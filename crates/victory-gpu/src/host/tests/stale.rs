//! Title bytes under flights' shader writes (`stale_under_flights`,
//! `Shadow::merge_under_writes`): why a merge refuses, who declared what it
//! met, and the waits that are left.

use super::super::*;
use crate::frame_budget::{Declarer, StaleRefusal};
use crate::shadow::{Shadow, find, register, unregister};
use std::sync::Arc;

const K64: usize = 64 << 10;

/// Title memory of the test's own, `len` bytes, zeroed; write-watched if
/// `watched` (as the title's memory is), else plain. Never freed: a copy
/// may outlive the test in a list's hold. Memory that an earlier test
/// released with copies still in the tables over it is passed over (and
/// kept, so that it is not handed out again): those copies would be found
/// instead of the test's.
fn title_memory(len: usize, watched: bool) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
    };
    // (Not in the bindings.)
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let flags = MEM_RESERVE | MEM_COMMIT | if watched { MEM_WRITE_WATCH } else { 0 };
    loop {
        // SAFETY: an allocation of the test's own.
        let at = unsafe { VirtualAlloc(std::ptr::null(), len, flags, PAGE_READWRITE) } as usize;
        assert_ne!(at, 0);
        if !crate::shadow::tabled_over(at as u64, (at + len) as u64) {
            return at;
        }
    }
}

fn put(at: usize, bytes: &[u8]) {
    // SAFETY: committed memory of the test's own.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), at as *mut u8, bytes.len()) };
}

fn get(at: usize, len: usize) -> Vec<u8> {
    // SAFETY: committed memory of the test's own.
    unsafe { std::slice::from_raw_parts(at as *const u8, len).to_vec() }
}

/// The host copy of a title buffer `[at, at + len)`, registered as `id`.
fn copy_of(id: usize, at: usize, len: usize) -> Arc<Shadow> {
    register(id, at as u64, len as u64);
    find(at as u64, len as u64).expect("a host copy")
}

/// A copy of plain memory: the first refresh finds it not write-watched,
/// and a merge under a declared range refuses for that.
#[test]
fn warp_stale_reason_not_watched() {
    let Some(_host) = host() else { return };
    let at = title_memory(K64, false);
    let copy = copy_of(0x57a1_e001, at, K64);
    copy.update(at, K64);
    assert!(copy.unwatched());
    copy.writing(at + 0x100, 0x100);
    put(at + 0x180, &[7; 8]);
    let mut runs_left = 512;
    let refused = copy
        .merge_under_writes(at, K64, &mut runs_left)
        .unwrap_err();
    assert_eq!(refused.why, StaleRefusal::NotWatched);
    assert_eq!(runs_left, 512, "nothing put in");
    unregister(0x57a1_e001, at as u64);
}

/// A copy never refreshed (whether its memory is watched is not known yet)
/// of memory the kernel does not watch: neither the journal nor the kernel
/// can say what was written.
#[test]
fn warp_stale_reason_journal_silent() {
    let Some(_host) = host() else { return };
    let at = title_memory(K64, false);
    let copy = copy_of(0x57a1_e002, at, K64);
    assert!(!copy.unwatched());
    copy.writing(at + 0x100, 0x100);
    let refused = copy.merge_under_writes(at, K64, &mut 512).unwrap_err();
    assert_eq!(refused.why, StaleRefusal::JournalSilent);
    unregister(0x57a1_e002, at as u64);
}

/// A card copy whose first busy page has 513 changed runs, one more than
/// the budget: nothing is merged, and the reason says it was the first
/// page. With one run on the first page and the budget's worth on the
/// second, the second page refuses as a later one.
#[test]
fn warp_stale_reason_card_runs_out_first_page() {
    let Some(host) = host() else { return };
    if !host.card_ready() {
        return;
    }
    const LEN: usize = 1 << 20;
    let at = title_memory(LEN, true);
    let copy = copy_of(0x57a1_e003, at, LEN);
    if !copy.carded() {
        unregister(0x57a1_e003, at as u64);
        return; // no card copies here
    }
    copy.update(at, LEN);
    assert!(!copy.unwatched());
    copy.writing(at, 2 * K64);
    // Every other byte of the first page's first 1,026 bytes: 513 runs.
    for i in 0..513 {
        put(at + 2 * i, &[0x5a]);
    }
    let mut runs_left = 512;
    let refused = copy
        .merge_under_writes(at, K64, &mut runs_left)
        .unwrap_err();
    assert_eq!(refused.why, StaleRefusal::CardRunsOutFirst);
    assert_eq!(runs_left, 512, "merged = 0");
    assert!(
        crate::shadow::copy_bytes(&copy)[..1026]
            .iter()
            .all(|&b| b == 0),
        "the CPU side as it was"
    );
    // The first page as it was but one run; 512 runs on the second.
    for i in 1..513 {
        put(at + 2 * i, &[0]);
    }
    for i in 0..512 {
        put(at + K64 + 2 * i, &[0xa5]);
    }
    let mut runs_left = 512;
    let refused = copy
        .merge_under_writes(at, 2 * K64, &mut runs_left)
        .unwrap_err();
    assert_eq!(refused.why, StaleRefusal::CardRunsOutLater);
    assert_eq!(runs_left, 511, "the first page's run went in");
    assert_eq!(crate::shadow::copy_bytes(&copy)[0], 0x5a);
    unregister(0x57a1_e003, at as u64);
}

/// A copy in the PC's memory whose snapshot is gone: the title's bytes
/// cannot be told from the shaders'.
#[test]
fn warp_stale_reason_no_snapshot() {
    let Some(_host) = host() else { return };
    let at = title_memory(K64, true);
    let copy = copy_of(0x57a1_e004, at, K64);
    copy.update(at, K64);
    assert!(!copy.unwatched());
    copy.writing(at + 0x100, 0x100);
    copy.drop_snapshot_for_test();
    put(at + 0x180, &[9; 8]);
    let refused = copy.merge_under_writes(at, K64, &mut 512).unwrap_err();
    assert_eq!(refused.why, StaleRefusal::NoSnapshot);
    unregister(0x57a1_e004, at as u64);
}

/// A root UAV's declaration, made through a list and a flight, keeps its
/// site: the relay's record of the range says so, and so does a refusal
/// that meets it (by the flight sent just before).
#[test]
fn warp_stale_pending_source_root_tail() {
    let Some(host) = host() else { return };
    let flights = flights::TestFlights::on(false, false);
    let at = title_memory(K64, false);
    let copy = copy_of(0x57a1_e005, at, K64);
    copy.update(at, K64);
    let hold = flights::QueueHold::new(host);
    let list = host.new_list().unwrap();
    copy.refresh(&list, (at + 0x200) as u64, 0x200);
    copy.gpu_writes_from(&list, (at + 0x200) as u64, 0x200, Declarer::RootTail);
    list.close();
    host.send_for_test(&list);
    let range = (at + 0x200, at + 0x400);
    let (number, by) = copy
        .declared_by_for_test(range)
        .expect("declared with the relay");
    assert_eq!(by, Declarer::RootTail);
    assert!(number > 0);
    // The next submission's look (its number is the one after the flight).
    crate::shadow::preparing(number + 1);
    let refused = copy.merge_under_writes(at, K64, &mut 512).unwrap_err();
    assert_eq!(refused.why, StaleRefusal::NotWatched);
    assert_eq!(refused.by, Declarer::RootTail);
    assert!(refused.newest, "declared by the flight just before");
    hold.release();
    flights.land_all();
    list.reset();
    unregister(0x57a1_e005, at as u64);
}

/// Whether the thread's counters saw a wait for stale spans or a refusal
/// since the last look (`crate::frame_budget::take_seen`).
fn stale_seen() -> (bool, usize) {
    use crate::frame_budget::{Seen, Wait};
    let seen = crate::frame_budget::take_seen();
    let waited = seen.iter().any(|s| {
        matches!(
            s,
            Seen::Wait {
                kind: Wait::Stale,
                ..
            }
        )
    });
    let refused = seen
        .iter()
        .filter(|s| matches!(s, Seen::StaleRefused(_)))
        .count();
    (waited, refused)
}

/// A flight whose shaders may write `[at, at + len)` of `copy` (declared,
/// nothing written), sent behind a hold on the queue: it stays in the air
/// until the hold goes.
fn held_flight(
    host: &Host,
    copy: &Arc<Shadow>,
    at: usize,
    len: usize,
) -> (flights::QueueHold, HostList) {
    let hold = flights::QueueHold::new(host);
    let list = host.new_list().unwrap();
    copy.refresh(&list, at as u64, len as u64);
    copy.gpu_writes(&list, at as u64, len as u64);
    list.close();
    host.send_for_test(&list);
    (hold, list)
}

/// A list that refreshes `ranges` of `copy`, closed.
fn refresher(host: &Host, copy: &Arc<Shadow>, ranges: &[(usize, usize)]) -> HostList {
    let list = host.new_list().unwrap();
    for &(a, e) in ranges {
        copy.refresh(&list, a as u64, (e - a) as u64);
    }
    list.close();
    list
}

/// The title writes the page a flight may write, between two pages a
/// submission refreshes. No refresh of it writes that page, so nothing is
/// stale: the submission goes with no wait (the memory is not watched, so a
/// look at the page would refuse and wait). The next submission that
/// refreshes the page, with nothing in the air, takes the title's bytes.
#[test]
fn warp_stale_pending_between_refreshed_ranges_no_settle() {
    let Some(host) = host() else { return };
    let flights = flights::TestFlights::on(false, false);
    let at = title_memory(3 * K64, false);
    let copy = copy_of(0x57a1_e011, at, 3 * K64);
    copy.update(at, 3 * K64);
    let (hold, one) = held_flight(host, &copy, at + K64, K64);
    put(at + K64 + 0x40, &[0x77; 32]);
    let _ = stale_seen();
    let two = refresher(host, &copy, &[(at, at + K64), (at + 2 * K64, at + 3 * K64)]);
    // Were it to wait, the hold goes after a while.
    let released = hold.released().after(2000);
    host.send_for_test(&two);
    let sent_with_no_wait = flights::in_air() == 2;
    let (waited, refused) = stale_seen();
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert!(sent_with_no_wait, "no wait");
    assert!(!waited && refused == 0, "nothing stale");
    let three = refresher(host, &copy, &[(at + K64, at + 2 * K64)]);
    host.send_for_test(&three);
    flights.land_all();
    assert_eq!(
        &crate::shadow::copy_bytes(&copy)[K64 + 0x40..K64 + 0x60],
        &[0x77; 32],
        "the next refresh of the page takes the title's bytes"
    );
    for list in [one, two, three] {
        list.reset();
    }
    unregister(0x57a1_e011, at as u64);
}

/// The title writes inside a range a flight may write and the submission
/// refreshes: still merged beside the flight's writes, with no wait, and the
/// landing keeps the title's bytes.
#[test]
fn warp_stale_pending_inside_refreshed_range_still_merged() {
    let Some(host) = host() else { return };
    let flights = flights::TestFlights::on(false, false);
    let at = title_memory(2 * K64, true);
    let copy = copy_of(0x57a1_e012, at, 2 * K64);
    copy.update(at, 2 * K64);
    let (hold, one) = held_flight(host, &copy, at + 0x1000, 0x1000);
    put(at + 0x1100, &[0x3c; 8]);
    let _ = stale_seen();
    let two = refresher(host, &copy, &[(at, at + K64)]);
    let released = hold.released().after(2000);
    host.send_for_test(&two);
    let sent_with_no_wait = flights::in_air() == 2;
    let (waited, refused) = stale_seen();
    let held = crate::shadow::copy_bytes(&copy)[0x1100..0x1108].to_vec();
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert!(sent_with_no_wait, "no wait");
    assert!(!waited && refused == 0, "merged, not refused");
    assert_eq!(held, [0x3c; 8], "the title's bytes went into the copy");
    assert_eq!(
        get(at + 0x1100, 8),
        [0x3c; 8],
        "and stay the title's after the landing"
    );
    for list in [one, two] {
        list.reset();
    }
    unregister(0x57a1_e012, at as u64);
}

/// A look for stale spans sees the copies' pending writes as they are when
/// it looks, not as they were when the submission's ranges were gathered:
/// once the flight that declared the range is done, the same ranges are not
/// stale any more.
#[test]
fn warp_stale_relook_after_narrowed_wait_uses_current_pending() {
    let Some(host) = host() else { return };
    let _flights = flights::TestFlights::on(false, false);
    let at = title_memory(K64, false);
    let copy = copy_of(0x57a1_e013, at, K64);
    copy.update(at, K64);
    let range = (at + 0x200, at + 0x400);
    copy.writing(range.0, range.1 - range.0);
    let list = refresher(host, &copy, &[(at, at + K64)]);
    let refreshed = refreshed_ranges(&[&list]);
    let stale = stale_under_flights(&refreshed, true);
    assert_eq!(stale.len(), 1, "not watched: refused");
    copy.copy_back_done(&[range]);
    assert!(
        stale_under_flights(&refreshed, false).is_empty(),
        "nothing pending now"
    );
    list.reset();
    unregister(0x57a1_e013, at as u64);
}

/// One walk gathers every refreshed copy (unwatched ones too, for the looks
/// for stale spans); the asking covers only those that may be watched, each
/// by the span of its ranges.
#[test]
fn warp_ask_once_skips_unwatched_from_shared_walk() {
    let Some(host) = host() else { return };
    let (plain, watched) = (title_memory(2 * K64, false), title_memory(2 * K64, true));
    let a = copy_of(0x57a1_e014, plain, 2 * K64);
    let b = copy_of(0x57a1_e015, watched, 2 * K64);
    a.update(plain, 2 * K64);
    b.update(watched, 2 * K64);
    assert!(a.unwatched() && !b.unwatched());
    let list = host.new_list().unwrap();
    for (copy, base) in [(&a, plain), (&b, watched)] {
        copy.refresh(&list, (base + 0x100) as u64, 0x100);
        copy.refresh(&list, (base + K64) as u64, 0x100);
        copy.refresh(&list, (base + 0x180) as u64, 0x100);
    }
    list.close();
    let refreshed = refreshed_ranges(&[&list]);
    assert_eq!(refreshed.len(), 2, "both copies, unfiltered");
    let ranges_of = |copy: &Arc<Shadow>| refreshed[&(Arc::as_ptr(copy) as usize)].1.clone();
    assert_eq!(
        ranges_of(&a),
        [
            (plain + 0x100, plain + 0x280),
            (plain + K64, plain + K64 + 0x100)
        ]
    );
    assert_eq!(
        asked_spans(&refreshed),
        [(watched + 0x100, watched + K64 + 0x100)]
    );
    list.reset();
    unregister(0x57a1_e014, plain as u64);
    unregister(0x57a1_e015, watched as u64);
}
