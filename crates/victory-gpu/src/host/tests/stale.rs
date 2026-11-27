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
