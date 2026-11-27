//! R5's refreshes over flights' homes done on the card (H2,
//! `crate::host`'s `late_lift::lift_refreshes`) instead of waiting for the
//! flights that write those homes; what still waits, and how it is counted.

use super::*;
use crate::host::flights::{
    LandingHeld, R5_H8, R5_REFRESHES_LIFTED, R5_REFRESHES_REFUSED, READ, blocked, homes_written,
    in_air, r5_stats, refresh_waits, waits_left,
};

/// The bytes at `at` (title memory or a mapping of ours).
fn bytes(at: *const u8, len: usize) -> Vec<u8> {
    // SAFETY: the callers' memory holds `len` bytes there, and the card is
    // done with it.
    unsafe { std::slice::from_raw_parts(at, len).to_vec() }
}

/// What [`refresh_over_home`] leaves: the home, the exact copy and where it
/// starts, the bytes the flight wrote in the home's part of it, whether the
/// submission that refreshed it went with no wait (both in the air at once),
/// and the two lists.
struct OverHome {
    region: TestRegion,
    home: Arc<Shadow>,
    copy: Arc<Shadow>,
    view: u64,
    new: Vec<u8>,
    sent_with_no_wait: bool,
    lists: [crate::host::HostList; 2],
}

/// The bytes of the exact copy over the home, and past it.
const PART: u64 = 0x800;

/// An exact copy partly over a home (and partly past it), and a flight
/// whose shaders write the home's part of it; then a submission that
/// refreshes the copy (`written`: and declares a shader write of its own
/// over it there), sent while the flight is held, and everything landed.
fn refresh_over_home(host: &crate::host::Host, flights: &TestFlights, written: bool) -> OverHome {
    let region = TestRegion::new(8 * K64 as usize);
    let (start, len) = (region.base + K64, 2 * K64);
    register(0xc0de_5c01, start, len);
    let home = find(start, len).expect("a home");
    assert!(home.is_home());
    let view = start + len - PART;
    let copy = crate::shadow::exact(view, 2 * PART).expect("an exact copy");
    assert!(!copy.is_home());
    fill(start, 4 * K64, 9);
    copy.update(view as usize, 2 * PART as usize);
    let new = vec![0xc3u8; PART as usize];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, view, &from, PART);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    copy.refresh(&two, view, 2 * PART);
    if written {
        two.defer_gpu_write(&copy, view as usize, 2 * PART as usize);
    }
    two.close();
    assert_eq!(blocked(&[&two]), None, "no read of the CPU's");
    assert!(
        refresh_waits(&[&two]),
        "a refresh over a home a flight writes"
    );
    // Were it to wait, the hold goes after a while (soon if it should).
    let released = hold.released().after(if written { 200 } else { 5000 });
    host.send_for_test(&two);
    let sent_with_no_wait = in_air() == 2;
    hold.release();
    flights.land_all();
    released.join().unwrap();
    OverHome {
        region,
        home,
        copy,
        view,
        new,
        sent_with_no_wait,
        lists: [one, two],
    }
}

impl OverHome {
    fn done(self) {
        for list in &self.lists {
            list.reset();
        }
        unregister(0xc0de_5c01, self.region.base + K64);
        drop((self.lists, self.home, self.copy));
        // Not released: the exact copy stays in its table, over this memory.
        let _ = self.region;
    }
}

/// An exact copy partly over a home; a flight writes the home's part; the
/// next submission binds the copy for reading. Its refresh is lifted (no
/// wait, counted): the copy holds the flight's bytes over the home and the
/// title's past it. After the landing the copy's next refresh takes what the
/// title wrote there since.
#[test]
fn warp_r5_refresh_over_a_flights_home_is_lifted() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let before = r5_stats();
    let over = refresh_over_home(host, &flights, false);
    let after = r5_stats();
    assert!(over.sent_with_no_wait, "no wait");
    assert_eq!(after[0], before[0], "no wait for the refresh");
    assert_eq!(
        after[R5_REFRESHES_LIFTED],
        before[R5_REFRESHES_LIFTED] + 1,
        "lifted"
    );
    let held = crate::shadow::copy_bytes(&over.copy);
    assert_eq!(
        &held[..PART as usize],
        &over.new[..],
        "the flight's bytes, from the card"
    );
    assert!(
        same(over.view + PART, &held[PART as usize..]),
        "the title's past the home"
    );
    assert!(same(over.view, &over.new), "title memory has them");
    // What the title writes there since is taken by the next refresh.
    fill(over.view, PART, 0x44);
    let three = host.new_list().unwrap();
    over.copy.refresh(&three, over.view, 2 * PART);
    three.close();
    host.execute(&[&three]);
    assert!(
        same(over.view, crate::shadow::copy_bytes(&over.copy)),
        "the next refresh"
    );
    three.reset();
    drop(three);
    over.done();
}

/// As above, but the next submission declares a shader write of its own over
/// the copy where the flight writes the home: the refresh is not lifted, and
/// the submission waits as before (both counted); the copy and title memory
/// are what the waited refresh gives.
#[test]
fn warp_r5_refresh_lift_refused_when_the_copy_is_written() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let before = r5_stats();
    let over = refresh_over_home(host, &flights, true);
    let after = r5_stats();
    assert!(!over.sent_with_no_wait, "it waited");
    assert_eq!(after[0], before[0] + 1, "a wait for the refresh");
    assert_eq!(
        after[R5_REFRESHES_REFUSED],
        before[R5_REFRESHES_REFUSED] + 1,
        "refused"
    );
    assert_eq!(after[R5_REFRESHES_LIFTED], before[R5_REFRESHES_LIFTED]);
    assert!(
        same(over.view, &over.new),
        "title memory has the flight's bytes"
    );
    assert!(
        same(over.view, crate::shadow::copy_bytes(&over.copy)),
        "the refresh came after the flight"
    );
    over.done();
}

/// The H2 wait of [`warp_r5_refresh_lift_refused_when_the_copy_is_written`]
/// as this thread counted it (`crate::frame_budget::take_seen`): one H2
/// wait, its fence and landing parts within its time and not both nothing
/// (the card is held while the submission waits), and the exact copy's kind
/// in the H2 table with one wait, one refresh, its bytes over the home as
/// alias bytes, and the wait's whole time (it is the wait's only kind).
#[test]
fn warp_r5_h2_wait_is_attributed_to_the_exact_copy() {
    use crate::frame_budget::{Culprit, Seen, Wait, take_seen};
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let _ = take_seen();
    let over = refresh_over_home(host, &flights, true);
    let seen = take_seen();
    assert!(!over.sent_with_no_wait, "it waited");
    // A copy never handed to the card: no way noted.
    let key = 3 << 8;
    assert_eq!(over.copy.budget_key(), key, "a view's own copy");
    assert_eq!(
        crate::shadow::budget_name(key),
        "view copy, last handed as no way noted"
    );
    let waits: Vec<[u64; 3]> = seen
        .iter()
        .filter_map(|s| match *s {
            Seen::Wait {
                kind: Wait::H2,
                total,
                fence,
                land,
            } => Some([total, fence, land]),
            _ => None,
        })
        .collect();
    let [[total, fence, land]] = waits[..] else {
        panic!("one H2 wait: {seen:?}");
    };
    assert!(fence + land <= total, "{fence} + {land} within {total}");
    // The relay waited for the held card and then landed meanwhile.
    assert!(fence + land > 0, "the wait's parts were seen ({total} µs)");
    let culprits: Vec<Culprit> = seen
        .iter()
        .filter_map(|s| match *s {
            Seen::Culprit(c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(
        culprits,
        vec![Culprit {
            key,
            waits: 1,
            refreshes: 1,
            alias: PART,
            us: total,
        }],
        "its bytes over the home, and the wait's whole time"
    );
    over.done();
}

/// A list whose "shader" writes `bytes` into `copy` at title address `at`
/// (a copy on the card), declared as shaders' writes over `[at, at +
/// bytes)`, with no refresh of its own first; closed.
fn copy_writer(
    host: &crate::host::Host,
    copy: &Arc<Shadow>,
    at: u64,
    from: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    bytes: u64,
) -> crate::host::HostList {
    let list = host.new_list().unwrap();
    // SAFETY: an open list of the host's; live buffers, the range in both.
    unsafe {
        list.list
            .CopyBufferRegion(&copy.buffer, copy.offset(at), from, 0, bytes)
    };
    list.defer_gpu_write(copy, at as usize, bytes as usize);
    list.close();
    list
}

/// As [`warp_r5_refresh_over_a_flights_home_is_lifted`], but a second
/// flight, sent between the two, writes the exact copy over the home's part
/// (declared). A copy on the card from the home there would go over what
/// that flight's shaders wrote in the copy, and its copy-back would take the
/// home's bytes for theirs: the refresh is not lifted, the submission waits
/// (both counted), and title memory ends with the second flight's bytes.
/// Nothing lands until the card ran all three, so that a lifted copy would
/// have gone over them first. (The first flight writes the bytes the home
/// holds: where the title's bytes change under a copy's shaders' writes,
/// the title's are kept, `Shadow::merge_under_writes`.)
#[test]
fn warp_r5_refresh_lift_refused_when_a_flight_writes_the_copy() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let region = TestRegion::new(8 * K64 as usize);
    let (start, len) = (region.base + K64, 2 * K64);
    register(0xc0de_5c02, start, len);
    let home = find(start, len).expect("a home");
    assert!(home.is_home());
    let view = start + len - PART;
    let copy = crate::shadow::exact(view, 2 * PART).expect("an exact copy");
    assert!(!copy.is_home());
    fill(start, 4 * K64, 9);
    copy.update(view as usize, 2 * PART as usize);
    let (new, theirs) = (
        bytes(view as *const u8, PART as usize),
        vec![0x5du8; PART as usize],
    );
    let (from, other) = (source(host, &new), source(host, &theirs));
    let before = r5_stats();
    let hold = QueueHold::new(host);
    let one = writer(host, &home, view, &from, PART);
    host.send_for_test(&one);
    let mid = copy_writer(host, &copy, view, &other, PART);
    host.send_for_test(&mid);
    let two = host.new_list().unwrap();
    copy.refresh(&two, view, 2 * PART);
    two.close();
    assert!(
        refresh_waits(&[&two]),
        "a refresh over a home a flight writes"
    );
    // Were it to wait, the hold goes after a while.
    let released = hold.released().after(200);
    host.send_for_test(&two);
    let held = LandingHeld::new();
    host.card_done_for_test();
    drop(held);
    released.join().unwrap();
    flights.land_all();
    let after = r5_stats();
    assert_eq!(
        after[R5_REFRESHES_REFUSED],
        before[R5_REFRESHES_REFUSED] + 1,
        "refused"
    );
    assert_eq!(after[R5_REFRESHES_LIFTED], before[R5_REFRESHES_LIFTED]);
    assert_eq!(after[0], before[0] + 1, "a wait for the refresh");
    assert!(
        same(view, &theirs),
        "title memory has the second flight's bytes"
    );
    for list in [&one, &mid, &two] {
        list.reset();
    }
    unregister(0xc0de_5c02, start);
    drop((one, mid, two, hold, home, copy));
    // Not released: the exact copy stays in its table, over this memory.
    let _ = region;
}

/// One scripted sequence of submissions: shaders' writes into a home, and
/// in each submission after the first a refresh of an exact copy over what
/// the submission before wrote, read on the card into a buffer of ours.
/// Waited for one by one, and sent as flights held on the queue, so that
/// each refresh is over a home a flight in the air writes and is taken on
/// the card (H2): the same title memory and the same bytes read each way.
#[test]
fn warp_r5_matches_waited_submissions_with_lifted_refreshes() {
    let Some(host) = warp13() else { return };
    const N: usize = 50;
    let run = |flights: bool| -> (u64, Vec<u8>) {
        let _core = Switched::on(false);
        let flying = flights.then(|| TestFlights::on(true, true));
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 4 * K64);
        register(0xc0de_5e01, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        fill(start, 6 * K64, 0x21);
        let copy = crate::shadow::exact(start, len).expect("an exact copy");
        assert!(!copy.is_home());
        copy.update(start as usize, len as usize);
        let sources: Vec<_> = (0..N)
            .map(|i| source(host, &[i as u8 ^ 0x5c; 256]))
            .collect();
        let (out, data) = host.shared_buffer((64 * N) as u64).unwrap();
        let at = |i: usize| start + ((i as u64 * 9973) % (len - 256)) / 4 * 4;
        let lifted = r5_stats()[R5_REFRESHES_LIFTED];
        let mut lists = Vec::new();
        // With flights every submission is sent while the ones before are
        // held on the queue (were one to wait, the hold goes after a while).
        let hold = flights.then(|| QueueHold::new(host));
        let released = hold.as_ref().map(|hold| hold.released().after(3000));
        for (i, from) in sources.iter().enumerate() {
            let list = host.new_list().unwrap();
            list.defer_refresh(&home, at(i) as usize, 256);
            if i > 0 {
                copy.refresh(&list, at(i - 1), 64);
            }
            // SAFETY: an open list of the host's; live buffers, the ranges
            // in them.
            unsafe {
                if i > 0 {
                    list.list.CopyBufferRegion(
                        &out,
                        64 * i as u64,
                        &copy.buffer,
                        copy.offset(at(i - 1)),
                        64,
                    );
                }
                list.list
                    .CopyBufferRegion(&home.buffer, home.offset(at(i)), from, 0, 256);
            }
            list.defer_gpu_write(&home, at(i) as usize, 256);
            list.close();
            if flights {
                host.send_for_test(&list);
            } else {
                host.execute(&[&list]);
            }
            lists.push(list);
        }
        if flights {
            assert_eq!(in_air(), N, "no submission waited");
        }
        let lifted = r5_stats()[R5_REFRESHES_LIFTED] - lifted;
        drop(hold);
        if let Some(flying) = &flying {
            flying.land_all();
        }
        if let Some(released) = released {
            released.join().unwrap();
        }
        // (Slot 0 is not written: the first submission reads nothing.)
        let read = bytes(data.wrapping_add(64), 64 * (N - 1));
        let memory = bytes(start as *const u8, 6 * K64 as usize);
        let hash = memory.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        for list in &lists {
            list.reset();
        }
        drop(lists);
        if flights {
            assert!(lifted > 0, "refreshes were taken on the card");
        }
        unregister(0xc0de_5e01, start);
        drop((home, copy, out));
        drop(flying);
        // Not released: the exact copy stays in its table, over this memory.
        let _ = region;
        (hash, read)
    };
    let waited = run(false);
    let flown = run(true);
    assert_eq!(flown.1, waited.1, "the same bytes read");
    assert_eq!(flown.0, waited.0, "the same title memory");
}

/// A texture's memory is a home's, which a flight writes: the next
/// submission's texture check waits for the flight (H8), and the staging
/// state sees the flight's bytes.
#[cfg(feature = "diagnostics")]
#[test]
fn warp_r5_texture_over_a_flights_home_waits() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let region = TestRegion::new(4 * K64 as usize);
    let (start, len) = (region.base + K64, 2 * K64);
    register(0xc0de_5d01, start, len);
    let home = find(start, len).expect("a home");
    assert!(home.is_home());
    fill(start, len, 3);
    let (staging, texture, size) = crate::texture::staging_warp_tests::staging_at(start as *mut u8);
    assert!(size as u64 <= len);
    let new = [0x6eu8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    two.defer_texture(staging.clone(), false);
    two.close();
    assert_eq!(blocked(&[&two]), Some(READ));
    let waits = r5_stats()[R5_H8];
    let released = hold.released().after(200);
    host.send_for_test(&two);
    released.join().unwrap();
    assert_eq!(r5_stats()[R5_H8], waits + 1, "it waited");
    assert!(same(start, &new), "the flight's bytes");
    flights.land_all();
    one.reset();
    two.reset();
    drop((one, two, hold, staging, texture));
    unregister(0xc0de_5d01, start);
    drop(home);
    region.release();
}

// ---- The H2 split (`crate::frame_budget::h2_line`) ----

/// Title memory for a window over a home: a region that holds a whole cell
/// of windows (4 MiB, on its line), and the cell's start. The window an ask
/// there makes lies inside the region, which is never released: the window
/// stays in its table, and no later test's memory lies under it.
fn window_region() -> (TestRegion, u64) {
    const CELL: u64 = 4 << 20;
    let region = TestRegion::new(3 * CELL as usize);
    let cell = region.base.next_multiple_of(CELL);
    (region, cell)
}

/// A window over title memory that a home holds part of is a copy, not a
/// home, and keeps the core's reason (`Origin::Kept`): a refresh of it over
/// the home bytes a flight writes, reaching past the home's pages, is
/// counted as C2-cross under H2, with the refresh's source.
#[test]
fn warp_r5_window_over_home_is_c2_cross() {
    use crate::frame_budget::{H2Class, Seen, Source, take_seen};
    use crate::shadow::{Kept, Origin};
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let (region, cell) = window_region();
    let (start, len) = (cell + 2 * K64, K64);
    register(0xc0de_5f01, start, len);
    let home = find(start, len).expect("a home");
    assert!(home.is_home());
    // An ask outside buffers that reaches into the home.
    let window = find(start - 0x100, 0x200).expect("a window");
    assert!(
        window.is_window() && !window.is_home(),
        "a copy, not a home"
    );
    let (from, bytes) = window.span();
    assert!(
        cell <= from && from + bytes <= cell + (4 << 20),
        "in the region"
    );
    assert_eq!(window.origin(), Origin::Kept(Kept::Why(Why::Overlap)));
    fill(cell, 8 * K64, 4);
    let new = [0x3cu8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    window.refresh_from(&two, start - 0x100, 0x200, Source::Vertex);
    two.close();
    assert!(
        refresh_waits(&[&two]),
        "a refresh over a home a flight writes"
    );
    let _ = take_seen();
    let released = hold.released().after(200);
    host.send_for_test(&two);
    hold.release();
    flights.land_all();
    released.join().unwrap();
    let met: Vec<(Source, H2Class)> = take_seen()
        .iter()
        .filter_map(|s| match *s {
            Seen::H2Source { source, class, .. } => Some((source, class)),
            _ => None,
        })
        .collect();
    assert_eq!(met, [(Source::Vertex, H2Class::C2Cross)]);
    for list in [&one, &two] {
        list.reset();
    }
    unregister(0xc0de_5f01, start);
    drop((one, two, hold, home, window));
    // Not released: the window stays in its table, over this memory.
    let _ = region;
}

/// Two homes side by side, and a flight that writes both across their
/// border: what flights write in homes is one joined range there.
struct Adjacent {
    region: TestRegion,
    homes: [Arc<Shadow>; 2],
    /// The border between the two homes.
    border: u64,
}

impl Adjacent {
    fn new() -> Adjacent {
        let region = TestRegion::new(8 * K64 as usize);
        let (a, b) = (region.base + K64, region.base + 2 * K64);
        register(0xc0de_5f02, a, K64);
        register(0xc0de_5f03, b, K64);
        let homes = [find(a, K64).expect("a home"), find(b, K64).expect("a home")];
        assert!(homes.iter().all(|h| h.is_home()));
        assert!(!Arc::ptr_eq(&homes[0], &homes[1]), "two homes");
        fill(region.base, 8 * K64, 6);
        Adjacent {
            region,
            homes,
            border: b,
        }
    }

    /// Lists whose shaders write `bytes` (`len` of them) on each side of the
    /// border, closed.
    fn writers(
        &self,
        host: &crate::host::Host,
        from: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        len: u64,
    ) -> [crate::host::HostList; 2] {
        [
            writer(host, &self.homes[0], self.border - len, from, len),
            writer(host, &self.homes[1], self.border, from, len),
        ]
    }

    fn done(self) {
        unregister(0xc0de_5f02, self.region.base + K64);
        unregister(0xc0de_5f03, self.border);
        drop(self.homes);
        // Not released: exact copies stay in their table, over this memory.
        let _ = self.region;
    }
}

/// One submission refreshes three exact copies over home bytes flights
/// write: one its shaders write too (refused: written), one a flight in the
/// air writes (refused: a pending write), and one across the border of two
/// homes (split, not refused). The submission waits once; the refusal is
/// counted once for the submission, under its first piece's reason, and
/// per piece.
#[test]
fn warp_lift_counts_refusal_per_submission() {
    use crate::frame_budget::{Refusal, Seen, take_seen};
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let adjacent = Adjacent::new();
    let border = adjacent.border;
    let (written, pending) = (border - 0x8000, border - 0x4000);
    let [one, two, three] = [written, pending, border - 0x100]
        .map(|at| crate::shadow::exact(at, 0x200).expect("an exact copy"));
    for (copy, at) in [(&one, written), (&two, pending), (&three, border - 0x100)] {
        copy.update(at as usize, 0x200);
    }
    let new = [0x5au8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let mut sent: Vec<crate::host::HostList> = adjacent.writers(host, &from, 0x100).into();
    sent.push(writer(host, &adjacent.homes[0], written, &from, 0x100));
    sent.push(writer(host, &adjacent.homes[0], pending, &from, 0x100));
    sent.push(copy_writer(host, &two, pending, &from, 0x100));
    for list in &sent {
        host.send_for_test(list);
    }
    let four = host.new_list().unwrap();
    one.refresh(&four, written, 0x200);
    four.defer_gpu_write(&one, written as usize, 0x200);
    two.refresh(&four, pending, 0x200);
    three.refresh(&four, border - 0x100, 0x200);
    four.close();
    assert!(refresh_waits(&[&four]));
    let _ = take_seen();
    let before = r5_stats();
    let released = hold.released().after(200);
    host.send_for_test(&four);
    released.join().unwrap();
    flights.land_all();
    let refused: Vec<(Refusal, [u32; Refusal::ALL.len()])> = take_seen()
        .iter()
        .filter_map(|s| match *s {
            Seen::H2Refused { first, pieces } => Some((first, pieces)),
            _ => None,
        })
        .collect();
    let mut pieces = [0; Refusal::ALL.len()];
    pieces[Refusal::Written as usize] = 1;
    pieces[Refusal::Pending as usize] = 1;
    assert_eq!(refused, [(Refusal::Written, pieces)], "once, first written");
    let after = r5_stats();
    assert_eq!(
        after[R5_REFRESHES_REFUSED],
        before[R5_REFRESHES_REFUSED] + 1
    );
    assert_eq!(after[0], before[0] + 1, "one wait");
    for list in sent.iter().chain([&four]) {
        list.reset();
    }
    drop((sent, four, hold, one, two, three));
    adjacent.done();
}

/// An exact copy across the border of two homes side by side, which a
/// flight writes on both sides: the one range of home bytes under the
/// refresh is split by home and taken on the card, with no wait, and the
/// copy then holds the flight's bytes on both sides.
#[test]
fn warp_r5_piece_over_two_adjacent_homes_is_split() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let adjacent = Adjacent::new();
    let at = adjacent.border - 0x100;
    let copy = crate::shadow::exact(at, 0x200).expect("an exact copy");
    copy.update(at as usize, 0x200);
    assert!(
        crate::shadow::home_of(at, 0x200).is_none(),
        "no one home holds it"
    );
    let new: Vec<u8> = (0..0x100u32).map(|i| (i * 11 + 5) as u8).collect();
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let sides = adjacent.writers(host, &from, 0x100);
    for list in &sides {
        host.send_for_test(list);
    }
    let two = host.new_list().unwrap();
    copy.refresh(&two, at, 0x200);
    two.close();
    assert!(refresh_waits(&[&two]));
    assert!(
        homes_written()
            .iter()
            .any(|&(a, b)| a as u64 <= at && at + 0x200 <= b as u64),
        "one joined range over both homes"
    );
    let before = r5_stats();
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 3, "sent behind the flights with no wait");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    let after = r5_stats();
    assert_eq!(after[0], before[0], "no wait");
    assert_eq!(after[R5_REFRESHES_LIFTED], before[R5_REFRESHES_LIFTED] + 1);
    let held = crate::shadow::copy_bytes(&copy);
    assert_eq!(&held[..0x100], &new[..], "below the border, from the card");
    assert_eq!(&held[0x100..0x200], &new[..], "above it");
    for list in sides.iter().chain([&two]) {
        list.reset();
    }
    drop((sides, two, hold, copy));
    adjacent.done();
}
// ---- The heap buffer pass over a copy partly in a home ----

/// A home, and a title buffer around it whose copy is no home (a home holds
/// part of it), with a view heap over the buffer: the region, the home, the
/// buffer's start and the heap.
struct AroundHome {
    region: TestRegion,
    home: Arc<Shadow>,
    buffer: u64,
    heap: crate::draw::test_heap::ViewHeap,
}

impl AroundHome {
    fn new() -> AroundHome {
        let region = TestRegion::new(8 * K64 as usize);
        let (home_at, buffer) = (region.base + 2 * K64, region.base + K64);
        register(0xc0de_6001, home_at, K64);
        let home = find(home_at, K64).expect("a home");
        assert!(home.is_home());
        register(0xc0de_6002, buffer, 4 * K64);
        let copy = find(buffer, 4 * K64).expect("a copy");
        assert!(!copy.is_home(), "a home holds part of it");
        fill(region.base, 8 * K64, 8);
        let heap = crate::draw::test_heap::ViewHeap::over(buffer as usize, 4 * K64 as usize);
        AroundHome {
            region,
            home,
            buffer,
            heap,
        }
    }

    /// A flight's shaders write the home at its start plus 0x100; then a
    /// list runs the heap buffer pass (closed). The flight and the list.
    fn written_then_pass(
        &self,
        host: &crate::host::Host,
    ) -> (crate::host::HostList, crate::host::HostList) {
        let new = [0x6bu8; 0x100];
        let from = source(host, &new);
        let one = writer(host, &self.home, self.home.span().0 + 0x100, &from, 0x100);
        host.send_for_test(&one);
        let two = host.new_list().unwrap();
        self.heap.pass(&two, crate::draw::test_heap::test_frame());
        two.close();
        (one, two)
    }

    fn done(self) {
        unregister(0xc0de_6001, self.home.span().0);
        unregister(0xc0de_6002, self.buffer);
        drop((self.heap, self.home));
        let _ = self.region;
    }
}

/// Two views of the buffer, one below the home and one above it: the pass
/// refreshes each view's range, not the span between, so its submission
/// does not refresh the home's bytes the flight writes.
#[test]
fn warp_heap_pass_two_views_around_a_flight_home_do_not_wait() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let around = AroundHome::new();
    // [buffer, +768) and [buffer + 3 * 64 KiB, +768).
    around.heap.put(1, 0);
    around.heap.put(2, (3 * K64 / 12) as u32);
    let hold = QueueHold::new(host);
    let (one, two) = around.written_then_pass(host);
    assert!(!refresh_waits(&[&two]), "no refresh over the home");
    let before = r5_stats();
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "sent behind the flight with no wait");
    assert_eq!(r5_stats()[0], before[0], "no H2 wait");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    one.reset();
    two.reset();
    drop((one, two, hold));
    around.done();
}

/// A view of the buffer over the home bytes the flight writes: the pass's
/// refresh of it meets H2, and the card takes those bytes, with no wait.
#[test]
fn warp_heap_pass_view_over_flight_home_is_lifted() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let around = AroundHome::new();
    around.heap.put(1, 0);
    // From the home's start plus 4 bytes, over what the flight writes.
    around.heap.put(2, (K64 / 12 + 1) as u32);
    let hold = QueueHold::new(host);
    let (one, two) = around.written_then_pass(host);
    assert!(refresh_waits(&[&two]), "a refresh over the home");
    let before = r5_stats();
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "sent behind the flight with no wait");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    let after = r5_stats();
    assert_eq!(after[0], before[0], "no H2 wait");
    assert_eq!(
        after[R5_REFRESHES_LIFTED],
        before[R5_REFRESHES_LIFTED] + 1,
        "lifted"
    );
    one.reset();
    two.reset();
    drop((one, two, hold));
    around.done();
}

// ---- The H2 lift under R5 alone ----

/// A home, and a window over title memory that the home holds part of (a
/// copy, made by an ask outside buffers that reaches into the home): the
/// region, the home, the window and the home's start.
struct WindowOverHome {
    region: TestRegion,
    home: Arc<Shadow>,
    window: Arc<Shadow>,
    start: u64,
}

impl WindowOverHome {
    fn new(seed: u8) -> WindowOverHome {
        let (region, cell) = window_region();
        let start = cell + 2 * K64;
        register(0xc0de_6101, start, K64);
        let home = find(start, K64).expect("a home");
        assert!(home.is_home());
        let window = crate::shadow::find_any(start - 0x100, 0x200).expect("a window");
        assert!(window.is_window() && !window.is_home());
        let (from, bytes) = window.span();
        assert!(
            cell <= from && from + bytes <= cell + (4 << 20),
            "in the region"
        );
        fill(cell, 8 * K64, seed);
        window.update((start - 0x100) as usize, 0x200);
        WindowOverHome {
            region,
            home,
            window,
            start,
        }
    }

    /// The window's bytes at title address `at`, as its host copy holds them.
    fn held(&self, at: u64, len: usize) -> Vec<u8> {
        let from = (at - self.window.span().0) as usize;
        crate::shadow::copy_bytes(&self.window)[from..from + len].to_vec()
    }

    fn done(self) {
        unregister(0xc0de_6101, self.start);
        drop((self.home, self.window));
        // Not released: the window stays in its table, over this memory.
        let _ = self.region;
    }
}

/// A flight F0 reads a window's bytes over a home; a later flight F1 writes
/// the home there; while both are held on the queue the title's memory
/// there changes (as a card's write half way through would). The next
/// submission's refresh of the window over those bytes is taken on the
/// card: the CPU copies only around them, so F0, which runs before F1 on
/// the queue, reads the bytes from before, and the window then holds F1's.
#[test]
fn warp_r5_window_cpu_refresh_skips_lifted_piece() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let over = WindowOverHome::new(2);
    let start = over.start;
    let before = bytes(start as *const u8, 0x100);
    let (out, data) = host.shared_buffer(0x100).unwrap();
    let hold = QueueHold::new(host);
    let zero = host.new_list().unwrap();
    // SAFETY: an open list of the host's; live buffers, the range in both.
    unsafe {
        zero.list.CopyBufferRegion(
            &out,
            0,
            &over.window.buffer,
            over.window.offset(start),
            0x100,
        )
    };
    zero.close();
    host.send_for_test(&zero);
    let new = [0xe1u8; 0x100];
    let from = source(host, &new);
    let one = writer(host, &over.home, start, &from, 0x100);
    host.send_for_test(&one);
    fill(start, 0x100, 0x77);
    fill(start - 0x100, 0x100, 0x99);
    let title_below = bytes((start - 0x100) as *const u8, 0x100);
    let two = host.new_list().unwrap();
    over.window.refresh(&two, start - 0x100, 0x200);
    two.close();
    assert!(refresh_waits(&[&two]));
    let lifted = r5_stats()[R5_REFRESHES_LIFTED];
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 3, "sent behind the flights with no wait");
    assert_eq!(r5_stats()[R5_REFRESHES_LIFTED], lifted + 1, "lifted");
    assert_eq!(over.held(start, 0x100), before, "the CPU left the piece");
    assert_eq!(
        over.held(start - 0x100, 0x100),
        title_below,
        "and took what lies around it"
    );
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert_eq!(
        bytes(data, 0x100),
        before,
        "the older flight read the bytes from before"
    );
    assert_eq!(
        over.held(start, 0x100),
        new,
        "the card took the flight's bytes"
    );
    for list in [&zero, &one, &two] {
        list.reset();
    }
    drop((zero, one, two, hold, out));
    over.done();
}

/// A root shader resource view bound as a one-view table over a window
/// whose bytes a flight writes in a home: its refresh is taken on the card
/// (counted under its source), with no wait, and the window holds the
/// flight's bytes.
#[test]
fn warp_r5_root_srv_window_over_flight_home_is_lifted() {
    use crate::frame_budget::{H2Outcome, Seen, Source, take_seen};
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let over = WindowOverHome::new(3);
    let heap = crate::draw::test_heap::ViewHeap::new(true);
    let new = [0x2du8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &over.home, over.start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    let at = over.start - 0x100;
    let (mut state, root) = heap.root_srv(host, &two, at);
    two.close();
    assert!(refresh_waits(&[&two]));
    let _ = take_seen();
    let waits = r5_stats()[0];
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "sent behind the flight with no wait");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert_eq!(r5_stats()[0], waits, "no wait");
    let met: Vec<(Source, H2Outcome)> = take_seen()
        .iter()
        .filter_map(|s| match *s {
            Seen::H2Source {
                source, outcome, ..
            } => Some((source, outcome)),
            _ => None,
        })
        .collect();
    assert_eq!(met, [(Source::RootSrv, H2Outcome::Lifted)]);
    assert_eq!(over.held(over.start, 0x100), new);
    state.tables_done();
    one.reset();
    two.reset();
    drop((one, two, hold, root, heap));
    over.done();
}

/// An index buffer's refresh (a window a draw's ask finds) over home bytes
/// a flight writes is taken on the card, with no wait.
#[test]
fn warp_r5_index_refresh_over_flight_home_is_lifted() {
    use crate::frame_budget::{H2Outcome, Seen, Source, take_seen};
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let over = WindowOverHome::new(5);
    let new = [0x4bu8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &over.home, over.start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    let at = over.start - 0x80;
    let index = crate::shadow::find_any(at, 0x100).expect("the window");
    assert!(
        Arc::ptr_eq(&index, &over.window),
        "the window that is there"
    );
    index.refresh_from(&two, at, 0x100, Source::Index);
    two.close();
    let _ = take_seen();
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "sent behind the flight with no wait");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    let met: Vec<(Source, H2Outcome)> = take_seen()
        .iter()
        .filter_map(|s| match *s {
            Seen::H2Source {
                source, outcome, ..
            } => Some((source, outcome)),
            _ => None,
        })
        .collect();
    assert_eq!(met, [(Source::Index, H2Outcome::Lifted)]);
    assert_eq!(over.held(over.start, 0x80), new[..0x80]);
    one.reset();
    two.reset();
    drop((one, two, hold, index));
    over.done();
}

/// With R5 and `VICTORY_LIFT_READS` off a fill
/// from a home a flight writes is not lifted: it waits for the flight
/// (H8), and takes its bytes.
#[test]
fn warp_r5_lift_does_not_lift_fills_without_lift_reads() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let region = TestRegion::new(4 * K64 as usize);
    let (start, len) = (region.base + K64, K64);
    register(0xc0de_6201, start, len);
    let home = find(start, len).expect("a home");
    fill(start, len, 1);
    let written = [0x93u8; 64];
    let from = source(host, &written);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, start + 0x100, &from, 64);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    let (_, out) = two.scratch(64).expect("scratch");
    two.defer_fill(out, (start + 0x100) as *const u8, 64);
    two.close();
    assert_eq!(blocked(&[&two]), Some(READ));
    let (left, waits) = (waits_left(), r5_stats()[R5_H8]);
    let released = hold.released().after(200);
    host.send_for_test(&two);
    released.join().unwrap();
    assert_eq!(waits_left(), left, "not lifted");
    assert_eq!(r5_stats()[R5_H8], waits + 1, "it waited");
    flights.land_all();
    assert_eq!(bytes(out, 64), written, "the fill took the flight's bytes");
    one.reset();
    two.reset();
    drop((one, two, hold));
    unregister(0xc0de_6201, start);
    drop(home);
    region.release();
}

/// A refresh lifted in one submission leaves its piece stale on the CPU:
/// once the flights landed, the next submission's refresh, with nothing in
/// the air, takes the piece on the CPU, as the title wrote it since.
#[test]
fn warp_r5_lifted_refresh_then_cpu_refresh_next_submission() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let over = WindowOverHome::new(7);
    let new = [0x18u8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &over.home, over.start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    over.window.refresh(&two, over.start - 0x100, 0x200);
    two.close();
    let lifted = r5_stats()[R5_REFRESHES_LIFTED];
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(r5_stats()[R5_REFRESHES_LIFTED], lifted + 1, "lifted");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert_eq!(over.held(over.start, 0x100), new, "from the card");
    // The title writes the piece since; nothing is in the air.
    fill(over.start, 0x80, 0x61);
    let three = host.new_list().unwrap();
    over.window.refresh(&three, over.start - 0x100, 0x200);
    three.close();
    assert!(!refresh_waits(&[&three]));
    host.send_for_test(&three);
    flights.land_all();
    assert_eq!(
        over.held(over.start - 0x100, 0x200),
        bytes((over.start - 0x100) as *const u8, 0x200),
        "the next refresh took the piece on the CPU"
    );
    for list in [&one, &two, &three] {
        list.reset();
    }
    drop((one, two, three, hold));
    over.done();
}

/// A guard only (the software adapter may not fault as a card does): the
/// title frees the region under a lifted piece while the submission that
/// copies from its home is held on the queue. The free waits for the
/// flights that hold the home; the device is still there after.
#[test]
fn warp_r5_lifted_piece_after_title_frees_region() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = r5_on();
    let region = TestRegion::new(8 * K64 as usize);
    let start = region.base + 2 * K64;
    register(0xc0de_6301, start, K64);
    let home = find(start, K64).expect("a home");
    let copy = crate::shadow::exact(start - 0x100, 0x200).expect("an exact copy");
    fill(region.base, 8 * K64, 9);
    copy.update((start - 0x100) as usize, 0x200);
    let new = [0x0fu8; 0x100];
    let from = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, start, &from, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    copy.refresh(&two, start - 0x100, 0x200);
    two.close();
    let lifted = r5_stats()[R5_REFRESHES_LIFTED];
    let released = hold.released().after(200);
    host.send_for_test(&two);
    assert_eq!(r5_stats()[R5_REFRESHES_LIFTED], lifted + 1, "lifted");
    unregister(0xc0de_6301, start);
    drop(home);
    region.release();
    released.join().unwrap();
    flights.land_all();
    assert!(
        crate::descriptor::host::device_removed().is_none(),
        "the device is still there"
    );
    one.reset();
    two.reset();
    drop((one, two, hold, copy));
}
