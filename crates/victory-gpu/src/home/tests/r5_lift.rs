//! R5's reads of homes on the card (`VICTORY_R5_LIFT=1`,
//! `crate::host::r5_lift`): fills and copies of homes' bytes that flights
//! write, and refreshes of copies over them, done on the card behind the
//! flights instead of waiting for them; and what still waits.

use super::*;
use crate::host::flights::{
    LandingHeld, R5_H8, R5_REFRESHES_LIFTED, R5_REFRESHES_REFUSED, READ, aborted, blocked,
    homes_written, in_air, r5_stats, refresh_waits, waits_left,
};
use windows::core::Interface;

/// The core on, and flights with R5 and its lifts (in that order, so that
/// the flights land before the core goes off).
fn lifting() -> (Switched, TestFlights) {
    let core = Switched::on(false);
    (core, TestFlights::r5_lifting(true))
}

/// The bytes at `at` (title memory or a mapping of ours).
fn bytes(at: *const u8, len: usize) -> Vec<u8> {
    // SAFETY: the callers' memory holds `len` bytes there, and the card is
    // done with it.
    unsafe { std::slice::from_raw_parts(at, len).to_vec() }
}

/// Whether title memory `[from, to)` touches `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: u64, to: u64) -> bool {
    ranges
        .iter()
        .any(|&(a, b)| (a as u64) < to && from < b as u64)
}

/// A flight's shaders write a home's range; the next submission fills
/// upload memory from it. The fill is lifted (no wait: both are in the air
/// together, and the lifted reads' counter moved), and once both ran it
/// holds the card's bytes.
#[test]
fn warp_r5_lifted_fill_from_a_home_sees_the_flights_bytes() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = lifting();
    let region = TestRegion::new(4 * K64 as usize);
    let (start, len) = (region.base + K64, K64);
    register(0xc0de_5a01, start, len);
    let home = find(start, len).expect("a home under flights");
    assert!(home.is_home());
    fill(start, len, 1);
    let written = [0xa5u8; 64];
    let from = source(host, &written);
    let hold = QueueHold::new(host);
    let one = writer(host, &home, start + 0x100, &from, 64);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    let (_, out) = two.scratch(64).expect("scratch");
    two.defer_fill(out, (start + 0x100) as *const u8, 64);
    two.close();
    assert_eq!(
        blocked(&[&two]),
        Some(READ),
        "a read of what a flight writes"
    );
    let (left, waits) = (waits_left(), r5_stats()[R5_H8]);
    // Were it to wait, the hold would go after a while.
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "sent behind the flight with no wait");
    assert_eq!(waits_left(), left + 1, "lifted");
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert_eq!(bytes(out, 64), written, "the fill took what the card wrote");
    assert_eq!(r5_stats()[R5_H8], waits, "no wait");
    assert!(same(start + 0x100, &written));
    one.reset();
    two.reset();
    drop((one, two, hold));
    unregister(0xc0de_5a01, start);
    drop(home);
    region.release();
}

/// A flight writes the home `src` at `from`; the next submission's title
/// buffer copy from there into another home at `to` is lifted: on the card
/// behind the flight, with no wait. While that flight flies the destination
/// is among what flights write in homes; once it landed the title memory
/// there is the first flight's bytes, and `copy`, a copy over the
/// destination, takes them at its next refresh: it was told at the landing
/// (`told` names the case).
fn copy_on_the_card(
    host: &crate::host::Host,
    flights: &TestFlights,
    src: &Arc<Shadow>,
    (from, to): (u64, u64),
    copy: Option<(&Arc<Shadow>, u64, u64)>,
    told: &str,
) {
    let new: Vec<u8> = (0..0x100u32).map(|i| (i * 7 + 3) as u8).collect();
    let source_buffer = source(host, &new);
    let hold = QueueHold::new(host);
    let one = writer(host, src, from, &source_buffer, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    two.defer_copy(to as *mut u8, from as *const u8, 0x100);
    two.close();
    assert_eq!(
        blocked(&[&two]),
        Some(READ),
        "{told}: a read of what a flight writes"
    );
    let left = waits_left();
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(in_air(), 2, "{told}: sent behind the flight with no wait");
    assert_eq!(waits_left(), left + 1, "{told}: lifted");
    assert!(
        touches(&homes_written(), to, to + 0x100),
        "{told}: the destination is written by a flight"
    );
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert!(same(to, &new), "{told}: the card copied the flight's bytes");
    assert!(!touches(&homes_written(), to, to + 0x100), "{told}: landed");
    if let Some((copy, view, len)) = copy {
        // On the software adapter the card is this process, and write
        // watching saw it: forgotten, as a card's write is never seen.
        // SAFETY: the test's own allocation.
        unsafe { mem::ResetWriteWatch(view as *const c_void, len as usize) };
        copy.update(view as usize, len as usize);
        assert!(
            same(view, crate::shadow::copy_bytes(copy)),
            "{told}: the copy was told"
        );
    }
    one.reset();
    two.reset();
}

/// A title buffer copy between two homes on the card: between two homes of
/// their own (two buffers), and between two homes in one chunk of the pool
/// (one buffer: through a bounce buffer).
#[test]
fn warp_r5_copy_home_to_home_on_the_card() {
    let Some(host) = warp13() else { return };
    {
        let (_core, flights) = lifting();
        let region = TestRegion::new(8 * K64 as usize);
        let (a, b) = (region.base + K64, region.base + 4 * K64);
        register(0xc0de_5b01, a, K64);
        register(0xc0de_5b02, b, K64);
        let (one, two) = (find(a, K64).expect("a home"), find(b, K64).expect("a home"));
        assert!(one.is_home() && two.is_home());
        assert_ne!(one.buffer.as_raw(), two.buffer.as_raw(), "two buffers");
        fill(a, 6 * K64, 2);
        let (view, len) = (b + 0x380, 0x200);
        let copy = crate::shadow::exact(view, len).expect("an exact copy");
        copy.update(view as usize, len as usize);
        copy_on_the_card(
            host,
            &flights,
            &one,
            (a + 0x200, b + 0x400),
            Some((&copy, view, len)),
            "two homes",
        );
        unregister(0xc0de_5b01, a);
        unregister(0xc0de_5b02, b);
        drop((one, two, copy, flights));
        // Not released: the exact copy stays in its table, over this memory.
        let _ = region;
    }
    // Two homes in one chunk of the pool share its buffer.
    let _core = Switched::pool();
    let flights = TestFlights::r5_lifting(true);
    let chunk = TestChunk::new(8 * K64 as usize, crate::pool::PoolChunk::Committed);
    let place = Place::new(4 * K64 as usize);
    place.map(0, 4 * K64, &chunk, 0);
    let (a, b) = (place.base + 0x100, place.base + 2 * K64 + 0x100);
    register(0xc0de_5b03, a, 0x1000);
    register(0xc0de_5b04, b, 0x1000);
    let (one, two) = (
        find(a, 0x1000).expect("a home"),
        find(b, 0x1000).expect("a home"),
    );
    assert_eq!(
        (one.home_kind(), two.home_kind()),
        (Some(Kind::Pool), Some(Kind::Pool))
    );
    assert_eq!(
        one.buffer.as_raw(),
        two.buffer.as_raw(),
        "one chunk's buffer"
    );
    fill(a, 0x1000, 4);
    fill(b, 0x1000, 5);
    copy_on_the_card(
        host,
        &flights,
        &one,
        (a + 0x200, b + 0x400),
        None,
        "one chunk",
    );
    unregister(0xc0de_5b03, a);
    unregister(0xc0de_5b04, b);
    drop((one, two, flights));
    place.unmap(0, 4 * K64);
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
    let (_core, flights) = lifting();
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
    let (_core, flights) = lifting();
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
    let (_core, flights) = lifting();
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

/// A flight writes a home at `a`; the next submission copies from there
/// into a second home at `to`, then refreshes an exact copy of a view over
/// the second home (`0x200` bytes at `view`) that holds the destination
/// (`over`), or lies past it. Over it the plan is aborted
/// (`LiftAbort::Viewed`, counted): the copy's refresh on the CPU would take
/// the home's bytes before the card's copy wrote them. The submission waits
/// for the flight, and once it landed the exact copy holds the copied bytes.
/// Past it the copy is lifted, with no wait.
fn copy_into_a_viewed_home(host: &crate::host::Host, over: bool) {
    let (_core, flights) = lifting();
    let region = TestRegion::new(8 * K64 as usize);
    let (a, b) = (region.base + K64, region.base + 4 * K64);
    register(0xc0de_5b05, a, K64);
    register(0xc0de_5b06, b, K64);
    let (one_home, two_home) = (find(a, K64).expect("a home"), find(b, K64).expect("a home"));
    assert!(one_home.is_home() && two_home.is_home());
    fill(a, 6 * K64, 6);
    let view = b + 0x380;
    let copy = crate::shadow::exact(view, 0x200).expect("an exact copy");
    copy.update(view as usize, 0x200);
    let to = if over { b + 0x400 } else { b + 0x800 };
    let new: Vec<u8> = (0..0x100u32).map(|i| (i * 5 + 1) as u8).collect();
    let source_buffer = source(host, &new);
    let (viewed, waits, left) = (aborted().1, r5_stats()[R5_H8], waits_left());
    let hold = QueueHold::new(host);
    let one = writer(host, &one_home, a + 0x200, &source_buffer, 0x100);
    host.send_for_test(&one);
    let two = host.new_list().unwrap();
    two.defer_copy(to as *mut u8, (a + 0x200) as *const u8, 0x100);
    copy.refresh(&two, view, 0x200);
    two.close();
    assert_eq!(
        blocked(&[&two]),
        Some(READ),
        "a read of what a flight writes"
    );
    // Were it to wait, the hold goes after a while (soon if it should).
    let released = hold.released().after(if over { 200 } else { 5000 });
    host.send_for_test(&two);
    if over {
        assert_eq!(aborted().1, viewed + 1, "a copy over the home is refreshed");
        assert_eq!(r5_stats()[R5_H8], waits + 1, "it waited");
        assert_eq!(waits_left(), left, "nothing lifted");
    } else {
        assert_eq!(in_air(), 2, "sent behind the flight with no wait");
        assert_eq!(aborted().1, viewed, "the copy lies past the destination");
        assert_eq!(waits_left(), left + 1, "lifted");
    }
    hold.release();
    flights.land_all();
    released.join().unwrap();
    assert!(same(to, &new), "the copied bytes");
    if over {
        let held = crate::shadow::copy_bytes(&copy);
        let at = (to - view) as usize;
        assert_eq!(&held[at..at + 0x100], &new[..], "the exact copy took them");
    }
    one.reset();
    two.reset();
    unregister(0xc0de_5b05, a);
    unregister(0xc0de_5b06, b);
    drop((one, two, hold, one_home, two_home, copy, flights));
    // Not released: the exact copy stays in its table, over this memory.
    let _ = region;
}

/// [`copy_into_a_viewed_home`], over the destination and past it.
#[test]
fn warp_r5_lifted_copy_into_a_home_aborts_when_a_view_over_it_is_refreshed() {
    let Some(host) = warp13() else { return };
    copy_into_a_viewed_home(host, true);
    copy_into_a_viewed_home(host, false);
}

/// A flight writes a home; a second flight's shaders write a copy that is
/// no home (memory of the process heap gets none) over `[x, x + 0x100)`;
/// the next submission copies from the home into the copy at `x + 0x80`,
/// lifted into the copy's buffer while the second flight still has its
/// range pending. Nothing lands until the card ran all three: title memory
/// ends with the second flight's bytes below `x + 0x80` and the copied
/// bytes from there, as the queue's order says.
#[test]
fn warp_r5_lifted_copy_into_a_copy_a_flight_writes() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = lifting();
    let region = TestRegion::new(4 * K64 as usize);
    let (a, len) = (region.base + K64, K64);
    register(0xc0de_5b07, a, len);
    let home = find(a, len).expect("a home");
    assert!(home.is_home());
    fill(a, len, 7);
    let block = vec![0x11u8; 2 * K64 as usize];
    let base = block.as_ptr() as u64;
    let x = base + 0x1000;
    register(0xc0de_5b08, base, K64);
    let copy = find(base, K64).expect("a copy");
    assert!(!copy.is_home());
    copy.update(x as usize, 0x200);
    let new: Vec<u8> = (0..0x100u32).map(|i| (i * 3 + 2) as u8).collect();
    let theirs = vec![0x7au8; 0x100];
    let (from, other) = (source(host, &new), source(host, &theirs));
    let left = waits_left();
    let hold = QueueHold::new(host);
    let one = writer(host, &home, a + 0x200, &from, 0x100);
    host.send_for_test(&one);
    let mid = copy_writer(host, &copy, x, &other, 0x100);
    host.send_for_test(&mid);
    let two = host.new_list().unwrap();
    two.defer_copy((x + 0x80) as *mut u8, (a + 0x200) as *const u8, 0x100);
    two.close();
    assert_eq!(
        blocked(&[&two]),
        Some(READ),
        "a read of what a flight writes"
    );
    let released = hold.released().after(5000);
    host.send_for_test(&two);
    assert_eq!(waits_left(), left + 1, "lifted");
    let held = LandingHeld::new();
    hold.release();
    host.card_done_for_test();
    drop(held);
    flights.land_all();
    released.join().unwrap();
    assert!(same(x, &theirs[..0x80]), "the second flight's bytes");
    assert!(same(x + 0x80, &new), "the copied bytes over them");
    for list in [&one, &mid, &two] {
        list.reset();
    }
    unregister(0xc0de_5b07, a);
    unregister(0xc0de_5b08, base);
    drop((one, mid, two, hold, home, copy, flights));
    drop(block);
    region.release();
}

/// A texture's memory is a home's, which a flight writes: the next
/// submission's texture check cannot be lifted (counted), it waits for the
/// flight as before, and the staging state sees the flight's bytes.
#[cfg(feature = "diagnostics")]
#[test]
fn warp_r5_lift_abort_waits() {
    let Some(host) = warp13() else { return };
    let (_core, flights) = lifting();
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
    let (textures, waits) = (aborted().0, r5_stats()[R5_H8]);
    let released = hold.released().after(200);
    host.send_for_test(&two);
    released.join().unwrap();
    assert_eq!(aborted().0, textures + 1, "a texture reads it");
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

/// One scripted sequence of submissions: shaders' writes into a home, fills
/// of upload memory and title buffer copies into a second home from what the
/// submission before wrote, bottom-of-pipe writes and a fence signal behind
/// each. Waited for one by one, sent as flights, and sent as flights with
/// R5's lifts: the same title memory, fills and fence value each way, and
/// the lifts did lift.
#[test]
fn warp_r5_matches_waited_submissions_with_lifts() {
    let Some(host) = warp13() else { return };
    const N: usize = 50;
    let run = |flights: bool, lift: bool| -> (u64, Vec<Vec<u8>>, u64) {
        let _core = Switched::on(false);
        let flying = flights.then(|| TestFlights::r5_lifting(lift));
        let left = waits_left();
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 4 * K64);
        let other = start + 5 * K64;
        register(0xc0de_5e01, start, len);
        register(0xc0de_5e02, other, K64);
        let (home, second) = (
            find(start, len).expect("a home"),
            find(other, K64).expect("a home"),
        );
        assert!(home.is_home() && second.is_home());
        fill(start, 6 * K64, 0x21);
        let mut fence = std::ptr::null();
        crate::fence::create(
            std::ptr::null(),
            0,
            0,
            &crate::fence::FENCE_IIDS[0],
            &mut fence,
        );
        let sources: Vec<_> = (0..N)
            .map(|i| source(host, &[i as u8 ^ 0x5c; 256]))
            .collect();
        let at = |i: usize| start + ((i as u64 * 9973) % (len - 256)) / 4 * 4;
        let mut lists = Vec::new();
        let mut outs = Vec::new();
        // With the lifts every submission is sent while the ones before are
        // held on the queue, so that each reads what a flight writes (were
        // one to wait, the hold goes after a while).
        let hold = lift.then(|| QueueHold::new(host));
        let released = hold.as_ref().map(|hold| hold.released().after(3000));
        for (i, from) in sources.iter().enumerate() {
            let list = host.new_list().unwrap();
            list.defer_refresh(&home, at(i) as usize, 256);
            // SAFETY: an open list of the host's; live buffers.
            unsafe {
                list.list
                    .CopyBufferRegion(&home.buffer, home.offset(at(i)), from, 0, 256)
            };
            list.defer_gpu_write(&home, at(i) as usize, 256);
            if i > 0 {
                // A fill and a copy of what the submission before wrote.
                let (_, out) = list.scratch(64).expect("scratch");
                list.defer_fill(out, at(i - 1) as *const u8, 64);
                outs.push(out);
                list.defer_copy(
                    (other + 64 * i as u64) as *mut u8,
                    at(i - 1) as *const u8,
                    64,
                );
            }
            // A bottom-of-pipe value past the home, in the region.
            list.defer_write(
                (start + len + 8 * i as u64) as *mut u8,
                crate::queue::Written::Value(i as u64 * 3 + 1),
                8,
            );
            list.close();
            if flights {
                host.send_for_test(&list);
                if !crate::host::flights::signal_behind(
                    crate::fence::FenceRef::new(fence),
                    i as u64 + 1,
                ) {
                    crate::fence::set(fence, i as u64 + 1);
                }
            } else {
                host.execute(&[&list]);
                crate::fence::set(fence, i as u64 + 1);
            }
            lists.push(list);
        }
        if lift {
            assert_eq!(in_air(), N, "no submission waited");
            assert_eq!(
                waits_left(),
                left + N as u64 - 1,
                "each read after the first was lifted"
            );
        }
        drop(hold);
        if let Some(flying) = &flying {
            flying.land_all();
        }
        if let Some(released) = released {
            released.join().unwrap();
        }
        let fills: Vec<Vec<u8>> = outs.iter().map(|&out| bytes(out, 64)).collect();
        let memory = bytes(start as *const u8, 6 * K64 as usize);
        let hash = memory.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
        });
        let value = crate::fence::completed(fence);
        for list in &lists {
            list.reset();
        }
        drop(lists);
        crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(fence, 0x10)(fence);
        unregister(0xc0de_5e01, start);
        unregister(0xc0de_5e02, other);
        drop((home, second));
        drop(flying);
        region.release();
        (hash, fills, value)
    };
    let waited = run(false, false);
    let flown = run(true, false);
    let lifted = run(true, true);
    assert_eq!(waited.2, N as u64);
    for (got, how) in [(&flown, "flown"), (&lifted, "lifted")] {
        assert_eq!(got.2, waited.2, "{how}: the same fence value");
        assert_eq!(got.1, waited.1, "{how}: the same fills");
        assert_eq!(got.0, waited.0, "{how}: the same title memory");
    }
}
