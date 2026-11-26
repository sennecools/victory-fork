//! The release of the title's memory at its end
//! (`VICTORY_MEMORY_RELEASE_AT_END`, on by default; `=0` keeps what was done
//! before: nothing).
//!
//! The memory core opens the title's own memory as heaps: chunks of the
//! page pool, sections committed whole, through
//! `OpenExistingHeapFromFileMapping` (`crate::pool`), and parts of its
//! private regions through `OpenExistingHeapFromAddress1` (`crate::home`).
//! The host and the chunk table are statics, never dropped, and every way
//! the process ends is a `TerminateProcess`. Measured on the card after a
//! clean boot, a run with the pool on left about 3.4 GB of the system's
//! commit charge behind once the process was gone (0.5 GB with the core
//! and no pool, 0.7 GB with the core off), which only a restart gave back:
//! the sections opened as heaps are not let go when the process is run
//! down. So before each `TerminateProcess` the runtime itself makes,
//! [`title_memory`] lets them go in an order that keeps the card from
//! reading anything freed:
//!
//! 1. no submission reaches the card any more (one after this is dropped,
//!    with a line in the log the first time: `crate::host::end_submissions`);
//! 2. everything sent is waited for, until the deadline; on a removed device
//!    nothing is waited for. If the card is still busy at the deadline, or a
//!    submission that began before step 1 still holds the submission lock
//!    (it may still send lists), nothing is let go (a heap freed under a
//!    running list is a page fault, and a device removal the run's tooling
//!    counts).
//!    A heap sync pass still running (the ahead helper's) is waited for
//!    too, under the same deadline; later passes do nothing;
//! 3. the flights of the relay and their lists go, with what they would do
//!    once landed;
//! 4. the view heap slots that name a home's or a chunk's buffer let it go;
//! 5. every home and every `VICTORY_INPLACE` placement is let go (retired)
//!    and the directory of raw-address shaders goes whole, with the copies
//!    and homes it numbers;
//! 6. the chunk table is emptied: each opened chunk's heap and buffer go;
//! 7. what was retired goes, and with the last `Shadow` of a home of the
//!    title's private memory its claim (and its heap over those pages);
//! 8. the queue is waited for once more, so that the runtime's deferred
//!    destruction runs.
//!
//! It runs once, on a thread of its own, which the caller waits for until
//! the deadline and a little more: a caller that holds a lock of the
//! runtime's (a host GPU failure is reported from inside a submission) then
//! ends the process anyway. One line in the log says what was let go, the
//! chunks still held by something the release did not reach, and the
//! process's and the system's commit charge before and after.
//!
//! What it does not reach: lists the title recorded and did not submit keep
//! what they name until they are reset; a home or a chunk buffer held by
//! one of them is counted as held (`chunks_held`, `claims_left`). The
//! device itself stays (the host is a `OnceLock`). And nothing covers a
//! process that ends without the runtime: a kill from outside (the run's
//! tooling asks for an orderly stop first, `VICTORY_STOP_EVENT` in
//! `victory-hle`), a fail-fast (`0xC0000409`, which runs no code of ours)
//! or a hang.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::core::Interface;

/// Whether the release runs (`VICTORY_MEMORY_RELEASE_AT_END`, on unless
/// `0`). Read once.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        switch(
            std::env::var("VICTORY_MEMORY_RELEASE_AT_END")
                .ok()
                .as_deref(),
        )
    });
    on && !test_off()
}

/// The switch's value `value`: on unless it is `0`.
fn switch(value: Option<&str>) -> bool {
    value.is_none_or(|v| v.trim() != "0")
}

#[cfg(test)]
thread_local! {
    static TEST_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_off() -> bool {
    TEST_OFF.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn test_off() -> bool {
    false
}

/// Set by the first [`title_memory`]: later calls do nothing.
static BEGUN: AtomicBool = AtomicBool::new(false);

/// Set by the release's thread once it is done: a later call made while it
/// runs waits for this (until its own deadline), so that the process does
/// not end halfway through.
static DONE: AtomicBool = AtomicBool::new(false);

/// Whether the release has begun: nothing is opened as a home or a heap of
/// the title's memory any more, and no submission runs.
pub(crate) fn released() -> bool {
    crate::host::ending()
}

/// How the wait for the card ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Waited {
    /// Everything sent has run.
    Done,
    /// The deadline came first; nothing was let go.
    Busy,
    /// The device is removed: nothing was waited for.
    Removed,
    /// No host device was ever made.
    NoDevice,
}

/// What [`title_memory`] let go, in counts and MB.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Released {
    pub waited: Waited,
    /// Chunks of the pool opened as heaps let go, their MB, and those whose
    /// buffer something the release did not reach still held.
    pub chunks: u64,
    pub chunk_mb: u64,
    pub chunks_held: u64,
    /// Homes let go, numbers of the raw-address directory let go, view heap
    /// slots that let a home's or chunk's buffer go, flights let go, retired
    /// objects dropped.
    pub homes: u64,
    pub numbered: u64,
    pub slots: u64,
    pub flights: u64,
    pub retired: u64,
    /// Claims of homes on the title's private memory still standing after
    /// it, and their MB (homes something still holds).
    pub claims_left: u64,
    pub claims_left_mb: u64,
    /// The process's private commit and the system's commit charge, in MB,
    /// before and after (0: not read).
    pub private_mb: [u64; 2],
    pub commit_mb: [u64; 2],
    /// Milliseconds it took.
    pub ms: u64,
}

impl Released {
    fn new(waited: Waited) -> Released {
        Released {
            waited,
            chunks: 0,
            chunk_mb: 0,
            chunks_held: 0,
            homes: 0,
            numbered: 0,
            slots: 0,
            flights: 0,
            retired: 0,
            claims_left: 0,
            claims_left_mb: 0,
            private_mb: [0; 2],
            commit_mb: [0; 2],
            ms: 0,
        }
    }
}

/// How much longer than its deadline the caller waits for the release's
/// thread before it gives up on it.
const GRACE: Duration = Duration::from_millis(500);

/// Releases the title's memory the core opened as heaps (see the module),
/// waiting for the card at most `deadline`; returns what it let go. `None`:
/// the switch is off, it ran already, or it did not finish within the
/// deadline and [`GRACE`] (the caller goes on ending the process). A call
/// while another runs waits for that one as long (and returns `None`). Safe
/// to call from any thread and more than once.
pub fn title_memory(deadline: Duration) -> Option<Released> {
    if !on() {
        return None;
    }
    if BEGUN.swap(true, Ordering::SeqCst) {
        let until = Instant::now() + deadline + GRACE;
        while !DONE.load(Ordering::SeqCst) && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(5));
        }
        return None;
    }
    crate::host::end_submissions();
    let until = Instant::now() + deadline;
    let (tell, told) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("victory memory release".into())
        .spawn(move || {
            let released = release(until);
            DONE.store(true, Ordering::SeqCst);
            let _ = tell.send(released);
        });
    if spawned.is_err() {
        tracing::warn!("memory release: no thread for it; nothing released");
        return None;
    }
    match told.recv_timeout(deadline + GRACE) {
        Ok(released) => Some(released),
        Err(_) => {
            tracing::warn!(
                deadline_ms = deadline.as_millis() as u64,
                "memory release: not done in time; the process ends without it"
            );
            None
        }
    }
}

/// The release itself, on its own thread (see the module for the order).
fn release(until: Instant) -> Released {
    let begun = Instant::now();
    let private = private_mb();
    let commit = commit_mb();
    let host = crate::host::host_made();
    let waited = match host.map(|host| host.drain(until)) {
        None => Waited::NoDevice,
        Some(crate::host::Drained::Reached) => Waited::Done,
        Some(crate::host::Drained::TimedOut) => Waited::Busy,
        Some(crate::host::Drained::Removed) => Waited::Removed,
    };
    let mut released = Released::new(waited);
    released.private_mb[0] = private;
    released.commit_mb[0] = commit;
    if waited == Waited::Busy {
        released.ms = begun.elapsed().as_millis() as u64;
        tracing::warn!(
            ms = released.ms,
            "memory release: the card was still busy at the deadline; nothing released"
        );
        return released;
    }
    if !crate::descriptor::sync_quiet(until) {
        released.ms = begun.elapsed().as_millis() as u64;
        tracing::warn!(
            ms = released.ms,
            "memory release: a heap sync pass still ran at the deadline; nothing released"
        );
        return released;
    }
    released.flights = crate::host::let_go_flights();
    let buffers = crate::home::home_buffers();
    released.slots = crate::descriptor::let_go_slots(&|resource| {
        buffers.contains(&(resource.as_raw() as usize)) || crate::pool::is_chunk_buffer(resource)
    });
    drop(buffers);
    released.homes = crate::home::release_all();
    released.numbered = crate::memory::release_all();
    let chunks = crate::pool::release_all();
    released.chunks = chunks.opened;
    released.chunk_mb = chunks.bytes >> 20;
    released.chunks_held = chunks.held;
    released.retired = crate::host::bury_all();
    let (claims, bytes) = crate::home::claims_left();
    released.claims_left = claims;
    released.claims_left_mb = bytes >> 20;
    if waited == Waited::Done
        && let Some(host) = host
    {
        let _ = host.drain(until.max(Instant::now() + Duration::from_millis(100)));
    }
    released.private_mb[1] = private_mb();
    released.commit_mb[1] = commit_mb();
    released.ms = begun.elapsed().as_millis() as u64;
    let r = &released;
    tracing::info!(
        waited = ?r.waited,
        released_chunks = r.chunks,
        released_chunk_mb = r.chunk_mb,
        chunks_held = r.chunks_held,
        homes = r.homes,
        numbered = r.numbered,
        slots = r.slots,
        flights = r.flights,
        retired = r.retired,
        claims_left = r.claims_left,
        claims_left_mb = r.claims_left_mb,
        private_mb_before = r.private_mb[0],
        private_mb_after = r.private_mb[1],
        commit_mb_before = r.commit_mb[0],
        commit_mb_after = r.commit_mb[1],
        ms = r.ms,
        "memory release at the title's end (VICTORY_MEMORY_RELEASE_AT_END=0 turns it off)"
    );
    released
}

/// The process's private commit, in MB (0 if not read).
fn private_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure, of which the call
    // fills the extended form when told its size.
    let read = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    };
    if read == 0 {
        0
    } else {
        c.PrivateUsage as u64 >> 20
    }
}

/// The system's commit charge, in MB (0 if not read).
fn commit_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetPerformanceInfo, PERFORMANCE_INFORMATION,
    };
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized structure the call fills.
    let read = unsafe { K32GetPerformanceInfo(&mut info, info.cb) };
    if read == 0 {
        0
    } else {
        (info.CommitTotal as u64).saturating_mul(info.PageSize as u64) >> 20
    }
}

/// Tests: as before any release (submissions run, homes are made again).
#[cfg(test)]
pub(crate) fn reset_for_test() {
    BEGUN.store(false, Ordering::SeqCst);
    DONE.store(false, Ordering::SeqCst);
    crate::host::reset_for_test();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::home::tests::Switched;
    use windows_sys::Win32::System::Memory as mem;

    const K64: u64 = 64 << 10;

    /// The release's switch: on unless it is 0.
    #[test]
    fn the_switch_is_on_unless_it_is_zero() {
        assert!(switch(None));
        assert!(switch(Some("1")));
        assert!(switch(Some("")));
        assert!(!switch(Some("0")));
        assert!(!switch(Some(" 0 ")));
    }

    /// Back to before any release when the test ends, however it ends.
    struct Reset;

    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_OFF.with(|off| off.set(false));
            reset_for_test();
        }
    }

    /// A section as `xmem` makes a committed chunk of the pool, told to the
    /// GPU side as one, and its handle.
    fn committed_chunk(bytes: u64) -> usize {
        // SAFETY: plain section creation.
        let handle = unsafe {
            mem::CreateFileMappingW(
                windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                std::ptr::null(),
                mem::PAGE_READWRITE | mem::SEC_COMMIT,
                0,
                bytes as u32,
                std::ptr::null(),
            )
        } as usize;
        assert_ne!(handle, 0);
        crate::inplace::title_memory_section(handle, bytes, true);
        crate::pool::title_memory_pool_chunk(handle, bytes, crate::pool::PoolChunk::Committed);
        handle
    }

    /// The handle `handle`'s section is told gone and closed.
    fn close_chunk(handle: usize) {
        crate::pool::title_memory_pool_chunk_gone(handle);
        // SAFETY: the test's own handle.
        unsafe { windows_sys::Win32::Foundation::CloseHandle(handle as _) };
    }

    /// Whether the tests were asked to run on the software adapter
    /// (`VICTORY_D3D12_WARP=1`): then it must open sections as heaps, and a
    /// test that cannot open its chunk fails instead of passing empty.
    fn warp_asked() -> bool {
        std::env::var("VICTORY_D3D12_WARP").is_ok_and(|v| v.trim() == "1")
    }

    /// The host on the software adapter, if it can open a section as a
    /// heap (`ID3D12Device3`).
    fn warp() -> Option<&'static crate::host::Host> {
        let host = crate::host::host().filter(|host| {
            host.device
                .cast::<windows::Win32::Graphics::Direct3D12::ID3D12Device3>()
                .is_ok()
        });
        assert!(
            host.is_some() || !warp_asked(),
            "VICTORY_D3D12_WARP=1: a host that opens sections as heaps"
        );
        host
    }

    /// A chunk opened as a heap: the release empties the table, lets its
    /// buffer out of the set of chunk buffers, and drops every reference it
    /// held to the heap (only the test's own is left); a second release does
    /// nothing; the section's handle then closes as any other.
    #[test]
    fn warp_release_frees_an_opened_chunk_and_runs_once() {
        let Some(host) = warp() else { return };
        let _on = Switched::on(false);
        crate::pool::set_test(true);
        let _reset = Reset;
        let handle = committed_chunk(4 * K64);
        let placed = match crate::pool::home(handle, K64, K64) {
            Ok(placed) => placed,
            // The adapter opens no section as a heap: nothing to release.
            Err(why) => {
                close_chunk(handle);
                assert!(!warp_asked(), "the chunk is opened: {why:?}");
                return;
            }
        };
        assert_eq!(
            crate::pool::title_memory_pool_chunk_known(handle),
            Some(true)
        );
        let heap = placed.heap.clone();
        let buffer = placed.buffer.clone();
        drop(placed);
        assert!(crate::pool::is_chunk_buffer(&buffer));
        drop(buffer);
        let before = crate::host::references(&heap);
        assert!(before > 1, "the table holds the heap too");
        let released = title_memory(Duration::from_secs(5)).expect("the first release runs");
        assert_eq!(released.waited, Waited::Done);
        assert_eq!(released.chunks, 1);
        assert_eq!(released.chunk_mb, (4 * K64) >> 20);
        assert_eq!(released.chunks_held, 0, "nothing else held its buffer");
        assert_eq!(crate::pool::title_memory_pool_chunk_known(handle), None);
        assert_eq!(crate::host::references(&heap), 1, "only the test's own");
        assert!(title_memory(Duration::from_secs(5)).is_none(), "once");
        assert!(
            crate::pool::home(handle, 0, K64).is_err(),
            "nothing is opened after it"
        );
        drop(heap);
        close_chunk(handle);
        let _ = host;
    }

    /// After the release a submission is dropped, and counted; before it
    /// one runs.
    #[test]
    fn warp_release_refuses_submissions_after_it() {
        let Some(host) = warp() else { return };
        let _on = Switched::on(false);
        let _reset = Reset;
        let submit = || {
            let list = host.new_list().expect("a host list");
            list.close();
            host.execute(&[&list]);
        };
        let refused = crate::host::refused_count();
        submit();
        assert_eq!(crate::host::refused_count(), refused, "before: it runs");
        assert!(title_memory(Duration::from_secs(5)).is_some());
        submit();
        assert_eq!(crate::host::refused_count(), refused + 1, "after: dropped");
        assert!(released());
        // Nor is a home found or made, or the directory made again.
        let memory = vec![0u8; 4 * K64 as usize];
        assert!(matches!(
            crate::home::resolve(memory.as_ptr() as u64, K64, false, false, &mut || false),
            crate::home::Found::Copy(_)
        ));
        let list = host.new_list().expect("a host list");
        let heap = crate::descriptor::host::HostHeap::new(false, 4).expect("a host heap");
        assert!(crate::memory::ready(&list, &heap, &[], &[], false).is_none());
        list.close();
    }

    /// The card still busy at the deadline (the queue waits for a fence the
    /// test signals only afterwards): nothing is let go, and the opened
    /// chunk stays opened. A second caller meanwhile returns only once the
    /// first is done.
    #[test]
    fn warp_release_lets_nothing_go_while_the_card_is_busy() {
        let Some(host) = warp() else { return };
        let _on = Switched::on(false);
        crate::pool::set_test(true);
        let _reset = Reset;
        let handle = committed_chunk(2 * K64);
        match crate::pool::home(handle, 0, K64) {
            Ok(placed) => drop(placed),
            Err(why) => {
                close_chunk(handle);
                assert!(!warp_asked(), "the chunk is opened: {why:?}");
                return;
            }
        }
        let hold = crate::host::flights::QueueHold::new(host);
        let second = std::thread::spawn(|| {
            while !BEGUN.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            let again = title_memory(Duration::from_secs(5));
            (again.is_none(), DONE.load(Ordering::SeqCst))
        });
        let began = Instant::now();
        let released = title_memory(Duration::from_millis(500)).expect("the release runs");
        assert!(
            began.elapsed() >= Duration::from_millis(500),
            "until the deadline"
        );
        assert_eq!(released.waited, Waited::Busy);
        assert_eq!(
            (released.chunks, released.homes, released.retired),
            (0, 0, 0)
        );
        assert_eq!(
            crate::pool::title_memory_pool_chunk_known(handle),
            Some(true),
            "the chunk stays opened"
        );
        assert_eq!(second.join().expect("the second caller"), (true, true));
        drop(hold);
        close_chunk(handle);
    }

    /// A heap sync pass still running at the deadline (the ahead helper's,
    /// which a refused submission does not wait for): nothing is let go, and
    /// the opened chunk stays opened, since the pass would keep it again.
    #[test]
    fn warp_release_lets_nothing_go_while_a_heap_sync_pass_runs() {
        let Some(_host) = warp() else { return };
        let _on = Switched::on(false);
        crate::pool::set_test(true);
        let _reset = Reset;
        let handle = committed_chunk(2 * K64);
        match crate::pool::home(handle, 0, K64) {
            Ok(placed) => drop(placed),
            Err(why) => {
                close_chunk(handle);
                assert!(!warp_asked(), "the chunk is opened: {why:?}");
                return;
            }
        }
        let pass = crate::descriptor::hold_sync_for_test();
        let released = title_memory(Duration::from_millis(300)).expect("the release runs");
        assert_eq!(released.waited, Waited::Done, "the queue itself was idle");
        assert_eq!(
            (released.chunks, released.homes, released.retired),
            (0, 0, 0)
        );
        assert_eq!(
            crate::pool::title_memory_pool_chunk_known(handle),
            Some(true),
            "the chunk stays opened"
        );
        drop(pass);
        close_chunk(handle);
    }

    /// With the switch off nothing is done: no release, submissions run,
    /// and an opened chunk stays opened.
    #[test]
    fn warp_release_switched_off_changes_nothing() {
        let Some(host) = warp() else { return };
        let _on = Switched::on(false);
        crate::pool::set_test(true);
        let _reset = Reset;
        TEST_OFF.with(|off| off.set(true));
        let handle = committed_chunk(2 * K64);
        let opened = crate::pool::home(handle, 0, K64).is_ok();
        assert!(opened || !warp_asked(), "the chunk is opened");
        assert!(title_memory(Duration::from_secs(5)).is_none());
        assert!(!released());
        let refused = crate::host::refused_count();
        let list = host.new_list().expect("a host list");
        list.close();
        host.execute(&[&list]);
        assert_eq!(crate::host::refused_count(), refused, "it runs");
        assert_eq!(
            crate::pool::title_memory_pool_chunk_known(handle),
            Some(opened),
            "the chunk stays as it was"
        );
        close_chunk(handle);
    }
}
