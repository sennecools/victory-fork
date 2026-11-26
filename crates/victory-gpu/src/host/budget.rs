//! The host's side of the frame budget (`crate::frame_budget`): its waits
//! for flights measured, and the line's figures. Kept here so that
//! `host.rs` holds only the calls.

use super::*;
pub(super) use crate::frame_budget::Wait;

/// `wait()`, a wait of the submit thread for flights, measured as one of
/// kind `kind` if it waited (`crate::frame_budget::Measure`).
pub(super) fn measured<T>(kind: Wait, wait: impl FnOnce() -> T) -> T {
    let measure = crate::frame_budget::Measure::start();
    let result = wait();
    measure.end(kind, relay());
    result
}

/// H2: the wait of a submission for the flights that write home bytes
/// under refreshes of copies that are no home
/// ([`refresh_over_flight_homes`]), measured. The microseconds waited, if
/// it waited (as [`relay_wait_with`] says them).
pub(super) fn h2_wait(args: &[(&'static str, u64)], runs: &[&HostList]) -> Option<u64> {
    measured(Wait::H2, || {
        relay_wait_with(WHY_READ, args, |line| {
            !refresh_over_flight_homes(&line.homes_written, runs)
        })
    })
}

/// [`retake_flight_homes`], its time counted.
pub(super) fn retake(writers: &[Writer], runs: &[&HostList]) {
    let retaking = std::time::Instant::now();
    retake_flight_homes(writers, runs);
    crate::frame_budget::retaken(retaking.elapsed());
}

/// The frame budget's line, per frame over `frames`.
pub(super) fn line(frames: f64) {
    crate::frame_budget::line(
        frames,
        STARTED.load(Ordering::Relaxed),
        crate::descriptor::views_in_use(),
        u64::from(victory_shader::emit::view_heap_limit()) + 1,
    );
}
