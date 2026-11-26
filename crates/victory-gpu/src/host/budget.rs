//! The host's side of the frame budget (`crate::frame_budget`): its waits
//! for flights measured, which copies' refreshes made a submission wait for
//! H2, and the line's figures. Kept here so that `host.rs` holds only the
//! calls.

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
/// (`homes`, in order and joined, as it was before the wait) under refreshes
/// of copies that are no home ([`refresh_over_flight_homes`]), measured, and
/// the refreshes that made it wait sent to the H2 table. The microseconds
/// waited, if it waited (as [`relay_wait_with`] says them).
pub(super) fn h2_wait(
    args: &[(&'static str, u64)],
    homes: &[(usize, usize)],
    runs: &[&HostList],
) -> Option<u64> {
    let measure = crate::frame_budget::Measure::start();
    let waited = relay_wait_with(WHY_READ, args, |line| {
        !refresh_over_flight_homes(&line.homes_written, runs)
    });
    if let Some(us) = measure.end(Wait::H2, true) {
        h2_waited(us, homes, runs);
        if let Some((class, bytes)) = h2_class(homes, runs) {
            crate::frame_budget::h2_class(class, bytes, us);
        }
    }
    waited
}

/// The class of an H2 wait: that of the first refresh in `runs` of a copy
/// that is no home over `homes` (as [`refresh_over_flight_homes`] finds
/// it, but counting nothing: that one is also the wait's predicate, asked
/// at every landing), and that refresh's bytes over `homes`.
fn h2_class(
    homes: &[(usize, usize)],
    runs: &[&HostList],
) -> Option<(crate::frame_budget::H2Class, u64)> {
    for list in runs {
        for step in &list.deferred().copies {
            for (shadow, at, len) in step.refreshes() {
                let end = at.saturating_add(len);
                if shadow.is_home() {
                    continue;
                }
                let Some(first) = crate::frame_budget::first_overlap(homes, at, end) else {
                    continue;
                };
                let bytes = crate::frame_budget::overlap_bytes(homes, at, end);
                let class = crate::frame_budget::H2Class::of(
                    shadow.origin(),
                    shadow.carded(),
                    in_home_pages(first, at, end),
                );
                return Some((class, bytes));
            }
        }
    }
    None
}

/// Whether `[at, end)` lies inside the 64 KiB pages of the home that holds
/// title byte `first`.
fn in_home_pages(first: usize, at: usize, end: usize) -> bool {
    const PAGE: u64 = 1 << 16;
    crate::shadow::home_of(first as u64, 1).is_some_and(|home| {
        let (start, len) = home.span();
        let (from, to) = (start / PAGE * PAGE, (start + len).next_multiple_of(PAGE));
        from <= at as u64 && end as u64 <= to
    })
}

/// A submission waited `us` microseconds for H2: each refresh of a copy
/// that is no home over `homes` goes to the H2 table with its copy's kind
/// and its bytes over `homes`.
fn h2_waited(us: u64, homes: &[(usize, usize)], runs: &[&HostList]) {
    let mut refreshes = Vec::new();
    // One list's lock at a time, and not under the line's.
    for list in runs {
        for step in &list.deferred().copies {
            for (shadow, at, len) in step.refreshes() {
                if shadow.is_home() {
                    continue;
                }
                let alias = crate::frame_budget::overlap_bytes(homes, at, at.saturating_add(len));
                if alias > 0 {
                    refreshes.push((shadow.budget_key(), alias));
                }
            }
        }
    }
    crate::frame_budget::h2_waited(us, &refreshes);
}

/// [`retake_flight_homes`], its time counted.
pub(super) fn retake(writers: &[Writer], runs: &[&HostList]) {
    let retaking = std::time::Instant::now();
    retake_flight_homes(writers, runs);
    crate::frame_budget::retaken(retaking.elapsed());
}

/// The frame budget's two lines, per frame over `frames`.
pub(super) fn line(frames: f64) {
    crate::frame_budget::line(
        frames,
        STARTED.load(Ordering::Relaxed),
        crate::descriptor::views_in_use(),
        u64::from(victory_shader::emit::view_heap_limit()) + 1,
    );
    crate::frame_budget::wave1_line(frames);
}
