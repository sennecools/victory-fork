//! The heap sync run ahead of the submission, on a thread of its own
//! (`VICTORY_HEAP_SYNC_AHEAD=1`, off by default; off, [`super::sync_heaps`]
//! runs in place in the submission, as before).
//!
//! Lists read the host heap when they run, so a record the title copied
//! into its heap as bytes before its `ExecuteCommandLists` must be in the
//! host heap before ours: the pass for a submission must start after the
//! title's call and end before the submission goes to the card. In place it
//! runs on the submitting thread before the submission is prepared (45 to
//! 49 ms a frame in the R5 profile), though nothing in the preparing reads
//! what it makes. Here the submission asks for a pass when it begins
//! ([`sync_ahead`]), and waits for it only if it has not ended by the time
//! the lists go to the card ([`sync_ahead_wait`]): the waits for flights
//! and the preparing run meanwhile.
//!
//! A pass covers the requests made before it began: the helper reads the
//! number of the last request before the pass looks at a page, and marks
//! that number done when it ends. The title's bytes written before its
//! call come before our request in the title thread's order, and the
//! pass's look at the pages after it, so a pass that covers a request
//! sees them. A pass that began before a request never satisfies it.
//!
//! The submitting thread waits while it holds the submission lock. The
//! pass takes nothing the submitting thread holds there: in place it ran
//! under that lock on the same thread, whose mutex is not reentrant, so
//! nothing on its way takes the lock. Other threads already mirror the
//! same slots while it runs (draws' tables, the heap buffer pass); the
//! helper is one more of them, ordered by the slots' claims as they are.
//!
//! A request names the heaps its submission's lists bind; a pass looks at
//! the heaps of the requests it covers (`super::sync_heaps`).
//!
//! With `VICTORY_HEAP_SYNC_EARLY=<ms>` (off by default) the helper also
//! runs passes of its own while submissions come ([`early`]), over the
//! heaps of the requests the last pass covered; those answer no request.

use std::sync::{Arc, Condvar, Mutex};

/// Whether the heap sync runs ahead on its own thread
/// (`VICTORY_HEAP_SYNC_AHEAD=1`).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_AHEAD").is_some_and(|v| v == "1"))
}

/// How long the helper waits with no request before it runs a pass of its
/// own, while submissions come (`VICTORY_HEAP_SYNC_EARLY=<ms>`, with
/// `VICTORY_HEAP_SYNC_AHEAD=1`; unset or 0, the default: none, it waits for
/// a request, as before). The title copies its records into its heaps while
/// it records, and the first submission of a frame found all of them for
/// its pass to do (cool_fox measured 8 ms waited for); a pass of its own
/// meanwhile leaves the requested one only what was written since. Such a
/// pass answers no request: a request is answered by a pass begun after it,
/// as without the switch, and one that comes during an own pass waits for
/// that and then its own. Only within [`EARLY_WINDOW`] of the last request:
/// a title that stops submitting is not looked at for ever.
///
/// Ported from cool_fox f6db479 (4 ms by default there), by local.
fn early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("VICTORY_HEAP_SYNC_EARLY")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// How long after the last request the helper still runs passes of its own
/// ([`early`]).
pub(super) const EARLY_WINDOW: std::time::Duration = std::time::Duration::from_millis(500);

/// The helper of the process, started at the first request; `None` if no
/// thread could be started (the pass then runs in place).
static AHEAD: std::sync::OnceLock<Option<Ahead>> = std::sync::OnceLock::new();

/// At a submission's start: asks the helper for a pass over `heaps` (the
/// heaps its lists bind), and returns the request's number for
/// [`sync_ahead_wait`]. `None` with `VICTORY_HEAP_SYNC_AHEAD` off, or with
/// no helper: the caller runs [`super::sync_heaps`] in place.
pub(crate) fn sync_ahead(heaps: &[usize]) -> Option<u64> {
    if !on() {
        return None;
    }
    AHEAD
        .get_or_init(|| Ahead::start_with(super::sync_heaps, early()))
        .as_ref()?
        .request(heaps)
}

/// Before the submission of request `seq` ([`sync_ahead`]) goes to the
/// card: waits until a pass that began after the request has ended. On the
/// profile, a wait is `sync heaps wait` (`seq`; `behind`: requests not yet
/// covered when it began). If the helper failed, the pass runs here.
///
/// Then the slots a pass left waiting for a texture whose host copy was
/// made meanwhile are made ([`super::make_waiting`]), as when the pass ran
/// in place before the submission's own `make_waiting`: the next pass sees
/// their records as the ones last seen, and leaves them.
pub(crate) fn sync_ahead_wait(seq: u64, heaps: &[usize]) {
    wait_and_make(AHEAD.get().and_then(Option::as_ref), seq, heaps);
}

/// [`sync_ahead_wait`] with the helper `ahead` (`None`: none, the pass over
/// `heaps`, the request's, runs here).
pub(super) fn wait_and_make(ahead: Option<&Ahead>, seq: u64, heaps: &[usize]) {
    match ahead {
        None => super::sync_heaps(heaps),
        Some(ahead) => {
            let behind = ahead.behind(seq);
            if behind > 0 {
                let _wait = crate::profile::span_args(
                    "sync heaps wait",
                    &[("seq", seq), ("behind", behind)],
                );
                if !ahead.wait(seq) {
                    super::sync_heaps(heaps);
                }
            }
        }
    }
    // (One lock when no texture was made.)
    super::make_waiting();
}

/// What the helper and the submissions share.
#[derive(Default)]
struct State {
    /// The number of the last request.
    asked: u64,
    /// The number of the last request a pass that ended covered.
    done: u64,
    /// The helper is to end once it has covered every request (tests).
    stop: bool,
    /// A pass panicked: the helper is gone, and passes run in place.
    failed: bool,
    /// When the last request was made ([`early`]).
    asked_at: Option<std::time::Instant>,
    /// The heaps of the requests no pass has begun to cover, each once.
    heaps: Vec<usize>,
    /// The heaps the last pass that covered requests looked at: what a
    /// pass of its own looks at ([`early`]).
    recent: Vec<usize>,
}

/// A thread that runs `pass` over the heaps of requests, as described
/// above.
pub(super) struct Ahead {
    shared: Arc<(Mutex<State>, Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Ahead {
    /// Starts the helper, with `pass` as its pass. `None` if no thread
    /// could be started.
    #[cfg(test)]
    pub(super) fn start(pass: impl Fn(&[usize]) + Send + 'static) -> Option<Self> {
        Self::start_with(pass, None)
    }

    /// [`Ahead::start`], with passes of its own every `early` while
    /// requests come ([`early`]; `None`: none).
    pub(super) fn start_with(
        pass: impl Fn(&[usize]) + Send + 'static,
        early: Option<std::time::Duration>,
    ) -> Option<Self> {
        let shared = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let thread = std::thread::Builder::new()
            .name("heap sync".into())
            .spawn({
                let shared = shared.clone();
                move || run(&shared, pass, early)
            })
            .map_err(
                |e| tracing::warn!(error = %e, "no heap sync thread: the heap sync runs in place"),
            )
            .ok()?;
        Some(Self {
            shared,
            thread: Some(thread),
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A request for a pass over `heaps`: its number. `None` once the
    /// helper failed.
    pub(super) fn request(&self, heaps: &[usize]) -> Option<u64> {
        let seq = {
            let mut state = self.state();
            if state.failed {
                return None;
            }
            for &heap in heaps {
                if !state.heaps.contains(&heap) {
                    state.heaps.push(heap);
                }
            }
            state.asked += 1;
            state.asked_at = Some(std::time::Instant::now());
            state.asked
        };
        self.shared.1.notify_all();
        Some(seq)
    }

    /// Requests up to `seq` that no ended pass covered yet.
    pub(super) fn behind(&self, seq: u64) -> u64 {
        seq.saturating_sub(self.state().done)
    }

    /// Waits until a pass that covers request `seq` has ended. False if the
    /// helper failed instead (the pass did not run).
    pub(super) fn wait(&self, seq: u64) -> bool {
        let mut state = self.state();
        while state.done < seq && !state.failed {
            state = self.shared.1.wait(state).unwrap_or_else(|e| e.into_inner());
        }
        !state.failed || state.done >= seq
    }
}

impl Drop for Ahead {
    fn drop(&mut self) {
        self.state().stop = true;
        self.shared.1.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The helper's loop: a pass for the requests made before it, until told
/// to stop with none left; with `early` ([`early`]), passes of its own
/// between them, which cover none.
fn run(
    shared: &(Mutex<State>, Condvar),
    pass: impl Fn(&[usize]),
    early: Option<std::time::Duration>,
) {
    let (state, changed) = shared;
    loop {
        // Read before the pass looks at any page: what it covers (`None`:
        // nothing, a pass of its own), and the heaps it looks at.
        let (covering, heaps) = {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            let covering = loop {
                if state.asked != state.done {
                    break Some(state.asked);
                }
                if state.stop {
                    return;
                }
                let lately = state.asked_at.is_some_and(|at| at.elapsed() < EARLY_WINDOW);
                match early.filter(|_| lately) {
                    None => state = changed.wait(state).unwrap_or_else(|e| e.into_inner()),
                    Some(every) => {
                        let (back, waited) = changed
                            .wait_timeout(state, every)
                            .unwrap_or_else(|e| e.into_inner());
                        state = back;
                        if waited.timed_out() && state.asked == state.done && !state.stop {
                            break None;
                        }
                    }
                }
            };
            if covering.is_some() {
                state.recent = std::mem::take(&mut state.heaps);
            }
            (covering, state.recent.clone())
        };
        let passed = {
            let _pass = match covering {
                Some(covering) => {
                    crate::profile::span_args("sync heaps ahead", &[("covering", covering)])
                }
                None => crate::profile::span("sync heaps early"),
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pass(&heaps)))
        };
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        if passed.is_err() {
            tracing::error!(
                "the heap sync thread failed: the heap sync runs in place from here on"
            );
            state.failed = true;
        } else if let Some(covering) = covering {
            state.done = state.done.max(covering);
        }
        drop(state);
        changed.notify_all();
        if passed.is_err() {
            return;
        }
    }
}
