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

use std::sync::{Arc, Condvar, Mutex};

/// Whether the heap sync runs ahead on its own thread
/// (`VICTORY_HEAP_SYNC_AHEAD=1`).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_AHEAD").is_some_and(|v| v == "1"))
}

/// The helper of the process, started at the first request; `None` if no
/// thread could be started (the pass then runs in place).
static AHEAD: std::sync::OnceLock<Option<Ahead>> = std::sync::OnceLock::new();

/// At a submission's start: asks the helper for a pass over the heaps, and
/// returns the request's number for [`sync_ahead_wait`]. `None` with
/// `VICTORY_HEAP_SYNC_AHEAD` off, or with no helper: the caller runs
/// [`super::sync_heaps`] in place.
pub(crate) fn sync_ahead() -> Option<u64> {
    if !on() {
        return None;
    }
    AHEAD
        .get_or_init(|| Ahead::start(super::sync_heaps))
        .as_ref()?
        .request()
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
pub(crate) fn sync_ahead_wait(seq: u64) {
    wait_and_make(AHEAD.get().and_then(Option::as_ref), seq);
}

/// [`sync_ahead_wait`] with the helper `ahead` (`None`: none, the pass runs
/// here).
pub(super) fn wait_and_make(ahead: Option<&Ahead>, seq: u64) {
    match ahead {
        None => super::sync_heaps(),
        Some(ahead) => {
            let behind = ahead.behind(seq);
            if behind > 0 {
                let _wait = crate::profile::span_args(
                    "sync heaps wait",
                    &[("seq", seq), ("behind", behind)],
                );
                if !ahead.wait(seq) {
                    super::sync_heaps();
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
}

/// A thread that runs `pass` for requests, as described above.
pub(super) struct Ahead {
    shared: Arc<(Mutex<State>, Condvar)>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Ahead {
    /// Starts the helper, with `pass` as its pass. `None` if no thread
    /// could be started.
    pub(super) fn start(pass: impl Fn() + Send + 'static) -> Option<Self> {
        let shared = Arc::new((Mutex::new(State::default()), Condvar::new()));
        let thread = std::thread::Builder::new()
            .name("heap sync".into())
            .spawn({
                let shared = shared.clone();
                move || run(&shared, pass)
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

    /// A request for a pass: its number. `None` once the helper failed.
    pub(super) fn request(&self) -> Option<u64> {
        let seq = {
            let mut state = self.state();
            if state.failed {
                return None;
            }
            state.asked += 1;
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
/// to stop with none left.
fn run(shared: &(Mutex<State>, Condvar), pass: impl Fn()) {
    let (state, changed) = shared;
    loop {
        // Read before the pass looks at any page: what it covers.
        let covering = {
            let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
            while state.asked == state.done && !state.stop {
                state = changed.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.asked == state.done {
                return;
            }
            state.asked
        };
        let passed = {
            let _pass = crate::profile::span_args("sync heaps ahead", &[("covering", covering)]);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(&pass))
        };
        let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
        if passed.is_err() {
            tracing::error!(
                "the heap sync thread failed: the heap sync runs in place from here on"
            );
            state.failed = true;
        } else {
            state.done = state.done.max(covering);
        }
        drop(state);
        changed.notify_all();
        if passed.is_err() {
            return;
        }
    }
}
