//! The host's part of the release at the title's end (`crate::release`):
//! no submission after it, a bounded wait for what the queue holds, and the
//! flights and retired objects let go. Here because it needs the host's own
//! fields and statics.

use std::sync::atomic::AtomicU64;
use std::time::Instant;

use super::*;

/// Set by the release: every submission from then on is dropped
/// ([`refused`]).
#[cfg(not(test))]
static ENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

// Tests: the thread that released (the test) alone sees it, so that tests
// running beside it still submit, open and resolve.
#[cfg(test)]
thread_local! {
    static ENDING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Submissions dropped since [`ENDING`] was set.
static REFUSED: AtomicU64 = AtomicU64::new(0);

/// No submission reaches the card from now on.
pub(crate) fn end_submissions() {
    #[cfg(not(test))]
    {
        ENDING.store(true, Ordering::SeqCst);
        // The card-awake sends (`VICTORY_CARD_AWAKE`) go on their own queue,
        // which the release's wait does not cover: none after this, and what
        // was on the way has run. Not under test, where ENDING is per thread
        // and the awake tests own the switch.
        super::awake::end();
        // The sweeper lets nothing go from now on; what it holds waits
        // for the release's burial (`bury_all`).
        super::sweeper::stop();
    }
    #[cfg(test)]
    ENDING.with(|ending| ending.set(true));
}

/// Whether submissions are dropped ([`end_submissions`]).
#[cfg(not(test))]
pub(crate) fn ending() -> bool {
    ENDING.load(Ordering::SeqCst)
}

/// Whether submissions are dropped ([`end_submissions`]).
#[cfg(test)]
pub(crate) fn ending() -> bool {
    ENDING.with(std::cell::Cell::get)
}

/// Asked by [`Host::submit`] with the submission lock held: whether this
/// submission is dropped, as every one after the release is. The first is
/// logged; the release has taken the lock after the flag was set, so a
/// submission that got past this ran before its wait.
pub(super) fn refused(lists: usize) -> bool {
    if !ending() {
        return false;
    }
    if REFUSED.fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::info!(
            lists,
            "memory release: a submission after the release is dropped (the title is ending)"
        );
    }
    true
}

/// Submissions dropped after the release so far (tests).
#[cfg(test)]
pub(crate) fn refused_count() -> u64 {
    REFUSED.load(Ordering::Relaxed)
}

/// Tests: submissions reach the card again.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    ENDING.with(|ending| ending.set(false));
}

/// The host if it was made: the release never makes one.
pub(crate) fn host_made() -> Option<&'static Host> {
    HOST.get().and_then(Option::as_ref)
}

/// How a wait for the queue ended ([`Host::drain`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Drained {
    /// Everything sent to the queue before it has run.
    Reached,
    /// The deadline came first: the card may still use what was sent.
    TimedOut,
    /// The device is removed: nothing runs any more.
    Removed,
}

impl Host {
    /// Whether the device is removed (its fence then reads all ones).
    pub(crate) fn removed(&self) -> bool {
        // SAFETY: plain queries.
        unsafe {
            self.fence.GetCompletedValue() == u64::MAX
                || self.device.GetDeviceRemovedReason().is_err()
        }
    }

    /// Waits, until `until`, for everything the queue was sent: the
    /// submission lock is taken first (a submission under way finishes its
    /// sending; one after it finds [`ending`] set), then a fence of its own
    /// is signalled on the queue and waited for. If the lock is not had by
    /// the deadline (the thread that holds it waits for something else, or
    /// is this one) the wait has timed out: that thread got past
    /// [`refused`] before the release began and may still send lists, which
    /// reach what the release would let go through view heap slots and the
    /// raw-address directory, so nothing may be let go.
    pub(crate) fn drain(&self, until: Instant) -> Drained {
        if self.removed() {
            return Drained::Removed;
        }
        let held = loop {
            match self.submitted.try_lock() {
                Ok(held) => break held,
                Err(std::sync::TryLockError::Poisoned(e)) => break e.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) if Instant::now() >= until => {
                    return if self.removed() {
                        Drained::Removed
                    } else {
                        Drained::TimedOut
                    };
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        };
        // SAFETY: plain D3D12 calls on objects of ours; the event is ours
        // and waited on here only.
        let reached = unsafe {
            let Ok(fence) = self
                .device
                .CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)
            else {
                return if self.removed() {
                    Drained::Removed
                } else {
                    Drained::TimedOut
                };
            };
            if self.queue.Signal(&fence, 1).is_err() {
                return if self.removed() {
                    Drained::Removed
                } else {
                    Drained::TimedOut
                };
            }
            drop(held);
            if fence.GetCompletedValue() >= 1 {
                true
            } else {
                let Ok(event) = CreateEventW(None, false, false, None) else {
                    return Drained::TimedOut;
                };
                let left = until.saturating_duration_since(Instant::now());
                let waited = fence.SetEventOnCompletion(1, event).is_ok()
                    && WaitForSingleObject(event, left.as_millis() as u32).0 == 0;
                windows_sys::Win32::Foundation::CloseHandle(event.0 as _);
                waited && fence.GetCompletedValue() >= 1
            }
        };
        match reached {
            _ if self.removed() => Drained::Removed,
            true => Drained::Reached,
            false => Drained::TimedOut,
        }
    }
}

/// The flights in the line ([`late`]) and the lists of landed ones not yet
/// given back are let go, with what they would do once landed (their
/// copy-back, the title's fence signals behind them): the title is ending
/// and the card is done with them. Returns the flights let go.
pub(crate) fn let_go_flights() -> u64 {
    let (items, landed) = {
        let mut line = line();
        line.written.clear();
        line.homes_written.clear();
        line.home_writers.clear();
        line.lists = 0;
        (
            std::mem::take(&mut line.line),
            std::mem::take(&mut line.landed),
        )
    };
    let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
    drop(items);
    drop(landed);
    LINE_MOVED.notify_all();
    flights as u64
}

/// Drops every retired object ([`retire`]): only once the card is done with
/// everything sent ([`Host::drain`]). Returns how many there were.
pub(crate) fn bury_all() -> u64 {
    let all = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    let n = all.len() as u64;
    drop(all);
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    /// A submission that holds the lock past the deadline may still send
    /// lists: the wait times out (so nothing is let go) and does not
    /// signal around it. With the lock free again the queue drains.
    ///
    /// With the fence signalled without the lock, as first written, an idle
    /// queue read `Reached` here.
    #[test]
    fn warp_drain_times_out_while_a_submission_holds_the_lock() {
        let Some(host) = host() else { return };
        std::thread::scope(|scope| {
            let (held, is_held) = channel();
            let (go, gone) = channel::<()>();
            scope.spawn(move || {
                let _lock = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
                held.send(()).expect("the test waits");
                let _ = gone.recv();
            });
            is_held.recv().expect("the lock is held");
            let began = Instant::now();
            let drained = host.drain(began + Duration::from_millis(200));
            let _ = go.send(());
            assert_eq!(drained, Drained::TimedOut);
            assert!(
                began.elapsed() >= Duration::from_millis(200),
                "until the deadline"
            );
        });
        assert_eq!(
            host.drain(Instant::now() + Duration::from_secs(5)),
            Drained::Reached
        );
    }
}
