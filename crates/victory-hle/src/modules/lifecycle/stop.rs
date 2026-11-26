//! The orderly stop the run's tooling asks for (`VICTORY_STOP_EVENT`): the
//! name of a manual-reset event it made. A thread of ours waits for it;
//! once it is set the title ends as when its window is closed
//! ([`super::end_title`]): it is suspended, the pipeline cache is saved and
//! the title's memory is released (`victory_gpu::release`) before the
//! process ends. Without it, the tooling's forced stop ends the process with
//! none of that. A relaunched `victory` inherits the variable and waits for
//! the same event. An event that cannot be opened is logged once, and the
//! run goes on as without the variable.

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{
    INFINITE, OpenEventW, SYNCHRONIZATION_SYNCHRONIZE, WaitForSingleObject,
};

/// The variable that names the event.
pub(crate) const STOP_EVENT: &str = "VICTORY_STOP_EVENT";

/// Starts the wait for the event `VICTORY_STOP_EVENT` names, if it names
/// one; a set event ends the title ([`super::end_title`]).
pub fn watch_stop_event() {
    let Some(name) = std::env::var_os(STOP_EVENT) else {
        return;
    };
    let name = name.to_string_lossy().into_owned();
    if watch(&name, super::stop_requested) {
        tracing::info!(
            event = name,
            "waiting for the run's orderly stop ({STOP_EVENT})"
        );
    }
}

/// Opens the event `name` (to wait on it only) and runs `stop` once on a
/// thread of its own when it is set; whether it was opened.
pub(crate) fn watch(name: &str, stop: impl FnOnce() + Send + 'static) -> bool {
    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: a nul-terminated name; the handle is closed by the thread.
    let event = unsafe { OpenEventW(SYNCHRONIZATION_SYNCHRONIZE, 0, wide.as_ptr()) };
    if event.is_null() {
        // SAFETY: a plain query.
        let error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        tracing::warn!(
            event = name,
            error,
            "{STOP_EVENT} names no event this process can open; the run ends as without it"
        );
        return false;
    }
    let event = Handle(event);
    let spawned = std::thread::Builder::new()
        .name("victory stop event".into())
        .spawn(move || {
            let event = event;
            // SAFETY: our handle, opened with SYNCHRONIZE.
            if unsafe { WaitForSingleObject(event.0, INFINITE) } == WAIT_OBJECT_0 {
                stop();
            }
        });
    if spawned.is_err() {
        tracing::warn!("no thread to wait for {STOP_EVENT}; the run ends as without it");
        return false;
    }
    true
}

/// The event's handle, closed when the waiting thread is done with it.
struct Handle(HANDLE);

// SAFETY: a kernel handle, usable from any thread.
unsafe impl Send for Handle {}

impl Drop for Handle {
    fn drop(&mut self) {
        // SAFETY: our own handle, closed once.
        unsafe { CloseHandle(self.0) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;
    use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

    /// A manual-reset event of the test's, as the run's tooling makes one.
    struct Event(HANDLE, String);

    impl Event {
        fn new(tag: &str) -> Event {
            let name = format!("Local\\victory-stop-test-{tag}-{}", std::process::id());
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: a nul-terminated name; no attributes.
            let handle = unsafe { CreateEventW(std::ptr::null(), 1, 0, wide.as_ptr()) };
            assert!(!handle.is_null());
            Event(handle, name)
        }

        fn set(&self) {
            // SAFETY: our own event.
            assert_ne!(unsafe { SetEvent(self.0) }, 0);
        }
    }

    // SAFETY: a kernel handle, usable from any thread.
    unsafe impl Sync for Event {}

    impl Drop for Event {
        fn drop(&mut self) {
            // SAFETY: our own handle.
            unsafe { CloseHandle(self.0) };
        }
    }

    /// An event set from another thread runs the stop once, and only then.
    #[test]
    fn a_set_event_runs_the_stop_once() {
        let event = Event::new("set");
        let (tell, told) = mpsc::channel();
        assert!(watch(&event.1, move || tell.send(()).unwrap()));
        assert!(
            told.recv_timeout(Duration::from_millis(200)).is_err(),
            "not before it is set"
        );
        std::thread::scope(|s| {
            s.spawn(|| event.set());
        });
        told.recv_timeout(Duration::from_secs(5))
            .expect("the stop ran");
        assert!(
            told.recv_timeout(Duration::from_millis(200)).is_err(),
            "once (its sender went with it)"
        );
    }

    /// A name no event has is not watched: the run goes on as without it.
    #[test]
    fn an_event_that_cannot_be_opened_is_not_watched() {
        let name = format!("Local\\victory-stop-test-none-{}", std::process::id());
        assert!(!watch(&name, || panic!("never")));
    }
}
