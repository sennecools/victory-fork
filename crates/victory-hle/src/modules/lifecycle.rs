//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first suspends it (its suspend
//! routines run, and it has a second to save) and then ends the process.
//! Closing the window (the title's, or the one `victory-gpu` presents in)
//! does the same here; a new launch then starts `victory` again, which
//! waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use victory_gpu::window::{Placement, place};
use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HMENU};
use windows_sys::core::{PCSTR, PCWSTR, w};

use crate::{HleModule, hle_export};

type Routine = extern "system" fn(quiesced: u8, context: *mut c_void);

/// Registration id, routine and context of each suspend routine.
static ROUTINES: Mutex<Vec<(usize, Routine, usize)>> = Mutex::new(Vec::new());

fn routines() -> MutexGuard<'static, Vec<(usize, Routine, usize)>> {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long the console lets suspend routines run before it ends the title.
const SUSPEND_DEADLINE: Duration = Duration::from_secs(1);

const ERROR_INVALID_PARAMETER: u32 = 87;

pub extern "system" fn RegisterAppStateChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    let Some(routine) = routine else {
        return ERROR_INVALID_PARAMETER;
    };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    routines().push((id, routine, context as usize));
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    routines().retain(|&(id, ..)| id != registration);
}

/// The window procedure a title window had before ours.
const TITLE_PROC: PCWSTR = w!("victory.wndproc");

/// The title's windows, with our procedure in front of the title's; a
/// top-level one larger than the host screen is shrunk to fit
/// ([`victory_gpu::window::place`]), whatever `[window] size` says.
pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: PCWSTR,
    name: PCWSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let asked = Placement::asked(ex_style, style, x, y, width, height);
    let at = if parent.is_null() {
        place(asked)
    } else {
        asked
    };
    // SAFETY: the title's own arguments, forwarded (the size fitted).
    let hwnd = unsafe {
        wm::CreateWindowExW(
            at.ex_style,
            class,
            name,
            at.style,
            at.x,
            at.y,
            at.width,
            at.height,
            parent,
            menu,
            instance,
            param,
        )
    };
    hook(hwnd);
    hwnd
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: PCSTR,
    name: PCSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let asked = Placement::asked(ex_style, style, x, y, width, height);
    let at = if parent.is_null() {
        place(asked)
    } else {
        asked
    };
    // SAFETY: the title's own arguments, forwarded (the size fitted).
    let hwnd = unsafe {
        wm::CreateWindowExA(
            at.ex_style,
            class,
            name,
            at.style,
            at.x,
            at.y,
            at.width,
            at.height,
            parent,
            menu,
            instance,
            param,
        )
    };
    hook(hwnd);
    hwnd
}

/// Puts our procedure in front of the title's.
fn hook(hwnd: HWND) {
    if hwnd.is_null() {
        return;
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = wm::GetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC);
        wm::SetPropW(hwnd, TITLE_PROC, title as *mut c_void);
        wm::SetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC, window_proc as *const () as isize);
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // A user close (close button, Alt+F4, window or taskbar menu); a
    // console title never sees one, nor a WM_CLOSE from it.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_CLOSE as usize {
        window_closed();
        return 0;
    }
    // Alt or F10 puts a window into menu mode: the system runs a modal menu
    // loop on the thread that owns the window, which is the title's. The
    // title then stops (no frames, no input) until a key ends the loop, and
    // nothing shows it, since the window has no menu bar; the title
    // aborts itself about half a minute later. A console has no window
    // menu. The mouse still opens it from the title bar.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_KEYMENU as usize {
        return 0;
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = wm::GetPropW(hwnd, TITLE_PROC);
        if message == wm::WM_NCDESTROY {
            wm::RemovePropW(hwnd, TITLE_PROC);
        }
        let title = std::mem::transmute::<*mut c_void, wm::WNDPROC>(title);
        wm::CallWindowProcW(title, hwnd, message, wparam, lparam)
    }
}

/// The user closed the title's window or the one `victory-gpu` presents in
/// ([`victory_gpu::window::on_close`]): the title ends as a console ends
/// it. Returns at once, so a window thread keeps pumping meanwhile.
pub(crate) fn window_closed() {
    end_title("window closed", Box::new(|| {}));
}

/// Suspends the title as the console does when it ends it, runs `then`
/// (once), then ends the process. Returns at once; later calls do nothing.
pub(crate) fn end_title(why: &'static str, then: Box<dyn FnOnce() + Send>) {
    static ENDING: AtomicBool = AtomicBool::new(false);
    if ENDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let suspend = routines().clone();
    tracing::info!(routines = suspend.len(), "{why}: suspending the title");
    // Whoever comes first (the routines or the deadline) runs `then`; the
    // other waits for it, so the process never ends halfway through.
    let then = std::sync::Arc::new(Mutex::new(Some(then)));
    let finish = move |how: &str| {
        let mut then = then.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!("{how}; ending the title");
        if let Some(then) = then.take() {
            then();
        }
        end();
    };
    let late = finish.clone();
    // Not on the calling thread: titles hand the suspend to their window
    // thread and wait for it inside the routine.
    std::thread::spawn(move || {
        for (_, routine, context) in suspend {
            routine(1, context as *mut c_void);
        }
        finish("title suspended");
    });
    std::thread::spawn(move || {
        std::thread::sleep(SUSPEND_DEADLINE);
        late("title did not finish suspending in time");
    });
}

/// Set in the environment of a relaunched `victory`: an inherited handle
/// of the process it replaces.
const RELAUNCHED_FROM: &str = "VICTORY_RELAUNCHED_FROM";

/// Starts `victory` again with `args` (after the title has suspended); the
/// new process waits until this one has ended.
pub(crate) fn relaunch(args: Vec<std::ffi::OsString>) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    let mut me: HANDLE = std::ptr::null_mut();
    // SAFETY: an inheritable duplicate of our own process handle.
    let duplicated = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentProcess(),
            GetCurrentProcess(),
            &mut me,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } != 0;
    let spawned = std::env::current_exe().and_then(|exe| {
        let mut command = std::process::Command::new(exe);
        command.args(&args);
        if duplicated {
            command.env(RELAUNCHED_FROM, (me as usize).to_string());
        }
        command.spawn()
    });
    match spawned {
        Ok(child) => tracing::info!(pid = child.id(), ?args, "relaunched"),
        Err(e) => tracing::warn!(%e, "relaunch failed; the title just ends"),
    }
}

/// In a relaunched `victory`: waits (30 s at most) for the process it
/// replaces to be gone, so the two never share the title's files.
pub fn wait_for_previous() {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let Some(handle) = std::env::var(RELAUNCHED_FROM)
        .ok()
        .and_then(|h| h.parse::<usize>().ok())
    else {
        return;
    };
    // SAFETY: removing our own variable before any other thread reads it.
    unsafe { std::env::remove_var(RELAUNCHED_FROM) };
    let handle = handle as HANDLE;
    // SAFETY: the inherited process handle the previous process gave us.
    unsafe {
        WaitForSingleObject(handle, 30_000);
        CloseHandle(handle);
    }
}

fn end() {
    // Compiled host pipelines kept for the next run.
    #[cfg(windows)]
    victory_gpu::pipeline::save_pipeline_cache();
    // SAFETY: ends this process, as the console ends a suspended title.
    unsafe { TerminateProcess(GetCurrentProcess(), 0) };
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "lifecycle",
        dlls: &[
            "api-ms-win-core-psm-appnotify-*",
            "ext-ms-win-rtcore-ntuser-window-*",
        ],
        exports: vec![
            hle_export!(RegisterAppStateChangeNotification),
            hle_export!(UnregisterAppStateChangeNotification),
            hle_export!(CreateWindowExW),
            hle_export!(CreateWindowExA),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = routines().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    #[test]
    fn hooked_windows_still_reach_the_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("before"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        let mut text = [0u16; 16];
        // SAFETY: our window; the buffer holds 16 characters.
        unsafe {
            wm::SetWindowTextW(hwnd, w!("after"));
            let n = wm::GetWindowTextW(hwnd, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "after");
            assert!(!wm::GetPropW(hwnd, TITLE_PROC).is_null());
            wm::DestroyWindow(hwnd);
        }
    }

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them (no menu loop, no minimising).
    extern "system" fn recording(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == wm::WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        // SAFETY: the system's default procedure, called as the system would.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("menu"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns. `recording` stands in for the
        // title's procedure and the original is put back before the window
        // is destroyed.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            let command = |c: u32, key: u8| {
                wm::SendMessageW(hwnd, wm::WM_SYSCOMMAND, c as usize, LPARAM::from(key))
            };
            command(wm::SC_KEYMENU, 0); // Alt or F10
            command(wm::SC_KEYMENU, b' '); // Alt+Space
            command(wm::SC_KEYMENU | 1, 0); // the low bits are the system's
            command(wm::SC_MINIMIZE, 0);
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::SC_MINIMIZE as usize]);
    }
}
