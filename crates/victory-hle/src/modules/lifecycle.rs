//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first suspends it (its suspend
//! routines run, and it has a second to save) and then ends the process.
//! Closing the title's window does the same here; a new launch then starts
//! `victory` again, which waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
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
        place(asked, menu)
    } else {
        asked
    };
    // SAFETY: the title's own arguments, forwarded (placed on the host screen).
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
    keep_client(hwnd, at.client);
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
        place(asked, menu)
    } else {
        asked
    };
    // SAFETY: the title's own arguments, forwarded (placed on the host screen).
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
    keep_client(hwnd, at.client);
    hook(hwnd);
    hwnd
}

/// How the title's top-level window is sized on the host screen, in screen
/// pixels. Only the window: the title renders at the size it chooses, and
/// the swap chain stretches its frame to the window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WindowSize {
    /// The title's own size (its render size), shrunk to the primary
    /// screen's work area if larger.
    #[default]
    Title,
    /// A client area of `width` x `height`, centred in the primary screen's
    /// work area (shrunk to it, keeping proportions, if it would not fit).
    Pixels { width: u32, height: u32 },
    /// Borderless over the whole primary screen.
    Fullscreen,
}

static WINDOW: OnceLock<WindowSize> = OnceLock::new();

/// Sets how the title's window is sized; call before the title runs. Later
/// calls do nothing.
pub fn set_window(size: WindowSize) {
    if WINDOW.set(size).is_err() {
        tracing::debug!(?size, "window size already set");
    }
}

/// Makes window sizes physical pixels: the process becomes per-monitor DPI
/// aware, so on a screen scaled to 150 % a 1920x1080 window is 1920x1080
/// pixels and its frame is drawn sharp, not scaled up by Windows. Call
/// before any window exists.
///
/// shortcut: a window moved to a screen with another scale keeps its size
/// in pixels (`WM_DPICHANGED` is passed on, not acted on).
pub fn use_physical_pixels() {
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::UI::HiDpi::{
        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, DPI_AWARENESS_PER_MONITOR_AWARE,
        GetAwarenessFromDpiAwarenessContext, GetThreadDpiAwarenessContext,
        SetProcessDpiAwarenessContext,
    };
    // SAFETY: a predefined awareness context.
    if unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) } != 0 {
        return;
    }
    // SAFETY: plain queries of this thread's state.
    let (error, aware) = unsafe {
        (
            GetLastError(),
            GetAwarenessFromDpiAwarenessContext(GetThreadDpiAwarenessContext()),
        )
    };
    // Already aware (a manifest, or a second call) is fine.
    if aware != DPI_AWARENESS_PER_MONITOR_AWARE {
        tracing::warn!(
            error,
            "not DPI aware: Windows scales the window with the display scale"
        );
    }
}

/// A window's styles and rectangle, and for a sized window the client area
/// it must end up with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placement {
    ex_style: u32,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    client: Option<(i32, i32)>,
}

impl Placement {
    fn asked(ex_style: u32, style: u32, x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            ex_style,
            style,
            x,
            y,
            width,
            height,
            client: None,
        }
    }
}

/// What the host screen offers the title's top-level window.
struct Screen {
    /// The primary screen's work area (without the taskbar), if known.
    area: Option<RECT>,
    /// The primary screen's size, if known.
    size: Option<(i32, i32)>,
    /// How much wider and taller a window of the title's style is than its
    /// client area.
    frame: (i32, i32),
}

/// The title's top-level window, placed as [`set_window`] says.
fn place(asked: Placement, menu: HMENU) -> Placement {
    let size = WINDOW.get().copied().unwrap_or_default();
    let mut area = RECT::default();
    // SAFETY: `area` receives the primary screen's work area; plain metric
    // queries of the primary screen.
    let (known, screen) = unsafe {
        (
            wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) != 0,
            (
                wm::GetSystemMetrics(wm::SM_CXSCREEN),
                wm::GetSystemMetrics(wm::SM_CYSCREEN),
            ),
        )
    };
    let screen = Screen {
        area: known.then_some(area),
        size: (screen.0 > 0 && screen.1 > 0).then_some(screen),
        frame: frame(asked.style, asked.ex_style, !menu.is_null()),
    };
    let placed = plan(size, &screen, asked);
    if placed != asked {
        tracing::info!(
            ?size,
            asked = ?(asked.width, asked.height),
            window = ?(placed.width, placed.height),
            client = ?placed.client,
            "title window placed"
        );
    }
    placed
}

/// Where the title's top-level window goes; falls back to the title's own
/// size (shrunk to the work area) when the screen is not known.
fn plan(size: WindowSize, screen: &Screen, asked: Placement) -> Placement {
    match size {
        WindowSize::Fullscreen => {
            if let Some((width, height)) = screen.size {
                return Placement {
                    ex_style: borderless_ex(asked.ex_style),
                    style: borderless(asked.style),
                    x: 0,
                    y: 0,
                    width,
                    height,
                    client: None,
                };
            }
        }
        WindowSize::Pixels { width, height } => {
            if let Some(area) = &screen.area {
                let (fw, fh) = screen.frame;
                // The client area fits what the work area leaves inside a frame.
                let inside = RECT {
                    right: area.right - fw,
                    bottom: area.bottom - fh,
                    ..*area
                };
                let wanted = (
                    i32::try_from(width).unwrap_or(i32::MAX),
                    i32::try_from(height).unwrap_or(i32::MAX),
                );
                let (cw, ch) = fit_into(&inside, wanted.0, wanted.1);
                let (x, y, width, height) = centred(area, cw + fw, ch + fh);
                return Placement {
                    x,
                    y,
                    width,
                    height,
                    client: Some((cw, ch)),
                    ..asked
                };
            }
        }
        WindowSize::Title => {}
    }
    // A console title sizes its window to its render target (the title:
    // 3840x2160 on a Series X), which can exceed the host's
    // screen: then it is shrunk to fit, keeping its proportions, and
    // centred.
    let Some(area) = &screen.area else {
        return asked;
    };
    if fit_into(area, asked.width, asked.height) == (asked.width, asked.height) {
        return asked;
    }
    let (x, y, width, height) = centred(area, asked.width, asked.height);
    Placement {
        x,
        y,
        width,
        height,
        ..asked
    }
}

/// How much wider and taller than its client area a top-level window of
/// these styles is, at the system DPI (an estimate: [`keep_client`]
/// measures the real one).
fn frame(style: u32, ex_style: u32, menu: bool) -> (i32, i32) {
    use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, GetDpiForSystem};
    // Windows gives a top-level window that is not a popup a caption.
    let style = if style & (wm::WS_POPUP | wm::WS_CHILD) == 0 {
        style | wm::WS_CAPTION
    } else {
        style
    };
    let mut rect = RECT::default();
    // SAFETY: `rect` is ours.
    let known = unsafe {
        AdjustWindowRectExForDpi(
            &mut rect,
            style,
            i32::from(menu),
            ex_style,
            GetDpiForSystem(),
        )
    } != 0;
    if known {
        (rect.right - rect.left, rect.bottom - rect.top)
    } else {
        (0, 0)
    }
}

/// A sized window whose client area came out other than planned (its frame
/// differs from the estimate, as on a screen whose scale changed since sign
/// in) is resized by the difference and centred again.
fn keep_client(hwnd: HWND, client: Option<(i32, i32)>) {
    let Some((cw, ch)) = client else {
        return;
    };
    if hwnd.is_null() {
        return;
    }
    let (mut got, mut outer, mut area) = (RECT::default(), RECT::default(), RECT::default());
    // SAFETY: a window this thread just created; the rectangles are ours.
    let known = unsafe {
        wm::GetClientRect(hwnd, &mut got) != 0
            && wm::GetWindowRect(hwnd, &mut outer) != 0
            && wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) != 0
    };
    let (gw, gh) = (got.right - got.left, got.bottom - got.top);
    if !known || (gw, gh) == (cw, ch) {
        return;
    }
    let width = outer.right - outer.left + cw - gw;
    let height = outer.bottom - outer.top + ch - gh;
    let (x, y, width, height) = centred(&area, width, height);
    tracing::debug!(got = ?(gw, gh), want = ?(cw, ch), "title window client area corrected");
    // SAFETY: a window this thread just created.
    unsafe {
        wm::SetWindowPos(
            hwnd,
            std::ptr::null_mut(),
            x,
            y,
            width,
            height,
            wm::SWP_NOZORDER | wm::SWP_NOACTIVATE,
        );
    }
}

/// `width` x `height` shrunk to fit `area` if needed, and centred in it:
/// left, top, width, height.
fn centred(area: &RECT, width: i32, height: i32) -> (i32, i32, i32, i32) {
    let (w, h) = fit_into(area, width, height);
    let left = area.left + (area.right - area.left - w) / 2;
    let top = area.top + (area.bottom - area.top - h) / 2;
    (left, top, w, h)
}

/// `style` without a frame: no caption, border or sizing edge, a popup.
/// The system menu and minimize box stay, so Alt+F4 and the taskbar can
/// still close and minimise it.
fn borderless(style: u32) -> u32 {
    let frame = wm::WS_CAPTION | wm::WS_THICKFRAME | wm::WS_MAXIMIZEBOX;
    (style & !frame) | wm::WS_POPUP
}

/// `ex_style` without the extended frame edges.
fn borderless_ex(ex_style: u32) -> u32 {
    ex_style
        & !(wm::WS_EX_DLGMODALFRAME
            | wm::WS_EX_WINDOWEDGE
            | wm::WS_EX_CLIENTEDGE
            | wm::WS_EX_STATICEDGE)
}

/// `width` x `height` scaled down to fit `area`, or unchanged if it fits
/// (or is `CW_USEDEFAULT`).
fn fit_into(area: &RECT, width: i32, height: i32) -> (i32, i32) {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if width <= 0 || height <= 0 || aw <= 0 || ah <= 0 || (width <= aw && height <= ah) {
        return (width, height);
    }
    let scale = f64::min(
        f64::from(aw) / f64::from(width),
        f64::from(ah) / f64::from(height),
    );
    (
        (f64::from(width) * scale) as i32,
        (f64::from(height) * scale) as i32,
    )
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
        end_title("window closed", Box::new(|| {}));
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
    fn oversized_windows_shrink_to_the_screen() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_into(&area, 3840, 2160), (1848, 1040));
        assert_eq!(fit_into(&area, 1280, 720), (1280, 720));
        assert_eq!(
            fit_into(&area, wm::CW_USEDEFAULT, 0),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn sized_windows_are_centred_and_shrink_only_when_too_big() {
        // A 4K screen with a 60-pixel taskbar.
        let area = RECT {
            left: 0,
            top: 0,
            right: 3840,
            bottom: 2100,
        };
        assert_eq!(centred(&area, 1920, 1080), (960, 510, 1920, 1080));
        assert_eq!(centred(&area, 3840, 2160), (53, 0, 3733, 2100));
        // A work area that does not start at the origin (taskbar on the left).
        let left_bar = RECT {
            left: 100,
            top: 0,
            right: 1920,
            bottom: 1080,
        };
        assert_eq!(centred(&left_bar, 1280, 720), (370, 180, 1280, 720));
    }

    #[test]
    fn borderless_drops_the_frame_and_keeps_the_rest() {
        let style = wm::WS_OVERLAPPEDWINDOW | wm::WS_VISIBLE | wm::WS_CLIPCHILDREN;
        let plain = borderless(style);
        assert_eq!(
            plain & (wm::WS_CAPTION | wm::WS_THICKFRAME | wm::WS_MAXIMIZEBOX),
            0
        );
        assert_ne!(plain & wm::WS_POPUP, 0);
        // Alt+F4 and the taskbar still close and minimise it.
        assert_ne!(plain & wm::WS_SYSMENU, 0);
        assert_ne!(plain & wm::WS_MINIMIZEBOX, 0);
        assert_ne!(plain & wm::WS_VISIBLE, 0);
        assert_ne!(plain & wm::WS_CLIPCHILDREN, 0);
        let ex = borderless_ex(wm::WS_EX_OVERLAPPEDWINDOW | wm::WS_EX_APPWINDOW);
        assert_eq!(ex & (wm::WS_EX_WINDOWEDGE | wm::WS_EX_CLIENTEDGE), 0);
        assert_ne!(ex & wm::WS_EX_APPWINDOW, 0);
    }

    /// A 4K primary screen with a 60-pixel taskbar, and a frame 16 wider
    /// and 39 taller than the client area.
    fn screen() -> Screen {
        Screen {
            area: Some(RECT {
                left: 0,
                top: 0,
                right: 3840,
                bottom: 2100,
            }),
            size: Some((3840, 2160)),
            frame: (16, 39),
        }
    }

    fn asked(width: i32, height: i32) -> Placement {
        Placement::asked(0, wm::WS_OVERLAPPEDWINDOW, 7, 8, width, height)
    }

    #[test]
    fn a_pixel_size_is_a_client_area_centred_in_the_work_area() {
        let placed = plan(
            WindowSize::Pixels {
                width: 1920,
                height: 1080,
            },
            &screen(),
            asked(3840, 2160),
        );
        assert_eq!(placed.client, Some((1920, 1080)));
        assert_eq!((placed.width, placed.height), (1936, 1119));
        assert_eq!((placed.x, placed.y), (952, 490));
        assert_eq!(placed.style, wm::WS_OVERLAPPEDWINDOW);
    }

    #[test]
    fn a_pixel_size_too_big_for_the_screen_keeps_its_proportions() {
        let placed = plan(
            WindowSize::Pixels {
                width: 7680,
                height: 4320,
            },
            &screen(),
            asked(3840, 2160),
        );
        let (cw, ch) = placed.client.unwrap();
        // The client fits what the work area leaves inside the frame.
        assert!(cw + 16 <= 3840 && ch + 39 <= 2100, "{cw}x{ch}");
        // As tall as the work area allows (less a rounding pixel).
        assert!(ch >= 2060, "{ch}");
        assert!((f64::from(cw) / f64::from(ch) - 16.0 / 9.0).abs() < 0.01);
        assert_eq!((placed.width, placed.height), (cw + 16, ch + 39));
    }

    #[test]
    fn fullscreen_covers_the_screen_without_a_frame() {
        let placed = plan(WindowSize::Fullscreen, &screen(), asked(3840, 2160));
        assert_eq!(
            (placed.x, placed.y, placed.width, placed.height),
            (0, 0, 3840, 2160)
        );
        assert_eq!(placed.style & wm::WS_CAPTION, 0);
        assert_ne!(placed.style & wm::WS_POPUP, 0);
        assert_eq!(placed.client, None);
    }

    #[test]
    fn the_title_size_is_kept_when_it_fits_and_fitted_when_not() {
        assert_eq!(
            plan(WindowSize::Title, &screen(), asked(1280, 720)),
            asked(1280, 720)
        );
        let placed = plan(WindowSize::Title, &screen(), asked(3840, 2160));
        assert_eq!(
            (placed.x, placed.y, placed.width, placed.height),
            (53, 0, 3733, 2100)
        );
    }

    #[test]
    fn an_unknown_screen_falls_back_to_what_the_title_asked() {
        let unknown = Screen {
            area: None,
            size: None,
            frame: (0, 0),
        };
        for size in [
            WindowSize::Title,
            WindowSize::Fullscreen,
            WindowSize::Pixels {
                width: 1920,
                height: 1080,
            },
        ] {
            assert_eq!(
                plan(size, &unknown, asked(3840, 2160)),
                asked(3840, 2160),
                "{size:?}"
            );
        }
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
