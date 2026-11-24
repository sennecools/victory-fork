//! The size, place and closing of the window the title is shown in, on the
//! host's primary screen. That is the window the host presents in when the
//! title shows none, which is the usual case (a console title keeps its
//! windows hidden); the title's own top-level windows (created through
//! `victory-hle`'s window hooks) are only fitted to the screen.

use std::sync::OnceLock;

use windows_sys::Win32::Foundation::{HWND, RECT};
use windows_sys::Win32::UI::WindowsAndMessaging as wm;

/// How the window the host presents in is sized on the host screen, in
/// screen pixels. Only the window: the title renders at the size it
/// chooses, and the swap chain stretches its frame to the window.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum WindowSize {
    /// A client area of the frame's size, shrunk to the primary screen's
    /// work area (keeping proportions) if larger, and always centred in it.
    /// A title window that fits keeps its position ([`place`]).
    #[default]
    Title,
    /// A client area of `width` x `height`, centred in the primary screen's
    /// work area (shrunk to it, keeping proportions, if it would not fit).
    Pixels { width: u32, height: u32 },
    /// Borderless over the whole primary screen.
    Fullscreen,
}

static WINDOW: OnceLock<WindowSize> = OnceLock::new();

/// Sets how the window the host presents in is sized; call before the
/// title runs. Later calls do nothing.
pub fn set_window(size: WindowSize) {
    if WINDOW.set(size).is_err() {
        tracing::debug!(?size, "window size already set");
    }
}

fn window_size() -> WindowSize {
    WINDOW.get().copied().unwrap_or_default()
}

/// What a user close of the window the host presents in does (the
/// runtime's: suspend the title, then end it).
static ON_CLOSE: OnceLock<fn()> = OnceLock::new();

/// Sets what closing the window the host presents in does (close button,
/// Alt+F4, the window or taskbar menu). Without it the window is destroyed
/// and the title runs on without one. Later calls do nothing.
pub fn on_close(close: fn()) {
    let _ = ON_CLOSE.set(close);
}

/// Runs what [`on_close`] set; `false` if nothing is set.
pub(crate) fn close() -> bool {
    run_close(ON_CLOSE.get())
}

/// Runs `hook`, if there is one; whether there was.
fn run_close(hook: Option<&fn()>) -> bool {
    match hook {
        Some(close) => {
            close();
            true
        }
        None => false,
    }
}

/// What the window the host presents in does with a message before the
/// default handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Handling {
    /// A user close: [`close`], or the default handling if it does nothing.
    Close,
    /// Dropped.
    Ignore,
    /// The default handling.
    Default,
}

/// How the window the host presents in handles `message`.
pub(crate) fn presenting_message(message: u32, wparam: usize) -> Handling {
    let command = (message == wm::WM_SYSCOMMAND).then_some(wparam & 0xFFF0);
    if message == wm::WM_CLOSE || command == Some(wm::SC_CLOSE as usize) {
        return Handling::Close;
    }
    // Alt or F10 puts the window into menu mode, and the keys the keyboard
    // pad uses (arrows, Enter) then work the window menu, where they can
    // move, minimise or close the window. A console has no window menu.
    // The mouse still opens it from the title bar.
    if command == Some(wm::SC_KEYMENU as usize) {
        return Handling::Ignore;
    }
    Handling::Default
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
pub struct Placement {
    pub ex_style: u32,
    pub style: u32,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub client: Option<(i32, i32)>,
}

impl Placement {
    /// A window as its creator asked for it, with no client area to keep.
    pub fn asked(ex_style: u32, style: u32, x: i32, y: i32, width: i32, height: i32) -> Self {
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

/// The primary screen's work area (without the taskbar), if known.
fn work_area() -> Option<RECT> {
    let mut area = RECT::default();
    // SAFETY: `area` receives the primary screen's work area.
    let known =
        unsafe { wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) };
    (known != 0).then_some(area)
}

/// What the host screen offers the window the host presents in.
struct Screen {
    /// The primary screen's work area, if known.
    area: Option<RECT>,
    /// The primary screen's size, if known.
    size: Option<(i32, i32)>,
    /// How much wider and taller the window is than its client area.
    frame: (i32, i32),
}

impl Screen {
    /// The primary screen, for a window of [`PRESENTING`] styles.
    fn primary() -> Self {
        // SAFETY: plain metric queries of the primary screen.
        let size = unsafe {
            (
                wm::GetSystemMetrics(wm::SM_CXSCREEN),
                wm::GetSystemMetrics(wm::SM_CYSCREEN),
            )
        };
        Self {
            area: work_area(),
            size: (size.0 > 0 && size.1 > 0).then_some(size),
            frame: frame(PRESENTING, 0),
        }
    }
}

/// The title's top-level window on the host screen: shrunk to the primary
/// screen's work area if larger, whatever [`set_window`] says. The title
/// may read back its window's client size or styles, so its windows are
/// not resized to another size or restyled; `"<width>x<height>"` and
/// `"fullscreen"` apply to the window the host presents in.
pub fn place(asked: Placement) -> Placement {
    let placed = fit_title(work_area().as_ref(), asked);
    if placed != asked {
        tracing::info!(
            asked = ?(asked.width, asked.height),
            window = ?(placed.width, placed.height),
            "title window fitted to the screen"
        );
    }
    placed
}

/// A title's top-level window as asked if it fits `area` (or `area` is not
/// known), else shrunk to it, keeping its proportions, and centred.
fn fit_title(area: Option<&RECT>, asked: Placement) -> Placement {
    // A console title sizes its window to its render target (the title:
    // 3840x2160 on a Series X), which can exceed the host's
    // screen.
    let Some(area) = area else {
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

/// The styles of the window the host presents in when the title shows
/// none: an ordinary window with a caption, shown once placed.
const PRESENTING: u32 = wm::WS_OVERLAPPEDWINDOW;

/// The window the host presents a `width` x `height` frame in, placed as
/// [`set_window`] says.
pub(crate) fn place_presenting(width: u32, height: u32) -> Placement {
    let size = window_size();
    let placed = plan_presenting(size, &Screen::primary(), width, height);
    tracing::info!(
        ?size,
        frame = ?(width, height),
        window = ?(placed.width, placed.height),
        client = ?placed.client,
        "presenting window placed"
    );
    placed
}

/// Where the presenting window for a `width` x `height` frame goes, as
/// [`WindowSize`] says. Falls back to a window whose client area is the
/// frame's size, where Windows puts it, when the screen is not known.
fn plan_presenting(size: WindowSize, screen: &Screen, width: u32, height: u32) -> Placement {
    let side = |n: u32| i32::try_from(n).unwrap_or(i32::MAX);
    let frame_size = (side(width), side(height));
    let (fw, fh) = screen.frame;
    let asked = Placement::asked(
        0,
        PRESENTING,
        wm::CW_USEDEFAULT,
        wm::CW_USEDEFAULT,
        frame_size.0.saturating_add(fw),
        frame_size.1.saturating_add(fh),
    );
    let client = match size {
        WindowSize::Fullscreen => {
            if let Some((width, height)) = screen.size {
                return Placement {
                    style: borderless(PRESENTING),
                    x: 0,
                    y: 0,
                    width,
                    height,
                    ..asked
                };
            }
            // The screen's size is not known: as "title".
            frame_size
        }
        WindowSize::Pixels { width, height } => (side(width), side(height)),
        WindowSize::Title => frame_size,
    };
    let Some(area) = &screen.area else {
        return asked;
    };
    let ((x, y, width, height), client) = framed(area, screen.frame, client);
    Placement {
        x,
        y,
        width,
        height,
        client: Some(client),
        ..asked
    }
}

/// A window `frame` wider and taller than its client area, whose client
/// area is `client` shrunk (keeping proportions) to what `area` leaves
/// inside the frame, centred in `area`: left, top, width and height, and
/// the client area.
fn framed(
    area: &RECT,
    (fw, fh): (i32, i32),
    (cw, ch): (i32, i32),
) -> ((i32, i32, i32, i32), (i32, i32)) {
    let inside = RECT {
        right: area.right - fw,
        bottom: area.bottom - fh,
        ..*area
    };
    let (cw, ch) = fit_into(&inside, cw, ch);
    (centred(area, cw + fw, ch + fh), (cw, ch))
}

/// How much wider and taller than its client area a top-level window of
/// these styles (with a caption, no menu bar) is, at the system DPI (an
/// estimate: [`keep_client`] measures the real one).
fn frame(style: u32, ex_style: u32) -> (i32, i32) {
    use windows_sys::Win32::UI::HiDpi::{AdjustWindowRectExForDpi, GetDpiForSystem};
    let mut rect = RECT::default();
    // SAFETY: `rect` is ours.
    let known =
        unsafe { AdjustWindowRectExForDpi(&mut rect, style, 0, ex_style, GetDpiForSystem()) } != 0;
    if known {
        (rect.right - rect.left, rect.bottom - rect.top)
    } else {
        (0, 0)
    }
}

/// A sized window whose client area came out other than planned (its frame
/// differs from the estimate, as on a screen whose scale changed since sign
/// in) is given the planned client area inside its real frame, shrunk if
/// that no longer fits the work area, and centred again. Call on the thread
/// that just created `hwnd`.
pub(crate) fn keep_client(hwnd: HWND, client: Option<(i32, i32)>) {
    let Some(want) = client else {
        return;
    };
    if hwnd.is_null() {
        return;
    }
    let (mut got, mut outer) = (RECT::default(), RECT::default());
    // SAFETY: a window this thread just created; the rectangles are ours.
    let measured = unsafe {
        wm::GetClientRect(hwnd, &mut got) != 0 && wm::GetWindowRect(hwnd, &mut outer) != 0
    };
    if !measured {
        return;
    }
    let Some(area) = work_area() else {
        return;
    };
    let got_client = (got.right - got.left, got.bottom - got.top);
    if got_client == want {
        return;
    }
    let real_frame = (
        outer.right - outer.left - got_client.0,
        outer.bottom - outer.top - got_client.1,
    );
    let ((x, y, width, height), client) = framed(&area, real_frame, want);
    tracing::debug!(got = ?got_client, ?want, ?client, "window client area corrected");
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let area = work_area_4k();
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
    }

    /// A 4K primary screen's work area, with a 60-pixel taskbar.
    fn work_area_4k() -> RECT {
        RECT {
            left: 0,
            top: 0,
            right: 3840,
            bottom: 2100,
        }
    }

    /// A 4K primary screen with a 60-pixel taskbar, and a frame 16 wider
    /// and 39 taller than the client area.
    fn screen() -> Screen {
        Screen {
            area: Some(work_area_4k()),
            size: Some((3840, 2160)),
            frame: (16, 39),
        }
    }

    fn asked(width: i32, height: i32) -> Placement {
        let ex_style = wm::WS_EX_APPWINDOW;
        Placement::asked(ex_style, wm::WS_OVERLAPPEDWINDOW, 7, 8, width, height)
    }

    #[test]
    fn a_title_window_is_kept_when_it_fits_and_fitted_when_not() {
        let area = work_area_4k();
        assert_eq!(fit_title(Some(&area), asked(1280, 720)), asked(1280, 720));
        let placed = fit_title(Some(&area), asked(3840, 2160));
        assert_eq!(
            (placed.x, placed.y, placed.width, placed.height),
            (53, 0, 3733, 2100)
        );
        // Never restyled, and no client area forced on it.
        assert_eq!(
            (placed.ex_style, placed.style, placed.client),
            (wm::WS_EX_APPWINDOW, wm::WS_OVERLAPPEDWINDOW, None)
        );
    }

    #[test]
    fn a_title_window_on_an_unknown_screen_is_as_asked() {
        assert_eq!(fit_title(None, asked(3840, 2160)), asked(3840, 2160));
    }

    #[test]
    fn the_presenting_window_fits_a_4k_frame_to_the_work_area() {
        // The case this exists for: a 3840x2160 frame on a 4K screen.
        let placed = plan_presenting(WindowSize::Title, &screen(), 3840, 2160);
        let (cw, ch) = placed.client.unwrap();
        // Frame and all inside the work area, centred, with the frame's
        // proportions.
        assert_eq!((placed.width, placed.height), (cw + 16, ch + 39));
        assert!(placed.width <= 3840 && placed.height <= 2100, "{placed:?}");
        assert!(placed.x >= 0 && placed.y >= 0, "{placed:?}");
        assert_eq!(placed.x, (3840 - placed.width) / 2);
        assert_eq!(placed.y, (2100 - placed.height) / 2);
        assert!((f64::from(cw) / f64::from(ch) - 16.0 / 9.0).abs() < 0.01);
        assert_eq!((placed.ex_style, placed.style), (0, PRESENTING));
    }

    #[test]
    fn a_presenting_window_that_fits_keeps_the_frame_size_centred() {
        let placed = plan_presenting(WindowSize::Title, &screen(), 1280, 720);
        assert_eq!(placed.client, Some((1280, 720)));
        assert_eq!((placed.width, placed.height), (1296, 759));
        assert_eq!((placed.x, placed.y), (1272, 670));
    }

    #[test]
    fn a_pixel_size_sets_the_presenting_window_not_the_frame() {
        let placed = plan_presenting(
            WindowSize::Pixels {
                width: 1920,
                height: 1080,
            },
            &screen(),
            3840,
            2160,
        );
        assert_eq!(placed.client, Some((1920, 1080)));
        assert_eq!(
            (placed.x, placed.y, placed.width, placed.height),
            (952, 490, 1936, 1119)
        );
        assert_eq!(placed.style, PRESENTING);
    }

    #[test]
    fn a_pixel_size_too_big_for_the_screen_keeps_its_proportions() {
        let placed = plan_presenting(
            WindowSize::Pixels {
                width: 7680,
                height: 4320,
            },
            &screen(),
            3840,
            2160,
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
    fn a_fullscreen_presenting_window_is_a_borderless_popup() {
        let placed = plan_presenting(WindowSize::Fullscreen, &screen(), 3840, 2160);
        assert_eq!(
            (placed.x, placed.y, placed.width, placed.height),
            (0, 0, 3840, 2160)
        );
        assert_eq!(placed.style, borderless(PRESENTING));
        assert_eq!(placed.style & wm::WS_CAPTION, 0);
        assert_ne!(placed.style & wm::WS_POPUP, 0);
        assert_eq!(placed.client, None);
    }

    #[test]
    fn without_a_screen_the_presenting_window_frames_the_frame_size() {
        let framed = Screen {
            area: None,
            size: None,
            frame: (16, 39),
        };
        for size in [
            WindowSize::Title,
            WindowSize::Fullscreen,
            WindowSize::Pixels {
                width: 1920,
                height: 1080,
            },
        ] {
            let placed = plan_presenting(size, &framed, 3840, 2160);
            assert_eq!(
                placed,
                Placement::asked(
                    0,
                    PRESENTING,
                    wm::CW_USEDEFAULT,
                    wm::CW_USEDEFAULT,
                    3856,
                    2199
                ),
                "{size:?}"
            );
        }
    }

    #[test]
    fn a_larger_real_frame_still_fits_the_work_area() {
        let area = work_area_4k();
        // As estimated: the planned window.
        assert_eq!(
            framed(&area, (16, 39), (1920, 1080)),
            ((952, 490, 1936, 1119), (1920, 1080))
        );
        // A client planned to fill the height under a 39-pixel frame, in a
        // real frame of 60: shrunk so frame and all stay inside, centred.
        let ((x, y, width, height), (cw, ch)) = framed(&area, (24, 60), (3664, 2061));
        assert!(cw + 24 <= 3840 && ch + 60 <= 2100, "{cw}x{ch}");
        assert!(ch >= 2039, "{ch}");
        assert_eq!((width, height), (cw + 24, ch + 60));
        assert_eq!((x, y), ((3840 - width) / 2, (2100 - height) / 2));
    }

    #[test]
    fn a_close_runs_the_hook_if_there_is_one() {
        use std::sync::atomic::{AtomicBool, Ordering};
        static CLOSED: AtomicBool = AtomicBool::new(false);
        fn closing() {
            CLOSED.store(true, Ordering::SeqCst);
        }
        assert!(!run_close(None));
        assert!(!CLOSED.load(Ordering::SeqCst));
        let hook: fn() = closing;
        assert!(run_close(Some(&hook)));
        assert!(CLOSED.load(Ordering::SeqCst));
    }

    #[test]
    fn user_closes_and_the_key_menu_are_caught() {
        let command = |c: u32| presenting_message(wm::WM_SYSCOMMAND, c as usize);
        // Close button, Alt+F4, window and taskbar menus; the low four bits
        // are the system's.
        assert_eq!(command(wm::SC_CLOSE), Handling::Close);
        assert_eq!(command(wm::SC_CLOSE | 2), Handling::Close);
        assert_eq!(presenting_message(wm::WM_CLOSE, 0), Handling::Close);
        // Alt, F10 and Alt+Space.
        assert_eq!(command(wm::SC_KEYMENU), Handling::Ignore);
        assert_eq!(
            presenting_message(wm::WM_SYSCOMMAND, wm::SC_KEYMENU as usize | 1),
            Handling::Ignore
        );
        for other in [wm::SC_MINIMIZE, wm::SC_MOVE, wm::SC_MOUSEMENU] {
            assert_eq!(command(other), Handling::Default);
        }
        assert_eq!(presenting_message(wm::WM_PAINT, 0), Handling::Default);
        // The command's value under another message is no command.
        assert_eq!(
            presenting_message(wm::WM_KEYDOWN, wm::SC_CLOSE as usize),
            Handling::Default
        );
    }
}
