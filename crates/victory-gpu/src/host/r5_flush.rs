//! R5 flush (`VICTORY_R5_FLUSH=1`): less work on the submitting thread
//! before a submission's lists go to the card, and counters that say where
//! the rest of it goes. See [`on`].

use super::*;

/// R5 flush (`VICTORY_R5_FLUSH=1`, off by default). Read once.
///
/// - The submission's one write-watch asking (`crate::watch::ask_once`) and
///   a flush's asking ahead (`crate::watch::ask_ahead`) leave out the spans
///   whose first block the guards already hold as view memory: the kernel
///   cannot answer for view memory, so every run of joined spans holding
///   one failed and was asked again by halves, and the span alone once
///   more. `crate::watch::writes` answers those copies from the guards, as
///   it did after those failed askings.
/// - A refresh of a placed copy (a home of the memory core, or a buffer in
///   place) records no step: its refresh copies nothing, and the step was
///   only there to hold it until the list is reset. The list holds it in
///   [`Kept`] instead (once per copy), so nothing at submission walks it. A
///   card copy whose CPU side is a home is not placed and keeps its step.
/// - A line every ten seconds with the frame-rate line says, per frame:
///   failed kernel askings and their bytes, the guards' `VirtualProtect`
///   calls, their microseconds and the blocks guarded again, the guards'
///   faults, the spans left out of the askings, the flushes' refreshes of
///   placed copies, of view copies and of card copies with a home, the
///   distinct 64 KiB blocks the flushes copied against the bytes they
///   copied, the refresh steps of placed copies and the copies kept
///   without one, and the time of the texture checks.
pub(crate) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_ON.with(std::cell::Cell::get) {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_R5_FLUSH").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    static TEST_ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Tests: [`on`] as asked for this thread while it lives, then as the
/// process says again.
#[cfg(test)]
pub(crate) struct TestSwitch;

#[cfg(test)]
impl TestSwitch {
    pub(crate) fn set(on: bool) -> TestSwitch {
        TEST_ON.with(|t| t.set(Some(on)));
        TestSwitch
    }
}

#[cfg(test)]
impl Drop for TestSwitch {
    fn drop(&mut self) {
        TEST_ON.with(|t| t.set(None));
    }
}

/// What [`STATS`] counts, since the last line.
#[derive(Clone, Copy)]
pub(crate) enum Count {
    /// Kernel write-watch askings that failed (the range is not
    /// write-watched), and their bytes.
    FailedAsks,
    FailedBytes,
    /// The guards' `VirtualProtect` calls that make view blocks read-only,
    /// their microseconds (with the switch on), and the blocks so guarded.
    Protects,
    ProtectUs,
    Guarded,
    /// Writes into guarded blocks taken by the guards' fault handler.
    Faults,
    /// Spans left out of the askings as known view memory, and their bytes.
    ViewSpans,
    ViewSpanBytes,
    /// A flush's refreshes of placed copies, of copies of view memory, and
    /// of card copies whose CPU side is a home.
    PlacedRefreshes,
    ViewRefreshes,
    CardHomeRefreshes,
    /// The distinct 64 KiB blocks a flush copied, summed over flushes, and
    /// the bytes it copied (a block taken by two copies counts twice here).
    Blocks,
    BlockBytes,
    /// Refresh steps of placed copies run, and placed copies a list held
    /// with no step.
    PlacedSteps,
    KeptHomes,
    /// Microseconds of the texture checks of the lists' steps.
    TextureUs,
}

const COUNTS: usize = Count::TextureUs as usize + 1;

static STATS: [std::sync::atomic::AtomicU64; COUNTS] =
    [const { std::sync::atomic::AtomicU64::new(0) }; COUNTS];

/// Adds `n` to `c`. A relaxed addition only, whatever the switch: safe in
/// the guards' fault handler.
pub(crate) fn count(c: Count, n: u64) {
    STATS[c as usize].fetch_add(n, Ordering::Relaxed);
}

/// Tests: `c` now.
#[cfg(test)]
pub(crate) fn counted(c: Count) -> u64 {
    STATS[c as usize].load(Ordering::Relaxed)
}

/// The placed copies a list holds with no refresh step ([`on`]), until it
/// is reset: each once.
#[derive(Default)]
pub(super) struct Kept {
    homes: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    by: std::collections::HashSet<usize>,
}

impl Kept {
    /// Holds `shadow` until the list is reset.
    pub(super) fn keep(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        if self.by.insert(std::sync::Arc::as_ptr(shadow) as usize) {
            self.homes.push(shadow.clone());
            count(Count::KeptHomes, 1);
        }
    }

    /// Tests: the copies held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.homes.len()
    }
}

thread_local! {
    /// The blocks the flush running on this thread has copied so far
    /// ([`flush_begins`]); `None` outside a flush.
    static BLOCKS: std::cell::RefCell<Option<std::collections::HashSet<usize>>> =
        const { std::cell::RefCell::new(None) };
}

/// With [`on`]: a flush begins on this thread; [`copied`] notes its blocks
/// until [`flush_ends`]. The copies the flush hands to other threads
/// (`VICTORY_WIDE_REFRESH=1`) are not counted.
pub(super) fn flush_begins(refreshes: &[ShadowRefresh]) {
    for (shadow, ranges) in refreshes {
        if shadow.is_placed() {
            count(Count::PlacedRefreshes, 1);
        } else if shadow.card_home().is_some() {
            count(Count::CardHomeRefreshes, 1);
        } else if let Some(&(from, _)) = ranges.as_slice().first()
            && crate::watch::known_view_at(from)
        {
            count(Count::ViewRefreshes, 1);
        }
    }
    BLOCKS.with(|b| *b.borrow_mut() = Some(std::collections::HashSet::new()));
}

/// The flush begun by [`flush_begins`] ends: its distinct blocks counted.
pub(super) fn flush_ends() {
    if let Some(blocks) = BLOCKS.with(|b| b.borrow_mut().take()) {
        count(Count::Blocks, blocks.len() as u64);
    }
}

/// Title memory `[from, to)` was copied into a host copy
/// (`crate::shadow::Shadow`'s copy), for the flush running on this thread.
pub(crate) fn copied(from: usize, to: usize) {
    if !on() || from >= to {
        return;
    }
    BLOCKS.with(|b| {
        if let Some(blocks) = b.borrow_mut().as_mut() {
            blocks.extend(from / crate::watch::PAGE..to.div_ceil(crate::watch::PAGE));
            count(Count::BlockBytes, (to - from) as u64);
        }
    });
}

/// For the frame-rate line, with [`on`]: [`STATS`] per frame since the
/// last one.
pub(super) fn line(frames: f64) {
    if !on() {
        return;
    }
    let s: Vec<f64> = STATS
        .iter()
        .map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames)
        .collect();
    let mb = |b: f64| b / (1 << 20) as f64;
    tracing::info!(
        "R5 flush per frame (VICTORY_R5_FLUSH=1): {:.1} failed kernel askings ({:.1} MB); {:.1} spans left out of the askings \
         as view memory ({:.1} MB); guards: {:.1} VirtualProtect calls ({:.2} ms) guarding {:.0} blocks, {:.1} faults; flushes \
         refreshed {:.1} placed copies, {:.1} view copies, {:.1} card copies with a home, and copied {:.0} distinct 64 KiB blocks \
         ({:.1} MB) in {:.1} MB; {:.1} refresh steps of placed copies, {:.1} placed copies held with no step; texture checks \
         {:.2} ms",
        s[Count::FailedAsks as usize],
        mb(s[Count::FailedBytes as usize]),
        s[Count::ViewSpans as usize],
        mb(s[Count::ViewSpanBytes as usize]),
        s[Count::Protects as usize],
        s[Count::ProtectUs as usize] / 1000.0,
        s[Count::Guarded as usize],
        s[Count::Faults as usize],
        s[Count::PlacedRefreshes as usize],
        s[Count::ViewRefreshes as usize],
        s[Count::CardHomeRefreshes as usize],
        s[Count::Blocks as usize],
        mb(s[Count::Blocks as usize] * crate::watch::PAGE as f64),
        mb(s[Count::BlockBytes as usize]),
        s[Count::PlacedSteps as usize],
        s[Count::KeptHomes as usize],
        s[Count::TextureUs as usize] / 1000.0,
    );
}

/// Tests: the list's refresh steps, and the placed copies it holds with no
/// step ([`Kept`]).
#[cfg(test)]
impl HostList {
    pub(crate) fn r5_flush_held(&self) -> (usize, usize) {
        let deferred = self.deferred();
        let steps = deferred
            .copies
            .iter()
            .filter(|step| matches!(step, Step::Refresh(..)))
            .count();
        (steps, deferred.kept_homes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::watch::PAGE;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile, VirtualAlloc,
        VirtualFree,
    };

    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// B: a submission refreshes a copy of view memory the guards hold and
    /// a copy of write-watched memory. With the switch the view's span is
    /// not asked of the kernel (no failed asking), without it the asking
    /// fails; either way a write into each since the last submission is in
    /// its copy before the list runs.
    #[test]
    fn warp_r5_flush_view_spans_are_not_asked_of_the_kernel() {
        let Some(host) = host() else { return };
        const SIZE: usize = 2 * PAGE;
        // SAFETY: a fresh pagefile section mapped once (the view keeps the
        // section alive) and a fresh write-watched allocation, both the
        // test's own until the end.
        let (view, watched) = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize;
            CloseHandle(section);
            let watched = VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize;
            assert!(view != 0 && watched != 0);
            (view, watched)
        };
        let fill = |at: usize, len: usize, value: u8| {
            // SAFETY: the test's own memory (a guarded view block is opened by
            // the guards' fault handler).
            unsafe { (at as *mut u8).write_bytes(value, len) };
        };
        crate::shadow::register(0xc0de_f101, view as u64, SIZE as u64);
        crate::shadow::register(0xc0de_f102, watched as u64, SIZE as u64);
        let one = crate::shadow::find(view as u64, SIZE as u64).expect("a copy of the view");
        let two =
            crate::shadow::find(watched as u64, SIZE as u64).expect("a copy of the watched memory");
        assert!(!one.is_placed() && !two.is_placed());
        let submit = || {
            let list = host.new_list().unwrap();
            one.refresh(&list, view as u64, SIZE as u64);
            two.refresh(&list, watched as u64, SIZE as u64);
            list.close();
            let failed = counted(Count::FailedAsks);
            let left_out = counted(Count::ViewSpans);
            host.execute(&[&list]);
            list.reset();
            (
                counted(Count::FailedAsks) - failed,
                counted(Count::ViewSpans) - left_out,
            )
        };
        for (on, value) in [(true, 1u8), (true, 2), (false, 3), (true, 4)] {
            let _switch = TestSwitch::set(on);
            // After the first submission only the view's second block is
            // written: the first stays guarded, which is what the guards
            // know a view by (a written block is open until it is copied).
            if value == 1 {
                fill(view, SIZE, value);
            } else {
                fill(view + PAGE, PAGE, value);
            }
            fill(watched, SIZE, value.wrapping_mul(7));
            let (failed, left_out) = submit();
            let bytes = crate::shadow::copy_bytes(&one);
            assert!(
                bytes[..PAGE].iter().all(|&b| b == 1) && bytes[PAGE..].iter().all(|&b| b == value),
                "the view's write is in its copy ({value})"
            );
            assert!(
                crate::shadow::copy_bytes(&two)
                    .iter()
                    .all(|&b| b == value.wrapping_mul(7)),
                "the watched write is in its copy ({value})"
            );
            // The first submission takes the view in: only then do the
            // guards hold it.
            if value == 1 {
                continue;
            }
            if on {
                assert_eq!(failed, 0, "no failed kernel asking");
                assert!(left_out >= 1, "the view's span left out");
            } else if crate::watch::light() {
                // `VICTORY_LIGHT_REFRESH=1` leaves the span out by its own
                // bits (a block of it is read-only by our hand), from the
                // asking ahead and from `writes`: not asked either, though
                // not counted as left out by the R5 flush.
                assert_eq!(failed, 0, "with the light refresh the view's span is not asked");
                assert_eq!(left_out, 0);
            } else {
                assert!(
                    failed >= 1,
                    "without the switch the view's span is asked and fails"
                );
                assert_eq!(left_out, 0);
            }
        }
        crate::shadow::unregister(0xc0de_f101, view as u64);
        crate::shadow::unregister(0xc0de_f102, watched as u64);
        drop((one, two));
        crate::watch::title_memory_remapped(view, SIZE);
        // SAFETY: the test's own view and allocation, no list uses them.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            VirtualFree(watched as *mut _, 0, MEM_RELEASE);
        }
    }
}
