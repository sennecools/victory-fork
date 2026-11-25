//! The write-watch journal as epoch tables (`VICTORY_EPOCH_JOURNAL=1`; off
//! by default). What [`super::Journal`] keeps in one map behind one lock,
//! for every asking of every thread, is kept here as a number per 4 KiB
//! page, so that askings about different memory do not wait for each other
//! and the answers for a submission's asking ahead are read with no lock
//! but a read-mostly one. That lock was what kept `VICTORY_WIDE_REFRESH`
//! from scaling: every refresh on every thread took it.
//!
//! - A table holds, for each kernel page, the asking that last found it
//!   written (0: never), as an offset from a base that [`Epochs::rebase`]
//!   moves before the offsets run out. A table covers 1 GiB of addresses
//!   and is made by the first asking that finds a page there. Tables are
//!   never freed: the title frees its memory with the host's own call,
//!   which the runtime does not see, so a reservation freed and another
//!   made at its addresses only looks written once more.
//!   Beside its kernel pages a table holds, for each of our pages, the
//!   highest of its kernel pages', so an answer reads one entry a page.
//! - An asking holds the locks of the 16 MB spans it asks about (taken
//!   modulo their number, so spans 4 GiB apart share one) from
//!   before the kernel's answer until its pages are noted, and takes its
//!   number after the answer. So a page whose dirty bit one asking reset is
//!   noted before another asking about it begins, and a write the kernel
//!   reports after a copy took the page has a higher number than the
//!   copy's.
//! - The runs of a submission's asking ahead ([`Epochs::ask_ahead`]) are
//!   asked on the threads kept for the refreshes, each under its own
//!   locks. The asking that stands is a short list ([`Ahead`]); its number
//!   is the highest of its runs'.
//!
//! The answers are in 64 KiB pages, as the journal gives them (a page's
//! number is the highest of its kernel pages'), so host copies keep their
//! 64 KiB grain.

use super::{KERNEL_PAGE, PAGE, pieces_of, runs_of, shards};
use std::cell::RefCell;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock};

/// Kernel pages in one of ours.
const SUB: usize = PAGE / KERNEL_PAGE;
/// The addresses a table covers: 1 GiB.
const TABLE_BITS: u32 = 30;
/// Kernel pages in a table.
const TABLE_PAGES: usize = 1 << (TABLE_BITS - KERNEL_PAGE.trailing_zeros());
/// A table's entries: one per kernel page, then one per page of ours (the
/// highest of its kernel pages').
const TABLE_LEN: usize = TABLE_PAGES + TABLE_PAGES / SUB;
/// The most room for the kernel's answer a thread keeps between askings
/// (2 MiB: an asking ahead's longest run).
const FOUND_KEPT: usize = super::AHEAD_MAX * SUB;
/// The user address space of 64-bit Windows (128 TiB), which the tables
/// cover.
const SPACE: usize = 1 << 47;
/// How far the askings may run past the base before it is moved: half of
/// what an offset holds, so threads that took their number before the
/// move still fit.
const LIMIT: u64 = 1 << 31;
/// The most pieces of work one asking ahead is cut into.
const JOBS: usize = 8;

/// The tables, if `VICTORY_EPOCH_JOURNAL=1`.
pub(super) fn on() -> Option<&'static Epochs> {
    static ON: OnceLock<bool> = OnceLock::new();
    static EPOCHS: OnceLock<Epochs> = OnceLock::new();
    ON.get_or_init(|| std::env::var_os("VICTORY_EPOCH_JOURNAL").is_some_and(|v| v == "1"))
        .then(|| EPOCHS.get_or_init(|| Epochs::new(LIMIT, true)))
}

/// The asking that stands for a submission's refreshes ([`Epochs::ask_ahead`],
/// [`Epochs::ask_once`]).
#[derive(Default)]
struct Ahead {
    /// The page ranges asked, by first page.
    ranges: Vec<(usize, usize)>,
    /// The furthest end of `ranges[..=i]`.
    reach: Vec<usize>,
    /// Higher than the number of every asking of `ranges`.
    now: u64,
    /// The asking stands for the whole submission ([`super::ask_once`]).
    kept: bool,
}

impl Ahead {
    fn set(&mut self, mut ranges: Vec<(usize, usize)>, now: u64) {
        ranges.sort_unstable();
        self.reach = ranges
            .iter()
            .scan(0, |reach, r| {
                *reach = r.1.max(*reach);
                Some(*reach)
            })
            .collect();
        self.ranges = ranges;
        self.now = now;
    }

    fn clear(&mut self) {
        self.ranges.clear();
        self.reach.clear();
    }

    /// Whether pages `[first, last)` lie in one range asked.
    fn covers(&self, first: usize, last: usize) -> bool {
        let i = self.ranges.partition_point(|r| r.0 <= first);
        i > 0 && self.reach[i - 1] >= last
    }
}

/// Page ranges asked about and the highest number of their askings.
#[derive(Default)]
struct Asked {
    ranges: Vec<(usize, usize)>,
    now: u64,
}

impl Asked {
    fn add(&mut self, other: Self) {
        self.ranges.extend(other.ranges);
        self.now = self.now.max(other.now);
    }
}

/// The journal's state (see the module's text).
pub(super) struct Epochs {
    /// One per 1 GiB of addresses: its kernel pages' offsets
    /// ([`TABLE_PAGES`] of them) and its pages' ([`TABLE_LEN`] in all), or
    /// null before a page there was written.
    tables: Box<[AtomicPtr<AtomicU32>]>,
    /// The number of the last asking.
    epoch: AtomicU64,
    /// What an offset of 0 in the tables stands for.
    base: AtomicU64,
    /// Odd while the base moves; changed by every move.
    moves: AtomicU64,
    /// How far `epoch` may run past `base`.
    limit: u64,
    /// An asking's locks, one per 16 MB of addresses (taken modulo their
    /// number, ascending, as `super::MAPPING`'s).
    spans: Box<[Mutex<()>]>,
    ahead: RwLock<Ahead>,
    /// Whether runs of an asking ahead go to the threads kept for the
    /// refreshes.
    wide: bool,
}

// The asking's room for the kernel's answer.
thread_local! {
    static FOUND: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// `f` with this thread's room for the kernel's answer, cut back to
/// [`FOUND_KEPT`] after it, so the threads that ever asked about much
/// memory at once do not each keep room for it.
fn with_found<R>(f: impl FnOnce(&mut Vec<usize>) -> R) -> R {
    FOUND.with_borrow_mut(|found| {
        let answer = f(found);
        if found.len() > FOUND_KEPT {
            found.truncate(FOUND_KEPT);
            found.shrink_to(FOUND_KEPT);
        }
        answer
    })
}

impl Epochs {
    fn new(limit: u64, wide: bool) -> Self {
        Self {
            tables: (0..SPACE >> TABLE_BITS)
                .map(|_| AtomicPtr::default())
                .collect(),
            epoch: AtomicU64::new(0),
            base: AtomicU64::new(0),
            moves: AtomicU64::new(0),
            limit,
            spans: (0..super::SHARDS).map(|_| Mutex::new(())).collect(),
            ahead: RwLock::default(),
            wide,
        }
    }

    fn ahead(&self) -> std::sync::RwLockReadGuard<'_, Ahead> {
        self.ahead.read().unwrap_or_else(|e| e.into_inner())
    }

    fn ahead_mut(&self) -> std::sync::RwLockWriteGuard<'_, Ahead> {
        self.ahead.write().unwrap_or_else(|e| e.into_inner())
    }

    /// The table holding address `at`, if one was made.
    fn table(&self, at: usize) -> Option<&[AtomicU32]> {
        let table = self.tables.get(at >> TABLE_BITS)?.load(Ordering::Acquire);
        // SAFETY: a table is `TABLE_LEN` entries made by `table_or_make`
        // and freed only when `self` is dropped.
        (!table.is_null()).then(|| unsafe { std::slice::from_raw_parts(table, TABLE_LEN) })
    }

    /// [`Epochs::table`], made if there is none; `None` outside the address
    /// space.
    fn table_or_make(&self, at: usize) -> Option<&[AtomicU32]> {
        let slot = self.tables.get(at >> TABLE_BITS)?;
        if slot.load(Ordering::Acquire).is_null() {
            let made: Box<[AtomicU32]> = (0..TABLE_LEN).map(|_| AtomicU32::new(0)).collect();
            let made = Box::into_raw(made).cast::<AtomicU32>();
            if slot
                .compare_exchange(
                    std::ptr::null_mut(),
                    made,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                // SAFETY: ours, made above and not shared: another thread's
                // table went in first.
                drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(made, TABLE_LEN)) });
            }
        }
        self.table(at)
    }

    /// The locks of pages `[first, last)`, ascending.
    fn lock(&self, first: usize, last: usize) -> Vec<MutexGuard<'_, ()>> {
        let len = last.saturating_sub(first).saturating_mul(PAGE).max(1);
        shards(first.saturating_mul(PAGE), len)
            .into_iter()
            .map(|s| self.spans[s].lock().unwrap_or_else(|e| e.into_inner()))
            .collect()
    }

    /// A new asking's number. Taken with the locks of what it notes held.
    fn next(&self) -> u64 {
        self.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Notes the kernel pages of `[from, to)` written by asking `now`, with
    /// their locks held.
    fn mark(&self, from: usize, to: usize, now: u64) {
        // The base holds still while the locks are held (a move takes them
        // all), and the askings are less than the limit past it.
        let offset = u32::try_from(now - self.base.load(Ordering::Relaxed)).unwrap_or(u32::MAX);
        let mut at = from / KERNEL_PAGE * KERNEL_PAGE;
        while at < to.min(SPACE) {
            let Some(table) = self.table_or_make(at) else {
                return;
            };
            let end = to.min(((at >> TABLE_BITS) + 1) << TABLE_BITS);
            let first = (at / KERNEL_PAGE) % TABLE_PAGES;
            let last = first + end.div_ceil(KERNEL_PAGE) - at / KERNEL_PAGE;
            for entry in &table[first..last] {
                entry.store(offset, Ordering::Relaxed);
            }
            for entry in &table[TABLE_PAGES + first / SUB..TABLE_PAGES + last.div_ceil(SUB)] {
                entry.fetch_max(offset, Ordering::Relaxed);
            }
            at = (at / KERNEL_PAGE + last - first) * KERNEL_PAGE;
        }
    }

    /// The pages of `[first, last)` written since tracking began, with the
    /// asking that last found each written, in page order (as
    /// [`super::writes`] gives them).
    fn fold(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        loop {
            let moves = self.moves.load(Ordering::Acquire);
            if moves % 2 == 1 {
                std::thread::yield_now();
                continue;
            }
            let base = self.base.load(Ordering::Relaxed);
            let mut written = Vec::new();
            let mut page = first;
            while page < last && page < SPACE / PAGE {
                let at = page * PAGE;
                let next = ((at >> TABLE_BITS) + 1) << TABLE_BITS;
                let end = last.min(next / PAGE);
                if let Some(table) = self.table(at) {
                    for page in page..end {
                        let top =
                            table[TABLE_PAGES + page % (TABLE_PAGES / SUB)].load(Ordering::Relaxed);
                        if top != 0 {
                            written.push((page, base + u64::from(top)));
                        }
                    }
                }
                page = end;
            }
            fence(Ordering::Acquire);
            if self.moves.load(Ordering::Relaxed) == moves {
                return written;
            }
        }
    }

    /// Moves the base if the askings ran far enough past it. Called with no
    /// lock of an asking held.
    fn rebase_if_due(&self) {
        let (now, base) = (
            self.epoch.load(Ordering::Relaxed),
            self.base.load(Ordering::Relaxed),
        );
        if now.saturating_sub(base) > self.limit {
            self.rebase();
        }
    }

    /// Moves the base to just below the last asking, with every page
    /// written so far noted as written by that asking: later than it was
    /// for some, so a copy that took such a page before that asking takes
    /// it once more, and no copy misses a write.
    fn rebase(&self) {
        let _all: Vec<_> = self
            .spans
            .iter()
            .map(|s| s.lock().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let now = self.epoch.load(Ordering::SeqCst);
        if now - self.base.load(Ordering::Relaxed) <= self.limit {
            // Another thread moved it.
            return;
        }
        let moves = self.moves.load(Ordering::Relaxed);
        self.moves.store(moves + 1, Ordering::Relaxed);
        fence(Ordering::Release);
        for slot in &self.tables {
            if slot.load(Ordering::Acquire).is_null() {
                continue;
            }
            // SAFETY: as in `table`.
            let table =
                unsafe { std::slice::from_raw_parts(slot.load(Ordering::Acquire), TABLE_LEN) };
            for entry in table {
                if entry.load(Ordering::Relaxed) != 0 {
                    entry.store(1, Ordering::Relaxed);
                }
            }
        }
        self.base.store(now - 1, Ordering::Relaxed);
        self.moves.store(moves + 2, Ordering::Release);
    }

    /// One asking of the kernel about pages `[first, last)`, noted: its
    /// number, or `None` (nothing noted) if the range is not write-watched
    /// memory of one reservation.
    fn ask_kernel(&self, first: usize, last: usize) -> Option<u64> {
        self.rebase_if_due();
        let _held = self.lock(first, last);
        with_found(|found| {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            let count = kernel_ask(first, last, found)?;
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, started.elapsed().as_micros() as u64);
                super::slow(started.elapsed().as_micros() as u64);
            }
            let now = self.next();
            for &at in &found[..count] {
                self.mark(at, at + 1, now);
            }
            Some(now)
        })
    }

    /// [`super::ask_run`]: one asking for `pieces` (page ranges in order),
    /// by halves if it fails; a piece alone is left to `writes`.
    fn ask_run(&self, pieces: &[(usize, usize)], asked: &mut Asked) {
        if pieces.len() < 2 {
            return;
        }
        let first = pieces[0].0;
        let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
        if let Some(now) = self.ask_kernel(first, last) {
            asked.ranges.push((first, last));
            asked.now = asked.now.max(now);
            return;
        }
        let (a, b) = pieces.split_at(pieces.len() / 2);
        self.ask_run(a, asked);
        self.ask_run(b, asked);
    }

    /// `ask` for each of `items`, on the threads kept for the refreshes if
    /// there are several.
    fn spread<T: Send + Sync + 'static>(
        &'static self,
        items: Vec<T>,
        ask: fn(&Self, &T, &mut Asked),
    ) -> Asked {
        let mut all = Asked::default();
        if !self.wide || items.len() < 2 {
            for item in &items {
                ask(self, item, &mut all);
            }
            return all;
        }
        let per = items.len().div_ceil(JOBS);
        let items = Arc::new(items);
        let out = Arc::new(Mutex::new(Asked::default()));
        // On the profile: askings for a refresh, and the journal's share of
        // them, count as such on the threads that make them.
        let refresh = crate::profile::in_refresh();
        let jobs: Vec<Box<dyn FnOnce() + Send>> = (0..items.len())
            .step_by(per)
            .map(|from| {
                let (items, out) = (items.clone(), out.clone());
                Box::new(move || {
                    let _refresh = refresh.then(crate::profile::refresh_scope);
                    let _journal = crate::profile::journal_timer();
                    let mut mine = Asked::default();
                    for item in &items[from..(from + per).min(items.len())] {
                        ask(self, item, &mut mine);
                    }
                    out.lock().unwrap_or_else(|e| e.into_inner()).add(mine);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        // (The wait for them is not the journal's share.)
        crate::profile::handed_off(|| crate::shadow::wide_run(jobs));
        all.add(std::mem::take(
            &mut *out.lock().unwrap_or_else(|e| e.into_inner()),
        ));
        all
    }

    /// The runs of `pieces` (page ranges in order), each asked as one.
    fn ask_runs(&'static self, pieces: &[(usize, usize)]) -> Asked {
        let runs: Vec<Vec<(usize, usize)>> = runs_of(pieces)
            .into_iter()
            .map(|run| pieces[run].to_vec())
            .collect();
        self.spread(runs, |epochs, run, asked| epochs.ask_run(run, asked))
    }

    /// [`super::ask_ahead`].
    pub(super) fn ask_ahead(&'static self, ranges: &[(usize, usize)]) {
        let pieces = pieces_of(ranges);
        if pieces.len() < 2 {
            return;
        }
        // On the profile: the journal's share of a refresh (its locks, its
        // tables), less the kernel askings made under it.
        let _journal = crate::profile::journal_timer();
        if self.ahead().kept {
            return;
        }
        let asked = self.ask_runs(&pieces);
        let mut ahead = self.ahead_mut();
        // An asking for the whole submission went in meanwhile: it stands.
        if !ahead.kept {
            let now = self.standing(asked.now);
            ahead.set(asked.ranges, now);
        }
    }

    /// The number of an asking ahead whose runs' highest was `asked`: the
    /// last asking's, which is no lower. The journal's number comes before
    /// any other thread's asking about the pages meanwhile (its lock is
    /// held throughout); with a lower one, pages those askings noted would
    /// look written after it and be copied at every refresh while it
    /// stands. A higher number is as safe: it is taken before the
    /// refreshes read title memory, and an asking that finds a write made
    /// after that read takes its number later still.
    fn standing(&self, asked: u64) -> u64 {
        asked.max(self.epoch.load(Ordering::SeqCst))
    }

    /// [`super::ask_ahead_done`].
    pub(super) fn ask_ahead_done(&self) {
        let mut ahead = self.ahead_mut();
        if !ahead.kept {
            ahead.clear();
        }
    }

    /// [`super::ask_once`]: the ranges close together asked as by
    /// [`Epochs::ask_ahead`], then those that lie alone, one asking each.
    pub(super) fn ask_once(&'static self, ranges: &[(usize, usize)]) {
        let _journal = crate::profile::journal_timer();
        self.ask_once_end();
        let pieces = pieces_of(ranges);
        let mut asked = if pieces.len() < 2 {
            Asked::default()
        } else {
            self.ask_runs(&pieces)
        };
        let mut held = Ahead::default();
        held.set(asked.ranges.clone(), 0);
        let mut lone: Vec<(usize, usize)> = pieces
            .into_iter()
            .filter(|&(a, b)| !held.covers(a, b))
            .collect();
        // Those inside one before them (in page order) once.
        let mut reach = 0;
        lone.retain(|&(_, last)| {
            let inside = last <= reach;
            reach = reach.max(last);
            !inside
        });
        asked.add(self.spread(lone, |epochs, &(first, last), asked| {
            if let Some(now) = epochs.ask_kernel(first, last) {
                asked.ranges.push((first, last));
                asked.now = asked.now.max(now);
            }
        }));
        let mut ahead = self.ahead_mut();
        let now = self.standing(asked.now);
        ahead.set(asked.ranges, now);
        ahead.kept = !ahead.ranges.is_empty();
    }

    /// [`super::ask_once_end`].
    pub(super) fn ask_once_end(&self) {
        let mut ahead = self.ahead_mut();
        if ahead.kept {
            ahead.kept = false;
            ahead.clear();
        }
    }

    /// [`super::wrote`]: the pages count as written by a new asking, which
    /// is the one that stands from now on.
    pub(super) fn wrote(&self, at: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut ahead = self.ahead_mut();
        if !ahead.kept {
            return;
        }
        self.rebase_if_due();
        let end = at.saturating_add(len);
        let _held = self.lock(at / PAGE, end.div_ceil(PAGE));
        let now = self.next();
        self.mark(at, end, now);
        ahead.now = now;
    }

    /// [`super::gpu_wrote`].
    pub(super) fn gpu_wrote(&self, at: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut ahead = self.ahead_mut();
        self.rebase_if_due();
        let end = at.saturating_add(len);
        let _held = self.lock(at / PAGE, end.div_ceil(PAGE));
        let now = self.next();
        self.mark(at, end, now);
        if !ahead.ranges.is_empty() {
            ahead.now = now;
        }
    }

    /// [`super::asked`].
    pub(super) fn asked(&self, start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        let ahead = self.ahead();
        ahead
            .covers(first, last)
            .then(|| (ahead.now, self.fold(first, last)))
    }

    /// [`super::writes`].
    pub(super) fn writes(
        &self,
        start: usize,
        end: usize,
        guarded: bool,
    ) -> Option<(u64, Vec<(usize, u64)>)> {
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        let _journal = crate::profile::journal_timer();
        if let Some(answer) = self.asked(start, end) {
            return Some(answer);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // As the journal: view memory the guards hold is not asked about.
        let known_view = guarded && super::known_view(first * PAGE);
        self.rebase_if_due();
        let _held = self.lock(first, last);
        with_found(|found| {
            let count = if known_view {
                None
            } else {
                kernel_ask(first, last, found)
            };
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(
                if known_view {
                    crate::copystat::C::WatchKnownView
                } else {
                    crate::copystat::C::WatchFailed
                },
                count.is_none() as u64,
            );
            let (count, views) = match count {
                Some(count) => (count, Vec::new()),
                None if !guarded => return None,
                None => {
                    #[cfg(feature = "diagnostics")]
                    let guarding = std::time::Instant::now();
                    let views = super::guard(first * PAGE, last * PAGE);
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(
                        crate::copystat::C::GuardUs,
                        guarding.elapsed().as_micros() as u64,
                    );
                    (0, views?)
                }
            };
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::Watches, 1);
                add(C::WatchUs, started.elapsed().as_micros() as u64);
                super::slow(started.elapsed().as_micros() as u64);
            }
            let now = self.next();
            for &at in &found[..count] {
                self.mark(at, at + 1, now);
            }
            for page in views {
                self.mark(page * PAGE, (page + 1) * PAGE, now);
            }
            Some((now, self.fold(first, last)))
        })
    }
}

impl Drop for Epochs {
    fn drop(&mut self) {
        for slot in &self.tables {
            let table = slot.load(Ordering::Acquire);
            if !table.is_null() {
                // SAFETY: made by `table_or_make`, no longer referred to.
                drop(unsafe {
                    Box::from_raw(std::ptr::slice_from_raw_parts_mut(table, TABLE_LEN))
                });
            }
        }
    }
}

/// Asks the kernel for the written kernel pages of pages `[first, last)`
/// and resets them: their addresses are `found[..count]`. `None` if the
/// range is not write-watched memory of one reservation.
fn kernel_ask(first: usize, last: usize, found: &mut Vec<usize>) -> Option<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    // Kept as long as it grew: a shorter answer reuses the room.
    found.resize(count.max(found.len()), 0);
    let mut granularity = 0;
    let asking = crate::profile::refresh_stamp();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    // (A failed asking found nothing: its time is the kernel's still.)
    let asked = if failed { &[][..] } else { &found[..count] };
    crate::profile::kernel_asked(asking, ((last - first) * PAGE) as u64, asked);
    (!failed).then_some(count)
}

#[cfg(test)]
mod tests {
    use super::super::{
        Journal, ask_ahead_done_in, ask_ahead_in, ask_once_end_in, ask_once_in, asked_in,
        writes_in, wrote_in,
    };
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Write-watched memory of ours, committed, freed when dropped.
    struct Region(usize, usize);

    impl Region {
        /// Its pages are all there and none counts as written, as in title
        /// memory the title has used: the kernel counts a page's first
        /// touch as a write, even a read's.
        fn new(len: usize) -> Self {
            use windows_sys::Win32::System::Memory::ResetWriteWatch;
            // SAFETY: a fresh write-watched allocation, freed on drop.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            } as usize;
            assert_ne!(at, 0);
            for page in (at..at + len).step_by(KERNEL_PAGE) {
                // SAFETY: inside the allocation, committed read-write.
                unsafe { (page as *mut u8).write_volatile(0) };
            }
            // SAFETY: our allocation.
            assert_eq!(unsafe { ResetWriteWatch(at as *const _, len) }, 0);
            Self(at, len)
        }

        fn write(&self, offset: usize, value: u8) {
            assert!(offset < self.1);
            // SAFETY: inside our committed read-write allocation.
            unsafe { ((self.0 + offset) as *mut u8).write_volatile(value) };
        }
    }

    impl Drop for Region {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// Tables of a test's own, which live as long as the test process (the
    /// threads kept for the refreshes ask with them).
    fn epochs(limit: u64) -> &'static Epochs {
        Box::leak(Box::new(Epochs::new(limit, true)))
    }

    /// A host copy of `[start, start + len)` as `crate::shadow::Shadow`
    /// keeps one: the bytes and, per 64 KiB page, the asking after which it
    /// was last taken (0: never).
    struct Copy {
        start: usize,
        len: usize,
        data: Vec<u8>,
        copied: Vec<u64>,
    }

    impl Copy {
        fn new(start: usize, len: usize) -> Self {
            let pages = (start + len).div_ceil(PAGE) - start / PAGE;
            Self {
                start,
                len,
                data: vec![0; len],
                copied: vec![0; pages],
            }
        }

        /// `Shadow::copy_stale`: copies the pages of `[from, to)` (inside
        /// the copy) written after their last copy, notes the whole ones
        /// taken at `now`; the pages copied, from the copy's first.
        fn refresh(
            &mut self,
            from: usize,
            to: usize,
            now: u64,
            written: &[(usize, u64)],
        ) -> Vec<usize> {
            let (start, end) = (self.start, self.start + self.len);
            let base = start / PAGE;
            let mut writes = written.iter().peekable();
            let mut taken = Vec::new();
            for page in from / PAGE..=(to - 1) / PAGE {
                let write = match writes.peek() {
                    Some(&&(p, w)) if p == page => {
                        writes.next();
                        w
                    }
                    _ => 0,
                };
                let at = &mut self.copied[page - base];
                if *at == 0 || write > *at {
                    if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                        *at = now;
                    }
                    let (lo, hi) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
                    // SAFETY: inside the caller's committed region.
                    let title = unsafe { std::slice::from_raw_parts(lo as *const u8, hi - lo) };
                    self.data[lo - start..hi - start].copy_from_slice(title);
                    taken.push(page - base);
                }
            }
            taken
        }

        /// Whether the copy holds the title's bytes of `[from, to)`.
        fn holds(&self, from: usize, to: usize) -> bool {
            // SAFETY: inside the caller's committed region.
            let title = unsafe { std::slice::from_raw_parts(from as *const u8, to - from) };
            self.data[from - self.start..to - self.start] == *title
        }
    }

    /// The written pages of an answer, from `base`.
    fn pages(answer: &[(usize, u64)], base: usize) -> Vec<usize> {
        answer.iter().map(|&(page, _)| page - base / PAGE).collect()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self, below: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % below as u64) as usize
        }
    }

    /// An asking notes the kernel pages it found, and a 64 KiB page is
    /// reported with the asking that found one of them.
    #[test]
    fn an_asking_marks_the_kernel_pages_it_found() {
        let e = epochs(LIMIT);
        let r = Region::new(4 * PAGE);
        let (_, before) = e.writes(r.0, r.0 + 4 * PAGE, false).expect("watched");
        assert!(before.is_empty());
        r.write(PAGE + 5 * KERNEL_PAGE + 8, 1);
        let (now, after) = e.writes(r.0, r.0 + 4 * PAGE, false).expect("watched");
        assert_eq!(after, [(r.0 / PAGE + 1, now)]);
        let table = e.table(r.0).expect("made by the asking");
        let marked: Vec<usize> = (0..4 * SUB)
            .filter(|&k| table[(r.0 / KERNEL_PAGE + k) % TABLE_PAGES].load(Ordering::Relaxed) != 0)
            .collect();
        assert_eq!(marked, [SUB + 5]);
        // Another asker after the kernel's bit was reset still sees it.
        let (later, again) = e
            .writes(r.0 + PAGE, r.0 + 2 * PAGE, false)
            .expect("watched");
        assert!(later > now);
        assert_eq!(again, [(r.0 / PAGE + 1, now)]);
        let plain = vec![0u8; PAGE];
        assert!(
            e.writes(
                plain.as_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                false
            )
            .is_none()
        );
    }

    /// A page's entry is the highest of its kernel pages', through marks
    /// that cross a table's end and through a move of the base.
    #[test]
    fn a_page_holds_its_kernel_pages_highest() {
        let e = epochs(4);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // The last 4 MiB of one table and the first of the next.
        let from = (3 << TABLE_BITS) - (4 << 20);
        let len = 8 << 20;
        let check = |e: &Epochs| {
            for at in (from..from + len).step_by(PAGE) {
                let table = e.table(at).expect("made");
                let i = (at / KERNEL_PAGE) % TABLE_PAGES;
                let top = table[i..i + SUB]
                    .iter()
                    .map(|kernel| kernel.load(Ordering::Relaxed))
                    .max();
                let page = table[TABLE_PAGES + i / SUB].load(Ordering::Relaxed);
                assert_eq!(Some(page), top, "page at {at:#x}");
            }
        };
        e.mark(from, from + len, e.next());
        for _ in 0..200 {
            let at = from + rng.next(len);
            let end = (at + rng.next(3 * PAGE)).min(from + len);
            let _held = e.lock(at / PAGE, end.div_ceil(PAGE));
            e.mark(at, end.max(at + 1), e.next());
        }
        check(e);
        e.rebase_if_due();
        assert!(e.base.load(Ordering::Relaxed) > 0, "moved");
        check(e);
    }

    /// Only the pages written after a copy took them are stale for it, and
    /// a page the refresh took only in part stays stale for the rest.
    #[test]
    fn stale_picks_and_a_partial_page_stays_stale() {
        let e = epochs(LIMIT);
        let r = Region::new(3 * PAGE);
        let mut copy = Copy::new(r.0, 3 * PAGE);
        let (now, written) = e.writes(r.0, r.0 + 3 * PAGE, false).expect("watched");
        assert_eq!(
            copy.refresh(r.0, r.0 + 3 * PAGE, now, &written),
            [0, 1, 2],
            "never taken"
        );
        let (now, written) = e.writes(r.0, r.0 + 3 * PAGE, false).expect("watched");
        assert!(copy.refresh(r.0, r.0 + 3 * PAGE, now, &written).is_empty());
        r.write(PAGE + 10, 7);
        r.write(2 * PAGE - 10, 9);
        // The first half of page 1 only: copied, the page still stale.
        let half = r.0 + PAGE + PAGE / 2;
        let (now, written) = e.writes(r.0, half, false).expect("watched");
        assert_eq!(copy.refresh(r.0, half, now, &written), [1]);
        assert!(copy.holds(r.0, half));
        let (now, written) = e.writes(half, r.0 + 3 * PAGE, false).expect("watched");
        assert_eq!(
            copy.refresh(half, r.0 + 3 * PAGE, now, &written),
            [1],
            "the rest of page 1"
        );
        assert!(copy.holds(r.0, r.0 + 3 * PAGE));
        // Taken in two parts, never whole: stale until a refresh takes it whole.
        let (now, written) = e.writes(r.0, r.0 + 3 * PAGE, false).expect("watched");
        assert_eq!(copy.refresh(r.0, r.0 + 3 * PAGE, now, &written), [1]);
        let (now, written) = e.writes(r.0, r.0 + 3 * PAGE, false).expect("watched");
        assert!(copy.refresh(r.0, r.0 + 3 * PAGE, now, &written).is_empty());
    }

    /// The numbers are moved down before their offsets run out: pages
    /// written before stay written (copied once more at most), pages never
    /// written stay unreported, and no copy misses a write.
    #[test]
    fn epochs_are_rebased_before_they_wrap() {
        let e = epochs(4);
        let r = Region::new(16 * PAGE);
        let mut copy = Copy::new(r.0, 15 * PAGE);
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..64 {
            // Never page 15, outside the copy and never written.
            r.write(rng.next(15 * PAGE), round as u8);
            let (now, written) = e.writes(r.0, r.0 + 15 * PAGE, false).expect("watched");
            copy.refresh(r.0, r.0 + 15 * PAGE, now, &written);
            assert!(copy.holds(r.0, r.0 + 15 * PAGE), "round {round}");
            assert!(e.epoch.load(Ordering::Relaxed) - e.base.load(Ordering::Relaxed) <= 5);
        }
        assert!(e.base.load(Ordering::Relaxed) > 0, "moved");
        let (_, written) = e.writes(r.0, r.0 + 16 * PAGE, false).expect("watched");
        assert!(!pages(&written, r.0).contains(&15));
    }

    /// Our own write while a submission's one asking stands counts as
    /// written after it, and the asking that stands moves with it.
    #[test]
    fn a_write_of_ours_moves_the_asking_that_stands() {
        let e = epochs(LIMIT);
        let r = Region::new(2 * PAGE);
        e.ask_once(&[(r.0, r.0 + PAGE)]);
        let (before, _) = e.asked(r.0, r.0 + PAGE).expect("asked");
        e.wrote(r.0 + 8, 4);
        let (now, written) = e.writes(r.0, r.0 + PAGE, false).expect("watched");
        assert!(now > before);
        assert_eq!(written, [(r.0 / PAGE, now)]);
        assert_eq!(e.asked(r.0, r.0 + PAGE), Some((now, written)));
        // Not asked about: the kernel is.
        assert!(e.asked(r.0 + PAGE, r.0 + 2 * PAGE).is_none());
        e.ask_once_end();
        assert!(e.asked(r.0, r.0 + PAGE).is_none());
    }

    /// The same random writes, askings and refreshes on two regions, one
    /// kept by the journal and one by the epoch tables: every refresh
    /// copies the same pages and leaves the copy holding the title's bytes.
    #[test]
    fn epoch_refresh_copies_the_same_bytes_as_the_journal() {
        const SIZE: usize = 40 << 20;
        let journal = Mutex::new(Journal::new());
        let e = epochs(LIMIT);
        let (a, b) = (Region::new(SIZE), Region::new(SIZE));
        // Nested and overlapping copies, some not on page bounds, in three
        // groups far enough apart to be asked about apart.
        let shapes: [(usize, usize); 9] = [
            (0, 4 << 20),
            (PAGE + 100, 5 * PAGE),
            (2 * PAGE, PAGE / 2),
            (3 * PAGE - 8, 2 * PAGE + 16),
            (20 << 20, 4 << 20),
            ((21 << 20) + KERNEL_PAGE, 3 * PAGE + 8),
            ((22 << 20) + 7, PAGE),
            (36 << 20, 4 << 20),
            ((37 << 20) + PAGE / 3, 9 * PAGE),
        ];
        let mut copies: Vec<(Copy, Copy)> = shapes
            .iter()
            .map(|&(at, len)| (Copy::new(a.0 + at, len), Copy::new(b.0 + at, len)))
            .collect();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut standing = false;
        let (mut stood, mut from_stand) = (0, 0);
        for step in 0..3000 {
            match rng.next(10) {
                // The title writes (no asking may stand: it does not tell).
                0..=2 => {
                    if standing {
                        ask_once_end_in(&journal);
                        e.ask_once_end();
                        standing = false;
                    }
                    for _ in 0..1 + rng.next(8) {
                        let (at, len) = shapes[rng.next(shapes.len())];
                        let offset = at + rng.next(len);
                        let value = rng.next(256) as u8;
                        a.write(offset, value);
                        b.write(offset, value);
                    }
                }
                // A refresh of part of a copy.
                3..=5 => {
                    let i = rng.next(copies.len());
                    let (at, len) = shapes[i];
                    let from = at + rng.next(len);
                    let to = from + 1 + rng.next(at + len - from);
                    let ja = asked_in(&journal, a.0 + from, a.0 + to);
                    let eb = e.asked(b.0 + from, b.0 + to);
                    assert_eq!(ja.is_some(), eb.is_some(), "step {step}: asked");
                    from_stand += usize::from(ja.is_some());
                    let (jn, jw) =
                        writes_in(&journal, a.0 + from, a.0 + to, false).expect("watched");
                    let (en, ew) = e.writes(b.0 + from, b.0 + to, false).expect("watched");
                    assert_eq!(
                        pages(&jw, a.0),
                        pages(&ew, b.0),
                        "step {step}: written pages"
                    );
                    let (ca, cb) = &mut copies[i];
                    let ta = ca.refresh(a.0 + from, a.0 + to, jn, &jw);
                    let tb = cb.refresh(b.0 + from, b.0 + to, en, &ew);
                    assert_eq!(ta, tb, "step {step}: pages copied");
                    assert!(ca.holds(a.0 + from, a.0 + to), "step {step}: journal copy");
                    assert!(cb.holds(b.0 + from, b.0 + to), "step {step}: epoch copy");
                }
                // A submission's one asking, over some of the copies.
                6 => {
                    let picked: Vec<(usize, usize)> = shapes
                        .iter()
                        .filter(|_| rng.next(3) != 0)
                        .map(|&(at, len)| (at, at + len))
                        .collect();
                    let ra: Vec<_> = picked.iter().map(|&(f, t)| (a.0 + f, a.0 + t)).collect();
                    let rb: Vec<_> = picked.iter().map(|&(f, t)| (b.0 + f, b.0 + t)).collect();
                    ask_once_in(&journal, &ra);
                    e.ask_once(&rb);
                    standing = journal.lock().unwrap().kept;
                    assert_eq!(standing, e.ahead().kept, "step {step}: stands");
                    stood += usize::from(standing);
                }
                // Our own write while it stands, told.
                7 => {
                    let (at, len) = shapes[rng.next(shapes.len())];
                    let offset = at + rng.next(len);
                    let n = 1 + rng.next((at + len - offset).min(3 * PAGE));
                    let value = rng.next(256) as u8;
                    for k in offset..offset + n {
                        a.write(k, value);
                        b.write(k, value);
                    }
                    wrote_in(&journal, a.0 + offset, n);
                    e.wrote(b.0 + offset, n);
                }
                // An asking ahead for a flush, then its refreshes.
                8 => {
                    let ra: Vec<_> = shapes
                        .iter()
                        .map(|&(at, len)| (a.0 + at, a.0 + at + len))
                        .collect();
                    let rb: Vec<_> = shapes
                        .iter()
                        .map(|&(at, len)| (b.0 + at, b.0 + at + len))
                        .collect();
                    ask_ahead_in(&journal, &ra);
                    e.ask_ahead(&rb);
                    for (i, &(at, len)) in shapes.iter().enumerate() {
                        let (jn, jw) =
                            writes_in(&journal, a.0 + at, a.0 + at + len, false).expect("watched");
                        let (en, ew) = e.writes(b.0 + at, b.0 + at + len, false).expect("watched");
                        let (ca, cb) = &mut copies[i];
                        let ta = ca.refresh(a.0 + at, a.0 + at + len, jn, &jw);
                        let tb = cb.refresh(b.0 + at, b.0 + at + len, en, &ew);
                        assert_eq!(ta, tb, "step {step}: pages copied ahead");
                        assert!(
                            cb.holds(b.0 + at, b.0 + at + len),
                            "step {step}: epoch copy ahead"
                        );
                    }
                    ask_ahead_done_in(&journal);
                    e.ask_ahead_done();
                }
                _ => {
                    ask_once_end_in(&journal);
                    e.ask_once_end();
                    standing = false;
                }
            }
        }
        assert!(
            stood > 100 && from_stand > 100,
            "the asking that stands was used: {stood}, {from_stand}"
        );
    }

    /// Threads asking about overlapping copies while the title writes, one
    /// of them asking ahead: once the writes stop, every copy brought up to
    /// date holds the title's bytes.
    #[test]
    fn concurrent_askers_miss_no_write() {
        const SIZE: usize = 64 * PAGE;
        const THREADS: usize = 4;
        let e = epochs(LIMIT);
        let r = Arc::new(Region::new(SIZE));
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let still = Arc::new(std::sync::Barrier::new(THREADS + 1));
        let shapes: [(usize, usize); THREADS] = [
            (0, SIZE),
            (3 * PAGE + 5, 20 * PAGE),
            (16 * PAGE, 40 * PAGE + 3),
            (60 * PAGE - 1, 4 * PAGE + 1),
        ];
        let askers: Vec<_> = shapes
            .iter()
            .enumerate()
            .map(|(n, &(at, len))| {
                let (r, done, still) = (r.clone(), done.clone(), still.clone());
                std::thread::spawn(move || {
                    let (start, end) = (r.0 + at, r.0 + at + len);
                    let mut copy = Copy::new(start, len);
                    let all: Vec<(usize, usize)> = shapes
                        .iter()
                        .map(|&(a, l)| (r.0 + a, r.0 + a + l))
                        .collect();
                    while !done.load(Ordering::SeqCst) {
                        if n == 0 {
                            e.ask_ahead(&all);
                        }
                        let (now, written) = e.writes(start, end, false).expect("watched");
                        copy.refresh(start, end, now, &written);
                        if n == 0 {
                            e.ask_ahead_done();
                        }
                    }
                    still.wait();
                    let (now, written) = e.writes(start, end, false).expect("watched");
                    copy.refresh(start, end, now, &written);
                    assert!(copy.holds(start, end), "copy {n}");
                })
            })
            .collect();
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for round in 0..200_000 {
            r.write(rng.next(SIZE), round as u8);
        }
        done.store(true, Ordering::SeqCst);
        still.wait();
        for asker in askers {
            asker.join().unwrap();
        }
    }
}
