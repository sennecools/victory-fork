//! The journal's stamps: for each 64 KiB page of title memory, the number of
//! the last asking that found it written (0: never), and for each group of
//! [`GROUP`] pages a summary, so an answer costs a look per group of the
//! range and one per page of the groups written since the caller's stamp,
//! not one per page ever written.
//!
//! - A table covers 1 GiB of addresses: a stamp per page, then a floor and a
//!   top per group. It is made by the first mark there and never freed: the
//!   title frees its memory with the host's own call, which the runtime does
//!   not see, so a reservation freed and another made at its addresses only
//!   looks written once more. A table is 132 KiB, so the tables are bounded
//!   by the address space touched (some 13 MiB for 100 GiB).
//! - A mark over a whole group raises the group's floor, one store for its
//!   64 pages (the GPU's big writes, `super::gpu_wrote`); a mark over part of
//!   one raises the stamps of its pages. A page's stamp is the higher of its
//!   own and its group's floor ([`Stamps::since`]).
//! - The top of a group is the highest of its pages' stamps and its floor:
//!   a group whose top is not above a caller's stamp is skipped whole.
//! - Stamps only rise (`fetch_max`): a copy that took a page after asking
//!   `n` must take it again iff the page's stamp is above `n`, so no copy
//!   that overlaps another loses a write the other's asking found.
//!
//! - The parts of a page (its 4 KiB pieces, the kernel's pages): for a
//!   page last written in part, a stamp per part, in a map beside the
//!   tables ([`Stamps::parts`]). A part mark makes the page's entry if it
//!   has none, every part at the page's stamp before the mark (what was
//!   written before is not known by parts), and raises the parts marked; a
//!   whole mark at or above the page's stamp drops the entry, an older one
//!   raises every part. A group's floor counts for every part of its
//!   pages. A copy that took a page after asking `n` takes the parts whose
//!   stamp is above `n`, the whole page if it has no entry
//!   ([`Stamps::stale_words`]). Adapted from cool_fox's 3ead56e
//!   (`PAGE_PARTS`, `parts_since`), 673be0e (`Shard::note`) and 0d5e4c8
//!   (`PageHash`), author `local`, as carried in Belobug/VIctory-Belobug-AMD
//!   (read at cd8dfc06), GPL-3.0-or-later; there the entries sit beside a
//!   map of pages, here beside the tables. An entry is 128 bytes, and only
//!   a page whose newest write was in part has one: a whole write at or
//!   above every one of its parts' stamps drops it, of the page alone or of
//!   its whole group (the floor). The map is bounded by the pages last
//!   written in part.
//!
//! Marks are made with the journal's lock held (`super::Journal`), between
//! the kernel's answer and the asking's number being handed out. The stamp
//! or floor is stored before the top, the top with release order, and a
//! reader loads the top first with acquire order, so whoever sees a top sees
//! the stamps under it. The parts map has a lock of its own, taken under
//! the journal's.

use super::{KERNEL_PAGE, PAGE};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Pages in a group, the unit of the summary (4 MiB).
const GROUP: usize = 64;
/// The addresses a table covers: 1 GiB.
const TABLE_BITS: u32 = 30;
/// Pages in a table.
const TABLE_PAGES: usize = (1 << TABLE_BITS) / PAGE;
/// Groups in a table.
const GROUPS: usize = TABLE_PAGES / GROUP;
/// Where a table's floors begin, after a stamp per page.
const FLOORS: usize = TABLE_PAGES;
/// Where its tops begin, after a floor per group.
const TOPS: usize = FLOORS + GROUPS;
/// A table's entries: a stamp per page, then a floor per group, then a top
/// per group.
const TABLE_LEN: usize = TOPS + GROUPS;
/// The 4 KiB parts of a page, a bit each.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// Every part of a page.
pub(crate) const WHOLE: u16 = u16::MAX;
/// The user address space of 64-bit Windows (128 TiB), which the tables
/// cover; pages past it are never written.
const SPACE_PAGES: usize = (1 << 47) / PAGE;

/// The stamps of every page (see the module's text).
pub(super) struct Stamps {
    /// One per 1 GiB of addresses, null before a page there was marked;
    /// made at the first mark.
    tables: OnceLock<Box<[AtomicPtr<AtomicU64>]>>,
    /// Page number -> the stamp of each of its parts, for pages last
    /// written in part; a page that is not here was written whole, as far
    /// as anyone knows (see the module's text).
    parts: Mutex<PageParts>,
}

/// The parts map of [`Stamps`].
type PageParts = HashMap<usize, [u64; PARTS], BuildHasherDefault<PageHash>>;

/// A cheap hash of page numbers (adapted from cool_fox's 0d5e4c8,
/// `PageHash`).
#[derive(Default)]
struct PageHash(u64);

impl std::hash::Hasher for PageHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15)
            .rotate_left(26)
            .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// Notes page `page`, whose stamp was `before` (its group's floor
/// included), written in `parts` by asking `now`, in the parts map
/// (cool_fox's `Shard::note`, 673be0e).
fn note_parts(map: &mut PageParts, page: usize, before: u64, parts: u16, now: u64) {
    if parts == WHOLE {
        if now >= before {
            map.remove(&page);
        } else if let Some(all) = map.get_mut(&page) {
            all.iter_mut().for_each(|at| *at = (*at).max(now));
        }
        return;
    }
    let all = map.entry(page).or_insert([before; PARTS]);
    for (k, at) in all.iter_mut().enumerate() {
        if parts >> k & 1 != 0 {
            *at = (*at).max(now);
        }
    }
}

/// The parts of page `page` written by an asking after `taken`, with its
/// group's floor `floor` (cool_fox's `parts_since`, 3ead56e): the whole page
/// if the map does not know them.
fn parts_since(map: &PageParts, page: usize, floor: u64, taken: u64) -> u16 {
    let parts = map.get(&page).map_or(WHOLE, |all| {
        (0..PARTS)
            .filter(|&k| all[k].max(floor) > taken)
            .fold(0, |m, k| m | 1 << k)
    });
    // No part above `taken` means the caller's page is not stale; should one
    // ask anyway, it takes the whole page.
    if parts == 0 { WHOLE } else { parts }
}

/// The parts of a page that its bytes `[from, to)` lie in (`from < to <=
/// PAGE`).
fn parts_of(from: usize, to: usize) -> u16 {
    let (a, b) = (from / KERNEL_PAGE, to.div_ceil(KERNEL_PAGE));
    ((u32::MAX >> (32 - (b - a))) << a) as u16
}

/// The byte runs `[from, to)` of a page that `parts` covers, in order.
pub(crate) fn part_runs(parts: u16) -> impl Iterator<Item = (usize, usize)> {
    let mut left = u32::from(parts);
    std::iter::from_fn(move || {
        if left == 0 {
            return None;
        }
        let a = left.trailing_zeros() as usize;
        let b = a + (!(left >> a)).trailing_zeros() as usize;
        left &= u32::MAX.checked_shl(b as u32).unwrap_or(0);
        Some((a * KERNEL_PAGE, b.min(PARTS) * KERNEL_PAGE))
    })
}

/// The part of one group that lies in a range asked about.
struct Group<'a> {
    /// The first page of the part.
    first: usize,
    /// The part's stamps.
    pages: &'a [AtomicU64],
    floor: &'a AtomicU64,
    top: &'a AtomicU64,
}

impl Stamps {
    pub(super) const fn new() -> Self {
        Self {
            tables: OnceLock::new(),
            parts: Mutex::new(HashMap::with_hasher(BuildHasherDefault::new())),
        }
    }

    /// The parts map, locked.
    fn parts(&self) -> std::sync::MutexGuard<'_, PageParts> {
        self.parts.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Table `index`, if one was made.
    fn table(&self, index: usize) -> Option<&[AtomicU64]> {
        let table = self.tables.get()?.get(index)?.load(Ordering::Acquire);
        // SAFETY: a table is `TABLE_LEN` entries made by `table_or_make`
        // and freed only when `self` is dropped.
        (!table.is_null()).then(|| unsafe { std::slice::from_raw_parts(table, TABLE_LEN) })
    }

    /// [`Stamps::table`], made if there is none.
    fn table_or_make(&self, index: usize) -> &[AtomicU64] {
        let slots = self.tables.get_or_init(|| {
            (0..SPACE_PAGES / TABLE_PAGES)
                .map(|_| AtomicPtr::default())
                .collect()
        });
        let slot = &slots[index];
        if slot.load(Ordering::Acquire).is_null() {
            let made: Box<[AtomicU64]> = (0..TABLE_LEN).map(|_| AtomicU64::new(0)).collect();
            let made = Box::into_raw(made).cast::<AtomicU64>();
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
        self.table(index).expect("made above")
    }

    /// Notes pages `[first, last)` written by asking `now`.
    pub(super) fn mark(&self, first: usize, last: usize, now: u64) {
        let last = last.min(SPACE_PAGES);
        let mut map = None;
        let mut page = first;
        while page < last {
            let index = page / TABLE_PAGES;
            let base = index * TABLE_PAGES;
            let end = last.min(base + TABLE_PAGES);
            let table = self.table_or_make(index);
            let (from, to) = (page - base, end - base);
            for group in from / GROUP..to.div_ceil(GROUP) {
                let (a, b) = ((group * GROUP).max(from), ((group + 1) * GROUP).min(to));
                let map = map.get_or_insert_with(|| self.parts());
                if b - a == GROUP {
                    let floor = table[FLOORS + group]
                        .fetch_max(now, Ordering::Relaxed)
                        .max(now);
                    // The group's entries the floor covers go: every part of
                    // theirs is at or below it, and the floor counts for
                    // every part of the group's pages.
                    if !map.is_empty() {
                        for k in a..b {
                            let page = base + k;
                            if map
                                .get(&page)
                                .is_some_and(|all| all.iter().all(|&at| at <= floor))
                            {
                                map.remove(&page);
                            }
                        }
                    }
                } else {
                    for k in a..b {
                        Self::note(table, map, base, k, now, WHOLE);
                    }
                }
                // After the stamps under it (see the module's text).
                table[TOPS + group].fetch_max(now, Ordering::Release);
            }
            page = end;
        }
    }

    /// Notes page `k` of `table` (whose first page is `base`) written in
    /// `parts` by asking `now`: its parts, then its stamp. The caller raises
    /// the group's top after.
    fn note(table: &[AtomicU64], map: &mut PageParts, base: usize, k: usize, now: u64, parts: u16) {
        let floor = table[FLOORS + k / GROUP].load(Ordering::Relaxed);
        let before = table[k].load(Ordering::Relaxed).max(floor);
        note_parts(map, base + k, before, parts, now);
        table[k].fetch_max(now, Ordering::Relaxed);
    }

    /// Notes parts `parts` of page `page` written by asking `now`.
    pub(super) fn mark_parts(&self, page: usize, parts: u16, now: u64) {
        if parts == WHOLE {
            return self.mark(page, page + 1, now);
        }
        if parts == 0 || page >= SPACE_PAGES {
            return;
        }
        let (index, k) = (page / TABLE_PAGES, page % TABLE_PAGES);
        let table = self.table_or_make(index);
        Self::note(table, &mut self.parts(), index * TABLE_PAGES, k, now, parts);
        // After the stamp under it (see the module's text).
        table[TOPS + k / GROUP].fetch_max(now, Ordering::Release);
    }

    /// Notes bytes `[from, to)` written by asking `now`: the parts they lie
    /// in of the pages at either end, the pages between whole.
    pub(super) fn mark_bytes(&self, from: usize, to: usize, now: u64) {
        let mut marker = Marker::new(self, now);
        marker.add(from, to);
        marker.end();
    }

    /// Notes the kernel's answer `found` (the addresses of the written
    /// pages, in order, each `granularity` bytes) written by asking `now`.
    pub(super) fn mark_found(&self, found: &[usize], granularity: usize, now: u64) {
        let size = granularity.max(KERNEL_PAGE);
        let mut marker = Marker::new(self, now);
        for &at in found {
            marker.add(at, at.saturating_add(size));
        }
        marker.end();
    }

    /// The parts of the groups in pages `[first, last)` that have a table,
    /// in page order.
    fn groups(&self, first: usize, last: usize) -> impl Iterator<Item = Group<'_>> {
        let last = last.min(SPACE_PAGES);
        let mut page = first;
        std::iter::from_fn(move || {
            while page < last {
                let index = page / TABLE_PAGES;
                let base = index * TABLE_PAGES;
                let Some(table) = self.table(index) else {
                    // Nothing marked in this table.
                    page = base + TABLE_PAGES;
                    continue;
                };
                let (from, group) = (page - base, (page - base) / GROUP);
                let end = last.min(base + (group + 1) * GROUP);
                let part = Group {
                    first: page,
                    pages: &table[from..end - base],
                    floor: &table[FLOORS + group],
                    top: &table[TOPS + group],
                };
                page = end;
                return Some(part);
            }
            None
        })
    }

    /// Appends to `out` the pages of `[first, last)` whose stamp is above
    /// `since`, with that stamp, in page order (0: every page ever marked).
    pub(super) fn since(&self, first: usize, last: usize, since: u64, out: &mut Vec<(usize, u64)>) {
        for group in self.groups(first, last) {
            if group.top.load(Ordering::Acquire) <= since {
                continue;
            }
            let floor = group.floor.load(Ordering::Relaxed);
            for (k, stamp) in group.pages.iter().enumerate() {
                let stamp = stamp.load(Ordering::Relaxed).max(floor);
                if stamp > since {
                    out.push((group.first + k, stamp));
                }
            }
        }
    }

    /// Calls `f` with each page of `[first, last)` (cut to the copy) that
    /// the copy whose notes are `taken` must take from title memory before
    /// it holds the title's bytes, in page order, with the parts of it to
    /// take: a page it never took whole ([`WHOLE`]), and a page whose stamp
    /// is above the asking it was taken at (its parts stamped above that
    /// asking, or [`WHOLE`] if it has no parts entry; see the module's
    /// text). This is the one rule every copy decides by: the parts hold
    /// every byte written since the copy took the page. A group of the
    /// range whose pages were all taken, and whose top is not above the group's
    /// floor in `taken`, is skipped with one compare: the cost is a look
    /// per group of the range and one per page of the groups written since
    /// the copy took them. Stops when `f` breaks, and says so.
    pub(super) fn stale_words(
        &self,
        first: usize,
        last: usize,
        taken: &Taken,
        mut f: impl FnMut(usize, u16) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let (first, last) = (
            first.max(taken.first),
            last.min(taken.first + taken.at.len()),
        );
        if first >= last {
            return ControlFlow::Continue(());
        }
        // The parts map, locked at the first page written since the copy
        // took it.
        let mut map = None;
        // The table looked at last, by its index.
        let mut table: Option<(usize, Option<&[AtomicU64]>)> = None;
        for group in first / GROUP..=(last - 1) / GROUP {
            let (a, b) = ((group * GROUP).max(first), ((group + 1) * GROUP).min(last));
            let word = group - taken.first / GROUP;
            let untaken = taken.untaken[word] & bits(a % GROUP, b - group * GROUP);
            let page = group * GROUP;
            let index = page / TABLE_PAGES;
            let stamps = match table {
                Some((at, stamps)) if at == index => stamps,
                _ => {
                    let stamps = self.table(index);
                    table = Some((index, stamps));
                    stamps
                }
            };
            let in_table = (page - index * TABLE_PAGES) / GROUP;
            let (top, floor) = stamps.map_or((0, 0), |t| {
                let top = t[TOPS + in_table].load(Ordering::Acquire);
                (top, t[FLOORS + in_table].load(Ordering::Relaxed))
            });
            if untaken == 0 && top <= taken.floor[word] {
                continue;
            }
            for page in a..b {
                if untaken & (1 << (page % GROUP)) != 0 {
                    f(page, WHOLE)?;
                    continue;
                }
                // Never marked: nothing written since it was taken.
                let Some(t) = stamps else {
                    continue;
                };
                let (k, at) = (page - index * TABLE_PAGES, taken.at[page - taken.first]);
                if t[k].load(Ordering::Relaxed).max(floor) > at {
                    let map = map.get_or_insert_with(|| self.parts());
                    f(page, parts_since(map, page, floor, at))?;
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// The tables made so far.
    #[cfg(test)]
    fn tables_made(&self) -> usize {
        self.tables.get().map_or(0, |slots| {
            slots
                .iter()
                .filter(|slot| !slot.load(Ordering::Relaxed).is_null())
                .count()
        })
    }
}

/// Gathers the byte ranges of one asking's writes, in order, into marks: a
/// parts mark per page written in part, one mark per run of pages written
/// whole (whose whole groups raise a floor, [`Stamps::mark`]).
struct Marker<'a> {
    stamps: &'a Stamps,
    now: u64,
    /// The page last added to, and its parts so far.
    page: Option<(usize, u16)>,
    /// Pages written whole, not yet marked.
    run: Option<(usize, usize)>,
}

impl<'a> Marker<'a> {
    fn new(stamps: &'a Stamps, now: u64) -> Self {
        Self {
            stamps,
            now,
            page: None,
            run: None,
        }
    }

    /// Adds bytes `[from, to)`. Ranges out of order are marked as well, in
    /// more marks.
    fn add(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        let (first, last) = (from / PAGE, (to - 1) / PAGE);
        let end = |page: usize| to.min((page + 1) * PAGE) - page * PAGE;
        self.part(first, parts_of(from - first * PAGE, end(first)));
        if first < last {
            if first + 1 < last {
                self.flush_page();
                self.run_of(first + 1, last);
            }
            self.part(last, parts_of(0, end(last)));
        }
    }

    /// Adds parts `parts` of page `page`.
    fn part(&mut self, page: usize, parts: u16) {
        match &mut self.page {
            Some((at, had)) if *at == page => *had |= parts,
            _ => {
                self.flush_page();
                self.page = Some((page, parts));
            }
        }
    }

    /// Adds pages `[first, last)` whole to the run, or marks the run and
    /// starts another.
    fn run_of(&mut self, first: usize, last: usize) {
        match &mut self.run {
            Some((_, end)) if *end == first => *end = last,
            _ => {
                self.flush_run();
                self.run = Some((first, last));
            }
        }
    }

    /// Marks the page last added to: whole into the run, else its parts.
    fn flush_page(&mut self) {
        if let Some((page, parts)) = self.page.take() {
            if parts == WHOLE {
                self.run_of(page, page + 1);
            } else {
                self.stamps.mark_parts(page, parts, self.now);
            }
        }
    }

    fn flush_run(&mut self) {
        if let Some((first, last)) = self.run.take() {
            self.stamps.mark(first, last, self.now);
        }
    }

    /// Marks what is left.
    fn end(mut self) {
        self.flush_page();
        self.flush_run();
    }
}

impl Drop for Stamps {
    fn drop(&mut self) {
        let Some(slots) = self.tables.get_mut() else {
            return;
        };
        for slot in slots.iter_mut() {
            let table = *slot.get_mut();
            if !table.is_null() {
                // SAFETY: made by `table_or_make` as a box of `TABLE_LEN`
                // entries; nothing refers to it once `self` goes.
                drop(unsafe {
                    Box::from_raw(std::ptr::slice_from_raw_parts_mut(table, TABLE_LEN))
                });
            }
        }
    }
}

/// What a host copy of title memory took of it (`crate::shadow::Shadow`):
/// for each 64 KiB page, the asking after which the copy last took it
/// whole, and for each group of [`GROUP`] pages (the stamps' own groups, at
/// the same addresses) the pages never taken whole and the lowest asking a
/// taken page was taken at, so [`Stamps::stale_words`] can pass over a
/// group with one compare.
///
/// For every page of the copy, either it was never taken whole, or the
/// copy holds the title's bytes as of the asking it was taken at and the
/// group's floor is not above that asking. Stamps only rise.
pub(crate) struct Taken {
    /// The copy's first page.
    first: usize,
    /// For each page from `first` on: the asking after which it was last
    /// taken whole (0: never).
    at: Vec<u64>,
    /// For each group from the one holding `first`: a bit per page, set
    /// while the page was never taken whole; clear for pages outside the
    /// copy (padding).
    untaken: Vec<u64>,
    /// For each such group: the lowest `at` of its pages that were taken
    /// (`u64::MAX` if none was).
    floor: Vec<u64>,
}

impl Taken {
    /// The notes of a copy of pages `[first, first + pages)`, none taken.
    pub(crate) fn new(first: usize, pages: usize) -> Self {
        let last = first + pages;
        let groups = if pages == 0 {
            0
        } else {
            (last - 1) / GROUP - first / GROUP + 1
        };
        let untaken = (0..groups)
            .map(|k| {
                let group = first / GROUP + k;
                let (a, b) = ((group * GROUP).max(first), ((group + 1) * GROUP).min(last));
                bits(a % GROUP, b - group * GROUP)
            })
            .collect();
        Self {
            first,
            at: vec![0; pages],
            untaken,
            floor: vec![u64::MAX; groups],
        }
    }

    /// Notes pages `[first, last)` (cut to the copy) taken whole after
    /// asking `now`, and brings the floors of their groups up to date.
    pub(crate) fn take(&mut self, first: usize, last: usize, now: u64) {
        let (first, last) = (first.max(self.first), last.min(self.first + self.at.len()));
        if first >= last {
            return;
        }
        for page in first..last {
            let at = &mut self.at[page - self.first];
            *at = (*at).max(now);
            self.untaken[page / GROUP - self.first / GROUP] &= !(1 << (page % GROUP));
        }
        for group in first / GROUP..=(last - 1) / GROUP {
            let word = group - self.first / GROUP;
            let (a, b) = (
                (group * GROUP).max(self.first),
                ((group + 1) * GROUP).min(self.first + self.at.len()),
            );
            let untaken = self.untaken[word];
            self.floor[word] = (a..b)
                .filter(|&page| untaken & (1 << (page % GROUP)) == 0)
                .map(|page| self.at[page - self.first])
                .min()
                .unwrap_or(u64::MAX);
        }
    }

    /// Whether pages `[first, last)` are all pages of the copy.
    pub(crate) fn holds(&self, first: usize, last: usize) -> bool {
        self.first <= first && last <= self.first + self.at.len()
    }

    /// The pages of the copy taken whole at least once.
    pub(crate) fn taken_pages(&self) -> usize {
        self.at.len()
            - self
                .untaken
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum::<usize>()
    }

    /// The asking after which page `page` was last taken whole (0: never,
    /// or not a page of the copy).
    #[cfg(test)]
    pub(crate) fn at(&self, page: usize) -> u64 {
        page.checked_sub(self.first)
            .and_then(|k| self.at.get(k))
            .copied()
            .unwrap_or(0)
    }
}

/// The bits of pages `[a, b)` of a group (`a < b <= GROUP`).
fn bits(a: usize, b: usize) -> u64 {
    debug_assert!(a < b && b <= GROUP);
    (u64::MAX >> (GROUP - b)) & (u64::MAX << a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self, below: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % below as u64) as usize
        }
    }

    /// Random marks, small, over whole groups and straddling groups and
    /// tables, answer as a map of every page's highest mark (the journal's
    /// map before the stamps): `since` for any stamp, and
    /// `stale_words` for copies around the range.
    #[test]
    fn stamps_match_a_page_model() {
        let stamps = Stamps::new();
        let mut model: BTreeMap<usize, u64> = BTreeMap::new();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // About the end of one table and the start of the next.
        let near = 3 * TABLE_PAGES - 1000;
        let space = 2000;
        let mut now = 0u64;
        for step in 0..6000 {
            // Not always the newest: the stamps keep the highest.
            now += 1;
            let at = if rng.next(5) == 0 { now / 2 + 1 } else { now };
            let first = near + rng.next(space);
            let len = match rng.next(4) {
                0 => GROUP * (1 + rng.next(4)) + rng.next(GROUP),
                1 => GROUP - first % GROUP,
                _ => 1 + rng.next(8),
            };
            stamps.mark(first, first + len, at);
            for page in first..first + len {
                let stamp = model.entry(page).or_default();
                *stamp = (*stamp).max(at);
            }
            let (a, b) = (
                near - 100 + rng.next(space + 300),
                near - 100 + rng.next(space + 300),
            );
            let (first, last) = (a.min(b), a.max(b) + 1);
            let since = [0, rng.next(now as usize + 1) as u64, now][rng.next(3)];
            let want: Vec<(usize, u64)> = model
                .range(first..last)
                .filter(|&(_, &w)| w > since)
                .map(|(&p, &w)| (p, w))
                .collect();
            let mut got = Vec::new();
            stamps.since(first, last, since, &mut got);
            assert_eq!(
                got, want,
                "step {step}: pages {first}..{last} since {since}"
            );
            // A copy around the range, pages taken at random askings
            // (some never, some in runs), asked about at random edges in
            // and around it: `stale_words` says what the page rule says.
            let (cf, cl) = (first - rng.next(70), last + rng.next(70));
            let mut taken = Taken::new(cf, cl - cf);
            let mut page = cf;
            while page < cl {
                let most = if rng.next(4) == 0 { 90 } else { 3 };
                let n = (1 + rng.next(most)).min(cl - page);
                let at = match rng.next(6) {
                    0 => None,
                    1 => Some(model.get(&page).copied().unwrap_or(1)),
                    2 => Some(since.max(1)),
                    3 => Some(1 + rng.next(now as usize) as u64),
                    _ => Some(now),
                };
                if let Some(at) = at {
                    taken.take(page, page + n, at);
                }
                page += n;
            }
            let (qa, qb) = (
                cf - 5 + rng.next(cl - cf + 10),
                cf - 5 + rng.next(cl - cf + 10),
            );
            let (qa, qb) = (qa.min(qb), qa.max(qb) + 1);
            let want: Vec<usize> = (qa.max(cf)..qb.min(cl))
                .filter(|&p| {
                    let at = taken.at(p);
                    at == 0 || model.get(&p).is_some_and(|&w| w > at)
                })
                .collect();
            let mut got = Vec::new();
            let flow = stamps.stale_words(qa, qb, &taken, |p, _| {
                got.push(p);
                ControlFlow::Continue(())
            });
            assert!(flow.is_continue());
            assert_eq!(got, want, "step {step}: pages {qa}..{qb} of {cf}..{cl}");
            let floors = (cf / GROUP..=(cl - 1) / GROUP).all(|group| {
                let pages = (group * GROUP).max(cf)..((group + 1) * GROUP).min(cl);
                let low = pages.map(|p| taken.at(p)).filter(|&at| at != 0).min();
                taken.floor[group - cf / GROUP] == low.unwrap_or(u64::MAX)
            });
            assert!(
                floors,
                "step {step}: every floor is its group's lowest take"
            );
            // Stopped at the first.
            let mut seen = 0;
            let flow = stamps.stale_words(qa, qb, &taken, |_, _| {
                seen += 1;
                ControlFlow::Break(())
            });
            assert_eq!(
                (seen, flow.is_break()),
                (usize::from(!want.is_empty()), !want.is_empty())
            );
            assert_eq!(
                taken.taken_pages(),
                (cf..cl).filter(|&p| taken.at(p) != 0).count()
            );
        }
        assert_eq!(stamps.tables_made(), 2, "the two tables marked, no more");
    }

    /// One page marked in a GiB: an answer for the whole GiB looks at its
    /// groups and at the 64 pages of the one written, and returns that page.
    #[test]
    fn since_skips_clean_groups() {
        let stamps = Stamps::new();
        let base = 5 * TABLE_PAGES;
        stamps.mark(base + 1000, base + 1001, 7);
        assert_eq!(stamps.groups(base, base + TABLE_PAGES).count(), GROUPS);
        let written = stamps
            .groups(base, base + TABLE_PAGES)
            .filter(|g| g.top.load(Ordering::Relaxed) > 0)
            .count();
        assert_eq!(written, 1, "one group to look into");
        let mut out = Vec::new();
        stamps.since(base, base + TABLE_PAGES, 0, &mut out);
        assert_eq!(out, [(base + 1000, 7)]);
        out.clear();
        stamps.since(base, base + TABLE_PAGES, 7, &mut out);
        assert!(out.is_empty());
        // No table, nothing to look at.
        assert_eq!(stamps.groups(0, base).count(), 0);
    }

    /// Where no page was ever marked (no table), a copy must take what it
    /// never took and nothing else; a group taken whole whose top is not
    /// above its floor is skipped, one written since is looked into.
    #[test]
    fn stale_words_without_tables_and_by_floor() {
        let stamps = Stamps::new();
        let base = 7 * TABLE_PAGES + 10;
        let mut taken = Taken::new(base, 200);
        let stale = |taken: &Taken, a: usize, b: usize| {
            let mut out = Vec::new();
            let _ = stamps.stale_words(a, b, taken, |p, _| {
                out.push(p);
                ControlFlow::Continue(())
            });
            out
        };
        assert_eq!(
            stale(&taken, 0, usize::MAX / PAGE),
            (base..base + 200).collect::<Vec<_>>()
        );
        taken.take(base, base + 200, 5);
        assert!(stale(&taken, base, base + 200).is_empty());
        assert_eq!(taken.taken_pages(), 200);
        stamps.mark(base + 70, base + 71, 6);
        stamps.mark(base + 150, base + 151, 4);
        assert_eq!(stale(&taken, base, base + 200), [base + 70]);
        taken.take(base + 70, base + 71, 6);
        assert!(stale(&taken, base, base + 200).is_empty());
    }

    /// Marks over the same GiB again and again make one table.
    #[test]
    fn repeated_marks_do_not_grow() {
        let stamps = Stamps::new();
        let base = 9 * TABLE_PAGES;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for now in 1..=10_000 {
            let first = base + rng.next(TABLE_PAGES - 300);
            stamps.mark(first, first + 1 + rng.next(300), now);
        }
        assert_eq!(stamps.tables_made(), 1);
        // Past the address space: nothing made, nothing found.
        stamps.mark(usize::MAX / PAGE - 10, usize::MAX / PAGE, 1);
        assert_eq!(stamps.tables_made(), 1);
        let mut out = Vec::new();
        stamps.since(SPACE_PAGES, usize::MAX / PAGE, 0, &mut out);
        assert!(out.is_empty());
    }

    /// The parts helpers: the parts of byte ranges, and back to runs.
    #[test]
    fn parts_and_runs() {
        assert_eq!(parts_of(0, PAGE), WHOLE);
        assert_eq!(parts_of(0, 1), 1);
        assert_eq!(parts_of(KERNEL_PAGE - 1, KERNEL_PAGE + 1), 0b11);
        assert_eq!(parts_of(PAGE - 1, PAGE), 1 << 15);
        let runs = |parts| part_runs(parts).collect::<Vec<_>>();
        assert_eq!(runs(0), []);
        assert_eq!(runs(WHOLE), [(0, PAGE)]);
        assert_eq!(
            runs(0b1000_0000_0110_0001),
            [
                (0, KERNEL_PAGE),
                (5 * KERNEL_PAGE, 7 * KERNEL_PAGE),
                (15 * KERNEL_PAGE, PAGE)
            ]
        );
    }

    /// What a copy of page `page` alone, taken after asking `taken` (0:
    /// never), must take: its parts, 0 if nothing.
    fn due(stamps: &Stamps, page: usize, taken: u64) -> u16 {
        let mut copy = Taken::new(page, 1);
        if taken != 0 {
            copy.take(page, page + 1, taken);
        }
        let mut parts = 0;
        let _ = stamps.stale_words(page, page + 1, &copy, |_, p| {
            parts = p;
            ControlFlow::Continue(())
        });
        parts
    }

    /// The page's stamp.
    fn stamp(stamps: &Stamps, page: usize) -> u64 {
        let mut out = Vec::new();
        stamps.since(page, page + 1, 0, &mut out);
        out.first().map_or(0, |p| p.1)
    }

    /// Ported from cool_fox's `the_parts_of_a_page_written_since_it_was_taken`
    /// (3ead56e, as carried in Belobug/VIctory-Belobug-AMD), on the stamp
    /// tables.
    #[test]
    fn the_parts_of_a_page_written_since_it_was_taken() {
        let stamps = Stamps::new();
        let base = 11 * TABLE_PAGES;
        assert_eq!(due(&stamps, base + 7, 0), WHOLE);
        stamps.mark(base + 7, base + 8, 3);
        // Parts 0 and 5 at 4, part 5 again at 6: at 2 the whole page is due.
        stamps.mark_parts(base + 7, 0b10_0001, 4);
        assert_eq!(stamp(&stamps, base + 7), 4);
        stamps.mark_parts(base + 7, 0b10_0000, 6);
        assert_eq!(due(&stamps, base + 7, 2), WHOLE);
        assert_eq!(due(&stamps, base + 7, 3), 0b10_0001);
        assert_eq!(due(&stamps, base + 7, 4), 0b10_0000);
        assert_eq!(due(&stamps, base + 7, 5), 0b10_0000);
        assert_eq!(due(&stamps, base + 7, 6), 0);
        // Written whole: all of it, whoever took it before.
        stamps.mark(base + 7, base + 8, 8);
        assert_eq!(due(&stamps, base + 7, 7), WHOLE);
        // A page never written before: what was not named was never written.
        stamps.mark_parts(base + 9, 0b100, 9);
        assert_eq!(due(&stamps, base + 9, 5), 0b100);
    }

    /// Ported from cool_fox's `notes_only_go_up` (673be0e): noted out of order, a
    /// page's stamp and its parts' only go up; and a group's floor counts
    /// for every part of its pages.
    #[test]
    fn notes_only_go_up() {
        let stamps = Stamps::new();
        let page = 12 * TABLE_PAGES + 3;
        stamps.mark(page, page + 1, 9);
        stamps.mark_parts(page, 0b11, 5);
        assert_eq!(stamp(&stamps, page), 9);
        assert_eq!(due(&stamps, page, 8), WHOLE, "all of it at 9");
        stamps.mark_parts(page, 0b100, 12);
        assert_eq!(due(&stamps, page, 9), 0b100);
        // An older whole note leaves the newer part as it is.
        stamps.mark(page, page + 1, 10);
        assert_eq!(due(&stamps, page, 10), 0b100);
        assert_eq!(due(&stamps, page, 9), WHOLE);
        assert_eq!(stamp(&stamps, page), 12);
        // The whole group written at 14: all of the page before it, then the
        // parts written after it alone.
        let group = page / GROUP * GROUP;
        stamps.mark(group, group + GROUP, 14);
        assert_eq!(due(&stamps, page, 12), WHOLE);
        assert_eq!(due(&stamps, page, 14), 0);
        stamps.mark_parts(page, 0b1000, 15);
        assert_eq!(due(&stamps, page, 14), 0b1000);
        assert_eq!(due(&stamps, page, 13), WHOLE);
        // An older group write under a newer part: the part stays newest.
        stamps.mark(group, group + GROUP, 13);
        assert_eq!(due(&stamps, page, 14), 0b1000);
    }

    /// Random writes on 4 KiB parts (parts marks, byte ranges, whole pages
    /// and groups, the kernel's answers, some marked out of order but all
    /// above every number handed out, as the journal hands them out) and
    /// refreshes of a copy that takes what `stale_words` says, in the parts
    /// it says, against a model of every part's writes: the parts taken
    /// hold every part written since the page was taken (so the copy holds
    /// the title's parts after each refresh), and are those parts exactly
    /// when one asking wrote the page since.
    #[test]
    fn parts_hold_every_write_since_the_take() {
        let stamps = Stamps::new();
        let mut rng = Rng(0x6c07_8965_2f1e_ab09);
        // Over the end of one table and into the next, five groups and more.
        let (first, pages) = (4 * TABLE_PAGES - 130, 340);
        let part = |page: usize, k: usize| (page - first) * PARTS + k;
        // The title's parts (the number of their last write) and the copy's.
        let (mut title, mut copy) = (vec![0u64; pages * PARTS], vec![0u64; pages * PARTS]);
        // Per part the highest asking that wrote it; per page the askings.
        let mut high = vec![0u64; pages * PARTS];
        let mut askings: Vec<std::collections::BTreeSet<u64>> = vec![Default::default(); pages];
        let mut taken = Taken::new(first, pages);
        let (mut handed, mut top, mut writes) = (0u64, 0u64, 0u64);
        let (mut exact, mut whole, mut copied, mut wanted) = (0, 0, 0u64, 0u64);
        for step in 0..20_000 {
            if rng.next(3) != 0 {
                // One asking's marks; its number above every one handed out.
                let now = handed + 1 + rng.next(3) as u64;
                top = top.max(now);
                let mut written: Vec<(usize, usize)> = Vec::new();
                for _ in 0..1 + rng.next(3) {
                    let at = (first + rng.next(pages)) * PAGE;
                    let room = (first + pages) * PAGE - at;
                    match rng.next(5) {
                        0 => {
                            let parts = 1 + rng.next(usize::from(WHOLE)) as u16;
                            stamps.mark_parts(at / PAGE, parts, now);
                            written.extend(part_runs(parts).map(|(a, b)| (at + a, at + b)));
                        }
                        1 => {
                            let from = at + rng.next(PAGE);
                            let to = (from + 1 + rng.next(3 * PAGE)).min(at + room);
                            stamps.mark_bytes(from, to, now);
                            written.push((from, to));
                        }
                        2 => {
                            let most = if rng.next(3) == 0 { 150 } else { 4 };
                            let n = (1 + rng.next(most)).min(room / PAGE);
                            stamps.mark(at / PAGE, at / PAGE + n, now);
                            written.push((at, at + n * PAGE));
                        }
                        _ => {
                            let mut found: Vec<usize> = (0..1 + rng.next(40))
                                .map(|_| {
                                    at + rng.next(room.min(4 * PAGE)) / KERNEL_PAGE * KERNEL_PAGE
                                })
                                .collect();
                            found.sort_unstable();
                            found.dedup();
                            stamps.mark_found(&found, KERNEL_PAGE, now);
                            written.extend(found.iter().map(|&a| (a, a + KERNEL_PAGE)));
                        }
                    }
                }
                for (from, to) in written {
                    for k in from / KERNEL_PAGE..to.div_ceil(KERNEL_PAGE) {
                        let (page, k) = (k * KERNEL_PAGE / PAGE, k % PARTS);
                        writes += 1;
                        title[part(page, k)] = writes;
                        high[part(page, k)] = high[part(page, k)].max(now);
                        askings[page - first].insert(now);
                    }
                }
                continue;
            }
            // A refresh of some pages at the newest number, which it hands
            // out: what is marked from now on is above it.
            let now = top;
            handed = now;
            let (a, b) = (first + rng.next(pages), first + rng.next(pages));
            let (a, b) = (a.min(b), a.max(b) + 1);
            let mut seen = Vec::new();
            let _ = stamps.stale_words(a, b, &taken, |page, parts| {
                seen.push((page, parts));
                ControlFlow::Continue(())
            });
            for page in a..b {
                let at = taken.at(page);
                let since: u16 = (0..PARTS)
                    .filter(|&k| high[part(page, k)] > at)
                    .fold(0, |m, k| m | 1 << k);
                let want = if at == 0 { WHOLE } else { since };
                let got = seen.iter().find(|s| s.0 == page).map_or(0, |s| s.1);
                assert_eq!(got & want, want, "step {step}: page {page} taken at {at}");
                assert_eq!(got == 0, want == 0, "step {step}: page {page} stale or not");
                if at != 0 && askings[page - first].range(at + 1..).count() == 1 {
                    assert_eq!(got, want, "step {step}: page {page}, one asking since");
                    exact += 1;
                }
                whole += usize::from(got == WHOLE);
                copied += u64::from(got.count_ones());
                wanted += u64::from(want.count_ones());
                for (lo, hi) in part_runs(got) {
                    for k in lo / KERNEL_PAGE..hi / KERNEL_PAGE {
                        copy[part(page, k)] = title[part(page, k)];
                    }
                }
                if got != 0 {
                    taken.take(page, page + 1, now);
                }
                let parts = part(page, 0)..part(page, PARTS);
                assert_eq!(
                    copy[parts.clone()],
                    title[parts],
                    "step {step}: page {page} held"
                );
            }
        }
        assert!(
            exact > 1000 && whole > 100,
            "both cases seen: {exact} exact, {whole} whole"
        );
        assert!(
            copied < wanted * 4,
            "parts copied {copied} for {wanted} written"
        );
        // The table past the boundary and the one before it, no more.
        assert_eq!(stamps.tables_made(), 2);
    }

    /// The parts entries of pages `[first, last)`.
    fn entries(stamps: &Stamps, first: usize, last: usize) -> usize {
        stamps
            .parts()
            .keys()
            .filter(|&&p| (first..last).contains(&p))
            .count()
    }

    /// A group's pages written in part, then the whole group: the floor
    /// covers their entries and they go, so the map holds only pages last
    /// written in part; a part written after the floor (out of order)
    /// keeps its entry. Copies taken before and after are told the right
    /// parts all along.
    #[test]
    fn a_whole_group_write_drops_the_entries_it_covers() {
        let stamps = Stamps::new();
        let group = 13 * TABLE_PAGES + 5 * GROUP;
        let next = group + GROUP;
        for (k, now) in (0..GROUP).step_by(3).zip(1..) {
            stamps.mark_parts(group + k, 1 << (k % PARTS), now);
        }
        stamps.mark_parts(next + 1, 0b11, 40);
        assert_eq!(entries(&stamps, group, next), GROUP.div_ceil(3));
        assert_eq!(due(&stamps, group + 3, 1), 1 << 3);
        // One page written in part at 30, then the whole group at 25 (an
        // older number), then at 50.
        stamps.mark_parts(group + 7, 0b100, 30);
        stamps.mark(group, next, 25);
        assert_eq!(
            entries(&stamps, group, next),
            1,
            "the part above the floor stays"
        );
        assert_eq!(due(&stamps, group + 7, 25), 0b100);
        assert_eq!(due(&stamps, group + 3, 24), WHOLE);
        assert_eq!(due(&stamps, group + 3, 25), 0);
        stamps.mark(group, next, 50);
        assert_eq!(entries(&stamps, group, next), 0, "covered by the floor");
        assert_eq!(
            entries(&stamps, next, next + GROUP),
            1,
            "another group's stays"
        );
        assert_eq!(due(&stamps, group + 7, 30), WHOLE);
        assert_eq!(due(&stamps, group + 7, 50), 0);
        // Written in part again: a new entry from the floor, exact parts.
        stamps.mark_parts(group + 9, 0b1_0000, 51);
        assert_eq!(due(&stamps, group + 9, 50), 0b1_0000);
        assert_eq!(due(&stamps, group + 9, 49), WHOLE);
        assert_eq!(entries(&stamps, group, next), 1);
        assert_eq!(due(&stamps, next + 1, 39), 0b11);
    }
}
