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
//! Marks are made with the journal's lock held (`super::Journal`), between
//! the kernel's answer and the asking's number being handed out. The stamp
//! or floor is stored before the top, the top with release order, and a
//! reader loads the top first with acquire order, so whoever sees a top sees
//! the stamps under it.

use super::PAGE;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};

/// Pages in a group, the unit of the summary (4 MiB).
const GROUP: usize = 64;
/// The addresses a table covers: 1 GiB.
const TABLE_BITS: u32 = 30;
/// Pages in a table.
const TABLE_PAGES: usize = (1 << TABLE_BITS) / PAGE;
/// Groups in a table.
const GROUPS: usize = TABLE_PAGES / GROUP;
/// A table's entries: a stamp per page, then a floor per group, then a top
/// per group.
const TABLE_LEN: usize = TABLE_PAGES + 2 * GROUPS;
/// The user address space of 64-bit Windows (128 TiB), which the tables
/// cover; pages past it are never written.
const SPACE_PAGES: usize = (1 << 47) / PAGE;

/// The stamps of every page (see the module's text).
pub(super) struct Stamps {
    /// One per 1 GiB of addresses, null before a page there was marked;
    /// made at the first mark.
    tables: OnceLock<Box<[AtomicPtr<AtomicU64>]>>,
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
        }
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
        let mut page = first;
        while page < last {
            let index = page / TABLE_PAGES;
            let base = index * TABLE_PAGES;
            let end = last.min(base + TABLE_PAGES);
            let table = self.table_or_make(index);
            let (from, to) = (page - base, end - base);
            for group in from / GROUP..to.div_ceil(GROUP) {
                let (a, b) = ((group * GROUP).max(from), ((group + 1) * GROUP).min(to));
                if b - a == GROUP {
                    table[TABLE_PAGES + group].fetch_max(now, Ordering::Relaxed);
                } else {
                    for stamp in &table[a..b] {
                        stamp.fetch_max(now, Ordering::Relaxed);
                    }
                }
                // After the stamps under it (see the module's text).
                table[TABLE_PAGES + GROUPS + group].fetch_max(now, Ordering::Release);
            }
            page = end;
        }
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
                    floor: &table[TABLE_PAGES + group],
                    top: &table[TABLE_PAGES + GROUPS + group],
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

    /// Whether no page of `[first, last)` has a stamp above what a copy
    /// took it at: `taken[k]` for page `first + k` (as many as the pages).
    pub(super) fn none_above(&self, first: usize, last: usize, taken: &[u64]) -> bool {
        debug_assert_eq!(taken.len(), last - first);
        self.groups(first, last).all(|group| {
            let taken = &taken[group.first - first..][..group.pages.len()];
            let top = group.top.load(Ordering::Acquire);
            if taken.iter().all(|&t| top <= t) {
                return true;
            }
            let floor = group.floor.load(Ordering::Relaxed);
            group
                .pages
                .iter()
                .zip(taken)
                .all(|(stamp, &t)| stamp.load(Ordering::Relaxed).max(floor) <= t)
        })
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
    /// map before the stamps): `since` for any stamp, and `none_above`.
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
            let taken: Vec<u64> = (first..last)
                .map(|p| match rng.next(3) {
                    0 => model.get(&p).copied().unwrap_or(0),
                    1 => since,
                    _ => now,
                })
                .collect();
            let above = (first..last).any(|p| model.get(&p).is_some_and(|&w| w > taken[p - first]));
            assert_eq!(
                stamps.none_above(first, last, &taken),
                !above,
                "step {step}"
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
}
