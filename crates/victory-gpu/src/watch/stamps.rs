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
use std::ops::ControlFlow;
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

    /// Calls `f` with each page of `[first, last)` (cut to the copy) that
    /// the copy whose notes are `taken` must take from title memory before
    /// it holds the title's bytes, in page order: a page it never took
    /// whole, and a page whose stamp is above the asking it was taken at.
    /// This is the one rule every copy decides by. A group of the range
    /// whose pages were all taken, and whose top is not above the group's
    /// floor in `taken`, is skipped with one compare: the cost is a look
    /// per group of the range and one per page of the groups written since
    /// the copy took them. Stops when `f` breaks, and says so.
    pub(super) fn stale_words(
        &self,
        first: usize,
        last: usize,
        taken: &Taken,
        mut f: impl FnMut(usize) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        let (first, last) = (
            first.max(taken.first),
            last.min(taken.first + taken.at.len()),
        );
        if first >= last {
            return ControlFlow::Continue(());
        }
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
                let top = t[TABLE_PAGES + GROUPS + in_table].load(Ordering::Acquire);
                (top, t[TABLE_PAGES + in_table].load(Ordering::Relaxed))
            });
            if untaken == 0 && top <= taken.floor[word] {
                continue;
            }
            for page in a..b {
                let stale = untaken & (1 << (page % GROUP)) != 0 || {
                    let stamp = stamps.map_or(0, |t| {
                        t[page - index * TABLE_PAGES]
                            .load(Ordering::Relaxed)
                            .max(floor)
                    });
                    stamp > taken.at[page - taken.first]
                };
                if stale {
                    f(page)?;
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
            let flow = stamps.stale_words(qa, qb, &taken, |p| {
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
            let flow = stamps.stale_words(qa, qb, &taken, |_| {
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
            let _ = stamps.stale_words(a, b, taken, |p| {
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
}
