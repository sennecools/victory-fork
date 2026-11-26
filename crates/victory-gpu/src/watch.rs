//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).
//!
//! The journal's numbers are kept in [`stamps`]' tables: one per page and a
//! summary per 4 MiB, so an answer costs what was written since the asker's
//! last take, not what was ever written.

use std::collections::BTreeMap;
use std::sync::Mutex;

mod bits;
mod stamps;

use bits::{Bits, Blocks};
use stamps::Stamps;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    // Only whether it can be tracked: no page is wanted (none is stamped
    // after `u64::MAX`).
    if len == 0 || writes(start, end, views, u64::MAX).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// For each page, the highest asking that found it written. Marked
    /// with this lock held only, so that an asking's marks are made before
    /// its number is handed out and before another asking of the same
    /// memory begins (the guards rely on that, [`guard_blocks`]).
    stamps: Stamps,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

impl Journal {
    const fn new() -> Self {
        Self {
            asked: 0,
            stamps: Stamps::new(),
            found: Vec::new(),
            ahead: Vec::new(),
            ahead_asked: 0,
            kept: false,
        }
    }
}

impl Journal {
    /// The pages of `[first, last)` last found written by an asking after
    /// `since` (0: since tracking began), with that asking, in page order.
    fn pages(&self, first: usize, last: usize, since: u64) -> Vec<(usize, u64)> {
        let mut pages = Vec::new();
        self.stamps.since(first, last, since, &mut pages);
        pages
    }

    /// Notes `pages` (in order, each as often as it comes) written by asking
    /// `now`: a mark per run of pages that follow each other.
    fn mark_pages(&self, pages: impl IntoIterator<Item = usize>, now: u64) {
        let mut run: Option<(usize, usize)> = None;
        for page in pages {
            match &mut run {
                Some((_, end)) if page == *end => *end += 1,
                Some((first, end)) if (*first..*end).contains(&page) => {}
                _ => {
                    if let Some((first, end)) = run {
                        self.stamps.mark(first, end, now);
                    }
                    run = Some((page, page + 1));
                }
            }
        }
        if let Some((first, end)) = run {
            self.stamps.mark(first, end, now);
        }
    }
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal::new());

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    ask_ahead_in(&JOURNAL, &left_to_ask(ranges));
}

/// `ranges` without those the kernel would refuse, for the askings ahead
/// ([`without_known_views`] with `VICTORY_R5_FLUSH=1`, else
/// [`without_guarded`]).
fn left_to_ask(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    if crate::host::r5_flush::on() {
        without_known_views(ranges)
    } else {
        without_guarded(ranges)
    }
}

/// `ranges` without those of which a block is read-only by our hand, by the
/// bits ([`PROTECTED`]), for the askings ahead:
/// the kernel would refuse the asking, and, asked together with others,
/// theirs (cool_fox 2aa8134). [`writes`] answers them from the bits or the
/// guards.
fn without_guarded(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    ranges
        .iter()
        .filter(|&&(start, end)| !protected_bits(start, end))
        .copied()
        .collect()
}

/// `ranges` without those whose first block the guards hold as view memory
/// ([`known_view`]), for the askings ahead (`VICTORY_R5_FLUSH=1`): the
/// kernel cannot answer for them, and a run of joined ranges holding one
/// fails and is asked again by halves. [`writes`] answers them from the
/// guards, as it does after such a failed asking.
fn without_known_views(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    use crate::host::r5_flush::{Count, count};
    if view_ask() {
        return ranges.to_vec();
    }
    let g = guards();
    let (mut spans, mut bytes) = (0, 0);
    let left = ranges
        .iter()
        .copied()
        .filter(|&(start, end)| {
            let view = start < end && known_view_in(&g, start);
            if view {
                spans += 1;
                bytes += (end - start) as u64;
            }
            !view
        })
        .collect();
    drop(g);
    count(Count::ViewSpans, spans);
    count(Count::ViewSpanBytes, bytes);
    left
}

/// `ranges` as page ranges in order, empty ones left out.
fn pieces_of(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    pieces.sort_unstable();
    pieces
}

/// `pieces` (page ranges in order) cut into the runs asked about together:
/// each run `[i, k)` of pieces closer than [`AHEAD_GAP`] to the one before
/// and spanning at most [`AHEAD_MAX`] pages.
fn runs_of(pieces: &[(usize, usize)]) -> Vec<std::ops::Range<usize>> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        runs.push(i..k);
        i = k;
    }
    runs
}

/// [`ask_ahead`] in `journal`.
fn ask_ahead_in(journal: &Mutex<Journal>, ranges: &[(usize, usize)]) {
    let pieces = pieces_of(ranges);
    if pieces.len() < 2 {
        return;
    }
    // On the profile: the journal's share of a refresh (its lock, its map),
    // less the kernel askings made under it.
    let _journal = crate::profile::journal_timer();
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    for run in runs_of(&pieces) {
        ask_run(j, &pieces[run]);
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    ask_ahead_done_in(&JOURNAL);
}

/// [`ask_ahead_done`] in `journal`.
fn ask_ahead_done_in(journal: &Mutex<Journal>) {
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    ask_once_in(&JOURNAL, &left_to_ask(ranges));
}

/// [`ask_once`] in `journal`, whatever [`once`] says.
fn ask_once_in(journal: &Mutex<Journal>, ranges: &[(usize, usize)]) {
    let _journal = crate::profile::journal_timer();
    ask_once_end_in(journal);
    ask_ahead_in(journal, ranges);
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    ask_once_end_in(&JOURNAL);
}

/// [`ask_once_end`] in `journal`.
fn ask_once_end_in(journal: &Mutex<Journal>) {
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    wrote_in(&JOURNAL, at, len);
}

/// [`wrote`] in `journal`.
fn wrote_in(journal: &Mutex<Journal>, at: usize, len: usize) {
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    j.stamps
        .mark(at / PAGE, at.saturating_add(len).div_ceil(PAGE), now);
}

/// The GPU wrote title memory `[at, at + len)` through a home of the memory
/// core (`crate::home::after_wait`), which write watching cannot see: its
/// pages count as written now, whether or not an asking stands, so that
/// every copy over them takes them again at its next refresh. ([`wrote`]
/// acts only while a submission's one asking stands.)
pub(crate) fn gpu_wrote(at: usize, len: usize) {
    gpu_wrote_in(&JOURNAL, at, len);
}

/// [`gpu_wrote`] in `journal`.
fn gpu_wrote_in(journal: &Mutex<Journal>, at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    j.asked += 1;
    let now = j.asked;
    // An asking that stands moves with it, as in `wrote_in`.
    if !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    j.stamps
        .mark(at / PAGE, at.saturating_add(len).div_ceil(PAGE), now);
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let asking = crate::profile::refresh_stamp();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    // (A failed asking found nothing: its time is the kernel's still.)
    let found = if failed { &[][..] } else { &j.found[..count] };
    crate::profile::kernel_asked(asking, ((last - first) * PAGE) as u64, found);
    if failed {
        failed_ask(first, last);
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.mark_pages(j.found[..count].iter().map(|&at| at / PAGE), j.ahead_asked);
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages last found written by an asking after `since` and that asking, with
/// the number of the asking that stands (as [`writes`] gives it). `None` if
/// the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize, since: u64) -> Option<(u64, Vec<(usize, u64)>)> {
    asked_in(&JOURNAL, start, end, since)
}

/// [`asked`] in `journal`.
fn asked_in(
    journal: &Mutex<Journal>,
    start: usize,
    end: usize,
    since: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.pages(first, last, since)))
}

/// The pages of `[start, end)` (whole pages) last found written by an
/// asking after `since` (0: since tracking began; a copy passes the asking
/// after which it last took the range): `(page number, asking)` in page
/// order, and this asking's number, which is higher than any in the list
/// but those the kernel just reported. `None` if the memory is not
/// write-watched (nor, with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(
    start: usize,
    end: usize,
    guarded: bool,
    since: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    writes_in(&JOURNAL, start, end, guarded, since)
}

/// [`writes`] in `journal`.
fn writes_in(
    journal: &Mutex<Journal>,
    start: usize,
    end: usize,
    guarded: bool,
    since: u64,
) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let _journal = crate::profile::journal_timer();
    let mut j = journal.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((j.ahead_asked, j.pages(first, last, since)));
    }
    // Read-only since an asking noted it: the journal has every write of it.
    if guarded && published(first * PAGE, last * PAGE) {
        j.asked += 1;
        return Some((j.asked, j.pages(first, last, since)));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    // (Any block of the range, by the bits.)
    let known_view =
        guarded && (protected_bits(first * PAGE, last * PAGE) || known_view(first * PAGE));
    let asking = crate::profile::refresh_stamp();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known_view
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    if !known_view {
        // (A failed asking found nothing: its time is the kernel's still.)
        let found = if failed { &[][..] } else { &j.found[..count] };
        crate::profile::kernel_asked(asking, ((last - first) * PAGE) as u64, found);
        if failed {
            failed_ask(first, last);
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    let guarded = if failed {
        count = 0;
        if !guarded {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let guarding = std::time::Instant::now();
        let written = guard(first * PAGE, last * PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    j.mark_pages(
        j.found[..count]
            .iter()
            .map(|&at| at / PAGE)
            .chain(guarded.iter().copied()),
        now,
    );
    // What the guards made clean of it is noted: from now on the bits tell
    // of it (the journal's lock is still held, no other asking of it came
    // between).
    if failed {
        publish(&guarded);
    }
    Some((now, j.pages(first, last, since)))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        // shortcut: every page ever written, filtered by `seen` below; a
        // stamp per page taken (one `since` for the range) is later work.
        let Some((_, pages)) = writes(from, to, true, 0) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// The blocks that are read-only by our hand, not written since, and of
/// which the journal holds every write from before that: for those, what
/// [`writes`] answers is in the journal, and neither the kernel nor the
/// guards need be asked; what [`nothing_new`] knows of view memory, it knows
/// from these bits.
///
/// cool_fox measured, before these bits, a frame asking about view memory
/// some 1,500 times, nearly always to hear that nothing was written, on up
/// to seven threads at once, each under the guards' one lock: 94 ms a frame
/// of all threads' time, most of it waiting. Ported from cool_fox 2aa8134
/// (its `VICTORY_LIGHT_REFRESH`; only its watch part), 74045e4
/// ([`PROTECTED`]) and 7bc7c07 (the handler), with the race fixes of
/// 2be0515/689df31 ([`unguard`]).
///
/// Why that is so: a block's bit is set under the guards' lock while the
/// block is clean, by the asking that made it clean, after that asking noted
/// in the journal what it found ([`publish`]; no other asking of the block
/// comes between: the journal's lock is held from before the guards are
/// asked until after). It is cleared under
/// the guards' lock before the block becomes writable again ([`unguard`]),
/// before another view of its page does ([`open`], [`views_mapped`]), or
/// when it is forgotten ([`title_memory_remapped`]). So while the bit is set
/// nothing wrote the block through any address, and who sees the bit and
/// looks at the journal after that finds every earlier write in it. A write
/// that comes while the bit is looked at is no different from one that
/// comes just after an asking: the next asking finds the block open.
static PUBLISHED: Bits = Bits::new();

/// The blocks read-only by our hand, or about to be: set under the guards'
/// lock ahead of the protection, cleared under it once the block is open
/// again (or forgotten, [`title_memory_remapped`]), on every path. So a
/// block read-only by our hand always has its bit set, and every block in
/// `Guards::protected` does. Who finds none set has no guard to open and
/// need not wait for the lock: the kernel is not asked about such a range
/// (it would refuse), nor such a range among the askings ahead
/// ([`without_guarded`]), and who writes title memory, or faults on it, and
/// finds no block of it set ([`writing`], [`title_memory_written`], the
/// handler of [`install`]) takes no lock.
static PROTECTED: Bits = Bits::new();

/// An asking made blocks of `pages` clean and noted them in the journal
/// ([`PUBLISHED`]).
fn publish(pages: &[usize]) {
    if pages.is_empty() {
        return;
    }
    let g = guards();
    for &page in pages {
        // (Holes of placeholders too: nothing can be there before the
        // runtime maps it, and it tells of that before the title is.)
        if g.clean.contains(page) || g.held.contains(page) {
            PUBLISHED.set(page);
        }
    }
}

/// Whether `block` lies in a placeholder: a reservation in which pages come
/// to be only by mapping a view (or an allocation) in its place, which an
/// ordinary commit cannot do.
///
/// The holes of placeholders are known for what they are (`Guards::held`):
/// a block noted absent that lies in a placeholder is taken as absent until
/// the runtime tells of a mapping there ([`title_memory_remapped`]) or an
/// allocation ([`title_memory_allocated`]), and is known by its bits
/// ([`PUBLISHED`]); the other holes are asked about with the guards' lock
/// not held. cool_fox measured, before its change, most ranges asked about
/// having holes, reservations the title maps its pages into as it streams
/// (1,125 of a frame's 1,530 asked view ranges): each asking put a question
/// to Windows for each hole, under the guards' lock, and 111 ms a frame of
/// all threads' time went into waiting for that lock. Ported from cool_fox
/// 74045e4 (its `VICTORY_KNOWN_HOLES`).
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk).
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryVirtualMemory(
            process: *mut std::ffi::c_void,
            at: *const std::ffi::c_void,
            class: u32,
            out: *mut std::ffi::c_void,
            len: usize,
            returned: *mut usize,
        ) -> i32;
    }
    // (Class 3 answers with the kind of memory in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The runtime gave the title memory at `[at, at + len)`, allocated or
/// committed (`victory-hle` xmem): what was known of holes there is asked
/// again ([`placeholder`]).
pub fn title_memory_allocated(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    guards().held.remove_range(
        at / BLOCK,
        at.saturating_add(len).div_ceil(BLOCK),
        |block| PUBLISHED.clear(block),
    );
}

/// Once a frame: a hundredth of the known holes ([`placeholder`]) are
/// looked at, in case one was filled with nobody telling
/// ([`title_memory_allocated`], [`title_memory_remapped`]), which no title
/// we know does.
pub(crate) fn look_at_holes() {
    const PARTS: usize = 100;
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let blocks: Vec<usize> = {
        let g = guards();
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part = g.held.first_from(from, g.held.len() / PARTS + 1);
        FROM.store(
            part.last().map_or(0, |b| b + 1),
            std::sync::atomic::Ordering::Relaxed,
        );
        part
    };
    let mut until = 0;
    for block in blocks {
        if block >= until {
            until = uncommitted_end(block);
        }
        if block >= until {
            // (Or unmapped memory was mapped since the part was taken: told
            // already, and forgotten.)
            let mut g = guards();
            if g.held.remove(block) {
                PUBLISHED.clear(block);
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{:#x}", block * BLOCK),
                        "a placeholder of the title's was filled with nobody telling (its holes are taken as known)"
                    );
                });
            }
        }
    }
}

/// Whether nothing of `[start, end)` can have been written since an asking
/// noted it in the journal ([`PUBLISHED`]).
fn published(start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether a block of `[start, end)` is read-only by our hand, by the bits
/// ([`PROTECTED`]).
fn protected_bits(start: usize, end: usize) -> bool {
    PROTECTED.any(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether nothing of `[start, end)` was written that a copy with the notes
/// `copied` (the asking after which each page from page `base` on was taken)
/// does not hold, as [`writes`] would answer with nothing asked: the
/// submission's asking covers the range, or its blocks are known by their
/// bits ([`PUBLISHED`]). `false` if something would have to be asked, or a
/// page was never taken. A flush of host copies leaves out those that have
/// nothing new so, before any thread is woken and anything asked for them
/// (`crate::shadow::Shadow::nothing_new`; ported from cool_fox 2aa8134, its
/// `VICTORY_NOTHING_NEW`).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first
        .checked_sub(base)
        .and_then(|from| copied.get(from..last - base))
    else {
        return false;
    };
    if taken.contains(&0) {
        return false;
    }
    // The bits first, then the journal, as in `writes`.
    let known = published(start, end);
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    (known || j.ahead.iter().any(|&(a, b)| a <= first && last <= b))
        && j.stamps.none_above(first, last, taken)
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    !view_ask() && known_view_in(&guards(), start)
}

/// Whether the kernel is asked about view memory anyway
/// (`VICTORY_VIEW_ASK=1` or `VICTORY_GUARD_ASK=1`, [`known_view`]).
fn view_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// [`known_view`] with the guards' lock held, whatever `VICTORY_VIEW_ASK`
/// says.
fn known_view_in(g: &Guards, start: usize) -> bool {
    let first = start / BLOCK;
    g.protected.contains(first) || g.absent.contains(first)
}

/// Whether the guards hold the block at `start` as view memory, whatever
/// `VICTORY_VIEW_ASK` says: for the R5 flush's counters
/// (`crate::host::r5_flush`).
pub(crate) fn known_view_at(start: usize) -> bool {
    known_view_in(&guards(), start)
}

/// A kernel asking about pages `[first, last)` failed: counted
/// (`crate::host::r5_flush`).
fn failed_ask(first: usize, last: usize) {
    use crate::host::r5_flush::{Count, count};
    count(Count::FailedAsks, 1);
    count(Count::FailedBytes, ((last - first) * PAGE) as u64);
}

/// The guards' sets of blocks, as words of bits (cool_fox measured, before
/// its change, the title's write faults waiting for the guards' lock most of
/// their time, and a look at a range walking a tree of every guarded block:
/// 586 faults a frame took 69 ms of the title's threads; ported from
/// cool_fox 7bc7c07, its `VICTORY_GUARD_BITS`).
struct Guards {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them.
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Blocks,
    /// Of those, the ones in placeholders ([`placeholder`]).
    held: Blocks,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one makes the others
    /// no longer clean ([`open`]). A write range can be huge (`fread`), so
    /// only these are looked up.
    aliased: Blocks,
    /// Blocks opened by a write fault, and when ([`rest`]).
    opened: std::collections::HashMap<usize, std::time::Instant, OpenedHash>,
}

/// The hasher of [`Guards::opened`] (one that can be made in a `static`).
type OpenedHash = std::hash::BuildHasherDefault<std::collections::hash_map::DefaultHasher>;

impl Guards {
    /// Whether `block` is left open for now: the title has just written it
    /// ([`rest`]).
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        !rest.is_zero()
            && self
                .opened
                .get(&block)
                .is_some_and(|at| at.elapsed() < rest)
    }

    /// A write fault opened `block` ([`rest`]).
    fn opened(&mut self, block: usize) {
        const MANY: usize = 1 << 13;
        if self.opened.len() >= MANY {
            self.opened.retain(|_, at| at.elapsed().as_millis() < 50);
            if self.opened.len() >= MANY {
                self.opened.clear();
            }
        }
        self.opened.insert(block, std::time::Instant::now());
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: Blocks::new(),
    clean: Blocks::new(),
    absent: Blocks::new(),
    held: Blocks::new(),
    aliased: Blocks::new(),
    opened: std::collections::HashMap::with_hasher(OpenedHash::new()),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let known = !*ASK
        .get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
        && (protected_bits(start, start + 1) || {
            let g = guards();
            let first = start / BLOCK;
            g.protected.contains(first) || g.absent.contains(first)
        });
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    Some(guard_blocks(start / BLOCK, end.div_ceil(BLOCK)))
}

/// How long a block the title has just opened by a write fault is left open
/// by the refreshes that take it: [`REST`].
///
/// cool_fox measured 220 of a frame's 586 faults coming within 3 ms after a
/// refresh had taken the block and made it read-only, while the title was
/// still writing it, and what a refresh takes asked about again by the next
/// within a millisecond: made read-only, written, opened, up to a dozen
/// times a frame. A block that rests is told as written at each asking, at
/// every view of its page, so nothing is missed; it is made read-only at the
/// first asking after its rest. Ported from cool_fox 7bc7c07 (its
/// `VICTORY_GUARD_REST`, microseconds there, 3,000 unless given).
fn rest() -> std::time::Duration {
    #[cfg(test)]
    if let Some(rest) = REST_IN_TEST.get() {
        return rest;
    }
    REST
}

/// [`rest`]: what cool_fox runs.
const REST: std::time::Duration = std::time::Duration::from_millis(3);

#[cfg(test)]
thread_local! {
    /// [`rest`], for what this thread asks and faults.
    static REST_IN_TEST: std::cell::Cell<Option<std::time::Duration>> = const { std::cell::Cell::new(None) };
}

/// For the tests: how long blocks rest after a write on this thread.
#[cfg(test)]
fn set_rest_in_test(rest: std::time::Duration) {
    REST_IN_TEST.set(Some(rest));
}

/// Makes the other views of `block`'s page read-only, as [`protect`] does;
/// whether all of them are now (none rests or failed).
///
/// A block read-only by our hand whose page has other views can be left
/// with one of those writable: a write fault opens only the view it came
/// through ([`open`]), and a view that rests ([`rest`]) or cannot be
/// guarded stays open. A block is clean, and so known by its bits, only
/// while every other view of its page is read-only by our hand: a write
/// told at one that is not ([`writing`], [`title_memory_written`]) takes no
/// lock and opens nothing. [`protect`] keeps such a block from being clean;
/// the next asking, finding it read-only and not clean, must not note it
/// clean either until the other views are guarded. With
/// `VICTORY_VIEW_GUARD=0` the other views are left as they are: true.
fn cover(g: &mut Guards, block: usize) -> bool {
    if !view_guard() {
        return true;
    }
    let shown = crate::inplace::pages_of(block * BLOCK);
    if shown.len() < 2 || shown.len() > MANY_VIEWS {
        return true;
    }
    let mut all = true;
    for alias in shown {
        let alias = alias / BLOCK;
        g.aliased.insert(alias);
        if alias != block && !protect_alias(g, alias) {
            all = false;
        }
    }
    all
}

/// [`guard`] of blocks `[first, last)`, a view: the blocks to take are read
/// off the sets by words, and the guards' lock is let go between the runs
/// of blocks made read-only, for the title's faults that wait for it (each
/// run is looked at anew under the lock). Ported from cool_fox 7bc7c07.
///
/// Letting go is safe: no other asking guards these blocks meanwhile (the
/// caller holds the journal's lock), and what a fault or the runtime does
/// meanwhile only opens or forgets blocks. A block opened after the look
/// below was not clean at it, or is told as written at the next asking; one
/// forgotten is made read-only again, as any block that is not clean.
fn guard_blocks(first: usize, last: usize) -> Vec<usize> {
    let g = guards();
    if g.clean.all(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        return Vec::new();
    }
    // Holes that an ordinary commit may have filled, with no word of the
    // runtime's: one question for the blocks of a region, none for those of
    // placeholders ([`placeholder`]). The questions are put with the lock
    // let go: what the runtime tells meanwhile takes a block out of the set,
    // and it is looked at as any other below.
    let unsure: Vec<usize> = g
        .absent
        .iter(first, last)
        .filter(|&block| !g.held.contains(block))
        .collect();
    drop(g);
    let mut came = Vec::new();
    let mut uncommitted_until = first;
    for block in unsure {
        if block >= uncommitted_until {
            uncommitted_until = uncommitted_end(block);
        }
        if block >= uncommitted_until {
            came.push(block);
        }
    }
    let mut g = guards();
    for block in came {
        g.absent.remove(block);
    }
    // The range's blocks that are neither clean nor (still) holes: all
    // told as written.
    let mut written = Vec::new();
    g.clean.in_neither(&g.absent, first, last, &mut written);
    let rest = rest();
    let mut run: Option<(usize, usize)> = None;
    for &block in &written {
        if let Some((from, to)) = run
            && to != block
        {
            protect(&mut g, from, to);
            run = None;
            drop(g);
            g = guards();
        }
        if g.protected.contains(block) {
            // Written through the runtime, or through another view of its
            // page ([`open`]), still read-only: clean once every view of it
            // is read-only ([`cover`]).
            if let Some((from, to)) = run.take() {
                protect(&mut g, from, to);
            }
            if cover(&mut g, block) {
                g.clean.insert(block);
            }
        } else if !g.rests(block, rest) {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            // Just written by the title: left open, told as written.
            protect(&mut g, from, to);
        }
    }
    if let Some((from, to)) = run {
        protect(&mut g, from, to);
    }
    // (A block is a page.)
    written
        .iter()
        .flat_map(|&block| block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE)
        .collect()
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                // A block with a view beside it that could not be guarded,
                // or rests, is not clean: a write through that view is not
                // seen, nor (by the bits) one told there.
                if alias != block && !protect_alias(g, alias) {
                    g.clean.remove(block);
                }
            }
        }
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks no
/// longer clean because another view of theirs was written, and blocks
/// opened because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it. Whether it is
/// read-only by our hand now.
fn protect_alias(g: &mut Guards, block: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    if g.protected.contains(block) {
        return true;
    }
    // Just written by the title ([`rest`]): left open.
    if g.rests(block, rest()) {
        return false;
    }
    // Ahead of the protection.
    PROTECTED.set(block);
    let mut old = 0;
    let timed = protect_timer();
    // SAFETY: title view memory; only its protection changes.
    let done =
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    protected(timed, usize::from(done));
    if done {
        g.protected.insert(block);
    } else {
        PROTECTED.clear(block);
    }
    done
}

/// With `VICTORY_R5_FLUSH=1`: when a `VirtualProtect` of the guards starts,
/// for [`protected`].
fn protect_timer() -> Option<std::time::Instant> {
    crate::host::r5_flush::on().then(std::time::Instant::now)
}

/// A `VirtualProtect` of the guards, begun at `timed`, made `blocks` blocks
/// read-only: counted (`crate::host::r5_flush`).
fn protected(timed: Option<std::time::Instant>, blocks: usize) {
    use crate::host::r5_flush::{Count, count};
    count(Count::Protects, 1);
    count(Count::Guarded, blocks as u64);
    if let Some(timed) = timed {
        count(Count::ProtectUs, timed.elapsed().as_micros() as u64);
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        let timed = protect_timer();
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        protected(timed, if done { to - from } else { 0 });
        done
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.insert_all(from..to);
        g.clean.insert_all(from..to);
    };
    // Ahead of the protection (`PROTECTED`).
    (from..to).for_each(|block| PROTECTED.set(block));
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
            continue;
        }
        PROTECTED.clear(block);
        if block < uncommitted_end(block) {
            g.absent.insert(block);
            if placeholder(block) {
                g.held.insert(block);
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The blocks are opened
/// ([`unguard`]); the other views of their pages stay read-only and are no
/// longer clean (told as written before the first block is writable). The
/// write that faulted (or is ours, `crate::host::copy_mapped`) goes through
/// one view, and every guard over the pages must copy them again.
///
/// The other views were opened too once (a change of protection each, under
/// the guards' lock, with other faults waiting) and made read-only again
/// when the page was next taken, though the title writes through one of
/// them: cool_fox measured 586 faults a frame opening 880 blocks. A write
/// through one left as it was faults, and opens it. Ported from cool_fox
/// 7bc7c07 (its `VICTORY_OPEN_ONE_VIEW`).
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return false;
    }
    let blocks: Vec<usize> = g.protected.iter(first, last).collect();
    let Guards { aliased, clean, .. } = &mut *g;
    aliased.remove_range(first, last, |block| {
        for alias in crate::inplace::pages_of(block * BLOCK) {
            let alias = alias / BLOCK;
            let was = clean.remove(alias);
            PUBLISHED.clear(alias);
            #[cfg(feature = "diagnostics")]
            ALIASES[1].fetch_add(
                u64::from(was && alias != block),
                std::sync::atomic::Ordering::Relaxed,
            );
            #[cfg(not(feature = "diagnostics"))]
            let _ = was;
        }
    });
    unguard(g, &blocks);
    !blocks.is_empty()
}

/// Gives write access back to `blocks`, guarded ones: all of them are told
/// as written ([`PUBLISHED`]) before the first can be written, for a write
/// through one is a write of the others' bytes, and who asks about one by
/// its bits takes no lock. Their [`PROTECTED`] bits go once they are
/// writable.
///
/// The race fix of cool_fox 2be0515 (also 689df31, the same change on
/// another line): opened one range after another, one view was writable
/// for some microseconds while another was still known by its bits.
fn unguard(g: &mut Guards, blocks: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    for &block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PUBLISHED.clear(block);
    }
    for &block in blocks {
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
        PROTECTED.clear(block);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    // The guards' lock only if a block of it is read-only by our hand (an
    // other view's is opened only with its own; [`protect`] and [`cover`]
    // leave no view clean beside one they could not guard).
    let end = at.saturating_add(len);
    if PROTECTED.any(at / BLOCK, end.div_ceil(BLOCK)) {
        open(&mut guards(), at, end);
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
    crate::home::cpu_touched(at, len, crate::home::Touch::Wrote);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    // Nothing to count or open if no block of it is read-only by our hand.
    if PROTECTED.any(at / BLOCK, end.div_ceil(BLOCK)) {
        let mut g = guards();
        let guarded = g.protected.count(at / BLOCK, end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
    crate::home::cpu_touched(at, len, crate::home::Touch::Wrote);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    // All views together, under the lock: none may be writable while
    // another is known by its bits (cool_fox 2be0515, [`unguard`]).
    let blocks: Vec<usize> = ranges
        .iter()
        .flat_map(|&(at, n)| {
            g.protected
                .iter(at / BLOCK, at.saturating_add(n).div_ceil(BLOCK))
        })
        .collect();
    #[cfg(feature = "diagnostics")]
    ALIASES[2].fetch_add(
        u64::from(!blocks.is_empty()),
        std::sync::atomic::Ordering::Relaxed,
    );
    unguard(&mut g, &blocks);
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
///
/// One lock per 16 MB of addresses (taken modulo [`SHARDS`]), not one for
/// all title memory: a read holds the locks of the addresses it reads, a
/// change those of the addresses it changes. With one lock, every texture
/// refresh waited for every remap anywhere in title memory, and the remaps
/// are kernel calls of 0.1-0.8 ms each, some 1,000-2,700 a second while the
/// world streams in (run mem-W3a): up to 1.1 s of a frame's texture
/// preparation was waiting for it (mem-W0). Several locks are always taken
/// in ascending order, by readers and changes alike, so none waits on
/// another in a circle.
static MAPPING: [std::sync::RwLock<()>; SHARDS] = [const { std::sync::RwLock::new(()) }; SHARDS];

/// Locks in [`MAPPING`], and the bits of address each covers (16 MB).
const SHARDS: usize = 256;
const SHARD_BITS: u32 = 24;

/// The locks of [`MAPPING`] covering `[at, at + len)`, ascending; every
/// one for `len` 0 (the whole of title memory) or a range of [`SHARDS`]
/// spans or more.
fn shards(at: usize, len: usize) -> Vec<usize> {
    if len == 0 {
        return (0..SHARDS).collect();
    }
    let first = at >> SHARD_BITS;
    let last = at.saturating_add(len - 1) >> SHARD_BITS;
    if last - first + 1 >= SHARDS {
        return (0..SHARDS).collect();
    }
    let mut all: Vec<usize> = (first..=last).map(|s| s % SHARDS).collect();
    all.sort_unstable();
    all
}

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    _guards: Vec<std::sync::RwLockWriteGuard<'static, ()>>,
    /// When it was had, how long it was waited for (µs), and where.
    #[cfg(feature = "diagnostics")]
    taken: (std::time::Instant, u64, &'static std::panic::Location<'static>),
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again. All of
/// title memory; [`title_memory_remapping_range`] for part of it.
#[track_caller]
pub fn title_memory_remapping() -> Remapping {
    title_memory_remapping_range(0, 0)
}

/// [`title_memory_remapping`] for `[at, at + len)` only: every address
/// whose mapping the change touches, the parts of views it takes away and
/// maps again included. Reads elsewhere go on.
#[track_caller]
pub fn title_memory_remapping_range(at: usize, len: usize) -> Remapping {
    #[cfg(feature = "diagnostics")]
    let asked = std::time::Instant::now();
    let guards = shards(at, len)
        .into_iter()
        .map(|s| MAPPING[s].write().unwrap_or_else(|e| e.into_inner()))
        .collect();
    Remapping {
        _guards: guards,
        #[cfg(feature = "diagnostics")]
        taken: (
            std::time::Instant::now(),
            asked.elapsed().as_micros() as u64,
            std::panic::Location::caller(),
        ),
    }
}

/// Diagnostics: the changes of the mapping by where the runtime made them since the
/// last asking: file, line, and [changes, µs waited for in-place reads to end, µs held,
/// longest µs held]. The texture refreshes wait on these ("waiting for title memory to
/// hold still").
#[cfg(feature = "diagnostics")]
static REMAPS: std::sync::Mutex<Vec<(&'static str, u32, [u64; 4])>> = std::sync::Mutex::new(Vec::new());

#[cfg(feature = "diagnostics")]
impl Drop for Remapping {
    fn drop(&mut self) {
        let (at, waited, place) = self.taken;
        let held = at.elapsed().as_micros() as u64;
        let mut all = REMAPS.lock().unwrap_or_else(|e| e.into_inner());
        let i = match all.iter().position(|r| r.0 == place.file() && r.1 == place.line()) {
            Some(i) => i,
            None => {
                all.push((place.file(), place.line(), [0; 4]));
                all.len() - 1
            }
        };
        let k = &mut all[i].2;
        k[0] += 1;
        k[1] += waited;
        k[2] += held;
        k[3] = k[3].max(held);
    }
}

/// [`REMAPS`], taken (and reset).
#[cfg(feature = "diagnostics")]
pub(crate) fn remaps_taken() -> Vec<(&'static str, u32, [u64; 4])> {
    std::mem::take(&mut *REMAPS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] Vec<std::sync::RwLockReadGuard<'static, ()>>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory. All of title memory; [`title_memory_reading_range`]
/// for part of it.
pub fn title_memory_reading() -> Reading {
    title_memory_reading_range(0, 0)
}

/// [`title_memory_reading`] for `[at, at + len)` only: changes of mapping
/// elsewhere go on meanwhile.
pub fn title_memory_reading_range(at: usize, len: usize) -> Reading {
    Reading(
        shards(at, len)
            .into_iter()
            .map(|s| MAPPING[s].read().unwrap_or_else(|e| e.into_inner()))
            .collect(),
    )
}

/// [`title_memory_reading_range`] if it can be had at once: `None` while
/// the runtime changes a mapping there (or waits to), for a caller that has
/// another way to read and would rather not wait.
pub fn title_memory_reading_now(at: usize, len: usize) -> Option<Reading> {
    let mut guards = Vec::new();
    for s in shards(at, len) {
        match MAPPING[s].try_read() {
            Ok(guard) => guards.push(guard),
            Err(std::sync::TryLockError::Poisoned(e)) => guards.push(e.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        }
    }
    Some(Reading(guards))
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::home::cpu_touched(at, len, crate::home::Touch::Remapped);
    // The directory of raw-address shaders looks at its pages there again.
    crate::memory::remapped(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let Guards {
        protected,
        clean,
        absent,
        held,
        aliased,
        ..
    } = &mut *g;
    absent.remove_range(range.start, range.end, |block| {
        if held.remove(block) {
            PUBLISHED.clear(block);
        }
    });
    protected.remove_range(range.start, range.end, |block| {
        clean.remove(block);
        PUBLISHED.clear(block);
        PROTECTED.clear(block);
    });
    aliased.remove_range(range.start, range.end, |_| {});
    // What rested there was the old mapping's ([`rest`]): the new one is
    // guarded at the next asking, as any other.
    if !g.opened.is_empty() {
        if range.len() < g.opened.len() {
            for block in range {
                g.opened.remove(&block);
            }
        } else {
            g.opened.retain(|block, _| !range.contains(block));
        }
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's. It is the first vectored handler in the process
/// and also the one that takes the faults of our own copies
/// (`crate::safecopy`, `VICTORY_SAFE_COPY=1`), which register it through
/// here.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        // A fault of our own copies (`crate::safecopy`): whatever it is, if
        // it is not a block we guard, the copy fails and goes on from its
        // failing exit. Never so for any other instruction, so with
        // `VICTORY_SAFE_COPY` off (no such copy runs) this is as before.
        let ours = crate::safecopy::failing_exit(r.ExceptionAddress as usize);
        // SAFETY: the system passes the context of the fault.
        let context = unsafe { (*info).ContextRecord };
        let fail = move |exit: usize| {
            // SAFETY: the context the system passes, to go on from: the
            // copy's own failing exit, in the same leaf.
            unsafe { (*context).Rip = exit as u64 };
            EXCEPTION_CONTINUE_EXECUTION
        };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return match ours {
                Some(exit) => fail(exit),
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        if write_fault_taken(r.ExceptionInformation[1]) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        match ours {
            Some(exit) => fail(exit),
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// What the handler of [`install`] does with a write fault at `at`: whether
/// the write can go on (a block of ours opened, or one already writable).
fn write_fault_taken(at: usize) -> bool {
    let protected = PROTECTED.any(at / BLOCK, at / BLOCK + 1);
    #[cfg(test)]
    if let Some(between) = BETWEEN_IN_TEST.get() {
        between(at);
    }
    // Another thread opened the block since this one faulted ([`PROTECTED`]
    // is cleared only once a block is writable again): nothing to wait for
    // the guards' lock for, which a guard pass may hold across its kernel
    // calls (cool_fox 7bc7c07). A clear bit alone does not say the block is
    // not ours: it may have been opened and guarded again since the bit was
    // read (a guard pass lets the lock go between its runs), so a block not
    // writable now is looked at under the lock.
    if !protected && writable(at) {
        return true;
    }
    // Held across both checks: another thread may have opened the block
    // since this one faulted, and a copy may have guarded it again since
    // (then it is ours again, not read-only for some other reason).
    let mut g = guards();
    if open(&mut g, at, at + 1) {
        // (Left open for a while, [`rest`].)
        if !rest().is_zero() {
            g.opened(at / BLOCK);
        }
        crate::host::r5_flush::count(crate::host::r5_flush::Count::Faults, 1);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardFaults, 1);
        return true;
    }
    writable(at)
}

#[cfg(test)]
thread_local! {
    /// Run by [`write_fault_taken`] on this thread between its look at the
    /// bits and its look at the memory, for a test of what may come between.
    static BETWEEN_IN_TEST: std::cell::Cell<Option<fn(usize)>> = const { std::cell::Cell::new(None) };
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// For a test that checks how blocks are guarded again after a write
    /// fault, not how they rest: no block written on this thread rests
    /// ([`rest`]), whatever [`REST`] is. With a rest, a block
    /// the test has just written is told as written, and left open, at every
    /// asking within the rest (at every view of its page); a test that
    /// expects it clean at the next asking would hold only if the rest were
    /// over by then. `a_block_just_written_rests` and
    /// `no_write_to_resting_views_is_lost_between_askers` check the rest.
    fn no_rest() {
        set_rest_in_test(std::time::Duration::ZERO);
    }

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false, 0).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false, 0).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false, 0).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        // An asker that took the range after `now` (a texture's last
        // asking) hears of what was written since only.
        let (_, none) = writes(start, end, false, now).expect("watched");
        assert!(none.is_empty(), "nothing since {now}");
        // SAFETY: committed pages of ours.
        unsafe { ((p + 3 * PAGE) as *mut u8).write(1) };
        let (newest, since) = writes(start, end, false, later).expect("watched");
        assert_eq!(
            since,
            [(p / PAGE + 3, newest)],
            "only the page written since"
        );
        let (_, all) = writes(start, end, false, 0).expect("watched");
        assert_eq!(all, [(p / PAGE + 1, now), (p / PAGE + 3, newest)]);
        assert!(
            writes(start, end, false, newest)
                .expect("watched")
                .1
                .is_empty()
        );
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true,
                0
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// Our own copy (`crate::safecopy`) into a guarded view block faults
    /// into the handler, which opens the block and lets the copy go on: it
    /// lands and the block counts as written. Into a read-only page not
    /// guarded by us it fails, and the process goes on.
    #[test]
    fn safe_copies_into_guarded_blocks_open_them() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY, UnmapViewOfFile,
            VirtualProtect,
        };
        install();
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let bytes = b"from our copy";
        // SAFETY: 13 bytes of our view, nothing else refers to them.
        assert!(unsafe {
            crate::safecopy::copy((view + 100) as *mut u8, bytes.as_ptr(), bytes.len())
        });
        // SAFETY: the bytes just copied.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, bytes);
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // Read-only for some other reason: the copy fails.
        let (block, len) = (view + BLOCK, bytes.len());
        let mut old = 0;
        // SAFETY: a block of our view; the guards were taken off it first.
        unsafe {
            open(&mut guards(), view, view + 2 * BLOCK);
            VirtualProtect(block as *const _, BLOCK, PAGE_READONLY, &mut old);
            let copied = crate::safecopy::copy(block as *mut u8, bytes.as_ptr(), len);
            assert!(!copied);
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write fault on a block that another thread opened since (its
    /// [`PROTECTED`] bit clear, the memory writable) is answered without the
    /// guards' lock, while another thread holds it as a guard pass does
    /// across its kernel calls.
    #[test]
    fn faults_on_opened_blocks_take_no_lock() {
        // SAFETY: a fresh read-write allocation of ours, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                BLOCK,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        assert!(!PROTECTED.any(p / BLOCK, p / BLOCK + 1));
        let held = guards();
        let (done, taken) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            done.send(write_fault_taken(p + 8)).ok();
        });
        let taken = taken.recv_timeout(std::time::Duration::from_secs(10));
        drop(held);
        assert_eq!(
            taken,
            Ok(true),
            "taken at once, the guards' lock not waited for"
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A write faults on a guarded block; before the handler looks at the
    /// bits another thread opens the block (as `writing` does, noting no
    /// rest), and before it looks at the memory a guard pass makes the block
    /// read-only again. The bit read was clear, the memory is not writable:
    /// the block is ours again, and the handler opens it under the guards'
    /// lock rather than taking the title's write for a crash.
    #[test]
    fn a_block_guarded_again_during_a_fault_is_opened() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        no_rest();
        // SAFETY: a pagefile section of one block, mapped read-write; both
        // are let go at the end.
        let (section, view) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                BLOCK as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
            (section, view)
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + BLOCK).is_some());
        assert!(!writable(view), "guarded");
        // The fault's thread reads the bit after another thread opened the
        // block; the guard pass comes between that read and the look at the
        // memory.
        writing(view, 8);
        assert!(writable(view) && !PROTECTED.any(view / BLOCK, view / BLOCK + 1));
        BETWEEN_IN_TEST.set(Some(|at| {
            protect(&mut guards(), at / BLOCK, at / BLOCK + 1)
        }));
        let taken = write_fault_taken(view + 8);
        BETWEEN_IN_TEST.set(None);
        assert!(taken, "the title's write goes on");
        assert!(writable(view), "opened again");
        title_memory_remapped(view, BLOCK);
        // SAFETY: our view and its section's handle.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: view as *mut _,
            });
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }
    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        no_rest();
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        no_rest();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        no_rest();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        no_rest();
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        no_rest();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        no_rest();
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// What a copy of a range has taken: the asking after which it took
    /// each page, as `crate::shadow::Shadow::copy_stale` notes it.
    fn take(start: usize, end: usize, guarded: bool, copied: &mut [u64]) {
        let (now, written) = writes(start, end, guarded, 0).expect("watched or a view");
        let base = start / PAGE;
        for (page, at) in copied.iter_mut().enumerate() {
            let write = written
                .iter()
                .find(|w| w.0 == base + page)
                .map_or(0, |w| w.1);
            if *at == 0 || write > *at {
                *at = now;
            }
        }
    }

    /// A range nothing wrote is known to be as it was without asking
    /// (`nothing_new`), and is so no more once something wrote it: in
    /// write-watched memory from the submission's asking, in view memory
    /// from the bits. Ported from cool_fox 2aa8134's
    /// `unchanged_ranges_are_known`, at the watch's level (ours copies the
    /// ends of a range as before, so the range is whole pages).
    #[test]
    fn unchanged_ranges_are_known() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        const PAGES: usize = 6;
        // (With a rest the written block is not known until a taking after
        // the rest: `a_block_just_written_rests` checks it is not.)
        no_rest();
        for view in [false, true] {
            // SAFETY: memory of the test's own, never given back (it stays
            // in the journal and the guards).
            let at = unsafe {
                if view {
                    let section = CreateFileMappingW(
                        -1isize as _,
                        std::ptr::null(),
                        PAGE_READWRITE,
                        0,
                        (PAGES * PAGE) as u32,
                        std::ptr::null(),
                    );
                    MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize
                } else {
                    VirtualAlloc(
                        std::ptr::null(),
                        PAGES * PAGE,
                        MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                        PAGE_READWRITE,
                    ) as usize
                }
            };
            assert_ne!(at, 0);
            let (start, end) = (at, at + (PAGES - 1) * PAGE);
            let mut copied = [0u64; PAGES - 1];
            let known = |copied: &[u64]| {
                // The submission's one asking, as a flush has it (again if
                // another test's submission ended it).
                let mut known = false;
                for _ in 0..100 {
                    ask_once(&[(start, end), (at + (PAGES - 1) * PAGE, at + PAGES * PAGE)]);
                    known = nothing_new(start, end, start / PAGE, copied);
                    if known {
                        break;
                    }
                }
                ask_once_end();
                known
            };
            assert!(
                !nothing_new(start, end, start / PAGE, &copied),
                "view {view}: never taken"
            );
            take(start, end, view, &mut copied);
            // (Taken twice: the first asking of view memory guards it.)
            take(start, end, view, &mut copied);
            assert!(known(&copied), "view {view}: taken");
            // SAFETY: the test's own memory.
            unsafe { ((at + 2 * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(!known(&copied), "view {view}: written");
            take(start, end, view, &mut copied);
            assert!(known(&copied), "view {view}: taken again");
            // Outside the range: as it was.
            // SAFETY: as above.
            unsafe { ((at + (PAGES - 1) * PAGE + 8) as *mut u8).write_volatile(1) };
            assert!(known(&copied), "view {view}: written outside it");
            // A longer range was never taken whole.
            assert!(!nothing_new(
                start,
                at + PAGES * PAGE,
                start / PAGE,
                &copied
            ));
            if view {
                title_memory_remapped(at, PAGES * PAGE);
                assert!(!nothing_new(start, end, start / PAGE, &copied), "forgotten");
            }
        }
    }

    /// Of view memory asked about by many threads, written by the title
    /// through any of its views, or by us after telling of it (`writing`),
    /// and with views of its pages mapped and unmapped meanwhile: every
    /// asking after a write tells of it, at every view, whatever is known by
    /// the bits. Ported from cool_fox 2aa8134/7bc7c07/2be0515's
    /// `writes_to_views_are_not_lost` (without its writes past the guards,
    /// which our line does not have).
    fn writes_to_views_are_not_lost<const VIEWS: usize>(rest_us: u64) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            UnmapViewOfFile,
        };
        const PAGES: usize = 12;
        const THREADS: usize = 8;
        let rest = std::time::Duration::from_micros(rest_us);
        // SAFETY: views of a section of our own, which stay (at their
        // addresses other tests' memory would be told of these writes).
        let (section, all): (usize, [usize; VIEWS]) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (PAGES * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let all = std::array::from_fn(|_| {
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize;
                assert!(at != 0 && at.is_multiple_of(PAGE));
                title_memory_remapped(at, PAGES * PAGE);
                crate::inplace::title_memory_view(at, PAGES * PAGE, section as usize, 0);
                at
            });
            (section as usize, all)
        };
        let by_bits = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            if VIEWS > 1 {
                s.spawn(move || {
                    set_rest_in_test(rest);
                    let mut last = [0u64; VIEWS];
                    for round in 0..300usize {
                        let (page, asked) = (round * 5 % PAGES, round % VIEWS);
                        // SAFETY (all): a view of one page of the section,
                        // this thread's alone.
                        let one = unsafe {
                            MapViewOfFile(
                                section as _,
                                FILE_MAP_WRITE,
                                0,
                                (page * PAGE) as u32,
                                PAGE,
                            )
                            .Value as usize
                        };
                        assert_ne!(one, 0);
                        title_memory_remapped(one, PAGE);
                        crate::inplace::title_memory_view(one, PAGE, section, (page * PAGE) as u64);
                        unsafe { ((one + THREADS * 8) as *mut u64).write_volatile(round as u64) };
                        let (now, written) = writes(
                            all[asked] + page * PAGE,
                            all[asked] + (page + 1) * PAGE,
                            true,
                            0,
                        )
                        .expect("a view");
                        let number = written.first().map_or(0, |w| w.1);
                        assert!(
                            number > last[asked],
                            "page {page}, through a new view: {number} is not after {}",
                            last[asked]
                        );
                        last[asked] = now.max(last[asked]);
                        {
                            let _changing = title_memory_remapping();
                            title_memory_remapped(one, PAGE);
                            unsafe {
                                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                                    Value: one as *mut _,
                                })
                            };
                        }
                        last = [last[asked]; VIEWS];
                    }
                });
            }
            for t in 0..THREADS {
                let by_bits = &by_bits;
                s.spawn(move || {
                    set_rest_in_test(rest);
                    let mut last = [[0u64; PAGES]; VIEWS];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..500 {
                        let page = next(PAGES);
                        let from = page - next(page.min(3) + 1);
                        let to = page + 1 + next((PAGES - page).min(3));
                        // Written through one view, asked about at another.
                        let (through, asked) = (all[next(VIEWS)], next(VIEWS));
                        let (at, last) = (all[asked], &mut last[asked]);
                        if next(4) == 0 {
                            // Nothing written: what is asked counts as before.
                            by_bits.fetch_add(
                                usize::from(published(at + from * PAGE, at + to * PAGE)),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            let (now, _) = writes(at + from * PAGE, at + to * PAGE, true, 0).expect("a view");
                            for l in &mut last[from..to] {
                                *l = now;
                            }
                            continue;
                        }
                        let word = through + page * PAGE + t * 8;
                        let told = next(4) == 0;
                        if told {
                            writing(word, 8);
                        }
                        // SAFETY: a word of the test's views, this thread's.
                        unsafe { (word as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, true, 0).expect("a view");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(
                            number > last[page],
                            "{VIEWS} views, page {page}, through {:x} (told {told}), asked at {:x}: {number} is not after {}",
                            through / PAGE + page,
                            at / PAGE + page,
                            last[page]
                        );
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        // A block that rests after a write ([`rest`]) is not clean, so not
        // known by its bits, until an asking after its rest; here the title
        // writes every page again within milliseconds, and with three views
        // a page is clean only once none of its views rests: with a rest of
        // milliseconds the bits may answer none of the askings above.
        // Without a rest they must.
        if rest.is_zero() {
            assert!(by_bits.into_inner() > 0, "the bits answered none");
        }
        // Either way: once nothing has been written for longer than any
        // rest, one asking at each view has every view known by its bits.
        set_rest_in_test(rest);
        std::thread::sleep(rest + std::time::Duration::from_millis(1));
        for &at in &all {
            writes(at, at + PAGES * PAGE, true, 0).expect("a view");
        }
        for &at in &all {
            assert!(
                published(at, at + PAGES * PAGE),
                "nothing written since: the bits know the view at {at:x}"
            );
        }
    }

    #[test]
    fn no_write_to_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(0);
    }

    /// The same with blocks that rest after a write ([`rest`], here 50
    /// microseconds whatever [`REST`] is).
    #[test]
    fn no_write_to_resting_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(50);
        writes_to_views_are_not_lost::<3>(50);
    }

    /// A write opens the view it goes through and no other: the others stay
    /// read-only, are told as written, and take writes of their own. Ported
    /// from cool_fox 7bc7c07.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        set_rest_in_test(std::time::Duration::ZERO);
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(writable(two) && !writable(one));
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view that was left as it was, after the other.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(
            guard(one, one + BLOCK),
            Some(vec![page(one)]),
            "not clean, though never opened"
        );
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // Of a page that was told at one view, a kernel write to the other
        // finds it open.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block the title has just written is not made read-only, and is told
    /// as written at each asking, at every view of it, until it has rested
    /// ([`rest`], set here for this thread). Ported from
    /// cool_fox 7bc7c07.
    #[test]
    fn a_block_just_written_rests() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        let rests = |rests: bool| {
            // (Not resting: a rest that is over at once, so that the views
            // are still made read-only together.)
            set_rest_in_test(if rests {
                std::time::Duration::from_secs(3600)
            } else {
                std::time::Duration::from_nanos(1)
            })
        };
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        rests(true);
        assert_eq!(
            guard(one, one + BLOCK),
            Some(Vec::new()),
            "nothing written: as it was"
        );
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(writable(two), "no fault for the next write");
            assert!(writes(two, two + BLOCK, true, 0).is_some());
            assert!(!published(two, two + BLOCK) && !published(one, one + BLOCK));
            unsafe { ((two + 8) as *mut u8).write(1) };
        }
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block that rests and is then mapped anew (the runtime tells of it,
    /// [`title_memory_remapped`]) rests no more: the new mapping is guarded
    /// at the next asking, as any other. (A rest left over from an old
    /// mapping was kept for the whole rest, here an hour.)
    #[test]
    fn a_block_mapped_anew_rests_no_more() {
        set_rest_in_test(std::time::Duration::from_secs(3600));
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        // SAFETY: a view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(writable(one), "it rests");
        title_memory_remapped(one, BLOCK);
        crate::inplace::title_memory_view(one, BLOCK, section, 0);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]), "mapped anew");
        assert!(!writable(one), "guarded again, not resting");
        drop_views(section, &[one, two], BLOCK);
    }

    /// The same of pages shown at three addresses, each of which has copies
    /// of its own, with views of single pages mapped beside them meanwhile.
    #[test]
    fn no_write_through_another_view_is_lost_between_askers() {
        writes_to_views_are_not_lost::<3>(0);
    }

    /// A page with a view beside the guarded one that cannot be made
    /// read-only (partly committed: one kernel page of its block): a write
    /// told there (`writing`) is told at the guarded view, at every asking.
    /// The guarded block must not be noted clean, nor known by its bits,
    /// while the other
    /// view is writable: the told write takes no lock there and would open
    /// nothing. (Which view shows which pages is what the runtime tells; the
    /// test tells it so of a view of another section.)
    #[test]
    fn a_write_told_at_a_view_that_cannot_be_guarded_is_not_lost() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };
        set_rest_in_test(std::time::Duration::ZERO);
        // SAFETY: two fresh pagefile sections of one block, the second
        // reserved with one kernel page committed; each mapped once. The
        // views keep the sections alive once the handles are closed.
        let (guarded, other) = unsafe {
            let map = |flags: u32| {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | flags,
                    0,
                    BLOCK as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK).Value as usize;
                assert_ne!(view, 0);
                (section, view)
            };
            let (section, guarded) = map(0);
            let (reserved, other) = map(SEC_RESERVE);
            CloseHandle(reserved);
            let committed = VirtualAlloc(other as _, KERNEL_PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(committed as usize, other);
            crate::inplace::title_memory_view(guarded, BLOCK, section as usize, 0);
            crate::inplace::title_memory_view(other, BLOCK, section as usize, 0);
            CloseHandle(section);
            (guarded, other)
        };
        let page = guarded / PAGE;
        assert_eq!(guard(guarded, guarded + BLOCK), Some(vec![page]));
        for asking in 1..3 {
            assert_eq!(
                guard(guarded, guarded + BLOCK),
                Some(vec![page]),
                "asking {asking}: another view of it is writable"
            );
            assert!(!published(guarded, guarded + BLOCK), "asking {asking}");
        }
        assert!(writable(other));
        writing(other + 8, 8);
        // SAFETY: the committed kernel page of the other view.
        unsafe { ((other + 8) as *mut u64).write_volatile(1) };
        assert_eq!(
            guard(guarded, guarded + BLOCK),
            Some(vec![page]),
            "the told write"
        );
        for view in [guarded, other] {
            title_memory_remapped(view, BLOCK);
            // SAFETY: a view mapped above.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
    }

    /// A range of a placeholder with views in it is told as written right
    /// whatever is mapped into it, unmapped from it or allocated there, and
    /// it is known whole by its bits, its holes among it, until the runtime
    /// tells of
    /// such a change. Ported from cool_fox 74045e4 (our runtime tells of an
    /// unmapping as of any change, `title_memory_remapped`).
    #[test]
    fn holes_are_known_until_the_runtime_tells() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
            UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 8;
        no_rest();
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (BLOCKS * PAGE) as u32,
                std::ptr::null(),
            );
            (at, section)
        };
        assert_ne!(at, 0);
        let map = |block: usize| unsafe {
            let to = at + block * PAGE;
            VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                to as *const _,
                (block * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, to);
            title_memory_remapped(to, PAGE);
        };
        map(1);
        map(5);
        assert!(
            placeholder(at / PAGE) && placeholder(at / PAGE + 7) && !placeholder(at / PAGE + 5)
        );
        let end = at + BLOCKS * PAGE;
        let ask = |since: u64| -> (u64, Vec<usize>) {
            let (now, written) = writes(at, end, true, 0).expect("a view");
            let written = written
                .into_iter()
                .filter(|w| w.1 > since)
                .map(|w| w.0 - at / PAGE);
            (now, written.collect())
        };
        let (first, all) = ask(0);
        assert_eq!(all, (0..BLOCKS).collect::<Vec<_>>());
        assert!(published(at, end), "with its holes");
        // Written; mapped into; unmapped.
        // SAFETY (all): views of the test's own.
        unsafe { ((at + 5 * PAGE) as *mut u8).write_volatile(1) };
        assert!(!published(at, end));
        let (second, all) = ask(first);
        assert_eq!(all, [5]);
        map(2);
        assert!(!published(at, end));
        let (third, all) = ask(second);
        assert_eq!(all, [2]);
        unsafe { ((at + 2 * PAGE) as *mut u8).write_volatile(1) };
        let (fourth, all) = ask(third);
        assert_eq!(all, [2]);
        title_memory_remapped(at + 5 * PAGE, PAGE);
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS {
                Value: (at + 5 * PAGE) as *mut _,
            };
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
        }
        let (fifth, all) = ask(fourth);
        assert_eq!(all, [5]);
        assert!(published(at, end));
        // Told of memory there, though there is none: asked about again.
        title_memory_allocated(at, PAGE);
        assert!(!published(at, end));
        let (_, all) = ask(fifth);
        assert_eq!(all, Vec::<usize>::new());
        // A reservation of the ordinary kind is no placeholder.
        let plain =
            unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(!placeholder(plain / PAGE));
        look_at_holes();
        assert!(published(at + PAGE, end), "the others are as they were");
        title_memory_remapped(at, BLOCKS * PAGE);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{
        SHARDS, shards, title_memory_reading, title_memory_reading_now, title_memory_reading_range,
        title_memory_remapping, title_memory_remapping_range,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// One of these tests at a time: the first takes every lock, which the
    /// second's "at once" would meet.
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// The locks a range takes: those of its 16 MB spans, ascending, and
    /// all of them for the whole of title memory or a range that wraps.
    #[test]
    fn a_range_takes_the_locks_of_its_spans() {
        let mb = 1 << 20;
        assert_eq!(shards(0, 0).len(), SHARDS);
        assert_eq!(shards(0x1_0000_0000 + 3 * mb, mb), vec![0]);
        assert_eq!(shards(0x1_0000_0000 + 15 * mb, 2 * mb), vec![0, 1]);
        // Spans that wrap past the last lock come back to the first.
        assert_eq!(shards(255 << 24, 32 * mb), vec![0, 255]);
        assert_eq!(shards(0, 256 << 24).len(), SHARDS);
    }

    /// A read in place holds off a change of the addresses it reads, and
    /// only those: a change elsewhere goes on meanwhile.
    #[test]
    fn a_read_holds_off_changes_of_its_own_addresses_only() {
        let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // Addresses no other test takes, far apart in the locks.
        let (here, there) = (0x7_0000_0000 + (40 << 24), 0x7_0000_0000 + (90 << 24));
        let reading = title_memory_reading_range(here, 64 << 10);
        // Elsewhere: at once.
        let elsewhere = std::thread::spawn(move || drop(title_memory_remapping_range(there, 64 << 10)));
        elsewhere.join().unwrap();
        // Here: not until the read lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                let _changing = title_memory_remapping_range(here + (32 << 10), 1);
                changed.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        // A change holds its addresses: a read there cannot be had at once,
        // one elsewhere can.
        let changing = title_memory_remapping_range(here, 1);
        let (asked, answer) = mpsc::channel();
        std::thread::spawn(move || {
            asked
                .send((title_memory_reading_now(here, 1).is_some(), title_memory_reading_now(there, 1).is_some()))
                .unwrap();
        });
        assert_eq!(answer.recv().unwrap(), (false, true));
        drop(changing);
    }
}

#[cfg(test)]
mod wrote_tests {
    use super::{PAGE, ask_once, ask_once_end, asked, writes, wrote};

    #[test]
    fn wrote_keeps_ahead_asked_at_or_above_written_timestamps() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        // SAFETY: private write-watched 64 KiB page owned by this test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);

        ask_once(&[(p, p + PAGE)]);
        wrote(p, PAGE);

        let (now, written) = writes(p, p + PAGE, false, 0).expect("write-watched page");
        let (asked_now, asked_written) = asked(p, p + PAGE, 0).expect("in ask_once range");
        ask_once_end();

        // SAFETY: freeing our private reservation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };

        assert_eq!(now, asked_now);
        let page = p / PAGE;
        let &(_, w) = written
            .iter()
            .find(|&&(pg, _)| pg == page)
            .expect("page recorded by wrote()");
        assert_eq!(asked_written, written);
        assert!(
            now >= w,
            "writes() returned now={now} < written timestamp w={w} after wrote()"
        );
    }
}

#[cfg(test)]
mod journal_tests {
    //! The journal against host copies of title memory kept as
    //! `crate::shadow::Shadow` keeps them. (The epoch tables' tests, which
    //! checked them against the journal of ordered maps the stamps replace.)
    use super::{
        Journal, KERNEL_PAGE, PAGE, ask_ahead_done_in, ask_ahead_in, ask_once_end_in, ask_once_in,
        asked_in, gpu_wrote_in, writes_in, wrote_in,
    };
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, ResetWriteWatch, VirtualAlloc,
        VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Write-watched memory of ours, committed, freed when dropped.
    struct Region(usize, usize);

    impl Region {
        /// Its pages are all there and none counts as written, as in title
        /// memory the title has used: the kernel counts a page's first
        /// touch as a write, even a read's.
        fn new(len: usize) -> Self {
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

        /// The lowest asking a page of `[from, to)` was taken at (0: one
        /// never was): what was written before it the copy holds.
        fn since(&self, from: usize, to: usize) -> u64 {
            let base = self.start / PAGE;
            self.copied[from / PAGE - base..=(to - 1) / PAGE - base]
                .iter()
                .copied()
                .min()
                .unwrap_or(0)
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

    struct Rng(u64);

    impl Rng {
        fn next(&mut self, below: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % below as u64) as usize
        }
    }

    /// Random writes of the title's, writes of the GPU's (unseen by the
    /// kernel, told as `gpu_wrote` does, big ones over whole groups of the
    /// stamps), our own told writes, askings ahead and a submission's one
    /// asking, and refreshes of nested and overlapping copies, half of them
    /// asking only for what was written since their lowest take: every
    /// refresh leaves the copy holding the title's bytes.
    #[test]
    fn refreshes_copy_every_write() {
        const SIZE: usize = 40 << 20;
        let journal = Mutex::new(Journal::new());
        let r = Region::new(SIZE);
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
        let mut copies: Vec<Copy> = shapes
            .iter()
            .map(|&(at, len)| Copy::new(r.0 + at, len))
            .collect();
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut standing = false;
        let (mut stood, mut from_stand, mut gpu) = (0, 0, 0);
        for step in 0..3000 {
            match rng.next(11) {
                // The title writes (no asking may stand: it does not tell).
                0..=2 => {
                    if standing {
                        ask_once_end_in(&journal);
                        standing = false;
                    }
                    for _ in 0..1 + rng.next(8) {
                        let (at, len) = shapes[rng.next(shapes.len())];
                        r.write(at + rng.next(len), rng.next(256) as u8);
                    }
                }
                // A refresh of part of a copy.
                3..=5 => {
                    let i = rng.next(copies.len());
                    let (at, len) = shapes[i];
                    let from = r.0 + at + rng.next(len);
                    let to = from + 1 + rng.next(r.0 + at + len - from);
                    let since = if i.is_multiple_of(2) {
                        copies[i].since(from, to)
                    } else {
                        0
                    };
                    from_stand += usize::from(asked_in(&journal, from, to, since).is_some());
                    let (now, written) =
                        writes_in(&journal, from, to, false, since).expect("watched");
                    assert!(
                        written.iter().all(|w| w.1 > since),
                        "step {step}: since {since}"
                    );
                    copies[i].refresh(from, to, now, &written);
                    assert!(copies[i].holds(from, to), "step {step}: copy {i}");
                }
                // A submission's one asking, over some of the copies.
                6 => {
                    let picked: Vec<(usize, usize)> = shapes
                        .iter()
                        .filter(|_| rng.next(3) != 0)
                        .map(|&(at, len)| (r.0 + at, r.0 + at + len))
                        .collect();
                    ask_once_in(&journal, &picked);
                    standing = journal.lock().unwrap().kept;
                    stood += usize::from(standing);
                }
                // Our own write while it stands, told.
                7 => {
                    let (at, len) = shapes[rng.next(shapes.len())];
                    let offset = at + rng.next(len);
                    let n = 1 + rng.next((at + len - offset).min(3 * PAGE));
                    let value = rng.next(256) as u8;
                    for k in offset..offset + n {
                        r.write(k, value);
                    }
                    wrote_in(&journal, r.0 + offset, n);
                }
                // The GPU writes, unseen by the write watch, and is told.
                8 => {
                    let offset = rng.next(SIZE);
                    let most = [3 * PAGE, 12 << 20][rng.next(2)];
                    let n = 1 + rng.next((SIZE - offset).min(most));
                    let value = rng.next(256) as u8;
                    for k in (offset..offset + n).step_by(1 + rng.next(2 * KERNEL_PAGE)) {
                        r.write(k, value);
                    }
                    // SAFETY: our allocation; what the GPU writes the write
                    // watch does not see.
                    let pages = r.0 + offset / KERNEL_PAGE * KERNEL_PAGE;
                    let end = (r.0 + offset + n).next_multiple_of(KERNEL_PAGE);
                    assert_eq!(
                        unsafe { ResetWriteWatch(pages as *const _, end - pages) },
                        0
                    );
                    gpu_wrote_in(&journal, r.0 + offset, n);
                    // Over a whole group of the stamps (4 MiB, aligned).
                    let group = 64 * PAGE;
                    gpu += usize::from(
                        (r.0 + offset).next_multiple_of(group) + group <= r.0 + offset + n,
                    );
                }
                // An asking ahead for a flush, then its refreshes.
                9 => {
                    let all: Vec<_> = shapes
                        .iter()
                        .map(|&(at, len)| (r.0 + at, r.0 + at + len))
                        .collect();
                    ask_ahead_in(&journal, &all);
                    for (i, &(from, to)) in all.iter().enumerate() {
                        let since = if i.is_multiple_of(2) {
                            copies[i].since(from, to)
                        } else {
                            0
                        };
                        let (now, written) =
                            writes_in(&journal, from, to, false, since).expect("watched");
                        copies[i].refresh(from, to, now, &written);
                        assert!(copies[i].holds(from, to), "step {step}: copy {i} ahead");
                    }
                    ask_ahead_done_in(&journal);
                }
                _ => {
                    ask_once_end_in(&journal);
                    standing = false;
                }
            }
        }
        assert!(
            stood > 100 && from_stand > 100,
            "the asking that stands was used: {stood}, {from_stand}"
        );
        assert!(gpu > 10, "big writes of the GPU: {gpu}");
    }

    /// Threads asking about overlapping copies while the title writes, one
    /// of them asking ahead: once the writes stop, every copy brought up to
    /// date holds the title's bytes.
    #[test]
    fn concurrent_askers_miss_no_write() {
        const SIZE: usize = 64 * PAGE;
        const THREADS: usize = 4;
        let journal: &'static Mutex<Journal> = Box::leak(Box::new(Mutex::new(Journal::new())));
        let r = std::sync::Arc::new(Region::new(SIZE));
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let still = std::sync::Arc::new(std::sync::Barrier::new(THREADS + 1));
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
                            ask_ahead_in(journal, &all);
                        }
                        let since = copy.since(start, end);
                        let (now, written) =
                            writes_in(journal, start, end, false, since).expect("watched");
                        copy.refresh(start, end, now, &written);
                        if n == 0 {
                            ask_ahead_done_in(journal);
                        }
                    }
                    still.wait();
                    let since = copy.since(start, end);
                    let (now, written) =
                        writes_in(journal, start, end, false, since).expect("watched");
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
