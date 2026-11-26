//! Pages of memory we own, cut into pieces: the card's memory for the host
//! copies of textures the title fills ([`TEXTURES`]), and the PC's upload
//! memory for their staging buffers (`crate::texture_pool`, [`STAGING`]).
//!
//! A texture page is a DEFAULT heap of [`PAGE`] bytes that takes only
//! textures that are neither targets nor depth
//! (`ALLOW_ONLY_NON_RT_DS_TEXTURES`); each texture is placed in a piece of
//! one (`CreatePlacedResource`). A texture larger than a page gets a heap of
//! its own size, let go when its piece comes back. Nothing here is ever a
//! tiled or reserved resource.
//!
//! cool_fox measured an allocation at about 310 microseconds whatever its
//! size, on one thread, and a texture placed in a page at 6: its pass over
//! the textures shaders pick by index got through 350-500 of the 24,000
//! queued when the world loads in each of its 250 ms turns, some forty
//! seconds of a world without its textures.
//!
//! **Ripe pages.** cool_fox lost the device to page faults of lists already
//! on the card, in a texture made the same tenth of a second as the first
//! piece of a page made for it: the title gives a slot that those lists read
//! by index a new texture. Why the card does not have the new page's
//! addresses yet is not known. A page that every submission begun before its
//! making has left behind ([`Ripe`]) has no such lists, and a piece is only
//! ever cut from such a page: with none that has room, the texture is not
//! placed (the caller makes it an allocation of its own, as with the pages
//! off). Pages are made ahead so that one is ripe when it is wanted: the
//! spare pages, empty ones kept, are those made ahead, [`AHEAD`] while the
//! card has room and [`AHEAD_PRESSED`] when it has little ([`pressed`]).
//! One count for both, so a page made ahead is never one let go as a spare
//! too many, and made again.
//!
//! **Rest.** A piece given back is used again only [`REST`] later (nothing
//! the card was told before that is still running: Windows resets a card
//! that takes that long over one submission), and by then no list names it,
//! as for allocations of their own: a texture's piece comes back when the
//! last reference to the host texture goes (an object of ours the texture
//! holds says so, `Reclaim`), a staging buffer's when its `crate::texture`
//! side is dropped and the submissions running then have finished
//! (`crate::host::retire`).
//!
//! **Commit.** Only letting a heap go returns the card's memory and the
//! process's commit: an empty page past the spare count goes. To empty
//! pages, [`Pages::compaction`] picks the emptiest and says where each of
//! their live pieces could go instead; moving the textures is the budget
//! manager's (change 2, T7).
//!
//! Ported from cool_fox (`pool.rs`), which this follows in its pages,
//! pieces, rest, pages made ahead and the back-off after a page the host
//! refused ([`Pages::grow`]).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

/// Bytes of a page.
pub(crate) const PAGE: u64 = 64 << 20;
/// Spare texture pages, which are those made ahead, while the card has room
/// ([`pressed`]).
pub(crate) const AHEAD: usize = 12;
/// The same when the card has little room.
pub(crate) const AHEAD_PRESSED: usize = 2;
/// Milliseconds until a piece given back is used again.
pub(crate) const REST: u64 = 2500;
/// Milliseconds no page is made after the host refused one ([`Pages::grow`]).
const BACK_OFF: u64 = 500;
/// The card has little room from this share of its budget in use (%).
const PRESSED_PERCENT: u128 = 85;
/// The size of a heap of its own is a multiple of this.
const OWN_GRAIN: u64 = 64 << 10;

/// Whether the card has little room, by its memory (budget and use, as
/// `crate::host::Host::video_memory` says): from [`PRESSED_PERCENT`] of the
/// budget in use, or when that is not known.
pub(crate) fn pressed(memory: Option<[u64; 4]>) -> bool {
    !matches!(memory, Some([budget, used, ..])
        if u128::from(used) * 100 < u128::from(budget) * PRESSED_PERCENT)
}

/// How many texture pages are made ahead and kept empty.
pub(crate) fn ahead(pressed: bool) -> usize {
    if pressed { AHEAD_PRESSED } else { AHEAD }
}

/// The free ranges of a page: start and length, never two that touch.
#[derive(Clone)]
pub(crate) struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    pub(crate) fn new(size: u64) -> Ranges {
        Ranges(BTreeMap::from([(0, size)]))
    }

    fn find(&self, len: u64, align: u64) -> Option<(u64, u64, u64)> {
        self.0.iter().find_map(|(&start, &size)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + size).then_some((start, size, at))
        })
    }

    fn fits(&self, len: u64, align: u64) -> bool {
        self.find(len, align).is_some()
    }

    /// The lowest place for `len` bytes at a multiple of `align`.
    pub(crate) fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.find(len, align)?;
        self.cut(start, size, at, len);
        Some(at)
    }

    /// Takes `len` bytes at `at`, if all of them are free.
    #[cfg_attr(not(test), allow(dead_code))]
    fn take_at(&mut self, at: u64, len: u64) -> bool {
        let Some((&start, &size)) = self.0.range(..=at).next_back() else {
            return false;
        };
        if at + len > start + size {
            return false;
        }
        self.cut(start, size, at, len);
        true
    }

    fn cut(&mut self, start: u64, size: u64, at: u64, len: u64) {
        self.0.remove(&start);
        if at > start {
            self.0.insert(start, at - start);
        }
        if at + len < start + size {
            self.0.insert(at + len, start + size - at - len);
        }
    }

    pub(crate) fn give(&mut self, mut start: u64, mut len: u64) {
        if let Some((&before, &size)) = self.0.range(..start).next_back()
            && before + size == start
        {
            self.0.remove(&before);
            (start, len) = (before, len + size);
        }
        if let Some(size) = self.0.remove(&(start + len)) {
            len += size;
        }
        self.0.insert(start, len);
    }

    pub(crate) fn whole(&self, size: u64) -> bool {
        self.0.get(&0) == Some(&size)
    }

    /// Whether `at` lies in a free range.
    #[cfg(test)]
    pub(crate) fn free_at(&self, at: u64) -> bool {
        self.0
            .range(..=at)
            .next_back()
            .is_some_and(|(&start, &len)| at < start + len)
    }
}

/// A part of a page.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    pub(crate) page: usize,
    pub(crate) start: u64,
    pub(crate) len: u64,
}

/// A live piece, and where it could lie instead ([`Pages::compaction`]).
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Move {
    pub(crate) from: Piece,
    pub(crate) to: Piece,
}

/// Submissions begun and finished (`crate::host::submissions`): a page made
/// while submission `n` was the newest begun is ripe once `n` has finished.
#[derive(Clone, Copy)]
pub(crate) struct Ripe {
    pub(crate) started: u64,
    pub(crate) finished: u64,
}

impl Ripe {
    /// Every page is ripe.
    pub(crate) const ALL: Ripe = Ripe {
        started: 0,
        finished: u64::MAX,
    };

    pub(crate) fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        Ripe { started, finished }
    }

    fn is<B>(self, page: &Page<B>) -> bool {
        page.made <= self.finished
    }
}

/// How a set of pages is used.
#[derive(Clone, Copy)]
pub(crate) struct Rule {
    /// Pages are made ahead, and a piece is cut only from a ripe one.
    ahead: bool,
    /// The largest piece cut from a page.
    largest: u64,
    /// Larger pieces get a page of their own size (else none).
    own: bool,
}

/// Textures: pages made ahead, ripe ones only, a page of its own for a
/// texture larger than one.
pub(crate) const TEXTURES: Rule = Rule {
    ahead: true,
    largest: PAGE,
    own: true,
};

/// Staging buffers: any page, pieces of half a page at most.
pub(crate) const STAGING: Rule = Rule {
    ahead: false,
    largest: PAGE / 2,
    own: false,
};

struct Page<B> {
    memory: B,
    size: u64,
    /// A heap of its own size for one piece, let go as soon as it is empty.
    own: bool,
    free: Ranges,
    /// The pieces taken and not given back: start, then length and
    /// alignment.
    live: BTreeMap<u64, (u64, u64)>,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
}

impl<B> Page<B> {
    fn whole(&self) -> bool {
        self.free.whole(self.size)
    }

    fn live_bytes(&self) -> u64 {
        self.live.values().map(|&(len, _)| len).sum()
    }

    /// A shared page that is empty: a spare one.
    fn spare(&self) -> bool {
        !self.own && self.whole()
    }
}

/// Pages and the pieces cut from them.
pub(crate) struct Pages<B> {
    rule: Rule,
    /// `None`: a page let go, its place to be used again.
    pages: Vec<Option<Page<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty shared pages kept (and, with [`Rule::ahead`], made ahead).
    spare: usize,
    /// The spare count fell since the last look for empty pages.
    fewer: bool,
    /// No page is made before this (ms): the host refused the last one.
    retry_growth_at: u64,
}

impl<B: Clone> Pages<B> {
    pub(crate) const fn new(rule: Rule, spare: usize) -> Pages<B> {
        Pages {
            rule,
            pages: Vec::new(),
            resting: VecDeque::new(),
            spare,
            fewer: false,
            retry_growth_at: 0,
        }
    }

    /// Sets how many empty shared pages are kept; those past it go at the
    /// next look ([`Pages::reclaim`]).
    pub(crate) fn set_spare(&mut self, spare: usize) {
        self.fewer |= spare < self.spare;
        self.spare = spare;
    }

    /// A piece of `len` bytes at a multiple of `align`, the memory it lies
    /// in and whether that part was ever used before; `make` makes a page of
    /// the size it is given. `None`: too big, no page to be had, or (with
    /// [`Rule::ahead`]) no ripe page with room.
    pub(crate) fn take(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        mut make: impl FnMut(u64) -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        if len == 0 {
            return None;
        }
        self.rested(now);
        if len > self.rule.largest {
            if !self.rule.own {
                return None;
            }
            // shortcut: a heap of its own is made for the texture at once,
            // so it is not ripe: it is what the texture had before pages (a
            // committed resource is a heap of its own size too).
            let size = len.next_multiple_of(OWN_GRAIN);
            let memory = self.grow(now, size, &mut make)?;
            let i = self.add(memory, size, true, ripe.started);
            return self.cut(i, len, align);
        }
        let i = match self.room(len, align, ripe) {
            Some(i) => i,
            None if self.rule.ahead => {
                // One made now is ripe at once if nothing runs.
                self.make_ahead(now, ripe, &mut make);
                self.room(len, align, ripe)?
            }
            None => {
                let memory = self.grow(now, PAGE, &mut make)?;
                self.add(memory, PAGE, false, ripe.started)
            }
        };
        let taken = self.cut(i, len, align);
        if self.rule.ahead {
            self.make_ahead(now, ripe, &mut make);
        }
        taken
    }

    /// The page to cut `len` bytes from: one in use before an empty one
    /// (which stays whole, to be let go), the oldest empty one first.
    fn room(&self, len: u64, align: u64, ripe: Ripe) -> Option<usize> {
        let usable = |p: &Page<B>| !p.own && (!self.rule.ahead || ripe.is(p));
        let mut empty: Option<(u64, usize)> = None;
        for (i, page) in self.pages.iter().enumerate() {
            let Some(page) = page.as_ref().filter(|p| usable(p)) else {
                continue;
            };
            if page.whole() {
                if len <= page.size && empty.is_none_or(|(made, _)| page.made < made) {
                    empty = Some((page.made, i));
                }
            } else if page.free.fits(len, align) {
                return Some(i);
            }
        }
        empty.map(|(_, i)| i)
    }

    fn cut(&mut self, i: usize, len: u64, align: u64) -> Option<(Piece, B, bool)> {
        let page = self.pages[i].as_mut()?;
        let start = page.free.take(len, align)?;
        page.live.insert(start, (len, align));
        let again = start < page.used_to;
        page.used_to = page.used_to.max(start + len);
        Some((
            Piece {
                page: i,
                start,
                len,
            },
            page.memory.clone(),
            again,
        ))
    }

    /// Makes one page when fewer than the spare count are empty.
    fn make_ahead(&mut self, now: u64, ripe: Ripe, make: &mut impl FnMut(u64) -> Option<B>) {
        let empty = self.pages.iter().flatten().filter(|p| p.spare()).count();
        if empty < self.spare
            && let Some(memory) = self.grow(now, PAGE, make)
        {
            self.add(memory, PAGE, false, ripe.started);
        }
    }

    /// A new page of `size` bytes from `make`, unless the host refused one
    /// less than [`BACK_OFF`] ago: after a refusal (out of memory) the next
    /// asks wait that long, so that a burst of small uploads is not a burst
    /// of failing 64 MiB allocations.
    fn grow(&mut self, now: u64, size: u64, make: &mut impl FnMut(u64) -> Option<B>) -> Option<B> {
        if now < self.retry_growth_at {
            return None;
        }
        let page = make(size);
        if page.is_none() {
            self.retry_growth_at = now.saturating_add(BACK_OFF);
        }
        page
    }

    fn add(&mut self, memory: B, size: u64, own: bool, made: u64) -> usize {
        let i = match self.pages.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.pages.push(None);
                self.pages.len() - 1
            }
        };
        self.pages[i] = Some(Page {
            memory,
            size,
            own,
            free: Ranges::new(size),
            live: BTreeMap::new(),
            used_to: 0,
            made,
        });
        i
    }

    /// Gives `piece` back at `now`: free again [`REST`] later.
    pub(crate) fn give(&mut self, piece: Piece, now: u64) {
        if let Some(page) = self.pages[piece.page].as_mut() {
            page.live.remove(&piece.start);
        }
        self.resting.push_back((now + REST, piece));
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// pages past those kept.
    pub(crate) fn rested(&mut self, now: u64) {
        self.reclaim(now, false);
    }

    /// [`Pages::rested`]; `sweep`: looks for empty pages past the spare
    /// count even if no piece came free.
    pub(crate) fn reclaim(&mut self, now: u64, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(page) = self.pages[piece.page].as_mut() {
                page.free.give(piece.start, piece.len);
                any |= page.whole();
            }
        }
        if !(sweep || any || std::mem::take(&mut self.fewer)) {
            return;
        }
        // The newest empty pages go first: those kept are the ripest.
        let mut empty: Vec<(u64, usize)> = Vec::new();
        for (i, slot) in self.pages.iter_mut().enumerate() {
            match slot {
                Some(page) if page.own && page.whole() => *slot = None,
                Some(page) if page.spare() => empty.push((page.made, i)),
                _ => {}
            }
        }
        if empty.len() > self.spare {
            empty.sort_unstable_by(|a, b| b.cmp(a));
            for &(_, i) in &empty[..empty.len() - self.spare] {
                self.pages[i] = None;
            }
        }
    }

    /// The pages there are.
    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.pages.iter().flatten().count()
    }

    /// The bytes of the pages there are.
    pub(crate) fn bytes(&self) -> u64 {
        self.pages.iter().flatten().map(|p| p.size).sum()
    }

    /// The memory of page `i`, if it is there.
    #[cfg(test)]
    pub(crate) fn memory(&self, i: usize) -> Option<&B> {
        Some(&self.pages.get(i)?.as_ref()?.memory)
    }

    /// The first page whose memory `is` says yes to.
    #[cfg(test)]
    pub(crate) fn position(&self, is: impl Fn(&B) -> bool) -> Option<usize> {
        self.pages
            .iter()
            .position(|p| p.as_ref().is_some_and(|p| is(&p.memory)))
    }

    /// Whether the piece at `start` in page `i` is no holder's (resting or
    /// free again).
    #[cfg(test)]
    pub(crate) fn given(&self, i: usize, start: u64) -> bool {
        self.resting
            .iter()
            .any(|&(_, p)| p.page == i && p.start == start)
            || self.pages[i]
                .as_ref()
                .is_some_and(|p| p.free.free_at(start))
    }

    /// How to empty up to `most` of the emptiest shared pages: for each of
    /// their live pieces, a free place of the same length and alignment in
    /// a ripe page that holds live pieces and stays. A page is chosen only
    /// if all its live pieces have such a place; it is empty once they have
    /// moved there and its resting pieces have rested. Nothing is taken:
    /// the mover takes each place with [`Pages::claim`] and gives the old
    /// piece back once the texture has moved (the budget manager, which is
    /// to come).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn compaction(&self, most: usize, ripe: Ripe) -> Vec<Move> {
        let holds = |p: &Page<B>| !p.own && !p.live.is_empty() && ripe.is(p);
        let mut emptiest: Vec<(u64, usize)> = self
            .pages
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.as_ref().filter(|p| holds(p)).map(|p| (p.live_bytes(), i)))
            .collect();
        emptiest.sort_unstable();
        let mut free: HashMap<usize, Ranges> = HashMap::new();
        let mut chosen: HashSet<usize> = HashSet::new();
        let mut taking: HashSet<usize> = HashSet::new();
        let mut moves = Vec::new();
        for &(_, from) in &emptiest {
            if chosen.len() >= most {
                break;
            }
            if taking.contains(&from) {
                continue;
            }
            let Some(page) = self.pages[from].as_ref() else {
                continue;
            };
            let mut trial = free.clone();
            let mut planned = Vec::new();
            let fits = page.live.iter().all(|(&start, &(len, align))| {
                let place = emptiest.iter().find_map(|&(_, to)| {
                    if to == from || chosen.contains(&to) {
                        return None;
                    }
                    let ranges = trial.entry(to).or_insert_with(|| {
                        let page = self.pages[to].as_ref();
                        page.map_or_else(|| Ranges(BTreeMap::new()), |p| p.free.clone())
                    });
                    Some((to, ranges.take(len, align)?))
                });
                place
                    .inspect(|&(to, at)| {
                        planned.push(Move {
                            from: Piece {
                                page: from,
                                start,
                                len,
                            },
                            to: Piece {
                                page: to,
                                start: at,
                                len,
                            },
                        });
                    })
                    .is_some()
            });
            if fits {
                free = trial;
                chosen.insert(from);
                taking.extend(planned.iter().map(|m| m.to.page));
                moves.extend(planned);
            }
        }
        moves
    }

    /// Takes exactly the place `to` ([`Pages::compaction`]) at alignment
    /// `align`, if it is still free, and the memory it lies in.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn claim(&mut self, to: Piece, align: u64) -> Option<B> {
        let page = self.pages.get_mut(to.page)?.as_mut()?;
        if !to.start.is_multiple_of(align.max(1)) || !page.free.take_at(to.start, to.len) {
            return None;
        }
        page.live.insert(to.start, (to.len, align));
        page.used_to = page.used_to.max(to.start + to.len);
        Some(page.memory.clone())
    }
}

pub(crate) use d3d::{first_write, in_page, place, reclaim, room_left};

/// The texture pages' D3D12 side.
mod d3d {
    use std::ffi::c_void;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::{GUID, HRESULT, IUnknown, Interface};

    use super::{AHEAD_PRESSED, PAGE, Pages, Piece, Ripe, TEXTURES, ahead};

    fn now() -> u64 {
        crate::host::now_ms()
    }

    /// Until the card's room is known, as when it has little.
    pub(super) static PAGES: Mutex<Pages<ID3D12Heap>> =
        Mutex::new(Pages::new(TEXTURES, AHEAD_PRESSED));

    /// The spare count [`reclaim`] last set, which [`pages`] gives [`PAGES`]
    /// each time it is taken: a look that found the pages busy loses
    /// nothing.
    static SPARE: AtomicUsize = AtomicUsize::new(AHEAD_PRESSED);

    fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Pieces of textures that went, and when, not yet in [`PAGES`]: a
    /// texture gives its piece back from inside the runtime's destruction of
    /// it (`release`), which may hold locks of the runtime's own, while
    /// [`PAGES`] is held across the making of a page (`CreateHeap`). Taking
    /// only this, never held across a call to the runtime, keeps the two
    /// from waiting on each other.
    static GIVEN: Mutex<Vec<(u64, Piece)>> = Mutex::new(Vec::new());

    /// [`PAGES`], with the pieces given back meanwhile in it and the spare
    /// count of the last look ([`SPARE`]).
    fn pages() -> std::sync::MutexGuard<'static, Pages<ID3D12Heap>> {
        let mut pages = lock(&PAGES);
        given_into(&mut pages);
        pages.set_spare(SPARE.load(Ordering::Relaxed));
        pages
    }

    fn given_into(pages: &mut Pages<ID3D12Heap>) {
        for (at, piece) in std::mem::take(&mut *lock(&GIVEN)) {
            pages.give(piece, at);
        }
    }

    /// Sets the spare count by the card's room (`pressed`), and lets go of
    /// empty pages past it. Only that sweep is passed over while the pages
    /// are busy: the count is applied at the next piece ([`pages`]), which
    /// lets the empty pages past it go too.
    pub(crate) fn reclaim(pressed: bool) {
        SPARE.store(ahead(pressed), Ordering::Relaxed);
        if let Ok(mut pages) = PAGES.try_lock() {
            given_into(&mut pages);
            pages.set_spare(SPARE.load(Ordering::Relaxed));
            pages.reclaim(now(), true);
        }
    }

    /// Bytes of the pieces textures hold now.
    static HELD: AtomicU64 = AtomicU64::new(0);

    /// Textures not placed in a page (no ripe page with room, or the page
    /// was refused).
    static UNRIPE: AtomicU64 = AtomicU64::new(0);

    /// Bytes of the texture pages no texture holds: memory of the card in
    /// use by us that new textures take first.
    pub(crate) fn room_left() -> u64 {
        lock(&PAGES)
            .bytes()
            .saturating_sub(HELD.load(Ordering::Relaxed))
    }

    /// An object the host texture holds and releases when it goes: its
    /// piece is free then.
    #[repr(C)]
    struct Reclaim {
        methods: &'static [*const (); 3],
        references: AtomicU32,
        piece: Piece,
        /// The texture's address, while it is in [`UNTOUCHED`].
        texture: usize,
    }

    /// Placed textures the card has neither uploaded into nor written by a
    /// copy (by address): what their piece held before still shows in them
    /// ([`first_write`]).
    static UNTOUCHED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

    /// Whether `texture` is a placed one the card writes for the first
    /// time, which it no longer is after this. The caller records the
    /// aliasing barrier and gives every texel a value before anything else
    /// uses it.
    pub(crate) fn first_write(texture: &ID3D12Resource) -> bool {
        let mut set = lock(&UNTOUCHED);
        set.as_mut()
            .is_some_and(|set| set.remove(&(texture.as_raw() as usize)))
    }

    struct Methods([*const (); 3]);
    // SAFETY: function addresses only.
    unsafe impl Sync for Methods {}

    static RECLAIM: Methods = Methods([
        query as *const (),
        add_ref as *const (),
        release as *const (),
    ]);

    unsafe extern "system" fn query(
        this: *mut c_void,
        iid: *const GUID,
        out: *mut *mut c_void,
    ) -> HRESULT {
        const E_NOINTERFACE: HRESULT = HRESULT(0x8000_4002_u32 as i32);
        const E_POINTER: HRESULT = HRESULT(0x8000_4003_u32 as i32);
        // SAFETY: COM's arguments.
        unsafe {
            if out.is_null() {
                return E_POINTER;
            }
            if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
                add_ref(this);
                *out = this;
                return HRESULT(0);
            }
            *out = std::ptr::null_mut();
        }
        E_NOINTERFACE
    }

    unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
        // SAFETY: one of ours, alive while a reference to it is.
        unsafe {
            (*this.cast::<Reclaim>())
                .references
                .fetch_add(1, Ordering::Relaxed)
                + 1
        }
    }

    unsafe extern "system" fn release(this: *mut c_void) -> u32 {
        // SAFETY: as above; the last reference frees the box it was made as.
        unsafe {
            let left = (*this.cast::<Reclaim>())
                .references
                .fetch_sub(1, Ordering::AcqRel)
                - 1;
            if left == 0 {
                let gone = Box::from_raw(this.cast::<Reclaim>());
                // Before the texture's address can be another's.
                if let Some(set) = lock(&UNTOUCHED).as_mut() {
                    set.remove(&gone.texture);
                }
                HELD.fetch_sub(gone.piece.len, Ordering::Relaxed);
                lock(&GIVEN).push((now(), gone.piece));
            }
            left
        }
    }

    /// The private data key of [`Reclaim`].
    const KEY: GUID = GUID::from_u128(0x6f0e2f0c_5a59_4b1e_9a3f_7f4d2c9be713);

    /// Whether `texture` was placed in a page.
    pub(crate) fn in_page(texture: &ID3D12Resource) -> bool {
        let mut size = 0u32;
        // SAFETY: a size query of the texture's private data.
        unsafe { texture.GetPrivateData(&KEY, &mut size, None) }.is_ok() && size != 0
    }

    /// Makes `texture` give `piece` back when the last reference to it goes.
    fn give_back_with(texture: &ID3D12Resource, piece: Piece) {
        let made = Box::into_raw(Box::new(Reclaim {
            methods: &RECLAIM.0,
            references: AtomicU32::new(1),
            piece,
            texture: texture.as_raw() as usize,
        }));
        // SAFETY: a COM object of ours with one reference, which this takes
        // and drops at the end; the texture takes its own.
        unsafe {
            let object = IUnknown::from_raw(made.cast());
            if texture.SetPrivateDataInterface(&KEY, &object).is_err() {
                // Never given back: the piece stays taken, nothing is reused
                // under a texture that may be alive.
                std::mem::forget(object);
                return;
            }
        }
        // (Taken out again when it goes, which only a held object says.)
        lock(&UNTOUCHED)
            .get_or_insert_with(Default::default)
            .insert(texture.as_raw() as usize);
    }

    /// A texture page (or a heap of its own) of `size` bytes.
    pub(super) fn make(device: &ID3D12Device, size: u64) -> Option<ID3D12Heap> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: size,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation.
        if let Err(e) = unsafe { device.CreateHeap(&desc, &mut heap) } {
            tracing::warn!(error = %e, size, "no page of texture memory");
        }
        if let Some(heap) = &heap {
            crate::host::name(heap, || {
                if size == PAGE {
                    "textures the title fills (a page)".into()
                } else {
                    "a texture the title fills (a heap of its own)".into()
                }
            });
        }
        heap
    }

    /// Where `desc` lies in a page: the description with its alignment, and
    /// the piece's length and alignment. Small textures lie at multiples of
    /// 4 KiB where the host allows it (the top level is 64 KiB at most),
    /// 64 KiB each otherwise.
    ///
    /// # Safety
    /// `desc` is a valid description.
    pub(super) unsafe fn layout(
        device: &ID3D12Device,
        desc: *const D3D12_RESOURCE_DESC,
    ) -> Option<(D3D12_RESOURCE_DESC, u64, u64)> {
        // SAFETY: the caller's description.
        let mut d = unsafe { *desc };
        d.Alignment = u64::from(D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT);
        // SAFETY: plain queries.
        let mut info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
        if info.SizeInBytes == u64::MAX || info.Alignment != d.Alignment {
            d.Alignment = 0;
            // SAFETY: as above.
            info = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
        }
        (info.SizeInBytes != u64::MAX).then_some((d, info.SizeInBytes, info.Alignment.max(1)))
    }

    /// A texture of `desc` (no target, depth or unordered access) placed in
    /// a page, in `state`; `None`: not placed (no ripe page with room, or
    /// the host refused).
    ///
    /// # Safety
    /// `desc` is a valid description.
    pub(crate) unsafe fn place(
        device: &ID3D12Device,
        desc: *const D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        // SAFETY: the caller's description.
        unsafe { place_with(device, desc, state, Ripe::now()) }
    }

    /// [`place`] with the pages `ripe` says are ripe.
    ///
    /// # Safety
    /// As for [`place`].
    unsafe fn place_with(
        device: &ID3D12Device,
        desc: *const D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
        ripe: Ripe,
    ) -> Option<ID3D12Resource> {
        // SAFETY: the caller's description.
        let (d, len, align) = unsafe { layout(device, desc) }?;
        let Some((piece, heap, _)) =
            pages().take(len, align, now(), ripe, |size| make(device, size))
        else {
            // shortcut: with no ripe page that has room, the caller makes
            // the texture a separate committed allocation (as with the pages
            // off), in place of waiting for a page to ripen or starting the
            // texture with its tail only (the budget manager's, T7). A wait
            // here would hold the title's thread on submissions that may
            // need it.
            if len <= PAGE {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        "textures not placed in a page (no ripe page with room, or the page was refused), each an allocation of its own: {n}"
                    );
                }
            }
            return None;
        };
        let mut made: Option<ID3D12Resource> = None;
        // SAFETY: the piece is no one else's; the heap lives while a piece
        // of it is taken (the texture holds it).
        let result =
            unsafe { device.CreatePlacedResource(&heap, piece.start, &d, state, None, &mut made) };
        match (result, made) {
            (Ok(()), Some(made)) => {
                HELD.fetch_add(piece.len, Ordering::Relaxed);
                give_back_with(&made, piece);
                Some(made)
            }
            _ => {
                lock(&PAGES).give(piece, now());
                None
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::super::REST;
        use super::*;

        fn rgba(width: u64, height: u32) -> D3D12_RESOURCE_DESC {
            D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: width,
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            }
        }

        /// The process's commit (`PagefileUsage`).
        fn commit() -> u64 {
            use windows_sys::Win32::System::ProcessStatus::{
                K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let mut c = PROCESS_MEMORY_COUNTERS {
                cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
                ..Default::default()
            };
            // SAFETY: our own process and a sized structure.
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
            c.PagefileUsage as u64
        }

        /// A placed texture's piece comes back when its last reference
        /// goes, not before.
        #[test]
        fn warp_a_texture_gives_its_piece_back_when_it_goes() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let desc = rgba(256, 256);
            let state = D3D12_RESOURCE_STATE_COMMON;
            // SAFETY: a valid description.
            let t = unsafe { place_with(&host.device, &desc, state, Ripe::ALL) }
                .expect("a texture placed in a ripe page");
            assert!(in_page(&t));
            assert_eq!(
                crate::host::references(&t),
                1,
                "what the texture holds is no reference to it"
            );
            let resting = || pages().resting.len();
            let before = resting();
            let other = t.clone();
            drop(t);
            assert_eq!(resting(), before);
            drop(other);
            assert_eq!(resting(), before + 1);
            // SAFETY: a plain query.
            assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        }

        /// Placed textures in `pages`: `count` of `desc`, all ripe.
        fn fill(
            device: &ID3D12Device,
            pages: &mut Pages<ID3D12Heap>,
            desc: &D3D12_RESOURCE_DESC,
            count: usize,
        ) -> Vec<(Piece, ID3D12Resource)> {
            // SAFETY: a valid description.
            let (d, len, align) = unsafe { layout(device, desc) }.unwrap();
            (0..count)
                .map(|_| {
                    let (piece, heap, _) = pages
                        .take(len, align, 0, Ripe::ALL, |size| make(device, size))
                        .unwrap();
                    let mut t: Option<ID3D12Resource> = None;
                    // SAFETY: the piece is ours alone, in a live heap.
                    unsafe {
                        device.CreatePlacedResource(
                            &heap,
                            piece.start,
                            &d,
                            D3D12_RESOURCE_STATE_COMMON,
                            None,
                            &mut t,
                        )
                    }
                    .unwrap();
                    (piece, t.unwrap())
                })
                .collect()
        }

        /// Pages made and filled: `N` full ones (four 16 MiB textures each)
        /// and the spares made ahead meanwhile.
        const N: usize = 6;

        fn full_pages(device: &ID3D12Device) -> (Pages<ID3D12Heap>, Vec<(Piece, ID3D12Resource)>) {
            let mut pages: Pages<ID3D12Heap> = Pages::new(TEXTURES, N);
            let textures = fill(device, &mut pages, &rgba(2048, 2048), 4 * N);
            assert!(pages.count() >= N, "{} pages", pages.count());
            (pages, textures)
        }

        /// The commit after pages were made and let go once: WARP commits
        /// some memory of its own for its first ones and keeps it.
        fn warm(device: &ID3D12Device) -> u64 {
            let (pages, textures) = full_pages(device);
            let_go(pages, textures);
            commit()
        }

        /// Empties `pages` and lets every one go.
        fn let_go(mut pages: Pages<ID3D12Heap>, textures: Vec<(Piece, ID3D12Resource)>) {
            let given: Vec<Piece> = textures.iter().map(|&(p, _)| p).collect();
            drop(textures);
            for piece in given {
                pages.give(piece, 0);
            }
            pages.set_spare(0);
            pages.reclaim(REST, true);
            assert_eq!(pages.count(), 0);
        }

        /// Pages made, filled, emptied and let go give the process's commit
        /// back: at least a page's bytes for each full one, and all that
        /// their making took. (WARP commits an empty heap only in part.)
        #[test]
        fn warp_pages_let_go_return_their_commit() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let start = warm(&host.device);
            let (pages, textures) = full_pages(&host.device);
            let full = commit();
            let_go(pages, textures);
            let end = commit();
            let back = full.saturating_sub(end);
            assert!(
                back >= N as u64 * PAGE * 9 / 10,
                "{} MiB of commit back for {N} full pages",
                back >> 20
            );
            assert!(
                end <= start + PAGE / 4,
                "{} MiB kept",
                end.saturating_sub(start) >> 20
            );
            // SAFETY: a plain query.
            assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        }

        /// WARP only: evicting full pages does not give their commit back.
        /// This documents the fact measured on the card and proves nothing
        /// about it (WARP's heaps are the PC's memory).
        #[test]
        fn warp_only_evict_alone_keeps_the_commit() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let device = &host.device;
            let start = warm(device);
            let (pages, textures) = full_pages(device);
            let pageables: Vec<Option<ID3D12Pageable>> = pages
                .pages
                .iter()
                .flatten()
                .map(|p| p.memory.cast().ok())
                .collect();
            let before = commit();
            // SAFETY: live heaps of ours that no work uses.
            unsafe { device.Evict(&pageables) }.unwrap();
            let after = commit();
            assert!(
                before.saturating_sub(after) < PAGE,
                "evicting {N} full pages gave {} MiB of commit back",
                before.saturating_sub(after) >> 20
            );
            // SAFETY: as above.
            unsafe { device.MakeResident(&pageables) }.unwrap();
            drop(pageables);
            // Letting them go does.
            let_go(pages, textures);
            assert!(commit() <= start + PAGE / 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted() -> (
        std::cell::Cell<u32>,
        impl Fn(&std::cell::Cell<u32>) -> Option<u32>,
    ) {
        (std::cell::Cell::new(0), |made: &std::cell::Cell<u32>| {
            made.set(made.get() + 1);
            Some(made.get())
        })
    }

    #[test]
    fn ranges_are_cut_aligned_and_join_again() {
        let mut r = Ranges::new(1024);
        assert_eq!(r.take(100, 64), Some(0));
        assert_eq!(r.take(100, 64), Some(128));
        assert_eq!(r.take(28, 4), Some(100));
        assert_eq!(r.take(2000, 4), None);
        r.give(128, 100);
        r.give(0, 100);
        assert!(!r.whole(1024));
        r.give(100, 28);
        assert!(r.whole(1024));
        assert!(r.take_at(512, 64));
        assert!(!r.take_at(500, 64));
        assert!(!r.free_at(512));
        assert!(r.free_at(576));
    }

    #[test]
    fn the_card_is_pressed_from_85_percent_of_its_budget() {
        assert!(!pressed(Some([100, 84, 0, 0])));
        assert!(pressed(Some([100, 85, 0, 0])));
        assert!(pressed(Some([0, 0, 0, 0])));
        assert!(pressed(None));
        assert_eq!(ahead(false), AHEAD);
        assert_eq!(ahead(true), AHEAD_PRESSED);
    }

    #[test]
    fn a_failed_page_is_not_retried_for_each_small_upload() {
        let mut pages: Pages<u32> = Pages::new(STAGING, 2);
        let calls = std::cell::Cell::new(0);
        let refuse = |_| {
            calls.set(calls.get() + 1);
            None
        };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pages.take(size, 512, 12, Ripe::ALL, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        let one = |_| Some(1);
        assert!(
            pages
                .take(24080, 512, 12 + BACK_OFF - 1, Ripe::ALL, one)
                .is_none()
        );
        assert!(
            pages
                .take(24080, 512, 12 + BACK_OFF, Ripe::ALL, one)
                .is_some()
        );
    }

    #[test]
    fn a_reclaim_lets_idle_pages_go_without_reusing_live_or_resting_pieces() {
        let mut pages = Pages::new(STAGING, 8);
        let (held, _, _) = pages.take(PAGE / 2, 1, 0, Ripe::ALL, |_| Some(1)).unwrap();
        let (resting, _, _) = pages.take(PAGE / 2, 1, 0, Ripe::ALL, |_| Some(2)).unwrap();
        pages.give(resting, 0);
        for i in 0..12 {
            pages.add(i, PAGE, false, 0);
        }
        pages.reclaim(0, true);
        assert_eq!(pages.count(), 8 + 1);
        pages.set_spare(2);
        pages.reclaim(REST - 1, false);
        assert_eq!(pages.count(), 2 + 1);
        assert_eq!(pages.resting.len(), 1);
        pages.set_spare(0);
        pages.reclaim(REST, true);
        assert_eq!(pages.count(), 1);
        assert!(pages.resting.is_empty());
        pages.give(held, REST);
        pages.reclaim(2 * REST, true);
        assert_eq!(pages.count(), 0);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pages: Pages<u32> = Pages::new(STAGING, 2);
        let (made, next) = counted();
        let make = |_| next(&made);
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pages
                .take(65536 * (1 + i % 7), 65536, 0, Ripe::ALL, make)
                .unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.page != p.page
                || p.start + p.len <= q.start
                || q.start + q.len <= p.start));
            taken.push(p);
        }
        let count = pages.count();
        assert!(count >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pages.give(first, 10);
        let (p, _, _) = pages
            .take(first.len, 65536, 10 + REST - 1, Ripe::ALL, make)
            .unwrap();
        assert_ne!((p.page, p.start), (first.page, first.start));
        let (q, _, again) = pages
            .take(first.len, 65536, 10 + REST, Ripe::ALL, make)
            .unwrap();
        assert_eq!((q.page, q.start, again), (first.page, first.start, true));
        // Everything back: the empty pages past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pages.give(p, 20);
        }
        pages.reclaim(20 + REST, false);
        assert_eq!(pages.count(), count.min(2));
        // Too big for a staging page, and a page that cannot be made.
        assert!(pages.take(PAGE, 65536, 30, Ripe::ALL, make).is_none());
        let mut none: Pages<u32> = Pages::new(STAGING, 2);
        assert!(none.take(65536, 65536, 0, Ripe::ALL, |_| None).is_none());
    }

    #[test]
    fn pieces_come_only_from_pages_older_than_the_lists_that_run() {
        let mut pages: Pages<u32> = Pages::new(TEXTURES, 2);
        let (made, next) = counted();
        let make = |_| next(&made);
        let at = |started, finished| Ripe { started, finished };
        let take = |pages: &mut Pages<u32>, ripe| {
            pages
                .take(PAGE / 2, 65536, 0, ripe, make)
                .map(|(_, page, _)| page)
        };
        // Nothing runs: at once, and pages are made ahead, one a time, until
        // two are empty.
        assert_eq!(take(&mut pages, at(0, 0)), Some(1));
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pages, at(0, 0)), Some(1));
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs: 2 and 3 are ripe.
        assert_eq!(take(&mut pages, at(5, 4)), Some(2));
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pages, at(5, 4)), Some(2));
        assert_eq!(take(&mut pages, at(6, 4)), Some(3));
        assert_eq!(take(&mut pages, at(6, 4)), Some(3));
        // None is ripe with room: not placed, and no page is made past the
        // spare count (4 and 5 are empty); placed once 5 has finished.
        assert_eq!(take(&mut pages, at(7, 4)), None);
        assert_eq!(made.get(), 5);
        assert_eq!(take(&mut pages, at(7, 4)), None);
        assert_eq!(made.get(), 5);
        assert_eq!(take(&mut pages, at(7, 5)), Some(4));
        // A whole page's worth, from a ripe empty one.
        assert_eq!(
            pages
                .take(PAGE, 65536, 0, at(8, 7), make)
                .map(|(_, page, _)| page),
            Some(5)
        );
    }

    #[test]
    fn a_piece_larger_than_a_page_has_a_heap_of_its_own_let_go_with_it() {
        let mut pages: Pages<u32> = Pages::new(TEXTURES, 0);
        let sizes = std::cell::RefCell::new(Vec::new());
        let make = |size| {
            sizes.borrow_mut().push(size);
            Some(sizes.borrow().len() as u32)
        };
        let len = PAGE + 3 * 65536 + 100;
        let (piece, _, _) = pages.take(len, 65536, 0, Ripe::ALL, make).unwrap();
        assert_eq!(*sizes.borrow(), [len.next_multiple_of(65536)]);
        assert_eq!((piece.start, piece.len), (0, len));
        assert_eq!(pages.count(), 1);
        // Never shared, never kept as a spare.
        pages.set_spare(4);
        assert!(pages.take(65536, 65536, 0, Ripe::ALL, make).is_some());
        assert_ne!(sizes.borrow()[1], len.next_multiple_of(65536));
        pages.give(piece, 0);
        pages.reclaim(REST, true);
        assert!(pages.pages.iter().flatten().all(|p| !p.own));
        // Staging is never given a heap of its own.
        let mut staging: Pages<u32> = Pages::new(STAGING, 2);
        assert!(staging.take(len, 65536, 0, Ripe::ALL, make).is_none());
    }

    #[test]
    fn pages_made_ahead_are_the_spares_and_never_go_and_come_back() {
        let mut pages: Pages<u32> = Pages::new(TEXTURES, AHEAD);
        let (made, next) = counted();
        let make = |_| next(&made);
        let (piece, _, _) = pages.take(65536, 65536, 0, Ripe::ALL, make).unwrap();
        for _ in 0..AHEAD + 4 {
            pages.take(65536, 65536, 0, Ripe::ALL, make).unwrap();
        }
        // One in use and the spares, made ahead one per piece.
        assert_eq!(pages.count(), 1 + AHEAD);
        pages.reclaim(REST, true);
        assert_eq!(pages.count(), 1 + AHEAD);
        let before = made.get();
        // Under pressure the spares past the smaller count go, newest first,
        // and none is made again.
        pages.set_spare(AHEAD_PRESSED);
        pages.reclaim(REST, false);
        assert_eq!(pages.count(), 1 + AHEAD_PRESSED);
        assert!(pages.pages[..1 + AHEAD_PRESSED].iter().all(Option::is_some));
        pages.take(65536, 65536, REST, Ripe::ALL, make).unwrap();
        assert_eq!(made.get(), before);
        pages.give(piece, REST);
        assert_eq!(pages.count(), 1 + AHEAD_PRESSED);
    }

    #[test]
    fn a_compaction_empties_the_emptiest_page() {
        let mut pages: Pages<u32> = Pages::new(TEXTURES, 1);
        let (made, next) = counted();
        let make = |_| next(&made);
        let quarter = PAGE / 4;
        // Three pages of four quarters each, and a spare one.
        let mut taken: Vec<Piece> = (0..12)
            .map(|_| pages.take(quarter, 65536, 0, Ripe::ALL, make).unwrap().0)
            .collect();
        assert_eq!(pages.count(), 4);
        // Each keeps two quarters; one of page 0's then rests.
        let gone: Vec<Piece> = (0..3)
            .flat_map(|page| {
                let taken = &taken;
                taken
                    .iter()
                    .copied()
                    .filter(move |q| q.page == page)
                    .skip(2)
            })
            .collect();
        for &p in &gone {
            pages.give(p, 0);
        }
        taken.retain(|p| !gone.contains(p));
        pages.reclaim(REST, false);
        let resting = taken.iter().position(|p| p.page == 0).unwrap();
        pages.give(taken.remove(resting), REST);
        assert_eq!(pages.compaction(0, Ripe::ALL), []);
        // Page 0's live quarter goes to page 1; page 2 has nowhere to go
        // then (page 1 has one free quarter left), and the empty spare page
        // is no place to move to.
        let moves = pages.compaction(1, Ripe::ALL);
        assert_eq!(pages.compaction(4, Ripe::ALL), moves);
        assert_eq!(moves.len(), 1);
        let m = moves[0];
        assert_eq!((m.from.page, m.to.page, m.to.len), (0, 1, quarter));
        assert!(taken.contains(&m.from));
        assert!(pages.claim(m.to, 65536).is_some());
        assert!(pages.claim(m.to, 65536).is_none());
        // The texture moved: its old piece comes back, and page 0 is empty
        // and goes once both its pieces have rested.
        pages.give(m.from, REST);
        pages.set_spare(0);
        pages.reclaim(2 * REST - 1, true);
        assert!(pages.memory(0).is_some());
        pages.reclaim(2 * REST, true);
        assert!(pages.memory(0).is_none());
        assert_eq!(pages.count(), 2);
        // No two live pieces overlap.
        let page = pages.pages[1].as_ref().unwrap();
        let mut end = 0;
        for (&start, &(len, _)) in &page.live {
            assert!(start >= end);
            end = start + len;
        }
        assert_eq!(page.live.len(), 3);
    }

    #[test]
    fn a_compaction_moves_nothing_that_has_nowhere_to_go_or_is_not_ripe() {
        let mut pages: Pages<u32> = Pages::new(TEXTURES, 0);
        let piece = |page, start, len| Piece { page, start, len };
        let three = PAGE / 4 * 3;
        pages.add(1, PAGE, false, 0);
        pages.add(2, PAGE, false, 0);
        assert!(pages.claim(piece(0, 0, three), 65536).is_some());
        assert!(pages.claim(piece(1, 0, three), 65536).is_some());
        assert_eq!(pages.compaction(4, Ripe::ALL), []);
        // Two pages made while submission 5 ran, a quarter used in each.
        let quarter = PAGE / 4;
        pages.add(3, PAGE, false, 5);
        pages.add(4, PAGE, false, 5);
        assert!(pages.claim(piece(2, 0, quarter), 65536).is_some());
        assert!(pages.claim(piece(3, 0, quarter), 65536).is_some());
        assert!(pages.claim(piece(3, 100, quarter), 65536).is_none());
        let at = |finished| Ripe {
            started: 9,
            finished,
        };
        assert_eq!(pages.compaction(4, at(4)), []);
        let moves = pages.compaction(4, at(5));
        assert_eq!(moves.len(), 1);
        assert_eq!((moves[0].from.page, moves[0].to.page), (2, 3));
    }
}
