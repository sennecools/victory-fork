//! Counters for the frame budget of the title's submit thread (wave 0 of the
//! bottleneck-first roadmap): what its waits for flights are made of, which
//! copies make it wait before a refresh over a home (H2), how many parts the
//! title's submissions are cut into, and how many descriptor views are in
//! use. Release builds count them too; they change nothing that runs, and
//! [`line`] says them once a second beside the frame lines.
//!
//! [`wave1_line`] says, beside it, what decides which of wave 1's steps are
//! needed: H2 waits by class.
//!
//! Each counting site is one relaxed atomic add, but for these: the relay's
//! clock at a phase change is a few atomic stores under a sequence number,
//! without a lock ([`AtomicClock`]); and one lock, the H2 table's, taken
//! after an H2 wait ([`h2_waited`]) and by [`line`]. None is on the path of
//! a draw.
//!
//! shortcut: no read tripwire on merged pages (the title's CPU reads of
//! pages a landing merged): the guards and the write watch see writes only,
//! and catching reads would change what the title sees. W2.8 adds it.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Notes what a counter counted in this thread's log ([`Seen`], tests only;
/// nothing in other builds).
macro_rules! seen {
    ($seen:expr) => {
        #[cfg(test)]
        SEEN.with(|log| log.borrow_mut().push($seen));
    };
}

/// What made the submit thread wait for flights, before or while it
/// prepares a submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Wait {
    /// A submission that is waited for settles every flight first.
    Sync,
    /// A submission sent ahead settles every flight first (`late_blocked`
    /// gave a reason other than a read the relay can wait for alone).
    Settle,
    /// A title buffer copy, a fill or a texture reads what flights write
    /// (H8): until those flights have landed.
    H8,
    /// A copy's refresh over home bytes a flight writes (H2): until those
    /// flights have landed.
    H2,
    /// The settle made for the memory core's frees when nothing else brings
    /// a moment with nothing in the air.
    Quiet,
    /// While preparing: title memory a flight may write changed and is
    /// refreshed.
    Stale,
    /// While preparing: too few card results left.
    Most,
}

impl Wait {
    const ALL: [Wait; 7] = [
        Wait::Sync,
        Wait::Settle,
        Wait::H8,
        Wait::H2,
        Wait::Quiet,
        Wait::Stale,
        Wait::Most,
    ];

    fn name(self) -> &'static str {
        match self {
            Wait::Sync => "sync",
            Wait::Settle => "settle",
            Wait::H8 => "H8",
            Wait::H2 => "H2",
            Wait::Quiet => "quiet",
            Wait::Stale => "stale",
            Wait::Most => "most",
        }
    }
}

/// Per [`Wait`]: the waits, and their microseconds in all, in the card's
/// fence and in landing (since the last [`line`]).
static WAITED: [[AtomicU64; 4]; Wait::ALL.len()] =
    [const { [const { AtomicU64::new(0) }; 4] }; Wait::ALL.len()];

/// Microseconds the submit thread spent taking home bytes flights write
/// again for card copies whose CPU side is a home (`retake_flight_homes`,
/// next to H2 and part of its cost, though no wait).
static RETAKE_US: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// This thread's waits for flights so far, and the microseconds it
    /// waited for the card's fence itself in them (a settle without the
    /// relay): [`Measure`] looks at what changed.
    static MINE: Cell<[u64; 2]> = const { Cell::new([0, 0]) };
}

/// The calling thread waited for flights (`relay_wait_with`, or a settle
/// without the relay), `fence_us` of it for the card's fence itself.
pub(crate) fn waited_here(fence_us: u64) {
    MINE.with(|mine| {
        let [waits, us] = mine.get();
        mine.set([waits + 1, us + fence_us]);
    });
}

/// Where the thread that lands flights is (`relay_worker`): the time it
/// spends in each [`Phase`] is what a wait for it was made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Looking at the line, or with nothing to do.
    Idle,
    /// Waiting for the card's fence: for the first flight of the line, or
    /// for a fetch of card copies' pieces.
    Card,
    /// Landing flights on the CPU: merges, telling, the line.
    Land,
}

/// What the relay's clock holds ([`CLOCK`]): microseconds of ended time in
/// [`Phase::Card`] and [`Phase::Land`], and the phase it is in with when
/// that began ([`pack`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Clock {
    spent: [u64; 2],
    state: u64,
}

/// Bits of [`Clock::state`] below the phase: the phase's start, in
/// microseconds since [`EPOCH`].
const PHASE_SHIFT: u32 = 62;

/// A phase and its start as one word.
fn pack(phase: Phase, start: u64) -> u64 {
    (phase as u64) << PHASE_SHIFT | start & ((1 << PHASE_SHIFT) - 1)
}

/// The phase and start of a [`pack`]ed word.
fn unpack(state: u64) -> (Phase, u64) {
    let phase = match state >> PHASE_SHIFT {
        1 => Phase::Card,
        2 => Phase::Land,
        _ => Phase::Idle,
    };
    (phase, state & ((1 << PHASE_SHIFT) - 1))
}

impl Clock {
    /// The clock once the relay entered `phase` at `now`, and the phase it
    /// was in.
    fn entered(self, phase: Phase, now: u64) -> (Clock, Phase) {
        let (was, since) = unpack(self.state);
        let mut spent = self.spent;
        let ended = now.saturating_sub(since);
        match was {
            Phase::Card => spent[0] += ended,
            Phase::Land => spent[1] += ended,
            Phase::Idle => {}
        }
        let state = pack(phase, now);
        (Clock { spent, state }, was)
    }

    /// The microseconds in [`Phase::Card`] and [`Phase::Land`] up to `now`,
    /// the current phase's so far included.
    fn read(self, now: u64) -> [u64; 2] {
        let (phase, since) = unpack(self.state);
        let mut spent = self.spent;
        let open = now.saturating_sub(since);
        match phase {
            Phase::Card => spent[0] += open,
            Phase::Land => spent[1] += open,
            Phase::Idle => {}
        }
        spent
    }
}

/// Where the relay's clock counts from.
static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Microseconds since [`EPOCH`].
fn now_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// The relay's clock, without a lock: written by the thread that lands
/// flights alone, a few times per landing; read by any thread. A sequence
/// number, odd while a write is under way, tells a reader to read again
/// (a sequence lock: the writer never waits, a reader only for a write in
/// progress). Ended card and landing microseconds, then the packed phase.
struct AtomicClock {
    sequence: AtomicU64,
    spent: [AtomicU64; 2],
    state: AtomicU64,
}

impl AtomicClock {
    /// What the clock holds (as its one writer sees it).
    fn held(&self) -> Clock {
        Clock {
            spent: [0, 1].map(|i| self.spent[i].load(Ordering::Relaxed)),
            state: self.state.load(Ordering::Relaxed),
        }
    }

    /// The relay enters `phase` now: the phase it was in. From one thread
    /// only (two writers would break the sequence). The sequence is odd
    /// before the time is taken, so that no reader's time between this
    /// one's and the stores below counts the phase that ends twice.
    fn enter(&self, phase: Phase) -> Phase {
        let sequence = self.sequence.load(Ordering::Relaxed);
        self.sequence.store(sequence + 1, Ordering::SeqCst);
        // The odd sequence before the stores below.
        std::sync::atomic::fence(Ordering::Release);
        let (next, was) = self.held().entered(phase, now_us());
        for (i, spent) in next.spent.iter().enumerate() {
            self.spent[i].store(*spent, Ordering::Relaxed);
        }
        self.state.store(next.state, Ordering::Relaxed);
        self.sequence.store(sequence + 2, Ordering::SeqCst);
        was
    }

    /// The microseconds in [`Phase::Card`] and [`Phase::Land`] so far: one
    /// consistent copy and a time taken while no write was under way.
    fn read(&self) -> [u64; 2] {
        loop {
            let before = self.sequence.load(Ordering::SeqCst);
            if before % 2 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let clock = self.held();
            let now = now_us();
            // The loads above before the sequence's second look.
            std::sync::atomic::fence(Ordering::Acquire);
            if self.sequence.load(Ordering::SeqCst) == before {
                return clock.read(now);
            }
        }
    }
}

static CLOCK: AtomicClock = AtomicClock {
    sequence: AtomicU64::new(0),
    spent: [const { AtomicU64::new(0) }; 2],
    state: AtomicU64::new(0),
};

/// The relay is in a phase until this is dropped; then in the one before.
pub(crate) struct InPhase(Phase);

impl Drop for InPhase {
    fn drop(&mut self) {
        CLOCK.enter(self.0);
    }
}

/// The thread that lands flights enters `phase` (until the result is
/// dropped).
pub(crate) fn phase(phase: Phase) -> InPhase {
    InPhase(CLOCK.enter(phase))
}

/// The start of a wait of the submit thread for flights ([`Measure::end`]).
pub(crate) struct Measure {
    started: Instant,
    relay: [u64; 2],
    mine: [u64; 2],
}

impl Measure {
    pub(crate) fn start() -> Measure {
        Measure {
            started: Instant::now(),
            relay: CLOCK.read(),
            mine: MINE.with(Cell::get),
        }
    }

    /// Counts what came after [`Measure::start`] as a wait of kind `kind`,
    /// if this thread waited for flights meanwhile. `relayed`: the relay
    /// lands flights, so the time not in its card wait or landing is
    /// neither; else this thread landed them itself, and what was not its
    /// fence wait was landing. The microseconds waited, if it waited.
    pub(crate) fn end(self, kind: Wait, relayed: bool) -> Option<u64> {
        let mine = MINE.with(Cell::get);
        if mine[0] == self.mine[0] {
            return None;
        }
        let relay = CLOCK.read();
        let total = self.started.elapsed().as_micros() as u64;
        let [card, land] = [0, 1].map(|i| relay[i].saturating_sub(self.relay[i]));
        let own_fence = mine[1] - self.mine[1];
        let (fence, land) = split(total, card + own_fence, land, relayed);
        let counts = &WAITED[kind as usize];
        counts[0].fetch_add(1, Ordering::Relaxed);
        counts[1].fetch_add(total, Ordering::Relaxed);
        counts[2].fetch_add(fence, Ordering::Relaxed);
        counts[3].fetch_add(land, Ordering::Relaxed);
        seen!(Seen::Wait {
            kind,
            total,
            fence,
            land
        });
        Some(total)
    }
}

/// A wait of `total` microseconds: its part in the card's fence and its
/// part landing, each at most what is left of `total`. Without the relay
/// what was not the fence was landing.
fn split(total: u64, fence: u64, land: u64, relayed: bool) -> (u64, u64) {
    let fence = fence.min(total);
    let land = if relayed { land } else { total - fence };
    (fence, land.min(total - fence))
}

/// Of a submission sent ahead with R5 (whether it waited for H2 or not),
/// the time the submit thread took home bytes again for card copies whose
/// CPU side is a home.
pub(crate) fn retaken(spent: Duration) {
    RETAKE_US.fetch_add(spent.as_micros() as u64, Ordering::Relaxed);
}

/// Bytes of `[from, to)` inside `ranges` (in order and joined).
pub(crate) fn overlap_bytes(ranges: &[(usize, usize)], from: usize, to: usize) -> u64 {
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges[at..]
        .iter()
        .take_while(|r| r.0 < to)
        .map(|r| r.1.min(to).saturating_sub(r.0.max(from)) as u64)
        .sum()
}

/// The H2 waits of one kind of copy (`Shadow::budget_key`): the waits it
/// took part in, its refreshes in them, the bytes of those refreshes over
/// homes flights wrote (alias bytes), and its share of the waits'
/// microseconds (by alias bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Culprit {
    pub(crate) key: u16,
    pub(crate) waits: u64,
    pub(crate) refreshes: u64,
    pub(crate) alias: u64,
    pub(crate) us: u64,
}

impl Culprit {
    fn merge(&mut self, other: &Culprit) {
        self.waits += other.waits;
        self.refreshes += other.refreshes;
        self.alias += other.alias;
        self.us += other.us;
    }
}

/// The kinds of copy kept apart in the H2 table; the rest is one sum.
pub(crate) const TOP: usize = 8;

/// The H2 table: up to [`TOP`] kinds of copy, and what was pushed out of it
/// (`other`, its key unused). A kind not there when the table is full
/// pushes out the one with the least time, whose figures go to `other`.
#[derive(Debug)]
struct Top {
    slots: [Culprit; TOP],
    used: usize,
    other: Culprit,
}

impl Top {
    const fn new() -> Top {
        Top {
            slots: [Culprit {
                key: 0,
                waits: 0,
                refreshes: 0,
                alias: 0,
                us: 0,
            }; TOP],
            used: 0,
            other: Culprit {
                key: 0,
                waits: 0,
                refreshes: 0,
                alias: 0,
                us: 0,
            },
        }
    }

    fn add(&mut self, culprit: &Culprit) {
        let used = &mut self.slots[..self.used];
        if let Some(slot) = used.iter_mut().find(|s| s.key == culprit.key) {
            slot.merge(culprit);
            return;
        }
        if self.used < TOP {
            self.slots[self.used] = *culprit;
            self.used += 1;
            return;
        }
        let least = (0..TOP)
            .min_by_key(|&i| (self.slots[i].us, self.slots[i].alias))
            .unwrap_or(0);
        let out = self.slots[least];
        self.other.merge(&out);
        self.slots[least] = *culprit;
    }

    /// The kinds kept, most time first, and the sum pushed out; the table
    /// is empty after.
    fn take(&mut self) -> (Vec<Culprit>, Culprit) {
        let mut kept = self.slots[..self.used].to_vec();
        kept.sort_by(|a, b| (b.us, b.alias, a.key).cmp(&(a.us, a.alias, b.key)));
        let other = self.other;
        *self = Top::new();
        (kept, other)
    }
}

static H2_TOP: Mutex<Top> = Mutex::new(Top::new());

/// One H2 wait of `us` microseconds, by the kinds of copy it waited for:
/// each refresh's kind and alias bytes. The time is shared by alias bytes
/// (evenly if there are none).
fn shares(us: u64, refreshes: &[(u16, u64)]) -> Vec<Culprit> {
    let mut kinds: Vec<Culprit> = Vec::new();
    for &(key, alias) in refreshes {
        let at = match kinds.iter().position(|k| k.key == key) {
            Some(at) => at,
            None => {
                kinds.push(Culprit {
                    key,
                    waits: 1,
                    ..Culprit::default()
                });
                kinds.len() - 1
            }
        };
        kinds[at].refreshes += 1;
        kinds[at].alias += alias;
    }
    let bytes: u64 = kinds.iter().map(|k| k.alias).sum();
    let n = kinds.len() as u64;
    for kind in &mut kinds {
        kind.us = if bytes > 0 {
            (u128::from(us) * u128::from(kind.alias) / u128::from(bytes)) as u64
        } else {
            us / n
        };
    }
    kinds
}

/// A submission waited `us` microseconds for H2; `refreshes`: the kind of
/// copy (`Shadow::budget_key`) and the alias bytes of each refresh that
/// made it wait.
pub(crate) fn h2_waited(us: u64, refreshes: &[(u16, u64)]) {
    if refreshes.is_empty() {
        return;
    }
    let kinds = shares(us, refreshes);
    let mut top = lock(&H2_TOP);
    for kind in &kinds {
        top.add(kind);
        seen!(Seen::Culprit(*kind));
    }
}

/// What one H2 wait was made by, from the first refresh that made it wait
/// (`crate::host`'s H2 site, once per wait): what wave 1's alias fixes
/// would each take away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum H2Class {
    /// A buffer view's own copy whose buffer's copy is a home of the core.
    C1a,
    /// A buffer view's own copy whose buffer's copy is a card copy with a
    /// home as its CPU side.
    C1b,
    /// A copy kept for an overlap with a home, inside that home's pages.
    C2In,
    /// A copy kept for an overlap that crosses pages or homes.
    C2Cross,
    /// A card copy that kept its staging buffer, off a 16-byte step of its
    /// home.
    C3,
    Other,
}

impl H2Class {
    const ALL: [H2Class; 6] = [
        H2Class::C1a,
        H2Class::C1b,
        H2Class::C2In,
        H2Class::C2Cross,
        H2Class::C3,
        H2Class::Other,
    ];

    fn name(self) -> &'static str {
        match self {
            H2Class::C1a => "C1a",
            H2Class::C1b => "C1b",
            H2Class::C2In => "C2-in",
            H2Class::C2Cross => "C2-cross",
            H2Class::C3 => "C3",
            H2Class::Other => "other",
        }
    }

    /// The class of a refresh of `origin`'s copy (`carded`: a copy in the
    /// card's memory) over home bytes; `in_home`: whether the refresh lies
    /// inside the pages of the home that holds its first byte over homes.
    pub(crate) fn of(origin: crate::shadow::Origin, carded: bool, in_home: bool) -> H2Class {
        use crate::home::Why;
        use crate::shadow::{Kept, Origin};
        match origin {
            Origin::Exact {
                base_is_card: true, ..
            } => H2Class::C1b,
            Origin::Exact {
                base_is_home: true, ..
            } => H2Class::C1a,
            Origin::Kept(Kept::Why(Why::Overlap)) if in_home => H2Class::C2In,
            Origin::Kept(Kept::Why(Why::Overlap)) => H2Class::C2Cross,
            Origin::Kept(Kept::Misaligned) if carded => H2Class::C3,
            _ => H2Class::Other,
        }
    }
}

/// Per [`H2Class`]: waits, the alias bytes of their first refreshes, and
/// the waits' microseconds.
static H2_CLASSES: [[AtomicU64; 3]; H2Class::ALL.len()] =
    [const { [const { AtomicU64::new(0) }; 3] }; H2Class::ALL.len()];

/// One H2 wait of `us` microseconds, of `class`, its first refresh `bytes`
/// over homes.
pub(crate) fn h2_class(class: H2Class, bytes: u64, us: u64) {
    let counts = &H2_CLASSES[class as usize];
    counts[0].fetch_add(1, Ordering::Relaxed);
    counts[1].fetch_add(bytes, Ordering::Relaxed);
    counts[2].fetch_add(us, Ordering::Relaxed);
    seen!(Seen::H2Class { class, bytes, us });
}

/// The first byte of `[from, to)` inside `ranges` (in order and joined).
pub(crate) fn first_overlap(ranges: &[(usize, usize)], from: usize, to: usize) -> Option<usize> {
    if from >= to {
        return None;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).filter(|r| r.0 < to).map(|r| r.0.max(from))
}

/// What was counted on this thread, in order (tests: the process's counters
/// are shared with every other test).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Seen {
    Wait {
        kind: Wait,
        total: u64,
        fence: u64,
        land: u64,
    },
    Culprit(Culprit),
    H2Class {
        class: H2Class,
        bytes: u64,
        us: u64,
    },
}

#[cfg(test)]
thread_local! {
    static SEEN: std::cell::RefCell<Vec<Seen>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// What this thread counted since the last call (tests).
#[cfg(test)]
pub(crate) fn take_seen() -> Vec<Seen> {
    SEEN.with(|seen| std::mem::take(&mut *seen.borrow_mut()))
}

/// What the queues do with the title's submissions ([`count`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Part {
    /// `ExecuteCommandLists` calls of the title.
    TitleSubmits,
    /// Of a submission cut at its waits (`submit_cut`): lists run at once.
    AtOnce,
    /// Parts given to the queue behind waits for memory values.
    Held,
    /// Waits for memory values in those parts (another queue's writes, as
    /// often as not: cross-queue waits within a chain).
    ValueWaits,
    /// Held parts that ran (`release_held`).
    HeldRun,
    /// Of those, run at a present without their waits being met.
    Forced,
    /// The title's queue waits for a fence (`ID3D12CommandQueue::Wait`).
    QueueWaits,
    /// Of those, for a value not reached yet (or behind what the queue
    /// holds).
    QueueWaitsBehind,
    /// The title's CPU waits for a fence value not reached yet.
    FenceWaits,
    /// Of those, blocking (no event).
    FenceBlocking,
}

const PARTS: usize = 10;

static PARTS_SEEN: [AtomicU64; PARTS] = [const { AtomicU64::new(0) }; PARTS];

/// `n` more of `part`.
pub(crate) fn count(part: Part, n: u64) {
    PARTS_SEEN[part as usize].fetch_add(n, Ordering::Relaxed);
}

/// The host's submissions at the last [`line`].
static SUBMITTED_BEFORE: AtomicU64 = AtomicU64::new(0);

/// The limit of shader-visible view heaps on resource binding tiers 1 and
/// 2: what one persistent heap holding every view would have to fit.
const VIEW_LIMIT: u64 = 1_000_000;

/// For the frame line, once a second: what was counted since the last one,
/// per frame over `frames`. `submitted`: the host's submissions so far;
/// `views`: the view table's entries in use now and the most ever in use
/// (`crate::descriptor::views_in_use`); `heap`: the slots of a host view
/// heap.
pub(crate) fn line(frames: f64, submitted: u64, views: [u64; 2], heap: u64) {
    let per = |n: u64| n as f64 / frames;
    let ms = |us: u64| us as f64 / 1000.0 / frames;
    let p: [u64; PARTS] = std::array::from_fn(|i| PARTS_SEEN[i].swap(0, Ordering::Relaxed));
    let host = submitted.saturating_sub(SUBMITTED_BEFORE.swap(submitted, Ordering::Relaxed));
    let waits: Vec<String> = Wait::ALL
        .iter()
        .filter_map(|&kind| {
            let w: [u64; 4] =
                std::array::from_fn(|i| WAITED[kind as usize][i].swap(0, Ordering::Relaxed));
            (w[0] > 0).then(|| {
                format!(
                    "{} {:.2} for {:.1} ms (fence {:.1}, landing {:.1})",
                    kind.name(),
                    per(w[0]),
                    ms(w[1]),
                    ms(w[2]),
                    ms(w[3])
                )
            })
        })
        .collect();
    let (kept, other) = lock(&H2_TOP).take();
    let culprits: Vec<String> = kept
        .iter()
        .take(4)
        .map(|c| describe(&crate::shadow::budget_name(c.key), c, frames))
        .chain((other.waits > 0).then(|| describe("others", &other, frames)))
        .collect();
    tracing::info!(
        "frame budget per frame: {:.1} host submissions from {:.1} title submissions (cut: {:.1} run at once, {:.1} parts \
         held behind {:.1} waits for memory values, {:.1} held parts run, {:.2} forced at a present); queue fence waits \
         {:.1} ({:.1} not reached); title CPU fence waits {:.2} ({:.2} blocking); waits for flights: {}; H2 by copy: {}; \
         H2 retakes {:.2} ms; descriptor views in use {} (peak {}) against {VIEW_LIMIT}, host view heap {heap}",
        per(host),
        per(p[Part::TitleSubmits as usize]),
        per(p[Part::AtOnce as usize]),
        per(p[Part::Held as usize]),
        per(p[Part::ValueWaits as usize]),
        per(p[Part::HeldRun as usize]),
        per(p[Part::Forced as usize]),
        per(p[Part::QueueWaits as usize]),
        per(p[Part::QueueWaitsBehind as usize]),
        per(p[Part::FenceWaits as usize]),
        per(p[Part::FenceBlocking as usize]),
        if waits.is_empty() {
            "none".to_owned()
        } else {
            waits.join(", ")
        },
        if culprits.is_empty() {
            "none".to_owned()
        } else {
            culprits.join("; ")
        },
        ms(RETAKE_US.swap(0, Ordering::Relaxed)),
        views[0],
        views[1],
    );
}

/// The frame budget's second line, beside [`line`]: what decides which of
/// wave 1's steps are needed, per frame over `frames`.
pub(crate) fn wave1_line(frames: f64) {
    let per = |n: u64| n as f64 / frames;
    let mb = |b: u64| b as f64 / f64::from(1 << 20) / frames;
    let classes: Vec<String> = H2Class::ALL
        .iter()
        .filter_map(|&class| {
            let c: [u64; 3] =
                std::array::from_fn(|i| H2_CLASSES[class as usize][i].swap(0, Ordering::Relaxed));
            (c[0] > 0).then(|| {
                format!(
                    "{} {:.2} ({:.2} MB, {:.1} ms)",
                    class.name(),
                    per(c[0]),
                    mb(c[1]),
                    c[2] as f64 / 1000.0 / frames
                )
            })
        })
        .collect();
    tracing::info!(
        "frame budget for wave 1 per frame: H2 by class {}",
        if classes.is_empty() {
            "none".to_owned()
        } else {
            classes.join(", ")
        },
    );
}

/// One entry of the H2 table for the line.
fn describe(name: &str, c: &Culprit, frames: f64) -> String {
    format!(
        "{name}: {:.2} waits, {:.1} refreshes, {:.2} MB over homes, {:.1} ms",
        c.waits as f64 / frames,
        c.refreshes as f64 / frames,
        c.alias as f64 / f64::from(1 << 20) / frames,
        c.us as f64 / 1000.0 / frames
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_clock_counts_each_phase_up_to_now() {
        let mut clock = Clock {
            spent: [0, 0],
            state: 0,
        };
        let mut enter = |phase: Phase, now: u64| {
            let (next, was) = clock.entered(phase, now);
            clock = next;
            (was, clock)
        };
        let (was, at10) = enter(Phase::Card, 10);
        assert_eq!(was, Phase::Idle);
        assert_eq!(at10.read(14), [4, 0]);
        assert_eq!(enter(Phase::Land, 20).0, Phase::Card);
        // A fetch's wait inside a landing, and back.
        assert_eq!(enter(Phase::Card, 23).0, Phase::Land);
        assert_eq!(enter(Phase::Land, 25).0, Phase::Card);
        let (was, idle) = enter(Phase::Idle, 30);
        assert_eq!(was, Phase::Land);
        assert_eq!(idle.read(100), [12, 8]);
        for phase in [Phase::Idle, Phase::Card, Phase::Land] {
            assert_eq!(unpack(pack(phase, 123_456)), (phase, 123_456));
        }
    }

    #[test]
    fn the_atomic_clock_reads_whole_while_the_relay_writes() {
        // Its own clock, written by one thread while this one reads: each
        // read is one the writer left (card and landing only grow, and their
        // sum never passes the time since the clock began).
        static TEST_CLOCK: AtomicClock = AtomicClock {
            sequence: AtomicU64::new(0),
            spent: [const { AtomicU64::new(0) }; 2],
            state: AtomicU64::new(0),
        };
        let began = now_us();
        let writer = std::thread::spawn(|| {
            for i in 0..20_000 {
                TEST_CLOCK.enter([Phase::Card, Phase::Land, Phase::Idle][i % 3]);
            }
        });
        let mut last = [0, 0];
        while !writer.is_finished() {
            let read = TEST_CLOCK.read();
            assert!(
                read[0] >= last[0] && read[1] >= last[1],
                "{read:?} after {last:?}"
            );
            assert!(read[0] + read[1] <= now_us() - began + 1);
            last = read;
        }
        writer.join().unwrap();
    }

    #[test]
    fn a_wait_is_split_into_fence_and_landing_within_its_time() {
        // With the relay: what the relay did, the rest is neither.
        assert_eq!(split(100, 60, 30, true), (60, 30));
        // Never more than the wait (the relay's phases began before it).
        assert_eq!(split(100, 80, 50, true), (80, 20));
        assert_eq!(split(100, 150, 0, true), (100, 0));
        // Without it this thread landed what it did not wait for.
        assert_eq!(split(100, 70, 0, false), (70, 30));
    }

    #[test]
    fn a_measure_counts_only_a_thread_that_waited() {
        // On a thread of its own: `MINE` and the log are per thread.
        std::thread::spawn(|| {
            assert_eq!(Measure::start().end(Wait::Most, false), None);
            let measure = Measure::start();
            waited_here(0);
            let total = measure.end(Wait::Most, false).expect("it waited");
            // Without the relay, what was not the fence was landing.
            assert_eq!(
                take_seen(),
                vec![Seen::Wait {
                    kind: Wait::Most,
                    total,
                    fence: 0,
                    land: total
                }]
            );
        })
        .join()
        .unwrap();
    }

    #[test]
    fn overlap_bytes_counts_only_bytes_in_the_ranges() {
        let ranges = [(10, 20), (30, 40), (50, 60)];
        assert_eq!(overlap_bytes(&ranges, 0, 10), 0);
        assert_eq!(overlap_bytes(&ranges, 15, 35), 10);
        assert_eq!(overlap_bytes(&ranges, 0, 100), 30);
        assert_eq!(overlap_bytes(&ranges, 40, 50), 0);
        assert_eq!(overlap_bytes(&[], 0, 100), 0);
        // A reference: byte by byte.
        for from in 0..70 {
            for to in from..70 {
                let reference = (from..to)
                    .filter(|&b| ranges.iter().any(|r| r.0 <= b && b < r.1))
                    .count() as u64;
                assert_eq!(overlap_bytes(&ranges, from, to), reference, "{from}..{to}");
            }
        }
    }

    #[test]
    fn a_wait_is_shared_by_alias_bytes() {
        let kinds = shares(1000, &[(1, 300), (2, 100), (1, 100)]);
        assert_eq!(
            kinds,
            vec![
                Culprit {
                    key: 1,
                    waits: 1,
                    refreshes: 2,
                    alias: 400,
                    us: 800
                },
                Culprit {
                    key: 2,
                    waits: 1,
                    refreshes: 1,
                    alias: 100,
                    us: 200
                },
            ]
        );
        let even = shares(90, &[(1, 0), (2, 0), (3, 0)]);
        assert!(even.iter().all(|k| k.us == 30));
    }

    /// A small deterministic generator for the model test.
    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn the_h2_table_matches_a_full_count_while_it_has_room() {
        let mut seed = 0x9e37_79b9_7f4a_7c15;
        let mut top = Top::new();
        let mut reference: std::collections::BTreeMap<u16, Culprit> = Default::default();
        for _ in 0..2000 {
            let key = (next(&mut seed) % TOP as u64) as u16;
            let c = Culprit {
                key,
                waits: 1,
                refreshes: 1 + next(&mut seed) % 4,
                alias: next(&mut seed) % 100_000,
                us: next(&mut seed) % 5_000,
            };
            top.add(&c);
            reference
                .entry(key)
                .or_insert(Culprit {
                    key,
                    ..Culprit::default()
                })
                .merge(&c);
        }
        let (kept, other) = top.take();
        assert_eq!(other, Culprit::default());
        let mut expected: Vec<Culprit> = reference.into_values().collect();
        expected.sort_by(|a, b| (b.us, b.alias, a.key).cmp(&(a.us, a.alias, b.key)));
        assert_eq!(kept, expected);
        // Taken: empty again.
        assert_eq!(top.take(), (Vec::new(), Culprit::default()));
    }

    #[test]
    fn the_h2_table_keeps_the_heavy_kinds_and_loses_nothing() {
        let mut seed = 0x0123_4567_89ab_cdef;
        let mut top = Top::new();
        let (mut waits, mut us) = (0, 0);
        for i in 0..5000u64 {
            // Four heavy kinds, every few waits; many light ones between.
            let (key, weight) = if i % 3 == 0 {
                ((i / 3 % 4) as u16, 10_000)
            } else {
                (100 + (next(&mut seed) % 200) as u16, next(&mut seed) % 10)
            };
            let c = Culprit {
                key,
                waits: 1,
                refreshes: 1,
                alias: weight,
                us: weight,
            };
            waits += 1;
            us += weight;
            top.add(&c);
        }
        let (kept, other) = top.take();
        assert_eq!(kept.len(), TOP);
        let heavy: Vec<u16> = kept.iter().take(4).map(|c| c.key).collect();
        let mut sorted = heavy.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3], "{kept:?}");
        // Every wait and microsecond is in the table or in `other`.
        let sum = |f: fn(&Culprit) -> u64| kept.iter().map(f).sum::<u64>() + f(&other);
        assert_eq!(sum(|c| c.waits), waits);
        assert_eq!(sum(|c| c.us), us);
    }

    #[test]
    fn h2_classes_follow_the_copys_origin() {
        use crate::home::Why;
        use crate::shadow::{Kept, Origin};
        let exact = |base_is_home, base_is_card| Origin::Exact {
            base_is_home,
            base_is_card,
        };
        let overlap = Origin::Kept(Kept::Why(Why::Overlap));
        let cases = [
            (exact(true, false), false, false, H2Class::C1a),
            (exact(false, true), false, false, H2Class::C1b),
            // `exact_of` never sets both (`home_kind` is `None` for a card
            // copy); if both were set, `base_is_card` would win.
            (exact(true, true), false, false, H2Class::C1b),
            (exact(false, false), false, false, H2Class::Other),
            (overlap, false, true, H2Class::C2In),
            (overlap, false, false, H2Class::C2Cross),
            (Origin::Kept(Kept::Misaligned), true, false, H2Class::C3),
            (Origin::Kept(Kept::Misaligned), false, false, H2Class::Other),
            (
                Origin::Kept(Kept::Why(Why::View)),
                false,
                false,
                H2Class::Other,
            ),
            (Origin::Made, true, true, H2Class::Other),
        ];
        for (origin, carded, in_home, class) in cases {
            assert_eq!(H2Class::of(origin, carded, in_home), class, "{origin:?}");
        }
    }

    #[test]
    fn first_overlap_is_the_first_byte_in_the_ranges() {
        let ranges = [(10, 20), (30, 40)];
        for from in 0..50 {
            for to in from..50 {
                let reference = (from..to).find(|&b| ranges.iter().any(|r| r.0 <= b && b < r.1));
                assert_eq!(first_overlap(&ranges, from, to), reference, "{from}..{to}");
            }
        }
    }
}
