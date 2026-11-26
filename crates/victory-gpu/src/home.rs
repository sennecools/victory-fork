//! The memory core (`VICTORY_MEMORY_CORE=1`; off by default): every title
//! byte the card uses has at most one *home*, the one place the GPU reads
//! and writes it, and every ask for those bytes resolves to that home. A
//! nested or overlapping ask gets an offset into the same home, never a
//! second copy that could disagree with it.
//!
//! This first slice covers one class of memory, **C**:
//! title buffers and windows of memory outside buffers whose bytes lie in
//! one `XMemVirtualAlloc` region (private memory, committed read-write) and
//! that get no copy in the card's memory. Their home is a part of the
//! title's own allocation opened as a host heap
//! (`OpenExistingHeapFromAddress1`, [`crate::inplace::open`]) with one
//! buffer placed over it. There is nothing to copy either way: no refresh,
//! no write-watch asking, no snapshot, no copy-back, and no buffer of ours
//! in the PC's memory beside the title's. Everything else keeps today's
//! path, **X**: a host copy kept current every submission
//! ([`crate::shadow`]). A big title buffer is offered a copy in the card's
//! memory first (**G**) and becomes C only when the card turns it down.
//!
//! With the memory pool on as well (`VICTORY_MEMORY_POOL=1`,
//! [`crate::pool`]), a second class, **C-pool**: title buffers and windows
//! whose bytes are one run of one committed chunk of the page pool at one
//! title view. Their home is the buffer placed over the whole chunk's heap
//! (`OpenExistingHeapFromFileMapping`), at the run's offset in the chunk
//! ([`Kind::Pool`]); it pins nothing. A remap of its title addresses lets it
//! go, as the frees do a C home (`Touch::Remapped`). View memory that is
//! scattered, in a reserved chunk, or in no chunk keeps its copy.
//!
//! What this module holds:
//!
//! - [`classify`]: the order of the rules, as a pure function over what a
//!   caller can find out ([`Ask`]), so that the order is tested without a
//!   device;
//! - the region table: each `XMemVirtualAlloc` reservation with the
//!   console's raw protection, kind and flags words (the memory-type bits
//!   `xmem` drops for the host are kept here as hints, for the census line
//!   only in this slice);
//! - the homes in the tables (for the one-home rule) and the homes whose
//!   heaps stand (for the title's frees and protections);
//! - [`title_virtual_free`] and [`title_virtual_protect`]: the title's
//!   `VirtualFree` and `VirtualProtect`, which `victory-hle` sends here when
//!   the switch is on. While a heap opened on its pages stands the host
//!   refuses both with error 87, so the homes there are let go first; a
//!   free that is still refused is done later and the title told it was
//!   done, a protection that is still refused is refused (and the region is
//!   never given a home again);
//! - [`cpu_touched`]: the one entry point for everything other than the GPU
//!   that touches title bytes;
//! - [`after_wait`]: the shaders wrote a home, and a copy elsewhere over the
//!   same bytes (which write watching cannot tell) must take them again;
//! - a line every ten seconds with what the core did and the census.
//!
//! What it assumes is what `crate::inplace` assumes: every submission is
//! waited for. With `VICTORY_LATE_WAIT` or the relay nothing is given a
//! home ([`Why::Late`]), unless R5 is on (`VICTORY_R5=1` with the relay's
//! second stage, `crate::host::r5`): then the flights land what homes need
//! too. At each landing, before the title is told anything, the copies over
//! the bytes their shaders wrote are told ([`after_wait`]); a copy over a
//! home is not refreshed while a flight may write the home; a protection of
//! the title's over a home a flight holds waits for the flights in the air
//! before it is refused ([`title_virtual_protect`]); frees that waited are
//! looked at again after each landing ([`after_landing`]); and a home let go
//! goes only once the flights that may name it are done
//! (`crate::host::retire`, dropped when `FINISHED` passes the submissions
//! running at the time).
//!
//! shortcuts with R5: a home's pages that the title frees, and the
//! directory's entries for them, still wait for the sweep (with flights in
//! the air, the forced one at most 1.5 s apart), not for the flights that
//! name them; a protection that is refused waits for every flight in the
//! air, not only those that name the home; a bottom-of-pipe write into a
//! home is done on the CPU at its flight's landing, while a later flight may
//! already read the home on the card; and a title buffer copy done on the
//! CPU into a home is not ordered after a flight in the air that reads it,
//! as the relay already assumes for host copies.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::ID3D12Device13;
use windows::core::Interface;
use windows_sys::Win32::Foundation::{GetLastError, SetLastError};
use windows_sys::Win32::System::Memory as mem;

use crate::host::HostList;
use crate::shadow::Shadow;

/// The host's page: what the title's frees and protections are rounded to.
const PAGE_4K: usize = 4096;
/// `ERROR_INVALID_PARAMETER`: the host's answer while a heap stands over
/// the pages.
const REFUSED_BY_HEAP: u32 = 87;

/// The place title bytes live for the GPU, as the rest of the runtime asks
/// for it: what a list reads, what its shaders may write. A host copy and a
/// home are both [`Shadow`]s; `Shadow::refresh` and `Shadow::gpu_writes` go
/// through this, so their callers (`draw.rs`, `memory.rs`,
/// `descriptor/host.rs`) do not tell the two apart.
pub(crate) trait Home: Send + Sync {
    /// `list` will read `[at, at + len)`: those bytes are current for it
    /// when it is submitted, and the home stays alive until the list is
    /// reset.
    fn visible(self: &Arc<Self>, list: &HostList, at: u64, len: u64);
    /// The shaders of `list` may write `[at, at + len)`.
    fn gpu_may_write(self: &Arc<Self>, list: &HostList, at: u64, len: u64);

    // ---- Slice 4 (`crate::card_home`, `VICTORY_MEMORY_CARDS=1`) ----

    /// The card wrote `[at, at + len)` into the home outside any list (a
    /// copy in the card's memory whose CPU side the home is, merged into it
    /// on the card): every copy over those bytes takes them again at its next
    /// refresh, as write watching does not see the card's writes.
    fn gpu_wrote(self: &Arc<Self>, at: u64, len: u64) {
        crate::watch::gpu_wrote(at as usize, len as usize);
    }
}

/// Something other than the GPU touched title bytes ([`cpu_touched`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Touch {
    /// The title or the runtime wrote through a path watching does not see
    /// (DirectStorage through our views, a kernel read into title memory).
    /// A home is the memory itself: nothing to do for it.
    Wrote,
    /// The runtime maps or unmaps pages there (`xmem`).
    Remapped,
    /// The title is about to decommit or release it (`VirtualFree`).
    Freeing,
    /// The title is about to change its protection (`VirtualProtect`).
    Protecting,
}

/// `VICTORY_MEMORY_CORE=1`: the core is on. Read once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MEMORY_CORE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "memory core on (VICTORY_MEMORY_CORE=1): title buffers and windows in private title memory are used by the card in place; VICTORY_INPLACE is ignored"
            );
        }
        on
    });
    on || test_on()
}

/// [`on`], for `victory-hle`: whether it sends the title's `VirtualFree`
/// and `VirtualProtect` here and tells regions ([`title_memory_region`]).
pub fn memory_core_on() -> bool {
    on()
}

// The tests' switches, per thread: a test that turns the core on turns it on
// for the lookups it makes itself, not for the tests running beside it.
#[cfg(test)]
thread_local! {
    static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static NO_DEVICE13: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_on() -> bool {
    // (And the relay's threads, landing a test's flights.)
    TEST_ON.with(std::cell::Cell::get) || crate::host::test_relay_sees()
}

#[cfg(not(test))]
fn test_on() -> bool {
    false
}

/// The most title memory pinned by homes, in bytes
/// (`VICTORY_MEMORY_CORE_MB`, 4,096): opening a part locks its pages into
/// the PC's memory, those the title never touched too.
fn most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| most_of(std::env::var("VICTORY_MEMORY_CORE_MB").ok().as_deref()))
}

/// [`most`] for a setting: MB, 4,096 unless it is a number.
fn most_of(setting: Option<&str>) -> u64 {
    setting
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(4096)
        .saturating_mul(1 << 20)
}

/// Title memory promised to homes, in bytes (the 64 KiB pages each pins).
static RESERVED: AtomicU64 = AtomicU64::new(0);

/// What [`classify`] is told about the process before it looks at a range.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gate {
    pub(crate) on: bool,
    /// Late waits or the relay: submissions are not all waited for.
    pub(crate) late: bool,
    /// R5 (`crate::host::r5`): the flights land what homes need, so late
    /// submissions do not keep ranges out of homes.
    pub(crate) flights_safe: bool,
    /// The device can open parts of an allocation (`ID3D12Device13`).
    pub(crate) device13: bool,
    /// The host refused three openings in a row with the same error.
    pub(crate) latched: bool,
    /// The memory pool is on (`crate::pool`): view memory in its chunks
    /// may have homes.
    pub(crate) pool: bool,
}

impl Gate {
    fn process() -> Gate {
        Gate {
            on: on(),
            late: crate::host::late(),
            flights_safe: crate::host::r5(),
            device13: device13(),
            latched: LATCHED.load(Ordering::Relaxed),
            pool: crate::pool::on(),
        }
    }
}

/// Whether the host device is an `ID3D12Device13`: asked once, when the
/// first range is classified (`false` before there is a device).
fn device13() -> bool {
    #[cfg(test)]
    if NO_DEVICE13.with(std::cell::Cell::get) {
        return false;
    }
    match DEVICE13.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let Some(host) = crate::host::host() else {
                return false;
            };
            let yes = host.device.cast::<ID3D12Device13>().is_ok();
            DEVICE13.store(if yes { 1 } else { 2 }, Ordering::Relaxed);
            if !yes && on() {
                tracing::warn!(
                    "memory core: the device cannot open parts of an allocation (no ID3D12Device13); nothing is placed in place, every range keeps its copy"
                );
            }
            yes
        }
    }
}

/// What [`device13`] found: 0 not asked yet, 1 yes, 2 no.
static DEVICE13: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Whether something that stops every home of the core is known to hold
/// now: late waits or the relay without R5, a device that is no
/// `ID3D12Device13` (once asked), the host's refusals latched. Creates no
/// device (`crate::pool::memory_pool_homes_possible`).
pub(crate) fn homes_ruled_out() -> bool {
    ruled_out().is_some()
}

/// [`homes_ruled_out`], with the reason.
pub(crate) fn ruled_out() -> Option<Why> {
    if crate::host::late() && !crate::host::r5() {
        Some(Why::Late)
    } else if DEVICE13.load(Ordering::Relaxed) == 2 {
        Some(Why::NoDevice13)
    } else if LATCHED.load(Ordering::Relaxed) {
        Some(Why::Latched)
    } else {
        None
    }
}

/// Set once the host refused three openings in a row with the same error:
/// nothing more is asked of it this run.
static LATCHED: AtomicBool = AtomicBool::new(false);

/// Why a range keeps a copy (or, for [`Why::Refused`], why its opening came
/// to nothing). Counted in [`TURNED_DOWN`] in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// The switch is off (not counted).
    Off,
    Late,
    NoDevice13,
    Latched,
    /// A home holds part of the range only.
    Overlap,
    /// View memory (the page pool), with the memory pool off
    /// (`crate::pool`).
    View,
    /// View memory not one run of one section: views of several places,
    /// or holes between them.
    Scattered,
    /// View memory of a section that is no chunk of the memory pool.
    NotPool,
    /// View memory of a chunk of the pool made reserved (no room in the
    /// PC's commit charge, or an old Windows).
    ReservedChunk,
    /// View memory of a chunk the host refused to open as a heap.
    ChunkRefused,
    /// View memory of an unopened chunk while the shared part of the card's
    /// budget has no room for it (`crate::pool::fits_shared`); asked again
    /// later.
    OverBudget,
    /// View memory of an unopened chunk the title maps less than
    /// `VICTORY_MEMORY_POOL_OPEN_PCT` of (`crate::pool::home`); asked again
    /// later.
    BelowShare,
    /// View memory whose run reaches past its chunk's end: the census of
    /// views and the chunk table disagree (never expected).
    PastChunk,
    /// View memory whose mapping the runtime is changing right now.
    Remapping,
    /// Not inside an `XMemVirtualAlloc` region (the process heap, memory
    /// from elsewhere).
    NotRegion,
    /// The region's protection was refused while a home stood, or a free
    /// of the title's there was given up.
    Blacklisted,
    /// A free or protection of the title's in the region is under way or
    /// waiting.
    Busy,
    /// Not private memory of the region committed read-write over the span.
    NotCommitted,
    Budget,
    Refused,
}

const WHYS: [(Why, &str); 19] = [
    (Why::Late, "late waits"),
    (Why::NoDevice13, "no ID3D12Device13"),
    (Why::Latched, "after the host's refusals"),
    (Why::Overlap, "partly in a home"),
    (Why::View, "view memory"),
    (Why::Scattered, "view memory from several places"),
    (Why::NotPool, "view memory in no pool chunk"),
    (Why::ReservedChunk, "view memory in a reserved pool chunk"),
    (
        Why::ChunkRefused,
        "view memory in a pool chunk the host refused",
    ),
    (
        Why::OverBudget,
        "view memory in a pool chunk the shared budget has no room for",
    ),
    (
        Why::BelowShare,
        "view memory in a pool chunk mapped below VICTORY_MEMORY_POOL_OPEN_PCT",
    ),
    (Why::PastChunk, "view memory past its pool chunk's end"),
    (Why::Remapping, "view memory being remapped"),
    (Why::NotRegion, "not an XMemVirtualAlloc region"),
    (Why::Blacklisted, "blacklisted region"),
    (Why::Busy, "region busy with a title free or protection"),
    (Why::NotCommitted, "not committed read-write"),
    (Why::Budget, "over VICTORY_MEMORY_CORE_MB"),
    (Why::Refused, "refused by the host"),
];

/// Ranges that kept a copy since the start, by [`WHYS`].
static TURNED_DOWN: [AtomicU64; WHYS.len()] = [const { AtomicU64::new(0) }; WHYS.len()];

fn turned_down(why: Why) {
    if let Some(i) = why_index(why) {
        TURNED_DOWN[i].fetch_add(1, Ordering::Relaxed);
    }
}

/// How many reasons [`WHYS`] counts (`crate::memory` counts its own by them).
pub(crate) const WHY_COUNT: usize = WHYS.len();

/// `why`'s place in [`WHYS`] (none for [`Why::Off`]).
pub(crate) fn why_index(why: Why) -> Option<usize> {
    WHYS.iter().position(|w| w.0 == why)
}

/// The words of the reason at `i` of [`WHYS`].
pub(crate) fn why_label(i: usize) -> &'static str {
    WHYS.get(i).map_or("", |w| w.1)
}

/// Ranges that kept a copy for `why` since the start.
pub(crate) fn turned_down_count(why: Why) -> u64 {
    why_index(why).map_or(0, |i| TURNED_DOWN[i].load(Ordering::Relaxed))
}

/// Since the start: homes made (windows, buffers); asks that joined a home
/// that holds them; big buffers given a card copy when asked first; opens
/// and their microseconds.
static MADE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Since the start: the title's frees of GPU regions, of them in regions
/// that had a home, done later, their milliseconds of waiting, done later
/// that failed; the title's protections of GPU regions, of them in regions
/// that had a home, refused; GPU writes into homes told to copies; frees
/// that waited given up ([`GIVE_UP_MS`]); homes a protection took out of
/// `crate::memory`'s directory at once; with R5, protections refused at
/// first that waited for the flights in the air, and their microseconds.
static TOLD: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

/// How long a free of the title's waits for the heaps over its pages to go
/// before it is given up ([`run_pending`]).
const GIVE_UP_MS: u64 = 30_000;

/// What a home over the range is, for the one-home rule.
pub(crate) enum Held<H> {
    None,
    /// One home holds all of it.
    Whole(H),
    /// Homes hold part of it.
    Part,
}

/// What the title's views say of a range ([`Ask::view`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Viewed<P> {
    /// No view holds it: private memory, or memory of no one's.
    No,
    /// One run of one committed chunk of the memory pool, opened: its home
    /// there (C-pool).
    Pool(P),
    /// View memory that keeps its copy, and why.
    Copy(Why),
}

/// What [`classify`] asks, in its order: each only once it is needed.
pub(crate) trait Ask {
    type Home;
    /// A home in a chunk of the memory pool.
    type Pool;
    /// A home over the range ([`Held`]).
    fn holder(&mut self) -> Held<Self::Home>;
    /// For a big buffer: whether the card took a copy of it.
    fn card(&mut self) -> bool;
    /// Whether the range is view memory, and if so whether it has a home in
    /// a chunk of the pool.
    fn view(&mut self) -> Viewed<Self::Pool>;
    /// The region that holds the range's pages: `None` if none does,
    /// `Some(Err(why))` if it may not be given a home now
    /// ([`Why::Blacklisted`], [`Why::Busy`]).
    fn region(&mut self) -> Option<Result<(), Why>>;
    /// Whether the pages are the region's, committed read-write.
    fn committed(&mut self) -> bool;
    /// Claims the pages for a home: the region's say, asked again
    /// ([`Why::NotRegion`], [`Why::Blacklisted`], [`Why::Busy`]), then the
    /// budget ([`Why::Budget`]).
    fn claim(&mut self) -> Result<(), Why>;
}

/// What [`classify`] decides for a range.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Class<H, P> {
    /// Today's copy (X).
    Copy(Why),
    /// The card took a copy (G): the copy path, with that card side.
    Card,
    /// The home that holds the range (C, joined).
    Home(H),
    /// Claimed: open the pages and place a buffer (C, new).
    Place,
    /// A new home in a chunk of the memory pool (C-pool).
    Pool(P),
}

/// The rules of the memory core, in order. `big`: a title buffer large
/// enough to be offered a card copy.
///
/// 1. The process: the switch on, every submission waited for (or the
///    flights landing what homes need, [`Gate::flights_safe`]), a device
///    that opens parts of an allocation, the host not latched off.
/// 2. A big buffer is offered a copy in the card's memory first (G): the
///    card reads it there far faster than from the PC's memory, also when a
///    home (a window made first over its bytes) holds it. A card copy is a
///    copy, which may lie over a home ([`after_wait`] keeps it current).
/// 3. One home per byte: a range that one home holds whole is that home; a
///    range that homes hold only part of keeps a copy (a second home over the
///    same bytes, or one over two homes, could not be one buffer).
/// 4. View memory (the page pool's views) that is one run of one chunk of
///    the memory pool, committed and opened, is a new home in that chunk
///    (C-pool, `crate::pool`); other view memory keeps its copy.
/// 5. The range's pages lie in one `XMemVirtualAlloc` region that the title
///    is not freeing or protecting ([`Why::Busy`]) and whose protection was
///    never refused ([`Why::Blacklisted`]).
/// 6. Those pages are private memory of that allocation, committed
///    read-write.
/// 7. The pages fit in `VICTORY_MEMORY_CORE_MB`: they are claimed, and the
///    caller opens them.
pub(crate) fn classify<A: Ask>(gate: Gate, big: bool, ask: &mut A) -> Class<A::Home, A::Pool> {
    if !gate.on {
        return Class::Copy(Why::Off);
    }
    if gate.late && !gate.flights_safe {
        return Class::Copy(Why::Late);
    }
    if !gate.device13 {
        return Class::Copy(Why::NoDevice13);
    }
    if gate.latched {
        return Class::Copy(Why::Latched);
    }
    if big && ask.card() {
        return Class::Card;
    }
    match ask.holder() {
        Held::Whole(home) => return Class::Home(home),
        Held::Part => return Class::Copy(Why::Overlap),
        Held::None => {}
    }
    match ask.view() {
        Viewed::No => {}
        Viewed::Pool(home) => return Class::Pool(home),
        Viewed::Copy(why) => return Class::Copy(why),
    }
    match ask.region() {
        None => return Class::Copy(Why::NotRegion),
        Some(Err(why)) => return Class::Copy(why),
        Some(Ok(())) => {}
    }
    if !ask.committed() {
        return Class::Copy(Why::NotCommitted);
    }
    if let Err(why) = ask.claim() {
        return Class::Copy(why);
    }
    Class::Place
}

/// An `XMemVirtualAlloc` reservation.
struct Region {
    /// Bytes (rounded to the host's page).
    len: usize,
    /// The console's words as the title gave them: the allocation kind (page
    /// size bits included), the XMem flags, the protection (memory type and
    /// GPU access bits above the host's 0x1ff included).
    kind: u32,
    flags: u64,
    protect: u32,
    /// Homes ever placed in it: a free or protection of a region that never
    /// had one goes straight to the host.
    homes_ever: u64,
    /// Frees and protections of the title under way or waiting: no home is
    /// placed meanwhile.
    busy: u32,
    /// A protection was refused while a home stood: no home again.
    blacklisted: bool,
    /// One of our own heaps (slice 3, [`memory_own_alloc`]), not the
    /// title's.
    own: bool,
    /// Which region this is, of all ever told: a release leaves the table
    /// only with the region it freed ([`released`]), not one of ours made
    /// at the same base right after the host freed it.
    serial: u64,
}

impl Region {
    /// Whether a home may be claimed in the region now.
    fn may_home(&self) -> Result<(), Why> {
        if self.blacklisted {
            Err(Why::Blacklisted)
        } else if self.busy > 0 {
            Err(Why::Busy)
        } else {
            Ok(())
        }
    }
}

/// A free of the title's the host refused while a heap stood, done later:
/// its arguments, the pages it frees, its region, when it was asked for.
struct Pending {
    at: usize,
    len: usize,
    kind: u32,
    pages: (usize, usize),
    region: usize,
    /// The region's [`Region::serial`].
    serial: u64,
    since: u64,
}

/// The core's tables, behind one lock taken last: nothing is called with it
/// held that takes another lock or drops a [`Shadow`].
struct Core {
    /// By base address.
    regions: BTreeMap<usize, Region>,
    /// The 64 KiB pages each home pins, as (first, bytes), with how many
    /// homes pin exactly those: from the claim to the home's end.
    live: BTreeMap<(u64, u64), u32>,
    /// The homes in the shadow tables by the title address they start at:
    /// their length, number and the home.
    tabled: BTreeMap<u64, Vec<(u64, u64, Weak<Shadow>)>>,
    /// The longest entry of `tabled` ever.
    tabled_longest: u64,
    pending: Vec<Pending>,
    /// Title memory a free or a remap let the homes over go from, whose
    /// homes in `crate::memory`'s directory go at the next sweep
    /// ([`run_pending`]), when no submission runs: a shader of one running
    /// now may read them.
    unnumber: Vec<(u64, u64)>,
    /// Slice 4 (`crate::card_home`): copies in the card's memory let go with
    /// their home, which go out of the directory at the same sweep, each by
    /// itself (not by range: theirs may lie over other homes still in use).
    unnumber_cards: Vec<Weak<Shadow>>,
}

static CORE: Mutex<Core> = Mutex::new(Core {
    regions: BTreeMap::new(),
    live: BTreeMap::new(),
    tabled: BTreeMap::new(),
    tabled_longest: 0,
    pending: Vec::new(),
    unnumber: Vec::new(),
    unnumber_cards: Vec::new(),
});

fn core() -> std::sync::MutexGuard<'static, Core> {
    CORE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Held by [`run_pending`] from taking the waiting frees to their last host
/// call, and by a commit of the title's inside a region from cancelling the
/// waiting decommits of its pages to the host's commit
/// ([`title_memory_committing`]): a decommit that waited never runs after
/// the title committed its pages again. Held too by a release of the
/// title's ([`title_virtual_free`]) and by a new region
/// ([`title_memory_region`]) while they drop the waiting frees of a region
/// that ended: none of those runs against a reservation made at the same
/// addresses since. Taken before any other lock.
static FREEING: Mutex<()> = Mutex::new(());

fn freeing() -> std::sync::MutexGuard<'static, ()> {
    FREEING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Frees waiting in [`Core::pending`] (read without the lock).
static PENDING: AtomicU64 = AtomicU64::new(0);

impl Core {
    /// The region holding `at`: its base and itself.
    fn region_at(&mut self, at: usize) -> Option<(usize, &mut Region)> {
        let (&base, region) = self.regions.range_mut(..=at).next_back()?;
        (at < base.saturating_add(region.len)).then_some((base, region))
    }
}

/// `XMemVirtualAlloc` reserved `[at, at + len)` (with the console's words
/// as the title gave them). A region already in the table that overlaps is
/// gone (the title freed it some way the hooks did not see), and its frees
/// that wait go with it.
///
/// shortcut: a decommit of the title's refused while a release of the same
/// region on another thread is done, and the addresses reserved again
/// meanwhile, can put its wait in the list after the new region came; the
/// title would have to free one region on two threads at once.
pub fn title_memory_region(at: usize, len: usize, kind: u32, flags: u64, protect: u32) {
    if len == 0 {
        return;
    }
    let len = len.next_multiple_of(PAGE_4K);
    let end = at.saturating_add(len);
    // No waiting free runs meanwhile ([`run_pending`]): one of a stale
    // region would free the new region's pages.
    let _freeing = freeing();
    let mut core = core();
    let stale: Vec<usize> = core
        .regions
        .range(..end)
        .rev()
        .take_while(|(base, region)| base.saturating_add(region.len) > at)
        .map(|(base, _)| *base)
        .collect();
    // The frees of a stale region, or over these pages, that wait: the host
    // freed those pages already, since it gave them out again.
    drop_pending(&mut core, |p| {
        stale.contains(&p.region) || (p.pages.0 < end && at < p.pages.1)
    });
    for base in stale {
        core.regions.remove(&base);
    }
    core.regions
        .insert(at, Region::new(len, kind, flags, protect, false));
}

impl Region {
    fn new(len: usize, kind: u32, flags: u64, protect: u32, own: bool) -> Region {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        Region {
            serial: SERIAL.fetch_add(1, Ordering::Relaxed),
            len,
            kind,
            flags,
            protect,
            homes_ever: 0,
            busy: 0,
            blacklisted: false,
            own,
        }
    }
}

/// The homes in the tables, for the one-home rule: `home` was put there.
pub(crate) fn tabled(home: &Arc<Shadow>) {
    let (start, len) = home.span();
    let mut core = core();
    core.tabled_longest = core.tabled_longest.max(len);
    core.tabled
        .entry(start)
        .or_default()
        .push((len, home.id(), Arc::downgrade(home)));
}

/// What home over title memory `[start, end)` the tables hold.
fn holder(start: u64, end: u64) -> Held<Arc<Shadow>> {
    let core = core();
    let mut part = false;
    for (&first, homes) in core
        .tabled
        .range(start.saturating_sub(core.tabled_longest)..end)
        .rev()
    {
        for (len, _, home) in homes {
            if first + len <= start {
                continue;
            }
            if first <= start
                && first + len >= end
                && let Some(home) = home.upgrade()
            {
                return Held::Whole(home);
            }
            part = true;
        }
    }
    if part { Held::Part } else { Held::None }
}

/// The pages a home over title memory `[first, first + len)` pins: the
/// 64 KiB pages its heap is opened over ([`resolve`]), which the host's
/// frees and protections of any of their 4 KiB pages run into.
pub(crate) fn pinned(first: u64, len: u64) -> (u64, u64) {
    let (at, bytes) = crate::inplace::aligned_span(first, len).unwrap_or((first, len));
    (at, at.saturating_add(bytes))
}

/// Takes the homes whose pinned pages ([`pinned`]) lie over `[at, end)`
/// out of [`Core::tabled`] and returns them (alive).
fn untable_over(at: u64, end: u64) -> Vec<Arc<Shadow>> {
    const K64: u64 = 64 << 10;
    let mut core = core();
    // A home's pinned pages start up to 64 KiB before its first byte and
    // end up to 64 KiB after its last.
    let firsts: Vec<u64> = core
        .tabled
        .range(at.saturating_sub(core.tabled_longest.saturating_add(K64))..end.saturating_add(K64))
        .map(|(&first, _)| first)
        .collect();
    let mut taken = Vec::new();
    for first in firsts {
        let Some(homes) = core.tabled.get_mut(&first) else {
            continue;
        };
        homes.retain(|&(len, _, ref home)| {
            let (from, to) = pinned(first, len);
            if to <= at || from >= end {
                return true;
            }
            taken.extend(home.upgrade());
            false
        });
        if homes.is_empty() {
            core.tabled.remove(&first);
        }
    }
    taken
}

/// The asks of [`classify`] for a range of the live tables.
struct LiveAsk<'a> {
    start: u64,
    end: u64,
    /// The 64 KiB pages under the range.
    at: u64,
    bytes: u64,
    /// The region found by `region`.
    base: usize,
    card: &'a mut dyn FnMut() -> bool,
    /// The memory pool is on ([`Gate::pool`]).
    pool: bool,
    /// For a home in a chunk of the pool: the mapping of the range held
    /// still from the look at its views until the home is in the tables
    /// ([`New::home`]), so that a remap of it either comes first (and the
    /// views say what it shows now) or lets the home go after.
    reading: Option<crate::watch::Reading>,
}

impl Ask for LiveAsk<'_> {
    type Home = Arc<Shadow>;
    type Pool = crate::inplace::Placed;

    fn holder(&mut self) -> Held<Arc<Shadow>> {
        holder(self.start, self.end)
    }

    fn card(&mut self) -> bool {
        (self.card)()
    }

    fn view(&mut self) -> Viewed<crate::inplace::Placed> {
        use crate::inplace::{Cover, cover};
        let (start, len) = (self.start as usize, (self.end - self.start) as usize);
        match cover(start, len) {
            Cover::None => return Viewed::No,
            _ if !self.pool => return Viewed::Copy(Why::View),
            Cover::Holes | Cover::Scattered => return Viewed::Copy(Why::Scattered),
            Cover::Run(..) => {}
        }
        // Not waited for: the caller holds the shadow tables, which a remap
        // under way takes to let homes go (`crate::shadow::let_go_homes`).
        let Some(reading) = crate::watch::title_memory_reading_now(start, len) else {
            return Viewed::Copy(Why::Remapping);
        };
        // Asked again with the mapping held still.
        let Cover::Run(section, offset) = cover(start, len) else {
            return Viewed::Copy(Why::Scattered);
        };
        match crate::pool::home(section, offset, self.end - self.start) {
            Ok(placed) => {
                self.reading = Some(reading);
                Viewed::Pool(placed)
            }
            Err(why) => Viewed::Copy(why),
        }
    }

    fn region(&mut self) -> Option<Result<(), Why>> {
        let mut core = core();
        let (base, region) = core.region_at(self.at as usize)?;
        if (self.at + self.bytes) as usize > base + region.len {
            return None;
        }
        self.base = base;
        Some(region.may_home())
    }

    fn committed(&mut self) -> bool {
        region_whole(self.base, self.at as usize, self.bytes as usize)
    }

    fn claim(&mut self) -> Result<(), Why> {
        let mut core = core();
        // Asked again under the core's lock, which a free or protection of
        // the title's in the region takes to mark it busy before it calls the
        // host ([`busy`]): either this claim comes first, and the call finds
        // `homes_ever` raised and goes through the protocol, or the call
        // marked the region first and nothing is claimed in it until the
        // call is done.
        match core.region_at(self.base) {
            Some((base, region)) if base == self.base => region.may_home()?,
            // Released or replaced since `region` found it.
            _ => return Err(Why::NotRegion),
        }
        if !crate::inplace::try_reserve_at(&RESERVED, self.bytes, most()) {
            return Err(Why::Budget);
        }
        if let Some((_, region)) = core.region_at(self.base) {
            region.homes_ever += 1;
        }
        *core.live.entry((self.at, self.bytes)).or_default() += 1;
        Ok(())
    }
}

/// Whether `[at, at + bytes)` is private memory of the allocation at
/// `base`, committed read-write throughout: what a part may be opened on.
/// One question to Windows, as `crate::inplace::private_whole`, which this
/// adds the allocation's base to: the region table may hold a region the
/// title freed some way the hooks did not see.
fn region_whole(base: usize, at: usize, bytes: usize) -> bool {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0
        && info.AllocationBase as usize == base
        && info.State == mem::MEM_COMMIT
        && info.Type == mem::MEM_PRIVATE
        && info.Protect == mem::PAGE_READWRITE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at.saturating_add(bytes)
}

/// Gives back a claim whose opening came to nothing, or a home's at its end.
fn unclaim(at: u64, bytes: u64) {
    {
        let mut core = core();
        if let Some(count) = core.live.get_mut(&(at, bytes)) {
            *count -= 1;
            if *count == 0 {
                core.live.remove(&(at, bytes));
            }
        }
    }
    crate::inplace::release_at(&RESERVED, bytes);
}

/// The last refusal of the host and how many in a row had its code (an
/// opening that succeeds ends the row).
static REFUSALS: Mutex<(i32, u32)> = Mutex::new((0, 0));

/// An opening succeeded: the row of refusals ends.
fn opened() {
    *REFUSALS.lock().unwrap_or_else(|e| e.into_inner()) = (0, 0);
}

fn refused(error: &windows::core::Error, start: u64, len: u64) {
    turned_down(Why::Refused);
    let mut last = REFUSALS.lock().unwrap_or_else(|e| e.into_inner());
    let code = error.code().0;
    *last = if last.0 == code {
        (code, last.1 + 1)
    } else {
        (code, 1)
    };
    let n = TURNED_DOWN[WHYS.len() - 1].load(Ordering::Relaxed);
    if n <= 8 {
        tracing::warn!(start = format_args!("{start:#x}"), len, error = %error, "memory core: title memory not opened in place; a copy instead");
    }
    if last.1 >= 3 && !LATCHED.swap(true, Ordering::Relaxed) {
        tracing::warn!(
            code = format_args!("{code:#010x}"),
            "memory core: the host refused three openings in a row with the same error; nothing more is opened this run"
        );
    }
}

/// The kinds of home.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// C: a part of an `XMemVirtualAlloc` region opened as a heap, its pages
    /// pinned and counted in `VICTORY_MEMORY_CORE_MB`.
    Private,
    /// C-pool: a run of a chunk of the memory pool, in the chunk's heap
    /// (`crate::pool`), which pins nothing.
    Pool,
}

/// What a range of title memory resolves to.
pub(crate) enum Found {
    /// The home that holds it.
    Home(Arc<Shadow>),
    /// A new home, for the caller to make ([`New::home`]).
    New(New),
    /// The card took a copy: the copy path, with the card side it made.
    Card,
    /// Today's copy.
    Copy,
}

/// A new home, opened and placed, not yet in the tables.
pub(crate) struct New {
    placed: crate::inplace::Placed,
    kind: Kind,
    /// For [`Kind::Pool`]: the mapping held still ([`LiveAsk::reading`]).
    _reading: Option<crate::watch::Reading>,
}

impl New {
    /// The home's `Shadow` for title memory `[start, start + len)` (a window
    /// if `window`), in the tables of the one-home rule ([`tabled`]); then
    /// the mapping may change again.
    pub(crate) fn home(self, start: u64, len: u64, window: bool) -> Arc<Shadow> {
        let home = Arc::new(Shadow::placed(
            self.placed,
            start,
            len,
            window,
            Some(self.kind),
        ));
        tabled(&home);
        home
    }
}

/// The home of title memory `[start, start + len)` (a window if `window`):
/// what [`classify`] decides, and the opening if it is a new one. `card` is
/// asked only for a `big` buffer, and only when the rules come to it.
pub(crate) fn resolve(
    start: u64,
    len: u64,
    window: bool,
    big: bool,
    card: &mut dyn FnMut() -> bool,
) -> Found {
    // After the release at the title's end no home is made or found.
    // (Asked without the core's lock: a resolve that passed this just
    // before the flag was set can table its home after `release_all`. It is
    // a leak, not a fault: no submission runs then, and the home keeps its
    // heap and claim, counted as `claims_left` or `chunks_held`.)
    if crate::release::released() {
        return Found::Copy;
    }
    resolve_with(Gate::process(), start, len, window, big, card)
}

/// [`resolve`] with the gate given (tests).
fn resolve_with(
    gate: Gate,
    start: u64,
    len: u64,
    window: bool,
    big: bool,
    card: &mut dyn FnMut() -> bool,
) -> Found {
    let Some(end) = start.checked_add(len.max(1)) else {
        return Found::Copy;
    };
    let Some((at, bytes)) = crate::inplace::aligned_span(start, len) else {
        return Found::Copy;
    };
    let mut ask = LiveAsk {
        start,
        end,
        at,
        bytes,
        base: 0,
        card,
        pool: gate.pool,
        reading: None,
    };
    match classify(gate, big, &mut ask) {
        Class::Copy(why) => {
            if why != Why::Off {
                turned_down(why);
            }
            Found::Copy
        }
        Class::Card => {
            MADE[3].fetch_add(1, Ordering::Relaxed);
            Found::Card
        }
        Class::Home(home) => {
            MADE[2].fetch_add(1, Ordering::Relaxed);
            Found::Home(home)
        }
        Class::Pool(placed) => {
            crate::pool::made(len);
            Found::New(New {
                placed,
                kind: Kind::Pool,
                _reading: ask.reading.take(),
            })
        }
        Class::Place => match open_home(start, at, bytes) {
            Ok(placed) => {
                opened();
                MADE[usize::from(!window)].fetch_add(1, Ordering::Relaxed);
                Found::New(New {
                    placed,
                    kind: Kind::Private,
                    _reading: None,
                })
            }
            Err(error) => {
                unclaim(at, bytes);
                // A free or protection of the title's that came in after the
                // claim is not the host refusing to open: not in the row.
                let went_busy = core()
                    .region_at(ask.base)
                    .is_none_or(|(base, region)| base != ask.base || region.busy > 0);
                match error {
                    _ if went_busy => turned_down(Why::Busy),
                    Some(error) => refused(&error, start, len),
                    None => turned_down(Why::Refused),
                }
                Found::Copy
            }
        },
    }
}

/// Opens the pages `[at, at + bytes)` as a heap and places one buffer over
/// all of them; `start` is the title's first byte in them.
fn open_home(
    start: u64,
    at: u64,
    bytes: u64,
) -> Result<crate::inplace::Placed, Option<windows::core::Error>> {
    let host = crate::host::host().ok_or(None)?;
    let begun = std::time::Instant::now();
    let opened = crate::inplace::open(
        &host.device,
        crate::inplace::From::Part(at as usize, bytes as usize),
    )
    .and_then(|heap| {
        crate::inplace::place(&host.device, &heap, 0, bytes).map(|buffer| (heap, buffer))
    });
    MADE[4].fetch_add(1, Ordering::Relaxed);
    MADE[5].fetch_add(begun.elapsed().as_micros() as u64, Ordering::Relaxed);
    let (heap, buffer) = opened.map_err(Some)?;
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() } + (start - at);
    Ok(crate::inplace::Placed {
        buffer,
        heap,
        gpu,
        bias: start - at,
    })
}

/// A home's `Shadow` is gone (`start`, `len`: its title span; `id`: its
/// number; `kind`: its kind): the tables forget it, and a home of
/// [`Kind::Private`] gives its pages back.
pub(crate) fn gone(start: u64, len: u64, id: u64, kind: Kind) {
    let Some((at, bytes)) = crate::inplace::aligned_span(start, len) else {
        return;
    };
    {
        let mut core = core();
        if let Some(homes) = core.tabled.get_mut(&start) {
            homes.retain(|home| home.1 != id);
            if homes.is_empty() {
                core.tabled.remove(&start);
            }
        }
    }
    match kind {
        Kind::Private => unclaim(at, bytes),
        Kind::Pool => crate::pool::gone(len),
    }
}

/// Something other than the GPU touched title memory `[at, at + len)`.
/// For [`Touch::Wrote`] nothing is done (a home is the memory itself;
/// copies keep the watch paths that called this). For the others every home
/// over the range is let go: out of the tables (the next ask makes a copy or
/// a home anew) and out of the directory of `crate::memory`; its heap goes
/// when no list and no slot hold it any more. Returns how many were let go.
///
/// The directory is read by shaders of a submission that may run now (this
/// is called on the title's threads): after a free or a remap its entries go
/// at the next sweep, when none runs ([`run_pending`]); the free waits for
/// that anyway, since the directory holds the home. A protection cannot
/// wait, so it takes them out at once, and is counted: a dispatch running
/// then, or recorded before and submitted after, that reads the home through
/// the directory takes the fault path for its pages (reads no title bytes
/// there) for that once.
pub(crate) fn cpu_touched(at: usize, len: usize, how: Touch) -> u64 {
    if how == Touch::Wrote {
        return 0;
    }
    // `VICTORY_INPLACE`'s placements (the core off).
    let gone = crate::inplace::let_go(at, len);
    if !on() {
        return gone;
    }
    gone + let_go_over(at, len, how == Touch::Protecting)
}

/// The homes over `[at, at + len)` are let go ([`cpu_touched`]); `now`:
/// out of `crate::memory`'s directory at once, else at the next sweep.
fn let_go_over(at: usize, len: usize, now: bool) -> u64 {
    let end = at.saturating_add(len) as u64;
    let mut homes = untable_over(at as u64, end);
    if homes.is_empty() {
        return 0;
    }
    // Copies in the card's memory whose CPU side is one of them
    // (`crate::card_home`) hold it: let go with it.
    let cards = crate::shadow::let_go_homes(&homes);
    if !now {
        // (Weak: held here they would hold the home's heap. The directory
        // holds a numbered one alive until the sweep forgets it.)
        core()
            .unnumber_cards
            .extend(cards.iter().map(Arc::downgrade));
    }
    homes.extend(cards);
    if now {
        for home in &homes {
            if crate::memory::forget(home) {
                TOLD[10].fetch_add(1, Ordering::Relaxed);
            }
        }
    } else {
        // A range, not the homes: held here they would hold their heaps.
        core().unnumber.push((at as u64, end));
    }
    let gone = homes.len() as u64;
    for home in homes {
        if home.home_kind() == Some(Kind::Pool) {
            crate::pool::let_go();
            // The view heap slots made on it are made anew at their next use.
            let (start, len) = home.span();
            crate::descriptor::host::pool_let_go(start, start.saturating_add(len));
        }
        crate::host::retire(home);
    }
    gone
}

/// At the title's end (`crate::release`): the buffers of every home in the
/// tables (their COM pointers), for the view heap slots that name them.
pub(crate) fn home_buffers() -> std::collections::HashSet<usize> {
    let homes: Vec<Arc<Shadow>> = core()
        .tabled
        .values()
        .flatten()
        .filter_map(|(_, _, home)| home.upgrade())
        .collect();
    // (Dropped without the core's lock: a home's end takes it.)
    homes
        .iter()
        .map(|home| home.buffer.as_raw() as usize)
        .collect()
}

/// At the title's end (`crate::release`): every home is let go as a
/// protection lets the homes over its pages go (out of the tables and the
/// directory at once, retired), and so are `VICTORY_INPLACE`'s placements,
/// every one of them. A home of [`Kind::Private`] gives its claim back when
/// its `Shadow` goes ([`gone`]). Not asked of [`on`]: with the core off the
/// tables are empty. Returns how many were let go.
pub(crate) fn release_all() -> u64 {
    crate::inplace::let_go_all(0, usize::MAX) + let_go_over(0, usize::MAX, true)
}

/// The claims of homes of [`Kind::Private`] still standing, and their
/// bytes: homes something still holds (`crate::release` reports them).
pub(crate) fn claims_left() -> (u64, u64) {
    let core = core();
    core.live
        .iter()
        .fold((0, 0), |(n, bytes), (&(_, len), &count)| {
            (n + u64::from(count), bytes + len * u64::from(count))
        })
}

/// The homes over the memory frees and remaps let homes go from since the
/// last sweep go out of `crate::memory`'s directory (at the sweep: no
/// submission runs).
fn unnumber() {
    let ranges = std::mem::take(&mut core().unnumber);
    for (from, to) in ranges {
        crate::memory::forget_homes_over(from, to);
    }
    // Slice 4: the card copies let go with their homes, each by itself.
    let cards = std::mem::take(&mut core().unnumber_cards);
    for card in cards.iter().filter_map(Weak::upgrade) {
        crate::memory::forget(&card);
    }
}

/// The pages of a title `VirtualFree(at, len, kind)` in a region, and the
/// region's base and serial; `None` if they are not a region's, the call is of a kind
/// that never touches a home (placeholders), or the host would refuse it
/// for its own shape whatever stood there: a release that is not of the
/// whole allocation from its base, a decommit of length 0 not at the base,
/// a range past the region's end or over more than its one allocation.
/// Such a call goes to the host as it is, and its refusal is the title's.
fn free_span(at: usize, len: usize, kind: u32) -> Option<(usize, usize, usize, u64)> {
    let (base, region_end, serial) = {
        let mut core = core();
        let (base, region) = core.region_at(at)?;
        (base, base + region.len, region.serial)
    };
    let (from, to) = match kind {
        mem::MEM_RELEASE if at == base && len == 0 => (base, region_end),
        mem::MEM_DECOMMIT if len == 0 && at == base => (base, region_end),
        mem::MEM_DECOMMIT if len > 0 => (
            at / PAGE_4K * PAGE_4K,
            at.checked_add(len)?.next_multiple_of(PAGE_4K),
        ),
        _ => return None,
    };
    if to > region_end {
        return None;
    }
    (allocation_of(from) == Some(base) && allocation_of(to - 1) == Some(base))
        .then_some((from, to, base, serial))
}

/// The base of the host allocation holding `at` (one question to Windows).
fn allocation_of(at: usize) -> Option<usize> {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    (n != 0 && info.State != mem::MEM_FREE).then_some(info.AllocationBase as usize)
}

/// Marks the region at `base` busy for a call of the title's to the host:
/// no home is claimed in it until [`unbusy`]. Returns whether the region is
/// one (then [`unbusy`] follows the call) and whether it ever had a home
/// (then the call goes through the core's protocol).
fn busy(base: usize, which: usize) -> (bool, bool) {
    let mut core = core();
    let Some((found, region)) = core.region_at(base) else {
        return (false, false);
    };
    if found != base {
        return (false, false);
    }
    // Our own heaps' frees are counted apart ([`own_free`]).
    let title = !region.own;
    if title {
        TOLD[which].fetch_add(1, Ordering::Relaxed);
    }
    region.busy += 1;
    if region.homes_ever == 0 {
        return (true, false);
    }
    if title {
        TOLD[which + 1].fetch_add(1, Ordering::Relaxed);
    }
    (true, true)
}

/// By base only: a region made at the same base since (after the host's
/// release) loses at most a busy count it never had, which saturates at 0.
fn unbusy(base: usize) {
    let mut core = core();
    if let Some((found, region)) = core.region_at(base)
        && found == base
    {
        region.busy = region.busy.saturating_sub(1);
    }
}

/// The title's `VirtualFree`, with the core on (`victory-hle` sends it
/// here). Memory that is no region's, or a region that never had a home,
/// goes straight to the host and its answer (last error included) is the
/// title's. Otherwise the homes over the pages are let go first
/// ([`cpu_touched`]); then the host. If it still refuses with error 87 (a
/// list recorded and not yet submitted, or a descriptor slot, holds a heap
/// there), the free waits in a list and is done at a sweep once nothing
/// holds the heap (or given up after [`GIVE_UP_MS`]), and the title is told
/// it was done (TRUE): the title's pages stay committed until then, and an
/// allocation of its at that very address would fail meanwhile. A call the
/// host would refuse for its own shape ([`free_span`]) never waits.
pub fn title_virtual_free(at: usize, len: usize, kind: u32) -> i32 {
    // SAFETY: the title's own arguments, for the host function it called.
    let host = || unsafe { mem::VirtualFree(at as *mut c_void, len, kind) };
    // SAFETY: a plain read of this thread's last error.
    let before = unsafe { GetLastError() };
    let Some((from, to, base, serial)) = free_span(at, len, kind) else {
        return host();
    };
    // A release ends the region and its waiting frees: from before the
    // host's call to the region's end in the table, no waiting free runs
    // ([`run_pending`]) and no region is told ([`title_memory_region`]), so
    // neither can meet a reservation of the title's made at these addresses
    // meanwhile. Taken first, as [`run_pending`] takes it.
    let _freeing = (kind == mem::MEM_RELEASE).then(freeing);
    match busy(base, 0) {
        (false, _) => return host(),
        (true, false) => {
            let done = host();
            // SAFETY: a plain read of this thread's last error.
            let error = unsafe { GetLastError() };
            unbusy(base);
            if done != 0 && kind == mem::MEM_RELEASE {
                released(base, serial);
            }
            // SAFETY: as the host left it.
            unsafe { SetLastError(error) };
            return done;
        }
        (true, true) => {}
    }
    cpu_touched(from, to - from, Touch::Freeing);
    let done = host();
    // SAFETY: a plain read of this thread's last error.
    let error = unsafe { GetLastError() };
    if done != 0 || error != REFUSED_BY_HEAP {
        unbusy(base);
        if done != 0 && kind == mem::MEM_RELEASE {
            released(base, serial);
        }
        // SAFETY: as the host left it.
        unsafe { SetLastError(error) };
        return done;
    }
    TOLD[2].fetch_add(1, Ordering::Relaxed);
    {
        let mut core = core();
        core.pending.push(Pending {
            at,
            len,
            kind,
            pages: (from, to),
            region: base,
            serial,
            since: crate::host::now_ms(),
        });
        PENDING.fetch_add(1, Ordering::Release);
    }
    // Done, as the title is told: its last error as it was before.
    // SAFETY: a plain write of this thread's last error.
    unsafe { SetLastError(before) };
    1
}

/// The region at `base` with serial `serial` was released: it leaves the
/// table, and the frees of it that wait go with it (the host freed their
/// pages with the rest). Called with [`FREEING`] held. A region of ours
/// ([`memory_own_alloc`], which does not take [`FREEING`]) made at the same
/// base once the host had freed it is another region and stays.
fn released(base: usize, serial: u64) {
    let mut core = core();
    if core.regions.get(&base).is_some_and(|r| r.serial == serial) {
        core.regions.remove(&base);
    }
    drop_pending(&mut core, |p| p.region == base && p.serial == serial);
}

/// Drops the waiting frees `which` picks, with their marks of their
/// regions' busy counts.
fn drop_pending(core: &mut Core, which: impl Fn(&Pending) -> bool) {
    let pending = std::mem::take(&mut core.pending);
    let (gone, kept): (Vec<Pending>, Vec<Pending>) = pending.into_iter().partition(&which);
    core.pending = kept;
    for p in &gone {
        if let Some((found, region)) = core.region_at(p.region)
            && found == p.region
        {
            region.busy = region.busy.saturating_sub(1);
        }
    }
    PENDING.fetch_sub(gone.len() as u64, Ordering::Release);
}

/// The title's `VirtualProtect`, with the core on. As
/// [`title_virtual_free`]; a protection cannot wait for a sweep, so when the
/// host still refuses after the homes were let go, what the retire queue
/// holds for finished submissions is dropped and it is asked once more;
/// with R5 (`crate::host::r5`) the flights in the air when it was refused
/// are waited for first, and the lists they ran let go, as those may be
/// what holds the heap (`crate::host::wait_for_flights_now`, bounded by what
/// is in the air, counted). Refused again, the title gets FALSE with error
/// 87 (what it got with `VICTORY_INPLACE`), it is counted, and the region is
/// never given a home again.
// `old` is the title's own pointer, given to the host function it called as
// the title gave it.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn title_virtual_protect(at: usize, len: usize, new: u32, old: *mut u32) -> i32 {
    // SAFETY: the title's own arguments, for the host function it called.
    let host = || unsafe { mem::VirtualProtect(at as *const c_void, len, new, old) };
    let base = {
        let mut core = core();
        // A protection past the region's end (over two regions) goes to the
        // host as it is: one region's busy mark could not cover it.
        let Some(base) = core
            .region_at(at)
            .filter(|(base, region)| at.checked_add(len).is_some_and(|e| e <= base + region.len))
            .map(|(base, _)| base)
        else {
            drop(core);
            return host();
        };
        base
    };
    match busy(base, 5) {
        (false, _) => return host(),
        (true, false) => {
            let done = host();
            // SAFETY: a plain read of this thread's last error.
            let error = unsafe { GetLastError() };
            unbusy(base);
            // SAFETY: as the host left it.
            unsafe { SetLastError(error) };
            return done;
        }
        (true, true) => {}
    }
    cpu_touched(at, len, Touch::Protecting);
    let mut done = host();
    // SAFETY: a plain read of this thread's last error.
    let mut error = unsafe { GetLastError() };
    if done == 0 && error == REFUSED_BY_HEAP {
        // Flights in the air may hold the heap (their lists, the homes let
        // go retired behind them): those are waited for, then let go.
        if let Some(us) = crate::host::wait_for_flights_now() {
            TOLD[11].fetch_add(1, Ordering::Relaxed);
            TOLD[12].fetch_add(us, Ordering::Relaxed);
        }
        crate::host::bury_finished();
        done = host();
        // SAFETY: as above.
        error = unsafe { GetLastError() };
    }
    if done == 0 && error == REFUSED_BY_HEAP {
        if TOLD[7].fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::warn!(
                at = format_args!("{at:#x}"),
                len,
                "memory core: the title's VirtualProtect was refused while a heap over its pages is held; the region gets no home again"
            );
        }
        let mut core = core();
        if let Some((found, region)) = core.region_at(base)
            && found == base
        {
            region.blacklisted = true;
        }
    }
    unbusy(base);
    // SAFETY: as the host left it.
    unsafe { SetLastError(error) };
    done
}

/// A commit of the title's inside a region under way
/// ([`title_memory_committing`]): while it lives no decommit that waited
/// runs.
#[must_use]
pub struct Committing(#[allow(dead_code)] Option<std::sync::MutexGuard<'static, ()>>);

/// `XMemVirtualAlloc` is about to commit `[at, at + len)` again, inside a
/// region (not a new reservation); the returned guard is held across the
/// host's commit. A decommit of those pages that waits
/// ([`title_virtual_free`]) must not run any more: it would take the pages
/// the title is to use. The title asked for them decommitted and then for
/// new pages, which read zeros: so they are zeroed (they never left), and
/// the wait keeps only what lies outside them. The waiting frees are run
/// under the same lock ([`run_pending`]), so one is either done before the
/// commit (which then commits the pages anew) or cut here.
///
/// shortcut: a commit through the host's `VirtualAlloc` itself, which the
/// title's imports bind to directly, is not seen; the title commits its GPU
/// memory through `XMemVirtualAlloc`.
pub fn title_memory_committing(at: usize, len: usize) -> Committing {
    // A free that waits was put in the list before the title was told it
    // was done, so before it could ask for these pages again.
    if PENDING.load(Ordering::Acquire) == 0 || len == 0 {
        return Committing(None);
    }
    let freeing = freeing();
    cancel_decommits(at, len);
    Committing(Some(freeing))
}

/// [`title_memory_committing`] with [`FREEING`] held.
fn cancel_decommits(at: usize, len: usize) {
    let (at, end) = (
        at / PAGE_4K * PAGE_4K,
        at.saturating_add(len).next_multiple_of(PAGE_4K),
    );
    let mut zero = Vec::new();
    {
        let mut core = core();
        let pending = std::mem::take(&mut core.pending);
        let mut kept = Vec::with_capacity(pending.len() + 1);
        for p in pending {
            let (from, to) = p.pages;
            if p.kind != mem::MEM_DECOMMIT || to <= at || from >= end {
                kept.push(p);
                continue;
            }
            let (a, b) = (from.max(at), to.min(end));
            zero.push((a, b));
            let pieces: Vec<(usize, usize)> = [(from, a), (b, to)]
                .into_iter()
                .filter(|&(x, y)| x < y)
                .collect();
            // One wait, one busy mark of its region: as many as it is cut in.
            if let Some((_, region)) = core.region_at(p.region) {
                region.busy = (region.busy + pieces.len() as u32).saturating_sub(1);
            }
            if pieces.is_empty() {
                PENDING.fetch_sub(1, Ordering::Relaxed);
            } else if pieces.len() == 2 {
                PENDING.fetch_add(1, Ordering::Relaxed);
            }
            for (x, y) in pieces {
                kept.push(Pending {
                    at: x,
                    len: y - x,
                    pages: (x, y),
                    ..p
                });
            }
        }
        core.pending = kept;
    }
    for (a, b) in zero {
        if region_rw(a, b - a) {
            // SAFETY: pages of the title's that its free never decommitted,
            // committed read-write.
            unsafe { std::ptr::write_bytes(a as *mut u8, 0, b - a) };
        }
    }
}

/// Whether `[at, at + len)` is committed read-write (one question).
fn region_rw(at: usize, len: usize) -> bool {
    // SAFETY: plain data filled in by the query.
    let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == mem::MEM_COMMIT
        && info.Protect == mem::PAGE_READWRITE
        && (info.BaseAddress as usize).saturating_add(info.RegionSize) >= at.saturating_add(len)
}

/// At the sweep, when no submission runs: the homes frees and remaps let go
/// leave `crate::memory`'s directory ([`cpu_touched`]), then the frees that
/// waited ([`title_virtual_free`]) whose heaps are gone are done. One still
/// refused after [`GIVE_UP_MS`] is given up: said and counted, its pages
/// zeroed if it was a decommit (a commit of them again reads zeros), and the
/// region given no home again (something still holds a heap there).
fn run_pending() {
    run_pending_after(GIVE_UP_MS);
}

/// With R5 (`crate::host::r5`), after flights landed and their lists were
/// let go (`crate::host`'s `land`): a free that waited may find its heaps
/// gone now, rather than at the next sweep, which behind flights comes only
/// every second and a half. Nothing if none waits. The directory's entries
/// over what frees and remaps let go stay for the sweep, when no submission
/// runs (flights may be in the air here): a free whose heap the directory
/// holds waits for that sweep as before. Takes [`FREEING`] and the core's
/// lock, as [`run_pending`] does; the caller holds no lock of ours.
pub(crate) fn after_landing() {
    if PENDING.load(Ordering::Acquire) == 0 {
        return;
    }
    run_pending_with(GIVE_UP_MS, false);
}

/// [`run_pending`], giving up after `give_up_ms` (tests).
fn run_pending_after(give_up_ms: u64) {
    run_pending_with(give_up_ms, true);
}

/// [`run_pending`], giving up after `give_up_ms`; `unnumbers`: the
/// directory's entries over what was let go are emptied first (at the
/// sweep).
fn run_pending_with(give_up_ms: u64, unnumbers: bool) {
    let _freeing = freeing();
    let due = if PENDING.load(Ordering::Acquire) == 0 {
        Vec::new()
    } else {
        std::mem::take(&mut core().pending)
    };
    // A home placed since (none should be: the region is busy).
    for p in &due {
        cpu_touched(p.pages.0, p.pages.1 - p.pages.0, Touch::Freeing);
    }
    if unnumbers {
        unnumber();
    }
    let mut kept: Vec<Pending> = Vec::new();
    // Regions a waiting release ended this pass: their other waiting frees
    // go with them (the host freed their pages), never to the host.
    let mut ended: Vec<usize> = Vec::new();
    for p in due {
        if ended.contains(&p.region) {
            PENDING.fetch_sub(1, Ordering::Release);
            continue;
        }
        // SAFETY: the title's own arguments, as it gave them.
        let done = unsafe { mem::VirtualFree(p.at as *mut c_void, p.len, p.kind) };
        // SAFETY: a plain read of this thread's last error.
        let error = unsafe { GetLastError() };
        let waited = crate::host::now_ms().saturating_sub(p.since);
        if done == 0 && error == REFUSED_BY_HEAP && waited < give_up_ms {
            kept.push(p);
            continue;
        }
        TOLD[3].fetch_add(waited, Ordering::Relaxed);
        if done == 0 && error == REFUSED_BY_HEAP {
            if TOLD[9].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(
                    at = format_args!("{:#x}", p.at),
                    len = p.len,
                    waited,
                    "memory core: a free of the title's still refused after it waited; given up, the region gets no home again"
                );
            }
            if p.kind == mem::MEM_DECOMMIT && region_rw(p.pages.0, p.pages.1 - p.pages.0) {
                // SAFETY: pages of the title's that it decommitted and that
                // stayed committed read-write.
                unsafe { std::ptr::write_bytes(p.pages.0 as *mut u8, 0, p.pages.1 - p.pages.0) };
            }
            let mut core = core();
            if let Some((found, region)) = core.region_at(p.region)
                && found == p.region
            {
                region.blacklisted = true;
            }
        } else if done == 0 {
            TOLD[4].fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                at = format_args!("{:#x}", p.at),
                error,
                "memory core: a free of the title's that waited failed"
            );
        }
        unbusy(p.region);
        PENDING.fetch_sub(1, Ordering::Release);
        if done != 0 && p.kind == mem::MEM_RELEASE {
            ended.push(p.region);
            let before = kept.len();
            kept.retain(|k| k.region != p.region);
            PENDING.fetch_sub((before - kept.len()) as u64, Ordering::Release);
            // With those put in the list since this pass took it.
            released(p.region, p.serial);
        }
    }
    core().pending.extend(kept);
}

/// After a submission's wait (with R5, at the landing of the flights that
/// declared them, before the title is told they are done): the shaders of
/// its lists may have written these homes' ranges (`declared`: each copy
/// with its ranges, as `Host::submit` gathers them). Write watching does not see GPU writes, so
/// a copy over the same bytes elsewhere (a buffer view's exact copy, a
/// buffer partly in a home) is told: its pages there count as written and
/// its next refresh takes them again (`crate::watch::gpu_wrote`). Normally
/// no copy lies over a home and nothing is done.
///
/// A home in a chunk of the pool ([`Kind::Pool`]) writes pages the title
/// may show at other views too: the copies there are looked for as well
/// (`crate::pool::shown_at`), and the placed textures over all of those
/// ranges are told (`crate::pool::textures_told`), since a texture's host
/// copy there would otherwise keep what it had (a copy's write-back tells
/// them; nothing is written back from such a home). The other views are
/// looked for per home, whose ranges each lie in its one run of pages: two
/// homes side by side at the title's addresses need not be one run.
///
/// shortcut: host copies of textures over a home of [`Kind::Private`] are
/// not looked for.
pub(crate) fn after_wait(declared: &[(Arc<Shadow>, Vec<(usize, usize)>)]) {
    if !on() {
        return;
    }
    // Each byte looked for once: `crate::memory` declares every home it
    // numbered for every dispatch of its shaders. The pool's ranges are
    // merged per home before their other views are looked for.
    let mut private: Vec<(usize, usize)> = Vec::new();
    let mut pool: std::collections::HashMap<usize, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    for (home, written) in declared {
        let ranges = match home.home_kind() {
            // (Also a copy in the card's memory whose CPU side is a home:
            // what its shaders wrote reaches the home by its merge,
            // `crate::card_home`, which tells it.)
            None => continue,
            Some(Kind::Private) => &mut private,
            Some(Kind::Pool) => pool.entry(Arc::as_ptr(home) as usize).or_default(),
        };
        ranges.extend(written.iter().copied().filter(|(from, to)| from < to));
    }
    let mut ranges = merged(private);
    if !pool.is_empty() {
        let shown = pool_shown(pool.into_values().flat_map(merged));
        ranges.extend(shown);
        ranges = merged(ranges);
    }
    tell_copies(ranges);
}

/// Slice 3 (`crate::memory`, `VICTORY_MEMORY_DIRECTORY=1`): after a
/// submission's wait, raw-address shaders may have written the pages of
/// chunks of the pool that the directory names (`ranges`, in order and
/// apart), which have no home there: as for a C-pool home's ranges
/// ([`after_wait`]), the copies over those pages at every view, and the
/// placed textures there, are told.
pub(crate) fn directory_written(ranges: Vec<(usize, usize)>) {
    if ranges.is_empty() {
        return;
    }
    tell_copies(pool_shown(ranges.into_iter()));
}

/// The title ranges that show the pool pages under `ranges` at every view
/// (`crate::pool::shown_at`; each range lies in one run of pages), merged,
/// with the placed textures over them told (`crate::pool::textures_told`).
fn pool_shown(ranges: impl Iterator<Item = (usize, usize)>) -> Vec<(usize, usize)> {
    let shown = merged(
        ranges
            .flat_map(|(from, to)| crate::pool::shown_at(from, to))
            .collect(),
    );
    for &(from, to) in &shown {
        crate::pool::textures_told(from, to);
    }
    shown
}

/// The copies over `ranges` (merged) take those bytes again: the GPU wrote
/// them where write watching does not see it.
fn tell_copies(ranges: Vec<(usize, usize)>) {
    for (from, to) in ranges {
        for (a, b) in crate::shadow::copies_over(from as u64, to as u64) {
            crate::watch::gpu_wrote(a, b - a);
            TOLD[8].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// `ranges` (start, end) in order, those that overlap or touch joined.
fn merged(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    merged
}

// ---- Slice 3: our own heaps (`VICTORY_MEMORY_OWN=1`) ----

/// `VICTORY_MEMORY_OWN=1` with the core on: the memory we give the title
/// out of our own allocator (its D3D12 heaps, its descriptor heaps,
/// committed resources when it names no allocator, `XMemAllocDefault`'s
/// blocks of 64 KiB and more) is a region of our own below 128 GiB, 64 KiB
/// aligned and write-watched ([`memory_own_alloc`]), which the core gives
/// homes in as it does the title's regions: what was class O, process-heap
/// memory no home may be opened on (`Why::NotRegion`), becomes C. Read
/// once.
pub(crate) fn own_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_MEMORY_OWN").is_some_and(|v| v == "1");
        if asked && !on() {
            tracing::warn!("VICTORY_MEMORY_OWN=1 does nothing without VICTORY_MEMORY_CORE=1");
        } else if asked {
            tracing::info!(
                "memory core own heaps on (VICTORY_MEMORY_OWN=1): our heaps for the title are watched regions below 128 GiB the card may use in place"
            );
        }
        asked && on()
    });
    on || test_own()
}

/// [`own_on`], for `victory-hle` (`XMemAllocDefault`).
pub fn memory_own_on() -> bool {
    own_on()
}

#[cfg(test)]
thread_local! {
    static TEST_OWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_own() -> bool {
    TEST_OWN.with(std::cell::Cell::get) && on()
}

#[cfg(not(test))]
fn test_own() -> bool {
    false
}

/// Our own heaps since the start: regions made, their bytes, refused by
/// the host, given back (at a sweep, [`run_own_frees`]), their bytes, given
/// back that the host refused.
static OWN: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Our own heaps to give back at the next sweep ([`memory_own_free`]), by
/// base. Its own lock, taken with no other held.
static OWN_FREES: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// `len` zeroed bytes aligned to `align` (a power of two), as a region of
/// our own: a reservation of its own, committed read-write and
/// write-watched, below 128 GiB (where the title's GPU memory is, and as
/// far as `crate::memory`'s directory reaches, so that raw-address shaders
/// find it too) and above 4 GiB (kept for the title's 32-bit memory), its
/// length rounded up to
/// 64 KiB so that no other allocation shares its last 64 KiB page (a home
/// pins whole 64 KiB pages, and a free of a page under one fails). Told to
/// the region table as ours. None if the host refuses it; the caller then
/// takes heap memory as before.
pub fn memory_own_alloc(len: usize, align: usize) -> Option<std::ptr::NonNull<u8>> {
    const K64: usize = 64 << 10;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    if len == 0 || !align.is_power_of_two() {
        return None;
    }
    let bytes = len.checked_next_multiple_of(K64)?;
    let mut requirements = mem::MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (4usize << 30) as *mut c_void,
        // The directory's last page stays empty (`crate::memory`).
        HighestEndingAddress: ((victory_shader::emit::MEMORY_PAGES as usize - 1) * K64 - 1)
            as *mut c_void,
        // Reservations are 64 KiB aligned anyway.
        Alignment: if align > K64 { align } else { 0 },
    };
    let mut parameter = mem::MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = mem::MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: a reservation of our own; the parameter points at
    // `requirements`, which outlives the call.
    let at = unsafe {
        mem::VirtualAlloc2(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            std::ptr::null(),
            bytes,
            mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
            mem::PAGE_READWRITE,
            &mut parameter,
            1,
        )
    };
    if at.is_null() {
        if OWN[2].fetch_add(1, Ordering::Relaxed) < 4 {
            // SAFETY: a plain read of this thread's last error.
            let error = unsafe { GetLastError() };
            tracing::warn!(
                bytes,
                error,
                "memory core: no region of our own for a heap; heap memory instead"
            );
        }
        return None;
    }
    let base = at as usize;
    {
        // Without `FREEING` (we may be called with any lock held): a waiting
        // free of the title's is of pages the host still holds, which it
        // cannot have given us, so none is over these.
        let mut core = core();
        let end = base + bytes;
        let stale: Vec<usize> = core
            .regions
            .range(..end)
            .rev()
            .take_while(|(b, region)| b.saturating_add(region.len) > base)
            .map(|(b, _)| *b)
            .collect();
        for b in stale {
            core.regions.remove(&b);
        }
        core.regions.insert(
            base,
            Region::new(
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                0,
                mem::PAGE_READWRITE,
                true,
            ),
        );
    }
    OWN[0].fetch_add(1, Ordering::Relaxed);
    OWN[1].fetch_add(bytes as u64, Ordering::Relaxed);
    std::ptr::NonNull::new(at.cast())
}

/// Gives back a region [`memory_own_alloc`] made, at the next sweep
/// ([`run_own_frees`]): for a caller that may hold any lock (the drop of
/// memory of ours with no allocator: heaps, descriptor heaps, resources
/// made without one), as the release takes the core's path (the homes over it let go
/// first, the release waiting while a list still holds one), whose lock
/// comes first. A caller that holds none of ours takes
/// [`memory_own_free_now`].
pub fn memory_own_free(at: usize) {
    OWN_FREES.lock().unwrap_or_else(|e| e.into_inner()).push(at);
}

/// Gives back a region [`memory_own_alloc`] made now, through the core's
/// path as [`memory_own_free`]'s sweep does: for a caller that holds no
/// lock of ours (`XMemFreeDefault`, on the title's threads), so that a
/// title that frees many big blocks between sweeps does not keep them all
/// committed.
pub fn memory_own_free_now(at: usize) {
    release_own(at);
}

/// The regions of ours given back since the last sweep are released
/// ([`release_own`]).
fn run_own_frees() {
    let due = std::mem::take(&mut *OWN_FREES.lock().unwrap_or_else(|e| e.into_inner()));
    for at in due {
        release_own(at);
    }
}

/// The region of ours at `at` is released, as a title's release is
/// ([`title_virtual_free`]), and counted apart from the title's.
fn release_own(at: usize) {
    let len = core()
        .region_at(at)
        .filter(|(base, region)| *base == at && region.own)
        .map_or(0, |(_, region)| region.len);
    OWN[3].fetch_add(1, Ordering::Relaxed);
    OWN[4].fetch_add(len as u64, Ordering::Relaxed);
    if title_virtual_free(at, 0, mem::MEM_RELEASE) == 0
        && OWN[5].fetch_add(1, Ordering::Relaxed) < 4
    {
        // SAFETY: a plain read of this thread's last error.
        let error = unsafe { GetLastError() };
        tracing::warn!(
            at = format_args!("{at:#x}"),
            error,
            "memory core: a region of our own could not be released"
        );
    }
}

/// The own heaps' part of the ten-second lines, with the switch on.
fn own_line() -> Option<String> {
    if !own_on() {
        return None;
    }
    let o: Vec<u64> = OWN.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let waiting = OWN_FREES.lock().unwrap_or_else(|e| e.into_inner()).len();
    Some(format!(
        "memory core own heaps: {} regions made ({} MB), {} refused by the host (heap memory instead); {} given back ({} MB, {} refused), {waiting} to give back at the next sweep",
        o[0],
        o[1] >> 20,
        o[2],
        o[3],
        o[4] >> 20,
        o[5],
    ))
}

/// Once per sweep (`crate::inplace::tick`): our own heaps given back, the
/// frees that waited, and the line every ten seconds.
pub(crate) fn tick() {
    if !on() {
        return;
    }
    run_own_frees();
    run_pending();
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = crate::host::now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 10_000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    for line in lines() {
        tracing::info!("{line}");
    }
}

/// The ten-second lines: what the core did, then the census of regions by
/// the console's words.
fn lines() -> Vec<String> {
    let made: Vec<u64> = MADE.iter().map(|m| m.load(Ordering::Relaxed)).collect();
    let told: Vec<u64> = TOLD.iter().map(|t| t.load(Ordering::Relaxed)).collect();
    let down: Vec<String> = WHYS
        .iter()
        .zip(&TURNED_DOWN)
        .map(|(w, n)| format!("{} {}", n.load(Ordering::Relaxed), w.1))
        .collect();
    // Before the core's lock: `crate::memory` has its own.
    let views = crate::memory::let_go_views_waiting();
    let directory = crate::memory::line();
    let pool = crate::pool::line();
    let own = own_line();
    let core = core();
    let (homes, pinned) = core.live.iter().fold((0u64, 0u64), |a, (&(_, bytes), &n)| {
        (a.0 + u64::from(n), a.1 + bytes * u64::from(n))
    });
    // (Only with R5 do protections wait for flights.)
    let waited = if crate::host::r5() {
        format!(
            "{} waited for flights in the air ({} ms), ",
            told[11],
            told[12] / 1000
        )
    } else {
        String::new()
    };
    let mut lines = vec![format!(
        "memory core: {homes} homes pin {} MB (made {} windows, {} buffers; {} asks joined a home; {} big buffers given a card copy first; {} opens, {} us); \
         kept a copy: {}; title frees of GPU regions {} ({} in regions that had a home, {} done later, {} ms waited, {} failed later, {} given up, {} waiting), \
         protections {} ({} in regions that had a home, {waited}{} refused, {} homes taken out of the shaders' directory at once); \
         GPU writes into homes told to copies over them {}; view heaps whose slots still hold a let-go home {views}; {pool}",
        pinned >> 20,
        made[0],
        made[1],
        made[2],
        made[3],
        made[4],
        made[5],
        down.join(", "),
        told[0],
        told[1],
        told[2],
        told[3],
        told[4],
        told[9],
        core.pending.len(),
        told[5],
        told[6],
        told[7],
        told[10],
        told[8],
    )];
    // Slice 4: copies in the card's memory whose CPU side is a home.
    lines.extend(crate::card_home::line());
    // Slice 3: the directory of raw-address shaders, our own heaps.
    lines.extend(directory);
    lines.extend(own);
    // The census: regions by the console's protection above the host's
    // bits, its page-size kind bits and the XMem flags, with the bytes homes
    // pin in them; our own heaps (slice 3) apart.
    let mut census: BTreeMap<(bool, u32, u32, u64), (u64, u64, u64)> = BTreeMap::new();
    for (&base, region) in &core.regions {
        let key = (
            region.own,
            region.protect & !0x1ff,
            region.kind & 0xe040_0000,
            region.flags,
        );
        let in_homes: u64 = core
            .live
            .range((base as u64, 0)..((base + region.len) as u64, 0))
            .map(|(&(_, bytes), &n)| bytes * u64::from(n))
            .sum();
        let entry = census.entry(key).or_default();
        entry.0 += 1;
        entry.1 += region.len as u64;
        entry.2 += in_homes;
    }
    for ((own, protect, kind, flags), (n, bytes, in_homes)) in census {
        if own {
            lines.push(format!(
                "memory core census: our own heaps: {n} regions, {} MB, {} MB in homes",
                bytes >> 20,
                in_homes >> 20
            ));
            continue;
        }
        lines.push(format!(
            "memory core census: protection bits {protect:#x}, kind bits {kind:#x}, flags {flags:#x}: {n} regions, {} MB, {} MB in homes",
            bytes >> 20,
            in_homes >> 20
        ));
    }
    lines
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inplace::Rig;
    use crate::shadow::{find, register, unregister};

    const K64: u64 = 64 << 10;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// One test at a time over the core's tables and counters, which are
    /// the process's.
    fn serial() -> std::sync::MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets the core on for this thread (and, if asked, the device without
    /// `ID3D12Device13`) for as long as it lives, with the other tests of the
    /// core waiting.
    pub(crate) struct Switched(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl Switched {
        pub(crate) fn on(no_device13: bool) -> Switched {
            let one = serial();
            TEST_ON.with(|on| on.set(true));
            NO_DEVICE13.with(|no| no.set(no_device13));
            Switched(one)
        }
    }

    impl Drop for Switched {
        fn drop(&mut self) {
            TEST_ON.with(|on| on.set(false));
            NO_DEVICE13.with(|no| no.set(false));
            TEST_OWN.with(|own| own.set(false));
            crate::pool::set_test(false);
            crate::card_home::set_test(false);
            crate::memory::set_test(false);
        }
    }

    impl Switched {
        /// [`Switched::on`] with the memory pool on too.
        fn pool() -> Switched {
            let switched = Switched::on(false);
            crate::pool::set_test(true);
            switched
        }
    }

    /// The core and the pool on for the thread that calls it, for a thread
    /// a test starts (the switches are per thread).
    fn pool_on_here() {
        TEST_ON.with(|on| on.set(true));
        crate::pool::set_test(true);
    }

    /// Fixed answers, and the asks in the order they came.
    struct Fixed {
        holder: Option<bool>,
        card: bool,
        view: Viewed<u32>,
        region: Option<Result<(), Why>>,
        committed: bool,
        claim: Result<(), Why>,
        asked: Vec<&'static str>,
    }

    impl Fixed {
        fn good() -> Fixed {
            Fixed {
                holder: None,
                card: false,
                view: Viewed::No,
                region: Some(Ok(())),
                committed: true,
                claim: Ok(()),
                asked: Vec::new(),
            }
        }
    }

    impl Ask for Fixed {
        type Home = u32;
        type Pool = u32;
        fn holder(&mut self) -> Held<u32> {
            self.asked.push("holder");
            match self.holder {
                None => Held::None,
                Some(true) => Held::Whole(7),
                Some(false) => Held::Part,
            }
        }
        fn card(&mut self) -> bool {
            self.asked.push("card");
            self.card
        }
        fn view(&mut self) -> Viewed<u32> {
            self.asked.push("view");
            self.view
        }
        fn region(&mut self) -> Option<Result<(), Why>> {
            self.asked.push("region");
            self.region
        }
        fn committed(&mut self) -> bool {
            self.asked.push("committed");
            self.committed
        }
        fn claim(&mut self) -> Result<(), Why> {
            self.asked.push("claim");
            self.claim
        }
    }

    const OPEN: Gate = Gate {
        on: true,
        late: false,
        flights_safe: false,
        device13: true,
        latched: false,
        pool: true,
    };

    /// R5: with flights in the air the rules go on as with every submission
    /// waited for once the flights land what homes need
    /// (`flights_safe`); without it late submissions keep every range a
    /// copy, asking nothing. The switch alone, with nothing late, changes
    /// nothing.
    #[test]
    fn classify_lets_homes_under_safe_flights() {
        let safe = Gate {
            late: true,
            flights_safe: true,
            ..OPEN
        };
        let mut f = Fixed::good();
        assert_eq!(classify(safe, false, &mut f), Class::Place);
        assert_eq!(f.asked, ["holder", "view", "region", "committed", "claim"]);
        let mut f = Fixed {
            holder: Some(true),
            ..Fixed::good()
        };
        assert_eq!(classify(safe, false, &mut f), Class::Home(7));
        let mut f = Fixed::good();
        assert_eq!(
            classify(Gate { late: true, ..OPEN }, false, &mut f),
            Class::Copy(Why::Late)
        );
        assert!(f.asked.is_empty());
        let mut f = Fixed::good();
        assert_eq!(
            classify(
                Gate {
                    flights_safe: true,
                    ..OPEN
                },
                false,
                &mut f
            ),
            Class::Place
        );
        // The gate's other rules still come first.
        let mut f = Fixed::good();
        assert_eq!(
            classify(
                Gate {
                    device13: false,
                    ..safe
                },
                false,
                &mut f
            ),
            Class::Copy(Why::NoDevice13)
        );
    }

    /// Every rule of [`classify`], in its order, and that a rule asks
    /// nothing it does not need (the card is asked only for a big buffer,
    /// and before anything else about the range).
    #[test]
    fn classify_follows_the_rules_in_order() {
        let run = |gate: Gate, big: bool, fixed: &mut Fixed| classify(gate, big, fixed);
        // 1: the gate, asking nothing.
        for (gate, why) in [
            (Gate { on: false, ..OPEN }, Why::Off),
            (Gate { late: true, ..OPEN }, Why::Late),
            (
                Gate {
                    device13: false,
                    ..OPEN
                },
                Why::NoDevice13,
            ),
            (
                Gate {
                    latched: true,
                    ..OPEN
                },
                Why::Latched,
            ),
        ] {
            let mut f = Fixed::good();
            assert_eq!(run(gate, true, &mut f), Class::Copy(why));
            assert!(f.asked.is_empty(), "{why:?} asked {:?}", f.asked);
        }
        // 2: a big buffer asks the card first; given one: G, whatever home
        // holds it (all of it, or part).
        for holder in [None, Some(true), Some(false)] {
            let mut f = Fixed {
                card: true,
                holder,
                ..Fixed::good()
            };
            assert_eq!(run(OPEN, true, &mut f), Class::Card, "{holder:?}");
            assert_eq!(f.asked, ["card"]);
        }
        // 3: a home holds it whole: that home, nothing else asked.
        let mut f = Fixed {
            holder: Some(true),
            ..Fixed::good()
        };
        assert_eq!(run(OPEN, false, &mut f), Class::Home(7));
        assert_eq!(f.asked, ["holder"]);
        // A big one the card turns down joins it too.
        let mut f = Fixed {
            holder: Some(true),
            ..Fixed::good()
        };
        assert_eq!(run(OPEN, true, &mut f), Class::Home(7));
        assert_eq!(f.asked, ["card", "holder"]);
        // 3: partly: a copy.
        for big in [false, true] {
            let mut f = Fixed {
                holder: Some(false),
                ..Fixed::good()
            };
            assert_eq!(run(OPEN, big, &mut f), Class::Copy(Why::Overlap));
            assert_eq!(f.asked.last(), Some(&"holder"));
        }
        // Refused by the card: it goes on and is placed (C).
        let mut f = Fixed::good();
        assert_eq!(run(OPEN, true, &mut f), Class::Place);
        assert_eq!(
            f.asked,
            ["card", "holder", "view", "region", "committed", "claim"]
        );
        // A small one never asks the card.
        let mut f = Fixed {
            card: true,
            ..Fixed::good()
        };
        assert_eq!(run(OPEN, false, &mut f), Class::Place);
        assert!(!f.asked.contains(&"card"));
        // 4: one run of an opened chunk of the pool: a new home there, with
        // nothing about regions asked; a big one the card turned down too.
        for big in [false, true] {
            let mut f = Fixed {
                view: Viewed::Pool(9),
                ..Fixed::good()
            };
            assert_eq!(run(OPEN, big, &mut f), Class::Pool(9));
            assert_eq!(f.asked.last(), Some(&"view"));
        }
        // But a big one the card takes stays G.
        let mut f = Fixed {
            view: Viewed::Pool(9),
            card: true,
            ..Fixed::good()
        };
        assert_eq!(run(OPEN, true, &mut f), Class::Card);
        // 4-7, each stopping there.
        for (f, why, last) in [
            (
                Fixed {
                    view: Viewed::Copy(Why::View),
                    ..Fixed::good()
                },
                Why::View,
                "view",
            ),
            (
                Fixed {
                    view: Viewed::Copy(Why::Scattered),
                    ..Fixed::good()
                },
                Why::Scattered,
                "view",
            ),
            (
                Fixed {
                    view: Viewed::Copy(Why::ReservedChunk),
                    ..Fixed::good()
                },
                Why::ReservedChunk,
                "view",
            ),
            (
                Fixed {
                    region: None,
                    ..Fixed::good()
                },
                Why::NotRegion,
                "region",
            ),
            (
                Fixed {
                    region: Some(Err(Why::Blacklisted)),
                    ..Fixed::good()
                },
                Why::Blacklisted,
                "region",
            ),
            (
                Fixed {
                    region: Some(Err(Why::Busy)),
                    ..Fixed::good()
                },
                Why::Busy,
                "region",
            ),
            (
                Fixed {
                    committed: false,
                    ..Fixed::good()
                },
                Why::NotCommitted,
                "committed",
            ),
            (
                Fixed {
                    claim: Err(Why::Busy),
                    ..Fixed::good()
                },
                Why::Busy,
                "claim",
            ),
            (
                Fixed {
                    claim: Err(Why::Budget),
                    ..Fixed::good()
                },
                Why::Budget,
                "claim",
            ),
        ] {
            let mut f = f;
            assert_eq!(run(OPEN, false, &mut f), Class::Copy(why));
            assert_eq!(f.asked.last(), Some(&last), "{why:?}");
        }
    }

    /// The region table keeps the console's words as given, the host's
    /// protection of the memory is what `xmem` gave it, and a release
    /// through the core forgets the region.
    #[test]
    fn region_table_records_hints_and_forgets_released_regions() {
        let _one = serial();
        // SAFETY: an allocation of the test's own.
        let base = unsafe {
            mem::VirtualAlloc(
                std::ptr::null(),
                4 * K64 as usize,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(base, 0);
        title_memory_region(base, 4 * K64 as usize - 100, 0xa000_3000, 0x3, 0x404);
        {
            let mut core = core();
            let (found, region) = core.region_at(base + 5).expect("recorded");
            assert_eq!(found, base);
            assert_eq!(
                (region.len, region.kind, region.flags, region.protect),
                (4 * K64 as usize, 0xa000_3000, 0x3, 0x404)
            );
            assert!(
                core.region_at(base + 4 * K64 as usize).is_none(),
                "past its end"
            );
        }
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer.
        unsafe { mem::VirtualQuery(base as *const c_void, &mut info, size_of_val(&info)) };
        assert_eq!(
            info.Protect,
            mem::PAGE_READWRITE,
            "the host's protection is unchanged"
        );
        // A region over it replaces it (it was freed unseen).
        title_memory_region(base + K64 as usize, K64 as usize, 0, 0, 4);
        assert!(!core().regions.contains_key(&base));
        title_memory_region(base, 4 * K64 as usize, 0, 0, 4);
        assert_eq!(title_virtual_free(base, 0, mem::MEM_RELEASE), 1);
        assert!(core().region_at(base).is_none(), "forgotten when released");
    }

    /// The core's budget is its own setting and its own counter.
    #[test]
    fn memory_core_budget_is_its_own() {
        assert_eq!(most_of(None), 4096 << 20);
        assert_eq!(most_of(Some(" 512 ")), 512 << 20);
        assert_eq!(most_of(Some("lots")), 4096 << 20);
        assert_eq!(most_of(Some("18446744073709551615")), u64::MAX, "saturates");
        let count = AtomicU64::new(0);
        let limit = most_of(Some("1"));
        assert!(crate::inplace::try_reserve_at(&count, 16 * K64, limit));
        assert!(
            !crate::inplace::try_reserve_at(&count, K64, limit),
            "1 MB is 16 pages"
        );
        crate::inplace::release_at(&count, K64);
        assert!(crate::inplace::try_reserve_at(&count, K64, limit));
    }

    /// The host on the software adapter, if it can open parts of an
    /// allocation (`ID3D12Device13`).
    fn warp13() -> Option<&'static crate::host::Host> {
        crate::host::host().filter(|host| host.device.cast::<ID3D12Device13>().is_ok())
    }

    /// Title memory as `xmem` reserves it (committed, read-write,
    /// write-watched), told to the core as a region.
    struct TestRegion {
        base: u64,
    }

    impl TestRegion {
        /// [`TestRegion::new`] below 128 GiB.
        fn low(bytes: usize) -> TestRegion {
            let base = low(
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
                mem::PAGE_READWRITE,
            );
            title_memory_region(
                base as usize,
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                3,
                0x404,
            );
            TestRegion { base }
        }

        fn new(bytes: usize) -> TestRegion {
            // SAFETY: an allocation of the test's own.
            let base = unsafe {
                mem::VirtualAlloc(
                    std::ptr::null(),
                    bytes,
                    mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
                    mem::PAGE_READWRITE,
                )
            } as u64;
            assert_ne!(base, 0);
            title_memory_region(
                base as usize,
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                3,
                0x404,
            );
            TestRegion { base }
        }

        /// Released through the core, as the title's `VirtualFree` is.
        fn release(self) {
            assert_eq!(
                title_virtual_free(self.base as usize, 0, mem::MEM_RELEASE),
                1
            );
        }
    }

    fn fill(at: u64, len: u64, seed: u8) {
        for i in 0..len {
            // SAFETY: committed memory of the test's own.
            unsafe { ((at + i) as *mut u8).write((i as u8).wrapping_mul(13).wrapping_add(seed)) };
        }
    }

    fn same(at: u64, bytes: &[u8]) -> bool {
        // SAFETY: committed memory of the test's own.
        unsafe { std::slice::from_raw_parts(at as *const u8, bytes.len()) == bytes }
    }

    fn state(at: u64) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer.
        unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
        info.State
    }

    fn turned(why: Why) -> u64 {
        let i = WHYS.iter().position(|w| w.0 == why).unwrap();
        TURNED_DOWN[i].load(Ordering::Relaxed)
    }

    /// A small title buffer in a region is its own home: the GPU reads the
    /// title's bytes at the buffer's address, also those the CPU wrote after
    /// the home was made, and a refresh copies nothing.
    #[test]
    fn warp_core_places_a_small_buffer_and_the_gpu_reads_title_bytes_live() {
        let Some(host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64 + 0x100, 3 * K64 - 0x200);
        register(0xc0de_0501, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home(), "a home, not a copy");
        let mut rig = Rig::new(&host.device).unwrap();
        fill(start, len, 1);
        let got = rig
            .fetch(&home.buffer, home.offset(start), len as usize)
            .unwrap();
        assert!(same(start, &got), "the GPU reads the title's bytes");
        fill(start, len, 99);
        let got = rig
            .fetch(&home.buffer, home.offset(start), len as usize)
            .unwrap();
        assert!(
            same(start, &got),
            "and what the CPU wrote since, with nothing in between"
        );
        assert_eq!(
            home.update(start as usize, len as usize),
            0,
            "a refresh copies nothing"
        );
        assert!(home.unwatched(), "no write-watch asking is made for it");
        unregister(0xc0de_0501, start);
        drop((home, rig));
        region.release();
    }

    /// What the GPU writes through a home is at the title's address once it
    /// has run, with nothing copied back.
    #[test]
    fn warp_core_gpu_write_lands_in_title_memory_with_no_copy_back() {
        let Some(host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(4 * K64 as usize);
        let (start, len) = (region.base + 0x40, 2 * K64);
        register(0xc0de_0601, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let mut rig = Rig::new(&host.device).unwrap();
        rig.store(&home.buffer, home.offset(start + 16), &[0xa5; 64])
            .unwrap();
        assert!(
            same(start + 16, &[0xa5; 64]),
            "the GPU's write is at the title's address"
        );
        home.writing(start as usize, len as usize);
        assert_eq!(
            home.copy_back(start as usize, len as usize),
            0,
            "nothing to copy back"
        );
        unregister(0xc0de_0601, start);
        drop((home, rig));
        region.release();
    }

    /// A window home and a title buffer inside it are one home: the same
    /// buffer, the same GPU address for the same title byte.
    #[test]
    fn warp_core_nested_buffer_and_window_resolve_to_one_home() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        const CELL: u64 = 4 << 20;
        let region = TestRegion::new(3 * CELL as usize);
        let cell = (region.base + CELL - 1) & !(CELL - 1);
        let window = find(cell + 0x1000, 16).expect("a window");
        assert!(window.is_home() && window.is_window());
        assert_eq!(window.span(), (cell, CELL));
        let (start, len) = (cell + 2 * K64 + 0x40, 0x1000);
        register(0xc0de_0701, start, len);
        let buffer = find(start, len).expect("the buffer");
        assert!(Arc::ptr_eq(&buffer, &window), "one home");
        assert_eq!(buffer.gpu(start + 8), window.gpu(start + 8));
        assert_eq!(buffer.buffer.as_raw(), window.buffer.as_raw());
        // Let go, as the title's free would: out of both tables.
        assert_eq!(cpu_touched(cell as usize, CELL as usize, Touch::Freeing), 1);
        // The shaders' directory forgets it at the sweep, not now.
        assert!(core().unnumber.contains(&(cell, cell + CELL)));
        run_pending();
        assert!(core().unnumber.is_empty(), "at the sweep");
        unregister(0xc0de_0701, start);
        drop((buffer, window));
        region.release();
    }

    /// A copy over a home's bytes (a buffer view's exact copy) takes the
    /// pages the GPU wrote through the home again: write watching does not
    /// see a card's writes, `after_wait` tells the journal.
    #[test]
    fn warp_core_overlapping_copy_sees_gpu_writes() {
        let Some(host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 4 * K64);
        register(0xc0de_0801, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let (view, bytes) = (start + 0x2000, 0x1000u64);
        let copy = crate::shadow::exact(view, bytes).expect("an exact copy");
        assert!(!copy.is_home());
        fill(start, len, 5);
        copy.update(view as usize, bytes as usize);
        assert!(
            same(view, crate::shadow::copy_bytes(&copy)),
            "the copy took the title's bytes"
        );
        let new = [0x5au8; 0x1000];
        let mut rig = Rig::new(&host.device).unwrap();
        rig.store(&home.buffer, home.offset(view), &new).unwrap();
        assert!(same(view, &new));
        // On the software adapter the GPU is this process, and write watching
        // saw it: forgotten, as a card's write is never seen.
        // SAFETY: the test's own allocation.
        unsafe { mem::ResetWriteWatch(region.base as *const c_void, 8 * K64 as usize) };
        copy.update(view as usize, bytes as usize);
        assert!(
            !same(view, crate::shadow::copy_bytes(&copy)),
            "the copy cannot know"
        );
        after_wait(&[(home.clone(), vec![(view as usize, (view + bytes) as usize)])]);
        copy.update(view as usize, bytes as usize);
        assert!(
            same(view, crate::shadow::copy_bytes(&copy)),
            "told, it takes the page again"
        );
        unregister(0xc0de_0801, start);
        drop((home, copy, rig));
        // Not released: the exact copy stays in its table, over this memory.
        let _ = region;
    }

    /// The title's decommit of a home's pages: with only the tables holding
    /// the home it is let go and the pages are decommitted at once; with a
    /// list holding it, the title is told TRUE, the pages wait, and they are
    /// decommitted at the sweep after the list let it go.
    #[test]
    fn warp_core_free_lets_the_home_go_then_frees() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 2 * K64);
        register(0xc0de_0901, start, len);
        assert!(find(start, len).expect("a home").is_home());
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1
        );
        assert_eq!(state(start), mem::MEM_RESERVE, "decommitted at once");
        unregister(0xc0de_0901, start);
        // Held as a recorded list holds what it reads.
        let (start, len) = (region.base + 4 * K64, K64);
        register(0xc0de_0902, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home());
        let waiting = TOLD[2].load(Ordering::Relaxed);
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1,
            "told it was done"
        );
        assert_eq!(TOLD[2].load(Ordering::Relaxed), waiting + 1, "done later");
        assert_eq!(
            state(start),
            mem::MEM_COMMIT,
            "still committed under the heap"
        );
        run_pending();
        assert_eq!(state(start), mem::MEM_COMMIT, "still held");
        drop(held);
        run_pending();
        assert_eq!(
            state(start),
            mem::MEM_RESERVE,
            "decommitted once nothing holds the heap"
        );
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        unregister(0xc0de_0902, start);
        region.release();
    }

    /// C (`VICTORY_R5_FLUSH=1`): a list that refreshes a home records no
    /// refresh step and holds the home once, until it is reset: the title's
    /// decommit of its pages waits while the list is open, also after the
    /// list ran, and is done at the sweep after the reset. Without the
    /// switch the refresh is a step, as before.
    #[test]
    fn warp_r5_flush_a_home_is_held_with_no_refresh_step() {
        let Some(host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_f201, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home() && home.is_placed());
        {
            let _switch = crate::host::r5_flush::TestSwitch::set(false);
            let list = host.new_list().unwrap();
            home.refresh(&list, start, len);
            assert_eq!(list.r5_flush_held(), (1, 0), "without the switch: a step");
            list.close();
            list.reset();
        }
        let _switch = crate::host::r5_flush::TestSwitch::set(true);
        let list = host.new_list().unwrap();
        home.refresh(&list, start, len);
        home.refresh(&list, start + 0x100, 0x100);
        assert_eq!(list.r5_flush_held(), (0, 1), "no step, held once");
        list.close();
        host.execute(&[&list]);
        drop(home);
        let waiting = TOLD[2].load(Ordering::Relaxed);
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1,
            "told it was done"
        );
        assert_eq!(TOLD[2].load(Ordering::Relaxed), waiting + 1, "done later");
        run_pending();
        assert_eq!(state(start), mem::MEM_COMMIT, "held by the list");
        list.reset();
        assert_eq!(list.r5_flush_held(), (0, 0));
        run_pending();
        assert_eq!(
            state(start),
            mem::MEM_RESERVE,
            "decommitted once the list let it go"
        );
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        unregister(0xc0de_f201, start);
        drop(list);
        region.release();
    }

    /// C (`VICTORY_R5_FLUSH=1`): a card copy whose CPU side is a home is no
    /// placed copy: its refresh keeps its step (the card's side is sent up
    /// from it).
    #[test]
    fn warp_r5_flush_a_card_copy_on_a_home_keeps_its_step() {
        let Some(host) = warp13() else { return };
        if !host.card_ready() {
            return;
        }
        let _on = Switched::pool();
        crate::card_home::set_test(true);
        let _switch = crate::host::r5_flush::TestSwitch::set(true);
        let bytes = 32 * K64;
        let chunk = TestChunk::new(2 * bytes as usize, crate::pool::PoolChunk::Committed);
        let p = Place::new(bytes as usize);
        p.map(0, bytes, &chunk, 0);
        let (start, len) = (p.base + K64, 1 << 20);
        register(0xc0de_f202, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(copy.card_home().is_some(), "its CPU side is a home");
        assert!(!copy.is_placed(), "not placed itself");
        let list = host.new_list().unwrap();
        copy.refresh(&list, start, len);
        assert_eq!(list.r5_flush_held(), (1, 0), "its step is kept");
        list.close();
        list.reset();
        unregister(0xc0de_f202, start);
        drop((copy, list));
        p.unmap(0, bytes);
    }

    /// A decommit that waits, then the title commits the middle of those
    /// pages again: that part reads zeros and is never decommitted; the rest
    /// still is, once the heap goes.
    #[test]
    fn warp_core_recommit_cancels_a_waiting_decommit() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 3 * K64);
        register(0xc0de_0a01, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home());
        fill(start, len, 3);
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1
        );
        let pending = PENDING.load(Ordering::Relaxed);
        // The title's commit of the middle page, as `xmem` makes it: the
        // guard held across the host's commit.
        let committing = title_memory_committing((start + K64) as usize, K64 as usize);
        assert!(committing.0.is_some(), "the waiting frees held back");
        // SAFETY: pages of the test's own allocation.
        let again = unsafe {
            mem::VirtualAlloc(
                (start + K64) as *const c_void,
                K64 as usize,
                mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
        };
        assert_eq!(again as u64, start + K64);
        // The sweep's run of the waiting frees waits for the commit's end.
        let swept = Arc::new(AtomicBool::new(false));
        let sweep = std::thread::spawn({
            let swept = swept.clone();
            move || {
                run_pending();
                swept.store(true, Ordering::Relaxed);
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!swept.load(Ordering::Relaxed), "held back by the commit");
        drop(committing);
        sweep.join().unwrap();
        assert!(swept.load(Ordering::Relaxed));
        assert_eq!(PENDING.load(Ordering::Relaxed), pending + 1, "cut in two");
        assert!(
            same(start + K64, &[0u8; K64 as usize]),
            "the page committed again reads zeros"
        );
        drop(held);
        run_pending();
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        assert_eq!(state(start), mem::MEM_RESERVE, "the first page went");
        assert_eq!(
            state(start + K64),
            mem::MEM_COMMIT,
            "the page committed again stays"
        );
        assert_eq!(
            state(start + 2 * K64),
            mem::MEM_RESERVE,
            "the last page went"
        );
        unregister(0xc0de_0a01, start);
        region.release();
    }

    /// The title's protection of a home's pages: with only the tables
    /// holding the home it is let go and the protection is made; with a list
    /// holding it the title gets FALSE with error 87, it is counted, and the
    /// region gets no home again.
    #[test]
    fn warp_core_protect_lets_go_or_refuses_and_blacklists() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_1001, start, len);
        assert!(find(start, len).expect("a home").is_home());
        let mut old = 0;
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READONLY, &mut old),
            1
        );
        assert_eq!(old, mem::PAGE_READWRITE);
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READWRITE, &mut old),
            1
        );
        unregister(0xc0de_1001, start);
        let (start, len) = (region.base + 4 * K64, K64);
        register(0xc0de_1002, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home());
        let refused = TOLD[7].load(Ordering::Relaxed);
        // SAFETY: a plain write of this thread's last error.
        unsafe { SetLastError(0) };
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READONLY, &mut old),
            0
        );
        // SAFETY: a plain read of it.
        assert_eq!(unsafe { GetLastError() }, REFUSED_BY_HEAP);
        assert_eq!(TOLD[7].load(Ordering::Relaxed), refused + 1, "counted");
        let blacklisted = turned(Why::Blacklisted);
        register(0xc0de_1003, region.base + 6 * K64, K64);
        let copy = find(region.base + 6 * K64, K64).expect("a copy");
        assert!(!copy.is_home(), "the region gets no home again");
        assert_eq!(turned(Why::Blacklisted), blacklisted + 1);
        unregister(0xc0de_1002, start);
        unregister(0xc0de_1003, region.base + 6 * K64);
        drop((held, copy));
        region.release();
    }

    /// Memory of the process heap is never given a home, whatever it looks
    /// like to Windows: the heap's own frees and protections never reach
    /// the core, so a heap opened there could make them fail.
    #[test]
    fn warp_core_process_heap_memory_is_never_placed() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let block = vec![0u8; 4 * K64 as usize];
        let (start, len) = (block.as_ptr() as u64, 2 * K64);
        register(0xc0de_1101, start, len);
        let before = turned(Why::NotRegion);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home());
        assert_eq!(turned(Why::NotRegion), before + 1);
        unregister(0xc0de_1101, start);
        drop((copy, block));
    }

    /// A device that cannot open parts of an allocation: nothing is placed,
    /// and the title's calls go to the host as they are.
    #[test]
    fn warp_core_without_device13_is_off() {
        let Some(_host) = crate::host::host() else {
            return;
        };
        let _on = Switched::on(true);
        let region = TestRegion::new(4 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_1201, start, len);
        let before = turned(Why::NoDevice13);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home());
        assert_eq!(turned(Why::NoDevice13), before + 1);
        let mut old = 0;
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READONLY, &mut old),
            1
        );
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READWRITE, &mut old),
            1
        );
        unregister(0xc0de_1201, start);
        drop(copy);
        region.release();
    }

    /// With late waits nothing is given a home, and it is counted.
    #[test]
    fn warp_core_refuses_with_late_waits() {
        let _one = serial();
        let region = TestRegion::new(4 * K64 as usize);
        let before = turned(Why::Late);
        let gate = Gate { late: true, ..OPEN };
        let found = resolve_with(gate, region.base + K64, K64, false, false, &mut || {
            panic!("the card is not asked")
        });
        assert!(matches!(found, Found::Copy));
        assert_eq!(turned(Why::Late), before + 1);
        region.release();
    }

    // ---- R5 (`crate::host::r5`): homes under flights in the air ----

    use crate::host::flights::{QueueHold, TestFlights};

    /// The core on, and flights with R5 (in that order, so that the flights
    /// land before the core goes off).
    fn r5_on() -> (Switched, TestFlights) {
        let core = Switched::on(false);
        (core, TestFlights::on(true, true))
    }

    /// A buffer of ours in the PC's memory holding `bytes`, for a list to
    /// copy from as a shader would write.
    fn source(
        host: &crate::host::Host,
        bytes: &[u8],
    ) -> windows::Win32::Graphics::Direct3D12::ID3D12Resource {
        let (buffer, data) = host
            .shared_buffer(bytes.len().next_multiple_of(256) as u64)
            .unwrap();
        // SAFETY: the mapping holds that many bytes; the card does not use it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        buffer
    }

    /// A list whose "shader" writes `bytes` into `home` at title address
    /// `at` (a copy on the card), declared as shaders' writes over
    /// `[at, at + bytes)`; closed.
    fn writer(
        host: &crate::host::Host,
        home: &Arc<Shadow>,
        at: u64,
        from: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        bytes: u64,
    ) -> crate::host::HostList {
        let list = host.new_list().unwrap();
        list.defer_refresh(home, at as usize, bytes as usize);
        // SAFETY: an open list of the host's; live buffers, the range in both.
        unsafe {
            list.list
                .CopyBufferRegion(&home.buffer, home.offset(at), from, 0, bytes)
        };
        list.defer_gpu_write(home, at as usize, bytes as usize);
        list.close();
        list
    }

    /// R5's reads of homes on the card (`VICTORY_R5_LIFT=1`).
    mod r5_lift;

    /// A flight's shaders write a home's range; the next submission fills
    /// upload memory from that range: it waits for that flight (a read of
    /// what flights write), and once it is prepared its fill holds the
    /// card's bytes, not the title's from before.
    #[test]
    fn warp_r5_fill_of_a_home_waits_for_the_flight_that_writes_it() {
        let Some(host) = warp13() else { return };
        let (_core, flights) = r5_on();
        let region = TestRegion::new(4 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_5201, start, len);
        let home = find(start, len).expect("a home under flights");
        assert!(home.is_home(), "R5: a home while submissions fly");
        fill(start, len, 1);
        let written = [0xa5u8; 64];
        let from = source(host, &written);
        let hold = QueueHold::new(host);
        let one = writer(host, &home, start + 0x100, &from, 64);
        host.send_for_test(&one);
        assert_eq!(crate::host::flights::in_air(), 1, "held on the queue");
        let mut out = vec![0u8; 64];
        let two = host.new_list().unwrap();
        two.defer_fill(out.as_mut_ptr(), (start + 0x100) as *const u8, 64);
        two.close();
        assert_eq!(
            crate::host::flights::blocked(&[&two]),
            Some(crate::host::flights::READ),
            "a read of what a flight writes"
        );
        let released = hold.released().after(200);
        host.send_for_test(&two);
        released.join().unwrap();
        assert_eq!(out, written, "the fill took what the card wrote");
        assert!(same(start + 0x100, &written));
        flights.land_all();
        one.reset();
        two.reset();
        drop((one, two, hold));
        unregister(0xc0de_5201, start);
        drop(home);
        region.release();
    }

    /// An exact copy over a home's bytes; a flight's shaders write the home
    /// there, and a fence of the title's is signalled behind the flight. By
    /// the time the fence is set the copy has been told (its next refresh
    /// takes the card's bytes): the landing tells copies before it tells the
    /// title.
    #[test]
    fn warp_r5_copy_over_a_home_is_told_at_landing() {
        let Some(host) = warp13() else { return };
        let (_core, flights) = r5_on();
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 4 * K64);
        register(0xc0de_5301, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let (view, bytes) = (start + 0x2000, 0x1000u64);
        let copy = crate::shadow::exact(view, bytes).expect("an exact copy");
        assert!(!copy.is_home());
        fill(start, len, 5);
        copy.update(view as usize, bytes as usize);
        assert!(same(view, crate::shadow::copy_bytes(&copy)));
        let mut fence = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut fence
            ),
            crate::object::S_OK
        );
        let new = [0x5au8; 0x1000];
        let from = source(host, &new);
        let hold = QueueHold::new(host);
        let one = writer(host, &home, view, &from, bytes);
        host.send_for_test(&one);
        assert!(
            crate::host::flights::signal_behind(crate::fence::FenceRef::new(fence), 1),
            "the signal waits behind the flight"
        );
        assert_eq!(crate::fence::completed(fence), 0);
        hold.release();
        let waited = std::time::Instant::now();
        while crate::fence::completed(fence) < 1 {
            assert!(waited.elapsed().as_secs() < 10, "the flight lands");
            std::thread::yield_now();
        }
        assert!(same(view, &new), "the card's bytes are in title memory");
        // On the software adapter the card is this process, and write
        // watching saw it: forgotten, as a card's write is never seen.
        // SAFETY: the test's own allocation.
        unsafe { mem::ResetWriteWatch(region.base as *const c_void, 8 * K64 as usize) };
        copy.update(view as usize, bytes as usize);
        assert!(
            same(view, crate::shadow::copy_bytes(&copy)),
            "told before the fence was set: the refresh takes the card's bytes"
        );
        flights.land_all();
        one.reset();
        drop((one, hold));
        // SAFETY: the fence's own release, our last reference.
        assert_eq!(
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(fence, 0x10)(
                fence
            ),
            0
        );
        unregister(0xc0de_5301, start);
        drop((home, copy));
        // Not released: the exact copy stays in its table, over this memory.
        let _ = region;
    }

    /// An exact copy over a home's bytes is refreshed by a submission while
    /// a flight's shaders write the home there: it waits for that flight, so
    /// the refresh takes the card's bytes (nothing else would tell the copy
    /// before its next refresh), and it is counted.
    #[test]
    fn warp_r5_x_refresh_over_flight_written_home_waits() {
        let Some(host) = warp13() else { return };
        let (_core, flights) = r5_on();
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, 4 * K64);
        register(0xc0de_5401, start, len);
        let home = find(start, len).expect("a home");
        let (view, bytes) = (start + 0x3000, 0x800u64);
        let copy = crate::shadow::exact(view, bytes).expect("an exact copy");
        assert!(!copy.is_home());
        fill(start, len, 9);
        copy.update(view as usize, bytes as usize);
        let new = [0xc3u8; 0x800];
        let from = source(host, &new);
        let hold = QueueHold::new(host);
        let one = writer(host, &home, view, &from, bytes);
        host.send_for_test(&one);
        let two = host.new_list().unwrap();
        copy.refresh(&two, view, bytes);
        two.close();
        assert_eq!(
            crate::host::flights::blocked(&[&two]),
            None,
            "no read of the CPU's"
        );
        assert!(crate::host::flights::refresh_waits(&[&two]));
        let counted = crate::host::flights::r5_stats()[0];
        let released = hold.released().after(200);
        host.send_for_test(&two);
        released.join().unwrap();
        assert_eq!(crate::host::flights::r5_stats()[0], counted + 1, "counted");
        assert!(
            same(view, crate::shadow::copy_bytes(&copy)),
            "the refresh came after the flight: the card's bytes"
        );
        assert!(same(view, &new));
        // A refresh of the home itself does not wait.
        let three = host.new_list().unwrap();
        home.refresh(&three, view, bytes);
        three.close();
        let hold = QueueHold::new(host);
        let four = writer(host, &home, view, &from, bytes);
        host.send_for_test(&four);
        assert!(!crate::host::flights::refresh_waits(&[&three]));
        hold.release();
        flights.land_all();
        for list in [&one, &two, &three, &four] {
            list.reset();
        }
        drop((one, two, three, four, hold));
        unregister(0xc0de_5401, start);
        drop((home, copy));
        let _ = region;
    }

    /// The title frees a home's pages while a flight's list holds the home:
    /// told TRUE, the pages wait, and they are decommitted once the flight
    /// landed and its list was let go, without a sweep. The title protects
    /// another home's pages while a flight holds that one: it waits for the
    /// flights in the air, succeeds, and the region is not blacklisted.
    #[test]
    fn warp_r5_free_and_protect_with_a_flight_holding_the_home() {
        let Some(host) = warp13() else { return };
        let (_core, flights) = r5_on();
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_5601, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let hold = QueueHold::new(host);
        let list = host.new_list().unwrap();
        home.refresh(&list, start, len);
        list.close();
        host.send_owned_for_test(list);
        drop(home);
        let waiting = TOLD[2].load(Ordering::Relaxed);
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1,
            "told it was done"
        );
        assert_eq!(TOLD[2].load(Ordering::Relaxed), waiting + 1, "done later");
        assert_eq!(state(start), mem::MEM_COMMIT, "held by the flight's list");
        hold.release();
        flights.land_all();
        assert_eq!(
            state(start),
            mem::MEM_RESERVE,
            "decommitted after the landing, with no sweep"
        );
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        unregister(0xc0de_5601, start);
        // A protection, with a flight holding the home.
        let (start, len) = (region.base + 4 * K64, K64);
        register(0xc0de_5602, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let hold = QueueHold::new(host);
        let list = host.new_list().unwrap();
        home.refresh(&list, start, len);
        list.close();
        host.send_owned_for_test(list);
        drop(home);
        let (waited, refused) = (
            TOLD[11].load(Ordering::Relaxed),
            TOLD[7].load(Ordering::Relaxed),
        );
        let released = hold.released().after(200);
        let mut old = 0;
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READONLY, &mut old),
            1,
            "done once the flight landed"
        );
        released.join().unwrap();
        assert_eq!(TOLD[11].load(Ordering::Relaxed), waited + 1, "it waited");
        assert_eq!(TOLD[7].load(Ordering::Relaxed), refused, "not refused");
        assert_eq!(
            title_virtual_protect(start as usize, 4096, mem::PAGE_READWRITE, &mut old),
            1
        );
        assert_eq!(old, mem::PAGE_READONLY);
        let again = find(start, len).expect("a home again");
        assert!(again.is_home(), "the region is not blacklisted");
        drop(again);
        unregister(0xc0de_5602, start);
        drop(hold);
        region.release();
    }

    /// One scripted sequence of submissions, waited for one by one and then
    /// sent as flights: shaders' writes into a home, fills from what earlier
    /// ones wrote, bottom-of-pipe writes and a fence signal behind each: the
    /// same title memory, fills and fence value either way.
    #[test]
    fn warp_r5_matches_waited_submissions() {
        let Some(host) = warp13() else { return };
        const N: usize = 50;
        let run = |flights: bool| -> (u64, Vec<Vec<u8>>, u64) {
            let _core = Switched::on(false);
            let flying = flights.then(|| TestFlights::on(true, true));
            let region = TestRegion::new(8 * K64 as usize);
            let (start, len) = (region.base + K64, 4 * K64);
            register(0xc0de_5701, start, len);
            let home = find(start, len).expect("a home");
            assert!(home.is_home());
            fill(start, 6 * K64, 0x21);
            let mut fence = std::ptr::null();
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut fence,
            );
            let mut fills = vec![vec![0u8; 64]; N];
            let mut lists = Vec::new();
            let sources: Vec<_> = (0..N)
                .map(|i| source(host, &[i as u8 ^ 0x5c; 256]))
                .collect();
            for i in 0..N {
                let at = start + ((i as u64 * 9973) % (len - 256)) / 4 * 4;
                let list = host.new_list().unwrap();
                list.defer_refresh(&home, at as usize, 256);
                // SAFETY: an open list of the host's; live buffers.
                unsafe {
                    list.list
                        .CopyBufferRegion(&home.buffer, home.offset(at), &sources[i], 0, 256)
                };
                list.defer_gpu_write(&home, at as usize, 256);
                if i > 0 {
                    // A fill of what the submission before wrote.
                    let before = start + (((i as u64 - 1) * 9973) % (len - 256)) / 4 * 4;
                    list.defer_fill(fills[i].as_mut_ptr(), before as *const u8, 64);
                }
                // A bottom-of-pipe value past the home, in the region.
                list.defer_write(
                    (start + len + 8 * i as u64) as *mut u8,
                    crate::queue::Written::Value(i as u64 * 3 + 1),
                    8,
                );
                list.close();
                if flights {
                    host.send_for_test(&list);
                    if !crate::host::flights::signal_behind(
                        crate::fence::FenceRef::new(fence),
                        i as u64 + 1,
                    ) {
                        crate::fence::set(fence, i as u64 + 1);
                    }
                } else {
                    host.execute(&[&list]);
                    crate::fence::set(fence, i as u64 + 1);
                }
                lists.push(list);
            }
            if let Some(flying) = &flying {
                flying.land_all();
            }
            let bytes = 6 * K64 as usize;
            // SAFETY: the test's own memory.
            let memory = unsafe { std::slice::from_raw_parts(start as *const u8, bytes) };
            let hash = memory.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
            });
            let value = crate::fence::completed(fence);
            for list in &lists {
                list.reset();
            }
            drop(lists);
            crate::object::slot::<extern "system" fn(crate::object::This) -> u32>(fence, 0x10)(
                fence,
            );
            unregister(0xc0de_5701, start);
            drop(home);
            drop(flying);
            region.release();
            (hash, fills, value)
        };
        let waited = run(false);
        let flown = run(true);
        assert_eq!(waited.2, N as u64);
        assert_eq!(flown.2, waited.2, "the same fence value");
        assert_eq!(flown.1, waited.1, "the same fills");
        assert_eq!(flown.0, waited.0, "the same title memory");
    }

    /// With the directory switch on: two flights, each behind a change of a
    /// home's directory entry, both on the queue before either runs. Each
    /// one's list reads the entry as it was when that flight was sent: the
    /// second's change did not reach the first's staging.
    #[test]
    fn warp_r5_directory_follows_queue_order() {
        let Some(host) = warp13() else { return };
        let core = Switched::on(false);
        crate::memory::set_test(true);
        let flights = TestFlights::on(true, true);
        let region = TestRegion::low(12 << 20);
        let cell = region.base.next_multiple_of(4 << 20);
        let (start, len) = (cell + K64 + 0x100, 3 * K64);
        register(0xc0de_5501, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let page = cell + 2 * K64;
        let hold = QueueHold::new(host);
        let one = host.new_list().unwrap();
        crate::memory::ready(&one, &heap, &[page + 0x10], &[], false).expect("ready");
        let table = crate::memory::directory_buffer().unwrap();
        let (out, data) = host.shared_buffer(16).unwrap();
        // SAFETY: the mapping holds 16 bytes; the card does not use it yet.
        unsafe { std::ptr::write_bytes(data, 0xff, 16) };
        let seen = |list: &crate::host::HostList, at: u64| {
            // SAFETY: an open list of the host's; live buffers, the ranges
            // inside both.
            unsafe {
                list.list
                    .CopyBufferRegion(&out, at, &table, page / K64 * 8, 8)
            };
        };
        let (shaders, entry) = crate::memory::entries_of(page).unwrap();
        assert_ne!(entry.0, 0, "numbered for the next submission");
        assert_ne!(shaders, entry, "not yet in the card's table");
        seen(&one, 0);
        one.close();
        host.send_for_test(&one);
        // Changed for the next flight: the home is let go of.
        assert!(crate::memory::forget(&home));
        let two = host.new_list().unwrap();
        seen(&two, 8);
        two.close();
        host.send_for_test(&two);
        assert_eq!(crate::host::flights::in_air(), 2, "both held");
        hold.release();
        flights.land_all();
        let read = |at: usize| {
            // SAFETY: the mapping holds 16 bytes; the card is done with it.
            unsafe { std::slice::from_raw_parts(data.add(at).cast::<u32>(), 2).to_vec() }
        };
        assert_eq!(
            read(0),
            [entry.0, entry.1],
            "the first flight saw its own table"
        );
        assert_eq!(read(8), [0, 0], "the second saw the change");
        assert_eq!(crate::memory::entries_of(page).unwrap(), ((0, 0), (0, 0)));
        one.reset();
        two.reset();
        drop((one, two, hold, flights));
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(region.base, region.base + (12 << 20));
        unregister(0xc0de_5501, start);
        drop((heap, home, out));
        drop(core);
        region.release();
        run_pending();
    }

    /// Six threads (three queues of two threads, as the title has) send
    /// flights over one home, each its own stretch of it with a value
    /// written at the bottom of the pipe, each fill reading what another
    /// thread's flight wrote last. No removal, and title memory ends as the
    /// last flight of each thread left it. `VICTORY_R5_STRESS` submissions
    /// in all (10,000; run with `VICTORY_D3D12_DEBUG=1 VICTORY_D3D12_GBV=1`
    /// for the debug layer and GPU-based validation of the host's device).
    #[test]
    fn warp_r5_stress() {
        let Some(host) = warp13() else { return };
        const THREADS: u64 = 6;
        const STRETCH: u64 = 0x400;
        let total: u64 = std::env::var("VICTORY_R5_STRESS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10_000);
        let each = total.div_ceil(THREADS);
        let (_core, flights) = r5_on();
        let region = TestRegion::new(4 * K64 as usize);
        let (start, len) = (region.base + K64, 2 * K64);
        register(0xc0de_5801, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        fill(start, len, 0);
        let threads: Vec<_> = (0..THREADS)
            .map(|t| {
                let home = home.clone();
                std::thread::spawn(move || {
                    // This thread's submissions fly too.
                    TestFlights::here();
                    TEST_ON.with(|on| on.set(true));
                    let host = crate::host::host().unwrap();
                    let at = start + t * STRETCH;
                    let other = start + ((t + 1) % THREADS) * STRETCH;
                    let sources: Vec<_> = (0..4u8)
                        .map(|k| source(host, &[t as u8 * 16 + k; STRETCH as usize]))
                        .collect();
                    let mut out = vec![0u8; 16];
                    let mut kept = Vec::new();
                    let mut last = 0;
                    for i in 0..each {
                        let list = host.new_list().unwrap();
                        list.defer_refresh(&home, at as usize, STRETCH as usize);
                        // SAFETY: an open list of the host's; live buffers.
                        unsafe {
                            list.list.CopyBufferRegion(
                                &home.buffer,
                                home.offset(at),
                                &sources[(i % 4) as usize],
                                0,
                                STRETCH,
                            )
                        };
                        list.defer_gpu_write(&home, at as usize, STRETCH as usize);
                        if i % 7 == 0 {
                            list.defer_fill(out.as_mut_ptr(), other as *const u8, 16);
                        }
                        list.defer_write(
                            (start + len + 8 * t) as *mut u8,
                            crate::queue::Written::Value(i + 1),
                            8,
                        );
                        list.close();
                        last = host.send_for_test(&list);
                        // Kept (not reset) until its flight has landed.
                        kept.push(list);
                        if kept.len() > 64 {
                            crate::host::flights::land_through(last);
                            for list in kept.drain(..) {
                                list.reset();
                            }
                        }
                    }
                    crate::host::flights::land_through(last);
                    for list in kept {
                        list.reset();
                    }
                    // The last fill read the other thread's stretch whole, as
                    // one of its flights left it (or as it was filled, if
                    // none had landed yet).
                    let next = (t + 1) % THREADS;
                    let initial: Vec<u8> = (0..16u8).map(|j| j.wrapping_mul(13)).collect();
                    assert!(
                        out == initial
                            || (out.iter().all(|&b| b == out[0])
                                && u64::from(out[0] / 16) == next
                                && out[0] % 16 < 4),
                        "a fill of thread {t} read {out:?}"
                    );
                    TEST_ON.with(|on| on.set(false));
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("a stress thread");
        }
        flights.land_all();
        // SAFETY: a plain query.
        assert!(
            unsafe { host.device.GetDeviceRemovedReason() }.is_ok(),
            "no removal"
        );
        for t in 0..THREADS {
            let last = (each - 1) % 4;
            assert!(
                same(
                    start + t * STRETCH,
                    &[t as u8 * 16 + last as u8; STRETCH as usize]
                ),
                "thread {t}'s stretch as its last flight wrote it"
            );
            // SAFETY: the test's own memory.
            let value = unsafe { ((start + len + 8 * t) as *const u64).read_unaligned() };
            assert_eq!(value, each, "thread {t}'s last bottom-of-pipe value");
        }
        unregister(0xc0de_5801, start);
        drop(home);
        drop(flights);
        region.release();
    }

    /// The latch counts refusals in a row: an opening that succeeds between
    /// them ends the row.
    #[test]
    fn refusal_latch_counts_only_refusals_in_a_row() {
        let _one = serial();
        // The latch as it was, also if an assertion fails: set, it would
        // keep every other test's range a copy.
        struct Restore(bool);
        impl Drop for Restore {
            fn drop(&mut self) {
                opened();
                LATCHED.store(self.0, Ordering::Relaxed);
            }
        }
        let _restore = Restore(LATCHED.swap(false, Ordering::Relaxed));
        opened();
        let error = windows::core::Error::from(windows::core::HRESULT(0x8007_000e_u32 as i32));
        refused(&error, 0, 1);
        refused(&error, 0, 1);
        opened();
        refused(&error, 0, 1);
        refused(&error, 0, 1);
        assert!(!LATCHED.load(Ordering::Relaxed), "two, an opening, two");
        refused(&error, 0, 1);
        assert!(LATCHED.load(Ordering::Relaxed), "three in a row");
    }

    /// A title call in a region marks it busy while the host is asked, also
    /// in a region that never had a home: no home is claimed there meanwhile.
    #[test]
    fn a_call_in_a_region_holds_back_claims_while_the_host_is_asked() {
        let _one = serial();
        let region = TestRegion::new(4 * K64 as usize);
        let base = region.base as usize;
        assert_eq!(busy(base, 0), (true, false), "a region, never a home");
        let mut card = || false;
        let mut ask = LiveAsk {
            start: region.base,
            end: region.base + K64,
            at: region.base,
            bytes: K64,
            base,
            card: &mut card,
            pool: false,
            reading: None,
        };
        assert_eq!(ask.claim(), Err(Why::Busy), "busy: nothing claimed");
        unbusy(base);
        assert_eq!(ask.claim(), Ok(()), "claimed once the call is done");
        unclaim(region.base, K64);
        region.release();
    }

    /// A free the host would refuse for its own shape is the host's
    /// refusal, also with a heap standing in the region: it does not wait,
    /// and the title is not told it was done.
    #[test]
    fn warp_core_a_free_the_host_refuses_for_its_shape_does_not_wait() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_1301, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home());
        let waiting = TOLD[2].load(Ordering::Relaxed);
        for (at, len, kind) in [
            // Past the region's end.
            (region.base + 7 * K64, 2 * K64, mem::MEM_DECOMMIT),
            // A release with a length, or not from the base.
            (region.base, K64, mem::MEM_RELEASE),
            (start, 0, mem::MEM_RELEASE),
        ] {
            // SAFETY: a plain write of this thread's last error.
            unsafe { SetLastError(0) };
            assert_eq!(
                title_virtual_free(at as usize, len as usize, kind),
                0,
                "{at:#x}+{len:#x} {kind:#x}"
            );
            // SAFETY: a plain read of it.
            assert_ne!(unsafe { GetLastError() }, 0, "the host's own error");
        }
        assert_eq!(TOLD[2].load(Ordering::Relaxed), waiting, "none waits");
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        unregister(0xc0de_1301, start);
        drop(held);
        region.release();
    }

    /// A small buffer's home pins the whole 64 KiB page around it: the
    /// title's protection and decommit of a neighbour 4 KiB page in that
    /// page, outside the buffer, let the home go too and are done at once.
    #[test]
    fn warp_core_a_neighbour_page_in_a_homes_64k_lets_it_go() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64 + 0x100, 0x200);
        let page = (region.base + K64 + 0x8000) as usize;
        register(0xc0de_1501, start, len);
        assert!(find(start, len).expect("a home").is_home());
        let mut old = 0;
        assert_eq!(
            title_virtual_protect(page, 4096, mem::PAGE_READONLY, &mut old),
            1,
            "the protection of a neighbour page is made"
        );
        assert_eq!(
            title_virtual_protect(page, 4096, mem::PAGE_READWRITE, &mut old),
            1
        );
        assert!(find(start, len).expect("a home again").is_home());
        let waiting = TOLD[2].load(Ordering::Relaxed);
        assert_eq!(title_virtual_free(page, 4096, mem::MEM_DECOMMIT), 1);
        assert_eq!(state(page as u64), mem::MEM_RESERVE, "decommitted at once");
        assert_eq!(TOLD[2].load(Ordering::Relaxed), waiting, "nothing waited");
        assert!(
            core()
                .unnumber
                .contains(&(page as u64, (page + 4096) as u64)),
            "the sweep takes it out of the directory"
        );
        run_pending();
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        unregister(0xc0de_1501, start);
        region.release();
    }

    /// Reserves `bytes` at `at` again, as the title's next `XMemVirtualAlloc`
    /// may, and tells the core: the region a release ended, reused.
    fn reuse(at: u64, bytes: usize) -> TestRegion {
        // SAFETY: addresses the test's own allocation released.
        let again = unsafe {
            mem::VirtualAlloc(
                at as *const c_void,
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT | MEM_WRITE_WATCH,
                mem::PAGE_READWRITE,
            )
        } as u64;
        assert_eq!(again, at, "the released addresses are free");
        title_memory_region(at as usize, bytes, mem::MEM_RESERVE, 3, 0x404);
        TestRegion { base: at }
    }

    /// A release done while a decommit of the region waits ends that wait:
    /// after the title reserves the same addresses again the sweep frees
    /// none of its new pages, and the new region is not left busy. Both
    /// ways: the release done at once, and the release itself waiting.
    #[test]
    fn warp_core_a_release_ends_the_regions_waiting_frees() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        const BYTES: usize = 8 * K64 as usize;
        for release_waits in [false, true] {
            let region = TestRegion::new(BYTES);
            let base = region.base;
            let (start, len) = (base + K64, K64);
            register(0xc0de_1601, start, len);
            let held = find(start, len).expect("a home");
            assert!(held.is_home());
            if release_waits {
                assert_eq!(title_virtual_free(base as usize, 0, mem::MEM_RELEASE), 1);
            }
            assert_eq!(
                title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
                1
            );
            assert_eq!(
                PENDING.load(Ordering::Relaxed),
                1 + u64::from(release_waits),
                "waiting"
            );
            unregister(0xc0de_1601, start);
            drop(held);
            let failed = TOLD[4].load(Ordering::Relaxed);
            if release_waits {
                run_pending();
            } else {
                assert_eq!(title_virtual_free(base as usize, 0, mem::MEM_RELEASE), 1);
            }
            assert_eq!(state(base), mem::MEM_FREE, "released");
            assert_eq!(
                PENDING.load(Ordering::Relaxed),
                0,
                "the decommit went with it"
            );
            assert!(core().pending.is_empty());
            assert_eq!(
                TOLD[4].load(Ordering::Relaxed),
                failed,
                "never asked of the host"
            );
            let again = reuse(base, BYTES);
            fill(start, len, 11);
            run_pending();
            assert_eq!(state(start), mem::MEM_COMMIT, "the new pages stay");
            let mut want = vec![0u8; len as usize];
            for (i, b) in want.iter_mut().enumerate() {
                *b = (i as u8).wrapping_mul(13).wrapping_add(11);
            }
            assert!(same(start, &want), "with the title's bytes");
            assert_eq!(
                core().region_at(base as usize).map(|(_, r)| r.busy),
                Some(0),
                "the new region is not busy"
            );
            again.release();
        }
    }

    /// A free still refused after it waited long enough is given up: the
    /// region gets no home again, and its pages read zeros.
    #[test]
    fn warp_core_a_free_that_waits_too_long_is_given_up() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::new(8 * K64 as usize);
        let (start, len) = (region.base + K64, K64);
        register(0xc0de_1401, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home());
        fill(start, len, 7);
        assert_eq!(
            title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1
        );
        let given_up = TOLD[9].load(Ordering::Relaxed);
        run_pending_after(0);
        assert_eq!(TOLD[9].load(Ordering::Relaxed), given_up + 1, "given up");
        assert_eq!(PENDING.load(Ordering::Relaxed), 0);
        assert!(same(start, &[0u8; K64 as usize]), "its pages read zeros");
        let blacklisted = turned(Why::Blacklisted);
        register(0xc0de_1402, region.base + 4 * K64, K64);
        let copy = find(region.base + 4 * K64, K64).expect("a copy");
        assert!(!copy.is_home(), "the region gets no home again");
        assert_eq!(turned(Why::Blacklisted), blacklisted + 1);
        unregister(0xc0de_1401, start);
        unregister(0xc0de_1402, region.base + 4 * K64);
        drop((held, copy));
        region.release();
    }

    /// A section as `xmem` makes a chunk of the memory pool, told to the GPU
    /// side as one; gone (and told) when dropped.
    struct TestChunk {
        handle: usize,
        committed: bool,
    }

    impl TestChunk {
        fn new(bytes: usize, how: crate::pool::PoolChunk) -> TestChunk {
            let committed = how == crate::pool::PoolChunk::Committed;
            // SAFETY: plain section creation.
            let handle = unsafe {
                mem::CreateFileMappingW(
                    windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    mem::PAGE_READWRITE
                        | if committed {
                            mem::SEC_COMMIT
                        } else {
                            mem::SEC_RESERVE
                        },
                    0,
                    bytes as u32,
                    std::ptr::null(),
                )
            } as usize;
            assert_ne!(handle, 0);
            crate::inplace::title_memory_section(handle, bytes as u64, committed);
            crate::pool::title_memory_pool_chunk(handle, bytes as u64, how);
            TestChunk { handle, committed }
        }
    }

    impl Drop for TestChunk {
        fn drop(&mut self) {
            crate::pool::title_memory_pool_chunk_gone(self.handle);
            // SAFETY: the test's own handle; views keep the section alive.
            unsafe { windows_sys::Win32::Foundation::CloseHandle(self.handle as _) };
        }
    }

    /// Title addresses as `xmem` reserves them for physical pages: one
    /// placeholder, mapped and unmapped as `xmem` maps pages.
    struct Place {
        base: u64,
        bytes: usize,
    }

    /// A reservation of `bytes` below 128 GiB (where `crate::memory`'s
    /// directory reaches), of `kind` and `protect`.
    fn low(bytes: usize, kind: u32, protect: u32) -> u64 {
        let mut requirements = mem::MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (4usize << 30) as *mut c_void,
            HighestEndingAddress: ((1usize << 37) - 1) as *mut c_void,
            Alignment: 0,
        };
        let mut parameter = mem::MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = mem::MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // SAFETY: a reservation of the test's own; the parameter points at
        // `requirements`, which outlives the call.
        let base = unsafe {
            mem::VirtualAlloc2(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                std::ptr::null(),
                bytes,
                kind,
                protect,
                &mut parameter,
                1,
            )
        } as u64;
        assert_ne!(base, 0);
        base
    }

    impl Place {
        /// [`Place::new`] below 128 GiB.
        fn low(bytes: usize) -> Place {
            let base = low(
                bytes,
                mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                mem::PAGE_NOACCESS,
            );
            Place { base, bytes }
        }

        fn new(bytes: usize) -> Place {
            // SAFETY: a reservation of the test's own.
            let base = unsafe {
                mem::VirtualAlloc2(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    std::ptr::null(),
                    bytes,
                    mem::MEM_RESERVE | mem::MEM_RESERVE_PLACEHOLDER,
                    mem::PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                )
            } as u64;
            assert_ne!(base, 0);
            Place { base, bytes }
        }

        /// Shows `len` bytes of `chunk` from `offset` on at `base + at`, as
        /// `xmem`'s `map` does: under the remapping lock, the GPU side told.
        fn map(&self, at: u64, len: u64, chunk: &TestChunk, offset: u64) {
            let address = self.base + at;
            let _changing =
                crate::watch::title_memory_remapping_range(address as usize, len as usize);
            // SAFETY: splits the test's placeholder (failing harmlessly when
            // it is the whole of one), then maps the section over that part.
            let view = unsafe {
                mem::VirtualFree(
                    address as *mut c_void,
                    len as usize,
                    mem::MEM_RELEASE | mem::MEM_PRESERVE_PLACEHOLDER,
                );
                mem::MapViewOfFile3(
                    chunk.handle as _,
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    address as *const c_void,
                    offset,
                    len as usize,
                    mem::MEM_REPLACE_PLACEHOLDER,
                    mem::PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as u64, address);
            if !chunk.committed {
                // SAFETY: the view just mapped, of a reserved section.
                let done = unsafe {
                    mem::VirtualAlloc(
                        address as *const c_void,
                        len as usize,
                        mem::MEM_COMMIT,
                        mem::PAGE_READWRITE,
                    )
                };
                assert!(!done.is_null());
            }
            crate::watch::title_memory_remapped(address as usize, len as usize);
            crate::inplace::title_memory_view(address as usize, len as usize, chunk.handle, offset);
        }

        /// Takes the view at `base + at` (of `len` bytes) away, as `xmem`'s
        /// `unmap` does.
        fn unmap(&self, at: u64, len: u64) {
            let address = self.base + at;
            let _changing =
                crate::watch::title_memory_remapping_range(address as usize, len as usize);
            crate::watch::title_memory_remapped(address as usize, len as usize);
            // SAFETY: a view the test mapped; it becomes a placeholder again.
            unsafe {
                mem::UnmapViewOfFile2(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    mem::MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: address as *mut c_void,
                    },
                    mem::MEM_PRESERVE_PLACEHOLDER,
                )
            };
        }
    }

    impl Drop for Place {
        fn drop(&mut self) {
            /// Joins placeholders (not in windows-sys' memory module).
            const MEM_COALESCE_PLACEHOLDERS: u32 = 1;
            // SAFETY: the test's placeholders, every view unmapped: joined
            // again, then released.
            unsafe {
                mem::VirtualFree(
                    self.base as *mut c_void,
                    self.bytes,
                    mem::MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
                );
                mem::VirtualFree(self.base as *mut c_void, 0, mem::MEM_RELEASE);
            }
        }
    }

    /// A title buffer in one run of a committed chunk of the pool is a home
    /// in the chunk's heap at the run's offset: the GPU reads the title's
    /// bytes there, and what the CPU wrote since, with nothing copied.
    #[test]
    fn warp_pool_a_buffer_in_one_run_of_a_chunk_is_a_home_at_its_offset() {
        let Some(host) = warp13() else { return };
        let _on = Switched::pool();
        let chunk = TestChunk::new(8 * K64 as usize, crate::pool::PoolChunk::Committed);
        let place = Place::new(4 * K64 as usize);
        place.map(0, 4 * K64, &chunk, 2 * K64);
        let (start, len) = (place.base + K64 + 0x40, 2 * K64);
        register(0xc0de_2001, start, len);
        let home = find(start, len).expect("a home");
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        assert_eq!(
            home.offset(start),
            3 * K64 + 0x40,
            "at the run's offset in the chunk"
        );
        let mut rig = Rig::new(&host.device).unwrap();
        fill(start, len, 3);
        let got = rig
            .fetch(&home.buffer, home.offset(start), len as usize)
            .unwrap();
        assert!(same(start, &got), "the GPU reads the title's bytes");
        fill(start, len, 77);
        let got = rig
            .fetch(&home.buffer, home.offset(start), len as usize)
            .unwrap();
        assert!(same(start, &got), "and what the CPU wrote since");
        assert_eq!(
            home.update(start as usize, len as usize),
            0,
            "a refresh copies nothing"
        );
        assert!(home.unwatched(), "no write-watch asking, no guard");
        // A window over the same run is the same chunk's buffer.
        let window = find(place.base + 2 * K64 + 0x10, 16).expect("a window");
        assert_eq!(window.home_kind(), Some(Kind::Pool));
        unregister(0xc0de_2001, start);
        drop((home, window, rig));
        place.unmap(0, 4 * K64);
    }

    /// A buffer view made into a view heap slot over a C-pool home is made
    /// on the home, as every other use of the buffer is; a remap that lets
    /// the home go has the slot made anew at its next use, on what the
    /// addresses show now, and walked as changed by the heap buffer pass.
    #[test]
    fn warp_pool_a_slot_on_a_home_is_made_anew_after_a_remap() {
        use windows::core::Interface;
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::new(2 * K64 as usize);
        place.map(0, 2 * K64, &a, 0);
        let (start, len) = (place.base + 0x100, 0x1000);
        register(0xc0de_2501, start, len);
        let home = find(start, len).expect("a home");
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        // The buffer as the title placed it (dimension 1, its width), and a
        // view of 64 of its elements of 16 bytes: format, dimension (1:
        // buffer), mapping, a pad, first element, elements, stride, flags.
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 64, 16, 0, 0]);
        (desc[10], desc[11]) = (start as u32, (start >> 32) as u32);
        let view = crate::descriptor::View::Placed(
            false,
            [1, 0, 0, 0, len as u32, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
            desc,
        );
        let mut records = vec![[0u32; 8]; 4];
        let handle = records.as_mut_ptr() as u64 + 32;
        crate::descriptor::record(handle, view);
        heap.mirror(1, handle, true);
        assert!(heap.made(1));
        assert_eq!(heap.kept_raw(1), Some(home.buffer.as_raw()), "on the home");
        drop(heap.buffer_changes());
        let (anew, _) = crate::pool::told_counts();
        // The title maps other pages there and writes no descriptor.
        place.unmap(0, 2 * K64);
        place.map(0, 2 * K64, &b, 0);
        heap.pool_slots_let_go();
        assert!(!heap.made(1), "made from no record");
        assert_eq!(heap.buffer_changes(), [1], "walked as changed");
        assert_eq!(crate::pool::told_counts().0, anew + 1);
        heap.mirror(1, handle, true);
        let new = find(start, len).expect("a home");
        assert!(!Arc::ptr_eq(&home, &new), "the remap let the first go");
        assert!(heap.made(1));
        assert_eq!(
            heap.kept_raw(1),
            Some(new.buffer.as_raw()),
            "made anew on the new home"
        );
        unregister(0xc0de_2501, start);
        drop((heap, home, new, records));
        place.unmap(0, 2 * K64);
    }

    /// The GPU wrote two C-pool homes side by side at the title's
    /// addresses, in two chunks: the copies over the same pages at other
    /// views are told for each (the two ranges are no one run, and looked
    /// for together their other views were missed).
    #[test]
    fn warp_pool_gpu_writes_are_told_at_other_views_per_home() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        // The same pages at two places: `p`, where the homes are, and `q`.
        let (p, q) = (Place::new(2 * K64 as usize), Place::new(2 * K64 as usize));
        for place in [&p, &q] {
            place.map(0, K64, &a, 0);
            place.map(K64, K64, &b, 0);
        }
        let len = 0x1000;
        let (first, second) = (p.base + K64 - len, p.base + K64);
        register(0xc0de_2601, first, len);
        register(0xc0de_2602, second, len);
        let one = find(first, len).expect("a home");
        let two = find(second, len).expect("a home");
        assert_eq!(one.home_kind(), Some(Kind::Pool));
        assert_eq!(two.home_kind(), Some(Kind::Pool));
        // Copies over both homes' pages at `q`.
        let copies = [
            crate::shadow::exact(q.base + K64 - len, len).expect("a copy"),
            crate::shadow::exact(q.base + K64, len).expect("a copy"),
        ];
        let before = TOLD[8].load(Ordering::Relaxed);
        let range = |at: u64| vec![(at as usize, (at + len) as usize)];
        after_wait(&[(one.clone(), range(first)), (two.clone(), range(second))]);
        assert_eq!(
            TOLD[8].load(Ordering::Relaxed),
            before + 2,
            "both copies told"
        );
        unregister(0xc0de_2601, first);
        unregister(0xc0de_2602, second);
        drop((one, two, copies));
        for place in [&p, &q] {
            place.unmap(0, K64);
            place.unmap(K64, K64);
        }
    }

    /// With `VICTORY_MEMORY_CARDS=1` (`crate::card_home`) a big buffer's
    /// copy in the card's memory can have a C-pool home as its CPU side.
    /// What is merged into that home shows at another view of its pages:
    /// told there, a copy at that view takes the pages again.
    #[test]
    fn warp_pool_a_card_copy_on_a_pool_home_tells_its_other_views() {
        let Some(host) = warp13() else { return };
        if !host.card_ready() {
            return;
        }
        let _on = Switched::pool();
        crate::card_home::set_test(true);
        let bytes = 32 * K64;
        let chunk = TestChunk::new(2 * bytes as usize, crate::pool::PoolChunk::Committed);
        // The same pages at two places: `p`, where the card copy is, and `q`.
        let (p, q) = (Place::new(bytes as usize), Place::new(bytes as usize));
        for place in [&p, &q] {
            place.map(0, bytes, &chunk, 0);
        }
        let (start, len) = (p.base + K64, 1 << 20);
        register(0xc0de_2701, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(copy.carded(), "in the card's memory");
        let home = copy.card_home().expect("its CPU side is a home").clone();
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        assert!(copy.home_kind().is_none(), "the card copy is no home itself");
        let at = 3 * K64 + 0x100;
        let (view, small) = (q.base + K64 + at, 0x1000u64);
        let other = crate::shadow::exact(view, small).expect("an exact copy");
        assert!(!other.is_home());
        fill(start, len, 9);
        other.update(view as usize, small as usize);
        assert!(same(view, crate::shadow::copy_bytes(&other)));
        // The merge's store into the home, as the card makes it.
        let new = [0xa5u8; 0x1000];
        let mut rig = Rig::new(&host.device).unwrap();
        rig.store(&home.buffer, home.offset(start + at), &new).unwrap();
        assert!(same(view, &new), "the same pages");
        copy.card_merged_shown(at as usize, (at + small) as usize);
        other.update(view as usize, small as usize);
        assert!(
            same(view, crate::shadow::copy_bytes(&other)),
            "told, the other view's copy takes the pages again"
        );
        unregister(0xc0de_2701, start);
        drop((home, copy, other, rig));
        for place in [&p, &q] {
            place.unmap(0, bytes);
        }
    }

    /// A buffer over views of two chunks keeps its copy (one placed buffer
    /// cannot show two heaps); one inside the first view alone is a home.
    #[test]
    fn warp_pool_a_view_across_two_chunks_stays_a_copy() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::new(4 * K64 as usize);
        place.map(0, 2 * K64, &a, 0);
        place.map(2 * K64, 2 * K64, &b, 0);
        let scattered = turned(Why::Scattered);
        let (start, len) = (place.base + K64, 2 * K64);
        register(0xc0de_2101, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home());
        assert_eq!(turned(Why::Scattered), scattered + 1);
        let (inside, small) = (place.base + 0x100, 0x1000);
        register(0xc0de_2102, inside, small);
        let home = find(inside, small).expect("a home");
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        unregister(0xc0de_2101, start);
        unregister(0xc0de_2102, inside);
        drop((copy, home));
        place.unmap(0, 2 * K64);
        place.unmap(2 * K64, 2 * K64);
    }

    /// Chunks of the smallest size `VICTORY_MEMORY_POOL_CHUNK_MB` takes
    /// (16 MiB) side by side: a buffer inside one is a home in it at its
    /// offset, and one across their seam keeps its copy.
    #[test]
    fn warp_pool_small_chunks_are_homes_and_a_view_across_them_stays_a_copy() {
        const MIB16: u64 = 16 << 20;
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        let (a, b) = (
            TestChunk::new(MIB16 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(MIB16 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::new(2 * MIB16 as usize);
        place.map(0, MIB16, &a, 0);
        place.map(MIB16, MIB16, &b, 0);
        let (inside, len) = (place.base + MIB16 + 3 * K64, K64);
        register(0xc0de_2111, inside, len);
        let home = find(inside, len).expect("a home");
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        assert_eq!(
            home.offset(inside),
            3 * K64,
            "at its offset in the second chunk"
        );
        let scattered = turned(Why::Scattered);
        let across = place.base + MIB16 - K64;
        register(0xc0de_2112, across, 2 * K64);
        let copy = find(across, 2 * K64).expect("a copy");
        assert!(!copy.is_home(), "across the seam: a copy");
        assert_eq!(turned(Why::Scattered), scattered + 1);
        unregister(0xc0de_2111, inside);
        unregister(0xc0de_2112, across);
        drop((home, copy));
        place.unmap(0, MIB16);
        place.unmap(MIB16, MIB16);
    }

    /// With `VICTORY_MEMORY_POOL_OPEN_PCT`, a chunk is opened only once
    /// `xmem` tells that the title maps its share of it: before, a buffer in
    /// it keeps its copy, counted for that, and the chunk stays unopened;
    /// after, the next buffer is a home; and an opened chunk stays opened
    /// when the share drops again.
    #[test]
    fn warp_pool_a_chunk_opens_only_once_the_title_maps_its_share() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        crate::pool::set_test_open_pct(50);
        let chunk = TestChunk::new(4 * K64 as usize, crate::pool::PoolChunk::Committed);
        let opened = || crate::pool::title_memory_pool_chunk_known(chunk.handle);
        let place = Place::new(4 * K64 as usize);
        place.map(0, K64, &chunk, 0);
        let below = turned(Why::BelowShare);
        let (start, len) = (place.base + 0x100, 0x1000);
        register(0xc0de_2121, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home(), "a quarter mapped: a copy");
        assert_eq!(turned(Why::BelowShare), below + 1);
        assert_eq!(opened(), Some(false), "not opened");
        // A second page mapped: half of it, which `xmem` tells.
        place.map(K64, K64, &chunk, K64);
        crate::pool::title_memory_pool_chunk_mapped(chunk.handle, true);
        let (next, small) = (place.base + K64 + 0x200, 0x1000);
        register(0xc0de_2122, next, small);
        let home = find(next, small).expect("a home");
        assert_eq!(home.home_kind(), Some(Kind::Pool));
        assert_eq!(home.offset(next), K64 + 0x200);
        assert_eq!(opened(), Some(true), "opened");
        // Less than the share again: the chunk stays opened.
        crate::pool::title_memory_pool_chunk_mapped(chunk.handle, false);
        let third = place.base + 0x4000;
        register(0xc0de_2123, third, small);
        let again = find(third, small).expect("a home");
        assert_eq!(again.home_kind(), Some(Kind::Pool));
        assert_eq!(turned(Why::BelowShare), below + 1, "not asked of the share");
        unregister(0xc0de_2121, start);
        unregister(0xc0de_2122, next);
        unregister(0xc0de_2123, third);
        drop((copy, home, again));
        place.unmap(0, K64);
        place.unmap(K64, K64);
    }

    /// A chunk made reserved (no room in the commit charge) keeps its views'
    /// copies, counted; and with the pool off, view memory is counted as
    /// before.
    #[test]
    fn warp_pool_a_reserved_chunk_keeps_its_copies() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::pool();
        let chunk = TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::ShortOfCommit);
        let place = Place::new(2 * K64 as usize);
        place.map(0, 2 * K64, &chunk, 0);
        let reserved = turned(Why::ReservedChunk);
        let (start, len) = (place.base + 0x80, K64);
        register(0xc0de_2201, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home());
        assert_eq!(turned(Why::ReservedChunk), reserved + 1);
        unregister(0xc0de_2201, start);
        drop(copy);
        let committed = TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed);
        place.unmap(0, 2 * K64);
        place.map(0, 2 * K64, &committed, 0);
        crate::pool::set_test(false);
        let view = turned(Why::View);
        register(0xc0de_2202, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(!copy.is_home(), "the pool off: a copy");
        assert_eq!(turned(Why::View), view + 1);
        unregister(0xc0de_2202, start);
        drop(copy);
        place.unmap(0, 2 * K64);
    }

    /// A remap lets the home go: the next ask is a home in what the
    /// addresses show now. A home a list still holds keeps reading the pages
    /// it was made on through its heap, also once the view is gone and its
    /// section closed.
    #[test]
    fn warp_pool_a_remap_lets_the_home_go_and_a_held_home_keeps_its_pages() {
        let Some(host) = warp13() else { return };
        let _on = Switched::pool();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::new(2 * K64 as usize);
        place.map(0, 2 * K64, &a, 0);
        let (start, len) = (place.base + 0x200, 0x2000);
        fill(start, len, 1);
        let old: Vec<u8> = (0..len)
            .map(|i| (i as u8).wrapping_mul(13).wrapping_add(1))
            .collect();
        register(0xc0de_2301, start, len);
        let held = find(start, len).expect("a home");
        assert_eq!(held.home_kind(), Some(Kind::Pool));
        // The title maps other pages there.
        place.unmap(0, 2 * K64);
        place.map(0, 2 * K64, &b, 0);
        fill(start, len, 50);
        let new = find(start, len).expect("a home");
        assert!(!Arc::ptr_eq(&held, &new), "the remap let the first go");
        assert_eq!(new.home_kind(), Some(Kind::Pool));
        let mut rig = Rig::new(&host.device).unwrap();
        let got = rig
            .fetch(&new.buffer, new.offset(start), len as usize)
            .unwrap();
        assert!(
            same(start, &got),
            "the new home shows what the addresses show"
        );
        // The first, held: its heap still is the first section.
        place.unmap(0, 2 * K64);
        drop(a);
        let got = rig
            .fetch(&held.buffer, held.offset(start), len as usize)
            .unwrap();
        assert_eq!(got, old, "a held home reads the pages it was made on");
        unregister(0xc0de_2301, start);
        drop((held, new, rig, b));
    }

    /// Maps, unmaps and remaps on one thread while another resolves homes
    /// over the same addresses and reads them on the GPU: no fault, no
    /// deadlock, every home shows one of the two chunks' bytes whole, and
    /// once the remaps stop the home shows what the addresses show.
    #[test]
    fn warp_pool_remaps_race_with_homes_the_gpu_holds() {
        let Some(host) = warp13() else { return };
        let _on = Switched::pool();
        let bytes = 2 * K64 as usize;
        let (a, b) = (
            Arc::new(TestChunk::new(bytes, crate::pool::PoolChunk::Committed)),
            Arc::new(TestChunk::new(bytes, crate::pool::PoolChunk::Committed)),
        );
        let place = Arc::new(Place::new(bytes));
        // Each chunk's bytes, written through the first mapping of it.
        for (chunk, seed) in [(&a, 0x11u8), (&b, 0x22)] {
            place.map(0, bytes as u64, chunk, 0);
            // SAFETY: mapped read-write.
            unsafe { std::ptr::write_bytes(place.base as *mut u8, seed, bytes) };
            place.unmap(0, bytes as u64);
        }
        place.map(0, bytes as u64, &a, 0);
        let stop = Arc::new(AtomicBool::new(false));
        let remapper = {
            let (place, a, b, stop) = (place.clone(), a.clone(), b.clone(), stop.clone());
            std::thread::spawn(move || {
                pool_on_here();
                let mut on_a = true;
                let mut n = 0u32;
                while !stop.load(Ordering::Relaxed) && n < 400 {
                    place.unmap(0, bytes as u64);
                    on_a = !on_a;
                    place.map(0, bytes as u64, if on_a { &a } else { &b }, 0);
                    n += 1;
                }
                on_a
            })
        };
        let (start, len) = (place.base + 0x100, 0x1000);
        let mut rig = Rig::new(&host.device).unwrap();
        let mut homes = 0;
        for k in 0..60usize {
            register(0xc0de_2400 + k, start, len);
            if let Some(found) = find(start, len)
                && found.is_home()
            {
                homes += 1;
                let got = rig
                    .fetch(&found.buffer, found.offset(start), len as usize)
                    .unwrap();
                assert!(
                    got.iter().all(|&x| x == 0x11) || got.iter().all(|&x| x == 0x22),
                    "a home shows one chunk's pages whole"
                );
            }
            unregister(0xc0de_2400 + k, start);
        }
        stop.store(true, Ordering::Relaxed);
        let on_a = remapper.join().expect("the remapper never faulted");
        register(0xc0de_24ff, start, len);
        let last = find(start, len).expect("a home");
        assert!(last.is_home());
        let got = rig
            .fetch(&last.buffer, last.offset(start), len as usize)
            .unwrap();
        assert!(
            got.iter().all(|&x| x == if on_a { 0x11 } else { 0x22 }),
            "the home shows what the addresses show"
        );
        assert!(homes > 0 || turned(Why::Remapping) > 0);
        unregister(0xc0de_24ff, start);
        drop((last, rig));
        place.unmap(0, bytes as u64);
    }

    // ---- Slice 3: the directory (`crate::memory`) and our own heaps ----

    /// The core, the pool and the directory on for this thread.
    fn directory() -> Switched {
        let switched = Switched::pool();
        crate::memory::set_test(true);
        switched
    }

    /// The offset in `place` of the first 4 MiB cell whose next cell is
    /// still `place`'s (reserved: a dispatch that names an address in the
    /// cell finds nothing to number after it).
    fn cell_in(place: &Place) -> u64 {
        const CELL: u64 = 4 << 20;
        let at = place.base.next_multiple_of(CELL) - place.base;
        assert!(at + 2 * CELL <= place.bytes as u64);
        at
    }

    /// The buffer placed over all of `chunk`, as its raw pointer.
    fn chunk_buffer(chunk: &TestChunk) -> *mut c_void {
        crate::pool::home(chunk.handle, 0, K64)
            .expect("an opened chunk")
            .buffer
            .as_raw()
    }

    /// With the directory on, view memory of two chunks (no buffer or window
    /// could be one home for it) is reached by raw-address shaders page by
    /// page: each page's entry is its chunk's number and its offset in the
    /// chunk, the chunk's buffer is that number's view, nothing is copied,
    /// and the card's table takes the entries only at the next submission's
    /// head.
    #[test]
    fn warp_directory_scattered_view_pages_resolve_to_their_chunks() {
        let Some(host) = warp13() else { return };
        let _on = directory();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(4 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::low(12 << 20);
        let at = cell_in(&place);
        place.map(at, 2 * K64, &a, 0);
        place.map(at + 2 * K64, 2 * K64, &b, 2 * K64);
        let start = place.base + at;
        fill(start, 4 * K64, 5);
        let scattered = turned(Why::Scattered);
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        let ready = crate::memory::ready(&list, &heap, &[start + 0x40], &[], false).expect("ready");
        let entry = |k: u64| crate::memory::entries_of(start + k * K64).unwrap();
        let (one, two) = (entry(0).1, entry(2).1);
        assert_ne!(one.0, 0, "a number");
        assert_ne!(two.0, 0, "a number");
        assert_ne!(one.0, two.0, "one number per chunk");
        assert_eq!([entry(0).1, entry(1).1], [(one.0, 0), (one.0, K64 as u32)]);
        assert_eq!(
            [entry(2).1, entry(3).1],
            [(two.0, 2 * K64 as u32), (two.0, 3 * K64 as u32)],
            "each page at its offset in its chunk, the whole cell at once"
        );
        assert_eq!(
            heap.kept_raw(ready.base + one.0),
            Some(chunk_buffer(&a)),
            "the number's view is the chunk's buffer"
        );
        assert_eq!(heap.kept_raw(ready.base + two.0), Some(chunk_buffer(&b)));
        assert!(
            crate::shadow::copies_over(start, start + 4 * K64).is_empty(),
            "no copy of the pages"
        );
        assert_eq!(turned(Why::Scattered), scattered, "and no ask for one");
        assert_eq!(entry(0).0, (0, 0), "the card's table not yet");
        // The card reads the title's bytes at the entries' offsets.
        let mut rig = Rig::new(&host.device).unwrap();
        let buffer = crate::pool::home(b.handle, 0, K64).unwrap().buffer;
        let got = rig.fetch(&buffer, 3 * K64, K64 as usize).unwrap();
        assert!(same(start + 3 * K64, &got), "the page in place");
        list.close();
        host.execute(&[&list]);
        assert_eq!(entry(0).0, one, "the card's table at the submission's head");
        assert_eq!(entry(3).0, (two.0, 3 * K64 as u32));
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(start, start + (8 << 20));
        drop((list, heap, rig, buffer));
        place.unmap(at, 2 * K64);
        place.unmap(at + 2 * K64, 2 * K64);
    }

    /// A remap of a chunk page changes its entry, and only in queue order:
    /// the card's table keeps the old entry until the next submission, whose
    /// lists all see the new one (a copy of ours at its head), and a list of
    /// the submission before saw the old one.
    #[test]
    fn warp_directory_remaps_reach_the_card_in_queue_order() {
        let Some(host) = warp13() else { return };
        let _on = directory();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(4 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::low(12 << 20);
        let at = cell_in(&place);
        place.map(at, K64, &a, K64);
        let start = place.base + at;
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        crate::memory::ready(&list, &heap, &[start], &[], false).expect("ready");
        let table = crate::memory::directory_buffer().unwrap();
        let (out, data) = host.shared_buffer(16).unwrap();
        let page = start / K64;
        // What a list of the submission sees of the entry, copied to `at`.
        let seen = |list: &crate::host::HostList, at: u64| {
            // SAFETY: an open list of the host's; live buffers, the ranges
            // inside both.
            unsafe { list.list.CopyBufferRegion(&out, at, &table, page * 8, 8) };
        };
        let read = |at: usize| {
            // SAFETY: the mapping holds 16 bytes; the card is done with it.
            unsafe { std::slice::from_raw_parts(data.add(at).cast::<u32>(), 2).to_vec() }
        };
        seen(&list, 0);
        list.close();
        host.execute(&[&list]);
        let old = crate::memory::entries_of(start).unwrap().0;
        assert_eq!(old.1, K64 as u32, "chunk a at its second page");
        assert_eq!(
            read(0),
            [old.0, old.1],
            "the list saw the entry made for it"
        );
        list.reset();
        // The title maps chunk b's last page there: the CPU's table changes
        // at the next submission's head, the card's not before. Chunk b has
        // no number yet, and none is given at a head (no list of the
        // submission has its view): the entry is empty for that submission.
        place.unmap(at, K64);
        place.map(at, K64, &b, 3 * K64);
        assert_eq!(crate::memory::entries_of(start).unwrap().0, old, "not yet");
        seen(&list, 8);
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            crate::memory::entries_of(start).unwrap(),
            ((0, 0), (0, 0)),
            "no number at a submission's head"
        );
        assert_eq!(read(8), [0, 0], "the submission's list saw it whole");
        // The next list's dispatch numbers chunk b, and writes its view.
        list.reset();
        let ready = crate::memory::ready(&list, &heap, &[], &[], false).expect("ready");
        let numbered = crate::memory::entries_of(start).unwrap().1;
        assert_eq!(numbered.1, 3 * K64 as u32, "chunk b's last page");
        assert_ne!(numbered.0, old.0, "chunk b's number");
        assert_eq!(
            heap.kept_raw(ready.base + numbered.0),
            Some(chunk_buffer(&b))
        );
        seen(&list, 0);
        list.close();
        host.execute(&[&list]);
        let new = crate::memory::entries_of(start).unwrap();
        assert_eq!(new.0, new.1, "both tables agree after the submission");
        assert_eq!(new.0, numbered);
        assert_eq!(read(0), [new.0.0, new.0.1], "and its list saw it");
        // Back to chunk a, which has a number: taken at the head itself.
        list.reset();
        place.unmap(at, K64);
        place.map(at, K64, &a, K64);
        list.close();
        host.execute(&[&list]);
        assert_eq!(crate::memory::entries_of(start).unwrap(), (old, old));
        // Unmapped: empty, at the next submission.
        list.reset();
        place.unmap(at, K64);
        list.close();
        host.execute(&[&list]);
        assert_eq!(crate::memory::entries_of(start).unwrap(), ((0, 0), (0, 0)));
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(start, start + (8 << 20));
        drop((list, heap, out));
    }

    /// Raw-address shaders that may store reach chunk pages with no copy to
    /// compare: after the wait, a copy of the same pages at another view is
    /// told, so it takes them again.
    #[test]
    fn warp_directory_raw_writes_into_chunk_pages_are_told_to_copies() {
        let Some(host) = warp13() else { return };
        let _on = directory();
        let chunk = TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed);
        let (p, q) = (Place::low(12 << 20), Place::low(2 * K64 as usize));
        let at = cell_in(&p);
        p.map(at, K64, &chunk, 0);
        q.map(0, K64, &chunk, 0);
        let start = p.base + at;
        let copy = crate::shadow::exact(q.base + 0x100, 0x100).expect("a copy at the other view");
        assert!(!copy.is_home());
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        crate::memory::ready(&list, &heap, &[start], &[], true).expect("ready");
        assert_ne!(crate::memory::entries_of(start).unwrap().1.0, 0);
        let told = TOLD[8].load(Ordering::Relaxed);
        list.close();
        host.execute(&[&list]);
        assert!(
            TOLD[8].load(Ordering::Relaxed) > told,
            "the copy at the other view told"
        );
        // A list that only reads tells nothing.
        list.reset();
        crate::memory::ready(&list, &heap, &[start], &[], false).expect("ready");
        let told = TOLD[8].load(Ordering::Relaxed);
        list.close();
        host.execute(&[&list]);
        assert_eq!(TOLD[8].load(Ordering::Relaxed), told);
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(start, start + (8 << 20));
        crate::shadow::let_go_copies_for_test(q.base, q.base + K64);
        drop((list, heap, copy));
        p.unmap(at, K64);
        q.unmap(0, K64);
    }

    /// A chunk page a host copy holds bytes of (memory of several places
    /// bound as one view) is reached by raw-address shaders through a copy
    /// too, so that both kinds of use see the same bytes: one made after the
    /// page got its chunk's entry takes the page at the next list's
    /// dispatch.
    #[test]
    fn warp_directory_a_copy_over_chunk_pages_takes_them() {
        let Some(host) = warp13() else { return };
        let _on = directory();
        let (a, b) = (
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
            TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed),
        );
        let place = Place::low(12 << 20);
        let at = cell_in(&place);
        place.map(at, K64, &a, 0);
        place.map(at + K64, K64, &b, 0);
        let start = place.base + at;
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        let ready = crate::memory::ready(&list, &heap, &[start], &[], false).expect("ready");
        let number = |k: u64| crate::memory::entries_of(start + k * K64).unwrap().1.0;
        let (one, two) = (number(0), number(1));
        assert_eq!(heap.kept_raw(ready.base + one), Some(chunk_buffer(&a)));
        assert_eq!(heap.kept_raw(ready.base + two), Some(chunk_buffer(&b)));
        // A buffer view over both pages: a copy (two chunks).
        let copy = crate::shadow::exact(start + K64 - 0x100, 0x200).expect("a copy");
        assert!(!copy.is_home());
        let ready = crate::memory::ready(&list, &heap, &[start], &[], false).expect("ready");
        for k in 0..2 {
            let n = number(k);
            assert_ne!(n, 0);
            assert_ne!(n, [one, two][k as usize], "no longer the chunk's");
            assert_ne!(heap.kept_raw(ready.base + n), Some(chunk_buffer(&a)));
            assert_ne!(heap.kept_raw(ready.base + n), Some(chunk_buffer(&b)));
        }
        list.close();
        host.execute(&[&list]);
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(start, start + (8 << 20));
        drop((list, heap, copy));
        place.unmap(at, K64);
        place.unmap(at + K64, K64);
    }

    /// A home's buffer starts before its first title byte (the 64 KiB page
    /// under it): its entries and its view count from the buffer's start, so
    /// the card reads the title's bytes at an entry's offset. With the
    /// directory off the entries go straight into the card's table.
    #[test]
    fn warp_directory_entries_of_a_home_count_from_its_buffer() {
        let Some(host) = warp13() else { return };
        let _on = Switched::on(false);
        let region = TestRegion::low(12 << 20);
        let cell = region.base.next_multiple_of(4 << 20);
        let (start, len) = (cell + K64 + 0x100, 3 * K64);
        register(0xc0de_3101, start, len);
        let home = find(start, len).expect("a home");
        assert!(home.is_home());
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        let page = cell + 2 * K64;
        let ready = crate::memory::ready(&list, &heap, &[page + 0x10], &[], false).expect("ready");
        let (shaders, next) = crate::memory::entries_of(page).unwrap();
        assert_eq!(shaders, next, "straight into the card's table");
        assert_ne!(next.0, 0);
        assert_eq!(
            u64::from(next.1),
            home.offset(page),
            "from the buffer's start"
        );
        assert_eq!(
            heap.kept_raw(ready.base + next.0),
            Some(home.buffer.as_raw())
        );
        fill(page, K64, 21);
        let mut rig = Rig::new(&host.device).unwrap();
        let got = rig
            .fetch(&home.buffer, u64::from(next.1), K64 as usize)
            .unwrap();
        assert!(same(page, &got), "the title's bytes at the entry's offset");
        list.close();
        host.execute(&[&list]);
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(region.base, region.base + (12 << 20));
        unregister(0xc0de_3101, start);
        drop((list, heap, home, rig));
        region.release();
        run_pending();
    }

    /// With `VICTORY_MEMORY_OWN=1` our memory for the title's heaps is a
    /// region of our own below 128 GiB: a buffer in it gets a home (class O is
    /// C), and giving it back while a home is held waits for the home, then
    /// releases the region.
    #[test]
    fn warp_own_heap_memory_is_a_region_with_homes() {
        let Some(_host) = warp13() else { return };
        let _on = Switched::on(false);
        TEST_OWN.with(|own| own.set(true));
        let memory = crate::resource::Memory::zeroed(3 * K64 + 0x123, 64).expect("memory");
        let base = memory.address();
        assert!(
            (4 << 30..1 << 37).contains(&base),
            "below 128 GiB, above 4 GiB"
        );
        assert_eq!(base % K64, 0, "64 KiB aligned");
        assert_eq!(state(base), mem::MEM_COMMIT);
        {
            let mut core = core();
            let (found, region) = core.region_at(base as usize).expect("a region");
            assert_eq!(found, base as usize);
            assert!(region.own, "ours");
            assert_eq!(region.len as u64, 4 * K64, "whole 64 KiB pages");
        }
        // SAFETY: committed memory of ours.
        let bytes = unsafe { std::slice::from_raw_parts(base as *const u8, 4 * K64 as usize) };
        assert!(bytes.iter().all(|&x| x == 0), "zeroed");
        let (start, len) = (base + 0x100, 2 * K64);
        register(0xc0de_3201, start, len);
        let held = find(start, len).expect("a home");
        assert!(held.is_home(), "in place, not a copy");
        drop(memory);
        assert_eq!(state(base), mem::MEM_COMMIT, "given back at the sweep");
        run_own_frees();
        assert_eq!(state(base), mem::MEM_COMMIT, "waits while the home is held");
        unregister(0xc0de_3201, start);
        drop(held);
        run_pending();
        assert_eq!(state(base), mem::MEM_FREE, "released once nothing holds it");
        assert!(
            core().region_at(base as usize).is_none(),
            "out of the table"
        );
    }

    /// Without `VICTORY_MEMORY_OWN=1` our heap memory stays the process
    /// heap's: no region, so no home (`Why::NotRegion`).
    #[test]
    fn own_heaps_off_keep_heap_memory() {
        let _on = Switched::on(false);
        let memory = crate::resource::Memory::zeroed(4 * K64, 64).expect("memory");
        assert!(core().region_at(memory.address() as usize).is_none());
        drop(memory);
        assert!(
            OWN_FREES.lock().unwrap().is_empty(),
            "nothing given back through the core"
        );
    }

    /// `XMemFreeDefault`'s path: a region of ours given back at once, not at
    /// the next sweep, through the core (out of the table, counted apart).
    #[test]
    fn own_region_given_back_at_once() {
        let _on = Switched::on(false);
        let at = memory_own_alloc(K64 as usize + 1, 16).expect("a region of ours");
        let base = at.as_ptr() as usize;
        assert!(core().region_at(base).is_some_and(|(_, r)| r.own));
        let given = OWN[3].load(Ordering::Relaxed);
        memory_own_free_now(base);
        assert_eq!(state(base as u64), mem::MEM_FREE, "released now");
        assert!(core().region_at(base).is_none(), "out of the table");
        assert!(OWN[3].load(Ordering::Relaxed) > given, "counted as ours");
    }

    /// A region freed and one of ours made at the same base before the
    /// release reached the table: the release takes only its own region.
    #[test]
    fn a_release_leaves_a_new_region_at_its_base() {
        let _on = Switched::on(false);
        let at = memory_own_alloc(K64 as usize, 16).expect("a region of ours");
        let base = at.as_ptr() as usize;
        let old = core().region_at(base).map(|(_, r)| r.serial).unwrap();
        core().regions.insert(
            base,
            Region::new(
                K64 as usize,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                0,
                mem::PAGE_READWRITE,
                true,
            ),
        );
        released(base, old);
        assert!(
            core().region_at(base).is_some_and(|(_, r)| r.serial != old),
            "the new region stays"
        );
        memory_own_free_now(base);
        assert!(core().region_at(base).is_none());
    }

    /// With the directory on, a chunk that goes loses its number at the next
    /// dispatch: its entries are emptied, the number names nothing, and the
    /// heap's view of it is made null, so the directory no longer holds the
    /// chunk's buffer (and with it its heap and section).
    #[test]
    fn warp_directory_a_chunk_that_goes_loses_its_number() {
        let Some(host) = warp13() else { return };
        let _on = directory();
        let chunk = TestChunk::new(2 * K64 as usize, crate::pool::PoolChunk::Committed);
        let place = Place::low(12 << 20);
        let at = cell_in(&place);
        place.map(at, 2 * K64, &chunk, 0);
        let start = place.base + at;
        let heap = crate::descriptor::host::HostHeap::new(false, 8).expect("a view heap");
        let list = host.new_list().unwrap();
        let ready = crate::memory::ready(&list, &heap, &[start], &[], false).expect("ready");
        let number = crate::memory::entries_of(start).unwrap().1.0;
        assert_ne!(number, 0);
        assert!(crate::memory::names(number));
        assert_eq!(
            heap.kept_raw(ready.base + number),
            Some(chunk_buffer(&chunk))
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
        crate::pool::title_memory_pool_chunk_gone(chunk.handle);
        let ready = crate::memory::ready(&list, &heap, &[], &[], false).expect("ready");
        assert!(!crate::memory::names(number), "the number names nothing");
        assert_eq!(heap.kept_raw(ready.base + number), None, "a null view");
        assert_eq!(
            crate::memory::entries_of(start + K64).unwrap().1,
            (0, 0),
            "every page of it emptied"
        );
        list.close();
        host.execute(&[&list]);
        assert_eq!(
            crate::memory::entries_of(start).unwrap(),
            ((0, 0), (0, 0)),
            "in the card's table at the head"
        );
        list.reset();
        crate::memory::forget_heap(&heap);
        crate::memory::forget_for_test(start, start + (8 << 20));
        drop((list, heap));
        place.unmap(at, 2 * K64);
    }
}
