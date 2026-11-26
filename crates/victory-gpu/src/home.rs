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
//! home ([`Why::Late`]).

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
    TEST_ON.with(std::cell::Cell::get)
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
    /// The device can open parts of an allocation (`ID3D12Device13`).
    pub(crate) device13: bool,
    /// The host refused three openings in a row with the same error.
    pub(crate) latched: bool,
}

impl Gate {
    fn process() -> Gate {
        Gate {
            on: on(),
            late: crate::host::late(),
            device13: device13(),
            latched: LATCHED.load(Ordering::Relaxed),
        }
    }
}

/// Whether the host device is an `ID3D12Device13`: asked once, when the
/// first range is classified (`false` before there is a device).
fn device13() -> bool {
    use std::sync::atomic::AtomicU8;
    static KNOWN: AtomicU8 = AtomicU8::new(0);
    #[cfg(test)]
    if NO_DEVICE13.with(std::cell::Cell::get) {
        return false;
    }
    match KNOWN.load(Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => {
            let Some(host) = crate::host::host() else {
                return false;
            };
            let yes = host.device.cast::<ID3D12Device13>().is_ok();
            KNOWN.store(if yes { 1 } else { 2 }, Ordering::Relaxed);
            if !yes && on() {
                tracing::warn!(
                    "memory core: the device cannot open parts of an allocation (no ID3D12Device13); nothing is placed in place, every range keeps its copy"
                );
            }
            yes
        }
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
    /// View memory (the page pool): slice 2.
    View,
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

const WHYS: [(Why, &str); 11] = [
    (Why::Late, "late waits"),
    (Why::NoDevice13, "no ID3D12Device13"),
    (Why::Latched, "after the host's refusals"),
    (Why::Overlap, "partly in a home"),
    (Why::View, "view memory"),
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
    if let Some(i) = WHYS.iter().position(|w| w.0 == why) {
        TURNED_DOWN[i].fetch_add(1, Ordering::Relaxed);
    }
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
/// `crate::memory`'s directory at once.
static TOLD: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];

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

/// What [`classify`] asks, in its order: each only once it is needed.
pub(crate) trait Ask {
    type Home;
    /// A home over the range ([`Held`]).
    fn holder(&mut self) -> Held<Self::Home>;
    /// For a big buffer: whether the card took a copy of it.
    fn card(&mut self) -> bool;
    /// Whether the range is view memory.
    fn view(&mut self) -> bool;
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
pub(crate) enum Class<H> {
    /// Today's copy (X).
    Copy(Why),
    /// The card took a copy (G): the copy path, with that card side.
    Card,
    /// The home that holds the range (C, joined).
    Home(H),
    /// Claimed: open the pages and place a buffer (C, new).
    Place,
}

/// The rules of the memory core, in order. `big`: a title buffer large
/// enough to be offered a card copy.
///
/// 1. The process: the switch on, every submission waited for, a device
///    that opens parts of an allocation, the host not latched off.
/// 2. A big buffer is offered a copy in the card's memory first (G): the
///    card reads it there far faster than from the PC's memory, also when a
///    home (a window made first over its bytes) holds it. A card copy is a
///    copy, which may lie over a home ([`after_wait`] keeps it current).
/// 3. One home per byte: a range that one home holds whole is that home; a
///    range that homes hold only part of keeps a copy (a second home over the
///    same bytes, or one over two homes, could not be one buffer).
/// 4. View memory (the page pool's views) keeps its copy in this slice.
/// 5. The range's pages lie in one `XMemVirtualAlloc` region that the title
///    is not freeing or protecting ([`Why::Busy`]) and whose protection was
///    never refused ([`Why::Blacklisted`]).
/// 6. Those pages are private memory of that allocation, committed
///    read-write.
/// 7. The pages fit in `VICTORY_MEMORY_CORE_MB`: they are claimed, and the
///    caller opens them.
pub(crate) fn classify<A: Ask>(gate: Gate, big: bool, ask: &mut A) -> Class<A::Home> {
    if !gate.on {
        return Class::Copy(Why::Off);
    }
    if gate.late {
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
    if ask.view() {
        return Class::Copy(Why::View);
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
}

static CORE: Mutex<Core> = Mutex::new(Core {
    regions: BTreeMap::new(),
    live: BTreeMap::new(),
    tabled: BTreeMap::new(),
    tabled_longest: 0,
    pending: Vec::new(),
    unnumber: Vec::new(),
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
    core.regions.insert(
        at,
        Region {
            len,
            kind,
            flags,
            protect,
            homes_ever: 0,
            busy: 0,
            blacklisted: false,
        },
    );
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
}

impl Ask for LiveAsk<'_> {
    type Home = Arc<Shadow>;

    fn holder(&mut self) -> Held<Arc<Shadow>> {
        holder(self.start, self.end)
    }

    fn card(&mut self) -> bool {
        (self.card)()
    }

    fn view(&mut self) -> bool {
        crate::inplace::cover(self.start as usize, (self.end - self.start) as usize)
            != crate::inplace::Cover::None
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

/// What a range of title memory resolves to.
pub(crate) enum Found {
    /// The home that holds it.
    Home(Arc<Shadow>),
    /// A new home: the caller makes the `Shadow` and tables it ([`tabled`]).
    New(crate::inplace::Placed),
    /// The card took a copy: the copy path, with the card side it made.
    Card,
    /// Today's copy.
    Copy,
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
        Class::Place => match open_home(start, at, bytes) {
            Ok(placed) => {
                opened();
                MADE[usize::from(!window)].fetch_add(1, Ordering::Relaxed);
                Found::New(placed)
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
/// number): its pages are given back and the tables forget it.
pub(crate) fn gone(start: u64, len: u64, id: u64) {
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
    unclaim(at, bytes);
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
    let homes = untable_over(at as u64, end);
    if homes.is_empty() {
        return 0;
    }
    crate::shadow::let_go_homes(&homes);
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
        crate::host::retire(home);
    }
    gone
}

/// The homes over the memory frees and remaps let homes go from since the
/// last sweep go out of `crate::memory`'s directory (at the sweep: no
/// submission runs).
fn unnumber() {
    let ranges = std::mem::take(&mut core().unnumber);
    for (from, to) in ranges {
        crate::memory::forget_homes_over(from, to);
    }
}

/// The pages of a title `VirtualFree(at, len, kind)` in a region, and the
/// region's base; `None` if they are not a region's, the call is of a kind
/// that never touches a home (placeholders), or the host would refuse it
/// for its own shape whatever stood there: a release that is not of the
/// whole allocation from its base, a decommit of length 0 not at the base,
/// a range past the region's end or over more than its one allocation.
/// Such a call goes to the host as it is, and its refusal is the title's.
fn free_span(at: usize, len: usize, kind: u32) -> Option<(usize, usize, usize)> {
    let (base, region_end) = {
        let mut core = core();
        let (base, region) = core.region_at(at)?;
        (base, base + region.len)
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
        .then_some((from, to, base))
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
    TOLD[which].fetch_add(1, Ordering::Relaxed);
    region.busy += 1;
    if region.homes_ever == 0 {
        return (true, false);
    }
    TOLD[which + 1].fetch_add(1, Ordering::Relaxed);
    (true, true)
}

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
    let Some((from, to, base)) = free_span(at, len, kind) else {
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
                released(base);
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
            released(base);
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
            since: crate::host::now_ms(),
        });
        PENDING.fetch_add(1, Ordering::Release);
    }
    // Done, as the title is told: its last error as it was before.
    // SAFETY: a plain write of this thread's last error.
    unsafe { SetLastError(before) };
    1
}

/// The region at `base` was released: it leaves the table, and the frees of
/// it that wait go with it (the host freed their pages with the rest).
/// Called with [`FREEING`] held.
fn released(base: usize) {
    let mut core = core();
    core.regions.remove(&base);
    drop_pending(&mut core, |p| p.region == base);
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
/// [`title_virtual_free`]; a protection cannot wait, so when the host still
/// refuses after the homes were let go, what the retire queue holds for
/// finished submissions is dropped and it is asked once more; refused again,
/// the title gets FALSE with error 87 (what it got with `VICTORY_INPLACE`),
/// it is counted, and the region is never given a home again.
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

/// [`run_pending`], giving up after `give_up_ms` (tests).
fn run_pending_after(give_up_ms: u64) {
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
    unnumber();
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
            released(p.region);
        }
    }
    core().pending.extend(kept);
}

/// After a submission's wait: the shaders of its lists may have written
/// these homes' ranges (`declared`: each copy with its ranges, as
/// `Host::submit` gathers them). Write watching does not see GPU writes, so
/// a copy over the same bytes elsewhere (a buffer view's exact copy, a
/// buffer partly in a home) is told: its pages there count as written and
/// its next refresh takes them again (`crate::watch::gpu_wrote`). Normally
/// no copy lies over a home and nothing is done.
///
/// shortcut: host copies of textures over a home are not looked for.
pub(crate) fn after_wait(declared: &[(Arc<Shadow>, Vec<(usize, usize)>)]) {
    if !on() {
        return;
    }
    // Each byte looked for once: `crate::memory` declares every home it
    // numbered for every dispatch of its shaders.
    let mut ranges: Vec<(usize, usize)> = declared
        .iter()
        .filter(|(home, _)| home.is_home())
        .flat_map(|(_, ranges)| ranges.iter().copied())
        .filter(|&(from, to)| from < to)
        .collect();
    if ranges.is_empty() {
        return;
    }
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match merged.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => merged.push((from, to)),
        }
    }
    for (from, to) in merged {
        for (a, b) in crate::shadow::copies_over(from as u64, to as u64) {
            crate::watch::gpu_wrote(a, b - a);
            TOLD[8].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Once per sweep (`crate::inplace::tick`): the frees that waited, and the
/// line every ten seconds.
pub(crate) fn tick() {
    if !on() {
        return;
    }
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
    let core = core();
    let (homes, pinned) = core.live.iter().fold((0u64, 0u64), |a, (&(_, bytes), &n)| {
        (a.0 + u64::from(n), a.1 + bytes * u64::from(n))
    });
    let mut lines = vec![format!(
        "memory core: {homes} homes pin {} MB (made {} windows, {} buffers; {} asks joined a home; {} big buffers given a card copy first; {} opens, {} us); \
         kept a copy: {}; title frees of GPU regions {} ({} in regions that had a home, {} done later, {} ms waited, {} failed later, {} given up, {} waiting), \
         protections {} ({} in regions that had a home, {} refused, {} homes taken out of the shaders' directory at once); \
         GPU writes into homes told to copies over them {}; view heaps whose slots still hold a let-go home {views}",
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
    // The census: regions by the console's protection above the host's
    // bits, its page-size kind bits and the XMem flags, with the bytes homes
    // pin in them.
    let mut census: BTreeMap<(u32, u32, u64), (u64, u64, u64)> = BTreeMap::new();
    for (&base, region) in &core.regions {
        let key = (
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
    for ((protect, kind, flags), (n, bytes, in_homes)) in census {
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
        }
    }

    /// Fixed answers, and the asks in the order they came.
    struct Fixed {
        holder: Option<bool>,
        card: bool,
        view: bool,
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
                view: false,
                region: Some(Ok(())),
                committed: true,
                claim: Ok(()),
                asked: Vec::new(),
            }
        }
    }

    impl Ask for Fixed {
        type Home = u32;
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
        fn view(&mut self) -> bool {
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
        device13: true,
        latched: false,
    };

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
        // 4-7, each stopping there.
        for (f, why, last) in [
            (
                Fixed {
                    view: true,
                    ..Fixed::good()
                },
                Why::View,
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
}
