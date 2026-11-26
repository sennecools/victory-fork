//! The memory pool (`VICTORY_MEMORY_POOL=1`, with `VICTORY_MEMORY_CORE=1`;
//! off by default): the title's page pool as the card's own memory, the
//! second slice of the memory core (`crate::home`).
//!
//! With the switch on, `xmem` backs each allocation of physical pages by
//! chunks of 256 MiB, each a section made on the first map or write into
//! it, committed whole (`SEC_COMMIT`) when the PC has room for it, and
//! tells this module ([`title_memory_pool_chunk`]). A committed section can
//! be opened as a heap (`OpenExistingHeapFromFileMapping`,
//! `crate::inplace::open`): each chunk is opened once, when a home is first
//! wanted in it, with one buffer placed over all of it, if the shared part
//! of the card's budget has room for the whole chunk ([`fits_shared`]).
//! A title buffer or a
//! window whose bytes are one run of one chunk's pages at one title view
//! (`crate::inplace::Cover::Run`) then has its home in that buffer, at the
//! run's offset in the section (class C-pool, [`home`]): the card reads and
//! writes the pages themselves, which every view of them shows, and nothing
//! is copied either way.
//!
//! What the opening is, measured: on the software adapter a section heap
//! pins nothing and opens in 0.2 ms for 256 MiB, and it outlives the views
//! and the handle (the heap is the section, `crate::inplace`). On the card
//! (RX 9070 XT, on-card probe) it opens, shaders read it at 53-55 GB/s, the
//! CPU's writes are seen with no copy, and it charges the shared budget its
//! own size only.
//!
//! These heaps are never tile-mapped: tile mappings into such a heap remove
//! the device (`crate::inplace`). They are only ever given to
//! `CreatePlacedResource`; the title's tiled heap stays CPU views of the
//! pages (`xmem`'s `map_alias`).
//!
//! A chunk that is not committed (no room in the PC's commit charge, a
//! Windows before build 20348, a process where the core can make no home,
//! or the host refused) keeps today's path: its views are copied, guarded
//! by the watch.
//!
//! Buffer views the title writes into view heaps are made on the C-pool
//! home as every other use of the buffer is (one place for the card's
//! reads and writes of those bytes). Such a slot is made once, on the
//! chunk's buffer at the offset the home had then; a remap or free that
//! lets the home go would leave it showing the old pages (through the heap,
//! which keeps them) until the title writes the descriptor again. So the
//! slots made on a chunk's buffer are known by their title range
//! (`crate::descriptor::host::pool_let_go`), and those over a home let go
//! are made anew from their record at their next use: the draw that reads
//! them through a table, or the heap buffer pass for slots shaders pick by
//! index. shortcut: slots shaders pick by index that a list reads after the
//! frame's heap buffer pass ran keep the old pages until the next frame's,
//! as a copy there keeps the bytes of that pass's refresh; dynamic slots
//! (made for one list) are not looked at. shortcut: a remap lets a C-pool
//! home go at once, but its entries in `crate::memory`'s directory go only
//! at the next sweep (`crate::home::cpu_touched`): a raw-address dispatch
//! recorded in between reads the old pages through the directory.
//!
//! The GPU's writes into a C-pool home are not seen by the guards: after a
//! submission's wait, the copies over the same pages at every view are told
//! (`crate::home::after_wait`), and so are the placed textures there
//! ([`textures_told`]): the journal marks those pages written, which a
//! draw's look at its textures sees, and the texture pass makes the slots
//! of textures picked by index due (`crate::descriptor::host::pool_written`).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use windows::Win32::Graphics::Direct3D12::{ID3D12Heap, ID3D12Resource};

use crate::home::Why;

/// `VICTORY_MEMORY_POOL=1` with the memory core on. Read once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_MEMORY_POOL").is_some_and(|v| v == "1");
        let on = asked && crate::home::on();
        if on {
            tracing::info!(
                "memory pool on (VICTORY_MEMORY_POOL=1): the page pool is made in committed chunks of 256 MiB, and title buffers and windows in one run of one are used by the card in place"
            );
        } else if asked {
            tracing::warn!("VICTORY_MEMORY_POOL=1 is ignored without VICTORY_MEMORY_CORE=1");
        }
        on
    });
    on || test_on()
}

/// [`on`], for `xmem`: whether it makes the page pool in chunks.
pub fn memory_pool_on() -> bool {
    on()
}

// The tests' switch, per thread, as `crate::home`'s.
#[cfg(test)]
thread_local! {
    static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_on() -> bool {
    TEST_ON.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn test_on() -> bool {
    false
}

/// Sets the pool on for this thread's lookups (tests).
#[cfg(test)]
pub(crate) fn set_test(on: bool) {
    TEST_ON.with(|t| t.set(on));
}

/// How `xmem` made a chunk of the pool ([`title_memory_pool_chunk`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolChunk {
    /// Committed whole: it can be opened as a heap.
    Committed,
    /// Reserved: the PC's commit charge had no room for it.
    ShortOfCommit,
    /// Reserved: a Windows before build 20348 cannot open a section as a
    /// heap.
    OldWindows,
    /// Reserved: the host refused to make it committed.
    Refused,
    /// Reserved: the memory core can make no home in this process
    /// ([`memory_pool_homes_possible`]), so the commit would buy nothing.
    NoHomes,
}

/// A chunk's heap and the buffer placed over all of it.
struct Opened {
    heap: ID3D12Heap,
    buffer: ID3D12Resource,
    /// The buffer's GPU address.
    gpu: u64,
    /// Once its chunk took it: whether the buffer has a place in
    /// [`CHUNK_BUFFERS`] (else it is counted in [`CHUNK_BUFFERS_OVER`]).
    listed: Option<bool>,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Opened {}

impl Drop for Opened {
    fn drop(&mut self) {
        use windows::core::Interface;
        // Out of the set before the buffer is let go: a buffer made later at
        // the same address is no chunk's.
        match self.listed {
            Some(true) => unlist(self.buffer.as_raw() as usize),
            Some(false) => {
                CHUNK_BUFFERS_OVER.fetch_sub(1, Ordering::SeqCst);
            }
            None => {}
        }
    }
}

/// The opened chunks' placed buffers (their COM pointers; 0: an empty
/// place), so that [`is_chunk_buffer`], asked for every view heap slot made
/// with the pool on, takes no lock. 1,024 chunks of 256 MiB are 256 GiB.
static CHUNK_BUFFERS: [AtomicUsize; 1024] = [const { AtomicUsize::new(0) }; 1024];

/// Opened chunks' buffers that found no place in [`CHUNK_BUFFERS`]: while
/// there are any, [`is_chunk_buffer`] looks in the chunks' table too.
static CHUNK_BUFFERS_OVER: AtomicUsize = AtomicUsize::new(0);

/// One past the last place of [`CHUNK_BUFFERS`] ever taken: the places
/// looked at.
static CHUNK_BUFFERS_TOP: AtomicUsize = AtomicUsize::new(0);

/// The places of [`CHUNK_BUFFERS`] that may hold a buffer.
fn listed() -> &'static [AtomicUsize] {
    &CHUNK_BUFFERS[..CHUNK_BUFFERS_TOP.load(Ordering::SeqCst)]
}

/// Puts `raw` in [`CHUNK_BUFFERS`]; whether there was a place. Done before
/// the buffer is handed to anyone (under the chunks' lock), so whoever has
/// it finds it.
fn list(raw: usize) -> bool {
    let Some(i) = CHUNK_BUFFERS.iter().position(|place| {
        place
            .compare_exchange(0, raw, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok()
    }) else {
        return false;
    };
    CHUNK_BUFFERS_TOP.fetch_max(i + 1, Ordering::SeqCst);
    true
}

/// Whether `raw` is in [`CHUNK_BUFFERS`].
fn is_listed(raw: usize) -> bool {
    listed()
        .iter()
        .any(|place| place.load(Ordering::SeqCst) == raw)
}

/// Takes `raw` out of [`CHUNK_BUFFERS`].
fn unlist(raw: usize) {
    if let Some(place) = listed()
        .iter()
        .find(|place| place.load(Ordering::SeqCst) == raw)
    {
        place.store(0, Ordering::SeqCst);
    }
}

/// A chunk of the pool, as `xmem` made it.
struct Chunk {
    bytes: u64,
    committed: bool,
    /// Opened when a home was first wanted in it.
    opened: Option<Opened>,
    /// The host refused to open it: never asked again.
    refused: bool,
    /// Which chunk made at this handle it is ([`MADE_CHUNKS`]): an opening
    /// made with the lock let go goes in only if the chunk is still the one
    /// it was made for ([`home`]).
    serial: u64,
}

/// Chunks made so far.
static MADE_CHUNKS: AtomicU64 = AtomicU64::new(0);

/// The chunks by their section's handle. Taken after every other lock of
/// the runtime's that is held when a home is resolved (the shadow tables);
/// nothing that takes another lock of ours runs with it held.
static CHUNKS: Mutex<BTreeMap<usize, Chunk>> = Mutex::new(BTreeMap::new());

fn chunks() -> std::sync::MutexGuard<'static, BTreeMap<usize, Chunk>> {
    CHUNKS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The pool's counters ([`COUNTS`]), since the start unless it says now.
mod at {
    /// Chunks made committed.
    pub(super) const COMMITTED: usize = 0;
    /// Chunks made reserved: short of commit, on an old Windows, because the
    /// host refused, because no home can be made (`super::PoolChunk`).
    pub(super) const SHORT: usize = 1;
    pub(super) const OLD: usize = 2;
    pub(super) const REFUSED: usize = 3;
    pub(super) const NO_HOMES: usize = 4;
    /// Chunks opened as heaps, refused by the host, the microseconds the
    /// openings took, and the slowest one's.
    pub(super) const OPENED: usize = 5;
    pub(super) const OPEN_REFUSED: usize = 6;
    pub(super) const OPEN_US: usize = 7;
    pub(super) const OPEN_MOST_US: usize = 8;
    /// Asks for a home in an unopened chunk turned down because the shared
    /// budget has no room for the chunk ([`super::fits_shared`]).
    pub(super) const OVER_BUDGET: usize = 9;
    /// C-pool homes made; now: C-pool homes and their bytes; homes a remap
    /// or free let go.
    pub(super) const MADE: usize = 10;
    pub(super) const LIVE: usize = 11;
    pub(super) const LIVE_BYTES: usize = 12;
    pub(super) const LET_GO: usize = 13;
    /// Ranges of GPU writes into C-pool homes told to the placed textures
    /// ([`super::textures_told`]), those that lie where placed textures do.
    pub(super) const TEXTURE_RANGES: usize = 14;
    /// View heap slots made on a C-pool home that a let-go of the home had
    /// made anew at their next use ([`super::slots_made_anew`]).
    pub(super) const SLOTS_ANEW: usize = 15;
    pub(super) const ALL: usize = 16;
}

static COUNTS: [AtomicU64; at::ALL] = [const { AtomicU64::new(0) }; at::ALL];

/// What a chunk's opening must leave of the shared part of the card's
/// budget (the PC's memory the card uses, where the system also moves
/// what does not fit on the card).
const SHARED_MARGIN: u64 = 1 << 30;

/// After an ask turned down for the budget: the milliseconds before the
/// budget is asked again (the asks in between are turned down too).
const BUDGET_AGAIN_MS: u64 = 1000;

/// When an ask was last turned down for the budget (`crate::host::now_ms`,
/// plus one; 0: never).
static OVER_BUDGET_AT: AtomicU64 = AtomicU64::new(0);

/// Whether a chunk of `bytes` may be opened as a heap, by the shared part
/// of the card's budget (`budget`) and its use now (`used`), in bytes: the
/// heap charges that budget its whole size (on-card probe). A budget of 0
/// is none reported (a software adapter): nothing to keep to.
pub(crate) fn fits_shared(budget: u64, used: u64, bytes: u64) -> bool {
    budget == 0 || used.saturating_add(bytes).saturating_add(SHARED_MARGIN) <= budget
}

/// Whether the shared budget has room for a chunk of `bytes` now; asked of
/// the host at most once a [`BUDGET_AGAIN_MS`] after a no.
fn shared_room(host: &crate::host::Host, bytes: u64) -> bool {
    let now = crate::host::now_ms();
    let last = OVER_BUDGET_AT.load(Ordering::Relaxed);
    if last != 0 && now.saturating_add(1) < last.saturating_add(BUDGET_AGAIN_MS) {
        return false;
    }
    // `None`: the adapter cannot be asked; the probe's card charges the
    // chunk's size only, and nothing else is known.
    let Some([_, _, budget, used]) = host.video_memory() else {
        return true;
    };
    if fits_shared(budget, used, bytes) {
        return true;
    }
    OVER_BUDGET_AT.store(now.saturating_add(1), Ordering::Relaxed);
    if COUNTS[at::OVER_BUDGET].load(Ordering::Relaxed) == 0 {
        tracing::warn!(
            budget_mb = budget >> 20,
            used_mb = used >> 20,
            chunk_mb = bytes >> 20,
            "memory pool: the shared budget has no room to open a chunk of the page pool; its views keep their copies until it has"
        );
    }
    false
}

/// `xmem` made the section `section` of `bytes` for a chunk of the pool,
/// as `how` says. A chunk at the same handle before is gone (the handle
/// was closed, [`title_memory_pool_chunk_gone`]).
pub fn title_memory_pool_chunk(section: usize, bytes: u64, how: PoolChunk) {
    let i = match how {
        PoolChunk::Committed => at::COMMITTED,
        PoolChunk::ShortOfCommit => at::SHORT,
        PoolChunk::OldWindows => at::OLD,
        PoolChunk::Refused => at::REFUSED,
        PoolChunk::NoHomes => at::NO_HOMES,
    };
    if COUNTS[i].fetch_add(1, Ordering::Relaxed) == 0 && i != at::COMMITTED {
        tracing::warn!(
            ?how,
            bytes,
            "memory pool: a chunk of the page pool is reserved; its views keep their copies"
        );
    }
    let chunk = Chunk {
        bytes,
        committed: how == PoolChunk::Committed,
        opened: None,
        refused: false,
        serial: MADE_CHUNKS.fetch_add(1, Ordering::Relaxed),
    };
    let old = chunks().insert(section, chunk);
    drop(old);
}

/// The section `section` of a chunk is about to be closed: its heap is let
/// go (homes in it keep their own hold, until the tables and lists let them
/// go too). Without this the heap would keep the section, and its commit
/// charge, after the title freed its pages.
pub fn title_memory_pool_chunk_gone(section: usize) {
    let old = chunks().remove(&section);
    drop(old);
}

/// Whether the GPU side knows `section` as a chunk of the pool (made, and
/// not gone yet), and if so whether it is opened as a heap. Only for
/// `xmem`'s tests (another crate's, so not behind `cfg(test)`); the
/// runtime never asks it.
#[doc(hidden)]
pub fn title_memory_pool_chunk_known(section: usize) -> Option<bool> {
    chunks().get(&section).map(|chunk| chunk.opened.is_some())
}

/// Whether the memory core can make a C-pool home in this process: the
/// pool on, and none of what stops every home (late waits or the relay, no
/// `ID3D12Device13`, the host's refusals latched) known to hold. Asked by
/// `xmem` when it makes a chunk, so that it commits none for nothing;
/// creates no device (what is not known yet does not stop it).
pub fn memory_pool_homes_possible() -> bool {
    on() && !crate::home::homes_ruled_out()
}

/// The home of `len` title bytes that are the bytes from `offset` on of the
/// chunk with section `section`: a buffer over the whole chunk, with the
/// title bytes at `offset` in it. The chunk is opened now if no home was
/// wanted in it before, with the chunks' lock let go meanwhile: the opening
/// takes the host's own time. (Two threads that open one chunk at once both
/// open it; the first opening to go in is the chunk's, the other is let go.)
pub(crate) fn home(section: usize, offset: u64, len: u64) -> Result<crate::inplace::Placed, Why> {
    let placed = |opened: &Opened| crate::inplace::Placed {
        buffer: opened.buffer.clone(),
        heap: opened.heap.clone(),
        gpu: opened.gpu + offset,
        bias: offset,
    };
    let (bytes, serial) = {
        let chunks = chunks();
        let Some(chunk) = chunks.get(&section) else {
            return Err(Why::NotPool);
        };
        if !chunk.committed {
            return Err(Why::ReservedChunk);
        }
        if chunk.refused {
            return Err(Why::ChunkRefused);
        }
        // A run lies inside its section (`crate::inplace::cover`): one past
        // it is the census and the chunk table telling different sizes.
        if offset
            .checked_add(len.max(1))
            .is_none_or(|end| end > chunk.bytes)
        {
            return Err(Why::PastChunk);
        }
        if let Some(opened) = &chunk.opened {
            return Ok(placed(opened));
        }
        (chunk.bytes, chunk.serial)
    };
    let Some(host) = crate::host::host() else {
        return Err(Why::ChunkRefused);
    };
    // Not latched: asked again once the budget has room.
    if !shared_room(host, bytes) {
        COUNTS[at::OVER_BUDGET].fetch_add(1, Ordering::Relaxed);
        return Err(Why::OverBudget);
    }
    // The opening goes through a handle of its own, taken while the chunk
    // is known: `xmem` may close the chunk's handle meanwhile (and a section
    // made then may get the same number).
    let Some(own) = ({
        let chunks = chunks();
        match chunks.get(&section) {
            Some(chunk) if chunk.serial == serial => duplicate(section),
            _ => return Err(Why::NotPool),
        }
    }) else {
        return Err(Why::ChunkRefused);
    };
    let begun = std::time::Instant::now();
    let opened =
        crate::inplace::open(&host.device, crate::inplace::From::Section(own)).and_then(|heap| {
            crate::inplace::place(&host.device, &heap, 0, bytes).map(|buffer| (heap, buffer))
        });
    // SAFETY: the handle is ours (`duplicate`); the heap holds the section
    // itself.
    unsafe { windows_sys::Win32::Foundation::CloseHandle(own as _) };
    let opened = opened.map(|(heap, buffer)| {
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Opened {
            heap,
            buffer,
            gpu,
            listed: None,
        }
    });
    let took = begun.elapsed().as_micros() as u64;
    COUNTS[at::OPEN_US].fetch_add(took, Ordering::Relaxed);
    COUNTS[at::OPEN_MOST_US].fetch_max(took, Ordering::Relaxed);
    // An opening that does not go in is let go once the lock is.
    let mut spare = None;
    let found = {
        let mut chunks = chunks();
        match chunks.get_mut(&section) {
            // Closed meanwhile (and perhaps made anew at the same handle).
            None => Err(Why::NotPool),
            Some(chunk) if chunk.serial != serial => Err(Why::NotPool),
            Some(chunk) => match (opened, &chunk.opened) {
                (opened, Some(theirs)) => {
                    spare = opened.ok();
                    Ok(placed(theirs))
                }
                (Ok(mut opened), None) => {
                    use windows::core::Interface;
                    COUNTS[at::OPENED].fetch_add(1, Ordering::Relaxed);
                    crate::host::name(&opened.buffer, || {
                        format!("page pool chunk {section:#x} in place")
                    });
                    let listed = list(opened.buffer.as_raw() as usize);
                    if !listed {
                        CHUNK_BUFFERS_OVER.fetch_add(1, Ordering::SeqCst);
                    }
                    opened.listed = Some(listed);
                    let found = placed(&opened);
                    chunk.opened = Some(opened);
                    Ok(found)
                }
                (Err(error), None) => {
                    chunk.refused = true;
                    if COUNTS[at::OPEN_REFUSED].fetch_add(1, Ordering::Relaxed) < 8 {
                        tracing::warn!(bytes, %error, "memory pool: a chunk was not opened as a heap; its views keep their copies");
                    }
                    Err(Why::ChunkRefused)
                }
            },
        }
    };
    drop(spare);
    found
}

/// A handle of our own to the section with handle `section`, with the same
/// access; `None` if Windows gives none. The caller closes it.
fn duplicate(section: usize) -> Option<usize> {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut own = std::ptr::null_mut();
    // SAFETY: a handle of this process, known to the pool while its lock is
    // held; the out pointer is ours.
    let done = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            section as _,
            GetCurrentProcess(),
            &mut own,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    };
    (done != 0).then_some(own as usize)
}

/// A C-pool home of `len` bytes was made.
pub(crate) fn made(len: u64) {
    COUNTS[at::MADE].fetch_add(1, Ordering::Relaxed);
    COUNTS[at::LIVE].fetch_add(1, Ordering::Relaxed);
    COUNTS[at::LIVE_BYTES].fetch_add(len, Ordering::Relaxed);
}

/// A C-pool home of `len` bytes is gone.
pub(crate) fn gone(len: u64) {
    COUNTS[at::LIVE].fetch_sub(1, Ordering::Relaxed);
    COUNTS[at::LIVE_BYTES].fetch_sub(len, Ordering::Relaxed);
}

/// A C-pool home was let go (`crate::home::cpu_touched`).
pub(crate) fn let_go() {
    COUNTS[at::LET_GO].fetch_add(1, Ordering::Relaxed);
}

/// `n` view heap slots made on C-pool homes a let-go let go are made anew
/// at their next use (`crate::descriptor::host::pool_let_go`).
pub(crate) fn slots_made_anew(n: u64) {
    COUNTS[at::SLOTS_ANEW].fetch_add(n, Ordering::Relaxed);
}

/// Whether `buffer` is the buffer placed over a chunk of the pool (the
/// buffer of every C-pool home in it). Takes no lock ([`CHUNK_BUFFERS`])
/// unless an opened chunk's buffer found no place there; then the chunks'
/// lock, the last.
pub(crate) fn is_chunk_buffer(buffer: &ID3D12Resource) -> bool {
    use windows::core::Interface;
    let raw = buffer.as_raw();
    if is_listed(raw as usize) {
        return true;
    }
    CHUNK_BUFFERS_OVER.load(Ordering::SeqCst) != 0
        && chunks().values().any(|chunk| {
            chunk
                .opened
                .as_ref()
                .is_some_and(|o| o.buffer.as_raw() == raw)
        })
}

/// The slots made anew ([`slots_made_anew`]) and the ranges told to the
/// textures ([`textures_told`]) so far (tests).
#[cfg(test)]
pub(crate) fn told_counts() -> (u64, u64) {
    (
        COUNTS[at::SLOTS_ANEW].load(Ordering::Relaxed),
        COUNTS[at::TEXTURE_RANGES].load(Ordering::Relaxed),
    )
}

/// The GPU wrote title memory `[from, to)` of C-pool homes (or another view
/// of those pages, [`shown_at`]), which the guards do not see. The part
/// where placed textures lie (`crate::descriptor::host::pool_written`) is
/// told to them: the journal marks its pages written by the GPU
/// (`crate::watch::gpu_wrote`), which a draw's look at the textures it
/// reads sees, as a copy's write-back would have told them; and the slots
/// of placed textures shaders pick by index are made due again.
pub(crate) fn textures_told(from: usize, to: usize) {
    let Some((start, end)) = crate::descriptor::host::pool_written(from, to.saturating_sub(from))
    else {
        return;
    };
    COUNTS[at::TEXTURE_RANGES].fetch_add(1, Ordering::Relaxed);
    crate::watch::gpu_wrote(start, end - start);
}

/// The title ranges that show the pages under title memory `[from, to)`
/// (a C-pool home's bytes, which shaders wrote), at every view of them, as
/// `(start, end)`: the home's own range among them. Copies over any of
/// them (another view's buffer copy, the same pages mapped twice) take the
/// pages again (`crate::home::after_wait`).
///
/// shortcut: not the views of a page shown in very many views
/// (`crate::watch::MANY_VIEWS`, the title's page for unmapped places),
/// which `crate::inplace::views_of_section` leaves out; and the views are
/// those of the mapping after the wait: if the range was remapped between
/// the recording and the wait, the other views of the pages the GPU wrote
/// are missed (their copies keep what they had).
pub(crate) fn shown_at(from: usize, to: usize) -> Vec<(usize, usize)> {
    let len = to.saturating_sub(from);
    match crate::inplace::cover(from, len) {
        crate::inplace::Cover::Run(section, offset) => {
            crate::inplace::views_of_section(section, offset, len)
                .into_iter()
                .map(|(at, bytes)| (at, at.saturating_add(bytes)))
                .chain(std::iter::once((from, to)))
                .collect()
        }
        _ => vec![(from, to)],
    }
}

/// The pool's part of the memory core's ten-second line.
pub(crate) fn line() -> String {
    let c: Vec<u64> = COUNTS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let [kept, dropped, held, due] =
        [0, 1, 2, 3].map(|i| crate::descriptor::host::POOL_WRITES[i].load(Ordering::Relaxed));
    format!(
        "memory pool: chunks {} committed, {} reserved ({} short of commit, {} before build 20348, {} refused by the host, {} where no home can be made); \
         {} opened as heaps when a home was first wanted ({} refused, {} us, slowest {} us; {} asks turned down by the shared budget); \
         {} C-pool homes {} MB (made {}, {} let go by remaps; {} view heap slots on them made anew after a let-go); {} GPU-written ranges told to placed textures \
         ({kept} under texture views kept for the texture pass, {dropped} dropped; {held} slots held their bytes, {due} uploaded before and made due)",
        c[at::COMMITTED],
        c[at::SHORT] + c[at::OLD] + c[at::REFUSED] + c[at::NO_HOMES],
        c[at::SHORT],
        c[at::OLD],
        c[at::REFUSED],
        c[at::NO_HOMES],
        c[at::OPENED],
        c[at::OPEN_REFUSED],
        c[at::OPEN_US],
        c[at::OPEN_MOST_US],
        c[at::OVER_BUDGET],
        c[at::LIVE],
        c[at::LIVE_BYTES] >> 20,
        c[at::MADE],
        c[at::LET_GO],
        c[at::SLOTS_ANEW],
        c[at::TEXTURE_RANGES],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The chunk buffers' set finds what it holds and nothing else, and lets
    /// a buffer go. (Odd numbers: no COM pointer of another test's chunk is
    /// one. Which place a buffer takes is not asked: a test beside it may
    /// open a chunk.)
    #[test]
    fn the_chunk_buffers_set_finds_what_it_holds() {
        let [a, b, c] = [0x5f01usize, 0x5f03, 0x5f05];
        assert!(list(a) && list(b));
        assert!(is_listed(a) && is_listed(b) && !is_listed(c));
        unlist(a);
        assert!(!is_listed(a) && is_listed(b), "only the one let go");
        assert!(list(c));
        assert!(is_listed(c));
        unlist(b);
        unlist(c);
        assert!(!is_listed(b) && !is_listed(c));
    }

    /// A chunk is opened only if the shared budget keeps its margin after
    /// it; a budget of 0 (none reported) stops nothing.
    #[test]
    fn a_chunk_is_opened_only_within_the_shared_budget() {
        const GIB: u64 = 1 << 30;
        let chunk = 256 << 20;
        assert!(fits_shared(16 * GIB, 2 * GIB, chunk));
        assert!(
            fits_shared(16 * GIB, 15 * GIB - chunk, chunk),
            "exactly the margin left"
        );
        assert!(!fits_shared(16 * GIB, 15 * GIB - chunk + 1, chunk));
        assert!(!fits_shared(16 * GIB, 16 * GIB, chunk), "over: refused");
        assert!(!fits_shared(GIB / 2, 0, chunk), "a budget below the margin");
        assert!(fits_shared(0, u64::MAX, chunk), "no budget reported");
    }
}
