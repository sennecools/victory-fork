//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from blocks of 64 MiB made once (`VICTORY_TEXTURE_POOL=1`;
//! off by default: every host copy of such a texture and every staging
//! buffer is an allocation of its own, as before; `=texture` or `=staging`:
//! only that half).
//!
//! Each such texture is a committed resource of its own and its staging
//! buffer another. cool_fox measured an allocation at about 310
//! microseconds whatever its size, on one thread, and a texture placed in a
//! block at 6: its pass over the textures shaders pick by index got through
//! 350-500 of the 24,000 queued when the world loads in each of its 250 ms
//! turns, some forty seconds of a world without its textures.
//!
//! A piece given back is used again only [`REST`] later (nothing the card
//! was told before that is still running: Windows resets a card that takes
//! that long over one submission), and by then no list names it, as for
//! allocations of their own: a texture's piece comes back when the last
//! reference to the host texture goes (an object of ours the texture holds
//! says so, [`Reclaim`]), a staging buffer's when its `crate::texture` side
//! is dropped and the submissions running then have finished
//! (`crate::host::retire`).
//!
//! Ported from cool_fox (`pool.rs`), which this follows in its blocks,
//! pieces, rest, spare blocks, blocks made ahead ([`ahead`]) and the back-off
//! after a block the host refused ([`Pool::grow`]).

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Bytes of a block; pieces of more than half of one are not pooled (with
/// blocks made ahead, [`ahead`]: of more than one).
const BLOCK: u64 = 64 << 20;
/// Blocks with nothing in them that are kept.
const SPARE: usize = 2;
/// The same while the card has room ([`room`]): 4 GiB of them.
const SPARE_KEPT: usize = 64;
/// The most empty blocks of staging buffers kept, which are the PC's memory
/// (cool_fox measured 5.2 GB of them held in one run before this cap).
const STAGING_KEPT: usize = 8;
/// How many empty blocks are kept now ([`room`]).
static SPARES: AtomicUsize = AtomicUsize::new(SPARE);

/// Milliseconds until a piece given back is used again.
const REST: u64 = 2500;

/// Milliseconds a pool makes no block after the host refused one
/// ([`Pool::grow`]).
const BACK_OFF: u64 = 500;

fn now() -> u64 {
    crate::host::now_ms()
}

/// `VICTORY_TEXTURE_POOL`: which halves are on (textures, staging).
fn halves() -> (bool, bool) {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = switched();
        if on != (false, false) {
            tracing::info!(
                textures = on.0,
                staging = on.1,
                ripe_blocks = ahead(),
                "textures the title fills and their staging buffers are cut from 64 MiB blocks (VICTORY_TEXTURE_POOL)"
            );
        }
        on
    })
}

/// What `VICTORY_TEXTURE_POOL` says, read each time ([`halves`] keeps it;
/// [`ahead`] reads it too, from inside [`halves`]'s first call).
fn switched() -> (bool, bool) {
    match std::env::var("VICTORY_TEXTURE_POOL").as_deref() {
        Ok("1") => (true, true),
        Ok("texture") => (true, false),
        Ok("staging") => (false, true),
        _ => (false, false),
    }
}

/// Whether host copies of textures the title fills are cut from blocks.
pub(crate) fn textures_on() -> bool {
    halves().0
}

/// Whether texture staging buffers are cut from blocks.
pub(crate) fn staging_on() -> bool {
    halves().1
}

/// Sets how many empty blocks stay, by the card's memory (budget and use):
/// [`SPARE_KEPT`] below 85 % of the budget, else [`SPARE`]; and lets go of
/// those past it. Asked once a second (the sweep); does nothing, and asks
/// the card nothing, with the pool off.
pub(crate) fn room() {
    let (textures, staging) = halves();
    if !textures && !staging {
        return;
    }
    let limit = spares(crate::host::host().and_then(|h| h.video_memory()));
    SPARES.store(limit, Ordering::Relaxed);
    // A pool no texture is cut from for a while lets its empty blocks go
    // here; a busy one at its next piece.
    d3d::reclaim(limit);
}

fn spares(memory: Option<[u64; 4]>) -> usize {
    match memory {
        Some([budget, used, ..]) if u128::from(used) * 100 < u128::from(budget) * 85 => SPARE_KEPT,
        _ => SPARE,
    }
}

/// How many blocks of textures are made ahead, so that one has been there
/// for a while when its first piece is taken (`VICTORY_RIPE_BLOCKS`, a
/// count; unset: 12 while the texture half of `VICTORY_TEXTURE_POOL` is
/// on, as cool_fox runs, and with the pool off nothing is cut from blocks
/// anyway; 0: none, and a new block is used at once, the setup in which
/// cool_fox lost the device).
///
/// cool_fox lost the device to page faults of lists already on the card,
/// in a texture made the same tenth of a second as the first piece of a
/// block made for it: the title gives a slot that those lists read by index
/// a new texture. Why the card does not have the new block's addresses yet
/// is not known. A block that every submission begun before its making has
/// left behind ([`Ripe`]) has no such lists: pieces are cut from those
/// first.
fn ahead() -> usize {
    static AHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *AHEAD.get_or_init(|| {
        std::env::var("VICTORY_RIPE_BLOCKS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(if switched().0 { 12 } else { 0 })
    })
}

/// Submissions begun and finished (`crate::host::submissions`), and how
/// many blocks to make ahead (0: every block is ripe at once).
#[derive(Clone, Copy)]
struct Ripe {
    started: u64,
    finished: u64,
    ahead: usize,
}

impl Ripe {
    const ALL: Ripe = Ripe {
        started: 0,
        finished: 0,
        ahead: 0,
    };

    fn now() -> Ripe {
        let (started, finished) = crate::host::submissions();
        // Not more than are kept ([`room`]): they would go and be made again.
        Ripe {
            started,
            finished,
            ahead: ahead().min(SPARES.load(Ordering::Relaxed)),
        }
    }

    fn is<B>(self, block: &Block<B>) -> bool {
        self.ahead == 0 || block.made <= self.finished
    }
}

/// Pieces cut from blocks that were not ripe, for want of a ripe one.
static UNRIPE: AtomicU64 = AtomicU64::new(0);

/// The free ranges of a block: start and length, never two that touch.
struct Ranges(BTreeMap<u64, u64>);

impl Ranges {
    fn new(size: u64) -> Ranges {
        Ranges(BTreeMap::from([(0, size)]))
    }

    /// The lowest place for `len` bytes at a multiple of `align`.
    fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        let (start, size, at) = self.0.iter().find_map(|(&start, &size)| {
            let at = start.next_multiple_of(align);
            (at + len <= start + size).then_some((start, size, at))
        })?;
        self.0.remove(&start);
        if at > start {
            self.0.insert(start, at - start);
        }
        if at + len < start + size {
            self.0.insert(at + len, start + size - at - len);
        }
        Some(at)
    }

    fn give(&mut self, mut start: u64, mut len: u64) {
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

    fn whole(&self, size: u64) -> bool {
        self.0.get(&0) == Some(&size)
    }
}

/// A part of a block.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    block: usize,
    start: u64,
    len: u64,
}

struct Block<B> {
    memory: B,
    free: Ranges,
    /// The end of the highest piece ever taken: above it the memory is as
    /// it was made (zeros).
    used_to: u64,
    /// The newest submission begun when it was made ([`Ripe`]).
    made: u64,
}

struct Pool<B> {
    /// `None`: a block let go, its place to be used again.
    blocks: Vec<Option<Block<B>>>,
    /// Pieces given back, and when each may be used again.
    resting: VecDeque<(u64, Piece)>,
    /// The empty blocks it was last told to keep, and the most it keeps.
    spare: usize,
    most: usize,
    /// No block is made before this (ms): the host refused the last one.
    retry_growth_at: u64,
}

impl<B: Clone> Pool<B> {
    const fn new(most: usize) -> Pool<B> {
        Pool {
            blocks: Vec::new(),
            resting: VecDeque::new(),
            spare: 0,
            most,
            retry_growth_at: 0,
        }
    }

    /// A piece of `len` bytes and the memory it lies in, and whether that
    /// part was ever used before; `make` makes a block when none has room.
    fn take(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        self.take_ripe(len, align, now, Ripe::ALL, make)
    }

    /// The same from a block that is ripe, if one has room; `make` makes one
    /// ahead too, when few are empty.
    fn take_ripe(
        &mut self,
        len: u64,
        align: u64,
        now: u64,
        ripe: Ripe,
        mut make: impl FnMut() -> Option<B>,
    ) -> Option<(Piece, B, bool)> {
        // With blocks made ahead, what would be an allocation of its own (and
        // new when the card first reads it) lies in one of those too.
        if len == 0 || len > if ripe.ahead > 0 { BLOCK } else { BLOCK / 2 } {
            return None;
        }
        self.rested(now);
        let mut found = None;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if let Some(block) = block
                && ripe.is(block)
                && let Some(start) = block.free.take(len, align)
            {
                found = Some((i, start));
                break;
            }
        }
        if found.is_none() && ripe.ahead > 0 {
            // The one that has been there the longest.
            let mut unripe: Vec<(u64, usize)> = self
                .blocks
                .iter()
                .enumerate()
                .filter_map(|(i, b)| Some((b.as_ref()?.made, i)))
                .collect();
            unripe.sort_unstable();
            found = unripe
                .into_iter()
                .find_map(|(_, i)| Some((i, self.blocks[i].as_mut()?.free.take(len, align)?)));
        }
        // A refused block is not asked for again at each small texture:
        // the free ranges there are stay usable meanwhile.
        let (i, start) = match found {
            Some(found) => found,
            None => {
                let memory = self.grow(now, &mut make)?;
                let i = self.add(memory, ripe.started);
                (i, self.blocks[i].as_mut()?.free.take(len, align)?)
            }
        };
        if ripe.ahead > 0 {
            if !ripe.is(self.blocks[i].as_ref()?) {
                let n = UNRIPE.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    tracing::info!(
                        "textures cut from blocks made while lists ran that still run: {n}"
                    );
                }
            }
            let empty = self
                .blocks
                .iter()
                .flatten()
                .filter(|b| b.free.whole(BLOCK))
                .count();
            if empty < ripe.ahead
                && let Some(memory) = self.grow(now, &mut make)
            {
                self.add(memory, ripe.started);
            }
        }
        let block = self.blocks[i].as_mut()?;
        let again = start < block.used_to;
        block.used_to = block.used_to.max(start + len);
        Some((
            Piece {
                block: i,
                start,
                len,
            },
            block.memory.clone(),
            again,
        ))
    }

    /// A new block from `make`, unless the host refused one less than
    /// [`BACK_OFF`] ago: after a refusal (out of memory) the next asks wait
    /// that long, so that a burst of small uploads is not a burst of failing
    /// 64 MiB allocations.
    fn grow(&mut self, now: u64, make: &mut impl FnMut() -> Option<B>) -> Option<B> {
        if now < self.retry_growth_at {
            return None;
        }
        let block = make();
        if block.is_none() {
            self.retry_growth_at = now.saturating_add(BACK_OFF);
        }
        block
    }

    fn add(&mut self, memory: B, made: u64) -> usize {
        let i = match self.blocks.iter().position(Option::is_none) {
            Some(i) => i,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[i] = Some(Block {
            memory,
            free: Ranges::new(BLOCK),
            used_to: 0,
            made,
        });
        i
    }

    fn give(&mut self, piece: Piece, now: u64) {
        self.resting.push_back((now + REST, piece));
    }

    /// Makes the pieces that rested long enough free, and lets go of empty
    /// blocks past those kept.
    fn rested(&mut self, now: u64) {
        self.reclaim(now, SPARES.load(Ordering::Relaxed), false);
    }

    /// [`Pool::rested`] with `limit` empty blocks kept (at most its own
    /// most); `sweep`: looks for empty blocks past it even if no piece came
    /// free.
    fn reclaim(&mut self, now: u64, limit: usize, sweep: bool) {
        let mut any = false;
        while let Some(&(ready, piece)) = self.resting.front()
            && ready <= now
        {
            self.resting.pop_front();
            if let Some(block) = self.blocks[piece.block].as_mut() {
                block.free.give(piece.start, piece.len);
                any |= block.free.whole(BLOCK);
            }
        }
        // (Without a piece given back too, once there is less room.)
        let most = limit.min(self.most);
        let previous = std::mem::replace(&mut self.spare, most);
        if sweep || any || most < previous {
            let mut spare = 0;
            for block in &mut self.blocks {
                if block.as_ref().is_some_and(|b| b.free.whole(BLOCK)) {
                    spare += 1;
                    if spare > most {
                        *block = None;
                    }
                }
            }
        }
    }

    fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }
}

pub(crate) use d3d::*;

/// The pools' D3D12 side.
mod d3d {
    use std::ffi::c_void;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::{GUID, HRESULT, IUnknown, Interface};

    use super::{BLOCK, Piece, Pool, Ripe, SPARE_KEPT, STAGING_KEPT, now};

    /// A mapped upload buffer's memory.
    #[derive(Clone)]
    pub(super) struct Upload(ID3D12Resource, *mut u8);
    // SAFETY: the mapping is written by the owners of its pieces, each its
    // own.
    unsafe impl Send for Upload {}

    pub(super) static TEXTURES: Mutex<Pool<ID3D12Heap>> = Mutex::new(Pool::new(SPARE_KEPT));
    pub(super) static STAGING: Mutex<Pool<Upload>> = Mutex::new(Pool::new(STAGING_KEPT));

    fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Pieces of textures that went, and when, not yet in [`TEXTURES`]:
    /// a texture gives its piece back from inside the runtime's destruction
    /// of it ([`release`]), which may hold locks of the runtime's own, while
    /// [`TEXTURES`] is held across the making of a block (`CreateHeap`).
    /// Taking only this, never held across a call to the runtime, keeps
    /// the two from waiting on each other.
    static GIVEN: Mutex<Vec<(u64, Piece)>> = Mutex::new(Vec::new());

    /// [`TEXTURES`], with the pieces given back meanwhile in it.
    fn textures() -> std::sync::MutexGuard<'static, Pool<ID3D12Heap>> {
        let mut pool = lock(&TEXTURES);
        given_into(&mut pool);
        pool
    }

    fn given_into(pool: &mut Pool<ID3D12Heap>) {
        for (at, piece) in std::mem::take(&mut *lock(&GIVEN)) {
            pool.give(piece, at);
        }
    }

    /// [`super::room`]'s look at the pools, passed over while one is busy.
    pub(super) fn reclaim(limit: usize) {
        let tick = now();
        if let Ok(mut pool) = TEXTURES.try_lock() {
            given_into(&mut pool);
            pool.reclaim(tick, limit, true);
        }
        if let Ok(mut pool) = STAGING.try_lock() {
            pool.reclaim(tick, limit, true);
        }
    }

    /// Bytes of the pieces textures hold now.
    static HELD: AtomicU64 = AtomicU64::new(0);

    /// Bytes of the texture blocks no texture holds: memory of the card in
    /// use by us that new textures take first (0 with the pool off).
    pub(crate) fn texture_room() -> u64 {
        if !super::textures_on() {
            return 0;
        }
        (lock(&TEXTURES).blocks() as u64 * BLOCK).saturating_sub(HELD.load(Ordering::Relaxed))
    }

    /// Part of a block of mapped upload memory, back in the pool when
    /// dropped.
    pub(crate) struct StagingPiece(Piece);

    impl Drop for StagingPiece {
        fn drop(&mut self) {
            lock(&STAGING).give(self.0, now());
        }
    }

    /// `bytes` of upload memory, all zeros: the piece, the buffer it lies
    /// in, where in that buffer, and its mapping. `None`: not pooled (the
    /// staging half off, too big, or no block to be had).
    pub(crate) fn staging(
        host: &crate::host::Host,
        bytes: u64,
    ) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
        if !super::staging_on() {
            return None;
        }
        stage(host, bytes)
    }

    /// [`staging`] whatever the switch says.
    fn stage(
        host: &crate::host::Host,
        bytes: u64,
    ) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
        // Copies from a buffer begin at multiples of 512.
        let bytes = bytes.next_multiple_of(512);
        let (piece, memory, again) = lock(&STAGING).take(bytes, 512, now(), || {
            let (buffer, data) = host.upload_buffer(BLOCK)?;
            crate::host::label(&buffer, "texture staging block", BLOCK);
            Some(Upload(buffer, data))
        })?;
        let owned = StagingPiece(piece);
        // SAFETY: the piece lies in the block's mapping.
        let data = unsafe { memory.1.add(piece.start as usize) };
        if again {
            // What is not detiled into it (an unmapped texture) uploads
            // zeros, as from a buffer of its own.
            // SAFETY: the piece is ours alone.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        }
        Some((owned, memory.0, piece.start, data))
    }

    /// Tests: whether the staging pool holds `buffer` as one of its blocks,
    /// and whether the piece at `start` in it has been given back (resting
    /// or free again: no longer any holder's).
    #[cfg(test)]
    pub(crate) fn staging_piece(buffer: &ID3D12Resource, start: u64) -> (bool, bool) {
        let pool = lock(&STAGING);
        let Some(i) = pool
            .blocks
            .iter()
            .position(|b| b.as_ref().is_some_and(|b| b.memory.0 == *buffer))
        else {
            return (false, false);
        };
        let given = pool
            .resting
            .iter()
            .any(|&(_, p)| p.block == i && p.start == start)
            || pool.blocks[i].as_ref().is_some_and(|b| {
                b.free
                    .0
                    .range(..=start)
                    .next_back()
                    .is_some_and(|(&at, &len)| start < at + len)
            });
        (true, given)
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

    /// Pooled textures the card has neither uploaded into nor written by a
    /// copy (by address): what their piece held before still shows in them
    /// ([`first_write`]).
    static UNTOUCHED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

    /// Whether `texture` is a pooled one the card writes for the first
    /// time, which it no longer is after this (false with the texture half
    /// off, without taking a lock). The caller records the aliasing barrier
    /// and gives every texel a value before anything else uses it.
    pub(crate) fn first_write(texture: &ID3D12Resource) -> bool {
        if !super::textures_on() {
            return false;
        }
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

    /// Whether `texture` was cut from a block (false with the texture half
    /// off, without asking the texture).
    pub(crate) fn in_block(texture: &ID3D12Resource) -> bool {
        if !super::textures_on() {
            return false;
        }
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

    /// A texture of `desc` (no target, depth or unordered access) in a
    /// block, in `state`; `None`: not pooled (the texture half off, too big,
    /// or the host refused).
    ///
    /// # Safety
    /// `desc` is a valid description.
    pub(crate) unsafe fn texture(
        device: &ID3D12Device,
        desc: *const D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        if !super::textures_on() {
            return None;
        }
        // SAFETY: the caller's description.
        unsafe { place(device, desc, state) }
    }

    /// [`texture`] whatever the switch says.
    ///
    /// # Safety
    /// As for [`texture`].
    unsafe fn place(
        device: &ID3D12Device,
        desc: *const D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        // Small textures lie at multiples of 4 KiB where the host allows it
        // (the top level is 64 KiB at most): 64 KiB each otherwise.
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
        if info.SizeInBytes == u64::MAX {
            return None;
        }
        let (piece, heap, _) = textures().take_ripe(
            info.SizeInBytes,
            info.Alignment.max(1),
            now(),
            Ripe::now(),
            || {
                let desc = D3D12_HEAP_DESC {
                    SizeInBytes: BLOCK,
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
                    tracing::warn!(error = %e, "no block of texture memory");
                }
                if let Some(heap) = &heap {
                    crate::host::name(heap, || "textures the title fills (a block)".into());
                }
                heap
            },
        )?;
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
                lock(&TEXTURES).give(piece, now());
                None
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A pooled texture's piece comes back when its last reference
        /// goes, not before; its staging piece begins as zeros (asked of
        /// the pools whatever `VICTORY_TEXTURE_POOL` says).
        #[test]
        fn warp_a_texture_gives_its_piece_back_when_it_goes() {
            let Some(host) = crate::host::host() else {
                return;
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 256,
                Height: 256,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            // SAFETY: a valid description.
            let Some(t) = (unsafe { place(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON) })
            else {
                return;
            };
            assert_eq!(
                crate::host::references(&t),
                1,
                "what the texture holds is no reference to it"
            );
            let resting = || textures().resting.len();
            let before = resting();
            let other = t.clone();
            drop(t);
            assert_eq!(resting(), before);
            drop(other);
            assert_eq!(resting(), before + 1);
            // SAFETY: the piece's mapping, 4096 bytes of it ours.
            unsafe {
                let (piece, _, at, data) = stage(host, 4096).unwrap();
                assert!(at.is_multiple_of(512));
                std::ptr::write_bytes(data, 0xAB, 4096);
                drop(piece);
                lock(&STAGING).rested(u64::MAX);
                let (_piece, _, _, data) = stage(host, 4096).unwrap();
                assert!((0..4096).all(|i| *data.add(i) == 0));
                assert!(host.device.GetDeviceRemovedReason().is_ok());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn empty_blocks_stay_while_the_card_has_room() {
        assert_eq!(spares(Some([100, 84, 0, 0])), SPARE_KEPT);
        assert_eq!(spares(Some([100, 85, 0, 0])), SPARE);
        assert_eq!(spares(Some([0, 0, 0, 0])), SPARE);
        assert_eq!(spares(None), SPARE);
    }

    #[test]
    fn a_failed_block_is_not_retried_for_each_small_upload() {
        let mut pool: Pool<u32> = Pool::new(2);
        let calls = std::cell::Cell::new(0);
        let refuse = || {
            calls.set(calls.get() + 1);
            None
        };
        for size in [704008, 179720, 179720, 24080, 48648, 524288, 89616, 351760] {
            assert!(pool.take(size, 512, 12, refuse).is_none());
        }
        assert_eq!(calls.get(), 1);
        assert!(
            pool.take(24080, 512, 12 + BACK_OFF - 1, || Some(1))
                .is_none()
        );
        assert!(pool.take(24080, 512, 12 + BACK_OFF, || Some(1)).is_some());
    }

    #[test]
    fn a_reclaim_lets_idle_blocks_go_without_reusing_live_or_resting_pieces() {
        let mut pool = Pool::new(STAGING_KEPT);
        let (held, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(1)).unwrap();
        let (resting, _, _) = pool.take(BLOCK / 2, 1, 0, || Some(2)).unwrap();
        pool.give(resting, 0);
        for i in 0..12 {
            pool.add(i, 0);
        }
        pool.reclaim(0, SPARE_KEPT, true);
        assert_eq!(pool.blocks(), STAGING_KEPT + 1);
        pool.reclaim(REST - 1, SPARE, true);
        assert_eq!(pool.blocks(), SPARE + 1);
        assert_eq!(pool.resting.len(), 1);
        pool.reclaim(REST, 0, true);
        assert_eq!(pool.blocks(), 1);
        assert!(pool.resting.is_empty());
        pool.give(held, REST);
        pool.reclaim(2 * REST, 0, true);
        assert_eq!(pool.blocks(), 0);
    }

    #[test]
    fn pieces_never_overlap_and_rest_before_they_are_used_again() {
        let mut pool: Pool<u32> = Pool::new(SPARE);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let mut taken: Vec<Piece> = Vec::new();
        for i in 0..500u64 {
            let (p, _, again) = pool.take(65536 * (1 + i % 7), 65536, 0, make).unwrap();
            assert!(!again);
            assert!(taken.iter().all(|q| q.block != p.block
                || p.start + p.len <= q.start
                || q.start + q.len <= p.start));
            taken.push(p);
        }
        let blocks = pool.blocks();
        assert!(blocks >= 2);
        // Given back at 10: not used again before 10 + REST.
        let first = taken[0];
        pool.give(first, 10);
        let (p, _, _) = pool.take(first.len, 65536, 10 + REST - 1, make).unwrap();
        assert_ne!((p.block, p.start), (first.block, first.start));
        let (q, _, again) = pool.take(first.len, 65536, 10 + REST, make).unwrap();
        assert_eq!((q.block, q.start, again), (first.block, first.start, true));
        // Everything back: the empty blocks past the spare ones go.
        for p in taken.drain(1..).chain([p, q]) {
            pool.give(p, 20);
        }
        pool.reclaim(20 + REST, SPARE, false);
        assert_eq!(pool.blocks(), blocks.min(SPARE));
        // Too big for a block, and a block that cannot be made.
        assert!(pool.take(BLOCK, 65536, 30, make).is_none());
        let mut none: Pool<u32> = Pool::new(SPARE);
        assert!(none.take(65536, 65536, 0, || None).is_none());
    }

    #[test]
    fn pieces_come_from_blocks_older_than_the_lists_that_run() {
        let mut pool: Pool<u64> = Pool::new(SPARE_KEPT);
        let made = std::cell::Cell::new(0);
        let make = || {
            made.set(made.get() + 1);
            Some(made.get())
        };
        let at = |started, finished| Ripe {
            started,
            finished,
            ahead: 2,
        };
        let take =
            |pool: &mut Pool<u64>, ripe| pool.take_ripe(BLOCK / 2, 65536, 0, ripe, make).unwrap().1;
        // Nothing runs: at once, and two are made ahead, one a time.
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 2);
        assert_eq!(take(&mut pool, at(0, 0)), 1);
        assert_eq!(made.get(), 3);
        // 4 is made while submission 5 runs.
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(made.get(), 4);
        assert_eq!(take(&mut pool, at(5, 4)), 2);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        assert_eq!(take(&mut pool, at(6, 4)), 3);
        // None is ripe: the oldest; it is ripe once 5 has finished.
        assert_eq!(take(&mut pool, at(7, 4)), 4);
        assert_eq!(take(&mut pool, at(7, 5)), 4);
        // A whole block's worth too.
        assert_eq!(
            pool.take_ripe(BLOCK, 65536, 0, at(8, 7), make).unwrap().1,
            5
        );
        // Off: the first with room, whatever its age, none ahead.
        let n = made.get();
        assert_eq!(pool.take(BLOCK / 2, 65536, 0, make).unwrap().1, 6);
        assert_eq!(made.get(), n);
        assert!(pool.take(BLOCK, 65536, 0, make).is_none());
    }
}
