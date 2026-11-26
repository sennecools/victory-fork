//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory was written since the last detiling
//! ([`crate::watch`]), in write-watched memory; in physical-page views,
//! which cannot be watched, a content hash of the whole memory at every
//! submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Only textures read through descriptor tables are brought up to date
//! (directly indexed views are not known per draw); plane 0 is uploaded
//! (every mip and array slice), depth and stencil planes are not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A staging buffer: an upload buffer, mapped, that a texture's
/// subresources are detiled into and copied from. Its D3D12 buffer is
/// retired when it goes (`crate::host::retire`): dropped only once every
/// submission running then has finished, whoever lets go of it last.
pub(crate) struct Buf {
    /// Some until it is dropped. With [`Buf::piece`], the block the piece
    /// lies in.
    buffer: Option<ID3D12Resource>,
    /// Its mapping (with a piece, where the piece starts in the block's).
    data: *mut u8,
    /// Where in `buffer` it starts: 0, or with a piece where that lies
    /// (copies from it add this to the footprints' offsets).
    base: u64,
    /// Cut from a block of `crate::texture_pool` (`VICTORY_TEXTURE_POOL`):
    /// retired with the buffer when it goes, back in the pool once the
    /// submissions running then have finished.
    piece: Option<crate::texture_pool::StagingPiece>,
    /// Its bytes as made: the texture's, or with [`transient`] a size class
    /// ([`class_of`]).
    bytes: u64,
    /// With [`transient`]: the fence of the last submission sent ahead that
    /// read it, kept when it goes back to the pool; the next write into it
    /// waits for that fence (`Staging::detile`).
    fence: std::sync::atomic::AtomicU64,
}

// SAFETY: the mapping is written only at submission, under its texture's
// `last` lock, or by nobody while it lies in the pool.
unsafe impl Send for Buf {}
// SAFETY: as above.
unsafe impl Sync for Buf {}

impl Buf {
    /// An upload buffer of `bytes` bytes, counted in [`STAGED`].
    fn new(bytes: u64) -> Option<Buf> {
        let (buffer, data) = host()?.upload_buffer(bytes)?;
        crate::host::label(&buffer, "texture staging", bytes);
        STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        STAGED[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        Some(Buf {
            buffer: Some(buffer),
            data,
            base: 0,
            piece: None,
            bytes,
            fence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// [`Buf::new`], cut from a block of `crate::texture_pool` if its
    /// staging half is on (all zeros, as a new buffer); a buffer of its own
    /// otherwise. Not for [`transient`] buffers, whose pool asks whether a
    /// list holds a buffer by its references, which a block's buffer has
    /// for all its pieces.
    fn pooled(bytes: u64) -> Option<Buf> {
        let host = host()?;
        let Some((piece, buffer, base, data)) = crate::texture_pool::staging(host, bytes) else {
            return Buf::new(bytes);
        };
        STAGED[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        STAGED[1].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        Some(Buf {
            buffer: Some(buffer),
            data,
            base,
            piece: Some(piece),
            bytes,
            fence: std::sync::atomic::AtomicU64::new(0),
        })
    }

    fn buffer(&self) -> &ID3D12Resource {
        self.buffer.as_ref().expect("a staging buffer lives until it is dropped")
    }

    /// Whether only this holds the D3D12 buffer: no list that recorded a
    /// copy from it is still to be reset (a list keeps what it copies from
    /// until then, and is reset only after its last submission ran).
    fn unheld(&self) -> bool {
        crate::host::references(self.buffer()) == 1
    }

    /// Before the CPU writes it for `staging`: waits for the submissions
    /// sent ahead that may still copy from it (the last one that read it
    /// for this texture, and for the one it served before the pool).
    fn wait_to_write(&self, staging: &Staging) {
        use std::sync::atomic::Ordering::Relaxed;
        let fence = staging.last_upload.swap(0, Relaxed).max(self.fence.swap(0, Relaxed));
        if fence != 0 {
            host().expect("an upload has a host").wait_for_staging(fence);
        }
    }
}

impl Drop for Buf {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        STAGED[0].fetch_sub(1, Relaxed);
        STAGED[1].fetch_sub(self.bytes, Relaxed);
        // A submission sent ahead may still copy from it, and a staging
        // buffer freed under one would be a GPU page fault. A guard: no
        // such order was found (suspected in run r2b-standin-2, which lost
        // the device at a fault inside freed staging allocations; not
        // shown). Whatever holds it last, the buffer itself waits for every
        // submission running.
        if let Some(buffer) = self.buffer.take() {
            crate::host::retire(buffer);
        }
        // A piece of a block waits the same way, and then rests.
        if let Some(piece) = self.piece.take() {
            crate::host::retire(piece);
        }
    }
}

/// A texture's upload state: the staging buffer laid out as the host
/// texture's subresources, and what was last detiled.
pub(crate) struct Staging {
    /// The staging buffer. Always there without [`transient`]; with it,
    /// only from a detile or a recorded copy until the upload has run
    /// ([`release_uploaded`]).
    buf: Mutex<Option<Arc<Buf>>>,
    /// Made with [`transient`] on.
    transient: bool,
    /// With [`transient`]: a buffer was given to the texture for a copy a
    /// list recorded, and holds none of its bytes yet (or a detiling found
    /// no buffer to be had); the next refresh detiles whether the memory
    /// changed or not. If it cannot, a texture never detiled gets zeros; a
    /// texture detiled before keeps its host copy's texels, and no list
    /// copies from the buffer until it is filled
    /// ([`Staging::record_copies`]).
    fresh: std::sync::atomic::AtomicBool,
    /// With [`transient`]: the last upload a list asked for recorded no
    /// copy ([`Recorded::Later`]) and none since did. The pass over
    /// textures picked by index keeps such a texture's slots due until a
    /// list copies it ([`copy_pending`]).
    later: std::sync::atomic::AtomicBool,
    /// Detilings done into the staging buffer (for [`Staging::refresh`]).
    detiles: std::sync::atomic::AtomicU64,
    /// When a list last recorded a copy from the staging buffer
    /// (`crate::host::moment`).
    copied: std::sync::atomic::AtomicU64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    /// The bytes the subresources take in the staging buffer.
    bytes: u64,
    memory: *const u8,
    last: Mutex<Last>,
    /// Whether the staging buffer was detiled since a submitted list last
    /// uploaded it, and whether one has uploaded it at all: a list records
    /// the upload only then. (Every list that sampled a title texture
    /// uploaded all of it again, changed or not: 5,411 subresource copies
    /// of 1,120 textures in one frame of run 18.)
    dirty: std::sync::atomic::AtomicBool,
    uploaded: std::sync::atomic::AtomicBool,
    /// The fence of the last late submission that actually reads this
    /// staging buffer. Checked only when new contents will be written.
    last_upload: std::sync::atomic::AtomicU64,
    /// When a list last recorded a use of it (`crate::host::moment`).
    used: std::sync::atomic::AtomicU64,
    /// Levels on demand (`crate::tiles`): the texture's host copy was made
    /// without memory of its own when the staging buffer was made.
    on_demand: bool,
    /// Made by the pass over textures picked by index for one upload
    /// ([`once`]): filled at the list's submission without asking the write
    /// watch or the guards, and never looked at again. A draw that samples
    /// the texture through a table takes it back to the rule of the others.
    once: std::sync::atomic::AtomicBool,
    /// For such a texture: the levels that were the title's at the last
    /// detiling (a bit a level; the staging buffer holds zeros for the
    /// others), and what that was judged by ([`Staging::judgement`]).
    detiled: std::sync::atomic::AtomicU32,
    judged: [std::sync::atomic::AtomicU64; 2],
    /// For such a texture: when an upload last left a held level without
    /// tiles (`crate::host::moment`, never 0; 0: none did). The upload is
    /// recorded again about a second later if the card has room by then
    /// ([`upload`]).
    waiting: std::sync::atomic::AtomicU64,
}

/// What tells a texture's memory changed (see [`content_hash`]). Kept in
/// memory only: no hash of it is stored anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hash {
    Blake3(blake3::Hash),
    Digest(Digest),
}

/// Whether textures' memory is told changed by [`digest`] instead of
/// blake3 (`VICTORY_TEXTURE_DIGEST=1`; off by default: blake3, as before).
/// cool_fox measured blake3 at 0.2 to 0.27 ms a megabyte, four times the
/// detiling it is to spare, and the digest at 0.035. Also for
/// [`fingerprint`].
fn cheap_hash() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_DIGEST").is_some_and(|v| v == "1"))
}

/// The hash of a texture's memory: [`digest`] with [`cheap_hash`], else
/// blake3.
fn content_hash(bytes: &[u8]) -> Hash {
    if cheap_hash() {
        Hash::Digest(digest(bytes))
    } else {
        Hash::Blake3(blake3::hash(bytes))
    }
}

/// A 128-bit digest of `bytes`, to tell a texture's memory changed: four
/// lanes of folded multiplications over 64 bytes at a time. Not a
/// cryptographic one: nobody makes up textures to collide.
pub(crate) fn digest(bytes: &[u8]) -> Digest {
    const K: [u64; 8] = [
        0xa076_1d64_78bd_642f,
        0xe703_7ed1_a0b4_28db,
        0x8ebc_6af0_9c88_c6e3,
        0x5899_65cc_7537_4cc3,
        0x1d8e_4e27_c47d_124f,
        0xeb44_acca_b455_d165,
        0x9e37_79b9_7f4a_7c15,
        0xc2b2_ae3d_27d4_eb4f,
    ];
    let fold = |a: u64, b: u64| {
        let p = u128::from(a) * u128::from(b);
        p as u64 ^ (p >> 64) as u64
    };
    let word = |b: &[u8]| {
        let mut w = [0; 8];
        w[..b.len()].copy_from_slice(b);
        u64::from_le_bytes(w)
    };
    let mut lanes = [K[0], K[1], K[2], K[3]];
    let mut step = |block: &[u8; 64]| {
        for (i, lane) in lanes.iter_mut().enumerate() {
            let a = word(&block[16 * i..16 * i + 8]);
            let b = word(&block[16 * i + 8..16 * i + 16]);
            // (What went in is added again: a product with a factor of zero
            // forgets neither the lane nor the block.)
            *lane = fold(a ^ K[4 + i], b ^ *lane)
                .wrapping_add(a.rotate_left(23) ^ lane.rotate_left(41))
                ^ b;
        }
    };
    let mut blocks = bytes.chunks_exact(64);
    for block in &mut blocks {
        step(block.try_into().unwrap_or(&[0; 64]));
    }
    let rest = blocks.remainder();
    let mut last = [0u8; 64];
    last[..rest.len()].copy_from_slice(rest);
    step(&last);
    let len = bytes.len() as u64;
    let low = fold(lanes[0] ^ K[1], lanes[1] ^ len) ^ fold(lanes[2] ^ K[3], lanes[3] ^ K[5]);
    let high = fold(lanes[0].rotate_left(32) ^ K[6], lanes[2] ^ len)
        ^ fold(lanes[1].rotate_left(32) ^ K[7], lanes[3] ^ K[2]);
    Digest(u128::from(low) << 64 | u128::from(high))
}

/// See [`digest`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Digest(u128);

/// What a texture's staging buffer holds: the hash of the memory detiled
/// (memory not watched), or the write-watch asking it was detiled after.
#[derive(Default)]
struct Last {
    hash: Option<Hash>,
    asked: u64,
    /// The frame a partly mapped texture was last read in.
    partial: Option<u64>,
    /// The frame a hash last found the texture's memory changed in (see
    /// [`guard_skips`]).
    changed: Option<u64>,
    /// The hash of what the staging buffer holds, if a hash was what it was
    /// last detiled after (none once the write-watch path detiled it): what
    /// [`Staging::note_held`] says the host copy holds. `hash` alone would
    /// name bytes a detiling after the write-watch replaced, and memory that
    /// went back to those bytes would be taken as uploaded.
    held: Option<Hash>,
}

impl Last {
    /// Takes `hash` as the hash of the texture's memory; true when it
    /// differs from the last one, and then the frame is noted.
    fn rehash(&mut self, hash: Hash) -> bool {
        if self.hash == Some(hash) {
            return false;
        }
        self.hash = Some(hash);
        self.held = Some(hash);
        self.changed = Some(crate::frame::current());
        true
    }
}

/// A guarded texture in view memory is hashed every eighth frame only, by
/// its address so that textures take turns.
const HASH_EVERY: u64 = 8;
/// For this many frames after a hash found a guarded texture changed, it is
/// hashed at every check. A texture the title rewrites through another
/// view (startup screens, video frames) changes where the guard cannot see
/// it; hashed every eighth frame only, it showed repeated and stale frames
/// and flickered (#269). Two turns, so a texture that keeps changing never
/// leaves this state while one that stops goes back to taking turns.
const RECENTLY_CHANGED: u64 = 2 * HASH_EVERY;

/// Whether a guarded texture in view memory is passed over at a check in
/// `frame`: the guard saw no write to it (`unwritten`), it is not its turn
/// to be hashed, and no hash found it changed lately.
fn guard_skips(unwritten: bool, frame: u64, start: usize, changed: Option<u64>) -> bool {
    let turn = frame
        .wrapping_add((start >> 16) as u64)
        .is_multiple_of(HASH_EVERY);
    let recent = changed.is_some_and(|at| frame.wrapping_sub(at) < RECENTLY_CHANGED);
    unwritten && !turn && !recent
}

impl Last {
    /// Sparse view blocks can be reported as written on every guard query
    /// even when unchanged. Keep their expensive reads bounded regardless.
    fn partial_due(&self, frame: u64) -> bool {
        self.partial.is_none_or(|at| frame.saturating_sub(at) >= 30)
    }

    /// All-zero reads from partially mapped memory are not uploaded.
    /// Do not record their hash as if they had replaced the staging data.
    fn accept_partial(&mut self, bytes: &[u8]) -> bool {
        if bytes.iter().all(|&b| b == 0) {
            return false;
        }
        self.rehash(content_hash(bytes))
    }
}

// SAFETY: the staging mapping is written only at submission, under
// `last`'s lock; the title memory is only read.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

static STAGING: Mutex<Option<HashMap<usize, Arc<Staging>>>> = Mutex::new(None);

/// The buffer a texture the host writes is copied through into the texture
/// of a view the title placed in its memory in another format
/// ([`alias`]): every level of every layer as each of the two formats lays
/// it out for a copy (the same bytes: the texels are as wide), the written
/// texture's layout first.
struct Through {
    buffer: ID3D12Resource,
    parts: Vec<(
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    )>,
}

/// The most levels times layers [`alias`] copies (a cube of nine levels).
const ALIAS_PARTS: u32 = 54;

// SAFETY: a device-local buffer only lists record copies with.
unsafe impl Send for Through {}
// SAFETY: as above.
unsafe impl Sync for Through {}

/// The copy buffers of [`alias`], by the placed view's texture.
static THROUGH: Mutex<Option<HashMap<usize, Arc<Through>>>> = Mutex::new(None);

/// Brings `placed`, the texture of a view the title placed in the memory
/// of `source` in another format (`crate::resource::alias_of`), up to date
/// with what the host wrote to `source`: the texels of every level and
/// layer are copied as they are, through a buffer (the host copies between
/// two formats of one texel width in no other way). Recorded the first time
/// the list reads the view
/// and again whenever the list used `source` in another way since (wrote
/// it, or read it itself). Leaves `placed` in the copy destination state,
/// as [`upload`] does. False if the copy cannot be made: the caller fills
/// the texture from title memory, as before.
pub(crate) fn alias(list: &HostList, placed: This, source: This) -> bool {
    const TARGET_OR_UNORDERED: u32 = 0x4 | 0x8;
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(to), Some(from)) = (
        crate::resource::host_resource(placed),
        crate::resource::host_resource(source),
    ) else {
        return false;
    };
    let Some(host) = host() else { return false };
    let through = {
        let mut map = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        match map.get(&(placed as usize)) {
            Some(through) => through.clone(),
            None => {
                // SAFETY: live textures.
                let (of_from, of_to) = unsafe { (from.GetDesc(), to.GetDesc()) };
                // Levels times layers (a volume's slices are one part).
                let count = |d: &D3D12_RESOURCE_DESC| {
                    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                        1
                    } else {
                        u32::from(d.DepthOrArraySize)
                    };
                    u32::from(d.MipLevels) * layers
                };
                let parts = count(&of_from);
                if parts == 0 || parts > ALIAS_PARTS || parts != count(&of_to) {
                    return false;
                }
                let mut rows = [(); 2]
                    .map(|()| vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); parts as usize]);
                let mut bytes = [0u64; 2];
                // SAFETY: plain queries; the out pointers are ours and
                // hold `parts` layouts each.
                unsafe {
                    host.device.GetCopyableFootprints(
                        &of_from,
                        0,
                        parts,
                        0,
                        Some(rows[0].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[0]),
                    );
                    host.device.GetCopyableFootprints(
                        &of_to,
                        0,
                        parts,
                        0,
                        Some(rows[1].as_mut_ptr()),
                        None,
                        None,
                        Some(&mut bytes[1]),
                    );
                }
                let alike = rows[0].iter().zip(&rows[1]).all(|(a, b)| {
                    let (fa, fb) = (a.Footprint, b.Footprint);
                    a.Offset == b.Offset
                        && (fa.Width, fa.Height, fa.Depth, fa.RowPitch)
                            == (fb.Width, fb.Height, fb.Depth, fb.RowPitch)
                });
                if bytes[0] == 0 || bytes[0] != bytes[1] || !alike {
                    return false;
                }
                let Some(buffer) = host.uav_buffer(bytes[0]) else {
                    return false;
                };
                crate::host::label(&buffer, "texture alias copy", bytes[0]);
                crate::host::name(&buffer, || {
                    format!(
                        "alias copy {}x{} f{} to f{}",
                        of_from.Width, of_from.Height, of_from.Format.0, of_to.Format.0
                    )
                });
                tracing::info!(
                    view = ?placed,
                    texture = ?source,
                    width = of_from.Width,
                    height = of_from.Height,
                    parts,
                    written_as = of_from.Format.0,
                    read_as = of_to.Format.0,
                    "a view placed in the memory of a texture the host writes, in another format: read from that texture by a copy on the host"
                );
                let [of_written, of_read] = rows;
                let through = Arc::new(Through {
                    buffer,
                    parts: of_written.into_iter().zip(of_read).collect(),
                });
                map.insert(placed as usize, through.clone());
                through
            }
        }
    };
    let first = list.first_use(placed as usize);
    // The source's first use in the list records nothing (its prelude
    // brings it to the state); a use in another way since the last copy
    // shows as a change of state.
    let before = list.transition(source, COPY_SOURCE).map(|(texture, before)| {
        let state = crate::host::legal_state(texture, COPY_SOURCE);
        let barrier = crate::host::transition(
            texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(state as i32),
        );
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
        before
    });
    if !first && before.is_none() {
        return true;
    }
    card_writes(list, placed, to, None);
    crate::list::ensure_state(list, placed, COPY_DEST);
    let part = |texture: &ID3D12Resource, index: usize| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index as u32,
        },
    };
    let rows = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&through.buffer) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, read, written) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("alias copy of {source:?} into {placed:?}"));
    // SAFETY: an open list and live resources; the buffer is as big as
    // the footprints and rests in the common state between copies.
    unsafe {
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, common, written)]);
        for (index, (of_written, _)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&rows(of_written), 0, 0, 0, &part(from, index), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, written, read)]);
        for (index, (_, of_read)) in through.parts.iter().enumerate() {
            list.list
                .CopyTextureRegion(&part(to, index), 0, 0, 0, &rows(of_read), None);
        }
        list.list
            .ResourceBarrier(&[crate::host::transition(&through.buffer, read, common)]);
    }
    list.keep(&through.buffer);
    // A draw that also writes the source (a view of it handled before
    // this one) needs it in that state again; the next read of the alias
    // then copies once more, as it must after a write.
    if let Some(before) = before
        && before & TARGET_OR_UNORDERED != 0
    {
        crate::list::ensure_state(list, source, before);
    }
    true
}

/// Whether a texture refresh that finds the runtime remapping title memory
/// goes on without waiting for it (`VICTORY_TEXTURE_NO_WAIT=1`; off by
/// default).
///
/// A texture whose memory is all mapped is read in place, under a lock
/// the runtime takes alone while it changes a mapping
/// (`crate::watch::title_memory_reading`), so that no page goes away under
/// the read. While the world streams in the runtime remaps all the time,
/// and each of the frame's 150-360 refreshes waited for it: runs 99-101
/// have 4, 15 and 16 frame lines with over 300 ms of "textures", up to
/// 3,708 ms, and on every one of them the wait for this lock is the whole
/// of it (1,345 of 1,381 ms; 3,630 of 3,708; the write-watch askings 3-12
/// ms, the texture's own lock 0).
///
/// With the switch the refresh asks for the lock without waiting. If the
/// runtime has it, the texture is read through the kernel instead, which
/// needs no lock: it fails if a page is not there, and then the texture is
/// left as it is and read at its next refresh, a frame later. A read
/// through the kernel costs about 0.6 ms a megabyte against 0.3 in place.
///
/// What changes for the picture: a texture that arrives while the runtime
/// is remapping may show a frame later. It never shows partly read.
fn no_wait() -> bool {
    static ON: crate::flip::Pair = crate::flip::Pair::new("VICTORY_TEXTURE_NO_WAIT");
    ON.get() == Some("1")
}

/// Since the last frame-rate line, with [`no_wait`]: refreshes that read
/// through the kernel because title memory was being remapped, and those
/// left for the next refresh because a page was not there.
pub(crate) static UNLOCKED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Since the last frame-rate line, of texture refreshes on every thread, in
/// microseconds: waiting for the texture's own lock (another thread is
/// refreshing the same texture), asking the write watch and the guards,
/// waiting for title memory to hold still (the lock the runtime takes
/// alone while it remaps, `crate::watch::title_memory_reading`), and the
/// refreshes in all.
///
/// While the world streams in, the frame line's "textures" is seconds a
/// frame in some runs and tenths in others for the same megabytes (2,566
/// ms for 110 MB in run 93, 314 ms for 132 MB in run 91; before these
/// counters, runs 81, 83, 85 and 94 each had lines of 1.5-2.9 s), and the
/// submitting thread is not running while it lasts (run 93: 240-310 ms of
/// its own time a frame beside 1,100-1,800 ms of "textures"): it waits.
/// These say for what.
pub(crate) static REFRESH_WAITS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// `f`'s result, its time added to [`REFRESH_WAITS`]' place `slot`.
fn waited<T>(slot: usize, f: impl FnOnce() -> T) -> T {
    let started = std::time::Instant::now();
    let out = f();
    REFRESH_WAITS[slot].fetch_add(started.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    out
}

/// Whether a staging buffer made for one upload of a texture picked by
/// index is filled without asking the write watch and the guards
/// (`VICTORY_TEXTURE_ONCE=1`; off by default: with them, as every other).
/// What they say is for the next refresh, and there is none: the pass lets
/// go of the staging state after recording the upload
/// (`crate::draw`'s pass over textures picked by index). Asking the guards
/// guards the texture's blocks (a change of protection, under the lock every
/// other refresh wants), and the title's next write there faults to open
/// them again.
fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_ONCE").is_some_and(|v| v == "1"))
}

thread_local! {
    /// The thread records the upload of a texture for the pass over
    /// textures picked by index, which lets go of its staging state at once
    /// ([`once`]).
    pub(crate) static PICKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// Room to read a texture's memory into: one buffer per submitting
    /// thread. Each texture kept its own, a second copy in memory of every
    /// write-watched or partly mapped texture ever sampled, for the run.
    static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Calls `f` with `size` bytes of this thread's read buffer (their
/// contents are whatever the last texture left).
fn with_read<R>(size: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    READ.with(|read| {
        let mut read = read.borrow_mut();
        if read.len() < size {
            read.resize(size, 0);
        }
        f(&mut read[..size])
    })
}

/// Whether a texture all of whose memory is mapped is read through the
/// kernel's copy instead of in place (`VICTORY_TEXTURE_COPY=1`; off by
/// default). Round 3 of the memory work read every such texture that way
/// to be safe from pages taken away under the read; it cost 0.6 ms per MB
/// against 0.3 in place (90-105 ms a frame while the world streams in, runs
/// 72 and 73, against 20-40), and both runs with it lost the graphics
/// device, for a reason not found. The read in place is safe now for
/// another reason (`crate::watch::title_memory_reading`); this switch is
/// left to tell whether the copy was what lost the device.
fn through_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_COPY").is_some_and(|v| v == "1"))
}

/// Calls `f` with the `size` bytes of title memory at `memory`, read
/// through the kernel into this thread's read buffer; pages that are not
/// there read as zeros. The bytes are those a read in place would see.
fn with_title<R>(memory: *const u8, size: usize, f: impl FnOnce(&[u8]) -> R) -> R {
    with_read(size, |read| {
        crate::host::copy_mapped(read.as_mut_ptr(), memory, size, false);
        f(read)
    })
}

/// Staging buffers alive (also those only a list still holds, and with
/// [`transient`] those in the pool) and their bytes.
static STAGED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Staging {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        // With [`transient`] the buffer goes back to the pool: now if no
        // list holds it, else at a sweep after those lists were reset (the
        // list whose reset drops this one may be one of them).
        let buf = self.buf.get_mut().unwrap_or_else(|e| e.into_inner()).take();
        if self.transient
            && let Some(buf) = buf
            && let Ok(buf) = Arc::try_unwrap(buf)
        {
            buf.fence.fetch_max(*self.last_upload.get_mut(), Relaxed);
            if buf.unheld() {
                give_back(buf);
            } else {
                ORPHANS.lock().unwrap_or_else(|e| e.into_inner()).push(buf);
            }
        }
    }
}

/// With [`transient`]: staging buffers whose texture let go of its staging
/// state while a list still held them, for the pool once none does.
static ORPHANS: Mutex<Vec<Buf>> = Mutex::new(Vec::new());

/// With [`transient`], at the sweep: the [`ORPHANS`] no list holds any more
/// go back to the pool.
fn pool_orphans() {
    let unheld: Vec<Buf> = {
        let mut orphans = ORPHANS.lock().unwrap_or_else(|e| e.into_inner());
        if orphans.is_empty() {
            return;
        }
        let (unheld, held) = std::mem::take(&mut *orphans).into_iter().partition(Buf::unheld);
        *orphans = held;
        unheld
    };
    for buf in unheld {
        give_back(buf);
    }
}

/// Whether a texture keeps its staging buffer only until its upload has run
/// (`VICTORY_TEXTURE_STAGING_TRANSIENT=1`; off by default: one buffer per
/// sampled texture, kept as long as the texture is staged, as before).
///
/// A staging buffer is system memory the size of all of its texture's
/// levels, and nearly every live host copy of a filled texture had one beside
/// it (1.5-3 GB of the runner's private bytes, `runs\perf\ram-budget.md`).
/// Its bytes are needed only from a detiling until the list that uploads them
/// has run: a change later is found against title memory and detiled again.
/// With the switch a texture is given a buffer when a list records a copy or
/// a check at a submission finds its memory changed, and the buffer goes back
/// to a pool once the upload ran (no list holds it, at the sweep:
/// [`release_uploaded`]); the pool keeps buffers by size class up to
/// [`pool_cap`] and the next texture of that class takes one. A copy
/// recorded again (a texture picked by index brought up to date, a level on
/// demand asked again) detiles the texture again into the buffer it is
/// given, at that list's check, and the next list that uploads it copies it
/// ([`Staging::record_copies`]; a texture picked by index stays due for
/// that list, [`copy_pending`]).
fn transient() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_TEXTURE_STAGING_TRANSIENT").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                pool_mb = pool_cap() >> 20,
                "texture staging buffers are kept only until their upload has run, then pooled (VICTORY_TEXTURE_STAGING_TRANSIENT=1)"
            );
        }
        on
    })
}

/// The most bytes the pool of [`transient`] staging buffers keeps
/// (`VICTORY_TEXTURE_STAGING_POOL_MB`, 256 by default; 0: none, each buffer
/// is let go once its upload ran).
fn pool_cap() -> u64 {
    static CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_STAGING_POOL_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(256)
            .saturating_mul(1 << 20)
    })
}

/// The size class of a staging buffer of `bytes`: a quarter step of its
/// power of two, at least 64 KiB (an upload buffer's own grain), so a
/// buffer made for one texture serves others of nearly its size and wastes
/// less than a quarter of itself.
fn class_of(bytes: u64) -> u64 {
    let bytes = bytes.max(64 << 10);
    let step = 1u64 << (63 - bytes.leading_zeros()).saturating_sub(2);
    bytes.div_ceil(step).saturating_mul(step)
}

/// Spare things of a size class each, the oldest let go first when they
/// take more than a cap ([`transient`] staging buffers).
struct Pool<T> {
    by_class: HashMap<u64, Vec<(u64, T)>>,
    /// Each one's class, by when it came (the oldest first).
    order: std::collections::BTreeMap<u64, u64>,
    bytes: u64,
    next: u64,
}

impl<T> Pool<T> {
    fn new() -> Self {
        Self {
            by_class: HashMap::new(),
            order: std::collections::BTreeMap::new(),
            bytes: 0,
            next: 0,
        }
    }

    /// The newest one of `class`.
    fn take(&mut self, class: u64) -> Option<T> {
        let spare = self.by_class.get_mut(&class)?;
        let (at, item) = spare.pop()?;
        if spare.is_empty() {
            self.by_class.remove(&class);
        }
        self.order.remove(&at);
        self.bytes -= class;
        Some(item)
    }

    /// Keeps `item` of `class` within `cap` bytes: what it lets go of to
    /// stay within (the oldest first; `item` itself if it is bigger than
    /// `cap`).
    fn put(&mut self, class: u64, item: T, cap: u64) -> Vec<T> {
        if class > cap {
            return vec![item];
        }
        let at = self.next;
        self.next += 1;
        self.by_class.entry(class).or_default().push((at, item));
        self.order.insert(at, class);
        self.bytes += class;
        let mut out = Vec::new();
        while self.bytes > cap {
            let Some((at, class)) = self.order.pop_first() else {
                break;
            };
            let spare = self.by_class.get_mut(&class).expect("a pooled item's class");
            // Each class's items are in the order they came.
            let (first, item) = spare.remove(0);
            debug_assert_eq!(first, at);
            if spare.is_empty() {
                self.by_class.remove(&class);
            }
            self.bytes -= class;
            out.push(item);
        }
        out
    }

    fn len(&self) -> usize {
        self.order.len()
    }
}

static POOL: Mutex<Option<Pool<Buf>>> = Mutex::new(None);

/// With [`transient`], since the start: buffers made, and their bytes;
/// taken from the pool, and their bytes; let go of by their texture once
/// the upload ran, and their bytes; pooled ones let go to keep the pool
/// within its cap; detilings into a buffer given to a texture for a copy
/// (what keeping no buffer costs); uploads that recorded no copy and left
/// it to a later list ([`Recorded::Later`]).
static TRANSIENT: [std::sync::atomic::AtomicU64; 9] = [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// A staging buffer of `bytes` bytes' class, from the pool if it has one.
fn take_buf(bytes: u64) -> Option<Buf> {
    use std::sync::atomic::Ordering::Relaxed;
    let class = class_of(bytes);
    let pooled = POOL.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|pool| pool.take(class));
    if let Some(buf) = pooled {
        TRANSIENT[2].fetch_add(1, Relaxed);
        TRANSIENT[3].fetch_add(class, Relaxed);
        return Some(buf);
    }
    let buf = Buf::new(class)?;
    TRANSIENT[0].fetch_add(1, Relaxed);
    TRANSIENT[1].fetch_add(class, Relaxed);
    Some(buf)
}

/// Puts `buf`, which no list holds, back in the pool; what the pool lets go
/// of to stay within [`pool_cap`] is retired.
fn give_back(buf: Buf) {
    let gone = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        pool.get_or_insert_with(Pool::new).put(buf.bytes, buf, pool_cap())
    };
    TRANSIENT[6].fetch_add(gone.len() as u64, std::sync::atomic::Ordering::Relaxed);
    drop(gone);
}

/// Whether a texture's staging buffer may go back to the pool
/// ([`transient`]): what it holds was uploaded by a list that has run (no
/// list holds it any more, `unheld`), nothing detiled since waits for a
/// list's upload (`dirty`), it does not wait for its first detiling
/// (`fresh`), and no list recorded a copy from it for `IDLE_FRAMES` frames
/// (`idle`: one copied every frame keeps its buffer).
fn releasable(uploaded: bool, dirty: bool, fresh: bool, unheld: bool, idle: bool) -> bool {
    uploaded && !dirty && !fresh && unheld && idle
}

/// With [`transient`], at the sweep: of `stagings` (the staging table's,
/// under its lock), the buffers whose upload has run go back to the pool
/// ([`releasable`]).
fn release_uploaded<'a>(stagings: impl Iterator<Item = &'a Arc<Staging>>) {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 2;
    let (mut went, mut bytes) = (0u64, 0u64);
    for s in stagings.filter(|s| s.transient) {
        if let Some(buf) = s.release_if_uploaded(IDLE_FRAMES) {
            went += 1;
            bytes += buf.bytes;
            give_back(buf);
        }
    }
    if went > 0 {
        TRANSIENT[4].fetch_add(went, Relaxed);
        TRANSIENT[5].fetch_add(bytes, Relaxed);
        crate::host::trail(|| format!("sweep: {went} staging buffers back to the pool after their upload ({} KiB)", bytes >> 10));
    }
}

/// With [`transient`], for the frame line's texture refreshes: the staging
/// buffers alive and pooled, and what the pool and the releases did since
/// the start; empty without the switch.
pub(crate) fn staging_note() -> String {
    use std::sync::atomic::Ordering::Relaxed;
    if !transient() {
        return String::new();
    }
    let mb = |b: u64| b >> 20;
    let (pooled, pooled_bytes) = POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or((0, 0), |pool| (pool.len(), pool.bytes));
    let t = TRANSIENT.each_ref().map(|c| c.load(Relaxed));
    format!(
        "; staging (transient): {} buffers {} MB live, of them {pooled} {} MB pooled (cap {} MB); since the start {} made {} MB, \
         {} reused {} MB, {} released after their upload {} MB, {} let go by the pool's cap, {} detiled again into a buffer given for a copy, \
         {} uploads left to a later list",
        STAGED[0].load(Relaxed),
        mb(STAGED[1].load(Relaxed)),
        mb(pooled_bytes),
        mb(pool_cap()),
        t[0],
        mb(t[1]),
        t[2],
        mb(t[3]),
        t[4],
        mb(t[5]),
        t[6],
        t[7],
        t[8],
    )
}

/// For the profile's memory sample (`VICTORY_PROFILE_SAMPLES`): the bytes
/// of the staging buffers alive.
pub(crate) fn staging_bytes() -> u64 {
    STAGED[1].load(std::sync::atomic::Ordering::Relaxed)
}

/// For the memory line: staging buffers alive, their bytes, and how many
/// of them their texture still has (the others wait for a list's reset, or
/// with [`transient`] lie in the pool).
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 3] {
    use std::sync::atomic::Ordering::Relaxed;
    // (The table's lock, then each buffer's: the order everywhere.)
    let kept = STAGING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map_or(0, |map| {
            map.values()
                .filter(|s| s.buf.lock().unwrap_or_else(|e| e.into_inner()).is_some())
                .count() as u64
        });
    [STAGED[0].load(Relaxed), STAGED[1].load(Relaxed), kept]
}

/// Staging buffers let go as idle since the start, and their bytes.
static LET_GO: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// For the memory line: staging buffers let go as idle, and their bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn let_go() -> [u64; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    [LET_GO[0].load(Relaxed), LET_GO[1].load(Relaxed)]
}

/// Lets go of the staging buffers no list recorded a use of for `IDLE_MS`
/// and `IDLE_FRAMES` frames and no list holds: a staging buffer is system memory the size of its
/// texture, kept so that an unchanged texture is not uploaded again, and
/// those of textures no longer sampled were kept for the run (2,246 of
/// them, 2.1 GB, at the end of run 24). The next list that samples such a
/// texture makes a staging buffer anew, detiles and uploads once more.
/// Called before a submission starts, by the thread that submits. What it
/// lets go of goes to `gone`, to be dropped as is (no list names it, so no
/// running submission does; and a staging buffer itself waits for the
/// submissions running when it goes, [`Buf`]): at once when the sweep runs
/// whole, under the lock as before slices. With [`transient`] it also gives
/// back to the pool the buffers whose upload has run.
pub(crate) fn let_go_idle(gone: &mut crate::host::LetGo) {
    use std::sync::atomic::Ordering::Relaxed;
    const IDLE_FRAMES: u64 = 4;
    if transient() {
        pool_orphans();
    }
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let Some(map) = map.as_mut() else {
        return;
    };
    if map.is_empty() {
        return;
    }
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    // A list that recorded an upload or a check holds the staging until it
    // is reset: one only the table holds is named by no list, recorded or
    // running.
    let idle: Vec<usize> = map
        .iter()
        .filter(|(_, s)| {
            Arc::strong_count(s) == 1
                && crate::host::long_ago(s.used.load(Relaxed), idle_ms, IDLE_FRAMES)
        })
        .map(|(key, _)| *key)
        .collect();
    let (mut went, mut bytes) = (0u64, 0u64);
    for s in idle.into_iter().filter_map(|key| map.remove(&key)) {
        LET_GO[0].fetch_add(1, Relaxed);
        LET_GO[1].fetch_add(s.bytes, Relaxed);
        went += 1;
        bytes += s.bytes;
        gone.push_dropped(s);
    }
    if went > 0 {
        crate::host::trail(|| format!("sweep: {went} staging buffers of textures let go ({} KiB), held by no list", bytes >> 10));
    }
    if transient() {
        release_uploaded(map.values());
    }
}

/// Textures the host GPU wrote by a copy or resolve: their contents are on
/// the host, not in title memory, which stays as it was.
static HOST_WRITTEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// A host copy or resolve writes `resource`.
pub(crate) fn host_written(resource: This) {
    let new = {
        let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        set.get_or_insert_with(Default::default).insert(resource as usize)
    };
    // Diagnostics (run 177): where its title memory is, for placed views
    // the title reads that memory through ([`placed_in_host_written`]).
    if new && let Some((at, desc)) = crate::resource::texture_memory(resource) {
        // SAFETY: 56 bytes in the description's layout.
        let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
        let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
        if !at.is_null() && bytes > 0 {
            HOST_WRITTEN_AT
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(at as u64, (bytes, words[4], words[6], words[7], words[8], resource as usize));
        }
    }
}

/// The title memory of textures the host copied into, by address: bytes
/// and description words 4, 6, 7 and 8 (width, height, layers and levels,
/// format).
static HOST_WRITTEN_AT: Mutex<std::collections::BTreeMap<u64, (u64, u32, u32, u32, u32, usize)>> =
    Mutex::new(std::collections::BTreeMap::new());

/// A texture the title placed at `address` (description words `desc`) was
/// made: says so once if it lies in the memory of a texture the host
/// copied into ([`host_written`]). Its own host copy is filled from title
/// memory, which the copy never reached: the copied texels are on the
/// card in the other texture, where no shader that reads this view finds
/// them.
pub(crate) fn placed_in_host_written(address: u64, desc: &[u32; 14]) {
    let hit = {
        let map = HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner());
        map.range(..=address)
            .next_back()
            .filter(|(at, d)| address < **at + d.0)
            .map(|(at, d)| (*at, *d))
    };
    if let Some((at, (bytes, width, height, layers_levels, format, _))) = hit {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 200 {
            tracing::warn!(
                view_address = format_args!("{address:#x}"),
                view_offset = address - at,
                view_width = desc[4],
                view_height = desc[6],
                view_layers_levels = format_args!("{:#x}", desc[7]),
                view_format = desc[8],
                copied_address = format_args!("{at:#x}"),
                copied_bytes = bytes,
                copied_width = width,
                copied_height = height,
                copied_layers_levels = format_args!("{layers_levels:#x}"),
                copied_format = format,
                "HERO LEAD: a placed texture view lies in the memory of a texture the host copied into: it is filled from title memory, which the copy never reached"
            );
        }
    }
}

/// Whether the host GPU wrote `resource` by a copy or resolve
/// ([`host_written`]): its host copy cannot be made again from title
/// memory.
pub(crate) fn host_wrote(resource: usize) -> bool {
    HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&resource))
}

/// The title released a resource.
pub(crate) fn forget(resource: usize) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&resource);
    }
    let mut set = HOST_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(set) = set.as_mut() {
        set.remove(&resource);
    }
    // Its memory may be given to other textures now (run 177: most of the
    // "HERO LEAD" lines were views in memory a released texture had held).
    HOST_WRITTEN_AT.lock().unwrap_or_else(|e| e.into_inner()).retain(|_, d| d.5 != resource);
    let mut through = THROUGH.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(through) = through.as_mut() {
        through.remove(&resource);
    }
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Records, once per list, the copy of `resource`'s staging buffer into
/// its host copy, and the detiling that refreshes the staging buffer at
/// submission if the memory changed. Leaves the texture in the copy
/// destination state.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    let host_written = HOST_WRITTEN
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|set| set.contains(&(resource as usize)));
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    if host_written || !list.first_use(resource as usize) {
        return;
    }
    // The pass over textures picked by index, for a texture without a
    // staging state: nothing to upload if its host copy holds these bytes.
    if PICKING.get() && same() && !is_staged(resource) && unchanged(resource) {
        let n = SAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n == 1 || n.is_multiple_of(10_000) {
            tracing::info!(
                so_far = n,
                "uploads of textures picked by index left out: the host copy holds those bytes (VICTORY_TEXTURE_SAME=1)"
            );
        }
        return;
    }
    let Some(staging) = staging(resource, texture) else {
        return;
    };
    use std::sync::atomic::Ordering::Relaxed;
    staging.used.store(crate::host::moment(), Relaxed);
    if !PICKING.get() {
        // Sampled through a table: kept and checked from here on.
        staging.once.store(false, Relaxed);
    }
    // Levels on demand: a view of the texture now starts at another
    // level, or another texture now starts inside its span, with no byte
    // of its memory changed. The levels are judged anew: detiled again at
    // this list's submission, and uploaded by this list.
    if staging.on_demand && staging.judgement() != staging.judged() {
        staging.forget();
        staging.dirty.store(true, Relaxed);
    }
    // Levels on demand: the last upload left a held level without tiles
    // (the card had no room, or no heap could be had), and nothing else
    // would record an upload again until the title writes the texture: the
    // level stayed empty, the texture blurred or black. Asked again about
    // once a second, and only when the card has room; the staging buffer
    // still holds the level's bytes (with [`transient`] it may have gone
    // back to the pool: the levels are then detiled again into the buffer
    // the texture is given, and the next list uploads them).
    if staging.on_demand {
        let since = staging.waiting.load(Relaxed);
        if since != 0 && crate::host::long_ago(since, 1000, 2) {
            if crate::resource::room_lately() {
                staging.waiting.store(0, Relaxed);
                staging.dirty.store(true, Relaxed);
                crate::tiles::STATS[10].fetch_add(1, Relaxed);
            } else {
                staging.waiting.store(crate::host::moment().max(1), Relaxed);
            }
        }
    }
    // The host texture has the staging buffer's contents: only the check
    // at submission is left, and what it finds changed the next list that
    // samples the texture uploads (a frame late where every list of the
    // frame was recorded before that check).
    if staging.uploaded.load(Relaxed) && !staging.dirty.load(Relaxed) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexSkipped, 1);
        list.defer_texture(staging, false);
        return;
    }
    // A texture cut from a block (`crate::texture_pool`) may lie where
    // another did: what the card holds of that one is done with before the
    // first upload into this one, and before its transition below (a use).
    if !staging.uploaded.load(Relaxed) && crate::texture_pool::in_block(texture) {
        let aliasing = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    // SAFETY: a borrowed pointer the barrier never releases.
                    pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
                }),
            },
        };
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[aliasing]) };
    }
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    match staging.record_copies(list, texture, resource) {
        Recorded::Copies => {
            staging.later.store(false, Relaxed);
            // Every subresource is written by this list: a host copy into
            // the texture later needs no zeros ([`first_written`]).
            crate::texture_pool::first_write(texture);
        }
        Recorded::Later => {
            // Only the check, which fills the buffer the texture was given;
            // the next list that uploads the texture copies it (the pass
            // over textures picked by index keeps their slots due for one:
            // [`copy_pending`]).
            staging.later.store(true, Relaxed);
            TRANSIENT[8].fetch_add(1, Relaxed);
            list.defer_texture(staging, false);
            return;
        }
        Recorded::Nothing => return,
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::TexUploads, 1);
    // When the list is submitted, its copy takes what the staging buffer
    // holds by then: noted by the texture step (`uploads`) before its check,
    // which sets `dirty` again if it detiles (the next list then uploads once
    // more), and the submission's fence noted so that a later detiling waits
    // for the copy to have run (`Staging::detile`).
    // Levels on demand (`crate::tiles`): after the check below, in the
    // order of this list's steps, and before the list is executed, the
    // levels the staging buffer holds as the title's are given the card's
    // memory if they have none. The list copies every level, so each one
    // given here is written whole by it (the rest are copies into no
    // memory, which the card drops).
    let on_demand = if staging.on_demand { crate::resource::host_tiled(resource) } else { None };
    let Some(on_demand) = on_demand else {
        // Then what the list's copy puts in the host copy is known
        // ([`unchanged`]).
        let noted = same()
            .then(|| crate::resource::host_copy(resource))
            .flatten()
            .map(|copy| (staging.clone(), copy));
        // Once per submission however many of its lists sample the texture.
        list.defer_texture(staging, true);
        if let Some((staging, copy)) = noted {
            list.defer_host(move || staging.note_held(&copy));
        }
        return;
    };
    list.defer_texture(staging.clone(), true);
    list.defer_host(move || {
        let waiting = on_demand.uploaded(staging.detiled.load(Relaxed), staging.memory as u64);
        staging.waiting.store(if waiting { crate::host::moment().max(1) } else { 0 }, Relaxed);
        // A texture picked by index has no staging buffer after this
        // upload: its slots are made due again by the texture pass.
        if waiting {
            crate::descriptor::host::levels_waiting(staging.memory as u64);
        }
    });
}

/// Levels on demand: the levels a list claimed for its zeros
/// (`crate::tiles::Tiled::claim`), held by the list's submission-time
/// step. A list that is reset without having been submitted gives the
/// claim up, or the next list that writes those levels would send no zeros
/// and what its write does not cover would be what the tiles' last owner
/// left.
struct Claimed {
    on_demand: crate::resource::OnDemand,
    fresh: u32,
    ran: std::sync::atomic::AtomicBool,
}

impl Drop for Claimed {
    fn drop(&mut self) {
        if self.fresh != 0
            && !*self.ran.get_mut()
            && let Some(tiled) = self.on_demand.tiled()
        {
            tiled.unclaim(self.fresh);
        }
    }
}

/// Levels on demand (`crate::tiles`): the card itself is about to write
/// title texture `resource` in `list` (a copy or a resolve into level
/// `level`, or into every level). If its host copy `texture` was made
/// without memory, the levels written get theirs when the list is
/// submitted, ahead of it on the queue; the ones that have none yet are
/// first written with zeros, here in the same list, so that what the
/// card's write does not cover is not what the tiles' last owner left.
/// To be called before the list is brought to the state of that write.
/// shortcut: two lists that write the same new level, submitted in the
/// other order than recorded, lose the first one's write under the zeros
/// of the other; so does an upload recorded and not yet submitted.
pub(crate) fn card_writes(list: &HostList, resource: This, texture: &ID3D12Resource, level: Option<u32>) {
    // (`None`: a copy of the whole resource, which writes every texel.)
    if crate::texture_pool::first_write(texture) {
        first_written(list, resource, texture, level.is_some());
    }
    let Some(on_demand) = crate::resource::host_tiled(resource) else {
        return;
    };
    let Some(tiled) = on_demand.tiled() else {
        return;
    };
    let levels = match level {
        Some(level) => tiled.with_level(level),
        None => tiled.every_level(),
    };
    if levels == 0 {
        return;
    }
    let fresh = tiled.claim(levels);
    // Of them, the ones whose zeros could not be recorded.
    let mut unzeroed = if host().is_none() { fresh } else { 0 };
    if fresh != 0
        && let Some(host) = host()
    {
        const COPY_DEST: u32 = 0x400;
        crate::list::ensure_state(list, resource, COPY_DEST);
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        for level in (0..u32::from(d.MipLevels).min(32)).filter(|&level| fresh & (1u32 << level) != 0) {
            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut bytes = 0u64;
            // SAFETY: outputs for one subresource.
            unsafe {
                host.device
                    .GetCopyableFootprints(&d, level, 1, 0, Some(&mut footprint), None, None, Some(&mut bytes));
            }
            let Some((buffer, data)) = list.upload(host, bytes) else {
                // (With the levels that share its tiles, if it is one of
                // the small ones.)
                unzeroed |= tiled.with_level(level);
                continue;
            };
            // SAFETY: a mapped buffer of `bytes` bytes, just made.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: level },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above; the list holds the buffer until it is reset.
                pResource: unsafe { std::mem::transmute_copy(&buffer) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
    }
    // A new level without its zeros gets no tiles from this list: the
    // card's write into it is lost (the level reads zeros, as before the
    // write), which is less wrong than a part write over what the tiles'
    // last owner left. The next list that writes it claims it again.
    unzeroed &= fresh;
    if unzeroed != 0 {
        tiled.unclaim(unzeroed);
    }
    let levels = levels & !unzeroed;
    if levels == 0 {
        return;
    }
    let memory = crate::resource::texture_memory(resource).map_or(0, |(memory, _)| memory as u64);
    let step = Claimed {
        on_demand,
        fresh: fresh & !unzeroed,
        ran: std::sync::atomic::AtomicBool::new(false),
    };
    list.defer_host(move || {
        step.ran.store(true, std::sync::atomic::Ordering::Relaxed);
        step.on_demand.card_wrote(levels, memory);
    });
}

/// A texture cut from a block of `crate::texture_pool` that a host copy or
/// resolve writes before any upload did (after which [`upload`] records
/// none): the aliasing barrier that makes it the block's occupant, and
/// zeros in every subresource, so that what the write does not cover reads
/// as in a texture of its own, not as what the piece held before (without
/// `zero`, the barrier only: the write covers every texel). A subresource
/// whose zeros find no upload buffer keeps those bytes (logged).
fn first_written(list: &HostList, resource: This, texture: &ID3D12Resource, zero: bool) {
    let aliasing = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                pResourceBefore: std::mem::ManuallyDrop::new(None),
                // SAFETY: a borrowed pointer the barrier never releases.
                pResourceAfter: unsafe { std::mem::transmute_copy(texture) },
            }),
        },
    };
    // SAFETY: an open list and a live texture.
    unsafe { list.list.ResourceBarrier(&[aliasing]) };
    let Some(host) = host().filter(|_| zero) else {
        return;
    };
    const COPY_DEST: u32 = 0x400;
    crate::list::ensure_state(list, resource, COPY_DEST);
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    // One plane (pooled textures have one).
    let layers = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(d.DepthOrArraySize)
    };
    let mut unzeroed = 0u32;
    for sub in 0..u32::from(d.MipLevels).max(1) * layers {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut bytes = 0u64;
        // SAFETY: outputs for one subresource.
        unsafe {
            host.device.GetCopyableFootprints(
                &d,
                sub,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut bytes),
            );
        }
        let Some((buffer, data)) = list.upload(host, bytes) else {
            unzeroed += 1;
            continue;
        };
        // SAFETY: a mapped buffer of `bytes` bytes, just made.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above; the list holds the buffer until it is reset.
            pResource: unsafe { std::mem::transmute_copy(&buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
    if unzeroed != 0 {
        tracing::warn!(
            subresources = unzeroed,
            "a pooled texture written by a host copy first could not be zeroed: no upload buffer (VICTORY_TEXTURE_POOL)"
        );
    }
}

/// What [`Staging::record_copies`] recorded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Recorded {
    /// The copies: the list uploads the staging buffer.
    Copies,
    /// Nothing yet ([`transient`]): the buffer the texture was given is
    /// filled by the list's check at submission, and the next list that
    /// uploads the texture copies it: one that samples it through a table,
    /// or the pass over textures picked by index, which keeps the
    /// texture's slots due until then ([`copy_pending`]).
    Later,
    /// Nothing: no staging buffer could be had.
    Nothing,
}

impl Staging {
    /// Records into `list` the copy of every subresource of the staging
    /// buffer into `texture` (the host copy of title texture `resource`, in
    /// the copy destination state). With [`transient`] a texture without a
    /// buffer is given one first, to be filled at the list's submission
    /// ([`Staging::fresh`]), and the list holds the buffer itself until it
    /// is reset: the texture may let go of it before.
    ///
    /// With [`transient`], a texture detiled before whose buffer holds none
    /// of its bytes (it went back to the pool after its upload ran) records
    /// no copy: its host copy holds the bytes of that detiling, and a copy
    /// from a buffer the check at submission could not fill (memory partly
    /// mapped and all zero, or a read that failed while the runtime
    /// remapped it) would put zeros in their place, where without the
    /// switch the last bytes were copied again. The list's check fills the
    /// buffer, and the next list that uploads the texture copies it
    /// ([`Recorded::Later`]).
    fn record_copies(&self, list: &HostList, texture: &ID3D12Resource, resource: This) -> Recorded {
        use std::sync::atomic::Ordering::Relaxed;
        // Held while the copies are recorded: the sweep gives back a buffer
        // only when this is all that holds it.
        let buf = {
            let mut slot = self.buf.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() && self.transient {
                let Some(buf) = take_buf(self.bytes) else {
                    once!("no staging buffer for a texture upload: the texture is uploaded at a later use");
                    return Recorded::Nothing;
                };
                crate::host::name(buf.buffer(), || format!("staging of {resource:?} (pooled)"));
                *slot = Some(Arc::new(buf));
                self.fresh.store(true, Relaxed);
            }
            let Some(buf) = slot.clone() else {
                return Recorded::Nothing;
            };
            if self.transient && self.fresh.load(Relaxed) && self.detiles.load(Relaxed) != 0 {
                return Recorded::Later;
            }
            buf
        };
        for (i, footprint) in self.footprints.iter().enumerate() {
            #[cfg(feature = "diagnostics")]
            {
                let f = footprint.Footprint;
                // SAFETY: live resources.
                let (gpu, d) = unsafe { (buf.buffer().GetGPUVirtualAddress(), texture.GetDesc()) };
                let len = u64::from(f.RowPitch) * u64::from(self.rows[i]) * u64::from(f.Depth);
                list.note_read(
                    || {
                        format!(
                            "texture upload {resource:?} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                            d.Width,
                            d.Height,
                            d.DepthOrArraySize,
                            d.Format.0,
                            d.MipLevels,
                            self.footprints.len(),
                            buf.bytes,
                            footprint.Offset,
                            f.Width,
                            f.Height,
                            f.Depth,
                            f.RowPitch
                        )
                    },
                    (gpu + buf.base + footprint.Offset, len),
                );
            }
            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed pointer the location never releases.
                pResource: unsafe { std::mem::transmute_copy(texture) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: i as u32,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: as above.
                pResource: unsafe { std::mem::transmute_copy(buf.buffer()) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: buf.base + footprint.Offset,
                        ..*footprint
                    },
                },
            };
            // SAFETY: an open list; both locations name live resources.
            unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
        }
        // Without the switch the buffer lives as long as the staging state,
        // which the list's texture step holds until the list is reset; with
        // it the texture may give the buffer back before, so the list holds
        // it too.
        if self.transient {
            list.keep(buf.buffer());
            self.copied.store(crate::host::moment(), Relaxed);
        }
        Recorded::Copies
    }

    /// With [`transient`]: takes the staging buffer away if it may go back
    /// to the pool ([`releasable`]), with the fence of the last submission
    /// sent ahead that read it. Passed over if another thread has the
    /// buffer's lock.
    fn release_if_uploaded(&self, idle_frames: u64) -> Option<Buf> {
        use std::sync::atomic::Ordering::Relaxed;
        let mut slot = self.buf.try_lock().ok()?;
        let buf = slot.as_ref()?;
        // (Asked under the lock, under which a detiling or a copy being
        // recorded takes its own reference first: one that holds the
        // buffer shows in the count.)
        let unheld = Arc::strong_count(buf) == 1 && buf.unheld();
        if !releasable(
            self.uploaded.load(Relaxed),
            self.dirty.load(Relaxed),
            self.fresh.load(Relaxed),
            unheld,
            crate::host::long_ago(self.copied.load(Relaxed), 0, idle_frames),
        ) {
            return None;
        }
        let buf = Arc::try_unwrap(slot.take()?).ok()?;
        buf.fence.fetch_max(self.last_upload.swap(0, Relaxed), Relaxed);
        Some(buf)
    }

    /// Note an upload before checking for changed contents. The check sets
    /// dirty again if the next list needs to upload newer contents.
    pub(crate) fn note_upload(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        self.uploaded.store(true, Relaxed);
        self.dirty.store(false, Relaxed);
    }

    pub(crate) fn note_upload_fence(&self, fence: u64) {
        self.last_upload.store(fence, std::sync::atomic::Ordering::Relaxed);
    }

    /// Levels on demand: what says, beside the mapping, which levels of
    /// the texture are the title's now. First the lowest level a live view
    /// of the texture shows (0 without a view that says): the title's own
    /// word, a texture it holds from level 3 on is viewed from level 3.
    /// Then where the next texture placed in this one's span starts, in
    /// bytes from this one's start (`u64::MAX`: none does): for a texture
    /// viewed from level 0 although a neighbour sits where its large
    /// levels would be.
    fn judgement(&self) -> [u64; 2] {
        let start = self.memory as u64;
        let first = crate::descriptor::host::first_viewed(start).map_or(0, u64::from);
        let limit = crate::descriptor::host::next_start(start, start.saturating_add(self.layout.size))
            .map_or(u64::MAX, |at| at - start);
        [first, limit]
    }

    /// What [`Staging::judgement`] said at the last detiling.
    fn judged(&self) -> [u64; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [self.judged[0].load(Relaxed), self.judged[1].load(Relaxed)]
    }

    /// Levels on demand: the levels that are the title's now, a bit a
    /// level. A level is held if no view starts above it, its bytes end
    /// before the next texture starts, and all of them are mapped. The
    /// smallest levels share the texture's first block and are held
    /// together. (The console layout: that block first, then the other
    /// levels from the smallest up, level 0 last:
    /// `victory_formats::texture::Layout`.) A texture this cannot tell
    /// the levels of holds them all.
    fn held_levels(&self, [first, limit]: [u64; 2]) -> u32 {
        let mips = &self.layout.mips;
        if self.layout.slices != 1 || mips.is_empty() || mips.len() > 32 {
            return u32::MAX;
        }
        let tail = (self.layout.first_tail_mip as usize).min(mips.len());
        let there = |from: u64, to: u64| {
            to > from && to <= limit && committed(self.memory.wrapping_add(from as usize), (to - from) as usize)
        };
        let mut held = 0u32;
        for (level, m) in mips.iter().enumerate().take(tail) {
            let end = if level == 0 { self.layout.slice_size } else { mips[level - 1].offset };
            if level as u64 >= first && there(m.offset, end) {
                held |= 1 << level;
            }
        }
        if tail < mips.len() && there(0, self.layout.alignment.min(self.layout.slice_size)) {
            for level in tail..mips.len() {
                if level as u64 >= first {
                    held |= 1 << level;
                }
            }
        }
        held
    }

    /// The host copy `copy` holds what the staging buffer was filled from
    /// ([`unchanged`]): called after the check at a submission whose list
    /// copies the staging buffer into it.
    fn note_held(&self, copy: &crate::resource::HostTexture) {
        let hash = self.last.lock().unwrap_or_else(|e| e.into_inner()).held;
        *copy.holds.lock().unwrap_or_else(|e| e.into_inner()) = hash;
    }

    /// Forgets what the staging buffer was last filled from: the next
    /// refresh detiles anew, whether the memory changed or not.
    fn forget(&self) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Last::default();
    }

    /// The title memory the texture is read from: address and bytes.
    pub(crate) fn memory(&self) -> (usize, usize) {
        (self.memory as usize, self.layout.size as usize)
    }

    /// Detiles the texture memory into the staging buffer if it changed
    /// since the last time (and is mapped at all).
    pub(crate) fn refresh(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        // Counted with the host copies' refreshes (the profiler's split).
        let _refresh = crate::profile::refresh_scope();
        let read = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
    }

    /// Returns the bytes of title memory read.
    fn refresh_now(&self) -> usize {
        let whole = std::time::Instant::now();
        let read = self.refresh_timed();
        REFRESH_WAITS[3].fetch_add(whole.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        read
    }

    fn refresh_timed(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let mut last = waited(0, || self.last.lock().unwrap_or_else(|e| e.into_inner()));
        // A buffer given for a copy ([`transient`]) holds another texture's
        // bytes, or none: detiled whatever the write watch, the guards and
        // the hash say, and zeros if the memory cannot be read now (what a
        // new buffer held before the switch). Cleared only once the buffer
        // is filled: a list that records an upload meanwhile goes by it
        // ([`Staging::record_copies`]).
        let fresh = self.fresh.load(Relaxed);
        if fresh {
            // (When a partly mapped texture was last read stays: it is not
            // polled more often for having been given a buffer.)
            *last = Last {
                partial: last.partial,
                ..Last::default()
            };
        }
        let detiles = self.detiles.load(Relaxed);
        let read = self.refresh_locked(&mut last);
        if fresh {
            if self.detiles.load(Relaxed) != detiles {
                TRANSIENT[7].fetch_add(1, Relaxed);
                self.fresh.store(false, Relaxed);
            } else if detiles == 0 && self.zero() {
                self.fresh.store(false, Relaxed);
            }
            // Else detiled before (no list copies this buffer until it is
            // filled, and the host copy keeps the bytes it has), or no
            // buffer to be had: tried again at the next check, which starts
            // anew as this one did.
        }
        read
    }

    /// Zeros the staging buffer (with [`transient`], one given for the
    /// first copy of a texture never detiled that nothing could be detiled
    /// into: what a new buffer held before the switch). Called under
    /// `last`'s lock. False if the texture has no buffer.
    fn zero(&self) -> bool {
        // Held while it is written, as in [`Staging::detile`].
        let buf = self.buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let Some(buf) = buf else {
            return false;
        };
        buf.wait_to_write(self);
        // SAFETY: the mapped buffer, `buf.bytes` long; no submission reads
        // it any more (waited for above).
        unsafe { std::ptr::write_bytes(buf.data, 0, buf.bytes as usize) };
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// [`Staging::refresh_timed`] with the texture's lock held.
    fn refresh_locked(&self, last: &mut Last) -> usize {
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        // Made for one upload ([`once`]): filled once, from what the memory
        // holds, and not asked about again.
        let alone = self.once.load(std::sync::atomic::Ordering::Relaxed);
        if alone && last.hash.is_some() {
            return 0;
        }
        if !alone
            && let Some((now, written)) =
                waited(1, || crate::watch::writes(start, start + size, false))
        {
            if last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked) {
                return 0;
            }
            last.asked = now;
            // What it holds now is known by no hash.
            last.held = None;
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page.
            with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                self.detile(read);
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, size as u64);
            return size;
        }
        // View memory has no write-watch, and a texture there was hashed
        // whole at every check: 229 MB and 51 ms a frame in a gameplay view
        // where two textures change (run 58). It is guarded now, as buffers
        // in view memory are: a write of the title's to it faults once and
        // is noted, a mapping is announced by the runtime. While neither
        // happened the texture is as it was. What the guard cannot see (a
        // write to the same pages through another view) is still found by
        // the hash, which is taken every eighth frame only, by the
        // texture's address, so the textures take turns; and at every check
        // for a while once it found the texture changed ([`guard_skips`]).
        // `VICTORY_TEXTURE_GUARD=0` hashes at every check, as before.
        static GUARD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !alone
            && *GUARD
                .get_or_init(|| std::env::var_os("VICTORY_TEXTURE_GUARD").is_none_or(|v| v != "0"))
            && let Some((now, written)) =
                waited(1, || crate::watch::writes(start, start + size, true))
        {
            let unwritten = last.asked != 0 && written.iter().all(|&(_, w)| w <= last.asked);
            last.asked = now;
            if guard_skips(unwritten, crate::frame::current(), start, last.changed) {
                return 0;
            }
        }
        // From here on the texture's memory may be read in place. Held until
        // that read is done: the runtime changes what title memory shows
        // only with this taken alone, so what `committed` finds mapped
        // stays mapped (see `crate::watch::title_memory_reading`; without
        // it a page went between the asking and the read in runs 71 and
        // 74).
        let _mapped = if no_wait() {
            // Not waited for: while the runtime remaps title memory the
            // texture is read through the kernel instead, whole or not at
            // all (`no_wait`).
            let Some(mapped) = crate::watch::title_memory_reading_now(self.memory as usize, size) else {
                return self.refresh_unlocked(last, size);
            };
            mapped
        } else {
            // Only the texture's own addresses: remaps elsewhere go on.
            waited(2, || crate::watch::title_memory_reading_range(self.memory as usize, size))
        };
        if !committed(self.memory, size) {
            // Partly mapped: a streamed texture with only some mips' pages
            // there, the usual state while it streams in. Read through the
            // copy, which stands unmapped pages (zeros), and upload what
            // is there; nothing at all mapped changes nothing.
            // Poll at most every 30 frames: the guard conservatively reports
            // unprotectable sparse blocks even without an actual write.
            let frame = crate::frame::current();
            if !last.partial_due(frame) {
                return 0;
            }
            last.partial = Some(frame);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
            let changed = with_read(size, |read| {
                crate::host::copy_mapped(read.as_mut_ptr(), self.memory, size, false);
                if !last.accept_partial(read) {
                    return false;
                }
                self.detile(read);
                true
            });
            #[cfg(feature = "diagnostics")]
            if changed {
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
            }
            let _ = changed;
            return size;
        }
        // Fully mapped again: a later partial mapping starts a new poll.
        last.partial = None;
        if through_copy() {
            return with_title(self.memory, size, |tiled| {
                let hash = content_hash(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
                if !last.rehash(hash) {
                    return size;
                }
                self.detile(tiled);
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
                size
            });
        }
        // SAFETY: committed memory of the texture's layout size, and it
        // stays committed while `_mapped` is held.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = content_hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if !last.rehash(hash) {
            return size;
        }
        self.detile(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        size
    }

    /// A refresh while the runtime is remapping title memory ([`no_wait`]):
    /// the texture is read through the kernel, which fails if a page of it
    /// is not there at this moment. Then nothing is taken and the texture
    /// is looked at again at its next refresh (what was learnt about writes
    /// to it is forgotten, so that it is not passed over as unwritten).
    /// Returns the bytes read.
    fn refresh_unlocked(&self, last: &mut Last, size: usize) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let read = with_read(size, |read| {
            if !crate::host::read_into(self.memory, read) {
                return false;
            }
            if last.rehash(content_hash(read)) {
                self.detile(read);
            }
            true
        });
        if read {
            UNLOCKED[0].fetch_add(1, Relaxed);
            size
        } else {
            UNLOCKED[1].fetch_add(1, Relaxed);
            last.asked = 0;
            0
        }
    }

    /// Detiles `tiled` (the texture's memory) into the staging buffer (with
    /// [`transient`], one given to it now if it has none).
    fn detile(&self, tiled: &[u8]) {
        // Held while it is written (the sweep gives back only a buffer
        // nothing else holds), and not under the buffer's lock: a list
        // recording an upload of the texture meanwhile does not wait.
        let buf = {
            let mut slot = self.buf.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() && self.transient {
                match take_buf(self.bytes) {
                    Some(buf) => *slot = Some(Arc::new(buf)),
                    None => {
                        once!("no staging buffer to detile a texture into: it is detiled at a later check");
                        // Not taken as detiled: the next refresh starts anew.
                        self.fresh.store(true, std::sync::atomic::Ordering::Relaxed);
                        return;
                    }
                }
            }
            let Some(buf) = slot.clone() else {
                return;
            };
            buf
        };
        // A submission sent ahead may still copy from the staging buffer.
        buf.wait_to_write(self);
        self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
        self.detiles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Levels on demand: which levels are the title's is judged here,
        // once, for what the staging buffer holds from now on. The step
        // that gives levels their memory goes by this answer.
        let held = if self.on_demand {
            use std::sync::atomic::Ordering::Relaxed;
            let judged = self.judgement();
            self.judged[0].store(judged[0], Relaxed);
            self.judged[1].store(judged[1], Relaxed);
            let held = self.held_levels(judged);
            self.detiled.store(held, Relaxed);
            held
        } else {
            u32::MAX
        };
        #[cfg(feature = "diagnostics")]
        if tiled.iter().all(|&b| b == 0) {
            tracing::debug!(
                memory = ?self.memory,
                desc = ?self.layout.desc,
                "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
            );
        }
        for (i, (footprint, &rows)) in self.footprints.iter().zip(&self.rows).enumerate() {
            let (mip, slice) = (i as u32 % self.mips, i as u32 / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.bytes);
            // SAFETY: the subresource's part of the mapped staging buffer, cut
            // to its end ([`subresource_len`]).
            let linear = unsafe {
                std::slice::from_raw_parts_mut(buf.data.add(footprint.Offset as usize), len)
            };
            if mip < 32 && held & (1 << mip) == 0 {
                // Not the title's: zeros, not what lies in that memory.
                linear.fill(0);
                continue;
            }
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                // With [`transient`] the buffer may hold another texture's
                // bytes: this subresource and the ones after it are zeros
                // instead, as in a new buffer.
                if self.transient {
                    let at = (footprint.Offset as usize).min(self.bytes as usize);
                    // SAFETY: inside the mapped buffer, which is at least
                    // `self.bytes` long.
                    unsafe { std::ptr::write_bytes(buf.data.add(at), 0, self.bytes as usize - at) };
                }
                return;
            }
        }
    }
}

/// The bytes of the staging buffer (`total` long) a subresource at `offset`
/// may use: `pitch * rows * depth`, or what is left of the buffer if that is
/// less. `GetCopyableFootprints` leaves out the row pitch's padding after
/// the last row of the last subresource, so the product can run past the
/// buffer's end (a 4-row texture: 896 bytes of staging, 1,024 of rows).
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether the pass over textures picked by index leaves out the upload of
/// a texture whose host copy was filled from the bytes title memory holds
/// now (`VICTORY_TEXTURE_SAME=1`; off by default: uploaded, as before).
/// cool_fox measured half of the repeated uploads of a frame that streams
/// equal in every byte, and an upload costs a staging buffer, a detiling
/// and a copy more than the look. What the host copy holds is known from
/// checks that hash the memory (not from the write watch's, which do not:
/// with `VICTORY_TEXTURE_ONCE=1` the pass's own uploads hash it).
fn same() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_SAME").is_some_and(|v| v == "1"))
}

/// Uploads [`same`] left out, since the start.
static SAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether `resource`'s host copy was filled from the bytes its title
/// memory holds now (all of it mapped, and of the same hash): an upload
/// would bring nothing new. Not for a texture of levels on demand.
pub(crate) fn unchanged(resource: This) -> bool {
    let Some(copy) = crate::resource::host_copy(resource) else {
        return false;
    };
    if copy.tiled.is_some() {
        return false;
    }
    let Some(held) = *copy.holds.lock().unwrap_or_else(|e| e.into_inner()) else {
        return false;
    };
    let Some((memory, desc)) = crate::resource::texture_memory(resource) else {
        return false;
    };
    let Some(layout) = crate::resource::texture_layout(&desc) else {
        return false;
    };
    let size = layout.planes[0].layout.size as usize;
    with_read(size, |read| {
        crate::host::read_into(memory, read) && content_hash(read) == held
    })
}

/// With [`same`]: where a title texture's memory is, address and bytes, as
/// [`staged_memory`] says it of a staged one (for the look again at a
/// texture whose upload was left out). `None` with the switch off.
pub(crate) fn left_out_memory(resource: This) -> Option<(u64, u64)> {
    if !same() {
        return None;
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let layout = crate::resource::texture_layout(&desc)?;
    Some((memory as u64, layout.planes[0].layout.size))
}

/// Lets go of a title texture's staging buffer (lists that recorded an
/// upload from it hold it until they have run); the next upload makes one
/// anew.
pub(crate) fn unstage(resource: This) {
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(map) = map.as_mut() {
        map.remove(&(resource as usize));
    }
}

/// Where a staged title texture's memory is: address and bytes.
pub(crate) fn staged_memory(resource: This) -> Option<(u64, u64)> {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = map.as_ref()?.get(&(resource as usize))?.memory();
    Some((at as u64, len as u64))
}

/// A fingerprint of the title memory `[at, at + len)`, never 0: of all of it up to
/// 48 KiB, else of twelve pieces of 4 KiB spread evenly over it, the first
/// and the last included. Read through the kernel, so pages not there count
/// as zeros. (A texture whose change falls between the pieces is missed.)
pub(crate) fn fingerprint(at: u64, len: u64) -> u64 {
    const PIECE: usize = 4 << 10;
    const PIECES: usize = 12;
    thread_local! {
        static READ: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let len = len as usize;
    READ.with(|read| {
        let mut read = read.borrow_mut();
        read.resize(PIECES * PIECE, 0);
        let mut hasher = blake3::Hasher::new();
        // With [`cheap_hash`] the pieces' digests folded instead.
        let cheap = cheap_hash();
        let mut print = 0u128;
        let mut take = |from: usize, bytes: usize| {
            read[..bytes].fill(0);
            crate::host::copy_mapped(read.as_mut_ptr(), (at as usize + from) as *const u8, bytes, false);
            if cheap {
                print = print.rotate_left(37) ^ digest(&read[..bytes]).0;
            } else {
                hasher.update(&read[..bytes]);
            }
        };
        if len <= PIECES * PIECE {
            take(0, len);
        } else {
            for n in 0..PIECES - 1 {
                take(((len - PIECE) / (PIECES - 1) * n) & !0xfff, PIECE);
            }
            take(len - PIECE, PIECE);
        }
        if cheap {
            return (print as u64 ^ (print >> 64) as u64) | 1;
        }
        let hash = hasher.finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&hash.as_bytes()[..8]);
        u64::from_le_bytes(first) | 1
    })
}

/// Has the next list that uploads a title texture with a staging buffer
/// copy it whole, changed or not since: what the check at a submission
/// finds changed is copied by the next list that samples the texture
/// through a table, and there may be none for one that shaders now pick by
/// index alone.
pub(crate) fn upload_again(resource: This) {
    let staging = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()
            .and_then(|map| map.get(&(resource as usize)).cloned())
    };
    if let Some(staging) = staging {
        staging
            .uploaded
            .store(false, std::sync::atomic::Ordering::Relaxed);
        staging
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .partial = None;
    }
}

/// Whether a title texture has a staging buffer: a list recorded its
/// upload, and the lists that sample it check it at submission.
/// For the frame listing: whether a list has uploaded `resource`'s staging
/// buffer, and whether it was detiled since. None: it has none.
#[cfg(feature = "diagnostics")]
pub(crate) fn staged_state(resource: This) -> Option<(bool, bool)> {
    use std::sync::atomic::Ordering::Relaxed;
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let staging = map.as_ref()?.get(&(resource as usize))?;
    Some((staging.uploaded.load(Relaxed), staging.dirty.load(Relaxed)))
}

pub(crate) fn is_staged(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref().is_some_and(|map| map.contains_key(&(resource as usize)))
}

/// With [`transient`]: whether the last upload of `resource` a list asked
/// for recorded no copy and left it to the next list that uploads the
/// texture ([`Recorded::Later`]). The pass over textures picked by index,
/// whose slot may be the only thing that uploads it, keeps it due for that
/// list.
pub(crate) fn copy_pending(resource: This) -> bool {
    let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    map.as_ref()
        .and_then(|map| map.get(&(resource as usize)))
        .is_some_and(|s| s.later.load(std::sync::atomic::Ordering::Relaxed))
}

/// The bytes of a title texture's memory, and whether all of it is mapped,
/// if the texture has been staged for upload.
pub(crate) fn staged(resource: This) -> Option<(u64, bool)> {
    let s = {
        let map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        map.as_ref()?.get(&(resource as usize))?.clone()
    };
    let size = s.layout.size as usize;
    Some((size as u64, committed(s.memory, size)))
}

/// Whether `[at, at + len)` is committed memory (streamed textures may be
/// unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    if by_query() {
        return committed_by_query(at, len);
    }
    if len == 0 {
        return true;
    }
    let (there, of) = readable(at as usize, len, true);
    there == of
}

/// Whether the two questions above and below are put to `VirtualQuery`, as
/// they were (`VICTORY_COMMITTED_QUERY=1`). `VirtualQuery` answers with the
/// length of the run of pages in the same state, which it finds by walking
/// the page tables to the run's end: 8.8 ms for one question in the title's
/// 1.1 GB block (round 6). The pass over the heap's views asks three or
/// more of them for every texture slot that is due, 3,000 slots a frame
/// when the card has no room (run 75): that pass then ran 208-264 ms, and
/// while it ran the submitting thread's write-watch askings waited behind
/// the walks (run 76: 8-22 askings a frame of 6-21 ms each, 46-172 ms).
fn by_query() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COMMITTED_QUERY").is_some_and(|v| v == "1"))
}

/// Of the kernel's pages that hold `[at, at + len)` (`len > 0`): how many
/// can be read, and how many there are; with `all`, it stops at the first
/// that cannot. One question to the kernel for up to 256 pages at once
/// (`QueryWorkingSetEx`: the state of each page asked about, no walk); a
/// page not in the working set may still be there (never touched, or
/// trimmed), and a one-byte read through the kernel says.
/// A page that is committed and may not be read counts as not there, which
/// is what the callers ask for: they read or write the memory next.
fn readable(at: usize, len: usize, all: bool) -> (usize, usize) {
    use windows_sys::Win32::System::ProcessStatus::{K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    const KERNEL_PAGE: usize = 4096;
    let Some(end_incl) = at.checked_add(len - 1) else {
        return (0, 1);
    };
    let (first, last) = (at / KERNEL_PAGE, end_incl / KERNEL_PAGE);
    let of = last - first + 1;
    let mut there = 0;
    // SAFETY: plain data.
    let mut info: [PSAPI_WORKING_SET_EX_INFORMATION; 256] = unsafe { std::mem::zeroed() };
    let mut page = first;
    while page <= last {
        let n = (last - page + 1).min(info.len());
        for (i, entry) in info[..n].iter_mut().enumerate() {
            entry.VirtualAddress = ((page + i) * KERNEL_PAGE) as *mut _;
        }
        // SAFETY: `n` entries of the size given; our own process.
        let asked = unsafe {
            K32QueryWorkingSetEx(
                GetCurrentProcess(),
                info.as_mut_ptr().cast(),
                (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
            )
        } != 0;
        for (i, entry) in info[..n].iter().enumerate() {
            // Bit 0 of the answer: the page is in the working set.
            // SAFETY: plain data the query filled in.
            let valid = asked && unsafe { entry.VirtualAttributes.Flags } & 1 != 0;
            let mut byte = [0u8];
            if valid || crate::host::read_into(((page + i) * KERNEL_PAGE) as *const u8, &mut byte) {
                there += 1;
            } else if all {
                return (there, of);
            }
        }
        page += n;
    }
    (there, of)
}

/// Whether a region `VirtualQuery` described is committed memory that may be
/// read: not no-access (the host's form of the console's GPU-only memory)
/// and no guard page, as [`readable`] counts them.
fn is_readable_commit(info: &windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_GUARD, PAGE_NOACCESS};
    info.State == MEM_COMMIT
        && info.Protect != 0
        && (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0
}

/// Whether all of `[at, at + len)` is committed memory that may be read
/// ([`is_readable_commit`]), by `VirtualQuery`.
pub(crate) fn committed_by_query(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        if n == 0 || !is_readable_commit(&info) {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// How many bytes of `[at, at + len)` are committed memory: all of a
/// texture that is there, some of one streaming in (its smaller mips lie
/// at the start of its memory, and their pages are mapped first), none of
/// one not there. With neither the first nor the last page committed the
/// answer is none without a look at the pages between (two queries for the
/// slots whose texture is away, which are asked again and again).
pub(crate) fn mapped_bytes(at: *const u8, len: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    if len == 0 {
        return 0;
    }
    let start = at as usize;
    let Some(end) = start.checked_add(len) else {
        return 0;
    };
    if !committed(at, 1) && !committed((end - 1) as *const u8, 1) {
        return 0;
    }
    if !by_query() {
        static BY_PAGE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *BY_PAGE.get_or_init(|| std::env::var_os("VICTORY_MAPPED_BY_PAGE").is_some_and(|v| v == "1")) {
            // By page, without the walk: whole pages, cut to the range.
            let (there, _) = readable(start, len, false);
            return (there as u64 * 4096).min(len as u64);
        }
        // By 64 KiB block, one page asked of each: the title's memory is
        // mapped in blocks of that size (the console's page), and the
        // answer is a measure of how much of a texture has arrived, not a
        // promise that every byte can be read (the upload reads through the
        // kernel). By page, a texture half there cost a failed kernel read
        // for each of its missing pages, every frame it stayed due: 1,663
        // textures "uploaded in part" kept the pass over the heap's views at
        // 90-160 ms a frame (run 81). `VICTORY_MAPPED_BY_PAGE=1`: by page.
        const BLOCK: usize = 64 << 10;
        let mut there = 0u64;
        let mut block = start & !(BLOCK - 1);
        while block < end {
            let (lo, hi) = (block.max(start), block.saturating_add(BLOCK).min(end));
            if readable(lo, 1, true).0 == 1 {
                there += (hi - lo) as u64;
            }
            block = block.saturating_add(BLOCK);
        }
        return there;
    }
    let (mut p, mut there) = (start, 0u64);
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize).min(end);
        if n == 0 || next <= p {
            break;
        }
        if is_readable_commit(&info) {
            there += (next - p) as u64;
        }
        p = next;
    }
    there
}

/// The staging state of a title texture, made on first use.
fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    // Asked before the table is locked: the sweep lets go of a staging
    // buffer with its texture's host side locked, in that order.
    let on_demand = crate::resource::host_tiled(resource).is_some();
    let mut map = STAGING.lock().unwrap_or_else(|e| e.into_inner());
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(s) = map.get(&(resource as usize)) {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let s = Arc::new(make_staging(memory, layout, texture, on_demand, transient(), || {
        format!("staging of {resource:?}")
    })?);
    tracing::debug!(?desc, total = s.bytes, "texture contents uploaded from title memory");
    map.insert(resource as usize, s.clone());
    Some(s)
}

/// The staging state of a texture whose console layout is `layout`, read
/// from title memory at `memory`, for its host copy `texture`: with a
/// staging buffer (`name`d for the debug tools) unless `transient`.
fn make_staging(
    memory: *const u8,
    layout: victory_formats::texture::Layout,
    texture: &ID3D12Resource,
    on_demand: bool,
    transient: bool,
    name: impl FnOnce() -> String,
) -> Option<Staging> {
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // With the switch a buffer is given when one is needed.
    let buf = if transient {
        None
    } else {
        let buf = Buf::pooled(total)?;
        // (A block's buffer has its own name.)
        if buf.piece.is_none() {
            crate::host::name(buf.buffer(), || {
                format!(
                    "{} {}x{}x{} f{}",
                    name(),
                    d.Width,
                    d.Height,
                    d.DepthOrArraySize,
                    d.Format.0
                )
            });
        }
        Some(Arc::new(buf))
    };
    Some(Staging {
        buf: Mutex::new(buf),
        transient,
        fresh: std::sync::atomic::AtomicBool::new(false),
        later: std::sync::atomic::AtomicBool::new(false),
        detiles: std::sync::atomic::AtomicU64::new(0),
        copied: std::sync::atomic::AtomicU64::new(0),
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        bytes: total,
        memory,
        last: Mutex::default(),
        dirty: std::sync::atomic::AtomicBool::new(true),
        uploaded: std::sync::atomic::AtomicBool::new(false),
        last_upload: std::sync::atomic::AtomicU64::new(0),
        used: std::sync::atomic::AtomicU64::new(crate::host::moment()),
        on_demand,
        once: std::sync::atomic::AtomicBool::new(once() && !on_demand && PICKING.get()),
        detiled: std::sync::atomic::AtomicU32::new(0),
        judged: [const { std::sync::atomic::AtomicU64::new(0) }; 2],
        waiting: std::sync::atomic::AtomicU64::new(0),
    })
}

#[cfg(test)]
mod guard_tests {
    use super::{HASH_EVERY, RECENTLY_CHANGED, guard_skips};

    /// Of `frames`, the ones a guarded texture at `start` is hashed in.
    fn hashed(
        frames: std::ops::Range<u64>,
        unwritten: bool,
        start: usize,
        changed: Option<u64>,
    ) -> Vec<u64> {
        frames
            .filter(|&f| !guard_skips(unwritten, f, start, changed))
            .collect()
    }

    /// Unwritten and unchanged, a texture is hashed on its turn only, one
    /// frame in eight; a write the guard saw has it hashed at once.
    #[test]
    fn a_still_texture_takes_its_turn() {
        assert_eq!(hashed(100..132, true, 0x3_0000, None), [101, 109, 117, 125]);
        assert_eq!(hashed(100..132, true, 0x4_0000, None), [100, 108, 116, 124]);
        assert_eq!(hashed(100..108, false, 0x3_0000, None).len(), 8);
    }

    /// A texture a hash found changed is hashed at every check for the
    /// next 16 frames, so one rewritten every frame through another view
    /// (where the guard cannot see the writes) is never shown stale; after
    /// that it goes back to taking turns (#269).
    #[test]
    fn a_changing_texture_is_hashed_at_every_frame() {
        let changed = Some(101);
        let after = 101 + RECENTLY_CHANGED;
        assert_eq!(
            hashed(101..after, true, 0x3_0000, changed),
            (101..after).collect::<Vec<_>>()
        );
        let later = hashed(after..after + 2 * HASH_EVERY, true, 0x3_0000, changed);
        assert_eq!(later, [117, 125]);
        // A change noted in the frame counter's last values still counts.
        assert!(!guard_skips(true, 3, 0x3_0000, Some(u64::MAX - 2)));
    }
}

#[cfg(test)]
mod digest_tests {
    use super::{Digest, digest};

    /// The digest tells every length and every changed byte apart, zeros
    /// among them.
    #[test]
    fn a_digest_tells_bytes_apart() {
        let zeros = [0u8; 400];
        let mut seen: Vec<Digest> = (0..400).map(|n| digest(&zeros[..n])).collect();
        for at in 0..400 {
            for value in [1u8, 0x80, 0xff] {
                let mut bytes = zeros;
                bytes[at] = value;
                seen.push(digest(&bytes));
            }
        }
        let text: Vec<u8> = (0..400u32).map(|i| ((i * 37) >> 2) as u8).collect();
        for at in 0..399 {
            let mut bytes = text.clone();
            bytes.swap(at, at + 1);
            if bytes != text {
                seen.push(digest(&bytes));
            }
        }
        seen.push(digest(&text));
        let n = seen.len();
        seen.sort_unstable_by_key(|d| d.0);
        seen.dedup();
        assert_eq!(seen.len(), n);
        assert_eq!(digest(&text), digest(&text.clone()));
    }
}

#[cfg(test)]
mod title_memory_tests {
    use super::{Last, with_title};

    #[test]
    fn repeated_sparse_guard_reports_do_not_bypass_partial_throttle() {
        let mut last = Last::default();
        let mut reads = Vec::new();
        // Simulate every submission reaching the sparse read path due to
        // a conservative guard report (including repeated queries per frame).
        for frame in 100..191 {
            for _ in 0..32 {
                if last.partial_due(frame) {
                    reads.push(frame);
                    last.partial = Some(frame);
                }
            }
        }
        assert_eq!(reads, [100, 130, 160, 190]);
        // A fully mapped path clears the partial polling state.
        last.partial = None;
        assert!(last.partial_due(191));
    }

    /// A detiling after the write-watch leaves no hash as what the staging
    /// buffer holds, and memory back at the hashed bytes does not bring it
    /// back (`VICTORY_TEXTURE_SAME`'s note of what the host copy holds).
    #[test]
    fn a_watched_detiling_forgets_what_the_staging_buffer_held() {
        let mut last = Last::default();
        let a = super::content_hash(&[0x11; 64]);
        assert!(last.rehash(a));
        assert_eq!(last.held, Some(a));
        // The watch path detiled other bytes.
        last.held = None;
        // The memory is back at the first bytes: no detiling, nothing held.
        assert!(!last.rehash(a));
        assert_eq!(last.held, None);
        let b = super::content_hash(&[0x22; 64]);
        assert!(last.rehash(b));
        assert_eq!(last.held, Some(b));
    }

    #[test]
    fn skipped_zero_partial_read_does_not_replace_staged_hash() {
        let mut last = Last::default();
        let data = [0x51; 64];
        let zero = [0; 64];
        assert!(!last.accept_partial(&zero));
        assert_eq!(last.hash, None);
        assert!(last.accept_partial(&data));
        assert_eq!(last.hash, Some(super::content_hash(&data)));
        assert!(!last.accept_partial(&zero));
        assert_eq!(last.hash, Some(super::content_hash(&data)));
        // A later fully mapped zero texture must not be skipped as equal
        // to the nonzero contents still in staging.
        assert_ne!(last.hash, Some(super::content_hash(&zero)));
        assert!(!last.accept_partial(&data));
        assert!(last.accept_partial(&[0xa2; 64]));
    }

    /// Title memory whose tail is reserved and not committed reads through
    /// the copy as its bytes and then zeros, without a fault; so does
    /// memory a page of which is taken away between two reads, and the
    /// zeros are zeros whatever the read buffer held before.
    #[test]
    fn an_uncommitted_tail_reads_as_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const HEAD: usize = 0xb_0000;
        const SIZE: usize = 0x20_0000;
        // SAFETY: a private reservation of our own; only committed pages
        // are written in place, and nothing else reads them in place.
        unsafe {
            // The read buffer holds something else first.
            let other = vec![0x77u8; SIZE];
            with_title(other.as_ptr(), SIZE, |bytes| assert!(bytes.iter().all(|&b| b == 0x77)));
            let base = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            assert_eq!(VirtualAlloc(base.cast(), HEAD, MEM_COMMIT, PAGE_READWRITE), base.cast());
            base.write_bytes(0x5a, HEAD);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..HEAD].iter().all(|&b| b == 0x5a), "what is mapped");
                assert!(bytes[HEAD..].iter().all(|&b| b == 0), "what is not: zeros");
            });
            // The title takes a page in the middle away: zeros from then
            // on, the rest as before.
            assert_ne!(VirtualFree(base.add(0x4_0000).cast(), 0x1_0000, MEM_DECOMMIT), 0);
            with_title(base, SIZE, |bytes| {
                assert!(bytes[..0x4_0000].iter().all(|&b| b == 0x5a));
                assert!(bytes[0x4_0000..0x5_0000].iter().all(|&b| b == 0));
                assert!(bytes[0x5_0000..HEAD].iter().all(|&b| b == 0x5a));
                assert!(bytes[HEAD..].iter().all(|&b| b == 0));
            });
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// The texture both device-loss runs named last (16 texels wide, 4 or 8
    /// high, format 10, layout 0x100, console flags 0x300000; its memory
    /// began 0x300 into a page in run 73 and 0x700 in run 72, and here also
    /// 0xf00, so that a page line runs through it): what the detiling is
    /// given and what it writes are the same read in place and read through
    /// the copy, and the rows it writes end where the staging buffer the
    /// host sizes for it ends (896 bytes for 4 rows of 128 bytes 256 apart,
    /// 1,920 for 8).
    #[test]
    fn the_small_texture_of_runs_72_and_73_detiles_the_same_both_ways() {
        use windows_sys::Win32::System::Memory::*;
        for (height, within) in [(4u32, 0x300usize), (8, 0x700), (4, 0xf00), (8, 0xf00)] {
            // The description as the log printed it, word for word.
            let words: [u32; 14] = [3, 0, 0, 0, 16, 0, height, 65537, 10, 1, 0, 256, 3_145_728, 0];
            // SAFETY: fourteen words are a resource description (56 bytes,
            // plain numbers).
            let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
            let layout = crate::resource::texture_layout(&desc)
                .expect("a console layout")
                .planes[0]
                .layout
                .clone();
            let size = layout.size as usize;
            let (row, pitch) = (16 * 8, 256usize);
            assert!(size >= row * height as usize, "the layout holds every texel");
            // Committed pages and a reserved one after them; the texture
            // begins `within` bytes into the first, as the title placed it.
            // SAFETY: our own reservation; the committed part is written
            // and read in place, by this thread alone.
            unsafe {
                let pages = size.div_ceil(4096) + 1;
                let base = VirtualAlloc(std::ptr::null(), (pages + 1) * 4096, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
                assert!(!base.is_null());
                assert_eq!(VirtualAlloc(base.cast(), pages * 4096, MEM_COMMIT, PAGE_READWRITE), base.cast());
                for i in 0..pages * 4096 {
                    base.add(i).write((i * 7 + i / 251) as u8);
                }
                let memory = base.add(within);
                let in_place = std::slice::from_raw_parts(memory, size);
                // The bytes: the same in place and through the copy.
                with_title(memory, size, |copied| assert_eq!(copied, in_place));
                // What the detiling writes: the same, and inside the bytes
                // the host's staging buffer for this texture has.
                let staged = (height as usize - 1) * pitch + row;
                assert_eq!(staged, if height == 4 { 896 } else { 1920 });
                let detile = |tiled: &[u8]| {
                    // The slice `Staging::detile` hands over: whole rows.
                    let mut linear = vec![0xeeu8; pitch * height as usize];
                    layout
                        .detile(tiled, 0, 0, &mut linear, pitch, pitch * height as usize)
                        .expect("detiled");
                    linear
                };
                let direct = detile(in_place);
                let through = with_title(memory, size, |copied| detile(copied));
                assert_eq!(direct, through);
                for y in 0..height as usize {
                    assert!(direct[y * pitch..][..row].iter().any(|&b| b != 0xee), "row {y} written");
                    assert!(direct[y * pitch + row..(y + 1) * pitch].iter().all(|&b| b == 0xee), "nothing past row {y}");
                }
                // A buffer of exactly the staged bytes is enough.
                let mut exact = vec![0u8; staged];
                layout
                    .detile(in_place, 0, 0, &mut exact, pitch, pitch * height as usize)
                    .expect("the staging buffer's bytes are enough");
                assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
            }
        }
    }

    /// A subresource's slice of the staging buffer ends with the buffer.
    #[test]
    fn subresource_len_stays_inside_staging_bytes() {
        // 4 rows of pitch 256, the last 128 long: 896 bytes staged, 1,024 of rows.
        assert_eq!(super::subresource_len(0, 256, 4, 1, 896), 896);
        assert_eq!(super::subresource_len(512, 256, 4, 1, 896), 384);
        assert_eq!(super::subresource_len(0, 256, 4, 1, 2048), 1024);
        assert_eq!(super::subresource_len(1024, 256, 4, 1, 896), 0);
    }

    /// A range past the address space's end, a no-access page and a guard
    /// page are not there to read; a read-write page is.
    #[test]
    fn overflowing_ranges_and_unreadable_pages_are_not_there() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE,
            VirtualAlloc, VirtualFree,
        };
        assert!(!super::committed(usize::MAX as *const u8, 64));
        assert!(!super::committed_by_query(usize::MAX as *const u8, 64));
        assert_eq!(super::readable(usize::MAX - 8, 64, true), (0, 1));
        assert_eq!(super::mapped_bytes(usize::MAX as *const u8, 64), 0);
        let page = |protect| {
            // SAFETY: a page of our own, freed below.
            let at =
                unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, protect) };
            assert!(!at.is_null());
            at
        };
        // SAFETY: a page `page` made.
        let free = |at| assert_ne!(unsafe { VirtualFree(at, 0, MEM_RELEASE) }, 0);
        let none = page(PAGE_NOACCESS);
        assert!(!super::committed(none.cast(), 4096));
        assert!(!super::committed_by_query(none.cast(), 4096));
        assert_eq!(super::mapped_bytes(none.cast(), 4096), 0);
        free(none);
        let guard = page(PAGE_READWRITE | PAGE_GUARD);
        assert!(!super::committed_by_query(guard.cast(), 4096));
        free(guard);
        let rw = page(PAGE_READWRITE);
        assert!(super::committed(rw.cast(), 4096));
        assert!(super::committed_by_query(rw.cast(), 4096));
        assert_eq!(super::mapped_bytes(rw.cast(), 4096), 4096);
        free(rw);
    }
}

#[cfg(test)]
mod staging_tests {
    use super::{Pool, class_of, releasable};

    /// Size classes are quarter steps of a power of two from 64 KiB on: a
    /// buffer is never smaller than asked for and wastes less than a quarter.
    #[test]
    fn staging_size_classes_waste_under_a_quarter() {
        assert_eq!(class_of(0), 64 << 10);
        assert_eq!(class_of(896), 64 << 10);
        assert_eq!(class_of(64 << 10), 64 << 10);
        assert_eq!(class_of((64 << 10) + 1), 80 << 10);
        assert_eq!(class_of(1 << 20), 1 << 20);
        assert_eq!(class_of((1 << 20) + 1), 1280 << 10);
        assert_eq!(class_of(22 << 20), 24 << 20);
        for bytes in (1u64..40).map(|i| i * 777_777) {
            let class = class_of(bytes);
            let at_least = bytes.max(64 << 10);
            assert!(class >= at_least);
            assert!(class <= at_least + at_least / 4, "{bytes}: {class}");
            assert_eq!(class_of(class), class, "a class is its own class");
        }
    }

    /// The pool hands back the newest item of the class asked for, nothing
    /// of another class, and keeps within its cap by letting the oldest go.
    #[test]
    fn staging_pool_reuses_by_class_and_keeps_its_cap() {
        let mut pool = Pool::new();
        assert!(pool.put(64, 1, 256).is_empty());
        assert!(pool.put(64, 2, 256).is_empty());
        assert!(pool.put(128, 3, 256).is_empty());
        assert_eq!((pool.len(), pool.bytes), (3, 256));
        assert_eq!(pool.take(96), None, "no item of another class");
        assert_eq!(pool.take(64), Some(2), "the newest of its class");
        assert_eq!((pool.len(), pool.bytes), (2, 192));
        // Over the cap: the oldest go first, whatever their class.
        assert_eq!(pool.put(128, 4, 256), vec![1]);
        assert_eq!((pool.len(), pool.bytes), (2, 256));
        assert_eq!(pool.put(64, 5, 256), vec![3]);
        assert_eq!(pool.take(128), Some(4));
        assert_eq!(pool.take(128), None);
        assert_eq!(pool.take(64), Some(5));
        assert_eq!((pool.len(), pool.bytes), (0, 0));
        // Bigger than the cap: not kept at all; a cap of 0 keeps nothing.
        assert_eq!(pool.put(512, 6, 256), vec![6]);
        assert_eq!(pool.put(64, 7, 0), vec![7]);
        assert_eq!((pool.len(), pool.bytes), (0, 0));
    }

    /// A staging buffer goes back to the pool only once what it holds was
    /// uploaded by a list that has run, nothing newer waits for an upload,
    /// it is not waiting for its first detiling, and it was not copied from
    /// in the last frames.
    #[test]
    fn staging_release_policy() {
        assert!(releasable(true, false, false, true, true));
        assert!(!releasable(false, false, false, true, true), "never uploaded");
        assert!(!releasable(true, true, false, true, true), "detiled since: the next list uploads it");
        assert!(!releasable(true, false, true, true, true), "given for a copy, not detiled yet");
        assert!(!releasable(true, false, false, false, true), "a list still holds it");
        assert!(!releasable(true, false, false, true, false), "copied from lately");
    }
}

/// WARP: the staging buffer's lifetime against submissions, and transient
/// staging buffers (`VICTORY_D3D12_WARP=1`, diagnostics builds).
#[cfg(all(test, feature = "diagnostics"))]
mod staging_warp_tests {
    use super::{Arc, HostList, Recorded, Staging, TRANSIENT, make_staging};
    use crate::host::{host, references};
    use crate::object::This;
    use std::sync::atomic::Ordering::Relaxed;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32G32_UINT, DXGI_SAMPLE_DESC};
    use windows::core::Interface;

    /// Title memory of 64 KiB, write-watched as the title's is, and a
    /// texture in it: 16 by 4 texels of 8 bytes (format 10, the console's
    /// layout 0x100), its host copy in the copy destination state.
    struct Fixture {
        /// The allocation, and the texture's memory in it.
        base: *mut u8,
        memory: *mut u8,
        layout: victory_formats::texture::Layout,
        texture: ID3D12Resource,
    }

    const MEMORY: usize = 64 << 10;
    /// The host copy's rows as a copy lays them out: 256 apart, 128 long.
    const PITCH: usize = 256;
    const ROW: usize = 128;
    const ROWS: usize = 4;
    const STAGED: usize = (ROWS - 1) * PITCH + ROW;

    impl Fixture {
        fn new() -> Fixture {
            Fixture::at(0, true)
        }

        /// The texture's memory `offset` bytes into the allocation, which
        /// is write-watched if `watched` (memory that is not is checked by
        /// its mapping and hash, as view memory is).
        fn at(offset: usize, watched: bool) -> Fixture {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            let host = host().expect("a WARP host device (VICTORY_D3D12_WARP=1)");
            let layout = layout();
            assert!(offset + layout.size as usize <= MEMORY);
            // SAFETY: a new committed allocation the test owns.
            let watch = if watched { MEM_WRITE_WATCH } else { 0 };
            let base = unsafe {
                VirtualAlloc(std::ptr::null(), MEMORY, MEM_RESERVE | MEM_COMMIT | watch, PAGE_READWRITE)
            }
            .cast::<u8>();
            assert!(!base.is_null());
            // SAFETY: inside the allocation (asserted above).
            let memory = unsafe { base.add(offset) };
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 16,
                Height: ROWS as u32,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R32G32_UINT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a texture.
            unsafe {
                let dest = D3D12_RESOURCE_STATE_COPY_DEST;
                host.device
                    .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, dest, None, &mut texture)
                    .unwrap();
            }
            let fixture = Fixture {
                base,
                memory,
                layout,
                texture: texture.unwrap(),
            };
            fixture.fill(0x11);
            fixture
        }

        /// Writes the texture's memory, byte `i` being `seed + i * 7 + i / 251`.
        fn fill(&self, seed: u8) {
            for i in 0..self.layout.size as usize {
                // SAFETY: inside the allocation.
                unsafe { self.memory.add(i).write(seed.wrapping_add((i * 7 + i / 251) as u8)) };
            }
        }

        /// The texture's rows as the CPU detiles its memory now.
        fn expected(&self) -> Vec<u8> {
            // SAFETY: the committed allocation, read by this thread alone.
            let tiled = unsafe { std::slice::from_raw_parts(self.memory, self.layout.size as usize) };
            let mut linear = vec![0u8; STAGED];
            self.layout
                .detile(tiled, 0, 0, &mut linear, PITCH, PITCH * ROWS)
                .expect("detiled");
            rows(&linear)
        }

        fn staging(&self, transient: bool) -> Arc<Staging> {
            Arc::new(
                make_staging(self.memory, self.layout.clone(), &self.texture, false, transient, || "test staging".into())
                    .expect("a staging state"),
            )
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: the allocation `at` made.
            unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
        }
    }

    /// The console layout of the fixture's texture.
    fn layout() -> victory_formats::texture::Layout {
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, ROWS as u32, 65537, 10, 1, 0, 256, 3_145_728, 0];
        // SAFETY: fourteen words are a resource description (plain numbers).
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        crate::resource::texture_layout(&desc)
            .expect("a console layout")
            .planes[0]
            .layout
            .clone()
    }

    /// `staging` in the staging table as the state of title texture
    /// `resource`, until this goes (`unstage`), whether the test passes or
    /// not.
    struct Staged(This);

    impl Staged {
        fn new(resource: This, staging: &Arc<Staging>) -> Staged {
            super::STAGING
                .lock()
                .unwrap()
                .get_or_insert_with(Default::default)
                .insert(resource as usize, staging.clone());
            Staged(resource)
        }
    }

    impl Drop for Staged {
        fn drop(&mut self) {
            super::unstage(self.0);
        }
    }

    /// The `ROW` bytes of each of the `ROWS` rows of copy-layout bytes.
    fn rows(linear: &[u8]) -> Vec<u8> {
        (0..ROWS).flat_map(|y| linear[y * PITCH..][..ROW].iter().copied()).collect()
    }

    /// The title texture the tests name (only printed).
    fn key() -> This {
        0x5_0000usize as This
    }

    /// Records the upload of `staging` as `crate::texture::upload` does (its
    /// copies with `upload`, then its step at submission, which only checks
    /// without or when no copy was recorded), then a copy of the host
    /// texture into a read-back buffer; runs the list (waited for), resets
    /// it and returns the texture's rows.
    fn upload_and_read(f: &Fixture, staging: &Arc<Staging>, upload: bool) -> Vec<u8> {
        upload_and_read_as(f, staging, upload).0
    }

    /// [`upload_and_read`], and what was recorded (None: only a check).
    fn upload_and_read_as(
        f: &Fixture,
        staging: &Arc<Staging>,
        upload: bool,
    ) -> (Vec<u8>, Option<Recorded>) {
        let host = host().unwrap();
        let list = host.new_list().unwrap();
        let recorded = upload.then(|| staging.record_copies(&list, &f.texture, key()));
        assert_ne!(recorded, Some(Recorded::Nothing), "a staging buffer to be had");
        list.defer_texture(staging.clone(), recorded == Some(Recorded::Copies));
        let back = read_into(&list, f);
        list.close();
        host.execute(&[&list]);
        list.reset();
        (read_back(&back), recorded)
    }

    /// Records the copy of the host texture into a new read-back buffer.
    fn read_into(list: &HostList, f: &Fixture) -> ID3D12Resource {
        let host = host().unwrap();
        let back = host.read_back_buffer(STAGED as u64).unwrap();
        // SAFETY: a live texture.
        let d = unsafe { f.texture.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        // SAFETY: outputs for one subresource.
        unsafe { host.device.GetCopyableFootprints(&d, 0, 1, 0, Some(&mut footprint), None, None, None) };
        assert_eq!(footprint.Footprint.RowPitch as usize, PITCH);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(&back) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&f.texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        let (dest, source) = (D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        // SAFETY: an open list and live resources; the texture rests in the
        // copy destination state between lists.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(&f.texture, dest, source)]);
            list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.list.ResourceBarrier(&[crate::host::transition(&f.texture, source, dest)]);
        }
        list.keep(&back);
        back
    }

    fn read_back(back: &ID3D12Resource) -> Vec<u8> {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of `STAGED` bytes the card has written
        // and nothing writes any more.
        unsafe {
            back.Map(0, None, Some(&mut data)).unwrap();
            let linear = std::slice::from_raw_parts(data.cast::<u8>(), STAGED).to_vec();
            back.Unmap(0, None);
            rows(&linear)
        }
    }

    fn buffer_of(staging: &Staging) -> Option<ID3D12Resource> {
        staging.buf.lock().unwrap().as_ref().map(|b| b.buffer().clone())
    }

    /// Guards against a staging buffer freed while a submission runs
    /// (suspected in the device loss of run r2b-standin-2, a page fault
    /// inside freed texture-staging allocations while textures picked by
    /// index were being made ready; not shown): a list records an upload
    /// from a texture's staging buffer, the texture lets go of its staging
    /// state (`unstage`, as the pass over textures picked by index does
    /// after every upload), and the last holder lets go while a submission
    /// is still on the card. The buffer must outlive that submission.
    #[test]
    fn warp_a_staging_buffer_let_go_during_a_submission_outlives_it() {
        let f = Fixture::new();
        let host = host().unwrap();
        let staging = f.staging(false);
        let buffer = buffer_of(&staging).expect("a staging buffer");
        let list = host.new_list().unwrap();
        assert_eq!(staging.record_copies(&list, &f.texture, key()), Recorded::Copies);
        list.defer_texture(staging.clone(), true);
        super::STAGING
            .lock()
            .unwrap()
            .get_or_insert_with(Default::default)
            .insert(key() as usize, staging);
        super::unstage(key());
        // The list holds the staging state alone now, and ours is the only
        // other reference to the buffer.
        assert_eq!(references(&buffer), 2);
        list.close();
        let running = crate::host::Running::start();
        // The holder lets go while the submission runs.
        list.reset();
        assert_eq!(references(&buffer), 2, "the buffer waits for the submission running when it was let go");
        drop(running);
        assert_eq!(references(&buffer), 1, "and goes once that submission has finished");
    }

    /// Transient staging buffers: given for the copy a list records, filled
    /// at its submission whatever the write watch says, back in the pool
    /// once the upload ran, taken again by the next copy with the same
    /// bytes arriving; a change of the memory found by a check gets a
    /// buffer of its own and arrives at the next upload.
    #[test]
    fn warp_transient_staging_uploads_the_same_bytes() {
        let f = Fixture::new();
        let staging = f.staging(true);
        assert!(buffer_of(&staging).is_none(), "made without a buffer");
        let expected = f.expected();
        assert_eq!(upload_and_read(&f, &staging, true), expected, "the first upload");
        // Detiled at that submission after its upload was noted: the next
        // list uploads once more, as without the switch.
        assert!(staging.dirty.load(Relaxed));
        assert!(staging.release_if_uploaded(0).is_none(), "detiled and not uploaded since");
        assert_eq!(upload_and_read(&f, &staging, true), expected);
        // (By its address: a reference of ours would hold it.)
        let buffer = buffer_of(&staging).map(|b| b.as_raw() as usize).expect("kept until the sweep");
        let given = staging.release_if_uploaded(0).expect("released once its upload ran");
        assert!(buffer_of(&staging).is_none());
        super::give_back(given);
        // Asked to upload again (a texture picked by index): a buffer from
        // the pool, detiled anew although nothing was written, by the
        // list's check; the next list copies it.
        let reused = TRANSIENT[2].load(Relaxed);
        staging.uploaded.store(false, Relaxed);
        assert_eq!(
            upload_and_read_as(&f, &staging, true),
            (expected.clone(), Some(Recorded::Later)),
            "no copy from a buffer not filled yet"
        );
        assert_eq!(TRANSIENT[2].load(Relaxed), reused + 1, "taken from the pool");
        assert_eq!(buffer_of(&staging).map(|b| b.as_raw() as usize), Some(buffer), "the same buffer again");
        assert!(!staging.fresh.load(Relaxed) && staging.dirty.load(Relaxed), "filled by the check");
        assert_eq!(
            upload_and_read_as(&f, &staging, true),
            (expected.clone(), Some(Recorded::Copies)),
            "the same bytes from a pooled buffer"
        );
        assert_eq!(upload_and_read(&f, &staging, true), expected);
        super::give_back(staging.release_if_uploaded(0).expect("released"));
        // The title writes the texture while it has no buffer. A list that
        // only samples it checks it at submission: the change is detiled
        // into a buffer it is given then, and the next list uploads it.
        f.fill(0x5d);
        let changed = f.expected();
        assert_ne!(changed, expected);
        assert_eq!(upload_and_read(&f, &staging, false), expected, "a check copies nothing");
        assert!(buffer_of(&staging).is_some() && staging.dirty.load(Relaxed));
        assert_eq!(upload_and_read(&f, &staging, true), changed, "the change arrives");
    }

    /// A transient staging buffer is not taken back while a list that
    /// recorded a copy from it is still to be reset, whatever its texture's
    /// state says, nor before its first detiling; after the list was reset
    /// it is.
    #[test]
    fn warp_staging_not_reused_while_a_list_holds_it() {
        let f = Fixture::new();
        let host = host().unwrap();
        let staging = f.staging(true);
        assert_eq!(upload_and_read(&f, &staging, true), f.expected());
        assert_eq!(upload_and_read(&f, &staging, true), f.expected());
        // Recorded and not yet run, while the texture's state says uploaded.
        let list = host.new_list().unwrap();
        assert_eq!(staging.record_copies(&list, &f.texture, key()), Recorded::Copies);
        assert!(staging.release_if_uploaded(0).is_none(), "a list holds it");
        list.close();
        list.reset();
        assert!(staging.release_if_uploaded(0).is_some(), "the list was reset");
        // A buffer given for a copy is not taken back before it was filled
        // (and, the texture having been detiled before, not copied from).
        let list = host.new_list().unwrap();
        assert_eq!(staging.record_copies(&list, &f.texture, key()), Recorded::Later);
        assert!(staging.fresh.load(Relaxed));
        list.close();
        list.reset();
        assert!(staging.release_if_uploaded(0).is_none(), "not filled yet");
    }

    /// The order of the pass over textures picked by index with transient
    /// staging buffers: a list records the upload, the texture lets go of
    /// its staging state at once (`unstage`), and the list's reset drops it
    /// while the list still holds the buffer: the buffer goes back to the
    /// pool at the next sweep, not before.
    #[test]
    fn warp_transient_staging_let_go_by_its_texture_is_pooled_after_the_list() {
        let f = Fixture::new();
        let host = host().unwrap();
        let staging = f.staging(true);
        let list = host.new_list().unwrap();
        assert_eq!(staging.record_copies(&list, &f.texture, key()), Recorded::Copies);
        list.defer_texture(staging.clone(), true);
        let buffer = buffer_of(&staging).map(|b| b.as_raw() as usize).expect("given for the copy");
        super::STAGING
            .lock()
            .unwrap()
            .get_or_insert_with(Default::default)
            .insert(key() as usize, staging);
        super::unstage(key());
        let back = read_into(&list, &f);
        list.close();
        host.execute(&[&list]);
        assert_eq!(read_back(&back), f.expected());
        let pooled = |raw: usize| {
            let pool = super::POOL.lock().unwrap();
            let raw_of = |b: &super::Buf| b.buffer().as_raw() as usize;
            pool.as_ref()
                .is_some_and(|pool| pool.by_class.values().flatten().any(|(_, b)| raw_of(b) == raw))
        };
        list.reset();
        assert!(!pooled(buffer), "the list held the buffer when its staging state went");
        assert!(super::ORPHANS.lock().unwrap().iter().any(|b| b.buffer().as_raw() as usize == buffer));
        super::pool_orphans();
        assert!(pooled(buffer), "pooled at the sweep after the list's reset");
    }

    /// A texture uploaded whole whose buffer went back to the pool, asked
    /// to upload again (`upload_again`, a texture picked by index) while
    /// its memory is partly mapped and all zero, which a refresh does not
    /// take: its host copy keeps the texels it has, where a copy from the
    /// buffer it is given would have put zeros there. Once the memory can
    /// be read again the buffer is filled and copied.
    #[test]
    fn warp_transient_staging_keeps_an_uploaded_texture_when_its_memory_cannot_be_read() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const PAGE: usize = 4096;
        // The texture's memory across the end of the first page, not
        // write-watched: a partly mapped read is taken only if not all
        // zero.
        let size = layout().size as usize;
        let f = Fixture::at(PAGE - size.min(PAGE) / 2, false);
        // SAFETY: the second page of the fixture's allocation.
        let second = unsafe { f.base.add(PAGE) };
        let staging = f.staging(true);
        let expected = f.expected();
        assert_eq!(upload_and_read(&f, &staging, true), expected);
        assert_eq!(upload_and_read(&f, &staging, true), expected);
        super::give_back(
            staging
                .release_if_uploaded(0)
                .expect("released once its upload ran"),
        );
        // All zero, and the second page gone.
        // SAFETY: the fixture's memory; the decommitted page is not read
        // but through the kernel's copy, which stands it.
        unsafe {
            std::ptr::write_bytes(f.memory, 0, size);
            assert_ne!(VirtualFree(second.cast(), PAGE, MEM_DECOMMIT), 0);
        }
        assert!(!super::committed(f.memory, size), "partly mapped");
        let _staged = Staged::new(key(), &staging);
        super::upload_again(key());
        let (rows, recorded) = upload_and_read_as(&f, &staging, true);
        assert_eq!(
            rows, expected,
            "no zeros over the texels the host copy holds"
        );
        assert_eq!(recorded, Some(Recorded::Later));
        assert!(
            staging.fresh.load(Relaxed),
            "not filled: tried again at the next check"
        );
        let (rows, recorded) = upload_and_read_as(&f, &staging, true);
        assert_eq!(recorded, Some(Recorded::Later), "still no copy");
        assert_eq!(rows, expected);
        // Mapped and written again: filled by the next check, then copied.
        // SAFETY: the page decommitted above.
        assert!(
            !unsafe { VirtualAlloc(second.cast(), PAGE, MEM_COMMIT, PAGE_READWRITE) }.is_null()
        );
        f.fill(0x2b);
        let changed = f.expected();
        assert_ne!(changed, expected);
        assert_eq!(
            upload_and_read_as(&f, &staging, true),
            (expected, Some(Recorded::Later))
        );
        assert_eq!(
            upload_and_read_as(&f, &staging, true),
            (changed, Some(Recorded::Copies))
        );
    }

    /// The pass over textures picked by index with transient staging
    /// buffers, through `crate::texture::upload` itself: a texture uploaded
    /// whole whose buffer went back to the pool and whose memory the title
    /// then wrote is brought up to date (`upload_again`, then `upload`).
    /// That list leaves the copy to the next one and says so
    /// (`copy_pending`), which keeps the slot due; the next pass's list
    /// copies the new bytes, with no list sampling the texture through a
    /// table in between.
    #[test]
    fn warp_transient_staging_index_picked_upload_is_copied_by_the_next_pass() {
        let f = Fixture::new();
        let host = host().unwrap();
        let staging = f.staging(true);
        // A title texture `upload` finds in the staging table and nowhere
        // else: no title resource is behind it (`key` would be read as one
        // for its host copy's state), so the list moves no state.
        let resource = 0usize as This;
        let _staged = Staged::new(resource, &staging);
        // One pass's upload of the texture: its list, run and reset; the
        // texture's rows after it, and whether the copy was left for later.
        let pass = || {
            let list = host.new_list().unwrap();
            super::upload_again(resource);
            super::upload(&list, resource, &f.texture);
            let pending = super::copy_pending(resource);
            let back = read_into(&list, &f);
            list.close();
            host.execute(&[&list]);
            list.reset();
            (read_back(&back), pending)
        };
        let expected = f.expected();
        assert_eq!(pass(), (expected.clone(), false), "the first upload");
        assert_eq!(pass(), (expected.clone(), false));
        super::give_back(staging.release_if_uploaded(0).expect("released once its upload ran"));
        f.fill(0x6e);
        let changed = f.expected();
        assert_ne!(changed, expected);
        assert_eq!(pass(), (expected, true), "left to the next list, the slot kept due");
        assert!(!staging.fresh.load(Relaxed) && staging.dirty.load(Relaxed), "filled by the check");
        assert_eq!(pass(), (changed, false), "the new bytes arrive");
    }
}
