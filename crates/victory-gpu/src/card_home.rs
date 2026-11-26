//! The memory core's class **G** (slice 4, `VICTORY_MEMORY_CARDS=1`, only
//! with `VICTORY_MEMORY_CORE=1`; off by default): a big title buffer the card
//! keeps a copy of in its own memory (`crate::shadow::cards`) gets the home
//! of its bytes (`crate::home`) as the copy's CPU side, instead of a staging
//! buffer of ours in the PC's memory that title memory is copied into.
//!
//! What changes for such a copy, and what does not:
//!
//! - The card side, the hashes of its 64 KiB pieces and the compare after
//!   the lists are as before.
//! - The refresh still asks write watching which pages the title wrote, but
//!   copies nothing (`Shadow::copy`): the pages are noted, and before the
//!   submission they go up from the home to the card side on the card
//!   (`Shadow::card_upload`), the title's own memory read over the bus as
//!   the staging buffer was.
//! - What the shaders wrote is merged into the home on the card, behind the
//!   compare, instead of fetched and merged on the CPU after one more wait
//!   ([`record_merge`]). A merge needs the card side's bytes from before the
//!   lists ran, which the staging buffer was; the home is not (the title may
//!   write it while the card runs). So ahead of the lists, once the changes
//!   went up, the declared ranges of such copies are copied aside on the
//!   card ([`before`]), into a reference in the card's memory (in the PC's
//!   if the card has no room). The merge takes, byte by byte, the card
//!   side's byte where it differs from the reference (a shader wrote it),
//!   else the home's (the title's, perhaps written while the card ran), as
//!   `crate::shadow::merge_piece` does, and stores the words that hold such
//!   a byte. After the wait the merged pieces go up again from the home with
//!   their hashes made anew, and copies of the same bytes elsewhere are told
//!   (`Shadow::card_merged`).
//! - A copy whose declared ranges cover half of it or more (a copy the
//!   shaders reach through `crate::memory`'s directory is declared written
//!   whole for every list with writes) has its reference **kept** between
//!   submissions instead ([`Kept`]): a buffer of the copy's size in the
//!   card's memory that holds what the card side held before the lists. Its
//!   card side only changes where something went up to it (the title's
//!   writes, and the pieces merged after the last submission), so only those
//!   ranges are copied into the kept reference, behind the upload, and the
//!   declared ranges themselves are not copied at all. A kept reference
//!   nothing declared for [`KEPT_IDLE`] submissions is let go.
//! - A free or protection of the title's that lets the home go lets the copy
//!   go too (`crate::shadow::let_go_homes`): it holds the home's heap.
//!
//! The copy's CPU side is asked of the core as a buffer that is not offered
//! the card (`crate::home::resolve`), so whatever kinds of memory the core
//! gives homes come here the same way. A copy whose bytes get no home, or a
//! home in which the copy does not start on a 16-byte step, keeps its
//! staging buffer (counted), as do windows given a copy in the card's memory
//! (`VICTORY_CARD_WINDOWS_MB`).
//!
//! What it assumes is what the core assumes: every submission is waited for
//! (the core gives no home with late waits or the relay).
//!
//! shortcut: a kept reference follows the card side through what went up
//! and what was merged; a shader's write to a copy that no list declared
//! changes the card side behind it. Where a later declared range of the copy
//! is flagged, such bytes differ from the kept reference and are merged into
//! the home as the shaders' (a per-submission reference would have taken
//! them for the bytes from before, and left them on the card side only).
//!
//! shortcut: the merge's read of a home's word and its store of the merged
//! word are two steps; a title byte written to the same word in between is
//! put back to what the merge read. The CPU's merge has the same window
//! (`Shadow::card_merge`), longer, but there the race log (`merge_race`)
//! can see it; nothing sees it here.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::shadow::{CardSpan, Shadow};

/// The merge of a declared range of a copy in the card's memory into the
/// home that is its CPU side. A thread per 16-byte step of the range (the
/// range widened to steps): a step in a piece the compare did not flag is
/// left alone; else every byte in which the card side differs from the
/// reference goes into the home, and each word that holds one is stored.
/// The bytes stored are summed per group, and added once per group to a
/// counter in the card's memory.
const MERGE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Span : register(b0) {
    uint from;      // the first step's offset in the copy (a multiple of 16)
    uint to;        // the end of the steps (a multiple of 16)
    uint reference; // where the first step lies in `before`
    uint unused0;
    uint flags;     // the dword of the range's first piece in `differs`, in bytes; 0xffffffff: not compared
    uint first;     // the range's first piece
    uint skew;      // where the copy starts in its piece 0
    uint unused1;
};
ByteAddressBuffer now : register(t0);     // the card side, as the shaders left it
ByteAddressBuffer before : register(t1);  // the same bytes before the lists ran
RWByteAddressBuffer home : register(u0);  // title memory, bound at the copy's first byte
RWByteAddressBuffer differs : register(u1);
RWByteAddressBuffer stats : register(u2); // bytes stored so far, in its first dword
groupshared uint group_stored;
uint bytes_of(uint x) {
    return ((x & 0xffu) != 0u ? 0xffu : 0u) | ((x & 0xff00u) != 0u ? 0xff00u : 0u)
        | ((x & 0xff0000u) != 0u ? 0xff0000u : 0u) | ((x & 0xff000000u) != 0u ? 0xff000000u : 0u);
}
// Merges the step at `at`; returns the bytes it stored.
uint merge_step(uint at) {
    if (at >= to) return 0u;
    if (flags != 0xffffffffu && differs.Load(flags + (((at + skew) >> 16) - first) * 4u) == 0u) return 0u;
    uint4 n = now.Load4(at);
    uint4 b = before.Load4(reference + (at - from));
    uint4 m = uint4(bytes_of(n.x ^ b.x), bytes_of(n.y ^ b.y), bytes_of(n.z ^ b.z), bytes_of(n.w ^ b.w));
    if (all(m == 0u)) return 0u;
    uint4 h = home.Load4(at);
    uint4 r = (h & ~m) | (n & m);
    uint stored = 0u;
    if (m.x != 0u) { home.Store(at, r.x); stored += 4u; }
    if (m.y != 0u) { home.Store(at + 4u, r.y); stored += 4u; }
    if (m.z != 0u) { home.Store(at + 8u, r.z); stored += 4u; }
    if (m.w != 0u) { home.Store(at + 12u, r.w); stored += 4u; }
    return stored;
}
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint index : SV_GroupIndex) {
    if (index == 0u) group_stored = 0u;
    GroupMemoryBarrierWithGroupSync();
    uint stored = merge_step(from + id.x * 16u);
    if (stored != 0u) InterlockedAdd(group_stored, stored);
    GroupMemoryBarrierWithGroupSync();
    if (index == 0u && group_stored != 0u) stats.InterlockedAdd(0, group_stored);
}
"#;

/// The most 16-byte steps one dispatch of [`MERGE`] takes (65,535 groups of
/// 64 threads).
const STEPS_PER_DISPATCH: u64 = 65_535 * 64;

/// The most bytes of reference kept in the card's memory; a submission that
/// needs more has it in the PC's memory.
const REFERENCE_CARD_MOST: u64 = 512 << 20;

/// Submissions in a row whose reference would fit the card again before a
/// reference in the PC's memory goes back to the card's ([`room_move`]).
const ROOM_BACK_AFTER: u32 = 16;

/// The most bytes of kept references ([`Kept`]), all copies together.
const KEPT_MOST: u64 = 512 << 20;

/// Submissions with no declared range of its copy after which a kept
/// reference is let go: it costs a copy of what goes up to the copy, which
/// a reference made per submission does not.
const KEPT_IDLE: u64 = 64;

/// Submissions after the card's budget or the card turned a kept reference
/// down before another as big or bigger is tried.
const KEPT_RETRY: u64 = 16;

/// `VICTORY_MEMORY_CARDS=1` with the memory core on (`crate::home::on`), and
/// the merge's pipeline made: copies in the card's memory get a home as
/// their CPU side.
pub(crate) fn on() -> bool {
    switch() && crate::home::on() && merge().is_some()
}

/// `VICTORY_MEMORY_CARDS=1` (read once), or a test's switch.
fn switch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_MEMORY_CARDS").is_some_and(|v| v == "1");
        if on && !crate::home::on() {
            tracing::warn!("VICTORY_MEMORY_CARDS=1 does nothing without VICTORY_MEMORY_CORE=1");
        } else if on {
            tracing::info!(
                "memory core cards on (VICTORY_MEMORY_CARDS=1): copies in the card's memory have the title's own memory as their CPU side where it has a home"
            );
        }
        on
    });
    on || test_on()
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

/// Sets the tests' switch for this thread (`crate::home`'s tests of the
/// pool, which hold the core's switch themselves).
#[cfg(test)]
pub(crate) fn set_test(on: bool) {
    TEST_ON.with(|t| t.set(on));
}

#[cfg(not(test))]
fn test_on() -> bool {
    false
}

/// [`MERGE`]'s root signature and pipeline; the counter its groups add the
/// bytes they stored to (in the card's memory: no atomics over the bus), a
/// mapped buffer in the PC's memory it is copied into behind the merges,
/// and the count last taken from there ([`stored`]).
struct Merge {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    stats: ID3D12Resource,
    seen: ID3D12Resource,
    seen_data: *mut u32,
    taken: AtomicU32,
}

// SAFETY: D3D12 objects (free-threaded); the counter's mapping is read only
// after a submission was waited for, under the submission lock.
unsafe impl Send for Merge {}
// SAFETY: as above.
unsafe impl Sync for Merge {}

/// The merge once asked for: `Some(None)` if it could not be made.
static MERGE_MADE: OnceLock<Option<Merge>> = OnceLock::new();

/// The merge, made at the first ask once there is a host; `None` (and no
/// homes for card copies) if it cannot be made.
fn merge() -> Option<&'static Merge> {
    if let Some(made) = MERGE_MADE.get() {
        return made.as_ref();
    }
    let host = crate::host::host()?;
    MERGE_MADE
        .get_or_init(|| match make_merge(host) {
            Ok(merge) => Some(merge),
            Err(e) => {
                tracing::warn!(error = %e, "VICTORY_MEMORY_CARDS: the merge on the card could not be made; card copies keep their staging buffers");
                None
            }
        })
        .as_ref()
}

fn make_merge(host: &crate::host::Host) -> windows::core::Result<Merge> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; the out pointer
    // is ours.
    unsafe {
        D3DCompile(
            MERGE.as_ptr().cast(),
            MERGE.len(),
            s!("card home merge"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )?;
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // (Committed, so it starts at 0.)
    let stats = host
        .uav_buffer(256)
        .ok_or_else(windows::core::Error::empty)?;
    crate::host::name(&stats, || "card home merge: bytes stored".into());
    let (seen, data) = host
        .shared_buffer(256)
        .ok_or_else(windows::core::Error::empty)?;
    crate::host::name(&seen, || {
        "card home merge: bytes stored, for the CPU".into()
    });
    // SAFETY: the mapping holds the counter's copy; nothing ran with it yet.
    unsafe { data.cast::<u32>().write_volatile(0) };
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes)?;
        let pso: ID3D12PipelineState =
            host.device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
        Ok(Merge {
            root,
            pso,
            stats,
            seen,
            seen_data: data.cast(),
            taken: AtomicU32::new(0),
        })
    }
}

/// The steps a range `[from, to)` of a copy is merged in: widened to
/// multiples of 16 (inside the copy's buffers, which are multiples of
/// 64 KiB, and inside its home: the copy starts on a step of it).
fn steps(from: u32, to: u32) -> (u32, u32) {
    (from & !15, to.saturating_add(15) & !15)
}

/// The reference's buffer: its bytes, and whether it is in the card's
/// memory.
struct Room {
    buffer: ID3D12Resource,
    bytes: u64,
    on_card: bool,
}

/// A copy's reference kept between submissions: what its card side held
/// before the lists, at the copy's own offsets. Brought up to date with
/// what goes up to the card side, behind the upload ([`before`]). Kept by
/// the copy's address ([`Before::kept`]), which the weak hold keeps from
/// being given to another copy.
struct Kept {
    copy: Weak<Shadow>,
    buffer: ID3D12Resource,
    bytes: u64,
    /// The last submission ([`Before::submissions`]) that declared a range
    /// of the copy.
    used: u64,
}

/// Where the card side's bytes of a declared range from before the lists
/// lie, for its merge ([`record_merge`]).
#[derive(Clone)]
pub(crate) struct Place {
    /// The copy's kept reference, or `None`: the submission's room.
    kept: Option<ID3D12Resource>,
    /// Where the range's first step lies in it.
    at: u32,
}

/// The reference, the kept ones, and the list of ours that copies into
/// them, reused by every submission (each is waited for before the next is
/// prepared).
struct Before {
    room: Option<Room>,
    /// Submissions in a row since the room went to the PC's memory whose
    /// reference would have fit the card's ([`room_move`]).
    small: u32,
    kept: BTreeMap<usize, Kept>,
    kept_bytes: u64,
    /// No kept reference of `.1` bytes or more is made before submission
    /// `.0`: the card's budget or the card turned one of `.1` bytes down
    /// ([`KEPT_RETRY`], [`backed_off`]).
    retry: (u64, u64),
    /// Submissions prepared so far that had card copies' references to set
    /// aside or kept references to age ([`before`], [`idle`]).
    submissions: u64,
    list: Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
}

// SAFETY: D3D12 objects (free-threaded), used under the submission lock.
unsafe impl Send for Before {}

static BEFORE: Mutex<Before> = Mutex::new(Before {
    room: None,
    small: 0,
    kept: BTreeMap::new(),
    kept_bytes: 0,
    retry: (0, 0),
    submissions: 0,
    list: None,
});

/// What [`room`] does with the reference there is for a submission that
/// needs `need` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RoomMove {
    /// It holds them where it is.
    Keep,
    /// A bigger one (or the first).
    Grow,
    /// It is in the PC's memory and the need fit the card's for
    /// [`ROOM_BACK_AFTER`] submissions in a row: it is made in the card's
    /// memory again.
    Back,
}

/// [`RoomMove`] for a room of `room` bytes (`on_card`: in the card's
/// memory), `small` submissions in a row having fit the card's before this
/// one. Also returns the new count.
fn room_move(room: Option<(u64, bool)>, need: u64, small: u32) -> (RoomMove, u32) {
    let Some((bytes, on_card)) = room else {
        return (RoomMove::Grow, 0);
    };
    if bytes < need {
        return (RoomMove::Grow, 0);
    }
    if on_card || need > REFERENCE_CARD_MOST {
        return (RoomMove::Keep, 0);
    }
    let small = small + 1;
    if small >= ROOM_BACK_AFTER {
        (RoomMove::Back, 0)
    } else {
        (RoomMove::Keep, small)
    }
}

/// Whether the card's budget has room for `bytes` more.
fn card_has_room(host: &crate::host::Host, bytes: u64) -> bool {
    host.video_memory()
        .is_none_or(|[budget, used, ..]| used + bytes <= budget / 100 * 97)
}

/// A buffer of at least `need` bytes for the reference: the one there is,
/// or a bigger one (in the card's memory while it is at most
/// [`REFERENCE_CARD_MOST`] and the card has room, else in the PC's memory).
/// One in the PC's memory goes back to the card's once the need shrank
/// ([`room_move`]); if the card has no room then, it stays.
fn room<'a>(host: &crate::host::Host, before: &'a mut Before, need: u64) -> Option<&'a Room> {
    let (step, small) = room_move(
        before.room.as_ref().map(|r| (r.bytes, r.on_card)),
        need,
        before.small,
    );
    before.small = small;
    match step {
        RoomMove::Keep => {}
        RoomMove::Back => {
            let bytes = need.next_multiple_of(16 << 20);
            if card_has_room(host, bytes)
                && let Some(buffer) = host.uav_buffer(bytes)
            {
                // The old one is no longer read: its last submission was
                // waited for.
                ROOM_MOVES.fetch_add(1, Ordering::Relaxed);
                set_room(
                    before,
                    Room {
                        buffer,
                        bytes,
                        on_card: true,
                    },
                );
            }
        }
        RoomMove::Grow => {
            // The old one is no longer read: its last submission was waited for.
            before.room = None;
            let bytes = need.next_multiple_of(16 << 20);
            let on_card = (bytes <= REFERENCE_CARD_MOST && card_has_room(host, bytes))
                .then(|| host.uav_buffer(bytes))
                .flatten();
            let room = match on_card {
                Some(buffer) => Room {
                    buffer,
                    bytes,
                    on_card: true,
                },
                // Not mapped by us: the card writes and reads it, nothing else.
                None => Room {
                    buffer: host.shared_buffer(bytes)?.0,
                    bytes,
                    on_card: false,
                },
            };
            set_room(before, room);
        }
    }
    before.room.as_ref()
}

/// `room` is the reference from now on.
fn set_room(before: &mut Before, room: Room) {
    crate::host::label(&room.buffer, "card home merge reference", room.bytes);
    crate::host::name(&room.buffer, || {
        "card copies' bytes before the lists, for the merge into their homes".into()
    });
    ROOM_BYTES.store(
        room.bytes | u64::from(room.on_card) << 63,
        Ordering::Relaxed,
    );
    before.room = Some(room);
}

/// The kept reference of `copy` (made now if it has none, its declared
/// steps `declared` bytes are half of it or more, and the kept references
/// and the card have room), marked used by submission `now`; and whether it
/// was made now (it is then filled whole).
fn kept_for(
    host: &crate::host::Host,
    before: &mut Before,
    copy: &Arc<Shadow>,
    declared: u64,
    now: u64,
) -> Option<(ID3D12Resource, bool)> {
    let key = Arc::as_ptr(copy) as usize;
    if let Some(kept) = before.kept.get_mut(&key) {
        kept.used = now;
        return Some((kept.buffer.clone(), false));
    }
    let (.., bytes) = copy.card_homed()?;
    if declared * 2 < bytes
        || before.kept_bytes + bytes > KEPT_MOST
        || backed_off(before.retry, now, bytes)
    {
        return None;
    }
    // Asking the card's budget, or a buffer it turns down, is not done every
    // submission for one as big; a smaller one may still fit.
    let Some(buffer) = card_has_room(host, bytes)
        .then(|| host.uav_buffer(bytes))
        .flatten()
    else {
        before.retry = (now + KEPT_RETRY, bytes);
        return None;
    };
    crate::host::label(&buffer, "card home kept reference", bytes);
    crate::host::name(&buffer, || {
        "a card copy's bytes before the lists, kept, for the merge into its home".into()
    });
    before.kept.insert(
        key,
        Kept {
            copy: Arc::downgrade(copy),
            buffer: buffer.clone(),
            bytes,
            used: now,
        },
    );
    before.kept_bytes += bytes;
    KEPT[0].fetch_add(1, Ordering::Relaxed);
    Some((buffer, true))
}

/// Whether no kept reference of `bytes` is asked for at submission `now`:
/// one as big or bigger was turned down ([`Before::retry`]) less than
/// [`KEPT_RETRY`] submissions ago.
fn backed_off((until, refused): (u64, u64), now: u64, bytes: u64) -> bool {
    now < until && bytes >= refused
}

/// Lets go the kept references of copies that are gone or that nothing
/// declared for [`KEPT_IDLE`] submissions. (A submission that still reads
/// one holds its buffer, [`Place`].)
fn let_go_kept(before: &mut Before) {
    let now = before.submissions;
    let mut freed = 0;
    before.kept.retain(|_, k| {
        let keep = k.copy.strong_count() > 0 && now - k.used <= KEPT_IDLE;
        if !keep {
            freed += k.bytes;
        }
        keep
    });
    if freed != 0 {
        before.kept_bytes -= freed;
        KEPT[1].fetch_add(1, Ordering::Relaxed);
    }
    KEPT_BYTES.store(before.kept_bytes, Ordering::Relaxed);
}

/// A submission that needs no [`before`] (`Host::card_before`): while there
/// are kept references, it counts towards letting go those nothing
/// declares, and those of copies that are gone are let go now.
pub(crate) fn idle() {
    if KEPT_BYTES.load(Ordering::Relaxed) == 0 {
        return;
    }
    let mut before = BEFORE.lock().unwrap_or_else(|e| e.into_inner());
    before.submissions += 1;
    let_go_kept(&mut before);
}

/// Before a submission's lists, once what changed went up to the card sides
/// (`Host::card_before`): records a list of ours that sets aside the card
/// side's bytes of each range in `spans` (a copy whose CPU side is a home,
/// and the range's offsets in it) for the merge [`record_merge`] makes.
///
/// A copy with a kept reference ([`Kept`]), or one made now (its ranges are
/// half of it or more), has nothing of its ranges copied: its kept reference
/// takes `uploads`' ranges of it (what just went up to its card side), or
/// all of it when made now. The others' ranges are copied into the room.
///
/// Returns the closed list, to run ahead of the lists (`None`: nothing to
/// copy), and where each range lies; `None` if there is no memory for it.
pub(crate) fn before(
    host: &crate::host::Host,
    spans: &[(&Arc<Shadow>, u32, u32)],
    uploads: &[(Arc<Shadow>, Vec<(usize, usize)>)],
) -> Option<(Option<ID3D12CommandList>, Vec<Place>)> {
    let mut before = BEFORE.lock().unwrap_or_else(|e| e.into_inner());
    before.submissions += 1;
    let now = before.submissions;
    // The copies, each once, and the bytes of their declared steps.
    let mut declared: HashMap<usize, (&Arc<Shadow>, u64)> = HashMap::new();
    for &(copy, from, to) in spans {
        let (from, to) = steps(from, to);
        declared
            .entry(Arc::as_ptr(copy) as usize)
            .or_insert((copy, 0))
            .1 += u64::from(to - from);
    }
    // In one order from run to run (the room for kept references goes to
    // the most declared first), not the map's; copies of one span alike.
    let mut declared: Vec<(usize, (&Arc<Shadow>, u64))> = declared.into_iter().collect();
    declared.sort_unstable_by_key(|&(_, (copy, bytes))| (std::cmp::Reverse(bytes), copy.span()));
    // The kept references still declared are used now; then those let go
    // leave their room to the copies that may get one now.
    for (key, _) in &declared {
        if let Some(kept) = before.kept.get_mut(key) {
            kept.used = now;
        }
    }
    let_go_kept(&mut before);
    // Per copy, its kept reference and whether it is filled now.
    let copies: HashMap<usize, (&Arc<Shadow>, Option<(ID3D12Resource, bool)>)> = declared
        .into_iter()
        .map(|(key, (copy, bytes))| (key, (copy, kept_for(host, &mut before, copy, bytes, now))))
        .collect();
    KEPT_BYTES.store(before.kept_bytes, Ordering::Relaxed);
    let kept_of = |copy: &Arc<Shadow>| {
        copies
            .get(&(Arc::as_ptr(copy) as usize))
            .and_then(|(_, k)| k.as_ref())
    };
    let mut places = Vec::with_capacity(spans.len());
    let mut need = 0u64;
    let mut skipped = 0u64;
    for &(copy, from, to) in spans {
        let (from, to) = steps(from, to);
        match kept_of(copy) {
            Some((buffer, _)) => {
                places.push(Place {
                    kept: Some(buffer.clone()),
                    at: from,
                });
                skipped += u64::from(to - from);
            }
            None => {
                // A submission's ranges lie in copies of at most 1 GiB each,
                // and each copy's ranges are apart (`Host::card_before` joins
                // them).
                places.push(Place {
                    kept: None,
                    at: u32::try_from(need).ok()?,
                });
                need += u64::from(to - from);
            }
        }
    }
    if need > u64::from(u32::MAX) {
        return None;
    }
    // What goes into kept references: copies filled now, whole; the others
    // kept, what went up to them.
    let mut fills: Vec<(ID3D12Resource, &ID3D12Resource, u64, u64)> = Vec::new();
    for (copy, kept) in copies.values() {
        if let (Some((buffer, true)), Some((.., bytes))) = (kept, copy.card_homed()) {
            fills.push((buffer.clone(), &copy.buffer, 0, bytes));
        }
    }
    for (copy, ranges) in uploads {
        let key = Arc::as_ptr(copy) as usize;
        if kept_of(copy).is_some_and(|(_, filled)| *filled) {
            continue;
        }
        let Some(kept) = before.kept.get(&key) else {
            continue;
        };
        for &(a, b) in ranges {
            fills.push((kept.buffer.clone(), &copy.buffer, a as u64, (b - a) as u64));
        }
    }
    if need == 0 && fills.is_empty() {
        FLOW[flow::SKIPPED].fetch_add(skipped, Ordering::Relaxed);
        return Some((None, places));
    }
    if before.list.is_none() {
        // SAFETY: plain object creation.
        let made = unsafe {
            (|| -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
                let allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        before.list = made.ok();
    }
    let room = match need {
        0 => None,
        _ => Some(room(host, &mut before, need)?.buffer.clone()),
    };
    let (allocator, list) = before.list.as_ref()?;
    // SAFETY: the list's last run finished (its submission was waited for
    // before this one was prepared); live buffers, the ranges inside both
    // (the steps lie in the copy's buffer and the room holds `need`; a kept
    // reference has its copy's size, and what went up lies in the copy).
    let recorded = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            for (&(copy, from, to), place) in spans.iter().zip(&places) {
                let (Some(room), None) = (&room, &place.kept) else {
                    continue;
                };
                let (from, to) = steps(from, to);
                list.CopyBufferRegion(
                    room,
                    u64::from(place.at),
                    &copy.buffer,
                    u64::from(from),
                    u64::from(to - from),
                );
            }
            for (kept, card, at, bytes) in &fills {
                list.CopyBufferRegion(kept, *at, *card, *at, *bytes);
            }
            list.Close()?;
            list.cast()
        })()
    };
    FLOW[flow::REFERENCE].fetch_add(need, Ordering::Relaxed);
    FLOW[flow::KEPT].fetch_add(fills.iter().map(|f| f.3).sum(), Ordering::Relaxed);
    FLOW[flow::SKIPPED].fetch_add(skipped, Ordering::Relaxed);
    recorded.ok().map(|list| (Some(list), places))
}

/// A barrier on every buffer's unordered access.
fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// Records into `list` (the compare's, `Host::card_before`), behind its
/// compares and one barrier, the merge of each of `ranges` (a copy whose CPU
/// side is a home, its declared range, where [`before`] put it, and its
/// flags as [`record_merge`] takes them), and then the copy of the count of
/// bytes stored to where [`stored`] reads it. Nothing if there are none.
///
/// Between the merges of two different copies, another barrier: copies may
/// lie over the same home words (a big buffer inside another, whose home is
/// the outer one's), and a merge that read a word before the other stored
/// it would store back the old bytes of what the other's shaders wrote.
/// (The ranges of one copy are apart; their widened steps may share a word,
/// which both merges make the same.)
///
/// # Safety
///
/// As [`record_merge`].
pub(crate) unsafe fn record_merges<'a>(
    list: &ID3D12GraphicsCommandList,
    differs: &ID3D12Resource,
    ranges: impl Iterator<Item = (&'a Shadow, &'a CardSpan, &'a Place, Option<u32>)>,
) {
    let mut ranges = ranges.peekable();
    if ranges.peek().is_none() {
        return;
    }
    crate::profile::count(crate::profile::Count::UavCard, 1);
    // SAFETY: the caller's: an open list; what the compares wrote first.
    unsafe { list.ResourceBarrier(&[uav_barrier()]) };
    let mut last: Option<*const Shadow> = None;
    for (copy, span, reference, flags) in ranges {
        if last.is_some_and(|last| !std::ptr::eq(last, copy)) {
            crate::profile::count(crate::profile::Count::UavCard, 1);
            // SAFETY: the caller's: an open list.
            unsafe { list.ResourceBarrier(&[uav_barrier()]) };
        }
        // SAFETY: the caller's.
        if unsafe { record_merge(list, copy, span, reference, flags, differs) } {
            last = Some(copy);
        }
    }
    let Some(merge) = merge().filter(|_| last.is_some()) else {
        return;
    };
    // SAFETY: an open list; the counter was used for unordered access in it
    // (by the merges just recorded), and decays at the end of the
    // submission as buffers do.
    unsafe {
        list.ResourceBarrier(&[crate::host::transition(
            &merge.stats,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&merge.seen, 0, &merge.stats, 0, 4);
    }
}

/// Records into `list`, behind the compare whose results are `differs`, the
/// merge of declared range `span` of `copy` (a copy whose CPU side is a
/// home) into the home: `reference` is where [`before`] put the range's
/// bytes from before the lists (the room, or the copy's kept reference),
/// `flags` the dword of its first piece in
/// `differs` (in bytes) if the compare looked at it (else every piece is
/// merged). Nothing for a copy without a home; whether it was recorded.
/// [`record_merges`] records the merges of a submission.
///
/// # Safety
///
/// `list` is open and runs after the submission's lists and the compare,
/// in the same submission as the list [`before`] gave.
unsafe fn record_merge(
    list: &ID3D12GraphicsCommandList,
    copy: &Shadow,
    span: &CardSpan,
    reference: &Place,
    flags: Option<u32>,
    differs: &ID3D12Resource,
) -> bool {
    let (Some((home, bias, _)), Some(merge)) = (copy.card_homed(), merge()) else {
        return false;
    };
    let room = match &reference.kept {
        Some(kept) => kept.clone(),
        None => {
            let Some(room) = BEFORE
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .room
                .as_ref()
                .map(|r| r.buffer.clone())
            else {
                return false;
            };
            room
        }
    };
    let reference = reference.at;
    let (from, to) = steps(span.from, span.to);
    FLOW[flow::RANGES].fetch_add(1, Ordering::Relaxed);
    // SAFETY: the caller's; root views of live buffers, and every step the
    // shader takes lies in the copy's card side, the reference and the home
    // (`steps`). The home is bound at the copy's first byte (`bias`, a
    // multiple of 16), so the shader's offsets are the copy's, which fit in
    // 32 bits (a copy has at most 1 GiB) wherever the copy lies in the home.
    unsafe {
        list.SetComputeRootSignature(&merge.root);
        list.SetPipelineState(&merge.pso);
        list.SetComputeRootShaderResourceView(1, copy.buffer.GetGPUVirtualAddress());
        list.SetComputeRootShaderResourceView(2, room.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(3, home.buffer.GetGPUVirtualAddress() + bias);
        list.SetComputeRootUnorderedAccessView(4, differs.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(5, merge.stats.GetGPUVirtualAddress());
        let mut at = from;
        while at < to {
            let n = u64::from(to - at).min(STEPS_PER_DISPATCH * 16) as u32;
            let constants = [
                at,
                at + n,
                reference + (at - from),
                0,
                flags.unwrap_or(u32::MAX),
                span.first,
                span.skew,
                0,
            ];
            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
            list.Dispatch((n / 16).div_ceil(64), 1, 1);
            at += n;
        }
    }
    true
}

/// After the wait (`Host::card_back`): the merges of the submission stored
/// into piece `[from, to)` of `copy` (or may have); see
/// `Shadow::card_merged`.
pub(crate) fn merged(copy: &Shadow, from: usize, to: usize) {
    FLOW[flow::PIECES].fetch_add(1, Ordering::Relaxed);
    copy.card_merged(from, to);
}

/// After the wait: the bytes the merges stored into homes since the last
/// ask. (The counter on the card only grows, modulo 2^32; a submission
/// stores far less than 4 GiB.)
pub(crate) fn stored() -> u64 {
    let Some(merge) = merge() else {
        return 0;
    };
    // SAFETY: the counter's copy's mapping; the merges that add to it have
    // run (their submission was waited for) and the next is not prepared.
    let now = unsafe { merge.seen_data.read_volatile() };
    let bytes = now.wrapping_sub(merge.taken.swap(now, Ordering::Relaxed));
    FLOW[flow::STORED].fetch_add(u64::from(bytes), Ordering::Relaxed);
    u64::from(bytes)
}

/// Since the start: card copies made with a home as their CPU side; of them
/// alive, and their bytes (staging buffers not made); card copies that kept
/// a staging buffer (their bytes got no home); of them those whose home
/// they would have started off a 16-byte step in.
static CARDS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// [`FLOW`]'s counters.
mod flow {
    /// Bytes that went up from homes to card sides.
    pub(super) const UPLOADED: usize = 0;
    /// Bytes of declared ranges copied into the room.
    pub(super) const REFERENCE: usize = 1;
    /// Ranges merged into homes on the card.
    pub(super) const RANGES: usize = 2;
    /// Pieces of them merged.
    pub(super) const PIECES: usize = 3;
    /// Bytes the merges stored into homes.
    pub(super) const STORED: usize = 4;
    /// Bytes copied into kept references (what went up to their copies, and
    /// the copies whose kept reference was made, whole).
    pub(super) const KEPT: usize = 5;
    /// Bytes of declared ranges not copied: their merge reads a kept
    /// reference.
    pub(super) const SKIPPED: usize = 6;
    pub(super) const ALL: usize = 7;
}

/// Since the last line: see [`flow`].
static FLOW: [AtomicU64; flow::ALL] = [const { AtomicU64::new(0) }; flow::ALL];

/// The reference's bytes, with bit 63 set when it is in the card's memory.
static ROOM_BYTES: AtomicU64 = AtomicU64::new(0);

/// Times the room went back from the PC's memory to the card's.
static ROOM_MOVES: AtomicU64 = AtomicU64::new(0);

/// Since the start: kept references made, and times some were let go.
static KEPT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The kept references' bytes now.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);

/// A card copy of `size` bytes was made with a home as its CPU side.
pub(crate) fn made(size: u64) {
    CARDS[0].fetch_add(1, Ordering::Relaxed);
    CARDS[1].fetch_add(1, Ordering::Relaxed);
    CARDS[2].fetch_add(size, Ordering::Relaxed);
}

/// One such copy (`size` bytes) is gone.
pub(crate) fn gone(size: u64) {
    CARDS[1].fetch_sub(1, Ordering::Relaxed);
    CARDS[2].fetch_sub(size, Ordering::Relaxed);
}

/// A card copy keeps its staging buffer: its bytes got no home.
pub(crate) fn kept() {
    if on() {
        CARDS[3].fetch_add(1, Ordering::Relaxed);
    }
}

/// A card copy keeps its staging buffer: it would start off a 16-byte step
/// in its home.
pub(crate) fn misaligned() {
    CARDS[4].fetch_add(1, Ordering::Relaxed);
}

/// `bytes` went up from homes to card sides.
pub(crate) fn uploaded(bytes: u64) {
    FLOW[flow::UPLOADED].fetch_add(bytes, Ordering::Relaxed);
}

/// The core's ten-second line's part for card copies (`crate::home`), with
/// the switch on. It runs under the core's lock, so it only looks whether
/// the merge was made and never makes it.
pub(crate) fn line() -> Option<String> {
    if !switch() || !crate::home::on() {
        return None;
    }
    let cards: Vec<u64> = CARDS.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let moved: Vec<u64> = FLOW.iter().map(|f| f.swap(0, Ordering::Relaxed)).collect();
    let room = ROOM_BYTES.load(Ordering::Relaxed);
    Some(format!(
        "memory core cards: {} copies in the card's memory have their home as the CPU side ({} made), {} MB of staging not in the PC's memory; \
         {} kept a staging buffer ({} for their alignment in the home){}; in the last ten seconds {} MB went up from homes, {} MB of reference \
         copied on the card, {} MB into kept references, {} MB of declared ranges not copied (their merge read a kept reference), \
         {} ranges merged into homes on the card ({} pieces, {} KB stored); reference {} MB in the {} memory ({} times back to the card's), \
         {} MB of kept references ({} made, let go {} times)",
        cards[1],
        cards[0],
        cards[2] >> 20,
        cards[3],
        cards[4],
        if MERGE_MADE.get().is_some_and(Option::is_none) {
            ", the merge could not be made"
        } else {
            ""
        },
        moved[flow::UPLOADED] >> 20,
        moved[flow::REFERENCE] >> 20,
        moved[flow::KEPT] >> 20,
        moved[flow::SKIPPED] >> 20,
        moved[flow::RANGES],
        moved[flow::PIECES],
        moved[flow::STORED] >> 10,
        (room & !(1 << 63)) >> 20,
        if room >> 63 == 1 { "card's" } else { "PC's" },
        ROOM_MOVES.load(Ordering::Relaxed),
        KEPT_BYTES.load(Ordering::Relaxed) >> 20,
        KEPT[0].load(Ordering::Relaxed),
        KEPT[1].load(Ordering::Relaxed),
    ))
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;
    use std::sync::Arc;

    use windows_sys::Win32::System::Memory as mem;

    use super::*;
    use crate::inplace::Rig;
    use crate::shadow::{card_dirty, find, register, unregister};

    const K64: u64 = 64 << 10;
    /// The host's write watching (not in windows-sys' memory module).
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    /// A card copy's size at least (`card_limits`' default: 1 MiB).
    const BIG: u64 = 1 << 20;

    /// The memory core and the switch on for this thread, the core's other
    /// tests waiting.
    struct Switched {
        _core: crate::home::tests::Switched,
    }

    impl Switched {
        fn on() -> Switched {
            let core = crate::home::tests::Switched::on(false);
            TEST_ON.with(|on| on.set(true));
            Switched { _core: core }
        }
    }

    impl Drop for Switched {
        fn drop(&mut self) {
            TEST_ON.with(|on| on.set(false));
        }
    }

    /// Title memory as `xmem` reserves it (committed, read-write,
    /// write-watched), told to the core as a region.
    struct Region {
        base: u64,
        bytes: usize,
    }

    impl Region {
        fn new(bytes: usize) -> Region {
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
            crate::home::title_memory_region(
                base as usize,
                bytes,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                3,
                0x404,
            );
            Region { base, bytes }
        }

        /// Forgets what the title wrote so far.
        fn reset_watch(&self) {
            // SAFETY: the test's own allocation.
            unsafe { mem::ResetWriteWatch(self.base as *const c_void, self.bytes) };
        }

        /// Released through the core, as the title's `VirtualFree` is.
        fn release(self) {
            assert_eq!(
                crate::home::title_virtual_free(self.base as usize, 0, mem::MEM_RELEASE),
                1
            );
        }
    }

    fn fill(at: u64, len: u64, seed: u8) {
        for i in 0..len {
            // SAFETY: committed memory of the test's own.
            unsafe { ((at + i) as *mut u8).write((i as u8).wrapping_mul(29).wrapping_add(seed)) };
        }
    }

    fn title(at: u64, len: u64) -> Vec<u8> {
        // SAFETY: committed memory of the test's own.
        unsafe { std::slice::from_raw_parts(at as *const u8, len as usize).to_vec() }
    }

    fn state(at: u64) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: mem::MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer.
        unsafe { mem::VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
        info.State
    }

    /// The host on the software adapter, if it has what card copies with a
    /// home need: `ID3D12Device13` and the card's work.
    fn warp() -> Option<&'static crate::host::Host> {
        let host = crate::host::host()?;
        (host.device.cast::<ID3D12Device13>().is_ok() && host.card_ready()).then_some(host)
    }

    /// A card copy of title buffer `[start, start + len)`, registered as
    /// resource `id`: with the switches on, its CPU side is a home.
    fn card_copy(id: usize, start: u64, len: u64) -> Arc<Shadow> {
        register(id, start, len);
        let copy = find(start, len).expect("a copy");
        assert!(copy.carded(), "in the card's memory");
        copy
    }

    /// The ranges waiting to go up for `copy`, taken.
    fn dirty_of(copy: &Arc<Shadow>) -> Vec<(usize, usize)> {
        card_dirty(0)
            .into_iter()
            .filter(|(s, _)| Arc::ptr_eq(s, copy))
            .flat_map(|(_, r)| r)
            .collect()
    }

    /// Runs the upload of `ranges` from the CPU side to the card side, as
    /// `Host::card_before` records it.
    fn upload(rig: &mut Rig, copy: &Shadow, ranges: &[(usize, usize)]) {
        // SAFETY: an open list of the rig's; the ranges lie in the copy.
        rig.run(|list| unsafe {
            copy.card_upload(list, ranges);
        })
        .unwrap();
    }

    /// With the switches on, a big buffer's card copy has its home as the
    /// CPU side: no staging buffer. A refresh copies nothing; only the pages
    /// the title wrote are noted, and the upload takes them from the home:
    /// a page the title did not write is not sent up again.
    #[test]
    fn warp_cards_upload_only_the_written_pages() {
        let Some(host) = warp() else { return };
        let _on = Switched::on();
        let region = Region::new(4 * BIG as usize);
        let (start, len) = (region.base + K64 + 0x100, BIG + 3 * K64);
        fill(start, len, 1);
        let copy = card_copy(0xca4d_0101, start, len);
        let (home, at, _) = copy.card_homed().expect("its CPU side is a home");
        let home = home.clone();
        assert!(home.is_home() && copy.is_home());
        assert_eq!(home.offset(start), at);
        let mut rig = Rig::new(&host.device).unwrap();
        // The first refresh takes all of it, and copies none of it.
        assert_eq!(
            copy.update(start as usize, len as usize),
            0,
            "nothing copied on the CPU"
        );
        let all = dirty_of(&copy);
        assert_eq!(all, [(0, len as usize)]);
        upload(&mut rig, &copy, &all);
        assert_eq!(
            rig.fetch(&copy.buffer, 0, len as usize).unwrap(),
            title(start, len),
            "the card side holds the title's bytes"
        );
        // The title writes one page; meanwhile the card side of another
        // page is changed behind the copy's back.
        region.reset_watch();
        let page = (start + 5 * K64) & !(K64 - 1);
        fill(page + 0x10, 0x40, 77);
        let other = (page + 3 * K64 - start) as usize;
        rig.store(&copy.buffer, other as u64, &[0xee; 32]).unwrap();
        assert_eq!(copy.update(start as usize, len as usize), 0);
        let written = dirty_of(&copy);
        let lo = (page - start) as usize;
        assert_eq!(
            written,
            [(lo, lo + K64 as usize)],
            "only the page the title wrote"
        );
        upload(&mut rig, &copy, &written);
        let card = rig.fetch(&copy.buffer, 0, len as usize).unwrap();
        assert_eq!(
            card[lo..lo + K64 as usize],
            title(page, K64)[..],
            "the written page went up"
        );
        assert_eq!(
            card[other..other + 32],
            [0xee; 32],
            "a page the title did not write was not sent up"
        );
        unregister(0xca4d_0101, start);
        drop((home, copy, rig));
        region.release();
    }

    /// What the merge on the card stores into the home is what the CPU's
    /// merge (`merge_piece`) makes of the same bytes: the card side's byte
    /// where the shaders changed it, the title's elsewhere, also in a word
    /// both changed. With the compare's flags only the flagged pieces are
    /// merged.
    #[test]
    fn warp_cards_gpu_merge_equals_the_cpu_merge() {
        let Some(host) = warp() else { return };
        let _on = Switched::on();
        let region = Region::new(4 * BIG as usize);
        let (start, len) = (region.base + 2 * K64 + 0x40, BIG + K64);
        fill(start, len, 3);
        let copy = card_copy(0xca4d_0201, start, len);
        let (home, ..) = copy.card_homed().expect("its CPU side is a home");
        let home = home.clone();
        let mut rig = Rig::new(&host.device).unwrap();
        copy.update(start as usize, len as usize);
        upload(&mut rig, &copy, &dirty_of(&copy));
        // The declared range: pieces 1 to 4 of the copy, cut off mid-step.
        let span = copy
            .card_span(
                (start + K64 - 0x40 + 8) as usize,
                (start + 5 * K64 - 0x40 + 0x1f3) as usize,
            )
            .unwrap();
        let (from, to) = (span.from as usize, span.to as usize);
        let (list, places) =
            before(host, &[(&copy, span.from, span.to)], &[]).expect("a reference");
        assert!(places[0].kept.is_none(), "a quarter of the copy: the room");
        rig.run_closed(&list.expect("a copy into the room"))
            .unwrap();
        let held = rig.fetch(&copy.buffer, 0, len as usize).unwrap();
        let room = BEFORE
            .lock()
            .unwrap()
            .room
            .as_ref()
            .map(|r| r.buffer.clone())
            .unwrap();
        let (a, b) = steps(span.from, span.to);
        let (a, b) = (a as usize, b as usize);
        assert_eq!(
            rig.fetch(&room, u64::from(places[0].at), b - a).unwrap(),
            held[a..b],
            "the reference holds the card side's steps"
        );
        // The shaders write bytes and words in the range (and one beside
        // it); the title writes others while the card runs, one byte in a
        // word a shader writes too.
        let writes: [(usize, &[u8]); 5] = [
            (from + 3, &[0x11]),
            (from + 0x100, &[0x22; 8]),
            (from + K64 as usize + 0x2001, &[0x33, 0x34]),
            (to - 2, &[0x44]),
            (to + 0x20, &[0x55]),
        ];
        for (at, bytes) in writes {
            rig.store(&copy.buffer, at as u64, bytes).unwrap();
        }
        fill(start + from as u64 + 0x101, 1, 0x66);
        fill(start + from as u64 + 0x5000, 0x30, 0x67);
        let now = rig.fetch(&copy.buffer, 0, len as usize).unwrap();
        // The CPU's merge over the widened steps.
        let beside = title(start + to as u64 + 0x20, 1);
        let mut expected = title(start + a as u64, (b - a) as u64);
        let mut reference = held[a..b].to_vec();
        crate::shadow::merge_piece(&now[a..b], &mut reference, &mut expected);
        let flags = host.shared_buffer(K64).unwrap();
        stored();
        // SAFETY: an open list of the rig's; the merge runs alone.
        rig.run(|list| unsafe {
            record_merges(
                list,
                &flags.0,
                std::iter::once((&*copy, &span, &places[0], None)),
            );
        })
        .unwrap();
        assert_eq!(
            title(start + a as u64, (b - a) as u64),
            expected,
            "the card's merge is the CPU's"
        );
        assert_eq!(
            title(start + to as u64 + 0x20, 1),
            beside,
            "past the steps nothing is merged"
        );
        assert_eq!(
            stored(),
            4 * 5,
            "five words stored (two for the eight bytes, one for each other change)"
        );
        // Again with the compare's flags (the second piece not flagged):
        // only the flagged pieces' changes reach the home.
        let title_before = title(start + a as u64, (b - a) as u64);
        rig.store(&copy.buffer, (from + 0x10) as u64, &[0x77])
            .unwrap();
        rig.store(&copy.buffer, (from + K64 as usize + 0x3000) as u64, &[0x78])
            .unwrap();
        // SAFETY: the mapping holds 64 KiB; the card does not use it now.
        unsafe {
            let words = flags.1.cast::<u32>();
            for p in 0..span.pieces as usize {
                words.add(p).write(u32::from(p != 1));
            }
        }
        // SAFETY: the merge alone, with the same reference: the first
        // merge's bytes differ from it too and are stored again, unchanged.
        rig.run(|list| unsafe {
            record_merges(
                list,
                &flags.0,
                std::iter::once((&*copy, &span, &places[0], Some(0))),
            );
        })
        .unwrap();
        let after = title(start + a as u64, (b - a) as u64);
        assert_eq!(after[from + 0x10 - a], 0x77, "a flagged piece is merged");
        let skipped = from + K64 as usize + 0x3000 - a;
        assert_eq!(
            after[skipped], title_before[skipped],
            "a piece not flagged is not"
        );
        stored();
        unregister(0xca4d_0201, start);
        drop((home, copy, rig));
        region.release();
    }

    /// Without a home the card copy keeps its staging buffer, as before: the
    /// switch off, memory that is no region of the title's, or a buffer that
    /// would start off a 16-byte step of its home.
    #[test]
    fn warp_cards_fallback_when_no_home() {
        let Some(_host) = warp() else { return };
        // The core on, the switch off.
        {
            let _core = crate::home::tests::Switched::on(false);
            let region = Region::new(4 * BIG as usize);
            let start = region.base + K64;
            let copy = card_copy(0xca4d_0301, start, BIG);
            assert!(copy.card_homed().is_none() && !copy.is_home());
            unregister(0xca4d_0301, start);
            drop(copy);
            region.release();
        }
        let _on = Switched::on();
        // Not a region: process memory the core was never told of.
        // SAFETY: an allocation of the test's own.
        let base = unsafe {
            mem::VirtualAlloc(
                std::ptr::null(),
                4 * BIG as usize,
                mem::MEM_RESERVE | mem::MEM_COMMIT,
                mem::PAGE_READWRITE,
            )
        } as u64;
        assert_ne!(base, 0);
        let kept = CARDS[3].load(Ordering::Relaxed);
        let copy = card_copy(0xca4d_0302, base + K64, BIG);
        assert!(copy.card_homed().is_none(), "no region, no home");
        assert_eq!(CARDS[3].load(Ordering::Relaxed), kept + 1, "counted");
        unregister(0xca4d_0302, base + K64);
        drop(copy);
        // Off a 16-byte step.
        let region = Region::new(4 * BIG as usize);
        let start = region.base + K64 + 8;
        let misaligned = CARDS[4].load(Ordering::Relaxed);
        let copy = card_copy(0xca4d_0303, start, BIG);
        assert!(copy.card_homed().is_none());
        assert_eq!(CARDS[4].load(Ordering::Relaxed), misaligned + 1);
        unregister(0xca4d_0303, start);
        drop(copy);
        region.release();
        // SAFETY: the test's own allocation.
        unsafe { mem::VirtualFree(base as *mut c_void, 0, mem::MEM_RELEASE) };
    }

    /// The title decommits a home's pages while a card copy whose CPU side
    /// it is still lives (held as a recorded list holds what it reads): the
    /// copy goes from the tables with the home, the title is told the free
    /// was done, the pages stay committed while the copy holds the heap (its
    /// upload still reads them), and they are decommitted at the sweep after
    /// the copy is gone.
    #[test]
    fn warp_cards_release_of_a_home_while_a_card_copy_refers_to_it() {
        let Some(host) = warp() else { return };
        let _on = Switched::on();
        let region = Region::new(4 * BIG as usize);
        let (start, len) = (region.base + K64, BIG);
        fill(start, len, 9);
        let copy = card_copy(0xca4d_0401, start, len);
        assert!(copy.card_homed().is_some());
        assert_eq!(
            crate::home::title_virtual_free(start as usize, len as usize, mem::MEM_DECOMMIT),
            1,
            "told it was done"
        );
        assert_eq!(
            state(start),
            mem::MEM_COMMIT,
            "still committed under the heap"
        );
        assert!(
            crate::shadow::tabled_copy(start).is_none(),
            "out of the tables"
        );
        // Still used: its upload reads the home.
        let mut rig = Rig::new(&host.device).unwrap();
        upload(&mut rig, &copy, &[(0, 4096)]);
        assert_eq!(
            rig.fetch(&copy.buffer, 0, 4096).unwrap(),
            title(start, 4096)
        );
        crate::home::tick();
        assert_eq!(state(start), mem::MEM_COMMIT, "held by the copy");
        drop((copy, rig));
        crate::home::tick();
        assert_eq!(
            state(start),
            mem::MEM_RESERVE,
            "decommitted once the copy let the heap go"
        );
        unregister(0xca4d_0401, start);
        region.release();
    }

    /// Through `Host::execute`, as a submission of the title's goes: two
    /// copies in the card's memory over one home (a big buffer inside
    /// another, whose home is the outer one's), the list's writes to both
    /// card sides (as shaders' writes, declared), two of them to different
    /// bytes of one word of title memory, and a write of the title's to
    /// another byte of a word they write while the lists run (through the
    /// home, as another view of its memory would). After it title memory
    /// holds every one of them and nothing else changed; the next
    /// submission sends the merged pieces up again from the home, with what
    /// the title wrote there between the two, to both card sides.
    #[test]
    fn warp_cards_a_submission_merges_into_the_home_and_sends_it_up_again() {
        let Some(host) = warp() else { return };
        let _on = Switched::on();
        let region = Region::new(8 * BIG as usize);
        let (outer, outer_len) = (region.base + K64, 2 * BIG + K64);
        let (inner, inner_len) = (outer + 4 * K64, BIG);
        fill(outer, outer_len, 5);
        let a = card_copy(0xca4d_0501, outer, outer_len);
        let b = card_copy(0xca4d_0502, inner, inner_len);
        let home = a.card_homed().expect("its CPU side is a home").0.clone();
        assert!(
            b.card_homed().is_some_and(|(h, ..)| Arc::ptr_eq(h, &home)),
            "the inner one's CPU side is the outer one's home"
        );
        let original = title(outer, outer_len);
        // What the list writes, from a buffer of ours.
        let bytes = [0xa1u8, 0xa2, 0xa3, 0xa4, 0xa5, 0xb1, 0x7e];
        let (source, data) = host.shared_buffer(256).unwrap();
        // SAFETY: the mapping holds 256 bytes; the card does not use it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        // A's word at 0x2000 (bytes 2 to 5, over two words), A's and B's
        // shared word at the inner buffer's 0x3000 (byte 0 by A, 1 by B),
        // and the title's byte 1 of A's word at 0x2000.
        let shared = inner - outer + 0x3000;
        let list = host.new_list().unwrap();
        list.defer_refresh(&a, outer as usize, outer_len as usize);
        list.defer_refresh(&b, inner as usize, inner_len as usize);
        // SAFETY: an open list of the host's; live buffers, the ranges in them.
        unsafe {
            list.list.CopyBufferRegion(&a.buffer, 0x2002, &source, 0, 4);
            list.list.CopyBufferRegion(&a.buffer, shared, &source, 4, 1);
            list.list.CopyBufferRegion(&b.buffer, 0x3001, &source, 5, 1);
            list.list
                .CopyBufferRegion(&home.buffer, home.offset(outer + 0x2001), &source, 6, 1);
        }
        list.defer_gpu_write(&a, (outer + 0x2000) as usize, 0x10);
        list.defer_gpu_write(&a, (outer + shared) as usize, 4);
        list.defer_gpu_write(&b, (inner + 0x3000) as usize, 4);
        list.close();
        host.execute(&[&list]);
        let mut expected = original.clone();
        expected[0x2001] = 0x7e;
        expected[0x2002..0x2006].copy_from_slice(&bytes[..4]);
        expected[shared as usize] = 0xa5;
        expected[shared as usize + 1] = 0xb1;
        let now = title(outer, outer_len);
        assert_eq!(
            now[0x2000..0x2010],
            expected[0x2000..0x2010],
            "the shaders' bytes and the title's of the same word"
        );
        assert_eq!(
            now[shared as usize..shared as usize + 4],
            expected[shared as usize..shared as usize + 4],
            "both copies' bytes of one word"
        );
        assert!(now == expected, "nothing else of title memory changed");
        // The title writes the merged piece between the submissions; the
        // next one takes it, and the merged pieces, up from the home.
        // SAFETY: committed memory of the test's own.
        unsafe { ((outer + 0x2008) as *mut u8).write(0x44) };
        list.reset();
        list.defer_refresh(&a, outer as usize, outer_len as usize);
        list.defer_refresh(&b, inner as usize, inner_len as usize);
        list.close();
        host.execute(&[&list]);
        let mut rig = Rig::new(&host.device).unwrap();
        assert_eq!(
            rig.fetch(&a.buffer, 0x2000, 16).unwrap(),
            title(outer + 0x2000, 16),
            "the outer card side has the merge and the title's write"
        );
        assert_eq!(
            rig.fetch(&a.buffer, shared, 4).unwrap(),
            title(outer + shared, 4),
            "and the inner one's shader byte"
        );
        assert_eq!(
            rig.fetch(&b.buffer, 0x3000, 4).unwrap(),
            title(inner + 0x3000, 4),
            "the inner card side has the outer one's shader byte"
        );
        list.reset();
        drop(list);
        unregister(0xca4d_0502, inner);
        unregister(0xca4d_0501, outer);
        drop((home, a, b, rig, source));
        region.release();
    }

    /// The room goes back to the card's memory once the need fit it for
    /// [`ROOM_BACK_AFTER`] submissions in a row, and grows when it is short.
    #[test]
    fn the_room_goes_back_to_the_card_when_the_need_shrinks() {
        const MB: u64 = 1 << 20;
        assert_eq!(room_move(None, 8 * MB, 0), (RoomMove::Grow, 0));
        assert_eq!(
            room_move(Some((16 * MB, true)), 32 * MB, 3),
            (RoomMove::Grow, 0)
        );
        assert_eq!(
            room_move(Some((16 * MB, true)), 8 * MB, 0),
            (RoomMove::Keep, 0)
        );
        assert_eq!(
            room_move(Some((1024 * MB, false)), 600 * MB, 7),
            (RoomMove::Keep, 0),
            "past the card's most: it stays, and the count starts again"
        );
        let mut small = 0;
        for n in 1..ROOM_BACK_AFTER {
            let (step, now) = room_move(Some((1024 * MB, false)), 8 * MB, small);
            assert_eq!((step, now), (RoomMove::Keep, n));
            small = now;
        }
        assert_eq!(
            room_move(Some((1024 * MB, false)), REFERENCE_CARD_MOST, small),
            (RoomMove::Back, 0)
        );
    }

    /// A kept reference turned down holds back only those as big or bigger,
    /// for [`KEPT_RETRY`] submissions.
    #[test]
    fn a_refused_kept_reference_holds_back_only_those_as_big() {
        const MB: u64 = 1 << 20;
        assert!(!backed_off((0, 0), 1, 64 * MB), "nothing refused yet");
        let retry = (10 + KEPT_RETRY, 64 * MB);
        assert!(backed_off(retry, 11, 64 * MB));
        assert!(backed_off(retry, 11, 128 * MB));
        assert!(!backed_off(retry, 11, 32 * MB), "a smaller one is tried");
        assert!(backed_off(retry, 10 + KEPT_RETRY - 1, 64 * MB));
        assert!(!backed_off(retry, 10 + KEPT_RETRY, 64 * MB), "tried again");
    }

    /// FLOW's counters now.
    fn flow_now() -> [u64; flow::ALL] {
        std::array::from_fn(|i| FLOW[i].load(Ordering::Relaxed))
    }

    /// A copy declared written whole (as `crate::memory` declares the copies
    /// it numbered) keeps its reference: the first submission fills it whole
    /// and copies nothing into the room; the next copies into it only what
    /// went up (a page the title wrote and the piece merged before), and its
    /// merge still keeps a byte the title wrote while the card ran over one
    /// that went up before. Nothing declared for a while, it is let go.
    #[test]
    fn warp_cards_a_whole_declaration_keeps_its_reference() {
        let Some(host) = warp() else { return };
        let _on = Switched::on();
        let region = Region::new(8 * BIG as usize);
        let (start, len) = (region.base + K64, 2 * BIG + K64);
        fill(start, len, 7);
        let a = card_copy(0xca4d_0601, start, len);
        let (home, _, size) = a.card_homed().expect("its CPU side is a home");
        let home = home.clone();
        let original = title(start, len);
        let bytes = [0xc1u8, 0xc2, 0x6b];
        let (source, data) = host.shared_buffer(256).unwrap();
        // SAFETY: the mapping holds 256 bytes; the card does not use it yet.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len()) };
        let made = KEPT[0].load(Ordering::Relaxed);
        let list = host.new_list().unwrap();
        list.defer_refresh(&a, start as usize, len as usize);
        // SAFETY: an open list of the host's; live buffers, the ranges in them.
        unsafe { list.list.CopyBufferRegion(&a.buffer, 0x2002, &source, 0, 2) };
        list.defer_gpu_write(&a, start as usize, len as usize);
        list.close();
        let was = flow_now();
        host.execute(&[&list]);
        let now = flow_now();
        assert_eq!(
            KEPT[0].load(Ordering::Relaxed),
            made + 1,
            "a kept reference made"
        );
        assert_eq!(
            now[flow::REFERENCE] - was[flow::REFERENCE],
            0,
            "nothing into the room"
        );
        assert_eq!(now[flow::KEPT] - was[flow::KEPT], size, "filled whole");
        assert_eq!(
            now[flow::SKIPPED] - was[flow::SKIPPED],
            len,
            "the declared range not copied"
        );
        let mut expected = original.clone();
        expected[0x2002..0x2004].copy_from_slice(&bytes[..2]);
        assert!(
            title(start, len) == expected,
            "the shaders' bytes and nothing else"
        );
        // The title writes a byte of page 3 between the submissions; the next
        // submission's shaders write the word beside it, and the title writes
        // that byte again while the card runs (through the home, as another
        // view of its memory would).
        // SAFETY: committed memory of the test's own.
        unsafe { ((start + 0x3_0010) as *mut u8).write(0x5a) };
        list.reset();
        list.defer_refresh(&a, start as usize, len as usize);
        // SAFETY: as above.
        unsafe {
            list.list
                .CopyBufferRegion(&a.buffer, 0x3_0000, &source, 0, 1);
            list.list
                .CopyBufferRegion(&home.buffer, home.offset(start + 0x3_0010), &source, 2, 1);
        }
        list.defer_gpu_write(&a, start as usize, len as usize);
        list.close();
        let was = flow_now();
        host.execute(&[&list]);
        let now = flow_now();
        let up = now[flow::UPLOADED] - was[flow::UPLOADED];
        assert!(
            (2 * K64..size / 4).contains(&up),
            "the merged piece and the title's page went up, not the copy: {up}"
        );
        assert_eq!(
            now[flow::KEPT] - was[flow::KEPT],
            up,
            "only those into the kept reference"
        );
        assert_eq!(now[flow::REFERENCE] - was[flow::REFERENCE], 0);
        assert_eq!(KEPT[0].load(Ordering::Relaxed), made + 1, "the same one");
        expected[0x3_0000] = 0xc1;
        expected[0x3_0010] = 0x6b;
        let after = title(start, len);
        assert_eq!(
            after[0x3_0010], 0x6b,
            "the title's write while the card ran stays"
        );
        assert_eq!(after[0x3_0000], 0xc1, "the shader's byte is merged");
        assert!(after == expected, "nothing else of title memory changed");
        list.reset();
        drop(list);
        // Nothing declared: let go after `KEPT_IDLE` submissions, those that
        // need no `before` (`idle`) counted too.
        let bytes_kept = KEPT_BYTES.load(Ordering::Relaxed);
        assert!(bytes_kept >= size);
        let (list, places) = before(host, &[], &[]).expect("nothing to do");
        assert!(list.is_none() && places.is_empty());
        for _ in 1..KEPT_IDLE {
            idle();
        }
        assert_eq!(
            KEPT_BYTES.load(Ordering::Relaxed),
            bytes_kept,
            "kept for `KEPT_IDLE` submissions"
        );
        idle();
        assert_eq!(
            KEPT_BYTES.load(Ordering::Relaxed),
            bytes_kept - size,
            "let go"
        );
        unregister(0xca4d_0601, start);
        drop((home, a, source));
        region.release();
    }
}
