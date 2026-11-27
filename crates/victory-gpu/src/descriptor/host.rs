//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::Mutex;

use victory_shader::emit::GuardView;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, view_at};
use crate::host::host;

pub(crate) struct HostHeap {
    id: u64,
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more, then
    /// null slots up to the null views they end with ([`GuardView`]).
    count: u32,
    /// Dynamic slots past `count`: [`DYNAMIC`] in view heaps, none in
    /// sampler heaps. (Upstream falls back to 2^16 on a capped host heap;
    /// ours keeps 2^18 there: 2^16 and 2^17 both ran out in this title.)
    dynamic: u32,
    sampler: bool,
    /// The record (view index and generation) each slot was last made from;
    /// [`CLAIMED`] while one thread changes the slot ([`HostHeap::claim`]).
    /// Changed a slot at a time, with no lock over the heap: what a write
    /// changes in the queue of textures and the sets of buffer views is
    /// logged and applied by whatever reads them next ([`NoteLog`]), and a
    /// write of a slot that holds no placed texture view leaves the spans'
    /// lock alone ([`HostHeap::spanned`]).
    made: Box<[std::sync::atomic::AtomicU64]>,
    /// The buffer record each slot was last left for a draw from (a buffer
    /// view mirrored not for a draw: a null view of its kind, its word, no
    /// host resource kept), 0 if the slot's last change was anything else.
    /// Written by every claim of the slot as it is let go, before `made`
    /// ([`Claimed`]), so `made` 0 and this record together say what the
    /// slot shows. None in sampler heaps (no buffer views) or heaps made
    /// without it (tests). The pattern follows cool_fox's `quick` record
    /// (53c5256, e4d0837), which keeps the key a slot was last made from
    /// beside `made` and had to forget it on every path that changes the
    /// slot; ours is forgotten by the claim itself.
    left: Box<[std::sync::atomic::AtomicU64]>,
    /// Tests: host descriptors [`HostHeap::mirror_marking`] wrote into this
    /// heap's slots.
    #[cfg(test)]
    writes: std::sync::atomic::AtomicU64,
    /// The changes [`HostHeap::note_buffer`]
    /// makes to `textures`, `buffers` and `changed`, in the order they were
    /// made, until whatever reads those takes them ([`NoteLog`]).
    log: NoteLog,
    /// The key of the record each slot held when it was last looked at by
    /// the pass over records copied in as bytes, or mirrored by a call that
    /// wrote it (0: never, or no view there; [`HostHeap::sort_copied_page`]).
    /// Made on the first pass.
    seen: std::sync::OnceLock<Box<[std::sync::atomic::AtomicU64]>>,
    /// Title slots whose changed record the pass over records copied in as
    /// bytes left to their draws, as tables held them lately
    /// ([`HostHeap::sort_copied_page`]): once no table holds one for
    /// [`TABLED_FRAMES`], the pass reads it again whether or not its page was
    /// written since ([`HostHeap::retake_left`]). Without that, a record a
    /// draw never made, in a slot no table reads any more, went on showing
    /// the view from before for as long as nothing else on its page changed.
    /// At most [`LEFT_MOST`] slots; taken by the pass alone, once a pass.
    left_to_tables: Mutex<std::collections::BTreeSet<u32>>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    /// In shards of [`KEPT_SHARD`] slots, each under its own lock.
    kept: Box<[Mutex<Vec<Option<ID3D12Resource>>>]>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
    /// How many dynamic slots are free (changed under `next`'s lock): with
    /// fewer than a table needs, the search is not made. Once the world
    /// had taken them all, every later table of the frame searched every
    /// slot before it was refused.
    free: std::sync::atomic::AtomicU32,
    /// The shortest table that found no run of free slots since slots were
    /// last given back (changed under `next`'s lock; `u32::MAX`: none).
    /// Until something is given back the free runs only shrink, so a table
    /// of that length or longer is refused without the search. Enough
    /// slots were free, but in pieces, and every such table walked the
    /// whole ring before it was refused: 0.33 ms each with 131,072 slots,
    /// 1.0 s of the recording threads in a frame that refused 3,143 (run
    /// 45), under the lock all of them need. (A free run that lies across
    /// the place the search starts from is seen by neither.)
    too_long: std::sync::atomic::AtomicU32,
    /// View heaps: the bytes per element of the buffer view at each title
    /// slot, for shaders that pick a buffer by an index they compute
    /// (`victory_shader::emit::SLOT_STRIDES`): an upload buffer of
    /// `SLOT_STRIDES_COUNT` dwords, its GPU address and its mapping.
    strides: Option<(ID3D12Resource, u64, usize)>,
    /// Title slots holding buffer views. Such a view gets its host slot
    /// (and its memory a host copy) only when something reads it: a draw
    /// through a table, or [`HostHeap::buffer_slots`] for shaders that pick
    /// buffers from the whole heap.
    buffers: Mutex<std::collections::BTreeSet<u32>>,
    /// Of those and of the slots that left the set: the ones written since
    /// [`HostHeap::buffer_changes`] last took them.
    changed: Mutex<std::collections::HashSet<u32>>,
    /// With `VICTORY_SLOT_BITS=1` ([`slot_bits`]): `buffers` and `changed`
    /// as bits, changed with no lock and no log; the two sets above stay
    /// empty then.
    bits: Option<BufferBits>,
    /// With [`away_ms`]: the slots made from a texture whose memory was
    /// found away, with what each was made from and when that was first
    /// found ([`HostHeap::texture_slot_gone`]).
    away: Mutex<std::collections::HashMap<u32, (u64, u64)>>,
    /// Title slots holding a view of a texture the title may fill itself (a
    /// placed texture, a texture object), with the frame from which they
    /// are due: such a texture is uploaded when a draw reads it through a
    /// table; one a shader picks by a computed index is brought up to date
    /// from here ([`HostHeap::texture_slots_due`]).
    textures: Mutex<TextureQueue>,
    /// Per title slot, the frame (plus one; 0: never) in which a draw last
    /// read it through a table, or made a table it lies in (with
    /// [`TABLED_READ`] set if the draw read it). The draw that
    /// reads such a slot brings its texture up to date itself and the slot
    /// then leaves `textures`; a write of a slot that was a table's within
    /// the last [`TABLED_FRAMES`] frames is due only [`TABLED_GRACE`]
    /// frames later, by when its draw has read it. A title that builds a
    /// table per draw writes thousands of texture views a frame into slots
    /// only tables read (the title: some 2,500 a frame in its
    /// opening cutscene, a few hundred textures over and over), and every
    /// one of them stood in the queue of textures picked by index, ahead
    /// of or behind the ones shaders do pick by index.
    tabled: Vec<std::sync::atomic::AtomicU32>,
    /// With `VICTORY_HEAP_PASS_TABLED=1`, per title slot: the frame (plus
    /// one; 0: never) in which a draw last read it through a table or made
    /// a table it lies in, for the heap buffer pass alone
    /// ([`HostHeap::buffer_slot_tabled_lately`], [`TABLED_BUFFER_FRAMES`]).
    /// Apart from `tabled`, which the heap sync and the queue of textures
    /// picked by index read with their [`TABLED_FRAMES`] window: a buffer
    /// slot a table held is not left by the sync for 240 frames, so a
    /// record the title copies into it still reaches the set of changes,
    /// and the pass makes it once no table held it for a few frames.
    /// Made on the first mark; never with the switch off.
    buffer_tabled: std::sync::OnceLock<Box<[std::sync::atomic::AtomicU32]>>,
    /// What the heap buffer pass keeps of this heap from pass to pass
    /// (`crate::draw::HeapRanges`) with `VICTORY_HEAP_SWEEP_OWN=1`: under a
    /// lock of its own, not the one of every heap's, and let go with the
    /// heap. Unused with the switch off.
    pub(crate) buffer_sweep: Mutex<crate::draw::HeapRanges>,
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// View heaps, per title slot: [`IN_SPANS`] if its last
    /// write left it in `placed` (it may have left as a stale one since),
    /// and above that bit a count of its writes that looked at `placed`.
    /// Changed under the slot's claim and the spans' lock. A stale slot
    /// is made null only if no write of it looked at the spans since it
    /// went stale ([`HostHeap::unmake_stale`]). Empty in sampler heaps.
    spanned: Box<[std::sync::atomic::AtomicU32]>,
    /// The heap buffer pass's claim and walk ([`WalkGate`]).
    walk: std::sync::Arc<WalkGate>,
    /// The slots made on C-pool homes ([`PoolSlots`]), and the number of
    /// the newest let-go of such a home ([`pool_let_go`]) this heap looked
    /// at ([`HostHeap::pool_slots_let_go`]).
    pool_slots: Mutex<PoolSlots>,
    pool_seen: std::sync::atomic::AtomicU64,
    /// How many slots `pool_slots` holds (changed under its lock).
    pool_noted: std::sync::atomic::AtomicUsize,
}

/// The heap buffer pass of one view heap (`State::heap_buffers` in
/// `crate::draw`): the frame whose pass was claimed last, and the frames
/// whose walk over the heap's buffer views has ended. The walk gives the
/// buffer views the title wrote their host slots; until it reaches one,
/// such a slot holds a null view of its kind and its word
/// ([`HostHeap::mirror`]), and a shader that picks it reads zeros. A list
/// whose shader picks buffers from the heap after another list claimed
/// the pass is not run before the walk it lost to has ended
/// ([`WalkGate::wait`]), as when every such list waited for the walk's
/// lock while it recorded.
pub(crate) struct WalkGate {
    /// The heap's id ([`HostHeap::id`]), for the profiler.
    heap: u64,
    /// The last frame whose pass was claimed, plus one (0: none).
    claimed: std::sync::atomic::AtomicU64,
    /// The highest claim (a frame plus one) whose walk has ended. Walks
    /// run one at a time and end under the lock they run under
    /// (`HEAP_RANGES` in `crate::draw`). So when a newer claim's walk ends
    /// before an older one's (two claims a frame apart), the older has not
    /// begun. A list that lost to the older claim bound before the newer
    /// walk began: it found the older claim to be the last. The newer walk
    /// took every slot changed until then, so its end lets that list go
    /// too, as the lock let it go after whichever walk held it.
    walked: Mutex<u64>,
    /// Told when `walked` moves.
    ended: std::sync::Condvar,
}

/// A claimed walk ([`WalkGate::claim`]): it counts as ended when this is
/// dropped, whatever way the pass leaves.
pub(crate) struct Walking {
    gate: std::sync::Arc<WalkGate>,
    mark: u64,
}

impl Drop for Walking {
    fn drop(&mut self) {
        self.gate.end(self.mark);
    }
}

impl WalkGate {
    pub(crate) fn new(heap: u64) -> Self {
        WalkGate {
            heap,
            claimed: std::sync::atomic::AtomicU64::new(0),
            walked: Mutex::new(0),
            ended: std::sync::Condvar::new(),
        }
    }

    /// Claims the pass of `frame` without a lock: `Ok` for the one caller
    /// that runs it, `Err` with the claim it lost to for every other
    /// (the newest one, at least `frame` plus one). A caller that read an
    /// older frame than the last claim loses to that claim.
    pub(crate) fn claim(self: &std::sync::Arc<Self>, frame: u64) -> Result<Walking, u64> {
        let mark = frame.saturating_add(1);
        let last = self
            .claimed
            .fetch_max(mark, std::sync::atomic::Ordering::AcqRel);
        if last < mark {
            Ok(Walking {
                gate: self.clone(),
                mark,
            })
        } else {
            Err(last)
        }
    }

    /// Whether the walk of claim `mark` ([`WalkGate::claim`]) has ended.
    pub(crate) fn walked(&self, mark: u64) -> bool {
        *self.walked.lock().unwrap_or_else(|e| e.into_inner()) >= mark
    }

    /// How long a wait for a walk ([`WalkGate::wait`]) lasts before it is
    /// told in the log, and before a submission goes on without the walk.
    /// (The longest walk measured took under a second.)
    const LONG: std::time::Duration = std::time::Duration::from_secs(10);

    /// Waits until the walk of claim `mark` has ended, and says whether it
    /// has; a span on the profile when it had not ended at the call
    /// (`at_submit`: the wait is the submission's, not the recording
    /// list's). The waiting thread holds no lock a walk takes (see
    /// `crate::host::wait_for_walks`).
    ///
    /// A wait of [`WalkGate::LONG`] is told once in the log. A submission
    /// then goes on without the walk (`false`): it may hold locks that
    /// every other submission, fence signal and present needs
    /// (`crate::queue::release_held`), and a walking thread that is gone
    /// without unwinding never ends its walk. The slots that walk had not
    /// reached then read zeros, which [`HostHeap::mirror`] keeps safe. A
    /// recording list waits on, as it did for the walk's lock.
    pub(crate) fn wait(&self, mark: u64, at_submit: bool) -> bool {
        self.wait_within(mark, at_submit, Self::LONG)
    }

    /// [`WalkGate::wait`], with `long` for [`WalkGate::LONG`].
    fn wait_within(&self, mark: u64, at_submit: bool, long: std::time::Duration) -> bool {
        let mut walked = self.walked.lock().unwrap_or_else(|e| e.into_inner());
        if *walked >= mark {
            return true;
        }
        let _wait = crate::profile::span_args(
            "heap walk wait",
            &[
                ("heap", self.heap),
                ("frame", mark - 1),
                ("submit", u64::from(at_submit)),
            ],
        );
        let started = std::time::Instant::now();
        let step = long.min(std::time::Duration::from_secs(1));
        let mut told = false;
        loop {
            if *walked >= mark {
                // (The span is written without the lock the walk's end takes.)
                drop(walked);
                return true;
            }
            if !told && started.elapsed() >= long {
                told = true;
                // Told without that lock too.
                drop(walked);
                tracing::warn!(
                    heap = self.heap,
                    frame = mark - 1,
                    at_submit,
                    ms = long.as_millis() as u64,
                    "the walk of the heap buffer pass has not ended: a submission goes on without it, a recording list waits on"
                );
                if at_submit {
                    return false;
                }
                walked = self.walked.lock().unwrap_or_else(|e| e.into_inner());
                continue;
            }
            walked = self
                .ended
                .wait_timeout(walked, step)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// The walk of claim `mark` has ended. Called after the walk's last
    /// [`HostHeap::mirror`], so a waiter that sees `walked` move also sees
    /// every slot and word the walk wrote (the lock orders them).
    fn end(&self, mark: u64) {
        let mut walked = self.walked.lock().unwrap_or_else(|e| e.into_inner());
        *walked = (*walked).max(mark);
        drop(walked);
        self.ended.notify_all();
    }
}

/// Whether the heap buffer pass walks a slot it makes once: the walk
/// leaves the slot out of the next pass's set
/// ([`HostHeap::mirror_walked`]). `VICTORY_HEAP_WALK_ONCE=0`: the walk
/// marks the slot changed and the next pass walks it again, as before.
fn walk_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_WALK_ONCE").is_none_or(|v| v != "0"))
}

/// The spans of every view heap made ([`HostHeap::placed`]), for the
/// texture code, which has no heap at hand ([`next_start`],
/// [`first_viewed`]).
static PLACED_SPANS: Mutex<Vec<std::sync::Weak<Mutex<Spans>>>> = Mutex::new(Vec::new());

/// Where the next texture placed inside `[start, end)` starts, past the
/// first bytes of the one at `start` ([`HEAD`]): what is there of the
/// texture at `start` ends there, whatever its full size. `None`: no view
/// in a slot shaders pick by index describes memory in that range.
pub(crate) fn next_start(start: u64, end: u64) -> Option<u64> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).next_start(start, end))
        .min()
}

/// The lowest level that a view of the placed texture at `start` shows,
/// over the slots shaders pick by index that hold one now. The title
/// views a texture from the first level it holds. `None`: no such slot.
pub(crate) fn first_viewed(start: u64) -> Option<u32> {
    let heaps = PLACED_SPANS.lock().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        .filter_map(std::sync::Weak::upgrade)
        .filter_map(|spans| spans.lock().unwrap_or_else(|e| e.into_inner()).first_viewed(start))
        .min()
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose first level with memory changed since a texture pass last took
/// them ([`HostHeap::texture_views_remade`]).
static RESHOWN: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Counts the level changes [`levels_changed`] is told of, each counted
/// before its start is pushed to [`RESHOWN`] (under its lock; a start a
/// mirror tells the pass again is pushed without a count): a mirror that
/// resolved a texture view before its slot's claim and finds this moved
/// once it holds the claim resolves the view again
/// ([`HostHeap::mirror_marking`]). Otherwise a level change between
/// the two would be missed: the remade pass, looking at the slot before
/// the claim, finds it made from another record and leaves it to its
/// maker, which then writes a view of the levels from before.
static LEVELS_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The changes of a texture's levels so far ([`LEVELS_SEEN`]), for the
/// frame budget's line.
pub(crate) fn levels_seen() -> u64 {
    LEVELS_SEEN.load(std::sync::atomic::Ordering::Relaxed)
}

/// What the frame budget counts a mirror that changes a slot as: from
/// `view` (`None`: the slot is made null), `later`: left for later.
/// `entry`: the record names a table entry, not a view held whole. Only
/// plain matches: nothing looked up, no lock.
fn mirror_kind(
    view: Option<&View>,
    sampler: bool,
    later: bool,
    entry: bool,
) -> (crate::frame_budget::Mirror, bool) {
    use crate::frame_budget::Mirror;
    const BUFFER: u32 = 1;
    let kind = match view {
        None => Mirror::Null,
        Some(_) if sampler => Mirror::Sampler,
        Some(View::Sampler(_) | View::ConsoleSampler(_)) => Mirror::Sampler,
        // (Left for a draw or not: the title's placed textures are.)
        Some(View::Placed(_, resource, _)) if resource[0] != BUFFER => Mirror::PlacedTexture,
        Some(view) if later && view.is_buffer() => Mirror::BufferDeferred,
        Some(view @ (View::Shader(..) | View::Unordered(..))) if !view.is_buffer() => {
            Mirror::ObjectTexture
        }
        Some(_) => Mirror::Other,
    };
    // A pre-made descriptor per table entry serves a view that is an entry
    // and needs no host copy of title memory.
    let eligible = entry && matches!(kind, Mirror::Sampler | Mirror::ObjectTexture);
    (kind, eligible)
}

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    let mut reshown = RESHOWN.lock().unwrap_or_else(|e| e.into_inner());
    LEVELS_SEEN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    reshown.push(start);
}

/// Levels on demand (`crate::tiles`): where the placed textures start
/// whose last upload left a held level without tiles (the card had no
/// room, or no heap could be had), and when (`crate::host::moment`). A
/// texture pass makes their slots due again once the card has room
/// ([`HostHeap::texture_slots_waiting`]).
static WAITING: Mutex<std::collections::BTreeMap<u64, u64>> = Mutex::new(std::collections::BTreeMap::new());

/// The last upload of the placed texture at `start` left a held level
/// without tiles. A texture picked by index is uploaded once and then
/// only when its memory is found changed, and its staging buffer goes
/// with the upload: without this the level stayed empty (the texture
/// blurred, or black if no level had tiles) for as long as the title left
/// its memory alone, where a texture refused for want of room is tried
/// again.
pub(crate) fn levels_waiting(start: u64) {
    WAITING.lock().unwrap_or_else(|e| e.into_inner()).insert(start, crate::host::moment());
}

/// Title memory by the placed texture views that describe it: per span of
/// memory (its start, its end at full size) the title slots holding a
/// view of a texture that takes exactly that span. A console title need
/// not clear a descriptor when it streams a texture out, so a newer view
/// that cannot be there together with an older one is all that says the
/// older one is stale ([`Rule`] says when that is). Views of one span with
/// different descriptions (the formats of one family, say) are one
/// texture's.
///
/// Spans may overlap: a texture's smallest levels lie at the start of its
/// memory, and a title that holds only those puts the next texture where
/// the larger ones would be (the title: three 512x256 textures
/// of 0x2c000 bytes each whose starts are 0x2000 and 0x4000 apart, all
/// three read by the same frame's draws, run 188). What is there of a
/// texture ends where the next one starts.
/// No host heap is needed, so it is tested without one.
#[derive(Default)]
pub(crate) struct Spans {
    /// When a newer view makes an older one stale.
    rule: Rule,
    by_start: std::collections::BTreeMap<u64, (u64, std::collections::BTreeSet<u32>)>,
    by_slot: std::collections::HashMap<u32, u64>,
    /// The level each slot's view starts at ([`Spans::shows`]).
    first: std::collections::HashMap<u32, u32>,
    /// The bytes a placed texture takes, by resource description: worked
    /// out once for each (a title writes the views of a few thousand
    /// descriptions, thousands of times a frame).
    sizes: std::collections::HashMap<[u32; 14], Option<u64>>,
}

/// When a view the title writes makes an older view of other memory stale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Rule {
    /// When it starts where the older one starts: at the same address
    /// with another length, or within the older one's first [`HEAD`]
    /// bytes (or the older within its own). No two textures share the
    /// block of their smallest levels. A view that only reaches into the
    /// address range of an older texture's larger levels says nothing.
    #[default]
    Start,
    /// Whenever their spans at full size overlap (the rule until run
    /// 188, where it made null the slots of textures packed as above:
    /// each view written took its neighbours out of the queue and off the
    /// card, for good). `VICTORY_STALE_TEXTURES=overlap`.
    Overlap,
    /// Never; a view that would be stale under [`Rule::Overlap`] is left
    /// out of the spans instead. `VICTORY_STALE_TEXTURES=0`, as before.
    Never,
}

/// The first bytes of a placed texture that are its own under
/// [`Rule::Start`]: the 4 KiB block its smallest levels share. (Of 129
/// pairs of overlapping neighbours in the listings of runs 188 and 189,
/// none starts closer than 0x2000 to the one before it.)
const HEAD: u64 = 4096;

impl Spans {
    /// The bytes of title memory a placed texture with the resource
    /// description `desc` takes (`crate::resource::placed_bytes`).
    fn bytes(&mut self, desc: &[u32; 14]) -> Option<u64> {
        if let Some(&known) = self.sizes.get(desc) {
            return known;
        }
        let bytes = crate::resource::placed_bytes(desc);
        self.sizes.insert(*desc, bytes);
        bytes
    }

    /// The starts of the spans that overlap `[start, end)`, each taken to
    /// end where the next one starts: the last span that starts before
    /// `start`, if it reaches past it at full size, and every span that
    /// starts inside.
    fn overlapping(&self, start: u64, end: u64) -> Vec<u64> {
        let before = self
            .by_start
            .range(..start)
            .next_back()
            .filter(|(_, span)| span.0 > start)
            .map(|(at, _)| *at);
        before.into_iter().chain(self.by_start.range(start..end).map(|(at, _)| *at)).collect()
    }

    /// Slot `index` holds a view of a texture in `[start, end)`. `newer`:
    /// the title wrote it now, so the slots whose views cannot be there
    /// with it ([`Rule`]) are stale: they are returned, and forgotten.
    /// Otherwise (a draw made the slot from what it held), and under
    /// [`Rule::Never`], a view that clashes with other spans is left out.
    fn note(&mut self, index: u32, start: u64, end: u64, newer: bool) -> Vec<u32> {
        let same = |span: Option<&(u64, std::collections::BTreeSet<u32>)>| span.is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(self.by_start.get(&start)) {
            return Vec::new();
        }
        self.forget(index);
        if end <= start {
            return Vec::new();
        }
        let rule = self.rule;
        // The spans the view cannot be there with.
        let clashing: Vec<u64> = match rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let mut stale = Vec::new();
        for at in clashing {
            if at == start && same(self.by_start.get(&at)) {
                continue;
            }
            if !newer || rule == Rule::Never {
                return stale;
            }
            if let Some((old_end, slots)) = self.by_start.remove(&at) {
                // The first of them are said with both spans, to tell a
                // replacement from a neighbour in one launch's log.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
                    tracing::info!(
                        slot = index,
                        start = format_args!("{start:#x}"),
                        end = format_args!("{end:#x}"),
                        stale_start = format_args!("{at:#x}"),
                        stale_end = format_args!("{old_end:#x}"),
                        stale_slots = slots.len(),
                        rule = ?rule,
                        "a placed texture view makes older views of other memory stale"
                    );
                }
                for slot in slots {
                    self.by_slot.remove(&slot);
                    stale.push(slot);
                }
            }
        }
        self.by_start.entry(start).or_insert_with(|| (end, Default::default())).1.insert(index);
        self.by_slot.insert(index, start);
        stale
    }

    /// The view in slot `index` starts at level `first` of its texture.
    fn shows(&mut self, index: u32, first: u32) {
        self.first.insert(index, first);
    }

    /// The lowest level a view of the texture at `start` starts at.
    fn first_viewed(&self, start: u64) -> Option<u32> {
        self.by_start.get(&start)?.1.iter().filter_map(|slot| self.first.get(slot)).min().copied()
    }

    /// The first span that starts inside `[start, end)` past the first
    /// [`HEAD`] bytes of it.
    fn next_start(&self, start: u64, end: u64) -> Option<u64> {
        self.by_start.range(start.saturating_add(HEAD)..end).next().map(|(at, _)| *at)
    }

    /// The slots that hold a view of the texture at `start`.
    fn slots_at(&self, start: u64) -> Vec<u32> {
        self.by_start.get(&start).map(|span| span.1.iter().copied().collect()).unwrap_or_default()
    }

    /// What the spans say of a view in slot `index` of a texture in
    /// `[start, end)`, and the first span that speaks against it: by the
    /// same rule as [`Spans::note`].
    fn claim(&self, index: u32, start: u64, end: u64) -> (Claim, Option<(u64, u64)>) {
        let same = |at: u64| self.by_start.get(&at).is_some_and(|s| s.0 == end);
        if self.by_slot.get(&index) == Some(&start) && same(start) {
            return (Claim::Own, None);
        }
        let clashing: Vec<u64> = match self.rule {
            Rule::Start => self
                .by_start
                .range(start.saturating_sub(HEAD - 1)..start.saturating_add(HEAD))
                .map(|(at, _)| *at)
                .collect(),
            Rule::Overlap | Rule::Never => self.overlapping(start, end),
        };
        let against: Vec<(u64, u64)> = clashing
            .into_iter()
            .filter(|&at| !(at == start && same(at)))
            .filter_map(|at| self.by_start.get(&at).map(|s| (at, s.0)))
            .collect();
        match against.first() {
            None => (Claim::Free, None),
            Some(&span) if against.iter().all(|s| s.0 == start) => (Claim::SameStart, Some(span)),
            Some(&span) => (Claim::Other, Some(span)),
        }
    }

    /// Slot `index` holds no placed texture view any more.
    fn forget(&mut self, index: u32) {
        self.first.remove(&index);
        let Some(start) = self.by_slot.remove(&index) else {
            return;
        };
        if let Some(span) = self.by_start.get_mut(&start) {
            span.1.remove(&index);
            if span.1.is_empty() {
                self.by_start.remove(&start);
            }
        }
    }

    /// The slots whose texture's memory overlaps `[start, end)`.
    fn slots_in(&self, start: u64, end: u64) -> Vec<u32> {
        self.overlapping(start, end)
            .iter()
            .filter_map(|at| self.by_start.get(at))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }

    /// The slots of the textures that hold bytes of `[start, end)`, for a
    /// write there: as [`Spans::slots_in`], but when a texture starts at
    /// `start` itself the one before it is left out (it ends where that
    /// one starts, and a request that fills a texture begins at its first
    /// byte: `slots_in` would name its neighbour with every such write).
    fn slots_holding(&self, start: u64, end: u64) -> Vec<u32> {
        if end <= start {
            return Vec::new();
        }
        let before = if self.by_start.contains_key(&start) {
            None
        } else {
            self.by_start.range(..start).next_back().filter(|(_, span)| span.0 > start).map(|(_, span)| span)
        };
        before
            .into_iter()
            .chain(self.by_start.range(start..end).map(|(_, span)| span))
            .flat_map(|span| span.1.iter().copied())
            .collect()
    }
}

/// Title memory written behind the guards' back whose bytes are there
/// (DirectStorage's reads into physical pages, at the addresses the title
/// has them mapped; `crate::watch::title_memory_written`), since a texture
/// pass last took the ranges ([`page_written`]). Kept apart from
/// [`REMAPPED`] so that what such writes do is counted by itself.
static PAGE_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots whose texture had been uploaded
/// before the write (out of the queue, or in it after an upload): they are
/// the ones that kept the bytes from before, and the ones made due when
/// `VICTORY_PAGE_WRITES` asks for it.
pub(crate) static PAGE_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Frames from the pass that takes a written range to the look at its
/// slots: the levels of one texture arrive in a few requests, and a look
/// two frames on takes those of the next frame with it.
const PAGE_WRITE_AFTER: u64 = 2;

/// The bytes of a write the guards do not see are in title memory `[at,
/// at + len)`: the placed textures there changed with no descriptor
/// written and nothing mapped. Kept for the next texture pass
/// ([`HostHeap::texture_slots_page_written`]), as [`remapped`] keeps its
/// ranges; locks one mutex and calls nothing.
pub(crate) fn page_written(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    PAGE_WRITES[0].fetch_add(1, Relaxed);
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        PAGE_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Title memory the GPU wrote through homes in chunks of the memory pool
/// (`crate::pool::textures_told`), since a texture pass last took the
/// ranges ([`pool_written`]).
static POOL_WRITTEN: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Since the start: such ranges kept for a pass; ranges dropped because no
/// pass took them; slots of placed textures that held bytes of them when a
/// pass took them; of those, the slots uploaded before, made due again.
pub(crate) static POOL_WRITES: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The GPU wrote title memory `[at, at + len)` through a home in a chunk of
/// the memory pool: the placed textures there changed with nothing the
/// guards see, and are made due at the next texture pass
/// ([`HostHeap::texture_slots_page_written`]). Returns the part of the
/// range where placed textures lie (between the lowest start and the
/// highest end of their spans, [`SPANNED`]), `None` if none of it does.
/// Called after a submission's wait; locks one mutex and calls nothing.
///
/// Ranges no pass took are dropped past a bound (counted): the slots of
/// textures picked by index there are then not made due by them; a draw
/// that reads such a texture through a table still sees the write in the
/// journal (`crate::pool::textures_told`).
pub(crate) fn pool_written(at: usize, len: usize) -> Option<(usize, usize)> {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    const MOST: usize = 1 << 15;
    let (lowest, highest) = (SPANNED[0].load(Relaxed), SPANNED[1].load(Relaxed));
    let (start, end) = (
        (at as u64).max(lowest),
        (at.saturating_add(len) as u64).min(highest),
    );
    if start >= end {
        return None;
    }
    let mut ranges = POOL_WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    POOL_WRITES[0].fetch_add(1, Relaxed);
    if ranges.len() >= MOST {
        POOL_WRITES[1].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
    Some((start as usize, end as usize))
}

/// Title memory of homes in chunks of the memory pool that a remap, free or
/// protection let go (`crate::home::cpu_touched`), each with its number,
/// the newest last ([`pool_let_go`]); at most [`POOL_GONE_MOST`] kept.
static POOL_GONE: Mutex<std::collections::VecDeque<(u64, u64, u64)>> =
    Mutex::new(std::collections::VecDeque::new());

/// The number of the newest range in [`POOL_GONE`] (0: none so far).
static POOL_GONE_LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Ranges [`POOL_GONE`] keeps: a heap that has not looked since more went
/// makes all its slots on C-pool homes anew ([`HostHeap::pool_slots_let_go`]).
const POOL_GONE_MOST: usize = 1 << 12;

/// The memory core let a home in a chunk of the memory pool go, over title
/// memory `[start, end)`: the view heap slots made on it show the pages it
/// had, through the chunk's heap, and are made anew from their records at
/// their next use ([`HostHeap::pool_slots_let_go`]). Called on the title's
/// threads with the runtime's own locks held: locks one mutex of its own
/// and calls nothing.
pub(crate) fn pool_let_go(start: u64, end: u64) {
    use std::sync::atomic::Ordering::{Relaxed, Release};
    let mut gone = POOL_GONE.lock().unwrap_or_else(|e| e.into_inner());
    let n = POOL_GONE_LAST.load(Relaxed) + 1;
    if gone.len() >= POOL_GONE_MOST {
        gone.pop_front();
    }
    gone.push_back((n, start, end));
    POOL_GONE_LAST.store(n, Release);
}

/// A view heap's slots made on C-pool homes (`crate::pool`), by the title
/// memory their views read: which to make anew when a home is let go
/// ([`pool_let_go`]).
#[derive(Default)]
struct PoolSlots {
    /// Each slot's range (start, end).
    by_slot: std::collections::HashMap<u32, (u64, u64)>,
    /// The same, by (start, slot), to the end.
    by_start: std::collections::BTreeMap<(u64, u32), u64>,
    /// The longest range ever noted: a range over `[start, end)` begins
    /// at most this far before `start`.
    longest: u64,
}

impl PoolSlots {
    /// Slot `index` now shows `span` of title memory on a C-pool home, or
    /// (`None`) no such home.
    fn set(&mut self, index: u32, span: Option<(u64, u64)>) {
        if let Some((start, _)) = self.by_slot.remove(&index) {
            self.by_start.remove(&(start, index));
        }
        if let Some((start, end)) = span {
            self.by_slot.insert(index, (start, end));
            self.by_start.insert((start, index), end);
            self.longest = self.longest.max(end - start);
        }
    }

    /// Takes out the slots whose range overlaps `[start, end)`.
    fn take_over(&mut self, start: u64, end: u64) -> Vec<u32> {
        let from = start.saturating_sub(self.longest);
        let over: Vec<(u64, u32)> = self
            .by_start
            .range((from, 0)..(end, 0))
            .filter(|&(_, &to)| to > start)
            .map(|(&key, _)| key)
            .collect();
        for key in &over {
            self.by_start.remove(key);
            self.by_slot.remove(&key.1);
        }
        over.into_iter().map(|(_, slot)| slot).collect()
    }

    /// Takes out every slot.
    fn take_all(&mut self) -> Vec<u32> {
        self.by_start.clear();
        self.longest = 0;
        self.by_slot.drain().map(|(slot, _)| slot).collect()
    }
}

/// The title memory a slot made from `view` on `kept` reads, if `kept` is
/// the buffer of a chunk of the memory pool (a C-pool home's, `crate::pool`):
/// the view's range and the whole buffer it is of, joined.
fn pool_slot_span(view: &View, kept: Option<&ID3D12Resource>) -> Option<(u64, u64)> {
    let kept = kept?;
    if !crate::pool::on() || !crate::pool::is_chunk_buffer(kept) {
        return None;
    }
    let span = match *view {
        View::Constant(address, size) => Some((address, address.saturating_add(u64::from(size)))),
        _ => {
            let range = |r: Option<(u64, u64)>| r.map(|(at, len)| (at, at.saturating_add(len)));
            match (range(buffer_range(view)), range(buffer_extent(view))) {
                (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
                (a, b) => a.or(b),
            }
        }
    }?;
    Some((span.0, span.1.max(span.0 + 1)))
}

/// Whose the memory of a texture view is, by [`Spans`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Claim {
    /// The slot's own.
    Own,
    /// No other texture's.
    Free,
    /// Another texture's that starts at the same address.
    SameStart,
    /// Another texture's.
    Other,
}

/// What the last whole look at the heap's placed texture views found
/// ([`HostHeap::texture_sweep`]): views; of them shown by their host slot;
/// in the queue; read through tables lately; without memory; then the ones
/// with memory that nothing brings up to date, by [`Claim`] (own, free,
/// same start, other), and of all those the ones whose host slot shows
/// another view. And since the start: the ones put back into the queue.
pub(crate) static SWEPT: [std::sync::atomic::AtomicU64; 11] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// A look at the heap under way: the next slot, what it counted, and when
/// (milliseconds) it went on last.
static SWEEP: Mutex<(u32, [u64; 10], u64)> = Mutex::new((0, [0; 10], 0));

/// Title memory the runtime mapped, unmapped or is about to write since a
/// texture pass last took the ranges ([`remapped`]).
static REMAPPED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// The lowest start and the highest end of any span of [`Spans`] so far
/// (the start is `u64::MAX` until there is one): what the runtime does to
/// memory outside is none of the texture pass's business.
static SPANNED: [std::sync::atomic::AtomicU64; 2] = [
    std::sync::atomic::AtomicU64::new(u64::MAX),
    std::sync::atomic::AtomicU64::new(0),
];

/// How long a slot made from a texture whose memory is not mapped stays as
/// it is, in milliseconds (`VICTORY_TEXTURE_AWAY_MS`; unset or 0, the
/// default: made null at the first look, as before).
///
/// A title that moves a texture in its streaming memory unmaps it before it
/// maps it again, and what the slot showed went black for the pictures in
/// between (cool_fox saw faces, books and posters black for 0.3 to 2 s, and
/// runs with 3000). Kept, the slot shows the host copy it was made from,
/// which stays until the slot is made null.
fn away_ms() -> u64 {
    static WAIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WAIT.get_or_init(|| {
        std::env::var("VICTORY_TEXTURE_AWAY_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0)
    })
}

/// Since the start: slots whose view a newer view of other memory there
/// made stale; ranges of memory noted as mapped, unmapped or written;
/// ranges dropped because no pass took them; slots made due by such
/// ranges; and made slots whose memory was found unmapped (made null).
pub(crate) static STALE: [std::sync::atomic::AtomicU64; 5] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// When a view the title writes makes older views of other memory stale
/// ([`Rule`]): where it starts in the block of their smallest levels, by
/// default; `VICTORY_STALE_TEXTURES=0`: never, as before;
/// `VICTORY_STALE_TEXTURES=overlap`: wherever their spans overlap, the rule
/// until run 188.
fn stale_rule() -> Rule {
    static RULE: std::sync::OnceLock<Rule> = std::sync::OnceLock::new();
    *RULE.get_or_init(|| match std::env::var("VICTORY_STALE_TEXTURES").as_deref() {
        Ok("0") => Rule::Never,
        Ok("overlap") => Rule::Overlap,
        _ => Rule::Start,
    })
}

/// The runtime mapped or unmapped title memory `[at, at + len)`, or is
/// about to write it through the kernel (`crate::watch`): the placed
/// textures there may have arrived, gone, or changed with no descriptor
/// written. The range is kept for the next texture pass
/// ([`HostHeap::texture_slots_remapped`]). Called with the runtime's own
/// locks held: this locks one mutex of its own and calls nothing.
pub(crate) fn remapped(at: usize, len: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    /// Ranges kept until a pass takes them.
    /// (A world's first minutes note tens of thousands between two passes:
    /// 131,072 were dropped in one run at the 32,768 there were.)
    const MOST: usize = 1 << 18;
    let (start, end) = (at as u64, at.saturating_add(len) as u64);
    if end <= SPANNED[0].load(Relaxed) || start >= SPANNED[1].load(Relaxed) {
        return;
    }
    let mut ranges = REMAPPED.lock().unwrap_or_else(|e| e.into_inner());
    STALE[1].fetch_add(1, Relaxed);
    // Pages are mapped one after the other.
    if let Some(last) = ranges.last_mut()
        && last.1 == start
    {
        last.1 = end;
        return;
    }
    if ranges.len() >= MOST {
        // No pass runs (no shader that picks textures by index is bound):
        // the slots there are looked at when they are due, as before.
        STALE[2].fetch_add(ranges.len() as u64, Relaxed);
        ranges.clear();
    }
    ranges.push((start, end));
}

/// Frames for which a slot a draw read through a table counts as a table's
/// slot ([`HostHeap::tabled`]).
const TABLED_FRAMES: u64 = 240;

/// Frames a write of a table's slot waits in the queue for the draw that
/// reads it, before the slot is taken for one shaders pick by index.
/// Frames as `crate::frame::current` counts them: in its opening cutscene
/// The title begins two or three of them per frame it presents
/// (435 begun in the 353 s of run 52's, for some 140 presented).
const TABLED_GRACE: u64 = 8;

/// Frames for which a slot a draw read through a table is left to such a
/// draw by the heap buffer pass (`VICTORY_HEAP_PASS_TABLED=1`,
/// [`HostHeap::buffer_slot_tabled_lately`]); cool_fox's `COPIED_FRAMES`.
const TABLED_BUFFER_FRAMES: u32 = 8;

/// In [`HostHeap::tabled`]: a draw read the slot in that frame.
const TABLED_READ: u32 = 1 << 31;

/// Since the start: writes of texture views into slots that count as a
/// table's (due later), and slots that left the queue because a draw read
/// them through a table.
pub(crate) static TABLED_WRITES: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether slots draws read through tables leave the queue of textures
/// picked by index; `VICTORY_TABLED_TEXTURES=1` keeps them in it as
/// before.
fn tabled_out() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLED_TEXTURES").is_none_or(|v| v != "1"))
}

/// What the queue of textures picked by index knows of a title slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waiting {
    /// The frame from which the slot is due.
    due: u64,
    /// The frame it was written in.
    written: u64,
    /// Which write of the queue put it here: the pass hands it back with
    /// what it did, so a slot written again meanwhile stays as written.
    write: u64,
    /// Looks in a row that found nothing to upload (its memory not mapped,
    /// or no more of it than at the last upload).
    looks: u8,
    /// Bytes of its memory that were mapped when it was last uploaded
    /// without all of it there; 0: not uploaded yet.
    mapped: u64,
    /// For a slot uploaded whole and looked at again
    /// ([`TextureQueue::recheck`]): where its memory is (address, bytes),
    /// the fingerprint it had at the upload, how many looks found it
    /// unchanged, and the frame of the upload. A fingerprint of 0: none.
    checked: (u64, u64, u64, u8, u64),
}

/// The queue of title slots whose texture view waits to be brought up to
/// date for shaders that pick textures by index: which are due, and in
/// which order (no host heap needed, so it is tested without one).
#[derive(Default)]
pub(crate) struct TextureQueue {
    slots: std::collections::BTreeMap<u32, Waiting>,
    writes: u64,
    /// Slots held back because the card had no room for the host copy
    /// they need ([`TextureQueue::hold`]), each with the write it was
    /// held as.
    held: std::collections::BTreeMap<u32, u64>,
}

impl TextureQueue {
    /// A texture view was written into slot `index` in `frame`, due from
    /// frame `due`. `seen`: it was not written, a draw or the pass made
    /// the host slot from a view the queue may hold already, which then
    /// stays as it is.
    fn written(&mut self, index: u32, frame: u64, due: u64, seen: bool) {
        if seen && self.slots.contains_key(&index) {
            return;
        }
        self.writes += 1;
        let write = self.writes;
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    fn remove(&mut self, index: u32) {
        self.slots.remove(&index);
    }

    /// Up to `max` slots due at `frame`, each with its write number, in
    /// the order to serve them:
    /// - first the ones looked at before (their memory was not mapped, or
    ///   only part of it): the look is a query, and memory that has
    ///   arrived since is what the title streamed in last;
    /// - then the others, the newest write first and the oldest next, in
    ///   turns: slot order served the low slots of a backlog of
    ///   25,000-40,000 while the textures of the shot on screen waited
    ///   (run 52: 98% of the drawn pixels had no colour texture in the
    ///   first 3D frames).
    fn due(&self, frame: u64, max: usize) -> Vec<(u32, u64)> {
        let mut again: Vec<(u64, u32, u64)> = Vec::new();
        let mut fresh: Vec<(u64, u32, u64)> = Vec::new();
        for (&index, w) in self.slots.iter().filter(|(_, w)| w.due <= frame) {
            let class = if w.looks > 0 || w.mapped > 0 { &mut again } else { &mut fresh };
            class.push((w.written, index, w.write));
        }
        // By the number of the write, counted over the run.
        let newest_first = |a: &(u64, u32, u64), b: &(u64, u32, u64)| b.2.cmp(&a.2);
        again.sort_unstable_by(newest_first);
        fresh.sort_unstable_by(newest_first);
        let mut order: Vec<(u32, u64)> = again.iter().map(|&(_, index, write)| (index, write)).collect();
        let (mut new, mut old) = (0, fresh.len());
        while new < old && order.len() < max {
            order.push((fresh[new].1, fresh[new].2));
            new += 1;
            if new < old && order.len() < max {
                old -= 1;
                order.push((fresh[old].1, fresh[old].2));
            }
        }
        order.truncate(max);
        order
    }

    /// The entry of `index` if it is still the write the pass was handed.
    fn same(&mut self, index: u32, write: u64) -> Option<&mut Waiting> {
        self.slots.get_mut(&index).filter(|w| w.write == write)
    }

    /// Slot `index` is up to date: out of the queue until written again.
    fn done(&mut self, index: u32, write: u64) {
        if self.same(index, write).is_some() {
            self.slots.remove(&index);
        }
    }

    /// Nothing to upload at slot `index` now: looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    fn later(&mut self, index: u32, write: u64, frame: u64, most: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = frame + (1u64 << w.looks.min(16)).min(most.max(1));
            w.looks = w.looks.saturating_add(1);
        }
    }

    /// The card has no room for the host copy slot `index` needs: it is
    /// not handed out again before frame `due`, or before the card has
    /// room again ([`TextureQueue::room`]). Nothing else of it changes: it
    /// is not a slot "waiting for its memory", and when it is due it has
    /// its old place among the slots never looked at. (It stayed due, so
    /// every pass on a full card looked at the same few hundred slots
    /// again and stopped there, and the slots behind them were never
    /// reached: the ones whose host copy exists and only wants bringing up
    /// to date, and the ones whose memory went, which is what lets host
    /// copies go and makes room.)
    fn hold(&mut self, index: u32, write: u64, due: u64) {
        if let Some(w) = self.same(index, write) {
            w.due = w.due.max(due);
            self.held.insert(index, write);
        }
    }

    /// The card has room again at `frame`: the slots held back for lack of
    /// it are due now, as if they had never been held. Returns how many.
    fn room(&mut self, frame: u64) -> usize {
        let held = std::mem::take(&mut self.held);
        let mut back = 0;
        for (index, write) in held {
            if let Some(w) = self.same(index, write) {
                w.due = w.due.min(frame);
                back += 1;
            }
        }
        back
    }

    /// Slot `index` was uploaded with `mapped` bytes of its memory there,
    /// not all of it: looked at again from frame `due`.
    fn partly(&mut self, index: u32, write: u64, due: u64, mapped: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 0, mapped.max(1));
        }
    }

    /// Slot `index` was uploaded whole from memory `at` (address, bytes)
    /// with fingerprint `print`: looked at again from frame `due`, for a
    /// title that fills a texture's memory after it wrote the view (the
    /// `looks`-th such look; `uploaded` is the upload's frame).
    #[allow(clippy::too_many_arguments)]
    fn recheck(&mut self, index: u32, write: u64, due: u64, at: (u64, u64), print: u64, looks: u8, uploaded: u64) {
        if let Some(w) = self.same(index, write) {
            (w.due, w.looks, w.mapped) = (due, 1, 0);
            w.checked = (at.0, at.1, print, looks, uploaded);
        }
    }

    /// What [`TextureQueue::recheck`] noted of slot `index`, if it is
    /// waiting for such a look.
    fn checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.slots.get(&index).map(|w| w.checked).filter(|c| c.2 != 0)
    }

    /// The memory slot `index`'s texture lies in was mapped, unmapped or
    /// written in `frame`: the slot is looked at from frame `due` as one
    /// written now, whatever was known of it (a pass at work on it does
    /// not take it out).
    fn changed(&mut self, index: u32, frame: u64, due: u64) {
        self.writes += 1;
        let write = self.writes;
        let due = self.slots.get(&index).map_or(due, |w| w.due.min(due));
        self.slots.insert(index, Waiting { due, written: frame, write, looks: 0, mapped: 0, checked: (0, 0, 0, 0, 0) });
    }

    /// Bytes of slot `index`'s memory that were mapped at its last upload
    /// (0: not uploaded, or all of it was).
    fn mapped(&self, index: u32) -> u64 {
        self.slots.get(&index).map_or(0, |w| w.mapped)
    }

    /// For the frame line: slots in the queue; of them due at `frame`;
    /// waiting for their memory (looked at, never uploaded); uploaded in
    /// part; and the frames the oldest due one has waited since its write.
    fn census(&self, frame: u64) -> [u64; 5] {
        let mut c = [self.slots.len() as u64, 0, 0, 0, 0];
        for w in self.slots.values() {
            if w.mapped > 0 {
                c[3] += 1;
            } else if w.looks > 0 {
                c[2] += 1;
            }
            if w.due <= frame {
                c[1] += 1;
                c[4] = c[4].max(frame.saturating_sub(w.written));
            }
        }
        c
    }
}

/// Whether `view` is of a texture the title may fill itself: a texture it
/// placed in memory, or a texture object.
pub(crate) fn is_texture_view(view: &View) -> bool {
    match view {
        View::Placed(false, resource, _) => resource[0] != 1,
        View::Shader(r, _) => crate::resource::buffer_bytes(*r as crate::object::This).is_none(),
        _ => false,
    }
}

/// Buffer views left null because the host would refuse them
/// ([`buffer_view_ok`]), since the start.
pub(crate) static REFUSED_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host device is gone, and why (the first time it is asked
/// after that): for passes that make many views, to name the one that did
/// it.
pub(crate) fn device_removed() -> Option<String> {
    let host = host()?;
    // SAFETY: a plain query.
    let reason = unsafe { host.device.GetDeviceRemovedReason() };
    reason.err().map(|e| e.to_string())
}

/// Writes of title view-heap slots with a placed texture view and with a
/// view of a texture object, since the start (diagnostic counts).
pub(crate) static PLACED_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static OBJECT_TEXTURE_SLOTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Bytes per element of a buffer view, as the console's descriptor holds
/// them for indexed reads (structured: the stride; raw: 1, the index is the
/// byte; typed: the format's); `None` if `view` is not a buffer view.
fn view_stride(view: &View) -> Option<u32> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    // SAFETY: buffer views' union members.
    let (stride, raw, format) = unsafe {
        if unordered {
            let d = uav_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0, d.Format)
        } else {
            let d = srv_desc(words);
            let b = d.Anonymous.Buffer;
            (b.StructureByteStride, b.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0, d.Format)
        }
    };
    Some(if stride > 0 {
        stride
    } else if raw {
        1
    } else {
        crate::resource::element(format.0 as u32).map_or(0, |(bytes, _)| bytes as u32)
    })
}

/// The dword of the table of strides by slot
/// (`victory_shader::emit::SLOT_STRIDES`) for a slot holding `view`: the
/// element stride of a buffer view; [`SLOT_CONSTANT`] for a constant buffer
/// view or a buffer view of no element size we know (read as a constant
/// buffer; neither is a buffer with a stride); 0 for anything else, which
/// shaders read as a null view.
///
/// [`SLOT_CONSTANT`]: victory_shader::emit::SLOT_CONSTANT
fn slot_word(view: Option<&View>) -> u32 {
    use victory_shader::emit::SLOT_CONSTANT;
    match view {
        Some(View::Constant(..)) => SLOT_CONSTANT,
        Some(view) => match view_stride(view) {
            Some(0) => SLOT_CONSTANT,
            Some(stride) => stride,
            None => 0,
        },
        None => 0,
    }
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// For device-removed reports: the class of a faulting host GPU address
/// `va`. Low: below 4 GiB (no host heap or buffer of ours lies there; a
/// null view read as another kind gave such addresses). Mid: within 4 GiB
/// of a host descriptor heap (a view index past the heap's end: the 24-bit
/// "no descriptor" index is 512 MiB of views past the start). Wild:
/// anywhere else. With the heap nearest to it.
pub(crate) fn fault_class(va: u64) -> String {
    fault_class_in(va, &GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()))
}

/// [`fault_class`] against the heaps `ranges` ([`GPU_RANGES`]). Pure, for
/// testing.
fn fault_class_in(va: u64, ranges: &[(u64, u64, u64, bool)]) -> String {
    const FOUR_GIB: u64 = 1 << 32;
    let distance = |&&(start, bytes, _, _): &&(u64, u64, u64, bool)| {
        if va < start {
            start - va
        } else {
            va.saturating_sub(start.saturating_add(bytes))
        }
    };
    let nearest = ranges.iter().min_by_key(distance);
    let class = match nearest {
        _ if va < FOUR_GIB => "low",
        Some(heap) if distance(&heap) < FOUR_GIB => "mid",
        _ => "wild",
    };
    match nearest {
        Some(&(start, bytes, _, sampler)) => format!(
            "; fault class {class}, nearest the {} heap {start:#x}+{bytes:#x}",
            if sampler { "sampler" } else { "view" }
        ),
        None => format!("; fault class {class}, no host heap"),
    }
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// The null views every view heap ends with ([`GuardView`]).
const GUARDS: u32 = GuardView::ALL.len() as u32;

/// Picks, once, the size of every host view heap and tells the translator
/// its last slot now (`set_view_heap_limit`) instead of when the title's
/// heap is made: a shader keeps the limit it was translated with, and the
/// title makes 25 pipelines (their shaders translated at once) before it
/// makes its view heap. [`HEAP_VIEWS`] where the host takes that many
/// (AMD), else [`PORTABLE_VIEWS`] (NVIDIA). Called before a shader is
/// translated and before a view heap is made.
///
/// Every host clamps: a view index past the heap hung the capped host's
/// GPU, and on AMD a compute program of the title's world that reads the
/// 24-bit "no descriptor" marker as a view index took a page fault past
/// the heap's end and removed the device (RDNA3 and RDNA4).
///
/// [`HEAP_VIEWS`]: victory_shader::emit::HEAP_VIEWS
pub(crate) fn probe_view_limit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let Some(host) = host() else { return };
        // SAFETY: plain heap creation; the heap is released at once.
        let made: windows::core::Result<ID3D12DescriptorHeap> = unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: victory_shader::emit::HEAP_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        };
        let views = if made.is_ok() {
            victory_shader::emit::HEAP_VIEWS
        } else {
            PORTABLE_VIEWS
        };
        tracing::info!(
            views,
            "host view heaps hold {views} views; view indices are clamped to the null views at their end"
        );
        victory_shader::emit::set_view_heap_limit(views - 1);
    });
}

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out in the world ("no free dynamic descriptor slots", run 11;
/// upstream PR #336 takes 2^18). On a host heap capped at 1,000,000 views
/// (NVIDIA) these come out of the title's slots, whose highest seen is
/// about 720,000: 2^17 left 868,000 of them, and ran out in the opening
/// cutscene (a frame asks for up to 16,000 tables while the frame before
/// still holds its own; a draw refused its table read slots never made
/// for it, and the card stopped within frames). 2^18 leaves 737,000; a
/// table past them is made afresh in these slots like any other.
pub(crate) const DYNAMIC: u32 = 1 << 18;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        Self::with_ways(sampler, count, slot_bits(), true)
    }

    /// [`HostHeap::new`]; `bits`: the sets of buffer views are bits
    /// ([`slot_bits`]); `left`: slots keep the record they were last left
    /// for a draw from ([`HostHeap::left`]; false only for tests' reference
    /// heaps, which then make every such mirror in full).
    fn with_ways(sampler: bool, count: u32, bits: bool, left: bool) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        let extra = if sampler { 0 } else { DYNAMIC };
        let (heap, count, size) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, count),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else {
            // A view heap ends at the slot shaders clamp view indices to
            // (`probe_view_limit`), whatever the title's own count: the
            // title's slots, the dynamic ones, null slots, and the null
            // views of every kind (`make_guards`).
            probe_view_limit();
            let size = victory_shader::emit::view_heap_limit() + 1;
            let room = size - GUARDS - extra;
            let fitted = if count <= room {
                count
            } else {
                // shortcut: descriptors past the host's limit are left out; a title that
                // reaches them needs remapping. They read the null views instead.
                tracing::warn!(
                    count,
                    "host view heap capped at {size}; views past it read a null view"
                );
                room
            };
            match make(size) {
                Ok(heap) => (heap, fitted, size),
                Err(e) => {
                    tracing::warn!(error = %e, count, size, "no host view heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(size) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let strides = (!sampler)
            .then(|| {
                let bytes = u64::from(victory_shader::emit::SLOT_STRIDES_COUNT) * 4;
                let (buffer, data) = host.upload_buffer(bytes)?;
                // SAFETY: the buffer's whole mapping; a live buffer.
                let gpu = unsafe {
                    std::ptr::write_bytes(data, 0, bytes as usize);
                    buffer.GetGPUVirtualAddress()
                };
                crate::host::label(&buffer, "stride table", bytes);
                crate::host::name(&buffer, || "element strides by view heap slot".to_owned());
                Some((buffer, gpu, data as usize))
            })
            .flatten();
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let h = HostHeap {
            id,
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            made: (0..count)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            left: (0..if sampler || !left { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
            #[cfg(test)]
            writes: std::sync::atomic::AtomicU64::new(0),
            log: NoteLog::default(),
            seen: std::sync::OnceLock::new(),
            left_to_tables: Mutex::default(),
            kept: (0..(count + extra).div_ceil(KEPT_SHARD))
                .map(|shard| {
                    let slots = (count + extra - shard * KEPT_SHARD).min(KEPT_SHARD);
                    Mutex::new(vec![None; slots as usize])
                })
                .collect(),
            next: Mutex::new((0, vec![false; DYNAMIC as usize])),
            free: std::sync::atomic::AtomicU32::new(DYNAMIC),
            too_long: std::sync::atomic::AtomicU32::new(u32::MAX),
            strides,
            buffers: Mutex::default(),
            changed: Mutex::default(),
            bits: bits.then(|| BufferBits {
                buffers: Slots::new(count as usize),
                changed: Slots::new(count as usize),
            }),
            away: Mutex::default(),
            textures: Mutex::default(),
            tabled: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            buffer_tabled: std::sync::OnceLock::new(),
            buffer_sweep: Mutex::new(crate::draw::HeapRanges::new()),
            placed: {
                let spans = std::sync::Arc::<Mutex<Spans>>::default();
                if !sampler {
                    PLACED_SPANS
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(std::sync::Arc::downgrade(&spans));
                }
                spans
            },
            spanned: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
            walk: std::sync::Arc::new(WalkGate::new(id)),
            pool_slots: Mutex::default(),
            pool_noted: std::sync::atomic::AtomicUsize::new(0),
            // No slot made on a home before this heap was.
            pool_seen: std::sync::atomic::AtomicU64::new(
                POOL_GONE_LAST.load(std::sync::atomic::Ordering::Acquire),
            ),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, size) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        if !sampler {
            h.make_guards(size - 1);
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the null views a view heap whose last slot is `last` ends with
    /// ([`GuardView`]), each of the kind and texture dimension of the
    /// accesses that read it: none reads a null view of another kind (on
    /// AMD a null texture view read as a buffer gives a wild address) or
    /// dimension (as [`null_like`] and [`null_uav_desc`] avoid for the
    /// title's slots).
    fn make_guards(&self, last: u32) {
        let Some(host) = host() else { return };
        for view in GuardView::ALL {
            let at = self.cpu(view.slot(last));
            // Null views as for a title's description of that dimension.
            let texture = |dimension: D3D12_SRV_DIMENSION| {
                let mut words = [0; 10];
                words[1] = dimension.0 as u32;
                null_like(&words, at);
            };
            let unordered = |dimension: D3D12_UAV_DIMENSION| {
                null_uav(&null_uav_desc(Some(dimension.0 as u32)), at);
            };
            match view {
                GuardView::Texture2D => texture(D3D12_SRV_DIMENSION_TEXTURE2D),
                GuardView::Constant => {
                    let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
                    // SAFETY: a slot of our heap; a zero location makes a
                    // null view.
                    unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
                }
                GuardView::Buffer => null_buffer_view(at),
                GuardView::UnorderedBuffer => null_uav(&null_raw_uav_desc(), at),
                GuardView::UnorderedTexture2D => unordered(D3D12_UAV_DIMENSION_TEXTURE2D),
                GuardView::Texture1D => texture(D3D12_SRV_DIMENSION_TEXTURE1D),
                GuardView::Texture1DArray => texture(D3D12_SRV_DIMENSION_TEXTURE1DARRAY),
                GuardView::Texture2DArray => texture(D3D12_SRV_DIMENSION_TEXTURE2DARRAY),
                GuardView::Texture3D => texture(D3D12_SRV_DIMENSION_TEXTURE3D),
                GuardView::TextureCube => texture(D3D12_SRV_DIMENSION_TEXTURECUBE),
                GuardView::UnorderedTexture1D => unordered(D3D12_UAV_DIMENSION_TEXTURE1D),
                GuardView::UnorderedTexture1DArray => {
                    unordered(D3D12_UAV_DIMENSION_TEXTURE1DARRAY);
                }
                GuardView::UnorderedTexture2DArray => {
                    unordered(D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
                }
                GuardView::UnorderedTexture3D => unordered(D3D12_UAV_DIMENSION_TEXTURE3D),
            }
        }
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

/// A sampler heap of our own, for lists that set one of our root signatures
/// (`SAMPLER_HEAP_DIRECTLY_INDEXED`) while the title binds no sampler heap.
/// Its one slot starts as a point sampler (`HostHeap::new`), so an indexed
/// read of it returns something defined.
pub(crate) fn fallback_samplers() -> Option<&'static HostHeap> {
    static HEAP: std::sync::OnceLock<Option<HostHeap>> = std::sync::OnceLock::new();
    HEAP.get_or_init(|| HostHeap::new(true, 1)).as_ref()
}

/// For the profiler: where slots of a heap's `made` are claimed or looked
/// at, as the `site` of `made long hold` ([`MadeHold`]).
pub(crate) mod hold {
    /// A view the title wrote or copied (`mirror`, not for a draw).
    pub(crate) const VIEW: u64 = 1;
    /// A draw's bind of a table (`mirror` for a draw).
    pub(crate) const TABLE: u64 = 2;
    /// The walk of the heap buffer pass (`mirror_walked`).
    pub(crate) const WALK: u64 = 3;
    /// `sync_heaps` before a submission (`mirror_synced`).
    pub(crate) const SYNC: u64 = 4;
    /// The texture pass's uploads (`mirror_at`).
    pub(crate) const UPLOAD: u64 = 5;
    /// Slots that waited for their texture's host copy (`make_waiting`).
    pub(crate) const WAITING: u64 = 6;
    /// The texture pass's sweep over placed texture views.
    pub(crate) const SWEEP: u64 = 7;
    /// Levels on demand: views made again (`texture_views_remade`).
    pub(crate) const REMADE: u64 = 8;
    /// A texture's memory gone (`texture_slot_gone`).
    pub(crate) const GONE: u64 = 9;
    /// The texture pass asking whether a slot is made (`texture_slot_made_from`).
    pub(crate) const MADE_FROM: u64 = 10;
    /// The profile's sample of the made slots (`made_keys`).
    pub(crate) const SAMPLE: u64 = 11;
}

/// A stretch of work on a heap's slots at a site ([`hold`]) while the
/// profiler records, from [`HostHeap::hold`] until this is dropped: a
/// `made long hold` event if it lasted 200 µs or more, args `heap`, `site`
/// and `created` (host textures made or tried meanwhile, `crate::resource`'s
/// `host texture make`). Nothing while the profiler does not record.
struct MadeHold {
    start: u64,
    heap: u64,
    site: u64,
    created: u64,
}

impl Drop for MadeHold {
    fn drop(&mut self) {
        if self.start != 0 {
            crate::profile::since(
                "made long hold",
                self.start,
                200,
                &[
                    ("heap", self.heap),
                    ("site", self.site),
                    (
                        "created",
                        crate::profile::textures_made().wrapping_sub(self.created),
                    ),
                ],
            );
        }
    }
}

impl HostHeap {
    /// Work on slots at `site` ([`hold`]) begins: while the profiler
    /// records, it is timed ([`MadeHold`]).
    #[inline]
    fn hold(&self, site: u64) -> MadeHold {
        let start = crate::profile::stamp();
        let created = if start != 0 {
            crate::profile::textures_made()
        } else {
            0
        };
        MadeHold {
            start,
            heap: self.id,
            site,
            created,
        }
    }

    /// For the profile's sample (`VICTORY_PROFILE_SAMPLES`): the keys of
    /// the views the made slots were made from (0 for a slot not made), a
    /// slot at a time, each once no thread changes it.
    pub(crate) fn made_keys(&self) -> Vec<u64> {
        let _hold = self.hold(hold::SAMPLE);
        (0..self.count)
            .map(|index| self.claim(index, |_| false).err().unwrap_or(0))
            .collect()
    }
}

impl HostHeap {
    /// Monotonic unique identifier for this host heap, used to key per-heap
    /// caches without pointer-reuse aliasing.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The heap buffer pass's claim and walk.
    pub(crate) fn walk_gate(&self) -> &std::sync::Arc<WalkGate> {
        &self.walk
    }

    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        if self.sampler || n == 0 || n > self.dynamic {
            return None;
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicAsked, 1);
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let enough = self.free.load(std::sync::atomic::Ordering::Relaxed) >= n
            && n < self.too_long.load(std::sync::atomic::Ordering::Relaxed);
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts.take(if enough { DYNAMIC as usize } else { 0 }) {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            if enough {
                self.too_long.fetch_min(n as u32, std::sync::atomic::Ordering::Relaxed);
            }
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::DynamicRefused, 1);
                add(C::DynamicNs, started.elapsed().as_nanos() as u64);
            }
            return None;
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::DynamicNs, started.elapsed().as_nanos() as u64);
        taken[first..first + n].fill(true);
        self.free.fetch_sub(n as u32, std::sync::atomic::Ordering::Relaxed);
        *next = ((first + n) % DYNAMIC as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The dynamic slots free now (tests: what a list took is given back).
    #[cfg(test)]
    pub(crate) fn free_dynamic(&self) -> u32 {
        self.free.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// The GPU address of the table of element strides by title slot (view
    /// heaps).
    pub(crate) fn strides_gpu(&self) -> Option<u64> {
        self.strides.as_ref().map(|s| s.1)
    }

    /// The table's buffer, for a list to hold.
    pub(crate) fn strides_buffer(&self) -> Option<&ID3D12Resource> {
        self.strides.as_ref().map(|s| &s.0)
    }

    /// Up to `max` title slots whose texture view is due at `frame`, each
    /// with its write number, in the order to serve them
    /// ([`TextureQueue::due`]), and the queue's counts
    /// ([`TextureQueue::census`]).
    pub(crate) fn texture_slots_due(&self, frame: u64, max: usize) -> (Vec<(u32, u64)>, [u64; 5]) {
        let textures = self.texture_queue();
        (if max == 0 { Vec::new() } else { textures.due(frame, max) }, textures.census(frame))
    }

    /// Every title slot of the queue that is due at `frame`, and whether
    /// it was uploaded before (in part, or whole and waiting for another
    /// look): for the card accounting line
    /// (`crate::resource::card_accounting`).
    pub(crate) fn texture_slots_all_due(&self, frame: u64) -> Vec<(u32, bool)> {
        let textures = self.texture_queue();
        textures
            .slots
            .iter()
            .filter(|(_, w)| w.due <= frame)
            .map(|(&index, w)| (index, w.mapped > 0 || w.checked.2 != 0))
            .collect()
    }

    /// The texture view of title slot `index` (as handed out with `write`)
    /// was brought up to date: due again from frame `again` (uploaded with
    /// that many bytes of its memory there, not all), or not until the
    /// slot is written again.
    pub(crate) fn texture_slot_done(&self, index: u32, write: u64, again: Option<(u64, u64)>) {
        let mut textures = self.texture_queue();
        match again {
            Some((frame, mapped)) => textures.partly(index, write, frame, mapped),
            None => textures.done(index, write),
        }
    }

    /// Title slot `index` was uploaded whole: see [`TextureQueue::recheck`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn texture_slot_recheck(
        &self,
        index: u32,
        write: u64,
        due: u64,
        at: (u64, u64),
        print: u64,
        looks: u8,
        uploaded: u64,
    ) {
        let mut textures = self.texture_queue();
        textures.recheck(index, write, due, at, print, looks, uploaded);
    }

    /// See [`TextureQueue::checked`].
    pub(crate) fn texture_slot_checked(&self, index: u32) -> Option<(u64, u64, u64, u8, u64)> {
        self.texture_queue().checked(index)
    }

    /// Nothing to upload at title slot `index` now (its memory is not
    /// mapped, or no more of it than before): looked at again after 1, 2,
    /// 4, ... frames, at most `most`.
    pub(crate) fn texture_slot_later(&self, index: u32, write: u64, frame: u64, most: u64) {
        let mut textures = self.texture_queue();
        textures.later(index, write, frame, most);
    }

    /// See [`TextureQueue::hold`].
    pub(crate) fn texture_slot_held(&self, index: u32, write: u64, due: u64) {
        self.texture_queue().hold(index, write, due);
    }

    /// See [`TextureQueue::room`].
    pub(crate) fn texture_slots_room(&self, frame: u64) -> usize {
        self.texture_queue().room(frame)
    }

    /// Bytes of title slot `index`'s memory that were mapped when it was
    /// last uploaded in part (0: not uploaded, or whole).
    pub(crate) fn texture_slot_mapped(&self, index: u32) -> u64 {
        self.texture_queue().mapped(index)
    }

    /// Takes the ranges of title memory the runtime mapped, unmapped or
    /// wrote since the last call ([`remapped`]) and makes the slots whose
    /// placed texture lies there due from the next frame (by when a write
    /// under way has landed): memory that arrived is uploaded, memory that
    /// went makes the slot null ([`HostHeap::texture_slot_gone`]), and a
    /// texture streamed into the place of another of the same description
    /// is uploaded although no descriptor says so. Slots tables read are
    /// left to their draws. Returns how many slots it made due.
    /// shortcut: the ranges are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    pub(crate) fn texture_slots_remapped(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        self.texture_slots_page_written(frame);
        let ranges = std::mem::take(&mut *REMAPPED.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_in(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        // A table's slot waits for its draw, as a write of it does: the
        // draw takes it out again. Left out altogether, a slot that shaders
        // also pick by index kept what it showed before.
        let lately: Vec<bool> = slots.iter().map(|&slot| self.tabled_lately(slot, frame)).collect();
        let mut textures = self.texture_queue();
        for (&slot, lately) in slots.iter().zip(lately) {
            textures.changed(slot, frame, frame + if lately { TABLED_GRACE } else { 1 });
        }
        STALE[3].fetch_add(slots.len() as u64, Relaxed);
        slots.len()
    }

    /// Goes on with a look at all of the heap's placed texture views, a
    /// part at a time: a view whose memory is there, whose host slot shows
    /// nothing (or another view) and which is not in the queue will stay
    /// that way until the title writes it again, and a shader that picks it
    /// reads black. One whose memory no newer view describes otherwise goes
    /// back into the queue; the others are counted ([`SWEPT`]).
    /// `VICTORY_TEXTURE_SWEEP=0` turns it off, `count` only counts.
    pub(crate) fn texture_sweep(&self, heap: &super::Heap, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        /// Slots looked at at a time, and the least time between.
        const SLOTS: u32 = 1 << 15;
        const EVERY_MS: u64 = 100;
        static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
        let mode = *MODE.get_or_init(|| match std::env::var("VICTORY_TEXTURE_SWEEP").as_deref() {
            Ok("0") => 0,
            Ok("count") => 1,
            _ => 2,
        });
        if mode == 0 || self.sampler {
            return;
        }
        let now = crate::host::now_ms();
        let Ok(mut sweep) = SWEEP.try_lock() else {
            return;
        };
        if now.saturating_sub(sweep.2) < EVERY_MS {
            return;
        }
        sweep.2 = now;
        let (from, mut counts) = (sweep.0.min(self.count), sweep.1);
        let to = from.saturating_add(SLOTS).min(self.count);
        let _span = crate::profile::span_args(
            "texture sweep",
            &[("heap", self.id), ("slots", u64::from(to - from))],
        );
        self.sweep_part(
            from..to,
            |index| heap.handle(index),
            frame,
            mode == 2,
            &mut counts,
        );
        sweep.0 = to;
        sweep.1 = counts;
        if to >= self.count {
            for (to, counted) in SWEPT.iter().zip(counts) {
                to.store(counted, Relaxed);
            }
            (sweep.0, sweep.1) = (0, [0; 10]);
        }
    }

    /// [`HostHeap::texture_sweep`] of some slots, `handle` saying where the
    /// title's descriptors lie.
    fn sweep_part(
        &self,
        slots: std::ops::Range<u32>,
        handle: impl Fn(u32) -> Option<u64>,
        frame: u64,
        put_back: bool,
        counts: &mut [u64; 10],
    ) {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        /// Put back per part, and only while the queue is shorter than
        /// this: a view the title forgot to clear may be among them, and
        /// the queue serves the newest first.
        const MOST: u32 = 128;
        const SHORT: usize = 4_000;
        let mut back = 0;
        // The slots whose host slot does not show what the title's says.
        let mut unmade = Vec::new();
        {
            let _hold = self.hold(hold::SWEEP);
            // The memory is the title's, and a heap may be larger than what
            // of it is there: asked once per page.
            let mut page = (u64::MAX, false);
            for index in slots {
                let Some(handle) = handle(index) else {
                    break;
                };
                if page.0 != handle >> 12 {
                    page = (
                        handle >> 12,
                        crate::texture::committed(handle as *const u8, 12),
                    );
                }
                if !page.1 {
                    continue;
                }
                // SAFETY: a descriptor slot of the title's heap, there.
                let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
                if tag != VIEW_TAG || generation >> super::KIND_SHIFT == super::CONSTANT {
                    continue;
                }
                let key = u64::from(generation) << 32 | u64::from(i);
                let shown = self
                    .made
                    .get(index as usize)
                    .map_or(0, |made| made.load(Acquire));
                // (A slot another thread is making: counted as one that
                // shows what the title's says, which it is about to.)
                if shown == key || shown == CLAIMED {
                    counts[0] += 1;
                    counts[1] += 1;
                } else {
                    unmade.push((index, i, generation, shown != 0));
                }
            }
        }
        for (index, i, generation, other) in unmade {
            let Some(View::Placed(false, resource, desc)) = view_at(i, generation) else {
                continue;
            };
            if resource[0] == 1 {
                continue;
            }
            counts[0] += 1;
            let queued = self.texture_queue().slots.contains_key(&index);
            if queued {
                counts[2] += 1;
                continue;
            }
            if self.tabled_lately(index, frame) {
                counts[3] += 1;
                continue;
            }
            let start = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            if !crate::texture::committed(start as *const u8, 1) {
                counts[4] += 1;
                continue;
            }
            let (claim, against) = {
                let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
                let Some(bytes) = placed.bytes(&resource) else {
                    continue;
                };
                placed.claim(index, start, start.saturating_add(bytes))
            };
            counts[5 + claim as usize] += 1;
            counts[9] += u64::from(other);
            static SAID: [std::sync::atomic::AtomicU32; 4] =
                [const { std::sync::atomic::AtomicU32::new(0) }; 4];
            if SAID[claim as usize].fetch_add(1, Relaxed) < 12 {
                let (width, height, format) = (resource[4], resource[6], resource[8]);
                tracing::info!(
                    "a texture view nothing brings up to date: slot {index}, {claim:?}, {width}x{height} f{format} \
                     mips {} at {start:#x}, shows {}; against {against:x?}; its view {:x?}",
                    resource[7] >> 16,
                    if other { "another view" } else { "nothing" },
                    &desc[..10]
                );
            }
            if put_back && back < MOST && matches!(claim, Claim::Own | Claim::Free) {
                let mut textures = self.texture_queue();
                if textures.slots.len() < SHORT {
                    textures.written(index, frame, frame + 1, false);
                    back += 1;
                    SWEPT[10].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Takes the ranges of title memory written behind the guards' back
    /// since the last call ([`page_written`]) and finds the slots whose
    /// placed texture holds bytes of them ([`Spans::slots_holding`]). A
    /// slot whose texture was uploaded before the write shows the bytes
    /// from before: with `VICTORY_PAGE_WRITES` (`crate::watch::page_writes`,
    /// bit 2) it is due again [`PAGE_WRITE_AFTER`] frames on, as memory
    /// mapped under it would make it ([`HostHeap::texture_slots_remapped`]);
    /// its host copy exists, so the upload needs no room on the card. A
    /// slot still waiting for its first upload is left as it is: that
    /// upload takes the bytes that are there by then. Slots tables read
    /// are left to their draws. Counted either way; returns how many slots
    /// it made due.
    ///
    /// The GPU's writes into homes in chunks of the memory pool
    /// ([`pool_written`]) are taken here too, and their slots are always
    /// made due, from the next frame: those bytes are new for certain, as a
    /// copy's write-back would have said, and landed before the wait that
    /// told them (no write under way to wait for).
    fn texture_slots_page_written(&self, frame: u64) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        let pool = std::mem::take(&mut *POOL_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        let mut due = 0;
        if !pool.is_empty() {
            let [held, uploaded, made] = self.slots_written_under(&pool, frame, Some(1));
            POOL_WRITES[2].fetch_add(held as u64, Relaxed);
            POOL_WRITES[3].fetch_add(uploaded as u64, Relaxed);
            due += made;
        }
        let ranges = std::mem::take(&mut *PAGE_WRITTEN.lock().unwrap_or_else(|e| e.into_inner()));
        if ranges.is_empty() {
            return due;
        }
        let after = (crate::watch::page_writes() & 2 != 0).then_some(PAGE_WRITE_AFTER);
        let [held, uploaded, made] = self.slots_written_under(&ranges, frame, after);
        PAGE_WRITES[2].fetch_add(held as u64, Relaxed);
        PAGE_WRITES[3].fetch_add(uploaded as u64, Relaxed);
        due + made
    }

    /// [`HostHeap::texture_slots_page_written`] for `ranges` (start, end):
    /// the slots that held bytes of them, those of them uploaded before,
    /// and those made due (the latter only with `due`: that many frames on).
    fn slots_written_under(
        &self,
        ranges: &[(u64, u64)],
        frame: u64,
        due: Option<u64>,
    ) -> [usize; 3] {
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            ranges.iter().flat_map(|&(start, end)| placed.slots_holding(start, end)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let held = slots.len();
        let mut textures = self.texture_queue();
        // Uploaded before: out of the queue, or in it after an upload (in
        // part, or whole and being looked at again).
        // (Counted for the black census: a slot out of the queue is one
        // the pass was done with; without `due` its write goes unnoticed.)
        let mut out_of_queue = 0u64;
        slots.retain(|slot| match textures.slots.get(slot) {
            None => {
                out_of_queue += 1;
                true
            }
            Some(w) => w.mapped > 0 || w.checked.2 != 0,
        });
        crate::census::PAGE_WRITES_LOST[0].fetch_add(out_of_queue, std::sync::atomic::Ordering::Relaxed);
        let Some(after) = due else {
            crate::census::PAGE_WRITES_LOST[1].fetch_add(out_of_queue, std::sync::atomic::Ordering::Relaxed);
            return [held, slots.len(), 0];
        };
        for &slot in &slots {
            textures.changed(slot, frame, frame + after);
        }
        [held, slots.len(), slots.len()]
    }

    /// Whether host slot `index` is made from the very record the title
    /// has at `handle` now. Its host texture then exists (a slot keeps what
    /// it views), so bringing that texture up to date takes no memory of
    /// the card.
    pub(crate) fn texture_slot_made_from(&self, index: u32, handle: u64) -> bool {
        if index >= self.count {
            return false;
        }
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        let key = u64::from(generation) << 32 | u64::from(i);
        if tag != VIEW_TAG {
            return false;
        }
        // And the slot holds a host resource: a slot is also noted as made
        // from a view the host could not make (`make` gave nothing, the
        // slot is null and keeps nothing). The host copy of such a
        // texture is held by no slot, may have been let go, and would be
        // made anew here, with the card full. `made` first, then `kept`,
        // as `mirror` takes them (the slot claimed, so that no other
        // thread changes it between the two).
        // A slot settled on another record says so before the claim (as a
        // look just before it was made would). A slot another thread has
        // claimed goes on to the claim: many claims leave the slot as it was
        // (`texture_slot_gone` of a null slot, this look's own), and a "no"
        // for those would count the texture as refused.
        if self.made.get(index as usize).is_some_and(|made| {
            let now = made.load(std::sync::atomic::Ordering::Acquire);
            now != key && now != CLAIMED
        }) {
            return false;
        }
        let _hold = self.hold(hold::MADE_FROM);
        let Ok(claim) = self.claim(index, |_| true) else {
            return false;
        };
        claim.old != 0 && claim.old == key && self.kept_at(index, |kept| kept.is_some())
    }

    /// Levels on demand (`crate::tiles`): makes again the host slots of the
    /// placed textures whose first level with memory changed since the
    /// last call ([`levels_changed`]), each from the view it was made
    /// from, so that it shows the texture from that level ([`make`]). Only
    /// slots that are made; a slot draws read through a table is made by
    /// its draw. Returns how many it made.
    /// shortcut: the starts are kept in one place for all heaps; the heap
    /// whose pass runs first takes them (a title has one view heap).
    /// Levels on demand (`crate::tiles`): the card has room, so the slots
    /// of up to [`Self::WAITING_A_PASS`] placed textures whose upload left
    /// a held level without tiles at least a second ago
    /// ([`levels_waiting`]) are due again from the next frame, as memory
    /// mapped under them would make them
    /// ([`HostHeap::texture_slots_remapped`]): the upload then gives the
    /// level its tiles. To be called only while the card has room: without
    /// it the upload would give nothing again. Slots tables read are left
    /// to their draws (`texture.rs` `upload` asks again for those). Returns
    /// how many slots it made due.
    /// shortcut: kept in one place for all heaps, as the starts below.
    pub(crate) fn texture_slots_waiting(&self, frame: u64) -> usize {
        if self.sampler {
            return 0;
        }
        let starts: Vec<u64> = {
            let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
            if waiting.is_empty() {
                return 0;
            }
            let starts: Vec<u64> = waiting
                .iter()
                .filter(|&(_, &since)| crate::host::long_ago(since, 1000, 2))
                .map(|(&start, _)| start)
                .take(Self::WAITING_A_PASS)
                .collect();
            for start in &starts {
                waiting.remove(start);
            }
            starts
        };
        if starts.is_empty() {
            return 0;
        }
        let mut slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        slots.sort_unstable();
        slots.dedup();
        slots.retain(|&slot| !self.tabled_lately(slot, frame));
        let mut textures = self.texture_queue();
        for &slot in &slots {
            textures.changed(slot, frame, frame + 1);
        }
        crate::tiles::STATS[10].fetch_add(slots.len() as u64, std::sync::atomic::Ordering::Relaxed);
        slots.len()
    }

    /// Textures a pass asks again for their levels
    /// ([`HostHeap::texture_slots_waiting`]): the first ones given tiles
    /// may take the room the others were asked for, and an upload that
    /// gives nothing is a whole texture staged and copied for nothing.
    const WAITING_A_PASS: usize = 128;

    pub(crate) fn texture_views_remade(&self) -> usize {
        if self.sampler {
            return 0;
        }
        let mut starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        // Each start once: a slot is made again once a pass.
        starts.sort_unstable();
        starts.dedup();
        // Ordered with a writer's claim and its look at [`LEVELS_SEEN`]:
        // a slot this pass finds not made from its record yet was claimed
        // after the count moved, and its maker resolves the view again.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let mut span = crate::profile::span_args("views remade", &[("heap", self.id)]);
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        if let Some(span) = span.as_mut() {
            span.arg("slots", slots.len() as u64);
        }
        // The record each made slot was made from (looked at once no
        // thread changes it), and what its view is made from, found with no
        // claim held ([`resolve`]); then each slot still made from that
        // record is written under its claim. A slot made from another
        // record meanwhile was made by its maker from the levels there now
        // ([`LEVELS_SEEN`]).
        let found: Vec<(u32, u64, Resolved)> = slots
            .into_iter()
            .filter_map(|slot| {
                let key = self.claim(slot, |_| false).err().filter(|&key| key != 0)?;
                let view = view_at(key as u32, (key >> 32) as u32)?;
                Some((slot, key, resolve(&view)?))
            })
            .collect();
        #[cfg(test)]
        if let Some(hook) = BEFORE_REMADE.take() {
            hook(self);
        }
        let _hold = self.hold(hold::REMADE);
        let mut remade = 0;
        for (slot, key, resolved) in found {
            let Ok(_claim) = self.claim(slot, |now| now == key) else {
                continue;
            };
            if let Some(kept) = write(&resolved, self.cpu(slot)) {
                self.keep(slot, kept);
                remade += 1;
            }
        }
        if let Some(span) = span.as_mut() {
            span.arg("remade", remade as u64);
        }
        remade
    }

    /// The memory of the texture title slot `index` views is not mapped
    /// (any more): a host slot made from it becomes null again, so that
    /// the host copy can be let go (`crate::resource::let_go_idle`; a slot
    /// keeps what it views). True if the slot was made.
    ///
    /// With `VICTORY_TEXTURE_AWAY_MS` ([`away_ms`]) not at the first look,
    /// only once the memory is still away that long after it was first
    /// found away (the slot made from the same view meanwhile).
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let _hold = self.hold(hold::GONE);
        let Ok(mut claim) = self.claim(index, |key| key != 0) else {
            return false;
        };
        let wait = away_ms();
        if wait != 0 {
            let now = crate::host::now_ms();
            let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
            match away.get(&index) {
                Some(&(was, since)) if was == claim.value && now.saturating_sub(since) < wait => {
                    return false;
                }
                Some(&(was, _)) if was == claim.value => {
                    away.remove(&index);
                }
                _ => {
                    away.insert(index, (claim.value, now));
                    return false;
                }
            }
        }
        self.unmake(&mut claim, index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    }

    /// The memory of the texture title slot `index` views is there (again):
    /// see [`HostHeap::texture_slot_gone`]. Nothing without [`away_ms`].
    pub(crate) fn texture_slot_here(&self, index: u32) {
        if away_ms() == 0 {
            return;
        }
        let mut away = self.away.lock().unwrap_or_else(|e| e.into_inner());
        if !away.is_empty() {
            away.remove(&index);
        }
    }

    /// Makes host slot `index`, claimed by `claim`, null again, as it is
    /// before its view is made: a null view of the kind it held. (Its word
    /// fits that kind already.)
    fn unmake(&self, claim: &mut Claimed<'_>, index: u32) {
        let key = std::mem::take(&mut claim.value);
        if key == 0 {
            return;
        }
        match view_at(key as u32, (key >> 32) as u32) {
            Some(view) => null_for(&view, self.cpu(index)),
            None => null_view(self.cpu(index)),
        }
        self.keep(index, None);
    }

    /// Makes null the host slots a placed texture view written into
    /// another slot made stale ([`HostHeap::note_buffer`]: each with what
    /// [`HostHeap::spanned`] counted of it then), a slot at a time, after the
    /// write's own slot is made (a thread waits for a claim only while it
    /// holds none, so two writes that make each other's slots stale do not
    /// wait for each other): a slot is made null only if no write of it
    /// looked at the spans since, so what another thread wrote into it
    /// after the stale verdict is left as it is, and what it wrote before
    /// is made null, as one after the other would have it.
    fn unmake_stale(&self, stale: &[(u32, u32)]) {
        use std::sync::atomic::Ordering::Relaxed;
        for &(slot, writes) in stale {
            let Ok(mut claim) = self.claim(slot, |key| key != 0) else {
                continue;
            };
            // (Under the claim: the writes of the slot are, too.)
            let now = self
                .spanned
                .get(slot as usize)
                .map(|s| s.load(Relaxed) & !IN_SPANS);
            if now.is_some_and(|now| now != writes) {
                continue;
            }
            self.unmake(&mut claim, slot);
        }
    }

    /// Title slot `index` is a table's ([`HostHeap::tabled`]). `read`: a
    /// draw read it through the table and brought what it views up to
    /// date, so the slot leaves the queue of textures picked by index; the
    /// caller makes the host slot in place ([`HostHeap::mirror`]), as the
    /// pass would have, for a shader that picks it by index all the same.
    pub(crate) fn texture_slot_tabled(&self, index: u32, read: bool) {
        self.tabled_in(index, read, crate::frame::current());
    }

    /// [`HostHeap::texture_slot_tabled`] in frame `frame`.
    fn tabled_in(&self, index: u32, read: bool, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(stamp) = self.tabled.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        let old = stamp.load(Relaxed);
        if !read {
            if old & !TABLED_READ != now {
                stamp.store(now, Relaxed);
            }
            return;
        }
        // Once per slot and frame: tables are read by every draw.
        if old != now | TABLED_READ {
            stamp.store(now | TABLED_READ, Relaxed);
            if tabled_out() {
                let mut textures = self.texture_queue();
                if textures.slots.remove(&index).is_some() {
                    TABLED_WRITES[1].fetch_add(1, Relaxed);
                }
            }
        }
    }

    /// Title slot `index`, a buffer view's or none, lies in a table a draw
    /// reads or had made afresh: what it holds is the draw's to make
    /// ([`HostHeap::buffer_slot_tabled_lately`]; `VICTORY_HEAP_PASS_TABLED=1`
    /// alone calls this).
    pub(crate) fn buffer_slot_tabled(&self, index: u32) {
        self.buffer_tabled_in(index, crate::frame::current());
    }

    /// [`HostHeap::buffer_slot_tabled`] in frame `frame`: in
    /// [`HostHeap::buffer_tabled`], not in `tabled`.
    pub(crate) fn buffer_tabled_in(&self, index: u32, frame: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.sampler || index >= self.count {
            return;
        }
        let stamps = self.buffer_tabled.get_or_init(|| {
            (0..self.count)
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect()
        });
        let Some(stamp) = stamps.get(index as usize) else {
            return;
        };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        if stamp.load(Relaxed) != now {
            stamp.store(now, Relaxed);
        }
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_BUFFER_FRAMES`] frames before `frame`: the heap buffer pass
    /// leaves it to the draws that read it through their tables
    /// (`VICTORY_HEAP_PASS_TABLED=1`). Not asked with the switch off.
    pub(crate) fn buffer_slot_tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self
            .buffer_tabled
            .get()
            .and_then(|stamps| stamps.get(index as usize))
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed));
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        stamp != 0 && now.wrapping_sub(stamp) & !TABLED_READ <= TABLED_BUFFER_FRAMES
    }

    /// Whether title slot `index` was a table's within the last
    /// [`TABLED_FRAMES`] frames before `frame` (never, if such slots are
    /// to be queued as any other).
    fn tabled_lately(&self, index: u32, frame: u64) -> bool {
        let stamp = self.tabled_stamp(index) & !TABLED_READ;
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        tabled_out() && stamp != 0 && u64::from(now.wrapping_sub(stamp)) <= TABLED_FRAMES
    }

    /// What `tabled` holds for title slot `index`; 0: never a table's.
    fn tabled_stamp(&self, index: u32) -> u32 {
        self.tabled
            .get(index as usize)
            .map_or(0, |s| s.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// The queue's part of a write of title slot `index` in `frame`:
    /// `texture`, whether it now holds a view of a texture the title may
    /// fill; `seen` as for [`HostHeap::note_buffer`].
    fn queue_texture(&self, index: u32, texture: bool, seen: bool, frame: u64) {
        let stamp = if tabled_out() { self.tabled_stamp(index) } else { 0 };
        let now = (frame as u32).wrapping_add(1) & !TABLED_READ;
        // A table's slot lately: the draw that reads it uploads its
        // texture, so its write waits for that draw.
        let lately = stamp != 0 && u64::from(now.wrapping_sub(stamp & !TABLED_READ)) <= TABLED_FRAMES;
        // Made now by a draw of this frame that read it through a table:
        // up to date.
        let read_now = seen && stamp == now | TABLED_READ;
        let note = if texture && !read_now {
            let due = if lately && !seen {
                TABLED_WRITES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                frame + TABLED_GRACE
            } else {
                0
            };
            TextureNote::Written {
                index,
                frame,
                due,
                seen,
            }
        } else {
            TextureNote::Remove(index)
        };
        self.note_textures(&[note]);
    }

    /// Takes the slots a write made stale ([`Spans::note`]) out of the
    /// queue of textures picked by index.
    fn remove_stale(&self, stale: &[(u32, u32)]) {
        if stale.is_empty() {
            return;
        }
        STALE[0].fetch_add(stale.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let removed: Vec<TextureNote> = stale
            .iter()
            .map(|&(slot, _)| TextureNote::Remove(slot))
            .collect();
        self.note_textures(&removed);
    }

    /// Changes to the queue of textures picked by index from a write of a
    /// slot: logged ([`NoteLog`]). A writer that finds the log long applies
    /// it itself, if the queue is free: the texture pass may leave a heap
    /// alone for a long time.
    fn note_textures(&self, notes: &[TextureNote]) {
        if self.log.textures(notes) {
            drop(self.texture_queue_if_free());
        }
    }

    /// The queue of textures picked by index, with every change logged to
    /// it so far applied ([`NoteLog`]).
    fn texture_queue(&self) -> std::sync::MutexGuard<'_, TextureQueue> {
        let textures = self.textures.lock().unwrap_or_else(|e| e.into_inner());
        self.texture_notes_applied(textures)
    }

    /// [`HostHeap::texture_queue`], unless another thread holds the queue.
    fn texture_queue_if_free(&self) -> Option<std::sync::MutexGuard<'_, TextureQueue>> {
        let textures = match self.textures.try_lock() {
            Ok(textures) => textures,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        Some(self.texture_notes_applied(textures))
    }

    fn texture_notes_applied<'a>(
        &self,
        mut textures: std::sync::MutexGuard<'a, TextureQueue>,
    ) -> std::sync::MutexGuard<'a, TextureQueue> {
        // (Taken under the queue's lock: the changes of two callers are
        // applied in the order they were logged.)
        for note in self.log.take_textures() {
            note.apply(&mut textures);
        }
        textures
    }

    /// The set of title slots holding buffer views, with every change
    /// logged to it and to the set of changes so far applied ([`NoteLog`]).
    fn buffer_set(&self) -> std::sync::MutexGuard<'_, std::collections::BTreeSet<u32>> {
        let buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        self.buffer_notes_applied(buffers)
    }

    /// [`HostHeap::buffer_set`], unless another thread holds the set.
    fn buffer_set_if_free(
        &self,
    ) -> Option<std::sync::MutexGuard<'_, std::collections::BTreeSet<u32>>> {
        let buffers = match self.buffers.try_lock() {
            Ok(buffers) => buffers,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return None,
        };
        Some(self.buffer_notes_applied(buffers))
    }

    fn buffer_notes_applied<'a>(
        &self,
        mut buffers: std::sync::MutexGuard<'a, std::collections::BTreeSet<u32>>,
    ) -> std::sync::MutexGuard<'a, std::collections::BTreeSet<u32>> {
        let notes = self.log.take_buffers();
        if !notes.is_empty() {
            let mut changed = self.changed.lock().unwrap_or_else(|e| e.into_inner());
            for note in notes {
                if note.apply(&mut buffers) {
                    changed.insert(note.index);
                }
            }
        }
        buffers
    }

    /// The title slots that hold buffer views.
    pub(crate) fn buffer_slots(&self) -> Vec<u32> {
        if let Some(bits) = &self.bits {
            return bits.buffers.all();
        }
        self.buffer_set().iter().copied().collect()
    }

    /// Of [`HostHeap::buffer_slots`], those of one of `parts` parts: every
    /// slot is in the same part each time (`VICTORY_HEAP_SPREAD=1`).
    pub(crate) fn buffer_slots_of(&self, part: u32, parts: u32) -> Vec<u32> {
        let mut slots = self.buffer_slots();
        slots.retain(|slot| slot % parts == part);
        slots
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
        if let Some(bits) = &self.bits {
            return bits.changed.take();
        }
        // The set of buffer views first: what is logged goes into both.
        drop(self.buffer_set());
        let mut changed = self.changed.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *changed).into_iter().collect()
    }

    /// Records what title slot `index` now holds for shaders that pick
    /// buffers from the heap: whether it is a buffer view at all (the
    /// caller writes the table of strides, [`HostHeap::write_word`]).
    /// `seen`: the slot was not written now, a draw or a pass made its
    /// host slot from what it held. `mark`: a slot that is or was a buffer
    /// view goes into the set of changes ([`HostHeap::mirror_walked`]).
    /// Returns the title slots whose placed texture view the write made
    /// stale ([`Spans`]), each with what [`HostHeap::spanned`] counted of
    /// it then: out of the queue of textures already, their host slots are
    /// the caller's to make null ([`HostHeap::unmake_stale`]). The caller
    /// holds the claim on slot `index` ([`HostHeap::claim`]).
    fn note_buffer(
        &self,
        index: u32,
        view: Option<&View>,
        seen: bool,
        mark: bool,
    ) -> Vec<(u32, u32)> {
        if self.sampler {
            return Vec::new();
        }
        // Measured for the texture side of the same gap (textures a shader
        // picks by a computed index are neither made, if placed, nor
        // uploaded): slots written with a placed texture view, and with a
        // view of a texture object.
        let texture = match view {
            Some(View::Placed(false, resource, _)) if resource[0] != 1 => {
                PLACED_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            Some(View::Shader(r, _)) if crate::resource::buffer_bytes(*r as crate::object::This).is_none() => {
                OBJECT_TEXTURE_SLOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                true
            }
            _ => false,
        };
        let frame = crate::frame::current();
        // The memory a placed texture view describes. Not for a slot
        // tables read (thousands of writes a frame, each read by its
        // draw): no pass looks at it, and the same texture's view in a
        // slot shaders pick by index says what it says.
        let placed_view = match view {
            Some(View::Placed(false, resource, desc))
                if resource[0] != 1 && !self.tabled_lately(index, frame) =>
            {
                // (And the level the view starts at: word 4 of a texture
                // view's description, after format, dimension, mapping
                // and a pad.)
                Some((resource, u64::from(desc[10]) | u64::from(desc[11]) << 32, desc[4]))
            }
            _ => None,
        };
        // A write of anything but a placed texture view into a slot no such view was noted for has
        // nothing to forget in the spans, and leaves their lock alone. No
        // other write makes such a slot stale meanwhile, so its change to
        // the queue is logged without it.
        let spanned = self.spanned.get(index as usize);
        let stale = if placed_view.is_none()
            && spanned.is_some_and(|spanned| {
                spanned.load(std::sync::atomic::Ordering::Relaxed) & IN_SPANS == 0
            }) {
            self.queue_texture(index, texture, seen, frame);
            Vec::new()
        } else {
            use std::sync::atomic::Ordering::Relaxed;
            let mut placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            // A slot a time, the slot's own change to the queue and the
            // removals of the slots it makes stale are logged under the
            // spans' lock: in the order the spans saw the writes. A removal
            // logged after the stale slot's own newer write, by a thread
            // that does not hold its claim, took that write's slot out of
            // the queue while the spans held it.
            self.queue_texture(index, texture, seen, frame);
            let span = placed_view.and_then(|(resource, start, first)| {
                Some((start, start.saturating_add(placed.bytes(resource)?), first))
            });
            let stale = match span {
                Some((start, end, first)) => {
                    SPANNED[0].fetch_min(start, Relaxed);
                    SPANNED[1].fetch_max(end, Relaxed);
                    placed.rule = stale_rule();
                    let stale = placed.note(index, start, end, !seen);
                    placed.shows(index, first);
                    stale
                }
                None => {
                    placed.forget(index);
                    Vec::new()
                }
            };
            // Each stale slot with the count of its writes so far.
            let stale: Vec<(u32, u32)> = stale
                .into_iter()
                .map(|slot| {
                    let seen = self.spanned.get(slot as usize);
                    (slot, seen.map_or(0, |seen| seen.load(Relaxed) & !IN_SPANS))
                })
                .collect();
            // (A view that clashes with the spans is left out of them.)
            if let Some(spanned) = spanned {
                let writes = (spanned.load(Relaxed) & !IN_SPANS).wrapping_add(IN_SPANS << 1);
                let held = if placed.by_slot.contains_key(&index) {
                    IN_SPANS
                } else {
                    0
                };
                spanned.store(writes | held, Relaxed);
            }
            self.remove_stale(&stale);
            stale
        };
        let note = BufferNote {
            index,
            buffer: view.and_then(view_stride).is_some(),
            mark,
        };
        if let Some(bits) = &self.bits {
            // Under the slot's claim, as the log's: two writes of one slot
            // change its bits in their order, and those of other slots
            // other bits.
            note.apply_bits(bits);
        } else if self.log.buffers(note) {
            // Applied by the writer once the log is long, as the queue's
            // (`note_textures`): the heap buffer pass runs only for lists
            // whose shaders pick buffers from the heap.
            drop(self.buffer_set_if_free());
        }
        stale
    }

    /// Writes `word` ([`slot_word`]) into the table of strides for title
    /// slot `index`. Shaders read the table and then the slot while we
    /// write both, so a slot being rewritten gets 0 before its descriptor
    /// (a null view to them) and its own word only after it: a word that
    /// said "buffer" over a texture view still in the slot had the texture
    /// read as a buffer, a page fault on AMD. A fence after a 0 and before
    /// any other word keeps the stores in that order for the card (both
    /// are write-combined memory). This narrows the window, it does not
    /// close it: a shader that read the old word just before may still
    /// read the new descriptor.
    fn write_word(&self, index: u32, word: u32) {
        debug_assert!(
            index >= self.count || claims_held() > 0,
            "title slot {index} changed without its claim"
        );
        if let Some((_, _, data)) = &self.strides
            && index < victory_shader::emit::SLOT_STRIDES_COUNT
        {
            let fence = || std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            if word != 0 {
                fence();
            }
            // SAFETY: inside the mapping; shaders read it while we write (a
            // dword, either value is one the slot had).
            unsafe { (*data as *mut u32).add(index as usize).write_volatile(word) };
            if word == 0 {
                fence();
            }
        }
    }

    /// The word the table of strides holds for title slot `index` (tests).
    #[cfg(test)]
    pub(crate) fn word(&self, index: u32) -> Option<u32> {
        let (_, _, data) = self.strides.as_ref()?;
        (index < victory_shader::emit::SLOT_STRIDES_COUNT)
            // SAFETY: inside the mapping.
            .then(|| unsafe { (*data as *const u32).add(index as usize).read_volatile() })
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            let given = slots.iter().filter(|taken| **taken).count() as u32;
            self.free.fetch_add(given, std::sync::atomic::Ordering::Relaxed);
            self.too_long.store(u32::MAX, std::sync::atomic::Ordering::Relaxed);
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot.
    pub(crate) fn make_at(&self, index: u32, view: &View) {
        let kept = make(view, self.cpu(index));
        if kept.is_none() {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        // Every change of a title slot goes through its claim, which writes
        // `made` and `left` ([`Claimed`]); dynamic slots have neither.
        debug_assert!(
            index >= self.count || claims_held() > 0,
            "title slot {index} changed without its claim"
        );
        let Some(shard) = self.kept.get((index / KEPT_SHARD) as usize) else {
            return;
        };
        let mut kept = shard.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut((index % KEPT_SHARD) as usize) {
            // A running submission may still read the old resource through
            // this slot.
            if let Some(old) = std::mem::replace(slot, resource) {
                crate::host::retire(old);
            }
        }
    }

    /// At the title's end (`crate::release`, with the card done with
    /// everything sent): the slots whose host resource `ours` names hold it
    /// no longer. The resources go to `out`, for the caller to drop with no
    /// lock held.
    pub(crate) fn let_go_kept(
        &self,
        ours: &dyn Fn(&ID3D12Resource) -> bool,
        out: &mut Vec<ID3D12Resource>,
    ) {
        for shard in &self.kept {
            let mut kept = shard.lock().unwrap_or_else(|e| e.into_inner());
            for slot in kept.iter_mut() {
                if slot.as_ref().is_some_and(ours) {
                    out.extend(slot.take());
                }
            }
        }
    }

    /// `look` at what slot `index` views ([`HostHeap::kept`]); false past
    /// the heap.
    fn kept_at(&self, index: u32, look: impl FnOnce(Option<&ID3D12Resource>) -> bool) -> bool {
        let Some(shard) = self.kept.get((index / KEPT_SHARD) as usize) else {
            return false;
        };
        let kept = shard.lock().unwrap_or_else(|e| e.into_inner());
        kept.get((index % KEPT_SHARD) as usize)
            .is_some_and(|slot| look(slot.as_ref()))
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view that reads
    /// zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// `len` bytes of `buffer` from byte `offset` (a multiple of 16): a
    /// root UAV with a size, whose accesses past it do nothing instead of
    /// reaching whatever the host placed after the buffer.
    pub(crate) fn raw_uav_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device.CreateUnorderedAccessView(
                buffer,
                None::<&ID3D12Resource>,
                Some(&desc),
                self.cpu(index),
            )
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` (a dynamic one) a raw unordered access view of
    /// nothing: reads give zeros, writes are dropped.
    pub(crate) fn null_raw_uav_at(&self, index: u32) {
        self.keep(index, None);
        null_uav(&null_raw_uav_desc(), self.cpu(index));
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(size),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of `bytes`
    /// bytes (whole 256s, at most 64 KiB) of `buffer` from host GPU address
    /// `gpu` (on 256 bytes, the view inside the buffer): a root constant
    /// buffer view with a size, whose reads past it give zeros instead of
    /// reaching whatever the host placed after the buffer. The slot keeps
    /// the buffer.
    pub(crate) fn constant_view_at(
        &self,
        index: u32,
        buffer: &ID3D12Resource,
        gpu: u64,
        bytes: u32,
    ) {
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: constant_bytes(bytes),
        };
        // SAFETY: a slot of our heap; the range lies in the live buffer.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

    /// The keys last seen of this heap's slots ([`HostHeap::seen`]), made
    /// on first use.
    fn seen(&self) -> &[std::sync::atomic::AtomicU64] {
        self.seen.get_or_init(|| {
            (0..self.count)
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect()
        })
    }

    /// A call wrote the record of key `key` ([`record_key`], read before
    /// its mirror) into title slot `index` and mirrored it (`super::record`,
    /// `super::copy_simple`): the key is the one last seen, so the next pass
    /// leaves the slot if it still holds that record and the slot shows it
    /// ([`HostHeap::sort_copied_page`]). A mirror that finds no view in a
    /// slot notes what it read too (0, or a released view's dead key),
    /// whether or not it changes the slot ([`HostHeap::mirror_marking`]).
    /// Nothing before the first pass.
    pub(crate) fn saw_key(&self, index: u32, key: u64) {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(seen) = self.seen.get().and_then(|seen| seen.get(index as usize))
            // (Most mirrors of a draw find the key there already: the cache
            // line is left clean.)
            && seen.load(Relaxed) != key
        {
            seen.store(key, Relaxed);
        }
    }

    /// For one changed page of the title's heap (`super::sync_heaps`):
    /// title slots `slots`, the record of slot `i` at title address
    /// `start + i * step`, and `page` the bytes read from title address
    /// `from` on. A record that straddles the page's start is the range's
    /// before it. A record is left if its key (the 12 bytes that name a view,
    /// [`page_key`]) is the one last seen in its slot ([`HostHeap::seen`])
    /// and the slot is made from it or from nothing; a slot made from another
    /// record is mirrored again whatever was seen, so no write that a pass
    /// did not see (a call's, a draw's, a record copied back as bytes over
    /// one a call made) leaves it showing that other record. A mirror that
    /// finds no view in a slot notes the key it read as seen, so a live
    /// record copied back over bytes a mirror found no view in (a draw's
    /// that made the slot null, or a refused one that left it null) differs
    /// from the one last seen and is mirrored again. Of the others,
    /// one of a slot a table's lately is left to its draws, its key not taken
    /// as seen, so it is looked at again next time; the rest are seen from
    /// here on and go into `part.make`, in slot order, for the caller to
    /// mirror ([`HostHeap::mirror_synced`]) once every part of the heap is
    /// sorted. Nothing is mirrored here.
    ///
    /// A record that is no view of ours (key 0) into a slot made from none is
    /// left too: its mirror would leave the slot as it is (cool_fox measured
    /// 32,000 such records of 41,000 a frame in one place of the title). The
    /// compare reads 12 bytes a record and keeps 8 a slot, against the 32
    /// bytes a record of the copy of records it replaced (cool_fox 1f1a433,
    /// 53c5256), and takes no lock over the heap.
    pub(crate) fn sort_copied_page(
        &self,
        slots: std::ops::Range<u64>,
        (start, step): (u64, u64),
        from: usize,
        page: &[u8],
        part: &mut SyncPart,
    ) {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        let first = ((from as u64).saturating_sub(start))
            .div_ceil(step)
            .max(slots.start);
        let Some(records) = ((start + first * step) as usize)
            .checked_sub(from)
            .and_then(|at| page.get(at..))
        else {
            return;
        };
        let count = (records.len() as u64 / step).min(slots.end.saturating_sub(first));
        part.looked += count;
        let (seen, frame, tables) = (self.seen(), crate::frame::current(), sync_tables());
        for (n, record) in records
            .chunks_exact(step as usize)
            .take(count as usize)
            .enumerate()
        {
            let index = first as u32 + n as u32;
            let (Some(last), Some(made)) =
                (seen.get(index as usize), self.made.get(index as usize))
            else {
                break;
            };
            let key = page_key(record);
            if key == last.load(Relaxed) && {
                let shown = made.load(Acquire);
                shown == key || shown == 0
            } {
                continue;
            }
            // The 4 KiB pieces of title memory the records not left here
            // are in (each once: the records are in slot order), for the
            // frame budget.
            let piece = (start + u64::from(index) * step) >> 12;
            if part.pieces.last() != Some(&piece) {
                part.pieces.push(piece);
            }
            // A table's slot lately: left to its draws, and looked at again
            // next time.
            if !tables && self.tabled_lately(index, frame) {
                COPIED_TABLED.fetch_add(1, Relaxed);
                part.left.push(index);
                continue;
            }
            last.store(key, Relaxed);
            part.make.push(index);
        }
    }

    /// For one pass over this heap (`super::sync_heaps`), after its pages are
    /// sorted: the slots the pass left to tables (`left`, this pass's) are
    /// noted ([`HostHeap::left_to_tables`]), and those noted before that no
    /// table holds lately any more are read again from the title's heap at
    /// `start` (records of `step` bytes): their keys are seen from here on
    /// and they are handed on to be mirrored (`part.make`, kept sorted and
    /// without repeats). Costs one lock of the set (the pass's alone) and a
    /// look at each slot in it, then a read of 12 bytes per slot read again;
    /// nothing when the set is empty. No lock of the mirror path is taken.
    pub(crate) fn retake_left(&self, left: &[u32], (start, step): (u64, u64), part: &mut SyncPart) {
        let frame = crate::frame::current();
        let aged: Vec<u32> = {
            let mut set = self
                .left_to_tables
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if set.is_empty() && left.is_empty() {
                return;
            }
            let aged: Vec<u32> = set
                .iter()
                .copied()
                .filter(|&slot| !self.tabled_lately(slot, frame))
                .collect();
            for slot in &aged {
                set.remove(slot);
            }
            let (most, mut refused) = (left_most(), 0u64);
            for &slot in left {
                if set.contains(&slot) {
                    continue;
                }
                if set.len() >= most {
                    refused += 1;
                    continue;
                }
                set.insert(slot);
            }
            if refused > 0 {
                LEFT_DROPPED.fetch_add(refused, std::sync::atomic::Ordering::Relaxed);
            }
            aged
        };
        if aged.is_empty() {
            return;
        }
        let seen = self.seen();
        let mut record = [0u8; 12];
        for slot in aged {
            let Some(last) = seen.get(slot as usize) else {
                continue;
            };
            // (Read by the safe copy: the title may have unmapped the page.)
            let at = start + u64::from(slot) * step;
            if !crate::host::read_into(at as *const u8, &mut record) {
                continue;
            }
            last.store(page_key(&record), std::sync::atomic::Ordering::Relaxed);
            part.make.push(slot);
        }
        part.make.sort_unstable();
        part.make.dedup();
    }

    /// [`HostHeap::mirror`] of a slot `super::sync_heaps` found a new
    /// record in. Any thread may call it.
    pub(crate) fn mirror_synced(&self, index: u32, handle: u64) {
        self.mirror_marking(index, handle, false, true, hold::SYNC);
    }

    /// The key [`HostHeap::sort_copied_page`] last saw in slot `index`
    /// (tests; `None` before the first pass).
    #[cfg(test)]
    pub(crate) fn seen_key(&self, index: u32) -> Option<u64> {
        let seen = self.seen.get()?.get(index as usize)?;
        Some(seen.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since, and returns the view recorded there. Buffer
    /// views are made only for draws (`for_draw`), which also copy their
    /// data: a host copy of a buffer is made on first use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) -> Option<View> {
        let site = if for_draw { hold::TABLE } else { hold::VIEW };
        self.mirror_marking(index, handle, for_draw, true, site)
    }

    /// [`HostHeap::mirror`] from `site` ([`hold`]), for the profiler.
    pub(crate) fn mirror_at(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        site: u64,
    ) -> Option<View> {
        self.mirror_marking(index, handle, for_draw, true, site)
    }

    /// [`HostHeap::mirror`] for a draw, by the walk of the heap buffer
    /// pass, which takes the slot's copy and range from the view this
    /// returns at once: the slot is not marked changed for the next pass
    /// ([`HostHeap::buffer_changes`]) by what this makes or empties. That
    /// mark had every slot the walk made walked again by the next pass,
    /// only to find the same copy. A write after this one marks the slot
    /// itself: the title's writes and copies, and the draws and the
    /// texture pass that make a slot from a record no write told us of.
    /// `VICTORY_HEAP_WALK_ONCE=0`: marked, as before.
    pub(crate) fn mirror_walked(&self, index: u32, handle: u64) -> Option<View> {
        self.mirror_marking(index, handle, true, !walk_once(), hold::WALK)
    }

    /// The slots of this heap made on C-pool homes that a let-go since the
    /// last look lets go ([`pool_let_go`]; all of them if more went than
    /// were kept) count as made from no record: the next mirror of each (a
    /// draw that reads it through a table, the heap buffer pass, which
    /// walks them as changed) makes it anew from its record, on what its
    /// memory is now. Until then it shows what it showed. Called before a
    /// mirror and the heap buffer pass, with no claim and no lock of the
    /// heap's held; one atomic load when nothing went.
    ///
    /// shortcut: a slot made by a thread that found the home just before the
    /// let-go, and noted after this look, keeps the old pages until the title
    /// writes it again.
    pub(crate) fn pool_slots_let_go(&self) {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        let last = POOL_GONE_LAST.load(Acquire);
        let seen = self.pool_seen.load(Relaxed);
        // Another thread looking at the same let-goes does them for both.
        if last == seen
            || self
                .pool_seen
                .compare_exchange(seen, last, Relaxed, Relaxed)
                .is_err()
        {
            return;
        }
        let ranges: Option<Vec<(u64, u64)>> = {
            let gone = POOL_GONE.lock().unwrap_or_else(|e| e.into_inner());
            // Every let-go since `seen` is still kept.
            gone.front()
                .is_some_and(|&(n, _, _)| n <= seen + 1)
                .then(|| {
                    gone.iter()
                        .filter(|&&(n, _, _)| n > seen && n <= last)
                        .map(|&(_, start, end)| (start, end))
                        .collect()
                })
        };
        let slots: Vec<u32> = {
            let mut pool = self.pool_slots.lock().unwrap_or_else(|e| e.into_inner());
            let slots = match ranges {
                Some(ranges) => ranges
                    .iter()
                    .flat_map(|&(start, end)| pool.take_over(start, end))
                    .collect(),
                None => pool.take_all(),
            };
            self.pool_noted.store(pool.by_slot.len(), Relaxed);
            slots
        };
        if slots.is_empty() {
            return;
        }
        for &slot in &slots {
            if let Ok(mut claim) = self.claim(slot, |made| made != 0) {
                claim.value = 0;
            }
        }
        crate::pool::slots_made_anew(slots.len() as u64);
        if let Some(bits) = &self.bits {
            for slot in slots {
                bits.changed.put(slot);
            }
            return;
        }
        self.changed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(slots);
    }

    /// Slot `index` was just made from `view`, showing `kept`: noted among
    /// the slots made on C-pool homes if it is one ([`PoolSlots`]), and
    /// taken out of them otherwise.
    fn note_pool_slot(&self, index: u32, view: Option<&View>, kept: Option<&ID3D12Resource>) {
        use std::sync::atomic::Ordering::Relaxed;
        let span = view.and_then(|view| pool_slot_span(view, kept));
        // No lock while the heap has no such slot (the pool off, mostly).
        if span.is_none() && self.pool_noted.load(Relaxed) == 0 {
            return;
        }
        let mut pool = self.pool_slots.lock().unwrap_or_else(|e| e.into_inner());
        pool.set(index, span);
        self.pool_noted.store(pool.by_slot.len(), Relaxed);
    }

    /// [`HostHeap::mirror`]; `mark`: a slot that is or was a buffer view
    /// goes into the set the next heap buffer pass walks. `site`: who asks
    /// ([`hold`]), for the profiler.
    fn mirror_marking(
        &self,
        index: u32,
        handle: u64,
        for_draw: bool,
        mark: bool,
        site: u64,
    ) -> Option<View> {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            crate::census::PAST_HEAP[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            crate::census::PAST_HEAP[1].fetch_max(u64::from(index), std::sync::atomic::Ordering::Relaxed);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return None;
        }
        // Slots on C-pool homes let go since: made anew below if this is one.
        self.pool_slots_let_go();
        // SAFETY: the handle is a descriptor slot in title memory.
        let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        // Generations start at 1, so a made slot's key is never 0.
        let key = u64::from(generation) << 32 | u64::from(i);
        // For every mirror ([`SKIPPED_MADE`]: table binds, the walk of the
        // heap buffer pass, the texture pass, the title's writes,
        // `sync_heaps`): a slot made from this record already, of a view
        // still there, is done with before the claim, as the claim below
        // would find it (a plain load: `CLAIMED` is never a key, and a change
        // just after it is the same race as one just after the claim's look).
        // A record whose resource was released has no view and goes on to
        // the claim, which makes its slot null.
        //
        // `made` itself is what is asked: it holds a key only once the
        // slot's descriptor, word and kept resource are in place (the
        // claim stores it last, with release), and every change of the
        // slot goes through a claim, so no second record of it can say a
        // key the slot no longer shows. (cool_fox keeps such a second
        // record beside its `made`, and had to forget it where a slot
        // flips to its late view, e4d0837. Our one second record,
        // [`HostHeap::left`], says what a slot made from nothing was left
        // for a draw from, and is written by the claim itself.)
        if tag == VIEW_TAG
            && self
                .made
                .get(index as usize)
                .is_some_and(|made| made.load(std::sync::atomic::Ordering::Acquire) == key)
            && let Some(view) = view_at(i, generation)
        {
            SKIPPED_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(view);
        }
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            let _hold = self.hold(site);
            #[cfg(test)]
            if let Some(hook) = BEFORE_NULL.take() {
                hook(self);
            }
            // What was read is noted as seen (0 for bytes of no view, the
            // dead key for a released view), so that a live record copied
            // back over it later does not look like the one the pass saw last
            // ([`HostHeap::sort_copied_page`]).
            let read = if tag == VIEW_TAG { key } else { 0 };
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            let Ok(mut claim) = self.claim(index, |made| made != 0 || for_draw) else {
                // Refused: the slot shows nothing and this is no draw's
                // mirror, so it stays as it is. The pass may have noted a live
                // record it found here a moment ago, which these bytes
                // replaced; noted as is, that record copied back would be
                // left in a null slot.
                self.saw_key(index, read);
                return None;
            };
            // Noted under the claim when it changes the slot: a pass that
            // made the slot from a live record between this read and the
            // claim noted that record as seen, and the null made here must
            // not leave it so.
            self.saw_key(index, read);
            if claim.old != 0 {
                let (kind, _) = mirror_kind(None, self.sampler, false, false);
                crate::frame_budget::mirrored(kind, false);
                claim.value = 0;
                self.write_word(index, 0);
                self.make_null(index);
                self.note_pool_slot(index, None, None);
                let _ = self.note_buffer(index, None, false, mark);
            } else {
                // Left for later and stale since: out of the sets and the stride table.
                self.write_word(index, 0);
                let _ = self.note_buffer(index, None, false, mark);
            }
            return None;
        };
        // What the slot's host descriptor is made from is found, or made,
        // before the slot is claimed ([`resolve`]): a host copy of a buffer
        // or a host texture made under the claim kept every thread that
        // asked for the slot meanwhile waiting for it. Not for a slot that
        // shows this record already (the most common case by far, a draw's
        // table entry: the claim below lets it be), nor for a view left
        // for later. Dropped after the claim, if unused: nothing it holds
        // is let go under the claim.
        let mut resolved: Option<Option<Resolved>> = None;
        #[cfg(test)]
        if let (Some(hook), Some(made)) = (BEFORE_LOOK.take(), self.made.get(index as usize)) {
            hook(made);
        }
        let mut refused_placed = None;
        let mut again = false;
        let mut levels = 0;
        let (mut claim, waiting, later) = loop {
            let (_, later) = Self::deferred(&view, for_draw);
            let shown = || {
                self.made
                    .get(index as usize)
                    .is_some_and(|made| made.load(std::sync::atomic::Ordering::Acquire) == key)
            };
            if !later && resolved.is_none() && (again || !shown()) {
                levels = LEVELS_SEEN.load(std::sync::atomic::Ordering::SeqCst);
                // Summed for the walk of the heap buffer pass (`make_us`).
                let making = crate::profile::part_start();
                let found = resolve(&view);
                crate::profile::part_end(making);
                // (What `refused_for_now` asks of a refused view.)
                if found.is_none() {
                    refused_placed = placed_texture(&view);
                }
                resolved = Some(found);
            }
            #[cfg(test)]
            if let (Some(hook), Some(made)) = (BEFORE_CLAIM.take(), self.made.get(index as usize)) {
                hook(made);
            }
            // The slot shows this record already: nothing to change.
            let Ok(claim) = self.claim(index, |made| made != key) else {
                if matches!(resolved, Some(Some(_))) {
                    RESOLVED_UNUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                return Some(view);
            };
            // Asked again under the claim, as before: a texture may have
            // got its host copy meanwhile.
            let (waiting, later) = Self::deferred(&view, for_draw);
            // A texture view resolved before a change of its texture's
            // levels ([`LEVELS_SEEN`]) shows the levels from before it.
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            let old_levels = matches!(
                resolved,
                Some(Some(
                    Resolved::Srv { shadow: None, .. } | Resolved::Uav { shadow: None, .. }
                ))
            ) && LEVELS_SEEN.load(std::sync::atomic::Ordering::SeqCst) != levels;
            if later || (resolved.is_some() && !old_levels) {
                break (claim, waiting, later);
            }
            // Claimed with nothing found for it (the slot changed after the
            // look above, or the texture got its host copy), or with a
            // texture view of levels changed since: let go, nothing
            // changed, and found first (again) after all.
            drop(claim);
            RESOLVED_AGAIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            resolved = None;
            again = true;
        };
        if later && matches!(resolved, Some(Some(_))) {
            RESOLVED_UNUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        let _hold = self.hold(site);
        // On the profile: slots changed against the distinct records they
        // changed to, by who mirrored them.
        crate::profile::record_changed(site, key);
        let kind = {
            let entry = generation >> super::KIND_SHIFT == 0;
            let (kind, eligible) = mirror_kind(Some(&view), self.sampler, later, entry);
            crate::frame_budget::mirrored(kind, eligible);
            kind
        };
        // Nothing to shaders while the slot changes (`write_word`).
        self.write_word(index, 0);
        // Views of other textures in the memory this one describes are
        // stale: their host slots read nothing from here on, and let go of
        // the host copies they held, once this slot is made
        // (`unmake_stale`).
        #[cfg(test)]
        if let Some(hook) = BEFORE_NOTE.take() {
            hook();
        }
        let stale: Vec<(u32, u32)> = self
            .note_buffer(index, Some(&view), for_draw, mark)
            .into_iter()
            .filter(|&(stale, _)| stale != index)
            .collect();
        // A level change after the look under the claim and before the slot
        // was in the spans: the remade pass did not find the slot there. The
        // view is written as resolved, and its start told again once the
        // claim is let go, for the next pass, which finds the slot.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        let reshow = match &view {
            View::Placed(false, _, desc)
                if matches!(
                    resolved,
                    Some(Some(
                        Resolved::Srv { shadow: None, .. } | Resolved::Uav { shadow: None, .. }
                    ))
                ) && LEVELS_SEEN.load(std::sync::atomic::Ordering::SeqCst) != levels =>
            {
                Some(u64::from(desc[10]) | u64::from(desc[11]) << 32)
            }
            _ => None,
        };
        // Another view of the placed texture the slot shows already (a
        // title that streams a texture in writes its view again with more
        // mips each time some arrive): the slot goes on showing the old
        // view until a draw or the texture pass makes the new one and
        // uploads what it shows. Made null here, the texture read as
        // nothing for as long as it waited in the queue, every time it got
        // sharper.
        // The texture the slot waits for (`super::wait_for`, after the
        // claim: it takes a global lock and may make the texture).
        let mut wait = waiting;
        if later
            && keep_views()
            && view_at(claim.old as u32, (claim.old >> 32) as u32)
                .is_some_and(|old| same_placed_texture(&old, &view))
        {
            KEPT_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // The old view's word too: both are of one placed texture.
            self.write_word(index, slot_word(Some(&view)));
        } else {
            let mut value = if later { 0 } else { key };
            let kept = if later {
                None
            } else {
                // Summed for the walk of the heap buffer pass (`make_us`).
                let making = crate::profile::part_start();
                let kept = resolved
                    .as_ref()
                    .and_then(Option::as_ref)
                    .and_then(|resolved| write(resolved, self.cpu(index)));
                crate::profile::part_end(making);
                #[cfg(test)]
                if kept.is_some() {
                    self.writes
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                kept
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
                if !later {
                    wait = wait.or(self.refused_for_now(&mut value, index, &view, refused_placed));
                }
            }
            let kept = kept.flatten();
            self.note_pool_slot(index, Some(&view), kept.as_ref());
            self.keep(index, kept);
            // What the slot holds now, made or a null view of its kind.
            self.write_word(index, slot_word(Some(&view)));
            // Shown to other threads only now, with its descriptor and its
            // word in place.
            claim.value = value;
            // A buffer view left for a draw: the record the slot is left
            // from ([`HostHeap::left`]). Not a placed texture's or a waiting
            // texture's view, whose deferral also changes the queue of
            // textures, the spans and the texture's waits.
            if later
                && mark
                && waiting.is_none()
                && kind == crate::frame_budget::Mirror::BufferDeferred
                && claim.left_at.is_some()
            {
                if claim.old == 0
                    && let Some(again) = left_again(claim.left_before(), key)
                {
                    crate::frame_budget::left_again(again);
                }
                claim.left = key;
            }
        }
        drop(claim);
        if let Some(start) = reshow {
            RESHOWN_AGAIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // No level changed again: not counted in `LEVELS_SEEN`, which
            // would make every mirror of a texture view meanwhile resolve
            // again or tell its start too.
            RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
        }
        if let Some(texture) = wait {
            super::wait_for(texture, handle);
        }
        self.unmake_stale(&stale);
        Some(view)
    }

    /// Whether a mirror of `view` (`for_draw`: a draw's) leaves its slot for
    /// later: a buffer view not for a draw, or a view of a texture object
    /// with no host copy yet (that texture, which the slot then waits for).
    /// A view of a texture object that has no host copy yet waits for it,
    /// as buffer views wait for a draw: the slot is made when the texture
    /// is first used (`super::make_waiting`), or when a draw reads the slot
    /// through a table.
    fn deferred(view: &View, for_draw: bool) -> (Option<usize>, bool) {
        let waiting = if for_draw {
            None
        } else {
            view.waiting_texture()
        };
        (
            waiting,
            !for_draw && (view.is_buffer() || waiting.is_some()),
        )
    }

    /// Claims slot `index` of `made` for this thread, if `wanted` says so
    /// of what it holds: until the claim is dropped, the slot holds
    /// [`CLAIMED`] and no other thread looks at or changes the host slot,
    /// its word or what it keeps (it waits). Dropped, the slot holds the
    /// claim's `value`, which starts as what it held. `Err`: what it holds,
    /// not wanted.
    ///
    /// A thread waits for a claim only while it holds none itself, so
    /// claims cannot wait for each other in a ring.
    fn claim(&self, index: u32, wanted: impl Fn(u64) -> bool) -> Result<Claimed<'_>, u64> {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        let Some(slot) = self.made.get(index as usize) else {
            return Err(0);
        };
        let mut waited = 0;
        let mut spins = 0u32;
        let claimed = loop {
            let now = slot.load(Acquire);
            if now == CLAIMED {
                if spins == 0 {
                    waited = crate::profile::stamp();
                }
                spins = spins.saturating_add(1);
                if spins < 64 {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
                continue;
            }
            if !wanted(now) {
                break Err(now);
            }
            if slot
                .compare_exchange_weak(now, CLAIMED, Acquire, Relaxed)
                .is_ok()
            {
                if cfg!(debug_assertions) {
                    CLAIMS_HELD.with(|held| held.set(held.get() + 1));
                }
                break Ok(Claimed {
                    slot,
                    left_at: self.left.get(index as usize),
                    old: now,
                    value: now,
                    left: 0,
                });
            }
        };
        if spins > 0 {
            crate::profile::since("made claim wait", waited, 50, &[("heap", self.id)]);
        }
        claimed
    }

    /// Whether slot `index` is backed by a made host descriptor (tests).
    #[cfg(test)]
    pub(crate) fn made(&self, index: u32) -> bool {
        self.claim(index, |_| false).is_err_and(|key| key != 0)
    }

    /// The record slot `index` was last left for a draw from (tests;
    /// [`HostHeap::left`]), once no thread changes it.
    #[cfg(test)]
    pub(crate) fn left_key(&self, index: u32) -> u64 {
        let Ok(mut claim) = self.claim(index, |_| true) else {
            unreachable!("every slot is wanted");
        };
        claim.left = claim.left_before();
        claim.left
    }

    /// The host resource slot `index` keeps, as its raw pointer (tests).
    #[cfg(test)]
    pub(crate) fn kept_raw(&self, index: u32) -> Option<*mut std::ffi::c_void> {
        let mut raw = None;
        self.kept_at(index, |kept| {
            raw = kept.map(windows::core::Interface::as_raw);
            true
        });
        raw
    }

    /// Host slot `index` was just made null from `view`. If its texture is
    /// one the host refused for lack of memory and tries again
    /// (`crate::resource::host_waits`), the slot counts as not made, so
    /// that a look after the texture is made makes it: the texture's own
    /// making (`super::make_waiting`, for a view of a texture object), a
    /// draw that reads it through a table, or the pass for textures picked
    /// by index, which looks at it again in `RETRY_FRAMES` frames (the
    /// pass takes the slot out of its queue on the failure). Counted as
    /// made, it read zeros for the rest of the run.
    /// `made`: what the claimed slot will hold ([`Claimed::value`]);
    /// `placed`: the view's [`placed_texture`], looked up before the claim.
    /// Returns the texture object the slot is to wait for, which the
    /// caller tells once it holds the claim no more (`super::wait_for`).
    fn refused_for_now(
        &self,
        made: &mut u64,
        index: u32,
        view: &View,
        placed: Option<View>,
    ) -> Option<usize> {
        /// Frames until the pass looks at such a slot again: the try
        /// itself waits a second or more (`crate::resource::refused_after`).
        const RETRY_FRAMES: u64 = 16;
        let texture = placed.as_ref().unwrap_or(view).waiting_texture()?;
        *made = 0;
        crate::census::VIEW_NULL_WAITING[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // (`make_waiting` makes only slots that view the texture object.)
        let wait = placed.is_none().then_some(texture);
        let frame = crate::frame::current();
        if self.tabled_lately(index, frame) {
            return wait;
        }
        let Some(spanned) = self.spanned.get(index as usize) else {
            let mut textures = self.texture_queue();
            textures.changed(index, frame, frame + RETRY_FRAMES);
            return wait;
        };
        // A slot at a time: logged after the write's own change, and under
        // the spans' lock, so that a removal from a write that made the
        // slot stale since comes after it; or left out, if it came before
        // (the slot's write in the spans, and no longer there).
        let note = TextureNote::Changed {
            index,
            frame,
            due: frame + RETRY_FRAMES,
        };
        let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
        let in_spans = spanned.load(std::sync::atomic::Ordering::Relaxed) & IN_SPANS != 0;
        if !in_spans || placed.by_slot.contains_key(&index) {
            self.note_textures(&[note]);
        }
        wait
    }

    /// Whether host slot `index` shows `buffer`; `None` if it shows no
    /// resource (diagnostics: a buffer view's slot must show the host copy
    /// that is brought up to date for it).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn shows(&self, index: u32, buffer: &ID3D12Resource) -> Option<bool> {
        let mut shows = None;
        self.kept_at(index, |shown| {
            shows = shown.map(|shown| {
                windows::core::Interface::as_raw(shown) == windows::core::Interface::as_raw(buffer)
            });
            true
        });
        shows
    }

    /// What the queue of textures picked by index knows of title slot
    /// `index` at `frame`, in a few words (diagnostics).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn texture_slot_state(&self, index: u32, frame: u64) -> String {
        let textures = self.texture_queue();
        match textures.slots.get(&index) {
            None => "not queued".into(),
            Some(w) => format!(
                "queued due={} written@{} looks={} part_mapped={} recheck={}",
                if w.due <= frame { "now".to_string() } else { format!("@{}", w.due) },
                w.written,
                w.looks,
                w.mapped,
                w.checked.2 != 0
            ),
        }
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        if index >= self.count {
            return None;
        }
        let key = self.claim(index, |_| false).err()?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at(key as u32, (key >> 32) as u32)?;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

// A child module beside this file, so it shares this module's items and
// `once!`.
#[path = "buffer_views.rs"]
mod buffer_views;
pub(crate) use buffer_views::{buffer_copy, buffer_extent, buffer_range};
use buffer_views::{buffer_srv, buffer_uav};
#[cfg(test)]
use buffer_views::{buffer_view_ok, elements_inside, own_bytes_if, raw_window};

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Whether the pass over records copied in as bytes also mirrors slots
/// that draws read through tables lately (`VICTORY_HEAP_SYNC_TABLES=1`;
/// off by default: [`HostHeap::sort_copied_page`]).
fn sync_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_TABLES").is_some_and(|v| v == "1"))
}

/// The key of a record as `made` and [`HostHeap::seen`] hold it, from its
/// first 12 bytes (`record`: at least those): the view index and generation
/// of a record of ours, 0 for anything else.
fn page_key(record: &[u8]) -> u64 {
    let word = |at: usize| {
        record
            .get(at..at + 4)
            .and_then(|b| b.try_into().ok())
            .map_or(0, u32::from_le_bytes)
    };
    if word(0) == VIEW_TAG {
        u64::from(word(8)) << 32 | u64::from(word(4))
    } else {
        0
    }
}

/// [`page_key`] of the record the title has at `handle`.
pub(crate) fn record_key(handle: u64) -> u64 {
    // SAFETY: the handle is a descriptor slot in title memory.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag == VIEW_TAG {
        u64::from(generation) << 32 | u64::from(i)
    } else {
        0
    }
}

/// Changed pages of a heap the heap sync gives a job at a time.
pub(crate) const SYNC_PAGES: usize = 8;

/// Slots to mirror the heap sync gives a job at a time.
pub(crate) const SYNC_SLOTS: usize = 256;

/// What a job of the heap sync found on its pages of a heap.
#[derive(Default)]
pub(crate) struct SyncPart {
    /// Pages read.
    pub(crate) pages: u64,
    /// Records on them looked at.
    pub(crate) looked: u64,
    /// Profile ticks reading the pages.
    pub(crate) read_ticks: u64,
    /// The slots to mirror, in slot order.
    pub(crate) make: Vec<u32>,
    /// The changed ones left to the draws of tables, for
    /// [`HostHeap::retake_left`].
    pub(crate) left: Vec<u32>,
    /// Records looked at on pages our own writes had changed, and the
    /// 4 KiB pieces of title memory with records the key compare did not
    /// leave (their numbers, each once per page): for the frame budget.
    pub(crate) own_looked: u64,
    pub(crate) pieces: Vec<u64>,
}

/// Mirrors done with before their slot's claim as the slot was made from
/// the record already, of a view still there (a plain look; a record of a
/// released resource still goes on and makes its slot null): table binds,
/// the walk of the heap buffer pass, the texture pass, the title's writes
/// and `sync_heaps`. The same slots are made as with the claim. Only grows.
///
/// Ported from cool_fox 53c5256 and the unlocked first look of 6c17ce9:
/// there most asks of five recording threads were of slots made already,
/// behind one lock another thread held to make a descriptor (cool_fox
/// measured 15 % of a recording thread's time in the title's opening).
pub(crate) static SKIPPED_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Views resolved before their slot's claim ([`resolve`]) that were not
/// written: another thread made the slot from the record meanwhile, or the
/// view was left for later. Only grows.
pub(crate) static RESOLVED_UNUSED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Claims of a slot let go to resolve its view first: the slot changed
/// after the look before the claim, its texture got its host copy in
/// between, or its texture's levels changed after the view was resolved
/// ([`LEVELS_SEEN`]). Only grows.
pub(crate) static RESOLVED_AGAIN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Placed texture views written from levels that changed while their
/// slot entered the spans: their start was told again ([`LEVELS_SEEN`]).
/// Only grows.
pub(crate) static RESHOWN_AGAIN: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// With the frame line: views resolved before their slot's claim and not
/// written ([`RESOLVED_UNUSED`]) and claims let go to resolve again
/// ([`RESOLVED_AGAIN`]), per frame over `frames`, since the last line;
/// and the starts told again ([`RESHOWN_AGAIN`]) since then.
/// Nothing while there are none.
pub(crate) fn claims_line(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LINED: [std::sync::atomic::AtomicU64; 3] =
        [const { std::sync::atomic::AtomicU64::new(0) }; 3];
    let [unused, again, reshown] =
        [&RESOLVED_UNUSED, &RESOLVED_AGAIN, &RESHOWN_AGAIN].map(|n| n.load(Relaxed));
    let unused = unused.wrapping_sub(LINED[0].swap(unused, Relaxed));
    let again = again.wrapping_sub(LINED[1].swap(again, Relaxed));
    let reshown = reshown.wrapping_sub(LINED[2].swap(reshown, Relaxed));
    if unused == 0 && again == 0 && reshown == 0 {
        return;
    }
    tracing::info!(
        "view heap slots: per frame {:.1} views resolved before the slot's claim and not written \
         (made by another thread meanwhile, or left for later), {:.2} claims let go to resolve \
         the view first; {reshown} placed texture views told again to the remade pass",
        unused as f64 / frames,
        again as f64 / frames
    );
}

/// Slots of a heap [`HostHeap::left_to_tables`] holds at most: a view
/// heap's tables of a frame (some 16,000 tables of a few slots, run 45)
/// fit many times over. Slots left past it are not read again unless their
/// page is written (as before the set), and are counted in
/// [`LEFT_DROPPED`]. A slot is a 4-byte key in a tree node holding up to 11
/// of them: well under 1 MB at most per heap.
const LEFT_MOST: usize = 1 << 16;

/// [`LEFT_MOST`], or what a test on this thread set ([`LEFT_MOST_HERE`]).
fn left_most() -> usize {
    #[cfg(test)]
    if let Some(most) = LEFT_MOST_HERE.get() {
        return most;
    }
    LEFT_MOST
}

#[cfg(test)]
thread_local! {
    /// Tests: [`left_most`] on this thread.
    pub(crate) static LEFT_MOST_HERE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

/// Slots left to tables that [`HostHeap::left_to_tables`] had no room for.
/// Only grows.
pub(crate) static LEFT_DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Since the last frame line: slots on changed heap pages that
/// [`HostHeap::sort_copied_page`] left to their draws (tables' slots).
pub(crate) static COPIED_TABLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        if unordered {
            crate::resource::DEPTH_UNORDERED[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A view of a target whose format the host refuses, in the title's
    // format: in the stand-in the host copy has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// The title's unordered access view of a depth texture, as a view of the
/// texture's colour twin `twin` ([`crate::resource::Twin`]): the same mip
/// and slices, in the colour format of the twin's family for a depth or
/// typeless format (R16_UNORM for 16-bit depth, R32_FLOAT for 32-bit; the
/// bits are the same). Without a description: mip 0 of every slice.
/// The twin of a stencil plane (R8) stands for the views of that plane
/// only, which become R8_UINT views of its one plane; `None` for a view of
/// the depth plane, which the twin does not stand for. The depth plane has
/// a twin of its own, viewed as a depth texture's without stencil: a view
/// in the format of the depth plane of a depth-stencil format (21, 46)
/// takes the twin's colour format too.
fn twin_uav(
    twin: &ID3D12Resource,
    desc: Option<&D3D12_UNORDERED_ACCESS_VIEW_DESC>,
) -> Option<D3D12_UNORDERED_ACCESS_VIEW_DESC> {
    // SAFETY: a live resource.
    let d = unsafe { twin.GetDesc() };
    let mut desc = desc.cloned().unwrap_or_else(|| {
        let mut all = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        if d.DepthOrArraySize > 1 {
            all.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            all.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: u32::from(d.DepthOrArraySize),
                ..Default::default()
            };
        }
        all
    });
    // A stencil plane's twin (R8 typeless, 60): the title's view is of the
    // stencil plane if it names plane 1 or a stencil view format (22, 47).
    if d.Format.0 == 60 {
        let mut plane = 0;
        match desc.ViewDimension {
            D3D12_UAV_DIMENSION_TEXTURE2D => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2D };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2D = view;
            }
            D3D12_UAV_DIMENSION_TEXTURE2DARRAY => {
                // SAFETY: the member the dimension names.
                let mut view = unsafe { desc.Anonymous.Texture2DArray };
                plane = std::mem::take(&mut view.PlaneSlice);
                desc.Anonymous.Texture2DArray = view;
            }
            _ => {}
        }
        if plane != 1 && !matches!(desc.Format.0, 22 | 47) {
            return None;
        }
        desc.Format.0 = 62;
        return Some(desc);
    }
    // None, typeless and depth formats (R16: 0, 53, 55; R32: 0, 39, 40),
    // and those of a depth-stencil format's depth plane (19 to 21, 44 to
    // 46), for the twin of such a plane.
    desc.Format.0 = match (desc.Format.0, d.Format.0) {
        (0 | 53 | 55 | 19..=21 | 44..=46, 53) => 56,
        (0 | 39 | 40 | 19..=21 | 44..=46, 39) => 41,
        (other, _) => other,
    };
    Some(desc)
}

/// The bytes of the host's constant buffer view for a title's of `size`:
/// whole 256s, and at most the 64 KiB a shader can index. The host removes
/// the device for a larger view, and the title's heap holds such sizes in
/// slots it does not use.
fn constant_bytes(size: u32) -> u32 {
    super::constant_staging_bytes(size).next_multiple_of(256)
}

/// What the host descriptor of a view is made from, found (and its host
/// resource or copy made, if it had none) with no claim of a slot held
/// ([`resolve`]); written into a slot under its claim ([`write`]). It holds
/// the host resource and the copy a buffer view reads, so neither is let
/// go in between (`crate::shadow`'s sweep and `crate::resource::let_go_idle`
/// let go only what nothing else holds).
enum Resolved {
    Constant {
        desc: D3D12_CONSTANT_BUFFER_VIEW_DESC,
        shadow: std::sync::Arc<crate::shadow::Shadow>,
        address: u64,
    },
    /// A shader resource view of `resource`; `shadow`, of a buffer view:
    /// the copy it views and the title address it was asked for.
    Srv {
        resource: ID3D12Resource,
        desc: Option<D3D12_SHADER_RESOURCE_VIEW_DESC>,
        shadow: Option<(std::sync::Arc<crate::shadow::Shadow>, u64)>,
    },
    /// An unordered access view, as [`Resolved::Srv`].
    Uav {
        resource: ID3D12Resource,
        desc: Option<D3D12_UNORDERED_ACCESS_VIEW_DESC>,
        shadow: Option<(std::sync::Arc<crate::shadow::Shadow>, u64)>,
    },
    Sampler(D3D12_SAMPLER_DESC),
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot. Where no slot is claimed (dynamic slots,
/// tests); a claimed slot is written from what [`resolve`] found before.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    write(&resolve(view)?, at)
}

/// Writes the host descriptor `resolved` says at `at`: the host resource it
/// views (none for a sampler). Makes nothing else and takes no lock
/// (`None`: no host).
fn write(resolved: &Resolved, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let device = &host()?.device;
    let (kept, shadow) = match resolved {
        Resolved::Constant {
            desc,
            shadow,
            address,
        } => {
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(desc), at) };
            (Some(shadow.buffer.clone()), Some((shadow, *address)))
        }
        Resolved::Srv {
            resource,
            desc,
            shadow,
        } => {
            // SAFETY: a live resource and a slot of our heap.
            unsafe {
                device.CreateShaderResourceView(resource, desc.as_ref().map(std::ptr::from_ref), at)
            };
            let shadow = shadow.as_ref().map(|(shadow, address)| (shadow, *address));
            (Some(resource.clone()), shadow)
        }
        Resolved::Uav {
            resource,
            desc,
            shadow,
        } => {
            // SAFETY: a live resource and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    resource,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            let shadow = shadow.as_ref().map(|(shadow, address)| (shadow, *address));
            (Some(resource.clone()), shadow)
        }
        Resolved::Sampler(desc) => {
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(desc, at) };
            (None, None)
        }
    };
    if let Some((shadow, address)) = shadow {
        shadow.handed(crate::shadow::Hand::View, address);
    }
    Some(kept)
}

/// What the host descriptor for `view` is made from ([`Resolved`]); the
/// host resource it views made now if this is its first use. `None` if the
/// host cannot make one. Never under a claim of a slot ([`claims_held`]):
/// what it makes may take long (a host copy of a buffer, a texture) and
/// takes global locks, and every thread waiting for the slot would wait
/// for it.
fn resolve(view: &View) -> Option<Resolved> {
    debug_assert_eq!(claims_held(), 0, "a view resolved under a claim of a slot");
    host()?;
    match view {
        &View::Constant(address, size) => {
            if address == 0 || size == 0 {
                return None;
            }
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: constant_bytes(size),
            };
            Some(Resolved::Constant {
                desc,
                shadow,
                address,
            })
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                // The title's views of textures with a stencil plane, as it
                // describes them (format, dimension, component mapping, a
                // pad, then first level, levels, plane): each distinct one
                // once. They say how its shaders read the stencil plane
                // (the lighting takes each pixel's material class and its
                // "inside a room" bit from it).
                #[cfg(feature = "diagnostics")]
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() }.Format.0;
                    if matches!(own, 19..=22 | 44..=47) {
                        static SEEN: Mutex<Vec<(i32, Option<[u32; 10]>)>> = Mutex::new(Vec::new());
                        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
                        if seen.len() < 64 && !seen.contains(&(own, *desc)) {
                            seen.push((own, *desc));
                            tracing::info!(
                                ?resource,
                                texture_format = own,
                                view = ?desc,
                                "shader view of a depth-stencil texture"
                            );
                        }
                    }
                }
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // For the black census: a view of a 2D texture whose last
                // partial upload found its first levels without pages, and
                // the ones that start in them (nothing clamps those yet).
                // One load and two adds, no lock.
                let held = crate::resource::held_from(resource);
                if held > 0 {
                    let first = match &desc {
                        None => Some(0),
                        // SAFETY: the Texture2D member of a Texture2D view.
                        Some(d) if d.ViewDimension == D3D12_SRV_DIMENSION_TEXTURE2D => {
                            Some(unsafe { d.Anonymous.Texture2D.MostDetailedMip })
                        }
                        Some(_) => None,
                    };
                    if let Some(first) = first {
                        crate::census::VIEWS_BELOW_HELD[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if first < held {
                            crate::census::VIEWS_BELOW_HELD[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                // Levels on demand (`crate::tiles`): a texture made level by
                // level is shown from the first level that is the title's
                // and has memory, if the title's view starts before it (a
                // level without memory reads zeros: a black texture). The
                // title's own view mostly starts there already.
                if let Some(first) = crate::resource::first_shown(resource)
                    && first > 0
                {
                    // SAFETY: a live resource.
                    let own = unsafe { texture.GetDesc() };
                    // The texture's own view (the title gave none): all of it.
                    let whole = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: own.Format,
                        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                        Shader4ComponentMapping: 5768,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_SRV {
                                MostDetailedMip: 0,
                                MipLevels: u32::MAX,
                                PlaneSlice: 0,
                                ResourceMinLODClamp: 0.0,
                            },
                        },
                    };
                    crate::tiles::show_from(
                        desc.get_or_insert(whole),
                        first,
                        u32::from(own.MipLevels),
                    );
                }
                Some(Resolved::Srv {
                    resource: texture.clone(),
                    desc,
                    shadow: None,
                })
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc))
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => buffer_uav(data as u64, len, uav_desc(desc)),
                    _ => None,
                };
            };
            // The first such views of depth textures are logged as the
            // title describes them (format, dimension, then mip, slices and
            // plane): they say which planes shaders write, and as what.
            if crate::resource::depth_and_unordered(resource) {
                static LOGGED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 512 {
                    tracing::info!(
                        ?resource,
                        view = ?desc,
                        "unordered access view of a depth texture"
                    );
                }
            }
            let mut desc = desc.as_ref().map(uav_desc);
            // A depth texture's unordered access is its colour twin's: the
            // host gives a depth texture none.
            let twin = crate::resource::twin_resource(resource)
                .and_then(|twin| Some((twin, twin_uav(twin, desc.as_ref())?)))
                // A view of the depth plane of a texture with a stencil
                // plane: of that plane's own twin.
                .or_else(|| {
                    let twin = crate::resource::twin_plane_resource(resource)?;
                    Some((twin, twin_uav(twin, desc.as_ref())?))
                });
            let texture = match twin {
                Some((twin, view)) => {
                    desc = Some(view);
                    twin
                }
                None => texture,
            };
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            Some(Resolved::Uav {
                resource: texture.clone(),
                desc,
                shadow: None,
            })
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            Some(Resolved::Sampler(desc))
        }
        View::ConsoleSampler(words) => Some(Resolved::Sampler(crate::sampler::console(words))),
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return resolve(&view);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(
                    dimension = resource[0],
                    "placed unordered texture view not mirrored"
                );
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            // An empty title buffer: nothing to view, and no host copy to
            // look for (#70: such views were in flight at device removals).
            if len == 0 {
                return None;
            }
            if *unordered {
                buffer_uav(address, len, uav_desc(words))
            } else {
                buffer_srv(address, len, srv_desc(words))
            }
        }
    }
}

/// What a slot of [`HostHeap::made`] holds while a thread changes it
/// ([`HostHeap::claim`]). No record's key: its two kind bits say 3, a kind
/// no record has (`super::KIND_SHIFT`), and a slot is made only from a
/// record that names a view.
const CLAIMED: u64 = u64::MAX;

/// Slots of [`HostHeap::kept`] under one lock.
const KEPT_SHARD: u32 = 1 << 12;

/// The bit of [`HostHeap::spanned`] that says the slot's last write left
/// it in the spans; the bits above it count the slot's writes.
const IN_SPANS: u32 = 1;

/// Whether a heap's sets of buffer views (the slots holding one, and those
/// changed since the heap buffer pass took them) are bits of atomic words
/// ([`BufferBits`]), changed by the writer of a slot with no lock and no
/// log (`VICTORY_SLOT_BITS=1`, off by default: a
/// `BTreeSet` and a `HashSet` under locks of their own, as before). Either
/// way the sets hold the same slots; the slots come out in slot order.
///
/// Ported from cool_fox c3e643e (its bit sets only; the parallel heap
/// buffer pass of that commit is another port). cool_fox measured a pass
/// through a set of 70,000 slots under its lock keeping every writer of
/// slots waiting for 5 ms of a frame.
fn slot_bits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SLOT_BITS").is_some_and(|v| v == "1"))
}

/// A heap's sets of buffer views as bits ([`slot_bits`]).
struct BufferBits {
    /// [`HostHeap::buffers`].
    buffers: Slots,
    /// [`HostHeap::changed`].
    changed: Slots,
}

/// A set of a heap's slots, a bit each in atomic words: put, removed and
/// taken with no lock. A slot past its words (the count it was made for,
/// rounded up to 64) is never in it.
struct Slots(Box<[std::sync::atomic::AtomicU64]>);

impl Slots {
    fn new(count: usize) -> Self {
        Slots(
            (0..count.div_ceil(64))
                .map(|_| std::sync::atomic::AtomicU64::new(0))
                .collect(),
        )
    }

    fn put(&self, slot: u32) {
        if let Some(bits) = self.0.get(slot as usize / 64) {
            bits.fetch_or(1 << (slot % 64), std::sync::atomic::Ordering::AcqRel);
        }
    }

    /// Whether it was there.
    fn remove(&self, slot: u32) -> bool {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let bit = 1 << (slot % 64);
        self.0.get(slot as usize / 64).is_some_and(|bits| {
            // (A slot not there, most of them, leaves the word unwritten.)
            bits.load(Acquire) & bit != 0 && bits.fetch_and(!bit, AcqRel) & bit != 0
        })
    }

    /// The slots of word `n`, whose bits are `bits`, into `out` in order.
    fn of(n: usize, mut bits: u64, out: &mut Vec<u32>) {
        while bits != 0 {
            out.push(n as u32 * 64 + bits.trailing_zeros());
            bits &= bits - 1;
        }
    }

    /// Those in the set, in order.
    fn all(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for (n, bits) in self.0.iter().enumerate() {
            Self::of(n, bits.load(std::sync::atomic::Ordering::Acquire), &mut out);
        }
        out
    }

    /// Those in the set, in order, and none is left.
    fn take(&self) -> Vec<u32> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let mut out = Vec::new();
        for (n, bits) in self.0.iter().enumerate() {
            if bits.load(Acquire) != 0 {
                Self::of(n, bits.swap(0, AcqRel), &mut out);
            }
        }
        out
    }
}

/// Whether a buffer view of record `key` left for a draw in a slot made
/// from nothing that was last left for a draw from `left`
/// ([`HostHeap::left`]) leaves the slot as it is: the same record again,
/// or a constant buffer view where one was left before (a deferred
/// constant view is a null constant view and the constant word whatever
/// its address, in no set of buffer views). How, if so.
fn left_again(left: u64, key: u64) -> Option<crate::frame_budget::LeftAgain> {
    use crate::frame_budget::LeftAgain;
    let constant = |key: u64| (key >> 32) as u32 >> super::KIND_SHIFT == super::CONSTANT;
    if left == 0 {
        None
    } else if left == key {
        Some(LeftAgain::Same)
    } else if constant(left) && constant(key) {
        Some(LeftAgain::Constant)
    } else {
        None
    }
}

/// A claim on one slot of [`HostHeap::made`] ([`HostHeap::claim`]).
struct Claimed<'a> {
    slot: &'a std::sync::atomic::AtomicU64,
    /// The slot's [`HostHeap::left`], if the heap keeps it.
    left_at: Option<&'a std::sync::atomic::AtomicU64>,
    /// What the slot held when it was claimed.
    old: u64,
    /// What it holds once the claim is dropped.
    value: u64,
    /// The buffer record the slot is left for a draw from once the claim is
    /// dropped ([`HostHeap::left`]); 0, the default: any other change, or
    /// none.
    left: u64,
}

impl Claimed<'_> {
    /// The buffer record the slot was last left for a draw from when it was
    /// claimed (0: none). Exact: only claims write it.
    fn left_before(&self) -> u64 {
        self.left_at
            .map_or(0, |left| left.load(std::sync::atomic::Ordering::Relaxed))
    }
}

impl Drop for Claimed<'_> {
    fn drop(&mut self) {
        // A slot left for a draw is made from nothing.
        debug_assert!(
            self.left == 0 || self.value == 0,
            "a slot left for a draw shows a made record"
        );
        // Before `made`, with release: a thread that reads `made` 0 from
        // this store reads this record or a later one. (Left as it is when
        // unchanged: most claims are of slots no deferral touched.)
        if let Some(left) = self.left_at
            && left.load(std::sync::atomic::Ordering::Relaxed) != self.left
        {
            left.store(self.left, std::sync::atomic::Ordering::Release);
        }
        // After the host slot, its word and what it keeps: a thread that
        // reads this value sees them too.
        self.slot
            .store(self.value, std::sync::atomic::Ordering::Release);
        if cfg!(debug_assertions) {
            CLAIMS_HELD.with(|held| held.set(held.get() - 1));
        }
    }
}

thread_local! {
    /// Debug builds: the claims of slots ([`HostHeap::claim`]) this thread
    /// holds now (release builds do not count them).
    static CLAIMS_HELD: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Tests: called once, by this thread's next mirror of a slot, with the
    /// slot's [`HostHeap::made`], just before it claims the slot (after
    /// the view was resolved, if it was).
    static BEFORE_CLAIM: std::cell::Cell<Option<fn(&std::sync::atomic::AtomicU64)>> =
        const { std::cell::Cell::new(None) };
    /// Tests: called once, by this thread's next mirror of a slot that did
    /// not show its record at the first look, with the slot's
    /// [`HostHeap::made`], before the look that asks whether to resolve.
    static BEFORE_LOOK: std::cell::Cell<Option<fn(&std::sync::atomic::AtomicU64)>> =
        const { std::cell::Cell::new(None) };
    /// Tests: called once, by this thread's next mirror that finds no view
    /// in a slot, with the heap, after the record is read and before the
    /// slot is claimed.
    pub(crate) static BEFORE_NULL: std::cell::Cell<Option<fn(&HostHeap)>> =
        const { std::cell::Cell::new(None) };
    /// Tests: called once, by this thread's next mirror that changes a
    /// slot, under the claim, before the write is noted in the spans.
    static BEFORE_NOTE: std::cell::Cell<Option<fn()>> = const { std::cell::Cell::new(None) };
    /// Tests: called once, by this thread's next pass over views to make
    /// again, with the heap, between finding the views and claiming their
    /// slots.
    static BEFORE_REMADE: std::cell::Cell<Option<fn(&HostHeap)>> =
        const { std::cell::Cell::new(None) };
}

/// How many claims of slots ([`HostHeap::claim`]) this thread holds now;
/// 0 in release builds, which do not count them. Nothing that may make a
/// host resource or take a global lock runs while one is held
/// ([`resolve`], `crate::shadow`'s copies).
pub(crate) fn claims_held() -> u32 {
    CLAIMS_HELD.with(std::cell::Cell::get)
}

/// A change to a heap's queue of textures picked by index from a write of
/// a slot ([`HostHeap::queue_texture`], [`HostHeap::note_buffer`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextureNote {
    /// [`TextureQueue::written`].
    Written {
        index: u32,
        frame: u64,
        due: u64,
        seen: bool,
    },
    /// [`TextureQueue::remove`].
    Remove(u32),
    /// [`TextureQueue::changed`].
    Changed { index: u32, frame: u64, due: u64 },
}

impl TextureNote {
    fn apply(self, textures: &mut TextureQueue) {
        match self {
            TextureNote::Written {
                index,
                frame,
                due,
                seen,
            } => textures.written(index, frame, due, seen),
            TextureNote::Remove(index) => textures.remove(index),
            TextureNote::Changed { index, frame, due } => textures.changed(index, frame, due),
        }
    }
}

/// A change to a heap's sets of buffer views from a write of slot `index`
/// ([`HostHeap::note_buffer`]): `buffer`, whether it now holds a buffer
/// view; `mark`, whether the change goes into the set of changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferNote {
    index: u32,
    buffer: bool,
    mark: bool,
}

impl BufferNote {
    /// Applies it to the set of slots holding buffer views; true if the
    /// slot goes into the set of changes.
    fn apply(self, buffers: &mut std::collections::BTreeSet<u32>) -> bool {
        let was = if self.buffer {
            buffers.insert(self.index);
            true
        } else {
            buffers.remove(&self.index)
        };
        was && self.mark
    }

    /// [`BufferNote::apply`] to the sets as bits ([`slot_bits`]).
    fn apply_bits(self, bits: &BufferBits) {
        let was = if self.buffer {
            bits.buffers.put(self.index);
            true
        } else {
            bits.buffers.remove(self.index)
        };
        if was && self.mark {
            bits.changed.put(self.index);
        }
    }
}

/// Logs of the changes writes of slots make to a heap's queue of textures
/// and sets of buffer views ([`HostHeap::made`]):
/// each thread appends to a log of its own (one of [`LOG_STRIPES`]), so
/// threads that write different slots wait for no lock in common, and
/// whatever reads the queue or the sets applies the changes logged until
/// then ([`HostHeap::texture_queue`], [`HostHeap::buffer_set`]).
///
/// Each change is numbered as it is logged, under its log's lock, and a
/// take applies exactly the changes numbered below what the numbering had
/// reached when it began, in the order of their numbers; one numbered
/// later that it found already waits for the next take. So the changes
/// are applied in the order they were logged, over all logs and takes: a
/// change logged after another (two writes of one slot, each under the
/// slot's claim; a removal of a stale slot and the slot's own write, each
/// under the spans' lock) is applied after it.
#[derive(Default)]
struct NoteLog {
    /// Numbers the changes, for their order over all logs.
    next: std::sync::atomic::AtomicU64,
    /// Changes logged since the last take, of each kind (an upper bound: 0
    /// lets a reader skip the logs).
    texture_count: std::sync::atomic::AtomicUsize,
    buffer_count: std::sync::atomic::AtomicUsize,
    textures: [Mutex<Vec<(u64, TextureNote)>>; LOG_STRIPES],
    buffers: [Mutex<Vec<(u64, BufferNote)>>; LOG_STRIPES],
    /// Changes a take found that were numbered after it began, for the
    /// next take (under the queue's or the set's lock, as the takes).
    later_textures: Mutex<Vec<(u64, TextureNote)>>,
    later_buffers: Mutex<Vec<(u64, BufferNote)>>,
}

/// Logs per kind of [`NoteLog`]: threads beyond that share them.
const LOG_STRIPES: usize = 16;

/// Changes of a kind logged since the last take, past which the writer
/// that logs one more applies them itself, if the queue or the set is
/// free ([`HostHeap::note_textures`]): a heap that no pass reads for a
/// while (loading, or a heap no list picks buffers from) kept every write
/// it had.
const LOG_MOST: usize = 1 << 14;

/// The [`NoteLog`] log of this thread.
fn log_stripe() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local! {
        static STRIPE: usize = NEXT.fetch_add(1, Relaxed) % LOG_STRIPES;
    }
    STRIPE.with(|stripe| *stripe)
}

impl NoteLog {
    /// Logs `notes`; true if the log is longer than [`LOG_MOST`].
    fn textures(&self, notes: &[TextureNote]) -> bool {
        use std::sync::atomic::Ordering::{Relaxed, Release};
        let mut log = self.textures[log_stripe()]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for &note in notes {
            log.push((self.next.fetch_add(1, Relaxed), note));
        }
        drop(log);
        self.texture_count.fetch_add(notes.len(), Release) + notes.len() > LOG_MOST
    }

    /// Logs `note`; true if the log is longer than [`LOG_MOST`].
    fn buffers(&self, note: BufferNote) -> bool {
        use std::sync::atomic::Ordering::{Relaxed, Release};
        let mut log = self.buffers[log_stripe()]
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        log.push((self.next.fetch_add(1, Relaxed), note));
        drop(log);
        self.buffer_count.fetch_add(1, Release) + 1 > LOG_MOST
    }

    /// The texture changes logged so far, in the order they were logged.
    /// To be called under the queue's lock.
    fn take_textures(&self) -> Vec<TextureNote> {
        self.take(&self.texture_count, &self.textures, &self.later_textures)
    }

    /// The buffer changes logged so far, in order; under the set's lock.
    fn take_buffers(&self) -> Vec<BufferNote> {
        self.take(&self.buffer_count, &self.buffers, &self.later_buffers)
    }

    fn take<T>(
        &self,
        count: &std::sync::atomic::AtomicUsize,
        logs: &[Mutex<Vec<(u64, T)>>],
        later: &Mutex<Vec<(u64, T)>>,
    ) -> Vec<T> {
        use std::sync::atomic::Ordering::{Acquire, Relaxed};
        // Every change numbered below this is in its log by the time the
        // log's lock is taken below: it is numbered and logged under that
        // lock, and the number was given out before this was read.
        let below = self.next.load(Acquire);
        let mut later = later.lock().unwrap_or_else(|e| e.into_inner());
        // (A change logged but not counted yet is taken all the same, and
        // one counted but taken already leaves a look that finds nothing.
        // One not counted yet and not taken is still being logged: it is
        // as if this take came before it.)
        if count.load(Acquire) == 0 && later.is_empty() {
            return Vec::new();
        }
        count.store(0, Relaxed);
        let mut all = std::mem::take(&mut *later);
        for log in logs {
            all.append(&mut log.lock().unwrap_or_else(|e| e.into_inner()));
        }
        all.sort_unstable_by_key(|&(at, _)| at);
        *later = all.split_off(all.partition_point(|&(at, _)| at < below));
        all.into_iter().map(|(_, note)| note).collect()
    }
}

/// Writes of a slot with another view of the placed texture it showed,
/// since the start: the slot kept showing the old view.
pub(crate) static KEPT_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether a slot written with another view of the placed texture it
/// shows keeps showing the old one until the new is made;
/// `VICTORY_KEEP_VIEWS=0` makes it null at once, as before.
fn keep_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_VIEWS").is_none_or(|v| v != "0"))
}

/// Whether both are shader views of one texture the title placed in
/// memory, the same address and the same resource description, that
/// differ in their mips at most. The old view shows a sharper or blurrier
/// picture of the same thing; one of another array slice, plane or format
/// (a video's ring of pictures) would show an older picture instead.
fn same_placed_texture(a: &View, b: &View) -> bool {
    match (a, b) {
        (View::Placed(false, resource, desc), View::Placed(false, other, other_desc)) => {
            resource[0] != 1
                && resource == other
                && desc[10..12] == other_desc[10..12]
                && (0..10).all(|i| desc[i] == other_desc[i] || mip_word(desc[1], i))
        }
        _ => false,
    }
}

/// Whether word `i` of a shader view's description (the host's layout) is
/// one of its mip range or its least detail clamp, for view dimension
/// `dimension`. Word 3 is padding; the union starts at word 4.
fn mip_word(dimension: u32, i: usize) -> bool {
    let clamp = match D3D12_SRV_DIMENSION(dimension as i32) {
        D3D12_SRV_DIMENSION_TEXTURE1D
        | D3D12_SRV_DIMENSION_TEXTURE3D
        | D3D12_SRV_DIMENSION_TEXTURECUBE => 6,
        D3D12_SRV_DIMENSION_TEXTURE2D => 7,
        D3D12_SRV_DIMENSION_TEXTURE1DARRAY | D3D12_SRV_DIMENSION_TEXTURECUBEARRAY => 8,
        D3D12_SRV_DIMENSION_TEXTURE2DARRAY => 9,
        // Multisampled views have no mips.
        _ => return i == 3,
    };
    matches!(i, 3..=5) || i == clamp
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// With `VICTORY_NULL_MAGENTA=1`, a 2D colour texture view that shows
/// nothing yet shows magenta: on the screen, a texture that is not there is
/// told from one whose contents are black. False if it is to be null.
#[cfg(feature = "diagnostics")]
fn magenta_for(desc: &[u32; 16], at: D3D12_CPU_DESCRIPTOR_HANDLE) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_NULL_MAGENTA").is_some_and(|v| v == "1")) {
        return false;
    }
    // Block-compressed colour (BC1 to BC3, BC7) and 8-bit RGBA and BGRA.
    let colour = matches!(desc[0], 27..=29 | 70..=78 | 87..=91 | 97..=99);
    if desc[1] != D3D12_SRV_DIMENSION_TEXTURE2D.0 as u32 || !colour {
        return false;
    }
    let (Some(host), Some(texture)) = (host(), magenta()) else {
        return false;
    };
    // SAFETY: a slot of our heap and a live texture.
    unsafe { host.device.CreateShaderResourceView(texture, None, at) };
    true
}

/// One magenta texel, written by the processor.
#[cfg(feature = "diagnostics")]
fn magenta() -> Option<&'static ID3D12Resource> {
    static TEXTURE: std::sync::OnceLock<Option<ID3D12Resource>> = std::sync::OnceLock::new();
    TEXTURE
        .get_or_init(|| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 1,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: valid descriptions and an out pointer; the texel is
            // written while nothing reads the texture.
            unsafe {
                host()?
                    .device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut texture,
                    )
                    .ok()?;
                let texture = texture?;
                crate::host::label(&texture, "magenta texel", 4);
                texture.Map(0, None, None).ok()?;
                let texel = [255u8, 0, 255, 255];
                texture
                    .WriteToSubresource(0, None, texel.as_ptr().cast(), 4, 4)
                    .ok()?;
                texture.Unmap(0, None);
                Some(texture)
            }
        })
        .as_ref()
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            #[cfg(feature = "diagnostics")]
            if magenta_for(desc, at) {
                return;
            }
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    null_uav(&desc, at);
}

/// A null unordered access view described by `desc` at `at`: reads give
/// zeros, writes are dropped.
fn null_uav(desc: &D3D12_UNORDERED_ACCESS_VIEW_DESC, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(desc),
            at,
        )
    };
}

/// A raw buffer UAV of one dword, as null views of raw buffers are made.
fn null_raw_uav_desc() -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
        Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_UAV {
                NumElements: 1,
                Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                ..Default::default()
            },
        },
    }
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, one of the title's compute programs, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

impl Drop for HostHeap {
    fn drop(&mut self) {
        crate::draw::forget_heap(self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A set of slots as bits ([`Slots`]) holds what was put in and not
    /// removed, in slot order, and a take empties it. (Ported from
    /// cool_fox c3e643e.)
    #[test]
    fn a_set_of_slots_holds_what_was_put_in() {
        let set = Slots::new(200);
        for slot in [199, 0, 64, 63, 130, 64, 4000] {
            set.put(slot);
        }
        assert_eq!(set.all(), [0, 63, 64, 130, 199], "none past its words");
        assert!(set.remove(64) && !set.remove(64) && !set.remove(5) && !set.remove(4000));
        assert_eq!(set.take(), [0, 63, 130, 199]);
        assert!(set.take().is_empty() && set.all().is_empty());
    }

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// What the texture code asks of the spans (levels on demand): where
    /// the next texture starts inside a span, the lowest level a view of a
    /// texture shows, and a texture's slots.
    #[test]
    fn spans_say_the_next_start_and_the_lowest_level_viewed() {
        let mut spans = Spans::default();
        // A 512x512 texture of 0x56000 bytes viewed from levels 3 and 1,
        // and the next one 0x6000 after it (run 188).
        spans.note(1, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(1, 3);
        spans.note(2, 0x1_0000, 0x1_0000 + 0x5_6000, true);
        spans.shows(2, 1);
        spans.note(3, 0x1_6000, 0x1_6000 + 0x5_6000, true);
        assert_eq!(spans.next_start(0x1_0000, 0x1_0000 + 0x5_6000), Some(0x1_6000));
        assert_eq!(spans.next_start(0x1_6000, 0x1_6000 + 0x5_6000), None);
        assert_eq!(spans.first_viewed(0x1_0000), Some(1));
        assert_eq!(spans.first_viewed(0x1_6000), None);
        assert_eq!(spans.slots_at(0x1_0000), vec![1, 2]);
        // The view from level 1 goes: the texture is viewed from level 3.
        spans.forget(2);
        assert_eq!(spans.first_viewed(0x1_0000), Some(3));
    }

    /// Views of one placed texture are told from views of others by the
    /// address and the resource description, not by what they show.
    #[test]
    fn views_of_one_placed_texture_are_told_by_address_and_description() {
        let texture = [3, 0, 0, 0, 512, 0, 512, 0x8_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |resource: [u32; 14], first_mip: u32, address: u32| {
            let mut desc = [0u32; 16];
            (desc[0], desc[1], desc[4], desc[5], desc[10], desc[11]) = (98, 4, first_mip, 8 - first_mip, address, 0xb);
            View::Placed(false, resource, desc)
        };
        let low = view(texture, 3, 0x1000_0000);
        // more mips of the same texture
        assert!(same_placed_texture(&low, &view(texture, 0, 0x1000_0000)));
        // another address, another description, a buffer, an unordered view
        assert!(!same_placed_texture(&low, &view(texture, 3, 0x2000_0000)));
        let mut wider = texture;
        wider[4] = 1024;
        assert!(!same_placed_texture(&low, &view(wider, 3, 0x1000_0000)));
        let mut buffer = texture;
        buffer[0] = 1;
        assert!(!same_placed_texture(&view(buffer, 3, 0x1000_0000), &view(buffer, 3, 0x1000_0000)));
        let View::Placed(_, resource, desc) = low.clone() else { unreachable!() };
        assert!(!same_placed_texture(&low, &View::Placed(true, resource, desc)));
        assert!(!same_placed_texture(&low, &View::Shader(0, None)));
        // Of the same texture, but not only its mips: another plane or
        // format; a clamp of the least detail is a mip range still.
        let with = |word: usize, value: u32| {
            let mut other = desc;
            other[word] = value;
            View::Placed(false, resource, other)
        };
        assert!(!same_placed_texture(&low, &with(6, 1)), "plane");
        assert!(!same_placed_texture(&low, &with(0, 99)), "format");
        assert!(
            same_placed_texture(&low, &with(7, 2.0f32.to_bits())),
            "clamp"
        );
        // Array slices of one texture (a ring of video pictures).
        let slice = |first: u32| {
            let mut array = desc;
            (array[1], array[6], array[7]) = (5, first, 1);
            View::Placed(false, resource, array)
        };
        assert!(same_placed_texture(&slice(2), &slice(2)));
        assert!(!same_placed_texture(&slice(2), &slice(3)));
    }

    /// A faulting address is low below 4 GiB, mid within 4 GiB of a host
    /// heap and wild elsewhere, with the heap nearest to it.
    #[test]
    fn a_fault_is_classed_by_its_distance_from_the_heaps() {
        const GIB: u64 = 1 << 30;
        // Heaps of 2^21 views of 32 bytes (64 MiB): the "no descriptor"
        // index lies past their end, not inside them.
        const HEAP: u64 = (1 << 21) * 32;
        let heaps = [(16 * GIB, HEAP, 32, false), (40 * GIB, HEAP, 32, true)];
        assert!(
            fault_class_in(37 << 20, &heaps)
                .starts_with("; fault class low, nearest the view heap")
        );
        // The 24-bit "no descriptor" index read as a view past the heap's end.
        let past = 16 * GIB + 0x00FF_FFFF * 32;
        assert!(past > 16 * GIB + HEAP);
        assert_eq!(
            fault_class_in(past, &heaps),
            format!(
                "; fault class mid, nearest the view heap {:#x}+{HEAP:#x}",
                16 * GIB
            )
        );
        assert!(
            fault_class_in(15 * GIB, &heaps)
                .starts_with("; fault class mid, nearest the view heap")
        );
        assert!(
            fault_class_in(42 * GIB, &heaps)
                .starts_with("; fault class mid, nearest the sampler heap")
        );
        assert!(fault_class_in(28 * GIB, &heaps).starts_with("; fault class wild"));
        assert_eq!(
            fault_class_in(28 * GIB, &[]),
            "; fault class wild, no host heap"
        );
    }

    /// A constant buffer view larger than the host takes is cut to what it
    /// takes: the device, which such a view removes, stays.
    #[test]
    fn an_oversized_constant_buffer_view_leaves_the_device_alive() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        // SAFETY: a plain device query.
        let alive = || unsafe { host.device.GetDeviceRemovedReason() }.is_ok();
        // Made first: an allocation that fails on a removed device must not
        // end the test early.
        let (buffer, _) = host.upload_buffer(1 << 20).expect("an upload buffer");
        let memory = vec![0u8; 2 << 20];
        let at = (memory.as_ptr() as u64).next_multiple_of(64 << 10);
        for size in [969_472u32, 896_256] {
            assert!(make(&View::Constant(at, size), heap.cpu(0)).is_some());
            assert!(alive(), "a view of {size} bytes");
        }
        // SAFETY: a live upload buffer.
        heap.constant_at(
            heap.count + 1,
            unsafe { buffer.GetGPUVirtualAddress() },
            969_472,
        );
        assert!(alive(), "a view of the list's own memory");
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
    }

    /// At the title's end the slots that view a resource the release names
    /// let it go, and only those: the others keep theirs.
    #[test]
    fn let_go_kept_takes_only_the_named_resources() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let (ours, _) = host.upload_buffer(64 << 10).expect("an upload buffer");
        let (theirs, _) = host.upload_buffer(64 << 10).expect("an upload buffer");
        // Dynamic slots: a title slot changes only under its claim.
        let at = |k: u32| heap.count + k;
        heap.keep(at(0), Some(ours.clone()));
        heap.keep(at(2), Some(theirs.clone()));
        heap.keep(at(3), Some(ours.clone()));
        let before = crate::host::references(&ours);
        let mut out = Vec::new();
        heap.let_go_kept(&|resource| resource == &ours, &mut out);
        assert_eq!(out.len(), 2, "both slots that view it");
        drop(out);
        assert_eq!(crate::host::references(&ours), before - 2, "the slots hold it no longer");
        assert!(!heap.kept_at(at(0), |kept| kept.is_some()));
        assert!(!heap.kept_at(at(3), |kept| kept.is_some()));
        assert!(
            heap.kept_at(at(2), |kept| kept == Some(&theirs)),
            "the other one stays"
        );
    }

    /// The texture that stands for missing ones can be made and viewed.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_magenta_texture_is_made() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let texture = magenta().expect("the texture");
        // SAFETY: a slot of our heap and a live texture.
        unsafe {
            host.device
                .CreateShaderResourceView(texture, None, heap.cpu(0))
        };
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    #[test]
    fn views_past_the_end_are_cut_to_the_elements_inside() {
        assert_eq!(elements_inside(0, 10, 16, 160), 10);
        assert_eq!(elements_inside(4, 10, 16, 160), 6);
        assert_eq!(elements_inside(4, 10, 16, 167), 6);
        // Nothing inside: as it was, for the host check to refuse.
        assert_eq!(elements_inside(10, 10, 16, 160), 10);
        assert_eq!(elements_inside(0, 7, 4, 1 << 40), 7);
    }

    /// Whose the memory of a view is: the slot's own span, no other's, or
    /// one that says otherwise (by the rule of starts).
    #[test]
    fn spans_say_whose_memory_a_view_is() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x1_0000, 0x3_0000, true).is_empty());
        assert_eq!(spans.claim(1, 0x1_0000, 0x3_0000), (Claim::Own, None));
        // Another slot's view of the same texture, or of memory nothing describes.
        assert_eq!(spans.claim(2, 0x1_0000, 0x3_0000), (Claim::Free, None));
        assert_eq!(spans.claim(2, 0x3_0000, 0x4_0000), (Claim::Free, None));
        // The same start with another length; another start in its first block.
        assert_eq!(spans.claim(2, 0x1_0000, 0x2_0000), (Claim::SameStart, Some((0x1_0000, 0x3_0000))));
        assert_eq!(spans.claim(2, 0x1_0800, 0x2_0000), (Claim::Other, Some((0x1_0000, 0x3_0000))));
    }

    /// A view that fell out of the queue with its memory there and nothing
    /// to show for it goes back; one a newer view speaks against does not.
    #[test]
    fn a_texture_sweep_finds_views_nothing_brings_up_to_date() {
        let Some(host) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // Below 2^40, as a title's memory is.
        // SAFETY: new memory, left to the process.
        let at = unsafe {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            VirtualAlloc(
                0x7d_4000_0000usize as *const _,
                1 << 20,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as u64
        };
        if at == 0 {
            return; // taken
        }
        let mut slots = vec![[0u32; 8]; 8];
        let base = slots.as_mut_ptr() as u64;
        let handle = |index: u32| Some(base + u64::from(index) * 32);
        let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let view = |width: u32, address: u64| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            let mut resource = resource;
            resource[4] = width;
            View::Placed(false, resource, desc)
        };
        super::super::record(handle(1).unwrap(), view(64, at));
        super::super::record(handle(2).unwrap(), view(64, at + 0x8_0000));
        // A newer, wider texture in the second one's memory.
        host.note_buffer(5, Some(&view(128, at + 0x8_0000)), false, true);
        let queued = || -> Vec<u32> {
            host.texture_slots_due(u64::MAX, 8)
                .0
                .into_iter()
                .map(|d| d.0)
                .collect()
        };
        for slot in queued() {
            host.textures.lock().unwrap().remove(slot);
        }
        let mut counts = [0; 10];
        host.sweep_part(0..8, handle, 50, false, &mut counts);
        assert_eq!(counts, [2, 0, 0, 0, 0, 0, 1, 1, 0, 0]);
        assert_eq!(queued(), []);
        host.sweep_part(0..8, handle, 50, true, &mut counts);
        assert_eq!(queued(), [1]);
    }

    /// The table of strides tells shaders a buffer view (its stride) from a
    /// constant buffer view or a buffer of no known element size
    /// (`SLOT_CONSTANT`), and those from anything else (0): a constant
    /// buffer read at a 0 reads the null view, not a texture.
    #[test]
    fn the_table_of_strides_tells_constant_buffers_from_textures() {
        use victory_shader::emit::SLOT_CONSTANT;
        // Format, dimension (1: buffer), mapping, a pad, first element,
        // elements, stride, flags (1: raw).
        let buffer = |format: u32, stride: u32, flags: u32| {
            View::Shader(0, Some([format, 1, 0, 0, 0, 0, 64, stride, flags, 0]))
        };
        assert_eq!(slot_word(Some(&buffer(0, 12, 0))), 12);
        assert_eq!(slot_word(Some(&buffer(1, 0, 1))), 1);
        // A typed buffer of a format we do not size: read as before.
        assert_eq!(slot_word(Some(&buffer(0xFFFF, 0, 0))), SLOT_CONSTANT);
        let constant = View::Constant(0x1_0000, 256);
        assert_eq!(slot_word(Some(&constant)), SLOT_CONSTANT);
        // A texture (dimension 4: 2D), a placed texture, nothing.
        let texture = View::Shader(0, Some([28, 4, 0, 0, 0, 0, 0, 0, 0, 0]));
        assert_eq!(slot_word(Some(&texture)), 0);
        let resource = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        assert_eq!(slot_word(Some(&View::Placed(false, resource, [0; 16]))), 0);
        assert_eq!(slot_word(None), 0);
    }

    /// The title rewrites a slot from a constant buffer view to a texture
    /// view and empties it: the table follows each write (what the run that
    /// removed the device read as a constant buffer was a texture view).
    #[test]
    fn a_rewritten_slot_has_its_word_in_the_table() {
        use victory_shader::emit::SLOT_CONSTANT;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let mut slots = vec![[0u32; 8]; 8];
        let at = slots.as_mut_ptr() as u64 + 32;
        super::super::record(at, View::Constant(0x7d_4000_0000, 256));
        heap.mirror(1, at, false);
        assert_eq!(heap.word(1), Some(SLOT_CONSTANT), "a constant buffer view");
        let texture = [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0];
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (0x4000_0000, 0x7d);
        super::super::record(at, View::Placed(false, texture, desc));
        heap.mirror(1, at, false);
        assert_eq!(heap.word(1), Some(0), "a texture view");
        super::super::record(at, View::Constant(0x7d_4000_0000, 256));
        heap.mirror(1, at, false);
        // Emptied, and found so by a draw (until then the slot holds the
        // null view of what it held, and its word still fits it).
        // SAFETY: the record's first words, in `slots`.
        unsafe { (at as *mut [u32; 3]).write_unaligned([0; 3]) };
        heap.mirror(1, at, false);
        assert_eq!(heap.word(1), Some(SLOT_CONSTANT), "emptied, not yet seen");
        heap.mirror(1, at, true);
        assert_eq!(heap.word(1), Some(0), "emptied");
    }

    /// The heap buffer pass is claimed once a frame: every other caller
    /// of the frame loses to that claim, and one that read an older frame
    /// loses to the newest. A newer walk that ends lets go the lists that
    /// lost to an older claim too (it began after their binds), and the
    /// older walk's end takes nothing back.
    #[test]
    fn the_heap_buffer_pass_is_claimed_once_a_frame() {
        let gate = std::sync::Arc::new(WalkGate::new(u64::MAX - 10));
        let first = gate.claim(5).expect("the first claim of frame 5");
        assert_eq!(gate.claim(5).err(), Some(6), "the frame's second caller");
        assert!(!gate.walked(6));
        let second = gate.claim(6).expect("the first claim of frame 6");
        assert_eq!(gate.claim(5).err(), Some(7), "a caller that read frame 5");
        assert_eq!(gate.claim(6).err(), Some(7));
        assert!(!gate.walked(6) && !gate.walked(7));
        drop(second);
        assert!(gate.walked(6) && gate.walked(7));
        drop(first);
        assert!(gate.walked(7), "not back to the older claim");
        assert!(!gate.walked(8));
        // Ended already: nothing to wait for.
        assert!(gate.wait(7, true));
    }

    /// A submission does not wait for ever for a walk that does not end
    /// (its thread gone without unwinding): after the long wait it goes on
    /// without the walk. A recording list waits on until the walk ends.
    #[test]
    fn a_submission_gives_up_a_walk_that_does_not_end() {
        let gate = std::sync::Arc::new(WalkGate::new(u64::MAX - 11));
        let walking = gate.claim(2).expect("the first claim of frame 2");
        let long = std::time::Duration::from_millis(50);
        let began = std::time::Instant::now();
        assert!(!gate.wait_within(3, true, long), "the walk has not ended");
        assert!(began.elapsed() >= long);
        assert!(!gate.walked(3), "giving up ends no walk");
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                let began = std::time::Instant::now();
                (gate.wait_within(3, false, long), began.elapsed())
            });
            std::thread::sleep(4 * long);
            drop(walking);
            let (ended, waited) = waiter.join().unwrap();
            assert!(ended && waited >= 3 * long, "waited {waited:?}");
        });
        assert!(gate.wait_within(3, true, long), "ended since");
    }

    /// The walk of the heap buffer pass takes what it makes from the set it
    /// walks and does not put it back: a slot it made is not walked again
    /// by the next pass; one the title wrote since is, and so is one a draw
    /// made from a record no write told us of.
    #[test]
    fn a_slot_the_pass_makes_is_not_walked_again() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        // A title buffer in memory of its own: a view of it is left for the
        // walk when the title writes it, and the walk finds its host copy.
        const SIZE: usize = 1 << 16;
        // SAFETY: new memory, left to the process.
        let memory = unsafe {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            VirtualAlloc(
                std::ptr::null(),
                SIZE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(memory, 0);
        crate::shadow::register(memory, memory as u64, SIZE as u64);
        // The buffer as the title placed it (dimension 1, its width), and
        // a view of 64 of its elements: format, dimension (1: buffer),
        // mapping, a pad, first element, elements, stride, flags.
        let buffer = |stride: u32| {
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 64, stride, 0, 0]);
            (desc[10], desc[11]) = (memory as u32, (memory as u64 >> 32) as u32);
            View::Placed(
                false,
                [1, 0, 0, 0, SIZE as u32, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
                desc,
            )
        };
        let mut slots = vec![[0u32; 8]; 8];
        let one = slots.as_mut_ptr() as u64 + 32;
        let two = one + 32;
        let changes = |heap: &HostHeap| {
            let mut slots = heap.buffer_changes();
            slots.sort_unstable();
            slots
        };
        let none: [u32; 0] = [];
        // The title writes two buffer views: left for the walk, both walked.
        super::super::record(one, buffer(12));
        heap.mirror(1, one, false);
        super::super::record(two, buffer(16));
        heap.mirror(2, two, false);
        assert!(!heap.made(1) && !heap.made(2), "left for the walk");
        assert_eq!(heap.word(1), Some(12), "a null buffer view under its word");
        assert_eq!(changes(&heap), [1, 2]);
        // The walk makes them and takes their copy: nothing for the next pass.
        let view = heap
            .mirror_marking(1, one, true, false, hold::WALK)
            .expect("a buffer view");
        assert!(
            buffer_copy(&view).is_some(),
            "the copy the walk brings up to date"
        );
        heap.mirror_marking(2, two, true, false, hold::WALK);
        assert!(heap.made(1) && heap.made(2));
        assert_eq!(changes(&heap), none, "made by the walk");
        // Walked once more with nothing written: still nothing.
        heap.mirror_marking(1, one, true, false, hold::WALK);
        assert_eq!(changes(&heap), none);
        // The title writes slot 1 again: walked again.
        super::super::record(one, buffer(20));
        heap.mirror(1, one, false);
        assert!(!heap.made(1), "left for the walk again");
        assert_eq!(changes(&heap), [1], "written by the title");
        heap.mirror_marking(1, one, true, false, hold::WALK);
        assert!(heap.made(1));
        assert_eq!(heap.word(1), Some(20));
        assert_eq!(changes(&heap), none);
        // Slot 2's record changes with no write we see (a byte copy into a
        // slot tables read): the draw that makes it marks it.
        super::super::record(two, buffer(24));
        heap.mirror(2, two, true);
        assert_eq!(changes(&heap), [2], "made by a draw from a new record");
        // The title empties slot 1's record with no write we see: the walk
        // empties the host slot and drops its range at once.
        // SAFETY: the record's first words, in `slots`.
        unsafe { (one as *mut [u32; 3]).write_unaligned([0; 3]) };
        assert!(
            heap.mirror_marking(1, one, true, false, hold::WALK)
                .is_none()
        );
        assert!(!heap.made(1));
        assert_eq!(heap.word(1), Some(0));
        assert_eq!(changes(&heap), none, "emptied by the walk");
        // Through the walk's own entry, which reads the switch (once for
        // the process): marked only with `VICTORY_HEAP_WALK_ONCE=0`.
        super::super::record(two, buffer(32));
        heap.mirror(2, two, false);
        assert_eq!(changes(&heap), [2]);
        assert!(heap.mirror_walked(2, two).is_some());
        assert_eq!(heap.word(2), Some(32));
        let again: &[u32] = if walk_once() { &[] } else { &[2] };
        assert_eq!(changes(&heap), again, "made through `mirror_walked`");
        // As before (`VICTORY_HEAP_WALK_ONCE=0`): the walk marks what it
        // made, and the next pass walks it again.
        super::super::record(one, buffer(28));
        heap.mirror(1, one, false);
        assert_eq!(changes(&heap), [1]);
        heap.mirror_marking(1, one, true, true, hold::TABLE);
        assert_eq!(changes(&heap), [1], "the double visit");
        drop(heap);
        crate::shadow::unregister(memory, memory as u64);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn a_structured_view_made_raw_passes_the_host_check() {
        // 100 elements of 20 bytes from element 8, in a copy that starts 32
        // bytes before the view's memory: bytes 192..2192.
        let (first, count) = raw_window(32, 8, 100, 20).unwrap();
        assert_eq!((first, count), (48, 500));
        assert!(buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 4096));
        // the format a raw view must name; the structured one's is refused
        assert!(!buffer_view_ok(DXGI_FORMAT_UNKNOWN, first, count, 0, true, 4096));
        // past the buffer, not on 16 bytes, no elements
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, first, count, 0, true, 2000));
        assert!(raw_window(4, 0, 100, 20).is_none());
        assert!(!buffer_view_ok(DXGI_FORMAT_R32_TYPELESS, 0, 0, 0, true, 4096));
        // a structured view the host takes as it is
        assert!(buffer_view_ok(DXGI_FORMAT_UNKNOWN, 1, 10, 12, false, 4096));
    }

    /// The queue of textures picked by index serves the slots looked at
    /// before first, then the newest write and the oldest in turns; a
    /// slot whose memory is not there is looked at again after
    /// 1, 2, 4 frames up to the limit; and what the pass reports about a
    /// slot written again meanwhile is dropped.
    #[test]
    fn the_texture_queue_serves_the_newest_writes_first() {
        let mut q = TextureQueue::default();
        // Slots 10..18 written in frames 1..9, slot 5 last.
        for i in 0..8u32 {
            q.written(10 + i, u64::from(i) + 1, 0, false);
        }
        q.written(5, 9, 0, false);
        let order: Vec<u32> = q.due(9, 100).into_iter().map(|d| d.0).collect();
        // newest, oldest, newest, oldest, and so on
        assert_eq!(order, [5, 10, 17, 11, 16, 12, 15, 13, 14]);
        assert_eq!(q.due(9, 2).len(), 2);
        assert_eq!(q.census(9), [9, 9, 0, 0, 8]);
        // Slot 17's memory is not mapped: again next frame, then in 2, 4, 4.
        let write = |q: &TextureQueue, index: u32| q.due(u64::MAX, 100).iter().find(|d| d.0 == index).unwrap().1;
        let w17 = write(&q, 17);
        let mut frame = 9;
        for wait in [1, 2, 4, 4] {
            q.later(17, w17, frame, 4);
            assert!(!q.due(frame + wait - 1, 100).iter().any(|d| d.0 == 17));
            frame += wait;
            // Due again, and ahead of every slot never looked at.
            assert_eq!(q.due(frame, 100)[0].0, 17);
        }
        assert_eq!(q.census(9)[2], 1);
        q.done(17, w17);
        assert!(!q.due(u64::MAX, 100).iter().any(|d| d.0 == 17));
        // Uploaded with part of its memory there: kept, with what was there.
        let w16 = write(&q, 16);
        q.partly(16, w16, 12, 4096);
        assert_eq!(q.mapped(16), 4096);
        assert_eq!(q.due(12, 100)[0].0, 16);
        // The title writes slot 16 again while the pass works on it: what
        // the pass then says about the old write changes nothing.
        q.written(16, 13, 0, false);
        q.done(16, w16);
        q.later(16, w16, 13, 4);
        assert_eq!(q.mapped(16), 0);
        assert_eq!(q.due(13, 100)[0].0, 16);
        // A draw that makes the host slot (`seen`) keeps what the queue knows,
        let w15 = write(&q, 15);
        q.partly(15, w15, 20, 8192);
        q.written(15, 14, 0, true);
        assert_eq!(q.mapped(15), 8192);
        // and queues a slot the queue did not hold.
        q.written(99, 14, 0, true);
        assert!(q.due(14, 100).iter().any(|d| d.0 == 99));
        // A write due later (a table's slot) is not handed out before.
        q.written(98, 14, 16, false);
        assert!(!q.due(15, 100).iter().any(|d| d.0 == 98));
        assert!(q.due(16, 100).iter().any(|d| d.0 == 98));
    }

    /// Slots held back for lack of room on the card are not handed out
    /// until their frame, keep their place, are not counted as waiting
    /// for their memory, and all come back at once when the card has room;
    /// a slot written again meanwhile is the title's new write.
    #[test]
    fn slots_held_for_lack_of_room_come_back_when_there_is_room() {
        let mut q = TextureQueue::default();
        for i in 0..4u32 {
            q.written(20 + i, 1, 0, false);
        }
        let due = q.due(1, 100);
        assert_eq!(due.len(), 4);
        // The card is full: slots 20 and 21 are held back until frame 101.
        for &(index, write) in due.iter().filter(|d| d.0 < 22) {
            q.hold(index, write, 101);
        }
        let now: Vec<u32> = q.due(2, 100).into_iter().map(|d| d.0).collect();
        assert!(!now.contains(&20) && !now.contains(&21));
        assert!(now.contains(&22) && now.contains(&23));
        assert_eq!(q.census(2)[2], 0, "held slots do not wait for their memory");
        // Due again at their frame, in the order they had.
        assert_eq!(q.due(101, 100), due);
        // Slot 21 is written again while held: due as written.
        q.written(21, 3, 0, false);
        assert!(q.due(3, 100).iter().any(|d| d.0 == 21));
        // Room again at frame 5: slot 20 is due at once, and only it was
        // still held as it was.
        assert_eq!(q.room(5), 1);
        assert!(q.due(5, 100).iter().any(|d| d.0 == 20));
        assert_eq!(q.room(6), 0);
    }

    /// Textures of which only the smallest levels are held, packed into
    /// one another's address range (the three of run 188: 0x2c000 bytes
    /// each at full size, their starts 0x2000 and 0x4000 apart): writing
    /// any of them, in any order and again, leaves the others.
    #[test]
    fn packed_neighbours_stay() {
        let mut spans = Spans::default();
        assert_eq!(spans.rule, Rule::Start);
        let at = 0xb_86bf_8000u64;
        let (a, b, c) = (at, at + 0x2000, at + 0x6000);
        assert!(spans.note(105_998, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        assert!(spans.note(108_521, c, c + 0x2c000, true).is_empty());
        // The first written again (as a title does when levels arrive),
        // from another slot too: its span reaches over the other two.
        assert!(spans.note(7, a, a + 0x2c000, true).is_empty());
        assert!(spans.note(106_004, b, b + 0x2c000, true).is_empty());
        let mut all = spans.slots_in(0, u64::MAX);
        all.sort_unstable();
        assert_eq!(all, [7, 105_998, 106_004, 108_521]);
        // A draw makes a slot of the second texture: it is one of them.
        assert!(spans.note(9, b, b + 0x2c000, false).is_empty());
        assert_eq!(spans.by_slot.get(&9), Some(&b));
    }

    /// A texture written where another starts replaces it: at the same
    /// address with another length, or inside the block of its smallest
    /// levels. The packed neighbours of the one replaced stay.
    #[test]
    fn a_texture_written_at_the_start_of_another_replaces_it() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        // Another length from the same start: slot 1 is stale, slot 2 not.
        assert_eq!(spans.note(3, at, at + 0xb000, true), [1]);
        // A start inside the first 4 KiB of slot 3's texture: stale too.
        assert_eq!(spans.note(4, at + 0x800, at + 0x1800, true), [3]);
        // And the other way: a texture that starts just before one.
        assert_eq!(spans.note(5, at + 0x1800, at + 0x4800, true), [2]);
        let mut left = spans.slots_in(0, u64::MAX);
        left.sort_unstable();
        assert_eq!(left, [4, 5]);
        // Exactly 4 KiB apart: both there.
        assert!(spans.note(6, at + 0x2800, at + 0x3800, true).is_empty());
        // A draw's view that clashes is left out, and nothing is stale.
        assert!(spans.note(8, at + 0x900, at + 0x2900, false).is_empty());
        assert_eq!(spans.by_slot.get(&8), None);
        assert_eq!(spans.by_slot.len(), 3);
    }

    /// A slot written with another texture leaves its old span; the same
    /// view written again changes nothing.
    #[test]
    fn a_slot_rewritten_moves_to_its_new_span() {
        let mut spans = Spans::default();
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert!(spans.note(2, 0x12000, 0x3e000, true).is_empty());
        assert!(spans.note(1, 0x10000, 0x3c000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [1]);
        // Slot 1 now views a texture far away.
        assert!(spans.note(1, 0x90000, 0x91000, true).is_empty());
        assert_eq!(spans.slots_in(0x10000, 0x10001), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x12000, 0x12001), [2]);
        assert_eq!(spans.slots_in(0x90000, 0x90001), [1]);
        spans.forget(1);
        spans.forget(2);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps, unmaps or writes names the textures that
    /// are there: among packed neighbours the one whose held levels the
    /// range lies in (it ends where the next one starts), and the ones
    /// that start inside the range.
    #[test]
    fn memory_changed_under_packed_neighbours_names_the_ones_there() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // Inside the first one's held levels.
        assert_eq!(spans.slots_in(at + 0x1000, at + 0x2000), [1]);
        // Inside the second's: not the first, which ends where it starts.
        assert_eq!(spans.slots_in(at + 0x3000, at + 0x4000), [2]);
        // Past the last start: the last one.
        assert_eq!(spans.slots_in(at + 0x20000, at + 0x21000), [3]);
        // A range over all of them.
        assert_eq!(spans.slots_in(at, at + 0x8000), [1, 2, 3]);
        // Past the end of all: none.
        assert_eq!(spans.slots_in(at + 0x32000, at + 0x33000), [] as [u32; 0]);
    }

    /// A write names the textures that hold its bytes: the one that
    /// starts where the write starts, not the neighbour before it, whose
    /// held levels end there.
    #[test]
    fn a_write_at_the_start_of_a_packed_texture_names_it_alone() {
        let mut spans = Spans::default();
        let at = 0x7_0000_0000u64;
        assert!(spans.note(1, at, at + 0x2c000, true).is_empty());
        assert!(spans.note(2, at + 0x2000, at + 0x2e000, true).is_empty());
        assert!(spans.note(3, at + 0x6000, at + 0x32000, true).is_empty());
        // A request that fills the second texture's held levels.
        assert_eq!(spans.slots_in(at + 0x2000, at + 0x6000), [1, 2]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x6000), [2]);
        // From inside the first one's held levels on: both.
        assert_eq!(spans.slots_holding(at + 0x1000, at + 0x3000), [1, 2]);
        // A whole 64 KiB page: every texture that starts in it.
        assert_eq!(spans.slots_holding(at, at + 0x10000), [1, 2, 3]);
        // Past the last start, the last one; past all, or no bytes, none.
        assert_eq!(spans.slots_holding(at + 0x20000, at + 0x21000), [3]);
        assert_eq!(spans.slots_holding(at + 0x32000, at + 0x33000), [] as [u32; 0]);
        assert_eq!(spans.slots_holding(at + 0x2000, at + 0x2000), [] as [u32; 0]);
    }

    /// A write behind the guards' back under a placed texture: a slot
    /// waiting for its first upload is left alone, one uploaded before is
    /// counted, and made due two frames on when that is asked for. (The
    /// ranges are handed over here, not through the list all heaps share.)
    #[test]
    fn a_page_write_makes_an_uploaded_texture_slot_due_when_asked() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fed_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(
            false,
            [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0],
            desc,
        );
        assert!(heap.note_buffer(1, Some(&view), false, true).is_empty());
        let written = [(at, at + 0x1000)];
        // Waiting for its first upload: held, not uploaded, nothing done.
        let after = Some(PAGE_WRITE_AFTER);
        assert_eq!(heap.slots_written_under(&written, 50, after), [1, 0, 0]);
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Uploaded and out of the queue. Memory elsewhere: nothing.
        let elsewhere = [(at + 0x10_0000, at + 0x10_1000)];
        assert_eq!(heap.slots_written_under(&elsewhere, 60, after), [0, 0, 0]);
        // Under it, counted only: it stays out of the queue.
        assert_eq!(heap.slots_written_under(&written, 60, None), [1, 1, 0]);
        assert_eq!(slots(u64::MAX), []);
        // Under it, asked for: due two frames on, not before.
        assert_eq!(heap.slots_written_under(&written, 60, after), [1, 1, 1]);
        assert_eq!(slots(61), []);
        assert_eq!(slots(62), [1]);
    }

    /// `VICTORY_STALE_TEXTURES=0`: nothing is ever stale, and a view that
    /// overlaps others is left out of the spans, as before.
    #[test]
    fn never_stale_leaves_overlapping_views_out() {
        let mut spans = Spans { rule: Rule::Never, ..Default::default() };
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x2000, 0x4000, true).is_empty());
        assert!(spans.note(3, 0x1000, 0x2000, true).is_empty());
        assert_eq!(spans.slots_in(0, u64::MAX), [1]);
    }

    /// The rule until run 188 (`VICTORY_STALE_TEXTURES=overlap`): a view
    /// written over memory that older views describe otherwise makes
    /// those stale; views of the same span are one texture's.
    #[test]
    fn a_newer_view_of_other_memory_makes_the_older_stale() {
        let mut spans = Spans { rule: Rule::Overlap, ..Default::default() };
        // Three textures side by side, the first in two slots.
        assert!(spans.note(1, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert!(spans.note(3, 0x3000, 0x4000, true).is_empty());
        assert!(spans.note(4, 0x8000, 0x9000, true).is_empty());
        // The same slot written with the same span again: nothing changes.
        assert!(spans.note(2, 0x1000, 0x3000, true).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        assert_eq!(spans.slots_in(0x4000, 0x8000), [] as [u32; 0]);
        // A draw makes a slot whose view overlaps them: left out, nothing stale.
        assert!(spans.note(9, 0x2000, 0x3800, false).is_empty());
        assert_eq!(spans.slots_in(0x2000, 0x3800), [1, 2, 3]);
        // The title writes a texture over the end of the first and all of
        // the second: their three slots are stale.
        let mut stale = spans.note(5, 0x2000, 0x4000, true);
        stale.sort_unstable();
        assert_eq!(stale, [1, 2, 3]);
        assert_eq!(spans.slots_in(0, u64::MAX), [5, 4]);
        // Another start in the same memory: stale in turn. Another length
        // from the same start too.
        assert_eq!(spans.note(6, 0x2800, 0x3000, true), [5]);
        assert_eq!(spans.note(7, 0x2800, 0x2c00, true), [6]);
        // A slot that moves to another texture leaves its span behind.
        assert!(spans.note(7, 0x8000, 0x9000, true).is_empty());
        assert_eq!(spans.slots_in(0x2800, 0x2c00), [] as [u32; 0]);
        assert_eq!(spans.slots_in(0x8000, 0x8001), [4, 7]);
        spans.forget(4);
        spans.forget(7);
        assert!(spans.by_start.is_empty() && spans.by_slot.is_empty());
    }

    /// Memory the runtime maps under a slot's texture makes the slot due
    /// again from the next frame; a stale slot leaves the queue.
    #[test]
    fn remapped_memory_makes_its_texture_slots_due() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fee_0000_0000u64;
        let view = |address: u64, width: u32| {
            let mut desc = [0u32; 16];
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(false, [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc)
        };
        let len = crate::resource::placed_bytes(&[
            3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0,
        ])
        .unwrap();
        assert!(
            heap.note_buffer(1, Some(&view(at, 64)), false, true)
                .is_empty()
        );
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        // Memory elsewhere: nothing. Under the texture: due from the next frame.
        remapped((at + 2 * len) as usize, 0x1000);
        remapped(at as usize, 0x1000);
        assert_eq!(heap.texture_slots_remapped(50), 1);
        assert_eq!(slots(50), []);
        assert_eq!(slots(51), [1]);
        // A wider texture written at the same address: slot 1 is stale.
        let stale = heap.note_buffer(2, Some(&view(at, 128)), false, true);
        assert_eq!(stale.iter().map(|&(slot, _)| slot).collect::<Vec<_>>(), [1]);
        assert_eq!(slots(u64::MAX), [2]);
    }

    /// The GPU's write through a home in a chunk of the memory pool under an
    /// uploaded placed texture makes its slot due two frames on, whatever
    /// `VICTORY_PAGE_WRITES` says, and is counted; memory elsewhere does
    /// nothing.
    #[test]
    fn a_pool_write_makes_an_uploaded_texture_slot_due() {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        // 64x64 BC7, one mip, at an address no test and no title uses.
        let at = 0x7fec_0000_0000u64;
        let mut desc = [0u32; 16];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0], desc);
        assert!(heap.note_buffer(1, Some(&view), false, true).is_empty());
        let (queued, write) = heap.texture_slots_due(u64::MAX, 8).0[0];
        assert_eq!(queued, 1);
        heap.texture_slot_done(1, write, None);
        assert_eq!(slots(u64::MAX), []);
        let due = POOL_WRITES[3].load(Relaxed);
        // Past every span: not kept, and said so.
        assert_eq!(pool_written(0x7fff_0000_0000, 0x1000), None);
        pool_written((at + 0x10_0000) as usize, 0x1000);
        assert_eq!(
            pool_written(at as usize, 0x1000),
            Some((at as usize, at as usize + 0x1000))
        );
        assert_eq!(heap.texture_slots_page_written(60), 1);
        assert!(POOL_WRITES[3].load(Relaxed) > due);
        assert_eq!(slots(60), []);
        assert_eq!(slots(61), [1], "the next frame: the GPU's write has landed");
    }

    /// The slots made on C-pool homes are found by any overlap with a range
    /// let go, however long their own range, and taken out once.
    #[test]
    fn pool_slots_are_found_by_overlap_and_taken_once() {
        let mut slots = PoolSlots::default();
        slots.set(1, Some((0x1000, 0x2000)));
        slots.set(2, Some((0x10_0000, 0x50_0000)));
        slots.set(3, Some((0x2000, 0x3000)));
        slots.set(4, Some((0x8000, 0x9000)));
        slots.set(4, None);
        let mut over = |start, end| {
            let mut got = slots.take_over(start, end);
            got.sort_unstable();
            got
        };
        assert_eq!(over(0x2000, 0x2001), [3], "touching is no overlap");
        assert_eq!(over(0x40_0000, 0x40_0010), [2], "inside a long range");
        assert_eq!(over(0x8000, 0x9000), [0u32; 0], "a slot noted gone");
        assert_eq!(over(0, u64::MAX), [1]);
        assert_eq!(over(0, u64::MAX), [0u32; 0], "taken out");
        slots.set(5, Some((0x1000, 0x2000)));
        assert_eq!(slots.take_all(), [5]);
    }

    /// A heap looks at the let-goes since its last look once; one that
    /// missed some (more went than were kept) takes all its slots.
    #[test]
    fn a_heap_takes_its_pool_slots_a_let_go_lets_go() {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let note = |slot: u32, start: u64| {
            let mut pool = heap.pool_slots.lock().unwrap();
            pool.set(slot, Some((start, start + 0x1000)));
            heap.pool_noted.store(pool.by_slot.len(), Relaxed);
        };
        let at = 0x7fed_0000_0000u64;
        note(1, at);
        note(2, at + 0x10_0000);
        pool_let_go(at + 0x800, at + 0x900);
        heap.pool_slots_let_go();
        assert_eq!(heap.buffer_changes(), [1]);
        heap.pool_slots_let_go();
        assert_eq!(heap.buffer_changes(), [0u32; 0], "looked at once");
        // More let-goes than are kept, none over slot 2: it goes too.
        for n in 0..POOL_GONE_MOST as u64 + 1 {
            let from = at + 0x100_0000 + n * 0x1000;
            pool_let_go(from, from + 1);
        }
        heap.pool_slots_let_go();
        assert_eq!(heap.buffer_changes(), [2]);
        assert_eq!(heap.pool_noted.load(Relaxed), 0);
    }

    /// A slot of a draw's table is left to the draws by the heap buffer
    /// pass (`VICTORY_HEAP_PASS_TABLED=1`) for [`TABLED_BUFFER_FRAMES`]
    /// frames after the last table that held it, and a slot never a
    /// table's is not. The marks are the pass's alone: the heap sync and
    /// the queue of textures picked by index ([`TABLED_FRAMES`]) do not
    /// see them, and a texture table's mark is not one. (After cool_fox
    /// 1f1a433's `copied_tabled`.)
    #[test]
    fn a_table_s_buffer_slot_is_left_to_its_draws_for_some_frames() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 1000;
        assert!(!heap.buffer_slot_tabled_lately(3, frame), "never a table's");
        heap.buffer_tabled_in(3, frame);
        heap.buffer_tabled_in(4, frame);
        for at in frame..=frame + u64::from(TABLED_BUFFER_FRAMES) {
            assert!(heap.buffer_slot_tabled_lately(3, at), "frame {at}");
            assert!(heap.buffer_slot_tabled_lately(4, at), "frame {at}");
            assert!(!heap.tabled_lately(3, at), "not the sync's, frame {at}");
        }
        let after = frame + u64::from(TABLED_BUFFER_FRAMES) + 1;
        assert!(!heap.buffer_slot_tabled_lately(3, after));
        assert!(!heap.buffer_slot_tabled_lately(4, after));
        // A texture table's mark is the sync's and the queue's, not the
        // pass's.
        heap.tabled_in(5, true, frame);
        heap.tabled_in(6, false, frame);
        assert!(heap.tabled_lately(5, frame + 1) && heap.tabled_lately(6, frame + 1));
        assert!(!heap.buffer_slot_tabled_lately(5, frame + 1));
        assert!(!heap.buffer_slot_tabled_lately(6, frame + 1));
        // A table again: the frames count from it.
        heap.buffer_tabled_in(3, after);
        assert!(heap.buffer_slot_tabled_lately(3, after + 1));
        assert!(!heap.buffer_slot_tabled_lately(2, after), "another slot");
        // Past the heap's slots: never.
        assert!(!heap.buffer_slot_tabled_lately(8, after));
    }

    /// A slot a draw read through a table leaves the queue; the title's
    /// next write of it waits for its draw, and is due after the grace
    /// frames if none reads it.
    #[test]
    fn slots_read_through_tables_leave_the_texture_queue() {
        let Some(heap) = HostHeap::new(false, 8) else {
            return; // no host GPU
        };
        let frame = 100;
        let slots = |at: u64| -> Vec<u32> { heap.texture_slots_due(at, 8).0.into_iter().map(|d| d.0).collect() };
        let texture = View::Placed(false, [3, 0, 0, 0, 4, 0, 4, 0x1_0001, 28, 1, 0, 0x105, 0, 0], [0; 16]);
        assert!(is_texture_view(&texture));
        heap.queue_texture(2, true, false, frame);
        heap.queue_texture(3, true, false, frame);
        assert_eq!(slots(frame), [3, 2]);
        heap.tabled_in(3, true, frame);
        assert_eq!(slots(frame), [2]);
        // Made in place by the draw that read it: still out.
        heap.queue_texture(3, true, true, frame);
        assert_eq!(slots(frame), [2]);
        // Written again in the next frame: due once the grace frames have
        // passed without a draw that reads it.
        heap.queue_texture(3, true, false, frame + 1);
        assert_eq!(slots(frame + TABLED_GRACE), [2]);
        assert_eq!(slots(frame + 1 + TABLED_GRACE), [3, 2]);
        // The draw comes after all: out again.
        heap.tabled_in(3, true, frame + 1);
        assert_eq!(slots(u64::MAX), [2]);
        // A slot of a table no draw read stays queued; its next write waits.
        heap.tabled_in(2, false, frame + 1);
        assert_eq!(slots(frame + 1), [2]);
        heap.queue_texture(2, true, false, frame + 2);
        assert_eq!(slots(frame + 2), []);
        // Long after the last table, a write is due at once.
        heap.queue_texture(2, true, false, frame + 3 + TABLED_FRAMES);
        assert_eq!(slots(frame + 3 + TABLED_FRAMES), [2]);
    }

    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    /// A view heap ends at the slot shaders clamp view indices to, however
    /// few slots the title asks for, and its null views of every kind and
    /// dimension there come after every slot it hands out and leave the
    /// device alive (with `VICTORY_D3D12_DEBUG` in a diagnostics build, the
    /// debug layer finds no error in them; run it alone then, as other tests
    /// share the device).
    #[test]
    fn view_heaps_end_with_the_null_views_shaders_clamp_to() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let _ = host.log_messages();
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        assert_eq!(host.log_messages(), None);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        let last = victory_shader::emit::view_heap_limit();
        // SAFETY: a live heap.
        let size = unsafe { heap.heap.GetDesc() }.NumDescriptors;
        assert_eq!(size, last + 1);
        let lowest = GuardView::ALL.iter().map(|v| v.slot(last)).min().unwrap();
        assert_eq!(lowest, last + 1 - GUARDS);
        assert!(heap.count + heap.dynamic <= lowest);
        // A heap the title fills to the end leaves the null views out.
        let Some(full) = HostHeap::new(false, last) else {
            return;
        };
        assert_eq!(full.count + full.dynamic, lowest);
    }

    #[test]
    fn null_cbv_does_not_allocate_shadow_at_zero() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        assert!(make(&View::Constant(0, 0), heap.cpu(0)).is_none());
        assert!(make(&View::Constant(0, 256), heap.cpu(0)).is_none());
        assert!(make(&View::Constant(0x1000, 0), heap.cpu(0)).is_none());
        // Making through `make_at` falls back to `null_for` (canonical null CBV)
        // without keeping a bogus VA-0 shadow buffer.
        heap.make_at(heap.count, &View::Constant(0, 0));
        assert!(heap.kept_at(heap.count, |kept| kept.is_none()));
        heap.constant_at(heap.count + 1, 0, 0);
        assert!(heap.kept_at(heap.count + 1, |kept| kept.is_none()));
    }
}

/// The heap sync: a page at once, the early skip, the pass run ahead.
#[cfg(test)]
mod sync_tests;

/// `made` changed a slot at a time by many threads.
#[cfg(test)]
mod made_tests {
    use super::*;

    /// Slots of the heaps these tests make.
    const SLOTS: u32 = 16;

    /// Memory of its own for [`views`]: 1 MiB, left to the process.
    pub(super) fn memory() -> u64 {
        // SAFETY: new memory.
        let at = unsafe {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            VirtualAlloc(
                std::ptr::null(),
                1 << 20,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as u64
        };
        assert_ne!(at, 0);
        at
    }

    /// The views the tests write: two constant buffer views; a buffer
    /// placed in memory, with no host copy (left for a draw, then a null
    /// view under its word); two textures placed at one start in memory at
    /// `at` ([`memory`]), each of which makes the other's slots stale; one
    /// placed further on; and no view at all.
    pub(super) fn views(at: u64) -> Vec<Option<View>> {
        // The buffer and the constant buffer views in title memory that is
        // not there, as in the tests above: a host copy of memory that is
        // would stay for the rest of the tests, next to their own.
        let away = 0x7d_5000_0000;
        let buffer = {
            let address = away + 0x2_0000;
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 64, 12, 0, 0]);
            (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
            View::Placed(
                false,
                [1, 0, 0, 0, 1 << 16, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
                desc,
            )
        };
        vec![
            Some(View::Constant(away, 256)),
            Some(View::Constant(away + 0x1000, 512)),
            Some(buffer),
            Some(texture(at, 64)),
            Some(texture(at, 128)),
            Some(texture(at + 0x1_0000, 64)),
            None,
        ]
    }

    /// A view of a `width`x64 BC7 texture placed at `address`: a 2D view of
    /// its one level (an unknown dimension removes the device).
    pub(super) fn texture(address: u64, width: u32) -> View {
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&[98, 4, 5768, 0, 0, 1, 0, 0, 0, 0]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        View::Placed(
            false,
            [3, 0, 0, 0, width, 0, 64, 0x1_0001, 98, 1, 0, 0x105, 0, 0],
            desc,
        )
    }

    /// Writes `view` as the title's record at `handle` (zeros for none).
    pub(super) fn write(handle: u64, view: &Option<View>) {
        match view {
            Some(view) => super::super::record(handle, view.clone()),
            // SAFETY: the record's first words, in the caller's memory.
            None => unsafe { (handle as *mut [u32; 3]).write_unaligned([0; 3]) },
        }
    }

    pub(super) fn next(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// What slot `index` of `made` holds, once no thread changes it.
    pub(super) fn key(heap: &HostHeap, index: u32) -> u64 {
        heap.claim(index, |_| false).err().expect("not claimed")
    }

    /// The slots in the queue of textures picked by index, in its order.
    fn queued(heap: &HostHeap) -> Vec<(u32, u64)> {
        heap.texture_slots_due(u64::MAX, SLOTS as usize).0
    }

    /// The frame budget's mirrors by kind, as this thread counted them
    /// (`crate::frame_budget::take_seen`): a sampler, the same slot made
    /// null, a placed texture and a buffer view left for a draw, one each;
    /// only the sampler (a table entry needing no host copy) is eligible
    /// for a pre-made descriptor. A mirror of what a slot shows already
    /// counts nothing. A buffer view left for a draw again counts as left
    /// again, of the same record, and so does a constant buffer view left
    /// where another one was, at another address; a placed texture view
    /// left again does not. With the sets of buffer views as sets and as
    /// bits.
    #[test]
    fn sync_counts_kinds() {
        use crate::frame_budget::{LeftAgain, Mirror, Seen, take_seen};
        for bits in [false, true] {
            let (Some(heap), Some(samplers)) = (
                HostHeap::with_ways(false, SLOTS, bits, true),
                HostHeap::new(true, SLOTS),
            ) else {
                return; // no host GPU
            };
            let views = views(memory());
            let mut records = vec![[0u32; 8]; 4];
            let base = records.as_mut_ptr() as u64;
            let at = |slot: u64| base + slot * 32;
            // A valid point sampler (D3D12_SAMPLER_DESC's words): wrap on
            // every axis, one sample, compared never, every level.
            let sampler = [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, f32::MAX.to_bits()];
            let _ = take_seen();
            write(at(0), &Some(View::Sampler(sampler)));
            samplers.mirror(0, at(0), false);
            samplers.mirror(0, at(0), false);
            write(at(0), &None);
            samplers.mirror(0, at(0), false);
            write(at(1), &views[3]);
            heap.mirror(1, at(1), false);
            heap.mirror(1, at(1), false);
            write(at(2), &views[2]);
            heap.mirror(2, at(2), false);
            heap.mirror(2, at(2), false);
            write(at(3), &views[0]);
            heap.mirror(3, at(3), false);
            write(at(3), &views[1]);
            heap.mirror(3, at(3), false);
            let seen: Vec<Seen> = take_seen()
                .into_iter()
                .filter(|s| matches!(s, Seen::Mirror { .. } | Seen::LeftAgain(_)))
                .collect();
            let mirror = |kind| Seen::Mirror {
                kind,
                eligible: false,
            };
            assert_eq!(
                seen,
                vec![
                    Seen::Mirror {
                        kind: Mirror::Sampler,
                        eligible: true
                    },
                    mirror(Mirror::Null),
                    mirror(Mirror::PlacedTexture),
                    mirror(Mirror::PlacedTexture),
                    mirror(Mirror::BufferDeferred),
                    mirror(Mirror::BufferDeferred),
                    Seen::LeftAgain(LeftAgain::Same),
                    mirror(Mirror::BufferDeferred),
                    mirror(Mirror::BufferDeferred),
                    Seen::LeftAgain(LeftAgain::Constant),
                ],
                "bits {bits}"
            );
            drop(records);
        }
    }

    /// One thread's writes, mirrors (of new records and of records already
    /// made, which skip the claim), walks and texture questions over the
    /// slots leave a heap whose sets of buffer views are bits alike after
    /// every step with one whose sets are sets: what each slot is made from,
    /// its word, the sets of buffer views and the queue of textures. (Two
    /// heaps: each takes a whole view heap of the card.)
    #[test]
    fn bits_make_what_the_sets_make() {
        let ways = [false, true];
        let Some(heaps) = ways
            .iter()
            .map(|&bits| HostHeap::with_ways(false, SLOTS, bits, true))
            .collect::<Option<Vec<_>>>()
        else {
            return; // no host GPU
        };
        let views = views(memory());
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let base = records.as_mut_ptr() as u64;
        let mut rng = 0x6a09_e667_f3bc_c908u64;
        let sorted = |mut slots: Vec<u32>| {
            slots.sort_unstable();
            slots
        };
        let quick_before = SKIPPED_MADE.load(std::sync::atomic::Ordering::Relaxed);
        // Slot looks that found it null, made, and with a word other than 0.
        let mut seen = [0u64; 3];
        // (Other tests count stale views too.)
        let stale_before = STALE[0].load(std::sync::atomic::Ordering::Relaxed);
        for step in 0..3_000 {
            let r = next(&mut rng);
            let slot = (r % u64::from(SLOTS)) as u32;
            let at = base + u64::from(slot) * 32;
            match r >> 8 & 15 {
                0 => {
                    let gone: Vec<bool> = heaps
                        .iter()
                        .map(|heap| heap.texture_slot_gone(slot))
                        .collect();
                    assert!(gone.iter().all(|&g| g == gone[0]), "step {step}");
                }
                1 => {
                    let made: Vec<bool> = heaps
                        .iter()
                        .map(|heap| heap.texture_slot_made_from(slot, at))
                        .collect();
                    assert!(made.iter().all(|&m| m == made[0]), "step {step}");
                }
                2 => {
                    for heap in &heaps {
                        heap.mirror_walked(slot, at);
                    }
                }
                // Mirrors of what is there already: they skip the claim.
                3 | 4 => {
                    for heap in &heaps {
                        heap.mirror(slot, at, r & 1 << 20 != 0);
                    }
                }
                op => {
                    write(at, &views[(r >> 16) as usize % views.len()]);
                    for heap in &heaps {
                        heap.mirror(slot, at, op & 1 == 1);
                    }
                }
            }
            let first = &heaps[0];
            for s in 0..SLOTS {
                let made = key(first, s);
                seen[usize::from(made != 0)] += 1;
                seen[2] += u64::from(first.word(s).is_some_and(|word| word != 0));
            }
            for (heap, way) in heaps.iter().zip(ways).skip(1) {
                for s in 0..SLOTS {
                    assert_eq!(
                        key(first, s),
                        key(heap, s),
                        "step {step}, slot {s}, {way:?}"
                    );
                    assert_eq!(
                        first.word(s),
                        heap.word(s),
                        "step {step}, slot {s}, {way:?}"
                    );
                }
                assert_eq!(
                    first.buffer_slots(),
                    heap.buffer_slots(),
                    "step {step}, {way:?}"
                );
                assert_eq!(queued(first), queued(heap), "step {step}, {way:?}");
            }
            if step % 5 == 0 {
                let changes: Vec<Vec<u32>> = heaps
                    .iter()
                    .map(|heap| sorted(heap.buffer_changes()))
                    .collect();
                for (changed, way) in changes.iter().zip(ways) {
                    assert_eq!(&changes[0], changed, "step {step}, {way:?}");
                }
            }
        }
        assert!(
            SKIPPED_MADE.load(std::sync::atomic::Ordering::Relaxed) > quick_before,
            "mirrors done with before the claim"
        );
        // Slots were made, left null and given words along the way.
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
        assert!(
            STALE[0].load(std::sync::atomic::Ordering::Relaxed) > stale_before,
            "stale views"
        );
        assert!(alive());
    }

    /// Whether the host device is still there.
    pub(super) fn alive() -> bool {
        // SAFETY: a plain device query.
        host().is_some_and(|host| unsafe { host.device.GetDeviceRemovedReason() }.is_ok())
    }

    /// The texture pass's question on a slot another thread holds a claim on
    /// that changes nothing waits for the claim and answers as without it,
    /// not "not made" at once: a texture whose slot is made would count as
    /// refused.
    #[test]
    fn a_quick_look_at_a_claimed_slot_waits_for_the_claim() {
        let Some(heap) = HostHeap::with_ways(false, SLOTS, false, true) else {
            return; // no host GPU
        };
        let views = views(memory());
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let at = records.as_mut_ptr() as u64 + 3 * 32;
        write(at, &views[3]);
        heap.mirror(3, at, true);
        let before = heap.texture_slot_made_from(3, at);
        let answer = std::thread::scope(|scope| {
            let claim = heap.claim(3, |_| true).expect("not claimed");
            let asked = scope.spawn(|| heap.texture_slot_made_from(3, at));
            std::thread::sleep(std::time::Duration::from_millis(50));
            assert!(!asked.is_finished(), "waits for the claim");
            drop(claim);
            asked.join().expect("the look")
        });
        assert_eq!(answer, before);
    }

    /// Threads that write, mirror and empty the same slot and slots of
    /// their own at once, with views that make each other's slots stale,
    /// all finish (no two wait for each other), and leave every slot made
    /// from a view with that view's word: a word never says another kind
    /// than the descriptor under it (the null slot crash). Written once
    /// more by one thread, every slot then holds what it holds on a fresh
    /// heap written the same way by one thread alone.
    #[test]
    fn many_threads_leave_each_slot_with_its_own_word() {
        if let Some(heap) = HostHeap::new(false, SLOTS) {
            many_threads_on(heap);
        }
    }

    /// The same with the sets of buffer views as bits
    /// (`VICTORY_SLOT_BITS=1`).
    #[test]
    fn many_threads_leave_each_slot_with_its_own_word_with_bits() {
        if let Some(heap) = HostHeap::with_ways(false, SLOTS, true, true) {
            many_threads_on(heap);
        }
    }

    /// [`many_threads_leave_each_slot_with_its_own_word`] on `heap`.
    fn many_threads_on(heap: HostHeap) {
        const THREADS: u64 = 8;
        let heap = std::sync::Arc::new(heap);
        let views = std::sync::Arc::new(views(memory()));
        let (done, finished) = std::sync::mpsc::channel();
        for t in 0..THREADS {
            let (heap, views, done) = (heap.clone(), views.clone(), done.clone());
            std::thread::spawn(move || {
                // Records of its own: the title's, which one thread writes.
                let mut records = vec![[0u32; 8]; SLOTS as usize];
                let base = records.as_mut_ptr() as u64;
                let mut rng = 0x2545_f491_4f6c_dd1d ^ (t + 1).wrapping_mul(0x9e37_79b9);
                for _ in 0..20_000 {
                    let r = next(&mut rng);
                    // Half of the time slot 0, which every thread writes.
                    let slot = if r & 1 == 0 {
                        0
                    } else {
                        (r >> 1) as u32 % SLOTS
                    };
                    let at = base + u64::from(slot) * 32;
                    match r >> 8 & 15 {
                        0 => {
                            heap.texture_slot_gone(slot);
                        }
                        1 => {
                            heap.texture_slot_made_from(slot, at);
                        }
                        2 => {
                            heap.mirror_walked(slot, at);
                        }
                        3 => {
                            heap.buffer_changes();
                            heap.texture_slots_due(u64::MAX, 4);
                        }
                        op => {
                            write(at, &views[(r >> 16) as usize % views.len()]);
                            heap.mirror(slot, at, op & 1 == 1);
                        }
                    }
                }
                let _ = done.send(());
            });
        }
        // Meanwhile: a slot that shows the same record before and after its
        // word is read has that record's word, or 0 (nothing, while it
        // changes), never another.
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (heap, stop) = (heap.clone(), stop.clone());
            std::thread::spawn(move || {
                use std::sync::atomic::Ordering::{Acquire, Relaxed};
                let (mut looks, mut wrong) = (0u64, Vec::new());
                while !stop.load(Relaxed) {
                    for s in 0..SLOTS {
                        let before = heap.made[s as usize].load(Acquire);
                        let word = heap.word(s);
                        if before == 0
                            || before == CLAIMED
                            || heap.made[s as usize].load(Acquire) != before
                        {
                            continue;
                        }
                        looks += 1;
                        let view = view_at(before as u32, (before >> 32) as u32);
                        if word != Some(0)
                            && word != Some(slot_word(view.as_ref()))
                            && wrong.len() < 8
                        {
                            wrong.push((s, before, word));
                        }
                    }
                }
                (looks, wrong)
            })
        };
        for _ in 0..THREADS {
            finished
                .recv_timeout(std::time::Duration::from_secs(300))
                .expect("every thread finishes: no two wait for each other");
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let (looks, wrong) = reader.join().unwrap();
        assert!(looks > 0);
        assert_eq!(wrong, [], "slots whose word is not their record's");
        for s in 0..SLOTS {
            let key = key(&heap, s);
            if key != 0 {
                let view = view_at(key as u32, (key >> 32) as u32).expect("made from a view");
                assert_eq!(heap.word(s), Some(slot_word(Some(&view))), "slot {s}");
            }
        }
        assert_eq!(spanned_but_not_queued(&heap), [], "slots in the spans");
        // Once more, a write per slot by one thread: constant buffer views
        // and the placed buffer, which leave nothing stale behind.
        let Some(alone) = HostHeap::new(false, SLOTS) else {
            return;
        };
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let base = records.as_mut_ptr() as u64;
        for s in 0..SLOTS {
            let at = base + u64::from(s) * 32;
            write(at, &views[s as usize % 3]);
            heap.mirror(s, at, true);
            alone.mirror(s, at, true);
        }
        for s in 0..SLOTS {
            assert_eq!(key(&heap, s), key(&alone, s), "slot {s}");
            assert_eq!(heap.word(s), alone.word(s), "slot {s}");
        }
        assert_eq!(heap.buffer_slots(), alone.buffer_slots());
        let slots = |heap: &HostHeap| {
            let mut slots: Vec<u32> = queued(heap).into_iter().map(|(slot, _)| slot).collect();
            slots.sort_unstable();
            slots
        };
        assert_eq!(slots(&heap), slots(&alone));
        assert!(alive());
    }

    /// The slots the spans hold a placed texture view for that are not in
    /// the queue of textures picked by index. One after the other, a write
    /// of such a view puts its slot into both, a draw that makes the slot
    /// from it puts it into the queue if the spans take it, and a newer
    /// view that makes it stale takes it out of both; nothing in these
    /// tests takes a slot out of the queue alone.
    fn spanned_but_not_queued(heap: &HostHeap) -> Vec<u32> {
        let spanned: Vec<u32> = {
            let placed = heap.placed.lock().unwrap();
            placed.by_slot.keys().copied().collect()
        };
        let textures = heap.texture_queue();
        let mut missing: Vec<u32> = spanned
            .into_iter()
            .filter(|slot| !textures.slots.contains_key(slot))
            .collect();
        missing.sort_unstable();
        missing
    }

    /// The key of the record at `handle` ([`HostHeap::mirror`]), its tag
    /// not looked at.
    pub(super) fn raw_key(handle: u64) -> u64 {
        // SAFETY: the caller's record.
        let [_, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
        u64::from(generation) << 32 | u64::from(i)
    }

    /// Threads that write placed texture views that make each other's
    /// slots stale, into the same slots at once: a removal of a stale
    /// slot from the queue never lands after a newer write of that slot,
    /// which left it in the spans. (The removal was logged by the thread
    /// that wrote the other slot, after the stale slot's own write had
    /// logged its change to the queue and before it noted its spans: the
    /// slot stayed in the spans, made, and out of the queue, and the
    /// texture a shader picked there by index was never uploaded.)
    ///
    /// The threads write in steps, all at once: each writes a slot of its
    /// own, and the two slots of a pair are written with views of two
    /// textures at one start, each of which makes the other stale. After
    /// every step, as one after the other in either order: the slot
    /// written later is in the spans and in the queue, the other in
    /// neither.
    #[test]
    fn a_stale_slot_leaves_the_queue_only_before_its_newer_write() {
        const THREADS: u32 = 8;
        const STEPS: u32 = 2_000;
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let heap = std::sync::Arc::new(heap);
        let at = memory();
        let step = std::sync::Arc::new(std::sync::Barrier::new(THREADS as usize + 1));
        let stale_before = STALE[0].load(std::sync::atomic::Ordering::Relaxed);
        let threads: Vec<_> = (0..THREADS)
            .map(|t| {
                let (heap, step) = (heap.clone(), step.clone());
                // Pair t / 2, at a start of its own.
                let view = texture(at + u64::from(t / 2) * 0x1_0000, 64 << (t & 1));
                std::thread::spawn(move || {
                    let mut record = [0u32; 8];
                    let handle = record.as_mut_ptr() as u64;
                    for _ in 0..STEPS {
                        step.wait();
                        // A record of its own each time: a newer view.
                        write(handle, &Some(view.clone()));
                        heap.mirror(t, handle, false);
                        step.wait();
                    }
                })
            })
            .collect();
        let mut wrong = Vec::new();
        for n in 0..STEPS {
            step.wait();
            step.wait();
            let missing = spanned_but_not_queued(&heap);
            if !missing.is_empty() && wrong.len() < 8 {
                wrong.push((n, missing));
            }
            // Of each pair, one is in the spans.
            let placed = heap.placed.lock().unwrap();
            for pair in 0..THREADS / 2 {
                let held = (0..2).filter(|&i| placed.by_slot.contains_key(&(pair * 2 + i)));
                assert_eq!(held.count(), 1, "step {n}, pair {pair}");
            }
        }
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(
            wrong,
            [],
            "steps that left slots in the spans and out of the queue"
        );
        assert!(
            STALE[0].load(std::sync::atomic::Ordering::Relaxed) > stale_before,
            "stale views"
        );
        // Made slots show a view with its own word.
        for s in 0..SLOTS {
            let key = key(&heap, s);
            if key != 0 {
                let view = view_at(key as u32, (key >> 32) as u32).expect("made from a view");
                assert_eq!(heap.word(s), Some(slot_word(Some(&view))), "slot {s}");
            }
        }
        assert!(alive());
    }

    /// Threads that make the same slot from records of their own at once:
    /// the slot ends up made from the last record one of them made it
    /// from, with that record's word; no claim is published over the key
    /// of a write that came after it.
    #[test]
    fn a_slot_shows_the_last_record_it_was_made_from() {
        const THREADS: u64 = 8;
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let heap = std::sync::Arc::new(heap);
        let views = std::sync::Arc::new(views(memory()));
        let lasts: Vec<u64> = (0..THREADS)
            .map(|t| {
                let (heap, views) = (heap.clone(), views.clone());
                std::thread::spawn(move || {
                    let mut record = [0u32; 8];
                    let at = record.as_mut_ptr() as u64;
                    let mut rng = 0xbb67_ae85_84ca_a73b ^ (t + 1).wrapping_mul(0x9e37_79b9);
                    for _ in 0..20_000 {
                        // The constant buffer views and the placed buffer:
                        // each is made, none waits for anything.
                        write(at, &views[(next(&mut rng) % 3) as usize]);
                        heap.mirror(0, at, true);
                    }
                    raw_key(at)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        let shown = key(&heap, 0);
        assert!(lasts.contains(&shown), "{shown:x} of {lasts:x?}");
        let view = view_at(shown as u32, (shown >> 32) as u32).expect("made from a view");
        assert_eq!(heap.word(0), Some(slot_word(Some(&view))));
        assert!(alive());
    }

    /// Threads that mirror one record into one slot at once, each finding
    /// what the view is made from before it claims the slot: the slot is
    /// written once, from the record, and shows the record's host copy with
    /// its word.
    #[test]
    fn a_view_resolved_before_the_claim_is_made_once() {
        const THREADS: usize = 8;
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let views = views(memory());
        let Some(View::Constant(address, size)) = views[0] else {
            unreachable!("views()[0] is a constant buffer view")
        };
        let mut record = [0u32; 8];
        let at = record.as_mut_ptr() as u64;
        write(at, &views[0]);
        let start = std::sync::Barrier::new(THREADS);
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    start.wait();
                    heap.mirror(0, at, true);
                    assert_eq!(claims_held(), 0, "no claim left held");
                });
            }
        });
        assert_eq!(heap.writes.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(key(&heap, 0), raw_key(at));
        let copy = crate::shadow::find(address, u64::from(size)).expect("its host copy");
        assert_eq!(
            heap.kept_raw(0),
            Some(windows::core::Interface::as_raw(&copy.buffer)),
            "the slot keeps the record's host copy"
        );
        assert_eq!(heap.word(0), Some(slot_word(views[0].as_ref())));
        assert!(alive());
    }

    thread_local! {
        /// The key the test of a slot changed between the look and the claim
        /// puts back at the look.
        static SHOWN: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// A slot that showed the record at the look before the claim, and
    /// another by the claim: the claim is let go, the view resolved, and
    /// the slot made from the record after all.
    #[test]
    fn a_slot_changed_between_the_look_and_the_claim_is_resolved_first() {
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let views = views(memory());
        let mut record = [0u32; 8];
        let at = record.as_mut_ptr() as u64;
        write(at, &views[0]);
        heap.mirror(0, at, true);
        assert_eq!(key(&heap, 0), raw_key(at));
        let again = RESOLVED_AGAIN.load(std::sync::atomic::Ordering::Relaxed);
        // Past the first look (a slot that shows its record is done with
        // there): the slot shows another record at it, this one at the next
        // look, and another again by the claim.
        let shown = key(&heap, 0);
        heap.claim(0, |_| true).expect("not claimed").value = 0;
        SHOWN.set(shown);
        BEFORE_LOOK.set(Some(|made| {
            made.store(SHOWN.get(), std::sync::atomic::Ordering::Release)
        }));
        BEFORE_CLAIM.set(Some(|made| {
            made.store(0, std::sync::atomic::Ordering::Release)
        }));
        heap.mirror(0, at, true);
        assert!(RESOLVED_AGAIN.load(std::sync::atomic::Ordering::Relaxed) > again);
        assert_eq!(heap.writes.load(std::sync::atomic::Ordering::Relaxed), 2);
        assert_eq!(key(&heap, 0), raw_key(at));
        assert_eq!(heap.word(0), Some(slot_word(views[0].as_ref())));
        assert_eq!(claims_held(), 0);
        assert!(alive());
    }

    /// A texture view resolved before its texture's levels changed
    /// (`levels_changed` between the resolve and the claim) is resolved
    /// again under a new look, and the slot written once, from the view as
    /// it is after the change.
    #[test]
    fn a_texture_view_resolved_before_its_levels_changed_is_resolved_again() {
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let at = memory();
        let mut record = [0u32; 8];
        let handle = record.as_mut_ptr() as u64;
        write(handle, &Some(texture(at, 64)));
        heap.mirror(3, handle, false);
        let again = RESOLVED_AGAIN.load(std::sync::atomic::Ordering::Relaxed);
        BEFORE_CLAIM.set(Some(|_| super::levels_changed(0)));
        heap.mirror_marking(3, handle, true, false, hold::WALK);
        assert!(BEFORE_CLAIM.take().is_none(), "the hook ran");
        assert!(heap.kept_raw(3).is_some(), "a host texture for the view");
        assert!(RESOLVED_AGAIN.load(std::sync::atomic::Ordering::Relaxed) > again);
        assert_eq!(heap.writes.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(key(&heap, 3), raw_key(handle));
        assert!(alive());
    }

    /// A level change after the claim's look and before the slot enters the
    /// spans (where the remade pass looks): the view is written as resolved,
    /// and its start told again, so the next pass makes the slot again.
    #[test]
    fn a_level_change_before_the_slot_is_in_the_spans_is_told_again() {
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let at = memory();
        let mut record = [0u32; 8];
        let handle = record.as_mut_ptr() as u64;
        // Not in the spans before: a draw's first write of the slot.
        write(handle, &Some(texture(at, 64)));
        let reshown = RESHOWN_AGAIN.load(std::sync::atomic::Ordering::Relaxed);
        BEFORE_NOTE.set(Some(|| super::levels_changed(0)));
        heap.mirror_marking(3, handle, true, false, hold::WALK);
        assert!(BEFORE_NOTE.take().is_none(), "the hook ran");
        assert!(heap.kept_raw(3).is_some(), "a host texture for the view");
        assert!(RESHOWN_AGAIN.load(std::sync::atomic::Ordering::Relaxed) > reshown);
        assert_eq!(key(&heap, 3), raw_key(handle));
        // The next pass finds the slot in the spans and makes it again.
        assert_eq!(heap.texture_views_remade(), 1);
        assert_eq!(key(&heap, 3), raw_key(handle));
        assert!(alive());
    }

    /// A slot made from another record between the pass's look and its
    /// claim is left as that record's maker made it.
    #[test]
    fn views_remade_leave_a_slot_made_anew_meanwhile() {
        /// A key no record of the tests has.
        const OTHER: u64 = 7 << 32 | 0x7777;
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let at = memory();
        let mut record = [0u32; 8];
        let handle = record.as_mut_ptr() as u64;
        write(handle, &Some(texture(at, 64)));
        heap.mirror(3, handle, false);
        heap.mirror_marking(3, handle, true, false, hold::WALK);
        assert!(heap.kept_raw(3).is_some(), "a host texture for the view");
        super::levels_changed(at);
        BEFORE_REMADE.set(Some(|heap| {
            heap.made[3].store(OTHER, std::sync::atomic::Ordering::Release)
        }));
        assert_eq!(heap.texture_views_remade(), 0, "the slot is another's now");
        assert_eq!(key(&heap, 3), OTHER);
        assert!(alive());
    }

    /// Levels on demand: the slots of a placed texture whose first level
    /// with memory changed are written again from the record they were
    /// made from, found before their claims, and keep that record; a slot
    /// written with another view since is left with that view's record.
    #[test]
    fn views_remade_keep_their_record() {
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let at = memory();
        let views = views(at);
        let mut records = vec![[0u32; 8]; 2];
        let (one, two) = (
            records.as_mut_ptr() as u64,
            records.as_mut_ptr() as u64 + 32,
        );
        // Two slots made from views of the placed texture at `at`.
        let view = texture(at, 64);
        for (slot, handle) in [(3, one), (4, two)] {
            write(handle, &Some(view.clone()));
            heap.mirror(slot, handle, false);
            heap.mirror_marking(slot, handle, true, false, hold::WALK);
            assert_eq!(key(&heap, slot), raw_key(handle), "slot {slot} made");
        }
        assert!(heap.kept_raw(3).is_some(), "a host texture for the view");
        // Slot 4 written with a constant buffer view meanwhile.
        write(two, &views[0]);
        heap.mirror(4, two, true);
        super::levels_changed(at);
        let remade = heap.texture_views_remade();
        assert_eq!(remade, 1, "slot 3 made again");
        assert_eq!(key(&heap, 3), raw_key(one));
        assert!(heap.kept_raw(3).is_some());
        assert_eq!(key(&heap, 4), raw_key(two), "slot 4 keeps its new record");
        assert!(alive());
    }

    /// Slots written while the heap buffer pass takes the changes over and
    /// over are each in some take, or in the one after the writes: the
    /// walk sees every slot written while it ran, in its next pass at the
    /// latest. The set of buffer views ends as the last writes left it.
    #[test]
    fn the_walk_sees_every_slot_written_while_it_ran() {
        const THREADS: u32 = 8;
        // Two slots of its own per thread.
        const _: () = assert!(THREADS * 2 == SLOTS);
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        let heap = std::sync::Arc::new(heap);
        let views = std::sync::Arc::new(views(memory()));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let walk = {
            let (heap, stop) = (heap.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut seen = std::collections::BTreeSet::new();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    seen.extend(heap.buffer_changes());
                }
                seen
            })
        };
        // Per thread: the slots it wrote a buffer view into, and what each
        // of its slots held last (a buffer view or not).
        let writes: Vec<(Vec<u32>, [bool; 2])> = (0..THREADS)
            .map(|t| {
                let (heap, views) = (heap.clone(), views.clone());
                std::thread::spawn(move || {
                    let mut records = [[0u32; 8]; 2];
                    let base = records.as_mut_ptr() as u64;
                    let mut rng =
                        0x3c6e_f372_fe94_f82b ^ u64::from(t + 1).wrapping_mul(0x9e37_79b9);
                    let (mut buffers, mut last) = (Vec::new(), [false; 2]);
                    for _ in 0..10_000 {
                        let r = next(&mut rng);
                        let own = (r & 1) as usize;
                        let slot = t * 2 + own as u32;
                        let at = base + own as u64 * 32;
                        let buffer = r >> 8 & 1 == 1;
                        write(at, &views[if buffer { 2 } else { 0 }]);
                        heap.mirror(slot, at, false);
                        if buffer {
                            buffers.push(slot);
                        }
                        last[own] = buffer;
                    }
                    (buffers, last)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut seen = walk.join().unwrap();
        seen.extend(heap.buffer_changes());
        let mut holding = Vec::new();
        for (t, (buffers, last)) in writes.iter().enumerate() {
            for slot in buffers {
                assert!(seen.contains(slot), "slot {slot}");
            }
            for (own, &buffer) in last.iter().enumerate() {
                if buffer {
                    holding.push(t as u32 * 2 + own as u32);
                }
            }
        }
        assert_eq!(heap.buffer_slots(), holding);
    }

    /// A heap whose queue and sets no pass reads keeps its logs short: the
    /// writer that finds one long applies it.
    #[test]
    fn logs_no_pass_reads_stay_short() {
        let Some(heap) = HostHeap::new(false, SLOTS) else {
            return; // no host GPU
        };
        fn longest<T>(logs: &[Mutex<Vec<T>>]) -> usize {
            logs.iter()
                .map(|log| log.lock().unwrap().len())
                .max()
                .unwrap_or(0)
        }
        let views = views(memory());
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let base = records.as_mut_ptr() as u64;
        let (mut most, mut last) = ((0, 0), [false; SLOTS as usize]);
        let mut rng = 0xa54f_f53a_5f1d_36f1u64;
        for n in 0..3 * LOG_MOST {
            let r = next(&mut rng);
            let slot = (r % u64::from(SLOTS)) as u32;
            let at = base + u64::from(slot) * 32;
            // A buffer view or a constant buffer view: each write logs a
            // change to the queue and one to the sets (as bits, changes
            // the sets at once).
            let buffer = r >> 8 & 1 == 1;
            write(at, &views[if buffer { 2 } else { 0 }]);
            heap.mirror(slot, at, false);
            last[slot as usize] = buffer;
            if n % 1024 == 0 {
                most.0 = most.0.max(longest(&heap.log.textures));
                most.1 = most.1.max(longest(&heap.log.buffers));
            }
        }
        assert!(most.0 <= LOG_MOST && most.1 <= LOG_MOST, "{most:?}");
        if heap.bits.is_some() {
            // Sets as bits (`VICTORY_SLOT_BITS=1`): each write changes its
            // bits under the slot's claim and logs no buffer change at all.
            assert!(most.0 > 0 && most.1 == 0, "{most:?}");
        } else {
            assert!(most.0 > 0 && most.1 > 0, "{most:?}");
        }
        // What was applied on the way is what the sets hold.
        let holding: Vec<u32> = (0..SLOTS).filter(|&slot| last[slot as usize]).collect();
        assert_eq!(heap.buffer_slots(), holding);
    }

    /// Changes logged by many threads come out in the order they were
    /// logged: a slot's last change wins, as when each was applied at once
    /// under the queue's lock, also when another thread takes the logs
    /// while they are written. Each slot's changes are made under a lock of
    /// the slot's own, as under its claim.
    #[test]
    fn logged_changes_come_out_in_their_order() {
        const THREADS: u64 = 8;
        let log = NoteLog::default();
        let queue = Mutex::new((
            TextureQueue::default(),
            std::collections::BTreeSet::new(),
            0usize,
        ));
        let claims: Vec<Mutex<Option<(bool, bool)>>> =
            (0..SLOTS).map(|_| Mutex::new(None)).collect();
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let (log, queue, claims) = (&log, &queue, &claims);
                s.spawn(move || {
                    let mut rng = 0x5851_f42d_4c95_7f2d ^ (t + 1).wrapping_mul(0x9e37_79b9);
                    for _ in 0..20_000 {
                        let r = next(&mut rng);
                        if r & 63 == 0 {
                            let mut queue = queue.lock().unwrap();
                            for note in log.take_textures() {
                                note.apply(&mut queue.0);
                            }
                            for note in log.take_buffers() {
                                if note.apply(&mut queue.1) {
                                    queue.2 += 1;
                                }
                            }
                            continue;
                        }
                        let index = (r >> 8) as u32 % SLOTS;
                        let (texture, buffer) = (r >> 16 & 1 == 1, r >> 17 & 1 == 1);
                        let mut last = claims[index as usize].lock().unwrap();
                        let note = if texture {
                            TextureNote::Written {
                                index,
                                frame: 1,
                                due: 0,
                                seen: false,
                            }
                        } else {
                            TextureNote::Remove(index)
                        };
                        log.textures(&[note]);
                        log.buffers(BufferNote {
                            index,
                            buffer,
                            mark: true,
                        });
                        *last = Some((texture, buffer));
                    }
                });
            }
        });
        let mut queue = queue.into_inner().unwrap();
        for note in log.take_textures() {
            note.apply(&mut queue.0);
        }
        for note in log.take_buffers() {
            note.apply(&mut queue.1);
        }
        assert!(log.take_textures().is_empty() && log.take_buffers().is_empty());
        for (index, last) in claims.into_iter().enumerate() {
            let Some((texture, buffer)) = last.into_inner().unwrap() else {
                continue;
            };
            let index = index as u32;
            assert_eq!(queue.0.slots.contains_key(&index), texture, "slot {index}");
            assert_eq!(queue.1.contains(&index), buffer, "slot {index}");
        }
    }
}

#[cfg(test)]
mod own_copy_tests {
    use super::own_bytes_if;

    /// A buffer view gets a copy of its own bytes only when it has no host
    /// view on its buffer's copy: off that copy's elements, and not a
    /// structured view that can be a raw view of the same bytes.
    #[test]
    fn a_view_off_its_copys_elements_gets_its_own_bytes() {
        let base = 0xd_9000_0000u64;
        // On an element: the buffer's copy (the two cases of every run's
        // log are the ones below).
        assert_eq!(own_bytes_if(true, 24, base, 3, 10, 12, 0, false), None);
        assert_eq!(own_bytes_if(true, 0, base, 0, 10, 12, 0, false), None);
        // A typed view of 12-byte elements, 5,222,240 bytes into its copy
        // (8 past an element): its own bytes, from its first element.
        assert_eq!(own_bytes_if(true, 5_222_240, base, 3, 10, 12, 0, false), Some((base + 36, 120)));
        // A structured view of 32-byte elements 8 bytes in: not on 16, so
        // no raw view of the copy either.
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 32, 32, true), Some((base + 64, 128)));
        // The same 16 bytes in: off its elements, but a raw view of the
        // copy's bytes does (as before).
        assert_eq!(own_bytes_if(true, 16, base, 2, 4, 32, 32, true), None);
        // No elements, no size, or the switch off: as before.
        assert_eq!(own_bytes_if(true, 8, base, 2, 0, 32, 32, true), None);
        assert_eq!(own_bytes_if(true, 8, base, 2, 4, 0, 0, false), None);
        assert_eq!(own_bytes_if(false, 8, base, 2, 4, 32, 32, true), None);
    }
}
