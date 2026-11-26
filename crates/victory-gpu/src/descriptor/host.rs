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
    made: Box<[std::sync::atomic::AtomicU64]>,
    /// Whether `made` is changed a slot at a time, with no lock over the
    /// heap ([`made_atomic`]); otherwise under `made_lock`, as before.
    atomic: bool,
    /// The heap-wide lock over `made` when it is not changed a slot at a
    /// time: held across every look at and change of a slot, as the lock
    /// that held `made` itself was.
    made_lock: Mutex<()>,
    /// While the profiler records: who took `made_lock` last, its site
    /// ([`hold`]) above the thread id, for the `made lock wait` of whoever
    /// finds it held. Written only while recording ([`HostHeap::hold`]).
    holder: std::sync::atomic::AtomicU64,
    /// With [`HostHeap::atomic`]: the changes [`HostHeap::note_buffer`]
    /// makes to `textures`, `buffers` and `changed`, in the order they were
    /// made, until whatever reads those takes them ([`NoteLog`]).
    log: NoteLog,
    /// A fingerprint of the record each slot held when the pass over
    /// records copied in as bytes last looked at it
    /// ([`HostHeap::mirror_copied`]; 0: never looked at). Made on first use.
    copied: Mutex<Vec<u64>>,
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
    /// The title memory each placed texture view of this heap describes,
    /// by slot and by address ([`Spans`]): which slots a newer view of
    /// other memory there makes stale, and which to look at again when the
    /// runtime maps, unmaps or writes memory ([`remapped`]).
    placed: std::sync::Arc<Mutex<Spans>>,
    /// With [`HostHeap::atomic`], per title slot: [`IN_SPANS`] if its last
    /// write left it in `placed` (it may have left as a stale one since),
    /// and above that bit a count of its writes that looked at `placed`.
    /// Changed under the slot's claim and the spans' lock. A stale slot
    /// is made null only if no write of it looked at the spans since it
    /// went stale ([`HostHeap::unmake_stale`]). Empty otherwise.
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

/// The first level with memory of the placed texture at `start` changed:
/// its slots are made again at the next texture pass.
pub(crate) fn levels_changed(start: u64) {
    RESHOWN.lock().unwrap_or_else(|e| e.into_inner()).push(start);
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
        Self::with_made(sampler, count, made_atomic())
    }

    /// [`HostHeap::new`]; `atomic`: `made` is changed a slot at a time
    /// ([`made_atomic`]).
    fn with_made(sampler: bool, count: u32, atomic: bool) -> Option<Self> {
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
            atomic,
            made_lock: Mutex::new(()),
            holder: std::sync::atomic::AtomicU64::new(0),
            log: NoteLog::default(),
            copied: Mutex::new(Vec::new()),
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
            textures: Mutex::default(),
            tabled: (0..if sampler { 0 } else { count })
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
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
            spanned: (0..if atomic && !sampler { count } else { 0 })
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

/// For the profiler: where a heap's `made` lock is taken, as the `site` of
/// `made lock wait` (the waiter's) and `holder_site` (the holder's), and
/// of `made long hold`.
pub(crate) mod hold {
    /// A view the title wrote or copied (`mirror`, not for a draw).
    pub(crate) const VIEW: u64 = 1;
    /// A draw's bind of a table (`mirror` for a draw).
    pub(crate) const TABLE: u64 = 2;
    /// The walk of the heap buffer pass (`mirror_walked`).
    pub(crate) const WALK: u64 = 3;
    /// `sync_heaps` before a submission (`mirror_copied`).
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

/// A hold of a heap's `made_lock` while the profiler records, from just
/// after it is taken until this is dropped (in [`MadeLocked`], before its
/// guard, so dropped first): a `made long hold` event if it lasted 200 µs
/// or more, args `heap`, `site` ([`hold`]) and `created` (host textures
/// made or tried meanwhile, `crate::resource`'s `host texture make`).
/// Nothing while the profiler does not record.
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
    /// `made_lock` was just taken at `site` ([`hold`]): while the profiler
    /// records, says so to waiters and times the hold.
    #[inline]
    fn hold(&self, site: u64) -> MadeHold {
        let start = crate::profile::stamp();
        let created = if start != 0 {
            self.holder.store(
                site << 32 | u64::from(crate::profile::thread_id()),
                std::sync::atomic::Ordering::Relaxed,
            );
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

    /// Takes the heap-wide lock over `made`, unless it is changed a slot at
    /// a time ([`HostHeap::atomic`]): `None` then. `site` ([`hold`]): a wait
    /// for it is timed only when another thread holds it and the profiler
    /// records, and on the profile only when long (`made lock wait`, with
    /// who held it when the wait began), and the hold is timed
    /// ([`MadeHold`]); `None`, neither (a look for tests and diagnostics).
    fn lock_made(&self, site: Option<u64>) -> Option<MadeLocked<'_>> {
        if self.atomic {
            return None;
        }
        let guard = match self.made_lock.try_lock() {
            Ok(made) => made,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => {
                let waited = site.map_or(0, |_| crate::profile::stamp());
                // (Who took it last: the holder now, unless it has only
                // just been taken and not said so yet.)
                let holder = if waited != 0 {
                    self.holder.load(std::sync::atomic::Ordering::Relaxed)
                } else {
                    0
                };
                let made = self.made_lock.lock().unwrap_or_else(|e| e.into_inner());
                crate::profile::since(
                    "made lock wait",
                    waited,
                    50,
                    &[
                        ("heap", self.id),
                        ("site", site.unwrap_or(0)),
                        ("holder", holder & u64::from(u32::MAX)),
                        ("holder_site", holder >> 32),
                    ],
                );
                made
            }
        };
        Some(MadeLocked {
            _hold: site.map(|site| self.hold(site)),
            _guard: guard,
        })
    }

    /// For the profile's sample (`VICTORY_PROFILE_SAMPLES`): the keys of
    /// the views the made slots were made from (0 for a slot not made), as
    /// read under `made`'s lock (with [`HostHeap::atomic`], a slot at a
    /// time, each once no thread changes it).
    pub(crate) fn made_keys(&self) -> Vec<u64> {
        let _made = self.lock_made(Some(hold::SAMPLE));
        (0..self.count)
            .map(|index| self.claim(index, |_| false).err().unwrap_or(0))
            .collect()
    }
}

/// The heap-wide lock over a heap's `made` ([`HostHeap::lock_made`]), with
/// the hold's timer, which is dropped first.
struct MadeLocked<'a> {
    _hold: Option<MadeHold>,
    _guard: std::sync::MutexGuard<'a, ()>,
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
            let _made = self.lock_made(Some(hold::SWEEP));
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
        slots.retain(|slot| textures.slots.get(slot).is_none_or(|w| w.mapped > 0 || w.checked.2 != 0));
        let Some(after) = due else {
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
        let _made = self.lock_made(Some(hold::MADE_FROM));
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
        let starts = std::mem::take(&mut *RESHOWN.lock().unwrap_or_else(|e| e.into_inner()));
        if starts.is_empty() {
            return 0;
        }
        let mut span = crate::profile::span_args("views remade", &[("heap", self.id)]);
        let slots: Vec<u32> = {
            let placed = self.placed.lock().unwrap_or_else(|e| e.into_inner());
            starts.iter().flat_map(|&start| placed.slots_at(start)).collect()
        };
        if let Some(span) = span.as_mut() {
            span.arg("slots", slots.len() as u64);
        }
        let _made = self.lock_made(Some(hold::REMADE));
        let mut remade = 0;
        for slot in slots {
            let Ok(claim) = self.claim(slot, |key| key != 0) else {
                continue;
            };
            let key = claim.old;
            let Some(view) = view_at(key as u32, (key >> 32) as u32) else {
                continue;
            };
            if let Some(kept) = make(&view, self.cpu(slot)) {
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
    pub(crate) fn texture_slot_gone(&self, index: u32) -> bool {
        let _made = self.lock_made(Some(hold::GONE));
        let Ok(mut claim) = self.claim(index, |key| key != 0) else {
            return false;
        };
        self.unmake(&mut claim, index);
        STALE[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
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
    /// [`HostHeap::spanned`] counted of it then). Under the heap-wide lock,
    /// as the write found them. A slot at a time otherwise, after the
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
    /// slot: applied now under the heap-wide lock, logged otherwise
    /// ([`NoteLog`]). A writer that finds the log long applies it itself,
    /// if the queue is free: the texture pass may leave a heap alone for a
    /// long time.
    fn note_textures(&self, notes: &[TextureNote]) {
        if self.atomic {
            if self.log.textures(notes) {
                drop(self.texture_queue_if_free());
            }
        } else {
            let mut textures = self.texture_queue();
            for &note in notes {
                note.apply(&mut textures);
            }
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
        self.buffer_set().iter().copied().collect()
    }

    /// The title slots whose buffer view was written, or went, since the
    /// last call.
    pub(crate) fn buffer_changes(&self) -> Vec<u32> {
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
        if !self.atomic {
            self.queue_texture(index, texture, seen, frame);
        }
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
        // A slot a time ([`HostHeap::atomic`]): a write of anything but a
        // placed texture view into a slot no such view was noted for has
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
            if self.atomic {
                self.queue_texture(index, texture, seen, frame);
            }
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
            if self.atomic {
                self.remove_stale(&stale);
            }
            stale
        };
        if !self.atomic {
            self.remove_stale(&stale);
        }
        let note = BufferNote {
            index,
            buffer: view.and_then(view_stride).is_some(),
            mark,
        };
        if self.atomic {
            // Applied by the writer once the log is long, as the queue's
            // (`note_textures`): the heap buffer pass runs only for lists
            // whose shaders pick buffers from the heap.
            if self.log.buffers(note) {
                drop(self.buffer_set_if_free());
            }
        } else {
            let mut buffers = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
            if note.apply(&mut buffers) {
                self.changed
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(index);
            }
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

    /// For a slot on a page of the title's heap that changed with no call
    /// (`super::sync_heaps`: the title copies records into its heap as
    /// bytes): [`HostHeap::mirror`], if the `bytes` bytes of the record at
    /// `handle` are not the ones this last saw there. True if it did.
    ///
    /// The page says only that something on it was written, and `mirror`
    /// by itself would take every buffer or placed-texture record on it for
    /// newly written each time (it leaves such a slot unmade until a draw
    /// or a pass uses it): the texture queue's entry for the slot would
    /// start again, and the buffer pass would look at 2,048 slots for one
    /// that changed.
    /// `record`: the record's bytes as the caller read them (through the
    /// kernel, which fails on a page the title unmapped meanwhile where a
    /// read in place would fault: the title unmaps heap pages too).
    pub(crate) fn mirror_copied(&self, index: u32, handle: u64, record: &[u8]) -> bool {
        if index >= self.count {
            return false;
        }
        // A slot draws read through a table lately is left to its draws,
        // as the passes over slots picked by index leave it
        // (`texture_slots_remapped`, `slots_written_under`): the title
        // writes its per-draw tables into the same heap every frame, and
        // each draw makes what its table shows in a slot of its own. Those
        // were most of the 26,000 to 43,000 records a frame this pass
        // mirrored in run 200 and in the play session after it, each then
        // made and brought up to date once more by the pass over buffer
        // views (some 50 ms a frame). `VICTORY_HEAP_SYNC_TABLES=1`: they
        // are mirrored too, as in run 200.
        if !sync_tables() && self.tabled_lately(index, crate::frame::current()) {
            COPIED_TABLED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
        let n = record.len();
        // Never 0, which is "never looked at". (A multiply-and-fold over
        // the record's words: BLAKE3 here was most of the 35 to 42 ms a
        // frame this pass took over some 250,000 to 300,000 records, run
        // 200.)
        let print = record_print(record) | 1;
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if copied.is_empty() {
                copied.resize(self.count as usize, 0);
            }
            let Some(last) = copied.get_mut(index as usize) else {
                return false;
            };
            // A slot never looked at that holds nothing: as it was.
            if *last == print || (*last == 0 && record[..n].iter().all(|&b| b == 0)) {
                return false;
            }
            *last = print;
        }
        self.mirror_marking(index, handle, false, true, hold::SYNC);
        true
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
        {
            let _made = self.lock_made(None);
            for &slot in &slots {
                if let Ok(mut claim) = self.claim(slot, |made| made != 0) {
                    claim.value = 0;
                }
            }
        }
        crate::pool::slots_made_anew(slots.len() as u64);
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
        // A wait timed only when another thread holds it, and on the
        // profile only when long (the walk of the heap buffer pass and
        // `sync_heaps` before a submission both mirror many slots).
        let made = self.lock_made(Some(site));
        let view = (tag == VIEW_TAG).then(|| view_at(i, generation)).flatten();
        let Some(view) = view else {
            // Not a view of ours (never written, or of a resource the
            // title released): nothing to read.
            let Ok(mut claim) = self.claim(index, |made| made != 0 || for_draw) else {
                return None;
            };
            if claim.old != 0 {
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
        // The slot shows this record already: nothing to change (the most
        // common case by far, a draw's table entry).
        let Ok(mut claim) = self.claim(index, |made| made != key) else {
            return Some(view);
        };
        // On the profile: slots changed against the distinct records they
        // changed to, by who mirrored them.
        crate::profile::record_changed(site, key);
        // Nothing to shaders while the slot changes (`write_word`).
        self.write_word(index, 0);
        // Views of other textures in the memory this one describes are
        // stale: their host slots read nothing from here on, and let go of
        // the host copies they held. Under the heap-wide lock now, as
        // before; otherwise once this slot is made (`unmake_stale`).
        let stale: Vec<(u32, u32)> = self
            .note_buffer(index, Some(&view), for_draw, mark)
            .into_iter()
            .filter(|&(stale, _)| stale != index)
            .collect();
        if made.is_some() {
            self.unmake_stale(&stale);
        }
        // A view of a texture object that has no host copy yet waits for
        // it, as buffer views wait for a draw: the slot is made when the
        // texture is first used (`super::make_waiting`), or here when a
        // draw reads the slot through a table.
        let waiting = if for_draw { None } else { view.waiting_texture() };
        if let Some(texture) = waiting {
            super::wait_for(texture, handle);
        }
        let later = !for_draw && (view.is_buffer() || waiting.is_some());
        // Another view of the placed texture the slot shows already (a
        // title that streams a texture in writes its view again with more
        // mips each time some arrive): the slot goes on showing the old
        // view until a draw or the texture pass makes the new one and
        // uploads what it shows. Made null here, the texture read as
        // nothing for as long as it waited in the queue, every time it got
        // sharper.
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
                let kept = make(&view, self.cpu(index));
                crate::profile::part_end(making);
                kept
            };
            if kept.is_none() && !self.sampler {
                null_for(&view, self.cpu(index));
                if !later {
                    self.refused_for_now(&mut value, index, handle, &view);
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
        }
        drop(claim);
        if made.is_none() {
            self.unmake_stale(&stale);
        }
        Some(view)
    }

    /// Claims slot `index` of `made` for this thread, if `wanted` says so
    /// of what it holds: until the claim is dropped, the slot holds
    /// [`CLAIMED`] and no other thread looks at or changes the host slot,
    /// its word or what it keeps (it waits). Dropped, the slot holds the
    /// claim's `value`, which starts as what it held. `Err`: what it holds,
    /// not wanted.
    ///
    /// Under the heap-wide lock ([`HostHeap::lock_made`]) a claim never
    /// waits. Without it a thread waits for a claim only while it holds
    /// none itself, so claims cannot wait for each other in a ring.
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
                break Ok(Claimed {
                    slot,
                    old: now,
                    value: now,
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
        let _made = self.lock_made(None);
        self.claim(index, |_| false).is_err_and(|key| key != 0)
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
    /// `made`: what the claimed slot will hold ([`Claimed::value`]).
    fn refused_for_now(&self, made: &mut u64, index: u32, handle: u64, view: &View) {
        /// Frames until the pass looks at such a slot again: the try
        /// itself waits a second or more (`crate::resource::refused_after`).
        const RETRY_FRAMES: u64 = 16;
        let placed = placed_texture(view);
        let Some(texture) = placed.as_ref().unwrap_or(view).waiting_texture() else {
            return;
        };
        *made = 0;
        // (`make_waiting` makes only slots that view the texture object.)
        if placed.is_none() {
            super::wait_for(texture, handle);
        }
        let frame = crate::frame::current();
        if self.tabled_lately(index, frame) {
            return;
        }
        let Some(spanned) = self.spanned.get(index as usize) else {
            let mut textures = self.texture_queue();
            textures.changed(index, frame, frame + RETRY_FRAMES);
            return;
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
        let key = {
            let _made = self.lock_made(None);
            self.claim(index, |_| false).err()?
        };
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

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`] finds the host copy by. [`buffer_range`] is the part of it the
/// view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`] finds it by the
/// whole buffer the view is of ([`buffer_extent`]). Looked up by the range
/// the view reads, as it was, the copy is another one wherever the two
/// lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(view: &View) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    // A view off its buffer's elements is made on a copy of its own bytes
    // (`own_bytes`): the same question `buffer_srv` and `buffer_uav` ask,
    // of the copy they find, so that the copy brought up to date is the
    // one the slot shows.
    if let Some((base, width)) = buffer_extent(view)
        && let Some((first, count, element, stride, unknown)) = buffer_fields(view)
        && let Some(whole) = crate::shadow::find(base, width)
        && let Some((start, bytes)) = own_bytes(whole.offset(base), base, first, count, element, stride, unknown)
        && let Some(own) = crate::shadow::exact(start, bytes)
    {
        return Some((own, start, bytes));
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((shadow, address, len));
    }
    crate::shadow::find(address, len).map(|shadow| (shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

/// Whether the pass over records copied in as bytes also mirrors slots
/// that draws read through tables lately (`VICTORY_HEAP_SYNC_TABLES=1`;
/// off by default: [`HostHeap::mirror_copied`]).
fn sync_tables() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_TABLES").is_some_and(|v| v == "1"))
}

/// Since the last frame line: slots on changed heap pages that
/// [`HostHeap::mirror_copied`] left to their draws (tables' slots).
pub(crate) static COPIED_TABLED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A 64-bit fingerprint of a descriptor record's bytes (16 or 32 of them):
/// equal records give equal prints, and two different records the same
/// print about once in 2^64. Not a hash anyone can attack: the records
/// are our own.
fn record_print(record: &[u8]) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut h = K ^ record.len() as u64;
    let mut words = record.chunks_exact(8);
    for word in &mut words {
        let w = u64::from_le_bytes(word.try_into().unwrap_or([0; 8]));
        h = (h ^ w).wrapping_mul(K);
        h ^= h >> 29;
    }
    for &b in words.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(K);
    }
    h ^ (h >> 32)
}

/// Whether a buffer view off its buffer's elements gets a copy of its own
/// bytes (`VICTORY_VIEW_OWN_COPY=0`: it is refused and reads zeros, as
/// before).
fn own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_OWN_COPY").is_none_or(|v| v != "0"))
}

/// The title memory (start, bytes) of a buffer view that cannot be made on
/// the host copy of its buffer: `count` elements of `element` bytes from
/// element `first` of the buffer at `base`, which lies `offset` bytes into
/// that copy, where `offset` is not a whole number of elements and the view
/// cannot be a raw view of the same bytes either ([`raw_window`]:
/// `unknown`, the view has no format, and `stride` is its structure's).
/// The host counts a view's first element from its buffer's start, so such
/// a view has no host view there; it is made on a copy of its own bytes
/// (`crate::shadow::exact`), where it starts at element 0. `None`: the
/// view is made on its buffer's copy, as every other.
fn own_bytes(offset: u64, base: u64, first: u64, count: u32, element: u64, stride: u32, unknown: bool) -> Option<(u64, u64)> {
    own_bytes_if(own_copy_on(), offset, base, first, count, element, stride, unknown)
}

#[allow(clippy::too_many_arguments)]
fn own_bytes_if(
    on: bool,
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
) -> Option<(u64, u64)> {
    if !on || element == 0 || count == 0 || offset.is_multiple_of(element) {
        return None;
    }
    if unknown && raw_window(offset, first, count, stride).is_some() {
        return None;
    }
    let start = base.checked_add(first.checked_mul(element)?)?;
    Some((start, u64::from(count) * element))
}

/// Of a buffer view: its first element, its elements, the bytes of one (a
/// raw view's: 4), its structure's stride, and whether it has no format.
/// As [`buffer_srv`] and [`buffer_uav`] read them from the same words.
fn buffer_fields(view: &View) -> Option<(u64, u32, u64, u32, bool)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => (d[..10].try_into().ok()?, *unordered),
        _ => return None,
    };
    if unordered {
        let d = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(d.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        Some((b.FirstElement, b.NumElements, element, b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    } else {
        let d = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        Some((b.FirstElement, b.NumElements, element_size(&d), b.StructureByteStride, d.Format == DXGI_FORMAT_UNKNOWN))
    }
}

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| u32::try_from(bytes / 4).ok().map(|dwords| (start / 4, dwords)))
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
fn buffer_view_ok(format: DXGI_FORMAT, first: u64, count: u32, stride: u32, raw: bool, width: u64) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// How many of `count` elements of `element` bytes from `first` on lie in
/// `width` bytes: the console lets a view reach past its buffer's end, the
/// host refuses one. A view with no element inside keeps its count (and is
/// refused).
fn elements_inside(first: u64, count: u32, element: u64, width: u64) -> u32 {
    let room = (width / element.max(1)).saturating_sub(first);
    match u32::try_from(room) {
        Ok(0) | Err(_) => count,
        Ok(room) => count.min(room),
    }
}

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, offset) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(offset, address, b.FirstElement, b.NumElements, element_size(&desc), b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, 0)
            }
            None => (shadow, offset),
        }
    };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, structured.FirstElement, structured.NumElements, structured.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let unit = if window.is_some() { 4 } else { element };
    buffer.NumElements = elements_inside(buffer.FirstElement, buffer.NumElements, unit, width);
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    shadow.handed(crate::shadow::Hand::View, address);
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, own) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(shadow.offset(address), address, b.FirstElement, b.NumElements, element, b.StructureByteStride, unknown)
            .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, true)
            }
            None => (shadow, false),
        }
    };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = if own { 0 } else { shadow.offset(address) };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| raw_window(offset, buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride))
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(offset, element, "unordered buffer view not on an element of its host copy");
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let unit = if window.is_some() { 4 } else { element };
    buffer.NumElements = elements_inside(buffer.FirstElement, buffer.NumElements, unit, width);
    let (first, count, stride) = (buffer.FirstElement, buffer.NumElements, buffer.StructureByteStride);
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    shadow.handed(crate::shadow::Hand::View, address);
    Some(Some(shadow.buffer.clone()))
}

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

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
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
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            shadow.handed(crate::shadow::Hand::View, address);
            Some(Some(shadow.buffer.clone()))
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
                    crate::tiles::show_from(desc.get_or_insert(whole), first, u32::from(own.MipLevels));
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
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
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
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
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
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

/// Whether [`HostHeap::made`] is changed a slot at a time, each slot
/// claimed by the thread that changes it, with no lock over the heap; the
/// host descriptor is made outside any such lock, what a write changes
/// in the queue of textures and the sets of buffer views is logged and
/// applied by whatever reads them next ([`NoteLog`]), and a write of a
/// slot that holds no placed texture view leaves the spans' lock alone
/// ([`HostHeap::spanned`]).
/// `VICTORY_MADE_ATOMIC=1`; otherwise one lock over the heap's `made` is
/// held across all of it, as before.
fn made_atomic() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MADE_ATOMIC").is_some_and(|v| v == "1"))
}

/// A claim on one slot of [`HostHeap::made`] ([`HostHeap::claim`]).
struct Claimed<'a> {
    slot: &'a std::sync::atomic::AtomicU64,
    /// What the slot held when it was claimed.
    old: u64,
    /// What it holds once the claim is dropped.
    value: u64,
}

impl Drop for Claimed<'_> {
    fn drop(&mut self) {
        // After the host slot, its word and what it keeps: a thread that
        // reads this value sees them too.
        self.slot
            .store(self.value, std::sync::atomic::Ordering::Release);
    }
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
}

/// Logs of the changes writes of slots make to a heap's queue of textures
/// and sets of buffer views, with `VICTORY_MADE_ATOMIC` ([`made_atomic`]):
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
        heap.constant_at(1, unsafe { buffer.GetGPUVirtualAddress() }, 969_472);
        assert!(alive(), "a view of the list's own memory");
        assert_eq!(constant_bytes(480), 512);
        assert_eq!(constant_bytes(64 << 10), 64 << 10);
        assert_eq!(constant_bytes(u32::MAX), 64 << 10);
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
        // The profiler records nothing in tests: no holder was ever said.
        assert_eq!(heap.holder.load(std::sync::atomic::Ordering::Relaxed), 0);
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
        heap.make_at(0, &View::Constant(0, 0));
        assert!(heap.kept_at(0, |kept| kept.is_none()));
        heap.constant_at(1, 0, 0);
        assert!(heap.kept_at(1, |kept| kept.is_none()));
    }
}

/// `made` changed a slot at a time (`VICTORY_MADE_ATOMIC=1`) against the
/// heap-wide lock.
#[cfg(test)]
mod made_tests {
    use super::*;

    /// Slots of the heaps these tests make.
    const SLOTS: u32 = 16;

    /// Memory of its own for [`views`]: 1 MiB, left to the process.
    fn memory() -> u64 {
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
    fn views(at: u64) -> Vec<Option<View>> {
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
    fn texture(address: u64, width: u32) -> View {
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
    fn write(handle: u64, view: &Option<View>) {
        match view {
            Some(view) => super::super::record(handle, view.clone()),
            // SAFETY: the record's first words, in the caller's memory.
            None => unsafe { (handle as *mut [u32; 3]).write_unaligned([0; 3]) },
        }
    }

    fn next(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// What slot `index` of `made` holds, once no thread changes it.
    fn key(heap: &HostHeap, index: u32) -> u64 {
        let _made = heap.lock_made(None);
        heap.claim(index, |_| false).err().expect("not claimed")
    }

    /// The slots in the queue of textures picked by index, in its order.
    fn queued(heap: &HostHeap) -> Vec<(u32, u64)> {
        heap.texture_slots_due(u64::MAX, SLOTS as usize).0
    }

    /// One thread's writes, mirrors, walks and passes over the slots, made
    /// on a heap under the heap-wide lock and on one changed a slot at a
    /// time, leave both alike after every step: what each slot is made
    /// from, its word, the sets of buffer views and the queue of textures.
    #[test]
    fn a_slot_at_a_time_makes_what_the_heap_wide_lock_makes() {
        let (Some(locked), Some(atomic)) = (
            HostHeap::with_made(false, SLOTS, false),
            HostHeap::with_made(false, SLOTS, true),
        ) else {
            return; // no host GPU
        };
        let views = views(memory());
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let base = records.as_mut_ptr() as u64;
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        // Slot looks that found it null, made, and with a word other than 0.
        let mut seen = [0u64; 3];
        let sorted = |mut slots: Vec<u32>| {
            slots.sort_unstable();
            slots
        };
        // (Other tests count stale views too.)
        let stale_before = STALE[0].load(std::sync::atomic::Ordering::Relaxed);
        for step in 0..3_000 {
            let r = next(&mut rng);
            let slot = (r % u64::from(SLOTS)) as u32;
            let at = base + u64::from(slot) * 32;
            match r >> 8 & 15 {
                0 => {
                    assert_eq!(
                        locked.texture_slot_gone(slot),
                        atomic.texture_slot_gone(slot),
                        "step {step}"
                    );
                }
                1 => {
                    assert_eq!(
                        locked.texture_slot_made_from(slot, at),
                        atomic.texture_slot_made_from(slot, at),
                        "step {step}"
                    );
                }
                2 => {
                    for heap in [&locked, &atomic] {
                        heap.mirror_marking(slot, at, true, false, hold::WALK);
                    }
                }
                op => {
                    write(at, &views[(r >> 16) as usize % views.len()]);
                    for heap in [&locked, &atomic] {
                        heap.mirror(slot, at, op & 1 == 1);
                    }
                }
            }
            for s in 0..SLOTS {
                let made = key(&locked, s);
                assert_eq!(made, key(&atomic, s), "step {step}, slot {s}");
                assert_eq!(locked.word(s), atomic.word(s), "step {step}, slot {s}");
                seen[usize::from(made != 0)] += 1;
                seen[2] += u64::from(locked.word(s).is_some_and(|word| word != 0));
            }
            assert_eq!(locked.buffer_slots(), atomic.buffer_slots(), "step {step}");
            assert_eq!(queued(&locked), queued(&atomic), "step {step}");
            if step % 5 == 0 {
                assert_eq!(
                    sorted(locked.buffer_changes()),
                    sorted(atomic.buffer_changes()),
                    "step {step}"
                );
            }
        }
        // Slots were made, left null and given words along the way.
        assert!(seen.iter().all(|&n| n > 0), "{seen:?}");
        assert!(
            STALE[0].load(std::sync::atomic::Ordering::Relaxed) > stale_before,
            "stale views"
        );
        assert!(alive());
    }

    /// Whether the host device is still there.
    fn alive() -> bool {
        // SAFETY: a plain device query.
        host().is_some_and(|host| unsafe { host.device.GetDeviceRemovedReason() }.is_ok())
    }

    /// Threads that write, mirror and empty the same slot and slots of
    /// their own at once, with views that make each other's slots stale,
    /// all finish (no two wait for each other), and leave every slot made
    /// from a view with that view's word: a word never says another kind
    /// than the descriptor under it (the null slot crash). Written once
    /// more by one thread, every slot then holds what it holds on a heap
    /// under the heap-wide lock written the same way.
    #[test]
    fn many_threads_leave_each_slot_with_its_own_word() {
        const THREADS: u64 = 8;
        let Some(heap) = HostHeap::with_made(false, SLOTS, true) else {
            return; // no host GPU
        };
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
        let Some(locked) = HostHeap::with_made(false, SLOTS, false) else {
            return;
        };
        let mut records = vec![[0u32; 8]; SLOTS as usize];
        let base = records.as_mut_ptr() as u64;
        for s in 0..SLOTS {
            let at = base + u64::from(s) * 32;
            write(at, &views[s as usize % 3]);
            heap.mirror(s, at, true);
            locked.mirror(s, at, true);
        }
        for s in 0..SLOTS {
            assert_eq!(key(&heap, s), key(&locked, s), "slot {s}");
            assert_eq!(heap.word(s), locked.word(s), "slot {s}");
        }
        assert_eq!(heap.buffer_slots(), locked.buffer_slots());
        let slots = |heap: &HostHeap| {
            let mut slots: Vec<u32> = queued(heap).into_iter().map(|(slot, _)| slot).collect();
            slots.sort_unstable();
            slots
        };
        assert_eq!(slots(&heap), slots(&locked));
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

    /// The key of the record at `handle` ([`HostHeap::mirror`]).
    fn record_key(handle: u64) -> u64 {
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
        let Some(heap) = HostHeap::with_made(false, SLOTS, true) else {
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
        let Some(heap) = HostHeap::with_made(false, SLOTS, true) else {
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
                    record_key(at)
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

    /// Slots written while the heap buffer pass takes the changes over and
    /// over are each in some take, or in the one after the writes: the
    /// walk sees every slot written while it ran, in its next pass at the
    /// latest. The set of buffer views ends as the last writes left it.
    #[test]
    fn the_walk_sees_every_slot_written_while_it_ran() {
        const THREADS: u32 = 8;
        // Two slots of its own per thread.
        const _: () = assert!(THREADS * 2 == SLOTS);
        let Some(heap) = HostHeap::with_made(false, SLOTS, true) else {
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
        let Some(heap) = HostHeap::with_made(false, SLOTS, true) else {
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
            // change to the queue and one to the sets.
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
        assert!(most.0 > 0 && most.1 > 0, "{most:?}");
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
mod record_print_tests {
    use super::record_print;

    /// Equal records print alike; a record that differs in any one byte,
    /// or in its length, prints otherwise.
    #[test]
    fn a_record_that_changed_prints_otherwise() {
        let record: Vec<u8> = (0..32u8).map(|i| i.wrapping_mul(37).wrapping_add(5)).collect();
        let print = record_print(&record);
        assert_eq!(print, record_print(&record.clone()));
        for at in 0..32 {
            for bit in 0..8 {
                let mut other = record.clone();
                other[at] ^= 1 << bit;
                assert_ne!(record_print(&other), print, "byte {at} bit {bit}");
            }
        }
        assert_ne!(record_print(&record[..16]), print);
        assert_ne!(record_print(&[0u8; 32]), record_print(&[0u8; 16]));
        // A record of zeros is told from "never looked at" by the caller's `| 1`.
        assert_ne!(record_print(&[0u8; 32]) | 1, 0);
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
