//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::census::Dropped;
use crate::frame_budget::Source;
use crate::host::{HostList, OpKind};
use crate::object::This;
use crate::pipeline::host::HostPipeline;
use crate::shadow::Hand;
use crate::spans::Span;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake, by the first 16 hex
/// digits of their hash as the frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// The sampler heap we bind when the title binds none (a value no title heap
/// address takes).
const SAMPLERS_OURS: usize = usize::MAX;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours (and with
/// `VICTORY_INDIRECT_READABLE=1` no barrier either: the scratch buffer
/// then stays readable as arguments until the next clamp,
/// `crate::indirect`), until one of these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1"))
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// The key of the pipeline bound last ([`HostPipeline::key`]), for the
    /// list's ops ([`HostList::op`]).
    host_key: u64,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root constant buffer views bound as one-view tables in this list
    /// ([`crate::pipeline::host::bounded_root_cbvs`]), likewise, with the
    /// bytes the view holds: of their host copy, of the title's bytes in
    /// the host list's scratch (left out when the list is cut,
    /// [`State::replay`]), or a null view.
    root_cbv_slots: std::collections::HashMap<(usize, u64), RootCbvSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

/// Why a draw is dropped for the tables [`State::tables`] made: one of them
/// found no free dynamic slots (`u32::MAX` for its offset).
fn table_refused(made: &[(u8, u32)]) -> Option<Dropped> {
    made.iter()
        .any(|m| m.1 == u32::MAX)
        .then_some(Dropped::TableFull)
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.root_cbv_slots.clear();
        self.tables_gone = true;
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// A root constant buffer view's one-view table: its dynamic slot, the
/// host copy the slot shows (none: the list's scratch, or with no bytes a
/// null view), and the bytes the view holds.
type RootCbvSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>, u64);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Of [`HEAP_BUFFER_MICROS`], the microseconds of the pass over textures
/// picked by index that runs inside it (`heap_textures`), and the buffer
/// slots the pass over buffer views looked at, and of those the ones it
/// left to the draws that read them through tables (`VICTORY_HEAP_PASS_TABLED=1`);
/// since the last frame line.
pub(crate) static HEAP_PASS_PARTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB,
/// other views as all that is left of their buffer. A constant buffer view
/// in a window of memory outside buffers is bound from a copy that holds the
/// whole 64 KiB ([`root_view_copy`], [`root_cbv_reach`]), and an unaligned
/// one takes 64 KiB of the list's scratch, because the title's memory goes
/// on past a window's end; in a title buffer's copy it is still cut at the
/// buffer's end, taken to hold the title's constant blocks. (Other views
/// were taken as 1 MiB: 1,452 of the 2,021 mesh vertex streams of a
/// cutscene frame reach further, and what lay past the first megabyte was
/// never brought up to date - meshes without bodies, exploded triangles.
/// `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = root_view_copy(address, srv)?;
    shadow.handed(if srv { Hand::RootSrv } else { Hand::RootCbv }, address);
    let len = shadow
        .left(address)
        .min(if srv { limit } else { ROOT_CBV_REACH });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh_from(list, address, len, Source::RootView);
        list.buffer_read(&shadow.buffer);
        return Some(gpu);
    }
    // The whole reach in the list's scratch, so that no read runs past the
    // end of a scratch chunk; the title's bytes in the first `len`.
    let reach = if root_cbv_reach() {
        ROOT_CBV_REACH
    } else {
        len
    };
    let (gpu, to) = list.scratch(reach)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

/// The most a root constant buffer view reads from its address: a constant
/// buffer of 4,096 vectors of 16 bytes.
const ROOT_CBV_REACH: u64 = 64 << 10;

/// Whether a root constant buffer view near the end of a window of memory
/// outside buffers is bound from a window that holds all it can read (on by
/// default; `VICTORY_ROOT_CBV_REACH=0`: the window its address names, as
/// before).
///
/// A root descriptor has no size, and a root view is found by its address
/// alone, which names its 4 MiB cell's window (or the cell's tail). The
/// title's memory goes on past the cell's end, and a constant block that
/// straddles a 4 MiB line is read past the end of the window's host buffer,
/// into whatever the host placed after it: as root shader resource views
/// were before they became bounded tables
/// (`crate::pipeline::host::bounded_root_srvs`). Where that is memory the
/// host has freed, the card reads freed memory. This is the suspected cause
/// of page faults "inside freed" allocations of whatever kind lay there
/// before (texture staging, host textures, other windows) in relay runs
/// with `VICTORY_WINDOW_KB=1024`, where more host buffers come and go; no
/// run has confirmed it yet. [`ROOT_CBV_SHORT`] counts such views with the
/// switch on and off.
///
/// limit: windows only. A title buffer's copy is kept as its address names
/// it: it is rounded up to 64 KiB, but a view in its last bytes still has
/// less than 64 KiB of it left, and the title is taken to keep its constant
/// blocks inside the buffer (counted in [`ROOT_CBV_SHORT`] too). Whatever
/// copy is taken, a root constant buffer view is bound as a one-view table
/// whose view holds only what that copy has
/// ([`crate::pipeline::host::bounded_root_cbvs`], on by default): a shader
/// reads at offsets it computes, also past 64 KiB, and those reads give
/// zeros past the view.
pub(crate) fn root_cbv_reach() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_CBV_REACH").is_none_or(|v| v != "0"))
}

/// Root constant buffer views whose copy, as their address names it, ends
/// less than [`ROOT_CBV_REACH`] on, since the start: in a window of memory
/// outside buffers (with [`root_cbv_reach`] on or off); of those, bound
/// from a window that holds the whole reach; not, because no window held it
/// (the short one kept); and in a title buffer's copy (kept).
pub(crate) static ROOT_CBV_SHORT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Root constant buffer views made one-view tables of title bytes
/// ([`crate::pipeline::host::bounded_root_cbvs`]) since the start; of
/// those, views that hold less than 64 KiB because their copy ends sooner
/// (reads past the view give zeros: through a root descriptor they reached
/// past the copy's end); and draws dropped because no view heap was bound
/// (a root descriptor needed none) or no dynamic slot was free.
pub(crate) static ROOT_CBV_TABLED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Makes dynamic slot `index` of `host` the constant buffer view of a root
/// constant buffer view at title `address` ([`State::root_cbv`]): the host
/// copy it shows (none for the list's scratch or a null view) and the bytes
/// the view holds (0: a null view).
fn root_cbv_view(
    list: &HostList,
    host: &crate::descriptor::host::HostHeap,
    index: u32,
    address: Option<u64>,
) -> (Option<std::sync::Arc<crate::shadow::Shadow>>, u64) {
    let Some((a, shadow)) = address.and_then(|a| Some((a, root_view_copy(a, false)?))) else {
        host.constant_at(index, 0, 0);
        return (None, 0);
    };
    shadow.handed(Hand::RootCbv, a);
    let len = shadow.left(a).min(ROOT_CBV_REACH);
    let bytes = len.next_multiple_of(256);
    let gpu = shadow.gpu(a);
    // SAFETY: a live buffer.
    let (base, width) = unsafe {
        (
            shadow.buffer.GetGPUVirtualAddress(),
            shadow.buffer.GetDesc().Width,
        )
    };
    ROOT_CBV_TABLED[0].fetch_add(1, Ordering::Relaxed);
    if len < ROOT_CBV_REACH {
        ROOT_CBV_TABLED[1].fetch_add(1, Ordering::Relaxed);
    }
    if gpu.is_multiple_of(256) && gpu - base + bytes <= width {
        host.constant_view_at(index, &shadow.buffer, gpu, bytes as u32);
        shadow.refresh_from(list, a, len, Source::RootView);
        list.buffer_read(&shadow.buffer);
        return (Some(shadow), bytes);
    }
    // The title's bytes in the list's scratch (whole 256s of it are the
    // list's), filled when the list is submitted: every bind of the address
    // in the list reads the same bytes.
    let Some((gpu, to)) = list.scratch(bytes) else {
        host.constant_at(index, 0, 0);
        return (None, 0);
    };
    list.defer_fill(to, a as *const u8, len as usize);
    // SAFETY: the list's scratch, `bytes` of it from `to`: what the view
    // holds past the title's bytes reads zeros, not what an earlier list
    // left there.
    unsafe { std::ptr::write_bytes(to.add(len as usize), 0, (bytes - len) as usize) };
    host.constant_at(index, gpu, bytes as u32);
    (None, bytes)
}

/// Counts a root constant buffer view in [`ROOT_CBV_SHORT`]`[kind]`, and
/// tells the first of each kind with the copy's range and how far the view
/// is from its end.
fn short_root_cbv(kind: usize, what: &str, shadow: &crate::shadow::Shadow, address: u64) {
    static FIRST: [std::sync::Once; 4] = [const { std::sync::Once::new() }; 4];
    ROOT_CBV_SHORT[kind].fetch_add(1, Ordering::Relaxed);
    FIRST[kind].call_once(|| {
        let (start, len) = shadow.span();
        tracing::info!(
            "root constant buffer view {what}: copy {start:#x} + {len:#x}, the view at {address:#x} with {:#x} bytes \
             of the copy left (a constant buffer reads up to {ROOT_CBV_REACH:#x})",
            shadow.left(address)
        );
    });
}

/// The host copy a root view at `address` is bound from ([`root_view`]):
/// the one its address names, or, for a constant buffer view that would
/// end less than [`ROOT_CBV_REACH`] on in a window of memory outside
/// buffers, a window that holds the whole reach ([`root_cbv_reach`]). A
/// title buffer's copy is kept (see the limit there). Each constant buffer
/// view with less than the reach left is counted ([`ROOT_CBV_SHORT`]).
fn root_view_copy(address: u64, srv: bool) -> Option<std::sync::Arc<crate::shadow::Shadow>> {
    let shadow = crate::shadow::find(address, 1)?;
    if srv || shadow.left(address) >= ROOT_CBV_REACH {
        return Some(shadow);
    }
    if !shadow.is_window() {
        short_root_cbv(
            3,
            "near the end of a title buffer's copy (kept)",
            &shadow,
            address,
        );
        return Some(shadow);
    }
    short_root_cbv(0, "near the end of a window", &shadow, address);
    if !root_cbv_reach() {
        return Some(shadow);
    }
    let whole =
        crate::shadow::find(address, ROOT_CBV_REACH).filter(|s| s.left(address) >= ROOT_CBV_REACH);
    match whole {
        Some(whole) => {
            short_root_cbv(
                1,
                "bound from a window that holds its reach",
                &whole,
                address,
            );
            Some(whole)
        }
        None => {
            short_root_cbv(
                2,
                "near the end of a window, no window holds its reach (the short one kept)",
                &shadow,
                address,
            );
            Some(shadow)
        }
    }
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        list.end_mesh_batch();
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        // Root constant views in the old host list's scratch: that list's
        // fills, made when its part is submitted (their slots stay held).
        self.root_cbv_slots
            .retain(|_, (_, shadow, bytes)| shadow.is_some() || *bytes == 0);
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_GROUP_BASE, USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        // Set between the parts of a large dispatch (`group_base`).
        if kind == USER_GROUP_BASE {
            return 0;
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false, true)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    /// With `note` a drop is counted by its reason ([`Dropped`]); the
    /// indirect path, whose drops are not those of `DROPPED`, passes false.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool, note: bool) -> bool {
        let drop = |why: Dropped| {
            if note {
                why.note();
            }
        };
        if crate::pipeline::is_skipped(self.pipeline as This) {
            drop(Dropped::Skipped);
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            drop(Dropped::NoPipeline);
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            drop(Dropped::Other);
            return false;
        }
        match self.bind(list, p, false) {
            Ok(()) => {}
            Err(why) => {
                drop(why);
                return false;
            }
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.handed(Hand::Vertex, v.address);
            shadow.refresh_from(list, v.address, u64::from(v.size), Source::Vertex);
            list.buffer_read(&shadow.buffer);
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                drop(Dropped::Other);
                return false;
            };
            shadow.handed(Hand::Index, v.address);
            shadow.refresh_from(list, v.address, u64::from(v.size), Source::Index);
            list.buffer_read(&shadow.buffer);
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// The heaps a list must have bound before one of our root signatures
    /// is set: the title's, and our sampler heap when it binds none. Every
    /// signature we make indexes its sampler heap directly, and D3D12
    /// requires one to be bound before such a signature (the debug layer's
    /// id 1419; NVIDIA removes the device for it). Compute lists that only
    /// read views are the case.
    fn wanted(title: [usize; 2]) -> [usize; 2] {
        let sampler = if title[1] != 0 {
            title[1]
        } else {
            SAMPLERS_OURS
        };
        [title[0], sampler]
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// `Ok` when the draw or dispatch may run. `Err(why)` says why it
    /// must be dropped ([`Dropped`]): a root view or table that could not
    /// be given a slot, no view heap, or no free table slots. Counting
    /// the drop by that reason is the caller's ([`State::bound`]).
    pub(crate) fn bind(
        &mut self,
        list: &HostList,
        p: &HostPipeline,
        compute: bool,
    ) -> Result<(), Dropped> {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        let want = Self::wanted(self.heaps);
        if self.host_heaps != want {
            let mut heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            if want[1] == SAMPLERS_OURS {
                match crate::descriptor::host::fallback_samplers() {
                    Some(h) => heaps.push(Some(h.heap.clone())),
                    None => once!("no sampler heap of ours to bind"),
                }
            }
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // Its submission's heap sync looks at these heaps.
            self.heaps
                .iter()
                .filter(|&&h| h != 0)
                .for_each(|&h| list.binds_heap(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = want;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        self.host_key = p.key;
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        for stage in &p.stages {
            let t = stage.translated();
            if t.stores {
                // The list's next clamp waits for what it stores
                // (`crate::indirect::precise`).
                list.may_write();
            }
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, compute);
            STORES.set(true);
            if let Some(why) = table_refused(&made) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return Err(why);
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return Err(Dropped::MemoryWithoutHeap);
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let constant_table = !srv && crate::pipeline::host::bounded_root_cbvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return Err(Dropped::RootSrv);
                        }
                        if constant_table && !self.root_cbv_table(list, None, param, compute) {
                            return Err(Dropped::RootCbv);
                        }
                        continue;
                    };
                    if constant_table {
                        if !self.root_cbv_table(list, Some(address), param, compute) {
                            return Err(Dropped::RootCbv);
                        }
                        continue;
                    }
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return Err(Dropped::RootSrv);
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return Err(Dropped::RootUav);
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        Ok(())
    }

    /// [`State::bind`] for the callers that only ask whether the draw runs:
    /// a draw it drops is counted by its reason.
    fn bound(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        match self.bind(list, p, compute) {
            Ok(()) => true,
            Err(why) => {
                why.note();
                false
            }
        }
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            s.handed(Hand::RootUav, a);
            let len = s.left(a).min(limit);
            s.refresh_from(list, a, len, Source::RootUav);
            list.buffer_uav(&s.buffer);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.handed(Hand::RootSrv, *a);
            s.refresh_from(list, *a, s.left(*a), Source::RootSrv);
            list.buffer_read(&s.buffer);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root constant buffer view as a one-view table
    /// ([`crate::pipeline::host::bounded_root_cbvs`]; [`State::root_cbv`]).
    /// False if the draw must be dropped (no view heap, no slot).
    fn root_cbv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
    ) -> bool {
        let Some((index, _)) = self.root_cbv(list, address) else {
            return false;
        };
        let Some((_, host)) = self.view_heap() else {
            return false;
        };
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// The bound title view heap and its host view heap.
    fn view_heap(
        &self,
    ) -> Option<(
        &'static crate::descriptor::Heap,
        &'static crate::descriptor::host::HostHeap,
    )> {
        match self.heaps[0] {
            0 => None,
            h => {
                let heap = crate::descriptor::heap(h as This);
                Some((heap, heap.host()?))
            }
        }
    }

    /// The one-view table a root constant buffer view at title `address`
    /// is bound as ([`crate::pipeline::host::bounded_root_cbvs`]; without
    /// it the caller binds a root descriptor at the host address
    /// [`root_view`] gives, as before): its dynamic slot in the bound view
    /// heap and the bytes the view holds. The view holds only what its host
    /// copy has from `address` on (at most the 64 KiB a host view may), so
    /// a read past that gives zeros instead of whatever the host placed
    /// after the copy: of the copy its address names ([`root_view_copy`],
    /// which takes a window that holds the 64 KiB where the named one ends
    /// sooner), or for an address not on 256 bytes there (the console needs
    /// 16) of the title's bytes in the list's scratch; a null view without
    /// an address or a copy. One slot per address per list (a view in the
    /// scratch per host list), a copy brought up to date at each bind.
    /// `None` if the draw must be dropped: no view heap or no slot (counted
    /// in [`ROOT_CBV_TABLED`]).
    fn root_cbv(&mut self, list: &HostList, address: Option<u64>) -> Option<(u32, u64)> {
        let Some((heap, host)) = self.view_heap() else {
            ROOT_CBV_TABLED[2].fetch_add(1, Ordering::Relaxed);
            once!("a root constant buffer view without a bound view heap: the draw is dropped");
            return None;
        };
        // The title heap: `tables_done` gives the slot back by it.
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        if let Some((index, shadow, bytes)) = self.root_cbv_slots.get(&key) {
            if let (Some(shadow), Some(a)) = (shadow, address) {
                shadow.handed(Hand::RootCbv, a);
                shadow.refresh_from(list, a, shadow.left(a).min(ROOT_CBV_REACH), Source::RootCbv);
                list.buffer_read(&shadow.buffer);
            }
            return Some((*index, *bytes));
        }
        let Some(index) = host.dynamic(1) else {
            ROOT_CBV_TABLED[3].fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.dynamic.push((key.0, index, 1));
        let (shadow, bytes) = root_cbv_view(list, host, index, address);
        self.root_cbv_slots.insert(key, (index, shadow, bytes));
        Some((index, bytes))
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        self.heap_buffers_in(list, crate::frame::current())
    }

    /// [`State::heap_buffers`] in frame `frame` (the current one; tests
    /// name their own).
    fn heap_buffers_in(&self, list: &HostList, frame: u64) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        // Keyed per host heap so binding a second view heap on the same list
        // keeps its strides buffer and refreshes its views too.
        if !list.first_use(std::ptr::from_ref(host) as usize) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // Once per frame per host heap: avoids re-scanning 81,000 views on
        // every list while never starving a second view heap bound in the same
        // frame, and releases all cached `Arc<Shadow>` refs on `HostHeap::drop`.
        // Claimed before the lock (`claim_pass`): the other lists of the
        // frame go on recording, and their submissions wait for the walk.
        let walking = if heap_claim() {
            let gate = host.walk_gate();
            let walking = claim_pass(gate, frame, heap_wait_at_submit(), |mark| {
                list.needs_walk(gate.clone(), mark);
            });
            if walking.is_none() {
                return Some(gpu);
            }
            walking
        } else {
            None
        };
        let started = std::time::Instant::now();
        // Every list that gets here waits for the lock, the pass's own list
        // too (`won`: it runs the pass).
        let mut lock_wait =
            crate::profile::span_args("heap pass lock wait", &[("heap", host.id())]);
        // The heap's own lock with `VICTORY_HEAP_SWEEP_OWN=1`
        // (`HostHeap::buffer_sweep`), else the one of every heap's.
        let mut global = None;
        let mut own: Option<std::sync::MutexGuard<'_, HeapRanges>> = None;
        let entry: &mut HeapRanges = if heap_sweep_own() {
            own.insert(host.buffer_sweep.lock().unwrap_or_else(|e| e.into_inner()))
        } else {
            global
                .insert(HEAP_RANGES.lock().unwrap_or_else(|e| e.into_inner()))
                .get_or_insert_with(Default::default)
                .entry(host.id())
                .or_insert_with(HeapRanges::new)
        };
        let won = entry.last_frame != frame;
        if let Some(wait) = lock_wait.as_mut() {
            wait.arg("won", u64::from(won));
        }
        drop(lock_wait);
        if !won {
            return Some(gpu);
        }
        entry.last_frame = frame;
        let _pass = crate::profile::span_args("heap buffer pass", &[("heap", host.id())]);
        let passes = &mut entry.passes;
        let ranges = &mut entry.ranges;
        // Slots on C-pool homes let go since are walked as changed.
        host.pool_slots_let_go();
        let changed = host.buffer_changes();
        let plan = pass_slots(*passes, heap_spread());
        let full = plan == PassSlots::All;
        let slots = match plan {
            PassSlots::All => {
                ranges.clear();
                host.buffer_slots()
            }
            PassSlots::Changed => changed,
            PassSlots::ChangedAndPart(part) => {
                // The written ones, and this pass's part of the others.
                let parts = HEAP_ROUND as u32;
                forget_part(ranges, part, parts);
                let mut slots = host.buffer_slots_of(part, parts);
                slots.extend(changed);
                slots.sort_unstable();
                slots.dedup();
                slots
            }
        };
        *passes += 1;
        // The walk's parts are timed in sums, not in a span each (some
        // 60,000 slots a pass): the views `mirror` makes (`make_us`) and the
        // copies looked up (`copy_us`).
        let mut walk = crate::profile::span_args(
            "heap walk",
            &[("slots", slots.len() as u64), ("full", u64::from(full))],
        );
        let timed = walk.is_some();
        if timed {
            crate::profile::parts_begin();
        }
        let mut copy_ticks = 0;
        // Of the jobs on other threads (`VICTORY_HEAP_PASS_THREADS`): the
        // ticks and the count of the views they made.
        let mut jobs_made = (0, 0);
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive =
            !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
        // Slots draws read through tables lately are the draws' to make
        // (`VICTORY_HEAP_PASS_TABLED=1`).
        let tabled = heap_pass_tabled().then_some(frame);
        let threads = heap_pass_threads();
        if threads > 1 && slots.len() > HEAP_WALK_JOB {
            let handles: Vec<Option<u64>> = slots.iter().map(|&i| heap.handle(i)).collect();
            let walked = walk_spread(host, &slots, &handles, threads, tabled, timed, alive);
            copy_ticks = walked.copy_ticks;
            jobs_made = (walked.make_ticks, walked.made);
            if let Some((why, job)) = walked.removed
                && !GONE.swap(true, Ordering::Relaxed)
            {
                let at = job * HEAP_WALK_JOB;
                let last: Vec<String> = slots[at..slots.len().min(at + HEAP_WALK_JOB)]
                    .iter()
                    .filter_map(|&i| {
                        crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}"))
                    })
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(
                    why,
                    "the host device went while buffer views of the heap were made; one of: {last:#?}"
                );
                crate::host::log_removal_context();
            }
            // In slot order of the walk, as one thread leaves them.
            for (index, found) in walked.found {
                ranges.remove(&index);
                if let Some(range) = found {
                    ranges.insert(index, range);
                }
            }
        } else {
            for (n, &index) in slots.iter().enumerate() {
                ranges.remove(&index);
                let Some(handle) = heap.handle(index) else {
                    continue;
                };
                if tabled.is_some_and(|frame| host.buffer_slot_tabled_lately(index, frame)) {
                    HEAP_PASS_PARTS[2].fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                // Its copy and range are taken below from what this returns:
                // it does not mark the slot for the next pass.
                let Some(view) = host.mirror_walked(index, handle) else {
                    continue;
                };
                #[cfg(test)]
                WALK_HOOK.with_borrow_mut(|hook| {
                    if let Some(hook) = hook {
                        hook();
                    }
                });
                // Every 64 views while the device lives (a cheap query).
                if alive
                    && n % 64 == 63
                    && let Some(why) = crate::descriptor::host::device_removed()
                {
                    alive = false;
                    GONE.store(true, Ordering::Relaxed);
                    let last: Vec<String> = slots[n - 63..=n]
                        .iter()
                        .filter_map(|&i| {
                            crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}"))
                        })
                        .map(|s| s.chars().take(200).collect())
                        .collect();
                    tracing::error!(
                        why,
                        slot = index,
                        "the host device went while buffer views of the heap were made; one of: {last:#?}"
                    );
                    crate::host::log_removal_context();
                }
                let Some(range) = walked_copy(host, index, &view, timed, &mut copy_ticks) else {
                    continue;
                };
                ranges.insert(index, range);
            }
        }
        if let Some(walk) = walk.as_mut() {
            let (make_ticks, made) = crate::profile::parts_end();
            walk.arg("made", made + jobs_made.1);
            walk.arg("make_us", crate::profile::micros(make_ticks + jobs_made.0));
            walk.arg("copy_us", crate::profile::micros(copy_ticks));
            walk.arg("threads", threads as u64);
        }
        drop(walk);
        // The walk has ended: its last `mirror` is behind it, and what is
        // left of the pass writes no slot of the heap's buffer views. The
        // lists that lost the claim may run (`Host::submit`).
        drop(walking);
        let views = ranges.len() as u32;
        let mut spanning = crate::profile::span_args("heap spans", &[("ranges", u64::from(views))]);
        // Per host copy, the ranges its picked views cover, in order, those
        // less than 4 KiB apart joined (a refresh takes 4 KiB parts anyway).
        // The bytes between views further apart are not refreshed here: a
        // refresh of the copy's whole span took the home bytes a flight
        // writes between two views, and the submission waited for that
        // flight (H2). Every binding that reads such bytes has a refresh of
        // its own: the views of tables (`readable`, also the unnamed views of
        // a small table the pass leaves to the draw with
        // `VICTORY_HEAP_PASS_TABLED=1`, `State::tables`), vertex and index
        // buffers, and indirect arguments.
        let mut spans: std::collections::HashMap<
            usize,
            (std::sync::Arc<crate::shadow::Shadow>, Vec<(u64, u64)>),
        > = std::collections::HashMap::new();
        for (shadow, from, to) in ranges.values() {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            spans
                .entry(key)
                .or_insert_with(|| (shadow.clone(), Vec::new()))
                .1
                .push((*from, *to));
        }
        for (_, joined) in spans.values_mut() {
            join_near(joined, HEAP_PASS_JOIN);
        }
        drop(global);
        drop(own);
        // Its deferred steps are the pass's (the `run_deferred` span says so).
        list.ran_heap_pass();
        let bytes: u64 = spans
            .values()
            .flat_map(|(_, joined)| joined)
            .map(|(from, to)| to - from)
            .sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, joined) in spans.values() {
            // Once per copy, then every range of it: a check per range would
            // find the frame noted by the first and leave the others stale.
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            for &(from, to) in joined {
                shadow.refresh_from(list, from, to - from, Source::HeapPass);
            }
        }
        if let Some(span) = spanning.as_mut() {
            span.arg("copies", spans.len() as u64);
            span.arg("mb", bytes >> 20);
        }
        drop(spanning);
        let textures_began = std::time::Instant::now();
        self.heap_textures(list, heap, host);
        HEAP_PASS_PARTS[0].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The frame whose early pass has run: taken only by a draw with a
        // view heap, so a frame whose first draws have none still gets it.
        static EARLY_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let bake = bake && left > 0;
        if !bake && (early_ms == 0 || EARLY_FRAME.load(Ordering::Relaxed) == frame) {
            return;
        }
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let budget = if bake {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if EARLY_FRAME.swap(frame, Ordering::Relaxed) != frame {
            (early_ms, 64 << 20)
        } else {
            // Another thread's draw took this frame's early pass.
            return;
        };
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        self.heap_textures(list, heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        // On the profile, which caller's pass it is: 1 for one made early
        // or for a bake (`State::early_textures`, with a budget of its
        // own), 0 for the heap buffer pass's.
        let early = ("early", u64::from(PASS_BUDGET.get().is_some()));
        let _pass = crate::profile::span_args("heap texture pass", &[early]);
        /// Bytes of texture memory staged per frame.
        // `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames. Under
        // `VICTORY_FLIP` a pair of values compares two ([`crate::flip`]).
        static PER_FRAME_MB: crate::flip::Pair = crate::flip::Pair::new("VICTORY_TEXTURE_STAGE_MB");
        let per_frame: u64 = PER_FRAME_MB.parsed().unwrap_or(192u64).saturating_mul(1 << 20);
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut queue = crate::profile::span_args("texture pass: queue", &[early]);
        let mut room = crate::resource::room();
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        host.texture_slots_remapped(frame);
        // Views that fell out of the queue with their memory there.
        host.texture_sweep(heap, frame);
        // Levels on demand: the slots of textures whose first level with
        // memory changed since the last pass are made again from it.
        host.texture_views_remade();
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room {
            host.texture_slots_waiting(frame);
        }
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        if let Some(queue) = queue.as_mut() {
            queue.arg("queued", census[0]);
            queue.arg("due", census[1]);
        }
        drop(queue);
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        // `VICTORY_TEXTURE_STAGE_MS` (a pair under `VICTORY_FLIP`, as above).
        static SPAN_MS: crate::flip::Pair = crate::flip::Pair::new("VICTORY_TEXTURE_STAGE_MS");
        let span = std::time::Duration::from_millis(SPAN_MS.parsed().unwrap_or(250));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && due.len() >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        let began = std::time::Instant::now();
        let mut uploads = crate::profile::span_args("texture pass: uploads", &[early]);
        // Slots looked at, uploaded, their bytes, and why the loop stopped:
        // 0 at the queue's end, 1 out of time, 2 out of bytes, 3 too many
        // refused for lack of room, 4 the device went.
        let (mut looked, mut uploaded, mut uploaded_bytes, mut stop) = (0u64, 0u64, 0u64, 0u64);
        for (index, write) in due {
            if SPENT.load(Ordering::Relaxed) >= bytes_at_most {
                stop = 2;
                break;
            }
            if began.elapsed() > span {
                stop = 1;
                break;
            }
            looked += 1;
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint(at, len) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                        Some(wait) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, looks + 1, uploaded)
                        }
                        None => host.texture_slot_done(index, write, None),
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                if n < 200 || n.is_multiple_of(1000) {
                    tracing::warn!(
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            if let Some(View::Placed(false, resource, desc)) = heap.handle(index).and_then(crate::descriptor::view) {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(&resource).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                host.texture_slot_here(index);
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(View::Shader(object, _)) = heap.handle(index).and_then(crate::descriptor::view)
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                if !has_its_texture {
                    refused += 1;
                    crate::census::TEXTURE_PASS_HELD[1].fetch_add(1, Ordering::Relaxed);
                    if full_card_pass() {
                        crate::census::TEXTURE_PASS_HELD[0].fetch_add(1, Ordering::Relaxed);
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            stop = 3;
                            break;
                        }
                    } else if refused > 256 {
                        stop = 3;
                        break;
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(500) {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            let view = heap.handle(index).and_then(|handle| {
                host.mirror_at(index, handle, true, crate::descriptor::host::hold::UPLOAD)
            });
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if tabled {
                crate::texture::upload_again(resource);
            }
            if let Some(view) = &view {
                // (Not tabled: the staging state made here goes below.)
                crate::texture::PICKING.set(!tabled);
                readable(list, sampled.kind, view);
                crate::texture::PICKING.set(false);
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                // (The loop is the pass's last part: the end of the pass.)
                stop = 4;
                break;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here (only the staging buffer copied
                // again, `upload_again`), so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more. With transient
                // staging buffers the copy may be left to the next list
                // (`texture::copy_pending`): the slot stays due for it.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                let again = again.or_else(|| crate::texture::copy_pending(resource).then_some((frame + 1, 0)));
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                // (An upload left out as the host copy holds those bytes
                // has no staging state: looked at again all the same.)
                crate::texture::staged_memory(resource)
                    .or_else(|| crate::texture::left_out_memory(resource))
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = crate::resource::room();
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            uploaded += 1;
            uploaded_bytes += bytes;
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let print = crate::texture::fingerprint(at, len);
                    host.texture_slot_recheck(index, write, frame + RECHECK_FRAMES[0], (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
        crate::census::TEXTURE_PASS_STOP[stop as usize].fetch_add(1, Ordering::Relaxed);
        if let Some(uploads) = uploads.as_mut() {
            uploads.arg("looked", looked);
            uploads.arg("uploaded", uploaded);
            uploads.arg("mb", uploaded_bytes >> 20);
            uploads.arg("stop", stop);
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        // It may write the arguments a window or a batch holds clamped.
        list.end_mesh_batch();
        self.window = None;
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            Dropped::Skipped.note();
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            Dropped::BuilderOff.note();
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            Dropped::NoPipeline.note();
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            DROPPED.fetch_add(1, Ordering::Relaxed);
            Dropped::GraphicsDispatch.note();
            return;
        }
        if !self.bound(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        list.op(OpKind::Dispatch, self.host_key);
        match group_base(p) {
            Some((param, dword)) if x > crate::indirect::PART => {
                // More than the host runs in one dimension: in parts.
                // SAFETY: an open list with everything set; the parameter
                // and dword are the pipeline's.
                unsafe {
                    let mut base = 0;
                    while base < x {
                        list.list.SetComputeRoot32BitConstant(param, base, dword);
                        list.list.Dispatch((x - base).min(crate::indirect::PART), y, z);
                        base += crate::indirect::PART;
                    }
                    list.list.SetComputeRoot32BitConstant(param, 0, dword);
                }
            }
            // SAFETY: an open list with everything set.
            _ => unsafe { list.list.Dispatch(x, y, z) },
        }
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (`table` values with the view the table starts at; `const` values
    /// with the view at that heap index, if one is, since dynamic resources
    /// may index the heap with one, though most are plain numbers; `host`
    /// values, which VIctory computes, with none), the host addresses of
    /// its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                // A host word is an address VIctory computes, not a heap
                // index: it gets no view.
                let (root, index) = match u.root {
                    RootType::Table => ("table", Some(v / 32)),
                    RootType::Other(_) => ("host", None),
                    _ => ("const", Some(v)),
                };
                let _ = write!(text, " u{}={v:#x} {root}", u.index);
                if let Some(index) = index
                    && let Some(slot) = views.and_then(|h| h.describe_slot(index))
                {
                    let at = if root == "table" {
                        "heap"
                    } else {
                        "as heap index"
                    };
                    let _ = write!(text, " [{at} {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        list.end_mesh_batch();
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        // (`prepare_for` counts its own reasons, so a drop is counted once:
        // the mesh reason is for a dispatch the test switches keep out and
        // for a list that has no mesh dispatch.)
        let gated = mesh_test() != 1 && mesh_kept(self.pipeline);
        if !gated {
            Dropped::Mesh.note();
        }
        let ran = gated
            && self.prepare_for(list, false, true, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    list.op(OpKind::Mesh, self.host_key);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(x, 1, 1) };
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => {
                    Dropped::Mesh.note();
                    false
                }
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles in a geometry stage.
    fn rect_gs(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rect_gs)
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            } else if heap_pass_tabled() {
                // Not the heap buffer pass's to make for a while.
                host.buffer_slot_tabled(index);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        let mut oob: Vec<u8> = Vec::new();
        for u in used.iter().filter(tables) {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let Some(end) = table.checked_add(u64::from(u.table_offset_dwords) * 4) else {
                oob.push(u.slot);
                if !fresh.contains(&u.slot) {
                    fresh.push(u.slot);
                }
                continue;
            };
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                oob.push(u.slot);
                if !fresh.contains(&u.slot) {
                    fresh.push(u.slot);
                }
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let must_relocate = oob.contains(&slot);
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                if must_relocate {
                    made.push((slot, u32::MAX));
                }
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                if must_relocate {
                    made.push((slot, u32::MAX));
                }
                continue;
            }
            let Some(host) = heap.host() else {
                if must_relocate {
                    made.push((slot, u32::MAX));
                }
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                if must_relocate {
                    made.push((slot, u32::MAX));
                }
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                }
                (table, n, h.finish())
            };
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // With `VICTORY_HEAP_PASS_TABLED=1` a small table's slots are its
            // own, named or not (below), and the heap buffer pass leaves them
            // to the draw: the buffer views among them the shader does not
            // name are brought up to date here too, as the pass would have
            // (`pass_refresh`: a refresh only, no state and no write).
            let unnamed_tabled = heap_pass_tabled() && n <= 64 && title_first.is_some();
            let named = |k: u32| entries.iter().any(|u| u.table_offset_dwords / 8 == k);
            if let Some(&first) = self.fresh_tables.get(&key) {
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in 0..n {
                    let picked = selected.binary_search(&k).is_ok();
                    if named(k) || !(picked || unnamed_tabled) {
                        continue;
                    }
                    let Some(view) = crate::descriptor::view(table + u64::from(k) * 32) else {
                        continue;
                    };
                    if !picked {
                        pass_refresh(list, &view);
                    } else if let Some(kind) = view_kind(&view) {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // With `VICTORY_HEAP_PASS_TABLED=1` they are such a table's,
            // whatever they hold: the heap buffer pass leaves them to the
            // draw. (Of a big one, which may lie over slots shaders pick by
            // index, only the ones the shader names.)
            if heap_pass_tabled()
                && let Some(title) = title_first
            {
                for k in (0..n).filter(|&k| n <= 64 || named(k)) {
                    host.buffer_slot_tabled(title + k);
                }
            }
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        if address == 0 || size == 0 {
                            host.constant_at(index, 0, 0);
                            continue;
                        }
                        let size = crate::descriptor::constant_staging_bytes(size);
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.constant_at(index, 0, 0);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        } else if unnamed_tabled {
                            // An unnamed slot of a small table, which the
                            // heap buffer pass leaves to the draw (above).
                            pass_refresh(list, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        list.end_mesh_batch();
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with a rectangle geometry stage takes the title's
        // vertices as they are.
        let rects = self.topology == RECTANGLES && !self.rect_gs();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            list.op(OpKind::Draw, self.host_key);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            Dropped::Other.note();
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let Some((from, src_bytes, dst_bytes)) = rect_vertex_span(v, first, count, rects) else {
                once!(slot, "rectangle vertices outside their buffer");
                Dropped::Other.note();
                return false;
            };
            if crate::shadow::find(from, src_bytes).is_none() {
                once!(slot, "rectangle vertices outside their buffer");
                Dropped::Other.note();
                return false;
            }
            let Some((gpu, to)) = list.scratch(u64::from(dst_bytes)) else {
                Dropped::Other.note();
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, v.stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading_range(from, n * 3 * stride);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: dst_bytes,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        list.end_mesh_batch();
        self.window = None;
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Rectangles run only where a geometry stage completes them: it
        // takes each three indices' vertices as the console does.
        let rects = self.topology == RECTANGLES && !self.rect_gs();
        if rects {
            once!("indexed rectangles not drawn");
            Dropped::Other.note();
        }
        let ran = !rects && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            list.op(OpKind::Indexed, self.host_key);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        if crate::indirect::executes_nothing(max) {
            return true;
        }
        // Skip before the argument clamp can mutate the indirect buffer.
        if crate::pipeline::is_skipped(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Rectangles run only where a geometry stage completes them: an
        // indirect draw's vertex range is not known here to expand.
        if matches!(kind, DRAW | INDEXED) && self.topology == RECTANGLES && !p.rect_gs {
            why("rectangles without a geometry stage");
            return false;
        }
        let matches_kind = match kind {
            DRAW | INDEXED => !p.compute && !p.mesh,
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        // A dispatch that may have more groups than the host runs in one
        // dimension, by a program that takes the number of its first group.
        let parts = if kind == DISPATCH && max == 1 && count.is_none() {
            group_base(p).zip(host_signature(DISPATCH, crate::indirect::PART_BYTES as u32))
        } else {
            None
        };
        let windowed = windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        if !windowed || list.first_use(WINDOW_USE) {
            self.window = None;
        }
        // A one-record mesh draw joins the list's batch of them
        // (`crate::indirect::grouped`): its record is clamped with the
        // batch's, by the clamp recorded before the batch's first draw.
        // Any other execution ends the batch first.
        let grouped = crate::indirect::grouped()
            && kind == MESH
            && max == 1
            && count.is_none()
            && stride == crate::indirect::MESH_STRIDE
            && !p.stages.iter().any(|stage| stage.translated().stores)
            && offset.is_multiple_of(4)
            && offset <= u64::from(u32::MAX - stride);
        args.handed(
            if grouped {
                Hand::MeshBatch
            } else {
                Hand::Arguments
            },
            arguments,
        );
        if let Some((s, at)) = &counter {
            s.handed(Hand::Count, *at);
        }
        let batched = if grouped {
            args.refresh_from(list, arguments, len, Source::Indirect);
            list.buffer_read(&args.buffer);
            // It marks what it records (a batch's end and clamp).
            let _span = list.span_held();
            crate::indirect::grouped_mesh(list, args.id(), &args.buffer, offset as u32)
        } else {
            None
        };
        let in_batch = batched.is_some();
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = if let Some((bounded, at, opened)) = batched {
            // The batch's clamp wrote the scratch buffer a window reads.
            self.window = None;
            if opened {
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
            }
            Some((bounded, at))
        } else {
            list.end_mesh_batch();
            self.window
                .as_ref()
                .filter(|w| same(w) && offset >= w.base)
                .filter(|w| {
                    (offset - w.base) % u64::from(w.step) == 0
                        && (offset - w.base) / u64::from(w.step) < u64::from(w.records)
                })
                .map(|w| (w.scratch.clone(), offset - w.base))
        };
        let hit_window = hit.is_some() && !in_batch;
        // Whether a batch of clamps took the execution
        // (`crate::indirect::batch`): its scratch buffer stays as the batch
        // left it.
        let mut clamp_batched = false;
        let (bounded, at) = match hit {
            Some(hit) => hit,
            None => 'bounded: {
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                args.refresh_from(
                    list,
                    arguments,
                    len.max(u64::from(step) * u64::from(records)),
                    Source::Indirect,
                );
                list.buffer_read(&args.buffer);
                if let Some((s, at)) = &counter {
                    s.refresh_from(list, *at, 4, Source::Indirect);
                    list.buffer_read(&s.buffer);
                }
                // Bounded first: a clamp binds a compute pipeline of its own
                // and copies clamped active records into a scratch buffer, at
                // offset 0 for a clamp of the execution's own, at the
                // execution's slot for a batch of clamps.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // The list's batch of clamps takes it, or opens with it
                // (`crate::indirect::batch`); not a window's.
                let batched = (crate::indirect::batch() && !windowed)
                    .then(|| {
                        let src = (&args.buffer, offset);
                        crate::indirect::bound_batched(
                            list,
                            src,
                            count_arg,
                            (kind, step, records),
                            parts.is_some(),
                        )
                    })
                    .flatten();
                if let Some((scratch, at, opened)) = batched {
                    if opened {
                        self.host_pipeline = 0;
                        self.host_roots[1] = 0;
                    }
                    self.window = None;
                    clamp_batched = true;
                    break 'bounded (scratch, at);
                }
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = {
                    let _span = list.spanned(Span::Clamp);
                    if parts.is_some() {
                        crate::indirect::bound_split(list, (&args.buffer, offset))
                    } else if windowed && records > 1 {
                        crate::indirect::bound_ahead(
                            list,
                            (&args.buffer, offset),
                            (kind, step, records),
                        )
                    } else {
                        crate::indirect::bound(
                            list,
                            (&args.buffer, offset),
                            count_arg,
                            (kind, step, records),
                        )
                    }
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        let ran = match kind {
            // (Not counted by reason: the drops of `DROPPED` are the direct
            // calls'; these are counted as indirect in the frame line.)
            DRAW | INDEXED => self.prepare_for(list, kind == INDEXED, false, false),
            MESH => self.prepare_for(list, false, true, false),
            DISPATCH => self.bind(list, p, true).is_ok(),
            _ => false,
        };
        if !ran {
            // A window hit did not run `bound`/`bound_ahead`, so `bounded` is
            // still in `COMMON` (from the previous draw's `done`); emitting
            // `done` (`INDIRECT_ARGUMENT -> COMMON`) without an `ExecuteIndirect`
            // implicit promotion would be a mismatched barrier. (A scratch
            // buffer the list keeps readable, `VICTORY_INDIRECT_READABLE=1`,
            // is in `INDIRECT_ARGUMENT` either way and `done` leaves it so.)
            // A batch's draws leave that to the end of the batch.
            if !hit_window && !in_batch && !clamp_batched {
                crate::indirect::done(list, &bounded);
            }
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        list.op(OpKind::Indirect, self.host_key);
        if let Some((s, _)) = &counter {
            list.buffer_read(&s.buffer);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            if let Some(((param, dword), signature)) = &parts {
                for part in 0..crate::indirect::SPLIT {
                    list.list.SetComputeRoot32BitConstant(*param, part * crate::indirect::PART, *dword);
                    let at = at + u64::from(part) * crate::indirect::PART_BYTES;
                    list.list.ExecuteIndirect(signature, 1, &bounded, at, None, 0);
                }
                list.list.SetComputeRoot32BitConstant(*param, 0, *dword);
            } else {
                list.list.ExecuteIndirect(
                    &signature,
                    max,
                    &bounded,
                    at,
                    counter.as_ref().map(|(s, _)| &s.buffer),
                    counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
                )
            }
        };
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // Back to the common state for the next call, unless the list
        // keeps the scratch buffer readable as arguments
        // (`VICTORY_INDIRECT_READABLE=1`): then the next draw of the window
        // reads it with no barrier, and the next clamp moves it back. A
        // batch's draws read it in that state until the batch ends
        // (`HostList::end_mesh_batch`), which does this once for them all,
        // and a batch of clamps keeps its scratch buffer readable until the
        // next one's clamp or the list's close.
        if !in_batch && !clamp_batched {
            crate::indirect::done(list, &bounded);
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// Where a compute pipeline's program takes the number of its first thread
/// group (`victory_shader::emit::USER_GROUP_BASE`): the root parameter of
/// its user data and the dword in it.
fn group_base(p: &crate::pipeline::host::HostPipeline) -> Option<(u32, u32)> {
    let stage = p.stages.first().filter(|_| p.compute)?;
    let (param, _) = stage.user_data?;
    let base = RootType::Other(victory_shader::emit::USER_GROUP_BASE);
    let word = stage
        .translated()
        .host
        .user_data
        .iter()
        .find(|u| u.root == base)?;
    Some((param, word.index))
}

/// Computes `(src_address, src_bytes, expanded_dst_bytes)` for rectangle
/// expansion over `first..first + count`, rejecting overflows, out-of-bounds
/// views, and expanded sizes exceeding `u32::MAX`.
fn rect_vertex_span(v: VertexView, first: u32, count: u32, rects: u64) -> Option<(u64, u64, u32)> {
    let stride = u64::from(v.stride);
    if v.address == 0 || stride == 0 {
        return None;
    }
    let total_vertices = u64::from(first) + u64::from(count);
    let end_bytes = total_vertices.checked_mul(stride)?;
    if end_bytes > u64::from(v.size) {
        return None;
    }
    let from = u64::from(first)
        .checked_mul(stride)
        .and_then(|off| v.address.checked_add(off))?;
    let src_bytes = rects.checked_mul(3)?.checked_mul(stride)?;
    let dst_bytes = rects.checked_mul(6)?.checked_mul(stride)?;
    let dst_bytes = u32::try_from(dst_bytes).ok()?;
    Some((from, src_bytes, dst_bytes))
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
pub(crate) fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// What the heap buffer pass does for a buffer view it makes, done by a
/// draw for a slot the pass leaves to it (`VICTORY_HEAP_PASS_TABLED=1`): the
/// view's range of its host copy brought up to date, nothing else. Not
/// [`readable`]: for an unordered access view that would change the
/// buffer's state and declare the shaders' writes (a snapshot and a
/// copy-back), for a slot the shader may never touch.
fn pass_refresh(list: &HostList, view: &View) {
    if let Some((shadow, _, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh_from(list, address, len, Source::HeapPass);
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, resource, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh_from(list, address, len, Source::Table);
        // The resource the slot shows, whose state the view needs.
        let resource = resource.as_ref().unwrap_or(&shadow.buffer);
        if kind == ResourceType::Uav {
            list.buffer_uav(resource);
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        } else {
            list.buffer_read(resource);
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

type HeapRangeMap =
    std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>;

/// What the heap buffer pass keeps of a heap: the last frame its buffer
/// views were looked at, its pass counter, and the host copy and range each
/// slot's view reads.
pub(crate) struct HeapRanges {
    last_frame: u64,
    passes: u64,
    ranges: HeapRangeMap,
}

impl HeapRanges {
    /// Before the heap's first pass.
    pub(crate) fn new() -> Self {
        Self {
            last_frame: u64::MAX,
            passes: 0,
            ranges: HeapRangeMap::new(),
        }
    }
}

/// Per host view heap (keyed by monotonic [`crate::descriptor::host::HostHeap::id`]):
/// the last frame its buffer views were scanned, its pass counter, and the host
/// shadow ranges cached across passes. Entries are removed on `HostHeap::drop`.
static HEAP_RANGES: std::sync::Mutex<Option<std::collections::HashMap<u64, HeapRanges>>> =
    std::sync::Mutex::new(None);

/// Whether the heap buffer pass of a frame is claimed per heap before
/// [`HEAP_RANGES`] is taken ([`claim_pass`]); `VICTORY_HEAP_CLAIM=0`: every
/// list that binds a shader picking buffers from the heap takes the lock
/// first, and waits there for the whole walk to learn that the frame's
/// pass ran, as before.
fn heap_claim() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_CLAIM").is_none_or(|v| v != "0"))
}

/// Whether a list that lost the claim of the heap buffer pass
/// ([`claim_pass`]) waits for the walk when it is submitted
/// (`crate::host::wait_for_walks`). `VICTORY_HEAP_WAIT_AT_SUBMIT=0` does
/// not take the wait away, it moves it: the list waits while it records,
/// at the bind that lost, until the walk has ended (the lock had it wait
/// there).
fn heap_wait_at_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_WAIT_AT_SUBMIT").is_none_or(|v| v != "0"))
}

/// Passes after which every buffer slot of the heap was looked at again
/// by the heap buffer pass, whether no write changed it: a view of a
/// released resource goes stale without a write, and a slot written before
/// its memory had a host copy is only made then.
const HEAP_ROUND: u64 = 16;

/// Whether the heap buffer pass looks again at a part of the slots no
/// write changed at every pass, each slot at every [`HEAP_ROUND`]th
/// (`VICTORY_HEAP_SPREAD=1`), instead of at all of them at every
/// [`HEAP_ROUND`]th pass (off, the default, as before). A slot is looked
/// at as often either way, and the first pass looks at all of them. Ours
/// measured 200-720 ms for a whole pass on a thread that records the frame
/// (`docs/perf-notes/heap-pass.md`); cool_fox measured some 115 ms for one
/// of 72,000 slots, against 11 ms for the others.
fn heap_spread() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SPREAD").is_some_and(|v| v == "1"))
}

/// Threads the walk of the heap buffer pass runs on, the one that binds
/// among them (`VICTORY_HEAP_PASS_THREADS`, 2 to 16, in jobs of
/// [`HEAP_WALK_JOB`] slots on threads of the pass's own, made for the
/// walk: not the ones of any other work, so it waits for none). Unset, 0 or
/// 1, the default: on the thread that binds alone, as before. A walk of
/// one job or less runs there either way.
fn heap_pass_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_HEAP_PASS_THREADS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(1, |n| n.clamp(1, 16))
    })
}

/// Whether the heap buffer pass leaves the slots draws read through their
/// tables within the last few frames to those draws
/// (`VICTORY_HEAP_PASS_TABLED=1`; off, the default: it makes them and
/// brings up to date the copies they show, as before). Such a draw makes
/// what its table holds and brings what it reads up to date itself
/// (`table_entry`, `tables`), so for the title's per-draw tables the pass
/// did the same work again. With the switch on, the draws mark the buffer
/// slots of their tables too, not only the texture slots, in marks of the
/// pass's own: the heap sync and the queue of textures picked by index do
/// not read them, so a record the title copies into such a slot still
/// reaches the set of changes, and the pass makes it once no table held
/// the slot for a few frames
/// ([`crate::descriptor::host::HostHeap::buffer_slot_tabled`]).
fn heap_pass_tabled() -> bool {
    #[cfg(test)]
    if TABLED_HERE.with(std::cell::Cell::get) {
        return true;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_TABLED").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// Tests: [`heap_pass_tabled`] on for this thread.
    static TABLED_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether what the heap buffer pass keeps of a heap lies on the heap
/// under a lock of its own (`VICTORY_HEAP_SWEEP_OWN=1`,
/// [`crate::descriptor::host::HostHeap::buffer_sweep`]) and goes with it,
/// instead of in [`HEAP_RANGES`] under one lock for every heap (off, the
/// default, as before): the passes of two heaps in one frame do not wait
/// for each other.
fn heap_sweep_own() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SWEEP_OWN").is_some_and(|v| v == "1"))
}

/// Slots of a job of the walk of the heap buffer pass on several threads
/// ([`heap_pass_threads`]).
const HEAP_WALK_JOB: usize = 256;

/// Which slots a pass of the heap buffer pass looks at
/// ([`pass_slots`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassSlots {
    /// Every slot that holds a buffer view; what was kept is forgotten.
    All,
    /// The slots changed since the last pass.
    Changed,
    /// Those, and the slots of this part of [`HEAP_ROUND`] parts
    /// ([`heap_spread`]); what was kept of that part is forgotten.
    ChangedAndPart(u32),
}

/// The slots pass number `passes` (from 0) of a heap looks at; `spread`:
/// [`heap_spread`].
fn pass_slots(passes: u64, spread: bool) -> PassSlots {
    if passes == 0 || (!spread && passes.is_multiple_of(HEAP_ROUND)) {
        PassSlots::All
    } else if spread {
        PassSlots::ChangedAndPart((passes % HEAP_ROUND) as u32)
    } else {
        PassSlots::Changed
    }
}

/// Forgets what `ranges` holds of the slots of part `part` of `parts`
/// ([`crate::descriptor::host::HostHeap::buffer_slots_of`]).
fn forget_part<T>(ranges: &mut std::collections::HashMap<u32, T>, part: u32, parts: u32) {
    ranges.retain(|&slot, _| slot % parts != part);
}

/// What the walk of the heap buffer pass keeps of a slot: the host copy its
/// buffer view shows, and the range of it the view reads.
type HeapRange = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// Ranges of one copy the heap buffer pass refreshes apart only if this many
/// bytes or more lie between them: the 4 KiB parts a refresh takes.
const HEAP_PASS_JOIN: u64 = 4 << 10;

/// `ranges` (start, end) sorted, and joined where they overlap or less than
/// `gap` bytes lie between them.
fn join_near(ranges: &mut Vec<(u64, u64)>, gap: u64) {
    ranges.sort_unstable();
    ranges.dedup_by(|next, previous| {
        if next.0 >= previous.1.saturating_add(gap) {
            return false;
        }
        previous.1 = previous.1.max(next.1);
        true
    });
}

/// The copy and range the view `view` of title slot `index` shows
/// (`buffer_copy`, not the one the view's range alone would find), with
/// the ticks taken added to `copy_ticks` when `timed`.
fn walked_copy(
    host: &crate::descriptor::host::HostHeap,
    index: u32,
    view: &View,
    timed: bool,
    copy_ticks: &mut u64,
) -> Option<HeapRange> {
    let copy_began = if timed { crate::profile::stamp() } else { 0 };
    let copy = crate::descriptor::host::buffer_copy(view);
    *copy_ticks += crate::profile::elapsed(copy_began);
    let (shadow, resource, address, len) = copy?;
    #[cfg(feature = "diagnostics")]
    if host.shows(index, resource.as_ref().unwrap_or(&shadow.buffer)) == Some(false) {
        HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = (host, index, resource);
    Some((shadow, address, address + len))
}

/// What the walk on several threads found ([`walk_spread`]).
struct SpreadWalk {
    /// Per slot looked at, in the order given: what to keep of it.
    found: Vec<(u32, Option<HeapRange>)>,
    /// Ticks of the copies looked up, and of the views made with their
    /// count (summed by each job's thread while the profile records).
    copy_ticks: u64,
    make_ticks: u64,
    made: u64,
    /// The device went in this job: why, and the job's number.
    removed: Option<(String, usize)>,
}

/// The walk of the heap buffer pass over `slots` (with their records'
/// `handles`) in jobs of [`HEAP_WALK_JOB`] slots on `threads` threads, this
/// one among them, as the walk on one thread does it: each slot is
/// mirrored (`mirror_walked`; slots of different jobs at once, which a
/// mirror allows from any thread) and its copy looked up; the caller keeps
/// what was found, in slot order. `tabled`: the frame for
/// [`heap_pass_tabled`]. `alive`: the device was there; it is asked about
/// after each job (a cheap query), and the first job after which it went is
/// named.
fn walk_spread(
    host: &crate::descriptor::host::HostHeap,
    slots: &[u32],
    handles: &[Option<u64>],
    threads: usize,
    tabled: Option<u64>,
    timed: bool,
    alive: bool,
) -> SpreadWalk {
    let jobs = slots.len().div_ceil(HEAP_WALK_JOB);
    let found: Vec<std::sync::OnceLock<Vec<(u32, Option<HeapRange>)>>> =
        (0..jobs).map(|_| std::sync::OnceLock::new()).collect();
    let ticks = [const { AtomicU64::new(0) }; 3];
    let removed = std::sync::Mutex::new(None::<(String, usize)>);
    let gone = std::sync::atomic::AtomicBool::new(!alive);
    each_on(threads, jobs, "victory-heap-walk", &|job| {
        let at = job * HEAP_WALK_JOB;
        let end = slots.len().min(at + HEAP_WALK_JOB);
        // A span of its own on this thread sums the views it makes.
        let span = if timed {
            crate::profile::span_args("heap walk job", &[("slots", (end - at) as u64)])
        } else {
            None
        };
        if span.is_some() {
            crate::profile::parts_begin();
        }
        let mut copy_ticks = 0;
        let mut skipped = 0;
        let looked: Vec<_> = (at..end)
            .map(|n| {
                let index = slots[n];
                let range = handles[n].and_then(|handle| {
                    if tabled.is_some_and(|frame| host.buffer_slot_tabled_lately(index, frame)) {
                        skipped += 1;
                        return None;
                    }
                    let view = host.mirror_walked(index, handle)?;
                    walked_copy(host, index, &view, timed, &mut copy_ticks)
                });
                (index, range)
            })
            .collect();
        if span.is_some() {
            let (make, made) = crate::profile::parts_end();
            ticks[1].fetch_add(make, Ordering::Relaxed);
            ticks[2].fetch_add(made, Ordering::Relaxed);
        }
        drop(span);
        ticks[0].fetch_add(copy_ticks, Ordering::Relaxed);
        if skipped > 0 {
            HEAP_PASS_PARTS[2].fetch_add(skipped, Ordering::Relaxed);
        }
        if !gone.load(Ordering::Relaxed)
            && let Some(why) = crate::descriptor::host::device_removed()
            && !gone.swap(true, Ordering::Relaxed)
        {
            *removed.lock().unwrap_or_else(|e| e.into_inner()) = Some((why, job));
        }
        let _ = found[job].set(looked);
    });
    let [copy_ticks, make_ticks, made] = ticks.map(AtomicU64::into_inner);
    SpreadWalk {
        found: found
            .into_iter()
            .filter_map(std::sync::OnceLock::into_inner)
            .flatten()
            .collect(),
        copy_ticks,
        make_ticks,
        made,
        removed: removed.into_inner().unwrap_or_else(|e| e.into_inner()),
    }
}

/// Calls `f` with every number below `count`, on this thread and on up to
/// `threads - 1` more made for the call, and returns when every call has.
/// A panic of one call is the caller's once all have ended. With fewer than
/// two threads or numbers, on this thread alone. The threads are named
/// `name-1` on. (Also the heap sync's.)
pub(crate) fn each_on(threads: usize, count: usize, name: &str, f: &(dyn Fn(usize) + Sync)) {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let run = || {
        loop {
            let n = next.fetch_add(1, Ordering::Relaxed);
            if n >= count {
                break;
            }
            f(n);
        }
    };
    if threads < 2 || count < 2 {
        run();
        return;
    }
    std::thread::scope(|s| {
        for i in 1..threads.min(count) {
            // (A thread that cannot be made is only missed: the others
            // take its numbers.)
            let _ = std::thread::Builder::new()
                .name(format!("{name}-{i}"))
                .spawn_scoped(s, run);
        }
        run();
    });
}

#[cfg(test)]
thread_local! {
    /// Called by the walk of the heap buffer pass on this thread after
    /// each view it mirrors: a test parks the walk there.
    static WALK_HOOK: std::cell::RefCell<Option<Box<dyn FnMut()>>> = const { std::cell::RefCell::new(None) };
}

/// Claims the heap buffer pass of `frame` on `gate`'s heap, before any
/// lock: the walk to run (`Some`; it counts as ended when dropped), or
/// `None` for a list whose frame's pass is claimed already. Such a list
/// returns at once instead of waiting for the walk's lock: with
/// `at_submit`, the walk it lost to goes to `defer` (the list's, waited
/// for when it is submitted); else it is waited for here, at its bind,
/// as the lock had it wait, but not for the rest of the pass nor for
/// other heaps' walks.
fn claim_pass(
    gate: &std::sync::Arc<crate::descriptor::host::WalkGate>,
    frame: u64,
    at_submit: bool,
    defer: impl FnOnce(u64),
) -> Option<crate::descriptor::host::Walking> {
    match gate.claim(frame) {
        Ok(walking) => Some(walking),
        Err(mark) => {
            if at_submit {
                if !gate.walked(mark) {
                    defer(mark);
                }
            } else {
                gate.wait(mark, false);
            }
            None
        }
    }
}

/// Drops cached dynamic-buffer range state (and its held `Arc<Shadow>` refs)
/// when a `HostHeap` is destroyed.
pub(crate) fn forget_heap(id: u64) {
    if let Some(map) = HEAP_RANGES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        map.remove(&id);
    }
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

/// A title view heap with buffer views for tests that run the heap buffer
/// pass ([`State::heap_buffers`]).
#[cfg(test)]
pub(crate) mod test_heap {
    use super::*;

    /// The bytes of the title buffer [`ViewHeap::new`] makes.
    pub(crate) const VIEW_MEMORY: usize = 1 << 17;

    /// A title view heap of eight slots, and a title buffer whose views the
    /// slots may hold. The heap is released on drop, also when an assertion
    /// fails, and the buffer with it if the heap made it.
    pub(crate) struct ViewHeap {
        pub(crate) h: This,
        /// The title buffer's start and bytes.
        pub(crate) memory: usize,
        size: usize,
        /// Whether the heap made (and registered) the buffer.
        own: bool,
    }

    impl ViewHeap {
        /// The heap over a title buffer in memory of its own
        /// ([`VIEW_MEMORY`] bytes, write-watched or not; the memory is left
        /// to the process).
        pub(crate) fn new(watched: bool) -> ViewHeap {
            // SAFETY: new memory, left to the process.
            let memory = unsafe {
                use windows_sys::Win32::System::Memory::{
                    MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
                };
                const MEM_WRITE_WATCH: u32 = 0x20_0000;
                let watch = if watched { MEM_WRITE_WATCH } else { 0 };
                VirtualAlloc(
                    std::ptr::null(),
                    VIEW_MEMORY,
                    MEM_RESERVE | MEM_COMMIT | watch,
                    PAGE_READWRITE,
                ) as usize
            };
            assert_ne!(memory, 0);
            crate::shadow::register(memory, memory as u64, VIEW_MEMORY as u64);
            let mut heap = ViewHeap::over(memory, VIEW_MEMORY);
            heap.own = true;
            heap
        }

        /// The heap over the title buffer `[memory, memory + size)`, which
        /// the caller registered and unregisters.
        pub(crate) fn over(memory: usize, size: usize) -> ViewHeap {
            use crate::descriptor::{HeapDesc, create_heap};
            use crate::object::{S_OK, guid};
            #[repr(C)]
            struct Extended(HeapDesc, [u64; 2]);
            let desc = HeapDesc {
                kind: 0,
                count: 8,
                flags: 1,
                nodes: 1,
            };
            let iid = guid(
                0x8efb471d,
                0x616c,
                0x4f49,
                [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
            );
            let extended = Extended(desc, [0; 2]);
            let mut h: This = std::ptr::null();
            assert_eq!(
                create_heap(std::ptr::null(), &extended.0, &iid, &mut h),
                S_OK
            );
            ViewHeap {
                h,
                memory,
                size,
                own: false,
            }
        }

        pub(crate) fn host(&self) -> &'static crate::descriptor::host::HostHeap {
            crate::descriptor::heap(self.h)
                .host()
                .expect("host view heap")
        }

        /// A view of 64 elements of 12 bytes of the buffer, from element
        /// `first` (byte `12 * first`).
        pub(crate) fn view(&self, first: u32) -> View {
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, 5768, 0, first, 0, 64, 12, 0, 0]);
            (desc[10], desc[11]) = (self.memory as u32, (self.memory as u64 >> 32) as u32);
            let width = self.size as u32;
            View::Placed(
                false,
                [1, 0, 0, 0, width, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
                desc,
            )
        }

        /// [`ViewHeap::view`] for unordered access.
        pub(crate) fn uav(&self, first: u32) -> View {
            let mut desc = [0u32; 16];
            desc[..10].copy_from_slice(&[0, 1, first, 0, 64, 12, 0, 0, 0, 0]);
            (desc[10], desc[11]) = (self.memory as u32, (self.memory as u64 >> 32) as u32);
            let width = self.size as u32;
            View::Placed(
                true,
                [1, 0, 0, 0, width, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
                desc,
            )
        }

        /// Slot `slot` holds [`ViewHeap::view`] of `first`.
        pub(crate) fn put(&self, slot: u32, first: u32) {
            let handle = crate::descriptor::heap(self.h).handle(slot).unwrap();
            crate::descriptor::record(handle, self.view(first));
        }

        /// The buffer's host copy.
        pub(crate) fn copy(&self) -> std::sync::Arc<crate::shadow::Shadow> {
            crate::shadow::find(self.memory as u64, self.size as u64).expect("the buffer's copy")
        }

        /// A root shader resource view of title `address` bound in `list`
        /// as a one-view table of the heap ([`State::root_srv_table`]), under
        /// a root signature of one such table: the state, whose tables are
        /// given back with `tables_done`, and the root signature, which must
        /// live until the list is reset.
        pub(crate) fn root_srv(
            &self,
            host: &crate::host::Host,
            list: &HostList,
            address: u64,
        ) -> (State, ID3D12RootSignature) {
            // A root signature of one table of one view, for the bind.
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                ..Default::default()
            };
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &param,
                ..Default::default()
            };
            let mut blob = None;
            // SAFETY: the description and what it points to live through the
            // call; the out pointer is ours.
            unsafe {
                D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
            }
            .expect("a root signature");
            let blob = blob.unwrap();
            // SAFETY: the blob's own bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                )
            };
            // SAFETY: a serialized root signature.
            let root: ID3D12RootSignature =
                unsafe { host.device.CreateRootSignature(0, bytes) }.unwrap();
            // SAFETY: an open list; our heap and root signature, which live
            // until the list is reset.
            unsafe {
                list.list
                    .SetDescriptorHeaps(&[Some(self.host().heap.clone())]);
                list.list.SetComputeRootSignature(&root);
            }
            let mut state = State::default();
            state.heaps[0] = self.h as usize;
            assert!(state.root_srv_table(list, Some(address), 0, true, false));
            (state, root)
        }

        /// The heap buffer pass of a list that binds the heap, in `frame`.
        pub(crate) fn pass(&self, list: &HostList, frame: u64) {
            let mut state = State::default();
            state.heaps[0] = self.h as usize;
            assert!(state.heap_buffers_in(list, frame).is_some());
        }
    }

    impl Drop for ViewHeap {
        fn drop(&mut self) {
            use crate::object::slot;
            slot::<extern "system" fn(This) -> u32>(self.h, 0x10)(self.h);
            if self.own {
                crate::shadow::unregister(self.memory, self.memory as u64);
            }
        }
    }

    /// A frame of its own for a test's heap pass: the pass runs once per
    /// frame per heap.
    pub(crate) fn test_frame() -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(1 << 40);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// A table that found no slots drops the draw as "table full"; a draw
    /// without a host pipeline is counted by that reason, for a draw, a
    /// mesh dispatch and a dispatch alike (`VICTORY_D3D12_WARP=1`).
    #[test]
    fn a_dropped_draw_is_counted_by_its_reason() {
        let _census = crate::census::lock_tests();
        use crate::census::DROPPED_BY;
        assert_eq!(table_refused(&[(0, 64), (1, 0)]), None);
        assert_eq!(
            table_refused(&[(0, 64), (1, u32::MAX)]),
            Some(Dropped::TableFull)
        );
        let table = DROPPED_BY[Dropped::TableFull as usize].load(Ordering::Relaxed);
        table_refused(&[(1, u32::MAX)]).unwrap().note();
        assert_eq!(
            DROPPED_BY[Dropped::TableFull as usize].load(Ordering::Relaxed),
            table + 1
        );
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let none = || DROPPED_BY[Dropped::NoPipeline as usize].load(Ordering::Relaxed);
        let list = host.new_list().unwrap();
        let mut state = State::default();
        let was = none();
        assert!(
            !state.prepare_for(&list, false, false, true),
            "no pipeline set"
        );
        let quiet = none();
        assert!(
            !state.prepare_for(&list, false, false, false),
            "no pipeline set"
        );
        assert_eq!(none(), quiet, "not counted when told not to");
        assert_eq!(none(), was + 1, "a draw");
        state.dispatch_mesh(&list, 1);
        assert_eq!(none(), was + 2, "a mesh dispatch");
        state.dispatch(&list, 1, 1, 1);
        assert_eq!(none(), was + 3, "a dispatch");
        list.close();
        list.reset();
    }

    /// One of our root signatures needs a sampler heap bound before it is
    /// set: ours stands in for a title that binds none, which a compute list
    /// that only reads views does.
    #[test]
    fn a_list_without_a_sampler_heap_gets_ours() {
        // The title binds both: nothing of ours.
        assert_eq!(State::wanted([0x10, 0x20]), [0x10, 0x20]);
        // Views only: our sampler heap after them.
        assert_eq!(State::wanted([0x10, 0]), [0x10, SAMPLERS_OURS]);
        // Neither: our sampler heap alone.
        assert_eq!(State::wanted([0, 0]), [0, SAMPLERS_OURS]);
        // A title sampler heap stays without views.
        assert_eq!(State::wanted([0, 0x20]), [0, 0x20]);
    }

    /// A root constant view 256 bytes before a 4 MiB line of memory outside
    /// buffers: the window its address names ends 256 bytes on, while a
    /// constant buffer reads up to 64 KiB. It is bound from a window that
    /// holds all of that; a root shader resource view keeps the window its
    /// address names (it is bound as a bounded table). With
    /// `VICTORY_ROOT_CBV_REACH=0` the first assertion fails (WARP).
    #[cfg(windows)]
    #[test]
    fn warp_a_root_constant_view_at_a_windows_end_is_bound_from_a_window_that_holds_its_reach() {
        let Some(_) = crate::host::host() else {
            return;
        };
        let short = ROOT_CBV_SHORT[0].load(Ordering::Relaxed);
        // Memory outside buffers that no other test asks for.
        let address = 0x7f_3c00_0000u64 + (4 << 20) - 256;
        let named = crate::shadow::find(address, 1).expect("a window of memory outside buffers");
        assert!(named.is_window());
        assert_eq!(named.left(address), 256);
        let bound = root_view_copy(address, false).expect("a window");
        assert!(
            bound.left(address) >= ROOT_CBV_REACH,
            "a root constant view is bound with {} bytes of its copy left, less than a constant buffer reads",
            bound.left(address)
        );
        assert_eq!(bound.gpu(address) % 256, address % 256);
        assert!(
            ROOT_CBV_SHORT[0].load(Ordering::Relaxed) > short,
            "the short window is counted"
        );
        let srv = root_view_copy(address, true).expect("a window");
        assert!(std::sync::Arc::ptr_eq(&srv, &named));
    }

    /// Root constant views in a title buffer of 0x38400 bytes (the size of
    /// the copy just below the fault of a device removal): one 256 bytes
    /// before its end, one at its start, and one 240 bytes before its end,
    /// not on 256 bytes in the copy. A host root signature makes a title's
    /// root constant view a one-view table, and each is bound as one whose
    /// view holds what the copy has from its address on: 256 bytes, the 64
    /// KiB a view may, and 256 bytes of the list's scratch with the title's
    /// 240 in front. On the card a shader reads the title's bytes inside the
    /// views and zeros past the short ones, where the title's memory goes on
    /// but the copy ends (through a root descriptor, which has no size, such
    /// a read reaches whatever the host placed after the copy). Binding
    /// again in the list takes no new slot, and once the list has run all
    /// its slots are free again. With `VICTORY_ROOT_CBV_TABLES=0` the first
    /// assertion fails: the root signature has a root descriptor (WARP).
    #[cfg(windows)]
    #[test]
    fn warp_a_root_constant_view_holds_only_what_its_copy_has() {
        use crate::descriptor::{HeapDesc, create_heap, heap};
        use crate::object::{S_OK, This, guid, slot};
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        const LEN: usize = 0x38400;
        const MAPPED: usize = 0x40000;
        assert!(
            crate::pipeline::host::tabled_root_view(D3D12_ROOT_PARAMETER_TYPE_CBV, 0),
            "a title's root constant view is a one-view table in a host root signature"
        );
        assert!(!crate::pipeline::host::tabled_root_view(
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            victory_shader::emit::SLOT_STRIDES
        ));
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        // What newer titles pass: the description, then no memory of theirs.
        #[repr(C)]
        struct Extended(HeapDesc, [u64; 2]);
        let desc = HeapDesc {
            kind: 0,
            count: 8,
            flags: 1,
            nodes: 1,
        };
        let iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let extended = Extended(desc, [0; 2]);
        let mut h: This = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &extended.0, &iid, &mut h),
            S_OK
        );
        let views = heap(h).host().expect("host view heap");
        // A title buffer in memory of its own, which goes on past the
        // buffer's end; every dword holds its offset and a mark.
        // SAFETY: new memory, left to the process.
        let memory = unsafe {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            VirtualAlloc(
                std::ptr::null(),
                MAPPED,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(memory, 0);
        let word = |offset: usize| 0x8000_0000 | offset as u32;
        for offset in (0..MAPPED).step_by(4) {
            // SAFETY: inside the new memory.
            unsafe { ((memory + offset) as *mut u32).write(word(offset)) };
        }
        crate::shadow::register(memory, memory as u64, LEN as u64);
        // The title releases its heap and its buffer, also when an
        // assertion fails: later tests find no copy of this memory.
        struct Release(This, usize);
        impl Drop for Release {
            fn drop(&mut self) {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
                crate::shadow::unregister(self.1, self.1 as u64);
            }
        }
        let _release = Release(h, memory);
        let free = views.free_dynamic();
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.heaps[0] = h as usize;
        let near_end = (memory + LEN - 0x100) as u64;
        let (short, bytes) = state.root_cbv(&list, Some(near_end)).expect("a slot");
        assert_eq!(
            bytes, 0x100,
            "the view holds what the copy has from its address on"
        );
        let (whole, bytes) = state.root_cbv(&list, Some(memory as u64)).expect("a slot");
        assert_eq!(bytes, ROOT_CBV_REACH, "the view holds the most a view may");
        let unaligned = (memory + LEN - 0xf0) as u64;
        let (odd, bytes) = state.root_cbv(&list, Some(unaligned)).expect("a slot");
        assert_eq!(
            bytes, 0x100,
            "whole 256s of scratch, the title's 240 bytes in front"
        );
        let (null, bytes) = state.root_cbv(&list, None).expect("a slot");
        assert_eq!(bytes, 0, "a null view");
        // Bound again in the list: the same slots, and no others taken.
        for (address, index, bytes) in [
            (Some(near_end), short, 0x100),
            (Some(unaligned), odd, 0x100),
            (None, null, 0),
        ] {
            assert_eq!(state.root_cbv(&list, address), Some((index, bytes)));
        }
        assert_eq!(
            views.free_dynamic(),
            free - 4,
            "one slot per view in the list"
        );
        // The whole buffer brought up to date in its copy: a read past the
        // whole view that reached the copy would find the title's bytes.
        let copy = crate::shadow::find(memory as u64, LEN as u64).expect("the buffer's copy");
        copy.refresh(&list, memory as u64, LEN as u64);

        // On the card: rows 0 and 15 of the short view, row 16 past it,
        // the last row of the whole one, and a row past it at an index the
        // shader computes (in dword 4), where the copy goes on; rows 0 and
        // 14 of the unaligned view, row 15 past the title's bytes in it and
        // row 16 past it.
        const SOURCE: &str = r#"
#define RS "DescriptorTable(CBV(b0)), DescriptorTable(CBV(b1)), DescriptorTable(CBV(b2)), UAV(u0)"
cbuffer Short : register(b0) { uint4 near_end[4096]; };
cbuffer Whole : register(b1) { uint4 start[4096]; };
cbuffer Odd : register(b2) { uint4 odd[4096]; };
RWByteAddressBuffer o : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint past = o.Load(16);
    o.Store4(0, uint4(near_end[0].x, near_end[15].w, near_end[16].x, start[4095].w));
    o.Store(20, start[past].x);
    o.Store4(32, uint4(odd[0].x, odd[14].w, odd[15].x, odd[16].x));
}
"#;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names outlive the call; the out pointer is ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("bounded"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the test shader compiles");
        let code = code.unwrap();
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: the root signature is in the shader.
        let root: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytes) }.unwrap();
        // SAFETY: the bytecode and root signature live through the call.
        let pso: ID3D12PipelineState = unsafe {
            host.device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
        }
        .unwrap();
        let (out, data) = host.shared_buffer(256).unwrap();
        // Row 4,112: byte 0x10100 of the copy, inside it.
        const PAST: usize = 4096 + 16;
        // SAFETY: the mapping of 256 bytes; nothing runs yet.
        unsafe { (data as *mut u32).add(4).write(PAST as u32) };
        // SAFETY: an open list; our heap, root signature, pipeline and
        // buffers, which live until the list has run.
        unsafe {
            let l = &list.list;
            l.SetDescriptorHeaps(&[Some(views.heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootDescriptorTable(0, views.gpu_handle(short));
            l.SetComputeRootDescriptorTable(1, views.gpu_handle(whole));
            l.SetComputeRootDescriptorTable(2, views.gpu_handle(odd));
            l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: the mapping of 256 bytes; the list has run.
        let read = unsafe { (data as *const [u32; 12]).read_volatile() };
        state.tables_done();
        list.reset();
        assert_eq!(
            read[..4],
            [word(LEN - 0x100), word(LEN - 4), 0, word(0xfffc)],
            "the title's bytes inside the views, zeros past the short one"
        );
        assert_eq!(
            read[5],
            0,
            "a row past the whole view reads zeros where the copy holds {:#x}",
            word(PAST * 16)
        );
        assert_eq!(
            read[8..],
            [word(LEN - 0xf0), word(LEN - 4), 0, 0],
            "the title's bytes in the scratch, zeros past them where its memory holds {:#x}",
            word(LEN)
        );
        assert_eq!(
            views.free_dynamic(),
            free,
            "the list's slots are given back"
        );
    }

    /// Without `VICTORY_HEAP_SPREAD` every slot is looked at at every
    /// [`HEAP_ROUND`]th pass; with it the first pass looks at all, and
    /// then each of the [`HEAP_ROUND`] parts once a round, in turn.
    #[test]
    fn a_spread_pass_looks_at_each_part_once_a_round() {
        for passes in 0..3 * HEAP_ROUND {
            let whole = passes % HEAP_ROUND == 0;
            assert_eq!(
                pass_slots(passes, false),
                if whole {
                    PassSlots::All
                } else {
                    PassSlots::Changed
                },
                "pass {passes}"
            );
        }
        assert_eq!(pass_slots(0, true), PassSlots::All);
        for round in 0..3 {
            let mut parts: Vec<u32> = (1..=HEAP_ROUND)
                .map(|n| match pass_slots(round * HEAP_ROUND + n, true) {
                    PassSlots::ChangedAndPart(part) => part,
                    other => panic!("pass {n} of round {round}: {other:?}"),
                })
                .collect();
            parts.sort_unstable();
            assert_eq!(parts, (0..HEAP_ROUND as u32).collect::<Vec<_>>());
        }
    }

    /// Round by round every slot is looked at again once, whatever was
    /// written between, and what a slot holds is what was found last.
    /// (Ported from cool_fox 2aa8134, on our map of ranges.)
    #[test]
    fn every_slot_is_looked_at_again_in_a_round() {
        const SLOTS: u32 = 1000;
        let mut ranges: std::collections::HashMap<u32, (u64, u64)> = Default::default();
        for slot in (0..SLOTS).step_by(3) {
            ranges.insert(slot, (0, 1));
        }
        let parts = HEAP_ROUND as u32;
        for pass in 1..=HEAP_ROUND {
            let PassSlots::ChangedAndPart(part) = pass_slots(pass, true) else {
                panic!("pass {pass} is not a part's");
            };
            forget_part(&mut ranges, part, parts);
            for slot in (0..SLOTS).step_by(3).filter(|slot| slot % parts == part) {
                assert!(!ranges.contains_key(&slot), "slot {slot} not forgotten");
                ranges.insert(slot, (pass, pass));
            }
            assert_eq!(ranges.len(), SLOTS.div_ceil(3) as usize);
        }
        assert!(
            ranges.values().all(|r| r.0 != 0),
            "a slot not looked at in a round"
        );
    }

    /// Every number below the count is called once, on as many threads as
    /// asked or on this one alone.
    #[test]
    fn each_on_calls_every_number_once() {
        for threads in [0, 1, 2, 4, 16] {
            for count in [0, 1, 2, 7, 1000] {
                let calls: Vec<AtomicU64> = (0..count).map(|_| AtomicU64::new(0)).collect();
                each_on(threads, count, "victory-heap-walk", &|n| {
                    calls[n].fetch_add(1, Ordering::Relaxed);
                });
                assert!(
                    calls.iter().all(|c| c.load(Ordering::Relaxed) == 1),
                    "{threads} threads, {count} numbers"
                );
            }
        }
    }

    /// The walk of the heap buffer pass on several threads
    /// (`VICTORY_HEAP_PASS_THREADS`) makes and finds for every slot what it
    /// does on one, in slot order; with `VICTORY_HEAP_PASS_TABLED=1` a slot
    /// a draw's table held lately is left out, on either.
    #[test]
    fn the_walk_on_threads_finds_what_it_finds_on_one() {
        use crate::descriptor::{HeapDesc, create_heap, heap};
        use crate::object::{S_OK, This, guid};
        if crate::host::host().is_none() {
            return; // no host GPU
        }
        #[repr(C)]
        struct Extended(HeapDesc, [u64; 2]);
        let desc = HeapDesc {
            kind: 0,
            count: 2048,
            flags: 1,
            nodes: 1,
        };
        let iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let extended = Extended(desc, [0; 2]);
        let mut h: This = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &extended.0, &iid, &mut h),
            S_OK
        );
        let views = heap(h).host().expect("host view heap");
        // A title buffer in memory of its own and a view of 64 of its
        // elements in 1,200 slots (five jobs), as in
        // `a_bind_that_loses_the_claim_returns_while_the_walk_runs`.
        const SIZE: usize = 1 << 16;
        const SLOTS: u32 = 1200;
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
        let mut record = [0u32; 16];
        record[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 64, 12, 0, 0]);
        (record[10], record[11]) = (memory as u32, (memory as u64 >> 32) as u32);
        for slot in 0..SLOTS {
            let view = View::Placed(
                false,
                [1, 0, 0, 0, SIZE as u32, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
                record,
            );
            crate::descriptor::record(heap(h).handle(slot).unwrap(), view);
        }
        let slots = views.buffer_slots();
        assert_eq!(slots, (0..SLOTS).collect::<Vec<_>>());
        let handles: Vec<Option<u64>> = slots.iter().map(|&i| heap(h).handle(i)).collect();
        // First on four threads, which make the slots, then on one.
        let four = walk_spread(views, &slots, &handles, 4, None, false, true);
        assert!(views.made(0) && views.made(SLOTS - 1), "made by the walk");
        let one = walk_spread(views, &slots, &handles, 1, None, false, true);
        assert!(four.removed.is_none() && one.removed.is_none());
        let same = |a: &SpreadWalk, b: &SpreadWalk| {
            a.found.len() == b.found.len()
                && a.found.iter().zip(&b.found).all(|((i, x), (j, y))| {
                    i == j
                        && match (x, y) {
                            (Some(x), Some(y)) => {
                                std::sync::Arc::ptr_eq(&x.0, &y.0) && (x.1, x.2) == (y.1, y.2)
                            }
                            (None, None) => true,
                            _ => false,
                        }
                })
        };
        assert!(same(&four, &one), "the same slots, copies and ranges");
        assert!(
            four.found.iter().all(|(_, r)| r.is_some()),
            "every slot shows its copy"
        );
        let found: Vec<u32> = four.found.iter().map(|&(i, _)| i).collect();
        assert_eq!(found, slots, "in slot order");
        // A slot of a draw's table lately: left to the draw, on both.
        views.buffer_slot_tabled(700);
        let frame = crate::frame::current();
        let tabled =
            [4, 1].map(|n| walk_spread(views, &slots, &handles, n, Some(frame), false, true));
        assert!(same(&tabled[0], &tabled[1]));
        for (index, range) in &tabled[0].found {
            assert_eq!(range.is_none(), *index == 700, "slot {index}");
        }
        // A slot of a table made afresh, then no table for nine frames:
        // left to the draw at first, then made by the pass.
        let first = frame + 100;
        views.buffer_tabled_in(701, first);
        let walk = |at: u64| walk_spread(views, &slots, &handles, 4, Some(at), false, true);
        let shown = |w: &SpreadWalk| {
            w.found
                .iter()
                .find(|&&(i, _)| i == 701)
                .map(|(_, r)| r.is_some())
        };
        assert_eq!(shown(&walk(first)), Some(false), "left to the draw");
        assert_eq!(shown(&walk(first + 9)), Some(true), "made by the pass");
    }

    #[test]
    fn heap_ranges_tracks_frames_per_heap_and_cleans_up_on_forget() {
        let mut guard = HEAP_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let map = guard.get_or_insert_with(Default::default);
        let id_a = u64::MAX - 10;
        let id_b = u64::MAX - 11;
        map.remove(&id_a);
        map.remove(&id_b);

        let claim = |map: &mut std::collections::HashMap<u64, HeapRanges>, id: u64, frame: u64| {
            let e = map.entry(id).or_insert_with(|| HeapRanges {
                last_frame: u64::MAX,
                passes: 0,
                ranges: HeapRangeMap::new(),
            });
            if e.last_frame == frame {
                false
            } else {
                e.last_frame = frame;
                e.passes += 1;
                true
            }
        };

        // Heap A and Heap B both run in frame 5 (no cross-heap starvation).
        assert!(claim(map, id_a, 5));
        assert!(claim(map, id_b, 5));
        // Second call for Heap A in the same frame skips.
        assert!(!claim(map, id_a, 5));
        // Next frame runs again.
        assert!(claim(map, id_a, 6));
        drop(guard);

        // Dropping a HostHeap calls forget_heap(id), freeing the map entry.
        forget_heap(id_a);
        forget_heap(id_b);
        let guard = HEAP_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let map = guard.as_ref().unwrap();
        assert!(!map.contains_key(&id_a));
        assert!(!map.contains_key(&id_b));
    }

    /// `claim_pass` alone, on gates with no heap and no list (the pass
    /// itself is in `a_bind_that_loses_the_claim_returns_while_the_walk_runs`):
    /// it takes no lock, so a second claim of the frame returns while
    /// `HEAP_RANGES` is held, with the walk its submission is to wait for,
    /// and a second heap's pass is claimed in the same frame. With
    /// `VICTORY_HEAP_WAIT_AT_SUBMIT=0` the second claim waits at its bind
    /// until the walk has ended, and not for the lock.
    #[test]
    fn a_second_claim_of_the_frame_does_not_wait_for_the_lock() {
        use crate::descriptor::host::WalkGate;
        use std::sync::atomic::AtomicBool;
        let gate = std::sync::Arc::new(WalkGate::new(u64::MAX - 20));
        let other = std::sync::Arc::new(WalkGate::new(u64::MAX - 21));
        let frame = 7;
        let first = |_: u64| panic!("the first claim of a frame waits for nothing");
        let walking = claim_pass(&gate, frame, true, first);
        assert!(walking.is_some());
        let lock = HEAP_RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let (sent, got) = std::sync::mpsc::channel();
        let second = &gate;
        std::thread::scope(|s| {
            s.spawn(move || {
                let mut needed = None;
                let walk = claim_pass(second, frame, true, |mark| needed = Some(mark));
                sent.send((walk.is_none(), needed)).unwrap();
            });
            let returned = got.recv_timeout(std::time::Duration::from_secs(5));
            assert_eq!(
                returned,
                Ok((true, Some(frame + 1))),
                "returned while the walk ran"
            );
        });
        assert!(!gate.walked(frame + 1));
        assert!(
            claim_pass(&other, frame, true, first).is_some(),
            "another heap"
        );
        // At its bind (`VICTORY_HEAP_WAIT_AT_SUBMIT=0`): until the walk ends.
        let ended = AtomicBool::new(false);
        std::thread::scope(|s| {
            let waiter = s.spawn(|| {
                let walk = claim_pass(&gate, frame, false, |_| panic!("waited for at the bind"));
                (walk.is_none(), ended.load(Ordering::SeqCst))
            });
            std::thread::sleep(std::time::Duration::from_millis(300));
            ended.store(true, Ordering::SeqCst);
            drop(walking);
            assert_eq!(
                waiter.join().unwrap(),
                (true, true),
                "let go after the walk"
            );
        });
        drop(lock);
        assert!(gate.walked(frame + 1));
        assert!(
            claim_pass(&gate, frame + 1, true, first).is_some(),
            "the next frame"
        );
    }

    /// Through `State::heap_buffers`, on a view heap with one buffer view:
    /// while one list's walk is held after it mirrored that view (it holds
    /// `HEAP_RANGES`), a second list's bind in the same frame returns, and
    /// that list is left with the one walk its submission waits for. The
    /// walk counts as ended only once the held walk goes on, not before
    /// its last mirror.
    #[test]
    fn a_bind_that_loses_the_claim_returns_while_the_walk_runs() {
        use crate::descriptor::{HeapDesc, View, create_heap, heap};
        use crate::object::{S_OK, This, guid, slot};
        use std::sync::mpsc::channel;
        use std::time::Duration;
        if !heap_claim() || !heap_wait_at_submit() {
            return; // the switches are set once for the process
        }
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        // What newer titles pass: the description, then no memory of theirs.
        #[repr(C)]
        struct Extended(HeapDesc, [u64; 2]);
        let desc = HeapDesc {
            kind: 0,
            count: 8,
            flags: 1,
            nodes: 1,
        };
        let iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let extended = Extended(desc, [0; 2]);
        let mut h: This = std::ptr::null();
        assert_eq!(
            create_heap(std::ptr::null(), &extended.0, &iid, &mut h),
            S_OK
        );
        let views = heap(h).host().expect("host view heap");
        // A title buffer in memory of its own and a view of 64 of its
        // elements in slot 1, left for the walk (as in the host heap's
        // `a_slot_the_pass_makes_is_not_walked_again`).
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
        let mut view = [0u32; 16];
        view[..10].copy_from_slice(&[0, 1, 5768, 0, 0, 0, 64, 12, 0, 0]);
        (view[10], view[11]) = (memory as u32, (memory as u64 >> 32) as u32);
        let view = View::Placed(
            false,
            [1, 0, 0, 0, SIZE as u32, 0, 1, 0x1_0001, 0, 1, 0, 1, 0, 0],
            view,
        );
        crate::descriptor::record(heap(h).handle(1).unwrap(), view);
        assert!(!views.made(1), "left for the walk");
        assert_eq!(views.buffer_slots(), [1]);
        let gate = views.walk_gate().clone();
        let (frame, mark) = (7, 8);
        // Each thread makes its own list (a list stays on its thread) and
        // binds the heap; `at` is the title heap's address.
        let at = h as usize;
        let bind = move |list: &HostList| {
            let mut state = State::default();
            state.heaps[0] = at;
            state.heap_buffers_in(list, frame).is_some()
        };
        let (parked, is_parked) = channel();
        let (go_on, gone_on) = channel::<()>();
        let (returned, has_returned) = channel();
        let wait = Duration::from_secs(30);
        std::thread::scope(|s| {
            // (Dropped with this closure if a check below fails.)
            let go_on = go_on;
            let walker = s.spawn(move || {
                // Held until the test says so, or has failed.
                WALK_HOOK.set(Some(Box::new(move || {
                    parked.send(()).unwrap();
                    let _ = gone_on.recv_timeout(wait);
                })));
                let list = host.new_list().unwrap();
                let bound = bind(&list);
                WALK_HOOK.set(None);
                (bound, list.walks_needed().len())
            });
            is_parked
                .recv_timeout(wait)
                .expect("the walk mirrors the view");
            assert!(!gate.walked(mark), "held after a mirror: not ended");
            s.spawn(move || {
                let list = host.new_list().unwrap();
                let bound = bind(&list);
                returned.send((bound, list.walks_needed())).unwrap();
            });
            let (bound, walks) = has_returned
                .recv_timeout(wait)
                .expect("returned while the walk was held");
            assert!(bound);
            assert_eq!(walks.len(), 1, "the walk its submission waits for");
            assert!(std::sync::Arc::ptr_eq(&walks[0].0, &gate));
            assert_eq!(walks[0].1, mark);
            assert!(!gate.walked(mark), "the walk is still held");
            go_on.send(()).unwrap();
            assert_eq!(
                walker.join().unwrap(),
                (true, 0),
                "the list that walks waits for none"
            );
        });
        assert!(gate.walked(mark));
        assert!(views.made(1), "made by the walk");
        // The title releases its heap, and with it the pass's ranges.
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        crate::shadow::unregister(memory, memory as u64);
    }

    // ---- Where refreshes are asked for (`crate::frame_budget::Source`) ----

    use test_heap::{VIEW_MEMORY, ViewHeap, test_frame};

    /// The refresh steps of `list` for `copy` (title address, bytes, source).
    fn steps_of(list: &HostList, copy: &crate::shadow::Shadow) -> Vec<(usize, usize, Source)> {
        list.refresh_steps_for_test()
            .into_iter()
            .filter(|&(id, ..)| id == copy.id())
            .map(|(_, at, len, source)| (at, len, source))
            .collect()
    }

    /// The heap buffer pass tags its refreshes as its own.
    #[test]
    fn warp_h2_source_tag_heap_pass() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        heap.put(1, 0);
        let list = host.new_list().unwrap();
        heap.pass(&list, test_frame());
        let copy = heap.copy();
        assert_eq!(
            steps_of(&list, &copy),
            [(heap.memory, 64 * 12, Source::HeapPass)]
        );
        list.close();
        list.reset();
    }

    /// A root shader resource view bound as a one-view table tags its
    /// refresh as such.
    #[test]
    fn warp_h2_source_tag_root_srv() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        let list = host.new_list().unwrap();
        let at = heap.memory + 0x100;
        let (mut state, _root) = heap.root_srv(host, &list, at as u64);
        let copy = heap.copy();
        assert_eq!(
            steps_of(&list, &copy),
            [(at, VIEW_MEMORY - 0x100, Source::RootSrv)]
        );
        state.tables_done();
        list.close();
        list.reset();
    }

    /// A buffer view in a table a draw binds tags its refresh as the
    /// table's.
    #[test]
    fn warp_h2_source_tag_table_readable() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        let list = host.new_list().unwrap();
        readable(&list, ResourceType::Srv, &heap.view(4));
        let copy = heap.copy();
        assert_eq!(
            steps_of(&list, &copy),
            [(heap.memory + 48, 64 * 12, Source::Table)]
        );
        list.close();
        list.reset();
    }

    // ---- The heap buffer pass refreshes per joined range ----

    /// [`join_near`] against a byte map: two bytes are in one range exactly
    /// when every byte between them is covered or within the gap of a
    /// covered byte below.
    #[test]
    fn join_near_matches_a_byte_map() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for _ in 0..500 {
            let gap = 1 + next(16);
            let ranges: Vec<(u64, u64)> = (0..1 + next(8))
                .map(|_| {
                    let from = next(200);
                    (from, from + 1 + next(20))
                })
                .collect();
            let mut joined = ranges.clone();
            join_near(&mut joined, gap);
            // The reference: covered bytes, then runs joined over gaps.
            let mut covered = [false; 256];
            for &(from, to) in &ranges {
                covered[from as usize..to as usize].fill(true);
            }
            let mut reference: Vec<(u64, u64)> = Vec::new();
            for byte in 0..256u64 {
                if !covered[byte as usize] {
                    continue;
                }
                match reference.last_mut() {
                    Some(last) if byte < last.1 + gap => last.1 = byte + 1,
                    _ => reference.push((byte, byte + 1)),
                }
            }
            assert_eq!(joined, reference, "{ranges:?} gap {gap}");
        }
    }

    /// Views of one copy that overlap or lie less than 4 KiB apart are one
    /// refresh; one far from them is a refresh of its own, and the bytes
    /// between are not refreshed by the pass.
    #[test]
    fn warp_heap_pass_joined_ranges_refresh_each_view() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        // [0, 768), [120, 888), [1200, 1968), [32772, 33540).
        for (slot, first) in [(1, 0), (2, 10), (3, 100), (4, 2731)] {
            heap.put(slot, first);
        }
        let list = host.new_list().unwrap();
        heap.pass(&list, test_frame());
        let copy = heap.copy();
        assert_eq!(
            steps_of(&list, &copy),
            [
                (heap.memory, 1968, Source::HeapPass),
                (heap.memory + 32772, 768, Source::HeapPass),
            ]
        );
        list.close();
        list.reset();
    }

    /// A copy of memory that is not write-watched (copied whole by every
    /// refresh, at most once per `VICTORY_UNWATCHED_FRAMES` frames) has
    /// every one of its ranges refreshed by the pass that takes it, not only
    /// the first.
    #[test]
    fn warp_heap_pass_unwatched_copy_refreshes_every_range() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(false);
        let copy = heap.copy();
        copy.update(heap.memory, 0x100);
        assert!(copy.unwatched());
        heap.put(1, 0);
        heap.put(2, 2731);
        for frame in [test_frame(), test_frame()] {
            let list = host.new_list().unwrap();
            heap.pass(&list, frame);
            assert_eq!(
                steps_of(&list, &copy),
                [
                    (heap.memory, 768, Source::HeapPass),
                    (heap.memory + 32772, 768, Source::HeapPass),
                ]
            );
            list.close();
            list.reset();
        }
    }

    /// A table view whose bytes lie between two of the pass's views, which
    /// the pass no longer refreshes, gets a refresh of its own exact range
    /// in the same list.
    #[test]
    fn warp_heap_pass_gap_read_by_table_view_gets_exact_refresh() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        heap.put(1, 0);
        heap.put(2, 2731);
        let list = host.new_list().unwrap();
        heap.pass(&list, test_frame());
        readable(&list, ResourceType::Srv, &heap.view(1000));
        let copy = heap.copy();
        let steps = steps_of(&list, &copy);
        assert!(
            steps.contains(&(heap.memory + 12000, 768, Source::Table)),
            "{steps:?}"
        );
        assert!(
            steps
                .iter()
                .filter(|s| s.2 == Source::HeapPass)
                .all(|&(at, len, _)| at + len <= heap.memory + 12000 || heap.memory + 12768 <= at),
            "the pass does not refresh the gap: {steps:?}"
        );
        list.close();
        list.reset();
    }

    /// Refresh steps: title address, bytes, source.
    type Steps = Vec<(usize, usize, Source)>;

    /// A small table made afresh (it holds a constant view) whose first slot,
    /// which the shader does not name, holds a buffer view (`uav`: for
    /// unordered access) between two of the heap buffer pass's views, 4 KiB
    /// and more from each; the table is bound, the pass runs, and the table
    /// is bound again after a title copy. The refresh steps of the copy over
    /// the slot's bytes after the first bind and at the end, and the list's
    /// declared writes.
    fn unnamed_slot(tabled: bool, uav: bool) -> Option<([Steps; 2], usize)> {
        let host = crate::host::host()?;
        struct Tabled;
        impl Drop for Tabled {
            fn drop(&mut self) {
                TABLED_HERE.with(|on| on.set(false));
            }
        }
        TABLED_HERE.with(|on| on.set(tabled));
        let _tabled = Tabled;
        let heap = ViewHeap::new(true);
        let gap = heap.memory + 12000;
        let view = if uav { heap.uav(1000) } else { heap.view(1000) };
        crate::descriptor::record(crate::descriptor::heap(heap.h).handle(0).unwrap(), view);
        let constant = crate::descriptor::heap(heap.h).handle(1).unwrap();
        crate::descriptor::record(constant, View::Constant(heap.memory as u64, 256));
        heap.put(2, 0);
        heap.put(3, 2731);
        let table = crate::descriptor::heap(heap.h).handle(0).unwrap();
        let used = [UsedResource {
            kind: ResourceType::Cbv,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }];
        let list = host.new_list().unwrap();
        let mut state = State::default();
        state.heaps[0] = heap.h as usize;
        set(&mut state.compute, 0, Root::Table(table));
        let copy = heap.copy();
        let over = |list: &HostList| -> Steps {
            steps_of(list, &copy)
                .into_iter()
                .filter(|&(at, len, _)| at < gap + 768 && gap < at + len)
                .collect()
        };
        assert_eq!(
            state.tables(&list, &used, &[], true).len(),
            1,
            "made afresh"
        );
        let bound = over(&list);
        heap.pass(&list, crate::frame::current());
        // A title copy in between: the next bind of the same table, found
        // made in the list, refreshes again what it refreshed.
        list.defer_copy(
            heap.memory as *mut u8,
            (heap.memory + 0x1_0000) as *const u8,
            4,
        );
        assert_eq!(state.tables(&list, &used, &[], true).len(), 1);
        let end = over(&list);
        let writes = list.gpu_writes_for_test();
        state.tables_done();
        list.close();
        list.reset();
        Some(([bound, end], writes))
    }

    /// With `VICTORY_HEAP_PASS_TABLED=1` the unnamed slot is the table's, so
    /// the pass leaves it to the draw, and the draw refreshes its exact
    /// range as the pass would have (nothing else: for an unordered access
    /// view no state change and no declared write), at both binds. With the
    /// switch off the pass makes the slot, and the draw does nothing for it.
    #[test]
    fn warp_heap_pass_unnamed_slot_of_a_tabled_table_gets_exact_refresh() {
        for uav in [false, true] {
            let Some(([bound, end], writes)) = unnamed_slot(true, uav) else {
                return; // no host GPU
            };
            let refreshed = |steps: &[(usize, usize, Source)]| {
                steps
                    .iter()
                    .all(|&(_, len, source)| len == 768 && source == Source::HeapPass)
            };
            assert_eq!(bound.len(), 1, "uav {uav}: at the bind: {bound:?}");
            assert_eq!(end.len(), 2, "uav {uav}: and again: {end:?}");
            assert!(refreshed(&end), "uav {uav}: as the pass refreshes: {end:?}");
            assert_eq!(writes, 0, "uav {uav}: no declared write");
        }
        let ([bound, _], writes) = unnamed_slot(false, false).expect("a host GPU");
        assert!(bound.is_empty(), "no step of the draw's: {bound:?}");
        assert_eq!(writes, 0);
    }

    /// A range asked for twice in a list is one step, with the tag it got
    /// first; another range of the same copy is a step of its own.
    #[test]
    fn warp_h2_source_tag_dedup_first_wins() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let heap = ViewHeap::new(true);
        let copy = heap.copy();
        let (at, len) = (heap.memory as u64 + 0x400, 0x200);
        let list = host.new_list().unwrap();
        copy.refresh_from(&list, at, len, Source::Vertex);
        copy.refresh_from(&list, at, len, Source::Index);
        copy.refresh(&list, at, len);
        copy.refresh_from(&list, at, len / 2, Source::Index);
        assert_eq!(
            steps_of(&list, &copy),
            [
                (at as usize, len as usize, Source::Vertex),
                (at as usize, len as usize / 2, Source::Index),
            ]
        );
        list.close();
        list.reset();
    }
    #[test]
    fn rect_vertex_span_rejects_overflow_and_out_of_bounds() {
        let view = VertexView {
            address: 0x1000_0000,
            size: 192,
            stride: 16,
        };
        // Valid: 2 rectangles (6 vertices starting at 0) -> 96 src bytes, 192 dst bytes.
        assert_eq!(rect_vertex_span(view, 0, 6, 2), Some((0x1000_0000, 96, 192)));
        // Valid with non-zero first: 1 rect (3 vertices at first=3) -> 48 src bytes, 96 dst bytes.
        assert_eq!(
            rect_vertex_span(view, 3, 3, 1),
            Some((0x1000_0000 + 48, 48, 96))
        );
        // u32 overflow in `first + count`: 0x8000_0000 + 0x8000_0000 wraps to 0 in u32!
        assert_eq!(
            rect_vertex_span(view, 0x8000_0000, 0x8000_0000, 0x8000_0000 / 3),
            None
        );
        // `v.address + first * stride` u64 overflow.
        let high = VertexView {
            address: u64::MAX - 16,
            size: 256,
            stride: 16,
        };
        assert_eq!(rect_vertex_span(high, 4, 3, 1), None);
        // Expanded 6-vertex buffer exceeding u32::MAX must not silently truncate SizeInBytes.
        let huge = VertexView {
            address: 0x1000_0000,
            size: u32::MAX,
            stride: 16,
        };
        assert_eq!(
            rect_vertex_span(huge, 0, 0x0FFF_FFFC, 0x0FFF_FFFC / 3),
            None
        );
    }
}
