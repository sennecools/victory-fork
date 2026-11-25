//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. While pipelines are
//! asked for by the hundred (loading, a world's first minutes) a frame may
//! wait longer ([`LOADING_FRAME`], [`LOADING_FRAME_COMPUTE`]), up to a bound
//! for the whole run (`VICTORY_PIPELINE_GATE_S`): a pass the title runs
//! only once (one that unpacks a model, say) is lost if its pipeline is not
//! there, and a draw comes again in the next frame. The recording threads of
//! a frame share the base budget; past it each waits for its own pipeline
//! in the frame's loading window ([`Window`]), whose time counts once
//! however many threads wait in it. No frame waits for a pipeline the host
//! driver has been at for longer than most take ([`slow`]).
//!
//! Once the world runs (a frame of [`WORLD_DRAWS`] draws or more), one
//! pipeline's uses wait for it one frame's allowance in all, not in every
//! frame: one the host driver takes minutes to compile, or that stands far
//! back in the queue, is used in every frame meanwhile and would spend each
//! frame's whole budget. Before the world that does not apply: counted
//! from the first loading screen, it gave up on the passes the title runs
//! once while its world loads, and those never came back.
//!
//! While the ray-tracing structure builder is off, the compute programs
//! that shade ray-tracing hits and the builder's own stages are not made
//! at all ([`leaves_out`]).
//!
//! The loading-time waits and the slow-driver cutoff come from the AMD
//! fork of the fork (commit 4436f8f1, GPL-3.0), as does leaving the
//! hit-shading programs out (there by their names, here by their shape),
//! and the per-thread loading window (its pull request 6, GPL-3.0).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// How long its uses have waited for it once the world runs, in
    /// microseconds.
    waited_us: AtomicU64,
    /// When a worker began it ([`slow`]).
    begun: OnceLock<Instant>,
    /// Why [`leaves_out`] did not make it, if it did not.
    left_out: OnceLock<LeftOut>,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            waited_us: AtomicU64::new(0),
            begun: OnceLock::new(),
            left_out: OnceLock::new(),
        }
    }

    /// Whether it is a compute pipeline.
    fn computes(&self) -> bool {
        matches!(self.shaders.as_slice(), [only] if only.stage == "compute")
    }

    /// What is left of the waiting its uses are given once the world runs:
    /// one frame's `allowance` in all.
    fn wait_left(&self, allowance: Duration) -> Duration {
        let waited = Duration::from_micros(self.waited_us.load(Ordering::Relaxed));
        allowance.saturating_sub(waited)
    }

    /// Counts a use's wait against `allowance`; whether it was the one that
    /// used up the pipeline's waiting (true at most once, so it is named
    /// once).
    fn count_wait(&self, waited: Duration, allowance: Duration) -> bool {
        // Relaxed: the count guards no other data (the host pipeline is
        // published through `made`). Uses that wait at once each saw some
        // left, so together they may wait a little over the allowance. A
        // wait is at most the allowance and is only counted while some was
        // left, so the sum stays far below u64::MAX.
        let waited = waited.min(allowance).as_micros() as u64;
        let allowance = allowance.as_micros() as u64;
        let before = self.waited_us.fetch_add(waited, Ordering::Relaxed);
        before < allowance && before + waited >= allowance
    }

    fn make(&self) {
        let _ = self.begun.set(Instant::now());
        self.made.get_or_init(|| {
            if let Some(why) = self.left_out_while_dead() {
                let _ = self.left_out.set(why);
                let counter = match why {
                    LeftOut::ShadesHits => &LEFT_OUT[0],
                    LeftOut::Builds => &LEFT_OUT[1],
                };
                // Each kind's own count decides: a sum loaded afterwards
                // can step over a power of two when two workers race.
                let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_power_of_two() {
                    let [hit_shading, builder] = left_out_counts();
                    tracing::info!(
                        hit_shading,
                        builder,
                        "compute pipelines left out: the ray-tracing structure builder is off (VICTORY_RT_RESOLVE=1 makes them)"
                    );
                }
                return None;
            }
            HostPipeline::new(&self.shaders, self.graphics.as_deref())
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }

    /// Why it is not made while nothing builds the ray-tracing structures
    /// ([`leaves_out`]), if it is not. Translates a compute program (a
    /// worker does, before the host compile it saves).
    fn left_out_while_dead(&self) -> Option<LeftOut> {
        let [shader] = self.shaders.as_slice() else {
            return None;
        };
        let shape = || shader.translated().map(Shape::of);
        leaves_out(structures_dead(), shader.stage == "compute", shape)
    }
}

/// Why a compute pipeline is left out while the structure builder is off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeftOut {
    /// It shades what a ray hit ([`shades_hits`]).
    ShadesHits,
    /// It is a stage of the structure builder (`super::of_builder`).
    Builds,
}

/// Compute pipelines left out by [`leaves_out`] so far: hit shading, the
/// builder's stages.
static LEFT_OUT: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

fn left_out_counts() -> [u64; 2] {
    LEFT_OUT.each_ref().map(|n| n.load(Ordering::Relaxed))
}

/// What [`leaves_out`] looks at in a translated compute program.
#[derive(Debug, Clone, Copy)]
struct Shape<'a> {
    /// Root SRVs and UAVs the translation binds
    /// ([`victory_shader::emit::HostShader`]).
    root_srvs: &'a [u8],
    root_uavs: &'a [u8],
    /// User-data dwords it reads.
    user_data: usize,
    /// User SGPRs and LDS bytes of its dispatch settings.
    user_sgprs: u8,
    lds_bytes: u32,
    /// It reads a spilled user-data table (the builder's stages).
    spilled: bool,
}

impl Shape<'_> {
    fn of(t: &super::Translated) -> Shape<'_> {
        Shape {
            root_srvs: &t.host.root_srvs,
            root_uavs: &t.host.root_uavs,
            user_data: t.host.user_data.len(),
            // No dispatch settings: no shape that matches.
            user_sgprs: t.compute.map_or(u8::MAX, |c| c.user_sgprs()),
            lds_bytes: t.compute.map_or(u32::MAX, |c| c.lds_bytes()),
            spilled: t.spilled,
        }
    }
}

/// Whether a compute program has the shape of the title's hit-shading
/// programs: its only root view the heap's strides (`SLOT_STRIDES`: it
/// picks buffer views from the heap), no root UAV, 4 user-data dwords, 12
/// user SGPRs, no LDS. In an offline census of the title's compute
/// programs this is all of the some 1,000 that shade ray-tracing hits and
/// 3 of the other 1,600 (they match in every other respect too, so likely
/// more of the same).
///
/// Never the programs that walk the structures (they must run, or rooms
/// are left unlit): each reads title memory by address (`SLOT_MEMORY`),
/// lists the pages it found no copy of (`SLOT_FAULTS`) and uses 2 KiB of
/// LDS, and any one of those rules it out.
fn shades_hits(s: &Shape) -> bool {
    s.root_srvs == [victory_shader::emit::SLOT_STRIDES]
        && s.root_uavs.is_empty()
        && s.user_data == 4
        && s.user_sgprs == 12
        && s.lds_bytes == 0
}

/// Whether nothing builds the ray-tracing structures in this run: the
/// structure builder's stages are off (`VICTORY_BVH_BUILDER` is not 1) and
/// `VICTORY_RT_RESOLVE=1` does not ask for their readers all the same.
fn structures_dead() -> bool {
    static DEAD: OnceLock<bool> = OnceLock::new();
    *DEAD.get_or_init(|| {
        !super::builder_allowed() && std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v != "1")
    })
}

/// Why a pipeline is left out while the structures are `dead`, if it is: a
/// compute program (`shape`: its translation, `None` if it has none) that
/// shades ray-tracing hits ([`shades_hits`]) or is a stage of the builder.
/// With the builder off no ray hits anything there is to shade, and each
/// hit-shading program takes AMD's compiler 5 to 9 s: some 1,000 of them,
/// all asked for at the world's first frame, stood in the queue before
/// the passes that make the picture. The builder's own stages never run
/// with the builder off. The programs that walk the structures are made:
/// left out, rooms were not lit.
///
/// After the AMD fork of the fork (commit 4436f8f1), which leaves the
/// hit-shading programs out by a name in their containers; here their
/// shape decides.
fn leaves_out<'a>(
    dead: bool,
    compute: bool,
    shape: impl FnOnce() -> Option<Shape<'a>>,
) -> Option<LeftOut> {
    if !dead || !compute {
        return None;
    }
    let shape = shape()?;
    if shape.spilled {
        Some(LeftOut::Builds)
    } else if shades_hits(&shape) {
        Some(LeftOut::ShadesHits)
    } else {
        None
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused", "left out" or "being made"; `None` if it
/// has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match (p.made.get(), p.left_out.get()) {
        (Some(Some(_)), _) => None,
        (None, _) => Some(format!("being made {:?}", p.shaders)),
        (Some(None), Some(why)) => Some(format!(
            "{}, left out while the ray-tracing structure builder is off {:?}",
            match why {
                LeftOut::ShadesHits => "shades ray-tracing hits",
                LeftOut::Builds => "a stage of the structure builder",
            },
            p.shaders
        )),
        (Some(None), None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// The draws a title frame records from which on the world is taken to
/// run (the menus and loading screens record a few dozen; the world some
/// 5,000). The same mark the world captures use.
const WORLD_DRAWS: u64 = 500;

/// Whether a frame has recorded [`WORLD_DRAWS`] draws or more.
static WORLD: AtomicBool = AtomicBool::new(false);

/// At a present: `recorded`, the draws the title has recorded so far
/// (dropped ones included).
pub(crate) fn frame_recorded(recorded: u64) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let before = LAST.swap(recorded, Ordering::Relaxed);
    if recorded.saturating_sub(before) >= WORLD_DRAWS && !WORLD.swap(true, Ordering::Relaxed) {
        tracing::info!(
            frame = crate::frame::current(),
            "the world runs: a pipeline's uses now wait one frame's allowance in all"
        );
    }
}

/// Whether the world runs ([`frame_recorded`]).
fn in_world() -> bool {
    WORLD.load(Ordering::Relaxed)
}

/// Whether the host's driver has been at a pipeline for longer than most
/// take (one of the title's takes 85 s), `begun` ago: no frame waits for
/// it. A pass a title runs once (it unpacks a model, say) is lost if its
/// pipeline is not there, and a draw comes again in the next frame, so a
/// compute pipeline is given longer.
fn is_slow(computes: bool, begun: Option<Duration>) -> bool {
    let most = if computes { 10 } else { 3 };
    begun.is_some_and(|elapsed| elapsed > Duration::from_secs(most))
}

/// [`is_slow`] of `p`.
fn slow(p: &Pending) -> bool {
    is_slow(p.computes(), p.begun.get().map(Instant::elapsed))
}

/// What a frame may wait while pipelines are asked for by the hundred
/// (loading, a world's first minutes). All such waits of a run together
/// are bounded (`VICTORY_PIPELINE_GATE_S`); the title's GPU watchdog
/// bounds one frame's (30 s).
const LOADING_FRAME: Duration = Duration::from_secs(4);
/// The same for compute pipelines, waited for apart: the passes that unpack
/// a world's models are dispatched once, and in a run whose pipelines were
/// all still to be loaded 130 of them were dropped, and with them walls,
/// floor, ceiling and faces (they never came back).
const LOADING_FRAME_COMPUTE: Duration = Duration::from_secs(12);

/// `VICTORY_PIPELINE_GATE_S`: the bound on all waits past the frame's
/// budget, per kind (graphics, compute); 0: none of them.
fn gate_total() -> Duration {
    static TOTAL: OnceLock<Duration> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        let seconds = std::env::var("VICTORY_PIPELINE_GATE_S")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(120);
        tracing::info!(
            seconds,
            "host pipeline waits past the budget, in all per kind"
        );
        Duration::from_secs(seconds)
    })
}

/// What a frame may wait for pipelines of one kind: the `budget`, or the
/// `loading` frame's while the gate has that much left of its `total` after
/// `spent` (and what is left of it before that runs out).
fn frame_allowance(
    budget: Duration,
    loading: Duration,
    total: Duration,
    spent: Duration,
) -> Duration {
    if budget.is_zero() {
        // `VICTORY_PIPELINE_WAIT_MS=0` turns all waiting off.
        return Duration::ZERO;
    }
    budget.max(loading.min(total.saturating_sub(spent)))
}

/// What frames waited past their budget since the start, in microseconds;
/// graphics, then compute.
static PAST_BUDGET: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What of a wait of `waited` lies past the frame's `budget`, the frame's
/// earlier waits of its kind having taken `before`: a frame whose waits
/// are each below the budget may together be far past it.
fn past_budget(before: Duration, waited: Duration, budget: Duration) -> Duration {
    (before + waited)
        .saturating_sub(budget)
        .saturating_sub(before.saturating_sub(budget))
}

/// What a frame may wait for pipelines of `p`'s kind now.
fn allowance(p: &Pending) -> Duration {
    let compute = p.computes();
    let spent = Duration::from_micros(PAST_BUDGET[usize::from(compute)].load(Ordering::Relaxed));
    let loading = if compute {
        LOADING_FRAME_COMPUTE
    } else {
        LOADING_FRAME
    };
    frame_allowance(wait_budget(), loading, gate_total(), spent)
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    if p.made.get().is_none() {
        start(p, REQUESTED);
        // Once the world runs only what a use really waited counts: one
        // that got nothing because the frame's budget was taken by others
        // leaves the pipeline's own waiting for a later frame. One that
        // could not be made counts as made (with no host pipeline): it is
        // neither waited for nor named.
        let allowance = allowance(p);
        let world = in_world();
        let left = if world {
            p.wait_left(allowance)
        } else {
            allowance
        };
        if !left.is_zero() && !slow(p) {
            let waited = wait(p, crate::frame::current(), left, allowance);
            if world && p.count_wait(waited, allowance) && p.made.get().is_none() {
                let shaders: Vec<String> = p
                    .shaders
                    .iter()
                    .map(|s| format!("{} {}", s.stage, s.hash))
                    .collect();
                tracing::info!(
                    shaders = %shaders.join(" + "),
                    "a pipeline is still being made after a frame's allowance of waiting; its uses no longer wait"
                );
            }
        }
    }
    p.made.get()?.as_ref()
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    // A pipeline `VICTORY_SKIP_PIPELINES` names is not started early
    // either: the early list keeps what an earlier run drew with, skipped
    // or not, and the host compile the skip avoids would run here.
    #[cfg(feature = "diagnostics")]
    if super::skipped_shader(&p.shaders).is_some() {
        return;
    }
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, EARLY);
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Whether the threads of a frame each wait for their own pipeline past the
/// base budget. With the shared budget the first waiter of a frame took the
/// whole allowance, loading extension and all, and any other thread that
/// asked for a pipeline in that frame returned at once. At the start of a
/// world the title binds hundreds of compute pipelines within a few seconds
/// from several threads, so their dispatches (model unpacking, run once)
/// were dropped although their pipelines were made 2-30 ms later, and
/// characters' heads and a room's walls were missing for the session. Now
/// only the base budget is shared that way; past it the frame has a loading
/// window ([`Window`]) in which every thread waits for the pipeline it
/// needs. On by default; `VICTORY_PIPELINE_WAIT_OWN=0` restores the old
/// sharing (it needs the shared budget).
fn own_extension() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_OWN").is_none_or(|v| v != "0"))
}

/// Waits that returned without the pipeline because other threads had
/// taken the frame's allowance and no loading window was left to wait in
/// (graphics, compute); their draws are dropped.
static REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// When the first of them was, and how many milliseconds after it they were
/// last logged.
static REFUSED_FIRST: OnceLock<Instant> = OnceLock::new();
static REFUSED_LOGGED: AtomicU64 = AtomicU64::new(0);

fn refused(compute: bool) {
    REFUSED_FIRST.get_or_init(Instant::now);
    REFUSED[usize::from(compute)].fetch_add(1, Ordering::Relaxed);
}

/// Logs the waits refused since the last such line, a second after it at
/// the earliest. Called when a pipeline is waited for, so the waits of a
/// burst are in one line and the last of them are not left unreported.
fn log_refused() {
    let Some(first) = REFUSED_FIRST.get() else {
        return;
    };
    let now = first.elapsed().as_millis() as u64;
    let last = REFUSED_LOGGED.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000
        || (REFUSED[0].load(Ordering::Relaxed) == 0 && REFUSED[1].load(Ordering::Relaxed) == 0)
    {
        return;
    }
    if REFUSED_LOGGED
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        tracing::warn!(
            "pipeline waits refused since the last such line: compute {}, graphics {} (their draws are dropped)",
            REFUSED[1].swap(0, Ordering::Relaxed),
            REFUSED[0].swap(0, Ordering::Relaxed),
        );
    }
}

/// Blocks until `p` is made, is a slow one, or `until` passes.
fn block_until(p: &Pending, until: Instant) {
    let _wait = crate::profile::span_args("pipeline wait", &[("compute", u64::from(p.computes()))]);
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() && !slow(p) {
        let rest = until.saturating_duration_since(Instant::now());
        if rest.is_zero() {
            break;
        }
        // Woken now and then to see whether it has become a slow one.
        let rest = rest.min(Duration::from_millis(250));
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
}

/// A frame's loading window: the time its threads may wait for pipelines
/// of one kind past the base budget. Time counts while at least one thread
/// waits in it, once however many do, and not between waits; that time is
/// what goes to the gate.
struct Window {
    frame: u64,
    length: Duration,
    /// What the waits that have ended took of it.
    used: Duration,
    /// The threads waiting in it, and since when one has been.
    waiting: u32,
    since: Instant,
}

impl Window {
    /// A thread of `frame` begins to wait at `now`: until when it may. A
    /// first wait of a frame opens its window, `length` long. `None` for a
    /// thread still recording an earlier frame, which does not take the
    /// window from the frame that has it.
    fn enter(slot: &mut Option<Window>, frame: u64, now: Instant, length: Duration) -> Option<Instant> {
        match slot {
            Some(window) if window.frame == frame => {}
            Some(window) if window.frame > frame => return None,
            _ => {
                *slot = Some(Window {
                    frame,
                    length,
                    used: Duration::ZERO,
                    waiting: 0,
                    since: now,
                });
            }
        }
        let window = slot.as_mut()?;
        if window.waiting == 0 {
            window.since = now;
        }
        window.waiting += 1;
        Some(window.since + window.length.saturating_sub(window.used))
    }

    /// The wait [`Window::enter`] allowed a thread of `frame` ends at
    /// `now`: what it adds to the gate, which is the time since a thread
    /// began to wait if this was the last one waiting. A frame that began
    /// meanwhile has its own window.
    fn leave(slot: &mut Option<Window>, frame: u64, now: Instant) -> Duration {
        let Some(window) = slot.as_mut().filter(|window| window.frame == frame) else {
            return Duration::ZERO;
        };
        window.waiting = window.waiting.saturating_sub(1);
        if window.waiting > 0 {
            return Duration::ZERO;
        }
        let waited = now
            .saturating_duration_since(window.since)
            .min(window.length.saturating_sub(window.used));
        window.used += waited;
        waited
    }
}

/// Waits for `p` while frame `frame` has some of `allowance` left for
/// pipelines of its kind, at most `most`; how long it waited. With
/// [`own_extension`] the threads of the frame share only the base budget,
/// and what `allowance` gives past it is the frame's loading [`Window`].
fn wait(p: &Pending, frame: u64, most: Duration, allowance: Duration) -> Duration {
    log_refused();
    if allowance.is_zero() {
        return Duration::ZERO;
    }
    // Per kind (graphics, compute): the frame the allowance belongs to, what
    // was waited of it (or is taken by a thread waiting now), and what the
    // waits that ended waited (for the gate).
    static SPENT: [Mutex<(u64, Duration, Duration)>; 2] =
        [const { Mutex::new((u64::MAX, Duration::ZERO, Duration::ZERO)) }; 2];
    let compute = p.computes();
    let frame_spent = &SPENT[usize::from(compute)];
    let shared = shared_budget();
    let own = shared && own_extension();
    let base = wait_budget();
    let first = if own { allowance.min(base) } else { allowance };
    let left = {
        let mut spent = frame_spent.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO, Duration::ZERO);
        }
        let left = first.saturating_sub(spent.1).min(most);
        if shared {
            spent.1 += left;
        }
        left
    };
    let started = Instant::now();
    let mut waited = Duration::ZERO;
    if !left.is_zero() {
        block_until(p, started + left);
        waited = started.elapsed().min(left);
        let past = {
            let mut spent = frame_spent.lock().unwrap_or_else(|e| e.into_inner());
            // A frame that began meanwhile has its own budget.
            if spent.0 == frame {
                if shared {
                    spent.1 = spent.1.saturating_sub(left - waited);
                } else {
                    spent.1 += waited;
                }
                let before = spent.2;
                spent.2 += waited;
                past_budget(before, waited, base)
            } else {
                past_budget(Duration::ZERO, waited, base)
            }
        };
        // A wait is at most a frame's allowance (seconds), so the sum in
        // microseconds stays far below u64::MAX.
        PAST_BUDGET[usize::from(compute)].fetch_add(past.as_micros() as u64, Ordering::Relaxed);
    }
    let mut own_waited = false;
    let rest = most.saturating_sub(waited);
    if own && !rest.is_zero() && p.made.get().is_none() && !slow(p) {
        // The first thread whose pipeline is not there after the base
        // budget opens the frame's loading window, for what the allowance
        // (the loading frame's, bounded by the gate) gives past the base
        // budget, and every thread of the frame waits for its own pipeline
        // inside it, in the world a pipeline's own allowance at most. The
        // frame stands waiting no longer than its first waiter alone held
        // it before, however many pipelines its threads wait for (the
        // title's GPU watchdog bounds a frame), and that time goes to the
        // gate.
        static WINDOWS: [Mutex<Option<Window>>; 2] = [const { Mutex::new(None) }; 2];
        let window = &WINDOWS[usize::from(compute)];
        let length = allowance.saturating_sub(base);
        let entered = Instant::now();
        let until = {
            let mut window = window.lock().unwrap_or_else(|e| e.into_inner());
            Window::enter(&mut window, frame, entered, length)
        };
        if let Some(until) = until {
            let until = until.min(entered + rest);
            own_waited = until > entered;
            block_until(p, until);
            let ended = Instant::now();
            waited += ended.saturating_duration_since(entered).min(rest);
            let past = {
                let mut window = window.lock().unwrap_or_else(|e| e.into_inner());
                Window::leave(&mut window, frame, ended)
            };
            PAST_BUDGET[usize::from(compute)].fetch_add(past.as_micros() as u64, Ordering::Relaxed);
        }
    }
    if shared && left.is_zero() && !own_waited && p.made.get().is_none() && !slow(p) {
        refused(compute);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
    waited
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

struct Queue {
    early: VecDeque<Arc<Pending>>,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            early: VecDeque::new(),
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self) -> Option<Arc<Pending>> {
        let p = self.requested.pop_front().or_else(|| {
            // An early entry can have been promoted and claimed already.
            std::iter::from_fn(|| self.early.pop_front())
                .find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        })?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            p.make();
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        AVAILABLE.notify_one();
    }
}

fn next() -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop() {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// Starts the workers (half the logical processors, 2 to 8) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8));
        for i in 0..workers {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(|| {
                    // Below the title's own threads (loading, recording).
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                    }
                    loop {
                        next().make();
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        let [left_out_hit_shading, left_out_builder] = left_out_counts();
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            left_out_hit_shading,
            left_out_builder,
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    /// A pipeline's uses wait one frame's allowance in all: a use that got
    /// no waiting does not count, and the wait that uses it up is told so
    /// once.
    #[test]
    fn a_pipeline_is_waited_for_one_allowance_in_all() {
        let allowance = Duration::from_millis(500);
        let p = Pending::new(Vec::new(), None);
        assert_eq!(p.wait_left(allowance), allowance);
        assert!(!p.count_wait(Duration::ZERO, allowance));
        assert_eq!(p.wait_left(allowance), allowance);
        assert!(!p.count_wait(allowance / 2, allowance));
        assert_eq!(p.wait_left(allowance), allowance / 2);
        assert!(p.count_wait(allowance / 2, allowance));
        assert!(p.wait_left(allowance).is_zero());
        // A use that was waiting at the same time: counted, not named again.
        assert!(!p.count_wait(allowance, allowance));
        assert!(p.wait_left(allowance).is_zero());
        // A larger allowance (the loading gate's) leaves more.
        assert_eq!(p.wait_left(allowance * 4), allowance * 2);
    }

    /// A frame waits the loading allowance while the gate has it, what is
    /// left of the gate after that, and never less than the budget; a zero
    /// budget turns all waiting off.
    #[test]
    fn the_loading_gate_bounds_the_frame_allowance() {
        let ms = Duration::from_millis;
        let s = Duration::from_secs;
        let budget = ms(500);
        assert_eq!(
            frame_allowance(budget, s(12), s(120), Duration::ZERO),
            s(12)
        );
        assert_eq!(frame_allowance(budget, s(4), s(120), s(118)), s(2));
        assert_eq!(frame_allowance(budget, s(4), s(120), s(120)), budget);
        assert_eq!(frame_allowance(budget, s(4), s(120), s(500)), budget);
        // VICTORY_PIPELINE_GATE_S=0: the budget only.
        assert_eq!(
            frame_allowance(budget, s(12), Duration::ZERO, Duration::ZERO),
            budget
        );
        assert_eq!(
            frame_allowance(Duration::ZERO, s(12), s(120), Duration::ZERO),
            Duration::ZERO
        );
    }

    /// The gate counts what a frame's waits together spent past its
    /// budget, not only single waits longer than the budget.
    #[test]
    fn the_gate_counts_a_frames_waits_together() {
        let ms = Duration::from_millis;
        let budget = ms(500);
        assert_eq!(past_budget(Duration::ZERO, ms(400), budget), Duration::ZERO);
        assert_eq!(past_budget(ms(400), ms(400), budget), ms(300));
        assert_eq!(past_budget(ms(800), ms(400), budget), ms(400));
        assert_eq!(past_budget(Duration::ZERO, ms(4000), budget), ms(3500));
        // Eight waits of 499 ms: 3,492 ms past the budget.
        let mut before = Duration::ZERO;
        let mut past = Duration::ZERO;
        for _ in 0..8 {
            past += past_budget(before, ms(499), budget);
            before += ms(499);
        }
        assert_eq!(past, ms(8 * 499 - 500));
    }

    /// The accounting of a frame's loading window on given instants.
    #[test]
    fn a_loading_window_counts_the_time_some_thread_waits() {
        let (secs, ms) = (Duration::from_secs, Duration::from_millis);
        let t0 = Instant::now();
        let mut slot = None;
        // Two threads wait at the same time: both until the window ends,
        // and the time is charged once, when the last of them returns.
        assert_eq!(Window::enter(&mut slot, 7, t0, secs(4)), Some(t0 + secs(4)));
        assert_eq!(Window::enter(&mut slot, 7, t0 + ms(10), secs(4)), Some(t0 + secs(4)));
        assert_eq!(Window::leave(&mut slot, 7, t0 + ms(30)), Duration::ZERO);
        assert_eq!(Window::leave(&mut slot, 7, t0 + ms(50)), ms(50));
        // A later wait of the frame has what is left, from when it begins:
        // the time nobody waited is not counted.
        assert_eq!(Window::enter(&mut slot, 7, t0 + secs(2), secs(9)), Some(t0 + secs(2) + ms(3950)));
        assert_eq!(Window::leave(&mut slot, 7, t0 + secs(3)), secs(1));
        // A thread still recording the frame before neither waits nor
        // takes the window, and its return charges nothing.
        assert_eq!(Window::enter(&mut slot, 6, t0 + secs(3), secs(4)), None);
        assert_eq!(Window::leave(&mut slot, 6, t0 + secs(3)), Duration::ZERO);
        // A wait that runs to the end of the window charges what was left
        // of it, and a wait after that returns at once.
        assert_eq!(Window::enter(&mut slot, 7, t0 + secs(5), secs(4)), Some(t0 + secs(5) + ms(2950)));
        assert_eq!(Window::leave(&mut slot, 7, t0 + secs(9)), ms(2950));
        assert_eq!(Window::enter(&mut slot, 7, t0 + secs(10), secs(4)), Some(t0 + secs(10)));
        assert_eq!(Window::leave(&mut slot, 7, t0 + secs(10)), Duration::ZERO);
        // The next frame opens its own window; a wait of the frame before
        // that ends afterwards does not charge it.
        assert_eq!(Window::enter(&mut slot, 7, t0 + secs(11), secs(4)), Some(t0 + secs(11)));
        assert_eq!(Window::enter(&mut slot, 8, t0 + secs(12), secs(2)), Some(t0 + secs(14)));
        assert_eq!(Window::leave(&mut slot, 7, t0 + secs(13)), Duration::ZERO);
        assert_eq!(Window::leave(&mut slot, 8, t0 + secs(13)), secs(1));
    }

    /// A thread that asks for a pipeline while another thread of the frame
    /// waits for its own gets it too, instead of returning at once.
    #[test]
    fn a_second_thread_of_the_frame_waits_for_its_pipeline() {
        if !own_extension() || !shared_budget() || wait_budget() < Duration::from_millis(400) {
            return;
        }
        // A frame no other test names, so none of them has its budget.
        let frame = 1 << 40;
        let allowance = Duration::from_secs(4);
        let pending = || {
            let p = Pending::new(Vec::new(), None);
            // A worker has it and is not done.
            p.queue_state.store(MAKING, Ordering::Relaxed);
            Arc::new(p)
        };
        let (first, second) = (pending(), pending());
        std::thread::scope(|scope| {
            // The first thread takes the frame's base budget.
            scope.spawn(|| wait(&first, frame, allowance, allowance));
            std::thread::sleep(Duration::from_millis(50));
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(100));
                second.make();
                std::thread::sleep(Duration::from_millis(50));
                first.make();
            });
            let waited = wait(&second, frame, allowance, allowance);
            assert!(second.made.get().is_some());
            assert!(waited > Duration::ZERO);
        });
    }

    /// The shape of a hit-shading program, as the census found it, with
    /// `change` made to it.
    fn hit_shader(change: impl FnOnce(&mut Shape<'static>)) -> Shape<'static> {
        let mut s = Shape {
            root_srvs: &[victory_shader::emit::SLOT_STRIDES],
            root_uavs: &[],
            user_data: 4,
            user_sgprs: 12,
            lds_bytes: 0,
            spilled: false,
        };
        change(&mut s);
        s
    }

    /// The hit-shading shape is left out, and every way off it is made.
    #[test]
    fn hit_shading_programs_are_told_by_their_shape() {
        use victory_shader::emit::{SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES};
        assert!(shades_hits(&hit_shader(|_| ())));
        let off = [
            hit_shader(|s| s.root_srvs = &[]),
            hit_shader(|s| s.root_srvs = &[SLOT_MEMORY]),
            hit_shader(|s| s.root_srvs = &[SLOT_MEMORY, SLOT_STRIDES]),
            hit_shader(|s| s.root_srvs = &[SLOT_STRIDES, 0]),
            hit_shader(|s| s.root_uavs = &[SLOT_FAULTS]),
            hit_shader(|s| s.user_data = 3),
            hit_shader(|s| s.user_data = 5),
            hit_shader(|s| s.user_sgprs = 11),
            hit_shader(|s| s.user_sgprs = 13),
            hit_shader(|s| s.lds_bytes = 512),
        ];
        for s in off {
            assert!(!shades_hits(&s), "{s:?}");
        }
    }

    /// The programs that walk the structures are never left out: each
    /// reads memory by address, reports faults and uses 2 KiB of LDS, and
    /// any one of the three rules it out. The user-data dwords and user
    /// SGPRs are each pair the census found in a walker, and the hit
    /// shaders' pair.
    #[test]
    fn walkers_are_made() {
        use victory_shader::emit::{SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES};
        let srvs: [&[u8]; 2] = [&[SLOT_MEMORY], &[SLOT_MEMORY, SLOT_STRIDES]];
        let found = [
            (6, 6),
            (8, 7),
            (10, 7),
            (10, 8),
            (8, 12),
            (9, 12),
            (10, 12),
            (12, 12),
            (13, 12),
            (4, 12),
        ];
        for root_srvs in srvs {
            for (user_data, user_sgprs) in found {
                let walker = Shape {
                    root_srvs,
                    root_uavs: &[SLOT_FAULTS],
                    user_data,
                    user_sgprs,
                    lds_bytes: 2048,
                    spilled: false,
                };
                assert_eq!(leaves_out(true, true, || Some(walker)), None, "{walker:?}");
            }
            // A hit shader given any one of the three a walker has is made.
            let memory = hit_shader(|s| s.root_srvs = root_srvs);
            let faults = hit_shader(|s| s.root_uavs = &[SLOT_FAULTS]);
            let lds = hit_shader(|s| s.lds_bytes = 2048);
            for s in [memory, faults, lds] {
                assert!(!shades_hits(&s), "{s:?}");
            }
        }
    }

    /// Hit shaders and the builder's stages are left out, only while the
    /// structures are dead and only as compute programs; the translation
    /// is not looked at otherwise, and one that does not translate is not
    /// left out (it is not made anyway).
    #[test]
    fn left_out_only_while_the_structures_are_dead() {
        let never = || -> Option<Shape<'static>> { panic!("looked at the translation") };
        let hit = hit_shader(|_| ());
        let builder = hit_shader(|s| {
            s.root_srvs = &[victory_shader::emit::SLOT_MEMORY];
            s.user_data = 9;
            s.user_sgprs = 2;
            s.spilled = true;
        });
        let other = hit_shader(|s| s.user_data = 2);
        let (hits, builds) = (Some(LeftOut::ShadesHits), Some(LeftOut::Builds));
        assert_eq!(leaves_out(true, true, || Some(hit)), hits);
        assert_eq!(leaves_out(true, true, || Some(builder)), builds);
        assert_eq!(leaves_out(true, true, || Some(other)), None);
        assert_eq!(leaves_out(true, true, || None), None);
        // VICTORY_RT_RESOLVE=1 or VICTORY_BVH_BUILDER=1: all are made.
        assert_eq!(leaves_out(false, true, never), None);
        assert_eq!(leaves_out(true, false, never), None);
        // A pipeline without shaders is not one.
        assert_eq!(Pending::new(Vec::new(), None).left_out_while_dead(), None);
    }

    /// No frame waits for a pipeline the driver has been at for over 3 s
    /// (graphics) or 10 s (compute); one not begun yet is waited for.
    #[test]
    fn slow_pipelines_are_not_waited_for() {
        let s = Duration::from_secs;
        assert!(!is_slow(false, None));
        assert!(!is_slow(true, None));
        assert!(!is_slow(false, Some(s(3))));
        assert!(is_slow(false, Some(s(4))));
        assert!(!is_slow(true, Some(s(10))));
        assert!(is_slow(true, Some(s(11))));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop().unwrap(), &earlier));
        assert!(queue.pop().is_none());
    }
}
