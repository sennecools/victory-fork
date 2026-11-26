//! The black-surface census: counters (release builds) that sort the
//! surfaces that show black into two classes before anything is changed
//! for them.
//!
//! The timing class: a texture is there but its levels were not yet (levels
//! missing at an upload, views made null while the host copy waits, a pass
//! that holds slots back for want of room, writes nobody looked at). It
//! goes away when the memory arrives.
//!
//! The deterministic class: a texture or a draw the host never takes (no
//! layout for the description, a plane that is not uploaded, a view past
//! the host heap, a draw dropped for a reason). It stays black.
//!
//! Every site is one relaxed add on one of the statics below: no lock is
//! taken on the texture pass, the view resolve or the draw path. The one
//! line that reads them is [`log`], called from the five-second block of
//! the host's frame log.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Zeros of levels not mapped yet that an upload of a partly mapped texture
/// carried to the host copy (every level is uploaded): the uploads, and the
/// bytes of their levels without pages. Asked of `held_levels([0,
/// u64::MAX])`, which looks at committed pages only. Textures made level by
/// level (`on_demand`) are left out: their missing levels are zeroed and
/// hidden on purpose.
pub(crate) static ZERO_LEVELS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Views resolved onto a texture whose last partial upload had levels
/// missing, and of them the ones that start (MostDetailedMip) below the
/// first level held. Nothing clamps those views yet.
pub(crate) static VIEWS_BELOW_HELD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Slots made null because their texture's host copy is not there yet
/// (`refused_for_now`), slots made late once it was
/// (`descriptor::make_waiting`), and slots left unmade to wait for it
/// (`descriptor::wait_for`).
pub(crate) static VIEW_NULL_WAITING: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Passes over the textures picked by index, by what ended them: the
/// queue's end, the time, the bytes, too many refused for lack of room,
/// the device.
pub(crate) static TEXTURE_PASS_STOP: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Slots the pass held back for lack of room, and the slots it refused in
/// all (a held slot is one of the refused; the rest were left due).
pub(crate) static TEXTURE_PASS_HELD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Writes under placed texture slots that found the slot out of the queue
/// (uploaded and done), and of those the ones not made due again for want
/// of the page-writes bit that does it.
pub(crate) static PAGE_WRITES_LOST: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What the guards and the partial-poll throttle did to texture checks:
/// guarded checks skipped by turn, textures a hash found changed after the
/// guard let them through, partial polls skipped, all-zero partial reads
/// not uploaded.
pub(crate) static GUARD_LOOKS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Textures whose contents are never uploaded, by reason (count, then
/// bytes of the title's memory, for each): no layout for the dimension,
/// no layout for the swizzle word, no format (or an extent the layout
/// library refuses), an XOR mode.
pub(crate) static UPLOAD_REFUSED: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

/// Looks at slots past the host view heap, and the highest slot seen.
pub(crate) static PAST_HEAP: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Textures with a stencil plane, whose plane is not uploaded, and the
/// bytes of the planes after the first.
pub(crate) static PLANES_LEFT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Why a draw or dispatch did not run on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dropped {
    /// The title's pipeline is skipped (`crate::pipeline::is_skipped`).
    Skipped = 0,
    /// A stage of the ray-tracing structure builder, which is off.
    BuilderOff,
    /// No host pipeline (yet).
    NoPipeline,
    /// A descriptor table found no free dynamic slots.
    TableFull,
    /// A shader that reads title memory by address, and no view heap.
    MemoryWithoutHeap,
    /// A root UAV with nothing to bind or no view heap.
    RootUav,
    /// A root SRV with no view heap or no slot.
    RootSrv,
    /// A root constant buffer view with no view heap or no slot.
    RootCbv,
    /// A mesh dispatch that did not run (test switches, no mesh shaders,
    /// or the list has no mesh dispatch).
    Mesh,
    /// A dispatch with a graphics pipeline.
    GraphicsDispatch,
    /// Any other drop: a pipeline of the wrong kind for the call, an index
    /// buffer title memory does not hold, rectangles not drawn.
    Other,
}

/// How many [`Dropped`] reasons there are.
pub(crate) const DROP_REASONS: usize = 11;

/// What each [`Dropped`] reason dropped, since the start: the draws, mesh
/// dispatches and dispatches the title called directly. They sum to what
/// `draw::DROPPED` counts. Drops of ExecuteIndirect and of indirect mesh
/// dispatches are counted in the frame line's "indirect" figure instead.
pub(crate) static DROPPED_BY: [AtomicU64; DROP_REASONS] =
    [const { AtomicU64::new(0) }; DROP_REASONS];

impl Dropped {
    /// Counts one draw or dispatch dropped for this reason.
    pub(crate) fn note(self) {
        DROPPED_BY[self as usize].fetch_add(1, Relaxed);
    }
}

/// The census as one line of the five-second block. The gauges (`waiting`:
/// textures and slots in `descriptor`'s table of slots waiting for a host
/// copy) are read here, under that table's lock, once in five seconds.
pub(crate) fn log() {
    let get =
        |counters: &[AtomicU64]| -> Vec<u64> { counters.iter().map(|c| c.load(Relaxed)).collect() };
    let zero = get(&ZERO_LEVELS);
    let below = get(&VIEWS_BELOW_HELD);
    let null = get(&VIEW_NULL_WAITING);
    let refusals: Vec<u64> = crate::resource::REFUSALS
        .iter()
        .map(|c| c.load(Relaxed))
        .collect();
    let [waiting_textures, waiting_slots] = crate::descriptor::waiting_gauge();
    let stop = get(&TEXTURE_PASS_STOP);
    let held = get(&TEXTURE_PASS_HELD);
    let lost = get(&PAGE_WRITES_LOST);
    let looks = get(&GUARD_LOOKS);
    let refused = get(&UPLOAD_REFUSED);
    let past = get(&PAST_HEAP);
    let planes = get(&PLANES_LEFT);
    let dropped = get(&DROPPED_BY);
    let mb = |bytes: u64| bytes >> 20;
    tracing::info!(
        "black census, since the start. Timing class: {} uploads of partly mapped textures carried {} MB of levels \
         with no pages ({} views of partly held textures made, {} of them start below the first level held); {} slots \
         made null while their texture waited for its host copy, {} made late, {} left unmade to wait for it, now \
         {waiting_textures} textures and \
         {waiting_slots} slots waiting (host copies refused: {} in all, {} for good at once, {} made on a later try, \
         {} given up); texture passes ended by the queue {} / time {} / bytes {} / room {} / device {}, passes with no \
         room {}, slots refused for room {} (held back: {}), brought up to date though the card was full {}, found changed \
         at a recheck {} of {}; writes under placed slots out of the queue {} ({} not made due again); guarded checks \
         skipped by turn {}, found changed {}, partial polls skipped {}, all-zero partial reads {}. Deterministic \
         class: uploads refused for want of a layout by dimension {} ({} MB), by swizzle word {} ({} MB), for want of a \
         format {} ({} MB), in an XOR mode {} ({} MB); {} textures with planes not uploaded ({} MB); {} looks at slots \
         past the host heap (highest {}); draws dropped by reason: skipped {}, builder off {}, no host pipeline {}, \
         table full {}, memory without a view heap {}, root UAV {}, root SRV {}, root CBV {}, mesh {}, dispatch with a \
         graphics pipeline {}, other {}",
        zero[0],
        mb(zero[1]),
        below[0],
        below[1],
        null[0],
        null[1],
        null[2],
        refusals[0],
        refusals[1],
        refusals[2],
        refusals[3],
        stop[0],
        stop[1],
        stop[2],
        stop[3],
        stop[4],
        crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        held[1],
        held[0],
        crate::draw::HEAP_TEXTURES_REUPLOADED_FULL.load(Relaxed),
        crate::draw::HEAP_TEXTURES_RECHECKED[1].load(Relaxed),
        crate::draw::HEAP_TEXTURES_RECHECKED[0].load(Relaxed)
            + crate::draw::HEAP_TEXTURES_RECHECKED[1].load(Relaxed),
        lost[0],
        lost[1],
        looks[0],
        looks[1],
        looks[2],
        looks[3],
        refused[0],
        mb(refused[1]),
        refused[2],
        mb(refused[3]),
        refused[4],
        mb(refused[5]),
        refused[6],
        mb(refused[7]),
        planes[0],
        mb(planes[1]),
        past[0],
        past[1],
        dropped[0],
        dropped[1],
        dropped[2],
        dropped[3],
        dropped[4],
        dropped[5],
        dropped[6],
        dropped[7],
        dropped[8],
        dropped[9],
        dropped[10],
    );
}

/// Tests that assert exact changes of the counters above take this, so
/// that tests running on other threads do not add to them meanwhile.
#[cfg(test)]
static CENSUS_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Takes [`CENSUS_TESTS`] (a test that failed while holding it is no reason
/// to fail the next).
#[cfg(test)]
pub(crate) fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
    CENSUS_TESTS.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each reason has a counter of its own, in the order of the line.
    #[test]
    fn each_reason_counts_in_its_own_slot() {
        let _census = lock_tests();
        let before: Vec<u64> = DROPPED_BY.iter().map(|c| c.load(Relaxed)).collect();
        let all = [
            Dropped::Skipped,
            Dropped::BuilderOff,
            Dropped::NoPipeline,
            Dropped::TableFull,
            Dropped::MemoryWithoutHeap,
            Dropped::RootUav,
            Dropped::RootSrv,
            Dropped::RootCbv,
            Dropped::Mesh,
            Dropped::GraphicsDispatch,
            Dropped::Other,
        ];
        assert_eq!(all.len(), DROP_REASONS);
        for (i, why) in all.into_iter().enumerate() {
            assert_eq!(why as usize, i);
            why.note();
        }
        for (i, was) in before.into_iter().enumerate() {
            assert_eq!(DROPPED_BY[i].load(Relaxed), was + 1, "reason {i}");
        }
    }
}
