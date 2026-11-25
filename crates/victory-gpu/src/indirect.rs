//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! Each execution is followed by a move of the scratch buffer back to the
//! common state, and the next clamp moves it from common to unordered
//! access. With `VICTORY_INDIRECT_READABLE=1` ([`readable`]; off by
//! default) the scratch buffer is left readable as indirect arguments after
//! a clamp and stays so until the next clamp into it, which moves it
//! straight back to unordered access; it goes back to the common state
//! once, when the list closes. The executions that read it then record no
//! barrier of their own, so draws that read records clamped together
//! (`crate::draw`'s windows) follow each other with nothing between them.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! Every record charges the budget on its own, with an atomic read and a
//! compare and swap on the budget word, in memory the CPU maps. Two
//! switches, both off by default, change that: `VICTORY_CLAMP_GROUP=1`
//! ([`by_group`]) takes a clamp whose thread groups charge once each
//! ([`GROUP_SOURCE`]), and `VICTORY_INDIRECT_CARD_BUDGET=1` keeps the
//! budget word in the card's own memory (`HostList::budget`). What a record
//! costs and how much work a list may do are the same with either. The
//! clamp also notes what compute records asked for, for a line of the frame
//! summary: two more atomics per record, which `VICTORY_CLAMP_LEAN=1`
//! ([`lean`]; off by default) leaves out.
//!
//! The title draws its meshes one record per call, and each such call
//! takes a clamp of its own. With `VICTORY_GROUPED_MESH_CLAMP=1`
//! ([`grouped`]; off by default) a run of up to [`MESH_BATCH`] of them,
//! from one argument buffer with nothing between them that may write
//! memory, takes one clamp ([`MeshBatch`], [`MESH_BATCH_SOURCE`]): it is
//! recorded before the run's first draw, the offsets of the records drawn
//! after it go to upload memory it reads when the list runs, and each draw
//! reads its record from its own slot. Every record is bounded and charged
//! as by a clamp of its own, in draw order.
//!
//! Every clamp waits for every earlier shader write before its dispatch
//! and again after it: on the card a wait for the whole pipe to drain,
//! twice per clamp. `VICTORY_CLAMP_PRECISE=1` ([`precise`]; off by default)
//! keeps only the waits the clamp needs ([`Waits`]).
//!
//! With `VICTORY_CLAMP_BATCH=1` ([`batch`]; off by default) the indirect
//! executions of a list between two commands that may write their
//! arguments share one clamp ([`ClampBatch`], [`BATCH_SOURCE`]): it is
//! recorded before the first of them, with its waits, and the records of
//! each execution recorded after it go to a slot of their own in the
//! batch's scratch buffer, which the clamp reads the list of from upload
//! memory when the list runs.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{D3D_SHADER_MACRO, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, OpKind, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: one dispatch, written as
    // SPLIT records of at most 65,535 groups in x each.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
#ifdef NOTE_COMPUTE
        // Diagnostics (left out with VICTORY_CLAMP_LEAN=1), in the budget
        // buffer's words after the mesh ones:
        // the largest x a compute record asked for, and the records that
        // asked for more groups in x than one dispatch runs (they are cut
        // to 65,535 below unless run in parts, so the groups past it never
        // run). Not the
        // garbage of a slot nothing wrote.
        if (asked.x <= (1u << 24) && asked.y <= 64u && asked.z <= 64u) {
            uint noted;
            budget.InterlockedMax(12, asked.x, noted);
            if (asked.x > 65535u) {
                budget.InterlockedAdd(16, 1u, noted);
            }
        }
#endif
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && asked.x <= SPLIT * 65535u) {
            g.x = asked.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_at + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

/// [`SOURCE`] with one charge of the budget per thread group
/// (`VICTORY_CLAMP_GROUP=1`, [`by_group`]): the same constants, buffers and
/// root signature, the same bounds, and the same result for every record
/// but one thing. Which records of a group still fit when the budget runs
/// out is their order in the group, where the per-record clamp leaves it
/// to the order the card served their race for the budget word.
const GROUP_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: one dispatch, written as
    // SPLIT records of at most 65,535 groups in x each.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64 threads racing for one word, costs a
// round for each record of a window, and in memory the CPU maps (where the
// budget is unless VICTORY_INDIRECT_CARD_BUDGET=1) every atomic crosses
// the bus on a card of its own.
groupshared uint costs[64];
groupshared uint fits[64];
// What the group's mesh records asked for: the largest group count, and
// the groups in all.
groupshared uint largest_asked;
groupshared uint all_asked;
#ifdef NOTE_COMPUTE
// Diagnostics (left out with VICTORY_CLAMP_LEAN=1), of the group's compute
// records: the largest x asked for, and the records that asked for more
// groups in x than one dispatch runs.
groupshared uint widest_asked;
groupshared uint cut_asked;
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
#ifdef NOTE_COMPUTE
        widest_asked = 0;
        cut_asked = 0;
#endif
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 wanted = src_args.Load3(src_at);
#ifdef NOTE_COMPUTE
        // Not the garbage of a slot nothing wrote.
        if (wanted.x <= (1u << 24) && wanted.y <= 64u && wanted.z <= 64u) {
            asked = wanted.x;
        }
#endif
        uint3 g = min(wanted, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && wanted.x <= SPLIT * 65535u) {
            g.x = wanted.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        // Diagnostics, in the budget buffer's next words (below): the
        // largest group count a mesh record asked for, and the groups
        // asked in all.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
#ifdef NOTE_COMPUTE
    if (active && kind == 2) {
        // And of compute records, in the words after those: a record over
        // 65,535 is cut to that above unless run in parts, so the groups
        // past it never run.
        uint was;
        InterlockedMax(widest_asked, asked, was);
        if (asked > 65535u) {
            InterlockedAdd(cut_asked, 1u, was);
        }
    }
#endif
    // The group's own records: none for a group past a count buffer's
    // count (an execution may ask for thousands and count a few), which
    // touches the budget not at all, as its threads do not when every
    // record charges on its own.
    uint first = id.x - gi;
    uint n = eff_count > first ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest.
        // Only the group's own records are walked (one, clamped alone).
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
#ifdef NOTE_COMPUTE
    if (gi == 0 && kind == 2 && n > 0) {
        uint was;
        budget.InterlockedMax(12, widest_asked, was);
        budget.InterlockedAdd(16, cut_asked, was);
    }
#endif
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_at + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

/// The clamp of a run of one-record mesh draws ([`MeshBatch`],
/// `VICTORY_GROUPED_MESH_CLAMP=1`): one group of [`MESH_BATCH`] threads,
/// one per record. The same root signature as [`SOURCE`]'s, whose
/// constants it does not read; in the count buffer's place the batch's
/// words: how many records it holds, then the byte offset of each in the
/// argument buffer, written as the draws are recorded and read when the
/// list runs. Record i goes to byte 12 i of the scratch buffer, bounded as
/// [`SOURCE`] bounds a mesh record clamped alone, and the records are
/// charged in draw order, each as a clamp of its own would charge it.
/// Slots past the count read nothing and cost nothing.
const MESH_BATCH_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
#define MOST 32
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer batch : register(t1);
RWByteAddressBuffer dst_args : register(u0);
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
groupshared uint costs[MOST];
groupshared uint asked[MOST];
groupshared uint fits[MOST];
[RootSignature(RS)]
[numthreads(MOST, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint n = min(batch.Load(0), uint(MOST));
    uint3 g = uint3(0, 0, 0);
    asked[lane] = 0;
    if (lane < n) {
        // Mesh dispatches: the console's are one-dimensional.
        g = src_args.Load3(batch.Load(4 + lane * 4));
        asked[lane] = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
    }
    costs[lane] = g.x * g.y * g.z * 128u;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && n > 0) {
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a record asked for, and the groups asked in all.
        uint largest = 0;
        uint all = 0;
        for (uint i = 0; i < n; i++) {
            largest = max(largest, asked[i]);
            all += min(asked[i], 1u << 24);
        }
        uint was;
        budget.InterlockedMax(4, largest, was);
        budget.InterlockedAdd(8, all, was);
        // What the records charge, from `before` on, in draw order: a
        // record that does not fit whole runs empty and takes the rest.
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < n) {
        dst_args.Store3(lane * 12, fits[lane] != 0 ? g : uint3(0, 0, 0));
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let clamp = pick(by_group(), make);
            if clamp.is_some() && lean() {
                tracing::info!(
                    "the argument clamp takes no notes of indirect compute records \
                     (VICTORY_CLAMP_LEAN=1)"
                );
            }
            clamp
        })
        .as_ref()
}

/// Whether the clamp charges a list's budget once per thread group
/// (`VICTORY_CLAMP_GROUP=1`; off by default).
///
/// The clamp charges record by record: an atomic read and a compare and
/// swap loop on the budget word for each, every thread of a group racing
/// for it, so a window of n records takes n rounds and about n * n / 2
/// atomics. With the switch the records' costs go to memory the group
/// shares, one thread charges their sum with one compare and swap (retried
/// only if another group got there first) and every thread reads whether
/// its record fitted ([`GROUP_SOURCE`]). A group past a count buffer's
/// count touches the budget not at all, either way.
fn by_group() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_GROUP").is_some_and(|v| v == "1"))
}

/// The clamp the lists use, made by `make` from a source. `grouped`
/// ([`by_group`]): the one that charges once per thread group, or, if the
/// host does not take that one, the per-record clamp after all (one
/// warning: an execution with a slower clamp, not a dropped one).
fn pick(
    grouped: bool,
    make: impl Fn(&'static str) -> windows::core::Result<Clamp>,
) -> Option<Clamp> {
    if grouped {
        match make(GROUP_SOURCE) {
            Ok(c) => {
                tracing::info!(
                    "the argument clamp charges the budget once per thread group \
                     (VICTORY_CLAMP_GROUP=1)"
                );
                return Some(c);
            }
            Err(e) => tracing::warn!(
                error = %e,
                "no argument clamp that charges once per thread group (VICTORY_CLAMP_GROUP=1): \
                 the one that charges record by record is used"
            ),
        }
    }
    match make(SOURCE) {
        Ok(c) => Some(c),
        Err(e) => {
            tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
            None
        }
    }
}

fn make(source: &'static str) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_from(device, source, !lean())
}

/// Whether the clamp leaves out its notes of what indirect compute records
/// asked for (`VICTORY_CLAMP_LEAN=1`; off by default).
///
/// The notes are the largest group count in x and the records cut to what
/// one dispatch runs: two more atomics on the budget buffer per compute
/// record (per group of them, charging once per group), read back at the
/// next submission (`HostList::budget`) for the frame line "indirect
/// compute dispatches". With the switch the clamp is compiled without them
/// (`NOTE_COMPUTE` in the source), the two words stay zero and that line is
/// not logged; what a record is clamped to and charged is the same.
pub(crate) fn lean() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_LEAN").is_some_and(|v| v == "1"))
}

/// The clamp for `device` compiled from `source` ([`SOURCE`],
/// [`GROUP_SOURCE`], [`MESH_BATCH_SOURCE`] or [`BATCH_SOURCE`]); `notes`:
/// with the atomics that note what indirect compute records asked for
/// (`NOTE_COMPUTE` in the source; without them with [`lean`]).
fn make_from(device: &ID3D12Device, source: &str, notes: bool) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // The list of defines ends with an entry of nulls.
    let defines = [
        D3D_SHADER_MACRO {
            Name: s!("NOTE_COMPUTE"),
            Definition: s!("1"),
        },
        D3D_SHADER_MACRO::default(),
    ];
    let defines = notes.then_some(defines.as_ptr());
    // SAFETY: the source, names and defines live through the call; out
    // pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
            defines,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`
/// (until [`done`], or with [`readable`] until the next clamp into it).
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    bound_as(list, src, count, spec, (0, readable()))
}

/// [`bound`] (`how` 0) and [`bound_ahead`] (`how` [`AHEAD`]); `keep`:
/// whether the scratch buffer stays readable afterwards ([`readable`]).
fn bound_as(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    (how, keep): (u32, bool),
) -> Option<ID3D12Resource> {
    // A batch's draws read the scratch buffer this clamp writes.
    list.end_mesh_batch();
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let clamped = clamp_into(list, c, &dst, budget, src, count, spec, (how, keep));
    clamped.then(|| noted(list, spec, dst))
}

/// Whether a list's scratch buffer stays readable as indirect arguments
/// between executions (`VICTORY_INDIRECT_READABLE=1`; off by default).
///
/// Every indirect execution is followed by a move of the list's clamped
/// arguments back to the common state ([`done`]), and the next clamp moves
/// them from common to unordered access. Drivers may treat a barrier into
/// or out of the common state as a full wait, so even the window path
/// (`VICTORY_INDIRECT_WINDOW=1`), whose draws skip the clamp, records one
/// barrier between every two draws.
///
/// With the switch the list remembers that a clamp left its scratch buffer
/// readable as indirect arguments (`HostList::indirect_state`): executions
/// read it with no barrier of their own, the next clamp moves it straight
/// back to unordered access, and it returns to common once, when the list
/// closes or a bigger buffer replaces it. The waits for earlier shader
/// writes around the clamp are the same either way. Said once in the log,
/// when the first clamp asks.
fn readable() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_INDIRECT_READABLE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "the indirect scratch buffer stays readable between executions \
                 (VICTORY_INDIRECT_READABLE=1)"
            );
        }
        on
    })
}

/// A clamp just recorded on `list`: noted as its op if it dispatched.
fn noted(list: &HostList, (_, _, records): (u32, u32, u32), dst: ID3D12Resource) -> ID3D12Resource {
    if records > 0 {
        list.op(OpKind::Clamp, 0);
    }
    dst
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    bound_as(list, src, None, spec, (AHEAD, readable()))
}

/// The parts [`bound_split`] makes of a dispatch, the most groups of each,
/// and the bytes between them.
pub(crate) const SPLIT: u32 = 2;
pub(crate) const PART: u32 = 65_535;
pub(crate) const PART_BYTES: u64 = 12;

/// [`bound`] for one dispatch that a program with a group base runs
/// (`victory_shader::emit::USER_GROUP_BASE`): [`SPLIT`] records follow each
/// other in the scratch buffer, of which the first has the first [`PART`]
/// groups in x and the next the ones after (empty if there are none). The
/// console runs 115,200 groups in one dispatch (the title's final
/// gather of indirect light: 8 for each 16x16 tile of 2560x1440); clamped
/// to what the host runs, the tiles at the bottom of the screen kept what
/// they showed before.
pub(crate) fn bound_split(list: &HostList, src: (&ID3D12Resource, u64)) -> Option<ID3D12Resource> {
    let spec = (2, 12, 1);
    list.end_mesh_batch();
    validate_layout(spec, src.1, None)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let dst = list.indirect_scratch(u64::from(SPLIT) * PART_BYTES)?;
    let how = (IN_PARTS, readable());
    let clamped = clamp_into(list, c, &dst, budget, src, None, spec, how);
    clamped.then(|| noted(list, spec, dst))
}

/// Bits of the clamp's `has_count`.
const AHEAD: u32 = 2;
const IN_PARTS: u32 = 4;

/// [`record_clamp_as`] into `list`'s scratch buffer `dst`, from the state
/// the list left it in. `how`: [`AHEAD`], [`IN_PARTS`], or 0. `keep`
/// ([`readable`]): afterwards the list knows the buffer is readable as
/// indirect arguments, [`done`] leaves it so, and the next clamp starts
/// from there. Without it nothing is remembered: the buffer starts every
/// clamp common, where [`done`] returns it after the execution.
#[allow(clippy::too_many_arguments)]
fn clamp_into(
    list: &HostList,
    c: &Clamp,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    (how, keep): (u32, bool),
) -> bool {
    let before = list.indirect_state(dst);
    let waits = Waits::for_list(list);
    let target = (dst, before);
    let recorded =
        record_clamp_waiting(c, &list.list, waits, target, budget, src, count, spec, how);
    if recorded {
        waits.noted(list, spec.2 > 0);
        if keep {
            list.indirect_readable(dst);
        }
    }
    recorded
}

/// Whether a clamp keeps only the waits it needs (`VICTORY_CLAMP_PRECISE=1`;
/// off by default; [`Waits::for_list`]).
///
/// Without the switch every clamp waits for every earlier shader write
/// before its dispatch, and again after it. With it:
/// - after the dispatch, no wait: the clamp writes its scratch buffer,
///   whose move from unordered access to indirect arguments orders those
///   writes before the execution that reads them, and the list's budget,
///   which only a later clamp reads, and that clamp waits for it;
/// - before it, a wait for every earlier shader write only if something
///   recorded since the list's last such wait may have written memory
///   (`HostList::may_write`: a draw or dispatch whose shaders store, a
///   command of the title's that may write memory such as a copy or a
///   clear, a buffer moved to be written; a list starts so); a wait for
///   the writes to the budget alone if only clamps wrote since; else none.
///   A title barrier on a buffer or a barrier group is such a wait.
///
/// What the clamp reads (the title's arguments and count, the budget) is
/// then ordered as before. A write the shaders make that their translation
/// does not know of is no longer waited for at every clamp.
pub(crate) fn precise() -> bool {
    #[cfg(test)]
    if let Some(on) = PRECISE_HERE.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_CLAMP_PRECISE").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "argument clamps keep only the waits they need (VICTORY_CLAMP_PRECISE=1)"
            );
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    /// Tests: [`precise`] as a test on this thread sets it, whatever the
    /// switch says (`tests::with_precise`).
    static PRECISE_HERE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// What a clamp waits for before its dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Wait<'a> {
    /// Every earlier shader write.
    All,
    /// The writes to one buffer (the list's budget).
    On(&'a ID3D12Resource),
    /// Nothing.
    Nothing,
}

/// The waits a clamp records around its dispatch: before it, [`Wait`];
/// after it, for every shader write or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Waits<'a> {
    before: Wait<'a>,
    after: bool,
}

impl<'a> Waits<'a> {
    /// No wait at all.
    const NONE: Waits<'static> = Waits {
        before: Wait::Nothing,
        after: false,
    };

    /// Every earlier shader write waited for before the dispatch and after
    /// it, as without [`precise`]. Diagnostics: `VICTORY_BARRIER_TEST=indirect`
    /// leaves both out (the state changes stay).
    fn both() -> Waits<'static> {
        #[cfg(feature = "diagnostics")]
        if crate::copystat::barrier_test() & 2 != 0 {
            return Waits::NONE;
        }
        Waits {
            before: Wait::All,
            after: true,
        }
    }

    /// What a clamp about to be recorded on `list` waits for ([`precise`]).
    fn for_list(list: &'a HostList) -> Self {
        let both = Waits::both();
        if !precise() || both == Waits::NONE {
            return both;
        }
        Self::after_list(list.since_wait(), list.budget_buffer())
    }

    /// With [`precise`]: the waits of a clamp after a list recorded `since`
    /// since its last wait for every earlier shader write, `budget` the
    /// buffer the list's clamps charge.
    fn after_list(since: crate::host::SinceWait, budget: Option<&'a ID3D12Resource>) -> Self {
        use crate::host::SinceWait;
        let before = match (since, budget) {
            (SinceWait::Nothing, _) => Wait::Nothing,
            (SinceWait::Clamps, Some(budget)) => Wait::On(budget),
            (SinceWait::Clamps, None) | (SinceWait::Anything, _) => Wait::All,
        };
        Waits {
            before,
            after: false,
        }
    }

    /// The clamp is recorded on `list` (a dispatch if `dispatched`): what
    /// it left unwaited for is noted there. Diagnostics: counted.
    fn noted(self, list: &HostList, dispatched: bool) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            let left_out = u64::from(self.before != Wait::All) + u64::from(!self.after);
            add(C::ClampWaitsLeftOut, left_out);
            if matches!(self.before, Wait::On(_)) {
                add(C::ClampBudgetWaits, 1);
            }
        }
        if self.before == Wait::All {
            list.waited_for_writes();
        }
        if self.after {
            list.waited_for_writes();
        } else if dispatched {
            list.clamp_wrote();
        }
    }
}

/// Tests: [`record_clamp_waiting`] with both waits for every earlier shader
/// write ([`Waits::both`]).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    target: (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    how: u32,
) -> bool {
    record_clamp_waiting(c, l, Waits::both(), target, budget, src, count, spec, how)
}

/// Records the clamp into `dst`, which is in state `before` (common, or
/// indirect arguments after an earlier clamp into it that the list kept
/// readable), with `waits` around its dispatch, and leaves `dst` in
/// `INDIRECT_ARGUMENT`. `how`: [`AHEAD`], the records are a window clamped
/// ahead of their draws ([`bound_ahead`]); [`IN_PARTS`], one dispatch in
/// parts ([`bound_split`]). False, with nothing recorded, if the layout or
/// the addresses do not fit the clamp.
#[allow(clippy::too_many_arguments)]
fn record_clamp_waiting(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    waits: Waits<'_>,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    how: u32,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [
        kind,
        stride,
        records,
        args_offset,
        has_count | how,
        count_offset,
    ];
    record_gpu_clamp(
        c,
        l,
        waits,
        (dst, before),
        constants,
        [src_gpu, count_gpu, dst_gpu, budget],
        records.div_ceil(64),
    );
    true
}

/// Records `groups` groups of clamp `c` with `constants` and the source,
/// count, scratch and budget addresses, between `waits`, moving `dst` from
/// `before` to unordered access and on to `INDIRECT_ARGUMENT` (no dispatch
/// for none: the state changes only).
fn record_gpu_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    waits: Waits<'_>,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    constants: [u32; 6],
    [src_gpu, count_gpu, dst_gpu, budget]: [u64; 4],
    groups: u32,
) {
    let launch = (groups > 0).then_some(|| {
        // SAFETY: an open list with `c`'s root signature and pipeline set;
        // live source, count, scratch destination, and budget buffers.
        // `src` and `count` stay in `COMMON` / `GENERIC_READ` via implicit
        // buffer promotion.
        unsafe {
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(groups, 1, 1);
        }
    });
    record_clamp_between(c, l, waits, (dst, before), launch);
}

/// Records clamp `c` between `waits`, moving `dst` from `before` to
/// unordered access and on to `INDIRECT_ARGUMENT`: `launch`, if given, binds
/// the clamp's inputs and runs it, after `c`'s root signature and pipeline
/// are set (none: the state changes only). Counted as one clamp
/// (`C::IndirectBounded`, `Count::Clamps`), whatever it bounds. Shared by
/// a clamp of its own ([`record_gpu_clamp`]) and a batch's
/// ([`record_batch_clamp`]).
fn record_clamp_between(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    waits: Waits<'_>,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    launch: Option<impl FnOnce()>,
) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    let wait = match waits.before {
        Wait::On(buffer) => uav_barrier_on(buffer),
        Wait::All | Wait::Nothing => uav_barrier(),
    };
    let first = [
        wait,
        transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let (skip_first, skip_second) = (
        usize::from(waits.before == Wait::Nothing),
        usize::from(!waits.after),
    );
    {
        // (A wait for the budget's writes alone is no global barrier.)
        use crate::profile::{Count, count};
        count(Count::Clamps, 1);
        count(Count::UavClampFirst, u64::from(waits.before == Wait::All));
        count(Count::UavClampSecond, u64::from(waits.after));
    }
    // SAFETY: an open list; our root signature and pipeline; `dst`, a live
    // buffer, transitions `before -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    // Moving it out of `INDIRECT_ARGUMENT` orders this clamp's writes after
    // every earlier execution's reads of the scratch buffer, and its move
    // back orders them before the executions that read them; nothing else
    // writes it.
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip_first..]);
        if let Some(launch) = launch {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            launch();
        }
        l.ResourceBarrier(&second[skip_second..]);
    }
}

/// Whether an execution of at most `max` records runs nothing, whatever its
/// arguments and count say: `State::execute_indirect` ends it before the
/// argument refresh, the clamp (a wait for every earlier shader write and
/// two state changes) and the host's execution of no records. Only `max`
/// says so on the CPU: a count buffer is mostly written by the title's
/// passes on the card, which run after the call is recorded, so what its
/// memory holds at recording is not what the call reads.
pub(crate) fn executes_nothing(max: u32) -> bool {
    max == 0
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
/// Records nothing for a scratch buffer the list keeps readable as
/// indirect arguments ([`readable`]): the next execution reads it as it
/// is, the next clamp moves it back to unordered access, and the list
/// returns it to common when it closes.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    if list.indirect_state(scratch) == D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT {
        return;
    }
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Whether runs of one-record mesh draws share one clamp
/// (`VICTORY_GROUPED_MESH_CLAMP=1`; off by default; [`MeshBatch`]).
///
/// The title draws its meshes one record per call (3,460 a frame in
/// gameplay), and each call's clamp is a compute dispatch of its own
/// between two waits for every earlier shader write. With the switch,
/// `crate::draw` gives a one-record mesh draw without a count buffer, whose
/// shaders store to no buffer, to the list's open batch ([`grouped_mesh`]):
/// the batch's clamp is recorded before its first draw and clamps the
/// records of up to [`MESH_BATCH`] draws in one dispatch. Anything that may
/// write memory or use the scratch buffer ends the batch first
/// (`HostList::end_mesh_batch`): every other draw and dispatch, every other
/// clamp, barriers, copies, clears, resolves, value writes, a change of
/// targets, the list's close. A batch also ends at a draw from another
/// argument buffer, and when it is full.
///
/// What a record is clamped to and charged is what a clamp of its own
/// gives it, in the order the title draws; the budget is charged when the
/// batch's clamp runs, before the draws, as a clamp of its own charges it
/// before its draw. The scratch buffer is left as [`readable`] says, as
/// after any clamp. Not tied to `VICTORY_INDIRECT_WINDOW=1`: with both, the
/// mesh draws a batch takes leave no window open.
pub(crate) fn grouped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GROUPED_MESH_CLAMP").is_some_and(|v| v == "1"))
}

/// The most records one batch clamps ([`MESH_BATCH_SOURCE`]'s `MOST`).
pub(crate) const MESH_BATCH: u32 = 32;
/// The bytes of a mesh record (the x, y and z groups) and of its slot in
/// the scratch buffer.
pub(crate) const MESH_STRIDE: u32 = 12;

/// The clamp of [`MESH_BATCH_SOURCE`], made at the first batch. A host that
/// does not make it is told once, and the draws take a clamp each.
fn batch_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let device = &host()?.device;
            match make_from(device, MESH_BATCH_SOURCE, false) {
                Ok(c) => {
                    tracing::info!(
                        records = MESH_BATCH,
                        "runs of one-record mesh draws share a clamp (VICTORY_GROUPED_MESH_CLAMP=1)"
                    );
                    Some(c)
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no clamp for runs of mesh draws (VICTORY_GROUPED_MESH_CLAMP=1): \
                         each draw is clamped alone"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// A run of one-record mesh draws in one recording that share a clamp
/// (`HostList::mesh_batch`, [`grouped`]): the argument buffer's host copy
/// they read (its number), the list's upload words the clamp reads when
/// the list runs (the count, then each record's byte offset), how many
/// draws it has, and the scratch buffer that holds their records clamped,
/// the i-th at `i * MESH_STRIDE`.
pub(crate) struct MeshBatch {
    copy: u64,
    words: usize,
    used: u32,
    bounded: ID3D12Resource,
}

impl MeshBatch {
    /// Records the clamp of a new batch of records of `src` (host copy
    /// `copy`) into `list`, from the state the list left its scratch
    /// buffer in. `keep` ([`readable`]): the list then knows the buffer is
    /// readable as indirect arguments, as after [`bound`]. None, with
    /// nothing recorded, if the clamp, the budget, the scratch or upload
    /// memory is not there, or an address does not fit the clamp.
    fn open(list: &HostList, copy: u64, src: &ID3D12Resource, keep: bool) -> Option<Self> {
        Self::open_with(list, batch_clamp()?, copy, src, keep)
    }

    fn open_with(
        list: &HostList,
        c: &Clamp,
        copy: u64,
        src: &ID3D12Resource,
        keep: bool,
    ) -> Option<Self> {
        let budget = list.budget()?;
        let bounded = list.indirect_scratch(u64::from(MESH_BATCH * MESH_STRIDE))?;
        let (words_gpu, words) = list.scratch(u64::from(MESH_BATCH + 1) * 4)?;
        // SAFETY: live D3D12 buffer resources.
        let (src_gpu, dst_gpu) =
            unsafe { (src.GetGPUVirtualAddress(), bounded.GetGPUVirtualAddress()) };
        if !src_gpu.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
            return None;
        }
        // SAFETY: the list's own upload memory (256-byte aligned, alive
        // until its reset). The count stays zero until a draw is added.
        unsafe { words.cast::<u32>().write(0) };
        let before = list.indirect_state(&bounded);
        let addresses = [src_gpu, words_gpu, dst_gpu, budget];
        let waits = Waits::for_list(list);
        record_gpu_clamp(
            c,
            &list.list,
            waits,
            (&bounded, before),
            [0; 6],
            addresses,
            1,
        );
        crate::profile::count(crate::profile::Count::MeshBatches, 1);
        waits.noted(list, true);
        if keep {
            list.indirect_readable(&bounded);
        }
        list.op(OpKind::Clamp, 0);
        Some(Self {
            copy,
            words: words as usize,
            used: 0,
            bounded,
        })
    }

    /// Whether a draw of a record of host copy `copy` can join.
    fn accepts(&self, copy: u64) -> bool {
        self.copy == copy && self.used < MESH_BATCH
    }

    /// Adds the record at byte `offset` of the copy: the scratch buffer and
    /// where in it the draw reads the record clamped.
    fn append(&mut self, offset: u32) -> (ID3D12Resource, u64) {
        let at = self.used;
        self.used += 1;
        let words = self.words as *mut u32;
        // SAFETY: the batch's upload words, MESH_BATCH + 1 of them, owned by
        // the open list, which one thread records; `accepts` keeps `at`
        // below MESH_BATCH. The card reads them when the list runs.
        unsafe {
            words.add(at as usize + 1).write(offset);
            words.write(self.used);
        }
        (self.bounded.clone(), u64::from(at * MESH_STRIDE))
    }

    /// After the batch's last draw: what [`done`] does after an execution.
    pub(crate) fn finish(self, list: &HostList) {
        done(list, &self.bounded);
    }
}

/// A one-record mesh draw of the record at byte `offset` of `src` (host
/// copy `copy`) joins `list`'s open batch, or ends it and opens one
/// ([`grouped`]): the scratch buffer, where in it the draw reads its record,
/// and whether a clamp was recorded for it (which replaced the list's
/// compute root signature and pipeline). None if no batch can be opened:
/// the draw then takes a clamp of its own.
pub(crate) fn grouped_mesh(
    list: &HostList,
    copy: u64,
    src: &ID3D12Resource,
    offset: u32,
) -> Option<(ID3D12Resource, u64, bool)> {
    grouped_mesh_as(list, copy, offset, |list| {
        MeshBatch::open(list, copy, src, readable())
    })
}

/// [`grouped_mesh`], with a new batch opened by `open`.
fn grouped_mesh_as(
    list: &HostList,
    copy: u64,
    offset: u32,
    open: impl FnOnce(&HostList) -> Option<MeshBatch>,
) -> Option<(ID3D12Resource, u64, bool)> {
    let mut slot = list.mesh_batch();
    let opened = !slot.as_ref().is_some_and(|batch| batch.accepts(copy));
    if opened {
        if let Some(batch) = slot.take() {
            batch.finish(list);
        }
        *slot = open(list);
    }
    let (bounded, at) = slot.as_mut()?.append(offset);
    Some((bounded, at, opened))
}

/// Tests: a draw of the record at the start of `src` opens a batch on
/// `list`, whatever the switch says; false if the host makes no batch
/// clamp.
#[cfg(test)]
pub(crate) fn open_batch(list: &HostList, src: &ID3D12Resource) -> bool {
    grouped_mesh_as(list, 0, 0, |list| MeshBatch::open(list, 0, src, readable())).is_some()
}

/// The clamp of a batch of indirect executions ([`ClampBatch`],
/// `VICTORY_CLAMP_BATCH=1`): [`GROUP_SOURCE`]'s bounds and charge, for the
/// records of every execution of the batch in one dispatch. In the place of
/// the constants, the batch's list of executions in upload memory (`calls`,
/// written as they are recorded, read when the list runs): the dispatch's
/// own arguments (the groups in x, 1, 1), how many executions, then
/// [`BATCH_ENTRY`] bytes for each: its kind, stride, records and argument
/// offset as [`SOURCE`] takes them; the count and split bits, the count's
/// offset, where its slot starts in the scratch buffer and its first group;
/// whether its count lies in the argument buffer (else in `count_buf`).
/// Each group clamps records of one execution, 64 at a time, into that
/// execution's slot, as [`GROUP_SOURCE`] clamps them into a scratch buffer
/// of their own. The groups of different executions charge the budget in
/// the order the card runs them, as the groups of one execution do.
const BATCH_SOURCE: &str = r#"
#define RS "SRV(t0), SRV(t1), SRV(t2), UAV(u0), UAV(u1)"
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
ByteAddressBuffer calls : register(t2);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
// The list's layout (BATCH_HEAD, BATCH_ENTRY and ClampBatch::append in Rust):
// where the head keeps the count of executions, and where in an entry its
// slot and first group are and whether its count lies in the arguments.
static const uint HEAD = 16;
static const uint ENTRY = 48;
static const uint EXECUTIONS_AT = 12;
static const uint PLACE_AT = 16;
static const uint FIRST_GROUP_AT = 28;
static const uint IN_ARGS_AT = 32;
groupshared uint costs[64];
groupshared uint fits[64];
// What the group's mesh records asked for: the largest group count, and
// the groups in all.
groupshared uint largest_asked;
groupshared uint all_asked;
#ifdef NOTE_COMPUTE
// Diagnostics (left out with VICTORY_CLAMP_LEAN=1), of the group's compute
// records: the largest x asked for, and the records that asked for more
// groups in x than one dispatch runs.
groupshared uint widest_asked;
groupshared uint cut_asked;
#endif
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint gi : SV_GroupIndex) {
    // The execution whose records the group clamps: the last one whose
    // first group is at or before it (the same for every thread).
    uint executions = calls.Load(EXECUTIONS_AT);
    uint lo = 0;
    uint hi = executions;
    while (hi - lo > 1) {
        uint mid = (lo + hi) / 2;
        if (calls.Load(HEAD + mid * ENTRY + FIRST_GROUP_AT) <= group.x) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    uint entry = HEAD + lo * ENTRY;
    uint4 shape = calls.Load4(entry);
    uint4 place = calls.Load4(entry + PLACE_AT);
    uint kind = shape.x;
    uint stride = shape.y;
    uint records = shape.z;
    uint args_offset = shape.w;
    uint has_count = place.x;
    uint count_offset = place.y;
    uint dst_base = place.z;
    uint first_group = place.w;
    bool count_in_args = calls.Load(entry + IN_ARGS_AT) != 0;
    uint eff_count = records;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        uint counted = count_in_args ? src_args.Load(count_offset) : count_buf.Load(count_offset);
        eff_count = min(eff_count, counted);
    }
    // The record of the execution this thread clamps.
    uint first = (group.x - first_group) * 64u;
    uint rec = first + gi;
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = executions > 0 && rec < eff_count;
    uint src_at = args_offset + rec * stride;
    uint dst_at = dst_base + rec * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
#ifdef NOTE_COMPUTE
        widest_asked = 0;
        cut_asked = 0;
#endif
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 wanted = src_args.Load3(src_at);
#ifdef NOTE_COMPUTE
        // Not the garbage of a slot nothing wrote.
        if (wanted.x <= (1u << 24) && wanted.y <= 64u && wanted.z <= 64u) {
            asked = wanted.x;
        }
#endif
        uint3 g = min(wanted, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && wanted.x <= SPLIT * 65535u) {
            g.x = wanted.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
#ifdef NOTE_COMPUTE
    if (active && kind == 2) {
        uint was;
        InterlockedMax(widest_asked, asked, was);
        if (asked > 65535u) {
            InterlockedAdd(cut_asked, 1u, was);
        }
    }
#endif
    // The group's own records; none past a count buffer's count, which
    // touches the budget not at all.
    uint n = (executions > 0 && eff_count > first) ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest.
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
#ifdef NOTE_COMPUTE
    if (gi == 0 && kind == 2 && n > 0) {
        uint was;
        budget.InterlockedMax(12, widest_asked, was);
        budget.InterlockedAdd(16, cut_asked, was);
    }
#endif
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_base + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

/// Whether the indirect executions of a list share clamps
/// (`VICTORY_CLAMP_BATCH=1`; off by default; [`ClampBatch`]).
///
/// Each indirect execution takes a clamp of its own: a compute dispatch
/// between two waits for every earlier shader write (one with
/// `VICTORY_CLAMP_PRECISE=1`), thousands a frame. With the switch,
/// `crate::draw` gives an execution (not one of a window of
/// `VICTORY_INDIRECT_WINDOW=1`, nor a mesh draw a batch of
/// `VICTORY_GROUPED_MESH_CLAMP=1` takes) to the list's open batch
/// ([`bound_batched`]): the batch's clamp is recorded before its first
/// execution, with the waits a clamp of its own has, and clamps the records
/// of every execution of the batch, each into a slot of its own. An
/// execution joins the open batch only if nothing recorded since its clamp
/// may have written memory (`HostList::may_write`: a draw or dispatch whose
/// shaders store, a title barrier, copy, clear or value write, a buffer
/// moved to be written), no state change was recorded since then for its
/// argument or count buffer (the clamp read them as they were), its
/// arguments lie in the batch's argument buffer and its count in that or
/// the batch's count buffer, and it fits ([`BATCH_CALLS`] executions,
/// [`BATCH_BYTES`] of records; the bytes also keep the dispatch's groups far
/// below what one dispatch runs, [`BATCH_GROUPS`]); otherwise it opens the
/// next batch, after what was recorded. One whose records alone are more
/// than a batch holds takes a clamp of its own. The batch's scratch buffer
/// stays readable as indirect arguments until the next batch's clamp, and
/// goes back to common when the list closes ([`close_batch`]).
///
/// What a record is clamped to is what a clamp of its own gives it. The
/// budget is charged when the batch's clamp runs, before its first
/// execution, so for every execution of the batch at once; which records
/// of a list past its budget still fit is then the order the card runs the
/// batch's groups in, not the order of the executions. The batch charges
/// once per group, as [`GROUP_SOURCE`] does (`VICTORY_CLAMP_GROUP=1`).
///
/// The batch takes on `VICTORY_CLAMP_PRECISE`'s risk even with that switch
/// off: a later execution of the batch has its records read when the
/// batch's clamp runs, so a shader write its translation does not know of
/// (no `may_write`), recorded after the batch's clamp, is no longer waited
/// for: the execution runs on the records as they were before that write.
pub(crate) fn batch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLAMP_BATCH").is_some_and(|v| v == "1"))
}

/// The most executions one batch clamps.
pub(crate) const BATCH_CALLS: u32 = 128;
/// The bytes of a batch's scratch buffer: of the records of all its
/// executions, each slot starting at a multiple of 16 bytes.
pub(crate) const BATCH_BYTES: u64 = 256 << 10;
/// The most groups a batch's dispatch can have (in x), which no check needs:
/// every record takes at least 12 bytes of [`BATCH_BYTES`] (`min_stride`),
/// and each execution adds at most one group that is not full.
const BATCH_GROUPS: u32 = (BATCH_BYTES / 12 / 64) as u32 + BATCH_CALLS;
/// The bytes of [`BATCH_SOURCE`]'s list before the first execution, and of
/// each execution (`HEAD` and `ENTRY` there; [`ClampBatch::append`] writes
/// an entry as 12 words).
const BATCH_HEAD: u64 = 16;
const BATCH_ENTRY: u64 = 48;
const _: () = assert!(BATCH_GROUPS <= 65_535 && BATCH_ENTRY == 12 * 4);

/// The clamp of [`BATCH_SOURCE`], made at the first batch. A host that
/// does not make it is told once, and the executions take a clamp each.
fn clamp_of_batches() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let device = &host()?.device;
            match make_from(device, BATCH_SOURCE, !lean()) {
                Ok(c) => {
                    tracing::info!(
                        executions = BATCH_CALLS,
                        "indirect executions share clamps (VICTORY_CLAMP_BATCH=1)"
                    );
                    Some(c)
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "no clamp for batches of indirect executions (VICTORY_CLAMP_BATCH=1): \
                         each execution is clamped alone"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// A list's batch of clamps ([`batch`]): the open batch, if any, the
/// scratch buffer batches clamp into (made at the list's first batch, kept
/// with the list), and whether a batch's clamp in this recording left it
/// readable as indirect arguments.
#[derive(Default)]
pub(crate) struct BatchSlot {
    open: Option<ClampBatch>,
    scratch: Option<ID3D12Resource>,
    readable: bool,
}

impl BatchSlot {
    /// The list was reset: the batch of a recording that was never closed
    /// goes with it, and the scratch buffer is common again.
    pub(crate) fn reset(&mut self) {
        self.open = None;
        self.readable = false;
    }

    /// The scratch buffer, made if the list has none yet.
    fn scratch(&mut self) -> Option<ID3D12Resource> {
        if self.scratch.is_none() {
            let buffer = host()?.uav_buffer(BATCH_BYTES)?;
            crate::host::name(&buffer, || "indirect arguments of batches".to_owned());
            self.scratch = Some(buffer);
        }
        self.scratch.clone()
    }
}

/// The list closes: the open batch ends, and the scratch buffer goes back
/// to common if a batch left it readable (the list ends as it begins).
pub(crate) fn close_batch(list: &HostList) {
    let mut slot = list.clamp_batch();
    slot.open = None;
    if std::mem::take(&mut slot.readable)
        && let Some(scratch) = &slot.scratch
    {
        done_on(&list.list, scratch);
    }
}

/// An open batch of clamps ([`batch`], `HostList::clamp_batch`): what the
/// list had recorded when its clamp was ([`HostList::writes`], and
/// [`HostList::buffer_changes`] of the argument and the count buffer), the
/// argument and count buffers the clamp reads (the count buffer is the
/// argument buffer if the first execution has none), the list of executions
/// in upload memory, how many there are, their groups, and the bytes of the
/// scratch buffer their slots take.
pub(crate) struct ClampBatch {
    writes: u64,
    changes: (u64, u64),
    args: ID3D12Resource,
    counts: ID3D12Resource,
    calls_list: usize,
    calls: u32,
    groups: u32,
    used: u64,
}

/// One execution as a batch takes it: its words in the batch's list (all
/// but its slot and its first group), its argument buffer and count
/// buffer, the bytes of its slot and its groups.
struct Execution<'a> {
    words: [u32; 6],
    args: &'a ID3D12Resource,
    counts: Option<&'a ID3D12Resource>,
    bytes: u64,
    groups: u32,
}

impl<'a> Execution<'a> {
    /// The execution of `spec` records at `src`, bounded by the count at
    /// `count` if given (`how`: 0, or [`IN_PARTS`] for one dispatch in parts),
    /// if a batch can take it: its layout is one the clamp takes, it has
    /// records, its slot fits in a batch, and the offsets of its arguments
    /// and count fit the clamp's 32 bits.
    fn of(
        src: (&'a ID3D12Resource, u64),
        count: Option<(&'a ID3D12Resource, u64)>,
        (kind, stride, records): (u32, u32, u32),
        how: u32,
    ) -> Option<Self> {
        let read = validate_layout((kind, stride, records), src.1, count.map(|(_, off)| off))?;
        let bytes = if how & IN_PARTS != 0 {
            u64::from(SPLIT) * PART_BYTES
        } else {
            read
        };
        let fits = |off: u64, len: u64| {
            off.checked_add(len)
                .is_some_and(|end| end <= u64::from(u32::MAX))
        };
        if records == 0
            || bytes > BATCH_BYTES
            || !fits(src.1, read)
            || count.is_some_and(|(_, off)| !fits(off, 4))
        {
            return None;
        }
        let has_count = u32::from(count.is_some());
        let words = [
            kind,
            stride,
            records,
            src.1 as u32,
            has_count | how,
            count.map_or(0, |(_, off)| off as u32),
        ];
        Some(Self {
            words,
            args: src.0,
            counts: count.map(|(buffer, _)| buffer),
            bytes,
            groups: records.div_ceil(64),
        })
    }
}

impl ClampBatch {
    /// Records the clamp of a new batch whose first execution is `first`
    /// into `list`, from the state the list left `slot`'s scratch buffer
    /// in, and leaves the buffer readable as indirect arguments. None, with
    /// nothing recorded, if the clamp's budget, scratch buffer, upload
    /// memory or command signature is not there, or an address does not fit
    /// the clamp.
    fn open(list: &HostList, c: &Clamp, slot: &mut BatchSlot, first: &Execution) -> Option<Self> {
        let budget = list.budget()?;
        let scratch = slot.scratch()?;
        let signature = crate::draw::host_signature(2, 12)?;
        let table_bytes = BATCH_HEAD + u64::from(BATCH_CALLS) * BATCH_ENTRY;
        let (table, table_at, table_gpu, words) = list.scratch_in(table_bytes)?;
        let counts = first.counts.unwrap_or(first.args).clone();
        // SAFETY: live D3D12 buffer resources.
        let (args_gpu, counts_gpu, dst_gpu) = unsafe {
            (
                first.args.GetGPUVirtualAddress(),
                counts.GetGPUVirtualAddress(),
                scratch.GetGPUVirtualAddress(),
            )
        };
        if ![args_gpu, counts_gpu, dst_gpu, budget]
            .iter()
            .all(|a| a.is_multiple_of(4))
        {
            return None;
        }
        // SAFETY: the list's own upload memory (256-byte aligned, alive
        // until its reset). No execution until one is added: no groups.
        unsafe { words.cast::<[u32; 4]>().write([0, 1, 1, 0]) };
        let before = if slot.readable {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        let waits = Waits::for_list(list);
        record_batch_clamp(
            c,
            &list.list,
            waits,
            (&scratch, before),
            [args_gpu, counts_gpu, table_gpu, dst_gpu, budget],
            (&signature, &table, table_at),
        );
        waits.noted(list, true);
        slot.readable = true;
        list.op(OpKind::Clamp, 0);
        Some(Self {
            writes: list.writes(),
            changes: (
                list.buffer_changes(first.args),
                list.buffer_changes(&counts),
            ),
            args: first.args.clone(),
            counts,
            calls_list: words as usize,
            calls: 0,
            groups: 0,
            used: 0,
        })
    }

    /// Whether `e` can join: its buffers are the ones the clamp reads; it
    /// fits; nothing that may write memory was recorded since the batch's
    /// clamp, nor a change of its buffers' states. A buffer moved to be
    /// written already ends the batch as a write; the change count also
    /// ends it at a move between read states (to `COPY_SOURCE` and back,
    /// say), so the batch's executions read their buffers in the states the
    /// clamp read them in, whatever `HostList::buffer_state` counts as a
    /// write.
    fn accepts(&self, list: &HostList, e: &Execution) -> bool {
        let same = |a: &ID3D12Resource, b: &ID3D12Resource| a.as_raw() == b.as_raw();
        let kept = |buffer: &ID3D12Resource, then: u64| list.buffer_changes(buffer) == then;
        self.calls < BATCH_CALLS
            && self.used + e.bytes <= BATCH_BYTES
            && same(e.args, &self.args)
            && e.counts
                .is_none_or(|c| same(c, &self.args) || same(c, &self.counts))
            && list.writes() == self.writes
            && kept(e.args, self.changes.0)
            && e.counts.is_none_or(|c| {
                let then = if same(c, &self.args) {
                    self.changes.0
                } else {
                    self.changes.1
                };
                kept(c, then)
            })
    }

    /// Adds `e`: where in the scratch buffer its records are clamped to.
    fn append(&mut self, e: &Execution) -> u64 {
        let at = self.used;
        let in_args = e.counts.is_some_and(|c| c.as_raw() == self.args.as_raw());
        let [kind, stride, records, args_offset, flags, count_offset] = e.words;
        let entry: [u32; 12] = [
            kind,
            stride,
            records,
            args_offset,
            flags,
            count_offset,
            at as u32,
            self.groups,
            u32::from(in_args),
            0,
            0,
            0,
        ];
        let words = self.calls_list as *mut u8;
        // SAFETY: the batch's list in the open list's upload memory, room
        // for BATCH_CALLS executions after its head; `accepts` keeps
        // `calls` below that. One thread records the list; the card reads
        // the words when the list runs.
        unsafe {
            let slot = words.add((BATCH_HEAD + u64::from(self.calls) * BATCH_ENTRY) as usize);
            slot.cast::<[u32; 12]>().write(entry);
        }
        self.calls += 1;
        self.groups += e.groups;
        debug_assert!(self.groups <= BATCH_GROUPS, "BATCH_BYTES bounds the groups");
        self.used = (at + e.bytes).next_multiple_of(16);
        // SAFETY: as above: the head, the dispatch's groups and the count.
        unsafe {
            words
                .cast::<[u32; 4]>()
                .write([self.groups, 1, 1, self.calls])
        };
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ClampBatchCalls, 1);
        crate::profile::count(crate::profile::Count::ClampBatchCalls, 1);
        at
    }
}

/// [`bound`] for an execution a batch of clamps takes ([`batch`]): `spec`
/// records at `src`, bounded by the count at `count` if given, or one
/// dispatch in parts (`parts`, as [`bound_split`] clamps it). It joins
/// `list`'s open batch or opens one: the batch's scratch buffer, where the
/// execution reads its records there (the parts of a dispatch follow each
/// other [`PART_BYTES`] apart), and whether a clamp was recorded for it
/// (which replaced the list's compute root signature and pipeline). None
/// if no batch takes it: it then takes a clamp of its own.
pub(crate) fn bound_batched(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    parts: bool,
) -> Option<(ID3D12Resource, u64, bool)> {
    bound_batched_with(list, clamp_of_batches()?, src, count, spec, parts)
}

fn bound_batched_with(
    list: &HostList,
    c: &Clamp,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    parts: bool,
) -> Option<(ID3D12Resource, u64, bool)> {
    let (spec, how) = if parts {
        ((2, 12, 1), IN_PARTS)
    } else {
        (spec, 0)
    };
    let e = Execution::of(src, count, spec, how)?;
    let mut slot = list.clamp_batch();
    let joins = slot.open.as_ref().is_some_and(|b| b.accepts(list, &e));
    if !joins {
        slot.open = None;
        let opened = ClampBatch::open(list, c, &mut slot, &e)?;
        slot.open = Some(opened);
    }
    let scratch = slot.scratch.clone()?;
    let at = slot.open.as_mut()?.append(&e);
    Some((scratch, at, !joins))
}

/// Records `c`, a batch's clamp ([`BATCH_SOURCE`]), with the argument,
/// count, list, scratch and budget addresses, between `waits`, moving
/// `dst` from `before` to unordered access and on to `INDIRECT_ARGUMENT`.
/// Its groups are read when the list runs, from the head of the list in
/// upload memory: `signature` dispatches them from `table` at `at`.
fn record_batch_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    waits: Waits<'_>,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    [args_gpu, counts_gpu, list_gpu, dst_gpu, budget]: [u64; 5],
    (signature, table, at): (&ID3D12CommandSignature, &ID3D12Resource, u64),
) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ClampBatches, 1);
    let launch = || {
        // SAFETY: an open list with `c`'s root signature and pipeline set;
        // live argument, count, scratch and budget buffers, and the list's
        // upload memory for the batch's list (in `GENERIC_READ`, which holds
        // `INDIRECT_ARGUMENT`, as upload memory always is). The argument and
        // count buffers are where the list moved them to be read, or common
        // (promoted).
        unsafe {
            l.SetComputeRootShaderResourceView(0, args_gpu);
            l.SetComputeRootShaderResourceView(1, counts_gpu);
            l.SetComputeRootShaderResourceView(2, list_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.ExecuteIndirect(signature, 1, table, at, None, 0);
        }
    };
    record_clamp_between(c, l, waits, (dst, before), Some(launch));
}

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

/// An unordered access barrier on `buffer` alone: its earlier shader
/// writes finish before later work reads or writes it.
fn uav_barrier_on(buffer: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(buffer) },
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        /// With the clamp that charges record by record.
        fn new() -> Option<Self> {
            Self::with(SOURCE)
        }

        /// With the clamp compiled from `source`: [`SOURCE`], or
        /// [`GROUP_SOURCE`] as with `VICTORY_CLAMP_GROUP=1`.
        fn with(source: &str) -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                // With its notes of compute records, as with no switch.
                let clamp =
                    make_from(&device, source, true).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            signature_of(&self.device, kind, stride)
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    /// A command signature of one record of `kind` (PC argument numbering)
    /// and `stride` bytes.
    fn signature_of(device: &ID3D12Device, kind: u32, stride: u32) -> ID3D12CommandSignature {
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
            ..Default::default()
        };
        let desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: stride,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: valid single-argument command signature description.
        unsafe {
            device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
        }
        sig.unwrap()
    }

    /// The name of `device`'s adapter, and whether it is a software one
    /// (unnamed and not, if the adapter does not say).
    pub(super) fn adapter_of(device: &ID3D12Device) -> (String, bool) {
        // SAFETY: plain DXGI adapter queries.
        unsafe {
            let luid = device.GetAdapterLuid();
            CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(luid))
                .and_then(|a| a.GetDesc1())
                .map(|d| {
                    let name = String::from_utf16_lossy(&d.Description)
                        .trim_end_matches('\0')
                        .to_owned();
                    (name, d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0)
                })
                .unwrap_or_else(|_| ("an unnamed adapter".to_owned(), false))
        }
    }

    /// The host, if it is on a software adapter (`VICTORY_D3D12_WARP=1` in
    /// a diagnostics build): for the tests that force on what a switch
    /// records. A plain `cargo test` runs the host's tests on the card,
    /// and what the switches keep off there by default (a scratch buffer
    /// moved from arguments straight to unordered access, the first
    /// `WriteBufferImmediate`) must not reach it through a test either.
    fn software_host() -> Option<&'static crate::host::Host> {
        super::host().filter(|host| adapter_of(&host.device).1)
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        card_compare_stores(crate::host::CARD_COMPARE, false);
    }

    /// The same of the compare with a group per piece
    /// (`VICTORY_CARD_HASH_WIDE=1`).
    #[test]
    fn warp_wide_card_compare_stores_the_hash_it_made_when_told() {
        card_compare_stores(crate::host::CARD_COMPARE_WIDE, true);
    }

    /// `source` is the compare with a group per piece (`wide`) or with a
    /// thread per piece.
    fn card_compare_stores(source: &str, wide: bool) {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare =
            compile_cs(&warp.device, source).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list
                    .Dispatch(if wide { PIECES } else { PIECES.div_ceil(64) }, 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Every byte counts, a piece's first and last among them.
        for (at, flagged) in [
            (0, [1, 0, 0]),
            (PIECE - 1, [1, 0, 0]),
            (PIECE, [0, 1, 0]),
            (2 * PIECE + 4096, [0, 0, 1]),
            (size - 1, [0, 0, 1]),
        ] {
            // SAFETY: inside the mapping.
            unsafe { now_ptr.add(at).write_volatile(!content[at]) };
            assert_eq!(run(&mut warp, 2), flagged, "byte {at}");
        }
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // Two 16-byte steps that change places, in the steps of one thread
        // or of two, and the same change in two threads.
        for (a, b) in [
            (PIECE, PIECE + 16),
            (PIECE, PIECE + 1024),
            (PIECE + 48, PIECE + 5 * 1024 + 160),
        ] {
            for i in 0..16 {
                // SAFETY: inside the mapping.
                unsafe { std::ptr::swap(now_ptr.add(a + i), now_ptr.add(b + i)) };
            }
            assert_eq!(run(&mut warp, 2), [0, 1, 0], "{a} and {b}");
        }
        for at in [PIECE + 32, PIECE + 64] {
            // SAFETY: inside the mapping.
            unsafe {
                now_ptr
                    .add(at)
                    .write_volatile(now_ptr.add(at).read_volatile() ^ 0x80)
            };
        }
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
    }

    /// The fetch inside flights (`crate::host::CARD_FETCH`), on the software
    /// adapter, in one list behind the compare as `Host::card_before`
    /// records it: each flagged piece takes a ticket and, while it is
    /// within the slots of the lowest ticket in the air, its bytes go to
    /// its slot; an unflagged piece's entry says so; a ticket past the
    /// slots is written but copies nothing.
    #[test]
    fn warp_card_fetch_copies_flagged_pieces_to_their_slots() {
        card_fetch_copies(crate::host::CARD_COMPARE, false);
    }

    /// The same behind the compare with a group per piece
    /// (`VICTORY_CARD_HASH_WIDE=1`): the fetch reads the compare's flags
    /// and the copy's bytes, never a hash, so it takes either compare's.
    #[test]
    fn warp_card_fetch_copies_pieces_the_wide_compare_flagged() {
        card_fetch_copies(crate::host::CARD_COMPARE_WIDE, true);
    }

    /// `compare` is the compare with a group per piece (`wide`) or with a
    /// thread per piece.
    fn card_fetch_copies(compare: &str, wide: bool) {
        use crate::host::{CARD_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare =
            compile_cs(&warp.device, compare).expect("the card compare must compile on WARP");
        let fetch =
            compile_cs(&warp.device, CARD_FETCH).expect("the card fetch must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        // The copy starts 40 bytes into its piece 0: piece 0 is short, the
        // others straddle title pieces and start off a multiple of 16.
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 1u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mappings hold what is written.
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let pieces = PIECES + 1;
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> Vec<u32> {
            // SAFETY: our open list, live buffers, the shaders' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(if wide { pieces } else { 1 }, 1, 1);
                if mode == 2 {
                    warp.list.ResourceBarrier(&[uav_barrier()]);
                    warp.list.SetComputeRootSignature(&fetch.root);
                    warp.list.SetPipelineState(&fetch.pso);
                    // Two groups for four pieces: the loop over pieces.
                    let constants = [0u32, SKEW, size as u32, pieces, 0, landed, slots, 2];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, table_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, fetched_gpu);
                    warp.list.Dispatch(2, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            (0..=pieces as usize).map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() }).collect()
        };
        // SAFETY (each): inside the mapping of the copy.
        let set = |at: usize, value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // SAFETY (each): the slot's mapping holds `INFLIGHT_SLOT` bytes.
        let slot = |len: usize| unsafe { std::slice::from_raw_parts(fetched_ptr, len).to_vec() };
        run(&mut warp, 0, 0);
        // Piece 2 of the copy (title piece 2: bytes 2 * 64 KiB - 40 on), its
        // slot from the multiple of 16 below to the one above its end.
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        assert_ne!(widened, from);
        set(from + 100, content[from + 100] ^ 0x40);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table, [1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted, "piece 2 in slot 0");
        // Two pieces flagged with one slot, ticket 0 not landed: tickets 1
        // and 2 get none and the slot keeps what it held.
        set(5, content[5] ^ 1);
        set(PIECE + 7, content[PIECE + 7] ^ 1);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table[0], 3);
        let mut tickets = [table[1], table[2]];
        tickets.sort_unstable();
        assert_eq!(tickets, [1, 2]);
        assert_eq!(&table[3..], [INFLIGHT_NONE, INFLIGHT_NONE]);
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted);
        // Every ticket up to 2 landed: the next one, 3, gets slot 0. Piece
        // 0 starts at the copy's start.
        set(9, content[9] ^ 2);
        let table = run(&mut warp, 2, 3);
        assert_eq!(table, [4, 3, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        let mut piece0 = content[..PIECE - SKEW as usize].to_vec();
        piece0[5] ^= 1;
        piece0[9] ^= 2;
        assert_eq!(slot(PIECE - SKEW as usize), piece0);
    }

    /// The compare with a thread per piece and the one with a group per
    /// piece (`VICTORY_CARD_HASH_WIDE=1`), each with hashes and results of
    /// its own, over one copy on the software adapter: through 120 passes
    /// of every kind over runs of its pieces, with bytes changed and
    /// changed back between them, both flag the pieces whose bytes are not
    /// those last hashed, and no other. The copy starts 40 bytes into its
    /// piece 0, and the passes are dispatched as `Host::card_before`
    /// dispatches them (`crate::host::hash_parts`).
    #[test]
    fn warp_wide_and_narrow_card_compares_flag_the_same_pieces() {
        use crate::host::{CARD_COMPARE, CARD_COMPARE_WIDE, hash_parts};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compares = [(CARD_COMPARE, false), (CARD_COMPARE_WIDE, true)].map(|(source, wide)| {
            (
                compile_cs(&warp.device, source).expect("the card compare must compile on WARP"),
                wide,
            )
        });
        const PIECE: usize = 64 << 10;
        const SKEW: usize = 40;
        const PIECES: usize = 6;
        // The dwords of the results before the first piece's.
        const AT: usize = 3;
        let size = (PIECES - 1) * PIECE;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let sides = [(); 2].map(|()| {
            (
                warp.shared_buffer(16 * PIECES as u64),
                warp.shared_buffer(4 * (AT + PIECES) as u64),
            )
        });
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // The bytes a pass hashes for a piece: the piece as the shaders cut
        // it, widened to 16 bytes at both ends.
        let hashed = |piece: usize| {
            let (lo, hi) = (
                (piece * PIECE).saturating_sub(SKEW),
                ((piece + 1) * PIECE - SKEW).min(size),
            );
            (lo & !15)..((hi + 15) & !15)
        };
        let mut now = content.clone();
        // Each piece's bytes when a pass last stored its hash.
        let mut stored: Vec<Vec<u8>> = (0..PIECES)
            .map(|piece| now[hashed(piece)].to_vec())
            .collect();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut draw = |below: usize| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as usize % below
        };
        let mut changed: Vec<usize> = Vec::new();
        let (mut flagged, mut clean) = (0, 0);
        for pass in 0..120 {
            // The first pass stores every piece's hash; then one to three
            // bytes change, or the last one changed goes back.
            let (mode, first, pieces) = if pass == 0 {
                (0, 0, PIECES)
            } else {
                for _ in 0..1 + draw(3) {
                    let at = match changed.pop() {
                        Some(at) if draw(3) == 0 => at,
                        back => {
                            changed.extend(back);
                            let at = draw(size);
                            changed.push(at);
                            at
                        }
                    };
                    now[at] = if now[at] == content[at] {
                        !content[at]
                    } else {
                        content[at]
                    };
                    // SAFETY: inside the mapping.
                    unsafe { now_ptr.add(at).write_volatile(now[at]) };
                }
                let first = draw(PIECES);
                ([0, 1, 1, 2, 2][draw(5)], first, 1 + draw(PIECES - first))
            };
            // SAFETY: our open list, live buffers, the compares' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                for ((compare, wide), ((_, _, hashes_gpu), (_, differs_ptr, differs_gpu))) in
                    compares.iter().zip(&sides)
                {
                    for dword in 0..AT + PIECES {
                        differs_ptr.cast::<u32>().add(dword).write_volatile(7);
                    }
                    warp.list.SetComputeRootSignature(&compare.root);
                    warp.list.SetPipelineState(&compare.pso);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, *hashes_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, *differs_gpu);
                    for (skip, own, groups) in hash_parts(*wide, pieces as u32) {
                        let constants = [
                            first as u32 + skip,
                            SKEW as u32,
                            size as u32,
                            own,
                            (AT as u32 + skip) * 4,
                            mode,
                        ];
                        warp.list
                            .SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        warp.list.Dispatch(groups, 1, 1);
                    }
                }
            }
            warp.submit_and_wait();
            // What both must have left in their results: 7 where a pass
            // stores nothing.
            let mut wanted = vec![7u32; AT + PIECES];
            for i in 0..pieces {
                let piece = first + i;
                if mode != 0 {
                    wanted[AT + i] = u32::from(stored[piece] != now[hashed(piece)]);
                    flagged += wanted[AT + i];
                    clean += 1 - wanted[AT + i];
                }
                if mode != 1 {
                    stored[piece] = now[hashed(piece)].to_vec();
                }
            }
            for ((_, wide), (_, (_, differs_ptr, _))) in compares.iter().zip(&sides) {
                // SAFETY: the mapping holds these dwords.
                let got: Vec<u32> = (0..AT + PIECES)
                    .map(|dword| unsafe { differs_ptr.cast::<u32>().add(dword).read_volatile() })
                    .collect();
                assert_eq!(
                    got, wanted,
                    "pass {pass} (kind {mode}, {pieces} pieces from {first}), a group per piece: {wide}"
                );
            }
        }
        assert!(
            flagged > 50 && clean > 50,
            "the passes flagged {flagged} pieces and found {clean} unchanged"
        );
    }

    /// The passes before a submission's lists as `Host::card_before` records
    /// them with `VICTORY_CARD_HASH_WIDE=1`, on the software adapter: every
    /// range sent up to two copies in the card's memory first, one barrier
    /// for both copies, then the passes that hash them. The hashes are
    /// those of the bytes just sent up: a compare finds nothing changed,
    /// flags a piece whose bytes went up with no pass behind them, and
    /// finds nothing again once that range was sent up the same way.
    #[test]
    fn warp_wide_hash_passes_follow_all_copies_behind_one_barrier() {
        use crate::host::{CARD_COMPARE_WIDE, hash_parts};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, CARD_COMPARE_WIDE)
            .expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 2;
        let size = PIECE * PIECES as usize;
        // Per copy: the card's side, the CPU's side (mapped), the hashes.
        let copies = [(); 2].map(|()| {
            (
                warp.default_uav(size as u64),
                warp.shared_buffer(size as u64),
                warp.default_uav(16 * u64::from(PIECES)),
            )
        });
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(4 * 2 * u64::from(PIECES));
        for (n, (_, (_, staging_ptr, _), _)) in copies.iter().enumerate() {
            let content: Vec<u8> = (0..size)
                .map(|i| (i * 31 + i / 251 + n * 7) as u8)
                .collect();
            // SAFETY: the mapping holds `size` bytes.
            unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), *staging_ptr, size) };
        }
        // One pass of kind `mode` over each copy, the second copy's results
        // behind the first's.
        let passes = |warp: &WarpHarness, mode: u32| {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                for (n, (card, _, hashes)) in copies.iter().enumerate() {
                    warp.list
                        .SetComputeRootShaderResourceView(1, card.GetGPUVirtualAddress());
                    warp.list
                        .SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
                    for (skip, own, groups) in hash_parts(true, PIECES) {
                        let constants = [
                            skip,
                            0,
                            size as u32,
                            own,
                            (n as u32 * PIECES + skip) * 4,
                            mode,
                        ];
                        warp.list
                            .SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        warp.list.Dispatch(groups, 1, 1);
                    }
                }
            }
        };
        // The ranges go up to both copies and, with `hash`, are hashed
        // behind one barrier; then, in a list of its own as after a
        // submission's lists, the compare. What it left in its results.
        let send = |warp: &mut WarpHarness, range: (usize, usize), hash: bool| -> [u32; 4] {
            let (from, to) = range;
            // SAFETY: our open list and live buffers; the ranges lie in
            // them. Both copies are in the common state when a list starts
            // (buffers return to it after every run), a copy into them
            // makes them copy destinations, and the barrier names that.
            unsafe {
                for (card, (staging, _, _), _) in &copies {
                    warp.list.CopyBufferRegion(
                        card,
                        from as u64,
                        staging,
                        from as u64,
                        (to - from) as u64,
                    );
                }
                if hash {
                    let barriers = copies
                        .iter()
                        .map(|(card, _, _)| {
                            transition(
                                card,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )
                        })
                        .collect::<Vec<_>>();
                    warp.list.ResourceBarrier(&barriers);
                    passes(warp, 0);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above; the mapping holds four dwords.
            unsafe {
                for dword in 0..4 {
                    differs_ptr.cast::<u32>().add(dword).write_volatile(7);
                }
                warp.list.ResourceBarrier(&[uav_barrier()]);
            }
            passes(warp, 1);
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2, 3]
                .map(|dword| unsafe { differs_ptr.cast::<u32>().add(dword).read_volatile() })
        };
        assert_eq!(send(&mut warp, (0, size), true), [0, 0, 0, 0]);
        // A byte of the second copy's second piece.
        let at = PIECE + 4321;
        // SAFETY: inside the mapping.
        unsafe {
            copies[1]
                .1
                .1
                .add(at)
                .write_volatile(!copies[1].1.1.add(at).read_volatile())
        };
        assert_eq!(
            send(&mut warp, (at & !15, (at & !15) + 16), false),
            [0, 0, 0, 1]
        );
        assert_eq!(
            send(&mut warp, (at & !15, (at & !15) + 16), true),
            [0, 0, 0, 0]
        );
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// So does the compare of copies in the card's memory, as the host
    /// makes it: with a thread per piece, or with a group per piece when
    /// the tests run with `VICTORY_CARD_HASH_WIDE=1`.
    #[test]
    fn card_compare_builds() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        assert!(host.card_ready());
    }

    #[test]
    fn only_an_execution_of_no_records_ends_early() {
        assert!(executes_nothing(0));
        assert!(!executes_nothing(1));
        assert!(!executes_nothing(64));
        assert!(!executes_nothing(u32::MAX));
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        two_clamps_in_one_list(SOURCE, false);
    }

    /// The same two calls with the scratch buffer kept readable
    /// (`VICTORY_INDIRECT_READABLE=1`): no return to the common state after
    /// an execution, and the second clamp takes the scratch buffer straight
    /// from indirect arguments back to unordered access. Every record runs
    /// and is charged as without the switch.
    #[test]
    fn warp_second_clamp_takes_a_readable_scratch_buffer_straight_back() {
        two_clamps_in_one_list(SOURCE, true);
    }

    /// The same two calls through the clamp that charges once per thread
    /// group (`VICTORY_CLAMP_GROUP=1`), with the scratch buffer returned to
    /// common and kept readable: of the 64 groups each call dispatches, 63
    /// lie past the count and leave the budget alone, and the one with
    /// records charges what its records cost, as record by record.
    #[test]
    fn warp_group_clamp_charges_only_the_records_below_the_count() {
        two_clamps_in_one_list(GROUP_SOURCE, false);
        two_clamps_in_one_list(GROUP_SOURCE, true);
    }

    /// Two counted calls in one list on WARP, clamped by `source`; `keep`:
    /// the scratch buffer stays readable as arguments between them, as a
    /// list records them with `VICTORY_INDIRECT_READABLE=1`.
    fn two_clamps_in_one_list(source: &str, keep: bool) {
        let Some(mut warp) = WarpHarness::with(source) else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        // Where the scratch buffer rests between the calls.
        let rest = if keep {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            0,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        if !keep {
            done_on(&warp.list, &dst_buf);
        }

        // Snapshot `dst_buf` after Call #1 (from where it rests, copy
        // 5 * STRIDE bytes, then return `dst_buf` there before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by
        // `done_on`, or left in `INDIRECT_ARGUMENT` by the clamp.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                rest,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                rest,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        // Kept readable, the scratch buffer goes straight from arguments
        // back to unordered access.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, rest),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            0,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        if !keep {
            done_on(&warp.list, &dst_buf);
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// One dispatch in parts: groups past what the host runs in one
    /// dimension go into the second record; what nothing wrote stays
    /// clamped, or empty.
    #[test]
    fn warp_large_dispatch_is_made_in_parts() {
        large_dispatch_in_parts(SOURCE);
    }

    /// The tests of one clamp that take a `source`, through the clamp that
    /// charges once per thread group (`VICTORY_CLAMP_GROUP=1`): a dispatch
    /// in parts, a window clamped ahead, a budget that runs out, and
    /// windows read with no barrier between their executions come out as
    /// they do record by record.
    #[test]
    fn warp_group_clamp_makes_a_large_dispatch_in_parts() {
        large_dispatch_in_parts(GROUP_SOURCE);
    }

    #[test]
    fn warp_group_clamp_leaves_unfilled_window_slots_empty_and_uncharged() {
        window_slots(GROUP_SOURCE);
    }

    #[test]
    fn warp_group_clamp_saturates_the_budget_without_wraparound() {
        budget_saturates(GROUP_SOURCE);
    }

    #[test]
    fn warp_group_clamp_window_executions_read_their_records() {
        window_executions(GROUP_SOURCE);
    }

    fn large_dispatch_in_parts(source: &str) {
        let Some(mut warp) = WarpHarness::with(source) else {
            return;
        };
        let cases: [([u32; 3], [u32; 6]); 6] = [
            ([115_200, 1, 1], [65_535, 1, 1, 49_665, 1, 1]),
            ([65_535, 1, 1], [65_535, 1, 1, 0, 1, 1]),
            ([65_536, 1, 1], [65_535, 1, 1, 1, 1, 1]),
            ([5, 3, 2], [5, 3, 2, 0, 3, 2]),
            ([0, 1, 1], [0, 1, 1, 0, 1, 1]),
            // Past both parts: as before, what one dispatch runs.
            ([200_000, 1, 1], [65_535, 1, 1, 0, 1, 1]),
        ];
        for (asked, parts) in cases {
            let (src_buf, src_ptr, _) = warp.shared_buffer(16);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(64);
            let dst_buf = warp.default_uav(64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, 64);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (2, 12, 1),
                IN_PARTS,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list and live buffers.
            unsafe {
                warp.list
                    .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 24)
            };
            warp.submit_and_wait();
            // SAFETY: the mapping holds 24 bytes the copy wrote.
            let made: [u32; 6] = unsafe { readback_ptr.cast::<[u32; 6]>().read_volatile() };
            assert_eq!(made, parts, "{asked:?}");
        }
    }

    /// The two words the clamp notes of indirect compute records (the
    /// largest group count in x asked for, the records cut to what one
    /// dispatch runs) are written unless the clamp is compiled without its
    /// notes (`VICTORY_CLAMP_LEAN=1`, [`lean`]): that one leaves them zero,
    /// and clamps and charges the same. So for both clamps, the one that
    /// charges record by record and the one that charges once per thread
    /// group.
    #[test]
    fn warp_compute_notes_are_left_out_by_the_lean_clamp_only() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        for (grouped, notes) in [(false, false), (false, true), (true, false), (true, true)] {
            let source = if grouped { GROUP_SOURCE } else { SOURCE };
            let clamp =
                make_from(&warp.device, source, notes).expect("the clamp must compile either way");
            let (src_buf, src_ptr, _) = warp.shared_buffer(16);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(64);
            let dst_buf = warp.default_uav(64);
            let asked = [100_000u32, 1, 1];
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(asked.as_ptr().cast(), src_ptr, 12);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, 64);
            }
            assert!(record_clamp_as(
                &clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (2, 12, 1),
                0,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list and live buffers.
            unsafe {
                warp.list
                    .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, 12)
            };
            warp.submit_and_wait();
            // SAFETY: the work is done; the mappings hold these words.
            let (made, words) = unsafe {
                (
                    readback_ptr.cast::<[u32; 3]>().read_volatile(),
                    budget_ptr.cast::<[u32; 5]>().read_volatile(),
                )
            };
            let which = format!("by group {grouped}, notes {notes}");
            assert_eq!(made, [65_535, 1, 1], "{which}");
            assert_eq!(words[0], 65_535 * 64, "{which}");
            let noted = if notes { [100_000, 1] } else { [0, 0] };
            assert_eq!(words[3..], noted, "{which}");
        }
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        window_slots(SOURCE);
    }

    fn window_slots(source: &str) {
        let Some(mut warp) = WarpHarness::with(source) else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                if ahead { AHEAD } else { 0 },
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        budget_saturates(SOURCE);
    }

    fn budget_saturates(source: &str) {
        let Some(mut warp) = WarpHarness::with(source) else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            0,
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        scratch_across_submissions(false);
    }

    /// The same with the scratch buffer kept readable
    /// (`VICTORY_INDIRECT_READABLE=1`): the list's tracked state of it
    /// across clamps, growth and a reset.
    #[test]
    fn host_list_keeps_its_scratch_readable_across_clamps_growth_and_reset() {
        scratch_across_submissions(true);
    }

    /// Clamps into a host list's scratch buffer, grows it, and runs the
    /// list twice. `keep`: the scratch buffer stays readable as arguments
    /// after a clamp (`VICTORY_INDIRECT_READABLE=1`); otherwise each
    /// execution's [`done`] returns it to the common state.
    fn scratch_across_submissions(keep: bool) {
        // Forced on, the switch's barriers run on a software adapter only.
        let host = if keep { software_host() } else { super::host() };
        let Some(host) = host else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }
        let common = D3D12_RESOURCE_STATE_COMMON;
        // Where a clamp and the end of its execution leave the buffer.
        let rest = if keep {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            common
        };
        let clamped = |at: u64, spec: (u32, u32, u32)| {
            let count = Some((&count_buf, 0));
            let scratch = bound_as(&list, (&src_buf, at), count, spec, (0, keep)).unwrap();
            done(&list, &scratch);
            scratch
        };

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`, which with `keep`
        // stays readable as arguments between them.
        let b1 = clamped(0, (0, 16, 64));
        assert_eq!(
            list.indirect_state(&b1),
            rest,
            "where a clamp and its execution leave the scratch buffer"
        );
        let b2 = clamped(64, (1, 20, 128));
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2), rest);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let grown = list.indirect_scratch(5000 * 20).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            common,
            "a new scratch buffer is common"
        );
        let b3 = clamped(0, (1, 20, 5000));
        assert_eq!(b3.as_raw(), grown.as_raw());
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3), rest);
        assert_eq!(
            list.indirect_state(&b1),
            common,
            "the replaced buffer went back to common"
        );

        list.close();
        assert_eq!(
            list.indirect_state(&b3),
            common,
            "a closed list leaves its scratch buffer common"
        );
        // With the debug layer on (`VICTORY_D3D12_DEBUG=1`, diagnostics):
        // every barrier recorded named the state the buffer was in. Read
        // before the submission, which moves the messages to the log.
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations, from the common state again.
        assert_eq!(
            list.indirect_state(&b3),
            common,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = clamped(0, (1, 20, 4096));
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4), rest);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty. (Racing for the
    /// word record by record, which of them fits is up to the card.)
    #[test]
    fn warp_a_group_charges_its_records_in_lane_order() {
        let Some(mut warp) = WarpHarness::with(GROUP_SOURCE) else {
            return;
        };
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;
        const BUDGET_CAP: u32 = 1 << 27;
        let counts = [100u32, 250, 50, 100];
        let total = counts.len() * STRIDE as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            for (i, &count) in counts.iter().enumerate() {
                let words = [count, 1, 7, 9];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STRIDE as usize),
                    16,
                );
            }
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (budget_ptr as *mut u32).write_volatile(BUDGET_CAP - 300);
        }
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DRAW, STRIDE, counts.len() as u32),
            0,
        ));
        // SAFETY: an open list; the clamped records are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                    .to_vec(),
            )
        };
        assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits");
        for i in 1..counts.len() {
            assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
        }
        assert_eq!(
            spent, BUDGET_CAP,
            "the record that did not fit took the rest"
        );
    }

    /// Which clamp the lists take ([`pick`]): without the switch the
    /// per-record one, and the group clamp is not made at all; with
    /// `VICTORY_CLAMP_GROUP=1` the group clamp, and if the host does not
    /// make it the per-record one after all, so indirect executions go on.
    /// None only if that one fails too.
    #[test]
    fn a_group_clamp_that_cannot_be_made_falls_back_to_the_record_clamp() {
        let Some(warp) = WarpHarness::new() else {
            return;
        };
        // Whether a clamp came out, and which were asked for (true: the
        // group clamp), when the host refuses the ones in `refused`.
        let tried = |grouped: bool, refused: &[&str]| {
            let asked = std::cell::RefCell::new(Vec::new());
            let made = pick(grouped, |source| {
                asked.borrow_mut().push(source == GROUP_SOURCE);
                if refused.contains(&source) {
                    return Err(windows::core::Error::empty());
                }
                make_from(&warp.device, source, true)
            });
            (made.is_some(), asked.into_inner())
        };
        assert_eq!(tried(false, &[]), (true, vec![false]));
        assert_eq!(tried(true, &[]), (true, vec![true]));
        assert_eq!(tried(true, &[GROUP_SOURCE]), (true, vec![true, false]));
        assert_eq!(
            tried(true, &[GROUP_SOURCE, SOURCE]),
            (false, vec![true, false])
        );
        assert_eq!(tried(false, &[SOURCE]), (false, vec![false]));
    }

    /// A command signature of one dispatch record of `stride` bytes.
    fn dispatch_signature(device: &ID3D12Device, stride: u32) -> ID3D12CommandSignature {
        signature_of(
            device,
            D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.0 as u32,
            stride,
        )
    }

    /// The list's budget starts empty at each submission, wherever it
    /// lives: a recording whose clamps spend all of it is followed, after
    /// the reset, by one whose small record still runs.
    #[test]
    fn host_list_budget_starts_empty_at_each_submission() {
        budget_starts_empty(false);
    }

    /// The same with the budget in the card's own memory
    /// (`VICTORY_INDIRECT_CARD_BUDGET=1`), which the list zeroes itself
    /// before its first clamp.
    #[test]
    fn host_list_card_budget_starts_empty_at_each_submission() {
        budget_starts_empty(true);
    }

    /// `card`: the list's budget is in the card's own memory whatever the
    /// switch says (on a software adapter only: [`software_host`]);
    /// otherwise where the switch puts it.
    fn budget_starts_empty(card: bool) {
        let host = if card { software_host() } else { super::host() };
        let Some(host) = host else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        if card {
            assert!(list.force_card_budget(), "no budget in the card's memory");
        }
        let _ = host.log_messages();
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let signature = dispatch_signature(&host.device, STRIDE);
        // Three records of 65,535 x 16 groups: about half the budget each.
        let Some((big, big_ptr)) = host.shared_buffer(3 * u64::from(STRIDE)) else {
            return;
        };
        let Some((small, small_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            for i in 0..3 {
                let words = [65535u32, 16, 1];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    big_ptr.add(i * STRIDE as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), small_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        // Clamps the small record and executes it, as
        // `State::execute_indirect` does; then the list runs.
        let small_runs = || {
            let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
            // SAFETY: an open list; the worker's root signature and
            // pipeline; buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list
                    .ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
            }
            done(&list, &scratch);
            list.close();
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            unsafe { (hits_ptr as *const u32).read_volatile() }
        };
        for round in 0..2 {
            // Spends the budget (clamped, not executed), then one group.
            let spent = bound(&list, (&big, 0), None, (KIND_DISPATCH, STRIDE, 3)).unwrap();
            done(&list, &spent);
            // The small record comes after the big ones spent the budget:
            // it runs empty, in every round alike (none carries over).
            assert_eq!(
                small_runs(),
                0,
                "round {round}: a spent budget empties the next record"
            );
        }
        // A recording with only the small record: the budget is empty
        // again, so it runs.
        assert_eq!(
            small_runs(),
            1,
            "the budget starts empty at the next submission"
        );
        assert_eq!(list.budget_on_card(), card || crate::host::card_budget());
    }

    /// What the clamp notes of a list's records (mesh: the largest group
    /// count asked for and the groups asked in all; compute, unless the
    /// tests run with `VICTORY_CLAMP_LEAN=1`: the largest x asked for and
    /// the records cut) is in the list's mapped words once it has run,
    /// where the next submission takes it for the frame line.
    #[test]
    fn host_list_budget_notes_are_in_the_mapped_words_after_a_run() {
        budget_notes(false);
    }

    /// The same with the budget in the card's own memory
    /// (`VICTORY_INDIRECT_CARD_BUDGET=1`): the list copies the notes to
    /// the mapped words when it closes, and the mapped budget word itself
    /// is not charged. The closed list runs twice in one call to the
    /// queue: it leaves both budget buffers as it found them, and its
    /// second run zeroes the card's copy again, so the notes are those of
    /// one run.
    #[test]
    fn host_list_card_budget_copies_its_notes_to_the_mapped_words() {
        budget_notes(true);
    }

    /// `card`: as for [`budget_starts_empty`].
    fn budget_notes(card: bool) {
        let host = if card { software_host() } else { super::host() };
        let Some(host) = host else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        if card {
            assert!(list.force_card_budget(), "no budget in the card's memory");
        }
        let _ = host.log_messages();
        const KIND_DISPATCH: u32 = 2;
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 12;
        let Some((mesh, mesh_ptr)) = host.shared_buffer(2 * u64::from(STRIDE)) else {
            return;
        };
        let Some((compute, compute_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            let records = [5u32, 1, 1, 519, 1, 1];
            std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), mesh_ptr, 24);
            let record = [100_000u32, 1, 1];
            std::ptr::copy_nonoverlapping(record.as_ptr().cast::<u8>(), compute_ptr, 12);
        }
        // Twice: the second recording finds the words of the first, which
        // its submission takes and zeroes before it runs.
        for round in 0..2 {
            let scratch = bound(&list, (&mesh, 0), None, (KIND_MESH, STRIDE, 2)).unwrap();
            done(&list, &scratch);
            let scratch = bound(&list, (&compute, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
            done(&list, &scratch);
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            if card {
                host.execute(&[&list, &list]);
            } else {
                host.execute(&[&list]);
            }
            list.reset();
            let on_card = list.budget_on_card();
            assert_eq!(on_card, card || crate::host::card_budget());
            let words = list.budget_words().unwrap();
            // The compute record is cut to 65,535 groups of 64 threads.
            let charged = (5 + 519) * 128 + 65_535 * 64;
            assert_eq!(
                words[0],
                if on_card { 0 } else { charged },
                "round {round}: the budget is charged where it lives"
            );
            assert_eq!(
                words[1..3],
                [519, 524],
                "round {round}: the mesh records' notes"
            );
            let noted = if lean() { [0, 0] } else { [100_000, 1] };
            assert_eq!(
                words[3..],
                noted,
                "round {round}: the compute record's notes"
            );
        }
    }

    /// [`bound`], [`bound_ahead`] and [`bound_split`] take the mode from the
    /// switch ([`readable`]): unset, every clamp's execution ends with the
    /// scratch buffer common; with `VICTORY_INDIRECT_READABLE=1` in the
    /// test's environment it stays readable until the list closes. Where
    /// the budget they charge lives follows its switch too
    /// (`VICTORY_INDIRECT_CARD_BUDGET=1`: the card's own memory).
    #[test]
    fn host_list_clamps_follow_the_switch() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: the mapping holds 256 bytes.
        unsafe { std::ptr::write_bytes(src_ptr, 0, 256) };
        let rest = if readable() {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        };
        // Each clamp, then what follows its execution.
        let executed = |scratch: Option<ID3D12Resource>| {
            let scratch = scratch.unwrap();
            done(&list, &scratch);
            assert_eq!(list.indirect_state(&scratch), rest);
        };
        executed(bound(&list, (&src_buf, 0), None, (2, 12, 1)));
        executed(bound_ahead(&list, (&src_buf, 0), (10, 32, 4)));
        executed(bound_split(&list, (&src_buf, 0)));
        assert_eq!(list.budget_on_card(), crate::host::card_budget());
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// The clamp of a batch of mesh draws (`VICTORY_GROUPED_MESH_CLAMP=1`)
    /// reads the records its words name, in that order, and no others:
    /// each bounded as a mesh record clamped alone, charged in draw order
    /// (the first fits, the second does not and takes the rest, the third
    /// costs nothing), the notes of the three in the budget's next words.
    /// The offsets are written after the clamp is recorded, before the list
    /// runs, as the draws of a batch add them; the fourth offset, past the
    /// count, names a slot of all ones that is never read.
    #[test]
    fn warp_mesh_batch_charges_only_requested_records_in_draw_order() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let clamp =
            make_from(&warp.device, MESH_BATCH_SOURCE, false).expect("batch clamp must compile");
        let (_src, src, src_gpu) = warp.shared_buffer(256);
        let (_words, words, words_gpu) = warp.shared_buffer(256);
        let (_budget, budget, budget_gpu) = warp.shared_buffer(256);
        let (readback, output, _) = warp.shared_buffer(256);
        let dst = warp.default_uav(512);
        // SAFETY: mapped test buffers of 256 bytes; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(src, 0xff, 256);
            std::ptr::write_bytes(budget, 0, 256);
            std::ptr::write_bytes(words, 0, 256);
            src.add(16).cast::<[u32; 3]>().write([3, 1, 1]);
            src.add(48).cast::<[u32; 3]>().write([2, 2, 2]);
            src.add(80).cast::<[u32; 3]>().write([0, 0, 1]);
            budget.cast::<u32>().write((1 << 27) - 3 * 128);
        }
        // SAFETY: a live buffer.
        let dst_gpu = unsafe { dst.GetGPUVirtualAddress() };
        let addresses = [src_gpu, words_gpu, dst_gpu, budget_gpu];
        let common = D3D12_RESOURCE_STATE_COMMON;
        let waits = Waits::both();
        record_gpu_clamp(
            &clamp,
            &warp.list,
            waits,
            (&dst, common),
            [0; 6],
            addresses,
            1,
        );
        // SAFETY: the mapped words, written before the list runs; an open
        // list, and the scratch buffer in INDIRECT_ARGUMENT after the clamp.
        unsafe {
            words.cast::<[u32; 5]>().write([3, 48, 16, 80, 0]);
            done_on(&warp.list, &dst);
            warp.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback, 0, &dst, 0, 48);
        }
        warp.submit_and_wait();
        // SAFETY: the list has run; mapped buffers.
        unsafe {
            assert_eq!(
                output.cast::<[u32; 12]>().read(),
                [2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                "the records in draw order, the slot past the count untouched"
            );
            assert_eq!(
                budget.cast::<[u32; 5]>().read(),
                [1 << 27, 3, 5, 0, 0],
                "spent, the largest group count asked for, the groups asked in all"
            );
        }
    }

    /// The mesh records of [`batches_match_single_clamps`]: 40 of them, 32
    /// bytes apart as the title keeps them (the rest of each record ones):
    /// full dispatches, slots of all ones, ones over 65,535 groups, small
    /// ones and ones that cost nothing, so the budget runs out part way.
    fn mesh_records() -> Vec<[u32; 3]> {
        (0..40u32)
            .map(|i| {
                let x = match i % 5 {
                    0 => 65_535,
                    1 => u32::MAX,
                    2 => 40_000 + i,
                    3 => 3,
                    _ => 70_000,
                };
                let y = if i % 7 == 6 { 0 } else { 1 + i % 3 };
                [x, y, 1]
            })
            .collect()
    }

    /// What clamps of their own make of `records` in turn, from an empty
    /// budget: each clamped record, and the budget's words after them
    /// (spent; the largest group count asked for and the groups asked in
    /// all; no compute notes).
    fn clamped_alone(records: &[[u32; 3]]) -> (Vec<[u32; 3]>, [u32; 5]) {
        const BUDGET: u32 = 1 << 27;
        let (mut spent, mut largest, mut all) = (0u32, 0u32, 0u32);
        let out = records
            .iter()
            .map(|&[x, y, z]| {
                largest = largest.max(x);
                all = all.wrapping_add(x.min(1 << 24));
                let g = [x.min(65_535), y.min(1), z.min(1)];
                let cost = g[0] * g[1] * g[2] * 128;
                let fits = cost == 0 || (spent < BUDGET && BUDGET - spent >= cost);
                if cost > 0 && spent < BUDGET {
                    spent = if BUDGET - spent <= cost {
                        BUDGET
                    } else {
                        spent + cost
                    };
                }
                if fits { g } else { [0; 3] }
            })
            .collect();
        (out, [spent, largest, all, 0, 0])
    }

    /// A batch of mesh draws (`VICTORY_GROUPED_MESH_CLAMP=1`) gives every
    /// record what a clamp of its own gives it, and charges the budget as
    /// those do, with each of the clamp's switches on and off: the scratch
    /// buffer kept readable (`VICTORY_INDIRECT_READABLE=1`), the clamp that
    /// charges once per thread group (`VICTORY_CLAMP_GROUP=1`), the budget
    /// in the card's own memory (`VICTORY_INDIRECT_CARD_BUDGET=1`), and the
    /// clamp without its compute notes (`VICTORY_CLAMP_LEAN=1`). 40 draws,
    /// so a full batch ends and a second one opens; the budget runs out
    /// part way, so later records run empty in both.
    #[test]
    fn host_list_mesh_batches_match_single_clamps_under_every_switch() {
        let Some(host) = software_host() else {
            return;
        };
        let batch = make_from(&host.device, MESH_BATCH_SOURCE, false).expect("batch clamp");
        for keep in [false, true] {
            for card in [false, true] {
                for grouped_source in [false, true] {
                    for notes in [true, false] {
                        let source = if grouped_source { GROUP_SOURCE } else { SOURCE };
                        let single = make_from(&host.device, source, notes).expect("clamp");
                        let how = format!(
                            "readable {keep}, card budget {card}, group clamp {grouped_source}, \
                             notes {notes}"
                        );
                        batches_match_single_clamps(host, (&single, &batch), keep, card, &how);
                    }
                }
            }
        }
    }

    /// [`host_list_mesh_batches_match_single_clamps_under_every_switch`]
    /// with the single clamp `single`, the batch clamp `batch`, the scratch
    /// buffer kept readable (`keep`) and the budget in the card's memory
    /// (`card`).
    fn batches_match_single_clamps(
        host: &crate::host::Host,
        (single, batch): (&Clamp, &Clamp),
        keep: bool,
        card: bool,
        how: &str,
    ) {
        const KIND_MESH: u32 = 10;
        const STEP: u32 = 32;
        const COPY: u64 = 7;
        let records = mesh_records();
        let n = records.len();
        let Some((src, src_ptr)) = host.shared_buffer(n as u64 * u64::from(STEP)) else {
            return;
        };
        // SAFETY: a mapped buffer of this size; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xff, n * STEP as usize);
            for (i, record) in records.iter().enumerate() {
                src_ptr
                    .add(i * STEP as usize)
                    .cast::<[u32; 3]>()
                    .write_unaligned(*record);
            }
        }
        // Runs `n` draws' clamps (`each`: the scratch buffer and where the
        // draw reads its record) on a new list, copying every clamped
        // record out as a draw would read it; the records and the budget's
        // words once the list has run.
        let run = |each: &dyn Fn(&HostList, u32) -> (ID3D12Resource, u64)| {
            let list = host.new_list().expect("a host list");
            if card {
                assert!(list.force_card_budget(), "no budget in the card's memory");
            }
            let _ = host.log_messages();
            let (out, out_ptr) = host
                .shared_buffer(n as u64 * 12)
                .expect("a read-back buffer");
            let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
            for i in 0..n as u32 {
                let (bounded, at) = each(&list, i * STEP);
                // SAFETY: an open list; the scratch buffer is in
                // INDIRECT_ARGUMENT after a clamp (and through a batch),
                // and goes back there; buffers held until the list ran.
                unsafe {
                    list.list
                        .ResourceBarrier(&[transition(&bounded, args, copy)]);
                    list.list
                        .CopyBufferRegion(&out, u64::from(i) * 12, &bounded, at, 12);
                    list.list
                        .ResourceBarrier(&[transition(&bounded, copy, args)]);
                }
                list.keep(&bounded);
            }
            list.close();
            assert!(
                list.mesh_batch().is_none(),
                "{how}: the close ended the batch"
            );
            assert_eq!(
                host.log_messages(),
                None,
                "{how}: the debug layer found an error"
            );
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer of `n` records.
            let clamped = unsafe {
                std::slice::from_raw_parts(out_ptr.cast::<[u32; 3]>().cast_const(), n).to_vec()
            };
            let words = list.budget_words().expect("budget words");
            (clamped, words, list.budget_on_card())
        };
        let alone = run(&|list, offset| {
            let budget = list.budget().unwrap();
            let dst = list.indirect_scratch(12).unwrap();
            let src = (&src, u64::from(offset));
            let spec = (KIND_MESH, crate::indirect::MESH_STRIDE, 1);
            assert!(clamp_into(
                list,
                single,
                &dst,
                budget,
                src,
                None,
                spec,
                (0, keep)
            ));
            done(list, &dst);
            (dst, 0)
        });
        let opened = std::cell::RefCell::new(Vec::new());
        let grouped = run(&|list, offset| {
            let (bounded, at, new) = grouped_mesh_as(list, COPY, offset, |list| {
                MeshBatch::open_with(list, batch, COPY, &src, keep)
            })
            .expect("a batch");
            opened.borrow_mut().push(new);
            assert_eq!(
                list.indirect_state(&bounded),
                if keep {
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
                } else {
                    D3D12_RESOURCE_STATE_COMMON
                },
                "{how}: the list's tracked state of the scratch buffer in a batch"
            );
            (bounded, at)
        });
        let (expected, words) = clamped_alone(&records);
        assert!(
            expected.contains(&[0; 3]) && expected.iter().any(|r| r[0] > 0),
            "the records must both fit and run empty"
        );
        assert_eq!(alone.0, expected, "{how}: clamps of their own");
        assert_eq!(grouped.0, expected, "{how}: a batch");
        // On the card the spent word stays there; the notes come back.
        let charged = if alone.2 {
            [0, words[1], words[2], 0, 0]
        } else {
            words
        };
        assert_eq!(
            alone.1[..5],
            charged,
            "{how}: the budget of clamps of their own"
        );
        assert_eq!(grouped.1, alone.1, "{how}: the budget of a batch");
        assert_eq!(alone.2, grouped.2);
        let opens: Vec<usize> = (opened.borrow().iter().enumerate())
            .filter_map(|(i, &new)| new.then_some(i))
            .collect();
        assert_eq!(
            opens,
            [0, 32],
            "{how}: a full batch ends and the next opens"
        );
    }

    /// A batch of mesh draws takes the draws of one argument buffer, ends
    /// before another clamp is recorded (which writes the scratch buffer
    /// its draws read) and at the list's close, and leaves the scratch
    /// buffer where the list's tracking expects it, readable or common.
    #[test]
    fn host_list_ends_a_mesh_batch_before_another_clamp() {
        let Some(host) = software_host() else {
            return;
        };
        let batch = make_from(&host.device, MESH_BATCH_SOURCE, false).expect("batch clamp");
        let Some((src, src_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: the mapping holds 256 bytes.
        unsafe { std::ptr::write_bytes(src_ptr, 0, 256) };
        for keep in [false, true] {
            let list = host.new_list().expect("a host list");
            let _ = host.log_messages();
            let join = |offset| {
                grouped_mesh_as(&list, 1, offset, |list| {
                    MeshBatch::open_with(list, &batch, 1, &src, keep)
                })
                .expect("a batch")
            };
            let (first, _, opened) = join(0);
            assert!(opened);
            let (_, at, opened) = join(12);
            assert_eq!((at, opened), (12, false), "the next draw joins");
            // Another copy's record opens a batch of its own.
            let (_, at, opened) = grouped_mesh_as(&list, 2, 24, |list| {
                MeshBatch::open_with(list, &batch, 2, &src, keep)
            })
            .expect("a batch");
            assert_eq!((at, opened), (0, true), "another argument buffer");
            // A clamp of its own ends it first.
            let alone = bound_as(&list, (&src, 0), None, (10, 12, 1), (0, keep)).unwrap();
            assert!(list.mesh_batch().is_none(), "the clamp ended the batch");
            assert_eq!(alone.as_raw(), first.as_raw(), "one scratch buffer");
            done(&list, &alone);
            let rest = if keep {
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
            } else {
                D3D12_RESOURCE_STATE_COMMON
            };
            assert_eq!(list.indirect_state(&alone), rest);
            // One open at the close ends there, before the scratch buffer
            // goes back to common.
            join(36);
            assert!(list.mesh_batch().is_some());
            list.close();
            assert!(list.mesh_batch().is_none(), "the close ended the batch");
            assert_eq!(list.indirect_state(&alone), D3D12_RESOURCE_STATE_COMMON);
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
        }
    }

    /// PC argument kinds, for [`batch_calls`].
    const KIND_DRAW: u32 = 0;
    const KIND_INDEXED: u32 = 1;
    const KIND_DISPATCH: u32 = 2;
    const KIND_MESH: u32 = 10;

    /// Where an execution of [`batch_calls`] finds its count, if it has
    /// one: at a byte offset of the count buffer or of the argument buffer.
    #[derive(Clone, Copy)]
    enum Counted {
        No,
        InCounts(u64),
        InArgs(u64),
    }

    /// An execution of [`batch_calls`]: its records' offset in the argument
    /// buffer, kind, stride and records, its count, whether it is one
    /// dispatch in parts, and the count's value.
    struct Call {
        offset: u64,
        spec: (u32, u32, u32),
        count: Counted,
        parts: bool,
        counted: u32,
    }

    impl Call {
        /// The bytes an execution reads of its clamped records.
        fn read(&self) -> usize {
            if self.parts {
                return (SPLIT as usize) * PART_BYTES as usize;
            }
            let (_, stride, records) = self.spec;
            let active = match self.count {
                Counted::No => records,
                _ => records.min(self.counted),
            };
            (stride * active) as usize
        }
    }

    /// Synthetic executions of mixed kinds, strides and counts, with
    /// records the clamp must bound (out of range, or more than the host
    /// runs), written into `args` and `counts` (mapped): the executions.
    /// Every record fits the list's budget, so what each is clamped to
    /// does not depend on the order the card charges them in.
    fn batch_calls(args: *mut u8, counts: *mut u8) -> Vec<Call> {
        let mut calls = Vec::new();
        let mut at = 0u64;
        let put = |calls: &mut Vec<Call>, call: Call, words: &[u32]| {
            // SAFETY: the mapped buffers hold 8 KiB and 256 bytes, which
            // these executions stay within.
            unsafe {
                let to = args.add(call.offset as usize).cast::<u32>();
                for (i, w) in words.iter().enumerate() {
                    to.add(i).write_unaligned(*w);
                }
                match call.count {
                    Counted::No => {}
                    Counted::InCounts(off) => counts
                        .add(off as usize)
                        .cast::<u32>()
                        .write_unaligned(call.counted),
                    Counted::InArgs(off) => args
                        .add(off as usize)
                        .cast::<u32>()
                        .write_unaligned(call.counted),
                }
            }
            calls.push(call);
        };
        let mut next = |len: u64| {
            let here = at;
            at = (at + len).next_multiple_of(16);
            here
        };
        // Draws, three records, two counted (in the count buffer).
        let offset = next(48);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_DRAW, 16, 3),
                count: Counted::InCounts(0),
                parts: false,
                counted: 2,
            },
            &[3, 1, 0, 0, 0xffff_ffff, 0, 5, 6, 9, 9, 9, 9],
        );
        // Indexed draws: an index count past 2^24, then instances past
        // what 2^26 vertex instances allow.
        let offset = next(40);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_INDEXED, 20, 2),
                count: Counted::No,
                parts: false,
                counted: 0,
            },
            &[0xffff_ffff, 0, 1, 2, 3, 100, 1 << 30, 4, 5, 6],
        );
        // Dispatches, 70 records (two groups of the clamp): wide, too
        // many groups in all, and plain.
        let offset = next(70 * 12);
        let mut words = Vec::new();
        for i in 0..70u32 {
            words.extend_from_slice(match (i, i % 3) {
                (0, _) => &[70_000, 1, 1],
                (_, 1) => &[65_535, 65_535, 65_535],
                (_, 2) => &[0xffff_ffff, 0xffff_ffff, 0xffff_ffff],
                _ => &[3, 2, 1],
            });
        }
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_DISPATCH, 12, 70),
                count: Counted::No,
                parts: false,
                counted: 0,
            },
            &words,
        );
        // One mesh record of all ones (a slot nothing wrote).
        let offset = next(32);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_MESH, 32, 1),
                count: Counted::No,
                parts: false,
                counted: 0,
            },
            &[0xffff_ffff, 0xffff_ffff, 0xffff_ffff, 7, 7, 7, 7, 7],
        );
        // Dispatches 16 bytes apart, counted in the argument buffer.
        let offset = next(64);
        let count_at = next(4);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_DISPATCH, 16, 4),
                count: Counted::InArgs(count_at),
                parts: false,
                counted: 3,
            },
            &[1, 2, 3, 11, 70_000, 2, 1, 12, 4, 5, 6, 13, 7, 8, 9, 14],
        );
        // One dispatch in parts, past one dispatch's groups.
        let offset = next(12);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_DISPATCH, 12, 1),
                count: Counted::No,
                parts: true,
                counted: 0,
            },
            &[100_000, 1, 1],
        );
        // Draws counted past their records (in the count buffer).
        let offset = next(32);
        put(
            &mut calls,
            Call {
                offset,
                spec: (KIND_DRAW, 16, 2),
                count: Counted::InCounts(4),
                parts: false,
                counted: 5,
            },
            &[1 << 25, 1, 0, 0, 6, 1000, 0, 0],
        );
        calls
    }

    /// Executions that run the list's budget out, written into `args`
    /// (mapped) from byte `from` on: two dispatches of 1,024 by 1,023
    /// groups leave 131,072 vertex instances of the budget, then draws of
    /// 65,536, 65,536 and one (the last runs empty) and a dispatch (empty).
    fn spending_calls(args: *mut u8, from: u64) -> Vec<Call> {
        let plain = |offset: u64, spec: (u32, u32, u32)| Call {
            offset,
            spec,
            count: Counted::No,
            parts: false,
            counted: 0,
        };
        let calls = [
            (plain(from, (KIND_DISPATCH, 12, 1)), &[1024, 1023, 1][..]),
            (plain(from + 16, (KIND_DISPATCH, 12, 1)), &[1024, 1023, 1]),
            (
                plain(from + 32, (KIND_DRAW, 16, 3)),
                &[65_536, 1, 0, 0, 65_536, 1, 0, 0, 1, 1, 0, 0],
            ),
            (plain(from + 80, (KIND_DISPATCH, 12, 1)), &[3, 2, 1]),
        ];
        calls
            .into_iter()
            .map(|(call, words)| {
                // SAFETY: the mapped buffer holds 8 KiB; these executions
                // stay within 128 bytes of `from`, which the caller keeps
                // below that.
                unsafe {
                    let to = args.add(call.offset as usize).cast::<u32>();
                    for (i, w) in words.iter().enumerate() {
                        to.add(i).write_unaligned(*w);
                    }
                }
                call
            })
            .collect()
    }

    /// How a test clamps an execution of [`batch_calls`]: the scratch buffer,
    /// where the execution reads its records there, and whether a clamp was
    /// recorded for it.
    type Each<'a> = dyn Fn(&HostList, &Call) -> (ID3D12Resource, u64, bool) + 'a;

    /// The executions of [`batch_calls`] clamped in one batch give every
    /// record what clamps of their own give it, and charge the budget the
    /// same: against [`SOURCE`] and [`GROUP_SOURCE`], with and without the
    /// clamps' notes (`VICTORY_CLAMP_LEAN=1` builds the batch's clamp
    /// without), with the budget in mapped words and in the card's memory,
    /// with both waits and with `VICTORY_CLAMP_PRECISE=1`. One clamp is
    /// recorded for them all, each reads its own slot, and the debug layer
    /// stays silent. The executions of [`spending_calls`], each in a batch
    /// of its own (a note that what follows may write before each), run the
    /// budget out as [`GROUP_SOURCE`]'s clamps of their own do, in the order
    /// of the executions.
    #[test]
    fn host_list_batched_clamps_match_clamps_of_their_own() {
        let Some(host) = software_host() else {
            return;
        };
        let (Some((args, args_ptr)), Some((counts, counts_ptr))) =
            (host.shared_buffer(8 << 10), host.shared_buffer(256))
        else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(args_ptr, 0, 8 << 10);
            std::ptr::write_bytes(counts_ptr, 0, 256);
        }
        let calls = batch_calls(args_ptr, counts_ptr);
        let spending = spending_calls(args_ptr, 4 << 10);
        let count_of = |call: &Call| match call.count {
            Counted::No => None,
            Counted::InCounts(off) => Some((&counts, off)),
            Counted::InArgs(off) => Some((&args, off)),
        };
        let from_to = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        // Runs the clamps of `calls` on a new list (`each`: the scratch
        // buffer and where the execution reads its records; `apart`: a note
        // that what follows may write before each), copying what each
        // execution reads out; those bytes, the budget's words and whether
        // a clamp was recorded for each.
        let run = |calls: &[Call], apart: bool, card: bool, precise: bool, each: &Each<'_>| {
            let total: usize = calls.iter().map(|c| c.read().next_multiple_of(16)).sum();
            with_precise(precise, || {
                let list = host.new_list().expect("a host list");
                if card {
                    assert!(list.force_card_budget(), "no budget in the card's memory");
                }
                let _ = host.log_messages();
                let (out, out_ptr) = host
                    .shared_buffer(total as u64)
                    .expect("a read-back buffer");
                // As `State::execute_indirect` does before a clamp.
                list.buffer_read(&args);
                list.buffer_read(&counts);
                let mut opened = Vec::new();
                let mut to = 0u64;
                for call in calls {
                    if apart {
                        list.may_write();
                    }
                    let (bounded, at, new) = each(&list, call);
                    opened.push(new);
                    // SAFETY: an open list; the scratch buffer is in
                    // INDIRECT_ARGUMENT after a clamp, and goes back there;
                    // buffers held until the list ran.
                    unsafe {
                        list.list
                            .ResourceBarrier(&[transition(&bounded, from_to, copy)]);
                        list.list
                            .CopyBufferRegion(&out, to, &bounded, at, call.read() as u64);
                        list.list
                            .ResourceBarrier(&[transition(&bounded, copy, from_to)]);
                    }
                    list.keep(&bounded);
                    to += call.read().next_multiple_of(16) as u64;
                }
                list.close();
                let how = format!("card budget {card}, precise {precise}");
                assert_eq!(
                    host.log_messages(),
                    None,
                    "{how}: the debug layer found an error"
                );
                host.execute(&[&list]);
                list.reset();
                // SAFETY: the list has run; a mapped buffer of `total` bytes.
                let bytes =
                    unsafe { std::slice::from_raw_parts(out_ptr.cast_const(), total).to_vec() };
                (bytes, list.budget_words().expect("budget words"), opened)
            })
        };
        let word =
            |bytes: &[u8], at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        for grouped in [false, true] {
            for notes in [true, false] {
                let source = if grouped { GROUP_SOURCE } else { SOURCE };
                let single = make_from(&host.device, source, notes).expect("clamp");
                let batched = make_from(&host.device, BATCH_SOURCE, notes).expect("batch clamp");
                let alone = |list: &HostList, call: &Call| {
                    let budget = list.budget().unwrap();
                    let (spec, how, bytes) = if call.parts {
                        (
                            (KIND_DISPATCH, 12, 1),
                            IN_PARTS,
                            u64::from(SPLIT) * PART_BYTES,
                        )
                    } else {
                        let (_, stride, records) = call.spec;
                        (call.spec, 0, u64::from(stride * records))
                    };
                    let dst = list.indirect_scratch(bytes).unwrap();
                    let src = (&args, call.offset);
                    let count = count_of(call);
                    assert!(clamp_into(
                        list,
                        &single,
                        &dst,
                        budget,
                        src,
                        count,
                        spec,
                        (how, true)
                    ));
                    (dst, 0, true)
                };
                let together = |list: &HostList, call: &Call| {
                    let src = (&args, call.offset);
                    let count = count_of(call);
                    bound_batched_with(list, &batched, src, count, call.spec, call.parts)
                        .expect("a batch takes the execution")
                };
                for card in [false, true] {
                    for precise in [false, true] {
                        let expected = run(&calls, false, card, precise, &alone);
                        let got = run(&calls, false, card, precise, &together);
                        let how = format!(
                            "group clamp {grouped}, notes {notes}, card budget {card}, precise \
                             {precise}"
                        );
                        assert_eq!(got.0, expected.0, "{how}: the records clamped");
                        assert_eq!(got.1, expected.1, "{how}: the budget's words");
                        let mut once = vec![false; calls.len()];
                        once[0] = true;
                        assert_eq!(got.2, once, "{how}: one clamp for every execution");
                    }
                }
                // The budget run out, in the order of the executions: per
                // group, as GROUP_SOURCE charges.
                if grouped {
                    for card in [false, true] {
                        let expected = run(&spending, true, card, false, &alone);
                        let got = run(&spending, true, card, false, &together);
                        let how = format!("spent budget, notes {notes}, card budget {card}");
                        assert_eq!(got.0, expected.0, "{how}: the records clamped");
                        assert_eq!(got.1, expected.1, "{how}: the budget's words");
                        // (The card's copy of the budget hands back only
                        // its notes.)
                        if !card {
                            assert_eq!(got.1[0], 1 << 27, "{how}: the budget is spent");
                        }
                        assert_eq!(got.2, vec![true; spending.len()], "{how}: a batch each");
                        let draws: Vec<u32> = (0..12).map(|i| word(&got.0, 32 + i * 4)).collect();
                        assert_eq!(
                            draws,
                            [65_536, 1, 0, 0, 65_536, 1, 0, 0, 0, 0, 0, 0],
                            "{how}: the draw past the budget runs empty"
                        );
                        let last: Vec<u32> = (0..3).map(|i| word(&got.0, 80 + i * 4)).collect();
                        assert_eq!(last, [0, 0, 0], "{how}: the dispatch past the budget");
                    }
                }
                // The bounds themselves, for a few records: an index count
                // past 2^24, instances past 2^26 vertex instances, groups
                // past what a dispatch runs, a mesh record of all ones, the
                // parts of 100,000 groups.
                let got = run(&calls, false, false, false, &together).0;
                let word = |at: usize| word(&got, at);
                let start = |i: usize| -> usize {
                    calls[..i]
                        .iter()
                        .map(|c| c.read().next_multiple_of(16))
                        .sum()
                };
                assert_eq!([word(start(1)), word(start(1) + 4)], [1 << 24, 0]);
                assert_eq!(
                    [word(start(1) + 20), word(start(1) + 24)],
                    [100, (1 << 26) / 100]
                );
                let dispatch = start(2);
                assert_eq!(
                    [word(dispatch), word(dispatch + 4), word(dispatch + 8)],
                    [65_535, 1, 1]
                );
                let tail = (12..24).step_by(4).map(|off| word(dispatch + off));
                assert!(tail.eq([0, 0, 0]), "too many groups: empty");
                assert_eq!(
                    [word(start(3)), word(start(3) + 4), word(start(3) + 8)],
                    [65_535, 1, 1]
                );
                let parts = start(5);
                assert_eq!(
                    [word(parts), word(parts + 12)],
                    [65_535, 100_000 - 65_535],
                    "a dispatch in parts"
                );
            }
        }
    }

    /// A batch of clamps takes executions until something may have written
    /// their arguments: moving the argument buffer to be written (for a
    /// dispatch that stores to it) ends it, and the next execution opens a
    /// batch whose clamp, recorded after that dispatch, reads what it
    /// wrote; so does a note that what follows may write, and a move of the
    /// argument buffer between read states with no such note (to
    /// `COPY_SOURCE` and back). An execution counted in another buffer than
    /// the batch's opens one too, and one whose records are more than a
    /// batch holds takes none. The debug layer stays silent.
    #[test]
    fn host_list_batch_ends_where_the_arguments_may_be_written() {
        const WRITER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer args : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    args.Store(32, 9);
}
"#;
        let Some(host) = software_host() else {
            return;
        };
        let batched = make_from(&host.device, BATCH_SOURCE, true).expect("batch clamp");
        let writer = compile_cs(&host.device, WRITER_CS).expect("writer shader");
        let (Some((args, args_ptr)), Some((counts, counts_ptr))) =
            (host.shared_buffer(256), host.shared_buffer(256))
        else {
            return;
        };
        // SAFETY: mapped buffers of 256 bytes; nothing runs yet. Mesh
        // records at 0 and 32, a count at 64 of the count buffer.
        unsafe {
            std::ptr::write_bytes(args_ptr, 0, 256);
            std::ptr::write_bytes(counts_ptr, 0, 256);
            args_ptr.cast::<[u32; 3]>().write([5, 1, 1]);
            args_ptr.add(32).cast::<[u32; 3]>().write([7, 1, 1]);
            counts_ptr.add(64).cast::<u32>().write(1);
        }
        let list = host.new_list().expect("a host list");
        let _ = host.log_messages();
        let (out, out_ptr) = host.shared_buffer(256).expect("a read-back buffer");
        let from_to = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let spec = (KIND_MESH, MESH_STRIDE, 1);
        let mut copied = 0u64;
        let mut clamp = |offset: u64, count: Option<(&ID3D12Resource, u64)>| {
            let (bounded, at, opened) =
                bound_batched_with(&list, &batched, (&args, offset), count, spec, false)
                    .expect("a batch takes the execution");
            // SAFETY: an open list; the scratch buffer is in
            // INDIRECT_ARGUMENT after a batch's clamp, and goes back there;
            // buffers held until the list ran.
            unsafe {
                list.list
                    .ResourceBarrier(&[transition(&bounded, from_to, copy)]);
                list.list.CopyBufferRegion(&out, copied, &bounded, at, 12);
                list.list
                    .ResourceBarrier(&[transition(&bounded, copy, from_to)]);
            }
            copied += 16;
            opened
        };
        list.buffer_read(&args);
        assert!(clamp(0, None), "the first execution opens a batch");
        assert!(!clamp(32, None), "the next one joins");
        // A dispatch that stores to the argument buffer: moving the buffer
        // to be written already ends the batch (`HostList::may_write`).
        list.buffer_uav(&args);
        // SAFETY: an open list; the writer's root signature and pipeline,
        // and a buffer held until the list ran.
        unsafe {
            list.list.SetComputeRootSignature(&writer.root);
            list.list.SetPipelineState(&writer.pso);
            list.list
                .SetComputeRootUnorderedAccessView(0, args.GetGPUVirtualAddress());
            list.list.Dispatch(1, 1, 1);
        }
        list.buffer_read(&args);
        assert!(clamp(32, None), "after the write, a new batch");
        assert!(!clamp(0, None), "which the next one joins");
        list.may_write();
        assert!(clamp(0, None), "after what may write, a new batch");
        list.buffer_read(&counts);
        assert!(
            clamp(0, Some((&counts, 64))),
            "a count in a buffer the batch's clamp does not read"
        );
        assert!(
            !clamp(32, Some((&counts, 64))),
            "counted in the same buffer"
        );
        // A move between read states, which notes no write: only the
        // argument buffer's change count ends the batch.
        let writes = list.writes();
        list.buffer_state(&args, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list.buffer_read(&args);
        assert_eq!(list.writes(), writes, "no write noted");
        assert!(
            clamp(0, None),
            "after the buffer's state changed, a new batch"
        );
        assert!(!clamp(32, None), "which the next one joins");
        // More records than a batch holds: none takes them.
        let big = (KIND_MESH, MESH_STRIDE, (BATCH_BYTES / 12 + 1) as u32);
        assert!(bound_batched_with(&list, &batched, (&args, 0), None, big, false).is_none());
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer of 256 bytes.
        let got: Vec<u32> = (0..9)
            .map(|i| unsafe { out_ptr.add(i * 16).cast::<u32>().read_volatile() })
            .collect();
        assert_eq!(
            got,
            [5, 7, 9, 5, 5, 5, 9, 5, 9],
            "each execution's record as it was when it ran"
        );
    }

    /// A batch takes [`BATCH_CALLS`] executions with nothing that may write
    /// between them, and the next one opens a batch; the last slot gets its
    /// execution's record. An execution whose slot ends exactly at
    /// [`BATCH_BYTES`] joins, and the next one opens a batch. The debug
    /// layer stays silent.
    #[test]
    fn host_list_batch_fills_to_its_bounds() {
        let Some(host) = software_host() else {
            return;
        };
        let batched = make_from(&host.device, BATCH_SOURCE, false).expect("batch clamp");
        // The bytes of the execution that fills a batch after one of a
        // single record (its slot rounded to 16 bytes).
        let rest = BATCH_BYTES - 16;
        let records = (rest / 12) as u32;
        assert_eq!(u64::from(records) * 12, rest, "a whole number of records");
        let len = 4096 + rest;
        let (Some((args, args_ptr)), Some((out, out_ptr))) =
            (host.shared_buffer(len), host.shared_buffer(256))
        else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet. Mesh
        // records 16 bytes apart, the i-th asking for i + 1 groups; the
        // filling execution's last record asks for 7.
        unsafe {
            std::ptr::write_bytes(args_ptr, 0, len as usize);
            for i in 0..=BATCH_CALLS {
                let at = args_ptr.add(i as usize * 16).cast::<[u32; 3]>();
                at.write([i + 1, 1, 1]);
            }
            let last = 4096 + rest - 12;
            args_ptr
                .add(last as usize)
                .cast::<[u32; 3]>()
                .write([7, 1, 1]);
        }
        let list = host.new_list().expect("a host list");
        let _ = host.log_messages();
        list.buffer_read(&args);
        let from_to = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let mut copied = 0u64;
        let mut clamp = |offset: u64, spec: (u32, u32, u32), read: Option<u64>| {
            let (bounded, at, opened) =
                bound_batched_with(&list, &batched, (&args, offset), None, spec, false)
                    .expect("a batch takes the execution");
            if let Some(read) = read {
                // SAFETY: an open list; the scratch buffer is in
                // INDIRECT_ARGUMENT after a batch's clamp, and goes back
                // there; buffers held until the list ran.
                unsafe {
                    list.list
                        .ResourceBarrier(&[transition(&bounded, from_to, copy)]);
                    list.list
                        .CopyBufferRegion(&out, copied, &bounded, at + read, 12);
                    list.list
                        .ResourceBarrier(&[transition(&bounded, copy, from_to)]);
                }
                copied += 16;
            }
            (opened, at)
        };
        let one = (KIND_MESH, MESH_STRIDE, 1);
        let mut opened = Vec::new();
        for i in 0..=BATCH_CALLS {
            let read = (i >= BATCH_CALLS - 1).then_some(0);
            let (new, at) = clamp(u64::from(i) * 16, one, read);
            if i == BATCH_CALLS - 1 {
                assert_eq!(at, u64::from(BATCH_CALLS - 1) * 16, "the last slot");
            }
            opened.push(new);
        }
        let mut once = vec![false; opened.len()];
        once[0] = true;
        once[BATCH_CALLS as usize] = true;
        assert_eq!(
            opened, once,
            "a batch opens at the first execution and the 129th"
        );
        // A slot that ends exactly at BATCH_BYTES.
        list.may_write();
        assert!(clamp(0, one, None).0, "after what may write, a new batch");
        let fill = (KIND_MESH, MESH_STRIDE, records);
        let (joined, at) = clamp(4096, fill, Some(rest - 12));
        assert!(!joined, "a slot ending at BATCH_BYTES joins");
        assert_eq!(at + rest, BATCH_BYTES, "the slot ends at BATCH_BYTES");
        assert!(clamp(0, one, None).0, "a full batch takes no more");
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer of 256 bytes.
        let got: Vec<[u32; 3]> = (0..3)
            .map(|i| unsafe { out_ptr.add(i * 16).cast::<[u32; 3]>().read_volatile() })
            .collect();
        assert_eq!(
            got,
            [[BATCH_CALLS, 1, 1], [BATCH_CALLS + 1, 1, 1], [7, 1, 1]],
            "the last slot of a full batch, the next batch's first, the last record of a full slot"
        );
    }

    /// Runs `f` with [`precise`] `on` on this thread.
    fn with_precise<R>(on: bool, f: impl FnOnce() -> R) -> R {
        PRECISE_HERE.set(Some(on));
        let out = f();
        PRECISE_HERE.set(None);
        out
    }

    /// With [`precise`], a clamp waits for every earlier shader write only
    /// after what may have written memory, for the budget alone after
    /// clamps, and for nothing after a wait; never after its dispatch.
    #[test]
    fn precise_waits_follow_what_the_list_recorded() {
        use crate::host::SinceWait;
        let after = |since| Waits::after_list(since, None);
        let before = |before| Waits {
            before,
            after: false,
        };
        assert_eq!(after(SinceWait::Nothing), before(Wait::Nothing));
        assert_eq!(after(SinceWait::Anything), before(Wait::All));
        assert_eq!(
            after(SinceWait::Clamps),
            before(Wait::All),
            "with no budget buffer known, every earlier write"
        );
        assert!(SinceWait::Nothing < SinceWait::Clamps && SinceWait::Clamps < SinceWait::Anything);
        assert_eq!(
            Waits::both(),
            Waits {
                before: Wait::All,
                after: true
            }
        );
    }

    /// A host list notes what may have written memory since its last wait
    /// for every earlier shader write, and its clamps wait for that with
    /// `VICTORY_CLAMP_PRECISE=1` (and for everything twice without it): a
    /// list starts as written and is so again after a reset; a clamp leaves
    /// its budget's writes; a wait leaves nothing; a buffer moved to be
    /// written, or a command noted as one that may write, leaves anything.
    #[test]
    fn host_list_notes_what_may_have_written_since_its_last_wait() {
        use crate::host::SinceWait;
        let Some(host) = software_host() else {
            return;
        };
        let (Some((src, src_ptr)), Some(other)) = (host.shared_buffer(256), host.uav_buffer(256))
        else {
            return;
        };
        // SAFETY: the mapping holds 256 bytes.
        unsafe { std::ptr::write_bytes(src_ptr, 0, 256) };
        let spec = (10, MESH_STRIDE, 1);
        for card in [false, true] {
            let list = host.new_list().expect("a host list");
            if card {
                assert!(list.force_card_budget(), "no budget in the card's memory");
            }
            let _ = host.log_messages();
            let clamp = |list: &HostList| {
                let dst = bound_as(list, (&src, 0), None, spec, (0, false)).expect("a clamp");
                done(list, &dst);
            };
            assert_eq!(
                list.since_wait(),
                SinceWait::Anything,
                "a list starts as written"
            );
            with_precise(true, || {
                let all = Waits {
                    before: Wait::All,
                    after: false,
                };
                assert_eq!(Waits::for_list(&list), all);
                clamp(&list);
                assert_eq!(
                    list.since_wait(),
                    SinceWait::Clamps,
                    "the clamp's own writes"
                );
                let budget = list.budget_buffer().expect("a budget after a clamp");
                assert_eq!(Waits::for_list(&list).before, Wait::On(budget));
                clamp(&list);
                assert_eq!(list.since_wait(), SinceWait::Clamps);
                // SAFETY: an open list.
                unsafe { list.list.ResourceBarrier(&[uav_barrier()]) };
                list.waited_for_writes();
                assert_eq!(Waits::for_list(&list).before, Wait::Nothing);
                list.buffer_read(&src);
                assert_eq!(
                    list.since_wait(),
                    SinceWait::Nothing,
                    "a read writes nothing"
                );
                clamp(&list);
                assert_eq!(list.since_wait(), SinceWait::Clamps);
                list.buffer_uav(&other);
                assert_eq!(
                    list.since_wait(),
                    SinceWait::Anything,
                    "a buffer to be written"
                );
                assert_eq!(Waits::for_list(&list), all);
                clamp(&list);
                list.may_write();
                assert_eq!(Waits::for_list(&list), all);
            });
            with_precise(false, || {
                assert_eq!(Waits::for_list(&list), Waits::both());
                clamp(&list);
                assert_eq!(
                    list.since_wait(),
                    SinceWait::Nothing,
                    "the wait after the clamp"
                );
            });
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            assert_eq!(
                list.since_wait(),
                SinceWait::Anything,
                "a reset list starts as written"
            );
        }
    }

    /// With `VICTORY_CLAMP_PRECISE=1`, clamps of their own and batches of
    /// mesh draws give every record what they give it with both waits for
    /// every earlier shader write, and charge the budget the same: with the
    /// budget in mapped words and in the card's memory, the scratch buffer
    /// kept readable and not. The first records are clamped alone, the next
    /// in batches, the last alone again; writes noted on the way bring the
    /// wait for every earlier write back, and waits recorded on the way
    /// leave none. The debug layer stays silent.
    #[test]
    fn host_list_precise_clamps_match_the_waited_ones() {
        const STEP: u32 = 32;
        const COPY: u64 = 9;
        let Some(host) = software_host() else {
            return;
        };
        let batch = make_from(&host.device, MESH_BATCH_SOURCE, false).expect("batch clamp");
        let records = mesh_records();
        let n = records.len();
        let Some((src, src_ptr)) = host.shared_buffer(n as u64 * u64::from(STEP)) else {
            return;
        };
        // SAFETY: a mapped buffer of this size; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0xff, n * STEP as usize);
            for (i, record) in records.iter().enumerate() {
                src_ptr
                    .add(i * STEP as usize)
                    .cast::<[u32; 3]>()
                    .write_unaligned(*record);
            }
        }
        let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
        let run = |precise: bool, card: bool, keep: bool| {
            with_precise(precise, || {
                let list = host.new_list().expect("a host list");
                if card {
                    assert!(list.force_card_budget(), "no budget in the card's memory");
                }
                let _ = host.log_messages();
                let (out, out_ptr) = host
                    .shared_buffer(n as u64 * 12)
                    .expect("a read-back buffer");
                for i in 0..n as u32 {
                    if i % 9 == 4 {
                        list.end_mesh_batch();
                        list.may_write();
                    }
                    if i % 11 == 7 {
                        list.end_mesh_batch();
                        // SAFETY: an open list.
                        unsafe { list.list.ResourceBarrier(&[uav_barrier()]) };
                        list.waited_for_writes();
                    }
                    let offset = i * STEP;
                    let alone = !(10..30).contains(&i);
                    let (bounded, at) = if alone {
                        let spec = (10, MESH_STRIDE, 1);
                        let at = (&src, u64::from(offset));
                        (
                            bound_as(&list, at, None, spec, (0, keep)).expect("a clamp"),
                            0,
                        )
                    } else {
                        let (bounded, at, _) = grouped_mesh_as(&list, COPY, offset, |list| {
                            MeshBatch::open_with(list, &batch, COPY, &src, keep)
                        })
                        .expect("a batch");
                        (bounded, at)
                    };
                    // SAFETY: an open list; the scratch buffer is in
                    // INDIRECT_ARGUMENT after a clamp (and through a
                    // batch), and goes back there; buffers held until the
                    // list ran.
                    unsafe {
                        list.list
                            .ResourceBarrier(&[transition(&bounded, args, copy)]);
                        list.list
                            .CopyBufferRegion(&out, u64::from(i) * 12, &bounded, at, 12);
                        list.list
                            .ResourceBarrier(&[transition(&bounded, copy, args)]);
                    }
                    if alone {
                        done(&list, &bounded);
                    }
                    list.keep(&bounded);
                }
                list.close();
                let how = format!("precise {precise}, card budget {card}, readable {keep}");
                assert_eq!(
                    host.log_messages(),
                    None,
                    "{how}: the debug layer found an error"
                );
                host.execute(&[&list]);
                list.reset();
                // SAFETY: the list has run; a mapped buffer of `n` records.
                let clamped = unsafe {
                    std::slice::from_raw_parts(out_ptr.cast::<[u32; 3]>().cast_const(), n).to_vec()
                };
                (clamped, list.budget_words().expect("budget words"))
            })
        };
        let (expected, _) = clamped_alone(&records);
        for card in [false, true] {
            for keep in [false, true] {
                let waited = run(false, card, keep);
                let precise = run(true, card, keep);
                let how = format!("card budget {card}, readable {keep}");
                assert_eq!(waited.0, expected, "{how}: with both waits");
                assert_eq!(precise.0, expected, "{how}: precise");
                assert_eq!(precise.1, waited.1, "{how}: the budget's words");
            }
        }
    }

    /// The bytes of the dispatch records `groups`: three words each, `step`
    /// bytes apart, the rest of each record ones.
    fn dispatch_bytes(groups: &[u32], step: u32) -> Vec<u8> {
        let mut bytes = vec![0xffu8; groups.len() * step as usize];
        for (i, &g) in groups.iter().enumerate() {
            for (w, word) in [g, 1, 1].into_iter().enumerate() {
                let at = i * step as usize + w * 4;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    /// Records the dispatch records `groups` ([`dispatch_bytes`]) into a
    /// fresh mapped buffer of `warp`'s device.
    fn dispatch_records(warp: &WarpHarness, groups: &[u32], step: u32) -> ID3D12Resource {
        let bytes = dispatch_bytes(groups, step);
        let (buffer, ptr, _) = warp.shared_buffer(bytes.len() as u64);
        // SAFETY: the mapping holds the records' bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
        buffer
    }

    /// Two windows in one list, as `crate::draw` records them with
    /// `VICTORY_INDIRECT_READABLE=1`: each clamps its records ahead, then
    /// one execution per record reads the scratch buffer with no barrier
    /// between them; the second clamp moves the scratch buffer straight
    /// from arguments back to unordered access. Every record runs once,
    /// with the group count it asked for.
    #[test]
    fn warp_window_executions_read_their_records_with_no_barrier_between() {
        window_executions(SOURCE);
    }

    fn window_executions(source: &str) {
        let Some(mut warp) = WarpHarness::with(source) else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        // The title names 12 bytes and keeps its records 32 apart.
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let sources: Vec<ID3D12Resource> = windows
            .iter()
            .map(|g| dispatch_records(&warp, g, STEP))
            .collect();
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(u64::from(STEP) * 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let mut state = D3D12_RESOURCE_STATE_COMMON;
        for (groups, src) in windows.iter().zip(&sources) {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, state),
                budget_gpu,
                (src, 0),
                None,
                (KIND_DISPATCH, STEP, groups.len() as u32),
                AHEAD,
            ));
            state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: an open list; the worker's root signature and
            // pipeline; the scratch buffer is readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                for i in 0..groups.len() as u64 {
                    let at = i * u64::from(STEP);
                    warp.list.ExecuteIndirect(&sig, 1, &dst_buf, at, None, 0);
                }
            }
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (hits, spent) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
            )
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        assert_eq!(hits, expected, "every record of both windows runs once");
        assert_eq!(spent, expected * 64, "and is charged what it asked for");
    }

    /// The windows of `warp_window_executions_read_their_records_with_no_barrier_between`
    /// through a host list (`bound_ahead` with the scratch buffer kept
    /// readable, the list's tracked scratch state, `Host::execute`), twice:
    /// on the first recording the scratch buffer starts common, on the
    /// second (after the reset) it starts common again. With the debug
    /// layer on (`VICTORY_D3D12_DEBUG=1` in a diagnostics build) no barrier
    /// names a state the buffer is not in. On a software adapter only
    /// ([`software_host`]): the switch is forced on here.
    #[test]
    fn host_list_window_executions_need_no_barrier_of_their_own() {
        let Some(host) = software_host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let signature = dispatch_signature(&host.device, STRIDE);
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let mut sources = Vec::new();
        for groups in windows {
            let bytes = dispatch_bytes(groups, STEP);
            let Some((buffer, ptr)) = host.shared_buffer(bytes.len() as u64) else {
                return;
            };
            // SAFETY: the mapping holds the records' bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len()) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            for (groups, src) in windows.iter().zip(&sources) {
                let spec = (KIND_DISPATCH, STEP, groups.len() as u32);
                let scratch = bound_as(&list, (src, 0), None, spec, (AHEAD, true)).unwrap();
                assert_eq!(
                    list.indirect_state(&scratch),
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
                );
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                }
                for i in 0..groups.len() as u64 {
                    let at = i * u64::from(STEP);
                    // SAFETY: as above; the scratch buffer is readable as
                    // arguments.
                    unsafe {
                        list.list
                            .ExecuteIndirect(&signature, 1, &scratch, at, None, 0)
                    };
                    // What `State::execute_indirect` records after an
                    // execution: nothing, for a buffer kept readable.
                    done(&list, &scratch);
                }
                assert_eq!(
                    list.indirect_state(&scratch),
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
                );
            }
            list.close();
            // With the debug layer on: nothing it objects to was recorded.
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every record of both windows runs once");
        }
    }
}

/// The benchmark of the clamp, its barriers and its budget (ignored by
/// default).
#[cfg(test)]
mod bench;

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
