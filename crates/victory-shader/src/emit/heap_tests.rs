//! End-to-end tests of view indices past the host's view heap (module
//! notes, "Heap indices"): synthetic GFX10 compute programs whose
//! descriptors come from a table or from the view heap itself at an index
//! the program computes from data, lowered to DXIL, validated by `dxil.dll`
//! when present, and run on WARP only. Each access past the heap reads the
//! null view of its own kind and texture dimension, and a buffer picked
//! from the heap at a slot holding no buffer view reads the null buffer
//! view: zeros, stores and atomics dropped, the device alive. With views
//! of each kind in the null views' places, each access finds its own.

use windows::Win32::Graphics::Direct3D12::*;

use super::GuardView as G;
use super::host::{Dispatch, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 128.. inline, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// `v_mov_b32 vdst, src0`.
const fn v_mov(vdst: u8, src0: u16) -> u32 {
    vop1(0x001, vdst, src0)
}

/// SMEM `op s[dst..], s[base..], soffset offset:byte_off` (`soffset`: an
/// SGPR, or 125 for none).
const fn smem(op: u8, dst: u8, base: u8, soffset: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | (op as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (soffset as u32) << 25 | (byte_off & 0x1F_FFFF),
    ]
}

/// MUBUF `op vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn mubuf(op: u8, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32 & 0x7F) << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// MUBUF `op vdata, vindex, s[srsrc:srsrc+3], 0 idxen`.
const fn mubuf_indexed(op: u8, vdata: u8, vindex: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32 & 0x7F) << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vindex as u32,
    ]
}

/// `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    mubuf(0x1C, vdata, srsrc, byte_off)
}

/// MIMG `op vdata, v[vaddr..], s[rsrc:rsrc+7], s[ssamp:ssamp+3] dmask:0x1
/// dim:dim`, unnormalized unless a sample (`op` 0x20 on).
const fn mimg(op: u8, dim: u8, vdata: u8, vaddr: u8, rsrc: u8, ssamp: u8) -> [u32; 2] {
    let unorm = (op < 0x20) as u32;
    [
        0x3C << 26 | (op as u32 & 0x7F) << 18 | unorm << 12 | 1 << 8 | (dim as u32 & 7) << 3,
        (ssamp as u32 >> 2 & 0x1F) << 21
            | (rsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

/// A used-resource entry of the table at `s4`, at dword `table_offset_dwords`.
fn used(kind: ResourceType, table_offset_dwords: u32, stride: u32, dwords: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 4,
        slot: 1,
    }
}

/// Lowers a one-thread wave32 compute program with 4 root constants in
/// `s0..s3`, the address of a table with the entries `used` in `s4` and,
/// with `heap`, the view heap's address in `s6` (root type `ResourceHeap`:
/// the program picks descriptors from the heap itself).
fn lower_cs(words: &[u32], used: Vec<UsedResource>, heap: bool) -> HostShader {
    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let mut layout = vec![
        param(RootType::Constants, 4, 0, 0),
        param(RootType::Table, 1, 4, 1),
    ];
    if heap {
        layout.push(param(RootType::ResourceHeap, 1, 6, 2));
    }
    let user_sgprs = if heap { 7 } else { 5 };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: user_sgprs << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// A program's compute pipeline on WARP.
struct Warp {
    host: Host,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

impl Warp {
    /// None without a WARP device.
    fn new(shader: &HostShader) -> Option<Self> {
        let Ok(host) = Host::new(true) else {
            eprintln!("no SM 6.6 WARP device; skipped");
            return None;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("compute_pipeline failed: {e}"));
        Some(Warp { host, root, pso })
    }

    /// Runs `shader` with root constants `k`, the table at heap slot 0,
    /// `views`, `samplers` and the root SRVs `roots` after the user data,
    /// and returns the UAVs' contents in the order of the views.
    fn run(
        &self,
        shader: &HostShader,
        k: [u32; 4],
        views: &[View],
        samplers: &[D3D12_SAMPLER_DESC],
        roots: &[Vec<u32>],
    ) -> Vec<Vec<u32>> {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Constants, 0) => k[u.dword as usize],
                (RootType::Table, 1) => 0,
                // Words the host computes (view addresses): 0.
                (RootType::Other(_), _) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let mut bound = vec![(0, Root::Constants(ud))];
        for (n, data) in roots.iter().enumerate() {
            bound.push((
                1 + n as u32,
                Root::View {
                    srv: true,
                    data: data.clone(),
                },
            ));
        }
        self.host
            .dispatch(
                &self.root,
                &self.pso,
                &Dispatch {
                    roots: bound,
                    views,
                    samplers,
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{k:x?}: dispatch failed: {e}"))
    }
}

/// The heap's last slot, which the views of [`View::Guards`] end at.
fn last() -> u32 {
    super::view_heap_limit()
}

/// Buffer reads through table entries past the heap: a constant buffer with
/// scalar and vector loads, a raw SRV with scalar and vector loads, a raw
/// UAV with a vector load and a store. The heap's last slot is a null 2D
/// texture view, and on AMD a null texture view read as a buffer gives a
/// wild address (the device is removed). Each clamped index reads the null
/// view of its handle's kind instead: every read gives zeros and the store
/// is dropped; with real views of each kind in the null views' places,
/// each access finds its own kind's.
#[test]
fn a_heap_index_past_the_heap_reads_the_null_view_of_its_kind() {
    let words = [
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        // s_load_dwordx4: the output UAV at slot 0; at slots s0 + 1, s0 + 2
        // and s0 the raw SRV, the constant buffer and a UAV.
        smem(0x02, 8, 4, 125, 0)[0],
        smem(0x02, 8, 4, 125, 0)[1],
        smem(0x02, 12, 4, 6, 32)[0],
        smem(0x02, 12, 4, 6, 32)[1],
        smem(0x02, 16, 4, 6, 64)[0],
        smem(0x02, 16, 4, 6, 64)[1],
        smem(0x02, 20, 4, 6, 0)[0],
        smem(0x02, 20, 4, 6, 0)[1],
        // s_buffer_load_dword: the constant buffer at 0x20, the SRV at 4.
        smem(0x08, 24, 16, 125, 0x20)[0],
        smem(0x08, 24, 16, 125, 0x20)[1],
        smem(0x08, 25, 12, 125, 4)[0],
        smem(0x08, 25, 12, 125, 4)[1],
        // buffer_load_dword: the SRV at 8, the UAV at 0, the constant
        // buffer at 0x24.
        mubuf(0x0C, 1, 12, 8)[0],
        mubuf(0x0C, 1, 12, 8)[1],
        mubuf(0x0C, 2, 20, 0)[0],
        mubuf(0x0C, 2, 20, 0)[1],
        mubuf(0x0C, 6, 16, 0x24)[0],
        mubuf(0x0C, 6, 16, 0x24)[1],
        v_mov(3, 24), // v_mov_b32 v3, s24
        v_mov(4, 25), // v_mov_b32 v4, s25
        buffer_store_dword(3, 8, 0)[0],
        buffer_store_dword(3, 8, 0)[1],
        buffer_store_dword(4, 8, 4)[0],
        buffer_store_dword(4, 8, 4)[1],
        buffer_store_dword(1, 8, 8)[0],
        buffer_store_dword(1, 8, 8)[1],
        buffer_store_dword(2, 8, 12)[0],
        buffer_store_dword(2, 8, 12)[1],
        buffer_store_dword(6, 8, 20)[0],
        buffer_store_dword(6, 8, 20)[1],
        v_mov(5, 0x87), // v_mov_b32 v5, 7
        // Through the UAV at slot s0, to its dword 4.
        buffer_store_dword(5, 20, 16)[0],
        buffer_store_dword(5, 20, 16)[1],
        S_ENDPGM,
    ];
    let shader = lower_cs(
        &words,
        vec![
            used(ResourceType::Uav, 0, 0, 4),
            used(ResourceType::Srv, 8, 0, 4),
            used(ResourceType::Cbv, 16, 0, 4),
        ],
        false,
    );
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(warp) = Warp::new(&shader) else {
        return;
    };
    let run = |k: u32, views: &[View]| warp.run(&shader, [k, 0, 0, 0], views, &[], &[]);
    let dwords = |base: u32| (0..16).map(|k| base + k).collect::<Vec<u32>>();
    let (srv, cbv) = (dwords(0xB000), dwords(0xC000));

    // The host's null views at the heap's end: zeros, the store dropped.
    let views = [
        View::RawUav(8),
        View::Buffer {
            data: &srv,
            stride: Stride::Raw,
        },
        View::Cbv(&cbv),
        View::Guards(&[]),
    ];
    let inside = run(0, &views);
    let want = [cbv[8], srv[1], srv[2], 0, 7, cbv[9], 0, 0];
    assert_eq!(inside[0], want, "in range");
    for k in [last(), 1 << 25, 0x07FF_FF00] {
        let out = run(k, &views);
        assert_eq!(out[0], [0; 8], "index {k:#x}");
    }

    // Views of each kind in the null views' places: each access reads the
    // one of its kind.
    let (srv2, cbv2) = (dwords(0xE000), dwords(0xD000));
    let marked = [
        View::Texture(1, 1, &[[0.5; 4]]),
        View::Cbv(&cbv2),
        View::Buffer {
            data: &srv2,
            stride: Stride::Raw,
        },
        View::RawUav(8),
    ];
    let views = [View::RawUav(8), View::Guards(&marked)];
    let out = run(1 << 25, &views);
    let want = [cbv2[8], srv2[1], srv2[2], 0, 0, cbv2[9], 0, 0];
    assert_eq!(out[0], want, "the marked views' contents");
    let want = [0, 0, 0, 0, 7, 0, 0, 0];
    assert_eq!(out[1], want, "the store in the null UAV's place");
}

/// More accesses through table entries past the heap, each expecting what a
/// null view of its kind gives: a structured buffer load (idxen), a typed
/// buffer load (`buffer_load_format_x`), a texture atomic through a UAV
/// (dropped), and raw loads through descriptors chosen by SCC
/// (`s_cselect`, [`crate::resource::Source::Select`]) and by control flow
/// (a join, [`crate::resource::Source::Phi`]) between the clamped one and
/// one in range.
#[test]
fn heap_indices_past_the_heap_read_zeros_for_every_access_kind() {
    let mut words = vec![
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        v_mov(7, 0x80),         // v7 = 0: element index
        v_mov(2, 0x80),         // v2 = 0: texel x
        v_mov(3, 0x80),         // v3 = 0: texel y
        v_mov(5, 0x87),         // v5 = 7
    ];
    // s_load_dwordx4/x8: the output UAV (entry 0); at entries s0 + 1 to
    // s0 + 3 the structured SRV, the typed SRV and the texture UAV; entry
    // 1 again, in range.
    for (op, dst, soffset, entry) in [
        (0x02, 8, 125, 0),
        (0x02, 12, 6, 1),
        (0x02, 16, 6, 2),
        (0x03, 20, 6, 3),
        (0x02, 36, 125, 1),
    ] {
        words.extend(smem(op, dst, 4, soffset, 32 * entry));
    }
    words.extend([
        0xBF06_8001,            // s_cmp_eq_u32 s1, 0
        sop2(0x0B, 40, 12, 36), // s_cselect_b64 s[40:41], s[12:13], s[36:37]
        sop2(0x0B, 42, 14, 38), // s_cselect_b64 s[42:43], s[14:15], s[38:39]
        0xBF06_8002,            // s_cmp_eq_u32 s2, 0
        0xBF85_0003,            // s_cbranch_scc1 -> the clamped one
    ]);
    // The raw SRV at entry 4 (in range) or s0 + 4, into s[28:31].
    words.extend(smem(0x02, 28, 4, 125, 128));
    words.push(0xBF82_0002); // s_branch -> the join
    words.extend(smem(0x02, 28, 4, 6, 128));
    words.extend(smem(0x08, 33, 28, 125, 0)); // s_buffer_load_dword s33, s[28:31]
    words.extend(mubuf_indexed(0x0C, 9, 7, 12)); // buffer_load_dword v9: structured
    words.extend(mubuf_indexed(0x00, 8, 7, 16)); // buffer_load_format_x v8: typed
    words.extend(mubuf_indexed(0x0C, 10, 7, 40)); // buffer_load_dword v10: selected
    words.push(v_mov(11, 33)); // v_mov_b32 v11, s33: joined
    // image_atomic_add v5, v[2:3], s[20:27] dmask:0x1 unorm dim:2D
    words.extend([0xF044_1108, 0x0005_0502]);
    for (k, v) in [9, 8, 10, 11].into_iter().enumerate() {
        words.extend(buffer_store_dword(v, 8, 4 * k as u16));
    }
    words.push(S_ENDPGM);
    let shader = lower_cs(
        &words,
        vec![
            used(ResourceType::Uav, 0, 0, 4),
            used(ResourceType::Srv, 8, 4, 4),
            used(ResourceType::Srv, 16, 0, 4),
            used(ResourceType::Uav, 24, 0, 8),
            used(ResourceType::Srv, 32, 0, 4),
        ],
        false,
    );
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(warp) = Warp::new(&shader) else {
        return;
    };
    let run =
        |k: [u32; 3], views: &[View]| warp.run(&shader, [k[0], k[1], k[2], 0], views, &[], &[]);
    let structured = [0x5100, 0x5101];
    let typed = [1.5f32.to_bits(), 0, 0, 0];
    let raw = [0x7700, 0x7701];
    let views = [
        View::RawUav(4),
        View::Buffer {
            data: &structured,
            stride: Stride::Structured(4),
        },
        View::Buffer {
            data: &typed,
            stride: Stride::Typed,
        },
        View::UintTextureUav(1, 1),
        View::Buffer {
            data: &raw,
            stride: Stride::Raw,
        },
        View::Guards(&[]),
    ];
    // (s0, s1, s2): the index, and whether the choices take the clamped
    // descriptor (0) or the one in range; the output, the texel.
    let past = 1 << 25;
    let cases = [
        ([0, 0, 0], [0x5100, typed[0], 0x5100, 0x7700], 7),
        ([past, 0, 0], [0, 0, 0, 0], 0),
        ([past, 1, 1], [0, 0, 0x5100, 0x7700], 0),
        ([last() - 1, 0, 0], [0, 0, 0, 0], 0),
    ];
    for (k, want, texel) in cases {
        let out = run(k, &views);
        assert_eq!(out[0], want, "{k:x?}");
        assert_eq!(out[1], [texel], "{k:x?}: the texel");
    }

    // WARP reads these kinds through a null texture view as zeros too, so
    // zeros do not show which null view was read. Views of each kind in
    // the null views' places do: the buffer loads read the buffer one, the
    // atomic lands in the texture UAV one.
    let marked = [0xA100, 0xA101];
    let guards = [
        View::Texture(1, 1, &[[0.5; 4]]),
        View::Cbv(&[0; 4]),
        View::Buffer {
            data: &marked,
            stride: Stride::Structured(4),
        },
        View::RawUav(4),
        View::UintTextureUav(1, 1),
    ];
    let out = run([past, 0, 0], &[View::RawUav(4), View::Guards(&guards)]);
    // The typed load (a typed handle on the structured view) is not checked.
    let got = [out[0][0], out[0][2], out[0][3]];
    assert_eq!(got, [0xA100; 3], "structured, selected and joined loads");
    assert_eq!(out[2], [7], "the atomic in the texture UAV's place");
}

/// Texture accesses of other dimensions through table entries past the
/// heap: a 3D and a 2D array load, a cube sample, and atomics through a 3D
/// and a 2D array UAV. Each clamped index reads the null view of its own
/// dimension, not the null 2D view in the heap's last slot (a view of
/// another dimension is undefined in D3D12): the loads and the sample give
/// zeros and the atomics are dropped. With views of each dimension in the
/// null views' places, each access finds its own. The 1D and 1D array null
/// views are routed the same way ([`super::GuardView::of`] tests their
/// kinds) but not run here.
#[test]
fn heap_indices_past_the_heap_read_the_null_view_of_their_dimension() {
    let mut words = vec![
        s_mov_b32(5, 0x80),     // s5 = 0
        sop2(0x1E, 6, 0, 0x85), // s_lshl_b32 s6, s0, 5: the index from data
        v_mov(0, 0x80),         // v0 = 0
        v_mov(1, 0x80),         // v1 = 0
        v_mov(2, 0x80),         // v2 = 0
        v_mov(3, 0x81),         // v3 = 1: the array layer
        // v[4:6]: the centre of the cube's +x face (s, t in [1, 2], face 0).
        v_mov(4, 0xFF),
        1.5f32.to_bits(),
        v_mov(5, 0xFF),
        1.5f32.to_bits(),
        v_mov(6, 0x80),
    ];
    // s_load_dwordx4/x8: the output UAV (entry 0); at entries s0 + 1 to
    // s0 + 5 the 3D SRV, the 2D array SRV, the cube SRV, the 3D UAV and
    // the 2D array UAV; the sampler at sampler heap index 0.
    for (op, dst, soffset, byte) in [
        (0x02, 8, 125, 0),
        (0x03, 12, 6, 32),
        (0x03, 20, 6, 64),
        (0x03, 28, 6, 96),
        (0x03, 36, 6, 128),
        (0x03, 44, 6, 160),
        (0x02, 52, 125, 0),
    ] {
        words.extend(smem(op, dst, 4, soffset, byte));
    }
    words.extend(mimg(0x27, 3, 7, 4, 28, 52)); // image_sample_lz v7, cube
    words.extend(mimg(0x00, 2, 4, 0, 12, 0)); // image_load v4, 3D at (0, 0, 0)
    words.extend(mimg(0x00, 5, 5, 1, 20, 0)); // image_load v5, 2D array layer 1
    words.push(v_mov(6, 0x87)); // v6 = 7
    words.extend(mimg(0x11, 2, 6, 0, 36, 0)); // image_atomic_add v6, 3D UAV
    words.extend(mimg(0x11, 5, 6, 0, 44, 0)); // image_atomic_add v6, 2D array UAV
    for (k, v) in [4, 5, 7].into_iter().enumerate() {
        words.extend(buffer_store_dword(v, 8, 4 * k as u16));
    }
    words.push(S_ENDPGM);
    let shader = lower_cs(
        &words,
        vec![
            used(ResourceType::Uav, 0, 0, 8),
            used(ResourceType::Srv, 8, 0, 8),
            used(ResourceType::Uav, 32, 0, 8),
        ],
        false,
    );
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(warp) = Warp::new(&shader) else {
        return;
    };
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let run = |k: u32, views: &[View]| warp.run(&shader, [k, 0, 0, 0], views, &samplers, &[]);
    // A texel of red `r`.
    let red = |r: f32| [r, 0.0, 0.0, 1.0];
    // Faces +x -x +y -y +z -z red `base` + face.
    let faces = |base: f32| (0..6).map(|f| red(base + f as f32)).collect::<Vec<_>>();
    let floats = |w: &[u32]| w.iter().map(|&x| f32::from_bits(x)).collect::<Vec<f32>>();

    // The host's null views: zeros, the atomics dropped.
    let (volume, layers, cube) = ([red(0.25), red(0.75)], [red(1.0), red(2.0)], faces(30.0));
    let views = [
        View::RawUav(4),
        View::Volume(1, 1, 2, &volume),
        View::Array(1, 1, 2, &layers),
        View::Cube(1, &cube),
        View::UintVolumeUav(1, 1),
        View::UintArrayUav(1, 1),
        View::Guards(&[]),
    ];
    let inside = run(0, &views);
    assert_eq!(floats(&inside[0]), [0.25, 2.0, 30.0, 0.0], "in range");
    assert_eq!(inside[1..], [[7], [7]], "in range: the atomics");
    for k in [last() - 1, last(), 1 << 25, 0x07FF_FF00] {
        let out = run(k, &views);
        assert_eq!(out, [vec![0; 4], vec![0], vec![0]], "index {k:#x}");
    }

    // WARP reads a null view of another dimension as zeros too, so zeros
    // do not show which null view was read. Views of each dimension in the
    // null views' places do; places no access here reads hold a 2D texture.
    let (volume, layers, cube) = ([red(4.0), red(4.5)], [red(5.0), red(6.0)], faces(40.0));
    let filler = [red(0.5)];
    let marked = [0xE000, 0xE001];
    let guards = [
        View::Texture(1, 1, &filler),
        View::Cbv(&[0; 4]),
        View::Buffer {
            data: &marked,
            stride: Stride::Raw,
        },
        View::RawUav(4),
        View::UintTextureUav(1, 1),
        View::Texture(1, 1, &filler),
        View::Texture(1, 1, &filler),
        View::Array(1, 1, 2, &layers),
        View::Volume(1, 1, 2, &volume),
        View::Cube(1, &cube),
        View::Texture(1, 1, &filler),
        View::Texture(1, 1, &filler),
        View::UintArrayUav(1, 1),
        View::UintVolumeUav(1, 1),
    ];
    for (view, at) in [
        (G::Texture2DArray, 7),
        (G::Texture3D, 8),
        (G::TextureCube, 9),
        (G::UnorderedTexture2DArray, 12),
        (G::UnorderedTexture3D, 13),
    ] {
        assert_eq!(G::ALL[at], view);
    }
    let out = run(1 << 25, &[View::RawUav(4), View::Guards(&guards)]);
    assert_eq!(
        floats(&out[0]),
        [4.0, 6.0, 40.0, 0.0],
        "the marked views' texels"
    );
    // UAVs in order: the buffer and 2D ones untouched, the atomics in the
    // 2D array and 3D ones.
    assert_eq!(
        out[1..],
        [vec![0; 4], vec![0], vec![7], vec![7]],
        "the atomics"
    );
}

/// Buffers the program picks from the view heap itself (`ResourceHeap`) at
/// an index it computes: a raw load and a raw store. At a slot that holds
/// a buffer view they reach it; at a slot that holds a texture view (the
/// host's table of strides says 0) they read the null buffer view of their
/// kind instead, not the texture view as a buffer (on AMD a wild address);
/// past the heap, too. With buffer views in the null views' places, the
/// accesses at the texture's slot find them.
#[test]
fn a_buffer_picked_from_the_heap_at_a_texture_slot_reads_the_null_buffer_view() {
    let mut words = vec![
        s_mov_b32(5, 0x80),     // s5 = 0: the table's high half
        s_mov_b32(7, 0x80),     // s7 = 0: the heap's high half
        sop2(0x1E, 8, 0, 0x85), // s_lshl_b32 s8, s0, 5: the load's index from data
        sop2(0x1E, 9, 1, 0x85), // s_lshl_b32 s9, s1, 5: the store's
    ];
    words.extend(smem(0x02, 12, 4, 125, 0)); // the output UAV, table entry 0
    words.extend(smem(0x02, 16, 6, 8, 0)); // heap slot s0
    words.extend(smem(0x02, 20, 6, 9, 0)); // heap slot s1
    words.extend(mubuf(0x0C, 1, 16, 4)); // buffer_load_dword v1, its dword 1
    words.push(v_mov(2, 0x87)); // v2 = 7
    words.extend(buffer_store_dword(2, 20, 8)); // to dword 2 of slot s1
    words.extend(buffer_store_dword(1, 12, 0));
    words.push(S_ENDPGM);
    let shader = lower_cs(&words, vec![used(ResourceType::Uav, 0, 0, 4)], true);
    assert_eq!(
        shader.root_srvs,
        [super::SLOT_STRIDES],
        "the table of strides"
    );
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(warp) = Warp::new(&shader) else {
        return;
    };
    // Slots: 0 the output, 1 a texture, 2 a raw buffer, 3 a raw UAV; the
    // table says which hold buffer views (raw: 1 byte per element).
    let mut strides = vec![0u32; super::SLOT_STRIDES_COUNT as usize];
    for slot in [0, 2, 3] {
        strides[slot] = 1;
    }
    let roots = [strides];
    let data = [0xB000, 0xB001];
    let texel = [[0.5; 4]];
    let listed = || {
        vec![
            View::RawUav(4),
            View::Texture(1, 1, &texel),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
            View::RawUav(4),
        ]
    };
    let run =
        |s0: u32, s1: u32, views: &[View]| warp.run(&shader, [s0, s1, 0, 0], views, &[], &roots);
    let mut views = listed();
    views.push(View::Guards(&[]));
    // (s0, s1): the output's dword 0, the UAV's dwords.
    let past = 1 << 25;
    let cases = [
        ((2, 3), 0xB001, [0, 0, 7, 0]),
        ((1, 1), 0, [0; 4]),
        ((past, past), 0, [0; 4]),
    ];
    for ((s0, s1), loaded, stored) in cases {
        let out = run(s0, s1, &views);
        assert_eq!(out[0][0], loaded, "({s0:#x}, {s1:#x}): the load");
        assert_eq!(out[1], stored, "({s0:#x}, {s1:#x}): the store");
    }

    // With buffer views in the null views' places, the accesses at the
    // texture's slot find them.
    let marked = [0xA000, 0xA001];
    let guards = [
        View::Texture(1, 1, &texel),
        View::Cbv(&[0; 4]),
        View::Buffer {
            data: &marked,
            stride: Stride::Raw,
        },
        View::RawUav(4),
    ];
    let mut views = listed();
    views.push(View::Guards(&guards));
    let out = run(1, 1, &views);
    assert_eq!(
        out[0][0], 0xA001,
        "the load in the null buffer view's place"
    );
    assert_eq!(out[1], [0; 4], "the UAV at slot 3 untouched");
    assert_eq!(out[2], [0, 0, 7, 0], "the store in the null UAV's place");
}

/// A scalar buffer load (`s_buffer_load_dword`) through a descriptor the
/// program picks from the view heap itself at an index it computes: raw at
/// a buffer view's slot, as a constant buffer at a constant buffer view's
/// ([`super::SLOT_CONSTANT`]), and at a texture view's slot (the table
/// says 0) through the null constant buffer view, not the texture view read
/// as a constant buffer: on AMD that is the texture's address shifted 8
/// bits down, the page fault that removed the device in the first frame of
/// the world. With a constant buffer view in the null view's place, the
/// load at the texture's slot finds it.
#[test]
fn a_constant_buffer_picked_from_the_heap_at_a_texture_slot_reads_the_null_view() {
    let mut words = vec![
        s_mov_b32(5, 0x80),     // s5 = 0: the table's high half
        s_mov_b32(7, 0x80),     // s7 = 0: the heap's high half
        sop2(0x1E, 8, 0, 0x85), // s_lshl_b32 s8, s0, 5: the index from data
    ];
    words.extend(smem(0x02, 12, 4, 125, 0)); // the output UAV, table entry 0
    words.extend(smem(0x02, 16, 6, 8, 0)); // heap slot s0
    words.extend(smem(0x08, 20, 16, 125, 4)); // s_buffer_load_dword s20, its dword 1
    words.push(v_mov(1, 20)); // v_mov_b32 v1, s20
    words.extend(buffer_store_dword(1, 12, 0));
    words.push(S_ENDPGM);
    let shader = lower_cs(&words, vec![used(ResourceType::Uav, 0, 0, 4)], true);
    assert_eq!(
        shader.root_srvs,
        [super::SLOT_STRIDES],
        "the table of strides"
    );
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(warp) = Warp::new(&shader) else {
        return;
    };
    // Slots: 0 the output, 1 a texture, 2 a constant buffer, 3 a raw buffer.
    let mut strides = vec![0u32; super::SLOT_STRIDES_COUNT as usize];
    (strides[0], strides[2], strides[3]) = (1, super::SLOT_CONSTANT, 1);
    let roots = [strides];
    let (constants, data) = ([0xC000, 0xC001, 0, 0], [0xB000, 0xB001]);
    let texel = [[0.5; 4]];
    let listed = || {
        vec![
            View::RawUav(4),
            View::Texture(1, 1, &texel),
            View::Cbv(&constants),
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
        ]
    };
    let run = |s0: u32, views: &[View]| warp.run(&shader, [s0, 0, 0, 0], views, &[], &roots)[0][0];
    let mut views = listed();
    views.push(View::Guards(&[]));
    for (s0, loaded) in [(3, 0xB001), (2, 0xC001), (1, 0), (1 << 25, 0)] {
        assert_eq!(run(s0, &views), loaded, "slot {s0:#x}");
    }

    // A constant buffer view in the null one's place: the load at the
    // texture's slot reads it.
    let marked = [0xA000, 0xA001, 0, 0];
    let guards = [
        View::Texture(1, 1, &texel),
        View::Cbv(&marked),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(4),
    ];
    let mut views = listed();
    views.push(View::Guards(&guards));
    assert_eq!(
        run(1, &views),
        0xA001,
        "the load in the null constant buffer view's place"
    );
}
