//! End-to-end tests of descriptor fields the console compiler reads as
//! numbers from loads that nothing uses as a descriptor: synthetic GFX10
//! programs lowered to DXIL, validated by `dxil.dll` when present, and run
//! on the host GPU and WARP, the fields rebuilt from the host views.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `s_mov_b32 s1, 0`: the high half of the table's address at s[0:1].
const S_MOV_S1_0: u32 = 0xBE81_0380;
/// `s_waitcnt vmcnt(0) lgkmcnt(0)`.
const S_WAITCNT: u32 = 0xBF8C_0070;

/// SMEM `s_load_dwordx4` `s[dst:dst+3]`, `s[0:1]`, `offset` (soffset null).
const fn s_load_x4(dst: u8, offset: u32) -> [u32; 2] {
    [
        0xF400_0000 | 2 << 18 | (dst as u32) << 6,
        125 << 25 | offset,
    ]
}

/// `v_mov_b32 vdst, s<src>`, then MUBUF `buffer_store_dword vdst, off,
/// s[16:19], 0 offset:byte`.
fn store(vdst: u8, src: u8, byte: u32) -> [u32; 3] {
    [
        0x7E00_0000 | u32::from(vdst) << 17 | 0x01 << 9 | u32::from(src),
        0x38 << 26 | 0x1C << 18 | byte,
        0x80 << 24 | (16 >> 2) << 16 | u32::from(vdst) << 8,
    ]
}

/// Lowers a wave32 compute program of one thread with the table of views at
/// s0: `first` (a descriptor of `dwords` at dword 0, with `stride`) and a raw
/// UAV at dword 8 (heap index 1), the result the program stores there.
fn build(words: &[u32], first: ResourceType, dwords: u8, stride: u32) -> HostShader {
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(first, 0, dwords, stride),
            range(ResourceType::Uav, 8, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
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
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with `views` at heap indices 0.. (the UAV last) on the
/// hardware GPU and on WARP (each skipped without a device) and returns,
/// per one that ran, the UAV's dwords. The table starts at heap index 0;
/// the view addresses the host would tell read 0.
fn run(shader: &HostShader, views: &[View]) -> Vec<(&'static str, Vec<u32>)> {
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words[0].clone()));
    }
    out
}

#[test]
fn an_image_read_only_for_its_size_takes_the_kind_a_sibling_load_samples() {
    // Two image descriptors loaded from one table by computed indices: one
    // read only for its size, the other read as a 3D image (the puddle
    // passes size one texture of a table and sample others). The sized one
    // takes the 3D kind, so its depth (dword 4) is the host view's, not the
    // 1 of the 2D texture nothing else would tell.
    let mut words = vec![
        S_MOV_S1_0,
        0x7E0C_0280, // v_mov_b32 v6, 0
        0x7E0E_0280, // v_mov_b32 v7, 0
        0x7E10_0280, // v_mov_b32 v8, 0
        0x7E08_0500, // v_readfirstlane_b32 s4, v0 (the thread's index, 0)
        0x8F05_8104, // s_lshl_b32 s5, s4, 1 (0, another computed term)
        0xF40C_0200,
        0x0800_0000, // s_load_dwordx8 s[8:15], s[0:1], s4: read for its size
        0xF40C_0600,
        0x0A00_0000, // s_load_dwordx8 s[24:31], s[0:1], s5: read as an image
    ];
    words.extend(s_load_x4(16, 0x20)); // s[16:19]: the result's UAV
    words.push(S_WAITCNT);
    words.extend([0xF000_1110, 0x0006_0D06]); // image_load v13, v[6:8], s[24:31] dmask:0x1 3D
    words.push(S_WAITCNT);
    words.extend(store(0, 12, 0)); // dword 4: depth - 1
    words.extend(store(1, 11, 4)); // dword 3: last level and type
    // buffer_store_dword v13, off, s[16:19], 0 offset:8
    words.extend([
        0x38 << 26 | 0x1C << 18 | 8,
        0x80 << 24 | (16 >> 2) << 16 | 13 << 8,
    ]);
    words.push(S_ENDPGM);
    let shader = build(&words, ResourceType::Srv, 8, 0);
    let texels: Vec<[f32; 4]> = (0..4 * 2 * 3).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let views = [View::Texture3D(4, 2, 3, &texels), View::RawUav(8)];
    let results = run(&shader, &views);
    assert!(!results.is_empty());
    for (label, got) in results {
        assert_eq!(got[0], 2, "{label}: depth - 1");
        assert_eq!(got[1], 0xA000_0FAC, "{label}: a 3D image of one level");
    }
}

#[test]
fn the_record_count_of_a_buffer_view_never_read_through_comes_from_the_host_view() {
    // A buffer descriptor whose record count (dword 2) the code tests and
    // nothing reads through, as a program does that skips its work for an
    // empty buffer: the count from the host's raw view, in elements of the
    // used entry's stride (bytes for none).
    let mut words = vec![S_MOV_S1_0];
    words.extend(s_load_x4(8, 0)); // s[8:11]: the buffer's descriptor
    words.extend(s_load_x4(16, 0x20)); // s[16:19]: the result's UAV
    words.push(S_WAITCNT);
    words.extend(store(0, 10, 0));
    words.push(S_ENDPGM);
    let data: Vec<u32> = (0..32).collect();
    for (stride, want) in [(0, 128), (16, 8)] {
        let shader = build(&words, ResourceType::Srv, 4, stride);
        let views = [
            View::Buffer {
                data: &data,
                stride: Stride::Raw,
            },
            View::RawUav(8),
        ];
        let results = run(&shader, &views);
        assert!(!results.is_empty());
        for (label, got) in results {
            assert_eq!(got[0], want, "{label}, stride {stride}");
        }
    }
}
