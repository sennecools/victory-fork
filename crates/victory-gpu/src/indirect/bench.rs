//! A frame's worth of one-record indirect draws on the host GPU, clamped one
//! at a time and in windows, recorded as a build with no switch set records
//! them and with each of the three switches of `crate::indirect`, alone and
//! together:
//!
//! - `VICTORY_INDIRECT_READABLE=1`: the scratch buffer stays readable as
//!   arguments until the next clamp, where it otherwise goes back to the
//!   common state after every execution;
//! - `VICTORY_CLAMP_GROUP=1`: the clamp charges the budget once per thread
//!   group, where it otherwise charges record by record;
//! - `VICTORY_INDIRECT_CARD_BUDGET=1`: the budget is in the card's own
//!   memory, zeroed in the list and its notes copied out at the end, where
//!   it otherwise is in memory the CPU maps.
//!
//! Windows of 32 are what `VICTORY_INDIRECT_WINDOW=1` records for draws
//! that walk a buffer. The switches are read once per process, so the
//! benchmark does not go through them: it records each case's commands
//! itself, as `HostList` and `State::execute_indirect` do. (On a card
//! they say which cases run: below.)
//!
//! A measurement, not a check: it asserts only that every clamp was
//! recorded and that the budget was charged what the draws cost. Ignored
//! by default; run it by name with
//! `cargo test -p victory-gpu --release --lib indirect::bench -- --ignored --nocapture`.
//! It needs no title files: the draws are tiny triangles of ours. It runs
//! on the adapter the layer takes, so on the card unless
//! `VICTORY_D3D12_WARP=1` is set in a build with the `diagnostics` feature
//! (`--features victory-gpu/diagnostics`); on the software adapter it
//! records one frame per case to show that it runs, and its times mean
//! nothing.
//!
//! On a card this is the first run there of what the switches keep off by
//! default: a scratch buffer moved from arguments straight back to
//! unordered access, the group clamp (a compute shader of a new shape:
//! memory the group shares, two waits for the whole group) and
//! `WriteBufferImmediate`. So on a card a case runs only if each of its
//! switches is set to 1 in the benchmark's environment, the group clamp is
//! compiled only for `VICTORY_CLAMP_GROUP=1`, and the cases left out are
//! named. With no switch set a card runs the two cases a build with no
//! switch records, and nothing new; the software adapter runs all twelve.

use super::*;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

use crate::host::compile;

/// Draws a frame (a frame of the lit scene holds some 3,200 indirect
/// executions), records a window holds, and how far apart the records are
/// (the title keeps them 32 bytes apart).
const DRAWS: u32 = 3200;
const WINDOW: u32 = 32;
const STEP: u32 = 32;
/// Frames timed per case on a card; the median counts.
const RUNS: usize = 9;
/// The words of a list's budget buffer (`HostList::budget`): the budget and
/// the clamp's four notes after it.
const WORDS: u32 = 5;

const SHADERS: &str = r#"
#define RS "RootFlags(0)"
[RootSignature(RS)]
float4 vs(uint v : SV_VertexID, uint i : SV_InstanceID) : SV_Position {
    float2 p = float2((v & 1) ? 0.05 : 0.0, (v & 2) ? 0.05 : 0.0) - 0.5;
    return float4(p, 0.5, 1);
}
float4 ps() : SV_Target { return float4(1, 0.5, 0.25, 1); }
"#;

/// What one case records: the records a clamp takes (1, or a window), and
/// the three switches.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Case {
    window: u32,
    /// `VICTORY_INDIRECT_READABLE=1`.
    readable: bool,
    /// `VICTORY_CLAMP_GROUP=1`.
    grouped: bool,
    /// `VICTORY_INDIRECT_CARD_BUDGET=1`.
    card: bool,
}

impl Case {
    fn name(self) -> String {
        let how = if self.window > 1 {
            format!("windows of {}", self.window)
        } else {
            "one at a time".to_owned()
        };
        let mut switches = Vec::new();
        if self.readable {
            switches.push("READABLE");
        }
        if self.grouped {
            switches.push("CLAMP_GROUP");
        }
        if self.card {
            switches.push("CARD_BUDGET");
        }
        if switches.is_empty() {
            switches.push("no switch");
        }
        format!("{how}, {}", switches.join(" + "))
    }

    /// Whether a card runs the case: only with each of its switches set in
    /// the benchmark's environment.
    fn asked_for(self) -> bool {
        (!self.readable || readable())
            && (!self.grouped || by_group())
            && (!self.card || crate::host::card_budget())
    }
}

#[test]
#[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
fn bench_indirect_draw_barriers() {
    let Some(host) = super::host() else {
        eprintln!("no host GPU");
        return;
    };
    let device = &host.device;
    let (adapter, software) = super::tests::adapter_of(device);
    // The cases: one record at a time and in windows, each with no
    // switch, each switch alone, the two that change what a clamp
    // records together, and all three. (Bits of `i`: the switches.) A
    // card runs those whose switches are set.
    let (cases, left_out): (Vec<Case>, Vec<Case>) = [1, WINDOW]
        .into_iter()
        .flat_map(|window| {
            [0u32, 1, 2, 4, 3, 7].map(|i| Case {
                window,
                readable: i & 1 != 0,
                grouped: i & 2 != 0,
                card: i & 4 != 0,
            })
        })
        .partition(|case| software || case.asked_for());
    // The per-record clamp, and the one that charges once per group if a
    // case takes it; both with their notes of compute records, as with no
    // switch.
    let clamp_of = |source| make_from(device, source).expect("the clamp must compile");
    let clamps = [
        Some(clamp_of(SOURCE)),
        cases
            .iter()
            .any(|c| c.grouped)
            .then(|| clamp_of(GROUP_SOURCE)),
    ];
    // SAFETY: plain D3D12 object creation with valid descriptions; the
    // objects outlive every list recorded and run below.
    unsafe {
        let runs = if software { 1 } else { RUNS };
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .unwrap();
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .unwrap();
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            .unwrap();
        list.Close().unwrap();
        // What zeroes the card's copy of the budget; without it a list
        // keeps the mapped words, and so do the cases here.
        let list2 = list.cast::<ID3D12GraphicsCommandList2>().ok();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = CreateEventW(None, false, false, None).unwrap();
        let frequency = queue.GetTimestampFrequency().unwrap() as f64;

        // The pipeline: tiny triangles into a small target.
        let vs = compile(SHADERS, s!("vs"), s!("vs_5_1")).unwrap();
        let ps = compile(SHADERS, s!("ps"), s!("ps_5_1")).unwrap();
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).unwrap();
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            VS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: vs.as_ptr().cast(),
                BytecodeLength: vs.len(),
            },
            PS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: ps.as_ptr().cast(),
                BytecodeLength: ps.len(),
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        desc.BlendState.RenderTarget[0].RenderTargetWriteMask =
            D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc).unwrap();
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &target_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut target,
            )
            .unwrap();
        let target = target.unwrap();
        let rtvs: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap();
        let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
        device.CreateRenderTargetView(&target, None, rtv);

        // The title's records: three vertices, one instance, 32 bytes apart.
        let (src, src_ptr) = host.shared_buffer(u64::from(DRAWS * STEP)).unwrap();
        for i in 0..DRAWS as usize {
            let words = [3u32, 1, 0, 0];
            std::ptr::copy_nonoverlapping(
                words.as_ptr().cast::<u8>(),
                src_ptr.add(i * STEP as usize),
                16,
            );
        }
        // The budget's words in memory the CPU maps, and the same in the
        // card's own memory.
        let (budget, budget_ptr) = host.shared_buffer(256).unwrap();
        let budget_gpu = budget.GetGPUVirtualAddress();
        let card_budget = host.uav_buffer(256).unwrap();
        let card_budget_gpu = card_budget.GetGPUVirtualAddress();
        let scratch = host
            .uav_buffer(u64::from(WINDOW * STEP).max(64 << 10))
            .unwrap();
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: 16,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            device
                .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
            sig.unwrap()
        };
        let stamps: ID3D12QueryHeap = {
            let mut heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .unwrap();
            heap.unwrap()
        };
        let mut back: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: 16,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .unwrap();
        let back = back.unwrap();
        let mut submitted = 0u64;

        // One frame's list: GPU ms (timestamps around the list), CPU ms
        // recording it, ms from submission to the end of its run.
        let mut frame = |case: Case| -> [f64; 3] {
            let (common, arguments) = (
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            );
            let clamp = clamps[usize::from(case.grouped)]
                .as_ref()
                .expect("a clamp for every case that runs");
            let card = list2.as_ref().filter(|_| case.card);
            allocator.Reset().unwrap();
            list.Reset(&allocator, None).unwrap();
            let recording = std::time::Instant::now();
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 256.0,
                Height: 256.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: 256,
                bottom: 256,
            }]);
            list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            list.SetGraphicsRootSignature(&root);
            // The card's copy of the budget is zeroed in the list before
            // its first clamp (`HostList::zero_card_budget`).
            if let Some(list2) = card {
                let zeros: [_; WORDS as usize] =
                    std::array::from_fn(|word| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
                        Dest: card_budget_gpu + 4 * word as u64,
                        Value: 0,
                    });
                let copied = D3D12_RESOURCE_STATE_COPY_DEST;
                list.ResourceBarrier(&[transition(&card_budget, common, copied)]);
                list2.WriteBufferImmediate(WORDS, zeros.as_ptr(), None);
                list.ResourceBarrier(&[transition(
                    &card_budget,
                    copied,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                )]);
            }
            let mut state = common;
            for first in (0..DRAWS).step_by(case.window as usize) {
                let records = case.window.min(DRAWS - first);
                assert!(record_clamp_as(
                    clamp,
                    &list,
                    (&scratch, state),
                    if card.is_some() {
                        card_budget_gpu
                    } else {
                        budget_gpu
                    },
                    (&src, u64::from(first * STEP)),
                    None,
                    (0, STEP, records),
                    if case.window > 1 { AHEAD } else { 0 },
                ));
                state = arguments;
                list.SetPipelineState(&pso);
                for k in 0..records {
                    list.ExecuteIndirect(&signature, 1, &scratch, u64::from(k * STEP), None, 0);
                    // Without the switch every execution is followed by the
                    // scratch buffer's move back to common (`done`).
                    if !case.readable {
                        list.ResourceBarrier(&[transition(&scratch, arguments, common)]);
                        state = common;
                    }
                }
            }
            // What the list's close records: a scratch buffer kept readable
            // goes back to common, and the card's copy of the budget hands
            // its notes to the mapped words (`HostList::card_budget_out`).
            if state != common {
                list.ResourceBarrier(&[transition(&scratch, state, common)]);
            }
            if card.is_some() {
                let (written, read) = (
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                list.ResourceBarrier(&[transition(&card_budget, written, read)]);
                list.ResourceBarrier(&[transition(
                    &budget,
                    common,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
                list.CopyBufferRegion(&budget, 4, &card_budget, 4, u64::from(4 * (WORDS - 1)));
                list.ResourceBarrier(&[transition(&card_budget, read, common)]);
            }
            list.EndQuery(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
            list.Close().unwrap();
            let recorded = recording.elapsed().as_secs_f64() * 1e3;
            let running = std::time::Instant::now();
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            submitted += 1;
            queue.Signal(&fence, submitted).unwrap();
            fence.SetEventOnCompletion(submitted, event).unwrap();
            WaitForSingleObject(event, INFINITE);
            let ran = running.elapsed().as_secs_f64() * 1e3;
            let mut data = std::ptr::null_mut();
            back.Map(0, None, Some(&mut data)).unwrap();
            let t = std::slice::from_raw_parts(data.cast::<u64>(), 2);
            let gpu = t[1].saturating_sub(t[0]) as f64 / frequency * 1e3;
            back.Unmap(0, None);
            // The mapped budget word: charged three vertices a draw, unless
            // the clamps charged the card's copy. Zeroed for the next frame
            // as a submission zeroes it.
            let charged = (budget_ptr as *mut u32).read_volatile();
            let wanted = if card.is_some() { 0 } else { 3 * DRAWS };
            assert_eq!(charged, wanted, "{}: the mapped budget word", case.name());
            (budget_ptr as *mut u32).write_volatile(0);
            [gpu, recorded, ran]
        };

        // Warm up, then the cases in turn, so drift falls on all alike.
        for &case in &cases {
            frame(case);
        }
        let mut times: Vec<Vec<[f64; 3]>> = vec![Vec::new(); cases.len()];
        for _ in 0..runs {
            for (i, &case) in cases.iter().enumerate() {
                times[i].push(frame(case));
            }
        }
        let median = |v: &[[f64; 3]], k: usize| {
            let mut x: Vec<f64> = v.iter().map(|t| t[k]).collect();
            x.sort_by(f64::total_cmp);
            x[x.len() / 2]
        };
        eprintln!(
            "indirect draws on {adapter}: {DRAWS} one-record draws a frame, medians of {runs} \
             frames"
        );
        if software {
            eprintln!("  a software adapter: these times say nothing about a card");
        }
        if list2.is_none() {
            eprintln!(
                "  no WriteBufferImmediate here: the CARD_BUDGET cases kept the mapped words"
            );
        }
        for case in &left_out {
            eprintln!(
                "  left out on a card (set each of its switches to 1 to run it): {}",
                case.name()
            );
        }
        for (i, case) in cases.iter().enumerate() {
            let gpu = median(&times[i], 0);
            eprintln!(
                "  {:<52} GPU {gpu:7.2} ms ({:5.2} us a draw), recording {:6.2} ms, submit to \
                 done {:7.2} ms",
                case.name(),
                gpu * 1e3 / f64::from(DRAWS),
                median(&times[i], 1),
                median(&times[i], 2),
            );
        }
        // Each case against the one with no switch that clamps as it does.
        for (i, case) in cases.iter().enumerate() {
            let Some(base) = cases
                .iter()
                .position(|c| c.window == case.window && !(c.readable || c.grouped || c.card))
                .filter(|&base| base != i)
            else {
                continue;
            };
            let (a, b) = (median(&times[base], 0), median(&times[i], 0));
            eprintln!(
                "  {} -> {}: GPU time {:+.1} %",
                cases[base].name(),
                case.name(),
                (b / a - 1.0) * 100.0
            );
        }
        assert!(
            times
                .iter()
                .flatten()
                .all(|t| t.iter().all(|ms| ms.is_finite())),
            "a frame gave no time"
        );
    }
}
