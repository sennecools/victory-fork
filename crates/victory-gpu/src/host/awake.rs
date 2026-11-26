//! Keeps the card's clocks up between submissions (`VICTORY_CARD_AWAKE=1`;
//! off by default). With the relay the card stands idle many times a frame,
//! for a millisecond or two each, while the CPU prepares the next
//! submission. The driver takes that for a light load and lowers the clocks,
//! and what comes next starts slow: cool_fox measured a replayed frame whose
//! lists take 28.6 ms sent in one go at 37.8 ms with 1.2 ms between its
//! submissions, and at 28.6 ms again with this on.
//!
//! One group of 64 threads counts on a compute queue of its own, over and
//! over: it takes one or two of the card's few thousand wave slots, and the
//! card is never idle. Nothing is sent once no submission has gone out for
//! [`QUIET_MS`]: a title that stands still, not a menu, which goes on
//! drawing and so keeps the card awake too (cool_fox saw the compute queue
//! at 100 % there as well).
//!
//! `VICTORY_CARD_AWAKE=turns`: [`TURN_MS`] on, as long off, by turns, to
//! compare the two in one run (the log says when each turn starts).
//!
//! The counting thread is started at the first submission ([`sent`]) and
//! is detached: its handle is forgotten, so it is never stopped and keeps
//! its own reference to the device for the life of the process. Dropping
//! an [`Awake`] stops the thread, waits for what it sent and closes its
//! event, but only for callers of [`start`] that keep the handle.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use windows::Win32::Foundation::CloseHandle;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::Interface;

/// About 2 ms at full clocks: some 500 sends a second.
const COUNT: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer sum : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    [loop] for (uint i = 0; i < 400000u; i++) x = x * 1664525u + 1013904223u;
    sum.Store(id.x * 4u, x);
}
"#;

/// Quiet this long after the last submission.
const QUIET_MS: u64 = 250;

/// With `VICTORY_CARD_AWAKE=turns`: this long on, this long off.
const TURN_MS: u64 = 20_000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    On,
    /// On and off by turns, to compare the two in one run.
    ByTurns,
}

/// `VICTORY_CARD_AWAKE`: unset or anything else (off, the default), `1`
/// (on), `turns` (on and off by turns).
fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| parse_mode(std::env::var_os("VICTORY_CARD_AWAKE").as_deref()))
}

fn parse_mode(value: Option<&std::ffi::OsStr>) -> Mode {
    match value.and_then(|v| v.to_str()) {
        Some("1") => Mode::On,
        Some("turns") => Mode::ByTurns,
        _ => Mode::Off,
    }
}

fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// When a submission was last sent ([`now_ms`] plus one; 0: never).
static LAST: AtomicU64 = AtomicU64::new(0);

/// Whether the card is to be kept busy at `now`, the last submission sent
/// at `last` (as in [`LAST`]).
fn wanted_at(mode: Mode, last: u64, now: u64) -> bool {
    mode != Mode::Off
        && last != 0
        && now + 1 < last + QUIET_MS
        && (mode != Mode::ByTurns || now / TURN_MS % 2 == 1)
}

fn wanted() -> bool {
    wanted_at(mode(), LAST.load(Ordering::Relaxed), now_ms())
}

/// A submission of the title's went to `device`'s queue (the late path,
/// `Host::submit`): the counting starts with the first, and goes on until
/// [`QUIET_MS`] after the last.
pub(super) fn sent(device: &ID3D12Device) {
    let mode = mode();
    if mode == Mode::Off {
        return;
    }
    let now = now_ms();
    let before = LAST.swap(now + 1, Ordering::Relaxed);
    if mode == Mode::ByTurns && (before == 0 || (before - 1) / TURN_MS != now / TURN_MS) {
        tracing::info!(
            on = now / TURN_MS % 2 == 1,
            "the card is kept awake, by turns"
        );
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        // Runs as long as the process does.
        std::mem::forget(start(device, wanted));
    });
}

/// The thread that keeps the card busy; it ends when this is dropped
/// ([`sent`] forgets its own, so that one runs as long as the process).
pub(super) struct Awake {
    stop: std::sync::Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Awake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Keeps the card of `device` busy while `wanted`.
pub(super) fn start(
    device: &ID3D12Device,
    wanted: impl Fn() -> bool + Send + 'static,
) -> Option<Awake> {
    let stop = std::sync::Arc::new(AtomicBool::new(false));
    let (device, ended) = (device.clone(), stop.clone());
    let thread = std::thread::Builder::new()
        .name("victory-card-awake".into())
        .spawn(move || {
            if let Err(e) = run(&device, &wanted, &ended) {
                tracing::warn!(error = %e, "the card is not kept awake (VICTORY_CARD_AWAKE)");
            }
        });
    match thread {
        Ok(thread) => Some(Awake {
            stop,
            thread: Some(thread),
        }),
        Err(e) => {
            tracing::warn!(error = %e, "no thread to keep the card awake (VICTORY_CARD_AWAKE)");
            None
        }
    }
}

fn run(
    device: &ID3D12Device,
    wanted: &dyn Fn() -> bool,
    stop: &AtomicBool,
) -> windows::core::Result<()> {
    let code = super::compile(
        COUNT,
        windows::core::s!("main"),
        windows::core::s!("cs_5_1"),
    )?;
    // SAFETY: plain object creation; the code lives through the calls, the
    // root signature is inside it. Two closed lists, sent in turn, whose
    // objects live until what was sent last has run; the event is ours and
    // closed once nothing waits on it.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso: ID3D12PipelineState =
            device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })?;
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            ..Default::default()
        })?;
        let mut sum: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 256,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut sum,
        )?;
        let sum = sum.ok_or_else(windows::core::Error::empty)?;
        let allocator: ID3D12CommandAllocator =
            device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)?;
        let mut lists = Vec::new();
        for _ in 0..2 {
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, &pso)?;
            list.SetComputeRootSignature(&root);
            list.SetComputeRootUnorderedAccessView(0, sum.GetGPUVirtualAddress());
            list.Dispatch(1, 1, 1);
            list.Close()?;
            lists.push(Some(list.cast::<ID3D12CommandList>()?));
        }
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let mut value = 0u64;
        let mut result = Ok(());
        while !stop.load(Ordering::Relaxed) && result.is_ok() {
            if !wanted() {
                std::thread::sleep(Duration::from_millis(2));
                continue;
            }
            // Never more than two on the way.
            if value >= 2 && fence.GetCompletedValue() + 2 <= value {
                result = fence.SetEventOnCompletion(value - 1, event);
                if result.is_ok() {
                    WaitForSingleObject(event, 1000);
                    result = device.GetDeviceRemovedReason();
                }
                continue;
            }
            queue.ExecuteCommandLists(&lists[(value % 2) as usize..][..1]);
            value += 1;
            result = queue.Signal(&fence, value);
        }
        // What was sent runs before its lists and the event go.
        if result.is_ok()
            && fence.GetCompletedValue() < value
            && fence.SetEventOnCompletion(value, event).is_ok()
        {
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_asked() {
        assert_eq!(parse_mode(None), Mode::Off);
        assert_eq!(parse_mode(Some("0".as_ref())), Mode::Off);
        assert_eq!(parse_mode(Some("yes".as_ref())), Mode::Off);
        assert_eq!(parse_mode(Some("1".as_ref())), Mode::On);
        assert_eq!(parse_mode(Some("turns".as_ref())), Mode::ByTurns);
    }

    #[test]
    fn wanted_only_shortly_after_a_submission() {
        // Never sent: nothing.
        assert!(!wanted_at(Mode::On, 0, 10));
        // Sent at 100 (stored as 101): on until the quiet time has passed.
        assert!(wanted_at(Mode::On, 101, 100));
        assert!(wanted_at(Mode::On, 101, 100 + QUIET_MS - 1));
        assert!(!wanted_at(Mode::On, 101, 100 + QUIET_MS));
        // Off is off whatever was sent.
        assert!(!wanted_at(Mode::Off, 101, 100));
    }

    #[test]
    fn by_turns_on_in_odd_turns_only() {
        let even = 2 * TURN_MS + 10;
        let odd = 3 * TURN_MS + 10;
        assert!(!wanted_at(Mode::ByTurns, even + 1, even));
        assert!(wanted_at(Mode::ByTurns, odd + 1, odd));
        assert!(wanted_at(Mode::On, even + 1, even));
    }
}
