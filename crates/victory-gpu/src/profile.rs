//! The frame-anatomy profiler (`VICTORY_PROFILE`): spans of what each thread
//! does (a submission and its phases, the recording of each title list, the
//! heap buffer and texture passes, pipeline waits, presents) and, for each
//! waited submission, when the card ran each host list, written as a Chrome
//! trace (`{"traceEvents": [...]}`, complete `X` events, microseconds) that
//! opens in Perfetto (ui.perfetto.dev) or `chrome://tracing`. The `frame`
//! events (present to present, with the frame's draws) delimit frames, so a
//! reader can make one "where did the frame go" table per frame.
//!
//! `VICTORY_PROFILE=<path>` writes there; `=1` writes `<cache>\profile.json`
//! (the app points this at Victory's cache folder; `%TEMP%\victory-profile.json`
//! if it did not). Recording starts at the first presented frame that
//! recorded more than [`FIRST_DRAWS`] draws (`VICTORY_PROFILE_DELAY=<s>`:
//! that many seconds after the profiler was first asked instead, in practice
//! when the title makes its first command list) and lasts `VICTORY_PROFILE_SECONDS` (60), or
//! until the file holds [`MOST_BYTES`]. The closing `]}` is written after
//! every chunk and written over by the next, so the file is valid JSON
//! whenever the run ends.
//!
//! Release-buildable (no `diagnostics` feature) and off unless the switch is
//! set: a span then costs one relaxed atomic load. Each thread records into a
//! buffer of its own; one writer thread drains them ten times a second.
//!
//! The card's side ([`gpu`]): in a submission that is waited for (the default;
//! not those sent ahead with `VICTORY_LATE_WAIT` or the relay), a timestamp
//! before and after each host list, resolved by a list at the end of the same
//! submission and read after the wait the submission makes anyway; put on the
//! CPU clock with `GetClockCalibration`. They show as the process "GPU". Our
//! other sends (the card copies' uploads in `card_before`, their fetches in
//! `card_back`) are not stamped and show as gaps there.
//!
//! The recording of each title list (open to close) shows as the process
//! "title lists", one line per list (lists overlap on one thread, and the
//! title can close a list on another thread than the one that opened it).
//!
//! Threads are named by their Rust name (ours) or `thread <id>`; the names
//! the title gives its threads are not written.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Draws a presented frame records past which recording starts (the menus
/// record a few dozen, the world some 5,000).
const FIRST_DRAWS: u64 = 1000;
/// The most the file may hold; recording stops there.
const MOST_BYTES: u64 = 512 << 20;
/// Seconds recorded unless `VICTORY_PROFILE_SECONDS` says otherwise.
const SECONDS: u64 = 60;

/// Not decided yet; off; on and waiting for its first frame; recording;
/// done.
const UNSET: u8 = 0;
const OFF: u8 = 1;
const ARMED: u8 = 2;
const ON: u8 = 3;
const DONE: u8 = 4;

static STATE: AtomicU8 = AtomicU8::new(UNSET);
/// Where `=1` writes; the app points this at Victory's cache folder.
static DEFAULT_PATH: OnceLock<PathBuf> = OnceLock::new();
static CONFIG: OnceLock<Option<Config>> = OnceLock::new();
/// When the profiler was armed ([`now`]).
static ARMED_AT: AtomicU64 = AtomicU64::new(0);

struct Config {
    path: PathBuf,
    seconds: u64,
    /// Seconds after arming to start at, instead of the first busy frame.
    delay: Option<f64>,
}

/// Points `VICTORY_PROFILE=1` at `path` (the app passes
/// `<cache>\profile.json`). Call before the first presented frame.
pub fn use_profile_path(path: PathBuf) {
    let _ = DEFAULT_PATH.set(path);
}

/// Whether spans are recorded now.
#[inline]
pub(crate) fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        ON => true,
        UNSET => {
            decide();
            false
        }
        _ => false,
    }
}

#[cold]
fn decide() {
    let config = CONFIG.get_or_init(config_from_env);
    let next = if config.is_some() { ARMED } else { OFF };
    if STATE
        .compare_exchange(UNSET, next, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
        && let Some(config) = config
    {
        ARMED_AT.store(now(), Ordering::Relaxed);
        tracing::info!(
            path = %config.path.display(),
            seconds = config.seconds,
            delay = ?config.delay,
            "profile armed"
        );
    }
}

fn config_from_env() -> Option<Config> {
    let value = std::env::var_os("VICTORY_PROFILE")?;
    if value.is_empty() || value.to_str() == Some("0") {
        return None;
    }
    let path = if value.to_str() == Some("1") {
        DEFAULT_PATH
            .get()
            .cloned()
            .unwrap_or_else(|| std::env::temp_dir().join("victory-profile.json"))
    } else {
        PathBuf::from(value)
    };
    let seconds = std::env::var("VICTORY_PROFILE_SECONDS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&s| s > 0)
        .unwrap_or(SECONDS);
    let delay = std::env::var("VICTORY_PROFILE_DELAY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|d| d.is_finite() && *d >= 0.0);
    Some(Config {
        path,
        seconds,
        delay,
    })
}

/// The clock: `QueryPerformanceCounter` ticks (what `GetClockCalibration`
/// gives on the CPU side), never 0.
#[cfg(windows)]
fn now() -> u64 {
    let mut ticks = 0i64;
    // SAFETY: writes one integer.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut ticks) };
    (ticks as u64).max(1)
}

/// Ticks of [`now`] per second.
#[cfg(windows)]
fn frequency() -> u64 {
    static FREQUENCY: OnceLock<u64> = OnceLock::new();
    *FREQUENCY.get_or_init(|| {
        let mut frequency = 0i64;
        // SAFETY: writes one integer.
        unsafe {
            windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut frequency)
        };
        (frequency as u64).max(1)
    })
}

#[cfg(not(windows))]
fn now() -> u64 {
    static BASE: OnceLock<std::time::Instant> = OnceLock::new();
    BASE.get_or_init(std::time::Instant::now)
        .elapsed()
        .as_nanos() as u64
        + 1
}

#[cfg(not(windows))]
fn frequency() -> u64 {
    1_000_000_000
}

/// Which line of the trace an event goes to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Track {
    /// The thread that recorded it.
    Thread,
    /// The frames line (present to present).
    Frames,
    /// The card's line.
    Gpu,
    /// The line of one title list (by id): lists overlap on one thread and
    /// can be closed on another thread than the one that opened them, so
    /// their spans would not nest on the threads' lines.
    List(u32),
}

/// The most named numbers an event carries.
const MOST_ARGS: usize = 6;

/// Arguments an event carries: up to [`MOST_ARGS`] named numbers.
type Args = [(&'static str, u64); MOST_ARGS];

/// One span, in [`now`] ticks.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Event {
    name: &'static str,
    start: u64,
    end: u64,
    track: Track,
    args: Args,
    args_len: u8,
}

impl Event {
    fn new(
        name: &'static str,
        start: u64,
        end: u64,
        track: Track,
        args: &[(&'static str, u64)],
    ) -> Event {
        let mut event = Event {
            name,
            start,
            end,
            track,
            args: [("", 0); MOST_ARGS],
            args_len: 0,
        };
        for &(key, value) in args.iter().take(MOST_ARGS) {
            event.args[usize::from(event.args_len)] = (key, value);
            event.args_len += 1;
        }
        event
    }
}

/// A thread's events since the writer last took them.
struct Buffer {
    tid: u32,
    name: String,
    events: Mutex<Vec<Event>>,
}

static BUFFERS: Mutex<Vec<Arc<Buffer>>> = Mutex::new(Vec::new());

thread_local! {
    static LOCAL: std::cell::OnceCell<Arc<Buffer>> = const { std::cell::OnceCell::new() };
}

#[cfg(windows)]
fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

#[cfg(not(windows))]
fn thread_id() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    thread_local! {
        static ID: u32 = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    ID.with(|id| *id)
}

fn register() -> Arc<Buffer> {
    let tid = thread_id();
    let name = std::thread::current()
        .name()
        .map_or_else(|| format!("thread {tid}"), str::to_owned);
    let buffer = Arc::new(Buffer {
        tid,
        name,
        events: Mutex::new(Vec::with_capacity(256)),
    });
    BUFFERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(buffer.clone());
    buffer
}

fn push(event: Event) {
    // (A thread being torn down has no buffer any more: its event is lost.)
    let _ = LOCAL.try_with(|local| {
        let buffer = local.get_or_init(register);
        buffer
            .events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    });
}

/// Records `events` on the calling thread's buffer, while recording.
pub(crate) fn push_all(events: Vec<Event>) {
    if STATE.load(Ordering::Relaxed) != ON || events.is_empty() {
        return;
    }
    let _ = LOCAL.try_with(|local| {
        let buffer = local.get_or_init(register);
        buffer
            .events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(events);
    });
}

/// A span of the calling thread: from [`span`] until dropped.
#[must_use = "a span ends when it is dropped"]
pub(crate) struct Span {
    name: &'static str,
    start: u64,
    args: Args,
    args_len: u8,
}

impl Span {
    /// Adds a named number ([`MOST_ARGS`] at most).
    pub(crate) fn arg(&mut self, key: &'static str, value: u64) {
        if usize::from(self.args_len) < self.args.len() {
            self.args[usize::from(self.args_len)] = (key, value);
            self.args_len += 1;
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if STATE.load(Ordering::Relaxed) == ON {
            let args = &self.args[..usize::from(self.args_len)];
            push(Event::new(
                self.name,
                self.start,
                now(),
                Track::Thread,
                args,
            ));
        }
    }
}

/// A span named `name` from now until the result is dropped; `None` (and
/// nothing recorded) while the profiler does not record.
#[inline]
pub(crate) fn span(name: &'static str) -> Option<Span> {
    span_args(name, &[])
}

/// [`span`] with named numbers ([`MOST_ARGS`] at most).
#[inline]
pub(crate) fn span_args(name: &'static str, args: &[(&'static str, u64)]) -> Option<Span> {
    if !on() {
        return None;
    }
    let mut span = Span {
        name,
        start: now(),
        args: [("", 0); MOST_ARGS],
        args_len: 0,
    };
    for &(key, value) in args {
        span.arg(key, value);
    }
    Some(span)
}

/// The start of a span that ends in another call ([`since`]); 0 while the
/// profiler does not record.
#[inline]
pub(crate) fn stamp() -> u64 {
    if on() { now() } else { 0 }
}

/// Ticks from `start` ([`stamp`]) until now; 0 for a `start` of 0, with no
/// look at the clock.
#[inline]
pub(crate) fn elapsed(start: u64) -> u64 {
    if start == 0 {
        0
    } else {
        now().saturating_sub(start)
    }
}

/// `ticks` of the clock ([`stamp`], [`elapsed`]) in whole microseconds.
pub(crate) fn micros(ticks: u64) -> u64 {
    (u128::from(ticks) * 1_000_000 / u128::from(frequency())) as u64
}

/// Records a span named `name` of the calling thread from `start`
/// ([`stamp`]; 0: nothing) until now, if it lasted `least_us` microseconds
/// or more: a wait timed only when it happens, kept only when it is long.
#[inline]
pub(crate) fn since(name: &'static str, start: u64, least_us: u64, args: &[(&'static str, u64)]) {
    if start == 0 || STATE.load(Ordering::Relaxed) != ON {
        return;
    }
    let end = now();
    if micros(end.saturating_sub(start)) >= least_us {
        push(Event::new(name, start, end, Track::Thread, args));
    }
}

thread_local! {
    /// The ticks and the count of the parts timed on this thread
    /// ([`part_start`], [`part_end`]) since a span began to sum them
    /// ([`parts_begin`]); `None` while no span sums them.
    static PARTS: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// A span of the calling thread begins to sum the time of parts timed
/// deeper down (a call made many times inside it, too short and too many
/// for a span each). The caller holds a recording span.
pub(crate) fn parts_begin() {
    let _ = PARTS.try_with(|parts| parts.set(Some((0, 0))));
}

/// The ticks and the count of the parts timed since [`parts_begin`]; they
/// are no longer summed after it.
pub(crate) fn parts_end() -> (u64, u64) {
    PARTS
        .try_with(std::cell::Cell::take)
        .ok()
        .flatten()
        .unwrap_or_default()
}

/// The start of a part ([`part_end`]): 0, with no look at the clock,
/// unless recording and a span of this thread sums the parts.
#[inline]
pub(crate) fn part_start() -> u64 {
    if on()
        && PARTS
            .try_with(|parts| parts.get().is_some())
            .unwrap_or(false)
    {
        now()
    } else {
        0
    }
}

/// The part begun at `start` ([`part_start`]) ends: its time is added to
/// the sum of the span that sums them.
#[inline]
pub(crate) fn part_end(start: u64) {
    if start == 0 {
        return;
    }
    let ticks = elapsed(start);
    let _ = PARTS.try_with(|parts| {
        if let Some((sum, count)) = parts.get() {
            parts.set(Some((sum + ticks, count + 1)));
        }
    });
}

/// Records the recording of title list `list` from `start` ([`stamp`]; 0:
/// nothing) until now, on the list's own line, with the closing thread.
#[inline]
pub(crate) fn list_recorded(list: u32, start: u64) {
    if start != 0 && STATE.load(Ordering::Relaxed) == ON {
        push(Event::new(
            "title list",
            start,
            now(),
            Track::List(list),
            &[
                ("list", u64::from(list)),
                ("thread", u64::from(thread_id())),
            ],
        ));
    }
}

/// At every present, with `recorded`, the draws the title has recorded so
/// far: starts recording at the first busy frame (or after the delay), and
/// while recording puts the frame (from the last present) on the frames
/// line.
pub(crate) fn presented(recorded: u64) {
    let state = STATE.load(Ordering::Relaxed);
    if state == OFF || state == DONE {
        return;
    }
    if state == UNSET {
        decide();
    }
    static LAST_RECORDED: AtomicU64 = AtomicU64::new(0);
    static LAST_PRESENT: AtomicU64 = AtomicU64::new(0);
    static FRAMES: AtomicU64 = AtomicU64::new(0);
    let draws = recorded.saturating_sub(LAST_RECORDED.swap(recorded, Ordering::Relaxed));
    let at = now();
    let last = LAST_PRESENT.swap(at, Ordering::Relaxed);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    match STATE.load(Ordering::Relaxed) {
        ARMED => {
            let Some(Some(config)) = CONFIG.get() else {
                return;
            };
            let due = match config.delay {
                Some(delay) => {
                    let armed = ARMED_AT.load(Ordering::Relaxed);
                    at.saturating_sub(armed) as f64 / frequency() as f64 >= delay
                }
                // (Not at the first present: its draws are all since the start.)
                None => last != 0 && draws > FIRST_DRAWS,
            };
            if due {
                start(config, frame, draws);
            }
        }
        ON if last != 0 => push(Event::new(
            "frame",
            last,
            at,
            Track::Frames,
            &[("frame", frame), ("draws", draws)],
        )),
        _ => {}
    }
}

fn start(config: &'static Config, frame: u64, draws: u64) {
    if STATE
        .compare_exchange(ARMED, ON, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    // The trace's 0.
    let at = now();
    tracing::info!(frame, draws, path = %config.path.display(), "profile recording");
    let spawned = std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(move || write_out(config, at));
    if let Err(error) = spawned {
        tracing::warn!(%error, "no profile writer");
        STATE.store(DONE, Ordering::Relaxed);
    }
}

/// The writer thread: drains the buffers ten times a second until the time
/// or the size is up.
fn write_out(config: &'static Config, base: u64) {
    let mut writer = match Writer::create(&config.path, base, frequency()) {
        Ok(writer) => writer,
        Err(error) => {
            tracing::warn!(path = %config.path.display(), %error, "no profile");
            STATE.store(DONE, Ordering::Relaxed);
            return;
        }
    };
    let started = std::time::Instant::now();
    let length = std::time::Duration::from_secs(config.seconds);
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
        let done = started.elapsed() >= length || writer.bytes() >= MOST_BYTES;
        if done {
            STATE.store(DONE, Ordering::Relaxed);
        }
        let buffers: Vec<Arc<Buffer>> = BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let taken: Vec<(u32, &str, Vec<Event>)> = buffers
            .iter()
            .map(|b| {
                // The thread keeps a buffer as large: its next pushes do not
                // allocate.
                let mut held = b.events.lock().unwrap_or_else(|e| e.into_inner());
                let capacity = held.capacity();
                let events = std::mem::replace(&mut *held, Vec::with_capacity(capacity));
                drop(held);
                (b.tid, b.name.as_str(), events)
            })
            .collect();
        if let Err(error) = writer.write(&taken) {
            tracing::warn!(path = %config.path.display(), %error, "profile cut short");
            STATE.store(DONE, Ordering::Relaxed);
            return;
        }
        if done {
            break;
        }
    }
    tracing::info!(
        path = %config.path.display(),
        mb = writer.bytes() >> 20,
        "profile written"
    );
}

/// What closes the trace; written after every chunk and over by the next.
const TRAILER: &str = "\n]}\n";

/// The trace file.
struct Writer {
    file: std::fs::File,
    /// Where the trailer starts.
    at: u64,
    /// Lines whose name was written (pid, tid).
    named: std::collections::HashSet<(u32, u32)>,
    /// Ticks at the trace's 0, and per second.
    base: u64,
    frequency: f64,
}

impl Writer {
    fn create(path: &Path, base: u64, frequency: u64) -> std::io::Result<Writer> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        let head = format!(
            "{{\"displayTimeUnit\":\"ms\",\"otherData\":{{\"version\":\"{}\",\
             \"clock\":\"qpc\",\"clock_frequency\":{frequency}}},\"traceEvents\":[\n\
             {{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{{\"name\":\"CPU\"}}}},\n\
             {{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":2,\"tid\":0,\"args\":{{\"name\":\"GPU\"}}}},\n\
             {{\"name\":\"process_sort_index\",\"ph\":\"M\",\"pid\":2,\"tid\":0,\"args\":{{\"sort_index\":-1}}}},\n\
             {{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":3,\"tid\":0,\"args\":{{\"name\":\"title lists\"}}}},\n\
             {{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":0,\"args\":{{\"name\":\"frames\"}}}},\n\
             {{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":2,\"tid\":1,\"args\":{{\"name\":\"host queue\"}}}}",
            env!("CARGO_PKG_VERSION")
        );
        file.write_all(head.as_bytes())?;
        file.write_all(TRAILER.as_bytes())?;
        Ok(Writer {
            file,
            at: head.len() as u64,
            named: std::collections::HashSet::new(),
            base,
            frequency: frequency as f64,
        })
    }

    fn bytes(&self) -> u64 {
        self.at + TRAILER.len() as u64
    }

    /// Microseconds from the trace's 0 to `ticks`.
    fn micros(&self, ticks: u64) -> f64 {
        ticks.wrapping_sub(self.base) as i64 as f64 * 1e6 / self.frequency
    }

    /// Appends each thread's events (thread id, name, events).
    fn write(&mut self, threads: &[(u32, &str, Vec<Event>)]) -> std::io::Result<()> {
        use std::fmt::Write as _;
        use std::io::{Seek, Write};
        let mut chunk = String::new();
        for (tid, name, events) in threads {
            if events.is_empty() {
                continue;
            }
            for event in events {
                let (pid, tid) = match event.track {
                    Track::Thread => (1, *tid),
                    Track::Frames => (1, 0),
                    Track::Gpu => (2, 1),
                    Track::List(list) => (3, list),
                };
                if (pid == 1 && tid != 0 || pid == 3) && self.named.insert((pid, tid)) {
                    let _ = write!(
                        chunk,
                        ",\n{{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":{pid},\"tid\":{tid},\"args\":{{\"name\":"
                    );
                    if pid == 3 {
                        let _ = write!(chunk, "\"list {tid}\"");
                    } else {
                        json_string(&mut chunk, name);
                    }
                    chunk.push_str("}}");
                }
                chunk.push_str(",\n{\"name\":");
                json_string(&mut chunk, event.name);
                let ts = self.micros(event.start);
                let dur = event.end.saturating_sub(event.start) as f64 * 1e6 / self.frequency;
                let _ = write!(
                    chunk,
                    ",\"ph\":\"X\",\"ts\":{ts:.3},\"dur\":{dur:.3},\"pid\":{pid},\"tid\":{tid}"
                );
                if event.args_len > 0 {
                    chunk.push_str(",\"args\":{");
                    for (i, (key, value)) in
                        event.args[..usize::from(event.args_len)].iter().enumerate()
                    {
                        if i > 0 {
                            chunk.push(',');
                        }
                        json_string(&mut chunk, key);
                        let _ = write!(chunk, ":{value}");
                    }
                    chunk.push('}');
                }
                chunk.push('}');
            }
        }
        if chunk.is_empty() {
            return Ok(());
        }
        self.file.seek(std::io::SeekFrom::Start(self.at))?;
        self.file.write_all(chunk.as_bytes())?;
        self.at += chunk.len() as u64;
        self.file.write_all(TRAILER.as_bytes())?;
        Ok(())
    }
}

/// `text` as a JSON string.
fn json_string(out: &mut String, text: &str) {
    use std::fmt::Write;
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The card's timestamps around the host lists of a waited submission.
#[cfg(windows)]
pub(crate) mod gpu {
    use super::{Event, Track};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_HEAP_FLAG_NONE, D3D12_HEAP_PROPERTIES,
        D3D12_HEAP_TYPE_READBACK, D3D12_QUERY_HEAP_DESC, D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
        D3D12_QUERY_TYPE_TIMESTAMP, D3D12_RANGE, D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator, ID3D12CommandList,
        ID3D12CommandQueue, ID3D12Device, ID3D12GraphicsCommandList, ID3D12QueryHeap,
        ID3D12Resource,
    };
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::core::Interface;

    /// Timestamps a submission may take: two per stamped part.
    const MOST: u32 = 256;

    /// A part of a submission's lists to stamp: `lists[first..end]`, shown as
    /// `name` with `arg`.
    pub(crate) struct Part {
        pub(crate) first: usize,
        pub(crate) end: usize,
        pub(crate) name: &'static str,
        pub(crate) arg: (&'static str, u64),
    }

    /// What [`read`] needs of a stamped submission.
    pub(crate) struct Plan {
        parts: Vec<(&'static str, (&'static str, u64))>,
        submission: u64,
    }

    pub(crate) struct Stamps {
        heap: ID3D12QueryHeap,
        back: ID3D12Resource,
        /// Backs every list below; none is ever reset.
        allocator: ID3D12CommandAllocator,
        /// Closed lists that each end one timestamp query, by index.
        marks: Vec<ID3D12GraphicsCommandList>,
        /// Closed lists that resolve the first `n` queries, by `n`.
        resolves: Vec<Option<ID3D12GraphicsCommandList>>,
        /// Card ticks per second, and a pair of card and CPU ticks taken at
        /// the same moment.
        frequency: u64,
        calibration: (u64, u64),
        calibrated: std::time::Instant,
    }

    // SAFETY: D3D12 objects (free-threaded), used behind `STAMPS`' lock.
    unsafe impl Send for Stamps {}

    /// Made at the first stamped submission; `Some(None)` if that failed.
    static STAMPS: Mutex<Option<Option<Stamps>>> = Mutex::new(None);

    impl Stamps {
        pub(crate) fn new(
            device: &ID3D12Device,
            queue: &ID3D12CommandQueue,
        ) -> windows::core::Result<Stamps> {
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let readback = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(MOST) * 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            // SAFETY: plain object creation and queries.
            unsafe {
                device.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: MOST,
                        NodeMask: 0,
                    },
                    &mut heap,
                )?;
                device.CreateCommittedResource(
                    &readback,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut back,
                )?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let frequency = queue.GetTimestampFrequency()?;
                let mut calibration = (0, 0);
                queue.GetClockCalibration(&mut calibration.0, &mut calibration.1)?;
                Ok(Stamps {
                    heap: heap.ok_or_else(windows::core::Error::empty)?,
                    back: back.ok_or_else(windows::core::Error::empty)?,
                    allocator,
                    marks: Vec::new(),
                    resolves: vec![None; MOST as usize + 1],
                    frequency: frequency.max(1),
                    calibration,
                    calibrated: std::time::Instant::now(),
                })
            }
        }

        /// A closed list of `record`'s commands, on the shared allocator.
        fn closed(
            &self,
            device: &ID3D12Device,
            record: impl FnOnce(&ID3D12GraphicsCommandList),
        ) -> windows::core::Result<ID3D12GraphicsCommandList> {
            // SAFETY: a new list on our allocator, which records nothing
            // else meanwhile (the caller holds `STAMPS`' lock).
            unsafe {
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &self.allocator,
                    None,
                )?;
                record(&list);
                list.Close()?;
                Ok(list)
            }
        }

        fn mark(
            &mut self,
            device: &ID3D12Device,
            index: u32,
        ) -> windows::core::Result<ID3D12CommandList> {
            while self.marks.len() <= index as usize {
                let at = self.marks.len() as u32;
                // SAFETY: an open list and our heap.
                let list = self.closed(device, |list| unsafe {
                    list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, at)
                })?;
                self.marks.push(list);
            }
            self.marks[index as usize].cast()
        }

        fn resolve(
            &mut self,
            device: &ID3D12Device,
            count: u32,
        ) -> windows::core::Result<ID3D12CommandList> {
            if self.resolves[count as usize].is_none() {
                // SAFETY: an open list, our heap and read-back buffer.
                let list = self.closed(device, |list| unsafe {
                    list.ResolveQueryData(
                        &self.heap,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        count,
                        &self.back,
                        0,
                    )
                })?;
                self.resolves[count as usize] = Some(list);
            }
            match &self.resolves[count as usize] {
                Some(list) => list.cast(),
                None => Err(windows::core::Error::empty()),
            }
        }

        /// `lists` with a timestamp before and after each part and a resolve
        /// at the end; `None` if there are too many parts.
        pub(crate) fn around(
            &mut self,
            device: &ID3D12Device,
            lists: &[Option<ID3D12CommandList>],
            parts: &[Part],
            submission: u64,
        ) -> windows::core::Result<Option<(Vec<Option<ID3D12CommandList>>, Plan)>> {
            let count = parts.len() as u32 * 2;
            if count == 0 || count > MOST {
                return Ok(None);
            }
            let mut out = Vec::with_capacity(lists.len() + count as usize + 1);
            let mut next = 0;
            for (k, part) in parts.iter().enumerate() {
                let (first, end) = (
                    part.first.clamp(next, lists.len()),
                    part.end.min(lists.len()),
                );
                out.extend_from_slice(&lists[next..first]);
                out.push(Some(self.mark(device, 2 * k as u32)?));
                out.extend_from_slice(&lists[first..end.max(first)]);
                out.push(Some(self.mark(device, 2 * k as u32 + 1)?));
                next = end.max(first);
            }
            out.extend_from_slice(&lists[next..]);
            out.push(Some(self.resolve(device, count)?));
            let plan = Plan {
                parts: parts.iter().map(|p| (p.name, p.arg)).collect(),
                submission,
            };
            Ok(Some((out, plan)))
        }

        /// Card ticks on the CPU clock (`QueryPerformanceCounter`).
        fn to_cpu(&self, gpu: u64) -> u64 {
            let (gpu0, cpu0) = self.calibration;
            let delta = i128::from(gpu) - i128::from(gpu0);
            let cpu = i128::from(cpu0)
                + delta * i128::from(super::frequency()) / i128::from(self.frequency);
            cpu.max(1) as u64
        }

        /// After the stamped submission ran: the card's events, a part each
        /// and one for the whole submission.
        pub(crate) fn read(&mut self, queue: &ID3D12CommandQueue, plan: Plan) -> Vec<Event> {
            let count = plan.parts.len() * 2;
            let mut ticks = vec![0u64; count];
            // SAFETY: the read-back buffer, which the resolve (it has run)
            // wrote; `count` values fit in it.
            unsafe {
                let range = D3D12_RANGE {
                    Begin: 0,
                    End: count * 8,
                };
                let mut data = std::ptr::null_mut();
                if self.back.Map(0, Some(&range), Some(&mut data)).is_err() || data.is_null() {
                    return Vec::new();
                }
                std::ptr::copy_nonoverlapping(data.cast::<u64>(), ticks.as_mut_ptr(), count);
                self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            }
            // The two clocks drift apart slowly: matched again every second.
            if self.calibrated.elapsed() > std::time::Duration::from_secs(1) {
                let mut calibration = (0, 0);
                // SAFETY: a plain query.
                if unsafe { queue.GetClockCalibration(&mut calibration.0, &mut calibration.1) }
                    .is_ok()
                {
                    self.calibration = calibration;
                    self.calibrated = std::time::Instant::now();
                }
            }
            let mut events = Vec::with_capacity(plan.parts.len() + 1);
            let mut whole: Option<(u64, u64)> = None;
            for (k, (name, arg)) in plan.parts.iter().enumerate() {
                let (begin, end) = (ticks[2 * k], ticks[2 * k + 1]);
                if begin == 0 || end < begin {
                    continue;
                }
                let (begin, end) = (self.to_cpu(begin), self.to_cpu(end));
                whole = Some(whole.map_or((begin, end), |(a, b)| (a.min(begin), b.max(end))));
                events.push(Event::new(
                    name,
                    begin,
                    end,
                    Track::Gpu,
                    &[("submission", plan.submission), *arg],
                ));
            }
            if let Some((begin, end)) = whole {
                events.push(Event::new(
                    "gpu submission",
                    begin,
                    end,
                    Track::Gpu,
                    &[
                        ("submission", plan.submission),
                        ("parts", plan.parts.len() as u64),
                    ],
                ));
            }
            events
        }
    }

    /// [`Stamps::around`] on the shared stamps, made at the first call;
    /// `None` when they could not be made or the parts are too many.
    pub(crate) fn around(
        device: &ID3D12Device,
        queue: &ID3D12CommandQueue,
        lists: &[Option<ID3D12CommandList>],
        parts: &[Part],
        submission: u64,
    ) -> Option<(Vec<Option<ID3D12CommandList>>, Plan)> {
        let mut stamps = STAMPS.lock().unwrap_or_else(|e| e.into_inner());
        let stamps = stamps
            .get_or_insert_with(|| match Stamps::new(device, queue) {
                Ok(stamps) => Some(stamps),
                Err(error) => {
                    tracing::warn!(%error, "no GPU timestamps for the profile");
                    None
                }
            })
            .as_mut()?;
        match stamps.around(device, lists, parts, submission) {
            Ok(planned) => planned,
            Err(error) => {
                tracing::warn!(%error, "a submission's GPU timestamps not taken");
                None
            }
        }
    }

    /// After the submission [`around`] planned ran: its events go to the
    /// trace.
    pub(crate) fn done(queue: &ID3D12CommandQueue, plan: Plan) {
        let events = match STAMPS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            Some(Some(stamps)) => stamps.read(queue, plan),
            _ => return,
        };
        super::push_all(events);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Just enough of a JSON parser to tell valid from not.
    struct Json<'a> {
        bytes: &'a [u8],
        at: usize,
    }

    impl Json<'_> {
        fn space(&mut self) {
            while self.at < self.bytes.len() && b" \t\r\n".contains(&self.bytes[self.at]) {
                self.at += 1;
            }
        }
        fn eat(&mut self, byte: u8) -> bool {
            self.space();
            if self.bytes.get(self.at) == Some(&byte) {
                self.at += 1;
                true
            } else {
                false
            }
        }
        fn value(&mut self) -> Result<(), usize> {
            self.space();
            match self.bytes.get(self.at).copied() {
                Some(b'{') => {
                    self.at += 1;
                    if self.eat(b'}') {
                        return Ok(());
                    }
                    loop {
                        self.space();
                        self.string()?;
                        if !self.eat(b':') {
                            return Err(self.at);
                        }
                        self.value()?;
                        if self.eat(b'}') {
                            return Ok(());
                        }
                        if !self.eat(b',') {
                            return Err(self.at);
                        }
                    }
                }
                Some(b'[') => {
                    self.at += 1;
                    if self.eat(b']') {
                        return Ok(());
                    }
                    loop {
                        self.value()?;
                        if self.eat(b']') {
                            return Ok(());
                        }
                        if !self.eat(b',') {
                            return Err(self.at);
                        }
                    }
                }
                Some(b'"') => self.string(),
                Some(b'-' | b'0'..=b'9') => {
                    let start = self.at;
                    while self.at < self.bytes.len()
                        && b"+-.eE0123456789".contains(&self.bytes[self.at])
                    {
                        self.at += 1;
                    }
                    let text =
                        std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| start)?;
                    text.parse::<f64>().map(|_| ()).map_err(|_| start)
                }
                Some(b't') if self.bytes[self.at..].starts_with(b"true") => {
                    self.at += 4;
                    Ok(())
                }
                Some(b'f') if self.bytes[self.at..].starts_with(b"false") => {
                    self.at += 5;
                    Ok(())
                }
                Some(b'n') if self.bytes[self.at..].starts_with(b"null") => {
                    self.at += 4;
                    Ok(())
                }
                _ => Err(self.at),
            }
        }
        fn string(&mut self) -> Result<(), usize> {
            if self.bytes.get(self.at) != Some(&b'"') {
                return Err(self.at);
            }
            self.at += 1;
            while let Some(&b) = self.bytes.get(self.at) {
                self.at += 1;
                match b {
                    b'"' => return Ok(()),
                    b'\\' => {
                        let escaped = *self.bytes.get(self.at).ok_or(self.at)?;
                        self.at += if escaped == b'u' { 5 } else { 1 };
                    }
                    0..0x20 => return Err(self.at),
                    _ => {}
                }
            }
            Err(self.at)
        }
    }

    fn valid(text: &str) -> Result<(), usize> {
        let mut json = Json {
            bytes: text.as_bytes(),
            at: 0,
        };
        json.value()?;
        json.space();
        if json.at == text.len() {
            Ok(())
        } else {
            Err(json.at)
        }
    }

    #[test]
    fn the_parser_tells_valid_from_not() {
        assert!(valid(r#"{"a":[1,-2.5e3,"x\"y",true,null,{}]}"#).is_ok());
        assert!(valid(r#"{"a":[1,]}"#).is_err());
        assert!(valid(r#"{"a":1"#).is_err());
        assert!(valid("[\"\u{1}\"]").is_err());
    }

    #[test]
    fn the_writer_leaves_valid_json_after_every_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.json");
        let base = 1_000_000;
        let mut writer = Writer::create(&path, base, 10_000_000).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(valid(&text), Ok(()), "{text}");
        let span = |name, start, end, track| {
            Event::new(name, start, end, track, &[("lists", 3), ("late", 0)])
        };
        writer
            .write(&[
                (
                    7,
                    "victory \"main\"\\",
                    vec![span("submit", base + 10, base + 2_010, Track::Thread)],
                ),
                (9, "thread 9", Vec::new()),
            ])
            .unwrap();
        writer
            .write(&[
                (
                    7,
                    "victory \"main\"\\",
                    vec![span("gpu wait", base + 20, base + 1_020, Track::Thread)],
                ),
                (
                    9,
                    "thread 9",
                    vec![span("frame", base - 5, base + 4_000, Track::Frames)],
                ),
                (
                    9,
                    "thread 9",
                    vec![
                        Event::new("host list", base + 30, base + 40, Track::Gpu, &[]),
                        Event::new("title list", base, base + 50, Track::List(12), &[]),
                        Event::new("title list", base + 60, base + 70, Track::List(12), &[]),
                    ],
                ),
            ])
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(valid(&text), Ok(()), "{text}");
        assert!(text.ends_with(TRAILER));
        assert_eq!(writer.bytes(), text.len() as u64);
        // Complete events with times in microseconds (10 MHz ticks).
        assert_eq!(text.matches("\"ph\":\"X\"").count(), 6);
        assert!(text.contains(
            "{\"name\":\"submit\",\"ph\":\"X\",\"ts\":1.000,\"dur\":200.000,\"pid\":1,\"tid\":7,\
             \"args\":{\"lists\":3,\"late\":0}}"
        ));
        assert!(text.contains("\"ts\":-0.500,\"dur\":400.500,\"pid\":1,\"tid\":0"));
        assert!(text.contains(
            "\"name\":\"host list\",\"ph\":\"X\",\"ts\":3.000,\"dur\":1.000,\"pid\":2,\"tid\":1}"
        ));
        // Each thread named once, escaped; one with no spans of its own (only
        // frames, the card's and lists') not at all; each list's line once.
        assert_eq!(
            text.matches("\"tid\":7,\"args\":{\"name\":\"victory \\\"main\\\"\\\\\"}")
                .count(),
            1
        );
        assert_eq!(text.matches("\"args\":{\"name\":\"thread 9\"}").count(), 0);
        assert!(text.contains("\"ts\":6.000,\"dur\":1.000,\"pid\":3,\"tid\":12}"));
        assert_eq!(
            text.matches("\"pid\":3,\"tid\":12,\"args\":{\"name\":\"list 12\"}")
                .count(),
            1
        );
    }

    #[test]
    fn spans_record_nothing_while_off() {
        // Whatever the environment, recording starts only at a busy present
        // or after a delay, neither of which this test makes.
        assert!(!on());
        assert!(span("nothing").is_none());
        assert_eq!(stamp(), 0);
        list_recorded(1, 0);
        assert_eq!(elapsed(0), 0);
        since("nothing", 0, 0, &[]);
        // No part is timed, summed or not.
        assert_eq!(part_start(), 0);
        parts_begin();
        assert_eq!(part_start(), 0);
        part_end(0);
        assert_eq!(parts_end(), (0, 0));
        assert_eq!(parts_end(), (0, 0));
    }

    /// The parts timed on a thread are summed and counted from
    /// `parts_begin` to `parts_end`, which takes the sum: nothing is kept
    /// after it, nor with no span to sum them.
    #[test]
    fn parts_are_summed_from_begin_to_end() {
        // (Their starts from the clock itself: `part_start` gives one only
        // while recording.)
        assert!(elapsed(now()) < frequency(), "under a second");
        parts_begin();
        let first = now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        part_end(first);
        part_end(now());
        // One not timed (start 0) is not counted.
        part_end(0);
        let (ticks, count) = parts_end();
        assert_eq!(count, 2);
        assert!(micros(ticks) >= 2_000, "{ticks} ticks");
        assert_eq!(parts_end(), (0, 0), "taken by the first end");
        part_end(now());
        assert_eq!(parts_end(), (0, 0), "no span sums the parts");
        // A second span starts from nothing.
        parts_begin();
        part_end(now());
        assert_eq!(parts_end().1, 1);
    }

    #[test]
    fn micros_are_whole_microseconds_of_the_clock() {
        assert_eq!(micros(frequency()), 1_000_000);
        assert_eq!(micros(3 * frequency()), 3_000_000);
        assert_eq!(micros(0), 0);
    }

    #[test]
    fn an_event_keeps_six_args_and_drops_the_rest() {
        let args = [
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
        ];
        let event = Event::new("walk", 1, 2, Track::Thread, &args);
        assert_eq!(usize::from(event.args_len), MOST_ARGS);
        assert_eq!(event.args[..], args[..MOST_ARGS]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.json");
        let mut writer = Writer::create(&path, 0, 1_000_000).unwrap();
        writer.write(&[(3, "thread 3", vec![event])]).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(valid(&text), Ok(()), "{text}");
        assert!(text.contains("\"args\":{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"e\":5,\"f\":6}}"));
    }

    /// On the software adapter: stamps around two parts of a submission come
    /// back in order, on the CPU clock, around the moment it ran.
    #[cfg(windows)]
    #[test]
    fn gpu_stamps_bracket_each_part_on_warp() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter, IDXGIFactory4,
        };
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        let factory: IDXGIFactory4 =
            unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }.unwrap();
        let adapter: IDXGIAdapter = unsafe { factory.EnumWarpAdapter() }.unwrap();
        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.unwrap();
        let device = device.unwrap();
        let queue: ID3D12CommandQueue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
        }
        .unwrap();
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }.unwrap();
        let empty = |_: ()| -> ID3D12CommandList {
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            }
            .unwrap();
            unsafe { list.Close() }.unwrap();
            list.cast().unwrap()
        };
        let lists = vec![Some(empty(())), Some(empty(())), Some(empty(()))];
        let mut stamps = gpu::Stamps::new(&device, &queue).unwrap();
        let parts = [
            gpu::Part {
                first: 0,
                end: 2,
                name: "host list",
                arg: ("list", 1),
            },
            gpu::Part {
                first: 2,
                end: 3,
                name: "card compare",
                arg: ("list", 0),
            },
        ];
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let event = unsafe { CreateEventW(None, false, false, None) }.unwrap();
        for submission in 1..=2u64 {
            let (all, plan) = stamps
                .around(&device, &lists, &parts, submission)
                .unwrap()
                .unwrap();
            // Two marks a part and the resolve.
            assert_eq!(all.len(), lists.len() + 2 * parts.len() + 1);
            let before = now();
            unsafe {
                queue.ExecuteCommandLists(&all);
                queue.Signal(&fence, submission).unwrap();
                fence.SetEventOnCompletion(submission, event).unwrap();
                WaitForSingleObject(event, INFINITE);
            }
            let after = now();
            let events = stamps.read(&queue, plan);
            assert_eq!(events.len(), 3, "{events:?}");
            let names: Vec<&str> = events.iter().map(|e| e.name).collect();
            assert_eq!(names, ["host list", "card compare", "gpu submission"]);
            assert!(
                events
                    .iter()
                    .all(|e| e.track == Track::Gpu && e.start <= e.end)
            );
            assert!(events[0].end <= events[1].start);
            assert_eq!(events[2].args[0], ("submission", submission));
            // On the CPU clock: within the submission's moment, give or take
            // the calibration's own error (a millisecond).
            let slack = frequency() / 1000;
            assert!(
                events[2].start + slack >= before,
                "{} {before}",
                events[2].start
            );
            assert!(events[2].end <= after + slack, "{} {after}", events[2].end);
        }
    }
}
