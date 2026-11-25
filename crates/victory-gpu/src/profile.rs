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
//! those sent ahead with `VICTORY_LATE_WAIT` or the relay only with
//! `VICTORY_PROFILE_FLIGHT_STAMPS=1`, [`flight_stamps`]), a timestamp before
//! and after each host list, resolved by a list at the end of the same
//! submission into a read-back slot of its own and read after the wait the
//! submission makes anyway, or when it lands; put on the CPU clock with
//! `GetClockCalibration`. They show as the process "GPU". Our other sends
//! (the card copies' uploads in `card_before`, their fetches in `card_back`)
//! are not stamped and show as gaps there.
//!
//! Counters ([`count`], [`record_changed`]; `"ph":"C"` events): what was
//! recorded or done between two presents, written at each present (global
//! UAV barriers by origin, the split of copy refreshes into kernel askings,
//! journal work and copies, host view slots changed and the distinct records
//! they were changed to). With `VICTORY_PROFILE_SAMPLES=1` ([`samples`]) the
//! writer thread adds each recording thread's processor time (the process
//! "thread cpu") and the memory split.
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
    /// A counter event (`"ph":"C"`) of the CPU process: sums since the last
    /// present ([`count`]) and the samples of [`samples`].
    Counter,
    /// The processor time of one thread (by id) over a sample's interval,
    /// on the process "thread cpu" ([`samples`]).
    ThreadCpu(u32),
}

/// The most named numbers an event carries.
const MOST_ARGS: usize = 10;

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
    /// A handle to the thread for its processor time ([`samples`]); 0 if
    /// none was opened. Kept for the process's life (a handle per thread
    /// that recorded).
    #[cfg(windows)]
    handle: usize,
}

static BUFFERS: Mutex<Vec<Arc<Buffer>>> = Mutex::new(Vec::new());

thread_local! {
    static LOCAL: std::cell::OnceCell<Arc<Buffer>> = const { std::cell::OnceCell::new() };
}

/// The calling thread's id, as the trace's `tid` names it.
#[cfg(windows)]
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

#[cfg(not(windows))]
pub(crate) fn thread_id() -> u32 {
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
        #[cfg(windows)]
        handle: if samples() {
            use windows_sys::Win32::System::Threading::{
                OpenThread, THREAD_QUERY_LIMITED_INFORMATION,
            };
            // SAFETY: a plain open of this thread for queries.
            unsafe { OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid) as usize }
        } else {
            0
        },
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

thread_local! {
    /// Host textures this thread made while recording ([`texture_made`]).
    static TEXTURES_MADE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The calling thread made a host texture; counted only while recording
/// (the caller holds a recording span), so that a span around something
/// that may make one can say whether it did ([`textures_made`]).
pub(crate) fn texture_made() {
    let _ = TEXTURES_MADE.try_with(|n| n.set(n.get() + 1));
}

/// Host textures the calling thread made while recording, so far.
pub(crate) fn textures_made() -> u64 {
    TEXTURES_MADE.try_with(std::cell::Cell::get).unwrap_or(0)
}

/// Whether submissions sent ahead (`VICTORY_LATE_WAIT`, the relay) are
/// stamped on the card too (`VICTORY_PROFILE_FLIGHT_STAMPS=1`; off by
/// default): the stamps go into the flight and are read when it lands
/// ([`gpu`]). It adds the stamps' small lists to every flight sent while
/// recording.
pub(crate) fn flight_stamps() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PROFILE_FLIGHT_STAMPS").is_some_and(|v| v == "1"))
}

/// Whether the writer thread samples while recording
/// (`VICTORY_PROFILE_SAMPLES=1`; off by default): every second the
/// processor time of each thread that recorded, and every two seconds the
/// memory split and the distinct views the view heaps' slots were made
/// from ([`sample_memory`]). The samples take the heaps' `made` locks for a
/// copy of their slots and ask Windows about the page pools' pages.
pub(crate) fn samples() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PROFILE_SAMPLES").is_some_and(|v| v == "1"))
}

/// What is counted while recording ([`count`]), summed between two
/// presents: each present writes the sums since the one before as counter
/// events, a group each ([`GROUPS`]). The sums themselves only grow, so a
/// span can take the part its own work added ([`refresh_split`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Count {
    /// Global UAV barriers recorded (waits for every earlier shader
    /// write), by origin: a title barrier naming a buffer; a title barrier
    /// on every resource; a title barrier group; the indirect clamp's wait
    /// before its dispatch and the one after; the compare and the fetch of
    /// copies in the card's memory.
    UavTitleBuffer,
    UavTitleGlobal,
    UavTitleGroup,
    UavClampFirst,
    UavClampSecond,
    UavCard,
    /// Indirect argument clamps recorded, and of them those that open a
    /// batch of mesh draws.
    Clamps,
    MeshBatches,
    /// Refreshes of host copies and texture staging ([`refresh_scope`];
    /// thread time, summed over the threads that refresh): clock ticks in
    /// the kernel's write-watch askings and the bytes they asked about, clock
    /// ticks in the journal's own work (its lock, its map, the copies'
    /// page stamps), clock ticks copying title memory (`ReadProcessMemory`),
    /// and the 4 KiB pages the askings found written with the distinct
    /// 64 KiB pages they lie in.
    AskTicks,
    AskBytes,
    JournalTicks,
    CopyTicks,
    Pages4k,
    Pages64k,
    /// Host view slots whose record changed, by who mirrored it (the
    /// `descriptor::host::hold` sites view, table, walk, sync and upload).
    SlotsView,
    SlotsTable,
    SlotsWalk,
    SlotsSync,
    SlotsUpload,
}

const COUNTS_N: usize = Count::SlotsUpload as usize + 1;
static COUNTS: [AtomicU64; COUNTS_N] = [const { AtomicU64::new(0) }; COUNTS_N];

/// How a counter's sum is written.
#[derive(Clone, Copy)]
enum Unit {
    Plain,
    /// Clock ticks, as microseconds.
    Micros,
    /// Bytes, as KiB.
    Kib,
    /// Times this (pages, as KiB).
    Times(u64),
}

/// The counter events written at each present: a name and its sums.
const GROUPS: &[(&str, &[(&str, Count, Unit)])] = &[
    (
        "uav barriers",
        &[
            ("title_buffer", Count::UavTitleBuffer, Unit::Plain),
            ("title_global", Count::UavTitleGlobal, Unit::Plain),
            ("title_group", Count::UavTitleGroup, Unit::Plain),
            ("clamp_first", Count::UavClampFirst, Unit::Plain),
            ("clamp_second", Count::UavClampSecond, Unit::Plain),
            ("card", Count::UavCard, Unit::Plain),
            ("clamps", Count::Clamps, Unit::Plain),
            ("mesh_batches", Count::MeshBatches, Unit::Plain),
        ],
    ),
    (
        "refresh split",
        &[
            ("kernel_us", Count::AskTicks, Unit::Micros),
            ("kernel_kb", Count::AskBytes, Unit::Kib),
            ("journal_us", Count::JournalTicks, Unit::Micros),
            ("copy_us", Count::CopyTicks, Unit::Micros),
            ("written_4k_kb", Count::Pages4k, Unit::Times(4)),
            ("written_64k_kb", Count::Pages64k, Unit::Times(64)),
        ],
    ),
];

/// Adds `n` to `what` while recording.
#[inline]
pub(crate) fn count(what: Count, n: u64) {
    if on() {
        COUNTS[what as usize].fetch_add(n, Ordering::Relaxed);
    }
}

/// `sum` written in `unit`.
fn in_unit(sum: u64, unit: Unit) -> u64 {
    match unit {
        Unit::Plain => sum,
        Unit::Micros => micros(sum),
        Unit::Kib => sum >> 10,
        Unit::Times(n) => sum.saturating_mul(n),
    }
}

/// Who mirrors a view slot, as [`record_changed`] keeps them: the
/// `descriptor::host::hold` sites 1 to 5, with their counts of slots.
const KEY_SITES: usize = 5;
const SITE_SLOTS: [Count; KEY_SITES] = [
    Count::SlotsView,
    Count::SlotsTable,
    Count::SlotsWalk,
    Count::SlotsSync,
    Count::SlotsUpload,
];
const KEY_SHARDS: usize = 16;

/// One shard of [`KEYS`]; `None` until its first key.
type KeyShard = Mutex<Option<std::collections::HashSet<u64>>>;

/// The distinct records (by view key) host view slots were made from since
/// the last present, by site, in shards by key.
static KEYS: [[KeyShard; KEY_SHARDS]; KEY_SITES] =
    [const { [const { Mutex::new(None) }; KEY_SHARDS] }; KEY_SITES];

/// While recording: host view slot changed to the record `key` (a view's
/// generation and index), mirrored by `site` (`descriptor::host::hold`).
/// Counted, and the key kept for the count of distinct records changed to
/// in the frame.
#[inline]
pub(crate) fn record_changed(site: u64, key: u64) {
    if !on() {
        return;
    }
    let Some(s) = (site as usize).checked_sub(1).filter(|&s| s < KEY_SITES) else {
        return;
    };
    COUNTS[SITE_SLOTS[s] as usize].fetch_add(1, Ordering::Relaxed);
    let shard = (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60) as usize % KEY_SHARDS;
    KEYS[s][shard]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(Default::default)
        .insert(key);
}

/// The distinct keys [`record_changed`] kept per site since the last
/// call, which empties them.
fn keys_taken() -> [u64; KEY_SITES] {
    let mut out = [0u64; KEY_SITES];
    for (site, shards) in KEYS.iter().enumerate() {
        for shard in shards {
            if let Some(set) = shard.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                out[site] += set.len() as u64;
                set.clear();
            }
        }
    }
    out
}

/// The sums at the last present, against which the next writes its own.
static LAST_COUNTS: Mutex<[u64; COUNTS_N]> = Mutex::new([0; COUNTS_N]);

/// At a present while recording: the counter events of the frame that
/// ends `at`, the sums since the last present ([`GROUPS`], and the record
/// keys), as one event per group with anything in it.
fn push_counts(at: u64) {
    let now: [u64; COUNTS_N] = std::array::from_fn(|i| COUNTS[i].load(Ordering::Relaxed));
    let since: [u64; COUNTS_N] = {
        let mut last = LAST_COUNTS.lock().unwrap_or_else(|e| e.into_inner());
        let since = std::array::from_fn(|i| now[i].saturating_sub(last[i]));
        *last = now;
        since
    };
    for &(name, sums) in GROUPS {
        if sums.iter().all(|&(_, what, _)| since[what as usize] == 0) {
            continue;
        }
        let args: Vec<(&'static str, u64)> = sums
            .iter()
            .map(|&(key, what, unit)| (key, in_unit(since[what as usize], unit)))
            .collect();
        push(Event::new(name, at, at, Track::Counter, &args));
    }
    let keys = keys_taken();
    let slots = |s: usize| since[SITE_SLOTS[s] as usize];
    if (0..KEY_SITES).any(|s| slots(s) > 0) {
        const NAMES: [(&str, &str); KEY_SITES] = [
            ("view_slots", "view_keys"),
            ("table_slots", "table_keys"),
            ("walk_slots", "walk_keys"),
            ("sync_slots", "sync_keys"),
            ("upload_slots", "upload_keys"),
        ];
        let args: Vec<(&'static str, u64)> = NAMES
            .iter()
            .enumerate()
            .flat_map(|(s, &(a, b))| [(a, slots(s)), (b, keys[s])])
            .collect();
        push(Event::new("record keys", at, at, Track::Counter, &args));
    }
}

thread_local! {
    /// [`refresh_scope`]: whether this thread refreshes host copies now;
    /// the clock ticks it spent asking the kernel and copying so far (the
    /// journal's share is what its timers saw less these); whether a
    /// journal timer runs ([`journal_timer`]).
    static REFRESH: std::cell::Cell<(bool, u64, u64, bool)> =
        const { std::cell::Cell::new((false, 0, 0, false)) };
}

/// While recording, the calling thread refreshes host copies (or texture
/// staging) until the result is dropped: the kernel askings, the journal's
/// work and the copies it does meanwhile are counted ([`Count::AskTicks`]
/// and on). `None` (and nothing counted) while the profiler does not
/// record.
pub(crate) fn refresh_scope() -> Option<RefreshScope> {
    if !on() {
        return None;
    }
    let was = REFRESH
        .try_with(|r| {
            let (was, kernel, copy, timing) = r.get();
            r.set((true, kernel, copy, timing));
            was
        })
        .ok()?;
    Some(RefreshScope { was })
}

/// See [`refresh_scope`].
#[must_use = "the scope ends when it is dropped"]
pub(crate) struct RefreshScope {
    was: bool,
}

impl Drop for RefreshScope {
    fn drop(&mut self) {
        let was = self.was;
        let _ = REFRESH.try_with(|r| {
            let (_, kernel, copy, timing) = r.get();
            r.set((was, kernel, copy, timing));
        });
    }
}

/// The start of a timed part of a refresh: 0, with no look at the clock,
/// unless the calling thread is in a [`refresh_scope`].
#[inline]
pub(crate) fn refresh_stamp() -> u64 {
    // (One load while the profiler is off: no look at the thread's state.)
    if STATE.load(Ordering::Relaxed) == ON && REFRESH.try_with(|r| r.get().0).unwrap_or(false) {
        now()
    } else {
        0
    }
}

/// A kernel write-watch asking begun at `start` ([`refresh_stamp`]; 0:
/// not counted) asked about `bytes` and found the 4 KiB pages at
/// `found` (in address order, as the kernel gives them).
pub(crate) fn kernel_asked(start: u64, bytes: u64, found: &[usize]) {
    if start == 0 {
        return;
    }
    let ticks = elapsed(start);
    let _ = REFRESH.try_with(|r| {
        let (on, kernel, copy, timing) = r.get();
        r.set((on, kernel + ticks, copy, timing));
    });
    let (pages, blocks) = pages_and_blocks(found);
    COUNTS[Count::AskTicks as usize].fetch_add(ticks, Ordering::Relaxed);
    COUNTS[Count::AskBytes as usize].fetch_add(bytes, Ordering::Relaxed);
    COUNTS[Count::Pages4k as usize].fetch_add(pages, Ordering::Relaxed);
    COUNTS[Count::Pages64k as usize].fetch_add(blocks, Ordering::Relaxed);
}

/// Of the addresses of written 4 KiB pages in address order: how many,
/// and how many distinct 64 KiB blocks they lie in.
fn pages_and_blocks(found: &[usize]) -> (u64, u64) {
    let mut blocks = 0u64;
    let mut last = None;
    for &at in found {
        let block = at >> 16;
        if last != Some(block) {
            blocks += 1;
            last = Some(block);
        }
    }
    (found.len() as u64, blocks)
}

/// A copy of title memory begun at `start` ([`refresh_stamp`]; 0: not
/// counted) has ended.
pub(crate) fn copied(start: u64) {
    if start == 0 {
        return;
    }
    let ticks = elapsed(start);
    let _ = REFRESH.try_with(|r| {
        let (on, kernel, copy, timing) = r.get();
        r.set((on, kernel, copy + ticks, timing));
    });
    COUNTS[Count::CopyTicks as usize].fetch_add(ticks, Ordering::Relaxed);
}

/// [`copied`] when dropped, of a copy begun now: nothing outside a
/// [`refresh_scope`].
pub(crate) fn copy_timer() -> CopyTimer {
    CopyTimer {
        start: refresh_stamp(),
    }
}

/// See [`copy_timer`].
#[must_use = "the timer ends when it is dropped"]
pub(crate) struct CopyTimer {
    start: u64,
}

impl Drop for CopyTimer {
    fn drop(&mut self) {
        copied(self.start);
    }
}

/// Times the journal's work from now until dropped, less the kernel
/// askings and copies made meanwhile ([`kernel_asked`], [`copied`]):
/// `None` outside a [`refresh_scope`] and inside another timer (an outer
/// one counts the inner's time).
pub(crate) fn journal_timer() -> Option<JournalTimer> {
    // (One load while the profiler is off: no look at the thread's state.)
    if STATE.load(Ordering::Relaxed) != ON {
        return None;
    }
    scoped_journal_timer()
}

/// [`journal_timer`] once the profiler is known to record: the thread's
/// own state decides.
fn scoped_journal_timer() -> Option<JournalTimer> {
    let (on, kernel, copy, timing) = REFRESH.try_with(std::cell::Cell::get).ok()?;
    if !on || timing {
        return None;
    }
    let _ = REFRESH.try_with(|r| r.set((on, kernel, copy, true)));
    Some(JournalTimer {
        start: now(),
        kernel,
        copy,
    })
}

/// See [`journal_timer`].
#[must_use = "the timer ends when it is dropped"]
pub(crate) struct JournalTimer {
    start: u64,
    kernel: u64,
    copy: u64,
}

impl Drop for JournalTimer {
    fn drop(&mut self) {
        let ticks = elapsed(self.start);
        let (kernel, copy) = REFRESH
            .try_with(|r| {
                let (on, kernel, copy, _) = r.get();
                r.set((on, kernel, copy, false));
                (kernel, copy)
            })
            .unwrap_or((self.kernel, self.copy));
        let others = kernel.saturating_sub(self.kernel) + copy.saturating_sub(self.copy);
        COUNTS[Count::JournalTicks as usize]
            .fetch_add(ticks.saturating_sub(others), Ordering::Relaxed);
    }
}

/// The refresh sums so far ([`Count::AskTicks`] to [`Count::Pages64k`]),
/// for [`refresh_args`].
pub(crate) fn refresh_split() -> [u64; 6] {
    std::array::from_fn(|i| COUNTS[Count::AskTicks as usize + i].load(Ordering::Relaxed))
}

/// Adds to `span` what the refreshes did since `before` ([`refresh_split`]):
/// `kernel_us`, `kernel_mb`, `journal_us`, `copy_us`, `kb_4k` and `kb_64k`
/// (the written pages found, at 4 KiB and at 64 KiB grain). Only the
/// calling thread's submission refreshes meanwhile (the submission lock),
/// but the times are thread time summed over the threads that refresh for
/// it: with `VICTORY_WIDE_REFRESH` the flush's workers refresh side by side
/// and the parts can add up to more than the span's wall time.
pub(crate) fn refresh_args(span: &mut Span, before: [u64; 6]) {
    let now = refresh_split();
    let d: [u64; 6] = std::array::from_fn(|i| now[i].saturating_sub(before[i]));
    span.arg("kernel_us", micros(d[0]));
    span.arg("kernel_mb", d[1] >> 20);
    span.arg("journal_us", micros(d[2]));
    span.arg("copy_us", micros(d[3]));
    span.arg("kb_4k", d[4] * 4);
    span.arg("kb_64k", d[5] * 64);
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
        ON if last != 0 => {
            push(Event::new(
                "frame",
                last,
                at,
                Track::Frames,
                &[("frame", frame), ("draws", draws)],
            ));
            push_counts(at);
        }
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
    let mut sampler = Sampler::default();
    let mut ticks = 0u64;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(100));
        ticks += 1;
        if samples() && ticks.is_multiple_of(10) {
            sampler.threads();
            if ticks.is_multiple_of(20) {
                sample_memory();
            }
        }
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

/// The writer thread's samples of the threads' processor time
/// ([`samples`]): per thread id, the time at the last sample (clock ticks)
/// and its processor time then (100 ns units).
#[derive(Default)]
struct Sampler {
    last: std::collections::HashMap<u32, (u64, u64)>,
}

impl Sampler {
    /// One event per thread that recorded, on the process "thread cpu":
    /// its processor time (kernel and user, `cpu_us`) from its last sample
    /// to now.
    #[cfg(windows)]
    fn threads(&mut self) {
        use windows_sys::Win32::Foundation::FILETIME;
        use windows_sys::Win32::System::Threading::GetThreadTimes;
        let buffers: Vec<Arc<Buffer>> = BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let mut events = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            if buffer.handle == 0 {
                continue;
            }
            let mut times = [FILETIME::default(); 4];
            let [created, exited, kernel, user] = &mut times;
            // SAFETY: a handle of ours, opened for queries; four times out.
            let ok =
                unsafe { GetThreadTimes(buffer.handle as _, created, exited, kernel, user) } != 0;
            if !ok {
                continue;
            }
            let time =
                |t: &FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
            let cpu = time(&times[2]) + time(&times[3]);
            let at = now();
            if let Some((then, was)) = self.last.insert(buffer.tid, (at, cpu))
                && cpu >= was
            {
                events.push(Event::new(
                    "cpu",
                    then,
                    at,
                    Track::ThreadCpu(buffer.tid),
                    &[("cpu_us", (cpu - was) / 10)],
                ));
            }
        }
        push_all(events);
    }

    #[cfg(not(windows))]
    fn threads(&mut self) {}
}

/// The memory split and the distinct views of the view heaps' slots, as
/// two counter events ([`samples`]): `memory` in MB (the page pools the
/// title registered and how much of them is resident, host copies of
/// title memory, of them the windows outside buffers, snapshot buffers,
/// copies in the card's memory (each with a CPU side as large), texture
/// staging buffers, upload memory of the lists) and `live keys` (distinct
/// views the made slots were made from, the made slots).
#[cfg(windows)]
fn sample_memory() {
    let at = now();
    let (pool, resident) = crate::resource::page_pools_resident();
    let [copies, windows, snapshots, card] = crate::shadow::memory_split();
    let staging = crate::texture::staging_bytes();
    let upload = crate::host::upload_bytes();
    let mb = |b: u64| b >> 20;
    push(Event::new(
        "memory",
        at,
        at,
        Track::Counter,
        &[
            ("pool_mb", mb(pool)),
            ("pool_resident_mb", mb(resident)),
            ("copies_mb", mb(copies)),
            ("windows_mb", mb(windows)),
            ("snapshots_mb", mb(snapshots)),
            ("card_mb", mb(card)),
            ("staging_mb", mb(staging)),
            ("upload_mb", mb(upload)),
        ],
    ));
    let (keys, slots) = crate::descriptor::live_keys();
    push(Event::new(
        "live keys",
        at,
        at,
        Track::Counter,
        &[("live_keys", keys), ("made_slots", slots)],
    ));
}

#[cfg(not(windows))]
fn sample_memory() {}

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
             {{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":4,\"tid\":0,\"args\":{{\"name\":\"thread cpu\"}}}},\n\
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
                    Track::Frames | Track::Counter => (1, 0),
                    Track::Gpu => (2, 1),
                    Track::List(list) => (3, list),
                    Track::ThreadCpu(thread) => (4, thread),
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
                if event.track == Track::Counter {
                    // (A counter has a moment, no length.)
                    let _ = write!(
                        chunk,
                        ",\"ph\":\"C\",\"ts\":{ts:.3},\"pid\":{pid},\"tid\":{tid}"
                    );
                } else {
                    let dur = event.end.saturating_sub(event.start) as f64 * 1e6 / self.frequency;
                    let _ = write!(
                        chunk,
                        ",\"ph\":\"X\",\"ts\":{ts:.3},\"dur\":{dur:.3},\"pid\":{pid},\"tid\":{tid}"
                    );
                }
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

/// The card's timestamps around the host lists of a submission: waited
/// ones, and with [`super::flight_stamps`] those sent ahead too, read when
/// they land.
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
    /// Submissions that may be stamped and not yet read at once (flights in
    /// the air), each with its own part of the read-back buffer.
    const SLOTS: u32 = 128;

    /// A part of a submission's lists to stamp: `lists[first..end]`, shown as
    /// `name` with `arg`.
    pub(crate) struct Part {
        pub(crate) first: usize,
        pub(crate) end: usize,
        pub(crate) name: &'static str,
        pub(crate) arg: (&'static str, u64),
    }

    /// What [`read`] needs of a stamped submission: its parts, its number
    /// and the read-back slot its stamps are resolved into (given back when
    /// it is read, [`done`], or forgotten, [`forget`]).
    pub(crate) struct Plan {
        parts: Vec<(&'static str, (&'static str, u64))>,
        submission: u64,
        slot: u32,
    }

    pub(crate) struct Stamps {
        /// Two banks of [`MOST`] queries: a slot uses bank `slot % 2`.
        heap: ID3D12QueryHeap,
        /// [`MOST`] values for each of the slots.
        back: ID3D12Resource,
        /// Backs every mark list; none is ever reset.
        allocator: ID3D12CommandAllocator,
        /// Closed lists that each end one timestamp query, by index, made
        /// when first needed.
        marks: Vec<Option<ID3D12GraphicsCommandList>>,
        /// Per slot, an allocator and a list that resolves the slot's
        /// queries into its part of `back`, recorded again for each
        /// submission stamped in the slot (the one before it has been read
        /// by then).
        resolves: Vec<Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
        /// Slots whose submission has not been read yet.
        busy: Vec<bool>,
        /// The slot to try first: slots are taken in turn, so submissions
        /// one after another use the two banks by turns.
        next: u32,
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
            Self::with_slots(device, queue, SLOTS)
        }

        /// [`Stamps::new`] with `slots` read-back slots.
        pub(crate) fn with_slots(
            device: &ID3D12Device,
            queue: &ID3D12CommandQueue,
            slots: u32,
        ) -> windows::core::Result<Stamps> {
            let slots = slots.max(1);
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let readback = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(slots) * u64::from(MOST) * 8,
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
                        Count: 2 * MOST,
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
                if let Some(back) = &back {
                    crate::host::label(back, "profile timestamp read-back", desc.Width);
                }
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
                    resolves: (0..slots).map(|_| None).collect(),
                    busy: vec![false; slots as usize],
                    next: 0,
                    frequency: frequency.max(1),
                    calibration,
                    calibrated: std::time::Instant::now(),
                })
            }
        }

        fn mark(
            &mut self,
            device: &ID3D12Device,
            index: u32,
        ) -> windows::core::Result<ID3D12CommandList> {
            if self.marks.len() <= index as usize {
                self.marks.resize(index as usize + 1, None);
            }
            if let Some(list) = &self.marks[index as usize] {
                return list.cast();
            }
            // SAFETY: a new list on our allocator, which records nothing
            // else meanwhile (the caller holds `STAMPS`' lock); our heap.
            let list = unsafe {
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &self.allocator,
                    None,
                )?;
                list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, index);
                list.Close()?;
                list
            };
            let cast = list.cast();
            self.marks[index as usize] = Some(list);
            cast
        }

        /// The list that resolves `count` queries of `slot`'s bank into the
        /// slot's part of the read-back buffer, recorded now.
        fn resolve(
            &mut self,
            device: &ID3D12Device,
            slot: u32,
            count: u32,
        ) -> windows::core::Result<ID3D12CommandList> {
            let held = &mut self.resolves[slot as usize];
            if held.is_none() {
                // SAFETY: plain object creation.
                unsafe {
                    let allocator: ID3D12CommandAllocator =
                        device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                    let list: ID3D12GraphicsCommandList = device.CreateCommandList(
                        0,
                        D3D12_COMMAND_LIST_TYPE_DIRECT,
                        &allocator,
                        None,
                    )?;
                    list.Close()?;
                    *held = Some((allocator, list));
                }
            }
            let Some((allocator, list)) = held else {
                return Err(windows::core::Error::empty());
            };
            // SAFETY: the slot was free, so the card has run its last
            // resolve (it was read after its submission ran); our heap and
            // read-back buffer, `count` queries of the slot's bank.
            unsafe {
                allocator.Reset()?;
                list.Reset(&*allocator, None)?;
                list.ResolveQueryData(
                    &self.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    bank(slot),
                    count,
                    &self.back,
                    u64::from(slot) * u64::from(MOST) * 8,
                );
                list.Close()?;
            }
            list.cast()
        }

        /// A free slot, taken; `None` if every slot's submission is still
        /// to be read.
        fn take_slot(&mut self) -> Option<u32> {
            let n = self.busy.len() as u32;
            let slot = (0..n)
                .map(|k| (self.next + k) % n)
                .find(|&s| !self.busy[s as usize])?;
            self.busy[slot as usize] = true;
            self.next = (slot + 1) % n;
            Some(slot)
        }

        /// `lists` with a timestamp before and after each part and a resolve
        /// at the end; `None` if there are too many parts or no slot is
        /// free.
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
            let Some(slot) = self.take_slot() else {
                return Ok(None);
            };
            match self.around_in(device, lists, parts, slot) {
                Ok(out) => {
                    let plan = Plan {
                        parts: parts.iter().map(|p| (p.name, p.arg)).collect(),
                        submission,
                        slot,
                    };
                    Ok(Some((out, plan)))
                }
                Err(error) => {
                    self.busy[slot as usize] = false;
                    Err(error)
                }
            }
        }

        fn around_in(
            &mut self,
            device: &ID3D12Device,
            lists: &[Option<ID3D12CommandList>],
            parts: &[Part],
            slot: u32,
        ) -> windows::core::Result<Vec<Option<ID3D12CommandList>>> {
            let count = parts.len() as u32 * 2;
            let base = bank(slot);
            let mut out = Vec::with_capacity(lists.len() + count as usize + 1);
            let mut next = 0;
            for (k, part) in parts.iter().enumerate() {
                let (first, end) = (
                    part.first.clamp(next, lists.len()),
                    part.end.min(lists.len()),
                );
                out.extend_from_slice(&lists[next..first]);
                out.push(Some(self.mark(device, base + 2 * k as u32)?));
                out.extend_from_slice(&lists[first..end.max(first)]);
                out.push(Some(self.mark(device, base + 2 * k as u32 + 1)?));
                next = end.max(first);
            }
            out.extend_from_slice(&lists[next..]);
            out.push(Some(self.resolve(device, slot, count)?));
            Ok(out)
        }

        /// Card ticks on the CPU clock (`QueryPerformanceCounter`).
        fn to_cpu(&self, gpu: u64) -> u64 {
            let (gpu0, cpu0) = self.calibration;
            let delta = i128::from(gpu) - i128::from(gpu0);
            let cpu = i128::from(cpu0)
                + delta * i128::from(super::frequency()) / i128::from(self.frequency);
            cpu.max(1) as u64
        }

        /// The slot of a plan whose submission will not be read is free
        /// (tests: a submission not waited for keeps its slot, which the
        /// card may still resolve into).
        #[cfg(test)]
        pub(crate) fn forget(&mut self, plan: Plan) {
            self.free(plan.slot);
        }

        fn free(&mut self, slot: u32) {
            if let Some(busy) = self.busy.get_mut(slot as usize) {
                *busy = false;
            }
        }

        /// After the stamped submission ran: the card's events, a part each
        /// and one for the whole submission. Its slot is free again.
        pub(crate) fn read(&mut self, queue: &ID3D12CommandQueue, plan: Plan) -> Vec<Event> {
            let count = plan.parts.len() * 2;
            let mut ticks = vec![0u64; count];
            let offset = plan.slot as usize * MOST as usize * 8;
            // SAFETY: the read-back buffer's slot, which the resolve (it has
            // run) wrote; `count` values fit in it.
            unsafe {
                let range = D3D12_RANGE {
                    Begin: offset,
                    End: offset + count * 8,
                };
                let mut data = std::ptr::null_mut();
                if self.back.Map(0, Some(&range), Some(&mut data)).is_ok() && !data.is_null() {
                    std::ptr::copy_nonoverlapping(
                        data.cast::<u8>().add(offset).cast::<u64>(),
                        ticks.as_mut_ptr(),
                        count,
                    );
                    self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
                } else {
                    ticks.clear();
                }
            }
            let Plan {
                parts,
                submission,
                slot,
            } = plan;
            self.free(slot);
            if ticks.is_empty() {
                return Vec::new();
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
            let mut events = Vec::with_capacity(parts.len() + 1);
            let mut whole: Option<(u64, u64)> = None;
            for (k, (name, arg)) in parts.iter().enumerate() {
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
                    &[("submission", submission), *arg],
                ));
            }
            if let Some((begin, end)) = whole {
                events.push(Event::new(
                    "gpu submission",
                    begin,
                    end,
                    Track::Gpu,
                    &[("submission", submission), ("parts", parts.len() as u64)],
                ));
            }
            events
        }
    }

    /// The first query of `slot`'s bank.
    fn bank(slot: u32) -> u32 {
        slot % 2 * MOST
    }

    /// [`Stamps::around`] on the shared stamps, made at the first call;
    /// `None` when they could not be made, the parts are too many or no
    /// slot is free.
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

    /// After the submission [`around`] planned ran (waited for, or landed
    /// with the relay): its events go to the trace.
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

    /// The count of host textures made is the calling thread's own.
    #[test]
    fn textures_made_are_counted_per_thread() {
        let before = textures_made();
        texture_made();
        texture_made();
        assert_eq!(textures_made() - before, 2);
        let other = std::thread::spawn(textures_made).join().unwrap();
        assert_eq!(other, 0);
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
    fn an_event_keeps_ten_args_and_drops_the_rest() {
        let args = [
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("d", 4),
            ("e", 5),
            ("f", 6),
            ("g", 7),
            ("h", 8),
            ("i", 9),
            ("j", 10),
            ("k", 11),
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
        assert!(text.contains(
            "\"args\":{\"a\":1,\"b\":2,\"c\":3,\"d\":4,\"e\":5,\"f\":6,\"g\":7,\"h\":8,\"i\":9,\
             \"j\":10}}"
        ));
    }

    /// Counter events have a moment and no length, on the CPU process;
    /// a thread's processor time goes to the process "thread cpu" under
    /// the thread's id, and names no line after the writing thread.
    #[test]
    fn counters_and_thread_samples_are_written_apart_from_spans() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile.json");
        let mut writer = Writer::create(&path, 0, 1_000_000).unwrap();
        writer
            .write(&[(
                5,
                "victory-profile",
                vec![
                    Event::new("uav barriers", 30, 30, Track::Counter, &[("card", 2)]),
                    Event::new("cpu", 10, 40, Track::ThreadCpu(77), &[("cpu_us", 25)]),
                ],
            )])
            .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(valid(&text), Ok(()), "{text}");
        assert!(text.contains(
            "{\"name\":\"uav barriers\",\"ph\":\"C\",\"ts\":30.000,\"pid\":1,\"tid\":0,\
             \"args\":{\"card\":2}}"
        ));
        assert!(text.contains(
            "{\"name\":\"cpu\",\"ph\":\"X\",\"ts\":10.000,\"dur\":30.000,\"pid\":4,\"tid\":77,\
             \"args\":{\"cpu_us\":25}}"
        ));
        assert!(!text.contains("victory-profile"), "{text}");
    }

    /// The 4 KiB pages a write-watch asking found, and the 64 KiB blocks
    /// they lie in (the kernel gives them in address order).
    #[test]
    fn written_pages_are_counted_at_both_grains() {
        assert_eq!(pages_and_blocks(&[]), (0, 0));
        let found = [0x1_0000, 0x1_1000, 0x1_f000, 0x2_0000, 0x5_0000, 0x5_3000];
        assert_eq!(pages_and_blocks(&found), (6, 3));
        assert_eq!(pages_and_blocks(&[0x7_0000]), (1, 1));
    }

    /// Counts and the refresh split stay untouched while the profiler does
    /// not record: no scope, no stamp, no timer.
    #[test]
    fn counters_count_nothing_while_off() {
        assert!(!on());
        let before = COUNTS[Count::UavCard as usize].load(Ordering::Relaxed);
        count(Count::UavCard, 3);
        record_changed(4, 99);
        assert_eq!(
            COUNTS[Count::UavCard as usize].load(Ordering::Relaxed),
            before
        );
        assert!(refresh_scope().is_none());
        assert_eq!(refresh_stamp(), 0);
        assert!(journal_timer().is_none());
        let split = refresh_split();
        kernel_asked(0, 1 << 20, &[0x1000]);
        copied(0);
        assert_eq!(refresh_split(), split);
    }

    /// Inside a refresh scope a journal timer counts its time less the
    /// kernel askings and copies made under it; a timer inside another
    /// counts nothing of its own; out of the scope nothing is timed.
    #[test]
    fn the_journal_share_leaves_out_kernel_and_copy_time() {
        // (Set by hand, and the timer taken past the profiler's own check:
        // `refresh_scope` and `journal_timer` need the profiler recording.)
        REFRESH.with(|r| r.set((true, 0, 0, false)));
        // The profiler off: not even a scope's thread times its journal.
        assert!(journal_timer().is_none());
        let before = refresh_split();
        {
            let timer = scoped_journal_timer().expect("in a scope");
            assert!(scoped_journal_timer().is_none(), "one timer at a time");
            // (Starts from the clock itself: `refresh_stamp` gives one only
            // while recording.)
            let asking = now();
            std::thread::sleep(std::time::Duration::from_millis(3));
            kernel_asked(asking, 8 << 20, &[0x1_0000, 0x1_1000]);
            let copying = now();
            std::thread::sleep(std::time::Duration::from_millis(3));
            copied(copying);
            drop(timer);
        }
        let after = refresh_split();
        let d: Vec<u64> = (0..6).map(|i| after[i] - before[i]).collect();
        // (Other tests may count meanwhile on their threads: at least.)
        assert!(micros(d[0]) >= 3_000, "kernel {d:?}");
        assert!(d[1] >= 8 << 20);
        assert!(micros(d[3]) >= 3_000, "copy {d:?}");
        assert!(d[4] >= 2 && d[5] >= 1);
        // The journal's share: its whole time less those two, so well under
        // the 6 ms slept.
        let journal = micros(d[2]);
        assert!(journal < 3_000, "journal {journal} us");
        REFRESH.with(|r| r.set((false, 0, 0, false)));
        assert!(scoped_journal_timer().is_none());
    }

    /// A present writes what was counted since the one before, a counter
    /// event per group with anything in it, and the record keys' slots
    /// against the distinct keys, which start again from nothing.
    #[test]
    fn a_present_writes_the_counts_since_the_last() {
        let taken = |at: u64| -> Vec<Event> {
            LOCAL.with(|l| {
                let buffer = l.get_or_init(register);
                let mut events = buffer.events.lock().unwrap();
                let mine: Vec<Event> = events.iter().filter(|e| e.start == at).copied().collect();
                events.retain(|e| e.start != at);
                mine
            })
        };
        let arg = |e: &Event, key: &str| {
            e.args[..usize::from(e.args_len)]
                .iter()
                .find(|a| a.0 == key)
                .map(|a| a.1)
        };
        // Whatever came before is written first.
        push_counts(1);
        let _ = taken(1);
        let _ = keys_taken();
        COUNTS[Count::UavTitleGroup as usize].fetch_add(5, Ordering::Relaxed);
        COUNTS[Count::SlotsSync as usize].fetch_add(3, Ordering::Relaxed);
        for key in [11u64, 12, 11] {
            let shard = key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 60;
            KEYS[3][shard as usize % KEY_SHARDS]
                .lock()
                .unwrap()
                .get_or_insert_with(Default::default)
                .insert(key);
        }
        push_counts(2);
        let events = taken(2);
        let barriers = events
            .iter()
            .find(|e| e.name == "uav barriers")
            .expect("barriers");
        assert_eq!(barriers.track, Track::Counter);
        assert_eq!(arg(barriers, "title_group"), Some(5));
        assert_eq!(arg(barriers, "card"), Some(0));
        let keys = events
            .iter()
            .find(|e| e.name == "record keys")
            .expect("keys");
        assert_eq!(arg(keys, "sync_slots"), Some(3));
        assert_eq!(arg(keys, "sync_keys"), Some(2));
        assert_eq!(usize::from(keys.args_len), 2 * KEY_SITES);
        // Nothing new: nothing written.
        push_counts(3);
        assert!(taken(3).iter().all(|e| e.name != "record keys"));
        assert_eq!(keys_taken(), [0; KEY_SITES]);
    }

    /// A unit turns a sum into what its counter shows.
    #[test]
    fn counter_units_scale_their_sums() {
        assert_eq!(in_unit(7, Unit::Plain), 7);
        assert_eq!(in_unit(3 * frequency(), Unit::Micros), 3_000_000);
        assert_eq!(in_unit(5 << 10, Unit::Kib), 5);
        assert_eq!(in_unit(6, Unit::Times(64)), 384);
        // Every count is in at most one group, and no group holds more
        // than an event carries.
        let mut seen = std::collections::HashSet::new();
        for &(_, sums) in GROUPS {
            assert!(sums.len() <= MOST_ARGS);
            for &(_, what, _) in sums {
                assert!(seen.insert(what as usize), "{what:?} twice");
            }
        }
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

    /// On the software adapter: submissions stamped one after another and
    /// sent before any is read (flights in the air) each read back their own
    /// stamps, in the order they ran; with every slot taken a submission is
    /// not stamped, and a read or a forgotten plan frees its slot.
    #[cfg(windows)]
    #[test]
    fn flights_in_the_air_read_their_own_stamps_on_warp() {
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
        let empty = || -> Option<ID3D12CommandList> {
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            }
            .unwrap();
            unsafe { list.Close() }.unwrap();
            Some(list.cast().unwrap())
        };
        let lists = vec![empty(), empty()];
        let parts = |list: u64| {
            [0, 1].map(|i| gpu::Part {
                first: i,
                end: i + 1,
                name: "host list",
                arg: ("list", list + i as u64),
            })
        };
        let mut stamps = gpu::Stamps::with_slots(&device, &queue, 3).unwrap();
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let event = unsafe { CreateEventW(None, false, false, None) }.unwrap();
        // Three flights sent, none read: every slot is taken.
        let mut plans = Vec::new();
        for submission in 1..=3u64 {
            let (all, plan) = stamps
                .around(&device, &lists, &parts(10 * submission), submission)
                .unwrap()
                .unwrap();
            unsafe {
                queue.ExecuteCommandLists(&all);
                queue.Signal(&fence, submission).unwrap();
            }
            plans.push(plan);
        }
        assert!(
            stamps
                .around(&device, &lists, &parts(40), 4)
                .unwrap()
                .is_none(),
            "no slot free"
        );
        unsafe {
            fence.SetEventOnCompletion(3, event).unwrap();
            WaitForSingleObject(event, INFINITE);
        }
        let mut last_end = 0;
        let mut third = None;
        for (k, plan) in plans.into_iter().enumerate() {
            if k == 2 {
                third = Some(plan);
                continue;
            }
            let events = stamps.read(&queue, plan);
            assert_eq!(events.len(), 3, "{events:?}");
            let submission = k as u64 + 1;
            assert_eq!(events[0].args[1], ("list", 10 * submission));
            assert_eq!(events[1].args[1], ("list", 10 * submission + 1));
            assert_eq!(events[2].args[0], ("submission", submission));
            assert!(events[0].end <= events[1].start);
            assert!(events[2].start >= last_end, "flights ran in order");
            last_end = events[2].end;
        }
        // The two read slots are free again; the forgotten one too.
        stamps.forget(third.unwrap());
        for submission in 5..=7u64 {
            let planned = stamps
                .around(&device, &lists, &parts(0), submission)
                .unwrap();
            assert!(planned.is_some(), "slot for {submission}");
        }
    }
}
