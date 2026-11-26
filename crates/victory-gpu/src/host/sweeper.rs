//! The sweeper: a thread of its own that lets go of the host objects
//! nothing names any more (copies of windows of title memory, staging
//! buffers of textures, host copies of textures the title fills) and trims
//! the pipeline library's mapping, outside the submission lock.
//!
//! A pass starts at most once a second ([`PASS_MS`]) and goes on a step
//! every [`TICK`]: a step looks at no more than [`ENTRIES_PER_STEP`]
//! entries and lets go of no more than [`STAGING_BYTES_PER_STEP`] bytes of
//! staging, so the lock of each table (the one that hands its objects out)
//! is held only briefly while the title records and the host submits.
//!
//! Why the card cannot use what it lets go of, with no submission lock and
//! with flights sent ahead (`VICTORY_RELAY`) in the air:
//! - an object goes only when nothing that could hand it to a list or a
//!   slot holds it: a list holds what it uses until it is reset (flights
//!   keep their lists until they land), a slot keeps a reference to what it
//!   views, and the table's lock is held from the look to the removal, so
//!   none is handed out in between and none after;
//! - what goes is retired by the newest submission started
//!   ([`super::retire`]): it is dropped only once every submission begun
//!   before it went has finished on the card, so one that touched it
//!   without holding it (a copy recorded while it prepared) has run. One
//!   begun after cannot name it.
//!
//! What this thread retires waits in its own queue ([`OWN`]) and is dropped
//! here, outside the tables' locks: the D3D12 releases stay off the thread
//! that submits.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Duration;

use super::{FINISHED, RETIRED, Retired};
use std::sync::atomic::Ordering;

/// How long the sweeper rests between steps.
const TICK: Duration = Duration::from_millis(50);
/// A pass starts at most this often, in milliseconds.
const PASS_MS: u64 = 1000;
/// The entries of the tables a step looks at.
const ENTRIES_PER_STEP: usize = 512;
/// The bytes of staging buffers a step lets go of (the oldest idle first).
const STAGING_BYTES_PER_STEP: u64 = 64 << 20;

thread_local! {
    /// Set on the sweeper's thread (and by tests that act as it).
    static ON_SWEEPER: Cell<bool> = const { Cell::new(false) };
}

/// What the sweeper retired ([`super::retire`]), by the submission started
/// when it went, oldest first; dropped by the sweeper once the card is done
/// with those ([`bury_own`]).
static OWN: Mutex<VecDeque<(u64, Retired)>> = Mutex::new(VecDeque::new());

/// Held by a step while it runs: [`stop`] waits for it.
static STEP: Mutex<()> = Mutex::new(());

/// Whether this thread is the sweeper's: what it retires waits in [`OWN`].
pub(super) fn on_sweeper() -> bool {
    ON_SWEEPER.with(Cell::get)
}

/// Keeps `retired` until submission `started` has finished ([`bury_own`]).
pub(super) fn hold(started: u64, retired: Retired) {
    OWN.lock()
        .unwrap_or_else(|e| e.into_inner())
        .push_back((started, retired));
}

/// Drops what the sweeper retired that the card is done with. Taken out
/// under the queue's lock, dropped without it.
fn bury_own() {
    let finished = FINISHED.load(Ordering::Acquire);
    let done: Vec<(u64, Retired)> = {
        let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
        let n = own.iter().take_while(|(at, _)| *at <= finished).count();
        own.drain(..n).collect()
    };
    drop(done);
}

/// Starts the sweeper's thread, once (at the first submission). Not under
/// test: a sweeper beside a test would let go of, and bury, another test's
/// objects; tests run its steps themselves.
pub(super) fn start() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        if cfg!(test) {
            return;
        }
        let spawned = std::thread::Builder::new()
            .name("victory-sweep".into())
            .spawn(run);
        if spawned.is_err() {
            tracing::warn!("no thread for the sweep of idle host objects: nothing is let go");
        }
    });
}

/// The sweeper's thread: a step every [`TICK`] until the title's memory
/// is released ([`super::ending`]).
fn run() {
    // Below the title's own threads (loading, recording).
    // SAFETY: a plain call on our own thread.
    unsafe {
        use windows::Win32::System::Threading::*;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    ON_SWEEPER.with(|on| on.set(true));
    let mut sweeper = Sweeper::default();
    let mut parts = LiveParts::default();
    loop {
        std::thread::sleep(TICK);
        let _step = STEP.lock().unwrap_or_else(|e| e.into_inner());
        if super::ending() {
            return;
        }
        sweeper.step(super::now_ms(), &mut parts);
    }
}

/// At the release of the title's memory, after no submission may start
/// ([`super::end_submissions`]): a step under way ends first, none comes
/// after it, and what the sweeper still holds goes to the host's retired
/// objects, which the release drops once the card is done
/// ([`super::bury_all`]).
pub(super) fn stop() {
    let _step = STEP.lock().unwrap_or_else(|e| e.into_inner());
    let own = std::mem::take(&mut *OWN.lock().unwrap_or_else(|e| e.into_inner()));
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend(own);
}

/// How much a step may still do: entries to look at and bytes of staging
/// to let go of. A step gets on by one entry at least, whatever is left.
pub(crate) struct Budget {
    entries: usize,
    bytes: u64,
}

impl Budget {
    /// A budget of `entries` entries to look at and `bytes` bytes of
    /// staging to let go of.
    pub(crate) fn new(entries: usize, bytes: u64) -> Self {
        Self { entries, bytes }
    }

    /// A step's budget.
    fn step() -> Self {
        Self::new(ENTRIES_PER_STEP, STAGING_BYTES_PER_STEP)
    }

    /// `n` entries looked at.
    pub(crate) fn looked(&mut self, n: usize) {
        self.entries = self.entries.saturating_sub(n);
    }

    /// `bytes` let go of.
    pub(crate) fn let_go(&mut self, bytes: u64) {
        self.bytes = self.bytes.saturating_sub(bytes);
    }

    /// Whether no entry is left to look at.
    pub(crate) fn spent(&self) -> bool {
        self.entries == 0
    }

    /// Whether no byte is left to let go of.
    pub(crate) fn bytes_spent(&self) -> bool {
        self.bytes == 0
    }
}

/// Calls `look` on `entries`, keys and values in order, while `budget`
/// lasts, one at least: the key of the last one looked at if the budget ran
/// out before the end. A walk of the sweep goes on after that key at its
/// next step.
pub(crate) fn sweep_walk<K: Copy, V>(
    entries: impl Iterator<Item = (K, V)>,
    budget: &mut Budget,
    mut look: impl FnMut(K, V),
) -> Option<K> {
    let mut entries = entries.peekable();
    while let Some((key, value)) = entries.next() {
        budget.looked(1);
        look(key, value);
        if budget.spent() && entries.peek().is_some() {
            return Some(key);
        }
    }
    None
}

/// The sweeper's state between steps: the pass under way, and when the
/// last one began.
#[derive(Default)]
struct Sweeper {
    pass: Option<SweepPart>,
    began: Option<u64>,
}

impl Sweeper {
    /// One step at `now` (milliseconds, [`super::now_ms`]): what it retired
    /// and the card is done with is dropped, then a pass goes on, or begins
    /// if the last one began [`PASS_MS`] ago, by one budget.
    fn step(&mut self, now: u64, parts: &mut impl SweepParts) {
        bury_own();
        if self.pass.is_none() {
            if self.began.is_some_and(|began| now < began + PASS_MS) {
                return;
            }
            self.began = Some(now);
            self.pass = Some(SweepPart::default());
        }
        let mut budget = Budget::step();
        if let Some(part) = &mut self.pass
            && go_on(part, &mut budget, parts)
        {
            self.pass = None;
        }
        bury_own();
    }
}

/// The parts of a pass of the sweep, in order, each with where it goes on
/// from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SweepPart {
    /// The pipeline library's mapped file (`crate::pipeline::trim_library`)
    /// and the pool's empty blocks (`crate::texture_pool::room`).
    #[default]
    Library,
    /// Copies of windows of title memory, after the key given
    /// (`crate::shadow::let_go_idle`).
    Windows(Option<(u64, u64)>),
    /// Staging buffers of textures, the oldest idle first: at the place
    /// given in the pass's list of them (`crate::texture::let_go_staging`;
    /// none: the list is to be made, `crate::texture::idle_staging`).
    Staging(Option<usize>),
    /// Host copies of textures the title fills, after the address given
    /// (`crate::resource::let_go_idle`).
    Textures(Option<usize>),
    Done,
}

/// What the parts of a pass of the sweep do ([`SweepPart`]): those of the
/// layer ([`LiveParts`]), or a test's.
trait SweepParts {
    /// `crate::pipeline::trim_library` and `crate::texture_pool::room`.
    fn library(&mut self);
    /// `crate::shadow::let_go_idle`.
    fn windows(&mut self, from: Option<(u64, u64)>, budget: &mut Budget) -> Option<(u64, u64)>;
    /// `crate::texture::idle_staging` (none) or `crate::texture::let_go_staging`.
    fn staging(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize>;
    /// `crate::resource::let_go_idle`.
    fn textures(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize>;
}

/// Goes on with the pass at `part` while `budget` lasts: whether it is
/// done. Each call gets on by one step at least (an entry looked at, or a
/// part done).
fn go_on(part: &mut SweepPart, budget: &mut Budget, parts: &mut impl SweepParts) -> bool {
    let mut got_on = false;
    loop {
        if *part == SweepPart::Done {
            return true;
        }
        if got_on && budget.spent() {
            return false;
        }
        got_on = true;
        *part = match *part {
            SweepPart::Library => {
                parts.library();
                budget.looked(1);
                SweepPart::Windows(None)
            }
            SweepPart::Windows(from) => match parts.windows(from, budget) {
                Some(at) => SweepPart::Windows(Some(at)),
                None => SweepPart::Staging(None),
            },
            SweepPart::Staging(from) => match parts.staging(from, budget) {
                Some(at) => SweepPart::Staging(Some(at)),
                None => SweepPart::Textures(None),
            },
            SweepPart::Textures(from) => match parts.textures(from, budget) {
                Some(at) => SweepPart::Textures(Some(at)),
                None => SweepPart::Done,
            },
            SweepPart::Done => SweepPart::Done,
        };
    }
}

/// The parts of the sweep the layer runs, each a span on the profile (one
/// per step of it), with the pass's list of idle staging buffers.
#[derive(Default)]
struct LiveParts {
    staging: Vec<usize>,
}

impl SweepParts for LiveParts {
    fn library(&mut self) {
        let _part = crate::profile::span("sweep: library");
        crate::pipeline::trim_library();
        // The blocks of textures and staging kept empty, by the card's
        // room (nothing without `VICTORY_TEXTURE_POOL`).
        crate::texture_pool::room();
    }

    fn windows(&mut self, from: Option<(u64, u64)>, budget: &mut Budget) -> Option<(u64, u64)> {
        let _part = crate::profile::span("sweep: copies");
        crate::shadow::let_go_idle(from, budget)
    }

    fn staging(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize> {
        let _part = crate::profile::span("sweep: staging");
        let Some(at) = from else {
            self.staging = crate::texture::idle_staging(budget);
            return (!self.staging.is_empty()).then_some(0);
        };
        let at = at + crate::texture::let_go_staging(&self.staging[at..], budget);
        if at < self.staging.len() {
            return Some(at);
        }
        self.staging = Vec::new();
        None
    }

    fn textures(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize> {
        let _part = crate::profile::span("sweep: resources");
        crate::resource::let_go_idle(from, budget)
    }
}

/// Tests: runs `f` as the sweeper's thread would: what it retires waits in
/// [`OWN`] (dropped by [`bury_own_for_test`], not here).
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn as_sweeper<R>(f: impl FnOnce() -> R) -> R {
    ON_SWEEPER.with(|on| on.set(true));
    let r = f();
    ON_SWEEPER.with(|on| on.set(false));
    r
}

/// Tests that use the sweeper's own queue ([`OWN`]) hold this while they
/// run, so that two of them on parallel test threads never see each
/// other's objects. (Those that also start a submission with
/// `crate::host::Running` still need `--test-threads=1`, as every test
/// of `Running` does.)
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn own_queue_for_test() -> std::sync::MutexGuard<'static, ()> {
    static ONE: Mutex<()> = Mutex::new(());
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Tests: [`bury_own`], and how many the sweeper still holds.
#[cfg(all(test, feature = "diagnostics"))]
pub(crate) fn bury_own_for_test() -> usize {
    bury_own();
    OWN.lock().unwrap_or_else(|e| e.into_inner()).len()
}

#[cfg(test)]
mod tests {
    use super::{Budget, SweepPart, SweepParts, Sweeper, go_on, sweep_walk};

    /// A budget of `entries` entries and no byte limit.
    fn entries(entries: usize) -> Budget {
        Budget::new(entries, u64::MAX)
    }

    /// A walk on a budget of one entry looks at one entry a step, and going
    /// on after the key each stops at looks at every entry of the pass once,
    /// in order, as one walk with room for all does; a key put in before the
    /// place reached waits for the next pass, one taken out after it is not
    /// looked at.
    #[test]
    fn a_walk_in_steps_looks_at_what_a_whole_one_does() {
        use std::ops::Bound::{Excluded, Unbounded};
        let mut map: std::collections::BTreeMap<u64, u64> = (0..10).map(|k| (k * 10, k)).collect();
        let mut whole = Vec::new();
        assert_eq!(
            sweep_walk(map.iter(), &mut entries(usize::MAX), |&k, _| whole.push(k)),
            None
        );
        assert_eq!(whole, (0..10).map(|k| k * 10).collect::<Vec<_>>());
        let (mut stepped, mut from, mut steps) = (Vec::new(), None, 0);
        loop {
            steps += 1;
            let after = from.map_or(Unbounded, Excluded);
            let mut looked = 0;
            let stopped = sweep_walk(map.range((after, Unbounded)), &mut entries(1), |&k, _| {
                stepped.push(k);
                looked += 1;
            })
            .copied();
            assert_eq!(looked, 1, "one entry a step");
            if steps == 3 {
                // After 0, 10 and 20: one before, one ahead.
                map.insert(5, 0);
                map.remove(&50);
            }
            match stopped {
                Some(at) => from = Some(at),
                None => break,
            }
        }
        assert_eq!(stepped, [0, 10, 20, 30, 40, 60, 70, 80, 90]);
        assert_eq!(steps, 9, "no step is left over after the last entry");
        // A spent budget still looks at one.
        let mut spent = entries(0);
        let mut looked = Vec::new();
        assert_eq!(
            sweep_walk(map.iter(), &mut spent, |&k, _| looked.push(k)).copied(),
            Some(0)
        );
        assert_eq!(looked, [0]);
        let empty: std::collections::BTreeMap<u64, u64> = Default::default();
        assert_eq!(
            sweep_walk(empty.iter(), &mut entries(1), |_, _| panic!("nothing")),
            None
        );
    }

    /// Parts that look at one entry each per call of a walk `steps` long,
    /// and spend the step's budget: what they were called with, in order.
    struct Parts {
        steps: u64,
        calls: Vec<String>,
    }

    impl Parts {
        fn new(steps: u64) -> Self {
            Self {
                steps,
                calls: Vec::new(),
            }
        }

        /// Where a walk of `steps` entries stops after `from`, spending the
        /// budget unless it ends.
        fn walk(&self, from: Option<u64>, budget: &mut Budget) -> Option<u64> {
            let next = from.map_or(0, |at| at + 1);
            budget.looked(1);
            (budget.spent() && next + 1 < self.steps).then_some(next)
        }
    }

    impl SweepParts for Parts {
        fn library(&mut self) {
            self.calls.push("library".into());
        }

        fn windows(&mut self, from: Option<(u64, u64)>, budget: &mut Budget) -> Option<(u64, u64)> {
            self.calls.push(format!("windows {from:?}"));
            self.walk(from.map(|(at, _)| at), budget).map(|at| (at, 0))
        }

        fn staging(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize> {
            self.calls.push(format!("staging {from:?}"));
            self.walk(from.map(|at| at as u64), budget)
                .map(|at| at as usize)
        }

        fn textures(&mut self, from: Option<usize>, budget: &mut Budget) -> Option<usize> {
            self.calls.push(format!("textures {from:?}"));
            self.walk(from.map(|at| at as u64), budget)
                .map(|at| at as usize)
        }
    }

    /// A pass goes through its parts in order, on from where each step
    /// stopped, one entry a step on a budget of one, and in one call on a
    /// budget with room for all.
    #[test]
    fn a_pass_of_the_sweep_ends_in_steps_and_whole() {
        let mut parts = Parts::new(3);
        let mut part = SweepPart::default();
        let mut steps = 1;
        while !go_on(&mut part, &mut entries(1), &mut parts) {
            steps += 1;
            assert!(steps < 100, "a pass in steps ends");
        }
        assert_eq!(part, SweepPart::Done);
        let walked = |kind: &str, key: &dyn Fn(u64) -> String| {
            let mut calls = vec![format!("{kind} None")];
            calls.extend((0..2).map(|at| format!("{kind} Some({})", key(at))));
            calls
        };
        let mut expected = vec!["library".to_string()];
        expected.extend(walked("windows", &|at| format!("({at}, 0)")));
        expected.extend(walked("staging", &|at| at.to_string()));
        expected.extend(walked("textures", &|at| at.to_string()));
        assert_eq!(parts.calls, expected);
        // The library a step, then three entries a part, one a step.
        assert_eq!(steps, 10);

        let mut parts = Parts::new(3);
        let mut whole = SweepPart::default();
        assert!(
            go_on(&mut whole, &mut entries(usize::MAX), &mut parts),
            "a pass with room for all ends in one call"
        );
        assert_eq!(
            parts.calls,
            ["library", "windows None", "staging None", "textures None"]
        );
    }

    /// A pass begins at most once a second, by when the last one began,
    /// and goes on at every step until it is done.
    #[test]
    fn a_pass_begins_once_a_second() {
        let mut sweeper = Sweeper::default();
        let mut parts = Parts::new(1);
        sweeper.step(5_000, &mut parts);
        assert_eq!(
            sweeper.pass, None,
            "a pass of one entry a part ends in one step"
        );
        assert_eq!(parts.calls.len(), 4);
        sweeper.step(5_999, &mut parts);
        assert_eq!(parts.calls.len(), 4, "not before a second has gone");
        sweeper.step(6_000, &mut parts);
        assert_eq!(parts.calls.len(), 8);
    }

    /// What the sweeper retires waits in its own queue and is dropped by its
    /// next burial, not where it was let go: then at once if no submission
    /// was running, else once those running have finished.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn what_the_sweeper_retires_waits_for_the_running_submissions() {
        use super::super::{Running, retire};
        use super::{as_sweeper, bury_own_for_test, own_queue_for_test};
        use std::sync::Arc;
        let _one = own_queue_for_test();
        let idle: Arc<()> = Arc::default();
        assert_eq!(bury_own_for_test(), 0);
        as_sweeper(|| retire(idle.clone()));
        assert_eq!(
            Arc::strong_count(&idle),
            2,
            "not dropped where it was let go"
        );
        assert_eq!(
            bury_own_for_test(),
            0,
            "none ran: dropped at the next burial"
        );
        assert_eq!(Arc::strong_count(&idle), 1);
        let running = Running::start();
        as_sweeper(|| retire(idle.clone()));
        assert_eq!(bury_own_for_test(), 1, "a submission runs");
        assert_eq!(Arc::strong_count(&idle), 2);
        drop(running);
        assert_eq!(bury_own_for_test(), 0, "it finished");
        assert_eq!(Arc::strong_count(&idle), 1);
    }

    /// At the release what the sweeper holds goes to the host's retired
    /// objects, and still waits for the submission running.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn what_the_sweeper_holds_at_the_end_waits_for_the_release() {
        use super::super::{Running, retire};
        use super::{as_sweeper, bury_own_for_test, own_queue_for_test, stop};
        use std::sync::Arc;
        let _one = own_queue_for_test();
        let idle: Arc<()> = Arc::default();
        let running = Running::start();
        as_sweeper(|| retire(idle.clone()));
        stop();
        assert_eq!(bury_own_for_test(), 0, "handed over");
        assert_eq!(
            Arc::strong_count(&idle),
            2,
            "held by the host's retired objects"
        );
        drop(running);
        assert_eq!(Arc::strong_count(&idle), 1);
    }
}
