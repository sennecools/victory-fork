//! R5 (`VICTORY_R5=1`): the memory core's homes under the relay, the parts
//! of [`super`] that only R5 uses. See [`r5`].

use super::*;

/// R5 (`VICTORY_R5=1`, off by default; only with the memory core on,
/// `VICTORY_MEMORY_CORE=1`, and the relay's second stage, `VICTORY_RELAY=2`,
/// [`relay_exact`]): the memory core's homes under the relay. Read once.
///
/// Without it the core gives no home while submissions fly
/// (`crate::home::Why::Late`): a home has no snapshot and no copy-back, and
/// what the core does after a submission assumed it was waited for. With it
/// the flights land that too, on the thread that lands them:
///
/// - at each landing, after the copies' copy-back and before the title is
///   told anything (read-backs, bottom-of-pipe writes, fence signals), the
///   copies over bytes the flights' shaders wrote in homes are told
///   (`crate::home::after_wait`, `crate::memory::after_wait`), as the
///   waited submission did after its wait;
/// - a refresh of a copy that is no home, over bytes of a home a flight's
///   shaders may write, waits until those flights have landed
///   ([`refresh_over_flight_homes`]): what it would take may be half
///   written, and nothing would tell it the rest; a card copy whose CPU side
///   is a home sends those bytes up from the home again, behind those
///   flights on the queue ([`retake_flight_homes`]);
/// - the directory of raw-address shaders changes in queue order with each
///   flight's own staging (`crate::memory::before_submission`, a [`Ring`]),
///   and card copies whose CPU side is a home each set their reference aside
///   with a list of their own flight's (`crate::card_home::before`) and land
///   with no fetch ([`Host::relay_card_back`]);
/// - a protection of the title's refused over a home waits for the flights
///   in the air before it is refused for good ([`wait_for_flights_now`]),
///   and frees that waited are looked at again after each landing
///   (`crate::home::after_landing`);
/// - a wait for a memory value, and a blocking wait for a fence, wait only
///   for the flights concerned ([`late_met`], [`settle_for_fence_at`]).
///
/// What it leaves as it was is in `crate::home`'s shortcuts.
///
/// Lock order on the landing thread: the line's lock is not held while it
/// lands; `CARD_WORK` and [`QUEUE_ORDER`] are taken and given back by the
/// card's part; the copies' tables, the watch journal and the directory's
/// lock by the telling, one after another. The core's lock (`crate::home`)
/// is not taken there: [`land`], on the threads that reset landed lists,
/// takes it through `crate::home::after_landing` after the frees' own lock
/// and before the copies' tables and the retire queue.
pub(crate) fn r5() -> bool {
    #[cfg(test)]
    if test_flights() {
        return test_r5();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_R5").is_some_and(|v| v == "1");
        let on = asked && crate::home::on() && relay_exact();
        if on {
            tracing::info!(
                "R5 on (VICTORY_R5=1): the memory core's homes under the relay; the flights land what homes need"
            );
        } else if asked {
            tracing::warn!(
                "VICTORY_R5=1 does nothing without VICTORY_MEMORY_CORE=1 and VICTORY_RELAY=2"
            );
        }
        on
    })
}

/// R5's S2 (`VICTORY_R5_LIFT=1`, off by default; only with [`r5`]): reads of
/// title memory that wait for flights are done on the card instead, behind
/// the flights on the queue, also where the bytes are a home's (`lift_mode`
/// in fills mode; the home's placed buffer is the source or destination), and
/// a copy's refresh over a home a flight writes takes those bytes on the card
/// (`late_lift::lift_refreshes`). Read once.
pub(crate) fn r5_lift() -> bool {
    #[cfg(test)]
    if test_flights() {
        return test_r5() && TEST_FLIGHTS.load(Ordering::Acquire) & TEST_LIFT != 0;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_R5_LIFT").is_some_and(|v| v == "1");
        let on = asked && r5();
        if on {
            tracing::info!(
                "R5 lifts on (VICTORY_R5_LIFT=1): reads of homes flights write are done on the card, \
                 behind the flights"
            );
        } else if asked {
            tracing::warn!("VICTORY_R5_LIFT=1 does nothing without VICTORY_R5=1");
        }
        on
    })
}

/// Tests with flights in the air: the relay's second stage on, and R5 if
/// asked, for the test's own thread and the relay's threads (named
/// `victory-relay` and `victory-late-wait`) while [`TestFlights`] lives.
/// Other threads see the switches as they are. Only one test at a time holds
/// it (its lock), and the process's `STARTED` and `FINISHED` count its
/// submissions: with `--test-threads=1`, as the WARP tests run.
#[cfg(test)]
static TEST_FLIGHTS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
thread_local! {
    static TEST_FLIGHTS_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Bits of [`TEST_FLIGHTS`]: flights on; R5 on; the memory core on (for the
/// relay's threads; the test's own thread sets `crate::home`'s); R5's lifts.
#[cfg(test)]
const TEST_FLYING: u8 = 1;
#[cfg(test)]
const TEST_R5: u8 = 2;
#[cfg(test)]
const TEST_CORE: u8 = 4;
#[cfg(test)]
const TEST_LIFT: u8 = 8;

/// Whether this thread sees the test's flights ([`TEST_FLIGHTS`]).
#[cfg(test)]
pub(crate) fn test_flights() -> bool {
    let bits = TEST_FLIGHTS.load(Ordering::Acquire);
    bits & TEST_FLYING != 0 && (TEST_FLIGHTS_HERE.with(std::cell::Cell::get) || test_relay_thread())
}

#[cfg(test)]
fn test_relay_thread() -> bool {
    matches!(
        std::thread::current().name(),
        Some("victory-relay" | "victory-late-wait")
    )
}

#[cfg(test)]
fn test_r5() -> bool {
    TEST_FLIGHTS.load(Ordering::Acquire) & TEST_R5 != 0
}

/// For `crate::home`'s test switch: whether this is a relay thread of a
/// test whose flights have the memory core on.
#[cfg(test)]
pub(crate) fn test_relay_sees() -> bool {
    let bits = TEST_FLIGHTS.load(Ordering::Acquire);
    bits & TEST_FLYING != 0 && bits & TEST_CORE != 0 && test_relay_thread()
}

/// Of the ranges a submission's shaders may write, per host copy: those in
/// homes of the memory core, and in card copies whose CPU side is one
/// (`Shadow::is_home`): title memory itself, written by the card while it
/// runs.
pub(super) fn home_ranges(back: &[Declared]) -> impl Iterator<Item = (usize, usize)> + '_ {
    back.iter()
        .filter(|(shadow, _)| shadow.is_home())
        .flat_map(|(_, ranges)| ranges.iter().copied())
}

/// Of [`home_ranges`], who declared them ([`Line::home_writers`]): `None` for
/// homes the shaders write in place, or a card copy whose CPU side is a home
/// (held weakly, so that its address is given to no other copy while this
/// lives). Its ranges in order and joined.
pub(super) type Writer = (
    Option<std::sync::Weak<crate::shadow::Shadow>>,
    Vec<(usize, usize)>,
);

/// `back`'s [`home_ranges`] joined into `writers`, each with who declared it.
pub(super) fn join_writers(writers: &mut Vec<Writer>, back: &[Declared]) {
    for (shadow, ranges) in back.iter().filter(|(shadow, _)| shadow.is_home()) {
        let carded = shadow.card_home().is_some();
        let at = writers.iter().position(|(by, _)| match by {
            None => !carded,
            Some(by) => carded && std::ptr::eq(by.as_ptr(), std::sync::Arc::as_ptr(shadow)),
        });
        let at = at.unwrap_or_else(|| {
            writers.push((
                carded.then(|| std::sync::Arc::downgrade(shadow)),
                Vec::new(),
            ));
            writers.len() - 1
        });
        join_ranges(&mut writers[at].1, ranges.iter().copied());
    }
}

/// With [`r5`] and flights in the air: whether `runs` refresh a range of a
/// host copy that is no home over title memory `homes_written` (in order and
/// joined: what flights' shaders may write in homes, [`Line::homes_written`]).
/// Such a copy (an exact copy of a buffer view over a home's bytes, a buffer
/// partly in a home, a texture's buffer) takes the title's bytes from the
/// home, which the card may be half way through writing; no copy-back comes
/// for it (nothing was declared on the copy), and the telling at the landing
/// marks the pages written only for its next refresh. So the submission
/// waits like a read of that memory, until those flights have landed. A
/// card copy whose CPU side is a home does not wait: its refresh sends those
/// bytes up from the home again ([`retake_flight_homes`]).
pub(super) fn refresh_over_flight_homes(
    homes_written: &[(usize, usize)],
    runs: &[&HostList],
) -> bool {
    if homes_written.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| {
            step.refreshes().any(|(shadow, at, len)| {
                !shadow.is_home() && touches(homes_written, at, at.saturating_add(len))
            })
        })
    })
}

/// With [`r5`] and flights in the air: of the refreshes in `runs` of card
/// copies whose CPU side is a home, the bytes the flights' shaders may write
/// in homes (`writers`, [`Line::home_writers`]) go up from the home again
/// before this submission (`Shadow::card_home_written`). Their write
/// watching sees only the title's writes, and what the flights' shaders
/// wrote in the home (in place, or merged from another copy) is told to them
/// only at those flights' landings, after this submission is prepared. The
/// upload runs on the queue behind those flights, so it takes their bytes.
///
/// A copy's own declared ranges are not taken again: its flights merged
/// them into the home from its own card side, in queue order, so for those
/// bytes the two already agree (and a copy that raw-address shaders reach is
/// declared written whole for every list with writes).
pub(super) fn retake_flight_homes(writers: &[Writer], runs: &[&HostList]) {
    if writers.is_empty() {
        return;
    }
    // One buffer for every refresh step (`join_ranges` joins in place).
    let mut others = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            for (shadow, at, len) in step.refreshes() {
                if shadow.card_home().is_none() {
                    continue;
                }
                let end = at.saturating_add(len);
                others.clear();
                join_ranges(
                    &mut others,
                    writers
                        .iter()
                        .filter(|(by, _)| {
                            !by.as_ref().is_some_and(|by| {
                                std::ptr::eq(by.as_ptr(), std::sync::Arc::as_ptr(shadow))
                            })
                        })
                        .flat_map(|(_, ranges)| ranges.iter().copied())
                        .filter(|&(from, to)| from < end && at < to)
                        .map(|(from, to)| (from.max(at), to.min(end))),
                );
                for &(from, to) in &others {
                    shadow.card_home_written(from, to);
                }
            }
        }
    }
}

/// Since the last frame-rate line, with [`r5`]: submissions that waited for
/// flights before a copy's refresh over a home they write
/// ([`refresh_over_flight_homes`]); waits for a memory value that waited for
/// the flights concerned only ([`late_met`]); blocking fence waits that
/// waited for the flights ahead of the fence's signals only
/// ([`settle_for_fence_at`]); the title's protections that waited for the
/// flights in the air ([`wait_for_flights_now`]). Then what
/// `super::late_lift` counts (its `R5_` places).
pub(super) static R5_STATS: [std::sync::atomic::AtomicU64; 9] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 9];

/// Whether flights in the air, or being landed, may still change title
/// memory `[address, address + bytes)` ([`late_met`] with [`r5`]): their
/// shaders may write it ([`Line::written`] holds the flights being landed
/// until their landing ends), or a bottom-of-pipe write over it waits in the
/// line or in hand.
pub(super) fn value_unsure(line: &Line, address: usize, bytes: usize) -> bool {
    let over = |at: usize, n: usize| at < address + bytes && address < at + n;
    touches(&line.written, address, address + bytes)
        || line.in_hand.iter().any(|&(at, _, n)| over(at, n))
        || line
            .line
            .iter()
            .any(|item| matches!(item, Late::Write(at, _, n) if over(*at, *n)))
}

/// The title blocks on `fence` (no event) until `reached` says its value
/// is there ([`late`]). With [`r5`] only until no signal of that fence waits
/// in the line or is being landed, not until everything in the air has
/// landed; without it [`settle_for_fence`].
pub(crate) fn settle_for_fence_at(fence: crate::object::This, reached: impl Fn() -> bool) {
    if !r5() {
        settle_for_fence();
        return;
    }
    let waited = relay_wait(WHY_FENCE, |line| {
        reached()
            || (!line.settling
                && !line
                    .line
                    .iter()
                    .any(|item| matches!(item, Late::Signal(f, _) if f.as_this() == fence)))
    });
    if waited {
        R5_STATS[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// With [`r5`] and flights in the air: waits until every flight sent
/// before the call has landed, then resets the lists they ran
/// ([`land`]): for a protection of the title's that the host refused while
/// a heap over its pages is held (`crate::home::title_virtual_protect`),
/// which cannot wait for a sweep. The microseconds it waited, or `None` if
/// it did not wait (nothing was in the air, or R5 is off). From a thread that holds no lock
/// of ours: the landing takes none it holds, and it is bounded by what is in
/// the air now.
pub(crate) fn wait_for_flights_now() -> Option<u64> {
    if !r5() || !flying() {
        return None;
    }
    let last = STARTED.load(Ordering::Acquire);
    let waiting = std::time::Instant::now();
    let waited = relay_wait(WHY_SYNC, |line| {
        FINISHED.load(Ordering::Acquire) >= last
            || (!line.settling
                && !line
                    .line
                    .iter()
                    .any(|item| matches!(item, Late::Run(flight) if flight.number <= last)))
    });
    land();
    recycled();
    if !waited {
        return None;
    }
    R5_STATS[3].fetch_add(1, Ordering::Relaxed);
    Some(waiting.elapsed().as_micros() as u64)
}

/// The number of the last submission that finished (with [`late`]: whose
/// flight landed).
pub(crate) fn finished() -> u64 {
    FINISHED.load(Ordering::Acquire)
}

/// Things of ours a submission's lists use on the card, reused by later
/// submissions: each is taken again only once the submission that last used
/// it has finished ([`FINISHED`]). With every submission waited for, the one
/// there is is always free again; with flights ([`late`]) each flight in the
/// air holds its own (`crate::memory`'s directory staging, `crate::card_home`'s
/// list).
pub(crate) struct Ring<T> {
    held: Vec<(u64, T)>,
}

/// The most things a [`Ring`] keeps that no submission in the air uses,
/// beside the one it hands out.
const RING_SPARE: usize = 4;

impl<T> Ring<T> {
    pub(crate) const fn new() -> Self {
        Self { held: Vec::new() }
    }

    /// One whose last submission has finished by `finished`, taken out (the
    /// oldest such); `None` if every one is still in use. Of the others that
    /// finished, at most [`RING_SPARE`] are kept: a burst of flights made
    /// more than the queue holds afterwards.
    pub(crate) fn take(&mut self, finished: u64) -> Option<T> {
        let at = (0..self.held.len())
            .filter(|&i| self.held[i].0 <= finished)
            .min_by_key(|&i| self.held[i].0)?;
        let taken = self.held.swap_remove(at).1;
        let mut spare = 0;
        self.held.retain(|&(number, _)| {
            if number > finished {
                return true;
            }
            spare += 1;
            spare <= RING_SPARE
        });
        Some(taken)
    }

    /// `item` is used by submission `number` from now on.
    pub(crate) fn put(&mut self, number: u64, item: T) {
        self.held.push((number, item));
    }
}

/// [`Host::relay_card_back`] for the ranges of copies whose CPU side is a
/// home (`crate::card_home`, with [`r5`]), as the `reference` arm of
/// [`Host::card_back`] after a wait: what their shaders wrote was merged
/// into the home on the card, in the flight itself behind its compare, so
/// nothing is fetched. The pieces the compare flagged (every piece of a
/// range it did not compare) go up again from the home before the next
/// submission, with their hashes made anew, and the copies over those bytes
/// elsewhere are told (`crate::card_home::merged`); with a home in a chunk of
/// the pool, at its other views too (`Shadow::card_merged_shown`).
///
/// Each flight's compare stored the hashes it made ([`flagged_pieces`]), so
/// it flagged what its own shaders changed since the compare before it; the
/// next submission's upload runs on the queue behind every flight sent
/// before it, whose merges the home holds by then.
pub(super) fn relay_card_homed(plans: &[&CardPlan]) {
    if !plans
        .iter()
        .any(|p| p.ranges.iter().any(|r| r.reference.is_some()))
    {
        return;
    }
    let differs = {
        let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        work.as_ref()
            .map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
    };
    let Some((differs, dwords)) = differs else {
        return;
    };
    // SAFETY: the mapping of the results. The compares of these plans have
    // run, and their dwords are given to no other plan before these are done
    // with (`card_done`).
    let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
    for plan in plans {
        for r in plan.ranges.iter().filter(|r| r.reference.is_some()) {
            // The merged pieces, joined where they touch, for the other views
            // of a home in a chunk of the pool.
            let mut run: Option<(usize, usize)> = None;
            for p in 0..r.span.pieces {
                if r.compared && differs.get((r.at + p) as usize).is_some_and(|&d| d == 0) {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo >= hi {
                    continue;
                }
                crate::card_home::merged(&r.shadow, lo, hi);
                run = match run {
                    Some((from, to)) if lo <= to && hi >= from => Some((from.min(lo), to.max(hi))),
                    Some((from, to)) => {
                        r.shadow.card_merged_shown(from, to);
                        Some((lo, hi))
                    }
                    None => Some((lo, hi)),
                };
            }
            if let Some((from, to)) = run {
                r.shadow.card_merged_shown(from, to);
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
    }
    // What the merges on the card stored into homes.
    COPY_BACK[1].fetch_add(crate::card_home::stored(), Ordering::Relaxed);
}

/// Tests with submissions in the air ([`r5`]): the switches, a hold on the
/// host queue, and what a test sends and waits for.
#[cfg(test)]
pub(crate) mod flights {
    use super::*;

    /// The relay's second stage (and R5, and the memory core for the
    /// relay's threads, as asked) on for this thread while it lives
    /// ([`TEST_FLIGHTS`]); one test at a time. On drop everything in the air
    /// lands first. Take it after `crate::home`'s test switch, so that it
    /// goes first.
    pub(crate) struct TestFlights(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    impl TestFlights {
        pub(crate) fn on(r5: bool, core: bool) -> TestFlights {
            Self::with(r5, core, false)
        }

        /// [`TestFlights::on`] with R5 and the memory core, and R5's lifts
        /// ([`r5_lift`]) as asked.
        pub(crate) fn r5_lifting(lift: bool) -> TestFlights {
            Self::with(true, true, lift)
        }

        fn with(r5: bool, core: bool, lift: bool) -> TestFlights {
            static ONE: Mutex<()> = Mutex::new(());
            let one = ONE.lock().unwrap_or_else(|e| e.into_inner());
            let bits = TEST_FLYING
                | if r5 { TEST_R5 } else { 0 }
                | if core { TEST_CORE } else { 0 }
                | if lift { TEST_LIFT } else { 0 };
            TEST_FLIGHTS.store(bits, Ordering::Release);
            TEST_FLIGHTS_HERE.with(|here| here.set(true));
            TestFlights(one)
        }

        /// Another thread of the test's sees the flights too, until it ends.
        pub(crate) fn here() {
            TEST_FLIGHTS_HERE.with(|here| here.set(true));
        }

        /// Waits until nothing is in the air or being landed, and resets
        /// the lists landed flights ran ([`land`]).
        pub(crate) fn land_all(&self) {
            relay_wait(WHY_SYNC, |line| line.line.is_empty() && !line.settling);
            land();
            recycled();
        }
    }

    impl Drop for TestFlights {
        fn drop(&mut self) {
            self.land_all();
            TEST_FLIGHTS.store(0, Ordering::Release);
            TEST_FLIGHTS_HERE.with(|here| here.set(false));
        }
    }

    /// The host queue waits for a fence of the test's: what is sent after
    /// this stays on it, in the air, until [`QueueHold::release`] (or the
    /// drop, so that a failed test leaves no queue standing).
    pub(crate) struct QueueHold(Released);

    /// The hold's fence, for a thread that releases it.
    #[derive(Clone)]
    pub(crate) struct Released(ID3D12Fence);

    // SAFETY: a D3D12 fence, free-threaded.
    unsafe impl Send for Released {}

    impl Released {
        pub(crate) fn release(&self) {
            // SAFETY: our fence; a signal from the CPU.
            let _ = unsafe { self.0.Signal(1) };
        }

        /// Released by another thread after `ms`.
        pub(crate) fn after(&self, ms: u64) -> std::thread::JoinHandle<()> {
            let released = self.clone();
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(ms));
                released.release();
            })
        }
    }

    impl QueueHold {
        pub(crate) fn new(host: &Host) -> QueueHold {
            // SAFETY: a fence of our own; the wait goes on the queue under
            // the submission lock, between two submissions.
            let fence: ID3D12Fence =
                unsafe { host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.expect("a fence");
            let submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: as above.
            unsafe { host.queue.Wait(&fence, 1) }.expect("a wait on the queue");
            drop(submitted);
            QueueHold(Released(fence))
        }

        pub(crate) fn release(&self) {
            self.0.release();
        }

        /// The fence, for a thread that releases it.
        pub(crate) fn released(&self) -> Released {
            self.0.clone()
        }
    }

    /// The host's submission lock, held as a submission holds it (a test
    /// that waits under it for what a submission waits for).
    pub(crate) fn submission_lock(host: &Host) -> std::sync::MutexGuard<'_, u64> {
        host.submitted.lock().unwrap_or_else(|e| e.into_inner())
    }

    impl Drop for QueueHold {
        fn drop(&mut self) {
            self.release();
        }
    }

    /// Flights are not landed while it lives (the line's lock is held): what
    /// the card runs meanwhile is all done before any of their copy-backs.
    /// Nothing that takes the line is called while it is held.
    pub(crate) struct LandingHeld(#[allow(dead_code)] std::sync::MutexGuard<'static, Line>);

    impl LandingHeld {
        pub(crate) fn new() -> LandingHeld {
            LandingHeld(line())
        }
    }

    impl Host {
        /// Waits until the card ran everything sent so far (a signal of a
        /// fence of the test's behind it on the queue).
        pub(crate) fn card_done_for_test(&self) {
            // SAFETY: a fence of our own; the signal goes on the queue under
            // the submission lock, between two submissions, and the wait
            // with no event returns once it is reached.
            unsafe {
                let fence: ID3D12Fence = self
                    .device
                    .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                    .expect("a fence");
                let submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.Signal(&fence, 1).expect("a signal on the queue");
                drop(submitted);
                fence
                    .SetEventOnCompletion(1, HANDLE::default())
                    .expect("a wait for the fence");
            }
        }

        /// Sends `list` (closed) as a flight ([`Host::execute_own`]); the
        /// caller keeps it and does not reset it before the flight landed.
        pub(crate) fn send_for_test(&self, list: &HostList) -> u64 {
            self.execute_own(list)
        }

        /// Sends `list` (closed) as a flight that owns it, as a queue's
        /// submission does ([`Host::execute_late`]): reset and kept for
        /// reuse once the flight landed.
        pub(crate) fn send_owned_for_test(&self, list: HostList) {
            self.execute_late(vec![crate::list::detached_for_test(list)], vec![0]);
        }
    }

    /// Waits until submission `number` (and every one before it) has
    /// landed, as other threads go on sending.
    pub(crate) fn land_through(number: u64) {
        relay_wait(WHY_SYNC, |_| FINISHED.load(Ordering::Acquire) >= number);
    }

    /// Flights in the air (or being landed).
    pub(crate) fn in_air() -> usize {
        let line = line();
        line.line
            .iter()
            .filter(|l| matches!(l, Late::Run(_)))
            .count()
            + usize::from(line.settling)
    }

    /// Why `runs`, prepared now, would wait for flights in the air
    /// ([`late_blocked`]).
    pub(crate) fn blocked(runs: &[&HostList]) -> Option<usize> {
        late_blocked(runs)
    }

    /// Whether `runs` would wait for a copy's refresh over a home a flight
    /// writes ([`refresh_over_flight_homes`]).
    pub(crate) fn refresh_waits(runs: &[&HostList]) -> bool {
        let homes = line().homes_written.clone();
        refresh_over_flight_homes(&homes, runs)
    }

    /// R5's counters ([`R5_STATS`]), now.
    pub(crate) fn r5_stats() -> [u64; R5_STATS.len()] {
        std::array::from_fn(|i| R5_STATS[i].load(Ordering::Relaxed))
    }

    /// Of [`r5_stats`]: H8 waits, refreshes' submissions lifted, refused.
    pub(crate) const R5_H8: usize = super::super::late_lift::R5_H8;
    pub(crate) const R5_REFRESHES_LIFTED: usize = super::super::late_lift::R5_REFRESHES_LIFTED;
    pub(crate) const R5_REFRESHES_REFUSED: usize = super::super::late_lift::R5_REFRESHES_REFUSED;

    /// The waits for flights to land that lifted reads left out
    /// (`LIFT_STATS`), now.
    pub(crate) fn waits_left() -> u64 {
        LIFT_STATS[LIFT_WAITS_LEFT].load(Ordering::Relaxed)
    }

    /// Plans of lifted reads aborted because a texture reads what the card
    /// writes, and because a copy over a lifted copy's home is refreshed.
    pub(crate) fn aborted() -> (u64, u64) {
        let at = |why: LiftAbort| LIFT_ABORTED[why as usize].load(Ordering::Relaxed);
        (at(LiftAbort::TextureAfter), at(LiftAbort::Viewed))
    }

    /// What flights' shaders may write in homes ([`Line::homes_written`]).
    pub(crate) fn homes_written() -> Vec<(usize, usize)> {
        line().homes_written.clone()
    }

    /// `late_signal`, for a fence of the title's: behind the flights.
    pub(crate) fn signal_behind(fence: crate::fence::FenceRef, value: u64) -> bool {
        late_signal(fence, value).is_ok()
    }

    /// Why a submission waits for a read of what flights write.
    pub(crate) const READ: usize = WHY_READ;

    /// A ring hands out what its submission finished with, the oldest
    /// first, and nothing still in use.
    #[test]
    fn a_ring_hands_out_only_what_finished() {
        let mut ring: Ring<&str> = Ring::new();
        assert_eq!(ring.take(10), None);
        ring.put(3, "a");
        ring.put(5, "b");
        assert_eq!(ring.take(2), None, "both in use");
        assert_eq!(ring.take(5), Some("a"), "the oldest that finished");
        ring.put(7, "a");
        assert_eq!(ring.take(6), Some("b"));
        assert_eq!(ring.take(6), None, "a is used by 7");
        assert_eq!(ring.take(7), Some("a"));
        // After a burst, only a few that finished stay beside the one taken.
        let mut ring: Ring<u64> = Ring::new();
        for n in 1..=10 {
            ring.put(n, n);
        }
        ring.put(20, 20);
        assert_eq!(ring.take(10), Some(1));
        assert_eq!(
            ring.held.len(),
            RING_SPARE + 1,
            "four spare and the one in use"
        );
        assert!(ring.held.iter().any(|&(n, _)| n == 20));
    }

    /// A value the title waits for is unsure while a flight's shaders may
    /// write it or a write of it waits in the line, and only there.
    #[test]
    fn a_value_is_unsure_only_where_flights_may_change_it() {
        let mut line = Line {
            line: Vec::new(),
            written: vec![(0x1000, 0x2000)],
            homes_written: Vec::new(),
            home_writers: Vec::new(),
            lists: 0,
            target: 0,
            settling: false,
            poke: false,
            landed: Vec::new(),
            in_hand: Vec::new(),
        };
        assert!(value_unsure(&line, 0x1ff8, 8));
        assert!(!value_unsure(&line, 0x2000, 8));
        line.line
            .push(Late::Write(0x3000, crate::queue::Written::Value(1), 8));
        assert!(value_unsure(&line, 0x3004, 4));
        assert!(!value_unsure(&line, 0x3008, 4));
        line.in_hand
            .push((0x4000, crate::queue::Written::Value(2), 4));
        assert!(value_unsure(&line, 0x4000, 8));
    }
}
