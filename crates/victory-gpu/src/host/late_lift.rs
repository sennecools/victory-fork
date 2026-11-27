//! R5's S2 (`VICTORY_R5_LIFT=1`, [`r5_lift`]): reads of title memory that
//! would wait for flights are done on the card instead, where the bytes are
//! a home's too; and what the submitting thread's waits for flights say on
//! the profile ([`wait_args`]).
//!
//! With R5 the submitting thread waits for flights before preparing for two
//! reasons only, both "title memory read": a title buffer copy, a fill or a
//! texture that reads (or a copy that writes) what a flight's shaders may
//! write (H8, `blocked_by`), and a refresh of a copy that is no home over a
//! home's bytes a flight's shaders may write (H2,
//! [`refresh_over_flight_homes`]). With the switch:
//!
//! - H8 is lifted as `VICTORY_LIFT_READS=2` lifts it (`lift_reads`), and a
//!   home that holds the whole range is a source or destination like a sole
//!   copy (`crate::shadow::sole_or_home`): the copy reads or writes the
//!   home's placed buffer at `Shadow::offset`, through a bounce buffer
//!   within one buffer (two homes in one chunk of the pool share it). A
//!   lifted copy into a home is declared as the list's shaders' writes are
//!   (`Step::card_writes`): it is in the flight's `homes_written` until the
//!   flight lands, and the landing tells the copies over those bytes, as for
//!   a shader's write (`crate::home::after_wait`).
//! - H2 is lifted by [`lift_refreshes`]: the refresh is done on the CPU as
//!   before, and the card then copies the part over `homes_written` from the
//!   home's placed buffer into the copy at the head of the submission, behind
//!   the flights that write it.
//!
//! What still waits is counted: `LiftAbort` for H8, [`R5_REFRESHES_REFUSED`]
//! for H2.

use super::*;

/// Places of [`R5_STATS`] from here on: microseconds of H2's waits; H8's
/// waits and their microseconds; submissions whose refreshes over flights'
/// homes were done on the card ([`lift_refreshes`]); and those that waited
/// all the same.
pub(super) const R5_H2_US: usize = 4;
pub(super) const R5_H8: usize = 5;
pub(super) const R5_H8_US: usize = 6;
pub(super) const R5_REFRESHES_LIFTED: usize = 7;
pub(super) const R5_REFRESHES_REFUSED: usize = 8;

/// The `kind` argument of a "relay wait" span ([`wait_args`]): what read
/// first what flights write.
const KIND_FILL: u64 = 1;
const KIND_COPY: u64 = 2;
const KIND_TEXTURE: u64 = 3;
/// A refresh of a copy that is no home over a home's bytes (H2).
const KIND_H2: u64 = 4;

/// The parts of `[from, to)` that lie in `ranges` (in order and joined).
fn overlaps(
    ranges: &[(usize, usize)],
    from: usize,
    to: usize,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges[at..]
        .iter()
        .take_while(move |r| r.0 < to)
        .map(move |r| (r.0.max(from), r.1.min(to)))
        .filter(|r| r.0 < r.1)
}

/// The arguments of the profile's "relay wait" span of a submission about
/// to wait for flights before it is prepared (none without the profiler):
/// `kind` (fill 1, copy 2, texture 3: H8; 4: H2, `h2`), `bytes` (what its
/// steps read or write of what flights' shaders may write, in homes for H2),
/// `back` (how many submissions back the youngest flight that declared any
/// of it was sent; 0 if none is in the line, as when it is being landed), and
/// `homes_only` (1 if every range those flights declared over it is a
/// home's).
pub(super) fn wait_args(runs: &[&HostList], h2: bool) -> Vec<(&'static str, u64)> {
    if !crate::profile::on() {
        return Vec::new();
    }
    // The line taken once: what flights write, and per flight its number
    // and the ranges it declared (whether a home's; read backs are none).
    let (against, flights) = {
        let line = line();
        let against = if h2 {
            line.homes_written.clone()
        } else {
            line.written.clone()
        };
        let flights: Vec<(u64, Vec<(usize, usize, bool)>)> = line
            .line
            .iter()
            .filter_map(|item| match item {
                Late::Run(flight) => Some(flight),
                _ => None,
            })
            .map(|flight| {
                let mut ranges: Vec<(usize, usize, bool)> = flight
                    .back
                    .iter()
                    .flat_map(|(shadow, ranges)| {
                        let home = shadow.is_home();
                        ranges.iter().map(move |&(from, to)| (from, to, home))
                    })
                    .collect();
                ranges.extend(
                    flight
                        .read_backs
                        .iter()
                        .map(|&(from, to)| (from, to, false)),
                );
                (flight.number, ranges)
            })
            .collect();
        (against, flights)
    };
    let mut kind = 0;
    let mut over: Vec<(usize, usize)> = Vec::new();
    let mut note = |what: u64, from: usize, len: usize| {
        let to = from.saturating_add(len);
        if touches(&against, from, to) {
            if kind == 0 {
                kind = what;
            }
            join_ranges(&mut over, overlaps(&against, from, to));
        }
    };
    // One list's lock at a time, and not under the line's.
    for list in runs {
        for step in &list.deferred().copies {
            if h2 {
                for (shadow, at, len) in step.refreshes() {
                    if !shadow.is_home() {
                        note(KIND_H2, at, len);
                    }
                }
                continue;
            }
            match step {
                &Step::Copy(dst, src, len) => {
                    note(KIND_COPY, src, len);
                    note(KIND_COPY, dst, len);
                }
                &Step::Fill(_, src, len) => note(KIND_FILL, src, len),
                Step::Texture { staging, .. } => {
                    let (at, len) = staging.memory();
                    note(KIND_TEXTURE, at, len);
                }
                _ => {}
            }
        }
    }
    let bytes: usize = over.iter().map(|(from, to)| to - from).sum();
    let (mut youngest, mut homes_only) = (0, true);
    for (number, ranges) in &flights {
        let mut declared = false;
        for &(from, to, home) in ranges {
            if touches(&over, from, to) {
                declared = true;
                homes_only &= home;
            }
        }
        if declared {
            youngest = youngest.max(*number);
        }
    }
    // This submission is the next one started.
    let back = match youngest {
        0 => 0,
        youngest => (STARTED.load(Ordering::Acquire) + 1).saturating_sub(youngest),
    };
    vec![
        ("kind", kind),
        ("bytes", bytes as u64),
        ("back", back),
        ("homes_only", u64::from(youngest != 0 && homes_only)),
    ]
}

/// With [`r5_lift`], of a plan of lifted reads (`lift_reads`; `steps` as the
/// plan saw them): whether a lifted copy writes a home and the submission
/// refreshes another copy over those bytes (one that is not that home: an
/// exact copy of a view, a buffer partly in the home, a card copy whose CPU
/// side it is). The refresh takes them on the CPU, before the card's copy, so
/// the plan is aborted (`LiftAbort::Viewed`) and the submission waits.
pub(super) fn over_lifted_homes<C>(
    lifts: &[Lift<std::sync::Arc<crate::shadow::Shadow>, C>],
    steps: &[(usize, usize, Seen)],
    runs: &[&HostList],
) -> bool {
    // Where each list's steps start in `steps`: they stand list by list,
    // each step at its place in its list, so a lift's step is found at
    // once (a plan can hold thousands of lifts over as many steps).
    let mut starts: Vec<usize> = Vec::new();
    for (i, &(list, _, _)) in steps.iter().enumerate() {
        if starts.len() <= list {
            starts.resize(list + 1, i);
        }
    }
    let mut homes = Vec::new();
    for lift in lifts {
        let LiftTo::Copy { dst, .. } = &lift.to else {
            continue;
        };
        if dst.home_kind().is_none() {
            continue;
        }
        let at = |&&(list, step, _): &&(usize, usize, Seen)| list == lift.list && step == lift.step;
        let seen = starts
            .get(lift.list)
            .and_then(|start| steps.get(start + lift.step))
            .filter(at)
            // (Steps not in that order are looked for one by one.)
            .or_else(|| steps.iter().find(at));
        if let Some(&(_, _, Seen::Copy(at, _, len))) = seen {
            join_ranges(&mut homes, [(at, at + len)]);
        }
    }
    if homes.is_empty() {
        return false;
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| {
            step.refreshes().any(|(shadow, at, len)| {
                shadow.home_kind().is_none() && touches(&homes, at, at.saturating_add(len))
            })
        })
    })
}

/// A piece of a refresh [`lift_refreshes`] gives the card: the list's place
/// in the submission, the refresh step's in the list, the copy refreshed,
/// the home holding the piece, its title address and bytes.
struct Piece {
    list: usize,
    step: usize,
    copy: std::sync::Arc<crate::shadow::Shadow>,
    home: std::sync::Arc<crate::shadow::Shadow>,
    at: usize,
    len: usize,
}

use crate::frame_budget::Refusal;

/// The first refresh of a submission over home bytes flights write (H2):
/// the copy, its title range (start, end) and where it was asked for.
pub(super) type FirstRefresh = (
    std::sync::Arc<crate::shadow::Shadow>,
    usize,
    usize,
    crate::frame_budget::Source,
);

/// What [`plan_refresh_lifts`] found: the pieces, or the first reason a
/// piece was refused for, and per [`Refusal`] the pieces it refused; and the
/// first refresh that met H2, for the frame budget's class.
pub(super) struct RefreshPlan {
    pieces: Vec<Piece>,
    refused: Option<Refusal>,
    per_piece: [u32; Refusal::ALL.len()],
    first: Option<FirstRefresh>,
}

impl RefreshPlan {
    fn refuse(&mut self, why: Refusal) {
        self.refused.get_or_insert(why);
        self.per_piece[why as usize] += 1;
    }

    /// The first refresh that met H2.
    pub(super) fn first(&self) -> Option<&FirstRefresh> {
        self.first.as_ref()
    }

    /// The first reason a piece was refused for, and the pieces refused per
    /// [`Refusal`]; `None` if the card can take every piece.
    pub(super) fn refusal(&self) -> Option<(Refusal, [u32; Refusal::ALL.len()])> {
        self.refused.map(|first| (first, self.per_piece))
    }
}

/// The pieces of the refreshes of `runs` over `homes_written` the card can
/// take ([`lift_refreshes`]), or why not: every piece is looked at, so that
/// each reason is counted, and the first refused is the submission's reason.
/// One walk over the lists, which also finds the first refresh that met H2
/// ([`RefreshPlan::first`]); the copies' tables are looked at only after
/// it, with no list's lock held. Inserts nothing.
pub(super) fn plan_refresh_lifts(
    homes_written: &[(usize, usize)],
    runs: &[&HostList],
) -> RefreshPlan {
    use std::sync::Arc;
    let mut plan = RefreshPlan {
        pieces: Vec::new(),
        refused: None,
        per_piece: [0; Refusal::ALL.len()],
        first: None,
    };
    let twice = runs
        .iter()
        .enumerate()
        .any(|(i, a)| runs[..i].iter().any(|b| std::ptr::eq(*a, *b)));
    if twice {
        plan.refuse(Refusal::Twice);
    }
    // The pieces (list, step, copy, address, end), then their homes: the
    // copies' tables are not looked at under a list's lock. With them, the
    // card writes of the submission's steps (copy, address, end).
    let mut found = Vec::new();
    let mut writes = Vec::new();
    for (l, list) in runs.iter().enumerate() {
        for (i, step) in list.deferred().copies.iter().enumerate() {
            if let Some((shadow, at, len)) = step.card_writes() {
                writes.push((Arc::as_ptr(shadow), at, at.saturating_add(len)));
            }
            for (copy, at, len) in step.refreshes() {
                let end = at.saturating_add(len);
                if copy.is_home() || !touches(homes_written, at, end) {
                    continue;
                }
                if plan.first.is_none() {
                    plan.first = Some((copy.clone(), at, end, step.source()));
                }
                if twice {
                    continue;
                }
                let why = if !matches!(step, Step::Refresh(..)) {
                    Some(Refusal::NotRefresh)
                } else if copy.carded() {
                    Some(Refusal::Carded)
                } else {
                    None
                };
                for (from, to) in overlaps(homes_written, at, end) {
                    match why {
                        Some(why) => plan.refuse(why),
                        None => found.push((l, i, copy.clone(), from, to)),
                    }
                }
            }
        }
    }
    for (list, step, copy, from, to) in found {
        if copy.writes_pending_over(from, to - from) {
            plan.refuse(Refusal::Pending);
            continue;
        }
        let Some(home) = crate::shadow::home_of(from as u64, (to - from) as u64) else {
            let two = crate::shadow::home_of(from as u64, 1)
                .zip(crate::shadow::home_of(to as u64 - 1, 1))
                .is_some_and(|(a, b)| !Arc::ptr_eq(&a, &b));
            plan.refuse(if two {
                Refusal::TwoHomes
            } else {
                Refusal::NoHome
            });
            continue;
        };
        let ptr = Arc::as_ptr(&copy);
        if writes
            .iter()
            .any(|&(shadow, at, end)| shadow == ptr && from < end && at < to)
        {
            plan.refuse(Refusal::Written);
            continue;
        }
        plan.pieces.push(Piece {
            list,
            step,
            copy,
            home,
            at: from,
            len: to - from,
        });
    }
    // (Never empty without a refusal: the caller found such a refresh.)
    if plan.pieces.is_empty() && plan.refused.is_none() {
        plan.refuse(Refusal::NoHome);
    }
    plan
}

/// What [`lift_refreshes`] did with a submission's refreshes over flights'
/// homes.
pub(super) enum RefreshLift {
    /// Not asked for (`VICTORY_R5_LIFT` off).
    Off,
    /// Taken on the card.
    Lifted,
    /// Refused (the reasons are counted): the submission waits.
    Refused,
}

/// With [`r5_lift`] and flights in the air (`plan`: what [`plan_refresh_lifts`]
/// found for `runs`): the refreshes of `runs` over
/// homes' bytes flights' shaders may write (`homes_written`, as
/// [`refresh_over_flight_homes`] finds them) are done on the card instead of
/// waiting for those flights. Each such refresh stays a refresh on the CPU,
/// and the parts over `homes_written` become lifted copies
/// (`LiftedTo::Refresh`) from the home's placed buffer into the copy, behind
/// the refresh step: at the head of the submission, behind every flight on
/// the queue. They write over what the CPU took there, and once the flights
/// land the copy is told those bytes changed (`crate::home::after_wait`), so
/// its next refresh takes them again. What it did: with
/// [`RefreshLift::Lifted`] nothing is left to wait for.
///
/// Refused (counted; the submission waits as before) when the copy has a
/// declared write of its own over such a part in this submission: its
/// snapshot, taken on the CPU, would hold the bytes from before, and its
/// copy-back would write the home's bytes over the home after the title may
/// have changed them. Likewise when a flight in the air may write the copy
/// over such a part (`Shadow::writes_pending_over`): the card's copy would go
/// over what that flight's shaders wrote there, and its copy-back would take
/// the home's bytes for theirs. And when such a part is a lifted read's, the
/// copy's card side is in the card's memory (it goes up from the CPU side
/// before the lists), no one home holds a part, or a list is named twice.
/// Each refusal is counted with its reasons (`crate::frame_budget::h2_refused`).
pub(super) fn lift_refreshes(plan: RefreshPlan, runs: &[&HostList]) -> RefreshLift {
    if !r5_lift() {
        return RefreshLift::Off;
    }
    if let Some(first) = plan.refused {
        R5_STATS[R5_REFRESHES_REFUSED].fetch_add(1, Ordering::Relaxed);
        crate::frame_budget::h2_refused(first, plan.per_piece);
        return RefreshLift::Refused;
    }
    // From the last, so that the places found hold; behind each refresh in
    // the order found.
    for piece in plan.pieces.into_iter().rev() {
        let lifted = Lifted {
            src: piece.home,
            src_at: piece.at,
            len: piece.len,
            to: LiftedTo::Refresh(piece.copy),
        };
        runs[piece.list]
            .deferred()
            .copies
            .insert(piece.step + 1, Step::Lifted(Box::new(lifted)));
    }
    R5_STATS[R5_REFRESHES_LIFTED].fetch_add(1, Ordering::Relaxed);
    RefreshLift::Lifted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parts of a range in joined ranges.
    #[test]
    fn overlaps_are_the_parts_inside() {
        let ranges = [(10, 20), (30, 40), (50, 60)];
        assert_eq!(
            overlaps(&ranges, 15, 55).collect::<Vec<_>>(),
            [(15, 20), (30, 40), (50, 55)]
        );
        assert_eq!(overlaps(&ranges, 20, 30).count(), 0);
        assert_eq!(overlaps(&ranges, 0, 100).count(), 3);
    }

    /// One holder of title memory as `crate::shadow::sole_or_home` sees it:
    /// start, end, in place, a home, its buffer.
    type Holder = (usize, usize, bool, bool, usize);

    /// `crate::shadow::sole_or_home` over `holders` (with `homes`: else
    /// `sole`): the one holder of every byte of the range, named (and known
    /// to the plan) by its buffer, if `crate::shadow::sole_takes` takes it.
    fn resolve(holders: &[Holder], homes: bool, at: usize, len: usize) -> Option<(usize, usize)> {
        let mut over = holders.iter().filter(|h| h.0 < at + len && at < h.1);
        let one = over.next()?;
        if over.next().is_some() {
            return None;
        }
        let taken = crate::shadow::sole_takes(one.2, one.3, homes);
        (taken && one.0 <= at && at + len <= one.1).then_some((one.4, one.4))
    }

    /// The plan with R5's lifts takes a home holding the whole range as the
    /// source and destination of a copy (between two buffers, or through a
    /// bounce buffer within one chunk of the pool), and the source of a
    /// fill; two homes over the range, a home and a view's copy over the
    /// same bytes, or title memory in place that is no home are not one copy.
    /// Without R5's lifts a home is not taken, as before.
    #[test]
    fn plan_lifts_takes_a_home_as_source_and_destination() {
        // Two homes in one chunk of the pool (buffer 1), one of its own (2),
        // a copy (3), a view's copy over the second home (4), memory in
        // place that is no home (5).
        let holders: [Holder; 6] = [
            (0x1000, 0x2000, true, true, 1),
            (0x2000, 0x3000, true, true, 1),
            (0x5000, 0x6000, true, true, 2),
            (0x8000, 0x9000, false, false, 3),
            (0x2800, 0x2900, false, false, 4),
            (0xa000, 0xb000, true, false, 5),
        ];
        // The flights write every byte here.
        let written = [(0x1000, 0xc000)];
        let plan = |homes: bool, seen: Seen| {
            plan_lifts(
                &written,
                [(0, 0, seen)],
                true,
                |at, len| resolve(&holders, homes, at, len),
                |_, at, _| Some(at),
            )
        };
        let copy = |plan: Result<Vec<Lift<usize, usize>>, LiftAbort>| match plan {
            Ok(lifts) => match lifts.as_slice() {
                [
                    Lift {
                        src,
                        to: LiftTo::Copy { dst, bounce },
                        ..
                    },
                ] => Ok((*src, *dst, *bounce)),
                other => panic!("one copy: {other:?}"),
            },
            Err(why) => Err(why),
        };
        assert_eq!(
            copy(plan(true, Seen::Copy(0x5100, 0x1100, 0x100))),
            Ok((1, 2, false)),
            "two buffers"
        );
        assert_eq!(
            copy(plan(true, Seen::Copy(0x2100, 0x1100, 0x100))),
            Ok((1, 1, true)),
            "one chunk"
        );
        assert_eq!(
            copy(plan(true, Seen::Copy(0x8100, 0x5100, 0x100))),
            Ok((2, 3, false)),
            "into a copy"
        );
        assert_eq!(
            plan(true, Seen::Copy(0x5100, 0x1f80, 0x100)),
            Err(LiftAbort::NotSole),
            "two homes over the source"
        );
        assert_eq!(
            plan(true, Seen::Copy(0x5100, 0x2800, 0x80)),
            Err(LiftAbort::NotSole),
            "a home and a view's copy"
        );
        assert_eq!(
            plan(true, Seen::Copy(0x5100, 0xa100, 0x80)),
            Err(LiftAbort::NotSole),
            "in place"
        );
        assert_eq!(
            plan(false, Seen::Copy(0x5100, 0x1100, 0x100)),
            Err(LiftAbort::NotSole),
            "without R5's lifts"
        );
        assert_eq!(
            plan(true, Seen::Copy(0x8100, 0x8180, 0x100)),
            Err(LiftAbort::InPlace)
        );
        match plan(true, Seen::Fill(0x10_0000, 0x5100, 0x40)) {
            Ok(lifts) => assert!(matches!(
                lifts.as_slice(),
                [Lift {
                    src: 2,
                    to: LiftTo::Fill(0x10_0000),
                    ..
                }]
            )),
            Err(why) => panic!("a fill from a home: {why:?}"),
        }
    }
}
