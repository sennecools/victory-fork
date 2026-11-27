//! R5's refreshes over flights' homes, done on the card instead of waiting
//! for those flights ([`lift_refreshes`]), and what the submitting thread's
//! waits for flights say on the profile ([`wait_args`]).
//!
//! With R5 the submitting thread waits for flights before preparing for two
//! reasons only, both "title memory read": a title buffer copy, a fill or a
//! texture that reads (or a copy that writes) what a flight's shaders may
//! write (H8, `blocked_by`; lifted only as `VICTORY_LIFT_READS` lifts it,
//! never from or into a home), and a refresh of a copy that is no home over
//! a home's bytes a flight's shaders may write (H2,
//! [`refresh_over_flight_homes`]). H2 is lifted: the refresh is done on the
//! CPU around the part over `homes_written`, and the card copies that part
//! from the home's placed buffer into the copy at the head of the
//! submission, behind the flights that write it.
//!
//! What still waits for H2 is counted in [`R5_REFRESHES_REFUSED`], with its
//! reasons (`crate::frame_budget::h2_refused`).

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
        // Split by home: what flights write in homes is joined where it
        // touches, so one range can lie over two homes side by side.
        let Some(homes) = homes_over(from, to) else {
            plan.refuse(Refusal::NoHome);
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
        for (home, at, end) in homes {
            plan.pieces.push(Piece {
                list,
                step,
                copy: copy.clone(),
                home,
                at,
                len: end - at,
            });
        }
    }
    // (Never empty without a refusal: the caller found such a refresh.)
    if plan.pieces.is_empty() && plan.refused.is_none() {
        plan.refuse(Refusal::NoHome);
    }
    plan
}

/// The homes that hold `[from, to)` between them, in order, each with the
/// part it holds; `None` if a byte of it lies in no home.
fn homes_over(
    from: usize,
    to: usize,
) -> Option<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>> {
    let mut homes = Vec::new();
    let mut at = from;
    while at < to {
        let home = crate::shadow::home_of(at as u64, 1)?;
        let (start, len) = home.span();
        // The home holds byte `at`, so its end lies past it.
        let end = ((start + len) as usize).min(to);
        homes.push((home, at, end));
        at = end;
    }
    Some(homes)
}

/// With [`r5`] and flights in the air (`plan`: what [`plan_refresh_lifts`]
/// found for `runs`): the refreshes of `runs` over homes' bytes flights'
/// shaders may write (`homes_written`, as
/// [`refresh_over_flight_homes`] finds them) are done on the card instead of
/// waiting for those flights. Each such refresh stays a refresh on the CPU
/// around those bytes (the submission's batch leaves the pieces out,
/// `Batch::around`), and the pieces over `homes_written`, split by home,
/// become lifted copies (`LiftedTo::Refresh`) from the home's placed buffer
/// into the copy, behind the refresh step: at the head of the submission,
/// behind every flight on the queue. The CPU never writes a piece while a
/// flight may still be writing its home, so a flight sent before that reads
/// the copy sees no half-written bytes of a later one; once the flights land
/// the copy is told those bytes changed (`crate::home::after_wait`), and its
/// next refresh takes them on the CPU again (they were left stale). The
/// copy and title range of each piece, or `None` if refused.
///
/// Refused (counted; the submission waits as before) when the copy has a
/// declared write of its own over such a part in this submission: its
/// snapshot, taken on the CPU, would hold the bytes from before, and its
/// copy-back would write the home's bytes over the home after the title may
/// have changed them. Likewise when a flight in the air may write the copy
/// over such a part (`Shadow::writes_pending_over`): the card's copy would go
/// over what that flight's shaders wrote there, and its copy-back would take
/// the home's bytes for theirs. And when the copy's card side is in the
/// card's memory (it goes up from the CPU side before the lists), no one
/// home holds a part, or a list is named twice; and, as a guard only, when a
/// step that reads such a part is no refresh (a lifted read takes a sole
/// copy, never one over a home's bytes).
/// Each refusal is counted with its reasons (`crate::frame_budget::h2_refused`).
pub(super) fn lift_refreshes(plan: RefreshPlan, runs: &[&HostList]) -> Option<Vec<LiftedPiece>> {
    if let Some(first) = plan.refused {
        R5_STATS[R5_REFRESHES_REFUSED].fetch_add(1, Ordering::Relaxed);
        crate::frame_budget::h2_refused(first, plan.per_piece);
        return None;
    }
    let lifted = plan
        .pieces
        .iter()
        .map(|p| (p.copy.clone(), p.at, p.at + p.len))
        .collect();
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
    Some(lifted)
}

/// A piece of a refresh [`lift_refreshes`] gave the card: the copy, and the
/// title range (start, end) the card writes into it.
pub(super) type LiftedPiece = (std::sync::Arc<crate::shadow::Shadow>, usize, usize);

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
}
