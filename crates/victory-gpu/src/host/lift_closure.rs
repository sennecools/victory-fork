//! Lifted reads take along what depends on them (`VICTORY_LIFT_CLOSURE=1`,
//! [`lift_closure`]), with the plan's ranges joined once a round
//! (`VICTORY_LIFT_JOIN=1`, [`lift_join`]).
//!
//! [`plan_lifts`] walks a submission's steps in order and aborts the whole
//! plan when a title buffer copy left on the CPU writes what a lifted step
//! reads or writes ([`LiftAbort::CpuWrites`]): the CPU does all its copies
//! at submission, before the card runs any lifted one, so the two would
//! happen in the other order. The submission then waits for the flights.
//!
//! With the closure such a copy is lifted too, and so is everything that
//! depends on a lifted step or that it depends on, round by round, until
//! nothing more is in the way ([`plan_closure`]). The lift list records the
//! lifted steps in the order they stand in (`record_lifts`, each between
//! its own barriers), which is the order the CPU would have done them in,
//! so a step behind another reads what that one left.
//!
//! Ported from cool_fox's card steps (`VICTORY_CARD_STEPS`): cc03ecb (the
//! steps done on the card), 007fe55 (what depends on them taken along),
//! 3e64a64 (ranges joined once a round, each place looked up once). Their
//! list of ours with its own buffer between reading and writing is not
//! taken: our lift list copies from host copy to host copy (through a
//! bounce buffer within one) and fills into the lists' shared scratch, as
//! `VICTORY_LIFT_READS=2` does, and a home is never a source or destination
//! (`crate::shadow::sole`).

use super::*;

/// Whether lifted reads take along the copies and fills that depend on
/// them, or they on them, instead of aborting the plan
/// (`VICTORY_LIFT_CLOSURE=1`; off by default, and only where lifts are made:
/// `VICTORY_LIFT_READS` at `1` or `2`).
///
/// Taken along: every title buffer copy or fill of the submission that
/// writes what a lifted step reads or writes, or reads what one writes,
/// ahead of it or behind it. What is left on the CPU then writes none of
/// their bytes, and reads of them only where it stands ahead of the lifted
/// copy that writes them (it reads them before that copy, as it did). A
/// fill that cannot be lifted is not taken along but left on the CPU; it
/// aborts (`LiftAbort::BigScratch`) only behind a lifted copy that writes
/// what it reads, or where it reads what flights write. A texture behind a
/// lifted copy over its memory, a step of ours or a copy the card cannot do
/// still abort, as do more than [`CLOSURE_MOST`] lifted steps.
///
/// Why: cool_fox measured 10 ms a frame of waits for flights for a copy
/// whose destination another step of the submission read; ours aborts the
/// same case (`CpuWrites` on the frame-rate line).
pub(super) fn lift_closure() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIFT_CLOSURE").is_some_and(|v| v == "1"))
}

/// Whether [`plan_closure`] joins the ranges of what it lifted once a round
/// and looks up the host copy of each place once a plan
/// (`VICTORY_LIFT_JOIN=1`; off by default: each step's ranges are joined
/// as it is taken, and each step looks its host copies up). The plan is the
/// same either way. Only with [`lift_closure`].
///
/// Why: cool_fox measured 16 ms of the submitting thread before a
/// submission's preparing where every fill of some 2,800 a frame sorted the
/// ranges again.
pub(super) fn lift_join() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIFT_JOIN").is_some_and(|v| v == "1"))
}

/// The most steps one plan lifts with [`lift_closure`]
/// ([`LiftAbort::TooMany`]); cool_fox saw up to this many a submission.
pub(super) const CLOSURE_MOST: usize = 12288;

/// The title memory a copy or fill reads, and the title memory it writes
/// (empty for a fill: its upload memory is the list's own).
fn reads_writes(seen: Seen) -> Option<((usize, usize), (usize, usize))> {
    match seen {
        Seen::Copy(dst, src, len) => Some(((src, src + len), (dst, dst + len))),
        Seen::Fill(_, src, len) => Some(((src, src + len), (0, 0))),
        _ => None,
    }
}

/// As [`plan_lifts`], with [`lift_closure`]: of `steps` (list, step, what it
/// is) in submission order, the copies and fills to lift, in that order, and
/// how many of them were taken beyond those that touch what flights write
/// (including steps [`plan_lifts`] would lift too, behind a lifted copy
/// whose destination they read); or why none. `join`: [`lift_join`].
///
/// First taken, as by [`plan_lifts`]: every copy that reads or writes what
/// flights write (`written`), every fill that reads it. Then, round by round,
/// every copy that writes what a taken step reads or writes, or reads what a
/// taken copy writes, and every fill that can be lifted (`fills`, `scratch`)
/// that reads what a taken copy writes. Then each taken step is made a lift
/// as [`plan_lifts`] makes it (`sole`, `scratch`, `fills` as there), and the
/// steps left on the CPU are looked at in order: a texture or a fill over
/// what flights write, or over what a taken copy ahead of it writes,
/// aborts. As cool_fox's 3e64a64 plan, which skips such fills in its rounds
/// and aborts only for one behind a taken copy.
pub(super) fn plan_closure<T: Clone, C>(
    written: &[(usize, usize)],
    steps: impl IntoIterator<Item = (usize, usize, Seen)>,
    fills: bool,
    join: bool,
    mut sole: impl FnMut(usize, usize) -> Option<(usize, T)>,
    mut scratch: impl FnMut(usize, usize, usize) -> Option<C>,
) -> Result<(Vec<Lift<T, C>>, usize), LiftAbort> {
    let steps: Vec<(usize, usize, Seen)> = steps.into_iter().collect();
    if steps.iter().any(|s| matches!(s.2, Seen::Run)) {
        return Err(LiftAbort::Run);
    }
    let mut taken = vec![false; steps.len()];
    let mut more: Vec<usize> = Vec::new();
    for (i, &(_, _, seen)) in steps.iter().enumerate() {
        let first = match seen {
            Seen::Copy(dst, src, len) => {
                touches(written, src, src + len) || touches(written, dst, dst + len)
            }
            Seen::Fill(_, src, len) => touches(written, src, src + len),
            _ => false,
        };
        if first {
            more.push(i);
        }
    }
    let seeds = more.len();
    // What taken steps read, and what taken copies write (in order, joined).
    let (mut read, mut wrote): (Vec<(usize, usize)>, Vec<(usize, usize)>) =
        (Vec::new(), Vec::new());
    let mut count = 0;
    // Per step, whether a fill can be lifted (`fills` and a scratch chunk
    // holds it), asked once.
    let mut liftable: Vec<Option<bool>> = vec![None; steps.len()];
    while !more.is_empty() {
        let (mut reads, mut writes) = (Vec::new(), Vec::new());
        for i in std::mem::take(&mut more) {
            taken[i] = true;
            count += 1;
            let Some((r, w)) = reads_writes(steps[i].2) else {
                continue;
            };
            if join {
                reads.push(r);
                writes.push(w);
            } else {
                join_ranges(&mut read, [r]);
                join_ranges(&mut wrote, [w]);
            }
        }
        if join {
            join_ranges(&mut read, reads);
            join_ranges(&mut wrote, writes);
        }
        if count > CLOSURE_MOST {
            return Err(LiftAbort::TooMany);
        }
        for (i, &(_, _, seen)) in steps.iter().enumerate() {
            if taken[i] {
                continue;
            }
            let along = match seen {
                Seen::Copy(dst, src, len) => {
                    touches(&wrote, src, src + len)
                        || touches(&wrote, dst, dst + len)
                        || touches(&read, dst, dst + len)
                }
                // A fill that cannot be lifted stays on the CPU: it is in
                // the way only behind a taken copy that writes what it
                // reads, which the walk below aborts for.
                Seen::Fill(dst, src, len) => {
                    touches(&wrote, src, src + len)
                        && *liftable[i]
                            .get_or_insert_with(|| fills && scratch(steps[i].0, dst, len).is_some())
                }
                _ => false,
            };
            if along {
                more.push(i);
            }
        }
    }
    // Each place's host copy, looked up once with `join`.
    let mut held: std::collections::HashMap<(usize, usize), Option<(usize, T)>> =
        std::collections::HashMap::new();
    let mut find = |at: usize, len: usize| {
        if !join {
            return sole(at, len);
        }
        held.entry((at, len))
            .or_insert_with(|| sole(at, len))
            .clone()
    };
    let mut lifts = Vec::with_capacity(count);
    // What taken copies ahead of the step write.
    let mut behind: Vec<(usize, usize)> = Vec::new();
    for (i, &(list, step, seen)) in steps.iter().enumerate() {
        if !taken[i] {
            match seen {
                Seen::Texture(at, len)
                    if touches(written, at, at + len) || touches(&behind, at, at + len) =>
                {
                    return Err(LiftAbort::TextureAfter);
                }
                // A fill left on the CPU that reads what a taken copy ahead
                // of it writes would read it too early.
                Seen::Fill(_, src, len)
                    if touches(written, src, src + len) || touches(&behind, src, src + len) =>
                {
                    return Err(LiftAbort::BigScratch);
                }
                _ => {}
            }
            continue;
        }
        match seen {
            Seen::Copy(dst, src, len) => {
                if src < dst + len && dst < src + len {
                    return Err(LiftAbort::InPlace);
                }
                let (from, src_copy) = find(src, len).ok_or(LiftAbort::NotSole)?;
                let (to, dst_copy) = find(dst, len).ok_or(LiftAbort::NotSole)?;
                let bounce = from == to;
                if bounce && len > BOUNCE_MOST {
                    return Err(LiftAbort::SameBuffer);
                }
                lifts.push(Lift {
                    list,
                    step,
                    src: src_copy,
                    to: LiftTo::Copy {
                        dst: dst_copy,
                        bounce,
                    },
                    len,
                });
                join_ranges(&mut behind, [(dst, dst + len)]);
            }
            Seen::Fill(dst, src, len) => {
                let chunk = fills
                    .then(|| scratch(list, dst, len))
                    .flatten()
                    .ok_or(LiftAbort::BigScratch)?;
                let (_, src_copy) = find(src, len).ok_or(LiftAbort::NotSole)?;
                lifts.push(Lift {
                    list,
                    step,
                    src: src_copy,
                    to: LiftTo::Fill(chunk),
                    len,
                });
            }
            _ => {}
        }
    }
    Ok((lifts, count - seeds))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// As `lift_tests`: copies by the thousands of their address, 9000.. none;
    /// scratch from 0x10_0000 on.
    fn sole(at: usize, len: usize) -> Option<(usize, usize)> {
        let copy = at / 1000;
        (copy < 9 && (at + len - 1) / 1000 == copy).then_some((copy, copy))
    }

    fn scratch(_list: usize, at: usize, _len: usize) -> Option<usize> {
        at.checked_sub(0x10_0000)
    }

    /// The steps lifted (by place), and how many were taken along, with
    /// ranges joined each way: both give the same.
    fn planned(
        written: &[(usize, usize)],
        steps: &[Seen],
    ) -> Result<(Vec<usize>, usize), LiftAbort> {
        let run = |join| {
            let steps = steps.iter().enumerate().map(|(i, &seen)| (0, i, seen));
            plan_closure(written, steps, true, join, sole, scratch)
                .map(|(lifts, along)| (lifts.iter().map(|l| l.step).collect::<Vec<_>>(), along))
        };
        let joined = run(true);
        assert_eq!(joined, run(false), "{steps:?}");
        joined
    }

    const W: [(usize, usize); 1] = [(1000, 1100)];

    /// Without dependencies the closure lifts what `plan_lifts` lifts.
    #[test]
    fn apart_it_lifts_what_the_walk_lifts() {
        let steps = [
            Seen::Copy(2000, 1000, 16),
            Seen::Copy(5000, 4000, 16),
            Seen::Copy(3000, 2008, 8),
            Seen::Copy(1050, 6000, 4),
            Seen::Fill(0x10_0100, 3000, 8),
            Seen::Fill(0x10_0200, 7000, 8),
            Seen::Other,
        ];
        assert_eq!(planned(&W, &steps), Ok((vec![0, 2, 3, 4], 2)));
        let walk = plan_lifts(
            &W,
            steps.iter().enumerate().map(|(i, &s)| (0, i, s)),
            true,
            sole,
            scratch,
        )
        .expect("a plan");
        let walked: Vec<usize> = walk.iter().map(|l| l.step).collect();
        assert_eq!(walked, [0, 2, 3, 4]);
        // Ends that touch are apart.
        let steps = [
            Seen::Copy(2000, 1000, 100),
            Seen::Copy(1900, 9000, 100),
            Seen::Copy(2100, 9000, 5),
            Seen::Copy(900, 5000, 100),
        ];
        assert_eq!(planned(&W, &steps), Ok((vec![0], 0)));
    }

    /// What `plan_lifts` aborts for as a CPU copy in the way is taken along,
    /// ahead and behind; so is what reads what a taken copy writes.
    #[test]
    fn what_is_in_the_way_is_taken_along() {
        // A copy on the CPU writes a lifted copy's source, before or after.
        let ahead = [Seen::Copy(2000, 5000, 8), Seen::Copy(1000, 2000, 8)];
        let behind = [Seen::Copy(3000, 1096, 16), Seen::Copy(1100, 6000, 4)];
        for steps in [&ahead[..], &behind[..]] {
            let walk = steps.iter().enumerate().map(|(i, &s)| (0, i, s));
            assert_eq!(
                plan_lifts(&W, walk, true, sole, scratch),
                Err(LiftAbort::CpuWrites)
            );
            assert_eq!(planned(&W, steps), Ok((vec![0, 1], 1)));
        }
        let taken = Seen::Copy(2000, 1000, 100);
        // Reads what it writes, ahead of it too.
        assert_eq!(
            planned(&W, &[Seen::Fill(0x10_0000, 2099, 8), taken]),
            Ok((vec![0, 1], 1))
        );
        assert_eq!(
            planned(&W, &[Seen::Copy(7000, 2050, 10), taken]),
            Ok((vec![0, 1], 1))
        );
        // Writes what it writes.
        assert_eq!(
            planned(&W, &[taken, Seen::Copy(2098, 8000, 2)]),
            Ok((vec![0, 1], 1))
        );
        // A chain, round by round, with steps apart in between.
        let steps = [
            Seen::Copy(8000, 7000, 50),
            taken,
            Seen::Copy(3000, 2010, 20),
            Seen::Fill(0x10_0000, 6000, 10),
            Seen::Fill(0x10_0100, 3000, 20),
            Seen::Copy(4000, 3005, 5),
            Seen::Fill(0x10_0200, 4000, 5),
        ];
        assert_eq!(planned(&W, &steps), Ok((vec![1, 2, 4, 5, 6], 4)));
        // A fill is in nobody's way; nor a copy that shares a source.
        assert_eq!(
            planned(&W, &[taken, Seen::Fill(0x10_0000, 1000, 100)]),
            Ok((vec![0, 1], 0))
        );
        assert_eq!(
            planned(&W, &[Seen::Copy(7000, 1000, 10), taken]),
            Ok((vec![0, 1], 0))
        );
        assert_eq!(
            planned(&W, &[Seen::Copy(7000, 5000, 10), Seen::Copy(2000, 1000, 8)]),
            Ok((vec![1], 0))
        );
    }

    /// What cannot be lifted still aborts where it is in the way; a texture
    /// only behind the copy that writes its memory.
    #[test]
    fn what_cannot_be_lifted_aborts() {
        let taken = Seen::Copy(2000, 1000, 100);
        assert_eq!(planned(&W, &[taken, Seen::Run]), Err(LiftAbort::Run));
        assert_eq!(
            planned(&W, &[Seen::Texture(1050, 4)]),
            Err(LiftAbort::TextureAfter)
        );
        assert_eq!(
            planned(&W, &[taken, Seen::Texture(2050, 4)]),
            Err(LiftAbort::TextureAfter)
        );
        assert_eq!(
            planned(&W, &[Seen::Texture(2050, 4), taken]),
            Ok((vec![1], 0))
        );
        // A fill that cannot be lifted, behind the copy that writes what it
        // reads; or reading what flights write. Ahead of the copy it stays on
        // the CPU.
        assert_eq!(
            planned(&W, &[taken, Seen::Fill(0x0f_0000, 2050, 4)]),
            Err(LiftAbort::BigScratch)
        );
        assert_eq!(
            planned(&W, &[Seen::Fill(0x0f_0000, 1050, 4)]),
            Err(LiftAbort::BigScratch)
        );
        assert_eq!(
            planned(&W, &[Seen::Fill(0x0f_0000, 2050, 4), taken]),
            Ok((vec![1], 0))
        );
        // Into itself; across two copies.
        assert_eq!(
            planned(&W, &[Seen::Copy(1010, 1000, 20)]),
            Err(LiftAbort::InPlace)
        );
        assert_eq!(
            planned(&W, &[taken, Seen::Copy(9500, 2000, 4)]),
            Err(LiftAbort::NotSole)
        );
        // Too many.
        let many: Vec<Seen> = (0..=CLOSURE_MOST)
            .map(|i| Seen::Fill(0x10_0000 + i, 1000, 4))
            .collect();
        assert_eq!(planned(&W, &many), Err(LiftAbort::TooMany));
    }

    /// Whatever the steps: the CPU's steps in order, then the lifted ones in
    /// order, read and leave what all of them in order read and leave.
    #[test]
    fn lifted_after_the_cpu_does_what_the_order_does() {
        let mut seed = 12345u32;
        let mut random = move |n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (seed >> 16) % n
        };
        // All in one host copy (copies bounce), fills into scratch or into
        // memory the card cannot write.
        let one = |_: usize, _: usize| Some((0, ()));
        let into = |_: usize, at: usize, _: usize| (at >= 0x10_0000).then_some(());
        let (mut done, mut aborted, mut along) = (0, 0, 0);
        for _ in 0..4000 {
            let from = random(60) as usize;
            let written = [(from, from + 1 + random(6) as usize)];
            let steps: Vec<Seen> = (0..2 + random(9))
                .map(|_| {
                    let (at, other, len) = (
                        random(60) as usize,
                        random(60) as usize,
                        1 + random(8) as usize,
                    );
                    match random(8) {
                        0..4 => Seen::Copy(at, other, len),
                        4..6 => Seen::Fill(0x10_0000, at, len),
                        6 => Seen::Fill(0, at, len),
                        _ => Seen::Texture(at, len),
                    }
                })
                .collect();
            let plan = plan_closure(
                &written,
                steps.iter().enumerate().map(|(i, &s)| (0, i, s)),
                true,
                random(2) == 0,
                one,
                into,
            );
            let Ok((lifts, n)) = plan else {
                aborted += 1;
                continue;
            };
            done += 1;
            along += n;
            let reads = |s: &Seen| match *s {
                Seen::Texture(at, len) => (at, at + len),
                s => reads_writes(s).unwrap().0,
            };
            let start: Vec<u32> = (0..80).collect();
            let mut memory = start.clone();
            let mut wanted = Vec::new();
            for s in &steps {
                let (a, b) = reads(s);
                wanted.push(memory[a..b].to_vec());
                if let Seen::Copy(dst, src, len) = *s {
                    memory.copy_within(src..src + len, dst);
                }
            }
            let lifted = |i: usize| lifts.iter().any(|l| l.step == i);
            let mut cpu = start.clone();
            for (i, s) in steps.iter().enumerate().filter(|&(i, _)| !lifted(i)) {
                let (a, b) = reads(s);
                // What flights write is not read on the CPU.
                assert!(!touches(&written, a, b), "step {i} of {steps:?}");
                assert_eq!(cpu[a..b], wanted[i][..], "step {i} of {steps:?}");
                if let Seen::Copy(dst, src, len) = *s {
                    cpu.copy_within(src..src + len, dst);
                }
            }
            let mut card = cpu;
            for lift in &lifts {
                let s = &steps[lift.step];
                let (a, b) = reads(s);
                assert_eq!(
                    card[a..b],
                    wanted[lift.step][..],
                    "step {} of {steps:?}",
                    lift.step
                );
                if let Seen::Copy(dst, src, len) = *s {
                    card.copy_within(src..src + len, dst);
                }
            }
            assert_eq!(
                card,
                memory,
                "{steps:?}: {:?}",
                lifts.iter().map(|l| l.step).collect::<Vec<_>>()
            );
        }
        assert!(
            done > 500 && aborted > 100 && along > 100,
            "{done} {aborted} {along}"
        );
    }
}
