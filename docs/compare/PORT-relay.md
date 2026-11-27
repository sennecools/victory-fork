# Port lane: relay (work/cf-relay)

Base: `work/r5-s2` (ff27257). Items #6 (the card-steps closure, on top of
our `VICTORY_R5_LIFT`) and #7 (`host/awake.rs`, `VICTORY_CARD_AWAKE`) of
`docs/compare/cool-fox.md`, hand-ported from cool_fox onto our lift
planning and our `Host`. No cool_fox history is merged or cherry-picked.
Nothing here is XCL-derived. Every new behaviour is behind its own switch,
off by default; with every switch unset, behaviour is as on the base.

> **2026-11-27:** `VICTORY_R5_LIFT`, `crate::shadow::sole_or_home` and
> `over_lifted_homes` were removed. With `VICTORY_R5=1` a refresh over a
> home a flight writes is always taken on the card (H2), and a lifted copy
> or fill (H8, `VICTORY_LIFT_READS`) never reads from or writes into a
> home. What this file says about them is history.

## What the lane ports

All cool_fox commits named here have the author `local`.

| Ours | Item | What | From cool_fox |
|---|---|---|---|
| ff22454 | #6 | `host/lift_closure.rs`: `plan_closure`, called by `lift_reads` instead of `plan_lifts` when the closure is on. Copies and fills in the way of a lifted step are taken along, round by round. Uses our `sole`/`sole_or_home`, `scratch_chunk`, bounce buffers and the R5 `over_lifted_homes` check. New `LiftAbort::TooMany` (more than 12,288 steps), `LIFT_ALONG` stat, startup log fields, README rows | cc03ecb (steps done on the card), 007fe55 (dependents taken along), 3e64a64 (ranges joined once a round, each place looked up once) |
| 3674c1b | #6 | `stale_under_flights` keeps every stale span (budget 4,096 runs, not 512); `Host::settle_stale` waits only for the flights that may write them (`writes_pending_over`), asks the write watch again, up to 3 times, then settles as before | 007fe55 (the waits under shader writes) |
| 93e5b48 | #6 | Upload buffers of buffer-to-texture copies (`list.rs`) made in shared memory and kept in `Deferred::shared_uploads`; `scratch_chunk` finds them, so their fills can be lifted | 007fe55 (`FillTo::Buffer`, shared memory for the list's uploads) |
| b58ae77 | #7 | `host/awake.rs`: own compute queue, one 64-thread counting group (HLSL, `cs_5_1`, our `host::compile`), two sends at most on the way, while a late-path submission went out in the last 250 ms. Hook after the late path's `ExecuteCommandLists` and `Signal`, before `card_sent()` | 5f86fa3 |
| fdd0b64 | review | `over_lifted_homes` finds each lift's step by list start + place, not a walk over all steps (falls back to the walk on a mismatch); same result | ours |
| b6e4555 | review | The closure takes a fill only if it can be lifted; an untaken fill aborts (`BigScratch`) only over flights' writes or behind a taken copy that writes its source, as cool_fox 3e64a64 does. `LIFT_ALONG` docs corrected (it counts every step beyond the seeds, including some `plan_lifts` lifts too) | 3e64a64 (behaviour checked against it) |
| 2985292 | review | `VICTORY_LIFT_STALE` widens the look only with `relay_exact` (`VICTORY_RELAY=2`), as documented | ours |
| 426bdff | review | `awake.rs` comments: a menu keeps the card awake, a title that stands still goes quiet; the thread `sent()` starts is detached for the life of the process. Comments only | 040aa27 (wording) |
| fd77ae0 | docs | First version of this file | ours |
| 1868a5c | fix | With `VICTORY_CARD_AWAKE`, a free of the title's refused for a heap while card-awake sends were on the way holds the sends (`awake::hold`), waits for those on the way and asks again (3 tries, 1 ms apart): on WARP the sends kept heaps let go, and frees waited for a sweep | ours |
| 66d1aae | fix | `awake::end()`: no card-awake send after it, none on the way, the thread ends; for the release at the title's end (not called on this branch, see Known gaps) | ours |

## Switches

All are read once per process from the environment (`OnceLock`), so a
test cannot toggle them; set them on the test process.

| Switch | Default | Values | What it does | Depends on |
|---|---|---|---|---|
| `VICTORY_LIFT_CLOSURE` | off | `1` on; anything else off | Copies that write what a lifted step reads or writes, or read what a lifted copy writes, and liftable fills that read what a lifted copy writes, are lifted too, round by round, ahead or behind, instead of the plan aborting with "a CPU copy writes it". Recorded in step order, each step between its own barriers. Aborts: a texture behind a lifted copy over its memory, a step of ours, an unliftable fill over flights' writes or behind a taken copy, more than 12,288 steps ("too many steps taken along") | Lifts made: `VICTORY_LIFT_READS=1`/`2` or `VICTORY_R5_LIFT=1`, which need `VICTORY_RELAY=2`. With `VICTORY_LIFT_READS=count` it plans and counts only |
| `VICTORY_LIFT_JOIN` | off | `1` on | The closure joins its ranges once a round and looks each place's host copy up once a plan. Same plan; only planning time differs | `VICTORY_LIFT_CLOSURE=1` |
| `VICTORY_LIFT_STALE` | off | `1` on | Every stale span under flights' writes is kept (4,096 runs, not 512); `settle_stale` waits only for the flights that may write them, re-asks the write watch, looks again, up to 3 times, then settles everything as before | `VICTORY_RELAY=2` (`relay_exact`); without it nothing changes |
| `VICTORY_LIFT_UPLOADS` | off | `1` on | Texture upload buffers lie in shared memory (custom heap, write-back, L0) and their fills can be lifted like scratch-chunk fills, instead of aborting with "a fill into upload memory". The zeros of new texture levels stay in upload memory | Fills lifted: `VICTORY_LIFT_READS=2` or `VICTORY_R5_LIFT=1` (checked on every call) |
| `VICTORY_CARD_AWAKE` | off | `1` on; `turns` 20 s on / 20 s off, logged at each turn; anything else off | While a late-path submission went out in the last 250 ms, one group of 64 threads counts (400,000 multiply-adds, about 2 ms at full clocks per cool_fox) on its own compute queue, so the card does not drop its clocks between submissions | The late path (`VICTORY_LATE_WAIT`, the relay); with every submission waited for it does nothing |

## Build status here

Linux, target `x86_64-pc-windows-msvc`, `CARGO_FEATURE_PURE=1` (only so
blake3's build script skips MSVC assembly), `--locked`, own target dir.
Run on the lane's final code (before this docs commit):

- `cargo check --target x86_64-pc-windows-msvc --workspace --all-targets`:
  passes; one warning, the base's unused `depth_written` in `indirect.rs`.
- `cargo check --target x86_64-pc-windows-msvc -p victory --features diagnostics --all-targets`:
  passes, no warnings.
- `cargo clippy --target x86_64-pc-windows-msvc --workspace --all-targets`:
  the base's four warnings only (`texture.rs` precedence, `draw.rs`
  sort, `shadow.rs` collapsible if, `indirect.rs` `depth_written`), none
  in touched code.
- `cargo test -p victory-gpu` does not build on Linux (Windows-only
  modules), as on the base. **All tests in this lane were only
  type-checked, none ran in the crate.**
- The pure planning code (`touches`, `join_ranges`, `Seen`, `LiftAbort`,
  `Lift`/`LiftTo`, `plan_lifts`, `lift_closure.rs`) was copied into a
  scratch crate and its four `lift_closure` tests ran on Linux and passed.
  Removing one closure rule or the in-order texture check made the
  randomized test fail. That copy is not this crate; rerun them on the PC.

## Tests to run on the PC

1. Unit tests (pure, any Windows build):
   - `host::lift_closure::tests`: `apart_it_lifts_what_the_walk_lifts`,
     `what_is_in_the_way_is_taken_along`, `what_cannot_be_lifted_aborts`,
     `lifted_after_the_cpu_does_what_the_order_does` (each checks join on
     and off give the same plan).
   - `host::awake::tests`: `off_unless_asked`,
     `wanted_only_shortly_after_a_submission`,
     `by_turns_on_in_odd_turns_only`.
   - Touched, already there: `host::lift_tests`, `host::late_lift::tests`,
     `host::relay_tests`, `host::late_tests`, `shadow::relay_tests`.

   ```
   cargo test -p victory-gpu --locked host::lift_closure
   cargo test -p victory-gpu --locked host::awake
   cargo test -p victory-gpu --locked host::lift_tests host::late_lift host::relay_tests host::late_tests shadow::relay_tests
   ```
2. WARP tests, `VICTORY_D3D12_WARP=1`, `--features diagnostics`,
   `--test-threads=1`: `home::tests::r5_lift` (all `warp_r5_*`) and
   `home::tests::warp_core_refuses_with_late_waits`. These set R5, the
   core, lifts and the relay themselves (`TestFlights`); the new switches
   come from the environment. Run once with no new switch (must match the
   base), then once per line:
   - `VICTORY_LIFT_CLOSURE=1`
   - `VICTORY_LIFT_CLOSURE=1 VICTORY_LIFT_JOIN=1`
   - `VICTORY_LIFT_STALE=1`
   - `VICTORY_LIFT_UPLOADS=1`, then with `VICTORY_LIFT_CLOSURE=1`
   - `VICTORY_CARD_AWAKE=1`, then `turns` (no "the card is not kept
     awake" warning; clean exit)
   - all of the above together
3. Game runs, A/B against the base with `VICTORY_RELAY=2
   VICTORY_R5_LIFT=1` (and its core switches) and
   `VICTORY_FRAMETIME_DUMP=1`. One switch set at a time, in this order,
   then all combined:
   1. `VICTORY_LIFT_CLOSURE=1`
   2. `+ VICTORY_LIFT_JOIN=1`
   3. `VICTORY_LIFT_UPLOADS=1` (alone, then with the closure)
   4. `VICTORY_LIFT_STALE=1`
   5. `VICTORY_CARD_AWAKE=1`; then `turns` in one run (compare the on and
      off turns)
   6. all of them

   Measure, from the frametime CSV: `ms_per_frame`, `frame_ms_max`,
   `frame_ms_avg`, `cpu_prepare_ms`, `gpu_ms`, `gpu_pct`, `copies_ms`,
   `textures_ms`, `card_waits`, `card_wait_ms`. From the frame-rate lines:
   the lift line (plans without an abort, copies and fills lifted, "waits
   for flights to land left out", the "aborted:" reasons: "a CPU copy
   writes it" should drop with the closure, "a fill into upload memory"
   with the uploads, watch for "not one copy" and "too many steps taken
   along" growing; "steps taken beyond those over flights' writes"), the
   settles line (ms waiting in settles, flights each time) and its
   by-reason line ("title memory changed under shader writes" should drop
   with `LIFT_STALE`), and the title-bytes-under-writes line. For the
   closure and join, compare `cpu_prepare_ms` with and without the join.

## Risks to check on the PC

- **Uploads in shared memory** (`VICTORY_LIFT_UPLOADS`): a different heap
  kind on the card's side than upload memory, unmeasured on the RX 9070
  XT. Check textures fed through these buffers are right on WARP and the
  card, and that `textures_ms` does not grow.
- **Waits in `settle_stale` under the submission lock**
  (`VICTORY_LIFT_STALE`): the submitting thread waits for flights up to
  three times. A waited span is assumed to clear once its flight lands;
  if it does not, the three rounds pass and it settles anyway. Look for
  longer submit times, stalls, and title memory a submission late.
- **Planning cost of R5_LIFT plus the closure**: up to 12,288 lifts a
  plan, planned on the submitting thread under the submission lock.
  Compare `cpu_prepare_ms` with and without `VICTORY_LIFT_JOIN=1`.
- **More card work at the head of a submission**: more copies, each
  between its own barriers, behind flights. Lifted copies need one host
  copy or home, so aborts may move to "not one copy".
- **Order of lifted steps**: recorded in step order. A wrong result shows
  as bytes from before a copy. The randomized test checks the plan's
  order, not the recording; the WARP tests check the recording.
- **Card awake**: the thread is detached for the life of the process and
  keeps a device reference; it is killed at exit. Check exit, device
  removal or TDR, power and heat, and whether the 400,000-step loop is
  the right length on the target card. Frame times in `turns` should
  differ between turns.
- **Card awake and the title's frees**: on WARP the sends keep a heap the
  host let go until the device's queues at work ran on past it, so a
  decommit of a home's pages was refused (and waited for a sweep) while
  they ran. A refused free now holds the sends and waits for those on the
  way (some 2 ms a send on the card, two on the way) before it waits as
  before; a free that waits for a flight is tried again at landings
  without a hold. Whether the card's driver keeps heaps the same way is
  unknown: in an A/B with the switch, compare the memory core's line
  (frees waited, ms waited, given up, protections refused) with the
  switch off, and the title thread's frame time spikes.

## Known gaps

- Not ported, by choice: constant-buffer slot repointing (cc03ecb,
  30f7859; no `VICTORY_LIFT_CB_SLOT`): our lifted fills go to shared
  scratch chunks every constant-buffer fill fits, and rewriting a
  descriptor of a list that may still be on the card is a race. The owner
  can ask for it.
- Not ported, not applicable: `VICTORY_PLACED_ORDER` (we have no copies
  done at their place in a list); cool_fox's level-batched recording (step
  order is valid for the closure); the `watch::landing` re-checks (our
  `card_merge`/`card_merged` already tell the guards; cool_fox's feed only
  its `VICTORY_FILL_TITLE`); its replay options (`--awake`,
  `--no-restore`; we have no replay tool).
- Not ported: cool_fox's diagnostics for these items (blame, wide_write,
  why a span stayed stale).
- `VICTORY_CARD_AWAKE` hooks only the late path's submissions, as
  cool_fox does; waited submissions and other `ExecuteCommandLists` sites
  are not hooked.
- `crate::host::ending()` and `end_submissions` are not on this branch.
  Merged with the release (`crate::release`, R3 line), `end_submissions`
  in `host/ending.rs` must call `awake::end()` after it sets `ENDING`;
  until then the card-awake queue may send for up to 250 ms after the
  release (on its own buffer, nothing of the title's). Its `ENDED` is
  not read from `ending()`, which is per thread under test there.
- No unit tests for `VICTORY_LIFT_STALE`, `VICTORY_LIFT_UPLOADS` (cool_fox
  has none) or the `over_lifted_homes` index (needs real shadows): WARP
  tests and game runs only.
- Formatting: the base is not rustfmt-clean (`cargo fmt --all` rewrites
  about 60 files), so it was not run over the tree. New files pass
  `rustfmt --edition 2024 --check` with our `rustfmt.toml`; hunks in
  existing files were written in rustfmt style by hand.
- Review findings: all six were fixed (fdd0b64, b6e4555, 2985292,
  426bdff, fd77ae0). None open.
- #13 (submit thread, thin lists) is not in this lane.
