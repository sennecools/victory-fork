# Port lane: relay (work/cf-relay)

Base: `work/r5-s2`. Items #6 (the card-steps closure, on top of our
`VICTORY_R5_LIFT`) and #7 (`host/awake.rs`) of `docs/compare/cool-fox.md`,
hand-ported from cool_fox. Each commit names the cool_fox commits it took
from. Nothing here is XCL-derived.

## Switches

All are off by default. With every one unset, behaviour is as on the base.

| Switch | Needs | What it does |
|---|---|---|
| `VICTORY_LIFT_CLOSURE=1` | lifts made: `VICTORY_LIFT_READS=1`/`2` or `VICTORY_R5_LIFT=1` | copies and fills that depend on a lifted read (or that it depends on) are lifted too, round by round, instead of the plan aborting ("a CPU copy writes it"); up to 12,288 steps a plan |
| `VICTORY_LIFT_JOIN=1` | `VICTORY_LIFT_CLOSURE=1` | the closure joins its ranges once a round and looks each place's host copy up once; the plan is the same |
| `VICTORY_LIFT_STALE=1` | `VICTORY_RELAY=2` | refreshes of title memory changed under flights' writes look at every stale span (4,096 runs instead of 512) and wait only for the flights that may write them, up to three times, before settling everything as before |
| `VICTORY_LIFT_UPLOADS=1` | fills lifted: `VICTORY_LIFT_READS=2` or `VICTORY_R5_LIFT=1` | the upload buffers of buffer-to-texture copies are made in shared memory, so their fills can be lifted like fills of scratch chunks |
| `VICTORY_CARD_AWAKE=1` / `turns` | submissions sent ahead (`VICTORY_LATE_WAIT`, the relay) | a compute queue of its own keeps the card busy while a submission went out in the last 250 ms; `turns`: 20 s on, 20 s off, the log says when each turn starts |

## Tests to run on the PC

Here the tests were only type-checked for `x86_64-pc-windows-msvc`. The
pure planning tests in `host/lift_closure.rs` were also run on Linux, in a
scratch copy of the planning code, and passed. Nothing else was run.

1. Unit tests, Windows: `cargo test -p victory-gpu lift_closure`,
   `cargo test -p victory-gpu awake`, and `lift_tests`, `late_lift`,
   `relay_tests` and `late_tests` in the same crate, which this lane
   touches.
2. WARP relay tests (`VICTORY_D3D12_WARP=1`, `VICTORY_RELAY=2`): run them
   once with no new switch set (this must match the base), then once with
   each switch on alone:
   - `VICTORY_LIFT_READS=2` with `VICTORY_LIFT_CLOSURE=1`, then also with
     `VICTORY_LIFT_JOIN=1`;
   - `VICTORY_R5=1 VICTORY_R5_LIFT=1 VICTORY_LIFT_CLOSURE=1` (with
     `VICTORY_MEMORY_CORE=1`);
   - `VICTORY_LIFT_STALE=1`;
   - `VICTORY_LIFT_READS=2 VICTORY_LIFT_UPLOADS=1`, with and without the
     closure;
   - `VICTORY_CARD_AWAKE=1`, then `turns`.
3. Game runs, with the same sets. On the frame-rate line, look at the lift
   line's aborts (fewer "a CPU copy writes it" with the closure), its
   "steps taken beyond those over flights' writes", and the waits for
   flights.

## Risks to check on the PC

- **Uploads in shared memory** (`VICTORY_LIFT_UPLOADS`): this is a
  different heap kind on the card's side from upload memory. Check that
  textures fed from these buffers come out right on WARP and on the card,
  and that the time of the copies into textures does not grow.
- **Waits in `settle_stale` under the submission lock**
  (`VICTORY_LIFT_STALE`): the submitting thread waits for flights up to
  three times and asks the write watch again each time. Look for longer
  submit times or stalls, and for title memory that reaches the card a
  submission late.
- **Cost of R5_LIFT plus the closure**: with `VICTORY_R5_LIFT=1` and
  `VICTORY_LIFT_CLOSURE=1` a plan can hold up to 12,288 lifts, all planned
  on the submitting thread. Compare the submitting thread's time with and
  without `VICTORY_LIFT_JOIN=1`. Since the review, the R5 check of lifted
  homes no longer walks every step for each lift.
- **Order of lifted steps**: the lift list records the taken steps in
  submission order, each between its own barriers. A wrong result shows up
  as bytes from before a copy. The randomised planning test checks the
  plan's order, not the recording.
- **Card awake**: the counting thread is detached and lives until the
  process ends. Check device removal and exit with it on, the power draw,
  and that frame times with `turns` really differ between the turns.
