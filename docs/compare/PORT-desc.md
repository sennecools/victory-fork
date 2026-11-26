# Port lane `desc`: descriptor heaps from cool_fox

Branch `work/cf-desc`, from `work/r5-heapsync` (9750f9b). Items #1 to #3 of
the ranked list in `docs/compare/cool-fox.md` (branch
`work/cool-fox-compare`). Nothing here is XCL-derived. COPIED_GRACE is not
ported, as asked.

Every new switch is off by default. With all of them off, the code runs the
paths of `work/r5-heapsync`. The one difference: in the heap buffer pass the
device-removed check every 64 slots now runs after the copy lookup, and in a
few more cases. Only the log can change.

## What the lane ports

All cool_fox commits named here are by `local`, except 18bdf79 by `codec_d`.

| # | Our commit | What | From cool_fox |
| --- | --- | --- | --- |
| 1 | cb9245a | Quick mirror at every mirror site; unlocked first look in `texture_slot_made_from`; buffer-view and changed sets as atomic bits (`Slots`) | 53c5256 (quick mirror), 6c17ce9 (unlocked look), e4d0837 (the stale quick record: ours has no second record, see below), c3e643e (bit sets only) |
| 1 | 2c86826 | 512 striped locks taken by each slot claim | 6c17ce9 (its `Made` with 512 locks) |
| 1 | ff15ef5 | Review fix: the quick look waits for a claimed slot | - |
| 2 | bf712b3 | Heap buffer pass: walk on its own threads (256 slots a job), spread round, tabled skip, sweep state on the HostHeap | 6c17ce9 (walk in jobs), c3e643e (threads of its own), 2aa8134 (spread), 1f1a433 (`copied_tabled`), 18bdf79 (sweep state on the heap) |
| 2 | ea442e7 | Review fix: the pass's table marks are its own (8 frames), not the sync's 240-frame `tabled` | - |
| 3 | e056274 | Heap sync: records kept as bytes, 16-record memcmp, pages sorted and slots mirrored on threads, `shows_none` | 1f1a433 (byte copy), 53c5256 (16 at a time), 44629de (pages 8 a job, slots 256 a job, `shows_none`) |
| 3 | 4108c07 | Early rounds of the sync helper between submissions | f6db479 |
| 3 | ab2f1ee | `copy_simple` leaves a copy onto the same bytes unwritten | e8d3d91 (its VICTORY_SAME_RECORDS check only) |
| 3 | 2d3eb68 | Review fix: the same-records test sets its switch per thread | - |

Notes on adaptation:

- e4d0837: cool_fox clears a separate `quick[]` record when a slot flips.
  Ours reads `made` itself, which a claim stores last with Release, after
  the descriptor, stride word and kept resource. There is no second record
  to go stale, so the fix is built in.
- Stripes (history: `VICTORY_MADE_ATOMIC` and `VICTORY_MADE_STRIPES` were
  removed in change 1f, 26 Nov 2026; the per-slot claim is now the only
  design and the stripes measured slower). Ours has no heap-wide key array
  to stripe. The per-slot claim of `VICTORY_MADE_ATOMIC` already replaced
  the heap-wide lock, so `VICTORY_MADE_STRIPES` built a slot-at-a-time heap (NoteLog and
  `spanned` ordering kept) and each claim takes its stripe lock
  (`index % 512`) before the CAS. cb9245a first left the stripes out; the
  review flagged that as an unapproved scope change and 2c86826 ports them.
  Compare on the PC: ATOMIC claims spin on the slot, STRIPES claims sleep on
  the stripe.
- Pass threads: no land.rs pool here. `draw::each_on` uses
  `std::thread::scope` per walk; handles are worked out before the threads
  start, and the binding thread applies results in slot order.
- Byte copy: cool_fox takes a changed record as last seen before its tabled
  check. Ours checks first, as our prints do, so the same slots are made.
- Tabled skip: cool_fox gates it on VICTORY_HEAP_SYNC; ours has its own
  switch.
- Early rounds: on our VICTORY_HEAP_SYNC_AHEAD helper, not cool_fox's
  HEAP_SYNC_ASIDE. 4 ms and on by default there; off here. (History: since
  S9 of change 1, 26 Nov 2026, the helper is the only path and the early
  rounds always run, 4 ms after the last; see the note under Switches.)

## Switches

Since S9 of change 1 (26 Nov 2026) the heap sync's switches of this lane and
the two it worked with are removed: `VICTORY_HEAP_SYNC_BYTES`,
`VICTORY_HEAP_SYNC_THREADS`, `VICTORY_HEAP_SYNC_EARLY`,
`VICTORY_HEAP_SYNC_AHEAD` and `VICTORY_HEAP_SYNC_FAST`. What they turned on
is now the only path, as the play set ran it: the sync runs on its helper
thread (`descriptor/ahead.rs`) and the submission waits on its request's
number before its lists go to the card (in place, at that wait, only if the
helper cannot run); the helper runs passes of its own 4 ms after the last
(`EARLY`); the records are kept as bytes and compared 16 at a time
(`HostHeap::copied_begin`); a heap's changed pages are sorted and its slots
mirrored on 4 threads (`SYNC_THREADS` in `descriptor.rs`); and a record
whose slot is made from it already is left before the slot's claim
(`SKIPPED_MADE`). The sync also looks only at the heaps a submission's
lists bind. Their rows below are kept as the record of the port.
`VICTORY_SHOWS_NONE` is still a switch.

| Switch | Default | Values | What it does when on | Depends on | Read in |
| --- | --- | --- | --- | --- | --- |
| `VICTORY_QUICK_MIRROR` | off | `1` on | Every mirror (table binds, pass walk, texture pass, title writes, sync) of a slot whose `made` already holds the record's key, view still there, returns before the claim. `texture_slot_made_from` answers "no" early for a settled slot on another key (not for a claimed one). Skips counted in `QUICK_MIRRORED`. | - | `descriptor/host.rs` `quick_mirror` |
| `VICTORY_SLOT_BITS` | off | `1` on | Buffer-view set and changed set become atomic u64 bit sets, changed under the slot's claim with no lock. | The buffer half of NoteLog is bypassed too | `descriptor/host.rs` `slot_bits` |
| `VICTORY_HEAP_PASS_THREADS` | unset (1) | `0`/`1` one thread; `2`..`16` (clamped) | Pass walk in jobs of 256 slots on N scoped threads, the binding one among them. A walk of 256 slots or fewer stays on one thread. | Gains need `QUICK_MIRROR` (per-slot claims are always on since 1f) | `draw.rs` `heap_pass_threads` |
| `VICTORY_HEAP_SPREAD` | off | `1` on | Each pass looks at the changed slots plus one of 16 parts of the rest; first pass looks at all. Replaces the every-16th full pass. | - | `draw.rs` `heap_spread` |
| `VICTORY_HEAP_PASS_TABLED` | off | `1` on | Draws mark their tables' buffer slots (`table_entry`; a fresh table's slots if it has 64 or fewer, else the named ones) in `buffer_tabled`; the pass skips slots marked in the last 8 frames. Skips in `HEAP_PASS_PARTS[2]`, logged per frame when nonzero. The sync and texture queue do not read these marks. | - | `draw.rs` `heap_pass_tabled` |
| `VICTORY_HEAP_SWEEP_OWN` | off | `1` on | Pass state on the HostHeap under its own lock, dropped with the heap, not in the global `HEAP_RANGES` mutex. | - | `draw.rs` `heap_sweep_own` |
| `VICTORY_HEAP_SYNC_BYTES` (removed in S9: always) | off | `1` on | Sync keeps the records' bytes (32 B per view slot, made on first use) instead of an 8 B print, and compares 16 records (512 B) at a time. | - | (was `descriptor/host.rs` `sync_bytes`) |
| `VICTORY_HEAP_SYNC_THREADS` (removed in S9: always 4) | unset (1) | `0`/`1` one thread; `2`..`16` (clamped) | Sync reads, compares and sorts a heap's changed pages 8 a job, then mirrors the slots 256 a job, on N scoped threads (`each_on`). Slots are mirrored out of slot order, each ordered by its claim. | Keeps `VICTORY_HEAP_SYNC_FAST`'s early skip | (was `descriptor/host.rs` `sync_threads`) |
| `VICTORY_SHOWS_NONE` | off | `1` on | A changed record that is no view of ours, into a slot with `made` 0 and not claimed, is not mirrored. Counted in `SHOWED_NONE` (frame line, `shows_none` arg on `sync heaps`). | - | `descriptor/host.rs` `shows_none` |
| `VICTORY_HEAP_SYNC_EARLY` (removed in S9: always 4 ms) | unset (0: none) | `<ms>` > 0 | The sync helper runs passes of its own after `<ms>` with no request, while the last request is under 500 ms old. They answer no request. Profile: `sync heaps early`. | `VICTORY_HEAP_SYNC_AHEAD=1`, else no effect | (was `descriptor/ahead.rs` `early`) |
| `VICTORY_SAME_RECORDS` | off | `1` on | `copy_simple` (and `copy`) skip the write when the destination already holds the source's bytes; records are still mirrored. | - | `descriptor.rs` `same_records` |

Any of SYNC_BYTES, SYNC_THREADS or SHOWS_NONE took the new sync path
(`sort_copied_page` for all pages, then `mirror_synced`). Existing switches
these worked with: `VICTORY_HEAP_SYNC_AHEAD`,
`VICTORY_HEAP_SYNC_FAST`. (History: since S9 that path is the only one and
those switches are gone, see above.)

## Built here

Linux, Windows target, `CARGO_FEATURE_PURE=1` (blake3 build script only),
own target dir. Run by the lane at 2d3eb68 (this commit adds only this file):

- `cargo check --locked --target x86_64-pc-windows-msvc --workspace --all-targets`: passes.
- `cargo clippy --locked --target x86_64-pc-windows-msvc --workspace --all-targets`: passes.
- `cargo check --locked --target x86_64-pc-windows-msvc -p victory-gpu --all-targets --features diagnostics`: passes.
- `cargo clippy --locked --target x86_64-pc-windows-msvc -p victory-gpu --all-targets --features diagnostics`
  and `-p victory --all-targets --features diagnostics`: pass.
- Warnings left, all at the base, none in this lane's code:
  `indirect.rs:5549` (dead `depth_written`), `draw.rs:2952`
  (sort_unstable_by), `shadow.rs:4517` (collapsible if),
  `victory-formats texture.rs:1152` (precedence).

Not done here:

- No test ran. The tests were only type-checked: no MSVC linker or Wine
  here, and victory-gpu does not build for Linux (6 errors in `list.rs` and
  `resource.rs`, already at the base).
- `cargo fmt --all` reformats about 63 files of the base, so it was not run
  over the tree. Only this lane's hunks are formatted, checked with
  `rustfmt --check` against the changed lines. Running `cargo fmt --all` on
  the PC will touch many unrelated files.

## Tests to run on the PC

```
set VICTORY_D3D12_WARP=1
cargo test -p victory-gpu --features diagnostics -- --test-threads=1 <name>
```

HostHeap tests return early, and pass, without a host device. Check the run
really uses WARP.

New, `descriptor::host::tests`:
- `a_set_of_slots_holds_what_was_put_in` (from cool_fox c3e643e; no GPU)
- `a_table_s_buffer_slot_is_left_to_its_draws_for_some_frames`

New, `descriptor::host::made_tests`:
- `bits_and_quick_mirrors_make_what_the_sets_and_the_claims_make` (four heaps,
  each a whole 2^21-slot view heap, about 64 MB of descriptors each)
- `many_threads_leave_each_slot_with_its_own_word_with_bits_and_quick_mirrors`
- (`many_threads_leave_each_slot_with_its_own_word_under_stripes`: removed in
  1f with the stripes.)
- `a_quick_look_at_a_claimed_slot_waits_for_the_claim`

New, `descriptor::host::sync_tests`:
- `pages_sorted_on_threads_make_what_one_by_one_makes`
- `pages_sorted_with_bytes_make_what_one_by_one_makes` (removed in S9: the
  bytes are the only compare, so `pages_sorted_on_threads_*` checks them;
  the one-by-one reference is now a model in the test, `mirror_one`)
- `records_of_no_view_left_before_their_mirror_make_what_one_by_one_makes`
- `changed_records_are_those_that_differ_one_by_one` (from cool_fox 53c5256)
- `passes_of_its_own_run_between_requests_and_answer_none` (about 1.2 s of
  real-time sleeps)
- `records_copied_onto_the_same_bytes_leave_the_page_unwritten`

New, `draw::tests`:
- `a_spread_pass_looks_at_each_part_once_a_round`,
  `every_slot_is_looked_at_again_in_a_round` (from cool_fox 2aa8134; no GPU)
- `each_on_calls_every_number_once` (no GPU)
- `the_walk_on_threads_finds_what_it_finds_on_one` (1,200 slots, 4 threads
  against 1; tabled slot 700 skipped on both; fresh-table slot 701 left,
  then made after 9 frames)

Existing, re-run (touched paths): (`made_tests::a_slot_at_a_time_makes_what_the_heap_wide_lock_makes`:
removed in 1f with the heap-wide lock; its checks are in `bits_and_quick_mirrors_*`),
`made_tests::many_threads_leave_each_slot_with_its_own_word` (now through
`many_threads_on`), (`sync_tests::a_page_at_once_*`: removed in S9 with the
page-by-page sync), `sync_tests::the_early_skip_*`,
and the ahead tests (`Ahead::start` now goes through `start_with`).

Then run the whole `descriptor` and `draw` modules once with every switch
above set in the environment. Tests that pick their own way must still pass.

### A/B plan for game runs

Same scene, same length, `VICTORY_PROFILE_SAMPLES` as usual. Baseline: all
off. Then one set at a time against baseline, then combined. (History: A9,
A10 and A12, and the sync switches of C2, cannot be set since S9: what they
turned on always runs.)

| Run | Switches | Measure |
| --- | --- | --- |
| A0 | none | all buckets below, for the baseline |
| A3 | `QUICK_MIRROR=1` | `made claim wait`, `heap walk`, texture pass time, refusal count with a full card |
| A4 | `SLOT_BITS=1` | `heap walk`, `heap pass lock wait` |
| A5 | `HEAP_PASS_THREADS=4` (alone, then with A3+A4) | `heap walk`, `heap walk job` (per thread), `heap walk wait` |
| A6 | `HEAP_SPREAD=1` | frame-time spikes against the old every-16th-pass spikes; `heap walk` max |
| A7 | `HEAP_PASS_TABLED=1` | `heap walk`, per-frame tabled skip count, picture checks |
| A8 | `HEAP_SWEEP_OWN=1` | `heap pass lock wait` |
| A9 | `HEAP_SYNC_BYTES=1` | `sync heaps` (`read_us`, `mirror_us`), memory |
| A10 | `HEAP_SYNC_THREADS=4` (alone, then with A9) | `sync heaps` time against spawn cost |
| A11 | `SHOWS_NONE=1` | `sync heaps`, `shows_none` count |
| A12 | `HEAP_SYNC_AHEAD=1 HEAP_SYNC_EARLY=4` (against `AHEAD=1` alone) | `sync heaps wait`, `sync heaps early`, `sync heaps ahead` |
| A13 | `SAME_RECORDS=1` | heap sync page count, `sync heaps` |
| C1 | A3 + A4 + A5 + A8 | pass buckets together |
| C2 | C1 + A9 + A10 + A11 + `HEAP_SYNC_FAST=1` + `HEAP_SYNC_AHEAD=1` | sync buckets together |
| C3 | C2 + A12 + A13, then + A6, then + A7 | whole frame; picture checks |

Each run: look for device removal, missing or stale textures, missing or
garbled geometry, and hangs.

## Risks to check on the PC

- Threads per heap per pass: PASS_THREADS and SYNC_THREADS (4, always
  since S9) spawn scoped
  threads each pass, for each heap, with no pool. Many small passes pay for
  the spawns. Compare against the time saved.
- PASS_THREADS: (the heap-wide `made` lock the jobs once fought over is gone
  since 1f: claims are per slot.)
- Parallel mirrors make host copies (`shadow::find`, since 1f with no table
  lock held, put in the tables by a compare: `RACE_LOST` counts the losers)
  and views from several threads at once. Watch for device removal and, in diagnostics
  builds, `HEAP_VIEWS_ELSEWHERE`.
- SYNC_BYTES (always since S9): 32 B per title view slot, against 8 B per print (about 32 MB
  for a 1M-slot heap).
- PASS_TABLED: a slot a table held, read by index within 8 frames, may be
  stale. After 8 frames it is walked again only on a full or spread pass or
  after a new write. Look for flicker or missing geometry that goes away
  with the switch off. A slot with only a texture-table mark is not skipped.
- HEAP_SPREAD: a released resource's stale view is caught within 16 passes,
  a part at a time.
- SWEEP_OWN: state goes with the HostHeap, not `forget_heap`. Check a heap
  re-created in the same frame still gets its pass.
- SHOWS_NONE: a draw that makes a slot from an older view at the same moment
  keeps that view until the slot is written again.
- SYNC_EARLY (always, 4 ms, since S9): a submission can wait for an own pass already running; the
  helper wakes every `<ms>` while the title submits. Check CPU cost on the
  slow PC.
- MADE_STRIPES (removed in 1f, measured +35 ms): a thread taking a second
  claim in the same stripe while holding one would have deadlocked.
- QUICK_MIRROR: a claimed slot goes to the claim, as without the switch
  (ff15ef5). With a full card, compare the texture pass's refusal count on
  and off.
- SLOT_BITS: `buffer_changes` scans count/64 words a call (about 32k at
  2^21 slots).
- SAME_RECORDS: one extra read of the destination per copy. Check what the
  title reads back is the same.

## Known gaps

- Nothing is pushed by the lane agents.
- Not ported from cool_fox: COPIED_GRACE, `copied_wait`, `copied_before`
  (1f1a433), as asked; 18bdf79's staging caps, unwatched-copy intervals and
  strides-buffer identity check (ours has no deferred heap pass); 6c17ce9's
  VICTORY_VIEW_THREADS; 53c5256's HEAP_SYNC_ASIDE; 44629de's card-steps,
  land parts and kind counters; e8d3d91's titlewrite.rs; helper threads
  pinned to fast cores; the picked-UAV handling.
- `QUICK_MIRRORED` is counted but only tests read it; it is not on the frame
  line or the profile.
- `a_quick_look_at_a_claimed_slot_waits_for_the_claim` is weak if WARP makes
  no host texture for its BC7 view: the answer is then false both ways, and
  only the wait is checked.
- Review findings: all five were fixed (ea442e7, 2c86826, ff15ef5, 2d3eb68,
  and this file). None left open.
- The base is not rustfmt-clean; see "Built here".
