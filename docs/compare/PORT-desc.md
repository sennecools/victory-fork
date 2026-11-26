# Port lane `desc`: descriptor heaps from cool_fox

Branch `work/cf-desc`, from `work/r5-heapsync`. Items #1 to #3 of the ranked
list in `docs/compare/cool-fox.md` (branch `work/cool-fox-compare`). Nothing
here is XCL-derived. COPIED_GRACE is not ported, as asked.

Every switch is off by default. With all of them off, the code behaves as
`work/r5-heapsync` does.

## Switches

| Switch | Default | What it does when on | Where |
| --- | --- | --- | --- |
| `VICTORY_QUICK_MIRROR=1` | off | Every mirror of a slot that is already made from its record returns before the `made` lock or the slot's claim. The texture pass's `texture_slot_made_from` answers "no" before the lock for a slot settled on another record. | `descriptor/host.rs` `quick_mirror` |
| `VICTORY_SLOT_BITS=1` | off | The set of buffer-view slots and the set of changed slots are bits in atomic words, with no lock. | `descriptor/host.rs` `slot_bits` |
| `VICTORY_MADE_STRIPES=1` | off | Claims of slots are taken under 512 striped locks (slot `i` under lock `i % 512`). Implies the slot-at-a-time machinery of `VICTORY_MADE_ATOMIC=1`: no heap-wide lock, and queue/set changes logged in order (`NoteLog`). | `descriptor/host.rs` `made_stripes` |
| `VICTORY_HEAP_PASS_THREADS=N` | unset (1) | The walk of the heap buffer pass runs on N threads of its own (2 to 16), in jobs of 256 slots. | `draw.rs` `heap_pass_threads` |
| `VICTORY_HEAP_SPREAD=1` | off | The heap buffer pass looks again at 1/16 of the unchanged slots at every pass, not at all of them every 16th pass. | `draw.rs` `heap_spread` |
| `VICTORY_HEAP_PASS_TABLED=1` | off | The heap buffer pass leaves slots that a draw's table held in the last 8 frames to that draw. These marks are the pass's own (`buffer_tabled`). The heap sync and the texture queue do not read them. | `draw.rs` `heap_pass_tabled` |
| `VICTORY_HEAP_SWEEP_OWN=1` | off | What the heap buffer pass keeps of a heap lies on the heap, under a lock of its own, and not in one map for every heap. | `draw.rs` `heap_sweep_own` |
| `VICTORY_HEAP_SYNC_BYTES=1` | off | The heap sync keeps the records' bytes (32 B per slot, where a print is 8 B) and compares 16 records at a time. | `descriptor/host.rs` `sync_bytes` |
| `VICTORY_HEAP_SYNC_THREADS=N` | unset (1) | The heap sync reads, compares and sorts pages (8 a job) and mirrors slots (256 a job) on N threads (2 to 16). | `descriptor/host.rs` `sync_threads` |
| `VICTORY_SHOWS_NONE=1` | off | The heap sync leaves a changed record that is not a view of ours, in a slot made from no record, before it mirrors anything. | `descriptor/host.rs` `shows_none` |
| `VICTORY_HEAP_SYNC_EARLY=<ms>` | unset (0) | Needs `VICTORY_HEAP_SYNC_AHEAD=1`. The sync thread runs passes of its own after `<ms>` with no request, for 500 ms after the last request. Those passes answer no request. | `descriptor/ahead.rs` `early` |
| `VICTORY_SAME_RECORDS=1` | off | `copy_simple`/`copy` leave records that are copied onto the same bytes unwritten, so the page stays clean for the write watch. | `descriptor.rs` `same_records` |

Existing switches these work with: `VICTORY_MADE_ATOMIC=1`,
`VICTORY_HEAP_SYNC_AHEAD=1` and `VICTORY_HEAP_SYNC_FAST=1`.

### The striped-lock decision

The owner asked for cool_fox's striped `made` locks in item #1. cb9245a left
them out and pointed to `VICTORY_MADE_ATOMIC` instead. A reviewer flagged
that, so the stripes are now ported behind their own switch,
`VICTORY_MADE_STRIPES` (2c86826).

Ours has no heap-wide array of keys under one lock to stripe. The per-slot
claim is already what replaces the heap-wide lock. So the stripes are a lock
that each claim takes before its slot, and the switch turns on the
slot-at-a-time logging that keeps queue and set changes in order. Compare
these on the PC:

- `VICTORY_MADE_ATOMIC=1`: claims spin on the slot.
- `VICTORY_MADE_STRIPES=1`: claims sleep on the stripe.

## Tests to run on the PC (WARP)

The tests that make a `HostHeap` return early, and pass, when there is no
host device. Check that the run uses the WARP device:
`--features diagnostics` with `VICTORY_D3D12_WARP=1`.

```
set VICTORY_D3D12_WARP=1
cargo test -p victory-gpu --features diagnostics -- --test-threads=1 <name>
```

`descriptor::host::made_tests`:
- `bits_and_quick_mirrors_make_what_the_sets_and_the_lock_make`
- `many_threads_leave_each_slot_with_its_own_word_with_bits_and_quick_mirrors`
- `many_threads_leave_each_slot_with_its_own_word_under_stripes`
- `a_quick_look_at_a_claimed_slot_waits_for_the_claim`
- also run the existing `a_slot_at_a_time_makes_what_the_heap_wide_lock_makes`
  and `many_threads_leave_each_slot_with_its_own_word`

`descriptor::host::tests`:
- `a_set_of_slots_holds_what_was_put_in`
- `a_table_s_buffer_slot_is_left_to_its_draws_for_some_frames`

`descriptor::host::sync_tests`:
- `pages_sorted_on_threads_make_what_one_by_one_makes`
- `pages_sorted_with_bytes_make_what_one_by_one_makes`
- `records_of_no_view_left_before_their_mirror_make_what_one_by_one_makes`
- `changed_records_are_those_that_differ_one_by_one`
- `passes_of_its_own_run_between_requests_and_answer_none`
- `records_copied_onto_the_same_bytes_leave_the_page_unwritten`
- also run the existing `a_page_at_once_*` and `the_early_skip_*`

`draw::tests`:
- `a_spread_pass_looks_at_each_part_once_a_round`
- `every_slot_is_looked_at_again_in_a_round`
- `each_on_calls_every_number_once`
- `the_walk_on_threads_finds_what_it_finds_on_one`: this also covers a slot of
  a table made afresh, which the pass leaves at first and makes after 9 frames
  with no table

Also run the whole `descriptor` and `draw` test modules once with every
switch above set in the environment. The tests that pick a way themselves
must still pass.

## Risks to check on the PC

- **Threads per heap per pass.** `VICTORY_HEAP_PASS_THREADS` and
  `VICTORY_HEAP_SYNC_THREADS` spawn scoped threads for each pass, for each
  heap. They do not use a pool. Check the spawn cost against the time saved
  in the profile.
- **Memory.** `VICTORY_HEAP_SYNC_BYTES` keeps 32 B per title slot, where a
  print is 8 B. On a heap of about 1M slots that is about 32 MB.
- **`VICTORY_HEAP_PASS_TABLED`.** The pass's marks are now kept apart from
  the 240-frame window of the sync and the texture queue (ea442e7). Check
  that no buffer view stays stale in slots that a shader picks by index
  after a table held them. Visible symptoms would be flicker or missing
  geometry that goes away with the switch off. A slot that holds a texture
  table's mark is not skipped by the pass any more.
- **`VICTORY_HEAP_SYNC_EARLY`.** The thread wakes every `<ms>` while
  submissions come. Check the CPU cost on a slow machine, and that the
  requested passes do not get later.
- **`VICTORY_MADE_STRIPES`.** A thread that took a second claim while
  holding one in the same stripe would deadlock. No path does this now. If
  a game run hangs with the switch on, look there first, and compare the
  `made stripe wait` profile events with the `made claim wait` events of
  `VICTORY_MADE_ATOMIC=1`.
- **`VICTORY_QUICK_MIRROR`.** A claimed slot now goes to the lock, as it
  does without the switch (ff15ef5). With a full card, check that the
  texture pass's refusal count is the same with the switch on and off.
- **`VICTORY_SAME_RECORDS`.** Check that what the title reads back is the
  same, and that the heap sync's page count drops.

## Built here

Built on Linux, for the Windows target:

- `cargo check --target x86_64-pc-windows-msvc --workspace --all-targets`
  and `cargo clippy` with the same flags: no new warnings in the code this
  lane touches.
- `cargo fmt --check` fails on these files at the base already, so only new
  code is formatted.
- No test ran here. They need Windows and D3D12.
