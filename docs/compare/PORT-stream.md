# Port lane: cf-stream

Branch `work/cf-stream`, from `work/r5-s1`. Items #8 and #9 of
`docs/compare/cool-fox.md` (on `work/cool-fox-compare`), hand-ported from
cool_fox. No XCL-derived code.

| Commit | What |
|---|---|
| d838092 | #8 texture pool: textures the title fills and their staging buffers cut from 64 MiB blocks (cool_fox 9fea743, ripe blocks of 3b73cb0) |
| 8816e93 | #8 keep: host copies of title textures stay while the card has room (cool_fox 8d79927, 1f1a433) |
| 93a7254 | #8 away: a slot whose texture memory is briefly unmapped keeps its view (cool_fox 1f1a433) |
| ae0ab1c | #8 cheap hash: a cheap digest tells whether a texture's memory changed (cool_fox 2dfba4d) |
| 6b4f20e | #8 one-shot staging: a staging buffer made for one upload skips the write watch and the guards (cool_fox 2dfba4d) |
| 57eb649 | #8 same: the pass over textures picked by index leaves out uploads of bytes the host copy holds (cool_fox 8d79927) |
| 7f86be9 | #9 a DirectStorage file handle per worker (cool_fox fbc1830) |
| f34d7c3 | #9 submit timing: an automatic DirectStorage queue can wait for the title's submit (cool_fox 68e10e5) |
| 18fe5b4 | #9 in-flight floor: more buffered reads in flight than the staging buffer asked for (cool_fox e05d463) |
| 4d5b714 | Review fix: with the texture pool on, ripe blocks default to 12 |
| 9ea6706 | Review fix: the pool's empty blocks do not count as used card memory; the keep rule counts the kept room |
| 87d792d | Review fix: a pooled texture's aliasing barrier comes before its first transition |
| 210d652 | Review fix: a pooled texture that a host copy writes first is zeroed |
| e2a9c8c | Review fix: a texture that goes gives its pool piece back without the pool's lock |
| be14d7d | Review fix: SAME notes what the host copy holds only from a hash of what was detiled |
| f0ee459 | Review fix: SAME looks again only at textures whose upload it left out |

Item #13 (submit thread and thin lists) is not in this lane.

## Switches

Unset means exactly the old behaviour. Each switch is read once per
process (`OnceLock`), so every combination needs its own process: a
separate `cargo test` run, or a separate game run.

| Switch | Default | Meaning |
|---|---|---|
| `VICTORY_TEXTURE_POOL` | unset = off | `1`: host copies of textures the title fills, and their staging buffers, are cut from 64 MiB blocks (`texture_pool.rs`). `texture` or `staging`: only that half. |
| `VICTORY_RIPE_BLOCKS=<n>` | unset = 12 while the texture half of the pool is on; no effect with it off | Blocks made ahead. Pieces come first from blocks that every submission begun before their making has finished. `0` = none, which is cool_fox's device-loss setup (see the risks). |
| `VICTORY_TEXTURE_KEEP=1` | off | The sweep keeps idle host copies of title textures for 10 minutes while the card is under 88 % of its budget. Above that it lets the longest idle go first, down to 85 % (`host::vram_keep`). |
| `VICTORY_TEXTURE_AWAY_MS=<ms>` | unset / 0 = off | A slot whose texture memory is unmapped keeps its view for that long instead of being made null at the first look. cool_fox runs 3000. |
| `VICTORY_TEXTURE_DIGEST=1` | off | A cheap digest instead of blake3 tells whether a texture's memory changed. |
| `VICTORY_TEXTURE_ONCE=1` | off | A staging buffer made for one upload is filled once, without the write watch or the guards. |
| `VICTORY_TEXTURE_SAME=1` | off | The pass over textures picked by index leaves out an upload when the host copy already holds those bytes (by hash). |
| `VICTORY_DSTORAGE_IO=handles` | unset = one shared handle | Each DirectStorage worker reads on a file handle of its own, opened again from the title's handle. |
| `VICTORY_DSTORAGE_SUBMIT_FULL=1` | off (submits at half) | An automatic queue waits for the title's submit until it is full, instead of submitting itself at half. |
| `VICTORY_DSTORAGE_STAGING_MB=<n>` | unset / 0 = no floor | A floor on the staging bytes that buffered reads hold in flight. cool_fox uses 256. |

`VICTORY_BENCH_FILE` is only the input of the ignored `read_bench`
benchmark: a path to any large file on the PC.

## What could be checked here

Only `cargo check` and `cargo clippy` for `x86_64-pc-windows-msvc`
(`--workspace --all-targets`). Nothing was run here. `victory-gpu` does
not build its tests on Linux (its windows-only modules are named from
code that is not), and the texture and pool modules are windows-only.
Every test below has been type-checked, not run.

Note: the base branch is not clean under `cargo fmt --all --check` with
the pinned toolchain (many files predate the formatting), so the lane
formats only the code it adds. Run `cargo fmt --all` only if you want that
whole-tree change.

## Tests to run on the PC

```powershell
# 1. No switches: everything must pass as on r5-s1.
cargo test -p victory-gpu --lib -- texture_pool:: digest_tests title_memory_tests textures_are_kept_while_the_card_has_room
cargo test -p victory-hle --lib -- dstorage::

# 2. The pool on WARP, one thread (the test takes pieces of the shared pools).
$env:VICTORY_D3D12_WARP=1
cargo test -p victory-gpu --lib --features diagnostics -- texture_pool::d3d::tests::warp_a_texture_gives_its_piece_back_when_it_goes --test-threads=1 --nocapture
Remove-Item Env:VICTORY_D3D12_WARP

# 3. The ignored read benchmark (optional; any big file):
$env:VICTORY_BENCH_FILE="D:\some\big\file"; cargo test -p victory-hle --lib --release -- read_bench --ignored --nocapture
```

What each one checks:

- `texture_pool::tests`: `ranges_are_cut_aligned_and_join_again`,
  `empty_blocks_stay_while_the_card_has_room`,
  `a_failed_block_is_not_retried_for_each_small_upload`,
  `a_reclaim_lets_idle_blocks_go_without_reusing_live_or_resting_pieces`,
  `pieces_never_overlap_and_rest_before_they_are_used_again`,
  `pieces_come_from_blocks_older_than_the_lists_that_run`. These are pure
  pool logic with no device.
- `warp_a_texture_gives_its_piece_back_when_it_goes`: the piece comes back
  only when the last reference to the texture goes, and a reused staging
  piece starts as zeros. With no host device it returns at once and
  passes without checking anything, so make sure `VICTORY_D3D12_WARP=1`
  and `--features diagnostics` are both set.
- `digest_tests::a_digest_tells_bytes_apart` (DIGEST).
- `title_memory_tests::a_watched_detiling_forgets_what_the_staging_buffer_held`
  (review fix for SAME).
- `textures_are_kept_while_the_card_has_room` (KEEP's `vram_keep`).
- `dstorage::tests`: `half_a_queue_waits_for_the_submit`,
  `a_full_queue_submits_itself` (SUBMIT_FULL),
  `every_way_of_reading_gives_the_same_bytes` (IO, both ways),
  `the_staging_floor_only_raises_the_limit` (STAGING_MB).

## Game runs

Run once with no switch set first: it must match r5-s1. Then turn the
switches on one at a time, then all together:

- `VICTORY_TEXTURE_POOL=1` (ripe blocks at their default of 12)
- `VICTORY_TEXTURE_KEEP=1`, alone and then with the pool
- `VICTORY_TEXTURE_AWAY_MS=3000`
- `VICTORY_TEXTURE_DIGEST=1`, `VICTORY_TEXTURE_ONCE=1`,
  `VICTORY_TEXTURE_SAME=1`, alone and together
- `VICTORY_DSTORAGE_IO=handles`, `VICTORY_DSTORAGE_SUBMIT_FULL=1`,
  `VICTORY_DSTORAGE_STAGING_MB=256`

## Risks to check on the PC

- **Ripe blocks and device loss.** cool_fox's pool (9fea743) lost the
  device in two runs to page faults in the first texture of a block made
  that instant. 3b73cb0 fixed that with ripe blocks, which cool_fox runs
  at 12. Here they now default to 12 while the texture half of the pool is
  on. `VICTORY_RIPE_BLOCKS=0` with the pool on brings back cool_fox's known
  device-loss setup. cool_fox's later waits until new memory is resident
  (0322b39, f1d3eb2, its `VICTORY_WAIT_RESIDENT`) are **not ported**. If
  the device is lost with the pool on, that is the first suspect. Watch
  the log line "textures cut from blocks made while lists ran that still
  run". Each count there is a texture placed in an unripe block.
- **Pool spares and room.** Up to 64 empty 64 MiB blocks (4 GiB) plus the
  ripe blocks stay allocated and show in the card's `used`. `resource::room`,
  the sweep's idle tiers and the keep rule now take them off
  (`texture_pool::texture_room`). Check that with the pool on, host
  textures are not turned down earlier than with it off (the pass over
  textures picked by index).
- **Host-copy-first textures in blocks.** A pooled texture whose first
  write is a host copy or resolve now gets its aliasing barrier and zeros
  in every subresource in `texture::card_writes`. If no upload buffer can
  be had for the zeros, it warns and that subresource keeps the old texels.
  On WARP with the debug layer, look for aliasing or "resource not active"
  errors.
- **Aliasing barrier order.** It now comes before the first transition.
  Check that the debug layer is quiet about it.
- **Pool piece release.** Pieces of textures that go are queued and taken
  into the pool at its next lock, so they may rest slightly longer than
  2.5 s before reuse, never shorter.
- **SAME.** A stale texture after an upload was left out (video frames,
  streamed textures that change back) means `unchanged` is wrong. Turn
  SAME off to compare.
- **KEEP.** More card memory is held for longer. Watch the budget and
  whether buffer card copies or the first world frame's render targets
  fail to get room.
- **DirectStorage handles.** A handle per worker opens the title's file
  again (`ReOpenFile`, read access, sharing reads only). Check that load
  times drop, that the bytes read are the same (no corrupt assets), and
  whether the log warns "a file could not be opened again" (that file is
  then read on the first handle, as before).
