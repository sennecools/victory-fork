# Port lane: cf-stream

Branch `work/cf-stream`, from `work/r5-s1` (2ce3700). Items #8 and #9 of
`docs/compare/cool-fox.md` (on `work/cool-fox-compare`), hand-ported from
cool_fox. No XCL-derived code. Item #13 (submit thread and thin lists) is
not in this lane.

## What the lane ports

cool_fox authors as its history names them: `local` and `codec_d`.

### #8 Texture pool, keep, away, same, cheap hash, one-shot staging

| Ours | What | cool_fox source (author) |
|---|---|---|
| d838092 | Host textures the title fills, and their staging buffers, cut from 64 MiB blocks (`texture_pool.rs`) | 9fea743 (local, pool part only); d77556c (local, kept staging and per-pool maximum); 3b73cb0 (local, ripe blocks); 7d3e179 (local, empty-block rule only); ab75ac3 (codec_d, only the 500 ms back-off of `grow`) |
| 8816e93 | Keep: host copies of title textures stay while the card has room | 8d79927, 1f1a433 (local) |
| 93a7254 | Away: a slot whose texture memory is briefly unmapped keeps its view | 1f1a433 (local) |
| ae0ab1c | Cheap hash: a 128-bit digest instead of blake3 for the in-memory change hash and the recheck fingerprint | 2dfba4d (local) |
| 6b4f20e | One-shot staging: a staging state made by the index-picked pass skips the write watch and the guards | 2dfba4d (local, its "once") |
| 57eb649 | Same: the index-picked pass leaves out uploads of bytes the host copy holds | 8d79927 (local, its "unchanged"), adapted |

### #9 DirectStorage handle per worker, submit timing, in-flight floor

All in `crates/victory-hle/src/modules/dstorage.rs`.

| Ours | What | cool_fox source (author) |
|---|---|---|
| 7f86be9 | A file handle per worker (`ReOpenFile`, read access, shared reads) | fbc1830 (local, its `handles` way only) |
| f34d7c3 | An automatic queue can wait for the title's submit | 68e10e5 (local), default turned around |
| 18fe5b4 | A floor on the staging bytes buffered reads hold in flight | e05d463 (codec_d, DirectStorage part only) |

### Review fixes (ours, on the port)

| Ours | Fix |
|---|---|
| 4d5b714 | With the texture half of the pool on, ripe blocks default to 12; `=0` still opts out |
| 9ea6706 | `room()`, the default idle tiers and the keep rule take the pool's empty blocks off `used`; keep also counts `card_room_kept` + `targets_room_kept`, as 8d79927 does (corrects 8816e93's message) |
| 87d792d | A pooled texture's aliasing barrier comes before its first transition, as cool_fox has it |
| 210d652 | A pooled texture whose first write is a host copy or resolve gets the barrier and is zeroed in `texture::card_writes` (cool_fox has the same gap) |
| e2a9c8c | A texture that goes queues its piece without taking the pool's lock (possible lock-order inversion, not confirmed) |
| be14d7d | SAME notes what the host copy holds only from a hash of what was detiled (`Last::held`); the watch path clears it |
| f0ee459 | SAME rechecks only textures whose upload it left out (`HostTexture::left_out`) |

## Switches

All off by default: unset is exactly r5-s1. Each is read once per process
(`OnceLock`), so every combination needs its own process.

| Switch | Default | Values | What it does | Depends on |
|---|---|---|---|---|
| `VICTORY_TEXTURE_POOL` | off | `1` both halves; `texture`; `staging` | Host textures (1 sample, 1 plane, no clear value, no XOR layout) placed in owned 64 MiB DEFAULT-heap pages (`residency/pages.rs`, `ALLOW_ONLY_NON_RT_DS_TEXTURES`, `CreatePlacedResource`); staging buffers as pieces of 64 MiB upload pages. A piece rests 2.5 s before reuse. Texture pieces come only from ripe pages (older than every running submission); the spare pages are the pages made ahead: 12 below 85 % of budget, else 2. With no ripe page that has room the texture is not placed (an allocation of its own instead) and the unplaced-texture counter goes up. A texture over 64 MiB gets a heap of its own size, let go with it. Staging keeps 8 empty pages below 85 % of budget, else 2. After a refused page, none asked for 500 ms. | Staging half: only the non-transient path (not with `VICTORY_TEXTURE_STAGING_TRANSIENT=1`) |
| `VICTORY_RIPE_BLOCKS` (removed: ripe pages are the constant pair 12/2 in `residency/pages.rs`) | none | none | Was: n blocks made ahead | none |
| `VICTORY_TEXTURE_KEEP` (removed: a measured loser, host copies go by idle time alone) | off | `1` | `host::vram_keep`: host copies stay 10 min while use (less the pool's empty blocks, plus the kept room) is under 88 % of budget; above it the longest idle go first down to 85 % (5 s idle, 1.5 s from 95 %), in one unsliced sorted pass | Uses `texture_pool::texture_room` when the pool is on |
| `VICTORY_TEXTURE_AWAY_MS` | off | `<ms>`; `0` = off; cool_fox 3000 | A slot keeps its view until its memory has been unmapped that long, instead of null at the first look | None |
| `VICTORY_TEXTURE_DIGEST` | off (blake3) | `1` | cool_fox's 128-bit digest for the in-memory change hash and the recheck fingerprint. Nothing on disk uses these hashes. | None |
| `VICTORY_TEXTURE_ONCE` | off | `1` | A staging state made by the index-picked pass (untabled textures) is detiled once at submission without the write watch or the guards | None |
| `VICTORY_TEXTURE_SAME` | off | `1` | The index-picked pass skips an upload when all the texture's memory is mapped and hashes equal to what the host copy was filled from | Benefits watched-memory textures only with `VICTORY_TEXTURE_ONCE=1` |
| `VICTORY_DSTORAGE_IO` | one shared handle | `handles` | Each worker reads on a handle of its own, opened the first time it reads the file; on failure it warns and uses the title's handle | None |
| `VICTORY_DSTORAGE_SUBMIT_FULL` | off (submit at half) | `1` | An automatic queue starts its requests at the title's submit; submits itself only when full of unsubmitted commands with nothing running (warns) | Automatic queues only; `AUTO_SUBMIT_DISABLED` queues unchanged |
| `VICTORY_DSTORAGE_STAGING_MB` | off (title's 32 MiB) | `<n>` MiB; `0` = off; cool_fox 256 | Lowest in-flight staging limit, at start and after the title sets its buffer | None |

`VICTORY_BENCH_FILE` is only the input of the ignored `read_bench`.

## Build status here (Linux)

From the worktree, `CARGO_TARGET_DIR=$S/target-stream`, before each
commit:

- `CARGO_FEATURE_PURE=1 cargo check --locked --target x86_64-pc-windows-msvc --workspace --all-targets`:
  passes. One warning, older than the lane (indirect.rs `depth_written`).
- The same with `-p victory-gpu --features diagnostics`, and
  `cargo check -p victory --features diagnostics`: pass.
- `CARGO_FEATURE_PURE=1 cargo clippy --locked --target x86_64-pc-windows-msvc --workspace --all-targets`:
  4 warnings, all older than the lane and in code it does not touch
  (draw.rs, shadow.rs, indirect.rs, victory-formats texture.rs).
- `cargo test`: **not run**. `victory-gpu` does not build its tests on
  Linux (windows-only modules named from code that is not), and
  `victory-hle` depends on it and on `std::os::windows`. Every test below
  is type-checked for the Windows target only. The six pool logic tests
  and the digest test were also run as standalone rustc copies with
  stubbed host calls (7 pass); that is not a run of the real code.
- Formatting: the base is not clean under `cargo fmt --all --check`
  (about 63 files), so the lane formats only what it adds
  (`texture_pool.rs` whole; changed lines elsewhere, max width 100).

## Tests to run on the PC

```powershell
# 1. No switches: all must pass as on r5-s1.
cargo test -p victory-gpu --lib -- texture_pool:: residency::pages::tests digest_tests title_memory_tests textures_are_kept_while_the_card_has_room
cargo test -p victory-hle --lib -- dstorage::

# 2. The pool on WARP, one thread.
$env:VICTORY_D3D12_WARP=1
cargo test -p victory-gpu --lib --features diagnostics -- residency::pages::d3d::tests texture_pool::d3d::tests --test-threads=1 --nocapture
Remove-Item Env:VICTORY_D3D12_WARP

# 3. The existing texture and descriptor tests (staging, transient, guards,
#    Last, texture_slot_gone), on WARP as usual.
cargo test -p victory-gpu --lib -- texture:: descriptor::host

# 4. Optional benchmark, any large file:
$env:VICTORY_BENCH_FILE="D:\some\big\file"; cargo test -p victory-hle --lib --release -- read_bench --ignored --nocapture
```

Added tests:

- `residency::pages::tests` (pure logic): `ranges_are_cut_aligned_and_join_again`,
  `the_card_is_pressed_from_85_percent_of_its_budget`,
  `a_failed_page_is_not_retried_for_each_small_upload`,
  `a_reclaim_lets_idle_pages_go_without_reusing_live_or_resting_pieces`,
  `pieces_never_overlap_and_rest_before_they_are_used_again`,
  `pieces_come_only_from_pages_older_than_the_lists_that_run`,
  `a_piece_larger_than_a_page_has_a_heap_of_its_own_let_go_with_it`,
  `pages_made_ahead_are_the_spares_and_never_go_and_come_back`,
  `a_compaction_empties_the_emptiest_page`,
  `a_compaction_moves_nothing_that_has_nowhere_to_go_or_is_not_ripe`.
- `residency::pages::d3d::tests` (WARP): `warp_a_texture_gives_its_piece_back_when_it_goes`
  (places in a ripe page directly, ignoring the switch),
  `warp_pages_let_go_return_their_commit` (full pages emptied and let go
  lower the commit by at least a page each) and
  `warp_only_evict_alone_keeps_the_commit` (WARP only: documents the
  measured card fact, proves nothing about the card). With no host device
  they return at once and pass without checking: set both
  `VICTORY_D3D12_WARP=1` and `--features diagnostics`.
- `texture_pool::tests::empty_staging_pages_stay_while_the_card_has_room`
  and `texture_pool::d3d::tests::warp_a_staging_piece_begins_as_zeros`.
- `digest_tests::a_digest_tells_bytes_apart` (DIGEST).
- `title_memory_tests::a_watched_detiling_forgets_what_the_staging_buffer_held` (SAME fix).
- `textures_are_kept_while_the_card_has_room` (KEEP).
- `dstorage::tests`: `every_way_of_reading_gives_the_same_bytes` (IO,
  both ways; 3 extra threads), `half_a_queue_waits_for_the_submit`,
  `a_full_queue_submits_itself` (SUBMIT_FULL),
  `the_staging_floor_only_raises_the_limit` (STAGING_MB), and the existing
  `queue_options_control_submission_at_half_capacity` (the default).
- `read_bench` (ignored).

No test for AWAY or ONCE: cool_fox has none.

### A/B game runs

Same scene, same route, `VICTORY_FRAMETIME_DUMP=1` on every run.

1. Baseline: no switch. Must match r5-s1.
2. One at a time:
   - `VICTORY_TEXTURE_POOL=1` (ripe 12), then `=texture`, then `=staging`
   - `VICTORY_TEXTURE_KEEP=1` (since removed)
   - `VICTORY_TEXTURE_AWAY_MS=3000`
   - `VICTORY_TEXTURE_DIGEST=1`
   - `VICTORY_TEXTURE_ONCE=1`
   - `VICTORY_TEXTURE_SAME=1`, then `SAME=1` + `ONCE=1`
   - `VICTORY_DSTORAGE_IO=handles`
   - `VICTORY_DSTORAGE_SUBMIT_FULL=1`
   - `VICTORY_DSTORAGE_STAGING_MB=256`
3. Combined: the texture set (POOL=1, KEEP, AWAY_MS=3000, DIGEST, ONCE,
   SAME), the DirectStorage set, then all ten.

What to measure (frametime CSV columns and log lines):

| Switch | Look at |
|---|---|
| POOL | `textures_ms`, `prep`, `frame_ms_max` (fewer allocation spikes); log "textures not placed in a page (no ripe page with room, or the page was refused)" (each such texture is an allocation of its own; should stay small) |
| KEEP | `textures_mb` and `textures_ms` (fewer re-uploads), `card_*`, VRAM use against budget |
| AWAY | `heap_ms`, `heap_passes`; black flashes after unmaps |
| DIGEST | `texture_refresh_ms`, `heap_ms` |
| ONCE | `texture_refresh_ms`, `textures_ms` |
| SAME | `textures_mb`, `heap_ms`; log "uploads of textures picked by index left out" |
| IO, SUBMIT_FULL, STAGING_MB | Load time to first frame and after travel, `frame_ms_max` while streaming, `ram_mb`; dstorage warnings |

## Risks to check on the PC

- **Ripe pages and device loss.** cool_fox's pool (9fea743) lost the
  device to page faults in the first texture of a block made that
  instant; 3b73cb0 fixed it with ripe blocks. Pages follow that rule
  strictly: a texture is never cut from an unripe page (it becomes an
  allocation of its own instead, counted in the log). The one exception
  is a texture over 64 MiB, whose heap of its own is used at once, as a
  committed resource would be. cool_fox's waits until new memory is
  resident (0322b39, f1d3eb2) are not ported: first suspect on a device
  loss.
- **Pool spares and room.** Up to 12 empty pages (768 MiB) stay allocated
  below 85 % of budget, 2 above it; they are the pages made ahead. The
  spare count is applied at the next piece even when the sweep found the
  pages busy. `room()`, the idle tiers and KEEP take them off `used`.
  Check host textures are not turned down earlier than with the pool off,
  and that the unplaced-texture count stays small in load bursts.
- **Stale texels.** A pooled texture first written by a host copy is
  zeroed in `card_writes`; if no upload buffer is free it warns and keeps
  the old texels. A pooled texture never fully uploaded (transient staging
  returning nothing, a partial copy) can still show its piece's previous
  owner. Look for garbage in new textures.
- **Aliasing barrier.** Debug layer on WARP: no aliasing or "resource not
  active" errors.
- **Piece release.** Pieces are queued and taken at the next pool lock;
  they may rest a little longer than 2.5 s, never shorter.
- **KEEP.** More card memory held longer: watch the budget, and that
  buffer card copies and the first world frame's render targets still get
  room. Past 88 % the whole sorted pass runs in one slice: check its cost
  with many textures.
- **AWAY.** Kept views of unmapped memory must not show wrong data.
- **SAME.** A stale texture after a left-out upload (video frames, A-B-A
  content) means `unchanged` is wrong: turn SAME off to compare. The
  pass's uploaded count still includes skipped textures, at 0 bytes.
- **DIGEST.** A collision would leave a texture stale; compare with it off.
- **IO=handles.** Handles are workers x open files: watch the process
  handle count. Bytes must be the same (no corrupt assets). Warning "a
  file could not be opened again" means that file falls back.
- **SUBMIT_FULL.** Stalls or loads that never finish; frequent warnings
  "a queue full of commands that are not submitted submits itself".
- **STAGING_MB.** Buffered reads allocate per request: watch host RAM.

## Known gaps

- Not ported from #8: cool_fox's resident and mapped waits for new blocks
  (0322b39, f1d3eb2; not on the list), its texture book, `fresh()` list,
  residency and admission caps, the `allocation_costs` test, the
  `VICTORY_KEEP_FREED` switch of 7d3e179, the 30 s-from-68 % rule that
  8d79927 replaced, and from 2dfba4d the mapped-byte reuse, bake and
  remapping changes and the partial sweep.
- Not ported from #9: from 68e10e5 the trace (`VICTORY_DSTORAGE_TRACE`),
  re-writing to new pages when a list changes during a write, read
  retries, the louder page-list warning; from fbc1830 the `overlapped`
  and `direct` read modes and the xmem lock split.
- Review: all nine findings fixed (above). The lock-order inversion was
  not confirmed, only made impossible.
- Nothing ran here; all tests are type-checked only.
