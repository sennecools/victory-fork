# Port lane: cf-flush

Branch `work/cf-flush`, from `work/r5-flush`. Items #4 and #5 of
`docs/compare/cool-fox.md` (on `work/cool-fox-compare`), hand-ported from
cool_fox. No XCL-derived code. Every new behaviour is behind its own
`VICTORY_*` switch, off by default; with all of them unset each changed
path is the old code or an equivalent of it.

## 1. What the lane ports

All cool_fox commits named here have the author `local`.

| Item | Our commit | cool_fox source | What |
|---|---|---|---|
| #4 | 8e20d3c | 66afb2b | safecopy: `copy_mapped`, `read_into` and `write_mapped` (host.rs `read()` and the write closure) can copy with one `rep movsb` (new `safecopy.rs`). A fault inside it is taken by our write watch's vectored handler (`watch::on_write`), which stays first (`AddVectoredExceptionHandler(1, ..)`): a write into a block we guard opens it and the copy goes on; any other fault of the copy goes to its failing exit and the copy returns false, as the kernel call did. |
| #5 | 453253a | 2aa8134 (watch and nothing_new parts), 74045e4 (PROTECTED, guarded views not asked), 7bc7c07 (bit sets, lock let go between runs, handler without the lock), race fix 2be0515 = 689df31 | New `watch/bits.rs` (`Bits`, `Blocks`). GUARD_BITS, LIGHT_REFRESH, NOTHING_NEW. The race fix lives in the light paths (`unguard`, `open_together`, `views_mapped` opening all views at once). |
| #5 | 25b2ce7 | 7bc7c07 | GUARD_REST and OPEN_ONE_VIEW; `cover()`: a read-only, not clean block becomes clean only once every other view of its page is read-only again. |
| #5 | 3dad525 | 74045e4 (KNOWN_HOLES part) | KNOWN_HOLES: holes in placeholders (`NtQueryVirtualMemory`) held as absent until `title_memory_remapped` or the new `title_memory_allocated` (called from xmem `XMemVirtualAlloc`); other holes asked with the lock let go; `look_at_holes` at each flip re-checks a hundredth of the held holes. |
| review | 83be862 | (ours) | With LIGHT_REFRESH alone a block could be noted clean and published beside a view of its page that could not be guarded, and a write told at that view was lost. Light now goes through `guard_switched()` and `cover()` checks the other views under it. |
| doc | this commit | | This file. |

## 2. Switches

All off by default. Each is read once per process (`OnceLock`), so every
combination needs its own process: a separate `cargo test` run or a
separate game run.

| Switch | Default | Values | What it does | Depends on |
|---|---|---|---|---|
| `VICTORY_SAFE_COPY` | off | `1` = on; unset or anything else = off | Title memory is copied with one `rep movsb` instead of `ReadProcessMemory` / `NtWriteVirtualMemory`. Asking the switch registers the watch's handler. | Our handler being the first vectored handler. cool_fox had it on. |
| `VICTORY_GUARD_BITS` | off | `1` | The guards' sets (protected, clean, absent, aliased, held) are words of bits instead of `BTreeSet`s; a refresh reads them by words and lets go of the `GUARDS` lock between the runs of blocks it makes read-only. | The journal lock / epoch span locks keeping other askings off the same blocks. |
| `VICTORY_LIGHT_REFRESH` | off | `1` | Lock-free `PUBLISHED` / `PROTECTED` bits. A wholly published view range is answered from the journal or epoch tables without the guards; ranges with a protected block are not asked of the kernel nor joined into the ask-ahead; `writing`, `title_memory_written` and the fault handler skip the lock when no block of theirs is protected. Holds the 2be0515 race fix and the review fix (83be862). | Nothing; goes through `guard_switched()`. In our journal path the bits answer still takes the `JOURNAL` mutex; the epoch path is lock-free. |
| `VICTORY_NOTHING_NEW` | off | `1` | `Batch::flush`, `flush_one` and the wide-refresh jobs drop copies that `watch::nothing_new` / `Shadow::nothing_new` say have nothing new, before asking anything or waking a thread. | View memory only with `VICTORY_LIGHT_REFRESH=1`; write-watched memory without it. |
| `VICTORY_GUARD_REST` | off | `<milliseconds>`; unset or `0` = off (cool_fox runs 3; its unit was µs, 3000, on) | For that long after a write fault opens a block, refreshes leave it open and tell it as written at each asking. | Goes through `guard_switched()`; lock let go only with GUARD_BITS. |
| `VICTORY_OPEN_ONE_VIEW` | off | `1` | A write fault opens only the view it came through; the page's other views stay read-only and are marked not clean (bits cleared first). | Same as GUARD_REST. |
| `VICTORY_KNOWN_HOLES` | off | `1` | As in item 3dad525 above. | Held holes are also published only with `VICTORY_LIGHT_REFRESH=1`. |

## 3. Build status here (Linux, target `x86_64-pc-windows-msvc`)

Passed, with no errors and no new warnings in touched code (rerun at the
lane tip 83be862):

- `cargo check --locked --target x86_64-pc-windows-msvc --workspace --all-targets`:
  one pre-existing warning (`depth_written` never read, indirect.rs).
- `cargo clippy --locked --target x86_64-pc-windows-msvc --workspace --all-targets`:
  only pre-existing warnings (texture.rs, draw.rs, shadow.rs, indirect.rs,
  victory-formats).
- `cargo check -p victory-gpu --all-targets --features diagnostics` (Windows
  target); at item #4 also `cargo clippy -p victory -p victory-gpu
  --all-targets --features victory/diagnostics,victory-gpu/diagnostics`; at
  item #5 `cargo check -p victory --features diagnostics`.

Tests were only type-checked. No Windows linker here, and `watch` and
`safecopy` are `cfg(windows)`. `cargo test -p victory-gpu` on Linux does not
build (pre-existing errors in list.rs / resource.rs). `watch/bits.rs` alone,
compiled with `rustc --test` on Linux, passed both its tests with and without
`VICTORY_GUARD_BITS=1`.

Formatting: the base is not rustfmt-clean (`cargo fmt --all` rewrites ~60
files), so only this lane's hunks were formatted with our `rustfmt.toml`;
untouched code was left as it was.

## 4. Tests to run on the PC

Unit tests added:

- `safecopy::tests`: `the_switch_is_off_by_default`,
  `copies_every_length_and_alignment`, `faults_are_failures`, `copy_costs`
  (`#[ignore]`, kernel vs `rep movsb` into plain memory).
- `watch::bits::tests`: `bits_of_ranges`, `blocks_are_a_set`.
- `watch::tests`: `safe_copies_into_guarded_blocks_open_them`,
  `unchanged_ranges_are_known`, `no_write_to_views_is_lost_between_askers`,
  `no_write_through_another_view_is_lost_between_askers`,
  `no_write_to_resting_views_is_lost_between_askers`,
  `a_write_opens_the_view_it_goes_through`, `a_block_just_written_rests`,
  `a_write_told_at_a_view_that_cannot_be_guarded_is_not_lost`,
  `holes_are_known_until_the_runtime_tells`.

Runs (PowerShell, repo root; clear the variables between runs with
`Remove-Item Env:VICTORY_*`):

```powershell
# 1. No switches: everything must pass as on r5-flush.
cargo test -p victory-gpu --lib -- watch:: safecopy:: host::

# 2. One switch per run, then the combinations the tests check.
$env:VICTORY_SAFE_COPY=1;     cargo test -p victory-gpu --lib -- watch:: safecopy:: host::
$env:VICTORY_LIGHT_REFRESH=1; cargo test -p victory-gpu --lib -- watch::
$env:VICTORY_GUARD_BITS=1;    cargo test -p victory-gpu --lib -- watch::
$env:VICTORY_OPEN_ONE_VIEW=1; cargo test -p victory-gpu --lib -- watch::
$env:VICTORY_GUARD_REST=3;    cargo test -p victory-gpu --lib -- watch::
$env:VICTORY_LIGHT_REFRESH=1; $env:VICTORY_KNOWN_HOLES=1; cargo test -p victory-gpu --lib -- watch::
# All together:
$env:VICTORY_SAFE_COPY=1; $env:VICTORY_GUARD_BITS=1; $env:VICTORY_LIGHT_REFRESH=1; $env:VICTORY_NOTHING_NEW=1; $env:VICTORY_GUARD_REST=3; $env:VICTORY_OPEN_ONE_VIEW=1; $env:VICTORY_KNOWN_HOLES=1; cargo test -p victory-gpu --lib -- watch:: safecopy:: host::

# 3. Loop the three many-askers tests ~50 times with all switches on
#    (cool_fox saw 6 failures in 200 runs before its race fix):
1..50 | % { cargo test -p victory-gpu --lib -- between_askers; if ($LASTEXITCODE) { "fail $_" } }

# 4. Optional cost measurement:
cargo test -p victory-gpu --lib --release -- safecopy:: --ignored --nocapture
```

Tests that check something only with switches set:

- `a_write_opens_the_view_it_goes_through`: only with `VICTORY_OPEN_ONE_VIEW=1`.
- The three `*_between_askers` tests always check that no write is lost;
  only with `VICTORY_LIGHT_REFRESH=1` do they also check that the bits
  answered some askings.
- `a_block_just_written_rests` and `no_write_to_resting_views_is_lost_between_askers`
  set their rest per thread, whatever `VICTORY_GUARD_REST` says.
- `holes_are_known_until_the_runtime_tells`: answers always; bits only with
  `VICTORY_LIGHT_REFRESH=1` and `VICTORY_KNOWN_HOLES=1`.
- `unchanged_ranges_are_known`: view memory known by its bits only with
  `VICTORY_LIGHT_REFRESH=1`; without it, checks it is never known that way.
- `a_write_told_at_a_view_that_cannot_be_guarded_is_not_lost`: run it with
  `VICTORY_LIGHT_REFRESH=1` alone, the case of the review fix. It assumes
  `VirtualProtect` of a 64 KiB block whose view has one 4 KiB page committed
  fails and changes nothing; if it fails at the first asking, check that.
- `the_switch_is_off_by_default`: only when `VICTORY_SAFE_COPY` is unset.
- `host::tests::sparse_copies_preserve_islands_and_observe_remaps` goes through
  safecopy only with `VICTORY_SAFE_COPY=1`.

WARP: this lane touches no D3D12 code. Run the usual WARP suites once with
no switches and once with all seven on, to see that nothing else moved.

A/B game runs. Baseline: the usual r5-flush run with `VICTORY_R5_FLUSH=1`
(its counters line) and `VICTORY_PROFILE=1`. Then one switch at a time,
then NOTHING_NEW and KNOWN_HOLES each again with LIGHT_REFRESH, then all
seven. Measure:

| Switch | Expect to move |
|---|---|
| SAFE_COPY | `batch flush` `copy_us` and `kernel_us` (baseline copy part 15.1 ms, r5-batch-flush notes); frame-rate line `copies` |
| GUARD_BITS | `batch flush` `journal_us` (guard walks under the lock); fault wait on title threads |
| LIGHT_REFRESH | `batch flush` `journal_us`; `ask once`; R5 flush failed kernel askings |
| NOTHING_NEW | `batch flush` `refreshes`, `mb_asked`; span length |
| GUARD_REST | R5 flush faults, VirtualProtect calls and their ms, blocks guarded |
| OPEN_ONE_VIEW | R5 flush VirtualProtect calls and blocks guarded |
| KNOWN_HOLES | `batch flush` `journal_us`; `ask once` |

Plus frame-rate line `ms` / `prep` and the frame-time dump for every run.

## 5. Risks to check on the PC

- **Lost writes / staleness.** Run with `VICTORY_REFRESH_CHECK=1` and all
  switches on, and once each with LIGHT_REFRESH, OPEN_ONE_VIEW and
  GUARD_REST=3. Any "a host copy of title memory is stale after its refresh, three times in a row"
  warning not also seen without the switches is this lane's: bit ordering, views
  mapped beside guarded ones.
- **Holes filled with nobody telling** (KNOWN_HOLES): the warning "a
  placeholder of the title's was filled with nobody telling" (logged once)
  means held holes can be stale for up to a hundred frames.
- **safecopy and the handler order.** The copy's faults are turned into
  failures only because our handler is first. Anything registering a
  vectored handler ahead of ours (overlay, debugger helper) makes them
  crashes: watch for access violations in safecopy's leaf.
- **Lock re-entry** (SAFE_COPY): a write fault in our copy takes `GUARDS`
  (and inplace's lock for aliased blocks). A caller of `write_mapped` or
  `copy_mapped` holding either would hang. None found; watch for hangs.
- **Behaviour change** (SAFE_COPY): a block guarded again between
  `writing()` and the copy is now opened and written; before, the kernel's
  write failed silently.
- **rep movsb speed** into upload-heap or write-combined memory: compare
  `copy_us` and the ignored `copy_costs`.
- **Lock let go between runs** (GUARD_BITS): stale copies or a hang with
  GUARD_BITS alone.
- **GUARD_REST** does a HashMap insert inside the vectored handler, as
  cool_fox did.
- **Performance:** if a switch makes frames slower, report which and the
  buckets above.

## 6. Known gaps

- Not ported: CF's AVX2 `put_changed` / `victory_safe_put` (landing); its
  other safecopy uses (`zero()` in `read_pieces`, `read_there`, the
  `write_mapped` again-after-change fallback, shadow and land call sites);
  host test `landings_title_writes_and_light_refreshes_at_once` (landing).
- From 2aa8134: HEAP_SPREAD, LAND_WINDOW / `land_past`, SEND_CHANGED, the
  not-copied range ends and long-copy splitting. `unchanged_ranges_are_known`
  is at the watch level with whole pages, since we still copy partial ends.
- CF's journal `RwLock` / ASKING design; its views record read by many
  threads (inplace); the BESIDE bits (we have no COPY_BESIDE); writes past
  the guards in the many-askers test.
- 7bc7c07's fault book (`VICTORY_FAULT_BOOK`); 74045e4's LAND_TAKEN;
  3ead56e (PAGE_PARTS) not needed.
- CF test `no_mapping_is_lost_between_askers` not ported: it would hit our
  map-before-tell window, which exists without any switch (xmem maps with
  `MapViewOfFile3` before `title_memory_remapped`; an asking in between can
  guard the new view, the telling forgets that guard while the page stays
  read-only, and a later title write faults with no guard to open it).
  cool_fox closed it in 26d818c, not in this lane. KNOWN_HOLES narrows it
  for placeholder holes.
- Review findings: both fixed (83be862, and this file). None left open.
- Nothing run on Windows; the reasoning about the Windows tests was by hand.
