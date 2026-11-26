# Port lane: cf-flush

Branch `work/cf-flush`, from `work/r5-flush`. Items #4 and #5 of
`docs/compare/cool-fox.md` (on `work/cool-fox-compare`), hand-ported from
cool_fox. No XCL-derived code.

| Commit | What |
|---|---|
| 8e20d3c | #4 safecopy: copies of title memory as one `rep movsb` whose faults fail the copy (cool_fox 66afb2b) |
| 453253a | #5 guard bits, light refresh, nothing new (cool_fox 2aa8134, 74045e4, 7bc7c07, race fixes 2be0515 / 689df31) |
| 25b2ce7 | #5 GUARD_REST and OPEN_ONE_VIEW (cool_fox 7bc7c07) |
| 3dad525 | #5 KNOWN_HOLES (cool_fox 74045e4) |
| 83be862 | Review fix: with LIGHT_REFRESH alone, a block was noted clean beside a view of its page that could not be guarded |

## Switches

All are off by default. Unset means exactly the old behaviour. Each one is
read once per process (`OnceLock`), so every combination needs its own
process: a separate `cargo test` run, or a separate game run.

| Switch | Default | Meaning |
|---|---|---|
| `VICTORY_SAFE_COPY=1` | off | `copy_mapped`, `read_into` and `write_mapped` copy with one `rep movsb` (`safecopy.rs`) instead of `ReadProcessMemory` / `NtWriteVirtualMemory`. A fault inside the copy is taken by our vectored handler (`watch::on_write`) and fails the copy. |
| `VICTORY_GUARD_BITS=1` | off | The guards' sets are words of bits (`watch/bits.rs`). A refresh reads them by words and lets go of the guards' lock between the runs it makes read-only. |
| `VICTORY_LIGHT_REFRESH=1` | off | Lock-free `PUBLISHED` / `PROTECTED` bits. Askings about view memory that nothing wrote are answered from the journal without the guards. `writing`, `title_memory_written` and the fault handler skip the lock when no block of theirs is read-only by us. |
| `VICTORY_NOTHING_NEW=1` | off | A flush leaves out copies that it knows have nothing new. For view memory this only works with `VICTORY_LIGHT_REFRESH=1`. |
| `VICTORY_GUARD_REST=<ms>` | unset / 0 = off | A block that a write fault opened is left open for that many ms and told as written at each asking. cool_fox runs 3. |
| `VICTORY_OPEN_ONE_VIEW=1` | off | A write fault opens only the view it came through. The page's other views stay read-only but are no longer clean. |
| `VICTORY_KNOWN_HOLES=1` | off | Absent blocks inside placeholders are held as absent until the runtime tells of a mapping or an allocation there. Other holes are queried with the lock let go. Once a frame, a hundredth of the held holes is looked at again. |

## Tests to run on the PC

The tests use real Windows memory APIs, so they only run on Windows. Here
they were type-checked for `x86_64-pc-windows-msvc` but not run. Run from
the repo root in PowerShell. The filters cover `watch::tests`,
`watch::bits::tests`, `watch::epoch` and `safecopy::tests`:

```powershell
# 1. No switches: everything must pass as on r5-flush.
cargo test -p victory-gpu --lib -- watch:: safecopy::

# 2. Each switch combination below in a fresh shell, or clear the variables
#    between runs (Remove-Item Env:VICTORY_*).
$env:VICTORY_LIGHT_REFRESH=1; cargo test -p victory-gpu --lib -- watch:: safecopy::
$env:VICTORY_GUARD_BITS=1; cargo test -p victory-gpu --lib -- watch:: safecopy::
$env:VICTORY_OPEN_ONE_VIEW=1; cargo test -p victory-gpu --lib -- watch:: safecopy::
$env:VICTORY_GUARD_REST=3; cargo test -p victory-gpu --lib -- watch:: safecopy::
$env:VICTORY_LIGHT_REFRESH=1; $env:VICTORY_KNOWN_HOLES=1; cargo test -p victory-gpu --lib -- watch:: safecopy::
# All of #5 together, as cool_fox runs it, plus safecopy and nothing-new:
$env:VICTORY_LIGHT_REFRESH=1; $env:VICTORY_GUARD_BITS=1; $env:VICTORY_OPEN_ONE_VIEW=1; $env:VICTORY_GUARD_REST=3; $env:VICTORY_KNOWN_HOLES=1; $env:VICTORY_NOTHING_NEW=1; $env:VICTORY_SAFE_COPY=1; cargo test -p victory-gpu --lib -- watch:: safecopy::

# 3. The ignored cost measurement of safecopy (optional):
cargo test -p victory-gpu --lib --release -- safecopy:: --ignored --nocapture
```

Some tests check more, or check anything at all, only when switches are set:

- `a_write_opens_the_view_it_goes_through` only checks with `VICTORY_OPEN_ONE_VIEW=1`. Without it, it returns at once.
- `writes_to_views_are_not_lost` (`no_write_to_views_is_lost_between_askers`, `no_write_through_another_view_is_lost_between_askers`, `no_write_to_resting_views_is_lost_between_askers`) always checks that no write is lost. Only with `VICTORY_LIGHT_REFRESH=1` does it also check that the bits answered some askings.
- `a_block_just_written_rests` and `no_write_to_resting_views_is_lost_between_askers` set their rest per thread, so they test the rest whatever `VICTORY_GUARD_REST` says.
- `holes_are_known_until_the_runtime_tells` checks its answers in every configuration, but checks the bits only with `VICTORY_KNOWN_HOLES=1` and `VICTORY_LIGHT_REFRESH=1` both set.
- `unchanged_ranges_are_known` only checks that view memory is known by its bits with `VICTORY_LIGHT_REFRESH=1`. Without it, it checks that view memory is never known that way.
- `a_write_told_at_a_view_that_cannot_be_guarded_is_not_lost` (the review fix) always checks that the told write is reported. With `VICTORY_LIGHT_REFRESH=1`, `VICTORY_OPEN_ONE_VIEW=1` or a rest, it also checks that the block is never clean or published while the other view is writable. Run it with `VICTORY_LIGHT_REFRESH=1` alone: that is the case the fix is for. It assumes that `VirtualProtect` of a 64 KiB block whose view has only one 4 KiB page committed fails without changing anything. If it fails on the first `guard` asking, check that assumption first.
- `the_switch_is_off_by_default` (safecopy) only checks when `VICTORY_SAFE_COPY` is unset.

The D3D12 / WARP suites are untouched by this lane, but run them once with
the switches on to see that nothing else moved.

## Risks to check on the PC

- **Staleness.** Run the game with `VICTORY_REFRESH_CHECK=1` together with
  `VICTORY_LIGHT_REFRESH=1` and/or `VICTORY_OPEN_ONE_VIEW=1` (and once with
  `VICTORY_GUARD_REST=3`). Watch the log for "a host copy of title memory is
  stale after its refresh". Any such warning means a write the guards or the
  bits missed. Compare with the same run without the switches: warnings
  that also show up there are not this lane's.
- **Holes filled with nobody telling.** With `VICTORY_KNOWN_HOLES=1`, watch
  for "a placeholder of the title's was filled with nobody telling". It is
  logged once. It means the runtime committed into a placeholder without
  `title_memory_remapped` or `title_memory_allocated`, so held holes can be
  stale for up to a hundred frames. Report it with the context around it.
- **safecopy and other vectored handlers.** With `VICTORY_SAFE_COPY=1`, a
  fault inside the copy is only turned into a failed copy because our
  handler (`watch::install`, `on_write`) is the first vectored handler in
  the process. If anything ever registers a vectored handler ahead of ours
  (an overlay, a debugger helper, a later change of ours), a fault in a
  copy reaches that handler instead and can crash the process. Check for
  crashes with an access violation whose address is in `safecopy`'s leaf.
- **Map-before-tell window.** cool_fox's test
  `no_mapping_is_lost_between_askers` is not ported. On our line, a view
  that is mapped but not yet told (`title_memory_remapped` comes after the
  map) can be guarded by an asking that found its hole committed. The
  telling then forgets that guard while the page stays read-only. This
  window exists without any switch. `VICTORY_KNOWN_HOLES=1` does not widen
  it, as far as I can tell, but streaming-heavy scenes are where it would
  show up: faults on pages nobody guards, or stale copies. cool_fox closed
  it in 26d818c, which is not part of this lane.
- **Lock let go between runs** (`VICTORY_GUARD_BITS=1`). This is safe only
  because the journal's lock or the epoch span locks keep other askings off
  the same blocks. If a run shows stale copies or a hang with GUARD_BITS
  alone, report it.
- **Performance.** Each switch should help on its own (cool_fox's numbers
  are in the commit messages). If one makes frames slower, with the guards'
  share in the `VICTORY_R5_FLUSH` counters going up, report which one.
