# VIctory

Runs Grand Theft Auto VI (Xbox Series X|S) natively on Windows, from files
the user supplies.

The title's x86-64 code runs unchanged inside our process; VIctory supplies
the platform underneath it and converts the title's data, ahead of time where
it can and on first use where it must (shaders). See
[docs/architecture.md](docs/architecture.md).

## Status (2026-11-23)

**Not playable yet.** The title boots, runs its menus at its own 30 fps and
loads the world, where rendering fails.

| Stage | State |
|-------|-------|
| Boot, first-boot screens (accessibility, language, settings, legal) | works, 30 fps |
| Intro video with sound | works |
| Start screen, profile and saves | works |
| Loading into the world (about 50 s) | works |
| The world (first mission's intro) | reached and runs with indirect draws; its title card renders, the scene after it is still dark; 0.5-1.8 s per frame |

Tested on one machine: Windows 11, AMD RX 6700 XT.

> **Warning.** Entering the world on a real GPU can currently hang it, and
> Windows then resets the display driver (every monitor goes black for a
> moment). To look further, use a diagnostics build with
> `VICTORY_D3D12_WARP=1`, which renders on the software adapter.

What each part does today and what it still needs:

| Subsystem | Today | Needs |
|-----------|-------|-------|
| Loader | Runs the title's executable and its DLLs in our process; crash reports name the faulting code and are kept in `crashes.log` | – |
| Runtime services | One signed-in offline user, licences, packages, saves that survive a kill at any step, text entry dialog | Speech is silent; the dialog and relaunch are untested live |
| Network | Offline by design: the title cannot reach the network | – |
| Files and streaming | Archives and streamed data read, zlib and BCPack decoded in parallel | Loading is limited by writes into title memory |
| Input | XInput pads, keyboard, keyboard stand-in pad | Trigger rumble |
| Audio | Menu, video and loading sound; XMA voices and Opus streams decode | Reverb is silent; Opus results unverified against the title |
| GPU: menus | Draws, compute and presentation on host D3D12; only changed title memory is copied | Frame pacing drifts on one screen in release builds |
| GPU: world | Mesh pipelines, shader-written buffers, descriptor lifetimes, indirect execution (arguments clamped), pipelines made by worker threads and kept on disk | Most of the scene is still missing (untranslated shaders drop draws); frames take seconds; real queries, depth bounds, HDR and scaling are missing |
| Shaders | Console GPU code recompiled to DXIL at first use, cached on disk. Menus: all. World: mesh 8,191 of 8,506, vertex/pixel 10,617 of 12,346, compute 1,643 of 2,539 | 64-bit image atomics, merged geometry stages, the ray-tracing builder's shaders; some compute programs need a GPU with 64-lane waves |
| Textures | Console tiled layouts and detiling for the modes the title uses | Render-target metadata planes |
| Series S mode | Runs through the menus to the loading screen | The world, as above |

## Getting started

Requirements: Windows 10/11 x64, a CPU with AVX2, a Direct3D 12 GPU with
shader model 6.6 (and mesh shaders for the world), Rust (pinned in
`rust-toolchain.toml`).

The title also needs several GiB of available Windows commit at startup.
Check **Committed** under **Task Manager → Performance → Memory**. If
`crashes.log` reports Windows error 1455, close other applications or increase
the Windows paging file size, then retry.

```powershell
cargo xtask install-hooks                # asset-policy check on every commit
copy victory.example.toml victory.toml   # then set your paths (outside the repo)
cargo test --workspace                   # needs no game data
cargo run -p victory -- config           # check your paths
```

Run the title:

```powershell
cargo build --release -p victory
target\release\victory.exe run <game dir>\<the title's exe> --execute
```

The window takes the title's size (the size of the frames it renders),
shrunk to the primary screen. Set `[window] size` in `victory.toml` (or
`VICTORY_WINDOW_SIZE`) to `"1920x1080"` or any other client size, or to
`"fullscreen"` for a borderless window over the whole screen (Alt+F4 closes
it). Sizes are screen pixels: `victory` is DPI aware, so Windows display
scaling does not enlarge the window. This sizes only the window: the title
renders at its own size and its frame is stretched to the window, so a size
with another aspect ratio distorts the picture. A window the title shows
itself is only shrunk to fit the screen. Closing the window ends the title
as a console does: it is suspended first, so it can save.

The window caption shows the framerate. Saves, settings and `crashes.log`
are kept under the data folder (default `%LOCALAPPDATA%\victory\data`), derived
data under the cache folder.

Input: XInput pads. With none connected the keyboard is a pad while the
title's window has focus:

| Keys | Pad |
|------|-----|
| W A S D / I J K L | left / right stick |
| arrows | D-pad |
| Enter, Backspace, X, Y | A, B, X, Y |
| Q / E, Z / C | shoulders, triggers |
| F / H | stick clicks |
| Esc, Tab | Menu, View |

## Commands

```
victory inspect <exe|dll> [--imports] [--exports]
victory census [paths...] [--missing] [--strict]
victory lib-exports <import.lib> [--list]
victory shader-info [paths...] [--verbose] [--disasm]
victory translate [paths...] [--validate] [--pipelines] [--list] [--jobs N]
victory run <exe> [--execute] [--unimplemented trap|zero] [--strict]
            [--console series-x|series-s] [-- <title arguments>]
victory config
```

Logging: `VICTORY_LOG=victory_loader=trace,info` (a release build logs
info and above).

Host drivers that crash compiling a shader: with `VICTORY_PIPELINE_JOURNAL=1`
a pipeline whose creation ends the process is refused from the next run on
(its draws dropped, as for any refused pipeline) and listed in
`pipeline-journal\crashed.txt` under the cache folder. Each such pipeline
costs one crash; delete its line to try it again. Pipelines are still created
in parallel; when a crash leaves several in creation, they are listed in
`pipeline-journal\suspects.txt` and each is created alone until it succeeds,
so a repeat crash names only the pipeline that caused it.

## Debugging

A release build has no debugging aids. Build them in with

```powershell
cargo build --profile diag -p victory --features diagnostics   # target\diag\victory.exe
```

| Variable | Effect |
|----------|--------|
| `VICTORY_LOG=<target>=debug,...` | debug and trace output |
| `VICTORY_STACKS=20,60` | every thread's stack at those seconds |
| `VICTORY_WATCH=<rva>,...` | logs a title u32 when it changes |
| `VICTORY_KEYS=30:Enter,35:X` | presses keys at those seconds, focus or not (in release builds too, so unattended runs get past the start screen) |
| `VICTORY_CRASH_SAVE=<rva>:<len>,...` | saves title memory to the cache folder on a crash |
| `VICTORY_CAPTURE=<dir>` | saves presented frames (powers of two, or `VICTORY_CAPTURE_FRAMES=<list>`) |
| `VICTORY_CAPTURE_MENUS=1` | with `VICTORY_CAPTURE_WORLD`, also two frames of every 16 before the world (off by default: saving them makes the menus stutter) |
| `VICTORY_DUMP_TARGETS=1` | with the two above, saves every render target of those frames |
| `VICTORY_D3D12_DEBUG=1` | host debug layer |
| `VICTORY_D3D12_GBV=1` | with the above, GPU-based validation (bad descriptor use reported, not faulted; slow) |
| `VICTORY_D3D12_DRED=1` | on a GPU failure, also the text of the commands in flight and the names of the allocations hit (slow) |
| `VICTORY_SKIP_PIPELINES=<hash>,...` | omit draws and dispatches, including indirect ones, whose pipeline contains a listed 16-digit shader hash (release builds too), such as a program the card's driver never finishes compiling |
| `VICTORY_PIPELINE_WAIT_OWN=0` | only the first thread to wait for a pipeline in a frame gets the loading-time extension of the wait, as before (release builds too); by default every thread of the frame waits for its own pipeline within it, so a pass the title runs once is not dropped because another thread was waiting |
| `VICTORY_GPU_TIMING=1` | with the above, each list of a submission run and timed on its own; slow lists logged with their slowest commands |
| `VICTORY_FLIP=<n>` | compares two arms in one run (release builds too): one `flip` line a second, the arm changing every `n` lines in the order ABBA. A switch given a pair `a\|b` (`VICTORY_TEXTURE_STAGE_MB`, `VICTORY_TEXTURE_STAGE_MS`, `VICTORY_TEXTURE_NO_WAIT`) takes `a` under the first arm and `b` under the second |
| `VICTORY_D3D12_WARP=1` | host device on the software adapter (slow; cannot reset the driver) |
| `VICTORY_SWEEP_SLICE_MS=2` | the once-a-second sweep that lets go of idle host copies, staging buffers and the pipeline library's mapping (a pause of tens of milliseconds) runs in slices of about that many milliseconds (1 to 1000), one per submission at the same moment and by the same rules, until it is done; a submission that settles the flights only for the sweep (`VICTORY_LATE_WAIT=3`) still runs what is left of it at once (release builds too; off by default) |
| `VICTORY_TEXTURE_SWEEP=0` | no paced look over the heap's placed texture views for ones nothing brings up to date (`count`: only count them; default on, putting them back in the queue) |
| `VICTORY_HEAP_RECHECK=0` | a texture picked by index is uploaded once, not looked at again 4, 12, 40, 120 and 400 frames after its upload for memory the title filled after writing its view |
| `VICTORY_HEAP_CLAIM=0` | every list that binds a shader picking buffers from a view heap waits for the lock of the frame's heap buffer pass, and so for its whole walk, as before (release builds too); by default the pass is claimed per heap first and the other lists of the frame go on recording |
| `VICTORY_HEAP_WAIT_AT_SUBMIT=0` | with the claim, a list that lost it still waits for the walk, but while it records: at that bind, until the walk has ended (not for the walk's lock, as before the claim; release builds too); by default its submission waits for the walk instead, for 10 s at most |
| `VICTORY_HEAP_WALK_ONCE=0` | the walk of the heap buffer pass marks the slots it makes as changed, so the next pass walks them again, as before (release builds too); by default a slot the walk made is walked again only once something writes or makes it anew |
| `VICTORY_VIEW_GUARD=0` | a guarded copy of title memory does not see writes through another view of the same pages (on by default; a page shown in more than 64 views is guarded at each view by itself) |
| `VICTORY_EARLY_TEXTURES_MS=12` / `VICTORY_BAKE_TEXTURES_MS=1500` | milliseconds a frame for uploading textures picked by index before any shader asks for them, and before the draws of the title's decoration bake (`0`: each off) |
| `VICTORY_TEXTURE_ROOM_PERCENT=90` | share of the card's memory budget up to which textures picked by index get host copies (50 to 97) |
| `VICTORY_PLACED_FILLER=1` | placed textures keyed with the filler words of their description too, as before (two views that differ only there get two host copies) |
| `VICTORY_RELAY=1` / `VICTORY_RELAY=2` | what shaders wrote is copied back by a thread of its own, beside the submitting thread, at its first or second stage (off by default: with the second stage on by default an RX 9070 XT lost the device in a run while a submission was sent ahead) |
| `VICTORY_ROOT_CBV_REACH=0` | a root constant buffer view near the end of a window of memory outside buffers is bound from the window its address names, as before (release builds too); by default it is bound from a window that holds all 64 KiB a constant buffer can read, and an unaligned one takes 64 KiB of the list's scratch. Either way the memory line every 5 s counts such views, and those near the end of a title buffer's copy |
| `VICTORY_ROOT_CBV_TABLES=0` | root constant buffer views are bound as root descriptors, which have no size, as before (release builds too); by default each is a one-view table whose view holds only what its host copy has from its address on (at most 64 KiB), so a shader's read at a computed offset past the copy gives zeros instead of reaching whatever the host placed after it (a suspected cause of page faults inside freed allocations). A read more than 64 KiB from the root address gives zeros too, also where the copy goes on. A table needs a view heap: a draw whose title bound none is dropped (a root descriptor needed none). The memory line every 5 s counts the tables, those that hold less than 64 KiB, and the draws dropped for want of a view heap or a free descriptor slot |
| `VICTORY_LIFT_READS=count` / `1` / `2` | with `VICTORY_RELAY=2`, a submission whose title buffer copies or fills of upload memory read what submissions still in the air write does those reads on the card, behind them, instead of waiting for them to land: `count` only plans and counts (a line with the frame rate, aborts by reason), `1` lifts the copies, `2` the fills too (the lists' scratch chunks then lie in shared memory). Off by default: unmeasured |
| `VICTORY_FOLD_BUFFERS=1` | small title buffers inside one window cell share the cell's window as their copy (off by default: on an RX 9070 XT the title stayed on the loading screen with it on) |
| `VICTORY_CARD_HASH_WIDE=1` | the passes that find what shaders wrote to copies in the card's memory hash with a thread group per 64 KiB piece, not one thread, and those before a submission's lists run after all its copies, behind one barrier (off by default: a compute pipeline of a new shape, not yet run on an RX 9070 XT) |
| `VICTORY_EPOCH_JOURNAL=1` | what the write watch and the guards found the title wrote is kept as a number per 4 KiB page (tables of 1 GiB of addresses, made as pages there are written) instead of in one map behind one lock: askings about different memory lock only their own 16 MB spans (counted modulo 4 GiB, so spans that far apart share a lock), a submission's runs asked ahead go to the threads kept for the refreshes, and the refreshes answered by the asking that stands take no lock but a read-mostly one, so `VICTORY_WIDE_REFRESH=1` refreshes no longer wait for each other there (release builds too; off by default: not yet run on an RX 9070 XT). The host copies keep their 64 KiB grain |
| `VICTORY_UPSCALE_STANDIN=1` | in Series S mode, the title's upscaler (AMD FidelityFX, whose console-compiled passes do not run here, so the picture stays black) is stood in for: each upscale draws the rendered part of its colour input over its output with linear filtering, then the title's state is set again on the list; other dispatches go to the upscaler (release builds too; off by default: softer picture than the upscaler's, not yet run on an RX 9070 XT). Use with `VICTORY_CLEAR_TARGETS_ONLY=1` |
| `VICTORY_CLEAR_TARGETS_ONLY=1` | the title's clear value is passed to the host only for render and depth targets, so a texture only shaders write that comes with one (the upscaler target in Series S mode) is made instead of refused (release builds too; off by default) |
| `VICTORY_GROUPED_MESH_CLAMP=1` | a run of up to 32 indirect mesh draws of one record each, from one argument buffer, takes one argument clamp (one compute dispatch between two waits) instead of one each; every record is bounded and charged to the list's budget as by a clamp of its own, in draw order, and any other draw or dispatch, clamp, barrier, copy, clear, value write, change of targets or the list's close ends the run first (release builds too; off by default: not yet run on an RX 9070 XT). Works with or without `VICTORY_INDIRECT_WINDOW`, `VICTORY_INDIRECT_READABLE`, `VICTORY_CLAMP_GROUP`, `VICTORY_INDIRECT_CARD_BUDGET` and `VICTORY_CLAMP_LEAN` |
| `VICTORY_CLAMP_PRECISE=1` | an indirect argument clamp keeps only the waits it needs instead of a wait for every earlier shader write before and after its dispatch: none after (the move of its output to indirect arguments orders it for the execution, and the next clamp waits for the budget), and before it one for every earlier write only if something recorded since the list's last such wait may have written memory (a draw or dispatch whose shaders store, a command of the title's that may write, a buffer moved to be written), one for the budget alone if only clamps wrote, else none; what each record is clamped to and charged is the same (release builds too; off by default: a shader write its translation does not know of is then no longer waited for at every clamp; not yet run on an RX 9070 XT) |
| `VICTORY_FRAMETIME_DUMP=1` | writes `<cache>\frametime.csv`, one row per summary interval with the frame buckets and the worst and mean frame (any other value is a path; off by default) |
| `VICTORY_PROFILE=1` | writes `<cache>\profile.json`, a Chrome trace (opens in Perfetto) of what each thread does per frame (submission phases, list recording, heap passes, pipeline waits, presents) and when the card ran each host list (release builds too; any other value is a path; off by default). Starts at the first frame of over 1,000 draws (`VICTORY_PROFILE_DELAY=<s>`: that many seconds after the title's first command list instead) and records `VICTORY_PROFILE_SECONDS` (60) or 512 MB. Each frame also gets counters: global UAV barriers by origin, the refreshes of host copies and texture staging split into write-watch askings, journal work and copies (thread time summed over the refreshing threads, so with `VICTORY_WIDE_REFRESH` the parts can add up to more than the flush's wall time; with the written pages at 4 KiB and 64 KiB grain), and host view slots changed against the distinct records they changed to; each submission says which of its lists closed last, on which thread, and its waits before preparing |
| `VICTORY_PROFILE_FLIGHT_STAMPS=1` | with `VICTORY_PROFILE`, submissions sent ahead (`VICTORY_LATE_WAIT`, the relay) are stamped on the card too, each into a read-back slot of its own read when it lands, so relay runs get a card line (release builds too; off by default: it adds the stamps' small lists to every flight while recording) |
| `VICTORY_PROFILE_SAMPLES=1` | with `VICTORY_PROFILE`, the profile's writer thread adds every second each recording thread's processor time (to compare with what our spans cover) and every two seconds the memory split (the title's registered page pools and how much of them is resident, host copies, windows, snapshots, copies in the card's memory, texture staging, the lists' upload memory) and the distinct views the view heaps' slots are made from (release builds too; off by default: the samples copy each view heap's slots under its lock and ask Windows about every pool page) |
| `VICTORY_WIDE_BACK=0`, `VICTORY_CARD_BUFFERS_MIN_MB=16`, `VICTORY_CARD_BUFFERS_MB=512`, `VICTORY_LATE_WAIT_LISTS=128` | the earlier defaults: without the wide copy, card copies of title buffers from 16 MiB and up to 512 MB, 128 lists in the air (with `VICTORY_LATE_WAIT` or the relay) |
| `VICTORY_NULL_MAGENTA=1` | diagnostics build: a 2D colour texture view with nothing to show yet reads magenta instead of black |
| `VICTORY_SHADER_LOOP_LIMIT=<n>` | translated shader loops stop after `n` iterations per lane instead of 2^20 (own shader cache entries); if a lower `n` makes world frames much faster, loops hitting the limit are what makes them slow |
| `VICTORY_AUDIO_CAPTURE=<file.wav>` | records what the title plays (our process only) |
| `VICTORY_XMA_DUMP=<dir>` | saves the first 32 XMA voices' context and input |
| `VICTORY_CAPTION=<text>` | replaces the caption of the window we present in and of the title's own top-level windows (default `VIctory` for ours), e.g. a neutral one while the screen is shared |

Every build appends crashes and host GPU failures to `crashes.log` in the
title's data folder. For a removed host device it writes the driver's
device-removed data: the faulting address and its class, the draw or
dispatch each list stopped at with its pipeline, and the last ones sent
(`VICTORY_D3D12_DRED=lite`, the default; `0` turns it off).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
