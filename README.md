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
| `VICTORY_MADE_ATOMIC=1` | a view heap's host slots are made a slot at a time, each claimed by the thread that makes it, with no lock over the whole heap: draws, the heap buffer pass, the texture pass and the sync before a submission make different slots at once, and the host descriptor is made outside any heap-wide lock; what a write changes in the queue of textures picked by index and in the sets of buffer views is logged per thread and applied in the order it was logged by whatever reads them next, or by a writer once a log is long (release builds too). Off by default: one lock per heap is held across all of it, as before. Waits for a slot another thread is making go on the profile as `made claim wait` |
| `VICTORY_VIEW_GUARD=0` | a guarded copy of title memory does not see writes through another view of the same pages (on by default; a page shown in more than 64 views is guarded at each view by itself) |
| `VICTORY_EARLY_TEXTURES_MS=12` / `VICTORY_BAKE_TEXTURES_MS=1500` | milliseconds a frame for uploading textures picked by index before any shader asks for them, and before the draws of the title's decoration bake (`0`: each off) |
| `VICTORY_TEXTURE_ROOM_PERCENT=90` | share of the card's memory budget up to which textures picked by index get host copies (50 to 97) |
| `VICTORY_TEXTURE_STAGING_TRANSIENT=1` | a texture the title fills keeps its staging buffer (system memory the size of all its levels, which the CPU detiles into and the card copies from) only until the list that uploads it has run, instead of for as long as it is staged: the buffer goes back, at the once-a-second sweep, to a pool of buffers by size class (quarter steps of a power of two from 64 KiB) that later uploads take from; the pool keeps at most `VICTORY_TEXTURE_STAGING_POOL_MB` (256; `0`: none) and lets the oldest go past it. A texture uploaded again (one picked by index brought up to date, a level on demand asked again) is detiled again into the buffer it is given, by the check at the submission of the list that asked, and copied by the next list that uploads it (one picked by index stays due for the texture pass until then; until then its host copy keeps the texels it has, so memory that cannot be read at that moment leaves no zeros). The frame line of texture refreshes says, with the switch, the buffers live and pooled and how many were made, reused, released after their upload, let go by the cap and detiled again, and the uploads left to a later list (release builds too; off by default: not yet run on an RX 9070 XT) |
| `VICTORY_PLACED_FILLER=1` | placed textures keyed with the filler words of their description too, as before (two views that differ only there get two host copies) |
| `VICTORY_RELAY=1` / `VICTORY_RELAY=2` | what shaders wrote is copied back by a thread of its own, beside the submitting thread, at its first or second stage (off by default: with the second stage on by default an RX 9070 XT lost the device in a run while a submission was sent ahead) |
| `VICTORY_ROOT_CBV_REACH=0` | a root constant buffer view near the end of a window of memory outside buffers is bound from the window its address names, as before (release builds too); by default it is bound from a window that holds all 64 KiB a constant buffer can read, and an unaligned one takes 64 KiB of the list's scratch. Either way the memory line every 5 s counts such views, and those near the end of a title buffer's copy |
| `VICTORY_ROOT_CBV_TABLES=0` | root constant buffer views are bound as root descriptors, which have no size, as before (release builds too); by default each is a one-view table whose view holds only what its host copy has from its address on (at most 64 KiB), so a shader's read at a computed offset past the copy gives zeros instead of reaching whatever the host placed after it (a suspected cause of page faults inside freed allocations). A read more than 64 KiB from the root address gives zeros too, also where the copy goes on. A table needs a view heap: a draw whose title bound none is dropped (a root descriptor needed none). The memory line every 5 s counts the tables, those that hold less than 64 KiB, and the draws dropped for want of a view heap or a free descriptor slot |
| `VICTORY_LIFT_READS=count` / `1` / `2` | with `VICTORY_RELAY=2`, a submission whose title buffer copies or fills of upload memory read what submissions still in the air write does those reads on the card, behind them, instead of waiting for them to land: `count` only plans and counts (a line with the frame rate, aborts by reason), `1` lifts the copies, `2` the fills too (the lists' scratch chunks then lie in shared memory); with `VICTORY_R5_LIFT=1` the mode is `2`, whatever this says. Off by default: unmeasured |
| `VICTORY_FOLD_BUFFERS=1` | small title buffers inside one window cell share the cell's window as their copy (off by default: on an RX 9070 XT the title stayed on the loading screen with it on) |
| `VICTORY_CARD_HASH_WIDE=1` | the passes that find what shaders wrote to copies in the card's memory hash with a thread group per 64 KiB piece, not one thread, and those before a submission's lists run after all its copies, behind one barrier (off by default: a compute pipeline of a new shape, not yet run on an RX 9070 XT) |
| `VICTORY_EPOCH_JOURNAL=1` | what the write watch and the guards found the title wrote is kept as a number per 4 KiB page (tables of 1 GiB of addresses, made as pages there are written) instead of in one map behind one lock: askings about different memory lock only their own 16 MB spans (counted modulo 4 GiB, so spans that far apart share a lock), a submission's runs asked ahead go to the threads kept for the refreshes, and the refreshes answered by the asking that stands take no lock but a read-mostly one, so `VICTORY_WIDE_REFRESH=1` refreshes no longer wait for each other there (release builds too; off by default: not yet run on an RX 9070 XT). The host copies keep their 64 KiB grain |
| `VICTORY_MEMORY_CORE=1` | title buffers and windows in the title's own private GPU memory (an `XMemVirtualAlloc` region, committed read-write) are read and written by the card in place, through a heap opened on the title's pages and one buffer placed over each, instead of through a host copy kept current every submission: no refresh, write-watch asking, snapshot or copy-back for them, and no buffer of ours in the PC's memory beside the title's. A buffer inside such a home, or a window over it, is the same home. Big buffers given a copy in the card's memory keep it; one the card turns down is used in place. The title's `VirtualFree` and `VirtualProtect` on such pages let the placement go first (a free still refused is done once nothing holds the heap, or given up after 30 s with the region not used in place again; a protection still refused fails with error 87 and the region is not used in place again). Anything the host refuses keeps its copy; `VICTORY_INPLACE` is ignored. A line every ten seconds says what it did, with a census of the regions by the console's memory-type bits. `VICTORY_MEMORY_CORE_MB` caps the title memory it pins (4,096) (release builds too; off by default: the openings are a driver path nothing else uses) |
| `VICTORY_MEMORY_POOL=1` | with `VICTORY_MEMORY_CORE=1`: the title's physical pages are made in chunks of 256 MiB (`VICTORY_MEMORY_POOL_CHUNK_MB` below), each a section of its own made on the first map or write into it and committed whole when the PC's commit charge has room for it (with 1 GiB to spare), and each such chunk is opened as a heap, with one buffer placed over all of it, when a home is first wanted in it (and the title maps its share of it, `VICTORY_MEMORY_POOL_OPEN_PCT` below) and only if the shared part of the card's budget keeps 1 GiB to spare after it. A title buffer or window whose bytes are one run of one chunk's pages at one view is read and written by the card in place, at its offset in that buffer: no refresh, guard, snapshot or copy-back for it, and nothing pinned. Mapping other pages at its addresses lets it go, and the views of it the title wrote into descriptor heaps are made anew at their next use (a draw that reads them, or the pass over views shaders pick by index); placed textures over pages the card wrote there are seen as changed and uploaded again. Views over several chunks or with holes, chunks made reserved (no room, Windows before build 20348, or a process where the core can make no home), chunks the shared budget has no room for yet and anything the host refuses keep their copies, counted on the memory core's line with the chunks made, opened (and the slowest opening) and the MB in such homes. The pool's commit charge can pass what the title touches by up to a chunk's size per chunk, and the card holds an opened chunk resident whole however few homes it has (release builds too; off by default; without `VICTORY_MEMORY_CORE=1` it is ignored) |
| `VICTORY_MEMORY_POOL_CHUNK_MB=16` / `32` / `64` / `128` | with `VICTORY_MEMORY_POOL=1`: the pool's chunks are of that many MiB instead of 256. A chunk is committed whole when the title first touches it and opened whole for its first home, so smaller chunks keep the commit charge, the shared part of the card's budget and the working set closer to what the title touches and the homes need; more chunks are opened, and a buffer or window over two of them keeps its copy. The memory pool's line counts the chunks committed and opened by size and the MB committed and opened now against the MB in homes (release builds too; 256 by default; any other value is warned of and 256 used) |
| `VICTORY_MEMORY_POOL_OPEN_PCT=50` | with `VICTORY_MEMORY_POOL=1`: a chunk is opened as a heap only once the title maps at least that share of its pages (1 to 100) at some view; until then its buffers and windows keep their copies, counted on the memory core's line as "mapped below VICTORY_MEMORY_POOL_OPEN_PCT", and copies made then stay copies once it opens. An opened chunk stays opened when the title maps less of it again. It holds back the opening only: the chunk is still committed whole when first touched (release builds too; 0 by default: opened at the first home whatever is mapped) |
| `VICTORY_MEMORY_CARDS=1` | with `VICTORY_MEMORY_CORE=1`: a big title buffer with a copy in the card's memory whose bytes get a home of the core has that home, the title's own memory, as the copy's CPU side instead of a staging buffer in the PC's memory: the refresh copies nothing (the pages the title wrote go up from the home on the card), and what shaders wrote is merged into the home on the card behind the compare, a byte from the card where it changed during the submission and the title's elsewhere, instead of fetched and merged on the CPU after another wait. For that the declared ranges' card bytes are copied aside on the card before the lists, into a buffer that grows to the most a submission needed and is kept (in the card's memory up to 512 MB, else in the PC's, and back in the card's once 16 submissions in a row needed no more than that). A copy declared written for half of it or more (as a copy the shaders reach by raw address is, whole, for every list with writes) instead keeps a reference of its own size in the card's memory (up to 512 MB for all of them, let go after 64 submissions that declared none of it), into which only what went up to its card side is copied. A copy whose bytes get no home keeps its staging buffer. The homes of such copies count against `VICTORY_MEMORY_CORE_MB`, and their pages are pinned as every home's: the title's `VirtualProtect` and `VirtualFree` there take the core's path (a protection may be refused, a free may wait), where with a staging buffer both just worked. With `VICTORY_MEMORY_POOL=1` the home can be one in a chunk of the pool: it pins nothing and counts against no cap, a remap of its addresses lets it and the copy go, and what is merged into it is told at its other views. A line every ten seconds says what it did (release builds too; off by default: the kept references not yet run on an RX 9070 XT) |
| `VICTORY_MEMORY_DIRECTORY=1` | with `VICTORY_MEMORY_CORE=1`: the directory raw-address shaders find title memory by (the ray-tracing structure builder and what walks its structures) resolves pool memory page by page: with `VICTORY_MEMORY_POOL=1`, a page of a committed chunk is read and written by the card in place through the buffer over the whole chunk, at the page's offset in it, so memory whose views come from several runs or chunks needs no copy for these shaders (a cell of 4 MiB at a time, as a window's pages). Mapping other pages there changes those entries, and the table the shaders read takes every change only by a copy at the head of the next submission, so each submission sees it whole. After a submission whose shaders could store through it, copies of those pages at other views take them again. View memory bound as one view or descriptor over several places keeps its copy, and these shaders use that copy for its pages too, so both see the same bytes; counted on the line. A chunk the title lets go loses its entries at the next dispatch or submission. Shortcuts: a list recorded earlier in a submission than the dispatch at which a chunk page switched to a copy made later stores into that copy without declaring it written, so those stores are not copied back (as with any copy numbered after a list's last dispatch); every chunk page counts as written after a submission whose shaders could store; a page the title maps into a chunk that has no number yet is empty for the submission at whose head that change is taken (these shaders read zeros there and their stores are dropped) and is numbered at the next dispatch; numbers let go are not given again. A line every ten seconds with the memory core's says what was resolved and why the rest kept copies (release builds too; off by default: not yet run on an RX 9070 XT; ignored without `VICTORY_MEMORY_CORE=1`, and with late waits or with the relay unless `VICTORY_R5=1`) |
| `VICTORY_MEMORY_OWN=1` | with `VICTORY_MEMORY_CORE=1`: the memory we hand the title for its D3D12 heaps, its descriptor heaps, committed resources when it names no allocator, and `XMemAllocDefault`'s blocks of 64 KiB or more is a region of its own below 128 GiB (as far as raw-address shaders reach), 64 KiB aligned and write-watched, instead of process-heap memory above 2^40: the memory core can use title buffers in it in place, and its copies copy only written pages. Such a region is given back through the core (its homes let go first, the release waiting while a list still holds one): at once by `XMemFreeDefault`, at the next sweep for heaps and resources. Every `XMemAllocDefault` memory type takes it, CPU-only ones too. Counted on the memory core's line and census (release builds too; off by default: not yet run on an RX 9070 XT; ignored without `VICTORY_MEMORY_CORE=1`) |
| `VICTORY_R5=1` | with `VICTORY_MEMORY_CORE=1` and `VICTORY_RELAY=2`: the memory core gives homes while submissions are sent ahead too, and the thread that lands them does what homes need: copies over bytes a flight's shaders wrote in a home are told before the title is told the flight is done; a copy over a home is not refreshed while a flight may write the home (a card copy with a home sends those bytes up from the home again, behind the flight, except what its own shaders wrote); the directory of raw-address shaders and the references of card copies with a home are set up per flight; card copies with a home land with no fetch; a protection of the title's over a home a flight holds waits for the flights in the air before it is refused; a wait for a memory value, or a blocking wait for a fence, waits only for the flights concerned. What waited is counted per frame on the frame-rate line (release builds too; off by default: the openings and the overlap are both driver paths that need measuring on the card) |
| `VICTORY_R5_LIFT=1` | with `VICTORY_R5=1`: a submission that would wait for submissions still in the air because it reads title memory they may write does those reads on the card instead, at its head and behind them on the queue: title buffer copies and fills of upload memory are lifted as with `VICTORY_LIFT_READS=2` (that switch's mode is then `2`, whatever it says), now also where the bytes are a home's, read from or written into the home's own buffer (through a bounce buffer between two homes in one chunk of the pool; a copy into a home lands as a shader's write does, and the copies over it are told then), and a refresh of a copy that is no home over a home's bytes such a submission writes is done on the CPU and then copied over from the home on the card for those bytes. What still waits does as before and is counted: a texture or a step of ours over those bytes, a copy over a lifted copy's home that the submission refreshes, a refresh whose copy the same submission's shaders, or those of a submission still in the air, write there too. The R5 line on the frame-rate line splits the waits for title memory read by kind, with their milliseconds (release builds too; off by default: unmeasured) |
| `VICTORY_UPSCALE_STANDIN=1` | in Series S mode, the title's upscaler (AMD FidelityFX, whose console-compiled passes do not run here, so the picture stays black) is stood in for: each upscale draws the rendered part of its colour input over its output with linear filtering, then the title's state is set again on the list; other dispatches go to the upscaler (release builds too; off by default: softer picture than the upscaler's, not yet run on an RX 9070 XT). Use with `VICTORY_CLEAR_TARGETS_ONLY=1` |
| `VICTORY_CLEAR_TARGETS_ONLY=1` | the title's clear value is passed to the host only for render and depth targets, so a texture only shaders write that comes with one (the upscaler target in Series S mode) is made instead of refused (release builds too; off by default) |
| `VICTORY_GROUPED_MESH_CLAMP=1` | a run of up to 32 indirect mesh draws of one record each, from one argument buffer, takes one argument clamp (one compute dispatch between two waits) instead of one each; every record is bounded and charged to the list's budget as by a clamp of its own, in draw order, and any other draw or dispatch, clamp, barrier, copy, clear, value write, change of targets or the list's close ends the run first (release builds too; off by default: not yet run on an RX 9070 XT). Works with or without `VICTORY_INDIRECT_WINDOW`, `VICTORY_INDIRECT_READABLE`, `VICTORY_CLAMP_GROUP`, `VICTORY_INDIRECT_CARD_BUDGET` and `VICTORY_CLAMP_LEAN` |
| `VICTORY_CLAMP_PRECISE=1` | an indirect argument clamp keeps only the waits it needs instead of a wait for every earlier shader write before and after its dispatch: none after (the move of its output to indirect arguments orders it for the execution, and the next clamp waits for the budget), and before it one for every earlier write only if something recorded since the list's last such wait may have written memory (a draw or dispatch whose shaders store, a command of the title's that may write, a buffer moved to be written), one for the budget alone if only clamps wrote, else none; what each record is clamped to and charged is the same (release builds too; off by default: a shader write its translation does not know of is then no longer waited for at every clamp; not yet run on an RX 9070 XT) |
| `VICTORY_CLAMP_BATCH=1` | the indirect executions of a list between two commands that may write their arguments share one argument clamp (one compute dispatch of ours, recorded before the first of them with the waits a clamp has, reading the list of executions from upload memory when the list runs), each into a slot of its own, instead of a clamp each: an execution joins while nothing recorded since the clamp may have written memory (a draw or dispatch whose shaders store, a command of the title's that may write, a buffer moved to be written), its argument and count buffers kept their state, they are the batch's, and it fits (128 executions, 256 KiB of records; the 256 KiB bind, and keep the batch's dispatch far below 65,535 groups); anything else opens the next batch, and an execution larger than a batch takes a clamp of its own. What each record is clamped to is the same; the budget is charged for the whole batch when its clamp runs, once per thread group as with `VICTORY_CLAMP_GROUP=1`, so which records of a list past its budget still fit follows the order the card runs the batch's groups in (release builds too; off by default: as with `VICTORY_CLAMP_PRECISE`, a shader write its translation does not know of, recorded after a batch's clamp, is no longer waited for before the later executions of the batch; not yet run on an RX 9070 XT). Works with or without `VICTORY_CLAMP_PRECISE`, `VICTORY_CLAMP_GROUP`, `VICTORY_INDIRECT_CARD_BUDGET` and `VICTORY_CLAMP_LEAN`; executions of a `VICTORY_INDIRECT_WINDOW` window and mesh draws a `VICTORY_GROUPED_MESH_CLAMP` run takes stay outside the batch |
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
