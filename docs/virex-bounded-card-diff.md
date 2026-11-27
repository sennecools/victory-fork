# Bound fragmented card-write diff work

In late-exact submission preparation, the card-backed path budgets a
maximum of 4,096 changed runs across the submission (512 before every
stale span was merged). Previously a page
was fully scanned and all runs allocated even when the result exceeded
the remaining budget and would be discarded. Stop on the first excess
run instead, before allocating or scanning that run's remainder.

The accepted run list is unchanged. If the budget is exceeded, no
partial result is returned or applied, and the caller settles flights
exactly as before. Preserve unequal-length common-prefix behavior,
zero budget, and the unbounded CPU-side path.

A synthetic 64 KiB alternating-byte page contains 32,768 runs.
With budget 512 the new algorithm retains at most 512 runs and detects
the excess at byte offset 1,024 rather than scanning the whole page.
This does not reduce actual uploaded bytes or change the GPU wait
decision. It reduces CPU work and temporary allocation on rejected
fragmented pages while holding the reference lock. Real workload
frequency and frame-time benefit are unmeasured.

Python model: 37,768 exhaustive and seeded cases versus an independent
group-by reference. JavaScript model: 10,000 seeded cases versus a
per-byte run accumulator. Zero disagreements. These are algorithm
models, not native Rust tests.

    cargo test -p victory-gpu shadow::merge_tests --locked
    cargo test -p victory-gpu shadow::relay_tests --locked

Native Windows/Rust testing pending. No new dependencies or game data.
AI-assisted (ChatGPT).
