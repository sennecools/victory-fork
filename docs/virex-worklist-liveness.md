# Worklist register liveness

Port the solver from Marvisak/VIctory#1846 onto the current fork.
The original liveness function is identical at this base. Instead of
revisiting all reachable blocks in every sweep, revisit only predecessors
when a block's live-in set changes. Preserve live-out updates and zero
sets for unreachable blocks. No changes to SSA construction, shader
semantics, register representation or GPU commands.

The original upstream contribution was independently tested on Windows
with an RTX 3070, all 96 new CFG tests passed, and the three failures
were reported as pre-existing baseline failures. That is not validation
of this fork.

This port includes two tests: 4096 generated CFGs using the actual
register bitset and a 2048-block cyclic chain, compared with the old
full-sweep solver. An independent JavaScript model compared 4096
additional CFGs with both the sweep and a path-based oracle:
101548 blocks, 44607 sweep visits vs 20745 worklist visits.
No disagreements. These are operation counts, not runtime measurements.
Native Rust tests and shader translation timing remain pending.

    cargo test -p victory-shader ir::liveness_tests --locked

No game assets or dependencies. AI-assisted (ChatGPT).
