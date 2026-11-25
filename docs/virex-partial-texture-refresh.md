# Partial texture refresh correctness

Related to black or missing textures in #89 and #264. The link to
character skin artifacts is a hypothesis, not a verified fix.

A partial all-zero read is intentionally not detiled, but used to
replace the cached hash anyway. Keep the previous staged hash for
this rejected read. Otherwise a subsequent fully mapped all-zero
texture can incorrectly compare equal to bytes never staged.

Reset the partial polling timestamp on the ordinary fully mapped
read path, so a subsequent partial mapping starts a fresh poll.

## Review follow-up

The initial change allowed a guard-reported write to bypass the
30-frame partial read throttle. Codex correctly identified that
unguardable sparse view blocks are conservatively reported as written
on every query, even when unchanged. The bypass is removed entirely.
Partial reads retain the original 30-frame bound during continuous
partial mapping, including repeated reports within a frame. Actual
writes can still wait for the next poll, as on upstream main.

Two Rust regression tests cover repeated sparse polling and rejected
zero hashes. A JavaScript model exercised 15974400 queries across 64
initial frame offsets and 64 submission counts, with no mismatch
against the expected periodic read schedule. This is model validation,
not execution of the Rust tests.

Native Windows/Rust and image correctness tests remain pending.
The write-watch branch, 8-frame guard policy, GPU synchronization and
unlocked read path are unchanged. No FPS gain is claimed.
AI-assisted (ChatGPT).
