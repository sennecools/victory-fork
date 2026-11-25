# Shader cache resilience

The shader cache is optional, but its publication failures previously
propagated as shader translation failures. Cache::get_or_convert performs
conversion before writing the entry, so the translated HostShader is
already available when the cache directory or rename fails.

Retain the original translation result for this call. A successful
translation is returned directly (without decoding its own fresh entry);
a translation failure is returned without retrying. Existing cache hits
still decode from disk, and damaged entries still get removed and retried.
No cache format, key, graphics settings or shader semantics change.

Two Rust regressions cover a blocked publication and a translation error,
both checking that translation runs once. Native Windows/Rust tests are
pending. No in-game performance or visual claim. AI-assisted (ChatGPT).
