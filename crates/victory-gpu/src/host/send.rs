//! What goes up to copies in the card's memory goes up as it was when its
//! submission was prepared ([`own`]).
//!
//! The sends were copies out of the copies' CPU sides, which the card makes
//! when it comes to the submission. With the relay's second stage later
//! submissions are prepared by then, and their refreshes, and the merges of
//! the title's bytes under flights' shader writes
//! (`Shadow::merge_under_writes`), have written the same CPU sides: the
//! earlier submission's shaders would read a later one's bytes. A title
//! buffer copy is a command in the title's lists, done by us when its
//! submission is prepared, so "a value to a buffer, a draw that reads it" in
//! one submission and the same with another value in the next is ordinary,
//! and both draws read the second value.
//!
//! So each submission's bytes are taken out of the CPU sides into memory of
//! the submission's own when they are taken from the dirty lists, and the
//! card copies them from there, on the same queue as before (in queue order
//! behind the flights). That memory is used again only once the submission
//! has run (`Host::card_done`).
//!
//! Adapted from cool_fox 5b422ba and 18c0129 (author 'local'), with the
//! reuse pool of 52e0c03, via Belobug/VIctory-Belobug-AMD cd8dfc06,
//! GPL-3.0-or-later. Left out: the pairs of offset and word stored by a
//! dispatch (a later step), the up queue (53ed64e: a copy queue would need a
//! fence between it, the flights' writes and the submission's first read)
//! and the memory in the card's own heap (`VICTORY_SEND_BAR`).

use super::*;
use crate::shadow::Shadow;
use std::sync::atomic::AtomicU64;

/// Whether a submission's sends read memory of its own: with the relay's
/// second stage ([`relay_exact`]), under which submissions are prepared
/// while earlier ones wait on the queue and the merges under flights' writes
/// reach the CPU sides.
pub(super) fn own() -> bool {
    relay_exact()
}

/// Memory a submission's sends come from, mapped (upload memory).
pub(super) struct Memory {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

// SAFETY: a D3D12 object (free-threaded) and its mapping, with one owner.
unsafe impl Send for Memory {}

/// What submissions that have run gave back.
static SPARE: Mutex<Vec<Memory>> = Mutex::new(Vec::new());

/// At most this much is kept in [`SPARE`].
const KEPT: u64 = 256 << 20;

/// Memory of at least `size` bytes: the smallest spare one that fits, or a
/// new one.
fn memory(host: &Host, size: u64) -> Option<Memory> {
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        let fit = spare
            .iter()
            .enumerate()
            .filter(|(_, m)| m.size >= size)
            .min_by_key(|(_, m)| m.size)
            .map(|(i, _)| i);
        if let Some(i) = fit {
            return Some(spare.swap_remove(i));
        }
    }
    // By doubles, so that the next of about this size finds it: what
    // submissions send differs from frame to frame.
    let size = size.max(1 << 20).next_power_of_two();
    let (buffer, data) = host.upload_buffer(size)?;
    label(&buffer, "submission send memory", size);
    name(&buffer, || "bytes a submission sends up".into());
    Some(Memory {
        buffer,
        data: data as usize,
        size,
    })
}

/// The submission that `memory`'s sends belonged to has run.
pub(super) fn give_back(memory: Memory) {
    #[cfg(test)]
    OUT.fetch_sub(1, Ordering::Relaxed);
    let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
    spare.push(memory);
    spare.sort_by_key(|m| std::cmp::Reverse(m.size));
    while spare.iter().map(|m| m.size).sum::<u64>() > KEPT {
        spare.remove(0);
    }
}

/// Since the last frame-rate line: bytes taken, runs taken, microseconds of
/// taking them out of the CPU sides.
pub(super) static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Where a submission's bytes are taken to.
pub(super) struct Own {
    memory: Memory,
    /// Per entry of the dirty list it was taken from: the runs (offset in
    /// the copy, in the memory, bytes), or `None` for a copy whose CPU side
    /// is a home (`crate::card_home`), which goes up from the home as before.
    runs: Vec<Option<Vec<(usize, u64, usize)>>>,
}

/// Takes what `dirty` (`crate::shadow::card_dirty`) sends up out of the CPU
/// sides, as they are now. `None` if there is nothing to take, or no memory
/// for it: the sends then read the CPU sides, as before.
///
/// shortcut: the copies are made on this thread alone; the fork spreads big
/// ones over several (`copy_all`).
pub(super) fn take(
    host: &Host,
    dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
) -> Option<Own> {
    let began = std::time::Instant::now();
    let mut at = 0u64;
    let mut runs = Vec::with_capacity(dirty.len());
    for (shadow, ranges) in dirty {
        if shadow.card_homed().is_some() {
            runs.push(None);
            continue;
        }
        let mut mine = Vec::with_capacity(ranges.len());
        for &(a, b) in ranges {
            mine.push((a, at, b - a));
            at += ((b - a) as u64).next_multiple_of(16);
        }
        runs.push(Some(mine));
    }
    if at == 0 {
        return None;
    }
    let memory = memory(host, at)?;
    #[cfg(test)]
    {
        OUT.fetch_add(1, Ordering::Relaxed);
        LAST.store(memory.data, Ordering::Relaxed);
    }
    let mut bytes = 0;
    for ((shadow, _), runs) in dirty.iter().zip(&runs) {
        if let Some(runs) = runs {
            // SAFETY: the memory has room for every run behind the one
            // before it (`at` above), and it is ours until the submission
            // has run.
            unsafe { shadow.card_take(runs, memory.data as *mut u8) };
            bytes += runs.iter().map(|r| r.2 as u64).sum::<u64>();
        }
    }
    STATS[0].fetch_add(bytes, Ordering::Relaxed);
    STATS[1].fetch_add(
        runs.iter().flatten().map(|r| r.len() as u64).sum(),
        Ordering::Relaxed,
    );
    STATS[2].fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
    Some(Own { memory, runs })
}

impl Own {
    /// Records into `list` the sends of entry `i` of the dirty list it was
    /// taken from, `shadow`'s, from the memory to its card side; the bytes,
    /// or `None` if that copy's sends read its CPU side (`Shadow::card_upload`).
    ///
    /// # Safety
    ///
    /// `list` is open; `shadow` is entry `i`'s copy, and alive.
    pub(super) unsafe fn upload(
        &self,
        list: &ID3D12GraphicsCommandList,
        i: usize,
        shadow: &Shadow,
    ) -> Option<u64> {
        let runs = self.runs.get(i)?.as_ref()?;
        for &(a, from, n) in runs {
            // SAFETY: the caller's; live buffers, the ranges inside them.
            unsafe {
                list.CopyBufferRegion(
                    &shadow.buffer,
                    a as u64,
                    &self.memory.buffer,
                    from,
                    n as u64,
                )
            };
        }
        Some(runs.iter().map(|r| r.2 as u64).sum())
    }

    /// The memory, to be kept until the submission has run.
    pub(super) fn memory(self) -> Memory {
        self.memory
    }
}

/// The frame-rate line's part ([`STATS`]), per frame over `frames`.
pub(super) fn line(frames: f64) {
    if !own() {
        return;
    }
    let t: [f64; 3] = std::array::from_fn(|i| STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
    tracing::info!(
        "bytes of submissions' own sent up per frame (relay stage 2): {:.2} MB in {:.0} runs, {:.2} ms taking them out of \
         the CPU sides",
        t[0] / f64::from(1 << 20),
        t[1],
        t[2] / 1000.0,
    );
}

/// Memory taken by submissions and not given back yet (tests).
#[cfg(test)]
static OUT: AtomicU64 = AtomicU64::new(0);

/// The mapping of the memory taken last (tests).
#[cfg(test)]
static LAST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// [`OUT`] and [`LAST`] (tests).
#[cfg(test)]
pub(crate) fn out_for_test() -> (u64, usize) {
    (OUT.load(Ordering::Relaxed), LAST.load(Ordering::Relaxed))
}

/// Whether the spare memory holds the memory mapped at `data` (tests).
#[cfg(test)]
pub(crate) fn spare_for_test(data: usize) -> bool {
    SPARE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .any(|m| m.data == data)
}
