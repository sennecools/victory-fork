//! The upload ring (change 2, T3): one persistent buffer of 256 MiB in the
//! upload heap, mapped once at device start, from which the copy queue
//! (`super::copyq`) takes what it carries to the card.
//!
//! Allocations are whole granules of 64 KiB, the unit the design gives the
//! ring: far above the 512 bytes a placed footprint must be aligned to, so
//! every footprint offset in the ring is aligned without thought. They are
//! made without a lock: a compare and swap on the head. Each allocation is tagged
//! with the copy-fence value of the batch that reads it, and given back from
//! the tail once the copy fence has passed that value. Tags may arrive out of
//! order (workers post in any order); the tail only moves over a run of
//! allocations that are all tagged and passed, so a range is never reused
//! while an earlier one before it is still unread.
//!
//! When the ring is full, an allocation first gives back, without waiting,
//! what the copy fence's last completed value (as the uploader last read
//! it) allows; if that frees nothing, it returns `None` at once. Nothing
//! here waits for the card. An allocation that would cross the end takes
//! the rest of the ring as an allocation of its own, given back at once,
//! and starts again at the ring's beginning.
//!
//! Fence invariants:
//! - A granule is handed out again only after the tail moved past it, and
//!   the tail moves past an allocation only when its tag is `FREED` (never
//!   submitted) or the copy fence's completed value is at least its tag.
//! - The tail is moved by one thread at a time (`Core::retiring`), so a word
//!   is zeroed before the tail moves past it and no stale word is read.
//! - Allocation never reads the fence and never blocks: it retires only by
//!   the completed value the uploader stored, under a `try_lock`.
//! - A span holds a reference to the buffer, so its mapping lives as long
//!   as any span that may write it.

// Until T6 posts texture levels, only the tests post jobs and fill spans.
#![cfg_attr(not(test), allow(dead_code))]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// The ring's size: one buffer, charged to commit once at device start.
pub(crate) const RING_BYTES: u64 = 256 << 20;
/// The unit of allocation, and so the alignment of every offset.
pub(crate) const GRANULE: u64 = 64 << 10;

/// A granule word: the allocation's length in granules in the low bits,
/// its tag above them. 0 while free, or while its owner has not yet
/// published the length.
const LEN_BITS: u32 = 16;
const LEN_MASK: u64 = (1 << LEN_BITS) - 1;
/// The tag of an allocation dropped without being submitted: free at once.
const FREED: u64 = u64::MAX >> LEN_BITS;

/// The ring's bookkeeping, without the buffer (the model the tests check).
pub(crate) struct Core {
    granules: u64,
    /// Granules handed out since the start; only grows.
    head: AtomicU64,
    /// Granules given back since the start; only grows, never past `head`.
    tail: AtomicU64,
    /// One word per granule, set on the granule that starts an allocation.
    words: Box<[AtomicU64]>,
    /// Held while the tail moves: one retirer at a time.
    retiring: Mutex<()>,
    /// The copy fence's newest completed value given to [`Core::retire`]:
    /// what an allocation may retire by when the ring is full.
    completed: AtomicU64,
}

impl Core {
    /// A ring of `granules` granules.
    pub(crate) fn new(granules: u64) -> Core {
        assert!(
            granules > 0 && granules <= LEN_MASK,
            "a ring of {granules} granules"
        );
        Core {
            granules,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            words: (0..granules).map(|_| AtomicU64::new(0)).collect(),
            retiring: Mutex::new(()),
            completed: AtomicU64::new(0),
        }
    }

    /// [`Core::take`], and if the ring is full, once more after giving back
    /// what the last completed value allows (space freed by spans dropped
    /// since the uploader last retired). Never waits.
    fn alloc(&self, bytes: u64) -> Option<(u64, u64)> {
        self.take(bytes).or_else(|| {
            self.retire(self.completed.load(Ordering::Acquire));
            self.take(bytes)
        })
    }

    /// `bytes` bytes, rounded up to whole granules, as (first granule
    /// position, granules); its offset in bytes is the position's granule in
    /// the ring times [`GRANULE`]. `None` at once if they do not fit now.
    fn take(&self, bytes: u64) -> Option<(u64, u64)> {
        let n = bytes.div_ceil(GRANULE).max(1);
        if n > self.granules {
            return None;
        }
        loop {
            let head = self.head.load(Ordering::Relaxed);
            // Acquire: the retirer zeroed the words before it moved the tail.
            let tail = self.tail.load(Ordering::Acquire);
            let at = head % self.granules;
            if at + n > self.granules {
                // An allocation never wraps: the rest of the ring becomes an
                // allocation of its own, free at once, and the next try
                // starts at the ring's beginning.
                let pad = self.granules - at;
                if head + pad > tail + self.granules {
                    return None;
                }
                if self
                    .head
                    .compare_exchange_weak(head, head + pad, Ordering::AcqRel, Ordering::Relaxed)
                    .is_ok()
                {
                    self.words[at as usize].store((FREED << LEN_BITS) | pad, Ordering::Release);
                    self.retire(self.completed.load(Ordering::Acquire));
                }
                continue;
            }
            if head + n > tail + self.granules {
                return None;
            }
            if self
                .head
                .compare_exchange_weak(head, head + n, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                // Untagged until submitted: the tail stops here.
                self.words[at as usize].store(n, Ordering::Release);
                return Some((head, n));
            }
        }
    }

    /// Tags the allocation at `position` of `total` granules: the tail may
    /// pass it once the copy fence reaches `tag` (`FREED`: at once).
    fn tag(&self, position: u64, total: u64, tag: u64) {
        debug_assert!(tag > 0 && tag <= FREED);
        let word = (tag << LEN_BITS) | total;
        self.words[(position % self.granules) as usize].store(word, Ordering::Release);
    }

    /// Moves the tail over every allocation, oldest first, whose copy has
    /// passed (`completed`: the copy fence's completed value) or that was
    /// never submitted; stops at the first that is untagged or still due.
    /// Never waits: if another thread retires, this one returns.
    pub(crate) fn retire(&self, completed: u64) {
        self.completed.fetch_max(completed, Ordering::AcqRel);
        let Ok(_retiring) = self.retiring.try_lock() else {
            return;
        };
        // After the lock: the newest value any caller stored meanwhile.
        let completed = self.completed.load(Ordering::Acquire);
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            if tail == self.head.load(Ordering::Acquire) {
                return;
            }
            let slot = &self.words[(tail % self.granules) as usize];
            let word = slot.load(Ordering::Acquire);
            let (total, tag) = (word & LEN_MASK, word >> LEN_BITS);
            if total == 0 || tag == 0 || (tag != FREED && tag > completed) {
                return;
            }
            slot.store(0, Ordering::Relaxed);
            // Release: the zeroed word is seen before the granules are.
            self.tail.store(tail + total, Ordering::Release);
        }
    }

    /// Granules handed out and not yet given back.
    pub(crate) fn used(&self) -> u64 {
        // Tail first: head only grows and is never behind it, so this
        // cannot underflow while another thread retires.
        let tail = self.tail.load(Ordering::Acquire);
        self.head.load(Ordering::Acquire) - tail
    }
}

/// The ring on the device: the buffer, its mapping and its bookkeeping.
pub(crate) struct UploadRing {
    buffer: ID3D12Resource,
    data: *mut u8,
    core: Arc<Core>,
}

// SAFETY: the buffer is a free-threaded D3D12 object; `data` is its
// persistent mapping (never undone), and each byte of it is written only
// through the one `Span` that holds it.
unsafe impl Send for UploadRing {}
// SAFETY: as above.
unsafe impl Sync for UploadRing {}

impl UploadRing {
    /// The ring of [`RING_BYTES`] on `device`, mapped for writing.
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<UploadRing> {
        Self::with_bytes(device, RING_BYTES)
    }

    /// A ring of `bytes` (whole granules) on `device`.
    pub(crate) fn with_bytes(
        device: &ID3D12Device,
        bytes: u64,
    ) -> windows::core::Result<UploadRing> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer; the
        // mapping stays for the buffer's life (upload heaps may stay mapped).
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut buffer,
            )?;
            let buffer = buffer.ok_or_else(windows::core::Error::empty)?;
            buffer.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut data))?;
            super::label(&buffer.cast()?, "upload ring", bytes);
            Ok(UploadRing {
                buffer,
                data: data.cast(),
                core: Arc::new(Core::new(bytes / GRANULE)),
            })
        }
    }

    /// `bytes` bytes of the ring for one copy; `None` at once when the ring
    /// is full (the caller keeps its work for later).
    pub(crate) fn alloc(&self, bytes: u64) -> Option<Span> {
        let (position, total) = self.core.alloc(bytes)?;
        let offset = (position % self.core.granules) * GRANULE;
        Some(Span {
            core: self.core.clone(),
            buffer: Some(self.buffer.clone()),
            // SAFETY: inside the mapping (`take` keeps offset + bytes within
            // the ring).
            data: unsafe { self.data.add(offset as usize) },
            position,
            total,
            offset,
            len: bytes,
            tagged: false,
        })
    }

    /// Gives back what the copy fence has passed (`completed`, the value the
    /// uploader just read), and keeps it for allocations that find the ring
    /// full.
    pub(crate) fn retire(&self, completed: u64) {
        self.core.retire(completed);
    }

    /// The buffer copies read from.
    pub(crate) fn buffer(&self) -> &ID3D12Resource {
        &self.buffer
    }

    /// Bytes handed out and not yet given back.
    pub(crate) fn used_bytes(&self) -> u64 {
        self.core.used() * GRANULE
    }
}

/// One allocation of the ring: written by its owner, read by one copy
/// batch, given back after that batch's fence. Dropped without a tag (never
/// submitted), it is given back at once.
pub(crate) struct Span {
    core: Arc<Core>,
    /// The ring's buffer, held so that its mapping outlives the span
    /// (`None` only in the model tests, which have no buffer).
    buffer: Option<ID3D12Resource>,
    data: *mut u8,
    position: u64,
    total: u64,
    offset: u64,
    len: u64,
    tagged: bool,
}

// SAFETY: `data` points at bytes of the ring only this span may write; the
// buffer it holds is a free-threaded D3D12 object.
unsafe impl Send for Span {}

impl Span {
    /// Its offset in the ring's buffer (a multiple of [`GRANULE`]).
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// The bytes asked for.
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    /// Its bytes, to fill before the copy is posted.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        debug_assert!(self.buffer.is_some(), "a span of a ring with a buffer");
        // SAFETY: the mapping is live, since the span holds the buffer and
        // the mapping is never undone; the span holds these bytes of it alone
        // until it is given back, which needs its tag (set after the last
        // write, through `&mut self`).
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len as usize) }
    }

    /// The batch that reads it signals `value` on the copy fence: the span
    /// is given back once the fence passes it.
    pub(crate) fn retire_at(&mut self, value: u64) {
        assert!(!self.tagged, "a span is read by one batch");
        self.core.tag(self.position, self.total, value);
        self.tagged = true;
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if !self.tagged {
            self.core.tag(self.position, self.total, FREED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span of `core` as `UploadRing::alloc` makes it, without a buffer.
    fn span(core: &Arc<Core>, bytes: u64) -> Option<Span> {
        let (position, total) = core.alloc(bytes)?;
        let offset = (position % core.granules) * GRANULE;
        Some(Span {
            core: core.clone(),
            buffer: None,
            data: std::ptr::null_mut(),
            position,
            total,
            offset,
            len: bytes,
            tagged: false,
        })
    }

    #[test]
    fn allocations_are_whole_granules_in_order() {
        let core = Arc::new(Core::new(8));
        let a = span(&core, 1).unwrap();
        let b = span(&core, GRANULE + 1).unwrap();
        let c = span(&core, GRANULE).unwrap();
        assert_eq!(
            [a.offset(), b.offset(), c.offset()],
            [0, GRANULE, 3 * GRANULE]
        );
        assert_eq!(core.used(), 4);
        assert!(span(&core, 9 * GRANULE).is_none(), "larger than the ring");
    }

    #[test]
    fn full_returns_none_at_once_until_the_fence_passes() {
        let core = Arc::new(Core::new(4));
        let mut a = span(&core, 2 * GRANULE).unwrap();
        let mut b = span(&core, 2 * GRANULE).unwrap();
        a.retire_at(1);
        b.retire_at(2);
        let started = std::time::Instant::now();
        assert!(span(&core, 1).is_none(), "full");
        assert!(
            started.elapsed() < std::time::Duration::from_millis(50),
            "no wait"
        );
        core.retire(0);
        assert!(span(&core, 1).is_none(), "the fence has not moved");
        core.retire(1);
        assert_eq!(core.used(), 2);
        let c = span(&core, 2 * GRANULE).unwrap();
        assert_eq!(c.offset(), 0, "wrapped to the start");
        assert!(span(&core, 1).is_none());
        core.retire(2);
        assert_eq!(core.used(), 2, "only b went; c is untagged");
    }

    #[test]
    fn an_allocation_that_would_wrap_pads_to_the_start() {
        let core = Arc::new(Core::new(4));
        let mut a = span(&core, 3 * GRANULE).unwrap();
        a.retire_at(1);
        core.retire(1);
        // Head at granule 3: two granules do not fit before the end.
        let b = span(&core, 2 * GRANULE).unwrap();
        assert_eq!(b.offset(), 0);
        assert_eq!(core.used(), 2, "the padding was given back at once");
        drop(b);
        core.retire(0);
        assert_eq!(core.used(), 0);
    }

    /// On an empty ring whose head is not at granule 0, an allocation of
    /// more than half the ring still fits: the end is taken as padding of
    /// its own and given back, and the allocation starts at the beginning.
    #[test]
    fn an_empty_ring_at_a_later_head_fits_more_than_half_of_it() {
        let core = Arc::new(Core::new(8));
        let mut a = span(&core, 3 * GRANULE).unwrap();
        a.retire_at(1);
        core.retire(1);
        assert_eq!(core.used(), 0, "empty, head at granule 3");
        let b = span(&core, 6 * GRANULE).expect("six of eight granules on an empty ring");
        assert_eq!(b.offset(), 0);
        assert_eq!(core.used(), 6);
    }

    /// The lost wake-up: the oldest span is untagged and holds the tail
    /// while the rest are filled, tagged and passed; then it is dropped and
    /// nothing retires (the uploader is idle). An allocation still succeeds:
    /// it retires by the last completed value itself.
    #[test]
    fn a_full_ring_freed_by_a_dropped_span_allocates_with_no_retire() {
        let core = Arc::new(Core::new(4));
        let oldest = span(&core, GRANULE).unwrap();
        let mut rest: Vec<_> = (0..3).map(|_| span(&core, GRANULE).unwrap()).collect();
        for (tag, s) in (1..).zip(&mut rest) {
            s.retire_at(tag);
        }
        core.retire(3);
        assert_eq!(core.used(), 4, "the untagged oldest holds the tail");
        assert!(span(&core, GRANULE).is_none(), "full");
        drop(oldest);
        drop(rest);
        let again = span(&core, GRANULE).expect("room once the oldest was dropped");
        assert_eq!(again.offset(), 0);
        assert_eq!(core.used(), 1);
    }

    #[test]
    fn the_tail_waits_for_an_earlier_untagged_span() {
        let core = Arc::new(Core::new(4));
        let a = span(&core, GRANULE).unwrap();
        let mut b = span(&core, GRANULE).unwrap();
        b.retire_at(1);
        core.retire(10);
        assert_eq!(
            core.used(),
            2,
            "b is passed but a, before it, is still being written"
        );
        drop(a);
        core.retire(10);
        assert_eq!(core.used(), 0);
    }

    #[test]
    fn out_of_order_tags_retire_in_allocation_order() {
        let core = Arc::new(Core::new(4));
        let mut a = span(&core, GRANULE).unwrap();
        let mut b = span(&core, GRANULE).unwrap();
        // b's batch went first.
        b.retire_at(1);
        a.retire_at(2);
        core.retire(1);
        assert_eq!(core.used(), 2, "a (tag 2) holds the tail, and b behind it");
        core.retire(2);
        assert_eq!(core.used(), 0);
    }

    /// Many threads allocate, tag and drop while one retires with a
    /// completed value it raises: no granule is ever handed out while an
    /// earlier allocation over it is untagged or tagged past what was
    /// retired.
    #[test]
    fn concurrent_allocations_never_overlap_a_live_range() {
        const GRANULES: u64 = 64;
        let core = Arc::new(Core::new(GRANULES));
        // (first granule, granules, tag: 0 while untagged).
        let live = Arc::new(Mutex::new(Vec::<(u64, u64, u64)>::new()));
        let retired_up_to = Arc::new(AtomicU64::new(0));
        let next_tag = Arc::new(AtomicU64::new(1));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let retirer = {
            let (core, retired_up_to, next_tag, stop) = (
                core.clone(),
                retired_up_to.clone(),
                next_tag.clone(),
                stop.clone(),
            );
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    // The fence "completes" up to a little behind the tags given.
                    let completed = next_tag.load(Ordering::Relaxed).saturating_sub(3);
                    retired_up_to.fetch_max(completed, Ordering::SeqCst);
                    core.retire(completed);
                    std::thread::yield_now();
                }
            })
        };
        let workers: Vec<_> = (0..6u64)
            .map(|w| {
                let (core, live, retired_up_to, next_tag) = (
                    core.clone(),
                    live.clone(),
                    retired_up_to.clone(),
                    next_tag.clone(),
                );
                std::thread::spawn(move || {
                    let mut seed = w * 7919 + 1;
                    for _ in 0..4000 {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let bytes = (seed >> 33) % (5 * GRANULE) + 1;
                        let Some(mut s) = span(&core, bytes) else {
                            std::thread::yield_now();
                            continue;
                        };
                        // The granules this allocation holds, padding included.
                        let at = s.position % GRANULES;
                        let mine = (at, s.total);
                        let covers =
                            |(lo, n): (u64, u64), g: u64| (lo..lo + n).any(|x| x % GRANULES == g);
                        {
                            let mut live = live.lock().unwrap();
                            let up_to = retired_up_to.load(Ordering::SeqCst);
                            live.retain(|&(_, _, tag)| tag == 0 || tag > up_to);
                            for &(lo, n, _) in live.iter() {
                                for g in mine.0..mine.0 + mine.1 {
                                    assert!(
                                        !covers((lo, n), g % GRANULES),
                                        "granule {} handed out twice",
                                        g % GRANULES
                                    );
                                }
                            }
                            live.push((mine.0, mine.1, 0));
                        }
                        if seed & 3 == 0 {
                            // Never submitted.
                            live.lock().unwrap().retain(|&(lo, n, _)| (lo, n) != mine);
                            drop(s);
                        } else {
                            let tag = next_tag.fetch_add(1, Ordering::Relaxed);
                            {
                                let mut live = live.lock().unwrap();
                                for e in live.iter_mut() {
                                    if (e.0, e.1) == mine && e.2 == 0 {
                                        e.2 = tag;
                                    }
                                }
                            }
                            s.retire_at(tag);
                        }
                    }
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        retirer.join().unwrap();
        core.retire(u64::MAX >> LEN_BITS);
        assert_eq!(
            core.used(),
            0,
            "everything goes back once the fence passed it all"
        );
    }

    /// The commit of this process, in bytes.
    fn commit() -> u64 {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let mut c = PROCESS_MEMORY_COUNTERS {
            cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        // SAFETY: a plain query into a struct of the size given.
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        c.PagefileUsage as u64
    }

    /// The ring on the software adapter: charged to commit once when made
    /// (a lower bound: the host made first, so that its own start is not
    /// counted), and filled and given back four times it hands out the same
    /// offsets of the same buffer each time, and nothing stays used.
    #[test]
    fn the_ring_is_charged_to_commit_once_and_never_grows() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        let _ = crate::host::host();
        let Ok(adapter) = crate::host::warp_adapter() else {
            return; // no software adapter
        };
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain device creation.
        if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }.is_err() {
            return;
        }
        let device = device.unwrap();
        let before = commit();
        let ring = UploadRing::new(&device).expect("the ring");
        let made = commit();
        assert!(
            made >= before + RING_BYTES * 9 / 10,
            "the ring's {} MiB are charged when it is made ({} MiB more)",
            RING_BYTES >> 20,
            made.saturating_sub(before) >> 20
        );
        // SAFETY: a plain query.
        let width = || unsafe { ring.buffer().GetDesc() }.Width;
        let mut tag = 0;
        let mut first: Option<Vec<u64>> = None;
        for round in 0..4 {
            let mut spans = Vec::new();
            while let Some(mut s) = ring.alloc(4 * GRANULE - round) {
                s.bytes_mut().fill(round as u8);
                spans.push(s);
            }
            assert_eq!(
                spans.len() as u64,
                RING_BYTES / (4 * GRANULE),
                "filled whole"
            );
            let offsets: Vec<u64> = spans.iter().map(Span::offset).collect();
            assert_eq!(
                first.get_or_insert_with(|| offsets.clone()),
                &offsets,
                "round {round}"
            );
            for s in &mut spans {
                tag += 1;
                s.retire_at(tag);
            }
            drop(spans);
            ring.retire(tag);
            assert_eq!(ring.used_bytes(), 0);
            assert_eq!(width(), RING_BYTES, "the same buffer");
        }
    }
}
