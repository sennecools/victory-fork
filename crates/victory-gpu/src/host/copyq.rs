//! The copy queue (change 2, T3): one D3D12 copy queue and the uploader
//! thread that feeds it. Workers post jobs (bytes in the upload ring,
//! `super::ring`, and where they go) on a channel; the uploader cuts them
//! into batches every millisecond or 32 MiB, runs each batch on the copy
//! queue and signals the copy fence. Nothing is wired to textures yet
//! (T6 does that).
//!
//! The direct queue's side is [`execute`], the one way lists of ours reach
//! the host's direct queue (`Host::run_lists`): before them it makes the
//! direct queue wait on the GPU for every copy batch signalled so far, and
//! skips that wait when no batch was signalled since the last one.
//!
//! Fence invariants (each is tested below):
//! 1. Reads after copies: lists sent through [`execute`] run after every
//!    copy batch whose signal was issued before the call began.
//! 2. Copies after reads: a job names `named_by`, the direct-fence value of
//!    the newest submission that may name its destination (0: none). Its
//!    batch makes the copy queue wait for that value first, so a copy never
//!    overtakes direct work in flight that samples or moves its destination.
//! 3. No cycle, so no GPU hang: each queue waits only for a value whose
//!    signal was already issued on the other queue. The direct queue waits
//!    for `signalled`, which is stored after the copy queue's Signal
//!    returned; a batch waits only for a value at most `direct_issued`,
//!    which `Host::signal_submitted` stores after the direct queue's Signal
//!    returned. A job naming a later value is held until it is issued. Every
//!    wait thus points at an earlier issued signal, and waits cannot form a
//!    loop.
//! 4. A ring range is given back only once the copy fence passed the batch
//!    that read it (`super::ring`), and a batch's destinations are held
//!    until then too: a host texture the title lets go of lives until both
//!    the direct fence (the sweeper's contract, `super::sweeper`) and the
//!    copy fence passed every use.
//! 5. No thread here blocks on a fence while running: the uploader polls
//!    the copy fence and retires what passed. The one wait is at the end,
//!    when the queue is dropped, bounded, for the batches still in flight.
//!
//! TODO(T6): `Host::drain` (`super::ending`, the release at the title's end)
//! waits only for the direct queue. Once T6 posts real copies, it must also
//! wait, under the same deadline, for the copy fence to reach the last value
//! signalled ([`CopyQueue::signalled`]) before anything is let go.
//! Also, `Uploader::batch` lets a later ready job overtake an earlier held
//! one (its naming submission not yet issued), so T6 must order jobs to the
//! same destination itself.

// Until T6 posts texture levels, only the tests post jobs and fill spans.
#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::Interface;

use super::ring::{Span, UploadRing};

/// A batch is cut once its jobs carry this many bytes...
pub(crate) const BATCH_BYTES: u64 = 32 << 20;
/// ...or once its oldest job waited this long.
pub(crate) const BATCH_AGE: Duration = Duration::from_millis(1);
/// Batches on the copy queue at once; past that, jobs wait for one to pass.
const IN_FLIGHT: usize = 8;

/// Where a job's bytes go.
pub(crate) enum Dest {
    /// A subresource of a texture, from `footprint` (its offset counted
    /// from the start of the job's span).
    Texture {
        resource: ID3D12Resource,
        subresource: u32,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    },
    /// A buffer at `offset`, the span's length.
    Buffer {
        resource: ID3D12Resource,
        offset: u64,
    },
}

/// One copy: the bytes in the ring, where they go, and the newest
/// submission that may name the destination.
pub(crate) struct Job {
    span: Span,
    dest: Dest,
    /// The direct-fence value of the newest submission that may name the
    /// destination (0: none ever did). The copy waits for it.
    named_by: u64,
    /// Submissions started when the job was posted (`super::STARTED`): the
    /// job is to be seen by every submission after that.
    posted_at: u64,
}

// SAFETY: D3D12 resources are free-threaded; the span's bytes are the
// job's alone.
unsafe impl Send for Job {}

impl Job {
    pub(crate) fn new(span: Span, dest: Dest, named_by: u64) -> Job {
        Job {
            span,
            dest,
            named_by,
            posted_at: 0,
        }
    }
}

/// What the uploader and the queue's users share.
struct Shared {
    /// The newest value signalled on the copy fence (its Signal returned).
    signalled: AtomicU64,
    /// The newest value signalled on the direct fence (its Signal returned).
    direct_issued: AtomicU64,
    /// Jobs posted and not yet in a submitted batch, by `posted_at`.
    unsubmitted: Mutex<BTreeMap<u64, u64>>,
    /// How many jobs that is, read without the lock.
    pending: AtomicU64,
    /// Waits the direct queue was given for copies (counted for the tests
    /// and, later, the frame summary).
    direct_waits: AtomicU64,
    /// Jobs dropped because their batch could not be recorded or sent.
    failed: AtomicU64,
}

/// The copy queue, its fence and the uploader thread.
pub(crate) struct CopyQueue {
    fence: ID3D12Fence,
    ring: Arc<UploadRing>,
    shared: Arc<Shared>,
    /// The newest copy-fence value the direct queue was told to wait for.
    waited: AtomicU64,
    /// Held while a wait for copies goes to the direct queue.
    gate: Mutex<()>,
    sender: Option<Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

// SAFETY: D3D12 queues and fences are free-threaded; the rest is atomics,
// locks and a channel's sending side.
unsafe impl Send for CopyQueue {}
// SAFETY: as above.
unsafe impl Sync for CopyQueue {}

impl CopyQueue {
    /// A copy queue on `device` whose batches wait on `direct_fence` (the
    /// direct queue's), carrying bytes from `ring`.
    pub(crate) fn new(
        device: &ID3D12Device,
        direct_fence: &ID3D12Fence,
        ring: Arc<UploadRing>,
    ) -> windows::core::Result<CopyQueue> {
        // SAFETY: plain object creation.
        let (queue, fence) = unsafe {
            let queue: ID3D12CommandQueue =
                device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    ..Default::default()
                })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            (queue, fence)
        };
        if let Ok(object) = queue.cast() {
            super::label(&object, "copy queue", 0);
        }
        let shared = Arc::new(Shared {
            signalled: AtomicU64::new(0),
            direct_issued: AtomicU64::new(0),
            unsubmitted: Mutex::new(BTreeMap::new()),
            pending: AtomicU64::new(0),
            direct_waits: AtomicU64::new(0),
            failed: AtomicU64::new(0),
        });
        let (sender, receiver) = std::sync::mpsc::channel();
        let uploader = Uploader {
            device: device.clone(),
            queue,
            fence: fence.clone(),
            direct_fence: direct_fence.clone(),
            ring: ring.clone(),
            shared: shared.clone(),
            value: 0,
            flying: VecDeque::new(),
            free: Vec::new(),
        };
        let thread = std::thread::Builder::new()
            .name("copy uploader".into())
            .spawn(move || uploader.run(receiver))
            .map_err(|_| windows::core::Error::empty())?;
        Ok(CopyQueue {
            fence,
            ring,
            shared,
            waited: AtomicU64::new(0),
            gate: Mutex::new(()),
            sender: Some(sender),
            thread: Some(thread),
        })
    }

    /// The ring jobs take their bytes from.
    pub(crate) fn ring(&self) -> &UploadRing {
        &self.ring
    }

    /// Hands `job` to the uploader, stamped with the submissions started
    /// now; back (boxed, as it is large) if the uploader is gone. Never
    /// waits.
    pub(crate) fn post(&self, mut job: Job) -> Result<(), Box<Job>> {
        job.posted_at = super::STARTED.load(Ordering::Acquire);
        let Some(sender) = &self.sender else {
            return Err(Box::new(job));
        };
        self.note_posted(job.posted_at);
        sender.send(job).map_err(|e| {
            let job = e.0;
            self.shared.submitted(job.posted_at);
            Box::new(job)
        })
    }

    fn note_posted(&self, posted_at: u64) {
        let mut unsubmitted = self
            .shared
            .unsubmitted
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *unsubmitted.entry(posted_at).or_insert(0) += 1;
        self.shared.pending.fetch_add(1, Ordering::Release);
    }

    /// The direct queue signalled `value` on its fence (the Signal
    /// returned): batches may now wait for it (invariant 3).
    pub(crate) fn direct_signalled(&self, value: u64) {
        self.shared
            .direct_issued
            .fetch_max(value, Ordering::Release);
    }

    /// The newest direct-fence value issued: what a job may name.
    pub(crate) fn direct_issued(&self) -> u64 {
        self.shared.direct_issued.load(Ordering::Acquire)
    }

    /// The newest value signalled on the copy fence.
    pub(crate) fn signalled(&self) -> u64 {
        self.shared.signalled.load(Ordering::Acquire)
    }

    /// Jobs posted while fewer than `cut` submissions had started that are
    /// not yet in a submitted batch (T6: before submission `cut` they must
    /// be, or are counted as late).
    pub(crate) fn unsubmitted_before(&self, cut: u64) -> u64 {
        if self.shared.pending.load(Ordering::Acquire) == 0 {
            return 0;
        }
        let unsubmitted = self
            .shared
            .unsubmitted
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unsubmitted.range(..cut).map(|(_, n)| n).sum()
    }

    /// Waits the direct queue was given for copies.
    pub(crate) fn direct_waits(&self) -> u64 {
        self.shared.direct_waits.load(Ordering::Relaxed)
    }

    /// Jobs dropped because their batch could not be sent.
    pub(crate) fn failed(&self) -> u64 {
        self.shared.failed.load(Ordering::Relaxed)
    }

    /// Makes `direct` wait on the GPU for every copy batch signalled before
    /// this call (invariant 1). No wait when none was signalled since the
    /// last: two atomic loads.
    fn order_after_copies(&self, direct: &ID3D12CommandQueue) {
        let signalled = self.shared.signalled.load(Ordering::Acquire);
        // Acquire: the wait for `waited` went to the queue before it was
        // stored, so lists sent after this load run behind it.
        if signalled <= self.waited.load(Ordering::Acquire) {
            return;
        }
        let _gate = self.gate.lock().unwrap_or_else(|e| e.into_inner());
        // The newest under the lock: one wait covers every batch so far.
        let signalled = self.shared.signalled.load(Ordering::Acquire);
        if signalled <= self.waited.load(Ordering::Relaxed) {
            return;
        }
        // SAFETY: our fence, at a value whose signal was issued.
        match unsafe { direct.Wait(&self.fence, signalled) } {
            Ok(()) => {
                self.shared.direct_waits.fetch_add(1, Ordering::Relaxed);
                self.waited.store(signalled, Ordering::Release);
            }
            Err(e) => {
                tracing::warn!(error = %e, "the direct queue could not wait for the copy queue")
            }
        }
    }
}

impl Drop for CopyQueue {
    fn drop(&mut self) {
        // The uploader sends what is ready, waits for its batches and ends.
        drop(self.sender.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Runs closed `lists` on `direct`, the host's direct queue, after every
/// copy batch of `copy` signalled so far: the one way lists of ours go to
/// that queue (`Host::run_lists`; a test checks that no bare call is left).
///
/// # Safety
/// `lists` are closed and may run on `direct`; what they use lives until
/// they ran.
pub(crate) unsafe fn execute(
    direct: &ID3D12CommandQueue,
    copy: Option<&CopyQueue>,
    lists: &[Option<ID3D12CommandList>],
) {
    if let Some(copy) = copy {
        copy.order_after_copies(direct);
    }
    // SAFETY: as the caller promised.
    unsafe { direct.ExecuteCommandLists(lists) };
}

impl Shared {
    /// A job posted at `posted_at` is in a submitted batch (or gone).
    fn submitted(&self, posted_at: u64) {
        let mut unsubmitted = self.unsubmitted.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = unsubmitted.get_mut(&posted_at) {
            *n -= 1;
            if *n == 0 {
                unsubmitted.remove(&posted_at);
            }
        }
        self.pending.fetch_sub(1, Ordering::Release);
    }
}

/// A batch on the copy queue: its fence value, its list and the jobs it
/// holds (their destinations live until it passed).
struct Flying {
    value: u64,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    _jobs: Vec<Job>,
}

/// The uploader thread's own state.
struct Uploader {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    direct_fence: ID3D12Fence,
    ring: Arc<UploadRing>,
    shared: Arc<Shared>,
    /// The last value signalled on the copy fence.
    value: u64,
    flying: VecDeque<Flying>,
    /// Allocators and lists whose batches passed, closed.
    free: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>,
}

// SAFETY: D3D12 objects are free-threaded; the uploader is moved to its
// thread once.
unsafe impl Send for Uploader {}

impl Uploader {
    fn run(mut self, receiver: Receiver<Job>) {
        let mut waiting: VecDeque<Job> = VecDeque::new();
        // When the oldest job in `waiting` arrived.
        let mut since: Option<Instant> = None;
        let mut open = true;
        while open || !waiting.is_empty() {
            let got = if !open {
                // Closing: what is left waits for a batch to pass.
                std::thread::sleep(BATCH_AGE);
                Err(RecvTimeoutError::Disconnected)
            } else if waiting.is_empty() && self.flying.is_empty() {
                // Nothing to poll for: sleep until a job comes.
                receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
            } else {
                receiver.recv_timeout(BATCH_AGE)
            };
            match got {
                Ok(job) => {
                    waiting.push_back(job);
                    waiting.extend(receiver.try_iter());
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => open = false,
            }
            if !waiting.is_empty() {
                since.get_or_insert_with(Instant::now);
            }
            self.retire();
            let issued = self.shared.direct_issued.load(Ordering::Acquire);
            let ready: u64 = waiting
                .iter()
                .filter(|j| j.named_by <= issued)
                .map(|j| j.span.len())
                .sum();
            let due = since.is_some_and(|s| s.elapsed() >= BATCH_AGE);
            if ready > 0 && (ready >= BATCH_BYTES || due || !open) {
                self.batch(&mut waiting, issued);
                since = (!waiting.is_empty()).then(Instant::now);
            }
            if !open && !waiting.is_empty() && ready == 0 {
                // Closing, and the rest name submissions never signalled:
                // they are dropped (their ring spans go back untouched).
                tracing::warn!(
                    jobs = waiting.len(),
                    "copies dropped at the end: their submissions never came"
                );
                for job in waiting.drain(..) {
                    self.shared.submitted(job.posted_at);
                }
            }
        }
        self.finish();
    }

    /// Gives back the batches, allocators and ring ranges the copy fence
    /// passed. Never waits.
    fn retire(&mut self) {
        // SAFETY: a plain query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        while self.flying.front().is_some_and(|f| f.value <= completed) {
            let done = self.flying.pop_front().expect("a batch");
            // SAFETY: its batch has run.
            if unsafe { done.allocator.Reset() }.is_ok() {
                self.free.push((done.allocator, done.list));
            }
        }
        self.ring.retire(completed);
    }

    /// Sends one batch of the jobs whose naming submissions were issued
    /// (`issued`): up to [`BATCH_BYTES`], oldest first. With [`IN_FLIGHT`]
    /// batches on the queue, it sends nothing until one passed.
    fn batch(&mut self, waiting: &mut VecDeque<Job>, issued: u64) {
        if self.flying.len() >= IN_FLIGHT {
            return;
        }
        let mut jobs = Vec::new();
        let mut bytes = 0;
        let mut held = VecDeque::new();
        while let Some(job) = waiting.pop_front() {
            if job.named_by > issued || (bytes >= BATCH_BYTES && !jobs.is_empty()) {
                held.push_back(job);
            } else {
                bytes += job.span.len();
                jobs.push(job);
            }
        }
        *waiting = held;
        if jobs.is_empty() {
            return;
        }
        match self.send(&mut jobs) {
            Ok(flying) => self.flying.push_back(flying),
            Err(e) => {
                tracing::warn!(error = %e, jobs = jobs.len(), "a copy batch could not be sent");
                self.shared
                    .failed
                    .fetch_add(jobs.len() as u64, Ordering::Relaxed);
                for job in jobs.drain(..) {
                    self.shared.submitted(job.posted_at);
                }
            }
        }
    }

    /// Records `jobs` into one list, makes the copy queue wait for the
    /// newest submission they name (invariant 2), runs it and signals the
    /// copy fence. On an error before the run, `jobs` are left to the
    /// caller.
    fn send(&mut self, jobs: &mut Vec<Job>) -> windows::core::Result<Flying> {
        let (allocator, list) = match self.free.pop() {
            Some(pair) => pair,
            // SAFETY: plain object creation; the list is closed at once so
            // that every use starts with a reset.
            None => unsafe {
                let allocator: ID3D12CommandAllocator = self
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COPY,
                    &allocator,
                    None,
                )?;
                list.Close()?;
                (allocator, list)
            },
        };
        let source = self.ring.buffer();
        // SAFETY: the allocator's last batch passed (or it is new); live
        // resources, held by the jobs until the batch passed.
        unsafe {
            list.Reset(&allocator, None)?;
            for job in jobs.iter() {
                match &job.dest {
                    Dest::Buffer { resource, offset } => {
                        list.CopyBufferRegion(
                            resource,
                            *offset,
                            source,
                            job.span.offset(),
                            job.span.len(),
                        );
                    }
                    Dest::Texture {
                        resource,
                        subresource,
                        footprint,
                    } => {
                        let mut footprint = *footprint;
                        footprint.Offset += job.span.offset();
                        let to = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(resource),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                SubresourceIndex: *subresource,
                            },
                        };
                        let from = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(source),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                                PlacedFootprint: footprint,
                            },
                        };
                        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                    }
                }
            }
            list.Close()?;
            let named_by = jobs.iter().map(|j| j.named_by).max().unwrap_or(0);
            if named_by > 0 && self.direct_fence.GetCompletedValue() < named_by {
                self.queue.Wait(&self.direct_fence, named_by)?;
            }
            self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
            let value = self.value + 1;
            // From here the batch is on the queue: its spans are read and
            // its jobs are held until the fence passes, signal or not.
            let signalled = self.queue.Signal(&self.fence, value);
            for job in jobs.iter_mut() {
                job.span.retire_at(value);
                self.shared.submitted(job.posted_at);
            }
            self.value = value;
            let flying = Flying {
                value,
                allocator,
                list,
                _jobs: std::mem::take(jobs),
            };
            match signalled {
                Ok(()) => {
                    // Only now may the direct queue wait for it (invariant 3).
                    self.shared.signalled.store(value, Ordering::Release);
                    Ok(flying)
                }
                Err(e) => {
                    // Never published: no one waits for a value that may
                    // never come. The batch stays held for good.
                    tracing::warn!(error = %e, "the copy fence could not be signalled");
                    std::mem::forget(flying);
                    Err(e)
                }
            }
        }
    }

    /// At the end: waits, bounded, for the batches in flight, so that their
    /// lists, ranges and destinations outlive the card's use of them.
    fn finish(mut self) {
        if !self.flying.is_empty() {
            // SAFETY: our fence and a new event, waited for here.
            unsafe {
                if let Ok(event) = CreateEventW(None, false, false, None)
                    && self.fence.SetEventOnCompletion(self.value, event).is_ok()
                {
                    WaitForSingleObject(event, 10_000);
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                }
            }
        }
        self.retire();
        if !self.flying.is_empty() {
            // The card did not finish in time: leak rather than free what it
            // may still read.
            tracing::warn!(
                batches = self.flying.len(),
                "copy batches still running at the end"
            );
            std::mem::forget(std::mem::take(&mut self.flying));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::ring::GRANULE;
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::*;

    /// A device of its own on the software adapter with the debug layer
    /// (when Windows has it), a direct queue and its fence, a copy queue.
    struct Rig {
        device: ID3D12Device,
        direct: ID3D12CommandQueue,
        direct_fence: ID3D12Fence,
        direct_value: u64,
        copy: CopyQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        event: HANDLE,
        messages: Option<ID3D12InfoQueue>,
    }

    impl Rig {
        fn new() -> Option<Rig> {
            Self::with_ring(1 << 20)
        }

        fn with_ring(ring: u64) -> Option<Rig> {
            // SAFETY: plain object creation on a device of the test's own.
            unsafe {
                let mut layer: Option<ID3D12Debug> = None;
                let _ = D3D12GetDebugInterface(&mut layer);
                layer.iter().for_each(|l| l.EnableDebugLayer());
                let adapter = crate::host::warp_adapter().ok()?;
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let messages = device.cast::<ID3D12InfoQueue>().ok();
                let direct: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let direct_fence: ID3D12Fence =
                    device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let ring = Arc::new(UploadRing::with_bytes(&device, ring).ok()?);
                let copy = CopyQueue::new(&device, &direct_fence, ring).ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                list.Close().ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                Some(Rig {
                    device,
                    direct,
                    direct_fence,
                    direct_value: 0,
                    copy,
                    allocator,
                    list,
                    event,
                    messages,
                })
            }
        }

        /// The first error the debug layer reported since the last call.
        fn error(&self) -> Option<String> {
            let queue = self.messages.as_ref()?;
            let mut first = None;
            // SAFETY: each message is read into a buffer of the size the
            // queue gives, then the queue is emptied.
            unsafe {
                for i in 0..queue.GetNumStoredMessages() {
                    let mut size = 0;
                    if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                        continue;
                    }
                    let mut buffer = vec![0u64; size.div_ceil(8)];
                    let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                        let m = &*message;
                        let text =
                            std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text)
                            .trim_end_matches('\0')
                            .to_owned();
                        if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first.is_none() {
                            first = Some(text);
                        }
                    }
                }
                queue.ClearStoredMessages();
            }
            first
        }

        /// Records a direct list with `record`, sends it through
        /// [`execute`] and signals the direct fence; does not wait.
        fn send(&mut self, record: impl FnOnce(&ID3D12GraphicsCommandList)) -> u64 {
            if self.direct_value > 0 {
                self.wait_direct();
            }
            // SAFETY: the allocator's last list finished: its signal was
            // just waited for (or there was none).
            unsafe {
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
                record(&self.list);
                self.list.Close().unwrap();
                execute(
                    &self.direct,
                    Some(&self.copy),
                    &[Some(self.list.cast().unwrap())],
                );
                self.direct_value += 1;
                self.direct
                    .Signal(&self.direct_fence, self.direct_value)
                    .unwrap();
            }
            self.copy.direct_signalled(self.direct_value);
            self.direct_value
        }

        fn wait_direct(&self) {
            // SAFETY: our fence and event.
            unsafe {
                self.direct_fence
                    .SetEventOnCompletion(self.direct_value, self.event)
                    .unwrap();
                assert_eq!(
                    WaitForSingleObject(self.event, 20_000).0,
                    0,
                    "the direct queue finished"
                );
            }
        }

        /// Waits (on the CPU) until the uploader signalled `value`.
        fn until_signalled(&self, value: u64) {
            let start = Instant::now();
            while self.copy.signalled() < value {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "the uploader sent the batch"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        fn wait_copy(&self, value: u64) {
            self.until_signalled(value);
            // SAFETY: our fence and event.
            unsafe {
                self.copy
                    .fence
                    .SetEventOnCompletion(value, self.event)
                    .unwrap();
                assert_eq!(
                    WaitForSingleObject(self.event, 20_000).0,
                    0,
                    "the copy queue finished"
                );
            }
        }

        fn buffer(
            &self,
            kind: D3D12_HEAP_TYPE,
            size: u64,
            state: D3D12_RESOURCE_STATES,
        ) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: kind,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size,
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
            // SAFETY: plain creation.
            unsafe {
                self.device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    None,
                    &mut buffer,
                )
            }
            .unwrap();
            buffer.unwrap()
        }

        /// A job writing `bytes` copies of `value` into `dest`.
        fn job(&self, value: u8, bytes: u64, dest: Dest, named_by: u64) -> Job {
            let mut span = self.copy.ring().alloc(bytes).expect("room in the ring");
            span.bytes_mut().fill(value);
            Job::new(span, dest, named_by)
        }
    }

    /// The texels of a 16x16 RGBA8 level read back at rows 256 bytes apart.
    fn texels(out: &ID3D12Resource) -> Vec<u8> {
        read(out, 4096)
            .chunks(256)
            .flat_map(|row| row[..64].to_vec())
            .collect()
    }

    /// The first `n` bytes of a read-back buffer.
    fn read(buffer: &ID3D12Resource, n: usize) -> Vec<u8> {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer the card is done with, read and unmapped.
        unsafe {
            buffer
                .Map(0, Some(&D3D12_RANGE { Begin: 0, End: n }), Some(&mut data))
                .unwrap();
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), n).to_vec();
            buffer.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            bytes
        }
    }

    fn barrier(
        resource: &ID3D12Resource,
        from: D3D12_RESOURCE_STATES,
        to: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        crate::host::transition(resource, from, to)
    }

    /// With no copy signalled, [`execute`] gives the direct queue no wait;
    /// after a batch it gives one, and none again until the next batch.
    #[test]
    fn the_direct_queue_waits_only_once_per_new_batch() {
        let Some(mut rig) = Rig::new() else {
            return; // no software adapter
        };
        for _ in 0..50 {
            rig.send(|_| {});
        }
        assert_eq!(
            rig.copy.direct_waits(),
            0,
            "an empty copy queue costs no wait"
        );
        let out = rig.buffer(
            D3D12_HEAP_TYPE_READBACK,
            256,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let job = rig.job(
            7,
            256,
            Dest::Buffer {
                resource: out.clone(),
                offset: 0,
            },
            0,
        );
        assert!(rig.copy.post(job).is_ok());
        rig.until_signalled(1);
        rig.send(|_| {});
        rig.send(|_| {});
        assert_eq!(rig.copy.direct_waits(), 1, "one wait for the one batch");
        rig.wait_direct();
        assert_eq!(read(&out, 256), vec![7; 256]);
        assert_eq!(rig.error(), None);
    }

    /// A level copied on the copy queue into a texture in the common state
    /// (implicit promotion to copy destination, decay back), then read on
    /// the direct queue after the wait [`execute`] gives: equal, and the
    /// debug layer says nothing.
    #[test]
    fn a_level_copied_on_copy_reads_back_equal_on_direct() {
        let Some(mut rig) = Rig::new() else {
            return;
        };
        let texture = {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 16,
                Height: 16,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                rig.device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut texture,
                )
            }
            .unwrap();
            texture.unwrap()
        };
        let (job, footprint) = level_job(&rig.device, &rig.copy, &texture, 0x5a, 0);
        assert!(rig.copy.post(job).is_ok());
        rig.until_signalled(1);
        let out = rig.buffer(
            D3D12_HEAP_TYPE_READBACK,
            4096,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // No wait for the copy on the CPU: the direct queue's wait orders it.
        rig.send(|list| {
            // SAFETY: an open list; live resources.
            unsafe {
                list.ResourceBarrier(&[barrier(
                    &texture,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                copy_level_out(list, &texture, &out, footprint);
                list.ResourceBarrier(&[barrier(
                    &texture,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
        });
        rig.wait_direct();
        assert_eq!(texels(&out), vec![0x5a; 1024]);
        assert_eq!(rig.error(), None);
    }

    /// A copy job for level 0 of a 16x16 RGBA8 `texture`, every byte
    /// `value`, naming direct value `named_by`; and the level's footprint.
    fn level_job(
        device: &ID3D12Device,
        copy: &CopyQueue,
        texture: &ID3D12Resource,
        value: u8,
        named_by: u64,
    ) -> (Job, D3D12_PLACED_SUBRESOURCE_FOOTPRINT) {
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: plain queries.
        unsafe {
            let desc = texture.GetDesc();
            device.GetCopyableFootprints(
                &desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
        }
        // 16 rows of 64 bytes, 256 apart; the last row is not padded.
        assert_eq!((footprint.Footprint.RowPitch, total), (256, 15 * 256 + 64));
        let mut span = copy.ring().alloc(total).expect("room in the ring");
        span.bytes_mut().fill(value);
        let dest = Dest::Texture {
            resource: texture.clone(),
            subresource: 0,
            footprint,
        };
        (Job::new(span, dest, named_by), footprint)
    }

    /// Records the copy of level 0 of `texture` (in the copy source state)
    /// into `out` at `footprint`.
    ///
    /// # Safety
    /// An open list; live resources.
    unsafe fn copy_level_out(
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
        out: &ID3D12Resource,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) {
        // SAFETY: as the caller promised; the copies of the interface
        // pointers do not outlive the call.
        unsafe {
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(out),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        }
    }

    /// A title texture object of 16x16 RGBA8 texels in a default heap,
    /// whose host copy is made on first use. Left to the process.
    fn title_texture() -> crate::object::This {
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 16,
            Height: 16,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..Default::default()
        };
        // ID3D12Resource's interface id, as the title asks for it.
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let mut texture: crate::object::This = std::ptr::null();
        assert_eq!(
            crate::resource::create_committed(
                std::ptr::null(),
                (&raw const props).cast(),
                0,
                (&raw const desc).cast(),
                0,
                std::ptr::null(),
                &iid,
                &mut texture,
            ),
            crate::object::S_OK
        );
        texture
    }

    /// Part B, T3: a texture a title list moved to the pixel shader
    /// resource state through `ensure_state` is returned to the common
    /// state (by the next title list's prelude, from its real state) before
    /// the copy; the copy runs on the copy queue behind that submission,
    /// and a title list reading it on the direct queue, through the wait
    /// `Host::run_lists` gives, reads it back equal. No debug-layer error
    /// (with `VICTORY_D3D12_DEBUG` set the host has the layer).
    #[test]
    fn a_texture_a_title_list_left_shader_readable_goes_to_common_before_the_copy() {
        const COMMON: u32 = 0;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        const COPY_SOURCE: u32 = 0x800;
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let copy = host.copy_queue().expect("the host has a copy queue");
        let _ = host.log_messages();
        let texture = title_texture();
        let resource = crate::resource::host_resource(texture)
            .expect("a host copy")
            .clone();
        let hosted = crate::resource::host_copy(texture).expect("a host copy");
        let run = |state: &[u32], record: &dyn Fn(&ID3D12GraphicsCommandList)| {
            let list = host.new_list().expect("a host list");
            for &s in state {
                crate::list::ensure_state(&list, texture, s);
            }
            record(&list.list);
            list.close();
            host.execute(&[&list]);
            list.reset();
        };
        run(&[PIXEL_SHADER_RESOURCE], &|_| {});
        assert_eq!(
            hosted.state.load(Ordering::Relaxed),
            PIXEL_SHADER_RESOURCE,
            "left shader-readable"
        );
        run(&[COMMON], &|_| {});
        assert_eq!(
            hosted.state.load(Ordering::Relaxed),
            COMMON,
            "back to common"
        );
        let before = copy.signalled();
        // Read before the post: other submissions may wait for the copy too.
        let waits = copy.direct_waits();
        // The copy names the submission that moved it back.
        let (job, footprint) = level_job(&host.device, copy, &resource, 0x3c, copy.direct_issued());
        assert!(copy.post(job).is_ok());
        let start = Instant::now();
        while copy.signalled() <= before {
            assert!(
                start.elapsed() < Duration::from_secs(10),
                "the uploader sent the batch"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        let out = host.read_back_buffer(4096).expect("a read-back buffer");
        run(&[PIXEL_SHADER_RESOURCE, COPY_SOURCE], &|list| {
            // SAFETY: an open list; live resources (the test holds them).
            unsafe { copy_level_out(list, &resource, &out, footprint) };
        });
        assert!(
            copy.direct_waits() > waits,
            "the direct queue waited for the copy"
        );
        assert_eq!(texels(&out), vec![0x3c; 1024]);
        assert_eq!(host.log_messages(), None, "the debug layer says nothing");
    }

    /// Invariant 2: a direct submission that writes the destination is held
    /// on the card (it waits for a fence the test signals); a copy job that
    /// names it waits for it on the card, so the copy's bytes land last.
    #[test]
    fn a_copy_batch_waits_for_the_direct_submission_that_names_its_destination() {
        let Some(mut rig) = Rig::new() else {
            return;
        };
        let blocker: ID3D12Fence =
            // SAFETY: plain creation.
            unsafe { rig.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.unwrap();
        let source = rig.buffer(
            D3D12_HEAP_TYPE_UPLOAD,
            256,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        {
            let mut data = std::ptr::null_mut();
            // SAFETY: an upload buffer, mapped and written before any use.
            unsafe {
                source.Map(0, None, Some(&mut data)).unwrap();
                std::ptr::write_bytes(data.cast::<u8>(), 1, 256);
                source.Unmap(0, None);
            }
        }
        let named = rig.buffer(
            D3D12_HEAP_TYPE_READBACK,
            256,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: our queue and fence.
        unsafe { rig.direct.Wait(&blocker, 1) }.unwrap();
        let writes = rig.send(|list| {
            // SAFETY: an open list; live buffers.
            unsafe { list.CopyBufferRegion(&named, 0, &source, 0, 256) };
        });
        let job = rig.job(
            2,
            256,
            Dest::Buffer {
                resource: named.clone(),
                offset: 0,
            },
            writes,
        );
        assert!(rig.copy.post(job).is_ok());
        rig.until_signalled(1);
        // Long enough for the software adapter to run a copy of 256 bytes.
        std::thread::sleep(Duration::from_millis(100));
        // SAFETY: plain queries.
        unsafe {
            assert_eq!(
                rig.direct_fence.GetCompletedValue(),
                0,
                "the direct queue is still held"
            );
            assert_eq!(
                rig.copy.fence.GetCompletedValue(),
                0,
                "so the copy waits too"
            );
        }
        assert_eq!(read(&named, 256), vec![0; 256], "nothing landed yet");
        // SAFETY: our fence, signalled from the CPU.
        unsafe { blocker.Signal(1) }.unwrap();
        rig.wait_direct();
        rig.wait_copy(1);
        assert_eq!(
            read(&named, 256),
            vec![2; 256],
            "the copy landed after the direct write"
        );
        assert_eq!(rig.error(), None);
    }

    /// Invariant 3: a job naming a direct value not yet issued is held by
    /// the uploader (the copy queue is never given a wait for a signal that
    /// was not issued) and goes once it is.
    #[test]
    fn a_job_naming_a_later_submission_is_held_until_it_is_issued() {
        let Some(mut rig) = Rig::new() else {
            return;
        };
        let out = rig.buffer(
            D3D12_HEAP_TYPE_READBACK,
            256,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let job = rig.job(
            9,
            256,
            Dest::Buffer {
                resource: out.clone(),
                offset: 0,
            },
            1,
        );
        let cut = super::super::STARTED.load(Ordering::Acquire) + 1;
        assert!(rig.copy.post(job).is_ok());
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(
            rig.copy.signalled(),
            0,
            "held: direct value 1 was not issued"
        );
        assert_eq!(
            rig.copy.unsubmitted_before(cut),
            1,
            "and counted as not submitted"
        );
        rig.send(|_| {});
        rig.wait_copy(1);
        assert_eq!(rig.copy.unsubmitted_before(cut), 0);
        assert_eq!(read(&out, 256), vec![9; 256]);
        assert_eq!(rig.error(), None);
    }

    /// Invariants 1-3 together: many direct submissions and copy jobs,
    /// each job naming the newest issued submission, the direct side
    /// waiting through [`execute`]; everything finishes (no wait cycle) and
    /// each buffer holds its last copy.
    #[test]
    fn interleaved_submissions_and_copies_always_finish() {
        let Some(mut rig) = Rig::with_ring(4 << 20) else {
            return;
        };
        let outs: Vec<_> = (0..4)
            .map(|_| {
                rig.buffer(
                    D3D12_HEAP_TYPE_READBACK,
                    GRANULE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )
            })
            .collect();
        let mut posted = 0;
        for round in 0..200u32 {
            let target = &outs[round as usize % outs.len()];
            let named_by = rig.copy.direct_issued();
            let value = (round % 251) as u8 + 1;
            let mut span = loop {
                if let Some(span) = rig.copy.ring().alloc(GRANULE) {
                    break span;
                }
                // Full: the uploader gives ranges back as batches pass.
                std::thread::sleep(Duration::from_millis(1));
            };
            span.bytes_mut().fill(value);
            let dest = Dest::Buffer {
                resource: target.clone(),
                offset: 0,
            };
            assert!(rig.copy.post(Job::new(span, dest, named_by)).is_ok());
            posted += 1;
            if round % 3 == 0 {
                rig.send(|_| {});
                rig.wait_direct();
            }
        }
        let start = Instant::now();
        while rig.copy.unsubmitted_before(u64::MAX) > 0 {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "every job went out"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        rig.wait_copy(rig.copy.signalled());
        rig.send(|_| {});
        rig.wait_direct();
        assert!(posted > 0);
        for (i, out) in outs.iter().enumerate() {
            let last = (0..200u32)
                .filter(|r| *r as usize % outs.len() == i)
                .max()
                .unwrap();
            assert_eq!(
                read(out, 16),
                vec![(last % 251) as u8 + 1; 16],
                "buffer {i} holds its last copy"
            );
        }
        assert_eq!(rig.copy.failed(), 0);
        assert_eq!(rig.error(), None);
    }

    /// Every list of ours reaches the host's direct queue through
    /// [`execute`]: no bare `ExecuteCommandLists` on the host's queue is
    /// left in the crate. The other calls are on queues of their own (test
    /// rigs, the card-awake thread, probes) or the start-up self-test, which
    /// runs before the copy queue exists.
    #[test]
    fn no_bare_execute_on_the_host_queue_is_left() {
        fn files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    files(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut all = Vec::new();
        files(&root, &mut all);
        // Functions of host.rs where a bare `queue` is not the host's queue
        // as lists reach it: the start-up self-test (the host's queue before
        // the copy queue exists) and two tests on devices of their own.
        const OWN_QUEUE: &[&str] = &[
            "gpu_upload_self_test",
            "buffer_states_track_transitions_and_execute_on_d3d12",
            "lifted_copies_match_the_cpu_and_keep_the_debug_layer_quiet",
        ];
        let mut found = Vec::new();
        let mut checked = 0;
        for path in &all {
            let name = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(path).unwrap();
            let in_host = name == "host.rs" || name.starts_with("host/");
            let mut function = "";
            for (number, line) in text.lines().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if let Some(at) = code.find("fn ") {
                    let rest = &code[at + 3..];
                    let end = rest
                        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .unwrap_or(rest.len());
                    if end > 0 {
                        function = &rest[..end];
                    }
                }
                let Some(at) = code.find(".ExecuteCommandLists(") else {
                    continue;
                };
                checked += 1;
                let receiver = code[..at].trim_start();
                let receiver = receiver.rsplit([' ', '{', '(']).next().unwrap_or("");
                let host_queue = receiver.ends_with("host.queue")
                    || receiver.ends_with("queue()")
                    || (in_host && receiver.ends_with("self.queue"))
                    || (name == "host.rs" && receiver == "queue" && !OWN_QUEUE.contains(&function));
                // The wrapper itself, and the uploader's own copy queue.
                let ours = [("execute", "direct"), ("send", "self.queue")];
                let wrapper = name == "host/copyq.rs" && ours.contains(&(function, receiver));
                if host_queue && !wrapper {
                    found.push(format!("{name}:{}: {}", number + 1, line.trim()));
                }
            }
        }
        assert!(checked >= 10, "the scan found the calls ({checked})");
        assert!(
            found.is_empty(),
            "lists sent past `copyq::execute`:\n{}",
            found.join("\n")
        );
    }
}
