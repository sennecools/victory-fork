//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
mod ahead;
#[cfg(windows)]
pub(crate) mod host;
#[cfg(windows)]
pub(crate) use ahead::{sync_ahead, sync_ahead_wait};

/// The bytes of a title's constant buffer of `size` that the host can see:
/// at most the 64 KiB a shader can index, so only those are staged. The
/// scratch allocation and the host's view round up to 256 on their own.
#[cfg(any(windows, test))]
pub(crate) fn constant_staging_bytes(size: u32) -> u32 {
    size.min(64 << 10)
}

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    pub(crate) kind: u32,
    pub(crate) count: u32,
    pub(crate) flags: u32,
    pub(crate) nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
///
/// Read by every lookup and by [`sync_heaps`], written only when a heap is
/// made or dropped (a dropped heap waits for a [`sync_heaps`] that walks
/// it). It was a mutex, and [`sync_heaps`] holds it for its whole pass:
/// fork main counted the title's recording threads waiting 48.5 ms a frame
/// for it in [`shader_slot`] (run 225), each lookup itself a few
/// nanoseconds. Readers no longer wait for each other. Nothing a holder
/// calls takes it again. (Adapted from fork main a2537066.)
#[cfg(windows)]
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// One [`sync_heaps`] at a time, as when the list above was a mutex.
#[cfg(windows)]
static SYNCING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Who reads [`SHADER_HEAPS`] for a slot (`site` of `heap list wait`): a
/// `Create*View` call, a descriptor copy, a slot that waited for its
/// texture ([`make_waiting`]), dynamic slots given back
/// ([`release_dynamic`]).
#[cfg(windows)]
mod list_site {
    pub(super) const VIEW: u64 = 0;
    pub(super) const COPY: u64 = 1;
    pub(super) const WAITING: u64 = 2;
    pub(super) const RELEASE: u64 = 3;
}

/// [`SHADER_HEAPS`] read-locked for one of the [`list_site`] sites. On the
/// profile, a wait for it of 50 us or more is `heap list wait` (`site`):
/// readers no longer wait for each other, but still for a heap made or
/// dropped, and behind such a writer that waits for a [`sync_heaps`].
#[cfg(windows)]
fn heap_list(site: u64) -> std::sync::RwLockReadGuard<'static, Vec<(u64, u64, usize)>> {
    let asked = crate::profile::stamp();
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    crate::profile::since("heap list wait", asked, 50, &[("site", site)]);
    heaps
}

/// An owned COM reference to a shader-visible [`Heap`], keeping it (and its
/// host heap and backing memory) alive past the [`SHADER_HEAPS`] lookup.
#[cfg(windows)]
struct HeapRef(*const Heap);

#[cfg(windows)]
impl std::ops::Deref for HeapRef {
    type Target = Heap;

    fn deref(&self) -> &Heap {
        // SAFETY: `HeapRef` retains a COM reference on the boxed `Heap`.
        unsafe { &*self.0 }
    }
}

#[cfg(windows)]
impl Drop for HeapRef {
    fn drop(&mut self) {
        release(self.0.cast());
    }
}

/// The shader-visible heap holding `handle`, and the descriptor's index,
/// looked up for `site` ([`list_site`]).
#[cfg(windows)]
fn shader_slot(handle: u64, site: u64) -> Option<(HeapRef, u32)> {
    let heaps = heap_list(site);
    let &(_, _, at) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))?;
    if !object::try_add_ref(at as This) {
        return None;
    }
    let heap = HeapRef(at as *const Heap);
    let index = heap.index(handle);
    // Drop the registry lock before `index?` can drop `heap` (which acquires
    // `SHADER_HEAPS` if it happens to release the final reference).
    drop(heaps);
    Some((heap, index?))
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap (`site`: [`list_site`]).
#[cfg(windows)]
fn mirror(handle: u64, site: u64) {
    let Some((heap, index)) = shader_slot(handle, site) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                // The key of what is mirrored, read before the mirror: a
                // record copied over it meanwhile is not taken as seen.
                let key = host::record_key(handle);
                host.mirror(index, handle, false);
                // Seen as mirrored: the next pass leaves the record if the
                // slot still shows it.
                host.saw_key(index, key);
            }
        }
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
///
/// Only the heaps in `bound` (their [`Heap`] addresses) are looked at:
/// those a submission's lists bind (`crate::host::HostList::bound_heaps`),
/// the only ones its lists can read. A heap no submission binds is not
/// looked at, however many pages the title writes in it; when one binds it
/// again its pass takes every page written meanwhile, since the journal's
/// stamps only rise and its take asks for those above its last
/// (`crate::watch::Tracked::take`).
#[cfg(windows)]
pub(crate) fn sync_heaps(bound: &[usize]) {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0")) {
        return;
    }
    let _one = SYNCING.lock().unwrap_or_else(|e| e.into_inner());
    // Once the title's end has begun (`crate::release`), no pass runs: the
    // slots' kept resources, the homes and the chunks are being let go, and
    // a pass would keep or mirror them again.
    if crate::host::ending() {
        return;
    }
    // On the profile: what it did, and its time reading pages (`read_us`,
    // summed over the threads) and the pass of each heap (`mirror_us`: all
    // the time it took on the calling thread, that thread's share of the
    // reading and its waits for the other threads with it, so it overlaps
    // `read_us`); `skipped_made`: mirrors done with before their slot's
    // claim, the slot made from the record already (`host::SKIPPED_MADE`,
    // every mirror's, the pass's among them).
    let mut span = crate::profile::span("sync heaps");
    let (mut read_ticks, mut mirror_ticks) = (0u64, 0u64);
    let started = std::time::Instant::now();
    let (mut pages, mut looked, mut mirrored) = (0u64, 0u64, 0u64);
    let skipped_before = host::SKIPPED_MADE.load(Relaxed);
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter().filter(|h| bound.contains(&h.2)) {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        #[cfg(test)]
        tests_hook::walking(at);
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // The heap's changed pages sorted first, then the slots found
        // mirrored (`sync_parted_heap`).
        let taken = written.take(start as usize, end as usize);
        let (found, made_ticks) = sync_parted_heap(host, &taken, start, step, mirrored_count);
        pages += found.pages;
        looked += found.looked;
        mirrored += found.make.len() as u64;
        read_ticks += found.read_ticks;
        mirror_ticks += made_ticks;
    }
    drop(heaps);
    if let Some(span) = span.as_mut() {
        span.arg("pages", pages);
        span.arg("looked", looked);
        span.arg("mirrored", mirrored);
        span.arg("read_us", crate::profile::micros(read_ticks));
        span.arg("mirror_us", crate::profile::micros(mirror_ticks));
        span.arg(
            "skipped_made",
            host::SKIPPED_MADE.load(Relaxed) - skipped_before,
        );
    }
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, started.elapsed().as_micros() as u64]) {
        count.fetch_add(n, Relaxed);
    }
}

/// At the title's end (`crate::release`), after `crate::host::end_submissions`:
/// waits until `until` for a heap sync pass that is still running (the
/// helper's, `ahead`, which a refused submission does not wait for). A
/// pass that starts later sees `ending()` and does nothing. False if a pass
/// still runs at `until`.
#[cfg(windows)]
pub(crate) fn sync_quiet(until: std::time::Instant) -> bool {
    use std::sync::TryLockError;
    loop {
        match SYNCING.try_lock() {
            Ok(_) | Err(TryLockError::Poisoned(_)) => return true,
            Err(TryLockError::WouldBlock) if std::time::Instant::now() < until => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            Err(TryLockError::WouldBlock) => return false,
        }
    }
}

/// For the release's tests: a heap sync pass "still running".
#[cfg(all(windows, test))]
pub(crate) fn hold_sync_for_test() -> std::sync::MutexGuard<'static, ()> {
    SYNCING.lock().unwrap_or_else(|e| e.into_inner())
}

/// At the title's end (`crate::release`): the slots of every shader-visible
/// heap whose host resource `ours` names let it go (`HostHeap::let_go_kept`).
/// Returns how many did.
#[cfg(windows)]
pub(crate) fn let_go_slots(
    ours: &dyn Fn(&windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> bool,
) -> u64 {
    let mut out = Vec::new();
    {
        let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
        for &(_, _, at) in heaps.iter() {
            // SAFETY: registered heaps are live (removed when dropped).
            let heap = unsafe { &*(at as *const Heap) };
            if let Some(Some(host)) = heap.host.get() {
                host.let_go_kept(ours, &mut out);
            }
        }
    }
    let n = out.len() as u64;
    drop(out);
    n
}

/// Threads [`sync_parted_heap`] runs its jobs on, the calling one among
/// them (`crate::draw::each_on`). cool_fox measured 14 ms a frame of one
/// thread for 160 changed pages (44629de).
#[cfg(windows)]
const SYNC_THREADS: usize = 4;

/// [`sync_heaps`] of one heap: the pages of the changed ranges `taken` of
/// the title heap at `start` (records of `step` bytes, the first
/// `mirrored_count` with host slots) are read by the safe copy
/// (`crate::safecopy`), their keys compared with the ones last seen and
/// what each slot shows, and sorted ([`host::HostHeap::sort_copied_page`]),
/// [`host::SYNC_PAGES`] a job; then the slots found with new records are
/// mirrored, [`host::SYNC_SLOTS`] a job, the jobs on [`SYNC_THREADS`]
/// threads. A slot is mirrored only after every page of the heap is read,
/// and not in slot order (each slot's mirror is ordered by its claim, as
/// the draws' and the heap buffer pass's are).
///
/// Returns what was found, and the profile ticks it took (all of them: the
/// reading is on the other threads too).
///
/// Ported from cool_fox 44629de (pages read, compared and sorted on several
/// threads, the slots made 256 at a time), by local.
#[cfg(windows)]
fn sync_parted_heap(
    host: &host::HostHeap,
    taken: &[(usize, usize)],
    start: u64,
    step: u64,
    mirrored_count: u32,
) -> (host::SyncPart, u64) {
    let began = crate::profile::stamp();
    let mut parts: Vec<(usize, usize)> = Vec::new();
    for &(range_from, range_end) in taken {
        let mut page = range_from & !(crate::watch::PAGE - 1);
        while page < range_end {
            parts.push((
                page.max(range_from),
                (page + crate::watch::PAGE).min(range_end),
            ));
            page += crate::watch::PAGE;
        }
    }
    let threads = SYNC_THREADS;
    let jobs: Vec<&[(usize, usize)]> = parts.chunks(host::SYNC_PAGES).collect();
    let found: Vec<std::sync::Mutex<host::SyncPart>> =
        jobs.iter().map(|_| Default::default()).collect();
    crate::draw::each_on(threads, jobs.len(), "victory-heap-sync", &|n| {
        let mut part = host::SyncPart::default();
        let mut page_bytes: Vec<u8> = Vec::new();
        for &(from, to) in jobs[n] {
            // The page's records are read by the safe copy
            // (`crate::safecopy`), into a copy: a page the title unmaps
            // meanwhile fails the read where 2,048 reads in place would
            // fault.
            page_bytes.resize(to - from, 0);
            let reading = crate::profile::stamp();
            let read = crate::host::read_into(from as *const u8, &mut page_bytes);
            part.read_ticks += crate::profile::elapsed(reading);
            if !read {
                continue;
            }
            part.pages += 1;
            let slots = changed_slots(from, to, start as usize, step, mirrored_count);
            host.sort_copied_page(slots, (start, step), from, &page_bytes, &mut part);
        }
        *found[n].lock().unwrap_or_else(|e| e.into_inner()) = part;
    });
    let mut all = host::SyncPart::default();
    let mut left = Vec::new();
    for part in found {
        let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
        all.pages += part.pages;
        all.looked += part.looked;
        all.read_ticks += part.read_ticks;
        all.make.append(&mut part.make);
        left.append(&mut part.left);
    }
    // The slots left to tables before whose marks have aged: read again,
    // whether or not their pages were written since.
    host.retake_left(&left, (start, step), &mut all);
    let make = &all.make;
    crate::draw::each_on(
        threads,
        make.len().div_ceil(host::SYNC_SLOTS),
        "victory-heap-sync",
        &|n| {
            for &index in make.chunks(host::SYNC_SLOTS).nth(n).unwrap_or_default() {
                host.mirror_synced(index, start + u64::from(index) * step);
            }
        },
    );
    (all, crate::profile::elapsed(began))
}

/// For the profile's sample (`VICTORY_PROFILE_SAMPLES`): over the host
/// heaps of shader-visible heaps, the distinct views their made slots were
/// made from (which sizes a pool of one descriptor per distinct record),
/// and the made slots. Each heap's slots are read a slot at a time, each
/// once no thread changes it, and counted after.
#[cfg(windows)]
pub(crate) fn live_keys() -> (u64, u64) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let mut keys: Vec<u64> = Vec::new();
    for &(_, _, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        if let Some(Some(host)) = heap.host.get() {
            keys.extend(host.made_keys().into_iter().filter(|&k| k != 0));
        }
    }
    drop(heaps);
    let slots = keys.len() as u64;
    keys.sort_unstable();
    keys.dedup();
    (keys.len() as u64, slots)
}

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen (mirrored), and the microseconds it took.
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    host::claims_line(frames);
    let [pages, looked, mirrored, micros] = [0, 1, 2, 3].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if pages == 0 {
        return;
    }
    // A record whose key is the one last seen, in a slot that shows it or
    // nothing, is left before the question whether its slot was a table's,
    // so "left to the draws" below counts only the others.
    {
        // (`SKIPPED_MADE` only grows: the line's part since the last line.)
        static LINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let skipped = host::SKIPPED_MADE.load(std::sync::atomic::Ordering::Relaxed);
        let since = skipped.wrapping_sub(LINED.swap(skipped, std::sync::atomic::Ordering::Relaxed));
        tracing::info!(
            "view heap slots: per frame {:.0} mirrors whose slot was made from the record \
             already, done with before the slot's claim; the count left to the draws below is \
             of records the heap sync did not leave",
            since as f64 / frames
        );
    }
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.0} left to the \n         draws that read them through tables (VICTORY_HEAP_SYNC_TABLES=1 mirrors those too), {:.2} ms",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        host::COPIED_TABLED.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames,
        micros as f64 / 1000.0 / frames
    );
    // (`LEFT_DROPPED` only grows: the line's part since the last line.)
    static DROPPED_LINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dropped = host::LEFT_DROPPED.load(std::sync::atomic::Ordering::Relaxed);
    let since =
        dropped.wrapping_sub(DROPPED_LINED.swap(dropped, std::sync::atomic::Ordering::Relaxed));
    if since > 0 {
        tracing::info!(
            "heap sync: per frame {:.1} slots left to tables found no room to be read again once \
             their mark ages (read again only if their page is written)",
            since as f64 / frames
        );
    }
}

/// For tests: a heap [`sync_heaps`] stops at, until the test lets it go
/// on (a pass held while it walks that heap).
#[cfg(all(windows, test))]
pub(crate) mod tests_hook {
    use std::sync::mpsc::{Receiver, Sender};

    /// The heap's address, where to say the pass came to it, and where it
    /// is told to go on. Taken by the first pass that comes to it.
    pub(crate) static AT: std::sync::Mutex<Option<(usize, Sender<()>, Receiver<()>)>> =
        std::sync::Mutex::new(None);

    /// [`super::sync_heaps`] comes to the heap at `at`.
    pub(crate) fn walking(at: usize) {
        let held = {
            let mut hook = AT.lock().unwrap_or_else(|e| e.into_inner());
            match hook.as_ref() {
                Some(&(heap, _, _)) if heap == at => hook.take(),
                _ => None,
            }
        };
        if let Some((_, came, go)) = held {
            let _ = came.send(());
            let _ = go.recv();
        }
    }
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = heap_list(list_site::RELEASE);
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
        drop(heaps);
        if let Some(Some(host)) = self.host.get() {
            crate::memory::forget_heap(host);
        }
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them).
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is.
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<Shape, u32>,
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    shapes: Vec::new(),
    shape_index: std::collections::BTreeMap::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// The views of [`Views`] as its readers see them, without its lock: every
/// slot a draw reads through a table, the heap sync and the title's writes
/// look a view up here, on many threads at once, and the lock and the copy
/// of a view out from under it were the cost (`view_at`). Writers change it
/// under [`VIEWS`]' write lock alone, entry by entry, as they change
/// [`Views::list`] and [`Views::shapes`].
///
/// Entries live in chunks of [`CHUNK`] that are made as the table grows and
/// never moved or freed, so a reader holds no reference that a writer could
/// pull away. Each entry is a seqlock: its sequence word (odd while a writer
/// changes the entry) is read before and after its generation and words,
/// and a reader that finds it odd, or changed, reads again.
/// A view holds only plain words (no reference of any kind), so a view read
/// whole between two equal generations is the one that was written, and a
/// changed generation (the resource released, `forget`) is never returned.
mod table {
    use std::sync::atomic::{
        AtomicPtr, AtomicU32,
        Ordering::{Acquire, Relaxed, Release},
        fence,
    };

    /// Entries a chunk.
    pub(super) const CHUNK: usize = 1 << 12;
    /// Chunks: 4,194,304 entries in all. An entry past them is not in the
    /// table: readers look it up under the lock, as before
    /// (`super::entry_view`).
    pub(super) const CHUNKS: usize = 1 << 10;
    /// Words of an entry ([`super::View::words`]).
    pub(super) const WORDS: usize = 32;

    pub(super) struct Entry {
        /// Odd while a writer changes the entry; one more after each change.
        /// (Not the generation alone: an entry is written anew at the
        /// generation it has, so a reader could find the same generation
        /// before and after a write and take words of both.)
        sequence: AtomicU32,
        generation: AtomicU32,
        words: [AtomicU32; WORDS],
    }

    impl Entry {
        const fn new() -> Self {
            Entry {
                sequence: AtomicU32::new(0),
                generation: AtomicU32::new(0),
                words: [const { AtomicU32::new(0) }; WORDS],
            }
        }

        /// Changes the entry with `change`, seen by readers as one change.
        fn change(&self, change: impl FnOnce(&Self)) {
            let at = self.sequence.load(Relaxed);
            self.sequence.store(at.wrapping_add(1), Relaxed);
            fence(Release);
            change(self);
            self.sequence.store(at.wrapping_add(2), Release);
        }
    }

    /// One table of entries: the views ([`super::Views::list`]) or the
    /// shapes of placed views ([`super::Views::shapes`]).
    pub(super) struct Table {
        chunks: [AtomicPtr<Entry>; CHUNKS],
    }

    impl Table {
        pub(super) const fn new() -> Self {
            Table {
                chunks: [const { AtomicPtr::new(std::ptr::null_mut()) }; CHUNKS],
            }
        }

        /// Entry `i`, if its chunk is made.
        fn get(&self, i: u32) -> Option<&Entry> {
            let i = i as usize;
            let chunk = self.chunks.get(i / CHUNK)?.load(Acquire);
            // SAFETY: a made chunk is `CHUNK` entries, never freed.
            (!chunk.is_null()).then(|| unsafe { &*chunk.add(i % CHUNK) })
        }

        /// Entry `i`, its chunk made if it is not; `None` past the table.
        /// Called under the write lock of [`super::VIEWS`] alone.
        fn made(&self, i: u32) -> Option<&Entry> {
            let at = self.chunks.get(i as usize / CHUNK)?;
            if at.load(Acquire).is_null() {
                let chunk: Box<[Entry]> = (0..CHUNK).map(|_| Entry::new()).collect();
                at.store(Box::leak(chunk).as_mut_ptr(), Release);
            }
            self.get(i)
        }

        /// Writes entry `i`: `words` at `generation`. Under the write lock of
        /// [`super::VIEWS`].
        pub(super) fn write(&self, i: u32, generation: u32, words: &[u32; WORDS]) {
            let Some(entry) = self.made(i) else {
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(i, "view table full: views past it are read under the lock")
                });
                return;
            };
            entry.change(|entry| {
                entry.generation.store(generation, Relaxed);
                for (word, &w) in entry.words.iter().zip(words) {
                    word.store(w, Relaxed);
                }
            });
        }

        /// Entry `i` is at `generation` from now on, its words as they are
        /// (`forget`). Under the write lock of [`super::VIEWS`].
        pub(super) fn age(&self, i: u32, generation: u32) {
            if let Some(entry) = self.get(i) {
                entry.change(|entry| entry.generation.store(generation, Relaxed));
            }
        }

        /// The bytes of the chunks made so far (for the memory line).
        #[cfg(all(windows, feature = "diagnostics"))]
        pub(super) fn bytes(&self) -> u64 {
            let made = self
                .chunks
                .iter()
                .filter(|chunk| !chunk.load(Relaxed).is_null())
                .count();
            (made * CHUNK * size_of::<Entry>()) as u64
        }

        /// Tests: entry `i` as a writer leaves it while it changes it
        /// (`busy`), or after it (not).
        #[cfg(test)]
        pub(super) fn hold_busy(&self, i: u32, busy: bool) {
            if let Some(entry) = self.get(i) {
                let at = entry.sequence.load(Relaxed);
                if (at % 2 == 1) != busy {
                    entry.sequence.store(at.wrapping_add(1), Release);
                }
            }
        }

        /// The words of entry `i` if it is at `generation`: read whole,
        /// between two looks at its sequence that find it the same and no
        /// writer at it.
        pub(super) fn read(&self, i: u32, generation: u32) -> Option<[u32; WORDS]> {
            let entry = self.get(i)?;
            let mut spins = 0u32;
            loop {
                let before = entry.sequence.load(Acquire);
                if before % 2 == 1 {
                    #[cfg(test)]
                    super::BUSY_READS.with(|n| n.set(n.get() + 1));
                    spins += 1;
                    if spins < 64 {
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                    continue;
                }
                let now = entry.generation.load(Relaxed);
                let words = std::array::from_fn(|w| entry.words[w].load(Relaxed));
                fence(Acquire);
                if entry.sequence.load(Relaxed) == before {
                    return (now == generation).then_some(words);
                }
            }
        }
    }
}

/// The views of [`Views::list`] for readers ([`table`]).
static VIEW_TABLE: table::Table = table::Table::new();
/// The shapes of [`Views::shapes`] for readers ([`table`]), each written
/// once, at generation 1, before it is named.
static SHAPE_TABLE: table::Table = table::Table::new();

#[cfg(test)]
thread_local! {
    /// Tests: reads of the view table on this thread that found an entry
    /// being written and read again.
    static BUSY_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

impl View {
    /// The view as the words of an entry of [`table`]: what it is, then its
    /// fields.
    fn words(&self) -> [u32; table::WORDS] {
        let mut w = [0u32; table::WORDS];
        let wide = |w: &mut [u32], v: u64| (w[0], w[1]) = (v as u32, (v >> 32) as u32);
        match self {
            View::Constant(address, size) => {
                w[0] = 1;
                wide(&mut w[1..], *address);
                w[3] = *size;
            }
            View::Shader(r, desc) => {
                w[0] = 2;
                wide(&mut w[1..], *r as u64);
                w[3] = u32::from(desc.is_some());
                w[4..14].copy_from_slice(&desc.unwrap_or_default());
            }
            View::Unordered(r, counter, desc) => {
                w[0] = 3;
                wide(&mut w[1..], *r as u64);
                wide(&mut w[3..], *counter as u64);
                w[5] = u32::from(desc.is_some());
                w[6..16].copy_from_slice(&desc.unwrap_or_default());
            }
            View::Sampler(desc) => {
                w[0] = 4;
                w[1..14].copy_from_slice(desc);
            }
            View::ConsoleSampler(desc) => {
                w[0] = 5;
                w[1..28].copy_from_slice(desc);
            }
            View::Placed(unordered, resource, desc) => {
                w[0] = 6;
                w[1] = u32::from(*unordered);
                w[2..16].copy_from_slice(resource);
                w[16..32].copy_from_slice(desc);
            }
        }
        w
    }

    /// The view [`View::words`] wrote; `None` for words no view writes.
    fn from_words(w: &[u32; table::WORDS]) -> Option<View> {
        let wide = |at: usize| u64::from(w[at]) | u64::from(w[at + 1]) << 32;
        let desc = |has: u32, at: usize| {
            (has != 0).then(|| <[u32; 10]>::try_from(&w[at..at + 10]).unwrap_or_default())
        };
        Some(match w[0] {
            1 => View::Constant(wide(1), w[3]),
            2 => View::Shader(wide(1) as usize, desc(w[3], 4)),
            3 => View::Unordered(wide(1) as usize, wide(3) as usize, desc(w[5], 6)),
            4 => View::Sampler(w[1..14].try_into().ok()?),
            5 => View::ConsoleSampler(w[1..28].try_into().ok()?),
            6 => View::Placed(
                w[1] != 0,
                w[2..16].try_into().ok()?,
                w[16..32].try_into().ok()?,
            ),
            _ => return None,
        })
    }
}

/// A shape ([`Shape`]) as the words of an entry of [`table`].
fn shape_words(&(unordered, resource, rest): &Shape) -> [u32; table::WORDS] {
    let mut w = [0u32; table::WORDS];
    w[0] = u32::from(unordered);
    w[1..15].copy_from_slice(&resource);
    w[15..29].copy_from_slice(&rest);
    w
}

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

impl Views {
    /// The number of `shape`, a new one if it is not there yet; `None` when
    /// the shapes are full (the view is then a table entry).
    fn shape(&mut self, shape: Shape) -> Option<u32> {
        if let Some(&at) = self.shape_index.get(&shape) {
            return Some(at);
        }
        if self.shapes.len() >= SHAPES {
            return None;
        }
        let at = self.shapes.len() as u32;
        // Written for readers before any record names it.
        SHAPE_TABLE.write(at, 1, &shape_words(&shape));
        self.shapes.push(shape);
        self.shape_index.insert(shape, at);
        Some(at)
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
        let &i = self.index.get(view)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View) -> [u32; 2] {
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.list.len() as u32 - 1
                    }
                };
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                // For readers, before any record names it.
                VIEW_TABLE.write(i, self.list[i as usize].1, &view.words());
                self.index.insert(view, i);
                i
            }
        };
        [i, self.list[i as usize].1]
    }
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                let (shape, address) = shape_of(*unordered, resource, desc);
                if address >> PLACED_ADDRESS_BITS == 0 {
                    // Nearly every placed view has a shape seen before.
                    let known = views_read().shape_index.get(&shape).copied();
                    let shape = known.or_else(|| views_write().shape(shape));
                    if let Some(shape) = shape {
                        NAMED_WHOLE[1].fetch_add(1, Relaxed);
                        return placed_words(shape, address);
                    }
                }
            }
            _ => {}
        }
    }
    // A view written before keeps its entry and generation.
    if let Some(words) = views_read().entry_of(&view) {
        return words;
    }
    views_write().entry(view)
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, and one of the index with its share of
/// a tree node; the shapes likewise; and the chunks of the tables readers
/// read them from without the lock, [`table`], made so far); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> [u64; 8] {
    use std::sync::atomic::Ordering::Relaxed;
    let views = views_read();
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let (mut constants, mut placed) = (0, 0);
    for (view, _) in &views.list {
        match view {
            View::Constant(..) => constants += 1,
            View::Placed(..) => placed += 1,
            _ => {}
        }
    }
    [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.len() as u64 * shape * 3 / 2
            + VIEW_TABLE.bytes()
            + SHAPE_TABLE.bytes(),
        constants,
        placed,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ]
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// How many textures and slots wait for a host copy now (a gauge for the
/// black census, read under the table's lock).
#[cfg(windows)]
pub(crate) fn waiting_gauge() -> [u64; 2] {
    let waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
    [waiting.len() as u64, waiting.values().map(|slots| slots.len() as u64).sum()]
}

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    crate::census::VIEW_NULL_WAITING[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            let Some((heap, index)) = shader_slot(handle, list_site::WAITING) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                crate::census::VIEW_NULL_WAITING[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                host.mirror_at(index, handle, true, crate::descriptor::host::hold::WAITING);
            }
        }
    }
}

/// Writes the record of `view` into the descriptor slot at `handle`, and
/// mirrors it if the slot is in a shader-visible heap.
pub(crate) fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle, list_site::VIEW);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    let mut views = views_write();
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        // (Below the bits that say a record is no table entry.)
        let newer = ((generation + 1) & GENERATIONS).max(1);
        views.list[i as usize].1 = newer;
        VIEW_TABLE.age(i, newer);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own. Takes
/// no lock ([`table`]).
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => entry_view(i, generation, |i| {
            views_read().list.get(i as usize).cloned()
        }),
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => placed_of(i, generation),
        _ => None,
    }
}

/// The view of table entry `i` at `generation`: from the reader table
/// ([`table`]) without a lock, or, for an entry past it, from `list` (the
/// entry and its generation, read under the lock of [`VIEWS`]).
fn entry_view(
    i: u32,
    generation: u32,
    list: impl FnOnce(u32) -> Option<(View, u32)>,
) -> Option<View> {
    if (i as usize) / table::CHUNK < table::CHUNKS {
        return View::from_words(&VIEW_TABLE.read(i, generation)?);
    }
    let (view, current) = list(i)?;
    (current == generation).then_some(view)
}

/// The view of placed memory [`placed_words`] named.
fn placed_of(i: u32, generation: u32) -> Option<View> {
    let payload = payload(i, generation);
    let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
    let w = SHAPE_TABLE.read((payload >> PLACED_ADDRESS_BITS) as u32, 1)?;
    let resource: [u32; 14] = w[1..15].try_into().ok()?;
    let mut desc = [0u32; 16];
    desc[..10].copy_from_slice(&w[15..25]);
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc[12..].copy_from_slice(&w[25..29]);
    Some(View::Placed(w[0] != 0, resource, desc))
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    let view = match words(desc) {
        Some([lo, hi, size, _]) => View::Constant(u64::from(lo) | u64::from(hi) << 32, size),
        None => View::Constant(0, 0),
    };
    record(handle, view);
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // Records copied onto the same bytes are not written: their page stays
        // unwritten for the write watch, and the heap sync does not read it
        // for them. They are mirrored all the same. The title copies its
        // tables' records into its heaps each frame, many of them onto what
        // is there already; a write of the same bytes would still mark the
        // page written. What the title reads back is the same either way.
        // (Ported from the `VICTORY_SAME_RECORDS` check of cool_fox e8d3d91,
        // by local, there in its own write of the title's bytes, which this
        // port leaves out.)
        // SAFETY: both ranges are descriptor slots the title owns (they may
        // overlap: only read here).
        let same = len > 0
            && unsafe {
                std::slice::from_raw_parts(dst as *const u8, len as usize)
                    == std::slice::from_raw_parts(src as *const u8, len as usize)
            };
        if !same {
            // SAFETY: as above.
            unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        }
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step, list_site::COPY);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// Two views of every word the table keeps, each word unlike the other's.
    fn two_views() -> [View; 2] {
        let placed = |base: u32| {
            View::Placed(
                base % 2 == 1,
                std::array::from_fn(|w| base + w as u32),
                std::array::from_fn(|w| base * 7 + w as u32),
            )
        };
        [placed(1), placed(1000)]
    }

    /// Every kind of view comes back from the words of its entry as it was.
    #[test]
    fn a_view_comes_back_from_its_words() {
        let desc: [u32; 10] = std::array::from_fn(|w| w as u32 + 3);
        for view in [
            View::Constant(0x7d_5000_1234, 4096),
            View::Shader(0x1234_5678_9abc, None),
            View::Shader(0x1234_5678_9abc, Some(desc)),
            View::Unordered(0xdead_beef_0000, 0x77, Some(desc)),
            View::Unordered(0xdead_beef_0000, 0, None),
            View::Sampler(std::array::from_fn(|w| w as u32)),
            View::ConsoleSampler(std::array::from_fn(|w| 100 + w as u32)),
            two_views()[0].clone(),
            two_views()[1].clone(),
        ] {
            assert_eq!(View::from_words(&view.words()), Some(view));
        }
        assert_eq!(View::from_words(&[0; table::WORDS]), None);
    }

    /// Four readers and a writer that rewrites one entry over and over, for
    /// at most half a second: every read is one of the two views whole, never
    /// words of both, and both are read.
    #[test]
    fn view_table_seqlock_no_torn_reads() {
        static TABLE: table::Table = table::Table::new();
        const READS: u64 = if cfg!(debug_assertions) {
            100_000
        } else {
            1_000_000
        };
        let views = two_views();
        let words = views.clone().map(|v| v.words());
        TABLE.write(5, 1, &words[0]);
        let stop = std::sync::atomic::AtomicBool::new(false);
        let written = std::sync::atomic::AtomicU64::new(0);
        // Each reader stops at `READS` or after half a second.
        let until = std::time::Instant::now() + std::time::Duration::from_millis(500);
        std::thread::scope(|scope| {
            // The writer first, and some of its writes before any read.
            let writer = scope.spawn(|| {
                let mut n = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    TABLE.write(5, 1, &words[n % 2]);
                    n += 1;
                    written.store(n as u64, std::sync::atomic::Ordering::Relaxed);
                    // A short pause: readers get through between writes as
                    // well as across them.
                    for _ in 0..16 {
                        std::hint::spin_loop();
                    }
                }
            });
            while written.load(std::sync::atomic::Ordering::Relaxed) < 1_000
                && !writer.is_finished()
            {
                std::hint::spin_loop();
            }
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        let mut seen = [0u64; 2];
                        for n in 0..READS {
                            if n % 1024 == 0 && std::time::Instant::now() >= until {
                                break;
                            }
                            let read = TABLE.read(5, 1).expect("at its generation");
                            let view = View::from_words(&read).expect("a whole view");
                            let at = views
                                .iter()
                                .position(|v| *v == view)
                                .expect("a view written");
                            seen[at] += 1;
                        }
                        seen
                    })
                })
                .collect();
            // Every reader is joined and the writer stopped before anything
            // is asserted: a reader that failed ends the test, not hangs it.
            let seen: Vec<_> = readers.into_iter().map(|r| r.join()).collect();
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            let seen: Vec<[u64; 2]> = seen
                .into_iter()
                .map(|r| r.expect("a reader read only whole views"))
                .collect();
            assert!(seen.iter().all(|s| s[0] + s[1] > 0), "every reader read");
            let both = seen
                .iter()
                .fold([0u64; 2], |a, s| [a[0] + s[0], a[1] + s[1]]);
            assert!(both[0] > 0 && both[1] > 0, "both views read: {both:?}");
        });
    }

    /// An entry past the reader table is looked up as before, with its
    /// generation checked; one inside it is read from the table.
    #[test]
    fn an_entry_past_the_reader_table_is_read_under_the_lock() {
        let past = (table::CHUNK * table::CHUNKS) as u32;
        let view = two_views()[0].clone();
        let list = |i: u32| (i == past).then(|| (view.clone(), 7));
        assert_eq!(entry_view(past, 7, list), Some(view.clone()));
        assert_eq!(entry_view(past, 8, list), None, "another generation");
        assert_eq!(entry_view(past + 1, 7, list), None, "no entry");
        // Inside the table: the list is not asked.
        assert_eq!(
            entry_view(past - 1, 7, |_| panic!("asked the list")),
            None,
            "nothing written there"
        );
    }

    /// A read of an entry a writer is changing waits for it and reads what
    /// was written.
    #[test]
    fn view_table_busy_reads_retry() {
        static TABLE: table::Table = table::Table::new();
        let views = two_views();
        TABLE.write(9, 3, &views[0].words());
        TABLE.hold_busy(9, true);
        let reader = std::thread::spawn(|| {
            let read = TABLE.read(9, 3);
            (read, BUSY_READS.with(std::cell::Cell::get))
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!reader.is_finished(), "the read waits for the writer");
        TABLE.hold_busy(9, false);
        let (read, busy) = reader.join().unwrap();
        assert_eq!(
            read.and_then(|w| View::from_words(&w)),
            Some(views[0].clone())
        );
        assert!(busy > 0, "it found the entry busy");
    }

    /// A record of an entry whose generation moved on (its resource
    /// released) names nothing, and a chunk never made reads nothing.
    #[test]
    fn stale_generation_reads_none() {
        static TABLE: table::Table = table::Table::new();
        let views = two_views();
        TABLE.write(4, 1, &views[0].words());
        assert!(TABLE.read(4, 1).is_some());
        TABLE.age(4, 2);
        assert_eq!(TABLE.read(4, 1), None, "an older generation");
        assert!(TABLE.read(4, 2).is_some(), "the entry at its own");
        assert_eq!(
            TABLE.read(4 + table::CHUNK as u32 * 3, 1),
            None,
            "no chunk there"
        );
    }

    /// For the record, not gated: a read of the table against a read under
    /// an uncontended lock with a clone of the view, as before.
    #[test]
    fn view_at_microbench() {
        static TABLE: table::Table = table::Table::new();
        const N: u32 = 200_000;
        let view = two_views()[1].clone();
        TABLE.write(1, 1, &view.words());
        let locked = std::sync::RwLock::new(vec![(view.clone(), 1u32); 2]);
        let started = std::time::Instant::now();
        for _ in 0..N {
            std::hint::black_box(View::from_words(&TABLE.read(1, 1).unwrap()));
        }
        let table = started.elapsed();
        let started = std::time::Instant::now();
        for _ in 0..N {
            let list = locked.read().unwrap();
            let (v, g) = &list[1];
            std::hint::black_box((*g == 1).then(|| v.clone()));
        }
        let lock = started.elapsed();
        // Four threads at once, as the recording threads and the heap sync.
        let four = |read: &(dyn Fn() + Sync)| {
            let started = std::time::Instant::now();
            std::thread::scope(|scope| {
                for _ in 0..4 {
                    scope.spawn(|| (0..N).for_each(|_| read()));
                }
            });
            started.elapsed()
        };
        let table_four = four(&|| {
            std::hint::black_box(View::from_words(&TABLE.read(1, 1).unwrap()));
        });
        let lock_four = four(&|| {
            let list = locked.read().unwrap();
            let (v, g) = &list[1];
            std::hint::black_box((*g == 1).then(|| v.clone()));
        });
        let ns = |d: std::time::Duration| d.as_nanos() as f64 / f64::from(N);
        println!(
            "view_at: one thread: table {:.1} ns, lock and clone {:.1} ns a read; four threads: \
             table {:.1} ns, lock and clone {:.1} ns a read each",
            ns(table),
            ns(lock),
            ns(table_four),
            ns(lock_four)
        );
    }

    #[test]
    fn constant_buffer_staging_uses_only_the_host_addressable_prefix() {
        for (size, expected) in [
            (0, 0),
            (255, 255),
            (65_535, 65_535),
            (65_536, 65_536),
            (65_537, 65_536),
            (2_785_536, 65_536),
            (u32::MAX, 65_536),
        ] {
            assert_eq!(
                constant_staging_bytes(size),
                expected,
                "guest CBV size {size}"
            );
        }
    }

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            sync_heaps(&[h as usize]);
            assert!(host.made(5));
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses.
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ee0_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    #[test]
    fn null_cbv_overwrites_previous_descriptor() {
        let null = std::ptr::null();
        let slot = [0u64; 4];
        let at = slot.as_ptr() as u64;
        create_cbv(null, &[0x1000, 0, 256, 0], at);
        assert_eq!(view(at), Some(View::Constant(0x1000, 256)));
        create_cbv(null, std::ptr::null(), at);
        assert_eq!(view(at), Some(View::Constant(0, 0)));
    }

    #[cfg(windows)]
    #[test]
    fn shader_slot_retains_heap_across_title_release() {
        let desc = HeapDesc {
            kind: 0,
            count: 8,
            flags: SHADER_VISIBLE,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = std::ptr::null();
        assert_eq!(create(desc, none, &mut h), S_OK);
        let start = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);

        let (guard, index) =
            shader_slot(start + 32, list_site::VIEW).expect("registered shader-visible heap");
        assert_eq!(index, 1);

        // Title drops its reference while `guard` is still held: heap must stay alive.
        let rel = slot::<extern "system" fn(This) -> u32>(h, 0x10);
        assert_eq!(rel(h), 1, "shader_slot guard must keep Heap alive");
        // Backing memory remains mapped and readable/writable while `guard` is live.
        // SAFETY: slot 1 lies inside the still-live 8-descriptor heap.
        unsafe { ((start + 32) as *mut u32).write(0xCAFE_BABE) };
        // SAFETY: as above.
        assert_eq!(unsafe { ((start + 32) as *const u32).read() }, 0xCAFE_BABE);

        // Dropping the guard releases the final reference and unregisters the heap.
        drop(guard);
        assert!(shader_slot(start + 32, list_site::VIEW).is_none());

        // `try_add_ref` refuses to resurrect a zero-refcount object.
        let dead = Object::new(&FUNCTIONS);
        assert!(!object::try_add_ref(&raw const dead));
    }
}
