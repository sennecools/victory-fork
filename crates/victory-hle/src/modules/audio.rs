//! The console's audio hardware (`AcpHal.dll`):
//!
//! * the APU heap: host memory with 32-bit device addresses;
//! * SHAPE context arrays allocated from it;
//! * the audio coprocessor's command and message queues, where every
//!   command completes at once and is acknowledged as the hardware would;
//! * the XMA decoder: enabled XMA contexts are decoded on a thread of ours
//!   into their output rings, like the hardware does.
//!
//! SHAPE's other units (rate conversion, EQ, filters, mixing through
//! flowgraphs) are not emulated, so voices that need them stay silent.
//! Mixed audio leaves through the host's WASAPI, which titles and
//! middleware reach through COM like on PC. fmod's console build sets up
//! the coprocessor inside `System_Create` and waits there for its first
//! command to be acknowledged; the title uses only XMA contexts.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::{BTreeMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard};
use std::time::{Duration, Instant};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

#[cfg(all(windows, feature = "diagnostics"))]
pub mod capture;

const E_FAIL: HRESULT = 0x8000_4005_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// The APU heap.

struct Allocation {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: `Allocation` owns a heap allocation freed when the last `Arc` drops.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: allocated by `heap_alloc` with this layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// A live slice of an APU heap block that retains the backing [`Allocation`]
/// until dropped, preventing `ApuFree` from deallocating memory under active
/// XMA decoding or command notifications.
struct HeapSlice {
    _alloc: Arc<Allocation>,
    ptr: *mut u8,
    len: usize,
}

impl std::ops::Deref for HeapSlice {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr..ptr+len` lies inside `_alloc`, kept alive until drop.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for HeapSlice {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; the title and audio hardware share the block.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

struct Block {
    size: usize,
    apu: u32,
    alloc: Arc<Allocation>,
}

struct Heap {
    /// By host address.
    blocks: BTreeMap<usize, Block>,
    /// Host address by device address.
    by_apu: BTreeMap<u32, usize>,
    /// Free device address ranges, start to end, merged with neighbours.
    free: BTreeMap<u32, u32>,
    /// Set by `ApuCreateHeap` or the first allocation: the heap exists.
    created: bool,
}

static HEAP: Mutex<Heap> = Mutex::new(Heap {
    blocks: BTreeMap::new(),
    by_apu: BTreeMap::new(),
    free: BTreeMap::new(),
    created: false,
});

/// Device addresses start above 0, which means "none" to titles.
const APU_BASE: u32 = 0x0001_0000;

impl Heap {
    /// First fit in the free ranges.
    fn take(&mut self, size: u32, align: u32) -> Option<u32> {
        if self.free.is_empty() && !self.created {
            self.free.insert(APU_BASE, u32::MAX);
        }
        self.created = true;
        let (start, end, apu) = self.free.iter().find_map(|(&start, &end)| {
            let apu = start.checked_next_multiple_of(align)?;
            (apu.checked_add(size)? <= end).then_some((start, end, apu))
        })?;
        self.free.remove(&start);
        if start < apu {
            self.free.insert(start, apu);
        }
        if apu + size < end {
            self.free.insert(apu + size, end);
        }
        Some(apu)
    }

    fn give_back(&mut self, mut start: u32, mut end: u32) {
        if let Some((&before, &before_end)) = self.free.range(..start).next_back()
            && before_end == start
        {
            self.free.remove(&before);
            start = before;
        }
        if let Some(after_end) = self.free.remove(&end) {
            end = after_end;
        }
        self.free.insert(start, end);
    }
}

/// Zeroed host memory and its device address.
fn heap_alloc(size: usize, align: usize) -> Option<(*mut u8, u32)> {
    let layout = Layout::from_size_align(size.max(1), align.max(16)).ok()?;
    let mut heap = lock(&HEAP);
    let apu = heap.take(u32::try_from(layout.size()).ok()?, align as u32)?;
    // SAFETY: non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        heap.give_back(apu, apu + layout.size() as u32);
        return None;
    }
    let alloc = Arc::new(Allocation { ptr, layout });
    heap.blocks.insert(ptr as usize, Block { size, apu, alloc });
    heap.by_apu.insert(apu, ptr as usize);
    // The big ones are named in the log: they are the title's audio memory
    // given out of our allocator, which the memory line counts as ours.
    if size >= 32 << 20 {
        tracing::info!(mb = size >> 20, at = ?ptr, "audio memory for the title from our allocator");
    }
    Some((ptr, apu))
}

fn heap_free(ptr: *mut u8) {
    let mut heap = lock(&HEAP);
    let Some(block) = heap.blocks.remove(&(ptr as usize)) else {
        return;
    };
    heap.by_apu.remove(&block.apu);
    heap.give_back(block.apu, block.apu + block.alloc.layout.size() as u32);
    drop(heap);
    drop(block);
}

/// Sizes the heap once, before anything is allocated from it. The console
/// reserves both parts up front; we allocate per block, so the sizes only
/// decide whether the call is valid.
// shortcut: the cached and non-cached sizes are not enforced as limits;
// enforce them if a title relies on running out.
pub extern "system" fn ApuCreateHeap(cached: u32, _non_cached: u32) -> HRESULT {
    const ALREADY_ALLOCATED: HRESULT = 0x8ACA_0001_u32 as i32;
    if cached == 0 {
        return E_INVALIDARG;
    }
    let mut heap = lock(&HEAP);
    if heap.created {
        return ALREADY_ALLOCATED;
    }
    heap.created = true;
    heap.free.insert(APU_BASE, u32::MAX);
    S_OK
}

/// The host address of a device address, null outside the heap.
pub extern "system" fn ApuMapApuAddress(apu: u32) -> *mut c_void {
    heap_unmap(apu).map_or(std::ptr::null_mut(), |p| p.cast())
}

/// The device address of any byte of a heap block, 0 elsewhere.
fn heap_map(ptr: usize) -> u32 {
    let heap = lock(&HEAP);
    match heap.blocks.range(..=ptr).next_back() {
        Some((&start, block)) if ptr < start + block.size.max(1) => {
            block.apu + (ptr - start) as u32
        }
        _ => 0,
    }
}

/// The host address of a device address, if a heap block holds it.
fn heap_unmap(apu: u32) -> Option<*mut u8> {
    let heap = lock(&HEAP);
    let (&start, &host) = heap.by_apu.range(..=apu).next_back()?;
    let block = &heap.blocks[&host];
    (apu < start.saturating_add(block.size.max(1) as u32))
        .then(|| (host + (apu - start) as usize) as *mut u8)
}

/// `len` bytes of title memory at a device address, if one heap block holds
/// them all. Retains the block's backing allocation until dropped.
fn heap_bytes(apu: u32, len: usize) -> Option<HeapSlice> {
    let heap = lock(&HEAP);
    let (&start, &host) = heap.by_apu.range(..=apu).next_back()?;
    let block = &heap.blocks[&host];
    let offset = (apu - start) as usize;
    offset
        .checked_add(len)
        .is_some_and(|end| end <= block.size)
        .then(|| HeapSlice {
            _alloc: Arc::clone(&block.alloc),
            ptr: (host + offset) as *mut u8,
            len,
        })
}

pub extern "system" fn ApuAlloc(
    out: *mut *mut c_void,
    apu: *mut u32,
    size: u32,
    align: u32,
    _flags: u32,
) -> HRESULT {
    if out.is_null() || size == 0 || align < 4 || !align.is_power_of_two() {
        return E_INVALIDARG;
    }
    let Some((ptr, address)) = heap_alloc(size as usize, align as usize) else {
        return E_OUTOFMEMORY;
    };
    // SAFETY: checked non-null; optional out parameter checked below.
    unsafe {
        out.write(ptr.cast());
        if !apu.is_null() {
            apu.write(address);
        }
    }
    S_OK
}

pub extern "system" fn ApuFree(ptr: *mut c_void) {
    heap_free(ptr.cast());
}

pub extern "system" fn ApuMapVirtualAddress(ptr: *const c_void) -> u32 {
    heap_map(ptr as usize)
}

pub extern "system" fn ApuIsVirtualAddressValid(ptr: *const c_void, align: u32) -> bool {
    let apu = heap_map(ptr as usize);
    apu != 0 && (align == 0 || apu.is_multiple_of(align))
}

// SHAPE context arrays.

/// Six counts (rate conversion, EQ/compression, filter/volume, DMA, XMA,
/// PCM), then the six arrays and their device addresses.
#[repr(C, packed)]
pub struct ShapeContexts {
    counts: [u32; 6],
    arrays: [*mut c_void; 6],
    apu: [u32; 6],
}

/// Bytes per context: rate conversion, DMA and XMA as fmod clears them;
/// the others (not seen yet) get a generous upper bound.
const CONTEXT_BYTES: [usize; 6] = [32, 256, 256, 16, 64, 256];

static CONTEXTS: Mutex<Option<[usize; 6]>> = Mutex::new(None);

pub extern "system" fn AcpHalAllocateShapeContexts(contexts: *mut ShapeContexts) -> HRESULT {
    if contexts.is_null() {
        return E_INVALIDARG;
    }
    let mut allocated = lock(&CONTEXTS);
    if allocated.is_some() {
        return E_FAIL;
    }
    // SAFETY: the title passes its request; packed fields are copied out.
    let counts = unsafe { std::ptr::addr_of!((*contexts).counts).read_unaligned() };
    let mut arrays: [*mut c_void; 6] = [std::ptr::null_mut(); 6];
    let mut addresses = [0u32; 6];
    for i in 0..6 {
        if counts[i] == 0 {
            continue;
        }
        let Some((ptr, apu)) = heap_alloc(counts[i] as usize * CONTEXT_BYTES[i], 256) else {
            arrays.iter().for_each(|&p| heap_free(p.cast()));
            return E_OUTOFMEMORY;
        };
        (arrays[i], addresses[i]) = (ptr.cast(), apu);
    }
    // SAFETY: as above.
    unsafe {
        std::ptr::addr_of_mut!((*contexts).arrays).write_unaligned(arrays);
        std::ptr::addr_of_mut!((*contexts).apu).write_unaligned(addresses);
    }
    *allocated = Some(arrays.map(|p| p as usize));
    *XMA.write().unwrap_or_else(|e| e.into_inner()) =
        Xma::new(arrays[4] as usize, counts[4] as usize);
    S_OK
}

pub extern "system" fn AcpHalReleaseShapeContexts() -> HRESULT {
    match lock(&CONTEXTS).take() {
        Some(arrays) => {
            *XMA.write().unwrap_or_else(|e| e.into_inner()) = Xma::EMPTY;
            arrays.iter().for_each(|&p| heap_free(p as *mut u8));
            S_OK
        }
        None => E_FAIL,
    }
}

// The XMA decoder.
//
// An XMA context (16 words in the shared context array) names up to two
// input buffers of 2 KiB packets, a ring of 256-byte output blocks the
// hardware fills with 16-bit PCM, and the read offset the title advances
// by command. Words and bit fields, as titles write them:
// 0: input 0 size in packets (0-11), loop count (12-19, 255 forever),
//    valid inputs (20-21), output size in blocks (22-26), write block
//    (27-31, hardware);
// 1: input 1 size (0-11), loop end subframe (12-13), loop start subframes
//    to skip (17-19), subframes to skip at the start (24-26), sample rate
//    (27-28: 24, 32, 44.1, 48 kHz), stereo (29), output writable (31);
// 2: read position in bits (0-25), error status (26-30), error (31);
// 3: loop start in bits (0-25); 4: loop end in bits (0-25), current input
//    (31); 5, 6: input addresses; 7: output address; 8: overlap buffer;
//    9: read block (0-4).
// Titles read back only the write block, "writable" and the read block,
// and take whatever lies between read and write.
// shortcut: each enabled context decodes from a cold state and the overlap
// buffer is ignored (titles copy it between consecutive contexts of one
// voice); keep decoder state there if a title's seams turn out audible.

use victory_formats::xma;

const ENABLE_XMA: u32 = 5;
const ENABLE_XMAS: u32 = 6;
const DISABLE_XMA: u32 = 7;
const DISABLE_XMAS: u32 = 8;
const ADVANCE_XMA_READ: u32 = 18;

const BLOCK_BYTES: usize = 256;
const SAMPLES_PER_BLOCK: usize = BLOCK_BYTES / 2;
/// Error status: the input ran out.
const INPUT_INVALID: u32 = 4;
/// Decoded samples a voice holds at most, for a moment before they go into
/// the ring: the ring's free blocks (at most 31, the size field has 5 bits)
/// plus one stereo frame, rounded up. Between steps it holds less than a
/// block plus the rest of one frame.
const PCM_CAPACITY: usize = 32 * SAMPLES_PER_BLOCK + 2 * xma::FRAME_SAMPLES * 2;
/// A voice that could not get the samples its ring has room for is looked
/// at again this soon: a 256-byte output block lasts 2.7 ms at 48 kHz mono.
const WAITING_POLL: Duration = Duration::from_millis(2);
/// Every enabled voice is looked at this often, signalled or not, in case
/// the title changed its context words without a command.
// shortcut: the sweep is now the only thing that notices a writable bit or
// new input the title set by writing the context without a command (the
// old decoder looked at every context every 2 ms), so such a voice may
// start up to 20 ms late; shorten the sweep if a title relies on it.
const SWEEP: Duration = Duration::from_millis(20);

/// A voice's decoding side, held by the decoder thread for a whole step and
/// taken by enable and disable, never by the title's mixer commands. Its
/// buffers are kept from one enable to the next, so a running voice
/// allocates nothing.
struct Decoding {
    active: bool,
    /// Changed by every enable and disable; the output side carries the
    /// same number while it belongs to the same voice.
    generation: u32,
    decoder: Option<Box<xma::Decoder>>,
    /// An earlier voice's decoder, set up again for this voice's first frame.
    spare: Option<Box<xma::Decoder>>,
    /// Decoded samples not in the ring yet.
    pcm: VecDeque<i16>,
    /// Samples per channel still to drop (start or loop subframe skips).
    skip: usize,
    ended: bool,
    /// The current frame's bits, and its samples.
    frame: Vec<u8>,
    samples: Vec<i16>,
    since: Instant,
    frames: u32,
    bad_frames: u32,
    /// Milliseconds after enabling when the input ran out.
    ended_ms: Option<u128>,
    /// The longest a step decoded for, in microseconds.
    max_decode_us: u64,
}

impl Decoding {
    fn new() -> Decoding {
        Decoding {
            active: false,
            generation: 0,
            decoder: None,
            spare: None,
            pcm: VecDeque::new(),
            skip: 0,
            ended: false,
            frame: Vec::new(),
            samples: Vec::new(),
            since: Instant::now(),
            frames: 0,
            bad_frames: 0,
            ended_ms: None,
            max_decode_us: 0,
        }
    }

    /// Starts a new voice from a cold decoder, keeping the buffers.
    fn start(&mut self, skip: usize) {
        self.active = true;
        self.generation = self.generation.wrapping_add(1);
        if let Some(decoder) = self.decoder.take() {
            self.spare = Some(decoder);
        }
        self.pcm.clear();
        self.pcm.reserve(PCM_CAPACITY);
        self.skip = skip;
        self.ended = false;
        self.frame.clear();
        self.frame.reserve(xma::MAX_FRAME_BYTES);
        self.samples.clear();
        self.samples.reserve(xma::FRAME_SAMPLES * 2);
        self.since = Instant::now();
        self.frames = 0;
        self.bad_frames = 0;
        self.ended_ms = None;
        self.max_decode_us = 0;
    }
}

/// A voice's output side: what the title's commands touch. Taken briefly,
/// by `advance_read` and by the decoder thread to copy blocks into the ring.
struct Output {
    active: bool,
    generation: u32,
    /// Samples per channel written to the ring.
    produced: usize,
    /// The loudest sample written.
    peak: i16,
    /// Blocks the title took (command 18), how often, and how often it
    /// found the ring less than full.
    taken: usize,
    takes: u32,
    partial_takes: u32,
    /// The longest the decoder thread held this lock, in microseconds.
    max_fill_us: u64,
}

impl Output {
    fn new(active: bool, generation: u32) -> Output {
        Output {
            active,
            generation,
            produced: 0,
            peak: 0,
            taken: 0,
            takes: 0,
            partial_takes: 0,
            max_fill_us: 0,
        }
    }
}

/// One context. The lock order is `decoding`, then `output`.
struct Slot {
    decoding: Mutex<Decoding>,
    output: Mutex<Output>,
}

impl Slot {
    fn new() -> Slot {
        Slot {
            decoding: Mutex::new(Decoding::new()),
            output: Mutex::new(Output::new(false, 0)),
        }
    }
}

struct Xma {
    /// Host address of the context array.
    array: usize,
    /// One entry per context. A command for one voice never waits for
    /// another to decode, and `advance_read` (sent from the title's
    /// time-critical mixer thread) never waits for its own voice either.
    slots: Vec<Slot>,
    /// One bit per context: enabled or read from since the decoder thread
    /// last looked.
    pending: Vec<AtomicU64>,
    /// One bit per enabled context, for the sweep.
    enabled: Vec<AtomicU64>,
}

impl Xma {
    const EMPTY: Xma = Xma {
        array: 0,
        slots: Vec::new(),
        pending: Vec::new(),
        enabled: Vec::new(),
    };

    fn new(array: usize, count: usize) -> Xma {
        let bits =
            || -> Vec<AtomicU64> { (0..count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect() };
        Xma {
            array,
            slots: (0..count).map(|_| Slot::new()).collect(),
            pending: bits(),
            enabled: bits(),
        }
    }
}

/// Written only when the context arrays are allocated or released.
static XMA: RwLock<Xma> = RwLock::new(Xma::EMPTY);

fn xma() -> RwLockReadGuard<'static, Xma> {
    XMA.read().unwrap_or_else(|e| e.into_inner())
}

fn set_bit(words: &[AtomicU64], index: usize, on: bool) {
    let (word, bit) = (&words[index / 64], 1u64 << (index % 64));
    if on {
        word.fetch_or(bit, Ordering::Release);
    } else {
        word.fetch_and(!bit, Ordering::Release);
    }
}

/// A context's words in title memory.
#[derive(Clone, Copy)]
struct Context(*mut u32);

impl Context {
    fn get(self, word: usize) -> u32 {
        // SAFETY: inside the context array; the title reads and writes it
        // concurrently, like the hardware's.
        unsafe { self.0.add(word).read_volatile() }
    }

    fn set(self, word: usize, value: u32) {
        // SAFETY: as above.
        unsafe { self.0.add(word).write_volatile(value) }
    }

    fn field(self, word: usize, shift: u32, bits: u32) -> u32 {
        (self.get(word) >> shift) & ((1 << bits) - 1)
    }

    fn set_field(self, word: usize, shift: u32, bits: u32, value: u32) {
        let mask = ((1u32 << bits) - 1) << shift;
        self.set(word, (self.get(word) & !mask) | ((value << shift) & mask));
    }

    fn channels(self) -> usize {
        1 + self.field(1, 29, 1) as usize
    }

    fn writable(self) -> bool {
        self.field(1, 31, 1) == 1
    }

    /// The output blocks the title has left room for: none while the ring is
    /// not writable, all of them when the write block is on the read block
    /// of a ring that is.
    fn free_blocks(self, size: usize) -> usize {
        if !self.writable() {
            return 0;
        }
        let (write, read) = (self.field(0, 27, 5) as usize, self.field(9, 0, 5) as usize);
        match (read % size + size - write % size) % size {
            0 => size,
            n => n,
        }
    }
}

/// What a step left a voice waiting for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Step {
    /// Nothing until the title sends a command: disabled, done, or its ring
    /// is full.
    Idle,
    /// Its ring has room the decoder could not fill (no input yet, or only
    /// frames that decode to nothing): looked at again soon.
    Waiting,
}

/// Why decoding ahead stopped.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stop {
    /// Enough samples for the blocks the ring has room for.
    Full,
    /// Out of input while the ring did not need samples yet.
    Input,
    /// Frames that decoded to nothing, one after another.
    Empty,
    /// The input ran out: nothing more to decode.
    Ended,
}

/// One pass of the decoder thread.
#[derive(Default)]
struct Pass {
    stepped: u32,
    waiting: u32,
}

impl Xma {
    fn context(&self, index: u32) -> Option<Context> {
        ((index as usize) < self.slots.len() && self.array != 0)
            .then(|| Context((self.array + index as usize * 64) as *mut u32))
    }

    /// Marks a voice for the decoder thread's next pass; the caller wakes
    /// the thread.
    fn signal(&self, index: u32) {
        set_bit(&self.pending, index as usize, true);
    }

    fn enable(&self, index: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        let slot = &self.slots[index as usize];
        let mut voice = lock(&slot.decoding);
        let mut output = lock(&slot.output);
        context.set_field(2, 26, 6, 0);
        tracing::debug!(
            index,
            words = ?(0..10).map(|w| format!("{:08x}", context.get(w))).collect::<Vec<_>>(),
            "XMA voice enabled"
        );
        #[cfg(feature = "diagnostics")]
        dump_input(index, context);
        voice.start(context.field(1, 24, 3) as usize * xma::SUBFRAME_SAMPLES);
        *output = Output::new(true, voice.generation);
        set_bit(&self.enabled, index as usize, true);
        drop(output);
        drop(voice);
        self.signal(index);
    }

    fn disable(&self, index: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        let slot = &self.slots[index as usize];
        let mut voice = lock(&slot.decoding);
        if !voice.active {
            return;
        }
        voice.active = false;
        voice.generation = voice.generation.wrapping_add(1);
        let o = std::mem::replace(
            &mut *lock(&slot.output),
            Output::new(false, voice.generation),
        );
        set_bit(&self.enabled, index as usize, false);
        tracing::debug!(
            index,
            ms = voice.since.elapsed().as_millis(),
            produced = o.produced,
            taken = o.taken * SAMPLES_PER_BLOCK / context.channels(),
            takes = o.takes,
            partial_takes = o.partial_takes,
            frames = voice.frames,
            bad_frames = voice.bad_frames,
            peak = o.peak,
            input_ended_ms = ?voice.ended_ms,
            max_decode_us = voice.max_decode_us,
            max_fill_us = o.max_fill_us,
            read_bit = context.field(2, 0, 26),
            input_bits = context.field(0, 0, 12) * xma::PACKET_BYTES as u32 * 8,
            "XMA voice disabled"
        );
    }

    /// Moves the title's read block on, which frees output space. Takes
    /// only the output lock: a decode of the same voice does not hold it.
    fn advance_read(&self, index: u32, blocks: u32) {
        let Some(context) = self.context(index) else {
            return;
        };
        // The decoder thread writes the same words under the same lock.
        let mut output = lock(&self.slots[index as usize].output);
        let size = context.field(0, 22, 5).max(1);
        // The count is the title's: reduced first, so the sum stays small.
        let read = (context.field(9, 0, 5) + blocks % size) % size;
        context.set_field(9, 0, 5, read);
        if blocks > 0 {
            context.set_field(1, 31, 1, 1);
        }
        if output.active {
            output.taken += blocks as usize;
            output.takes += 1;
            output.partial_takes += u32::from(blocks < size);
        }
        drop(output);
        self.signal(index);
    }

    /// Steps the voices signalled since the last pass and those left
    /// waiting by it, plus every enabled voice when `sweep` is set.
    /// `waiting` is the decoder thread's own bitmask of waiting voices.
    fn pass(&self, waiting: &mut Vec<u64>, sweep: bool) -> Pass {
        let start = Instant::now();
        waiting.resize(self.pending.len(), 0);
        let mut pass = Pass::default();
        for (w, word) in self.pending.iter().enumerate() {
            let mut bits = word.swap(0, Ordering::Acquire) | std::mem::take(&mut waiting[w]);
            if sweep {
                bits |= self.enabled[w].load(Ordering::Acquire);
            }
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                pass.stepped += 1;
                if self.step(w * 64 + bit as usize) == Step::Waiting {
                    waiting[w] |= 1 << bit;
                    pass.waiting += 1;
                }
            }
        }
        let took = start.elapsed();
        if took.as_millis() >= 4 {
            tracing::debug!(
                stepped = pass.stepped,
                waiting = pass.waiting,
                ms = took.as_millis(),
                "slow XMA decoding pass"
            );
        }
        pass
    }

    /// Decodes ahead for one voice and fills its ring as far as it has room.
    fn step(&self, index: usize) -> Step {
        let Some(slot) = self.slots.get(index) else {
            return Step::Idle;
        };
        #[cfg(test)]
        tests::STEPPED.with_borrow_mut(|stepped| {
            // The decoding thread records its steps too; keep its list short.
            if stepped.len() < 1024 {
                stepped.push(index);
            }
        });
        let Some(context) = self.context(index as u32) else {
            return Step::Idle;
        };
        let mut voice = lock(&slot.decoding);
        if !voice.active {
            return Step::Idle;
        }
        fill(context, &slot.output, &mut voice)
    }

    /// Every voice, one after another, as the decoder thread would.
    #[cfg(test)]
    fn step_all(&self) {
        for index in 0..self.slots.len() {
            self.step(index);
        }
    }
}

/// Decodes into `voice.pcm` what the ring has room for, then copies whole
/// blocks into the ring under the output lock only.
fn fill(context: Context, output: &Mutex<Output>, voice: &mut Decoding) -> Step {
    let size = context.field(0, 22, 5) as usize;
    if size == 0 {
        return Step::Waiting;
    }
    {
        // The output side must belong to this voice before decoding moves
        // the context words on. It cannot change while the decoding lock is
        // held (enable and disable take that first); this is a safety net.
        let out = lock(output);
        if !out.active || out.generation != voice.generation {
            return Step::Idle;
        }
    }
    let mut stop = decode_ahead(context, voice, size);
    loop {
        let Some(writable) = publish(context, output, voice, size) else {
            return Step::Waiting;
        };
        let starving = writable && voice.pcm.len() < SAMPLES_PER_BLOCK;
        if matches!(stop, Stop::Input | Stop::Full) && starving {
            // The ring needs samples now (the title read on while this step
            // ran, or the input may end the voice): decode for it at once
            // rather than at the next poll.
            stop = decode_ahead(context, voice, size);
            continue;
        }
        return match stop {
            Stop::Ended if voice.pcm.is_empty() => Step::Idle,
            _ if !writable => Step::Idle,
            _ if starving => Step::Waiting,
            _ => Step::Idle,
        };
    }
}

/// Decodes frames until the voice holds samples for every block the title
/// has left room for, so the context words the title sees (read position,
/// valid inputs, current input, loop count) move on no earlier than the
/// hardware's would: at most one frame ahead of the ring. The input may
/// run out (which ends the voice and sets its error status) only when the
/// ring has room and less than a block is decoded.
fn decode_ahead(context: Context, voice: &mut Decoding, size: usize) -> Stop {
    let start = Instant::now();
    let mut empty_steps = 0u32;
    let stop = loop {
        // The read block is read without the output lock: a stale value
        // only makes the room look smaller.
        if voice.pcm.len() >= context.free_blocks(size) * SAMPLES_PER_BLOCK {
            break Stop::Full;
        }
        let may_end = context.writable() && voice.pcm.len() < SAMPLES_PER_BLOCK;
        let before = voice.pcm.len();
        if !decode(context, voice, may_end) {
            break if voice.ended {
                Stop::Ended
            } else {
                Stop::Input
            };
        }
        if voice.pcm.len() == before {
            empty_steps += 1;
            if empty_steps > 16 {
                break Stop::Empty;
            }
        } else {
            empty_steps = 0;
        }
    };
    voice.max_decode_us = voice.max_decode_us.max(start.elapsed().as_micros() as u64);
    stop
}

/// Copies decoded blocks into the ring while it is writable; the last,
/// partial block of an ended voice is padded with silence. Whether the
/// ring is still writable, or `None` when the ring could not be reached or
/// the output side belongs to another voice.
fn publish(
    context: Context,
    output: &Mutex<Output>,
    voice: &mut Decoding,
    size: usize,
) -> Option<bool> {
    let start = Instant::now();
    let mut out = lock(output);
    // A safety net: `fill` checked this before decoding.
    if !out.active || out.generation != voice.generation {
        return None;
    }
    let size = size as u32;
    let mut reached = true;
    while context.writable()
        && (voice.pcm.len() >= SAMPLES_PER_BLOCK || (voice.ended && !voice.pcm.is_empty()))
    {
        let write = context.field(0, 27, 5);
        let address = context.get(7) as usize + write as usize * BLOCK_BYTES;
        let Some(mut ring) = heap_bytes(address as u32, BLOCK_BYTES) else {
            reached = false;
            break;
        };
        for pair in ring.as_chunks_mut::<2>().0 {
            let sample = voice.pcm.pop_front().unwrap_or(0);
            out.peak = out.peak.max(sample.saturating_abs());
            pair.copy_from_slice(&sample.to_le_bytes());
        }
        out.produced += SAMPLES_PER_BLOCK / context.channels();
        let write = (write + 1) % size;
        context.set_field(0, 27, 5, write);
        if write == context.field(9, 0, 5) {
            // Full: nothing more until the title advances its read block.
            context.set_field(1, 31, 1, 0);
        }
    }
    let writable = context.writable();
    out.max_fill_us = out.max_fill_us.max(start.elapsed().as_micros() as u64);
    reached.then_some(writable)
}

/// Decodes the next frame of the current input into `voice.pcm`; false
/// when there is nothing left to decode, or when the input is used up and
/// `may_end` is not set (the voice then stays as it is).
fn decode(context: Context, voice: &mut Decoding, may_end: bool) -> bool {
    if voice.ended {
        return false;
    }
    let channels = context.channels();
    let mut current = context.field(4, 31, 1);
    let valid = context.field(0, 20, 2);
    if valid & (1 << current) == 0 {
        if valid & (1 << (1 - current)) == 0 {
            if !may_end {
                return false;
            }
            // Out of input: the overlap tail, then nothing.
            if let Some(decoder) = &mut voice.decoder {
                voice.samples.clear();
                decoder.flush(&mut voice.samples);
                voice.pcm.extend(&voice.samples);
            }
            voice.ended = true;
            voice.ended_ms = Some(voice.since.elapsed().as_millis());
            context.set_field(2, 26, 5, INPUT_INVALID);
            context.set_field(2, 31, 1, 1);
            return !voice.pcm.is_empty();
        }
        current = 1 - current;
        context.set_field(4, 31, 1, current);
        context.set_field(2, 0, 26, 0);
    }
    let (address, packets) = match current {
        0 => (context.get(5), context.field(0, 0, 12)),
        _ => (context.get(6), context.field(1, 0, 12)),
    };
    let Some(data) = heap_bytes(address & !0x7ff, packets as usize * xma::PACKET_BYTES) else {
        context.set_field(0, 20 + current, 1, 0);
        return true;
    };
    let mut at = context.field(2, 0, 26) as usize;
    if at < 32 {
        at = 32 + xma::packet(&data, 0).map_or(0, |p| p.first_frame_bit as usize);
    }
    let Some(frame) = xma::read_frame_into(&data, at, &mut voice.frame) else {
        // This input is used up.
        context.set_field(0, 20 + current, 1, 0);
        return true;
    };
    let spare = &mut voice.spare;
    let decoder = voice.decoder.get_or_insert_with(|| {
        const RATES: [u32; 4] = [24_000, 32_000, 44_100, 48_000];
        let fresh = xma::Decoder::new(channels, RATES[context.field(1, 27, 2) as usize]);
        match spare.take() {
            Some(mut decoder) => {
                *decoder = fresh;
                decoder
            }
            None => Box::new(fresh),
        }
    });
    let pcm = &mut voice.samples;
    pcm.clear();
    voice.frames += 1;
    if let Err(e) = decoder.decode(&voice.frame, frame.bits, pcm) {
        voice.bad_frames += 1;
        if voice.bad_frames <= 2 {
            tracing::debug!(
                at,
                bits = frame.bits,
                head = ?&voice.frame[..voice.frame.len().min(8)],
                %e,
                "XMA frame did not decode"
            );
        }
    }
    let mut next = frame.next.unwrap_or(0);
    let loops = context.field(0, 12, 8);
    let loop_end = context.field(4, 0, 26) as usize;
    let ends_loop = loop_end != 0
        && (frame.start == loop_end
            || (frame.start < loop_end && frame.next.is_none_or(|n| loop_end < n)));
    let mut next_skip = None;
    if loops > 0 && ends_loop {
        let keep = (context.field(1, 12, 2) as usize + 1) * xma::SUBFRAME_SAMPLES * channels;
        pcm.truncate(keep);
        next = context.field(3, 0, 26) as usize;
        next_skip = Some(context.field(1, 17, 3) as usize * xma::SUBFRAME_SAMPLES);
        if loops != 255 {
            context.set_field(0, 12, 8, loops - 1);
        }
    }
    let skip = (voice.skip * channels).min(pcm.len());
    voice.skip -= skip / channels;
    voice.pcm.extend(&pcm[skip..]);
    if let Some(skip) = next_skip {
        voice.skip = skip;
    }
    if next == 0 {
        context.set_field(0, 20 + current, 1, 0);
    }
    context.set_field(2, 0, 26, next as u32);
    true
}

/// `VICTORY_XMA_DUMP=<dir>`: saves the context words (`<n>-<index>.ctx`)
/// and input buffer 0 (`.xma`) of the first 32 voices enabled, to test
/// the decoder on a title's streams outside the title.
#[cfg(feature = "diagnostics")]
fn dump_input(index: u32, context: Context) {
    static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let Some(dir) = std::env::var_os("VICTORY_XMA_DUMP") else {
        return;
    };
    let n = COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if n >= 32 {
        return;
    }
    let dir = std::path::PathBuf::from(dir);
    let words: Vec<u8> = (0..16).flat_map(|w| context.get(w).to_le_bytes()).collect();
    let input = heap_bytes(
        context.get(5) & !0x7ff,
        context.field(0, 0, 12) as usize * xma::PACKET_BYTES,
    );
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(dir.join(format!("{n:02}-{index}.ctx")), words);
    let _ = std::fs::write(
        dir.join(format!("{n:02}-{index}.xma")),
        input.as_deref().unwrap_or(&[]),
    );
}

/// Starts the decoding thread on the first enabled voice, and has it look
/// at the voices signalled since its last pass now.
fn wake_decoder() {
    static DECODER: OnceLock<std::thread::Thread> = OnceLock::new();
    DECODER
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("victory XMA decoder".into())
                .spawn(decoder_thread)
                .map_or_else(|_| std::thread::current(), |h| h.thread().clone())
        })
        .unpark();
}

/// Steps signalled voices as they come, waiting ones every
/// [`WAITING_POLL`] and all enabled ones every [`SWEEP`]. A pass with
/// nothing to do reads one word per 64 contexts.
fn decoder_thread() {
    let mut waiting = Vec::new();
    let mut last_sweep = Instant::now();
    let (mut since, mut passes, mut stepped, mut busy) =
        (Instant::now(), 0u32, 0u64, Duration::ZERO);
    loop {
        let sweep = last_sweep.elapsed() >= SWEEP;
        if sweep {
            last_sweep = Instant::now();
        }
        let start = Instant::now();
        // The guard is dropped before parking, so allocating or releasing
        // the context arrays never waits for a timeout.
        let pass = xma().pass(&mut waiting, sweep);
        busy += start.elapsed();
        passes += 1;
        stepped += u64::from(pass.stepped);
        if since.elapsed().as_secs() >= 10 {
            tracing::debug!(
                passes,
                stepped,
                waiting = pass.waiting,
                busy_ms = busy.as_millis(),
                "XMA decoder in the last 10 s"
            );
            (since, passes, stepped, busy) = (Instant::now(), 0, 0, Duration::ZERO);
        }
        let timeout = match pass.waiting {
            0 => SWEEP.saturating_sub(last_sweep.elapsed()),
            _ => WAITING_POLL,
        };
        std::thread::park_timeout(timeout);
    }
}

/// Runs an XMA command; false for one this does not know.
fn xma_command(command: u32, word: impl Fn(usize) -> Option<u32>) -> bool {
    let xma = xma();
    let wake = match command {
        ENABLE_XMA => {
            xma.enable(word(0).unwrap_or(u32::MAX));
            true
        }
        DISABLE_XMA => {
            xma.disable(word(0).unwrap_or(u32::MAX));
            false
        }
        ENABLE_XMAS | DISABLE_XMAS => {
            for i in 0..xma.slots.len() as u32 {
                if word(i as usize / 32).unwrap_or(0) & (1 << (i % 32)) != 0 {
                    match command {
                        ENABLE_XMAS => xma.enable(i),
                        _ => xma.disable(i),
                    }
                }
            }
            command == ENABLE_XMAS
        }
        ADVANCE_XMA_READ => {
            xma.advance_read(word(0).unwrap_or(u32::MAX), word(1).unwrap_or(0));
            true
        }
        _ => return false,
    };
    drop(xma);
    // One wake for a whole mask of voices.
    if wake {
        wake_decoder();
    }
    true
}

// The audio coprocessor.

const LOAD_FLOWGRAPH: u32 = 1;
const REGISTER_MESSAGES: u32 = 2;
const UNREGISTER_MESSAGES: u32 = 3;

const FLOWGRAPH_COMPLETED: u32 = 0x02;
const COMMAND_COMPLETED: u32 = 0x10;
const DISCONNECTED: u32 = 0x80;

/// Audio frames of 128 samples at 48 kHz.
const FRAMES_PER_SECOND: u64 = 375;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Message {
    kind: u32,
    dropped: u32,
    usec: u32,
    data: [u8; 16],
}

struct State {
    connected: bool,
    /// Message kinds the title registered for.
    mask: u32,
    messages: VecDeque<Message>,
    /// Queue size given to `connect`.
    capacity: usize,
    /// Messages lost to a full queue since the last one queued.
    dropped: u32,
    since: Instant,
}

impl State {
    fn usec(&self) -> u64 {
        self.since.elapsed().as_micros() as u64
    }

    fn frame(&self) -> u32 {
        (self.usec() * FRAMES_PER_SECOND / 1_000_000) as u32
    }

    /// Queues a registered message; a full queue drops it and counts it in
    /// the next one that fits.
    fn post(&mut self, kind: u32, data: &[u8]) {
        if self.mask & kind == 0 {
            return;
        }
        if self.messages.len() >= self.capacity {
            self.dropped = self.dropped.saturating_add(1);
            tracing::debug!(kind, dropped = self.dropped, "coprocessor message dropped");
            return;
        }
        let mut message = Message {
            kind,
            dropped: std::mem::take(&mut self.dropped),
            usec: self.usec() as u32,
            data: [0; 16],
        };
        message.data[..data.len()].copy_from_slice(data);
        self.messages.push_back(message);
    }
}

/// The interface object: its method table, then our state.
#[repr(C)]
pub struct Acp {
    vtable: &'static AcpMethods,
    state: Mutex<State>,
}

#[repr(C)]
struct AcpMethods {
    connect: extern "system" fn(*mut Acp, u32, u32) -> HRESULT,
    disconnect: extern "system" fn(*mut Acp) -> HRESULT,
    submit: extern "system" fn(*mut Acp, u32, u64, u32, *const c_void, u32) -> HRESULT,
    pop_message: extern "system" fn(*mut Acp, *mut Message) -> bool,
    message_count: extern "system" fn(*mut Acp) -> u32,
    release: extern "system" fn(*mut Acp),
}

static ACP_METHODS: AcpMethods = AcpMethods {
    connect,
    disconnect,
    submit,
    pop_message,
    message_count,
    release,
};

fn state(this: *mut Acp) -> MutexGuard<'static, State> {
    // SAFETY: title code calls methods on the object `AcpHalCreate` made,
    // which lives until `release`.
    lock(unsafe { &(*this).state })
}

extern "system" fn connect(this: *mut Acp, commands: u32, messages: u32) -> HRESULT {
    if commands == 0 {
        return E_INVALIDARG;
    }
    let mut state = state(this);
    state.connected = true;
    state.capacity = messages as usize;
    S_OK
}

extern "system" fn disconnect(this: *mut Acp) -> HRESULT {
    let mut state = state(this);
    if !state.connected {
        return E_FAIL;
    }
    state.post(DISCONNECTED, &[]);
    state.connected = false;
    S_OK
}

/// Done at once: message registrations take effect, a flowgraph completes
/// unprocessed, a command with an id is acknowledged, and its notification
/// word is set.
extern "system" fn submit(
    this: *mut Acp,
    command: u32,
    id: u64,
    _frame: u32,
    data: *const c_void,
    notification: u32,
) -> HRESULT {
    static SEEN: Mutex<u64> = Mutex::new(0);
    if command < 64 && *lock(&SEEN) & 1 << command == 0 {
        *lock(&SEEN) |= 1 << command;
        tracing::info!(command, "first coprocessor command of this kind");
    }
    let mut state = state(this);
    let word = |at: usize| -> Option<u32> {
        // SAFETY: the command's data starts with these u32 fields.
        (!data.is_null()).then(|| unsafe { data.cast::<u32>().add(at).read_unaligned() })
    };
    match command {
        REGISTER_MESSAGES => state.mask |= word(0).unwrap_or(0),
        UNREGISTER_MESSAGES => state.mask &= !word(0).unwrap_or(0),
        LOAD_FLOWGRAPH => {
            let flowgraph = word(1).unwrap_or(0);
            state.post(FLOWGRAPH_COMPLETED, &flowgraph.to_le_bytes());
        }
        _ if xma_command(command, word) => {}
        // shortcut: context updates and the other SHAPE units' commands are
        // acknowledged but not carried out; add them when a title sends
        // them (each kind is logged once above).
        _ => {}
    }
    if id != 0 {
        let mut done = [0u8; 16];
        done[..4].copy_from_slice(&command.to_le_bytes());
        done[4..12].copy_from_slice(&id.to_le_bytes());
        done[12..].copy_from_slice(&state.frame().to_le_bytes());
        state.post(COMMAND_COMPLETED, &done);
    }
    if notification != 0
        && let Some(word) = heap_bytes(notification, 4)
    {
        // SAFETY: 4-byte slice in a retained heap block, written volatile for the title.
        unsafe { word.ptr.cast::<u32>().write_volatile(1) };
    }
    S_OK
}

extern "system" fn pop_message(this: *mut Acp, out: *mut Message) -> bool {
    let Some(message) = state(this).messages.pop_front() else {
        return false;
    };
    if !out.is_null() {
        // SAFETY: the title passes room for one message.
        unsafe { out.write_unaligned(message) };
    }
    true
}

extern "system" fn message_count(this: *mut Acp) -> u32 {
    state(this).messages.len() as u32
}

extern "system" fn release(this: *mut Acp) {
    // SAFETY: made by `AcpHalCreate` with `Box::into_raw`; released once.
    drop(unsafe { Box::from_raw(this) });
}

pub extern "system" fn AcpHalCreate(out: *mut *mut Acp) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let acp = Box::new(Acp {
        vtable: &ACP_METHODS,
        state: Mutex::new(State {
            connected: false,
            mask: 0,
            messages: VecDeque::new(),
            capacity: 0,
            dropped: 0,
            since: Instant::now(),
        }),
    });
    // SAFETY: checked non-null.
    unsafe { out.write(Box::into_raw(acp)) };
    S_OK
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "audio",
        dlls: &["acphal.dll"],
        exports: vec![
            hle_export!(AcpHalAllocateShapeContexts),
            hle_export!(AcpHalReleaseShapeContexts),
            hle_export!(AcpHalCreate),
            hle_export!(ApuAlloc),
            hle_export!(ApuCreateHeap),
            hle_export!(ApuFree),
            hle_export!(ApuIsVirtualAddressValid),
            hle_export!(ApuMapApuAddress),
            hle_export!(ApuMapVirtualAddress),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_maps_every_byte_of_a_block() {
        let _serial = lock(&CONTEXT_TESTS);
        let (mut ptr, mut apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut ptr, &mut apu, 0x100, 128, 1), S_OK);
        assert!(apu != 0 && apu.is_multiple_of(128));
        let inner = ptr.cast::<u8>().wrapping_add(0x40).cast_const().cast();
        assert_eq!(ApuMapVirtualAddress(inner), apu + 0x40);
        assert!(ApuIsVirtualAddressValid(ptr, 128));
        assert_eq!(heap_unmap(apu + 4), Some(ptr.cast::<u8>().wrapping_add(4)));
        ApuFree(ptr);
        assert_eq!(ApuMapVirtualAddress(ptr), 0);
        assert_eq!(ApuAlloc(&mut ptr, &mut apu, 0x100, 6, 1), E_INVALIDARG);
    }

    /// The context arrays and the heap exist once per process: tests that
    /// allocate from them run one at a time (a concurrent allocation could
    /// take a freed range a test expects back).
    static CONTEXT_TESTS: Mutex<()> = Mutex::new(());

    #[test]
    fn heap_reuses_freed_addresses() {
        let _serial = lock(&CONTEXT_TESTS);
        let (mut a, mut b, mut c) = (
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        let (mut apu_a, mut apu_b, mut apu_c) = (0, 0, 0);
        assert_eq!(ApuAlloc(&mut a, &mut apu_a, 0x1000, 0x1000, 1), S_OK);
        assert_eq!(ApuAlloc(&mut b, &mut apu_b, 0x1000, 0x1000, 1), S_OK);
        ApuFree(a);
        assert_eq!(ApuAlloc(&mut c, &mut apu_c, 0x800, 0x800, 1), S_OK);
        // The freed range is free again, whichever part `c` took.
        assert!(
            lock(&HEAP)
                .free
                .iter()
                .any(|(&s, &e)| s <= apu_a + 0x800 && apu_a + 0x1000 <= e)
                || apu_c == apu_a
        );
        assert_eq!(
            ApuMapApuAddress(apu_b + 8),
            b.cast::<u8>().wrapping_add(8).cast()
        );
        assert!(ApuMapApuAddress(0).is_null());
        assert_eq!(
            ApuCreateHeap(64 << 20, 0),
            0x8ACA_0001_u32 as i32,
            "too late"
        );
        ApuFree(b);
        ApuFree(c);
    }

    /// Two packets of three silent mono frames each: a fixed one-subframe
    /// layout, no transmitted coefficients.
    fn silent_stream() -> Vec<u8> {
        let mut data = vec![0xffu8; 2 * xma::PACKET_BYTES];
        let mut put = |at: usize, value: u32, width: usize| {
            for i in 0..width {
                let (p, bit) = (at + i, (value >> (width - 1 - i)) & 1);
                data[p / 8] = (data[p / 8] & !(0x80 >> (p % 8))) | ((bit as u8) << (7 - p % 8));
            }
        };
        for packet in 0..2 {
            let base = packet * xma::PACKET_BYTES * 8;
            put(base, 3 << 26, 32);
            for frame in 0..3 {
                let at = base + 32 + frame * 31;
                put(at, 31, 15); // length
                put(at + 15, 0b10, 2); // fixed layout, one 512-sample subframe
                put(at + 17, 0, 8 + 1 + 3 + 2); // gain, no trim, empty subframe
            }
        }
        data
    }

    #[test]
    fn xma_contexts_decode_into_their_output_rings() {
        let _serial = lock(&CONTEXT_TESTS);
        let mut request = ShapeContexts {
            counts: [0, 0, 0, 0, 4, 0],
            arrays: [std::ptr::null_mut(); 6],
            apu: [0; 6],
        };
        assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
        let context = Context(request.arrays[4].cast::<u32>().wrapping_add(16));

        let (mut input, mut input_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut input, &mut input_apu, 0x1000, 0x800, 1), S_OK);
        let stream = silent_stream();
        // SAFETY: two packets were allocated.
        unsafe { std::ptr::copy_nonoverlapping(stream.as_ptr(), input.cast(), stream.len()) };
        let (mut output, mut output_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(
            ApuAlloc(&mut output, &mut output_apu, 0x400, 0x100, 2),
            S_OK
        );
        // SAFETY: the ring was just allocated.
        unsafe { output.cast::<u8>().write_bytes(0x55, 0x400) };

        // As the title sets one up: two packets, input 0 valid, a ring of
        // four blocks, writable, 48 kHz mono, reading from the first frame.
        context.set(0, 2 | 1 << 20 | 4 << 22);
        context.set(1, 1 << 31 | 3 << 27);
        context.set(2, 32);
        context.set(5, input_apu);
        context.set(7, output_apu);

        let mut acp = std::ptr::null_mut();
        assert_eq!(AcpHalCreate(&mut acp), S_OK);
        // SAFETY: just created.
        let m = unsafe { (*acp).vtable };
        assert_eq!((m.connect)(acp, 0x1000, 0x1000), S_OK);
        let submit = |command: u32, words: &[u32]| {
            assert_eq!(
                (m.submit)(acp, command, 0xf1, u32::MAX, words.as_ptr().cast(), 0),
                S_OK
            );
        };
        submit(ENABLE_XMA, &[1]);
        xma().step_all();
        // Full: all four blocks written (silence), the write block wrapped
        // onto the read block, and the ring no longer writable.
        assert_eq!(context.field(0, 27, 5), 0);
        assert_eq!(context.field(1, 31, 1), 0);
        // SAFETY: the ring.
        let ring = unsafe { std::slice::from_raw_parts(output.cast::<u8>(), 0x400) };
        assert!(ring.iter().all(|&b| b == 0));

        // Read four blocks at a time until the input has run out and the
        // ring keeps room (the decoding thread may fill it too).
        for _ in 0..20 {
            submit(ADVANCE_XMA_READ, &[1, 4]);
            xma().step_all();
            if context.field(2, 31, 1) == 1 && context.field(1, 31, 1) == 1 {
                break;
            }
        }
        // Six frames and the flushed tail: 3,328 samples, 26 blocks.
        let produced = |x: &Xma| {
            let output = lock(&x.slots[1].output);
            output.active.then_some(output.produced)
        };
        assert_eq!(produced(&xma()), Some(26 * SAMPLES_PER_BLOCK));
        assert_eq!(context.field(0, 20, 2), 0, "input used up");
        assert_eq!(context.field(2, 26, 5), INPUT_INVALID);

        // A count near `u32::MAX` wraps the read block like any other
        // (3 + 0xffff_ffff blocks on a ring of four is block 2); the sum
        // used to overflow.
        context.set_field(9, 0, 5, 3);
        submit(ADVANCE_XMA_READ, &[1, u32::MAX]);
        assert_eq!(context.field(9, 0, 5), 2);

        submit(DISABLE_XMA, &[1]);
        assert_eq!(produced(&xma()), None);
        (m.release)(acp);
        ApuFree(input);
        ApuFree(output);
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }

    #[test]
    fn full_message_queues_count_drops() {
        let mut acp = std::ptr::null_mut();
        assert_eq!(AcpHalCreate(&mut acp), S_OK);
        // SAFETY: just created.
        let m = unsafe { (*acp).vtable };
        assert_eq!((m.connect)(acp, 16, 2), S_OK);
        let mask = COMMAND_COMPLETED;
        assert_eq!(
            (m.submit)(
                acp,
                REGISTER_MESSAGES,
                0,
                u32::MAX,
                (&raw const mask).cast(),
                0
            ),
            S_OK
        );
        for id in 1..=4 {
            assert_eq!((m.submit)(acp, 12, id, u32::MAX, std::ptr::null(), 0), S_OK);
        }
        assert_eq!((m.message_count)(acp), 2);
        let mut message = Message {
            kind: 0,
            dropped: 0,
            usec: 0,
            data: [0; 16],
        };
        assert!((m.pop_message)(acp, &mut message) && (m.pop_message)(acp, &mut message));
        assert_eq!((m.submit)(acp, 12, 5, u32::MAX, std::ptr::null(), 0), S_OK);
        assert!((m.pop_message)(acp, &mut message));
        let dropped = message.dropped;
        assert_eq!(dropped, 2);
        (m.release)(acp);
    }

    #[test]
    fn startup_as_fmod_does_it() {
        let _serial = lock(&CONTEXT_TESTS);
        // Context arrays: rate conversion, DMA and XMA.
        let mut request = ShapeContexts {
            counts: [0x200, 0, 0, 0x400, 0x200, 0],
            arrays: [std::ptr::null_mut(); 6],
            apu: [0; 6],
        };
        assert_eq!(size_of::<ShapeContexts>(), 0x60);
        assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
        let arrays = request.arrays;
        assert!(!arrays[0].is_null() && !arrays[3].is_null() && !arrays[4].is_null());
        assert!(arrays[1].is_null());
        // SAFETY: 0x200 rate-conversion contexts of 32 bytes were allocated.
        unsafe { arrays[0].cast::<u8>().write_bytes(0, 0x200 << 5) };

        // Connect, register every message but frame starts, then wait for
        // the registration to be acknowledged.
        let mut acp = std::ptr::null_mut();
        assert_eq!(AcpHalCreate(&mut acp), S_OK);
        // SAFETY: just created.
        let m = unsafe { (*acp).vtable };
        assert_eq!((m.connect)(acp, 0x804, 0x804), S_OK);
        let mask = 0xfe_u32;
        let data = (&raw const mask).cast();
        assert_eq!(
            (m.submit)(acp, REGISTER_MESSAGES, 1, u32::MAX, data, 0),
            S_OK
        );
        let mut message = Message {
            kind: 0,
            dropped: 0,
            usec: 0,
            data: [0; 16],
        };
        assert!((m.pop_message)(acp, &mut message));
        let (kind, dropped) = (message.kind, message.dropped);
        assert_eq!((kind, dropped), (COMMAND_COMPLETED, 0));
        assert_eq!(
            u64::from_le_bytes(message.data[4..12].try_into().unwrap()),
            1
        );
        assert!(!(m.pop_message)(acp, &mut message));
        assert_eq!((m.disconnect)(acp), S_OK);
        assert_eq!((m.message_count)(acp), 1, "disconnected");
        (m.release)(acp);
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }

    fn test_voice(skip: usize) -> Decoding {
        let mut voice = Decoding::new();
        voice.start(skip);
        voice
    }

    #[test]
    fn loop_end_keeps_current_samples_and_defers_loop_start_skip() {
        let _serial = lock(&CONTEXT_TESTS);
        let mut request = ShapeContexts {
            counts: [0, 0, 0, 0, 2, 0],
            arrays: [std::ptr::null_mut(); 6],
            apu: [0; 6],
        };
        assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
        let context = Context(request.arrays[4].cast::<u32>());

        let (mut input, mut input_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut input, &mut input_apu, 0x1000, 0x800, 1), S_OK);
        let stream = silent_stream();
        // SAFETY: two packets were allocated.
        unsafe { std::ptr::copy_nonoverlapping(stream.as_ptr(), input.cast(), stream.len()) };

        // 2 packets, loops = 1 (bits 12..20), input 0 valid (bit 20), ring size 4.
        context.set(0, 2 | (1 << 12) | (1 << 20) | (4 << 22));
        // loop_end_subframe = 1 (keep 2 subframes = 256 samples, bits 12..14),
        // loop_start_subframe skip = 2 (256 samples for next iteration, bits 17..20),
        // 48 kHz mono, writable.
        context.set(1, (1 << 12) | (2 << 17) | (3 << 27) | (1 << 31));
        context.set(2, 32); // read offset at first frame
        context.set(3, 32); // loop_start = bit 32
        context.set(4, 32); // loop_end = bit 32
        context.set(5, input_apu);

        let mut voice = test_voice(0);
        assert!(decode(context, &mut voice, true));
        // Current loop_end frame keeps 256 samples; loop_start skip of 256 is
        // deferred to the next frame rather than eating the current frame.
        assert_eq!(voice.pcm.len(), 2 * xma::SUBFRAME_SAMPLES);
        assert_eq!(voice.skip, 2 * xma::SUBFRAME_SAMPLES);
        assert_eq!(context.field(0, 12, 8), 0);
        assert_eq!(context.field(2, 0, 26), 32);

        ApuFree(input);
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }

    #[test]
    fn pump_keeps_decoding_across_empty_steps_without_injecting_silence() {
        let _serial = lock(&CONTEXT_TESTS);
        let mut request = ShapeContexts {
            counts: [0, 0, 0, 0, 2, 0],
            arrays: [std::ptr::null_mut(); 6],
            apu: [0; 6],
        };
        assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
        let context = Context(request.arrays[4].cast::<u32>());

        // Buffer 0 has an empty/exhausted packet (all 0xFF -> read_frame returns None),
        // buffer 1 has a valid stream with 6 frames.
        let (mut input0, mut input0_apu) = (std::ptr::null_mut(), 0);
        let (mut input1, mut input1_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut input0, &mut input0_apu, 0x800, 0x800, 1), S_OK);
        assert_eq!(ApuAlloc(&mut input1, &mut input1_apu, 0x1000, 0x800, 1), S_OK);
        let stream = silent_stream();
        // SAFETY: buffers were allocated above.
        unsafe {
            input0.cast::<u8>().write_bytes(0xff, 0x800);
            std::ptr::copy_nonoverlapping(stream.as_ptr(), input1.cast(), stream.len());
        }
        let (mut output, mut output_apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut output, &mut output_apu, 0x400, 0x100, 2), S_OK);

        // Input 0 (1 packet) and input 1 (2 packets) both valid; ring of 4 blocks, writable.
        context.set(0, 1 | (0b11 << 20) | (4 << 22));
        context.set(1, 2 | (3 << 27) | (1 << 31));
        context.set(2, 32);
        context.set(5, input0_apu);
        context.set(6, input1_apu);
        context.set(7, output_apu);

        // Start with skip = 512 (4 subframes): frame 0 of buffer 1 is also skipped whole.
        let mut voice = test_voice(xma::FRAME_SAMPLES);
        let ring = Mutex::new(Output::new(true, voice.generation));
        assert_eq!(
            fill(context, &ring, &mut voice),
            Step::Idle,
            "the ring is full"
        );

        // Buffer 0 became invalid, switched to buffer 1, skipped frame 0 whole,
        // and decoded frame 1 (512 samples = 4 blocks) without injecting any
        // spurious silence block during the buffer switch or full-frame skip.
        assert_eq!(context.field(0, 20, 1), 0, "input 0 consumed");
        assert_eq!(context.field(4, 31, 1), 1, "switched to input 1");
        assert_eq!(voice.skip, 0);
        assert_eq!(
            voice.frames, 2,
            "frame 0 skipped + frame 1 filled the 4-block ring"
        );
        assert_eq!(lock(&ring).produced, 4 * SAMPLES_PER_BLOCK);

        ApuFree(input0);
        ApuFree(input1);
        ApuFree(output);
        assert_eq!(AcpHalReleaseShapeContexts(), S_OK);
    }

    /// Allocates `count` XMA contexts and gives voice `index` the silent
    /// stream and a ring of four blocks, as the title sets one up.
    struct TestVoice {
        context: Context,
        input: *mut c_void,
        output: *mut c_void,
    }

    impl TestVoice {
        fn new(count: u32, index: usize) -> TestVoice {
            let mut request = ShapeContexts {
                counts: [0, 0, 0, 0, count, 0],
                arrays: [std::ptr::null_mut(); 6],
                apu: [0; 6],
            };
            assert_eq!(AcpHalAllocateShapeContexts(&mut request), S_OK);
            let context = Context(request.arrays[4].cast::<u32>().wrapping_add(16 * index));
            let (mut input, mut input_apu) = (std::ptr::null_mut(), 0);
            assert_eq!(ApuAlloc(&mut input, &mut input_apu, 0x1000, 0x800, 1), S_OK);
            let stream = silent_stream();
            // SAFETY: two packets were allocated.
            unsafe { std::ptr::copy_nonoverlapping(stream.as_ptr(), input.cast(), stream.len()) };
            let (mut output, mut output_apu) = (std::ptr::null_mut(), 0);
            assert_eq!(
                ApuAlloc(&mut output, &mut output_apu, 0x400, 0x100, 2),
                S_OK
            );
            context.set(0, 2 | 1 << 20 | 4 << 22);
            context.set(1, 1 << 31 | 3 << 27);
            context.set(2, 32);
            context.set(5, input_apu);
            context.set(7, output_apu);
            TestVoice {
                context,
                input,
                output,
            }
        }
    }

    impl Drop for TestVoice {
        fn drop(&mut self) {
            ApuFree(self.input);
            ApuFree(self.output);
            AcpHalReleaseShapeContexts();
        }
    }

    thread_local! {
        /// The slots this thread stepped, in order.
        pub(super) static STEPPED: std::cell::RefCell<Vec<usize>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// A pass looks only at the voices signalled since the last one (and
    /// those it left waiting), not at every context. The decoding thread
    /// (started by an earlier test) may take a signal first, so only this
    /// thread's own steps are counted.
    #[test]
    fn only_signalled_voices_are_stepped() {
        let _serial = lock(&CONTEXT_TESTS);
        let voice = TestVoice::new(200, 130);
        let stepped = || STEPPED.with_borrow_mut(std::mem::take);
        let xma = xma();
        let mut waiting = Vec::new();
        stepped();
        // Nothing enabled or signalled: nothing is stepped.
        xma.pass(&mut waiting, false);
        assert_eq!(stepped(), []);
        xma.enable(130);
        xma.pass(&mut waiting, false);
        assert!(stepped().iter().all(|&i| i == 130));
        // A sweep looks at every enabled voice, and only those.
        xma.pass(&mut waiting, true);
        assert_eq!(stepped(), [130]);
        assert_eq!(voice.context.field(1, 31, 1), 0, "the ring was filled");
        // Its ring is full, so it does not wait for input: a pass with
        // nothing signalled steps nothing, and neither does a sweep once
        // it is disabled.
        assert_eq!(waiting.iter().map(|w| w.count_ones()).sum::<u32>(), 0);
        xma.pass(&mut waiting, false);
        assert_eq!(stepped(), []);
        xma.disable(130);
        xma.pass(&mut waiting, true);
        assert_eq!(stepped(), []);
        drop(xma);
    }

    /// `advance_read`, sent from the title's mixer thread, does not wait for
    /// a decode of the same voice: it takes only the output lock.
    #[test]
    fn advance_read_does_not_wait_for_a_decode() {
        let _serial = lock(&CONTEXT_TESTS);
        let voice = TestVoice::new(4, 1);
        let xma = xma();
        xma.enable(1);
        voice.context.set_field(9, 0, 5, 0);
        // As if the decoder thread were in the middle of decoding voice 1.
        let decoding = lock(&xma.slots[1].decoding);
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            s.spawn(|| {
                xma.advance_read(1, 3);
                done.send(()).unwrap();
            });
            let returned = finished.recv_timeout(Duration::from_secs(5));
            drop(decoding);
            assert!(returned.is_ok(), "advance_read waited for the decode");
        });
        assert_eq!(voice.context.field(9, 0, 5), 3);
        assert_eq!(lock(&xma.slots[1].output).takes, 1);
        xma.disable(1);
        drop(xma);
    }

    /// The context words the title sees move on with the ring, not ahead of
    /// it: a full ring decodes nothing, and reading k blocks decodes only
    /// the frames that k blocks need (four blocks to a mono frame here).
    #[test]
    fn decoding_keeps_pace_with_the_ring() {
        let _serial = lock(&CONTEXT_TESTS);
        let voice = TestVoice::new(4, 1);
        let context = voice.context;
        // The silent stream's frames start at bits 32, 63 and 94 of packet
        // 0, then at bit 32 of packet 1.
        let next_frame = || context.field(2, 0, 26);
        let second_packet = (xma::PACKET_BYTES * 8 + 32) as u32;
        let xma = xma();
        xma.enable(1);
        xma.step_all();
        assert_eq!(context.field(1, 31, 1), 0, "the ring is full");
        assert_eq!(next_frame(), 63, "one frame for four blocks");
        // Full: nothing more is decoded however often the voice is stepped.
        xma.step_all();
        xma.step_all();
        assert_eq!(next_frame(), 63);
        // Four blocks read: one frame.
        xma.advance_read(1, 4);
        xma.step_all();
        assert_eq!(next_frame(), 94);
        // Two blocks read: one frame, half of it kept for later.
        xma.advance_read(1, 2);
        xma.step_all();
        assert_eq!(next_frame(), second_packet);
        // Two more: the kept half fills them, nothing is decoded.
        xma.advance_read(1, 2);
        xma.step_all();
        assert_eq!(next_frame(), second_packet);
        assert_eq!(context.field(0, 20, 2), 1, "input 0 is still valid");
        assert_eq!(context.field(4, 31, 1), 0, "still on input 0");
        assert_eq!(lock(&xma.slots[1].output).produced, 12 * SAMPLES_PER_BLOCK);
        xma.disable(1);
        drop(xma);
    }

    /// A voice re-enabled while the output side still belonged to the old
    /// one: the old voice's samples never reach the new ring.
    #[test]
    fn a_stale_voice_does_not_fill_the_ring() {
        let _serial = lock(&CONTEXT_TESTS);
        let voice = TestVoice::new(4, 1);
        // SAFETY: the ring was just allocated.
        unsafe { voice.output.cast::<u8>().write_bytes(0x55, 0x400) };
        let mut old = test_voice(0);
        let output = Mutex::new(Output::new(true, old.generation.wrapping_add(1)));
        assert_eq!(fill(voice.context, &output, &mut old), Step::Idle);
        assert_eq!(voice.context.field(2, 0, 26), 32, "nothing was decoded");
        // SAFETY: the ring.
        let ring = unsafe { std::slice::from_raw_parts(voice.output.cast::<u8>(), 0x400) };
        assert!(ring.iter().all(|&b| b == 0x55), "nothing was written");
        assert_eq!(voice.context.field(0, 27, 5), 0);
        assert_eq!(lock(&output).produced, 0);
    }

    #[test]
    fn heap_bytes_retains_allocation_across_apu_free() {
        let (mut ptr, mut apu) = (std::ptr::null_mut(), 0);
        assert_eq!(ApuAlloc(&mut ptr, &mut apu, 4096, 256, 1), S_OK);

        let mut slice = heap_bytes(apu, 4096).expect("live APU block");
        assert_eq!(Arc::strong_count(&slice._alloc), 2);

        // Freeing the block unmaps its APU range immediately, but defers
        // deallocation while the decoder's `HeapSlice` still holds a reference.
        ApuFree(ptr);
        assert!(ApuMapApuAddress(apu).is_null());
        assert_eq!(Arc::strong_count(&slice._alloc), 1);

        slice.fill(0x5A);
        assert!(slice.iter().all(|&b| b == 0x5A));
        // Overflow check: `offset + len` wrapping must not pass bounds checks.
        assert!(heap_bytes(apu, usize::MAX).is_none());
        drop(slice);
    }
}
