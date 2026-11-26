//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, GetLastError, HANDLE, INVALID_HANDLE_VALUE, SetLastError,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    // The console's words as given, for the memory core's region table.
    let (title_kind, title_protect) = (kind, protect);
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    // With the memory core on (`VICTORY_MEMORY_CORE=1`), pages committed
    // again inside a reservation: a decommit of the title's that the core
    // still holds back must not take them, from before the host's commit to
    // after it.
    let core = victory_gpu::memory_core_on();
    let _committing = (core && !constrained && kind & MEM_COMMIT != 0 && kind & MEM_RESERVE == 0)
        .then(|| victory_gpu::title_memory_committing(base as usize, size));
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    // With the memory core on: a new reservation that is not a placeholder
    // is a region the core may give homes in, with the bits the host was not
    // given kept as hints.
    if core && !at.is_null() && kind & MEM_RESERVE_PLACEHOLDER == 0 && kind & MEM_RESERVE != 0 {
        victory_gpu::title_memory_region(at as usize, size, title_kind, flags, title_protect);
    }
    // Memory there now: holes of placeholders known there are looked at
    // again.
    if !at.is_null() && !placeholder {
        victory_gpu::title_memory_allocated(at as usize, size);
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // With `VICTORY_MEMORY_OWN=1` (and the memory core on), a block of
    // 64 KiB or more is a watched region of its own below 128 GiB, which
    // the core may use in place for the GPU; its header says so.
    // shortcut: smaller blocks stay in the host heap (above 2^40, no home):
    // a region each would waste most of a 64 KiB page, and several in one
    // page would have a home over one pin the others.
    // shortcut: every memory type takes this path, CPU-only ones too (one
    // reservation each, up to 64 KiB of rounding): the memory-type bits of
    // the attributes are not decoded here yet; the core's census counts
    // our regions, so the cost shows on its line.
    let own = (size >= OWN_LEAST && victory_gpu::memory_own_on())
        .then(|| victory_gpu::memory_own_alloc(layout.size(), layout.align()))
        .flatten();
    let (block, mark) = match own {
        Some(block) => (block.as_ptr(), OWN_BLOCK),
        // shortcut: the host heap's addresses lie above 2^40, unlike the
        // console's; nothing seen builds ray-tracing structures in this
        // memory (the title uses its GPU heap). Sub-allocate from a
        // reservation below 2^40 if one does.
        // SAFETY: the layout has a non-zero size.
        None => (unsafe { alloc(layout) }, 0),
    };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align() | mark]);
        user.cast()
    }
}

/// The least block [`XMemAllocDefault`] makes a region of its own for with
/// `VICTORY_MEMORY_OWN=1`.
const OWN_LEAST: usize = 64 << 10;
/// In a block header's alignment word: the block is a region of our own
/// (`victory_gpu::memory_own_alloc`), given back through the memory core.
const OWN_BLOCK: usize = 1 << (usize::BITS - 1);

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            // A region of `XMemVirtualAlloc`'s: through the memory core when
            // it is on, which lets the homes in it go first.
            if victory_gpu::memory_core_on() {
                victory_gpu::title_virtual_free(address as usize, 0, MEM_RELEASE);
            } else {
                VirtualFree(address, 0, MEM_RELEASE);
            }
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        if align & OWN_BLOCK != 0 {
            // Now, on the freeing thread: no lock of ours is held here. A
            // committed resource's memory from the title's allocator comes
            // here too, but only from the title's own last Release; our
            // memory that a drop of ours may free with locks held (heaps,
            // descriptor heaps, resources with no allocator) waits for the
            // sweep instead.
            victory_gpu::memory_own_free_now(user.sub(align & !OWN_BLOCK) as usize);
            return;
        }
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

struct Section {
    handle: HANDLE,
    /// Made with `SEC_COMMIT`: its pages need no commit when a view shows
    /// them.
    committed: bool,
    /// A chunk of the memory pool's (`Pool`): the GPU side is told when
    /// it goes.
    pool: bool,
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // Before the handle is closed, and so before the host can give its
        // value to another section.
        if self.pool {
            victory_gpu::title_memory_pool_chunk_gone(self.handle as usize);
        }
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// The memory pool (`VICTORY_MEMORY_POOL=1` with `VICTORY_MEMORY_CORE=1`,
/// `victory_gpu::memory_pool_on`): an allocation of physical pages is
/// backed by chunks of [`Pool::chunk_pages`] pages, each a section of its
/// own made on the first map or write into it, committed whole
/// (`SEC_COMMIT`) when the PC has the room, so that the GPU side can open it
/// as a heap (`OpenExistingHeapFromFileMapping`, which refuses a reserved
/// section) and read and write the title's pages in place. A chunk the PC
/// has no room for, or one on a Windows that cannot open sections as heaps,
/// is reserved and committed as views show it, as without the pool.
///
/// Off, an allocation is one section made with it, reserved, as before:
/// the title asks for 6.1 GiB at once, which a busy PC often cannot commit
/// (error 1450). With the pool, the commit charge is that of the chunks the
/// title touches, each whole: it can pass what the title uses by up to a
/// chunk's size per chunk, so with small chunks
/// (`VICTORY_MEMORY_POOL_CHUNK_MB`, `victory_gpu::memory_pool_chunk_bytes`)
/// it follows what the title touches.
///
/// With `VICTORY_MEMORY_POOL_OPEN_PCT` (`victory_gpu::memory_pool_open_pct`)
/// each chunk counts the pages of it some view shows ([`Chunk::shown`]), and
/// the GPU side is told when that passes the share either way, so that it
/// opens a chunk as a heap only for one the title maps enough of.
///
/// With the pool `XMemAllocatePhysicalPages` makes no section and so never
/// fails for the host's sake; a chunk the host can make no section for
/// fails the map or write that first touches it instead.
#[derive(Clone, Copy)]
struct Pool {
    /// Pages of a chunk.
    chunk_pages: usize,
    /// The PC's commit charge still free, in bytes (`None`: not known).
    room: fn() -> Option<usize>,
    /// The Windows build.
    build: fn() -> u32,
    /// Whether the GPU side can make homes in a chunk at all
    /// (`victory_gpu::memory_pool_homes_possible`).
    homes: fn() -> bool,
    /// The share of a chunk's pages, in percent, whose mapping the GPU side
    /// is told of ([`Chunk::shown`]); 0: none told.
    open_pct: u32,
}

/// Pages of a chunk of `bytes` (`victory_gpu::memory_pool_chunk_bytes`: a
/// multiple of 16 MiB, so of pages).
fn chunk_pages(bytes: u64) -> usize {
    bytes as usize / PAGE
}

/// What a chunk must leave free of the PC's commit charge.
const COMMIT_HEADROOM: usize = 1 << 30;

/// The first Windows build whose Direct3D opens a section as a heap
/// (Windows Server 2022, Windows 11).
const FILE_MAPPING_BUILD: u32 = 20348;

impl Pool {
    /// The process's pool, if it is on (never under Wine: its sections are
    /// committed whole already, and opening them as heaps there is
    /// untried).
    fn process() -> Option<Pool> {
        (victory_gpu::memory_pool_on() && !on_wine()).then(|| Pool {
            chunk_pages: chunk_pages(victory_gpu::memory_pool_chunk_bytes()),
            room: commit_room,
            build: windows_build,
            homes: victory_gpu::memory_pool_homes_possible,
            open_pct: victory_gpu::memory_pool_open_pct(),
        })
    }

    /// How a chunk of `bytes` is to be made.
    fn kind(&self, bytes: usize) -> victory_gpu::PoolChunk {
        use victory_gpu::PoolChunk;
        if (self.build)() < FILE_MAPPING_BUILD {
            PoolChunk::OldWindows
        } else if !(self.homes)() {
            PoolChunk::NoHomes
        } else if (self.room)().is_none_or(|room| room < bytes.saturating_add(COMMIT_HEADROOM)) {
            PoolChunk::ShortOfCommit
        } else {
            PoolChunk::Committed
        }
    }
}

/// The PC's commit charge still free, in bytes.
fn commit_room() -> Option<usize> {
    let mut info = PERFORMANCE_INFORMATION {
        cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
        ..Default::default()
    };
    // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
    let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
    queried.then(|| {
        info.CommitLimit
            .saturating_sub(info.CommitTotal)
            .saturating_mul(info.PageSize)
    })
}

/// The Windows build, as the kernel says (not the compatibility answer of
/// `GetVersionEx`); 0 if it cannot be asked.
fn windows_build() -> u32 {
    static BUILD: LazyLock<u32> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        /// `RTL_OSVERSIONINFOW`.
        #[repr(C)]
        struct Version {
            size: u32,
            major: u32,
            minor: u32,
            build: u32,
            platform: u32,
            service_pack: [u16; 128],
        }
        type RtlGetVersion = unsafe extern "system" fn(*mut Version) -> i32;
        // SAFETY: plain lookups in ntdll, always loaded.
        let Some(get) = (unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            if ntdll.is_null() {
                None
            } else {
                GetProcAddress(ntdll, c"RtlGetVersion".as_ptr().cast())
            }
        }) else {
            return 0;
        };
        let mut version = Version {
            size: size_of::<Version>() as u32,
            major: 0,
            minor: 0,
            build: 0,
            platform: 0,
            service_pack: [0; 128],
        };
        // SAFETY: `RtlGetVersion` has this signature and fills the structure
        // its size field describes.
        let status = unsafe {
            std::mem::transmute::<unsafe extern "system" fn() -> isize, RtlGetVersion>(get)(
                &mut version,
            )
        };
        if status == 0 { version.build } else { 0 }
    });
    *BUILD
}

/// Pages of one allocation that share a section: the whole allocation
/// without the pool, a chunk of it with ([`Pool`]).
struct Chunk {
    bytes: usize,
    /// Made with the chunk (without the pool), or on its first map or write.
    section: std::sync::OnceLock<Section>,
    /// Taken to make `section`.
    making: Mutex<()>,
    /// With the pool: how to make `section`.
    pool: Option<Pool>,
    /// With the pool and its share ([`Pool::open_pct`]): the pages of the
    /// chunk some view shows now (keys of `Physical::shown`). Changed with
    /// the page table's lock held.
    mapped: AtomicUsize,
}

impl Chunk {
    /// A chunk whose section is already made.
    fn made(bytes: usize, section: Section) -> Chunk {
        Chunk {
            bytes,
            section: std::sync::OnceLock::from(section),
            making: Mutex::new(()),
            pool: None,
            mapped: AtomicUsize::new(0),
        }
    }

    /// The share of the chunk's pages whose mapping is counted, in percent
    /// ([`Pool::open_pct`]); `None`: nothing is counted.
    fn open_pct(&self) -> Option<u32> {
        self.pool.map(|pool| pool.open_pct).filter(|&pct| pct > 0)
    }

    /// `gained` pages of the chunk are shown by a view now that none showed,
    /// and `lost` no longer by any. With the pool's share, the GPU side is
    /// told when the pages shown pass it either way
    /// (`victory_gpu::title_memory_pool_chunk_mapped`). Called with the page
    /// table's lock held, as the GPU side is told of views, once per change
    /// of the page table with its net count ([`Shown`]): a view cut in two
    /// is unmapped whole and its parts mapped again, and a chunk near its
    /// share must not be told it dropped below and passed it again in
    /// between, where a home asked for meanwhile would stay a copy.
    fn shown(&self, gained: usize, lost: usize) {
        let Some(pct) = self.open_pct() else {
            return;
        };
        let Some(section) = self.section.get() else {
            return;
        };
        let before = self.mapped.load(Ordering::Relaxed);
        let after = (before + gained).saturating_sub(lost);
        self.mapped.store(after, Ordering::Relaxed);
        let pages = self.bytes / PAGE;
        let enough = |n: usize| n * 100 >= pct as usize * pages;
        if enough(before) != enough(after) {
            #[cfg(test)]
            SHARE_TOLD.fetch_add(1, Ordering::Relaxed);
            victory_gpu::title_memory_pool_chunk_mapped(section.handle as usize, enough(after));
        }
    }

    /// The chunk's section, made now if it is the first map or write into a
    /// chunk of the pool's; `None` if the host could not make one (a host
    /// failure of the map or write, as a failed commit is).
    fn section(&self) -> Option<&Section> {
        if let Some(section) = self.section.get() {
            return Some(section);
        }
        let _making = self.making.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(section) = self.section.get() {
            return Some(section);
        }
        let section = make_chunk(self.bytes, &self.pool?)?;
        Some(self.section.get_or_init(|| section))
    }
}

/// A section of `bytes`, committed or reserved; null if the host refused.
fn create_section(bytes: usize, committed: bool) -> HANDLE {
    // SAFETY: plain section creation.
    unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            PAGE_READWRITE | if committed { SEC_COMMIT } else { SEC_RESERVE },
            (bytes as u64 >> 32) as u32,
            bytes as u32,
            std::ptr::null(),
        )
    }
}

/// Makes the section of a chunk of the pool's: committed if [`Pool::kind`]
/// says so and the host gives it, else reserved. The GPU side is told of it
/// either way.
fn make_chunk(bytes: usize, pool: &Pool) -> Option<Section> {
    use victory_gpu::PoolChunk;
    let mut kind = pool.kind(bytes);
    let mut handle = std::ptr::null_mut();
    // A section made (a committed one refused, a reserved one made, too):
    // the map or write succeeds, and the title sees the last error it had
    // before.
    // SAFETY: plain thread state.
    let before = unsafe { GetLastError() };
    if kind == PoolChunk::Committed {
        handle = create_section(bytes, true);
        if handle.is_null() {
            kind = PoolChunk::Refused;
        }
    }
    let committed = !handle.is_null();
    if handle.is_null() {
        handle = create_section(bytes, false);
    }
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        commit_failed(bytes, &error, "a chunk of physical pages could not be made");
        return None;
    }
    victory_gpu::title_memory_section(handle as usize, bytes as u64, committed);
    victory_gpu::title_memory_pool_chunk(handle as usize, bytes as u64, kind);
    // Last: what the GPU side was told may log, and logging may set it.
    // SAFETY: plain thread state.
    unsafe { SetLastError(before) };
    Some(Section {
        handle,
        committed,
        pool: true,
    })
}

/// Where a page lives, or what a view starts with: a chunk and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    chunk: Arc<Chunk>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.chunk) as usize, self.index + k)
    }
}

/// The pages of chunks newly shown and no longer shown in one change of the
/// page table, told at its end with their net count ([`Chunk::shown`]).
/// Only chunks whose mapping is counted are kept.
#[derive(Default)]
struct Shown(Vec<(Arc<Chunk>, usize, usize)>);

impl Shown {
    fn add(&mut self, chunk: &Arc<Chunk>, gained: usize, lost: usize) {
        if gained == 0 && lost == 0 || chunk.open_pct().is_none() {
            return;
        }
        match self.0.iter_mut().find(|(c, ..)| Arc::ptr_eq(c, chunk)) {
            Some((_, g, l)) => {
                *g += gained;
                *l += lost;
            }
            None => self.0.push((chunk.clone(), gained, lost)),
        }
    }

    /// Tells each chunk its net count; with the page table's lock held.
    fn tell(self) {
        for (chunk, gained, lost) in self.0 {
            chunk.shown(gained, lost);
        }
    }
}

/// How often a chunk was told its share was passed (tests only).
#[cfg(test)]
static SHARE_TOLD: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    if n.checked_mul(PAGE).is_none_or(|s| s == 0) {
        return 0;
    }
    // SAFETY: the title passes room for `n` page numbers.
    let out = unsafe { std::slice::from_raw_parts_mut(pages, n) };
    if !allocate_pages(out, Pool::process()) {
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    1
}

/// [`XMemAllocatePhysicalPages`] of `pages.len()` pages (at least one),
/// their numbers written to `pages`; with `pool`, in chunks whose sections
/// are made later ([`Pool`]). False if the host refused the section.
fn allocate_pages(pages: &mut [usize], pool: Option<Pool>) -> bool {
    let (n, size) = (pages.len(), pages.len() * PAGE);
    let chunks: Vec<Arc<Chunk>> = match pool {
        Some(pool) => (0..n.div_ceil(pool.chunk_pages))
            .map(|c| {
                Arc::new(Chunk {
                    bytes: (n - c * pool.chunk_pages).min(pool.chunk_pages) * PAGE,
                    section: std::sync::OnceLock::new(),
                    making: Mutex::new(()),
                    pool: Some(pool),
                    mapped: AtomicUsize::new(0),
                })
            })
            .collect(),
        None => {
            // A pagefile-backed section of `size` bytes, its pages committed
            // when first mapped or written (`commit`). Under Wine a committed
            // section is a sparse file, free until touched, and committing a
            // reserved one page by page costs a wineserver request each that
            // scans every range committed so far: commit it whole there.
            let handle = create_section(size, on_wine());
            if handle.is_null() {
                let error = std::io::Error::last_os_error();
                tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
                return false;
            }
            // For the GPU side's in-place census: only a committed section
            // can be opened as a host heap.
            victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
            let section = Section {
                handle,
                committed: on_wine(),
                pool: false,
            };
            vec![Arc::new(Chunk::made(size, section))]
        }
    };
    let per_chunk = pool.map_or(n, |pool| pool.chunk_pages);
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for (index, out) in pages.iter_mut().enumerate() {
        let slice = Slice {
            chunk: chunks[index / per_chunk].clone(),
            index: index % per_chunk,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        *out = number;
    }
    true
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        let section = slice.chunk.section()?;
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        (section.committed || commit(at.Value as usize, len)).then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.chunk, &slice.chunk) && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    let mut physical = physical();
    let pages = (!pages.is_null()).then(|| {
        // SAFETY: the title passes `n` page numbers.
        unsafe { std::slice::from_raw_parts(pages, n) }
    });
    if map_pages(&mut physical, address as usize, n, pages) {
        address
    } else {
        std::ptr::null_mut()
    }
}

/// [`XMemMapPhysicalPages`] with the physical state held: unmaps
/// `[base, base + n pages)`, then maps `pages` there (none: only unmaps).
/// False if a page or the host fails.
///
/// Until the pages are mapped again: `unmap` takes whole views away and
/// maps what stays of them again, and a mapped view is a reservation until
/// it is committed. A read of title memory in place must not fall into that
/// gap: the remapping lock is held throughout, for the addresses the change
/// touches ([`changing`]).
#[track_caller]
fn map_pages(physical: &mut Physical, base: usize, n: usize, pages: Option<&[usize]>) -> bool {
    let _changing = changing(physical, base, n * PAGE);
    let mut shown = Shown::default();
    let mapped = remap(physical, base, n, pages, &mut shown);
    shown.tell();
    mapped
}

/// [`map_pages`] within its remapping lock; the pages shown and no longer
/// shown are added to `shown`.
fn remap(
    physical: &mut Physical,
    base: usize,
    n: usize,
    pages: Option<&[usize]>,
    shown: &mut Shown,
) -> bool {
    unmap(physical, base, n * PAGE, shown);
    let Some(pages) = pages else {
        return true;
    };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return false;
        };
        // One view per run of consecutive slices of one section (of one
        // chunk, with the memory pool).
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.chunk, &first.chunk) && s.index == first.index + k
                })
            })
            .count();
        if !map(physical, base + i * PAGE, run * PAGE, first, shown) {
            return false;
        }
        i += run;
    }
    true
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
///
/// The physical state is taken once for the call; the remapping lock per
/// address, for that address alone ([`changing`]). Taken once for the whole
/// call it was held up to 1.95 s (run mem-W3a), and every texture refresh
/// waited; taken per page when it covered all title memory, each page
/// waited for every in-place read under way and the refreshes queued behind
/// each (run amd-F). Each address is mapped on its own:
/// addresses that follow one another can lie in different placeholders,
/// which one view cannot span (run amd-H: error 487, and the title stopped).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
    let addresses = unsafe { std::slice::from_raw_parts(addresses, n) };
    let pages = (!pages.is_null()).then(|| unsafe { std::slice::from_raw_parts(pages, n) });
    let mut physical = physical();
    for (i, &address) in addresses.iter().enumerate() {
        let page = pages.map(|p| &p[i..=i]).filter(|p| p[0] != 0);
        if !map_pages(&mut physical, address as usize, 1, page) {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    let (to, from, len) = (to as usize, from as usize, len as usize);
    let mut physical = physical();
    // As in `map_pages`: no read in place meets the gap.
    let _changing = changing(&physical, to, len);
    let mut shown = Shown::default();
    let mapped = alias(&mut physical, to, from, len, &mut shown);
    shown.tell();
    mapped
}

/// [`map_alias`] within its remapping lock; the pages shown and no longer
/// shown are added to `shown`.
fn alias(physical: &mut Physical, to: usize, from: usize, len: usize, shown: &mut Shown) -> bool {
    unmap(physical, to, len, shown);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                chunk: s.chunk.clone(),
            })
        else {
            return false;
        };
        if !map(physical, to + k, PAGE, slice, shown) {
            return false;
        }
    }
    true
}

/// The remapping lock (`victory_gpu::title_memory_remapping_range`) for a
/// change of `[base, base + len)`: those addresses and the whole of the
/// views reaching past them, which `unmap` takes away and maps again in
/// part. Reads of title memory elsewhere go on meanwhile.
#[track_caller]
fn changing(physical: &Physical, base: usize, len: usize) -> victory_gpu::Remapping {
    let end = base.saturating_add(len);
    let holding = |at: usize| {
        physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(start, (size, _))| at < *start + size)
            .map(|(start, (size, _))| (*start, *start + size))
    };
    let lo = holding(base).map_or(base, |(start, _)| start.min(base));
    let hi = if len == 0 { end } else { holding(end - 1).map_or(end, |(_, stop)| stop.max(end)) };
    victory_gpu::title_memory_remapping_range(lo, (hi - lo).max(1))
}

/// Diagnostics: where the time under the remapping lock goes, since the last
/// line: [unmaps, their µs, kernel maps (placeholder split, view, commit),
/// their µs, µs of our own bookkeeping after a map (the GPU side's guards and
/// census)]. The texture refreshes wait for all of it.
#[cfg(feature = "diagnostics")]
static REMAP_TIME: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

#[cfg(feature = "diagnostics")]
fn remap_time(k: usize, since: std::time::Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    REMAP_TIME[k].fetch_add(since.elapsed().as_micros() as u64, Relaxed);
    if k != 4 {
        REMAP_TIME[k - 1].fetch_add(1, Relaxed);
    }
}

/// Logs [`REMAP_TIME`] at most every 5 s (diagnostics).
#[cfg(feature = "diagnostics")]
fn remap_time_line() {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let now = std::time::Instant::now();
    if last.is_some_and(|t| now.duration_since(t) < std::time::Duration::from_secs(5)) {
        return;
    }
    let secs = last.map_or(5.0, |t| now.duration_since(t).as_secs_f64());
    *last = Some(now);
    let [unmaps, unmap_us, maps, map_us, books] = [0, 1, 2, 3, 4].map(|i| REMAP_TIME[i].swap(0, Relaxed));
    tracing::info!(
        "remapping work per second over {secs:.0} s: {:.0} unmaps {:.1} ms, {:.0} kernel maps {:.1} ms, \
         {:.1} ms of our bookkeeping after them",
        unmaps as f64 / secs,
        unmap_us as f64 / 1e3 / secs,
        maps as f64 / secs,
        map_us as f64 / 1e3 / secs,
        books as f64 / 1e3 / secs,
    );
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder; the pages newly shown are added to `shown`.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice, shown: &mut Shown) -> bool {
    #[cfg(feature = "diagnostics")]
    let began = std::time::Instant::now();
    // With the memory pool, the chunk's section may be made now.
    let Some(section) = slice.chunk.section() else {
        return false;
    };
    // SAFETY: splits the placeholder around the range (failing harmlessly
    // when the range is the whole placeholder), then maps the section
    // there; both act only on placeholders.
    let view = unsafe {
        VirtualFree(
            base as *mut c_void,
            len,
            MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
        );
        MapViewOfFile3(
            section.handle,
            GetCurrentProcess(),
            base as *const c_void,
            (slice.index * PAGE) as u64,
            len,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    if !section.committed && !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    #[cfg(feature = "diagnostics")]
    remap_time(3, began);
    #[cfg(feature = "diagnostics")]
    let books = std::time::Instant::now();
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census (and,
    // with the memory pool, the homes it gives the range).
    victory_gpu::title_memory_view(
        base,
        len,
        section.handle as usize,
        (slice.index * PAGE) as u64,
    );
    let mut gained = 0;
    for k in 0..len.div_ceil(PAGE) {
        let addresses = physical.shown.entry(slice.key(k)).or_default();
        gained += usize::from(addresses.is_empty());
        addresses.push(base + k * PAGE);
    }
    shown.add(&slice.chunk, gained, 0);
    physical.views.insert(base, (len, slice));
    #[cfg(feature = "diagnostics")]
    remap_time(4, books);
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    commit_failed(len, &error, "committing physical pages failed");
    false
}

/// The host could not commit (or make a section of) `len` bytes of
/// physical pages: at the commit limit, the crash message says so; any
/// other error is warned of as `what`.
fn commit_failed(len: usize, error: &std::io::Error, what: &str) {
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "{what}");
    }
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it. The pages no longer shown, and those shown
/// again, are added to `shown`.
fn unmap(physical: &mut Physical, base: usize, len: usize, shown: &mut Shown) {
    #[cfg(feature = "diagnostics")]
    let began = std::time::Instant::now();
    #[cfg(feature = "diagnostics")]
    let _timed = Defer(move || {
        remap_time(1, began);
        remap_time_line();
    });
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        let mut lost = 0;
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                    lost += 1;
                }
            }
        }
        shown.add(&slice.chunk, 0, lost);
        victory_gpu::title_memory_remapped(at, size);
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        if at < base {
            map(physical, at, base - at, slice.clone(), shown);
        }
        if at + size > end {
            let rest = Slice {
                index: slice.index + (end - at) / PAGE,
                ..slice
            };
            map(physical, end, at + size - end, rest, shown);
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and what it uses: our process's commit without the
/// memory the graphics layer holds for the card. No tools memory: a retail
/// console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    // The system charges every byte the layer places in the card's memory
    // to this process's commit, one to one (measured on the RX 9070 XT:
    // 3 GiB of card buffers added 3 GiB of commit and nothing to the
    // working set). On the console those bytes are the title's own pages,
    // already counted once; here they are a second copy that is the
    // layer's, so they are left out.
    // shortcut: the layer's host copies in PC memory (shadows, upload
    // rings) stay counted as the title's: title memory opened as a heap
    // shows in the same use figure, so the two cannot be told apart there.
    let layer = victory_gpu::card_memory_in_use() as usize;
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage.saturating_sub(layer),
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th. (Before the layer's share was left out, "used" was the
    // whole commit: 27 GB in run 60's gameplay against a limit of 13.5 GB,
    // so a title budgeting its streaming from it believed itself far over.)
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            commit_mb = counters.PagefileUsage >> 20,
            layer_card_mb = layer >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

/// Runs its closure when dropped (diagnostics timing of a function with
/// several returns).
#[cfg(feature = "diagnostics")]
struct Defer<F: FnMut()>(F);

#[cfg(feature = "diagnostics")]
impl<F: FnMut()> Drop for Defer<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// One scatter call over consecutive addresses: pages out of
    /// order, an unmapped address (page number 0) between mapped ones, and an address
    /// apart from the others each show what the title asked for.
    #[test]
    fn a_scatter_of_many_pages_maps_each_address() {
        let mut pages = [0usize; 4];
        let mut n = 4;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let reserve = || {
            XMemVirtualAlloc(std::ptr::null_mut(), 6 * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE)
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        // Each page marked with its own byte, through a view of all four.
        assert_eq!(XMemMapPhysicalPages(b, 4, pages.as_ptr()), b);
        for k in 0..4 {
            // SAFETY: mapped read-write.
            unsafe { at(b, k).cast::<u8>().write(10 + k as u8) };
        }
        // a+0..a+2: pages 2, 0, 1 (out of order); a+3 unmapped; a+5: page 3.
        let addresses = [at(a, 0), at(a, 1), at(a, 2), at(a, 3), at(a, 5)];
        let scattered = [pages[2], pages[0], pages[1], 0, pages[3]];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 5, scattered.as_ptr()),
            1
        );
        // SAFETY: the mapped addresses are read-write.
        let read = |p: *mut c_void| unsafe { p.cast::<u8>().read() };
        assert_eq!([read(at(a, 0)), read(at(a, 1)), read(at(a, 2)), read(at(a, 5))], [12, 10, 11, 13]);
        assert_eq!(query(at(a, 3)).host.State, MEM_RESERVE);
        assert_eq!(query(at(a, 4)).host.State, MEM_RESERVE);
        // A write through the scatter's run shows at the page's other view.
        // SAFETY: mapped read-write.
        unsafe { at(a, 1).cast::<u8>().write(77) };
        assert_eq!(read(b), 77);
        let zeros = [0usize; 3];
        // Addresses that follow one another across two reservations (two
        // placeholders, which one view cannot span), in one call.
        let c = reserve();
        let d = XMemVirtualAlloc(at(c, 6), PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
        if d == at(c, 6) {
            let across = [at(c, 5), d];
            assert_eq!(XMemMapPhysicalPagesScatter(across.as_ptr(), 2, pages[..2].as_ptr()), 1);
            assert_eq!([read(at(c, 5)), read(d)], [77, 11]);
            assert_eq!(XMemMapPhysicalPagesScatter(across.as_ptr(), 2, zeros.as_ptr()), 1);
        }
        // Unmapping in one call.
        assert_eq!(XMemMapPhysicalPagesScatter(addresses.as_ptr(), 3, zeros.as_ptr()), 1);
        for k in 0..3 {
            assert_eq!(query(at(a, k)).host.State, MEM_RESERVE);
        }
        assert_eq!(read(at(a, 5)), 13, "the address apart keeps its page");
        assert_eq!(XMemMapPhysicalPages(b, 4, std::ptr::null()), b);
        assert_eq!(XMemMapPhysicalPagesScatter(addresses[4..].as_ptr(), 1, zeros.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// A pool of chunks of 4 pages on a PC with room and a Windows that
    /// opens sections as heaps.
    const ROOMY: Pool = Pool {
        chunk_pages: 4,
        room: all_the_room,
        build: windows_11,
        homes: || true,
        open_pct: 0,
    };

    fn all_the_room() -> Option<usize> {
        Some(usize::MAX)
    }

    fn windows_11() -> u32 {
        26100
    }

    /// `n` pages of one new allocation of the memory pool's.
    fn allocate_pool(n: usize, pool: Pool) -> Vec<usize> {
        let mut pages = vec![0; n];
        assert!(allocate_pages(&mut pages, Some(pool)));
        pages
    }

    /// The chunk of a page, and whether its section is made and committed.
    fn chunk_of(page: usize) -> (Arc<Chunk>, Option<bool>) {
        let chunk = physical()
            .pages
            .get(&page)
            .expect("known page")
            .chunk
            .clone();
        let made = chunk.section.get().map(|s| s.committed);
        (chunk, made)
    }

    /// With the memory pool, an allocation is chunks of their own, whose
    /// sections are made on the first map or write into each and committed
    /// whole: nothing before, and only the chunks touched.
    #[test]
    fn pool_chunks_are_made_on_the_first_map_or_write() {
        let pages = allocate_pool(10, ROOMY);
        let chunks: Vec<_> = [0, 4, 8].iter().map(|&i| chunk_of(pages[i])).collect();
        assert!(
            chunks.iter().all(|(_, made)| made.is_none()),
            "nothing made yet"
        );
        assert_eq!(
            [0, 1, 2].map(|c| chunks[c].0.bytes),
            [4 * PAGE, 4 * PAGE, 2 * PAGE]
        );
        // A map into the second chunk makes that one alone, committed.
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages[5..].as_ptr()), a);
        assert_eq!(chunk_of(pages[4]).1, Some(true));
        assert_eq!(chunk_of(pages[0]).1, None);
        assert_eq!(chunk_of(pages[8]).1, None);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(42) };
        assert_eq!(
            page_bytes(pages[5])[0],
            42,
            "the section shows through our views"
        );
        // A write into the first (DirectStorage's) makes it.
        assert!(write_physical(&pages[..1], 3, &[7]));
        assert_eq!(chunk_of(pages[0]).1, Some(true));
        assert_eq!(page_bytes(pages[0])[3], 7);
        assert_eq!(XMemMapPhysicalPages(a, 2, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(10, pages.as_ptr()), 1);
    }

    /// Short of commit room, or on a Windows before build 20348, a chunk is
    /// made reserved and committed as views show it, as without the pool.
    #[test]
    fn a_pool_chunk_short_of_commit_or_on_old_windows_stays_reserved() {
        let short = Pool {
            room: || Some(4 * PAGE),
            ..ROOMY
        };
        let old = Pool {
            build: || 19045,
            ..ROOMY
        };
        assert_eq!(short.kind(4 * PAGE), victory_gpu::PoolChunk::ShortOfCommit);
        assert_eq!(old.kind(4 * PAGE), victory_gpu::PoolChunk::OldWindows);
        assert_eq!(ROOMY.kind(4 * PAGE), victory_gpu::PoolChunk::Committed);
        let unknown = Pool {
            room: || None,
            ..ROOMY
        };
        assert_eq!(unknown.kind(PAGE), victory_gpu::PoolChunk::ShortOfCommit);
        let homeless = Pool {
            homes: || false,
            ..ROOMY
        };
        assert_eq!(homeless.kind(PAGE), victory_gpu::PoolChunk::NoHomes);
        // This PC's own answers can be had.
        assert!(windows_build() >= 10240, "the kernel's build number");
        assert!(commit_room().is_some());
        for pool in [short, old, homeless] {
            let pages = allocate_pool(3, pool);
            let a = XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!a.is_null());
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
            assert_eq!(chunk_of(pages[0]).1, Some(false), "reserved");
            assert_eq!(
                query(a).host.State,
                MEM_COMMIT,
                "committed as the view shows it"
            );
            // SAFETY: mapped read-write.
            unsafe { a.wrapping_byte_add(2 * PAGE).cast::<u8>().write(9) };
            assert_eq!(page_bytes(pages[2])[0], 9);
            assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
            assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        }
    }

    /// The chunk sizes the switch takes are whole pages: 16 per MiB.
    #[test]
    fn chunk_sizes_are_whole_pages() {
        for mb in [16u64, 32, 64, 128, 256] {
            assert_eq!(chunk_pages(mb << 20), mb as usize * 16);
        }
    }

    /// Pages that follow on across a chunk's end are two runs, and two views
    /// when mapped: no run, view or page window crosses a chunk, whatever
    /// the chunk's size (down to two pages, and the smallest the switch
    /// takes, 16 MiB).
    #[test]
    fn a_run_never_crosses_a_chunk() {
        for chunk in [2, 4, chunk_pages(16 << 20)] {
            let n = chunk + 2;
            let pages = allocate_pool(
                n,
                Pool {
                    chunk_pages: chunk,
                    ..ROOMY
                },
            );
            let lengths = |offset, len| {
                runs(&pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
            };
            assert_eq!(lengths(0, n * PAGE), Some(vec![chunk, 2]));
            assert_eq!(lengths((chunk - 1) * PAGE, 2 * PAGE), Some(vec![1, 1]));
            assert!(page_window(&pages, (chunk - 1) * PAGE, 2 * PAGE).is_none());
            assert!(page_window(&pages, chunk * PAGE, 2 * PAGE).is_some());
            let a = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!a.is_null());
            assert_eq!(XMemMapPhysicalPages(a, n, pages.as_ptr()), a);
            {
                let physical = physical();
                let views: Vec<usize> = physical
                    .views
                    .range(a as usize..a as usize + n * PAGE)
                    .map(|(_, v)| v.0)
                    .collect();
                assert_eq!(views, vec![chunk * PAGE, 2 * PAGE]);
            }
            // SAFETY: mapped read-write, across the two views.
            unsafe {
                a.wrapping_byte_add(chunk * PAGE - 1)
                    .cast::<[u8; 2]>()
                    .write_unaligned([1, 2])
            };
            assert_eq!(page_bytes(pages[chunk - 1])[PAGE - 1], 1);
            assert_eq!(page_bytes(pages[chunk])[0], 2);
            assert_eq!(XMemMapPhysicalPages(a, n, std::ptr::null()), a);
            assert_eq!(XMemFreePhysicalPages(n, pages.as_ptr()), 1);
        }
    }

    /// With the pool's share, a chunk counts the pages its views show, each
    /// once however many views show it, and the GPU side is told when that
    /// passes the share either way; without the share it is told nothing.
    #[test]
    fn the_mapped_share_of_a_chunk_is_told_when_it_passes() {
        let half = Pool {
            open_pct: 50,
            ..ROOMY
        };
        let pages = allocate_pool(4, half);
        let reserve = |n: usize| {
            let a = XMemVirtualAlloc(
                std::ptr::null_mut(),
                n * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            );
            assert!(!a.is_null());
            a
        };
        let (a, b) = (reserve(2), reserve(1));
        assert_eq!(XMemMapPhysicalPages(a, 1, pages.as_ptr()), a);
        let section = chunk_of(pages[0]).0.section.get().expect("made").handle as usize;
        let enough = || victory_gpu::title_memory_pool_chunk_mapped_enough(section);
        assert_eq!(enough(), Some(false), "a quarter");
        // The same page at a second place: still one page shown.
        assert_eq!(XMemMapPhysicalPages(b, 1, pages.as_ptr()), b);
        assert_eq!(enough(), Some(false), "a page counts once");
        assert_eq!(chunk_of(pages[0]).0.mapped.load(Ordering::Relaxed), 1);
        // A second page: half.
        let second = a.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 1, pages[1..].as_ptr()), second);
        assert_eq!(enough(), Some(true), "half");
        // The first page's first view gone: still shown at `b`.
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(enough(), Some(true), "still shown elsewhere");
        assert_eq!(XMemMapPhysicalPages(b, 1, std::ptr::null()), b);
        assert_eq!(enough(), Some(false), "a quarter again");
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(chunk_of(pages[1]).0.mapped.load(Ordering::Relaxed), 0);
        // A view cut in two, or a page mapped again where it is, keeps the
        // share without telling that it dropped below in between.
        let d = reserve(3);
        assert_eq!(XMemMapPhysicalPages(d, 3, pages.as_ptr()), d);
        assert_eq!(enough(), Some(true), "three quarters");
        let told = SHARE_TOLD.load(Ordering::Relaxed);
        let middle = d.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(chunk_of(pages[0]).0.mapped.load(Ordering::Relaxed), 2);
        assert_eq!(XMemMapPhysicalPages(d, 1, pages.as_ptr()), d);
        assert_eq!(chunk_of(pages[0]).0.mapped.load(Ordering::Relaxed), 2);
        let now = SHARE_TOLD.load(Ordering::Relaxed);
        assert_eq!(now, told, "never below in between");
        assert_eq!(enough(), Some(true), "half");
        assert_eq!(XMemMapPhysicalPages(d, 3, std::ptr::null()), d);
        assert_eq!(enough(), Some(false), "none");
        // Without the share nothing is counted or told.
        let other = allocate_pool(2, ROOMY);
        let c = reserve(2);
        assert_eq!(XMemMapPhysicalPages(c, 2, other.as_ptr()), c);
        let (chunk, _) = chunk_of(other[0]);
        let section = chunk.section.get().expect("made").handle as usize;
        assert_eq!(chunk.mapped.load(Ordering::Relaxed), 0);
        assert_eq!(
            victory_gpu::title_memory_pool_chunk_mapped_enough(section),
            Some(false)
        );
        drop(chunk);
        assert_eq!(XMemMapPhysicalPages(c, 2, std::ptr::null()), c);
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(2, other.as_ptr()), 1);
    }

    /// Once its pages are freed and its views unmapped, a chunk's section
    /// is closed and the GPU side lets the chunk go (and with it the heap it
    /// opened, which held the section and its commit charge).
    #[test]
    fn a_freed_and_unmapped_chunk_is_let_go_on_the_gpu_side() {
        let pages = allocate_pool(4, ROOMY);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 4, pages.as_ptr()), a);
        let section = {
            let (chunk, made) = chunk_of(pages[0]);
            assert_eq!(made, Some(true));
            chunk.section.get().expect("made").handle as usize
        };
        assert_eq!(
            victory_gpu::title_memory_pool_chunk_known(section),
            Some(false),
            "known, not opened (no home was wanted)"
        );
        // The pages freed first: the view still holds the chunk.
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
        assert!(victory_gpu::title_memory_pool_chunk_known(section).is_some());
        assert_eq!(XMemMapPhysicalPages(a, 4, std::ptr::null()), a);
        assert_eq!(
            victory_gpu::title_memory_pool_chunk_known(section),
            None,
            "the last view gone: the chunk is let go"
        );
    }
}
