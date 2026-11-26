//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled (but see the directory switch below);
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (one stage's table of structure addresses, another's
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.
//!
//! A home of the memory core (`crate::home`) is numbered as a copy is: its
//! buffer is the one placed over the title's pages, which starts up to
//! 64 KiB before the title's first byte (or, for a home in a chunk of the
//! pool, at the chunk's start), so its view and its entries' offsets count
//! from the buffer's start ([`Shadow::offset`]).
//!
//! ## The directory switch (`VICTORY_MEMORY_DIRECTORY=1`)
//!
//! With the memory core on (`VICTORY_MEMORY_CORE=1`), the third slice of the
//! core: the directory stops being a list of copies only filled.
//!
//! - **Pool pages are resolved page by page.** With the memory pool on
//!   (`VICTORY_MEMORY_POOL=1`, `crate::pool`), a page of view memory that
//!   is one committed, opened chunk's page has the *chunk* as its number:
//!   the chunk's buffer (placed over all of it) has its view in the slot
//!   run, and the entry's offset is the page's offset in the chunk. So
//!   view memory a buffer or window cannot be one home for (views of
//!   several runs or several chunks, `crate::home::Why::Scattered`) is read
//!   and written by raw-address shaders in place, page by page, with no
//!   copy, no refresh and no copy-back; a 4 MiB cell of such pages is filled
//!   at once, as a window's pages are. View memory bound as *one* view
//!   (a root view or a descriptor over several runs) still has a copy: one
//!   buffer view cannot be several places (counted on the line, until R6).
//!   A page such a copy holds bytes of is the copy's for raw-address
//!   shaders too, as without the switch, so that both kinds of use see the
//!   same bytes within a submission; a copy made after a page got its
//!   chunk's entry takes the page at the next dispatch that makes its pages
//!   ready ([`copy_made`]).
//! - **Entries change on remap.** Mapping other pages at a chunk page's
//!   address ([`remapped`]: `xmem`'s maps and unmaps, the queues' page
//!   mappings) has its entry looked at again: the new chunk and offset, or
//!   empty if the page is no chunk's any more.
//! - **In queue order.** The table shaders read is changed only by the
//!   card, by a copy at the head of each submission ([`before_submission`])
//!   of the entries that changed since the last one, from a table of ours on
//!   the CPU. Every list of a submission sees the table as it was when the
//!   submission was sent, whole: none sees a remap half done, and a change
//!   made while a submission runs reaches only the next.
//! - **Raw writes** into chunk pages are not seen by the guards: after the
//!   wait of a submission one of whose lists could store through the
//!   directory, the copies over the chunk pages it names at other views of
//!   those pages, and the placed textures there, take them again
//!   (`crate::home::directory_written`).
//!   shortcut: all of the chunk pages the directory names, as every
//!   numbered copy is declared written today; the translator's dirty masks
//!   (R4) would bound it.
//! - **Chunks that go** (`crate::pool::title_memory_pool_chunk_gone`) lose
//!   their number at the next list that makes its pages ready or the next
//!   submission's head ([`chunk_gone`]): the directory holds the chunk's
//!   buffer, and so its heap, its section and their commit charge, only
//!   until then. shortcut: a number let go is not given again, so chunks
//!   that come and go use up numbers the copies share.
//! - **A number is only taken where its view is written.** A remap looked
//!   at in a submission's head gives a page a chunk's number only if the
//!   chunk has one already; else the page is left empty and numbered at the
//!   next list that makes its pages ready. Every slot of a view heap's run
//!   is a null view from the run's start, so a shader of a list whose heap
//!   has not yet had a newer number's view written reads zeros there (and
//!   its stores are dropped) rather than whatever a reused slot held.
//!
//! shortcut: a chunk page's entry is that page only. An access of several
//! dwords that crosses a 64 KiB page's end at a chunk page reads on in the
//! chunk, not at the page the title mapped next; copies are whole, so with
//! them it did not happen. Aligned structure accesses do not cross.
//!
//! The switch needs every submission waited for (no `VICTORY_LATE_WAIT`,
//! no relay), or R5 (`crate::host::r5`): each copy at a submission's head
//! reads a staging buffer and runs a list that only that submission uses
//! until it has finished ([`crate::host::Ring`]), so a flight still in the
//! air when the next submission's entries are laid out keeps its own, and on
//! the one host queue every flight sees the table as it was when it was
//! sent.
//!
//! shortcuts with R5: [`after_wait`] tells the copies over chunk pages as
//! the table maps them at the flight's landing, not as the flight saw them,
//! so a remap between its send and its landing can leave copies at the old
//! views untold; and [`ready`], on the title's recording threads, makes the
//! view of a number let go null while flights in the air may still reach
//! that number through their own table (their reads there give zeros and
//! their stores are dropped, as for a protection and for one submission
//! without R5, but now for every flight in the air).
//!
//! shortcut: a home's pages are numbered when a dispatch names them or a
//! fault asks, not when the home is made.

use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::descriptor::host::HostHeap;
use crate::home::Why;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
pub(crate) const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// The pages of a [`CELL`].
const CELL_PAGES: u64 = CELL / PAGE;

/// `VICTORY_MEMORY_DIRECTORY=1` with the memory core on, every submission
/// waited for or R5 on (`crate::host::r5`). Read once.
pub(crate) fn directory_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let on = *ON.get_or_init(|| {
        let asked = std::env::var_os("VICTORY_MEMORY_DIRECTORY").is_some_and(|v| v == "1");
        let on = asked && crate::home::on() && (!crate::host::late() || crate::host::r5());
        if on {
            tracing::info!(
                "memory directory on (VICTORY_MEMORY_DIRECTORY=1): raw-address shaders reach pool pages page by page in place, and the directory changes on remap, at the head of each submission"
            );
        } else if asked {
            tracing::warn!(
                "VICTORY_MEMORY_DIRECTORY=1 does nothing without VICTORY_MEMORY_CORE=1, or with late waits or the relay unless VICTORY_R5=1"
            );
        }
        on
    });
    on || test_on()
}

// The tests' switch, per thread, as `crate::home`'s.
#[cfg(test)]
thread_local! {
    static TEST_ON: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn test_on() -> bool {
    TEST_ON.with(std::cell::Cell::get) && crate::home::on()
}

#[cfg(not(test))]
fn test_on() -> bool {
    false
}

/// Sets the directory switch for this thread (tests).
#[cfg(test)]
pub(crate) fn set_test(on: bool) {
    TEST_ON.with(|t| t.set(on));
}

/// What a number of the directory names.
enum Numbered {
    /// A host copy or a home ([`crate::shadow`]): its view is its buffer
    /// from the first byte, as far as its title bytes reach.
    Copy(Arc<Shadow>),
    /// With the directory on: a chunk of the memory pool (by its section's
    /// handle), its view the buffer placed over all of it.
    Chunk {
        section: usize,
        buffer: ID3D12Resource,
        bytes: u64,
    },
}

impl Numbered {
    fn shadow(&self) -> Option<&Arc<Shadow>> {
        match self {
            Numbered::Copy(shadow) => Some(shadow),
            Numbered::Chunk { .. } => None,
        }
    }
}

/// With the directory on: the table as the next submission's lists are to
/// see it, kept on the CPU (the shaders' table takes it at the head of each
/// submission, [`before_submission`]), and what that copy needs.
struct Next {
    /// As the shaders' table: two dwords per page.
    entries: Vec<u32>,
    /// The pages changed since the last copy.
    dirty: BTreeSet<u32>,
    /// What the copies at submissions' heads use, each taken again once its
    /// submission has finished.
    copiers: crate::host::Ring<Copier>,
}

/// What one copy at a submission's head uses ([`record`]): the changed
/// entries laid out for it (buffer, mapping, bytes), and the list that
/// copies them, with its allocator.
struct Copier {
    staging: Option<(ID3D12Resource, *mut u8, u64)>,
    list: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Number `n` is `copies[n - 1]`; `None` for a home the memory core let
    /// go ([`forget`]): its number is not given again.
    copies: Vec<Option<Numbered>>,
    /// The numbers let go (0-based), in the order they were.
    forgotten: Vec<u32>,
    /// Per host view heap (by address): the first slot of its run, how many
    /// copies have their view there, and how many of `forgotten` have had
    /// their view there made null.
    heaps: HashMap<usize, (u32, u32, usize)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// With the directory on: see [`Next`]. Made from the shaders' table
    /// the first time an entry changes with the switch on.
    next: Option<Next>,
    /// With the directory on: the pages whose entry is a chunk's, looked
    /// at again when their mapping changes ([`remapped`]).
    chunked: BTreeSet<u32>,
    /// With the directory on: pages a remap left empty because their chunk
    /// had no number yet ([`Memory::apply_remaps`]), numbered at the next
    /// [`ready`].
    wanted: Vec<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// Title memory whose mapping changed since the directory last looked
/// ([`remapped`]), as (address, bytes). Its own lock, taken with no other
/// held: the runtime tells it while it changes a mapping.
static REMAPS: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Whether a chunk was ever looked up for an entry ([`Memory::chunk`]):
/// until then no remap or gone chunk is noted. Set before the pool's
/// lookup, so that a chunk that goes after it is noted.
static CHUNKED_ANY: AtomicBool = AtomicBool::new(false);

/// Title memory host copies were made over since the directory last looked
/// ([`copy_made`]), as (address, bytes). Its own lock, taken with no other
/// held but the shadow tables'.
static COPIED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Whether the CPU's table ([`Next`]) was ever made: until then a
/// submission has nothing to copy.
static NEXT_MADE: AtomicBool = AtomicBool::new(false);

/// A chunk of the pool that went ([`chunk_gone`]): its section's handle and
/// the buffer that was placed over it, held until the directory lets go of
/// its number (so that no other buffer can be made at its address before).
struct Gone(usize, ID3D12Resource);

// SAFETY: a COM reference moved between threads only to be compared and
// released; D3D12 objects are free-threaded.
unsafe impl Send for Gone {}

/// The chunks that went since the directory last looked ([`chunk_gone`]).
/// Its own lock, taken with no other held but [`MEMORY`].
static GONE: Mutex<Vec<Gone>> = Mutex::new(Vec::new());

/// The directory's counters ([`DIR`]), since the start.
mod at {
    /// Entries filled: with a chunk of the pool, with a home, with a copy.
    pub(super) const CHUNK_PAGES: usize = 0;
    pub(super) const HOME_PAGES: usize = 1;
    pub(super) const COPY_PAGES: usize = 2;
    /// Chunks given a number.
    pub(super) const CHUNKS: usize = 3;
    /// Chunk pages looked at again after a remap, of them with an entry
    /// that changed.
    pub(super) const REMAP_LOOKED: usize = 4;
    pub(super) const REMAP_CHANGED: usize = 5;
    /// Submissions whose head copied entries, and the entries copied.
    pub(super) const UPDATES: usize = 6;
    pub(super) const UPDATED: usize = 7;
    /// Submissions after whose wait the copies over chunk pages were told.
    pub(super) const TOLD: usize = 8;
    /// Changes written straight into the shaders' table, out of queue order,
    /// as the host could not make or record the copy.
    pub(super) const STRAIGHT: usize = 9;
    /// Asks of pages that are not view memory that a copy (not a home)
    /// answered.
    pub(super) const COPY_OTHER: usize = 10;
    /// Asks of chunk pages a copy already held bytes of, and chunk pages
    /// whose entry became a copy's when one was made over them later.
    pub(super) const COPY_OVER: usize = 11;
    pub(super) const SWITCHED: usize = 12;
    /// Chunks that went whose number was let go ([`super::chunk_gone`]).
    pub(super) const CHUNKS_GONE: usize = 13;
    /// Pages a remap left empty for want of a number, numbered at the next
    /// list's dispatch.
    pub(super) const REMAP_WANTED: usize = 14;
    pub(super) const ALL: usize = 15;
}

static DIR: [AtomicU64; at::ALL] = [const { AtomicU64::new(0) }; at::ALL];

/// Asks of view memory a chunk could not answer (and a copy did), by
/// `crate::home::Why` ([`crate::home::why_index`]).
static COPY_WHY: [AtomicU64; crate::home::WHY_COUNT] =
    [const { AtomicU64::new(0) }; crate::home::WHY_COUNT];

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

/// What [`Memory::chunk`] found for a page.
enum Chunked {
    /// The chunk's number (0-based) and the page's offset in it.
    Yes(u32, u32),
    /// Not view memory: the copy path's.
    NotView,
    /// A chunk's page, but a host copy holds bytes of it (a root view or a
    /// descriptor over memory of several places): shaders that reach the
    /// page by address use that copy too, so that both see the same bytes.
    Copied,
    /// View memory no chunk answers for: a copy, and why.
    No(Why),
    /// A chunk's page whose chunk has no number, when none may be given
    /// now ([`Memory::apply_remaps`]).
    Unnumbered,
}

/// The bytes of a copy's buffer its view covers: as far as its title bytes
/// reach (a home's start lies past its buffer's, [`Shadow::offset`]).
fn view_bytes(shadow: &Shadow) -> u64 {
    let (start, len) = shadow.span();
    shadow.offset(start) + len
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::label(&buffer, "title memory by address", dwords * 4);
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            forgotten: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            next: None,
            chunked: BTreeSet::new(),
            wanted: Vec::new(),
        })
    }

    /// The entry of `page` as the next submission sees it: its number and
    /// offset.
    fn get(&self, page: u64) -> (u32, u32) {
        let i = page as usize * 2;
        match &self.next {
            Some(next) => (next.entries[i], next.entries[i + 1]),
            // SAFETY: inside the directory's mapping.
            None => unsafe {
                (
                    self.directory.1.add(i).read_volatile(),
                    self.directory.1.add(i + 1).read_volatile(),
                )
            },
        }
    }

    /// Sets the entry of `page`. With the directory on, in the CPU's table,
    /// for the next submission's head ([`before_submission`]); else in the
    /// shaders' table at once, so that a shader that sees a number sees its
    /// offset: a new entry's offset first, an emptied entry's number first.
    fn set(&mut self, page: u64, number: u32, offset: u32) {
        let i = page as usize * 2;
        let on = directory_on();
        if on && self.next.is_none() {
            let dwords = MEMORY_PAGES as usize * 2;
            // SAFETY: the directory's mapping holds that many dwords; no
            // copy into it runs (they are made only with this table).
            let entries = unsafe { std::slice::from_raw_parts(self.directory.1, dwords) }.to_vec();
            self.next = Some(Next {
                entries,
                dirty: BTreeSet::new(),
                copiers: crate::host::Ring::new(),
            });
            NEXT_MADE.store(true, Ordering::Relaxed);
        }
        if let Some(next) = &mut self.next {
            next.entries[i] = number;
            next.entries[i + 1] = offset;
            if on {
                next.dirty.insert(page as u32);
                return;
            }
        }
        // SAFETY: inside the directory's mapping.
        unsafe {
            if number == 0 {
                self.directory.1.add(i).write_volatile(number);
                self.directory.1.add(i + 1).write_volatile(offset);
            } else {
                self.directory.1.add(i + 1).write_volatile(offset);
                self.directory.1.add(i).write_volatile(number);
            }
        }
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        if self.get(page).0 != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // With the directory on, view memory is a chunk's page if it can be.
        // What the copy path counts the ask as: why a chunk did not answer,
        // or (`counted_over`) already counted as a chunk page a copy held.
        let mut view_why = None;
        let mut counted_over = false;
        if directory_on() {
            match self.chunk(page, true) {
                Chunked::Yes(n, offset) => {
                    self.set_chunk(page, n, offset);
                    self.chunk_cell(page);
                    return;
                }
                Chunked::NotView => {}
                Chunked::Copied => {
                    DIR[at::COPY_OVER].fetch_add(1, Ordering::Relaxed);
                    counted_over = true;
                }
                Chunked::No(why) => view_why = Some(why),
                // Not given with numbers allowed.
                Chunked::Unnumbered => return,
            }
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let counted = if shadow.home_kind().is_some() {
            at::HOME_PAGES
        } else {
            match view_why {
                // Counted as such above.
                _ if counted_over => {}
                Some(why) => {
                    if let Some(i) = crate::home::why_index(why) {
                        COPY_WHY[i].fetch_add(1, Ordering::Relaxed);
                    }
                }
                None => {
                    DIR[at::COPY_OTHER].fetch_add(1, Ordering::Relaxed);
                }
            }
            at::COPY_PAGES
        };
        let n = match self.copies.iter().position(|c| {
            c.as_ref()
                .and_then(Numbered::shadow)
                .is_some_and(|c| Arc::ptr_eq(c, &shadow))
        }) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(Some(Numbered::Copy(shadow.clone())));
                crate::frame_budget::directory_numbers(self.copies.len());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    if self.get(at / PAGE).0 == 0 {
                        self.set(at / PAGE, n as u32 + 1, shadow.offset(at) as u32);
                        DIR[counted].fetch_add(1, Ordering::Relaxed);
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                no_number_left();
                return;
            }
        };
        shadow.handed(crate::shadow::Hand::Memory, start);
        if self.get(page).0 == 0 {
            DIR[counted].fetch_add(1, Ordering::Relaxed);
        }
        self.set(page, n as u32 + 1, shadow.offset(start) as u32);
    }

    /// With the directory on: whether `page` is a page of a committed,
    /// opened chunk of the pool at one view, and if so the chunk's number
    /// (given now if it has none and `numbers`, else
    /// [`Chunked::Unnumbered`]) and the page's offset in it. Opens the chunk
    /// if no home was wanted in it before (`crate::pool::home`).
    fn chunk(&mut self, page: u64, numbers: bool) -> Chunked {
        use crate::inplace::{Cover, cover};
        let (start, len) = ((page * PAGE) as usize, PAGE as usize);
        if cover(start, len) == Cover::None {
            return Chunked::NotView;
        }
        if !crate::shadow::copies_over(start as u64, (start + len) as u64).is_empty() {
            return Chunked::Copied;
        }
        if !crate::pool::on() {
            return Chunked::No(Why::View);
        }
        if let Some(why) = crate::home::ruled_out() {
            return Chunked::No(why);
        }
        // The mapping held still from the look at the views to the entry
        // (a remap meanwhile is told after it, [`remapped`]).
        let Some(_reading) = crate::watch::title_memory_reading_now(start, len) else {
            return Chunked::No(Why::Remapping);
        };
        let (section, offset) = match cover(start, len) {
            Cover::Run(section, offset) => (section, offset),
            Cover::None => return Chunked::NotView,
            Cover::Holes | Cover::Scattered => return Chunked::No(Why::Scattered),
        };
        CHUNKED_ANY.store(true, Ordering::Relaxed);
        let placed = match crate::pool::home(section, offset, PAGE) {
            Ok(placed) => placed,
            Err(why) => return Chunked::No(why),
        };
        let Ok(within) = u32::try_from(placed.bias) else {
            return Chunked::No(Why::PastChunk);
        };
        let raw = placed.buffer.as_raw();
        let found = self.copies.iter().position(|c| {
            matches!(c, Some(Numbered::Chunk { section: s, buffer, .. })
                if *s == section && buffer.as_raw() == raw)
        });
        let n = match found {
            Some(n) => n,
            None if !numbers => return Chunked::Unnumbered,
            None if self.copies.len() < COPIES as usize => {
                // SAFETY: a live buffer.
                let bytes = unsafe { placed.buffer.GetDesc() }.Width;
                self.copies.push(Some(Numbered::Chunk {
                    section,
                    buffer: placed.buffer,
                    bytes,
                }));
                DIR[at::CHUNKS].fetch_add(1, Ordering::Relaxed);
                crate::frame_budget::directory_numbers(self.copies.len());
                self.copies.len() - 1
            }
            None => {
                no_number_left();
                return Chunked::No(Why::Refused);
            }
        };
        Chunked::Yes(n as u32, within)
    }

    /// `page`'s entry is chunk `n`'s (0-based) at `offset`.
    fn set_chunk(&mut self, page: u64, n: u32, offset: u32) {
        if self.get(page) != (n + 1, offset) {
            self.set(page, n + 1, offset);
            DIR[at::CHUNK_PAGES].fetch_add(1, Ordering::Relaxed);
        }
        self.chunked.insert(page as u32);
    }

    /// The other pages of `page`'s 4 MiB cell that are chunk pages and have
    /// no entry get theirs, as a window's pages would.
    fn chunk_cell(&mut self, page: u64) {
        let first = page / CELL_PAGES * CELL_PAGES;
        for other in first..first + CELL_PAGES {
            if other == page
                || other >= u64::from(MEMORY_PAGES) - 1
                || self.get(other).0 != 0
                || self.refused.contains(&other)
            {
                continue;
            }
            if let Chunked::Yes(n, offset) = self.chunk(other, true) {
                self.set_chunk(other, n, offset);
            }
        }
    }

    /// The chunk pages whose mapping changed since the last look
    /// ([`remapped`]) get their entry anew: the chunk and offset they show
    /// now, or none. A page whose mapping is changing right now is left
    /// empty and looked at again next time. Done at a submission's head,
    /// after its lists made their pages ready: no number is given here (no
    /// view heap of those lists would have its view), so a page whose chunk
    /// has none yet is left empty and numbered at the next [`ready`].
    fn apply_remaps(&mut self) {
        let ranges = std::mem::take(&mut *REMAPS.lock().unwrap_or_else(|e| e.into_inner()));
        let mut again = Vec::new();
        for (at, len) in ranges {
            let last = u64::from(MEMORY_PAGES) - 1;
            let from = (at / PAGE).min(last) as u32;
            let to = at.saturating_add(len).div_ceil(PAGE).min(last) as u32;
            let pages: Vec<u32> = self.chunked.range(from..to).copied().collect();
            for page in pages {
                let page = u64::from(page);
                DIR[at::REMAP_LOOKED].fetch_add(1, Ordering::Relaxed);
                let before = self.get(page);
                let (now, keep) = match self.chunk(page, false) {
                    Chunked::Yes(n, offset) => ((n + 1, offset), true),
                    Chunked::No(Why::Remapping) => {
                        again.push((page * PAGE, PAGE));
                        ((0, 0), true)
                    }
                    Chunked::Unnumbered => {
                        self.wanted.push(page);
                        DIR[at::REMAP_WANTED].fetch_add(1, Ordering::Relaxed);
                        ((0, 0), false)
                    }
                    Chunked::No(_) | Chunked::NotView | Chunked::Copied => ((0, 0), false),
                };
                if now != before {
                    self.set(page, now.0, now.1);
                    DIR[at::REMAP_CHANGED].fetch_add(1, Ordering::Relaxed);
                }
                if !keep {
                    self.chunked.remove(&(page as u32));
                }
            }
        }
        if !again.is_empty() {
            REMAPS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .extend(again);
        }
    }

    /// The chunk pages a host copy was made over since the last look
    /// ([`copy_made`]) take that copy instead ([`Chunked::Copied`]): their
    /// entries are emptied and the pages numbered anew, now on the copy path.
    /// Done when a list makes its pages ready ([`ready`]), so that the list
    /// brings the copy up to date and declares it as every numbered copy.
    /// shortcut: a list recorded before, in the same submission, reads the
    /// copy through the directory without having done so (as with any copy
    /// numbered after a list's last dispatch).
    fn apply_copies(&mut self) {
        let ranges = std::mem::take(&mut *COPIED.lock().unwrap_or_else(|e| e.into_inner()));
        for (at, len) in ranges {
            let last = u64::from(MEMORY_PAGES) - 1;
            let from = (at / PAGE).min(last) as u32;
            let to = at.saturating_add(len).div_ceil(PAGE).min(last) as u32;
            let pages: Vec<u32> = self.chunked.range(from..to).copied().collect();
            for page in pages {
                let start = u64::from(page) * PAGE;
                if crate::shadow::copies_over(start, start + PAGE).is_empty() {
                    continue;
                }
                self.chunked.remove(&page);
                self.set(u64::from(page), 0, 0);
                DIR[at::SWITCHED].fetch_add(1, Ordering::Relaxed);
                self.map(start);
            }
        }
    }

    /// The chunks that went since the last look ([`chunk_gone`]) lose their
    /// numbers: their entries are emptied, and their views in the heaps'
    /// runs are made null at those heaps' next [`ready`] (lists that hold
    /// the buffer keep it while they run). The buffers are released here:
    /// a buffer's release takes no lock of ours.
    fn forget_gone_chunks(&mut self) {
        let gone = std::mem::take(&mut *GONE.lock().unwrap_or_else(|e| e.into_inner()));
        for Gone(section, buffer) in &gone {
            let raw = buffer.as_raw();
            let found = self.copies.iter().position(|c| {
                matches!(c, Some(Numbered::Chunk { section: s, buffer, .. })
                    if s == section && buffer.as_raw() == raw)
            });
            let Some(n) = found else {
                continue;
            };
            let number = n as u32 + 1;
            let pages: Vec<u32> = (self.chunked.iter().copied())
                .filter(|&page| self.get(u64::from(page)).0 == number)
                .collect();
            for page in pages {
                self.set(u64::from(page), 0, 0);
                self.chunked.remove(&page);
            }
            self.copies[n] = None;
            self.forgotten.push(n as u32);
            DIR[at::CHUNKS_GONE].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The host copies noted since the last [`ready`] ([`copy_made`]) that
    /// lie over no chunk page are dropped from the queue, so that it does
    /// not grow while no raw-address shader runs: only those over chunk
    /// pages are looked at by [`Memory::apply_copies`].
    fn prune_copies(&self) {
        let mut copied = COPIED.lock().unwrap_or_else(|e| e.into_inner());
        let last = u64::from(MEMORY_PAGES) - 1;
        copied.retain(|&(at, len)| {
            let from = (at / PAGE).min(last) as u32;
            let to = at.saturating_add(len).div_ceil(PAGE).min(last) as u32;
            self.chunked.range(from..to).next().is_some()
        });
        // A copy made anew each frame over the same bytes: once.
        copied.sort_unstable();
        copied.dedup();
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

fn no_number_left() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            COPIES,
            "title memory by address: no number left for another copy; its pages read zeros"
        )
    });
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    // After the release at the title's end the directory is not made again:
    // the draw is dropped, as no submission runs it. Asked under the lock
    // the release takes the directory with ([`release_all`], after the flag
    // is set), so a draw that passed it is done with the directory first.
    if crate::release::released() {
        return None;
    }
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    if directory_on() {
        m.forget_gone_chunks();
        m.apply_copies();
        // Only as a chunk's page: one the title remapped to other memory
        // since gets no host copy unless a shader names it.
        for page in std::mem::take(&mut m.wanted) {
            if m.get(page).0 == 0
                && let Chunked::Yes(n, offset) = m.chunk(page, true)
            {
                m.set_chunk(page, n, offset);
                m.chunk_cell(page);
            }
        }
    }
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made, nulled) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            // All null at first: a number made after this heap's last
            // `ready` (by another list, or at a submission's head) may be
            // read through a slot whose view was not written yet, and a
            // reused slot may hold a view of a buffer that is gone.
            let base = heap.dynamic(COPIES + 1)?;
            for slot in base..=base + COPIES {
                heap.null_raw_uav_at(slot);
            }
            (base, 0, 0)
        }
    };
    for (n, numbered) in m.copies.iter().enumerate().skip(made as usize) {
        let slot = base + 1 + n as u32;
        match numbered {
            Some(Numbered::Copy(shadow)) => {
                heap.raw_uav_at(slot, &shadow.buffer, 0, view_bytes(shadow));
            }
            Some(Numbered::Chunk { buffer, bytes, .. }) => heap.raw_uav_at(slot, buffer, 0, *bytes),
            None => heap.null_raw_uav_at(slot),
        }
    }
    // The view of a copy let go since holds its buffer (and so a home's
    // heap): made null. The directory names it no more.
    for &n in m.forgotten.iter().skip(nulled) {
        if n < made {
            heap.null_raw_uav_at(base + 1 + n);
        }
    }
    m.heaps
        .insert(key, (base, m.copies.len() as u32, m.forgotten.len()));
    // Each copy once per list, whichever list numbered it: the builder alone
    // dispatches hundreds of times a frame. The title records its lists on
    // several threads, so a copy can be numbered by another list (or by the
    // frame's fault notes) after this list's first dispatch: this list's
    // shaders reach it through the directory all the same, and a copy it
    // has not brought up to date, or not declared written, would be read as
    // it stands and its stores never looked for. Copies are only ever
    // added, so the ones a list has not taken yet are the last ones.
    const READ: usize = usize::MAX - 0x6d;
    const READ_COPY: usize = usize::MAX - 0x1_0000;
    const WRITE_COPY: usize = usize::MAX - 0x2_0000;
    let first_read = list.first_use(READ);
    for (n, numbered) in m.copies.iter().enumerate().rev() {
        if !list.first_use(READ_COPY - n) {
            break;
        }
        match numbered {
            Some(Numbered::Copy(shadow)) => {
                let (start, len) = shadow.span();
                shadow.refresh(list, start, len);
            }
            // The title's pages themselves: nothing to bring up to date;
            // the chunk's heap stays while the list may run.
            Some(Numbered::Chunk { buffer, .. }) => list.keep(buffer),
            None => {}
        }
    }
    if writes {
        for (n, numbered) in m.copies.iter().enumerate().rev() {
            if !list.first_use(WRITE_COPY - n) {
                break;
            }
            match numbered {
                Some(Numbered::Copy(shadow)) => {
                    let (start, len) = shadow.span();
                    shadow.gpu_writes(list, start, len);
                }
                // Told after the wait ([`after_wait`]).
                Some(Numbered::Chunk { .. }) => {
                    list.first_use(CHUNKS_WRITTEN);
                }
                None => {}
            }
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

/// The key a list notes ([`HostList::first_use`]) when its shaders may
/// store through the directory into chunk pages ([`after_wait`]).
const CHUNKS_WRITTEN: usize = usize::MAX - 0x3_0000;

/// The runtime mapped or unmapped title memory `[at, at + len)`
/// (`crate::watch::title_memory_remapped`, `crate::inplace::title_memory_view`):
/// with the directory on, the chunk pages there get their entries anew at
/// the next submission's head ([`before_submission`]). Takes only its own
/// lock: called while the runtime changes the mapping.
pub(crate) fn remapped(at: usize, len: usize) {
    if len == 0 || !CHUNKED_ANY.load(Ordering::Relaxed) {
        return;
    }
    let mut remaps = REMAPS.lock().unwrap_or_else(|e| e.into_inner());
    let (at, len) = (at as u64, len as u64);
    // The runtime tells a remap, then the views it maps: one entry for both.
    if remaps.last() != Some(&(at, len)) {
        remaps.push((at, len));
    }
}

/// A host copy of title memory `[at, at + len)` was made (`crate::shadow`:
/// a title buffer's, a window's, a buffer view's exact copy, a copy in the
/// card's memory): with the directory on, the chunk pages there take it at
/// the next list that makes its pages ready ([`Memory::apply_copies`]).
/// Takes only its own lock: called with the shadow tables held.
pub(crate) fn copy_made(at: u64, len: u64) {
    if len == 0 || !CHUNKED_ANY.load(Ordering::Relaxed) {
        return;
    }
    COPIED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((at, len));
}

/// The chunk of the pool with section `section` went
/// (`crate::pool::title_memory_pool_chunk_gone`), its buffer `buffer`: with
/// the directory on, its number is let go at the next [`ready`] or
/// submission's head, so that the directory does not keep its heap (and the
/// section's commit charge) for the rest of the run. Takes only its own
/// lock.
pub(crate) fn chunk_gone(section: usize, buffer: &ID3D12Resource) {
    if !CHUNKED_ANY.load(Ordering::Relaxed) {
        return;
    }
    GONE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Gone(section, buffer.clone()));
}

/// At submission `number`, before its lists go to the card (`Host::submit`,
/// with every submission waited for, or with R5): with the directory on, the
/// chunks that went lose their numbers ([`chunk_gone`]), the chunk pages
/// remapped since get their entries anew ([`remapped`]), then the entries
/// changed since the last submission are copied into the shaders' table by
/// a list of ours that the caller runs ahead of the submission's lists.
/// None if nothing changed (or the switch is off).
pub(crate) fn before_submission(
    host: &crate::host::Host,
    number: u64,
) -> Option<ID3D12CommandList> {
    if !NEXT_MADE.load(Ordering::Relaxed) {
        return None;
    }
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let m = guard.as_mut()?;
    let on = directory_on();
    if on {
        m.forget_gone_chunks();
        m.apply_remaps();
        m.prune_copies();
    }
    let Memory {
        directory, next, ..
    } = m;
    let next = next.as_mut()?;
    if next.dirty.is_empty() {
        return None;
    }
    let pages = std::mem::take(&mut next.dirty);
    // Runs of pages side by side: one copy each.
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for page in pages {
        match runs.last_mut() {
            Some((first, count)) if *first + *count == page => *count += 1,
            _ => runs.push((page, 1)),
        }
    }
    let straight = |next: &Next, runs: &[(u32, u32)]| {
        for &(first, count) in runs {
            let from = first as usize * 2;
            let to = from + count as usize * 2;
            // SAFETY: inside the directory's mapping; nothing of the card's
            // reads it now (every submission is waited for). shortcut: with
            // R5 a flight in the air may read it meanwhile, and see some of
            // these entries early (only when no memory or list could be
            // had for the copy, counted).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    next.entries[from..to].as_ptr(),
                    directory.1.add(from),
                    to - from,
                )
            };
        }
    };
    if !on {
        // The switch went off (tests): the shaders' table is written again.
        straight(next, &runs);
        return None;
    }
    let bytes: u64 = runs.iter().map(|&(_, count)| u64::from(count) * 8).sum();
    let recorded = record(host, directory, next, &runs, bytes, number);
    DIR[at::UPDATES].fetch_add(1, Ordering::Relaxed);
    DIR[at::UPDATED].fetch_add(bytes / 8, Ordering::Relaxed);
    if recorded.is_none() {
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| {
            tracing::warn!("memory directory: no memory or list for the copy at a submission's head; its entries are written straight into the shaders' table")
        });
        DIR[at::STRAIGHT].fetch_add(1, Ordering::Relaxed);
        straight(next, &runs);
    }
    recorded
}

/// The list that copies the entries of `runs` (first page, pages) of
/// `next` into `directory`, through a staging buffer `bytes` long, for
/// submission `number`: a [`Copier`] whose last submission has finished
/// (`crate::host::finished`), or a new one. It is that submission's until
/// it finishes.
fn record(
    host: &crate::host::Host,
    directory: &(ID3D12Resource, *mut u32, u64),
    next: &mut Next,
    runs: &[(u32, u32)],
    bytes: u64,
    number: u64,
) -> Option<ID3D12CommandList> {
    let mut copier = match next.copiers.take(crate::host::finished()) {
        Some(copier) => copier,
        None => Copier {
            staging: None,
            list: new_copier(host)?,
        },
    };
    let recorded = record_with(host, directory, &next.entries, &mut copier, runs, bytes);
    // Its list may go to the card: kept until this submission finishes.
    next.copiers.put(number, copier);
    recorded
}

/// A list of ours (and its allocator), closed, for [`record`].
fn new_copier(
    host: &crate::host::Host,
) -> Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
    // SAFETY: plain object creation.
    let made = unsafe {
        (|| -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            list.Close()?;
            Ok((allocator, list))
        })()
    };
    made.ok()
}

/// [`record`] with `copier`, whose last submission has finished.
fn record_with(
    host: &crate::host::Host,
    directory: &(ID3D12Resource, *mut u32, u64),
    entries: &[u32],
    copier: &mut Copier,
    runs: &[(u32, u32)],
    bytes: u64,
) -> Option<ID3D12CommandList> {
    if copier.staging.as_ref().is_none_or(|s| s.2 < bytes) {
        let size = bytes.next_power_of_two().max(64 << 10);
        let (buffer, data) = host.shared_buffer(size)?;
        crate::host::label(&buffer, "title memory by address", size);
        crate::host::name(&buffer, || "title memory by address: changes".to_string());
        // (The one it replaces: its last copy has run.)
        copier.staging = Some((buffer, data, size));
    }
    let (staging, data, _) = copier.staging.as_ref()?;
    let mut at = 0usize;
    for &(first, count) in runs {
        let from = first as usize * 2;
        let dwords = count as usize * 2;
        // SAFETY: the staging mapping holds `bytes`; the card does not read
        // it now (its last copy ran in a submission that has finished).
        unsafe {
            std::ptr::copy_nonoverlapping(
                entries[from..from + dwords].as_ptr(),
                data.add(at).cast::<u32>(),
                dwords,
            )
        };
        at += dwords * 4;
    }
    let (allocator, list) = &copier.list;
    // SAFETY: the list's last run finished (its submission has finished
    // before this one was prepared); live buffers, the ranges inside both.
    let recorded = unsafe {
        (|| -> windows::core::Result<ID3D12CommandList> {
            allocator.Reset()?;
            list.Reset(allocator, None)?;
            let mut at = 0u64;
            for &(first, count) in runs {
                let len = u64::from(count) * 8;
                list.CopyBufferRegion(&directory.0, u64::from(first) * 8, staging, at, len);
                at += len;
            }
            list.Close()?;
            list.cast()
        })()
    };
    recorded.ok()
}

/// After a submission's wait (`Host::submit`; with R5 at its flight's
/// landing, `Host::relay_land`): if one of its lists could
/// store through the directory into chunk pages ([`ready`]), the copies
/// over those pages, at every view of them, take them again
/// (`crate::home::directory_written`): the guards do not see the card's
/// writes. See the module for what "those pages" are.
pub(crate) fn after_wait(runs: &[&HostList]) {
    if !CHUNKED_ANY.load(Ordering::Relaxed) || !runs.iter().any(|list| list.used(CHUNKS_WRITTEN)) {
        return;
    }
    // Runs of pages side by side in one chunk, in its order: each is one
    // run of one view, whose other views are looked for together.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    {
        let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = guard.as_ref() else {
            return;
        };
        let mut last: Option<(u64, (u32, u32))> = None;
        for &page in &m.chunked {
            let page = u64::from(page);
            let entry = m.get(page);
            if entry.0 == 0 {
                continue;
            }
            let (from, to) = ((page * PAGE) as usize, ((page + 1) * PAGE) as usize);
            match (ranges.last_mut(), last) {
                (Some(range), Some((before, (n, offset))))
                    if before + 1 == page
                        && n == entry.0
                        && u64::from(offset) + PAGE == u64::from(entry.1) =>
                {
                    range.1 = to
                }
                _ => ranges.push((from, to)),
            }
            last = Some((page, entry));
        }
    }
    DIR[at::TOLD].fetch_add(1, Ordering::Relaxed);
    crate::home::directory_written(ranges);
}

/// The memory core let `home` go (`crate::home::cpu_touched`): if it has a
/// number, the directory's entries for it go back to empty and the number
/// names nothing from now on (its views in the heaps' runs are made null at
/// their next [`ready`]; until then they hold its buffer). Whether it had
/// one.
///
/// A shader of a submission running meanwhile, or of a list recorded before
/// and submitted after, that reads the home through the directory takes the
/// fault path for its pages. So the core calls this at its tick
/// (`crate::home::tick`), when no submission runs ([`forget_homes_over`]),
/// except for a title's protection, which cannot wait and is counted. (With
/// the directory on, the shaders' table loses the entries at the next
/// submission's head.)
pub(crate) fn forget(home: &Arc<Shadow>) -> bool {
    let gone = {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = guard.as_mut() else {
            return false;
        };
        let Some(n) = m.copies.iter().position(|c| {
            c.as_ref()
                .and_then(Numbered::shadow)
                .is_some_and(|c| Arc::ptr_eq(c, home))
        }) else {
            return false;
        };
        m.forget_at(n)
    };
    // Dropped without the lock.
    drop(gone);
    true
}

/// At the title's end (`crate::release`): the directory goes whole, with
/// the copies and homes it numbers and the buffers of the chunks gone since
/// it last looked ([`chunk_gone`]). Returns the numbers it held.
pub(crate) fn release_all() -> u64 {
    let memory = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).take();
    let gone = std::mem::take(&mut *GONE.lock().unwrap_or_else(|e| e.into_inner()));
    let numbered = memory.as_ref().map_or(0, |m| {
        m.copies.iter().filter(|c| c.is_some()).count() as u64
    });
    // Dropped without the locks.
    drop((memory, gone));
    numbered
}

/// [`forget`] for every home of the memory core with a number whose pinned
/// pages (`crate::home::pinned`) lie over title memory `[start, end)`.
pub(crate) fn forget_homes_over(start: u64, end: u64) {
    let gone: Vec<Arc<Shadow>> = {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = guard.as_mut() else {
            return;
        };
        let over: Vec<usize> = (m.copies.iter().enumerate())
            .filter_map(|(n, c)| {
                let c = c.as_ref()?.shadow()?;
                if !c.is_home() {
                    return None;
                }
                let (first, len) = c.span();
                let (from, to) = crate::home::pinned(first, len);
                (from < end && start < to).then_some(n)
            })
            .collect();
        over.into_iter().filter_map(|n| m.forget_at(n)).collect()
    };
    // Dropped without the lock.
    drop(gone);
}

/// The view heaps whose slots still hold a home [`forget`] let go (made
/// null at their next [`ready`]), for the core's ten-second line.
///
/// shortcut: a view heap the title no longer uses keeps the home's heap
/// until it goes itself, and so a free of those pages waits (and is given
/// up after a while) and a protection of them is refused.
pub(crate) fn let_go_views_waiting() -> u64 {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_ref() else {
        return 0;
    };
    m.heaps
        .values()
        .filter(|&&(_, made, nulled)| m.forgotten.iter().skip(nulled).any(|&n| n < made))
        .count() as u64
}

/// The directory's part of the memory core's ten-second line, with the
/// switch on.
pub(crate) fn line() -> Option<String> {
    if !directory_on() {
        return None;
    }
    let d: Vec<u64> = DIR.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    let (numbered, chunked) = {
        let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().map_or((0, 0), |m| {
            (
                m.copies.iter().filter(|c| c.is_some()).count(),
                m.chunked.len(),
            )
        })
    };
    let why: Vec<String> = (0..crate::home::WHY_COUNT)
        .filter_map(|i| {
            let n = COPY_WHY[i].load(Ordering::Relaxed);
            (n > 0).then(|| format!("{n} {}", crate::home::why_label(i)))
        })
        .chain(std::iter::once(format!(
            "{} not view memory",
            d[at::COPY_OTHER]
        )))
        .collect();
    Some(format!(
        "memory directory: {numbered} numbers ({} pool chunks, {} let go as their chunk went); entries filled: {} pages of pool chunks ({chunked} now), {} of homes, {} of copies; \
         asks a copy answered: {} ({} chunk pages a copy held bytes of, {} chunk pages that took a copy made later); remaps: {} chunk pages looked at again, {} entries changed, {} left empty until the next dispatch (their chunk had no number); {} submission heads copied {} entries ({} written straight); \
         {} submissions' raw writes told to copies over chunk pages; homes of the core turned down as views of several places (any use, since the start; still copied until R6): {}",
        d[at::CHUNKS],
        d[at::CHUNKS_GONE],
        d[at::CHUNK_PAGES],
        d[at::HOME_PAGES],
        d[at::COPY_PAGES],
        why.join(", "),
        d[at::COPY_OVER],
        d[at::SWITCHED],
        d[at::REMAP_LOOKED],
        d[at::REMAP_CHANGED],
        d[at::REMAP_WANTED],
        d[at::UPDATES],
        d[at::UPDATED],
        d[at::STRAIGHT],
        d[at::TOLD],
        crate::home::turned_down_count(Why::Scattered),
    ))
}

impl Memory {
    /// Copy `n` (0-based) is let go: the directory's entries for it go back
    /// to empty, its number names nothing from now on.
    fn forget_at(&mut self, n: usize) -> Option<Arc<Shadow>> {
        let slot = self.copies.get_mut(n)?;
        if !matches!(slot, Some(Numbered::Copy(_))) {
            return None;
        }
        let Some(Numbered::Copy(home)) = slot.take() else {
            return None;
        };
        // Every entry is a page the copy holds (`Memory::map`).
        let (first, len) = home.span();
        let mut at = first / PAGE * PAGE;
        while at < first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
            if self.get(at / PAGE).0 == n as u32 + 1 {
                self.set(at / PAGE, 0, 0);
            }
            at += PAGE;
        }
        self.forgotten.push(n as u32);
        Some(home)
    }
}

/// Evicts cached descriptor-run state for `heap` when its owning `Heap` is
/// dropped so a later `Heap` allocated at the same address cannot reuse stale
/// `(base, made)` slot offsets.
pub(crate) fn forget_heap(heap: &HostHeap) {
    let key = std::ptr::from_ref(heap) as usize;
    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.heaps.remove(&key);
    }
}

/// The entry of the page of `address` in the shaders' table and in the
/// table the next submission sees (tests).
#[cfg(test)]
pub(crate) fn entries_of(address: u64) -> Option<((u32, u32), (u32, u32))> {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let m = guard.as_ref()?;
    if address / PAGE >= u64::from(MEMORY_PAGES) {
        return None;
    }
    let i = (address / PAGE) as usize * 2;
    // SAFETY: inside the directory's mapping.
    let shaders = unsafe {
        (
            m.directory.1.add(i).read_volatile(),
            m.directory.1.add(i + 1).read_volatile(),
        )
    };
    Some((shaders, m.get(address / PAGE)))
}

/// Whether directory number `number` (1-based) names a copy or chunk now
/// (tests).
#[cfg(test)]
pub(crate) fn names(number: u32) -> bool {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().is_some_and(|m| {
        (number as usize)
            .checked_sub(1)
            .and_then(|n| m.copies.get(n))
            .is_some_and(Option::is_some)
    })
}

/// The directory's buffer (tests).
#[cfg(test)]
pub(crate) fn directory_buffer() -> Option<ID3D12Resource> {
    let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    guard.as_ref().map(|m| m.directory.0.clone())
}

/// A test is done with title memory `[from, to)`: the copies numbered over
/// it are let go and every entry there is emptied, in both tables at once
/// (the process's directory outlives the test's memory), and the copies of
/// windows and buffer views there leave the shadow tables.
#[cfg(test)]
pub(crate) fn forget_for_test(from: u64, to: u64) {
    let gone: Vec<Arc<Shadow>> = {
        let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
        let Some(m) = guard.as_mut() else {
            return;
        };
        let over: Vec<usize> = (m.copies.iter().enumerate())
            .filter_map(|(n, c)| {
                let (first, len) = c.as_ref()?.shadow()?.span();
                (first < to && from < first + len).then_some(n)
            })
            .collect();
        let gone = over.into_iter().filter_map(|n| m.forget_at(n)).collect();
        for page in from / PAGE..to.div_ceil(PAGE).min(u64::from(MEMORY_PAGES) - 1) {
            m.set(page, 0, 0);
            m.chunked.remove(&(page as u32));
            if let Some(next) = &mut m.next {
                next.dirty.remove(&(page as u32));
            }
            let i = page as usize * 2;
            // SAFETY: inside the directory's mapping; no submission runs.
            unsafe {
                m.directory.1.add(i).write_volatile(0);
                m.directory.1.add(i + 1).write_volatile(0);
            }
        }
        gone
    };
    drop(gone);
    crate::shadow::let_go_copies_for_test(from, to);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::{HeapDesc, create_heap, heap};
    use crate::object::{S_OK, This, guid};

    #[test]
    fn dropping_a_view_heap_evicts_its_memory_heaps_entry() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let iid = guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create_heap(std::ptr::null(), &desc, &iid, &mut h), S_OK);
        let list = host.new_list().unwrap();
        let host_heap = heap(h).host().expect("host view heap");
        let key = std::ptr::from_ref(host_heap) as usize;

        assert!(ready(&list, host_heap, &[], &[], false).is_some());
        {
            let guard = MEMORY.lock().unwrap();
            assert!(guard.as_ref().unwrap().heaps.contains_key(&key));
        }

        // Release the title heap (COM vtable slot 2): `Heap::drop` must evict `key`.
        // SAFETY: `h` is a live COM object from `create_heap`.
        let release: extern "system" fn(This) -> u32 =
            unsafe { std::mem::transmute(*(*(h as *const *const *const ())).add(2)) };
        assert_eq!(release(h), 0);

        let guard = MEMORY.lock().unwrap();
        assert!(
            !guard.as_ref().unwrap().heaps.contains_key(&key),
            "MEMORY.heaps retained dropped HostHeap key {key:#x}"
        );
    }
}
