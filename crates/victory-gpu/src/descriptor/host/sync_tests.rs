//! The heap sync (`super::super::sync_heaps`): a changed page's records at
//! once and the early skip (`VICTORY_HEAP_SYNC_FAST=1`), and the pass run
//! ahead on its own thread (`super::super::ahead`).

use super::made_tests::{alive, key, memory, next, record_key, texture, views, write};
use super::*;
use crate::descriptor::ahead::Ahead;
use crate::descriptor::{self as descriptor, HeapDesc, Placement};
use crate::object::{S_OK, This, slot};
use std::sync::atomic::Ordering::{Acquire, Relaxed};
use std::time::Duration;

/// Slots of the heaps these tests make.
const SLOTS: u32 = 64;

/// [`sync_fast`] on until dropped.
struct Fast;

impl Fast {
    fn on() -> Self {
        SYNC_FAST_FOR_TESTS.store(true, Relaxed);
        Self
    }
}

impl Drop for Fast {
    fn drop(&mut self) {
        SYNC_FAST_FOR_TESTS.store(false, Relaxed);
    }
}

/// A title heap placed in write-watched memory of its own: the heap, the
/// memory, and the host heap behind it (`None`: no host GPU).
struct Placed {
    heap: This,
    memory: *mut std::ffi::c_void,
}

impl Placed {
    /// A shader-visible heap of `kind` ([`descriptor::VIEWS_KIND`] or
    /// [`descriptor::SAMPLERS`]) with [`SLOTS`] slots.
    fn new(kind: u32) -> Self {
        Self::with_slots(kind, SLOTS)
    }

    /// [`Placed::new`] with `slots` slots (in one page of memory).
    fn with_slots(kind: u32, slots: u32) -> Self {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: new memory.
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        #[repr(C)]
        struct Extended(HeapDesc, Placement);
        let step = u64::from(descriptor::increment(kind).expect("a kind with records"));
        assert!(u64::from(slots) * step <= crate::watch::PAGE as u64);
        let extended = Extended(
            HeapDesc {
                kind,
                count: slots,
                flags: descriptor::SHADER_VISIBLE,
                nodes: 1,
            },
            Placement {
                address: memory as u64,
                size: u64::from(slots) * step,
            },
        );
        let mut heap: This = std::ptr::null();
        assert_eq!(
            descriptor::create_heap(
                std::ptr::null(),
                &extended.0,
                &descriptor::HEAP_IIDS[0],
                &mut heap
            ),
            S_OK
        );
        Self { heap, memory }
    }

    fn host(&self) -> Option<&HostHeap> {
        descriptor::heap(self.heap).host()
    }

    /// The handle of slot `index`.
    fn at(&self, index: u32) -> u64 {
        descriptor::heap(self.heap)
            .handle(index)
            .expect("a slot of the heap")
    }

    /// Where the heap stands in the list of shader-visible heaps.
    fn registered(&self) -> usize {
        std::ptr::from_ref(descriptor::heap(self.heap)) as usize
    }

    /// The title's last release of the heap.
    fn release(heap: This) -> u32 {
        slot::<extern "system" fn(This) -> u32>(heap, 0x10)(heap)
    }
}

impl Drop for Placed {
    fn drop(&mut self) {
        if !self.heap.is_null() {
            Self::release(self.heap);
        }
        // SAFETY: this allocation belongs to the test.
        unsafe {
            windows_sys::Win32::System::Memory::VirtualFree(
                self.memory,
                0,
                windows_sys::Win32::System::Memory::MEM_RELEASE,
            )
        };
    }
}

/// Copies the `n` bytes of the record at `from` to `to`, as a title does.
fn copy_record(from: u64, to: u64, n: usize) {
    // SAFETY: both are records of at least `n` bytes, apart.
    unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, n) };
}

/// Every made slot of `heap` holds its view's word: never a word of
/// another kind than the descriptor under it (the null slot crash).
fn words_agree(heap: &HostHeap) {
    for s in 0..heap.count() {
        let key = key(heap, s);
        if key != 0 {
            let view = view_at(key as u32, (key >> 32) as u32).expect("made from a view");
            assert_eq!(heap.word(s), Some(slot_word(Some(&view))), "slot {s}");
        }
    }
}

/// A title texture object of 16x16 texels the title fills, whose host copy
/// is made on first use (`crate::resource::host_waits` until then). Left
/// to the process.
fn title_texture() -> This {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_DESC,
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, D3D12_RESOURCE_FLAG_NONE, D3D12_TEXTURE_LAYOUT_UNKNOWN,
    };
    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
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
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let iid = crate::object::guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );
    let mut texture: This = std::ptr::null();
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
        S_OK
    );
    assert!(
        crate::resource::host_waits(texture),
        "no host copy before its first use"
    );
    texture
}

/// Records over a page, some slots a table's lately, mirrored one by one
/// (`mirror_copied`) on one heap and a page at once (`mirror_copied_page`)
/// on another: after every round the same slots made from the same
/// records, with the same words and the same prints last seen.
#[test]
fn a_page_at_once_makes_what_one_by_one_makes() {
    page_against_one_by_one(false);
}

/// As above, with the early skip on for the page at once (and off for one
/// by one): the page and the skip together make what one by one makes.
#[test]
fn a_page_at_once_with_the_early_skip_makes_what_one_by_one_makes() {
    page_against_one_by_one(true);
}

/// The two tests above: the early skip on for the page at once if `skip`.
fn page_against_one_by_one(skip: bool) {
    let (Some(one), Some(page)) = (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS)) else {
        return; // no host GPU
    };
    let views = views(memory());
    let skipped = SKIPPED_MADE.load(Relaxed);
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let step = 32u64;
    let mut rng = 0x51f1_5eed_0bad_cafeu64;
    let mut mirrored = 0u64;
    for round in 0..200 {
        // Some slots written, some a table's.
        for _ in 0..8 {
            let r = next(&mut rng);
            let s = (r % u64::from(SLOTS)) as u32;
            write(
                base + u64::from(s) * step,
                &views[(r >> 16) as usize % views.len()],
            );
            if r >> 40 & 7 == 0 {
                one.texture_slot_tabled(s, false);
                page.texture_slot_tabled(s, false);
            }
            // Some made by a draw before the sync sees them (what the early
            // skip leaves).
            if r >> 44 & 3 == 0 {
                one.mirror(s, base + u64::from(s) * step, true);
                page.mirror(s, base + u64::from(s) * step, true);
            }
        }
        // The page as the safe copy (`crate::safecopy`) reads it.
        // SAFETY: `records` is `SLOTS` records of `step` bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(base as *const u8, (u64::from(SLOTS) * step) as usize)
        }
        .to_vec();
        let mut one_by_one = 0;
        for index in 0..u64::from(SLOTS) {
            let at = (index * step) as usize;
            if one.mirror_copied(
                index as u32,
                base + index * step,
                &bytes[at..at + step as usize],
            ) {
                one_by_one += 1;
            }
        }
        let (looked, made) = {
            let _fast = skip.then(Fast::on);
            page.mirror_copied_page(0..u64::from(SLOTS), (base, step), base as usize, &bytes)
        };
        assert_eq!(looked, u64::from(SLOTS), "round {round}");
        assert_eq!(made, one_by_one, "round {round}");
        mirrored += made;
        for s in 0..SLOTS {
            assert_eq!(key(&one, s), key(&page, s), "round {round}, slot {s}");
            assert_eq!(one.word(s), page.word(s), "round {round}, slot {s}");
            assert_eq!(
                one.copied_print(s),
                page.copied_print(s),
                "round {round}, slot {s}"
            );
        }
        assert_eq!(one.buffer_slots(), page.buffer_slots(), "round {round}");
    }
    assert!(mirrored > 0);
    if skip {
        assert!(SKIPPED_MADE.load(Relaxed) > skipped, "records skipped");
    }
    assert!(alive());
}

/// The early skip: a slot made from a record of a view still there is done
/// with before its claim (counted as skipped); a slot made from a record
/// whose resource was released since goes on to the claim and is made
/// null, as without the skip.
#[test]
fn the_early_skip_takes_no_claim_and_still_nulls_a_released_view() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let _fast = Fast::on();
    let heap = std::sync::Arc::new(heap);
    let mut records = vec![[0u32; 8]; 2];
    let (live, gone) = (
        records.as_mut_ptr() as u64,
        records.as_mut_ptr() as u64 + 32,
    );
    // A constant buffer view, made in slot 0.
    write(live, &Some(View::Constant(0x7d_5000_0000, 256)));
    heap.mirror(0, live, true);
    assert_eq!(
        key(&heap, 0),
        record_key(live),
        "slot 0 made from its record"
    );
    // Seen again by the sync: done at once.
    let skipped = SKIPPED_MADE.load(Relaxed);
    // (Called before any claim of the slot: left uncalled.)
    BEFORE_CLAIM.set(Some(|_| {}));
    let view = heap.mirror_marking(0, live, false, true, hold::SYNC);
    assert!(BEFORE_CLAIM.take().is_some(), "no claim asked for");
    assert_eq!(view, Some(View::Constant(0x7d_5000_0000, 256)));
    assert_eq!(SKIPPED_MADE.load(Relaxed), skipped + 1);
    // A view of a resource the title releases: its slot made from it (as a
    // draw would have made it), then the resource released.
    let resource = 0x7d_7777_0000usize;
    write(gone, &Some(View::Shader(resource, None)));
    let stale = record_key(gone);
    {
        let mut claim = heap.claim(1, |_| true).expect("not claimed");
        claim.value = stale;
        heap.write_word(1, slot_word(Some(&View::Constant(0, 256))));
    }
    descriptor::forget(resource);
    assert_eq!(
        view_at(stale as u32, (stale >> 32) as u32),
        None,
        "stale after release"
    );
    assert_eq!(heap.mirror_marking(1, gone, false, true, hold::SYNC), None);
    assert_eq!(key(&heap, 1), 0, "a released view's slot made null");
    assert_eq!(heap.word(1), Some(0));
    assert_eq!(SKIPPED_MADE.load(Relaxed), skipped + 1, "not skipped");
    assert!(alive());
}

/// A record the title copied into a placed heap as bytes, then a request
/// and its wait: the slot is made (as a direct call of `sync_heaps` makes
/// it).
#[test]
fn a_request_and_its_wait_leave_a_copied_record_made() {
    let placed = Placed::new(descriptor::SAMPLERS);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
    let ahead = Ahead::start(descriptor::sync_heaps).expect("a helper thread");
    let mut elsewhere = [0u64; 2];
    let source = elsewhere.as_mut_ptr() as u64;
    let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
    descriptor::create_sampler(std::ptr::null(), &sampler, source);
    copy_record(source, placed.at(5), 16);
    assert!(!host.made(5));
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    assert!(ahead.wait(seq));
    assert!(host.made(5));
    assert_eq!(ahead.behind(seq), 0);
}

/// A pass that began before a request does not answer it: with the first
/// pass held, what is written and asked for meanwhile waits for a second
/// pass, which sees it.
#[test]
fn a_pass_begun_before_a_request_does_not_answer_it() {
    use std::sync::atomic::AtomicU64;
    use std::sync::mpsc::channel;
    let written = std::sync::Arc::new(AtomicU64::new(0));
    let (came, come) = channel::<u64>();
    let (go, gone) = channel::<()>();
    let gone = std::sync::Mutex::new(gone);
    let ahead = Ahead::start({
        let written = written.clone();
        move |_: &[usize]| {
            // What the pass sees as it begins.
            let _ = came.send(written.load(Acquire));
            let _ = gone.lock().unwrap().recv();
        }
    })
    .expect("a helper thread");
    let first = ahead.request(&[]).expect("a request");
    let seen = come
        .recv_timeout(Duration::from_secs(30))
        .expect("the first pass");
    assert_eq!(seen, 0);
    // The title writes, and the next submission asks, while the pass runs.
    written.store(1, std::sync::atomic::Ordering::Release);
    let second = ahead.request(&[]).expect("a request");
    assert_eq!(ahead.behind(second), 2);
    go.send(()).unwrap();
    assert!(ahead.wait(first));
    // The second pass has begun (or is about to) and is held: not answered.
    let seen = come
        .recv_timeout(Duration::from_secs(30))
        .expect("a second pass");
    assert_eq!(seen, 1, "the second pass sees the write");
    assert_eq!(
        ahead.behind(second),
        1,
        "the first pass does not answer the second request"
    );
    go.send(()).unwrap();
    assert!(ahead.wait(second));
    assert_eq!(ahead.behind(second), 0);
}

/// A heap the title releases while the pass walks it: the heap's drop
/// waits for the pass (the list of heaps is read-locked through it), and
/// nothing reads freed memory.
#[test]
fn a_heap_dropped_while_the_pass_walks_it_waits_for_it() {
    let mut placed = Placed::new(descriptor::SAMPLERS);
    if placed.host().is_none() {
        return; // no host GPU
    }
    let mut elsewhere = [0u64; 2];
    let source = elsewhere.as_mut_ptr() as u64;
    let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
    descriptor::create_sampler(std::ptr::null(), &sampler, source);
    copy_record(source, placed.at(9), 16);
    let (came, come) = std::sync::mpsc::channel();
    let (go, gone) = std::sync::mpsc::channel();
    *descriptor::tests_hook::AT.lock().unwrap() = Some((placed.registered(), came, gone));
    let ahead = Ahead::start(descriptor::sync_heaps).expect("a helper thread");
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    come.recv_timeout(Duration::from_secs(30))
        .expect("the pass comes to the heap");
    // The title's last release, on a thread of its own.
    let heap = std::mem::replace(&mut placed.heap, std::ptr::null()) as usize;
    let dropped = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let releasing = {
        let dropped = dropped.clone();
        std::thread::spawn(move || {
            Placed::release(heap as This);
            dropped.store(true, Relaxed);
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    assert!(!dropped.load(Relaxed), "the heap's drop waits for the pass");
    go.send(()).unwrap();
    assert!(ahead.wait(seq));
    releasing.join().unwrap();
    assert!(dropped.load(Relaxed));
    assert!(alive());
}

/// The submitting thread waits for the pass, and makes the slots left
/// waiting, while it holds the host's submission lock, as a submission
/// does (`Host::submit`). The pass mirrors a placed texture's view, whose
/// host texture it looks up and makes, and more views of one title texture
/// object without a host copy than may wait for it, so that it makes that
/// texture's host copy (`descriptor::wait_for`): the wait ends, so nothing
/// on the pass's way takes the submission lock.
#[test]
fn a_wait_under_the_submission_lock_ends() {
    /// Slots that view the title texture: more than the 64 that may wait.
    const VIEWING: u32 = 72;
    let placed = Placed::with_slots(descriptor::VIEWS_KIND, VIEWING + 8);
    let (Some(_), Some(gpu)) = (placed.host(), crate::host::host()) else {
        return; // no host GPU
    };
    let placed_view = texture(memory(), 64);
    let title = title_texture();
    let title_view = View::Shader(title as usize, None);
    let mut elsewhere = [0u64; 8];
    let (source, other) = (
        elsewhere.as_mut_ptr() as u64,
        elsewhere.as_mut_ptr() as u64 + 32,
    );
    write(source, &Some(placed_view.clone()));
    copy_record(source, placed.at(VIEWING + 3), 32);
    write(other, &Some(title_view.clone()));
    for s in 0..VIEWING {
        copy_record(other, placed.at(s), 32);
    }
    let ahead = std::sync::Arc::new(Ahead::start(descriptor::sync_heaps).expect("a helper thread"));
    let (done, finished) = std::sync::mpsc::channel();
    {
        let ahead = ahead.clone();
        let heaps = [placed.registered()];
        std::thread::spawn(move || {
            let seq = ahead.request(&heaps).expect("a request");
            let held = crate::host::flights::submission_lock(gpu);
            crate::descriptor::ahead::wait_and_make(Some(&ahead), Some(seq), &heaps);
            drop(held);
            let _ = done.send(());
        });
    }
    if finished.recv_timeout(Duration::from_secs(120)).is_err() {
        // The pass is stuck in the heap: its drop would wait for it.
        std::mem::forget(placed);
        panic!("the wait under the submission lock does not end");
    }
    assert!(
        !crate::resource::host_waits(title),
        "the pass made the title texture's host copy"
    );
    let host = placed.host().unwrap();
    assert_eq!(host.word(VIEWING + 3), Some(slot_word(Some(&placed_view))));
    for s in 0..VIEWING {
        assert!(host.made(s), "slot {s} made");
        assert_eq!(host.word(s), Some(slot_word(Some(&title_view))), "slot {s}");
    }
    words_agree(host);
    assert!(alive());
}

/// A pass leaves slots that view a title texture without a host copy
/// waiting for it; the texture is made meanwhile (a draw uses it). The
/// next submission's wait makes them before it returns
/// (`descriptor::ahead::wait_and_make`), though its pass sees their records
/// as the ones it saw last.
#[test]
fn slots_left_waiting_are_made_by_the_next_wait() {
    let placed = Placed::new(descriptor::VIEWS_KIND);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
    let title = title_texture();
    let view = View::Shader(title as usize, None);
    let mut elsewhere = [0u64; 4];
    let source = elsewhere.as_mut_ptr() as u64;
    write(source, &Some(view.clone()));
    for s in 0..4 {
        copy_record(source, placed.at(s), 32);
    }
    let ahead = Ahead::start(descriptor::sync_heaps).expect("a helper thread");
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    crate::descriptor::ahead::wait_and_make(Some(&ahead), Some(seq), &[placed.registered()]);
    assert!(crate::resource::host_waits(title));
    for s in 0..4 {
        assert!(!host.made(s), "slot {s} waits for its texture");
    }
    // Made by a draw before the next submission.
    assert!(crate::resource::host_copy(title).is_some());
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    crate::descriptor::ahead::wait_and_make(Some(&ahead), Some(seq), &[placed.registered()]);
    for s in 0..4 {
        assert!(host.made(s), "slot {s} made with its texture");
        assert_eq!(host.word(s), Some(slot_word(Some(&view))), "slot {s}");
    }
    words_agree(host);
    assert!(alive());
}

/// Four threads write views into a placed heap (`Create*View`), copy
/// records into it as bytes and by calls (`CopyDescriptorsSimple`) and
/// make slots for draws that read them through tables, while submissions
/// ask the helper for passes and wait for them under the submission lock,
/// with the early skip on: all finish, and every slot's word is of the kind
/// of the descriptor under it, during and after.
#[test]
fn many_writers_and_the_pass_ahead_leave_each_slot_with_its_own_word() {
    const THREADS: u64 = 4;
    let placed = Placed::new(descriptor::VIEWS_KIND);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
    let _fast = Fast::on();
    let views = views(memory());
    let heap_at: Vec<u64> = (0..SLOTS).map(|s| placed.at(s)).collect();
    let heaps = [placed.registered()];
    let ahead = Ahead::start(descriptor::sync_heaps).expect("a helper thread");
    let stop = std::sync::atomic::AtomicBool::new(false);
    let skipped = SKIPPED_MADE.load(Relaxed);
    let (looks, wrong, passes) = std::thread::scope(|scope| {
        let (done, finished) = std::sync::mpsc::channel();
        for t in 0..THREADS {
            let (views, heap_at, done) = (&views, &heap_at, done.clone());
            scope.spawn(move || {
                // Records of its own to copy from.
                let mut records = vec![[0u32; 8]; 8];
                let base = records.as_mut_ptr() as u64;
                let mut rng = 0x2545_f491_4f6c_dd1d ^ (t + 1).wrapping_mul(0x9e37_79b9);
                for _ in 0..20_000 {
                    let r = next(&mut rng);
                    let slot = (r % u64::from(SLOTS)) as u32;
                    let at = heap_at[slot as usize];
                    let view = &views[(r >> 16) as usize % views.len()];
                    let from = base + (r >> 32) % 8 * 32;
                    match r >> 8 & 7 {
                        // A view written by a call, mirrored at once.
                        0 | 1 => write(at, view),
                        // A draw reads the slot through a table.
                        2..=4 => {
                            host.mirror(slot, at, true);
                        }
                        // A record copied as bytes: only the pass sees it.
                        5 | 6 => {
                            write(from, view);
                            copy_record(from, at, 32);
                        }
                        // A record copied by a call (`CopyDescriptorsSimple`),
                        // mirrored at once: the heap list read as the pass
                        // reads it.
                        _ => {
                            write(from, view);
                            descriptor::copy_simple(
                                std::ptr::null(),
                                1,
                                at,
                                from,
                                descriptor::VIEWS_KIND,
                            );
                        }
                    }
                }
                let _ = done.send(());
            });
        }
        let submitting = scope.spawn(|| {
            let gpu = crate::host::host().expect("the host GPU of the heap");
            let mut passes = 0u64;
            while !stop.load(Relaxed) {
                let seq = ahead.request(&heaps).expect("a request");
                // As a submission waits: under the submission lock.
                let _held = crate::host::flights::submission_lock(gpu);
                assert!(ahead.wait(seq));
                passes += 1;
            }
            passes
        });
        // Meanwhile: a slot that shows the same record before and after its
        // word is read has that record's word, or 0, never another.
        let (mut looks, mut wrong) = (0u64, Vec::new());
        let mut ended = 0;
        while ended < THREADS {
            match finished.recv_timeout(Duration::from_millis(1)) {
                Ok(()) => ended += 1,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                Err(e) => {
                    // (The submissions stop first: else the scope waits.)
                    stop.store(true, Relaxed);
                    panic!("a writer failed: {e}");
                }
            }
            for s in 0..SLOTS {
                let before = host.made[s as usize].load(Acquire);
                let word = host.word(s);
                if before == 0 || before == CLAIMED || host.made[s as usize].load(Acquire) != before
                {
                    continue;
                }
                looks += 1;
                let view = view_at(before as u32, (before >> 32) as u32);
                if word != Some(0) && word != Some(slot_word(view.as_ref())) && wrong.len() < 8 {
                    wrong.push((s, before, word));
                }
            }
        }
        stop.store(true, Relaxed);
        let passes = submitting.join().expect("every wait ends");
        (looks, wrong, passes)
    });
    assert!(passes > 0);
    // A last pass over what was copied last.
    let seq = ahead.request(&heaps).expect("a request");
    assert!(ahead.wait(seq));
    assert!(looks > 0);
    assert!(
        SKIPPED_MADE.load(Relaxed) > skipped,
        "records skipped before the claim"
    );
    assert_eq!(wrong, [], "slots whose word is not their record's");
    words_agree(host);
    assert!(alive());
}

/// Records over a page, some slots a table's lately, some made by a draw,
/// some written with bytes that are no view, mirrored one by one
/// (`mirror_copied`) on one heap, and on another sorted in parts of the
/// page on four threads (`sort_copied_page`) and the slots found mirrored
/// on four threads (`mirror_synced`): after every round the same slots
/// made from the same records, with the same words and buffer sets, and
/// the same records last seen. With prints or bytes (`bytes`), and with or
/// without `shows_none`.
fn sorted_against_one_by_one(bytes: bool, shows_none: bool) {
    let (Some(one), Some(sorted)) = (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS))
    else {
        return; // no host GPU
    };
    let views = views(memory());
    let none_before = SHOWED_NONE.load(Relaxed);
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let step = 32u64;
    let mut rng = 0x0dd5_eed5_1f1a_4330u64 ^ u64::from(bytes) ^ u64::from(shows_none) << 1;
    let mut mirrored = 0u64;
    for round in 0..200 {
        for _ in 0..8 {
            let r = next(&mut rng);
            let s = (r % u64::from(SLOTS)) as u32;
            let handle = base + u64::from(s) * step;
            if r >> 50 & 7 == 0 {
                // Bytes that are no view of ours.
                // SAFETY: a record of `records`.
                unsafe { (handle as *mut [u32; 8]).write_unaligned([(r >> 8) as u32 | 1; 8]) };
            } else {
                write(handle, &views[(r >> 16) as usize % views.len()]);
            }
            if r >> 40 & 7 == 0 {
                one.texture_slot_tabled(s, false);
                sorted.texture_slot_tabled(s, false);
            }
            if r >> 44 & 3 == 0 {
                one.mirror(s, handle, true);
                sorted.mirror(s, handle, true);
            }
        }
        // SAFETY: `records` is `SLOTS` records of `step` bytes.
        let page = unsafe {
            std::slice::from_raw_parts(base as *const u8, (u64::from(SLOTS) * step) as usize)
        }
        .to_vec();
        let mut one_by_one = 0;
        for index in 0..u64::from(SLOTS) {
            let at = (index * step) as usize;
            if one.mirror_copied(
                index as u32,
                base + index * step,
                &page[at..at + step as usize],
            ) {
                one_by_one += 1;
            }
        }
        // The page in eight parts of 256 bytes, as ranges the watch gives.
        let parts: Vec<(usize, usize)> = (0..8)
            .map(|p| (base as usize + p * 256, base as usize + (p + 1) * 256))
            .collect();
        let found: Vec<std::sync::Mutex<SyncPart>> =
            parts.iter().map(|_| Default::default()).collect();
        {
            let copied = sorted.copied_begin(step as usize, bytes);
            crate::draw::each_on(4, parts.len(), "test-sync", &|n| {
                let (from, to) = parts[n];
                let mut part = SyncPart::default();
                let at = from - base as usize;
                let slots = (at as u64 / step)..((to - base as usize) as u64).div_ceil(step);
                // SAFETY: the parts are apart from one another.
                unsafe {
                    sorted.sort_copied_page(
                        &copied,
                        slots,
                        (base, step),
                        from,
                        &page[at..to - base as usize],
                        shows_none,
                        &mut part,
                    )
                };
                *found[n].lock().unwrap() = part;
            });
        }
        let mut make = Vec::new();
        let mut nothing = 0;
        for part in found {
            let mut part = part.into_inner().unwrap();
            assert_eq!(part.looked, 8, "round {round}");
            nothing += part.nothing;
            make.append(&mut part.make);
        }
        assert!(make.is_sorted(), "round {round}");
        crate::draw::each_on(4, make.len().div_ceil(4), "test-sync", &|n| {
            for &index in make.chunks(4).nth(n).unwrap_or_default() {
                sorted.mirror_synced(index, base + u64::from(index) * step);
            }
        });
        if shows_none {
            assert_eq!(make.len() as u64 + nothing, one_by_one, "round {round}");
        } else {
            assert_eq!(nothing, 0, "round {round}");
            assert_eq!(make.len() as u64, one_by_one, "round {round}");
        }
        mirrored += one_by_one;
        for s in 0..SLOTS {
            assert_eq!(key(&one, s), key(&sorted, s), "round {round}, slot {s}");
            assert_eq!(one.word(s), sorted.word(s), "round {round}, slot {s}");
            let print = one.copied_print(s);
            if bytes {
                let seen = sorted.copied_record(s, step as usize).expect("bytes kept");
                if print == 0 {
                    assert!(seen.iter().all(|&b| b == 0), "round {round}, slot {s}");
                } else {
                    assert_eq!(record_print(&seen) | 1, print, "round {round}, slot {s}");
                }
            } else {
                assert_eq!(print, sorted.copied_print(s), "round {round}, slot {s}");
            }
        }
        assert_eq!(one.buffer_slots(), sorted.buffer_slots(), "round {round}");
        words_agree(&sorted);
    }
    assert!(mirrored > 0);
    if shows_none {
        assert!(SHOWED_NONE.load(Relaxed) > none_before, "records left");
    }
    assert!(alive());
}

/// [`sorted_against_one_by_one`] with prints.
#[test]
fn pages_sorted_on_threads_make_what_one_by_one_makes() {
    sorted_against_one_by_one(false, false);
}

/// [`sorted_against_one_by_one`] with the records' bytes as last seen.
#[test]
fn pages_sorted_with_bytes_make_what_one_by_one_makes() {
    sorted_against_one_by_one(true, false);
}

/// [`sorted_against_one_by_one`] with bytes, records of no view into slots
/// made from none left before their mirror.
#[test]
fn records_of_no_view_left_before_their_mirror_make_what_one_by_one_makes() {
    sorted_against_one_by_one(true, true);
}

/// The records [`changed_records`] gives are those that differ one by one,
/// in order, and it changes nothing (from cool_fox 53c5256's test).
#[test]
fn changed_records_are_those_that_differ_one_by_one() {
    for (step, count) in [(32usize, 128usize), (32, 2048), (16, 77), (32, 5), (32, 17)] {
        let mut seed = 7u64;
        let mut random = move || {
            seed = seed
                .wrapping_mul(0x5851_F42D_4C95_7F2D)
                .wrapping_add(0x1405_7B7E_F767_814F);
            (seed >> 33) as usize
        };
        let last: Vec<u8> = (0..step * count).map(|_| random() as u8).collect();
        let mut now = last.clone();
        let mut differ = Vec::new();
        for slot in 0..count {
            if random() % 9 == 0 {
                now[slot * step + random() % step] ^= 1 + (random() % 255) as u8;
                differ.push(1000 + slot as u32);
            }
        }
        let mut out = Vec::new();
        changed_records(&last, &now, step, 1000, &mut out);
        assert_eq!(out, differ);
        out.clear();
        changed_records(&now, &now, step, 1000, &mut out);
        assert!(out.is_empty());
    }
}

/// Passes of its own (`EARLY`): none before the first request; some after
/// it while requests are recent, none of which answers a request (one made
/// during such a pass, held, waits for a pass of its own); none once
/// requests have stopped.
#[test]
fn passes_of_its_own_run_between_requests_and_answer_none() {
    use std::sync::atomic::AtomicU64;
    use std::sync::mpsc::channel;
    let passes = std::sync::Arc::new(AtomicU64::new(0));
    let (came, come) = channel::<u64>();
    let (go, gone) = channel::<()>();
    let gone = std::sync::Mutex::new(gone);
    let ahead = Ahead::start_with(
        {
            let passes = passes.clone();
            move |_: &[usize]| {
                // The second pass, of its own, is held until the test lets
                // it go on.
                let n = passes.fetch_add(1, Relaxed) + 1;
                if n == 2 {
                    let _ = came.send(n);
                    let _ = gone.lock().unwrap().recv();
                }
            }
        },
        Some(Duration::from_millis(2)),
    )
    .expect("a helper thread");
    std::thread::sleep(Duration::from_millis(40));
    assert_eq!(passes.load(Relaxed), 0, "no pass before a request");
    let first = ahead.request(&[]).expect("a request");
    assert!(ahead.wait(first));
    // A pass of its own begins, and is held.
    assert_eq!(
        come.recv_timeout(Duration::from_secs(30))
            .expect("a pass of its own"),
        2
    );
    assert_eq!(ahead.behind(first), 0);
    let second = ahead.request(&[]).expect("a request");
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        ahead.behind(second),
        1,
        "a pass of its own answers no request"
    );
    go.send(()).unwrap();
    assert!(ahead.wait(second));
    assert!(
        passes.load(Relaxed) >= 3,
        "the request had a pass of its own"
    );
    // More passes of its own while requests are recent.
    let after = passes.load(Relaxed);
    std::thread::sleep(Duration::from_millis(100));
    assert!(passes.load(Relaxed) > after, "passes between requests");
    // None once requests have stopped.
    std::thread::sleep(super::super::ahead::EARLY_WINDOW + Duration::from_millis(100));
    let stopped = passes.load(Relaxed);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(
        passes.load(Relaxed),
        stopped,
        "no passes long after a request"
    );
}

/// A heap no submission binds is not looked at; when one binds it, its
/// pass makes every record copied into it meanwhile. Both heaps were taken
/// whole first (their low-water marks raised), and in between passes over
/// the other heap and another copy's askings about this one's memory (which
/// take the kernel's written pages) handed out numbers above its mark: its
/// stamps only rise, and its take asks for those above its mark.
#[test]
fn an_unbound_heap_catches_up_when_bound() {
    let (bound, unbound) = (
        Placed::new(descriptor::SAMPLERS),
        Placed::new(descriptor::SAMPLERS),
    );
    let (Some(host), Some(other)) = (bound.host(), unbound.host()) else {
        return; // no host GPU
    };
    let mut elsewhere = [0u64; 2];
    let source = elsewhere.as_mut_ptr() as u64;
    let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
    descriptor::create_sampler(std::ptr::null(), &sampler, source);
    // Both bound once: each taken whole.
    copy_record(source, bound.at(SLOTS - 1), 16);
    copy_record(source, unbound.at(SLOTS - 1), 16);
    descriptor::sync_heaps(&[bound.registered(), unbound.registered()]);
    assert!(host.made(SLOTS - 1) && other.made(SLOTS - 1));
    let (from, to) = (unbound.at(0) as usize, unbound.at(SLOTS - 1) as usize + 16);
    for round in 0..8 {
        copy_record(source, bound.at(round), 16);
        copy_record(source, unbound.at(round), 16);
        // Another copy over the unbound heap's memory asks meanwhile.
        crate::watch::writes(from, to, true, 0).expect("write-watched");
        descriptor::sync_heaps(&[bound.registered()]);
        assert!(host.made(round), "round {round}: the bound heap's record");
        assert!(
            !other.made(round),
            "round {round}: the unbound heap is left"
        );
    }
    descriptor::sync_heaps(&[unbound.registered()]);
    for round in 0..8 {
        assert!(other.made(round), "slot {round}: caught up");
    }
}

/// A pass looks at the heaps of the requests it covers, each once, and a
/// pass of its own at those of the last pass that covered requests.
#[test]
fn a_pass_looks_at_the_heaps_of_the_requests_it_covers() {
    use std::sync::mpsc::channel;
    let (came, come) = channel::<Vec<usize>>();
    let (go, gone) = channel::<()>();
    let gone = std::sync::Mutex::new(gone);
    let ahead = Ahead::start_with(
        move |heaps: &[usize]| {
            let _ = came.send(heaps.to_vec());
            let _ = gone.lock().unwrap().recv();
        },
        Some(Duration::from_millis(2)),
    )
    .expect("a helper thread");
    let first = ahead.request(&[1]).expect("a request");
    let seen = come.recv_timeout(Duration::from_secs(30)).expect("a pass");
    assert_eq!(seen, [1]);
    // Asked for while the first pass is held: the next pass's.
    ahead.request(&[2]).expect("a request");
    let third = ahead.request(&[3, 2]).expect("a request");
    go.send(()).unwrap();
    assert!(ahead.wait(first));
    let seen = come.recv_timeout(Duration::from_secs(30)).expect("a pass");
    assert_eq!(seen, [2, 3]);
    go.send(()).unwrap();
    assert!(ahead.wait(third));
    // A pass of its own, while requests are recent.
    let seen = come
        .recv_timeout(Duration::from_secs(30))
        .expect("a pass of its own");
    assert_eq!(seen, [2, 3]);
    assert_eq!(ahead.behind(third), 0);
    // (The helper ends once its passes stop being held.)
    drop(go);
}

/// With `VICTORY_SAME_RECORDS`, records copied onto the same bytes leave
/// their page unwritten for the write watch, and records that differ are
/// copied and mark it, as without the switch (which writes the same bytes
/// too). In write-watched memory that is no heap: no pass of another test
/// takes its pages.
#[test]
fn records_copied_onto_the_same_bytes_leave_the_page_unwritten() {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, ResetWriteWatch,
        VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let page = crate::watch::PAGE;
    // SAFETY: new memory.
    let memory = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            2 * page,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    };
    assert!(!memory.is_null());
    let (src, dst) = (memory as u64, memory as u64 + page as u64);
    // The kernel pages of `dst`'s page written since the last look.
    let written = || {
        let mut found = vec![std::ptr::null_mut::<std::ffi::c_void>(); page / 4096 + 1];
        let (mut count, mut granularity) = (found.len(), 0u32);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                1,
                dst as *const _,
                page,
                found.as_mut_ptr(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(failed, 0);
        count
    };
    // Records 0..8 of both alike, as the title left them.
    for k in 0..8u64 {
        // SAFETY: records of the allocation.
        unsafe {
            (dst as *mut [u32; 8])
                .add(k as usize)
                .write([k as u32 + 1; 8]);
            (src as *mut [u32; 8])
                .add(k as usize)
                .write([k as u32 + 1; 8]);
        }
    }
    // SAFETY: the allocation's own.
    assert_eq!(unsafe { ResetWriteWatch(memory, 2 * page) }, 0);
    let _ = written();
    // (On this thread alone: other tests copy records meanwhile.)
    descriptor::SAME_RECORDS_HERE.set(Some(true));
    descriptor::copy_simple(std::ptr::null(), 8, dst, src, descriptor::VIEWS_KIND);
    assert_eq!(written(), 0, "the same records: the page is not written");
    // One record differs: all eight are copied.
    // SAFETY: a record of the allocation.
    unsafe { (src as *mut [u32; 8]).add(5).write([99; 8]) };
    descriptor::copy_simple(std::ptr::null(), 8, dst, src, descriptor::VIEWS_KIND);
    assert!(written() > 0, "a record that differs is written");
    // SAFETY: as above.
    assert_eq!(unsafe { (dst as *const [u32; 8]).add(5).read() }, [99; 8]);
    // Without the switch (whatever the environment says) the same bytes
    // are written, and mark the page.
    descriptor::SAME_RECORDS_HERE.set(Some(false));
    let _ = written();
    descriptor::copy_simple(std::ptr::null(), 8, dst, src, descriptor::VIEWS_KIND);
    descriptor::SAME_RECORDS_HERE.set(None);
    assert!(written() > 0, "without the switch the page is written");
    // SAFETY: this allocation belongs to the test.
    unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
}
