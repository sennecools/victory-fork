//! The heap sync (`super::super::sync_heaps`): changed pages sorted on
//! threads, the early skip (`SKIPPED_MADE`), and the pass run ahead on its
//! own thread (`super::super::ahead`).

use super::made_tests::{alive, key, memory, next, texture, views, write};
use super::*;
use crate::descriptor::ahead::Ahead;
use crate::descriptor::{self as descriptor, HeapDesc, Placement};
use crate::object::{S_OK, This, slot};
use std::sync::atomic::Ordering::{Acquire, Relaxed};
use std::time::Duration;

/// Slots of the heaps these tests make.
const SLOTS: u32 = 64;

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
    title_texture_of(16, 16)
}

/// [`title_texture`] of `width` x `height` texels.
fn title_texture_of(width: u64, height: u32) -> This {
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
        Width: width,
        Height: height,
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

/// The early skip: a slot made from a record of a view still there is done
/// with before its claim (counted as skipped); a slot made from a record
/// whose resource was released since goes on to the claim and is made
/// null, as without the skip.
#[test]
fn the_early_skip_takes_no_claim_and_still_nulls_a_released_view() {
    let _census = crate::census::lock_tests();
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
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
    let _census = crate::census::lock_tests();
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
    let _census = crate::census::lock_tests();
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
    let _census = crate::census::lock_tests();
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
    let _census = crate::census::lock_tests();
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
    let _census = crate::census::lock_tests();
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

/// A view of a texture object with no host copy yet, mirrored not for a
/// draw, waits for its texture and keeps no record of a buffer left for a
/// draw ([`HostHeap::left`]): mirrored again, it is looked at under the
/// claim each time, so its wait and the queue of textures are told again.
/// With the sets of buffer views as sets and as bits.
#[test]
fn left_not_set_for_waiting_texture_deferral() {
    let _census = crate::census::lock_tests();
    for bits in [false, true] {
        let Some(heap) = HostHeap::with_ways(false, SLOTS, bits, true) else {
            return; // no host GPU
        };
        let title = title_texture();
        let view = Some(View::Shader(title as usize, None));
        // Left to the process: the texture's wait holds the handle.
        let records = Box::leak(Box::new([[0u32; 8]; 2]));
        let at = records.as_mut_ptr() as u64 + 32;
        write(at, &view);
        heap.mirror(1, at, false);
        assert!(!heap.made(1), "bits {bits}: waits for its texture");
        assert_eq!(heap.left_key(1), 0, "bits {bits}");
        BEFORE_CLAIM.set(Some(|_| {}));
        heap.mirror(1, at, false);
        assert!(BEFORE_CLAIM.take().is_none(), "bits {bits}: claimed again");
        assert_eq!(heap.left_key(1), 0, "bits {bits}");
        assert_eq!(heap.left_skips.load(Relaxed), 0, "bits {bits}");
        // Made by a draw, with its texture.
        heap.mirror(1, at, true);
        assert!(heap.made(1), "bits {bits}");
    }
    assert!(alive());
}

/// The black census counts the slots left unmade for a texture that waits
/// for its host copy (`wait_for`), the slots made late once it is there
/// (`make_waiting`), and shows both waiting gauges. A slot made null for a
/// texture the host refused (`refused_for_now`) is counted on its own.
#[test]
fn the_census_counts_slots_nulled_for_a_waiting_texture_and_made_late() {
    let _census = crate::census::lock_tests();
    use crate::census::VIEW_NULL_WAITING;
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
    // (Other tests that make slots reach these counters too, so they are
    // asked for "at least"; what must hold for this test's own slots is
    // asked of the slots.)
    let [late, waited] = [1, 2].map(|i| VIEW_NULL_WAITING[i].load(Relaxed));
    let ahead = Ahead::start(descriptor::sync_heaps).expect("a helper thread");
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    crate::descriptor::ahead::wait_and_make(Some(&ahead), Some(seq), &[placed.registered()]);
    assert!(crate::resource::host_waits(title));
    assert!(
        VIEW_NULL_WAITING[2].load(Relaxed) >= waited + 4,
        "four slots left unmade for the waiting texture"
    );
    for s in 0..4 {
        assert!(!host.made(s), "slot {s} left for its texture, not made");
    }
    let waiting = crate::descriptor::waiting_gauge();
    assert!(
        waiting[0] >= 1 && waiting[1] >= 4,
        "the gauge sees the waiting slots"
    );
    // Made by a draw, then the slots by the next wait.
    assert!(crate::resource::host_copy(title).is_some());
    let seq = ahead.request(&[placed.registered()]).expect("a request");
    crate::descriptor::ahead::wait_and_make(Some(&ahead), Some(seq), &[placed.registered()]);
    assert!(
        VIEW_NULL_WAITING[1].load(Relaxed) >= late + 4,
        "four slots made late with their texture"
    );
    for s in 0..4 {
        assert!(host.made(s), "slot {s} made with its texture");
    }
    assert!(alive());
}

/// A slot made null for a texture without its host copy is counted, and
/// returns the texture the slot waits for (`refused_for_now`).
#[test]
fn the_census_counts_a_slot_made_null_for_a_waiting_texture() {
    let _census = crate::census::lock_tests();
    use crate::census::VIEW_NULL_WAITING;
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let title = title_texture();
    let view = View::Shader(title as usize, None);
    let nulled = VIEW_NULL_WAITING[0].load(Relaxed);
    let mut made = 7;
    assert_eq!(
        heap.refused_for_now(&mut made, 3, &view, None),
        Some(title as usize)
    );
    assert_eq!(made, 0, "counted as not made");
    assert!(VIEW_NULL_WAITING[0].load(Relaxed) > nulled);
    // A view of nothing that waits is no such slot: it says so itself.
    let buffer = View::Constant(0x7d_5000_0000, 256);
    let mut made = 7;
    assert_eq!(heap.refused_for_now(&mut made, 3, &buffer, None), None);
    assert_eq!(made, 7, "left as it was");
}

/// A look at a slot past the host heap is counted, with the highest seen.
#[test]
fn the_census_counts_a_slot_past_the_host_heap() {
    let _census = crate::census::lock_tests();
    use crate::census::PAST_HEAP;
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let seen = PAST_HEAP[0].load(Relaxed);
    assert_eq!(heap.mirror(SLOTS + 3, 0, true), None);
    assert!(PAST_HEAP[0].load(Relaxed) > seen);
    assert!(PAST_HEAP[1].load(Relaxed) >= u64::from(SLOTS + 3));
}

/// Four threads write views into a placed heap (`Create*View`), copy
/// records into it as bytes and by calls (`CopyDescriptorsSimple`) and
/// make slots for draws that read them through tables, while submissions
/// ask the helper for passes and wait for them under the submission lock:
/// all finish, and every slot's word is of the kind
/// of the descriptor under it, during and after.
#[test]
fn many_writers_and_the_pass_ahead_leave_each_slot_with_its_own_word() {
    let _census = crate::census::lock_tests();
    const THREADS: u64 = 4;
    let placed = Placed::new(descriptor::VIEWS_KIND);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
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

/// The reference the sorted pass is held to, one record at a time: a
/// record whose key is not the one last seen in its slot (`last`, 0 before
/// the first look), or whose slot is made from another record, of a slot no
/// table's lately, is seen from then on and its slot mirrored as the sync
/// mirrors it. True if it was.
fn mirror_one(heap: &HostHeap, last: &mut [u64], index: u32, handle: u64) -> bool {
    let record = record_key(handle);
    let made = key(heap, index);
    if record == last[index as usize] && (made == record || made == 0) {
        return false;
    }
    if heap.tabled_lately(index, crate::frame::current()) {
        return false;
    }
    last[index as usize] = record;
    heap.mirror_synced(index, handle);
    true
}

/// Records over a page, some slots a table's lately, some made by a draw,
/// some written with bytes that are no view, mirrored one by one
/// ([`mirror_one`]) on one heap, and on another sorted in parts of the
/// page on four threads (`sort_copied_page`) and the slots found mirrored
/// on four threads (`mirror_synced`): after every round the same slots
/// made from the same records, with the same words and buffer sets, and
/// the same keys last seen.
#[test]
fn pages_sorted_on_threads_make_what_one_by_one_makes() {
    let (Some(one), Some(sorted)) = (HostHeap::new(false, SLOTS), HostHeap::new(false, SLOTS))
    else {
        return; // no host GPU
    };
    let views = views(memory());
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let step = 32u64;
    let mut rng = 0x0dd5_eed5_1f1a_4330u64 ^ 1;
    let (mut mirrored, mut left) = (0u64, 0u64);
    let mut last = vec![0u64; SLOTS as usize];
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
                // A mirror that finds no view notes what it read as seen.
                let record = record_key(handle);
                if record == 0 || view_at(record as u32, (record >> 32) as u32).is_none() {
                    last[s as usize] = record;
                }
            }
        }
        // SAFETY: `records` is `SLOTS` records of `step` bytes.
        let page = unsafe {
            std::slice::from_raw_parts(base as *const u8, (u64::from(SLOTS) * step) as usize)
        }
        .to_vec();
        let mut one_by_one = 0;
        for index in 0..SLOTS {
            if mirror_one(&one, &mut last, index, base + u64::from(index) * step) {
                one_by_one += 1;
            }
        }
        // The page in eight parts of 256 bytes, as ranges the watch gives.
        let parts: Vec<(usize, usize)> = (0..8)
            .map(|p| (base as usize + p * 256, base as usize + (p + 1) * 256))
            .collect();
        let found: Vec<std::sync::Mutex<SyncPart>> =
            parts.iter().map(|_| Default::default()).collect();
        crate::draw::each_on(4, parts.len(), "test-sync", &|n| {
            let (from, to) = parts[n];
            let mut part = SyncPart::default();
            let at = from - base as usize;
            let slots = (at as u64 / step)..((to - base as usize) as u64).div_ceil(step);
            sorted.sort_copied_page(
                slots,
                (base, step),
                from,
                &page[at..to - base as usize],
                &mut part,
            );
            *found[n].lock().unwrap() = part;
        });
        let mut make = Vec::new();
        for part in found {
            let mut part = part.into_inner().unwrap();
            assert_eq!(part.looked, 8, "round {round}");
            make.append(&mut part.make);
        }
        assert!(make.is_sorted(), "round {round}");
        crate::draw::each_on(4, make.len().div_ceil(4), "test-sync", &|n| {
            for &index in make.chunks(4).nth(n).unwrap_or_default() {
                sorted.mirror_synced(index, base + u64::from(index) * step);
            }
        });
        assert_eq!(make.len() as u64, one_by_one, "round {round}");
        mirrored += one_by_one;
        left += u64::from(SLOTS) - one_by_one;
        for s in 0..SLOTS {
            assert_eq!(key(&one, s), key(&sorted, s), "round {round}, slot {s}");
            assert_eq!(one.word(s), sorted.word(s), "round {round}, slot {s}");
            assert_eq!(
                sorted.seen_key(s),
                Some(last[s as usize]),
                "round {round}, slot {s}"
            );
        }
        assert_eq!(one.buffer_slots(), sorted.buffer_slots(), "round {round}");
        words_agree(&sorted);
    }
    assert!(mirrored > 0 && left > 0);
    assert!(alive());
}

/// The keys of records as the pass reads them from a page: the view index
/// and generation of a record of ours, 0 for any other bytes.
#[test]
fn a_page_key_is_the_view_a_record_names() {
    let mut record = [0u8; 32];
    record[..4].copy_from_slice(&VIEW_TAG.to_le_bytes());
    record[4..8].copy_from_slice(&7u32.to_le_bytes());
    record[8..12].copy_from_slice(&3u32.to_le_bytes());
    // Bytes past the first 12 name nothing.
    record[12..].fill(0xa5);
    assert_eq!(page_key(&record), 3 << 32 | 7);
    record[0] ^= 1;
    assert_eq!(page_key(&record), 0);
    assert_eq!(page_key(&record[..8]), 0);
}

/// Passes of its own (`EARLY`): none before the first request; some after
/// it while requests are recent, none of which answers a request (one made
/// during such a pass, held, waits for a pass of its own); none once
/// requests have stopped.
#[test]
fn passes_of_its_own_run_between_requests_and_answer_none() {
    let _census = crate::census::lock_tests();
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
    let _census = crate::census::lock_tests();
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

/// Records copied onto the same bytes leave their page unwritten for the
/// write watch, and records that differ are copied and mark it. In
/// write-watched memory that is no heap: no pass of another test takes its
/// pages.
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
    descriptor::copy_simple(std::ptr::null(), 8, dst, src, descriptor::VIEWS_KIND);
    assert_eq!(written(), 0, "the same records: the page is not written");
    // One record differs: all eight are copied.
    // SAFETY: a record of the allocation.
    unsafe { (src as *mut [u32; 8]).add(5).write([99; 8]) };
    descriptor::copy_simple(std::ptr::null(), 8, dst, src, descriptor::VIEWS_KIND);
    assert!(written() > 0, "a record that differs is written");
    // SAFETY: as above.
    assert_eq!(unsafe { (dst as *const [u32; 8]).add(5).read() }, [99; 8]);
    // SAFETY: this allocation belongs to the test.
    unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
}

/// The oracle the heap sync is held to (wave 1, step W1.8): random title
/// work on a placed heap (views written by calls and mirrored at once, records
/// copied in as bytes or by calls, draws that read slots through tables or
/// make tables, table marks that age, bytes that are no view, released
/// textures), and at every submission the pass ahead
/// ([`crate::descriptor::ahead::wait_and_make`]) against a reference heap
/// that is mirrored slowly: every slot no table had lately, made again from
/// the record the title holds now. After each submission each slot of the
/// two must agree on what it was made from (`made`), its word in the table of
/// strides and the host resource it keeps, and what it was made from must be
/// the record the title holds there (or nothing, for a view left for later).
/// The reference keeps no record of what its slots were left for a draw
/// from ([`HostHeap::left`]), so it makes every such mirror in full: the
/// heap's skips of buffer views left for a draw again are checked against
/// it. Calls write the record a slot holds again now and then.
/// Every [`GPU_EVERY`] submissions a compute read through a table of the
/// placed heap's host slots checks that the slots made from views of texture
/// objects show those textures.
///
/// Left out of the comparison, as the pass leaves them by design:
/// - slots a table had lately: their draws make them, and the pass looks at
///   them again next time;
/// - slots that went on showing an older view of the same placed texture
///   (`keep_views`);
/// - slots made from a record whose resource was released since, where the
///   record did not change: nothing tells the pass of a release, and the
///   slot keeps its host resource until it is written;
/// - placed texture views made null in one heap alone as stale (`Spans`)
///   while that heap holds a rival view of the same memory: which of the two
///   is the newer depends on the order the writes were seen in;
/// - views left for later (made only for draws) that one heap shows from a
///   draw of the same record and the other made null at a write of another
///   view it saw in between;
/// - the word of a slot made from nothing (a view left for later, or none):
///   it is the word of the last write the mirror saw, which differs.
struct Oracle {
    placed: Placed,
    reference: HostHeap,
    ahead: Ahead,
    views: Vec<Option<View>>,
    /// Texture objects with their host copies (held, so the sweep lets none
    /// go) and their sizes; none in a sampler heap.
    objects: Vec<(This, std::sync::Arc<crate::resource::HostTexture>, [u32; 2])>,
    /// Records outside the heap to copy from.
    scratch: Box<[[u32; 8]; 2]>,
    gpu: Option<GpuCheck>,
    /// Submissions, slots compared, slots left out (tables', kept, released),
    /// GPU checks, placed texture views left out as stale in one heap alone,
    /// views left for later in one heap alone.
    counts: [u64; 6],
}

/// Slots of the heaps the oracle makes.
const ORACLE_SLOTS: u32 = 64;

/// Submissions between two GPU checks of the oracle.
const GPU_EVERY: u64 = 8;

impl Oracle {
    /// An oracle over a heap of `kind`. `None` with no host GPU.
    fn new(kind: u32) -> Option<Self> {
        let placed = Placed::with_slots(kind, ORACLE_SLOTS);
        placed.host()?;
        let samplers = kind == descriptor::SAMPLERS;
        let reference = HostHeap::with_ways(samplers, ORACLE_SLOTS, slot_bits(), false)?;
        let (views, objects) = if samplers {
            let sampler = |filter: u32, address: u32| {
                let mut desc = [filter, address, address, address, 0, 1, 0, 0, 0, 0, 0, 0, 0];
                desc[11] = 0x7f7f_ffff;
                Some(View::Sampler(desc))
            };
            (
                vec![
                    sampler(21, 3),
                    sampler(0, 1),
                    sampler(21, 1),
                    sampler(1, 2),
                    None,
                ],
                Vec::new(),
            )
        } else {
            let objects: Vec<_> = [[16u32, 16u32], [32, 8], [8, 4]]
                .into_iter()
                .map(|[width, height]| {
                    let texture = title_texture_of(u64::from(width), height);
                    let host = crate::resource::host_copy(texture).expect("a host copy");
                    (texture, host, [width, height])
                })
                .collect();
            let mut views = views(memory());
            views.extend(
                objects
                    .iter()
                    .map(|&(t, _, _)| Some(View::Shader(t as usize, None))),
            );
            (views, objects)
        };
        Some(Self {
            placed,
            reference,
            ahead: Ahead::start(descriptor::sync_heaps)?,
            views,
            objects,
            scratch: Box::new([[0; 8]; 2]),
            gpu: if samplers { None } else { GpuCheck::new() },
            counts: [0; 6],
        })
    }

    fn real(&self) -> &HostHeap {
        self.placed.host().expect("the host heap")
    }

    /// The record's bytes of the heap's kind.
    fn step(&self) -> usize {
        if self.reference.sampler { 16 } else { 32 }
    }

    /// `view` written at `handle` by a call.
    fn call(&mut self, s: u32, view: &Option<View>) {
        let at = self.placed.at(s);
        match view {
            Some(view) => descriptor::record(at, view.clone()),
            // No view: zeros copied in by a call.
            None => {
                self.scratch[0] = [0; 8];
                let from = self.scratch.as_ptr() as u64;
                descriptor::copy_simple(std::ptr::null(), 1, at, from, self.kind());
            }
        }
    }

    fn kind(&self) -> u32 {
        if self.reference.sampler {
            descriptor::SAMPLERS
        } else {
            descriptor::VIEWS_KIND
        }
    }

    /// `view` copied into slot `s` from a record outside the heap: by a call
    /// (mirrored at once), or as bytes.
    fn copy(&mut self, s: u32, view: &Option<View>, by_call: bool) {
        let from = self.scratch.as_ptr() as u64 + 32;
        write(from, view);
        if by_call {
            descriptor::copy_simple(std::ptr::null(), 1, self.placed.at(s), from, self.kind());
        } else {
            copy_record(from, self.placed.at(s), self.step());
        }
    }

    /// Bytes that are no view of ours written into slot `s`.
    fn garbage(&mut self, s: u32, r: u64) {
        let words = [
            0x7e57_0000 | (r >> 20) as u32 & 0xffff,
            (r >> 36) as u32,
            1,
            2,
        ];
        // SAFETY: a record of the heap, at least 16 bytes.
        unsafe { (self.placed.at(s) as *mut [u32; 4]).write_unaligned(words) };
    }

    /// A draw reads slot `s` through a table (`tabled`: and marks it a
    /// table's first), in both heaps.
    fn draw(&mut self, s: u32, tabled: Option<bool>) {
        let at = self.placed.at(s);
        for heap in [self.real(), &self.reference] {
            if let Some(read) = tabled {
                heap.texture_slot_tabled(s, read);
            }
            heap.mirror(s, at, true);
        }
    }

    /// Slot `s` no table's for long: its stamp aged out, in both heaps.
    /// Marks age by frames, so the next pass is of a later frame, and looks
    /// for aged marks again ([`HostHeap::retake_left`]).
    fn age(&self, s: u32) {
        for heap in [self.real(), &self.reference] {
            if let Some(stamp) = heap.tabled.get(s as usize) {
                stamp.store(0, Relaxed);
            }
            heap.left_scanned.store(u64::MAX, Relaxed);
        }
    }

    /// One random step of title work.
    fn step_once(&mut self, rng: &mut u64) {
        let r = next(rng);
        let s = (r % u64::from(ORACLE_SLOTS)) as u32;
        let view = self.views[(r >> 16) as usize % self.views.len()].clone();
        match r >> 8 & 15 {
            // The record the slot holds, written again.
            0 if r >> 40 & 1 == 0 => {
                let again = descriptor::view(self.placed.at(s));
                self.call(s, &again);
            }
            0..=3 => self.call(s, &view),
            4..=6 => self.copy(s, &view, false),
            7 | 8 => self.copy(s, &view, true),
            9 | 10 => self.draw(s, None),
            11 => self.draw(s, Some(r >> 40 & 1 == 0)),
            12 => self.age(s),
            13 => self.garbage(s, r),
            14 if r >> 40 & 7 == 0 && !self.objects.is_empty() => {
                let (texture, _, _) = self.objects[(r >> 44) as usize % self.objects.len()];
                descriptor::forget(texture as usize);
            }
            _ => self.submit(),
        }
    }

    /// A submission: the pass ahead for the heap and its wait, the
    /// reference's slow mirror, and the comparison.
    fn submit(&mut self) {
        let heaps = [self.placed.registered()];
        let seq = self.ahead.request(&heaps).expect("a request");
        crate::descriptor::ahead::wait_and_make(Some(&self.ahead), Some(seq), &heaps);
        let frame = crate::frame::current();
        for s in 0..ORACLE_SLOTS {
            if !self.reference.tabled_lately(s, frame) {
                self.reference.mirror_synced(s, self.placed.at(s));
            }
        }
        self.counts[0] += 1;
        self.compare(frame);
        if self.counts[0].is_multiple_of(GPU_EVERY) {
            self.gpu_check();
        }
    }

    /// The comparison of every slot after a submission.
    fn compare(&mut self, frame: u64) {
        let n = self.counts[0];
        for s in 0..ORACLE_SLOTS {
            let real = self.placed.host().expect("the host heap");
            if real.tabled_lately(s, frame) {
                self.counts[2] += 1;
                continue;
            }
            let record = record_key(self.placed.at(s));
            let (made, slow) = (key(real, s), key(&self.reference, s));
            let of = |k: u64| {
                (k != 0)
                    .then(|| view_at(k as u32, (k >> 32) as u32))
                    .flatten()
            };
            // A record of a released view, unchanged since: the pass does
            // not look at it.
            let released = record != 0 && of(record).is_none() && made == record;
            // An older view of the same placed texture, shown on.
            let kept = matches!(
                (of(made), of(slow)),
                (Some(a), Some(b)) if a != b && same_placed_texture(&a, &b)
            );
            if released || kept {
                self.counts[2] += 1;
                continue;
            }
            // A placed texture view made null in one heap alone, as stale: a
            // rival there, another slot made from a view of a texture placed
            // at the same memory that is no view of the record's texture,
            // made it stale (`Spans`). The spans see the writes in another
            // order in each heap (the title's calls at once, the slow mirror
            // at the submission, slot by slot), so which of the two is the
            // newer differs. Both null is compared as any other slot.
            if (made == 0) != (slow == 0)
                && let Some(view) = of(record)
                && self.rival_in(if made == 0 { real } else { &self.reference }, s, &view)
            {
                self.counts[4] += 1;
                continue;
            }
            // A view left for later (of placed memory, or a buffer's: made
            // only for a draw) shown in one heap, from a draw of the same
            // record, and made null in the other by a mirror not for a draw:
            // one heap saw a write of another view in between and the other
            // did not (a table held the slot meanwhile, so the slow mirror
            // left it). Neither heap's pass skipped a record it should have
            // looked at.
            if (made == 0) != (slow == 0)
                && made.max(slow) == record
                && of(record).is_some_and(|view| HostHeap::deferred(&view, false).1)
            {
                self.counts[5] += 1;
                continue;
            }
            assert_eq!(
                made,
                slow,
                "the oracle: slot {s} made from another record, submission {n} \
                 (record {:?}, slow {:?})",
                of(record),
                of(slow)
            );
            assert!(
                made == record || made == 0,
                "the oracle: slot {s} made from {made:#x}, its record is {record:#x}, \
                 submission {n}"
            );
            // A slot made from nothing holds the word of what was last
            // written into it, a view left for later or none: the pass and
            // the slow mirror see other last writes.
            if made != 0 {
                assert_eq!(
                    real.word(s),
                    self.reference.word(s),
                    "the oracle: slot {s} word, submission {n}"
                );
            }
            assert_eq!(
                real.kept_raw(s),
                self.reference.kept_raw(s),
                "the oracle: slot {s} keeps another resource, submission {n}"
            );
            self.counts[1] += 1;
        }
    }

    /// Whether `heap` has a slot other than `s` made from a view of a texture
    /// placed at the memory of `view`'s, of another texture than `view`'s
    /// (one that makes `view` stale in the spans). `view` must be a placed
    /// texture view: not a placed buffer (`resource[0] != 1`, the dimension
    /// word of the resource description, 1 for a buffer).
    fn rival_in(&self, heap: &HostHeap, s: u32, view: &View) -> bool {
        let View::Placed(false, resource, desc) = view else {
            return false;
        };
        if resource[0] == 1 {
            return false;
        }
        (0..ORACLE_SLOTS).filter(|&other| other != s).any(|other| {
            let made = key(heap, other);
            let rival = (made != 0)
                .then(|| view_at(made as u32, (made >> 32) as u32))
                .flatten();
            matches!(
                &rival,
                Some(rival @ View::Placed(false, other_resource, other_desc))
                    if other_resource[0] != 1
                        && other_desc[10..12] == desc[10..12]
                        && !same_placed_texture(rival, view)
            )
        })
    }

    /// The slots of the placed heap made from a view of a texture object, read
    /// on the card through a table: each shows its texture's size.
    fn gpu_check(&mut self) {
        let Some(gpu) = &self.gpu else { return };
        let mut slots = Vec::new();
        let mut sizes = Vec::new();
        for s in 0..ORACLE_SLOTS {
            let made = key(self.real(), s);
            if made == 0 || made != record_key(self.placed.at(s)) {
                continue;
            }
            if let Some(View::Shader(t, None)) = view_at(made as u32, (made >> 32) as u32)
                && let Some((_, host, size)) = self.objects.iter().find(|o| o.0 as usize == t)
            {
                assert_eq!(
                    self.real().kept_raw(s),
                    Some(windows::core::Interface::as_raw(&host.texture)),
                    "the oracle: slot {s} keeps its texture"
                );
                slots.push(s);
                sizes.push(*size);
            }
        }
        if slots.is_empty() {
            return;
        }
        let read = gpu
            .sizes(self.real(), &slots)
            .expect("the card ran every dispatch");
        assert_eq!(read, sizes, "the oracle: the card reads slots {slots:?}");
        self.counts[3] += 1;
    }
}

/// A compute program that reads the size of the texture in a slot of a view
/// heap through a table of that one slot.
struct GpuCheck {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    out: ID3D12Resource,
    data: *mut u8,
}

impl GpuCheck {
    fn new() -> Option<Self> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::core::s;
        const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0), RootConstants(num32BitConstants = 1, b0)"
Texture2D<float4> t : register(t0);
RWByteAddressBuffer o : register(u0);
cbuffer Slot : register(b0) { uint at; };
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint w, h;
    t.GetDimensions(w, h);
    o.Store2(8 * at, uint2(w, h));
}
"#;
        let host = crate::host::host()?;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names outlive the call; the out pointer is ours.
        unsafe {
            D3DCompile(
                SOURCE.as_ptr().cast(),
                SOURCE.len(),
                s!("oracle"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )
        }
        .expect("the oracle's shader compiles");
        let code = code?;
        // SAFETY: the blob's own bytes.
        let bytes = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: the root signature is in the shader.
        let root: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytes) }.ok()?;
        // SAFETY: the bytecode and root signature live through the call.
        let pso: ID3D12PipelineState = unsafe {
            host.device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })
        }
        .ok()?;
        let (out, data) = host.shared_buffer(2048)?;
        Some(Self {
            root,
            pso,
            out,
            data,
        })
    }

    /// The size the card reads for the texture in each of `slots` of `heap`,
    /// on a queue of the check's own (the host's submissions are not
    /// involved), waited for; `None` if the card left out a dispatch (it
    /// leaves its mark, all ones, as it was).
    fn sizes(&self, heap: &HostHeap, slots: &[u32]) -> Option<Vec<[u32; 2]>> {
        let device = &crate::host::host().expect("the host GPU").device;
        // SAFETY: the mapping of 2,048 bytes; nothing of ours runs now.
        unsafe { std::ptr::write_bytes(self.data, 0xff, 8 * slots.len()) };
        // SAFETY: plain D3D12 calls on objects of ours; the list's heap,
        // root signature, pipeline and buffer live until it has run, which
        // the fence waits for.
        unsafe {
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("an allocator");
            let l: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .expect("a list");
            l.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&self.pso);
            l.SetComputeRootUnorderedAccessView(1, self.out.GetGPUVirtualAddress());
            // A slot a dispatch, as a table of one view.
            for (k, &s) in slots.iter().enumerate() {
                l.SetComputeRootDescriptorTable(0, heap.gpu_handle(s));
                l.SetComputeRoot32BitConstant(2, k as u32, 0);
                l.Dispatch(1, 1, 1);
            }
            l.Close().expect("the list closes");
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("a queue");
            queue.ExecuteCommandLists(&[Some(windows::core::Interface::cast(&l).expect("a list"))]);
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("a fence");
            queue.Signal(&fence, 1).expect("a signal");
            let until = std::time::Instant::now() + Duration::from_secs(30);
            while fence.GetCompletedValue() < 1 {
                assert!(std::time::Instant::now() < until, "the card ran the list");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        // SAFETY: the mapping; the list has run.
        let read: Vec<[u32; 2]> = (0..slots.len())
            .map(|k| unsafe { self.data.add(8 * k).cast::<[u32; 2]>().read_volatile() })
            .collect();
        (!read.contains(&[u32::MAX; 2])).then_some(read)
    }
}

/// [`Oracle`] over `rounds` submissions of a heap of `kind` from `seed`.
fn oracle_run(kind: u32, seed: u64, rounds: u64) {
    let _census = crate::census::lock_tests();
    let Some(mut oracle) = Oracle::new(kind) else {
        return; // no host GPU
    };
    let mut rng = seed;
    while oracle.counts[0] < rounds {
        oracle.step_once(&mut rng);
    }
    let [submissions, compared, left, gpu, stale, later] = oracle.counts;
    // Few slots are left out as stale or left for later: a rule that hid
    // real faults would hide many.
    assert!(stale * 20 < compared, "stale: {:?}", oracle.counts);
    assert!(later * 20 < compared, "left for later: {:?}", oracle.counts);
    assert!(
        compared > submissions * u64::from(ORACLE_SLOTS) / 4,
        "{:?}",
        oracle.counts
    );
    if kind == descriptor::VIEWS_KIND {
        assert!(left > 0, "slots left out: {:?}", oracle.counts);
        assert!(gpu > 0, "GPU checks: {:?}", oracle.counts);
        assert!(
            oracle.real().left_skips.load(Relaxed) > 0,
            "buffer views left for a draw again: {:?}",
            oracle.counts
        );
    }
    if kind == descriptor::VIEWS_KIND {
        // Every made slot holds its view's word (a released view's slot
        // keeps what it had).
        let real = oracle.real();
        for s in 0..ORACLE_SLOTS {
            let made = key(real, s);
            if let Some(view) = (made != 0)
                .then(|| view_at(made as u32, (made >> 32) as u32))
                .flatten()
            {
                assert_eq!(real.word(s), Some(slot_word(Some(&view))), "slot {s}");
            }
        }
    }
    assert!(alive());
}

/// [`Oracle`] on a view heap, from three seeds.
#[test]
fn the_heap_sync_makes_what_a_slow_mirror_makes_in_a_view_heap() {
    for seed in [
        0x9e37_79b9_7f4a_7c15,
        0x2545_f491_4f6c_dd1d,
        0x6a09_e667_f3bc_c908,
    ] {
        oracle_run(descriptor::VIEWS_KIND, seed, 160);
    }
}

/// [`Oracle`] on a sampler heap.
#[test]
fn the_heap_sync_makes_what_a_slow_mirror_makes_in_a_sampler_heap() {
    for seed in [0x9e37_79b9_7f4a_7c15, 0x0dd5_eed5_1f1a_4330] {
        oracle_run(descriptor::SAMPLERS, seed, 160);
    }
}

/// What comparing records with the ones the pass saw last missed: a slot no
/// table reads, made by a call from a record Y over the record X the pass
/// saw, then given X back as bytes before the next pass. The pass finds X as
/// it saw it; the slot shows Y, so it is mirrored again all the same.
#[test]
fn a_record_written_back_under_a_call_is_made_again() {
    let _census = crate::census::lock_tests();
    let placed = Placed::new(descriptor::VIEWS_KIND);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
    let (x, y) = (title_texture_of(16, 16), title_texture_of(32, 8));
    let _hosts = [x, y].map(|t| crate::resource::host_copy(t).expect("a host copy"));
    let mut elsewhere = [0u64; 4];
    let source = elsewhere.as_mut_ptr() as u64;
    write(source, &Some(View::Shader(x as usize, None)));
    copy_record(source, placed.at(5), 32);
    descriptor::sync_heaps(&[placed.registered()]);
    assert_eq!(key(host, 5), record_key(placed.at(5)), "the pass makes X");
    write(placed.at(5), &Some(View::Shader(y as usize, None)));
    assert_eq!(key(host, 5), record_key(placed.at(5)), "the call makes Y");
    copy_record(source, placed.at(5), 32);
    descriptor::sync_heaps(&[placed.registered()]);
    assert_eq!(
        key(host, 5),
        record_key(placed.at(5)),
        "slot 5 shows another record"
    );
}
/// The oracle's GPU check reads what a slot was made from: slots made from
/// views of three textures of other sizes, read through one-view tables.
#[test]
fn the_oracle_s_card_read_sees_each_slot_s_texture() {
    let _census = crate::census::lock_tests();
    let (Some(heap), Some(gpu)) = (HostHeap::new(false, 8), GpuCheck::new()) else {
        return; // no host GPU
    };
    let sizes = [[16u32, 16u32], [32, 8], [8, 4]];
    let mut records = vec![[0u32; 8]; 8];
    let base = records.as_mut_ptr() as u64;
    let mut hosts = Vec::new();
    for (k, &[width, height]) in sizes.iter().enumerate() {
        let texture = title_texture_of(u64::from(width), height);
        hosts.push(crate::resource::host_copy(texture).expect("a host copy"));
        let at = base + 32 * (2 - k as u64);
        write(at, &Some(View::Shader(texture as usize, None)));
        heap.mirror(2 - k as u32, at, true);
    }
    let read = gpu
        .sizes(&heap, &[0, 1, 2])
        .expect("the card runs the list");
    assert_eq!(read, [sizes[2], sizes[1], sizes[0]]);
}

/// Two view heaps read on the card in one list after the other: each shows
/// its own slots.
#[test]
fn the_oracle_s_card_read_tells_two_heaps_apart() {
    let _census = crate::census::lock_tests();
    let (Some(one), Some(two), Some(gpu)) = (
        HostHeap::new(false, 8),
        HostHeap::new(false, 8),
        GpuCheck::new(),
    ) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; 8];
    let base = records.as_mut_ptr() as u64;
    let mut hosts = Vec::new();
    let mut textures = Vec::new();
    for &[width, height] in &[[16u32, 16u32], [32, 8]] {
        let texture = title_texture_of(u64::from(width), height);
        hosts.push(crate::resource::host_copy(texture).expect("a host copy"));
        textures.push(texture);
    }
    for (heap, order) in [(&one, [0, 1]), (&two, [1, 0])] {
        for (s, &k) in order.iter().enumerate() {
            let at = base + 32 * s as u64;
            write(at, &Some(View::Shader(textures[k] as usize, None)));
            heap.mirror(s as u32, at, true);
        }
    }
    assert_eq!(
        gpu.sizes(&one, &[0, 1]),
        Some(vec![[16, 16], [32, 8]]),
        "the first heap"
    );
    assert_eq!(
        gpu.sizes(&two, &[0, 1]),
        Some(vec![[32, 8], [16, 16]]),
        "the second heap"
    );
    assert_eq!(
        gpu.sizes(&one, &[0, 1]),
        Some(vec![[16, 16], [32, 8]]),
        "the first heap again"
    );
}

/// Views of two texture objects with host copies (held: no sweep lets them
/// go), which a mirror not for a draw makes at once.
fn object_views() -> (
    Vec<Option<View>>,
    Vec<std::sync::Arc<crate::resource::HostTexture>>,
) {
    let textures = [title_texture_of(16, 16), title_texture_of(32, 8)];
    let hosts = textures
        .iter()
        .map(|&t| crate::resource::host_copy(t).expect("a host copy"))
        .collect();
    (
        textures
            .iter()
            .map(|&t| Some(View::Shader(t as usize, None)))
            .collect(),
        hosts,
    )
}

/// The slots a pass over `records` (`SLOTS` records of 32 bytes at `base`)
/// of `heap` hands on to be mirrored, and mirrors them.
fn pass_over(heap: &HostHeap, base: u64) -> Vec<u32> {
    let len = SLOTS as usize * 32;
    // SAFETY: the caller's records.
    let page = unsafe { std::slice::from_raw_parts(base as *const u8, len) }.to_vec();
    let mut part = SyncPart::default();
    heap.sort_copied_page(
        0..u64::from(SLOTS),
        (base, 32),
        base as usize,
        &page,
        &mut part,
    );
    for &s in &part.make {
        heap.mirror_synced(s, base + u64::from(s) * 32);
    }
    part.make
}

/// A record a call wrote and mirrored is seen: the next pass leaves it.
#[test]
fn sync_skips_own_api_writes() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    assert_eq!(pass_over(&heap, base), [], "nothing there");
    let at = base + 3 * 32;
    write(at, &views[0]);
    heap.mirror(3, at, false);
    heap.saw_key(3, record_key(at));
    assert_eq!(pass_over(&heap, base), [], "the call's record is seen");
    assert_eq!(key(&heap, 3), record_key(at));
    // Copied in as bytes: handed on.
    write(at, &views[1]);
    assert_eq!(pass_over(&heap, base), [3]);
    assert_eq!(key(&heap, 3), record_key(at));
}

/// A changed record of a slot a table's lately is left to its draws and not
/// taken as seen: once the slot ages out, the next pass hands it on.
#[test]
fn sync_tabled_slot_seen_not_taken() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let at = base + 5 * 32;
    write(at, &views[0]);
    assert_eq!(pass_over(&heap, base), [5]);
    heap.texture_slot_tabled(5, false);
    write(at, &views[1]);
    assert_eq!(pass_over(&heap, base), [], "left to the draws");
    assert_ne!(heap.seen_key(5), Some(record_key(at)), "not seen");
    heap.tabled[5].store(0, Relaxed);
    assert_eq!(pass_over(&heap, base), [5], "looked at again");
    assert_eq!(key(&heap, 5), record_key(at));
}

/// A slot a draw made from a record while a table held it, given back the
/// record the pass saw before, once no table holds it: made again from that
/// record, though the pass saw it last.
#[test]
fn sync_remakes_after_table_mirror_then_writeback() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let at = base + 7 * 32;
    write(at, &views[0]);
    assert_eq!(pass_over(&heap, base), [7]);
    let x = record_key(at);
    heap.texture_slot_tabled(7, false);
    write(at, &views[1]);
    heap.mirror(7, at, true);
    assert_eq!(pass_over(&heap, base), [], "left to the draws");
    heap.tabled[7].store(0, Relaxed);
    write(at, &views[0]);
    assert_eq!(record_key(at), x, "the record the pass saw");
    assert_eq!(pass_over(&heap, base), [7], "the slot shows another record");
    assert_eq!(key(&heap, 7), x);
}

/// Bytes that are no view, into a slot made from none, are not handed on;
/// into a slot made from a view, they are, and the slot is made null.
#[test]
fn shows_none_slot_not_mirrored() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    records[2] = [0x7e57_0001; 8];
    assert_eq!(
        pass_over(&heap, base),
        [],
        "no view into a slot made from none"
    );
    let at = base + 4 * 32;
    write(at, &views[0]);
    assert_eq!(pass_over(&heap, base), [4]);
    assert_ne!(key(&heap, 4), 0);
    records[4] = [0x7e57_0002; 8];
    assert_eq!(pass_over(&heap, base), [4], "no view over a made slot");
    assert_eq!(key(&heap, 4), 0, "made null");
}

/// A record copied in as bytes while its slot is a table's is left to the
/// draws; once the table mark ages, the next pass reads it again even with
/// nothing else written on its page, and the slot shows it.
#[test]
fn a_record_left_to_a_table_is_read_again_once_its_mark_ages() {
    let _census = crate::census::lock_tests();
    let placed = Placed::new(descriptor::VIEWS_KIND);
    let Some(host) = placed.host() else {
        return; // no host GPU
    };
    let textures = [title_texture_of(16, 16), title_texture_of(32, 8)];
    let _hosts = textures.map(|t| crate::resource::host_copy(t).expect("a host copy"));
    let mut elsewhere = [0u64; 8];
    let (x, y) = (
        elsewhere.as_mut_ptr() as u64,
        elsewhere.as_mut_ptr() as u64 + 32,
    );
    write(x, &Some(View::Shader(textures[0] as usize, None)));
    write(y, &Some(View::Shader(textures[1] as usize, None)));
    copy_record(x, placed.at(7), 32);
    descriptor::sync_heaps(&[placed.registered()]);
    assert_eq!(key(host, 7), record_key(placed.at(7)), "made from X");
    host.texture_slot_tabled(7, false);
    copy_record(y, placed.at(7), 32);
    descriptor::sync_heaps(&[placed.registered()]);
    assert_ne!(key(host, 7), record_key(placed.at(7)), "left to the draws");
    // Still a table's: left again.
    descriptor::sync_heaps(&[placed.registered()]);
    assert_ne!(key(host, 7), record_key(placed.at(7)), "left while tabled");
    host.tabled[7].store(0, Relaxed);
    // Marks age by frames, and the pass looks for aged ones once a frame:
    // the next pass is of a later frame.
    host.left_scanned.store(u64::MAX, Relaxed);
    descriptor::sync_heaps(&[placed.registered()]);
    assert_eq!(
        key(host, 7),
        record_key(placed.at(7)),
        "read again once the mark aged"
    );
    // And not again after that: the set holds it no more.
    assert!(host.left_to_tables.lock().unwrap().is_empty());
}

/// The slots left to tables are kept up to a limit: a slot already there is
/// not kept twice, and every slot past the limit is counted, not only the
/// first.
#[test]
fn slots_left_to_tables_past_the_limit_are_counted() {
    let _census = crate::census::lock_tests();
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    for s in 0..SLOTS {
        // Tables hold every slot, so none of them is read again yet.
        heap.texture_slot_tabled(s, false);
    }
    LEFT_MOST_HERE.set(Some(2));
    let dropped = LEFT_DROPPED.load(Relaxed);
    let mut part = SyncPart::default();
    heap.retake_left(&[1, 2, 2, 1, 3, 4, 3], (base, 32), &mut part);
    LEFT_MOST_HERE.set(None);
    assert_eq!(LEFT_DROPPED.load(Relaxed) - dropped, 3, "3, 4 and 3 again");
    let kept: Vec<u32> = heap
        .left_to_tables
        .lock()
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(kept, [1, 2]);
    assert!(part.make.is_empty(), "nothing aged");
}

/// The slots of a heap's [`HostHeap::left_to_tables`] set, in order.
fn left_set(heap: &HostHeap) -> Vec<u32> {
    heap.left_to_tables
        .lock()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

/// The set is looked at for aged marks in the first pass of a frame alone:
/// a later pass of the frame does not look again, with slots left or not,
/// and the first pass of the next frame does.
#[test]
fn warp_retake_scans_once_per_frame() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let at = (records.as_mut_ptr() as u64, 32);
    let frame = 1_000;
    heap.tabled_in(5, false, frame);
    heap.tabled_in(6, false, frame);
    let scans = LEFT_SCANS_HERE.get();
    let mut part = SyncPart::default();
    heap.retake_left_in(&[5], at, &mut part, frame);
    assert_eq!(LEFT_SCANS_HERE.get() - scans, 1);
    heap.retake_left_in(&[6], at, &mut part, frame);
    heap.retake_left_in(&[], at, &mut part, frame);
    assert_eq!(LEFT_SCANS_HERE.get() - scans, 1, "once in the frame");
    assert_eq!(left_set(&heap), [5, 6], "slots still noted in a later pass");
    heap.retake_left_in(&[], at, &mut part, frame + 1);
    assert_eq!(LEFT_SCANS_HERE.get() - scans, 2, "again in the next frame");
    assert!(part.make.is_empty(), "nothing aged");
}

/// A slot whose mark ages in a frame is read again by that frame's first
/// pass, and not again by a later one.
#[test]
fn warp_retake_aged_slot_retaken_same_frame() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let at = (records.as_mut_ptr() as u64, 32);
    let frame = 2_000;
    heap.tabled_in(5, false, frame);
    let mut part = SyncPart::default();
    heap.retake_left_in(&[5], at, &mut part, frame);
    heap.retake_left_in(&[], at, &mut part, frame + TABLED_FRAMES);
    assert!(part.make.is_empty(), "still a table's slot lately");
    let aged = frame + TABLED_FRAMES + 1;
    heap.retake_left_in(&[], at, &mut part, aged);
    assert_eq!(part.make, [5], "read again in the frame its mark aged");
    assert!(left_set(&heap).is_empty());
    let mut later = SyncPart::default();
    heap.tabled_in(6, false, aged);
    heap.retake_left_in(&[], at, &mut later, aged);
    heap.retake_left_in(&[6], at, &mut later, aged);
    assert!(later.make.is_empty(), "not read again in the frame");
    assert_eq!(left_set(&heap), [6]);
}

/// With the set full, nothing more goes in, and every slot left that is not
/// in it already is counted once per time it is left.
#[test]
fn warp_retake_full_set_skips_inserts_counts_left() {
    let _census = crate::census::lock_tests();
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let at = (records.as_mut_ptr() as u64, 32);
    let frame = 3_000;
    for s in 0..SLOTS {
        heap.tabled_in(s, false, frame);
    }
    LEFT_MOST_HERE.set(Some(3));
    let mut part = SyncPart::default();
    heap.retake_left_in(&[1, 2, 3], at, &mut part, frame);
    let dropped = LEFT_DROPPED.load(Relaxed);
    heap.retake_left_in(&[3, 4, 1, 5, 5], at, &mut part, frame);
    heap.retake_left_in(&[2, 6], at, &mut part, frame + 1);
    LEFT_MOST_HERE.set(None);
    assert_eq!(LEFT_DROPPED.load(Relaxed) - dropped, 4, "4, 5, 5 and 6");
    assert_eq!(left_set(&heap), [1, 2, 3]);
    assert!(part.make.is_empty(), "nothing aged");
}

/// Against the pass as it was (every pass looked at every slot in the set,
/// and refused each slot not in it once it was full), over frames that
/// advance by none to many between passes: the same slots are read again
/// in the same passes, the set holds the same slots, and the same number
/// of slots are counted as finding no room.
#[test]
fn warp_retake_left_dropped_equals_old_refused_sum() {
    let _census = crate::census::lock_tests();
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let at = (records.as_mut_ptr() as u64, 32);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: u64| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed % n
    };
    const MOST: usize = 12;
    // The reference: each slot's last table frame, and the set.
    let mut stamps = [None::<u64>; SLOTS as usize];
    let mut set = std::collections::BTreeSet::new();
    let lately = |stamps: &[Option<u64>], slot: u32, frame: u64| {
        stamps[slot as usize].is_some_and(|stamp| frame - stamp <= TABLED_FRAMES)
    };
    LEFT_MOST_HERE.set(Some(MOST));
    let (mut frame, mut refused_sum) = (10_000u64, 0u64);
    let dropped = LEFT_DROPPED.load(Relaxed);
    for _ in 0..3_000 {
        frame += match next(10) {
            0..=4 => 0,
            5..=8 => 1 + next(3),
            _ => 100 + next(200),
        };
        for _ in 0..next(6) {
            let slot = next(u64::from(SLOTS)) as u32;
            heap.tabled_in(slot, false, frame);
            stamps[slot as usize] = Some(frame);
        }
        // As the pass leaves them: tables' slots lately, some twice.
        let left: Vec<u32> = (0..next(8))
            .map(|_| next(u64::from(SLOTS)) as u32)
            .filter(|&slot| lately(&stamps, slot, frame))
            .collect();
        let aged: Vec<u32> = set
            .iter()
            .copied()
            .filter(|&slot| !lately(&stamps, slot, frame))
            .collect();
        for slot in &aged {
            set.remove(slot);
        }
        for &slot in &left {
            if set.contains(&slot) {
                continue;
            }
            if set.len() >= MOST {
                refused_sum += 1;
                continue;
            }
            set.insert(slot);
        }
        let mut part = SyncPart::default();
        heap.retake_left_in(&left, at, &mut part, frame);
        assert_eq!(part.make, aged, "frame {frame}");
        assert_eq!(left_set(&heap), set.iter().copied().collect::<Vec<_>>());
    }
    LEFT_MOST_HERE.set(None);
    assert!(refused_sum > 0, "the set was full at times");
    assert_eq!(LEFT_DROPPED.load(Relaxed) - dropped, refused_sum);
}

/// A slot a draw made null from bytes of no view, then given back the live
/// record the pass saw last: the pass mirrors it again, though the key is
/// the one it saw, and the slot shows the record.
#[test]
fn a_null_slot_whose_record_names_a_live_view_is_made_again() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let at = base + 6 * 32;
    write(at, &views[0]);
    assert_eq!(pass_over(&heap, base), [6]);
    let x = record_key(at);
    assert_eq!(key(&heap, 6), x);
    // Bytes of no view, which a draw reads through a table: made null.
    let bytes = records[6];
    records[6] = [0x7e57_0003; 8];
    heap.mirror(6, at, true);
    assert_eq!(key(&heap, 6), 0, "made null by the draw");
    // The live record back, as the pass saw it; the draw's mirror noted the
    // bytes it read as seen.
    records[6] = bytes;
    assert_eq!(heap.seen_key(6), Some(0));
    assert_eq!(pass_over(&heap, base), [6], "mirrored again");
    assert_eq!(key(&heap, 6), x);
    // Bytes of no view over a slot made from none are still left.
    records[9] = [0x7e57_0004; 8];
    assert_eq!(pass_over(&heap, base), [], "left");
}

/// Records of a constant buffer view, a placed buffer view and a placed
/// texture view, which a mirror outside a draw leaves for later, made null
/// by a draw from bytes of no view and then copied back as they were: the
/// next pass mirrors them again, and each slot is back in the set of buffer
/// views or the spans, with its word, as before the draw.
#[test]
fn a_view_left_for_later_and_nulled_by_a_draw_comes_back() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let views = views(memory());
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let slots = [(10u32, 0usize), (11, 2), (12, 3)];
    let (buffer, texture) = (11u32, 12u32);
    for (slot, view) in slots {
        write(base + u64::from(slot) * 32, &views[view]);
    }
    assert_eq!(pass_over(&heap, base), [10, 11, 12]);
    let state = |heap: &HostHeap| {
        (
            heap.buffer_slots(),
            slots.map(|(slot, _)| heap.word(slot)),
            heap.placed.lock().unwrap().by_slot.contains_key(&texture),
        )
    };
    let before = state(&heap);
    assert!(
        before.0.contains(&buffer),
        "the placed buffer view in the set"
    );
    // (A texture view's word is 0: no buffer.)
    assert!(
        before.1[..2].iter().all(|&word| word != Some(0)),
        "words: {:?}",
        before.1
    );
    assert!(before.2, "the placed texture in the spans");
    for (slot, _) in slots {
        let at = base + u64::from(slot) * 32;
        let bytes = records[slot as usize];
        records[slot as usize] = [0x7e57_0005; 8];
        heap.mirror(slot, at, true);
        records[slot as usize] = bytes;
    }
    let nulled = state(&heap);
    assert!(!nulled.0.contains(&buffer), "out of the set after the draw");
    assert_eq!(nulled.1, [Some(0); 3], "no words after the draw");
    assert!(!nulled.2, "out of the spans after the draw");
    assert_eq!(pass_over(&heap, base), [10, 11, 12], "mirrored again");
    assert_eq!(state(&heap), before);
}

thread_local! {
    /// [`a_pass_between_a_viewless_draw_s_read_and_its_claim_is_undone`]:
    /// the records' base, the slot, and the live record to copy back.
    static RACE: std::cell::Cell<(u64, u32, [u32; 8])> =
        const { std::cell::Cell::new((0, 0, [0; 8])) };
}

/// For [`BEFORE_NULL`]: the title copies the live record back, and a pass
/// makes the slot from it.
fn copy_back_and_pass(heap: &HostHeap) {
    let (base, slot, record) = RACE.get();
    // SAFETY: a record of the test's records.
    unsafe { ((base + u64::from(slot) * 32) as *mut [u32; 8]).write_unaligned(record) };
    assert_eq!(
        pass_over(heap, base),
        [slot],
        "the pass makes the live record"
    );
}

/// A draw reads bytes of no view in a slot; before it claims the slot, the
/// title copies a live record over them and a pass makes the slot from it;
/// the draw then makes the slot null. The next pass must make it from the
/// live record again: what the draw read is noted as seen under its claim,
/// after the pass's own note.
#[test]
fn a_pass_between_a_viewless_draw_s_read_and_its_claim_is_undone() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let slot = 6u32;
    let at = base + u64::from(slot) * 32;
    write(at, &views[0]);
    let live = records[slot as usize];
    let live_key = record_key(at);
    records[slot as usize] = [0x7e57_0006; 8];
    assert_eq!(pass_over(&heap, base), [], "bytes of no view, left");
    RACE.set((base, slot, live));
    BEFORE_NULL.set(Some(copy_back_and_pass));
    heap.mirror(slot, at, true);
    assert!(BEFORE_NULL.take().is_none(), "the hook ran");
    assert_eq!(record_key(at), live_key, "the live record is there");
    assert_eq!(key(&heap, slot), 0, "the draw made the slot null");
    assert_eq!(
        pass_over(&heap, base),
        [slot],
        "the next pass makes it again"
    );
    assert_eq!(key(&heap, slot), live_key);
}

/// A pass finds a live record in a slot made from nothing and notes it as
/// seen; before it mirrors the slot, the title writes bytes of no view
/// there, so the mirror, no draw's, is refused its claim; then the title
/// copies the record back. The refused mirror noted what it read, so the
/// next pass hands the slot on and makes it.
#[test]
fn a_refused_viewless_mirror_notes_what_it_read() {
    let Some(heap) = HostHeap::new(false, SLOTS) else {
        return; // no host GPU
    };
    let (views, _hosts) = object_views();
    let mut records = vec![[0u32; 8]; SLOTS as usize];
    let base = records.as_mut_ptr() as u64;
    let slot = 8u32;
    let at = base + u64::from(slot) * 32;
    assert_eq!(pass_over(&heap, base), [], "nothing there");
    write(at, &views[0]);
    let live = records[slot as usize];
    let live_key = record_key(at);
    // The pass's sort: the live record seen, the slot handed on.
    let len = SLOTS as usize * 32;
    // SAFETY: the test's records.
    let page = unsafe { std::slice::from_raw_parts(base as *const u8, len) }.to_vec();
    let mut part = SyncPart::default();
    heap.sort_copied_page(
        0..u64::from(SLOTS),
        (base, 32),
        base as usize,
        &page,
        &mut part,
    );
    assert_eq!(part.make, [slot]);
    assert_eq!(heap.seen_key(slot), Some(live_key));
    // Bytes of no view before the pass's mirror, which is refused.
    records[slot as usize] = [0x7e57_0008; 8];
    heap.mirror_synced(slot, at);
    assert_eq!(key(&heap, slot), 0, "left null");
    // The record back: the next pass makes the slot from it.
    records[slot as usize] = live;
    assert_eq!(pass_over(&heap, base), [slot]);
    assert_eq!(key(&heap, slot), live_key);
}
