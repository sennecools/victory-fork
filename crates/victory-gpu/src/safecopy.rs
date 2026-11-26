//! Copies between our memory and the title's that fail instead of faulting.
//!
//! Title memory may be unmapped, remapped or made read-only by the title's
//! other threads while we copy it, so a copy must report a missing page
//! instead of faulting. Copies once went through the kernel
//! (`ReadProcessMemory`, `NtWriteVirtualMemory`), which does, but each call
//! locks the pages it copies: cool_fox measured some 35,000 calls a frame
//! and 39 ms of kernel time on the submitting thread alone; ours was most
//! of the flush's copy part (15.1 ms, `r5-batch-flush.md`).
//!
//! [`copy`] is one `rep movsb`, and every copy between our memory and the
//! title's (`crate::host`) is made with it. A fault in it is taken by the
//! handler of [`crate::watch`] (`on_write`, the first vectored handler in
//! the process), which opens a block we guard ourselves and lets the copy
//! go on, and otherwise sends it to its failing exit ([`failing_exit`]).
//!
//! Ported from cool_fox 66afb2b (local). Its other helper there (the AVX2
//! store of changed dwords) belongs to the landing of shader writes and is
//! not taken.

std::arch::global_asm!(
    ".globl victory_safe_copy",
    ".globl victory_safe_copy_at",
    ".globl victory_safe_copy_after",
    ".globl victory_safe_copy_failed",
    // rcx destination, rdx source, r8 bytes; eax: whether all were copied.
    // A leaf that leaves the stack alone (the two registers it must give
    // back are kept in volatile ones), so a stack walk through it works.
    "victory_safe_copy:",
    "mov r10, rsi",
    "mov r11, rdi",
    "mov rdi, rcx",
    "mov rsi, rdx",
    "mov rcx, r8",
    "victory_safe_copy_at:",
    "rep movsb",
    "victory_safe_copy_after:",
    "mov eax, 1",
    "mov rsi, r10",
    "mov rdi, r11",
    "ret",
    "victory_safe_copy_failed:",
    "xor eax, eax",
    "mov rsi, r10",
    "mov rdi, r11",
    "ret",
);

unsafe extern "system" {
    fn victory_safe_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    fn victory_safe_copy_at();
    fn victory_safe_copy_after();
    fn victory_safe_copy_failed();
}

/// Copies `len` bytes. False if a page of the source cannot be read or one
/// of the destination cannot be written; then some of the bytes may have
/// been copied. The first copy registers the write watch's handler, which
/// takes the faults.
///
/// # Safety
/// Of the two ranges, what is not title memory must be valid, and no Rust
/// reference may cover the destination.
pub(crate) unsafe fn copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    crate::watch::install();
    // SAFETY: the handler is registered, so a fault on either side leaves
    // through the failing exit ([`failing_exit`]); the rest is the caller's.
    unsafe { victory_safe_copy(dst, src, len) != 0 }
}

/// For the fault handler: where a fault at instruction `at` is to go on, if
/// it is [`copy`]'s.
pub(crate) fn failing_exit(at: usize) -> Option<usize> {
    let from = victory_safe_copy_at as *const () as usize;
    let to = victory_safe_copy_after as *const () as usize;
    (from..to)
        .contains(&at)
        .then_some(victory_safe_copy_failed as *const () as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY,
        PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualProtect,
    };

    struct Pages(*mut u8, usize);

    impl Pages {
        fn new(len: usize) -> Pages {
            // SAFETY: plain allocation.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    len,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            assert!(!at.is_null());
            Pages(at.cast(), len)
        }

        fn fill(&self) {
            for i in 0..self.1 {
                // SAFETY: inside the allocation.
                unsafe { self.0.add(i).write((i * 31 + 7) as u8) };
            }
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn copies_every_length_and_alignment() {
        let (a, b) = (Pages::new(1 << 16), Pages::new(1 << 16));
        a.fill();
        for len in [0usize, 1, 3, 4, 8, 31, 64, 255, 4095, 4096, 4097, 60_000] {
            for (da, sa) in [(0usize, 0usize), (1, 0), (0, 3), (5, 2)] {
                // SAFETY: inside the two allocations.
                unsafe {
                    std::ptr::write_bytes(b.0, 0xEE, b.1);
                    assert!(copy(b.0.add(da), a.0.add(sa), len));
                    let to = std::slice::from_raw_parts(b.0, b.1);
                    let from = std::slice::from_raw_parts(a.0, a.1);
                    assert_eq!(to[da..da + len], from[sa..sa + len]);
                    assert!(to[..da].iter().chain(&to[da + len..]).all(|&x| x == 0xEE));
                }
            }
        }
    }

    /// Missing, forbidden and read-only pages: the copy fails, at any byte,
    /// and the process goes on.
    #[test]
    fn faults_are_failures() {
        let (a, b) = (Pages::new(3 << 12), Pages::new(3 << 12));
        a.fill();
        let mut old = 0;
        // SAFETY: pages of our allocations.
        unsafe {
            assert!(copy(b.0, a.0, b.1));
            VirtualProtect(a.0.add(4096).cast(), 4096, PAGE_NOACCESS, &mut old);
            assert!(!copy(b.0, a.0, b.1));
            assert!(copy(b.0, a.0, 4096));
            assert!(!copy(b.0, a.0.add(8191), 1));
            assert!(copy(b.0, a.0.add(8192), 4096));
            VirtualProtect(a.0.add(4096).cast(), 4096, PAGE_READWRITE, &mut old);
            VirtualFree(a.0.add(8192).cast(), 4096, MEM_DECOMMIT);
            assert!(!copy(b.0, a.0, b.1));
            VirtualProtect(b.0.add(4096).cast(), 4096, PAGE_READONLY, &mut old);
            assert!(!copy(b.0, a.0, 8192));
            assert!(!copy(std::ptr::without_provenance_mut(0x10), a.0, 4));
            assert!(!copy(b.0, std::ptr::without_provenance(usize::MAX - 7), 4));
            VirtualProtect(b.0.add(4096).cast(), 4096, PAGE_READWRITE, &mut old);
        }
        // Many faults, on several threads at once.
        let at = a.0 as usize;
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(move || {
                    let mut to = [0u8; 64];
                    for _ in 0..500 {
                        // SAFETY: a decommitted page and our own buffer.
                        assert!(!unsafe { copy(to.as_mut_ptr(), (at + 8192) as *const u8, 64) });
                    }
                });
            }
        });
    }

    /// What the copies cost (ignored: a measurement): pieces of scattered
    /// memory into plain memory, by the kernel and here.
    #[test]
    #[ignore]
    fn copy_costs() {
        use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const N: usize = 128 << 20;
        let from = Pages::new(N);
        from.fill();
        let to = Pages::new(N);
        for piece in [256usize, 4 << 10, 64 << 10] {
            let count = (N / piece).min(4096);
            // A different, scattered piece each time.
            let at = |i: usize| (i * 7919 % (N / piece)) * piece;
            let time = |f: &dyn Fn(usize) -> bool| {
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let t = std::time::Instant::now();
                    for i in 0..count {
                        assert!(f(at(i)));
                    }
                    best = best.min(t.elapsed().as_secs_f64());
                }
                (
                    best * 1e6 / count as f64,
                    count as f64 * piece as f64 / best / 1e9,
                )
            };
            // SAFETY: inside the allocations.
            let kernel = time(&|o| unsafe {
                let mut done = 0;
                let (src, dst) = (from.0.add(o).cast(), to.0.add(o).cast());
                ReadProcessMemory(GetCurrentProcess(), src, dst, piece, &mut done) != 0
            });
            // SAFETY: as above.
            let here = time(&|o| unsafe { copy(to.0.add(o), from.0.add(o), piece) });
            println!(
                "{piece:>6} bytes: kernel {:.2} us ({:.1} GB/s), rep movsb {:.2} us ({:.1} GB/s)",
                kernel.0, kernel.1, here.0, here.1
            );
        }
    }
}
