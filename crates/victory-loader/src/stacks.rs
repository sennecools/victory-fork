//! Where every thread is: `VICTORY_STACKS=<seconds>[,<seconds>...]` prints
//! each thread's unwound stack as `module+rva` that many seconds after
//! [`dump_stacks_on_schedule`], to find what a title waits on.

use std::ffi::c_void;

use windows_sys::Wdk::System::Threading::{
    NtQueryInformationThread, ThreadBasicInformation, ThreadQuerySetWin32StartAddress,
};
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_FULL_AMD64, GetThreadContext,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread,
    SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{MAX_FRAMES, locate, unwind_into};

/// Reads `VICTORY_STACKS` and, if set, starts a thread that prints every
/// thread's stack at each listed time (seconds from now).
pub fn dump_stacks_on_schedule() {
    let Ok(list) = std::env::var("VICTORY_STACKS") else {
        return;
    };
    let mut times: Vec<f64> = list
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    times.sort_by(f64::total_cmp);
    let start = std::time::Instant::now();
    std::thread::spawn(move || {
        for t in times {
            let at = std::time::Duration::from_secs_f64(t.max(0.0));
            std::thread::sleep(at.saturating_sub(start.elapsed()));
            dump_stacks();
        }
    });
}

/// `THREAD_BASIC_INFORMATION` (ntddk).
#[repr(C)]
struct BasicInfo {
    exit_status: i32,
    teb: *const usize,
    client_id: [usize; 2],
    affinity: usize,
    priority: i32,
    base_priority: i32,
}

/// `GetThreadContext` wants a 16-byte aligned context.
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// Prints every other thread's stack to stderr. Each thread is suspended
/// only while its context is read and unwound into a buffer allocated
/// beforehand (it may hold the heap lock).
pub fn dump_stacks() {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return;
    }
    let mut ids = Vec::new();
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            ids.push(entry.th32ThreadID);
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    eprintln!("VIctory: stacks of {} threads", ids.len());
    for id in ids {
        // SAFETY: opens a thread of our own process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        if thread.is_null() {
            continue;
        }
        let (frames, (priority, base)) = frames_of(thread);
        eprintln!(
            "  thread {id} \"{}\" start {} priority {priority} (base {base})",
            name_of(thread),
            locate(start_of(thread))
        );
        for (i, pc) in frames.into_iter().enumerate() {
            eprintln!("    #{i:<2} {}", locate(pc as usize));
        }
        // SAFETY: we opened it.
        unsafe { CloseHandle(thread) };
    }
}

/// The thread's frames and its current and base scheduling priority (0-31).
fn frames_of(thread: HANDLE) -> (Vec<u64>, (i32, i32)) {
    const STATUS_PENDING: i32 = 0x0000_0103;
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    let mut info: BasicInfo = BasicInfo {
        exit_status: 0,
        teb: std::ptr::null(),
        client_id: [0; 2],
        affinity: 0,
        priority: 0,
        base_priority: 0,
    };
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    context.0.ContextFlags = CONTEXT_FULL_AMD64;
    // SAFETY: suspend the thread BEFORE querying and reading its TEB so the
    // thread cannot exit (unmapping its TEB) or grow its stack (invalidating
    // StackLimit) between reading `info.teb` and capturing its context.
    unsafe {
        if SuspendThread(thread) == u32::MAX {
            return (frames, (0, 0));
        }
        let status = NtQueryInformationThread(
            thread,
            ThreadBasicInformation,
            (&raw mut info).cast(),
            size_of::<BasicInfo>() as u32,
            std::ptr::null_mut(),
        );
        if status == 0 && info.exit_status == STATUS_PENDING && !info.teb.is_null() {
            // SAFETY: the thread is suspended and still running (`STATUS_PENDING`),
            // so its TEB is mapped; NT_TIB holds StackBase at +8 and StackLimit at +0x10.
            let (high, low) = (info.teb.add(1).read(), info.teb.add(2).read());
            if low <= high && GetThreadContext(thread, &mut context.0) != 0 {
                unwind_into(context.0, low, high, &mut frames);
            }
        }
        ResumeThread(thread);
    }
    (frames, (info.priority, info.base_priority))
}

fn start_of(thread: HANDLE) -> usize {
    let mut start = 0usize;
    // SAFETY: writes one pointer-sized start address.
    unsafe {
        NtQueryInformationThread(
            thread,
            ThreadQuerySetWin32StartAddress,
            (&raw mut start).cast(),
            size_of::<usize>() as u32,
            std::ptr::null_mut(),
        )
    };
    start
}

fn name_of(thread: HANDLE) -> String {
    let mut name = std::ptr::null_mut();
    // SAFETY: the system allocates the string; we free it with LocalFree.
    unsafe {
        if GetThreadDescription(thread, &mut name) < 0 || name.is_null() {
            return String::new();
        }
        let len = (0..).take_while(|&i| *name.add(i) != 0).count();
        let text = String::from_utf16_lossy(std::slice::from_raw_parts(name, len));
        LocalFree(name.cast::<c_void>());
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn another_threads_stack_unwinds_to_its_start() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            stop_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: a thread of this process.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        let (frames, _) = frames_of(thread);
        // Waiting in the kernel, several frames down from the thread start.
        assert!(frames.len() >= 3, "{frames:?}");
        assert!(start_of(thread) != 0);
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
        dump_stacks();
        stop_tx.send(()).unwrap();
        parked.join().unwrap();
    }

    #[test]
    fn frames_of_handles_exited_thread_safely() {
        let (tx, rx) = std::sync::mpsc::channel();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let short = std::thread::spawn(move || {
            // SAFETY: plain query.
            tx.send(unsafe { GetCurrentThreadId() }).unwrap();
            go_rx.recv().ok();
        });
        let id = rx.recv().unwrap();
        // SAFETY: opens a thread of this process; keep the handle past exit.
        let thread = unsafe {
            OpenThread(
                THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                0,
                id,
            )
        };
        assert!(!thread.is_null());
        go_tx.send(()).unwrap();
        short.join().unwrap();
        // The thread has exited and its TEB is unmapped; `frames_of` must not fault.
        let (frames, _) = frames_of(thread);
        assert!(frames.is_empty());
        // SAFETY: opened above.
        unsafe { CloseHandle(thread) };
    }
}
