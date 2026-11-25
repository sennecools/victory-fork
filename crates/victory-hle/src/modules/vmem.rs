//! The title's `VirtualFree` and `VirtualProtect`, with the memory core on
//! (`VICTORY_MEMORY_CORE=1`, `victory_gpu`'s `home` module). The card may
//! use title memory in place, through a heap opened on the title's pages;
//! while such a heap stands the host refuses to decommit, release or
//! protect those pages (error 87). These calls go to the GPU side, which
//! lets its homes there go before it asks the host. With the switch off the
//! module is not registered at all and the title's calls bind to the host,
//! as before ([`super::all_with`]).
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::Memory as host;
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentProcessId, GetProcessId};
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

pub extern "system" fn VirtualFree(at: *mut c_void, len: usize, kind: u32) -> BOOL {
    victory_gpu::title_virtual_free(at as usize, len, kind)
}

pub extern "system" fn VirtualProtect(
    at: *const c_void,
    len: usize,
    new: u32,
    old: *mut u32,
) -> BOOL {
    victory_gpu::title_virtual_protect(at as usize, len, new, old)
}

/// Whether `process` is this one (the pseudo handle, or a handle of it).
fn own(process: HANDLE) -> bool {
    // SAFETY: plain queries; a bad handle gives 0.
    unsafe { process == GetCurrentProcess() || GetProcessId(process) == GetCurrentProcessId() }
}

pub extern "system" fn VirtualFreeEx(
    process: HANDLE,
    at: *mut c_void,
    len: usize,
    kind: u32,
) -> BOOL {
    if own(process) {
        return VirtualFree(at, len, kind);
    }
    // SAFETY: the title's own arguments, for the host function it called.
    unsafe { host::VirtualFreeEx(process, at, len, kind) }
}

pub extern "system" fn VirtualProtectEx(
    process: HANDLE,
    at: *const c_void,
    len: usize,
    new: u32,
    old: *mut u32,
) -> BOOL {
    if own(process) {
        return VirtualProtect(at, len, new, old);
    }
    // SAFETY: the title's own arguments, for the host function it called.
    unsafe { host::VirtualProtectEx(process, at, len, new, old) }
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "memory core",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-memory-*"],
        exports: vec![
            hle_export!(VirtualFree),
            hle_export!(VirtualFreeEx),
            hle_export!(VirtualProtect),
            hle_export!(VirtualProtectEx),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without the switch the title's calls bind to the host: no module
    /// answers for them, and the modules are today's exactly.
    #[test]
    fn off_registers_no_hooks() {
        let off = super::super::all_with(false);
        let on = super::super::all_with(true);
        let names = |all: &[HleModule]| all.iter().map(|m| m.name).collect::<Vec<_>>();
        assert!(!names(&off).contains(&"memory core"));
        assert_eq!(names(&on).len(), names(&off).len() + 1);
        for dll in [
            "kernel32.dll",
            "KernelBase.dll",
            "api-ms-win-core-memory-l1-1-0.dll",
        ] {
            for name in [
                "VirtualFree",
                "VirtualFreeEx",
                "VirtualProtect",
                "VirtualProtectEx",
            ] {
                let found = |all: &[HleModule]| {
                    all.iter()
                        .filter(|m| m.serves(dll))
                        .any(|m| m.exports.iter().any(|e| e.name == name))
                };
                assert!(
                    !found(&off),
                    "{dll}!{name} bound to the core with the switch off"
                );
                assert!(found(&on), "{dll}!{name} not bound to the core with it on");
            }
        }
        // The other modules are the same with it on.
        let rest: Vec<_> = names(&on)
            .into_iter()
            .filter(|n| *n != "memory core")
            .collect();
        assert_eq!(rest, names(&off));
    }

    /// With the core on, a free of memory that is no region of the title's
    /// goes to the host as it is, for the title's own process handle too.
    #[test]
    fn calls_on_other_memory_pass_through() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc,
        };
        // SAFETY: an allocation of the test's own.
        unsafe {
            let at = VirtualAlloc(
                std::ptr::null(),
                1 << 16,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            );
            assert!(!at.is_null());
            let mut old = 0;
            assert_eq!(
                VirtualProtectEx(GetCurrentProcess(), at, 4096, PAGE_READONLY, &mut old),
                1
            );
            assert_eq!(old, PAGE_READWRITE);
            assert_eq!(VirtualFree(at, 4096, MEM_DECOMMIT), 1);
            assert_eq!(VirtualFreeEx(GetCurrentProcess(), at, 0, MEM_RELEASE), 1);
            // Already released: the host's own refusal.
            assert_eq!(VirtualFree(at, 0, MEM_RELEASE), 0);
        }
    }
}
