//! Host memory for the textures the title fills, and for their staging
//! buffers, cut from pages of 64 MiB (`VICTORY_TEXTURE_POOL=1`; off by
//! default: every host copy of such a texture and every staging buffer is
//! an allocation of its own, as before; `=texture` or `=staging`: only that
//! half).
//!
//! Each such texture is a committed resource of its own and its staging
//! buffer another; with the switch on, both are pieces of pages that
//! `crate::residency::pages` owns: the card's memory for the textures
//! (placed in ripe pages only, pages made ahead, a texture larger than a
//! page in a heap of its own), the PC's upload memory for the staging
//! buffers. What a piece given back waits for before it is used again is
//! said there.

use crate::residency::pages;

/// The most empty pages of staging buffers kept while the card has room,
/// which are the PC's memory (cool_fox measured 5.2 GB of them held in one
/// run before this cap).
const STAGING_KEPT: usize = 8;
/// The same when the card has little room ([`pages::pressed`]).
const STAGING_PRESSED: usize = 2;

fn now() -> u64 {
    crate::host::now_ms()
}

/// `VICTORY_TEXTURE_POOL`: which halves are on (textures, staging).
fn halves() -> (bool, bool) {
    static ON: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("VICTORY_TEXTURE_POOL").as_deref() {
            Ok("1") => (true, true),
            Ok("texture") => (true, false),
            Ok("staging") => (false, true),
            _ => (false, false),
        };
        if on != (false, false) {
            tracing::info!(
                textures = on.0,
                staging = on.1,
                ripe_pages = pages::AHEAD,
                ripe_pages_pressed = pages::AHEAD_PRESSED,
                "textures the title fills and their staging buffers are cut from 64 MiB pages (VICTORY_TEXTURE_POOL)"
            );
        }
        on
    })
}

/// Whether host copies of textures the title fills are cut from pages.
pub(crate) fn textures_on() -> bool {
    halves().0
}

/// Whether texture staging buffers are cut from pages.
pub(crate) fn staging_on() -> bool {
    halves().1
}

/// Sets how many empty pages stay, by the card's memory (budget and use,
/// [`pages::pressed`]), and lets go of those past it. Asked once a second
/// (the sweep); does nothing, and asks the card nothing, with the pool off.
pub(crate) fn room() {
    let (textures, staging) = halves();
    if !textures && !staging {
        return;
    }
    let pressed = pages::pressed(crate::host::host().and_then(|h| h.video_memory()));
    // A set of pages no piece is cut from for a while lets its empty pages
    // go here; a busy one at its next piece.
    if textures {
        pages::reclaim(pressed);
    }
    d3d::reclaim(staging_kept(pressed));
}

fn staging_kept(pressed: bool) -> usize {
    if pressed {
        STAGING_PRESSED
    } else {
        STAGING_KEPT
    }
}

/// Bytes of the texture pages no texture holds: memory of the card in use
/// by us that new textures take first (0 with the pool off).
pub(crate) fn texture_room() -> u64 {
    if !textures_on() {
        return 0;
    }
    pages::room_left()
}

/// Whether `texture` is a placed one the card writes for the first time,
/// which it no longer is after this (false with the texture half off,
/// without taking a lock). The caller records the aliasing barrier and
/// gives every texel a value before anything else uses it.
pub(crate) fn first_write(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> bool {
    textures_on() && pages::first_write(texture)
}

/// Whether `texture` was cut from a page (false with the texture half off,
/// without asking the texture).
pub(crate) fn in_block(texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource) -> bool {
    textures_on() && pages::in_page(texture)
}

/// A texture of `desc` (no target, depth or unordered access) in a page, in
/// `state`; `None`: not pooled (the texture half off, no ripe page with
/// room, or the host refused).
///
/// # Safety
/// `desc` is a valid description.
pub(crate) unsafe fn texture(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    desc: *const windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    state: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES,
) -> Option<windows::Win32::Graphics::Direct3D12::ID3D12Resource> {
    if !textures_on() {
        return None;
    }
    // SAFETY: the caller's description.
    unsafe { pages::place(device, desc, state) }
}

#[cfg(test)]
pub(crate) use d3d::staging_piece;
pub(crate) use d3d::{StagingPiece, staging};

/// The staging pages' D3D12 side.
mod d3d {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use windows::Win32::Graphics::Direct3D12::*;

    use super::{STAGING_PRESSED, now};
    use crate::residency::pages::{Pages, Piece, Ripe, STAGING};

    /// A mapped upload buffer's memory.
    #[derive(Clone)]
    pub(super) struct Upload(ID3D12Resource, *mut u8);
    // SAFETY: the mapping is written by the owners of its pieces, each its
    // own.
    unsafe impl Send for Upload {}

    static PAGES: Mutex<Pages<Upload>> = Mutex::new(Pages::new(STAGING, STAGING_PRESSED));

    /// The count of empty pages [`reclaim`] last set, which [`stage`] gives
    /// [`PAGES`] each time it takes a piece: a look that found the pages
    /// busy loses nothing.
    static KEPT: AtomicUsize = AtomicUsize::new(STAGING_PRESSED);

    fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// [`super::room`]'s look at the staging pages: sets how many empty
    /// ones are kept, and lets go of those past it unless the pages are
    /// busy (then the next piece does).
    pub(super) fn reclaim(kept: usize) {
        KEPT.store(kept, Ordering::Relaxed);
        if let Ok(mut pages) = PAGES.try_lock() {
            pages.set_spare(kept);
            pages.reclaim(now(), true);
        }
    }

    /// Part of a page of mapped upload memory, back in the pool when
    /// dropped.
    pub(crate) struct StagingPiece(Piece);

    impl Drop for StagingPiece {
        fn drop(&mut self) {
            lock(&PAGES).give(self.0, now());
        }
    }

    /// `bytes` of upload memory, all zeros: the piece, the buffer it lies
    /// in, where in that buffer, and its mapping. `None`: not pooled (the
    /// staging half off, too big, or no page to be had).
    pub(crate) fn staging(
        host: &crate::host::Host,
        bytes: u64,
    ) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
        if !super::staging_on() {
            return None;
        }
        stage(host, bytes)
    }

    /// [`staging`] whatever the switch says.
    fn stage(
        host: &crate::host::Host,
        bytes: u64,
    ) -> Option<(StagingPiece, ID3D12Resource, u64, *mut u8)> {
        // Copies from a buffer begin at multiples of 512.
        let bytes = bytes.next_multiple_of(512);
        let mut pages = lock(&PAGES);
        pages.set_spare(KEPT.load(Ordering::Relaxed));
        let taken = pages.take(bytes, 512, now(), Ripe::ALL, |size| {
            let (buffer, data) = host.upload_buffer(size)?;
            crate::host::label(&buffer, "texture staging page", size);
            Some(Upload(buffer, data))
        });
        drop(pages);
        let (piece, memory, again) = taken?;
        let owned = StagingPiece(piece);
        // SAFETY: the piece lies in the page's mapping.
        let data = unsafe { memory.1.add(piece.start as usize) };
        if again {
            // What is not detiled into it (an unmapped texture) uploads
            // zeros, as from a buffer of its own.
            // SAFETY: the piece is ours alone.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        }
        Some((owned, memory.0, piece.start, data))
    }

    /// Tests: whether the staging pages hold `buffer` as one of theirs, and
    /// whether the piece at `start` in it has been given back (resting or
    /// free again: no longer any holder's).
    #[cfg(test)]
    pub(crate) fn staging_piece(buffer: &ID3D12Resource, start: u64) -> (bool, bool) {
        let pages = lock(&PAGES);
        match pages.position(|m| m.0 == *buffer) {
            Some(i) => (true, pages.given(i, start)),
            None => (false, false),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A staging piece begins as zeros, also when it was used before
        /// (asked of the pages whatever `VICTORY_TEXTURE_POOL` says).
        #[test]
        fn warp_a_staging_piece_begins_as_zeros() {
            let Some(host) = crate::host::host() else {
                return;
            };
            // SAFETY: the piece's mapping, 4096 bytes of it ours.
            unsafe {
                let (piece, buffer, at, data) = stage(host, 4096).unwrap();
                assert!(at.is_multiple_of(512));
                assert_eq!(staging_piece(&buffer, at), (true, false));
                std::ptr::write_bytes(data, 0xAB, 4096);
                drop(piece);
                assert_eq!(staging_piece(&buffer, at), (true, true));
                lock(&PAGES).rested(u64::MAX);
                let (_piece, _, _, data) = stage(host, 4096).unwrap();
                assert!((0..4096).all(|i| *data.add(i) == 0));
                assert!(host.device.GetDeviceRemovedReason().is_ok());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_staging_pages_stay_while_the_card_has_room() {
        assert_eq!(staging_kept(false), STAGING_KEPT);
        assert_eq!(staging_kept(true), STAGING_PRESSED);
    }
}
