//! The title's upscaler, stood in for (`VICTORY_UPSCALE_STANDIN=1`).
//!
//! In Series S mode the title renders at a low resolution and has
//! AMD's FidelityFX upscaler make the full-size picture. Its passes are
//! console-compiled compute programs the translator does not read: none of
//! them runs here, the output stays as it was made and the picture is
//! black.
//!
//! With the switch on, the title's lookup of the upscaler's `ffxDispatch`
//! gets [`dispatch`] instead ([`hook`]). An upscale dispatch is then one
//! pass of ours that draws the colour input over the output with linear
//! filtering (`crate::list::upscale`); the upscaler's own passes are not
//! recorded at all. Every other dispatch, and an upscale we cannot make
//! sense of, goes to the upscaler as before. The picture is softer than the
//! upscaler's would be.
//!
//! Off by default: unset, the lookup gets the upscaler's own function.

use std::ffi::c_void;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// The upscaler's own `ffxDispatch`.
static ORIGINAL: AtomicUsize = AtomicUsize::new(0);

/// Dispatches seen; upscales drawn by us; upscales handed on.
static CALLS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// `ffxDispatch(context, description)`.
type Dispatch = extern "system" fn(*mut c_void, *const u8) -> u32;

/// The description type of an upscale dispatch (`ffx_upscale.h`).
const UPSCALE: u64 = 0x0001_0001;

/// An upscale description (`ffxDispatchDescUpscale`): header 16 bytes, the
/// list, seven resources of 48 bytes (the pointer first), then the jitter
/// and motion scale, the render size and the upscale size.
const LIST: usize = 16;
const COLOUR: usize = 24;
const OUTPUT: usize = 24 + 6 * 48;
const RENDER: usize = OUTPUT + 48 + 16;
/// The bytes of a description read here; an upscale description is longer.
const LEN: usize = RENDER + 16;

/// `VICTORY_UPSCALE_STANDIN=1`: the title's upscaler is stood in for.
pub fn standin() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPSCALE_STANDIN").is_some_and(|v| v == "1"))
}

/// What the title's lookup of the upscaler's `ffxDispatch` gets: with the
/// switch on ([`standin`]) ours, which keeps `original` for what it hands
/// on; otherwise `original`.
pub fn hook(original: usize) -> usize {
    if !standin() {
        return original;
    }
    ORIGINAL.store(original, Relaxed);
    tracing::info!(
        original = format_args!("{original:#x}"),
        "the title's upscaler: its dispatch goes through ours (VICTORY_UPSCALE_STANDIN=1)"
    );
    dispatch as *const () as usize
}

fn hand_on(context: *mut c_void, description: *const u8) -> u32 {
    match ORIGINAL.load(Relaxed) {
        // (an error code of the interface: no backend)
        0 => 1,
        f => {
            // SAFETY: the address the upscaler exports its dispatch at.
            let f = unsafe { std::mem::transmute::<usize, Dispatch>(f) };
            f(context, description)
        }
    }
}

/// What an upscale dispatch names: the list, the colour input and the
/// output (title objects), the render size and the upscale size.
#[derive(Debug, PartialEq)]
struct Upscale {
    list: usize,
    colour: usize,
    output: usize,
    render: [u32; 2],
    size: [u32; 2],
}

/// The upscale in the first [`LEN`] bytes of an upscale description, if
/// it names a list, both textures and sizes from 1 to 16384.
fn read(bytes: &[u8; LEN]) -> Option<Upscale> {
    let word =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let pointer = |at: usize| (u64::from(word(at)) | u64::from(word(at + 4)) << 32) as usize;
    let upscale = Upscale {
        list: pointer(LIST),
        colour: pointer(COLOUR),
        output: pointer(OUTPUT),
        render: [word(RENDER), word(RENDER + 4)],
        size: [word(RENDER + 8), word(RENDER + 12)],
    };
    let sane = |d: [u32; 2]| (1..=16384).contains(&d[0]) && (1..=16384).contains(&d[1]);
    (upscale.list != 0
        && upscale.colour != 0
        && upscale.output != 0
        && sane(upscale.render)
        && sane(upscale.size))
    .then_some(upscale)
}

extern "system" fn dispatch(context: *mut c_void, description: *const u8) -> u32 {
    if description.is_null() {
        return hand_on(context, description);
    }
    // SAFETY: the title passes a description, which begins with its type.
    let kind = unsafe { description.cast::<u64>().read_unaligned() };
    let call = CALLS[0].fetch_add(1, Relaxed);
    if kind != UPSCALE {
        if call < 8 {
            tracing::info!(
                kind = format_args!("{kind:#x}"),
                "the title's upscaler: a dispatch that is no upscale, handed on"
            );
        }
        return hand_on(context, description);
    }
    // SAFETY: an upscale description is longer than this.
    let bytes = unsafe { &*description.cast::<[u8; LEN]>() };
    let upscale = read(bytes);
    if call < 2 {
        tracing::info!(?upscale, "the title's upscaler: an upscale dispatch");
    }
    let drawn = upscale.is_some_and(|u| {
        use crate::object::This;
        crate::list::upscale(
            u.list as This,
            u.colour as This,
            u.output as This,
            u.render,
            u.size,
        )
    });
    if drawn {
        CALLS[1].fetch_add(1, Relaxed);
        // (the interface's "ok")
        return 0;
    }
    if CALLS[2].fetch_add(1, Relaxed) < 4 {
        tracing::warn!("the title's upscaler: an upscale we could not draw ourselves, handed on");
    }
    hand_on(context, description)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn description(
        list: u64,
        colour: u64,
        output: u64,
        render: [u32; 2],
        size: [u32; 2],
    ) -> [u8; LEN] {
        let mut bytes = [0u8; LEN];
        bytes[..8].copy_from_slice(&UPSCALE.to_le_bytes());
        bytes[LIST..LIST + 8].copy_from_slice(&list.to_le_bytes());
        bytes[COLOUR..COLOUR + 8].copy_from_slice(&colour.to_le_bytes());
        bytes[OUTPUT..OUTPUT + 8].copy_from_slice(&output.to_le_bytes());
        for (i, w) in [render[0], render[1], size[0], size[1]]
            .into_iter()
            .enumerate()
        {
            bytes[RENDER + 4 * i..RENDER + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn an_upscale_description_is_read_at_its_offsets() {
        // The layout of `ffxDispatchDescUpscale`: the output is the seventh
        // 48-byte resource, the sizes follow two more 8-byte pairs.
        assert_eq!((COLOUR, OUTPUT, RENDER, LEN), (24, 312, 376, 392));
        let bytes = description(
            0x1234_5678_9abc_def0,
            0x10_0000,
            0x20_0000,
            [1280, 720],
            [1920, 1080],
        );
        assert_eq!(
            read(&bytes),
            Some(Upscale {
                list: 0x1234_5678_9abc_def0,
                colour: 0x10_0000,
                output: 0x20_0000,
                render: [1280, 720],
                size: [1920, 1080],
            })
        );
    }

    #[test]
    fn an_upscale_without_its_objects_or_with_odd_sizes_is_handed_on() {
        let ok = |l, c, o, r, s| read(&description(l, c, o, r, s)).is_some();
        assert!(ok(1, 2, 3, [1, 1], [16384, 16384]));
        assert!(!ok(0, 2, 3, [1280, 720], [1920, 1080]));
        assert!(!ok(1, 0, 3, [1280, 720], [1920, 1080]));
        assert!(!ok(1, 2, 0, [1280, 720], [1920, 1080]));
        assert!(!ok(1, 2, 3, [0, 720], [1920, 1080]));
        assert!(!ok(1, 2, 3, [1280, 720], [1920, 16385]));
    }

    #[test]
    fn unset_the_lookup_gets_the_upscalers_own_dispatch() {
        if standin() {
            return; // the switch is set for this test run
        }
        assert_eq!(hook(0x1234_5000), 0x1234_5000);
        assert_eq!(ORIGINAL.load(Relaxed), 0);
    }
}
