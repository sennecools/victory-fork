//! Buffer views of title memory: the host copy a buffer view is made on
//! ([`buffer_copy`]), and the host view there ([`buffer_srv`],
//! [`buffer_uav`]).

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{REFUSED_VIEWS, Resolved, View, srv_desc, uav_desc};

/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a buffer shader or unordered access view reads: its
/// address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The title memory the host slot of buffer view `view` is made on
/// (address, bytes): the whole buffer the view is of, which is what
/// [`make`](super::make) finds the host copy by. [`buffer_range`] is the
/// part of it the view reads.
pub(crate) fn buffer_extent(view: &View) -> Option<(u64, u64)> {
    match view {
        View::Shader(r, Some(_)) | View::Unordered(r, _, Some(_)) => {
            let (data, len) = crate::resource::buffer_bytes(*r as crate::object::This)?;
            Some((data as u64, len))
        }
        View::Placed(_, resource, desc) if resource[0] == 1 => Some((
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            u64::from(resource[4]) | u64::from(resource[5]) << 32,
        )),
        _ => None,
    }
}

/// Whether a buffer view is brought up to date in the host copy its slot
/// shows ([`buffer_copy`]). On by default; `VICTORY_VIEW_EXTENT=0` looks the
/// copy up by the view's range alone, as before.
fn view_extent() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_EXTENT").is_none_or(|v| v != "0"))
}

/// The host copy buffer view `view` reads, and the title range (address,
/// bytes) it reads there; `None` for other views and empty ranges.
///
/// The copy is the one the view's slot shows: [`make`](super::make) finds
/// it by the whole buffer the view is of ([`buffer_extent`]). Looked up by
/// the range the view reads, as it was, the copy is another one wherever
/// the two lookups differ: for memory outside buffer objects, a view of a part of a
/// buffer that crosses a window line (the buffer's window is two cells, the
/// part's one), and for buffer objects, a part that also lies in a smaller
/// buffer object. The slot then showed a copy nothing brought up to date,
/// and what shaders wrote through it was looked for in the other copy.
pub(crate) fn buffer_copy(
    view: &View,
) -> Option<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> {
    let (address, len) = buffer_range(view)?;
    if len == 0 {
        return None;
    }
    // A view off its buffer's elements is made on a copy of its own bytes
    // (`own_bytes`): the same question `buffer_srv` and `buffer_uav` ask,
    // of the copy they find, so that the copy brought up to date is the
    // one the slot shows.
    if let Some((base, width)) = buffer_extent(view)
        && let Some((first, count, element, stride, unknown)) = buffer_fields(view)
        && let Some(whole) = crate::shadow::find(base, width)
        && let Some((start, bytes)) = own_bytes(
            whole.offset(base),
            base,
            first,
            count,
            element,
            stride,
            unknown,
        )
        && let Some(own) = crate::shadow::exact(start, bytes)
    {
        return Some((own, start, bytes));
    }
    if view_extent()
        && let Some((base, width)) = buffer_extent(view)
        // (The console is lax: a view may begin past its buffer's end.)
        && address >= base
        && address - base < width.max(1)
        && let Some(shadow) = crate::shadow::find(base, width)
    {
        return Some((shadow, address, len));
    }
    crate::shadow::find(address, len).map(|shadow| (shadow, address, len))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// Whether a buffer view off its buffer's elements gets a copy of its own
/// bytes (`VICTORY_VIEW_OWN_COPY=0`: it is refused and reads zeros, as
/// before).
fn own_copy_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_OWN_COPY").is_none_or(|v| v != "0"))
}

/// The title memory (start, bytes) of a buffer view that cannot be made on
/// the host copy of its buffer: `count` elements of `element` bytes from
/// element `first` of the buffer at `base`, which lies `offset` bytes into
/// that copy, where `offset` is not a whole number of elements and the view
/// cannot be a raw view of the same bytes either ([`raw_window`]:
/// `unknown`, the view has no format, and `stride` is its structure's).
/// The host counts a view's first element from its buffer's start, so such
/// a view has no host view there; it is made on a copy of its own bytes
/// (`crate::shadow::exact`), where it starts at element 0. `None`: the
/// view is made on its buffer's copy, as every other.
fn own_bytes(
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
) -> Option<(u64, u64)> {
    own_bytes_if(
        own_copy_on(),
        offset,
        base,
        first,
        count,
        element,
        stride,
        unknown,
    )
}

/// Where a buffer placed in the same heap as a copy may start so that a
/// view whose first byte is `a` lies on one of its elements of `e` bytes:
/// the placed buffer's first byte `P` and the view's first element there.
/// `b` is the byte at the heap's offset 0. A placed buffer starts on a
/// 64 KiB line of the heap, and the host counts a view's elements from its
/// buffer's start, so `P = b + 64 KiB * j` for the smallest `j` with
/// `(a - P) % e == 0` and `P <= a`. `None` if there is no such `j`: when
/// the gcd `g` of 64 KiB and `e` does not divide `(a - b) % e` (a raw
/// view's 16 bytes or a constant buffer's 256 bytes then never move onto
/// the line by a placement), or when `j` would place the buffer past `a`.
///
/// Solved as `(64 KiB / g) * j = r / g (mod e / g)` with `r = (a - b) % e`:
/// `64 KiB / g` is a power of two and `e / g` has no factor of two left
/// (or `64 KiB / g` is 1), so the inverse exists and `j < e / g`.
#[cfg_attr(not(test), allow(dead_code))]
fn alias_base(a: u64, b: u64, e: u64) -> Option<(u64, u64)> {
    /// The inverse of `x` modulo `m`, for `x` and `m` coprime (0 for
    /// `m = 1`).
    fn inverse(x: u64, m: u64) -> Option<u64> {
        let (mut old, mut new) = (i128::from(x % m), i128::from(m));
        let (mut old_s, mut new_s) = (1i128, 0i128);
        while new != 0 {
            let q = old / new;
            (old, new) = (new, old - q * new);
            (old_s, new_s) = (new_s, old_s - q * new_s);
        }
        (old == 1).then(|| old_s.rem_euclid(i128::from(m)) as u64)
    }
    const LINE: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
    if e == 0 || a < b {
        return None;
    }
    let d = a - b;
    let r = d % e;
    let g = 1u64 << e.trailing_zeros().min(LINE.trailing_zeros());
    if !r.is_multiple_of(g) {
        return None;
    }
    let m = e / g;
    let j = (u128::from(r / g) * u128::from(inverse(LINE / g, m)?) % u128::from(m)) as u64;
    if j > d / LINE {
        return None;
    }
    let p = b + LINE * j;
    Some((p, (a - p) / e))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn own_bytes_if(
    on: bool,
    offset: u64,
    base: u64,
    first: u64,
    count: u32,
    element: u64,
    stride: u32,
    unknown: bool,
) -> Option<(u64, u64)> {
    if !on || element == 0 || count == 0 || offset.is_multiple_of(element) {
        return None;
    }
    if unknown && raw_window(offset, first, count, stride).is_some() {
        return None;
    }
    let start = base.checked_add(first.checked_mul(element)?)?;
    Some((start, u64::from(count) * element))
}

/// Of a buffer view: its first element, its elements, the bytes of one (a
/// raw view's: 4), its structure's stride, and whether it has no format.
/// As [`buffer_srv`] and [`buffer_uav`] read them from the same words.
fn buffer_fields(view: &View) -> Option<(u64, u32, u64, u32, bool)> {
    const BUFFER: u32 = 1;
    let (words, unordered): (&[u32; 10], bool) = match view {
        View::Shader(_, Some(d)) if d[1] == BUFFER => (d, false),
        View::Unordered(_, _, Some(d)) if d[1] == BUFFER => (d, true),
        View::Placed(unordered, resource, d) if resource[0] == BUFFER => {
            (d[..10].try_into().ok()?, *unordered)
        }
        _ => return None,
    };
    if unordered {
        let d = uav_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(d.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        Some((
            b.FirstElement,
            b.NumElements,
            element,
            b.StructureByteStride,
            d.Format == DXGI_FORMAT_UNKNOWN,
        ))
    } else {
        let d = srv_desc(words);
        // SAFETY: a buffer view's union member.
        let b = unsafe { d.Anonymous.Buffer };
        Some((
            b.FirstElement,
            b.NumElements,
            element_size(&d),
            b.StructureByteStride,
            d.Format == DXGI_FORMAT_UNKNOWN,
        ))
    }
}

/// A structured buffer view (`first`, `count` elements of `stride` bytes,
/// `offset` bytes into its host copy) as a raw view of the same bytes:
/// its first dword and its dwords. Translated shaders read untyped buffers
/// as raw bytes at `element * stride + offset` (the stride from the
/// used-resource entry, or from the table by heap slot for a buffer picked
/// by index), and a raw read through a structured view gives zeros on
/// NVIDIA (reported upstream with PR #193; structured reads through raw
/// views were right there). So the host's view is raw whenever it can be:
/// the bytes must start on 16 (the host's raw view alignment) and be whole
/// dwords. A view that cannot stays structured.
pub(super) fn raw_window(offset: u64, first: u64, count: u32, stride: u32) -> Option<(u64, u32)> {
    if stride == 0 {
        return None;
    }
    let start = first.checked_mul(u64::from(stride))?.checked_add(offset)?;
    let bytes = u64::from(count) * u64::from(stride);
    (start % 16 == 0 && bytes % 4 == 0)
        .then(|| {
            u32::try_from(bytes / 4)
                .ok()
                .map(|dwords| (start / 4, dwords))
        })
        .flatten()
}

/// Whether the host accepts a buffer view with these fields of a buffer of
/// `width` bytes. It removes the device for one it does not (view creation
/// returns nothing), and the console is laxer: the title's heap holds
/// views past their memory's end, of no elements, of typeless formats
/// without the raw flag, with strides the host has no structured buffers
/// for. Since every buffer view of the heap is made for shaders that pick
/// them by index, one such view ended the run (2026-11-24, run 10).
pub(super) fn buffer_view_ok(
    format: DXGI_FORMAT,
    first: u64,
    count: u32,
    stride: u32,
    raw: bool,
    width: u64,
) -> bool {
    use crate::resource::family;
    let f = format.0 as u32;
    let element = if raw {
        // raw: R32 typeless, counted in dwords, starting on 16 bytes
        if format != DXGI_FORMAT_R32_TYPELESS || stride != 0 || !first.is_multiple_of(4) {
            return false;
        }
        4
    } else if stride > 0 {
        // structured: no format, a stride the host takes
        if format != DXGI_FORMAT_UNKNOWN || stride > 2048 || !stride.is_multiple_of(4) {
            return false;
        }
        u64::from(stride)
    } else {
        // typed: a format with a size that is not typeless
        let typeless = family(f) == f && family(f + 1) == f;
        match crate::resource::element(f) {
            Some((bytes, 1)) if f != 0 && !typeless && bytes > 0 => bytes,
            _ => return false,
        }
    };
    count > 0
        && first
            .checked_add(u64::from(count))
            .and_then(|end| end.checked_mul(element))
            .is_some_and(|end| end <= width)
}

/// How many of `count` elements of `element` bytes from `first` on lie in
/// `width` bytes: the console lets a view reach past its buffer's end, the
/// host refuses one. A view with no element inside keeps its count (and is
/// refused).
pub(super) fn elements_inside(first: u64, count: u32, element: u64, width: u64) -> u32 {
    let room = (width / element.max(1)).saturating_sub(first);
    match u32::try_from(room) {
        Ok(0) | Err(_) => count,
        Ok(room) => count.min(room),
    }
}

/// What a shader resource view of title buffer memory at `address` (`len`
/// bytes) is made from on its host copy, made now if it has none
/// ([`resolve`](super::resolve)).
pub(super) fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
) -> Option<Resolved> {
    let shadow = crate::shadow::find(address, len)?;
    let offset = shadow.offset(address);
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, offset) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(
            offset,
            address,
            b.FirstElement,
            b.NumElements,
            element_size(&desc),
            b.StructureByteStride,
            unknown,
        )
        .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, 0)
            }
            None => (shadow, offset),
        }
    };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    // SAFETY: a buffer view's union member.
    let structured = unsafe { desc.Anonymous.Buffer };
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| {
            raw_window(
                offset,
                structured.FirstElement,
                structured.NumElements,
                structured.StructureByteStride,
            )
        })
        .flatten();
    if let Some((first, count)) = window {
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: first,
            NumElements: count,
            StructureByteStride: 0,
            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
        };
    }
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    if window.is_none() {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let unit = if window.is_some() { 4 } else { element };
    buffer.NumElements = elements_inside(buffer.FirstElement, buffer.NumElements, unit, width);
    let (first, count, stride) = (
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    );
    let raw = buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0;
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "buffer view the host would refuse left null"
        );
        return None;
    }
    // The console has no component mapping; one the host does not know is the default's.
    if desc.Shader4ComponentMapping != D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING {
        desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    }
    Some(Resolved::Srv {
        resource: shadow.buffer.clone(),
        desc: Some(desc),
        shadow: Some((shadow, address)),
    })
}

/// What an unordered access view of title buffer memory at `address`
/// (`len` bytes) is made from on its host copy, as [`buffer_srv`]; `None`
/// if the copy cannot be written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
pub(super) fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
) -> Option<Resolved> {
    let shadow = crate::shadow::find(address, len)?;
    // Off the elements of that copy: on a copy of its own bytes, from its
    // element 0 (`own_bytes`).
    let (shadow, own) = {
        // SAFETY: a buffer view's union member.
        let b = unsafe { desc.Anonymous.Buffer };
        let element = if b.StructureByteStride > 0 {
            u64::from(b.StructureByteStride)
        } else if b.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
            4
        } else {
            crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
        };
        let unknown = desc.Format == DXGI_FORMAT_UNKNOWN;
        match own_bytes(
            shadow.offset(address),
            address,
            b.FirstElement,
            b.NumElements,
            element,
            b.StructureByteStride,
            unknown,
        )
        .and_then(|(start, bytes)| crate::shadow::exact(start, bytes))
        {
            Some(own) => {
                desc.Anonymous.Buffer.FirstElement = 0;
                (own, true)
            }
            None => (shadow, false),
        }
    };
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = if own { 0 } else { shadow.offset(address) };
    // A structured view as a raw view of the same bytes (see `raw_window`).
    let window = (desc.Format == DXGI_FORMAT_UNKNOWN)
        .then(|| {
            raw_window(
                offset,
                buffer.FirstElement,
                buffer.NumElements,
                buffer.StructureByteStride,
            )
        })
        .flatten();
    if let Some((first, count)) = window {
        buffer.FirstElement = first;
        buffer.NumElements = count;
        buffer.StructureByteStride = 0;
        buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    } else {
        if element == 0 || offset % element != 0 {
            once!(
                offset,
                element,
                "unordered buffer view not on an element of its host copy"
            );
            return None;
        }
        buffer.FirstElement += offset / element;
    }
    buffer.CounterOffsetInBytes = 0;
    // SAFETY: a live buffer.
    let width = unsafe { shadow.buffer.GetDesc() }.Width;
    let unit = if window.is_some() { 4 } else { element };
    buffer.NumElements = elements_inside(buffer.FirstElement, buffer.NumElements, unit, width);
    let (first, count, stride) = (
        buffer.FirstElement,
        buffer.NumElements,
        buffer.StructureByteStride,
    );
    let raw = buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0;
    if window.is_some() {
        // (Left out at first: every structured unordered access view was then
        // refused by `buffer_view_ok` and left null, runs 15 and 16.)
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
    }
    if !buffer_view_ok(desc.Format, first, count, stride, raw, width) {
        REFUSED_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        once!(
            format = desc.Format.0,
            first,
            count,
            stride,
            raw,
            width,
            "unordered buffer view the host would refuse left null"
        );
        return None;
    }
    Some(Resolved::Uav {
        resource: shadow.buffer.clone(),
        desc: Some(desc),
        shadow: Some((shadow, address)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINE: u64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;

    /// The smallest `j` that [`alias_base`] solves for, found by trying
    /// each: the first 64 KiB line from `b` at or before `a` that lies a
    /// whole number of `e`-byte elements before `a`.
    fn tried(a: u64, b: u64, e: u64) -> Option<(u64, u64)> {
        let d = a.checked_sub(b)?;
        (0..=e.min(d / LINE))
            .find(|j| (d - LINE * j).is_multiple_of(e))
            .map(|j| (b + LINE * j, (d - LINE * j) / e))
    }

    /// What every placement must be: on a 64 KiB line from `b`, between `b`
    /// and `a`, a whole number of elements before `a`.
    fn placed_right(a: u64, b: u64, e: u64, (p, first): (u64, u64)) {
        assert!(
            (p - b).is_multiple_of(LINE),
            "{a:#x} {b:#x} {e}: off the line"
        );
        assert!(b <= p && p <= a, "{a:#x} {b:#x} {e}: outside");
        assert_eq!(p + first * e, a, "{a:#x} {b:#x} {e}: not on an element");
    }

    /// The alias math by rows: structured strides, a raw view's 16-byte
    /// line, a constant buffer's 256 bytes and the room before the view.
    #[test]
    fn alias_base_table() {
        let raw = u64::from(D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT);
        let constant = u64::from(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);
        // (view offset from b, element bytes, j, first element there)
        let rows: [(u64, u64, Option<(u64, u64)>); 16] = [
            // structured strides with a factor that is not two: a line on
            (2 * LINE + 0x3008, 12, Some((1, 6486))),
            (5 * LINE + 0x1238, 20, Some((4, 3510))),
            (5 * LINE + 0x123c, 20, Some((3, 6787))),
            (7 * LINE + 0x2008, 24, Some((2, 13995))),
            (40 * LINE + 0x100c, 36, Some((2, 69291))),
            (600 * LINE + 0x2a8, 2044, Some((254, 11094))),
            // stride 2044 = 4 * 511: up to 510 lines, about 33 MiB of room
            (510 * LINE, 2044, Some((510, 0))),
            (510 * LINE - 2044, 2044, None),
            (10 * LINE + 0x2a8, 2044, None),
            // on an element already: the copy's own start
            (20 * LINE + 40, 12, Some((0, (20 * LINE + 40) / 12))),
            // off the power of two of the element: no line helps
            (LINE + 0x38, 16, None),
            (LINE + 0x22, 12, None),
            // a raw view's 16 bytes: on 16 it needs no placement, off 16 none helps
            (LINE + 0x30, raw, Some((0, (LINE + 0x30) / raw))),
            (LINE + 0x34, raw, None),
            // a constant buffer's 256 bytes: the same
            (
                3 * LINE + 0x700,
                constant,
                Some((0, (3 * LINE + 0x700) / constant)),
            ),
            (3 * LINE + 0x740, constant, None),
        ];
        // b on a 64 KiB line (a home's line) and off one (a card side's start)
        for b in [0xd_9000_0000u64, 0x2_3456_7894] {
            for (d, e, want) in rows {
                let got = alias_base(b + d, b, e);
                assert_eq!(
                    got,
                    want.map(|(j, first)| (b + j * LINE, first)),
                    "view at b + {d:#x}, {e}-byte elements"
                );
                assert_eq!(got, tried(b + d, b, e), "b + {d:#x}, {e}");
                if let Some(got) = got {
                    placed_right(b + d, b, e, got);
                }
            }
        }
        // no elements, or a view before its heap: none
        assert_eq!(alias_base(0x1_0000, 0x1_0000, 0), None);
        assert_eq!(alias_base(0xffff, 0x1_0000, 12), None);
    }

    /// The alias math against trying every line: for each element size the
    /// host takes for strides (whole dwords up to 2048), a few odd and
    /// large sizes, and a raw and a constant buffer view's line, at 64
    /// pseudo-random offsets of up to 600 lines (about 37 MiB) past `b`.
    #[test]
    fn alias_base_is_the_first_line_that_fits() {
        let mut elements: Vec<u64> = (1..=512).map(|n| 4 * n).collect();
        elements.extend([1, 2, 3, 6, 256, 3 << 17]);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for e in elements {
            for _ in 0..64 {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let d = (seed >> 24) % (600 * LINE);
                let (a, b) = (0x8_0000_0000 + d, 0x8_0000_0000);
                let got = alias_base(a, b, e);
                assert_eq!(got, tried(a, b, e), "b + {d:#x}, {e}-byte elements");
                if let Some(got) = got {
                    placed_right(a, b, e, got);
                }
            }
        }
    }
}
