//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees, [`window`] the
//! size, place and closing of the window the title is shown in. Planned:
//! `present` (swap chain).

#[cfg(windows)]
mod capture;
#[cfg(windows)]
mod card_home;
#[cfg(windows)]
mod census;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
#[cfg(windows)]
mod flip;
pub mod frame;
pub mod frametime;
#[cfg(windows)]
mod home;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod pool;
#[cfg(windows)]
mod primitive;
#[cfg(windows)]
pub mod release;
pub mod profile;
pub mod query;
pub mod queue;
#[cfg(windows)]
mod residency;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod safecopy;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod tiles;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod texture_pool;
#[cfg(windows)]
mod upscale;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
pub mod window;

#[cfg(windows)]
pub use home::{
    Committing, memory_core_on, memory_own_alloc, memory_own_free_now, memory_own_on,
    title_memory_committing, title_memory_region, title_virtual_free, title_virtual_protect,
};
#[cfg(windows)]
pub use host::{card_memory_in_use, on_failure};
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use pool::{
    PoolChunk, memory_pool_chunk_bytes, memory_pool_homes_possible, memory_pool_on,
    memory_pool_open_pct, title_memory_pool_chunk, title_memory_pool_chunk_gone,
    title_memory_pool_chunk_known, title_memory_pool_chunk_mapped,
    title_memory_pool_chunk_mapped_enough,
};
#[cfg(windows)]
pub use upscale::{hook as upscaler_dispatch, standin as upscaler_standin};
#[cfg(windows)]
pub use watch::{
    Remapping, title_memory_allocated, title_memory_reading, title_memory_reading_range,
    title_memory_remapped, title_memory_remapping, title_memory_remapping_range,
    title_memory_written, writing as kernel_will_write,
};
