//! Where the host copies of the textures the title fills live on the card.
//!
//! [`pages`] owns the memory: pages of 64 MiB made by us, each texture
//! placed in a piece of one. Never a tiled or reserved resource (tile
//! mapping is a known device-removal trigger on the RDNA4 card).

pub(crate) mod pages;
