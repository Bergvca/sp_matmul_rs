//! Sparse CSR × CSR matrix multiplication with fused top-n selection.
//!
//! Rust port of `sparse_dot_topn`.
//!
//! The default driver column-chunks B so the dense `sums` accumulator and the streamed
//! B fragments stay resident in L1/L2 cache — this is the dominant performance lever
//! for very large, very sparse inputs and is built in, not bolted on.

#![deny(rust_2018_idioms)]
#![warn(missing_debug_implementations)]

pub mod csr;
pub mod index;
pub mod maxheap;
pub mod matmul;
pub mod matmul_topn;
pub mod chunked;
pub mod scalar;
pub mod tiled;
pub mod zip;

#[cfg(feature = "rayon")]
pub mod parallel;

#[cfg(feature = "python")]
pub mod python;

pub use crate::chunked::{default_chunk_cols, sp_matmul_topn_chunked, AccumMode, BProjection};
pub use crate::tiled::TiledB;
pub use crate::csr::{CsrMatrix, CsrView};
pub use crate::index::Index;
pub use crate::matmul::sp_matmul;
pub use crate::matmul_topn::{sp_matmul_topn, SortMode, TopNOptions};
pub use crate::scalar::Scalar;
pub use crate::zip::zip_sp_matmul_topn;

#[doc(hidden)]
pub use crate::matmul_topn::sp_matmul_topn_unchunked_for_tests;
