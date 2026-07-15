//! `Index` trait — the integer type of the `indptr` / `indices` arrays.
//!
//! Implemented for `i32` and `i64` to match the C++ extension.

use num_traits::PrimInt;
use std::fmt::Debug;

pub trait Index: PrimInt + Debug + Send + Sync + 'static {
    /// Narrow a `usize` to the index type. This is an unchecked, truncating
    /// cast on the hot path; callers must ensure the value fits (see
    /// [`Index::max_usize`] and `crate::csr::check_index_capacity`).
    fn from_usize(value: usize) -> Self;
    fn to_usize(self) -> usize;
    /// Largest `usize` value representable by this index type without
    /// overflow. Used to guard `from_usize` conversions. Provided in terms of
    /// `Bounded::max_value` so implementations cannot disagree with the
    /// type's actual range.
    fn max_usize() -> usize {
        Self::max_value().to_usize()
    }
}

impl Index for i32 {
    fn from_usize(value: usize) -> Self {
        value as i32
    }
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl Index for i64 {
    fn from_usize(value: usize) -> Self {
        value as i64
    }
    fn to_usize(self) -> usize {
        self as usize
    }
}
