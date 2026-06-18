//! `Index` trait — the integer type of the `indptr` / `indices` arrays.
//!
//! Implemented for `i32` and `i64` to match the C++ extension.

use num_traits::PrimInt;
use std::fmt::Debug;

pub trait Index: PrimInt + Debug + Send + Sync + 'static {
    fn from_usize(value: usize) -> Self;
    fn to_usize(self) -> usize;
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
