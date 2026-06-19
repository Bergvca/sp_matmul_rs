//! Bounded max-heap for top-n selection per row.
//!
//! Port of `maxheap.hpp`. A fixed-capacity heap keyed by score (`V`), tagged with the
//! source column index (`I`) and an insertion-order field used by the column-order
//! sort variant.
//!
//! The heap is structurally a *min*-heap by `val`: the root is the smallest of the
//! current top-n, which is what makes `push_pop` cheap (drop the smallest, insert the
//! new candidate, return the new minimum). This mirrors the C++ implementation
//! (`std::make_heap` with `std::greater<Score>`).
//!
//! See plan §3.2 — one heap is reused across all column-chunks of a single row.

use std::cmp::Ordering;

use crate::index::Index;
use crate::scalar::Scalar;

#[derive(Debug, Clone, Copy)]
pub struct Score<V: Scalar, I: Index> {
    pub order: u32,
    pub idx: I,
    pub val: V,
}

#[derive(Debug)]
pub struct MaxHeap<V: Scalar, I: Index> {
    capacity: usize,
    init: V,
    n_set: u32,
    heap: Vec<Score<V, I>>,
}

impl<V: Scalar, I: Index> MaxHeap<V, I> {
    pub fn new(capacity: usize, threshold: V) -> Self {
        let heap = (0..capacity)
            .map(|_| Score {
                order: u32::MAX,
                idx: I::zero(),
                val: threshold,
            })
            .collect();
        // All-equal values trivially satisfy the heap property, so no make_heap is
        // needed. The C++ constructor additionally calls `pop_heap`; with all entries
        // equal that has no observable effect, so we omit it.
        Self {
            capacity,
            init: threshold,
            n_set: 0,
            heap,
        }
    }

    /// Reset between rows. Returns the current threshold (root value).
    pub fn reset(&mut self) -> V {
        self.n_set = 0;
        for entry in &mut self.heap {
            entry.order = u32::MAX;
            entry.idx = I::zero();
            entry.val = self.init;
        }
        self.init
    }

    /// Number of real entries, capped at `capacity`. Counts `push_pop` calls — the
    /// caller is responsible for only calling `push_pop` when `val > min`, which is
    /// what keeps this equal to the count of real (non-sentinel) entries.
    pub fn n_set(&self) -> usize {
        (self.n_set as usize).min(self.capacity)
    }

    /// Push `(idx, val)`, pop the smallest. Returns the new root (== running threshold).
    ///
    /// Callers must guard with `val > current_min` — pushing values that wouldn't
    /// displace the root corrupts the `n_set`/sentinel accounting.
    pub fn push_pop(&mut self, idx: I, val: V) -> V {
        let last = self.capacity - 1;
        // pop_heap: move the min to `last`, then restore the heap over [0..last].
        self.heap.swap(0, last);
        sift_down(&mut self.heap, 0, last);
        // Overwrite the back slot with the new entry.
        self.heap[last] = Score {
            order: self.n_set,
            idx,
            val,
        };
        self.n_set = self.n_set.saturating_add(1);
        // push_heap: restore the heap over the full [0..capacity] range.
        sift_up(&mut self.heap, last);
        self.heap[0].val
    }

    /// Sort by insertion order — restores the column order in the output row.
    /// Invalidates the heap; caller must `reset` before reuse.
    pub fn sort_by_insertion_order(&mut self) {
        self.heap.sort_by_key(|s| s.order);
    }

    /// Sort by value descending — largest score first in the output row.
    /// Invalidates the heap; caller must `reset` before reuse.
    pub fn sort_by_value_desc(&mut self) {
        self.heap
            .sort_by(|a, b| b.val.partial_cmp(&a.val).unwrap_or(Ordering::Equal));
    }

    pub fn entries(&self) -> &[Score<V, I>] {
        &self.heap[..self.n_set()]
    }
}

/// Min-heap sift-down at index `i` over `heap[0..len]`.
fn sift_down<V: Scalar, I: Index>(heap: &mut [Score<V, I>], mut i: usize, len: usize) {
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut smallest = i;
        if left < len && lt(heap[left].val, heap[smallest].val) {
            smallest = left;
        }
        if right < len && lt(heap[right].val, heap[smallest].val) {
            smallest = right;
        }
        if smallest == i {
            return;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
}

/// Min-heap sift-up at index `i`.
fn sift_up<V: Scalar, I: Index>(heap: &mut [Score<V, I>], mut i: usize) {
    while i > 0 {
        let parent = (i - 1) / 2;
        if lt(heap[i].val, heap[parent].val) {
            heap.swap(i, parent);
            i = parent;
        } else {
            return;
        }
    }
}

#[inline]
fn lt<V: Scalar>(a: V, b: V) -> bool {
    a.partial_cmp(&b) == Some(Ordering::Less)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{NumCast, ToPrimitive};

    fn v<V: Scalar>(x: u8) -> V {
        <V as NumCast>::from(x).unwrap()
    }

    fn back<V: Scalar>(x: V) -> u8 {
        <V as ToPrimitive>::to_u8(&x).unwrap()
    }

    fn push_guarded<V: Scalar, I: Index>(
        heap: &mut MaxHeap<V, I>,
        min: &mut V,
        idx: usize,
        val: V,
    ) {
        if val.partial_cmp(min) == Some(Ordering::Greater) {
            *min = heap.push_pop(I::from_usize(idx), val);
        }
    }

    fn run_push_pop_orders_top_n<V: Scalar, I: Index>() {
        let mut heap = MaxHeap::<V, I>::new(3, v::<V>(0));
        let mut min = v::<V>(0);
        for (i, x) in [1u8, 5, 3, 8, 2, 7].iter().enumerate() {
            push_guarded(&mut heap, &mut min, i, v::<V>(*x));
        }
        heap.sort_by_value_desc();
        let got: Vec<u8> = heap.entries().iter().map(|s| back(s.val)).collect();
        assert_eq!(got, vec![8, 7, 5]);
    }

    fn run_threshold_filters<V: Scalar, I: Index>() {
        let threshold = v::<V>(4);
        let mut heap = MaxHeap::<V, I>::new(3, threshold);
        let mut min = threshold;
        for (i, x) in [1u8, 5, 3].iter().enumerate() {
            push_guarded(&mut heap, &mut min, i, v::<V>(*x));
        }
        heap.sort_by_value_desc();
        let got: Vec<u8> = heap.entries().iter().map(|s| back(s.val)).collect();
        assert_eq!(got, vec![5]);
    }

    fn run_insertion_order_sort_preserves_column_order<V: Scalar, I: Index>() {
        let mut heap = MaxHeap::<V, I>::new(3, v::<V>(0));
        let mut min = v::<V>(0);
        for (idx, val) in [(10usize, 5u8), (2, 3), (7, 8)].iter() {
            push_guarded(&mut heap, &mut min, *idx, v::<V>(*val));
        }
        heap.sort_by_insertion_order();
        let got: Vec<usize> = heap.entries().iter().map(|s| s.idx.to_usize()).collect();
        assert_eq!(got, vec![10, 2, 7]);
    }

    fn run_reset_clears_state<V: Scalar, I: Index>() {
        let mut heap = MaxHeap::<V, I>::new(2, v::<V>(0));
        let mut min = v::<V>(0);
        for (i, x) in [5u8, 3].iter().enumerate() {
            push_guarded(&mut heap, &mut min, i, v::<V>(*x));
        }
        heap.sort_by_value_desc();
        assert_eq!(heap.n_set(), 2);

        let returned_init = heap.reset();
        assert_eq!(heap.n_set(), 0);
        assert_eq!(returned_init.partial_cmp(&v::<V>(0)), Some(Ordering::Equal));

        let mut min2 = returned_init;
        for (i, x) in [9u8, 1, 7].iter().enumerate() {
            push_guarded(&mut heap, &mut min2, i + 100, v::<V>(*x));
        }
        heap.sort_by_value_desc();
        let got: Vec<u8> = heap.entries().iter().map(|s| back(s.val)).collect();
        assert_eq!(got, vec![9, 7]);
    }

    fn run_capacity_one<V: Scalar, I: Index>() {
        let mut heap = MaxHeap::<V, I>::new(1, v::<V>(0));
        let mut min = v::<V>(0);
        for (i, x) in [3u8, 7, 5].iter().enumerate() {
            push_guarded(&mut heap, &mut min, i, v::<V>(*x));
        }
        heap.sort_by_value_desc();
        let got: Vec<u8> = heap.entries().iter().map(|s| back(s.val)).collect();
        assert_eq!(got, vec![7]);
    }

    macro_rules! dtype_tests {
        ($mod_name:ident, $V:ty, $I:ty) => {
            mod $mod_name {
                use super::*;

                #[test]
                fn push_pop_orders_top_n() {
                    run_push_pop_orders_top_n::<$V, $I>();
                }

                #[test]
                fn threshold_filters() {
                    run_threshold_filters::<$V, $I>();
                }

                #[test]
                fn insertion_order_sort_preserves_column_order() {
                    run_insertion_order_sort_preserves_column_order::<$V, $I>();
                }

                #[test]
                fn reset_clears_state() {
                    run_reset_clears_state::<$V, $I>();
                }

                #[test]
                fn capacity_one() {
                    run_capacity_one::<$V, $I>();
                }
            }
        };
    }

    dtype_tests!(f32_i32, f32, i32);
    dtype_tests!(f32_i64, f32, i64);
    dtype_tests!(f64_i32, f64, i32);
    dtype_tests!(f64_i64, f64, i64);
    dtype_tests!(i32_i32, i32, i32);
    dtype_tests!(i32_i64, i32, i64);
    dtype_tests!(i64_i32, i64, i32);
    dtype_tests!(i64_i64, i64, i64);
}
