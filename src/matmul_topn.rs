//! `sp_matmul_topn` — sparse CSR × CSR multiplication keeping top-n per row.
//!
//! The public entry point delegates to the column-chunked driver in
//! `chunked.rs` (Phase 2). The Phase 1 single-chunk implementation is kept as
//! `sp_matmul_topn_unchunked_for_tests`, marked `#[doc(hidden)]` and only
//! intended for the chunk-invariance test as an independent oracle.

use crate::chunked::{sp_matmul_topn_chunked, AccumMode, BProjection};
use crate::csr::{matmul_topn_short_circuit, CsrMatrix, CsrView};
use crate::index::Index;
use crate::matmul::sp_matmul;
use crate::maxheap::MaxHeap;
use crate::scalar::Scalar;

const UNVISITED: usize = usize::MAX;
const HEAD_NIL: usize = usize::MAX - 1;

#[derive(Debug, Clone, Copy, Default)]
pub enum SortMode {
    /// Output rows in original column order (default — matches Python `sort=False`).
    #[default]
    ByColumn,
    /// Output rows with the largest value first (matches Python `sort=True`).
    ByValueDesc,
}

#[derive(Debug, Clone, Copy)]
pub struct TopNOptions<V: Scalar> {
    /// Minimum value to keep. `None` accepts all.
    pub threshold: Option<V>,
    /// Output row ordering.
    pub sort: SortMode,
    /// Expected output density per row (informs pre-allocation). `None` → 1.0.
    pub density_hint: Option<f64>,
    /// Column-chunk width in B. `None` → cache-size-derived default (see
    /// `chunked::default_chunk_cols`). Larger values approach the single-chunk
    /// path; `Some(usize::MAX)` collapses to one chunk after clamping to `ncols`.
    pub chunk_cols: Option<usize>,
    /// Worker thread count. `None` or `Some(1)` runs the sequential chunked
    /// driver with zero rayon overhead. `Some(n > 1)` dispatches to the
    /// parallel driver in `parallel.rs` when the `rayon` feature is on; without
    /// the feature the value is silently ignored. `Some(0)` panics.
    ///
    /// A fresh `rayon::ThreadPool` is built per call, so callers can run the
    /// crate from inside their own thread pool without oversubscription. See
    pub n_threads: Option<usize>,
    /// Accumulator bookkeeping in the chunked kernel — see [`AccumMode`].
    /// `Adaptive` (default) picks per A-row between the linked-list and the
    /// dense-scan path based on expected update density.
    pub accum_mode: AccumMode,
    /// Override the B-row → chunk projection strategy. `None` (default) keeps
    /// the heuristic in `chunked::pick_projection`.
    pub projection: Option<BProjection>,
    /// Process A-rows in blocks of this size per column-chunk (O2,
    /// loop-interchange) so B-chunk fragments are reused across rows while
    /// L2-hot. `None` (default) lets the kernel decide: large-enough shapes
    /// auto-enable the tiled+blocked kernel (see `chunked::auto_tile`).
    /// `Some(0)` forces the classic row-at-a-time kernel (benchmark escape
    /// hatch).
    pub row_block: Option<usize>,
    /// Pre-bucket B per column-chunk with chunk-local `u16` indices (O3,
    /// tiled CSR) — removes per-row chunk projection and halves index
    /// bandwidth. Falls back to plain CSR when the shape is unsupported.
    pub tile_b: bool,
}

impl<V: Scalar> Default for TopNOptions<V> {
    fn default() -> Self {
        Self {
            threshold: None,
            sort: SortMode::ByColumn,
            density_hint: None,
            chunk_cols: None,
            n_threads: None,
            accum_mode: AccumMode::Adaptive,
            projection: None,
            row_block: None,
            tile_b: false,
        }
    }
}

pub fn sp_matmul_topn<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
) -> CsrMatrix<V, I> {
    assert_eq!(
        a.ncols, b.nrows,
        "sp_matmul_topn: A.ncols ({}) must equal B.nrows ({})",
        a.ncols, b.nrows,
    );

    if let Some(out) = matmul_topn_short_circuit(a, b, top_n) {
        return out;
    }

    if let Some(out) = maybe_parallel(a, b, top_n, opts) {
        return out;
    }

    // Fast path: when top_n exceeds ncols, no threshold, and column order is requested,
    // the kernel degenerates into plain matmul.
    if top_n >= b.ncols && opts.threshold.is_none() && matches!(opts.sort, SortMode::ByColumn) {
        return sp_matmul(a, b);
    }

    sp_matmul_topn_chunked(a, b, top_n, opts)
}

/// Dispatch to the rayon-backed driver when `opts.n_threads > 1` and the
/// `rayon` feature is enabled; otherwise return `None` and let the caller
/// fall through to the sequential paths. Without `rayon`, this is a no-op.
#[cfg(feature = "rayon")]
fn maybe_parallel<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
) -> Option<CsrMatrix<V, I>> {
    let n_threads = opts.n_threads.unwrap_or(1);
    if n_threads > 1 {
        Some(crate::parallel::sp_matmul_topn_parallel(a, b, top_n, opts))
    } else {
        None
    }
}

#[cfg(not(feature = "rayon"))]
fn maybe_parallel<V: Scalar, I: Index>(
    _a: CsrView<'_, V, I>,
    _b: CsrView<'_, V, I>,
    _top_n: usize,
    _opts: TopNOptions<V>,
) -> Option<CsrMatrix<V, I>> {
    None
}

/// Phase 1 single-chunk reference implementation, retained as an independent
/// oracle for the chunk-invariance test. Not part of the public API; the
/// production path goes through [`sp_matmul_topn`].
#[doc(hidden)]
pub fn sp_matmul_topn_unchunked_for_tests<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
) -> CsrMatrix<V, I> {
    assert_eq!(
        a.ncols, b.nrows,
        "sp_matmul_topn_unchunked_for_tests: A.ncols ({}) must equal B.nrows ({})",
        a.ncols, b.nrows,
    );

    if let Some(out) = matmul_topn_short_circuit(a, b, top_n) {
        return out;
    }

    let nrows = a.nrows;
    let ncols = b.ncols;
    let threshold = opts.threshold.unwrap_or_else(V::min_value);
    let density = opts.density_hint.unwrap_or(1.0);
    let cap_hint = ((nrows as f64) * (top_n as f64) * density).ceil() as usize;

    let mut sums: Vec<V> = vec![V::default(); ncols];
    let mut next: Vec<usize> = vec![UNVISITED; ncols];
    let mut heap = MaxHeap::<V, I>::new(top_n, threshold);
    let mut c_indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    let mut c_indices: Vec<I> = Vec::with_capacity(cap_hint);
    let mut c_data: Vec<V> = Vec::with_capacity(cap_hint);
    c_indptr.push(I::zero());
    let mut nnz_total: usize = 0;

    for i in 0..nrows {
        let mut head: usize = HEAD_NIL;
        let mut length: usize = 0;
        let mut min = heap.reset();

        let jj_start = a.indptr[i].to_usize();
        let jj_end = a.indptr[i + 1].to_usize();
        for jj in jj_start..jj_end {
            let j = a.indices[jj].to_usize();
            let v = a.data[jj];
            let kk_start = b.indptr[j].to_usize();
            let kk_end = b.indptr[j + 1].to_usize();
            for kk in kk_start..kk_end {
                let k = b.indices[kk].to_usize();
                sums[k] += v * b.data[kk];
                if next[k] == UNVISITED {
                    next[k] = head;
                    head = k;
                    length += 1;
                }
            }
        }

        for _ in 0..length {
            if sums[head].partial_cmp(&min) == Some(std::cmp::Ordering::Greater) {
                min = heap.push_pop(I::from_usize(head), sums[head]);
            }
            let temp = head;
            head = next[head];
            next[temp] = UNVISITED;
            sums[temp] = V::default();
        }

        match opts.sort {
            SortMode::ByColumn => heap.sort_by_insertion_order(),
            SortMode::ByValueDesc => heap.sort_by_value_desc(),
        }
        let n_set = heap.n_set();
        for entry in heap.entries() {
            c_indices.push(entry.idx);
            c_data.push(entry.val);
        }
        nnz_total += n_set;
        c_indptr.push(I::from_usize(nnz_total));
    }

    CsrMatrix {
        nrows,
        ncols,
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type CsrParts = (Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<f64>);

    fn make_a_b() -> CsrParts {
        // A (2×3): [[1, 0, 2], [0, 3, 4]]
        // B (3×4): [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]]
        let a_indptr = vec![0i32, 2, 4];
        let a_indices = vec![0i32, 2, 1, 2];
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_indptr = vec![0i32, 2, 4, 6];
        let b_indices = vec![0i32, 3, 1, 3, 2, 3];
        let b_data = vec![1.0f64, 5.0, 1.0, 6.0, 1.0, 7.0];
        (a_indptr, a_indices, a_data, b_indptr, b_indices, b_data)
    }

    /// C = A·B:
    /// row 0: [1, 0, 2, 5+14] = [1, 0, 2, 19]   → top-2 by value: 19, 2
    /// row 1: [0, 3, 4, 18+28] = [0, 3, 4, 46]  → top-2 by value: 46, 4
    #[test]
    fn top2_by_value_desc() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let c = sp_matmul_topn(a, b, 2, opts);
        assert_eq!(c.indptr, vec![0, 2, 2 + 2]);
        // row 0
        assert_eq!(c.data[0], 19.0);
        assert_eq!(c.indices[0], 3);
        assert_eq!(c.data[1], 2.0);
        assert_eq!(c.indices[1], 2);
        // row 1
        assert_eq!(c.data[2], 46.0);
        assert_eq!(c.indices[2], 3);
        assert_eq!(c.data[3], 4.0);
        assert_eq!(c.indices[3], 2);
    }

    #[test]
    fn top2_by_column_order() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByColumn,
            ..Default::default()
        };
        let c = sp_matmul_topn(a, b, 2, opts);
        // Row 0 top-2 are at columns {2, 3}; by column order they should appear as (2,3).
        let row0: Vec<(i32, f64)> = (c.indptr[0] as usize..c.indptr[1] as usize)
            .map(|k| (c.indices[k], c.data[k]))
            .collect();
        assert_eq!(row0, vec![(2, 2.0), (3, 19.0)]);
        let row1: Vec<(i32, f64)> = (c.indptr[1] as usize..c.indptr[2] as usize)
            .map(|k| (c.indices[k], c.data[k]))
            .collect();
        assert_eq!(row1, vec![(2, 4.0), (3, 46.0)]);
    }

    #[test]
    fn threshold_filters() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            threshold: Some(10.0),
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let c = sp_matmul_topn(a, b, 4, opts);
        // Only 19 (row 0) and 46 (row 1) exceed 10.
        assert_eq!(c.indptr, vec![0, 1, 2]);
        assert_eq!(c.data, vec![19.0, 46.0]);
        assert_eq!(c.indices, vec![3, 3]);
    }

    #[test]
    fn top_n_zero_returns_zero_matrix() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let c = sp_matmul_topn(a, b, 0, TopNOptions::default());
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.indptr, vec![0, 0, 0]);
    }

    #[test]
    fn top_n_equals_ncols_matches_sp_matmul() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let c_topn = sp_matmul_topn(a, b, 4, TopNOptions::default());
        let c_plain = sp_matmul(a, b);
        assert_eq!(c_topn.indptr, c_plain.indptr);
        // indices may differ in order — sort each row first
        let sort_rows = |c: &CsrMatrix<f64, i32>| -> Vec<(i32, f64)> {
            let mut out = Vec::new();
            for i in 0..c.nrows {
                let mut row: Vec<(i32, f64)> = (c.indptr[i] as usize..c.indptr[i + 1] as usize)
                    .map(|k| (c.indices[k], c.data[k]))
                    .collect();
                row.sort_by_key(|(idx, _)| *idx);
                out.extend(row);
            }
            out
        };
        assert_eq!(sort_rows(&c_topn), sort_rows(&c_plain));
    }

    /// The unchunked oracle stays bit-equivalent to the chunked driver at the
    /// degenerate `chunk_cols >= ncols` setting on the small fixture. This is
    /// a tight local smoke test; `tests/chunk_invariance.rs` covers the breadth.
    #[test]
    fn unchunked_oracle_matches_chunked_one_chunk() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            chunk_cols: Some(usize::MAX),
            ..Default::default()
        };
        let chunked = sp_matmul_topn(a, b, 2, opts);
        let oracle = sp_matmul_topn_unchunked_for_tests(a, b, 2, opts);
        assert_eq!(chunked.indptr, oracle.indptr);
        assert_eq!(chunked.indices, oracle.indices);
        assert_eq!(chunked.data, oracle.data);
    }

    /// `n_threads = None` skips the parallel path; output must match the
    /// `Some(1)` setting which also routes through the sequential path.
    #[test]
    fn n_threads_none_and_one_agree() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts_none = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let opts_one = TopNOptions {
            n_threads: Some(1),
            ..opts_none
        };
        let c_none = sp_matmul_topn(a, b, 2, opts_none);
        let c_one = sp_matmul_topn(a, b, 2, opts_one);
        assert_eq!(c_none.indptr, c_one.indptr);
        assert_eq!(c_none.indices, c_one.indices);
        assert_eq!(c_none.data, c_one.data);
    }

    /// `n_threads = Some(4)` routes through the parallel driver; output must
    /// match the sequential `n_threads = None` baseline on the small fixture.
    #[cfg(feature = "rayon")]
    #[test]
    fn n_threads_many_matches_sequential() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let base = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let seq = sp_matmul_topn(a, b, 2, base);
        let par = sp_matmul_topn(
            a,
            b,
            2,
            TopNOptions {
                n_threads: Some(4),
                ..base
            },
        );
        assert_eq!(par.indptr, seq.indptr);
        assert_eq!(par.indices, seq.indices);
        assert_eq!(par.data, seq.data);
    }
}
