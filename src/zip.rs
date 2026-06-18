//! `zip_sp_matmul_topn` — combine top-n sub-matrices produced by independent chunks.
//!
//! Port of `zip_sp_matmul_topn.hpp`. Retained for the distributed/cluster use case
//! where chunks are computed on separate machines and must be materialised. The
//! in-process column-chunked driver in `chunked.rs` does the same thing via a
//! streaming merge and is the preferred single-node path.

use crate::csr::{CsrMatrix, CsrView};
use crate::index::Index;
use crate::maxheap::MaxHeap;
use crate::scalar::Scalar;

pub fn zip_sp_matmul_topn<V: Scalar, I: Index>(
    top_n: usize,
    c_chunks: &[CsrView<'_, V, I>],
) -> CsrMatrix<V, I> {
    if c_chunks.is_empty() {
        return CsrMatrix::zeros(0, 0);
    }

    let nrows = c_chunks[0].nrows;
    for chunk in c_chunks.iter().skip(1) {
        assert_eq!(
            chunk.nrows, nrows,
            "zip_sp_matmul_topn: all chunks must have the same nrows",
        );
    }

    // offset[j] = sum(c_chunks[0..j].ncols) — the column index in the zipped output
    // where chunk `j`'s columns start. Equivalent to the C++ quadratic loop, but a
    // straight prefix-sum is simpler and avoids the O(n²) shape.
    let mut offset: Vec<usize> = Vec::with_capacity(c_chunks.len());
    let mut acc: usize = 0;
    for chunk in c_chunks {
        offset.push(acc);
        acc += chunk.ncols;
    }
    let total_ncols = acc;

    if top_n == 0 {
        return CsrMatrix::zeros(nrows, total_ncols);
    }

    // Matches C++ `std::numeric_limits<eT>::min()` — see plan §5.4.
    let threshold = V::cpp_numeric_limits_min();
    let mut heap = MaxHeap::<V, I>::new(top_n, threshold);

    let mut indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    let mut indices: Vec<I> = Vec::new();
    let mut data: Vec<V> = Vec::new();
    indptr.push(I::zero());
    let mut nnz_total: usize = 0;

    for i in 0..nrows {
        let mut min = heap.reset();

        // Walk chunks in reverse to mirror the C++ insertion order. The
        // sort_by_insertion_order sort isn't used here, but matching insertion order
        // preserves bit-for-bit parity for value-tied entries via push_pop's
        // tie-breaking through the `order` field.
        for (j, chunk) in c_chunks.iter().enumerate().rev() {
            let start = chunk.indptr[i].to_usize();
            let end = chunk.indptr[i + 1].to_usize();
            let chunk_offset = I::from_usize(offset[j]);
            for k in start..end {
                let val = chunk.data[k];
                if val.partial_cmp(&min) == Some(std::cmp::Ordering::Greater) {
                    let abs_idx = chunk_offset + chunk.indices[k];
                    min = heap.push_pop(abs_idx, val);
                }
            }
        }

        heap.sort_by_value_desc();
        let n_set = heap.n_set();
        for entry in heap.entries() {
            indices.push(entry.idx);
            data.push(entry.val);
        }
        nnz_total += n_set;
        indptr.push(I::from_usize(nnz_total));
    }

    CsrMatrix {
        nrows,
        ncols: total_ncols,
        indptr,
        indices,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_chunks_returns_zero_matrix() {
        let chunks: Vec<CsrView<'_, f64, i32>> = vec![];
        let c = zip_sp_matmul_topn(5, &chunks);
        assert_eq!(c.nrows, 0);
        assert_eq!(c.ncols, 0);
    }

    #[test]
    fn top_n_zero_returns_zero_matrix() {
        let indptr = vec![0i32, 1, 2];
        let indices = vec![0i32, 1];
        let data = vec![1.0f64, 2.0];
        let chunk = CsrView::new(2, 3, &indptr, &indices, &data).unwrap();
        let c = zip_sp_matmul_topn(0, &[chunk]);
        assert_eq!(c.nrows, 2);
        assert_eq!(c.ncols, 3);
        assert_eq!(c.nnz(), 0);
    }

    /// Two chunks side by side. Each row has two nnz total, top_n=2 keeps both,
    /// sorted by value desc with offset applied to the second chunk's indices.
    #[test]
    fn two_chunks_offset_applied() {
        // chunk 0 (2x3): row 0 = {col 0: 3.0}, row 1 = {col 2: 1.0}
        let c0_indptr = vec![0i32, 1, 2];
        let c0_indices = vec![0i32, 2];
        let c0_data = vec![3.0f64, 1.0];
        let chunk0 = CsrView::new(2, 3, &c0_indptr, &c0_indices, &c0_data).unwrap();
        // chunk 1 (2x2): row 0 = {col 1: 5.0}, row 1 = {col 0: 2.0}
        let c1_indptr = vec![0i32, 1, 2];
        let c1_indices = vec![1i32, 0];
        let c1_data = vec![5.0f64, 2.0];
        let chunk1 = CsrView::new(2, 2, &c1_indptr, &c1_indices, &c1_data).unwrap();

        let c = zip_sp_matmul_topn(2, &[chunk0, chunk1]);
        assert_eq!(c.nrows, 2);
        assert_eq!(c.ncols, 5);
        assert_eq!(c.indptr, vec![0, 2, 4]);
        // Row 0 sorted desc: (col 1+3=4, 5.0), (col 0, 3.0)
        assert_eq!(c.data[0], 5.0);
        assert_eq!(c.indices[0], 4);
        assert_eq!(c.data[1], 3.0);
        assert_eq!(c.indices[1], 0);
        // Row 1 sorted desc: (col 0+3=3, 2.0), (col 2, 1.0)
        assert_eq!(c.data[2], 2.0);
        assert_eq!(c.indices[2], 3);
        assert_eq!(c.data[3], 1.0);
        assert_eq!(c.indices[3], 2);
    }
}
