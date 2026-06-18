//! `sp_matmul` — sparse CSR × CSR multiplication without top-n selection.
//!
//! Port of `sp_matmul.hpp`. Used internally when `top_n == B.ncols`.
//!
//! Two-pass: pass 1 sizes `C_indptr` and the total nnz via a per-row mask, pass 2
//! fills `C_indices`/`C_data` using a dense accumulator plus a linked-list of touched
//! columns. The intra-row column order in the output is the reverse linked-list walk
//! order — *not* sorted. Scipy and our parity tests handle this by sorting indices
//! before comparison.

use crate::csr::{CsrMatrix, CsrView};
use crate::index::Index;
use crate::scalar::Scalar;

const UNVISITED: usize = usize::MAX;
const HEAD_NIL: usize = usize::MAX - 1;

pub fn sp_matmul<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
) -> CsrMatrix<V, I> {
    assert_eq!(
        a.ncols, b.nrows,
        "sp_matmul: A.ncols ({}) must equal B.nrows ({})",
        a.ncols, b.nrows,
    );

    let nrows = a.nrows;
    let ncols = b.ncols;

    if nrows == 0 {
        return CsrMatrix::zeros(0, ncols);
    }
    if a.nnz() == 0 || b.nnz() == 0 {
        return CsrMatrix::zeros(nrows, ncols);
    }

    // Pass 1 — size c_indptr and total nnz.
    let mut mask: Vec<usize> = vec![UNVISITED; ncols];
    let mut c_indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    c_indptr.push(I::zero());
    let mut nnz_total: usize = 0;
    for i in 0..nrows {
        let mut row_nnz: usize = 0;
        let jj_start = a.indptr[i].to_usize();
        let jj_end = a.indptr[i + 1].to_usize();
        for jj in jj_start..jj_end {
            let j = a.indices[jj].to_usize();
            let kk_start = b.indptr[j].to_usize();
            let kk_end = b.indptr[j + 1].to_usize();
            for kk in kk_start..kk_end {
                let k = b.indices[kk].to_usize();
                if mask[k] != i {
                    mask[k] = i;
                    row_nnz += 1;
                }
            }
        }
        nnz_total += row_nnz;
        c_indptr.push(I::from_usize(nnz_total));
    }

    // Pass 2 — fill indices and data via dense accumulator + linked list.
    let mut sums: Vec<V> = vec![V::default(); ncols];
    let mut next: Vec<usize> = vec![UNVISITED; ncols];
    let mut c_indices: Vec<I> = Vec::with_capacity(nnz_total);
    let mut c_data: Vec<V> = Vec::with_capacity(nnz_total);

    for i in 0..nrows {
        let mut head: usize = HEAD_NIL;
        let mut length: usize = 0;

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
            if sums[head] != V::default() {
                c_indices.push(I::from_usize(head));
                c_data.push(sums[head]);
            }
            let temp = head;
            head = next[head];
            next[temp] = UNVISITED;
            sums[temp] = V::default();
        }
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

    /// 2×3 dense identity-like A and 3×2 B → check known product.
    /// A = [[1, 0, 2], [0, 3, 0]], B = [[1, 0], [0, 1], [1, 1]]
    /// C = A·B = [[3, 2], [0, 3]]
    #[test]
    fn small_known_product_f64_i32() {
        let a_indptr: Vec<i32> = vec![0, 2, 3];
        let a_indices: Vec<i32> = vec![0, 2, 1];
        let a_data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b_indptr: Vec<i32> = vec![0, 1, 2, 4];
        let b_indices: Vec<i32> = vec![0, 1, 0, 1];
        let b_data: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];

        let a = CsrView::new(2, 3, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(3, 2, &b_indptr, &b_indices, &b_data).unwrap();

        let c = sp_matmul(a, b);
        assert_eq!(c.nrows, 2);
        assert_eq!(c.ncols, 2);
        assert_eq!(c.indptr, vec![0, 2, 3]);

        // Pull row 0 and row 1, sort by index, then verify values.
        let row0: Vec<(i32, f64)> = (c.indptr[0] as usize..c.indptr[1] as usize)
            .map(|k| (c.indices[k], c.data[k]))
            .collect();
        let row1: Vec<(i32, f64)> = (c.indptr[1] as usize..c.indptr[2] as usize)
            .map(|k| (c.indices[k], c.data[k]))
            .collect();
        let mut row0_sorted = row0.clone();
        row0_sorted.sort_by_key(|(idx, _)| *idx);
        assert_eq!(row0_sorted, vec![(0, 3.0), (1, 2.0)]);
        assert_eq!(row1, vec![(1, 3.0)]);
    }

    #[test]
    fn empty_a_returns_zero_matrix() {
        let a_indptr: Vec<i32> = vec![0, 0, 0];
        let a_indices: Vec<i32> = vec![];
        let a_data: Vec<f64> = vec![];
        let b_indptr: Vec<i32> = vec![0, 1, 2, 3];
        let b_indices: Vec<i32> = vec![0, 1, 0];
        let b_data: Vec<f64> = vec![1.0, 2.0, 3.0];

        let a = CsrView::new(2, 3, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(3, 2, &b_indptr, &b_indices, &b_data).unwrap();
        let c = sp_matmul(a, b);
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.indptr, vec![0, 0, 0]);
    }

    #[test]
    fn zero_rows_returns_zero_rows() {
        let a_indptr: Vec<i32> = vec![0];
        let a_indices: Vec<i32> = vec![];
        let a_data: Vec<f64> = vec![];
        let b_indptr: Vec<i32> = vec![0, 1, 2, 3];
        let b_indices: Vec<i32> = vec![0, 1, 0];
        let b_data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let a = CsrView::new(0, 3, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(3, 2, &b_indptr, &b_indices, &b_data).unwrap();
        let c = sp_matmul(a, b);
        assert_eq!(c.nrows, 0);
        assert_eq!(c.ncols, 2);
        assert_eq!(c.indptr, vec![0]);
    }
}
