//! Tiled-CSR representation of B (plan O3, "Strategie C").
//!
//! `O(nnz(B))` preprocessing that buckets B's entries per column-chunk: every
//! (chunk, B-row) pair gets its own contiguous (indices, data) segment, with
//! column indices stored *chunk-local* as `u16`. The chunked kernel then
//! streams each segment directly — no `partition_point`, no cursor state —
//! and the index bandwidth halves versus 32-bit global indices.
//!
//! The build cost is amortised over all A-rows: B is re-walked once per A-row
//! by the kernel, so for `nrows(A) ≫ 1` the two extra passes over B vanish.

use crate::csr::CsrView;
use crate::index::Index;
use crate::scalar::Scalar;

/// B bucketed per column-chunk. Segments are ordered chunk-major: the segment
/// for (chunk `c`, B-row `j`) lives at `seg_ptr[c * nrows + j]..seg_ptr[c * nrows + j + 1]`,
/// so one chunk's data is one contiguous region — the streaming order of the
/// blocked kernel.
#[derive(Debug)]
pub struct TiledB<V> {
    pub chunk_cols: usize,
    pub n_chunks: usize,
    pub nrows: usize,
    pub ncols: usize,
    seg_ptr: Vec<u32>,
    idx: Vec<u16>,
    data: Vec<V>,
}

impl<V: Scalar> TiledB<V> {
    /// Tiling stores chunk-local column offsets in `u16` and global entry
    /// offsets in `u32`; shapes outside those bounds fall back to plain CSR.
    pub fn supports<I: Index>(b: &CsrView<'_, V, I>, chunk_cols: usize) -> bool {
        chunk_cols <= (u16::MAX as usize + 1) && b.nnz() <= u32::MAX as usize
    }

    /// Two counting-sort passes over B: count per (chunk, row) segment, prefix
    /// sum, then scatter entries into place. `O(nnz(B) + n_chunks * nrows)`.
    pub fn build<I: Index>(b: CsrView<'_, V, I>, chunk_cols: usize) -> Self {
        assert!(chunk_cols > 0, "chunk_cols must be > 0");
        assert!(
            Self::supports(&b, chunk_cols),
            "TiledB unsupported: chunk_cols={} nnz={}",
            chunk_cols,
            b.nnz()
        );
        let nrows = b.nrows;
        let ncols = b.ncols;
        let n_chunks = ncols.div_ceil(chunk_cols).max(1);
        let nnz = b.nnz();
        let nseg = n_chunks * nrows;

        let mut seg_ptr: Vec<u32> = vec![0; nseg + 1];
        for j in 0..nrows {
            let s = b.indptr[j].to_usize();
            let e = b.indptr[j + 1].to_usize();
            for kk in s..e {
                let c = b.indices[kk].to_usize() / chunk_cols;
                seg_ptr[c * nrows + j + 1] += 1;
            }
        }
        for i in 1..=nseg {
            seg_ptr[i] += seg_ptr[i - 1];
        }

        let mut cur: Vec<u32> = seg_ptr[..nseg].to_vec();
        let mut idx: Vec<u16> = vec![0; nnz];
        let mut data: Vec<V> = vec![V::default(); nnz];
        for j in 0..nrows {
            let s = b.indptr[j].to_usize();
            let e = b.indptr[j + 1].to_usize();
            for kk in s..e {
                let k = b.indices[kk].to_usize();
                let c = k / chunk_cols;
                let slot = c * nrows + j;
                let pos = cur[slot] as usize;
                cur[slot] += 1;
                idx[pos] = (k - c * chunk_cols) as u16;
                data[pos] = b.data[kk];
            }
        }

        TiledB {
            chunk_cols,
            n_chunks,
            nrows,
            ncols,
            seg_ptr,
            idx,
            data,
        }
    }

    /// The (chunk-local indices, values) segment of B-row `row` within `chunk`.
    #[inline(always)]
    pub fn segment(&self, chunk: usize, row: usize) -> (&[u16], &[V]) {
        let o = chunk * self.nrows + row;
        let s = self.seg_ptr[o] as usize;
        let e = self.seg_ptr[o + 1] as usize;
        (&self.idx[s..e], &self.data[s..e])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// B (3×4): [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]] — chunked fixture.
    fn make_b() -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        (
            vec![0i32, 2, 4, 6],
            vec![0i32, 3, 1, 3, 2, 3],
            vec![1.0f64, 5.0, 1.0, 6.0, 1.0, 7.0],
        )
    }

    #[test]
    fn segments_match_csr_rows() {
        let (ip, ix, d) = make_b();
        let b = CsrView::new(3, 4, &ip, &ix, &d).unwrap();
        for chunk_cols in [1usize, 2, 3, 4, 8] {
            let t = TiledB::build(b, chunk_cols);
            assert_eq!(t.n_chunks, 4usize.div_ceil(chunk_cols));
            // Reassemble each B-row from its chunk segments; must equal the CSR row.
            for j in 0..3 {
                let mut got: Vec<(usize, f64)> = Vec::new();
                for c in 0..t.n_chunks {
                    let (si, sd) = t.segment(c, j);
                    for (p, &kl) in si.iter().enumerate() {
                        got.push((c * chunk_cols + kl as usize, sd[p]));
                    }
                }
                let s = ip[j] as usize;
                let e = ip[j + 1] as usize;
                let want: Vec<(usize, f64)> = (s..e).map(|kk| (ix[kk] as usize, d[kk])).collect();
                assert_eq!(got, want, "chunk_cols={chunk_cols} row={j}");
            }
        }
    }

    #[test]
    fn supports_bounds() {
        let (ip, ix, d) = make_b();
        let b = CsrView::new(3, 4, &ip, &ix, &d).unwrap();
        assert!(TiledB::supports(&b, 4));
        assert!(TiledB::supports(&b, u16::MAX as usize + 1));
        assert!(!TiledB::supports(&b, u16::MAX as usize + 2));
    }
}
