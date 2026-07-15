//! CSR matrix types — borrowed view and owned matrix.

use crate::index::Index;
use crate::scalar::Scalar;

/// Borrowed CSR view. Wraps externally-owned buffers (e.g. numpy arrays).
#[derive(Debug, Clone, Copy)]
pub struct CsrView<'a, V: Scalar, I: Index> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: &'a [I],
    pub indices: &'a [I],
    pub data: &'a [V],
}

impl<'a, V: Scalar, I: Index> CsrView<'a, V, I> {
    /// Construct after validating buffer lengths.
    pub fn new(
        nrows: usize,
        ncols: usize,
        indptr: &'a [I],
        indices: &'a [I],
        data: &'a [V],
    ) -> Result<Self, CsrError> {
        if indptr.len() != nrows + 1 {
            return Err(CsrError::IndptrLength {
                expected: nrows + 1,
                got: indptr.len(),
            });
        }
        if indices.len() != data.len() {
            return Err(CsrError::IndicesDataMismatch {
                indices: indices.len(),
                data: data.len(),
            });
        }
        Ok(Self {
            nrows,
            ncols,
            indptr,
            indices,
            data,
        })
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Owned CSR matrix — the function return type.
#[derive(Debug, Clone)]
pub struct CsrMatrix<V: Scalar, I: Index> {
    pub nrows: usize,
    pub ncols: usize,
    pub indptr: Vec<I>,
    pub indices: Vec<I>,
    pub data: Vec<V>,
}

impl<V: Scalar, I: Index> CsrMatrix<V, I> {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            indptr: vec![I::zero(); nrows + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Marker substring present in every index-overflow message ([`CsrError::IndexOverflow`]
/// and the [`narrow_indptr`] panic). The Python bindings match on it to map
/// overflow panics from the drivers to `OverflowError`.
pub(crate) const INDEX_OVERFLOW_MARKER: &str = "exceeds the maximum value representable";

/// Narrow a running `indptr` total to `I`, panicking on index overflow.
///
/// The drivers call this once per row where `indptr` is extended — off the
/// inner loops, so the cost is one comparison per row — aborting the moment
/// overflow becomes certain instead of spending the rest of the run producing
/// a corrupted (wrapped/negative) `indptr`. The Python bindings map the panic
/// to `OverflowError` (see `python::detach_mapping_overflow`).
#[inline]
pub(crate) fn narrow_indptr<I: Index>(nnz_total: usize) -> I {
    if nnz_total > I::max_usize() {
        panic!(
            "{}",
            CsrError::IndexOverflow {
                what: "non-zero count (nnz)",
                value: nnz_total,
                max: I::max_usize(),
            }
        );
    }
    I::from_usize(nnz_total)
}

/// Verify a result with `nnz` non-zeros and `ncols` columns can be represented
/// with index type `I` without overflow.
///
/// The kernels narrow `usize` offsets to `I` via [`Index::from_usize`], an
/// unchecked truncating cast on the hot path. When the true output exceeds the
/// index range every `indptr` value up to `nnz` and every column index up to
/// `ncols - 1` must still fit; this checks both bounds against [`Index::max_usize`].
/// The sizes are taken as plain `usize` arguments (`data.len()` and the `ncols`
/// field), which are unaffected by any truncation in the stored index arrays.
pub fn check_index_capacity<I: Index>(nnz: usize, ncols: usize) -> Result<(), CsrError> {
    let max = I::max_usize();
    if nnz > max {
        return Err(CsrError::IndexOverflow {
            what: "non-zero count (nnz)",
            value: nnz,
            max,
        });
    }
    // Column indices run 0..ncols, so the largest stored value is ncols - 1.
    let max_col = ncols.saturating_sub(1);
    if max_col > max {
        return Err(CsrError::IndexOverflow {
            what: "largest column index",
            value: max_col,
            max,
        });
    }
    Ok(())
}

/// Short-circuit empty / degenerate matmul-topn inputs to a zero output.
/// Shared by every sp_matmul_topn entry point so the boilerplate stays in one place.
pub fn matmul_topn_short_circuit<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
) -> Option<CsrMatrix<V, I>> {
    if top_n == 0 || a.nrows == 0 || a.nnz() == 0 || b.nnz() == 0 {
        return Some(CsrMatrix::zeros(a.nrows, b.ncols));
    }
    None
}

#[derive(Debug, thiserror::Error)]
pub enum CsrError {
    #[error("indptr length mismatch: expected {expected}, got {got}")]
    IndptrLength { expected: usize, got: usize },
    #[error("indices and data length mismatch: indices={indices}, data={data}")]
    IndicesDataMismatch { indices: usize, data: usize },
    #[error(
        "output {what} ({value}) {} by the index type (max {max}); \
         use a wider index type or reduce the problem size",
        INDEX_OVERFLOW_MARKER
    )]
    IndexOverflow {
        what: &'static str,
        value: usize,
        max: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{check_index_capacity, narrow_indptr, INDEX_OVERFLOW_MARKER};

    #[test]
    fn narrow_indptr_boundary() {
        assert_eq!(narrow_indptr::<i32>(i32::MAX as usize), i32::MAX);
        assert_eq!(
            narrow_indptr::<i64>(i32::MAX as usize + 1),
            i32::MAX as i64 + 1
        );
    }

    #[test]
    #[should_panic(expected = "exceeds the maximum value representable")]
    fn narrow_indptr_panics_past_i32_max() {
        narrow_indptr::<i32>(i32::MAX as usize + 1);
    }

    /// The Python bindings match on the marker to translate overflow panics
    /// into `OverflowError`; pin that the rendered message contains it.
    #[test]
    fn overflow_message_contains_marker() {
        let err = check_index_capacity::<i32>(i32::MAX as usize + 1, 10).unwrap_err();
        assert!(err.to_string().contains(INDEX_OVERFLOW_MARKER));
    }

    #[test]
    fn i32_capacity_boundary() {
        let max = i32::MAX as usize;
        // nnz exactly at the limit fits; one past it overflows.
        assert!(check_index_capacity::<i32>(max, 10).is_ok());
        assert!(check_index_capacity::<i32>(max + 1, 10).is_err());
        // Largest column index is ncols - 1, so ncols == max + 1 still fits.
        assert!(check_index_capacity::<i32>(0, max + 1).is_ok());
        assert!(check_index_capacity::<i32>(0, max + 2).is_err());
        // Empty matrix (ncols == 0) never overflows on the column bound.
        assert!(check_index_capacity::<i32>(0, 0).is_ok());
    }

    #[test]
    fn i64_widens_the_ceiling() {
        // Sizes that overflow i32 are comfortably within i64.
        let over_i32 = i32::MAX as usize + 1;
        assert!(check_index_capacity::<i64>(over_i32, over_i32).is_ok());
    }
}
