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
}
