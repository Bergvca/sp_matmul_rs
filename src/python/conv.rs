//! Conversions between numpy arrays and the Rust-side CSR types.
//!
//! Inputs are borrowed read-only via `PyReadonlyArray1` — no copy. Outputs
//! consume the owned `Vec`s in `CsrMatrix` and hand them to numpy as buffer
//! owners.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::csr::{CsrMatrix, CsrView};
use crate::index::Index;
use crate::scalar::Scalar;

/// Borrow three numpy 1-D arrays as a `CsrView`. The view's `'py` lifetime
/// ties it to the input arrays' lifetimes; no data is copied.
pub(crate) fn borrow_csr_view<'py, V, I>(
    data: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    nrows: usize,
    ncols: usize,
) -> PyResult<CsrView<'py, V, I>>
where
    V: Scalar + numpy::Element,
    I: Index + numpy::Element,
{
    let data_arr = data.downcast::<PyArray1<V>>()?.readonly();
    let indptr_arr = indptr.downcast::<PyArray1<I>>()?.readonly();
    let indices_arr = indices.downcast::<PyArray1<I>>()?.readonly();

    // SAFETY-equivalent: as_slice returns Err on non-contiguous arrays,
    // which is the documented zero-copy contract.
    let data_slice: &[V] = data_arr.as_slice()?;
    let indptr_slice: &[I] = indptr_arr.as_slice()?;
    let indices_slice: &[I] = indices_arr.as_slice()?;

    // Promote the slice lifetimes from the temporary readonly view to 'py.
    // The underlying numpy buffer is held by the input Bound<PyAny> for the
    // entire call; the readonly view enforces no concurrent mutation.
    let data_slice: &'py [V] = unsafe { std::mem::transmute(data_slice) };
    let indptr_slice: &'py [I] = unsafe { std::mem::transmute(indptr_slice) };
    let indices_slice: &'py [I] = unsafe { std::mem::transmute(indices_slice) };

    CsrView::new(nrows, ncols, indptr_slice, indices_slice, data_slice)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Length of `indptr` minus one — the row count it implies. Validates the
/// array is 1-D and non-empty (indptr always has length nrows+1, so >= 1).
pub(crate) fn indptr_nrows<I>(indptr: &Bound<'_, PyAny>) -> PyResult<usize>
where
    I: Index + numpy::Element,
{
    let arr = indptr.downcast::<PyArray1<I>>()?;
    let len = arr.len();
    if len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "indptr must have length >= 1",
        ));
    }
    Ok(len - 1)
}

/// Build a `(data, indices, indptr)` tuple of numpy 1-D arrays from a
/// `CsrMatrix`. Each `Vec` is moved into numpy; no copy.
pub(crate) fn csr_to_py_tuple<'py, V, I>(
    py: Python<'py>,
    m: CsrMatrix<V, I>,
) -> PyResult<Bound<'py, PyTuple>>
where
    V: Scalar + numpy::Element,
    I: Index + numpy::Element,
{
    let data = m.data.into_pyarray(py);
    let indices = m.indices.into_pyarray(py);
    let indptr = m.indptr.into_pyarray(py);
    PyTuple::new(
        py,
        [data.into_any(), indices.into_any(), indptr.into_any()],
    )
}
