//! PyO3 + numpy bindings. Feature `python`.
//!
//! Exposes `sp_matmul`, `sp_matmul_topn`, `zip_sp_matmul_topn`, and the
//! `_rust_parallel_enabled` flag as the cdylib `sp_matmul_rs._core`. The
//! user-facing Python API lives in `python/sp_matmul_rs/api.py`; the
//! `py_*`-prefixed bindings here are an implementation detail.

#![allow(non_snake_case)]
// Bindings carry one positional per CSR buffer plus the kwargs already in the
// public Python API — the wrapper handles ergonomics, this is intentionally
// flat. `useless_conversion` fires inside the macro expansion of `dispatch_vi!`
// (format!() over Bound<PyString>'s Display impl); harmless.
#![allow(clippy::too_many_arguments, clippy::useless_conversion)]

use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

pub mod conv;
pub mod dispatch;

use crate::csr::{CsrMatrix, CsrView};
use crate::index::Index;
use crate::matmul_topn::{SortMode, TopNOptions};
use crate::scalar::Scalar;
use conv::{borrow_csr_view, csr_to_py_tuple, indptr_nrows};

#[pymodule]
fn _core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sp_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(py_sp_matmul_topn, m)?)?;
    m.add_function(wrap_pyfunction!(py_zip_sp_matmul_topn, m)?)?;
    m.add_function(wrap_pyfunction!(py_kernel_info, m)?)?;
    let parallel = cfg!(feature = "rayon");
    m.add("_rust_parallel_enabled", parallel)?;
    m.add("_has_openmp_support", parallel)?;
    m.add("__rust_build__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

/// Detected cache size and the kernel-tuning defaults derived from it.
#[pyfunction]
#[pyo3(name = "kernel_info")]
fn py_kernel_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    use crate::chunked::{
        default_chunk_cols, detect_l1d_bytes, DEFAULT_ROW_BLOCK, DENSE_MIN_DENSITY, FALLBACK_L1D_BYTES,
    };

    let detected = detect_l1d_bytes();
    let info = PyDict::new(py);
    info.set_item("l1d_bytes", detected.unwrap_or(FALLBACK_L1D_BYTES))?;
    info.set_item("l1d_detected", detected.is_some())?;
    info.set_item("l1d_fallback_bytes", FALLBACK_L1D_BYTES)?;

    let chunk_cols = PyDict::new(py);
    chunk_cols.set_item("int32", default_chunk_cols::<i32>())?;
    chunk_cols.set_item("int64", default_chunk_cols::<i64>())?;
    chunk_cols.set_item("float32", default_chunk_cols::<f32>())?;
    chunk_cols.set_item("float64", default_chunk_cols::<f64>())?;
    info.set_item("default_chunk_cols", chunk_cols)?;

    info.set_item("default_row_block", DEFAULT_ROW_BLOCK)?;
    info.set_item("dense_min_density", DENSE_MIN_DENSITY)?;
    info.set_item("parallel_enabled", cfg!(feature = "rayon"))?;
    Ok(info)
}

#[pyfunction]
#[pyo3(signature = (
    nrows, ncols,
    A_data, A_indptr, A_indices,
    B_data, B_indptr, B_indices,
    n_threads=None,
))]
#[pyo3(name = "sp_matmul")]
fn py_sp_matmul<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    A_data: &Bound<'py, PyAny>,
    A_indptr: &Bound<'py, PyAny>,
    A_indices: &Bound<'py, PyAny>,
    B_data: &Bound<'py, PyAny>,
    B_indptr: &Bound<'py, PyAny>,
    B_indices: &Bound<'py, PyAny>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    crate::dispatch_vi!(py, A_data, A_indptr, B_data, B_indptr, V, I => {
        run_sp_matmul::<V, I>(
            py, nrows, ncols,
            A_data, A_indptr, A_indices,
            B_data, B_indptr, B_indices,
            n_threads,
        )
    })
}

fn run_sp_matmul<'py, V, I>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    A_data: &Bound<'py, PyAny>,
    A_indptr: &Bound<'py, PyAny>,
    A_indices: &Bound<'py, PyAny>,
    B_data: &Bound<'py, PyAny>,
    B_indptr: &Bound<'py, PyAny>,
    B_indices: &Bound<'py, PyAny>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>>
where
    V: Scalar + numpy::Element,
    I: Index + numpy::Element,
{
    let b_nrows = indptr_nrows::<I>(B_indptr)?;
    let a = borrow_csr_view::<V, I>(A_data, A_indptr, A_indices, nrows, b_nrows)?;
    let b = borrow_csr_view::<V, I>(B_data, B_indptr, B_indices, b_nrows, ncols)?;
    // Sequential `sp_matmul` ignores n_threads; the wrapper short-circuits to
    // this binding only when n_threads <= 1, so we don't need to dispatch.
    let _ = n_threads;
    let c: CsrMatrix<V, I> = py.detach(|| crate::sp_matmul(a, b));
    csr_to_py_tuple(py, c)
}

#[pyfunction]
#[pyo3(signature = (
    top_n, nrows, ncols,
    A_data, A_indptr, A_indices,
    B_data, B_indptr, B_indices,
    threshold=None, density=1.0, sort=false,
    chunk_cols=None, n_threads=None,
))]
#[pyo3(name = "sp_matmul_topn")]
#[allow(clippy::too_many_arguments)]
fn py_sp_matmul_topn<'py>(
    py: Python<'py>,
    top_n: usize,
    nrows: usize,
    ncols: usize,
    A_data: &Bound<'py, PyAny>,
    A_indptr: &Bound<'py, PyAny>,
    A_indices: &Bound<'py, PyAny>,
    B_data: &Bound<'py, PyAny>,
    B_indptr: &Bound<'py, PyAny>,
    B_indices: &Bound<'py, PyAny>,
    threshold: Option<Bound<'py, PyAny>>,
    density: f64,
    sort: bool,
    chunk_cols: Option<usize>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    crate::dispatch_vi!(py, A_data, A_indptr, B_data, B_indptr, V, I => {
        run_sp_matmul_topn::<V, I>(
            py, top_n, nrows, ncols,
            A_data, A_indptr, A_indices,
            B_data, B_indptr, B_indices,
            threshold.as_ref(), density, sort, chunk_cols, n_threads,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn run_sp_matmul_topn<'py, V, I>(
    py: Python<'py>,
    top_n: usize,
    nrows: usize,
    ncols: usize,
    A_data: &Bound<'py, PyAny>,
    A_indptr: &Bound<'py, PyAny>,
    A_indices: &Bound<'py, PyAny>,
    B_data: &Bound<'py, PyAny>,
    B_indptr: &Bound<'py, PyAny>,
    B_indices: &Bound<'py, PyAny>,
    threshold: Option<&Bound<'py, PyAny>>,
    density: f64,
    sort: bool,
    chunk_cols: Option<usize>,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>>
where
    V: Scalar + numpy::Element + for<'a> pyo3::FromPyObject<'a>,
    I: Index + numpy::Element,
{
    let b_nrows = indptr_nrows::<I>(B_indptr)?;
    let a = borrow_csr_view::<V, I>(A_data, A_indptr, A_indices, nrows, b_nrows)?;
    let b = borrow_csr_view::<V, I>(B_data, B_indptr, B_indices, b_nrows, ncols)?;
    let threshold: Option<V> = match threshold {
        Some(t) => Some(t.extract()?),
        None => None,
    };
    let opts = TopNOptions {
        threshold,
        sort: if sort { SortMode::ByValueDesc } else { SortMode::ByColumn },
        density_hint: Some(density),
        chunk_cols,
        n_threads,
        ..Default::default()
    };
    let c: CsrMatrix<V, I> = py.detach(|| crate::sp_matmul_topn(a, b, top_n, opts));
    csr_to_py_tuple(py, c)
}

#[pyfunction]
#[pyo3(signature = (top_n, Z_max_nnz, nrows, B_ncols, data, indptr, indices))]
#[pyo3(name = "zip_sp_matmul_topn")]
fn py_zip_sp_matmul_topn<'py>(
    py: Python<'py>,
    top_n: usize,
    Z_max_nnz: usize,
    nrows: usize,
    B_ncols: &Bound<'py, PyAny>,
    data: &Bound<'py, PyList>,
    indptr: &Bound<'py, PyList>,
    indices: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = Z_max_nnz;
    let n_chunks = data.len();
    if n_chunks == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "zip_sp_matmul_topn: at least one chunk is required",
        ));
    }
    if indptr.len() != n_chunks || indices.len() != n_chunks {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "zip_sp_matmul_topn: data, indptr, indices must have the same length",
        ));
    }
    let ncols: Vec<usize> = B_ncols.extract()?;
    if ncols.len() != n_chunks {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "zip_sp_matmul_topn: B_ncols length must match number of chunks",
        ));
    }

    let first_data = data.get_item(0)?;
    let first_indptr = indptr.get_item(0)?;

    crate::dispatch_vi!(py, &first_data, &first_indptr, &first_data, &first_indptr, V, I => {
        run_zip::<V, I>(py, top_n, nrows, &ncols, data, indptr, indices)
    })
}

fn run_zip<'py, V, I>(
    py: Python<'py>,
    top_n: usize,
    nrows: usize,
    ncols: &[usize],
    data: &Bound<'py, PyList>,
    indptr: &Bound<'py, PyList>,
    indices: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyTuple>>
where
    V: Scalar + numpy::Element,
    I: Index + numpy::Element,
{
    let n_chunks = data.len();
    // Borrow each chunk's arrays via the readonly views. Held for the duration
    // of the call so the slices remain valid across `allow_threads`.
    let mut data_views = Vec::with_capacity(n_chunks);
    let mut indptr_views = Vec::with_capacity(n_chunks);
    let mut indices_views = Vec::with_capacity(n_chunks);
    for k in 0..n_chunks {
        let d = data.get_item(k)?;
        let p = indptr.get_item(k)?;
        let i = indices.get_item(k)?;
        let d_arr = d.downcast::<numpy::PyArray1<V>>()?.readonly();
        let p_arr = p.downcast::<numpy::PyArray1<I>>()?.readonly();
        let i_arr = i.downcast::<numpy::PyArray1<I>>()?.readonly();
        data_views.push(d_arr);
        indptr_views.push(p_arr);
        indices_views.push(i_arr);
    }

    let mut chunks: Vec<CsrView<'_, V, I>> = Vec::with_capacity(n_chunks);
    for k in 0..n_chunks {
        let d_slice = data_views[k].as_slice()?;
        let p_slice = indptr_views[k].as_slice()?;
        let i_slice = indices_views[k].as_slice()?;
        let view = CsrView::new(nrows, ncols[k], p_slice, i_slice, d_slice)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        chunks.push(view);
    }

    let c: CsrMatrix<V, I> = py.detach(|| crate::zip_sp_matmul_topn(top_n, &chunks));
    csr_to_py_tuple(py, c)
}
