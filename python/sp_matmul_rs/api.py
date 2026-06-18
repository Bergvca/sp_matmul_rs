from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import psutil
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from sp_matmul_rs import _core
from sp_matmul_rs.types import assert_idx_dtype, assert_supported_dtype, ensure_compatible_dtype

if TYPE_CHECKING:
    from numpy.types import DTypeLike

__all__ = ["sp_matmul", "sp_matmul_topn", "zip_sp_matmul_topn"]


_N_CORES = max(1, (psutil.cpu_count(logical=False) or 2) - 1)


def sp_matmul(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B.

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must have a {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A
            or the shape of B.T should match A.
            `B` must have a {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        n_threads: number of threads to use, `None` implies sequential processing, -1 will use all
            but one of the available cores.
        idx_dtype: dtype to use for the indices, defaults to 32-bit integers.

    Throws:
        TypeError: when A, B are not trivially convertable to a `CSR matrix`.

    Returns:
        C: result matrix.
    """
    idx_dtype = assert_idx_dtype(idx_dtype)
    n_threads: int = n_threads or 1
    if n_threads < 0:
        n_threads = _N_CORES

    if isinstance(A, csc_matrix) and isinstance(B, csc_matrix) and A.shape[0] == B.shape[1]:
        A = A.transpose()
        B = B.transpose()
    elif isinstance(A, (coo_matrix, csc_matrix)):
        A = A.tocsr(False)
    elif not isinstance(A, csr_matrix):
        msg = f"type of `A` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(A)}`"
        raise TypeError(msg)

    if not isinstance(B, (csr_matrix, coo_matrix, csc_matrix)):
        msg = f"type of `B` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(B)}`"
        raise TypeError(msg)

    A_nrows, A_ncols = A.shape
    B_nrows, B_ncols = B.shape

    if A_ncols == B_nrows:
        if isinstance(B, (coo_matrix, csc_matrix)):
            B = B.tocsr(False)
    elif A_ncols == B_ncols:
        B = B.transpose() if isinstance(B, csc_matrix) else B.transpose().tocsr(False)
        B_nrows, B_ncols = B.shape
    else:
        msg = (
            "Matrices `A` and `B` have incompatible shapes. `A.shape[1]` must be equal to `B.shape[0]` or `B.shape[1]`."
        )
        raise ValueError(msg)

    assert_supported_dtype(A)
    assert_supported_dtype(B)
    ensure_compatible_dtype(A, B)

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        C_indices = np.zeros(1, dtype=idx_dtype)
        C_data = np.zeros(1, dtype=A.dtype)
        return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    n_threads = _resolve_parallel(n_threads)

    kwargs = {
        "nrows": A_nrows,
        "ncols": B_ncols,
        "A_data": A.data,
        "A_indptr": A.indptr if idx_dtype is None else A.indptr.astype(idx_dtype),
        "A_indices": A.indices if idx_dtype is None else A.indices.astype(idx_dtype),
        "B_data": B.data,
        "B_indptr": B.indptr if idx_dtype is None else B.indptr.astype(idx_dtype),
        "B_indices": B.indices if idx_dtype is None else B.indices.astype(idx_dtype),
        "n_threads": n_threads if n_threads and n_threads > 1 else None,
    }
    return csr_matrix(_core.sp_matmul(**kwargs), shape=(A_nrows, B_ncols))


def sp_matmul_topn(
    A: csr_matrix | csc_matrix | coo_matrix,
    B: csr_matrix | csc_matrix | coo_matrix,
    top_n: int,
    threshold: int | float | None = None,
    sort: bool = False,
    density: float | None = None,
    n_threads: int | None = None,
    idx_dtype: DTypeLike | None = None,
) -> csr_matrix:
    """Compute A * B whilst only storing the `top_n` elements per row.

    Args:
        A: LHS of the multiplication, the number of columns of A determines the orientation of B.
            `A` must have a {32, 64}bit {int, float} dtype that is of the same kind as `B`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        B: RHS of the multiplication, the number of rows of B must match the number of columns of A
            or the shape of B.T should match A.
            `B` must have a {32, 64}bit {int, float} dtype that is of the same kind as `A`.
            Note the matrix is converted (copied) to CSR format if a CSC or COO matrix.
        top_n: the number of results to retain.
        sort: return C in a format where the first non-zero element of each row is the largest value.
        threshold: only return values greater than the threshold.
        density: the expected density of the result considering `top_n`. The expected number of non-zero
            elements in C should be <= (`density` * `top_n` * `A.shape[0]`) otherwise memory has to be
            reallocated. Set this only if you have a strong expectation; being wrong incurs a
            reallocation penalty.
        n_threads: number of threads to use, `None` implies sequential processing, -1 will use all
            but one of the available cores.
        idx_dtype: dtype to use for the indices, defaults to 32-bit integers.

    Throws:
        TypeError: when A, B are not trivially convertable to a `CSR matrix`.

    Returns:
        C: result matrix.
    """
    n_threads: int = n_threads or 1
    if n_threads < 0:
        n_threads = _N_CORES
    density: float = density or 1.0
    idx_dtype = assert_idx_dtype(idx_dtype)

    if isinstance(A, csc_matrix) and isinstance(B, csc_matrix) and A.shape[0] == B.shape[1]:
        A = A.transpose()
        B = B.transpose()
    elif isinstance(A, (coo_matrix, csc_matrix)):
        A = A.tocsr(False)
    elif not isinstance(A, csr_matrix):
        msg = f"type of `A` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(A)}`"
        raise TypeError(msg)

    if not isinstance(B, (csr_matrix, coo_matrix, csc_matrix)):
        msg = f"type of `B` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(B)}`"
        raise TypeError(msg)

    A_nrows, A_ncols = A.shape
    B_nrows, B_ncols = B.shape

    if A_ncols == B_nrows:
        if isinstance(B, (coo_matrix, csc_matrix)):
            B = B.tocsr(False)
    elif A_ncols == B_ncols:
        B = B.transpose() if isinstance(B, csc_matrix) else B.transpose().tocsr(False)
        B_nrows, B_ncols = B.shape
    else:
        msg = (
            "Matrices `A` and `B` have incompatible shapes. `A.shape[1]` must be equal to `B.shape[0]` or `B.shape[1]`."
        )
        raise ValueError(msg)

    if B_ncols == top_n and (sort is False) and (threshold is None):
        return sp_matmul(A, B, n_threads)

    assert_supported_dtype(A)
    assert_supported_dtype(B)
    ensure_compatible_dtype(A, B)

    # guard against top_n larger than number of cols
    top_n = min(top_n, B_ncols)

    # handle threshold
    if threshold is not None:
        threshold = int(np.rint(threshold)) if np.issubdtype(A.data.dtype, np.integer) else float(threshold)

    # basic check. if A or B are all zeros matrix, return all zero matrix directly
    if A.indices.size == 0 or B.indices.size == 0:
        C_indptr = np.zeros(A_nrows + 1, dtype=idx_dtype)
        C_indices = np.zeros(1, dtype=idx_dtype)
        C_data = np.zeros(1, dtype=A.dtype)
        return csr_matrix((C_data, C_indices, C_indptr), shape=(A_nrows, B_ncols))

    n_threads = _resolve_parallel(n_threads)

    kwargs = {
        "top_n": top_n,
        "nrows": A_nrows,
        "ncols": B_ncols,
        "threshold": threshold,
        "density": density,
        "sort": sort,
        "n_threads": n_threads if n_threads and n_threads > 1 else None,
        "A_data": A.data,
        "A_indptr": A.indptr if idx_dtype is None else A.indptr.astype(idx_dtype),
        "A_indices": A.indices if idx_dtype is None else A.indices.astype(idx_dtype),
        "B_data": B.data,
        "B_indptr": B.indptr if idx_dtype is None else B.indptr.astype(idx_dtype),
        "B_indices": B.indices if idx_dtype is None else B.indices.astype(idx_dtype),
    }
    return csr_matrix(_core.sp_matmul_topn(**kwargs), shape=(A_nrows, B_ncols))


def zip_sp_matmul_topn(top_n: int, C_mats: list[csr_matrix]) -> csr_matrix:
    """Compute zip-matrix C = zip_i C_i = zip_i A * B_i = A * B whilst keeping only the `top_n` elements.

    Combine the sub-matrices together and keep only the `top_n` elements per row.

    Args:
        top_n: the number of results to retain; should be smaller or equal to top_n used to obtain C_mats.
        C_mats: a list with each C_i sub-matrix, with format csr_matrix.

    Returns:
        C: zipped result matrix.

    Raises:
        TypeError: when not all elements of `C_mats` is a csr_matrix or trivially convertable.
        ValueError: when not all elements of `C_mats` has the same number of rows.
    """
    _nrows = []
    ncols = []
    data = []
    indptr = []
    indices = []
    for C in C_mats:
        if isinstance(C, (coo_matrix, csc_matrix)):
            C = C.tocsr(False)
        elif not isinstance(C, csr_matrix):
            msg = f"type of `C` must be one of `csr_matrix`, `csc_matrix` or `csr_matrix`, got `{type(C)}`"
            raise TypeError(msg)

        nrows, c_nc = C.shape
        _nrows.append(nrows)
        ncols.append(c_nc)
        data.append(C.data)
        indptr.append(C.indptr)
        indices.append(C.indices)

    ncols = np.asarray(ncols, int)
    total_cols = int(ncols.sum())
    if not np.all(np.diff(_nrows) == 0):
        msg = "Each `C` in `C_mats` should have the same number of rows."
        raise ValueError(msg)

    return csr_matrix(
        _core.zip_sp_matmul_topn(
            top_n=top_n,
            Z_max_nnz=nrows * top_n,
            nrows=nrows,
            B_ncols=ncols.tolist(),
            data=data,
            indptr=indptr,
            indices=indices,
        ),
        shape=(nrows, total_cols),
    )


def _resolve_parallel(n_threads: int) -> int:
    """If the user asked for >1 threads but the extension lacks parallel support, warn and clamp to 1."""
    if n_threads > 1 and not _core._rust_parallel_enabled:
        warnings.warn(
            "sp_matmul_rs: extension was compiled without parallelism (rayon) support, ignoring ``n_threads``",
            stacklevel=1,
        )
        return 1
    return n_threads
