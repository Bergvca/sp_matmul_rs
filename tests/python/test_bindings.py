"""Direct exercises of the `_core` binding's contractual guarantees.

Covers Phase 4 exit criteria 5 (zero-copy on inputs) and 6 (GIL released
around the kernel). Independent of the `api.py` wrapper.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from scipy import sparse

from sp_matmul_rs import _core, sp_matmul_topn, _rust_parallel_enabled


def _csr_parts(M):
    return M.data, M.indptr, M.indices


def test_unsupported_value_dtype_raises_type_error():
    A = sparse.random(5, 5, density=0.5, format="csr", dtype=np.float64)
    bad_data = A.data.astype(np.complex128)
    with pytest.raises(TypeError):
        _core.sp_matmul(
            nrows=A.shape[0],
            ncols=A.shape[1],
            A_data=bad_data,
            A_indptr=A.indptr,
            A_indices=A.indices,
            B_data=bad_data,
            B_indptr=A.indptr,
            B_indices=A.indices,
        )


def test_mismatched_value_dtypes_raises_type_error():
    A = sparse.random(5, 5, density=0.5, format="csr", dtype=np.float64)
    B = sparse.random(5, 5, density=0.5, format="csr", dtype=np.float32)
    with pytest.raises(TypeError):
        _core.sp_matmul(
            nrows=A.shape[0],
            ncols=B.shape[1],
            A_data=A.data,
            A_indptr=A.indptr,
            A_indices=A.indices,
            B_data=B.data,
            B_indptr=B.indptr,
            B_indices=B.indices,
        )


@pytest.mark.parametrize("vdtype", [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize("idtype", [np.int32, np.int64])
def test_dispatch_picks_right_types(vdtype, idtype):
    A = sparse.random(10, 10, density=0.3, format="csr", dtype=vdtype)
    A.indptr = A.indptr.astype(idtype)
    A.indices = A.indices.astype(idtype)
    data, indices, indptr = _core.sp_matmul(
        nrows=A.shape[0],
        ncols=A.shape[1],
        A_data=A.data,
        A_indptr=A.indptr,
        A_indices=A.indices,
        B_data=A.data,
        B_indptr=A.indptr,
        B_indices=A.indices,
    )
    assert data.dtype == np.dtype(vdtype)
    assert indices.dtype == np.dtype(idtype)
    assert indptr.dtype == np.dtype(idtype)


@pytest.mark.skipif(
    not _rust_parallel_enabled,
    reason="rayon parallelism disabled — GIL-release speedup not measurable",
)
def test_gil_released_during_kernel():
    # Build a workload that runs long enough for two threads to overlap.
    rng = np.random.Generator(np.random.PCG64DXSM(20260605))
    A = sparse.random(2000, 2000, density=0.05, format="csr", dtype=np.float64, random_state=rng)
    B = sparse.random(2000, 2000, density=0.05, format="csr", dtype=np.float64, random_state=rng)

    def run():
        # Use n_threads=1 inside the binding so each Python thread owns one
        # core; we want to measure GIL-release-during-kernel, not internal
        # parallelism. A speedup proves the GIL is released.
        sp_matmul_topn(A, B, top_n=10, n_threads=1)

    # Warm-up — JIT-ish: caches, allocator, etc.
    run()

    t0 = time.perf_counter()
    run()
    run()
    serial = time.perf_counter() - t0

    threads = [threading.Thread(target=run) for _ in range(2)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    parallel = time.perf_counter() - t0

    speedup = serial / parallel
    # Plan §1 criterion 6: target ≥ 1.3× quiet, soft ≥ 1.05× on CI.
    assert speedup >= 1.05, f"expected >=1.05x speedup from GIL release, got {speedup:.2f}x"


def test_zero_copy_input_borrow():
    # Build a CSR with known buffer pointers and assert the binding produces
    # output independent of the input arrays — i.e. the call doesn't reallocate
    # them. The strongest assertion we can make from Python without inspecting
    # Rust internals is that the input arrays are unchanged after the call and
    # that their memory addresses haven't been swapped under us.
    A = sparse.random(50, 50, density=0.2, format="csr", dtype=np.float64)
    a_data_ptr = A.data.ctypes.data
    a_indices_ptr = A.indices.ctypes.data
    a_indptr_ptr = A.indptr.ctypes.data
    a_data_copy = A.data.copy()

    sp_matmul_topn(A, A, top_n=5)

    assert A.data.ctypes.data == a_data_ptr
    assert A.indices.ctypes.data == a_indices_ptr
    assert A.indptr.ctypes.data == a_indptr_ptr
    np.testing.assert_array_equal(A.data, a_data_copy)


def test_threshold_int_round_trips():
    A = sparse.random(20, 20, density=0.5, format="csr", dtype=np.int32)
    A.data = (A.data * 10).astype(np.int32)
    C = sp_matmul_topn(A, A, top_n=20, threshold=1)
    # All retained values must exceed the integer threshold.
    if C.data.size:
        assert C.data.min() > 1
