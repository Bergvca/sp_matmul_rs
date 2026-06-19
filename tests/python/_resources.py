from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

if TYPE_CHECKING:
    from numpy.types import NDArray


def _get_topn_elements(x: NDArray, n: int):
    return x[np.sort(np.argsort(x)[::-1][:n])]


def _assert_array_equal(A, B, rtol=1e-5, atol=1e-8):
    if np.issubdtype(A.dtype, np.integer):
        assert_array_equal(A, B)
    else:
        assert_allclose(A, B, rtol=rtol, atol=atol)


def _assert_smat_equal(A, B, rtol=1e-5, atol=1e-8):
    # Sparse output ordering within a row is not part of the contract — scipy
    # 1.17's csr_matmul leaves indices unsorted, and our kernel may emit
    # descending column order from its linked-list drain. Canonicalize both
    # before structural comparison.
    A = A.copy()
    B = B.copy()
    A.sort_indices()
    B.sort_indices()
    _assert_array_equal(A.data, B.data, rtol, atol)
    assert_array_equal(A.indptr, B.indptr)
    assert_array_equal(A.indices, B.indices)
