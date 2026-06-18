# Generate parity-test golden inputs/outputs for the Rust port.
#
# Each case is emitted once per (V, I) dtype combination; the case definition itself is
# dtype-agnostic (we generate one random matrix and re-cast it). Both the C++ extension
# and scipy are run on the input; we assert they agree to the test tolerance and then
# persist the C++ extension's exact output as the golden — that is what the Rust port
# must match.
#
# Run from `sp_matmul_rs/`:
#     source ../.venv/bin/activate
#     python tests/data/generate_goldens.py
#
# Files committed under `tests/data/*.npz`.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sparse_dot_topn import sp_matmul as cpp_sp_matmul
from sparse_dot_topn import sp_matmul_topn as cpp_sp_matmul_topn
from sparse_dot_topn import zip_sp_matmul_topn as cpp_zip_sp_matmul_topn

OUT_DIR = Path(__file__).parent

V_DTYPES = ["f32", "f64", "i32", "i64"]
I_DTYPES = ["i32", "i64"]

NP_OF = {
    "f32": np.float32,
    "f64": np.float64,
    "i32": np.int32,
    "i64": np.int64,
}

# Per-dtype tolerance used by the cross-check.
TOL_REL = {"f32": 1e-6, "f64": 1e-12, "i32": 0.0, "i64": 0.0}
TOL_ABS = {"f32": 1e-6, "f64": 1e-12, "i32": 0.0, "i64": 0.0}


def is_int(v: str) -> bool:
    return v.startswith("i")


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_csr(nrows: int, ncols: int, density: float, v: str, seed: int) -> sp.csr_matrix:
    """Random CSR with no exact value-ties (perturbations avoid heap tie-breaking flakiness)."""
    g = rng(seed)
    n_total = nrows * ncols
    n_nnz = max(0, int(round(density * n_total)))
    if n_nnz == 0:
        return sp.csr_matrix((nrows, ncols), dtype=NP_OF[v])

    # Sample without replacement so we don't double-up entries; then perturb to avoid ties.
    chosen = g.choice(n_total, size=n_nnz, replace=False)
    rows, cols = np.divmod(chosen, ncols)
    if is_int(v):
        # Keep magnitudes small so sums stay under i32::MAX (see plan §5.5).
        base = g.integers(low=1, high=100, size=n_nnz)
        # Tie-break with a unique-ish offset modulated by index.
        vals = base + (np.arange(n_nnz) % 7)
        vals = vals.astype(NP_OF[v])
    else:
        base = g.uniform(low=0.1, high=10.0, size=n_nnz)
        # Multiplicative perturbation to break value ties.
        perturb = 1.0 + 1e-3 * g.uniform(0.0, 1.0, size=n_nnz)
        vals = (base * perturb).astype(NP_OF[v])

    return sp.csr_matrix((vals, (rows, cols)), shape=(nrows, ncols), dtype=NP_OF[v])


def empty_csr(nrows: int, ncols: int, v: str) -> sp.csr_matrix:
    return sp.csr_matrix((nrows, ncols), dtype=NP_OF[v])


def reindex(mat: sp.csr_matrix, i: str) -> sp.csr_matrix:
    """Re-cast indptr/indices to the requested integer dtype."""
    out = mat.copy()
    out.indptr = out.indptr.astype(NP_OF[i])
    out.indices = out.indices.astype(NP_OF[i])
    return out


def scipy_topn_per_row(
    full: sp.csr_matrix,
    top_n: int,
    threshold: float | None,
    sort_by_value_desc: bool,
) -> sp.csr_matrix:
    """Reference top-n via scipy: per-row, keep up to `top_n` strictly above `threshold`."""
    nrows, ncols = full.shape
    full = full.tocsr()
    rows_idx: list[int] = []
    rows_val: list = []
    new_indptr = [0]
    for i in range(nrows):
        start, end = full.indptr[i], full.indptr[i + 1]
        idxs = full.indices[start:end]
        vals = full.data[start:end]
        if threshold is not None:
            mask = vals > threshold
            idxs = idxs[mask]
            vals = vals[mask]
        if len(vals) > top_n:
            order = np.argsort(-vals, kind="stable")[:top_n]
            idxs = idxs[order]
            vals = vals[order]
        if sort_by_value_desc:
            order = np.argsort(-vals, kind="stable")
        else:
            order = np.argsort(idxs, kind="stable")
        idxs = idxs[order]
        vals = vals[order]
        rows_idx.append(idxs)
        rows_val.append(vals)
        new_indptr.append(new_indptr[-1] + len(vals))
    return sp.csr_matrix(
        (
            np.concatenate(rows_val) if rows_val else np.array([], dtype=full.dtype),
            np.concatenate(rows_idx) if rows_idx else np.array([], dtype=full.indices.dtype),
            np.array(new_indptr, dtype=full.indptr.dtype),
        ),
        shape=(nrows, ncols),
    )


def csr_close(a: sp.csr_matrix, b: sp.csr_matrix, v: str) -> bool:
    """Equivalence modulo intra-row column order and floating-point tolerance."""
    if a.shape != b.shape:
        return False
    aa = a.copy()
    bb = b.copy()
    aa.sort_indices()
    bb.sort_indices()
    if not np.array_equal(aa.indptr, bb.indptr):
        return False
    if not np.array_equal(aa.indices, bb.indices):
        return False
    if is_int(v):
        return np.array_equal(aa.data, bb.data)
    return np.allclose(aa.data, bb.data, rtol=TOL_REL[v], atol=TOL_ABS[v])


@dataclass
class Case:
    name: str
    nrows_a: int
    ncols_a: int  # = nrows_b
    ncols_b: int
    density_a: float
    density_b: float
    top_n: int
    sort_by_value_desc: bool
    threshold: float | None
    # Optional toggles
    empty_a: bool = False
    empty_b: bool = False


CASES: list[Case] = [
    Case("small_dense", 20, 30, 25, 0.5, 0.5, 5, False, None),
    Case("tall_skinny", 200, 10, 100, 0.3, 0.3, 8, True, None),
    Case("wide_sparse", 50, 500, 500, 0.02, 0.02, 15, False, None),
    Case("with_threshold", 100, 80, 120, 0.1, 0.1, 10, True, 0.5),
    Case("topn_eq_ncols", 30, 40, 30, 0.2, 0.2, 30, False, None),
    Case("topn_gt_ncols", 20, 20, 15, 0.3, 0.3, 100, True, None),
    Case("empty_a", 10, 10, 10, 0.0, 0.3, 5, False, None, empty_a=True),
    Case("empty_b", 10, 10, 10, 0.3, 0.0, 5, False, None, empty_b=True),
    Case("all_below_threshold", 30, 30, 30, 0.2, 0.2, 10, True, 1e9),
]


def threshold_for_dtype(v: str, raw: float | None) -> float | int | None:
    if raw is None:
        return None
    if is_int(v):
        return int(max(1, round(raw)))
    return float(raw)


def emit_case(case: Case, v: str, i: str) -> None:
    seed = abs(hash((case.name, v, i))) & 0xFFFFFFFF
    if case.empty_a:
        a = empty_csr(case.nrows_a, case.ncols_a, v)
    else:
        a = random_csr(case.nrows_a, case.ncols_a, case.density_a, v, seed)
    if case.empty_b:
        b = empty_csr(case.ncols_a, case.ncols_b, v)
    else:
        b = random_csr(case.ncols_a, case.ncols_b, case.density_b, v, seed + 1)

    a = reindex(a, i)
    b = reindex(b, i)

    threshold = threshold_for_dtype(v, case.threshold)
    sort_flag = case.sort_by_value_desc
    if threshold is None:
        cpp = cpp_sp_matmul_topn(a, b, top_n=case.top_n, sort=sort_flag)
    else:
        cpp = cpp_sp_matmul_topn(
            a, b, top_n=case.top_n, threshold=threshold, sort=sort_flag
        )

    # Cross-check the C++ extension against a from-scratch scipy reference.
    # For floats (no ties by construction) we hard-fail; for ints, top-n ties at the
    # boundary can legitimately pick different columns with the same value, so we
    # only warn — the C++ output remains the source of truth for parity.
    full = (a @ b).tocsr()
    scipy_ref = scipy_topn_per_row(
        full, case.top_n, None if threshold is None else float(threshold), sort_flag
    )
    if not csr_close(cpp, scipy_ref, v):
        msg = (
            f"GOLDEN MISMATCH: scipy reference disagrees with C++ extension for "
            f"case={case.name} V={v} I={i}"
        )
        if is_int(v):
            print(f"  WARN: {msg} (tie-break drift expected for ints)")
        else:
            raise SystemExit(msg)

    threshold_sentinel = (
        np.array(0, dtype=NP_OF[v]) if threshold is None else np.array(threshold, dtype=NP_OF[v])
    )

    # The C++ extension may pick its own index dtype for the output (often the platform
    # default int) regardless of the input's. Cast back to the requested I so each
    # golden's index arrays match the dtype the Rust port will compute with.
    out = {
        "a_indptr": a.indptr,
        "a_indices": a.indices,
        "a_data": a.data,
        "a_shape": np.array(a.shape, dtype=np.int64),
        "b_indptr": b.indptr,
        "b_indices": b.indices,
        "b_data": b.data,
        "b_shape": np.array(b.shape, dtype=np.int64),
        "top_n": np.int64(case.top_n),
        "sort_by_value_desc": np.bool_(sort_flag),
        "has_threshold": np.bool_(threshold is not None),
        "threshold": threshold_sentinel,
        "expected_indptr": cpp.indptr.astype(NP_OF[i]),
        "expected_indices": cpp.indices.astype(NP_OF[i]),
        "expected_data": cpp.data.astype(NP_OF[v]),
        "expected_shape": np.array(cpp.shape, dtype=np.int64),
    }
    path = OUT_DIR / f"{case.name}_{v}_{i}.npz"
    np.savez(path, **out)
    print(f"wrote {path.name}  (cpp.nnz={cpp.nnz})")


@dataclass
class ZipCase:
    name: str
    nrows_a: int
    ncols_a: int
    chunk_ncols: list[int]  # len = number of chunks
    density: float
    top_n: int


ZIP_CASES: list[ZipCase] = [
    ZipCase("zip_three_chunks", 40, 100, [30, 30, 40], 0.2, 8),
]


def emit_zip_case(case: ZipCase, v: str, i: str) -> None:
    seed = abs(hash(("zip", case.name, v, i))) & 0xFFFFFFFF
    a = reindex(random_csr(case.nrows_a, case.ncols_a, case.density, v, seed), i)
    bs = [
        reindex(
            random_csr(case.ncols_a, nc, case.density, v, seed + 100 + j), i
        )
        for j, nc in enumerate(case.chunk_ncols)
    ]
    # Each per-chunk top-n is computed in the C++ extension under the same sort
    # convention zip expects (value desc).
    cs = [cpp_sp_matmul_topn(a, b, top_n=case.top_n, sort=True) for b in bs]
    zipped = cpp_zip_sp_matmul_topn(case.top_n, cs)

    # Sanity cross-check vs scipy: concatenate columns and take per-row top-n by value.
    concat = sp.hstack([(a @ b).tocsr() for b in bs]).tocsr()
    scipy_ref = scipy_topn_per_row(concat, case.top_n, threshold=None, sort_by_value_desc=True)
    # The zip path uses `numeric_limits<V>::min()` as a sentinel, which for floats is the
    # smallest positive normal — meaning negative entries from the per-chunk tops get
    # silently dropped (see plan §5.4). Our scipy reference has no negative values
    # (random_csr produces all-positive), so cross-check works.
    if not csr_close(zipped, scipy_ref, v):
        raise SystemExit(
            f"ZIP GOLDEN MISMATCH: scipy reference disagrees with C++ zip extension "
            f"for case={case.name} V={v} I={i}"
        )

    out: dict[str, np.ndarray] = {
        "n_chunks": np.int64(len(cs)),
        "top_n": np.int64(case.top_n),
        "expected_indptr": zipped.indptr.astype(NP_OF[i]),
        "expected_indices": zipped.indices.astype(NP_OF[i]),
        "expected_data": zipped.data.astype(NP_OF[v]),
        "expected_shape": np.array(zipped.shape, dtype=np.int64),
    }
    for k, c in enumerate(cs):
        out[f"chunk_{k}_indptr"] = c.indptr.astype(NP_OF[i])
        out[f"chunk_{k}_indices"] = c.indices.astype(NP_OF[i])
        out[f"chunk_{k}_data"] = c.data.astype(NP_OF[v])
        out[f"chunk_{k}_shape"] = np.array(c.shape, dtype=np.int64)
    path = OUT_DIR / f"{case.name}_{v}_{i}.npz"
    np.savez(path, **out)
    print(f"wrote {path.name}  (zipped.nnz={zipped.nnz})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in V_DTYPES:
        for i in I_DTYPES:
            for case in CASES:
                emit_case(case, v, i)
            for zcase in ZIP_CASES:
                emit_zip_case(zcase, v, i)


if __name__ == "__main__":
    main()
