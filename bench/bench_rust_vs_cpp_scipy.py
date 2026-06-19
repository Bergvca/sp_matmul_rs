# Copyright (c) 2026 Chris van den Berg
"""Three-way benchmark: scipy.sparse vs sparse_dot_topn (C++) vs sp_matmul_rs (Rust).

Mirrors the C++ project's ``bench/bench_scipy_csr.py`` shape and parameter grid,
but builds the TF-IDF input from the EDGAR company-name sqlite snapshot bundled
in ``../sparse_dot_topn/dev_tests/database.sqlite`` (the original Kaggle CSV is
not in-tree).

The grid is the same as the C++ bench README:
    sp_matmul:       n_threads in {1, 2, 4, 8}
    sp_matmul_topn:  top_n in {10, 20, 30, 100, 1000}  x  n_threads in {1, 2, 4, 8}

Each cell is the **min** of ``REPEATS`` calls — same convention as richbench's
"Min" column. We also report ``mean`` for sanity. Outputs a Markdown table to
stdout and (optionally) writes it to a file via ``--out``.

Usage:
    python bench_rust_vs_cpp_scipy.py                # N_ROWS=20_000, repeats=5
    python bench_rust_vs_cpp_scipy.py --rows 100000  # larger sampled shape
    python bench_rust_vs_cpp_scipy.py --rows 0       # full 663k EDGAR corpus,
                                                     # self-similarity (A @ A.T)
    python bench_rust_vs_cpp_scipy.py --repeats 10 --out results.md
    python bench_rust_vs_cpp_scipy.py --only scipy,cpp   # subset

The TF-IDF matrices are cached as .npz next to this file so repeat invocations
skip the (slow) vectoriser fit.
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy
import sparse_dot_topn as sdtn
import sp_matmul_rs as sdtn_rs

HERE = Path(__file__).resolve().parent
DEFAULT_SQLITE = HERE.parent.parent / "sparse_dot_topn" / "dev_tests" / "database.sqlite"


def ngrams(string: str, n: int = 3) -> list[str]:
    string = re.sub(r"[,-./]|\sBD", r"", string)
    return ["".join(g) for g in zip(*[string[i:] for i in range(n)])]


def load_or_build_tfidf(sqlite_path: Path, n_rows: int) -> tuple[csr_matrix, csr_matrix]:
    """Build/load TF-IDF matrices A and B.

    ``n_rows <= 0`` uses the **entire** EDGAR corpus on both sides:
    ``A`` is the full 663 000-row TF-IDF matrix and ``B`` is its transpose,
    giving ``A @ B = A @ A.T`` — the self-similarity shape ``string_grouper``
    runs in practice. ``n_rows > 0`` keeps the sampled / two-halves layout
    used by the C++ bench README.
    """
    tag = "full" if n_rows <= 0 else str(n_rows)
    path_A = HERE / f"tfidf_A_{tag}.npz"
    path_B = HERE / f"tfidf_B_{tag}.npz"
    if path_A.exists() and path_B.exists():
        return load_npz(path_A), load_npz(path_B)

    if not sqlite_path.exists():
        sys.exit(f"sqlite source not found: {sqlite_path}")

    print(f"building TF-IDF from {sqlite_path} (n_rows={tag}) ...", flush=True)
    con = sqlite3.connect(sqlite_path)
    names = pd.read_sql_query("SELECT name FROM companies", con)["name"]
    con.close()
    names = names.str.lower().sample(frac=1.0, random_state=0).reset_index(drop=True)

    if n_rows <= 0:
        # Self-similarity: A is the full corpus, B is its transpose.
        vec = TfidfVectorizer(min_df=1).fit(names)
        A_out = vec.transform(names).tocsr()
        B_out = A_out.transpose().tocsr()
    else:
        # C++ bench layout: split the corpus in half, sample n_rows from each.
        hn = names.size // 2
        vec = TfidfVectorizer(min_df=1).fit(names)
        A = vec.transform(names[:hn])
        B = vec.transform(names[hn:])
        rng = np.random.default_rng(0)
        A_rows = np.sort(rng.choice(np.arange(hn), size=n_rows, replace=False))
        B_rows = np.sort(rng.choice(np.arange(hn), size=n_rows, replace=False))
        A_out = A[A_rows, :]
        B_out = B[B_rows, :].transpose().tocsr()

    save_npz(path_A, A_out)
    save_npz(path_B, B_out)
    return A_out, B_out


def time_call(fn, repeats: int) -> tuple[float, float, float]:
    """Return (min, mean, max) wall time in seconds across ``repeats`` calls."""
    # one untimed warm-up — avoids first-call effects (numpy reshape caches, etc.)
    fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), sum(ts) / len(ts), max(ts)


def fmt_speedup(baseline: float, candidate: float) -> str:
    """``richbench``-style speedup label."""
    if candidate <= baseline:
        return f"{baseline / candidate:.1f}x"
    return f"-{candidate / baseline:.1f}x"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rows",
        type=int,
        default=20_000,
        help="rows per side of the TF-IDF input; pass 0 to use the entire EDGAR corpus (A @ A.T)",
    )
    p.add_argument("--repeats", type=int, default=5, help="timed repeats per config")
    p.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE, help="EDGAR company sqlite")
    p.add_argument("--only", default="scipy,cpp,rust", help="comma-separated subset of {scipy,cpp,rust}")
    p.add_argument("--threads", default="1,2,4,8", help="comma-separated n_threads grid")
    p.add_argument("--top-ns", default="10,20,30,100,1000", help="comma-separated top_n grid")
    p.add_argument(
        "--topn-only",
        action="store_true",
        help="skip the 'sp_matmul (no top-n)' rows (avoids OOM on very large inputs)",
    )
    p.add_argument("--out", type=Path, default=None, help="write Markdown table to this file too")
    args = p.parse_args()

    engines = [e.strip() for e in args.only.split(",") if e.strip()]
    threads = [int(t) for t in args.threads.split(",")]
    top_ns = [int(t) for t in args.top_ns.split(",")]

    A, B = load_or_build_tfidf(args.sqlite, args.rows)
    print(
        f"A.shape={A.shape}  A.nnz={A.nnz}\n"
        f"B.shape={B.shape}  B.nnz={B.nnz}\n"
        f"scipy={scipy.__version__}  sparse_dot_topn={sdtn.__version__} "
        f"(openmp={sdtn._has_openmp_support})  sp_matmul_rs.parallel="
        f"{sdtn_rs.kernel_info()['parallel_enabled']}",
        flush=True,
    )

    # ---- assemble runs -----------------------------------------------------
    # Each row: (label, top_n_str, threads_str, dict[engine] -> callable)
    runs: list[tuple[str, str, str, dict]] = []
    if not args.topn_only:
        for nt in threads:
            runs.append(
                (
                    "sp_matmul (no top-n)",
                    "",
                    str(nt),
                    {
                        "scipy": (lambda A=A, B=B: A.dot(B)),
                        "cpp": (lambda A=A, B=B, nt=nt: sdtn.sp_matmul(A, B, n_threads=nt)),
                        "rust": (lambda A=A, B=B, nt=nt: sdtn_rs.sp_matmul(A, B, n_threads=nt)),
                    },
                )
            )
    for top_n in top_ns:
        for nt in threads:
            runs.append(
                (
                    "sp_matmul_topn",
                    str(top_n),
                    str(nt),
                    {
                        # scipy has no top-n; we benchmark its full multiply as the baseline.
                        "scipy": (lambda A=A, B=B: A.dot(B)),
                        "cpp": (lambda A=A, B=B, n=top_n, nt=nt: sdtn.sp_matmul_topn(A, B, n, n_threads=nt)),
                        "rust": (
                            lambda A=A, B=B, n=top_n, nt=nt: sdtn_rs.sp_matmul_topn(A, B, n, n_threads=nt)
                        ),
                    },
                )
            )

    # ---- run --------------------------------------------------------------
    rows = []
    for label, top_n_str, nt_str, fns in runs:
        print(f"  -> {label:<22} top_n={top_n_str or '-':<5} n_threads={nt_str}", flush=True)
        rec: dict[str, tuple[float, float, float] | None] = {"scipy": None, "cpp": None, "rust": None}
        for eng in engines:
            rec[eng] = time_call(fns[eng], args.repeats)
        rows.append((label, top_n_str, nt_str, rec))

    # ---- render -----------------------------------------------------------
    header = (
        "| Benchmark            | top_n | n_threads | "
        "Scipy (s) | C++ min (s) | C++ vs Scipy | Rust min (s) | Rust vs Scipy | Rust vs C++ |"
    )
    sep = (
        "| :------------------- | :---: | :-------: | "
        "--------: | ----------: | -----------: | -----------: | ------------: | ----------: |"
    )
    md_lines = [header, sep]
    for label, top_n_str, nt_str, rec in rows:
        sc = rec.get("scipy")
        cp = rec.get("cpp")
        ru = rec.get("rust")

        def cell(t):
            return f"{t[0]:.3f}" if t else "—"

        sc_speed_cpp = fmt_speedup(sc[0], cp[0]) if (sc and cp) else "—"
        sc_speed_rust = fmt_speedup(sc[0], ru[0]) if (sc and ru) else "—"
        cpp_speed_rust = fmt_speedup(cp[0], ru[0]) if (cp and ru) else "—"
        md_lines.append(
            f"| {label:<20} | {top_n_str or '-':^5} | {nt_str:^9} | "
            f"{cell(sc):>9} | {cell(cp):>11} | {sc_speed_cpp:>12} | "
            f"{cell(ru):>12} | {sc_speed_rust:>13} | {cpp_speed_rust:>11} |"
        )

    table = "\n".join(md_lines)
    print()
    print(table)

    if args.out is not None:
        args.out.write_text(table + "\n")
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
