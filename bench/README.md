# Benchmarks: sp_matmul_rs vs sparse_dot_topn (C++) vs Scipy

A three-way comparison on the same input as the upstream C++ project's
[`bench_scipy_csr.py`](https://github.com/ing-bank/sparse_dot_topn/tree/master/bench):
a word-level TF-IDF matrix over the EDGAR company-name list
([Kaggle: dattapiy/sec-edgar-companies-list](https://www.kaggle.com/datasets/dattapiy/sec-edgar-companies-list)).

The benchmark answers two questions:

1. How does the Rust port (`sp_matmul_rs`) compare to the legacy C++
   extension (`sparse_dot_topn`) on the workload `sparse_dot_topn` was tuned
   for?
2. How does each library compare to a plain `scipy.sparse` multiplication
   baseline (no top-n, single-threaded)?

The script (`bench_rust_vs_cpp_scipy.py`) covers the same `{top_n, n_threads}`
grid as the C++ bench README so the numbers are directly comparable to the
historical Apple M2 Pro / Intel i9 tables.

## Dependencies

```shell
uv pip install scipy scikit-learn pandas psutil
uv pip install sparse_dot_topn        # the C++ contender
uv pip install sp_matmul_rs           # the Rust contender (or maturin develop --release ...)
```

If you build `sparse_dot_topn` from source on macOS, enable OpenMP — otherwise
all `n_threads > 1` cells silently run single-threaded:

```shell
brew install libomp
CMAKE_ARGS="-DSDTN_ENABLE_OPENMP=ON -DOpenMP_ROOT=$(brew --prefix)/opt/libomp" \
  pip install sparse_dot_topn --no-build-isolation
python -c "import sparse_dot_topn; print(sparse_dot_topn._has_openmp_support)"
```

For `sp_matmul_rs`, default features include `rayon`; verify with
`python -c "import sp_matmul_rs; print(sp_matmul_rs.kernel_info()['parallel_enabled'])"`.

## Data

The original C++ bench reads `sec__edgar_company_info.csv` from
[Kaggle: dattapiy/sec-edgar-companies-list](https://www.kaggle.com/datasets/dattapiy/sec-edgar-companies-list).
This bench reuses the **same** company-name corpus, but loads it from
`../../sparse_dot_topn/dev_tests/database.sqlite` (663 000 names, the
`companies` table). The TF-IDF vectoriser (`sklearn.feature_extraction.text.TfidfVectorizer(min_df=1)`)
runs over the full corpus; we then sample `N_ROWS` rows from each half with a
fixed RNG seed and cache them as `tfidf_A_{N_ROWS}.npz` / `tfidf_B_{N_ROWS}.npz`
next to the script.

The default `--rows 20000` matches the C++ bench README. The cached input shape
is `A: (20 000, 193 190)` and `B: (193 190, 20 000)`, so results are directly
comparable to the M2 Pro table in the upstream README.

## Running

```shell
# Default sweep — N_ROWS=20_000, repeats=5, same grid as the C++ README.
python bench_rust_vs_cpp_scipy.py --out results_20000.md

# Subset (drop libraries, prune the grid).
python bench_rust_vs_cpp_scipy.py --only cpp,rust --threads 1,8 --top-ns 10,100

# Bigger shape.
python bench_rust_vs_cpp_scipy.py --rows 100000 --repeats 5
```

Each cell is the **min** of `--repeats` timed calls (one untimed warm-up
first) — the same convention as richbench's "Min" column. The first build runs
the TF-IDF vectoriser, which takes ~30 s; later runs hit the cache.

`Scipy` columns measure `A.dot(B)` (no top-n, no threading). The "vs Scipy"
columns are positive when the candidate is faster than Scipy and negative
when slower, matching richbench's sign convention.

## Results

### Apple Macbook Pro | M5 Pro | 48 GB RAM | macOS 26.4

`scipy 1.17.1`, `sparse_dot_topn 1.2.0` (OpenMP via Homebrew libomp 22.1.8),
`sp_matmul_rs 0.1.0` (rayon enabled). Shape
`A: (20 000, 193 190) × B: (193 190, 20 000)`, `repeats=5`. Times are the
per-call **min** in seconds.

| Benchmark            | top_n | n_threads | Scipy (s) | C++ min (s) | C++ vs Scipy | Rust min (s) | Rust vs Scipy | Rust vs C++ |
| :------------------- | :---: | :-------: | --------: | ----------: | -----------: | -----------: | ------------: | ----------: |
| sp_matmul (no top-n) |   -   |     1     |     0.110 |       0.089 |         1.2x |        0.125 |         -1.1x |       -1.4x |
| sp_matmul (no top-n) |   -   |     2     |     0.110 |       0.040 |         2.8x |        0.126 |         -1.1x |       -3.1x |
| sp_matmul (no top-n) |   -   |     4     |     0.110 |       0.022 |         5.0x |        0.127 |         -1.2x |       -5.7x |
| sp_matmul (no top-n) |   -   |     8     |     0.110 |       0.015 |         7.4x |        0.126 |         -1.1x |       -8.4x |
| sp_matmul_topn       |  10   |     1     |     0.110 |       0.102 |         1.1x |        0.083 |          1.3x |        1.2x |
| sp_matmul_topn       |  10   |     2     |     0.111 |       0.041 |         2.7x |        0.044 |          2.5x |       -1.1x |
| sp_matmul_topn       |  10   |     4     |     0.110 |       0.022 |         5.0x |        0.024 |          4.6x |       -1.1x |
| sp_matmul_topn       |  10   |     8     |     0.107 |       0.014 |         7.5x |        0.015 |          6.9x |       -1.1x |
| sp_matmul_topn       |  20   |     1     |     0.108 |       0.115 |        -1.1x |        0.105 |          1.0x |        1.1x |
| sp_matmul_topn       |  20   |     2     |     0.109 |       0.049 |         2.2x |        0.058 |          1.9x |       -1.2x |
| sp_matmul_topn       |  20   |     4     |     0.108 |       0.026 |         4.2x |        0.031 |          3.5x |       -1.2x |
| sp_matmul_topn       |  20   |     8     |     0.107 |       0.016 |         6.6x |        0.020 |          5.5x |       -1.2x |
| sp_matmul_topn       |  30   |     1     |     0.108 |       0.124 |        -1.2x |        0.119 |         -1.1x |        1.0x |
| sp_matmul_topn       |  30   |     2     |     0.109 |       0.054 |         2.0x |        0.065 |          1.7x |       -1.2x |
| sp_matmul_topn       |  30   |     4     |     0.108 |       0.028 |         3.8x |        0.034 |          3.2x |       -1.2x |
| sp_matmul_topn       |  30   |     8     |     0.109 |       0.017 |         6.3x |        0.021 |          5.3x |       -1.2x |
| sp_matmul_topn       |  100  |     1     |     0.109 |       0.215 |        -2.0x |        0.257 |         -2.4x |       -1.2x |
| sp_matmul_topn       |  100  |     2     |     0.107 |       0.102 |         1.0x |        0.136 |         -1.3x |       -1.3x |
| sp_matmul_topn       |  100  |     4     |     0.107 |       0.053 |         2.0x |        0.071 |          1.5x |       -1.3x |
| sp_matmul_topn       |  100  |     8     |     0.107 |       0.031 |         3.5x |        0.041 |          2.6x |       -1.3x |
| sp_matmul_topn       | 1000  |     1     |     0.107 |       0.739 |        -6.9x |        1.132 |        -10.6x |       -1.5x |
| sp_matmul_topn       | 1000  |     2     |     0.108 |       0.388 |        -3.6x |        0.604 |         -5.6x |       -1.6x |
| sp_matmul_topn       | 1000  |     4     |     0.108 |       0.202 |        -1.9x |        0.319 |         -2.9x |       -1.6x |
| sp_matmul_topn       | 1000  |     8     |     0.107 |       0.112 |        -1.1x |        0.180 |         -1.7x |       -1.6x |

### Reading the table

* **`sp_matmul` (no top-n).** `sp_matmul_rs.sp_matmul` is **sequential by
  design** — only the top-n driver is parallelised. That is why the Rust column
  is flat across thread counts. If you do not need top-n, Scipy's
  `A.dot(B)` is the better choice; this row exists only to show the
  point-of-reference. The C++ path here is OpenMP-parallel.
* **`sp_matmul_topn`, small `top_n` (10–30).** Both libraries beat Scipy
  decisively past 2 threads — the top-n filter is doing real work. Rust tracks
  C++ to within 10–20 %.
* **`sp_matmul_topn`, large `top_n` (100, 1000).** Both libraries lose to Scipy
  at low thread counts because the heap maintenance dominates; the **gap to
  the C++ extension widens to ~1.5–1.6×** at `top_n = 1000`. Top-n at a value
  this close to the dense result size is a known weak spot — see
  ["Migrating to v1"](../README.md#use-cases) for guidance.
* **Threading scales similarly.** Both libraries show ~7× scaling at
  `n_threads = 8`. Rust uses rayon thread pools (cached process-wide); C++
  uses OpenMP.

### Full EDGAR corpus — self-similarity (A @ A.T)

The sampled table above stresses the per-row top-n heap on a small input.
This second sweep stresses the **end-to-end string-matching workload** that
`string_grouper` actually runs in production: the full 663 000-row EDGAR
TF-IDF matrix multiplied by its own transpose. Shape:
`A: (663 000, 193 190) × B: (193 190, 663 000)`, `A.nnz = B.nnz = 2 285 459`,
`repeats=3`.

Scipy and the plain `sp_matmul` (no top-n) rows are **omitted** here — the
dense result is too large for `A.dot(B)` to materialise on a 48 GB machine
(empirically OOM-killed in <1 min). `top_n ∈ {30, 100, 1000}` is dropped
because output memory grows as `663 000 × top_n × (8 + 4) bytes`; at
`top_n = 100` that is already ~800 MB before any intermediate buffers, and
the single-thread runs cross the 30-minute mark per cell.

| Benchmark      | top_n | n_threads | C++ min (s) | Rust min (s) | Rust vs C++ |
| :------------- | :---: | :-------: | ----------: | -----------: | ----------: |
| sp_matmul_topn |  10   |     1     |     132.537 |       68.585 |        1.9x |
| sp_matmul_topn |  10   |     8     |      28.143 |       12.903 |        2.2x |
| sp_matmul_topn |  10   |    10     |      28.429 |       11.567 |        2.5x |
| sp_matmul_topn |  20   |     1     |     131.878 |       70.350 |        1.9x |
| sp_matmul_topn |  20   |     8     |      27.505 |       13.154 |        2.1x |
| sp_matmul_topn |  20   |    10     |      27.524 |       11.415 |        2.4x |

Reading the table:

* **At workload-realistic scale, Rust runs in roughly 40–50 % of the time of
  the C++ extension** across every configuration tested (best case 41 %, at
  `top_n=10, n_threads=10`). The 20 000-row sampled bench above shows the
  reverse picture at small/large `top_n` — meaning the 20 000-row table
  measures something different from what `string_grouper` actually pays for.
* **The L1/L2 cache-blocking driver is the reason.** At 20 000 rows the
  per-row working set still fits in cache, so the C++ algorithm's flat layout
  is competitive. At 663 000 rows it does not, and `sp_matmul_rs`'s chunked
  driver (`chunked.rs`) keeps the dense accumulator + streamed B fragments
  L1-resident — which is the whole point of column-chunking.
* **Rust scales further past 8 threads.** Going from 8 → 10 threads gives
  Rust another ~10 % (12.9 → 11.6 s at `top_n=10`); the C++ OpenMP build is
  flat (28.1 → 28.4 s). Both libraries top out around the M5 Pro's perf-core
  count, but rayon's per-block work-stealing handles the slack-thread case
  better than OpenMP's static schedule on this workload.

Reproduce:

```shell
python bench_rust_vs_cpp_scipy.py \
    --rows 0 --only cpp,rust --topn-only \
    --top-ns 10,20 --threads 1,8,10 --repeats 3 \
    --out results_full.md
```

### Caveats

* The original C++ bench README shows numbers from an Apple M2 Pro and an
  Intel i9 — those tables are **not** directly comparable to the M5 Pro
  results above (~50 % faster cores, larger caches).
* `sparse_dot_topn` wheels on PyPI ship **without** OpenMP and rely on a
  vendored libomp; for an apples-to-apples comparison, build from source as
  shown in the Dependencies section.
* `richbench` was avoided because it only pits **two** candidates against each
  other; this script reports all three side-by-side in one table.
