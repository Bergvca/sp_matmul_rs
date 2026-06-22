# sp\_matmul\_rs

[![Crates.io](https://img.shields.io/crates/v/sp_matmul_rs.svg)](https://crates.io/crates/sp_matmul_rs)
[![PyPI](https://img.shields.io/pypi/v/sp_matmul_rs.svg)](https://pypi.org/project/sp_matmul_rs/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**sp\_matmul\_rs** provides a fast way to perform a sparse matrix multiplication (SpGEMM) followed by top-n multiplication result selection.

It is a Rust port of [**sparse\_dot\_topn**](https://github.com/ing-bank/sparse_dot_topn), created and optimised for use in [**string\_grouper**](https://github.com/Bergvca/string_grouper) — where the dominant workload is matching every row of a very large TF-IDF matrix against every other row and keeping only the top-n nearest neighbours.

**sparse_dot_topn** is based on  [Gustavson's algorithm](https://www.researchgate.net/figure/Gustavson-algorithm-Sparse-matrix-matrix-multiplication-is-performed-in-a-row-wise_fig4_327077990) 
 but retains only the best _n_ values per output row. **sp\_matmul\_rs** retains the same algorithmic core 
but is built around L1/L2 cache-blocking: the default driver column-chunks `B` so the dense per-row 
accumulator and the streamed `B` fragments stay resident in L1 (chunk sizes are automatically calculated based on cache size), which is the dominant performance lever once the working set stops fitting in cache.
Parallelism is provided by `rayon` (enabled by default) and the Python distribution releases the GIL around the kernel.
On Apple M5 Pro over a 663k × 193k TF-IDF self-similarity workload **sp\_matmul\_rs** runs in as little as 41% of the time of **sparse\_dot\_topn** when retaining the top 10 values per row and utilising 10 cores.
See the benchmark directory for details.

## Usage

`sp_matmul_topn` supports `{CSR, CSC, COO}` matrices with `{32, 64}bit {int, float}` data.
Note that `COO` and `CSC` inputs are converted to the `CSR` format and are therefore slower.
The Python surface (`sp_matmul`, `sp_matmul_topn`, `zip_sp_matmul_topn`) mirrors `sparse_dot_topn`, so migration from the C++ extension is typically a one-line import swap.
Note that `sp_matmul_topn(A, B, top_n=B.shape[1])` is equal to `sp_matmul(A, B)` and `A.dot(B)`.

```python
import numpy as np
import scipy.sparse as sparse
from sp_matmul_rs import sp_matmul, sp_matmul_topn

A = sparse.random(1000, 100, density=0.1, format="csr", dtype=np.float64)
B = sparse.random(100, 2000, density=0.1, format="csr", dtype=np.float64)

# Compute C and retain the top 10 values per row
C = sp_matmul_topn(A, B, top_n=10)

# or parallelised matrix multiplication without top-n selection
C = sp_matmul(A, B, n_threads=2)
# or with top-n selection
C = sp_matmul_topn(A, B, top_n=10, n_threads=2)
# pass n_threads=-1 to use all but one physical core
C = sp_matmul_topn(A, B, top_n=10, n_threads=-1)

# If you are only interested in values above a certain threshold
C = sp_matmul_topn(A, B, top_n=10, threshold=0.8)

# If you set the threshold we cannot easily determine the number of non-zero
# entries beforehand. Therefore, we allocate memory for `ceil(top_n * A.shape[0] * density)`
# non-zero entries. You can set the expected density to reduce the amount pre-allocated
# entries. Note that if we allocate too little an expensive copy(ies) will need to happen.
C = sp_matmul_topn(A, B, top_n=10, threshold=0.8, density=0.1)
```

The package imports as `sp_matmul_rs` and does **not** share its namespace with `sparse_dot_topn` — both can be installed side-by-side in the same environment.

## Installation

**sp\_matmul\_rs** provides wheels for CPython 3.9 to 3.13 for:

* Windows (64bit)
* Linux (x86_64 and aarch64, both manylinux and musllinux)
* macOS (x86_64 and ARM)

```shell
pip install sp_matmul_rs
```

**sp\_matmul\_rs** relies on a Rust extension for the computationally intensive multiplication routine.
**Note that the wheels ship with `rayon` enabled, providing parallelisation out-of-the-box.**
If you need to disable threading at runtime, omit the `n_threads` argument (or pass `n_threads=1`).

Installing from source requires a Rust toolchain (1.75+) and [`maturin`](https://www.maturin.rs/):

```shell
pip install maturin
pip install sp_matmul_rs --no-binary sp_matmul_rs
```

### Supported

| Values             | Indices  | Python      | Platforms                                                                     |
|--------------------|----------|-------------|-------------------------------------------------------------------------------|
| f32, f64, i32, i64 | i32, i64 | 3.9 – 3.13  | manylinux x86_64/aarch64, musllinux x86_64, macOS x86_64/arm64, Windows x86_64 |

## Rust crate

For Rust callers, the same kernels are available without the Python bindings:

```toml
[dependencies]
sp_matmul_rs = "0.1"
```

```rust
use sp_matmul_rs::{sp_matmul_topn, CsrView, SortMode, TopNOptions};

let c = sp_matmul_topn::<f64, i32>(
    a_view,
    b_view,
    10,
    TopNOptions { sort: SortMode::ByValue, ..Default::default() },
);
```

### Features

| Feature  | Default | What it enables                                          |
|----------|---------|----------------------------------------------------------|
| `rayon`  | yes     | Parallel column-chunked driver (`parallel::*`).          |
| `python` | no      | PyO3 + numpy bindings — used by the Python distribution. |

### Module layout

| Module        | Responsibility                                                |
|---------------|---------------------------------------------------------------|
| `scalar`      | `Scalar` trait + impls for `f32, f64, i32, i64`               |
| `index`       | `Index` trait + impls for `i32, i64`                          |
| `csr`         | `CsrView` (borrowed) and `CsrMatrix` (owned)                  |
| `maxheap`     | Bounded max-heap retaining top-n scores per row               |
| `matmul`      | Sequential `sp_matmul` (no top-n)                             |
| `matmul_topn` | Sequential `sp_matmul_topn` (single full-column path)         |
| `chunked`     | Column-chunked driver — the L1/L2 cache-blocking pillar       |
| `zip`         | `zip_sp_matmul_topn` for distributed/cluster results          |
| `parallel`    | Rayon-backed variants (feature `rayon`)                       |
| `python`      | PyO3 + `numpy` bindings (feature `python`)                    |

## Build from source

```shell
# Rust crate
cargo build --release                       # default features: rayon
cargo build --release --no-default-features
cargo test
cargo bench

# Python extension (requires maturin and CPython >=3.9)
pip install maturin
maturin develop --release --features python,rayon
python -c "import sp_matmul_rs; print(sp_matmul_rs.kernel_info())"
```

## Benchmarks

`bench/bench_rust_vs_cpp_scipy.py` is a three-way comparison against the upstream C++ extension (`sparse_dot_topn`) and `scipy.sparse` on a word-level TF-IDF matrix over the EDGAR company-name corpus ([Kaggle: dattapiy/sec-edgar-companies-list](https://www.kaggle.com/datasets/dattapiy/sec-edgar-companies-list)).
Numbers below were measured on an Apple M5 Pro (10 performance cores), 48 GB, macOS 26.4, with both libraries built with parallelism enabled (`sparse_dot_topn` against Homebrew libomp, `sp_matmul_rs` with `rayon`).
Times are per-call **min** across timed runs (one untimed warm-up first).

### Full EDGAR corpus — A @ A.T (663 000 × 193 190)

The workload that `string_grouper` actually runs in production: every company name matched against every other.
Scipy and the no-top-n baseline are omitted because the dense result is too large to materialise on a 48 GB machine.
`repeats = 3`.

| Benchmark      | top_n | n_threads | C++ (s) | Rust (s) | Rust vs C++ |
| :------------- | :---: | :-------: | ------: | -------: | ----------: |
| sp_matmul_topn |  10   |     1     | 132.537 |   68.585 |        1.9x |
| sp_matmul_topn |  10   |     8     |  28.143 |   12.903 |        2.2x |
| sp_matmul_topn |  10   |    10     |  28.429 |   11.567 |        2.5x |
| sp_matmul_topn |  20   |     1     | 131.878 |   70.350 |        1.9x |
| sp_matmul_topn |  20   |     8     |  27.505 |   13.154 |        2.1x |
| sp_matmul_topn |  20   |    10     |  27.524 |   11.415 |        2.4x |

At workload-realistic scale, **sp\_matmul\_rs** consistently runs in 40–50 % of the time of the C++ extension and continues to scale past 8 threads where the OpenMP build plateaus.
The L1/L2 cache-blocking driver does the heavy lifting — at 663 000 rows the per-row working set no longer fits in cache, and the chunked layout keeps the dense accumulator L1-resident.

### Sampled shape — 20 000 × 193 190 × 20 000

The same shape as the upstream C++ bench README, included for direct comparability.
`repeats = 5`.

| Benchmark      | top_n | n_threads | Scipy (s) | C++ (s) | Rust (s) | Rust vs Scipy | Rust vs C++ |
| :------------- | :---: | :-------: | --------: | ------: | -------: | ------------: | ----------: |
| sp_matmul_topn |  10   |     1     |     0.110 |   0.102 |    0.083 |          1.3x |        1.2x |
| sp_matmul_topn |  10   |     8     |     0.107 |   0.014 |    0.015 |          6.9x |       -1.1x |
| sp_matmul_topn |  30   |     1     |     0.108 |   0.124 |    0.119 |         -1.1x |        1.0x |
| sp_matmul_topn |  30   |     8     |     0.109 |   0.017 |    0.021 |          5.3x |       -1.2x |
| sp_matmul_topn |  100  |     8     |     0.107 |   0.031 |    0.041 |          2.6x |       -1.3x |
| sp_matmul_topn | 1000  |     8     |     0.107 |   0.112 |    0.180 |         -1.7x |       -1.6x |

At this smaller shape the per-row working set still fits in cache, so the C++ extension's flat layout is competitive (Rust trails by ~1.2–1.6× at large `top_n`).
The sampled bench is included for direct comparability with the upstream C++ project's published table — **the full-corpus numbers above are the more relevant indicator** for real string-matching workloads.

The full `{top_n, n_threads}` grid (including the plain `sp_matmul` baseline and Scipy timings) lives in [`bench/README.md`](bench/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
