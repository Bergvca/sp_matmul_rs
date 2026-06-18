# sp_matmul_rs

[![Crates.io](https://img.shields.io/crates/v/sp_matmul_rs.svg)](https://crates.io/crates/sp_matmul_rs)
[![PyPI](https://img.shields.io/pypi/v/sp_matmul_rs.svg)](https://pypi.org/project/sp_matmul_rs/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Sparse CSR × CSR matrix multiplication with **fused top-n selection per row** —
a Rust port of [`sparse_dot_topn`](https://github.com/ing-bank/sparse_dot_topn).

The default driver column-chunks `B` so that the dense accumulator and the
streamed `B` fragments stay resident in L1/L2 cache; this is the dominant
performance lever for very large, very sparse inputs and is built in, not bolted
on. Parallelism is provided by `rayon` (enabled by default); the Python
distribution releases the GIL around the kernel.

## Python (PyPI)

```shell
pip install sp_matmul_rs
```

```python
import numpy as np
from scipy import sparse
from sp_matmul_rs import sp_matmul_topn

A = sparse.random(1000, 1000, density=0.01, format="csr", dtype=np.float64)
B = sparse.random(1000, 1000, density=0.01, format="csr", dtype=np.float64)

C = sp_matmul_topn(A, B, top_n=10, sort=True, n_threads=-1)
```

The package imports as `sp_matmul_rs` and does **not** share its namespace with
`sparse_dot_topn` — both can be installed side-by-side in the same environment.
The Python surface (`sp_matmul`, `sp_matmul_topn`, `zip_sp_matmul_topn`) mirrors
`sparse_dot_topn`, so migration is typically a one-line import swap.

### Supported

| Values         | Indices    | Python | Platforms                          |
|----------------|------------|--------|------------------------------------|
| f32, f64, i32, i64 | i32, i64 | 3.9 – 3.13 | manylinux x86_64/aarch64, musllinux x86_64, macOS x86_64/arm64, Windows x86_64 |

## Rust (crates.io)

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

## Module layout

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

## License

Apache-2.0. See [LICENSE](LICENSE).
