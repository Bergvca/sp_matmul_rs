# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] — 2026-07-15

### Fixed
- Index overflow now fails loudly instead of silently truncating. The kernels
  narrow offsets to the index dtype with an unchecked cast, so a result whose
  non-zero count exceeded `idx_dtype`'s range (`i32::MAX` for the default
  `int32`) previously returned a corrupted `indptr` (wrapped/negative values)
  with no warning. The drivers now check the running non-zero total as each
  row is finalised and abort the moment overflow becomes certain — a panic in
  the Rust API, mapped to `OverflowError` (with a message pointing to
  `idx_dtype=np.int64`) in the Python bindings. `zip_sp_matmul_topn`
  additionally validates the total zipped column width up front, before any
  work. Added `Index::max_usize()` (provided method, non-breaking) and
  `csr::check_index_capacity()` to support the checks.
- `sp_matmul` no longer desynchronises `indptr` from `indices`/`data` when a
  dot product cancels to exactly zero. The size pass counts every touched
  column, but the fill pass skipped exact-zero sums — a bug inherited from the
  C++ original (`sp_matmul.hpp` has the same mismatch), where any exact
  cancellation shifted all subsequent entries out of alignment with the row
  boundaries. Cancelled entries are now stored as explicit zeros, matching
  scipy's `csr_matmat` and the top-n kernels' no-threshold behaviour.

## [0.2.0] — 2026-07-09

Added optimisations for x86-64 CPU's based on FMA/AVX2

### Added
- Runtime AVX2+FMA dispatch for baseline x86-64 builds (including PyPI wheels):
  `process_row` / `process_row_block` dispatch once, per cached CPUID detection,
  to `#[target_feature(enable = "avx2,fma")]` clones of the same kernel bodies,
  recovering the ~17–18% FMA/AVX2 win without requiring `target-cpu=native`.
- `SP_MATMUL_RS_FORCE_BASELINE=1` kill switch to disable the AVX2+FMA clones
  (useful for A/B measurements).
- `kernel_info()['runtime_simd']` reporting the active path
  (`"avx2+fma"`, `"baseline"`, or `"compile-time"`).
- `chunk_cols` option exposed through the Python `sp_matmul_topn()` API, plus a
  `SP_MATMUL_RS_CHUNK_COLS` environment override.

  `chunk_cols` sets the column-chunk width of the cache-blocked kernel. Leave it
  as `None` (the default) to derive the width from the detected L1d cache size —
  inspect the per-dtype defaults via `kernel_info()['default_chunk_cols']`. Set
  it explicitly to tune for your workload; denser `B` rows can favour widths
  above the L1d-derived default. Any value yields identical results, only
  performance changes. Examples:
  ```python
  # Explicit width via the keyword argument
  C = sp_matmul_topn(A, B, top_n=10, chunk_cols=8192)
  ```
  ```shell
  # Or override the derived default for the whole process
  SP_MATMUL_RS_CHUNK_COLS=8192 python your_script.py
  ```
  The keyword argument takes precedence; the env var only applies when
  `chunk_cols` is `None`.
- x86-64 validation benchmarks and results (`bench/results_x86_i7-6500U.md`).
- `tests/simd_dispatch.rs` asserting dispatch/baseline equivalence across
  dtypes, projections, accum modes and chunk widths.

### Changed
- `Scalar` gained `mul_add_fused` (always-fused `llvm.fma` for floats), routed
  through a `const FMA: bool` generic so the fused/baseline branch folds at
  monomorphization.
- On aarch64 and compile-time `target_feature=fma` builds, the dispatch
  machinery compiles away entirely; native builds still get full features
  (including AVX-512 where present) at compile time.

## [0.1.0] — Initial release

First public release of `sp_matmul_rs`, a Rust port of
[`sparse_dot_topn`](https://github.com/ing-bank/sparse_dot_topn).

### Added
- Sequential CSR × CSR sparse multiplication: `sp_matmul`, `sp_matmul_topn`.
- Column-chunked driver as the default `sp_matmul_topn` path
  (`chunked::sp_matmul_topn_chunked`) — sized to fit L1/L2.
- Rayon-backed parallel variants behind the default `rayon` feature.
- `zip_sp_matmul_topn` for merging distributed per-chunk top-n results.
- Standalone PyO3 + numpy Python distribution (`sp_matmul_rs`) with a public
  API mirroring `sparse_dot_topn`: `sp_matmul`, `sp_matmul_topn`,
  `zip_sp_matmul_topn`.
- Supported dtype matrix: values `{f32, f64, i32, i64}`, indices `{i32, i64}`.
