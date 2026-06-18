# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


