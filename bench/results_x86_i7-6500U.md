# x86-64 validation: Intel i7-6500U (Skylake), Linux

Validation of the Apple-Silicon-tuned kernel heuristics on a small x86-64 box, driven by the
`string_grouper` EDGAR company-name workload (663 000 names, self-join). This machine is a
*validation* environment, not a tuning target — production x86-64 servers have more cores and
larger caches, so no constants were re-fitted to this laptop. Conclusions favour
hardware-derived formulas, runtime knobs, and build-flag guidance that generalize.

## Environment

- Intel Core i7-6500U @ 2.50 GHz (Skylake, AVX2+FMA), 2 cores / 4 threads (SMT), 1 socket.
- L1d 32 KiB per core (shared by SMT siblings), L2 256 KiB per core, L3 4 MiB shared.
- Linux 6.8, rustc 1.96.1, CPython 3.12; `maturin develop --release --features python,rayon`.
- `kernel_info()`: `l1d_bytes=32768, l1d_detected=True` (CPUID detection works on Linux/x86-64),
  `default_chunk_cols=2048` for all dtypes.
- CPU governor was `powersave` (could not be changed); mitigation: median/min of repeats,
  cool-downs between heavy runs. Run-to-run noise band ≈ 1–3 %; differences are called real
  only above max(5 %, 2× noise) ≈ 6 %.

## Workload and matrices

End-to-end call: `string_grouper.match_strings(names, number_of_processes=4)` over
`sec__edgar_company_info.csv` (663 000 rows) → trigram TF-IDF `A` (663 000 × 34 835,
11.76 M nnz ≈ 17.7 nnz/row), kernel call `sp_matmul_topn(A, A.T, top_n=20, threshold=0.8,
sort=True, n_threads=4)`.

> **Note on the cached `bench/tfidf_*.npz` matrices:** those are word-level TF-IDF
> (193 190-term vocabulary, ~3.4 nnz/row) — much sparser than what `string_grouper` actually
> produces (trigram analyzer, ~18–20 nnz/row, ~35 k vocabulary). Kernel-tuning conclusions in
> this file were measured on freshly dumped **true** `string_grouper` matrices; the npz shape
> is reported separately where used.

## End-to-end phase attribution (stock build)

| rows    | total (s) | kernel (s) | kernel % | vectorizer (s) | vectorizer % | other (s) |
| ------: | --------: | ---------: | -------: | -------------: | -----------: | --------: |
| 20 000  |      1.44 |       0.42 |     29 % |           0.95 |         66 % |      0.08 |
| 50 000  |      4.50 |       2.02 |     45 % |           2.28 |         51 % |      0.20 |
| 100 000 |     10.86 |       6.12 |     56 % |           4.36 |         40 % |      0.37 |
| 663 000 |    256.04 |     224.91 |     88 % |          28.53 |         11 % |      2.60 |

The kernel scales ~N^1.6, the TF-IDF vectorizer linearly, so at production scale the Rust
kernel dominates end-to-end time — kernel tuning is the right lever for this workload.

## chunk_cols sweep — true matrices, 5 % row sample of A, stock build

Median of 2 runs, ms. `top_n=20, threshold=0.8, sort=True`. Default (L1d-derived) is **2048**.

| chunk_cols | nt=1       | nt=2   | nt=4       |
| ---------: | ---------: | -----: | ---------: |
| 1024       | 31 237     |        | 13 729     |
| **2048**   | 26 980     | 15 058 | 12 213     |
| 4096       | 25 225     |        | 11 750     |
| 8192       | **24 761** |        | **11 665** |
| 16 384     | 25 237     |        | 12 184     |
| 32 768     | 26 736     |        |            |
| 65 536     | 28 686     |        |            |

- Sequential optimum is 8192 (scratch ≈ 96 KiB, i.e. an L2-resident budget): **8.9 %** faster
  than the default. At nt=4 the optimum (4096–8192) is only **4.5 %** off the default — below
  the significance bar.
- On the *sparser* cached-npz shape (3.4 nnz/row, 663 000 cols) the default 2048 was exactly
  optimal at every thread count, with larger chunks monotonically worse.
- The optimum is therefore **workload-dependent** (denser B rows amortize wider chunks), not a
  systematic platform offset: the L1d-derived formula stays as-is, and `sp_matmul_topn` now
  exposes `chunk_cols` (and the `SP_MATMUL_RS_CHUNK_COLS` env var) so heavy workloads can tune
  without rebuilding.
- SMT note: nt=4 > nt=2 > nt=1 throughout (12.2 s / 15.1 s / 27.0 s at default) — use logical
  cores, and no L1d-sharing penalty large enough to justify a thread-aware chunk formula.

## Build flags: `-C target-cpu=native` (AVX2 + FMA), true matrices

Stock manylinux-style builds take the non-FMA scalar path on x86-64
(`target_feature="fma"` is not in the baseline). Same sweep setup, median ms:

| config             | stock  | native | speedup    |
| :----------------- | -----: | -----: | ---------: |
| nt=1, chunk=2048   | 26 980 | 22 117 | **18.0 %** |
| nt=1, chunk=8192   | 24 761 | 20 523 | **17.1 %** |
| nt=4, chunk=2048   | 12 213 | 10 041 | **17.8 %** |
| nt=4, chunk=8192   | 11 665 |  9 682 | **17.0 %** |

- Correctness: the full Rust suite (314 tests, incl. parity vs C++ goldens at 1e-12 for f64)
  and the Python suite pass under `-C target-cpu=native` — FMA rounding stays inside the
  parity tolerances.
- On the sparser npz shape the native win was 10–14 % — the gain grows with arithmetic
  density, as expected for an FMA/vectorization effect.

## Production kernel path (auto tiling) — sanity check, true matrices, stock build

| config           | classic (`row_block=0`) | auto (production) | auto speedup |
| :--------------- | ----------------------: | ----------------: | -----------: |
| nt=1, chunk=2048 |                  53 434 |            26 980 |    **1.98×** |
| nt=4, chunk=2048 |                  22 492 |            12 213 |    **1.84×** |

The M5-calibrated `auto_tile` gates, `DEFAULT_ROW_BLOCK=2048`, projection heuristic and
`DENSE_MIN_DENSITY` all behave correctly on x86-64 for the real workload — no change needed.
(On the unrepresentative sparse npz shape the tiled path is mildly slower sequentially; the
auto gates are close to that boundary but the real workload is deep inside the win region.)

## End-to-end validation (full 663 000 rows, `number_of_processes=4`)

| build                                          | total (s) | kernel (s) | total vs baseline | kernel vs baseline |
| :--------------------------------------------- | --------: | ---------: | ----------------: | -----------------: |
| stock, default chunk                           |     256.0 |      224.9 |                 — |                  — |
| native, default chunk                          |     225.6 |      192.8 |       **−11.9 %** |            −14.3 % |
| native + `SP_MATMUL_RS_CHUNK_COLS=8192`        |     215.4 |      183.8 |       **−15.9 %** |            −18.3 % |

All three runs produce the identical 1 594 336 matches.

## Conclusions

1. **No kernel code/heuristic change is warranted for x86-64.** The L1d-derived
   `default_chunk_cols`, the auto-tiling gates, row-block default and projection heuristic all
   transfer from Apple Silicon to this machine (defaults are optimal or within ~5 % at
   production-like thread counts). No arch-gated constants were introduced; Apple Silicon
   behavior is untouched.
2. **Additive knob** (only code change): `sp_matmul_topn(..., chunk_cols=...)` and the
   `SP_MATMUL_RS_CHUNK_COLS` env var are now exposed through the Python API (the Rust binding
   already accepted it). Default `None` keeps the derived behavior byte-identical. Worth
   ~5–9 % on dense-ish workloads, more at low thread counts.
3. **FMA/AVX2 (biggest win, ~17–18 % kernel):** initially delivered as build-flag guidance
   (`RUSTFLAGS="-C target-cpu=native"`), since superseded by **runtime AVX2+FMA dispatch**
   baked into baseline builds — see the follow-up section below. From-source native builds
   remain equivalent (the dispatch machinery compiles away) and additionally get AVX-512 /
   `tune-cpu` where present.
4. Use **logical-core counts** for `n_threads` / `number_of_processes` on SMT machines.

---

# Follow-up: runtime AVX2+FMA dispatch (same machine, same matrices)

Implemented after the analysis above so that baseline x86-64 wheels get the FMA/AVX2 win
without build flags: `process_row` / `process_row_block` dispatch once, per cached CPUID
detection, to `#[target_feature(enable = "avx2,fma")]` clones of the same
`#[inline(always)]` kernel bodies (`const FMA: bool` selects `Scalar::mul_add_fused` at the
scatter sites). aarch64/Apple builds and `-C target-cpu=native` x86 builds compile the
machinery away entirely; `SP_MATMUL_RS_FORCE_BASELINE=1` disables it at runtime;
`kernel_info()['runtime_simd']` reports the active path. AVX-512 is deliberately not a
dispatch tier (scatter-bound kernel, downclocking on older Xeons, not measurable on this
box); native builds still get it at compile time.

Kernel A/B — true matrices, 5 % A sample, nt=4, chunk_cols=2048, median ms:

| build                                   | median ms | vs baseline |
| :-------------------------------------- | --------: | ----------: |
| stock build, forced baseline (env)      |    12 474 |           — |
| stock build, runtime dispatch (default) |    10 059 |  **−19.4 %** |
| `-C target-cpu=native` (reference)      |    10 041 |     −19.5 % |

The dispatched stock build matches the native build within noise — the clones capture
essentially the entire compile-flag win. Disassembly confirms all `vfmadd` instructions sit
inside the two `_avx2` clone symbols and none in the baseline path. Verified green:
`cargo test --features rayon` and `--no-default-features`, both also under
`SP_MATMUL_RS_FORCE_BASELINE=1`; the new `tests/simd_dispatch.rs` equivalence suite; clippy
`-D warnings`; 81 Python tests with `runtime_simd = "avx2+fma"` on a stock
`maturin develop` build.

End-to-end (full 663 000-row `match_strings`, `number_of_processes=4`, stock build with
runtime dispatch): **226.2 s** total / 194.1 s kernel — equal to the `-C target-cpu=native`
build (225.6 s / 192.8 s) and −11.6 % vs the pre-dispatch stock baseline (256.0 s / 224.9 s),
with the identical 1 594 336 matches. A default `pip install` build now performs like a
hand-flagged native build on AVX2-class hardware.
