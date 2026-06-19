//! TF-IDF-shaped benches for the chunked driver (Phase 2 §8.1).
//!
//! Three groups:
//! 1. `chunked_vs_unchunked` — fixed shape, sweep `chunk_cols`.
//! 2. `projection_strategy` — force binary search vs cursor on the same shape.
//! 3. `ncols_sweep` — fixed density, vary `ncols` to confirm the per-row scratch
//!    footprint stays roughly flat (eyeballed via top/htop, not asserted here).
//!
//! Inputs are generated deterministically from a fixed seed so different
//! invocations measure the same workload.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use sp_matmul_rs::{
    sp_matmul_topn, sp_matmul_topn_chunked, CsrMatrix, CsrView, SortMode, TopNOptions,
};

// Reach into the crate's internal projection enum via the public reexport path.
// The bench is in the same Cargo package, so `pub(crate)` is not visible; we
// instead exercise both strategies through `chunk_cols` extremes that the
// `pick_projection` heuristic resolves to either branch.

/// SplitMix64 — fast deterministic generator, no external dep.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Build a CSR matrix with ~`nnz_per_row` random nonzero columns per row,
/// with strictly-increasing column indices (suitable for chunked driver).
fn build_csr(seed: u64, nrows: usize, ncols: usize, nnz_per_row: usize) -> CsrMatrix<f64, i32> {
    let mut rng = SplitMix64::new(seed);
    let mut indptr: Vec<i32> = Vec::with_capacity(nrows + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    indptr.push(0);
    let target = nnz_per_row.min(ncols);
    // Reservoir-ish: pick `target` distinct columns per row by sampling without
    // replacement via a sorted-insertion buffer.
    let mut buf: Vec<i32> = Vec::with_capacity(target);
    for _ in 0..nrows {
        buf.clear();
        while buf.len() < target {
            let c = (rng.next_u64() % ncols as u64) as i32;
            if let Err(pos) = buf.binary_search(&c) {
                buf.insert(pos, c);
            }
        }
        for &c in &buf {
            indices.push(c);
            // TF-IDF-ish: values in (0, 1].
            data.push(0.1 + 0.9 * rng.next_f64_unit());
        }
        indptr.push(indices.len() as i32);
    }
    CsrMatrix {
        nrows,
        ncols,
        indptr,
        indices,
        data,
    }
}

fn as_view<'a>(m: &'a CsrMatrix<f64, i32>) -> CsrView<'a, f64, i32> {
    CsrView::new(m.nrows, m.ncols, &m.indptr, &m.indices, &m.data).unwrap()
}

fn chunked_vs_unchunked(c: &mut Criterion) {
    // Moderate shape: 4k × 16k, nnz/row ~ 8 in A and 16 in B.
    let nrows_a = 4_000usize;
    let inner = 8_000usize;
    let ncols_b = 16_000usize;
    let a = build_csr(0xA1A1, nrows_a, inner, 8);
    let b = build_csr(0xB2B2, inner, ncols_b, 16);
    let top_n = 10usize;

    let mut group = c.benchmark_group("chunked_vs_unchunked");
    group.throughput(Throughput::Elements(nrows_a as u64));
    for &cc in &[256usize, 1024, 4096, 16_384, ncols_b] {
        group.bench_with_input(BenchmarkId::from_parameter(cc), &cc, |bencher, &cc| {
            bencher.iter(|| {
                let opts = TopNOptions {
                    sort: SortMode::ByValueDesc,
                    chunk_cols: Some(cc),
                    ..Default::default()
                };
                sp_matmul_topn(as_view(&a), as_view(&b), top_n, opts)
            });
        });
    }
    group.finish();
}

fn projection_strategy(c: &mut Criterion) {
    // Two shapes that bias `pick_projection`:
    //   - Dense B → BinarySearch dominates.
    //   - Sparse B + many chunks → Cursor dominates.
    // We can't pin the strategy directly from outside the crate, but we drive
    // the heuristic toward each branch via chunk_cols.
    let nrows_a = 2_000usize;
    let inner = 4_000usize;
    let ncols_b = 32_000usize;
    let a = build_csr(0xC1C1, nrows_a, inner, 8);
    let b_dense = build_csr(0xD2D2, inner, ncols_b, 64);
    let b_sparse = build_csr(0xE3E3, inner, ncols_b, 4);
    let top_n = 10usize;

    let mut group = c.benchmark_group("projection_strategy");
    group.throughput(Throughput::Elements(nrows_a as u64));
    group.bench_function("dense_b_cc4096", |bencher| {
        bencher.iter(|| {
            let opts = TopNOptions {
                sort: SortMode::ByValueDesc,
                chunk_cols: Some(4096),
                ..Default::default()
            };
            sp_matmul_topn(as_view(&a), as_view(&b_dense), top_n, opts)
        });
    });
    group.bench_function("sparse_b_cc4096", |bencher| {
        bencher.iter(|| {
            let opts = TopNOptions {
                sort: SortMode::ByValueDesc,
                chunk_cols: Some(4096),
                ..Default::default()
            };
            sp_matmul_topn(as_view(&a), as_view(&b_sparse), top_n, opts)
        });
    });
    group.finish();
}

fn ncols_sweep(c: &mut Criterion) {
    // Same per-row density; varying ncols stresses per-row scratch sizing.
    let nrows_a = 1_000usize;
    let inner = 4_000usize;
    let top_n = 10usize;
    let mut group = c.benchmark_group("ncols_sweep");
    group.throughput(Throughput::Elements(nrows_a as u64));
    for &ncols in &[10_000usize, 100_000, 1_000_000] {
        let a = build_csr(0xF1F1 ^ ncols as u64, nrows_a, inner, 8);
        let b = build_csr(0xF2F2 ^ ncols as u64, inner, ncols, 8);
        group.bench_with_input(BenchmarkId::from_parameter(ncols), &ncols, |bencher, _| {
            bencher.iter(|| {
                let opts = TopNOptions {
                    sort: SortMode::ByValueDesc,
                    chunk_cols: None, // default heuristic
                    ..Default::default()
                };
                sp_matmul_topn_chunked(as_view(&a), as_view(&b), top_n, opts)
            });
        });
    }
    group.finish();
}

/// Phase 3 scaling bench: fixed TF-IDF-shaped input, sweep `n_threads`.
/// Wall time at `nt > 1` is expected to be strictly below `nt = 1` on at
/// least one shape; the comparison vs. the C++ OpenMP build lives in the
/// cross-language harness (see `bench/bench_rust_vs_cpp.py`).
fn parallel_scaling(c: &mut Criterion) {
    let nrows_a = 20_000usize;
    let inner = 50_000usize;
    let ncols_b = 200_000usize;
    let a = build_csr(0xAAAA, nrows_a, inner, 8);
    let b = build_csr(0xBBBB, inner, ncols_b, 8);
    let top_n = 10usize;

    let mut group = c.benchmark_group("parallel_scaling");
    group.throughput(Throughput::Elements(nrows_a as u64));
    for &nt in &[1usize, 2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(nt), &nt, |bencher, &nt| {
            bencher.iter(|| {
                let opts = TopNOptions {
                    sort: SortMode::ByValueDesc,
                    n_threads: Some(nt),
                    ..Default::default()
                };
                sp_matmul_topn(as_view(&a), as_view(&b), top_n, opts)
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    chunked_vs_unchunked,
    projection_strategy,
    ncols_sweep,
    parallel_scaling,
);
criterion_main!(benches);
