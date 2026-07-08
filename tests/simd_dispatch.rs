//! Runtime AVX2+FMA dispatch equivalence — the auto-dispatched kernel must
//! match the forced-baseline kernel within the per-dtype parity tolerance
//! (bit-exact multisets for ints), across projections, accumulator modes and
//! chunk widths.
//!
//! Single `#[test]`: the dispatch override is process-global and the libtest
//! harness is multithreaded, so all comparisons run in one serial body. On
//! non-x86-64 targets (or CPUs without AVX2/FMA) both paths are the same code
//! and the test degenerates to a self-comparison, which is fine.

mod common;

use common::assert_csr_equivalent;
use sp_matmul_rs::{
    force_simd_baseline_for_tests, sp_matmul_topn, AccumMode, CsrMatrix, CsrView, Index, Scalar,
    SortMode, TopNOptions,
};

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
}

/// Deterministic random CSR with `nnz_per_row` strictly increasing column
/// indices per row and small positive values (exact in every supported dtype).
fn build_csr<V, I>(seed: u64, nrows: usize, ncols: usize, nnz_per_row: usize) -> CsrMatrix<V, I>
where
    V: Scalar + From<i16>,
    I: Index,
{
    let mut rng = SplitMix64::new(seed);
    let mut indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    let mut indices: Vec<I> = Vec::new();
    let mut data: Vec<V> = Vec::new();
    indptr.push(I::from_usize(0));
    let target = nnz_per_row.min(ncols);
    let mut cols: Vec<usize> = Vec::with_capacity(target);
    for _ in 0..nrows {
        cols.clear();
        while cols.len() < target {
            let c = (rng.next_u64() % ncols as u64) as usize;
            if let Err(pos) = cols.binary_search(&c) {
                cols.insert(pos, c);
            }
        }
        for &c in &cols {
            indices.push(I::from_usize(c));
            data.push(<V as From<i16>>::from(1 + (rng.next_u64() % 97) as i16));
        }
        indptr.push(I::from_usize(indices.len()));
    }
    CsrMatrix {
        nrows,
        ncols,
        indptr,
        indices,
        data,
    }
}

fn view<V: Scalar, I: Index>(m: &CsrMatrix<V, I>) -> CsrView<'_, V, I> {
    CsrView::new(m.nrows, m.ncols, &m.indptr, &m.indices, &m.data).unwrap()
}

fn check_dtype<V, I>(seed: u64)
where
    V: Scalar + From<i16>,
    I: Index,
{
    let a = build_csr::<V, I>(seed, 120, 60, 12);
    let b = build_csr::<V, I>(seed ^ 0xB0B0, 60, 300, 24);
    let row_threshold = V::IS_FLOAT.then(|| <V as From<i16>>::from(5000));

    let accums: &[AccumMode] = if V::IS_FLOAT {
        &[AccumMode::Adaptive, AccumMode::Dense, AccumMode::LinkedList]
    } else {
        &[AccumMode::Adaptive]
    };

    for &chunk_cols in &[7usize, 64, 300, 10_000] {
        for &accum in accums {
            for threshold in [None, row_threshold] {
                let opts = TopNOptions::<V> {
                    threshold,
                    sort: SortMode::ByValueDesc,
                    chunk_cols: Some(chunk_cols),
                    accum_mode: accum,
                    ..Default::default()
                };

                force_simd_baseline_for_tests(true);
                let base = sp_matmul_topn(view(&a), view(&b), 10, opts);
                force_simd_baseline_for_tests(false);
                let auto = sp_matmul_topn(view(&a), view(&b), 10, opts);

                assert_csr_equivalent(
                    &auto,
                    &base.indptr,
                    &base.indices,
                    &base.data,
                    (base.nrows, base.ncols),
                );
            }
        }
    }
}

#[test]
fn dispatch_matches_baseline_all_dtypes() {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        eprintln!("simd_dispatch: avx2+fma detected = {avx2}");
    }

    check_dtype::<f64, i32>(0x51D0);
    check_dtype::<f64, i64>(0x51D1);
    check_dtype::<f32, i32>(0x51D2);
    check_dtype::<f32, i64>(0x51D3);
    check_dtype::<i32, i32>(0x51D4);
    check_dtype::<i64, i64>(0x51D5);

    // Leave the process-global override cleared for any other code in-process.
    force_simd_baseline_for_tests(false);
}
