
//! Property tests for sequential kernels — invariants checked without external goldens.
//!
//! Run on f64 × i32. The kernels are dtype-generic; correctness for other dtypes is
//! covered by `parity.rs`. We cap shapes/density tight so proptest's shrinking is
//! fast and the inner loop stays meaningful (see plan §3.8).

use std::collections::BTreeMap;

use proptest::prelude::*;
use sp_matmul_rs::{
    sp_matmul, sp_matmul_topn, zip_sp_matmul_topn, CsrMatrix, CsrView, SortMode, TopNOptions,
};

#[derive(Debug, Clone)]
struct CsrInput {
    nrows: usize,
    ncols: usize,
    indptr: Vec<i32>,
    indices: Vec<i32>,
    data: Vec<f64>,
}

impl CsrInput {
    fn view(&self) -> CsrView<'_, f64, i32> {
        CsrView::new(
            self.nrows,
            self.ncols,
            &self.indptr,
            &self.indices,
            &self.data,
        )
        .unwrap()
    }
}

prop_compose! {
    fn arb_csr(max_rows: usize, max_cols: usize)(
        nrows in 1usize..=max_rows,
        ncols in 1usize..=max_cols,
    )(
        nrows in Just(nrows),
        ncols in Just(ncols),
        // Generate per-row column sets via a bitmask; keep dense enough to exercise
        // overlap with B but not trivially saturated.
        rows in proptest::collection::vec(
            proptest::collection::vec(any::<u8>(), ncols..=ncols),
            nrows..=nrows,
        ),
        vals_pool in proptest::collection::vec(1u32..1_000_000, 1..2048),
    ) -> CsrInput {
        let mut indptr = Vec::with_capacity(nrows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        let mut cursor = 0usize;
        for row in &rows {
            for (c, mask) in row.iter().enumerate() {
                if *mask < 64 {  // ~25% density per row
                    indices.push(c as i32);
                    let raw = vals_pool[cursor % vals_pool.len()] as f64;
                    // Perturb slightly so floats stay tie-free.
                    let perturb = 1.0 + 1e-4 * ((cursor as f64).sin().abs());
                    data.push(0.1 + 0.9 * (raw / 1_000_000.0) * perturb);
                    cursor += 1;
                }
            }
            indptr.push(indices.len() as i32);
        }
        CsrInput { nrows, ncols, indptr, indices, data }
    }
}

prop_compose! {
    fn arb_a_b()(
        a in arb_csr(20, 25),
        ncols_b in 1usize..=30,
        b_seed in any::<u64>(),
    ) -> (CsrInput, CsrInput) {
        // Build B independently so its nrows == a.ncols.
        let nrows_b = a.ncols;
        let mut rng_state = b_seed | 1;
        let mut indptr = Vec::with_capacity(nrows_b + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for r in 0..nrows_b {
            // Simple deterministic-from-seed iteration; keep B density modest.
            for c in 0..ncols_b {
                rng_state = rng_state.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                let bit = (rng_state >> 33) & 0xFF;
                if bit < 64 {  // ~25% density per row
                    indices.push(c as i32);
                    let v = 0.1 + 0.9 * (((rng_state >> 8) & 0xFFFFFF) as f64) / (0xFFFFFF as f64);
                    let perturb = 1.0 + 1e-4 * (((r * 31 + c) as f64).cos().abs());
                    data.push(v * perturb);
                }
            }
            indptr.push(indices.len() as i32);
        }
        let b = CsrInput { nrows: nrows_b, ncols: ncols_b, indptr, indices, data };
        (a, b)
    }
}

fn rows_as_maps(c: &CsrMatrix<f64, i32>) -> Vec<BTreeMap<i32, f64>> {
    (0..c.nrows)
        .map(|i| {
            let s = c.indptr[i] as usize;
            let e = c.indptr[i + 1] as usize;
            (s..e).map(|k| (c.indices[k], c.data[k])).collect()
        })
        .collect()
}

fn maps_close(a: &[BTreeMap<i32, f64>], b: &[BTreeMap<i32, f64>]) -> bool {
    if a.len() != b.len() { return false; }
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() { return false; }
        for (ka, va) in ra.iter() {
            match rb.get(ka) {
                None => return false,
                Some(vb) => {
                    let diff = (va - vb).abs();
                    let mag = va.abs().max(vb.abs());
                    if !(diff <= 1e-12 || diff <= 1e-12 * mag) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    /// sp_matmul_topn with top_n >= ncols, no threshold, ByColumn must collapse to sp_matmul.
    #[test]
    fn topn_eq_ncols_equals_matmul((a, b) in arb_a_b()) {
        let av = a.view();
        let bv = b.view();
        let c_topn = sp_matmul_topn(av, bv, b.ncols, TopNOptions::default());
        let c_plain = sp_matmul(av, bv);
        prop_assert!(maps_close(&rows_as_maps(&c_topn), &rows_as_maps(&c_plain)));
    }

    /// Every output row has at most `top_n` entries.
    #[test]
    fn topn_le_actual_nnz((a, b) in arb_a_b(), top_n in 1usize..=12) {
        let c = sp_matmul_topn(
            a.view(),
            b.view(),
            top_n,
            TopNOptions {
                sort: SortMode::ByValueDesc,
                ..Default::default()
            },
        );
        for i in 0..c.nrows {
            let s = c.indptr[i] as usize;
            let e = c.indptr[i + 1] as usize;
            prop_assert!(e - s <= top_n, "row {} has {} > top_n {}", i, e - s, top_n);
        }
    }

    /// ByColumn and ByValueDesc must select the same per-row entries (just sorted differently).
    #[test]
    fn sort_modes_agree_as_set((a, b) in arb_a_b(), top_n in 1usize..=12) {
        let by_col = sp_matmul_topn(
            a.view(), b.view(), top_n,
            TopNOptions { sort: SortMode::ByColumn, ..Default::default() },
        );
        let by_val = sp_matmul_topn(
            a.view(), b.view(), top_n,
            TopNOptions { sort: SortMode::ByValueDesc, ..Default::default() },
        );
        prop_assert!(maps_close(&rows_as_maps(&by_col), &rows_as_maps(&by_val)));
    }

    /// With a threshold, every output value strictly exceeds it.
    #[test]
    fn threshold_filters_correctly((a, b) in arb_a_b(), top_n in 1usize..=10, t in 0.0f64..0.5) {
        let opts = TopNOptions {
            threshold: Some(t),
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let c = sp_matmul_topn(a.view(), b.view(), top_n, opts);
        for &v in &c.data {
            prop_assert!(v > t, "value {} did not exceed threshold {}", v, t);
        }
    }

    /// The chunked driver must produce the same per-row entries (as a set) at any
    /// `chunk_cols` as at the single-chunk setting. Pin Phase 2 §7.3.
    #[test]
    fn chunked_matches_unchunked(
        (a, b) in arb_a_b(),
        top_n in 1usize..=12,
        chunk_cols in prop_oneof![
            Just(1usize),
            4usize..64,
            64usize..512,
            Just(usize::MAX),
        ],
    ) {
        let opts_unchunked = TopNOptions {
            chunk_cols: Some(usize::MAX),
            ..Default::default()
        };
        let opts_chunked = TopNOptions {
            chunk_cols: Some(chunk_cols),
            ..Default::default()
        };
        let c_u = sp_matmul_topn(a.view(), b.view(), top_n, opts_unchunked);
        let c_c = sp_matmul_topn(a.view(), b.view(), top_n, opts_chunked);
        prop_assert!(maps_close(&rows_as_maps(&c_u), &rows_as_maps(&c_c)));
    }

    /// The parallel driver must produce the same per-row entries (as a set)
    /// at any `n_threads ∈ {2..=8}` as the sequential driver at the same
    /// `chunk_cols`. Pins Phase 3 §8.3 — threading invariance under fuzz.
    #[cfg(feature = "rayon")]
    #[test]
    fn parallel_matches_sequential(
        (a, b) in arb_a_b(),
        top_n in 1usize..=12,
        chunk_cols in prop_oneof![
            Just(1usize),
            4usize..64,
            64usize..512,
            Just(usize::MAX),
        ],
        n_threads in 2usize..=8,
    ) {
        let base = TopNOptions {
            chunk_cols: Some(chunk_cols),
            ..Default::default()
        };
        let par_opts = TopNOptions {
            n_threads: Some(n_threads),
            ..base
        };
        let c_seq = sp_matmul_topn(a.view(), b.view(), top_n, base);
        let c_par = sp_matmul_topn(a.view(), b.view(), top_n, par_opts);
        prop_assert!(maps_close(&rows_as_maps(&c_seq), &rows_as_maps(&c_par)));
    }

    /// Zipping a single chunk is equivalent (as a set per row) to truncating that chunk
    /// to top_n by value desc.
    #[test]
    fn zip_collapses_single_chunk((a, b) in arb_a_b(), top_n in 1usize..=10) {
        let chunk = sp_matmul_topn(
            a.view(), b.view(), top_n,
            TopNOptions { sort: SortMode::ByValueDesc, ..Default::default() },
        );
        let chunk_view = CsrView::new(
            chunk.nrows, chunk.ncols,
            &chunk.indptr, &chunk.indices, &chunk.data,
        ).unwrap();
        let zipped = zip_sp_matmul_topn(top_n, &[chunk_view]);
        prop_assert!(maps_close(&rows_as_maps(&chunk), &rows_as_maps(&zipped)));
    }
}

/// Edge cases: not proptest, just unit tests.
#[test]
fn empty_inputs_produce_zero_matrix() {
    let zero_indptr_2: Vec<i32> = vec![0, 0, 0];
    let nil_data_f: Vec<f64> = vec![];
    let nil_idx: Vec<i32> = vec![];
    let a = CsrView::new(2, 3, &zero_indptr_2, &nil_idx, &nil_data_f).unwrap();
    let b_indptr: Vec<i32> = vec![0, 1, 2, 3];
    let b_indices: Vec<i32> = vec![0, 1, 0];
    let b_data: Vec<f64> = vec![1.0, 2.0, 3.0];
    let b = CsrView::new(3, 2, &b_indptr, &b_indices, &b_data).unwrap();

    // empty A
    let c = sp_matmul_topn(a, b, 5, TopNOptions::default());
    assert_eq!(c.nnz(), 0);

    // top_n == 0
    let a_full = CsrView::new(3, 3, &b_indptr, &b_indices, &b_data).unwrap();
    let b_full = CsrView::new(3, 3, &b_indptr, &b_indices, &b_data).unwrap();
    let c = sp_matmul_topn(a_full, b_full, 0, TopNOptions::default());
    assert_eq!(c.nnz(), 0);

    // empty zip
    let empty: Vec<CsrView<'_, f64, i32>> = vec![];
    let z = zip_sp_matmul_topn(5, &empty);
    assert_eq!(z.nrows, 0);
}
