//! Phase 2 chunk-invariance tests.
//!
//! For each parity-golden case × dtype pair, run the chunked driver at several
//! `chunk_cols` and assert the result matches the single-chunk oracle
//! (`sp_matmul_topn_unchunked_for_tests`): chunk invariance for {1, 7, 64, 1024,
//! ncols, 10·ncols}.
//!
//! `chunk_cols == 1` runs ncols-many chunks and is expensive on the larger
//! goldens, so we restrict it to small fixtures via `SMALL_GOLDENS`.

mod common;

use common::{assert_csr_equivalent, load_case, GoldenCase};
use sp_matmul_rs::{
    sp_matmul_topn, sp_matmul_topn_unchunked_for_tests, AccumMode, CsrView, Index, Scalar,
    SortMode, TopNOptions,
};

/// Goldens cheap enough to run at chunk_cols=1 without test runtime regressing.
const SMALL_GOLDENS: &[&str] = &[
    "small_dense",
    "tall_skinny",
    "with_threshold",
    "topn_eq_ncols",
    "topn_gt_ncols",
    "empty_a",
    "empty_b",
    "all_below_threshold",
];

/// Run the full case at one `chunk_cols` setting and compare to the oracle.
fn check_case_at<V, I>(case: &GoldenCase<V, I>, chunk_cols: usize)
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let a = CsrView::new(
        case.a_shape.0,
        case.a_shape.1,
        &case.a_indptr,
        &case.a_indices,
        &case.a_data,
    )
    .unwrap();
    let b = CsrView::new(
        case.b_shape.0,
        case.b_shape.1,
        &case.b_indptr,
        &case.b_indices,
        &case.b_data,
    )
    .unwrap();
    let opts_chunked = TopNOptions::<V> {
        threshold: case.threshold,
        sort: if case.sort_by_value_desc {
            SortMode::ByValueDesc
        } else {
            SortMode::ByColumn
        },
        chunk_cols: Some(chunk_cols),
        ..Default::default()
    };
    // Oracle ignores chunk_cols entirely, so it runs the Phase 1 single-chunk body.
    let oracle = sp_matmul_topn_unchunked_for_tests(a, b, case.top_n, opts_chunked);
    let got = sp_matmul_topn(a, b, case.top_n, opts_chunked);
    assert_csr_equivalent(
        &got,
        &oracle.indptr,
        &oracle.indices,
        &oracle.data,
        (oracle.nrows, oracle.ncols),
    );

    // Forced dense-accumulator mode (O1) must match the oracle too. Float-only:
    // ints fall back to the linked list internally, so re-running them here
    // would duplicate the default-mode check above.
    if V::IS_FLOAT {
        let opts_dense = TopNOptions::<V> {
            accum_mode: AccumMode::Dense,
            ..opts_chunked
        };
        let got_dense = sp_matmul_topn(a, b, case.top_n, opts_dense);
        assert_csr_equivalent(
            &got_dense,
            &oracle.indptr,
            &oracle.indices,
            &oracle.data,
            (oracle.nrows, oracle.ncols),
        );
    }

    // Forced tiled+blocked kernel (O3+O2) must match the oracle for every
    // chunk width and block size — ints included (linked-list path). When the
    // chunk width exceeds TiledB's u16 bound this exercises the non-tiled
    // blocked path instead, which must be equally invariant.
    for rb in [1usize, 3, 2048] {
        let opts_blocked = TopNOptions::<V> {
            tile_b: true,
            row_block: Some(rb),
            ..opts_chunked
        };
        let got_blocked = sp_matmul_topn(a, b, case.top_n, opts_blocked);
        assert_csr_equivalent(
            &got_blocked,
            &oracle.indptr,
            &oracle.indices,
            &oracle.data,
            (oracle.nrows, oracle.ncols),
        );
    }
}

fn run_invariance<V, I>(name: &str, v: &str, i: &str)
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let case: GoldenCase<V, I> = load_case(name, v, i);
    let ncols = case.b_shape.1.max(1);
    let widths: Vec<usize> = {
        let mut w = vec![
            7usize,
            64,
            1024,
            ncols,
            ncols.saturating_mul(10).max(ncols + 1),
        ];
        if SMALL_GOLDENS.contains(&name) {
            w.push(1);
        }
        // Dedup; small-ncols cases may collapse several widths into the same value.
        w.sort_unstable();
        w.dedup();
        w
    };
    for cc in widths {
        check_case_at(&case, cc);
    }
}

macro_rules! invariance_case {
    ($case:literal) => {
        paste::paste! {
            #[test] fn [<$case _f32_i32>]() { run_invariance::<f32, i32>($case, "f32", "i32"); }
            #[test] fn [<$case _f32_i64>]() { run_invariance::<f32, i64>($case, "f32", "i64"); }
            #[test] fn [<$case _f64_i32>]() { run_invariance::<f64, i32>($case, "f64", "i32"); }
            #[test] fn [<$case _f64_i64>]() { run_invariance::<f64, i64>($case, "f64", "i64"); }
            #[test] fn [<$case _i32_i32>]() { run_invariance::<i32, i32>($case, "i32", "i32"); }
            #[test] fn [<$case _i32_i64>]() { run_invariance::<i32, i64>($case, "i32", "i64"); }
            #[test] fn [<$case _i64_i32>]() { run_invariance::<i64, i32>($case, "i64", "i32"); }
            #[test] fn [<$case _i64_i64>]() { run_invariance::<i64, i64>($case, "i64", "i64"); }
        }
    };
}

invariance_case!("small_dense");
invariance_case!("tall_skinny");
invariance_case!("wide_sparse");
invariance_case!("with_threshold");
invariance_case!("topn_eq_ncols");
invariance_case!("topn_gt_ncols");
invariance_case!("empty_a");
invariance_case!("empty_b");
invariance_case!("all_below_threshold");
