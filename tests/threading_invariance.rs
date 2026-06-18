//! Phase 3 threading-invariance tests.
//!
//! For each parity-golden case × dtype pair, run the public driver at
//! `n_threads ∈ {2, 4, 8}` and assert the result matches the `n_threads = 1`
//! (sequential chunked) output. The parallel path is bit-equivalent to the
//! sequential path on every input shape.
//!
//! Float equivalence uses the per-dtype tolerance; int equivalence is
//! per-row value-multiset (same tie-break carve-out as the parity test).
//!
//! Gated on `feature = "rayon"` — without it the parallel path is absent
//! from the module tree and these tests do not compile.

#![cfg(feature = "rayon")]

mod common;

use common::{assert_csr_equivalent, load_case, GoldenCase};
use sp_matmul_rs::{sp_matmul_topn, CsrView, Index, Scalar, SortMode, TopNOptions};

const PAR_THREAD_COUNTS: &[usize] = &[2, 4, 8];

fn check_case<V, I>(case: &GoldenCase<V, I>)
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
    let base = TopNOptions::<V> {
        threshold: case.threshold,
        sort: if case.sort_by_value_desc {
            SortMode::ByValueDesc
        } else {
            SortMode::ByColumn
        },
        ..Default::default()
    };
    let seq = sp_matmul_topn(a, b, case.top_n, base);
    for &nt in PAR_THREAD_COUNTS {
        let opts = TopNOptions {
            n_threads: Some(nt),
            ..base
        };
        let par = sp_matmul_topn(a, b, case.top_n, opts);
        assert_csr_equivalent(
            &par,
            &seq.indptr,
            &seq.indices,
            &seq.data,
            (seq.nrows, seq.ncols),
        );
    }
}

fn run_invariance<V, I>(name: &str, v: &str, i: &str)
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let case: GoldenCase<V, I> = load_case(name, v, i);
    check_case(&case);
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
