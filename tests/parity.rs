//! Parity tests against goldens produced by `tests/data/generate_goldens.py`.
//!
//! Each `.npz` golden is the C++ extension's exact output for a given case ×
//! `(V, I)` dtype combination; the Rust port must match it within the dtype's
//! tolerance (see plan §2).

mod common;

use common::{assert_csr_equivalent, load_case, load_zip_case, GoldenCase, ZipGoldenCase};
use sp_matmul_rs::{
    sp_matmul_topn, zip_sp_matmul_topn, CsrView, Index, Scalar, SortMode, TopNOptions,
};

fn run_case<V, I>(name: &str, v: &str, i: &str)
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let case: GoldenCase<V, I> = load_case(name, v, i);
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

    let opts = TopNOptions::<V> {
        threshold: case.threshold,
        sort: if case.sort_by_value_desc {
            SortMode::ByValueDesc
        } else {
            SortMode::ByColumn
        },
        ..Default::default()
    };
    let got = sp_matmul_topn(a, b, case.top_n, opts);
    assert_csr_equivalent(&got, &case.expected_indptr, &case.expected_indices, &case.expected_data, case.expected_shape);
}

fn run_zip_case<V, I>(name: &str, v: &str, i: &str)
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let case: ZipGoldenCase<V, I> = load_zip_case(name, v, i);
    let views: Vec<CsrView<'_, V, I>> = case
        .chunks
        .iter()
        .map(|c| CsrView::new(c.shape.0, c.shape.1, &c.indptr, &c.indices, &c.data).unwrap())
        .collect();
    let got = zip_sp_matmul_topn(case.top_n, &views);
    assert_csr_equivalent(&got, &case.expected_indptr, &case.expected_indices, &case.expected_data, case.expected_shape);
}

// Generate one #[test] per (case, V, I) combination.
macro_rules! parity_case {
    ($case:literal) => {
        paste::paste! {
            #[test] fn [<$case _f32_i32>]() { run_case::<f32, i32>($case, "f32", "i32"); }
            #[test] fn [<$case _f32_i64>]() { run_case::<f32, i64>($case, "f32", "i64"); }
            #[test] fn [<$case _f64_i32>]() { run_case::<f64, i32>($case, "f64", "i32"); }
            #[test] fn [<$case _f64_i64>]() { run_case::<f64, i64>($case, "f64", "i64"); }
            #[test] fn [<$case _i32_i32>]() { run_case::<i32, i32>($case, "i32", "i32"); }
            #[test] fn [<$case _i32_i64>]() { run_case::<i32, i64>($case, "i32", "i64"); }
            #[test] fn [<$case _i64_i32>]() { run_case::<i64, i32>($case, "i64", "i32"); }
            #[test] fn [<$case _i64_i64>]() { run_case::<i64, i64>($case, "i64", "i64"); }
        }
    };
}

macro_rules! zip_parity_case {
    ($case:literal) => {
        paste::paste! {
            #[test] fn [<$case _f32_i32>]() { run_zip_case::<f32, i32>($case, "f32", "i32"); }
            #[test] fn [<$case _f32_i64>]() { run_zip_case::<f32, i64>($case, "f32", "i64"); }
            #[test] fn [<$case _f64_i32>]() { run_zip_case::<f64, i32>($case, "f64", "i32"); }
            #[test] fn [<$case _f64_i64>]() { run_zip_case::<f64, i64>($case, "f64", "i64"); }
            #[test] fn [<$case _i32_i32>]() { run_zip_case::<i32, i32>($case, "i32", "i32"); }
            #[test] fn [<$case _i32_i64>]() { run_zip_case::<i32, i64>($case, "i32", "i64"); }
            #[test] fn [<$case _i64_i32>]() { run_zip_case::<i64, i32>($case, "i64", "i32"); }
            #[test] fn [<$case _i64_i64>]() { run_zip_case::<i64, i64>($case, "i64", "i64"); }
        }
    };
}

parity_case!("small_dense");
parity_case!("tall_skinny");
parity_case!("wide_sparse");
parity_case!("with_threshold");
parity_case!("topn_eq_ncols");
parity_case!("topn_gt_ncols");
parity_case!("empty_a");
parity_case!("empty_b");
parity_case!("all_below_threshold");

zip_parity_case!("zip_three_chunks");
