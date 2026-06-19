//! Shared helpers for parity tests: npz loader + CSR-equivalence assertion.

#![allow(dead_code)]

use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use npyz::npz::NpzArchive;
use sp_matmul_rs::{CsrMatrix, Index, Scalar};

type Archive = NpzArchive<BufReader<File>>;

pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

fn open(name: &str, v: &str, i: &str) -> Archive {
    let path = data_dir().join(format!("{name}_{v}_{i}.npz"));
    NpzArchive::open(&path)
        .unwrap_or_else(|e| panic!("failed to open golden {}: {}", path.display(), e))
}

fn read_array<T: npyz::Deserialize>(archive: &mut Archive, name: &str) -> Vec<T> {
    let npy = archive
        .by_name(name)
        .unwrap_or_else(|e| panic!("error reading entry {name}: {e}"))
        .unwrap_or_else(|| panic!("missing entry {name}"));
    npy.into_vec::<T>()
        .unwrap_or_else(|e| panic!("error decoding entry {name}: {e}"))
}

fn read_shape(archive: &mut Archive, name: &str) -> (usize, usize) {
    let arr = read_array::<i64>(archive, name);
    assert_eq!(arr.len(), 2, "shape array {name} must have len 2");
    (arr[0] as usize, arr[1] as usize)
}

fn read_scalar_i64(archive: &mut Archive, name: &str) -> i64 {
    let arr = read_array::<i64>(archive, name);
    assert_eq!(arr.len(), 1, "scalar {name} must have len 1");
    arr[0]
}

fn read_scalar_bool(archive: &mut Archive, name: &str) -> bool {
    let arr = read_array::<bool>(archive, name);
    assert_eq!(arr.len(), 1, "bool scalar {name} must have len 1");
    arr[0]
}

fn read_scalar_v<V: Scalar + npyz::Deserialize>(archive: &mut Archive, name: &str) -> V {
    let arr = read_array::<V>(archive, name);
    assert_eq!(arr.len(), 1, "scalar {name} must have len 1");
    arr[0]
}

pub struct GoldenCase<V: Scalar, I: Index> {
    pub a_indptr: Vec<I>,
    pub a_indices: Vec<I>,
    pub a_data: Vec<V>,
    pub a_shape: (usize, usize),
    pub b_indptr: Vec<I>,
    pub b_indices: Vec<I>,
    pub b_data: Vec<V>,
    pub b_shape: (usize, usize),
    pub top_n: usize,
    pub sort_by_value_desc: bool,
    pub threshold: Option<V>,
    pub expected_indptr: Vec<I>,
    pub expected_indices: Vec<I>,
    pub expected_data: Vec<V>,
    pub expected_shape: (usize, usize),
}

pub fn load_case<V, I>(name: &str, v: &str, i: &str) -> GoldenCase<V, I>
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let mut archive = open(name, v, i);
    let a_indptr = read_array::<I>(&mut archive, "a_indptr");
    let a_indices = read_array::<I>(&mut archive, "a_indices");
    let a_data = read_array::<V>(&mut archive, "a_data");
    let a_shape = read_shape(&mut archive, "a_shape");
    let b_indptr = read_array::<I>(&mut archive, "b_indptr");
    let b_indices = read_array::<I>(&mut archive, "b_indices");
    let b_data = read_array::<V>(&mut archive, "b_data");
    let b_shape = read_shape(&mut archive, "b_shape");
    let top_n = read_scalar_i64(&mut archive, "top_n") as usize;
    let sort_by_value_desc = read_scalar_bool(&mut archive, "sort_by_value_desc");
    let has_threshold = read_scalar_bool(&mut archive, "has_threshold");
    let threshold_val = read_scalar_v::<V>(&mut archive, "threshold");
    let threshold = if has_threshold {
        Some(threshold_val)
    } else {
        None
    };
    let expected_indptr = read_array::<I>(&mut archive, "expected_indptr");
    let expected_indices = read_array::<I>(&mut archive, "expected_indices");
    let expected_data = read_array::<V>(&mut archive, "expected_data");
    let expected_shape = read_shape(&mut archive, "expected_shape");

    GoldenCase {
        a_indptr,
        a_indices,
        a_data,
        a_shape,
        b_indptr,
        b_indices,
        b_data,
        b_shape,
        top_n,
        sort_by_value_desc,
        threshold,
        expected_indptr,
        expected_indices,
        expected_data,
        expected_shape,
    }
}

pub struct ZipChunk<V: Scalar, I: Index> {
    pub indptr: Vec<I>,
    pub indices: Vec<I>,
    pub data: Vec<V>,
    pub shape: (usize, usize),
}

pub struct ZipGoldenCase<V: Scalar, I: Index> {
    pub chunks: Vec<ZipChunk<V, I>>,
    pub top_n: usize,
    pub expected_indptr: Vec<I>,
    pub expected_indices: Vec<I>,
    pub expected_data: Vec<V>,
    pub expected_shape: (usize, usize),
}

pub fn load_zip_case<V, I>(name: &str, v: &str, i: &str) -> ZipGoldenCase<V, I>
where
    V: Scalar + npyz::Deserialize,
    I: Index + npyz::Deserialize,
{
    let mut archive = open(name, v, i);
    let n_chunks = read_scalar_i64(&mut archive, "n_chunks") as usize;
    let top_n = read_scalar_i64(&mut archive, "top_n") as usize;
    let mut chunks = Vec::with_capacity(n_chunks);
    for k in 0..n_chunks {
        let indptr = read_array::<I>(&mut archive, &format!("chunk_{k}_indptr"));
        let indices = read_array::<I>(&mut archive, &format!("chunk_{k}_indices"));
        let data = read_array::<V>(&mut archive, &format!("chunk_{k}_data"));
        let shape = read_shape(&mut archive, &format!("chunk_{k}_shape"));
        chunks.push(ZipChunk {
            indptr,
            indices,
            data,
            shape,
        });
    }
    let expected_indptr = read_array::<I>(&mut archive, "expected_indptr");
    let expected_indices = read_array::<I>(&mut archive, "expected_indices");
    let expected_data = read_array::<V>(&mut archive, "expected_data");
    let expected_shape = read_shape(&mut archive, "expected_shape");
    ZipGoldenCase {
        chunks,
        top_n,
        expected_indptr,
        expected_indices,
        expected_data,
        expected_shape,
    }
}

/// Assert `got` matches the expected CSR, modulo intra-row column order and per-dtype
/// floating-point tolerance.
///
/// Tie handling: at the top-n boundary the C++ and Rust heaps can legitimately pick
/// different columns when several entries share a value (see plan §5.3). For floats
/// the golden generator perturbs values to make ties impossible, so we hard-fail on
/// any per-column mismatch. For ints we relax the per-row check to value-multiset
/// equivalence — both algorithms must produce the same sorted value sequence, but
/// columns at tied values may differ.
pub fn assert_csr_equivalent<V, I>(
    got: &CsrMatrix<V, I>,
    expected_indptr: &[I],
    expected_indices: &[I],
    expected_data: &[V],
    expected_shape: (usize, usize),
) where
    V: Scalar,
    I: Index,
{
    assert_eq!(got.nrows, expected_shape.0, "nrows mismatch");
    assert_eq!(got.ncols, expected_shape.1, "ncols mismatch");
    assert_eq!(
        got.indptr.len(),
        expected_indptr.len(),
        "indptr length mismatch"
    );
    for (k, (g, e)) in got.indptr.iter().zip(expected_indptr.iter()).enumerate() {
        assert_eq!(g.to_usize(), e.to_usize(), "indptr[{k}] differs");
    }
    let (abs_tol, rel_tol) = V::parity_tol();
    let int_dtype = abs_tol == 0.0 && rel_tol == 0.0;
    for row in 0..got.nrows {
        let g_start = got.indptr[row].to_usize();
        let g_end = got.indptr[row + 1].to_usize();
        let e_start = expected_indptr[row].to_usize();
        let e_end = expected_indptr[row + 1].to_usize();
        assert_eq!(
            g_end - g_start,
            e_end - e_start,
            "row {row} nnz differs (got {} vs expected {})",
            g_end - g_start,
            e_end - e_start,
        );

        if int_dtype {
            // Compare as sorted value multisets — tolerates int tie-break divergence
            // at the top-n boundary.
            let mut got_vals: Vec<i64> = (g_start..g_end)
                .map(|k| got.data[k].to_i64().unwrap())
                .collect();
            let mut exp_vals: Vec<i64> = (e_start..e_end)
                .map(|k| expected_data[k].to_i64().unwrap())
                .collect();
            got_vals.sort_unstable();
            exp_vals.sort_unstable();
            assert_eq!(got_vals, exp_vals, "row {row}: value multiset differs",);
        } else {
            let mut got_row: Vec<(usize, V)> = (g_start..g_end)
                .map(|k| (got.indices[k].to_usize(), got.data[k]))
                .collect();
            let mut exp_row: Vec<(usize, V)> = (e_start..e_end)
                .map(|k| (expected_indices[k].to_usize(), expected_data[k]))
                .collect();
            got_row.sort_by_key(|(idx, _)| *idx);
            exp_row.sort_by_key(|(idx, _)| *idx);
            for ((gi, gv), (ei, ev)) in got_row.iter().zip(exp_row.iter()) {
                assert_eq!(gi, ei, "row {row}: column index mismatch ({gi} vs {ei})");
                let gv_f = gv.to_f64().unwrap();
                let ev_f = ev.to_f64().unwrap();
                let diff = (gv_f - ev_f).abs();
                let mag = gv_f.abs().max(ev_f.abs());
                let ok = diff <= abs_tol || diff <= rel_tol * mag;
                assert!(
                    ok,
                    "row {row} col {gi}: value mismatch (got {gv_f} vs expected {ev_f}, diff {diff})",
                );
            }
        }
    }
}
