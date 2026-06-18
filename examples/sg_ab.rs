//! O2/O3 A/B op de echte string_grouper TF-IDF-matrices.
//!
//! Laadt `dev_tests/bench_data/` (gedumpt door `dev_tests/dump_tfidf.py`),
//! bemonstert A als contiguë spans en vergelijkt het productiepad met:
//!   - geforceerde projectie (BinarySearch vs Cursor),
//!   - O2 rij-blocking (row_block-sweep),
//!   - O3 tiled CSR (tile_b), en de combinatie.
//! Elke variant moet bit-for-bit gelijk zijn aan base.
//!
//! Run: cargo run --release --example sg_ab
//! Env: RUNS (default 3), SAMPLE (default 0.05), SPANS (default 16),
//!      NT (default 0 = geen MT-pass)

use std::path::PathBuf;
use std::time::Instant;

use sp_matmul_rs::{
    default_chunk_cols, sp_matmul_topn, BProjection, CsrMatrix, CsrView, SortMode, TiledB,
    TopNOptions,
};

fn bench_dir() -> PathBuf {
    std::env::var("BENCH_DATA")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../dev_tests/bench_data")
        })
}

fn read_bytes(name: &str) -> Vec<u8> {
    let p = bench_dir().join(name);
    std::fs::read(&p).unwrap_or_else(|e| panic!("kan {} niet lezen: {e}", p.display()))
}

fn read_i64(name: &str) -> Vec<i64> {
    read_bytes(name)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_i32(name: &str) -> Vec<i32> {
    read_bytes(name)
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f64(name: &str) -> Vec<f64> {
    read_bytes(name)
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn load(label: &str, ncols: usize) -> CsrMatrix<f64, i32> {
    let indptr64 = read_i64(&format!("{label}_indptr.bin"));
    let indices = read_i32(&format!("{label}_indices.bin"));
    let data = read_f64(&format!("{label}_data.bin"));
    let nrows = indptr64.len() - 1;
    assert!(*indptr64.last().unwrap() <= i32::MAX as i64);
    let indptr: Vec<i32> = indptr64.into_iter().map(|x| x as i32).collect();
    CsrMatrix {
        nrows,
        ncols,
        indptr,
        indices,
        data,
    }
}

fn view(m: &CsrMatrix<f64, i32>) -> CsrView<'_, f64, i32> {
    CsrView::new(m.nrows, m.ncols, &m.indptr, &m.indices, &m.data).unwrap()
}

/// Neem `spans` contiguë rijspans, samen ~`frac` van alle rijen.
fn sample_rows(a: &CsrMatrix<f64, i32>, frac: f64, spans: usize) -> CsrMatrix<f64, i32> {
    let take = ((a.nrows as f64 * frac) as usize).max(spans).min(a.nrows);
    let span_len = take / spans;
    let stride = a.nrows / spans;
    let mut indptr: Vec<i32> = vec![0];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    for s in 0..spans {
        let lo = s * stride;
        let hi = (lo + span_len).min(a.nrows);
        for i in lo..hi {
            let rs = a.indptr[i] as usize;
            let re = a.indptr[i + 1] as usize;
            indices.extend_from_slice(&a.indices[rs..re]);
            data.extend_from_slice(&a.data[rs..re]);
            indptr.push(indices.len() as i32);
        }
    }
    CsrMatrix {
        nrows: indptr.len() - 1,
        ncols: a.ncols,
        indptr,
        indices,
        data,
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[v.len() / 2]
}

fn run_variant(
    a: CsrView<'_, f64, i32>,
    b: CsrView<'_, f64, i32>,
    top_n: usize,
    opts: TopNOptions<f64>,
    runs: usize,
) -> (CsrMatrix<f64, i32>, f64) {
    let mut times = Vec::with_capacity(runs);
    let mut out = None;
    for _ in 0..runs {
        let t0 = Instant::now();
        let c = sp_matmul_topn(a, b, top_n, opts);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        out = Some(c);
    }
    (out.unwrap(), median(times))
}

fn check_equal(base: &CsrMatrix<f64, i32>, c: &CsrMatrix<f64, i32>, name: &str) {
    assert_eq!(base.indptr, c.indptr, "{name}: indptr wijkt af");
    assert_eq!(base.indices, c.indices, "{name}: indices wijken af");
    assert!(
        base.data.iter().zip(&c.data).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{name}: data wijkt af"
    );
}

fn main() {
    let runs = env_usize("RUNS", 3);
    let spans = env_usize("SPANS", 16);
    let frac = env_f64("SAMPLE", 0.05);
    let nt = env_usize("NT", 0);
    let top_n = 20usize;
    let threshold = 0.8f64;

    let b = load("B", usize::MAX); // ncols volgt uit A
    let a = load("A", b.nrows);
    let b = CsrMatrix { ncols: a.nrows, ..b };
    println!(
        "A {}x{} nnz {} | B {}x{} nnz {}",
        a.nrows, a.ncols, a.indptr.last().unwrap(),
        b.nrows, b.ncols, b.indptr.last().unwrap()
    );

    let chunk_cols = default_chunk_cols::<f64>().min(b.ncols);
    let n_chunks = b.ncols.div_ceil(chunk_cols);
    let avg_b = *b.indptr.last().unwrap() as f64 / b.nrows as f64;
    println!(
        "chunk_cols={chunk_cols} n_chunks={n_chunks} | avg nnz/B-rij {avg_b:.1} vs 4*n_chunks={} (heuristiek kiest {})",
        4 * n_chunks,
        if avg_b >= (4 * n_chunks) as f64 { "BinarySearch" } else { "Cursor" }
    );

    let a_sub = sample_rows(&a, frac, spans);
    println!(
        "sample: {} rijen ({}%) in {spans} spans, runs={runs}\n",
        a_sub.nrows,
        (100.0 * a_sub.nrows as f64 / a.nrows as f64).round()
    );

    // O3-gate: tiled preprocessing apart klokken.
    let t0 = Instant::now();
    let tiled = TiledB::<f64>::build(view(&b), chunk_cols);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("TiledB::build over volle B: {build_ms:.1} ms (gate: <10% van één volle multiply)\n");
    drop(tiled);

    let base_opts = TopNOptions {
        threshold: Some(threshold),
        sort: SortMode::ByValueDesc,
        ..Default::default()
    };

    let tiled_rbs: Vec<usize> = std::env::var("TILED_RBS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![8, 32, 64, 256]);
    let skip_o2 = env_usize("SKIP_O2", 0) == 1;

    // row_block 0 = klassieke kernel; de productie-default (auto) kiest sinds
    // de O3+O2-port zelf het tiled+blocked pad op deze vorm.
    let classic_opts = TopNOptions { row_block: Some(0), ..base_opts };
    let mut variants: Vec<(String, TopNOptions<f64>)> = vec![
        ("base (klassiek)".into(), classic_opts),
        ("auto (productie)".into(), base_opts),
    ];
    if !skip_o2 {
        variants.push((
            "proj=BinarySearch".into(),
            TopNOptions { projection: Some(BProjection::BinarySearch), ..base_opts },
        ));
        variants.push((
            "proj=Cursor".into(),
            TopNOptions { projection: Some(BProjection::Cursor), ..base_opts },
        ));
        for rb in [8usize, 32, 64, 256] {
            variants.push((
                format!("O2 rb={rb} BS"),
                TopNOptions {
                    row_block: Some(rb),
                    projection: Some(BProjection::BinarySearch),
                    ..base_opts
                },
            ));
            variants.push((
                format!("O2 rb={rb} Cur"),
                TopNOptions {
                    row_block: Some(rb),
                    projection: Some(BProjection::Cursor),
                    ..base_opts
                },
            ));
        }
        variants.push((
            "O3 tiled rb=1".into(),
            TopNOptions { tile_b: true, ..base_opts },
        ));
    }
    for &rb in &tiled_rbs {
        variants.push((
            format!("O3+O2 tiled rb={rb}"),
            TopNOptions { tile_b: true, row_block: Some(rb), ..base_opts },
        ));
    }

    let av = view(&a_sub);
    let bv = view(&b);
    let mut base_out: Option<CsrMatrix<f64, i32>> = None;
    let mut base_ms = 0.0f64;
    let mut best = (f64::INFINITY, String::new());
    println!("{:<22} {:>10} {:>8}", "variant (seq)", "ms", "vs base");
    for (name, opts) in &variants {
        let (c, ms) = run_variant(av, bv, top_n, *opts, runs);
        match &base_out {
            None => {
                base_out = Some(c);
                base_ms = ms;
            }
            Some(b0) => check_equal(b0, &c, name),
        }
        if ms < best.0 {
            best = (ms, name.clone());
        }
        println!("{name:<22} {ms:>10.1} {:>8.3}", ms / base_ms);
    }
    println!(
        "\nbeste: {} ({:.1} ms, {:.3}x) | geschatte volle seq multiply: {:.1} s (base)",
        best.1,
        best.0,
        best.0 / base_ms,
        base_ms / 1e3 * (a.nrows as f64 / a_sub.nrows as f64)
    );
    println!(
        "TiledB::build = {:.1}% van één geschatte volle multiply",
        100.0 * build_ms / (base_ms * a.nrows as f64 / a_sub.nrows as f64)
    );

    if nt > 1 {
        println!("\nMT-pass (n_threads={nt}, volle A):");
        let avf = view(&a);
        let mut mt_variants: Vec<(String, TopNOptions<f64>)> = vec![
            (
                "base (klassiek)".into(),
                TopNOptions { n_threads: Some(nt), ..classic_opts },
            ),
            (
                "auto (productie)".into(),
                TopNOptions { n_threads: Some(nt), ..base_opts },
            ),
        ];
        for &rb in &tiled_rbs {
            mt_variants.push((
                format!("O3+O2 tiled rb={rb}"),
                TopNOptions {
                    n_threads: Some(nt),
                    tile_b: true,
                    row_block: Some(rb),
                    ..base_opts
                },
            ));
        }
        let mut mt_base: Option<CsrMatrix<f64, i32>> = None;
        let mut mt_base_ms = 0.0;
        for (name, opts) in &mt_variants {
            let (c, ms) = run_variant(avf, bv, top_n, *opts, runs);
            match &mt_base {
                None => {
                    mt_base = Some(c);
                    mt_base_ms = ms;
                }
                Some(b0) => check_equal(b0, &c, name),
            }
            println!("{name:<22} {ms:>10.1} {:>8.3}", ms / mt_base_ms);
        }
    }
}
