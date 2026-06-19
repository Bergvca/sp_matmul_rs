//! A/B experiment: dense-accumulator chunk mode vs linked-list bookkeeping. Run:
//!   cargo run --release --example dense_mode_ab
//!
//! Three sections:
//! 1. notebook shape (high update density, d ≈ 1.0) — dense expected to win
//! 2. TF-IDF shape (d ≈ 0.0003) — linked list must not regress under Adaptive
//! 3. density ladder (seq) — locate the dense/linked crossover for the
//!    Adaptive threshold (touched fraction ∈ {1/16, 1/8, 1/4})

use std::time::Instant;

use sp_matmul_rs::{sp_matmul_topn, AccumMode, CsrMatrix, CsrView, SortMode, TopNOptions};

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

fn build_csr(seed: u64, nrows: usize, ncols: usize, nnz_per_row: usize) -> CsrMatrix<f64, i32> {
    let mut rng = SplitMix64::new(seed);
    let mut indptr: Vec<i32> = Vec::with_capacity(nrows + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f64> = Vec::new();
    indptr.push(0);
    let target = nnz_per_row.min(ncols);
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

fn as_view(m: &CsrMatrix<f64, i32>) -> CsrView<'_, f64, i32> {
    CsrView::new(m.nrows, m.ncols, &m.indptr, &m.indices, &m.data).unwrap()
}

/// Median wall time in ms over `runs` timed runs (after one warmup).
fn bench(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
    accum_mode: AccumMode,
    n_threads: Option<usize>,
    runs: usize,
) -> (f64, usize) {
    let opts = TopNOptions {
        sort: SortMode::ByValueDesc,
        accum_mode,
        n_threads,
        ..Default::default()
    };
    let mut nnz = 0usize;
    let warm = sp_matmul_topn(as_view(a), as_view(b), top_n, opts);
    std::hint::black_box(&warm);
    let mut times: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let c = sp_matmul_topn(as_view(a), as_view(b), top_n, opts);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        nnz = c.nnz();
        std::hint::black_box(&c);
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (times[times.len() / 2], nnz)
}

fn section(
    label: &str,
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
    n_threads: Option<usize>,
    runs: usize,
) {
    let modes = [
        ("linked", AccumMode::LinkedList),
        ("dense", AccumMode::Dense),
        ("adaptive", AccumMode::Adaptive),
    ];
    let mut linked_ms = f64::NAN;
    for (name, mode) in modes {
        let (ms, nnz) = bench(a, b, top_n, mode, n_threads, runs);
        if name == "linked" {
            linked_ms = ms;
        }
        let rel = ms / linked_ms;
        println!("{label:<28} {name:<9} {ms:9.1} ms  (vs linked: {rel:5.3})  nnz={nnz}");
    }
}

fn main() {
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let top_n = 10;

    // 1. Notebook shape: 20000×10000 · 10000×20000, ~1% dense, d = 100·200/20000 = 1.0.
    let a_nb = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b_nb = build_csr(0xB2B2, 10_000, 20_000, 200);
    println!("== notebook shape (d=1.0, touched ~63%) ==");
    section("notebook seq", &a_nb, &b_nb, top_n, None, runs);
    section("notebook 10thr", &a_nb, &b_nb, top_n, Some(10), runs);

    // 2. TF-IDF shape (tfidf_like.rs parallel_scaling): d = 8·8/200000 = 0.00032.
    let a_tf = build_csr(0xAAAA, 20_000, 50_000, 8);
    let b_tf = build_csr(0xBBBB, 50_000, 200_000, 8);
    println!("\n== TF-IDF shape (d=0.00032, touched ~0.03%) ==");
    section("tfidf seq", &a_tf, &b_tf, top_n, None, runs);
    section("tfidf 10thr", &a_tf, &b_tf, top_n, Some(10), runs);

    // 3. Density ladder (seq): fixed A (100 nnz/row), B nnz/row swept so that
    //    d = 100·nnzB/20000. Touched fraction t = 1 − e^(−d).
    println!("\n== density ladder (seq) ==");
    for &nnz_b in &[6usize, 13, 27, 58, 139, 200] {
        let d = 100.0 * nnz_b as f64 / 20_000.0;
        let t = 1.0 - (-d).exp();
        let b = build_csr(0xC3C3 ^ nnz_b as u64, 10_000, 20_000, nnz_b);
        let label = format!("ladder d={d:.3} t={t:.3}");
        section(&label, &a_nb, &b, top_n, None, runs);
    }
}
