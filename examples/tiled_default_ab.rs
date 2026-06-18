//! Regressiecheck voor de tiled+blocked kernel (O3+O2) op vormen waar de
//! klassieke kernel al goed presteert: de notebookvorm (dense pad) en een
//! tiny vorm. Gebruikt vóór het wijzigen van de productie-default.
//!
//! Run: cargo run --release --example tiled_default_ab
//! Env: RUNS (default 5), RB (default 2048), NT (default 10)

use std::time::Instant;

use sp_matmul_rs::{
    sp_matmul_topn, CsrMatrix, CsrView, SortMode, TopNOptions,
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

fn view(m: &CsrMatrix<f64, i32>) -> CsrView<'_, f64, i32> {
    CsrView::new(m.nrows, m.ncols, &m.indptr, &m.indices, &m.data).unwrap()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[v.len() / 2]
}

fn bench(
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

fn compare(name: &str, a: &CsrMatrix<f64, i32>, b: &CsrMatrix<f64, i32>, top_n: usize, runs: usize, rb: usize, nt: Option<usize>) {
    let base_opts = TopNOptions {
        sort: SortMode::ByValueDesc,
        n_threads: nt,
        ..Default::default()
    };
    let tiled_opts = TopNOptions {
        tile_b: true,
        row_block: Some(rb),
        ..base_opts
    };
    let (c0, ms0) = bench(view(a), view(b), top_n, base_opts, runs);
    let (c1, ms1) = bench(view(a), view(b), top_n, tiled_opts, runs);
    assert_eq!(c0.indptr, c1.indptr, "{name}: indptr wijkt af");
    assert_eq!(c0.indices, c1.indices, "{name}: indices wijken af");
    assert!(
        c0.data.iter().zip(&c1.data).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{name}: data wijkt af"
    );
    println!(
        "{name:<34} base {ms0:>9.2} ms | tiled rb={rb} {ms1:>9.2} ms | ratio {:.3}",
        ms1 / ms0
    );
}

fn main() {
    let runs = env_usize("RUNS", 5);
    let rb = env_usize("RB", 2048);
    let nt = env_usize("NT", 10);

    // Notebookvorm: hoge update-dichtheid → dense pad.
    let a_nb = build_csr(42, 20_000, 10_000, 100);
    let b_nb = build_csr(43, 10_000, 20_000, 200);
    compare("notebook seq", &a_nb, &b_nb, 10, runs, rb, None);
    compare(&format!("notebook nt={nt}"), &a_nb, &b_nb, 10, runs, rb, Some(nt));

    // Tiny vorm: build-overhead mag de kleine call niet domineren.
    let a_t = build_csr(44, 1_000, 500, 5);
    let b_t = build_csr(45, 500, 1_000, 10);
    compare("tiny seq", &a_t, &b_t, 10, runs.max(50), rb, None);

    // Brede, dunne B (veel chunks, weinig nnz/B-rij): seg-lookup-overhead.
    let a_w = build_csr(46, 50_000, 5_000, 20);
    let b_w = build_csr(47, 5_000, 200_000, 30);
    compare("wijd-dun seq", &a_w, &b_w, 10, runs, rb, None);
}
