//! Quick probe: effect of chunk_cols on the notebook benchmark shape
//! (A: 20000x10000, B: 10000x20000, density 1%, top_n=10).
//! Run: cargo run --release --features rayon --example chunk_probe

use std::time::Instant;

use sp_matmul_rs::{sp_matmul_topn, CsrMatrix, CsrView, SortMode, TopNOptions};

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

fn time_it(label: &str, a: &CsrMatrix<f64, i32>, b: &CsrMatrix<f64, i32>, opts: TopNOptions<f64>) {
    // warmup
    let _ = sp_matmul_topn(as_view(a), as_view(b), 10, opts);
    let reps = 3;
    let t0 = Instant::now();
    for _ in 0..reps {
        let c = sp_matmul_topn(as_view(a), as_view(b), 10, opts);
        std::hint::black_box(c.nnz());
    }
    let dt = t0.elapsed().as_secs_f64() / reps as f64;
    println!("{label:<46} {:>8.1} ms", dt * 1e3);
}

fn main() {
    println!(
        "default_chunk_cols::<f64>() = {}",
        sp_matmul_rs::default_chunk_cols::<f64>()
    );

    // Notebook shape: 1% density -> 100 nnz/row in A, 200 nnz/row in B.
    let a = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b = build_csr(0xB2B2, 10_000, 20_000, 200);

    let base = TopNOptions::<f64> {
        sort: SortMode::ByValueDesc,
        ..Default::default()
    };

    for nt in [None, Some(10usize)] {
        let nt_label = match nt {
            None => "seq",
            Some(n) => {
                println!();
                &format!("{n}thr")
            }
        };
        for (label, cc) in [
            ("default (None)", None),
            ("2048", Some(2048usize)),
            ("4096", Some(4096)),
            ("16384", Some(16_384)),
            ("single chunk (usize::MAX)", Some(usize::MAX)),
        ] {
            let opts = TopNOptions {
                chunk_cols: cc,
                n_threads: nt,
                ..base
            };
            time_it(&format!("[{nt_label}] chunk_cols={label}"), &a, &b, opts);
        }
    }
}
