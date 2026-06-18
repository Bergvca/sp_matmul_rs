//! Profiling target: run the sequential kernel in a loop so `sample` can
//! attach. Run: cargo run --release --example profile_target [chunk_cols]

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

fn main() {
    let chunk_cols: Option<usize> = std::env::args()
        .nth(1)
        .map(|s| s.parse().expect("chunk_cols must be a number"));
    let a = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b = build_csr(0xB2B2, 10_000, 20_000, 200);
    let opts = TopNOptions::<f64> {
        sort: SortMode::ByValueDesc,
        chunk_cols,
        ..Default::default()
    };
    println!("pid: {}", std::process::id());
    for _ in 0..10 {
        let av = CsrView::new(a.nrows, a.ncols, &a.indptr, &a.indices, &a.data).unwrap();
        let bv = CsrView::new(b.nrows, b.ncols, &b.indptr, &b.indices, &b.data).unwrap();
        let c = sp_matmul_topn(av, bv, 10, opts);
        std::hint::black_box(c.nnz());
    }
}
