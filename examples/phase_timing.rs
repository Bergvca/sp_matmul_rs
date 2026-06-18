//! Phase-level timing of the sequential kernel: scatter vs drain vs heap/output.
//! Re-implements the single-chunk row kernel (C++-equivalent structure) with
//! per-phase accumulators. Run: cargo run --release --example phase_timing

use std::time::Instant;

use sp_matmul_rs::maxheap::MaxHeap;
use sp_matmul_rs::CsrMatrix;

const UNVISITED: usize = usize::MAX;
const HEAD_NIL: usize = usize::MAX - 1;

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

/// Single-chunk kernel, structurally identical to the C++ sp_matmul_topn,
/// with per-phase wall-clock accumulation (timer granularity: per row).
#[allow(clippy::too_many_arguments)]
fn kernel_timed(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
) -> (usize, f64, f64, f64) {
    let nrows = a.nrows;
    let ncols = b.ncols;
    const UNVISITED32: u32 = u32::MAX;
    const HEAD_NIL32: u32 = u32::MAX - 1;
    let mut sums: Vec<f64> = vec![0.0; ncols];
    let mut next: Vec<u32> = vec![UNVISITED32; ncols];
    let mut heap = MaxHeap::<f64, i32>::new(top_n, f64::MIN);
    let mut c_indices: Vec<i32> = Vec::with_capacity(nrows * top_n);
    let mut c_data: Vec<f64> = Vec::with_capacity(nrows * top_n);
    let mut nnz = 0usize;

    let (mut t_scatter, mut t_drain, mut t_out) = (0.0f64, 0.0, 0.0);

    for i in 0..nrows {
        let mut head: u32 = HEAD_NIL32;
        let mut length: usize = 0;
        let mut min = heap.reset();

        let t0 = Instant::now();
        let jj_start = a.indptr[i] as usize;
        let jj_end = a.indptr[i + 1] as usize;
        for jj in jj_start..jj_end {
            let j = a.indices[jj] as usize;
            let v = a.data[jj];
            let kk_start = b.indptr[j] as usize;
            let kk_end = b.indptr[j + 1] as usize;
            for kk in kk_start..kk_end {
                let k = b.indices[kk] as usize;
                unsafe {
                    *sums.get_unchecked_mut(k) += v * b.data.get_unchecked(kk);
                    let n = next.get_unchecked_mut(k);
                    if *n == UNVISITED32 {
                        *n = head;
                        head = k as u32;
                        length += 1;
                    }
                }
            }
        }
        let t1 = Instant::now();

        for _ in 0..length {
            unsafe {
                let temp = head as usize;
                let s = *sums.get_unchecked(temp);
                if s > min {
                    min = heap.push_pop(temp as i32, s);
                }
                head = *next.get_unchecked(temp);
                *next.get_unchecked_mut(temp) = UNVISITED32;
                *sums.get_unchecked_mut(temp) = 0.0;
            }
        }
        let t2 = Instant::now();

        heap.sort_by_value_desc();
        for entry in heap.entries() {
            c_indices.push(entry.idx);
            c_data.push(entry.val);
        }
        nnz += heap.n_set();
        let t3 = Instant::now();

        t_scatter += (t1 - t0).as_secs_f64();
        t_drain += (t2 - t1).as_secs_f64();
        t_out += (t3 - t2).as_secs_f64();
    }
    std::hint::black_box(&c_data);
    (nnz, t_scatter, t_drain, t_out)
}

fn main() {
    let a = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b = build_csr(0xB2B2, 10_000, 200_00 * 1, 200);

    // warmup
    let _ = kernel_timed(&a, &b, 10);
    for _ in 0..3 {
        let (nnz, ts, td, to) = kernel_timed(&a, &b, 10);
        println!(
            "nnz={nnz}  scatter={:7.1} ms  drain={:7.1} ms  sort+out={:6.1} ms  total={:7.1} ms",
            ts * 1e3,
            td * 1e3,
            to * 1e3,
            (ts + td + to) * 1e3
        );
    }
}
