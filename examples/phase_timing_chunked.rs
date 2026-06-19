//! Phase timing of the production *chunked* kernel (Fase A1, simd_test_plan).
//! Mirrors `chunked.rs` (safe indexing, u32 next[], dense/linked accum modes,
//! both projections) with per-phase timers. Rows are sampled (`sample` param)
//! so timer overhead stays negligible on fast shapes; reported phase times are
//! scaled estimates, the untimed total is the ground truth.
//! Run: cargo run --release --example phase_timing_chunked

use std::time::Instant;

use sp_matmul_rs::maxheap::MaxHeap;
use sp_matmul_rs::CsrMatrix;

const UNVISITED: u32 = u32::MAX;
const HEAD_NIL: u32 = u32::MAX - 1;

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

#[derive(Clone, Copy, PartialEq)]
enum Proj {
    BinarySearch,
    Cursor,
}

#[derive(Default)]
struct Phases {
    scatter: f64,
    drain: f64,
    out: f64,
    timed_rows: usize,
}

/// Chunked kernel mirroring `chunked::process_row`, with per-(row,chunk) phase
/// timers active on every `sample`-th row.
#[allow(clippy::too_many_arguments)]
fn kernel_timed(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
    chunk_cols: usize,
    proj: Proj,
    dense: bool,
    sample: usize,
) -> (usize, f64, Phases) {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let mut sums: Vec<f64> = vec![0.0; chunk_cols];
    let mut next: Vec<u32> = vec![UNVISITED; chunk_cols];
    let mut heap = MaxHeap::<f64, i32>::new(top_n, f64::NEG_INFINITY);
    let mut cursors: Vec<usize> = Vec::new();
    let mut c_indices: Vec<i32> = Vec::with_capacity(nrows * top_n);
    let mut c_data: Vec<f64> = Vec::with_capacity(nrows * top_n);
    let mut nnz = 0usize;
    let mut ph = Phases::default();

    let t_total = Instant::now();
    for i in 0..nrows {
        let timed = i % sample == 0;
        let mut min = heap.reset();
        let a_start = a.indptr[i] as usize;
        let a_end = a.indptr[i + 1] as usize;

        if proj == Proj::Cursor {
            cursors.clear();
            cursors.reserve(a_end - a_start);
            for jj in a_start..a_end {
                let j = a.indices[jj] as usize;
                cursors.push(b.indptr[j] as usize);
            }
        }

        let mut c0 = 0usize;
        while c0 < ncols {
            let chunk_width = (ncols - c0).min(chunk_cols);
            let chunk_end = c0 + chunk_width;
            let mut head: u32 = HEAD_NIL;
            let mut length: usize = 0;

            let t0 = timed.then(Instant::now);
            match (proj, dense) {
                (Proj::BinarySearch, false) => {
                    for jj in a_start..a_end {
                        let j = a.indices[jj] as usize;
                        let v = a.data[jj];
                        let row_start = b.indptr[j] as usize;
                        let row_end = b.indptr[j + 1] as usize;
                        let row_b_idx = &b.indices[row_start..row_end];
                        let lo = row_b_idx.partition_point(|x| (*x as usize) < c0);
                        let hi = row_b_idx.partition_point(|x| (*x as usize) < chunk_end);
                        for (slot, &k_idx) in row_b_idx[lo..hi].iter().enumerate() {
                            let off = lo + slot;
                            let k_local = k_idx as usize - c0;
                            sums[k_local] = v.mul_add(b.data[row_start + off], sums[k_local]);
                            if next[k_local] == UNVISITED {
                                next[k_local] = head;
                                head = k_local as u32;
                                length += 1;
                            }
                        }
                    }
                }
                (Proj::BinarySearch, true) => {
                    for jj in a_start..a_end {
                        let j = a.indices[jj] as usize;
                        let v = a.data[jj];
                        let row_start = b.indptr[j] as usize;
                        let row_end = b.indptr[j + 1] as usize;
                        let row_b_idx = &b.indices[row_start..row_end];
                        let lo = row_b_idx.partition_point(|x| (*x as usize) < c0);
                        let hi = row_b_idx.partition_point(|x| (*x as usize) < chunk_end);
                        for (slot, &k_idx) in row_b_idx[lo..hi].iter().enumerate() {
                            let off = lo + slot;
                            let k_local = k_idx as usize - c0;
                            sums[k_local] = v.mul_add(b.data[row_start + off], sums[k_local]);
                        }
                    }
                }
                (Proj::Cursor, false) => {
                    for (idx, jj) in (a_start..a_end).enumerate() {
                        let j = a.indices[jj] as usize;
                        let v = a.data[jj];
                        let stop_b = b.indptr[j + 1] as usize;
                        let mut cur = cursors[idx];
                        while cur < stop_b {
                            let k = b.indices[cur] as usize;
                            if k >= chunk_end {
                                break;
                            }
                            let k_local = k - c0;
                            sums[k_local] = v.mul_add(b.data[cur], sums[k_local]);
                            if next[k_local] == UNVISITED {
                                next[k_local] = head;
                                head = k_local as u32;
                                length += 1;
                            }
                            cur += 1;
                        }
                        cursors[idx] = cur;
                    }
                }
                (Proj::Cursor, true) => {
                    for (idx, jj) in (a_start..a_end).enumerate() {
                        let j = a.indices[jj] as usize;
                        let v = a.data[jj];
                        let stop_b = b.indptr[j + 1] as usize;
                        let mut cur = cursors[idx];
                        while cur < stop_b {
                            let k = b.indices[cur] as usize;
                            if k >= chunk_end {
                                break;
                            }
                            let k_local = k - c0;
                            sums[k_local] = v.mul_add(b.data[cur], sums[k_local]);
                            cur += 1;
                        }
                        cursors[idx] = cur;
                    }
                }
            }
            let t1 = timed.then(Instant::now);

            if dense {
                for (k_local, &s) in sums[..chunk_width].iter().enumerate() {
                    if s != 0.0 && s > min {
                        min = heap.push_pop((c0 + k_local) as i32, s);
                    }
                }
                sums[..chunk_width].fill(0.0);
            } else {
                for _ in 0..length {
                    let temp = head as usize;
                    if sums[temp] > min {
                        min = heap.push_pop((c0 + temp) as i32, sums[temp]);
                    }
                    head = next[temp];
                    next[temp] = UNVISITED;
                    sums[temp] = 0.0;
                }
            }

            if let (Some(t0), Some(t1)) = (t0, t1) {
                ph.scatter += (t1 - t0).as_secs_f64();
                ph.drain += t1.elapsed().as_secs_f64();
            }
            c0 += chunk_width;
        }

        let t2 = timed.then(Instant::now);
        heap.sort_by_value_desc();
        for entry in heap.entries() {
            c_indices.push(entry.idx);
            c_data.push(entry.val);
        }
        nnz += heap.n_set();
        if let Some(t2) = t2 {
            ph.out += t2.elapsed().as_secs_f64();
            ph.timed_rows += 1;
        }
    }
    let total = t_total.elapsed().as_secs_f64();
    std::hint::black_box(&c_data);
    (nnz, total, ph)
}

#[allow(clippy::too_many_arguments)]
fn report(
    label: &str,
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
    chunk_cols: usize,
    proj: Proj,
    dense: bool,
    sample: usize,
    runs: usize,
) {
    let _ = kernel_timed(a, b, top_n, chunk_cols, proj, dense, sample);
    for _ in 0..runs {
        let (nnz, total, ph) = kernel_timed(a, b, top_n, chunk_cols, proj, dense, sample);
        let scale = sample as f64;
        let (s, d, o) = (ph.scatter * scale, ph.drain * scale, ph.out * scale);
        let phase_sum = s + d + o;
        println!(
            "{label:<22} total={:7.1} ms | est: scatter={:7.1} ms ({:4.1}%)  drain={:7.1} ms ({:4.1}%)  sort+out={:5.1} ms ({:4.1}%) | phase-sum/total={:5.3}  nnz={nnz}",
            total * 1e3,
            s * 1e3,
            100.0 * s / phase_sum,
            d * 1e3,
            100.0 * d / phase_sum,
            o * 1e3,
            100.0 * o / phase_sum,
            phase_sum / total,
        );
    }
}

fn main() {
    let top_n = 10;
    // Production default on M5 Pro: L1d 128 KiB / 12 B = 8192 (next_pow2/2).
    let chunk_cols = 8192;

    // Notebook shape: BinarySearch + dense (adaptive picks dense at d=1.0).
    let a_nb = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b_nb = build_csr(0xB2B2, 10_000, 20_000, 200);
    println!("== notebook shape, dense accum (production path) ==");
    report(
        "notebook dense",
        &a_nb,
        &b_nb,
        top_n,
        chunk_cols,
        Proj::BinarySearch,
        true,
        1,
        3,
    );
    println!("== notebook shape, linked-list accum (old path, reference) ==");
    report(
        "notebook linked",
        &a_nb,
        &b_nb,
        top_n,
        chunk_cols,
        Proj::BinarySearch,
        false,
        1,
        3,
    );

    // TF-IDF shape: Cursor + linked list (adaptive picks linked at d=0.00032).
    // Sampled timing: 1 in 16 rows carries timers.
    let a_tf = build_csr(0xAAAA, 20_000, 50_000, 8);
    let b_tf = build_csr(0xBBBB, 50_000, 200_000, 8);
    println!("== TF-IDF shape, linked-list accum (production path), sampled 1/16 ==");
    report(
        "tfidf linked",
        &a_tf,
        &b_tf,
        top_n,
        chunk_cols,
        Proj::Cursor,
        false,
        16,
        3,
    );
}
