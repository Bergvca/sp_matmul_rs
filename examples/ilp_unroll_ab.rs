//! Fase A2 (simd_test_plan): ILP-headroom test on the production dense-chunk
//! kernel (notebook shape, BinarySearch projection, dense accum).
//!
//! Variants:
//! - base          — production dense scatter + scalar drain scan
//! - unroll4       — scatter manually unrolled 4× (loads before stores; column
//!                   indices within one B-row segment are strictly increasing,
//!                   hence distinct, so the 4 iterations are independent).
//!                   Simulates the memory-level parallelism SIMD could add.
//! - prefetch      — O4: `prfm pldl1keep` on sums[k] 8 entries ahead
//! - drainmax4     — drain scan grouped 4-way via max(): one branch per 4 slots
//! - unroll4+dmax4 — both
//!
//! Decision rule (plan): unroll gain ≤ ~3% ⇒ SIMD has no headroom on aarch64;
//! ≥ ~10% ⇒ proceed to Fase B microbenches.
//! Run: cargo run --release --example ilp_unroll_ab

use std::time::Instant;

use sp_matmul_rs::maxheap::MaxHeap;
use sp_matmul_rs::CsrMatrix;

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

#[derive(Clone, Copy)]
enum ScatterKind {
    Base,
    Unroll4,
    Prefetch,
    Unroll4Prefetch,
}

#[derive(Clone, Copy)]
enum DrainKind {
    Scan,
    Max4,
}

#[inline(always)]
fn prefetch_sums(sums: &[f64], k: usize) {
    if k < sums.len() {
        unsafe {
            std::arch::asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) sums.as_ptr().add(k),
                options(nostack, preserves_flags, readonly)
            );
        }
    }
}

/// Dense-accum chunked kernel (BinarySearch projection), scatter/drain variants.
fn kernel(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    top_n: usize,
    chunk_cols: usize,
    scatter: ScatterKind,
    drain: DrainKind,
) -> (usize, f64) {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let mut sums: Vec<f64> = vec![0.0; chunk_cols];
    let mut heap = MaxHeap::<f64, i32>::new(top_n, f64::NEG_INFINITY);
    let mut c_indices: Vec<i32> = Vec::with_capacity(nrows * top_n);
    let mut c_data: Vec<f64> = Vec::with_capacity(nrows * top_n);
    let mut nnz = 0usize;

    let t_total = Instant::now();
    for i in 0..nrows {
        let mut min = heap.reset();
        let a_start = a.indptr[i] as usize;
        let a_end = a.indptr[i + 1] as usize;

        let mut c0 = 0usize;
        while c0 < ncols {
            let chunk_width = (ncols - c0).min(chunk_cols);
            let chunk_end = c0 + chunk_width;

            for jj in a_start..a_end {
                let j = a.indices[jj] as usize;
                let v = a.data[jj];
                let row_start = b.indptr[j] as usize;
                let row_end = b.indptr[j + 1] as usize;
                let row_b_idx = &b.indices[row_start..row_end];
                let lo = row_b_idx.partition_point(|x| (*x as usize) < c0);
                let hi = row_b_idx.partition_point(|x| (*x as usize) < chunk_end);
                let seg_idx = &row_b_idx[lo..hi];
                let seg_dat = &b.data[row_start + lo..row_start + hi];

                match scatter {
                    ScatterKind::Base => {
                        for (slot, &k_idx) in seg_idx.iter().enumerate() {
                            let k_local = k_idx as usize - c0;
                            sums[k_local] = v.mul_add(seg_dat[slot], sums[k_local]);
                        }
                    }
                    ScatterKind::Unroll4 => {
                        let n = seg_idx.len();
                        let mut s = 0usize;
                        while s + 4 <= n {
                            // Indices within one B-row segment are strictly
                            // increasing → distinct → iterations independent.
                            let k0 = seg_idx[s] as usize - c0;
                            let k1 = seg_idx[s + 1] as usize - c0;
                            let k2 = seg_idx[s + 2] as usize - c0;
                            let k3 = seg_idx[s + 3] as usize - c0;
                            let s0 = sums[k0];
                            let s1 = sums[k1];
                            let s2 = sums[k2];
                            let s3 = sums[k3];
                            sums[k0] = v.mul_add(seg_dat[s], s0);
                            sums[k1] = v.mul_add(seg_dat[s + 1], s1);
                            sums[k2] = v.mul_add(seg_dat[s + 2], s2);
                            sums[k3] = v.mul_add(seg_dat[s + 3], s3);
                            s += 4;
                        }
                        for t in s..n {
                            let k_local = seg_idx[t] as usize - c0;
                            sums[k_local] = v.mul_add(seg_dat[t], sums[k_local]);
                        }
                    }
                    ScatterKind::Prefetch => {
                        const DIST: usize = 8;
                        let n = seg_idx.len();
                        for slot in 0..n {
                            if slot + DIST < n {
                                prefetch_sums(&sums, seg_idx[slot + DIST] as usize - c0);
                            }
                            let k_local = seg_idx[slot] as usize - c0;
                            sums[k_local] = v.mul_add(seg_dat[slot], sums[k_local]);
                        }
                    }
                    ScatterKind::Unroll4Prefetch => {
                        const DIST: usize = 16;
                        let n = seg_idx.len();
                        let mut s = 0usize;
                        while s + 4 <= n {
                            if s + DIST < n {
                                prefetch_sums(&sums, seg_idx[s + DIST] as usize - c0);
                            }
                            let k0 = seg_idx[s] as usize - c0;
                            let k1 = seg_idx[s + 1] as usize - c0;
                            let k2 = seg_idx[s + 2] as usize - c0;
                            let k3 = seg_idx[s + 3] as usize - c0;
                            let s0 = sums[k0];
                            let s1 = sums[k1];
                            let s2 = sums[k2];
                            let s3 = sums[k3];
                            sums[k0] = v.mul_add(seg_dat[s], s0);
                            sums[k1] = v.mul_add(seg_dat[s + 1], s1);
                            sums[k2] = v.mul_add(seg_dat[s + 2], s2);
                            sums[k3] = v.mul_add(seg_dat[s + 3], s3);
                            s += 4;
                        }
                        for t in s..n {
                            let k_local = seg_idx[t] as usize - c0;
                            sums[k_local] = v.mul_add(seg_dat[t], sums[k_local]);
                        }
                    }
                }
            }

            match drain {
                DrainKind::Scan => {
                    for (k_local, &s) in sums[..chunk_width].iter().enumerate() {
                        if s != 0.0 && s > min {
                            min = heap.push_pop((c0 + k_local) as i32, s);
                        }
                    }
                }
                DrainKind::Max4 => {
                    let w4 = chunk_width & !3usize;
                    let mut k = 0usize;
                    while k < w4 {
                        let s0 = sums[k];
                        let s1 = sums[k + 1];
                        let s2 = sums[k + 2];
                        let s3 = sums[k + 3];
                        let m = s0.max(s1).max(s2.max(s3));
                        if m > min {
                            for (off, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                                if s != 0.0 && s > min {
                                    min = heap.push_pop((c0 + k + off) as i32, s);
                                }
                            }
                        }
                        k += 4;
                    }
                    for k in w4..chunk_width {
                        let s = sums[k];
                        if s != 0.0 && s > min {
                            min = heap.push_pop((c0 + k) as i32, s);
                        }
                    }
                }
            }
            sums[..chunk_width].fill(0.0);

            c0 += chunk_width;
        }

        heap.sort_by_value_desc();
        for entry in heap.entries() {
            c_indices.push(entry.idx);
            c_data.push(entry.val);
        }
        nnz += heap.n_set();
    }
    let total = t_total.elapsed().as_secs_f64();
    std::hint::black_box(&c_data);
    (nnz, total)
}

fn bench(
    label: &str,
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    scatter: ScatterKind,
    drain: DrainKind,
    runs: usize,
    base_ms: Option<f64>,
) -> f64 {
    let chunk_cols = 8192;
    let _ = kernel(a, b, 10, chunk_cols, scatter, drain);
    let mut times: Vec<f64> = (0..runs)
        .map(|_| {
            let (nnz, t) = kernel(a, b, 10, chunk_cols, scatter, drain);
            std::hint::black_box(nnz);
            t * 1e3
        })
        .collect();
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = times[times.len() / 2];
    match base_ms {
        Some(b0) => println!("{label:<16} {med:8.1} ms  (vs base: {:5.3})", med / b0),
        None => println!("{label:<16} {med:8.1} ms  (base)"),
    }
    med
}

fn main() {
    let runs: usize = std::env::var("RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let a = build_csr(0xA1A1, 20_000, 10_000, 100);
    let b = build_csr(0xB2B2, 10_000, 20_000, 200);

    println!("notebook shape, dense accum, BinarySearch projection, seq, median of {runs}");
    let base = bench("base", &a, &b, ScatterKind::Base, DrainKind::Scan, runs, None);
    bench("unroll4", &a, &b, ScatterKind::Unroll4, DrainKind::Scan, runs, Some(base));
    bench("prefetch8", &a, &b, ScatterKind::Prefetch, DrainKind::Scan, runs, Some(base));
    bench("drainmax4", &a, &b, ScatterKind::Base, DrainKind::Max4, runs, Some(base));
    bench("unroll4+dmax4", &a, &b, ScatterKind::Unroll4, DrainKind::Max4, runs, Some(base));
    bench("unr4+pf16+dmax4", &a, &b, ScatterKind::Unroll4Prefetch, DrainKind::Max4, runs, Some(base));
}
