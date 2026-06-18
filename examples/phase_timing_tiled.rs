//! Phase timing of the production *tiled+blocked* kernel on the real
//! string_grouper TF-IDF dumps (B5 go/no-go: how much of the kernel is the
//! linked-list drain now that O2+O3 are the default?).
//!
//! Mirrors `chunked::process_row_block` (tiled path, adaptive dense/linked
//! accumulation — on this shape ~2/3 of the rows pick the dense path!) with
//! per-(row,chunk) phase timers on every `TIMER_SAMPLE`-th row, accounted per
//! accumulation mode. Reported phase times are scaled estimates; the
//! production run on the same sample is the timer-free ground truth. The
//! mirror's output is checked bit-for-bit against the production kernel.
//!
//! Run: cargo run --release --example phase_timing_tiled
//! Env: RUNS (default 3), SAMPLE (default 0.05), SPANS (default 16),
//!      TIMER_SAMPLE (default 8), RB (default 2048)

use std::path::PathBuf;
use std::time::Instant;

use sp_matmul_rs::maxheap::MaxHeap;
use sp_matmul_rs::{
    default_chunk_cols, sp_matmul_topn, CsrMatrix, CsrView, SortMode, TiledB, TopNOptions,
};

const UNVISITED: u32 = u32::MAX;
const HEAD_NIL: u32 = u32::MAX - 1;
const DENSE_MIN_DENSITY: f64 = 0.2;

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

/// Kopie van `chunked::scatter_dense_unrolled_local` (tiled dense pad).
#[inline(always)]
fn scatter_dense_unrolled_local(seg_idx: &[u16], seg_dat: &[f64], v: f64, sums: &mut [f64]) {
    let n = seg_idx.len();
    let mut s = 0;
    while s + 4 <= n {
        let k0 = seg_idx[s] as usize;
        let k1 = seg_idx[s + 1] as usize;
        let k2 = seg_idx[s + 2] as usize;
        let k3 = seg_idx[s + 3] as usize;
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
        let k_local = seg_idx[t] as usize;
        sums[k_local] = v.mul_add(seg_dat[t], sums[k_local]);
    }
}

#[derive(Default)]
struct Phases {
    s_dense: f64,
    d_dense: f64,
    s_linked: f64,
    d_linked: f64,
    out: f64,
    timed_rows: usize,
    chunks_dense: u64,
    chunks_linked: u64,
    upd_dense: u64,
    upd_linked: u64,
    uni_linked: u64,
}

/// Mirror van `chunked::process_row_block` (tiled pad, adaptief dense/linked),
/// met per-(rij,chunk) fase-timers op elke `timer_sample`-de rij.
#[allow(clippy::too_many_arguments)]
fn kernel_timed(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    tiled: &TiledB<f64>,
    top_n: usize,
    threshold: f64,
    chunk_cols: usize,
    row_block: usize,
    timer_sample: usize,
) -> (Vec<u32>, Vec<i32>, Vec<f64>, f64, Phases) {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let mut sums: Vec<f64> = vec![0.0; chunk_cols];
    let mut next: Vec<u32> = vec![UNVISITED; chunk_cols];
    let mut heaps: Vec<MaxHeap<f64, i32>> = (0..row_block)
        .map(|_| MaxHeap::new(top_n, threshold))
        .collect();
    let mut mins: Vec<f64> = vec![0.0; row_block];
    let mut use_dense: Vec<bool> = vec![false; row_block];
    let mut row_nset: Vec<u32> = Vec::with_capacity(nrows);
    let mut c_indices: Vec<i32> = Vec::with_capacity(nrows * top_n);
    let mut c_data: Vec<f64> = Vec::with_capacity(nrows * top_n);
    let mut ph = Phases::default();

    let t_total = Instant::now();
    let mut row_lo = 0;
    while row_lo < nrows {
        let row_hi = (row_lo + row_block).min(nrows);
        let block_rows = row_hi - row_lo;
        for r in 0..block_rows {
            let i = row_lo + r;
            mins[r] = heaps[r].reset();
            let mut updates = 0usize;
            for jj in a.indptr[i] as usize..a.indptr[i + 1] as usize {
                let j = a.indices[jj] as usize;
                updates += (b.indptr[j + 1] - b.indptr[j]) as usize;
            }
            use_dense[r] = updates as f64 / ncols.max(1) as f64 >= DENSE_MIN_DENSITY;
        }

        let mut c0 = 0;
        let mut chunk_id = 0;
        while c0 < ncols {
            let chunk_width = (ncols - c0).min(chunk_cols);
            for r in 0..block_rows {
                let i = row_lo + r;
                let timed = i % timer_sample == 0;
                let a_start = a.indptr[i] as usize;
                let a_end = a.indptr[i + 1] as usize;
                let dense = use_dense[r];
                let mut min = mins[r];
                let mut head: u32 = HEAD_NIL;
                let mut length: usize = 0;

                let t0 = timed.then(Instant::now);
                if dense {
                    for jj in a_start..a_end {
                        let j = a.indices[jj] as usize;
                        let (si, sd) = tiled.segment(chunk_id, j);
                        scatter_dense_unrolled_local(si, sd, a.data[jj], &mut sums);
                        if timed {
                            ph.upd_dense += si.len() as u64;
                        }
                    }
                } else {
                    for jj in a_start..a_end {
                        let j = a.indices[jj] as usize;
                        let v = a.data[jj];
                        let (si, sd) = tiled.segment(chunk_id, j);
                        for (p, &kl) in si.iter().enumerate() {
                            let k_local = kl as usize;
                            sums[k_local] = v.mul_add(sd[p], sums[k_local]);
                            if next[k_local] == UNVISITED {
                                next[k_local] = head;
                                head = k_local as u32;
                                length += 1;
                            }
                        }
                        if timed {
                            ph.upd_linked += si.len() as u64;
                        }
                    }
                }
                let t1 = timed.then(Instant::now);

                if dense {
                    let w4 = chunk_width & !3;
                    let mut k = 0;
                    while k < w4 {
                        let s0 = sums[k];
                        let s1 = sums[k + 1];
                        let s2 = sums[k + 2];
                        let s3 = sums[k + 3];
                        let m = s0.max(s1).max(s2.max(s3));
                        if m > min {
                            for (off, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                                if s != 0.0 && s > min {
                                    min = heaps[r].push_pop((c0 + k + off) as i32, s);
                                }
                            }
                        }
                        k += 4;
                    }
                    for (off, &s) in sums[w4..chunk_width].iter().enumerate() {
                        if s != 0.0 && s > min {
                            min = heaps[r].push_pop((c0 + w4 + off) as i32, s);
                        }
                    }
                    sums[..chunk_width].fill(0.0);
                } else {
                    for _ in 0..length {
                        let temp = head as usize;
                        if sums[temp] > min {
                            min = heaps[r].push_pop((c0 + temp) as i32, sums[temp]);
                        }
                        head = next[temp];
                        next[temp] = UNVISITED;
                        sums[temp] = 0.0;
                    }
                }

                if let (Some(t0), Some(t1)) = (t0, t1) {
                    let (s_t, d_t) = ((t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
                    if dense {
                        ph.s_dense += s_t;
                        ph.d_dense += d_t;
                        ph.chunks_dense += 1;
                    } else {
                        ph.s_linked += s_t;
                        ph.d_linked += d_t;
                        ph.chunks_linked += 1;
                        ph.uni_linked += length as u64;
                    }
                }
                mins[r] = min;
            }
            c0 += chunk_width;
            chunk_id += 1;
        }

        for r in 0..block_rows {
            let i = row_lo + r;
            let t2 = (i % timer_sample == 0).then(Instant::now);
            let heap = &mut heaps[r];
            heap.sort_by_value_desc();
            for entry in heap.entries() {
                c_indices.push(entry.idx);
                c_data.push(entry.val);
            }
            row_nset.push(heap.n_set() as u32);
            if let Some(t2) = t2 {
                ph.out += t2.elapsed().as_secs_f64();
                ph.timed_rows += 1;
            }
        }
        row_lo = row_hi;
    }
    let total = t_total.elapsed().as_secs_f64();
    std::hint::black_box(&c_data);
    (row_nset, c_indices, c_data, total, ph)
}

fn main() {
    let runs = env_usize("RUNS", 3);
    let spans = env_usize("SPANS", 16);
    let frac = env_f64("SAMPLE", 0.05);
    let timer_sample = env_usize("TIMER_SAMPLE", 8);
    let rb = env_usize("RB", 2048);
    let top_n = 20usize;
    let threshold = 0.8f64;

    let b = load("B", usize::MAX);
    let a = load("A", b.nrows);
    let b = CsrMatrix { ncols: a.nrows, ..b };
    println!(
        "A {}x{} nnz {} | B {}x{} nnz {}",
        a.nrows, a.ncols, a.indptr.last().unwrap(),
        b.nrows, b.ncols, b.indptr.last().unwrap()
    );

    let ncols = b.ncols;
    let chunk_cols = default_chunk_cols::<f64>().min(ncols);
    let n_chunks = ncols.div_ceil(chunk_cols);
    println!("chunk_cols={chunk_cols} n_chunks={n_chunks} row_block={rb} timer_sample=1/{timer_sample}");

    let a_sub = sample_rows(&a, frac, spans);

    // Modusmix van het sample (adaptieve keuze per rij, zoals productie).
    let mut dense_rows = 0usize;
    for i in 0..a_sub.nrows {
        let mut upd = 0usize;
        for jj in a_sub.indptr[i] as usize..a_sub.indptr[i + 1] as usize {
            let j = a_sub.indices[jj] as usize;
            upd += (b.indptr[j + 1] - b.indptr[j]) as usize;
        }
        if upd as f64 / ncols as f64 >= DENSE_MIN_DENSITY {
            dense_rows += 1;
        }
    }
    println!(
        "sample: {} rijen ({}%) in {spans} spans, runs={runs} | modusmix: {dense_rows} dense ({:.0}%), {} linked",
        a_sub.nrows,
        (100.0 * a_sub.nrows as f64 / a.nrows as f64).round(),
        100.0 * dense_rows as f64 / a_sub.nrows as f64,
        a_sub.nrows - dense_rows
    );

    let t0 = Instant::now();
    let tiled = TiledB::<f64>::build(view(&b), chunk_cols);
    println!("TiledB::build: {:.1} ms\n", t0.elapsed().as_secs_f64() * 1e3);

    // Productiereferentie (zelfde sample, geforceerd tiled+blocked): timer-vrije
    // ground truth + bit-for-bit-doel voor de mirror.
    let opts = TopNOptions {
        threshold: Some(threshold),
        sort: SortMode::ByValueDesc,
        tile_b: true,
        row_block: Some(rb),
        ..Default::default()
    };
    let av = view(&a_sub);
    let bv = view(&b);
    let mut prod_times = Vec::with_capacity(runs);
    let mut prod_out = None;
    for _ in 0..runs {
        let t = Instant::now();
        let c = sp_matmul_topn(av, bv, top_n, opts);
        prod_times.push(t.elapsed().as_secs_f64() * 1e3);
        prod_out = Some(c);
    }
    let prod_ms = median(prod_times);
    let prod = prod_out.unwrap();
    println!("productie tiled+blocked (timer-vrij): {prod_ms:.1} ms (mediaan van {runs})\n");

    let _ = kernel_timed(&a_sub, &b, &tiled, top_n, threshold, chunk_cols, rb, timer_sample);
    for run in 0..runs {
        let (row_nset, c_indices, c_data, total, ph) =
            kernel_timed(&a_sub, &b, &tiled, top_n, threshold, chunk_cols, rb, timer_sample);

        // Bit-for-bit gelijk aan productie (zelfde accumulatievolgorde).
        let nset: Vec<u32> = prod
            .indptr
            .windows(2)
            .map(|w| (w[1] - w[0]) as u32)
            .collect();
        assert_eq!(row_nset, nset, "row_nset wijkt af van productie");
        assert_eq!(c_indices, prod.indices, "indices wijken af van productie");
        assert!(
            c_data.iter().zip(&prod.data).all(|(x, y)| x.to_bits() == y.to_bits()),
            "data wijkt af van productie"
        );

        let scale = timer_sample as f64;
        let (sd, dd, sl, dl, o) = (
            ph.s_dense * scale,
            ph.d_dense * scale,
            ph.s_linked * scale,
            ph.d_linked * scale,
            ph.out * scale,
        );
        let phase_sum = sd + dd + sl + dl + o;
        let pct = |x: f64| 100.0 * x / phase_sum;
        println!(
            "run #{}: total={:.1} ms | phase-sum/total={:.3} | vs prod={:.3}",
            run + 1,
            total * 1e3,
            phase_sum / total,
            total / (prod_ms / 1e3)
        );
        println!(
            "  dense : scatter {:7.1} ms ({:4.1}%)  drain {:7.1} ms ({:4.1}%)",
            sd * 1e3, pct(sd), dd * 1e3, pct(dd)
        );
        println!(
            "  linked: scatter {:7.1} ms ({:4.1}%)  drain {:7.1} ms ({:4.1}%)",
            sl * 1e3, pct(sl), dl * 1e3, pct(dl)
        );
        println!("  sort+out: {:.1} ms ({:.1}%)", o * 1e3, pct(o));
        if run == 0 {
            println!(
                "  diag dense : {:.0} updates per (rij,chunk); drain scant {} slots ⇒ {:.1} ns/slot",
                ph.upd_dense as f64 / ph.chunks_dense.max(1) as f64,
                chunk_cols,
                dd * 1e9 / (ph.chunks_dense as f64 * scale * chunk_cols as f64)
            );
            println!(
                "  diag linked: {:.1} updates / {:.1} uniek per (rij,chunk) (ratio {:.2}); drain ≈ {:.1} ns per uniek element",
                ph.upd_linked as f64 / ph.chunks_linked.max(1) as f64,
                ph.uni_linked as f64 / ph.chunks_linked.max(1) as f64,
                ph.upd_linked as f64 / ph.uni_linked.max(1) as f64,
                dl * 1e9 / (ph.uni_linked as f64 * scale)
            );
        }
    }

    let scale_full = a.nrows as f64 / a_sub.nrows as f64;
    println!(
        "\nextrapolatie volle A, seq: {:.1} s (productie-sample × {:.0})",
        prod_ms / 1e3 * scale_full,
        scale_full
    );
}
