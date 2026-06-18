//! O6 A/B: dense-drain-scan varianten op de echte string_grouper-dumps.
//!
//! De fasemeting (phase_timing_tiled.rs, 2026-06-12) gaf: dense drain = 47%
//! van de seq-kernel op de sg-vorm. Kandidaten:
//!   - `neon8`/`neon16`  — NEON max-scan: per groep van 8/16 slots een
//!     vector-max vs `min`; alleen bij een hit scalair afhandelen. De scan is
//!     contigu/streaming, dus — anders dan de scatter (A2-falsificatie) —
//!     wél vectoriseerbaar.
//!   - `cmax`            — chunk-max early-out: de dense scatter houdt in
//!     registers een upper bound op alle geschreven sums bij (4 onafhankelijke
//!     fmaxnm-accumulatoren in de unrolled loop — gratis op een
//!     latency-bound loop, cf. de fmadd-les); is chunk_max ≤ min dan kan de
//!     hele scan vervallen (alleen de fill blijft). Bit-for-bit veilig:
//!     upper bound ≤ min ⇒ geen enkel slot zou pushen.
//!   - `cmax+neon8`      — combinatie (NEON-scan voor de niet-geskipte chunks).
//!
//! Elke variant wordt bit-for-bit geverifieerd tegen de productie-kernel.
//!
//! Run: cargo run --release --example dense_drain_ab
//! Env: RUNS (default 5), SAMPLE (default 0.05), SPANS (default 16),
//!      RB (default 2048)

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

/// Kopie van `chunked::scatter_dense_unrolled_local`.
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

/// Als `scatter_dense_unrolled_local`, maar houdt in 4 onafhankelijke
/// registers een running max van de geschreven waarden bij (fmaxnm — NaN
/// wordt genegeerd; productie pusht NaN toch nooit). Het resultaat is een
/// upper bound op de eindwaarden in dit segment (bij niet-negatieve data
/// exact): tussentijdse sums kunnen alleen hoger zijn dan de eindwaarde
/// als er negatieve producten volgen — skippen op de bound blijft correct.
#[inline(always)]
fn scatter_dense_unrolled_local_max(
    seg_idx: &[u16],
    seg_dat: &[f64],
    v: f64,
    sums: &mut [f64],
) -> f64 {
    let n = seg_idx.len();
    let mut s = 0;
    let mut m0 = f64::NEG_INFINITY;
    let mut m1 = f64::NEG_INFINITY;
    let mut m2 = f64::NEG_INFINITY;
    let mut m3 = f64::NEG_INFINITY;
    while s + 4 <= n {
        let k0 = seg_idx[s] as usize;
        let k1 = seg_idx[s + 1] as usize;
        let k2 = seg_idx[s + 2] as usize;
        let k3 = seg_idx[s + 3] as usize;
        let n0 = v.mul_add(seg_dat[s], sums[k0]);
        let n1 = v.mul_add(seg_dat[s + 1], sums[k1]);
        let n2 = v.mul_add(seg_dat[s + 2], sums[k2]);
        let n3 = v.mul_add(seg_dat[s + 3], sums[k3]);
        sums[k0] = n0;
        sums[k1] = n1;
        sums[k2] = n2;
        sums[k3] = n3;
        m0 = m0.max(n0);
        m1 = m1.max(n1);
        m2 = m2.max(n2);
        m3 = m3.max(n3);
        s += 4;
    }
    for t in s..n {
        let k_local = seg_idx[t] as usize;
        let nv = v.mul_add(seg_dat[t], sums[k_local]);
        sums[k_local] = nv;
        m0 = m0.max(nv);
    }
    m0.max(m1).max(m2.max(m3))
}

/// Productie max4-drain (kopie van het dense pad in `process_row_block`),
/// inclusief de fill-reset.
#[inline(always)]
fn drain_scan_max4(
    sums: &mut [f64],
    chunk_width: usize,
    c0: usize,
    mut min: f64,
    heap: &mut MaxHeap<f64, i32>,
) -> f64 {
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
                    min = heap.push_pop((c0 + k + off) as i32, s);
                }
            }
        }
        k += 4;
    }
    for (off, &s) in sums[w4..chunk_width].iter().enumerate() {
        if s != 0.0 && s > min {
            min = heap.push_pop((c0 + w4 + off) as i32, s);
        }
    }
    sums[..chunk_width].fill(0.0);
    min
}

/// NEON max-scan: per groep van `GROUP` (8/16) slots een vector-max; alleen
/// bij `!(groepsmax <= min)` scalair afhandelen (NaN-veilig: NaN ⇒ scan).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn drain_scan_neon<const GROUP: usize>(
    sums: &mut [f64],
    chunk_width: usize,
    c0: usize,
    mut min: f64,
    heap: &mut MaxHeap<f64, i32>,
) -> f64 {
    use std::arch::aarch64::*;
    let ptr = sums.as_ptr();
    let wg = chunk_width - chunk_width % GROUP;
    let mut k = 0;
    while k < wg {
        let mx = unsafe {
            let m01 = vmaxq_f64(vld1q_f64(ptr.add(k)), vld1q_f64(ptr.add(k + 2)));
            let m23 = vmaxq_f64(vld1q_f64(ptr.add(k + 4)), vld1q_f64(ptr.add(k + 6)));
            if GROUP == 8 {
                vmaxvq_f64(vmaxq_f64(m01, m23))
            } else {
                let m45 = vmaxq_f64(vld1q_f64(ptr.add(k + 8)), vld1q_f64(ptr.add(k + 10)));
                let m67 = vmaxq_f64(vld1q_f64(ptr.add(k + 12)), vld1q_f64(ptr.add(k + 14)));
                vmaxvq_f64(vmaxq_f64(vmaxq_f64(m01, m23), vmaxq_f64(m45, m67)))
            }
        };
        if !(mx <= min) {
            for off in 0..GROUP {
                let s = sums[k + off];
                if s != 0.0 && s > min {
                    min = heap.push_pop((c0 + k + off) as i32, s);
                }
            }
        }
        k += GROUP;
    }
    for (off, &s) in sums[wg..chunk_width].iter().enumerate() {
        if s != 0.0 && s > min {
            min = heap.push_pop((c0 + wg + off) as i32, s);
        }
    }
    sums[..chunk_width].fill(0.0);
    min
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn drain_scan_neon<const GROUP: usize>(
    sums: &mut [f64],
    chunk_width: usize,
    c0: usize,
    min: f64,
    heap: &mut MaxHeap<f64, i32>,
) -> f64 {
    drain_scan_max4(sums, chunk_width, c0, min, heap)
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Base,
    Neon8,
    Neon16,
    Cmax,
    CmaxNeon8,
}

impl Mode {
    fn track_max(self) -> bool {
        matches!(self, Mode::Cmax | Mode::CmaxNeon8)
    }
}

struct Counters {
    dense_chunks: u64,
    skipped: u64,
}

/// Mirror van `chunked::process_row_block` (tiled pad, adaptief dense/linked)
/// met een instelbare dense-drain-variant.
#[allow(clippy::too_many_arguments)]
fn kernel(
    a: &CsrMatrix<f64, i32>,
    b: &CsrMatrix<f64, i32>,
    tiled: &TiledB<f64>,
    top_n: usize,
    threshold: f64,
    chunk_cols: usize,
    row_block: usize,
    mode: Mode,
) -> (Vec<u32>, Vec<i32>, Vec<f64>, Counters) {
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
    let mut ctr = Counters {
        dense_chunks: 0,
        skipped: 0,
    };

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
                let a_start = a.indptr[i] as usize;
                let a_end = a.indptr[i + 1] as usize;
                let mut min = mins[r];

                if use_dense[r] {
                    ctr.dense_chunks += 1;
                    let mut chunk_max = f64::NEG_INFINITY;
                    if mode.track_max() {
                        for jj in a_start..a_end {
                            let j = a.indices[jj] as usize;
                            let (si, sd) = tiled.segment(chunk_id, j);
                            let m = scatter_dense_unrolled_local_max(si, sd, a.data[jj], &mut sums);
                            chunk_max = chunk_max.max(m);
                        }
                    } else {
                        for jj in a_start..a_end {
                            let j = a.indices[jj] as usize;
                            let (si, sd) = tiled.segment(chunk_id, j);
                            scatter_dense_unrolled_local(si, sd, a.data[jj], &mut sums);
                        }
                    }

                    if mode.track_max() && chunk_max <= min {
                        sums[..chunk_width].fill(0.0);
                        ctr.skipped += 1;
                    } else {
                        min = match mode {
                            Mode::Base | Mode::Cmax => {
                                drain_scan_max4(&mut sums, chunk_width, c0, min, &mut heaps[r])
                            }
                            Mode::Neon8 | Mode::CmaxNeon8 => {
                                drain_scan_neon::<8>(&mut sums, chunk_width, c0, min, &mut heaps[r])
                            }
                            Mode::Neon16 => {
                                drain_scan_neon::<16>(&mut sums, chunk_width, c0, min, &mut heaps[r])
                            }
                        };
                    }
                } else {
                    let mut head: u32 = HEAD_NIL;
                    let mut length: usize = 0;
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
                    }
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
                mins[r] = min;
            }
            c0 += chunk_width;
            chunk_id += 1;
        }

        for r in 0..block_rows {
            let heap = &mut heaps[r];
            heap.sort_by_value_desc();
            for entry in heap.entries() {
                c_indices.push(entry.idx);
                c_data.push(entry.val);
            }
            row_nset.push(heap.n_set() as u32);
        }
        row_lo = row_hi;
    }
    std::hint::black_box(&c_data);
    (row_nset, c_indices, c_data, ctr)
}

fn main() {
    let runs = env_usize("RUNS", 5);
    let spans = env_usize("SPANS", 16);
    let frac = env_f64("SAMPLE", 0.05);
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
    println!(
        "chunk_cols={chunk_cols} n_chunks={} row_block={rb} runs={runs}",
        ncols.div_ceil(chunk_cols)
    );

    let a_sub = sample_rows(&a, frac, spans);
    println!(
        "sample: {} rijen ({}%) in {spans} spans\n",
        a_sub.nrows,
        (100.0 * a_sub.nrows as f64 / a.nrows as f64).round()
    );

    let tiled = TiledB::<f64>::build(view(&b), chunk_cols);

    // Productiereferentie: timing-context + bit-for-bit-doel.
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
    let prod_nset: Vec<u32> = prod
        .indptr
        .windows(2)
        .map(|w| (w[1] - w[0]) as u32)
        .collect();
    println!("productie tiled+blocked: {prod_ms:.1} ms (mediaan van {runs})\n");

    let variants: Vec<(&str, Mode)> = vec![
        ("base (max4)", Mode::Base),
        ("neon8", Mode::Neon8),
        ("neon16", Mode::Neon16),
        ("cmax", Mode::Cmax),
        ("cmax+neon8", Mode::CmaxNeon8),
    ];

    let mut base_ms = 0.0f64;
    println!(
        "{:<14} {:>10} {:>9} {:>9} {:>12}",
        "variant (seq)", "ms", "vs base", "vs prod", "skip-rate"
    );
    for (name, mode) in &variants {
        // Warmup + verificatie.
        let (row_nset, c_indices, c_data, ctr) =
            kernel(&a_sub, &b, &tiled, top_n, threshold, chunk_cols, rb, *mode);
        assert_eq!(row_nset, prod_nset, "{name}: row_nset wijkt af");
        assert_eq!(c_indices, prod.indices, "{name}: indices wijken af");
        assert!(
            c_data.iter().zip(&prod.data).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{name}: data wijkt af"
        );

        let mut times = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            let out = kernel(&a_sub, &b, &tiled, top_n, threshold, chunk_cols, rb, *mode);
            times.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(&out);
        }
        let ms = median(times);
        if *mode == Mode::Base {
            base_ms = ms;
        }
        let skip = if mode.track_max() {
            format!("{:.1}%", 100.0 * ctr.skipped as f64 / ctr.dense_chunks.max(1) as f64)
        } else {
            "-".into()
        };
        println!(
            "{name:<14} {ms:>10.1} {:>9.3} {:>9.3} {:>12}",
            ms / base_ms,
            ms / prod_ms,
            skip
        );
    }

    let scale_full = a.nrows as f64 / a_sub.nrows as f64;
    println!(
        "\nextrapolatie volle A, seq (base): {:.1} s",
        base_ms / 1e3 * scale_full
    );
}
