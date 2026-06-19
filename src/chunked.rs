//! Column-chunked driver — the cache-blocking pillar.
//!
//! Splits B's column dimension
//! into tiles sized to fit L1d, processes one row of A across all column-chunks
//! while keeping a single per-row `MaxHeap` of size `top_n`. Avoids materialising
//! intermediate `C_j` chunks. Per-row scratch (`sums`, `next`) is sized to
//! `chunk_cols`, not `ncols` — this is the memory-footprint win.
//!
//! Two B-row projection strategies are implemented. Both produce
//! identical output and are selected internally via [`pick_projection`]:
//!
//! - [`BProjection::BinarySearch`] — `partition_point` per (A-entry × chunk).
//!   Cheap preprocessing, log-factor per chunk. Wins on dense B rows.
//! - [`BProjection::Cursor`] — per-A-entry cursor advanced across chunks. No log
//!   factor; each B-entry touched exactly once over the chunk sweep. Wins on
//!   sparse B rows / many chunks.

use std::cmp::Ordering;

use crate::csr::{matmul_topn_short_circuit, CsrMatrix, CsrView};
use crate::index::Index;
use crate::matmul_topn::{SortMode, TopNOptions};
use crate::maxheap::MaxHeap;
use crate::scalar::Scalar;
use crate::tiled::TiledB;

// u32 linked-list nodes keep the per-row scatter/drain working set half the
// size of usize nodes — the drain is a serial pointer chase, so whether
// `next[]` sits in L1d or L2 dominates kernel throughput. Local column
// offsets are < chunk_cols, which `resolve_chunk_cols` caps below the
// sentinels.
pub(crate) const UNVISITED: u32 = u32::MAX;
pub(crate) const HEAD_NIL: u32 = u32::MAX - 1;

/// L1d budget assumed when no cache size can be detected.
pub(crate) const FALLBACK_L1D_BYTES: usize = 64 * 1024;

/// Default column-chunk width derived from the L1d cache size.
///
/// Sizes the per-row scratch (`sums` + `next[]`) to roughly the L1d budget;
/// streamed B fragments, the heap, and cursors need little residency. The
/// L1d target is empirical: on Apple M-series the optimum chunk width put
/// scratch at ~L1d capacity, well below any L2-derived budget. Returns a
/// power of two within `[64, 1 << 20]`.
///
/// Detection: `sysctl` on macOS (the `cache-size` crate is CPUID-based and
/// returns `None` on Apple Silicon), `cache-size` elsewhere, 64 KiB fallback.
/// Callers can always override via `TopNOptions::chunk_cols`.
pub fn default_chunk_cols<V: Scalar>() -> usize {
    const FLOOR: usize = 64;
    const CEIL: usize = 1 << 20;

    let l1d = detect_l1d_bytes().unwrap_or(FALLBACK_L1D_BYTES);
    let per_col_bytes = std::mem::size_of::<V>() + std::mem::size_of::<u32>();
    let raw = (l1d / per_col_bytes.max(1)).max(1);
    let pow2 = raw.next_power_of_two() / 2;
    pow2.clamp(FLOOR, CEIL)
}

/// Detected L1d size in bytes, if available.
pub(crate) fn detect_l1d_bytes() -> Option<usize> {
    #[cfg(target_os = "macos")]
    if let Some(sz) = macos_sysctl_l1d() {
        return Some(sz);
    }
    cache_size::l1_cache_size()
}

/// L1d size via `sysctlbyname`. Prefers the performance-core value
/// (`hw.perflevel0.l1dcachesize`); the plain `hw.l1dcachesize` key reports
/// the smallest (efficiency-core) cache on heterogeneous chips.
#[cfg(target_os = "macos")]
fn macos_sysctl_l1d() -> Option<usize> {
    use std::ffi::CStr;
    use std::os::raw::{c_char, c_int, c_void};

    extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    fn get(name: &CStr) -> Option<usize> {
        let mut val: u64 = 0;
        let mut len: usize = std::mem::size_of::<u64>();
        // Writes 4 or 8 bytes depending on the key's C type; val is
        // zero-initialised and macOS is little-endian, so both are fine.
        let rc = unsafe {
            sysctlbyname(
                name.as_ptr(),
                &mut val as *mut u64 as *mut c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        (rc == 0 && (len == 4 || len == 8) && val > 0).then_some(val as usize)
    }

    let perf0 = CStr::from_bytes_with_nul(b"hw.perflevel0.l1dcachesize\0").unwrap();
    let plain = CStr::from_bytes_with_nul(b"hw.l1dcachesize\0").unwrap();
    get(perf0).or_else(|| get(plain))
}

/// Which B-row projection strategy to use inside the chunked kernel.
/// Picked automatically via [`pick_projection`]; overridable through
/// `TopNOptions::projection` (benchmarks / shape-specific tuning).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BProjection {
    BinarySearch,
    Cursor,
}

/// Accumulator bookkeeping mode for the chunked kernel (plan O1).
///
/// - `LinkedList` — the classic path: a `next[]` intrusive list records which
///   chunk columns were touched; the drain walks only those. Pays a
///   visited-check + branch per update and a pointer-chase per drain step.
/// - `Dense` — no bookkeeping: the scatter is a bare fused multiply-add and
///   the drain linearly scans the whole chunk, treating an exact-zero sum as
///   "untouched". Wins when a large fraction of each chunk is touched.
///   Float-only: integer dtypes silently fall back to `LinkedList` because a
///   genuine zero result (e.g. `2 - 2`) would otherwise be dropped. For
///   floats, dropping an exact-zero sum is the documented trade-off.
/// - `Adaptive` (default) — per A-row choice based on the expected update
///   density (`row nnz(A) × avg nnz(B-row) / ncols`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccumMode {
    #[default]
    Adaptive,
    LinkedList,
    Dense,
}

/// Update-density threshold above which `Adaptive` picks the dense mode.
/// Density `d` is expected updates per chunk column; the touched fraction is
/// `1 − e^(−d)`. Calibrated with the O1 ladder (examples/dense_mode_ab.rs,
/// M5 Pro, 2026-06-10). After the unrolled scatter + grouped drain the
/// measured crossover sits at d ≈ 0.13 (dense −3% at d=0.135, +34% at
/// d=0.065). 0.2 is deliberately conservative: just above the threshold
/// dense wins only a few percent, while below the crossover it loses tens
/// of percents — the asymmetry favours never landing in the regression
/// region over capturing the last ~3% in the d ∈ [0.13, 0.2) band.
pub(crate) const DENSE_MIN_DENSITY: f64 = 0.2;

/// Push one b-cursor per nonzero of A-row `i` — used by both the per-row and
/// the blocked kernels to seed the [`BProjection::Cursor`] scratch.
#[inline]
fn push_row_cursors<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    i: usize,
    cursors: &mut Vec<usize>,
) {
    for jj in a.indptr[i].to_usize()..a.indptr[i + 1].to_usize() {
        let j = a.indices[jj].to_usize();
        cursors.push(b.indptr[j].to_usize());
    }
}

/// Expected updates per output column for row `i` of A: exact per-row count
/// (sum of B-row nnz over the row's entries) divided by `ncols`. O(nnz(A-row));
/// the same `a.indices`/`b.indptr` reads happen again in the scatter, so this
/// pre-pass is cache-warm by construction.
fn row_update_density<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    i: usize,
) -> f64 {
    let start = a.indptr[i].to_usize();
    let end = a.indptr[i + 1].to_usize();
    let mut updates: usize = 0;
    for jj in start..end {
        let j = a.indices[jj].to_usize();
        updates += b.indptr[j + 1].to_usize() - b.indptr[j].to_usize();
    }
    updates as f64 / b.ncols.max(1) as f64
}

/// Pick the cheaper B-row projection strategy for the (A, B, chunk_cols) shape.
///
/// Heuristic: cursors win when avg(nnz(B-row)) is small relative to num_chunks
/// (linear walk beats many binary searches). Binary search wins when avg
/// B-row nnz is much larger than num_chunks (log factor cheaper than the walk).
/// The constant `4.0` is a placeholder; finalise from §8 bench data.
pub(crate) fn pick_projection<V: Scalar, I: Index>(
    _a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    chunk_cols: usize,
) -> BProjection {
    let cc = chunk_cols.max(1);
    let num_chunks = b.ncols.div_ceil(cc).max(1);
    let avg_nnz_b_row = if b.nrows == 0 {
        0.0
    } else {
        b.nnz() as f64 / b.nrows as f64
    };
    if avg_nnz_b_row >= (num_chunks as f64) * 4.0 {
        BProjection::BinarySearch
    } else {
        BProjection::Cursor
    }
}

/// Resolve a requested `chunk_cols` against `ncols` and the type default.
/// Capped below the u32 sentinels so local column offsets fit the `next[]`
/// node type for any chunk width.
pub(crate) fn resolve_chunk_cols<V: Scalar>(requested: Option<usize>, ncols: usize) -> usize {
    let raw = requested.unwrap_or_else(default_chunk_cols::<V>);
    assert!(raw > 0, "chunk_cols must be > 0");
    raw.min(ncols.max(1)).min((u32::MAX - 2) as usize)
}

/// Column-chunked sequential `sp_matmul_topn`.
///
/// Output is equivalent to the single-chunk path for all valid `chunk_cols`
/// (modulo dtype tolerance / integer tie-break rules at the top-n boundary —
/// see the parity comparator).
pub fn sp_matmul_topn_chunked<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
) -> CsrMatrix<V, I> {
    assert_eq!(
        a.ncols, b.nrows,
        "sp_matmul_topn_chunked: A.ncols ({}) must equal B.nrows ({})",
        a.ncols, b.nrows,
    );

    if let Some(out) = matmul_topn_short_circuit(a, b, top_n) {
        return out;
    }

    let chunk_cols = resolve_chunk_cols::<V>(opts.chunk_cols, b.ncols);
    let projection = opts
        .projection
        .unwrap_or_else(|| pick_projection(a, b, chunk_cols));
    dispatch::<V, I>(a, b, top_n, opts, chunk_cols, projection)
}

/// Row-block size for the auto-enabled tiled kernel. Swept on the
/// string_grouper TF-IDF shape and the notebook shape (examples/sg_ab.rs,
/// M5 Pro, 2026-06-12): gains flatten past 2048 sequentially and 2048 was
/// the multi-thread optimum (slices of ~nrows/4·n_threads rows clamp larger
/// values anyway).
pub(crate) const DEFAULT_ROW_BLOCK: usize = 2048;

/// Should the auto path use the tiled+blocked kernel (O3+O2)? Gates:
/// - ≥2 chunks — with a single chunk there is no cross-chunk reuse to win
///   and no projection overhead to remove;
/// - `TiledB` shape support (chunk-local `u16`, `u32` entry offsets);
/// - amortisation: the `O(nnz(B))` build must be small against the kernel
///   work, i.e. every B-row is re-walked often enough (`a.nnz() ≥ 2·nrows(B)`).
///   Deliberately mild: per entry the kernel (scattered fma + drain + heap)
///   measured ~25× the cost of the build's sequential counting sort, so even
///   at the gate boundary the build is a ~2% overhead against a >20% win;
/// - segment table stays proportional to B itself (`nseg ≤ 4·nnz(B)`), which
///   also rejects shapes whose mostly-empty segments would drag the kernel.
///
/// Measured (examples/tiled_default_ab.rs, sg_ab.rs): string_grouper shape
/// 0.72×, notebook shape 0.64×, wide-thin 0.98×; tiny calls (rejected here)
/// would pay ~8%.
pub(crate) fn auto_tile<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    chunk_cols: usize,
) -> bool {
    let n_chunks = b.ncols.div_ceil(chunk_cols.max(1));
    n_chunks >= 2
        && TiledB::supports(&b, chunk_cols)
        && a.nnz() >= 2 * b.nrows
        && n_chunks.saturating_mul(b.nrows) <= 4 * b.nnz()
}

/// Effective (build tiled?, row_block) pair. Explicit `tile_b`/`row_block`
/// options win; otherwise [`auto_tile`] decides. Shared by the sequential
/// dispatch and the parallel driver so both make the same choice.
pub(crate) fn resolve_blocking<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    chunk_cols: usize,
    opts: &TopNOptions<V>,
) -> (bool, Option<usize>) {
    if opts.row_block == Some(0) {
        // Explicit escape hatch: force the classic row-at-a-time kernel.
        (false, None)
    } else if opts.tile_b || opts.row_block.is_some() {
        (
            opts.tile_b && TiledB::supports(&b, chunk_cols),
            opts.row_block,
        )
    } else if auto_tile(a, b, chunk_cols) {
        (true, Some(DEFAULT_ROW_BLOCK))
    } else {
        (false, None)
    }
}

/// Route to the classic per-row driver or the blocked/tiled driver, based on
/// [`resolve_blocking`]. Shared by the sequential entry point and the
/// parallel driver's `n_threads <= 1` short-circuit.
pub(crate) fn dispatch<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
    chunk_cols: usize,
    projection: BProjection,
) -> CsrMatrix<V, I> {
    let (tile, row_block) = resolve_blocking(a, b, chunk_cols, &opts);
    if tile || row_block.is_some() {
        let tiled = tile.then(|| TiledB::build(b, chunk_cols));
        let opts = TopNOptions { row_block, ..opts };
        run_blocked(a, b, top_n, opts, chunk_cols, projection, tiled.as_ref())
    } else {
        run(a, b, top_n, opts, chunk_cols, projection)
    }
}

/// Inner driver with `projection` exposed — used by both the public entry point
/// and the chunked tests that pin a particular strategy.
pub(crate) fn run<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
    chunk_cols: usize,
    projection: BProjection,
) -> CsrMatrix<V, I> {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let threshold = opts.threshold.unwrap_or_else(V::min_value);
    let density = opts.density_hint.unwrap_or(1.0);
    let cap_hint = ((nrows as f64) * (top_n as f64) * density).ceil() as usize;

    let mut sums: Vec<V> = vec![V::default(); chunk_cols];
    let mut next: Vec<u32> = vec![UNVISITED; chunk_cols];
    let mut heap = MaxHeap::<V, I>::new(top_n, threshold);
    let mut c_indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    let mut c_indices: Vec<I> = Vec::with_capacity(cap_hint);
    let mut c_data: Vec<V> = Vec::with_capacity(cap_hint);
    c_indptr.push(I::zero());
    let mut nnz_total: usize = 0;

    let mut cursors: Vec<usize> = Vec::new();

    for i in 0..nrows {
        let n_set = process_row(
            a,
            b,
            i,
            opts.sort,
            chunk_cols,
            projection,
            opts.accum_mode,
            &mut sums,
            &mut next,
            &mut heap,
            &mut cursors,
            &mut c_indices,
            &mut c_data,
        );
        nnz_total += n_set;
        c_indptr.push(I::from_usize(nnz_total));
    }

    CsrMatrix {
        nrows,
        ncols,
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
    }
}

/// Scatter one B-row segment into `sums`, manually unrolled 4×. Column
/// indices within a segment are strictly increasing, hence distinct, so the
/// four load/fma/store groups are independent: the loads issue before the
/// stores, exposing memory-level parallelism the rolled loop serialises on
/// the load→fma→store chain. Alone worth ≤3% on Apple M5, but combined with
/// the grouped dense drain consistently the fastest variant (~7-9% total;
/// ~7-9% total speedup).
///
/// Returns the max of the values written, tracked in four independent
/// registers — free on this latency-bound loop. Accumulated per chunk it is
/// an upper bound on the final sums (intermediate values only exceed the
/// final value when later products are negative), so the drain may be
/// skipped entirely when the bound cannot beat the heap minimum (O6 in
/// ~99% skip rate on thresholded TF-IDF shapes).
/// Float `max` ignores NaN, which matches the drain: NaN sums never push.
#[inline(always)]
fn scatter_dense_unrolled<V: Scalar, I: Index>(
    seg_idx: &[I],
    seg_dat: &[V],
    v: V,
    c0: usize,
    sums: &mut [V],
) -> V {
    let n = seg_idx.len();
    let mut s = 0;
    let mut m0 = V::min_value();
    let mut m1 = V::min_value();
    let mut m2 = V::min_value();
    let mut m3 = V::min_value();
    while s + 4 <= n {
        let k0 = seg_idx[s].to_usize() - c0;
        let k1 = seg_idx[s + 1].to_usize() - c0;
        let k2 = seg_idx[s + 2].to_usize() - c0;
        let k3 = seg_idx[s + 3].to_usize() - c0;
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
        let k_local = seg_idx[t].to_usize() - c0;
        let nv = v.mul_add(seg_dat[t], sums[k_local]);
        sums[k_local] = nv;
        m0 = m0.max(nv);
    }
    m0.max(m1).max(m2.max(m3))
}

/// `scatter_dense_unrolled` for tiled segments: chunk-local `u16` indices,
/// no `c0` rebase. Same independence argument — indices within a segment are
/// strictly increasing — and the same returned max-of-written-sums.
#[inline(always)]
fn scatter_dense_unrolled_local<V: Scalar>(
    seg_idx: &[u16],
    seg_dat: &[V],
    v: V,
    sums: &mut [V],
) -> V {
    let n = seg_idx.len();
    let mut s = 0;
    let mut m0 = V::min_value();
    let mut m1 = V::min_value();
    let mut m2 = V::min_value();
    let mut m3 = V::min_value();
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

/// Drain the dense accumulator for one chunk into `heap` and reset `sums`.
///
/// Grouped four-wide via `max()` so the common "nothing beats the heap minimum"
/// case costs one branch per four slots instead of four; the whole scan is
/// skipped when `chunk_max` (the scatter's running max) cannot beat `min` (O6).
/// Float-only caller contract — exact-zero sums are treated as untouched, see
/// `AccumMode::Dense`. NaN sums never pass either compare. Reset via `fill`,
/// which lowers to memset. Returns the new heap minimum.
#[inline(always)]
fn drain_dense_chunk<V: Scalar, I: Index>(
    heap: &mut MaxHeap<V, I>,
    sums: &mut [V],
    mut min: V,
    chunk_max: V,
    c0: usize,
    chunk_width: usize,
) -> V {
    let zero = V::default();
    if chunk_max.partial_cmp(&min) == Some(Ordering::Greater) {
        let w4 = chunk_width & !3;
        let mut k = 0;
        while k < w4 {
            let s0 = sums[k];
            let s1 = sums[k + 1];
            let s2 = sums[k + 2];
            let s3 = sums[k + 3];
            let m = s0.max(s1).max(s2.max(s3));
            if m.partial_cmp(&min) == Some(Ordering::Greater) {
                for (off, s) in [s0, s1, s2, s3].into_iter().enumerate() {
                    if s != zero && s.partial_cmp(&min) == Some(Ordering::Greater) {
                        min = heap.push_pop(I::from_usize(c0 + k + off), s);
                    }
                }
            }
            k += 4;
        }
        for (off, &s) in sums[w4..chunk_width].iter().enumerate() {
            if s != zero && s.partial_cmp(&min) == Some(Ordering::Greater) {
                min = heap.push_pop(I::from_usize(c0 + w4 + off), s);
            }
        }
    }
    sums[..chunk_width].fill(zero);
    min
}

/// Drain the linked list of touched columns into `heap`, resetting `sums` and
/// `next` as we go. Returns the new heap minimum.
#[inline(always)]
fn drain_linked_chunk<V: Scalar, I: Index>(
    heap: &mut MaxHeap<V, I>,
    sums: &mut [V],
    next: &mut [u32],
    mut head: u32,
    length: usize,
    mut min: V,
    c0: usize,
) -> V {
    for _ in 0..length {
        let temp = head as usize;
        if sums[temp].partial_cmp(&min) == Some(Ordering::Greater) {
            min = heap.push_pop(I::from_usize(c0 + temp), sums[temp]);
        }
        head = next[temp];
        next[temp] = UNVISITED;
        sums[temp] = V::default();
    }
    min
}

/// Reusable scratch for [`process_row_block`]: one `sums`/`next` pair (drained
/// per (row, chunk), so shared across the block) plus per-row heap state that
/// must stay alive across the chunk sweep.
pub(crate) struct BlockScratch<V: Scalar, I: Index> {
    pub sums: Vec<V>,
    pub next: Vec<u32>,
    pub heaps: Vec<MaxHeap<V, I>>,
    pub mins: Vec<V>,
    pub use_dense: Vec<bool>,
    pub cursors: Vec<usize>,
    pub cursor_base: Vec<usize>,
}

impl<V: Scalar, I: Index> BlockScratch<V, I> {
    pub fn new(top_n: usize, threshold: V, chunk_cols: usize, block_rows: usize) -> Self {
        Self {
            sums: vec![V::default(); chunk_cols],
            next: vec![UNVISITED; chunk_cols],
            heaps: (0..block_rows)
                .map(|_| MaxHeap::new(top_n, threshold))
                .collect(),
            mins: vec![V::default(); block_rows],
            use_dense: vec![false; block_rows],
            cursors: Vec::new(),
            cursor_base: Vec::new(),
        }
    }
}

/// Chunk-major block kernel (plan O2/O3). Processes rows `[row_lo, row_hi)`
/// with the column-chunk loop outermost, so the B-fragment of one chunk stays
/// cache-hot across all rows of the block. With `tiled` set, B-segments are
/// streamed from the tiled-CSR buckets (no `partition_point`, no cursors).
///
/// Per-row accumulation order is identical to [`process_row`], so output is
/// bit-for-bit identical for every block size.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_row_block<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    row_lo: usize,
    row_hi: usize,
    sort: SortMode,
    chunk_cols: usize,
    projection: BProjection,
    accum: AccumMode,
    tiled: Option<&TiledB<V>>,
    scratch: &mut BlockScratch<V, I>,
    out_indices: &mut Vec<I>,
    out_data: &mut Vec<V>,
    row_nset: &mut Vec<u32>,
) {
    let ncols = b.ncols;
    let block_rows = row_hi - row_lo;
    debug_assert!(block_rows <= scratch.heaps.len());

    for r in 0..block_rows {
        let i = row_lo + r;
        scratch.mins[r] = scratch.heaps[r].reset();
        scratch.use_dense[r] = V::IS_FLOAT
            && match accum {
                AccumMode::LinkedList => false,
                AccumMode::Dense => true,
                AccumMode::Adaptive => row_update_density(a, b, i) >= DENSE_MIN_DENSITY,
            };
    }

    let need_cursors = tiled.is_none() && projection == BProjection::Cursor;
    scratch.cursors.clear();
    scratch.cursor_base.clear();
    if need_cursors {
        for i in row_lo..row_hi {
            scratch.cursor_base.push(scratch.cursors.len());
            push_row_cursors(a, b, i, &mut scratch.cursors);
        }
    }

    let sums = &mut scratch.sums[..];
    let next = &mut scratch.next[..];

    let mut c0 = 0;
    let mut chunk_id = 0;
    while c0 < ncols {
        let chunk_width = (ncols - c0).min(chunk_cols);
        let chunk_end = c0 + chunk_width;

        for r in 0..block_rows {
            let i = row_lo + r;
            let a_row_start = a.indptr[i].to_usize();
            let a_row_end = a.indptr[i + 1].to_usize();
            let use_dense = scratch.use_dense[r];
            let mut min = scratch.mins[r];
            let mut head: u32 = HEAD_NIL;
            let mut length: usize = 0;
            let mut chunk_max = V::min_value();

            if let Some(t) = tiled {
                if use_dense {
                    for jj in a_row_start..a_row_end {
                        let j = a.indices[jj].to_usize();
                        let (si, sd) = t.segment(chunk_id, j);
                        let m = scatter_dense_unrolled_local(si, sd, a.data[jj], sums);
                        chunk_max = chunk_max.max(m);
                    }
                } else {
                    for jj in a_row_start..a_row_end {
                        let j = a.indices[jj].to_usize();
                        let v = a.data[jj];
                        let (si, sd) = t.segment(chunk_id, j);
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
                }
            } else {
                match (projection, use_dense) {
                    (BProjection::BinarySearch, false) => {
                        for jj in a_row_start..a_row_end {
                            let j = a.indices[jj].to_usize();
                            let v = a.data[jj];
                            let row_start = b.indptr[j].to_usize();
                            let row_end = b.indptr[j + 1].to_usize();
                            let row_b_idx = &b.indices[row_start..row_end];
                            let lo = row_b_idx.partition_point(|x| Index::to_usize(*x) < c0);
                            let hi = row_b_idx.partition_point(|x| Index::to_usize(*x) < chunk_end);
                            for (slot, &k_idx) in row_b_idx[lo..hi].iter().enumerate() {
                                let off = lo + slot;
                                let k_local = k_idx.to_usize() - c0;
                                sums[k_local] = v.mul_add(b.data[row_start + off], sums[k_local]);
                                if next[k_local] == UNVISITED {
                                    next[k_local] = head;
                                    head = k_local as u32;
                                    length += 1;
                                }
                            }
                        }
                    }
                    (BProjection::BinarySearch, true) => {
                        for jj in a_row_start..a_row_end {
                            let j = a.indices[jj].to_usize();
                            let v = a.data[jj];
                            let row_start = b.indptr[j].to_usize();
                            let row_end = b.indptr[j + 1].to_usize();
                            let row_b_idx = &b.indices[row_start..row_end];
                            let lo = row_b_idx.partition_point(|x| Index::to_usize(*x) < c0);
                            let hi = row_b_idx.partition_point(|x| Index::to_usize(*x) < chunk_end);
                            let m = scatter_dense_unrolled(
                                &row_b_idx[lo..hi],
                                &b.data[row_start + lo..row_start + hi],
                                v,
                                c0,
                                sums,
                            );
                            chunk_max = chunk_max.max(m);
                        }
                    }
                    (BProjection::Cursor, false) => {
                        let base = scratch.cursor_base[r];
                        for (idx, jj) in (a_row_start..a_row_end).enumerate() {
                            let j = a.indices[jj].to_usize();
                            let v = a.data[jj];
                            let stop_b = b.indptr[j + 1].to_usize();
                            let mut cur = scratch.cursors[base + idx];
                            while cur < stop_b {
                                let k = b.indices[cur].to_usize();
                                if k >= chunk_end {
                                    break;
                                }
                                debug_assert!(k >= c0);
                                let k_local = k - c0;
                                sums[k_local] = v.mul_add(b.data[cur], sums[k_local]);
                                if next[k_local] == UNVISITED {
                                    next[k_local] = head;
                                    head = k_local as u32;
                                    length += 1;
                                }
                                cur += 1;
                            }
                            scratch.cursors[base + idx] = cur;
                        }
                    }
                    (BProjection::Cursor, true) => {
                        let base = scratch.cursor_base[r];
                        for (idx, jj) in (a_row_start..a_row_end).enumerate() {
                            let j = a.indices[jj].to_usize();
                            let v = a.data[jj];
                            let stop_b = b.indptr[j + 1].to_usize();
                            let cur = scratch.cursors[base + idx];
                            let seg_len = b.indices[cur..stop_b]
                                .partition_point(|x| Index::to_usize(*x) < chunk_end);
                            let m = scatter_dense_unrolled(
                                &b.indices[cur..cur + seg_len],
                                &b.data[cur..cur + seg_len],
                                v,
                                c0,
                                sums,
                            );
                            chunk_max = chunk_max.max(m);
                            scratch.cursors[base + idx] = cur + seg_len;
                        }
                    }
                }
            }

            min = if use_dense {
                drain_dense_chunk(&mut scratch.heaps[r], sums, min, chunk_max, c0, chunk_width)
            } else {
                drain_linked_chunk(&mut scratch.heaps[r], sums, next, head, length, min, c0)
            };

            scratch.mins[r] = min;
        }

        c0 = chunk_end;
        chunk_id += 1;
    }

    for r in 0..block_rows {
        let heap = &mut scratch.heaps[r];
        match sort {
            SortMode::ByColumn => heap.sort_by_insertion_order(),
            SortMode::ByValueDesc => heap.sort_by_value_desc(),
        }
        let n_set = heap.n_set();
        for entry in heap.entries() {
            out_indices.push(entry.idx);
            out_data.push(entry.val);
        }
        row_nset.push(n_set as u32);
    }
}

/// Sequential driver over [`process_row_block`] — `row_block` rows per block
/// (default 1), optionally streaming B from a [`TiledB`].
pub(crate) fn run_blocked<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
    chunk_cols: usize,
    projection: BProjection,
    tiled: Option<&TiledB<V>>,
) -> CsrMatrix<V, I> {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let threshold = opts.threshold.unwrap_or_else(V::min_value);
    let density = opts.density_hint.unwrap_or(1.0);
    let cap_hint = ((nrows as f64) * (top_n as f64) * density).ceil() as usize;

    let block_rows = opts.row_block.unwrap_or(1).max(1).min(nrows.max(1));
    let mut scratch = BlockScratch::<V, I>::new(top_n, threshold, chunk_cols, block_rows);
    let mut row_nset: Vec<u32> = Vec::with_capacity(nrows);
    let mut c_indices: Vec<I> = Vec::with_capacity(cap_hint);
    let mut c_data: Vec<V> = Vec::with_capacity(cap_hint);

    let mut lo = 0;
    while lo < nrows {
        let hi = (lo + block_rows).min(nrows);
        process_row_block(
            a,
            b,
            lo,
            hi,
            opts.sort,
            chunk_cols,
            projection,
            opts.accum_mode,
            tiled,
            &mut scratch,
            &mut c_indices,
            &mut c_data,
            &mut row_nset,
        );
        lo = hi;
    }

    let mut c_indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    c_indptr.push(I::zero());
    let mut running: usize = 0;
    for &n in &row_nset {
        running += n as usize;
        c_indptr.push(I::from_usize(running));
    }

    CsrMatrix {
        nrows,
        ncols,
        indptr: c_indptr,
        indices: c_indices,
        data: c_data,
    }
}

/// Per-row body of the chunked driver. Walks the column-chunks of B for row
/// `i` of A, fills `heap`, then appends the row's entries to `out_indices` /
/// `out_data`. Returns the number of entries appended.
///
/// Shared between [`run`] (sequential) and `parallel::run_row_slice`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_row<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    i: usize,
    sort: SortMode,
    chunk_cols: usize,
    projection: BProjection,
    accum: AccumMode,
    sums: &mut [V],
    next: &mut [u32],
    heap: &mut MaxHeap<V, I>,
    cursors: &mut Vec<usize>,
    out_indices: &mut Vec<I>,
    out_data: &mut Vec<V>,
) -> usize {
    let ncols = b.ncols;
    let mut min = heap.reset();
    let a_row_start = a.indptr[i].to_usize();
    let a_row_end = a.indptr[i + 1].to_usize();

    let use_dense = V::IS_FLOAT
        && match accum {
            AccumMode::LinkedList => false,
            AccumMode::Dense => true,
            AccumMode::Adaptive => row_update_density(a, b, i) >= DENSE_MIN_DENSITY,
        };

    if projection == BProjection::Cursor {
        cursors.clear();
        cursors.reserve(a_row_end - a_row_start);
        push_row_cursors(a, b, i, cursors);
    }

    let mut c0 = 0;
    while c0 < ncols {
        let chunk_width = (ncols - c0).min(chunk_cols);
        let chunk_end = c0 + chunk_width;
        let mut head: u32 = HEAD_NIL;
        let mut length: usize = 0;
        let mut chunk_max = V::min_value();

        match (projection, use_dense) {
            (BProjection::BinarySearch, false) => {
                for jj in a_row_start..a_row_end {
                    let j = a.indices[jj].to_usize();
                    let v = a.data[jj];
                    let row_start = b.indptr[j].to_usize();
                    let row_end = b.indptr[j + 1].to_usize();
                    let row_b_idx = &b.indices[row_start..row_end];
                    let lo = row_b_idx.partition_point(|x| Index::to_usize(*x) < c0);
                    let hi = row_b_idx.partition_point(|x| Index::to_usize(*x) < chunk_end);
                    for (slot, &k_idx) in row_b_idx[lo..hi].iter().enumerate() {
                        let off = lo + slot;
                        let k_local = k_idx.to_usize() - c0;
                        sums[k_local] = v.mul_add(b.data[row_start + off], sums[k_local]);
                        if next[k_local] == UNVISITED {
                            next[k_local] = head;
                            head = k_local as u32;
                            length += 1;
                        }
                    }
                }
            }
            (BProjection::BinarySearch, true) => {
                for jj in a_row_start..a_row_end {
                    let j = a.indices[jj].to_usize();
                    let v = a.data[jj];
                    let row_start = b.indptr[j].to_usize();
                    let row_end = b.indptr[j + 1].to_usize();
                    let row_b_idx = &b.indices[row_start..row_end];
                    let lo = row_b_idx.partition_point(|x| Index::to_usize(*x) < c0);
                    let hi = row_b_idx.partition_point(|x| Index::to_usize(*x) < chunk_end);
                    let m = scatter_dense_unrolled(
                        &row_b_idx[lo..hi],
                        &b.data[row_start + lo..row_start + hi],
                        v,
                        c0,
                        sums,
                    );
                    chunk_max = chunk_max.max(m);
                }
            }
            (BProjection::Cursor, false) => {
                for (idx, jj) in (a_row_start..a_row_end).enumerate() {
                    let j = a.indices[jj].to_usize();
                    let v = a.data[jj];
                    let stop_b = b.indptr[j + 1].to_usize();
                    let mut cur = cursors[idx];
                    while cur < stop_b {
                        let k = b.indices[cur].to_usize();
                        if k >= chunk_end {
                            break;
                        }
                        // Invariant: cur was advanced past prior chunk_end, so k >= c0.
                        debug_assert!(k >= c0);
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
            (BProjection::Cursor, true) => {
                for (idx, jj) in (a_row_start..a_row_end).enumerate() {
                    let j = a.indices[jj].to_usize();
                    let v = a.data[jj];
                    let stop_b = b.indptr[j + 1].to_usize();
                    let cur = cursors[idx];
                    // Invariant: cur was advanced past prior chunk_end, so all
                    // remaining indices are >= c0.
                    let seg_len =
                        b.indices[cur..stop_b].partition_point(|x| Index::to_usize(*x) < chunk_end);
                    let m = scatter_dense_unrolled(
                        &b.indices[cur..cur + seg_len],
                        &b.data[cur..cur + seg_len],
                        v,
                        c0,
                        sums,
                    );
                    chunk_max = chunk_max.max(m);
                    cursors[idx] = cur + seg_len;
                }
            }
        }

        min = if use_dense {
            drain_dense_chunk(heap, sums, min, chunk_max, c0, chunk_width)
        } else {
            drain_linked_chunk(heap, sums, next, head, length, min, c0)
        };

        c0 += chunk_width;
    }

    match sort {
        SortMode::ByColumn => heap.sort_by_insertion_order(),
        SortMode::ByValueDesc => heap.sort_by_value_desc(),
    }
    let n_set = heap.n_set();
    for entry in heap.entries() {
        out_indices.push(entry.idx);
        out_data.push(entry.val);
    }
    n_set
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrView;

    fn default_is_power_of_two_in_range<V: Scalar>() {
        let n = default_chunk_cols::<V>();
        assert!(n >= 64, "default {n} < floor");
        assert!(n <= (1 << 20), "default {n} > ceil");
        assert!(n.is_power_of_two(), "default {n} not a power of two");
    }

    #[test]
    fn default_chunk_cols_f32() {
        default_is_power_of_two_in_range::<f32>();
    }
    #[test]
    fn default_chunk_cols_f64() {
        default_is_power_of_two_in_range::<f64>();
    }
    #[test]
    fn default_chunk_cols_i32() {
        default_is_power_of_two_in_range::<i32>();
    }
    #[test]
    fn default_chunk_cols_i64() {
        default_is_power_of_two_in_range::<i64>();
    }

    type CsrParts = (Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<f64>);

    /// Same small fixture as `matmul_topn`'s unit tests.
    /// A (2×3): [[1, 0, 2], [0, 3, 4]]
    /// B (3×4): [[1, 0, 0, 5], [0, 1, 0, 6], [0, 0, 1, 7]]
    /// C = A·B:
    /// row 0: [1, 0, 2, 5+14] = [1, 0, 2, 19]
    /// row 1: [0, 3, 4, 18+28] = [0, 3, 4, 46]
    fn make_a_b() -> CsrParts {
        let a_indptr = vec![0i32, 2, 4];
        let a_indices = vec![0i32, 2, 1, 2];
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_indptr = vec![0i32, 2, 4, 6];
        let b_indices = vec![0i32, 3, 1, 3, 2, 3];
        let b_data = vec![1.0f64, 5.0, 1.0, 6.0, 1.0, 7.0];
        (a_indptr, a_indices, a_data, b_indptr, b_indices, b_data)
    }

    fn rows_sorted(c: &CsrMatrix<f64, i32>) -> Vec<Vec<(i32, f64)>> {
        (0..c.nrows)
            .map(|i| {
                let s = c.indptr[i] as usize;
                let e = c.indptr[i + 1] as usize;
                let mut row: Vec<(i32, f64)> = (s..e).map(|k| (c.indices[k], c.data[k])).collect();
                row.sort_by_key(|(idx, _)| *idx);
                row
            })
            .collect()
    }

    /// For each strategy, run the chunked driver at the given width and return
    /// the row-sorted (idx, val) lists.
    #[allow(clippy::type_complexity)]
    fn run_both(
        a: CsrView<'_, f64, i32>,
        b: CsrView<'_, f64, i32>,
        top_n: usize,
        opts: TopNOptions<f64>,
        chunk_cols: usize,
    ) -> (Vec<Vec<(i32, f64)>>, Vec<Vec<(i32, f64)>>) {
        let cc = chunk_cols.min(b.ncols.max(1));
        let c_bs = run::<f64, i32>(a, b, top_n, opts, cc, BProjection::BinarySearch);
        let c_cu = run::<f64, i32>(a, b, top_n, opts, cc, BProjection::Cursor);
        (rows_sorted(&c_bs), rows_sorted(&c_cu))
    }

    /// chunk_cols >= ncols collapses to one chunk; must match single-chunk semantics.
    #[test]
    fn one_big_chunk_matches_known_product() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let (bs, cu) = run_both(a, b, 2, opts, 4);
        // row 0 top-2: (3, 19) (2, 2); row 1 top-2: (3, 46) (2, 4)
        let expected = vec![vec![(2, 2.0), (3, 19.0)], vec![(2, 4.0), (3, 46.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// chunk_cols == 1 produces ncols-many chunks; output must still match.
    #[test]
    fn one_col_chunks_match() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByColumn,
            ..Default::default()
        };
        let (bs, cu) = run_both(a, b, 4, opts, 1);
        // top_n=4 keeps all nonzeros; sorted by column.
        let expected = vec![
            vec![(0, 1.0), (2, 2.0), (3, 19.0)],
            vec![(1, 3.0), (2, 4.0), (3, 46.0)],
        ];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// chunk_cols that does not divide ncols — the last chunk is short.
    #[test]
    fn non_divisor_chunk_width() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        // ncols = 4; chunk_cols = 3 → two chunks: [0..3), [3..4)
        let (bs, cu) = run_both(a, b, 4, opts, 3);
        let expected = vec![
            vec![(0, 1.0), (2, 2.0), (3, 19.0)],
            vec![(1, 3.0), (2, 4.0), (3, 46.0)],
        ];
        // Sort the two-chunk output by column (ByValueDesc gave us value order; resort here)
        let resort = |rs: Vec<Vec<(i32, f64)>>| -> Vec<Vec<(i32, f64)>> {
            rs.into_iter()
                .map(|mut r| {
                    r.sort_by_key(|(idx, _)| *idx);
                    r
                })
                .collect()
        };
        assert_eq!(resort(bs), expected);
        assert_eq!(resort(cu), expected);
    }

    /// All A-row entries land in one chunk — most chunks are empty for the row.
    #[test]
    fn most_chunks_empty_for_row() {
        // A is 1×10 with a single entry at col 3. B is 10×16 with B[3, :] only
        // populating columns 5 and 7. Both fall into the [4..8) chunk if width=4.
        let a_indptr = vec![0i32, 1];
        let a_indices = vec![3i32];
        let a_data = vec![2.0f64];
        let b_indptr = vec![0i32, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2];
        let b_indices = vec![5i32, 7];
        let b_data = vec![3.0f64, 5.0];
        let a = CsrView::new(1, 10, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(10, 16, &b_indptr, &b_indices, &b_data).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByColumn,
            ..Default::default()
        };
        // width=4 → 4 chunks: [0..4) [4..8) [8..12) [12..16). Only chunk 1 has work.
        let (bs, cu) = run_both(a, b, 4, opts, 4);
        let expected = vec![vec![(5, 6.0), (7, 10.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// A value found in a later chunk must displace an earlier-chunk value only
    /// if strictly greater. Tests the cross-chunk threshold accumulation.
    #[test]
    fn later_chunk_displaces_earlier() {
        // A: 1×3, all ones. B: 3×6, each B[j, :] places one value in different columns.
        // Columns: 0→2, 1→4, 2→1. With chunk_cols=2: chunks [0..2) [2..4) [4..6).
        // Output sums: col 0 = 2, col 1 = 1 (from B[2,1]), col 2 = ? (nothing from B[*,2])
        // Let's build it directly: result row = sum over j of A[0,j] * B[j,:]
        // Use values such that across-chunk displacement is observable.
        let a_indptr = vec![0i32, 3];
        let a_indices = vec![0i32, 1, 2];
        let a_data = vec![1.0f64, 1.0, 1.0];
        // B row 0 → col 0 with 5, col 5 with 1
        // B row 1 → col 2 with 7
        // B row 2 → col 4 with 3
        let b_indptr = vec![0i32, 2, 3, 4];
        let b_indices = vec![0i32, 5, 2, 4];
        let b_data = vec![5.0f64, 1.0, 7.0, 3.0];
        let a = CsrView::new(1, 3, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(3, 6, &b_indptr, &b_indices, &b_data).unwrap();
        // Expected dense product row 0: cols {0:5, 2:7, 4:3, 5:1}. Top-2 by value: 7, 5.
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let (bs, cu) = run_both(a, b, 2, opts, 2);
        // Heap top-2 contains (col 0, 5.0) and (col 2, 7.0); `rows_sorted` is column-sorted.
        let expected = vec![vec![(0, 5.0), (2, 7.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// top_n exactly fills with chunk 1; chunk 2 must not displace equal values.
    /// Regression guard for `>` (strict) in the heap push guard.
    #[test]
    fn equal_value_does_not_displace() {
        // A: 1×2 all ones. B: 2×4. row 0 → col 0 with 3 and col 1 with 3.
        // row 1 → col 3 with 3 (equal). With chunk_cols=2 and top_n=2:
        // chunk 1 [0..2): heap = {(0,3), (1,3)}, min = 3.
        // chunk 2 [2..4): col 3 with 3 — must not displace (3 is not > 3).
        let a_indptr = vec![0i32, 2];
        let a_indices = vec![0i32, 1];
        let a_data = vec![1.0f64, 1.0];
        let b_indptr = vec![0i32, 2, 3];
        let b_indices = vec![0i32, 1, 3];
        let b_data = vec![3.0f64, 3.0, 3.0];
        let a = CsrView::new(1, 2, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(2, 4, &b_indptr, &b_indices, &b_data).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByColumn,
            ..Default::default()
        };
        let (bs, cu) = run_both(a, b, 2, opts, 2);
        // Heap kept the first two equal values inserted (cols 0 and 1). Sort by col.
        let expected = vec![vec![(0, 3.0), (1, 3.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// Threshold pruning at the original (per-row) baseline works across chunks.
    #[test]
    fn threshold_filters_across_chunks() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            threshold: Some(10.0),
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        // width = 2 → two chunks; only 19 (row 0) and 46 (row 1) survive threshold 10.
        let (bs, cu) = run_both(a, b, 4, opts, 2);
        let expected = vec![vec![(3, 19.0)], vec![(3, 46.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// top_n smaller than per-chunk nonzeros — heap evicts within a chunk.
    #[test]
    fn topn_smaller_than_per_chunk_nnz() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        // top_n = 1 → just the largest value per row.
        let (bs, cu) = run_both(a, b, 1, opts, 4);
        let expected = vec![vec![(3, 19.0)], vec![(3, 46.0)]];
        assert_eq!(bs, expected);
        assert_eq!(cu, expected);
    }

    /// Forced Dense and forced LinkedList agree on the fixture for every
    /// chunk width × projection, for float values without zero sums or ties.
    #[test]
    fn dense_matches_linked_list() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        for chunk_cols in [1usize, 2, 3, 4, 8] {
            for projection in [BProjection::BinarySearch, BProjection::Cursor] {
                for top_n in [1usize, 2, 4] {
                    let mk = |accum_mode| TopNOptions::<f64> {
                        sort: SortMode::ByColumn,
                        accum_mode,
                        ..Default::default()
                    };
                    let dense =
                        run::<f64, i32>(a, b, top_n, mk(AccumMode::Dense), chunk_cols, projection);
                    let linked = run::<f64, i32>(
                        a,
                        b,
                        top_n,
                        mk(AccumMode::LinkedList),
                        chunk_cols,
                        projection,
                    );
                    assert_eq!(
                        rows_sorted(&dense),
                        rows_sorted(&linked),
                        "chunk_cols={chunk_cols} projection={projection:?} top_n={top_n}"
                    );
                }
            }
        }
    }

    /// Documented dense-mode caveat: a column whose products cancel to exactly
    /// 0.0 is treated as untouched and dropped; the linked-list path keeps it.
    #[test]
    fn dense_drops_exact_zero_sum_floats() {
        // A = [1, 1]; B row 0 = {0: 1.0, 1: 5.0}, row 1 = {0: -1.0, 2: 3.0}.
        // Product row: col 0 = 0.0 (cancel), col 1 = 5.0, col 2 = 3.0.
        let a_indptr = vec![0i32, 2];
        let a_indices = vec![0i32, 1];
        let a_data = vec![1.0f64, 1.0];
        let b_indptr = vec![0i32, 2, 4];
        let b_indices = vec![0i32, 1, 0, 2];
        let b_data = vec![1.0f64, 5.0, -1.0, 3.0];
        let a = CsrView::new(1, 2, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(2, 3, &b_indptr, &b_indices, &b_data).unwrap();
        let mk = |accum_mode| TopNOptions::<f64> {
            sort: SortMode::ByColumn,
            accum_mode,
            ..Default::default()
        };
        let linked = run::<f64, i32>(a, b, 3, mk(AccumMode::LinkedList), 4, BProjection::Cursor);
        let dense = run::<f64, i32>(a, b, 3, mk(AccumMode::Dense), 4, BProjection::Cursor);
        assert_eq!(
            rows_sorted(&linked),
            vec![vec![(0, 0.0), (1, 5.0), (2, 3.0)]],
        );
        assert_eq!(rows_sorted(&dense), vec![vec![(1, 5.0), (2, 3.0)]]);
    }

    /// Integer dtypes must ignore a forced Dense mode: a genuine zero result
    /// (2 - 2) stays in the output because they fall back to the linked list.
    #[test]
    fn dense_forced_on_ints_falls_back_to_linked_list() {
        let a_indptr = vec![0i32, 2];
        let a_indices = vec![0i32, 1];
        let a_data = vec![1i32, 1];
        let b_indptr = vec![0i32, 2, 4];
        let b_indices = vec![0i32, 1, 0, 2];
        let b_data = vec![2i32, 5, -2, 3];
        let a = CsrView::new(1, 2, &a_indptr, &a_indices, &a_data).unwrap();
        let b = CsrView::new(2, 3, &b_indptr, &b_indices, &b_data).unwrap();
        let opts = TopNOptions::<i32> {
            sort: SortMode::ByColumn,
            accum_mode: AccumMode::Dense,
            ..Default::default()
        };
        let c = run::<i32, i32>(a, b, 3, opts, 4, BProjection::Cursor);
        let mut row: Vec<(i32, i32)> = (0..c.nnz()).map(|k| (c.indices[k], c.data[k])).collect();
        row.sort_by_key(|(idx, _)| *idx);
        assert_eq!(row, vec![(0, 0), (1, 5), (2, 3)]);
    }

    /// row_update_density counts the exact per-row update total over ncols.
    #[test]
    fn row_update_density_exact() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        // Row 0 of A hits B-rows 0 (nnz 2) and 2 (nnz 2) → 4 updates over 4 cols.
        assert_eq!(row_update_density(a, b, 0), 1.0);
        // Row 1 hits B-rows 1 (nnz 2) and 2 (nnz 2) → likewise 1.0.
        assert_eq!(row_update_density(a, b, 1), 1.0);
    }

    /// pick_projection heuristic returns something sensible for both extremes.
    #[test]
    fn pick_projection_extremes() {
        // Dense B (nnz ≈ nrows * ncols), few chunks → BinarySearch.
        let dense_indptr: Vec<i32> = (0..=4).map(|i| i * 8).collect();
        let dense_indices: Vec<i32> = (0..32).map(|k| k % 8).collect();
        let dense_data = vec![1.0f64; 32];
        let dense = CsrView::new(4, 8, &dense_indptr, &dense_indices, &dense_data).unwrap();
        let a_ip = vec![0i32, 1];
        let a_idx = vec![0i32];
        let a_d = vec![1.0f64];
        let a = CsrView::new(1, 4, &a_ip, &a_idx, &a_d).unwrap();
        assert_eq!(pick_projection(a, dense, 8), BProjection::BinarySearch);

        // Sparse B, many chunks → Cursor.
        let sparse_indptr = vec![0i32, 1, 1, 1, 1];
        let sparse_indices = vec![0i32];
        let sparse_data = vec![1.0f64];
        let sparse = CsrView::new(4, 1024, &sparse_indptr, &sparse_indices, &sparse_data).unwrap();
        let a2 = CsrView::new(1, 4, &a_ip, &a_idx, &a_d).unwrap();
        assert_eq!(pick_projection(a2, sparse, 16), BProjection::Cursor);
    }

    /// auto_tile gates: each must individually veto the tiled+blocked path.
    #[test]
    fn auto_tile_gates() {
        // A 10×4 with 40 nnz (4 per row), B 4×40 with 20 nnz (5 per row).
        let a_ip: Vec<i32> = (0..=10).map(|i| i * 4).collect();
        let a_idx: Vec<i32> = (0..40).map(|k| k % 4).collect();
        let a_d = vec![1.0f64; 40];
        let a = CsrView::new(10, 4, &a_ip, &a_idx, &a_d).unwrap();
        let b_ip: Vec<i32> = (0..=4).map(|i| i * 5).collect();
        let b_idx: Vec<i32> = (0..20).map(|k| k * 2).collect();
        let b_d = vec![1.0f64; 20];
        let b = CsrView::new(4, 40, &b_ip, &b_idx, &b_d).unwrap();

        // All gates pass: 5 chunks, a.nnz 40 ≥ 2·4, nseg 20 ≤ 4·20.
        assert!(auto_tile(a, b, 8));
        // Single chunk → no reuse to win.
        assert!(!auto_tile(a, b, 40));
        // A too small to amortise the build (nnz 4 < 2·4).
        let a_small = CsrView::new(1, 4, &a_ip[..2], &a_idx[..4], &a_d[..4]).unwrap();
        assert!(!auto_tile(a_small, b, 8));
        // Segment table too large relative to nnz(B): 1 nnz per B-row,
        // 10 chunks → nseg 40 > 4·4.
        let bs_ip: Vec<i32> = vec![0, 1, 2, 3, 4];
        let bs_idx: Vec<i32> = vec![0, 10, 20, 30];
        let bs_d = vec![1.0f64; 4];
        let b_thin = CsrView::new(4, 40, &bs_ip, &bs_idx, &bs_d).unwrap();
        assert!(!auto_tile(a, b_thin, 4));
    }

    /// `row_block: Some(0)` is the explicit classic-kernel escape hatch.
    #[test]
    fn row_block_zero_forces_classic() {
        let a_ip: Vec<i32> = (0..=10).map(|i| i * 4).collect();
        let a_idx: Vec<i32> = (0..40).map(|k| k % 4).collect();
        let a_d = vec![1.0f64; 40];
        let a = CsrView::new(10, 4, &a_ip, &a_idx, &a_d).unwrap();
        let b_ip: Vec<i32> = (0..=4).map(|i| i * 5).collect();
        let b_idx: Vec<i32> = (0..20).map(|k| k * 2).collect();
        let b_d = vec![1.0f64; 20];
        let b = CsrView::new(4, 40, &b_ip, &b_idx, &b_d).unwrap();

        let auto = TopNOptions::<f64>::default();
        assert_eq!(
            resolve_blocking(a, b, 8, &auto),
            (true, Some(DEFAULT_ROW_BLOCK))
        );
        let classic = TopNOptions::<f64> {
            row_block: Some(0),
            ..Default::default()
        };
        assert_eq!(resolve_blocking(a, b, 8, &classic), (false, None));

        // Outputs of both paths agree bit-for-bit.
        let c_auto = sp_matmul_topn_chunked(
            a,
            b,
            3,
            TopNOptions {
                chunk_cols: Some(8),
                ..auto
            },
        );
        let c_classic = sp_matmul_topn_chunked(
            a,
            b,
            3,
            TopNOptions {
                chunk_cols: Some(8),
                ..classic
            },
        );
        assert_eq!(c_auto.indptr, c_classic.indptr);
        assert_eq!(c_auto.indices, c_classic.indices);
        assert_eq!(c_auto.data, c_classic.data);
    }
}
