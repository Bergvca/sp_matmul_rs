//! Rayon-backed parallel driver. Feature `rayon` (default-on).
//!
//! Unit of work is a slice of rows;
//! within a worker the column-chunk loop stays serial so per-row scratch
//! (`sums`, `next`, heap, optional cursors) stays hot. Per-thread row blocks
//! are concatenated through a prefix sum on per-row nnz counts, avoiding the
//! C++ build's `nrows × top_n` peak allocation.
//!
//! The public entry point [`sp_matmul_topn_parallel`] mirrors
//! [`crate::chunked::sp_matmul_topn_chunked`] for empty/degenerate inputs and
//! short-circuits to the sequential chunked driver when `n_threads <= 1` — no
//! rayon overhead, no thread-pool spin-up.
//!
//! Threading invariance: for fixed inputs and a fixed (`chunk_cols`,
//! projection) pair, every row's output is independent of `n_threads`. The
//! cross-row output order is by row index, also independent. Therefore
//! parallel results are bit-for-bit identical to sequential results.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::chunked::{
    self, process_row, process_row_block, resolve_chunk_cols, BProjection, BlockScratch, UNVISITED,
};
use crate::csr::{matmul_topn_short_circuit, CsrMatrix, CsrView};
use crate::index::Index;
use crate::matmul_topn::TopNOptions;
use crate::maxheap::MaxHeap;
use crate::scalar::Scalar;
use crate::tiled::TiledB;

/// Minimum block size — keeps per-block rayon overhead under control for very
/// small `nrows`. The `target_blocks` math (§4.1 of the plan) takes over for
/// larger inputs.
const ROW_BLOCK_HINT: usize = 64;

/// Per-thread output for one row slice. Rows are dense and ordered by row
/// index; `row_nset` holds the per-row entry counts so the final compaction
/// can build `c_indptr` via a prefix sum without re-walking `indices`.
#[derive(Debug)]
struct RowBlock<V: Scalar, I: Index> {
    row_lo: usize,
    row_hi: usize,
    row_nset: Vec<u32>,
    indices: Vec<I>,
    data: Vec<V>,
}

/// Partition `[0, nrows)` into contiguous row blocks for rayon's work-stealer.
///
/// Aims for ~4× `n_threads` blocks so heavy-tailed row weight can be
/// rebalanced by stealing; floors block size at `hint` so we never produce
/// blocks too tiny to amortise the thread-pool dispatch.
fn partition_rows(nrows: usize, n_threads: usize, hint: usize) -> Vec<(usize, usize)> {
    if nrows == 0 {
        return Vec::new();
    }
    let target_blocks = (n_threads * 4).max(1);
    let block_size = nrows.div_ceil(target_blocks).max(hint);
    (0..nrows)
        .step_by(block_size)
        .map(|lo| (lo, (lo + block_size).min(nrows)))
        .collect()
}

/// Process-lifetime cache of rayon pools, keyed by thread count. Spinning up
/// a pool costs ~100 µs of thread spawns per call — measurable against
/// sub-200 ms kernels. Pools idle at near-zero cost (parked threads), matching
/// OpenMP's persistent-team behaviour in the C++ build. The map stays tiny:
/// one entry per distinct `n_threads` ever requested.
fn pool_for(n_threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("rayon pool cache poisoned");
    Arc::clone(pools.entry(n_threads).or_insert_with(|| {
        Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .expect("failed to build rayon thread pool"),
        )
    }))
}

/// Resolve `opts.n_threads`. `None` maps to sequential (matches the Python
/// wrapper's default; avoids surprise oversubscription when the crate is
/// embedded). `Some(0)` is rejected loudly.
fn resolve_n_threads(requested: Option<usize>) -> usize {
    match requested {
        Some(0) => panic!("n_threads must be > 0, got 0"),
        Some(n) => n,
        None => 1,
    }
}

/// Process `[row_lo, row_hi)` through the chunked per-row kernel, writing
/// into block-local output buffers. Allocates per-block scratch — see
/// §4.2 of the phase-3 plan for the rationale on not threading per-thread
/// scratch through `rayon::ThreadLocal`.
#[allow(clippy::too_many_arguments)]
fn run_row_slice<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
    chunk_cols: usize,
    projection: BProjection,
    tiled: Option<&TiledB<V>>,
    row_lo: usize,
    row_hi: usize,
) -> RowBlock<V, I> {
    let threshold = opts.threshold.unwrap_or_else(V::min_value);

    let block_rows = row_hi - row_lo;
    // Halve the per-row density estimate vs. the sequential driver — many
    // blocks summed will reach the sequential cap; halving keeps each block's
    // initial allocation modest. Vec growth covers under-estimates cheaply.
    let density = opts.density_hint.unwrap_or(1.0) * 0.5;
    let est_nnz = ((block_rows as f64) * (top_n as f64) * density).ceil() as usize;
    let mut row_nset: Vec<u32> = Vec::with_capacity(block_rows);
    let mut indices: Vec<I> = Vec::with_capacity(est_nnz);
    let mut data: Vec<V> = Vec::with_capacity(est_nnz);

    if tiled.is_some() || opts.row_block.is_some() {
        let rb = opts.row_block.unwrap_or(1).max(1).min(block_rows.max(1));
        let mut scratch = BlockScratch::<V, I>::new(top_n, threshold, chunk_cols, rb);
        let mut lo = row_lo;
        while lo < row_hi {
            let hi = (lo + rb).min(row_hi);
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
                &mut indices,
                &mut data,
                &mut row_nset,
            );
            lo = hi;
        }
        return RowBlock {
            row_lo,
            row_hi,
            row_nset,
            indices,
            data,
        };
    }

    let mut sums: Vec<V> = vec![V::default(); chunk_cols];
    let mut next: Vec<u32> = vec![UNVISITED; chunk_cols];
    let mut heap = MaxHeap::<V, I>::new(top_n, threshold);
    let mut cursors: Vec<usize> = Vec::new();

    for i in row_lo..row_hi {
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
            &mut indices,
            &mut data,
        );
        debug_assert!(n_set <= u32::MAX as usize, "row n_set {n_set} > u32::MAX");
        row_nset.push(n_set as u32);
    }

    RowBlock {
        row_lo,
        row_hi,
        row_nset,
        indices,
        data,
    }
}

/// Concatenate `blocks` in row order, materialising `c_indptr` via a serial
/// prefix sum over `row_nset`. Cost is `O(total_nnz + nrows)` — negligible
/// against the multiplication work for the target shapes.
fn build_csr_from_blocks<V: Scalar, I: Index>(
    nrows: usize,
    ncols: usize,
    mut blocks: Vec<RowBlock<V, I>>,
) -> CsrMatrix<V, I> {
    // rayon::par_iter().collect() preserves order on indexed iterators, so the
    // sort is a defensive guard against future iterator-source changes.
    blocks.sort_by_key(|b| b.row_lo);
    debug_assert!(blocks.windows(2).all(|w| w[0].row_hi == w[1].row_lo));

    let total_nnz: usize = blocks.iter().map(|b| b.indices.len()).sum();
    let mut indptr: Vec<I> = Vec::with_capacity(nrows + 1);
    let mut indices: Vec<I> = Vec::with_capacity(total_nnz);
    let mut data: Vec<V> = Vec::with_capacity(total_nnz);
    indptr.push(I::zero());

    let mut running: usize = 0;
    for block in blocks {
        for &n in &block.row_nset {
            running += n as usize;
            indptr.push(I::from_usize(running));
        }
        indices.extend(block.indices);
        data.extend(block.data);
    }
    debug_assert_eq!(indptr.len(), nrows + 1);
    debug_assert_eq!(indices.len(), total_nnz);

    CsrMatrix {
        nrows,
        ncols,
        indptr,
        indices,
        data,
    }
}

fn run_parallel<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
    chunk_cols: usize,
    projection: BProjection,
    n_threads: usize,
) -> CsrMatrix<V, I> {
    let nrows = a.nrows;
    let ncols = b.ncols;
    let row_blocks = partition_rows(nrows, n_threads, ROW_BLOCK_HINT);

    let (tile, row_block) = chunked::resolve_blocking(a, b, chunk_cols, &opts);
    let opts = TopNOptions { row_block, ..opts };
    let tiled = tile.then(|| TiledB::build(b, chunk_cols));
    let tiled_ref = tiled.as_ref();

    let pool = pool_for(n_threads);

    let blocks_out: Vec<RowBlock<V, I>> = pool.install(|| {
        row_blocks
            .par_iter()
            .map(|&(lo, hi)| {
                run_row_slice(a, b, top_n, opts, chunk_cols, projection, tiled_ref, lo, hi)
            })
            .collect()
    });

    build_csr_from_blocks(nrows, ncols, blocks_out)
}

/// Parallel `sp_matmul_topn`. Equivalent to the sequential chunked driver
/// for every valid input shape; `n_threads <= 1` short-circuits to it.
///
/// Thread pools are cached per `n_threads` for the process lifetime (see
/// [`pool_for`]); the first call at a given thread count pays the spin-up.
pub fn sp_matmul_topn_parallel<V: Scalar, I: Index>(
    a: CsrView<'_, V, I>,
    b: CsrView<'_, V, I>,
    top_n: usize,
    opts: TopNOptions<V>,
) -> CsrMatrix<V, I> {
    assert_eq!(
        a.ncols, b.nrows,
        "sp_matmul_topn_parallel: A.ncols ({}) must equal B.nrows ({})",
        a.ncols, b.nrows,
    );

    if let Some(out) = matmul_topn_short_circuit(a, b, top_n) {
        return out;
    }

    let chunk_cols = resolve_chunk_cols::<V>(opts.chunk_cols, b.ncols);
    let projection = opts
        .projection
        .unwrap_or_else(|| chunked::pick_projection(a, b, chunk_cols));
    let n_threads = resolve_n_threads(opts.n_threads);
    if n_threads <= 1 {
        return chunked::dispatch(a, b, top_n, opts, chunk_cols, projection);
    }
    run_parallel(a, b, top_n, opts, chunk_cols, projection, n_threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matmul_topn::SortMode;

    #[test]
    fn row_partitioning_covers_all_rows() {
        let cases = [
            (0usize, 1usize),
            (1, 1),
            (1, 4),
            (256, 1),
            (256, 4),
            (10_000, 8),
            (97, 3),
        ];
        for (nrows, nt) in cases {
            let blocks = partition_rows(nrows, nt, ROW_BLOCK_HINT);
            if nrows == 0 {
                assert!(blocks.is_empty(), "expected empty for nrows=0");
                continue;
            }
            assert!(!blocks.is_empty(), "expected non-empty for nrows={nrows}");
            assert_eq!(blocks[0].0, 0, "first block must start at 0");
            assert_eq!(
                blocks.last().unwrap().1,
                nrows,
                "last block must end at nrows",
            );
            for win in blocks.windows(2) {
                assert_eq!(win[0].1, win[1].0, "blocks must be contiguous");
            }
            for &(lo, hi) in &blocks {
                assert!(lo < hi, "block [{lo}, {hi}) must be non-empty");
            }
        }
    }

    #[test]
    fn partition_handles_empty_input() {
        let blocks = partition_rows(0, 8, ROW_BLOCK_HINT);
        assert!(blocks.is_empty());
    }

    type CsrParts = (Vec<i32>, Vec<i32>, Vec<f64>, Vec<i32>, Vec<i32>, Vec<f64>);

    fn make_a_b() -> CsrParts {
        // Same fixture as chunked / matmul_topn unit tests.
        let a_indptr = vec![0i32, 2, 4];
        let a_indices = vec![0i32, 2, 1, 2];
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_indptr = vec![0i32, 2, 4, 6];
        let b_indices = vec![0i32, 3, 1, 3, 2, 3];
        let b_data = vec![1.0f64, 5.0, 1.0, 6.0, 1.0, 7.0];
        (a_indptr, a_indices, a_data, b_indptr, b_indices, b_data)
    }

    #[test]
    fn n_threads_one_returns_sequential() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            sort: SortMode::ByValueDesc,
            n_threads: Some(1),
            ..Default::default()
        };
        let par = sp_matmul_topn_parallel(a, b, 2, opts);
        let seq = chunked::sp_matmul_topn_chunked(a, b, 2, opts);
        assert_eq!(par.indptr, seq.indptr);
        assert_eq!(par.indices, seq.indices);
        assert_eq!(par.data, seq.data);
    }

    #[test]
    fn n_threads_many_matches_sequential() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let base = TopNOptions {
            sort: SortMode::ByValueDesc,
            ..Default::default()
        };
        let seq = chunked::sp_matmul_topn_chunked(a, b, 2, base);
        for nt in [2usize, 4, 8] {
            let opts = TopNOptions {
                n_threads: Some(nt),
                ..base
            };
            let par = sp_matmul_topn_parallel(a, b, 2, opts);
            assert_eq!(par.indptr, seq.indptr, "n_threads={nt}");
            assert_eq!(par.indices, seq.indices, "n_threads={nt}");
            assert_eq!(par.data, seq.data, "n_threads={nt}");
        }
    }

    #[test]
    fn empty_inputs_short_circuit() {
        let zero_indptr: Vec<i32> = vec![0, 0, 0];
        let nil_idx: Vec<i32> = vec![];
        let nil_data: Vec<f64> = vec![];
        let a = CsrView::new(2, 3, &zero_indptr, &nil_idx, &nil_data).unwrap();
        let b_indptr: Vec<i32> = vec![0, 1, 2, 3];
        let b_indices: Vec<i32> = vec![0, 1, 0];
        let b_data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let b = CsrView::new(3, 2, &b_indptr, &b_indices, &b_data).unwrap();
        let opts = TopNOptions {
            n_threads: Some(4),
            ..Default::default()
        };
        let c = sp_matmul_topn_parallel(a, b, 5, opts);
        assert_eq!(c.nnz(), 0);
        assert_eq!(c.indptr, vec![0, 0, 0]);
    }

    #[test]
    fn pool_cache_reuses_pools() {
        let p1 = pool_for(3);
        let p2 = pool_for(3);
        assert!(Arc::ptr_eq(&p1, &p2), "same n_threads must share a pool");
        assert_eq!(p1.current_num_threads(), 3);
        let p3 = pool_for(2);
        assert!(!Arc::ptr_eq(&p1, &p3), "distinct n_threads get distinct pools");
    }

    #[test]
    #[should_panic(expected = "n_threads must be > 0")]
    fn n_threads_zero_panics() {
        let (a_ip, a_idx, a_d, b_ip, b_idx, b_d) = make_a_b();
        let a = CsrView::new(2, 3, &a_ip, &a_idx, &a_d).unwrap();
        let b = CsrView::new(3, 4, &b_ip, &b_idx, &b_d).unwrap();
        let opts = TopNOptions {
            n_threads: Some(0),
            ..Default::default()
        };
        let _ = sp_matmul_topn_parallel(a, b, 2, opts);
    }
}
