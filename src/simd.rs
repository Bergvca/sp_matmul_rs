//! Runtime CPU-feature detection for the x86-64 kernel clones.
//!
//! Baseline x86-64 builds (PyPI wheels) cannot assume AVX2/FMA, so the chunked
//! kernels ship `#[target_feature(enable = "avx2,fma")]` clones selected here
//! at runtime (see `chunked::process_row` / `chunked::process_row_block`). The
//! decision is cached in a process-wide atomic: one relaxed load per row on the
//! steady state. `SP_MATMUL_RS_FORCE_BASELINE=1` disables the clones (A/B
//! benchmarking, debugging); builds that already enable `fma` at compile time
//! (`-C target-cpu=native`, aarch64) compile the machinery away entirely.

#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
mod imp {
    use std::sync::atomic::{AtomicU8, Ordering};

    const UNINIT: u8 = 0;
    const ENABLED: u8 = 1;
    const DISABLED: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNINIT);

    /// Should the dispatching kernels take the AVX2+FMA clone?
    #[inline(always)]
    pub(crate) fn avx2_fma_enabled() -> bool {
        match STATE.load(Ordering::Relaxed) {
            ENABLED => true,
            DISABLED => false,
            _ => init(),
        }
    }

    #[cold]
    fn init() -> bool {
        let forced_off = std::env::var("SP_MATMUL_RS_FORCE_BASELINE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        let ok = !forced_off
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        STATE.store(if ok { ENABLED } else { DISABLED }, Ordering::Relaxed);
        ok
    }

    /// Test-only override of the cached decision (the env var is read once, so
    /// tests cannot flip it in-process). `true` forces the baseline kernels;
    /// `false` clears the cache so the next call re-detects.
    pub fn force_simd_baseline_for_tests(on: bool) {
        STATE.store(if on { DISABLED } else { UNINIT }, Ordering::Relaxed);
    }

    /// Which floating-point kernel path this process uses: `"avx2+fma"`
    /// (runtime-dispatched clones) or `"baseline"` (detection failed or
    /// disabled via `SP_MATMUL_RS_FORCE_BASELINE`).
    pub fn runtime_simd_label() -> &'static str {
        if avx2_fma_enabled() {
            "avx2+fma"
        } else {
            "baseline"
        }
    }
}

#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
pub(crate) use imp::avx2_fma_enabled;
#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
pub use imp::force_simd_baseline_for_tests;
#[cfg(all(target_arch = "x86_64", not(target_feature = "fma")))]
pub use imp::runtime_simd_label;

#[cfg(not(all(target_arch = "x86_64", not(target_feature = "fma"))))]
mod imp {
    /// No runtime dispatch on this target: no-op, kept so tests compile everywhere.
    pub fn force_simd_baseline_for_tests(_on: bool) {}

    /// Which floating-point kernel path this build uses: `"compile-time"`
    /// (FMA is unconditional in [`crate::Scalar::mul_add`]) or `"baseline"`
    /// (target has no fused path).
    pub fn runtime_simd_label() -> &'static str {
        // aarch64 and `target_feature = "fma"` builds fuse unconditionally in
        // `Scalar::mul_add`; other targets have no fused path at all.
        #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
        {
            "compile-time"
        }
        #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
        {
            "baseline"
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", not(target_feature = "fma"))))]
pub use imp::{force_simd_baseline_for_tests, runtime_simd_label};
