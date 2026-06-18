//! `Scalar` trait — the element type of the matrices.
//!
//! Implemented for `f32, f64, i32, i64` to match the C++ extension's supported dtypes.

use num_traits::{NumAssign, NumCast};
use std::fmt::Debug;

pub trait Scalar:
    Copy + Default + PartialOrd + NumAssign + NumCast + Debug + Send + Sync + 'static
{
    /// `true` for floating-point dtypes. Gates the dense-accumulator chunk
    /// mode: that mode treats an exact-zero sum as "untouched", which is
    /// acceptable for floats but would silently drop genuine zero results for
    /// integer dtypes (e.g. `2 - 2`), so ints must stay on the linked-list path.
    const IS_FLOAT: bool;

    /// The smallest representable value, used as the heap sentinel.
    /// For floats this is `-∞`; for ints, `MIN`.
    fn min_value() -> Self;

    /// Mirrors C++'s `std::numeric_limits<V>::min()`: smallest *positive normal* for
    /// floats, type minimum for ints. The C++ zip path uses this as its sentinel
    /// threshold, which silently drops negative scores from the per-row top-n; we
    /// match it bit-for-bit (see plan §5.4).
    fn cpp_numeric_limits_min() -> Self;

    /// Per-dtype parity tolerance — bit-exact for ints, loose for floats. Returns
    /// `(abs_tol, rel_tol)`. The comparison the harness uses is
    /// `|a - b| <= abs OR |a - b| <= rel * max(|a|, |b|)`.
    fn parity_tol() -> (f64, f64);

    /// `self * b + acc`. Floats lower to hardware fused multiply-add where the
    /// target guarantees it (single rounding — matches clang's contracted C++
    /// kernel); ints, and float targets without FMA (baseline x86-64, where
    /// `std`'s `mul_add` becomes a libm call), use separate multiply + add.
    fn mul_add(self, b: Self, acc: Self) -> Self;

    /// Branchless maximum. Floats use `fN::max` (lowers to `fmax`; returns the
    /// non-NaN operand if one side is NaN), ints `Ord::max`. Used to group the
    /// dense drain scan four-wide with a single heap-threshold branch.
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_scalar_int {
    ($t:ty) => {
        impl Scalar for $t {
            const IS_FLOAT: bool = false;
            fn min_value() -> Self {
                <$t>::MIN
            }
            fn cpp_numeric_limits_min() -> Self {
                <$t>::MIN
            }
            fn parity_tol() -> (f64, f64) {
                (0.0, 0.0)
            }
            #[inline(always)]
            fn mul_add(self, b: Self, acc: Self) -> Self {
                self * b + acc
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                Ord::max(self, other)
            }
        }
    };
}
macro_rules! impl_scalar_float {
    ($t:ty, $tol:expr) => {
        impl Scalar for $t {
            const IS_FLOAT: bool = true;
            fn min_value() -> Self {
                <$t>::NEG_INFINITY
            }
            fn cpp_numeric_limits_min() -> Self {
                <$t>::MIN_POSITIVE
            }
            fn parity_tol() -> (f64, f64) {
                ($tol, $tol)
            }
            #[inline(always)]
            fn mul_add(self, b: Self, acc: Self) -> Self {
                #[cfg(any(target_arch = "aarch64", target_feature = "fma"))]
                {
                    <$t>::mul_add(self, b, acc)
                }
                #[cfg(not(any(target_arch = "aarch64", target_feature = "fma")))]
                {
                    self * b + acc
                }
            }
            #[inline(always)]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
        }
    };
}

impl_scalar_int!(i32);
impl_scalar_int!(i64);
impl_scalar_float!(f32, 1e-6);
impl_scalar_float!(f64, 1e-12);
