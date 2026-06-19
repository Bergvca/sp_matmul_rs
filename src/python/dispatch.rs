//! Runtime dtype dispatch for the PyO3 bindings.
//!
//! The bindings accept value/index arrays as `&Bound<'_, PyAny>` and dispatch
//! on the numpy dtype at the boundary, so each public function is bound
//! exactly once rather than 8× across the (V, I) cross-product.

use numpy::{dtype, PyArrayDescr, PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VKind {
    F32,
    F64,
    I32,
    I64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IKind {
    I32,
    I64,
}

pub fn dtype_of<'py>(arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArrayDescr>> {
    let untyped: &Bound<'py, PyUntypedArray> = arr.downcast()?;
    Ok(untyped.dtype())
}

pub fn dtypes_equiv(a: &Bound<'_, PyArrayDescr>, b: &Bound<'_, PyArrayDescr>) -> bool {
    a.is_equiv_to(b)
}

pub fn value_kind(py: Python<'_>, dt: &Bound<'_, PyArrayDescr>) -> Option<VKind> {
    if dt.is_equiv_to(&dtype::<f64>(py)) {
        return Some(VKind::F64);
    }
    if dt.is_equiv_to(&dtype::<f32>(py)) {
        return Some(VKind::F32);
    }
    if dt.is_equiv_to(&dtype::<i32>(py)) {
        return Some(VKind::I32);
    }
    if dt.is_equiv_to(&dtype::<i64>(py)) {
        return Some(VKind::I64);
    }
    None
}

pub fn index_kind(py: Python<'_>, dt: &Bound<'_, PyArrayDescr>) -> Option<IKind> {
    if dt.is_equiv_to(&dtype::<i32>(py)) {
        return Some(IKind::I32);
    }
    if dt.is_equiv_to(&dtype::<i64>(py)) {
        return Some(IKind::I64);
    }
    None
}

/// Dispatch on the (value, index) dtype pair of `a_data`/`a_indptr`, also
/// validating that `b_data`/`b_indptr` carry compatible dtypes. The body is
/// expanded once per supported (V, I) arm with `V` and `I` aliased to the
/// concrete Rust types. Unsupported combos return `PyTypeError`.
#[macro_export]
macro_rules! dispatch_vi {
    ($py:expr, $a_data:expr, $a_indptr:expr, $b_data:expr, $b_indptr:expr, $v:ident, $i:ident => $body:expr) => {{
        let __py: ::pyo3::Python<'_> = $py;
        let __a_data_dt = $crate::python::dispatch::dtype_of($a_data)?;
        let __a_indptr_dt = $crate::python::dispatch::dtype_of($a_indptr)?;
        let __b_data_dt = $crate::python::dispatch::dtype_of($b_data)?;
        let __b_indptr_dt = $crate::python::dispatch::dtype_of($b_indptr)?;
        if !$crate::python::dispatch::dtypes_equiv(&__a_data_dt, &__b_data_dt) {
            return ::std::result::Result::Err(::pyo3::exceptions::PyTypeError::new_err(
                "A and B must have the same value dtype",
            ));
        }
        if !$crate::python::dispatch::dtypes_equiv(&__a_indptr_dt, &__b_indptr_dt) {
            return ::std::result::Result::Err(::pyo3::exceptions::PyTypeError::new_err(
                "A and B must have the same index dtype",
            ));
        }
        let __vk = $crate::python::dispatch::value_kind(__py, &__a_data_dt);
        let __ik = $crate::python::dispatch::index_kind(__py, &__a_indptr_dt);
        match (__vk, __ik) {
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::F64),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I32),
            ) => {
                type $v = f64;
                type $i = i32;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::F64),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I64),
            ) => {
                type $v = f64;
                type $i = i64;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::F32),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I32),
            ) => {
                type $v = f32;
                type $i = i32;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::F32),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I64),
            ) => {
                type $v = f32;
                type $i = i64;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::I32),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I32),
            ) => {
                type $v = i32;
                type $i = i32;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::I32),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I64),
            ) => {
                type $v = i32;
                type $i = i64;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::I64),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I32),
            ) => {
                type $v = i64;
                type $i = i32;
                $body
            }
            (
                ::std::option::Option::Some($crate::python::dispatch::VKind::I64),
                ::std::option::Option::Some($crate::python::dispatch::IKind::I64),
            ) => {
                type $v = i64;
                type $i = i64;
                $body
            }
            _ => ::std::result::Result::Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                "unsupported dtype combo: values={}, indices={}",
                __a_data_dt.str()?,
                __a_indptr_dt.str()?
            ))),
        }
    }};
}
