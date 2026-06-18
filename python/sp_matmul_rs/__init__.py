"""sp_matmul_rs — sparse matrix multiplication with fused top-n selection.

Standalone Rust implementation. Functionally equivalent to the Python API
exposed by the legacy ``sparse_dot_topn`` package; see ``sp_matmul_rs.api`` for
the supported call surface.
"""
from sp_matmul_rs import _core
from sp_matmul_rs._core import __rust_build__, _has_openmp_support, _rust_parallel_enabled, kernel_info
from sp_matmul_rs.api import sp_matmul, sp_matmul_topn, zip_sp_matmul_topn

__all__ = [
    "sp_matmul",
    "sp_matmul_topn",
    "zip_sp_matmul_topn",
    "kernel_info",
    "_core",
    "_has_openmp_support",
    "_rust_parallel_enabled",
    "__rust_build__",
]
