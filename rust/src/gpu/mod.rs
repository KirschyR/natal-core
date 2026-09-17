//! GPU bypass scaffolding.
//!
//! Everything here sits behind the `gpu` cargo feature, which is off by
//! default: a stock build never compiles this module, so the CPU (rayon) path
//! remains the only executable engine and the offline development loop is
//! unaffected.
//!
//! ## Phasing
//!
//! See `GPU_insert_PLAN.md` §5 for the full plan. [`probe`] is dependency-free
//! and settles "does this host look like it has CUDA?" from the outside; with
//! P0-b, [`cuda`] adds `cudarc` and drives the real bindings (driver load,
//! device query, NVRTC compile-and-run). The actual device backends — context,
//! buffers, and layout transpose — arrive in P1, at which point this module
//! gains those submodules.

/// Dependency-free host discovery (`nvidia-smi`, toolkit directories).
pub mod probe;

/// `cudarc`-backed binding probe: dynamic driver loading, device queries, and
/// an NVRTC-compiled kernel that actually runs (P0-b).
pub mod cuda;
