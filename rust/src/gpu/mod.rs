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
//! and settles "does this host look like it has CUDA?" from the outside;
//! [`cuda`] drives the real bindings (driver load, device query, NVRTC
//! compile-and-run). P1 adds the device skeleton: [`context`] owns the
//! context and stream, [`buffers`] owns device allocations, and [`layout`]
//! performs the `(B, d0, …, dk) ↔ (d0, …, dk, B)` transpose the kernels need.
//! P2 adds [`kernels`] (NVRTC sources + launchers) and [`executor`], which
//! owns the uploaded state, enforces the §D5 memory budget, and runs the aging
//! stage; the remaining deterministic and sampling kernels follow in P3/P4.

/// RAII device allocations.
pub mod buffers;

/// CUDA context ownership and the device skeleton's entry point.
pub mod context;

/// `cudarc`-backed binding probe: dynamic driver loading, device queries, and
/// an NVRTC-compiled kernel that actually runs (P0-b).
pub mod cuda;

/// Device executor: uploaded state, kernel launches, and downloads.
pub mod executor;

/// NVRTC-compiled kernel sources and typed launchers.
pub mod kernels;

/// Batch-axis layout transpose between CPU and device conventions.
pub mod layout;

/// Dependency-free host discovery (`nvidia-smi`, toolkit directories).
pub mod probe;

/// Whether the hardware-dependent GPU tests must actually run.
///
/// The GPU server is this project's primary runtime, so hardware assertions
/// are the default: an unset `NATAL_GPU_REQUIRE` behaves like `1`, and a
/// missing or mismatched device is a hard failure rather than a silent skip.
/// A host without a matching GPU opts out explicitly with
/// `NATAL_GPU_REQUIRE=0`.
///
/// ## Returns
/// `true` unless `NATAL_GPU_REQUIRE` is exactly `0`.
#[cfg(test)]
pub(crate) fn hardware_required() -> bool {
    std::env::var("NATAL_GPU_REQUIRE").as_deref() != Ok("0")
}
