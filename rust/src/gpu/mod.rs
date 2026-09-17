//! GPU bypass scaffolding.
//!
//! Everything here sits behind the `gpu` cargo feature, which is off by
//! default: a stock build never compiles this module, so the CPU (rayon) path
//! remains the only executable engine and the offline development loop is
//! unaffected.
//!
//! ## Phasing
//!
//! See `GPU_insert_PLAN.md` §5 for the full plan. The feature currently
//! carries no dependency at all, because P0 only has to answer "can this host
//! run the bypass?" — a question the [`probe`] module settles with
//! `nvidia-smi` and a directory scan. The `cudarc` dependency and the actual
//! device backends arrive in P0-b/P1, at which point this module gains the
//! context, buffer, and layout submodules.

pub mod probe;
