//! P2 kernel-loader tests.
//!
//! Loading exercises NVRTC compilation end to end and asserts by default (the
//! GPU server is the primary runtime); `NATAL_GPU_REQUIRE=0` disables it.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;

use super::Kernels;

#[test]
fn aging_kernel_compiles_and_loads() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    Kernels::load(&context.context()).expect("aging kernel compiles and loads");
}
