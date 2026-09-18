//! P1 tests for the CUDA context wrapper.
//!
//! Hardware-dependent assertions assert by default, matching the P0 probes; a
//! CPU-only host disables them with `NATAL_GPU_REQUIRE=0`. The `Send`/`Sync`
//! assertion is compile-time and runs everywhere the `gpu` feature is on.

use super::GpuContext;
use crate::gpu::hardware_required;

#[test]
fn context_reports_the_sm120_device() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    assert_eq!(
        context.compute_capability().expect("capability query"),
        (12, 0),
        "the target device is sm_120"
    );
    assert!(
        !context.name().expect("name query").is_empty(),
        "the driver must report a device name"
    );
    let (free, total) = context.memory_info().expect("memory query");
    assert!(
        free > 0,
        "no free device memory: the GPU is held by other tenants"
    );
    assert!(free <= total, "free memory cannot exceed total memory");
    assert!(
        GpuContext::device_count().expect("device count") >= 1,
        "a live context implies at least one device"
    );
}

#[test]
fn gpu_context_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    // §D7 runs the device path inside `py.allow_threads`, whose closure
    // requires every capture to be `Send`. This fails to compile if cudarc's
    // handles ever lose that property.
    assert_send_sync::<GpuContext>();
}

#[test]
fn out_of_range_ordinal_is_rejected() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    assert!(
        GpuContext::new(9999).is_err(),
        "an impossible ordinal must report an error, not a context"
    );
}
