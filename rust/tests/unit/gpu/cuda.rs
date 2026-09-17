//! P0-b `cudarc` binding-level tests.
//!
//! The hardware tests assert by default: the GPU server is the primary runtime,
//! so "cannot drive the device" is a hard error. `cudarc` panics when no driver
//! is present, so a CPU-only host must skip explicitly with
//! `NATAL_GPU_REQUIRE=0`.
//!
//! `cuda_context_is_send` is a compile-time assertion and therefore runs on
//! every host, including the ones that disable the hardware gate.

use super::{CudaBindingProbe, ADD_ONE_EXPECTED, ADD_ONE_INPUT};
use crate::gpu::hardware_required;
use crate::gpu::probe::CudaProbe;

#[test]
fn binding_probe_runs_without_panicking() {
    let probe = CudaBindingProbe::run();
    println!("{}", probe.report());
    if hardware_required() {
        assert!(
            probe.is_usable(),
            "hardware assertions are enabled by default but the cudarc bindings are not usable \
             (error: {:?}); set NATAL_GPU_REQUIRE=0 to skip",
            probe.error
        );
    }
}

#[test]
fn driver_exposes_the_sm120_device() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let probe = CudaBindingProbe::run();
    assert_eq!(
        probe.device_count,
        Some(1),
        "expected exactly one visible device, got {:?} (error: {:?})",
        probe.device_count,
        probe.error
    );
    assert_eq!(
        probe.compute_capability,
        Some((12, 0)),
        "cudarc should agree with nvidia-smi on sm_120 (compute capability 12.0)"
    );
}

#[test]
fn mem_get_info_cross_checks_nvidia_smi() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let binding = CudaBindingProbe::run();
    let shell = CudaProbe::scan();
    let smi_device = shell
        .devices
        .first()
        .expect("a required CUDA host always reports a device through nvidia-smi");
    assert_eq!(
        binding
            .compute_capability
            .map(|(major, minor)| format!("{major}.{minor}")),
        smi_device.compute_capability,
        "cudarc and nvidia-smi must agree on the device's compute capability"
    );
    let free = binding
        .memory_free_bytes
        .expect("binding probe reports free memory");
    let total = binding
        .memory_total_bytes
        .expect("binding probe reports total memory");
    assert!(
        free > 0,
        "no free device memory: the GPU is held by other tenants"
    );
    assert!(free <= total, "free memory cannot exceed total memory");
    let smi_total = smi_device.memory_total_mib * 1024 * 1024;
    let drift = (total as i64 - smi_total as i64).unsigned_abs();
    assert!(
        drift * 10 <= smi_total,
        "cudaMemGetInfo total ({total} B) disagrees with nvidia-smi ({smi_total} B) by more than 10%"
    );
    println!(
        "cudaMemGetInfo: {free} B free of {total} B; \
         nvidia-smi: {} MiB free of {} MiB",
        smi_device.memory_free_mib(),
        smi_device.memory_total_mib
    );
}

#[test]
fn nvrtc_compiles_and_runs_add_one() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let probe = CudaBindingProbe::run();
    assert!(
        probe.nvrtc_ptx_bytes.unwrap_or(0) > 0,
        "NVRTC produced no PTX (error: {:?})",
        probe.error
    );
    let result = probe
        .add_one_result
        .expect("NVRTC compiled, so the kernel must have run");
    assert!(
        (result - ADD_ONE_EXPECTED).abs() <= 1e-6,
        "add_one({ADD_ONE_INPUT}) returned {result}, expected {ADD_ONE_EXPECTED}"
    );
}

#[test]
fn cuda_context_is_send() {
    fn assert_send<T: Send>() {}
    // §D7 launches the device path under `py.allow_threads`, whose closure
    // requires every captured handle to be `Send`. Cudarc's context and
    // stream are `Send + Sync`; this test fails to compile if that changes.
    assert_send::<cudarc::driver::CudaContext>();
    assert_send::<std::sync::Arc<cudarc::driver::CudaContext>>();
    assert_send::<cudarc::driver::CudaStream>();
    assert_send::<cudarc::driver::CudaSlice<f32>>();
}
