//! P0 CUDA-environment probe tests.
//!
//! The scan tests assert by default: the GPU server is the primary runtime, so
//! a missing or mismatched device is a hard failure. A CPU-only host disables
//! the hardware gate with `NATAL_GPU_REQUIRE=0`.
//!
//! The parsing and arithmetic tests are host-independent. They cover the
//! logic the scan delegates to, so the probe keeps real coverage on machines
//! that skip the hardware gate.

use super::{parse_device_line, CudaProbe};
use crate::gpu::hardware_required;

#[test]
fn scan_reports_without_panicking() {
    let probe = CudaProbe::scan();
    println!("{}", probe.report());
    if hardware_required() {
        assert!(
            probe.is_usable(),
            "hardware assertions are enabled by default but the CUDA environment is not usable; \
             set NATAL_GPU_REQUIRE=0 to skip"
        );
    }
}

#[test]
fn gpu_host_exposes_sm120_and_free_memory() {
    let probe = CudaProbe::scan();
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    assert!(
        probe.is_usable(),
        "hardware assertions are enabled by default but the CUDA environment is not usable; \
         set NATAL_GPU_REQUIRE=0 to skip"
    );
    let device = probe
        .devices
        .first()
        .expect("a usable probe always reports a device");
    assert_eq!(
        device.compute_capability.as_deref(),
        Some("12.0"),
        "expected sm_120 (compute capability 12.0) on the target GPU"
    );
    assert!(
        device.memory_free_mib() > 0,
        "no free device memory: the GPU is fully held by other tenants"
    );
    println!(
        "device 0: {} | {} MiB free of {} MiB",
        device.name,
        device.memory_free_mib(),
        device.memory_total_mib
    );
}

#[test]
fn device_line_parses_the_full_query_shape() {
    let device = parse_device_line("0, NVIDIA GeForce RTX 5090 D V2, 12.0, 24455, 19643")
        .expect("full query row parses");
    assert_eq!(device.index, 0);
    assert_eq!(device.name, "NVIDIA GeForce RTX 5090 D V2");
    assert_eq!(device.compute_capability.as_deref(), Some("12.0"));
    assert_eq!(device.memory_total_mib, 24455);
    assert_eq!(device.memory_used_mib, 19643);
    assert_eq!(device.memory_free_mib(), 4812);
}

#[test]
fn device_line_parses_the_basic_query_shape() {
    let device = parse_device_line("1, Some GPU, 8192, 0").expect("basic query row parses");
    assert_eq!(device.index, 1);
    assert_eq!(device.name, "Some GPU");
    assert!(device.compute_capability.is_none());
    assert_eq!(device.memory_free_mib(), 8192);
}

#[test]
fn device_line_rejects_unexpected_shapes() {
    assert!(parse_device_line("").is_none());
    assert!(parse_device_line("0, GPU, 12.0").is_none());
    assert!(parse_device_line("not-an-index, GPU, 12.0, 1, 1").is_none());
    assert!(parse_device_line("0, GPU, 12.0, lots, 1").is_none());
}

#[test]
fn unrecognized_compute_capability_is_dropped() {
    let device = parse_device_line("0, GPU, N/A, 1024, 0").expect("row parses");
    assert!(device.compute_capability.is_none());
}

#[test]
fn free_memory_saturates_instead_of_underflowing() {
    let device = parse_device_line("0, GPU, 12.0, 1024, 2048").expect("row parses");
    assert_eq!(device.memory_free_mib(), 0);
}
