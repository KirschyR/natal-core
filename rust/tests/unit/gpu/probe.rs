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
use std::path::PathBuf;

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

#[test]
fn default_probe_report_covers_unknown_branches() {
    let probe = CudaProbe::default();
    let report = probe.report();
    assert!(report.contains("nvidia-smi      : NOT FOUND"));
    assert!(report.contains("driver version  : <unknown>"));
    assert!(report.contains("CUDA version    : <unknown>"));
    assert!(report.contains("NVRTC           : NOT FOUND"));
    assert!(report.contains("toolkit roots   : <none>"));
    assert!(report.contains("devices         : <none>"));
    assert!(report.contains("usable          : false"));
    assert!(!probe.is_usable());
}

#[test]
fn nvrtc_names_follow_the_platform_convention() {
    if cfg!(windows) {
        assert!(super::is_nvrtc_name("nvrtc64_120_0.dll"));
        assert!(!super::is_nvrtc_name("nvrtc.so"));
    } else {
        assert!(super::is_nvrtc_name("libnvrtc.so"));
        assert!(super::is_nvrtc_name("libnvrtc.so.13"));
        assert!(!super::is_nvrtc_name("libcuda.so"));
    }
}

#[test]
fn loader_path_variable_is_defined() {
    assert!(!super::loader_path_var().is_empty());
}

#[test]
fn nvrtc_search_reports_platform_library_directories() {
    let roots = vec![PathBuf::from("/opt/natal-does-not-exist")];
    let dirs = super::nvrtc_search_dirs(&roots);
    assert!(!dirs.is_empty());
    // The toolkit scan itself must not panic on a missing directory.
    let _ = super::cuda_toolkit_roots();
}

#[test]
fn search_dirs_returns_the_first_matching_entry() {
    let unique = format!("natal-probe-{}", std::process::id());
    let dir = std::env::temp_dir().join(unique);
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let lib = dir.join("libnvrtc.so.99");
    std::fs::write(&lib, b"stub").expect("write stub");
    let found = super::search_dirs(std::slice::from_ref(&dir), |name| name == "libnvrtc.so.99");
    assert_eq!(found.as_deref(), Some(lib.as_path()));
    assert!(super::search_dirs(std::slice::from_ref(&dir), |_| false).is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn broken_smi_binary_degrades_gracefully() {
    let bad = PathBuf::from("/nonexistent/nvidia-smi");
    assert!(super::run_smi(&bad, &["--help"]).is_none());
    assert!(super::query_first(&bad, "driver_version").is_none());
    assert!(super::parse_cuda_version(&bad).is_none());
    assert!(super::query_devices(&bad).is_empty());
}

#[test]
fn report_lists_toolkit_roots_when_present() {
    let probe = CudaProbe {
        toolkit_roots: vec![PathBuf::from("/usr/local/cuda-99")],
        ..CudaProbe::default()
    };
    assert!(probe.report().contains("/usr/local/cuda-99"));
}

#[test]
fn search_dirs_skips_unreadable_directories() {
    assert!(super::search_dirs(&[PathBuf::from("/nonexistent/natal-dir")], |_| true).is_none());
}
