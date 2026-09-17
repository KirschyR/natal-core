//! Binding-level CUDA probe backed by `cudarc` (P0-b).
//!
//! [`super::probe`] answers "does this host look like it has CUDA?" from the
//! outside, by shelling out to `nvidia-smi`; this module answers the strictly
//! stronger question *"can the Rust process actually drive the device?"* by
//! going through the real bindings:
//!
//! 1. `cudarc` dynamically loads the CUDA driver and creates a context.
//! 2. The device reports its compute capability and free memory.
//! 3. NVRTC compiles a trivial kernel from source, the module is loaded, the
//!    kernel runs, and the result is copied back.
//! 4. The context handle is `Send`, which `py.allow_threads` will need (§D7).
//!
//! It deliberately keeps [`super::probe`] dependency-free: this module is only
//! compiled under the `gpu` feature, while the shell-based probe remains usable
//! in a stock build.
//!
//! ## Why NVRTC is the decisive assertion
//!
//! If the host has a device but no NVRTC, runtime kernel compilation — the
//! foundation of the whole bypass design (§D1) — is unavailable, and the
//! project would have to fall back to ahead-of-time `nvcc` compilation, which
//! changes the `maturin` packaging flow. A failed NVRTC step is therefore a
//! hard, reportable finding rather than a routine skip.
//!
//! On this project's server NVRTC is not in a standard toolkit directory: it
//! ships inside the Python `nvidia` packages under
//! `.../site-packages/nvidia/cu13/lib`. Expose that directory through
//! `LD_LIBRARY_PATH`, exactly as `cudarc`'s dynamic loader needs; the
//! dependency-free [`super::probe`] consults the same variable when scanning
//! for the library.
//!
//! ## Failure model
//!
//! `cudarc` loads the driver lazily and panics when no library is found, so
//! this probe wraps the whole binding interaction in
//! [`std::panic::catch_unwind`]. On a machine without CUDA the probe reports
//! `error: Some(...)` and `is_usable() == false` instead of unwinding. The
//! test harness turns that into a failure via
//! [`super::hardware_required`], exactly like the shell-based probe; a
//! CPU-only host opts out with `NATAL_GPU_REQUIRE=0`.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// CUDA C source for the P0-b end-to-end check.
///
/// The kernel is intentionally trivial: it exercises source-to-PTX
/// compilation, module loading, argument marshalling, launch, and copy-back,
/// without touching any natal arithmetic.
pub const ADD_ONE_SOURCE: &str = r#"
extern "C" __global__ void add_one(float* x) { x[0] += 1.0f; }
"#;

/// Name of the kernel in [`ADD_ONE_SOURCE`].
pub const ADD_ONE_KERNEL: &str = "add_one";

/// Value written to the device before launching [`ADD_ONE_KERNEL`].
pub const ADD_ONE_INPUT: f32 = 41.0;

/// Value [`ADD_ONE_KERNEL`] is expected to produce from [`ADD_ONE_INPUT`].
pub const ADD_ONE_EXPECTED: f32 = 42.0;

/// What `cudarc` learned about the machine.
///
/// Every field is `None` until the corresponding step succeeds, so a partially
/// working host is still described as precisely as possible.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CudaBindingProbe {
    /// Devices visible to the driver API.
    pub device_count: Option<i32>,
    /// Marketing name reported by the driver.
    pub device_name: Option<String>,
    /// `(major, minor)` compute capability, e.g. `(12, 0)` for sm_120.
    pub compute_capability: Option<(i32, i32)>,
    /// Free device memory in bytes, as `cudaMemGetInfo` sees it.
    pub memory_free_bytes: Option<usize>,
    /// Total device memory in bytes.
    pub memory_total_bytes: Option<usize>,
    /// Size in bytes of the PTX image NVRTC produced.
    pub nvrtc_ptx_bytes: Option<usize>,
    /// The value copied back from the device after launching the kernel.
    pub add_one_result: Option<f32>,
    /// Human-readable description of the first failing step, when any.
    pub error: Option<String>,
}

impl CudaBindingProbe {
    /// Drive the real CUDA bindings end to end.
    ///
    /// ## Returns
    /// A probe populated as far as the host allowed. A missing driver or a
    /// compilation failure is recorded in [`CudaBindingProbe::error`]; the
    /// call never panics, even when `cudarc` would.
    pub fn run() -> Self {
        match catch_unwind(AssertUnwindSafe(Self::run_inner)) {
            Ok(Ok(probe)) => probe,
            Ok(Err(error)) => Self {
                error: Some(error),
                ..Self::default()
            },
            Err(_) => Self {
                error: Some("CUDA driver library could not be loaded".to_owned()),
                ..Self::default()
            },
        }
    }

    /// The fallible body of [`CudaBindingProbe::run`].
    ///
    /// ## Returns
    /// A fully populated probe, or the first error encountered.
    fn run_inner() -> Result<Self, String> {
        let device_count =
            CudaContext::device_count().map_err(|err| format!("device_count failed: {err}"))?;
        let context =
            CudaContext::new(0).map_err(|err| format!("CudaContext::new(0) failed: {err}"))?;
        let device_name = context
            .name()
            .map_err(|err| format!("device name failed: {err}"))?;
        let capability = context
            .compute_capability()
            .map_err(|err| format!("compute capability failed: {err}"))?;
        let (memory_free_bytes, memory_total_bytes) = context
            .mem_get_info()
            .map_err(|err| format!("cudaMemGetInfo failed: {err}"))?;
        let (ptx, add_one_result) = compile_and_run(&context, capability)?;
        Ok(Self {
            device_count: Some(device_count),
            device_name: Some(device_name),
            compute_capability: Some(capability),
            memory_free_bytes: Some(memory_free_bytes),
            memory_total_bytes: Some(memory_total_bytes),
            nvrtc_ptx_bytes: Some(ptx),
            add_one_result: Some(add_one_result),
            error: None,
        })
    }

    /// Whether the host can run the GPU bypass.
    ///
    /// ## Returns
    /// `true` only when the driver, the device query, and the NVRTC
    /// compile-and-run step all succeeded.
    pub fn is_usable(&self) -> bool {
        self.error.is_none()
            && self.device_count.unwrap_or(0) > 0
            && self.compute_capability.is_some()
            && self.memory_free_bytes.is_some()
            && self.add_one_result.is_some()
    }

    /// Render the probe as a multi-line human-readable report.
    ///
    /// ## Returns
    /// One line per discovered fact, ending with the overall usability verdict.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "device count    : {}",
            self.device_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "<unknown>".to_owned())
        ));
        lines.push(format!(
            "device name     : {}",
            self.device_name.as_deref().unwrap_or("<unknown>")
        ));
        match self.compute_capability {
            Some((major, minor)) => lines.push(format!("compute capab.  : {major}.{minor}")),
            None => lines.push("compute capab.  : <unknown>".to_owned()),
        }
        match (self.memory_free_bytes, self.memory_total_bytes) {
            (Some(free), Some(total)) => lines.push(format!(
                "device memory   : {} MiB free of {} MiB",
                free / (1024 * 1024),
                total / (1024 * 1024)
            )),
            _ => lines.push("device memory   : <unknown>".to_owned()),
        }
        match self.nvrtc_ptx_bytes {
            Some(bytes) => lines.push(format!("NVRTC           : compiled {bytes} bytes of PTX")),
            None => lines.push("NVRTC           : NOT USABLE".to_owned()),
        }
        match self.add_one_result {
            Some(value) => lines.push(format!(
                "kernel run      : add_one({ADD_ONE_INPUT}) = {value}"
            )),
            None => lines.push("kernel run      : NOT USABLE".to_owned()),
        }
        match &self.error {
            Some(error) => lines.push(format!("error           : {error}")),
            None => lines.push("error           : <none>".to_owned()),
        }
        lines.push(format!("usable          : {}", self.is_usable()));
        lines.join("\n")
    }
}

/// Compile [`ADD_ONE_SOURCE`] with NVRTC and run it on `context`.
///
/// The architecture is pinned to the device's own compute capability so the
/// generated PTX does not rely on NVRTC's default target.
///
/// ## Parameters
/// - `context`: A live context for device 0.
/// - `capability`: The device's `(major, minor)` compute capability.
///
/// ## Returns
/// `(ptx_bytes, result)` — the size of the compiled image and the value copied
/// back from the device.
fn compile_and_run(
    context: &Arc<CudaContext>,
    capability: (i32, i32),
) -> Result<(usize, f32), String> {
    let opts = CompileOptions {
        options: vec![format!(
            "--gpu-architecture=compute_{}{}",
            capability.0, capability.1
        )],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(ADD_ONE_SOURCE, opts)
        .map_err(|err| format!("NVRTC compilation failed: {err}"))?;
    let ptx_bytes = ptx.as_bytes().map(<[u8]>::len).unwrap_or(0);
    let module = context
        .load_module(ptx)
        .map_err(|err| format!("loading the compiled module failed: {err}"))?;
    let function = module
        .load_function(ADD_ONE_KERNEL)
        .map_err(|err| format!("loading kernel `{ADD_ONE_KERNEL}` failed: {err}"))?;
    let stream = context.default_stream();
    let mut device = stream
        .clone_htod(&[ADD_ONE_INPUT])
        .map_err(|err| format!("host-to-device copy failed: {err}"))?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&function);
    launch.arg(&mut device);
    unsafe { launch.launch(config) }.map_err(|err| format!("kernel launch failed: {err}"))?;
    let host = stream
        .clone_dtoh(&device)
        .map_err(|err| format!("device-to-host copy failed: {err}"))?;
    let result = *host
        .first()
        .ok_or_else(|| "empty copy-back buffer".to_owned())?;
    Ok((ptx_bytes, result))
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/cuda.rs"]
mod tests;
