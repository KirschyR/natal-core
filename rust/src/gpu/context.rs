//! CUDA context and stream ownership for the device bypass.
//!
//! [`GpuContext`] is the single owner of the driver context and its default
//! stream. Every later device object (buffers, compiled kernels, executors)
//! borrows from it, so the context outlives all of them and there is exactly
//! one primary context per process.
//!
//! `cudarc`'s handles are `Send + Sync`, which the compile-time assertion in
//! the tests relies on: §D7 runs the whole device path inside
//! `py.allow_threads`, whose closure requires its captures to be `Send`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

/// A live CUDA context plus the default stream kernels launch on.
///
/// ## Lifetime
/// Dropping a [`GpuContext`] releases the stream and, once the last `Arc`
/// clone of the underlying context is gone, the context itself. Device
/// buffers must not outlive the context they were allocated under; keeping
/// them alongside a single [`GpuContext`] in an executor enforces this
/// naturally.
pub struct GpuContext {
    /// The primary context for one device ordinal.
    context: Arc<CudaContext>,
    /// The context's default stream; all uploads, kernels, and downloads
    /// are ordered on it.
    stream: Arc<CudaStream>,
}

impl GpuContext {
    /// Create a context for `ordinal` and select its default stream.
    ///
    /// ## Parameters
    /// - `ordinal`: Zero-based device index.
    ///
    /// ## Returns
    /// A ready context, or a description of why the driver refused.
    ///
    /// ## Errors
    /// Returns a description when CUDA is absent, the ordinal is invalid, or
    /// the driver rejects context creation.
    pub fn new(ordinal: usize) -> Result<Self, String> {
        let context = CudaContext::new(ordinal).map_err(|err| {
            format!("could not create a CUDA context for device {ordinal}: {err}")
        })?;
        let stream = context.default_stream();
        Ok(Self { context, stream })
    }

    /// Number of devices visible to the driver API.
    ///
    /// ## Returns
    /// The device count, or a description of the driver failure.
    pub fn device_count() -> Result<i32, String> {
        CudaContext::device_count().map_err(|err| format!("device count query failed: {err}"))
    }

    /// Marketing name of the context's device.
    ///
    /// ## Returns
    /// The driver-reported name, or a description of the failure.
    pub fn name(&self) -> Result<String, String> {
        self.context
            .name()
            .map_err(|err| format!("device name query failed: {err}"))
    }

    /// Compute capability of the context's device.
    ///
    /// ## Returns
    /// `(major, minor)`, e.g. `(12, 0)` for sm_120.
    pub fn compute_capability(&self) -> Result<(i32, i32), String> {
        self.context
            .compute_capability()
            .map_err(|err| format!("compute capability query failed: {err}"))
    }

    /// Free and total device memory in bytes.
    ///
    /// ## Returns
    /// `(free, total)`, the input the §D5 budget guard must use.
    pub fn memory_info(&self) -> Result<(usize, usize), String> {
        self.context
            .mem_get_info()
            .map_err(|err| format!("memory info query failed: {err}"))
    }

    /// The underlying context, for APIs that need it directly.
    ///
    /// ## Returns
    /// A clone of the shared context handle.
    pub fn context(&self) -> Arc<CudaContext> {
        Arc::clone(&self.context)
    }

    /// The default stream every device operation is ordered on.
    ///
    /// ## Returns
    /// A clone of the shared stream handle.
    pub fn stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/context.rs"]
mod tests;
