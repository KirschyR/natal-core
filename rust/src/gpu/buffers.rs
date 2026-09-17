//! RAII device buffers for the GPU bypass.
//!
//! [`DeviceBuffer`] is a thin owner of a `cudarc` device allocation. It exists
//! so the executor holds one vocabulary — `from_host`, `to_host`, `len` — and
//! never has to thread raw driver handles through the tick code. The device
//! slice is freed when the buffer drops, which is also when the context must
//! still be alive; keeping buffers in the same executor as the
//! [`super::context::GpuContext`] guarantees that ordering.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};

/// An owned device allocation of `T`.
///
/// `T` must satisfy `cudarc`'s `DeviceRepr`, which the device path uses only
/// with `f32` (§D2).
pub struct DeviceBuffer<T> {
    /// The underlying device allocation.
    slice: CudaSlice<T>,
}

impl<T: DeviceRepr> DeviceBuffer<T> {
    /// Copy `host` to a freshly allocated device buffer.
    ///
    /// ## Parameters
    /// - `stream`: Stream the transfer is ordered on.
    /// - `host`: Host values to upload.
    ///
    /// ## Returns
    /// The device buffer holding a copy of `host`.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the allocation or the
    /// transfer.
    pub fn from_host(stream: &Arc<CudaStream>, host: &[T]) -> Result<Self, String> {
        let slice = stream
            .clone_htod(host)
            .map_err(|err| format!("host-to-device copy failed: {err}"))?;
        Ok(Self { slice })
    }

    /// Copy the device contents back to a host vector.
    ///
    /// ## Parameters
    /// - `stream`: Stream the transfer is ordered on.
    ///
    /// ## Returns
    /// An owned host copy of the buffer.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the transfer.
    pub fn to_host(&self, stream: &Arc<CudaStream>) -> Result<Vec<T>, String> {
        // `clone_dtoh` synchronizes the stream before returning.
        stream
            .clone_dtoh(&self.slice)
            .map_err(|err| format!("device-to-host copy failed: {err}"))
    }

    /// Number of elements held.
    ///
    /// ## Returns
    /// The device allocation length in elements.
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Whether the buffer holds no elements.
    ///
    /// ## Returns
    /// `true` for an empty allocation.
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// The underlying device slice, for kernel argument marshalling.
    ///
    /// ## Returns
    /// A shared reference to the device allocation.
    pub fn slice(&self) -> &CudaSlice<T> {
        &self.slice
    }

    /// The underlying device slice, mutably, for kernel output arguments.
    ///
    /// ## Returns
    /// A mutable reference to the device allocation.
    pub fn slice_mut(&mut self) -> &mut CudaSlice<T> {
        &mut self.slice
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/buffers.rs"]
mod tests;
