//! P1 device-buffer tests.
//!
//! The transfer assertions assert by default; a CPU-only host disables them
//! with `NATAL_GPU_REQUIRE=0`. The L1 round trip combines the host transpose
//! with a device upload/download: integer state must survive bit for bit end
//! to end.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;
use crate::gpu::layout::{batch_to_inner, batch_to_outer};

use super::DeviceBuffer;

#[test]
fn device_round_trip_preserves_bits() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let host: Vec<f32> = (0..257).map(|value| value as f32).collect();
    let buffer = DeviceBuffer::from_host(&context.stream(), &host).expect("upload");
    assert_eq!(buffer.len(), host.len());
    assert!(!buffer.is_empty());
    let down = buffer.to_host(&context.stream()).expect("download");
    let expected: Vec<u32> = host.iter().map(|value| value.to_bits()).collect();
    let actual: Vec<u32> = down.iter().map(|value| value.to_bits()).collect();
    assert_eq!(actual, expected);
}

#[test]
fn layout_round_trip_through_the_device_is_bit_identical() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    // Representative spatial shape: (B=7, sexes=2, A=8, Z=9) as batch-minor.
    let axes = [2usize, 8, 9];
    let n_batch = 7;
    let outer: Vec<f32> = (0..axes.iter().product::<usize>() * n_batch)
        .map(|value| value as f32)
        .collect();
    let inner = batch_to_inner(&outer, &axes, n_batch).expect("transpose in");
    // The device sees the batch-minor slice unchanged: this is exactly what the
    // kernels will operate on.
    let buffer = DeviceBuffer::from_host(&context.stream(), &inner).expect("upload");
    let downloaded = buffer.to_host(&context.stream()).expect("download");
    let back = batch_to_outer(&downloaded, &axes, n_batch).expect("transpose out");
    let expected: Vec<u32> = outer.iter().map(|value| value.to_bits()).collect();
    let actual: Vec<u32> = back.iter().map(|value| value.to_bits()).collect();
    assert_eq!(actual, expected);
}
