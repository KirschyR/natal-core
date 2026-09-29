//! Route A (GPU FFT migration): `cudarc` cuFFT integration.
//!
//! Stage S0 of the M4 route-A GPU plan (`hexagon_spatial_test/results/
//! M4_routeA_productization_design.md` §9): activate the `cufft` binding and
//! prove a linear 2-D convolution through cuFFT matches the CPU reference.
//! No runtime migration path is wired here yet; that is S1/S2. Off by default
//! (this module only compiles under the `gpu` feature).

use cudarc::cufft::{CudaFft, sys::cufftType};
use cudarc::driver::CudaContext;

/// Linear ('same' crop) 2-D convolution of a real `rows x cols` signal with a
/// real `k x k` kernel using cuFFT r2c/c2r (f32), plus a host-side spectrum
/// multiply. Returns the central `rows x cols` region, normalised by the
/// transform size.
///
/// This is a correctness/S0 probe, not the production kernel (the production
/// path will multiply spectra on device and handle class batching).
///
/// ## Errors
/// Returns a string description when plan creation, transfer, or execution
/// fails.
pub fn fft_linear_conv_same(
    rows: usize,
    cols: usize,
    signal: &[f32],
    kernel: &[f32],
) -> Result<Vec<f32>, String> {
    let k = (kernel.len() as f64).sqrt() as usize;
    assert_eq!(k * k, kernel.len(), "kernel must be square");
    let fr = rows + k - 1;
    let fc = cols + k - 1;
    let spec_n = fr * (fc / 2 + 1);

    let mut sig = vec![0f32; fr * fc];
    for r in 0..rows {
        sig[r * fc..r * fc + cols].copy_from_slice(&signal[r * cols..r * cols + cols]);
    }
    let mut ker = vec![0f32; fr * fc];
    for r in 0..k {
        ker[r * fc..r * fc + k].copy_from_slice(&kernel[r * k..r * k + k]);
    }

    let ctx = CudaContext::new(0).map_err(|e| format!("cuda context: {e}"))?;
    let stream = ctx.default_stream();
    let r2c = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_R2C, stream.clone())
        .map_err(|e| format!("r2c plan: {e}"))?;
    let c2r = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_C2R, stream.clone())
        .map_err(|e| format!("c2r plan: {e}"))?;

    let sig_d = stream.clone_htod(&sig).map_err(|e| format!("{e}"))?;
    let ker_d = stream.clone_htod(&ker).map_err(|e| format!("{e}"))?;
    let mut sig_spec = stream
        .alloc_zeros::<cudarc::cufft::sys::float2>(spec_n)
        .map_err(|e| format!("{e}"))?;
    let mut ker_spec = stream
        .alloc_zeros::<cudarc::cufft::sys::float2>(spec_n)
        .map_err(|e| format!("{e}"))?;
    r2c.exec_r2c(&sig_d, &mut sig_spec).map_err(|e| format!("r2c: {e}"))?;
    r2c.exec_r2c(&ker_d, &mut ker_spec).map_err(|e| format!("r2c: {e}"))?;

    let sig_h = stream.clone_dtoh(&sig_spec).map_err(|e| format!("{e}"))?;
    let ker_h = stream.clone_dtoh(&ker_spec).map_err(|e| format!("{e}"))?;
    let mut prod = vec![cudarc::cufft::sys::float2 { x: 0.0, y: 0.0 }; spec_n];
    for i in 0..spec_n {
        let (a, b) = (sig_h[i], ker_h[i]);
        prod[i] = cudarc::cufft::sys::float2 {
            x: a.x * b.x - a.y * b.y,
            y: a.x * b.y + a.y * b.x,
        };
    }
    let mut prod_d = stream.clone_htod(&prod).map_err(|e| format!("{e}"))?;
    let mut out_d = stream.alloc_zeros::<f32>(fr * fc).map_err(|e| format!("{e}"))?;
    c2r.exec_c2r(&mut prod_d, &mut out_d).map_err(|e| format!("c2r: {e}"))?;
    let out = stream.clone_dtoh(&out_d).map_err(|e| format!("{e}"))?;

    let inv = 1.0f32 / (fr * fc) as f32;
    let r0 = (k - 1) / 2;
    let mut cropped = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            cropped[r * cols + c] = out[(r + r0) * fc + (c + r0)] * inv;
        }
    }
    Ok(cropped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_conv_same(rows: usize, cols: usize, signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let k = (kernel.len() as f64).sqrt() as usize;
        let r0 = (k - 1) / 2;
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let mut acc = 0f32;
                for kr in 0..k {
                    for kc in 0..k {
                        let sr = r as isize + kr as isize - r0 as isize;
                        let sc = c as isize + kc as isize - r0 as isize;
                        if sr >= 0 && sr < rows as isize && sc >= 0 && sc < cols as isize {
                            acc += signal[sr as usize * cols + sc as usize] * kernel[kr * k + kc];
                        }
                    }
                }
                out[r * cols + c] = acc;
            }
        }
        out
    }

    #[test]
    fn cufft_linear_conv_matches_cpu() {
        let (rows, cols, k) = (12usize, 16usize, 5usize);
        let mut signal = vec![0f32; rows * cols];
        let mut kernel = vec![0f32; k * k];
        for (i, v) in signal.iter_mut().enumerate() {
            *v = ((i * 2654435761usize) % 997) as f32 / 997.0;
        }
        let c = (k / 2) as f32;
        for r in 0..k {
            for cc in 0..k {
                let dr = r as f32 - c;
                let dc = cc as f32 - c;
                kernel[r * k + cc] = (-(dr * dr + dc * dc + dr * dc) / 2.0).exp();
            }
        }
        let gpu = fft_linear_conv_same(rows, cols, &signal, &kernel).expect("cufft conv");
        let cpu = cpu_conv_same(rows, cols, &signal, &kernel);
        let scale = cpu.iter().cloned().fold(0f32, f32::max).max(1e-9);
        let max_abs = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_abs / scale < 1e-4,
            "cuFFT conv mismatch: max_abs={max_abs} scale={scale}"
        );
    }
}
