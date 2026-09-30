//! Route-A device FFT migration (GPU cuFFT), deterministic spatial models.
//!
//! Implements `Hex_model_recon.md` §4 route A: the deterministic spatial
//! migration is a per-plane normalized convolution (productization design
//! §3.2c), so each `(sex, age, ztype)` plane advances with
//!
//! `plane' = plane·(1−rate) + K' ⊛ (rate·plane / Z)`
//!
//! where `⊛` is a zero-outside linear convolution computed with cuFFT.  This is
//! the isolated S2b device kernel: it owns its context/stream and takes host
//! slices so it can be unit-tested against the CPU CSR operator before the
//! executor/session wiring (S2c).  Compiled only under the `gpu` feature.

use cudarc::cufft::sys::{cufftType, float2};
use cudarc::cufft::CudaFft;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const FFT_MIGRATE_SOURCE: &str = r#"
struct cf2 { float x; float y; };

extern "C" __global__ void scatter_g(
    const float* plane, const float* rate, const float* zs, float* pad,
    unsigned long long total, int rows, int cols, int fr, int fc, int R,
    int rate_stride, int rate_off)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int r = (int)(i / (unsigned long long)fc);
    int c = (int)(i % (unsigned long long)fc);
    if (r < rows && c < cols) {
        unsigned long long idx = (unsigned long long)r * cols + c;
        float rr = rate[idx * (unsigned long long)rate_stride + (unsigned long long)rate_off];
        pad[i] = plane[idx] * rr / zs[idx];
    } else {
        pad[i] = 0.0f;
    }
}

extern "C" __global__ void complex_mul(cf2* spec, const cf2* ks, unsigned long long n, float inv)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    cf2 a = spec[i];
    cf2 b = ks[i];
    spec[i].x = (a.x * b.x - a.y * b.y) * inv;
    spec[i].y = (a.x * b.y + a.y * b.x) * inv;
}

extern "C" __global__ void combine(
    const float* plane, const float* rate, const float* padded, float* out,
    unsigned long long n, int cols, int fr, int fc, int R, int rate_stride, int rate_off)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = (int)(i / (unsigned long long)cols);
    int c = (int)(i % (unsigned long long)cols);
    float rr = rate[i * (unsigned long long)rate_stride + (unsigned long long)rate_off];
    float retain = plane[i] * (1.0f - rr);
    out[i] = retain + padded[(unsigned long long)(r + R) * (unsigned long long)fc + (c + R)];
}
"#;

/// Compiled launchers for the route-A device kernels.
struct FftKernels {
    scatter_g: CudaFunction,
    complex_mul: CudaFunction,
    combine: CudaFunction,
}

struct Scratch {
    pad: CudaSlice<f32>,
    spec: CudaSlice<float2>,
    padded: CudaSlice<f32>,
    out: CudaSlice<f32>,
}

/// Run one deterministic FFT migration tick for a spatial model.
///
/// ## Parameters
/// - `rows`, `cols`: grid dimensions; `n_demes = rows * cols`.
/// - `n_ages`, `n_ztypes`: model dimensions.
/// - `kernel`: `k*k` row-major Gaussian kernel (odd `k`).
/// - `z`: `(n_demes,)` normalization field `K' ⊛ m`.
/// - `rate`: `(n_demes, 2, n_ages)` migration rate, deme-major.
/// - `ind`: batch-minor `(2, A, Z, n_demes)`; `sperm`: `(A, Z, Z, n_demes)`.
///
/// ## Returns
/// `(ind', sperm')` in the same batch-minor layout.
///
/// ## Errors
/// Returns a description on CUDA/plan/launch failure or shape mismatch.
#[allow(clippy::too_many_arguments)] // mirrors the kernel's flat argument list
pub fn gpu_fft_migrate(
    rows: usize,
    cols: usize,
    n_ages: usize,
    n_ztypes: usize,
    kernel: &[f32],
    z: &[f32],
    rate: &[f32],
    ind: &[f32],
    sperm: &[f32],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let n_demes = rows * cols;
    let k = (kernel.len() as f64).sqrt() as usize;
    if kernel.len() != k * k || k % 2 == 0 {
        return Err("kernel must be odd and square".to_owned());
    }
    let r = (k - 1) / 2;
    let fr = rows + k - 1;
    let fc = cols + k - 1;
    let spec_n = fr * (fc / 2 + 1);
    let ind_planes = 2 * n_ages * n_ztypes;
    let sperm_planes = n_ages * n_ztypes * n_ztypes;
    if z.len() != n_demes || rate.len() != n_demes * 2 * n_ages {
        return Err("z/rate length mismatch".to_owned());
    }
    if ind.len() != n_demes * ind_planes || sperm.len() != n_demes * sperm_planes {
        return Err("state length mismatch".to_owned());
    }
    let rate_stride = 2 * n_ages;

    let context = CudaContext::new(0).map_err(|e| format!("cuda context: {e}"))?;
    let stream = context.default_stream();
    let (major, minor) = context
        .compute_capability()
        .map_err(|e| format!("compute capability: {e}"))?;
    let opts = CompileOptions {
        options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(FFT_MIGRATE_SOURCE, opts).map_err(|e| format!("nvrtc: {e}"))?;
    let module = context.load_module(ptx).map_err(|e| format!("load: {e}"))?;
    let kernels = FftKernels {
        scatter_g: module
            .load_function("scatter_g")
            .map_err(|e| format!("load scatter_g: {e}"))?,
        complex_mul: module
            .load_function("complex_mul")
            .map_err(|e| format!("load complex_mul: {e}"))?,
        combine: module
            .load_function("combine")
            .map_err(|e| format!("load combine: {e}"))?,
    };

    let r2c = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_R2C, stream.clone())
        .map_err(|e| format!("r2c plan: {e}"))?;
    let c2r = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_C2R, stream.clone())
        .map_err(|e| format!("c2r plan: {e}"))?;

    let z_d = stream.clone_htod(z).map_err(|e| format!("{e}"))?;
    let rate_d = stream.clone_htod(rate).map_err(|e| format!("{e}"))?;

    // Kernel spectrum (zero-padded kernel, r2c) computed once.
    let mut kernel_pad = vec![0f32; fr * fc];
    let center = k / 2;
    for kr in 0..k {
        for kc in 0..k {
            if kr == center && kc == center {
                continue; // K' excludes the center; retention (1-rate) covers it.
            }
            kernel_pad[kr * fc + kc] = kernel[kr * k + kc];
        }
    }
    let kp_d = stream.clone_htod(&kernel_pad).map_err(|e| format!("{e}"))?;
    let mut ks_d = stream
        .alloc_zeros::<float2>(spec_n)
        .map_err(|e| format!("{e}"))?;
    r2c.exec_r2c(&kp_d, &mut ks_d)
        .map_err(|e| format!("kernel r2c: {e}"))?;

    let mut scratch = Scratch {
        pad: stream
            .alloc_zeros::<f32>(fr * fc)
            .map_err(|e| format!("{e}"))?,
        spec: stream
            .alloc_zeros::<float2>(spec_n)
            .map_err(|e| format!("{e}"))?,
        padded: stream
            .alloc_zeros::<f32>(fr * fc)
            .map_err(|e| format!("{e}"))?,
        out: stream
            .alloc_zeros::<f32>(n_demes)
            .map_err(|e| format!("{e}"))?,
    };
    let inv = 1.0f32 / (fr * fc) as f32;
    let total_pad = (fr * fc) as u64;
    let total_n = n_demes as u64;
    let (rows_i, cols_i, fr_i, fc_i, r_i) =
        (rows as i32, cols as i32, fr as i32, fc as i32, r as i32);
    let spec_n_u = spec_n as u64;
    let rate_stride_i = rate_stride as i32;

    let mut run_plane = |plane: &[f32], rate_off: i32| -> Result<Vec<f32>, String> {
        let plane_d = stream.clone_htod(plane).map_err(|e| format!("{e}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(total_pad as u32);
            let mut lb = stream.launch_builder(&kernels.scatter_g);
            lb.arg(&plane_d);
            lb.arg(&rate_d);
            lb.arg(&z_d);
            lb.arg(&mut scratch.pad);
            lb.arg(&total_pad);
            lb.arg(&rows_i);
            lb.arg(&cols_i);
            lb.arg(&fr_i);
            lb.arg(&fc_i);
            lb.arg(&r_i);
            lb.arg(&rate_stride_i);
            lb.arg(&rate_off);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("scatter_g: {e}"))?;
        }
        r2c.exec_r2c(&scratch.pad, &mut scratch.spec)
            .map_err(|e| format!("plane r2c: {e}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(spec_n as u32);
            let mut lb = stream.launch_builder(&kernels.complex_mul);
            lb.arg(&mut scratch.spec);
            lb.arg(&ks_d);
            lb.arg(&spec_n_u);
            lb.arg(&inv);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("complex_mul: {e}"))?;
        }
        c2r.exec_c2r(&mut scratch.spec, &mut scratch.padded)
            .map_err(|e| format!("plane c2r: {e}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(total_n as u32);
            let mut lb = stream.launch_builder(&kernels.combine);
            lb.arg(&plane_d);
            lb.arg(&rate_d);
            lb.arg(&scratch.padded);
            lb.arg(&mut scratch.out);
            lb.arg(&total_n);
            lb.arg(&cols_i);
            lb.arg(&fr_i);
            lb.arg(&fc_i);
            lb.arg(&r_i);
            lb.arg(&rate_stride_i);
            lb.arg(&rate_off);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("combine: {e}"))?;
        }
        stream.clone_dtoh(&scratch.out).map_err(|e| format!("{e}"))
    };

    let mut out_ind = vec![0f32; ind.len()];
    for sex in 0..2usize {
        for age in 0..n_ages {
            for zt in 0..n_ztypes {
                let p = (sex * n_ages + age) * n_ztypes + zt;
                let plane = &ind[p * n_demes..(p + 1) * n_demes];
                let out = run_plane(plane, (sex * n_ages + age) as i32)?;
                out_ind[p * n_demes..(p + 1) * n_demes].copy_from_slice(&out);
            }
        }
    }
    let mut out_sperm = vec![0f32; sperm.len()];
    for age in 0..n_ages {
        for fz in 0..n_ztypes {
            for mz in 0..n_ztypes {
                let p = (age * n_ztypes + fz) * n_ztypes + mz;
                let plane = &sperm[p * n_demes..(p + 1) * n_demes];
                let out = run_plane(plane, age as i32)?;
                out_sperm[p * n_demes..(p + 1) * n_demes].copy_from_slice(&out);
            }
        }
    }
    Ok((out_ind, out_sperm))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::spatial::migrate_csr_deterministic;

    fn hex_kernel(k: usize, sigma: f64) -> Vec<f64> {
        let c = (k / 2) as f64;
        let mut ker = vec![0f64; k * k];
        let mut total = 0f64;
        for kr in 0..k {
            for kc in 0..k {
                let dr = kr as f64 - c;
                let dc = kc as f64 - c;
                let d2 = dr * dr + dc * dc + dr * dc;
                let w = (-d2 / (2.0 * sigma * sigma)).exp();
                ker[kr * k + kc] = w;
                total += w;
            }
        }
        for w in ker.iter_mut() {
            *w /= total;
        }
        ker
    }

    fn build_csr(
        rows: usize,
        cols: usize,
        k: usize,
        ker: &[f64],
    ) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let c = k / 2;
        let mut indptr = vec![0i64];
        let mut dest = Vec::new();
        let mut weights = Vec::new();
        for s in 0..rows * cols {
            let (r, cc) = (s / cols, s % cols);
            let mut offs: Vec<(i64, i64, f64)> = Vec::new();
            let mut sum = 0f64;
            for kr in 0..k {
                for kc in 0..k {
                    if kr == c && kc == c {
                        continue;
                    }
                    let w = ker[kr * k + kc];
                    if w <= 0.0 {
                        continue;
                    }
                    let (dr, dc) = (kr as i64 - c as i64, kc as i64 - c as i64);
                    let (nr, nc) = (r as i64 + dr, cc as i64 + dc);
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 {
                        continue;
                    }
                    offs.push((dr, dc, w));
                    sum += w;
                }
            }
            for (dr, dc, w) in offs {
                dest.push((r as i64 + dr) * cols as i64 + (cc as i64 + dc));
                weights.push(w / sum);
            }
            indptr.push(dest.len() as i64);
        }
        (indptr, dest, weights)
    }

    fn brute_z(rows: usize, cols: usize, k: usize, ker: &[f64]) -> Vec<f64> {
        let c = k / 2;
        let mut z = vec![0f64; rows * cols];
        for s in 0..rows * cols {
            let (r, cc) = (s / cols, s % cols);
            for kr in 0..k {
                for kc in 0..k {
                    if kr == c && kc == c {
                        continue;
                    }
                    let (dr, dc) = (kr as i64 - c as i64, kc as i64 - c as i64);
                    let (nr, nc) = (r as i64 + dr, cc as i64 + dc);
                    if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
                        z[s] += ker[kr * k + kc];
                    }
                }
            }
        }
        z
    }

    #[test]
    fn fft_migrate_matches_csr() {
        let (rows, cols, a, ztypes, k, sigma) = (4usize, 3usize, 2usize, 2usize, 3usize, 1.0f64);
        let n = rows * cols;
        let ker = hex_kernel(k, sigma);
        let (indptr, dest, weights) = build_csr(rows, cols, k, &ker);
        let z64 = brute_z(rows, cols, k, &ker);

        let ind_planes = 2 * a * ztypes;
        let sperm_planes = a * ztypes * ztypes;
        let mut rate = vec![0f64; n * 2 * a];
        let mut ind_dm = vec![0f64; n * ind_planes];
        let mut sperm_dm = vec![0f64; n * sperm_planes];
        let mut state = 0x1234u64;
        let mut rnd = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64
        };
        for v in rate.iter_mut() {
            *v = 0.1 + 0.4 * rnd();
        }
        for v in ind_dm.iter_mut() {
            *v = 100.0 * rnd();
        }
        for v in sperm_dm.iter_mut() {
            *v = 10.0 * rnd();
        }

        let (cpu_ind, cpu_sperm) = migrate_csr_deterministic(
            &ind_dm, &sperm_dm, &indptr, &dest, &weights, &rate, false, n, a, ztypes,
        )
        .expect("cpu csr migration");

        // Convert deme-major -> batch-minor (2,A,Z,B) / (A,Z,Z,B).
        let mut ind_bm = vec![0f32; ind_dm.len()];
        for d in 0..n {
            for p in 0..ind_planes {
                ind_bm[p * n + d] = ind_dm[d * ind_planes + p] as f32;
            }
        }
        let mut sperm_bm = vec![0f32; sperm_dm.len()];
        for d in 0..n {
            for p in 0..sperm_planes {
                sperm_bm[p * n + d] = sperm_dm[d * sperm_planes + p] as f32;
            }
        }

        let ker_f32: Vec<f32> = ker.iter().map(|v| *v as f32).collect();
        let z_f32: Vec<f32> = z64.iter().map(|v| *v as f32).collect();
        let rate_f32: Vec<f32> = rate.iter().map(|v| *v as f32).collect();
        let (gpu_ind_bm, gpu_sperm_bm) = gpu_fft_migrate(
            rows, cols, a, ztypes, &ker_f32, &z_f32, &rate_f32, &ind_bm, &sperm_bm,
        )
        .expect("gpu fft migration");

        let mut max_rel = 0f64;
        for d in 0..n {
            for p in 0..ind_planes {
                let got = gpu_ind_bm[p * n + d] as f64;
                let want = cpu_ind[d * ind_planes + p];
                max_rel = max_rel.max((got - want).abs() / want.abs().max(1e-6));
            }
            for p in 0..sperm_planes {
                let got = gpu_sperm_bm[p * n + d] as f64;
                let want = cpu_sperm[d * sperm_planes + p];
                max_rel = max_rel.max((got - want).abs() / want.abs().max(1e-6));
            }
        }
        assert!(max_rel < 1e-4, "fft vs csr max relative error {max_rel}");
    }
}
