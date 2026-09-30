//! Route-A device FFT migration (GPU cuFFT), deterministic spatial models.
//!
//! Implements `Hex_model_recon.md` §4 route A: the deterministic spatial
//! migration is a per-plane normalized convolution (productization design
//! §3.2c), so each `(sex, age, ztype)` plane advances with
//!
//! `plane' = plane·(1−rate) + K' ⊛ (rate·plane / Z)`
//!
//! where `⊛` is a zero-outside linear convolution computed with cuFFT.
//! [`FftMigrator`] owns the plan buffers, compiled kernels and cuFFT plans and
//! reuses a caller-provided stream, so the executor (S2c) can wire it into its
//! existing double-buffered state.  [`gpu_fft_migrate`] is a host-slice wrapper
//! used by the tests.  Compiled only under the `gpu` feature.

use std::sync::Arc;

use cudarc::cufft::sys::{cufftType, float2};
use cudarc::cufft::CudaFft;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const FFT_MIGRATE_SOURCE: &str = r#"
struct cf2 { float x; float y; };

extern "C" __global__ void scatter_g(
    const float* plane, const float* rate, const float* zs, float* pad,
    unsigned long long total, int rows, int cols, int fr, int fc, int R,
    int rate_stride, int rate_off, unsigned long long plane_base)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int r = (int)(i / (unsigned long long)fc);
    int c = (int)(i % (unsigned long long)fc);
    if (r < rows && c < cols) {
        unsigned long long idx = (unsigned long long)r * cols + c;
        float rr = rate[idx * (unsigned long long)rate_stride + (unsigned long long)rate_off];
        pad[i] = plane[plane_base + idx] * rr / zs[idx];
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
    unsigned long long n, int cols, int fr, int fc, int R, int rate_stride, int rate_off,
    unsigned long long plane_base)
{
    unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = (int)(i / (unsigned long long)cols);
    int c = (int)(i % (unsigned long long)cols);
    float rr = rate[i * (unsigned long long)rate_stride + (unsigned long long)rate_off];
    float retain = plane[plane_base + i] * (1.0f - rr);
    out[plane_base + i] = retain + padded[(unsigned long long)(r + R) * (unsigned long long)fc + (c + R)];
}
"#;

/// Host-side route-A plan handed from Python to the session at construction.
#[derive(Clone)]
pub struct MigrationPlanHost {
    /// Grid rows.
    pub rows: usize,
    /// Grid columns.
    pub cols: usize,
    /// Row-major `k*k` kernel (odd `k`).
    pub kernel: Vec<f32>,
    /// `(rows*cols,)` normalization field `K' ⊛ m`.
    pub z: Vec<f32>,
}

/// Reusable route-A device migrator for one rectangular topology and kernel.
pub struct FftMigrator {
    rows: usize,
    cols: usize,
    n_ages: usize,
    n_ztypes: usize,
    r: usize,
    fr: usize,
    fc: usize,
    spec_n: usize,
    rate_stride: i32,
    inv: f32,
    scatter_g: CudaFunction,
    complex_mul: CudaFunction,
    combine: CudaFunction,
    r2c: CudaFft,
    c2r: CudaFft,
    ks: CudaSlice<float2>,
    z: CudaSlice<f32>,
    pad: CudaSlice<f32>,
    spec: CudaSlice<float2>,
    padded: CudaSlice<f32>,
}

fn kernel_pad(rows: usize, cols: usize, k: usize, kernel: &[f32]) -> (usize, usize, Vec<f32>) {
    let fr = rows + k - 1;
    let fc = cols + k - 1;
    let center = k / 2;
    let mut pad = vec![0f32; fr * fc];
    for kr in 0..k {
        for kc in 0..k {
            if kr == center && kc == center {
                continue; // K' excludes the center; retention covers it.
            }
            pad[kr * fc + kc] = kernel[kr * k + kc];
        }
    }
    (fr, fc, pad)
}

impl FftMigrator {
    /// Build the migrator for `rows x cols` with an odd `k x k` kernel and the
    /// `(n_demes,)` normalization field `z`.
    ///
    /// ## Errors
    /// Returns a description on kernel shape / NVRTC / plan failures.
    #[allow(clippy::too_many_arguments)] // flat plan parameters
    pub fn new(
        context: &Arc<CudaContext>,
        rows: usize,
        cols: usize,
        n_ages: usize,
        n_ztypes: usize,
        kernel: &[f32],
        z: &[f32],
    ) -> Result<Self, String> {
        let k = (kernel.len() as f64).sqrt() as usize;
        if kernel.len() != k * k || k % 2 == 0 {
            return Err("kernel must be odd and square".to_owned());
        }
        let n_demes = rows * cols;
        if z.len() != n_demes {
            return Err("z length mismatch".to_owned());
        }
        let r = (k - 1) / 2;
        let (fr, fc, pad_host) = kernel_pad(rows, cols, k, kernel);
        let spec_n = fr * (fc / 2 + 1);

        let (major, minor) = context
            .compute_capability()
            .map_err(|e| format!("compute capability: {e}"))?;
        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        let ptx =
            compile_ptx_with_opts(FFT_MIGRATE_SOURCE, opts).map_err(|e| format!("nvrtc: {e}"))?;
        let module = context.load_module(ptx).map_err(|e| format!("load: {e}"))?;
        let stream = context.default_stream();
        let scatter_g = module
            .load_function("scatter_g")
            .map_err(|e| format!("load scatter_g: {e}"))?;
        let complex_mul = module
            .load_function("complex_mul")
            .map_err(|e| format!("load complex_mul: {e}"))?;
        let combine = module
            .load_function("combine")
            .map_err(|e| format!("load combine: {e}"))?;
        let r2c = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_R2C, stream.clone())
            .map_err(|e| format!("r2c plan: {e}"))?;
        let c2r = CudaFft::plan_2d(fr as i32, fc as i32, cufftType::CUFFT_C2R, stream.clone())
            .map_err(|e| format!("c2r plan: {e}"))?;

        let kp_d = stream.clone_htod(&pad_host).map_err(|e| format!("{e}"))?;
        let mut ks = stream
            .alloc_zeros::<float2>(spec_n)
            .map_err(|e| format!("{e}"))?;
        r2c.exec_r2c(&kp_d, &mut ks)
            .map_err(|e| format!("kernel r2c: {e}"))?;

        Ok(Self {
            rows,
            cols,
            n_ages,
            n_ztypes,
            r,
            fr,
            fc,
            spec_n,
            rate_stride: (2 * n_ages) as i32,
            inv: 1.0f32 / (fr * fc) as f32,
            scatter_g,
            complex_mul,
            combine,
            r2c,
            c2r,
            ks,
            z: stream.clone_htod(z).map_err(|e| format!("{e}"))?,
            pad: stream
                .alloc_zeros::<f32>(fr * fc)
                .map_err(|e| format!("{e}"))?,
            spec: stream
                .alloc_zeros::<float2>(spec_n)
                .map_err(|e| format!("{e}"))?,
            padded: stream
                .alloc_zeros::<f32>(fr * fc)
                .map_err(|e| format!("{e}"))?,
        })
    }

    /// Number of `(2,A,Z,B)` + `(A,Z,Z,B)` planes the migrator expects.
    pub fn plane_count(&self) -> usize {
        2 * self.n_ages * self.n_ztypes + self.n_ages * self.n_ztypes * self.n_ztypes
    }

    /// Estimate the resident FFT footprint in bytes (plan buffers only).
    pub fn footprint_bytes(&self) -> usize {
        let f32b = std::mem::size_of::<f32>();
        let cplx = std::mem::size_of::<float2>();
        self.pad.len() * f32b
            + self.padded.len() * f32b
            + self.spec.len() * cplx
            + self.ks.len() * cplx
            + self.z.len() * f32b
    }

    fn run_plane(
        &mut self,
        stream: &Arc<CudaStream>,
        plane: &CudaSlice<f32>,
        rate: &CudaSlice<f32>,
        rate_off: i32,
        out: &mut CudaSlice<f32>,
        plane_base: u64,
    ) -> Result<(), String> {
        let total = (self.fr * self.fc) as u64;
        let n = (self.rows * self.cols) as u64;
        let spec_n_u = self.spec_n as u64;
        let (rows_i, cols_i, fr_i, fc_i, r_i) = (
            self.rows as i32,
            self.cols as i32,
            self.fr as i32,
            self.fc as i32,
            self.r as i32,
        );
        {
            let cfg = LaunchConfig::for_num_elems(total as u32);
            let mut lb = stream.launch_builder(&self.scatter_g);
            lb.arg(plane);
            lb.arg(rate);
            lb.arg(&self.z);
            lb.arg(&mut self.pad);
            lb.arg(&total);
            lb.arg(&rows_i);
            lb.arg(&cols_i);
            lb.arg(&fr_i);
            lb.arg(&fc_i);
            lb.arg(&r_i);
            lb.arg(&self.rate_stride);
            lb.arg(&rate_off);
            lb.arg(&plane_base);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("scatter_g: {e}"))?;
        }
        self.r2c
            .exec_r2c(&self.pad, &mut self.spec)
            .map_err(|e| format!("plane r2c: {e}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(self.spec_n as u32);
            let mut lb = stream.launch_builder(&self.complex_mul);
            lb.arg(&mut self.spec);
            lb.arg(&self.ks);
            lb.arg(&spec_n_u);
            lb.arg(&self.inv);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("complex_mul: {e}"))?;
        }
        self.c2r
            .exec_c2r(&mut self.spec, &mut self.padded)
            .map_err(|e| format!("plane c2r: {e}"))?;
        {
            let cfg = LaunchConfig::for_num_elems(n as u32);
            let mut lb = stream.launch_builder(&self.combine);
            lb.arg(plane);
            lb.arg(rate);
            lb.arg(&self.padded);
            lb.arg(out);
            lb.arg(&n);
            lb.arg(&cols_i);
            lb.arg(&fr_i);
            lb.arg(&fc_i);
            lb.arg(&r_i);
            lb.arg(&self.rate_stride);
            lb.arg(&rate_off);
            lb.arg(&plane_base);
            unsafe { lb.launch(cfg) }.map_err(|e| format!("combine: {e}"))?;
        }
        Ok(())
    }

    /// Advance the whole batch-minor state one tick, writing into `ind_out` and
    /// `sperm_out` (double-buffered by the caller).
    ///
    /// ## Errors
    /// Returns a description on launch failures or length mismatch.
    pub fn migrate_inplace(
        &mut self,
        stream: &Arc<CudaStream>,
        ind: &CudaSlice<f32>,
        ind_out: &mut CudaSlice<f32>,
        sperm: &CudaSlice<f32>,
        sperm_out: &mut CudaSlice<f32>,
        rate: &CudaSlice<f32>,
    ) -> Result<(), String> {
        let n_demes = self.rows * self.cols;
        let ind_stride = 2 * self.n_ages * self.n_ztypes;
        let sperm_stride = self.n_ages * self.n_ztypes * self.n_ztypes;
        if ind.len() != n_demes * ind_stride
            || ind_out.len() != ind.len()
            || sperm.len() != n_demes * sperm_stride
            || sperm_out.len() != sperm.len()
        {
            return Err("state length mismatch".to_owned());
        }
        for sex in 0..2usize {
            for age in 0..self.n_ages {
                for zt in 0..self.n_ztypes {
                    let p = (sex * self.n_ages + age) * self.n_ztypes + zt;
                    self.run_plane(
                        stream,
                        ind,
                        rate,
                        (sex * self.n_ages + age) as i32,
                        ind_out,
                        (p * n_demes) as u64,
                    )?;
                }
            }
        }
        for age in 0..self.n_ages {
            for fz in 0..self.n_ztypes {
                for mz in 0..self.n_ztypes {
                    let p = (age * self.n_ztypes + fz) * self.n_ztypes + mz;
                    self.run_plane(
                        stream,
                        sperm,
                        rate,
                        age as i32,
                        sperm_out,
                        (p * n_demes) as u64,
                    )?;
                }
            }
        }
        Ok(())
    }
}

/// Host-slice convenience wrapper (tests, and a faithful reference).
///
/// ## Errors
/// Returns a description on CUDA/plan/launch failure or shape mismatch.
#[allow(clippy::too_many_arguments)] // mirrors the flat kernel argument list
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
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    if rate.len() != n_demes * 2 * n_ages
        || ind.len() != n_demes * ind_stride
        || sperm.len() != n_demes * sperm_stride
    {
        return Err("state/rate length mismatch".to_owned());
    }
    let context = CudaContext::new(0).map_err(|e| format!("cuda context: {e}"))?;
    let stream = context.default_stream();
    let mut migrator = FftMigrator::new(&context, rows, cols, n_ages, n_ztypes, kernel, z)?;
    let ind_d = stream.clone_htod(ind).map_err(|e| format!("{e}"))?;
    let sperm_d = stream.clone_htod(sperm).map_err(|e| format!("{e}"))?;
    let rate_d = stream.clone_htod(rate).map_err(|e| format!("{e}"))?;
    let mut ind_out = stream
        .alloc_zeros::<f32>(ind.len())
        .map_err(|e| format!("{e}"))?;
    let mut sperm_out = stream
        .alloc_zeros::<f32>(sperm.len())
        .map_err(|e| format!("{e}"))?;
    migrator.migrate_inplace(
        &stream,
        &ind_d,
        &mut ind_out,
        &sperm_d,
        &mut sperm_out,
        &rate_d,
    )?;
    let out_ind = stream.clone_dtoh(&ind_out).map_err(|e| format!("{e}"))?;
    let out_sperm = stream.clone_dtoh(&sperm_out).map_err(|e| format!("{e}"))?;
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

    fn check_fixture(
        rows: usize,
        cols: usize,
        a: usize,
        ztypes: usize,
        k: usize,
        sigma: f64,
        stay_after: bool,
    ) {
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
            &ind_dm, &sperm_dm, &indptr, &dest, &weights, &rate, stay_after, n, a, ztypes,
        )
        .expect("cpu csr migration");

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

    #[test]
    fn fft_migrate_matches_csr() {
        for &(rows, cols, a, zt, k, sigma) in &[
            (4usize, 3usize, 2usize, 2usize, 3usize, 1.0f64),
            (7usize, 5usize, 1usize, 2usize, 5usize, 1.4f64),
            (3usize, 3usize, 1usize, 1usize, 3usize, 0.9f64),
        ] {
            check_fixture(rows, cols, a, zt, k, sigma, false);
            check_fixture(rows, cols, a, zt, k, sigma, true);
        }
    }
}
