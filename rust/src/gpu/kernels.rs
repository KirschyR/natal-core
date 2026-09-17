//! Device kernel sources and launchers.
//!
//! Kernels are embedded as CUDA C strings and compiled at run time with NVRTC
//! (§D1), so no CUDA toolkit is needed at build time. Each launcher is a thin,
//! typed wrapper over the raw launch so callers never touch `cudarc` argument
//! marshalling directly.
//!
//! ## Layout contract
//!
//! Kernels operate on the batch-minor device layout produced by
//! [`super::layout::batch_to_inner`]: the batch axis is innermost, so a warp's
//! neighbouring threads read neighbouring addresses. For the aging kernel this
//! means the stride between adjacent age classes is a compile-time-friendly
//! `trailing_dims · n_batch`.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

/// CUDA C for the age-shift kernel.
///
/// Every age class moves down one slot (the oldest is dropped) and age 0 is
/// zeroed, matching `kernels::age_structured::aging` exactly. It reads `src`
/// and writes `dst` out of place: in-place shifting would race, because thread
/// `i` reads `i - age_stride` while thread `i - age_stride` writes it.
const AGING_SOURCE: &str = r#"
extern "C" __global__ void age_shift(
    const float* src,
    float* dst,
    unsigned long long total,
    int n_ages,
    unsigned long long age_stride)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    unsigned long long age = (i / age_stride) % (unsigned long long)n_ages;
    dst[i] = (age == 0ULL) ? 0.0f : src[i - age_stride];
}
"#;

/// Largest `n_ages` the density kernel supports (fixed local arrays).
const MAX_AGES: usize = 64;

/// CUDA C for the density-regulation scaling kernel.
///
/// One thread per batch element evaluates the equilibrium metrics and the
/// growth curve exactly as `kernels::equilibrium::equilibrium_metrics_core`
/// plus `kernels::density_regulation::regulation_scaling` do on the host, and
/// writes the scalar scaling factor. `ind` is batch-minor, so a batch element's
/// cells are strided by `n_batch`.
const DENSITY_SOURCE: &str = r#"
#define MAX_AGES 64

extern "C" __global__ void density_scaling(
    const float* ind,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int new_adult_age,
    const float* survival_rates,
    const float* reproduction_rates,
    const float* fertility,
    const float* competition_weights,
    const int* equilibrium_declared,
    const float* equilibrium_distribution,
    const float* external_expected_eggs,
    const float* carrying_capacity,
    const float* eggs_per_female,
    const float* sex_ratio,
    const int* growth_mode,
    const float* low_density_growth_rate,
    float* scaling_out)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    const float* surv = survival_rates + b * 2 * A;
    const float* compw = competition_weights + b * A;
    const float* fert = fertility + b * A;
    const float* repro = reproduction_rates + b * A;
    float eggs = eggs_per_female[b];
    float sr = sex_ratio[b];

    // p_reproducing[age] = clamp01(reproduction_rates[age]).
    float p_rep[MAX_AGES];
    for (int a = 0; a < A; ++a) {
        p_rep[a] = 0.0f;
    }
    for (int a = new_adult_age; a < A; ++a) {
        float x = repro[a];
        p_rep[a] = (x <= 0.0f) ? 0.0f : ((x >= 1.0f) ? 1.0f : x);
    }

    float dist[2 * MAX_AGES];
    float produced = 0.0f;
    float total_age_1;
    if (equilibrium_declared[b] != 0) {
        const float* declared = equilibrium_distribution + b * 2 * A;
        for (int a = 0; a < 2 * A; ++a) {
            dist[a] = declared[a];
        }
        for (int a = new_adult_age; a < A; ++a) {
            produced += dist[a] * p_rep[a] * fert[a] * eggs;
        }
        total_age_1 = dist[1] + dist[A + 1];
    } else {
        total_age_1 = carrying_capacity[b];
        for (int a = 0; a < 2 * A; ++a) {
            dist[a] = 0.0f;
        }
        dist[1] = total_age_1 * sr;
        dist[A + 1] = total_age_1 * (1.0f - sr);
        for (int a = 2; a < A; ++a) {
            dist[a] = dist[a - 1] * surv[a - 1];
            dist[A + a] = dist[A + a - 1] * surv[A + a - 1];
        }
        for (int a = new_adult_age; a < A; ++a) {
            produced += dist[a] * p_rep[a] * fert[a] * eggs;
        }
    }

    float expected_comp = produced * compw[0];
    for (int a = 1; a < new_adult_age; ++a) {
        expected_comp += (dist[a] + dist[A + a]) * compw[a];
    }
    float s0 = sr * surv[0] + (1.0f - sr) * surv[A];
    float extern_eggs = external_expected_eggs[b];
    float surv_eggs = (extern_eggs < 0.0f) ? produced : extern_eggs;
    float expected_surv = (surv_eggs > 0.0f && s0 > 1e-10f)
        ? total_age_1 / (surv_eggs * s0)
        : 1.0f;

    int mode = growth_mode[b];
    float scaling;
    if (mode == 0) {
        scaling = 1.0f;
    } else if (mode == 1) {
        float female_sum = 0.0f;
        float male_sum = 0.0f;
        for (int z = 0; z < Z; ++z) {
            female_sum += ind[((0 * A + 0) * Z + z) * n_batch + b];
            male_sum += ind[((1 * A + 0) * Z + z) * n_batch + b];
        }
        float total_age_0 = female_sum + male_sum;
        float cap = carrying_capacity[b];
        scaling = (total_age_0 > 0.0f)
            ? fminf(cap / total_age_0, 1.0f)
            : 1.0f;
    } else {
        float actual = 0.0f;
        for (int age = 0; age < new_adult_age; ++age) {
            float female_sum = 0.0f;
            float male_sum = 0.0f;
            for (int z = 0; z < Z; ++z) {
                female_sum += ind[((0 * A + age) * Z + z) * n_batch + b];
                male_sum += ind[((1 * A + age) * Z + z) * n_batch + b];
            }
            actual += (female_sum + male_sum) * compw[age];
        }
        float ratio = (expected_comp > 0.0f) ? actual / expected_comp : 1.0f;
        float r = low_density_growth_rate[b];
        float g;
        if (mode == 2) {
            g = fmaxf(0.0f, r - (r - 1.0f) * ratio);
        } else if (mode == 3) {
            g = r / (1.0f + (r - 1.0f) * ratio);
        } else if (mode == 4) {
            g = powf(r, 1.0f - ratio);
        } else {
            // Custom slots (>= 5) are not portable; the caller rejects them.
            g = 1.0f;
        }
        scaling = g * expected_surv;
    }
    scaling_out[b] = scaling;
}
"#;

/// Device arguments for [`Kernels::density_scaling`].
///
/// The per-deme ecology columns are uploaded as flat batch-major arrays; all
/// lengths follow the `n_batch`-element batch axis.
pub struct DensityBuffers<'a> {
    /// Batch-minor individual counts, `(2, A, Z, B)`.
    pub ind: &'a CudaSlice<f32>,
    /// `(B, 2, A)` age survival rates.
    pub survival_rates: &'a CudaSlice<f32>,
    /// `(B, A)` reproduction participation.
    pub reproduction_rates: &'a CudaSlice<f32>,
    /// `(B, A)` relative fertility.
    pub fertility: &'a CudaSlice<f32>,
    /// `(B, A)` juvenile competition weights.
    pub competition_weights: &'a CudaSlice<f32>,
    /// `(B,)` declaration flags, `0`/`1`.
    pub equilibrium_declared: &'a CudaSlice<i32>,
    /// `(B, 2, A)` declared distribution (zeros where undeclared).
    pub equilibrium_distribution: &'a CudaSlice<f32>,
    /// `(B,)` external egg override, negative for unused.
    pub external_expected_eggs: &'a CudaSlice<f32>,
    /// `(B,)` carrying capacity.
    pub carrying_capacity: &'a CudaSlice<f32>,
    /// `(B,)` eggs per female.
    pub eggs_per_female: &'a CudaSlice<f32>,
    /// `(B,)` sex ratio.
    pub sex_ratio: &'a CudaSlice<f32>,
    /// `(B,)` growth mode ids.
    pub growth_mode: &'a CudaSlice<i32>,
    /// `(B,)` low-density growth rates.
    pub low_density_growth_rate: &'a CudaSlice<f32>,
    /// `(B,)` scaling-factor output.
    pub scaling_out: &'a mut CudaSlice<f32>,
}

/// The compiled kernel set, loaded once per executor.
pub struct Kernels {
    /// `age_shift(const float*, float*, u64, int, u64)`.
    age_shift: CudaFunction,
    /// `density_scaling(...)`.
    density_scaling: CudaFunction,
}

impl Kernels {
    /// Compile [`AGING_SOURCE`] and load its kernels for `context`.
    ///
    /// ## Parameters
    /// - `context`: The device context the module is loaded into.
    ///
    /// ## Returns
    /// The ready kernel set, or a description of the compile/load failure.
    ///
    /// ## Errors
    /// Returns a description when NVRTC is unavailable, compilation fails, or
    /// the module or a kernel cannot be loaded.
    pub fn load(context: &Arc<CudaContext>) -> Result<Self, String> {
        let (major, minor) = context
            .compute_capability()
            .map_err(|err| format!("compute capability query failed: {err}"))?;
        let opts = CompileOptions {
            options: vec![format!("--gpu-architecture=compute_{major}{minor}")],
            ..Default::default()
        };
        let source = format!("{AGING_SOURCE}\n{DENSITY_SOURCE}");
        let ptx = compile_ptx_with_opts(source, opts)
            .map_err(|err| format!("NVRTC compilation failed: {err}"))?;
        let module = context
            .load_module(ptx)
            .map_err(|err| format!("loading the compiled module failed: {err}"))?;
        let age_shift = module
            .load_function("age_shift")
            .map_err(|err| format!("loading kernel `age_shift` failed: {err}"))?;
        let density_scaling = module
            .load_function("density_scaling")
            .map_err(|err| format!("loading kernel `density_scaling` failed: {err}"))?;
        Ok(Self {
            age_shift,
            density_scaling,
        })
    }

    /// Shift every age class down one slot and zero age 0, element-wise.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `src`: Batch-minor source slice of `total` elements.
    /// - `dst`: Batch-minor destination slice of `total` elements.
    /// - `total`: Number of elements in `src`/`dst`.
    /// - `n_ages`: Length of the age axis.
    /// - `age_stride`: Elements between adjacent age classes,
    ///   `trailing_dims · n_batch`.
    ///
    /// ## Returns
    /// `Ok(())` once the launch is enqueued (the caller synchronizes through
    /// the stream).
    ///
    /// ## Errors
    /// Returns a description when the launch parameters are inconsistent or
    /// the driver rejects the launch.
    pub fn age_shift(
        &self,
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
        dst: &mut CudaSlice<f32>,
        total: usize,
        n_ages: usize,
        age_stride: usize,
    ) -> Result<(), String> {
        if src.len() != total || dst.len() != total {
            return Err(format!(
                "age_shift expected {total} elements, got src={} dst={}",
                src.len(),
                dst.len()
            ));
        }
        if total == 0 {
            return Ok(());
        }
        if age_stride == 0 || n_ages == 0 {
            return Err("age_shift requires non-zero n_ages and age_stride".to_owned());
        }
        let total = total as u64;
        let n_ages = n_ages as i32;
        let age_stride = age_stride as u64;
        let config = LaunchConfig::for_num_elems(total as u32);
        let mut launch = stream.launch_builder(&self.age_shift);
        launch.arg(src);
        launch.arg(dst);
        launch.arg(&total);
        launch.arg(&n_ages);
        launch.arg(&age_stride);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("age_shift launch failed: {err}"))
    }

    /// Compute the density-regulation scaling factor for every batch element.
    ///
    /// Mirrors the host `scaling_factor` composition: equilibrium metrics from
    /// the ecology column, then the growth curve (`fixed`, `linear`,
    /// `beverton_holt`, or `ricker`) scaled by the equilibrium survival rate.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `buffers`: Device pointers for the ecology columns and output.
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    /// - `new_adult_age`: First adult age class.
    ///
    /// ## Returns
    /// `Ok(())` once the launch is enqueued.
    ///
    /// ## Errors
    /// Returns a description when dimensions are out of range (including
    /// `n_ages > 64`) or the driver rejects the launch.
    pub fn density_scaling(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut DensityBuffers<'_>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
    ) -> Result<(), String> {
        if n_batch == 0 || n_ages == 0 || n_ztypes == 0 {
            return Err("density_scaling requires non-zero dimensions".to_owned());
        }
        if n_ages > MAX_AGES {
            return Err(format!(
                "density_scaling supports at most {MAX_AGES} ages, got {n_ages}"
            ));
        }
        if new_adult_age == 0 || new_adult_age > n_ages {
            return Err(format!(
                "density_scaling needs 0 < new_adult_age <= n_ages, got {new_adult_age}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let config = LaunchConfig {
            grid_dim: ((n_batch as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.density_scaling);
        launch.arg(buffers.ind);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&new_adult_i);
        launch.arg(buffers.survival_rates);
        launch.arg(buffers.reproduction_rates);
        launch.arg(buffers.fertility);
        launch.arg(buffers.competition_weights);
        launch.arg(buffers.equilibrium_declared);
        launch.arg(buffers.equilibrium_distribution);
        launch.arg(buffers.external_expected_eggs);
        launch.arg(buffers.carrying_capacity);
        launch.arg(buffers.eggs_per_female);
        launch.arg(buffers.sex_ratio);
        launch.arg(buffers.growth_mode);
        launch.arg(buffers.low_density_growth_rate);
        launch.arg(&mut *buffers.scaling_out);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("density_scaling launch failed: {err}"))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/kernels.rs"]
mod tests;
