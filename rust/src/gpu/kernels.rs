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

/// Largest `n_ztypes` the reproduction kernel supports (fixed local arrays).
const MAX_Z: usize = 32;

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

/// CUDA C for the deterministic survival application.
///
/// Mirrors `kernels::age_structured::recruit_juveniles` (deterministic branch)
/// followed by `apply_survival_deterministic`: age-0 counts are rescaled by
/// `desired/total`, then every individual and sperm category is multiplied by
/// the combined age × viability rate. All kernels are element-wise in place.
const SURVIVAL_SOURCE: &str = r#"
extern "C" __global__ void recruit_factor(
    const float* ind,
    int n_batch,
    int n_ages,
    int n_ztypes,
    const float* scaling,
    float* factor)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    float female_sum = 0.0f;
    float male_sum = 0.0f;
    for (int z = 0; z < Z; ++z) {
        female_sum += ind[((0 * A + 0) * Z + z) * n_batch + b];
        male_sum += ind[((1 * A + 0) * Z + z) * n_batch + b];
    }
    float total = female_sum + male_sum;
    float desired = total * scaling[b];
    factor[b] = (total <= 0.0f || desired <= 0.0f) ? 0.0f : (desired / total);
}

extern "C" __global__ void survival_scale_ind(
    float* ind,
    unsigned long long total,
    const float* factor,
    const float* survival_rates,
    const float* viability,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int new_adult_age)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    int b = (int)(i % (unsigned long long)n_batch);
    unsigned long long t = i / (unsigned long long)n_batch;
    int z = (int)(t % (unsigned long long)n_ztypes);
    t /= (unsigned long long)n_ztypes;
    int age = (int)(t % (unsigned long long)n_ages);
    int sex = (int)(t / (unsigned long long)n_ages);
    float age_surv = survival_rates[b * 2 * n_ages + sex * n_ages + age];
    float viab = (age == new_adult_age - 1)
        ? viability[(b * 2 * n_ages + sex * n_ages + age) * n_ztypes + z]
        : 1.0f;
    float value = ind[i];
    if (age == 0) {
        value *= factor[b];
    }
    value *= age_surv * viab;
    ind[i] = value;
}

extern "C" __global__ void survival_scale_sperm(
    float* sperm,
    unsigned long long total,
    const float* survival_rates,
    const float* viability,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int new_adult_age)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    int b = (int)(i % (unsigned long long)n_batch);
    unsigned long long t = i / (unsigned long long)n_batch;
    int zm = (int)(t % (unsigned long long)n_ztypes);
    t /= (unsigned long long)n_ztypes;
    int zf = (int)(t % (unsigned long long)n_ztypes);
    int age = (int)(t / (unsigned long long)n_ztypes);
    float age_surv = survival_rates[b * 2 * n_ages + age];
    float viab = (age == new_adult_age - 1)
        ? viability[(b * 2 * n_ages + age) * n_ztypes + zf]
        : 1.0f;
    sperm[i] = sperm[i] * age_surv * viab;
}
"#;

/// CUDA C for the deterministic reproduction stage.
///
/// One thread per batch element reproduces `kernels::age_structured::reproduction`
/// statement by statement for `stochastic == false`: effective adult males,
/// the female × male mating matrix, sperm displacement, deterministic
/// fertilization from the offspring tensor, and zygote-viability scaling of
/// the newborn age class. The batch element owns its own sperm slice, so the
/// one-thread-per-batch decomposition has no cross-element races.
const REPRODUCTION_SOURCE: &str = r#"
#define MAX_Z 32

extern "C" __global__ void reproduction(
    float* ind,
    float* sperm,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int new_adult_age,
    int has_sex_chromosomes,
    const float* mating_rates,
    const float* sperm_displacement_rate,
    const float* reproduction_rates,
    const float* fertility,
    const float* eggs_per_female,
    const float* sex_ratio,
    const int* female_only,
    const int* male_only,
    const float* fecundity,
    const float* sexual_selection,
    const float* offspring,
    const float* zygote_viability,
    const float* female_compat,
    const float* male_compat)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;

    float effective[MAX_Z];
    for (int z = 0; z < Z; ++z) {
        effective[z] = 0.0f;
    }
    for (int age = new_adult_age; age < A; ++age) {
        float mr = mating_rates[b * 2 * A + A + age];
        for (int z = 0; z < Z; ++z) {
            effective[z] += ind[((1 * A + age) * Z + z) * n_batch + b] * mr;
        }
    }
    float eff_sum = 0.0f;
    for (int z = 0; z < Z; ++z) {
        eff_sum += effective[z];
    }
    if (eff_sum == 0.0f) {
        return;
    }

    float mating_prob[MAX_Z * MAX_Z];
    const float* sel = sexual_selection + b * Z * Z;
    for (int gf = 0; gf < Z; ++gf) {
        float row_sum = 0.0f;
        for (int gm = 0; gm < Z; ++gm) {
            float v = sel[gf * Z + gm] * effective[gm];
            mating_prob[gf * Z + gm] = v;
            row_sum += v;
        }
        if (isfinite(row_sum) && row_sum > 1e-10f) {
            for (int gm = 0; gm < Z; ++gm) {
                mating_prob[gf * Z + gm] /= row_sum;
            }
        } else {
            for (int gm = 0; gm < Z; ++gm) {
                mating_prob[gf * Z + gm] = 0.0f;
            }
        }
    }

    float p_displace = sperm_displacement_rate[b];
    p_displace = (p_displace <= 0.0f) ? 0.0f : ((p_displace >= 1.0f) ? 1.0f : p_displace);
    for (int age = new_adult_age; age < A; ++age) {
        float pm = mating_rates[b * 2 * A + age];
        pm = (pm <= 0.0f) ? 0.0f : ((pm >= 1.0f) ? 1.0f : pm);
        for (int gf = 0; gf < Z; ++gf) {
            float n_female = ind[((0 * A + age) * Z + gf) * n_batch + b];
            float mated = 0.0f;
            for (int gm = 0; gm < Z; ++gm) {
                mated += sperm[((age * Z + gf) * Z + gm) * n_batch + b];
            }
            float virgins = n_female - mated;
            if (virgins < 0.0f) {
                virgins = 0.0f;
            }
            float n_mating_virgins = virgins * pm;
            float p_remating = p_displace * pm;
            float removed = mated * p_remating;
            if (removed > 1e-10f && mated > 1e-10f) {
                float frac = removed / mated;
                if (frac > 1.0f) {
                    frac = 1.0f;
                }
                for (int gm = 0; gm < Z; ++gm) {
                    int idx = ((age * Z + gf) * Z + gm) * n_batch + b;
                    sperm[idx] -= sperm[idx] * frac;
                }
            }
            float n_new = n_mating_virgins + removed;
            if (n_new > 1e-10f) {
                for (int gm = 0; gm < Z; ++gm) {
                    int idx = ((age * Z + gf) * Z + gm) * n_batch + b;
                    sperm[idx] += n_new * mating_prob[gf * Z + gm];
                }
            }
        }
    }

    float offspring_acc[MAX_Z];
    for (int z = 0; z < Z; ++z) {
        offspring_acc[z] = 0.0f;
    }
    // NaN or negative eggs-per-female -> 0, matching the host `.max(0.0)`.
    float epf = eggs_per_female[b];
    if (!(epf > 0.0f)) {
        epf = 0.0f;
    }
    const float* ff = fecundity + b * 2 * Z;
    for (int age = new_adult_age; age < A; ++age) {
        float pr = reproduction_rates[b * A + age];
        pr = (pr <= 0.0f) ? 0.0f : ((pr >= 1.0f) ? 1.0f : pr);
        float ft = fertility[b * A + age];
        ft = (ft <= 0.0f) ? 0.0f : ((ft >= 1.0f) ? 1.0f : ft);
        for (int gf = 0; gf < Z; ++gf) {
            for (int gm = 0; gm < Z; ++gm) {
                float n_pairs = sperm[((age * Z + gf) * Z + gm) * n_batch + b];
                if (n_pairs <= 0.0f) {
                    continue;
                }
                float eggs_per_pair = epf * ff[gf] * ff[Z + gm] * ft;
                float n_total = n_pairs * pr * eggs_per_pair;
                if (n_total <= 1e-10f) {
                    continue;
                }
                const float* off = offspring + b * Z * Z * Z + (gf * Z + gm) * Z;
                for (int go = 0; go < Z; ++go) {
                    offspring_acc[go] += n_total * off[go];
                }
            }
        }
    }
    // No early return on "no recruits": the host `reproduction` always
    // overwrites age 0 with `n_female`/`n_male` (zero when `fertilize`
    // produced nothing), so the device must clear age 0 here too.

    float sr = sex_ratio[b];
    sr = (sr <= 0.0f) ? 0.0f : ((sr >= 1.0f) ? 1.0f : sr);
    const float* fcompat = female_compat + b * Z;
    const float* mcompat = male_compat + b * Z;
    const float* zyg = zygote_viability + b * 2 * Z;
    for (int go = 0; go < Z; ++go) {
        float n_g = offspring_acc[go];
        float n_f = 0.0f;
        float n_m = 0.0f;
        if (n_g > 1e-10f) {
            if (has_sex_chromosomes && female_only[go]) {
                n_f = n_g;
            } else if (has_sex_chromosomes && male_only[go]) {
                n_m = n_g;
            } else {
                float p_f;
                if (has_sex_chromosomes) {
                    float denom = fcompat[go] + mcompat[go];
                    p_f = (denom > 1e-10f) ? (fcompat[go] / denom) : 0.5f;
                    p_f = (p_f <= 0.0f) ? 0.0f : ((p_f >= 1.0f) ? 1.0f : p_f);
                } else {
                    p_f = sr;
                }
                n_f = n_g * p_f;
                n_m = n_g - n_f;
            }
        }
        ind[((0 * A + 0) * Z + go) * n_batch + b] = n_f * zyg[go];
        ind[((1 * A + 0) * Z + go) * n_batch + b] = n_m * zyg[Z + go];
    }
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

/// Device arguments for [`Kernels::reproduction`].
///
/// Genetics tables are uploaded per batch element (the executor expands the
/// variant bank and `deme_variants` on the host), so the kernel never indexes
/// a variant id.
pub struct ReproductionBuffers<'a> {
    /// `(B, 2, A)` mating rates.
    pub mating_rates: &'a CudaSlice<f32>,
    /// `(B,)` sperm displacement rates.
    pub sperm_displacement_rate: &'a CudaSlice<f32>,
    /// `(B, A)` reproduction participation.
    pub reproduction_rates: &'a CudaSlice<f32>,
    /// `(B, A)` relative fertility.
    pub fertility: &'a CudaSlice<f32>,
    /// `(B,)` eggs per female.
    pub eggs_per_female: &'a CudaSlice<f32>,
    /// `(B,)` sex ratios.
    pub sex_ratio: &'a CudaSlice<f32>,
    /// `(Z,)` female-only-by-sex-chromosome flags, shared across demes.
    pub female_only: &'a CudaSlice<i32>,
    /// `(Z,)` male-only-by-sex-chromosome flags, shared across demes.
    pub male_only: &'a CudaSlice<i32>,
    /// `(B, 2, Z)` fecundity fitness.
    pub fecundity: &'a CudaSlice<f32>,
    /// `(B, Z, Z)` sexual selection fitness.
    pub sexual_selection: &'a CudaSlice<f32>,
    /// `(B, Z, Z, Z)` offspring tensor.
    pub offspring: &'a CudaSlice<f32>,
    /// `(B, 2, Z)` zygote viability.
    pub zygote_viability: &'a CudaSlice<f32>,
    /// `(B, Z)` female sex-chromosome compatibility.
    pub female_compat: &'a CudaSlice<f32>,
    /// `(B, Z)` male sex-chromosome compatibility.
    pub male_compat: &'a CudaSlice<f32>,
}

/// The compiled kernel set, loaded once per executor.
pub struct Kernels {
    /// `age_shift(const float*, float*, u64, int, u64)`.
    age_shift: CudaFunction,
    /// `density_scaling(...)`.
    density_scaling: CudaFunction,
    /// `recruit_factor(...)`.
    recruit_factor: CudaFunction,
    /// `survival_scale_ind(...)`.
    survival_scale_ind: CudaFunction,
    /// `survival_scale_sperm(...)`.
    survival_scale_sperm: CudaFunction,
    /// `reproduction(...)`.
    reproduction: CudaFunction,
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
        let source =
            format!("{AGING_SOURCE}\n{DENSITY_SOURCE}\n{SURVIVAL_SOURCE}\n{REPRODUCTION_SOURCE}");
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
        let recruit_factor = module
            .load_function("recruit_factor")
            .map_err(|err| format!("loading kernel `recruit_factor` failed: {err}"))?;
        let survival_scale_ind = module
            .load_function("survival_scale_ind")
            .map_err(|err| format!("loading kernel `survival_scale_ind` failed: {err}"))?;
        let survival_scale_sperm = module
            .load_function("survival_scale_sperm")
            .map_err(|err| format!("loading kernel `survival_scale_sperm` failed: {err}"))?;
        let reproduction = module
            .load_function("reproduction")
            .map_err(|err| format!("loading kernel `reproduction` failed: {err}"))?;
        Ok(Self {
            age_shift,
            density_scaling,
            recruit_factor,
            survival_scale_ind,
            survival_scale_sperm,
            reproduction,
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

    /// Compute the deterministic age-0 recruitment factor for every batch
    /// element.
    ///
    /// `factor = desired / total` with `desired = total · scaling` and `total`
    /// the grouped age-0 count; `0` when either is non-positive. This is the
    /// deterministic branch of `recruit_juveniles`.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `ind`: Batch-minor individual counts.
    /// - `scaling`: Per-batch density scaling factors from
    ///   [`Kernels::density_scaling`].
    /// - `factor`: Per-batch output.
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn recruit_factor(
        &self,
        stream: &Arc<CudaStream>,
        ind: &CudaSlice<f32>,
        scaling: &CudaSlice<f32>,
        factor: &mut CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let config = LaunchConfig {
            grid_dim: ((n_batch as u32).div_ceil(128), 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.recruit_factor);
        launch.arg(ind);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(scaling);
        launch.arg(&mut *factor);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("recruit_factor launch failed: {err}"))
    }

    /// Apply deterministic survival to the individual plane in place.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `ind`: Batch-minor individual counts, mutated in place.
    /// - `factor`: Per-batch age-0 recruitment factors.
    /// - `survival_rates`: `(B, 2, A)` age survival rates.
    /// - `viability`: `(B, 2, A, Z)` viability fitness per batch.
    /// - `n_batch`, `n_ages`, `n_ztypes`, `new_adult_age`: Model dimensions.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn survival_scale_ind(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        factor: &CudaSlice<f32>,
        survival_rates: &CudaSlice<f32>,
        viability: &CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
    ) -> Result<(), String> {
        let total = ind.len() as u64;
        if total == 0 {
            return Ok(());
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let config = LaunchConfig::for_num_elems(total as u32);
        let mut launch = stream.launch_builder(&self.survival_scale_ind);
        launch.arg(&mut *ind);
        launch.arg(&total);
        launch.arg(factor);
        launch.arg(survival_rates);
        launch.arg(viability);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&new_adult_i);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("survival_scale_ind launch failed: {err}"))
    }

    /// Apply deterministic survival to the sperm plane in place.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `sperm`: Batch-minor stored sperm, mutated in place.
    /// - `survival_rates`: `(B, 2, A)` age survival rates (female row used).
    /// - `viability`: `(B, 2, A, Z)` viability fitness per batch.
    /// - `n_batch`, `n_ages`, `n_ztypes`, `new_adult_age`: Model dimensions.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn survival_scale_sperm(
        &self,
        stream: &Arc<CudaStream>,
        sperm: &mut CudaSlice<f32>,
        survival_rates: &CudaSlice<f32>,
        viability: &CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
    ) -> Result<(), String> {
        let total = sperm.len() as u64;
        if total == 0 {
            return Ok(());
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let config = LaunchConfig::for_num_elems(total as u32);
        let mut launch = stream.launch_builder(&self.survival_scale_sperm);
        launch.arg(&mut *sperm);
        launch.arg(&total);
        launch.arg(survival_rates);
        launch.arg(viability);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&new_adult_i);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("survival_scale_sperm launch failed: {err}"))
    }

    /// Run the deterministic reproduction stage in place.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `ind`, `sperm`: Batch-minor state, mutated in place.
    /// - `buffers`: Per-batch ecology and genetics tables.
    /// - `n_batch`, `n_ages`, `n_ztypes`, `new_adult_age`: Model dimensions.
    /// - `has_sex_chromosomes`: Whether sex is assigned by chromosome masks.
    ///
    /// ## Errors
    /// Returns a description when dimensions are out of range (including
    /// `n_ztypes > 32`) or the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn reproduction(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        sperm: &mut CudaSlice<f32>,
        buffers: &ReproductionBuffers<'_>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
        has_sex_chromosomes: bool,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "reproduction supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let sex_chrom_i = i32::from(has_sex_chromosomes);
        let config = LaunchConfig {
            grid_dim: ((n_batch as u32).div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.reproduction);
        launch.arg(&mut *ind);
        launch.arg(&mut *sperm);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&new_adult_i);
        launch.arg(&sex_chrom_i);
        launch.arg(buffers.mating_rates);
        launch.arg(buffers.sperm_displacement_rate);
        launch.arg(buffers.reproduction_rates);
        launch.arg(buffers.fertility);
        launch.arg(buffers.eggs_per_female);
        launch.arg(buffers.sex_ratio);
        launch.arg(buffers.female_only);
        launch.arg(buffers.male_only);
        launch.arg(buffers.fecundity);
        launch.arg(buffers.sexual_selection);
        launch.arg(buffers.offspring);
        launch.arg(buffers.zygote_viability);
        launch.arg(buffers.female_compat);
        launch.arg(buffers.male_compat);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("reproduction launch failed: {err}"))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/kernels.rs"]
mod tests;
