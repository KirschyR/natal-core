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

/// Largest CSR row (destination count) the stochastic migration scratch
/// supports; the prepare kernel uses fixed `MAX_Z`-sized row arrays.
pub const MAX_CSR_ROW: usize = MAX_Z;

/// Bounded 1-D launch for the stochastic sampling kernels.
///
/// The continuous branches inline gamma-based samplers, so their unrolled
/// local arrays raise register use above what a 1024-thread block can schedule
/// ("too many resources requested for launch"). A 64-thread block keeps the
/// register budget valid; these kernels are memory-bound, not occupancy-bound.
fn stochastic_grid(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(64), 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    }
}

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
    int discrete_actual,
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
        if (discrete_actual != 0) {
            // Discrete regulation compares the total age-0 count (both sexes)
            // rather than the age-structured competition-weighted juveniles.
            float female_sum = 0.0f;
            float male_sum = 0.0f;
            for (int z = 0; z < Z; ++z) {
                female_sum += ind[((0 * A + 0) * Z + z) * n_batch + b];
                male_sum += ind[((1 * A + 0) * Z + z) * n_batch + b];
            }
            actual = female_sum + male_sum;
        } else {
            for (int age = 0; age < new_adult_age; ++age) {
                float female_sum = 0.0f;
                float male_sum = 0.0f;
                for (int z = 0; z < Z; ++z) {
                    female_sum += ind[((0 * A + age) * Z + z) * n_batch + b];
                    male_sum += ind[((1 * A + age) * Z + z) * n_batch + b];
                }
                actual += (female_sum + male_sum) * compw[age];
            }
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

/// CUDA C for deterministic CSR migration.
///
/// Migration couples batch elements (demes), so it gathers per destination
/// from a host-built reverse CSR: each output cell is written by exactly one
/// thread that sums its incoming sources in a fixed order. That keeps the
/// device result deterministic (no atomics) and mirrors
/// `kernels::spatial::migrate_csr_deterministic`; the source's own "stay" mass
/// is the `self` term.
///
/// Three kernels split the work: females (plus the sperm mass that counts as
/// individuals), stored sperm, and males. All read the pre-migration planes
/// and write fresh output planes.
const MIGRATION_SOURCE: &str = r#"
#define MIG_EPS 1e-9f

extern "C" __global__ void migration_female(
    const float* ind_in,
    const float* sperm_in,
    float* ind_out,
    const float* rate,
    const int* indptr,
    const int* rev_indptr,
    const int* rev_src,
    const float* rev_weight,
    const float* row_sum_w,
    int stay_after,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int fz = idx % Z;
    int t = idx / Z;
    int age = t % A;
    int dst = t / A;

    float fr_dst = rate[dst * 2 * A + age];
    float stored = 0.0f;
    for (int mz = 0; mz < Z; ++mz) {
        stored += sperm_in[((age * Z + fz) * Z + mz) * n_batch + dst];
    }
    float female_total = ind_in[((0 * A + age) * Z + fz) * n_batch + dst];
    float virgin = female_total - stored;
    if (virgin < 0.0f && fabsf(virgin) < MIG_EPS) {
        virgin = 0.0f;
    }
    float outbound = virgin * fr_dst;
    int row_empty = (indptr[dst + 1] == indptr[dst]);
    float self;
    if (stay_after) {
        self = virgin - outbound * row_sum_w[dst];
    } else if (row_empty) {
        self = virgin;
    } else {
        self = virgin - outbound;
    }
    // Stored sperm that stays also counts as individuals in the female plane.
    for (int mz = 0; mz < Z; ++mz) {
        float value = sperm_in[((age * Z + fz) * Z + mz) * n_batch + dst];
        float out_sperm = value * fr_dst;
        if (stay_after) {
            self += value - out_sperm * row_sum_w[dst];
        } else if (row_empty) {
            self += value;
        } else {
            self += value - out_sperm;
        }
    }
    float total = self;
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        int src = rev_src[e];
        float w = rev_weight[e];
        float fr_src = rate[src * 2 * A + age];
        float s_stored = 0.0f;
        for (int mz = 0; mz < Z; ++mz) {
            s_stored += sperm_in[((age * Z + fz) * Z + mz) * n_batch + src];
        }
        float s_total = ind_in[((0 * A + age) * Z + fz) * n_batch + src];
        float s_virgin = s_total - s_stored;
        if (s_virgin < 0.0f && fabsf(s_virgin) < MIG_EPS) {
            s_virgin = 0.0f;
        }
        total += (s_virgin * fr_src) * w;
        for (int mz = 0; mz < Z; ++mz) {
            float sp = sperm_in[((age * Z + fz) * Z + mz) * n_batch + src];
            total += (sp * fr_src) * w;
        }
    }
    ind_out[((0 * A + age) * Z + fz) * n_batch + dst] = total;
}

extern "C" __global__ void migration_sperm(
    const float* sperm_in,
    float* sperm_out,
    const float* rate,
    const int* indptr,
    const int* rev_indptr,
    const int* rev_src,
    const float* rev_weight,
    const float* row_sum_w,
    int stay_after,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes * n_ztypes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int mz = idx % Z;
    int t = idx / Z;
    int fz = t % Z;
    t /= Z;
    int age = t % A;
    int dst = t / A;

    float fr_dst = rate[dst * 2 * A + age];
    float value = sperm_in[((age * Z + fz) * Z + mz) * n_batch + dst];
    float outbound = value * fr_dst;
    int row_empty = (indptr[dst + 1] == indptr[dst]);
    float total;
    if (stay_after) {
        total = value - outbound * row_sum_w[dst];
    } else if (row_empty) {
        total = value;
    } else {
        total = value - outbound;
    }
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        int src = rev_src[e];
        float w = rev_weight[e];
        float fr_src = rate[src * 2 * A + age];
        float moved = sperm_in[((age * Z + fz) * Z + mz) * n_batch + src];
        total += (moved * fr_src) * w;
    }
    sperm_out[((age * Z + fz) * Z + mz) * n_batch + dst] = total;
}

extern "C" __global__ void migration_male(
    const float* ind_in,
    float* ind_out,
    const float* rate,
    const int* indptr,
    const int* rev_indptr,
    const int* rev_src,
    const float* rev_weight,
    const float* row_sum_w,
    int stay_after,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int z = idx % Z;
    int t = idx / Z;
    int age = t % A;
    int dst = t / A;

    float mr_dst = rate[dst * 2 * A + A + age];
    float value = ind_in[((1 * A + age) * Z + z) * n_batch + dst];
    float outbound = value * mr_dst;
    int row_empty = (indptr[dst + 1] == indptr[dst]);
    float total;
    if (stay_after) {
        total = value - outbound * row_sum_w[dst];
    } else if (row_empty) {
        total = value;
    } else {
        total = value - outbound;
    }
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        int src = rev_src[e];
        float w = rev_weight[e];
        float mr_src = rate[src * 2 * A + A + age];
        float moved = ind_in[((1 * A + age) * Z + z) * n_batch + src];
        total += (moved * mr_src) * w;
    }
    ind_out[((1 * A + age) * Z + z) * n_batch + dst] = total;
}
"#;

/// CUDA C for the counter-based device RNG (§D3).
///
/// Philox4x32-10 keyed on `(seed, draw_site)` and counted by a 64-bit index:
/// every draw is a pure function of `(key, counter, site)`, so results are
/// reproducible run to run and independent of thread scheduling. `fill_uniform`
/// is the reference launcher the sampling kernels are built on.
const RNG_SOURCE: &str = r#"
__device__ __forceinline__ unsigned int philox_mulhi(unsigned int a, unsigned int b) {
    return (unsigned int)(((unsigned long long)a * (unsigned long long)b) >> 32);
}

__device__ __forceinline__ void philox4x32_10(
    unsigned int c0, unsigned int c1, unsigned int c2, unsigned int c3,
    unsigned int key0, unsigned int key1,
    unsigned int* o0, unsigned int* o1, unsigned int* o2, unsigned int* o3)
{
    const unsigned int M0 = 0xD2511F53u;
    const unsigned int M1 = 0xCD9E8D57u;
    const unsigned int W0 = 0x9E3779B9u;
    const unsigned int W1 = 0xBB67AE85u;
    for (int i = 0; i < 10; ++i) {
        unsigned int h0 = philox_mulhi(M0, c0);
        unsigned int l0 = M0 * c0;
        unsigned int h1 = philox_mulhi(M1, c2);
        unsigned int l1 = M1 * c2;
        unsigned int n0 = h1 ^ c1 ^ key0;
        unsigned int n1 = l1;
        unsigned int n2 = h0 ^ c3 ^ key1;
        unsigned int n3 = l0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        key0 += W0;
        key1 += W1;
    }
    *o0 = c0; *o1 = c1; *o2 = c2; *o3 = c3;
}

__device__ __forceinline__ float uniform_from_u32(unsigned int bits) {
    // 24-bit mantissa keeps every value exact in f32 and inside [0, 1).
    return (float)(bits >> 8) * (1.0f / 16777216.0f);
}

extern "C" __global__ void fill_uniform(
    float* out,
    unsigned long long n,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    unsigned int o0, o1, o2, o3;
    philox4x32_10(
        (unsigned int)(i & 0xFFFFFFFFu),
        (unsigned int)(i >> 32),
        site,
        0u,
        key0,
        key1,
        &o0, &o1, &o2, &o3);
    out[i] = uniform_from_u32(o0);
}
"#;

/// CUDA C for the device sampling library (§P4).
///
/// A per-thread `RngState` drives every sampler; each draw is a pure function
/// of `(key, counter, site)`, so a thread's stream is reproducible and has no
/// cross-thread dependency. Binomial and Poisson use exact methods in the
/// small-parameter regime (geometric skipping / Knuth product) and a normal
/// approximation with a continuity correction once the distribution is close
/// to symmetric; gamma uses Marsaglia-Tsang. Multinomial lifts the sequential
/// conditional-binomial loop to the outer axis all threads advance together
/// (§4.2 plan C).
const SAMPLING_SOURCE: &str = r#"
struct RngState {
    unsigned long long counter;
    unsigned int key0;
    unsigned int key1;
    unsigned int site;
    unsigned int buffer[4];
    int index;
};

__device__ __forceinline__ void rng_init(
    RngState* state, unsigned long long cell, unsigned int key0, unsigned int key1, unsigned int site)
{
    // High counter word identifies the cell, low word counts that cell's
    // draws, so no two cells ever share a Philox counter (which would make
    // their streams overlap).
    state->counter = (cell & 0xFFFFFFFFull) << 32;
    state->key0 = key0;
    state->key1 = key1;
    state->site = site;
    state->index = 4;
}

__device__ __forceinline__ unsigned int rng_next(RngState* state) {
    if (state->index >= 4) {
        unsigned int o0, o1, o2, o3;
        philox4x32_10(
            (unsigned int)(state->counter & 0xFFFFFFFFu),
            (unsigned int)(state->counter >> 32),
            state->site,
            0u,
            state->key0,
            state->key1,
            &o0, &o1, &o2, &o3);
        state->buffer[0] = o0;
        state->buffer[1] = o1;
        state->buffer[2] = o2;
        state->buffer[3] = o3;
        state->counter += 1ULL;
        state->index = 0;
    }
    return state->buffer[state->index++];
}

__device__ __forceinline__ float rng_uniform(RngState* state) {
    return uniform_from_u32(rng_next(state));
}

__device__ __forceinline__ float rng_normal(RngState* state) {
    float u1 = rng_uniform(state);
    if (u1 < 1e-7f) {
        u1 = 1e-7f;
    }
    float u2 = rng_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

// Exact binomial by geometric skipping below the threshold; normal
// approximation (mean-preserving, continuity-corrected) above it.
__device__ __forceinline__ float sample_binomial(RngState* state, float n, float p) {
    if (n <= 0.0f || p <= 0.0f) {
        return 0.0f;
    }
    if (p >= 1.0f) {
        return n;
    }
    int reflect = p > 0.5f;
    float pp = reflect ? (1.0f - p) : p;
    float mean = n * pp;
    float result;
    if (mean <= 512.0f) {
        float log1mp = logf(1.0f - pp);
        float successes = 0.0f;
        float remaining = n;
        while (remaining > 0.0f) {
            float u = rng_uniform(state);
            if (u <= 0.0f) {
                u = 1e-7f;
            }
            float failures = floorf(logf(u) / log1mp);
            if (failures >= remaining) {
                break;
            }
            successes += 1.0f;
            remaining -= (failures + 1.0f);
        }
        result = successes;
    } else {
        float sd = sqrtf(mean * (1.0f - pp));
        float z = rng_normal(state);
        result = floorf(mean + sd * z + 0.5f);
        if (result < 0.0f) {
            result = 0.0f;
        }
        if (result > n) {
            result = n;
        }
    }
    return reflect ? (n - result) : result;
}

// Exact Poisson (Knuth product) for small mean; normal approximation above.
__device__ __forceinline__ float sample_poisson(RngState* state, float lambda) {
    if (lambda <= 0.0f) {
        return 0.0f;
    }
    if (lambda < 10.0f) {
        // Knuth product method: exact for small mean.
        float limit = expf(-lambda);
        float count = 0.0f;
        float product = 1.0f;
        do {
            count += 1.0f;
            product *= rng_uniform(state);
        } while (product > limit);
        return count - 1.0f;
    }
    // PTRS (Hörmann transformed rejection): exact for lambda >= 10, O(1)
    // expected work. Unlike a normal approximation it preserves the Poisson
    // skew/kurtosis, which the CPU reference and the statistical gate require.
    float b = 0.931f + 2.53f * sqrtf(lambda);
    float a = -0.059f + 0.02483f * b;
    float inv_alpha = 1.1239f + 1.1328f / (b - 3.4f);
    float v_r = 0.9277f - 3.6224f / (b - 2.0f);
    for (;;) {
        float u = rng_uniform(state) - 0.5f;
        float v = rng_uniform(state);
        float us = 0.5f - fabsf(u);
        float k = floorf((2.0f * a / us + b) * u + lambda + 0.43f);
        if (us >= 0.07f && v <= v_r) {
            return k;
        }
        if (k < 0.0f || (us < 0.013f && v > us)) {
            continue;
        }
        if (logf(v * inv_alpha / (a / (us * us) + b))
            <= k * logf(lambda) - lambda - lgammaf(k + 1.0f))
        {
            return k;
        }
    }
}

// Marsaglia-Tsang gamma sampler. The shape < 1 boost is applied iteratively
// (rather than by recursive self-call) so the large reproduction kernel does
// not overflow the device call stack.
__device__ float sample_gamma(RngState* state, float shape, float scale) {
    float boost = 1.0f;
    if (shape < 1.0f) {
        float u = rng_uniform(state);
        if (u <= 0.0f) {
            u = 1e-7f;
        }
        boost = powf(u, 1.0f / shape);
        shape = shape + 1.0f;
    }
    float d = shape - 1.0f / 3.0f;
    float c = 1.0f / sqrtf(9.0f * d);
    for (;;) {
        float x = rng_normal(state);
        float v = 1.0f + c * x;
        if (v <= 0.0f) {
            continue;
        }
        v = v * v * v;
        float u = rng_uniform(state);
        if (u < 1.0f - 0.0331f * x * x * x * x) {
            return d * v * scale * boost;
        }
        if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) {
            return d * v * scale * boost;
        }
    }
}

// Continuous binomial analogue: Beta(n-1) proportion times n.
__device__ __forceinline__ float sample_continuous_binomial(
    RngState* state, float n, float p)
{
    // Mirrors `continuous_binomial`: a Beta-distributed proportion times n,
    // with the small-n mean path matching the host guards.
    if (!isfinite(n) || !isfinite(p)) {
        return 0.0f;
    }
    if (p <= 1e-12f) {
        return 0.0f;
    }
    if (p >= 1.0f - 1e-12f) {
        return n;
    }
    if (n <= 1.0f + 1e-12f) {
        return n * p;
    }
    float concentration = n - 1.0f;
    float alpha = fmaxf(p * concentration, 1e-12f);
    float beta = fmaxf((1.0f - p) * concentration, 1e-12f);
    float numerator = sample_gamma(state, alpha, 1.0f);
    float denominator = sample_gamma(state, beta, 1.0f);
    if (numerator == 0.0f) {
        return 0.0f;
    }
    return (numerator / (numerator + denominator)) * n;
}

// Continuous Poisson analogue: a Gamma(shape=lambda, scale=1) draw.
__device__ __forceinline__ float sample_continuous_poisson(RngState* state, float lambda)
{
    if (!isfinite(lambda) || lambda <= 1e-12f) {
        return 0.0f;
    }
    return sample_gamma(state, lambda, 1.0f);
}

// Continuous multinomial: normalized Gamma draws with a drift correction.
__device__ __forceinline__ void natal_continuous_multinomial(
    RngState* state, float n, const float* p, int K, float* out)
{
    if (n <= 1.0f + 1e-7f) {
        for (int k = 0; k < K; ++k) {
            out[k] = n * p[k];
        }
        return;
    }
    float concentration = n - 1.0f;
    float sum_gamma = 0.0f;
    for (int k = 0; k < K; ++k) {
        float alpha = p[k] * concentration;
        float value = (alpha <= 1e-12f) ? 0.0f : sample_gamma(state, alpha, 1.0f);
        out[k] = value;
        sum_gamma += value;
    }
    if (sum_gamma > 1e-12f) {
        float factor = n / sum_gamma;
        for (int k = 0; k < K; ++k) {
            out[k] *= factor;
        }
    } else {
        for (int k = 0; k < K; ++k) {
            out[k] = n * p[k];
        }
    }
    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        total += out[k];
    }
    float tolerance = 1e-6f * fmaxf(n, 1.0f);
    if (total > 1e-12f && fabsf(total - n) > tolerance) {
        float correction = n / total;
        for (int k = 0; k < K; ++k) {
            out[k] *= correction;
        }
    }
}

extern "C" __global__ void sample_into(
    float* out,
    unsigned long long n,
    unsigned int kind,
    float a,
    float b,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    RngState state;
    rng_init(&state, i, key0, key1, site);
    float value;
    if (kind == 0u) {
        value = rng_uniform(&state);
    } else if (kind == 1u) {
        value = sample_binomial(&state, a, b);
    } else if (kind == 2u) {
        value = sample_poisson(&state, a);
    } else if (kind == 3u) {
        value = sample_gamma(&state, a, b);
    } else {
        value = rng_normal(&state);
    }
    out[i] = value;
}

// Multinomial: the sequential conditional-binomial loop is the outer axis all
// threads advance together; each thread owns one row.
extern "C" __global__ void multinomial_seq(
    const float* probs,
    const float* totals,
    float* counts,
    int n_rows,
    int n_categories,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    int K = n_categories;
    RngState state;
    rng_init(&state, (unsigned long long)row, key0, key1, site);
    const float* p = probs + (long long)row * K;
    float* out = counts + (long long)row * K;
    float tail = 0.0f;
    for (int k = 0; k < K; ++k) {
        tail += p[k];
    }
    float remaining = totals[row];
    for (int k = 0; k < K - 1; ++k) {
        float draw = 0.0f;
        if (remaining > 0.0f && tail > 0.0f) {
            float conditional = p[k] / tail;
            if (conditional > 1.0f) {
                conditional = 1.0f;
            }
            draw = sample_binomial(&state, remaining, conditional);
            if (draw > remaining) {
                draw = remaining;
            }
        }
        out[k] = draw;
        remaining -= draw;
        tail -= p[k];
        if (remaining < 0.0f) {
            remaining = 0.0f;
        }
        if (tail < 0.0f) {
            tail = 0.0f;
        }
    }
    out[K - 1] = remaining;
}

#define NATAL_MAX_Z 32

// f32 tolerance for the virgin-count non-negativity check. The host uses a
// 1e-10 f64 threshold; the device must scale the tolerance with magnitude,
// because a sequential f32 sum of the sperm categories drifts by roughly
// `8 * eps * (|female| + |sperm|)` at count scales up to the 2^24 integer
// limit. States inside that drift are rounding noise and clamp to zero;
// meaningfully negative states are reported through `violation`.
#define NATAL_VIRGIN_EPS_ABS 1e-9f
#define NATAL_F32_EPS 1.19209290e-7f

__device__ __forceinline__ float natal_clamp01(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    if (x >= 1.0f) {
        return 1.0f;
    }
    return x;
}

// Stochastic recruit: resample the age-0 categories into `desired` draws with
// the multinomial (discrete or continuous, matching `recruit_juveniles`). One
// thread per batch element.
extern "C" __global__ void recruit_stochastic(
    float* ind,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int continuous,
    const float* scaling,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    float combined[2 * NATAL_MAX_Z];
    float female_sum = 0.0f;
    float male_sum = 0.0f;
    int cursor = 0;
    for (int sex = 0; sex < 2; ++sex) {
        for (int z = 0; z < Z; ++z) {
            float raw = ind[((sex * A + 0) * Z + z) * n_batch + b];
            float value = continuous ? raw : roundf(raw);
            combined[cursor++] = value;
            if (sex == 0) {
                female_sum += value;
            } else {
                male_sum += value;
            }
        }
    }
    float total = female_sum + male_sum;
    float total_counts = 0.0f;
    for (int i = 0; i < 2 * Z; ++i) {
        total_counts += combined[i];
    }
    float desired = 0.0f;
    if (total > 0.0f) {
        desired = continuous ? (total * scaling[b]) : roundf(total * scaling[b]);
    }
    if (total <= 0.0f || desired <= 0.0f) {
        for (int sex = 0; sex < 2; ++sex) {
            for (int z = 0; z < Z; ++z) {
                ind[((sex * A + 0) * Z + z) * n_batch + b] = 0.0f;
            }
        }
        return;
    }
    RngState state;
    rng_init(&state, (unsigned long long)b, key0, key1, site);
    float draws[2 * NATAL_MAX_Z];
    if (continuous) {
        float probs[2 * NATAL_MAX_Z];
        for (int k = 0; k < 2 * Z; ++k) {
            probs[k] = combined[k] / total_counts;
        }
        natal_continuous_multinomial(&state, desired, probs, 2 * Z, draws);
    } else {
        float remaining = desired;
        float tail = total_counts;
        for (int k = 0; k < 2 * Z - 1; ++k) {
            float draw = 0.0f;
            if (remaining > 0.0f && tail > 0.0f) {
                float conditional = natal_clamp01(combined[k] / tail);
                draw = sample_binomial(&state, remaining, conditional);
                if (draw > remaining) {
                    draw = remaining;
                }
            }
            draws[k] = draw;
            remaining -= draw;
            tail -= combined[k];
            if (remaining < 0.0f) {
                remaining = 0.0f;
            }
            if (tail < 0.0f) {
                tail = 0.0f;
            }
        }
        draws[2 * Z - 1] = remaining;
    }
    cursor = 0;
    for (int sex = 0; sex < 2; ++sex) {
        for (int z = 0; z < Z; ++z) {
            ind[((sex * A + 0) * Z + z) * n_batch + b] = draws[cursor++];
        }
    }
}

// Stochastic survival: `sample_survival_with_sperm` (discrete or continuous).
// One thread per (batch, age, zygote-type) owns its female cell, sperm column,
// and male cell, so there are no cross-thread races.
extern "C" __global__ void survival_stochastic(
    float* ind,
    float* sperm,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int continuous,
    int new_adult_age,
    const float* survival_rates,
    const float* viability,
    int* violation,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int cells = n_batch * n_ages * n_ztypes;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int g = i % Z;
    int t = i / Z;
    int age = t % A;
    int b = t / A;
    int target = new_adult_age - 1;
    float age_f = survival_rates[b * 2 * A + age];
    float age_m = survival_rates[b * 2 * A + A + age];
    float viab_f = (age == target) ? viability[(b * 2 * A + age) * Z + g] : 1.0f;
    float viab_m = (age == target) ? viability[(b * 2 * A + A + age) * Z + g] : 1.0f;
    float p_f = natal_clamp01(age_f * viab_f);
    float p_m = natal_clamp01(age_m * viab_m);

    RngState state;
    rng_init(&state, (unsigned long long)i, key0, key1, site);

    float n_f_raw = ind[((0 * A + age) * Z + g) * n_batch + b];
    float total_sperm = 0.0f;
    for (int mz = 0; mz < Z; ++mz) {
        total_sperm += sperm[((age * Z + g) * Z + mz) * n_batch + b];
    }
    float virgins = n_f_raw - total_sperm;
    if (virgins < 0.0f) {
        float tolerance = 8.0f * (fabsf(n_f_raw) + fabsf(total_sperm)) * NATAL_F32_EPS
            + NATAL_VIRGIN_EPS_ABS;
        if (virgins < -tolerance) {
            // The host errors on a meaningfully negative virgin count; record
            // it so the executor can return an explicit failure instead of
            // silently clamping a corrupted state. Magnitude-scaled so valid
            // large states whose f32 sum drifts positive are not misreported.
            violation[0] = 1;
        }
        virgins = 0.0f;
    }
    float n_virgins = continuous ? virgins : roundf(virgins);
    float new_sperm_sum = 0.0f;
    for (int mz = 0; mz < Z; ++mz) {
        int index = ((age * Z + g) * Z + mz) * n_batch + b;
        float count = continuous ? sperm[index] : roundf(sperm[index]);
        float survived = 0.0f;
        if (count > 1e-10f) {
            survived = continuous ? sample_continuous_binomial(&state, count, p_f)
                                  : sample_binomial(&state, count, p_f);
        }
        sperm[index] = survived;
        new_sperm_sum += survived;
    }
    float survived_virgins = 0.0f;
    if (n_virgins > 1e-10f) {
        survived_virgins = continuous ? sample_continuous_binomial(&state, n_virgins, p_f)
                                      : sample_binomial(&state, n_virgins, p_f);
    }
    ind[((0 * A + age) * Z + g) * n_batch + b] = new_sperm_sum + survived_virgins;

    float n_m_raw = ind[((1 * A + age) * Z + g) * n_batch + b];
    float n_m = continuous ? n_m_raw : roundf(n_m_raw);
    float survived_m = 0.0f;
    if (n_m > 1e-10f) {
        survived_m = continuous ? sample_continuous_binomial(&state, n_m, p_m)
                                : sample_binomial(&state, n_m, p_m);
    }
    ind[((1 * A + age) * Z + g) * n_batch + b] = survived_m;
}

// Sequential conditional-binomial multinomial over `K` categories, for use by
// a single thread (the batch element is the outer axis).
__device__ __forceinline__ void natal_multinomial(
    RngState* state, float total, const float* probs, int K, float* out)
{
    float remaining = total;
    float tail = 0.0f;
    for (int k = 0; k < K; ++k) {
        tail += probs[k];
    }
    for (int k = 0; k < K - 1; ++k) {
        float draw = 0.0f;
        if (remaining > 0.0f && tail > 0.0f) {
            float conditional = natal_clamp01(probs[k] / tail);
            draw = sample_binomial(state, remaining, conditional);
            if (draw > remaining) {
                draw = remaining;
            }
        }
        out[k] = draw;
        remaining -= draw;
        tail -= probs[k];
        if (remaining < 0.0f) {
            remaining = 0.0f;
        }
        if (tail < 0.0f) {
            tail = 0.0f;
        }
    }
    out[K - 1] = remaining;
}

// Stochastic reproduction: `sample_mating` + `fertilize` + zygote viability,
// stochastic (discrete or continuous). One thread per batch element.
extern "C" __global__ void reproduction_stochastic(
    float* ind,
    float* sperm,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int continuous,
    int new_adult_age,
    int has_sex_chromosomes,
    int fixed_egg_count,
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
    const float* male_compat,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;

    float effective[NATAL_MAX_Z];
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

    float mating_prob[NATAL_MAX_Z * NATAL_MAX_Z];
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

    RngState state;
    rng_init(&state, (unsigned long long)b, key0, key1, site);
    float p_displace = natal_clamp01(sperm_displacement_rate[b]);

    for (int age = new_adult_age; age < A; ++age) {
        float p_mating = natal_clamp01(mating_rates[b * 2 * A + age]);
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
            float n_mating_virgins = continuous
                ? sample_continuous_binomial(&state, virgins, p_mating)
                : sample_binomial(&state, roundf(virgins), p_mating);
            float p_remating = p_displace * p_mating;
            float n_remating = 0.0f;
            if (mated > 1e-10f && p_remating > 1e-10f) {
                if (continuous) {
                    float removed_frac = p_remating < 1.0f ? p_remating : 1.0f;
                    for (int gm = 0; gm < Z; ++gm) {
                        int index = ((age * Z + gf) * Z + gm) * n_batch + b;
                        sperm[index] -= sperm[index] * removed_frac;
                    }
                    n_remating = mated * removed_frac;
                } else {
                    for (int gm = 0; gm < Z; ++gm) {
                        int index = ((age * Z + gf) * Z + gm) * n_batch + b;
                        float count = sperm[index];
                        if (count > 1e-10f) {
                            float removed = sample_binomial(&state, roundf(count), p_remating);
                            float left = sperm[index] - removed;
                            sperm[index] = left < 0.0f ? 0.0f : left;
                            n_remating += removed;
                        }
                    }
                }
            }
            float n_new = n_mating_virgins + n_remating;
            if (n_new > 1e-10f) {
                float row[NATAL_MAX_Z];
                float drawn[NATAL_MAX_Z];
                for (int gm = 0; gm < Z; ++gm) {
                    row[gm] = mating_prob[gf * Z + gm];
                }
                if (continuous) {
                    natal_continuous_multinomial(&state, n_new, row, Z, drawn);
                    for (int gm = 0; gm < Z; ++gm) {
                        sperm[((age * Z + gf) * Z + gm) * n_batch + b] += drawn[gm];
                    }
                } else {
                    float n_int = roundf(n_new);
                    if (n_int > 0.0f) {
                        natal_multinomial(&state, n_int, row, Z, drawn);
                        for (int gm = 0; gm < Z; ++gm) {
                            sperm[((age * Z + gf) * Z + gm) * n_batch + b] += drawn[gm];
                        }
                    }
                }
            }
        }
    }

    float offspring_acc[NATAL_MAX_Z];
    for (int z = 0; z < Z; ++z) {
        offspring_acc[z] = 0.0f;
    }
    float epf = eggs_per_female[b];
    if (!(epf > 0.0f)) {
        epf = 0.0f;
    }
    const float* ff = fecundity + b * 2 * Z;
    for (int age = new_adult_age; age < A; ++age) {
        float pr = natal_clamp01(reproduction_rates[b * A + age]);
        float ft = natal_clamp01(fertility[b * A + age]);
        for (int gf = 0; gf < Z; ++gf) {
            for (int gm = 0; gm < Z; ++gm) {
                float n_pairs = sperm[((age * Z + gf) * Z + gm) * n_batch + b];
                if (n_pairs <= 0.0f) {
                    continue;
                }
                float eggs_per_pair = epf * ff[gf] * ff[Z + gm] * ft;
                float n_pairs_eff = continuous ? n_pairs : roundf(n_pairs);
                if (n_pairs_eff <= 0.0f) {
                    continue;
                }
                float n_reproducing = n_pairs_eff;
                if (pr < 1.0f - 1e-10f) {
                    n_reproducing = continuous
                        ? sample_continuous_binomial(&state, n_pairs_eff, pr)
                        : sample_binomial(&state, n_pairs_eff, pr);
                }
                float total_lambda = n_reproducing * eggs_per_pair;
                float n_total;
                if (fixed_egg_count) {
                    n_total = continuous ? total_lambda : roundf(total_lambda);
                } else {
                    n_total = continuous ? sample_continuous_poisson(&state, total_lambda)
                                         : sample_poisson(&state, total_lambda);
                }
                if (n_total <= 1e-10f) {
                    continue;
                }
                const float* off = offspring + b * Z * Z * Z + (gf * Z + gm) * Z;
                float p_surv = 0.0f;
                for (int go = 0; go < Z; ++go) {
                    p_surv += off[go];
                }
                if (p_surv <= 1e-10f) {
                    continue;
                }
                float n_viable = n_total;
                if (p_surv < 1.0f - 1e-10f) {
                    n_viable = continuous
                        ? sample_continuous_binomial(&state, n_total, p_surv)
                        : sample_binomial(&state, roundf(n_total), p_surv);
                }
                if (n_viable <= 1e-10f) {
                    continue;
                }
                float inv = 1.0f / p_surv;
                float prob_norm[NATAL_MAX_Z];
                float drawn[NATAL_MAX_Z];
                for (int go = 0; go < Z; ++go) {
                    prob_norm[go] = off[go] * inv;
                }
                if (continuous) {
                    natal_continuous_multinomial(&state, n_viable, prob_norm, Z, drawn);
                } else {
                    natal_multinomial(&state, roundf(n_viable), prob_norm, Z, drawn);
                }
                for (int go = 0; go < Z; ++go) {
                    offspring_acc[go] += drawn[go];
                }
            }
        }
    }

    float sr = natal_clamp01(sex_ratio[b]);
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
                    p_f = (denom > 1e-10f) ? natal_clamp01(fcompat[go] / denom) : 0.5f;
                } else {
                    p_f = sr;
                }
                float n_fem = continuous
                    ? sample_continuous_binomial(&state, n_g, p_f)
                    : sample_binomial(&state, roundf(n_g), p_f);
                n_f = n_fem;
                n_m = n_g - n_fem;
            }
        }
        float fv = 0.0f;
        if (n_f > 0.0f) {
            fv = continuous
                ? sample_continuous_binomial(&state, n_f, natal_clamp01(zyg[go]))
                : sample_binomial(&state, roundf(n_f), natal_clamp01(zyg[go]));
        }
        float mv = 0.0f;
        if (n_m > 0.0f) {
            mv = continuous
                ? sample_continuous_binomial(&state, n_m, natal_clamp01(zyg[Z + go]))
                : sample_binomial(&state, roundf(n_m), natal_clamp01(zyg[Z + go]));
        }
        ind[((0 * A + 0) * Z + go) * n_batch + b] = fv;
        ind[((1 * A + 0) * Z + go) * n_batch + b] = mv;
    }
}

// Outbound sampling: mirrors `sample_outbound`. `rate >= 1` moves everything;
// otherwise a continuous or discrete binomial draw.
__device__ __forceinline__ float sample_outbound_device(
    RngState* state, float value, float rate, int continuous)
{
    if (value <= 0.0f || rate <= 0.0f) {
        return 0.0f;
    }
    if (rate >= 1.0f) {
        return value;
    }
    return continuous ? sample_continuous_binomial(state, value, rate)
                      : sample_binomial(state, roundf(value), rate);
}

// Split an outbound amount across a CSR row's normalized probabilities and
// return the total moved. The discrete branch rounds to a trial count exactly
// as the host `distribute_csr_outbound` does.
__device__ __forceinline__ float migration_split(
    RngState* state, float outbound, const float* probs, int row_len, int continuous, float* drawn)
{
    if (continuous) {
        natal_continuous_multinomial(state, outbound, probs, row_len, drawn);
    } else {
        natal_multinomial(state, roundf(outbound), probs, row_len, drawn);
    }
    float moved = 0.0f;
    for (int pos = 0; pos < row_len; ++pos) {
        moved += drawn[pos];
    }
    return moved;
}

// Pass 1 of stochastic CSR migration: one thread per source deme computes the
// outbound draws and their multinomial split across the source's CSR row,
// storing per-entry forward amounts. Writing the source's own "stay" mass
// here keeps pass 2 a pure, order-fixed gather (no atomics, reproducible).
extern "C" __global__ void migration_stochastic_prepare(
    const float* ind_in,
    const float* sperm_in,
    float* ind_out,
    float* sperm_out,
    const float* rate,
    const int* indptr,
    const int* dest,
    const float* weights,
    float* fwd_f,
    float* fwd_s,
    float* fwd_m,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int continuous,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int src = blockIdx.x * blockDim.x + threadIdx.x;
    if (src >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int row_start = indptr[src];
    int row_end = indptr[src + 1];
    int row_len = row_end - row_start;
    float total_w = 0.0f;
    for (int e = row_start; e < row_end; ++e) {
        total_w += weights[e];
    }

    RngState state;
    rng_init(&state, (unsigned long long)src, key0, key1, site);

    float probs[NATAL_MAX_Z];
    float drawn[NATAL_MAX_Z];

    for (int age = 0; age < A; ++age) {
        float female_rate = rate[src * 2 * A + age];
        for (int gf = 0; gf < Z; ++gf) {
            float stored = 0.0f;
            for (int gm = 0; gm < Z; ++gm) {
                stored += sperm_in[((age * Z + gf) * Z + gm) * n_batch + src];
            }
            float female_total = ind_in[((0 * A + age) * Z + gf) * n_batch + src];
            float virgin = female_total - stored;
            if (virgin < 0.0f && fabsf(virgin) < MIG_EPS) {
                virgin = 0.0f;
            }
            float outbound = sample_outbound_device(&state, virgin, female_rate, continuous);
            float moved_total = 0.0f;
            if (outbound > 0.0f && row_len > 0 && total_w > 0.0f) {
                for (int pos = 0; pos < row_len; ++pos) {
                    probs[pos] = weights[row_start + pos] / total_w;
                }
                moved_total =
                    migration_split(&state, outbound, probs, row_len, continuous, drawn);
                for (int pos = 0; pos < row_len; ++pos) {
                    int e = row_start + pos;
                    fwd_f[(long long)e * A * Z + age * Z + gf] = drawn[pos];
                }
            } else {
                for (int pos = 0; pos < row_len; ++pos) {
                    fwd_f[(long long)(row_start + pos) * A * Z + age * Z + gf] = 0.0f;
                }
            }
            float female_stay = virgin - moved_total;
            int female_index = ((0 * A + age) * Z + gf) * n_batch + src;
            ind_out[female_index] = female_stay;
            for (int gm = 0; gm < Z; ++gm) {
                float value = sperm_in[((age * Z + gf) * Z + gm) * n_batch + src];
                float outbound_sperm = sample_outbound_device(&state, value, female_rate, continuous);
                float moved_sperm = 0.0f;
                if (outbound_sperm > 0.0f && row_len > 0 && total_w > 0.0f) {
                    for (int pos = 0; pos < row_len; ++pos) {
                        probs[pos] = weights[row_start + pos] / total_w;
                    }
                    moved_sperm =
                        migration_split(&state, outbound_sperm, probs, row_len, continuous, drawn);
                    for (int pos = 0; pos < row_len; ++pos) {
                        int e = row_start + pos;
                        fwd_s[((long long)e * A + age) * Z * Z + gf * Z + gm] = drawn[pos];
                    }
                } else {
                    for (int pos = 0; pos < row_len; ++pos) {
                        fwd_s[((long long)(row_start + pos) * A + age) * Z * Z + gf * Z + gm] = 0.0f;
                    }
                }
                float sperm_stay = value - moved_sperm;
                sperm_out[((age * Z + gf) * Z + gm) * n_batch + src] = sperm_stay;
                ind_out[female_index] += sperm_stay;
            }
        }
    }

    for (int age = 0; age < A; ++age) {
        float male_rate = rate[src * 2 * A + A + age];
        for (int z = 0; z < Z; ++z) {
            float value = ind_in[((1 * A + age) * Z + z) * n_batch + src];
            float outbound = sample_outbound_device(&state, value, male_rate, continuous);
            float moved_total = 0.0f;
            if (outbound > 0.0f && row_len > 0 && total_w > 0.0f) {
                for (int pos = 0; pos < row_len; ++pos) {
                    probs[pos] = weights[row_start + pos] / total_w;
                }
                moved_total =
                    migration_split(&state, outbound, probs, row_len, continuous, drawn);
                for (int pos = 0; pos < row_len; ++pos) {
                    int e = row_start + pos;
                    fwd_m[(long long)e * A * Z + age * Z + z] = drawn[pos];
                }
            } else {
                for (int pos = 0; pos < row_len; ++pos) {
                    fwd_m[(long long)(row_start + pos) * A * Z + age * Z + z] = 0.0f;
                }
            }
            ind_out[((1 * A + age) * Z + z) * n_batch + src] = value - moved_total;
        }
    }
}

// Pass 2 gather kernels: sum the stored per-entry forward amounts into each
// destination. `rev_entry[e]` is the CSR entry index of the e-th incoming edge.
extern "C" __global__ void migration_stochastic_gather_female(
    float* ind_out,
    const int* rev_indptr,
    const int* rev_entry,
    const float* fwd_f,
    const float* fwd_s,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int fz = i % Z;
    int t = i / Z;
    int age = t % A;
    int dst = t / A;
    float total = 0.0f;
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        long long entry = rev_entry[e];
        total += fwd_f[entry * A * Z + age * Z + fz];
        for (int mz = 0; mz < Z; ++mz) {
            total += fwd_s[(entry * A + age) * Z * Z + fz * Z + mz];
        }
    }
    ind_out[((0 * A + age) * Z + fz) * n_batch + dst] += total;
}

extern "C" __global__ void migration_stochastic_gather_sperm(
    float* sperm_out,
    const int* rev_indptr,
    const int* rev_entry,
    const float* fwd_s,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes * n_ztypes;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int mz = i % Z;
    int t = i / Z;
    int fz = t % Z;
    t /= Z;
    int age = t % A;
    int dst = t / A;
    float total = 0.0f;
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        long long entry = rev_entry[e];
        total += fwd_s[(entry * A + age) * Z * Z + fz * Z + mz];
    }
    sperm_out[((age * Z + fz) * Z + mz) * n_batch + dst] += total;
}

extern "C" __global__ void migration_stochastic_gather_male(
    float* ind_out,
    const int* rev_indptr,
    const int* rev_entry,
    const float* fwd_m,
    int n_batch,
    int n_ages,
    int n_ztypes)
{
    int cells = n_batch * n_ages * n_ztypes;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int z = i % Z;
    int t = i / Z;
    int age = t % A;
    int dst = t / A;
    float total = 0.0f;
    for (int e = rev_indptr[dst]; e < rev_indptr[dst + 1]; ++e) {
        long long entry = rev_entry[e];
        total += fwd_m[entry * A * Z + age * Z + z];
    }
    ind_out[((1 * A + age) * Z + z) * n_batch + dst] += total;
}
"#;

/// Observation projection for device-side history rows.
///
/// This replicates `output::observation::project` for the batch-minor `ind`
/// layout: one thread computes one output cell, accumulating genotype, then
/// age, then deme in the same order as the host so the tolerance stays at the
/// f32 arithmetic level. The row's tick column is added on the host, so the
/// kernel only writes the projected values.
const OBSERVATION_SOURCE: &str = r#"
extern "C" __global__ void observation_project(
    const float* ind,
    const float* mask,
    const int* selected,
    int n_groups,
    int plane,
    int n_selected,
    int n_sexes,
    int n_ages,
    int n_ztypes,
    int n_batch,
    int collapse,
    int aggregate,
    int out_d,
    int out_a,
    int row_offset,
    float* out)
{
    int out_len = n_groups * out_d * n_sexes * out_a;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= out_len) {
        return;
    }
    int t = v;
    int age_out = t % out_a;
    t /= out_a;
    int sex = t % n_sexes;
    t /= n_sexes;
    int destination = t % out_d;
    t /= out_d;
    int group = t;
    int start_d = aggregate ? 0 : destination;
    int end_d = aggregate ? n_selected : destination + 1;
    int start_a = collapse ? 0 : age_out;
    int end_a = collapse ? n_ages : age_out + 1;
    float result = 0.0f;
    for (int di = start_d; di < end_d; ++di) {
        int deme = selected[di];
        float deme_total = 0.0f;
        for (int age = start_a; age < end_a; ++age) {
            float genotype_total = 0.0f;
            for (int g = 0; g < n_ztypes; ++g) {
                int offset = (sex * n_ages + age) * n_ztypes + g;
                genotype_total +=
                    ind[((long long)offset) * n_batch + deme] * mask[group * plane + offset];
            }
            deme_total += genotype_total;
        }
        result += deme_total;
    }
    out[row_offset + v] = result;
}
"#;

/// Discrete-generation (two-age) lifecycle kernels.
///
/// These mirror `kernels::discrete_generation` for the batch-minor `(2, A, Z, B)`
/// layout with `A == 2`: reproduction samples adult-female matings and
/// fertilizes the pairs into age-0 offspring, and survival applies the density
/// scaling (computed by [`Kernels::density_scaling`] with `discrete_actual`)
/// and then age-0 viability. Both discrete and continuous sampling branches are
/// provided.
const DISCRETE_SOURCE: &str = r#"
extern "C" __global__ void discrete_reproduction(
    float* ind,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int stochastic,
    int continuous,
    int has_sex_chromosomes,
    const float* mating_rates,
    const float* reproduction_rates,
    const float* eggs_per_female,
    const float* sex_ratio,
    const int* female_only,
    const int* male_only,
    const float* fecundity,
    const float* sexual_selection,
    const float* offspring,
    const float* female_compat,
    const float* male_compat,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    int adult = A - 1;
    float female_adult_rate = natal_clamp01(mating_rates[b * 2 * A + adult]);
    float male_adult_rate = natal_clamp01(mating_rates[b * 2 * A + A + adult]);

    float effective[NATAL_MAX_Z];
    float females_total = 0.0f;
    float males_total = 0.0f;
    for (int z = 0; z < Z; ++z) {
        effective[z] = ind[((1 * A + adult) * Z + z) * n_batch + b] * male_adult_rate;
        females_total += ind[((0 * A + adult) * Z + z) * n_batch + b];
        males_total += effective[z];
    }
    if (males_total == 0.0f || females_total == 0.0f) {
        return;
    }

    float mating_prob[NATAL_MAX_Z * NATAL_MAX_Z];
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

    RngState state;
    rng_init(&state, (unsigned long long)b, key0, key1, site);

    float pair_counts[NATAL_MAX_Z * NATAL_MAX_Z];
    for (int i = 0; i < Z * Z; ++i) {
        pair_counts[i] = 0.0f;
    }
    for (int gf = 0; gf < Z; ++gf) {
        float n_female = ind[((0 * A + adult) * Z + gf) * n_batch + b];
        if (n_female <= 0.0f) {
            continue;
        }
        float n_mating;
        if (stochastic) {
            n_mating = continuous
                ? sample_continuous_binomial(&state, n_female, female_adult_rate)
                : sample_binomial(&state, roundf(n_female), female_adult_rate);
        } else {
            n_mating = n_female * female_adult_rate;
        }
        if (n_mating <= 1e-10f) {
            continue;
        }
        const float* row = mating_prob + gf * Z;
        if (stochastic) {
            float drawn[NATAL_MAX_Z];
            if (continuous) {
                natal_continuous_multinomial(&state, n_mating, row, Z, drawn);
            } else {
                natal_multinomial(&state, roundf(n_mating), row, Z, drawn);
            }
            for (int gm = 0; gm < Z; ++gm) {
                pair_counts[gf * Z + gm] += drawn[gm];
            }
        } else {
            for (int gm = 0; gm < Z; ++gm) {
                pair_counts[gf * Z + gm] += n_mating * row[gm];
            }
        }
    }

    float offspring_acc[NATAL_MAX_Z];
    for (int z = 0; z < Z; ++z) {
        offspring_acc[z] = 0.0f;
    }
    float p_reproduce = natal_clamp01(reproduction_rates[b * A + adult]);
    float epf = eggs_per_female[b];
    if (!(epf > 0.0f)) {
        epf = 0.0f;
    }
    const float* ff = fecundity + b * 2 * Z;
    for (int gf = 0; gf < Z; ++gf) {
        for (int gm = 0; gm < Z; ++gm) {
            float n_pairs = pair_counts[gf * Z + gm];
            if (n_pairs <= 0.0f) {
                continue;
            }
            float eggs_per_pair = epf * ff[gf] * ff[Z + gm];
            float n_total;
            if (stochastic) {
                float n_pairs_eff = continuous ? n_pairs : roundf(n_pairs);
                if (n_pairs_eff <= 0.0f) {
                    continue;
                }
                float n_reproducing = n_pairs_eff;
                if (p_reproduce < 1.0f - 1e-10f) {
                    n_reproducing = continuous
                        ? sample_continuous_binomial(&state, n_pairs_eff, p_reproduce)
                        : sample_binomial(&state, n_pairs_eff, p_reproduce);
                }
                float lambda = fmaxf(n_reproducing * eggs_per_pair, 0.0f);
                n_total = continuous ? sample_continuous_poisson(&state, lambda)
                                     : sample_poisson(&state, lambda);
            } else {
                n_total = n_pairs * p_reproduce * eggs_per_pair;
            }
            if (n_total <= 1e-10f) {
                continue;
            }
            const float* off = offspring + b * Z * Z * Z + (gf * Z + gm) * Z;
            float p_surv = 0.0f;
            for (int go = 0; go < Z; ++go) {
                p_surv += off[go];
            }
            if (stochastic) {
                if (p_surv <= 1e-10f) {
                    continue;
                }
                float n_viable = n_total;
                if (p_surv < 1.0f - 1e-10f) {
                    n_viable = continuous
                        ? sample_continuous_binomial(&state, n_total, p_surv)
                        : sample_binomial(&state, roundf(n_total), p_surv);
                }
                if (n_viable <= 1e-10f) {
                    continue;
                }
                float inv = 1.0f / p_surv;
                float prob_norm[NATAL_MAX_Z];
                float drawn[NATAL_MAX_Z];
                for (int go = 0; go < Z; ++go) {
                    prob_norm[go] = off[go] * inv;
                }
                if (continuous) {
                    natal_continuous_multinomial(&state, n_viable, prob_norm, Z, drawn);
                } else {
                    natal_multinomial(&state, roundf(n_viable), prob_norm, Z, drawn);
                }
                for (int go = 0; go < Z; ++go) {
                    offspring_acc[go] += drawn[go];
                }
            } else {
                for (int go = 0; go < Z; ++go) {
                    offspring_acc[go] += n_total * off[go];
                }
            }
        }
    }

    float sr = natal_clamp01(sex_ratio[b]);
    const float* fcompat = female_compat + b * Z;
    const float* mcompat = male_compat + b * Z;
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
                    p_f = (denom > 1e-10f) ? natal_clamp01(fcompat[go] / denom) : 0.5f;
                } else {
                    p_f = sr;
                }
                float n_fem;
                if (stochastic) {
                    n_fem = continuous
                        ? sample_continuous_binomial(&state, n_g, p_f)
                        : sample_binomial(&state, roundf(n_g), p_f);
                } else {
                    n_fem = n_g * p_f;
                }
                n_f = n_fem;
                n_m = n_g - n_fem;
            }
        }
        ind[((0 * A + 0) * Z + go) * n_batch + b] = n_f;
        ind[((1 * A + 0) * Z + go) * n_batch + b] = n_m;
    }
}

extern "C" __global__ void discrete_survival(
    float* ind,
    int n_batch,
    int n_ages,
    int n_ztypes,
    int stochastic,
    int continuous,
    const float* scaling,
    const float* survival_rates,
    const float* viability,
    unsigned int key0,
    unsigned int key1,
    unsigned int site)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batch) {
        return;
    }
    int A = n_ages;
    int Z = n_ztypes;
    RngState state;
    rng_init(&state, (unsigned long long)b, key0, key1, site);

    float combined[2 * NATAL_MAX_Z];
    float total = 0.0f;
    for (int sex = 0; sex < 2; ++sex) {
        for (int z = 0; z < Z; ++z) {
            float raw = ind[((sex * A + 0) * Z + z) * n_batch + b];
            float value = (stochastic && !continuous) ? roundf(raw) : raw;
            combined[sex * Z + z] = value;
            total += value;
        }
    }
    if (total <= 0.0f) {
        return;
    }
    float desired = (stochastic && !continuous) ? roundf(total * scaling[b]) : total * scaling[b];
    if (desired <= 0.0f) {
        for (int sex = 0; sex < 2; ++sex) {
            for (int z = 0; z < Z; ++z) {
                ind[((sex * A + 0) * Z + z) * n_batch + b] = 0.0f;
            }
        }
        return;
    }
    float probs[2 * NATAL_MAX_Z];
    float draws[2 * NATAL_MAX_Z];
    for (int k = 0; k < 2 * Z; ++k) {
        probs[k] = combined[k] / total;
    }
    if (stochastic) {
        if (continuous) {
            natal_continuous_multinomial(&state, desired, probs, 2 * Z, draws);
        } else {
            natal_multinomial(&state, roundf(desired), probs, 2 * Z, draws);
        }
    } else {
        for (int k = 0; k < 2 * Z; ++k) {
            draws[k] = combined[k] * (desired / total);
        }
    }
    for (int sex = 0; sex < 2; ++sex) {
        for (int z = 0; z < Z; ++z) {
            ind[((sex * A + 0) * Z + z) * n_batch + b] = draws[sex * Z + z];
        }
    }

    for (int z = 0; z < Z; ++z) {
        float f = ind[((0 * A + 0) * Z + z) * n_batch + b];
        float m = ind[((1 * A + 0) * Z + z) * n_batch + b];
        float rate_f = survival_rates[b * 2 * A + 0] * viability[(b * 2 * A + 0) * Z + z];
        float rate_m = survival_rates[b * 2 * A + A + 0] * viability[(b * 2 * A + A + 0) * Z + z];
        float nf;
        float nm;
        if (stochastic) {
            nf = continuous ? sample_continuous_binomial(&state, f, rate_f)
                            : sample_binomial(&state, roundf(f), rate_f);
            nm = continuous ? sample_continuous_binomial(&state, m, rate_m)
                            : sample_binomial(&state, roundf(m), rate_m);
        } else {
            nf = f * rate_f;
            nm = m * rate_m;
        }
        ind[((0 * A + 0) * Z + z) * n_batch + b] = nf;
        ind[((1 * A + 0) * Z + z) * n_batch + b] = nm;
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
    /// `migration_female(...)`.
    migration_female: CudaFunction,
    /// `migration_sperm(...)`.
    migration_sperm: CudaFunction,
    /// `migration_male(...)`.
    migration_male: CudaFunction,
    /// `fill_uniform(...)`.
    fill_uniform: CudaFunction,
    /// `sample_into(...)`.
    sample_into: CudaFunction,
    /// `multinomial_seq(...)`.
    multinomial_seq: CudaFunction,
    /// `recruit_stochastic(...)`.
    recruit_stochastic: CudaFunction,
    /// `survival_stochastic(...)`.
    survival_stochastic: CudaFunction,
    /// `reproduction_stochastic(...)`.
    reproduction_stochastic: CudaFunction,
    /// `migration_stochastic_prepare(...)`.
    migration_stochastic_prepare: CudaFunction,
    /// `migration_stochastic_gather_female(...)`.
    migration_stochastic_gather_female: CudaFunction,
    /// `migration_stochastic_gather_sperm(...)`.
    migration_stochastic_gather_sperm: CudaFunction,
    /// `migration_stochastic_gather_male(...)`.
    migration_stochastic_gather_male: CudaFunction,
    /// `observation_project(...)`.
    observation_project: CudaFunction,
    /// `discrete_reproduction(...)`.
    discrete_reproduction: CudaFunction,
    /// `discrete_survival(...)`.
    discrete_survival: CudaFunction,
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
        let source = format!(
            "{AGING_SOURCE}\n{DENSITY_SOURCE}\n{SURVIVAL_SOURCE}\n{REPRODUCTION_SOURCE}\n{MIGRATION_SOURCE}\n{RNG_SOURCE}\n{SAMPLING_SOURCE}\n{OBSERVATION_SOURCE}\n{DISCRETE_SOURCE}"
        );
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
        let migration_female = module
            .load_function("migration_female")
            .map_err(|err| format!("loading kernel `migration_female` failed: {err}"))?;
        let migration_sperm = module
            .load_function("migration_sperm")
            .map_err(|err| format!("loading kernel `migration_sperm` failed: {err}"))?;
        let migration_male = module
            .load_function("migration_male")
            .map_err(|err| format!("loading kernel `migration_male` failed: {err}"))?;
        let fill_uniform = module
            .load_function("fill_uniform")
            .map_err(|err| format!("loading kernel `fill_uniform` failed: {err}"))?;
        let sample_into = module
            .load_function("sample_into")
            .map_err(|err| format!("loading kernel `sample_into` failed: {err}"))?;
        let multinomial_seq = module
            .load_function("multinomial_seq")
            .map_err(|err| format!("loading kernel `multinomial_seq` failed: {err}"))?;
        let recruit_stochastic = module
            .load_function("recruit_stochastic")
            .map_err(|err| format!("loading kernel `recruit_stochastic` failed: {err}"))?;
        let survival_stochastic = module
            .load_function("survival_stochastic")
            .map_err(|err| format!("loading kernel `survival_stochastic` failed: {err}"))?;
        let reproduction_stochastic = module
            .load_function("reproduction_stochastic")
            .map_err(|err| format!("loading kernel `reproduction_stochastic` failed: {err}"))?;
        let migration_stochastic_prepare = module
            .load_function("migration_stochastic_prepare")
            .map_err(|err| {
                format!("loading kernel `migration_stochastic_prepare` failed: {err}")
            })?;
        let migration_stochastic_gather_female = module
            .load_function("migration_stochastic_gather_female")
            .map_err(|err| {
                format!("loading kernel `migration_stochastic_gather_female` failed: {err}")
            })?;
        let migration_stochastic_gather_sperm = module
            .load_function("migration_stochastic_gather_sperm")
            .map_err(|err| {
                format!("loading kernel `migration_stochastic_gather_sperm` failed: {err}")
            })?;
        let migration_stochastic_gather_male = module
            .load_function("migration_stochastic_gather_male")
            .map_err(|err| {
                format!("loading kernel `migration_stochastic_gather_male` failed: {err}")
            })?;
        let observation_project = module
            .load_function("observation_project")
            .map_err(|err| format!("loading kernel `observation_project` failed: {err}"))?;
        let discrete_reproduction = module
            .load_function("discrete_reproduction")
            .map_err(|err| format!("loading kernel `discrete_reproduction` failed: {err}"))?;
        let discrete_survival = module
            .load_function("discrete_survival")
            .map_err(|err| format!("loading kernel `discrete_survival` failed: {err}"))?;
        Ok(Self {
            age_shift,
            density_scaling,
            recruit_factor,
            survival_scale_ind,
            survival_scale_sperm,
            reproduction,
            migration_female,
            migration_sperm,
            migration_male,
            fill_uniform,
            sample_into,
            multinomial_seq,
            recruit_stochastic,
            survival_stochastic,
            reproduction_stochastic,
            migration_stochastic_prepare,
            migration_stochastic_gather_female,
            migration_stochastic_gather_sperm,
            migration_stochastic_gather_male,
            observation_project,
            discrete_reproduction,
            discrete_survival,
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
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn density_scaling(
        &self,
        stream: &Arc<CudaStream>,
        buffers: &mut DensityBuffers<'_>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
        discrete_actual: bool,
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
        let discrete_i = i32::from(discrete_actual);
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
        launch.arg(&discrete_i);
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

    /// Run one deterministic CSR migration step in place across batch elements.
    ///
    /// Reads the pre-migration `ind_in`/`sperm_in` planes and writes fresh
    /// `ind_out`/`sperm_out` planes (the caller swaps them). `rev_indptr` /
    /// `rev_src` / `rev_weight` are the host-built reverse CSR; `row_sum_w` is
    /// each source row's weight sum.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launches are ordered on.
    /// - `ind_in`, `sperm_in`: Batch-minor source state.
    /// - `ind_out`, `sperm_out`: Batch-minor destination state.
    /// - `rate`: `(B, 2, A)` per-sex, per-age migration rates.
    /// - `indptr`: `(B + 1)` source row pointers.
    /// - `rev_indptr`, `rev_src`, `rev_weight`: Reverse CSR.
    /// - `row_sum_w`: `(B)` per-source row weight sums.
    /// - `stay_after`: Whether unmoved mass stays at the source (row sums).
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    ///
    /// ## Errors
    /// Returns a description when a launch fails.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn migration(
        &self,
        stream: &Arc<CudaStream>,
        ind_in: &CudaSlice<f32>,
        sperm_in: &CudaSlice<f32>,
        ind_out: &mut CudaSlice<f32>,
        sperm_out: &mut CudaSlice<f32>,
        rate: &CudaSlice<f32>,
        indptr: &CudaSlice<i32>,
        rev_indptr: &CudaSlice<i32>,
        rev_src: &CudaSlice<i32>,
        rev_weight: &CudaSlice<f32>,
        row_sum_w: &CudaSlice<f32>,
        stay_after: bool,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
    ) -> Result<(), String> {
        let cells = n_batch * n_ages * n_ztypes;
        if cells == 0 {
            return Ok(());
        }
        let stay_after_i = i32::from(stay_after);
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let cfg = LaunchConfig::for_num_elems(cells as u32);

        let mut female = stream.launch_builder(&self.migration_female);
        female.arg(ind_in);
        female.arg(sperm_in);
        female.arg(&mut *ind_out);
        female.arg(rate);
        female.arg(indptr);
        female.arg(rev_indptr);
        female.arg(rev_src);
        female.arg(rev_weight);
        female.arg(row_sum_w);
        female.arg(&stay_after_i);
        female.arg(&n_batch_i);
        female.arg(&n_ages_i);
        female.arg(&n_ztypes_i);
        unsafe { female.launch(cfg) }
            .map(|_| ())
            .map_err(|err| format!("migration_female launch failed: {err}"))?;

        let mut male = stream.launch_builder(&self.migration_male);
        male.arg(ind_in);
        male.arg(&mut *ind_out);
        male.arg(rate);
        male.arg(indptr);
        male.arg(rev_indptr);
        male.arg(rev_src);
        male.arg(rev_weight);
        male.arg(row_sum_w);
        male.arg(&stay_after_i);
        male.arg(&n_batch_i);
        male.arg(&n_ages_i);
        male.arg(&n_ztypes_i);
        unsafe { male.launch(cfg) }
            .map(|_| ())
            .map_err(|err| format!("migration_male launch failed: {err}"))?;

        let sperm_cells = cells * n_ztypes;
        let sperm_cfg = LaunchConfig::for_num_elems(sperm_cells as u32);
        let mut sperm = stream.launch_builder(&self.migration_sperm);
        sperm.arg(sperm_in);
        sperm.arg(&mut *sperm_out);
        sperm.arg(rate);
        sperm.arg(indptr);
        sperm.arg(rev_indptr);
        sperm.arg(rev_src);
        sperm.arg(rev_weight);
        sperm.arg(row_sum_w);
        sperm.arg(&stay_after_i);
        sperm.arg(&n_batch_i);
        sperm.arg(&n_ages_i);
        sperm.arg(&n_ztypes_i);
        unsafe { sperm.launch(sperm_cfg) }
            .map(|_| ())
            .map_err(|err| format!("migration_sperm launch failed: {err}"))
    }

    /// Fill `out` with counter-based Philox4x32-10 uniforms in `[0, 1)`.
    ///
    /// Each element is a pure function of `(key0, key1, site, index)`, so the
    /// sequence is reproducible regardless of thread count or scheduling.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `out`: Destination uniforms.
    /// - `key0`, `key1`: 64-bit key split into two words (e.g. from the seed).
    /// - `site`: Draw-site counter distinguishing different draws in a tick.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    pub fn fill_uniform(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<f32>,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        let n = out.len() as u64;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n as u32);
        let mut launch = stream.launch_builder(&self.fill_uniform);
        launch.arg(&mut *out);
        launch.arg(&n);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("fill_uniform launch failed: {err}"))
    }

    /// Fill `out` by drawing one sample per element from a selected sampler.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `out`: Destination samples.
    /// - `kind`: `0` uniform, `1` binomial(`a`,`b`), `2` Poisson(`a`),
    ///   `3` gamma(`a` shape, `b` scale), other normal.
    /// - `a`, `b`: Distribution parameters.
    /// - `key0`, `key1`, `site`: Philox key and draw-site.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn sample_into(
        &self,
        stream: &Arc<CudaStream>,
        out: &mut CudaSlice<f32>,
        kind: u32,
        a: f32,
        b: f32,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        let n = out.len() as u64;
        if n == 0 {
            return Ok(());
        }
        let config = LaunchConfig::for_num_elems(n as u32);
        let mut launch = stream.launch_builder(&self.sample_into);
        launch.arg(&mut *out);
        launch.arg(&n);
        launch.arg(&kind);
        launch.arg(&a);
        launch.arg(&b);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("sample_into launch failed: {err}"))
    }

    /// Draw one multinomial row per batch element.
    ///
    /// `probs`/`counts` are `(n_rows, n_categories)` row-major and `totals` is
    /// `(n_rows,)`. The sequential conditional-binomial loop is the outer axis
    /// shared by all threads.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn multinomial_seq(
        &self,
        stream: &Arc<CudaStream>,
        probs: &CudaSlice<f32>,
        totals: &CudaSlice<f32>,
        counts: &mut CudaSlice<f32>,
        n_rows: usize,
        n_categories: usize,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_rows == 0 || n_categories == 0 {
            return Err("multinomial_seq requires non-zero rows and categories".to_owned());
        }
        let n_rows_i = n_rows as i32;
        let n_categories_i = n_categories as i32;
        let config = LaunchConfig::for_num_elems(n_rows as u32);
        let mut launch = stream.launch_builder(&self.multinomial_seq);
        launch.arg(probs);
        launch.arg(totals);
        launch.arg(&mut *counts);
        launch.arg(&n_rows_i);
        launch.arg(&n_categories_i);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("multinomial_seq launch failed: {err}"))
    }

    /// Resample the age-0 categories with the discrete multinomial.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn recruit_stochastic(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        scaling: &CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        continuous: bool,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "recruit_stochastic supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let continuous_i = i32::from(continuous);
        let config = stochastic_grid(n_batch);
        let mut launch = stream.launch_builder(&self.recruit_stochastic);
        launch.arg(&mut *ind);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&continuous_i);
        launch.arg(scaling);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("recruit_stochastic launch failed: {err}"))
    }

    /// Sample stochastic survival on individuals and stored sperm in place.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn survival_stochastic(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        sperm: &mut CudaSlice<f32>,
        survival_rates: &CudaSlice<f32>,
        viability: &CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        new_adult_age: usize,
        continuous: bool,
        violation: &mut CudaSlice<i32>,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        let cells = n_batch * n_ages * n_ztypes;
        if cells == 0 {
            return Ok(());
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let continuous_i = i32::from(continuous);
        let config = stochastic_grid(cells);
        let mut launch = stream.launch_builder(&self.survival_stochastic);
        launch.arg(&mut *ind);
        launch.arg(&mut *sperm);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&continuous_i);
        launch.arg(&new_adult_i);
        launch.arg(survival_rates);
        launch.arg(viability);
        launch.arg(&mut *violation);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("survival_stochastic launch failed: {err}"))
    }

    /// Run the stochastic reproduction stage in place.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn reproduction_stochastic(
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
        fixed_egg_count: bool,
        continuous: bool,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "reproduction_stochastic supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let new_adult_i = new_adult_age as i32;
        let sex_chrom_i = i32::from(has_sex_chromosomes);
        let fixed_eggs_i = i32::from(fixed_egg_count);
        let continuous_i = i32::from(continuous);
        let config = LaunchConfig {
            grid_dim: ((n_batch as u32).div_ceil(64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.reproduction_stochastic);
        launch.arg(&mut *ind);
        launch.arg(&mut *sperm);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&continuous_i);
        launch.arg(&new_adult_i);
        launch.arg(&sex_chrom_i);
        launch.arg(&fixed_eggs_i);
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
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("reproduction_stochastic launch failed: {err}"))
    }

    /// Run one deterministic-scatter stochastic CSR migration across the batch.
    ///
    /// Pass 1 samples each source's outbound and multinomial split into
    /// `fwd_*` (indexed by CSR entry) and writes the source's stay mass into
    /// `ind_out`/`sperm_out`; pass 2 gathers the forward amounts per
    /// destination. No atomics, so the result is reproducible run to run.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launches are ordered on.
    /// - `ind_in`, `sperm_in`: Batch-minor source state.
    /// - `ind_out`, `sperm_out`: Batch-minor destination state.
    /// - `rate`: `(B, 2, A)` migration rates.
    /// - `indptr`, `dest`, `weights`: CSR (destination and weight arrays are
    ///   only read by pass 1).
    /// - `fwd_f`, `fwd_s`, `fwd_m`: Entry-indexed forward scratch.
    /// - `rev_indptr`, `rev_entry`: Reverse CSR storing CSR entry indices.
    /// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
    /// - `key0`, `key1`, `site`: Philox key and draw-site.
    ///
    /// ## Errors
    /// Returns a description when a launch fails.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signatures.
    pub fn migration_stochastic(
        &self,
        stream: &Arc<CudaStream>,
        ind_in: &CudaSlice<f32>,
        sperm_in: &CudaSlice<f32>,
        ind_out: &mut CudaSlice<f32>,
        sperm_out: &mut CudaSlice<f32>,
        rate: &CudaSlice<f32>,
        indptr: &CudaSlice<i32>,
        dest: &CudaSlice<i32>,
        weights: &CudaSlice<f32>,
        fwd_f: &mut CudaSlice<f32>,
        fwd_s: &mut CudaSlice<f32>,
        fwd_m: &mut CudaSlice<f32>,
        rev_indptr: &CudaSlice<i32>,
        rev_entry: &CudaSlice<i32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        continuous: bool,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_batch == 0 || n_ages == 0 || n_ztypes == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "migration_stochastic supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let continuous_i = i32::from(continuous);
        let cells = n_batch * n_ages * n_ztypes;
        let grid_batch = stochastic_grid(n_batch);
        let grid_cells = LaunchConfig::for_num_elems(cells as u32);
        let grid_sperm = LaunchConfig::for_num_elems((cells * n_ztypes) as u32);

        let mut prepare = stream.launch_builder(&self.migration_stochastic_prepare);
        prepare.arg(ind_in);
        prepare.arg(sperm_in);
        prepare.arg(&mut *ind_out);
        prepare.arg(&mut *sperm_out);
        prepare.arg(rate);
        prepare.arg(indptr);
        prepare.arg(dest);
        prepare.arg(weights);
        prepare.arg(&mut *fwd_f);
        prepare.arg(&mut *fwd_s);
        prepare.arg(&mut *fwd_m);
        prepare.arg(&n_batch_i);
        prepare.arg(&n_ages_i);
        prepare.arg(&n_ztypes_i);
        prepare.arg(&continuous_i);
        prepare.arg(&key0);
        prepare.arg(&key1);
        prepare.arg(&site);
        unsafe { prepare.launch(grid_batch) }
            .map(|_| ())
            .map_err(|err| format!("migration_stochastic_prepare launch failed: {err}"))?;

        let mut female = stream.launch_builder(&self.migration_stochastic_gather_female);
        female.arg(&mut *ind_out);
        female.arg(rev_indptr);
        female.arg(rev_entry);
        female.arg(&*fwd_f);
        female.arg(&*fwd_s);
        female.arg(&n_batch_i);
        female.arg(&n_ages_i);
        female.arg(&n_ztypes_i);
        unsafe { female.launch(grid_cells) }
            .map(|_| ())
            .map_err(|err| format!("migration_stochastic_gather_female launch failed: {err}"))?;

        let mut sperm = stream.launch_builder(&self.migration_stochastic_gather_sperm);
        sperm.arg(&mut *sperm_out);
        sperm.arg(rev_indptr);
        sperm.arg(rev_entry);
        sperm.arg(&*fwd_s);
        sperm.arg(&n_batch_i);
        sperm.arg(&n_ages_i);
        sperm.arg(&n_ztypes_i);
        unsafe { sperm.launch(grid_sperm) }
            .map(|_| ())
            .map_err(|err| format!("migration_stochastic_gather_sperm launch failed: {err}"))?;

        let mut male = stream.launch_builder(&self.migration_stochastic_gather_male);
        male.arg(&mut *ind_out);
        male.arg(rev_indptr);
        male.arg(rev_entry);
        male.arg(&*fwd_m);
        male.arg(&n_batch_i);
        male.arg(&n_ages_i);
        male.arg(&n_ztypes_i);
        unsafe { male.launch(grid_cells) }
            .map(|_| ())
            .map_err(|err| format!("migration_stochastic_gather_male launch failed: {err}"))
    }

    /// Project the current device state into one history row.
    ///
    /// Mirrors `output::observation::project` for the batch-minor `ind` layout;
    /// the caller appends the tick column on the host.
    ///
    /// ## Parameters
    /// - `stream`: Stream the launch is ordered on.
    /// - `ind`: Batch-minor individual state, `(2, A, Z, B)`.
    /// - `mask`: Observation weights, `groups · S · A · Z`.
    /// - `selected`: Deme indices to keep, length `n_selected`.
    /// - `n_groups`, `plane`, `n_selected`, `n_sexes`, `n_ages`, `n_ztypes`,
    ///   `n_batch`: Projection dimensions (`plane = S · A · Z`).
    /// - `collapse`, `aggregate`: Reduction flags.
    /// - `out_d`, `out_a`: Output deme/age extents.
    /// - `row_offset`: Element offset of the destination row inside `out`.
    /// - `out`: Destination buffer; the row is written at `row_offset`.
    ///
    /// ## Errors
    /// Returns a description when a launch fails.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn observation_project(
        &self,
        stream: &Arc<CudaStream>,
        ind: &CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        selected: &CudaSlice<i32>,
        n_groups: usize,
        plane: usize,
        n_selected: usize,
        n_sexes: usize,
        n_ages: usize,
        n_ztypes: usize,
        n_batch: usize,
        collapse: bool,
        aggregate: bool,
        out_d: usize,
        out_a: usize,
        row_offset: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), String> {
        let out_len = n_groups * out_d * n_sexes * out_a;
        if out_len == 0 {
            return Ok(());
        }
        let n_groups_i = n_groups as i32;
        let plane_i = plane as i32;
        let n_selected_i = n_selected as i32;
        let n_sexes_i = n_sexes as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let n_batch_i = n_batch as i32;
        let collapse_i = collapse as i32;
        let aggregate_i = aggregate as i32;
        let out_d_i = out_d as i32;
        let out_a_i = out_a as i32;
        let row_offset_i = row_offset as i32;
        let grid = LaunchConfig::for_num_elems(out_len as u32);
        let mut builder = stream.launch_builder(&self.observation_project);
        builder.arg(ind);
        builder.arg(mask);
        builder.arg(selected);
        builder.arg(&n_groups_i);
        builder.arg(&plane_i);
        builder.arg(&n_selected_i);
        builder.arg(&n_sexes_i);
        builder.arg(&n_ages_i);
        builder.arg(&n_ztypes_i);
        builder.arg(&n_batch_i);
        builder.arg(&collapse_i);
        builder.arg(&aggregate_i);
        builder.arg(&out_d_i);
        builder.arg(&out_a_i);
        builder.arg(&row_offset_i);
        builder.arg(out);
        unsafe { builder.launch(grid) }
            .map(|_| ())
            .map_err(|err| format!("observation_project launch failed: {err}"))
    }

    /// Run the discrete-generation reproduction stage in place.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn discrete_reproduction(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        buffers: &ReproductionBuffers<'_>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        stochastic: bool,
        continuous: bool,
        has_sex_chromosomes: bool,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "discrete_reproduction supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let stochastic_i = i32::from(stochastic);
        let continuous_i = i32::from(continuous);
        let sex_chrom_i = i32::from(has_sex_chromosomes);
        let config = stochastic_grid(n_batch);
        let mut launch = stream.launch_builder(&self.discrete_reproduction);
        launch.arg(&mut *ind);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&stochastic_i);
        launch.arg(&continuous_i);
        launch.arg(&sex_chrom_i);
        launch.arg(buffers.mating_rates);
        launch.arg(buffers.reproduction_rates);
        launch.arg(buffers.eggs_per_female);
        launch.arg(buffers.sex_ratio);
        launch.arg(buffers.female_only);
        launch.arg(buffers.male_only);
        launch.arg(buffers.fecundity);
        launch.arg(buffers.sexual_selection);
        launch.arg(buffers.offspring);
        launch.arg(buffers.female_compat);
        launch.arg(buffers.male_compat);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("discrete_reproduction launch failed: {err}"))
    }

    /// Run the discrete-generation survival stage in place.
    ///
    /// `scaling` comes from [`Kernels::density_scaling`] with
    /// `discrete_actual = true`.
    ///
    /// ## Errors
    /// Returns a description when the driver rejects the launch.
    #[allow(clippy::too_many_arguments)] // Flat kernel arguments mirror the CUDA signature.
    pub fn discrete_survival(
        &self,
        stream: &Arc<CudaStream>,
        ind: &mut CudaSlice<f32>,
        n_batch: usize,
        n_ages: usize,
        n_ztypes: usize,
        stochastic: bool,
        continuous: bool,
        scaling: &CudaSlice<f32>,
        survival_rates: &CudaSlice<f32>,
        viability: &CudaSlice<f32>,
        key0: u32,
        key1: u32,
        site: u32,
    ) -> Result<(), String> {
        if n_batch == 0 {
            return Ok(());
        }
        if n_ztypes > MAX_Z {
            return Err(format!(
                "discrete_survival supports at most {MAX_Z} zygote types, got {n_ztypes}"
            ));
        }
        let n_batch_i = n_batch as i32;
        let n_ages_i = n_ages as i32;
        let n_ztypes_i = n_ztypes as i32;
        let stochastic_i = i32::from(stochastic);
        let continuous_i = i32::from(continuous);
        let config = stochastic_grid(n_batch);
        let mut launch = stream.launch_builder(&self.discrete_survival);
        launch.arg(&mut *ind);
        launch.arg(&n_batch_i);
        launch.arg(&n_ages_i);
        launch.arg(&n_ztypes_i);
        launch.arg(&stochastic_i);
        launch.arg(&continuous_i);
        launch.arg(scaling);
        launch.arg(survival_rates);
        launch.arg(viability);
        launch.arg(&key0);
        launch.arg(&key1);
        launch.arg(&site);
        unsafe { launch.launch(config) }
            .map(|_| ())
            .map_err(|err| format!("discrete_survival launch failed: {err}"))
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gpu/kernels.rs"]
mod tests;
