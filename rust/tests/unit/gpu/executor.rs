//! P2 executor tests.
//!
//! The budget arithmetic is host-only; the L1 aging cross-check needs a GPU
//! and asserts by default, disabled with `NATAL_GPU_REQUIRE=0`.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;
use crate::kernels::age_structured::{aging, reproduction, survival};
use crate::kernels::density_regulation::regulation_scaling;
use crate::kernels::discrete_generation::{
    reproduction as discrete_reproduction, survival as discrete_survival,
};
use crate::kernels::equilibrium::equilibrium_metrics;
use crate::kernels::rng::new_rng;
use crate::kernels::spatial::migrate_csr_deterministic;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use crate::output::observation::project;

use super::{GpuExecutor, HistorySpec};
use std::collections::HashMap;

/// Build a blueprint carrying only the dimensions `aging` reads.
///
/// ## Parameters
/// - `n_ages`: Length of the age axis.
/// - `n_ztypes`: Length of the zygote-type axis.
///
/// ## Returns
/// A blueprint with `2` sexes and no migration/initial state.
fn dimension_blueprint(n_ages: usize, n_ztypes: usize) -> Blueprint {
    Blueprint {
        n_sexes: 2,
        n_ages,
        n_ztypes,
        n_gtypes: n_ztypes,
        n_glabs: 1,
        new_adult_age: 1,
        adult_ages: (1..n_ages as i64).collect(),
        stochastic: false,
        continuous_sampling: false,
        fixed_egg_count: false,
        has_sex_chromosomes: false,
        extreme_speed_mode: 0,
        ztype_names: (0..n_ztypes).map(|z| format!("z{z}")).collect(),
        gtype_names: (0..n_ztypes).map(|g| format!("g{g}")).collect(),
        female_only_by_sex_chrom: vec![false; n_ztypes],
        male_only_by_sex_chrom: vec![false; n_ztypes],
        initial_individual_count: vec![],
        initial_sperm_storage: vec![],
        n_demes: 1,
        migration_indptr: vec![0, 0],
        migration_dest_idx: vec![],
        migration_weights: vec![],
    }
}

/// Raw bit patterns of an `f32` slice, for exact comparison.
///
/// ## Parameters
/// - `values`: Floating-point values to canonicalize.
///
/// ## Returns
/// One `u32` per value.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn state_bytes_matches_the_layout() {
    // (2·A·Z + A·Z²) · B · 4.
    assert_eq!(
        GpuExecutor::state_bytes(7, 8, 9),
        (2 * 8 * 9 + 8 * 9 * 9) * 7 * 4
    );
    assert_eq!(GpuExecutor::state_bytes(0, 8, 9), 0);
}

#[test]
fn device_aging_matches_the_cpu_reference_bit_for_bit() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 5;
    let n_ages = 8;
    let n_ztypes = 9;
    let ind_len = 2 * n_ages * n_ztypes * n_batch;
    let sperm_len = n_ages * n_ztypes * n_ztypes * n_batch;
    // Small integers are exact in f32, so every difference is a real bug.
    let ind_f32: Vec<f32> = (0..ind_len).map(|i| (i % 1000) as f32).collect();
    let sperm_f32: Vec<f32> = (0..sperm_len).map(|i| ((i * 7) % 1000) as f32).collect();

    let blueprint = dimension_blueprint(n_ages, n_ztypes);
    let mut ind_ref: Vec<f64> = ind_f32.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm_f32.iter().map(|v| f64::from(*v)).collect();
    // The host `aging` kernel is per-instance `(2, A, Z)`; the device kernel is
    // batch-aware, so the reference ages every batch element independently.
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    for batch in 0..n_batch {
        aging(
            &blueprint,
            &mut ind_ref[batch * ind_stride..(batch + 1) * ind_stride],
            &mut sperm_ref[batch * sperm_stride..(batch + 1) * sperm_stride],
        );
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind_f32, &sperm_f32)
        .expect("executor uploads and compiles");
    executor.age_tick().expect("aging launch");

    let ind_out = executor.download_ind().expect("download individuals");
    let sperm_out = executor.download_sperm().expect("download sperm");
    let ind_expected: Vec<f32> = ind_ref.iter().map(|v| *v as f32).collect();
    let sperm_expected: Vec<f32> = sperm_ref.iter().map(|v| *v as f32).collect();
    assert_eq!(bits(&ind_out), bits(&ind_expected));
    assert_eq!(bits(&sperm_out), bits(&sperm_expected));

    // Sanity: age 0 is zeroed for both planes.
    for batch in 0..n_batch {
        for sex in 0..2 {
            for z in 0..n_ztypes {
                assert_eq!(
                    ind_out[batch * ind_stride + sex * n_ages * n_ztypes + z],
                    0.0
                );
            }
        }
        for zf in 0..n_ztypes {
            for zm in 0..n_ztypes {
                assert_eq!(sperm_out[batch * sperm_stride + zf * n_ztypes + zm], 0.0);
            }
        }
    }
}

/// Host reference for `scaling_factor`, composed from the public building
/// blocks (`equilibrium_metrics` + `regulation_scaling`).
///
/// ## Parameters
/// - `blueprint`: Model dimensions.
/// - `ecology`: Per-deme ecology columns.
/// - `deme`: Deme index to evaluate.
/// - `ind`: That deme's `(2, A, Z)` individual counts.
///
/// ## Returns
/// The expected scaling factor.
fn cpu_scaling(blueprint: &Blueprint, ecology: &EcologyParams, deme: usize, ind: &[f64]) -> f64 {
    let n_ages = blueprint.n_ages;
    let n_ztypes = blueprint.n_ztypes;
    let mode = ecology.growth_mode[deme];
    if mode == 0 {
        return 1.0;
    }
    if mode == 1 {
        let mut female_sum = 0.0;
        let mut male_sum = 0.0;
        for z in 0..n_ztypes {
            female_sum += ind[(0 * n_ages + 0) * n_ztypes + z];
            male_sum += ind[(1 * n_ages + 0) * n_ztypes + z];
        }
        return regulation_scaling(
            1,
            female_sum + male_sum,
            ecology.carrying_capacity[deme],
            0.0,
            0.0,
        )
        .unwrap_or(1.0);
    }
    let mut actual = 0.0;
    for age in 0..blueprint.new_adult_age {
        let mut female_sum = 0.0;
        let mut male_sum = 0.0;
        for z in 0..n_ztypes {
            female_sum += ind[(0 * n_ages + age) * n_ztypes + z];
            male_sum += ind[(1 * n_ages + age) * n_ztypes + z];
        }
        actual += (female_sum + male_sum) * ecology.competition_weights[deme * n_ages + age];
    }
    let (expected_comp, expected_surv) = equilibrium_metrics(blueprint, ecology, deme);
    regulation_scaling(
        mode,
        actual,
        expected_comp,
        ecology.low_density_growth_rate[deme],
        expected_surv,
    )
    .unwrap_or(1.0)
}

/// Four demes, one per built-in growth mode (`none`, `fixed`, `linear`,
/// `beverton_holt`).
///
/// ## Returns
/// A blueprint plus per-deme ecology columns for the density cross-check.
fn density_fixture() -> (Blueprint, EcologyParams) {
    let n_demes = 4;
    let n_ages = 4;
    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_demes).flat_map(|_| values.iter().copied()).collect()
    };
    let ecology = EcologyParams {
        n_demes,
        carrying_capacity: vec![400.0; n_demes],
        eggs_per_female: vec![30.0; n_demes],
        sex_ratio: vec![0.5; n_demes],
        sperm_displacement_rate: vec![0.1; n_demes],
        low_density_growth_rate: vec![2.0; n_demes],
        growth_mode: vec![0, 1, 2, 3],
        external_expected_eggs: vec![-1.0; n_demes],
        survival_rates: tile(&[0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55]),
        mating_rates: tile(&[0.0, 0.9, 0.8, 0.7, 0.0, 0.9, 0.8, 0.7]),
        reproduction_rates: tile(&[0.0, 0.8, 0.7, 0.6]),
        fertility: tile(&[0.0, 1.0, 0.9, 0.8]),
        competition_weights: tile(&[1.0, 0.8, 0.7, 0.6]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_demes],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new(); n_demes],
    };
    (dimension_blueprint(n_ages, 2), ecology)
}

#[test]
fn device_density_scaling_matches_the_host_reference() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let ind_len = n_batch * 2 * n_ages * n_ztypes;
    let sperm_len = n_batch * n_ages * n_ztypes * n_ztypes;
    let ind: Vec<f32> = (0..ind_len).map(|i| ((i * 13) % 97) as f32).collect();
    let sperm = vec![0.0f32; sperm_len];
    let (blueprint, ecology) = density_fixture();
    let context = GpuContext::new(0).expect("device 0 context");
    let executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    let device = executor
        .density_scaling(&blueprint, &ecology)
        .expect("density launch");
    let stride = 2 * n_ages * n_ztypes;
    for deme in 0..n_batch {
        let local: Vec<f64> = ind[deme * stride..(deme + 1) * stride]
            .iter()
            .map(|value| f64::from(*value))
            .collect();
        let expected = cpu_scaling(&blueprint, &ecology, deme, &local) as f32;
        let got = device[deme];
        let tolerance = 1e-5f32 * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= tolerance,
            "deme {deme}: device {got} vs host {expected}"
        );
    }
}

/// All-ones genetics, so viability is neutral and the survival cross-check
/// isolates the density / recruitment / rate logic.
///
/// ## Parameters
/// - `n_ages`, `n_ztypes`: Model dimensions.
///
/// ## Returns
/// A neutral [`GeneticsTensors`].
fn identity_genetics(n_ages: usize, n_ztypes: usize) -> GeneticsTensors {
    GeneticsTensors {
        viability_fitness: vec![1.0; 2 * n_ages * n_ztypes],
        fecundity_fitness: vec![1.0; 2 * n_ztypes],
        sexual_selection_fitness: vec![1.0; n_ztypes * n_ztypes],
        zygote_viability_fitness: vec![1.0; 2 * n_ztypes],
        offspring_tensor: vec![0.0; n_ztypes * n_ztypes * n_ztypes],
        meiosis_map: vec![0.0; 2 * n_ztypes * n_ztypes],
        female_ztype_compatibility: vec![0.5; n_ztypes],
        male_ztype_compatibility: vec![0.5; n_ztypes],
    }
}

#[test]
fn device_survival_matches_the_host_reference() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    let ind: Vec<f32> = (0..n_batch * ind_stride)
        .map(|i| ((i * 7) % 61) as f32)
        .collect();
    let sperm: Vec<f32> = (0..n_batch * sperm_stride)
        .map(|i| ((i * 5) % 37) as f32)
        .collect();
    let genetics = identity_genetics(n_ages, n_ztypes);
    let variants = vec![genetics.clone()];

    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    for deme in 0..n_batch {
        let mut rng = new_rng(1234);
        survival(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            deme,
            &mut ind_ref[deme * ind_stride..(deme + 1) * ind_stride],
            &mut sperm_ref[deme * sperm_stride..(deme + 1) * sperm_stride],
        )
        .expect("host survival");
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("survival launch");
    let ind_out = executor.download_ind().expect("download individuals");
    let sperm_out = executor.download_sperm().expect("download sperm");
    for (index, (got, want)) in ind_out.iter().zip(ind_ref.iter()).enumerate() {
        let want = *want as f32;
        let tolerance = 1e-5f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "ind[{index}]: device {got} vs host {want}"
        );
    }
    for (index, (got, want)) in sperm_out.iter().zip(sperm_ref.iter()).enumerate() {
        let want = *want as f32;
        let tolerance = 1e-5f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "sperm[{index}]: device {got} vs host {want}"
        );
    }
}

/// Non-trivial genetics for the reproduction cross-check: identity sexual
/// selection and a Mendelian-like offspring tensor `go = (gf + gm) mod Z`.
///
/// ## Parameters
/// - `n_ages`, `n_ztypes`: Model dimensions.
///
/// ## Returns
/// Genetics with unit fecundity/zygote viability and a real offspring table.
fn reproduction_genetics(n_ages: usize, n_ztypes: usize) -> GeneticsTensors {
    let z = n_ztypes;
    let mut sexual_selection = vec![0.0; z * z];
    for g in 0..z {
        sexual_selection[g * z + g] = 1.0;
    }
    let mut offspring = vec![0.0; z * z * z];
    for gf in 0..z {
        for gm in 0..z {
            offspring[(gf * z + gm) * z + (gf + gm) % z] = 1.0;
        }
    }
    GeneticsTensors {
        viability_fitness: vec![1.0; 2 * n_ages * z],
        fecundity_fitness: vec![1.0; 2 * z],
        sexual_selection_fitness: sexual_selection,
        zygote_viability_fitness: vec![1.0; 2 * z],
        offspring_tensor: offspring,
        meiosis_map: vec![0.0; 2 * z * z],
        female_ztype_compatibility: vec![0.5; z],
        male_ztype_compatibility: vec![0.5; z],
    }
}

#[test]
fn device_reproduction_matches_the_host_reference() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    let mut ind = vec![0.0f32; n_batch * ind_stride];
    let mut sperm = vec![0.0f32; n_batch * sperm_stride];
    for batch in 0..n_batch {
        for age in 0..n_ages {
            for z in 0..n_ztypes {
                ind[batch * ind_stride + (0 * n_ages + age) * n_ztypes + z] =
                    ((batch + age + z + 1) * 3) as f32;
                ind[batch * ind_stride + (1 * n_ages + age) * n_ztypes + z] =
                    ((batch + age + z + 2) * 2) as f32;
            }
        }
        for age in 0..n_ages {
            for gf in 0..n_ztypes {
                for gm in 0..n_ztypes {
                    sperm[batch * sperm_stride + (age * n_ztypes + gf) * n_ztypes + gm] =
                        (age + gf + gm + batch + 1) as f32;
                }
            }
        }
    }
    let genetics = reproduction_genetics(n_ages, n_ztypes);
    let variants = vec![genetics.clone()];

    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    for deme in 0..n_batch {
        let mut rng = new_rng(42);
        reproduction(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            deme,
            &mut ind_ref[deme * ind_stride..(deme + 1) * ind_stride],
            &mut sperm_ref[deme * sperm_stride..(deme + 1) * sperm_stride],
        )
        .expect("host reproduction");
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .reproduction_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("reproduction launch");
    let ind_out = executor.download_ind().expect("download individuals");
    let sperm_out = executor.download_sperm().expect("download sperm");
    for (label, got, want) in [
        ("ind", &ind_out, &ind_ref),
        ("sperm", &sperm_out, &sperm_ref),
    ] {
        for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            let want = *want as f32;
            let tolerance = 1e-5f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "{label}[{index}]: device {got} vs host {want}"
            );
        }
    }
}

/// Build a non-trivial deterministic state for the lifecycle tests.
///
/// ## Parameters
/// - `n_batch`, `n_ages`, `n_ztypes`: Model dimensions.
///
/// ## Returns
/// `(individual_counts, sperm_storage)` in batch-major layout.
fn populated_state(n_batch: usize, n_ages: usize, n_ztypes: usize) -> (Vec<f32>, Vec<f32>) {
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    let mut ind = vec![0.0f32; n_batch * ind_stride];
    let mut sperm = vec![0.0f32; n_batch * sperm_stride];
    for batch in 0..n_batch {
        for age in 0..n_ages {
            for z in 0..n_ztypes {
                ind[batch * ind_stride + (0 * n_ages + age) * n_ztypes + z] =
                    ((batch + age + z + 1) * 3) as f32;
                ind[batch * ind_stride + (1 * n_ages + age) * n_ztypes + z] =
                    ((batch + age + z + 2) * 2) as f32;
            }
        }
        for age in 0..n_ages {
            for gf in 0..n_ztypes {
                for gm in 0..n_ztypes {
                    sperm[batch * sperm_stride + (age * n_ztypes + gf) * n_ztypes + gm] =
                        (age + gf + gm + batch + 1) as f32;
                }
            }
        }
    }
    (ind, sperm)
}

#[test]
fn device_full_tick_matches_the_host_reference() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let ind_stride = 2 * n_ages * n_ztypes;
    let sperm_stride = n_ages * n_ztypes * n_ztypes;
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let genetics = reproduction_genetics(n_ages, n_ztypes);
    let variants = vec![genetics.clone()];

    // Host reference tick: reproduction → survival → aging, per deme.
    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    for deme in 0..n_batch {
        let mut rng = new_rng(7);
        let ind_slice = &mut ind_ref[deme * ind_stride..(deme + 1) * ind_stride];
        let sperm_slice = &mut sperm_ref[deme * sperm_stride..(deme + 1) * sperm_stride];
        reproduction(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            deme,
            ind_slice,
            sperm_slice,
        )
        .expect("host reproduction");
        survival(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            deme,
            ind_slice,
            sperm_slice,
        )
        .expect("host survival");
        aging(&blueprint, ind_slice, sperm_slice);
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("device tick");
    let ind_out = executor.download_ind().expect("download individuals");
    let sperm_out = executor.download_sperm().expect("download sperm");
    // A whole tick compounds several f32 stages, so the tolerance is looser
    // than a single kernel's (still far below any biological effect).
    for (label, got, want) in [
        ("ind", &ind_out, &ind_ref),
        ("sperm", &sperm_out, &sperm_ref),
    ] {
        for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            let want = *want as f32;
            let tolerance = 1e-4f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "{label}[{index}]: device {got} vs host {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluator adversarial additions (independent of the author's fixtures).
//
// The author's tests only exercise: new_adult_age=1, has_sex_chromosomes=false,
// undeclared equilibrium, growth modes 0-3. These cover the branches left
// untested: declared equilibrium, positive external eggs, mode 4 (ricker),
// non-trivial new_adult_age, sex-chromosome sex assignment, and n_ztypes=1.
// ---------------------------------------------------------------------------

/// Blueprint with an explicit adult start age and optional sex chromosomes.
///
/// ## Parameters
/// - `n_ages`, `n_ztypes`, `new_adult_age`: Model dimensions.
/// - `sex_chrom`: Whether sex is assigned by chromosome masks.
///
/// ## Returns
/// A blueprint whose adult range is `new_adult_age..n_ages`.
fn evaluator_blueprint(
    n_ages: usize,
    n_ztypes: usize,
    new_adult_age: usize,
    sex_chrom: bool,
) -> Blueprint {
    let mut female_only = vec![false; n_ztypes];
    let mut male_only = vec![false; n_ztypes];
    if sex_chrom && n_ztypes >= 3 {
        female_only[0] = true;
        male_only[1] = true;
    }
    Blueprint {
        n_sexes: 2,
        n_ages,
        n_ztypes,
        n_gtypes: n_ztypes,
        n_glabs: 1,
        new_adult_age,
        adult_ages: (new_adult_age as i64..n_ages as i64).collect(),
        stochastic: false,
        continuous_sampling: false,
        fixed_egg_count: false,
        has_sex_chromosomes: sex_chrom,
        extreme_speed_mode: 0,
        ztype_names: (0..n_ztypes).map(|z| format!("z{z}")).collect(),
        gtype_names: (0..n_ztypes).map(|z| format!("g{z}")).collect(),
        female_only_by_sex_chrom: female_only,
        male_only_by_sex_chrom: male_only,
        initial_individual_count: vec![],
        initial_sperm_storage: vec![],
        n_demes: 1,
        migration_indptr: vec![0, 0],
        migration_dest_idx: vec![],
        migration_weights: vec![],
    }
}

/// Single-deme ecology for arbitrary `n_ages`/`new_adult_age`.
///
/// ## Parameters
/// - `n_ages`, `new_adult_age`: Model dimensions.
/// - `mode`: Density growth mode.
/// - `declared`: Whether to declare an equilibrium distribution.
/// - `external`: `external_expected_eggs` value (negative = unused).
///
/// ## Returns
/// Ecology columns sized for one deme.
fn evaluator_ecology(
    n_ages: usize,
    new_adult_age: usize,
    mode: i64,
    declared: bool,
    external: f64,
) -> EcologyParams {
    let mut survival = Vec::with_capacity(2 * n_ages);
    for sex in 0..2 {
        for age in 0..n_ages {
            survival.push(0.95 - 0.05 * age as f64 - 0.02 * sex as f64);
        }
    }
    let mut mating = Vec::with_capacity(2 * n_ages);
    for _sex in 0..2 {
        for age in 0..n_ages {
            mating.push(if age >= new_adult_age { 0.9 } else { 0.0 });
        }
    }
    let reproduction: Vec<f64> = (0..n_ages)
        .map(|age| if age >= new_adult_age { 0.8 } else { 0.0 })
        .collect();
    let fertility: Vec<f64> = (0..n_ages)
        .map(|age| if age >= new_adult_age { 0.9 } else { 0.0 })
        .collect();
    let competition: Vec<f64> = (0..n_ages).map(|age| 1.0 - 0.1 * age as f64).collect();
    let distribution: Vec<f64> = if declared {
        (0..2 * n_ages).map(|i| 1.0 + 0.5 * i as f64).collect()
    } else {
        vec![]
    };
    EcologyParams {
        n_demes: 1,
        carrying_capacity: vec![400.0],
        eggs_per_female: vec![30.0],
        sex_ratio: vec![0.5],
        sperm_displacement_rate: vec![0.1],
        low_density_growth_rate: vec![2.0],
        growth_mode: vec![mode],
        external_expected_eggs: vec![external],
        survival_rates: survival,
        mating_rates: mating,
        reproduction_rates: reproduction,
        fertility,
        competition_weights: competition,
        equilibrium_distribution: distribution,
        equilibrium_declared: vec![declared],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new()],
    }
}

/// Assert one relative-error contract with a small absolute floor.
///
/// ## Parameters
/// - `label`: Message prefix.
/// - `got`, `want`: Device and host values.
/// - `relative`: Maximum allowed relative error.
fn assert_relative(label: &str, got: f32, want: f32, relative: f32) {
    let tolerance = relative * want.abs().max(1.0);
    assert!(
        (got - want).abs() <= tolerance,
        "{label}: device {got} vs host {want} (|diff|={}, tol={tolerance})",
        (got - want).abs()
    );
}

/// Run the density-scaling kernel and compare one scaling factor to the host.
///
/// ## Parameters
/// - `mode`: Growth mode.
/// - `declared`: Whether the equilibrium is declared.
/// - `external`: `external_expected_eggs` value.
fn evaluator_density_case(mode: i64, declared: bool, external: f64) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 1;
    let n_ages = 4;
    let n_ztypes = 2;
    let blueprint = evaluator_blueprint(n_ages, n_ztypes, 1, false);
    let ecology = evaluator_ecology(n_ages, 1, mode, declared, external);
    let ind: Vec<f32> = (0..n_batch * 2 * n_ages * n_ztypes)
        .map(|i| ((i * 11) % 83) as f32)
        .collect();
    let sperm = vec![0.0f32; n_batch * n_ages * n_ztypes * n_ztypes];

    let context = GpuContext::new(0).expect("device 0 context");
    let executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    let device = executor
        .density_scaling(&blueprint, &ecology)
        .expect("density launch");
    let local: Vec<f64> = ind.iter().map(|value| f64::from(*value)).collect();
    let expected = cpu_scaling(&blueprint, &ecology, 0, &local) as f32;
    assert_relative(
        &format!("density mode={mode} declared={declared} external={external}"),
        device[0],
        expected,
        2e-6,
    );
}

#[test]
fn evaluator_density_declared_equilibrium_matches_host() {
    for mode in 0..=4 {
        evaluator_density_case(mode, true, -1.0);
    }
}

#[test]
fn evaluator_density_ricker_and_external_eggs_match_host() {
    evaluator_density_case(4, false, -1.0);
    evaluator_density_case(2, false, 123.0);
    evaluator_density_case(3, false, 123.0);
    evaluator_density_case(4, false, 250.0);
    evaluator_density_case(4, true, 250.0);
}

/// Full-tick cross-check for a single panmictic deme over arbitrary shape.
///
/// ## Parameters
/// - `n_ages`, `n_ztypes`, `new_adult_age`: Model dimensions.
/// - `mode`, `declared`, `external`: Ecology options.
/// - `sex_chrom`: Whether sex is assigned by chromosome masks.
fn evaluator_full_tick_case(
    n_ages: usize,
    n_ztypes: usize,
    new_adult_age: usize,
    mode: i64,
    declared: bool,
    external: f64,
    sex_chrom: bool,
) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 1;
    let blueprint = evaluator_blueprint(n_ages, n_ztypes, new_adult_age, sex_chrom);
    let ecology = evaluator_ecology(n_ages, new_adult_age, mode, declared, external);
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let mut genetics = reproduction_genetics(n_ages, n_ztypes);
    // Non-unit, sex-specific viability on the class the survival kernel targets.
    let target = new_adult_age - 1;
    for sex in 0..2 {
        for z in 0..n_ztypes {
            genetics.viability_fitness[(sex * n_ages + target) * n_ztypes + z] =
                0.5 + 0.1 * (sex + z) as f64;
        }
    }
    if sex_chrom {
        genetics.female_ztype_compatibility = (0..n_ztypes).map(|z| 0.7 - 0.1 * z as f64).collect();
        genetics.male_ztype_compatibility = (0..n_ztypes).map(|z| 0.2 + 0.1 * z as f64).collect();
    }
    let variants = vec![genetics.clone()];

    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    {
        let mut rng = new_rng(99);
        reproduction(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            0,
            &mut ind_ref,
            &mut sperm_ref,
        )
        .expect("host reproduction");
        survival(
            &mut rng,
            &blueprint,
            &ecology,
            &genetics,
            0,
            &mut ind_ref,
            &mut sperm_ref,
        )
        .expect("host survival");
        aging(&blueprint, &mut ind_ref, &mut sperm_ref);
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .tick(&blueprint, &ecology, &variants, &[0usize])
        .expect("device tick");
    let ind_out = executor.download_ind().expect("download individuals");
    let sperm_out = executor.download_sperm().expect("download sperm");
    let label = format!(
        "tick A={n_ages} Z={n_ztypes} adult={new_adult_age} mode={mode} \
         declared={declared} external={external} sex_chrom={sex_chrom}"
    );
    for (index, (got, want)) in ind_out.iter().zip(ind_ref.iter()).enumerate() {
        assert_relative(&format!("{label} ind[{index}]"), *got, *want as f32, 1.2e-6);
    }
    for (index, (got, want)) in sperm_out.iter().zip(sperm_ref.iter()).enumerate() {
        assert_relative(
            &format!("{label} sperm[{index}]"),
            *got,
            *want as f32,
            1.2e-6,
        );
    }
}

#[test]
fn evaluator_full_tick_new_adult_age_and_declared_host_match() {
    evaluator_full_tick_case(8, 2, 3, 2, false, -1.0, false);
    evaluator_full_tick_case(8, 2, 3, 2, true, -1.0, false);
    evaluator_full_tick_case(8, 3, 2, 3, true, 321.0, false);
    evaluator_full_tick_case(4, 2, 1, 4, false, -1.0, false);
}

#[test]
fn evaluator_full_tick_sex_chromosomes_host_match() {
    evaluator_full_tick_case(4, 3, 1, 2, false, -1.0, true);
    evaluator_full_tick_case(5, 4, 2, 3, true, -1.0, true);
}

#[test]
fn evaluator_full_tick_single_ztype_host_match() {
    evaluator_full_tick_case(4, 1, 1, 2, false, -1.0, false);
    evaluator_full_tick_case(4, 1, 1, 4, true, 10.0, false);
}

/// Construct a one-batch executor for `(n_ages, n_ztypes)`.
///
/// ## Parameters
/// - `n_ages`, `n_ztypes`: Model dimensions.
///
/// ## Returns
/// A ready executor (device context included).
fn evaluator_executor(n_ages: usize, n_ztypes: usize) -> GpuExecutor {
    let ind = vec![3.0f32; 2 * n_ages * n_ztypes];
    let sperm = vec![1.0f32; n_ages * n_ztypes * n_ztypes];
    let context = GpuContext::new(0).expect("device 0 context");
    GpuExecutor::new(context, 1, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles")
}

#[test]
fn evaluator_max_dimensions_are_accepted() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // MAX_AGES = 64 is inclusive: density scaling must run.
    let n_ages = 64;
    let blueprint = evaluator_blueprint(n_ages, 2, 1, false);
    let ecology = evaluator_ecology(n_ages, 1, 2, false, -1.0);
    let executor = evaluator_executor(n_ages, 2);
    let scaling = executor
        .density_scaling(&blueprint, &ecology)
        .expect("density at MAX_AGES");
    assert!(scaling[0].is_finite(), "scaling must be finite");

    // MAX_Z = 32 is inclusive: reproduction must run.
    let (blueprint, ecology) = (
        evaluator_blueprint(4, 32, 1, false),
        evaluator_ecology(4, 1, 2, false, -1.0),
    );
    let genetics = reproduction_genetics(4, 32);
    let variants = [genetics];
    let mut executor = evaluator_executor(4, 32);
    executor
        .reproduction_tick(&blueprint, &ecology, &variants, &[0usize])
        .expect("reproduction at MAX_Z");
}

/// Regression: CPU `reproduction` clears age-0 newborns when the stage
/// produces no recruits (`!has_any`), but the device kernel returns from its
/// `has_any` / `total` early-exit without touching age 0, preserving whatever
/// the caller seeded there. On tick 0 (or a direct `reproduction_tick` call)
/// the state then diverges.
#[test]
fn evaluator_reproduction_clears_newborns_when_no_recruits() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 1;
    let n_ages = 4;
    let n_ztypes = 1;
    let blueprint = evaluator_blueprint(n_ages, n_ztypes, 1, false);
    // Male adults mate (effective males > 0) but females never do, so no sperm
    // is seeded and `fertilize` produces no recruits at all.
    let ecology = EcologyParams {
        n_demes: 1,
        carrying_capacity: vec![400.0],
        eggs_per_female: vec![30.0],
        sex_ratio: vec![0.5],
        sperm_displacement_rate: vec![0.1],
        low_density_growth_rate: vec![2.0],
        growth_mode: vec![1],
        external_expected_eggs: vec![-1.0],
        survival_rates: vec![0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55],
        mating_rates: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9],
        reproduction_rates: vec![0.0, 0.8, 0.7, 0.6],
        fertility: vec![0.0, 1.0, 0.9, 0.8],
        competition_weights: vec![1.0, 0.8, 0.7, 0.6],
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new()],
    };
    // Nonzero individuals at every age plus empty stored sperm.
    let (ind, _) = populated_state(n_batch, n_ages, n_ztypes);
    let sperm = vec![0.0f32; n_batch * n_ages * n_ztypes * n_ztypes];
    let genetics = reproduction_genetics(n_ages, n_ztypes);
    let variants = [genetics.clone()];

    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    let mut rng = new_rng(3);
    reproduction(
        &mut rng,
        &blueprint,
        &ecology,
        &genetics,
        0,
        &mut ind_ref,
        &mut sperm_ref,
    )
    .expect("host reproduction");

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .reproduction_tick(&blueprint, &ecology, &variants, &[0usize])
        .expect("device reproduction");
    let ind_out = executor.download_ind().expect("download individuals");

    for (index, (got, want)) in ind_out.iter().zip(ind_ref.iter()).enumerate() {
        assert_relative(
            &format!("no-recruit reproduction ind[{index}]"),
            *got,
            *want as f32,
            1.2e-6,
        );
    }
}

/// Regression: CPU `compute_mating_probability_matrix` zeroes a female mating
/// row whose weighted male sum is `<= EPS` (1e-10, `rng.rs`), but the device
/// kernel uses a hardcoded `1e-12f` threshold. A row sum in `(1e-12, 1e-10]`
/// is therefore normalized on the GPU and suppressed on the CPU, so the GPU
/// produces offspring where the reference produces none.
#[test]
fn evaluator_mating_row_epsilon_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 1;
    let n_ages = 4;
    let n_ztypes = 1;
    let blueprint = evaluator_blueprint(n_ages, n_ztypes, 1, false);
    let mut ecology = evaluator_ecology(n_ages, 1, 2, false, -1.0);
    // Both sexes mate, but the selection table is ~0, so the row sum lands in
    // the divergence window.
    ecology.mating_rates = vec![0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9];
    let (mut ind, _) = populated_state(n_batch, n_ages, n_ztypes);
    for z in 0..n_ztypes {
        ind[(0 * n_ages + 0) * n_ztypes + z] = 0.0;
        ind[(1 * n_ages + 0) * n_ztypes + z] = 0.0;
    }
    let sperm = vec![0.0f32; n_batch * n_ages * n_ztypes * n_ztypes];
    let mut genetics = reproduction_genetics(n_ages, n_ztypes);
    genetics.sexual_selection_fitness = vec![1e-12];
    let variants = [genetics.clone()];

    let mut ind_ref: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let mut sperm_ref: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    let mut rng = new_rng(3);
    reproduction(
        &mut rng,
        &blueprint,
        &ecology,
        &genetics,
        0,
        &mut ind_ref,
        &mut sperm_ref,
    )
    .expect("host reproduction");

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .reproduction_tick(&blueprint, &ecology, &variants, &[0usize])
        .expect("device reproduction");
    let ind_out = executor.download_ind().expect("download individuals");
    for (index, (got, want)) in ind_out.iter().zip(ind_ref.iter()).enumerate() {
        assert_relative(&format!("epsilon ind[{index}]"), *got, *want as f32, 1.2e-6);
    }
}

#[test]
fn evaluator_over_max_dimensions_are_rejected() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_ages = 65;
    let blueprint = evaluator_blueprint(n_ages, 2, 1, false);
    let ecology = evaluator_ecology(n_ages, 1, 2, false, -1.0);
    let executor = evaluator_executor(n_ages, 2);
    assert!(
        executor.density_scaling(&blueprint, &ecology).is_err(),
        "density must reject n_ages > MAX_AGES"
    );

    let blueprint = evaluator_blueprint(4, 33, 1, false);
    let ecology = evaluator_ecology(4, 1, 2, false, -1.0);
    let genetics = reproduction_genetics(4, 33);
    let variants = [genetics];
    let mut executor = evaluator_executor(4, 33);
    assert!(
        executor
            .reproduction_tick(&blueprint, &ecology, &variants, &[0usize])
            .is_err(),
        "reproduction must reject n_ztypes > MAX_Z"
    );
}

#[test]
fn executor_rejects_malformed_inputs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    assert_eq!(executor.n_batch(), n_batch);
    let _ = executor.context().compute_capability();

    // Blueprint dimensions must agree with the executor.
    let mut bad_blueprint = blueprint.clone();
    bad_blueprint.n_ages = 5;
    assert!(executor.density_scaling(&bad_blueprint, &ecology).is_err());
    let genetics = reproduction_genetics(n_ages, n_ztypes);
    assert!(executor
        .survival_tick(
            &bad_blueprint,
            &ecology,
            &[genetics.clone()],
            &vec![0usize; n_batch]
        )
        .is_err());
    assert!(executor
        .reproduction_tick(
            &bad_blueprint,
            &ecology,
            &[genetics.clone()],
            &vec![0usize; n_batch]
        )
        .is_err());

    // Ecology column counts must match the batch.
    let mut bad_ecology = ecology.clone();
    bad_ecology.n_demes = 3;
    assert!(executor.density_scaling(&blueprint, &bad_ecology).is_err());

    // Every column length is validated (survival shown; the kernel indexes by b).
    let mut short_ecology = ecology.clone();
    short_ecology.survival_rates.pop();
    assert!(executor
        .density_scaling(&blueprint, &short_ecology)
        .is_err());
    let mut short_eggs = ecology.clone();
    short_eggs.eggs_per_female.clear();
    assert!(executor.density_scaling(&blueprint, &short_eggs).is_err());

    // Growth modes outside 0..=4 are rejected.
    let mut custom = ecology.clone();
    custom.growth_mode[0] = 5;
    assert!(executor.density_scaling(&blueprint, &custom).is_err());
    let mut negative = ecology.clone();
    negative.growth_mode[0] = -1;
    assert!(executor.density_scaling(&blueprint, &negative).is_err());

    // Per-batch variant ids must cover the batch.
    assert!(executor
        .survival_tick(&blueprint, &ecology, &[genetics.clone()], &[0usize])
        .is_err());
    assert!(executor
        .reproduction_tick(&blueprint, &ecology, &[genetics.clone()], &[0usize])
        .is_err());

    // Reproduction validates its ecology and genetics tables.
    let mut short_repro = ecology.clone();
    short_repro.mating_rates.pop();
    assert!(executor
        .reproduction_tick(
            &blueprint,
            &short_repro,
            &[genetics.clone()],
            &vec![0usize; n_batch]
        )
        .is_err());
    let mut bad_genetics = genetics.clone();
    bad_genetics.offspring_tensor.clear();
    assert!(executor
        .reproduction_tick(
            &blueprint,
            &ecology,
            &[bad_genetics],
            &vec![0usize; n_batch]
        )
        .is_err());

    // Sex-chromosome flags must be sized to the zygote-type axis.
    let mut bad_flags = blueprint.clone();
    bad_flags.female_only_by_sex_chrom.clear();
    assert!(executor
        .reproduction_tick(
            &bad_flags,
            &ecology,
            &[genetics.clone()],
            &vec![0usize; n_batch]
        )
        .is_err());
}

#[test]
fn executor_rejects_mismatched_state_lengths() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    assert!(GpuExecutor::new(context, 1, 4, 2, &[0.0f32; 3], &[0.0f32; 8]).is_err());
    let context = GpuContext::new(0).expect("device 0 context");
    assert!(GpuExecutor::new(context, 1, 4, 2, &[0.0f32; 16], &[0.0f32; 3]).is_err());
}

#[test]
fn memory_budget_helper_reports_over_budget() {
    assert!(super::ensure_memory_budget(10, 20).is_err());
    assert!(super::ensure_memory_budget(20, 10).is_ok());
    assert!(super::ensure_memory_budget(0, 0).is_ok());
}

#[test]
fn executor_rejects_bad_variants() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    let genetics = reproduction_genetics(n_ages, n_ztypes);
    let ids = vec![0usize; n_batch];

    // Out-of-range variant id.
    assert!(executor
        .survival_tick(
            &blueprint,
            &ecology,
            &[genetics.clone()],
            &vec![7usize; n_batch]
        )
        .is_err());
    assert!(executor
        .reproduction_tick(
            &blueprint,
            &ecology,
            &[genetics.clone()],
            &vec![7usize; n_batch]
        )
        .is_err());

    // Malformed variant tables are rejected by the copy helper.
    let mut short_viability = genetics.clone();
    short_viability.viability_fitness.truncate(3);
    assert!(executor
        .survival_tick(&blueprint, &ecology, &[short_viability], &ids)
        .is_err());
    let mut short_fecundity = genetics.clone();
    short_fecundity.fecundity_fitness.truncate(1);
    assert!(executor
        .reproduction_tick(&blueprint, &ecology, &[short_fecundity], &ids)
        .is_err());
}

/// Blueprint carrying a small non-trivial CSR (4 demes, one empty row).
///
/// ## Returns
/// A blueprint with destinations `0->{1,2}, 1->{0}, 2->{}, 3->{2}`.
fn migration_blueprint() -> Blueprint {
    let mut blueprint = dimension_blueprint(4, 2);
    blueprint.n_demes = 4;
    blueprint.migration_indptr = vec![0, 2, 3, 3, 4];
    blueprint.migration_dest_idx = vec![1, 2, 0, 2];
    blueprint.migration_weights = vec![0.5, 0.5, 1.0, 1.0];
    blueprint
}

#[test]
fn device_migration_matches_the_host_reference() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (_, mut ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let rate: Vec<f64> = (0..n_batch * 2 * n_ages)
        .map(|i| 0.1 + 0.02 * (i % 7) as f64)
        .collect();
    ecology.migration_rate = rate.clone();
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let indptr = [0i64, 2, 3, 3, 4];
    let dest = [1i64, 2, 0, 2];
    let weights = [0.5f64, 0.5, 1.0, 1.0];
    let blueprint = migration_blueprint();
    let ind_f64: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let sperm_f64: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();

    for stay_after in [false, true] {
        let (cpu_ind, cpu_sperm) = migrate_csr_deterministic(
            &ind_f64, &sperm_f64, &indptr, &dest, &weights, &rate, stay_after, n_batch, n_ages,
            n_ztypes,
        )
        .expect("host migration");
        let context = GpuContext::new(0).expect("device 0 context");
        let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
            .expect("executor uploads and compiles");
        executor
            .migrate_tick(&blueprint, &ecology, stay_after)
            .expect("device migration");
        let gpu_ind = executor.download_ind().expect("download individuals");
        let gpu_sperm = executor.download_sperm().expect("download sperm");
        for (label, got, want) in [
            ("ind", &gpu_ind, &cpu_ind),
            ("sperm", &gpu_sperm, &cpu_sperm),
        ] {
            for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                let want = *want as f32;
                let tolerance = 1.2e-6f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "stay_after={stay_after} {label}[{index}]: device {got} vs host {want}"
                );
            }
        }
    }
}

#[test]
fn migration_tick_validates_inputs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (_, mut ecology) = density_fixture();
    ecology.migration_rate = vec![0.1f64; 4 * 2 * 4];
    let (ind, sperm) = populated_state(4, 4, 2);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, 4, 4, 2, &ind, &sperm).expect("executor uploads and compiles");

    // Wrong CSR row-pointer count.
    let mut bad_indptr = migration_blueprint();
    bad_indptr.migration_indptr = vec![0, 2];
    assert!(executor.migrate_tick(&bad_indptr, &ecology, false).is_err());

    // An empty migration column is a no-op.
    let mut empty = ecology.clone();
    empty.migration_rate = vec![];
    assert!(executor
        .migrate_tick(&migration_blueprint(), &empty, false)
        .is_ok());

    // A destination outside the batch is rejected.
    let mut out_of_range = migration_blueprint();
    out_of_range.migration_dest_idx = vec![1, 9, 0, 2];
    assert!(executor
        .migrate_tick(&out_of_range, &ecology, false)
        .is_err());
}

/// Compare a device migration result against a host reference with f32
/// tolerance.
///
/// ## Parameters
/// - `label`: Prefix for assertion messages.
/// - `got`, `want`: Device and host values.
fn assert_migration_close(label: &str, got: &[f32], want: &[f64]) {
    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        let want = *want as f32;
        let tolerance = 1.2e-6f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "{label}[{index}]: device {got} vs host {want}"
        );
    }
}

#[test]
fn migration_cache_reuses_buffers_and_invalidates_on_new_csr() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (_, mut ecology) = density_fixture();
    let n_batch = 4;
    let n_ages = 4;
    let n_ztypes = 2;
    let rate_a: Vec<f64> = (0..n_batch * 2 * n_ages)
        .map(|i| 0.1 + 0.02 * (i % 7) as f64)
        .collect();
    let rate_b: Vec<f64> = (0..n_batch * 2 * n_ages)
        .map(|i| 0.05 + 0.03 * (i % 5) as f64)
        .collect();
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let indptr = [0i64, 2, 3, 3, 4];
    let dest = [1i64, 2, 0, 2];
    let weights = [0.5f64, 0.5, 1.0, 1.0];
    let blueprint = migration_blueprint();
    let ind_f64: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let sperm_f64: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");

    // Two ticks on the same CSR: the static buffers are built once and reused
    // while the changing rate column still drives each tick.
    ecology.migration_rate = rate_a.clone();
    executor
        .migrate_tick(&blueprint, &ecology, false)
        .expect("first device migration");
    let first_fingerprint = executor
        .migration_cache
        .as_ref()
        .expect("cache after first tick")
        .fingerprint;
    ecology.migration_rate = rate_b.clone();
    executor
        .migrate_tick(&blueprint, &ecology, false)
        .expect("second device migration");
    let second_fingerprint = executor
        .migration_cache
        .as_ref()
        .expect("cache after second tick")
        .fingerprint;
    assert_eq!(
        first_fingerprint, second_fingerprint,
        "the same CSR must reuse the cached buffers"
    );

    let (cpu_one_ind, cpu_one_sperm) = migrate_csr_deterministic(
        &ind_f64, &sperm_f64, &indptr, &dest, &weights, &rate_a, false, n_batch, n_ages, n_ztypes,
    )
    .expect("host first migration");
    let (cpu_two_ind, cpu_two_sperm) = migrate_csr_deterministic(
        &cpu_one_ind,
        &cpu_one_sperm,
        &indptr,
        &dest,
        &weights,
        &rate_b,
        false,
        n_batch,
        n_ages,
        n_ztypes,
    )
    .expect("host second migration");
    assert_migration_close(
        "cache-reuse ind",
        &executor.download_ind().expect("download ind"),
        &cpu_two_ind,
    );
    assert_migration_close(
        "cache-reuse sperm",
        &executor.download_sperm().expect("download sperm"),
        &cpu_two_sperm,
    );

    // A different CSR must invalidate the cached plan and match the host.
    let mut other = migration_blueprint();
    let other_dest = [2i64, 0, 1, 0];
    let other_weights = [0.25f64, 0.75, 0.5, 0.5];
    other.migration_dest_idx = other_dest.to_vec();
    other.migration_weights = other_weights.to_vec();
    executor
        .migrate_tick(&other, &ecology, false)
        .expect("third device migration with a new CSR");
    let third_fingerprint = executor
        .migration_cache
        .as_ref()
        .expect("cache after third tick")
        .fingerprint;
    assert_ne!(
        second_fingerprint, third_fingerprint,
        "a different CSR must rebuild the cache"
    );
    let (cpu_three_ind, cpu_three_sperm) = migrate_csr_deterministic(
        &cpu_two_ind,
        &cpu_two_sperm,
        &indptr,
        &other_dest,
        &other_weights,
        &rate_b,
        false,
        n_batch,
        n_ages,
        n_ztypes,
    )
    .expect("host third migration");
    assert_migration_close(
        "invalidate ind",
        &executor.download_ind().expect("download ind"),
        &cpu_three_ind,
    );
    assert_migration_close(
        "invalidate sperm",
        &executor.download_sperm().expect("download sperm"),
        &cpu_three_sperm,
    );
}

#[test]
fn stochastic_migration_reuses_cached_scratch_reproducibly() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n0 = 4;
    let n_ages = 3;
    let z = 2;
    let copies = 64;
    let (blueprint, ecology) = evaluator_block_diagonal(n0, n_ages, z, copies);
    let n_batch = blueprint.n_demes;
    let ind: Vec<f32> = (0..n_batch * 2 * n_ages * z)
        .map(|i| ((i * 7) % 41) as f32 + 1.0)
        .collect();
    let sperm: Vec<f32> = (0..n_batch * n_ages * z * z)
        .map(|i| ((i * 5) % 13) as f32)
        .collect();

    let run_twice = |ind: &[f32], sperm: &[f32]| -> (Vec<f32>, Vec<f32>) {
        let context = GpuContext::new(0).expect("device 0 context");
        let mut executor = GpuExecutor::new(context, n_batch, n_ages, z, ind, sperm)
            .expect("executor uploads and compiles");
        executor.set_seed(0xFEED);
        executor
            .migrate_tick_stochastic(&blueprint, &ecology)
            .expect("first stochastic migration");
        executor
            .migrate_tick_stochastic(&blueprint, &ecology)
            .expect("second stochastic migration");
        (
            executor.download_ind().expect("download ind"),
            executor.download_sperm().expect("download sperm"),
        )
    };

    let (a_ind, a_sperm) = run_twice(&ind, &sperm);
    let (b_ind, b_sperm) = run_twice(&ind, &sperm);
    assert_eq!(
        a_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "reusing the stochastic scratch must stay reproducible"
    );
    assert_eq!(
        a_sperm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b_sperm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "reusing the stochastic scratch must stay reproducible"
    );
}

#[test]
fn stochastic_migration_validates_inputs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (_, mut ecology) = density_fixture();
    ecology.migration_rate = vec![0.1f64; 4 * 2 * 4];
    let (ind, sperm) = populated_state(4, 4, 2);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, 4, 4, 2, &ind, &sperm).expect("executor uploads and compiles");

    // Missing CSR row pointers are rejected before any launch.
    let mut bad_indptr = migration_blueprint();
    bad_indptr.migration_indptr = vec![0, 2];
    assert!(executor
        .migrate_tick_stochastic(&bad_indptr, &ecology)
        .is_err());

    // An empty migration column is a no-op.
    let mut empty = ecology.clone();
    empty.migration_rate = vec![];
    assert!(executor
        .migrate_tick_stochastic(&migration_blueprint(), &empty)
        .is_ok());
}

#[test]
fn ensemble_rejects_mismatched_replicate_state() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_ages = 4;
    let n_ztypes = 2;
    let ind_one = vec![1.0f32; 2 * n_ages * n_ztypes];
    let sperm_one = vec![0.0f32; n_ages * n_ztypes * n_ztypes];

    // Wrong individual length per replicate.
    let context = GpuContext::new(0).expect("device 0 context");
    assert!(GpuExecutor::ensemble(
        context,
        3,
        n_ages,
        n_ztypes,
        &ind_one[..ind_one.len() - 1],
        &sperm_one,
        1,
    )
    .is_err());

    // Wrong sperm length per replicate.
    let context = GpuContext::new(0).expect("device 0 context");
    assert!(GpuExecutor::ensemble(
        context,
        3,
        n_ages,
        n_ztypes,
        &ind_one,
        &sperm_one[..sperm_one.len() - 1],
        1,
    )
    .is_err());
}

#[test]
fn device_history_projection_matches_host_project() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 3;
    let n_ages = 4;
    let n_ztypes = 2;
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let ind_f64: Vec<f64> = ind.iter().map(|value| f64::from(*value)).collect();
    let dims = [n_batch, 2, n_ages, n_ztypes];
    let plane = 2 * n_ages * n_ztypes;
    let mask_one: Vec<f64> = (0..plane)
        .map(|index| if index % 3 == 0 { 1.0 } else { 0.5 })
        .collect();
    let mask_two: Vec<f64> = (0..2 * plane)
        .map(|index| if index < plane { 1.0 } else { 0.25 })
        .collect();
    let selected = vec![2usize, 0];

    for (mask, selected) in [(&mask_one, vec![0usize, 2]), (&mask_two, selected)] {
        for (collapse, aggregate) in [(false, false), (true, false), (false, true), (true, true)] {
            let groups = mask.len() / plane;
            let out_d = if aggregate { 1 } else { selected.len() };
            let out_a = if collapse { 1 } else { n_ages };
            let width = groups * out_d * 2 * out_a;
            let spec = HistorySpec {
                capacity: 2,
                width,
                dims,
                mask: mask.clone(),
                selected: selected.clone(),
                collapse,
                aggregate,
            };
            let context = GpuContext::new(0).expect("device 0 context");
            let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
                .expect("executor uploads and compiles");
            executor
                .configure_history(&spec)
                .expect("configure history");
            executor.record_history_row().expect("record row");
            let got = executor.download_history_rows().expect("download rows");
            let want = project(&ind_f64, mask, dims, &selected, collapse, aggregate)
                .expect("host projection");
            assert_eq!(got.len(), want.len(), "width mismatch");
            for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                let want = *want as f32;
                let tolerance = 1.2e-6f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "groups={groups} collapse={collapse} aggregate={aggregate} cell[{index}]: \
                     device {got} vs host {want}"
                );
            }
        }
    }
}

#[test]
fn device_history_rejects_over_budget_and_empty_mask() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 3;
    let n_ages = 4;
    let n_ztypes = 2;
    let (ind, sperm) = populated_state(n_batch, n_ages, n_ztypes);
    let dims = [n_batch, 2, n_ages, n_ztypes];
    let plane = 2 * n_ages * n_ztypes;
    let selected = vec![0usize, 1, 2];
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");

    // Recording or downloading without a window is an explicit error / empty.
    assert!(executor.record_history_row().is_err());
    assert!(executor.history_width().is_none());
    assert!(executor
        .download_history_rows()
        .expect("empty download")
        .is_empty());

    // An observation-mode window larger than free memory is refused, which the
    // session treats as "stage on the host instead".
    let huge = HistorySpec {
        capacity: 1usize << 40,
        width: 8,
        dims,
        mask: vec![1.0; plane],
        selected: selected.clone(),
        collapse: false,
        aggregate: true,
    };
    assert!(executor.configure_history(&huge).is_err());

    // A store without a configured observation mask cannot be projected.
    let no_mask = HistorySpec {
        capacity: 2,
        width: 2,
        dims,
        mask: Vec::new(),
        selected: selected.clone(),
        collapse: false,
        aggregate: true,
    };
    assert!(executor.configure_history(&no_mask).is_err());

    // Dimensions, selection, and width are validated against the executor.
    let bad_dims = HistorySpec {
        capacity: 2,
        width: 2,
        dims: [2, 2, n_ages, n_ztypes],
        mask: vec![1.0; plane],
        selected: selected.clone(),
        collapse: false,
        aggregate: true,
    };
    assert!(executor.configure_history(&bad_dims).is_err());
    let bad_selected = HistorySpec {
        capacity: 2,
        width: 2,
        dims,
        mask: vec![1.0; plane],
        selected: vec![0, 9],
        collapse: false,
        aggregate: true,
    };
    assert!(executor.configure_history(&bad_selected).is_err());
    let bad_width = HistorySpec {
        capacity: 2,
        width: 999,
        dims,
        mask: vec![1.0; plane],
        selected: selected.clone(),
        collapse: false,
        aggregate: true,
    };
    assert!(executor.configure_history(&bad_width).is_err());
    let overflow = HistorySpec {
        capacity: usize::MAX,
        width: 24,
        dims,
        mask: vec![1.0; plane],
        selected: selected.clone(),
        collapse: false,
        aggregate: false,
    };
    assert!(executor.configure_history(&overflow).is_err());

    // A full window refuses further rows, and clearing drops the window.
    let spec = HistorySpec {
        capacity: 1,
        width: 8,
        dims,
        mask: vec![1.0; plane],
        selected,
        collapse: false,
        aggregate: true,
    };
    executor.configure_history(&spec).expect("configure");
    executor.record_history_row().expect("first row");
    assert!(executor.record_history_row().is_err(), "window must fill");
    assert_eq!(executor.history_width(), Some(8));
    executor.clear_history();
    assert!(executor.record_history_row().is_err(), "cleared window");
}

// ---------------------------------------------------------------------------
// Evaluator adversarial migration topologies (P5).
// ---------------------------------------------------------------------------

/// Tiled ecology for a general `(n_demes, n_ages)` migration case.
///
/// ## Parameters
/// - `n_demes`, `n_ages`: Model dimensions.
/// - `rate`: `(n_demes, 2, n_ages)` migration column.
///
/// ## Returns
/// An ecology whose only exercised column is `migration_rate`.
fn evaluator_migration_ecology(n_demes: usize, n_ages: usize, rate: &[f64]) -> EcologyParams {
    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_demes).flat_map(|_| values.iter().copied()).collect()
    };
    let survival: Vec<f64> = (0..2 * n_ages).map(|i| 0.9 - 0.01 * i as f64).collect();
    let per_age: Vec<f64> = (0..n_ages).map(|i| 0.5 + 0.01 * i as f64).collect();
    EcologyParams {
        n_demes,
        carrying_capacity: vec![500.0; n_demes],
        eggs_per_female: vec![10.0; n_demes],
        sex_ratio: vec![0.5; n_demes],
        sperm_displacement_rate: vec![0.1; n_demes],
        low_density_growth_rate: vec![3.0; n_demes],
        growth_mode: vec![2; n_demes],
        external_expected_eggs: vec![-1.0; n_demes],
        survival_rates: tile(&survival),
        mating_rates: tile(&per_age),
        reproduction_rates: tile(&per_age),
        fertility: tile(&per_age),
        competition_weights: tile(&per_age),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_demes],
        migration_rate: rate.to_vec(),
        custom_slots: vec![HashMap::new(); n_demes],
    }
}

/// Compare one device migration against the host reference for an arbitrary CSR.
///
/// ## Parameters
/// - `indptr`, `dest`, `weights`: CSR routing table.
/// - `rate`: Migration column.
/// - `n_demes`, `n_ages`, `n_ztypes`: Model dimensions.
/// - `stay_after`: Migration bookkeeping mode.
/// - `relative`: Allowed relative error.
/// - `label`: Message prefix.
fn evaluator_migration_case(
    indptr: &[i64],
    dest: &[i64],
    weights: &[f64],
    rate: &[f64],
    n_demes: usize,
    n_ages: usize,
    n_ztypes: usize,
    stay_after: bool,
    relative: f32,
    label: &str,
) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let mut blueprint = dimension_blueprint(n_ages, n_ztypes);
    blueprint.n_demes = n_demes;
    blueprint.migration_indptr = indptr.to_vec();
    blueprint.migration_dest_idx = dest.to_vec();
    blueprint.migration_weights = weights.to_vec();
    let ecology = evaluator_migration_ecology(n_demes, n_ages, rate);
    let (ind, sperm) = populated_state(n_demes, n_ages, n_ztypes);
    let ind_f64: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let sperm_f64: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    let (cpu_ind, cpu_sperm) = migrate_csr_deterministic(
        &ind_f64, &sperm_f64, indptr, dest, weights, rate, stay_after, n_demes, n_ages, n_ztypes,
    )
    .expect("host migration");
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_demes, n_ages, n_ztypes, &ind, &sperm)
        .expect("executor uploads and compiles");
    executor
        .migrate_tick(&blueprint, &ecology, stay_after)
        .expect("device migration");
    let gpu_ind = executor.download_ind().expect("download individuals");
    let gpu_sperm = executor.download_sperm().expect("download sperm");
    for (name, got, want) in [
        ("ind", &gpu_ind, &cpu_ind),
        ("sperm", &gpu_sperm, &cpu_sperm),
    ] {
        for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
            assert_relative(
                &format!("{label} stay_after={stay_after} {name}[{index}]"),
                *got,
                *want as f32,
                relative,
            );
        }
    }
}

/// Build a `(n_demes, 2, n_ages)` rate column with non-trivial values.
///
/// ## Parameters
/// - `n_demes`, `n_ages`: Model dimensions.
///
/// ## Returns
/// A deterministic rate column.
fn evaluator_rate(n_demes: usize, n_ages: usize) -> Vec<f64> {
    (0..n_demes * 2 * n_ages)
        .map(|i| 0.05 + 0.01 * (i % 9) as f64)
        .collect()
}

#[test]
fn evaluator_migration_topologies_match_host_tightly() {
    for stay_after in [false, true] {
        let rate = evaluator_rate(3, 4);
        // Ring with weights that do not sum to one (exercises row_sum_w).
        evaluator_migration_case(
            &[0, 2, 4, 6],
            &[1, 2, 0, 2, 0, 1],
            &[0.3, 0.4, 0.25, 0.35, 0.45, 0.15],
            &rate,
            3,
            4,
            2,
            stay_after,
            1.2e-6,
            "ring-nonunit",
        );
        let rate = evaluator_rate(2, 4);
        // Self-loop on deme 0, isolated (empty-row) deme 1.
        evaluator_migration_case(
            &[0, 1, 1],
            &[0],
            &[0.5],
            &rate,
            2,
            4,
            2,
            stay_after,
            1.2e-6,
            "selfloop-empty",
        );
        // Duplicate edges to the same destination plus a self-loop.
        evaluator_migration_case(
            &[0, 3, 3],
            &[1, 1, 0],
            &[0.2, 0.3, 0.5],
            &rate,
            2,
            4,
            2,
            stay_after,
            1.2e-6,
            "duplicate-edges",
        );
        // Larger dims: 4-deme cycle with heterogeneous weights, A=8, Z=3.
        let rate = evaluator_rate(4, 8);
        evaluator_migration_case(
            &[0, 2, 4, 6, 8],
            &[1, 3, 0, 2, 1, 3, 0, 2],
            &[0.2, 0.3, 0.4, 0.1, 0.25, 0.25, 0.3, 0.35],
            &rate,
            4,
            8,
            3,
            stay_after,
            1.2e-6,
            "cycle4-a8-z3",
        );
    }
}

#[test]
fn evaluator_migration_zero_rate_is_host_equivalent() {
    // The session short-circuits all-zero rates before calling the kernel, but
    // the executor kernel itself must still match the host reference when run.
    for stay_after in [false, true] {
        evaluator_migration_case(
            &[0, 2, 4, 6],
            &[1, 2, 0, 2, 0, 1],
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            &vec![0.0; 3 * 2 * 4],
            3,
            4,
            2,
            stay_after,
            1.2e-6,
            "zero-rate",
        );
    }
}

#[test]
fn evaluator_migration_empty_csr_and_overmigration_match_host() {
    for stay_after in [false, true] {
        // Fully empty CSR: every deme isolated, nnz = 0.
        evaluator_migration_case(
            &[0, 0, 0],
            &[],
            &[],
            &evaluator_rate(2, 4),
            2,
            4,
            2,
            stay_after,
            1.2e-6,
            "nnz-zero",
        );
        // Over-migration: rate > 1 drives outbound above the source count.
        let mut rate = evaluator_rate(3, 4);
        for value in rate.iter_mut() {
            *value = 1.5;
        }
        evaluator_migration_case(
            &[0, 2, 4, 6],
            &[1, 2, 0, 2, 0, 1],
            &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            &rate,
            3,
            4,
            2,
            stay_after,
            1.2e-6,
            "over-migration",
        );
    }
}

/// Compare one device stochastic survival run against the host distribution.
///
/// ## Parameters
/// - `continuous`: Whether to exercise the continuous-sampling branches.
fn survival_distribution_case(continuous: bool) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 4000usize;
    let n_ages = 4usize;
    let z = 2usize;

    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = continuous;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];

    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_batch).flat_map(|_| values.iter().copied()).collect()
    };
    let survival = tile(&[0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55]);
    let ecology = EcologyParams {
        n_demes: n_batch,
        carrying_capacity: vec![1.0e9; n_batch],
        eggs_per_female: vec![1.0; n_batch],
        sex_ratio: vec![0.5; n_batch],
        sperm_displacement_rate: vec![0.1; n_batch],
        low_density_growth_rate: vec![1.0; n_batch],
        growth_mode: vec![0; n_batch],
        external_expected_eggs: vec![-1.0; n_batch],
        survival_rates: survival.clone(),
        mating_rates: tile(&[0.0; 8]),
        reproduction_rates: tile(&[0.0; 4]),
        fertility: tile(&[0.0; 4]),
        competition_weights: tile(&[1.0; 4]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_batch],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new(); n_batch],
    };
    let mut one_ecology = ecology.clone();
    one_ecology.n_demes = 1;
    one_ecology.carrying_capacity.truncate(1);
    for column in [
        &mut one_ecology.eggs_per_female,
        &mut one_ecology.sex_ratio,
        &mut one_ecology.sperm_displacement_rate,
        &mut one_ecology.low_density_growth_rate,
        &mut one_ecology.external_expected_eggs,
    ] {
        column.truncate(1);
    }
    one_ecology.growth_mode.truncate(1);
    one_ecology.equilibrium_declared.truncate(1);
    one_ecology.survival_rates.truncate(8);
    one_ecology.mating_rates.truncate(8);
    one_ecology.reproduction_rates.truncate(4);
    one_ecology.fertility.truncate(4);
    one_ecology.competition_weights.truncate(4);
    one_ecology.custom_slots.truncate(1);

    let mut genetics = reproduction_genetics(n_ages, z);
    for sex in 0..2 {
        for zz in 0..z {
            genetics.viability_fitness[(sex * n_ages) * z + zz] = 0.5 + 0.2 * (sex + zz) as f64;
        }
    }
    let variants = vec![genetics.clone()];

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let one_ind: Vec<f64> = vec![100.0; ind_stride];
    let one_sperm: Vec<f64> = (0..sperm_stride).map(|i| (i % 3) as f64).collect();
    let ind_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_ind.iter().copied())
        .map(|v| v as f32)
        .collect();
    let sperm_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_sperm.iter().copied())
        .map(|v| v as f32)
        .collect();

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind_host, &sperm_host).expect("executor");
    executor.set_seed(0x0123_4567_89AB_CDEF);
    executor
        .survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("device stochastic survival");
    let device_ind = executor.download_ind().expect("download");
    let device_sperm = executor.download_sperm().expect("download");

    // Host distribution: one independent CPU trial per batch element.
    let mut cpu_sum_ind = vec![0.0f64; ind_stride];
    let mut cpu_sq_ind = vec![0.0f64; ind_stride];
    let mut cpu_sum_sperm = vec![0.0f64; sperm_stride];
    let mut cpu_sq_sperm = vec![0.0f64; sperm_stride];
    for trial in 0..n_batch {
        let mut rng = new_rng(10_000 + trial as u64);
        let mut ind = one_ind.clone();
        let mut sperm = one_sperm.clone();
        crate::kernels::age_structured::survival(
            &mut rng,
            &blueprint,
            &one_ecology,
            &genetics,
            0,
            &mut ind,
            &mut sperm,
        )
        .expect("host survival");
        for (slot, value) in ind.iter().enumerate() {
            cpu_sum_ind[slot] += value;
            cpu_sq_ind[slot] += value * value;
        }
        for (slot, value) in sperm.iter().enumerate() {
            cpu_sum_sperm[slot] += value;
            cpu_sq_sperm[slot] += value * value;
        }
    }
    let trials = n_batch as f64;
    for slot in 0..ind_stride {
        let cpu_mean = cpu_sum_ind[slot] / trials;
        let cpu_var = (cpu_sq_ind[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_ind[b * ind_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "ind[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
    for slot in 0..sperm_stride {
        let cpu_mean = cpu_sum_sperm[slot] / trials;
        let cpu_var = (cpu_sq_sperm[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_sperm[b * sperm_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "sperm[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
}

#[test]
fn device_stochastic_survival_matches_host_distribution() {
    survival_distribution_case(false);
}

#[test]
fn device_stochastic_survival_continuous_matches_host_distribution() {
    survival_distribution_case(true);
}

/// Compare one device stochastic reproduction run against the host
/// distribution.
///
/// ## Parameters
/// - `continuous`: Whether to exercise the continuous-sampling branches.
fn reproduction_distribution_case(continuous: bool) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 4000usize;
    let n_ages = 4usize;
    let z = 2usize;

    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = continuous;
    blueprint.fixed_egg_count = false;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];

    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_batch).flat_map(|_| values.iter().copied()).collect()
    };
    let ecology = EcologyParams {
        n_demes: n_batch,
        carrying_capacity: vec![1.0e9; n_batch],
        eggs_per_female: vec![10.0; n_batch],
        sex_ratio: vec![0.5; n_batch],
        sperm_displacement_rate: vec![0.1; n_batch],
        low_density_growth_rate: vec![1.0; n_batch],
        growth_mode: vec![0; n_batch],
        external_expected_eggs: vec![-1.0; n_batch],
        survival_rates: tile(&[1.0; 8]),
        mating_rates: tile(&[0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9]),
        reproduction_rates: tile(&[0.0, 0.8, 0.7, 0.6]),
        fertility: tile(&[0.0, 1.0, 0.9, 0.8]),
        competition_weights: tile(&[1.0; 4]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_batch],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new(); n_batch],
    };
    let mut one_ecology = ecology.clone();
    one_ecology.n_demes = 1;
    one_ecology.carrying_capacity.truncate(1);
    one_ecology.eggs_per_female.truncate(1);
    one_ecology.sex_ratio.truncate(1);
    one_ecology.sperm_displacement_rate.truncate(1);
    one_ecology.low_density_growth_rate.truncate(1);
    one_ecology.growth_mode.truncate(1);
    one_ecology.external_expected_eggs.truncate(1);
    one_ecology.equilibrium_declared.truncate(1);
    one_ecology.survival_rates.truncate(8);
    one_ecology.mating_rates.truncate(8);
    one_ecology.reproduction_rates.truncate(4);
    one_ecology.fertility.truncate(4);
    one_ecology.competition_weights.truncate(4);
    one_ecology.custom_slots.truncate(1);

    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics.clone()];

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let mut one_ind = vec![0.0f64; ind_stride];
    for sex in 0..2 {
        for age in 0..n_ages {
            for zz in 0..z {
                one_ind[(sex * n_ages + age) * z + zz] = if age == 0 { 0.0 } else { 100.0 };
            }
        }
    }
    let mut one_sperm = vec![0.0f64; sperm_stride];
    for age in 1..n_ages {
        for gf in 0..z {
            for gm in 0..z {
                one_sperm[(age * z + gf) * z + gm] = 3.0;
            }
        }
    }
    let ind_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_ind.iter().copied())
        .map(|v| v as f32)
        .collect();
    let sperm_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_sperm.iter().copied())
        .map(|v| v as f32)
        .collect();

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind_host, &sperm_host).expect("executor");
    executor.set_seed(0xDEAD_BEEF_1234_5678);
    executor
        .reproduction_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("device stochastic reproduction");
    let device_ind = executor.download_ind().expect("download");
    let device_sperm = executor.download_sperm().expect("download");

    let mut cpu_sum_ind = vec![0.0f64; ind_stride];
    let mut cpu_sq_ind = vec![0.0f64; ind_stride];
    let mut cpu_sum_sperm = vec![0.0f64; sperm_stride];
    let mut cpu_sq_sperm = vec![0.0f64; sperm_stride];
    for trial in 0..n_batch {
        let mut rng = new_rng(20_000 + trial as u64);
        let mut ind = one_ind.clone();
        let mut sperm = one_sperm.clone();
        crate::kernels::age_structured::reproduction(
            &mut rng,
            &blueprint,
            &one_ecology,
            &genetics,
            0,
            &mut ind,
            &mut sperm,
        )
        .expect("host reproduction");
        for (slot, value) in ind.iter().enumerate() {
            cpu_sum_ind[slot] += value;
            cpu_sq_ind[slot] += value * value;
        }
        for (slot, value) in sperm.iter().enumerate() {
            cpu_sum_sperm[slot] += value;
            cpu_sq_sperm[slot] += value * value;
        }
    }
    let trials = n_batch as f64;
    for slot in 0..ind_stride {
        let cpu_mean = cpu_sum_ind[slot] / trials;
        let cpu_var = (cpu_sq_ind[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_ind[b * ind_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "ind[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
    for slot in 0..sperm_stride {
        let cpu_mean = cpu_sum_sperm[slot] / trials;
        let cpu_var = (cpu_sq_sperm[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_sperm[b * sperm_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "sperm[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
}

#[test]
fn device_stochastic_reproduction_matches_host_distribution() {
    reproduction_distribution_case(false);
}

#[test]
fn device_stochastic_reproduction_continuous_matches_host_distribution() {
    reproduction_distribution_case(true);
}

/// Compare one device stochastic migration run against the host distribution.
///
/// ## Parameters
/// - `continuous`: Whether to exercise the continuous-sampling branches.
fn migration_distribution_case(continuous: bool) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // Independent pairs: even deme -> its odd partner (out-degree 1, empty
    // partner row), so pairs do not interact and can be pooled statistically.
    let n_batch = 2000usize;
    let pairs = n_batch / 2;
    let n_ages = 4usize;
    let z = 2usize;

    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = continuous;
    blueprint.n_demes = n_batch;
    let mut indptr = vec![0i64; n_batch + 1];
    let mut dest = Vec::new();
    for deme in 0..n_batch {
        let entries = if deme % 2 == 0 { 1 } else { 0 };
        indptr[deme + 1] = indptr[deme] + entries;
        if entries == 1 {
            dest.push(((deme + 1) % n_batch) as i64);
        }
    }
    blueprint.migration_indptr = indptr;
    blueprint.migration_dest_idx = dest;
    blueprint.migration_weights = vec![1.0f64; pairs];

    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_batch).flat_map(|_| values.iter().copied()).collect()
    };
    let rate_values: Vec<f64> = (0..n_batch * 2 * n_ages)
        .map(|i| 0.1 + 0.02 * (i % 5) as f64)
        .collect();
    let ecology = EcologyParams {
        n_demes: n_batch,
        carrying_capacity: vec![1.0e9; n_batch],
        eggs_per_female: vec![1.0; n_batch],
        sex_ratio: vec![0.5; n_batch],
        sperm_displacement_rate: vec![0.1; n_batch],
        low_density_growth_rate: vec![1.0; n_batch],
        growth_mode: vec![0; n_batch],
        external_expected_eggs: vec![-1.0; n_batch],
        survival_rates: tile(&[1.0; 8]),
        mating_rates: tile(&[0.0; 8]),
        reproduction_rates: tile(&[0.0; 4]),
        fertility: tile(&[0.0; 4]),
        competition_weights: tile(&[1.0; 4]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_batch],
        migration_rate: rate_values.clone(),
        custom_slots: vec![HashMap::new(); n_batch],
    };

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let one_ind: Vec<f64> = vec![100.0; ind_stride];
    let mut one_sperm = vec![0.0f64; sperm_stride];
    for age in 1..n_ages {
        for gf in 0..z {
            for gm in 0..z {
                one_sperm[(age * z + gf) * z + gm] = 3.0;
            }
        }
    }
    let ind_all: Vec<f64> = (0..n_batch).flat_map(|_| one_ind.iter().copied()).collect();
    let sperm_all: Vec<f64> = (0..n_batch)
        .flat_map(|_| one_sperm.iter().copied())
        .collect();
    let ind_host: Vec<f32> = ind_all.iter().map(|v| *v as f32).collect();
    let sperm_host: Vec<f32> = sperm_all.iter().map(|v| *v as f32).collect();

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind_host, &sperm_host).expect("executor");
    executor.set_seed(0x0BAD_F00D_1234_5678);
    executor
        .migrate_tick_stochastic(&blueprint, &ecology)
        .expect("device stochastic migration");
    let device_ind = executor.download_ind().expect("download");
    let device_sperm = executor.download_sperm().expect("download");

    let trials = 2000usize;
    let mut cpu_sum = [vec![0.0f64; ind_stride], vec![0.0f64; ind_stride]];
    let mut cpu_sq = [vec![0.0f64; ind_stride], vec![0.0f64; ind_stride]];
    let mut cpu_sum_sperm = [vec![0.0f64; sperm_stride], vec![0.0f64; sperm_stride]];
    let mut cpu_sq_sperm = [vec![0.0f64; sperm_stride], vec![0.0f64; sperm_stride]];
    for trial in 0..trials {
        let mut rngs: Vec<_> = (0..n_batch)
            .map(|src| new_rng(50_000 + (trial as u64) * 7919 + src as u64))
            .collect();
        let (ind, sperm) = crate::kernels::spatial::migrate_csr_stochastic_rngs(
            &mut rngs,
            &ind_all,
            &sperm_all,
            &blueprint.migration_indptr,
            &blueprint.migration_dest_idx,
            &blueprint.migration_weights,
            &rate_values,
            continuous,
            n_batch,
            n_ages,
            z,
        )
        .expect("host stochastic migration");
        for pair in 0..pairs {
            for parity in 0..2 {
                let deme = 2 * pair + parity;
                for slot in 0..ind_stride {
                    let value = ind[deme * ind_stride + slot];
                    cpu_sum[parity][slot] += value;
                    cpu_sq[parity][slot] += value * value;
                }
                for slot in 0..sperm_stride {
                    let value = sperm[deme * sperm_stride + slot];
                    cpu_sum_sperm[parity][slot] += value;
                    cpu_sq_sperm[parity][slot] += value * value;
                }
            }
        }
    }
    let samples = (pairs * trials) as f64;
    for parity in 0..2 {
        for slot in 0..ind_stride {
            let cpu_mean = cpu_sum[parity][slot] / samples;
            let cpu_var = (cpu_sq[parity][slot] / samples - cpu_mean * cpu_mean).max(0.0);
            let device_mean = (0..pairs)
                .map(|pair| device_ind[(2 * pair + parity) * ind_stride + slot] as f64)
                .sum::<f64>()
                / pairs as f64;
            let tolerance = 5.0 * (cpu_var / pairs as f64).sqrt() + 0.5;
            assert!(
                (device_mean - cpu_mean).abs() <= tolerance,
                "parity {parity} ind[{slot}]: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
            );
        }
        for slot in 0..sperm_stride {
            let cpu_mean = cpu_sum_sperm[parity][slot] / samples;
            let cpu_var = (cpu_sq_sperm[parity][slot] / samples - cpu_mean * cpu_mean).max(0.0);
            let device_mean = (0..pairs)
                .map(|pair| device_sperm[(2 * pair + parity) * sperm_stride + slot] as f64)
                .sum::<f64>()
                / pairs as f64;
            let tolerance = 5.0 * (cpu_var / pairs as f64).sqrt() + 0.5;
            assert!(
                (device_mean - cpu_mean).abs() <= tolerance,
                "parity {parity} sperm[{slot}]: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
            );
        }
    }
}

#[test]
fn device_stochastic_migration_matches_host_distribution() {
    migration_distribution_case(false);
}

#[test]
fn device_stochastic_migration_continuous_matches_host_distribution() {
    migration_distribution_case(true);
}

// ---------------------------------------------------------------------------
// Evaluator adversarial tests for P4 spatial stochastic migration.
// ---------------------------------------------------------------------------

/// Base CSR topology used by the block-diagonal evaluator test.
///
/// ## Returns
/// `(indptr, dest, weights)` for `n0 = 4` demes:
/// `0 -> {1,1,0}`, `1 -> {}`, `2 -> {3}`, `3 -> {0,2}`.
fn evaluator_migration_stoch_topology() -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    (
        vec![0, 3, 3, 4, 6],
        vec![1, 1, 0, 3, 0, 2],
        vec![0.3, 0.2, 0.5, 1.0, 0.4, 0.6],
    )
}

/// Base per-deme migration rate for the evaluator topology (A = 3).
///
/// ## Returns
/// `(n0, 2, 3)` rates including zero, <1, ==1 and >1 rows.
fn evaluator_migration_stoch_rates() -> Vec<f64> {
    vec![
        0.0, 0.3, 1.5, 0.0, 0.3, 1.5, // deme 0
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // deme 1
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // deme 2
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // deme 3
    ]
}

/// Build a block-diagonal blueprint/ecology with `copies` independent copies.
///
/// ## Parameters
/// - `n0`, `n_ages`, `n_ztypes`: Base model dimensions.
/// - `copies`: Number of independent copies.
///
/// ## Returns
/// `(blueprint, ecology)` for `n_batch = n0 * copies`.
fn evaluator_block_diagonal(
    n0: usize,
    n_ages: usize,
    n_ztypes: usize,
    copies: usize,
) -> (Blueprint, EcologyParams) {
    let (base_indptr, base_dest, base_weights) = evaluator_migration_stoch_topology();
    let base_rates = evaluator_migration_stoch_rates();
    let n_batch = n0 * copies;
    let mut indptr = vec![0i64];
    let mut dest = Vec::new();
    let mut weights = Vec::new();
    for copy in 0..copies {
        let offset = (copy * n0) as i64;
        for row in 0..n0 {
            let start = base_indptr[row] as usize;
            let end = base_indptr[row + 1] as usize;
            dest.extend(base_dest[start..end].iter().map(|d| d + offset));
            weights.extend_from_slice(&base_weights[start..end]);
            indptr.push(dest.len() as i64);
        }
    }
    let mut blueprint = dimension_blueprint(n_ages, n_ztypes);
    blueprint.stochastic = true;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = indptr;
    blueprint.migration_dest_idx = dest;
    blueprint.migration_weights = weights;
    let rate: Vec<f64> = (0..copies)
        .flat_map(|_| base_rates.iter().copied())
        .collect();
    let ecology = evaluator_migration_ecology(n_batch, n_ages, &rate);
    (blueprint, ecology)
}

/// Device one-shot stochastic migration helper.
///
/// ## Parameters
/// - `blueprint`, `ecology`: Model.
/// - `ind`, `sperm`: Host state (batch-major).
/// - `seed`: Device seed.
///
/// ## Returns
/// `(ind, sperm)` device results in batch-major order.
fn evaluator_run_stochastic_migration(
    blueprint: &Blueprint,
    ecology: &EcologyParams,
    ind: &[f32],
    sperm: &[f32],
    seed: u64,
) -> (Vec<f32>, Vec<f32>) {
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(
        context,
        blueprint.n_demes,
        blueprint.n_ages,
        blueprint.n_ztypes,
        ind,
        sperm,
    )
    .expect("executor");
    executor.set_seed(seed);
    executor
        .migrate_tick_stochastic(blueprint, ecology)
        .expect("device stochastic migration");
    (
        executor.download_ind().expect("download ind"),
        executor.download_sperm().expect("download sperm"),
    )
}

#[test]
fn evaluator_stochastic_migration_is_seed_reproducible() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n0 = 4;
    let n_ages = 3;
    let z = 2;
    let copies = 200;
    let (blueprint, ecology) = evaluator_block_diagonal(n0, n_ages, z, copies);
    let n_batch = blueprint.n_demes;
    let ind_len = n_batch * 2 * n_ages * z;
    let sperm_len = n_batch * n_ages * z * z;
    let ind: Vec<f32> = (0..ind_len).map(|i| ((i * 7) % 41) as f32 + 1.0).collect();
    let sperm: Vec<f32> = (0..sperm_len).map(|i| ((i * 5) % 13) as f32).collect();

    let (a_ind, a_sperm) =
        evaluator_run_stochastic_migration(&blueprint, &ecology, &ind, &sperm, 0xABCDEF);
    let (b_ind, b_sperm) =
        evaluator_run_stochastic_migration(&blueprint, &ecology, &ind, &sperm, 0xABCDEF);
    assert_eq!(
        a_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "same seed must reproduce ind bit-for-bit"
    );
    assert_eq!(
        a_sperm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b_sperm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "same seed must reproduce sperm bit-for-bit"
    );

    let (c_ind, _) =
        evaluator_run_stochastic_migration(&blueprint, &ecology, &ind, &sperm, 0x123456);
    assert_ne!(
        a_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c_ind.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "a different seed must change the result"
    );
}

#[test]
fn evaluator_stochastic_migration_topology_matches_host_mean() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n0 = 4usize;
    let n_ages = 3usize;
    let z = 2usize;
    let copies = 500usize;
    let trials = 300usize;
    let (blueprint, ecology) = evaluator_block_diagonal(n0, n_ages, z, copies);
    let n_batch = blueprint.n_demes;
    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let ind: Vec<f32> = (0..n_batch * ind_stride)
        .map(|i| ((i * 7) % 41) as f32 + 1.0)
        .collect();
    let sperm: Vec<f32> = (0..n_batch * sperm_stride)
        .map(|i| ((i * 5) % 13) as f32)
        .collect();
    let (device_ind, device_sperm) =
        evaluator_run_stochastic_migration(&blueprint, &ecology, &ind, &sperm, 0x5EED);

    // CPU reference: pooled mean/variance over `trials` independent runs.
    let ind_f64: Vec<f64> = ind.iter().map(|v| f64::from(*v)).collect();
    let sperm_f64: Vec<f64> = sperm.iter().map(|v| f64::from(*v)).collect();
    let mut sum = vec![vec![0.0f64; ind_stride]; n0];
    let mut sq = vec![vec![0.0f64; ind_stride]; n0];
    let mut sum_s = vec![vec![0.0f64; sperm_stride]; n0];
    let mut sq_s = vec![vec![0.0f64; sperm_stride]; n0];
    for trial in 0..trials {
        let mut rngs: Vec<_> = (0..n_batch)
            .map(|src| new_rng(90_000 + trial as u64 * 7919 + src as u64))
            .collect();
        let (host_ind, host_sperm) = crate::kernels::spatial::migrate_csr_stochastic_rngs(
            &mut rngs,
            &ind_f64,
            &sperm_f64,
            &blueprint.migration_indptr,
            &blueprint.migration_dest_idx,
            &blueprint.migration_weights,
            &ecology.migration_rate,
            false,
            n_batch,
            n_ages,
            z,
        )
        .expect("host stochastic migration");
        for copy in 0..copies {
            for base in 0..n0 {
                let deme = copy * n0 + base;
                for slot in 0..ind_stride {
                    let value = host_ind[deme * ind_stride + slot];
                    sum[base][slot] += value;
                    sq[base][slot] += value * value;
                }
                for slot in 0..sperm_stride {
                    let value = host_sperm[deme * sperm_stride + slot];
                    sum_s[base][slot] += value;
                    sq_s[base][slot] += value * value;
                }
            }
        }
    }
    let cpu_samples = (copies * trials) as f64;
    for base in 0..n0 {
        for slot in 0..ind_stride {
            let cpu_mean = sum[base][slot] / cpu_samples;
            let cpu_var = (sq[base][slot] / cpu_samples - cpu_mean * cpu_mean).max(0.0);
            let device_mean = (0..copies)
                .map(|copy| device_ind[(copy * n0 + base) * ind_stride + slot] as f64)
                .sum::<f64>()
                / copies as f64;
            let tolerance = 5.0 * (cpu_var / copies as f64).sqrt() + 0.5;
            assert!(
                (device_mean - cpu_mean).abs() <= tolerance,
                "ind base={base} slot={slot}: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
            );
        }
        for slot in 0..sperm_stride {
            let cpu_mean = sum_s[base][slot] / cpu_samples;
            let cpu_var = (sq_s[base][slot] / cpu_samples - cpu_mean * cpu_mean).max(0.0);
            let device_mean = (0..copies)
                .map(|copy| device_sperm[(copy * n0 + base) * sperm_stride + slot] as f64)
                .sum::<f64>()
                / copies as f64;
            let tolerance = 5.0 * (cpu_var / copies as f64).sqrt() + 0.5;
            assert!(
                (device_mean - cpu_mean).abs() <= tolerance,
                "sperm base={base} slot={slot}: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
            );
        }
    }
}

#[test]
fn execution_rejects_overwide_stochastic_csr_rows() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (_, mut ecology) = density_fixture();
    ecology.migration_rate = vec![0.2f64; 4 * 2 * 4];
    let (ind, sperm) = populated_state(4, 4, 2);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, 4, 4, 2, &ind, &sperm).expect("executor");
    let mut blueprint = migration_blueprint();
    blueprint.migration_indptr = vec![0, 33, 33, 33, 33];
    blueprint.migration_dest_idx = vec![1i64; 33];
    blueprint.migration_weights = vec![1.0f64 / 33.0; 33];
    assert!(executor
        .migrate_tick_stochastic(&blueprint, &ecology)
        .is_err());
}

#[test]
fn device_ensemble_matches_independent_cpu_runs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 2000usize;
    let n_ages = 4usize;
    let z = 2usize;
    let ticks = 3;

    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.fixed_egg_count = false;
    blueprint.n_demes = n_batch;

    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_batch).flat_map(|_| values.iter().copied()).collect()
    };
    let mut one_ecology = density_fixture().1;
    one_ecology.n_demes = 1;
    one_ecology.carrying_capacity.truncate(1);
    one_ecology.eggs_per_female.truncate(1);
    one_ecology.sex_ratio.truncate(1);
    one_ecology.sperm_displacement_rate.truncate(1);
    one_ecology.low_density_growth_rate.truncate(1);
    one_ecology.growth_mode.truncate(1);
    one_ecology.external_expected_eggs.truncate(1);
    one_ecology.equilibrium_declared.truncate(1);
    one_ecology.survival_rates.truncate(8);
    one_ecology.mating_rates.truncate(8);
    one_ecology.reproduction_rates.truncate(4);
    one_ecology.fertility.truncate(4);
    one_ecology.competition_weights.truncate(4);
    one_ecology.migration_rate.clear();
    one_ecology.custom_slots.truncate(1);
    one_ecology.growth_mode[0] = 0;
    one_ecology.mating_rates = vec![0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9];
    one_ecology.eggs_per_female[0] = 10.0;

    let ecology = EcologyParams {
        n_demes: n_batch,
        carrying_capacity: tile(&one_ecology.carrying_capacity),
        eggs_per_female: tile(&one_ecology.eggs_per_female),
        sex_ratio: tile(&one_ecology.sex_ratio),
        sperm_displacement_rate: tile(&one_ecology.sperm_displacement_rate),
        low_density_growth_rate: tile(&one_ecology.low_density_growth_rate),
        growth_mode: tile_i64(&one_ecology.growth_mode, n_batch),
        external_expected_eggs: tile(&one_ecology.external_expected_eggs),
        survival_rates: tile(&one_ecology.survival_rates),
        mating_rates: tile(&one_ecology.mating_rates),
        reproduction_rates: tile(&one_ecology.reproduction_rates),
        fertility: tile(&one_ecology.fertility),
        competition_weights: tile(&one_ecology.competition_weights),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_batch],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new(); n_batch],
    };
    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics.clone()];
    let deme_variants = vec![0usize; n_batch];

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let mut one_ind = vec![0.0f64; ind_stride];
    for sex in 0..2 {
        for age in 0..n_ages {
            for zz in 0..z {
                one_ind[(sex * n_ages + age) * z + zz] = if age == 0 { 0.0 } else { 100.0 };
            }
        }
    }
    let mut one_sperm = vec![0.0f64; sperm_stride];
    for age in 1..n_ages {
        for gf in 0..z {
            for gm in 0..z {
                one_sperm[(age * z + gf) * z + gm] = 3.0;
            }
        }
    }
    let ind_one_f32: Vec<f32> = one_ind.iter().map(|v| *v as f32).collect();
    let sperm_one_f32: Vec<f32> = one_sperm.iter().map(|v| *v as f32).collect();

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::ensemble(
        context,
        n_batch,
        n_ages,
        z,
        &ind_one_f32,
        &sperm_one_f32,
        0x5EED_1234_ABCD_0001,
    )
    .expect("ensemble executor");
    for _ in 0..ticks {
        executor
            .tick(&blueprint, &ecology, &variants, &deme_variants)
            .expect("ensemble tick");
    }
    let device_ind = executor.download_ind().expect("download");
    let device_sperm = executor.download_sperm().expect("download");

    let mut cpu_sum_ind = vec![0.0f64; ind_stride];
    let mut cpu_sq_ind = vec![0.0f64; ind_stride];
    let mut cpu_sum_sperm = vec![0.0f64; sperm_stride];
    let mut cpu_sq_sperm = vec![0.0f64; sperm_stride];
    for trial in 0..n_batch {
        let mut rng = new_rng(90_000 + trial as u64);
        let mut ind = one_ind.clone();
        let mut sperm = one_sperm.clone();
        for _ in 0..ticks {
            crate::kernels::age_structured::reproduction(
                &mut rng,
                &blueprint,
                &one_ecology,
                &genetics,
                0,
                &mut ind,
                &mut sperm,
            )
            .expect("host reproduction");
            crate::kernels::age_structured::survival(
                &mut rng,
                &blueprint,
                &one_ecology,
                &genetics,
                0,
                &mut ind,
                &mut sperm,
            )
            .expect("host survival");
            crate::kernels::age_structured::aging(&blueprint, &mut ind, &mut sperm);
        }
        for (slot, value) in ind.iter().enumerate() {
            cpu_sum_ind[slot] += value;
            cpu_sq_ind[slot] += value * value;
        }
        for (slot, value) in sperm.iter().enumerate() {
            cpu_sum_sperm[slot] += value;
            cpu_sq_sperm[slot] += value * value;
        }
    }
    let trials = n_batch as f64;
    for slot in 0..ind_stride {
        let cpu_mean = cpu_sum_ind[slot] / trials;
        let cpu_var = (cpu_sq_ind[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_ind[b * ind_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "ind[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
    for slot in 0..sperm_stride {
        let cpu_mean = cpu_sum_sperm[slot] / trials;
        let cpu_var = (cpu_sq_sperm[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device_sperm[b * sperm_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "sperm[{slot}]: device mean {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
}

/// Tile an `i64` column across `n` batch elements.
///
/// ## Parameters
/// - `values`, `n`: Source column and repeat count.
///
/// ## Returns
/// The tiled column.
fn tile_i64(values: &[i64], n: usize) -> Vec<i64> {
    (0..n).flat_map(|_| values.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// Discrete-generation (two-age) device statistical checks.
// ---------------------------------------------------------------------------

/// Tiled two-age discrete ecology for a distribution test.
fn discrete_ecology(n_batch: usize) -> EcologyParams {
    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_batch).flat_map(|_| values.iter().copied()).collect()
    };
    EcologyParams {
        n_demes: n_batch,
        carrying_capacity: vec![500.0; n_batch],
        eggs_per_female: vec![10.0; n_batch],
        sex_ratio: vec![0.5; n_batch],
        sperm_displacement_rate: vec![0.0; n_batch],
        low_density_growth_rate: vec![3.0; n_batch],
        growth_mode: vec![2; n_batch],
        external_expected_eggs: vec![-1.0; n_batch],
        survival_rates: tile(&[0.9, 0.8, 0.85, 0.75]),
        mating_rates: tile(&[0.0, 0.9, 0.0, 0.9]),
        reproduction_rates: tile(&[0.0, 0.8]),
        fertility: tile(&[0.0, 1.0]),
        competition_weights: tile(&[1.0, 0.8]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_batch],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new(); n_batch],
    }
}

/// Truncate a tiled discrete ecology to one deme.
fn discrete_one_ecology(ecology: &EcologyParams) -> EcologyParams {
    let mut one = ecology.clone();
    one.n_demes = 1;
    one.carrying_capacity.truncate(1);
    one.eggs_per_female.truncate(1);
    one.sex_ratio.truncate(1);
    one.sperm_displacement_rate.truncate(1);
    one.low_density_growth_rate.truncate(1);
    one.growth_mode.truncate(1);
    one.external_expected_eggs.truncate(1);
    one.equilibrium_declared.truncate(1);
    one.survival_rates.truncate(4);
    one.mating_rates.truncate(4);
    one.reproduction_rates.truncate(2);
    one.fertility.truncate(2);
    one.competition_weights.truncate(2);
    one.custom_slots.truncate(1);
    one
}

/// Compare one device discrete reproduction run against the host distribution.
fn discrete_reproduction_case(continuous: bool) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 2000usize;
    let n_ages = 2usize;
    let z = 2usize;
    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = continuous;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];
    let ecology = discrete_ecology(n_batch);
    let one_ecology = discrete_one_ecology(&ecology);
    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics.clone()];

    let ind_stride = 2 * n_ages * z;
    let mut one_ind = vec![0.0f64; ind_stride];
    for sex in 0..2 {
        for zz in 0..z {
            one_ind[(sex * n_ages + 1) * z + zz] = 100.0;
        }
    }
    let ind_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_ind.iter().copied())
        .map(|v| v as f32)
        .collect();
    let sperm_host = vec![0.0f32; n_batch * n_ages * z * z];

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind_host, &sperm_host).expect("executor");
    executor.set_seed(0x1111_2222_3333_4444);
    executor
        .discrete_reproduction_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("device discrete reproduction");
    let device = executor.download_ind().expect("download");

    let mut sum = vec![0.0f64; ind_stride];
    let mut sq = vec![0.0f64; ind_stride];
    for trial in 0..n_batch {
        let mut rng = new_rng(90_000 + trial as u64);
        let mut ind = one_ind.clone();
        discrete_reproduction(&mut rng, &blueprint, &one_ecology, &genetics, 0, &mut ind);
        for (slot, value) in ind.iter().enumerate() {
            sum[slot] += value;
            sq[slot] += value * value;
        }
    }
    let trials = n_batch as f64;
    for slot in 0..ind_stride {
        let cpu_mean = sum[slot] / trials;
        let cpu_var = (sq[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device[b * ind_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "continuous={continuous} discrete repro slot {slot}: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
}

#[test]
fn device_discrete_reproduction_matches_host_distribution() {
    discrete_reproduction_case(false);
}

#[test]
fn device_discrete_reproduction_continuous_matches_host_distribution() {
    discrete_reproduction_case(true);
}

/// Compare one device discrete survival run against the host distribution.
fn discrete_survival_case(continuous: bool) {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 2000usize;
    let n_ages = 2usize;
    let z = 2usize;
    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = continuous;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];
    let ecology = discrete_ecology(n_batch);
    let one_ecology = discrete_one_ecology(&ecology);
    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics.clone()];

    let ind_stride = 2 * n_ages * z;
    let mut one_ind = vec![0.0f64; ind_stride];
    for sex in 0..2 {
        for zz in 0..z {
            one_ind[(sex * n_ages) * z + zz] = 100.0;
        }
    }
    let ind_host: Vec<f32> = (0..n_batch)
        .flat_map(|_| one_ind.iter().copied())
        .map(|v| v as f32)
        .collect();
    let sperm_host = vec![0.0f32; n_batch * n_ages * z * z];

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind_host, &sperm_host).expect("executor");
    executor.set_seed(0x9999_8888_7777_6666);
    executor
        .discrete_survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("device discrete survival");
    let device = executor.download_ind().expect("download");

    let mut sum = vec![0.0f64; ind_stride];
    let mut sq = vec![0.0f64; ind_stride];
    for trial in 0..n_batch {
        let mut rng = new_rng(130_000 + trial as u64);
        let mut ind = one_ind.clone();
        discrete_survival(&mut rng, &blueprint, &one_ecology, &genetics, 0, &mut ind);
        for (slot, value) in ind.iter().enumerate() {
            sum[slot] += value;
            sq[slot] += value * value;
        }
    }
    let trials = n_batch as f64;
    for slot in 0..ind_stride {
        let cpu_mean = sum[slot] / trials;
        let cpu_var = (sq[slot] / trials - cpu_mean * cpu_mean).max(0.0);
        let device_mean = (0..n_batch)
            .map(|b| device[b * ind_stride + slot] as f64)
            .sum::<f64>()
            / trials;
        let tolerance = 5.0 * (cpu_var / trials).sqrt() + 0.5;
        assert!(
            (device_mean - cpu_mean).abs() <= tolerance,
            "continuous={continuous} discrete survival slot {slot}: device {device_mean} vs host {cpu_mean} (tol {tolerance})"
        );
    }
}

#[test]
fn device_discrete_survival_matches_host_distribution() {
    discrete_survival_case(false);
}

#[test]
fn device_discrete_survival_continuous_matches_host_distribution() {
    discrete_survival_case(true);
}

#[test]
fn restore_state_rejects_wrong_lengths() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 3usize;
    let n_ages = 4usize;
    let z = 2usize;
    let (ind, sperm) = populated_state(n_batch, n_ages, z);
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor =
        GpuExecutor::new(context, n_batch, n_ages, z, &ind, &sperm).expect("executor");
    assert!(executor
        .restore_state(&ind[..ind.len() - 1], &sperm, 0)
        .is_err());
    assert!(executor
        .restore_state(&ind, &sperm[..sperm.len() - 1], 0)
        .is_err());
    executor
        .restore_state(&ind, &sperm, 2)
        .expect("valid restore");
}

#[test]
fn stochastic_survival_rejects_negative_virgin_state() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let n_batch = 1usize;
    let n_ages = 4usize;
    let z = 2usize;
    let rate = evaluator_rate(n_batch, n_ages);
    let ecology = evaluator_migration_ecology(n_batch, n_ages, &rate);
    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics];
    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;

    // A meaningfully negative virgin count (sperm far exceeds females).
    let mut ind = vec![0.0f32; n_batch * ind_stride];
    let mut sperm = vec![0.0f32; n_batch * sperm_stride];
    let _ = &mut ind; // female age-0 stays zero
    for gm in 0..z {
        sperm[(0 * z + 0) * z + gm] = 10.0;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, z, &ind, &sperm).expect("exec");
    executor.set_seed(1);
    let error = executor
        .survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect_err("negative virgins must be rejected");
    assert!(error.contains("n_virgins"), "unexpected error: {error}");

    // A valid state (stored sperm == female) must not be rejected. The adults
    // sit at age 1 so recruitment (age 0 only) cannot disturb the comparison.
    let mut valid_ind = vec![0.0f32; n_batch * ind_stride];
    let mut valid_sperm = vec![0.0f32; n_batch * sperm_stride];
    valid_ind[(0 * n_ages + 1) * z + 0] = 1000.0;
    for gm in 0..z {
        valid_sperm[(1 * z + 0) * z + gm] = 500.0;
    }
    let context = GpuContext::new(0).expect("device 0 context");
    let mut tolerant =
        GpuExecutor::new(context, n_batch, n_ages, z, &valid_ind, &valid_sperm).expect("exec");
    tolerant.set_seed(1);
    tolerant
        .survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch])
        .expect("a valid stored==female state must not be rejected");
}

#[test]
fn evaluator_valid_large_virgin_state_is_not_rejected() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // A valid state (stored sperm mathematically equal to the female count) at
    // a realistic per-genotype scale (~1e5, Z=4) whose f32 sequential sum
    // overshoots by ~0.008. The host (f64) sees virgins == 0 and continues;
    // the fixed 1e-3f device tolerance must not abort this valid run.
    let n_batch = 1usize;
    let n_ages = 4usize;
    let z = 4usize;
    let rate = evaluator_rate(n_batch, n_ages);
    let ecology = evaluator_migration_ecology(n_batch, n_ages, &rate);
    let genetics = reproduction_genetics(n_ages, z);
    let variants = vec![genetics];
    let mut blueprint = dimension_blueprint(n_ages, z);
    blueprint.stochastic = true;
    blueprint.continuous_sampling = true;
    blueprint.n_demes = n_batch;
    blueprint.migration_indptr = vec![0; n_batch + 1];

    let ind_stride = 2 * n_ages * z;
    let sperm_stride = n_ages * z * z;
    let mut ind = vec![0.0f32; n_batch * ind_stride];
    let mut sperm = vec![0.0f32; n_batch * sperm_stride];
    // Age 1, female ztype 0 carries 100000 individuals; age 0 stays zero so
    // recruitment does not resample the juvenile class.
    ind[(0 * n_ages + 1) * z + 0] = 100000.0;
    let stored = [
        25000.25390625f32,
        25000.666015625,
        24999.32421875,
        24999.755859375,
    ];
    // Mathematical sum is exactly 100000.0 (virgins == 0).
    let math_sum: f64 = stored.iter().map(|v| f64::from(*v)).sum();
    assert_eq!(math_sum, 100000.0);
    for gm in 0..z {
        sperm[(1 * z + 0) * z + gm] = stored[gm];
    }

    let context = GpuContext::new(0).expect("device 0 context");
    let mut executor = GpuExecutor::new(context, n_batch, n_ages, z, &ind, &sperm).expect("exec");
    executor.set_seed(7);
    let result = executor.survival_tick(&blueprint, &ecology, &variants, &vec![0usize; n_batch]);
    assert!(
        result.is_ok(),
        "a valid stored==female state must not be rejected: {result:?}"
    );
}
