//! P2 executor tests.
//!
//! The budget arithmetic is host-only; the L1 aging cross-check needs a GPU
//! and asserts by default, disabled with `NATAL_GPU_REQUIRE=0`.

use crate::gpu::context::GpuContext;
use crate::gpu::hardware_required;
use crate::kernels::age_structured::{aging, reproduction, survival};
use crate::kernels::density_regulation::regulation_scaling;
use crate::kernels::equilibrium::equilibrium_metrics;
use crate::kernels::rng::new_rng;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;

use super::GpuExecutor;
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
