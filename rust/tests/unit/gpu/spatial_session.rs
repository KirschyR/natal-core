//! Device-path tests for `SpatialSession` (P5 wiring).
//!
//! Builds the session struct directly (the test module is a child of the
//! spatial session module) and exercises `enable_gpu` / `gpu_status` / the
//! `run_inner` device branch, including eligibility rejections. The success
//! path cross-checks the device run against the CPU run on the same fixture.

use super::{SpatialSession, SpatialTickCheckpoint};
use crate::gpu::hardware_required;
use crate::hooks::interpreter::HookProgram;
use crate::kernels::rng::{new_rng, stream_seed};
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use crate::sessions::status::ExecutionStatus;
use std::collections::HashMap;

/// Deterministic 3-deme ring fixture (mode 2, one genetics variant).
///
/// ## Returns
/// `(blueprint, ecology, genetics)`.
fn fixture() -> (Blueprint, EcologyParams, GeneticsTensors) {
    let n_demes = 3usize;
    let n_ages = 4usize;
    let z = 2usize;
    let new_adult_age = 1usize;
    let blueprint = Blueprint {
        n_sexes: 2,
        n_ages,
        n_ztypes: z,
        n_gtypes: z,
        n_glabs: 1,
        new_adult_age,
        adult_ages: (new_adult_age as i64..n_ages as i64).collect(),
        stochastic: false,
        continuous_sampling: false,
        fixed_egg_count: false,
        has_sex_chromosomes: false,
        extreme_speed_mode: 0,
        ztype_names: (0..z).map(|i| format!("z{i}")).collect(),
        gtype_names: (0..z).map(|i| format!("g{i}")).collect(),
        female_only_by_sex_chrom: vec![false; z],
        male_only_by_sex_chrom: vec![false; z],
        initial_individual_count: vec![],
        initial_sperm_storage: vec![],
        n_demes,
        migration_indptr: vec![0, 2, 4, 6],
        migration_dest_idx: vec![1, 2, 0, 2, 0, 1],
        migration_weights: vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    };
    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_demes).flat_map(|_| values.iter().copied()).collect()
    };
    let survival = tile(&[0.9, 0.8, 0.7, 0.6, 0.85, 0.75, 0.65, 0.55]);
    let mating = tile(&[0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9]);
    let ecology = EcologyParams {
        n_demes,
        carrying_capacity: vec![500.0; n_demes],
        eggs_per_female: vec![10.0; n_demes],
        sex_ratio: vec![0.5; n_demes],
        sperm_displacement_rate: vec![0.1; n_demes],
        low_density_growth_rate: vec![3.0; n_demes],
        growth_mode: vec![2; n_demes],
        external_expected_eggs: vec![-1.0; n_demes],
        survival_rates: survival,
        mating_rates: mating,
        reproduction_rates: tile(&[0.0, 0.8, 0.7, 0.6]),
        fertility: tile(&[0.0, 1.0, 0.9, 0.8]),
        competition_weights: tile(&[1.0, 0.8, 0.7, 0.6]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_demes],
        migration_rate: tile(&[0.1, 0.2, 0.15, 0.1, 0.1, 0.2, 0.15, 0.1]),
        custom_slots: vec![HashMap::new(); n_demes],
    };
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
    let genetics = GeneticsTensors {
        viability_fitness: vec![1.0; 2 * n_ages * z],
        fecundity_fitness: vec![1.0; 2 * z],
        sexual_selection_fitness: sexual_selection,
        zygote_viability_fitness: vec![1.0; 2 * z],
        offspring_tensor: offspring,
        meiosis_map: vec![0.0; 2 * z * z],
        female_ztype_compatibility: vec![0.5; z],
        male_ztype_compatibility: vec![0.5; z],
    };
    (blueprint, ecology, genetics)
}

/// Non-trivial stacked state for 3 demes.
///
/// ## Returns
/// `(individual_counts, sperm_storage)` in `(D, 2, A, Z)` / `(D, A, Z, Z)`.
fn initial_state() -> (Vec<f64>, Vec<f64>) {
    let n_demes = 3usize;
    let n_ages = 4usize;
    let z = 2usize;
    let mut ind = vec![0.0f64; n_demes * 2 * n_ages * z];
    let mut sperm = vec![0.0f64; n_demes * n_ages * z * z];
    for deme in 0..n_demes {
        for sex in 0..2 {
            for age in 0..n_ages {
                for g in 0..z {
                    ind[deme * 2 * n_ages * z + (sex * n_ages + age) * z + g] =
                        ((deme + sex + age + g + 1) * 5) as f64;
                }
            }
        }
        for age in 0..n_ages {
            for gf in 0..z {
                for gm in 0..z {
                    sperm[deme * n_ages * z * z + (age * z + gf) * z + gm] =
                        (deme + age + gf + gm + 1) as f64;
                }
            }
        }
    }
    (ind, sperm)
}

/// Build a spatial session directly.
///
/// ## Parameters
/// - `blueprint`, `ecology`, `variants`: Rust-side contracts.
/// - `deme_variants`: Per-deme variant ids.
/// - `stay_after_send`: Deterministic migration bookkeeping mode.
/// - `discrete`: Whether the discrete lifecycle is requested.
///
/// ## Returns
/// A ready, GPU-disabled session.
fn make_session(
    blueprint: Blueprint,
    ecology: EcologyParams,
    variants: Vec<GeneticsTensors>,
    deme_variants: Vec<usize>,
    stay_after_send: bool,
    discrete: bool,
) -> SpatialSession {
    let (ind, sperm) = initial_state();
    let seed = 0u64;
    SpatialSession {
        blueprint,
        ecology,
        variants,
        deme_variants: deme_variants.clone(),
        hooks: HookProgram::default(),
        seed,
        rngs: (0..deme_variants.len())
            .map(|deme| new_rng(stream_seed(seed, deme as i64)))
            .collect(),
        state_ind: ind,
        state_sperm: sperm,
        state_tick: 0,
        execution: ExecutionStatus::Ready,
        phase: 0,
        discrete,
        stay_after_send,
        eco_journal: Vec::new(),
        checkpoints: Vec::<SpatialTickCheckpoint>::new(),
        history_store: None,
        gpu: None,
    }
}

#[test]
fn spatial_device_branch_matches_cpu_and_covers_wiring() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    for stay_after in [false, true] {
        let mut gpu_session = make_session(
            blueprint.clone(),
            ecology.clone(),
            vec![genetics.clone()],
            vec![0, 0, 0],
            stay_after,
            false,
        );
        gpu_session.enable_gpu().expect("spatial enable_gpu");
        assert_eq!(gpu_session.gpu_status(), "enabled");
        for _ in 0..3 {
            gpu_session.run_inner().expect("device spatial tick");
        }
        assert_eq!(gpu_session.state_tick, 3);

        let mut cpu_session = make_session(
            blueprint.clone(),
            ecology.clone(),
            vec![genetics.clone()],
            vec![0, 0, 0],
            stay_after,
            false,
        );
        assert_eq!(cpu_session.gpu_status(), "disabled");
        for _ in 0..3 {
            cpu_session.run_inner().expect("cpu spatial tick");
        }
        assert_eq!(cpu_session.state_tick, 3);

        for (index, (got, want)) in gpu_session
            .state_ind
            .iter()
            .zip(cpu_session.state_ind.iter())
            .enumerate()
        {
            let want = *want as f32;
            let got = *got as f32;
            let tolerance = 1.2e-6f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "stay_after={stay_after} ind[{index}]: device {got} vs host {want}"
            );
        }
        for (index, (got, want)) in gpu_session
            .state_sperm
            .iter()
            .zip(cpu_session.state_sperm.iter())
            .enumerate()
        {
            let want = *want as f32;
            let got = *got as f32;
            let tolerance = 1.2e-6f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "stay_after={stay_after} sperm[{index}]: device {got} vs host {want}"
            );
        }
    }
}

#[test]
fn spatial_enable_gpu_rejects_ineligible_models() {
    let (blueprint, ecology, genetics) = fixture();

    let discrete = make_session(
        blueprint.clone(),
        ecology.clone(),
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        true,
    );
    let mut discrete = discrete;
    assert!(discrete.enable_gpu().is_err(), "discrete must reject");

    let (mut blueprint, ecology, genetics) = fixture();
    blueprint.continuous_sampling = true;
    let mut continuous = make_session(
        blueprint,
        ecology,
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    assert!(
        continuous.enable_gpu().is_err(),
        "continuous sampling must reject"
    );

    let (blueprint, mut ecology, genetics) = fixture();
    ecology.growth_mode[0] = 5;
    let mut custom = make_session(
        blueprint,
        ecology,
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    assert!(custom.enable_gpu().is_err(), "custom growth must reject");

    let (blueprint, ecology, genetics) = fixture();
    let mut hooked = make_session(
        blueprint,
        ecology,
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    hooked.hooks.n_hooks = 1;
    assert!(hooked.enable_gpu().is_err(), "hooks must reject");

    // Fewer variant ids than demes is rejected before any device work.
    let (blueprint, ecology, genetics) = fixture();
    let mut mismatch = make_session(
        blueprint,
        ecology,
        vec![genetics.clone()],
        vec![0, 0],
        false,
        false,
    );
    assert!(
        mismatch.enable_gpu().is_err(),
        "deme/variant mismatch must reject"
    );
}

#[test]
fn spatial_stochastic_session_device_tick_runs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (mut blueprint, ecology, genetics) = fixture();
    blueprint.stochastic = true;
    let mut session = make_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        false,
    );
    session.enable_gpu().expect("enable stochastic spatial gpu");
    assert_eq!(session.gpu_status(), "enabled");
    session.run_inner().expect("device stochastic spatial tick");
    assert_eq!(session.state_tick, 1);
    assert!(session.state_ind.iter().all(|value| value.is_finite()));
    assert!(session.state_sperm.iter().all(|value| value.is_finite()));
}
