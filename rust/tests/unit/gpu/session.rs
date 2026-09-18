//! Device-path tests for `AgeStructuredSession` (P2/P3 wiring).
//!
//! These build the session struct directly (the test module is a child of the
//! session module, so private fields are reachable) and exercise
//! `enable_gpu` / `gpu_status` / the `run_inner` device branch, including the
//! eligibility rejections. The success path cross-checks the device run against
//! the CPU run on the same fixture.

use pyo3::Python;

use super::AgeStructuredSession;
use crate::gpu::hardware_required;
use crate::model::blueprint::Blueprint;
use crate::model::ecology::EcologyParams;
use crate::model::genetics::GeneticsTensors;
use std::collections::HashMap;

/// Deterministic panmictic fixture: 2 sexes, 4 ages, 2 ztypes, mode 2.
///
/// ## Returns
/// `(blueprint, ecology, genetics)` for one deme.
fn fixture() -> (Blueprint, EcologyParams, GeneticsTensors) {
    let n_ages = 4usize;
    let z = 2usize;
    let new_adult_age = 1usize;
    let mut survival = Vec::new();
    for sex in 0..2 {
        for age in 0..n_ages {
            survival.push(0.9 - 0.05 * age as f64 - 0.02 * sex as f64);
        }
    }
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
        n_demes: 1,
        migration_indptr: vec![0, 0],
        migration_dest_idx: vec![],
        migration_weights: vec![],
    };
    let ecology = EcologyParams {
        n_demes: 1,
        carrying_capacity: vec![500.0],
        eggs_per_female: vec![10.0],
        sex_ratio: vec![0.5],
        sperm_displacement_rate: vec![0.1],
        low_density_growth_rate: vec![3.0],
        growth_mode: vec![2],
        external_expected_eggs: vec![-1.0],
        survival_rates: survival,
        mating_rates: vec![0.0, 0.9, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9],
        reproduction_rates: vec![0.0, 0.8, 0.7, 0.6],
        fertility: vec![0.0, 1.0, 0.9, 0.8],
        competition_weights: vec![1.0, 0.8, 0.7, 0.6],
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false],
        migration_rate: vec![],
        custom_slots: vec![HashMap::new()],
    };
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

/// Non-trivial initial state sized for the fixture.
///
/// ## Returns
/// `(individual_counts, sperm_storage)`.
fn initial_state() -> (Vec<f64>, Vec<f64>) {
    let n_ages = 4usize;
    let z = 2usize;
    let mut ind = vec![0.0f64; 2 * n_ages * z];
    let mut sperm = vec![0.0f64; n_ages * z * z];
    for sex in 0..2 {
        for age in 0..n_ages {
            for g in 0..z {
                ind[(sex * n_ages + age) * z + g] = ((sex + age + g + 1) * 5) as f64;
            }
        }
    }
    for age in 0..n_ages {
        for gf in 0..z {
            for gm in 0..z {
                sperm[(age * z + gf) * z + gm] = (age + gf + gm + 1) as f64;
            }
        }
    }
    (ind, sperm)
}

/// Build a session through the shared constructor.
///
/// ## Parameters
/// - `blueprint`, `params`, `genetics`: Rust-side contracts.
/// - `state_ind`, `state_sperm`: Panmictic initial state.
///
/// ## Returns
/// A ready, GPU-disabled session.
fn make_session(
    blueprint: Blueprint,
    params: EcologyParams,
    genetics: GeneticsTensors,
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
) -> AgeStructuredSession {
    AgeStructuredSession::assemble(blueprint, params, genetics, 0, state_ind, state_sperm)
}

#[test]
fn session_device_branch_matches_cpu_and_covers_wiring() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let mut gpu_session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu_session.enable_gpu().expect("enable_gpu succeeds");
        assert_eq!(gpu_session.gpu_status(), "enabled");
        let (tick, _history, stopped) = gpu_session
            .run_inner(py, 3, 0, None, 0)
            .expect("device run");
        assert_eq!(tick, 3);
        assert!(!stopped);

        let mut cpu_session = make_session(blueprint, params, genetics, ind, sperm);
        assert_eq!(cpu_session.gpu_status(), "disabled");
        let (cpu_tick, _history, cpu_stopped) =
            cpu_session.run_inner(py, 3, 0, None, 0).expect("cpu run");
        assert_eq!(cpu_tick, 3);
        assert!(!cpu_stopped);

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
                "ind[{index}]: device {got} vs host {want}"
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
                "sperm[{index}]: device {got} vs host {want}"
            );
        }
    });
}

#[test]
fn session_enable_gpu_rejects_ineligible_models() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (ind, sperm) = initial_state();

        let (mut blueprint, params, genetics) = fixture();
        blueprint.stochastic = true;
        let mut stochastic = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        assert!(stochastic.enable_gpu().is_err(), "stochastic must reject");

        let (mut blueprint, params, genetics) = fixture();
        blueprint.n_demes = 2;
        let mut spatial = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        assert!(spatial.enable_gpu().is_err(), "spatial must reject");

        let (blueprint, mut params, genetics) = fixture();
        params.growth_mode[0] = 5;
        let mut custom = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        assert!(custom.enable_gpu().is_err(), "custom growth must reject");

        let (blueprint, params, genetics) = fixture();
        let mut hooked = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        hooked.hooks.n_hooks = 1;
        assert!(hooked.enable_gpu().is_err(), "hooks must reject");

        // A callback-carrying program is rejected even with no declarative
        // hooks (`n_hooks == 0`), exercising the Python-callback arm.
        let (blueprint, params, genetics) = fixture();
        let mut callbacks = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        callbacks.hooks.n_hooks = 0;
        callbacks.hooks.python_callbacks = vec![vec![py.None()]];
        assert!(
            callbacks.enable_gpu().is_err(),
            "python callbacks must reject"
        );
    });
}
