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
use crate::output::history::{HistoryData, SharedHistory};
use crate::output::observation::project;
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
        let host_ind_before = gpu_session.state_ind.clone();
        let host_sperm_before = gpu_session.state_sperm.clone();
        for _ in 0..3 {
            gpu_session.run_inner().expect("device spatial tick");
        }
        assert_eq!(gpu_session.state_tick, 3);
        // Device ticks keep state resident: the host arrays are untouched
        // until an explicit sync (zero per-tick copy-back).
        assert_eq!(
            gpu_session.state_ind, host_ind_before,
            "device ticks must not copy individual state back to the host"
        );
        assert_eq!(
            gpu_session.state_sperm, host_sperm_before,
            "device ticks must not copy sperm state back to the host"
        );
        gpu_session.sync_gpu_state().expect("device state sync");

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
    session
        .run_steps(1, 0)
        .expect("device stochastic spatial tick");
    assert_eq!(session.state_tick, 1);
    assert!(session.state_ind.iter().all(|value| value.is_finite()));
    assert!(session.state_sperm.iter().all(|value| value.is_finite()));
}

/// Regression: the public single-tick `run_tick` must leave the host session
/// arrays consistent with the completed tick. With the zero-copy-back design,
/// the device path advances on the GPU but never refreshes `state_ind`, so a
/// subsequent `state_snapshot`/`observe_current`/`capture_checkpoint` returns
/// the *pre-tick* state under an advanced tick.
#[test]
fn evaluator_single_tick_syncs_host_state() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    let mut gpu = make_session(
        blueprint.clone(),
        ecology.clone(),
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    let mut cpu = make_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        false,
    );
    gpu.enable_gpu().expect("enable gpu");
    assert_eq!(gpu.gpu_status(), "enabled");

    gpu.run_tick().expect("gpu run_tick");
    cpu.run_tick().expect("cpu run_tick");
    assert_eq!(gpu.state_tick, cpu.state_tick);

    for (index, (got, want)) in gpu.state_ind.iter().zip(cpu.state_ind.iter()).enumerate() {
        let got = *got as f32;
        let want = *want as f32;
        let tolerance = 1.2e-6f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "host ind[{index}] stale after run_tick: gpu {got} vs cpu {want}"
        );
    }
}

/// Build a configured observation history store for the fixture dimensions.
///
/// ## Parameters
/// - `dims`, `mask`, `selected`, `collapse`, `aggregate`: Projection config.
///
/// ## Returns
/// A shared store whose `width` includes the tick column.
fn configured_history(
    dims: [usize; 4],
    mask: Vec<f64>,
    selected: Vec<usize>,
    collapse: bool,
    aggregate: bool,
) -> SharedHistory {
    let store = HistoryData::transient(1, dims, false);
    {
        let mut data = store.lock().unwrap();
        data.mask = mask;
        data.selected = selected;
        data.collapse_age = collapse;
        data.aggregate = aggregate;
        let zero = vec![0.0f64; dims.iter().product()];
        let projected =
            project(&zero, &data.mask, dims, &data.selected, collapse, aggregate).unwrap();
        data.width = 1 + projected.len();
    }
    store
}

#[test]
fn spatial_device_history_stages_and_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    let dims = [3usize, 2, 4, 2];
    let plane = 2 * 4 * 2;
    let mask = vec![1.0f64; plane];
    for (collapse, aggregate) in [(false, false), (true, true)] {
        for interval in [1i64, 2] {
            let selected = vec![0usize, 1, 2];
            let mut gpu = make_session(
                blueprint.clone(),
                ecology.clone(),
                vec![genetics.clone()],
                vec![0, 0, 0],
                false,
                false,
            );
            gpu.enable_gpu().expect("enable gpu");
            gpu.history_store = Some(configured_history(
                dims,
                mask.clone(),
                selected.clone(),
                collapse,
                aggregate,
            ));
            // Guard against a silent host fallback: the window must be usable.
            assert!(
                gpu.start_device_history(8),
                "device history staging must be available"
            );
            gpu.run_steps(4, interval).expect("gpu run_steps");

            let mut cpu = make_session(
                blueprint.clone(),
                ecology.clone(),
                vec![genetics.clone()],
                vec![0, 0, 0],
                false,
                false,
            );
            cpu.history_store = Some(configured_history(
                dims,
                mask.clone(),
                selected.clone(),
                collapse,
                aggregate,
            ));
            cpu.run_steps(4, interval).expect("cpu run_steps");

            let (g_flat, g_width, g_rows) = {
                let data = gpu.history_store.as_ref().unwrap().lock().unwrap();
                let (flat, rows) = data.flat_rows();
                (flat, data.width, rows)
            };
            let (c_flat, c_width, c_rows) = {
                let data = cpu.history_store.as_ref().unwrap().lock().unwrap();
                let (flat, rows) = data.flat_rows();
                (flat, data.width, rows)
            };
            assert_eq!(g_width, c_width);
            assert_eq!(g_rows, c_rows, "history row count");
            let expected: Vec<i64> = (0..=4).step_by(interval as usize).collect();
            assert_eq!(g_rows, expected.len(), "unexpected row count");
            for (row, tick) in expected.iter().enumerate() {
                assert_eq!(g_flat[row * g_width] as i64, *tick, "gpu tick");
                assert_eq!(c_flat[row * c_width] as i64, *tick, "cpu tick");
            }
            for (index, (got, want)) in g_flat.iter().zip(c_flat.iter()).enumerate() {
                let got = *got as f32;
                let want = *want as f32;
                let tolerance = 1e-4f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "collapse={collapse} aggregate={aggregate} interval={interval} [{index}]: \
                     device {got} vs host {want}"
                );
            }
        }
    }
}

#[test]
fn spatial_device_history_falls_back_for_raw_or_missing_window() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    let dims = [3usize, 2, 4, 2];

    // GPU disabled: no device window.
    let mut plain = make_session(
        blueprint.clone(),
        ecology.clone(),
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    assert!(!plain.start_device_history(2));

    // GPU enabled but no history store: no device window.
    let mut gpu = make_session(
        blueprint.clone(),
        ecology.clone(),
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        false,
    );
    gpu.enable_gpu().expect("enable gpu");
    assert!(!gpu.start_device_history(2));

    // Raw history is full-state: it keeps the host per-record path.
    let raw_width = 1 + 3 * 2 * 4 * 2 + 3 * 4 * 2 * 2;
    gpu.history_store = Some(HistoryData::transient(raw_width, dims, true));
    assert!(!gpu.start_device_history(2));
    gpu.run_steps(2, 1).expect("raw gpu run");
    let ticks: Vec<i64> = {
        let data = gpu.history_store.as_ref().unwrap().lock().unwrap();
        data.rows.iter().map(|row| row[0] as i64).collect()
    };
    assert_eq!(ticks, vec![0, 1, 2], "raw history must keep host recording");

    // Observation history on the same GPU session uses the device window.
    gpu.history_store = Some(configured_history(
        dims,
        vec![1.0; 16],
        vec![0, 1, 2],
        true,
        true,
    ));
    assert!(gpu.start_device_history(2));
}

/// Flatten a session's bound history store into `(flat values, width)`.
///
/// ## Parameters
/// - `session`: A spatial session with a bound history store.
///
/// ## Returns
/// `(flat row-major values, row width including the tick column)`.
fn history_flat(session: &SpatialSession) -> (Vec<f64>, usize) {
    let data = session.history_store.as_ref().unwrap().lock().unwrap();
    let (flat, _rows) = data.flat_rows();
    (flat, data.width)
}

/// Evaluator: non-zero start ticks and multi-run continuation must reconstruct
/// device history ticks exactly like the host path.
#[test]
fn evaluator_device_history_start_tick_and_continuation() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    let dims = [3usize, 2, 4, 2];
    let plane = 2 * 4 * 2;
    let mask = vec![1.0f64; plane];
    let selected = vec![0usize, 1, 2];
    // (n_ticks, record_interval) sequences; intervals of 0 mean "no recording".
    let sequences: [&[(i64, i64)]; 4] =
        [&[(3, 2), (3, 2)], &[(1, 0), (4, 2)], &[(2, 3)], &[(5, 1)]];
    for sequence in sequences {
        let mut gpu = make_session(
            blueprint.clone(),
            ecology.clone(),
            vec![genetics.clone()],
            vec![0, 0, 0],
            false,
            false,
        );
        gpu.enable_gpu().expect("enable gpu");
        gpu.history_store = Some(configured_history(
            dims,
            mask.clone(),
            selected.clone(),
            false,
            false,
        ));
        let mut cpu = make_session(
            blueprint.clone(),
            ecology.clone(),
            vec![genetics.clone()],
            vec![0, 0, 0],
            false,
            false,
        );
        cpu.history_store = Some(configured_history(
            dims,
            mask.clone(),
            selected.clone(),
            false,
            false,
        ));
        for &(n, interval) in sequence {
            gpu.run_steps(n, interval).expect("gpu run_steps");
            cpu.run_steps(n, interval).expect("cpu run_steps");
        }
        let (g_flat, g_width) = history_flat(&gpu);
        let (c_flat, c_width) = history_flat(&cpu);
        assert_eq!(g_width, c_width, "width for {sequence:?}");
        assert_eq!(
            g_flat.len(),
            c_flat.len(),
            "flat length for {sequence:?} (gpu {g_flat:?} vs cpu {c_flat:?})"
        );
        let g_ticks: Vec<i64> = g_flat.iter().step_by(g_width).map(|v| *v as i64).collect();
        let c_ticks: Vec<i64> = c_flat.iter().step_by(c_width).map(|v| *v as i64).collect();
        assert_eq!(g_ticks, c_ticks, "history ticks for {sequence:?}");
        for (index, (got, want)) in g_flat.iter().zip(c_flat.iter()).enumerate() {
            let got = *got as f32;
            let want = *want as f32;
            let tolerance = 1.2e-6f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "{sequence:?} [{index}]: device {got} vs host {want}"
            );
        }
    }
}

/// Evaluator: `configure_history` must fail explicitly on an over-budget window
/// (the session then keeps the host path; no silent engine fallback).
#[test]
fn evaluator_configure_history_over_budget_is_explicit() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = fixture();
    let dims = [3usize, 2, 4, 2];
    let plane = 2 * 4 * 2;
    let width = 2 * 3 * 1; // groups=1, out_d=3, out_a=1
    let mut session = make_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        false,
    );
    session.enable_gpu().expect("enable gpu");
    // A window far beyond any device: configure must error, not silently shrink.
    let huge = crate::gpu::executor::HistorySpec {
        capacity: usize::MAX / 8,
        width,
        dims,
        mask: vec![1.0f64; plane],
        selected: vec![0, 1, 2],
        collapse: true,
        aggregate: false,
    };
    let mut gpu = session.gpu.take().expect("executor");
    let result = gpu.configure_history(&huge);
    session.gpu = Some(gpu);
    assert!(result.is_err(), "over-budget history must be rejected");
}

#[test]
fn run_steps_rejects_overflowing_tick_span() {
    // The rejection builds a `PyValueError`; initialize the interpreter so the
    // test also passes when run in isolation.
    pyo3::prepare_freethreaded_python();
    let (blueprint, ecology, genetics) = fixture();
    let mut session = make_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        false,
    );
    session.run_steps(1, 0).expect("first tick");
    let error = session
        .run_steps(i64::MAX, 1)
        .expect_err("an overflowing tick span must be rejected");
    assert!(
        error.to_string().contains("overflow"),
        "unexpected message: {error}"
    );
}

#[test]
fn spatial_enable_gpu_accepts_continuous_sampling() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (mut blueprint, ecology, genetics) = fixture();
    blueprint.stochastic = true;
    blueprint.continuous_sampling = true;
    let mut session = make_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        false,
    );
    session
        .enable_gpu()
        .expect("continuous sampling must be accepted");
    assert_eq!(session.gpu_status(), "enabled");
    session.run_steps(1, 0).expect("continuous device tick");
    assert_eq!(session.state_tick, 1);
    assert!(session.state_ind.iter().all(|value| value.is_finite()));
    assert!(session.state_sperm.iter().all(|value| value.is_finite()));
}

/// Deterministic 3-deme discrete-generation (two-age) fixture.
///
/// ## Returns
/// `(blueprint, ecology, genetics)`.
fn discrete_fixture() -> (Blueprint, EcologyParams, GeneticsTensors) {
    let n_demes = 3usize;
    let n_ages = 2usize;
    let z = 2usize;
    let blueprint = Blueprint {
        n_sexes: 2,
        n_ages,
        n_ztypes: z,
        n_gtypes: z,
        n_glabs: 1,
        new_adult_age: 1,
        adult_ages: vec![1],
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
        migration_weights: vec![0.5; 6],
    };
    let tile = |values: &[f64]| -> Vec<f64> {
        (0..n_demes).flat_map(|_| values.iter().copied()).collect()
    };
    let ecology = EcologyParams {
        n_demes,
        carrying_capacity: vec![500.0; n_demes],
        eggs_per_female: vec![10.0; n_demes],
        sex_ratio: vec![0.5; n_demes],
        sperm_displacement_rate: vec![0.0; n_demes],
        low_density_growth_rate: vec![3.0; n_demes],
        growth_mode: vec![2; n_demes],
        external_expected_eggs: vec![-1.0; n_demes],
        survival_rates: tile(&[0.9, 0.8, 0.85, 0.75]),
        mating_rates: tile(&[0.0, 0.9, 0.0, 0.9]),
        reproduction_rates: tile(&[0.0, 0.8]),
        fertility: tile(&[0.0, 1.0]),
        competition_weights: tile(&[1.0, 0.8]),
        equilibrium_distribution: vec![],
        equilibrium_declared: vec![false; n_demes],
        migration_rate: tile(&[0.1, 0.2, 0.1, 0.15]),
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

/// Non-trivial two-age stacked state for 3 demes.
///
/// ## Returns
/// `(individual_counts, sperm_storage)`; the sperm plane stays zero (discrete).
fn discrete_initial_state() -> (Vec<f64>, Vec<f64>) {
    let n_demes = 3usize;
    let n_ages = 2usize;
    let z = 2usize;
    let mut ind = vec![0.0f64; n_demes * 2 * n_ages * z];
    for deme in 0..n_demes {
        for sex in 0..2 {
            for age in 0..n_ages {
                for g in 0..z {
                    ind[deme * 2 * n_ages * z + (sex * n_ages + age) * z + g] =
                        ((deme + sex + age + g + 1) * 5) as f64;
                }
            }
        }
    }
    let sperm = vec![0.0f64; n_demes * n_ages * z * z];
    (ind, sperm)
}

/// Build a spatial session holding the discrete two-age fixture state.
fn make_discrete_session(
    blueprint: Blueprint,
    ecology: EcologyParams,
    variants: Vec<GeneticsTensors>,
    deme_variants: Vec<usize>,
    stay_after_send: bool,
    discrete: bool,
) -> SpatialSession {
    let (ind, sperm) = discrete_initial_state();
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
fn spatial_discrete_device_tick_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (blueprint, ecology, genetics) = discrete_fixture();
    let mut gpu = make_discrete_session(
        blueprint.clone(),
        ecology.clone(),
        vec![genetics.clone()],
        vec![0, 0, 0],
        false,
        true,
    );
    gpu.enable_gpu().expect("enable discrete spatial gpu");
    assert_eq!(gpu.gpu_status(), "enabled");
    for _ in 0..3 {
        gpu.run_inner().expect("device discrete tick");
    }
    gpu.sync_gpu_state().expect("sync");

    let mut cpu = make_discrete_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        true,
    );
    for _ in 0..3 {
        cpu.run_inner().expect("cpu discrete tick");
    }
    assert_eq!(gpu.state_tick, cpu.state_tick);
    for (index, (got, want)) in gpu.state_ind.iter().zip(cpu.state_ind.iter()).enumerate() {
        let got = *got as f32;
        let want = *want as f32;
        let tolerance = 1.0e-4f32 * want.abs().max(1.0);
        assert!(
            (got - want).abs() <= tolerance,
            "discrete ind[{index}]: device {got} vs host {want}"
        );
    }
}

#[test]
fn spatial_discrete_stochastic_device_tick_runs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let (mut blueprint, ecology, genetics) = discrete_fixture();
    blueprint.stochastic = true;
    let mut session = make_discrete_session(
        blueprint,
        ecology,
        vec![genetics],
        vec![0, 0, 0],
        false,
        true,
    );
    session
        .enable_gpu()
        .expect("enable stochastic discrete spatial gpu");
    session
        .run_steps(2, 0)
        .expect("device stochastic discrete tick");
    assert_eq!(session.state_tick, 2);
    assert!(session.state_ind.iter().all(|value| value.is_finite()));
    assert!(session.state_sperm.iter().all(|value| value.is_finite()));
}

#[test]
fn spatial_gpu_restore_checkpoint_rewinds_device_state() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    let (blueprint, ecology, genetics) = fixture();
    let dims = [3usize, 2, 4, 2];
    // Raw rows are [tick, ind..., sperm...].
    let raw_width = 1 + 3 * 2 * 4 * 2 + 3 * 4 * 2 * 2;
    let make = || {
        let mut session = make_session(
            blueprint.clone(),
            ecology.clone(),
            vec![genetics.clone()],
            vec![0, 0, 0],
            false,
            false,
        );
        session.enable_gpu().expect("enable gpu");
        session.history_store = Some(HistoryData::transient(raw_width, dims, true));
        session
    };
    let mut restored = make();
    let mut reference = make();

    // Both run one tick; `restored` then advances to tick 2 and back to 1.
    restored.run_steps(1, 1).expect("restored tick 1");
    reference.run_steps(1, 1).expect("reference tick 1");
    restored.run_steps(1, 1).expect("restored tick 2");
    assert_eq!(restored.state_tick, 2);
    let outcome = restored
        .restore_from_checkpoint(1)
        .expect("restore checkpoint");
    assert_eq!(outcome, Some(1), "checkpoint at tick 1 must exist");
    assert_eq!(restored.state_tick, 1);

    // Advance both two ticks from tick 1 and compare the full device state.
    restored.run_steps(2, 1).expect("restored rerun");
    reference.run_steps(2, 1).expect("reference rerun");
    assert_eq!(restored.state_tick, reference.state_tick);
    assert_eq!(restored.state_ind.len(), reference.state_ind.len());
    for (index, (got, want)) in restored
        .state_ind
        .iter()
        .zip(reference.state_ind.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "device state diverged after restore at ind[{index}]: {got} vs {want}"
        );
    }
    for (index, (got, want)) in restored
        .state_sperm
        .iter()
        .zip(reference.state_sperm.iter())
        .enumerate()
    {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "device state diverged after restore at sperm[{index}]: {got} vs {want}"
        );
    }
}
