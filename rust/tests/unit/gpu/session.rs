//! Device-path tests for `AgeStructuredSession` (P2/P3 wiring).
//!
//! These build the session struct directly (the test module is a child of the
//! session module, so private fields are reachable) and exercise
//! `enable_gpu` / `gpu_status` / the `run_inner` device branch, including the
//! eligibility rejections. The success path cross-checks the device run against
//! the CPU run on the same fixture.

use numpy::{PyArrayMethods, PyUntypedArrayMethods};
use pyo3::Python;

use super::AgeStructuredSession;
use crate::gpu::hardware_required;
use crate::hooks::interpreter::HookProgram;
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
        blueprint.n_demes = 2;
        let mut spatial = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        assert!(spatial.enable_gpu().is_err(), "spatial must reject");

        let (blueprint, mut params, genetics) = fixture();
        params.growth_mode[0] = 5;
        let mut custom = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        assert!(custom.enable_gpu().is_err(), "custom growth must reject");

        // SAMPLE and stochastic-model hooks are device-supported from P7.3.
        if hardware_required() {
            let (blueprint, params, genetics) = fixture();
            let mut sample = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
            sample.hooks.n_hooks = 1;
            sample.hooks.op_types = vec![5];
            assert!(sample.enable_gpu().is_ok(), "SAMPLE hooks must be accepted");

            let (mut blueprint, params, genetics) = fixture();
            blueprint.stochastic = true;
            let mut stochastic =
                make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
            stochastic.hooks.n_hooks = 1;
            stochastic.hooks.op_types = vec![0];
            assert!(
                stochastic.enable_gpu().is_ok(),
                "hooks on a stochastic model must be accepted"
            );
        }

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

#[test]
fn session_gpu_ensemble_runs_and_covers_wiring() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (mut blueprint, params, genetics) = fixture();
        blueprint.stochastic = true;
        let (ind, sperm) = initial_state();
        let mut session = make_session(blueprint, params, genetics, ind, sperm);
        session.enable_gpu_ensemble(256).expect("enable ensemble");
        assert_eq!(session.gpu_status(), "enabled");
        let (tick, ind_arr, sperm_arr) = session.run_gpu_ensemble(py, 3).expect("run ensemble");
        assert_eq!(tick, 3);
        assert_eq!(ind_arr.len(), 256 * 2 * 4 * 2);
        assert_eq!(sperm_arr.len(), 256 * 4 * 2 * 2);
        assert!(ind_arr
            .readonly()
            .as_slice()
            .expect("array slice")
            .iter()
            .all(|value| value.is_finite()));
    });
}

#[test]
fn session_gpu_ensemble_rejects_ineligible() {
    let (blueprint, params, genetics) = fixture();
    let (ind, sperm) = initial_state();
    let mut session = make_session(blueprint, params, genetics, ind, sperm);
    assert!(session.enable_gpu_ensemble(0).is_err());
}

#[test]
fn session_accepts_continuous_sampling() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|_py| {
        let (mut blueprint, params, genetics) = fixture();
        blueprint.stochastic = true;
        blueprint.continuous_sampling = true;
        let (ind, sperm) = initial_state();

        let mut session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        session
            .enable_gpu()
            .expect("continuous sampling must be accepted");
        assert_eq!(session.gpu_status(), "enabled");

        let mut ensemble = make_session(blueprint, params, genetics, ind, sperm);
        ensemble
            .enable_gpu_ensemble(4)
            .expect("continuous ensemble must be accepted");
        assert_eq!(ensemble.gpu_status(), "enabled");
    });
}

#[test]
fn session_gpu_restore_device_state_rewinds() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let mut restored = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        restored.enable_gpu().expect("enable restored");
        restored.run(py, 1, 0, None, 0).expect("restored tick 1");
        let tick1_ind = restored.state_ind.clone();
        let tick1_sperm = restored.state_sperm.clone();
        restored.run(py, 1, 0, None, 0).expect("restored tick 2");
        assert_eq!(restored.state_tick, 2);

        // Simulate a host checkpoint restore, then rewind the device too.
        restored.state_ind = tick1_ind;
        restored.state_sperm = tick1_sperm;
        restored.state_tick = 1;
        restored
            .restore_device_state(1)
            .expect("device state restore");
        restored.run(py, 1, 0, None, 0).expect("restored rerun");

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.enable_gpu().expect("enable reference");
        reference.run(py, 2, 0, None, 0).expect("reference tick 2");

        assert_eq!(restored.state_tick, reference.state_tick);
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
    });
}

/// A deterministic hook program exercising SCALE, a tick condition, and CONVERT.
///
/// Slot order by event is: event 0 `SCALE(0.5)` over all cells; event 1
/// `SET(2.0)` over all cells guarded by `tick >= 1`; event 2 `CONVERT(z0→z1)`
/// with probability 0.25.
///
/// ## Returns
/// A hook program sized for the 2-sex, 4-age, 2-ztype fixture.
fn device_hook_program() -> HookProgram {
    let n_events = 4usize;
    let mut program = HookProgram::default();
    program.n_events = n_events as i64;
    program.n_hooks = 3;
    program.hook_offsets = vec![0, 1, 2, 3, 3];
    program.op_offsets = vec![0, 1, 2, 3];
    program.op_types = vec![0, 1, 11];
    program.zidx_offsets = vec![0, 2, 4, 6];
    program.zidx_data = vec![0, 1, 0, 1, 0, 1];
    program.age_offsets = vec![0, 4, 8, 12];
    program.age_data = vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3];
    program.sex_masks = vec![true; 6];
    program.params = vec![0.5, 2.0, 0.25];
    program.condition_offsets = vec![0, 0, 1, 1];
    program.condition_types = vec![3];
    program.condition_params = vec![1];
    program.deme_selector_types = vec![0, 0, 0];
    program.deme_selector_offsets = vec![0, 0, 0, 0];
    program.convert_source_z = vec![-1, -1, 0];
    program.convert_target_z = vec![-1, -1, 1];
    program
}

#[test]
fn session_device_hooks_match_cpu() {
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
        gpu_session.hooks = device_hook_program();
        gpu_session
            .enable_gpu()
            .expect("enable_gpu accepts supported hooks");
        let (tick, _history, stopped) = gpu_session
            .run_inner(py, 4, 0, None, 0)
            .expect("device hook run");
        assert_eq!(tick, 4);
        assert!(!stopped);

        let mut cpu_session = make_session(blueprint, params, genetics, ind, sperm);
        cpu_session.hooks = device_hook_program();
        cpu_session
            .run_inner(py, 4, 0, None, 0)
            .expect("cpu hook run");

        for (index, (got, want)) in gpu_session
            .state_ind
            .iter()
            .zip(cpu_session.state_ind.iter())
            .enumerate()
        {
            let want = *want as f32;
            let got = *got as f32;
            let tolerance = 1.2e-5f32 * want.abs().max(1.0);
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
            let tolerance = 1.2e-5f32 * want.abs().max(1.0);
            assert!(
                (got - want).abs() <= tolerance,
                "sperm[{index}]: device {got} vs host {want}"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Independent evaluator tests (P7.1 device declarative hook interpreter).
//
// These build raw CSR programs directly so every opcode, selector, condition
// form, and CONVERT direction can be exercised against the CPU golden path
// without going through the Python compiler.
// ---------------------------------------------------------------------------

/// One declarative operation in an evaluator-built CSR program.
struct EvalOp {
    op_type: i64,
    zidx: Vec<i64>,
    ages: Vec<i64>,
    sex: [bool; 2],
    param: f64,
    cond: Vec<(i64, i64)>,
    convert: Option<(i64, i64)>,
}

/// One hook slot (ops in execution order) plus its deme selector.
struct EvalHook {
    ops: Vec<EvalOp>,
    deme_type: i64,
    deme_data: Vec<i64>,
}

/// Build a CSR [`HookProgram`] from four events `[first, early, late, finish]`.
fn eval_program(events: [Vec<EvalHook>; 4]) -> HookProgram {
    let mut p = HookProgram::default();
    p.n_events = 4;
    p.hook_offsets = vec![0];
    p.op_offsets = vec![0];
    p.zidx_offsets = vec![0];
    p.age_offsets = vec![0];
    p.condition_offsets = vec![0];
    p.deme_selector_offsets = vec![0];
    let mut n_hooks = 0i64;
    for ev in &events {
        for hook in ev {
            for op in &hook.ops {
                p.op_types.push(op.op_type);
                p.zidx_data.extend_from_slice(&op.zidx);
                p.zidx_offsets.push(p.zidx_data.len() as i64);
                p.age_data.extend_from_slice(&op.ages);
                p.age_offsets.push(p.age_data.len() as i64);
                p.sex_masks.push(op.sex[0]);
                p.sex_masks.push(op.sex[1]);
                p.params.push(op.param);
                for &(t, q) in &op.cond {
                    p.condition_types.push(t);
                    p.condition_params.push(q);
                }
                p.condition_offsets.push(p.condition_types.len() as i64);
                let (src, dst) = op.convert.unwrap_or((-1, -1));
                p.convert_source_z.push(src);
                p.convert_target_z.push(dst);
            }
            p.op_offsets.push(p.op_types.len() as i64);
            p.deme_selector_types.push(hook.deme_type);
            p.deme_selector_data.extend_from_slice(&hook.deme_data);
            p.deme_selector_offsets
                .push(p.deme_selector_data.len() as i64);
        }
        n_hooks += ev.len() as i64;
        p.hook_offsets.push(n_hooks);
    }
    p.n_hooks = n_hooks;
    p
}

/// A full-coverage operation (both sexes, every age, every ztype).
fn eval_op(op_type: i64, param: f64) -> EvalOp {
    EvalOp {
        op_type,
        zidx: vec![0, 1],
        ages: vec![0, 1, 2, 3],
        sex: [true, true],
        param,
        cond: Vec::new(),
        convert: None,
    }
}

/// Wrap one operation in a single-op, wildcard-deme hook.
fn eval_hook(op: EvalOp) -> EvalHook {
    EvalHook {
        ops: vec![op],
        deme_type: 0,
        deme_data: Vec::new(),
    }
}

/// Run `ticks` ticks on the fixture, on the device or the CPU.
fn eval_run(program: HookProgram, ticks: i64, gpu: bool) -> (Vec<f64>, Vec<f64>) {
    let (blueprint, params, genetics) = fixture();
    let (ind, sperm) = initial_state();
    let mut session = make_session(blueprint, params, genetics, ind, sperm);
    session.hooks = program;
    if gpu {
        session.enable_gpu().expect("enable gpu");
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        session
            .run_inner(py, ticks, 0, None, 0)
            .expect("device/cpu run");
    });
    (session.state_ind, session.state_sperm)
}

/// Assert two `(ind, sperm)` pairs match within a relative tolerance.
fn eval_assert_close(
    got: &(Vec<f64>, Vec<f64>),
    want: &(Vec<f64>, Vec<f64>),
    rtol: f32,
    label: &str,
) {
    for (index, (g, w)) in got.0.iter().zip(want.0.iter()).enumerate() {
        let (g, w) = (*g as f32, *w as f32);
        assert!(
            (g - w).abs() <= rtol * w.abs().max(1.0),
            "{label} ind[{index}]: device {g} vs host {w}"
        );
    }
    for (index, (g, w)) in got.1.iter().zip(want.1.iter()).enumerate() {
        let (g, w) = (*g as f32, *w as f32);
        assert!(
            (g - w).abs() <= rtol * w.abs().max(1.0),
            "{label} sperm[{index}]: device {g} vs host {w}"
        );
    }
}

#[test]
fn evaluator_device_hook_each_opcode_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let cases: [(i64, f64); 5] = [(0, 1.5), (1, 3.0), (2, 2.0), (3, 2.0), (4, 0.5)];
    for (op_type, param) in cases {
        for (ticks, rtol) in [(1i64, 1.2e-6f32), (4, 1.2e-5)] {
            let make = || {
                eval_program([
                    vec![eval_hook(eval_op(op_type, param))],
                    vec![],
                    vec![],
                    vec![],
                ])
            };
            let got = eval_run(make(), ticks, true);
            let want = eval_run(make(), ticks, false);
            eval_assert_close(
                &got,
                &want,
                rtol,
                &format!("opcode {op_type} ticks {ticks}"),
            );
        }
    }
}

#[test]
fn evaluator_device_hooks_actually_execute_on_device() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let hooked = eval_run(
        eval_program([vec![eval_hook(eval_op(1, 5.0))], vec![], vec![], vec![]]),
        1,
        true,
    );
    let free = eval_run(HookProgram::default(), 1, true);
    let mut max_rel = 0.0f64;
    for (h, f) in hooked.0.iter().zip(free.0.iter()) {
        if f.abs() > 1.0 {
            max_rel = max_rel.max(((h - f) / f).abs());
        }
    }
    assert!(
        max_rel > 0.01,
        "device executed no hook effect (max relative change {max_rel})"
    );
}

#[test]
fn evaluator_device_hook_conditions_gate_correctly() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // ADD(4) only on tick 2.
    let make_fire = || {
        let mut op = eval_op(2, 4.0);
        op.cond = vec![(1, 2)];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(make_fire(), 4, true);
    let want = eval_run(make_fire(), 4, false);
    eval_assert_close(&got, &want, 1.2e-5, "tick==2 condition");

    // A never-true condition must leave the device state bit-identical to a
    // hook-free device run, proving the condition is evaluated rather than
    // ignored.
    let make_never = || {
        let mut op = eval_op(2, 4.0);
        op.cond = vec![(1, 99)];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let never = eval_run(make_never(), 4, true);
    let free = eval_run(HookProgram::default(), 4, true);
    assert_eq!(never.0, free.0, "never-true condition mutated ind");
    assert_eq!(never.1, free.1, "never-true condition mutated sperm");

    // MOD condition: fire on every even tick.
    let make_mod = || {
        let mut op = eval_op(2, 1.0);
        op.cond = vec![(2, 2)];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(make_mod(), 5, true);
    let want = eval_run(make_mod(), 5, false);
    eval_assert_close(&got, &want, 1.2e-5, "tick%2==0 condition");

    // RPN: tick>=1 AND NOT(tick==2).
    let make_rpn = || {
        let mut op = eval_op(2, 4.0);
        op.cond = vec![(3, 1), (1, 2), (102, 0), (100, 0)];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(make_rpn(), 4, true);
    let want = eval_run(make_rpn(), 4, false);
    eval_assert_close(&got, &want, 1.2e-5, "RPN AND/NOT condition");

    // OR: tick<1 or tick==3.
    let make_or = || {
        let mut op = eval_op(2, 4.0);
        op.cond = vec![(4, 1), (1, 3), (101, 0)];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(make_or(), 4, true);
    let want = eval_run(make_or(), 4, false);
    eval_assert_close(&got, &want, 1.2e-5, "RPN OR condition");
}

#[test]
fn evaluator_device_hook_selectors_match_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // Female-only KILL.
    let female = || {
        let mut op = eval_op(4, 0.5);
        op.sex = [true, false];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(female(), 2, true);
    let want = eval_run(female(), 2, false);
    eval_assert_close(&got, &want, 1.2e-5, "female-only KILL");

    // Male-only SCALE.
    let male = || {
        let mut op = eval_op(0, 0.5);
        op.sex = [false, true];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(male(), 2, true);
    let want = eval_run(male(), 2, false);
    eval_assert_close(&got, &want, 1.2e-5, "male-only SCALE");

    // Age and genotype subsets.
    let subset = || {
        let mut op = eval_op(3, 1.0);
        op.zidx = vec![1];
        op.ages = vec![1, 2];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(subset(), 2, true);
    let want = eval_run(subset(), 2, false);
    eval_assert_close(&got, &want, 1.2e-5, "zidx/age subset SUBTRACT");

    // No sex selected: the op is inert, so the device must be bit-identical to
    // a hook-free device run.
    let none = || {
        let mut op = eval_op(4, 0.9);
        op.sex = [false, false];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let inert = eval_run(none(), 2, true);
    let free = eval_run(HookProgram::default(), 2, true);
    assert_eq!(inert.0, free.0, "empty sex mask mutated ind");
    assert_eq!(inert.1, free.1, "empty sex mask mutated sperm");
}

#[test]
fn evaluator_device_hook_deme_selector_matches_host() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // Selector [0] matches the single panmictic deme.
    let matches = || {
        let mut hook = eval_hook(eval_op(1, 5.0));
        hook.deme_type = 1;
        hook.deme_data = vec![0];
        eval_program([vec![hook], vec![], vec![], vec![]])
    };
    let got = eval_run(matches(), 1, true);
    let want = eval_run(matches(), 1, false);
    eval_assert_close(&got, &want, 1.2e-6, "deme 0 matcher");

    // Selector [1] never matches in a panmictic run.
    let misses = || {
        let mut hook = eval_hook(eval_op(1, 5.0));
        hook.deme_type = 1;
        hook.deme_data = vec![1];
        eval_program([vec![hook], vec![], vec![], vec![]])
    };
    let missed = eval_run(misses(), 1, true);
    let free = eval_run(HookProgram::default(), 1, true);
    assert_eq!(missed.0, free.0, "non-matching deme selector ran the hook");
}

#[test]
fn evaluator_device_hook_female_sperm_scaling_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // A female-only reduction must scale stored sperm and virgin margin.
    let make = || {
        let mut op = eval_op(4, 0.5);
        op.sex = [true, false];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let got = eval_run(make(), 1, true);
    let want = eval_run(make(), 1, false);
    eval_assert_close(&got, &want, 1.2e-6, "female KILL sperm scaling");

    // Sperm is materially reduced versus a male-only KILL (which leaves the
    // female axis and its sperm untouched).
    let male_only = || {
        let mut op = eval_op(4, 0.5);
        op.sex = [false, true];
        eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
    };
    let male = eval_run(male_only(), 1, true);
    let sperm_delta: f64 = got
        .1
        .iter()
        .zip(male.1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        sperm_delta > 1.0,
        "female KILL did not scale sperm relative to male-only KILL (delta {sperm_delta})"
    );
}

#[test]
fn evaluator_device_hook_convert_matches_cpu_and_conserves() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    for (src, dst, prob) in [(0i64, 1i64, 0.3f64), (1, 0, 0.4)] {
        let make = || {
            let mut op = eval_op(11, prob);
            op.convert = Some((src, dst));
            eval_program([vec![eval_hook(op)], vec![], vec![], vec![]])
        };
        let got = eval_run(make(), 1, true);
        let want = eval_run(make(), 1, false);
        eval_assert_close(&got, &want, 1.2e-6, &format!("convert z{src}->z{dst}"));
        let g_total: f64 = got.0.iter().sum::<f64>() + got.1.iter().sum::<f64>();
        let c_total: f64 = want.0.iter().sum::<f64>() + want.1.iter().sum::<f64>();
        assert!(
            (g_total - c_total).abs() <= 1e-3 * c_total.abs().max(1.0),
            "convert z{src}->z{dst} totals: device {g_total} vs host {c_total}"
        );
    }
}

#[test]
fn evaluator_device_hook_multi_event_order_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    let make = || {
        let scale = eval_op(0, 0.9);
        let add = eval_op(2, 2.0);
        let kill_z0 = {
            let mut op = eval_op(4, 0.5);
            op.zidx = vec![0];
            op
        };
        let set_z1 = {
            let mut op = eval_op(1, 1.0);
            op.zidx = vec![1];
            op
        };
        eval_program([
            vec![eval_hook(scale)],
            vec![eval_hook(add)],
            vec![EvalHook {
                ops: vec![kill_z0, set_z1],
                deme_type: 0,
                deme_data: vec![],
            }],
            vec![],
        ])
    };
    let got = eval_run(make(), 4, true);
    let want = eval_run(make(), 4, false);
    eval_assert_close(&got, &want, 1.2e-5, "multi-event/multi-op ordering");
}

#[test]
fn evaluator_device_hook_eligibility_rejects_unsupported() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (ind, sperm) = initial_state();

        // Every declarative opcode is now device-supported, including SAMPLE.
        if hardware_required() {
            for op_type in [0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
                let (blueprint, params, genetics) = fixture();
                let mut session =
                    make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
                session.hooks = eval_program([
                    vec![eval_hook(eval_op(op_type, 1.0))],
                    vec![],
                    vec![],
                    vec![],
                ]);
                assert!(
                    session.enable_gpu().is_ok(),
                    "opcode {op_type} must be accepted"
                );
            }

            // Hooks on a stochastic model are accepted from P7.3.
            let (mut blueprint, params, genetics) = fixture();
            blueprint.stochastic = true;
            let mut stochastic =
                make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
            stochastic.hooks =
                eval_program([vec![eval_hook(eval_op(0, 0.5))], vec![], vec![], vec![]]);
            assert!(
                stochastic.enable_gpu().is_ok(),
                "stochastic hooks must be accepted"
            );
        }

        // Python callbacks are rejected even without declarative ops.
        let (blueprint, params, genetics) = fixture();
        let mut callbacks = make_session(blueprint, params, genetics, ind.clone(), sperm.clone());
        callbacks.hooks.python_callbacks = vec![vec![py.None()]];
        assert!(
            callbacks.enable_gpu().is_err(),
            "python callbacks must be rejected"
        );

        // The supported deterministic program is accepted.
        let (blueprint, params, genetics) = fixture();
        let mut ok = make_session(blueprint, params, genetics, ind, sperm);
        ok.hooks = eval_program([vec![eval_hook(eval_op(0, 0.5))], vec![], vec![], vec![]]);
        assert!(
            ok.enable_gpu().is_ok(),
            "supported deterministic hooks must be accepted"
        );
    });
}

#[test]
fn session_device_hooks_enabled_after_cpu_ticks_align_tick() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // Late SET(3.0) gated on `tick >= 3`: it must fire on tick 3 whichever
        // engine advances that tick.
        let build = || {
            let mut op = eval_op(1, 3.0);
            op.cond = vec![(3, 3)];
            eval_program([Vec::new(), Vec::new(), vec![eval_hook(op)], Vec::new()])
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();

        // Two CPU ticks, then switch to the device for two more.
        let mut mixed = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        mixed.hooks = build();
        mixed.run_inner(py, 2, 0, None, 0).expect("cpu warmup");
        mixed.enable_gpu().expect("enable gpu");
        mixed
            .run_inner(py, 2, 0, None, 0)
            .expect("gpu continuation");

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.hooks = build();
        reference
            .run_inner(py, 4, 0, None, 0)
            .expect("cpu reference");

        eval_assert_close(
            &(mixed.state_ind, mixed.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "hooks enabled after cpu ticks",
        );
    });
}

#[test]
fn session_device_hook_program_refresh_reuploads() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let early_set = |value: f64| {
            eval_program([
                Vec::new(),
                vec![eval_hook(eval_op(1, value))],
                Vec::new(),
                Vec::new(),
            ])
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();

        let mut session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        session.hooks = early_set(7.0);
        session.enable_gpu().expect("enable gpu");
        // Replacing the program after enabling must reach the device.
        super::install_hook_program(&mut session, early_set(1.0)).expect("refresh program");
        // A python-callback replacement is still rejected while active.
        let mut rejected = eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(0, 0.5))],
            Vec::new(),
            Vec::new(),
        ]);
        rejected.python_callbacks = vec![vec![py.None()]];
        assert!(
            super::install_hook_program(&mut session, rejected).is_err(),
            "callback refresh must be rejected"
        );
        session.run_inner(py, 2, 0, None, 0).expect("gpu run");

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.hooks = early_set(1.0);
        reference.run_inner(py, 2, 0, None, 0).expect("cpu run");

        eval_assert_close(
            &(session.state_ind, session.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "refreshed hook program",
        );
    });
}

// ---------------------------------------------------------------------------
// Independent evaluator tests for the §60 tick-alignment / CSR-refresh fixes.
// ---------------------------------------------------------------------------

/// A `tick % 2 == 0` ADD must fire on the same session ticks after a CPU warmup
/// switches the session to the device mid-run.
#[test]
fn evaluator_device_tick_alignment_matches_cpu_after_warmup() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let build = || {
            let mut op = eval_op(2, 5.0);
            op.cond = vec![(2, 2)];
            eval_program([Vec::new(), vec![eval_hook(op)], Vec::new(), Vec::new()])
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();

        let mut mixed = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        mixed.hooks = build();
        mixed.run_inner(py, 3, 0, None, 0).expect("cpu warmup");
        assert_eq!(mixed.state_tick, 3);
        mixed.enable_gpu().expect("enable gpu");
        mixed
            .run_inner(py, 3, 0, None, 0)
            .expect("gpu continuation");

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.hooks = build();
        reference
            .run_inner(py, 6, 0, None, 0)
            .expect("cpu reference");

        eval_assert_close(
            &(mixed.state_ind, mixed.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "tick%2 alignment after warmup",
        );
    });
}

/// A rejected refresh must leave both the host program and the device program
/// untouched, and the device must keep running the previously installed one.
#[test]
fn evaluator_device_rejected_program_refresh_is_atomic() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let set = |value: f64| {
            eval_program([
                Vec::new(),
                vec![eval_hook(eval_op(1, value))],
                Vec::new(),
                Vec::new(),
            ])
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let mut session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        session.hooks = set(7.0);
        session.enable_gpu().expect("enable gpu");

        let host_before = session.hooks.op_types.clone();
        let mut rejected_program = eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(0, 0.5))],
            Vec::new(),
            Vec::new(),
        ]);
        rejected_program.python_callbacks = vec![vec![py.None()]];
        let rejected = super::install_hook_program(&mut session, rejected_program);
        assert!(rejected.is_err(), "callback refresh must be rejected");
        assert_eq!(
            session.hooks.op_types, host_before,
            "rejected refresh polluted the host program"
        );

        session.run_inner(py, 2, 0, None, 0).expect("gpu run");
        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.hooks = set(7.0);
        reference.run_inner(py, 2, 0, None, 0).expect("cpu run");
        eval_assert_close(
            &(session.state_ind, session.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "device kept the installed program after a rejected refresh",
        );
    });
}

/// Clearing the program on a live device session must make it behave exactly
/// like a hook-free run.
#[test]
fn evaluator_device_program_clear_matches_hook_free() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let set = eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(1, 7.0))],
            Vec::new(),
            Vec::new(),
        ]);
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let mut session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        session.hooks = set;
        session.enable_gpu().expect("enable gpu");
        super::install_hook_program(&mut session, HookProgram::default()).expect("clear program");
        session.run_inner(py, 2, 0, None, 0).expect("gpu run");

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference
            .run_inner(py, 2, 0, None, 0)
            .expect("cpu hook-free");
        eval_assert_close(
            &(session.state_ind, session.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "cleared program behaves hook-free",
        );
    });
}

/// Run `ticks` ticks and return `(state_tick, stopped, ind, sperm)`.
fn eval_run_flow(program: HookProgram, ticks: i64, gpu: bool) -> (i64, bool, Vec<f64>, Vec<f64>) {
    let (blueprint, params, genetics) = fixture();
    let (ind, sperm) = initial_state();
    let mut session = make_session(blueprint, params, genetics, ind, sperm);
    session.hooks = program;
    if gpu {
        session.enable_gpu().expect("enable gpu");
    }
    pyo3::prepare_freethreaded_python();
    let stopped = Python::with_gil(|py| {
        let (_, _, stopped) = session
            .run_inner(py, ticks, 0, None, 0)
            .expect("device/cpu run");
        stopped
    });
    (
        session.state_tick,
        stopped,
        session.state_ind,
        session.state_sperm,
    )
}

#[test]
fn session_device_hook_stop_if_above_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // Early STOP_IF_ABOVE gated on `tick >= 2`: reproduction runs, then the
    // event stops the tick before survival/aging.
    let build = || {
        let mut stop = eval_op(8, 0.0);
        stop.cond = vec![(3, 2)];
        eval_program([Vec::new(), vec![eval_hook(stop)], Vec::new(), Vec::new()])
    };
    let gpu = eval_run_flow(build(), 6, true);
    let cpu = eval_run_flow(build(), 6, false);
    assert!(gpu.1, "device must report stopped");
    assert!(cpu.1, "cpu must report stopped");
    assert_eq!(gpu.0, cpu.0, "stop tick must match");
    assert_eq!(gpu.0, 2, "stop must fire at tick 2");
    eval_assert_close(&(gpu.2, gpu.3), &(cpu.2, cpu.3), 1.2e-5, "stop_if_above");
}

#[test]
fn session_device_hook_stop_if_zero_after_mutation_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // First-event SET(0) then a late STOP_IF_ZERO: the same tick stops after
    // the mutation, exercising intra-hook/op ordering.
    let build = || {
        let zero = eval_op(1, 0.0);
        let stop = eval_op(6, 0.0);
        eval_program([
            vec![eval_hook(zero)],
            Vec::new(),
            vec![eval_hook(stop)],
            Vec::new(),
        ])
    };
    let gpu = eval_run_flow(build(), 3, true);
    let cpu = eval_run_flow(build(), 3, false);
    assert!(gpu.1 && cpu.1, "both must stop");
    assert_eq!(gpu.0, cpu.0, "stop tick must match");
    assert_eq!(gpu.0, 0, "stop must fire at tick 0");
    eval_assert_close(&(gpu.2, gpu.3), &(cpu.2, cpu.3), 1.2e-5, "stop_if_zero");
}

// ---------------------------------------------------------------------------
// Independent evaluator tests for P7.2 device STOP_IF_* gating.
// ---------------------------------------------------------------------------

/// Run a stop-program on both engines and assert the stop tick, stop flag, and
/// resulting partial state agree.
fn eval_stop_pair<F: Fn() -> HookProgram>(
    make: F,
    ticks: i64,
    label: &str,
) -> (i64, bool, Vec<f64>) {
    let gpu = eval_run_flow(make(), ticks, true);
    let cpu = eval_run_flow(make(), ticks, false);
    assert_eq!(
        gpu.0, cpu.0,
        "{label}: stop tick device {} vs cpu {}",
        gpu.0, cpu.0
    );
    assert_eq!(
        gpu.1, cpu.1,
        "{label}: stopped flag device {} vs cpu {}",
        gpu.1, cpu.1
    );
    eval_assert_close(
        &(gpu.2.clone(), gpu.3.clone()),
        &(cpu.2.clone(), cpu.3.clone()),
        1.2e-5,
        label,
    );
    (gpu.0, gpu.1, cpu.2)
}

#[test]
fn evaluator_device_stop_above_below_zero_extinction_match_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();

    // early STOP_IF_ABOVE(0): the nonzero state stops tick 0 after reproduction.
    let above = || {
        eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(8, 0.0))],
            Vec::new(),
            Vec::new(),
        ])
    };
    let (tick, stopped, _) = eval_stop_pair(above, 5, "early stop_if_above");
    assert!(stopped, "stop_if_above must fire");
    assert_eq!(tick, 0);

    // first STOP_IF_BELOW(1e9) gated on tick>=1: stops before reproduction at tick 1.
    let below = || {
        let mut op = eval_op(7, 1e9);
        op.cond = vec![(3, 1)];
        eval_program([vec![eval_hook(op)], Vec::new(), Vec::new(), Vec::new()])
    };
    let (tick, stopped, _) = eval_stop_pair(below, 5, "first stop_if_below");
    assert!(stopped, "stop_if_below must fire");
    assert_eq!(tick, 1);

    // first SET(0) then late STOP_IF_ZERO: zeros the state, then stops at tick 0.
    let zero = || {
        eval_program([
            vec![eval_hook(eval_op(1, 0.0))],
            Vec::new(),
            vec![eval_hook(eval_op(6, 0.0))],
            Vec::new(),
        ])
    };
    let (tick, stopped, _) = eval_stop_pair(zero, 5, "late stop_if_zero");
    assert!(stopped, "stop_if_zero must fire");
    assert_eq!(tick, 0);

    // first SET(0) then early STOP_IF_EXTINCTION: the all-zero state stops at tick 0.
    let extinct = || {
        eval_program([
            vec![eval_hook(eval_op(1, 0.0))],
            vec![eval_hook(eval_op(9, 0.0))],
            Vec::new(),
            Vec::new(),
        ])
    };
    let (tick, stopped, _) = eval_stop_pair(extinct, 5, "early stop_if_extinction");
    assert!(stopped, "stop_if_extinction must fire");
    assert_eq!(tick, 0);

    // late STOP_IF_ABOVE gated on tick==3: no stop before tick 3.
    let late = || {
        let mut op = eval_op(8, 0.0);
        op.cond = vec![(1, 3)];
        eval_program([Vec::new(), Vec::new(), vec![eval_hook(op)], Vec::new()])
    };
    let (tick, stopped, _) = eval_stop_pair(late, 6, "late stop_if_above tick==3");
    assert!(stopped, "late stop_if_above must fire");
    assert_eq!(tick, 3);
}

#[test]
fn evaluator_device_stop_selector_subset_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();

    // A female age-1 z0 subset is below a huge threshold: stops at tick 0.
    let subset = || {
        let mut op = eval_op(7, 1e9);
        op.sex = [true, false];
        op.ages = vec![1];
        op.zidx = vec![0];
        eval_program([Vec::new(), vec![eval_hook(op)], Vec::new(), Vec::new()])
    };
    let (tick, stopped, _) = eval_stop_pair(subset, 4, "subset stop_if_below");
    assert!(stopped && tick == 0);

    // The same subset above a huge threshold never fires: parity must hold and
    // the run must not be reported stopped.
    let no_fire = || {
        let mut op = eval_op(8, 1e9);
        op.sex = [true, false];
        op.ages = vec![1];
        op.zidx = vec![0];
        eval_program([Vec::new(), vec![eval_hook(op)], Vec::new(), Vec::new()])
    };
    let (tick, stopped, _) = eval_stop_pair(no_fire, 4, "subset stop_if_above no fire");
    assert!(!stopped, "threshold must not be exceeded");
    assert_eq!(tick, 4);
}

#[test]
fn evaluator_device_stop_aborts_later_hooks_and_events() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    // Event 1 holds two hooks: the first zeroes z0 and then stops; the second
    // would set z1 to 100. The stop must abort the second hook as well as every
    // later event.
    let build = || {
        let killer = EvalHook {
            ops: vec![
                {
                    let mut zero = eval_op(1, 0.0);
                    zero.zidx = vec![0];
                    zero
                },
                eval_op(7, 1e9),
            ],
            deme_type: 0,
            deme_data: Vec::new(),
        };
        let survivor = EvalHook {
            ops: vec![{
                let mut set = eval_op(1, 100.0);
                set.zidx = vec![1];
                set
            }],
            deme_type: 0,
            deme_data: Vec::new(),
        };
        eval_program([
            Vec::new(),
            vec![killer, survivor],
            vec![eval_hook(eval_op(1, 999.0))],
            Vec::new(),
        ])
    };
    let (tick, stopped, cpu_ind) = eval_stop_pair(build, 3, "abort later hooks");
    assert!(stopped && tick == 0, "must stop at tick 0");
    // The later event's SET(999) must not have run: no cell is 999.
    assert!(
        cpu_ind.iter().all(|value| *value < 500.0),
        "a later event ran after the stop"
    );
}

#[test]
fn evaluator_device_stop_flag_does_not_leak_across_runs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // A stop op that can never fire (state stays positive): the run must
        // complete all ticks and never report stopped.
        let build = || {
            let mut op = eval_op(6, 0.0); // STOP_IF_ZERO
            op.cond = vec![(1, 0)]; // only at tick 0, where the state is nonzero
            eval_program([vec![eval_hook(op)], Vec::new(), Vec::new(), Vec::new()])
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let mut session = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        session.hooks = build();
        session.enable_gpu().expect("enable gpu");
        for _ in 0..3 {
            let (_, _, stopped) = session.run_inner(py, 2, 0, None, 0).expect("device run");
            assert!(!stopped, "stop flag leaked into a later run");
        }
        assert_eq!(session.state_tick, 6);

        let mut reference = make_session(blueprint, params, genetics, ind, sperm);
        reference.hooks = build();
        reference.run_inner(py, 6, 0, None, 0).expect("cpu run");
        eval_assert_close(
            &(session.state_ind, session.state_sperm),
            &(reference.state_ind, reference.state_sperm),
            1.2e-5,
            "repeated stop-program runs",
        );
    });
}

#[test]
fn session_device_hook_set_param_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // First-event `SET_PARAM(carrying_capacity, K * 0.5)` escalating on `first`
    // every tick, which later reproduction/survival stages must observe.
    let build = || {
        let mut p = eval_program([
            vec![eval_hook(eval_op(10, 0.0))],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ]);
        p.sp_param_ids = vec![0];
        p.sp_every = vec![1];
        p.sp_start = vec![0];
        p.rpn_offsets = vec![0, 3];
        p.rpn_kinds = vec![1, 0, 4];
        p.rpn_payload = vec![0, 0, 0];
        p.sp_literals = vec![0.5];
        p.has_set_param = true;
        p
    };
    let gpu = eval_run_flow(build(), 3, true);
    let cpu = eval_run_flow(build(), 3, false);
    assert_eq!(gpu.0, cpu.0, "tick must match");

    // The hook must change the trajectory relative to a hook-free device run.
    let plain = eval_run_flow(
        eval_program([Vec::new(), Vec::new(), Vec::new(), Vec::new()]),
        3,
        true,
    );
    let changed = gpu
        .2
        .iter()
        .zip(plain.2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3 * b.abs().max(1.0));
    assert!(changed, "set_param must change the trajectory");

    eval_assert_close(&(gpu.2, gpu.3), &(cpu.2, cpu.3), 1.2e-5, "set_param");
}

#[test]
fn session_device_stochastic_hooks_reproducible_and_execute() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // A SAMPLE hook on a stochastic model: sampling uses the device RNG.
    let build = || {
        eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(5, 10.0))],
            Vec::new(),
            Vec::new(),
        ])
    };
    let stochastic_fixture = || {
        let (mut blueprint, params, genetics) = fixture();
        blueprint.stochastic = true;
        (blueprint, params, genetics)
    };
    let run = |gpu: bool| -> (i64, bool, Vec<f64>, Vec<f64>) {
        let (blueprint, params, genetics) = stochastic_fixture();
        let (ind, sperm) = initial_state();
        let mut session = make_session(blueprint, params, genetics, ind, sperm);
        session.hooks = build();
        if gpu {
            session.enable_gpu().expect("enable gpu");
        }
        pyo3::prepare_freethreaded_python();
        let stopped = Python::with_gil(|py| {
            session
                .run_inner(py, 4, 0, None, 0)
                .expect("stochastic hook run")
                .2
        });
        (
            session.state_tick,
            stopped,
            session.state_ind,
            session.state_sperm,
        )
    };
    let first = run(true);
    let second = run(true);
    assert_eq!(first.0, second.0);
    assert!(!first.1 && !second.1);
    assert_eq!(
        first.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "same-seed device hook run must be bit-reproducible"
    );
    assert_eq!(
        first.3.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        second.3.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "same-seed device sperm must be bit-reproducible"
    );

    // Hook-free stochastic device run differs, proving the hook executed.
    let (blueprint, params, genetics) = stochastic_fixture();
    let (ind, sperm) = initial_state();
    let mut plain_session = make_session(blueprint, params, genetics, ind, sperm);
    plain_session.enable_gpu().expect("enable gpu");
    Python::with_gil(|py| {
        plain_session
            .run_inner(py, 4, 0, None, 0)
            .expect("plain run");
    });
    let changed = first
        .2
        .iter()
        .zip(plain_session.state_ind.iter())
        .any(|(a, b)| (a - b).abs() > 1e-3 * b.abs().max(1.0));
    assert!(changed, "SAMPLE must change the stochastic trajectory");
}

#[test]
fn evaluator_set_param_committed_on_stop_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // Event 0 runs SET_PARAM(carrying_capacity * 0.5) then STOP_IF_BELOW.
        // The CPU commits the set_param at the event boundary before honoring
        // the stop; the device must retain the same committed ecology.
        let build = || {
            let mut p = eval_program([
                vec![EvalHook {
                    ops: vec![eval_op(10, 0.0), eval_op(7, 1e9)],
                    deme_type: 0,
                    deme_data: Vec::new(),
                }],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ]);
            p.sp_param_ids = vec![0, -1];
            p.sp_every = vec![1, 1];
            p.sp_start = vec![0, 0];
            p.rpn_offsets = vec![0, 3, 3];
            p.rpn_kinds = vec![1, 0, 4];
            p.rpn_payload = vec![0, 0, 0];
            p.sp_literals = vec![0.5];
            p.has_set_param = true;
            p
        };
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();

        let mut gpu = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.hooks = build();
        gpu.enable_gpu().expect("enable gpu");
        let (_, _, gpu_stopped) = gpu.run_inner(py, 1, 0, None, 0).expect("gpu run");
        assert!(gpu_stopped, "gpu stop must fire");

        let mut cpu = make_session(blueprint, params, genetics, ind, sperm);
        cpu.hooks = build();
        let (_, _, cpu_stopped) = cpu.run_inner(py, 1, 0, None, 0).expect("cpu run");
        assert!(cpu_stopped, "cpu stop must fire");

        let g = gpu.params.eco_value(0, 0);
        let c = cpu.params.eco_value(0, 0);
        assert!(
            (g - c).abs() <= 1.2e-6 * c.abs().max(1.0),
            "set_param committed on stop diverged: device {g} vs host {c}"
        );
    });
}

// ---------------------------------------------------------------------------
// Independent evaluator tests for the §66 set_param-on-stop fix and the §65.5
// stochastic-equivalence sweep.
// ---------------------------------------------------------------------------

/// Build a session with an explicit RNG seed (both engines).
fn make_session_seeded(
    blueprint: Blueprint,
    params: EcologyParams,
    genetics: GeneticsTensors,
    seed: u64,
    state_ind: Vec<f64>,
    state_sperm: Vec<f64>,
) -> AgeStructuredSession {
    AgeStructuredSession::assemble(blueprint, params, genetics, seed, state_ind, state_sperm)
}

/// `SET_PARAM(carrying_capacity * 0.5)` followed by a `STOP_IF_BELOW` that
/// always fires, placed at one event.
fn setparam_stop_program(event: usize, stop_type: i64) -> HookProgram {
    let mut events: [Vec<EvalHook>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    events[event] = vec![EvalHook {
        ops: vec![eval_op(10, 0.0), eval_op(stop_type, 1e9)],
        deme_type: 0,
        deme_data: Vec::new(),
    }];
    let mut p = eval_program(events);
    p.sp_param_ids = vec![0, -1];
    p.sp_every = vec![1, 1];
    p.sp_start = vec![0, 0];
    p.rpn_offsets = vec![0, 3, 3];
    p.rpn_kinds = vec![1, 0, 4];
    p.rpn_payload = vec![0, 0, 0];
    p.sp_literals = vec![0.5];
    p.has_set_param = true;
    p
}

#[test]
fn evaluator_set_param_committed_on_stop_middle_and_late_events() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        for event in [1usize, 2] {
            let (blueprint, params, genetics) = fixture();
            let (ind, sperm) = initial_state();
            let mut gpu = make_session(
                blueprint.clone(),
                params.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            gpu.hooks = setparam_stop_program(event, 7);
            gpu.enable_gpu().expect("enable gpu");
            let (_, _, gpu_stopped) = gpu.run_inner(py, 1, 0, None, 0).expect("gpu run");

            let mut cpu = make_session(blueprint, params, genetics, ind, sperm);
            cpu.hooks = setparam_stop_program(event, 7);
            let (_, _, cpu_stopped) = cpu.run_inner(py, 1, 0, None, 0).expect("cpu run");

            assert!(gpu_stopped && cpu_stopped, "event {event}: both must stop");
            assert_eq!(gpu.state_tick, cpu.state_tick, "event {event}: tick");
            let g = gpu.params.eco_value(0, 0);
            let c = cpu.params.eco_value(0, 0);
            assert!(
                (g - c).abs() <= 1.2e-6 * c.abs().max(1.0),
                "event {event}: committed set_param diverged device {g} vs host {c}"
            );
            eval_assert_close(
                &(gpu.state_ind.clone(), gpu.state_sperm.clone()),
                &(cpu.state_ind.clone(), cpu.state_sperm.clone()),
                1.2e-5,
                &format!("event {event} partial state"),
            );
        }
    });
}

#[test]
fn evaluator_set_param_committed_on_stop_stochastic_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (mut blueprint, params, genetics) = fixture();
        blueprint.stochastic = true;
        let (ind, sperm) = initial_state();
        let program = setparam_stop_program(0, 7);

        let mut gpu = make_session_seeded(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            7,
            ind.clone(),
            sperm.clone(),
        );
        gpu.hooks = program;
        gpu.enable_gpu().expect("enable gpu");
        let (_, _, gpu_stopped) = gpu.run_inner(py, 1, 0, None, 0).expect("gpu run");

        let mut cpu = make_session_seeded(blueprint, params, genetics, 7, ind, sperm);
        cpu.hooks = setparam_stop_program(0, 7);
        let (_, _, cpu_stopped) = cpu.run_inner(py, 1, 0, None, 0).expect("cpu run");

        assert!(gpu_stopped && cpu_stopped, "both must stop");
        // The set_param write is deterministic; stochastic state may differ.
        let g = gpu.params.eco_value(0, 0);
        let c = cpu.params.eco_value(0, 0);
        assert!(
            (g - c).abs() <= 1.2e-6 * c.abs().max(1.0),
            "stochastic stop: committed set_param diverged device {g} vs host {c}"
        );
    });
}

#[test]
fn evaluator_set_param_persists_across_repeated_stops() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let initial = params.eco_value(0, 0);

        let mut gpu = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.hooks = setparam_stop_program(0, 7);
        gpu.enable_gpu().expect("enable gpu");

        let mut cpu = make_session(blueprint, params, genetics, ind, sperm);
        cpu.hooks = setparam_stop_program(0, 7);

        let mut expected = initial;
        for round in 0..3 {
            let (_, _, gs) = gpu.run_inner(py, 1, 0, None, 0).expect("gpu run");
            let (_, _, cs) = cpu.run_inner(py, 1, 0, None, 0).expect("cpu run");
            assert!(gs && cs, "round {round}: both must stop");
            expected *= 0.5;
            let g = gpu.params.eco_value(0, 0);
            let c = cpu.params.eco_value(0, 0);
            assert!(
                (g - c).abs() <= 1.2e-6 * c.abs().max(1.0),
                "round {round}: device {g} vs host {c}"
            );
            assert!(
                (c - expected).abs() <= 1.2e-6 * expected.abs().max(1.0),
                "round {round}: host {c} != expected {expected}"
            );
        }
    });
}

/// Draw one stochastic-hook trajectory total for one engine/seed.
fn stochastic_total<F: Fn() -> HookProgram>(make: &F, seed: u64, gpu: bool, ticks: i64) -> f64 {
    let (mut blueprint, params, genetics) = fixture();
    blueprint.stochastic = true;
    let (ind, sperm) = initial_state();
    let mut session = make_session_seeded(blueprint, params, genetics, seed, ind, sperm);
    session.hooks = make();
    if gpu {
        session.enable_gpu().expect("enable gpu");
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        session
            .run_inner(py, ticks, 0, None, 0)
            .expect("stochastic run");
    });
    session.state_ind.iter().sum()
}

#[test]
fn evaluator_device_stochastic_hooks_statistically_match_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    // SAMPLE(150) on a stochastic model. The device and host streams differ, so
    // only the distribution must match (§5.2).
    let make = || {
        eval_program([
            Vec::new(),
            vec![eval_hook(eval_op(5, 150.0))],
            Vec::new(),
            Vec::new(),
        ])
    };
    let n = 100u64;
    let cpu: Vec<f64> = (0..n)
        .map(|s| stochastic_total(&make, s, false, 4))
        .collect();
    let gpu: Vec<f64> = (0..n)
        .map(|s| stochastic_total(&make, s, true, 4))
        .collect();
    let mut gpu_pairs: Vec<(u64, f64)> = (0..n).map(|s| (s, gpu[s as usize])).collect();
    gpu_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut cpu_pairs: Vec<(u64, f64)> = (0..n).map(|s| (s, cpu[s as usize])).collect();
    cpu_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("STOCH GPU top: {:?}", &gpu_pairs[..5]);
    eprintln!("STOCH CPU top: {:?}", &cpu_pairs[..5]);

    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let variance = |values: &[f64]| {
        let m = mean(values);
        values.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (values.len() as f64 - 1.0)
    };
    let (mc, mg) = (mean(&cpu), mean(&gpu));
    let (sc, sg) = (variance(&cpu).sqrt(), variance(&gpu).sqrt());
    assert!(sc > 0.0 && sg > 0.0, "seeds did not produce variance");
    let se = (sc * sc / n as f64 + sg * sg / n as f64).sqrt();
    let t = (mg - mc) / se;
    let ratio = sg / sc;
    // Calibration: split each engine's sample in half and compare the halves.
    let half = cpu.len() / 2;
    let cpu_a = &cpu[..half];
    let cpu_b = &cpu[half..];
    let gpu_a = &gpu[..half];
    let gpu_b = &gpu[half..];
    let sd = |values: &[f64]| variance(values).sqrt();
    let control_cc = sd(cpu_a) / sd(cpu_b);
    let control_gg = sd(gpu_a) / sd(gpu_b);
    // A simple two-sample KS statistic on the (sorted) samples.
    let mut sorted_cpu = cpu.clone();
    let mut sorted_gpu = gpu.clone();
    sorted_cpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted_gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut ks = 0.0f64;
    let (mut i, mut j) = (0usize, 0usize);
    while i < sorted_cpu.len() && j < sorted_gpu.len() {
        let cdf_c = (i + 1) as f64 / sorted_cpu.len() as f64;
        let cdf_g = (j + 1) as f64 / sorted_gpu.len() as f64;
        ks = ks.max((cdf_c - cdf_g).abs());
        if sorted_cpu[i] <= sorted_gpu[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    eprintln!(
        "STOCH STAT n={n} cpu_mean={mc:.3} cpu_sd={sc:.3} gpu_mean={mg:.3} gpu_sd={sg:.3} t={t:.2} var_ratio={ratio:.3} control_cc={control_cc:.3} control_gg={control_gg:.3} ks={ks:.3}"
    );
    assert!(
        t.abs() < 4.0,
        "stochastic hook means differ: cpu={mc:.3} gpu={mg:.3} t={t:.2}"
    );
    assert!(
        (0.5..2.0).contains(&ratio),
        "stochastic hook variance ratio out of band: {ratio:.3}"
    );
    // Two-sample KS at n=100: 5% critical value is ~0.19.
    assert!(ks < 0.25, "stochastic hook ECDFs differ: ks={ks:.3}");
}

#[test]
fn session_gpu_particles_match_independent_cpu_runs() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let base = make_session(
            blueprint.clone(),
            params,
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        )
        .params
        .clone();
        let particle = |k: f64, eggs: f64| {
            let mut p = base.clone();
            p.carrying_capacity[0] = k;
            p.eggs_per_female[0] = eggs;
            p
        };
        let particles = vec![
            particle(300.0, 8.0),
            particle(500.0, 10.0),
            particle(800.0, 12.0),
        ];

        let mut gpu = make_session(
            blueprint.clone(),
            base.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.enable_gpu_particles_ecologies(particles.clone())
            .expect("enable particles");
        let (_, ind_flat, sperm_flat) = gpu.run_gpu_particles(py, 3).expect("run particles");
        let ind_values: Vec<f64> = ind_flat.readonly().as_slice().expect("ind slice").to_vec();
        let sperm_values: Vec<f64> = sperm_flat
            .readonly()
            .as_slice()
            .expect("sperm slice")
            .to_vec();

        let ind_block = 2 * 4 * 2;
        let sperm_block = 4 * 2 * 2;
        assert_eq!(ind_values.len(), particles.len() * ind_block);
        assert_eq!(sperm_values.len(), particles.len() * sperm_block);

        for (index, part) in particles.iter().enumerate() {
            let mut cpu = make_session(
                blueprint.clone(),
                part.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            cpu.run_inner(py, 3, 0, None, 0).expect("cpu run");
            for (cell, (got, want)) in ind_values[index * ind_block..(index + 1) * ind_block]
                .iter()
                .zip(cpu.state_ind.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                let tolerance = 1.2e-5f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "particle {index} ind[{cell}]: device {got} vs host {want}"
                );
            }
            for (cell, (got, want)) in sperm_values[index * sperm_block..(index + 1) * sperm_block]
                .iter()
                .zip(cpu.state_sperm.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                let tolerance = 1.2e-5f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "particle {index} sperm[{cell}]: device {got} vs host {want}"
                );
            }
        }

        // Distinct parameters must produce distinct trajectories.
        assert_ne!(
            &ind_values[..ind_block],
            &ind_values[ind_block..2 * ind_block],
            "particles with different parameters must differ"
        );
    });
}

// ---------------------------------------------------------------------------
// Independent evaluator tests for P9 GPU particles (per-particle parameters).
// ---------------------------------------------------------------------------

/// A single-deme ecology with a distinct value in every scalar/vector field,
/// indexed by particle `k`.
fn particle_ecology(base: &EcologyParams, k: usize) -> EcologyParams {
    let mut p = base.clone();
    let s = k as f64;
    p.carrying_capacity[0] = 300.0 + 200.0 * s;
    p.eggs_per_female[0] = 6.0 + 2.0 * s;
    p.sex_ratio[0] = 0.3 + 0.2 * s;
    p.sperm_displacement_rate[0] = 0.05 + 0.05 * s;
    p.low_density_growth_rate[0] = 2.0 + s;
    p.survival_rates = p
        .survival_rates
        .iter()
        .map(|v| v * (0.95 - 0.05 * s))
        .collect();
    p.mating_rates = p
        .mating_rates
        .iter()
        .map(|v| (v * (0.9 - 0.05 * s)).min(1.0))
        .collect();
    p.reproduction_rates = p
        .reproduction_rates
        .iter()
        .map(|v| v * (0.95 - 0.05 * s))
        .collect();
    p.fertility = p.fertility.iter().map(|v| v * (0.95 - 0.03 * s)).collect();
    p.competition_weights = p
        .competition_weights
        .iter()
        .map(|v| v * (0.8 + 0.1 * s))
        .collect();
    p
}

#[test]
fn evaluator_gpu_particles_all_fields_match_independent_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, base, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let particles: Vec<EcologyParams> = (0..3).map(|k| particle_ecology(&base, k)).collect();

        let mut gpu = make_session(
            blueprint.clone(),
            base.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.enable_gpu_particles_ecologies(particles.clone())
            .expect("enable particles");
        let (_, ind_arr, sperm_arr) = gpu.run_gpu_particles(py, 3).expect("run particles");
        let ind_flat: Vec<f64> = ind_arr.readonly().as_slice().expect("ind slice").to_vec();
        let sperm_flat: Vec<f64> = sperm_arr
            .readonly()
            .as_slice()
            .expect("sperm slice")
            .to_vec();

        let ind_block = 2 * 4 * 2;
        let sperm_block = 4 * 2 * 2;
        let mut blocks: Vec<Vec<f64>> = Vec::new();
        for (index, part) in particles.iter().enumerate() {
            let mut cpu = make_session(
                blueprint.clone(),
                part.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            cpu.run_inner(py, 3, 0, None, 0).expect("cpu run");

            let gblock = &ind_flat[index * ind_block..(index + 1) * ind_block];
            for (cell, (got, want)) in gblock.iter().zip(cpu.state_ind.iter()).enumerate() {
                let (got, want) = (*got as f32, *want as f32);
                assert!(
                    (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                    "particle {index} ind[{cell}]: device {got} vs host {want}"
                );
            }
            for (cell, (got, want)) in sperm_flat[index * sperm_block..(index + 1) * sperm_block]
                .iter()
                .zip(cpu.state_sperm.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                assert!(
                    (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                    "particle {index} sperm[{cell}]: device {got} vs host {want}"
                );
            }
            blocks.push(gblock.to_vec());
        }
        assert_ne!(blocks[0], blocks[1], "particle 0 vs 1 must differ");
        assert_ne!(blocks[1], blocks[2], "particle 1 vs 2 must differ");
    });
}

#[test]
fn evaluator_gpu_particles_hooks_and_set_param_match_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        // first: SET_PARAM(carrying_capacity*0.5) then SCALE 0.9, globally.
        let build = || {
            let mut q = HookProgram::default();
            q.n_events = 4;
            q.n_hooks = 1;
            q.hook_offsets = vec![0, 1, 1, 1, 1];
            q.op_offsets = vec![0, 2];
            q.op_types = vec![10, 0];
            q.zidx_offsets = vec![0, 2, 4];
            q.zidx_data = vec![0, 1, 0, 1];
            q.age_offsets = vec![0, 4, 8];
            q.age_data = vec![0, 1, 2, 3, 0, 1, 2, 3];
            q.sex_masks = vec![true, true, true, true];
            q.params = vec![0.0, 0.9];
            q.condition_offsets = vec![0, 0, 0];
            q.deme_selector_types = vec![0];
            q.deme_selector_offsets = vec![0, 0];
            q.convert_source_z = vec![-1, -1];
            q.convert_target_z = vec![-1, -1];
            q.sp_param_ids = vec![0, -1];
            q.sp_every = vec![1, 1];
            q.sp_start = vec![0, 0];
            q.rpn_offsets = vec![0, 3, 3];
            q.rpn_kinds = vec![1, 0, 4];
            q.rpn_payload = vec![0, 0, 0];
            q.sp_literals = vec![0.5];
            q.has_set_param = true;
            q
        };

        let (blueprint, base, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let particles: Vec<EcologyParams> = (0..3).map(|k| particle_ecology(&base, k)).collect();

        let mut gpu = make_session(
            blueprint.clone(),
            base.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.hooks = build();
        gpu.enable_gpu_particles_ecologies(particles.clone())
            .expect("enable particles");
        let (_, ind_arr, sperm_arr) = gpu.run_gpu_particles(py, 2).expect("run particles");
        let ind_flat: Vec<f64> = ind_arr.readonly().as_slice().expect("ind slice").to_vec();
        let sperm_flat: Vec<f64> = sperm_arr
            .readonly()
            .as_slice()
            .expect("sperm slice")
            .to_vec();

        let ind_block = 2 * 4 * 2;
        let sperm_block = 4 * 2 * 2;
        for (index, part) in particles.iter().enumerate() {
            let mut cpu = make_session(
                blueprint.clone(),
                part.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            cpu.hooks = build();
            cpu.run_inner(py, 2, 0, None, 0).expect("cpu run");
            for (cell, (got, want)) in ind_flat[index * ind_block..(index + 1) * ind_block]
                .iter()
                .zip(cpu.state_ind.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                assert!(
                    (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                    "hooked particle {index} ind[{cell}]: device {got} vs host {want}"
                );
            }
            for (cell, (got, want)) in sperm_flat[index * sperm_block..(index + 1) * sperm_block]
                .iter()
                .zip(cpu.state_sperm.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                assert!(
                    (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                    "hooked particle {index} sperm[{cell}]: device {got} vs host {want}"
                );
            }
        }
    });
}

#[test]
fn evaluator_gpu_particles_reject_invalid() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();

        let mut empty = make_session(
            blueprint.clone(),
            params.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        assert!(
            empty.enable_gpu_particles_ecologies(Vec::new()).is_err(),
            "empty particle list must reject"
        );
        assert!(
            empty.run_gpu_particles(py, 1).is_err(),
            "run without enable must reject"
        );

        let (mut nonpan, params2, genetics2) = fixture();
        nonpan.n_demes = 2;
        let mut session = make_session(nonpan, params2, genetics2, ind.clone(), sperm.clone());
        assert!(
            session
                .enable_gpu_particles_ecologies(vec![params.clone()])
                .is_err(),
            "non-panmictic must reject"
        );

        let (blueprint3, params3, genetics3) = fixture();
        let mut bad = params3.clone();
        bad.growth_mode[0] = 5;
        let mut custom = make_session(blueprint3, params3, genetics3, ind.clone(), sperm.clone());
        assert!(
            custom.enable_gpu_particles_ecologies(vec![bad]).is_err(),
            "custom growth must reject"
        );
    });
}

#[test]
fn session_gpu_particles_deme_selector_matches_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let base = make_session(
            blueprint.clone(),
            params,
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        )
        .params
        .clone();
        let particle = |k: f64| {
            let mut p = base.clone();
            p.carrying_capacity[0] = k;
            p
        };
        let particles = vec![particle(300.0), particle(500.0), particle(800.0)];

        // A deme selector `[0]`: every particle is a panmictic model (deme 0),
        // so all particles must scale, matching the independent CPU sessions.
        let program = || {
            let mut hook = eval_hook(eval_op(0, 0.5));
            hook.deme_type = 1;
            hook.deme_data = vec![0];
            eval_program([vec![hook], Vec::new(), Vec::new(), Vec::new()])
        };

        let mut gpu = make_session(
            blueprint.clone(),
            base.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.hooks = program();
        gpu.enable_gpu_particles_ecologies(particles.clone())
            .expect("enable particles");
        let (_, ind_flat, _) = gpu.run_gpu_particles(py, 2).expect("run particles");
        let ind_values: Vec<f64> = ind_flat.readonly().as_slice().expect("ind slice").to_vec();
        let ind_block = 2 * 4 * 2;

        for (index, part) in particles.iter().enumerate() {
            let mut cpu = make_session(
                blueprint.clone(),
                part.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            cpu.hooks = program();
            cpu.run_inner(py, 2, 0, None, 0).expect("cpu run");
            for (cell, (got, want)) in ind_values[index * ind_block..(index + 1) * ind_block]
                .iter()
                .zip(cpu.state_ind.iter())
                .enumerate()
            {
                let (got, want) = (*got as f32, *want as f32);
                let tolerance = 1.2e-5f32 * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= tolerance,
                    "particle {index} ind[{cell}]: device {got} vs host {want}"
                );
            }
        }
    });
}

#[test]
fn evaluator_gpu_particles_deme_selectors_follow_deme_zero() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, base, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let particles: Vec<EcologyParams> = (0..3).map(|k| particle_ecology(&base, k)).collect();

        // (selector type, data, whether the panmictic batch should match deme 0).
        let cases: Vec<(i64, Vec<i64>, bool)> = vec![
            (1, vec![0], true),
            (1, vec![1], false),
            (2, vec![0, 1], true),
            (2, vec![1, 2], false),
            (3, vec![0, 2], true),
        ];
        let ind_block = 2 * 4 * 2;
        let sperm_block = 4 * 2 * 2;
        for (sel_type, sel_data, should_match) in cases {
            let build = || {
                let mut p = eval_program([
                    vec![eval_hook(eval_op(0, 0.8))],
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ]);
                p.deme_selector_types = vec![sel_type];
                p.deme_selector_offsets = vec![0, sel_data.len() as i64];
                p.deme_selector_data = sel_data.clone();
                p
            };

            let mut gpu = make_session(
                blueprint.clone(),
                base.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            gpu.hooks = build();
            gpu.enable_gpu_particles_ecologies(particles.clone())
                .expect("enable particles");
            let (_, ind_arr, sperm_arr) = gpu.run_gpu_particles(py, 2).expect("run particles");
            let ind_flat: Vec<f64> = ind_arr.readonly().as_slice().expect("ind").to_vec();
            let sperm_flat: Vec<f64> = sperm_arr.readonly().as_slice().expect("sperm").to_vec();

            for (index, part) in particles.iter().enumerate() {
                let mut cpu = make_session(
                    blueprint.clone(),
                    part.clone(),
                    genetics.clone(),
                    ind.clone(),
                    sperm.clone(),
                );
                cpu.hooks = build();
                cpu.run_inner(py, 2, 0, None, 0).expect("cpu run");
                for (cell, (got, want)) in ind_flat[index * ind_block..(index + 1) * ind_block]
                    .iter()
                    .zip(cpu.state_ind.iter())
                    .enumerate()
                {
                    let (got, want) = (*got as f32, *want as f32);
                    assert!(
                        (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                        "sel {sel_type}/{sel_data:?} particle {index} ind[{cell}]: {got} vs {want}"
                    );
                }
                for (cell, (got, want)) in sperm_flat
                    [index * sperm_block..(index + 1) * sperm_block]
                    .iter()
                    .zip(cpu.state_sperm.iter())
                    .enumerate()
                {
                    let (got, want) = (*got as f32, *want as f32);
                    assert!(
                        (got - want).abs() <= 1.2e-5f32 * want.abs().max(1.0),
                        "sel {sel_type}/{sel_data:?} particle {index} sperm[{cell}]: {got} vs {want}"
                    );
                }
            }

            if !should_match {
                let mut free = make_session(
                    blueprint.clone(),
                    base.clone(),
                    genetics.clone(),
                    ind.clone(),
                    sperm.clone(),
                );
                free.enable_gpu_particles_ecologies(particles.clone())
                    .expect("enable particles");
                let (_, fi, fs) = free.run_gpu_particles(py, 2).expect("run particles");
                let fi: Vec<f64> = fi.readonly().as_slice().expect("ind").to_vec();
                let fs: Vec<f64> = fs.readonly().as_slice().expect("sperm").to_vec();
                assert_eq!(
                    ind_flat, fi,
                    "non-matching selector {sel_type}/{sel_data:?} mutated ind"
                );
                assert_eq!(
                    sperm_flat, fs,
                    "non-matching selector {sel_type}/{sel_data:?} mutated sperm"
                );
            }
        }
    });
}

#[test]
fn session_gpu_particles_with_replicates_match_cpu() {
    if !hardware_required() {
        eprintln!("SKIP: NATAL_GPU_REQUIRE=0 disables the hardware gate");
        return;
    }
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let (blueprint, params, genetics) = fixture();
        let (ind, sperm) = initial_state();
        let base = make_session(
            blueprint.clone(),
            params,
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        )
        .params
        .clone();
        let particle = |k: f64| {
            let mut p = base.clone();
            p.carrying_capacity[0] = k;
            p
        };
        let particles = vec![particle(300.0), particle(800.0)];
        let replicates = 3usize;

        let mut gpu = make_session(
            blueprint.clone(),
            base.clone(),
            genetics.clone(),
            ind.clone(),
            sperm.clone(),
        );
        gpu.enable_gpu_particles_ecologies_replicated(particles.clone(), replicates)
            .expect("enable particles");
        let (_, ind_flat, _) = gpu.run_gpu_particles(py, 2).expect("run particles");
        let ind_values: Vec<f64> = ind_flat.readonly().as_slice().expect("ind slice").to_vec();
        let ind_block = 2 * 4 * 2;
        assert_eq!(ind_values.len(), particles.len() * replicates * ind_block);

        for (p_index, part) in particles.iter().enumerate() {
            let mut cpu = make_session(
                blueprint.clone(),
                part.clone(),
                genetics.clone(),
                ind.clone(),
                sperm.clone(),
            );
            cpu.run_inner(py, 2, 0, None, 0).expect("cpu run");
            for replicate in 0..replicates {
                let start = (p_index * replicates + replicate) * ind_block;
                for (cell, (got, want)) in ind_values[start..start + ind_block]
                    .iter()
                    .zip(cpu.state_ind.iter())
                    .enumerate()
                {
                    let (got, want) = (*got as f32, *want as f32);
                    let tolerance = 1.2e-5f32 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tolerance,
                        "particle {p_index} replicate {replicate} ind[{cell}]: \
                         device {got} vs host {want}"
                    );
                }
            }
        }
    });
}
