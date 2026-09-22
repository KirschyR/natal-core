"""Frontend GPU entry points for deterministic declarative hooks (P7.1).

A model whose only hooks are deterministic state mutations
(SCALE/SET/ADD/SUBTRACT/KILL/CONVERT) may now run on the device. These tests
exercise the public ``enable_gpu`` path; they skip when the installed extension
has no usable CUDA device so CPU-only hosts stay green.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _build_hooked_pop(name: str, hook_calls: list | None = None) -> nt.AgeStructuredPopulation:
    """A small deterministic two-allele panmictic population with optional hooks."""
    species = nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    chain = (
        nt.AgeStructuredPopulation.setup(
            species=species, name=name, stochastic=False, continuous_sampling=False
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=8.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.8, 0.6, 0.0],
            male_age_based_survival=[1.0, 0.8, 0.6, 0.0],
        )
        .competition(juvenile_growth_mode="beverton_holt", carrying_capacity=500.0)
    )
    for items, kwargs in hook_calls or []:
        chain = chain.hooks(*items, **kwargs)
    return chain.build()


def test_gpu_hooks_supported_model_runs() -> None:
    """A deterministic declarative hook model enables and runs on the device."""
    pop = _build_hooked_pop(
        "__gpu_hooks_ok__",
        [(([nt.Op.scale(genotypes="*", factor=0.9)],), {"event": "early"})],
    )
    try:
        pop.enable_gpu()
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU unavailable: {exc}")
    assert pop.gpu_status() == "enabled"
    pop.run(3)
    assert pop.tick == 3
    assert np.isfinite(pop.state.individual_count).all()


def test_gpu_hooks_match_cpu() -> None:
    """The device hook interpreter tracks the CPU engine."""
    hooks = [(([nt.Op.scale(genotypes="*", factor=0.9)],), {"event": "early"})]
    gpu_pop = _build_hooked_pop("__gpu_hooks_parity__", hooks)
    try:
        gpu_pop.enable_gpu()
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU unavailable: {exc}")
    gpu_pop.run(3)

    cpu_pop = _build_hooked_pop("__cpu_hooks_parity__", hooks)
    cpu_pop.run(3)

    got = gpu_pop.state.individual_count
    want = cpu_pop.state.individual_count
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1.2e-5, atol=1e-4)


def test_gpu_hooks_unsupported_opcode_rejected() -> None:
    """Sampling hooks are reserved for P7.3 and must be rejected explicitly."""
    pop = _build_hooked_pop(
        "__gpu_hooks_sample__",
        [
            (
                (nt.Op.sample(genotypes="*", size=1),),
                {"event": "early"},
            )
        ],
    )
    with pytest.raises((RuntimeError, ValueError), match="opcode"):
        pop.enable_gpu()


def test_gpu_hooks_stop_gating_matches_cpu() -> None:
    """A device STOP_IF_* stops at the same tick as the CPU engine."""
    hooks = [
        (
            (
                nt.Op.stop_if_above(
                    genotypes="*", threshold=0.0, when="tick >= 2"
                ),
            ),
            {"event": "early"},
        )
    ]
    gpu_pop = _build_hooked_pop("__gpu_hooks_stopgate__", hooks)
    try:
        gpu_pop.enable_gpu()
    except (RuntimeError, ValueError) as exc:  # CPU-only host.
        pytest.skip(f"GPU unavailable: {exc}")
    gpu_pop.run(6)

    cpu_pop = _build_hooked_pop("__cpu_hooks_stopgate__", hooks)
    cpu_pop.run(6)

    assert gpu_pop.tick == cpu_pop.tick
    assert gpu_pop.tick == 2
    np.testing.assert_allclose(
        gpu_pop.state.individual_count,
        cpu_pop.state.individual_count,
        rtol=1.2e-5,
        atol=1e-4,
    )


def test_gpu_hooks_stochastic_model_rejected() -> None:
    """Hooks on a stochastic model stay host-only until P7.3."""
    species = nt.Species.from_dict(
        name="__gpu_hooks_stochastic__",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species,
            name="__gpu_hooks_stochastic__",
            stochastic=True,
            continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
            }
        )
        .hooks(
            nt.Op.scale(genotypes="*", factor=0.9),
            event="early",
        )
        .build()
    )
    with pytest.raises((RuntimeError, ValueError), match="deterministic"):
        pop.enable_gpu()
