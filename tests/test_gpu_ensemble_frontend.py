"""Frontend-level CUDA ensemble entry points.

These tests exercise ``AgeStructuredPopulation.enable_gpu_ensemble`` /
``run_gpu_ensemble``. They skip when the installed extension has no usable
CUDA device so CPU-only hosts stay green.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _build_pop(name: str, *, stochastic: bool = True) -> nt.AgeStructuredPopulation:
    """A small two-allele panmictic age-structured population."""
    species = nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    return (
        nt.AgeStructuredPopulation.setup(
            species=species, name=name, stochastic=stochastic, continuous_sampling=False
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
        .build()
    )


def test_enable_gpu_ensemble_rejects_zero_replicates() -> None:
    """``n_replicates`` must be positive even before any device work."""
    pop = _build_pop("__gpu_ens_zero__")
    with pytest.raises((RuntimeError, ValueError)):
        pop.enable_gpu_ensemble(0)


def test_run_gpu_ensemble_without_enable_raises() -> None:
    """Running before enabling is an explicit error, not a silent CPU run."""
    pop = _build_pop("__gpu_ens_missing__")
    with pytest.raises(RuntimeError):
        pop.run_gpu_ensemble(1)


def test_gpu_ensemble_shapes_and_leaves_cpu_state_untouched() -> None:
    """A successful ensemble returns stacked replicates without advancing CPU."""
    pop = _build_pop("__gpu_ens_shapes__")
    try:
        pop.enable_gpu_ensemble(64)
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU ensemble unavailable: {exc}")

    tick_before = pop.tick
    state_before = pop.state.individual_count.copy()
    tick, ind, sperm = pop.run_gpu_ensemble(3)
    state = pop.state.individual_count
    n_ages = state.shape[1]
    n_ztypes = state.shape[2]

    assert tick == 3
    assert ind.shape == (64, 2, n_ages, n_ztypes)
    assert sperm.shape == (64, n_ages, n_ztypes, n_ztypes)
    assert np.isfinite(ind).all()
    assert np.isfinite(sperm).all()
    # The ensemble is a separate experiment: the population's own timeline
    # and state are not advanced by it.
    assert pop.tick == tick_before
    np.testing.assert_array_equal(pop.state.individual_count, state_before)


def test_enable_gpu_ensemble_lazily_initializes_session() -> None:
    """``enable_gpu_ensemble`` must build the session when none exists yet."""
    pop = _build_pop("__gpu_ens_lazy__")
    pop._rust_lifecycle_backend = None  # pyright: ignore[reportPrivateUsage]
    try:
        pop.enable_gpu_ensemble(16)
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU ensemble unavailable: {exc}")
    assert pop._rust_lifecycle_backend is not None


def test_run_gpu_ensemble_missing_backend_raises() -> None:
    """A population without a native session raises the explicit guard error."""
    pop = _build_pop("__gpu_ens_nobackend__")
    pop._rust_lifecycle_backend = None  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="enable_gpu_ensemble"):
        pop.run_gpu_ensemble(1)


def test_single_population_gpu_path_runs() -> None:
    """``enable_gpu`` / ``gpu_status`` expose the single-population device path."""
    pop = _build_pop("__gpu_single__")
    assert pop.gpu_status() == "disabled"
    try:
        pop.enable_gpu()
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU unavailable: {exc}")
    assert pop.gpu_status() == "enabled"
    pop.run(3)
    assert pop.tick == 3
    assert np.isfinite(pop.state.individual_count).all()


def test_enable_gpu_lazily_initializes_session() -> None:
    """``enable_gpu`` must build the session when none exists yet."""
    pop = _build_pop("__gpu_single_lazy__")
    pop._rust_lifecycle_backend = None  # pyright: ignore[reportPrivateUsage]
    try:
        pop.enable_gpu()
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU unavailable: {exc}")
    assert pop._rust_lifecycle_backend is not None


def test_gpu_status_disabled_without_session() -> None:
    """A population without a native session reports ``disabled``."""
    pop = _build_pop("__gpu_single_nostatus__")
    pop._rust_lifecycle_backend = None  # pyright: ignore[reportPrivateUsage]
    assert pop.gpu_status() == "disabled"


def test_observe_gpu_ensemble_projects_each_replicate() -> None:
    """Ensemble readouts project through the population's Observation."""
    pop = _build_pop("__gpu_obs_ens__")
    try:
        pop.enable_gpu_ensemble(32)
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU ensemble unavailable: {exc}")
    _, ind, _ = pop.run_gpu_ensemble(2)
    observed = pop.observe_gpu_ensemble(ind)
    assert observed.shape[0] == ind.shape[0]
    assert np.isfinite(observed).all()
    expected = pop.observation.apply(ind[0])
    assert observed.shape[1:] == expected.shape
    np.testing.assert_array_equal(observed[0], expected)
    with pytest.raises(ValueError):
        pop.observe_gpu_ensemble(ind[0])
