"""Frontend GPU particle entry points (per-particle parameters).

Each particle carries its own parameter overrides and all particles advance
together on the device batch axis. These tests skip when the installed
extension has no usable CUDA device so CPU-only hosts stay green.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _build_pop(name: str) -> nt.AgeStructuredPopulation:
    """A small two-allele panmictic age-structured population."""
    species = nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    return (
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
        .build()
    )


def test_enable_gpu_particles_rejects_empty() -> None:
    """An empty particle list is a user error, before any device work."""
    pop = _build_pop("__gpu_particles_empty__")
    with pytest.raises(ValueError):
        pop.enable_gpu_particles([])


def test_run_gpu_particles_without_enable_raises() -> None:
    """Running before enabling is an explicit error, not a silent CPU run."""
    pop = _build_pop("__gpu_particles_missing__")
    with pytest.raises(RuntimeError):
        pop.run_gpu_particles(1)


def test_gpu_particles_shapes_and_distinct_parameters() -> None:
    """Each particle carries its own parameters and returns a stacked state."""
    pop = _build_pop("__gpu_particles_shapes__")
    try:
        pop.enable_gpu_particles(
            [
                {"carrying_capacity": 300.0},
                {"carrying_capacity": 800.0, "eggs_per_female": 12.0},
            ]
        )
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU particles unavailable: {exc}")

    tick, ind, sperm = pop.run_gpu_particles(3)

    assert tick == 3
    assert ind.shape == (2, 1, 2, 4, 3)
    assert sperm.shape == (2, 1, 4, 3, 3)
    assert np.isfinite(ind).all()
    assert np.isfinite(sperm).all()
    assert not np.allclose(ind[0, 0], ind[1, 0]), "particles must diverge"


def test_gpu_particles_with_inner_replicates_shapes() -> None:
    """Each particle can carry an inner replicate axis on the device batch."""
    pop = _build_pop("__gpu_particles_reps__")
    try:
        pop.enable_gpu_particles(
            [{"carrying_capacity": 300.0}, {"carrying_capacity": 800.0}],
            n_replicates=3,
        )
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    tick, ind, sperm = pop.run_gpu_particles(2)
    assert tick == 2
    assert ind.shape == (2, 3, 2, 4, 3)
    assert sperm.shape == (2, 3, 4, 3, 3)
    assert np.isfinite(ind).all()


def test_gpu_particles_per_particle_genetics_diverge() -> None:
    """Per-particle genetics overrides are accepted and change the trajectory."""
    pop = _build_pop("__gpu_particles_genetics__")
    n_fit = 2 * 4 * 3  # (sex, n_ages, n_ztypes) for the builder above.
    try:
        pop.enable_gpu_particles(
            [
                {"viability_fitness": np.ones(n_fit)},
                {"viability_fitness": np.full(n_fit, 0.5)},
            ]
        )
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU particles unavailable: {exc}")
    _, ind, _ = pop.run_gpu_particles(3)
    assert ind.shape == (2, 1, 2, 4, 3)
    assert not np.allclose(ind[0, 0], ind[1, 0]), "genetics variants must diverge"


def test_gpu_particles_chained_ecology_and_genetics_overrides() -> None:
    """Ecology and genetics overrides compose in one particle mapping."""
    pop = _build_pop("__gpu_particles_mixed__")
    try:
        pop.enable_gpu_particles(
            [
                {"carrying_capacity": 300.0, "viability_fitness": np.ones(2 * 4 * 3)},
                {
                    "carrying_capacity": 800.0,
                    "viability_fitness": np.full(2 * 4 * 3, 0.25),
                },
            ]
        )
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    _, ind, _ = pop.run_gpu_particles(2)
    assert ind.shape == (2, 1, 2, 4, 3)
    assert not np.allclose(ind[0, 0], ind[1, 0])


def test_gpu_particles_history_records_per_particle_rows() -> None:
    """Device history records per-particle observation rows with one readback."""
    pop = _build_pop("__gpu_particles_history__")
    try:
        pop.enable_gpu_particles(
            [{"carrying_capacity": 300.0}, {"carrying_capacity": 800.0}],
            n_replicates=2,
        )
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    n_ages = int(pop.state.individual_count.shape[1])
    n_ztypes = int(pop.state.individual_count.shape[2])
    mask = np.zeros((2, 2, n_ages, n_ztypes))
    mask[0, 0, :, :] = 1.0  # all females
    mask[1, 1, :, :] = 1.0  # all males
    tick, _ind, _sperm, history = pop.run_gpu_particles_history(
        4, mask, record_every=1
    )
    assert tick == 4
    # Documented order: (records, n_particles, n_replicates, n_groups, 2, A).
    assert history.shape == (5, 2, 2, 2, 2, n_ages)
    totals = history.sum(axis=(4, 5))  # (records, P, R, groups)
    init = pop.state.individual_count
    assert np.allclose(totals[0, :, :, 0], init[0].sum())
    assert np.allclose(totals[0, :, :, 1], init[1].sum())


def test_gpu_particles_history_rejects_bad_mask_and_missing_enable() -> None:
    """History needs an enabled particle batch and a size-correct mask."""
    pop = _build_pop("__gpu_particles_history_err__")
    with pytest.raises(RuntimeError):
        pop.run_gpu_particles_history(1, np.ones((1, 2, 4, 3)))
    try:
        pop.enable_gpu_particles([{"carrying_capacity": 300.0}])
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    with pytest.raises(ValueError):
        pop.run_gpu_particles_history(1, np.ones(7))


def test_enable_gpu_particles_rejects_zero_replicates() -> None:
    """``n_replicates`` must be positive even before any device work."""
    pop = _build_pop("__gpu_particles_zero_reps__")
    with pytest.raises(ValueError):
        pop.enable_gpu_particles([{"carrying_capacity": 300.0}], n_replicates=0)


def test_gpu_particles_cumulative_tick_and_reuse() -> None:
    """The returned tick is cumulative and re-enabling resets the state."""
    pop = _build_pop("__gpu_particles_reuse__")
    try:
        pop.enable_gpu_particles(
            [{"carrying_capacity": 300.0}, {"carrying_capacity": 800.0}]
        )
        first, _, _ = pop.run_gpu_particles(2)
        second, _, _ = pop.run_gpu_particles(3)
        # Re-enabling with the same batch size reuses the executor and resets.
        pop.enable_gpu_particles(
            [{"carrying_capacity": 100.0}, {"carrying_capacity": 900.0}]
        )
        reset, ind, _ = pop.run_gpu_particles(2)
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    assert first == 2
    assert second == 5, "tick must be cumulative across calls"
    assert reset == 2, "re-enable resets the device tick"
    assert ind.shape == (2, 1, 2, 4, 3)


def test_particles_backend_requires_gpu_build() -> None:
    """A CPU-only extension build gives an actionable particle error."""
    from natal.backends.rust.rust_backend import RustLifecycleBackend

    class _Stub:
        _session = object()

    backend = _Stub()
    with pytest.raises(RuntimeError, match="GPU particle support"):
        RustLifecycleBackend.enable_gpu_particles(backend, [])  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="GPU particle support"):
        RustLifecycleBackend.run_gpu_particles(backend, 1)  # type: ignore[arg-type]


def test_gpu_particles_lazily_initializes_session() -> None:
    """enable_gpu_particles creates the session when none exists yet."""
    pop = _build_pop("__gpu_particles_lazy__")
    pop._rust_lifecycle_backend = None  # noqa: SLF001
    try:
        pop.enable_gpu_particles([{"carrying_capacity": 300.0}])
    except RuntimeError as exc:  # CPU-only host / extension without gpu feature.
        pytest.skip(f"GPU particles unavailable: {exc}")
    assert pop.gpu_status() == "enabled"


def test_gpu_particles_history_axis_order_matches_doc() -> None:
    """`history` is ordered as documented: (records, P, R, n_groups, 2, A).

    With three particles, two replicates and two groups the documented order is
    unambiguous, unlike the all-2 shapes in the existing shapes test.
    """
    pop = _build_pop("__gpu_particles_hist_axes__")
    try:
        pop.enable_gpu_particles(
            [{"carrying_capacity": k} for k in (300.0, 500.0, 800.0)],
            n_replicates=2,
        )
    except RuntimeError as exc:  # CPU-only host.
        pytest.skip(f"GPU particles unavailable: {exc}")
    state = pop.state.individual_count
    n_ages = int(state.shape[1])
    n_ztypes = int(state.shape[2])
    mask = np.zeros((2, 2, n_ages, n_ztypes))
    mask[0, 0, :, :] = 1.0
    mask[1, 1, :, :] = 3.0
    _, _, _, history = pop.run_gpu_particles_history(2, mask, record_every=1)
    assert history.shape == (3, 3, 2, 2, 2, n_ages), history.shape
