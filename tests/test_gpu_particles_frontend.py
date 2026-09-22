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


def test_enable_gpu_particles_rejects_zero_replicates() -> None:
    """``n_replicates`` must be positive even before any device work."""
    pop = _build_pop("__gpu_particles_zero_reps__")
    with pytest.raises(ValueError):
        pop.enable_gpu_particles([{"carrying_capacity": 300.0}], n_replicates=0)


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
