"""M5 init check: natal post-reproduction age-0 vs the reference's newborns.

Builds the mapped mosquito model and uses an ``early`` callback hook (fires
after reproduction, before survival/density regulation) to read the age
distribution the engine writes into age 0.  At the equilibrium initial state the
reference writes ``eggs * total_female = 40`` into age 0 (all WT); this script
checks whether natal's reproduction writes the same total, so that the
engine/reference gap can be attributed to the density-regulation timing rather
than to reproduction.

Usage:
    python repro/m5_init_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import natal as nt
from natal.frontend.spatial import HexGrid, batch_setting

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_radial_reference as ref  # noqa: E402
import m5_engine_probe as eng  # noqa: E402

ARENA = 8
CENTER = (ARENA // 2) * ARENA + (ARENA // 2)

captured: dict[int, np.ndarray] = {}


@nt.hook(event="early", priority=0)
def capture(pop):
    """Read the age distribution right after reproduction, for the center deme."""
    if pop.deme_id == CENTER:
        captured[pop.tick] = np.array(pop.metrics.by_age)
    return 0


def main() -> int:
    m = n = ARENA
    kernel = ref.natal_hex_kernel()
    t, h, r = 1, 10.0, 0.0

    species = nt.Species.from_dict(name="M5InitCheck", structure={"chr1": {"loc": ["WT", "Drive"]}})
    drive = nt.HomingDrive(
        name="d", drive_allele="Drive", target_allele="WT", cas9_allele="Drive",
        drive_conversion_rate=1.0, late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0, functional_resistance_ratio=0.0,
        fecundity_scaling=1.0, viability_scaling=1.0,
    )
    female_survival, male_survival = eng._survival()
    mask = ref.release_geometry(m, n, r)
    demes = np.flatnonzero(mask.reshape(-1)).tolist()
    release, _ = eng._release_hook(t, h, demes)

    pop = (
        nt.SpatialPopulation.builder(
            species, n_demes=m * n, topology=HexGrid(rows=m, cols=n), pop_type="age_structured"
        )
        .setup(name="chk", stochastic=False, continuous_sampling=False)
        .age_structure(n_ages=eng.N_AGES, new_adult_age=eng.NEW_ADULT_AGE)
        .initial_state(individual_count=batch_setting(eng._initial_state(m, n)))
        .survival(female_age_based_survival=female_survival, male_age_based_survival=male_survival)
        .reproduction(
            female_age_based_mating_rate=[0.0, 0.0] + [1.0] * (eng.N_AGES - 2),
            male_age_based_mating_rate=[0.0, 0.0] + [1.0] * (eng.N_AGES - 2),
            eggs_per_female=eng.EGGS, sex_ratio=0.5, fixed_egg_count=True,
            sperm_displacement_rate=1.0,
        )
        .competition(
            juvenile_growth_mode="beverton_holt", low_density_growth_rate=5.0,
            age_1_carrying_capacity=eng.K_AGE1, competition_strength=5.0,
        )
        .presets(drive)
        .migration(kernel=kernel, migration_rate=1.0, strategy="kernel", migration_execution="csr")
        .hooks(release, deme=demes)
        .hooks(capture, event="early", deme=[CENTER])
        .build()
    )
    pop.run(n_steps=1, record_every=0)

    if not captured:
        print("no capture of the center deme")
        return 1
    first_key = next(iter(captured))
    ba = captured[first_key]

    ages = ref.model_ages()
    newborns_ref = ref.EGGS * float(ages[ref.CORD_F1 : ref.AGES].sum())
    print(f"natal center deme, AFTER reproduction (tick {first_key}, early hook):")
    print("  by_age =", np.round(ba, 6))
    print("  age0 =", round(float(ba[0]), 6), " age1 =", round(float(ba[1]), 6))
    print("  total =", round(float(ba.sum()), 6))
    print()
    print("reference newborns (age0 written) = eggs*total_female =", round(newborns_ref, 6))
    diff = abs(float(ba[0]) - newborns_ref)
    print("|natal age0 - reference newborns| =", round(diff, 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
