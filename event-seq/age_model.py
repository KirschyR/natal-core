"""Simplified single-population age-structured model for event-order study.

No space and no migration.  Structure, survival, reproduction, density
regulation and initial state follow the per-deme state of the M5 spatial model:

* 10 ages: 0-1 juveniles, 2-9 adults (female uses 2-9, male uses 2-6).
* initial WT|WT female/male counts as in ``m5_engine_probe._initial_state``.
* a homing drive is introduced at initialization as male ``Drive|Drive`` at
  age 2 (the first adult age), with count ``10%`` of that age's WT male count,
  **added on top** (WT counts unchanged).  Females are all WT.

One tick is decomposed into three events, executed in a configurable order:

* ``R`` -- reproduction: compute newborn genotype counts from the tick-start
  adults and write them into age 0 (split by sex ratio).
* ``S`` -- density regulation (Beverton-Holt form ``s1``) plus age survival.
* ``A`` -- aging: shift every age class down one, clear age 0.

The canonical orders are ``R-S-A`` (natal's engine order) and ``S-A-R`` (the
reference ``radial_release_optimizer/main.m`` order).  Reproduction always reads
the adults present at the **start** of the tick (before any event), matching
both the reference (which computes newborns before its transition) and natal
(whose reproduction is the first stage).
"""

from __future__ import annotations

import numpy as np

# --- structure --------------------------------------------------------------
F = 8
M = 5
N_AGES = 10
NEW_ADULT_AGE = 2
N_Z = 5  # paper genotype order: [dd, dw, ww, rw, dr]
N_SEX = 2  # 0 female, 1 male

# --- parameters -------------------------------------------------------------
EGGS = 40.0
LAMBDA = 5.0
SF = (F + 1) / 2.0
SEX_RATIO = 0.5

CORD_PURE_W = 2  # ww (WT|WT)
CORD_RELEASE_GENOTYPE = 0  # dd (Drive|Drive)
DRIVE_GENOTYPE = 0  # only dd counts as drive (homozygous)

_FEMALE_AGING = [1.0 - 1.0 / (F - i + 1) for i in range(1, F)]  # 7 values
_MALE_AGING = [1.0 - 1.0 / (M - i + 1) for i in range(1, M)]  # 4 values
FEMALE_SURVIVAL = np.array([1.0, 1.0, *_FEMALE_AGING, 0.0])
MALE_SURVIVAL = np.array([1.0, 1.0, *_MALE_AGING] + [0.0] * (N_AGES - 2 - len(_MALE_AGING)))
SURVIVAL = np.stack([FEMALE_SURVIVAL, MALE_SURVIVAL])  # (2, N_AGES)

INITIAL_FEMALE = np.array(
    [20.0, 0.2222222, 0.2222222, 0.1944444, 0.1666667, 0.1388889,
     0.1111111, 0.0833333, 0.0555556, 0.0277778]
)
INITIAL_MALE = np.array(
    [20.0, 0.2222222, 0.2222222, 0.1777778, 0.1333333, 0.0888889,
     0.0444444, 0.0, 0.0, 0.0]
)
DRIVE_FRACTION_OF_AGE2 = 0.1

# category labels in the 4x2 plotting order
CATEGORIES = ["A_F_W", "A_F_D", "A_M_W", "A_M_D", "J_F_W", "J_F_D", "J_M_W", "J_M_D"]


def homing_mats(dc: float = 1.0, gcr: float = 0.0, herr: float = 0.0,
                ddfitness: float = 1.0) -> np.ndarray:
    """Return the paper homing ``mats`` operator, shape ``(25, 5)``.

    Faithful translation of ``drive_generator.m`` (transpose + female fitness
    cost); ``mats[k * 5 + i, j]`` is offspring ``k`` from female ``i`` x male
    ``j`` in paper order ``[dd, dw, ww, rw, dr]``.
    """
    mat_dd = np.array([
        [1, dc / 2 + 1 / 2, 0, 0, 1 / 2],
        [dc / 2 + 1 / 2, dc * (dc / 2 + 1 / 2) - (dc / 4 + 1 / 4) * (dc + gcr - 1) + gcr * (dc / 4 + 1 / 4), 0, 0, dc / 4 + 1 / 4],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
    ])
    mat_dw = np.array([
        [0, -((herr - 1) ** 2 * (dc + gcr - 1)) / 2, (herr - 1) ** 2, (herr - 1) ** 2 / 2, 0],
        [(herr / 2 - 1 / 2) * (dc + gcr - 1), dc * (herr / 2 - 1 / 2) * (dc + gcr - 1) - (dc + gcr - 1) * ((herr / 2 - 1 / 2) * (dc + gcr - 1) + (dc * (herr - 1) ** 2) / 2 - gcr * (herr / 4 - 1 / 4)) + gcr * (herr / 4 - 1 / 4) * (dc + gcr - 1), (herr / 2 - 1 / 2) * (dc + gcr - 1) + dc * (herr - 1) ** 2 - gcr * (herr / 2 - 1 / 2), (herr / 4 - 1 / 4) * (dc + gcr - 1) + (dc * (herr - 1) ** 2) / 2 - gcr * (herr / 4 - 1 / 4), (herr / 4 - 1 / 4) * (dc + gcr - 1)],
        [1, dc / 2 + 1 / 2, 0, 0, 1 / 2],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
        [0, (herr / 4 - 1 / 4) * (dc + gcr - 1), 1 / 2 - herr / 2, 1 / 4 - herr / 4, 0],
    ])
    mat_ww = np.array([
        [0, 0, 0, 0, 0],
        [0, ((herr - 1) ** 2 * (dc + gcr - 1) ** 2) / 4, -((herr - 1) ** 2 * (dc + gcr - 1)) / 2, -((herr - 1) ** 2 * (dc + gcr - 1)) / 4, 0],
        [0, 1 / 2 - gcr / 2 - dc / 2, 1, 1 / 2, 0],
        [0, 1 / 4 - gcr / 4 - dc / 4, 1 / 2, 1 / 4, 0],
        [0, 0, 0, 0, 0],
    ])
    mat_rw = np.array([
        [0, 0, 0, 0, 0],
        [0, (gcr * (herr / 4 - 1 / 4) - (herr * (herr - 1) * (dc + gcr - 1)) / 2) * (dc + gcr - 1) + gcr * (herr / 4 - 1 / 4) * (dc + gcr - 1), herr * (herr - 1) * (dc + gcr - 1) - gcr * (herr / 2 - 1 / 2), (dc + gcr - 1) * (herr / 4 + (herr * (herr - 1)) / 2 - 1 / 4) - gcr * (herr / 4 - 1 / 4), (herr / 4 - 1 / 4) * (dc + gcr - 1)],
        [0, gcr / 2, 0, 1 / 2, 1 / 2],
        [0, 1 / 4 - dc / 4, 1 / 2, 1 / 2, 1 / 4],
        [0, (herr / 4 - 1 / 4) * (dc + gcr - 1), 1 / 2 - herr / 2, 1 / 4 - herr / 4, 0],
    ])
    mat_dr = np.array([
        [0, gcr / 2 + ((herr - 1) ** 2 / 2 - 1 / 2) * (dc + gcr - 1), 1 - (herr - 1) ** 2, 1 - (herr - 1) ** 2 / 2, 1 / 2],
        [gcr / 2 - (herr * (dc + gcr - 1)) / 2, gcr * (dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1)) + (dc + gcr - 1) * (dc * ((herr - 1) ** 2 / 2 - 1 / 2) - gcr * (herr / 4 + 1 / 4) + (herr * (dc + gcr - 1)) / 2) + dc * (gcr / 2 - (herr * (dc + gcr - 1)) / 2), (gcr * herr) / 2 - dc * ((herr - 1) ** 2 - 1) - (herr * (dc + gcr - 1)) / 2, gcr * (herr / 4 + 1 / 4) - (herr / 4 + 1 / 4) * (dc + gcr - 1) - dc * ((herr - 1) ** 2 / 2 - 1), dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1)],
        [0, 0, 0, 0, 0],
        [1 / 2, dc / 4 + 1 / 4, 0, 0, 1 / 4],
        [1 / 2, dc / 2 + gcr / 2 - (herr / 4 + 1 / 4) * (dc + gcr - 1), herr / 2, herr / 4 + 1 / 4, 1 / 2],
    ])
    f = np.sqrt(ddfitness)
    fit = np.array([f**2, f, 1, 1, f])
    cost = np.tile(fit, (5, 1))
    blocks = [m.T * cost for m in (mat_dd, mat_dw, mat_ww, mat_rw, mat_dr)]
    return np.vstack(blocks)


DEFAULT_MATS = homing_mats()


def initial_state() -> np.ndarray:
    """Return the ``(N_Z, N_SEX, N_AGES)`` initial state."""
    data = np.zeros((N_Z, N_SEX, N_AGES))
    data[CORD_PURE_W, 0, :] = INITIAL_FEMALE
    data[CORD_PURE_W, 1, :] = INITIAL_MALE
    data[CORD_RELEASE_GENOTYPE, 1, NEW_ADULT_AGE] = (
        DRIVE_FRACTION_OF_AGE2 * INITIAL_MALE[NEW_ADULT_AGE]
    )
    return data


def newborn_counts(data: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Newborn genotype counts from the current adult state, shape ``(N_Z,)``."""
    female_adults = data[:, 0, NEW_ADULT_AGE:]
    male_adults = data[:, 1, NEW_ADULT_AGE:]
    total_female = float(female_adults.sum())
    total_male = float(male_adults.sum())
    if total_female <= 0.0 or total_male <= 0.0:
        return np.zeros(N_Z)
    female_freq = female_adults.sum(axis=1) / total_female
    male_freq = male_adults.sum(axis=1) / total_male
    m3 = mats.reshape(N_Z, N_Z, N_Z)  # m3[k, i, j]
    prop = np.einsum("i,j,kij->k", female_freq, male_freq, m3)
    return EGGS * total_female * prop


def event_R(data: np.ndarray, newborns: np.ndarray) -> None:
    """Reproduction: write the newborn cohort into age 0 (split by sex)."""
    data[:, 0, 0] = SEX_RATIO * newborns
    data[:, 1, 0] = (1.0 - SEX_RATIO) * newborns


def event_S(data: np.ndarray) -> None:
    """Density regulation (Beverton-Holt) plus age survival, in place."""
    j0 = float(data[:, :, 0].sum())
    j1 = float(data[:, :, 1].sum())
    r = (j0 + 5.0 * j1) / (EGGS + 5.0 * (4.0 / (F + 1)))
    s1 = LAMBDA / ((LAMBDA - 1.0) * r + 1.0) * 2.0 / SF / EGGS
    data[:, :, 0] *= s1
    data *= SURVIVAL


def event_A(data: np.ndarray) -> None:
    """Aging: shift every age class down one, clear age 0."""
    data[:, :, 1:] = data[:, :, :-1].copy()
    data[:, :, 0] = 0.0


_EVENTS = {"R": event_R, "S": event_S, "A": event_A}


def classify(data: np.ndarray) -> dict[str, float]:
    """Split the state into the eight ``{A,J}_{F,M}_{W,D}`` categories."""
    adults = slice(NEW_ADULT_AGE, N_AGES)
    juveniles = slice(0, NEW_ADULT_AGE)
    out: dict[str, float] = {}
    for sex, sex_name in ((0, "F"), (1, "M")):
        for age_name, ages in (("A", adults), ("J", juveniles)):
            drive = float(data[DRIVE_GENOTYPE, sex, ages].sum())
            total = float(data[:, sex, ages].sum())
            out[f"{age_name}_{sex_name}_W"] = total - drive
            out[f"{age_name}_{sex_name}_D"] = drive
    return out


def run(order: str, n_ticks: int, mats: np.ndarray | None = None) -> list[dict[str, float]]:
    """Run ``n_ticks`` ticks under a given event ``order`` (e.g. ``"R-S-A"``).

    Reproduction reads the tick-start adults (before any event), so the order
    only changes when the newborn cohort is written and when density regulation
    acts on it.
    """
    mats = DEFAULT_MATS if mats is None else mats
    events = order.split("-")
    for name in events:
        if name not in _EVENTS:
            raise ValueError(f"unknown event {name!r} in order {order!r}")
    data = initial_state()
    history = [classify(data)]  # t = 0
    for _ in range(n_ticks):
        newborns = newborn_counts(data, mats)
        for name in events:
            if name == "R":
                event_R(data, newborns)
            else:
                _EVENTS[name](data)
        history.append(classify(data))
    return history
