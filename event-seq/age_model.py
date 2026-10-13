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

# Equilibrium juvenile density in the density ratio ``r = (j0 + 5*j1) / D``:
# age-0 total (= eggs = 40) + 5 * age-1 total (= 4/(F+1)).  Scaling the initial
# state by ``s`` scales this reference by ``s`` too, keeping the normalized
# dynamics unchanged while the population size grows.
DENSITY_REF = EGGS + 5.0 * (4.0 / (F + 1))

# category labels in the 4x3 plotting order: rows = (F_W, F_D, M_W, M_D),
# columns = (Adult, Juvenile, Egg).
CATEGORIES = [
    "A_F_W", "J_F_W", "E_F_W",
    "A_F_D", "J_F_D", "E_F_D",
    "A_M_W", "J_M_W", "E_M_W",
    "A_M_D", "J_M_D", "E_M_D",
]
# categories summed for the "total population" figure (eggs / age 0 excluded).
TOTAL_CATEGORIES = [c for c in CATEGORIES if not c.startswith("E_")]


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


def initial_state(scale: float = 1.0) -> np.ndarray:
    """Return the ``(N_Z, N_SEX, N_AGES)`` initial state, times ``scale``.

    ``scale`` multiplies every per-group count uniformly (a larger population
    with the same age/genotype composition); the density reference scales with
    it, so the normalized dynamics are unchanged.
    """
    data = np.zeros((N_Z, N_SEX, N_AGES))
    data[CORD_PURE_W, 0, :] = scale * INITIAL_FEMALE
    data[CORD_PURE_W, 1, :] = scale * INITIAL_MALE
    data[CORD_RELEASE_GENOTYPE, 1, NEW_ADULT_AGE] = (
        scale * DRIVE_FRACTION_OF_AGE2 * INITIAL_MALE[NEW_ADULT_AGE]
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


def event_S(data: np.ndarray, density_ref: float = DENSITY_REF) -> None:
    """Density regulation (Beverton-Holt) plus age survival, in place."""
    j0 = float(data[:, :, 0].sum())
    j1 = float(data[:, :, 1].sum())
    r = (j0 + 5.0 * j1) / density_ref
    s1 = LAMBDA / ((LAMBDA - 1.0) * r + 1.0) * 2.0 / SF / EGGS
    data[:, :, 0] *= s1
    data *= SURVIVAL


def event_A(data: np.ndarray) -> None:
    """Aging: shift every age class down one, clear age 0."""
    data[:, :, 1:] = data[:, :, :-1].copy()
    data[:, :, 0] = 0.0


_EVENTS = {"R": event_R, "S": event_S, "A": event_A}


def classify(
    data: np.ndarray,
    female_eggs: np.ndarray | None = None,
    male_eggs: np.ndarray | None = None,
) -> dict[str, float]:
    """Split the state into the twelve ``{E,A,J}_{F,M}_{W,D}`` categories.

    ``A`` sums adult ages (>= ``NEW_ADULT_AGE``); ``J`` is **age 1 only**.
    ``E`` is the **newborn cohort produced by reproduction this tick** (per-sex,
    per-genotype arrays ``female_eggs``/``male_eggs``), not the tick-boundary
    age-0 value, so it is phase-independent and comparable across event orders.
    When no cohort is supplied (e.g. at t = 0) age 0 is used.
    """
    category_ages = {
        "A": slice(NEW_ADULT_AGE, N_AGES),
        "J": slice(1, NEW_ADULT_AGE),
    }
    out: dict[str, float] = {}
    for sex, sex_name in ((0, "F"), (1, "M")):
        for age_name, ages in category_ages.items():
            drive = float(data[DRIVE_GENOTYPE, sex, ages].sum())
            total = float(data[:, sex, ages].sum())
            out[f"{age_name}_{sex_name}_W"] = total - drive
            out[f"{age_name}_{sex_name}_D"] = drive
    eggs = {
        "F": data[:, 0, 0] if female_eggs is None else female_eggs,
        "M": data[:, 1, 0] if male_eggs is None else male_eggs,
    }
    for sex_name, egg in eggs.items():
        drive = float(egg[DRIVE_GENOTYPE])
        total = float(np.sum(egg))
        out[f"E_{sex_name}_W"] = total - drive
        out[f"E_{sex_name}_D"] = drive
    return out


def run(
    order: str, n_ticks: int, mats: np.ndarray | None = None, scale: float = 1.0
) -> list[dict[str, float]]:
    """Run ``n_ticks`` ticks under a given event ``order`` (e.g. ``"R-S-A"``).

    Reproduction reads the tick-start adults (before any event), so the order
    only changes when the newborn cohort is written and when density regulation
    acts on it.  ``scale`` multiplies the initial state (and the density
    reference), for a larger population with identical normalized dynamics.
    """
    mats = DEFAULT_MATS if mats is None else mats
    events = order.split("-")
    for name in events:
        if name not in _EVENTS:
            raise ValueError(f"unknown event {name!r} in order {order!r}")
    data = initial_state(scale)
    density_ref = scale * DENSITY_REF
    history = [classify(data)]  # t = 0
    for _ in range(n_ticks):
        newborns = newborn_counts(data, mats)
        eggs_f = SEX_RATIO * newborns
        eggs_m = (1.0 - SEX_RATIO) * newborns
        for name in events:
            if name == "R":
                event_R(data, newborns)
            elif name == "S":
                event_S(data, density_ref)
            else:
                event_A(data)
        history.append(classify(data, eggs_f, eggs_m))
    return history


# --- stochastic mode --------------------------------------------------------
#
# natal's stochastic path uses *continuous* samplers (Beta/Gamma based) rather
# than integer Binomial/Poisson, because the reference counts are fractional
# (adults ~0.22).  The continuous samplers fall back to the deterministic mean
# when n <= 1, so small classes do not collapse while large classes (the ~40
# eggs) carry the noise.  These mirror ``rust/src/kernels/rng.rs``.

_EPS = 1e-12


def _continuous_poisson(rng: np.random.Generator, lam: float) -> float:
    """Continuous Poisson (``Gamma(shape=lambda, scale=1)``)."""
    if lam <= _EPS:
        return 0.0
    return float(rng.gamma(shape=lam, scale=1.0))


def _continuous_binomial(rng: np.random.Generator, n: float, p: float) -> float:
    """Continuous Binomial via a Beta proportion; mean when ``n <= 1``."""
    if p <= _EPS:
        return 0.0
    if p >= 1.0 - _EPS:
        return float(n)
    if n <= 1.0 + _EPS:
        return float(n) * p
    concentration = n - 1.0
    alpha = max(p * concentration, _EPS)
    beta = max((1.0 - p) * concentration, _EPS)
    numerator = rng.gamma(shape=alpha, scale=1.0)
    denominator = rng.gamma(shape=beta, scale=1.0)
    proportion = 0.0 if numerator == 0.0 else numerator / (numerator + denominator)
    return float(proportion * n)


def _continuous_multinomial(rng: np.random.Generator, n: float, p: np.ndarray) -> np.ndarray:
    """Continuous Multinomial via normalized Gamma draws; mean when ``n <= 1``."""
    n = float(n)
    p = np.asarray(p, dtype=float)
    if n <= 1.0 + _EPS:
        return n * p
    concentration = n - 1.0
    draws = np.zeros(len(p))
    for i, prob in enumerate(p):
        alpha = prob * concentration
        if alpha > _EPS:
            draws[i] = rng.gamma(shape=alpha, scale=1.0)
    total = float(draws.sum())
    if total > _EPS:
        return draws * (n / total)
    return n * p


class Simulation:
    """Single-population model with optional natal-style continuous sampling.

    The deterministic path reproduces :func:`run` exactly; the stochastic path
    samples continuous Poisson (egg count), continuous multinomial (offspring
    genotypes and density-recruited juveniles) and continuous binomial (sex
    assignment and survival), matching natal's ``continuous_sampling=True``
    semantics for fractional counts.
    """

    def __init__(self, *, dc: float = 1.0, stochastic: bool = False,
                 seed: int | None = None, scale: float = 1.0) -> None:
        """Create a simulation with the given drive conversion rate and scale."""
        self.mats = homing_mats(dc=dc)
        self.stochastic = stochastic
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.data = initial_state(scale)
        self.density_ref = scale * DENSITY_REF
        self._eggs_f = np.zeros(N_Z)
        self._eggs_m = np.zeros(N_Z)

    def _newborn(self) -> np.ndarray | None:
        data = self.data
        female_adults = data[:, 0, NEW_ADULT_AGE:]
        male_adults = data[:, 1, NEW_ADULT_AGE:]
        total_female = float(female_adults.sum())
        total_male = float(male_adults.sum())
        if total_female <= 0.0 or total_male <= 0.0:
            return None
        female_freq = female_adults.sum(axis=1) / total_female
        male_freq = male_adults.sum(axis=1) / total_male
        m3 = self.mats.reshape(N_Z, N_Z, N_Z)
        prop = np.einsum("i,j,kij->k", female_freq, male_freq, m3)
        if self.stochastic:
            total_eggs = _continuous_poisson(self.rng, EGGS * total_female)
            return _continuous_multinomial(self.rng, total_eggs, prop)
        return EGGS * total_female * prop

    def _event_R(self, newborns: np.ndarray | None) -> None:
        if newborns is None:
            self._eggs_f = np.zeros(N_Z)
            self._eggs_m = np.zeros(N_Z)
            return
        if self.stochastic:
            eggs_f = np.zeros(N_Z)
            eggs_m = np.zeros(N_Z)
            for z in range(N_Z):
                n = float(newborns[z])
                female = _continuous_binomial(self.rng, n, SEX_RATIO)
                eggs_f[z] = female
                eggs_m[z] = n - female
                self.data[z, 0, 0] = female
                self.data[z, 1, 0] = n - female
            self._eggs_f = eggs_f
            self._eggs_m = eggs_m
        else:
            self._eggs_f = SEX_RATIO * newborns
            self._eggs_m = (1.0 - SEX_RATIO) * newborns
            event_R(self.data, newborns)

    def _event_S(self) -> None:
        data = self.data
        if self.stochastic:
            j0 = float(data[:, :, 0].sum())
            j1 = float(data[:, :, 1].sum())
            r = (j0 + 5.0 * j1) / self.density_ref
            s1 = LAMBDA / ((LAMBDA - 1.0) * r + 1.0) * 2.0 / SF / EGGS
            age0 = data[:, :, 0].reshape(-1)
            total0 = float(age0.sum())
            if total0 > 0.0:
                draws = _continuous_multinomial(self.rng, total0 * s1, age0 / total0)
                data[:, :, 0] = draws.reshape(N_Z, N_SEX)
            for z in range(N_Z):
                for sex in range(N_SEX):
                    for age in range(N_AGES):
                        data[z, sex, age] = _continuous_binomial(
                            self.rng, data[z, sex, age], SURVIVAL[sex, age]
                        )
        else:
            event_S(data, self.density_ref)

    def step(self, order: str) -> None:
        """Advance the state by one tick under the given event order."""
        newborns = self._newborn()
        for name in order.split("-"):
            if name == "R":
                self._event_R(newborns)
            elif name == "S":
                self._event_S()
            elif name == "A":
                event_A(self.data)
            else:
                raise ValueError(f"unknown event {name!r} in order {order!r}")

    def run(self, order: str, n_ticks: int) -> list[dict[str, float]]:
        """Run ``n_ticks`` ticks; returns the per-tick category history."""
        history = [classify(self.data)]
        for _ in range(n_ticks):
            self.step(order)
            history.append(classify(self.data, self._eggs_f, self._eggs_m))
        return history
