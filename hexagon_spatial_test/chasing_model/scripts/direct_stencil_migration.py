"""Counter-based direct stencil stochastic migration (plan A).

This module implements the CSR-free, two-pass migration operator described in
``DirectStencilMigrationPlan.md``.  Each edge owns an independent
Philox4x32-10 stream keyed by ``(seed, role, tick, plane, source,
destination)``.  Pass 1 accumulates the per-source Gamma total ``S_s``; pass 2
replays the same Gamma draws while gathering into destinations in a fixed
offset order.  Only ``O_s``, ``S_s``, ``Z_s`` and the kernel template are kept,
so memory is ``O(n + kappa**2)`` per plane and no atomics are used.

The module is NumPy-only and does not import natal.  It is intended for the
operator-level prototype and the NumPy CPU demo, not for production code.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-10
_GAMMA_NORMAL_THRESHOLD = 1e8
_RESOLUTION_LIMIT = 2.028240960365167e31

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF
_U32 = np.uint64(32)
_ONE_U64 = np.uint64(1)
_INV_2POW24 = 1.0 / 16777216.0

_PHILOX_M0 = np.uint64(0xD2511F53)
_PHILOX_M1 = np.uint64(0xCD9E8D57)
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85

ROLE_OUTBOUND_A = 11
ROLE_OUTBOUND_B = 12
ROLE_GAMMA = 21


def _splitmix64(value: int) -> int:
    """Return one SplitMix64 output for a 64-bit integer."""
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    mixed = value
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & _MASK64
    return mixed ^ (mixed >> 31)


def philox4x32_10(
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    key0: np.uint32 | np.ndarray,
    key1: np.uint32 | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one Philox4x32-10 block for counter and key words.

    Args:
        c0: Low counter word.
        c1: Second counter word.
        c2: Third counter word.
        c3: High counter word.
        key0: Low key word.
        key1: High key word.

    Returns:
        Four uint32 output words with the broadcast input shape.
    """
    c0 = np.asarray(c0, dtype=np.uint32)
    c1 = np.asarray(c1, dtype=np.uint32)
    c2 = np.asarray(c2, dtype=np.uint32)
    c3 = np.asarray(c3, dtype=np.uint32)
    key0 = np.asarray(key0, dtype=np.uint32)
    key1 = np.asarray(key1, dtype=np.uint32)
    shape = np.broadcast_shapes(c0.shape, c1.shape, c2.shape, c3.shape, key0.shape, key1.shape)
    c0 = np.broadcast_to(c0, shape).copy()
    c1 = np.broadcast_to(c1, shape).copy()
    c2 = np.broadcast_to(c2, shape).copy()
    c3 = np.broadcast_to(c3, shape).copy()
    key0 = np.broadcast_to(key0, shape).copy()
    key1 = np.broadcast_to(key1, shape).copy()

    for _ in range(10):
        product0 = _PHILOX_M0 * c0.astype(np.uint64)
        high0 = (product0 >> _U32).astype(np.uint32)
        low0 = (product0 & np.uint64(_MASK32)).astype(np.uint32)
        product1 = _PHILOX_M1 * c2.astype(np.uint64)
        high1 = (product1 >> _U32).astype(np.uint32)
        low1 = (product1 & np.uint64(_MASK32)).astype(np.uint32)
        c0, c1, c2, c3 = high1 ^ c1 ^ key0, low1, high0 ^ c3 ^ key1, low0
        key0 = (key0.astype(np.uint64) + np.uint64(_PHILOX_W0)).astype(np.uint32)
        key1 = (key1.astype(np.uint64) + np.uint64(_PHILOX_W1)).astype(np.uint32)
    return c0, c1, c2, c3


def _role_key(seed: int, role: int) -> tuple[np.uint32, np.uint32]:
    mixed = _splitmix64((seed & _MASK64) ^ ((role * 0x9E3779B97F4A7C15) & _MASK64))
    return np.uint32(mixed & _MASK32), np.uint32((mixed >> 32) & _MASK32)


class PhiloxStream:
    """A vector of independent counter-based Philox streams.

    Each element is initialized with its own 64-bit edge id in the counter
    words ``(c0, c1)`` and a zero-based draw counter in ``(c2, c3)``.  A four
    word block is buffered per element; only active elements advance.
    """

    def __init__(self, edge_ids: np.ndarray, seed: int, role: int) -> None:
        """Initialize one stream per edge id.

        Args:
            edge_ids: Flat array of unique 64-bit edge identifiers.
            seed: Base seed shared by the run.
            role: Domain-separation tag for the stream family.
        """
        edge_ids = np.asarray(edge_ids, dtype=np.uint64).ravel()
        self.size = int(edge_ids.size)
        self.c0 = (edge_ids & np.uint64(_MASK32)).astype(np.uint32)
        self.c1 = (edge_ids >> _U32).astype(np.uint32)
        self.counter = np.zeros(self.size, dtype=np.uint64)
        self.key0, self.key1 = _role_key(seed, role)
        self.buffer = np.empty((self.size, 4), dtype=np.uint32)
        self.index = np.full(self.size, 4, dtype=np.int8)
        self._rows = np.arange(self.size, dtype=np.int64)

    def _refill(self, active: np.ndarray) -> None:
        need = active & (self.index >= 4)
        if not np.any(need):
            return
        c2 = (self.counter & np.uint64(_MASK32)).astype(np.uint32)
        c3 = (self.counter >> _U32).astype(np.uint32)
        o0, o1, o2, o3 = philox4x32_10(
            self.c0, self.c1, c2, c3, self.key0, self.key1
        )
        self.buffer[need, 0] = o0[need]
        self.buffer[need, 1] = o1[need]
        self.buffer[need, 2] = o2[need]
        self.buffer[need, 3] = o3[need]
        self.counter[need] += _ONE_U64
        self.index[need] = 0

    def next_uniform(self, active: np.ndarray | None = None) -> np.ndarray:
        """Return one uniform draw in ``[0, 1)`` per element.

        Args:
            active: Optional mask selecting elements that consume a draw.

        Returns:
            A float64 array; entries outside ``active`` are unspecified.
        """
        if active is None:
            active = np.ones(self.size, dtype=bool)
        active = np.asarray(active, dtype=bool)
        self._refill(active)
        safe_index = np.where(self.index < 4, self.index, 0)
        bits = self.buffer[self._rows, safe_index]
        values = (bits >> np.uint32(8)).astype(np.float64) * _INV_2POW24
        self.index[active] += 1
        return values

    def next_normal(self, active: np.ndarray | None = None) -> np.ndarray:
        """Return one standard normal draw per element (Box-Muller).

        Args:
            active: Optional mask selecting elements that consume draws.

        Returns:
            A float64 array; entries outside ``active`` are unspecified.
        """
        u1 = self.next_uniform(active)
        u1 = np.maximum(u1, 1e-7)
        u2 = self.next_uniform(active)
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def gamma_draw(shape: np.ndarray, edge_ids: np.ndarray, seed: int, role: int) -> np.ndarray:
    """Sample Gamma(shape, 1) with Marsaglia-Tsang per edge.

    Shapes at or below ``EPS`` return zero without advancing the edge stream.
    Very large shapes use the same normal approximation as the natal host
    sampler.  Rejection loops consume a variable number of uniforms; replaying
    an edge from the same ``(seed, role, edge_id)`` reproduces the same draw.

    Args:
        shape: Gamma shape values.
        edge_ids: Flat 64-bit edge identifiers aligned with ``shape``.
        seed: Base seed shared by the run.
        role: Domain-separation tag for the stream family.

    Returns:
        Gamma draws with the broadcast shape of ``shape``.
    """
    shape = np.asarray(shape, dtype=np.float64)
    flat_shape = shape.ravel()
    out = np.zeros(flat_shape.shape, dtype=np.float64)
    if flat_shape.size == 0:
        return out.reshape(shape.shape)
    stream = PhiloxStream(edge_ids, seed, role)

    resolved = flat_shape >= _RESOLUTION_LIMIT
    if np.any(resolved):
        out[resolved] = flat_shape[resolved]
    large = (flat_shape >= _GAMMA_NORMAL_THRESHOLD) & ~resolved
    if np.any(large):
        z = stream.next_normal(large)
        out[large] = np.maximum(flat_shape[large] + np.sqrt(flat_shape[large]) * z[large], 0.0)

    regular = (flat_shape > EPS) & ~large & ~resolved
    if np.any(regular):
        boosted_shape = flat_shape.copy()
        boost = np.ones_like(flat_shape)
        small = regular & (boosted_shape < 1.0)
        if np.any(small):
            u = np.maximum(stream.next_uniform(small), 1e-7)
            boost[small] = u[small] ** (1.0 / boosted_shape[small])
            boosted_shape[small] += 1.0

        d = np.where(regular, boosted_shape - 1.0 / 3.0, 1.0)
        c = np.where(regular, 1.0 / np.sqrt(9.0 * d), 0.0)
        pending = regular.copy()
        for _ in range(10000):
            if not np.any(pending):
                break
            x = stream.next_normal(pending)
            v = 1.0 + c * x
            good = pending & (v > 0.0)
            accept = np.zeros_like(pending)
            if np.any(good):
                u = stream.next_uniform(good)
                v3 = v * v * v
                with np.errstate(divide="ignore", invalid="ignore"):
                    quick_accept = u < 1.0 - 0.0331 * x * x * x * x
                    log_accept = np.log(u) < 0.5 * x * x + d * (1.0 - v3 + np.log(v3))
                accept = good & (quick_accept | log_accept)
                out[accept] = d[accept] * v3[accept] * boost[accept]
            pending = pending & ~accept
        else:  # pragma: no cover - astronomically unlikely for valid shapes
            raise RuntimeError("gamma rejection loop did not converge")
    return out.reshape(shape.shape)


def continuous_binomial(
    values: np.ndarray, rate: float, edge_ids: np.ndarray, seed: int
) -> np.ndarray:
    """Return natal's continuous-binomial outbound sample.

    Args:
        values: Source counts.
        rate: Migration rate.
        edge_ids: Source-level 64-bit stream ids aligned with ``values``.
        seed: Base seed shared by the run.

    Returns:
        Outbound counts with the shape of ``values``.
    """
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values)
    if rate <= EPS:
        return out
    if rate >= 1.0 - EPS:
        return values.copy()
    out[:] = values * rate
    big = values > 1.0 + EPS
    if np.any(big):
        concentration = values[big] - 1.0
        alpha = np.maximum(rate * concentration, EPS)
        beta = np.maximum((1.0 - rate) * concentration, EPS)
        ga = gamma_draw(alpha, edge_ids[big], seed, ROLE_OUTBOUND_A)
        gb = gamma_draw(beta, edge_ids[big], seed, ROLE_OUTBOUND_B)
        denominator = ga + gb
        proportion = np.where(denominator > EPS, ga / np.where(denominator > EPS, denominator, 1.0), rate)
        proportion = np.where(ga == 0.0, 0.0, proportion)
        out[big] = values[big] * proportion
    return out


def kernel_offsets(kernel: np.ndarray) -> list[tuple[int, int, float]]:
    """Return positive off-centre kernel offsets in scan order.

    Args:
        kernel: Square migration kernel including the centre.

    Returns:
        ``(dr, dc, weight)`` tuples in the kernel's row-major scan order.
    """
    size = kernel.shape[0]
    radius = size // 2
    offsets: list[tuple[int, int, float]] = []
    for kr in range(size):
        for kc in range(size):
            if kr == radius and kc == radius:
                continue
            weight = float(kernel[kr, kc])
            if weight > 0.0:
                offsets.append((kr - radius, kc - radius, weight))
    return offsets


def source_normalizers(rows: int, cols: int, kernel: np.ndarray) -> np.ndarray:
    """Return each source's valid kernel-weight sum ``Z_s``.

    Args:
        rows: Number of grid rows.
        cols: Number of grid columns.
        kernel: Square migration kernel including the centre.

    Returns:
        A flat ``rows * cols`` array of normalizers.
    """
    size = kernel.shape[0]
    radius = size // 2
    normalizer = np.zeros((rows, cols), dtype=np.float64)
    for kr in range(size):
        for kc in range(size):
            if kr == radius and kc == radius:
                continue
            weight = float(kernel[kr, kc])
            if weight <= 0.0:
                continue
            dr, dc = kr - radius, kc - radius
            r0, r1 = max(0, -dr), min(rows, rows - dr)
            c0, c1 = max(0, -dc), min(cols, cols - dc)
            if r0 < r1 and c0 < c1:
                normalizer[r0:r1, c0:c1] += weight
    return normalizer.ravel()


def _valid_sources(rows: int, cols: int, dr: int, dc: int) -> tuple[np.ndarray, np.ndarray]:
    r0, r1 = max(0, -dr), min(rows, rows - dr)
    c0, c1 = max(0, -dc), min(cols, cols - dc)
    if r0 >= r1 or c0 >= c1:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    rr, cc = np.mgrid[r0:r1, c0:c1]
    source = (rr * cols + cc).ravel().astype(np.int64)
    destination = ((rr + dr) * cols + (cc + dc)).ravel().astype(np.int64)
    return source, destination


def stencil_migrate_planes(
    fields: np.ndarray,
    rows: int,
    cols: int,
    kernel: np.ndarray,
    rate: float,
    seed: int,
    tick: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Migrate a batch of independent planes with the two-pass stencil.

    Args:
        fields: Array of shape ``(n_planes, rows * cols)``; a 1-D array is
            treated as one plane.
        rows: Number of grid rows.
        cols: Number of grid columns.
        kernel: Square migration kernel including the centre.
        rate: Migration rate.
        seed: Base seed shared by the run.
        tick: Tick index used to separate RNG streams across ticks.

    Returns:
        A tuple ``(migrated, diagnostics)``.  ``migrated`` has the input field
        shape.  ``diagnostics`` contains ``outbound``, ``source_gamma_sum``,
        ``normalizer`` and ``assigned`` arrays used by the operator check.
    """
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim == 1:
        fields = fields[None, :]
    planes, n = fields.shape
    if n != rows * cols:
        raise ValueError("field width must equal rows * cols")
    source_offset = np.arange(n, dtype=np.uint64)
    plane_base = (np.uint64(tick) * np.uint64(planes) + np.arange(planes, dtype=np.uint64)) * np.uint64(n * n)
    outbound_edge = plane_base[:, None] + source_offset[None, :]
    outbound = continuous_binomial(fields, rate, outbound_edge, seed)

    normalizer = source_normalizers(rows, cols, kernel)
    safe_normalizer = np.where(normalizer > 0.0, normalizer, 1.0)
    offsets = kernel_offsets(kernel)

    source_gamma_sum = np.zeros_like(fields)
    assigned = np.zeros_like(fields)
    for dr, dc, weight in offsets:
        source, destination = _valid_sources(rows, cols, dr, dc)
        if source.size == 0:
            continue
        edge = plane_base[:, None] + (
            source[None, :].astype(np.uint64) * np.uint64(n)
            + destination[None, :].astype(np.uint64)
        )
        normalized = weight / safe_normalizer[source]
        alpha = normalized[None, :] * (outbound[:, source] - 1.0)
        alpha = np.where(outbound[:, source] <= 1.0 + EPS, 0.0, alpha)
        alpha = np.where(alpha <= EPS, 0.0, alpha)
        draws = gamma_draw(alpha, edge, seed, ROLE_GAMMA)
        source_gamma_sum[:, source] += draws

    migrated = fields - outbound
    for dr, dc, weight in offsets:
        source, destination = _valid_sources(rows, cols, dr, dc)
        if source.size == 0:
            continue
        edge = plane_base[:, None] + (
            source[None, :].astype(np.uint64) * np.uint64(n)
            + destination[None, :].astype(np.uint64)
        )
        normalized = weight / safe_normalizer[source]
        alpha = normalized[None, :] * (outbound[:, source] - 1.0)
        alpha = np.where(outbound[:, source] <= 1.0 + EPS, 0.0, alpha)
        alpha = np.where(alpha <= EPS, 0.0, alpha)
        draws = gamma_draw(alpha, edge, seed, ROLE_GAMMA)
        gamma_sum = source_gamma_sum[:, source]
        stochastic = (outbound[:, source] > 1.0 + EPS) & (gamma_sum > EPS)
        safe_sum = np.where(stochastic, gamma_sum, 1.0)
        contribution = np.where(
            stochastic,
            outbound[:, source] * draws / safe_sum,
            outbound[:, source] * normalized[None, :],
        )
        migrated[:, destination] += contribution
        assigned[:, source] += contribution

    diagnostics = {
        "outbound": outbound,
        "source_gamma_sum": source_gamma_sum,
        "normalizer": normalizer,
        "assigned": assigned,
    }
    return migrated, diagnostics


def stencil_migrate_field(
    field: np.ndarray,
    rows: int,
    cols: int,
    kernel: np.ndarray,
    rate: float,
    seed: int,
    tick: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Migrate one flat field with the exact two-pass stencil.

    Args:
        field: Flat source counts of length ``rows * cols``.
        rows: Number of grid rows.
        cols: Number of grid columns.
        kernel: Square migration kernel including the centre.
        rate: Migration rate.
        seed: Base seed shared by the run.
        tick: Tick index used to separate RNG streams across ticks.

    Returns:
        A tuple ``(migrated, diagnostics)`` for one plane.
    """
    migrated, diagnostics = stencil_migrate_planes(
        np.asarray(field, dtype=np.float64)[None, :], rows, cols, kernel, rate, seed, tick
    )
    return migrated[0], diagnostics
