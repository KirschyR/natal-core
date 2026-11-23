"""PyTorch GPU implementation of the direct stencil stochastic migration.

This mirrors :mod:`direct_stencil_migration` on CUDA tensors.  Each edge keeps
an independent Philox4x32-10 stream keyed by a 64-bit edge id; pass 1
accumulates the per-source Gamma total, pass 2 replays the same Gamma draws in
a fixed offset order.  The implementation uses int64 tensors for the 32-bit
Philox words because CUDA PyTorch does not implement uint32 arithmetic.

The module is for the GPU prototype only.  It is not imported by the CPU code
path and does not modify natal.
"""

from __future__ import annotations

import direct_stencil_migration as cpu_core
import numpy as np
import torch

EPS = cpu_core.EPS
_GAMMA_NORMAL_THRESHOLD = 1e8
_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF
_INV_2POW24 = 1.0 / 16777216.0
_TARGET_BATCH_EDGES = 1_500_000

_PHILOX_M0 = 0xD2511F53
_PHILOX_M1 = 0xCD9E8D57
_PHILOX_W0 = 0x9E3779B9
_PHILOX_W1 = 0xBB67AE85

ROLE_OUTBOUND_A = cpu_core.ROLE_OUTBOUND_A
ROLE_OUTBOUND_B = cpu_core.ROLE_OUTBOUND_B
ROLE_GAMMA = cpu_core.ROLE_GAMMA


def _role_key(seed: int, role: int) -> tuple[int, int]:
    mixed = cpu_core._splitmix64(  # noqa: SLF001 - reuse the documented mixer
        (seed & cpu_core._MASK64) ^ ((role * 0x9E3779B97F4A7C15) & cpu_core._MASK64)
    )
    return mixed & _MASK32, (mixed >> 32) & _MASK32


def _mul32(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return high/low 32 bits of the unsigned 32-bit product."""
    a_hi = a >> 16
    a_lo = a & _MASK16
    b_hi = b >> 16
    b_lo = b & _MASK16
    cross = a_hi * b_lo + a_lo * b_hi
    total = a_lo * b_lo + (cross << 16)
    low = total & _MASK32
    high = (a_hi * b_hi + (total >> 32)) & _MASK32
    return high, low


def _philox4x32_10(
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
    key0: int,
    key1: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one Philox4x32-10 block for int64-encoded uint32 words."""
    key0_t = torch.as_tensor(key0, dtype=torch.int64, device=c0.device)
    key1_t = torch.as_tensor(key1, dtype=torch.int64, device=c0.device)
    for _ in range(10):
        high0, low0 = _mul32(torch.as_tensor(_PHILOX_M0, dtype=torch.int64, device=c0.device), c0)
        high1, low1 = _mul32(torch.as_tensor(_PHILOX_M1, dtype=torch.int64, device=c0.device), c2)
        c0, c1, c2, c3 = high1 ^ c1 ^ key0_t, low1, high0 ^ c3 ^ key1_t, low0
        key0_t = (key0_t + _PHILOX_W0) & _MASK32
        key1_t = (key1_t + _PHILOX_W1) & _MASK32
    return c0, c1, c2, c3


class _PhiloxStream:
    """Vector of independent counter-based streams on a CUDA device."""

    def __init__(self, edge_ids: torch.Tensor, seed: int, role: int) -> None:
        edge_ids = edge_ids.to(dtype=torch.int64).reshape(-1)
        self.size = int(edge_ids.numel())
        self.device = edge_ids.device
        self.c0 = edge_ids & _MASK32
        self.c1 = (edge_ids >> 32) & _MASK32
        self.counter = torch.zeros_like(edge_ids)
        self.key0, self.key1 = _role_key(seed, role)
        self.buffer = torch.empty((self.size, 4), dtype=torch.int64, device=self.device)
        self.index = torch.full((self.size,), 4, dtype=torch.int64, device=self.device)
        self.rows = torch.arange(self.size, dtype=torch.int64, device=self.device)

    def _refill(self, active: torch.Tensor) -> None:
        need = active & (self.index >= 4)
        if not bool(need.any()):
            return
        c2 = self.counter & _MASK32
        c3 = (self.counter >> 32) & _MASK32
        o0, o1, o2, o3 = _philox4x32_10(self.c0, self.c1, c2, c3, self.key0, self.key1)
        self.buffer[need, 0] = o0[need]
        self.buffer[need, 1] = o1[need]
        self.buffer[need, 2] = o2[need]
        self.buffer[need, 3] = o3[need]
        self.counter[need] += 1
        self.index[need] = 0

    def next_uniform(
        self, active: torch.Tensor | None = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        if active is None:
            active = torch.ones(self.size, dtype=torch.bool, device=self.device)
        self._refill(active)
        safe_index = torch.where(self.index < 4, self.index, torch.zeros_like(self.index))
        bits = self.buffer[self.rows, safe_index]
        values = (bits >> 8).to(dtype) * _INV_2POW24
        self.index[active] += 1
        return values

    def next_normal(
        self, active: torch.Tensor | None = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        u1 = self.next_uniform(active, dtype).clamp_min(1e-7)
        u2 = self.next_uniform(active, dtype)
        return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * u2)


def _gamma_draw(
    shape: torch.Tensor,
    edge_ids: torch.Tensor,
    seed: int,
    role: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Marsaglia-Tsang Gamma(shape, 1) per edge on the GPU."""
    original_shape = shape.shape
    flat_shape = shape.reshape(-1).to(dtype)
    stream = _PhiloxStream(edge_ids, seed, role)
    out = torch.zeros_like(flat_shape)

    resolved = flat_shape >= cpu_core._RESOLUTION_LIMIT  # noqa: SLF001
    if bool(resolved.any()):
        out[resolved] = flat_shape[resolved]
    large = (flat_shape >= _GAMMA_NORMAL_THRESHOLD) & ~resolved
    if bool(large.any()):
        z = stream.next_normal(large, dtype)
        out[large] = torch.clamp(flat_shape[large] + torch.sqrt(flat_shape[large]) * z[large], min=0.0)

    regular = (flat_shape > EPS) & ~large & ~resolved
    if bool(regular.any()):
        boosted = flat_shape.clone()
        boost = torch.ones_like(flat_shape)
        small = regular & (boosted < 1.0)
        if bool(small.any()):
            u = stream.next_uniform(small, dtype).clamp_min(1e-7)
            boost[small] = u[small].pow(1.0 / boosted[small])
            boosted[small] += 1.0

        d = torch.where(regular, boosted - 1.0 / 3.0, torch.ones_like(boosted))
        c = torch.where(regular, 1.0 / torch.sqrt(9.0 * d), torch.zeros_like(d))
        tiny = torch.finfo(dtype).tiny
        pending = regular
        for _ in range(10000):
            if not bool(pending.any()):
                break
            x = stream.next_normal(pending, dtype)
            v = 1.0 + c * x
            good = pending & (v > 0.0)
            accept = torch.zeros_like(pending)
            if bool(good.any()):
                u = stream.next_uniform(good, dtype)
                v3 = v * v * v
                log_u = torch.log(u.clamp_min(tiny))
                log_v = torch.log(v3.clamp_min(tiny))
                quick = u < 1.0 - 0.0331 * x * x * x * x
                slow = log_u < 0.5 * x * x + d * (1.0 - v3 + log_v)
                accept = good & (quick | slow)
                out[accept] = d[accept] * v3[accept] * boost[accept]
            pending = pending & ~accept
        else:  # pragma: no cover - astronomically unlikely for valid shapes
            raise RuntimeError("GPU gamma rejection loop did not converge")
    return out.reshape(original_shape)


def _continuous_binomial(
    values: torch.Tensor,
    rate: float,
    edge_ids: torch.Tensor,
    seed: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros_like(values)
    if rate <= EPS:
        return out
    if rate >= 1.0 - EPS:
        return values.clone()
    out[:] = values * rate
    big = values > 1.0 + EPS
    if bool(big.any()):
        concentration = values[big] - 1.0
        alpha = torch.clamp(rate * concentration, min=EPS)
        beta = torch.clamp((1.0 - rate) * concentration, min=EPS)
        ga = _gamma_draw(alpha, edge_ids[big], seed, ROLE_OUTBOUND_A, dtype)
        gb = _gamma_draw(beta, edge_ids[big], seed, ROLE_OUTBOUND_B, dtype)
        denominator = ga + gb
        proportion = torch.where(denominator > EPS, ga / denominator, torch.full_like(ga, rate))
        proportion = torch.where(ga == 0.0, torch.zeros_like(proportion), proportion)
        out[big] = values[big] * proportion
    return out


_GEOMETRY_CACHE: dict[tuple, tuple[torch.Tensor, list[tuple[float, torch.Tensor, torch.Tensor]]]] = {}


def _geometry(
    rows: int,
    cols: int,
    kernel: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor, torch.Tensor]]]:
    key = (rows, cols, kernel.shape[0], kernel.tobytes(), str(device), dtype)
    cached = _GEOMETRY_CACHE.get(key)
    if cached is not None:
        return cached
    normalizer = torch.as_tensor(
        cpu_core.source_normalizers(rows, cols, kernel), dtype=dtype, device=device
    )
    safe_normalizer = torch.where(normalizer > 0.0, normalizer, torch.ones_like(normalizer))
    items: list[tuple[float, torch.Tensor, torch.Tensor]] = []
    for dr, dc, weight in cpu_core.kernel_offsets(kernel):
        source, destination = cpu_core._valid_sources(rows, cols, dr, dc)  # noqa: SLF001
        if source.size == 0:
            continue
        items.append(
            (
                float(weight),
                torch.as_tensor(source, dtype=torch.int64, device=device),
                torch.as_tensor(destination, dtype=torch.int64, device=device),
            )
        )
    result = (safe_normalizer, items)
    _GEOMETRY_CACHE[key] = result
    return result


def _batches(items: list[tuple[float, torch.Tensor, torch.Tensor]], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def stencil_migrate_planes_torch(
    fields: torch.Tensor,
    rows: int,
    cols: int,
    kernel: np.ndarray,
    rate: float,
    seed: int,
    tick: int = 0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Migrate a batch of planes with the two-pass stencil on the GPU.

    Args:
        fields: ``(n_planes, rows * cols)`` tensor on the target device.
        rows: Number of grid rows.
        cols: Number of grid columns.
        kernel: Square migration kernel including the centre.
        rate: Migration rate.
        seed: Base seed shared by the run.
        tick: Tick index used to separate RNG streams across ticks.
        dtype: Floating dtype for the GPU arithmetic.
        device: CUDA device; defaults to the fields device.

    Returns:
        A tuple ``(migrated, diagnostics)`` with GPU tensors.
    """
    if device is None:
        device = fields.device
    fields = fields.to(device=device, dtype=dtype)
    planes, n = fields.shape
    if n != rows * cols:
        raise ValueError("field width must equal rows * cols")
    base = (
        torch.as_tensor(tick * planes, dtype=torch.int64, device=device)
        + torch.arange(planes, dtype=torch.int64, device=device)
    ) * (n * n)
    source_offset = torch.arange(n, dtype=torch.int64, device=device)
    outbound = _continuous_binomial(
        fields, rate, base[:, None] + source_offset[None, :], seed, dtype
    )

    normalizer, items = _geometry(rows, cols, kernel, device, dtype)
    max_offset_edges = max((int(source.numel()) for _, source, _ in items), default=1)
    batch_size = max(
        1,
        min(len(items), _TARGET_BATCH_EDGES // max(1, planes * max_offset_edges)),
    )
    source_gamma_sum = torch.zeros_like(fields)
    assigned = torch.zeros_like(fields)

    for batch in _batches(items, batch_size):
        source_parts: list[torch.Tensor] = []
        destination_parts: list[torch.Tensor] = []
        weight_parts: list[torch.Tensor] = []
        slices: list[tuple[int, int, float, torch.Tensor, torch.Tensor]] = []
        start = 0
        for weight, source, destination in batch:
            source_parts.append(source)
            destination_parts.append(destination)
            weight_parts.append(
                torch.full((source.numel(),), weight, dtype=dtype, device=device)
            )
            slices.append((start, start + source.numel(), weight, source, destination))
            start += source.numel()
        source_cat = torch.cat(source_parts)
        destination_cat = torch.cat(destination_parts)
        weight_cat = torch.cat(weight_parts)
        normalized = weight_cat / normalizer[source_cat]
        alpha = normalized[None, :] * (outbound[:, source_cat] - 1.0)
        alpha = torch.where(outbound[:, source_cat] <= 1.0 + EPS, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha <= EPS, torch.zeros_like(alpha), alpha)
        edge = base[:, None] + (source_cat * n + destination_cat)[None, :]
        draws = _gamma_draw(alpha, edge, seed, ROLE_GAMMA, dtype)
        for start, end, _weight, source, _destination in slices:
            source_gamma_sum[:, source] += draws[:, start:end]

    migrated = fields - outbound
    for batch in _batches(items, batch_size):
        source_parts = []
        destination_parts = []
        weight_parts = []
        slices = []
        start = 0
        for weight, source, destination in batch:
            source_parts.append(source)
            destination_parts.append(destination)
            weight_parts.append(
                torch.full((source.numel(),), weight, dtype=dtype, device=device)
            )
            slices.append((start, start + source.numel(), weight, source, destination))
            start += source.numel()
        source_cat = torch.cat(source_parts)
        destination_cat = torch.cat(destination_parts)
        weight_cat = torch.cat(weight_parts)
        normalized = weight_cat / normalizer[source_cat]
        alpha = normalized[None, :] * (outbound[:, source_cat] - 1.0)
        alpha = torch.where(outbound[:, source_cat] <= 1.0 + EPS, torch.zeros_like(alpha), alpha)
        alpha = torch.where(alpha <= EPS, torch.zeros_like(alpha), alpha)
        edge = base[:, None] + (source_cat * n + destination_cat)[None, :]
        draws = _gamma_draw(alpha, edge, seed, ROLE_GAMMA, dtype)
        for start, end, weight, source, destination in slices:
            normalized_slice = weight / normalizer[source]
            gamma_sum = source_gamma_sum[:, source]
            stochastic = (outbound[:, source] > 1.0 + EPS) & (gamma_sum > EPS)
            safe_sum = torch.where(stochastic, gamma_sum, torch.ones_like(gamma_sum))
            contribution = torch.where(
                stochastic,
                outbound[:, source] * draws[:, start:end] / safe_sum,
                outbound[:, source] * normalized_slice[None, :],
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
