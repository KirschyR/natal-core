"""Layered multiplicative response functions from the hex model.

All parameters here are placeholders.  They are intentionally uncalibrated and
exist only so the GIS -> K pipeline can be exercised end to end.
"""
from __future__ import annotations

import numpy as np

PARAMETERS = {
    "a1": 1.0,
    "a2": 1.0,
    "a3": 1.0,
    "a4": -1.0 / 800.0,
    "a5": 1.0,
    "p1": 0.01,
    "p2": 0.01,
    "p3": 0.05,
    "p4": 0.001,
    "p5": 0.1,
    "p6": 1.0,
    "p7": 1.0,
    "p8": 22.0,
    "p9": 6.0,
    "p10": 1.0,
    "p11": 28.0,
    "p12": 6.0,
    "base": 0.0,
    "k_scale": 10.0,
    "status": "uncalibrated",
}


def sanitize(values, fill: float = 0.0) -> np.ndarray:
    """Return a finite float64 array, replacing NaN/Inf with ``fill``."""
    array = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(array)
    if bad.any():
        array = array.copy()
        array[bad] = fill
    return array


def rain_func(rain, p=None) -> np.ndarray:
    p = PARAMETERS if p is None else p
    rain = sanitize(rain)
    value = p["a1"] * (1.0 - np.exp(-p["p1"] * rain)
                       - np.exp(-p["p2"] * (320.0 - rain)))
    return np.maximum(value, 0.0)


def water_func(rain, water_p, water_s, p=None) -> np.ndarray:
    p = PARAMETERS if p is None else p
    rain = sanitize(rain)
    water_p = sanitize(water_p)
    water_s = sanitize(water_s)
    value = p["a2"] * (
        1.0 - np.exp(-p["p2"] * (water_p + water_s * (1.0 - np.exp(-p["p3"] * rain))))
    )
    return np.maximum(value, 0.0)


def food_func(pop, p=None) -> np.ndarray:
    p = PARAMETERS if p is None else p
    pop = np.maximum(sanitize(pop), 0.0)
    return p["a3"] * (1.0 - np.exp(-p["p4"] * pop)) + p["p5"]


def altitude_func(alt, p=None) -> np.ndarray:
    p = PARAMETERS if p is None else p
    alt = sanitize(alt)
    value = p["a4"] * alt + p["p6"]
    value = np.where(alt < 800.0, value, 0.0)
    return np.maximum(value, 0.0)


def _gaussian_terms(temperature, p) -> np.ndarray:
    temperature = sanitize(temperature)
    return (
        p["p7"] * np.exp(-((temperature - p["p8"]) / p["p9"]) ** 2)
        + p["p10"] * np.exp(-((temperature - p["p11"]) / p["p12"]) ** 2)
    )


def temperature_func(tavg, tmin, tmax, p=None) -> np.ndarray:
    p = PARAMETERS if p is None else p
    value = (
        p["a5"]
        * np.cbrt(_gaussian_terms(tavg, p))
        * np.cbrt(_gaussian_terms(tmin, p))
        * np.cbrt(_gaussian_terms(tmax, p))
    )
    return np.maximum(value, 0.0)


def carrying_capacity(elevation, tavg, tmin, tmax, precip, population,
                      water_permanent, water_seasonal, land_mask, p=None):
    """Compute per-pixel capacity K (placeholder units)."""
    p = PARAMETERS if p is None else p
    rain = rain_func(precip, p)
    water = water_func(precip, water_permanent, water_seasonal, p)
    food = food_func(population, p)
    alt = altitude_func(elevation, p)
    temp = temperature_func(tavg, tmin, tmax, p)
    value = p["base"] + (rain + water) * food * alt * temp
    value = value * p["k_scale"]
    value = np.where(np.asarray(land_mask, dtype=bool), value, 0.0)
    value = sanitize(value)
    return np.maximum(value, 0.0).astype(np.float32)
