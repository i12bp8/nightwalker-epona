"""Vectorized 3D value noise, fBm and Worley noise (numpy)."""

from __future__ import annotations

import numpy as np

_M = np.uint64(0xFFFFFFFF)


def _hash(ix, iy, iz, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791)
    h = (h.astype(np.uint64) + np.uint64(seed * 0x9E3779B1)) & _M
    h = ((h ^ (h >> np.uint64(16))) * np.uint64(0x7FEB352D)) & _M
    h = ((h ^ (h >> np.uint64(15))) * np.uint64(0x846CA68B)) & _M
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def value(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """Smooth value noise in [0, 1] at points p (..., 3)."""
    fl = np.floor(p)
    f = p - fl
    i = fl.astype(np.int64)
    u = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        wx = u[..., 0] if dx else 1 - u[..., 0]
        for dy in (0, 1):
            wy = u[..., 1] if dy else 1 - u[..., 1]
            for dz in (0, 1):
                wz = u[..., 2] if dz else 1 - u[..., 2]
                out = out + wx * wy * wz * _hash(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm(p: np.ndarray, octaves: int = 5, seed: int = 0, lacunarity: float = 2.03, gain: float = 0.5) -> np.ndarray:
    """Fractal value noise normalized to roughly [0, 1]."""
    amp, total, norm = 1.0, 0.0, 0.0
    q = p
    for o in range(octaves):
        total = total + amp * value(q, seed + o * 101)
        norm += amp
        amp *= gain
        q = q * lacunarity
    return total / norm


def ridged(p: np.ndarray, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Ridged fBm: sharp creases, good for veins and cracks. ~[0, 1], ridges near 1."""
    amp, total, norm = 1.0, 0.0, 0.0
    q = p
    for o in range(octaves):
        n = 1.0 - np.abs(value(q, seed + o * 57) * 2 - 1)
        total = total + amp * n * n
        norm += amp
        amp *= 0.5
        q = q * 2.1
    return total / norm


def worley(p: np.ndarray, seed: int = 0):
    """Return (F1, F2) distances to the nearest feature points (cell size 1)."""
    fl = np.floor(p)
    i = fl.astype(np.int64)
    f1 = np.full(p.shape[:-1], 9.0)
    f2 = np.full(p.shape[:-1], 9.0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx, cy, cz = i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz
                fp = np.stack([cx + _hash(cx, cy, cz, seed), cy + _hash(cx, cy, cz, seed + 1),
                               cz + _hash(cx, cy, cz, seed + 2)], -1)
                d = np.linalg.norm(fp - p, axis=-1)
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
    return f1, f2


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)
