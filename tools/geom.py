"""Small procedural-mesh toolkit for building bones: lofts along curves, knobs, mirroring.

Conventions (match hs.bmd): Y up, +Z forward (head), +X = Epona's left. Front faces are clockwise when seen
from outside (GX), i.e. cross(b - a, c - a) points *into* the surface. Each Part keeps its own chart UVs in
[0, 1]; charts are packed into the body atlas later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Part:
    name: str
    pos: np.ndarray  # (N, 3) bind-pose model space
    nrm: np.ndarray  # (N, 3)
    uv: np.ndarray  # (N, 2) chart-local UVs in [0, 1]
    strips: list = field(default_factory=list)  # GX triangle strips (lists of vertex indices)
    tris: list = field(default_factory=list)  # (a, b, c)
    chart: str = ""  # parts with the same chart id share texture space (mirrors, repeated bones)
    chart_size: tuple = (1.0, 1.0)  # physical size of the chart in model units (u extent, v extent)
    drw: np.ndarray | None = None  # (N,) DRW1 matrix index per vertex (assigned by the rigging step)
    bind: object = None  # rigging hint: joint index, or callable(pos) -> DRW index
    kind: str = "bone"  # painting hint: bone / hoof / sinew / tooth / ...

    def triangles(self) -> np.ndarray:
        out = list(self.tris)
        for s in self.strips:
            for i in range(len(s) - 2):
                a, b, c = s[i], s[i + 1], s[i + 2]
                out.append((a, b, c) if i % 2 == 0 else (b, a, c))
        return np.array(out, np.int32).reshape(-1, 3)


# ------------------------------------------------------------------------------------------ frames
def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def frames_along(points: np.ndarray, up_hint=(0.0, 1.0, 0.0), normal_hints: np.ndarray | None = None):
    """Rotation-minimizing frames (t, n, b) along a polyline. n starts near up_hint (or follows normal_hints)."""
    p = np.asarray(points, float)
    t = np.gradient(p, axis=0)
    t = _norm(t)
    ns, bs = [], []
    hint = np.asarray(up_hint, float)
    for i in range(len(p)):
        h = normal_hints[i] if normal_hints is not None else (hint if i == 0 else ns[-1])
        n = h - np.dot(h, t[i]) * t[i]
        if np.linalg.norm(n) < 1e-6:
            n = np.cross(t[i], [1.0, 0, 0])
        n = _norm(n)
        ns.append(n)
        bs.append(np.cross(t[i], n))
    return t, np.array(ns), np.array(bs)


# ------------------------------------------------------------------------------------------ primitives
def loft(name: str, centers, radii_n, radii_b, sides: int = 10, up_hint=(0, 1, 0), normal_hints=None,
         power: float = 2.0, cap_start: bool = True, cap_end: bool = True, twist=None, offsets_n=None,
         offsets_b=None) -> Part:
    """Tube through `centers` with elliptical (super-elliptical) cross-sections.

    radii_n / radii_b: per-ring radius along the frame normal / binormal. power: 2 = ellipse, >2 = boxier.
    """
    c = np.asarray(centers, float)
    m = len(c)
    rn = np.broadcast_to(np.asarray(radii_n, float), (m,))
    rb = np.broadcast_to(np.asarray(radii_b, float), (m,))
    t, n, b = frames_along(c, up_hint, normal_hints)
    if twist is not None:
        tw = np.broadcast_to(np.asarray(twist, float), (m,))
        n2 = n * np.cos(tw)[:, None] + b * np.sin(tw)[:, None]
        b = np.cross(t, n2)
        n = n2
    on = np.zeros(m) if offsets_n is None else np.broadcast_to(np.asarray(offsets_n, float), (m,))
    ob = np.zeros(m) if offsets_b is None else np.broadcast_to(np.asarray(offsets_b, float), (m,))
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    ca, sa = np.cos(ang), np.sin(ang)
    e = 2.0 / power
    sx = np.sign(ca) * np.abs(ca) ** e
    sy = np.sign(sa) * np.abs(sa) ** e
    pos, nrm, uv = [], [], []
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    along = np.concatenate([[0], np.cumsum(seg)])
    length = max(along[-1], 1e-6)
    for i in range(m):
        ring = (c[i] + (sx[:, None] * rn[i] + on[i]) * n[i] + (sy[:, None] * rb[i] + ob[i]) * b[i])
        # normal of the super-ellipse: gradient of (x/rn)^p + (y/rb)^p
        gx = np.sign(sx) * np.abs(sx) ** (power - 1) / max(rn[i], 1e-6)
        gy = np.sign(sy) * np.abs(sy) ** (power - 1) / max(rb[i], 1e-6)
        rn_ = _norm(gx[:, None] * n[i] + gy[:, None] * b[i])
        # account for radius change along the tube (cone slope)
        if m > 1:
            i0, i1 = max(i - 1, 0), min(i + 1, m - 1)
            dr = (0.5 * (rn[i1] + rb[i1]) - 0.5 * (rn[i0] + rb[i0])) / max(along[i1] - along[i0], 1e-6)
            rn_ = _norm(rn_ - dr * t[i])
        pos.append(ring)
        nrm.append(rn_)
        uv.append(np.stack([ang / (2 * np.pi), np.full(sides + 1, along[i] / length)], 1))
    pos = np.concatenate(pos)
    nrm = np.concatenate(nrm)
    uv = np.concatenate(uv)
    strips = []
    w = sides + 1
    for i in range(m - 1):
        s = []
        for k in range(w):
            s += [i * w + k, (i + 1) * w + k]
        strips.append(s)
    part = Part(name, pos, nrm, uv, strips=strips,
                chart_size=(float(np.pi * (rn.mean() + rb.mean())), float(length)))
    if cap_start:
        _cap(part, 0, w, -t[0], c[0] + on[0] * n[0] + ob[0] * b[0], 0.0)
    if cap_end:
        _cap(part, (m - 1) * w, w, t[-1], c[-1] + on[-1] * n[-1] + ob[-1] * b[-1], 1.0)
    fix_winding(part)
    return part


def _cap(part: Part, start: int, w: int, normal, center, v):
    ring = list(range(start, start + w - 1))
    ci = len(part.pos)
    part.pos = np.vstack([part.pos, center])
    part.nrm = np.vstack([part.nrm, _norm(normal)])
    part.uv = np.vstack([part.uv, [0.5, v]])
    # reuse the ring vertices: the rim keeps its side normal, so the cap shades like a rounded end
    for k in range(len(ring)):
        a, b = ring[k], ring[(k + 1) % len(ring)]
        part.tris.append((ci, a, b))


def ellipsoid(name: str, center, radii, axes=None, sides: int = 10, rings: int = 7) -> Part:
    """Ellipsoid knob. axes: 3x3 rows = local x, y, z directions (default world axes); lofted along local y."""
    ax = np.eye(3) if axes is None else _norm(np.asarray(axes, float))
    rx, ry, rz = radii
    ts = np.linspace(-1, 1, rings + 2)[1:-1]
    ts = np.sin(ts * np.pi / 2)  # denser near the poles
    k = np.sqrt(np.clip(1 - ts ** 2, 0, 1))
    centers = np.asarray(center, float) + ts[:, None] * ry * ax[1]
    hints = np.repeat(ax[0][None], len(ts), 0)
    return loft(name, centers, rx * k, rz * k, sides=sides, normal_hints=hints)


def mirror(part: Part, name: str | None = None) -> Part:
    """Mirror across X (left <-> right). Keeps the chart (shared texture), flips winding."""
    p = Part(name or part.name.replace("_L", "_R"), part.pos * [-1, 1, 1], part.nrm * [-1, 1, 1],
             part.uv.copy(), strips=[list(s) for s in part.strips], tris=list(part.tris), chart=part.chart,
             chart_size=part.chart_size, bind=part.bind, kind=part.kind)
    fix_winding(p)
    return p


def fix_winding(part: Part) -> None:
    """Make every strip / triangle clockwise-from-outside (GX front face), using the vertex normals."""
    P, N = part.pos, part.nrm

    def sign(a, b, c):
        cr = np.cross(P[b] - P[a], P[c] - P[a])
        return float(np.dot(cr, N[a] + N[b] + N[c]))

    new_strips = []
    for s in part.strips:
        tot = 0.0
        for i in range(len(s) - 2):
            a, b, c = s[i], s[i + 1], s[i + 2]
            if i % 2:
                a, b = b, a
            tot += sign(a, b, c)
        if tot > 0:  # wrong way round: swap each pair (keeps the same triangles, flips all)
            s = [s[i ^ 1] for i in range(len(s))] if len(s) % 2 == 0 else list(reversed(s))
        new_strips.append(s)
    part.strips = new_strips
    part.tris = [(a, c, b) if sign(a, b, c) > 0 else (a, b, c) for a, b, c in part.tris]


def merge(parts: list[Part], name: str, chart: str | None = None) -> Part:
    """Concatenate parts into one (they must already share a chart layout)."""
    pos, nrm, uv, strips, tris = [], [], [], [], []
    off = 0
    for p in parts:
        pos.append(p.pos)
        nrm.append(p.nrm)
        uv.append(p.uv)
        strips += [[i + off for i in s] for s in p.strips]
        tris += [(a + off, b + off, c + off) for a, b, c in p.tris]
        off += len(p.pos)
    first = parts[0]
    return Part(name, np.vstack(pos), np.vstack(nrm), np.vstack(uv), strips, tris, chart or first.chart,
                first.chart_size, bind=first.bind, kind=first.kind)


def remap_uv(part: Part, u0: float, v0: float, u1: float, v1: float) -> None:
    """Squeeze the part's chart UVs into a sub-rectangle of its chart (for multi-piece charts)."""
    part.uv = np.stack([u0 + part.uv[:, 0] * (u1 - u0), v0 + part.uv[:, 1] * (v1 - v0)], 1)


def smooth_profile(t, r_mid, r_start, r_end, knob=0.18):
    """Long-bone radius profile: shaft r_mid with flared epiphyses at both ends."""
    t = np.asarray(t, float)
    a = np.exp(-(t / knob) ** 2)
    b = np.exp(-((1 - t) / knob) ** 2)
    return r_mid + (r_start - r_mid) * a + (r_end - r_mid) * b
