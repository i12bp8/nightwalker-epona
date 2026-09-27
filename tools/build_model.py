"""Assemble the undead "wight" Epona model (hs.bmd).

Layers, outside in (all but the bones reuse Epona's own triangles and skinning, so they deform exactly
like the original and cannot tear apart in any animation):
  1. hide   - her original body, alpha-cut into ragged wounds (tears live in the texture, not the mesh)
  2. muscle - an inset copy of the body, dried muscle with its own, smaller tears
  3. cavity - the same inset shell facing inward: what you see through the wounds on the far side
  4. bones  - a procedural skeleton (skull re-sculpted from her head, teeth, ribs, spine, legs, pelvis)
     fitted inside the muscle shell
Tack (saddle, blanket, girth, stirrups, bridle, bedroll), mane, tail and eyes are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

import bake
import bmd
import bmdwrite
import skeleton
from geom import Part, fix_winding

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
ATLAS = 2048  # body atlas working resolution (HD replacement size); 2x2 quadrants
LAYOUT = 1024  # the original body layout's size (its UV rects/seeds below are in this space)

# UV rects (1024 atlas px) of tack islands in the original body texture, see paint notes
TACK_RECTS = [(489, 0, 1022, 112), (797, 67, 1021, 389), (2, 364, 283, 1017), (506, 726, 612, 920),
              (968, 790, 1023, 1022), (686, 816, 958, 927), (694, 927, 962, 1022), (748, 61, 796, 179)]
BREASTCOLLAR_RECT = (506, 726, 612, 920)  # chest pendant
HOOF_SOLE_RECT = (617, 829, 686, 1022)
SADDLE_JOINTS = {21, 22, 23, 24, 25}
EAR_JOINTS = {16, 17}
FORELOCK_JOINT = 18
HEAD_JOINT = 15


@dataclass
class Geometry:
    """Everything needed to write the BMD and to paint/preview it."""
    shapes: list  # per original shape: list of (Part, is_original) blocks
    parts: list  # new/modified body parts (shape 1)


# ------------------------------------------------------------------------------------------ helpers
def uv_px(uv):
    w = uv - np.floor(uv)
    return w * LAYOUT


def in_rects(px, rects):
    m = np.zeros(len(px), bool)
    for x0, y0, x1, y1 in rects:
        m |= (px[:, 0] >= x0) & (px[:, 0] <= x1) & (px[:, 1] >= y0) & (px[:, 1] <= y1)
    return m


def envelope_drw(model: bmd.BMD, drw: bmdwrite.DrwLayout):
    """{frozenset((joint, weight)) -> new DRW1 index} for the original weighted matrices."""
    out = {}
    for d in range(len(model.drw_weighted)):
        if model.drw_weighted[d] and d < len(model.drw_weighted) - len(model.envelopes):
            env = model.envelopes[model.drw_index[d]]
            out[frozenset((j, round(w, 2)) for j, w in env)] = drw.remap_old(d)
    return out


def part_from_mesh(mesh: bmd.Mesh, tri_mask, name: str) -> Part:
    """Extract triangles of an original mesh as a Part (model space), keeping raw DRW per vertex."""
    tris = mesh.tris[tri_mask]
    used, inv = np.unique(tris.reshape(-1), return_inverse=True)
    p = Part(name, mesh.pos[used].copy(), mesh.nrm[used].copy(), mesh.uv[used].copy(),
             tris=[tuple(t) for t in inv.reshape(-1, 3)])
    p.raw = mesh.raw[used].copy()
    p.joint = mesh.joint[used].copy()
    p.fixed_uv = True
    return p


def subdivide(pos, nrm, uv, tris):
    """1 -> 4 midpoint subdivision with shared edge midpoints."""
    pos, nrm, uv = list(pos), list(nrm), list(uv)
    cache = {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            cache[k] = len(pos)
            pos.append((pos[a] + pos[b]) / 2)
            n = nrm[a] + nrm[b]
            nrm.append(n / max(np.linalg.norm(n), 1e-9))
            uv.append((uv[a] + uv[b]) / 2)
        return cache[k]

    out = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return np.array(pos), np.array(nrm), np.array(uv), out


def boundary_loops(tris):
    from collections import defaultdict
    cnt = defaultdict(int)
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            cnt[(min(e), max(e))] += 1
    edges = [e for e, n in cnt.items() if n == 1]
    adj = defaultdict(list)
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [v for v in adj[cur] if v != prev and v not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


# ------------------------------------------------------------------------------------------ skull
EYE_C = np.array([20.0, 198.0, 191.0])


def skull_slim(pos):
    """X scale that turns the fleshy head into a leaner skull (strongest at the jaw)."""
    y, z = pos[:, 1], pos[:, 2]
    return 0.9 - 0.08 * np.clip((196 - y) / 30, 0, 1) * np.clip((200 - z) / 40, 0, 1)


def on_head(pos):
    return (pos[:, 2] > 140) & (pos[:, 1] > 135)


def make_skull(body: bmd.Mesh, keep_mask, inner_mask, drw: bmdwrite.DrwLayout) -> list[Part]:
    """Re-sculpt the original head into a skull: slimmer, hollow cheeks, deep orbits, capped openings.

    Vertices keep Epona's own skinning (mouth1 drives the muzzle, mouth2 the jaw), so the skull's jaw still
    opens when she neighs.
    """
    t = body.tris[keep_mask]
    inner = inner_mask[keep_mask]
    used, inv = np.unique(t.reshape(-1), return_inverse=True)
    pos, nrm, uv = list(body.pos[used]), list(body.nrm[used]), list(body.uv[used])
    d = [drw.remap_old(int(x)) for x in body.raw[used][:, 3]]
    tris = [tuple(x) for x in inv.reshape(-1, 3)]
    for _ in range(2):
        cache = {}

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                cache[k] = len(pos)
                pos.append((pos[a] + pos[b]) / 2)
                n = nrm[a] + nrm[b]
                nrm.append(n / max(np.linalg.norm(n), 1e-9))
                uv.append((uv[a] + uv[b]) / 2)
                ka, kb = tuple(np.round(pos[a], 2)), tuple(np.round(pos[b], 2))
                d.append(d[a] if ka <= kb else d[b])  # position-based so UV-seam copies agree
            return cache[k]

        new = []
        for a_, b_, c_ in tris:
            ab, bc, ca = mid(a_, b_), mid(b_, c_), mid(c_, a_)
            new += [(a_, ab, ca), (ab, b_, bc), (ca, bc, c_), (ab, bc, ca)]
        tris = new
        inner = np.repeat(inner, 4)
    pos, nrm, uv = np.array(pos), np.array(nrm), np.array(uv)
    drw_v = np.array(d, np.int32)
    weld = _weld(pos)
    pos = _laplacian(pos, tris, weld, iterations=3, lam=0.45)
    pos = pos - welded_normals(pos, nrm, tris) * 2.0  # sit under the hide
    ax = np.abs(pos[:, 0])
    x, y, z = ax, pos[:, 1], pos[:, 2]
    sgn = np.where(pos[:, 0] >= 0, 1.0, -1.0)
    out = pos.copy()
    out[:, 0] = pos[:, 0] * skull_slim(pos)
    lateral = np.clip((ax - 5) / 6, 0, 1)

    def blob(cy, cz, ry, rz):
        dd = np.sqrt(((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2)
        return np.clip(1 - dd, 0, 1) ** 1.5

    def ridge(a, b, width):
        a, b = np.asarray(a, float), np.asarray(b, float)
        q = np.stack([z, y], 1)
        ab = b - a
        tt = np.clip(((q - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(q - (a + tt[:, None] * ab), axis=1)
        return np.clip(1 - dd / width, 0, 1) ** 2 * np.sin(np.pi * np.clip(tt, 0.05, 0.95))

    # temporal fossa behind the orbit, masseter hollow under it
    push = 3.8 * blob(205, 168, 8, 12) + 3.2 * blob(176, 172, 12, 15)
    # orbit: deep socket, raised rim, heavier brow bar above
    d_eye = np.sqrt((y - EYE_C[1]) ** 2 + (z - EYE_C[2]) ** 2)
    near = np.clip((ax - 10) / 4, 0, 1)
    push += 6.5 * np.clip(1 - d_eye / 9.5, 0, 1) ** 0.7 * near
    push -= (2.2 * np.clip(1 - np.abs(d_eye - 11.0) / 2.6, 0, 1) + 1.0 * (y > EYE_C[1] + 4)
             * np.clip(1 - np.abs(d_eye - 11.5) / 3.0, 0, 1)) * near
    # facial crest running forward under the eye, nasal notch, and a groove where the lips were
    push -= 1.6 * ridge((172, 186), (206, 181), 2.8) * lateral
    push += 3.0 * np.clip(1 - np.sqrt(((z - 207) / 11) ** 2 + ((y - 176) / 3.2) ** 2), 0, 1) * lateral
    out[:, 0] -= sgn * push
    whole = Part("skull", out, nrm, uv, tris=tris)
    _renormal(whole, weld)
    fix_winding(whole)
    tri_arr = np.array(whole.tris)
    skull = Part("skull", whole.pos, whole.nrm, whole.uv, tris=[tuple(t_) for t_ in tri_arr[~inner]])
    mouth = Part("mouth", whole.pos, whole.nrm, whole.uv, tris=[tuple(t_) for t_ in tri_arr[inner]])
    for p_, kind in ((skull, "skull"), (mouth, "mouth")):
        p_.drw = drw_v
        p_.fixed_uv = True
        p_.kind = kind

    # close the openings (neck cut, ear sockets) with fans so the skull reads solid from behind.
    # The source mesh splits vertices along UV seams, so find holes on position-welded vertices.
    key = np.round(whole.pos, 3)
    _, weld = np.unique(key, axis=0, return_inverse=True)
    weld = weld.reshape(-1)
    welded_tris = [(weld[a], weld[b], weld[c]) for a, b, c in whole.tris]
    rep = {}
    for i, w in enumerate(weld):
        rep.setdefault(w, i)
    caps = []
    loops = [lp for lp in boundary_loops(welded_tris) if len(lp) >= 6]
    for k, wloop in enumerate(loops):
        loop = [rep[w] for w in wloop]
        ring = skull.pos[loop]
        if ring[:, 2].mean() < 160:  # the neck cut is buried under the hide: no cap needed
            continue
        c = ring.mean(0)
        nrm_c = ring - c
        axis = np.linalg.svd(nrm_c)[2][2]  # plane normal
        # orient outward: away from the skull centre
        if np.dot(axis, c - np.array([0, 200, 185])) < 0:
            axis = -axis
        cen = c - axis * 2.0
        P = np.vstack([ring, cen])
        N = np.vstack([np.repeat(axis[None], len(ring), 0), axis])
        # planar UVs into a small chart
        e1 = np.linalg.svd(nrm_c)[2][0]
        e2 = np.cross(axis, e1)
        pu = (P - c) @ e1
        pv = (P - c) @ e2
        span = max(np.ptp(pu), np.ptp(pv), 1e-3)
        UV = np.stack([(pu - pu.min()) / span, (pv - pv.min()) / span], 1)
        n = len(ring)
        cap = Part(f"skull_cap{k}", P, N, UV, tris=[(n, i, (i + 1) % n) for i in range(n)])
        cap.chart = "skull_cap"
        cap.chart_size = (float(min(span, 12.0)), float(min(span, 12.0)))
        rim_drw = drw_v[loop]
        cap.drw = np.concatenate([rim_drw, [np.bincount(rim_drw).argmax()]]).astype(np.int32)
        cap.kind = "skull"
        fix_winding(cap)
        caps.append(cap)
    return [skull, mouth] + caps + fit_teeth(whole, drw_v)


def raycast(tri_pts, origin, direction):
    """Nearest hit of a ray against triangles (T, 3, 3). Returns (t, face normal, triangle) or None."""
    v0, v1, v2 = tri_pts[:, 0], tri_pts[:, 1], tri_pts[:, 2]
    e1, e2 = v1 - v0, v2 - v0
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    pv = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pv)
    ok = np.abs(det) > 1e-9
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    tv = np.asarray(origin, float) - v0
    u = np.einsum("ij,ij->i", tv, pv) * inv
    qv = np.cross(tv, e1)
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
    if not hit.any():
        return None
    k = np.nonzero(hit)[0][np.argmin(t[hit])]
    n = -np.cross(e1[k], e2[k])  # GX clockwise front faces -> outward
    return t[k], n / np.linalg.norm(n), k


def fit_teeth(skull: Part, drw_v) -> list[Part]:
    """Teeth grown out of the sculpted skull itself: an incisor arc under the upper lip, a lower arc on the
    chin, and rows of cheek teeth along both sides of the jaw (exposed through the torn cheeks). Each tooth
    rides on the matrix of the bone it is rooted in, so the jaw still opens."""
    from geom import loft
    from scipy.spatial import cKDTree
    tris = np.asarray(skull.tris)
    T = skull.pos[tris]
    tree = cKDTree(skull.pos)
    teeth = []

    def tooth(name, root, tip, width, thick, radial):
        root, tip = np.asarray(root, float), np.asarray(tip, float)
        n = 5
        tt = np.linspace(0, 1, n)
        c = root + tt[:, None] * (tip - root)
        taper = np.array([0.85, 1.0, 1.0, 0.92, 0.62])  # crown bulge, rounded tip
        p = loft(name, c, thick * taper, width * taper, sides=6, normal_hints=np.repeat(radial[None], n, 0),
                 power=2.8, cap_start=False)
        p.chart, p.kind = "tooth", "tooth"
        p.drw = np.full(len(p.pos), drw_v[tree.query(root)[1]], np.int32)
        teeth.append(p)

    # incisors: hang from the underside of the muzzle, around its front
    for x in (-4.3, -2.6, -0.9, 0.9, 2.6, 4.3):
        for z in np.arange(219.0, 203.0, -0.5):  # the frontmost underside hit near the lip
            h = raycast(T, (x, 135.0, z), (0, 1, 0))
            if h is not None and 135.0 + h[0] < 162:
                break
        y0 = 135.0 + h[0]
        radial = np.array([x * 0.25, 0.0, 1.0])
        radial /= np.linalg.norm(radial)
        tooth(f"inc_up{x:+.1f}", (x * 0.97, y0 + 1.6, z - 0.4), (x, y0 - 3.4, z + 0.4), 1.25, 0.72, radial)
    # lower incisors stand on the chin, under the slot
    for x in (-3.9, -2.4, -0.8, 0.8, 2.4, 3.9):
        for z in np.arange(210.0, 194.0, -0.5):
            h = raycast(T, (x, 162.0, z), (0, -1, 0))  # from inside the mouth slot down onto the chin
            if h is not None and h[0] < 16:
                break
        y0 = 162.0 - h[0]
        radial = np.array([x * 0.25, 0.0, 1.0])
        radial /= np.linalg.norm(radial)
        tooth(f"inc_lo{x:+.1f}", (x * 0.97, y0 - 1.6, z - 0.6), (x, y0 + 3.0, z + 0.3), 1.15, 0.68, radial)
    # cheek teeth: upper and lower rows along the side of the jaw, bitten together on the occlusal line
    occ_z = [176.0, 182.0, 188.0, 194.0, 198.5]
    occ_y = [178.0, 173.5, 169.0, 166.5, 165.8]
    for sx in (1, -1):
        for k, z in enumerate(np.linspace(178.0, 196.0, 5)):
            y = float(np.interp(z, occ_z, occ_y))
            h = raycast(T, (sx * 40.0, y, z), (-sx, 0, 0))
            if h is None:
                continue
            xs = sx * 40.0 - sx * h[0]
            n = h[1] if np.sign(h[1][0]) == sx else -h[1]
            radial = np.array([sx, 0.0, 0.0]) * 0.8 + n * 0.2
            radial /= np.linalg.norm(radial)
            face = np.array([xs, y, z]) + radial * 0.35
            tooth(f"cheek_up{sx}{k}", face - radial * 1.8 + [0, 4.6, 0], face + [0, 0.25, 0], 1.75, 1.05, radial)
            tooth(f"cheek_lo{sx}{k}", face - radial * 1.8 - [0, 4.2, 0], face - [0, 0.25, 0], 1.7, 1.0, radial)
    return teeth


def make_teeth() -> list[Part]:
    """Real teeth in the mouth slot: upper row on mouth1 (the muzzle), lower row on mouth2 (the jaw)."""
    from geom import loft
    teeth = []

    def tooth(name, root, tip, width, thick, bind, tangent):
        root, tip = np.asarray(root, float), np.asarray(tip, float)
        n = 4
        tt = np.linspace(0, 1, n)
        c = root + tt[:, None] * (tip - root)
        taper = 1.0 - 0.38 * tt ** 2
        radial = np.cross(tip - root, tangent)
        radial /= np.linalg.norm(radial)
        p = loft(name, c, thick * taper, width * taper, sides=6, normal_hints=np.repeat(radial[None], n, 0),
                 power=2.6, cap_start=False)
        p.chart = "tooth"
        p.kind = "tooth"
        p.bind = bind
        teeth.append(p)

    # upper incisors hang from the muzzle's underside (joint mouth1)
    for x in (-4.0, -2.6, -0.9, 0.9, 2.6, 4.0):
        zc = 207.7 - 0.075 * x * x
        edge = 151.3 + 2.1 * (abs(x) / 4.0) ** 2
        tangent = np.array([1.0, 0.0, -0.15 * x])
        tooth(f"inc_up{x:+.1f}", (x, edge + 1.7, zc + 0.2), (x * 1.02, edge - 2.3, zc - 0.7), 1.2, 0.7, 19,
              tangent / np.linalg.norm(tangent))
    # lower incisors stand on the chin (joint mouth2)
    for x in (-3.8, -2.45, -0.85, 0.85, 2.45, 3.8):
        zc = 204.2 - 0.07 * x * x
        edge = 148.1 + 4.6 * (abs(x) / 4.0) ** 1.5
        tangent = np.array([1.0, 0.0, -0.14 * x])
        tooth(f"inc_lo{x:+.1f}", (x, edge - 1.6, zc - 0.6), (x * 1.02, edge + 2.8, zc + 0.9), 1.15, 0.68, 20,
              tangent / np.linalg.norm(tangent))
    # cheek teeth toward the corners of the mouth
    for sx in (1, -1):
        for zc, edge, h in ((203.3, 158.4, 4.6), (200.9, 162.2, 4.8)):
            tooth(f"cheek_up{sx}{zc}", (sx * 4.3, edge + 1.6, zc), (sx * 4.2, edge - h * 0.7, zc + 0.2), 1.35, 0.95,
                  19, np.array([0.0, 0.0, 1.0]))
        tooth(f"cheek_lo{sx}", (sx * 4.8, 150.6, 201.2), (sx * 4.7, 154.6, 201.4), 1.35, 0.95, 20,
              np.array([0.0, 0.0, 1.0]))
    return teeth


def _weld(pos):
    _, w = np.unique(np.round(pos, 3), axis=0, return_inverse=True)
    return w.reshape(-1)


def _laplacian(pos, tris, weld, iterations=2, lam=0.5):
    """Smooth on welded vertices (seam copies move together); boundary vertices stay put."""
    from collections import defaultdict
    nw = weld.max() + 1
    nbr = defaultdict(set)
    ecount = defaultdict(int)
    for a, b, c in tris:
        for u, v in ((a, b), (b, c), (c, a)):
            wu, wv = weld[u], weld[v]
            nbr[wu].add(wv)
            nbr[wv].add(wu)
            ecount[(min(wu, wv), max(wu, wv))] += 1
    boundary = set()
    for (u, v), n in ecount.items():
        if n == 1:
            boundary |= {u, v}
    P = np.zeros((nw, 3))
    P[weld] = pos
    for _ in range(iterations):
        Q = P.copy()
        for w_, ns in nbr.items():
            if w_ in boundary or not ns:
                continue
            Q[w_] = P[w_] + lam * (P[list(ns)].mean(0) - P[w_])
        P = Q
    return P[weld]


def _renormal(part: Part, weld=None):
    P = part.pos
    if weld is None:
        weld = _weld(P)
    acc = np.zeros((weld.max() + 1, 3))
    for a, b, c in part.tris:
        n = np.cross(P[b] - P[a], P[c] - P[a])  # inward (GX clockwise) -> negate below
        acc[weld[a]] -= n
        acc[weld[b]] -= n
        acc[weld[c]] -= n
    acc = acc[weld]
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    new = acc / np.maximum(ln, 1e-9)
    # keep original orientation convention (outward): flip if it disagrees with the old normals
    flip = np.sum(new * part.nrm, axis=1) < 0
    if flip.mean() > 0.5:
        new = -new
    part.nrm = np.where(ln > 1e-9, new, part.nrm)


# ------------------------------------------------------------------------------------------ atlas packing
def occupied_mask(parts_fixed: list[Part], cell: int = 8) -> np.ndarray:
    g = ATLAS // cell
    occ = np.zeros((g, g), bool)
    for p in parts_fixed:
        tri = np.asarray(p.triangles())  # strips and triangle lists
        if len(tri) == 0:
            continue
        for a, b, c in tri:
            uv = p.uv[[a, b, c]]
            uv = uv - np.floor(uv.mean(0))
            lo = np.floor(uv.min(0) * g).astype(int)
            hi = np.ceil(uv.max(0) * g).astype(int)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x0, y0 = lo[0] + dx * g, lo[1] + dy * g
                    x1, y1 = hi[0] + dx * g, hi[1] + dy * g
                    xa, ya = max(x0, 0), max(y0, 0)
                    xb, yb = min(x1, g), min(y1, g)
                    if xa < xb and ya < yb:
                        occ[ya:yb, xa:xb] = True
    return ndimage.binary_dilation(occ, iterations=1)


def pack_charts(parts: list[Part], occ: np.ndarray, cell: int = 8):
    """Assign atlas rects to charts (shared by parts with the same chart id). Returns density used."""
    charts = {}
    for p in parts:
        if getattr(p, "fixed_uv", False):
            continue
        charts.setdefault(p.chart, []).append(p)
    sizes = {c: (max(p.chart_size[0] for p in ps), max(p.chart_size[1] for p in ps)) for c, ps in charts.items()}
    g = occ.shape[0]
    for density in np.arange(6.4, 1.0, -0.2):  # atlas px per model unit
        free = ~occ.copy()
        placed = {}
        ok = True
        for c in sorted(sizes, key=lambda k: -sizes[k][0] * sizes[k][1]):
            w, h = sizes[c]
            cw = int(np.ceil(w * density / cell)) + 1
            ch = int(np.ceil(h * density / cell)) + 1
            best = None
            for rot, (a, b) in enumerate(((cw, ch), (ch, cw))):
                if a > g or b > g:
                    continue
                s = np.pad(np.cumsum(np.cumsum(free, 0), 1), ((1, 0), (1, 0)))
                tot = s[b:, a:] - s[:-b, a:] - s[b:, :-a] + s[:-b, :-a]
                ys, xs = np.nonzero(tot == a * b)
                if len(ys):
                    k = np.argmin(ys * g + xs)
                    cand = (ys[k] * g + xs[k], ys[k], xs[k], a, b, rot)
                    if best is None or cand[0] < best[0]:
                        best = cand
            if best is None:
                ok = False
                break
            _, y0, x0, a, b, rot = best
            free[y0:y0 + b, x0:x0 + a] = False
            placed[c] = (x0 * cell, y0 * cell, a * cell, b * cell, rot)
        if ok:
            break
    else:
        raise RuntimeError("atlas full")
    pad = 3.0
    for c, ps in charts.items():
        x0, y0, a, b, rot = placed[c]
        for p in ps:
            u, v = p.uv[:, 0], p.uv[:, 1]
            if rot:
                u, v = v, u
            p.uv = np.stack([(x0 + pad + u * (a - 2 * pad)) / ATLAS, (y0 + pad + v * (b - 2 * pad)) / ATLAS], 1)
            p.atlas_rect = (x0, y0, a, b)
    return density, placed


# ------------------------------------------------------------------------------------------ rigging
def assign_drw(part: Part, model: bmd.BMD, drw: bmdwrite.DrwLayout, envs: dict) -> None:
    if part.drw is not None:
        return
    if isinstance(part.bind, int):
        part.drw = np.full(len(part.pos), drw.rigid(part.bind), np.int32)
        return
    if part.bind == "neck_blend":
        def env(*jw):
            return envs[frozenset((j, round(w, 2)) for j, w in jw)]
        z = part.pos[:, 2]
        d = np.full(len(z), drw.rigid(2), np.int32)
        d[z >= 50] = env((2, 0.75), (11, 0.25))
        d[z >= 60] = env((2, 0.5), (11, 0.5))
        d[z >= 72] = drw.rigid(11)
        d[z >= 100] = env((11, 0.75), (12, 0.25))
        d[z >= 118] = env((11, 0.5), (12, 0.5))
        d[z >= 132] = drw.rigid(12)
        d[z >= 150] = env((12, 0.5), (15, 0.5))
        d[z >= 158] = drw.rigid(15)
        part.drw = d
        return
    raise ValueError(f"part {part.name} has no binding")


def raw_attrs(part: Part, model: bmd.BMD, drw: bmdwrite.DrwLayout):
    """Positions/normals in the space the vertex's draw matrix expects (joint-local if rigid)."""
    pos = part.pos.copy()
    nrm = part.nrm.copy()
    for d in np.unique(part.drw):
        sel = part.drw == d
        if drw.flags[d] == 0:  # rigid: joint local
            j = drw.index[d]
            inv = np.linalg.inv(model.joint_world[j])
            pos[sel] = part.pos[sel] @ inv[:3, :3].T + inv[:3, 3]
            nrm[sel] = part.nrm[sel] @ inv[:3, :3].T
    return pos, nrm


# ------------------------------------------------------------------------------------------ layers
# one pixel (1024 layout) inside each tack island of the original body texture
TACK_SEEDS = {"saddle": (720, 40), "fender": (900, 200), "blanket": (100, 700), "stirrup_strap": (995, 900),
              "seat_trim": (820, 975), "pendant": (560, 820)}
DROP_ISLANDS = {"pendant"}
HEAD_ISLAND_SEED = (700, 500)  # a pixel of the main head/neck skin island
LEG_JOINTS = {3: 1, 4: 2, 5: 3, 6: 4, 7: 1, 8: 2, 9: 3, 10: 4, 27: 1, 28: 2, 29: 3, 30: 4, 31: 1, 32: 2, 33: 3,
              34: 4}  # joint -> leg segment


def classify_body(model: bmd.BMD, body: bmd.Mesh) -> dict:
    """Per-triangle masks over the original body mesh."""
    maps = bake.bake([body], "hs_body", (LAYOUT, LAYOUT))
    lab, _ = ndimage.label(maps["mask"])
    seeds = {name: lab[y, x] for name, (x, y) in TACK_SEEDS.items()}
    assert all(seeds.values()), seeds
    px = np.clip(uv_px(body.uv[body.tris].mean(1)).astype(int), 0, LAYOUT - 1)
    isl = lab[px[:, 1], px[:, 0]]
    for i in np.nonzero(isl == 0)[0]:  # centroid fell in a gap: use a vertex
        for v in body.tris[i]:
            q = np.clip(uv_px(body.uv[v]).astype(int), 0, LAYOUT - 1)
            if lab[q[1], q[0]]:
                isl[i] = lab[q[1], q[0]]
                break
    jt = body.joint[body.tris]
    cen = body.pos[body.tris].mean(1)
    saddle = np.isin(jt, list(SADDLE_JOINTS)).any(1)
    tack = np.isin(isl, [seeds[k] for k in seeds if k not in DROP_ISLANDS]) | saddle
    head_j = np.isin(jt, [15, 19, 20]).any(1)
    chest_straps = (cen[:, 2] > 70) & (cen[:, 1] < 185) & ~head_j & ~saddle & tack
    drop = np.isin(isl, [seeds[k] for k in DROP_ISLANDS]) | chest_straps
    tack &= ~drop
    hide = ~tack & ~drop
    ears = np.isin(jt, list(EAR_JOINTS)).any(1)
    forelock = (jt == FORELOCK_JOINT).all(1)
    head = hide & ~ears & ~forelock & (cen[:, 2] > 146) & (cen[:, 1] > 135)
    head_isl = lab[HEAD_ISLAND_SEED[1], HEAD_ISLAND_SEED[0]]
    inner_mouth = head & (isl != head_isl) & (cen[:, 2] > 185) & (cen[:, 1] < 172)
    headish = (cen[:, 2] > 140) & (cen[:, 1] > 135) & (head | ears | forelock | head_j)
    hoof = cen[:, 1] < 11
    return dict(tack=tack, hide=hide, head=head, inner_mouth=inner_mouth, blanket=isl == seeds["blanket"],
                muscle=hide & ~headish & ~hoof)


def tile_split(part: Part) -> Part:
    """Bring each triangle's UVs into [0,1) (the source wraps across tiles), splitting shared vertices when
    triangles disagree on the tile. Needed before squeezing a layout into one atlas quadrant."""
    tris = np.asarray(part.tris if part.tris else part.triangles())
    offs = np.floor(part.uv[tris].mean(1))
    key_to_new, pos, nrm, uv, src = {}, [], [], [], []
    new_tris = []
    for t, off in zip(tris, offs):
        nt = []
        for v in t:
            k = (int(v), int(off[0]), int(off[1]))
            if k not in key_to_new:
                key_to_new[k] = len(src)
                src.append(v)
                uv.append(np.clip(part.uv[v] - off, 0.0, 1.0))
            nt.append(key_to_new[k])
        new_tris.append(tuple(nt))
    src = np.array(src)
    out = Part(part.name, part.pos[src], part.nrm[src], np.array(uv), tris=new_tris, kind=part.kind)
    for attr in ("drw", "raw", "joint", "local_v"):
        val = getattr(part, attr, None)
        if val is not None:
            setattr(out, attr, np.asarray(val)[src])
    out.fixed_uv = True
    return out


def quadrant(part: Part, qx: int, qy: int) -> Part:
    part.uv = part.uv * 0.5 + np.array([qx * 0.5, qy * 0.5])
    return part


def welded_normals(pos, nrm, tris):
    """Area-weighted normals averaged over coincident vertices, so offset shells stay closed at seams."""
    weld = _weld(pos)
    acc = np.zeros((weld.max() + 1, 3))
    for a, b, c in tris:
        n = -np.cross(pos[b] - pos[a], pos[c] - pos[a])  # GX clockwise front faces -> outward
        acc[weld[a]] += n
        acc[weld[b]] += n
        acc[weld[c]] += n
    n = acc[weld]
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-9), nrm)
    return n


def shell(src: Part, depth, name: str, kind: str, inward: bool) -> Part:
    """Copy of a surface pushed `depth` units inward; inward=True flips it to face into the body."""
    n = welded_normals(src.pos, src.nrm, src.tris)
    d = depth(src) if callable(depth) else np.full(len(src.pos), float(depth))
    p = Part(name, src.pos - n * d[:, None], (-n if inward else n).copy(), src.uv.copy(),
             tris=[(a, c, b) for a, b, c in src.tris] if inward else list(src.tris), kind=kind)
    p.drw = src.drw.copy()
    p.joint = src.joint.copy()
    p.fixed_uv = True
    return p


def muscle_depth(part: Part):
    seg_ = np.array([LEG_JOINTS.get(int(j), 0) for j in part.joint])
    return np.select([seg_ >= 3, seg_ == 2], [1.0, 1.8], 2.6)


# ------------------------------------------------------------------------------------------ bones
KEEP_BONES = ("rib", "sternum", "sacrum", "radius", "carpus", "acc_carpal", "cannon", "sesamoid", "pastern", "femur",
              "patella", "tibia", "tarsus", "calcaneus", "mt_cannon", "mt_sesamoid", "mt_pastern")


def select_bones(parts: list[Part]) -> list[Part]:
    out = []
    for p in parts:
        base = p.name.rsplit("_", 1)[0] if p.name.endswith(("_L", "_R")) else p.name
        is_vert = p.name[:1] in "TLC" and p.name[1:2].isdigit() or p.name.startswith("Cd")
        if is_vert:
            if p.name.startswith("Cd"):
                continue  # tail vertebrae sat on top of the tail root: gone
            if p.name.endswith("_body") or p.name.startswith("C1_wing"):
                out.append(p)
            continue
        if base in KEEP_BONES or base.startswith("rib"):
            out.append(p)
    return out


def fit_inside(bones: list[Part], surface: Part, margin=1.2, skip=("Cd",)):
    """Move bone vertices that stick out of (or graze) the inner body shell back inside it. Pushes are
    smoothed over each bone so it shifts or narrows as a whole instead of getting dented."""
    from scipy.spatial import cKDTree
    S, SN = _surface_samples(surface, spacing=1.0)
    tree = cKDTree(S)
    moved = {}
    for p in bones:
        if p.name.startswith(skip):
            continue
        local = cKDTree(p.pos)
        nb = local.query_ball_point(p.pos, r=3.0)
        total = 0.0
        for _ in range(6):
            dist, idx = tree.query(p.pos, k=6)
            sd = np.einsum("ijk,ijk->ij", p.pos[:, None, :] - S[idx], SN[idx]).max(1)
            push = np.where(dist[:, 0] < 12.0, np.maximum(sd + margin, 0.0), 0.0)
            if push.max() < 0.05:
                break
            vec = SN[idx[:, 0]] * push[:, None]
            smooth = np.array([vec[n].mean(0) for n in nb])
            vec = np.where(np.linalg.norm(smooth, axis=1, keepdims=True) > np.linalg.norm(vec, axis=1,
                                                                                    keepdims=True) * 0.5,
                           smooth, vec)
            p.pos = p.pos - vec
            total += float(push.max())
        if total > 0:
            moved[p.name] = total
    return moved


def cavity_envelope(inner: Part):
    """Radius of the body cavity around (0, cy(z)) by height angle, measured from the inner shell."""
    P, _ = _surface_samples(inner, spacing=1.2)
    zs = np.arange(-60, 96, 4.0)
    ths = np.arange(0, 181, 6.0)
    cys, R = [], np.zeros((len(zs), len(ths)))
    for i, z in enumerate(zs):
        q = P[np.abs(P[:, 2] - z) < 3.0]
        c = q[np.abs(q[:, 0]) < 6]
        cy = (c[:, 1].min() + c[:, 1].max()) / 2 if len(c) else 160.0
        cy = min(cy, 165.0)
        cys.append(cy)
        th = np.degrees(np.arctan2(np.abs(q[:, 0]), q[:, 1] - cy))
        rad = np.hypot(q[:, 0], q[:, 1] - cy)
        for k, t in enumerate(ths):
            sel = np.abs(th - t) < 5
            R[i, k] = np.percentile(rad[sel], 20) if sel.sum() > 3 else np.nan
        row = R[i]
        good = ~np.isnan(row)
        R[i] = np.interp(ths, ths[good], row[good]) if good.any() else 40.0
    R = ndimage.uniform_filter(R, size=(3, 3), mode="nearest")
    cys = np.array(cys)

    def center(z):
        return float(np.interp(z, zs, cys))

    def radius(z, th):
        zi = np.interp(z, zs, np.arange(len(zs)))
        ti = np.interp(th, ths, np.arange(len(ths)))
        return float(ndimage.map_coordinates(R, [[zi], [ti]], order=1, mode="nearest")[0])

    return center, radius


def _surface_samples(part: Part, spacing=1.0):
    tris = np.asarray(part.tris if part.tris else part.triangles())
    a, b, c = part.pos[tris[:, 0]], part.pos[tris[:, 1]], part.pos[tris[:, 2]]
    na, nb, nc = part.nrm[tris[:, 0]], part.nrm[tris[:, 1]], part.nrm[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    k = np.maximum(1, np.ceil(area / spacing ** 2)).astype(int)
    idx = np.repeat(np.arange(len(tris)), k)
    r1, r2 = np.random.default_rng(1).random((2, len(idx)))
    s = np.sqrt(r1)
    w0, w1, w2 = (1 - s), s * (1 - r2), s * r2
    P = a[idx] * w0[:, None] + b[idx] * w1[:, None] + c[idx] * w2[:, None]
    N = na[idx] * w0[:, None] + nb[idx] * w1[:, None] + nc[idx] * w2[:, None]
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
    return P, N


# ------------------------------------------------------------------------------------------ gore strands
def make_strands(hide: Part, seed: int = 7) -> list[Part]:
    """Sinew and flesh stretched across the wounds, and torn strips dangling from their edges.

    Ends are rooted just under the torn hide and ride on that spot's own skin matrix, so a strand
    stretches with the body like the skin around it."""
    from scipy.spatial import cKDTree
    import paint_wight
    from geom import loft
    rng = np.random.default_rng(seed)
    S, SN = _surface_samples(hide, spacing=0.9)
    vtree = cKDTree(hide.pos)
    J = hide.joint[vtree.query(S)[1]]
    H, _ = paint_wight.wounds(S, SN, J)
    stree = cKDTree(S)
    edge = np.nonzero((H > -1.6) & (H < -0.2) & (S[:, 1] > 25))[0]
    edge = edge[rng.permutation(len(edge))]
    parts, used = [], []

    def far_from_used(p, r=4.0):
        return all(np.linalg.norm(p - q) > r for q in used)

    def build(name, pts, radius, flat, bind_a, bind_b):
        n = len(pts)
        t = np.linspace(0, 1, n)
        r = radius * (1.0 - 0.35 * np.sin(np.pi * t))  # thinner where stretched
        hint = np.cross(pts[-1] - pts[0], [0.0, 1.0, 0.0])
        hint = hint / max(np.linalg.norm(hint), 1e-6) if np.linalg.norm(hint) > 1e-6 else np.array([1.0, 0, 0])
        p = loft(name, pts, r * flat, r, sides=5, normal_hints=np.repeat(hint[None], n, 0))
        rings = len(p.pos) // 1
        ring_t = np.clip(p.uv[:, 1], 0, 1)
        p.drw = np.where(ring_t < 0.5, bind_a, bind_b).astype(np.int32)
        p.chart, p.kind = "strand", "strand"
        parts.append(p)

    # stretched across a wound
    n_across = 0
    for i in edge:
        if n_across >= 34:
            break
        a = S[i]
        if not far_from_used(a):
            continue
        d = np.linalg.norm(S[edge] - a, axis=1)
        cand = edge[(d > 7) & (d < 24)]
        if len(cand) == 0:
            continue
        j = cand[rng.integers(len(cand))]
        b = S[j]
        mid = (a + b) / 2
        k = stree.query(mid)[1]
        if H[k] < 1.5 or not far_from_used(b):
            continue
        L = np.linalg.norm(b - a)
        inward = -(SN[i] + SN[j]) / 2
        tt = np.linspace(0, 1, 7)[:, None]
        sag = np.sin(np.pi * tt) * (np.array([0, -1.0, 0]) * 0.18 * L + inward * 1.2)
        pts = a + (b - a) * tt + sag - (SN[i] * (1 - tt) + SN[j] * tt) * 0.9
        rad = rng.uniform(0.6, 1.35)
        da = hide.drw[vtree.query(a)[1]]
        db = hide.drw[vtree.query(b)[1]]
        build(f"strand{n_across}", pts, rad, rng.uniform(0.4, 1.0), da, db)
        used += [a, b]
        n_across += 1
    # torn strips hanging from wound edges
    n_hang = 0
    for i in edge[::-1]:
        if n_hang >= 18:
            break
        a = S[i]
        if not far_from_used(a, 5.0) or SN[i][1] > 0.2:  # hang from lower or side edges
            continue
        L = rng.uniform(5.0, 13.0)
        tt = np.linspace(0, 1, 6)[:, None]
        out = SN[i] * np.sin(np.pi * tt * 0.5) * 1.2
        pts = a - SN[i] * 0.8 + out + np.array([0, -1.0, 0]) * L * tt ** 1.2 + np.array([0, 0, 1.0]) * \
            rng.uniform(-1.5, 1.5) * tt
        d = hide.drw[vtree.query(a)[1]]
        build(f"tatter{n_hang}", pts, rng.uniform(0.9, 1.7), 0.3, d, d)
        used.append(a)
        n_hang += 1
    return parts


def make_intercostals(bones: list[Part], drw: bmdwrite.DrwLayout, seed: int = 11) -> list[Part]:
    """Dried strips of flesh webbed between neighbouring ribs. Each strip rides on its ribs' matrices."""
    from geom import loft
    rng = np.random.default_rng(seed)
    ribs = {p.name: p for p in bones if p.name.startswith("rib")}
    out = []
    for side in ("L", "R"):
        for i in range(1, 13):
            a, b = ribs.get(f"rib{i}_{side}"), ribs.get(f"rib{i + 1}_{side}")
            if a is None or b is None:
                continue
            ca, cb = _ring_centres(a), _ring_centres(b)
            for k in range(rng.integers(1, 4)):
                ta, tb = rng.uniform(0.35, 0.8), rng.uniform(0.35, 0.8)
                pa = ca[int(ta * (len(ca) - 1))]
                pb = cb[int(tb * (len(cb) - 1))]
                tt = np.linspace(0, 1, 5)[:, None]
                sag = np.sin(np.pi * tt) * np.array([0, -1.0, 0]) * rng.uniform(0.5, 2.5)
                pts = pa + (pb - pa) * tt + sag
                w = rng.uniform(0.9, 2.2)
                hint = np.array([np.sign(pa[0]) or 1.0, 0.0, 0.0])
                p = loft(f"icost{side}{i}_{k}", pts, 0.35 * np.ones(5), w * (1 - 0.3 * np.sin(np.pi * tt[:, 0])),
                         sides=5, normal_hints=np.repeat(hint[None], 5, 0))
                t_along = np.clip(p.uv[:, 1], 0, 1)
                p.drw = np.where(t_along < 0.5, drw.rigid(a.bind), drw.rigid(b.bind)).astype(np.int32)
                p.chart, p.kind = "strand", "strand"
                out.append(p)
    return out


def _ring_centres(p: Part, sides: int = 6):
    n = sides + 1
    m = len(p.strips) + 1
    return p.pos[:m * n].reshape(m, n, 3).mean(1)


# ------------------------------------------------------------------------------------------ build
def mat3_alpha_test(orig: bytes, material: str) -> tuple[int, bytes]:
    """MAT3 section with `material` switched to alpha-tested (alpha >= 128) like the mane, depth after texture."""
    import struct
    from gctex import _string_table, j3d_sections
    s = j3d_sections(orig)["MAT3"]
    size = struct.unpack_from(">I", orig, s + 4)[0]
    sec = bytearray(orig[s:s + size])
    n = struct.unpack_from(">H", sec, 8)[0]
    offs = struct.unpack_from(">30I", sec, 12)
    names = _string_table(bytes(sec), offs[2])
    remap = struct.unpack_from(f">{n}H", sec, offs[1])
    ref = offs[0] + remap[names.index("hair_m")] * 0x14C
    dst = offs[0] + remap[names.index(material)] * 0x14C
    sec[dst + 0x05] = sec[ref + 0x05]  # z compare after texturing
    sec[dst + 0x146:dst + 0x148] = sec[ref + 0x146:ref + 0x148]  # alpha compare: >= 128
    return s, bytes(sec)


def build():
    orig = (WORK / "hs.bmd").read_bytes()
    model = bmd.BMD(orig)
    meshes = model.meshes()
    by_mat = {m.material: m for m in meshes}
    body = by_mat["head_m"]
    drw = bmdwrite.DrwLayout(orig, [11, 35, 36])
    envs = envelope_drw(model, drw)
    cls = classify_body(model, body)

    def from_body(mask, name, kind):
        p = part_from_mesh(body, mask, name)
        p.drw = np.array([drw.remap_old(int(d)) for d in p.raw[:, 3]], np.int32)
        p.kind = kind
        return p

    # 1. outer hide and tack keep the original layout (top-left quadrant)
    hide = quadrant(tile_split(from_body(cls["hide"], "hide", "hide")), 0, 0)
    tack = quadrant(tile_split(from_body(cls["tack"], "tack", "tack")), 0, 0)
    bed_mesh = by_mat["z_nimotu_m"]
    bedroll = part_from_mesh(bed_mesh, np.ones(len(bed_mesh.tris), bool), "bedroll")
    bedroll.drw = np.array([drw.remap_old(int(d)) for d in bedroll.raw[:, 3]], np.int32)
    bedroll.kind = "tack"
    bedroll = quadrant(tile_split(bedroll), 0, 0)

    # 2./3. muscle shell and inward cavity shell (top-right quadrant)
    msrc = tile_split(from_body(cls["muscle"], "muscle_src", "muscle"))
    muscle = quadrant(shell(msrc, muscle_depth, "muscle", "muscle", inward=False), 1, 0)
    csrc = tile_split(from_body(cls["muscle"] | cls["blanket"], "cavity_src", "muscle"))
    cavity = quadrant(shell(csrc, lambda p: muscle_depth(p) + 0.8, "cavity", "muscle", inward=True), 1, 0)

    # 4. skull under the face (bottom-left quadrant), teeth, bones
    skull_parts = make_skull(body, cls["head"], cls["inner_mouth"], drw)
    for sp in skull_parts[:2]:
        split = quadrant(tile_split(sp), 0, 1)
        sp.__dict__.update(split.__dict__)
    inner = Part("inner", cavity.pos, -cavity.nrm, cavity.uv, tris=cavity.tris)  # outward-facing test surface
    skeleton.RIB_CENTER, skeleton.RIB_ENVELOPE = cavity_envelope(inner)
    bones = select_bones(skeleton.build_all())
    moved = fit_inside(bones, inner, margin=0.6)

    eyes = {}
    for mat in ("eye_m", "SC_eye_m"):
        m_ = by_mat[mat]
        e = part_from_mesh(m_, np.ones(len(m_.tris), bool), mat)
        e.drw = np.array([drw.remap_old(int(d)) for d in e.raw[:, 3]], np.int32)
        sg = np.sign(e.pos[:, 0])
        e.pos[:, 0] = e.pos[:, 0] * 0.9 - sg * 3.0  # sink the eye into the socket
        eyes[mat] = e

    fixed = [hide, tack, bedroll, muscle, cavity, skull_parts[0], skull_parts[1]]
    occ = occupied_mask(fixed)
    new_parts = skull_parts[2:] + bones
    for p in new_parts:
        p.local_v = p.uv[:, 1].copy()
    density, placed = pack_charts(new_parts, occ)
    for p in new_parts:
        assign_drw(p, model, drw, envs)

    strands = make_strands(hide) + make_intercostals(bones, drw)
    new_strands = strands
    for p in new_strands:
        p.local_v = p.uv[:, 1].copy()
    density2, placed2 = pack_charts(new_strands, occupied_mask(fixed + new_parts))
    skull_parts = [p for p in skull_parts if p.kind != "mouth"]  # the old inner lips dangled like a tongue
    body_parts = [hide, tack, muscle, cavity] + skull_parts + bones + strands
    shapes, positions, normals, uvs, model_pos = _assemble(model, meshes, drw, body_parts, eyes, bedroll)
    return dict(orig=orig, model=model, drw=drw, shapes=shapes, positions=positions, normals=normals, uvs=uvs,
                model_pos=model_pos, layers=dict(hide=hide, tack=tack, bedroll=bedroll, muscle=muscle,
                                                 cavity=cavity),
                parts=skull_parts + bones + strands, density=density, placed=placed, fitted=moved,
                mat3=mat3_alpha_test(orig, "head_m"))


def _assemble(model, meshes, drw, body_parts, eyes, bedroll):
    raw_pos, raw_nrm, raw_uv, model_pos_all = [], [], [], []

    def part_prims(p: Part):
        pr, nr = raw_attrs(p, model, drw)
        base = len(raw_pos)
        raw_pos.extend(pr)
        raw_nrm.extend(nr)
        raw_uv.extend(p.uv)
        model_pos_all.extend(p.pos)
        vx = [(int(p.drw[i]), base + i) for i in range(len(p.pos))]
        prims = [bmdwrite.Prim(bmdwrite.PRIM_STRIP, [vx[i] for i in s]) for s in p.strips]
        for t in p.tris:
            prims.append(bmdwrite.Prim(bmdwrite.PRIM_TRIANGLES, [vx[i] for i in t]))
        return prims

    def original_prims(mesh: bmd.Mesh):
        used, inv = np.unique(mesh.tris.reshape(-1), return_inverse=True)
        r = mesh.raw[used]
        base = len(raw_pos)
        raw_pos.extend(model.arrays[9][r[:, 0]])
        raw_nrm.extend(model.arrays[10][r[:, 1]])
        raw_uv.extend(model.arrays[13][r[:, 2]])
        model_pos_all.extend(mesh.pos[used])
        dr = [drw.remap_old(int(x)) for x in r[:, 3]]
        return [bmdwrite.Prim(bmdwrite.PRIM_TRIANGLES, [(dr[k], base + k) for k in tri])
                for tri in inv.reshape(-1, 3)]

    shapes = []
    for mesh in meshes:
        if mesh.material == "head_m":
            prims = []
            for p in body_parts:
                prims += part_prims(p)
            shapes.append(bmdwrite.OutShape(3, _merge_tri_prims(prims)))
        elif mesh.material in ("nuki_m", "hair_m"):
            shapes.append(bmdwrite.OutShape(3, _merge_tri_prims(original_prims(mesh))))
        elif mesh.material in eyes:
            shapes.append(bmdwrite.OutShape(0, _merge_tri_prims(part_prims(eyes[mesh.material]))))
        else:  # bedroll: its UVs moved into the atlas quadrant too
            shapes.append(bmdwrite.OutShape(0, _merge_tri_prims(part_prims(bedroll))))

    P = np.asarray(raw_pos, np.float32)
    N = np.clip(np.rint(np.asarray(raw_nrm) * 16384), -32768, 32767).astype(np.int16)
    T = np.clip(np.rint(np.asarray(raw_uv) * 4096), -32768, 32767).astype(np.int16)
    up, ip = np.unique(P, axis=0, return_inverse=True)
    un, inn = np.unique(N, axis=0, return_inverse=True)
    ut, it = np.unique(T, axis=0, return_inverse=True)
    ip, inn, it = ip.reshape(-1), inn.reshape(-1), it.reshape(-1)
    model_pos = np.zeros((len(up), 3))
    model_pos[ip] = np.asarray(model_pos_all)
    for sh in shapes:
        for pr in sh.prims:
            pr.verts = [(d, int(ip[v]), int(inn[v]), int(it[v])) for d, v in pr.verts]
    return shapes, up.astype(np.float64), un / 16384.0, ut / 4096.0, model_pos


def _merge_tri_prims(prims):
    """Pack consecutive triangle prims into bigger lists (<= 10 matrices each) for smaller display lists."""
    out = []
    cur = None
    for p in prims:
        if p.kind == bmdwrite.PRIM_TRIANGLES:
            if cur is not None and len({v[0] for v in cur.verts} | {v[0] for v in p.verts}) <= 10 \
                    and len(cur.verts) < 3000:
                cur.verts += p.verts
                continue
            cur = bmdwrite.Prim(bmdwrite.PRIM_TRIANGLES, list(p.verts))
            out.append(cur)
        else:
            out.append(p)
            cur = None
    return out
