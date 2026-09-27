"""Paint the body atlas of the undead ("wight") Epona.

The 2048 px atlas is baked back onto the layered model, so every texel knows its 3D position, normal,
joint and layer. Wounds are defined once in 3D and cut through the hide (alpha) and, a little smaller,
through the muscle, so the tears line up across UV seams and between layers:

    hide (rotting skin) -> dried muscle band -> bone / hollow body cavity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

import noise
import render
from noise import smoothstep
from paint_skeleton import lerp, luminance, occupancy_ao, ramp

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
SIZE = 2048

KIND = {"hide": 1, "tack": 2, "muscle": 3, "skull": 4, "mouth": 5, "cap": 6, "tooth": 7, "bone": 8,
        "strand": 9}

# ------------------------------------------------------------------------------------------ palette
SKIN_DARK = np.array([0.070, 0.060, 0.054])
SKIN_MID = np.array([0.215, 0.185, 0.158])
SKIN_LIGHT = np.array([0.380, 0.330, 0.280])
ROT = np.array([0.150, 0.155, 0.118])  # greenish-grey decay
MEAT_DARK = np.array([0.110, 0.030, 0.024])
MEAT = np.array([0.330, 0.100, 0.075])
MEAT_LIGHT = np.array([0.500, 0.230, 0.170])
FASCIA = np.array([0.640, 0.520, 0.450])
BLOOD = np.array([0.130, 0.030, 0.022])
BONE_LIGHT = np.array([0.760, 0.700, 0.580])
BONE_MID = np.array([0.560, 0.490, 0.380])
BONE_DARK = np.array([0.240, 0.195, 0.145])
TOOTH = np.array([0.800, 0.730, 0.560])
CAVITY = np.array([0.012, 0.010, 0.010])
MUD = np.array([0.150, 0.115, 0.080])
HORN = np.array([0.090, 0.080, 0.072])
LEATHER_DARK = np.array([0.045, 0.035, 0.030])
LEATHER = np.array([0.210, 0.160, 0.120])
GLOW = np.array([0.380, 0.950, 1.000])  # icy Twilight glow (the wight eyes)
GLOW_DEEP = np.array([0.020, 0.200, 0.240])

HEAD_J = (15, 16, 17, 18, 19, 20)
FRONT_LEG = {3: 1, 4: 2, 5: 3, 6: 4, 7: 1, 8: 2, 9: 3, 10: 4}
HIND_LEG = {27: 1, 28: 2, 29: 3, 30: 4, 31: 1, 32: 2, 33: 3, 34: 4}


# ------------------------------------------------------------------------------------------ baking
def bake(parts, size=SIZE):
    """UV-space rasterization of all body-atlas geometry: position, normal, kind, local v, joint."""
    h = w = size
    zbuf = np.full((h, w), np.inf)
    pos = np.zeros((h, w, 3), np.float32)
    nrm = np.zeros((h, w, 3), np.float32)
    kind = np.zeros((h, w), np.int8)
    lv = np.zeros((h, w), np.float32)
    joint = np.full((h, w), -1, np.int16)
    for pi, p in enumerate(parts):
        tris = np.asarray(p.tris if p.tris else p.triangles(), np.int32).reshape(-1, 3)
        if p.strips:
            tris = p.triangles()
        if len(tris) == 0:
            continue
        k = KIND.get(p.kind, KIND["bone"])
        v_local = getattr(p, "local_v", None)
        v_local = np.zeros(len(p.pos)) if v_local is None else v_local
        jn = getattr(p, "joint", None)
        jn = np.full(len(p.pos), -1) if jn is None else np.asarray(jn)
        uv = p.uv[tris]
        vid = tris.reshape(-1)
        xy = uv.reshape(-1, 2) * np.array([w, h])
        z = -np.abs(p.pos[vid, 0]) - 0.001 * pi
        z = np.where(p.pos[vid, 0] >= -1e-3, z, z + 1000)
        tri_joint = np.repeat(jn[tris[:, 0]], 3)
        attrs = np.concatenate([p.pos[vid], p.nrm[vid], v_local[vid][:, None], tri_joint[:, None]], 1)
        flat = np.arange(len(vid)).reshape(-1, 3)
        zbuf, abuf, tid = render.rasterize(xy, z, flat, (h, w), attrs, zbuf)
        cov = tid >= 0
        pos[cov] = abuf[cov][:, :3]
        nrm[cov] = abuf[cov][:, 3:6]
        lv[cov] = abuf[cov][:, 6]
        joint[cov] = np.rint(abuf[cov][:, 7]).astype(np.int16)
        kind[cov] = k
    n = np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = np.divide(nrm, n, out=np.zeros_like(nrm), where=n > 0)
    return dict(pos=pos, nrm=nrm, kind=kind, v=lv, joint=joint, mask=kind > 0)


# ------------------------------------------------------------------------------------------ wounds
def _ellipse(z, y, zc, yc, rz, ry):
    """Positive inside, roughly in units of distance from the rim."""
    return (1.0 - np.sqrt(((z - zc) / rz) ** 2 + ((y - yc) / ry) ** 2)) * min(rz, ry)


def wounds(P, N, J):
    """(hide_hole, muscle_hole) fields: a surface point is torn open where the value is > 0."""
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    rag = (noise.fbm(Pm / 8.0, 4, seed=11) - 0.5) * 16 + (noise.fbm(Pm / 2.0, 3, seed=12) - 0.5) * 4.5
    jj = J.astype(int)
    head = np.isin(jj, HEAD_J)
    fseg = np.vectorize(lambda j: FRONT_LEG.get(j, 0))(jj)
    hseg = np.vectorize(lambda j: HIND_LEG.get(j, 0))(jj)
    leg = (fseg > 0) | (hseg > 0)
    trunk = ~head & (fseg < 2) & (hseg < 2)  # body incl. shoulders/hips (skinned to the upper leg joints)
    side = x > 6
    H = np.full(len(P), -50.0)
    # ribcage: the lower barrel is rotted open on both flanks and under the belly
    H = np.maximum(H, np.where(trunk, _ellipse(z, y, 6, 134, 58, 32) + rag, -50))
    H = np.maximum(H, np.where(trunk & side, _ellipse(z, y, -52, 166, 16, 26) + rag * 0.8, -50))
    # neck: torn open along the side, vertebrae and neck muscle showing
    H = np.maximum(H, np.where(trunk & side, _ellipse(z, y, 102, 179, 30, 15) + rag * 0.8, -50))
    # shoulder and chest: torn meat
    H = np.maximum(H, np.where(trunk & side, _ellipse(z, y, 80, 168, 12, 15) + rag * 0.7, -50))
    H = np.maximum(H, np.where(trunk & (N[:, 2] > 0.4), _ellipse(x * 1.6, y, 0, 140, 14, 16) + rag * 0.6, -50))
    # hindquarter and thigh
    H = np.maximum(H, np.where(trunk & side, _ellipse(z, y, -90, 150, 14, 22) + rag * 0.7, -50))
    H = np.maximum(H, np.where(leg & (hseg == 1) & side, _ellipse(z, y, -86, 150, 13, 20) + rag * 0.7, -50))
    # face: skin gone from the nose bridge, around the eyes and the lips; patches cling to the forehead
    d_eye = np.hypot(y - 198.0, z - 191.0)
    face = np.maximum.reduce([
        _ellipse(z, y, 208, 186, 22, 24),  # muzzle and nasal bone
        13.0 - d_eye,  # orbit
        np.where(z > 193, _ellipse(z, y, 212, 157, 22, 9), -50),  # lips rotted off the teeth
        _ellipse(z, y, 187, 172, 14, 8.5),  # torn cheek over the molar rows
    ])
    cling = (noise.fbm(Pm / 6.0, 3, seed=13) - 0.66) * 30
    H = np.maximum(H, np.where(head & (jj != 16) & (jj != 17), np.minimum(face + rag * 0.6, 12 - cling), -50))
    # legs: forearm / gaskin tears, cannons skinned down to the bone at the front
    # the leg skin stays on around the back so the leg reads whole; tears open the front / outside
    outward = N[:, 0] * np.sign(P[:, 0]) > 0.55
    front = N[:, 2] > 0.35
    H = np.maximum(H, np.where((fseg == 2) & (front | outward), 7.5 - np.abs(y - 99.0) * 0.9 + rag * 0.5, -50))
    H = np.maximum(H, np.where((hseg == 2) & outward, 7.0 - np.abs(y - 106.0) * 0.8 + rag * 0.5, -50))
    H = np.maximum(H, np.where(((fseg == 3) | (hseg == 3)) & front & (y > 34),
                               np.minimum(y - 34.0, 62.0 - y) * 0.9 + rag * 0.45 - 2.5, -50))
    # ears: shrivelled and holed
    ear = np.isin(jj, (16, 17))
    H = np.maximum(H, np.where(ear, (noise.fbm(Pm / 2.2, 3, seed=14) - 0.6) * 30, -50))
    # scattered rot holes everywhere else
    H = np.maximum(H, (noise.fbm(Pm / 4.5, 4, seed=15) - 0.765) * 32)
    # croup: torn along the spine behind the saddle, what the rider sees
    top = N[:, 1] > 0.25
    croup = (1.0 - np.sqrt(((z + 88) / 24.0) ** 2 + (x / 10.0) ** 2)) * 10.0 + rag * 0.6
    H = np.maximum(H, np.where(trunk & top & (z < -58), croup, -50))
    # muscle tears are smaller: a band of torn meat shows before the bone / cavity
    band = np.where(leg, 1.2, 4.5)
    M = H - band
    # between the ribs the muscle is gone entirely where the hide is open
    return H, M


# ------------------------------------------------------------------------------------------ materials
def paint_hide(P, N, J, L, hp, sat, H, seed=100):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    mottle = noise.fbm(Pm / 16.0, 4, seed=seed)
    # Epona's own painted coat, drained of colour and dulled: keeps the TP look and her muscle shading
    shade = np.clip(L * 1.05 + hp * 1.2 + (mottle - 0.5) * 0.18, 0, 1)
    col = ramp(shade, (0.0, SKIN_DARK), (0.35, SKIN_MID * 1.35), (0.7, SKIN_LIGHT * 1.25),
               (1.0, np.array([0.56, 0.50, 0.44])))
    col = lerp(col, ROT, smoothstep(0.55, 0.8, noise.fbm(Pm / 7.0, 4, seed=seed + 1)) * 0.55)
    # bald leathery patches vs matted fur: fur keeps the painted hair strokes brighter
    bald = smoothstep(0.45, 0.62, noise.fbm(Pm / 11.0, 3, seed=seed + 2))
    col = lerp(col, col * np.array([0.72, 0.68, 0.66]) + 0.02, bald * 0.7)
    # feathering / mane-base hair (cream in the original): matted and filthy
    hair = (sat < 0.3) & (L > 0.42)
    hcol = ramp(np.clip((L - 0.35) / 0.55 + hp * 1.5, 0, 1), (0.0, np.array([0.05, 0.045, 0.04])),
                (1.0, np.array([0.34, 0.30, 0.25])))
    col = np.where(hair[:, None], hcol, col)
    # wound rims: skin curls back dark and wet
    rim = smoothstep(-2.0, 0.0, H)
    col = lerp(col, lerp(BLOOD * 0.7, MEAT * 0.6, noise.fbm(Pm / 1.1, 2, seed=seed + 3)), rim * 0.85)
    # dried blood running down from wounds above
    streak_n = noise.fbm(Pm * np.array([0.9, 0.07, 0.9]), 3, seed=seed + 4)
    run = np.zeros(len(P))
    return col, streak_n, run


def blood_runs(P, field_fn, seed=110):
    """How much blood has run down onto each point from wounds above it."""
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    acc = np.zeros(len(P))
    for dy, fall in ((2.5, 1.0), (6.0, 0.85), (11.0, 0.65), (18.0, 0.45), (27.0, 0.28)):
        acc = np.maximum(acc, smoothstep(-1.0, 1.5, field_fn(P + np.array([0, dy, 0]))) * fall)
    thin = smoothstep(0.52, 0.72, noise.fbm(Pm * np.array([1.1, 0.06, 1.1]), 3, seed=seed))
    return acc * thin


def paint_muscle(P, N, J, M, ao, seed=200):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    jj = J.astype(int)
    legish = np.isin(jj, list(FRONT_LEG) + list(HIND_LEG))
    along = np.where(legish[:, None], np.array([1.3, 0.12, 1.3]), np.array([1.1, 0.9, 0.13]))
    fib = noise.fbm(Pm * along, 3, seed=seed)
    fine = noise.fbm(Pm * along * 3.0, 2, seed=seed + 1)
    t = np.clip(fib * 0.8 + fine * 0.35 - 0.12, 0, 1)
    col = ramp(t, (0.0, MEAT_DARK), (0.55, MEAT), (1.0, MEAT_LIGHT))
    fascia = smoothstep(0.66, 0.8, noise.fbm(Pm * along * 0.7, 3, seed=seed + 2))
    col = lerp(col, FASCIA * (0.6 + 0.4 * fine)[:, None], fascia * 0.7)
    dry = smoothstep(0.55, 0.75, noise.fbm(Pm / 6.0, 3, seed=seed + 3))
    col = lerp(col, col * 0.45, dry * 0.6)  # blackened, dried-out patches
    col = lerp(col, BLOOD * 0.7, smoothstep(-1.5, 0.0, M) * 0.8)  # dark torn edges
    return col * (0.72 + 0.28 * ao)[:, None]


def paint_bone(P, N, ao, v, seed=300, clean=1.0):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    grain = noise.fbm(Pm * np.array([0.9, 0.35, 0.9]), 3, seed=seed)
    fine = noise.fbm(Pm * 2.4, 2, seed=seed + 1)
    stain = noise.fbm(Pm / 9.0, 4, seed=seed + 2)
    f1, f2 = noise.worley(Pm / 8.0, seed=seed + 3)
    crack = (1 - smoothstep(0.0, 0.026, f2 - f1)) * smoothstep(0.58, 0.72, noise.fbm(Pm / 9.0, 2, seed=seed + 4))
    ends = np.clip(np.maximum(1 - v / 0.13, (v - 0.87) / 0.13), 0, 1)
    light = np.clip(0.5 * N[:, 1] + 0.2, -1, 1)
    t = np.clip(0.42 + ao * 0.35 + light * 0.15 + (grain - 0.5) * 0.4 + (fine - 0.5) * 0.14 - ends * 0.15, 0, 1)
    col = ramp(t, (0.0, BONE_DARK), (0.45, BONE_MID), (1.0, BONE_LIGHT))
    col = lerp(col, col * np.array([0.78, 0.70, 0.58]), smoothstep(0.45, 0.8, stain) * 0.85)
    col = lerp(col, BONE_DARK * 0.5, crack * 0.7)
    # flesh and dried blood still clinging near joints and in crevices
    cling = np.clip(ends * 0.8 + (1 - ao) * 1.1, 0, 1) * smoothstep(0.45, 0.7, noise.fbm(Pm / 2.5, 3, seed=seed + 5))
    cling = np.maximum(cling * smoothstep(0.45, 0.7, noise.fbm(Pm / 5.0, 2, seed=seed + 7)),
                       smoothstep(0.6, 0.72, noise.fbm(Pm / 3.0, 3, seed=seed + 8)))  # rotten tissue in places
    col = lerp(col, lerp(BLOOD, MEAT * 0.85, noise.fbm(Pm / 1.2, 2, seed=seed + 6)), cling * 0.85 * clean)
    return col * (0.78 + 0.22 * ao)[:, None]


def paint_skull(P, N, ao, seed=400):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    col = paint_bone(P, N, ao, np.full(len(P), 0.5), seed=seed, clean=0.6)
    d_eye = np.hypot(y - 198.0, z - 191.0)
    side = smoothstep(9.0, 13.0, x)
    sock = smoothstep(10.0, 7.0, d_eye) * side
    col = lerp(col, lerp(CAVITY, GLOW_DEEP, smoothstep(7.0, 2.5, d_eye) * 0.7), sock)
    col = lerp(col, col * 0.5, smoothstep(13.5, 10.0, d_eye) * (1 - sock) * side)
    notch = np.where(x > 4, np.hypot((z - 207.0) / 11.0, (y - 176.0) / 3.2), 9.0)
    col = lerp(col, CAVITY, smoothstep(1.05, 0.8, notch))
    nostril = np.hypot((x - 6.5) / 3.6, (y - 171.0) / 5.0)
    col = lerp(col, CAVITY, smoothstep(1.1, 0.8, nostril) * (z > 214) * (N[:, 2] > 0.2))
    # old blood around the mouth and running from the sockets
    Pm = np.stack([x, y, z], 1)
    gore = np.maximum(smoothstep(170.0, 156.0, y) * (z > 190),
                      smoothstep(16.0, 8.0, d_eye) * smoothstep(198.0, 185.0, y))
    gore *= smoothstep(0.45, 0.65, noise.fbm(Pm * np.array([1.0, 0.25, 1.0]), 3, seed=seed + 1))
    return lerp(col, BLOOD, gore * 0.75)


def paint_tack(P, L, hp, seed=500):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    t = np.clip((L - 0.06) / 0.55 + hp * 1.4, 0, 1)
    col = ramp(t, (0.0, LEATHER_DARK), (0.55, LEATHER * 0.8), (1.0, LEATHER * 1.5))
    col = lerp(col, col * np.array([0.75, 0.8, 0.85]), 0.3)  # drained, cold
    etch = smoothstep(0.035, 0.11, hp) * smoothstep(0.25, 0.45, L)
    col = lerp(col, lerp(GLOW_DEEP, GLOW, 0.55), etch * 0.35)  # faint Twilight in the tooling
    wear = noise.fbm(Pm / 3.5, 4, seed=seed)
    col = lerp(col, col * 0.35, smoothstep(0.58, 0.75, wear))
    col = lerp(col, MUD, smoothstep(0.6, 0.8, noise.fbm(Pm / 9.0, 3, seed=seed + 1)) * 0.5)
    return col


def paint_hooves(P, col, J):
    x, y, z = np.abs(P[:, 0]), P[:, 1], P[:, 2]
    Pm = np.stack([x, y, z], 1)
    jj = J.astype(int)
    foot = np.isin(jj, (6, 10, 30, 34))
    hoof = foot & (y < 10.5)
    f1, f2 = noise.worley(Pm / np.array([2.5, 6.0, 2.5]), seed=600)
    hcol = lerp(HORN, HORN * 2.2, noise.fbm(Pm * np.array([2.0, 0.3, 2.0]), 3, seed=601))
    hcol = lerp(hcol, CAVITY, (1 - smoothstep(0.0, 0.05, f2 - f1)) * 0.8)
    col = np.where(hoof[:, None], hcol, col)
    mud = smoothstep(40.0, 4.0, y) * smoothstep(0.35, 0.65, noise.fbm(Pm / 3.0, 3, seed=602))
    return lerp(col, MUD, mud * 0.8 * np.isin(jj, list(FRONT_LEG) + list(HIND_LEG)))


# ------------------------------------------------------------------------------------------ atlas
def paint_atlas(r) -> np.ndarray:
    """RGBA body atlas (SIZE x SIZE), alpha 0 where the hide / muscle is torn through."""
    lay = r["layers"]
    parts = [lay["hide"], lay["tack"], lay["bedroll"], lay["muscle"], lay["cavity"]] + r["parts"]
    for p in r["parts"]:
        if p.name.startswith("skull_cap"):
            p.kind = "cap"
    maps = bake(parts)
    mask = maps["mask"]
    P, N = maps["pos"][mask].astype(float), maps["nrm"][mask].astype(float)
    K, V, J = maps["kind"][mask], maps["v"][mask], maps["joint"][mask]
    ao = occupancy_ao(parts, P, N)

    # the original texture drives the hide and tack detail (top-left quadrant holds the original layout)
    orig = np.asarray(Image.open(WORK / "orig" / "hs_body.png").convert("RGB").resize((SIZE // 2, SIZE // 2),
                                                                                         Image.BICUBIC), float) / 255
    blur = np.asarray(Image.fromarray((orig * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)),
                      float) / 255
    big = np.zeros((SIZE, SIZE, 3))
    big[:SIZE // 2, :SIZE // 2] = orig
    bigb = np.zeros((SIZE, SIZE, 3))
    bigb[:SIZE // 2, :SIZE // 2] = blur
    o = big[mask]
    L = luminance(o)
    hp = L - luminance(bigb[mask])
    sat = (o.max(1) - o.min(1)) / np.maximum(o.max(1), 1e-4)

    H, M = wounds(P, N, J)
    col = np.zeros((len(P), 3))
    alpha = np.ones(len(P))

    s = K == KIND["hide"]
    if s.any():
        c, _, _ = paint_hide(P[s], N[s], J[s], L[s], hp[s], sat[s], H[s])
        c = paint_hooves(P[s], c, J[s])
        Js = J[s]

        def field(Q):
            return wounds(Q, N[s], Js)[0]

        run = blood_runs(P[s], field)
        c = lerp(c, BLOOD, run * 0.85)
        col[s] = c
        alpha[s] = (H[s] <= 0).astype(float)
    s = K == KIND["muscle"]
    if s.any():
        col[s] = paint_muscle(P[s], N[s], J[s], M[s], ao[s])
        alpha[s] = (M[s] <= 0).astype(float)
    s = K == KIND["tack"]
    if s.any():
        col[s] = paint_tack(P[s], L[s], hp[s])
    s = K == KIND["skull"]
    if s.any():
        col[s] = paint_skull(P[s], N[s], ao[s])
    s = K == KIND["cap"]
    if s.any():
        col[s] = lerp(CAVITY, BONE_DARK, ao[s] * 0.4)
    s = K == KIND["mouth"]
    if s.any():
        col[s] = lerp(CAVITY, BLOOD, smoothstep(0.45, 0.8, noise.fbm(P[s] / 1.5, 3, seed=700)) * 0.8)
    s = K == KIND["tooth"]
    if s.any():
        g = noise.fbm(P[s] * np.array([2.5, 0.8, 2.5]), 3, seed=710)
        tc = lerp(TOOTH * 0.65, TOOTH, np.clip(0.35 + ao[s] * 0.5 + (g - 0.5) * 0.4, 0, 1))
        tc = lerp(tc, np.array([0.40, 0.28, 0.16]), smoothstep(0.55, 0.0, V[s]) * 0.8)
        col[s] = tc * (0.65 + 0.35 * ao[s])[:, None]
    s = K == KIND["bone"]
    if s.any():
        col[s] = paint_bone(P[s], N[s], ao[s], V[s])
    s = K == KIND["strand"]
    if s.any():
        f = noise.fbm(P[s] * np.array([2.0, 2.0, 2.0]), 3, seed=720)
        sc = ramp(np.clip(f * 1.2 - 0.1, 0, 1), (0.0, MEAT_DARK), (0.6, MEAT * 0.9), (1.0, FASCIA * 0.8))
        col[s] = sc * (0.75 + 0.25 * ao[s])[:, None]

    img = np.zeros((SIZE, SIZE, 4))
    img[mask, :3] = np.clip(col, 0, 1)
    img[mask, 3] = alpha
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    img = img[iy, ix]
    # keep alpha out of the unused gutters opaque so filtering never eats into painted islands
    return img


# ------------------------------------------------------------------------------------------ mane / tail
def paint_hair(name: str, seed: int) -> np.ndarray:
    """Matted, filthy, thinned-out hair with a faint icy sheen at the tips."""
    src = np.asarray(Image.open(WORK / "orig" / f"{name}.png").convert("RGBA").resize(
        (Image.open(WORK / "orig" / f"{name}.png").width * 4, Image.open(WORK / "orig" / f"{name}.png").height * 4),
        Image.BICUBIC), float) / 255
    h, w = src.shape[:2]
    L = luminance(src[..., :3])
    a = src[..., 3]
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    tip = 1 - smoothstep(0.0, 0.6, xx)
    p = np.stack([xx * 3.0, yy * 40.0, np.full_like(xx, seed)], -1)
    strand = noise.fbm(p, 3, seed=seed)
    clump = noise.fbm(np.stack([xx * 6, yy * 9, np.full_like(xx, seed + 1)], -1), 3, seed=seed + 1)
    t = np.clip((L - 0.25) / 0.6 * 0.8 + (strand - 0.5) * 0.55 + (clump - 0.5) * 0.3, 0, 1)
    col = ramp(t.reshape(-1), (0.0, np.array([0.018, 0.016, 0.015])), (0.5, np.array([0.105, 0.092, 0.080])),
               (1.0, np.array([0.300, 0.270, 0.235]))).reshape(h, w, 3)
    mud = smoothstep(0.55, 0.75, clump) * (1 - tip)
    col = col * (1 - mud[..., None] * 0.5) + MUD * mud[..., None] * 0.5
    sheen = tip * smoothstep(0.6, 0.85, strand) * 0.35
    col = col * (1 - sheen[..., None]) + GLOW * 0.55 * sheen[..., None]
    drop = smoothstep(0.48, 0.6, noise.fbm(np.stack([xx * 1.5, yy * 26.0, np.full_like(xx, seed + 3)], -1), 3,
                                           seed=seed + 4)) * (0.25 + 0.75 * tip)
    holes = smoothstep(0.6, 0.66, noise.fbm(np.stack([xx * 12, yy * 12, np.full_like(xx, seed)], -1), 3,
                                            seed=seed + 2)) * (0.3 + 0.7 * tip)
    alpha = np.where(a * (1 - drop) * (1 - holes) > 0.5, 1.0, 0.0)
    rgb = np.clip(col, 0, 1)
    if (alpha > 0.5).any():
        _, (iy, ix) = ndimage.distance_transform_edt(alpha < 0.5, return_indices=True)
        rgb = rgb[iy, ix]
    return np.concatenate([rgb, alpha[..., None]], -1)


def paint_eye() -> np.ndarray:
    """A cold ember deep in the socket, the wight's glowing eye."""
    n = 128
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n
    src = np.asarray(Image.open(WORK / "orig" / "hs_eye.png").convert("RGBA").resize((n, n), Image.BICUBIC),
                     float) / 255
    a = src[..., 3]
    ys, xs = np.nonzero(a > 0.5)
    cy, cx = ys.mean() / n, xs.mean() / n
    ry, rx = (ys.max() - ys.min()) / n / 2, (xs.max() - xs.min()) / n / 2
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2).reshape(-1)
    col = lerp(CAVITY, GLOW_DEEP, smoothstep(1.0, 0.5, r))
    col = lerp(col, GLOW, smoothstep(0.55, 0.05, r))
    col = lerp(col, np.array([0.92, 1.0, 1.0]), smoothstep(0.2, 0.0, r))
    return np.concatenate([col.reshape(n, n, 3), a[..., None]], -1)
