"""Paint the textures for the skeletal Epona model.

The body atlas is baked back onto the rebuilt geometry (every texel knows its 3D position, normal and what
kind of surface it belongs to), then painted procedurally: aged bone with ambient occlusion from the real
skeleton geometry, a skull with hollow orbits and a grin, horn hooves, dried sinew, and the tack blackened
and etched with Twilight teal. Mane, tail and eyes become ghostly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

import noise
import render
from noise import smoothstep

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
SIZE = 1024

KIND_IDS = {"tack": 1, "skull": 2, "bone": 3, "hoof": 4, "sinew": 5, "cap": 6, "hide": 7, "tooth": 8,
            "mouth": 9}

# ------------------------------------------------------------------------------------------ palette
BONE_LIGHT = np.array([0.845, 0.800, 0.670])
BONE_MID = np.array([0.650, 0.590, 0.470])
BONE_DARK = np.array([0.290, 0.250, 0.195])
GRIME = np.array([0.210, 0.180, 0.140])
TOOTH = np.array([0.900, 0.860, 0.720])
CAVITY = np.array([0.012, 0.014, 0.018])
FLESH = np.array([0.230, 0.110, 0.085])
FLESH_DARK = np.array([0.090, 0.045, 0.038])
SINEW_LIGHT = np.array([0.520, 0.360, 0.280])
HORN_DARK = np.array([0.060, 0.058, 0.060])
HORN_MID = np.array([0.200, 0.185, 0.170])
GLOW = np.array([0.330, 1.000, 0.880])  # Twilight teal
GLOW_DEEP = np.array([0.030, 0.260, 0.280])
LEATHER = np.array([0.070, 0.062, 0.058])


def lerp(a, b, t):
    t = np.asarray(t, float)
    if t.ndim == 1:
        t = t[:, None]
    return a + (b - a) * t


def ramp(t, *stops):
    t = np.clip(t, 0, 1)[..., None]
    out = np.broadcast_to(stops[0][1], t.shape[:-1] + (3,)).astype(float)
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        k = np.clip((t - p0) / (p1 - p0), 0, 1)
        out = np.where(t >= p0, c0 + (c1 - c0) * k, out)
    return out


def luminance(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


# ------------------------------------------------------------------------------------------ baking
def bake_parts(parts, size=SIZE):
    """Rasterize parts in UV space. Returns per-texel pos, nrm, kind, local-v, part index, mask."""
    h = w = size
    zbuf = np.full((h, w), np.inf)
    pos = np.zeros((h, w, 3))
    nrm = np.zeros((h, w, 3))
    extra = np.zeros((h, w, 3))  # kind, local v, part index
    covered = np.zeros((h, w), bool)
    for pi, p in enumerate(parts):
        tris = p.triangles() if hasattr(p, "triangles") else p.tris
        tris = np.asarray(tris, np.int32).reshape(-1, 3)
        if len(tris) == 0:
            continue
        kind = KIND_IDS.get(p.kind, 3)
        local_v = getattr(p, "local_v", None)
        if local_v is None:
            local_v = np.zeros(len(p.pos))
        uv = p.uv[tris]
        shift = np.floor(uv.mean(1))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tuv = uv - shift[:, None, :] + np.array([dx, dy])
                lo, hi = tuv.min(1), tuv.max(1)
                keep = (hi[:, 0] > 0) & (lo[:, 0] < 1) & (hi[:, 1] > 0) & (lo[:, 1] < 1)
                if not keep.any():
                    continue
                kt = tris[keep]
                vid = kt.reshape(-1)
                xy = tuv[keep].reshape(-1, 2) * np.array([w, h])
                z = -np.abs(p.pos[vid, 0]) - 0.001 * pi  # prefer the left (+x) instance of shared charts
                z = np.where(p.pos[vid, 0] >= -1e-3, z, z + 1000)
                attrs = np.concatenate([p.pos[vid], p.nrm[vid], np.full((len(vid), 1), kind),
                                        local_v[vid][:, None], np.full((len(vid), 1), pi)], 1)
                flat = np.arange(len(vid)).reshape(-1, 3)
                zbuf, abuf, tid = render.rasterize(xy, z, flat, (h, w), attrs, zbuf)
                cov = tid >= 0
                pos[cov] = abuf[cov][:, :3]
                nrm[cov] = abuf[cov][:, 3:6]
                extra[cov, 0] = kind
                extra[cov, 1] = abuf[cov][:, 7]
                extra[cov, 2] = pi
                covered |= cov
    n = np.linalg.norm(nrm, axis=2, keepdims=True)
    nrm = np.divide(nrm, n, out=np.zeros_like(nrm), where=n > 0)
    return dict(pos=pos, nrm=nrm, kind=extra[..., 0].round().astype(int), v=extra[..., 1],
                part=extra[..., 2].round().astype(int), mask=covered)


def occupancy_ao(parts, points, normals, cell=1.5):
    """Ambient occlusion from a blurred voxel occupancy grid of the whole model."""
    samples = []
    for p in parts:
        tris = np.asarray(p.triangles() if hasattr(p, "triangles") else p.tris, np.int32).reshape(-1, 3)
        if len(tris) == 0:
            continue
        a, b, c = p.pos[tris[:, 0]], p.pos[tris[:, 1]], p.pos[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        k = np.maximum(1, np.ceil(area / 1.0)).astype(int)
        idx = np.repeat(np.arange(len(tris)), k)
        r1, r2 = np.random.default_rng(0).random((2, len(idx)))
        s = np.sqrt(r1)
        samples.append(a[idx] * (1 - s)[:, None] + b[idx] * (s * (1 - r2))[:, None] + c[idx] * (s * r2)[:, None])
    S = np.vstack(samples)
    lo = S.min(0) - 12
    dims = np.ceil((S.max(0) + 12 - lo) / cell).astype(int) + 1
    grid = np.zeros(dims)
    ijk = np.floor((S - lo) / cell).astype(int)
    np.add.at(grid, tuple(ijk.T), 1.0)
    grid = np.minimum(grid, 3.0) / 3.0
    near = ndimage.gaussian_filter(grid, 2.0 / cell)
    far = ndimage.gaussian_filter(grid, 6.0 / cell)

    def sample(g, q):
        f = (q - lo) / cell
        return ndimage.map_coordinates(g, f.T, order=1, mode="nearest")

    occ = (sample(near, points + normals * 2.5) * 0.9 + sample(far, points + normals * 7.0) * 1.6)
    return np.clip(1.0 - occ * 2.2, 0.0, 1.0)


# ------------------------------------------------------------------------------------------ materials
def paint_bone(P, N, ao, v, seed=0):
    grain = noise.fbm(P * np.array([0.9, 0.35, 0.9]), 3, seed=seed)  # streaks along the long axis
    fine = noise.fbm(P * 2.2, 2, seed=seed + 1)
    stain = noise.fbm(P / 10.0, 4, seed=seed + 2)
    f1, f2 = noise.worley(P / 8.0, seed=seed + 3)
    crack = 1 - smoothstep(0.0, 0.028, f2 - f1)
    crack *= smoothstep(0.56, 0.7, noise.fbm(P / 9.0, 2, seed=seed + 4))
    ends = np.clip(np.maximum(1 - v / 0.12, (v - 0.88) / 0.12), 0, 1)  # epiphyses: rougher, darker
    light = np.clip(0.55 * N[:, 1] + 0.25, -1, 1)  # soft painted top light, TP-style
    t = np.clip(0.15 + ao * 0.72 + light * 0.18 + (grain - 0.5) * 0.35 + (fine - 0.5) * 0.12 - ends * 0.12, 0, 1)
    col = ramp(t, (0.0, BONE_DARK), (0.45, BONE_MID), (1.0, BONE_LIGHT))
    col = lerp(col, col * np.array([0.80, 0.75, 0.66]), smoothstep(0.5, 0.8, stain) * 0.8)
    col = lerp(col, GRIME, (1 - ao) ** 2 * 0.55)
    col = lerp(col, BONE_DARK * 0.5, crack * 0.75)
    # zombie remnants: dried flesh and sinew stuck in the crevices
    rem = smoothstep(0.62, 0.8, noise.fbm(P / 3.5, 3, seed=seed + 5)) * smoothstep(0.75, 0.35, ao)
    strands = smoothstep(0.55, 0.75, noise.fbm(P * np.array([0.35, 1.6, 0.35]), 3, seed=seed + 6))
    col = lerp(col, lerp(FLESH_DARK, FLESH, strands), rem * 0.85)
    col = col * (0.62 + 0.38 * ao)[:, None]
    mud = smoothstep(34.0, 6.0, P[:, 1]) * smoothstep(0.35, 0.7, noise.fbm(P / 3.0, 3, seed=seed + 8))
    col = lerp(col, np.array([0.20, 0.17, 0.13]), mud * 0.75)
    # the odd Twilight-lit fracture
    tw = crack * smoothstep(0.7, 0.8, noise.fbm(P / 25.0, 2, seed=seed + 7))
    col = lerp(col, GLOW, tw * 0.9)
    return col


def paint_skull(P, N, ao, seed=10):
    ax = np.abs(P[:, 0])
    x, y, z = ax, P[:, 1], P[:, 2]
    col = paint_bone(P, N, ao, np.full(len(P), 0.5), seed=seed)
    # orbits: deep hollow sockets glowing faintly from within
    d_eye = np.sqrt((y - 198.0) ** 2 + (z - 191.0) ** 2)
    side = smoothstep(9.0, 13.0, x)
    sock = smoothstep(10.0, 7.5, d_eye) * side
    inner = lerp(CAVITY, GLOW_DEEP, smoothstep(7.0, 3.0, d_eye) * 0.9)
    rim = smoothstep(13.5, 10.5, d_eye) * (1 - sock) * side
    col = lerp(col, col * 0.55, rim * smoothstep(0, 1, (198 - y) / 6 + 0.5))  # shadow under the brow
    col = lerp(col, inner, sock)
    # nasal notch and nostrils
    notch = np.where(x > 4, np.sqrt(((z - 207.0) / 11.0) ** 2 + ((y - 176.0) / 3.2) ** 2), 9.0)
    col = lerp(col, CAVITY, smoothstep(1.05, 0.8, notch))
    nostril = np.sqrt(((x - 6.5) / 3.6) ** 2 + ((y - 171.0) / 5.0) ** 2)
    col = lerp(col, CAVITY, smoothstep(1.1, 0.8, nostril) * (z > 214) * (N[:, 2] > 0.2))
    # sutures and a Twili sigil smouldering on the forehead
    su = np.abs(noise.value(np.stack([z / 12.0, y / 26.0, x / 26.0], 1), seed=seed + 4) - 0.5)
    col = lerp(col, BONE_DARK * 0.7, smoothstep(0.03, 0.0, su) * 0.7 * (y > 180))
    fh = np.stack([P[:, 0], (y - 214.0), (z - 181.0)], 1)
    ring = np.abs(np.sqrt(fh[:, 0] ** 2 + fh[:, 2] ** 2 * 0.8) - 6.0)
    spoke = np.minimum(np.abs(fh[:, 0]), np.abs(fh[:, 2] + 2.0) * 3)
    sigil = (smoothstep(0.9, 0.3, ring) + smoothstep(0.7, 0.2, spoke) * (np.hypot(fh[:, 0], fh[:, 2]) < 9)) \
        * (N[:, 1] > 0.3) * (y > 205)
    col = lerp(col, lerp(GLOW_DEEP, GLOW, 0.85), np.clip(sigil, 0, 1) * 0.9)
    return col


def paint_tooth(P, N, ao, v, seed=20):
    """Enamel, yellowed and stained toward the roots."""
    grain = noise.fbm(P * np.array([2.5, 0.8, 2.5]), 3, seed=seed)
    col = lerp(TOOTH * 0.7, TOOTH, np.clip(0.4 + ao * 0.5 + (grain - 0.5) * 0.4, 0, 1))
    col = lerp(col, np.array([0.52, 0.42, 0.27]), smoothstep(0.55, 0.0, v) * 0.75)  # v = 0 at the root
    return col * (0.7 + 0.3 * ao)[:, None]


def paint_mouth(P, ao, seed=25):
    """Inside of the mouth: dark, with rotten gum."""
    return lerp(CAVITY, FLESH_DARK, smoothstep(0.45, 0.8, noise.fbm(P / 1.5, 3, seed=seed)) * 0.8)


def paint_hoof(P, N, ao, seed=30):
    striae = noise.fbm(np.stack([np.arctan2(P[:, 0] - np.sign(P[:, 0]) * 25, P[:, 2]) * 12, P[:, 1] / 25, P[:, 2] * 0],
                                1), 3, seed=seed)
    col = ramp(np.clip(ao * 0.6 + (striae - 0.5) * 0.6 + P[:, 1] / 40, 0, 1), (0.0, HORN_DARK), (1.0, HORN_MID))
    f1, f2 = noise.worley(P / np.array([2.5, 5.0, 2.5]), seed=seed + 1)
    col = lerp(col, CAVITY, (1 - smoothstep(0.0, 0.05, f2 - f1)) * 0.8)
    col = lerp(col, BONE_MID * 0.7, smoothstep(10.0, 12.0, P[:, 1]) * 0.6)  # coronet band
    return col


def paint_sinew(P, N, ao, seed=40):
    fib = noise.fbm(P * np.array([2.0, 2.0, 0.25]), 3, seed=seed)
    col = ramp(np.clip(fib * 0.9 + ao * 0.4 - 0.2, 0, 1), (0.0, FLESH_DARK), (0.6, FLESH), (1.0, SINEW_LIGHT))
    return lerp(col, CAVITY, (1 - ao) * 0.5)


def paint_cap(P, N, ao):
    """Skull openings: occiput is bone around the foramen magnum, forehead patches are skull, the rest holes."""
    y, z = P[:, 1], P[:, 2]
    back = z < 165
    face = (z >= 185) & (y > 190) & (y < 219)
    out = lerp(CAVITY, BONE_DARK * 0.6, smoothstep(0.2, 0.9, ao) * 0.4)
    if back.any():
        bone = paint_bone(P[back], N[back], ao[back], np.full(back.sum(), 0.5), seed=60)
        foramen = np.hypot(P[back, 0], P[back, 1] - 206.0)
        out[back] = lerp(bone, CAVITY, smoothstep(5.0, 3.5, foramen))
    if face.any():
        out[face] = paint_skull(P[face], N[face], ao[face])
    return out


def paint_hide(P, N, ao, L, hp, edge, seed=70):
    """Desiccated, torn hide: Epona's painted fur shading kept as texture, drained to ash and rot."""
    mottle = noise.fbm(P / 14.0, 4, seed=seed)
    t = np.clip((L - 0.15) / 0.45 * 0.75 + (mottle - 0.5) * 0.5 + hp * 1.4, 0, 1)
    col = ramp(t, (0.0, np.array([0.03, 0.032, 0.036])), (0.55, np.array([0.15, 0.145, 0.14])),
               (1.0, np.array([0.30, 0.285, 0.26])))
    sores = smoothstep(0.6, 0.78, noise.fbm(P / 5.0, 4, seed=seed + 1))
    col = lerp(col, FLESH_DARK, sores * 0.6)
    # raw, dried flesh along every tear, fading into the skin
    rim = smoothstep(0.55, 0.0, edge)
    col = lerp(col, lerp(FLESH_DARK, FLESH, noise.fbm(P / 1.2, 2, seed=seed + 2)), rim * 0.9)
    # Twilight corruption creeping through the skin
    vein = noise.ridged(P / 9.0, 4, seed=seed + 3)
    v = smoothstep(0.86, 0.95, vein) * smoothstep(0.35, 0.6, noise.fbm(P / 25.0, 2, seed=seed + 4))
    col = lerp(col, lerp(GLOW_DEEP, GLOW, v), v * 0.8)
    return col * (0.6 + 0.4 * ao)[:, None]


def paint_tack(P, orig_rgb, hp, L, seed=50):
    Lt = np.clip((L - 0.08) / 0.55, 0, 1)
    col = ramp(np.clip(Lt + hp * 1.5, 0, 1), (0.0, CAVITY), (0.5, LEATHER * 1.45),
               (1.0, np.array([0.35, 0.34, 0.32])))
    etch = smoothstep(0.035, 0.11, hp) * smoothstep(0.25, 0.45, L)
    col = lerp(col, lerp(GLOW_DEEP, GLOW, 0.8), etch * 0.9)
    wear = noise.fbm(P / 4.0, 4, seed=seed)
    col = lerp(col, col * 0.4, smoothstep(0.6, 0.75, wear))
    holes = smoothstep(0.7, 0.74, noise.fbm(P / 5.0, 4, seed=seed + 1))
    return lerp(col, CAVITY, holes)


# ------------------------------------------------------------------------------------------ body atlas
def paint_body_atlas(r) -> np.ndarray:
    parts = [r["tack"]] + r["parts"] + r.get("painted_only", [])
    for p in r["parts"]:
        if p.kind == "skull" and p.name.startswith("skull_cap"):
            p.kind = "cap"
    maps = bake_parts(parts)
    mask = maps["mask"]
    ys, xs = np.nonzero(mask)
    P, N, K, V = maps["pos"][mask], maps["nrm"][mask], maps["kind"][mask], maps["v"][mask]
    ao = occupancy_ao(parts + [r["tack"]], P, N)
    col = np.zeros((len(P), 3))

    orig = np.asarray(Image.open(WORK / "orig" / "hs_body.png").convert("RGB").resize((SIZE, SIZE), Image.BICUBIC),
                      float) / 255.0
    Lfull = luminance(orig)
    Lblur = luminance(np.asarray(Image.fromarray((orig * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)),
                                 float) / 255.0)
    L, hp = Lfull[mask], (Lfull - Lblur)[mask]

    for kind, fn in (("bone", lambda s: paint_bone(P[s], N[s], ao[s], V[s])),
                     ("skull", lambda s: paint_skull(P[s], N[s], ao[s])),
                     ("hoof", lambda s: paint_hoof(P[s], N[s], ao[s])),
                     ("sinew", lambda s: paint_sinew(P[s], N[s], ao[s])),
                     ("tack", lambda s: paint_tack(P[s], None, hp[s], L[s])),
                     ("cap", lambda s: paint_cap(P[s], N[s], ao[s])),
                     ("hide", lambda s: paint_hide(P[s], N[s], ao[s], L[s], hp[s], V[s])),
                     ("tooth", lambda s: paint_tooth(P[s], N[s], ao[s], V[s])),
                     ("mouth", lambda s: paint_mouth(P[s], ao[s]))):
        s = K == KIND_IDS[kind]
        if s.any():
            col[s] = fn(s)
    img = np.zeros((SIZE, SIZE, 3))
    img[mask] = np.clip(col, 0, 1)
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    return img[iy, ix]


# ------------------------------------------------------------------------------------------ cards
def load_rgba(name: str, scale: int = 4) -> np.ndarray:
    img = Image.open(WORK / "orig" / f"{name}.png").convert("RGBA")
    img = img.resize((img.width * scale, img.height * scale), Image.BICUBIC)
    return np.asarray(img, float) / 255.0


def paint_hair(name: str, seed: int) -> np.ndarray:
    """Ghostly mane/tail: ashen black strands, Twilight-lit tips, moth-eaten and ragged."""
    src = load_rgba(name)
    h, w = src.shape[:2]
    L = luminance(src[..., :3])
    a = src[..., 3]
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    tipness = 1 - smoothstep(0.0, 0.6, xx)
    p = np.stack([xx * 3.0, yy * 40.0, np.full_like(xx, seed)], -1)
    strand = noise.fbm(p, 3, seed=seed)
    col = ramp(np.clip((L - 0.25) / 0.6 + (strand - 0.5) * 0.5, 0, 1), (0.0, np.array([0.015, 0.02, 0.026])),
               (0.5, np.array([0.10, 0.145, 0.155])), (1.0, np.array([0.40, 0.53, 0.52])))
    glow_t = np.clip(tipness * smoothstep(0.35, 0.75, strand) * 1.2, 0, 1)
    col = lerp(col.reshape(-1, 3), lerp(GLOW_DEEP, GLOW, glow_t.reshape(-1)), (glow_t * 0.9).reshape(-1)).reshape(h, w, 3)
    drop = smoothstep(0.5, 0.62, noise.fbm(np.stack([xx * 1.5, yy * 26.0, np.full_like(xx, seed + 3)], -1), 3,
                                           seed=seed + 1)) * tipness
    holes = smoothstep(0.6, 0.66, noise.fbm(np.stack([xx * 12, yy * 12, np.full_like(xx, seed)], -1), 3,
                                            seed=seed + 2)) * (0.35 + 0.65 * tipness)
    alpha = np.where(a * (1 - drop) * (1 - holes) > 0.5, 1.0, 0.0)
    rgb = np.clip(col, 0, 1)
    if (alpha > 0.5).any():
        _, (iy, ix) = ndimage.distance_transform_edt(alpha < 0.5, return_indices=True)
        rgb = rgb[iy, ix]
    return np.concatenate([rgb, alpha[..., None]], -1)


def paint_eye() -> np.ndarray:
    n = 128
    yy, xx = (np.mgrid[0:n, 0:n] + 0.5) / n
    a = load_rgba("hs_eye")[..., 3]
    ys, xs = np.nonzero(a > 0.5)
    cy, cx = ys.mean() / n, xs.mean() / n
    ry, rx = (ys.max() - ys.min()) / n / 2, (xs.max() - xs.min()) / n / 2
    r = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    col = lerp(CAVITY, GLOW_DEEP, smoothstep(1.0, 0.45, r).reshape(-1)).reshape(n, n, 3)
    col = lerp(col.reshape(-1, 3), GLOW, smoothstep(0.62, 0.0, r).reshape(-1)).reshape(n, n, 3)
    col = lerp(col.reshape(-1, 3), np.array([0.92, 1.0, 0.97]), smoothstep(0.22, 0.0, r).reshape(-1)).reshape(n, n, 3)
    return np.concatenate([col, a[..., None]], -1)


def transparent(w: int, h: int) -> np.ndarray:
    out = np.zeros((h, w, 4))
    out[..., :3] = CAVITY
    return out


def paint_reins() -> np.ndarray:
    src = load_rgba("tazuna")
    L = luminance(src[..., :3])
    h, w = L.shape
    yy, xx = np.mgrid[0:h, 0:w] / h
    fray = noise.fbm(np.stack([xx * 3, yy * 12, np.zeros_like(xx)], -1), 3, seed=130)
    col = ramp(np.clip(L * 1.4 + (fray - 0.5) * 0.5, 0, 1), (0.0, CAVITY), (0.5, LEATHER * 1.3),
               (1.0, np.array([0.26, 0.25, 0.24])))
    col = lerp(col.reshape(-1, 3), GLOW, (smoothstep(0.78, 0.9, fray) * 0.6).reshape(-1)).reshape(h, w, 3)
    return np.concatenate([col, np.ones((h, w, 1))], -1)
