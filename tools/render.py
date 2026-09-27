"""Tiny numpy software rasterizer used for UV baking and turntable previews of Epona."""

from __future__ import annotations

import math

import numpy as np


def look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = r, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def sample(tex: np.ndarray, uv: np.ndarray, wrap_s: int = 1, wrap_t: int = 1) -> np.ndarray:
    """Bilinear sample an (H, W, C) float texture at uv (N, 2) using GX wrap modes (0 clamp, 1 repeat, 2 mirror)."""
    h, w = tex.shape[:2]

    def wrap(c, mode):
        if mode == 1:
            return c - np.floor(c)
        if mode == 2:
            c = np.abs(c) % 2.0
            return np.where(c > 1.0, 2.0 - c, c)
        return np.clip(c, 0.0, 1.0)

    s = wrap(uv[:, 0], wrap_s) * w - 0.5
    t = wrap(uv[:, 1], wrap_t) * h - 0.5
    x0, y0 = np.floor(s).astype(int), np.floor(t).astype(int)
    fx, fy = (s - x0)[:, None], (t - y0)[:, None]
    x0m, x1m = x0 % w, (x0 + 1) % w
    y0m, y1m = y0 % h, (y0 + 1) % h
    return ((tex[y0m, x0m] * (1 - fx) + tex[y0m, x1m] * fx) * (1 - fy)
            + (tex[y1m, x0m] * (1 - fx) + tex[y1m, x1m] * fx) * fy)


def rasterize(xy: np.ndarray, z: np.ndarray, tris: np.ndarray, size: tuple[int, int], attrs: np.ndarray,
              zbuf: np.ndarray | None = None, persp_w: np.ndarray | None = None, cull_backfaces: bool = False,
              alpha_test=None):
    """Rasterize triangles.

    xy: (N, 2) pixel coords, z: (N,) depth (smaller = closer), attrs: (N, K) per-vertex attributes.
    Returns (zbuf, attr_buf (H, W, K), tri_id (H, W)); pixels not covered have tri_id == -1.
    alpha_test: optional (texture, wrap_s, wrap_t); attrs[:, :2] are then UVs and fragments whose texel
    alpha is < 0.5 are discarded before the depth write, like GX alpha compare.
    """
    h, w = size
    k = attrs.shape[1]
    if zbuf is None:
        zbuf = np.full((h, w), np.inf)
    abuf = np.zeros((h, w, k))
    tid = np.full((h, w), -1, np.int32)
    inv_w = np.ones(len(xy)) if persp_w is None else 1.0 / persp_w
    for ti, (a, b, c) in enumerate(tris):
        p = xy[[a, b, c]]
        area = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])
        if abs(area) < 1e-12 or (cull_backfaces and area < 0):
            continue
        x0, x1 = max(int(math.floor(p[:, 0].min())), 0), min(int(math.ceil(p[:, 0].max())), w - 1)
        y0, y1 = max(int(math.floor(p[:, 1].min())), 0), min(int(math.ceil(p[:, 1].max())), h - 1)
        if x0 > x1 or y0 > y1:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((p[1, 0] - gx) * (p[2, 1] - gy) - (p[2, 0] - gx) * (p[1, 1] - gy)) / area
        w1 = ((p[2, 0] - gx) * (p[0, 1] - gy) - (p[0, 0] - gx) * (p[2, 1] - gy)) / area
        w2 = 1.0 - w0 - w1
        eps = -1e-6
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        zz = w0 * z[a] + w1 * z[b] + w2 * z[c]
        region = zbuf[y0:y1 + 1, x0:x1 + 1]
        m = inside & (zz < region)
        if not m.any():
            continue
        iw = w0 * inv_w[a] + w1 * inv_w[b] + w2 * inv_w[c]
        pa = (w0[..., None] * attrs[a] * inv_w[a] + w1[..., None] * attrs[b] * inv_w[b]
              + w2[..., None] * attrs[c] * inv_w[c]) / iw[..., None]
        if alpha_test is not None:
            tex, ws, wt = alpha_test
            ok = np.zeros_like(m)
            ok[m] = sample(tex[..., 3:4], pa[m][:, :2], ws, wt)[:, 0] >= 0.5
            m &= ok
            if not m.any():
                continue
        region[m] = zz[m]
        abuf[y0:y1 + 1, x0:x1 + 1][m] = pa[m]
        tid[y0:y1 + 1, x0:x1 + 1][m] = ti
    return zbuf, abuf, tid


class Camera:
    def __init__(self, eye, target, fov_deg=30.0, size=(512, 512)):
        self.view = look_at(eye, target)
        self.size = size
        self.f = 1.0 / math.tan(math.radians(fov_deg) / 2)

    def project(self, pos: np.ndarray):
        h, w = self.size
        pv = pos @ self.view[:3, :3].T + self.view[:3, 3]
        depth = -pv[:, 2]
        sx = (pv[:, 0] * self.f / depth) * (h / 2) + w / 2
        sy = -(pv[:, 1] * self.f / depth) * (h / 2) + h / 2
        return np.stack([sx, sy], 1), depth


CULL_BACK = ("head_m", "eye_m", "SC_eye_m", "z_nimotu_m")  # hs.bmd materials with GX_CULL_BACK


def render(meshes, textures: dict, cam: Camera, light_dir=(0.4, 0.8, 0.5), ambient=0.42, bg=(0.06, 0.07, 0.09),
           tint=(1.0, 1.0, 1.0), skip=(), cull=CULL_BACK):
    """Render meshes with {texture_name: (rgba_float, wrap_s, wrap_t)}; alpha-tested like the game."""
    h, w = cam.size
    zbuf = np.full((h, w), np.inf)
    color = np.zeros((h, w, 3))
    color[:] = bg
    L = np.asarray(light_dir, float)
    L /= np.linalg.norm(L)
    # draw opaque first, alpha-tested cards after so they layer correctly via z-buffer anyway
    for mesh in sorted(meshes, key=lambda m: m.material in ("hair_m", "nuki_m", "SC_eye_m")):
        if mesh.material in skip or mesh.texture not in textures:
            continue
        tex, ws, wt = textures[mesh.texture]
        xy, depth = cam.project(mesh.pos)
        attrs = np.concatenate([mesh.uv, mesh.nrm], 1)
        # rasterize into a scratch z so alpha-tested texels can be rejected
        z_try, abuf, tid = rasterize(xy, depth, mesh.tris, (h, w), attrs, zbuf, persp_w=depth,
                                     cull_backfaces=mesh.material in cull,
                                     alpha_test=(tex, ws, wt) if tex.shape[-1] == 4 else None)
        cov = tid >= 0
        if not cov.any():
            continue
        uv = abuf[cov][:, :2]
        n = abuf[cov][:, 2:5]
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        texel = sample(tex, uv, ws, wt)
        keep = texel[:, 3] >= 0.5
        nl = n @ L
        lam = np.maximum(nl, 0.0) if mesh.material in cull else np.abs(nl)  # cards are lit on both sides
        shade = ambient + (1 - ambient) * lam
        rgb = texel[:, :3] * shade[:, None] * np.asarray(tint)
        ys, xs = np.nonzero(cov)
        ys, xs, rgb = ys[keep], xs[keep], rgb[keep]
        zbuf[ys, xs] = z_try[ys, xs]
        color[ys, xs] = rgb
    return np.clip(color, 0, 1)
