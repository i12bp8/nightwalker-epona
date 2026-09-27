"""Bake UV-space maps (bind-pose position, normal, dominant joint) for a texture of the Epona model."""

from __future__ import annotations

import numpy as np

import render


def bake(meshes, texture: str, size: tuple[int, int], prefer=lambda pos: -pos[:, 0]):
    """Return dict of maps for every texel of `texture`.

    Where several surfaces share texels (mirrored left/right halves), the one with the smallest
    `prefer` value wins; by default the +X (Epona's left) side.
    """
    h, w = size
    zbuf = np.full((h, w), np.inf)
    pos_map = np.zeros((h, w, 3))
    nrm_map = np.zeros((h, w, 3))
    joint_map = np.full((h, w), -1, np.int32)
    for mesh in meshes:
        if mesh.texture != texture or len(mesh.tris) == 0:
            continue
        tris = mesh.tris
        uv = mesh.uv[tris]  # (T, 3, 2)
        shift = np.floor(uv.mean(1))  # bring each triangle near the [0,1) tile
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                tuv = uv - shift[:, None, :] + np.array([dx, dy])
                lo, hi = tuv.min(1), tuv.max(1)
                keep = (hi[:, 0] > 0) & (lo[:, 0] < 1) & (hi[:, 1] > 0) & (lo[:, 1] < 1)
                if not keep.any():
                    continue
                kt = tris[keep]
                flat_idx = np.arange(len(kt) * 3).reshape(-1, 3)
                xy = tuv[keep].reshape(-1, 2) * np.array([w, h])
                vid = kt.reshape(-1)
                z = prefer(mesh.pos[vid])
                attrs = np.concatenate([mesh.pos[vid], mesh.nrm[vid], vid[:, None].astype(float)], 1)
                zbuf, abuf, tid = render.rasterize(xy, z, flat_idx, (h, w), attrs, zbuf)
                cov = tid >= 0
                pos_map[cov] = abuf[cov][:, :3]
                nrm_map[cov] = abuf[cov][:, 3:6]
                # dominant joint: from the first vertex of the covering triangle
                first_vid = kt[tid[cov]][:, 0]
                joint_map[cov] = mesh.joint[first_vid]
    n = np.linalg.norm(nrm_map, axis=2, keepdims=True)
    nrm_map = np.divide(nrm_map, n, out=np.zeros_like(nrm_map), where=n > 0)
    return {"pos": pos_map, "nrm": nrm_map, "joint": joint_map, "mask": joint_map >= 0}
