"""Pose a J3D model with a BCK animation (ANK1) and skin its vertices, for in-game-like previews and checks.

Joint transforms follow J3D: local = T * Rz * Ry * Rx * S, world = parent_world * local. Rigidly bound
vertices are stored in joint space; envelope vertices in bind-pose model space (skinned through the
inverse bind matrices).
"""

from __future__ import annotations

import copy
import math
import struct

import numpy as np

import bmd


class BCK:
    def __init__(self, data: bytes):
        s = 0x20
        (_tag, _size, self.loop, angmul, self.length, self.n_joints, sc, rc, tc, jo, so, ro, to) = \
            struct.unpack_from(">4sIBBHHHHHIIII", data, s)
        self.scale = np.frombuffer(data[s + so:s + so + sc * 4], ">f4").astype(float)
        self.rot = np.frombuffer(data[s + ro:s + ro + rc * 2], ">i2").astype(float) * (2 ** angmul)
        self.trans = np.frombuffer(data[s + to:s + to + tc * 4], ">f4").astype(float)
        self.tracks = []
        for j in range(self.n_joints):
            e = struct.unpack_from(">27H", data, s + jo + j * 54)
            # per axis x, y, z: (scale, rotation, translation) each as (count, index, tangent mode)
            self.tracks.append([[e[a * 9 + c * 3:a * 9 + c * 3 + 3] for c in range(3)] for a in range(3)])

    @staticmethod
    def _eval(arr, count, index, tan, t):
        if count == 1:
            return arr[index]
        stride = 3 if tan == 0 else 4
        keys = arr[index:index + count * stride].reshape(count, stride)
        times, vals = keys[:, 0], keys[:, 1]
        t_in = keys[:, 2]
        t_out = keys[:, 3] if stride == 4 else keys[:, 2]
        if t <= times[0]:
            return vals[0]
        if t >= times[-1]:
            return vals[-1]
        i = int(np.searchsorted(times, t) - 1)
        t0, t1 = times[i], times[i + 1]
        u = (t - t0) / (t1 - t0)
        h1, h2 = 2 * u ** 3 - 3 * u ** 2 + 1, -2 * u ** 3 + 3 * u ** 2
        h3, h4 = u ** 3 - 2 * u ** 2 + u, u ** 3 - u ** 2
        return h1 * vals[i] + h2 * vals[i + 1] + (h3 * t_out[i] + h4 * t_in[i + 1]) * (t1 - t0)

    def joint_srt(self, j, t):
        out = []
        for comp, arr in ((0, self.scale), (1, self.rot), (2, self.trans)):
            out.append([self._eval(arr, *self.tracks[j][axis][comp], t) for axis in range(3)])
        return out  # [scale xyz], [rot xyz (s16 units)], [trans xyz]


def local_matrix(scale, rot, trans):
    ax, ay, az = (r * math.pi / 32768.0 for r in rot)
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rz @ ry @ rx @ np.diag(scale)
    m[:3, 3] = trans
    return m


def pose(model: bmd.BMD, anim: BCK | None, t: float) -> list[np.ndarray]:
    """World matrix of every joint at frame t (bind pose if anim is None)."""
    if anim is None:
        return [m.copy() for m in model.joint_world]
    world = [None] * len(model.joint_local)
    order = sorted(range(len(world)), key=lambda j: _depth(model, j))
    for j in order:
        s, r, tr = anim.joint_srt(j, t)
        loc = local_matrix(s, r, tr)
        p = model.joint_parent[j]
        world[j] = loc if p is None else world[p] @ loc
    return world


def _depth(model, j):
    d = 0
    while model.joint_parent[j] is not None:
        j = model.joint_parent[j]
        d += 1
    return d


def draw_matrices(model: bmd.BMD, world) -> dict:
    """Matrix per DRW1 index: joint world for rigid entries, weighted skin matrix for envelopes."""
    inv_bind = [np.linalg.inv(m) for m in model.joint_world]
    out = {}
    n_draw = len(model.drw_weighted) - len(model.envelopes)
    for d in range(n_draw):
        if model.drw_weighted[d]:
            env = model.envelopes[model.drw_index[d]]
            out[d] = sum(w * (world[j] @ inv_bind[j]) for j, w in env)
        else:
            out[d] = world[model.drw_index[d]]
    return out


def posed_meshes(model: bmd.BMD, meshes, world):
    """Copies of the model's meshes with positions/normals skinned to the given joint matrices."""
    mats = draw_matrices(model, world)
    P, N = model.arrays[9], model.arrays[10]
    out = []
    for m in meshes:
        raw = m.raw
        pos = np.zeros((len(raw), 3))
        nrm = np.zeros((len(raw), 3))
        for d in np.unique(raw[:, 3]):
            sel = raw[:, 3] == d
            M = mats[int(d)]
            pos[sel] = P[raw[sel, 0]] @ M[:3, :3].T + M[:3, 3]
            nrm[sel] = N[raw[sel, 1]] @ M[:3, :3].T
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        c = copy.copy(m)
        c.pos, c.nrm = pos, nrm
        out.append(c)
    return out
