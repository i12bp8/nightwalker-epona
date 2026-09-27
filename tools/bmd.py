"""J3D BMD geometry reader: bind-pose triangles with UVs, normals and per-shape texture.

Enough of the format to map each texel of Epona's textures back onto the 3D model.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from gctex import _string_table, j3d_sections

GX_VA_PNMTXIDX, GX_VA_POS, GX_VA_NRM, GX_VA_CLR0, GX_VA_CLR1, GX_VA_TEX0 = 0, 9, 10, 11, 12, 13
COMP_FMT = {0: ">u1", 1: ">i1", 2: ">u2", 3: ">i2", 4: ">f4"}


@dataclass
class Mesh:
    shape: int
    material: str
    texture: str  # TEX1 name of the material's first texture
    pos: np.ndarray  # (N, 3) bind-pose model space
    nrm: np.ndarray  # (N, 3)
    uv: np.ndarray  # (N, 2)
    joint: np.ndarray  # (N,) dominant joint index
    tris: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.int32))
    raw: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.int32))  # (N, 4) pos/nrm/uv index, DRW1 index


def _joint_local(sx, sy, sz, rx, ry, rz, tx, ty, tz) -> np.ndarray:
    ax, ay, az = (r * math.pi / 32768.0 for r in (rx, ry, rz))
    cx, sx_, cy, sy_, cz, sz_ = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rxm = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    rym = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    rzm = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    m = np.eye(4)
    m[:3, :3] = rzm @ rym @ rxm @ np.diag([sx, sy, sz])
    m[:3, 3] = (tx, ty, tz)
    return m


class BMD:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.sec = j3d_sections(buf)
        self._read_joints()
        self._read_drw_evp()
        self._read_vtx()
        self._read_materials()
        self._read_hierarchy()

    # ----------------------------------------------------------------- joints
    def _read_joints(self):
        b, s = self.buf, self.sec["JNT1"]
        n, _, ent, remap, names = struct.unpack_from(">HHIII", b, s + 8)
        self.joint_names = _string_table(b, s + names)
        remap_t = struct.unpack_from(f">{n}H", b, s + remap)
        self.joint_local = []
        for i in range(n):
            o = s + ent + remap_t[i] * 0x40
            sx, sy, sz = struct.unpack_from(">3f", b, o + 4)
            rx, ry, rz = struct.unpack_from(">3h", b, o + 0x10)
            tx, ty, tz = struct.unpack_from(">3f", b, o + 0x18)
            self.joint_local.append(_joint_local(sx, sy, sz, rx, ry, rz, tx, ty, tz))

    def _read_drw_evp(self):
        b = self.buf
        s = self.sec["DRW1"]
        n, _, wo, do = struct.unpack_from(">HHII", b, s + 8)
        self.drw_weighted = list(b[s + wo:s + wo + n])
        self.drw_index = struct.unpack_from(f">{n}H", b, s + do)
        s = self.sec["EVP1"]
        n, _, co, io, wo, mo = struct.unpack_from(">HHIIII", b, s + 8)
        counts = list(b[s + co:s + co + n])
        self.envelopes = []
        k = 0
        for c in counts:
            idx = struct.unpack_from(f">{c}H", b, s + io + k * 2)
            w = struct.unpack_from(f">{c}f", b, s + wo + k * 4)
            self.envelopes.append(list(zip(idx, w)))
            k += c

    # ----------------------------------------------------------------- vertices
    def _read_vtx(self):
        b, s = self.buf, self.sec["VTX1"]
        size = struct.unpack_from(">I", b, s + 4)[0]
        fmt_off = struct.unpack_from(">I", b, s + 8)[0]
        offs = struct.unpack_from(">13I", b, s + 12)
        order = [GX_VA_POS, GX_VA_NRM, None, GX_VA_CLR0, GX_VA_CLR1] + [GX_VA_TEX0 + i for i in range(8)]
        valid = sorted([o for o in offs if o] + [size])
        self.arrays = {}
        o = s + fmt_off
        while True:
            attr, cnt, typ, frac = struct.unpack_from(">IIIB", b, o)
            o += 16
            if attr == 0xFF:
                break
            if attr not in order:
                continue
            start = offs[order.index(attr)]
            end = valid[valid.index(start) + 1]
            if attr in (GX_VA_CLR0, GX_VA_CLR1):
                continue
            ncomp = {GX_VA_POS: 3 if cnt == 1 else 2, GX_VA_NRM: 3}.get(attr, 2 if cnt == 1 else 1)
            dt = np.dtype(COMP_FMT[typ])
            raw = np.frombuffer(b[s + start:s + end], dt)
            raw = raw[:len(raw) // ncomp * ncomp].reshape(-1, ncomp).astype(np.float64)
            if typ != 4:
                raw /= float(1 << frac)
            self.arrays[attr] = raw

    # ----------------------------------------------------------------- materials
    def _read_materials(self):
        b, s = self.buf, self.sec["MAT3"]
        n = struct.unpack_from(">H", b, s + 8)[0]
        offs = struct.unpack_from(">30I", b, s + 12)
        self.mat_names = _string_table(b, s + offs[2])
        remap = struct.unpack_from(f">{n}H", b, s + offs[1])
        tex_names = [t for t in self._tex_names()]
        self.mat_texture = []
        for i in range(n):
            init = s + offs[0] + remap[i] * 0x14C
            tex_no = struct.unpack_from(">H", b, init + 0x84)[0]
            if tex_no == 0xFFFF:
                self.mat_texture.append("")
            else:
                tex1_idx = struct.unpack_from(">H", b, s + offs[15] + tex_no * 2)[0]
                self.mat_texture.append(tex_names[tex1_idx])

    def _tex_names(self):
        s = self.sec["TEX1"]
        str_off = struct.unpack_from(">I", self.buf, s + 0x10)[0]
        return _string_table(self.buf, s + str_off)

    # ----------------------------------------------------------------- scene graph
    def _read_hierarchy(self):
        b, s = self.buf, self.sec["INF1"]
        o = s + struct.unpack_from(">I", b, s + 0x14)[0]
        n_j = len(self.joint_local)
        self.joint_world = [None] * n_j
        self.shape_material = {}
        stack, last_joint, last_mat, cur_parent = [], None, None, None
        parent = [None] * n_j
        while True:
            typ, idx = struct.unpack_from(">HH", b, o)
            o += 4
            if typ == 0:
                break
            if typ == 1:
                stack.append((cur_parent, last_mat))
                cur_parent = last_joint
            elif typ == 2:
                cur_parent, last_mat = stack.pop()
                last_joint = cur_parent
            elif typ == 0x10:
                parent[idx] = cur_parent
                last_joint = idx
            elif typ == 0x11:
                last_mat = idx
            elif typ == 0x12:
                self.shape_material[idx] = last_mat
        for j in range(n_j):
            m, p, chain = np.eye(4), j, []
            while p is not None:
                chain.append(p)
                p = parent[p]
            for c in reversed(chain):
                m = m @ self.joint_local[c]
            self.joint_world[j] = m
        self.joint_parent = parent

    # ----------------------------------------------------------------- shapes
    def meshes(self) -> list[Mesh]:
        b, s = self.buf, self.sec["SHP1"]
        n = struct.unpack_from(">H", b, s + 8)[0]
        (shp_off, remap_off, _name_off, attr_off, mtx_tbl_off, prim_off, mtx_data_off,
         pkt_off) = struct.unpack_from(">8I", b, s + 12)
        remap = struct.unpack_from(f">{n}H", b, s + remap_off)
        pos_a, nrm_a, uv_a = self.arrays[GX_VA_POS], self.arrays.get(GX_VA_NRM), self.arrays[GX_VA_TEX0]
        out = []
        for si in range(n):
            e = s + shp_off + remap[si] * 0x28
            _mtype, npkt, a_off, first_mtx, first_pkt = struct.unpack_from(">BxHHHH", b, e)
            # vertex descriptor
            desc, o = [], s + attr_off + a_off
            while True:
                attr, typ = struct.unpack_from(">II", b, o)
                o += 8
                if attr == 0xFF:
                    break
                desc.append((attr, typ))
            mat = self.shape_material.get(si)
            mat_idx = mat if mat is not None else 0
            pos, nrm, uv, jnt, tris, raw = [], [], [], [], [], []
            slots = [0] * 10
            for p in range(npkt):
                psize, poff = struct.unpack_from(">II", b, s + pkt_off + (first_pkt + p) * 8)
                use_idx, mcount, mfirst = struct.unpack_from(">HHI", b, s + mtx_data_off + (first_mtx + p) * 8)
                first_mtx_use = use_idx
                for k in range(mcount):
                    v = struct.unpack_from(">H", b, s + mtx_tbl_off + (mfirst + k) * 2)[0]
                    if v != 0xFFFF:
                        slots[k] = v
                o, end = s + prim_off + poff, s + prim_off + poff + psize
                while o < end:
                    ptype = b[o]
                    if ptype == 0:
                        break
                    cnt = struct.unpack_from(">H", b, o + 1)[0]
                    o += 3
                    verts = []
                    for _ in range(cnt):
                        vals = {}
                        for attr, typ in desc:
                            if typ == 1:  # direct (matrix indices)
                                vals[attr] = b[o]
                                o += 1
                            elif typ == 2:
                                vals[attr] = b[o]
                                o += 1
                            elif typ == 3:
                                vals[attr] = struct.unpack_from(">H", b, o)[0]
                                o += 2
                        slot = vals.get(GX_VA_PNMTXIDX, 0) // 3
                        drw = slots[slot] if GX_VA_PNMTXIDX in vals else first_mtx_use

                        p_ = pos_a[vals[GX_VA_POS]]
                        n_ = nrm_a[vals[GX_VA_NRM]] if GX_VA_NRM in vals and nrm_a is not None else np.zeros(3)
                        if self.drw_weighted[drw]:
                            env = self.envelopes[self.drw_index[drw]]
                            dom = max(env, key=lambda x: x[1])[0]
                            wp, wn = p_, n_  # envelope vertices are stored in model space
                        else:
                            dom = self.drw_index[drw]
                            m = self.joint_world[dom]
                            wp = m[:3, :3] @ p_ + m[:3, 3]
                            wn = m[:3, :3] @ n_
                        pos.append(wp)
                        nrm.append(wn)
                        uv.append(uv_a[vals[GX_VA_TEX0]] if GX_VA_TEX0 in vals else (0.0, 0.0))
                        jnt.append(dom)
                        raw.append((vals[GX_VA_POS], vals.get(GX_VA_NRM, 0), vals.get(GX_VA_TEX0, 0), drw))
                        verts.append(len(pos) - 1)
                    if ptype == 0x98:  # triangle strip
                        for i in range(cnt - 2):
                            t = (verts[i], verts[i + 1], verts[i + 2])
                            tris.append(t if i % 2 == 0 else (t[1], t[0], t[2]))
                    elif ptype == 0xA0:  # fan
                        for i in range(1, cnt - 1):
                            tris.append((verts[0], verts[i], verts[i + 1]))
                    elif ptype == 0x90:
                        for i in range(0, cnt, 3):
                            tris.append(tuple(verts[i:i + 3]))
            nrm_arr = np.array(nrm, dtype=np.float64).reshape(-1, 3)
            ln = np.linalg.norm(nrm_arr, axis=1, keepdims=True)
            nrm_arr = np.divide(nrm_arr, ln, out=np.zeros_like(nrm_arr), where=ln > 0)
            out.append(Mesh(si, self.mat_names[mat_idx], self.mat_texture[mat_idx],
                            np.array(pos, dtype=np.float64).reshape(-1, 3), nrm_arr,
                            np.array(uv, dtype=np.float64).reshape(-1, 2), np.array(jnt, np.int32),
                            np.array(tris, np.int32).reshape(-1, 3), np.array(raw, np.int32).reshape(-1, 4)))
        return out
