"""Rebuild a J3D BMD with new geometry and textures, keeping the rig, materials and animation bindings.

Rewritten: INF1 counts, VTX1, DRW1 (optionally with extra rigid joint matrices), SHP1, TEX1 image data.
Copied verbatim: the file header tail, INF1 hierarchy, EVP1, JNT1, MAT3, TEX1 headers' sampler settings and
names. Shape/material/texture counts and order never change, so BCK/BTP animations and the game code that
indexes joints, materials and textures keep working.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from gctex import CMPR, j3d_sections

PRIM_TRIANGLES = 0x90
PRIM_STRIP = 0x98
MAX_PACKET_MTX = 10


@dataclass
class Prim:
    kind: int  # PRIM_TRIANGLES or PRIM_STRIP
    verts: list  # [(drw, pos_idx, nrm_idx, uv_idx), ...]


@dataclass
class OutShape:
    mtx_type: int  # 0 single matrix, 3 multi matrix
    prims: list = field(default_factory=list)


@dataclass
class OutTexture:
    fmt: int
    width: int
    height: int
    data: bytes


class DrwLayout:
    """DRW1 layout with extra rigid (single-joint) matrices inserted after the original rigid block."""

    def __init__(self, orig: bytes, extra_joints: list[int]):
        s = j3d_sections(orig)["DRW1"]
        n, _, fo, io = struct.unpack_from(">HHII", orig, s + 8)
        self.flags = list(orig[s + fo:s + fo + n])
        self.index = list(struct.unpack_from(f">{n}H", orig, s + io))
        self.n_rigid = self.flags.index(1)
        rigid_joints = {self.index[i]: i for i in range(self.n_rigid)}
        self.extra = [j for j in extra_joints if j not in rigid_joints]
        k = len(self.extra)
        self.flags = self.flags[:self.n_rigid] + [0] * k + self.flags[self.n_rigid:]
        self.index = self.index[:self.n_rigid] + self.extra + self.index[self.n_rigid:]
        self.shift = k
        self._rigid = dict(rigid_joints)
        for i, j in enumerate(self.extra):
            self._rigid[j] = self.n_rigid + i

    def remap_old(self, drw: int) -> int:
        return drw if drw < self.n_rigid else drw + self.shift

    def rigid(self, joint: int) -> int:
        return self._rigid[joint]

    def section(self) -> bytes:
        n = len(self.flags)
        body = bytearray(struct.pack(">4sIHHII", b"DRW1", 0, n, 0xFFFF, 0x14, 0))
        body += bytes(self.flags)
        _align(body, 2)
        idx_off = len(body)
        body += struct.pack(f">{n}H", *self.index)
        struct.pack_into(">I", body, 0x10, idx_off)
        return _finish(body)


def _align(buf: bytearray, a: int, fill: int = 0) -> None:
    while len(buf) % a:
        buf.append(fill)


def _finish(body: bytearray) -> bytes:
    _align(body, 32)
    struct.pack_into(">I", body, 4, len(body))
    return bytes(body)


def _packetize(prims: list[Prim]) -> list[tuple[list[int], list[Prim]]]:
    """Greedy grouping of primitives into packets that each use at most 10 matrices."""
    packets: list[tuple[list[int], list[Prim]]] = []
    cur_mtx: list[int] = []
    cur: list[Prim] = []
    for p in prims:
        need = sorted({v[0] for v in p.verts})
        assert len(need) <= MAX_PACKET_MTX, "a single primitive uses more than 10 matrices"
        merged = sorted(set(cur_mtx) | set(need))
        if len(merged) > MAX_PACKET_MTX:
            packets.append((cur_mtx, cur))
            cur_mtx, cur = need, [p]
        else:
            cur_mtx, cur = merged, cur + [p]
    if cur:
        packets.append((cur_mtx, cur))
    return packets


def build_bmd(orig: bytes, shapes: list[OutShape], positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray,
              drw: DrwLayout, textures: list[OutTexture], model_space_pos: np.ndarray,
              overrides: dict[str, bytes] | None = None) -> bytes:
    """Assemble the new BMD.

    positions/normals/uvs are the raw vertex arrays (positions of rigidly bound vertices in joint space).
    model_space_pos gives each position's bind-pose model-space location, for shape bounds.
    textures: one entry per TEX1 header, in TEX1 order (duplicates may reuse the same object).
    """
    sec = j3d_sections(orig)
    order = []
    off = 0x20
    n_sec = struct.unpack_from(">I", orig, 0x0C)[0]
    for _ in range(n_sec):
        tag, size = struct.unpack_from(">4sI", orig, off)
        order.append((tag.decode(), off, size))
        off += size

    shp1, n_packets = _shp1(orig, sec["SHP1"], shapes, model_space_pos)
    out_secs = []
    for tag, o, size in order:
        if tag == "INF1":
            blk = bytearray(orig[o:o + size])
            struct.pack_into(">II", blk, 0x0C, n_packets, len(positions))
            out_secs.append(bytes(blk))
        elif tag == "VTX1":
            out_secs.append(_vtx1(orig, o, positions, normals, uvs))
        elif tag == "DRW1":
            out_secs.append(drw.section())
        elif tag == "SHP1":
            out_secs.append(shp1)
        elif tag == "TEX1":
            out_secs.append(_tex1(orig, o, textures))
        elif overrides and tag in overrides:
            out_secs.append(overrides[tag])
        else:
            out_secs.append(orig[o:o + size])
    body = b"".join(out_secs)
    header = bytearray(orig[:0x20])
    struct.pack_into(">I", header, 8, 0x20 + len(body))
    return bytes(header) + body


def _vtx1(orig: bytes, o: int, positions, normals, uvs) -> bytes:
    fmt_off = struct.unpack_from(">I", orig, o + 8)[0]
    fmt = bytearray()
    p = o + fmt_off
    while True:
        entry = orig[p:p + 16]
        fmt += entry
        p += 16
        if struct.unpack_from(">I", entry, 0)[0] == 0xFF:
            break
    fmts = {}
    for i in range(0, len(fmt), 16):
        attr, cnt, typ, frac = struct.unpack_from(">IIIB", fmt, i)
        fmts[attr] = (cnt, typ, frac)
    assert fmts[9][1] == 4 and fmts[10][1] == 3 and fmts[13][1] == 3, "unexpected vertex formats"

    body = bytearray(struct.pack(">4sII", b"VTX1", 0, 0x40))
    body += b"\0" * (0x40 - len(body))
    body[0x40:] = fmt
    offsets = [0] * 13

    def put(slot, data):
        _align(body, 32)
        offsets[slot] = len(body)
        body.extend(data)

    put(0, np.asarray(positions, ">f4").tobytes())
    nfrac = fmts[10][2]
    put(1, np.clip(np.rint(np.asarray(normals) * (1 << nfrac)), -32768, 32767).astype(">i2").tobytes())
    tfrac = fmts[13][2]
    put(5, np.clip(np.rint(np.asarray(uvs) * (1 << tfrac)), -32768, 32767).astype(">i2").tobytes())
    struct.pack_into(">13I", body, 0x0C, *offsets)
    return _finish(body)


def _shp1(orig: bytes, s: int, shapes: list[OutShape], model_pos: np.ndarray) -> tuple[bytes, int]:
    n = struct.unpack_from(">H", orig, s + 8)[0]
    assert n == len(shapes)
    (_shp_off, _remap_off, _name_off, attr_off, mtx_tbl_off, _prim_off, _mtx_data_off,
     _pkt_off) = struct.unpack_from(">8I", orig, s + 12)
    desc_bytes = orig[s + attr_off:s + mtx_tbl_off]  # both vertex-descriptor lists, verbatim
    DESC_MULTI, DESC_SINGLE = 0, 40

    mtx_table: list[int] = []
    mtx_init: list[tuple[int, int, int]] = []
    draw_init: list[tuple[int, int]] = []
    dl = bytearray()
    shape_hdrs = []
    for sh in shapes:
        packets = _packetize(sh.prims) if sh.mtx_type == 3 else [
            ([sh.prims[0].verts[0][0]], sh.prims)]
        first_mtx, first_pkt = len(mtx_init), len(draw_init)
        used = []
        for mtxs, prims in packets:
            slot = {m: k for k, m in enumerate(mtxs)}
            mtx_init.append((mtxs[0], len(mtxs), len(mtx_table)))
            mtx_table.extend(mtxs)
            start = len(dl)
            for p in prims:
                dl += struct.pack(">BH", p.kind, len(p.verts))
                for d, pi, ni, ti in p.verts:
                    if sh.mtx_type == 3:
                        dl.append(slot[d] * 3)
                    else:
                        assert d == mtxs[0]
                    dl += struct.pack(">HHH", pi, ni, ti)
                    used.append(pi)
            _align(dl, 32)
            draw_init.append((len(dl) - start, start))
        pts = model_pos[np.unique(used)] if used else np.zeros((1, 3))
        radius = float(np.linalg.norm(pts, axis=1).max())
        lo, hi = pts.min(0), pts.max(0)
        shape_hdrs.append(struct.pack(">BBHHHHHf3f3f", sh.mtx_type, 0xFF, len(packets),
                                      DESC_MULTI if sh.mtx_type == 3 else DESC_SINGLE, first_mtx, first_pkt,
                                      0xFFFF, radius, *lo, *hi))

    body = bytearray(struct.pack(">4sIHH8I", b"SHP1", 0, n, 0xFFFF, *([0] * 8)))
    offs = {}
    _align(body, 4)
    offs["shape"] = len(body)
    for h in shape_hdrs:
        body += h
    offs["remap"] = len(body)
    body += struct.pack(f">{n}H", *range(n))
    _align(body, 32)
    offs["desc"] = len(body)
    body += desc_bytes
    _align(body, 4)
    offs["mtx"] = len(body)
    body += struct.pack(f">{len(mtx_table)}H", *mtx_table)
    _align(body, 32)
    offs["dl"] = len(body)
    body += dl
    _align(body, 32)
    offs["mtxinit"] = len(body)
    for m in mtx_init:
        body += struct.pack(">HHI", *m)
    _align(body, 32)
    offs["pkt"] = len(body)
    for d in draw_init:
        body += struct.pack(">II", *d)
    struct.pack_into(">8I", body, 12, offs["shape"], offs["remap"], 0, offs["desc"], offs["mtx"], offs["dl"],
                     offs["mtxinit"], offs["pkt"])
    return _finish(body), len(draw_init)


def _tex1(orig: bytes, o: int, textures: list[OutTexture]) -> bytes:
    count, _pad, hdr_off, str_off = struct.unpack_from(">HHII", orig, o + 8)
    assert count == len(textures)
    str_size = struct.unpack_from(">I", orig, o + 4)[0] - str_off
    strings = orig[o + str_off:o + str_off + str_size]
    headers = [bytearray(orig[o + hdr_off + 32 * i:o + hdr_off + 32 * (i + 1)]) for i in range(count)]

    body = bytearray(struct.pack(">4sIHHII", b"TEX1", 0, count, 0xFFFF, 0x20, 0))
    _align(body, 32)
    hdr_pos = len(body)
    body += b"\0" * (32 * count)
    data_at: dict[int, int] = {}
    for i, t in enumerate(textures):
        key = id(t)
        if key not in data_at:
            _align(body, 32)
            data_at[key] = len(body)
            body += t.data
        h = headers[i]
        h[0] = t.fmt
        struct.pack_into(">HH", h, 2, t.width, t.height)
        if t.fmt == CMPR:
            h[8] = 0  # no palette
            h[9] = 0
            struct.pack_into(">HI", h, 0x0A, 0, 0)
        h[0x10] = 0  # no mipmaps
        h[0x18] = 1
        this_hdr = hdr_pos + 32 * i
        struct.pack_into(">I", h, 0x1C, data_at[key] - this_hdr)
        body[this_hdr:this_hdr + 32] = h
    _align(body, 32)
    struct.pack_into(">I", body, 0x10, len(body))
    body += strings
    return _finish(body)
