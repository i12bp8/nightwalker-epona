"""Minimal GameCube/Wii asset helpers: Yaz0, RARC, J3D TEX1 and GX texture codecs.

Only what the Epona skin pipeline needs. Everything is pure Python + numpy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- Yaz0


def yaz0_decompress(data: bytes) -> bytes:
    if data[:4] != b"Yaz0":
        return data
    size = struct.unpack_from(">I", data, 4)[0]
    out = bytearray(size)
    src, dst = 16, 0
    while dst < size:
        code = data[src]
        src += 1
        for bit in range(7, -1, -1):
            if dst >= size:
                break
            if code & (1 << bit):
                out[dst] = data[src]
                dst += 1
                src += 1
            else:
                b1, b2 = data[src], data[src + 1]
                src += 2
                dist = ((b1 & 0x0F) << 8 | b2) + 1
                n = b1 >> 4
                if n == 0:
                    n = data[src] + 0x12
                    src += 1
                else:
                    n += 2
                s = dst - dist
                for i in range(n):
                    out[dst + i] = out[s + i]
                dst += n
    return bytes(out)


# --------------------------------------------------------------------------- RARC


def rarc_files(data: bytes) -> dict[str, bytes]:
    """Return {path: bytes} for every file in a (possibly Yaz0) RARC archive."""
    data = yaz0_decompress(data)
    assert data[:4] == b"RARC", data[:4]
    data_off = struct.unpack_from(">I", data, 0x0C)[0] + 0x20
    info = 0x20
    num_nodes, nodes_off, _num_entries, entries_off, _strsize, str_off = struct.unpack_from(">6I", data, info)
    nodes_off += 0x20
    entries_off += 0x20
    str_off += 0x20

    def cstr(off: int) -> str:
        end = data.index(b"\0", str_off + off)
        return data[str_off + off:end].decode("ascii")

    nodes = []
    for i in range(num_nodes):
        _typ, name_off, _hash, count, first = struct.unpack_from(">4sIHHI", data, nodes_off + i * 16)
        nodes.append((cstr(name_off), count, first))

    out: dict[str, bytes] = {}

    def walk(node_idx: int, prefix: str) -> None:
        _name, count, first = nodes[node_idx]
        for e in range(first, first + count):
            _fid, _h, type_name, off, size = struct.unpack_from(">HHIII", data, entries_off + e * 20)
            flags = type_name >> 24
            name = cstr(type_name & 0xFFFFFF)
            if name in (".", ".."):
                continue
            if flags & 0x02:
                walk(off, f"{prefix}{name}/")
            else:
                out[f"{prefix}{name}"] = yaz0_decompress(data[data_off + off:data_off + off + size])

    walk(0, "")
    return out


# --------------------------------------------------------------------------- GX formats

I4, I8, IA4, IA8, RGB565, RGB5A3, RGBA8, C4, C8, C14X2, CMPR = 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 14
FMT_NAMES = {I4: "I4", I8: "I8", IA4: "IA4", IA8: "IA8", RGB565: "RGB565", RGB5A3: "RGB5A3", RGBA8: "RGBA8",
             C4: "C4", C8: "C8", C14X2: "C14X2", CMPR: "CMPR"}
# (block_w, block_h, bits per pixel)
BLOCK = {I4: (8, 8, 4), I8: (8, 4, 8), IA4: (8, 4, 8), IA8: (4, 4, 16), RGB565: (4, 4, 16), RGB5A3: (4, 4, 16),
         RGBA8: (4, 4, 32), C4: (8, 8, 4), C8: (8, 4, 8), C14X2: (4, 4, 16), CMPR: (8, 8, 4)}


def level_size(fmt: int, w: int, h: int) -> int:
    bw, bh, bpp = BLOCK[fmt]
    pw = (w + bw - 1) // bw * bw
    ph = (h + bh - 1) // bh * bh
    return pw * ph * bpp // 8


def _rgb565(v):
    v = np.asarray(v, dtype=np.uint32)
    r = (v >> 11) & 0x1F
    g = (v >> 5) & 0x3F
    b = v & 0x1F
    return np.stack([(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)], -1).astype(np.uint8)


def _rgb5a3(v):
    v = np.asarray(v, dtype=np.uint32)
    out = np.zeros(v.shape + (4,), np.uint8)
    opaque = (v & 0x8000) != 0
    r5, g5, b5 = (v >> 10) & 0x1F, (v >> 5) & 0x1F, v & 0x1F
    r4, g4, b4, a3 = (v >> 8) & 0xF, (v >> 4) & 0xF, v & 0xF, (v >> 12) & 0x7
    out[..., 0] = np.where(opaque, (r5 << 3) | (r5 >> 2), r4 * 17)
    out[..., 1] = np.where(opaque, (g5 << 3) | (g5 >> 2), g4 * 17)
    out[..., 2] = np.where(opaque, (b5 << 3) | (b5 >> 2), b4 * 17)
    out[..., 3] = np.where(opaque, 255, (a3 << 5) | (a3 << 2) | (a3 >> 1))
    return out


def _palette(tlut_fmt: int, pal: bytes) -> np.ndarray:
    v = np.frombuffer(pal, ">u2")
    if tlut_fmt == 0:  # IA8
        i = (v & 0xFF).astype(np.uint8)
        a = (v >> 8).astype(np.uint8)
        return np.stack([i, i, i, a], -1)
    if tlut_fmt == 1:  # RGB565
        rgb = _rgb565(v)
        return np.concatenate([rgb, np.full(rgb.shape[:-1] + (1,), 255, np.uint8)], -1)
    return _rgb5a3(v)


def _deblock(blocks: np.ndarray, w: int, h: int, bw: int, bh: int) -> np.ndarray:
    """blocks: (nby, nbx, bh, bw, C) -> (h, w, C) image."""
    nby, nbx = blocks.shape[:2]
    img = blocks.transpose(0, 2, 1, 3, 4).reshape(nby * bh, nbx * bw, -1)
    return img[:h, :w]


def decode(fmt: int, w: int, h: int, data: bytes, tlut_fmt: int = 0, pal: bytes = b"") -> np.ndarray:
    """Decode a GX texture's base level to an RGBA uint8 array (h, w, 4)."""
    bw, bh, bpp = BLOCK[fmt]
    nbx, nby = (w + bw - 1) // bw, (h + bh - 1) // bh
    raw = np.frombuffer(data[:level_size(fmt, w, h)], np.uint8)

    if fmt == CMPR:
        sub = raw.reshape(nby, nbx, 2, 2, 8)  # 8x8 block = 2x2 DXT1 sub-blocks
        c0 = sub[..., 0].astype(np.uint16) << 8 | sub[..., 1]
        c1 = sub[..., 2].astype(np.uint16) << 8 | sub[..., 3]
        p0, p1 = _rgb565(c0).astype(np.int32), _rgb565(c1).astype(np.int32)
        four = (c0 > c1)[..., None]
        p2 = np.where(four, (2 * p0 + p1) // 3, (p0 + p1) // 2)
        p3 = np.where(four, (p0 + 2 * p1) // 3, 0)
        pal4 = np.stack([p0, p1, p2, p3], -2)  # (..., 4, 3)
        alpha = np.stack([np.full(c0.shape, 255), np.full(c0.shape, 255), np.full(c0.shape, 255),
                          np.where(c0 > c1, 255, 0)], -1)
        pal4 = np.concatenate([pal4, alpha[..., None]], -1).astype(np.uint8)  # (..., 4, 4)
        idx_bytes = sub[..., 4:8]  # 4 rows, each byte = 4 px, MSB first
        shifts = np.array([6, 4, 2, 0])
        idx = (idx_bytes[..., :, None] >> shifts) & 3  # (nby,nbx,2,2,4,4)
        px = np.take_along_axis(pal4[..., None, :, :].repeat(4, -3), idx[..., None].astype(np.int64).repeat(4, -1), -2)
        # px: (nby, nbx, sy, sx, 4row, 4col, 4ch) -> blocks (nby, nbx, 8, 8, 4)
        blocks = px.transpose(0, 1, 2, 4, 3, 5, 6).reshape(nby, nbx, 8, 8, 4)
        return _deblock(blocks, w, h, 8, 8)

    if fmt == RGBA8:
        b = raw.reshape(nby, nbx, 2, 16, 2)
        a, r = b[:, :, 0, :, 0], b[:, :, 0, :, 1]
        g, bl = b[:, :, 1, :, 0], b[:, :, 1, :, 1]
        blocks = np.stack([r, g, bl, a], -1).reshape(nby, nbx, 4, 4, 4)
        return _deblock(blocks, w, h, 4, 4)

    if bpp == 16:
        v = raw.view(">u2").reshape(nby, nbx, bh, bw)
        if fmt == IA8:
            i, a = (v & 0xFF).astype(np.uint8), (v >> 8).astype(np.uint8)
            blocks = np.stack([i, i, i, a], -1)
        elif fmt == RGB565:
            rgb = _rgb565(v)
            blocks = np.concatenate([rgb, np.full(rgb.shape[:-1] + (1,), 255, np.uint8)], -1)
        elif fmt == RGB5A3:
            blocks = _rgb5a3(v)
        else:  # C14X2
            blocks = _palette(tlut_fmt, pal)[v & 0x3FFF]
        return _deblock(blocks, w, h, bw, bh)

    if bpp == 8:
        v = raw.reshape(nby, nbx, bh, bw)
        if fmt == I8:
            blocks = np.stack([v, v, v, v], -1)
        elif fmt == IA4:
            i, a = (v & 0xF) * 17, (v >> 4) * 17
            blocks = np.stack([i, i, i, a], -1).astype(np.uint8)
        else:  # C8
            blocks = _palette(tlut_fmt, pal)[v]
        return _deblock(blocks, w, h, bw, bh)

    # 4bpp: I4 / C4
    v = raw.reshape(nby, nbx, bh, bw // 2)
    v = np.stack([v >> 4, v & 0xF], -1).reshape(nby, nbx, bh, bw)
    if fmt == I4:
        i = (v * 17).astype(np.uint8)
        blocks = np.stack([i, i, i, i], -1)
    else:
        blocks = _palette(tlut_fmt, pal)[v]
    return _deblock(blocks, w, h, bw, bh)


# --------------------------------------------------------------------------- BTI / TEX1


@dataclass
class Texture:
    name: str
    fmt: int
    width: int
    height: int
    wrap_s: int
    wrap_t: int
    tlut_fmt: int
    mip_count: int
    data: bytes  # base level only
    pal: bytes

    @property
    def fmt_name(self) -> str:
        return FMT_NAMES.get(self.fmt, str(self.fmt))

    def rgba(self) -> np.ndarray:
        return decode(self.fmt, self.width, self.height, self.data, self.tlut_fmt, self.pal)


def parse_bti_header(buf: bytes, hdr: int, name: str) -> Texture:
    (fmt, _alpha, w, h, ws, wt, _idx, tlut_fmt, pal_n, pal_off, _mip, _edge, _bias, _aniso, _minf, _magf,
     _minlod, _maxlod, mips, _pad, _lodbias, img_off) = struct.unpack_from(">BBHHBBBBHIBBBBBBbbBBhI", buf, hdr)
    base = level_size(fmt, w, h)
    data = buf[hdr + img_off:hdr + img_off + base]
    pal = buf[hdr + pal_off:hdr + pal_off + pal_n * 2] if fmt in (C4, C8, C14X2) else b""
    return Texture(name, fmt, w, h, ws, wt, tlut_fmt, mips, data, pal)


def parse_bti(buf: bytes, name: str) -> Texture:
    return parse_bti_header(buf, 0, name)


def _string_table(buf: bytes, off: int) -> list[str]:
    n = struct.unpack_from(">H", buf, off)[0]
    names = []
    for i in range(n):
        _h, soff = struct.unpack_from(">HH", buf, off + 4 + i * 4)
        end = buf.index(b"\0", off + soff)
        names.append(buf[off + soff:end].decode("shift_jis"))
    return names


def j3d_sections(buf: bytes) -> dict[str, int]:
    n = struct.unpack_from(">I", buf, 0x0C)[0]
    off, out = 0x20, {}
    for _ in range(n):
        tag, size = struct.unpack_from(">4sI", buf, off)
        out[tag.decode()] = off
        off += size
    return out


def bmd_textures(buf: bytes) -> list[Texture]:
    sec = j3d_sections(buf)["TEX1"]
    count, _pad, hdr_off, str_off = struct.unpack_from(">HHII", buf, sec + 8)
    names = _string_table(buf, sec + str_off)
    return [parse_bti_header(buf, sec + hdr_off + i * 32, names[i]) for i in range(count)]


# --------------------------------------------------------------------------- encoders


def _to565(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint32)
    return ((rgb[..., 0] >> 3) << 11) | ((rgb[..., 1] >> 2) << 5) | (rgb[..., 2] >> 3)


def encode_cmpr(rgba: np.ndarray) -> bytes:
    """Encode an (H, W, 4) uint8 image (H, W multiples of 8) as GX CMPR (DXT1 in 8x8 tiles).

    Blocks with any texel alpha < 128 use DXT1's 3-colour + transparent mode.
    """
    h, w = rgba.shape[:2]
    assert h % 8 == 0 and w % 8 == 0
    px = rgba.astype(np.float64)
    # (tile_y, tile_x, sub_y, sub_x, 4, 4, C)
    blocks = px.reshape(h // 8, 2, 4, w // 8, 2, 4, 4).transpose(0, 3, 1, 4, 2, 5, 6)
    rgb = blocks[..., :3].reshape(*blocks.shape[:4], 16, 3)
    alpha = blocks[..., 3].reshape(*blocks.shape[:4], 16)
    transparent = alpha < 128
    has_t = transparent.any(-1)

    # endpoints along the principal axis of the opaque texels
    wts = (~transparent).astype(np.float64)[..., None]
    cnt = np.maximum(wts.sum(-2), 1)
    mean = (rgb * wts).sum(-2) / cnt
    centered = (rgb - mean[..., None, :]) * wts
    cov = np.einsum("...ki,...kj->...ij", centered, centered)
    axis = np.ones(mean.shape) / np.sqrt(3)
    for _ in range(6):  # power iteration
        axis = np.einsum("...ij,...j->...i", cov, axis)
        axis /= np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-9)
    proj = np.einsum("...ki,...i->...k", rgb - mean[..., None, :], axis)
    big = 1e9
    lo = np.where(transparent, big, proj).min(-1)
    hi = np.where(transparent, -big, proj).max(-1)
    lo = np.where(np.isfinite(lo) & (lo < big / 2), lo, 0)
    hi = np.where(np.isfinite(hi) & (hi > -big / 2), hi, 0)
    e_hi = mean + hi[..., None] * axis
    e_lo = mean + lo[..., None] * axis
    c_hi, c_lo = _to565(e_hi), _to565(e_lo)

    # opaque blocks need c0 > c1 (4-colour mode); transparent-capable blocks need c0 <= c1
    c0 = np.where(has_t, np.minimum(c_hi, c_lo), np.maximum(c_hi, c_lo))
    c1 = np.where(has_t, np.maximum(c_hi, c_lo), np.minimum(c_hi, c_lo))
    same = (c0 == c1) & ~has_t
    c0 = np.where(same & (c0 < 0xFFFF), c0 + 0, c0)
    p0 = _rgb565(c0).astype(np.float64)
    p1 = _rgb565(c1).astype(np.float64)
    four = (c0 > c1)[..., None]
    p2 = np.where(four, (2 * p0 + p1) / 3, (p0 + p1) / 2)
    p3 = np.where(four, (p0 + 2 * p1) / 3, np.inf)
    pal = np.stack([p0, p1, p2, p3], -2)  # (..., 4, 3)
    d = ((rgb[..., :, None, :] - pal[..., None, :, :]) ** 2).sum(-1)  # (..., 16, 4)
    d = np.where(np.isnan(d), np.inf, d)
    idx = d.argmin(-1)
    idx = np.where(transparent, 3, idx)
    idx = np.where(same[..., None], 0, idx)

    shifts = np.array([6, 4, 2, 0])
    rows = (idx.reshape(*idx.shape[:-1], 4, 4) << shifts).sum(-1).astype(np.uint8)  # (..., 4)
    out = np.zeros(blocks.shape[:4] + (8,), np.uint8)
    out[..., 0] = c0 >> 8
    out[..., 1] = c0 & 0xFF
    out[..., 2] = c1 >> 8
    out[..., 3] = c1 & 0xFF
    out[..., 4:8] = rows
    return out.tobytes()
