"""Build the Nightwalker Epona Dusklight mod.

    python tools/extract.py <Horse.arc or extracted disc root>   # once: reference model/textures -> work/
    python tools/build.py [--install]

Produces mod/ (unpacked bundle), dist/<id>.dusk and previews/. --install copies the .dusk into Dusklight's
user mods folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import xxhash
from PIL import Image

import bmd
import bmdwrite
import build_model
import gctex
import anim
import paint_skeleton as paint
import paint_wight
import render

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
MOD = ROOT / "mod"
DIST = ROOT / "dist"
PREVIEWS = ROOT / "previews"
MOD_ID = "com.i12bp8.nightwalker_epona"
OVERLAY_BMD = MOD / "overlay" / "res" / "Object" / "Horse" / "archive" / "bmdr" / "hs.bmd"
TEXDIR = MOD / "textures" / "nightwalker_epona"
REIN_KEY = "tex1_16x16_94934198a2bc6db8_14"  # tazuna.bti stays on disc; replaced by hash

MOD_JSON = {
    "id": MOD_ID,
    "name": "Nightwalker Epona",
    "version": "2.0.0",
    "author": "i12bp8",
    "description": "Epona, back from the dead. A new model built on her original rig, so all her "
                   "animations still work.",
    "icon": "res/icon.png",
    "banner": "res/banner.png",
}


def to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def rgba(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 3:
        img = np.concatenate([img, np.ones(img.shape[:2] + (1,))], -1)
    return img


def embed(img_hd: np.ndarray, w: int, h: int) -> tuple[np.ndarray, bytes]:
    """Downsample an HD texture to its in-model size and CMPR-encode it (alpha kept 1-bit)."""
    im = Image.fromarray(to_u8(rgba(img_hd)), "RGBA")
    small = np.asarray(im.resize((w, h), Image.LANCZOS)).copy()
    a = np.asarray(im.getchannel("A").resize((w, h), Image.BOX))
    small[..., 3] = np.where(a >= 128, 255, 0)
    return small, gctex.encode_cmpr(small)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="copy the .dusk into Dusklight's mods folder")
    ap.add_argument("--no-previews", action="store_true")
    args = ap.parse_args()
    if not (WORK / "hs.bmd").exists():
        sys.exit("work/hs.bmd missing: run tools/extract.py first")
    t0 = time.time()

    print("[1/5] building model")
    r = build_model.build()
    print(f"      bone charts packed at {r['density']:.2f} px/unit")

    print("[2/5] painting textures")
    body = paint_wight.paint_atlas(r)
    hair = paint_wight.paint_hair("hs_hair", seed=200)
    tail = paint_wight.paint_hair("hs_tail", seed=210)
    tail2 = paint_wight.paint_hair("hs_tail#2", seed=220)
    eye = paint_wight.paint_eye()
    lid = paint.transparent(128, 128)
    reins = paint.paint_reins()

    print("[3/5] writing model + texture replacements")
    orig = r["orig"]
    tex1 = gctex.bmd_textures(orig)
    hd_by_name = {"hs_body": body, "hs_eye": eye, "hs_eye.1": lid, "hs_eye.2": lid, "hs_eye.3": lid,
                  "hs_hair": hair, "hs_tail": tail}
    out_tex, cache, embedded_preview = [], {}, {}
    seen_tail = 0
    for i, t in enumerate(tex1):
        name = t.name
        hd = hd_by_name[name]
        if name == "hs_tail":
            hd = tail if seen_tail == 0 else tail2
            seen_tail += 1
        key = (name, id(hd))
        if key not in cache:
            w_, h_ = (512, 512) if name == "hs_body" else (t.width, t.height)  # body atlas grew to 2x2
            small, data = embed(hd, w_, h_)
            cache[key] = (bmdwrite.OutTexture(gctex.CMPR, w_, h_, data), hd, small)
        out_tex.append(cache[key][0])
    if TEXDIR.exists():
        shutil.rmtree(TEXDIR)
    TEXDIR.mkdir(parents=True)
    written = set()
    for (name, _), (ot, hd, small) in cache.items():
        rep = f"tex1_{ot.width}x{ot.height}_{xxhash.xxh64(ot.data).hexdigest()}_{gctex.CMPR}"
        if rep in written:
            continue
        written.add(rep)
        Image.fromarray(to_u8(rgba(hd)), "RGBA").save(TEXDIR / f"{rep}.png", optimize=True)
        print(f"      {name:9s} -> {rep}.png ({hd.shape[1]}x{hd.shape[0]})")
        embedded_preview[name] = small
    Image.fromarray(to_u8(rgba(reins)), "RGBA").save(TEXDIR / f"{REIN_KEY}.png", optimize=True)

    data = bmdwrite.build_bmd(orig, r["shapes"], r["positions"], r["normals"], r["uvs"], r["drw"], out_tex,
                              r["model_pos"], overrides={"MAT3": r["mat3"][1]})
    OVERLAY_BMD.parent.mkdir(parents=True, exist_ok=True)
    OVERLAY_BMD.write_bytes(data)
    problems = validate_bmd(data)
    if problems:
        sys.exit("hs.bmd failed validation:\n  " + "\n  ".join(problems[:20]))
    check = bmd.BMD(data)  # parse our own output back as a sanity check
    meshes = check.meshes()
    n_tris = sum(len(m.tris) for m in meshes)
    print(f"      hs.bmd: {len(data) / 1024:.0f} KiB, {n_tris} triangles, {len(r['positions'])} positions")

    print("[4/5] previews, icon, banner")
    res = MOD / "res"
    res.mkdir(parents=True, exist_ok=True)
    tex = {
        "hs_body": (body, 1, 1),
        "hs_eye": (eye, 1, 1),
        "hs_eye.1": (lid, 1, 1),
        "hs_hair": (hair, 2, 1),
        "hs_tail": (tail, 1, 1),
    }
    tex = {k: (rgba(v[0]), v[1], v[2]) for k, v in tex.items()}
    bg = (0.035, 0.04, 0.05)
    if not args.no_previews:
        views = [((720, 175, -20), (0, 120, -20), 40), ((-330, 250, 420), (0, 135, 20), 40),
                 ((160, 225, 330), (6, 192, 182), 27), ((380, 360, -420), (0, 125, -30), 40)]
        ims = [render.render(meshes, tex, render.Camera(e, t, f, (640, 640)), bg=bg) for e, t, f in views]
        grid = np.concatenate([np.concatenate(ims[:2], 1), np.concatenate(ims[2:], 1)], 0)
        PREVIEWS.mkdir(exist_ok=True)
        Image.fromarray(to_u8(grid)).save(PREVIEWS / "turnaround.png")
        Image.fromarray(to_u8(posed_sheet(check, meshes, tex, bg))).save(PREVIEWS / "animated.png")
    banner = render.render(meshes, tex, render.Camera((620, 190, 60), (0, 128, -18), 30, (400, 1400)), bg=bg)
    Image.fromarray(to_u8(_twilight_bg(banner, bg))).save(res / "banner.png")
    icon = render.render(meshes, tex, render.Camera((-250, 250, 330), (4, 185, 150), 36, (512, 512)), bg=bg)
    Image.fromarray(to_u8(_twilight_bg(icon, bg))).save(res / "icon.png")
    (MOD / "mod.json").write_text(json.dumps(MOD_JSON, indent=2) + "\n")

    print("[5/5] packaging")
    DIST.mkdir(exist_ok=True)
    dusk = DIST / f"{MOD_ID}.dusk"
    with zipfile.ZipFile(dusk, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(MOD.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(MOD).as_posix())
    print(f"      {dusk} ({dusk.stat().st_size / 1024:.0f} KiB)")
    if args.install:
        dest = Path.home() / ".local/share/TwilitRealm/Dusklight/mods"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dusk, dest / dusk.name)
        print(f"      installed to {dest / dusk.name}")
    print(f"done in {time.time() - t0:.0f}s")


POSES = [("hs_wait_01", 0), ("hs_run_fast", 6), ("hs_excitement", 30), ("hs_jump_middle", 4), ("hs_turn_left", 8)]


def posed_sheet(model, meshes, tex, bg, size=420):
    """The model in real animation poses (idle, gallop, rearing, jump, turn) from two sides."""
    rows = []
    for eye in ((-640, 210, 60), (560, 330, -330)):
        ims = []
        for name, frame in POSES:
            a = anim.BCK((WORK / "bck" / f"{name}.bck").read_bytes())
            posed = anim.posed_meshes(model, meshes, anim.pose(model, a, frame))
            ims.append(render.render(posed, tex, render.Camera(eye, (0, 130, 0), 42, (size, size)), bg=bg))
        rows.append(np.concatenate(ims, 1))
    return np.concatenate(rows, 0)


def validate_bmd(data: bytes) -> list[str]:
    """Check the rebuilt file against the rules J3DModelLoader / aurora rely on. Returns problems found."""
    import struct
    errs = []
    if len(data) != struct.unpack_from(">I", data, 8)[0]:
        errs.append("header file size mismatch")
    sec = gctex.j3d_sections(data)
    for tag, off in sec.items():
        if off % 32 or struct.unpack_from(">I", data, off + 4)[0] % 32:
            errs.append(f"{tag} not 32-byte aligned")
    # draw matrices: rigid block first, then weighted; SHP1 may only use the first (count - envelopes)
    s = sec["DRW1"]
    n_drw, _, fo, io = struct.unpack_from(">HHII", data, s + 8)
    flags = data[s + fo:s + fo + n_drw]
    n_env = struct.unpack_from(">H", data, sec["EVP1"] + 8)[0]
    entry_num = n_drw - n_env
    first_w = flags.index(1)
    if any(flags[:first_w]) or not all(flags[first_w:entry_num]):
        errs.append("DRW1 rigid/weighted blocks out of order")
    # vertex arrays
    m = bmd.BMD(data)
    n_pos, n_nrm, n_uv = (len(m.arrays[a]) for a in (9, 10, 13))
    inf_packets, inf_vtx = struct.unpack_from(">II", data, sec["INF1"] + 0x0C)
    if inf_vtx > n_pos:
        errs.append("INF1 vertex count exceeds position array")
    s = sec["SHP1"]
    n_shp = struct.unpack_from(">H", data, s + 8)[0]
    shp_off, _, _, _, tbl_off, dl_off, mtx_off, pkt_off = struct.unpack_from(">8I", data, s + 12)
    total_packets = 0
    for i in range(n_shp):
        mt, _, npk, desc, fm, fp = struct.unpack_from(">BBHHHH", data, s + shp_off + i * 0x28)
        total_packets += npk
        multi = desc == 0
        for p in range(npk):
            use, cnt, first = struct.unpack_from(">HHI", data, s + mtx_off + (fm + p) * 8)
            tbl = struct.unpack_from(f">{cnt}H", data, s + tbl_off + first * 2)
            if cnt > 10 or any(entry_num <= t != 0xFFFF for t in tbl):
                errs.append(f"shape {i} packet {p}: bad matrix table {tbl}")
            size, off = struct.unpack_from(">II", data, s + pkt_off + (fp + p) * 8)
            if off % 32 or size % 32:
                errs.append(f"shape {i} packet {p}: display list not aligned")
            dl = data[s + dl_off + off:s + dl_off + off + size]
            k = 0
            stride = 7 if multi else 6
            while k < len(dl) and dl[k] != 0:
                op, cnt_v = dl[k], struct.unpack_from(">H", dl, k + 1)[0]
                if op not in (0x90, 0x98, 0xA0) or (op == 0x90 and cnt_v % 3) or cnt_v < 3:
                    errs.append(f"shape {i} packet {p}: bad primitive {op:#x} x{cnt_v}")
                    break
                for v in range(cnt_v):
                    base = k + 3 + v * stride
                    if multi and dl[base] // 3 >= cnt:
                        errs.append(f"shape {i} packet {p}: matrix slot out of range")
                        break
                    pi, ni, ti = struct.unpack_from(">HHH", dl, base + (1 if multi else 0))
                    if pi >= n_pos or ni >= n_nrm or ti >= n_uv:
                        errs.append(f"shape {i} packet {p}: vertex index out of range")
                        break
                k += 3 + cnt_v * stride
    if total_packets != inf_packets:
        errs.append(f"INF1 packet count {inf_packets} != {total_packets}")
    for t in gctex.bmd_textures(data):
        if len(t.data) != gctex.level_size(t.fmt, t.width, t.height):
            errs.append(f"texture {t.name} truncated")
    return errs


def _twilight_bg(img: np.ndarray, bg) -> np.ndarray:
    """Replace the flat background with a dusky Twilight gradient (orange horizon into teal-black)."""
    h, w = img.shape[:2]
    is_bg = np.all(np.abs(img - np.asarray(bg)) < 1e-6, axis=-1)
    y = np.linspace(0, 1, h)[:, None, None]
    x = np.linspace(-1, 1, w)[None, :, None]
    sky = (np.array([0.02, 0.05, 0.06]) * (1 - y) + np.array([0.20, 0.09, 0.03]) * y ** 2.2)
    sky = sky + np.array([0.08, 0.20, 0.19]) * np.exp(-((x * 1.2) ** 2 + ((y - 0.45) * 3) ** 2)) * 0.6
    sky = np.broadcast_to(sky, img.shape)
    return np.where(is_bg[..., None], sky, img)


if __name__ == "__main__":
    main()
