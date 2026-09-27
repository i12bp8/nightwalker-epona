"""Pull Epona's model and textures out of Horse.arc into work/.

Usage: python tools/extract.py <path to res/Object/Horse.arc | extracted disc root>

Writes work/hs.bmd, work/orig/<name>.png and work/keys.json (Dusklight replacement keys).
Nothing from here is shipped in the mod; it's only the reference the painter works from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xxhash
from PIL import Image

import gctex

try:  # much faster than the pure-Python fallback
    import oead

    gctex.yaz0_decompress = lambda d: bytes(oead.yaz0.decompress(d)) if d[:4] == b"Yaz0" else d
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"


def replacement_key(t: gctex.Texture) -> str:
    key = f"tex1_{t.width}x{t.height}_{xxhash.xxh64(t.data).hexdigest()}"
    if t.fmt in (gctex.C4, gctex.C8, gctex.C14X2):
        raw = np.frombuffer(t.data, np.uint8)
        if t.fmt == gctex.C4:
            raw = np.concatenate([raw >> 4, raw & 0xF])
        elif t.fmt == gctex.C14X2:
            raw = np.frombuffer(t.data, ">u2") & 0x3FFF
        lo, hi = int(raw.min()), int(raw.max())
        key += "_" + xxhash.xxh64(t.pal[lo * 2:(hi + 1) * 2]).hexdigest()
    return f"{key}_{t.fmt}"


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if src is None:
        sys.exit(__doc__)
    if src.is_dir():
        src = next(p for p in (src / "files/res/Object/Horse.arc", src / "res/Object/Horse.arc") if p.exists())
    files = gctex.rarc_files(src.read_bytes())
    (WORK / "orig").mkdir(parents=True, exist_ok=True)
    bmd_bytes = files["bmdr/hs.bmd"]
    (WORK / "hs.bmd").write_bytes(bmd_bytes)
    (WORK / "bck").mkdir(exist_ok=True)
    for name, data in files.items():  # animations, for posed previews
        if name.endswith(".bck"):
            (WORK / "bck" / Path(name).name).write_bytes(data)

    textures = gctex.bmd_textures(bmd_bytes) + [gctex.parse_bti(files["tex/tazuna.bti"], "tazuna")]
    keys: dict[str, dict] = {}
    for t in textures:
        key = replacement_key(t)
        # TEX1 repeats a few names; two distinct hs_tail images exist, so suffix by key when needed
        name = t.name if t.name not in keys or keys[t.name]["key"] == key else f"{t.name}#2"
        if name in keys:
            continue
        keys[name] = {"key": key, "width": t.width, "height": t.height, "format": t.fmt_name,
                      "wrap": [t.wrap_s, t.wrap_t]}
        Image.fromarray(t.rgba(), "RGBA").save(WORK / "orig" / f"{name}.png")
        print(f"{name:10s} {key}")
    (WORK / "keys.json").write_text(json.dumps(keys, indent=2))


if __name__ == "__main__":
    main()
