"""Procedural horse skeleton fitted to Epona's rig (bind pose of hs.bmd).

Every part names the joint it is rigidly bound to (or a function picking a blended matrix), so the game's
own BCK animations drive the bones. Left-side bones are built once and mirrored to the right.
Joint indices (JNT1 order): 0 center, 1 backbone2, 2 backbone1, 3-6 F_L_leg1-4, 7-10 F_R_leg1-4, 11 neck1,
12 neck2, 13 hair_L, 14 hair_R, 15 head, 16/17 ears, 18 hair_F, 19/20 mouth, 21 kura1 (saddle), 22-25 belts
and stirrups, 26 waist, 27-30 B_L_leg1-4, 31-34 B_R_leg1-4, 35-37 tail1-3.
"""

from __future__ import annotations

import numpy as np

from geom import Part, ellipsoid, loft, merge, mirror, remap_uv, smooth_profile

BACKBONE2, BACKBONE1, NECK1, NECK2, HEAD, WAIST = 1, 2, 11, 12, 15, 26
F_L = (3, 4, 5, 6)
B_L = (27, 28, 29, 30)
TAIL1, TAIL2, TAIL3 = 35, 36, 37
L_TO_R = {3: 7, 4: 8, 5: 9, 6: 10, 27: 31, 28: 32, 29: 33, 30: 34}


def catmull(points, n: int) -> np.ndarray:
    """Centripetal-ish Catmull-Rom through points, n samples."""
    p = np.asarray(points, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    seg = len(p) - 3
    out = []
    for s in np.linspace(0, seg, n, endpoint=True):
        i = min(int(s), seg - 1)
        t = s - i
        p0, p1, p2, p3 = p[i], p[i + 1], p[i + 2], p[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)


def resample_arc(points, n: int) -> np.ndarray:
    p = np.asarray(points, float)
    d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s = np.linspace(0, d[-1], n)
    return np.stack([np.interp(s, d, p[:, k]) for k in range(p.shape[1])], 1)


def seg(a, b, n):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.linspace(0, 1, n)[:, None] * (b - a)


def long_bone(name, a, b, r_mid, r_a, r_b, n=8, sides=8, flat=1.0, knob=0.2, up=(0, 0, 1), chart=None,
              bind=None) -> Part:
    t = np.linspace(0, 1, n)
    r = smooth_profile(t, r_mid, r_a, r_b, knob)
    p = loft(name, seg(a, b, n), r * flat, r, sides=sides, up_hint=up)
    p.chart = chart or name
    p.bind = bind
    return p


def knob(name, center, radii, axes=None, chart=None, bind=None, sides=7, rings=4) -> Part:
    p = ellipsoid(name, center, radii, axes=axes, sides=sides, rings=rings)
    p.chart = chart or name
    p.bind = bind
    return p


def paired(parts_left: list[Part]) -> list[Part]:
    out = []
    for p in parts_left:
        out.append(p)
        m = mirror(p)
        if isinstance(p.bind, int):
            m.bind = L_TO_R.get(p.bind, p.bind)
        out.append(m)
    return out


# ------------------------------------------------------------------------------------------ spine
NECK_LINE = [(0, 207, 153), (0, 205, 146), (0, 200, 138), (0, 193, 124), (0, 186, 110), (0, 180, 96),
             (0, 175, 83), (0, 173, 70), (0, 176, 60)]
BACK_LINE = [(0, 176, 60), (0, 179, 50), (0, 183, 35), (0, 186, 20), (0, 188, 0), (0, 190, -20),
             (0, 191, -40), (0, 192, -55), (0, 193, -70), (0, 192, -85), (0, 190, -96)]
TAIL_LINE = [(0, 190, -96), (0, 188.5, -110), (0, 186.5, -128), (0, 185, -148), (0, 184, -172)]


def _line_at(line, z):
    """Point on a (x, y, z) polyline at the given z (lines run with decreasing z)."""
    L = np.asarray(line, float)
    zs = L[:, 2][::-1]
    return np.array([0.0, np.interp(z, zs, L[:, 1][::-1]), z])


def spine_bind(z):
    if z > 118:
        return NECK2
    if z > 62:
        return NECK1
    if z > 7:
        return BACKBONE1
    if z > -60:
        return BACKBONE2
    if z > -98:
        return WAIST
    if z > -151:
        return TAIL1
    return TAIL2


def vertebra(name, z0, z1, line, r_body, spine_h=0.0, spine_back=0.0, spine_len=None, wings=0.0,
             wing_y=0.0, chart=None, sides=6) -> list[Part]:
    """One vertebra between z0 (front) and z1 (back): body + optional dorsal spine + transverse wings."""
    parts = []
    a, b = _line_at(line, z0), _line_at(line, z1)
    n = 4
    t = np.linspace(0, 1, n)
    r = smooth_profile(t, r_body * 0.82, r_body * 1.08, r_body * 1.12, 0.22)
    body = loft(name + "_body", seg(a, b, n), r, r * 1.05, sides=sides, up_hint=(0, 1, 0))
    body.chart = (chart or name) + "_body"
    parts.append(body)
    mid = (a + b) / 2
    bind = spine_bind(mid[2])
    if spine_h > 0:
        base = mid + [0, r_body * 0.7, 0]
        tip = mid + [0, r_body + spine_h, -spine_back]
        L = spine_len or (abs(z0 - z1) * 0.55)
        m = 4
        tt = np.linspace(0, 1, m)
        sp = loft(name + "_spine", seg(base, tip, m), 0.8 + 0.5 * tt ** 3, L * (1 - 0.3 * tt) / 2, sides=6,
                  up_hint=(1, 0, 0), cap_start=False)
        sp.chart = (chart or name) + "_spine"
        parts.append(sp)
    if wings > 0:
        for sgn in (1, -1):
            root = mid + [sgn * r_body * 0.6, wing_y, 0]
            tip = mid + [sgn * (r_body + wings), wing_y - wings * 0.12, 1.0]
            w = loft(name + ("_wingL" if sgn > 0 else "_wingR"), seg(root, tip, 3),
                     [0.9, 0.75, 0.6], [abs(z0 - z1) * 0.3, abs(z0 - z1) * 0.26, abs(z0 - z1) * 0.2], sides=5,
                     up_hint=(0, 1, 0), cap_start=False)
            w.chart = (chart or name) + "_wing"
            parts.append(w)
    for p in parts:
        p.bind = bind
    return parts


def build_spine() -> list[Part]:
    parts = []
    # cervical C1..C7 (C1 atlas: short with big wings; C2 long with a crest)
    cz = [(152, 143), (142, 125), (124, 110), (110, 97), (97, 85), (85, 74), (74, 64)]
    for i, (z0, z1) in enumerate(cz):
        name = f"C{i + 1}"
        if i == 0:
            parts += vertebra(name, z0, z1, NECK_LINE, 5.0, wings=10.0, wing_y=0.5, chart="atlas")
        elif i == 1:
            parts += vertebra(name, z0, z1, NECK_LINE, 5.2, spine_h=5.0, spine_back=0.0,
                              spine_len=abs(z0 - z1) * 0.8, chart="axis")
        else:
            parts += vertebra(name, z0, z1, NECK_LINE, 5.4 - 0.1 * i, spine_h=2.2, wings=4.5 + 0.6 * i,
                              wing_y=-1.5, chart="cerv")
    # thoracic T1..T16: spines tallest at the withers, sweeping back
    tz = np.linspace(62, -26, 17)
    tips = {0: 26, 1: 31, 2: 34, 3: 34, 4: 32, 5: 28, 6: 24, 7: 21, 8: 19, 9: 18, 10: 17, 11: 16.5, 12: 16,
            13: 16, 14: 16, 15: 16}
    for i in range(16):
        z0, z1 = tz[i] - 0.4, tz[i + 1] + 0.4
        yb = _line_at(BACK_LINE, (z0 + z1) / 2)[1]
        h = tips[i] - 4.2
        parts += vertebra(f"T{i + 1}", z0, z1, BACK_LINE, 4.2, spine_h=h, spine_back=6.0 if i < 8 else 3.0,
                          spine_len=4.6, chart="thor")
    # lumbar L1..L6 with long transverse wings
    lz = np.linspace(-26, -58, 7)
    for i in range(6):
        parts += vertebra(f"L{i + 1}", lz[i] - 0.4, lz[i + 1] + 0.4, BACK_LINE, 4.4, spine_h=12.0,
                          spine_back=-1.0, spine_len=4.6, wings=9.0 + 1.5 * min(i, 3), wing_y=0.0,
                          chart="lumb")
    # sacrum: fused block with a row of low spines
    sac = long_bone("sacrum", _line_at(BACK_LINE, -58), _line_at(BACK_LINE, -95), 5.2, 7.5, 3.5, n=7,
                    sides=10, flat=0.75, up=(0, 1, 0), bind=WAIST)
    parts.append(sac)
    for k, z in enumerate(np.linspace(-62, -90, 5)):
        base = _line_at(BACK_LINE, z) + [0, 3.5, 0]
        tip = base + [0, 12.5 - 1.8 * k, -2.5]
        sp = loft(f"S{k}_spine", seg(base, tip, 3), [0.8, 0.95, 1.1], [2.2, 1.9, 1.6], sides=6,
                  up_hint=(1, 0, 0), cap_start=False)
        sp.chart = "sacral_spine"
        sp.bind = WAIST
        parts.append(sp)
    # tail vertebrae, shrinking
    tzs = np.linspace(-96, -172, 11)
    for i in range(10):
        r = 3.4 - 1.8 * i / 9
        zz0, zz1 = tzs[i] - 0.6, tzs[i + 1] + 0.6
        ps = vertebra(f"Cd{i + 1}", zz0, zz1, TAIL_LINE, r, spine_h=2.5 if i < 3 else 0.0, spine_len=2.5,
                      wings=2.5 if i < 2 else 0.0, chart="tail", sides=6)
        parts += ps
    return parts


# ------------------------------------------------------------------------------------------ ribcage
def barrel(z):
    """Ribcage cross-section at z: (top y at the spine, bottom y, half width, centre y)."""
    # calibrated ~3 units inside Epona's own body shell (see build_model.fit_inside)
    top = np.interp(z, [-40, -20, 0, 20, 40, 60], [191, 190, 188, 186, 183, 177])
    bot = np.interp(z, [-40, -30, -20, -10, 0, 10, 20, 30, 50, 60, 70, 90],
                    [126.3, 121.1, 116.8, 115.4, 113.8, 112.5, 111.0, 110.0, 110.0, 110.6, 111.7, 118])
    half = np.interp(z, [-40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80],
                     [34.2, 35.2, 35.2, 34.1, 33.2, 32.6, 30.8, 29.7, 29.7, 29.2, 28.5, 25.5, 22])
    return top, bot, half


RIB_ENVELOPE = None  # optional callable (z, theta_deg) -> radius of the body cavity from (0, cy(z), z)
RIB_CENTER = None  # optional callable z -> cy


def rib_curve(i, n_ribs):
    """Centre line of rib i (0 = first rib behind the shoulder)."""
    f = i / (n_ribs - 1)
    z_top = 57 - 76 * f
    sternal = i < 8
    slope = np.interp(f, [0, 0.35, 1], [10, -8, -22])  # bottom end relative to top (+ = forward)
    theta_end = np.interp(f, [0, 0.5, 1], [172, 168, 128]) if sternal else np.interp(f, [0.5, 1], [150, 118])
    top_y = _line_at(BACK_LINE, z_top)[1]
    pts = []
    thetas = np.linspace(8, theta_end, 22)
    for k, th in enumerate(thetas):
        u = (th - 8) / (theta_end - 8)
        z = z_top + slope * u ** 1.3
        r = np.radians(th)
        s, c = np.sin(r), np.cos(r)
        if RIB_ENVELOPE is not None:  # follow the measured body cavity, a little inside it
            cy = RIB_CENTER(z)
            rad = RIB_ENVELOPE(z, th) - 2.3
            pts.append((s * rad, cy + c * rad, z))
            continue
        top, bot, half = barrel(z)
        cy = bot + (top - bot) * 0.46
        ry_up, ry_dn = top + 9 - cy, cy - bot
        ry = ry_up if c > 0 else ry_dn
        x = half * (np.abs(s) ** 0.85)
        y = cy + ry * np.sign(c) * (np.abs(c) ** 1.15)
        pts.append((x, y, z))
    pts = np.array(pts)
    # start at the vertebra: blend the first samples from the costovertebral joint outwards
    head = np.array([3.5, top_y + 1.5, z_top])
    w = np.clip(np.linspace(0, 1, 22) / 0.18, 0, 1)[:, None]
    w = w * w * (3 - 2 * w)
    pts = head * (1 - w) + pts * w
    if sternal:  # costal cartilage sweeps forward to the sternum
        end = np.array([4.5, sternum_y(z_top + slope + 6) + 3.5, z_top + slope + 6])
        pts = np.vstack([pts, (pts[-1] + end) / 2 + [1.5, -1.0, 0], end])
    return resample_arc(catmull(pts, 60), 11)


def sternum_y(z):
    return np.interp(z, [10, 30, 50, 70, 90, 105], [115, 112.5, 112.5, 114.5, 117, 124])


def build_ribcage() -> list[Part]:
    parts = []
    n_ribs = 13
    for i in range(n_ribs):
        c = rib_curve(i, n_ribs)
        m = len(c)
        t = np.linspace(0, 1, m)
        width = (2.3 + 0.9 * np.sin(np.pi * np.clip(t * 1.2, 0, 1))) * (1.0 if i > 1 else 1.25)
        thick = 1.05 + 0.25 * (1 - t)
        # cross-section: thin across the barrel surface normal, wide along the body (z)
        hints = []
        for k in range(m):
            top, bot, half = barrel(c[k, 2])
            cy = bot + (top - bot) * 0.46
            v = np.array([c[k, 0] / max(half, 1), (c[k, 1] - cy) / 40.0, 0])
            hints.append(v / max(np.linalg.norm(v), 1e-6))
        rib = loft(f"rib{i + 1}_L", c, thick, width, sides=6, normal_hints=np.array(hints))
        rib.chart = f"rib{i // 3}"
        z_top = c[0, 2]
        rib.bind = BACKBONE1 if z_top > 7 else BACKBONE2
        parts.append(rib)
    parts = paired(parts)
    # sternum keel + manubrium
    zs = np.linspace(8, 104, 12)
    cen = np.stack([np.zeros_like(zs), sternum_y(zs), zs], 1)
    st = loft("sternum", cen, np.interp(zs, [8, 40, 80, 104], [2.2, 3.6, 3.2, 2.6]),
              np.interp(zs, [8, 40, 80, 104], [2.6, 3.6, 3.0, 2.4]), sides=10, up_hint=(0, 1, 0))
    st.chart = "sternum"
    st.bind = BACKBONE1
    parts.append(st)
    return parts


# ------------------------------------------------------------------------------------------ limbs
def build_front_leg() -> list[Part]:
    L1, L2, L3, L4 = F_L
    parts = []
    # scapula: broad plate from the withers down to the shoulder joint, with a spine ridge
    top = np.array([30.0, 205.0, 60.0])
    neck = np.array([27.5, 158.0, 92.0])
    c = catmull([top, (top + neck) / 2 + [2.5, 0, 0], neck], 9)
    t = np.linspace(0, 1, 9)
    width = np.interp(t, [0, 0.15, 0.7, 1], [15.0, 14.0, 6.0, 5.2])
    sc = loft("scapula_L", c, 1.3 + 0.9 * t, width, sides=10, up_hint=(1, 0, 0.1), power=2.4)
    sc.chart = "scapula"
    sc.bind = BACKBONE1
    parts.append(sc)
    ridge = loft("scap_spine_L", catmull([top + [2.2, -6, -2], (top + neck) / 2 + [5.0, 2, -1],
                                          neck + [2.2, 8, -3]], 7), [0.9, 1.6, 2.2, 2.4, 2.0, 1.4, 0.9],
                 [0.8, 1.0, 1.1, 1.1, 1.0, 0.9, 0.8], sides=8, up_hint=(1, 0, 0))
    ridge.chart = "scapula_spine"
    ridge.bind = BACKBONE1
    parts.append(ridge)
    parts.append(knob("glenoid_L", neck + [0, -3.5, 2.5], (5.4, 5.0, 5.8), chart="joint_knob", bind=BACKBONE1))
    # humerus: shoulder -> elbow
    parts.append(long_bone("humerus_L", (25.5, 150.0, 100.0), (24.0, 116.5, 79.5), 4.2, 7.2, 6.0, n=10,
                           chart="humerus", bind=L1))
    parts.append(knob("tubercle_L", (27.0, 151.5, 104.5), (4.0, 4.6, 4.2), chart="joint_knob", bind=L1))
    # radius + ulna/olecranon
    parts.append(long_bone("radius_L", (23.5, 114.0, 79.5), (23.5, 74.5, 78.5), 3.4, 5.3, 4.6, n=10,
                           chart="radius", bind=L2))
    parts.append(long_bone("olecranon_L", (23.5, 111.5, 74.5), (23.5, 125.5, 67.5), 2.4, 3.2, 3.4, n=6,
                           flat=0.7, chart="olecranon", bind=L2))
    # carpus (knee) + accessory carpal
    parts.append(knob("carpus_L", (23.5, 71.0, 78.8), (5.4, 5.6, 5.2), chart="joint_knob", bind=L3))
    parts.append(knob("acc_carpal_L", (23.5, 72.5, 72.0), (1.8, 3.2, 2.6), chart="joint_knob", bind=L3))
    # cannon + splints
    parts.append(long_bone("cannon_L", (23.5, 67.0, 78.5), (23.5, 22.5, 78.2), 2.9, 4.0, 4.3, n=10,
                           flat=0.85, chart="cannon", bind=L3))
    # fetlock (sesamoids), pastern bones, hoof
    parts.append(knob("sesamoid_L", (23.5, 21.0, 74.0), (3.6, 2.8, 2.4), chart="joint_knob", bind=L4))
    parts.append(long_bone("pastern_L", (23.5, 20.5, 78.8), (23.5, 10.5, 89.0), 2.8, 4.0, 3.4, n=7,
                           chart="pastern", bind=L4))
    parts.append(hoof("hoof_L", (23.5, 0.0, 95.5), (23.5, 12.0, 90.0), (8.6, 10.0), (6.2, 6.6), L4))
    return paired(parts)


def build_hind_leg() -> list[Part]:
    L1, L2, L3, L4 = B_L
    parts = []
    # --- pelvis (bound to the waist) ---
    tc = np.array([33.5, 193.0, -38.0])  # point of hip (just under the hide)
    ts = np.array([8.0, 205.0, -50.0])  # tuber sacrale
    ace = np.array([28.5, 175.5, -78.4])  # hip socket (joint B_L_leg1)
    ti = np.array([16.0, 176.0, -117.0])  # point of buttock
    wing_mid = (tc + ts) / 2
    c = catmull([wing_mid, wing_mid * 0.45 + ace * 0.55 + [3, 2, 0], ace + [1.0, 2.0, 5.0]], 9)
    t = np.linspace(0, 1, 9)
    across = tc - ts
    across /= np.linalg.norm(across)
    ilium = loft("ilium_L", c, 1.6 + 1.2 * t, np.interp(t, [0, 0.25, 0.7, 1], [17.0, 13.0, 5.5, 5.5]), sides=10,
                 normal_hints=np.repeat(np.cross(across, c[-1] - c[0])[None] /
                                        np.linalg.norm(np.cross(across, c[-1] - c[0])), 9, 0), power=2.4)
    ilium.chart = "ilium"
    parts.append(ilium)
    parts.append(knob("coxae_L", tc + [0.3, -0.5, 0.5], (3.8, 3.6, 4.8), chart="joint_knob"))
    parts.append(knob("sacrale_L", ts + [0, 1, 0], (2.6, 3.2, 3.6), chart="joint_knob"))
    isch = catmull([ace + [0, 0, -3], (ace + ti) / 2 + [-1, 2, 0], ti], 8)
    t = np.linspace(0, 1, 8)
    ischium = loft("ischium_L", isch, 1.8 + 0.4 * t, np.interp(t, [0, 1], [5.0, 6.5]), sides=10,
                   up_hint=(1, 0.3, 0), power=2.2)
    ischium.chart = "ischium"
    parts.append(ischium)
    parts.append(knob("tuber_isch_L", ti + [0.5, 0.5, -1.5], (4.2, 4.6, 4.8), chart="joint_knob"))
    parts.append(long_bone("pubis_L", ace + [-3, -4, -1], (2.0, 165.0, -92.0), 2.0, 2.8, 2.6, n=5, sides=6,
                           chart="pubis"))
    parts.append(long_bone("symph_L", ti + [-3, -1, 2], (2.0, 165.0, -94.0), 1.8, 2.4, 2.6, n=5, sides=6,
                           chart="pubis"))
    # acetabulum rim
    rim = []
    for a in np.linspace(0, 2 * np.pi, 10):
        rim.append(ace + [2.2, 6.3 * np.sin(a), 6.3 * np.cos(a)])
    ac = loft("acetab_L", np.array(rim), 1.5, 1.5, sides=5, up_hint=(1, 0, 0), cap_start=False, cap_end=False)
    ac.chart = "acetab"
    parts.append(ac)
    for p in parts:
        p.bind = WAIST
    # --- femur ---
    parts.append(knob("fem_head_L", ace + [-0.5, 0, 0], (5.0, 5.0, 5.0), chart="joint_knob", bind=L1))
    parts.append(long_bone("femur_L", (30.5, 172.0, -79.5), (28.5, 126.0, -69.0), 4.4, 6.0, 6.6, n=10,
                           chart="femur", bind=L1))
    parts.append(knob("trochanter_L", (34.5, 180.5, -88.0), (3.4, 4.8, 4.4), chart="joint_knob", bind=L1))
    # --- stifle / tibia ---
    parts.append(knob("patella_L", (29.0, 126.0, -61.5), (2.8, 4.0, 2.6), chart="joint_knob", bind=L2))
    parts.append(long_bone("tibia_L", (28.5, 121.0, -70.5), (28.5, 85.5, -92.0), 3.7, 5.8, 4.8, n=10,
                           chart="tibia", bind=L2))
    # --- hock: tarsus + calcaneus (point of hock) ---
    parts.append(knob("tarsus_L", (28.5, 80.5, -93.0), (5.2, 6.4, 5.6), chart="joint_knob", bind=L3))
    parts.append(long_bone("calcaneus_L", (28.5, 82.0, -97.5), (28.5, 94.5, -103.5), 2.4, 3.6, 3.9, n=6,
                           flat=0.8, chart="olecranon", bind=L3))
    # --- cannon, splints ---
    parts.append(long_bone("mt_cannon_L", (28.5, 76.0, -92.3), (28.5, 24.5, -89.8), 3.0, 4.2, 4.3, n=10,
                           flat=0.85, chart="cannon", bind=L3))
    # --- fetlock, pastern, hoof ---
    parts.append(knob("mt_sesamoid_L", (28.5, 23.0, -94.3), (3.6, 2.8, 2.4), chart="joint_knob", bind=L4))
    parts.append(long_bone("mt_pastern_L", (28.5, 22.0, -89.5), (28.3, 10.5, -78.0), 2.8, 4.0, 3.4, n=7,
                           chart="pastern", bind=L4))
    parts.append(hoof("mt_hoof_L", (28.0, 0.0, -70.0), (28.2, 12.0, -75.5), (8.6, 10.0), (6.2, 6.6), L4))
    return paired(parts)


def hoof(name, bottom, top, r_bottom, r_top, bind) -> Part:
    """Hoof capsule: an elliptic truncated cone from the coronet down to the ground, capped."""
    n = 5
    t = np.linspace(0, 1, n)[:, None]
    centers = np.asarray(top, float) * (1 - t) + np.asarray(bottom, float) * t
    rx = np.interp(t[:, 0], [0, 1], [r_top[0], r_bottom[0]])
    rz = np.interp(t[:, 0], [0, 1], [r_top[1], r_bottom[1]])
    p = loft(name, centers, rx, rz, sides=14, normal_hints=np.repeat([[1.0, 0, 0]], n, 0), power=2.2)
    p.chart = "hoof"
    p.bind = bind
    p.kind = "hoof"
    return p


# ------------------------------------------------------------------------------------------ sinew
CREST = [(0, 226.5, 160), (0, 229.5, 148), (0, 230.0, 132), (0, 227.5, 115), (0, 223.0, 98), (0, 218.0, 82),
         (0, 214.0, 68), (0, 211.0, 55), (0, 208.5, 45)]


def build_nuchal() -> list[Part]:
    """Nuchal ligament: the cord from poll to withers that carries the (ghostly) mane."""
    c = resample_arc(catmull(CREST, 40), 15)
    t = np.linspace(0, 1, len(c))
    cord = loft("nuchal", c, 2.6 + 0.8 * np.sin(np.pi * t), 3.2 + 1.2 * np.sin(np.pi * t), sides=8,
                up_hint=(0, 1, 0))
    cord.chart = "nuchal"
    cord.kind = "sinew"
    cord.bind = "neck_blend"
    parts = [cord]
    # thin tendon strands from the cord down onto the neck vertebrae
    for i, z in enumerate([138, 118, 104, 90, 78]):
        top = _line_at(CREST, z) + [0, -2, 0]
        bot = _line_at(NECK_LINE, z - 3) + [0, 5, 0]
        s = loft(f"lamella{i}", seg(top, bot, 4), 0.55, 1.3, sides=5, up_hint=(1, 0, 0))
        s.chart = "strand"
        s.kind = "sinew"
        s.bind = "neck_blend"
        parts.append(s)
    return parts


def build_all() -> list[Part]:
    return build_spine() + build_ribcage() + build_front_leg() + build_hind_leg() + build_nuchal()
