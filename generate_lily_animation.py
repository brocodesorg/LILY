"""
3D Botanical Lily Bouquet in a Crystal Vase — Procedural Parametric Construction & Animation
===========================================================================================
Vertical 9:16 (1080x1920) cinematic 3D animation. A bouquet of Oriental Lilies (lilac, violet,
white and blush), lavender florets, buds and glossy leaves is progressively drawn, constructed
and revealed through parametric 3D curves & subsurface-lit meshes inside a cut-crystal goblet
with an ornate gold band, standing on a rustic wooden table with purple satin, a fallen lily,
loose petals, dew crystals and a dreamy bokeh garden of fairy lights.
"""

import os
import sys
import math
import time
import subprocess
import multiprocessing as mp
import numpy as np
import cv2

cv2.setNumThreads(1)

# ============================================================================
# CONFIGURATION (9:16 Instagram Reel Format)
# ============================================================================
WIDTH = 1080
HEIGHT = 1920
FPS = 30
DURATION = 13.0
TOTAL_FRAMES = int(FPS * DURATION)  # 390 frames
SHIFT = 3
SUBPIX = 1 << SHIFT

FOV_SCALE = 1580.0
SCREEN_CY = 0.50          # vertical screen anchor of the camera target
FINAL_CAM_DIST = 4.65     # camera distance of the final composed shot
REF_PX_PER_UNIT = FOV_SCALE / FINAL_CAM_DIST
VASE_FOOT_WORLD = np.array([0.0, -1.125, 0.0], dtype=np.float32)


# ============================================================================
# MATHEMATICAL & EASING UTILITIES
# ============================================================================
def clamp(x, lo=0.0, hi=1.0):
    return np.clip(x, lo, hi)


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(1e-8, (edge1 - edge0)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(1e-8, (edge1 - edge0)), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize_v3(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-8)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def rgb_to_bgr_tuple(c):
    return (int(round(c[2] * 255)), int(round(c[1] * 255)), int(round(c[0] * 255)))


def build_basis_from_dir(direction, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """Orthonormal 3x3 basis whose local Y-axis aligns with `direction`."""
    direction = np.asarray(direction, dtype=np.float32)
    y_axis = direction / max(1e-8, np.linalg.norm(direction))
    if abs(np.dot(y_axis, up_hint)) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x_axis = np.cross(up_hint, y_axis)
    x_axis /= max(1e-8, np.linalg.norm(x_axis))
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(1e-8, np.linalg.norm(z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1).astype(np.float32)


def eval_cubic_bezier(p0, p1, p2, p3, u):
    u = np.asarray(u, dtype=np.float32)[..., None]
    om = 1.0 - u
    pos = (om ** 3) * p0 + 3.0 * (om ** 2) * u * p1 + 3.0 * om * (u ** 2) * p2 + (u ** 3) * p3
    tan = 3.0 * (om ** 2) * (p1 - p0) + 6.0 * om * u * (p2 - p1) + 3.0 * (u ** 2) * (p3 - p2)
    return pos, tan


def compute_bishop_frame(T):
    """Twist-free Bishop parallel-transport frame (N1, N2) along tangents T (Nu, 3)."""
    Nu = len(T)
    N1 = np.zeros_like(T)
    N2 = np.zeros_like(T)
    t0 = T[0]
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(t0, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n1 = np.cross(t0, ref)
    n1 /= max(1e-8, np.linalg.norm(n1))
    n2 = np.cross(t0, n1)
    N1[0], N2[0] = n1, n2
    for i in range(1, Nu):
        ti = T[i]
        n1_proj = n1 - np.dot(n1, ti) * ti
        norm_n1 = np.linalg.norm(n1_proj)
        if norm_n1 < 1e-6:
            n1_proj = np.cross(ti, n2)
            norm_n1 = np.linalg.norm(n1_proj)
        n1 = n1_proj / max(1e-8, norm_n1)
        n2 = np.cross(ti, n1)
        N1[i], N2[i] = n1, n2
    return N1, N2


# ============================================================================
# 3D BOTANICAL GEOMETRY
# ============================================================================
class BotanicalElement:
    """
    Parametric 3D surface P(u, v) of shape (Nu, Nv, 3) with base colours, plus timing.
    layer: "main" (opaque, depth sorted) | "glass" (alpha blended crystal) | "late" (opaque, drawn after glass)
    """
    def __init__(self, kind, P, C, u_grid, v_grid, wire_t0, wire_t1, surf_t0, surf_t1,
                 sss_strength=0.35, spec_strength=0.25, shininess=18.0,
                 wire_color=(0.95, 0.78, 0.55), wire_stride_u=4, wire_stride_v=3,
                 is_petal=False, P_closed=None, papillae=None,
                 layer="main", quad_valid=None, wire_cross=True):
        self.kind = kind
        self.P = P.astype(np.float32)
        self.P_closed = P_closed.astype(np.float32) if P_closed is not None else None
        self.C = np.clip(C, 0.0, 1.0).astype(np.float32)
        self.u_grid = u_grid.astype(np.float32)
        self.v_grid = v_grid.astype(np.float32)
        self.wire_t0 = wire_t0
        self.wire_t1 = wire_t1
        self.surf_t0 = surf_t0
        self.surf_t1 = surf_t1
        self.sss_strength = sss_strength
        self.spec_strength = spec_strength
        self.shininess = shininess
        self.wire_color = np.array(wire_color, dtype=np.float32)
        self.wire_stride_u = wire_stride_u
        self.wire_stride_v = wire_stride_v
        self.is_petal = is_petal
        self.papillae = papillae or []
        self.layer = layer
        self.quad_valid = quad_valid
        self.wire_cross = wire_cross
        # precomputed quad parameters (constant over time)
        U, V = self.u_grid, self.v_grid
        u_q = 0.25 * (U[:-1, :-1] + U[1:, :-1] + U[1:, 1:] + U[:-1, 1:])
        v_q = 0.25 * (np.abs(V[:-1, :-1]) + np.abs(V[1:, :-1]) + np.abs(V[1:, 1:]) + np.abs(V[:-1, 1:]))
        self.eff_u = (0.85 * u_q + 0.15 * v_q).astype(np.float32)
        self.c_q = 0.25 * (self.C[:-1, :-1] + self.C[1:, :-1] + self.C[1:, 1:] + self.C[:-1, 1:])

    def get_positions_and_normals(self, bloom_t=1.0):
        if self.P_closed is not None and bloom_t < 0.999:
            P = self.P_closed * (1.0 - bloom_t) + self.P * bloom_t
        else:
            P = self.P
        p00 = P[:-1, :-1]
        p10 = P[1:, :-1]
        p11 = P[1:, 1:]
        p01 = P[:-1, 1:]
        d_u = 0.5 * ((p10 - p00) + (p11 - p01))
        d_v = 0.5 * ((p01 - p00) + (p11 - p10))
        normals = normalize_v3(np.cross(d_v, d_u))
        return P, p00, p10, p11, p01, normals


def generate_stem_tube(p0, p1, p2, p3, r_base, r_tip, Nu, Nv,
                       col_base, col_tip, wire_t0, wire_t1, surf_t0, surf_t1,
                       flare_tip=True):
    """Smooth twist-free tubular stem along a cubic Bezier spline."""
    p0, p1, p2, p3 = [np.asarray(p, dtype=np.float32) for p in (p0, p1, p2, p3)]
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    pos, tan = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(tan)
    N1, N2 = compute_bishop_frame(T)

    P = np.zeros((Nu, Nv, 3), dtype=np.float32)
    C = np.zeros((Nu, Nv, 3), dtype=np.float32)
    for i in range(Nu):
        n1, n2 = N1[i], N2[i]
        r = r_base * (1.0 - u[i]) + r_tip * u[i]
        if flare_tip and u[i] > 0.90:
            r *= 1.0 + 0.25 * smoothstep(0.90, 1.0, u[i])
        ridge = 1.0 + 0.025 * np.cos(6.0 * theta[i])
        rad = r * ridge
        P[i] = pos[i] + rad[:, None] * (np.cos(theta[i])[:, None] * n1 + np.sin(theta[i])[:, None] * n2)
        mix = u[i]
        base_c = (1.0 - mix) * col_base + mix * col_tip
        striation = 0.95 + 0.05 * np.cos(6.0 * theta[i])[:, None]
        C[i] = base_c * striation

    el = BotanicalElement(
        kind="stem", P=P, C=C, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.22, spec_strength=0.25, shininess=18.0,
        wire_color=(0.38, 0.88, 0.50), wire_stride_u=6, wire_stride_v=4
    )
    return el, pos, T


def generate_lanceolate_leaf(origin, azimuth, length, width, arch_drop, twist,
                             wire_t0, wire_t1, surf_t0, surf_t1):
    """Glossy lanceolate lily leaf with parallel venation and a natural arch."""
    Nu, Nv = 36, 15
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    w_prof = (
        width * (np.maximum(0.0, np.sin(math.pi * (U ** 0.66))) ** 0.92) * (1.0 - 0.12 * U)
        + 0.10 * width * ((1.0 - U) ** 3)
    )
    reach = length * (0.90 * U + 0.10 * np.sin(0.5 * math.pi * U))
    elev = length * (0.46 * U - arch_drop * (U ** 2.1))
    fold = 0.055 * width * (np.abs(V) ** 1.35) * (1.0 - 0.45 * U) - 0.025 * width * (V ** 4)
    tw_ang = twist * (U ** 1.35)
    x_loc = w_prof * V * np.cos(tw_ang) - fold * np.sin(tw_ang)
    z_loc = elev + w_prof * V * np.sin(tw_ang) + fold * np.cos(tw_ang)
    y_loc = reach

    P_local = np.stack([x_loc, z_loc, y_loc], axis=-1)
    rot = rot_y(azimuth)
    P_world = np.asarray(origin, dtype=np.float32) + P_local @ rot.T

    c_dark = np.array([0.06, 0.24, 0.11], dtype=np.float32)
    c_mid = np.array([0.13, 0.42, 0.18], dtype=np.float32)
    c_vein = np.array([0.34, 0.63, 0.30], dtype=np.float32)
    c_tip = np.array([0.22, 0.48, 0.18], dtype=np.float32)

    midrib_mask = np.exp(-((V / 0.10) ** 2))[..., None]
    parallel_veins = (0.5 + 0.5 * np.cos(V * 8.0 * math.pi))[..., None] ** 4
    edge_grad = (np.abs(V) ** 1.4)[..., None]

    C = (1.0 - edge_grad) * c_mid + edge_grad * c_dark
    C = C * (1.0 - 0.32 * (U ** 2)[..., None]) + c_tip * (0.32 * (U ** 2)[..., None])
    C = C * (1.0 - 0.48 * midrib_mask) + c_vein * (0.48 * midrib_mask)
    C = C + 0.06 * parallel_veins * c_vein

    return BotanicalElement(
        kind="leaf", P=P_world, C=C, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.30, spec_strength=0.42, shininess=28.0,
        wire_color=(0.45, 0.92, 0.54), wire_stride_u=7, wire_stride_v=4
    )


# ----------------------------------------------------------------------------
# Lily colour palettes (RGB 0..1) and papilla spot colours
# ----------------------------------------------------------------------------
LILY_VARIANTS = {
    # pale lilac-pink with white margins, soft pink heart (hero bloom)
    "lilac_white": dict(edge=(0.985, 0.96, 0.985), blush=(0.88, 0.72, 0.95), deep=(0.78, 0.46, 0.82),
                        nectar=(0.60, 0.78, 0.30), spot=(0.40, 0.08, 0.18), spots=16),
    # saturated violet-magenta bloom with dark maroon papillae
    "violet": dict(edge=(0.93, 0.84, 0.97), blush=(0.72, 0.52, 0.90), deep=(0.62, 0.26, 0.70),
                   nectar=(0.55, 0.74, 0.28), spot=(0.36, 0.05, 0.16), spots=44),
    # snow-white with pale green-yellow throat
    "white_green": dict(edge=(1.0, 0.995, 0.99), blush=(0.98, 0.95, 0.96), deep=(0.95, 0.86, 0.90),
                        nectar=(0.78, 0.88, 0.42), spot=(0.42, 0.10, 0.14), spots=10),
    # white with a breath of pink
    "white_pink": dict(edge=(0.995, 0.975, 0.985), blush=(0.97, 0.87, 0.94), deep=(0.92, 0.66, 0.82),
                       nectar=(0.70, 0.84, 0.36), spot=(0.42, 0.10, 0.18), spots=18),
    # deep orchid-lilac bloom
    "orchid_deep": dict(edge=(0.86, 0.74, 0.95), blush=(0.62, 0.42, 0.82), deep=(0.46, 0.22, 0.64),
                        nectar=(0.52, 0.70, 0.26), spot=(0.34, 0.04, 0.16), spots=36),
    # fallen pink-lilac bloom lying on the table
    "fallen_pink": dict(edge=(0.99, 0.94, 0.98), blush=(0.94, 0.72, 0.90), deep=(0.84, 0.50, 0.80),
                        nectar=(0.66, 0.82, 0.32), spot=(0.46, 0.10, 0.28), spots=14),
}


def generate_lily_tepal(center, basis, azimuth, is_inner, scale, openness,
                        color_variant, wire_t0, wire_t1, surf_t0, surf_t1, rng_seed):
    """High-resolution Oriental Lily tepal with pointed tip, recurve, ruffled margin and papillae."""
    rng = np.random.RandomState(rng_seed)
    Nu, Nv = 72, 33
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')

    w_max = (0.54 if is_inner else 0.44) * scale
    length = (1.26 if is_inner else 1.24) * scale

    envelope = np.maximum(0.0, np.sin(math.pi * (U ** 0.58))) ** 0.86
    tip_round = 1.0 - 0.10 * (U ** 3)
    w_profile = w_max * envelope * tip_round + 0.050 * scale * ((1.0 - U) ** 2.6)

    def build_tepal_surface(open_factor):
        theta_base = (0.18 if is_inner else 0.30)
        theta_flare = (0.74 if is_inner else 0.88) * open_factor + 0.14
        theta_recurve = (1.55 if is_inner else 1.72) * (open_factor ** 1.18)
        angles = (
            theta_base
            + theta_flare * smoothstep(0.04, 0.58, u)
            + (theta_recurve - theta_flare) * (smoothstep(0.42, 1.0, u) ** 1.25)
        )
        ds = length / (Nu - 1)
        r_mid = np.zeros(Nu, dtype=np.float32)
        y_mid = np.zeros(Nu, dtype=np.float32)
        r_mid[0] = (0.036 if is_inner else 0.078) * scale
        y_mid[0] = (0.035 if is_inner else 0.00) * scale
        for i in range(1, Nu):
            ang = 0.5 * (angles[i - 1] + angles[i])
            r_mid[i] = r_mid[i - 1] + ds * math.sin(ang)
            y_mid[i] = y_mid[i - 1] + ds * math.cos(ang)

        R_mid = r_mid[:, None]
        Y_mid = y_mid[:, None]
        Ang = angles[:, None]
        out_r = np.cos(Ang)
        out_y = -np.sin(Ang)

        furrow = 0.028 * scale * np.exp(-((V / 0.16) ** 2)) * (1.0 - 0.60 * U)
        reflex = 0.045 * scale * (V ** 2) * smoothstep(0.35, 0.90, U)
        ruffle_amp = (0.044 if is_inner else 0.028) * scale * open_factor
        ruffle_env = np.maximum(0.0, np.sin(math.pi * U)) * smoothstep(0.15, 0.88, U) * (np.abs(V) ** 2.3)
        phase = 0.85 if is_inner else -0.75
        ruffle = ruffle_amp * ruffle_env * np.sin(6.0 * math.pi * U + phase * np.sign(V))
        disp_out = furrow + reflex + ruffle

        wrap_r = R_mid + 0.16 * scale
        phi_local = (w_profile * V) / wrap_r
        cyl_blend = 1.0 - 0.55 * smoothstep(0.35, 0.85, U)
        phi_eff = phi_local * cyl_blend
        x_planar = (w_profile * V) * (1.0 - cyl_blend)

        r_base_surf = wrap_r * np.cos(phi_eff) - 0.16 * scale + disp_out * out_r
        x_base_surf = wrap_r * np.sin(phi_eff) + x_planar
        y_local = Y_mid + disp_out * out_y

        ca, sa = math.cos(azimuth), math.sin(azimuth)
        px = x_base_surf * ca + r_base_surf * sa
        pz = -x_base_surf * sa + r_base_surf * ca
        py = y_local
        P_flower = np.stack([px, py, pz], axis=-1)
        return center + P_flower @ basis.T

    P_full = build_tepal_surface(openness)
    P_semi = build_tepal_surface(max(0.28, openness * 0.45))

    # ---- pigmentation -------------------------------------------------------
    var = LILY_VARIANTS[color_variant]
    c_edge = np.array(var["edge"], dtype=np.float32)
    c_blush = np.array(var["blush"], dtype=np.float32)
    c_deep = np.array(var["deep"], dtype=np.float32)
    c_nectar = np.array(var["nectar"], dtype=np.float32)
    spot_rgb = np.array(var["spot"], dtype=np.float32)
    pap_core_bgr = rgb_to_bgr_tuple(spot_rgb)
    pap_rim_bgr = rgb_to_bgr_tuple(np.clip(spot_rgb * 0.5 + c_deep * 0.5, 0, 1))

    C = np.broadcast_to(c_edge, (Nu, Nv, 3)).copy()

    u_flush = smoothstep(0.04, 0.18, U) * (1.0 - smoothstep(0.52, 0.92, U))
    v_margin = 1.0 - smoothstep(0.38, 0.88, np.abs(V))
    flush_mask = (u_flush * v_margin)[..., None]
    core_mask = (
        smoothstep(0.06, 0.18, U) * (1.0 - smoothstep(0.34, 0.70, U)) * np.exp(-((V / 0.42) ** 2))
    )[..., None]

    C = C * (1.0 - flush_mask) + c_blush * flush_mask
    C = C * (1.0 - 0.80 * core_mask) + c_deep * (0.80 * core_mask)

    nectar_mask = (
        (1.0 - smoothstep(0.04, 0.26, U)) * np.exp(-((V / (0.34 - 0.16 * U)) ** 2))
    )[..., None]
    C = C * (1.0 - 0.90 * nectar_mask) + c_nectar * (0.90 * nectar_mask)

    midrib_vein = (
        smoothstep(0.14, 0.28, U) * (1.0 - smoothstep(0.58, 0.92, U)) * np.exp(-((V / 0.058) ** 2))
    )[..., None]
    c_midrib = 0.72 * c_deep + 0.28 * c_edge
    C = C * (1.0 - 0.55 * midrib_vein) + c_midrib * (0.55 * midrib_vein)

    vein_rib = (0.5 + 0.5 * np.cos(V * 18.0 * math.pi))[..., None]
    vein_strength = 0.045 * (1.0 - nectar_mask) * smoothstep(0.08, 0.88, U)[..., None]
    C = C - vein_strength * (1.0 - vein_rib) * np.array([0.05, 0.11, 0.07], dtype=np.float32)

    papillae = []
    n_pap = int(var["spots"] * (1.25 if is_inner else 1.0))
    for _ in range(n_pap):
        pu = rng.uniform(0.15, 0.58)
        pv_side = -1.0 if rng.uniform() < 0.5 else 1.0
        pv = pv_side * rng.uniform(0.08, 0.55)
        u_idx = int(np.clip(round(pu * (Nu - 1)), 1, Nu - 2))
        v_idx = int(np.clip(round(((pv + 1.0) * 0.5) * (Nv - 1)), 1, Nv - 2))
        rad_px = rng.uniform(1.5, 2.6) * scale * 1.5
        eff_u = 0.85 * pu + 0.15 * abs(pv)
        papillae.append((u_idx, v_idx, rad_px, pap_core_bgr, pap_rim_bgr, eff_u))

    return BotanicalElement(
        kind="petal", P=P_full, C=C, u_grid=U, v_grid=V, P_closed=P_semi,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.46, spec_strength=0.26, shininess=22.0,
        wire_color=(1.0, 0.72, 0.88), wire_stride_u=10, wire_stride_v=5,
        is_petal=True, papillae=papillae
    )


def generate_stamens_and_pistil(center, basis, scale, wire_t0, wire_t1, surf_t0, surf_t1):
    """6 curving stamens (cream filaments + maroon-brown anthers) and 1 central pistil."""
    elements = []

    # --- Pistil: green style with a 3-lobed pink-cream stigma
    Nu_s, Nv_s = 26, 13
    u = np.linspace(0.0, 1.0, Nu_s, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv_s, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    p0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p1 = np.array([0.02, 0.36, 0.03], dtype=np.float32) * scale
    p2 = np.array([0.04, 0.74, 0.07], dtype=np.float32) * scale
    p3 = np.array([0.02, 1.06, 0.14], dtype=np.float32) * scale
    pos, tan = eval_cubic_bezier(p0, p1, p2, p3, u)
    T = normalize_v3(tan)
    N1, N2 = compute_bishop_frame(T)

    P_style = np.zeros((Nu_s, Nv_s, 3), dtype=np.float32)
    C_style = np.zeros((Nu_s, Nv_s, 3), dtype=np.float32)
    c_green = np.array([0.62, 0.80, 0.38], dtype=np.float32)
    c_stigma = np.array([0.80, 0.50, 0.58], dtype=np.float32)

    for i in range(Nu_s):
        n1, n2 = N1[i], N2[i]
        tip_close = math.sin(math.pi * float(u[i])) ** 0.22 if u[i] > 0.92 else 1.0
        base_r = scale * 0.020 * (1.0 - 0.20 * u[i])
        if u[i] > 0.84:
            sw = math.sin(math.pi * (float(u[i]) - 0.84) / 0.16)
            lobe = 1.0 + 0.45 * np.cos(3.0 * theta[i])
            r = (base_r + scale * 0.022 * sw * lobe) * tip_close
        else:
            r = np.full_like(theta[i], base_r)
        P_style[i] = pos[i] + r[:, None] * (np.cos(theta[i])[:, None] * n1 + np.sin(theta[i])[:, None] * n2)
        stigma_mix = smoothstep(0.82, 0.92, u[i])
        C_style[i] = (1.0 - stigma_mix) * c_green + stigma_mix * c_stigma

    elements.append(BotanicalElement(
        kind="pistil", P=center + P_style @ basis.T, C=C_style, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.30, spec_strength=0.38, shininess=28.0,
        wire_color=(0.75, 0.95, 0.55), wire_stride_u=6, wire_stride_v=4
    ))

    # --- Six curved filaments + maroon versatile anthers
    for k in range(6):
        az = k * (math.pi / 3.0) + 0.26
        ca, sa = math.cos(az), math.sin(az)
        flare = (0.30 + 0.04 * (k % 2)) * scale
        h_tip = (0.90 + 0.05 * (k % 3)) * scale

        fp0 = np.array([0.028 * scale * ca, 0.02 * scale, 0.028 * scale * sa], dtype=np.float32)
        fp1 = np.array([0.075 * scale * ca, 0.34 * scale, 0.075 * scale * sa], dtype=np.float32)
        fp2 = np.array([0.205 * scale * ca, 0.70 * scale, 0.205 * scale * sa], dtype=np.float32)
        fp3 = np.array([flare * ca, h_tip, flare * sa], dtype=np.float32)

        Nu_f, Nv_f = 20, 7
        uf = np.linspace(0.0, 1.0, Nu_f, dtype=np.float32)
        vf = np.linspace(-1.0, 1.0, Nv_f, dtype=np.float32)
        Uf, Vf = np.meshgrid(uf, vf, indexing='ij')
        th_f = (Vf + 1.0) * math.pi

        fpos, ftan = eval_cubic_bezier(fp0, fp1, fp2, fp3, uf)
        Tf = normalize_v3(ftan)
        N1f, N2f = compute_bishop_frame(Tf)

        P_fil = np.zeros((Nu_f, Nv_f, 3), dtype=np.float32)
        C_fil = np.zeros((Nu_f, Nv_f, 3), dtype=np.float32)
        c_fil_base = np.array([0.70, 0.84, 0.46], dtype=np.float32)
        c_fil_tip = np.array([0.98, 0.95, 0.86], dtype=np.float32)

        for i in range(Nu_f):
            rf = scale * (0.011 * (1.0 - 0.45 * uf[i]))
            P_fil[i] = fpos[i] + rf * (np.cos(th_f[i])[:, None] * N1f[i] + np.sin(th_f[i])[:, None] * N2f[i])
            mix = smoothstep(0.15, 0.75, uf[i])
            C_fil[i] = (1.0 - mix) * c_fil_base + mix * c_fil_tip

        st_delay = 0.06 * k
        elements.append(BotanicalElement(
            kind="filament", P=center + P_fil @ basis.T, C=C_fil, u_grid=Uf, v_grid=Vf,
            wire_t0=wire_t0 + st_delay, wire_t1=wire_t1 + st_delay * 0.5,
            surf_t0=surf_t0 + st_delay, surf_t1=surf_t1 + st_delay * 0.5,
            sss_strength=0.35, spec_strength=0.25, shininess=20.0,
            wire_color=(0.95, 0.90, 0.65), wire_stride_u=5, wire_stride_v=3
        ))

        Nu_a, Nv_a = 12, 9
        ua = np.linspace(0.0, 1.0, Nu_a, dtype=np.float32)
        va = np.linspace(-1.0, 1.0, Nv_a, dtype=np.float32)
        Ua, Va = np.meshgrid(ua, va, indexing='ij')
        th_a = (Va + 1.0) * math.pi

        anther_len = 0.170 * scale
        rad_dir = np.array([ca, 0.0, sa], dtype=np.float32)
        tan_dir = np.array([-sa, 0.0, ca], dtype=np.float32)
        a_axis = normalize_v3(0.75 * tan_dir + 0.35 * np.array([0.0, 1.0, 0.0], dtype=np.float32) + 0.25 * rad_dir)
        a_n1 = normalize_v3(np.cross(a_axis, rad_dir))
        a_n2 = normalize_v3(np.cross(a_axis, a_n1))

        P_anth = np.zeros((Nu_a, Nv_a, 3), dtype=np.float32)
        C_anth = np.zeros((Nu_a, Nv_a, 3), dtype=np.float32)
        c_pollen_1 = np.array([0.40, 0.11, 0.10], dtype=np.float32)   # dark maroon-brown
        c_pollen_2 = np.array([0.66, 0.24, 0.16], dtype=np.float32)   # warm russet dust

        for i in range(Nu_a):
            s_along = (ua[i] - 0.5) * anther_len
            env = max(0.0, math.sin(math.pi * float(ua[i]))) ** 0.60
            bilobe = 1.0 + 0.28 * np.cos(2.0 * th_a[i])
            ra = 0.022 * scale * env * bilobe
            P_anth[i] = fp3 + s_along * a_axis + ra[:, None] * (
                np.cos(th_a[i])[:, None] * a_n1 + np.sin(th_a[i])[:, None] * a_n2
            )
            grain = 0.5 + 0.5 * np.sin(19.0 * ua[i] + 7.0 * th_a[i])[:, None]
            C_anth[i] = (1.0 - 0.45 * grain) * c_pollen_1 + (0.45 * grain) * c_pollen_2

        elements.append(BotanicalElement(
            kind="anther", P=center + P_anth @ basis.T, C=C_anth, u_grid=Ua, v_grid=Va,
            wire_t0=wire_t0 + 0.25 + st_delay, wire_t1=wire_t1 + 0.20 + st_delay * 0.5,
            surf_t0=surf_t0 + 0.25 + st_delay, surf_t1=surf_t1 + st_delay * 0.5,
            sss_strength=0.22, spec_strength=0.15, shininess=10.0,
            wire_color=(1.0, 0.65, 0.45), wire_stride_u=4, wire_stride_v=3
        ))

    return elements


BUD_VARIANTS = {
    "lilac": dict(body=(0.62, 0.72, 0.42), rose=(0.72, 0.52, 0.86), tip=(0.80, 0.72, 0.90)),
    "green": dict(body=(0.46, 0.66, 0.30), rose=(0.62, 0.78, 0.42), tip=(0.78, 0.86, 0.62)),
    "ivory": dict(body=(0.60, 0.74, 0.44), rose=(0.88, 0.80, 0.70), tip=(0.94, 0.90, 0.80)),
}


def generate_lily_bud(center, basis, scale, variant, wire_t0, wire_t1, surf_t0, surf_t1):
    """Unopened lily bud with three sepal ribs and a pointed tip."""
    Nu, Nv = 34, 21
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    length = 0.90 * scale
    r_prof = (
        0.21 * scale * (np.maximum(0.0, np.sin(math.pi * (U ** 0.66))) ** 1.12) * (1.0 - 0.12 * U)
        + 0.028 * scale * ((1.0 - U) ** 2)
    )
    rib = 1.0 + 0.08 * np.cos(3.0 * theta) * np.maximum(0.0, np.sin(math.pi * U))
    R = r_prof * rib
    Y = length * U
    P_bud = center + np.stack([R * np.cos(theta), Y, R * np.sin(theta)], axis=-1) @ basis.T

    bv = BUD_VARIANTS[variant]
    c_body = np.array(bv["body"], dtype=np.float32)
    c_rose = np.array(bv["rose"], dtype=np.float32)
    c_tip_c = np.array(bv["tip"], dtype=np.float32)

    rose_mix = smoothstep(0.18, 0.60, U)[..., None] * (1.0 - 0.40 * (0.5 + 0.5 * np.cos(3.0 * theta))[..., None])
    tip_mix = smoothstep(0.80, 0.99, U)[..., None]
    C = (1.0 - rose_mix) * c_body + rose_mix * c_rose
    C = (1.0 - 0.6 * tip_mix) * C + 0.6 * tip_mix * c_tip_c

    return BotanicalElement(
        kind="bud", P=P_bud, C=C, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.38, spec_strength=0.34, shininess=24.0,
        wire_color=(0.78, 0.82, 0.98), wire_stride_u=6, wire_stride_v=4
    )


# ----------------------------------------------------------------------------
# Lavender / larkspur florets on slender sprigs
# ----------------------------------------------------------------------------
FLORET_VARIANTS = {
    "lavender": (np.array([0.66, 0.56, 0.92], dtype=np.float32), np.array([0.88, 0.82, 0.98], dtype=np.float32)),
    "periwinkle": (np.array([0.56, 0.54, 0.94], dtype=np.float32), np.array([0.82, 0.82, 0.99], dtype=np.float32)),
    "orchid": (np.array([0.80, 0.56, 0.88], dtype=np.float32), np.array([0.94, 0.82, 0.96], dtype=np.float32)),
}


def generate_sprig(p0, p1, p2, p3, n_florets, r_floret, variant, seed,
                   wt0, wt1, st0, st1, first_u=0.38):
    """A thin green sprig plus one merged element holding all of its 5-petal florets."""
    rng = np.random.RandomState(seed)
    c_stem_a = np.array([0.24, 0.46, 0.20], dtype=np.float32)
    c_stem_b = np.array([0.40, 0.62, 0.30], dtype=np.float32)
    stem_el, pos, T = generate_stem_tube(p0, p1, p2, p3, 0.017, 0.010, 32, 8,
                                         c_stem_a, c_stem_b, wt0, wt1, st0, st1, flare_tip=False)
    N1, N2 = compute_bishop_frame(T)
    c_lav, c_light = FLORET_VARIANTS[variant]

    Nu_f, Nv_f = 5, 17
    u = np.linspace(0.0, 1.0, Nu_f, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv_f, dtype=np.float32)
    Uf, Vf = np.meshgrid(u, v, indexing='ij')
    TH = (Vf + 1.0) * math.pi

    P_all, C_all, U_all, V_all = [], [], [], []
    Nu_stem = len(pos)
    for k in range(n_florets):
        frac = k / max(1, n_florets - 1)
        uu = first_u + (1.0 - first_u) * frac
        idx = int(round(uu * (Nu_stem - 1)))
        phi = k * 2.4 + rng.uniform(-0.4, 0.4)
        outward = math.cos(phi) * N1[idx] + math.sin(phi) * N2[idx]
        direction = normalize_v3(0.75 * outward + 0.30 * T[idx] + 0.55 * np.array([0.0, 0.0, 1.0], dtype=np.float32))
        basis = build_basis_from_dir(direction)
        r0 = r_floret * (1.0 - 0.40 * frac) * rng.uniform(0.9, 1.1)
        center = pos[idx] + outward * 0.030

        lobes = 0.60 + 0.40 * np.abs(np.cos(2.5 * TH + rng.uniform(0, 0.3))) ** 0.8
        rad = Uf * lobes * r0
        x = rad * np.cos(TH)
        z = rad * np.sin(TH)
        y = 0.30 * r0 * (Uf ** 1.6) - 0.08 * r0 * np.exp(-((Uf / 0.2) ** 2))
        P_loc = np.stack([x, y, z], axis=-1)
        P_all.append(center + P_loc @ basis.T)

        centre_mix = (1.0 - smoothstep(0.06, 0.26, Uf))[..., None]
        rim_mix = (0.65 * smoothstep(0.55, 1.0, Uf))[..., None]
        vein = (1.0 - 0.10 * (0.5 + 0.5 * np.cos(5.0 * TH)))[..., None]
        Ck = ((1.0 - rim_mix) * c_lav + rim_mix * c_light) * vein
        Ck = (1.0 - centre_mix) * Ck + centre_mix * np.array([0.92, 0.88, 0.55], dtype=np.float32)
        C_all.append(Ck)

        U_all.append(Uf * 0.55 + 0.45 * frac)
        V_all.append(Vf)

    P_m = np.concatenate(P_all, axis=1)
    C_m = np.concatenate(C_all, axis=1)
    U_m = np.concatenate(U_all, axis=1)
    V_m = np.concatenate(V_all, axis=1)
    quad_valid = np.ones((Nu_f - 1, P_m.shape[1] - 1), dtype=bool)
    for k in range(1, n_florets):
        quad_valid[:, k * Nv_f - 1] = False

    fl_el = BotanicalElement(
        kind="floret", P=P_m, C=C_m, u_grid=U_m, v_grid=V_m,
        wire_t0=wt0 + 0.5, wire_t1=wt1 + 0.5, surf_t0=st0 + 0.4, surf_t1=st1 + 0.8,
        sss_strength=0.48, spec_strength=0.18, shininess=16.0,
        wire_color=(0.85, 0.80, 1.0), wire_stride_u=50, wire_stride_v=4,
        quad_valid=quad_valid, wire_cross=False
    )
    return [stem_el, fl_el]


def generate_loose_petal(center, azimuth, length, width, tilt, variant, wire_t0, wire_t1, surf_t0, surf_t1):
    """A single detached tepal lying on the table, gently cupped and curled at the tip."""
    Nu, Nv = 30, 13
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    w = width * (np.maximum(0.0, np.sin(math.pi * U ** 0.62)) ** 0.85) + 0.012 * (1 - U) ** 2
    x = w * V
    z = length * (U - 0.5)
    y = 0.045 * (V ** 2) * w / max(width, 1e-5) + 0.10 * length * (U ** 2.6) + 0.02 * np.sin(5.0 * U) * np.abs(V)
    P_loc = np.stack([x, y, z], axis=-1)
    R = rot_y(azimuth) @ rot_x(tilt)
    P = np.asarray(center, dtype=np.float32) + P_loc @ R.T

    var = LILY_VARIANTS[variant]
    c_edge = np.array(var["edge"], dtype=np.float32)
    c_blush = np.array(var["blush"], dtype=np.float32)
    mix = (1.0 - smoothstep(0.0, 0.75, U))[..., None] * (1.0 - smoothstep(0.3, 0.95, np.abs(V)))[..., None]
    C = c_edge * (1.0 - 0.7 * mix) + c_blush * 0.7 * mix
    return BotanicalElement(
        kind="loose_petal", P=P, C=C, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        sss_strength=0.5, spec_strength=0.22, shininess=18.0,
        wire_color=(1.0, 0.80, 0.92), wire_stride_u=6, wire_stride_v=4
    )


# ----------------------------------------------------------------------------
# Cut-crystal goblet + ornate gold band
# ----------------------------------------------------------------------------
VASE_PROFILE = [  # (radius, y) from foot centre up to rim
    (0.000, -1.125), (0.200, -1.125), (0.285, -1.117), (0.302, -1.097), (0.275, -1.070),
    (0.150, -1.035), (0.078, -1.000), (0.052, -0.950), (0.052, -0.890), (0.088, -0.835),
    (0.098, -0.785), (0.076, -0.735), (0.052, -0.690), (0.058, -0.645), (0.105, -0.600),
    (0.172, -0.540), (0.232, -0.420), (0.272, -0.270), (0.292, -0.150), (0.300, -0.100),
]


def _vase_profile_dense(n=240):
    pts = np.array(VASE_PROFILE, dtype=np.float32)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s /= s[-1]
    sd = np.linspace(0.0, 1.0, n, dtype=np.float32)
    r = np.interp(sd, s, pts[:, 0])
    y = np.interp(sd, s, pts[:, 1])
    k = 9
    kern = np.ones(k, dtype=np.float32) / k
    rp = np.pad(r, (k // 2, k // 2), mode='edge')
    yp = np.pad(y, (k // 2, k // 2), mode='edge')
    r_s = np.convolve(rp, kern, mode='valid')
    y_s = np.convolve(yp, kern, mode='valid')
    r_s[0], y_s[0] = r[0], y[0]
    r_s[-1], y_s[-1] = r[-1], y[-1]
    return r_s, y_s


def generate_vase(wire_t0, wire_t1, surf_t0, surf_t1):
    r_d, y_d = _vase_profile_dense()
    Nu, Nv = 96, 120
    idx = np.linspace(0, len(r_d) - 1, Nu)
    r_u = np.interp(idx, np.arange(len(r_d)), r_d).astype(np.float32)
    y_u = np.interp(idx, np.arange(len(y_d)), y_d).astype(np.float32)
    u = np.linspace(0.0, 1.0, Nu, dtype=np.float32)
    v = np.linspace(-1.0, 1.0, Nv, dtype=np.float32)
    U, V = np.meshgrid(u, v, indexing='ij')
    theta = (V + 1.0) * math.pi

    # cut-crystal flutes around the cup and the knop
    cup_mask = smoothstep(-0.64, -0.50, y_u)[:, None]
    knop_mask = np.exp(-(((y_u + 0.81) / 0.07) ** 2))[:, None]
    flutes = 1.0 + 0.020 * np.cos(10.0 * theta) * cup_mask + 0.030 * np.cos(8.0 * theta) * knop_mask
    R = r_u[:, None] * flutes
    P = np.stack([R * np.cos(theta), np.broadcast_to(y_u[:, None], R.shape), R * np.sin(theta)], axis=-1)

    C = np.ones((Nu, Nv, 3), dtype=np.float32)
    glass = BotanicalElement(
        kind="glass", P=P, C=C, u_grid=U, v_grid=V,
        wire_t0=wire_t0, wire_t1=wire_t1, surf_t0=surf_t0, surf_t1=surf_t1,
        wire_color=(0.85, 0.92, 1.0), wire_stride_u=7, wire_stride_v=8, layer="glass"
    )

    # --- gold band around the cup
    yb0, yb1 = -0.455, -0.325
    Nub, Nvb = 34, 120
    ub = np.linspace(0.0, 1.0, Nub, dtype=np.float32)
    Ub, Vb = np.meshgrid(ub, v, indexing='ij')
    thb = (Vb + 1.0) * math.pi
    yb = yb0 + (yb1 - yb0) * ub
    r_cup = np.interp(yb, y_d[len(y_d) // 2:], r_d[len(r_d) // 2:]).astype(np.float32)
    prof = (0.010 + 0.012 * np.maximum(0.0, np.sin(math.pi * ub)) ** 0.5
            + 0.007 * np.exp(-(((ub - 0.07) / 0.045) ** 2)) + 0.007 * np.exp(-(((ub - 0.93) / 0.045) ** 2)))
    medallion = 0.030 * np.exp(-(((thb - math.pi / 2) / 0.30) ** 2 + ((Ub - 0.5) / 0.30) ** 2))
    scroll = 0.0025 * np.sin(24.0 * thb) * np.exp(-(((Ub - 0.5) / 0.22) ** 2))
    Rb = (r_cup[:, None] + prof[:, None]) + medallion + scroll
    Pb = np.stack([Rb * np.cos(thb), np.broadcast_to(yb[:, None], Rb.shape), Rb * np.sin(thb)], axis=-1)

    gold_hi = np.array([0.98, 0.80, 0.38], dtype=np.float32)
    gold_mid = np.array([0.82, 0.60, 0.22], dtype=np.float32)
    gold_lo = np.array([0.46, 0.30, 0.10], dtype=np.float32)
    pat = 0.55 + 0.20 * np.sin(24.0 * thb) * np.exp(-(((Ub - 0.5) / 0.22) ** 2))
    edge_line = np.exp(-(((Ub - 0.08) / 0.05) ** 2)) + np.exp(-(((Ub - 0.92) / 0.05) ** 2))
    Cb = gold_mid * (1.0 - pat[..., None]) + gold_hi * pat[..., None]
    Cb = Cb * (1.0 - 0.45 * edge_line[..., None]) + gold_lo * 0.45 * edge_line[..., None]
    med = np.exp(-(((thb - math.pi / 2) / 0.30) ** 2 + ((Ub - 0.5) / 0.30) ** 2))[..., None]
    Cb = Cb * (1.0 - 0.5 * med) + gold_hi * 0.5 * med
    band = BotanicalElement(
        kind="band", P=Pb, C=Cb, u_grid=Ub, v_grid=Vb,
        wire_t0=wire_t0 + 1.0, wire_t1=wire_t1 + 1.0, surf_t0=surf_t0 + 1.0, surf_t1=surf_t1 + 0.8,
        sss_strength=0.10, spec_strength=0.95, shininess=36.0,
        wire_color=(1.0, 0.85, 0.45), wire_stride_u=3, wire_stride_v=8, layer="late"
    )
    return [glass, band]


# ============================================================================
# FULL SCENE ASSEMBLY
# ============================================================================
def _stem_to_flower(start, end, dirv, r_base, r_tip, wt0, wt1, st0, st1, Nu=40):
    start = np.asarray(start, dtype=np.float32)
    end = np.asarray(end, dtype=np.float32)
    dirv = np.asarray(dirv, dtype=np.float32)
    dirv = dirv / np.linalg.norm(dirv)
    d = float(np.linalg.norm(end - start))
    p1 = start + np.array([end[0] * 0.10, 0.45 * d, end[2] * 0.06], dtype=np.float32)
    p2 = end - dirv * (0.42 * d)
    c_a = np.array([0.20, 0.44, 0.18], dtype=np.float32)
    c_b = np.array([0.38, 0.62, 0.28], dtype=np.float32)
    el, _, _ = generate_stem_tube(start, p1, p2, end, r_base, r_tip, Nu, 11, c_a, c_b,
                                  wt0, wt1, st0, st1)
    return el


def build_lily_bouquet_scene():
    """
    Crystal goblet + gold band, 7 blooming lilies (lilac/violet/white/blush/orchid), 3 buds,
    6 lavender sprigs, 11 glossy leaves, 1 fallen lily and 2 loose petals on the table.
    """
    elements = []

    # 1. Crystal goblet and gold band (0.0s -> 5.0s)
    elements.extend(generate_vase(wire_t0=0.0, wire_t1=2.6, surf_t0=1.0, surf_t1=4.0))

    # 2. Blooming lilies -----------------------------------------------------
    flowers_spec = [
        # hero: pale lilac-pink, front & centre
        dict(center=(0.03, 0.54, 0.55), dir=(0.00, 0.35, 0.94), scale=0.50, openness=1.00, variant="lilac_white",
             rot=0.20, wire=(2.4, 5.0), surf=(4.4, 8.0), stamen=7.2, seed=100),
        # upper-left violet bloom
        dict(center=(-0.43, 1.00, 0.20), dir=(-0.30, 0.32, 0.90), scale=0.50, openness=0.96, variant="violet",
             rot=0.75, wire=(2.7, 5.3), surf=(4.7, 8.3), stamen=7.5, seed=200),
        # top-right white bloom
        dict(center=(0.32, 1.38, 0.00), dir=(0.12, 0.42, 0.90), scale=0.48, openness=0.98, variant="white_green",
             rot=0.40, wire=(3.0, 5.6), surf=(5.0, 8.6), stamen=7.8, seed=300),
        # lower-left large white bloom
        dict(center=(-0.66, 0.20, 0.35), dir=(-0.35, 0.25, 0.90), scale=0.47, openness=0.94, variant="white_pink",
             rot=-0.32, wire=(3.2, 5.8), surf=(5.2, 8.8), stamen=8.0, seed=400),
        # left side-view white bloom
        dict(center=(-0.98, 0.78, 0.00), dir=(-0.80, 0.40, 0.45), scale=0.42, openness=0.90, variant="white_pink",
             rot=0.10, wire=(3.4, 6.0), surf=(5.4, 9.0), stamen=8.2, seed=500),
        # right deep-orchid bloom
        dict(center=(0.70, 0.26, 0.25), dir=(0.62, 0.12, 0.78), scale=0.44, openness=0.92, variant="orchid_deep",
             rot=-0.42, wire=(3.6, 6.2), surf=(5.6, 9.2), stamen=8.4, seed=600),
        # far-right white-pink bloom, partly behind
        dict(center=(0.92, 0.72, -0.10), dir=(0.75, 0.40, 0.52), scale=0.40, openness=0.88, variant="white_pink",
             rot=0.55, wire=(3.8, 6.4), surf=(5.8, 9.4), stamen=8.6, seed=700),
    ]

    rng = np.random.RandomState(5)
    for i, f in enumerate(flowers_spec):
        center = np.array(f["center"], dtype=np.float32)
        dirv = np.array(f["dir"], dtype=np.float32)
        basis = build_basis_from_dir(dirv)
        w0, w1 = f["wire"]
        s0, s1 = f["surf"]
        # stem rising out of the goblet
        start = np.array([rng.uniform(-0.07, 0.07), -0.52, rng.uniform(-0.05, 0.05)], dtype=np.float32)
        end = center - (dirv / np.linalg.norm(dirv)) * 0.05
        elements.append(_stem_to_flower(start, end, dirv, 0.034, 0.026,
                                        wt0=w0 - 1.6, wt1=w0 + 0.9, st0=w0 - 1.0, st1=w0 + 1.4))
        for k in range(3):
            az = f["rot"] + k * (2.0 * math.pi / 3.0)
            dly = 0.12 * k
            elements.append(generate_lily_tepal(
                center=center, basis=basis, azimuth=az, is_inner=False, scale=f["scale"],
                openness=f["openness"], color_variant=f["variant"],
                wire_t0=w0 + dly, wire_t1=w1 + dly, surf_t0=s0 + dly, surf_t1=s1 + dly,
                rng_seed=f["seed"] + k))
        for k in range(3):
            az = f["rot"] + (k + 0.5) * (2.0 * math.pi / 3.0)
            dly = 0.18 + 0.12 * k
            elements.append(generate_lily_tepal(
                center=center, basis=basis, azimuth=az, is_inner=True, scale=f["scale"],
                openness=f["openness"], color_variant=f["variant"],
                wire_t0=w0 + dly, wire_t1=w1 + dly, surf_t0=s0 + dly, surf_t1=s1 + dly,
                rng_seed=f["seed"] + 10 + k))
        elements.extend(generate_stamens_and_pistil(
            center=center, basis=basis, scale=f["scale"],
            wire_t0=f["stamen"], wire_t1=f["stamen"] + 1.3,
            surf_t0=f["stamen"] + 0.4, surf_t1=f["stamen"] + 2.4))

    # 3. Buds on slender pedicels ---------------------------------------------
    buds_spec = [
        dict(center=(-0.66, 1.26, -0.06), dir=(-0.28, 0.95, 0.10), scale=0.46, variant="lilac", t=3.0),
        dict(center=(-0.40, 1.42, -0.14), dir=(-0.10, 0.98, 0.08), scale=0.42, variant="ivory", t=3.3),
        dict(center=(0.66, 0.96, 0.02), dir=(0.45, 0.88, 0.10), scale=0.42, variant="green", t=3.6),
    ]
    for b in buds_spec:
        center = np.array(b["center"], dtype=np.float32)
        dirv = np.array(b["dir"], dtype=np.float32)
        basis = build_basis_from_dir(dirv)
        start = np.array([rng.uniform(-0.05, 0.05), -0.52, rng.uniform(-0.05, 0.05)], dtype=np.float32)
        end = center - (dirv / np.linalg.norm(dirv)) * 0.01
        elements.append(_stem_to_flower(start, end, dirv, 0.030, 0.022,
                                        wt0=b["t"] - 1.5, wt1=b["t"] + 0.8, st0=b["t"] - 0.9, st1=b["t"] + 1.3))
        elements.append(generate_lily_bud(center, basis, b["scale"], b["variant"],
                                          wire_t0=b["t"] + 0.6, wire_t1=b["t"] + 3.0,
                                          surf_t0=b["t"] + 1.8, surf_t1=b["t"] + 4.4))

    # 4. Lavender sprigs --------------------------------------------------------
    sprigs_spec = [
        # (end point, control mid, florets, radius, variant, t)
        ((-1.14, 1.52, -0.10), (-0.6, 0.9, 0.0), 17, 0.100, "lavender", 4.0),
        ((0.92, 1.80, -0.15), (0.5, 1.2, 0.0), 16, 0.098, "periwinkle", 4.3),
        ((0.40, 1.00, 0.48), (0.2, 0.5, 0.2), 12, 0.092, "lavender", 4.6),
        ((-1.22, -0.12, 0.42), (-0.8, 0.2, 0.2), 14, 0.100, "orchid", 4.9),
        ((1.16, -0.14, 0.26), (0.8, 0.3, 0.1), 13, 0.096, "periwinkle", 5.2),
        ((-0.06, 1.28, 0.22), (-0.1, 0.6, 0.1), 9, 0.088, "lavender", 5.5),
    ]
    for i, (end, mid, n, r, var, tt) in enumerate(sprigs_spec):
        start = np.array([rng.uniform(-0.05, 0.05), -0.50, rng.uniform(-0.04, 0.04)], dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        mid = np.array(mid, dtype=np.float32)
        p1 = start + np.array([mid[0] * 0.20, 0.50, mid[2] * 0.10], dtype=np.float32)
        p2 = mid + (end - mid) * 0.25 + np.array([0.0, 0.18, 0.0], dtype=np.float32)
        elements.extend(generate_sprig(start, p1, p2, end, n, r, var, 900 + i,
                                       wt0=tt - 1.0, wt1=tt + 1.2, st0=tt - 0.4, st1=tt + 1.6, first_u=0.30))

    # 5. Glossy leaves near the rim and between the blooms ------------------------
    leaves_spec = [
        # origin, azimuth, length, width, arch, twist, delay
        ((-0.10, -0.12, 0.05), -1.55, 0.78, 0.115, 0.58, 0.18, 0.00),
        ((0.10, -0.10, 0.06), 1.50, 0.72, 0.110, 0.55, -0.16, 0.15),
        ((-0.06, -0.06, 0.04), -2.20, 0.62, 0.100, 0.40, 0.14, 0.30),
        ((0.06, -0.05, 0.03), 2.30, 0.60, 0.100, 0.38, -0.14, 0.45),
        ((-0.15, 0.20, 0.00), -0.90, 0.66, 0.100, 0.30, 0.12, 0.60),
        ((0.15, 0.25, 0.00), 0.95, 0.68, 0.105, 0.30, -0.12, 0.75),
        ((-0.05, 0.50, -0.05), -2.60, 0.60, 0.095, 0.24, 0.10, 0.90),
        ((0.08, 0.60, -0.05), 2.55, 0.60, 0.095, 0.24, -0.10, 1.05),
        ((-0.10, 0.05, 0.10), -1.15, 0.58, 0.095, 0.50, 0.12, 1.20),
        ((0.12, 0.00, 0.10), 1.20, 0.58, 0.095, 0.50, -0.12, 1.35),
        ((0.00, 0.75, -0.10), 0.20, 0.52, 0.085, 0.20, 0.10, 1.50),
    ]
    for org, az, ln, wd, arch, tw, dly in leaves_spec:
        elements.append(generate_lanceolate_leaf(
            origin=org, azimuth=az, length=ln, width=wd, arch_drop=arch, twist=tw,
            wire_t0=1.8 + dly, wire_t1=3.6 + dly, surf_t0=3.0 + dly, surf_t1=5.0 + dly))

    # 6. Fallen lily lying on the table + loose petals (5.8s -> 10.5s)
    fc = np.array((-0.80, -1.00, 0.60), dtype=np.float32)
    fdir = np.array((0.18, 0.80, 0.58), dtype=np.float32)
    fbasis = build_basis_from_dir(fdir)
    f_scale, f_rot = 0.42, 0.30
    for k in range(3):
        az = f_rot + k * (2.0 * math.pi / 3.0)
        elements.append(generate_lily_tepal(
            center=fc, basis=fbasis, azimuth=az, is_inner=False, scale=f_scale, openness=0.95,
            color_variant="fallen_pink", wire_t0=5.8 + 0.12 * k, wire_t1=8.0 + 0.12 * k,
            surf_t0=7.0 + 0.12 * k, surf_t1=10.0 + 0.12 * k, rng_seed=800 + k))
    for k in range(3):
        az = f_rot + (k + 0.5) * (2.0 * math.pi / 3.0)
        elements.append(generate_lily_tepal(
            center=fc, basis=fbasis, azimuth=az, is_inner=True, scale=f_scale, openness=0.95,
            color_variant="fallen_pink", wire_t0=6.0 + 0.12 * k, wire_t1=8.2 + 0.12 * k,
            surf_t0=7.2 + 0.12 * k, surf_t1=10.2 + 0.12 * k, rng_seed=810 + k))
    elements.extend(generate_stamens_and_pistil(
        center=fc, basis=fbasis, scale=f_scale, wire_t0=9.0, wire_t1=10.0, surf_t0=9.3, surf_t1=11.0))

    elements.append(generate_loose_petal((-0.62, -1.115, 0.88), 1.15, 0.36, 0.115, -0.04, "white_pink",
                                         wire_t0=8.4, wire_t1=9.6, surf_t0=9.0, surf_t1=10.8))
    elements.append(generate_loose_petal((0.78, -1.115, 0.45), -0.55, 0.32, 0.105, -0.04, "fallen_pink",
                                         wire_t0=8.8, wire_t1=10.0, surf_t0=9.4, surf_t1=11.2))
    return elements


# ============================================================================
# DREAMY GARDEN BACKDROP, FAIRY LIGHTS, WOODEN TABLE
# ============================================================================
def _fbm(h, w, seed, octaves=5):
    rng = np.random.RandomState(seed)
    out = np.zeros((h, w), dtype=np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gh, gw = 3 * 2 ** o, 2 * 2 ** o
        small = rng.rand(gh, gw).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    return out / tot


def _bgr(r, g, b):
    return (int(b), int(g), int(r))


def _premult_composite(img, layer_pm, alpha):
    """img: float32 BGR; layer_pm: premultiplied float32 BGR; alpha: float32 HxW."""
    return img * (1.0 - alpha[..., None]) + layer_pm


def create_dream_backdrop():
    H, W = HEIGHT, WIDTH
    rng = np.random.RandomState(11)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]

    d2 = ((xx - 0.5) / 0.68) ** 2 + ((yy - 0.38) / 0.52) ** 2
    glow = np.exp(-1.7 * d2)[..., None]
    edge = np.array([44, 30, 62], dtype=np.float32)
    mid = np.array([112, 88, 142], dtype=np.float32)
    core = np.array([204, 180, 226], dtype=np.float32)
    img = edge + (mid - edge) * glow + (core - mid) * (glow ** 3) * 0.85
    haze = np.exp(-(((xx - 0.5) / 0.38) ** 2 + ((yy - 0.14) / 0.14) ** 2))[..., None]
    img = img + haze * np.array([30, 14, 30], dtype=np.float32)
    noise = _fbm(H, W, 3)
    img = img * (0.86 + 0.26 * noise)[..., None]
    img = img[..., ::-1].copy()  # RGB -> BGR

    def blur_layer(draw_fn, sigma, strength=1.0):
        lay = np.zeros((H, W, 3), dtype=np.uint8)
        msk = np.zeros((H, W), dtype=np.uint8)
        draw_fn(lay, msk)
        lay_b = cv2.GaussianBlur(lay, (0, 0), sigma).astype(np.float32)
        a = cv2.GaussianBlur(msk, (0, 0), sigma).astype(np.float32) / 255.0 * strength
        return lay_b * strength, np.clip(a, 0, 1)

    # out-of-focus dark foliage framing the sides
    def draw_foliage(lay, msk):
        for _ in range(520):
            if rng.rand() < 0.5:
                cx = abs(rng.normal(0.0, 0.11)) * W
            else:
                cx = W - abs(rng.normal(0.0, 0.11)) * W
            cy = rng.uniform(0.0, 0.82) * H
            a_ax, b_ax = int(rng.uniform(26, 84)), int(rng.uniform(8, 22))
            ang = rng.uniform(0, 180)
            col = _bgr(rng.uniform(18, 52), rng.uniform(44, 100), rng.uniform(34, 76))
            cv2.ellipse(lay, (int(cx), int(cy)), (a_ax, b_ax), ang, 0, 360, col, -1, cv2.LINE_AA)
            cv2.ellipse(msk, (int(cx), int(cy)), (a_ax, b_ax), ang, 0, 360, 255, -1, cv2.LINE_AA)
    fol, fa = blur_layer(draw_foliage, 7.0, 0.95)
    img = _premult_composite(img, fol, fa)

    # blurred pink / lilac garden blossoms
    blossoms = [(0.12, 0.075), (0.02, 0.26), (0.10, 0.60), (0.03, 0.74), (0.93, 0.09),
                (0.97, 0.50), (0.90, 0.68), (0.84, 0.31), (0.15, 0.41), (0.08, 0.86)]

    def draw_blossoms(lay, msk):
        for (fx, fy) in blossoms:
            cx, cy = int(fx * W), int(fy * H)
            rad = rng.uniform(42, 66)
            base = rng.choice([0, 1, 2])
            cols = [(232, 150, 196), (214, 160, 214), (240, 178, 205)]
            col = _bgr(*cols[base])
            ph = rng.uniform(0, math.pi)
            for k in range(6):
                a = ph + k * math.pi / 3
                px = int(cx + math.cos(a) * rad * 0.55)
                py = int(cy + math.sin(a) * rad * 0.55)
                cv2.ellipse(lay, (px, py), (int(rad * 0.55), int(rad * 0.34)), math.degrees(a), 0, 360, col, -1, cv2.LINE_AA)
                cv2.ellipse(msk, (px, py), (int(rad * 0.55), int(rad * 0.34)), math.degrees(a), 0, 360, 255, -1, cv2.LINE_AA)
            cv2.circle(lay, (cx, cy), int(rad * 0.18), _bgr(240, 176, 96), -1, cv2.LINE_AA)
    bl, ba = blur_layer(draw_blossoms, 5.5, 0.88)
    img = _premult_composite(img, bl, ba)

    # hanging wisteria / lilac vines
    def draw_vines(lay, msk):
        for (vx, vy0, vy1, wd) in [(0.80, 0.0, 0.17, 38), (0.89, 0.0, 0.12, 30), (0.16, 0.0, 0.11, 32), (0.27, 0.02, 0.16, 40)]:
            n = 90
            for j in range(n):
                fr = j / n
                yv = (vy0 + (vy1 - vy0) * fr) * H
                xv = vx * W + rng.uniform(-wd, wd) * (1.0 - fr) ** 0.8
                r = int(rng.uniform(5, 10) * (1.0 - 0.45 * fr))
                col = _bgr(rng.uniform(165, 205), rng.uniform(130, 170), rng.uniform(210, 242))
                cv2.circle(lay, (int(xv), int(yv)), max(2, r), col, -1, cv2.LINE_AA)
                cv2.circle(msk, (int(xv), int(yv)), max(2, r), 255, -1, cv2.LINE_AA)
    vn, va = blur_layer(draw_vines, 3.5, 0.9)
    img = _premult_composite(img, vn, va)

    # large soft bokeh discs
    bk = np.zeros((H, W, 3), dtype=np.uint8)
    for _ in range(26):
        x = int(rng.uniform(0.0, 1.0) * W)
        y = int(rng.uniform(0.02, 0.78) * H)
        r = int(rng.uniform(26, 62))
        col = _bgr(rng.uniform(205, 255), rng.uniform(170, 225), rng.uniform(150, 235))
        cv2.circle(bk, (x, y), r, col, -1, cv2.LINE_AA)
        cv2.circle(bk, (x, y), r, tuple(int(c * 1.0) for c in col), 2, cv2.LINE_AA)
    bk = cv2.GaussianBlur(bk, (0, 0), 3.0)
    img = img + bk.astype(np.float32) * 0.13

    # vignette + darker top for title legibility
    vig = 1.0 - 0.34 * np.clip(((xx - 0.5) / 0.72) ** 2 + ((yy - 0.5) / 0.85) ** 2, 0, 1)
    top = 1.0 - 0.32 * (1.0 - smoothstep(0.0, 0.17, yy))
    img = img * (vig * top)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def create_fairy_lights():
    rng = np.random.RandomState(21)
    lights = []
    palette = [(255, 232, 160), (255, 220, 140), (255, 240, 200), (255, 205, 150), (250, 215, 235)]
    for i in range(110):
        if rng.rand() < 0.72:
            fx = (abs(rng.normal(0.0, 0.14)) if rng.rand() < 0.5 else 1.0 - abs(rng.normal(0.0, 0.14)))
            fy = rng.uniform(0.02, 0.80)
        else:
            fx = rng.uniform(0.2, 0.8)
            fy = rng.uniform(0.02, 0.34)
        fx = float(np.clip(fx, 0.01, 0.99))
        r = int(rng.choice([2, 3, 3, 4, 4, 5, 6, 8, 11]))
        c = palette[rng.randint(len(palette))]
        col = (c[2], c[1], c[0])
        lights.append((fx, fy, r, col, rng.uniform(0, 6.28), rng.uniform(0.8, 2.6)))
    return lights


FAIRY_LIGHTS = create_fairy_lights()


def draw_fairy_lights(canvas, t, shift_x, shift_y, fade):
    layer = np.zeros_like(canvas)
    for fx, fy, r, col, ph, sp in FAIRY_LIGHTS:
        a = (0.62 + 0.38 * math.sin(t * sp + ph)) * fade
        c = (int(col[0] * a), int(col[1] * a), int(col[2] * a))
        x = int(fx * WIDTH + shift_x * (0.6 + 0.4 * (r / 11.0)))
        y = int(fy * HEIGHT + shift_y)
        cv2.circle(layer, (x, y), r, c, -1, cv2.LINE_AA)
        if r >= 6:
            cv2.circle(layer, (x, y), r + 3, (int(c[0] * 0.35), int(c[1] * 0.35), int(c[2] * 0.35)), 2, cv2.LINE_AA)
    g1 = cv2.GaussianBlur(layer, (0, 0), 5.0)
    g2 = cv2.GaussianBlur(layer, (0, 0), 17.0)
    out = cv2.addWeighted(canvas, 1.0, layer, 0.95, 0.0)
    out = cv2.addWeighted(out, 1.0, g1, 1.25, 0.0)
    out = cv2.addWeighted(out, 1.0, g2, 1.05, 0.0)
    return out


# ---- wooden table with purple satin, dew crystals and bokeh reflections -----
TABLE_W = int(WIDTH * 1.7)
TABLE_H = 1500
TABLE_FOOT = (TABLE_W // 2, 300)
TABLE_BACK = 140


def create_table_texture():
    rng = np.random.RandomState(7)
    Wt, Ht = TABLE_W, TABLE_H
    cx = Wt / 2.0
    yy, xx = np.mgrid[0:Ht, 0:Wt].astype(np.float32)

    img = np.zeros((Ht, Wt, 3), dtype=np.float32)  # RGB
    g1 = cv2.resize(rng.rand(Ht // 3, max(2, Wt // 70)).astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_CUBIC)
    g2 = cv2.resize(rng.rand(Ht, max(2, Wt // 14)).astype(np.float32), (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    grain = 0.6 * g1 + 0.4 * g2

    y, h = float(TABLE_BACK), 46.0
    while y < Ht:
        y0, y1 = int(y), int(min(Ht, y + h))
        tone = 0.82 + 0.34 * rng.rand()
        img[y0:y1] = np.array([92, 72, 74], dtype=np.float32) * tone + np.array([6, 0, 10], dtype=np.float32)
        img[y0:min(Ht, y0 + 2)] *= 0.30
        for _ in range(2):
            xj = rng.randint(0, Wt - 3)
            img[y0:y1, xj:xj + 2] *= 0.35
        y += h
        h *= 1.10
    img *= (0.66 + 0.62 * grain)[..., None]

    light = np.exp(-(((xx - cx) / 760.0) ** 2))
    back_dark = 0.62 + 0.38 * smoothstep(TABLE_BACK, TABLE_BACK + 520, yy)
    img *= (0.50 + 0.62 * light * back_dark)[..., None]
    img += (np.exp(-(((xx - cx) / 420.0) ** 2 + ((yy - 330.0) / 150.0) ** 2)))[..., None] * np.array([34, 20, 30], dtype=np.float32)

    # purple satin: a wavy band at the back, and a drape sweeping in from the right
    ang = 0.62
    s = xx * math.cos(ang) + yy * math.sin(ang)
    folds = 0.5 + 0.5 * np.sin(s / 34.0 + 1.6 * np.sin(s / 118.0) + 0.9 * np.sin(xx / 150.0))
    sheen = np.clip(folds - 0.55, 0, 1) ** 1.6
    satin = (np.array([58, 44, 78], dtype=np.float32) * (0.55 + 0.7 * folds)[..., None]
             + np.array([140, 108, 168], dtype=np.float32) * (sheen * 1.3)[..., None])
    satin *= (0.60 + 0.55 * np.exp(-(((xx - cx) / 900.0) ** 2)))[..., None]

    back_band = (yy > TABLE_BACK - 16 + 10 * np.sin(xx / 77.0)) & (yy < TABLE_BACK + 50 + 22 * np.sin(xx / 93.0 + 1.0))
    x_b = cx + 250 - 0.55 * (yy - TABLE_BACK) + 28 * np.sin(yy / 61.0)
    drape = (xx > x_b) & (yy > TABLE_BACK + 10)
    fab = (back_band | drape).astype(np.float32)
    fab = cv2.GaussianBlur(fab, (0, 0), 3.0)[..., None]
    img = img * (1.0 - fab) + satin * fab

    # alpha: fade in the back edge of the table
    alpha = smoothstep(TABLE_BACK - 18, TABLE_BACK + 6, yy + 10 * np.sin(xx / 77.0) * (yy < TABLE_BACK + 60))
    alpha = np.clip(alpha, 0, 1).astype(np.float32)

    bgr = np.clip(img[..., ::-1], 0, 255).astype(np.uint8)

    # contact shadow under the goblet foot
    sh = np.zeros((Ht, Wt), dtype=np.uint8)
    cv2.ellipse(sh, (TABLE_FOOT[0], TABLE_FOOT[1] + 6), (150, 36), 0, 0, 360, 255, -1, cv2.LINE_AA)
    sh = cv2.GaussianBlur(sh, (0, 0), 16).astype(np.float32) / 255.0
    bgr = (bgr.astype(np.float32) * (1.0 - 0.62 * sh[..., None])).astype(np.uint8)
    # soft shadow under the fallen lily and loose petals
    for (sx_, sy_, ax, ay) in [(TABLE_FOOT[0] - 235, TABLE_FOOT[1] + 40, 120, 36),
                               (TABLE_FOOT[0] - 205, TABLE_FOOT[1] + 118, 70, 18),
                               (TABLE_FOOT[0] + 245, TABLE_FOOT[1] + 6, 62, 16)]:
        sh2 = np.zeros((Ht, Wt), dtype=np.uint8)
        cv2.ellipse(sh2, (sx_, sy_), (ax, ay), 0, 0, 360, 255, -1, cv2.LINE_AA)
        sh2 = cv2.GaussianBlur(sh2, (0, 0), 10).astype(np.float32) / 255.0
        bgr = (bgr.astype(np.float32) * (1.0 - 0.26 * sh2[..., None])).astype(np.uint8)

    # warm bokeh reflections, dew drops, crystals
    glow = np.zeros((Ht, Wt, 3), dtype=np.uint8)
    for _ in range(46):
        x = int(rng.uniform(0.18, 0.82) * Wt)
        y = int(rng.uniform(TABLE_BACK + 20, 760))
        r = int(rng.choice([2, 3, 3, 4, 5, 7]))
        c = (150 + rng.randint(0, 60), 205 + rng.randint(0, 40), 255)
        cv2.circle(glow, (x, y), r, c, -1, cv2.LINE_AA)
    gb = cv2.GaussianBlur(glow, (0, 0), 4.0)
    bgr = np.clip(bgr.astype(np.float32) + glow.astype(np.float32) * 0.7 + gb.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)

    for _ in range(95):
        x = int(rng.uniform(0.12, 0.88) * Wt)
        y = int(rng.uniform(TABLE_BACK + 30, 900))
        r = int(rng.choice([1, 1, 2, 2, 3]))
        cv2.circle(bgr, (x, y), r, (255, 245, 235), -1, cv2.LINE_AA)
        if r >= 2:
            cv2.line(bgr, (x - 5, y), (x + 5, y), (255, 235, 225), 1, cv2.LINE_AA)
            cv2.line(bgr, (x, y - 5), (x, y + 5), (255, 235, 225), 1, cv2.LINE_AA)

    gem_pos = [(-390, 150), (-330, 290), (-120, 305), (-60, 360), (150, 330), (310, 300), (420, 170),
               (260, 400), (-250, 440), (-30, 470), (120, 520), (340, 470), (-430, 330)]
    for gx, gy in gem_pos:
        cxg, cyg = TABLE_FOOT[0] + gx, TABLE_FOOT[1] + gy
        rg = int(rng.uniform(5, 11))
        pts = np.array([[cxg + rg * math.cos(a), cyg + rg * 0.8 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 7)[:-1]],
                       dtype=np.int32)
        cv2.fillConvexPoly(bgr, pts, (236, 214, 200), lineType=cv2.LINE_AA)
        cv2.polylines(bgr, [pts], True, (255, 250, 245), 1, cv2.LINE_AA)
        cv2.line(bgr, (cxg - rg // 2, cyg - rg // 3), (cxg + rg // 3, cyg + rg // 3), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(bgr, (cxg - rg // 3, cyg - rg // 3), max(1, rg // 4), (255, 255, 255), -1, cv2.LINE_AA)
    for _ in range(26):
        x = int(TABLE_FOOT[0] + rng.uniform(-520, 520))
        y = int(TABLE_FOOT[1] + rng.uniform(80, 520))
        r = int(rng.uniform(2, 5))
        cv2.circle(bgr, (x, y), r + 1, (96, 70, 56), -1, cv2.LINE_AA)
        cv2.circle(bgr, (x, y), r, (230, 200, 175), -1, cv2.LINE_AA)
        cv2.circle(bgr, (x - 1, y - 1), max(1, r // 2), (255, 255, 255), -1, cv2.LINE_AA)

    pm = (bgr.astype(np.float32) * alpha[..., None]).astype(np.uint8)
    return np.dstack([pm, (alpha * 255).astype(np.uint8)])


# Background image and table textures removed per request (clean void with sparkles/fairy lights)
DREAM_BG = None
TABLE_PM = None


# ============================================================================
# CAMERA, PROJECTION AND LIGHTING
# ============================================================================
def camera_at(p_all):
    azimuth = -0.12 + 0.26 * smootherstep(0.0, 0.86, p_all) + 0.04 * math.sin(
        max(0.0, (p_all - 0.72) / 0.28) * math.pi)
    elevation = 0.17 + 0.05 * smootherstep(0.10, 0.70, p_all) - 0.015 * smootherstep(0.80, 1.0, p_all)
    cam_rot = rot_x(elevation) @ rot_y(azimuth)
    y_target = -0.40 + 0.70 * smootherstep(0.0, 0.55, p_all)
    cam_target = np.array([0.0, y_target, 0.10], dtype=np.float32)
    cam_dist = 5.30 - (5.30 - FINAL_CAM_DIST) * smootherstep(0.0, 0.90, p_all)
    return cam_rot, cam_target, cam_dist, azimuth


def project_points(P_world, cam_rot, cam_target, cam_dist, fov_scale=FOV_SCALE):
    P_rel = (P_world - cam_target) @ cam_rot.T
    z_depth = cam_dist - P_rel[..., 2]
    z_safe = np.maximum(z_depth, 0.2)
    sx = (WIDTH * 0.5) + (P_rel[..., 0] / z_safe) * fov_scale
    sy = (HEIGHT * SCREEN_CY) - (P_rel[..., 1] / z_safe) * fov_scale
    return sx, sy, z_depth, P_rel


_cr, _ct, _cd, _ = camera_at(1.0)
_fx, _fy, _fz, _ = project_points(VASE_FOOT_WORLD[None, :], _cr, _ct, _cd)
FINAL_FOOT_SY = float(_fy[0])


def compute_lighting(normals_cam, base_rgb, sss_strength, spec_strength, shininess):
    """Studio botanical shader: key + fill + rim, subsurface scattering, waxy specular."""
    L_key = normalize_v3(np.array([-0.42, 0.62, 0.66], dtype=np.float32))
    L_fill = normalize_v3(np.array([0.65, -0.20, 0.73], dtype=np.float32))
    V_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    nz = normals_cam[..., 2:3]
    N = np.where(nz < 0.0, -normals_cam, normals_cam)

    ndotl_key = np.sum(N * L_key, axis=-1, keepdims=True)
    diff_key = np.clip((ndotl_key + 0.25) / 1.25, 0.0, 1.0)
    ndotl_fill = np.sum(N * L_fill, axis=-1, keepdims=True)
    diff_fill = np.clip((ndotl_fill + 0.40) / 1.40, 0.0, 1.0)

    ndotv = np.clip(np.sum(N * V_dir, axis=-1, keepdims=True), 0.0, 1.0)
    fresnel_rim = (1.0 - ndotv) ** 2.2
    back_trans = np.clip(-ndotl_key * 0.6 + 0.45, 0.0, 1.0)
    sss = sss_strength * (0.55 * fresnel_rim + 0.45 * back_trans)

    H_key = normalize_v3(L_key + V_dir)
    ndoth = np.clip(np.sum(N * H_key, axis=-1, keepdims=True), 0.0, 1.0)
    spec = spec_strength * (ndoth ** shininess)

    key_col = np.array([1.04, 1.01, 1.0], dtype=np.float32)
    fill_col = np.array([0.80, 0.84, 1.0], dtype=np.float32)
    sss_col = np.array([1.05, 0.84, 0.95], dtype=np.float32)

    lit_rgb = (
        base_rgb * (0.26 + 0.72 * diff_key * key_col + 0.22 * diff_fill * fill_col)
        + base_rgb * sss * sss_col
        + spec * np.array([1.0, 0.98, 0.96], dtype=np.float32)
    )
    lit_rgb = np.clip(lit_rgb, 0.0, 1.35)
    lit_rgb = lit_rgb / (1.0 + 0.18 * lit_rgb) * 1.14
    return np.clip(lit_rgb, 0.0, 1.0)


def compute_glass(normals_cam, y_world):
    """Crystal glass: fresnel rim, two window highlights and a faint water tint inside the cup."""
    nz = normals_cam[..., 2:3]
    N = np.where(nz < 0.0, -normals_cam, normals_cam)
    V_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    L1 = normalize_v3(np.array([-0.45, 0.55, 0.70], dtype=np.float32))
    L2 = normalize_v3(np.array([0.72, 0.28, 0.64], dtype=np.float32))
    ndotv = np.clip(np.sum(N * V_dir, axis=-1, keepdims=True), 0.0, 1.0)
    fres = (1.0 - ndotv) ** 2.0
    H1 = normalize_v3(L1 + V_dir)
    H2 = normalize_v3(L2 + V_dir)
    s1 = np.clip(np.sum(N * H1, axis=-1, keepdims=True), 0.0, 1.0) ** 70.0
    s2 = np.clip(np.sum(N * H2, axis=-1, keepdims=True), 0.0, 1.0) ** 45.0
    spec = 1.15 * s1 + 0.60 * s2
    water = ((y_world < -0.27) & (y_world > -0.62))[..., None].astype(np.float32)
    diff = 0.55 + 0.35 * np.clip(np.sum(N * L1, axis=-1, keepdims=True), 0.0, 1.0)
    tint = np.array([0.88, 0.93, 1.0], dtype=np.float32) * (1.0 - 0.45 * water) \
        + np.array([0.70, 0.90, 0.86], dtype=np.float32) * (0.45 * water)
    rgb = np.clip(tint * diff * 0.95 + spec, 0.0, 1.0)
    alpha = np.clip(0.15 + 0.78 * fres ** 0.85 + 0.95 * spec + 0.08 * water, 0.0, 0.93)
    return rgb, alpha[..., 0]


# ============================================================================
# PER-ELEMENT GATHER (quads, papillae, wire curves, wavefront particles)
# ============================================================================
SCENE_ELEMENTS = None


def init_worker():
    global SCENE_ELEMENTS
    SCENE_ELEMENTS = build_lily_bouquet_scene()


def _expand_quads(sx, sy):
    q_x = np.stack([sx[:-1, :-1], sx[1:, :-1], sx[1:, 1:], sx[:-1, 1:]], axis=-1)
    q_y = np.stack([sy[:-1, :-1], sy[1:, :-1], sy[1:, 1:], sy[:-1, 1:]], axis=-1)
    cx_q = np.mean(q_x, axis=-1, keepdims=True)
    cy_q = np.mean(q_y, axis=-1, keepdims=True)
    dx_q = q_x - cx_q
    dy_q = q_y - cy_q
    d_len = np.maximum(np.hypot(dx_q, dy_q), 1e-4)
    # 0.55 px expansion hides anti-aliasing seams between neighbouring quads
    return np.stack([q_x + (dx_q / d_len) * 0.55, q_y + (dy_q / d_len) * 0.55], axis=-1)


def gather_element(el, t, cam_rot, cam_target, cam_dist, global_wire_alpha, out):
    w_prog = smootherstep(el.wire_t0, el.wire_t1, t)
    s_prog = smootherstep(el.surf_t0, el.surf_t1, t)

    if el.is_petal:
        bloom_t = 0.28 + 0.72 * smootherstep(el.surf_t0, min(DURATION, el.surf_t1 + 1.6), t)
    else:
        bloom_t = 1.0

    P, p00, p10, p11, p01, normals = el.get_positions_and_normals(bloom_t)
    normals_cam = normals @ cam_rot.T
    sx, sy, z_depth, _ = project_points(P, cam_rot, cam_target, cam_dist)

    # ---- A. surface quads ---------------------------------------------------
    if s_prog > 0.001:
        eff_u = el.eff_u
        mask = eff_u <= (s_prog * 1.03)
        if el.quad_valid is not None:
            mask = mask & el.quad_valid

        if np.any(mask):
            dist_edge = (s_prog * 1.03) - eff_u
            rim_glow = np.exp(-((dist_edge / 0.055) ** 2)) * (dist_edge >= 0.0)
            is_glass = el.layer == "glass"

            if is_glass:
                y_q = 0.25 * (P[:-1, :-1, 1] + P[1:, :-1, 1] + P[1:, 1:, 1] + P[:-1, 1:, 1])
                lit_rgb, alpha_q = compute_glass(normals_cam, y_q)
                if s_prog < 0.995:
                    lit_rgb = np.clip(lit_rgb + 0.45 * rim_glow[..., None] * np.array([0.9, 0.9, 1.0], np.float32), 0, 1)
                    alpha_q = np.clip(alpha_q + 0.55 * rim_glow, 0.0, 0.92)
                z_ring = z_depth.mean(axis=1, keepdims=True)
                z_mid = 0.5 * (z_ring[:-1] + z_ring[1:])
                z_quad_full = 0.25 * (z_depth[:-1, :-1] + z_depth[1:, :-1] + z_depth[1:, 1:] + z_depth[:-1, 1:])
                back_q = (z_quad_full > z_mid)
                alpha_q = np.where(back_q, alpha_q * 0.85, alpha_q)
            else:
                lit_rgb = compute_lighting(normals_cam, el.c_q, el.sss_strength, el.spec_strength, el.shininess)
                if s_prog < 0.995:
                    glow_col = np.array([1.0, 0.86, 0.55], dtype=np.float32)
                    lit_rgb = np.clip(lit_rgb + 0.55 * rim_glow[..., None] * glow_col, 0.0, 1.0)

            pts_sub = _expand_quads(sx, sy)[mask]
            pts_int = np.rint(pts_sub * SUBPIX).astype(np.int32)
            z_q = 0.25 * (z_depth[:-1, :-1] + z_depth[1:, :-1] + z_depth[1:, 1:] + z_depth[:-1, 1:])[mask]
            bgr_q = np.rint(lit_rgb[mask][..., ::-1] * 255.0).astype(np.uint8)

            if is_glass:
                a_q = np.rint(alpha_q[mask] * 255.0).astype(np.uint8)
                out["glass"].append((pts_int, z_q, bgr_q, a_q, back_q[mask]))
            else:
                out[el.layer].append((pts_int, z_q, bgr_q))

            if 0.03 < s_prog < 0.98 and not is_glass:
                edge_mask = (dist_edge > 0.0) & (dist_edge < 0.04) & mask
                ey, ex = np.where(edge_mask)
                for idx_p in range(0, len(ey), 7):
                    iy, ix = ey[idx_p], ex[idx_p]
                    out["waves"].append((float(sx[iy, ix]), float(sy[iy, ix]), float(z_depth[iy, ix]) - 0.03,
                                         (165, 225, 255), 2.4))

            if el.papillae:
                for u_i, v_i, rad_px, core_bgr, rim_bgr, p_eff_u in el.papillae:
                    if p_eff_u <= (s_prog * 0.98):
                        n_z = float(normals_cam[min(u_i, normals_cam.shape[0] - 1),
                                                min(v_i, normals_cam.shape[1] - 1), 2])
                        if abs(n_z) > 0.14:
                            px_i = int(round(float(sx[u_i, v_i])))
                            py_i = int(round(float(sy[u_i, v_i])))
                            dx_u = float(sx[min(u_i + 1, sx.shape[0] - 1), v_i] - sx[max(u_i - 1, 0), v_i])
                            dy_u = float(sy[min(u_i + 1, sy.shape[0] - 1), v_i] - sy[max(u_i - 1, 0), v_i])
                            ang_deg = math.degrees(math.atan2(dy_u, dx_u))
                            r_maj = max(2, int(round(rad_px * 1.15)))
                            r_min = max(1, int(round(rad_px * 0.75)))
                            pz_d = float(z_depth[u_i, v_i]) - 0.032
                            out["pap"].append((pz_d, px_i, py_i, r_maj, r_min, ang_deg, core_bgr, rim_bgr))

    # ---- B. parametric wireframe curves ----------------------------------------
    wire_vis = global_wire_alpha * (1.0 - 0.82 * smootherstep(0.60, 1.0, s_prog))
    if w_prog > 0.01 and wire_vis > 0.04:
        Nu, Nv = P.shape[:2]
        max_u_idx = max(2, int(round(w_prog * Nu)))
        wc_bgr = tuple(int(round(c * 255.0 * (0.35 + 0.65 * wire_vis))) for c in el.wire_color[::-1])

        v_indices = list(range(0, Nv, el.wire_stride_v))
        if (Nv - 1) not in v_indices:
            v_indices.append(Nv - 1)
        for j in v_indices:
            pts_2d = np.stack([sx[:max_u_idx, j], sy[:max_u_idx, j]], axis=-1)
            pts_sub = np.rint(pts_2d * SUBPIX).astype(np.int32)
            z_m = float(np.mean(z_depth[:max_u_idx, j])) - 0.015
            out["wires"].append((pts_sub, z_m, wc_bgr, 1))
            if w_prog < 0.98 and (j % (el.wire_stride_v * 2) == 0):
                tip_u = max_u_idx - 1
                out["waves"].append((float(sx[tip_u, j]), float(sy[tip_u, j]), float(z_depth[tip_u, j]) - 0.04,
                                     (190, 240, 255), 3.0))
        if el.wire_cross:
            for i in range(0, max_u_idx, el.wire_stride_u):
                pts_2d = np.stack([sx[i, :], sy[i, :]], axis=-1)
                pts_sub = np.rint(pts_2d * SUBPIX).astype(np.int32)
                z_m = float(np.mean(z_depth[i, :])) - 0.015
                out["wires"].append((pts_sub, z_m, wc_bgr, 1))


# ============================================================================
# RASTERISATION HELPERS
# ============================================================================
def _draw_papilla(canvas, p):
    _, px_i, py_i, r_maj, r_min, ang_deg, core_bgr, rim_bgr = p
    cv2.ellipse(canvas, (px_i, py_i), (r_maj + 1, r_min + 1), ang_deg, 0, 360, rim_bgr, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (px_i, py_i), (r_maj, r_min), ang_deg, 0, 360, core_bgr, -1, lineType=cv2.LINE_AA)


def raster_opaque(canvas, entries, papillae):
    """Painter's algorithm: depth-sorted quads interleaved with raised botanical papillae."""
    if not entries:
        return
    all_pts = np.concatenate([e[0] for e in entries], axis=0)
    all_z = np.concatenate([e[1] for e in entries], axis=0)
    all_bgr = np.concatenate([e[2] for e in entries], axis=0).tolist()
    order = np.argsort(all_z)[::-1]
    papillae.sort(key=lambda x: x[0], reverse=True)
    n_pap, ptr = len(papillae), 0
    for idx in order:
        qz = float(all_z[idx])
        while ptr < n_pap and papillae[ptr][0] >= qz:
            _draw_papilla(canvas, papillae[ptr])
            ptr += 1
        cv2.fillConvexPoly(canvas, all_pts[idx], all_bgr[idx], lineType=cv2.LINE_AA, shift=SHIFT)
    while ptr < n_pap:
        _draw_papilla(canvas, papillae[ptr])
        ptr += 1


def _composite_layer(canvas, rgb, a):
    x, y, w, h = cv2.boundingRect(a)
    if w == 0 or h == 0:
        return
    roi = canvas[y:y + h, x:x + w].astype(np.float32)
    al = a[y:y + h, x:x + w, None].astype(np.float32) / 255.0
    roi = roi * (1.0 - al) + rgb[y:y + h, x:x + w].astype(np.float32) * al
    canvas[y:y + h, x:x + w] = np.clip(roi, 0, 255).astype(np.uint8)


def raster_glass(canvas, entries):
    """Alpha-blended crystal: far wall first, then near wall."""
    if not entries:
        return
    pts = np.concatenate([e[0] for e in entries], axis=0)
    zs = np.concatenate([e[1] for e in entries], axis=0)
    bgr = np.concatenate([e[2] for e in entries], axis=0).tolist()
    al = np.concatenate([e[3] for e in entries], axis=0).tolist()
    back = np.concatenate([e[4] for e in entries], axis=0)
    for flag in (True, False):
        idxs = np.nonzero(back == flag)[0]
        if len(idxs) == 0:
            continue
        idxs = idxs[np.argsort(zs[idxs])[::-1]]
        rgb_l = np.zeros_like(canvas)
        a_l = np.zeros(canvas.shape[:2], dtype=np.uint8)
        for i in idxs:
            cv2.fillConvexPoly(rgb_l, pts[i], bgr[i], lineType=cv2.LINE_AA, shift=SHIFT)
            cv2.fillConvexPoly(a_l, pts[i], int(al[i]), lineType=cv2.LINE_AA, shift=SHIFT)
        _composite_layer(canvas, rgb_l, a_l)


def draw_glass_rims(canvas, t, cam_rot, cam_target, cam_dist):
    """Crisp highlights on the goblet rim and foot edge once the crystal has formed."""
    for el in SCENE_ELEMENTS:
        if el.layer != "glass":
            continue
        s_prog = float(smootherstep(el.surf_t0, el.surf_t1, t))
        sx, sy, _, _ = project_points(el.P, cam_rot, cam_target, cam_dist)
        layer = np.zeros_like(canvas)
        for ring, thr, col, th in ((el.P.shape[0] - 1, 0.93, (255, 250, 250), 2),
                                   (el.P.shape[0] - 3, 0.90, (215, 205, 190), 1),
                                   (int(el.P.shape[0] * 0.12), 0.50, (150, 145, 140), 1)):
            if s_prog >= thr:
                pts = np.rint(np.stack([sx[ring], sy[ring]], axis=-1) * SUBPIX).astype(np.int32)
                cv2.polylines(layer, [pts], True, col, th, cv2.LINE_AA, SHIFT)
        canvas[:] = cv2.addWeighted(canvas, 1.0, cv2.GaussianBlur(layer, (0, 0), 1.2), 0.9, 0.0)
        canvas[:] = cv2.addWeighted(canvas, 1.0, layer, 0.45, 0.0)


def compose_background(t, cam_rot, cam_target, cam_dist, azimuth):
    """Clean dark background + floating fairy lights & sparkles (no background image or table)."""
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    fsx, fsy, fz, _ = project_points(VASE_FOOT_WORLD[None, :], cam_rot, cam_target, cam_dist)
    fsx, fsy, fz = float(fsx[0]), float(fsy[0]), float(fz[0])

    bg_fade = 0.18 + 0.82 * float(smootherstep(0.0, 2.8, t))
    dx = -azimuth * 55.0
    dy = 0.14 * (fsy - FINAL_FOOT_SY)

    # Draw fairy lights & sparkles against the clean background
    canvas = draw_fairy_lights(canvas, t, dx * 1.6, dy * 1.2, bg_fade)
    return canvas


# ============================================================================
# SINGLE-FRAME RENDERER
# ============================================================================
def render_frame(frame_idx, total_frames=TOTAL_FRAMES):
    """Render frame `frame_idx` at 1080x1920 and return a BGR uint8 image."""
    global SCENE_ELEMENTS
    if SCENE_ELEMENTS is None:
        SCENE_ELEMENTS = build_lily_bouquet_scene()

    t = (frame_idx / max(1, total_frames - 1)) * DURATION
    p_all = frame_idx / max(1, total_frames - 1)
    cam_rot, cam_target, cam_dist, azimuth = camera_at(p_all)

    canvas = compose_background(t, cam_rot, cam_target, cam_dist, azimuth)

    out = {"main": [], "glass": [], "late": [], "pap": [], "wires": [], "waves": []}
    global_wire_alpha = 1.0 - smootherstep(9.6, 11.4, t)
    for el in SCENE_ELEMENTS:
        if t < el.wire_t0:
            continue
        gather_element(el, t, cam_rot, cam_target, cam_dist, global_wire_alpha, out)

    raster_opaque(canvas, out["main"], out["pap"])
    raster_glass(canvas, out["glass"])
    draw_glass_rims(canvas, t, cam_rot, cam_target, cam_dist)
    raster_opaque(canvas, out["late"], [])

    # ---- parametric wireframes with additive glow ---------------------------------
    if out["wires"]:
        wire_layer = np.zeros_like(canvas)
        out["wires"].sort(key=lambda item: item[1], reverse=True)
        for pts_sub, _, wc_bgr, thick in out["wires"]:
            cv2.polylines(wire_layer, [pts_sub], isClosed=False, color=wc_bgr,
                          thickness=thick, lineType=cv2.LINE_AA, shift=SHIFT)
        canvas = cv2.addWeighted(canvas, 1.0, wire_layer, 0.58, 0.0)

    # ---- floating golden pollen & construction wavefront particles -------------------
    particle_layer = np.zeros_like(canvas)
    rng_p = np.random.RandomState(42)
    n_ambient = 110
    amb_pos = rng_p.uniform([-1.6, -1.2, -0.8], [1.6, 2.0, 0.9], size=(n_ambient, 3)).astype(np.float32)
    amb_phase = rng_p.uniform(0.0, 2.0 * math.pi, size=n_ambient)
    amb_pos[:, 1] += 0.16 * t + 0.08 * np.sin(t * 1.5 + amb_phase)
    amb_pos[:, 1] = ((amb_pos[:, 1] + 1.2) % 3.2) - 1.2
    amb_pos[:, 0] += 0.08 * np.cos(t * 1.1 + amb_phase)
    asx, asy, azd, _ = project_points(amb_pos, cam_rot, cam_target, cam_dist)
    palette = ((140, 210, 255), (235, 235, 255), (255, 205, 235), (170, 225, 255), (255, 225, 245), (150, 200, 255))
    for i in range(n_ambient):
        px, py = int(round(asx[i])), int(round(asy[i]))
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            twinkle = 0.4 + 0.6 * math.sin(t * 2.8 + amb_phase[i])
            rad = int(round((1.2 + 1.8 * (i % 3)) * (4.5 / max(2.5, azd[i]))))
            base_col = palette[i % len(palette)]
            col = tuple(int(channel * max(0.0, twinkle)) for channel in base_col)
            cv2.circle(particle_layer, (px, py), max(1, rad), col, -1, lineType=cv2.LINE_AA)
    for px, py, _, bgr, rad in out["waves"][:260]:
        ix, iy = int(round(px)), int(round(py))
        if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
            cv2.circle(particle_layer, (ix, iy), max(1, int(round(rad))), bgr, -1, lineType=cv2.LINE_AA)
    particle_blur = cv2.GaussianBlur(particle_layer, (0, 0), sigmaX=3.2)
    canvas = cv2.addWeighted(canvas, 1.0, particle_blur, 0.85, 0.0)
    canvas = cv2.addWeighted(canvas, 1.0, particle_layer, 0.55, 0.0)

    # ---- soft botanical bloom polish ---------------------------------------------------
    bright = np.maximum(canvas.astype(np.float32) - 195.0, 0.0).astype(np.uint8)
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=18.0)
    canvas = cv2.addWeighted(canvas, 1.0, bloom, 0.50, 0.0)

    draw_botanical_hud(canvas, t)
    return canvas


def draw_botanical_hud(canvas, t):
    """Animated title treatment; BROCODES is the only creator credit."""
    title_alpha = smootherstep(0.35, 1.55, t) * (1.0 - 0.16 * smootherstep(11.8, 12.8, t))
    if title_alpha <= 0.01:
        return
    overlay = canvas.copy()
    center_x = WIDTH // 2

    logo_y = 55
    logo_glow = np.zeros_like(canvas)
    petals = [((center_x, logo_y), (center_x - 13, logo_y - 11)),
              ((center_x, logo_y), (center_x, logo_y - 17)),
              ((center_x, logo_y), (center_x + 13, logo_y - 11))]
    for a, b in petals:
        cv2.line(logo_glow, a, b, (220, 70, 255), 3, cv2.LINE_AA)
    cv2.line(logo_glow, (center_x, logo_y), (center_x, logo_y + 7), (200, 100, 245), 2, cv2.LINE_AA)
    logo_glow = cv2.GaussianBlur(logo_glow, (0, 0), 7)
    overlay = cv2.addWeighted(overlay, 1.0, logo_glow, 0.55 * title_alpha, 0.0)
    for a, b in petals:
        cv2.line(overlay, a, b, (225, 100, 255), 2, cv2.LINE_AA)
    cv2.line(overlay, (center_x, logo_y), (center_x, logo_y + 7), (235, 180, 255), 1, cv2.LINE_AA)

    brand = "BROCODES"
    font = cv2.FONT_HERSHEY_SIMPLEX
    brand_scale, brand_thick = 0.43, 1
    (bw, bh), _ = cv2.getTextSize(brand, font, brand_scale, brand_thick)
    brand_y = 91
    bx = center_x - bw // 2
    rule_col = (135, 95, 150)
    cv2.line(overlay, (center_x - bw // 2 - 100, brand_y - 10), (center_x - bw // 2 - 18, brand_y - 10), rule_col, 1, cv2.LINE_AA)
    cv2.line(overlay, (center_x + bw // 2 + 18, brand_y - 10), (center_x + bw // 2 + 100, brand_y - 10), rule_col, 1, cv2.LINE_AA)
    cv2.putText(overlay, brand, (bx, brand_y), font, brand_scale, (238, 220, 246), brand_thick, cv2.LINE_AA)

    reveal = smootherstep(0.8, 2.15, t)
    title = "Lily 3D Animation"
    title_font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    title_scale = 1.18
    title_thick = 2
    (tw, th), _ = cv2.getTextSize(title, title_font, title_scale, title_thick)
    title_y = int(151 + 20 * (1.0 - reveal))
    tx = center_x - tw // 2
    glow = np.zeros_like(canvas)
    cv2.putText(glow, title, (tx, title_y), title_font, title_scale, (220, 35, 245), 5, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    overlay = cv2.addWeighted(overlay, 1.0, glow, 0.72 * title_alpha * reveal, 0.0)
    cv2.putText(overlay, title, (tx, title_y), title_font, title_scale, (230, 105, 250), title_thick, cv2.LINE_AA)

    subtitle = "B E A U T Y   I N   E V E R Y   F R A M E"
    sub_font, sub_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.31
    (sw, sh), _ = cv2.getTextSize(subtitle, sub_font, sub_scale, 1)
    sy = title_y + 31
    cv2.putText(overlay, subtitle, (center_x - sw // 2, sy), sub_font, sub_scale, (218, 198, 224), 1, cv2.LINE_AA)
    divider_y = sy + 24
    cv2.line(overlay, (center_x - 95, divider_y), (center_x - 15, divider_y), (150, 65, 190), 1, cv2.LINE_AA)
    cv2.line(overlay, (center_x + 15, divider_y), (center_x + 95, divider_y), (150, 65, 190), 1, cv2.LINE_AA)
    cv2.circle(overlay, (center_x, divider_y), 4, (215, 85, 245), -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, title_alpha * 0.88, canvas, 1.0 - title_alpha * 0.88, 0.0, dst=canvas)


# ============================================================================
# CLI ENTRY POINT (TEST FRAMES OR FULL MP4 RENDER)
# ============================================================================
def render_preview_keyframes(out_dir, test_times=None):
    os.makedirs(out_dir, exist_ok=True)
    init_worker()
    test_times = test_times or [1.8, 3.8, 6.4, 12.5]
    for sec in test_times:
        f_idx = int(round((sec / DURATION) * (TOTAL_FRAMES - 1)))
        t0 = time.time()
        img = render_frame(f_idx, TOTAL_FRAMES)
        out_path = os.path.join(out_dir, f"preview_{sec:.1f}s.png")
        cv2.imwrite(out_path, img)
        print(f"Rendered {out_path} in {time.time() - t0:.2f}s", flush=True)


def render_full_video(output_mp4):
    print(f"Rendering {TOTAL_FRAMES} frames at {WIDTH}x{HEIGHT} ({FPS} fps)...", flush=True)
    t_start = time.time()
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{WIDTH}x{HEIGHT}", "-pix_fmt", "bgr24", "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output_mp4,
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    num_workers = max(1, min(8, os.cpu_count() or 4))
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        for idx, frame in enumerate(pool.imap(render_frame, range(TOTAL_FRAMES), chunksize=2)):
            proc.stdin.write(frame.tobytes())
            if (idx + 1) % 15 == 0 or (idx + 1) == TOTAL_FRAMES:
                elapsed = time.time() - t_start
                fps_actual = (idx + 1) / max(1e-5, elapsed)
                eta = (TOTAL_FRAMES - idx - 1) / max(1e-5, fps_actual)
                print(f"  [{idx + 1:3d}/{TOTAL_FRAMES}] ({100.0 * (idx + 1) / TOTAL_FRAMES:5.1f}%) "
                      f"- {fps_actual:.2f} fps - ETA {eta/60:.1f} min", flush=True)

    proc.stdin.close()
    stderr_out = proc.stderr.read().decode("utf-8", errors="ignore")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {ret}:\n{stderr_out}")
    print(f"Done! Saved high-quality MP4 to: {output_mp4} in {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    out_folder = os.path.dirname(os.path.abspath(__file__))
    if "--preview" in sys.argv:
        extra = []
        i = sys.argv.index("--preview")
        for a in sys.argv[i + 1:]:
            try:
                extra.append(float(a))
            except ValueError:
                break
        render_preview_keyframes(out_folder, extra or None)
    else:
        render_full_video(os.path.join(out_folder, "lily_3d_botanical_animation.mp4"))