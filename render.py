"""
Render a still image + FOCtave funscripts into an MP4 with animated
electrode-glow overlays.

Each frame:
  - base image, brightness modulated by the volume channel
  - bloom (pre-blurred copy, screened on top, intensity modulated by volume)
  - 4 radial glows at the clicked electrode positions, radius and brightness
    driven by e1-e4 channel values
  - arcs between every pair of electrodes, brightness = geometric mean of
    the two endpoint intensities (visualises foc-stim's any-to-any current
    routing)

Usage (after running foctave.py and place.py):

    python render.py path/to/image.jpg

Defaults: looks for <image_stem>.electrodes.json for positions, the 5
funscripts next to the image, and an audio file matching the funscript
stem. Override any of them with flags.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG = "ffmpeg"


CHANNELS = ["e1", "e2", "e3", "e4", "volume"]
ELECTRODE_CHANNELS = ["e1", "e2", "e3", "e4"]
# Electrodes are visualised as a single polyline in order e1 -> e2 -> e3 -> e4,
# matching the "snake head -> necktie -> snake belly -> snake tail" mental model
# for a typical longitudinal 4-electrode placement.


def load_funscript(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(path.read_text(encoding="utf-8"))
    t = np.array([a["at"] for a in d["actions"]], dtype=np.float64)
    p = np.array([a["pos"] for a in d["actions"]], dtype=np.float64)
    return t, p


def find_funscript(stem_dir: Path, stem: str, ch: str) -> Path:
    candidates = [
        stem_dir / f"{stem}.{ch}.funscript",
        stem_dir.parent / f"{stem}.{ch}.funscript",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find {stem}.{ch}.funscript near {stem_dir}")


def find_audio(stem_dir: Path, stem: str) -> Path | None:
    for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        for candidate in (stem_dir / f"{stem}{ext}", stem_dir.parent / f"{stem}{ext}"):
            if candidate.exists():
                return candidate
    return None


def precompute_glow_stamp(max_radius: int) -> np.ndarray:
    """Generate a radial falloff stamp (float32, [0,1]) for an electrode glow.
    Composited by scaling this stamp's size and brightness per frame."""
    size = max_radius * 2 + 1
    yy, xx = np.mgrid[-max_radius:max_radius + 1, -max_radius:max_radius + 1]
    dist = np.sqrt(xx * xx + yy * yy) / max_radius
    # soft falloff: (1 - d^2)^2 for inside, 0 outside
    stamp = np.clip(1.0 - dist * dist, 0, 1) ** 2
    return stamp.astype(np.float32)


def stamp_glow(canvas: np.ndarray, cx: int, cy: int, stamp: np.ndarray,
               radius_scale: float, color: tuple[float, float, float],
               brightness: float) -> None:
    """Additive-blend a scaled electrode glow into canvas (float32 HxWx3)."""
    h, w = canvas.shape[:2]
    sh, sw = stamp.shape
    r_src = sh // 2
    # Effective radius for this frame
    r_eff = max(1, int(r_src * radius_scale))
    if r_eff < 1:
        return
    # Resample stamp to effective size using numpy (fast enough; for MVP fine)
    # Cheap nearest-neighbour via striding:
    idx = (np.linspace(0, sh - 1, r_eff * 2 + 1)).astype(np.int32)
    stamp_r = stamp[idx][:, idx]  # shape (2r+1, 2r+1)

    r = r_eff
    x0, x1 = cx - r, cx + r + 1
    y0, y1 = cy - r, cy + r + 1

    sx0 = max(0, -x0); sx1 = stamp_r.shape[1] - max(0, x1 - w)
    sy0 = max(0, -y0); sy1 = stamp_r.shape[0] - max(0, y1 - h)
    dx0 = max(0, x0); dx1 = min(w, x1)
    dy0 = max(0, y0); dy1 = min(h, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return

    patch = stamp_r[sy0:sy1, sx0:sx1] * brightness
    for c in range(3):
        canvas[dy0:dy1, dx0:dx1, c] += patch * color[c]


def _safe_lerp(a: np.ndarray, b: np.ndarray, ta: float, tb: float, tv: float) -> np.ndarray:
    if tb - ta < 1e-9:
        return b
    return ((tb - tv) / (tb - ta)) * a + ((tv - ta) / (tb - ta)) * b


def catmull_rom_polyline(control_points: list, samples_per_segment: int = 60,
                         alpha: float = 0.5) -> list[tuple[float, float]]:
    """Return (x, y) points sampled along a centripetal Catmull-Rom spline
    that passes exactly through each control point. With alpha=0.5 the curve
    never overshoots. Accepts 2+ control points; with just 2 it degenerates
    to a straight line."""
    pts = [np.array(p, dtype=np.float64) for p in control_points]
    n = len(pts)
    if n < 2:
        return [tuple(p) for p in pts]
    # Phantom endpoints mirror the first/last segment so endpoint tangents
    # behave smoothly instead of flying off.
    ctrl = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    out: list[tuple[float, float]] = []
    for seg in range(n - 1):
        p0, p1, p2, p3 = (ctrl[seg + i] for i in range(4))
        def knot(ti, a, b):
            d = float(np.linalg.norm(b - a))
            return ti + (d ** alpha if d > 0 else 1e-6)
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        for k in range(samples_per_segment):
            frac = k / samples_per_segment
            t = t1 + (t2 - t1) * frac
            A1 = _safe_lerp(p0, p1, t0, t1, t)
            A2 = _safe_lerp(p1, p2, t1, t2, t)
            A3 = _safe_lerp(p2, p3, t2, t3, t)
            B1 = _safe_lerp(A1, A2, t0, t2, t)
            B2 = _safe_lerp(A2, A3, t1, t3, t)
            C = _safe_lerp(B1, B2, t1, t2, t)
            out.append((float(C[0]), float(C[1])))
    out.append(tuple(pts[-1]))
    return out


def build_path(electrodes_ordered: list[tuple[int, int]],
               spacing_px: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample a centripetal Catmull-Rom spline through
    e1 -> e2 -> e3 -> e4 at ~every `spacing_px` and return
    (xy int array (N,2), barycentric weights (N,4)).

    Each path point's weight vector has two non-zero entries: its position
    along the current SEGMENT contributes to the two adjacent electrodes.
    A point halfway between e2 and e3 gets weights (0, 0.5, 0.5, 0), so its
    local intensity = 0.5 * e2_val + 0.5 * e3_val - the "signal between the
    points" behaviour, now following a smooth curve instead of a zigzag."""
    pts = [np.array(p, dtype=np.float64) for p in electrodes_ordered]
    n_electrodes = len(pts)
    xys: list[tuple[float, float]] = []
    weights: list[np.ndarray] = []
    # Phantom endpoints for the spline
    ctrl = [pts[0] + (pts[0] - pts[1])] + pts + [pts[-1] + (pts[-1] - pts[-2])]
    for seg in range(n_electrodes - 1):
        p0, p1, p2, p3 = (ctrl[seg + i] for i in range(4))
        def knot(ti, a, b, alpha=0.5):
            d = float(np.linalg.norm(b - a))
            return ti + (d ** alpha if d > 0 else 1e-6)
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        chord = float(np.linalg.norm(p2 - p1))
        n_samples = max(4, int(round(chord / spacing_px)))
        for k in range(n_samples):
            frac = k / n_samples
            t = t1 + (t2 - t1) * frac
            A1 = _safe_lerp(p0, p1, t0, t1, t)
            A2 = _safe_lerp(p1, p2, t1, t2, t)
            A3 = _safe_lerp(p2, p3, t2, t3, t)
            B1 = _safe_lerp(A1, A2, t0, t2, t)
            B2 = _safe_lerp(A2, A3, t1, t3, t)
            C = _safe_lerp(B1, B2, t1, t2, t)
            xys.append((float(C[0]), float(C[1])))
            w = np.zeros(n_electrodes, dtype=np.float32)
            w[seg] = 1.0 - frac
            w[seg + 1] = frac
            weights.append(w)
    xys.append((float(pts[-1][0]), float(pts[-1][1])))
    final_w = np.zeros(n_electrodes, dtype=np.float32)
    final_w[-1] = 1.0
    weights.append(final_w)
    return np.array(xys).astype(np.int32), np.array(weights)


def _stamp_at(canvas: np.ndarray, x: int, y: int, stamp: np.ndarray,
              color: tuple[float, float, float], brightness: float) -> None:
    """Additive-stamp a precomputed radial stamp into canvas at (x, y)."""
    h, w = canvas.shape[:2]
    sh = stamp.shape[0]
    r = sh // 2
    x0, x1 = x - r, x + r + 1
    y0, y1 = y - r, y + r + 1
    sx0 = max(0, -x0); sx1 = stamp.shape[1] - max(0, x1 - w)
    sy0 = max(0, -y0); sy1 = stamp.shape[0] - max(0, y1 - h)
    dx0 = max(0, x0); dx1 = min(w, x1)
    dy0 = max(0, y0); dy1 = min(h, y1)
    if dx0 >= dx1 or dy0 >= dy1:
        return
    patch = stamp[sy0:sy1, sx0:sx1] * brightness
    for c in range(3):
        canvas[dy0:dy1, dx0:dx1, c] += patch * color[c]


def _path_tangents(path_xys: np.ndarray) -> np.ndarray:
    """Per-sample unit perpendicular normal for a path. Shape (N, 2).
    Used by effects that need to jitter sideways off the curve."""
    n = len(path_xys)
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)
    pf = path_xys.astype(np.float32)
    diff = np.zeros_like(pf)
    diff[1:-1] = pf[2:] - pf[:-2]
    diff[0] = pf[1] - pf[0]
    diff[-1] = pf[-1] - pf[-2]
    lens = np.maximum(1e-3, np.linalg.norm(diff, axis=1, keepdims=True))
    tan = diff / lens
    # perpendicular: (-dy, dx)
    perp = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    return perp


def _bead_indices(n_samples: int, bead_spacing_samples: int) -> np.ndarray:
    """Evenly-spaced sub-sampling indices along a path. `bead_spacing_samples`
    is the stride in path-sample units — with build_path's 2 px per sample,
    stride=15 gives ~30 px between beads."""
    if n_samples <= 0:
        return np.zeros(0, dtype=np.int32)
    stride = max(1, int(bead_spacing_samples))
    return np.arange(stride // 2, n_samples, stride, dtype=np.int32)


def draw_path_beads(canvas: np.ndarray, path_xys: np.ndarray,
                    path_intensities: np.ndarray,
                    color: tuple[float, float, float],
                    stamp: np.ndarray, bead_stride: int,
                    brightness_scale: float = 200.0,
                    perp: np.ndarray | None = None,
                    t_s: float = 0.0, jitter: float = 0.0,
                    twinkle: float = 0.0) -> None:
    """Stamp one bead at every `bead_stride`-th path sample. Brightness is
    driven by `path_intensities` at that sample — i.e. the same barycentric
    × traveling-wave envelope the ribbon uses. Optional perpendicular jitter
    and per-bead twinkle turn the same positioning into 'sparks'.

    This unifies lights/sparks with ribbon: identical position logic along
    the curve, identical intensity envelope, only the stamp stride, stamp
    size, and jitter differ."""
    import math
    h, w = canvas.shape[:2]
    sh = stamp.shape[0]
    r = sh // 2
    idxs = _bead_indices(len(path_xys), bead_stride)
    for order, i in enumerate(idxs):
        inten = float(path_intensities[i])
        if inten <= 0.04:
            continue
        x = int(path_xys[i, 0])
        y = int(path_xys[i, 1])
        bright_mul = 1.0
        if jitter > 0 and perp is not None:
            phase = t_s * 14.0 + order * 1.37
            off = math.sin(phase) * jitter
            x = int(round(x + perp[i, 0] * off))
            y = int(round(y + perp[i, 1] * off))
        if twinkle > 0:
            phase = t_s * 6.0 + order * 0.91
            bright_mul *= (1.0 - twinkle) + twinkle * (0.5 + 0.5 * math.sin(phase))
        bright = inten * brightness_scale * bright_mul
        x0, x1 = x - r, x + r + 1
        y0, y1 = y - r, y + r + 1
        sx0 = max(0, -x0); sx1 = stamp.shape[1] - max(0, x1 - w)
        sy0 = max(0, -y0); sy1 = stamp.shape[0] - max(0, y1 - h)
        dx0 = max(0, x0); dx1 = min(w, x1)
        dy0 = max(0, y0); dy1 = min(h, y1)
        if dx0 >= dx1 or dy0 >= dy1:
            continue
        patch = stamp[sy0:sy1, sx0:sx1] * bright
        for c in range(3):
            canvas[dy0:dy1, dx0:dx1, c] += patch * color[c]


def draw_path_ribbon(canvas: np.ndarray, path_xys: np.ndarray,
                     path_intensities: np.ndarray, color: tuple[float, float, float],
                     stamp: np.ndarray, thickness_scale: float,
                     brightness_scale: float = 140.0) -> None:
    """Stamp a small radial glow at each path point with brightness scaled
    by that point's local intensity. Overlapping stamps add up, yielding a
    continuous ribbon whose per-pixel brightness matches the interpolated
    electrode values. `brightness_scale` controls peak additive brightness
    per fully-lit path point."""
    h, w = canvas.shape[:2]
    sh = stamp.shape[0]
    r_src = sh // 2

    r_eff = max(2, int(r_src * thickness_scale))
    idx = np.linspace(0, sh - 1, r_eff * 2 + 1).astype(np.int32)
    stamp_small = stamp[idx][:, idx]
    r = r_eff

    for (x, y), intensity in zip(path_xys, path_intensities):
        if intensity <= 0.05:
            continue
        x, y = int(x), int(y)
        x0, x1 = x - r, x + r + 1
        y0, y1 = y - r, y + r + 1
        sx0 = max(0, -x0); sx1 = stamp_small.shape[1] - max(0, x1 - w)
        sy0 = max(0, -y0); sy1 = stamp_small.shape[0] - max(0, y1 - h)
        dx0 = max(0, x0); dx1 = min(w, x1)
        dy0 = max(0, y0); dy1 = min(h, y1)
        if dx0 >= dx1 or dy0 >= dy1:
            continue
        patch = stamp_small[sy0:sy1, sx0:sx1] * (intensity * brightness_scale)
        for c in range(3):
            canvas[dy0:dy1, dx0:dx1, c] += patch * color[c]


def _prepare_scene(image_path: Path, electrodes: dict, max_dim: int):
    """Load, resize to match max_dim, pre-blur, and precompute the ribbon
    path for one scene. Returns a dict with everything the render loop needs."""
    base = Image.open(image_path).convert("RGB")
    iw, ih = base.size
    if max(iw, ih) > max_dim:
        sf = max_dim / max(iw, ih)
        nw, nh = int(iw * sf), int(ih * sf)
        nw -= nw % 2
        nh -= nh % 2
        base = base.resize((nw, nh), Image.LANCZOS)
        scaled = {k: (int(v[0] * sf), int(v[1] * sf)) for k, v in electrodes.items()}
    else:
        nw = iw - (iw % 2)
        nh = ih - (ih % 2)
        if (nw, nh) != (iw, ih):
            base = base.crop((0, 0, nw, nh))
        scaled = dict(electrodes)
    w, h = nw, nh

    base_arr = np.array(base, dtype=np.float32)
    blur_radius = max(6, min(w, h) * 0.025)
    bloom_arr = np.array(base.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
                         dtype=np.float32)
    ordered = [scaled[ch] for ch in ELECTRODE_CHANNELS]
    path_xys, path_weights = build_path(ordered, spacing_px=2.0)
    path_t = np.linspace(0.0, 1.0, len(path_xys), dtype=np.float32)
    path_perp = _path_tangents(path_xys)

    return {
        "w": w, "h": h,
        "base_arr": base_arr,
        "bloom_arr": bloom_arr,
        "electrodes": scaled,
        "path_xys": path_xys,
        "path_weights": path_weights,
        "path_t": path_t,
        "path_perp": path_perp,
    }


def render_multi(
    scenes: list,
    funscripts: dict,
    audio: Path | None,
    output: Path,
    fps: int,
    max_dim: int,
    duration_s: float | None,
    bloom_strength: float,
    base_dim_range: tuple[float, float],
    scene_duration_s: float | None = None,
    crossfade_s: float = 0.5,
    effect_opacity: float = 0.55,
    effect_style: str = "ribbon",
    ribbon_color: tuple[float, float, float] | None = None,
    progress=None,
) -> None:
    """Render a video that rotates through a list of scenes. Each scene is a
    dict with keys 'image_path' (Path) and 'electrodes' (dict e1..e4 -> (x,y))
    and an optional 'overrides' dict.

    `effect_style` selects how the signal is visualised:
      - "ribbon": diffuse glow along the whole path e1->e2->e3->e4 (default).
      - "lights": one discrete spot per segment, positioned by the intensity
        ratio of its endpoints (spot slides toward the hotter end).
      - "sparks": like lights, but with jittered flickering stamps for an
        electric-arc feel.

    `ribbon_color` overrides the amber tint used for ribbon/lights/sparks
    (RGB floats in [0,1]); pass None for the default warm amber."""
    if not scenes:
        raise ValueError("render_multi needs at least one scene")

    # First scene dictates output size
    first = _prepare_scene(Path(scenes[0]["image_path"]),
                            scenes[0]["electrodes"], max_dim)
    first["effect_opacity"] = float(
        scenes[0].get("overrides", {}).get("effect_opacity", effect_opacity))
    w, h = first["w"], first["h"]

    # Prepare every scene, forcing each to match the first scene's size so
    # compositing is uniform. Resize if needed.
    prepped = [first]
    for s in scenes[1:]:
        img = Image.open(Path(s["image_path"])).convert("RGB")
        iw, ih = img.size
        # Fit & center into (w, h): resize preserving aspect, then letterbox
        scale_fit = min(w / iw, h / ih)
        rw, rh = int(iw * scale_fit), int(ih * scale_fit)
        img_r = img.resize((rw, rh), Image.LANCZOS)
        canvas_img = Image.new("RGB", (w, h), (0, 0, 0))
        ox, oy = (w - rw) // 2, (h - rh) // 2
        canvas_img.paste(img_r, (ox, oy))
        # Map original-space electrodes -> letterboxed canvas coords
        fitted_elec = {k: (int(v[0] * scale_fit) + ox,
                            int(v[1] * scale_fit) + oy)
                       for k, v in s["electrodes"].items()}
        base_arr = np.array(canvas_img, dtype=np.float32)
        blur_radius = max(6, min(w, h) * 0.025)
        bloom_arr = np.array(canvas_img.filter(ImageFilter.GaussianBlur(radius=blur_radius)),
                             dtype=np.float32)
        ordered = [fitted_elec[ch] for ch in ELECTRODE_CHANNELS]
        path_xys, path_weights = build_path(ordered, spacing_px=2.0)
        path_t = np.linspace(0.0, 1.0, len(path_xys), dtype=np.float32)
        path_perp = _path_tangents(path_xys)
        prepped.append({
            "w": w, "h": h,
            "base_arr": base_arr, "bloom_arr": bloom_arr,
            "electrodes": fitted_elec,
            "path_xys": path_xys, "path_weights": path_weights,
            "path_t": path_t, "path_perp": path_perp,
            "effect_opacity": float(s.get("overrides", {}).get("effect_opacity",
                                                               effect_opacity)),
        })

    # Duration
    total_ms = max(fs[0][-1] for fs in funscripts.values())
    if duration_s is not None:
        total_ms = min(total_ms, duration_s * 1000)
    n_frames = int(total_ms / 1000 * fps)
    total_s = total_ms / 1000.0

    n_scenes = len(prepped)
    if scene_duration_s is None or scene_duration_s <= 0:
        scene_duration_s = max(1.0, total_s / max(1, n_scenes))
    crossfade_s = max(0.0, min(crossfade_s, scene_duration_s / 2))

    print(f"Rendering {n_frames} frames ({total_s:.1f}s) across {n_scenes} scene(s); "
          f"scene={scene_duration_s:.1f}s, crossfade={crossfade_s:.1f}s")

    # Glow stamps are resolution-dependent but scene-independent
    electrode_max_radius = int(min(w, h) * 0.18)
    stamp = precompute_glow_stamp(electrode_max_radius)
    ribbon_stamp = precompute_glow_stamp(int(min(w, h) * 0.035))

    electrode_colors = {
        "e1": (1.00, 0.35, 0.25),
        "e2": (1.00, 0.55, 0.20),
        "e3": (0.35, 1.00, 0.55),
        "e4": (0.30, 0.70, 1.00),
    }
    effect_color = ribbon_color if ribbon_color is not None else (1.0, 0.70, 0.30)

    # ffmpeg pipe
    cmd = [FFMPEG, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
    if audio and audio.exists():
        cmd += ["-i", str(audio)]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p", "-crf", "20"]
    if audio and audio.exists():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    if duration_s is not None:
        cmd += ["-t", str(duration_s)]
    cmd += [str(output)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    import time
    t_start = time.perf_counter()

    # Effect opacity scales how dominant the glow/ribbon is vs the base
    # image. Each prepped scene carries its own opacity (either overridden
    # for that scene or inherited from the global `effect_opacity`).

    # Stamp for bead-style effects (lights / sparks). Bigger than the ribbon
    # stamp so each bead reads as its own point of light rather than blending
    # into a continuous line.
    light_stamp = precompute_glow_stamp(int(min(w, h) * 0.06))

    def render_scene_frame(scene, vals, t_s):
        vol = vals["volume"]
        dim_lo, dim_hi = base_dim_range
        dim = dim_lo + (dim_hi - dim_lo) * vol
        canvas = scene["base_arr"] * dim + scene["bloom_arr"] * (bloom_strength * vol)
        e_values = np.array([vals["e1"], vals["e2"], vals["e3"], vals["e4"]],
                            dtype=np.float32)

        scene_opacity = scene.get("effect_opacity", effect_opacity)

        # --- Shared path-based intensity envelope (used by all styles) ---
        # Barycentric blend: each path sample's value is the weighted average
        # of its two adjacent electrodes. Traveling wave modulates this so
        # the "signal" visibly flows head-to-tail along the curve.
        path_intensity = scene["path_weights"] @ e_values
        wave = 0.60 + 0.40 * np.sin(2 * np.pi * (scene["path_t"] * 2.0 - t_s * 1.2))
        path_intensity = path_intensity * wave

        if effect_style == "lights":
            # Discrete beads along the same curve, same envelope — just a
            # sparser sub-sampling with a bigger stamp per point.
            draw_path_beads(canvas, scene["path_xys"], path_intensity,
                            effect_color, light_stamp,
                            bead_stride=scene.get("bead_stride", 14),
                            brightness_scale=190.0 * scene_opacity)
        elif effect_style == "sparks":
            # Same beads + perpendicular jitter off the curve + per-bead
            # twinkle for crackle. Still travels head-to-tail via `wave`.
            draw_path_beads(canvas, scene["path_xys"], path_intensity,
                            effect_color, light_stamp,
                            bead_stride=scene.get("bead_stride", 14),
                            brightness_scale=170.0 * scene_opacity,
                            perp=scene["path_perp"], t_s=t_s,
                            jitter=scene.get("spark_jitter_px", 6.0),
                            twinkle=0.55)
        else:
            # Ribbon: every path sample stamped with a soft small glow,
            # giving a continuous flowing line.
            ribbon_thickness = 0.55 + 0.45 * vol
            draw_path_ribbon(canvas, scene["path_xys"], path_intensity, effect_color,
                             ribbon_stamp, ribbon_thickness,
                             brightness_scale=140.0 * scene_opacity)

        # Anchor electrode glows on top (all styles)
        electrode_peak = 180.0 * scene_opacity
        for ch in ELECTRODE_CHANNELS:
            intensity = vals[ch]
            if intensity <= 0.02:
                continue
            cx, cy = scene["electrodes"][ch]
            stamp_glow(canvas, cx, cy, stamp,
                       0.35 + 0.65 * intensity,
                       electrode_colors[ch],
                       electrode_peak * intensity)
        return canvas

    try:
        for i in range(n_frames):
            t_ms = i * 1000 / fps
            t_s = t_ms / 1000.0
            vals = {ch: float(np.interp(t_ms, funscripts[ch][0], funscripts[ch][1])) / 100.0
                    for ch in CHANNELS}

            # Scene scheduling. Each scene occupies [i*D, (i+1)*D) seconds.
            # Crossfade happens in the first `crossfade_s` of each scene,
            # blending from the previous scene into the current one.
            if n_scenes == 1:
                active = 0
                fade_alpha = 1.0
            else:
                # Which scene index covers this time?
                cur_idx = int(t_s // scene_duration_s) % n_scenes
                t_in_scene = t_s - (t_s // scene_duration_s) * scene_duration_s
                if t_in_scene < crossfade_s and (t_s >= scene_duration_s):
                    # Crossfading from previous scene (cur_idx - 1) to current
                    fade_alpha = t_in_scene / max(crossfade_s, 1e-6)
                    prev_idx = (cur_idx - 1) % n_scenes
                    frame_prev = render_scene_frame(prepped[prev_idx], vals, t_s)
                    frame_cur = render_scene_frame(prepped[cur_idx], vals, t_s)
                    canvas = frame_prev * (1.0 - fade_alpha) + frame_cur * fade_alpha
                    active = cur_idx
                else:
                    active = cur_idx
                    fade_alpha = 1.0
                    canvas = render_scene_frame(prepped[active], vals, t_s)

            if n_scenes == 1:
                canvas = render_scene_frame(prepped[0], vals, t_s)

            frame = np.clip(canvas, 0, 255).astype(np.uint8)
            proc.stdin.write(frame.tobytes())

            if progress is not None and i % max(1, fps // 2) == 0:
                progress(i / max(1, n_frames),
                         f"frame {i}/{n_frames}  ({100*i/max(1,n_frames):.1f}%)")
            if i % fps == 0 and i > 0:
                elapsed = time.perf_counter() - t_start
                fps_now = i / elapsed
                eta = (n_frames - i) / max(fps_now, 0.01)
                print(f"\r  frame {i}/{n_frames}  ({100*i/n_frames:5.1f}%)  "
                      f"{fps_now:.1f}fps  ETA {eta:.0f}s", end="", flush=True)
        print()
    finally:
        proc.stdin.close()
        proc.wait()

    elapsed = time.perf_counter() - t_start
    if progress is not None:
        progress(1.0, f"Wrote {output.name}  ({elapsed:.1f}s)")
    print(f"Wrote {output}  ({elapsed:.1f}s render)")


def render(
    image_path: Path,
    electrodes: dict,
    funscripts: dict,
    audio: Path | None,
    output: Path,
    fps: int,
    max_dim: int,
    duration_s: float | None,
    bloom_strength: float,
    base_dim_range: tuple[float, float],
    effect_opacity: float = 0.55,
    effect_style: str = "ribbon",
    ribbon_color: tuple[float, float, float] | None = None,
    progress=None,
) -> None:
    """Single-image convenience wrapper around render_multi."""
    return render_multi(
        scenes=[{"image_path": image_path, "electrodes": electrodes}],
        funscripts=funscripts,
        audio=audio,
        output=output,
        fps=fps,
        max_dim=max_dim,
        duration_s=duration_s,
        bloom_strength=bloom_strength,
        base_dim_range=base_dim_range,
        effect_opacity=effect_opacity,
        effect_style=effect_style,
        ribbon_color=ribbon_color,
        progress=progress,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="Background still image")
    ap.add_argument("--electrodes", type=Path, default=None,
                    help="Path to electrodes JSON (default: <image>.electrodes.json)")
    ap.add_argument("--funscripts-stem", type=Path, default=None,
                    help="Stem path for funscripts (default: same as image stem)")
    ap.add_argument("--audio", type=Path, default=None,
                    help="Audio file to embed (default: <stem>.wav/.mp3/...)")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output MP4 path (default: <image_stem>.mp4)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-dim", type=int, default=1280,
                    help="Cap output resolution (default 1280 — good speed/quality balance)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Cap render duration in seconds (useful for previews)")
    ap.add_argument("--bloom", type=float, default=0.45,
                    help="Bloom strength 0-1 (default 0.45)")
    ap.add_argument("--min-dim", type=float, default=0.55,
                    help="Base brightness at silence 0-1 (default 0.55)")
    ap.add_argument("--max-base", type=float, default=1.0,
                    help="Base brightness at peak volume 0-1 (default 1.0)")
    args = ap.parse_args()

    image_path = args.image.resolve()
    if not image_path.exists():
        print(f"Not found: {image_path}", file=sys.stderr)
        return 1

    electrodes_path = (args.electrodes or image_path.with_suffix(".electrodes.json")).resolve()
    if not electrodes_path.exists():
        print(f"No electrodes file at {electrodes_path}.\nRun: python place.py \"{image_path}\"",
              file=sys.stderr)
        return 1
    ed = json.loads(electrodes_path.read_text(encoding="utf-8"))
    electrodes = {k: (int(v["x"]), int(v["y"])) for k, v in ed["electrodes"].items()}

    if args.funscripts_stem:
        stem_dir = args.funscripts_stem.parent
        stem = args.funscripts_stem.name
    else:
        stem_dir = image_path.parent
        stem = image_path.stem

    funscripts = {ch: load_funscript(find_funscript(stem_dir, stem, ch)) for ch in CHANNELS}

    audio = args.audio.resolve() if args.audio else find_audio(stem_dir, stem)
    if audio:
        print(f"Audio: {audio}")
    else:
        print("No audio found — rendering silent video.")

    output = (args.output or image_path.with_suffix(".mp4")).resolve()

    render(
        image_path=image_path,
        electrodes=electrodes,
        funscripts=funscripts,
        audio=audio,
        output=output,
        fps=args.fps,
        max_dim=args.max_dim,
        duration_s=args.duration,
        bloom_strength=args.bloom,
        base_dim_range=(args.min_dim, args.max_base),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
