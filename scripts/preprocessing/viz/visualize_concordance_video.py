"""
Concordance explainer video: group fixation density (blue cloud) with one HIGH-concordance
participant (green dot) and one LOW-concordance participant (red dot) drawn on top.

Purpose: make the concordance metric legible. The metric says participant X's fixation
density correlates r=0.69 with the group while participant Y's correlates r=0.15 -- this
renders what that actually looks like moment to moment.

Same windowing as the analysis, not a re-invention
--------------------------------------------------
Every quantity on screen comes from the concordance pipeline's own functions, imported
directly from scripts/analysis/calc_concordance_group.py (load_video_fixations,
build_participant_arrays, closed_form_tile_values) and plot_patch_grid.compute_patch_grid,
so the math cannot drift from the metric being illustrated:

- window = 1.0s, step = 0.25s, sigma = 46px, fixations weighted by their overlap duration
  with the window (ms) -- the confirmed parameters in METHODS_DECISIONS.md.
- The per-participant r values in the on-screen readout are the real leave-one-out
  concordance values (individual vs. group-minus-that-individual, over the 120 scene-tile
  centers), computed here exactly as calc_concordance_group.py computes them.

Blue cloud = the continuous density field, not the 120 tiles
------------------------------------------------------------
The metric correlates two 120-value vectors sampled at scene-tile centers. Those 120 values
are samples OF this field: the same duration-weighted sum of sigma=46px Gaussians. Drawing
the field rather than 120 discrete squares keeps it readable as "where the group looked"
while remaining the same quantity the tiles sample. Rendered on a downsampled grid
(--density_scale) and upscaled, since the field is smooth at sigma=46px.

The cloud excludes BOTH highlighted participants, so it is a fair common reference for the
green and red dots simultaneously. Each displayed r value is still that participant's true
leave-one-out r, whose group therefore includes the other highlighted participant -- a
one-of-N difference from the drawn cloud, noted here rather than silently glossed.

Frame-to-window mapping
-----------------------
A frame at time t shows the most recently COMPLETED analysis window (the largest
window_end_s <= t), so the cloud reads as "where the group looked over the last second"
rather than including gaze that hasn't happened yet. Frames before t=1.0s show window 0.
The cloud therefore updates every 0.25s (the analysis step) while the stimulus plays at its
native frame rate.

Dots
----
A participant can have several fixations inside a 1.0s window, and the metric weights each
by its overlap duration -- so each of that participant's overlapping fixations is drawn,
with opacity and radius scaled by its overlap weight. The biggest, most opaque dot is the
fixation carrying the most weight in that window's density vector.

Video I/O via the system ffmpeg/ffprobe binaries (raw RGB24 pipes) and Pillow for drawing,
matching visualize_fixation_dots.py -- this machine has no opencv-python.

Usage
-----
    python scripts/preprocessing/viz/visualize_concordance_video.py \\
        --video sesameindia_2 --code f329476c
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_concordance_group import (  # noqa: E402
    DISPLAY_H,
    DISPLAY_W,
    build_participant_arrays,
    closed_form_tile_values,
    load_video_fixations,
)
from plot_patch_grid import compute_patch_grid  # noqa: E402

OUTPUT_WIDTH, OUTPUT_HEIGHT = 1280, 720

HIGH_COLOR = (40, 200, 90)     # green: high-concordance participant
LOW_COLOR = (230, 45, 45)      # red:   low-concordance participant
CLOUD_COLOR = (70, 140, 255)   # blue:  group density

BG_OPACITY = 0.22
CLOUD_MAX_ALPHA = 0.80
CLOUD_GAMMA = 0.65             # <1 lifts low-density areas so the cloud's extent stays visible

# The background is desaturated to greyscale before dimming: several clips (this script's
# default included -- sesameindia_2 is a scene with a large blue muppet) have saturated
# colours that the blue density cloud and the green/red dots would otherwise disappear into.
HUD_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


# ── ffmpeg helpers (same approach as visualize_fixation_dots.py) ─────────────

def probe_video(path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", path]
    stream = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration_s = float(stream["duration"])
    n_frames = int(stream["nb_frames"]) if stream.get("nb_frames", "N/A") not in (None, "N/A") \
        else int(round(duration_s * fps))
    return {"fps": fps, "n_frames": n_frames, "duration_s": duration_s}


def open_frame_reader(path: str, w: int, h: int) -> subprocess.Popen:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={w}:{h}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "pipe:1"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)


def open_frame_writer(path: str, w: int, h: int, fps: float) -> subprocess.Popen:
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def read_frame(proc: subprocess.Popen, w: int, h: int) -> np.ndarray | None:
    n_bytes = w * h * 3
    buf = proc.stdout.read(n_bytes)
    return None if len(buf) < n_bytes else np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)


# ── concordance (identical math to calc_concordance_group.py) ────────────────

def compute_window_r(pid_arrays: dict, pids: list[str], n_windows: int,
                      window_s: float, step_s: float, sigma_px: float,
                      cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """(n_pids, n_windows) leave-one-out concordance r, via total_vec - individual_vec."""
    r_mat = np.full((len(pids), n_windows), np.nan)
    for w in range(n_windows):
        start_ms = w * step_s * 1000
        end_ms = start_ms + window_s * 1000
        vecs = np.array([closed_form_tile_values(*pid_arrays[p], start_ms, end_ms, sigma_px, cx, cy)
                         for p in pids])
        total = vecs.sum(axis=0)
        for i in range(len(pids)):
            vec, group_vec = vecs[i], total - vecs[i]
            if vec.std() > 0 and group_vec.std() > 0:
                r_mat[i, w] = np.corrcoef(vec, group_vec)[0, 1]
    return r_mat


def window_fixations(arrays: tuple, start_ms: float, end_ms: float) -> tuple[np.ndarray, ...]:
    """This participant's fixations overlapping the window, with their overlap weights (ms)."""
    startT, endT, img_x, img_y = arrays
    overlap = np.minimum(endT, end_ms) - np.maximum(startT, start_ms)
    m = overlap > 0
    return img_x[m], img_y[m], overlap[m]


def density_field(pid_arrays: dict, pids: list[str], start_ms: float, end_ms: float,
                   sigma_px: float, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Duration-weighted sum of sigma Gaussians over a grid -- the continuous field that
    closed_form_tile_values() samples at the 120 tile centers."""
    xs, ys, ws = [], [], []
    for pid in pids:
        x, y, w = window_fixations(pid_arrays[pid], start_ms, end_ms)
        xs.append(x), ys.append(y), ws.append(w)
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    if len(x) == 0:
        return np.zeros((len(gy), len(gx)), dtype=np.float32)

    dx2 = (gx[None, :] - x[:, None]) ** 2            # (n_fix, grid_w)
    dy2 = (gy[None, :] - y[:, None]) ** 2            # (n_fix, grid_h)
    gauss_x = np.exp(-dx2 / (2 * sigma_px ** 2))
    gauss_y = np.exp(-dy2 / (2 * sigma_px ** 2))
    # separable: each fixation contributes w * outer(gauss_y, gauss_x)
    return np.einsum("i,ih,iw->hw", w, gauss_y, gauss_x).astype(np.float32)


# ── drawing ─────────────────────────────────────────────────────────────────

def to_output_px(img_x: float, img_y: float, out_w: int, out_h: int) -> tuple[float, float]:
    """Image/top-left-origin display coords (as build_participant_arrays returns) -> output px."""
    return img_x * out_w / DISPLAY_W, img_y * out_h / DISPLAY_H


def cloud_overlay(field: np.ndarray, vmax: float, out_w: int, out_h: int) -> Image.Image:
    alpha = np.clip(field / vmax, 0, 1) ** CLOUD_GAMMA * CLOUD_MAX_ALPHA
    small = np.zeros((*field.shape, 4), dtype=np.uint8)
    small[..., 0], small[..., 1], small[..., 2] = CLOUD_COLOR
    small[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(small, mode="RGBA").resize((out_w, out_h), Image.BILINEAR)


def draw_participant(draw: ImageDraw.ImageDraw, arrays: tuple, start_ms: float, end_ms: float,
                      color: tuple, out_w: int, out_h: int, base_radius: int) -> None:
    x, y, w = window_fixations(arrays, start_ms, end_ms)
    if len(x) == 0:
        return
    w_norm = w / w.max()
    for xi, yi, wi in zip(x, y, w_norm):
        px, py = to_output_px(xi, yi, out_w, out_h)
        radius = base_radius * (0.55 + 0.45 * wi)
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     fill=color + (int(255 * (0.45 + 0.55 * wi)),),
                     outline=(255, 255, 255, 230), width=2)


def load_hud_font(size: int) -> ImageFont.ImageFont:
    for path in HUD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fmt_r(r: float) -> str:
    return "  no data" if np.isnan(r) else f"r = {r:5.2f}"


def rank_label(rank: int, n_total: int) -> str:
    """rank is 1=lowest mean r in this clip -- shown so the HUD's HIGH/LOW vs. higher/lower
    wording (see draw_hud) is never a clip-wide claim the numbers don't back up: a pair picked
    for being the extremes on one clip can be unremarkable on another (METHODS_DECISIONS.md)."""
    return f"rank {rank}/{n_total}"


def draw_hud(draw: ImageDraw.ImageDraw, video: str, high_pid: str, low_pid: str,
              high_r: float, low_r: float, high_mean: float, low_mean: float,
              win_start: float, win_end: float, n_group: int, out_w: int,
              high_rank: int, low_rank: int, n_total: int) -> None:
    font = load_hud_font(17)
    small = load_hud_font(14)
    pad, line_h = 14, 25
    box_w, box_h = 640, 168
    draw.rounded_rectangle([pad, pad, pad + box_w, pad + box_h], radius=10, fill=(0, 0, 0, 185))

    def row(i, text, f=font, color=(255, 255, 255, 255), swatch=None):
        y = pad + 12 + i * line_h
        x = pad + 14
        if swatch is not None:
            r = 7
            draw.ellipse([x, y + 3, x + 2 * r, y + 3 + 2 * r], fill=swatch + (255,),
                         outline=(255, 255, 255, 255), width=1)
            x += 2 * r + 10
        draw.text((x, y), text, font=f, fill=color)

    high_pctl = (high_rank - 1) / (n_total - 1) if n_total > 1 else 0.5
    low_pctl = (low_rank - 1) / (n_total - 1) if n_total > 1 else 0.5
    high_word = "HIGH" if high_pctl >= 0.8 else "higher"
    low_word = "LOW" if low_pctl <= 0.2 else "lower"

    row(0, f"{video}    window {win_start:.2f}-{win_end:.2f}s")
    row(1, f"group fixation density  (n={n_group})", swatch=CLOUD_COLOR)
    row(2, f"{high_pid}  {fmt_r(high_r)}   {high_word} of this pair  "
           f"(clip mean {high_mean:.2f}, {rank_label(high_rank, n_total)})", swatch=HIGH_COLOR)
    row(3, f"{low_pid}  {fmt_r(low_r)}   {low_word} of this pair  "
           f"(clip mean {low_mean:.2f}, {rank_label(low_rank, n_total)})", swatch=LOW_COLOR)
    row(4.35, "r = participant vs. leave-one-out group, 120 scene tiles", f=small,
        color=(205, 205, 205, 255))
    row(5.25, "window 1.0s, step 0.25s, sigma 46px, duration-weighted", f=small,
        color=(205, 205, 205, 255))


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Concordance explainer video: group density cloud + high/low participant dots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video", default="sesameindia_2",
                    help="Clip to render. Default has the widest high-vs-low concordance spread.")
    p.add_argument("--code", default="f329476c", help="I2MC parameter code on the fixation CSVs.")
    p.add_argument("--high_pid", default=None, help="Override the auto-picked highest-concordance pid.")
    p.add_argument("--low_pid", default=None, help="Override the auto-picked lowest-concordance pid.")
    p.add_argument("--fixation_dir", default="data/local_cache/fixations/adults",
                    help="LOCAL COPY of adult fixation CSVs (never the server mount).")
    p.add_argument("--participant_summary_path", default="data/local_cache/participant_summary.csv")
    p.add_argument("--stimuli_dir", default="stimuli/main_blocks")
    p.add_argument("--output_dir", default="figures/concordance")
    p.add_argument("--window_s", type=float, default=1.0)
    p.add_argument("--step_s", type=float, default=0.25)
    p.add_argument("--sigma_px", type=float, default=46.0)
    p.add_argument("--patch_diameter_px", type=float, default=180)
    p.add_argument("--patch_n_cols", type=int, default=15)
    p.add_argument("--output_width", type=int, default=OUTPUT_WIDTH)
    p.add_argument("--output_height", type=int, default=OUTPUT_HEIGHT)
    p.add_argument("--density_scale", type=int, default=4,
                    help="Density grid downsample factor vs. output size (field is smooth at sigma=46px).")
    p.add_argument("--dot_radius", type=int, default=11, help="Max dot radius in output px.")
    p.add_argument("--bg_opacity", type=float, default=BG_OPACITY)
    p.add_argument("--max_seconds", type=float, default=None, help="Render only the first N seconds.")
    return p.parse_args()


def main():
    args = parse_args()
    fixation_dir = PROJECT_ROOT / args.fixation_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = PROJECT_ROOT / args.stimuli_dir / f"{args.video}_stripped.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / args.stimuli_dir / f"{args.video}.mp4"
    if not video_path.exists():
        sys.exit(f"ERROR: no stimulus video for '{args.video}' in {args.stimuli_dir}")

    info = probe_video(str(video_path))
    fps, duration_s = info["fps"], info["duration_s"]
    n_frames = info["n_frames"] if args.max_seconds is None else int(round(args.max_seconds * fps))
    n_frames = min(n_frames, info["n_frames"])
    print(f"Video: {video_path.name}  fps={fps:.3f}  duration={duration_s:.1f}s")

    fix_df = load_video_fixations(fixation_dir, args.video, args.code,
                                   PROJECT_ROOT / args.participant_summary_path)
    pid_arrays = build_participant_arrays(fix_df)
    pids = sorted(pid_arrays)

    centers, grid_info = compute_patch_grid(args.patch_diameter_px, args.patch_n_cols,
                                             DISPLAY_W, DISPLAY_H)
    cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
    n_windows = max(0, int((duration_s - args.window_s) // args.step_s) + 1)
    print(f"Scene tiles: {grid_info['n_cols']}x{grid_info['n_rows']}={len(centers)}; "
          f"{len(pids)} participants; {n_windows} windows")

    print("Computing concordance (same math as calc_concordance_group.py)...")
    t0 = time.perf_counter()
    r_mat = compute_window_r(pid_arrays, pids, n_windows, args.window_s, args.step_s,
                              args.sigma_px, cx, cy)
    mean_r = np.nanmean(r_mat, axis=1)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    order = np.argsort(mean_r)  # ascending by mean_r; order[0] = lowest in this clip
    high_pid = args.high_pid or pids[order[-1]]
    low_pid = args.low_pid or pids[order[0]]
    hi, li = pids.index(high_pid), pids.index(low_pid)
    n_total = len(pids)
    high_rank = int(np.where(order == hi)[0][0]) + 1  # 1 = lowest in this clip
    low_rank = int(np.where(order == li)[0][0]) + 1
    print(f"  HIGH {high_pid}: mean r = {mean_r[hi]:.3f}  (clip rank {high_rank}/{n_total})")
    print(f"  LOW  {low_pid}: mean r = {mean_r[li]:.3f}  (clip rank {low_rank}/{n_total})")

    group_pids = [p for p in pids if p not in (high_pid, low_pid)]

    grid_w = args.output_width // args.density_scale
    grid_h = args.output_height // args.density_scale
    gx = (np.arange(grid_w) + 0.5) * DISPLAY_W / grid_w
    gy = (np.arange(grid_h) + 0.5) * DISPLAY_H / grid_h

    print(f"Computing group density fields ({grid_w}x{grid_h} grid)...")
    t0 = time.perf_counter()
    fields = np.stack([
        density_field(pid_arrays, group_pids, w * args.step_s * 1000,
                       (w * args.step_s + args.window_s) * 1000, args.sigma_px, gx, gy)
        for w in range(n_windows)
    ])
    vmax = float(np.percentile(fields.max(axis=(1, 2)), 95))
    print(f"  done in {time.perf_counter() - t0:.1f}s (vmax={vmax:.1f})")

    reader = open_frame_reader(str(video_path), args.output_width, args.output_height)
    out_path = output_dir / f"{args.video}_concordance_{high_pid}_vs_{low_pid}.mp4"
    writer = open_frame_writer(str(out_path), args.output_width, args.output_height, fps)

    cached_w, cached_cloud = -1, None
    for frame_idx in range(n_frames):
        bg = read_frame(reader, args.output_width, args.output_height)
        if bg is None:
            print(f"\n  stream ended early at frame {frame_idx}/{n_frames}")
            break

        t_s = frame_idx / fps
        # most recently COMPLETED analysis window: largest w with window_end_s <= t_s
        w = int(np.clip(np.floor((t_s - args.window_s) / args.step_s), 0, n_windows - 1))
        start_ms = w * args.step_s * 1000
        end_ms = start_ms + args.window_s * 1000

        if w != cached_w:
            cached_cloud = cloud_overlay(fields[w], vmax, args.output_width, args.output_height)
            cached_w = w

        grey = bg @ np.array([0.299, 0.587, 0.114])          # desaturate, then dim
        base = Image.fromarray(
            np.repeat((grey * args.bg_opacity).astype(np.uint8)[:, :, None], 3, axis=2),
            mode="RGB").convert("RGBA")
        base = Image.alpha_composite(base, cached_cloud)

        overlay = Image.new("RGBA", (args.output_width, args.output_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw_participant(draw, pid_arrays[low_pid], start_ms, end_ms, LOW_COLOR,
                          args.output_width, args.output_height, args.dot_radius)
        draw_participant(draw, pid_arrays[high_pid], start_ms, end_ms, HIGH_COLOR,
                          args.output_width, args.output_height, args.dot_radius)
        draw_hud(draw, args.video, high_pid, low_pid, r_mat[hi, w], r_mat[li, w],
                  mean_r[hi], mean_r[li], start_ms / 1000, end_ms / 1000, len(group_pids),
                  args.output_width, high_rank, low_rank, n_total)

        writer.stdin.write(np.array(Image.alpha_composite(base, overlay).convert("RGB")).tobytes())
        if frame_idx % int(round(fps)) == 0:
            print(f"  frame {frame_idx}/{n_frames}", end="\r")

    reader.stdout.close(), reader.wait()
    writer.stdin.close(), writer.wait()
    print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
