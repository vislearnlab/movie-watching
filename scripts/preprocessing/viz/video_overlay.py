"""
Shared ffmpeg frame I/O and gaze-density overlay helpers for the gaze video renderers
(visualize_entropy_segment.py). Moved out of visualize_concordance_video.py (now archived).
"""

import json
import subprocess

import numpy as np
from PIL import Image, ImageFont

CLOUD_COLOR = (70, 140, 255)   # blue: group density
CLOUD_MAX_ALPHA = 0.80
CLOUD_GAMMA = 0.65             # <1 lifts low-density areas so the cloud's extent stays visible
HUD_FONT_CANDIDATES = [
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
]


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


def cloud_overlay(field: np.ndarray, vmax: float, out_w: int, out_h: int) -> Image.Image:
    alpha = np.clip(field / vmax, 0, 1) ** CLOUD_GAMMA * CLOUD_MAX_ALPHA
    small = np.zeros((*field.shape, 4), dtype=np.uint8)
    small[..., 0], small[..., 1], small[..., 2] = CLOUD_COLOR
    small[..., 3] = (alpha * 255).astype(np.uint8)
    return Image.fromarray(small, mode="RGBA").resize((out_w, out_h), Image.BILINEAR)


def load_hud_font(size: int) -> ImageFont.ImageFont:
    for path in HUD_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
