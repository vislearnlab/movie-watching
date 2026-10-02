"""
Shared ffmpeg/ffprobe helpers for the stimulus_prep scripts in this directory.

Not a standalone script -- imported by stimulus_inventory.py and
representative_frames.py via:

    sys.path.insert(0, os.path.dirname(__file__))
    import _video_utils as vu

See METHODS_DECISIONS.md for why `_stripped.mp4` is the canonical "as displayed" video
(2026-09-25, "Video files used for Task 1 inventory").
"""

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def project_root() -> Path:
    """Repo root, resolved from this file's own location (scripts/preprocessing/stimulus_prep/)."""
    return Path(__file__).resolve().parents[3]


def check_ffmpeg_available() -> None:
    """Fail loudly rather than silently degrading -- every script here shells out to ffmpeg/ffprobe."""
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise RuntimeError(
                f"'{exe}' not found on PATH. Install ffmpeg (e.g. `brew install ffmpeg` on "
                "macOS, `apt-get install ffmpeg` on Linux) before continuing."
            )


def ffprobe_json(video_path: Path) -> dict:
    """Run ffprobe on a video and return its full metadata as a dict."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def get_video_metadata(video_path: Path) -> dict:
    """Native width/height, exact fps (as a fraction and as a decimal), and duration."""
    info = ffprobe_json(video_path)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")

    width, height = int(vstream["width"]), int(vstream["height"])

    fps_fraction = vstream.get("r_frame_rate", "0/1")
    num, den = fps_fraction.split("/")
    fps_decimal = float(num) / float(den) if float(den) != 0 else float(num)

    # Prefer container duration; fall back to the video stream's own duration field.
    duration = info["format"].get("duration") or vstream.get("duration")
    duration_sec = float(duration)

    return {
        "native_width": width,
        "native_height": height,
        "fps_fraction": fps_fraction,
        "fps_decimal": round(fps_decimal, 4),
        "duration_sec": round(duration_sec, 3),
    }


def block_from_filename(video_name: str, block_prefix_map: dict) -> str | None:
    """Look up a clip's block_id from its filename prefix (e.g. 'slow_hands' -> 'slow')."""
    for prefix, block_id in block_prefix_map.items():
        if video_name.startswith(prefix):
            return block_id
    return None


def _cropdetect_window(video_path: Path, start_sec: float, window_sec: float, limit: int) -> list[tuple]:
    """Run ffmpeg's cropdetect over one time window and return every crop=W:H:X:Y match."""
    cmd = [
        "ffmpeg", "-nostdin", "-ss", str(start_sec), "-i", str(video_path),
        "-t", str(window_sec),
        "-vf", f"cropdetect=limit={limit}:round=2:reset=1",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return _CROP_RE.findall(result.stderr)


def detect_letterbox_pillarbox(video_path: Path, native_width: int, native_height: int,
                                duration_sec: float, *, limit: int, n_samples: int,
                                sample_sec: float, bar_frac_threshold: float) -> dict:
    """
    Detect black bars baked into the video content by sampling ffmpeg's cropdetect at
    several points across the clip and taking the most frequently reported crop rect
    (robust to a few dark scenes being mistaken for bars).

    Bar fractions are relative to the *native* frame; because PsychoPy stretches the
    source frame uniformly to fill the display size (METHODS_DECISIONS.md, 2026-09-25), a bar
    occupying frac_x of native width occupies the same frac_x of displayed width.
    """
    window_sec = min(sample_sec, max(duration_sec - 0.1, 0.1))

    starts = [
        float(max(0.0, min(s, max(duration_sec - window_sec, 0.0))))
        for s in _linspace(0.1 * duration_sec, max(0.9 * duration_sec - window_sec, 0.1), n_samples)
    ]
    matches = []
    for start in starts:
        matches.extend(_cropdetect_window(video_path, start, window_sec, limit))

    if not matches:
        # No cropdetect output at all (e.g. a fully black sampled window) -- report as
        # "full frame, nothing detected" rather than guessing.
        w, h, x, y = native_width, native_height, 0, 0
    else:
        counts = Counter(matches)
        (w_s, h_s, x_s, y_s), _ = counts.most_common(1)[0]
        w, h, x, y = int(w_s), int(h_s), int(x_s), int(y_s)

    left_frac = x / native_width
    right_frac = (native_width - (x + w)) / native_width
    top_frac = y / native_height
    bottom_frac = (native_height - (y + h)) / native_height

    return {
        "crop_w": w, "crop_h": h, "crop_x": x, "crop_y": y,
        "left_bar_frac": round(left_frac, 4),
        "right_bar_frac": round(right_frac, 4),
        "top_bar_frac": round(top_frac, 4),
        "bottom_bar_frac": round(bottom_frac, 4),
        "is_pillarboxed": (left_frac + right_frac) > bar_frac_threshold,
        "is_letterboxed": (top_frac + bottom_frac) > bar_frac_threshold,
        "cropdetect_n_samples_used": n_samples,
    }


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Tiny stand-in for numpy.linspace so this module has no numpy dependency."""
    if n == 1:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def extract_frame(video_path: Path, timestamp_sec: float, out_path: Path,
                   width: int | None = None, height: int | None = None) -> Path:
    """Extract a single frame at `timestamp_sec` and save it as a PNG, optionally resized."""
    cmd = ["ffmpeg", "-y", "-nostdin", "-ss", str(timestamp_sec), "-i", str(video_path), "-frames:v", "1"]
    if width and height:
        cmd += ["-vf", f"scale={width}:{height}"]
    cmd += [str(out_path)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


def extract_display_frame(video_path: Path, timestamp_sec: float, out_path: Path,
                           display_width: int, display_height: int) -> Path:
    """Extract a frame exactly as displayed to participants (stretched to the display size)."""
    return extract_frame(video_path, timestamp_sec, out_path, width=display_width, height=display_height)
