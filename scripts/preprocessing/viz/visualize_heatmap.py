"""
Generate frame-by-frame gaze density heatmap videos aggregated across participants.

For each output video frame, gaze samples from all selected participants that
fall within that frame's time window are collected, converted to a 2-D density
map, smoothed, and colourised.  The result is an MP4 where you can watch where
the group was looking as the video plays.

Design choices (documented)
----------------------------
Frame rate
    Output video is written at the stimulus video's native frame rate (~30 fps).
    Each output frame covers one stimulus frame's worth of time.

Time window per frame
    By default each frame accumulates gaze from exactly one frame-duration
    (≈ 33 ms at 30 fps).  Pass --window_ms to widen it (e.g. 100 ms) for
    sparser groups such as infants — wider windows give more points per frame
    at the cost of temporal blur.

Per-participant weighting
    Each participant contributes at most one (x, y) point per frame: the mean
    of their valid gaze samples within the time window.  This gives every
    participant equal weight regardless of how many valid samples they happened
    to have in that frame.

Valid-data filter
    Only participants with valid_data=TRUE in participant_summary.csv are
    included.  Pass --include_invalid to override.

Coordinate system
    gaze_x / gaze_y are in display pixels (default 1920 × 1080).
    The Tobii y-axis is inverted (origin bottom-left), so Y is flipped before
    mapping to output pixel coordinates.

Point radius
    Before Gaussian blurring, each participant's gaze contribution is drawn as
    a filled circle of --point_radius pixels (default 0 = single pixel).
    Increasing point_radius makes individual fixations more visible and gives
    a harder-edged look before smoothing.

Gaussian blur
    Applied per frame after drawing gaze points.  --sigma sets the radius in
    output pixels (default 30).  Works in combination with --point_radius:
    effective visual size ≈ point_radius + sigma.

Normalisation  (--norm)
    Controls how the density map is scaled to [0, 1] before colouring.

    global (default)
        Two-pass: the 99th-percentile of per-frame blurred maxima across the
        whole video is used as the reference maximum.  Colour encodes both
        WHERE people looked and HOW concentrated attention was at each moment.
        A frame where gaze is scattered will appear cool/blue; a frame where
        everyone looks at the same spot will appear hot/red.  Robust to single
        outlier spikes (hence 99th percentile rather than absolute max).
        Requires computing all frames twice — roughly 2× the processing time.

    per_frame
        Each frame is normalised to its own maximum density.  The colourmap
        always uses its full range regardless of how concentrated attention is,
        so scattered frames look just as colourful as concentrated ones.
        Useful when you care only about the spatial distribution within each
        frame, not about changes in concentration over time.

    n_participants
        Density is divided by the number of participants with valid gaze for
        that video.  Full red = every participant looking at the same pixel;
        scattered gaze produces low, cool values.  Similar to global but the
        scale is interpretable (fraction of participants) and requires only
        a single pass.

Colormap
    "jet" by default (blue → green → yellow → red; red = highest density).
    Any matplotlib colormap name is accepted via --colormap.

Background overlay
    Pass --background to blend the heatmap transparently over each video frame.
    --alpha (default 0.6) sets the peak heatmap opacity.  Pixels with near-zero
    density become fully transparent.  Only the stripped video is used.

Output
    One MP4 per (video, participant-group), written to --output_dir
    (default preprocessing/viz/heatmaps/).
    Filename: {video_name}_{group_label}_heatmap.mp4

Usage examples
--------------
    # All adults, all videos, global normalisation (default)
    python preprocessing/viz/visualize_heatmap.py \\
        --videos all --participants adults

    # Kids + infants combined, wider window, overlaid on video
    python preprocessing/viz/visualize_heatmap.py \\
        --videos sesameus_1 --participants kids infants \\
        --window_ms 100 --background

    # Per-frame norm, larger gaze dots
    python preprocessing/viz/visualize_heatmap.py \\
        --videos pixar_birds --participants adults \\
        --norm per_frame --point_radius 15 --sigma 20

    # N-participants norm (single pass, interpretable scale)
    python preprocessing/viz/visualize_heatmap.py \\
        --videos all --participants adults --norm n_participants
"""

import argparse
import glob
import os
import re
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ── defaults ───────────────────────────────────────────────────────────────────

DISPLAY_WIDTH  = 1920
DISPLAY_HEIGHT = 1080
OUTPUT_WIDTH   = 1280
OUTPUT_HEIGHT  = 720
SIGMA          = 30     # Gaussian blur radius in output pixels
POINT_RADIUS   = 0      # per-participant gaze dot radius before blurring (0 = single pixel)
COLORMAP       = "jet"
ALPHA          = 0.6    # peak heatmap opacity when blending over video frame

GROUP_ALIASES = {"adults", "kids", "infants"}


# ── file helpers ───────────────────────────────────────────────────────────────

def find_participant_files(participant_dir: str) -> dict:
    csvs = glob.glob(os.path.join(participant_dir, "*.csv"))
    files = {"gaze": None}
    for f in csvs:
        bn = os.path.basename(f)
        if bn.startswith("."):
            continue
        if not (bn.endswith("_trial_order.csv")
                or bn.endswith("_validation_summary.csv")
                or bn.endswith("_validation.csv")):
            files["gaze"] = f
    return files


def read_gaze_csv(path: str) -> pd.DataFrame | None:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return None


def find_video_path(video_name: str, stimuli_dir: Path) -> str | None:
    stripped = stimuli_dir / f"{video_name}_stripped.mp4"
    if stripped.exists():
        return str(stripped)
    plain = stimuli_dir / f"{video_name}.mp4"
    if plain.exists():
        return str(plain)
    return None


# ── participant resolution ─────────────────────────────────────────────────────

def resolve_participants(
    raw_dir: Path,
    participant_args: list[str],
    valid_ids: set[str],
) -> dict[str, list[Path]]:
    """
    Return {group_label: [participant_dir, ...]} for the requested participants.

    participant_args can mix group names ("adults", "kids", "infants") and
    specific IDs ("MW001").  Only participants present in valid_ids are included.
    """
    group_dirs = {
        "adults":  raw_dir / "adults",
        "kids":    raw_dir / "kids",
        "infants": raw_dir / "infants",
    }

    all_on_disk: dict[str, tuple[str, Path]] = {}
    for group, gdir in group_dirs.items():
        if not gdir.exists():
            continue
        for d in sorted(gdir.iterdir()):
            if d.is_dir():
                all_on_disk[d.name.upper()] = (group, d)

    selected: dict[str, list[Path]] = {}

    groups_req = [a for a in participant_args if a.lower() in GROUP_ALIASES]
    ids_req    = [a for a in participant_args if a.lower() not in GROUP_ALIASES]

    for grp_arg in groups_req:
        grp  = grp_arg.lower()
        gdir = group_dirs[grp]
        if not gdir.exists():
            print(f"  WARNING: No directory for group '{grp}' at {gdir}")
            continue
        dirs = [d for d in sorted(gdir.iterdir()) if d.is_dir() and d.name in valid_ids]
        selected.setdefault(grp, []).extend(dirs)

    if ids_req:
        dirs = []
        for pid in ids_req:
            match = all_on_disk.get(pid.upper())
            if match is None:
                print(f"  WARNING: Participant '{pid}' not found.")
                continue
            _, pdir = match
            if pdir.name not in valid_ids:
                print(f"  WARNING: '{pid}' excluded (valid_data=FALSE).")
                continue
            dirs.append(pdir)
        if dirs:
            label = "_".join(p.name for p in dirs[:3])
            if len(dirs) > 3:
                label += f"_and{len(dirs)-3}more"
            selected[label] = dirs

    return selected


# ── gaze extraction ────────────────────────────────────────────────────────────

def extract_trial_gaze(gaze_df: pd.DataFrame, video_name: str) -> pd.DataFrame | None:
    """Return gaze rows for the named video trial, trial_time reset to 0 ms."""
    events     = gaze_df["events"].astype(str)
    start_mask = events.str.contains(rf"Video_{re.escape(video_name)}$", regex=True)
    if not start_mask.any():
        return None

    start_idx = gaze_df.index[start_mask][0]
    m = re.match(r"Trial_Start_(\d+)\|", gaze_df.loc[start_idx, "events"])
    if not m:
        return None
    trial_num = m.group(1)

    end_rows = gaze_df.index[events == f"Trial_End_{trial_num}"]
    if len(end_rows) == 0:
        return None

    trial_df = gaze_df.loc[start_idx : end_rows[0]].copy()
    first_t  = trial_df["trial_time"].dropna()
    if not first_t.empty:
        trial_df["trial_time"] = trial_df["trial_time"] - first_t.iloc[0]
    return trial_df


def load_all_participant_gaze(
    participant_dirs: list[Path],
    video_name: str,
    display_width: int,
    display_height: int,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pre-load cleaned gaze time series for all participants for one video.

    Returns list of (participant_id, trial_time_ms, gaze_x_px, gaze_y_px).
    Invalid samples (both eyes invalid, NaN, or out-of-display) are masked to
    NaN so they are excluded during per-frame averaging.
    """
    series = []
    for pdir in participant_dirs:
        files = find_participant_files(str(pdir))
        if files["gaze"] is None:
            print(f"  SKIP {pdir.name}: no gaze CSV")
            continue
        gaze_df = read_gaze_csv(files["gaze"])
        if gaze_df is None:
            print(f"  SKIP {pdir.name}: could not read gaze CSV")
            continue
        trial_df = extract_trial_gaze(gaze_df, video_name)
        if trial_df is None:
            continue  # participant did not watch this video

        df = trial_df[trial_df["trial_time"].notna() & (trial_df["trial_time"] >= 0)].copy()
        if df.empty:
            continue

        left_v  = pd.to_numeric(df.get("left_valid",  0), errors="coerce").fillna(0)
        right_v = pd.to_numeric(df.get("right_valid", 0), errors="coerce").fillna(0)
        gx = pd.to_numeric(df["gaze_x"], errors="coerce")
        gy = pd.to_numeric(df["gaze_y"], errors="coerce")

        no_eye = (left_v == 0) & (right_v == 0)
        gx = gx.where(~no_eye)
        gy = gy.where(~no_eye)

        in_bounds = (gx >= 0) & (gx <= display_width) & (gy >= 0) & (gy <= display_height)
        gx = gx.where(in_bounds)
        gy = gy.where(in_bounds)

        tt = pd.to_numeric(df["trial_time"], errors="coerce").to_numpy(dtype=np.float64)
        series.append((pdir.name, tt, gx.to_numpy(dtype=np.float64), gy.to_numpy(dtype=np.float64)))

    return series


# ── per-frame density ──────────────────────────────────────────────────────────

def frame_density(
    participant_series: list[tuple],
    t_start_ms: float,
    t_end_ms: float,
    display_width: int,
    display_height: int,
    output_width: int,
    output_height: int,
    point_radius: int = 0,
) -> np.ndarray | None:
    """
    Build a 2-D density map for one frame's time window.

    Each participant contributes one averaged (x, y) point for the window.
    If point_radius > 0, that point is drawn as a filled circle rather than
    a single pixel — giving more visual weight to individual fixations before
    Gaussian blurring.

    Returns float32 array (output_height, output_width), or None if no
    participant had valid gaze in this window.
    """
    density = np.zeros((output_height, output_width), dtype=np.float32)
    n = 0

    for _, tt, gx, gy in participant_series:
        mask = (tt >= t_start_ms) & (tt < t_end_ms)
        if not mask.any():
            continue
        win_gx = gx[mask]
        win_gy = gy[mask]
        valid  = ~(np.isnan(win_gx) | np.isnan(win_gy))
        if not valid.any():
            continue

        mean_x = win_gx[valid].mean()
        mean_y = win_gy[valid].mean()

        # Scale display → output pixels, flip Y (Tobii: bottom-left origin)
        px = int(np.clip(mean_x * output_width  / display_width,  0, output_width  - 1))
        py = int(np.clip((display_height - mean_y) * output_height / display_height, 0, output_height - 1))

        if point_radius > 0:
            cv2.circle(density, (px, py), point_radius, 1.0, thickness=-1)
        else:
            density[py, px] += 1.0
        n += 1

    return density if n > 0 else None


# ── rendering ──────────────────────────────────────────────────────────────────

def render_frame(
    density: np.ndarray | None,
    sigma: float,
    cmap_fn,
    bg_frame: np.ndarray | None,
    alpha: float,
    output_width: int,
    output_height: int,
    norm_value: float | None = None,
) -> np.ndarray:
    """
    Blur, normalise, colourise, and optionally blend over a background frame.

    norm_value controls scaling:
        None            → per-frame: normalise to frame's own max
        positive float  → divide by this value (global max or n_participants)

    Returns uint8 BGR image.
    """
    if density is None or density.max() == 0:
        if bg_frame is not None:
            return cv2.resize(bg_frame, (output_width, output_height))
        return np.zeros((output_height, output_width, 3), dtype=np.uint8)

    blurred = gaussian_filter(density, sigma=sigma)

    if norm_value is None:
        local_max = blurred.max()
        norm = blurred / local_max if local_max > 0 else blurred
    else:
        norm = np.clip(blurred / norm_value, 0.0, 1.0)

    rgba     = (cmap_fn(norm) * 255).astype(np.uint8)
    heat_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    if bg_frame is None:
        return heat_bgr

    bg = cv2.resize(bg_frame, (output_width, output_height))
    alpha_map = (norm * alpha).clip(0, 1)[:, :, np.newaxis]
    return ((1.0 - alpha_map) * bg + alpha_map * heat_bgr).astype(np.uint8)


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate frame-by-frame gaze heatmap videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos", nargs="+", required=True, metavar="VIDEO",
        help="Video name(s) or 'all'. E.g. --videos sesameus_1 pixar_birds",
    )
    parser.add_argument(
        "--participants", nargs="+", required=True, metavar="GROUP_OR_ID",
        help=(
            "Group(s) or specific ID(s). Groups: adults, kids, infants. "
            "Multiple groups are merged. E.g. --participants adults  or  --participants MW001 MW002"
        ),
    )
    parser.add_argument(
        "--norm", choices=["global", "per_frame", "n_participants"], default="global",
        help=(
            "Normalisation mode. "
            "'global': normalise by 99th-percentile of per-frame max across the video (two-pass, ~2x slower). "
            "'per_frame': normalise each frame to its own max (spatial distribution only, no concentration info). "
            "'n_participants': normalise by number of contributing participants (single pass, interpretable scale)."
        ),
    )
    parser.add_argument(
        "--point_radius", type=int, default=POINT_RADIUS,
        help=(
            "Radius in output pixels of each participant's gaze contribution before Gaussian blur. "
            "0 = single pixel (blur alone controls spread). "
            "Larger values give harder-edged dots."
        ),
    )
    parser.add_argument(
        "--output_dir", default="preprocessing/viz/heatmaps",
        help="Output directory (relative to project root).",
    )
    parser.add_argument(
        "--metadata", default="data/metadata/participant_summary.csv",
        help="Participant summary CSV (relative to project root).",
    )
    parser.add_argument(
        "--stimuli_dir", default="stimuli/main_blocks",
        help="Directory containing stimulus MP4 files (relative to project root).",
    )
    parser.add_argument(
        "--display_width",  type=int, default=DISPLAY_WIDTH,
        help="Display width used during experiment (px).",
    )
    parser.add_argument(
        "--display_height", type=int, default=DISPLAY_HEIGHT,
        help="Display height used during experiment (px).",
    )
    parser.add_argument(
        "--output_width",  type=int, default=OUTPUT_WIDTH,
        help="Output video width (px).",
    )
    parser.add_argument(
        "--output_height", type=int, default=OUTPUT_HEIGHT,
        help="Output video height (px).",
    )
    parser.add_argument(
        "--window_ms", type=float, default=None,
        help=(
            "Time window (ms) of gaze accumulated per frame. "
            "Defaults to one frame duration (1000/fps ≈ 33 ms at 30 fps). "
            "Widen for sparse groups (kids, infants)."
        ),
    )
    parser.add_argument(
        "--sigma", type=float, default=SIGMA,
        help="Gaussian blur radius in output pixels.",
    )
    parser.add_argument(
        "--colormap", default=COLORMAP,
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--background", action="store_true",
        help="Blend heatmap over each video frame (uses stripped .mp4).",
    )
    parser.add_argument(
        "--alpha", type=float, default=ALPHA,
        help="Peak heatmap opacity when using --background (0–1).",
    )
    parser.add_argument(
        "--include_invalid", action="store_true",
        help="Include participants whose valid_data flag is FALSE.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    raw_dir       = PROJECT_ROOT / "data" / "raw"
    stimuli_dir   = PROJECT_ROOT / args.stimuli_dir
    output_dir    = PROJECT_ROOT / args.output_dir
    metadata_path = PROJECT_ROOT / args.metadata

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── participant metadata ───────────────────────────────────────────────────
    if not metadata_path.exists():
        sys.exit(f"ERROR: Metadata not found: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    meta["valid_data"] = meta["valid_data"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
    valid_ids = (
        set(meta["participant_id"].astype(str))
        if args.include_invalid
        else set(meta.loc[meta["valid_data"], "participant_id"].astype(str))
    )
    print(f"Valid participants in metadata: {len(valid_ids)}")

    # ── video list ─────────────────────────────────────────────────────────────
    if len(args.videos) == 1 and args.videos[0].lower() == "all":
        video_names = [p.stem.replace("_stripped", "") for p in sorted(stimuli_dir.glob("*_stripped.mp4"))]
    else:
        video_names = args.videos

    if not video_names:
        sys.exit("ERROR: No videos to process.")

    # ── participant groups ─────────────────────────────────────────────────────
    group_map   = resolve_participants(raw_dir, args.participants, valid_ids)
    if not group_map:
        sys.exit("ERROR: No valid participants found.")

    group_label = "_".join(sorted(group_map.keys()))
    all_dirs    = [d for dirs in group_map.values() for d in dirs]

    cmap_fn = plt.colormaps[args.colormap]

    print(f"\nGroup label     : {group_label}")
    print(f"Participants    : {len(all_dirs)}")
    print(f"Videos          : {len(video_names)}")
    print(f"Norm mode       : {args.norm}")
    print(f"Point radius    : {args.point_radius} px")
    print(f"Gaussian sigma  : {args.sigma} px")
    print(f"Output dir      : {output_dir.relative_to(PROJECT_ROOT)}\n")

    # ── process each video ────────────────────────────────────────────────────
    for video_name in video_names:
        print(f"{'─'*60}")
        print(f"Video: {video_name}  |  group: {group_label}")

        vpath = find_video_path(video_name, stimuli_dir)
        if vpath is None and args.background:
            print(f"  WARNING: Video file not found — skipping background.")

        cap_probe = cv2.VideoCapture(vpath) if vpath else None
        fps       = cap_probe.get(cv2.CAP_PROP_FPS) if cap_probe else 30.0
        n_frames  = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT)) if cap_probe else None
        if cap_probe:
            cap_probe.release()

        frame_dur_ms = 1000.0 / fps
        window_ms    = args.window_ms if args.window_ms is not None else frame_dur_ms

        print(f"  fps={fps:.2f}  frame_dur={frame_dur_ms:.1f} ms  window={window_ms:.1f} ms")

        print(f"  Loading participant gaze...")
        participant_series = load_all_participant_gaze(
            all_dirs, video_name, args.display_width, args.display_height
        )
        n_contrib = len(participant_series)
        print(f"  Contributors: {n_contrib}")

        if n_contrib == 0:
            print(f"  SKIP: no valid gaze data found.")
            continue

        if n_frames is None or n_frames <= 0:
            max_t    = max(tt.max() for _, tt, _, _ in participant_series if len(tt) > 0)
            n_frames = int(np.ceil(max_t / frame_dur_ms))

        print(f"  Total frames: {n_frames}")

        # ── resolve norm_value ─────────────────────────────────────────────────
        norm_value: float | None

        if args.norm == "n_participants":
            norm_value = float(n_contrib)
            print(f"  Norm: n_participants = {n_contrib}")

        elif args.norm == "global":
            # Pass 1: collect blurred-frame maxima, find 99th percentile
            print(f"  Norm: global — pass 1/2 (computing frame maxima)...")
            frame_maxes = []
            for frame_idx in tqdm(range(n_frames), desc="  pass 1", unit="fr", leave=False):
                t_start = frame_idx * frame_dur_ms
                density = frame_density(
                    participant_series, t_start, t_start + window_ms,
                    args.display_width, args.display_height,
                    args.output_width,  args.output_height,
                    point_radius=args.point_radius,
                )
                if density is not None:
                    blurred = gaussian_filter(density, sigma=args.sigma)
                    frame_maxes.append(blurred.max())

            if not frame_maxes:
                print(f"  SKIP: all frames empty.")
                continue

            norm_value = float(np.percentile(frame_maxes, 99))
            if norm_value == 0:
                norm_value = float(max(frame_maxes))

            # Approximate peak value a single participant contributes after blur.
            # For point_radius=0: a spike of 1 spread by Gaussian sigma → peak ≈ 1/(2π·σ²).
            # This lets us express the norm in interpretable "effective participants" units.
            single_p_peak = 1.0 / (2 * np.pi * args.sigma ** 2)
            implied_n_at_99 = norm_value / single_p_peak

            n_empty = n_frames - len(frame_maxes)
            print(f"  Pass 1 stats:")
            print(f"    frames with data : {len(frame_maxes)}/{n_frames}  ({n_empty} empty)")
            print(f"    blurred max range: min={min(frame_maxes):.5f}  "
                  f"mean={np.mean(frame_maxes):.5f}  "
                  f"99th={norm_value:.5f}  "
                  f"abs_max={max(frame_maxes):.5f}")
            print(f"    single-participant peak (σ={args.sigma}px): {single_p_peak:.5f}")
            print(f"    99th pct ≈ {implied_n_at_99:.1f}/{n_contrib} participants co-fixating")
            print(f"  Norm: global 99th-pct max = {norm_value:.5f}  (pass 2/2 rendering...)")

        else:  # per_frame
            norm_value = None
            print(f"  Norm: per_frame")

        # ── render pass ────────────────────────────────────────────────────────
        out_name = f"{video_name}_{group_label}_heatmap.mp4"
        out_path = output_dir / out_name
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps, (args.output_width, args.output_height))

        cap = cv2.VideoCapture(vpath) if (args.background and vpath) else None

        desc = "  pass 2" if args.norm == "global" else "  rendering"
        for frame_idx in tqdm(range(n_frames), desc=desc, unit="fr"):
            t_start = frame_idx * frame_dur_ms
            density = frame_density(
                participant_series, t_start, t_start + window_ms,
                args.display_width, args.display_height,
                args.output_width,  args.output_height,
                point_radius=args.point_radius,
            )

            bg_frame = None
            if cap:
                ret, bg_frame = cap.read()
                if not ret:
                    bg_frame = None

            out_frame = render_frame(
                density, args.sigma, cmap_fn, bg_frame, args.alpha,
                args.output_width, args.output_height,
                norm_value=norm_value,
            )
            writer.write(out_frame)

        writer.release()
        if cap:
            cap.release()
        print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
