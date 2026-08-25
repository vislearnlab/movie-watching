"""
Batch face-gaze analysis across all 4 annotated clips.

For each clip computes:
  - Per-participant face-looking proportion (all frames + face-present frames only)
  - Random prop baseline: participant P's gaze from clip C checked against clip D's
    face masks — any hit is spurious (same logic as ISC random baseline)
  - Overall ISC (adult-adult, infant-infant, adult-infant) + cross-clip random baseline
  - Windowed ISC over time + windowed cross-clip random baseline

Outputs (analysis/results/):
  face_proportions.csv          per-participant, per-clip proportions
  random_prop_baseline.csv      per-participant cross-clip random baseline
  isc_summary.csv               overall ISC per clip, comparison type, actual vs random
  isc_timeseries.csv            windowed ISC over time per clip, actual vs random
  clip_face_metadata.csv        face count / area stats per clip

Usage:
    python analysis/run_all_clips.py \\
        --adults_dir data/adults \\
        --infants_dir data/infants \\
        --seg_output_dir preprocessing/segmentation/output \\
        --output_dir analysis/results
"""

import argparse
import glob
import os
import sys
import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# Import core helpers from face_gaze_analysis
sys.path.insert(0, os.path.dirname(__file__))
from face_gaze_analysis import (
    find_participant_files,
    extract_trial_gaze,
    gaze_to_video_coords,
    load_face_masks,
    compute_participant_time_series,
)

CLIPS = ["frank_complex", "frank_objects", "frank_play", "sesameus_1"]

# Max cross-clip pairs to use for the random baseline (subsampled if exceeded)
MAX_RANDOM_PAIRS = 500
RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_group(pid: str) -> str:
    return "infant" if pid.startswith("HMET") else "adult"


def compute_gaze_positions(
    trial_gaze: pd.DataFrame,
    fps: float,
    total_frames: int,
    display_w: int,
    display_h: int,
    video_w: int,
    video_h: int,
    flip_y: bool,
) -> dict[int, tuple[float, float]]:
    """
    Return {frame_idx: (vx, vy)} — the first valid in-bounds gaze position per frame,
    in video pixel coordinates. Used to apply a different clip's masks for baselines.
    """
    valid = trial_gaze.dropna(subset=["trial_time", "gaze_x", "gaze_y"])
    if valid.empty:
        return {}

    times_ms = valid["trial_time"].to_numpy(dtype=float)
    gx = valid["gaze_x"].to_numpy(dtype=float)
    gy = valid["gaze_y"].to_numpy(dtype=float)
    vx, vy = gaze_to_video_coords(gx, gy, display_w, display_h, video_w, video_h, flip_y)

    frame_indices = np.clip(
        np.round(times_ms / (1000.0 / fps)).astype(int), 0, total_frames - 1
    )

    pos: dict[int, tuple[float, float]] = {}
    for i, fi in enumerate(frame_indices):
        if fi not in pos and not np.isnan(vx[i]) and not np.isnan(vy[i]):
            pos[fi] = (float(vx[i]), float(vy[i]))
    return pos


def prop_from_gaze_positions(
    gaze_pos: dict[int, tuple[float, float]],
    frame_masks: dict[int, np.ndarray],
    face_present_only: bool = False,
) -> float:
    """
    Compute proportion of gaze frames that hit a face mask.

    gaze_pos        : {frame_idx: (vx, vy)}
    frame_masks     : {frame_idx: bool HxW mask}
    face_present_only : if True, only count frames where a face mask exists
                        (denominator = frames where face was on screen and gaze was valid)
    """
    n_valid = 0
    n_face = 0
    for fi, (vx, vy) in gaze_pos.items():
        mask = frame_masks.get(fi)
        if face_present_only and mask is None:
            continue
        n_valid += 1
        if mask is not None:
            row, col = int(vy), int(vx)
            h, w = mask.shape
            if 0 <= row < h and 0 <= col < w and mask[row, col]:
                n_face += 1
    return n_face / n_valid if n_valid > 0 else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Clip annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_clip_annotation(seg_output_dir: str, clip_name: str) -> dict:
    """Load annotation frames CSV, summary CSV, and face masks for a clip."""
    clip_dir = os.path.join(seg_output_dir, clip_name)
    frames_csv = os.path.join(clip_dir, f"{clip_name}_face_frames.csv")
    summary_csv = os.path.join(clip_dir, f"{clip_name}_face_summary.csv")
    masks_npz = os.path.join(clip_dir, f"{clip_name}_face_masks.npz")

    annot_df = pd.read_csv(frames_csv)
    summary_df = pd.read_csv(summary_csv)

    video_w = int(annot_df["video_width"].iloc[0])
    video_h = int(annot_df["video_height"].iloc[0])
    total_frames = int(summary_df["total_frames"].iloc[0])
    fps = float(summary_df["fps"].iloc[0])

    print(f"  {clip_name}: {video_w}x{video_h} @ {fps:.2f}fps, {total_frames} frames")
    frame_masks = load_face_masks(masks_npz)

    return dict(
        clip_name=clip_name,
        video_w=video_w,
        video_h=video_h,
        total_frames=total_frames,
        fps=fps,
        frame_masks=frame_masks,
        annot_df=annot_df,
        summary_df=summary_df,
    )


def compute_clip_face_metadata(clip_info: dict) -> dict:
    """Compute per-clip face count and area stats for scatter plots."""
    annot_df = clip_info["annot_df"]
    total_frames = clip_info["total_frames"]
    fps = clip_info["fps"]
    summary_df = clip_info["summary_df"]

    frame_stats = (
        annot_df.groupby("frame_idx")
        .agg(n_faces=("obj_id", "nunique"), total_area_frac=("mask_area_frac", "sum"))
        .reset_index()
    )

    # Average over ALL frames (denominator includes frames with 0 faces)
    mean_n_faces = frame_stats["n_faces"].sum() / total_frames
    mean_area_frac = frame_stats["total_area_frac"].sum() / total_frames
    max_n_faces = int(frame_stats["n_faces"].max()) if not frame_stats.empty else 0

    duration_s = total_frames / fps

    return dict(
        clip=clip_info["clip_name"],
        total_frames=total_frames,
        fps=fps,
        duration_s=round(duration_s, 2),
        video_width=clip_info["video_w"],
        video_height=clip_info["video_h"],
        n_unique_objects=int(summary_df["num_unique_objects"].iloc[0]),
        pct_frames_with_faces=float(summary_df["pct_frames_with_detections"].iloc[0]),
        mean_n_faces_per_frame=round(mean_n_faces, 4),
        mean_face_area_frac_per_frame=round(mean_area_frac, 4),
        max_simultaneous_faces=max_n_faces,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Participant loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all_participant_timeseries(
    adults_dir: str,
    infants_dir: str,
    clips_data: dict,
    display_w: int,
    display_h: int,
    flip_y: bool,
) -> tuple[dict, dict, list]:
    """
    Load face-looking time series for every participant across every clip they watched.

    Returns
    -------
    time_series  : {clip_name: {pid: np.ndarray}}   binary face-looking per frame
    gaze_pos_all : {clip_name: {pid: {frame_idx: (vx, vy)}}}  raw gaze positions
    prop_records : list of dicts (one per participant × clip)
    """
    time_series = {clip: {} for clip in CLIPS}
    gaze_pos_all = {clip: {} for clip in CLIPS}
    prop_records = []

    for group_label, data_dir in [("adult", adults_dir), ("infant", infants_dir)]:
        participant_dirs = sorted(
            d
            for pattern in ("MW*", "HMET*")
            for d in glob.glob(os.path.join(data_dir, pattern))
            if os.path.isdir(d)
        )

        for pdir in participant_dirs:
            pid = os.path.basename(pdir)
            files = find_participant_files(pdir)
            if files["gaze"] is None:
                continue

            watched_videos: set[str] = set()
            if files["trial_order"] is not None:
                try:
                    tdf = pd.read_csv(files["trial_order"])
                    watched_videos = set(tdf["video_name"].tolist())
                except Exception:
                    pass

            gaze_df = None
            for enc in ("utf-8", "utf-16", "latin-1"):
                try:
                    gaze_df = pd.read_csv(files["gaze"], encoding=enc, low_memory=False)
                    break
                except Exception:
                    continue
            if gaze_df is None:
                print(f"  [{group_label}] {pid}: could not read gaze CSV — skipping")
                continue

            for clip_name, clip_info in clips_data.items():
                if watched_videos and clip_name not in watched_videos:
                    continue

                trial_gaze = extract_trial_gaze(gaze_df, clip_name)
                if trial_gaze is None or trial_gaze.empty:
                    continue

                ts = compute_participant_time_series(
                    trial_gaze=trial_gaze,
                    frame_masks=clip_info["frame_masks"],
                    fps=clip_info["fps"],
                    total_frames=clip_info["total_frames"],
                    display_w=display_w,
                    display_h=display_h,
                    video_w=clip_info["video_w"],
                    video_h=clip_info["video_h"],
                    flip_y=flip_y,
                )

                # Raw gaze positions — used later to apply other clips' masks
                gpos = compute_gaze_positions(
                    trial_gaze=trial_gaze,
                    fps=clip_info["fps"],
                    total_frames=clip_info["total_frames"],
                    display_w=display_w,
                    display_h=display_h,
                    video_w=clip_info["video_w"],
                    video_h=clip_info["video_h"],
                    flip_y=flip_y,
                )

                valid_mask = ~np.isnan(ts)
                n_valid = int(valid_mask.sum())
                n_face = int(ts[valid_mask].sum()) if n_valid > 0 else 0
                prop = n_face / n_valid if n_valid > 0 else np.nan

                # prop when face was actually on screen
                prop_when_present = prop_from_gaze_positions(
                    gpos, clip_info["frame_masks"], face_present_only=True
                )

                time_series[clip_name][pid] = ts
                gaze_pos_all[clip_name][pid] = gpos
                prop_records.append(
                    dict(
                        participant_id=pid,
                        group=group_label,
                        clip=clip_name,
                        n_total_frames=clip_info["total_frames"],
                        n_valid_frames=n_valid,
                        n_face_frames=n_face,
                        prop_looking_at_face=round(prop, 4) if not np.isnan(prop) else np.nan,
                        prop_looking_at_face_when_present=(
                            round(prop_when_present, 4)
                            if not np.isnan(prop_when_present) else np.nan
                        ),
                    )
                )
                prop_str = f"{prop:.3f}" if not np.isnan(prop) else "NaN"
                print(f"  [{group_label}] {pid} | {clip_name}: valid={n_valid}/{clip_info['total_frames']}, face={n_face}, prop={prop_str}")

    return time_series, gaze_pos_all, prop_records


# ─────────────────────────────────────────────────────────────────────────────
# Random proportion baseline
# ─────────────────────────────────────────────────────────────────────────────

def compute_random_prop_baseline(
    gaze_pos_all: dict,
    clips_data: dict,
) -> pd.DataFrame:
    """
    For each (participant P, clip C), compute the random baseline proportion by
    applying P's actual gaze positions from clip C against the face masks of every
    OTHER clip D — any "hit" is spurious since clip D has different face locations.

    Averages the cross-clip proportions across all D ≠ C.
    """
    records = []

    for clip_name in CLIPS:
        other_clips = [c for c in CLIPS if c != clip_name]
        for pid, gpos in gaze_pos_all[clip_name].items():
            cross_props = []
            for other_clip in other_clips:
                other_masks = clips_data[other_clip]["frame_masks"]
                p = prop_from_gaze_positions(gpos, other_masks, face_present_only=False)
                if not np.isnan(p):
                    cross_props.append(p)

            baseline = float(np.mean(cross_props)) if cross_props else np.nan
            records.append(
                dict(
                    participant_id=pid,
                    group=get_group(pid),
                    clip=clip_name,
                    baseline_prop=round(baseline, 4) if not np.isnan(baseline) else np.nan,
                    n_clips_in_baseline=len(cross_props),
                )
            )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Overall ISC
# ─────────────────────────────────────────────────────────────────────────────

def _isc_stats(r_arr: np.ndarray) -> dict:
    r_arr = r_arr[~np.isnan(r_arr)]
    if len(r_arr) == 0:
        return dict(n_pairs=0, mean_r=np.nan, sd_r=np.nan, min_r=np.nan, max_r=np.nan)
    return dict(
        n_pairs=int(len(r_arr)),
        mean_r=round(float(r_arr.mean()), 6),
        sd_r=round(float(r_arr.std()), 6),
        min_r=round(float(r_arr.min()), 6),
        max_r=round(float(r_arr.max()), 6),
    )


def _pearson_matrix(mat_a: np.ndarray, mat_b: np.ndarray, min_valid: int = 10) -> np.ndarray:
    """
    Vectorized Pearson r for N pairs simultaneously.

    mat_a, mat_b : (N, T) — NaN where gaze is missing.
    Returns (N,) array of r values (NaN where insufficient valid overlap).
    """
    valid = ~(np.isnan(mat_a) | np.isnan(mat_b))
    n = valid.sum(axis=1).astype(float)

    a = np.where(valid, mat_a, 0.0)
    b = np.where(valid, mat_b, 0.0)

    sum_a = a.sum(axis=1)
    sum_b = b.sum(axis=1)
    sum_aa = (a * a).sum(axis=1)
    sum_bb = (b * b).sum(axis=1)
    sum_ab = (a * b).sum(axis=1)

    num = n * sum_ab - sum_a * sum_b
    denom = np.sqrt(np.maximum((n * sum_aa - sum_a**2) * (n * sum_bb - sum_b**2), 0.0))

    ok = (n >= min_valid) & (denom > 1e-10)
    return np.where(ok, num / denom, np.nan)


def _subsample_pairs(pairs: list, max_pairs: int) -> list:
    """Randomly subsample pairs if there are too many."""
    if len(pairs) <= max_pairs:
        return pairs
    idx = RNG.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[i] for i in idx]


def _pairs_to_matrix(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Stack list of (ts_a, ts_b) pairs into (N, max_T) matrices, padding with NaN."""
    n = len(pairs)
    len_a = max(len(ts_a) for ts_a, _ in pairs)
    len_b = max(len(ts_b) for _, ts_b in pairs)
    mat_a = np.full((n, len_a), np.nan)
    mat_b = np.full((n, len_b), np.nan)
    for i, (ts_a, ts_b) in enumerate(pairs):
        mat_a[i, : len(ts_a)] = ts_a
        mat_b[i, : len(ts_b)] = ts_b
    return mat_a, mat_b


def compute_overall_isc(time_series: dict, clips_data: dict) -> pd.DataFrame:
    """
    Compute overall (whole-clip) ISC for each clip and comparison type,
    plus a cross-clip random baseline.

    random baseline: correlate participant A's time series in clip C with
    participant B's time series in a DIFFERENT clip D.
    """
    records = []

    for clip_name, ts_dict in time_series.items():
        adults = {pid: ts for pid, ts in ts_dict.items() if get_group(pid) == "adult"}
        infants = {pid: ts for pid, ts in ts_dict.items() if get_group(pid) == "infant"}
        other_clips = [c for c in CLIPS if c != clip_name]

        for comparison in ("adult_adult", "infant_infant", "adult_infant"):
            # ── Actual within-clip ────────────────────────────────────────────
            if comparison == "adult_adult":
                actual_pairs = [(ts_a, ts_b) for (_, ts_a), (_, ts_b) in combinations(adults.items(), 2)]
            elif comparison == "infant_infant":
                actual_pairs = [(ts_a, ts_b) for (_, ts_a), (_, ts_b) in combinations(infants.items(), 2)]
            else:
                actual_pairs = [(ts_a, ts_b) for _, ts_a in adults.items() for _, ts_b in infants.items()]

            if actual_pairs:
                mat_a, mat_b = _pairs_to_matrix(actual_pairs)
                n = min(mat_a.shape[1], mat_b.shape[1])
                r_arr = _pearson_matrix(mat_a[:, :n], mat_b[:, :n])
            else:
                r_arr = np.array([])
            records.append(dict(clip=clip_name, comparison=comparison, isc_type="actual",
                                **_isc_stats(r_arr)))

            # ── Cross-clip random baseline ────────────────────────────────────
            random_pairs: list = []
            for other_clip in other_clips:
                ts_other = time_series[other_clip]
                adults_o = {pid: ts for pid, ts in ts_other.items() if get_group(pid) == "adult"}
                infants_o = {pid: ts for pid, ts in ts_other.items() if get_group(pid) == "infant"}

                if comparison == "adult_adult":
                    random_pairs += [(ts_a, ts_b) for pid_a, ts_a in adults.items()
                                     for pid_b, ts_b in adults_o.items() if pid_a != pid_b]
                elif comparison == "infant_infant":
                    random_pairs += [(ts_a, ts_b) for pid_a, ts_a in infants.items()
                                     for pid_b, ts_b in infants_o.items() if pid_a != pid_b]
                else:
                    random_pairs += [(ts_a, ts_b) for _, ts_a in adults.items()
                                     for _, ts_b in infants_o.items()]
                    random_pairs += [(ts_a, ts_b) for _, ts_a in infants.items()
                                     for _, ts_b in adults_o.items()]

            random_pairs = _subsample_pairs(random_pairs, MAX_RANDOM_PAIRS)
            if random_pairs:
                mat_a, mat_b = _pairs_to_matrix(random_pairs)
                n = min(mat_a.shape[1], mat_b.shape[1])
                r_arr = _pearson_matrix(mat_a[:, :n], mat_b[:, :n])
            else:
                r_arr = np.array([])
            records.append(dict(clip=clip_name, comparison=comparison, isc_type="random",
                                **_isc_stats(r_arr)))

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Windowed ISC
# ─────────────────────────────────────────────────────────────────────────────

def _windowed_r_for_pairs(
    pair_list: list[tuple[np.ndarray, np.ndarray]],
    window_frames: int,
    step_frames: int,
    n_windows: int,
) -> list[dict]:
    """
    Vectorized windowed Pearson r across all pairs simultaneously.

    Stacks pairs into (N, T) matrices once, then slices per window —
    avoiding a Python loop over pairs inside the window loop.
    """
    if not pair_list:
        return []

    mat_a, mat_b = _pairs_to_matrix(pair_list)
    usable_len = min(mat_a.shape[1], mat_b.shape[1])

    results = []
    for w_idx in range(n_windows):
        start = w_idx * step_frames
        end = start + window_frames
        if end > usable_len:
            break
        r_arr = _pearson_matrix(mat_a[:, start:end], mat_b[:, start:end])
        valid_r = r_arr[~np.isnan(r_arr)]
        if len(valid_r) > 0:
            results.append(
                dict(
                    window_idx=w_idx,
                    mean_r=round(float(valid_r.mean()), 6),
                    sd_r=round(float(valid_r.std()), 6),
                    n_pairs=int(len(valid_r)),
                )
            )
    return results


def compute_windowed_isc(
    time_series: dict,
    clips_data: dict,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Compute windowed ISC (sliding window) for each clip and comparison type,
    plus a cross-clip random baseline (same window index, different clip content).

    Returns a long-format DataFrame with columns:
      clip, comparison, isc_type, window_idx, window_start_ms, window_end_ms,
      mean_r, sd_r, n_pairs
    """
    records = []

    for clip_name, clip_info in clips_data.items():
        fps = clip_info["fps"]
        total_frames = clip_info["total_frames"]
        window_frames = max(1, int(round(window_sec * fps)))
        step_frames = max(1, int(round(step_sec * fps)))
        n_windows = max(0, (total_frames - window_frames) // step_frames + 1)

        ts_dict = time_series[clip_name]
        adults = {pid: ts for pid, ts in ts_dict.items() if get_group(pid) == "adult"}
        infants = {pid: ts for pid, ts in ts_dict.items() if get_group(pid) == "infant"}

        other_clips = [c for c in CLIPS if c != clip_name]

        for comparison in ("adult_adult", "infant_infant", "adult_infant"):
            # ── Actual within-clip pairs ─────────────────────────────────────
            if comparison == "adult_adult":
                actual_pairs = [(ts_a, ts_b) for (_, ts_a), (_, ts_b) in combinations(adults.items(), 2)]
            elif comparison == "infant_infant":
                actual_pairs = [(ts_a, ts_b) for (_, ts_a), (_, ts_b) in combinations(infants.items(), 2)]
            else:
                actual_pairs = [(ts_a, ts_b) for _, ts_a in adults.items() for _, ts_b in infants.items()]

            for wr in _windowed_r_for_pairs(actual_pairs, window_frames, step_frames, n_windows):
                records.append(
                    dict(
                        clip=clip_name,
                        comparison=comparison,
                        isc_type="actual",
                        window_start_ms=round(wr["window_idx"] * step_frames / fps * 1000, 1),
                        window_end_ms=round((wr["window_idx"] * step_frames + window_frames) / fps * 1000, 1),
                        **wr,
                    )
                )

            # ── Cross-clip random baseline ────────────────────────────────────
            # For each window position W, correlate participant A's window W
            # in this clip vs participant B's window W in a different clip.
            # Window alignment by index means same time-offset into each clip.
            random_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for other_clip in other_clips:
                ts_other = time_series[other_clip]
                adults_other = {pid: ts for pid, ts in ts_other.items() if get_group(pid) == "adult"}
                infants_other = {pid: ts for pid, ts in ts_other.items() if get_group(pid) == "infant"}

                if comparison == "adult_adult":
                    random_pairs += [
                        (ts_a, ts_b)
                        for pid_a, ts_a in adults.items()
                        for pid_b, ts_b in adults_other.items()
                        if pid_a != pid_b
                    ]
                elif comparison == "infant_infant":
                    random_pairs += [
                        (ts_a, ts_b)
                        for pid_a, ts_a in infants.items()
                        for pid_b, ts_b in infants_other.items()
                        if pid_a != pid_b
                    ]
                else:  # adult_infant
                    random_pairs += [
                        (ts_a, ts_b)
                        for _, ts_a in adults.items()
                        for _, ts_b in infants_other.items()
                    ] + [
                        (ts_a, ts_b)
                        for _, ts_a in infants.items()
                        for _, ts_b in adults_other.items()
                    ]

            random_pairs = _subsample_pairs(random_pairs, MAX_RANDOM_PAIRS)
            for wr in _windowed_r_for_pairs(random_pairs, window_frames, step_frames, n_windows):
                records.append(
                    dict(
                        clip=clip_name,
                        comparison=comparison,
                        isc_type="random",
                        window_start_ms=round(wr["window_idx"] * step_frames / fps * 1000, 1),
                        window_end_ms=round((wr["window_idx"] * step_frames + window_frames) / fps * 1000, 1),
                        **wr,
                    )
                )

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch face-gaze analysis for all annotated clips.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adults_dir", default="data/adults")
    parser.add_argument("--infants_dir", default="data/infants")
    parser.add_argument("--seg_output_dir", default="preprocessing/segmentation/output")
    parser.add_argument("--output_dir", default="analysis/results")
    parser.add_argument("--display_width", type=int, default=1920)
    parser.add_argument("--display_height", type=int, default=1080)
    parser.add_argument("--no_flip_y", action="store_true")
    parser.add_argument("--window_sec", type=float, default=2.0, help="ISC sliding window length (seconds)")
    parser.add_argument("--step_sec", type=float, default=1.0, help="ISC sliding window step (seconds)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    flip_y = not args.no_flip_y

    # ── Load clip annotations ─────────────────────────────────────────────────
    print("=== Loading clip annotations ===")
    clips_data = {}
    for clip_name in CLIPS:
        clips_data[clip_name] = load_clip_annotation(args.seg_output_dir, clip_name)

    # ── Clip face metadata ────────────────────────────────────────────────────
    print("\n=== Computing clip face metadata ===")
    meta_records = [compute_clip_face_metadata(info) for info in clips_data.values()]
    meta_df = pd.DataFrame(meta_records)
    meta_path = os.path.join(args.output_dir, "clip_face_metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved → {meta_path}")
    print(meta_df[["clip", "mean_n_faces_per_frame", "mean_face_area_frac_per_frame",
                    "pct_frames_with_faces", "max_simultaneous_faces"]].to_string(index=False))

    # ── Load participant time series ──────────────────────────────────────────
    print("\n=== Loading participant data ===")
    time_series, gaze_pos_all, prop_records = load_all_participant_timeseries(
        adults_dir=args.adults_dir,
        infants_dir=args.infants_dir,
        clips_data=clips_data,
        display_w=args.display_width,
        display_h=args.display_height,
        flip_y=flip_y,
    )

    # ── Face proportions ──────────────────────────────────────────────────────
    prop_df = pd.DataFrame(prop_records)
    prop_path = os.path.join(args.output_dir, "face_proportions.csv")
    prop_df.to_csv(prop_path, index=False)
    print(f"\nSaved → {prop_path}  ({len(prop_df)} rows)")

    print("\n  Summary by clip and group:")
    for (clip, grp), sub in prop_df.groupby(["clip", "group"]):
        vals = sub["prop_looking_at_face"].dropna()
        print(f"    {clip} | {grp}: n={len(vals)}, mean={vals.mean():.3f}, sd={vals.std():.3f}")

    # ── Random proportion baseline ────────────────────────────────────────────
    print("\n=== Computing random proportion baseline ===")
    baseline_df = compute_random_prop_baseline(gaze_pos_all, clips_data)
    baseline_path = os.path.join(args.output_dir, "random_prop_baseline.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print(f"Saved → {baseline_path}  ({len(baseline_df)} rows)")

    # ── Overall ISC ───────────────────────────────────────────────────────────
    print("\n=== Computing overall ISC ===")
    isc_df = compute_overall_isc(time_series, clips_data)
    isc_path = os.path.join(args.output_dir, "isc_summary.csv")
    isc_df.to_csv(isc_path, index=False)
    print(f"Saved → {isc_path}")

    print("\n  Actual ISC by clip:")
    actual = isc_df[isc_df["isc_type"] == "actual"]
    print(actual[["clip", "comparison", "n_pairs", "mean_r", "sd_r"]].to_string(index=False))

    # ── Windowed ISC ──────────────────────────────────────────────────────────
    print(f"\n=== Computing windowed ISC ({args.window_sec}s window, {args.step_sec}s step) ===")
    isc_ts_df = compute_windowed_isc(
        time_series, clips_data,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )
    isc_ts_path = os.path.join(args.output_dir, "isc_timeseries.csv")
    isc_ts_df.to_csv(isc_ts_path, index=False)
    print(f"Saved → {isc_ts_path}  ({len(isc_ts_df)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
