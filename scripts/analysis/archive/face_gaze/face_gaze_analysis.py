"""
Face-Looking Proportion and Inter-Subject Correlation (ISC) Analysis

Given SAM 3 face annotations for a video and eyetracking data from participants,
this script computes:

  1. Proportion of time each participant spent looking at a face region.
  2. ISC (pairwise Pearson correlation of face-looking time series):
       - Adult–adult ISC
       - Adult–infant ISC

Inputs
------
- Face annotation CSV produced by annotate_video.py
  (columns: video_name, frame_idx, time_ms, obj_id, bbox_x, bbox_y, bbox_w, bbox_h, …)
- Per-participant eyetracking CSVs in data/adults/ and data/infants/

Coordinate mapping
------------------
Gaze is recorded in display pixel coordinates. The video may be scaled or
centered on the display. This script assumes the video is presented
**full-screen** (scaled to fill the display) by default. Adjust
--display_width / --display_height to match the actual monitor resolution.

The Tobii coordinate system used in this experiment has Y=0 at the BOTTOM of
the screen (y increases upward), which is inverted relative to image coordinates
(y=0 at top). Use --no_flip_y if your Tobii data already uses top-left origin.

Usage
-----
    python analysis/face_gaze_analysis.py \
        --annotation_csv preprocessing/segmentation/output/sesameus_1_stripped_face_frames.csv \
        --video_name sesameus_1_stripped \
        --adults_dir data/adults \
        --infants_dir data/infants \
        --display_width 1920 \
        --display_height 1080 \
        --output_dir analysis/results
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Gaze data helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_participant_files(participant_dir: str) -> dict:
    """Return paths to gaze and trial-order CSVs for a participant folder."""
    csvs = glob.glob(os.path.join(participant_dir, "*.csv"))
    files = {"gaze": None, "trial_order": None}
    for f in csvs:
        bn = os.path.basename(f)
        if bn.endswith("_trial_order.csv") and not bn.startswith("."):
            files["trial_order"] = f
        elif bn.endswith("_validation_summary.csv") or bn.endswith("_validation.csv"):
            pass  # not needed here
        elif bn.count("_") == 2 and bn.endswith(".csv") and not bn.startswith("."):
            files["gaze"] = f
    return files


def extract_trial_gaze(gaze_df: pd.DataFrame, video_name: str) -> pd.DataFrame | None:
    """
    Extract gaze rows belonging to a specific video trial.

    Looks for 'Trial_Start_N|Video_{video_name}' events and returns the gaze
    data between Trial_Start and Trial_End for that trial, with trial_time
    reset to start at 0.
    """
    # The video_name in events may omit the _stripped suffix — try both
    candidate_names = [video_name]
    if video_name.endswith("_stripped"):
        candidate_names.append(video_name[: -len("_stripped")])

    events = gaze_df["events"].astype(str)
    start_mask = None
    for cname in candidate_names:
        m = events.str.contains(f"Video_{cname}$", regex=True)
        if m.any():
            start_mask = m
            break

    if start_mask is None or not start_mask.any():
        return None

    start_row_idx = gaze_df.index[start_mask][0]

    # Extract the trial index from the event string
    event_str = gaze_df.loc[start_row_idx, "events"]
    # e.g. "Trial_Start_2|Video_sesameus_1"
    trial_num = event_str.split("_")[2].split("|")[0]
    end_label = f"Trial_End_{trial_num}"

    end_rows = gaze_df.index[events == end_label]
    if len(end_rows) == 0:
        return None

    end_row_idx = end_rows[0]
    trial_df = gaze_df.loc[start_row_idx:end_row_idx].copy()

    # Ensure trial_time is relative to the start of this trial
    if "trial_time" in trial_df.columns:
        first_valid_t = trial_df["trial_time"].dropna()
        if not first_valid_t.empty:
            trial_df["trial_time"] = trial_df["trial_time"] - first_valid_t.iloc[0]

    return trial_df


def gaze_to_video_coords(
    gaze_x: np.ndarray,
    gaze_y: np.ndarray,
    display_w: int,
    display_h: int,
    video_w: int,
    video_h: int,
    flip_y: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map display-space gaze coordinates to video-frame pixel coordinates.

    Assumes the video fills the full display (uniform scaling). Clips results
    to [0, video_w-1] × [0, video_h-1]; out-of-bounds entries become NaN.

    flip_y=True: Tobii Y=0 is at bottom; flip to image Y=0 at top.
    """
    if flip_y:
        gaze_y = display_h - gaze_y

    vx = gaze_x * (video_w / display_w)
    vy = gaze_y * (video_h / display_h)

    # Mark out-of-bounds as NaN
    out_of_bounds = (vx < 0) | (vx >= video_w) | (vy < 0) | (vy >= video_h)
    vx = np.where(out_of_bounds, np.nan, vx)
    vy = np.where(out_of_bounds, np.nan, vy)

    return vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# Face-region lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_face_masks(masks_npz_path: str) -> dict[int, np.ndarray]:
    """
    Load the combined face masks saved by annotate_video.py.

    Returns {frame_idx: bool_array_HxW}. Only frames with detections are present;
    a frame absent from this dict had no face detected.
    """
    npz = np.load(masks_npz_path)
    return {int(k): npz[k] for k in npz.files}


def gaze_hits_mask(px: float, py: float, mask: np.ndarray) -> bool:
    """Return True if pixel (px, py) is within the face mask."""
    row, col = int(py), int(px)
    h, w = mask.shape
    if 0 <= row < h and 0 <= col < w:
        return bool(mask[row, col])
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-participant face-looking time series
# ─────────────────────────────────────────────────────────────────────────────

def compute_participant_time_series(
    trial_gaze: pd.DataFrame,
    frame_masks: dict[int, np.ndarray],
    fps: float,
    total_frames: int,
    display_w: int,
    display_h: int,
    video_w: int,
    video_h: int,
    flip_y: bool,
) -> np.ndarray:
    """
    Build a per-frame face-looking indicator for one participant.

    Returns a float array of length `total_frames` where:
      1.0  = participant was looking at a face this frame
      0.0  = participant was looking away from face
      NaN  = no valid gaze sample for this frame
    """
    ts = np.full(total_frames, np.nan)

    valid = trial_gaze.dropna(subset=["trial_time", "gaze_x", "gaze_y"])
    if valid.empty:
        return ts

    times_ms = valid["trial_time"].to_numpy(dtype=float)
    gx = valid["gaze_x"].to_numpy(dtype=float)
    gy = valid["gaze_y"].to_numpy(dtype=float)

    vx, vy = gaze_to_video_coords(gx, gy, display_w, display_h, video_w, video_h, flip_y)

    frame_indices = np.round(times_ms / (1000.0 / fps)).astype(int)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1)

    frame_gaze_hits: dict[int, list[float]] = {}
    for i, fi in enumerate(frame_indices):
        if np.isnan(vx[i]) or np.isnan(vy[i]):
            frame_gaze_hits.setdefault(fi, [])  # has gaze data but out of bounds
            continue
        mask = frame_masks.get(fi)
        hit = gaze_hits_mask(vx[i], vy[i], mask) if mask is not None else False
        frame_gaze_hits.setdefault(fi, []).append(float(hit))

    for fi, hits in frame_gaze_hits.items():
        if hits:
            ts[fi] = float(any(hits))
        # else: no in-bounds samples this frame → stays NaN

    return ts


# ─────────────────────────────────────────────────────────────────────────────
# ISC helpers
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_isc(
    time_series_dict: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """
    Compute pairwise Pearson correlations over the intersection of valid frames.

    Returns
    -------
    correlations : 1-D array of r values (one per participant pair)
    pairs        : list of (pid_a, pid_b) matching `correlations`
    """
    pids = list(time_series_dict.keys())
    correlations = []
    pairs = []

    for pid_a, pid_b in combinations(pids, 2):
        ts_a = time_series_dict[pid_a]
        ts_b = time_series_dict[pid_b]

        # Only use frames where both participants have valid gaze
        both_valid = ~(np.isnan(ts_a) | np.isnan(ts_b))
        if both_valid.sum() < 10:
            continue  # not enough overlap

        a_vals = ts_a[both_valid]
        b_vals = ts_b[both_valid]

        if a_vals.std() == 0 or b_vals.std() == 0:
            continue  # no variance — correlation undefined

        r, _ = stats.pearsonr(a_vals, b_vals)
        correlations.append(r)
        pairs.append((pid_a, pid_b))

    return np.array(correlations), pairs


def cross_group_isc(
    ts_group_a: dict[str, np.ndarray],
    ts_group_b: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """
    Compute all cross-group Pearson correlations (every a × every b pair).
    """
    correlations = []
    pairs = []

    for pid_a, ts_a in ts_group_a.items():
        for pid_b, ts_b in ts_group_b.items():
            both_valid = ~(np.isnan(ts_a) | np.isnan(ts_b))
            if both_valid.sum() < 10:
                continue
            a_vals = ts_a[both_valid]
            b_vals = ts_b[both_valid]
            if a_vals.std() == 0 or b_vals.std() == 0:
                continue
            r, _ = stats.pearsonr(a_vals, b_vals)
            correlations.append(r)
            pairs.append((pid_a, pid_b))

    return np.array(correlations), pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_participants(
    data_dir: str,
    group_label: str,
    video_name: str,
    frame_masks: dict[int, np.ndarray],
    fps: float,
    total_frames: int,
    display_w: int,
    display_h: int,
    video_w: int,
    video_h: int,
    flip_y: bool,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """
    Load all participants in `data_dir`, extract their face-looking time series,
    and return per-participant proportion records.
    """
    time_series = {}
    prop_records = []

    participant_dirs = sorted(
        [
            d for pattern in ("MW*", "HMET*")
            for d in glob.glob(os.path.join(data_dir, pattern))
            if os.path.isdir(d)
        ]
    )

    for pdir in participant_dirs:
        print(f"Processing {group_label} participant: {pdir}")
        pid = os.path.basename(pdir)
        files = find_participant_files(pdir)
        if files["gaze"] is None:
            continue

        # Check if this participant watched the target video
        if files["trial_order"] is not None:
            trial_df = pd.read_csv(files["trial_order"])
            # strip potential _stripped suffix for comparison
            bare_name = video_name.replace("_stripped", "")
            if bare_name not in trial_df["video_name"].values and video_name not in trial_df["video_name"].values:
                continue

        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                gaze_df = pd.read_csv(files["gaze"], encoding=enc, low_memory=False)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            print(f"    [{group_label}] {pid}: could not decode CSV — skipping")
            continue
        trial_gaze = extract_trial_gaze(gaze_df, video_name)

        if trial_gaze is None or trial_gaze.empty:
            print(f"    [{group_label}] {pid}: no matching trial found — skipping")
            continue

        ts = compute_participant_time_series(
            trial_gaze=trial_gaze,
            frame_masks=frame_masks,
            fps=fps,
            total_frames=total_frames,
            display_w=display_w,
            display_h=display_h,
            video_w=video_w,
            video_h=video_h,
            flip_y=flip_y,
        )

        valid_mask = ~np.isnan(ts)
        n_valid = int(valid_mask.sum())
        n_face = int(ts[valid_mask].sum()) if n_valid > 0 else 0
        prop = n_face / n_valid if n_valid > 0 else np.nan

        time_series[pid] = ts
        prop_records.append(
            dict(
                participant_id=pid,
                group=group_label,
                video_name=video_name,
                n_total_frames=total_frames,
                n_valid_frames=n_valid,
                pct_valid_frames=round(100 * n_valid / total_frames, 2),
                n_face_frames=n_face,
                prop_looking_at_face=round(prop, 4) if not np.isnan(prop) else np.nan,
            )
        )
        print(
            f"    [{group_label}] {pid}: "
            f"valid={n_valid}/{total_frames} frames, "
            f"face={n_face} frames, "
            f"prop={prop:.3f}" if not np.isnan(prop) else f"prop=NaN"
        )

    return time_series, prop_records


def main():
    parser = argparse.ArgumentParser(
        description="Face-looking proportion and ISC analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation_csv",
        required=True,
        help="Path to frames CSV from annotate_video.py.",
    )
    parser.add_argument(
        "--video_name",
        required=True,
        help=(
            "Video name to match in participant trial-order files "
            "(e.g. 'sesameus_1' or 'sesameus_1_stripped')."
        ),
    )
    parser.add_argument("--adults_dir", default="data/adults")
    parser.add_argument("--infants_dir", default="data/infants")
    parser.add_argument(
        "--display_width",
        type=int,
        default=1920,
        help="Width of the display (in pixels) used during the experiment.",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=1080,
        help="Height of the display (in pixels) used during the experiment.",
    )
    parser.add_argument(
        "--no_flip_y",
        action="store_true",
        help="Disable Y-axis flip (use if Tobii data already has top-left origin).",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/results",
        help="Directory where result CSVs are written.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    flip_y = not args.no_flip_y

    # ── Load annotation ───────────────────────────────────────────────────────
    print(f"Loading annotation: {args.annotation_csv}")
    annot_df = pd.read_csv(args.annotation_csv)

    if annot_df.empty:
        sys.exit("ERROR: annotation CSV is empty — run annotate_video.py first.")

    # Get video metadata from the annotation CSV
    video_w = int(annot_df["video_width"].iloc[0])
    video_h = int(annot_df["video_height"].iloc[0])
    # total_frames and fps come from the summary CSV; infer from max frame_idx + fps hint
    max_frame = int(annot_df["frame_idx"].max())
    max_time_ms = float(annot_df["time_ms"].max())
    fps = (max_frame / (max_time_ms / 1000.0)) if max_time_ms > 0 else 30.0

    # Try to find summary CSV for authoritative total_frames
    summary_glob = os.path.join(
        os.path.dirname(args.annotation_csv),
        "*_summary.csv",
    )
    total_frames = max_frame + 1  # fallback
    for sf in glob.glob(summary_glob):
        sdf = pd.read_csv(sf)
        if not sdf.empty and "total_frames" in sdf.columns:
            total_frames = int(sdf["total_frames"].iloc[0])
            fps = float(sdf["fps"].iloc[0])
            n_objects = int(sdf["num_unique_objects"].iloc[0])
            pct_detected = float(sdf["pct_frames_with_detections"].iloc[0])
            print(
                f"Video: {video_w}x{video_h} @ {fps:.2f}fps, {total_frames} frames\n"
                f"Detected {n_objects} face object(s) in {pct_detected:.1f}% of frames"
            )
            break
    else:
        print(f"Video: {video_w}x{video_h}, {total_frames} frames (estimated)")

    # ── Load face masks (preferred) or fall back to bboxes ───────────────────
    masks_npz_path = args.annotation_csv.replace("_frames.csv", "_masks.npz")
    if os.path.isfile(masks_npz_path):
        print(f"Loading face masks: {masks_npz_path}")
        frame_masks = load_face_masks(masks_npz_path)
    else:
        print("WARNING: masks npz not found — falling back to bounding boxes for hit-testing.")
        bbox_lookup = {}
        for _, row in annot_df.iterrows():
            fi = int(row["frame_idx"])
            bx, by, bw, bh = int(row["bbox_x"]), int(row["bbox_y"]), int(row["bbox_w"]), int(row["bbox_h"])
            m = np.zeros((video_h, video_w), dtype=bool)
            m[by:by+bh, bx:bx+bw] = True
            frame_masks[fi] = frame_masks.get(fi, np.zeros((video_h, video_w), dtype=bool)) | m
        frame_masks = bbox_lookup

    print(f"Face detections in {len(frame_masks)} frames.\n")

    # ── Load participants ─────────────────────────────────────────────────────
    print("=== Adults ===")
    adult_ts, adult_props = load_participants(
        data_dir=args.adults_dir,
        group_label="adult",
        video_name=args.video_name,
        frame_masks=frame_masks,
        fps=fps,
        total_frames=total_frames,
        display_w=args.display_width,
        display_h=args.display_height,
        video_w=video_w,
        video_h=video_h,
        flip_y=flip_y,
    )

    print("\n=== Infants ===")
    infant_ts, infant_props = load_participants(
        data_dir=args.infants_dir,
        group_label="infant",
        video_name=args.video_name,
        frame_masks=frame_masks,
        fps=fps,
        total_frames=total_frames,
        display_w=args.display_width,
        display_h=args.display_height,
        video_w=video_w,
        video_h=video_h,
        flip_y=flip_y,
    )

    # ── Save proportion results ───────────────────────────────────────────────
    all_props = adult_props + infant_props
    prop_df = pd.DataFrame(all_props)
    prop_path = os.path.join(args.output_dir, f"{args.video_name}_face_proportion.csv")
    prop_df.to_csv(prop_path, index=False)
    print(f"\nSaved proportion CSV → {prop_path}")

    # Group summary
    for grp, grp_df in prop_df.groupby("group"):
        valid = grp_df["prop_looking_at_face"].dropna()
        print(
            f"  {grp}: n={len(valid)}, "
            f"mean_prop={valid.mean():.3f}, "
            f"sd={valid.std():.3f}"
        )

    # ── ISC ───────────────────────────────────────────────────────────────────
    isc_records = []

    print("\n=== ISC ===")

    # Adult–adult
    if len(adult_ts) >= 2:
        aa_r, aa_pairs = pairwise_isc(adult_ts)
        aa_mean = float(np.mean(aa_r)) if len(aa_r) > 0 else np.nan
        aa_std = float(np.std(aa_r)) if len(aa_r) > 0 else np.nan
        print(
            f"  Adult–adult ISC: "
            f"n_pairs={len(aa_r)}, mean_r={aa_mean:.4f}, sd={aa_std:.4f}"
        )
        isc_records.append(
            dict(
                comparison="adult_adult",
                video_name=args.video_name,
                n_pairs=len(aa_r),
                mean_r=round(aa_mean, 6),
                sd_r=round(aa_std, 6),
                min_r=round(float(np.min(aa_r)), 6) if len(aa_r) > 0 else np.nan,
                max_r=round(float(np.max(aa_r)), 6) if len(aa_r) > 0 else np.nan,
            )
        )
    else:
        print(f"  Adult–adult ISC: fewer than 2 adults — skipping")

    # Infant–infant
    if len(infant_ts) >= 2:
        ii_r, ii_pairs = pairwise_isc(infant_ts)
        ii_mean = float(np.mean(ii_r)) if len(ii_r) > 0 else np.nan
        ii_std = float(np.std(ii_r)) if len(ii_r) > 0 else np.nan
        print(
            f"  Infant–infant ISC: "
            f"n_pairs={len(ii_r)}, mean_r={ii_mean:.4f}, sd={ii_std:.4f}"
        )
        isc_records.append(
            dict(
                comparison="infant_infant",
                video_name=args.video_name,
                n_pairs=len(ii_r),
                mean_r=round(ii_mean, 6),
                sd_r=round(ii_std, 6),
                min_r=round(float(np.min(ii_r)), 6) if len(ii_r) > 0 else np.nan,
                max_r=round(float(np.max(ii_r)), 6) if len(ii_r) > 0 else np.nan,
            )
        )

    # Adult–infant
    if adult_ts and infant_ts:
        ai_r, ai_pairs = cross_group_isc(adult_ts, infant_ts)
        ai_mean = float(np.mean(ai_r)) if len(ai_r) > 0 else np.nan
        ai_std = float(np.std(ai_r)) if len(ai_r) > 0 else np.nan
        print(
            f"  Adult–infant ISC: "
            f"n_pairs={len(ai_r)}, mean_r={ai_mean:.4f}, sd={ai_std:.4f}"
        )
        isc_records.append(
            dict(
                comparison="adult_infant",
                video_name=args.video_name,
                n_pairs=len(ai_r),
                mean_r=round(ai_mean, 6),
                sd_r=round(ai_std, 6),
                min_r=round(float(np.min(ai_r)), 6) if len(ai_r) > 0 else np.nan,
                max_r=round(float(np.max(ai_r)), 6) if len(ai_r) > 0 else np.nan,
            )
        )

    isc_path = os.path.join(args.output_dir, f"{args.video_name}_isc_results.csv")
    pd.DataFrame(isc_records).to_csv(isc_path, index=False)
    print(f"\nSaved ISC CSV → {isc_path}")

    # ── Per-frame time series CSV ─────────────────────────────────────────────
    # Wide format: rows=frames, columns=participants (1=looking at face, 0=not, NaN=no data)
    all_ts = {**adult_ts, **infant_ts}
    frame_indices = list(range(total_frames))
    ts_df = pd.DataFrame({"frame_idx": frame_indices})
    ts_df["time_ms"] = ts_df["frame_idx"] / fps * 1000.0
    ts_df["face_detected"] = ts_df["frame_idx"].map(
        lambda fi: int(fi in frame_masks)
    )
    for pid, ts in sorted(all_ts.items()):
        ts_df[pid] = ts
    ts_path = os.path.join(args.output_dir, f"{args.video_name}_face_timeseries.csv")
    ts_df.to_csv(ts_path, index=False)
    print(f"Saved per-frame time series CSV → {ts_path}  ({len(all_ts)} participants)")
    print("\nDone.")


if __name__ == "__main__":
    main()
