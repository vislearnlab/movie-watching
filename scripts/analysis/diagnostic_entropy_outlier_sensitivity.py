"""
Does pixar's lower entropy reflect a broad group effect, or a few participants?
Uses calc_entropy_by_segment.py's windows (cut-filtered, full-containment).

1. Leave-one-out on entropy: exclude each person, measure the shift. Direct, not
   a proxy.
2. Concordance on the same windows, segment-then-mean: do people flagged by (1)
   also show low group agreement?

frank/sesame/pixar only.

Output: entropy_outlier_sensitivity.csv -- (block, pid): n_segments_present,
entropy_drop (positive = removing them lowers entropy), mean_concordance_r.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from gaze_tiles import (  # noqa: E402
    DISPLAY_H, DISPLAY_W, build_participant_arrays, closed_form_tile_values, load_video_fixations,
)
from plot_patch_grid import compute_patch_grid  # noqa: E402

WINDOW_S, STEP_S, SIGMA = 1.0, 0.25, 46.0
N_FIX_FIXED, N_DRAWS, RNG_SEED = 75, 100, 0  # same as calc_entropy_by_segment.py

DURATIONS = {
    "frank_complex": 60.32, "frank_objects": 60.12, "frank_play": 60.92,
    "pixar_birds": 150.15,
    "sesameindia_1": 60.00, "sesameindia_2": 60.84, "sesameus_1": 60.5605, "sesameus_2": 60.827433,
}
BLOCKS = {
    "frank_complex": "frank", "frank_objects": "frank", "frank_play": "frank",
    "pixar_birds": "pixar",
    "sesameindia_1": "sesame", "sesameindia_2": "sesame", "sesameus_1": "sesame", "sesameus_2": "sesame",
}
BLOCK_VIDEOS = {"frank": [], "sesame": [], "pixar": []}
for v, b in BLOCKS.items():
    BLOCK_VIDEOS[b].append(v)

FIXATION_DIR = PROJECT_ROOT / "data" / "local_cache" / "fixations" / "adults"
PARTICIPANT_SUMMARY_PATH = PROJECT_ROOT / "data" / "local_cache" / "participant_summary.csv"
CUTS_PATH = PROJECT_ROOT / "annotations" / "scene-cuts" / "all_clips_cuts_frame_corrected.csv"  # see METHODS_DECISIONS 2026-10-24
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "entropy_by_segment" / "entropy_outlier_sensitivity.csv"

centers, grid_info = compute_patch_grid(240, 15, DISPLAY_W, DISPLAY_H)  # 2026-10-22: moved from 180px to 240px (50% overlap)
cx, cy = centers["center_x"].to_numpy(), centers["center_y"].to_numpy()
assert len(cx) == 120


def gauss_density(x, y, w, cx, cy, sigma):
    dist_sq = (x[:, None] - cx[None, :]) ** 2 + (y[:, None] - cy[None, :]) ** 2
    gauss = np.exp(-dist_sq / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    return (w[:, None] * gauss).sum(axis=0)


def entropy_from_vec(vec):
    total = vec.sum()
    if total <= 0:
        return np.nan
    p = vec / total
    nz = p[p > 0]
    return -(nz * np.log2(nz)).sum()


def segment_ids(dur, cut_times, n_windows):
    """Qualifying-window -> segment index (-1 if the window isn't fully inside any segment)."""
    t = np.arange(n_windows) * STEP_S
    bounds = [0.0] + sorted(cut_times) + [dur]
    seg_id = np.full(n_windows, -1, dtype=int)
    for seg_idx, (s, e) in enumerate(zip(bounds[:-1], bounds[1:])):
        if e - s <= 0:
            continue
        m = (t >= s) & (t + WINDOW_S <= e)
        seg_id[m] = seg_idx
    return seg_id


def video_entropy_segments(arrays, dur, seg_id, rng, exclude_pid=None):
    """Per-segment mean rarefied entropy for one video, optionally excluding one pid."""
    pids = [p for p in sorted(arrays) if p != exclude_pid]
    n_windows = len(seg_id)
    ent = np.full(n_windows, np.nan)
    for w in np.where(seg_id >= 0)[0]:
        s, e = w * STEP_S * 1000, (w * STEP_S + WINDOW_S) * 1000
        xs, ys, ws = [], [], []
        for p in pids:
            startT, endT, img_x, img_y = arrays[p]
            overlap = np.minimum(endT, e) - np.maximum(startT, s)
            m = overlap > 0
            if m.any():
                xs.append(img_x[m]); ys.append(img_y[m]); ws.append(overlap[m])
        if not xs:
            continue
        x, y, wt = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        if len(x) < N_FIX_FIXED:
            continue
        draw_ent = np.empty(N_DRAWS)
        for k in range(N_DRAWS):
            idx = rng.choice(len(x), size=N_FIX_FIXED, replace=False)
            draw_ent[k] = entropy_from_vec(gauss_density(x[idx], y[idx], wt[idx], cx, cy, SIGMA))
        ent[w] = draw_ent.mean()

    rows = []
    for seg_idx in sorted(set(seg_id[seg_id >= 0])):
        m = seg_id == seg_idx
        mean_ent = np.nanmean(ent[m])
        if not np.isnan(mean_ent):
            rows.append(mean_ent)
    return rows


def video_concordance(arrays, seg_id):
    """Per-pid list of per-segment mean concordance r (segment-then-mean, matching
    entropy's own aggregation -- not a flat mean across windows, which would
    overweight long segments)."""
    pids = sorted(arrays)
    r_by_pid_seg = {p: {} for p in pids}
    for w in np.where(seg_id >= 0)[0]:
        s, e = w * STEP_S * 1000, (w * STEP_S + WINDOW_S) * 1000
        seg_idx = seg_id[w]
        vecs = {}
        for p in pids:
            startT, endT, img_x, img_y = arrays[p]
            vecs[p] = closed_form_tile_values(startT, endT, img_x, img_y, s, e, SIGMA, cx, cy)
        total_vec = sum(vecs.values())
        for p in pids:
            own, group = vecs[p], total_vec - vecs[p]
            if own.std() == 0 or group.std() == 0:
                continue
            r_by_pid_seg[p].setdefault(seg_idx, []).append(np.corrcoef(own, group)[0, 1])
    return {p: [float(np.mean(rs)) for rs in segs.values()] for p, segs in r_by_pid_seg.items()}


def main():
    cuts_df = pd.read_csv(CUTS_PATH)
    rows = []

    for block, videos in BLOCK_VIDEOS.items():
        print(f"\n=== {block} ({len(videos)} clips) ===")
        video_data = {}
        block_pids = set()
        for video in videos:
            fix = load_video_fixations(FIXATION_DIR, video, "f329476c", PARTICIPANT_SUMMARY_PATH)
            arrays = build_participant_arrays(fix)
            dur = DURATIONS[video]
            n_windows = max(0, int((dur - WINDOW_S) // STEP_S) + 1)
            cut_times = cuts_df[cuts_df.video_name == video].cut_time_s.tolist()
            sid = segment_ids(dur, cut_times, n_windows)
            video_data[video] = (arrays, sid)
            block_pids |= set(arrays)
        block_pids = sorted(block_pids)
        print(f"{len(block_pids)} participants")

        rng = np.random.default_rng(RNG_SEED)
        baseline_segs = [e for v, (arrays, sid) in video_data.items() for e in video_entropy_segments(arrays, DURATIONS[v], sid, rng)]
        baseline_entropy = float(np.mean(baseline_segs))
        print(f"baseline block entropy: {baseline_entropy:.4f} ({len(baseline_segs)} segments)")

        print("computing concordance on entropy's windows...")
        conc_by_pid = {p: [] for p in block_pids}
        for video, (arrays, sid) in video_data.items():
            r_by_pid = video_concordance(arrays, sid)
            for p, rs in r_by_pid.items():
                conc_by_pid[p].extend(rs)

        print(f"running leave-one-out entropy for {len(block_pids)} participants...")
        for i, pid in enumerate(block_pids):
            rng = np.random.default_rng(RNG_SEED)
            segs = []
            n_present = 0
            for video, (arrays, sid) in video_data.items():
                if pid in arrays:
                    n_present += 1
                segs.extend(video_entropy_segments(arrays, DURATIONS[video], sid, rng, exclude_pid=pid))
            excl_entropy = float(np.mean(segs)) if segs else np.nan
            mean_r = float(np.mean(conc_by_pid[pid])) if conc_by_pid[pid] else np.nan
            rows.append(dict(
                block=block, pid=pid, n_clips_present=n_present,
                baseline_entropy=round(baseline_entropy, 4),
                entropy_excl=round(excl_entropy, 4) if not np.isnan(excl_entropy) else np.nan,
                entropy_drop=round(baseline_entropy - excl_entropy, 4) if not np.isnan(excl_entropy) else np.nan,
                mean_concordance_r=round(mean_r, 4) if not np.isnan(mean_r) else np.nan,
            ))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(block_pids)} done")

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows to {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    for block in BLOCK_VIDEOS:
        sub = out[out.block == block]
        print(f"\n{block}: baseline={sub.baseline_entropy.iloc[0]:.3f}")
        print(f"  entropy_drop: mean={sub.entropy_drop.mean():.4f}, max={sub.entropy_drop.max():.4f} "
              f"(pid={sub.loc[sub.entropy_drop.idxmax(), 'pid']}), min={sub.entropy_drop.min():.4f}")
        print(f"  concordance r vs entropy_drop correlation: {sub[['mean_concordance_r','entropy_drop']].corr().iloc[0,1]:.3f}")


if __name__ == "__main__":
    main()
