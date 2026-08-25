"""
Visualize per-frame face-looking data overlaid on the annotated video.

Reads the annotated MP4 (from annotate_video.py --render_video) and the
per-frame time series CSV (from face_gaze_analysis.py), and renders a new
video with a real-time panel showing:
  - Proportion of participants looking at a face each frame (bar chart area)
  - Separate adult vs infant traces
  - Frame-level face-detected indicator

Usage:
    python analysis/visualize_face_gaze.py \
        --video preprocessing/segmentation/output/sesameus_1_stripped_face_annotated.mp4 \
        --timeseries analysis/results/sesameus_1_face_timeseries.csv \
        --output analysis/results/sesameus_1_face_gaze_viz.mp4
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd


# ── Colours (BGR) ────────────────────────────────────────────────────────────
COL_ADULT  = (255, 160,  60)   # orange
COL_INFANT = ( 80, 200, 255)   # cyan
COL_FACE   = ( 60, 220,  60)   # green — face-detected indicator
COL_BG     = ( 30,  30,  30)   # panel background
COL_TEXT   = (220, 220, 220)


def draw_panel(
    panel: np.ndarray,
    frame_idx: int,
    history: list[dict],
    adult_cols: list[str],
    infant_cols: list[str],
    history_len: int,
):
    """
    Draw the gaze panel in-place.

    `history` is a list of per-frame dicts (most recent last), each with keys:
      frame_idx, face_detected, <pid>: float|nan, …
    """
    panel[:] = COL_BG
    H, W = panel.shape[:2]
    pad = 8
    inner_w = W - 2 * pad
    inner_h = H - 2 * pad

    # ── Rolling trace area ────────────────────────────────────────────────────
    trace_h = inner_h - 40  # leave 40px for text labels at bottom
    trace_y0 = pad
    trace_y1 = trace_y0 + trace_h

    # Background grid line at 0.5
    mid_y = trace_y0 + trace_h // 2
    cv2.line(panel, (pad, mid_y), (pad + inner_w, mid_y), (60, 60, 60), 1)

    def prop_to_y(p):
        return int(trace_y1 - p * trace_h)

    n = len(history)
    if n < 2:
        return

    # x positions: history[-1] is rightmost
    def frame_to_x(i):
        return pad + int(i * inner_w / (history_len - 1))

    # Draw adult prop trace (mean across participants)
    adult_pts = []
    infant_pts = []
    face_pts = []

    for i, entry in enumerate(history):
        x = frame_to_x(i)

        adult_vals = [entry[c] for c in adult_cols if not np.isnan(entry.get(c, np.nan))]
        infant_vals = [entry[c] for c in infant_cols if not np.isnan(entry.get(c, np.nan))]

        if adult_vals:
            adult_pts.append((x, prop_to_y(np.mean(adult_vals))))
        if infant_vals:
            infant_pts.append((x, prop_to_y(np.mean(infant_vals))))
        if entry.get("face_detected", 0):
            face_pts.append(x)

    # Shade face-detected frames as faint green column
    for x in face_pts:
        cv2.line(panel, (x, trace_y0), (x, trace_y1), (30, 60, 30), 1)

    # Draw traces
    for pts, colour in [(adult_pts, COL_ADULT), (infant_pts, COL_INFANT)]:
        for j in range(1, len(pts)):
            cv2.line(panel, pts[j - 1], pts[j], colour, 2, cv2.LINE_AA)

    # Current-frame vertical line
    cur_x = frame_to_x(n - 1)
    cv2.line(panel, (cur_x, trace_y0), (cur_x, trace_y1), (180, 180, 180), 1)

    # ── Labels ────────────────────────────────────────────────────────────────
    label_y = trace_y1 + 6
    font = cv2.FONT_HERSHEY_SIMPLEX

    cur = history[-1]

    adult_vals_cur = [cur[c] for c in adult_cols if not np.isnan(cur.get(c, np.nan))]
    infant_vals_cur = [cur[c] for c in infant_cols if not np.isnan(cur.get(c, np.nan))]
    ap = np.mean(adult_vals_cur) if adult_vals_cur else float("nan")
    ip = np.mean(infant_vals_cur) if infant_vals_cur else float("nan")

    adult_str  = f"Adults:  {ap:.0%}  (n={len(adult_vals_cur)})" if not np.isnan(ap) else "Adults: -"
    infant_str = f"Infants: {ip:.0%}  (n={len(infant_vals_cur)})" if not np.isnan(ip) else "Infants: -"
    frame_str  = f"frame {frame_idx}"

    cv2.putText(panel, adult_str,  (pad, label_y + 12), font, 0.38, COL_ADULT,  1, cv2.LINE_AA)
    cv2.putText(panel, infant_str, (pad, label_y + 26), font, 0.38, COL_INFANT, 1, cv2.LINE_AA)
    cv2.putText(panel, frame_str,  (W - 80, label_y + 12), font, 0.35, COL_TEXT, 1, cv2.LINE_AA)

    # Y-axis tick labels
    for frac, label in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]:
        y = prop_to_y(frac)
        cv2.putText(panel, label, (pad, y - 2), font, 0.32, (120, 120, 120), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay face-looking time series on annotated video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True,
                        help="Annotated MP4 from annotate_video.py.")
    parser.add_argument("--timeseries", required=True,
                        help="Per-frame time series CSV from face_gaze_analysis.py.")
    parser.add_argument("--output", required=True,
                        help="Output MP4 path.")
    parser.add_argument("--panel_height", type=int, default=180,
                        help="Height of the gaze panel appended below the video.")
    parser.add_argument("--history_sec", type=float, default=5.0,
                        help="Seconds of history shown in rolling trace.")
    args = parser.parse_args()

    # ── Load time series ──────────────────────────────────────────────────────
    ts_df = pd.read_csv(args.timeseries)
    pid_cols = [c for c in ts_df.columns if c not in ("frame_idx", "time_ms", "face_detected")]

    # Separate adult vs infant columns by ID prefix
    adult_cols  = [c for c in pid_cols if c.startswith("MW")]
    infant_cols = [c for c in pid_cols if c.startswith("HMET")]
    other_cols  = [c for c in pid_cols if c not in adult_cols + infant_cols]
    adult_cols += other_cols  # fold unknowns into adult bucket

    # Build per-frame lookup: frame_idx → dict of values
    ts_df = ts_df.set_index("frame_idx")
    ts_records = ts_df.to_dict("index")

    # ── Video setup ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    fps  = cap.get(cv2.CAP_PROP_FPS)
    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_h = H + args.panel_height
    history_len = max(2, int(args.history_sec * fps))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, out_h))

    panel = np.zeros((args.panel_height, W, 3), dtype=np.uint8)
    history = []

    print(f"Rendering {total} frames → {args.output}")
    print(f"  Adult participants: {len(adult_cols)}, Infant: {len(infant_cols)}")

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        entry = ts_records.get(fi, {"frame_idx": fi, "face_detected": 0})
        # Fill missing participant values as nan
        for c in adult_cols + infant_cols:
            if c not in entry:
                entry[c] = float("nan")

        history.append(entry)
        if len(history) > history_len:
            history.pop(0)

        draw_panel(panel, fi, history, adult_cols, infant_cols, history_len)

        combined = np.concatenate([frame, panel], axis=0)
        writer.write(combined)
        fi += 1

        if fi % 300 == 0:
            print(f"  frame {fi}/{total}")

    cap.release()
    writer.release()
    print(f"Done. Written {fi} frames.")


if __name__ == "__main__":
    main()
