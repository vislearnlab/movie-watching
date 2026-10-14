"""
Render a single scene-cut-to-scene-cut SEGMENT of a clip with the group's pooled fixation
density cloud overlaid, for visually inspecting why a segment's mean entropy (spread of the
group's collective attention, see METHODS_DECISIONS.md / entropy montage work) came out high
or low.

Same windowing as the rest of this analysis (window=1.0s, step=0.25s, sigma=46px), same
density-cloud rendering as visualize_concordance_video.py (reused directly, not reimplemented).
Window alignment is computed relative to the FULL clip's own timeline (not the segment's own
t=0), so it lines up exactly with the per-window entropy values the segment was selected from.

Usage
-----
    python scripts/preprocessing/viz/visualize_entropy_segment.py \\
        --video slow_hands --seg_start 10.50 --seg_end 13.75 --out_path out.mp4 \\
        --label "H=2.11b"
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "preprocessing" / "stimulus_prep"))

from calc_concordance_group import (  # noqa: E402
    DISPLAY_H, DISPLAY_W, build_participant_arrays, load_video_fixations,
)
from plot_patch_grid import compute_patch_grid  # noqa: E402
from visualize_concordance_video import (  # noqa: E402
    cloud_overlay, density_field, load_hud_font, open_frame_writer, probe_video, read_frame,
)

import subprocess as _subprocess  # noqa: E402


def open_segment_frame_reader(path: str, seg_start: float, seg_end: float, w: int, h: int):
    """Trims to [seg_start, seg_end) using ffmpeg's OWN seek (-ss/-to placed AFTER -i, so it
    decodes against real presentation timestamps), rather than counting+discarding frames
    ourselves -- manual frame counting assumes frame index N occurs at exactly N/fps, which
    isn't reliable (decoder reordering/buffering can drift), and produced segments that
    visibly ran one frame into the next scene."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ss", str(seg_start), "-to", str(seg_end),
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-sn", "pipe:1"]
    return _subprocess.Popen(cmd, stdout=_subprocess.PIPE, bufsize=10 ** 8)

OUTPUT_WIDTH, OUTPUT_HEIGHT = 960, 540
BG_OPACITY = 0.45


def main():
    p = argparse.ArgumentParser(description="Render one cut-to-cut segment with the group density cloud.")
    p.add_argument("--video", required=True)
    p.add_argument("--seg_start", type=float, required=True, help="Segment start, seconds from clip start.")
    p.add_argument("--seg_end", type=float, required=True, help="Segment end, seconds from clip start.")
    p.add_argument("--out_path", required=True)
    p.add_argument("--label", default="", help="Extra text for the HUD (e.g. entropy value).")
    p.add_argument("--code", default="f329476c")
    p.add_argument("--fixation_dir", default="data/local_cache/fixations/adults")
    p.add_argument("--participant_summary_path", default="data/local_cache/participant_summary.csv")
    p.add_argument("--stimuli_dir", default="stimuli/main_blocks")
    p.add_argument("--window_s", type=float, default=1.0)
    p.add_argument("--step_s", type=float, default=0.25)
    p.add_argument("--sigma_px", type=float, default=46.0)
    p.add_argument("--density_scale", type=int, default=4)
    p.add_argument("--end_margin_frames", type=float, default=2.0,
                    help="Trim this many frame-durations off seg_end before cutting the video. "
                         "Cut annotations are accurate to roughly a frame, not sub-frame -- "
                         "pixar_birds seg#33 was confirmed (via ffmpeg showinfo) to have its "
                         "true scene-change keyframe 4.7ms BEFORE the annotated cut_time_s, "
                         "which without this margin let that already-transitioned frame land "
                         "inside the 'old' segment.")
    p.add_argument("--start_margin_frames", type=float, default=2.0,
                    help="Symmetric to --end_margin_frames: push seg_start this many "
                         "frame-durations LATER before cutting, in case the annotated cut is "
                         "slightly EARLY and the first frame(s) still show the previous scene. "
                         "Not separately confirmed the way the end-margin case was -- added "
                         "proactively since the underlying annotation-precision issue is "
                         "symmetric and a start-side version wouldn't be visually obvious.")
    args = p.parse_args()

    video_path = PROJECT_ROOT / args.stimuli_dir / f"{args.video}_stripped.mp4"
    if not video_path.exists():
        video_path = PROJECT_ROOT / args.stimuli_dir / f"{args.video}.mp4"

    info = probe_video(str(video_path))
    fps = info["fps"]

    fix_df = load_video_fixations(PROJECT_ROOT / args.fixation_dir, args.video, args.code,
                                   PROJECT_ROOT / args.participant_summary_path)
    pid_arrays = build_participant_arrays(fix_df)
    pids = sorted(pid_arrays)

    grid_w = OUTPUT_WIDTH // args.density_scale
    grid_h = OUTPUT_HEIGHT // args.density_scale
    gx = (np.arange(grid_w) + 0.5) * DISPLAY_W / grid_w
    gy = (np.arange(grid_h) + 0.5) * DISPLAY_H / grid_h

    # Precompute vmax from a handful of windows spanning the segment, same approach as
    # visualize_concordance_video.py (95th percentile of each window's peak density).
    n_windows_total = int(args.seg_end / args.step_s) + 2
    sample_peaks = []
    w = int(np.floor(args.seg_start / args.step_s))
    while w * args.step_s < args.seg_end:
        s_ms, e_ms = w * args.step_s * 1000, (w * args.step_s + args.window_s) * 1000
        field = density_field(pid_arrays, pids, s_ms, e_ms, args.sigma_px, gx, gy)
        if field.max() > 0:
            sample_peaks.append(field.max())
        w += 1
    vmax = float(np.percentile(sample_peaks, 95)) if sample_peaks else 1.0

    font = load_hud_font(15)

    # No start margin at the true clip start (t=0) -- there's no previous scene to bleed from,
    # and seg_start there isn't a cut annotation at all.
    trim_start = args.seg_start + (args.start_margin_frames / fps if args.seg_start > 0 else 0.0)
    trim_end = args.seg_end - args.end_margin_frames / fps
    reader = open_segment_frame_reader(str(video_path), trim_start, trim_end,
                                        OUTPUT_WIDTH, OUTPUT_HEIGHT)
    writer = open_frame_writer(args.out_path, OUTPUT_WIDTH, OUTPUT_HEIGHT, fps)
    cached_w, cached_cloud = -1, None

    frame_idx_local = 0
    while True:
        bg = read_frame(reader, OUTPUT_WIDTH, OUTPUT_HEIGHT)
        if bg is None:
            break
        # ffmpeg's trimmed stream starts at (approximately) trim_start; frame_idx_local/fps
        # offset from there is accurate to within one frame, which is well under the 0.25s
        # analysis step -- fine for choosing which window to display.
        t_s = trim_start + frame_idx_local / fps
        frame_idx_local += 1
        w = int(np.floor((t_s - args.window_s) / args.step_s))
        w = max(w, int(np.floor(args.seg_start / args.step_s)))
        s_ms, e_ms = w * args.step_s * 1000, (w * args.step_s + args.window_s) * 1000

        if w != cached_w:
            field = density_field(pid_arrays, pids, s_ms, e_ms, args.sigma_px, gx, gy)
            cached_cloud = cloud_overlay(field, max(vmax, 1e-9), OUTPUT_WIDTH, OUTPUT_HEIGHT)
            cached_w = w

        grey = bg @ np.array([0.299, 0.587, 0.114])
        base = Image.fromarray(np.repeat((grey * BG_OPACITY).astype(np.uint8)[:, :, None], 3, axis=2),
                                "RGB").convert("RGBA")
        base = Image.alpha_composite(base, cached_cloud)
        draw = ImageDraw.Draw(base)
        draw.rectangle([0, 0, OUTPUT_WIDTH, 22], fill=(0, 0, 0, 180))
        draw.text((4, 3), f"{args.video}  t={t_s:.2f}s  {args.label}", font=font, fill=(255, 255, 255, 255))

        writer.stdin.write(np.array(base.convert("RGB")).tobytes())

    reader.stdout.close(); reader.wait()
    writer.stdin.close(); writer.wait()
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()
