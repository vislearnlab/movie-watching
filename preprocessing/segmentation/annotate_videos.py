"""
Batch video annotation using SAM 3, driven by stimuli_prompts.csv.

Writes ONE set of output files per video into output_dir/{video_stem}/:
  frames.csv     — bounding boxes & metadata for all rows × prompts
  masks.npz      — boolean masks keyed "r{row_idx}_{prompt_slug}|{global_fi}"
  annotated.mp4  — annotated video (optional); one segment per row, concatenated

Each prompt gets a unique, consistent colour and is labeled "{idx}. {prompt}"
at its mask centroid in the rendered video.

Incremental updates are supported:
  --skip_existing  — skip any (row, prompt) pair already recorded in frames.csv
  --render_only    — skip SAM3 entirely; re-render from existing NPZ/CSV
  --rows N M …     — restrict to specific row_idx values (re-processes only those)

Usage:
    python annotate_videos.py \\
        --csv data/stimuli_prompts.csv \\
        --output_dir preprocessing/segmentation/output \\
        [--render_video]          \\
        [--render_only]           \\
        [--chunk_size 500]        \\
        [--gpus 0 1 2 3]          \\
        [--rows 0 5 10]           \\
        [--skip_existing]

Prerequisites:
    Run prepare_stimuli_videos.ipynb first to generate stimuli_prompts.csv.
"""

import argparse
import gc
import logging
import os
import shutil
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Reduces CUDA memory fragmentation between sessions
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SAM3_REPO = os.path.join(os.path.dirname(__file__), "sam3")
if SAM3_REPO not in sys.path:
    sys.path.insert(0, SAM3_REPO)

from sam3.model_builder import build_sam3_video_predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BGR colours — one per unique prompt (cycled if more than len(COLOURS) prompts)
COLOURS = [
    (60,   60, 255),   # red
    (255, 180,  60),   # blue
    (60,  220,  60),   # green
    (0,   165, 255),   # orange
    (255,  60, 160),   # pink
    (0,   210, 190),   # teal
    (255,  60, 220),   # magenta
    (0,   210, 210),   # yellow
    (200,   0, 200),   # purple
    (100, 220, 255),   # light yellow
    (255, 128,   0),   # sky blue
    (0,   200, 100),   # lime
]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def prompt_idx_of(p: dict):
    """Return int prompt_idx from a prompt info dict, or None if absent/NaN."""
    raw = p.get("prompt_idx")
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    return int(raw)


def slugify(text: str) -> str:
    return (
        text.strip()
        .replace(" ", "_").replace("/", "-")
        .replace(",", "").replace("(", "").replace(")", "")
    )[:40]


def parse_coord(val) -> tuple[int, int] | None:
    """'x,y' string → (x, y) ints, or None if blank/NaN."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    parts = s.split(",")
    if len(parts) == 2:
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            pass
    log.warning(f"Could not parse coordinate '{val}' — ignoring.")
    return None


def mask_npz_key(row_idx: int, prompt_slug: str, global_fi: int) -> str:
    """Unique NPZ key scoped to (row, prompt, frame)."""
    return f"r{row_idx}_{prompt_slug}|{global_fi}"


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_to_jpeg(
    video_path: str,
    frames_dir: str,
    start_frame: int,
    end_frame: int,
) -> int:
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    n_requested = end_frame - start_frame
    local_idx = 0
    for _ in tqdm(range(n_requested), desc="  extract", leave=False, unit="f"):
        ret, frame = cap.read()
        if not ret:
            log.warning(f"Video ended at local frame {local_idx}; expected {n_requested}.")
            break
        cv2.imwrite(
            os.path.join(frames_dir, f"{local_idx:05d}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        local_idx += 1
    cap.release()
    return local_idx


def create_subchunk_symlink_dir(
    frames_dir: str,
    sub_start: int,
    sub_end: int,
    subchunk_dir: str,
) -> None:
    os.makedirs(subchunk_dir, exist_ok=True)
    for local_idx, src_idx in enumerate(range(sub_start, sub_end)):
        src = os.path.abspath(os.path.join(frames_dir, f"{src_idx:05d}.jpg"))
        dst = os.path.join(subchunk_dir, f"{local_idx:05d}.jpg")
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# Mask / drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin = int(np.where(rows)[0][0]);  rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0]);  cmax = int(np.where(cols)[0][-1])
    return cmin, rmin, cmax - cmin + 1, rmax - rmin + 1


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, colour: tuple, alpha: float = 0.45):
    overlay = frame.copy()
    overlay[mask] = colour
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_prompt_label(frame: np.ndarray, mask: np.ndarray, colour: tuple, label: str):
    """Draw `label` near the centroid of `mask` with a dark background box."""
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return
    bx, by, bw, bh = bbox
    cx, cy = bx + bw // 2, by + bh // 2

    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.5
    thickness = 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thickness)

    # Clamp to frame bounds
    h, w = frame.shape[:2]
    tx = max(0, min(cx - tw // 2, w - tw - 4))
    ty = max(th + 6, min(cy, h - bl - 4))

    cv2.rectangle(frame, (tx - 2, ty - th - 4), (tx + tw + 2, ty + bl), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty - 2), font, scale, colour, thickness, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# SAM 3 session
# ─────────────────────────────────────────────────────────────────────────────

def run_sam3_session(
    predictor,
    frames_dir: str,
    prompt: str,
    include_coord: tuple[int, int] | None = None,
    exclude_coord: tuple[int, int] | None = None,
) -> dict:
    """
    Run one SAM 3 session.  Always closes the session and flushes the CUDA
    cache afterwards — even on error — to prevent GPU memory accumulation.
    """
    response   = predictor.handle_request(request=dict(type="start_session", resource_path=frames_dir))
    session_id = response["session_id"]
    outputs: dict = {}

    try:
        req: dict = dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
        coords, labels = [], []
        if include_coord is not None:
            coords.append(list(include_coord)); labels.append(1)
        if exclude_coord is not None:
            coords.append(list(exclude_coord)); labels.append(0)
        if coords:
            req["point_coords"] = coords
            req["point_labels"] = labels

        predictor.handle_request(request=req)
        log.info(f"      Prompt '{prompt}' added — propagating…")

        for resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            outputs[resp["frame_index"]] = resp["outputs"]

        log.info(f"      Done: {len(outputs)} frames.")

    except Exception as e:
        log.error(f"      ERROR (prompt='{prompt}'): {e}")
        traceback.print_exc()

    finally:
        try:
            predictor.handle_request(request=dict(type="close_session", session_id=session_id))
        except Exception as ce:
            log.warning(f"      close_session error (non-fatal): {ce}")
        # Free reserved GPU memory immediately after every session
        torch.cuda.empty_cache()
        gc.collect()

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# Video rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_annotated_video(
    video_path: str,
    rows_data: list[dict],
    masks_dict: dict[str, np.ndarray],
    prompt_label_colours: dict[str, tuple],  # prompt → (colour, label_str)
    output_path: str,
    fps: float,
    video_W: int,
    video_H: int,
) -> None:
    """
    Render one annotated segment per row (in row order), concatenated into a
    single MP4.  Each prompt overlay is drawn with its assigned colour and
    labeled "{idx}. {prompt}" at the centroid of its mask.
    """
    log.info(f"    Rendering annotated video → {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_W, video_H))

    try:
        for row_data in rows_data:
            row_idx     = row_data["row_idx"]
            start_frame = row_data["start_frame"]
            end_frame   = row_data["end_frame"]

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for local_fi in tqdm(
                range(end_frame - start_frame),
                desc=f"  render row {row_idx}", leave=False, unit="f",
            ):
                ret, bgr = cap.read()
                if not ret:
                    break
                global_fi = start_frame + local_fi

                for pinfo in row_data.get("prompts", []):
                    pname   = pinfo["prompt"]
                    pidx    = prompt_idx_of(pinfo)
                    entry   = prompt_label_colours.get((pname, pidx))
                    if entry is None:
                        continue
                    colour, label_str = entry
                    mask_key = mask_npz_key(row_idx, slugify(pname), global_fi)
                    mask = masks_dict.get(mask_key)
                    if mask is not None and mask.any():
                        draw_mask_overlay(bgr, mask, colour)
                        draw_prompt_label(bgr, mask, colour, label_str)

                cv2.putText(
                    bgr,
                    f"row{row_idx} | {global_fi / fps:.2f}s | f{global_fi}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
                )
                writer.write(bgr)

            cap.release()
    finally:
        writer.release()

    log.info("    Render complete.")


# ─────────────────────────────────────────────────────────────────────────────
# Core per-video annotation
# ─────────────────────────────────────────────────────────────────────────────

def process_one_video(
    video_path: str,
    fps: float,
    video_W: int,
    video_H: int,
    rows_data: list[dict],
    output_dir: str,
    chunk_size: int,
    render_video: bool,
    predictor,
    skip_existing: bool = False,
    render_only: bool = False,
) -> None:
    """
    Annotate all temporal chunks (rows) for one video, writing a single set:
      frames.csv     — bounding boxes for all rows × prompts
      masks.npz      — boolean masks keyed "r{row_idx}_{prompt_slug}|{global_fi}"
      annotated.mp4  — optional; segments concatenated, each prompt color-labeled
    """
    video_stem = Path(video_path).stem.removesuffix("_stripped")
    video_out  = os.path.join(output_dir, video_stem)
    os.makedirs(video_out, exist_ok=True)

    frames_csv = os.path.join(video_out, "frames.csv")
    masks_path = os.path.join(video_out, "masks.npz")
    ann_path   = os.path.join(video_out, "annotated.mp4")

    # ── Assign a consistent colour to each unique (prompt, prompt_idx) pair ──────
    # Keying by (prompt, prompt_idx) means "person" with idx=1 and "person" with
    # idx=2 get distinct colours and labels rather than collapsing to one entry.
    seen_prompts: dict[tuple, tuple] = {}  # (prompt, pidx) → colour
    for row_data in rows_data:
        for p in row_data["prompts"]:
            key = (p["prompt"], prompt_idx_of(p))
            if key not in seen_prompts:
                seen_prompts[key] = COLOURS[len(seen_prompts) % len(COLOURS)]

    # (prompt, pidx) → (colour_bgr, label_str)
    prompt_label_colours: dict[tuple, tuple] = {
        (pname, pidx): (colour, f"{pidx}. {pname}" if pidx is not None else pname)
        for (pname, pidx), colour in seen_prompts.items()
    }

    # ── Load existing data for incremental / render-only runs ─────────────────
    all_records:    list[dict]            = []
    all_masks_dict: dict[str, np.ndarray] = {}
    done_combos:    set[tuple[int, str]]  = set()  # (row_idx, prompt)

    if skip_existing or render_only:
        if os.path.exists(frames_csv):
            edf         = pd.read_csv(frames_csv)
            all_records = edf.to_dict("records")
            done_combos = {(int(r["row_idx"]), str(r["prompt"])) for r in all_records}
            log.info(f"  Loaded {len(edf)} existing records ({len(done_combos)} (row,prompt) pairs).")
        if os.path.exists(masks_path):
            with np.load(masks_path, allow_pickle=False) as npz:
                all_masks_dict = dict(npz)
            log.info(f"  Loaded {len(all_masks_dict)} existing masks.")

    # ── SAM 3 processing (skipped when --render_only) ─────────────────────────
    if not render_only:
        for row_data in tqdm(rows_data, desc=f"  {video_stem}", leave=False):
            row_idx     = row_data["row_idx"]
            start_frame = row_data["start_frame"]
            end_frame   = row_data["end_frame"]
            n_frames    = end_frame - start_frame
            all_prompts = row_data["prompts"]

            if skip_existing:
                prompts_to_run = [
                    p for p in all_prompts if (row_idx, p["prompt"]) not in done_combos
                ]
                if not prompts_to_run:
                    log.info(f"  Row {row_idx}: all prompts cached — skipping SAM3.")
                    continue
                skipped = len(all_prompts) - len(prompts_to_run)
                if skipped:
                    log.info(f"  Row {row_idx}: {skipped} prompt(s) cached; running {len(prompts_to_run)} new.")
            else:
                prompts_to_run = all_prompts

            log.info(
                f"\n{'─'*60}\n"
                f"  Row {row_idx}  f{start_frame}–{end_frame - 1} ({n_frames} frames)\n"
                f"  Prompts: {[p['prompt'] for p in prompts_to_run]}"
            )

            tmp_dir    = tempfile.mkdtemp(prefix="sam3_")
            frames_dir = os.path.join(tmp_dir, "frames")

            try:
                n_ext = extract_frames_to_jpeg(video_path, frames_dir, start_frame, end_frame)
                if n_ext == 0:
                    log.warning(f"  Row {row_idx}: no frames extracted — skipping.")
                    continue

                if n_ext <= chunk_size:
                    subchunks = [(0, n_ext)]
                    sc_dirs   = [frames_dir]
                else:
                    subchunks = [
                        (s, min(s + chunk_size, n_ext))
                        for s in range(0, n_ext, chunk_size)
                    ]
                    sc_dirs = []
                    for sci, (ss, se) in enumerate(subchunks):
                        sc_dir = os.path.join(tmp_dir, f"sc_{sci:03d}")
                        create_subchunk_symlink_dir(frames_dir, ss, se, sc_dir)
                        sc_dirs.append(sc_dir)
                    log.info(f"  {n_ext} frames → {len(subchunks)} sub-chunks.")

                for pinfo in tqdm(prompts_to_run, desc=f"  row {row_idx}", leave=False):
                    prompt      = pinfo["prompt"]
                    prompt_slug = slugify(prompt)
                    prompt_idx  = pinfo.get("prompt_idx")
                    inc_coord   = pinfo.get("include_coord")
                    exc_coord   = pinfo.get("exclude_coord")

                    # Accumulate SAM3 outputs across sub-chunks
                    all_sc_outputs: dict[int, object] = {}
                    for sci, (ss, se) in enumerate(subchunks):
                        sc_out = run_sam3_session(
                            predictor, sc_dirs[sci], prompt, inc_coord, exc_coord
                        )
                        log.info(f"    sc {sci + 1}/{len(subchunks)}: {len(sc_out)}/{se - ss} frames.")
                        for sc_fi, out in sc_out.items():
                            all_sc_outputs[ss + sc_fi] = out

                    log.info(f"    '{prompt}': {len(all_sc_outputs)}/{n_ext} frames with masks.")

                    for local_fi in sorted(all_sc_outputs.keys()):
                        out       = all_sc_outputs[local_fi]
                        obj_ids   = out["out_obj_ids"].tolist()
                        bin_masks = out["out_binary_masks"]
                        global_fi = start_frame + local_fi
                        time_ms   = (global_fi / fps) * 1000.0

                        combined = np.zeros((video_H, video_W), dtype=bool)
                        for i, obj_id in enumerate(obj_ids):
                            mask = bin_masks[i]
                            if not mask.any():
                                continue
                            combined |= mask
                            bbox = mask_to_bbox(mask)
                            if bbox is None:
                                continue
                            bx, by, bw, bh = bbox
                            mask_area = int(mask.sum())
                            all_records.append(dict(
                                video_name       = video_stem,
                                row_idx          = row_idx,
                                prompt_idx       = prompt_idx,
                                prompt           = prompt,
                                frame_idx_global = global_fi,
                                frame_idx_local  = local_fi,
                                time_ms          = round(time_ms, 2),
                                obj_id           = obj_id,
                                bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                                mask_area_px     = mask_area,
                                mask_area_frac   = round(mask_area / (video_W * video_H), 6),
                                centroid_x       = bx + bw // 2,
                                centroid_y       = by + bh // 2,
                                video_width      = video_W,
                                video_height     = video_H,
                            ))

                        if combined.any():
                            all_masks_dict[mask_npz_key(row_idx, prompt_slug, global_fi)] = combined

                    done_combos.add((row_idx, prompt))

            except Exception as e:
                log.error(f"  Row {row_idx} error: {e}")
                traceback.print_exc()
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

            # ── Checkpoint after every row so a crash doesn't lose prior work ─
            frames_df = pd.DataFrame(all_records)
            frames_df.to_csv(frames_csv, index=False)
            np.savez_compressed(masks_path, **all_masks_dict)
            log.info(
                f"  Checkpoint → {frames_csv} ({len(frames_df)} rows) | "
                f"{len(all_masks_dict)} masks"
            )

    # ── Render annotated MP4 ───────────────────────────────────────────────────
    if render_video:
        if not all_masks_dict:
            log.warning(f"  No masks available for {video_stem} — skipping render.")
        else:
            render_annotated_video(
                video_path           = video_path,
                rows_data            = rows_data,
                masks_dict           = all_masks_dict,
                prompt_label_colours = prompt_label_colours,
                output_path          = ann_path,
                fps                  = fps,
                video_W              = video_W,
                video_H              = video_H,
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate video chunks with SAM 3 from stimuli_prompts.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to stimuli_prompts.csv (columns: row_idx, video_path, "
             "start_time_s, end_time_s, prompt; optional: prompt_idx, "
             "include_coord, exclude_coord, video_block).",
    )
    parser.add_argument(
        "--output_dir", default="preprocessing/segmentation/output",
        help="Root output directory; one sub-folder is created per video.",
    )
    parser.add_argument(
        "--render_video", action="store_true",
        help="Write annotated MP4 (one per video).",
    )
    parser.add_argument(
        "--render_only", action="store_true",
        help="Skip SAM3 entirely; re-render annotated.mp4 from existing NPZ/CSV. "
             "Implies --render_video.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500,
        help="Max frames per SAM3 session (rows with more frames are split).",
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="GPU IDs to use.  Default: all visible GPUs.",
    )
    parser.add_argument(
        "--rows", nargs="+", type=int, default=None,
        help="Process only these row_idx values (useful for debugging or re-running).",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip any (row, prompt) pair already present in the output frames.csv.",
    )
    args = parser.parse_args()

    if args.render_only:
        args.render_video = True

    _here = Path(os.getcwd())
    REPO_ROOT = _here
    while not (REPO_ROOT / "data").is_dir() and REPO_ROOT.parent != REPO_ROOT:
        REPO_ROOT = REPO_ROOT.parent

    CSV_PATH   = REPO_ROOT / args.csv
    OUTPUT_DIR = REPO_ROOT / args.output_dir

    log.info(f"CSV: {CSV_PATH}")
    if not CSV_PATH.exists():
        log.error(f"CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    log.info(f"{len(df)} rows in CSV.")

    required_cols = {"row_idx", "video_path", "start_time_s", "end_time_s", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(f"CSV missing columns: {missing}")
        sys.exit(1)

    for col in ("include_coord", "exclude_coord", "prompt_idx"):
        if col not in df.columns:
            df[col] = None

    n_before = len(df)
    df = df.dropna(subset=["video_path"]).copy()
    if len(df) < n_before:
        log.warning(f"Dropped {n_before - len(df)} rows with missing video_path.")

    if args.rows is not None:
        df = df[df["row_idx"].isin(args.rows)].copy()
        log.info(f"Filtered to rows {args.rows}: {len(df)} prompt rows remaining.")
        if df.empty:
            log.error("No rows match --rows; nothing to do.")
            sys.exit(1)

    # ── Read video metadata once per unique video ──────────────────────────────
    video_meta: dict[str, tuple] = {}
    for vp in df["video_path"].dropna().unique():
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps == 0:
            log.warning(f"Could not read metadata for {vp} — skipping all rows for this video.")
            continue
        video_meta[vp] = (fps, W, H)
        log.info(f"  {Path(vp).name}  {W}×{H} @ {fps:.3f} fps")

    # ── Build per-video row list ───────────────────────────────────────────────
    video_rows: dict[str, list[dict]] = {}

    for row_idx, group in df.groupby("row_idx"):
        first      = group.iloc[0]
        video_path = first["video_path"]
        if video_path not in video_meta:
            log.warning(f"Row {row_idx}: no metadata for {video_path} — skipping.")
            continue

        fps, W, H   = video_meta[video_path]
        start_s     = float(first["start_time_s"])
        end_s       = float(first["end_time_s"])
        start_frame = int(round(start_s * fps))
        end_frame   = int(round(end_s   * fps))

        if end_frame <= start_frame:
            log.warning(f"Row {row_idx}: zero-duration chunk ({start_s}s–{end_s}s) — skipping.")
            continue

        prompts = [
            {
                "prompt":        str(r["prompt"]),
                "prompt_idx":    r.get("prompt_idx"),
                "include_coord": parse_coord(r.get("include_coord")),
                "exclude_coord": parse_coord(r.get("exclude_coord")),
            }
            for _, r in group.iterrows()
        ]

        video_rows.setdefault(video_path, []).append({
            "row_idx":     int(row_idx),
            "start_frame": start_frame,
            "end_frame":   end_frame,
            "prompts":     prompts,
        })

    n_videos = len(video_rows)
    log.info(f"Processing {n_videos} video(s).")

    # ── Load SAM 3 (unless render-only) ───────────────────────────────────────
    predictor = None
    if not args.render_only:
        gpus = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
        log.info(f"Loading SAM 3 on GPU(s) {gpus} — ~2 minutes …")
        predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        log.info("SAM 3 ready.")

    # ── Main loop ──────────────────────────────────────────────────────────────
    n_done = n_error = 0
    video_bar = tqdm(total=n_videos, desc="videos", unit="vid", position=0)

    try:
        for video_path, rows_data in video_rows.items():
            fps, W, H  = video_meta[video_path]
            video_stem = Path(video_path).stem.removesuffix("_stripped")
            video_bar.set_postfix_str(f"{video_stem} ({len(rows_data)} rows)")
            log.info(f"\n{'='*60}\nVideo: {video_stem}  ({len(rows_data)} rows)\n{'='*60}")

            try:
                process_one_video(
                    video_path    = video_path,
                    fps           = fps,
                    video_W       = W,
                    video_H       = H,
                    rows_data     = rows_data,
                    output_dir    = str(OUTPUT_DIR),
                    chunk_size    = args.chunk_size,
                    render_video  = args.render_video,
                    predictor     = predictor,
                    skip_existing = args.skip_existing,
                    render_only   = args.render_only,
                )
                n_done += 1
            except Exception as e:
                log.error(f"  ERROR: {video_stem}: {e}")
                traceback.print_exc()
                n_error += 1

            video_bar.update(1)

    finally:
        video_bar.close()
        if predictor is not None:
            log.info("Shutting down SAM 3 …")
            try:
                predictor.shutdown()
            except Exception as e:
                log.warning(f"shutdown() error (non-fatal): {e}")
        torch.cuda.empty_cache()

    log.info(f"\nDone.  {n_done} video(s) completed | {n_error} error(s)")


if __name__ == "__main__":
    main()
