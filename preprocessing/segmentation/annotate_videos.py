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
import json
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
    """Return prompt_idx from a prompt info dict as a string, or None if absent/NaN."""
    raw = p.get("prompt_idx")
    if raw is None:
        return None
    if isinstance(raw, float) and np.isnan(raw):
        return None
    return str(raw)


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


def parse_coord_list(val) -> list[list[int]]:
    """JSON '[[x,y],…]' string → list of [x, y] pairs. Returns [] on blank/error."""
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    s = str(val).strip()
    if not s or s == "[]":
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        log.warning(f"Could not parse coord list '{s}' — ignoring.")
        return []


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

def _save_point_vis(
    frames_dir: str,
    coords_abs: list[list[int]],
    labels: list[int],
    save_path: str,
) -> None:
    """Overlay prompt points on the first frame of `frames_dir` and write to `save_path`."""
    frame_path = os.path.join(frames_dir, "00000.jpg")
    if not os.path.exists(frame_path):
        return
    img = cv2.imread(frame_path)
    if img is None:
        return
    for (x, y), lbl in zip(coords_abs, labels):
        colour = (0, 200, 0) if lbl == 1 else (0, 0, 200)   # green=include, red=exclude
        cv2.circle(img, (int(x), int(y)), 10, colour, -1)
        cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def run_sam3_session(
    predictor,
    frames_dir: str,
    prompt: str,
    include_coords: list[list[int]] | None = None,
    exclude_coords: list[list[int]] | None = None,
    video_W: int = 1920,
    video_H: int = 1080,
    debug_vis_path: str | None = None,
) -> dict:
    """
    Run one SAM 3 session.  Always closes the session and flushes the CUDA
    cache afterwards — even on error — to prevent GPU memory accumulation.

    include_coords: list of [x, y] absolute-pixel foreground points (label=1)
    exclude_coords: list of [x, y] absolute-pixel background points (label=0)
    video_W/video_H: frame dimensions, used to normalise coords to 0-1 range.
    debug_vis_path: if set, saves a JPEG of frame 0 with the points overlaid.
    """
    response   = predictor.handle_request(request=dict(type="start_session", resource_path=frames_dir))
    session_id = response["session_id"]
    outputs: dict = {}

    try:
        coords_abs: list[list[int]] = []
        labels:     list[int]       = []
        for c in (include_coords or []):
            coords_abs.append(list(c)); labels.append(1)
        for c in (exclude_coords or []):
            coords_abs.append(list(c)); labels.append(0)

        # Step 1: text prompt creates the object(s) and returns obj_ids
        text_resp = predictor.handle_request(request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt,
        ))

        # Step 2: if coordinates given, refine with a separate points-only call
        # using the obj_id returned by the text prompt (SAM3 forbids mixing text
        # and points in a single add_prompt request).
        #
        # SAM3 architecture note: add_prompt(text=...) only caches outputs for
        # frame 0.  A subsequent add_prompt(points=...) records an "add" action
        # in the session history, which causes propagate_in_video to choose
        # "propagation_partial".  That mode asserts cached_frame_outputs exists
        # for EVERY frame — so we must first run a full propagation (text-only)
        # to populate the cache before adding point prompts and propagating again.
        if coords_abs:
            coords_rel = [[x / video_W, y / video_H] for x, y in coords_abs]
            text_obj_ids = text_resp["outputs"]["out_obj_ids"].tolist()
            if text_obj_ids:
                # Warm-up pass: full VG propagation so cached_frame_outputs is
                # populated for all frames (required by propagation_partial).
                log.info(f"      Prompt '{prompt}': warm-up propagation to populate frame cache…")
                for _ in predictor.handle_stream_request(
                    request=dict(type="propagate_in_video", session_id=session_id)
                ):
                    pass  # discard; we only need the side-effect on the cache

                # Now add point refinement and do the real propagation.
                predictor.handle_request(request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    points=torch.tensor(coords_rel, dtype=torch.float32),
                    point_labels=torch.tensor(labels, dtype=torch.int32),
                    obj_id=int(text_obj_ids[0]),
                ))
            else:
                log.warning(f"      Prompt '{prompt}': text prompt returned no objects; skipping point refinement.")

            if debug_vis_path:
                _save_point_vis(frames_dir, coords_abs, labels, debug_vis_path)

        n_inc = len(include_coords or [])
        n_exc = len(exclude_coords or [])
        log.info(f"      Prompt '{prompt}' added (include_pts={n_inc}, exclude_pts={n_exc}) — propagating…")

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


def render_unified_video(
    video_path: str,
    rows_data: list[dict],
    masks_dict: dict[str, np.ndarray],
    prompt_label_colours: dict[tuple, tuple],
    output_path: str,
    fps: float,
    video_W: int,
    video_H: int,
) -> None:
    """
    Render one continuous pass from the earliest start_frame to the latest
    end_frame across all rows, overlaying every prompt's mask simultaneously
    on each frame.  Use this instead of render_annotated_video when prompts
    have overlapping time windows (e.g. datavyu CSV mode).
    """
    if not rows_data or not masks_dict:
        log.warning("  No rows/masks — skipping unified render.")
        return

    min_frame = min(r["start_frame"] for r in rows_data)
    max_frame = max(r["end_frame"]   for r in rows_data)
    log.info(f"    Unified render: frames {min_frame}–{max_frame - 1} → {output_path}")

    # Build prefix → (colour, label) from rows_data so we can look up by mask key
    # Mask key format: "r{row_idx}_{prompt_slug}|{global_fi}"
    key_prefix_to_style: dict[str, tuple] = {}
    for row_data in rows_data:
        rid = row_data["row_idx"]
        for p in row_data["prompts"]:
            slug   = slugify(p["prompt"])
            prefix = f"r{rid}_{slug}"
            style  = prompt_label_colours.get((p["prompt"], prompt_idx_of(p)),
                                              (COLOURS[0], p["prompt"]))
            key_prefix_to_style[prefix] = style

    # Pre-build per-frame overlay list
    frame_overlays: dict[int, list[tuple]] = {}
    for mask_key, mask in masks_dict.items():
        try:
            prefix, fi_str = mask_key.rsplit("|", 1)
            fi = int(fi_str)
        except (ValueError, AttributeError):
            continue
        if fi < min_frame or fi >= max_frame:
            continue
        style = key_prefix_to_style.get(prefix, (COLOURS[0], ""))
        frame_overlays.setdefault(fi, []).append((mask, style))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (video_W, video_H))

    try:
        for fi in tqdm(range(min_frame, max_frame), desc="  render", unit="f", leave=False):
            ret, bgr = cap.read()
            if not ret:
                break
            for mask, (colour, label) in frame_overlays.get(fi, []):
                if mask.any():
                    draw_mask_overlay(bgr, mask, colour)
                    if label:
                        draw_prompt_label(bgr, mask, colour, label)
            cv2.putText(
                bgr, f"f{fi} | {fi / fps:.2f}s",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
            )
            writer.write(bgr)
    finally:
        cap.release()
        writer.release()

    log.info("    Unified render complete.")


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
    debug_points: bool = False,
    unified_render: bool = False,
) -> None:
    """
    Annotate all temporal chunks (rows) for one video, writing a single set:
      frames.csv     — bounding boxes for all rows × prompts
      masks.npz      — boolean masks keyed "r{row_idx}_{prompt_slug}|{global_fi}"
      annotated.mp4  — optional render

    unified_render=True: render one continuous pass from min→max frame with all
      masks overlaid simultaneously (correct for datavyu mode where prompts overlap).
    unified_render=False: concatenate one clip per row (legacy behavior).
    """
    print(f"Video path: {video_path}; Video W: {video_W}; Video H: {video_H}")
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
    done_combos:    set[tuple[str, str]]  = set()  # (row_idx, prompt)

    if skip_existing or render_only:
        if os.path.exists(frames_csv):
            edf         = pd.read_csv(frames_csv)
            all_records = edf.to_dict("records")
            done_combos = {(str(r["row_idx"]), str(r["prompt"])) for r in all_records}
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
                    inc_coords  = pinfo.get("include_coords") or []
                    exc_coords  = pinfo.get("exclude_coords") or []

                    # Accumulate SAM3 outputs across sub-chunks
                    all_sc_outputs: dict[int, object] = {}
                    for sci, (ss, se) in enumerate(subchunks):
                        # Build debug vis path for first sub-chunk when coords are present
                        vis_path = None
                        if debug_points and sci == 0 and (inc_coords or exc_coords):
                            vis_dir  = os.path.join(video_out, "debug_points")
                            vis_path = os.path.join(vis_dir, f"{row_idx}_{prompt_slug}.jpg")

                        sc_out = run_sam3_session(
                            predictor, sc_dirs[sci], prompt,
                            # Only pass point coords on the first sub-chunk
                            include_coords=inc_coords if sci == 0 else None,
                            exclude_coords=exc_coords if sci == 0 else None,
                            video_W=video_W,
                            video_H=video_H,
                            debug_vis_path=vis_path,
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
        elif unified_render:
            render_unified_video(
                video_path           = video_path,
                rows_data            = rows_data,
                masks_dict           = all_masks_dict,
                prompt_label_colours = prompt_label_colours,
                output_path          = ann_path,
                fps                  = fps,
                video_W              = video_W,
                video_H              = video_H,
            )
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
        description="Annotate video chunks with SAM 3 from a stimuli CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to stimuli_prompts.csv or stimuli_videos_datavyu.csv. "
             "Format is auto-detected from column names.",
    )
    parser.add_argument(
        "--faces_and_hands", action="store_true", default=True,
        help="(datavyu CSV) Process only face/hand prompts (default: True).",
    )
    parser.add_argument(
        "--no_faces_and_hands", dest="faces_and_hands", action="store_false",
        help="(datavyu CSV) Process all prompts regardless of type.",
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
        "--debug_points", action="store_true",
        help="Save a JPEG of frame 0 with prompt points overlaid into "
             "{output_dir}/{video}/debug_points/ for every prompt that has coords.",
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

    # ── Detect and normalize CSV format ───────────────────────────────────────
    is_datavyu = "start_time_ms" in df.columns and "prompt_name" in df.columns

    if is_datavyu:
        log.info("Detected datavyu CSV format (stimuli_videos_datavyu.csv).")

        # Apply --faces_and_hands filter before any other processing
        if args.faces_and_hands and "faces_and_hands" in df.columns:
            before = len(df)
            df = df[df["faces_and_hands"].astype(str).str.lower() == "true"].reset_index(drop=True)
            log.info(f"  faces_and_hands filter: {before} → {len(df)} rows.")

        # Normalize to the column names the rest of main() expects
        df["row_idx"]      = df["prompt_idx"]          # unique per prompt
        df["start_time_s"] = df["start_time_ms"].astype(float) / 1000.0
        df["end_time_s"]   = df["end_time_ms"].astype(float) / 1000.0
        df["prompt"]       = df["prompt_name"]
        # include_coords / exclude_coords are JSON lists — keep as-is; set old-style to None
        df["include_coord"] = None
        df["exclude_coord"] = None
    else:
        log.info("Detected legacy CSV format (stimuli_prompts.csv).")

    required_cols = {"row_idx", "video_path", "start_time_s", "end_time_s", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        log.error(f"CSV missing columns: {missing}")
        sys.exit(1)

    for col in ("include_coord", "exclude_coord", "prompt_idx", "include_coords", "exclude_coords"):
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

        prompts = []
        for _, r in group.iterrows():
            # Prefer JSON list coords (datavyu format); fall back to single "x,y" (legacy)
            if r.get("include_coords") not in (None, "", "[]") and not (
                isinstance(r.get("include_coords"), float) and np.isnan(r.get("include_coords"))
            ):
                inc = parse_coord_list(r["include_coords"])
                exc = parse_coord_list(r.get("exclude_coords", "[]"))
            else:
                single_inc = parse_coord(r.get("include_coord"))
                single_exc = parse_coord(r.get("exclude_coord"))
                inc = [list(single_inc)] if single_inc else []
                exc = [list(single_exc)] if single_exc else []

            prompts.append({
                "prompt":        str(r["prompt"]),
                "prompt_idx":    r.get("prompt_idx"),
                "include_coords": inc,
                "exclude_coords": exc,
            })

        video_rows.setdefault(video_path, []).append({
            "row_idx":     str(row_idx),
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
        # Raise the Gloo collective-op timeout from the default 180 s so that a
        # slow propagation pass (e.g. long clips or many frames) does not leave
        # worker processes in a broken state when rank-0 exits early.
        os.environ.setdefault("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180")
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
                    video_path     = video_path,
                    fps            = fps,
                    video_W        = W,
                    video_H        = H,
                    rows_data      = rows_data,
                    output_dir     = str(OUTPUT_DIR),
                    chunk_size     = args.chunk_size,
                    render_video   = args.render_video,
                    predictor      = predictor,
                    skip_existing  = args.skip_existing,
                    render_only    = args.render_only,
                    debug_points   = args.debug_points,
                    unified_render = is_datavyu,
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
