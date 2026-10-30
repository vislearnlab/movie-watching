"""
Check merged annotations against the annotators' clicks.

Per video and category (face / hand / other):
  include_hit   fraction of include clicks within segment.TOL_PX of a mask with the
                click's text, on the click frame ±CLICK_SLACK
  control_hit   the same for N_CONTROL uniformly random points per click on the
                same frames (shuffle control: ≈ mask area fraction; large masks
                that cover everything would score high here too)
  exclude_viol  fraction of exclude clicks inside such a mask (should be 0)
  text_recall   fraction of include clicks SAM3's text prompt already found
                before click seeding (held-out recall of the text detector)
  big_face_frames  face masks covering > BIG_FACE of the frame (whole-body suspects)

Usage:
    python pipeline/evaluate.py [--video …]   → qc/report.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plan import CLICK_SLACK
from segment import TOL_PX

N_CONTROL = 20
BIG_FACE  = 0.15


def hit(masks_at, x, y, tol):
    """True if any mask in masks_at is within tol px of (x, y)."""
    for m in masks_at:
        y0, y1, x0, x1 = max(y - tol, 0), y + tol + 1, max(x - tol, 0), x + tol + 1
        win = m[y0:y1, x0:x1]
        if win.any():
            ys, xs = np.nonzero(win)
            if (((xs + x0 - x) ** 2 + (ys + y0 - y) ** 2).min()) <= tol ** 2:
                return True
    return False


def evaluate_video(stem, prompts, rng):
    vdir = Path(C.OUTPUT_DIR) / stem
    text_of = dict(zip(prompts.prompt_idx, prompts.sam_text))
    cat_of = dict(zip(prompts.sam_text, prompts.category))
    by_text = {}   # (sam_text, frame) → [mask]
    with np.load(vdir / "masks.npz") as z:
        for k in z.files:
            prefix, f = k.rsplit("|", 1)
            by_text.setdefault((text_of[int(prefix[1:].split("_", 1)[0])], int(f)), []).append(z[k])
    clicks = pd.read_csv(vdir / "clicks.csv")
    frames = pd.read_csv(vdir / "frames.csv")
    W, H = C.VIDEO_SIZE

    rows = []
    for cat in ("face", "hand", "other"):
        cc = clicks[clicks.sam_text.map(cat_of) == cat]
        inc, exc = cc[cc.label == 1], cc[cc.label == 0]
        real, ctrl, viol = [], [], []
        for c in inc.itertuples():
            near = [m for f in range(c.frame - CLICK_SLACK, c.frame + CLICK_SLACK + 1)
                    for m in by_text.get((c.sam_text, f), [])]
            real.append(hit(near, c.x, c.y, TOL_PX))
            pts = zip(rng.integers(0, W, N_CONTROL), rng.integers(0, H, N_CONTROL))
            ctrl += [hit(near, int(x), int(y), TOL_PX) for x, y in pts]
        for c in exc.itertuples():
            viol.append(hit(by_text.get((c.sam_text, c.frame), []), c.x, c.y, 0))
        fr = frames[frames.category == cat]
        rows.append(dict(
            video=stem, category=cat, n_include=len(inc), n_exclude=len(exc),
            include_hit=np.mean(real) if real else np.nan,
            control_hit=np.mean(ctrl) if ctrl else np.nan,
            exclude_viol=np.mean(viol) if viol else np.nan,
            text_recall=inc.status.isin(["already_tracked", "tracked_nearby"]).mean() if len(inc) else np.nan,
            unresolved=int((inc.status == "no_candidate").sum()),
            frames_with_mask=fr.frame_idx_global.nunique(),
            median_area_frac=fr.mask_area_frac.median() if len(fr) else np.nan,
            big_face_frames=int((fr.mask_area_frac > BIG_FACE).sum()) if cat == "face" else 0,
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", nargs="+")
    args = ap.parse_args()
    prompts = pd.read_csv(C.PROMPTS_CSV)
    rng = np.random.default_rng(0)
    rows = []
    for vdir in sorted(Path(C.OUTPUT_DIR).iterdir()):
        if (vdir / "masks.npz").exists() and (not args.video or vdir.name in args.video):
            rows += evaluate_video(vdir.name, prompts, rng)
    df = pd.DataFrame(rows)
    out = C.QC_DIR / "report.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(df[df.n_include + df.frames_with_mask > 0].to_string(index=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
