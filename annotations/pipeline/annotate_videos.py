"""
Shot-based SAM3 annotation of every prompt in config.PROMPTS_CSV.

  run    Process shots on one GPU for the prompts in --categories (default face
         hand; "other" is the objects pass). Workers share the shot list, longest
         shot first: each claims a shot by creating {shard}.lock, so any number of
         workers can run at once. Each finished shot is written atomically to
         {output}/{video}/shards/{categories}_shot{k}.{csv,npz,clicks.csv,json}. The
         .json stamp holds the signature of the shot's tasks (prompts, windows,
         clicks, boxes, PIPELINE_VERSION) and its iteration count. A shot reruns
         only when its signature changes, so editing inputs/corrections/ and
         rerunning redoes exactly the affected shots (delete stale *.lock files
         first; cycle.sh does).
  merge  Concatenate shards into {output}/{video}/frames.csv, masks.npz, clicks.csv.

Within a shot, two tracks that are the same object (segment.same_object) never
both keep a label on one frame. The winner is a real prompt over the auto
"face (unlabelled)" prompt, then a clicked track, then the specific text over the
generic noun, then the higher detector score. Across different prompts, a loser
that is the same object on most shared frames goes entirely.

Usage:
    python pipeline/annotate_videos.py run --gpu 0 [--video sesameus_1 --shot 3]
    python pipeline/annotate_videos.py run --gpu 0 --categories other          # objects pass
    python pipeline/annotate_videos.py merge
"""

import argparse
import dataclasses
import hashlib
import json
import time
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
from plan import plan_video
from segment import CUT_TEXT, EAR_TEXT, Det, Segmenter, attach_ears, drop_strays, same_object, subtract_parts

PIPELINE_VERSION = 4   # bump when segment/dedup logic changes, so every shot reruns
AUTO_ID0 = 100_000     # obj_id offset for auto-task tracks, which may share a label with a real prompt's tracks
BLACK_LEVEL = 12       # frames with mean intensity below this (fade troughs) get no masks
CLICK_COLS = ["video_name", "shot_idx", "sam_text", "text_used", "prompt_idx", "frame", "frame_used",
              "x", "y", "label", "status"]


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", text.strip().replace(" ", "_"))[:40]


def mask_key(prompt_idx: int, prompt_name: str, frame: int) -> str:
    return f"r{prompt_idx}_{slugify(prompt_name)}|{frame}"


def video_meta(stem):
    """(fps, n_frames, W, H); every stimulus must be full-length at the original 1280×720."""
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    meta = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    assert meta[2:] == C.VIDEO_SIZE, f"{stem}: {meta[2]}×{meta[3]}, expected {C.VIDEO_SIZE}"
    return meta


def all_units(categories, videos=None, shots=None):
    """[(stem, shot_idx, [tasks])] ordered by video and shot."""
    df = pd.read_csv(C.PROMPTS_CSV)
    units = {}
    for stem, g in df.groupby("video_stem"):
        if videos and stem not in videos:
            continue
        fps, n, _, _ = video_meta(stem)
        for t in plan_video(g, fps, n, categories):
            if shots is None or t.shot_idx in shots:
                units.setdefault((stem, t.shot_idx), []).append(t)
    return sorted((s, k, ts) for (s, k), ts in units.items())


def shard_base(stem, shot_idx, categories) -> Path:
    return Path(C.OUTPUT_DIR) / stem / "shards" / f"{'_'.join(categories)}_shot{shot_idx:03d}"


def signature(tasks) -> str:
    blob = json.dumps([PIPELINE_VERSION] + [dataclasses.asdict(t) for t in tasks], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def stamp(base: Path) -> dict | None:
    p = Path(f"{base}.json")
    return json.loads(p.read_text()) if p.exists() else None


def iteration_of(base: Path) -> int:
    """How many times this shot has been segmented (a pass-1 shard has no stamp and counts once)."""
    st = stamp(base)
    return st["iteration"] if st else int(Path(f"{base}.npz").exists())


def is_fresh(base: Path, sig: str) -> bool:
    st = stamp(base)
    return Path(f"{base}.npz").exists() and st is not None and st["sig"] == sig


def write_stamp(base: Path, sig: str) -> None:
    st = dict(sig=sig, iteration=iteration_of(base) + 1, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(f"{base}.tmp.json").write_text(json.dumps(st))
    os.replace(f"{base}.tmp.json", f"{base}.json")


def claim(base: Path, sig: str) -> bool:
    """Atomically claim a shot whose shard is missing or stale for this worker (O_EXCL on {base}.lock)."""
    if is_fresh(base, sig):
        return False
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(f"{base}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:   # another worker holds it
        return False
    return True


def read_frames(stem, a, b):
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    frames = []
    for _ in range(b - a):
        ok, f = cap.read()
        assert ok, f"{stem}: could not read frame {a + len(frames)}"
        frames.append(f)
    cap.release()
    return frames


def dedup(results):
    """{(task index, obj_id, frame)} to drop from one shot. results: [(task, TaskResult)].

    Same-object pairs: the weaker track goes on every frame where both cover the object.
    Across tasks, when that is most shared frames, it goes entirely, so an object keeps one label.
    Tracks of one task and group never overlap and are not compared.
    """
    tracks = []  # (ti, obj_id, frame→Det, group)
    for ti, (_, r) in enumerate(results):
        per_obj = {}
        for f, dets in r.frames.items():
            for d in dets:
                per_obj.setdefault(d.obj_id, {})[f] = d
        tracks += [(ti, oid, fr, r.tracks[oid]["group"]) for oid, fr in per_obj.items()]

    def rank(tr):
        ti, oid, fr, _ = tr
        meta = results[ti][1].tracks[oid]
        return (not results[ti][0].auto, meta["clicked"], meta["source"] != "generic",
                np.mean([d.prob for d in fr.values()]))

    gone = set()   # whole tracks dropped across tasks
    drop = set()
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            if (a[0], a[3]) == (b[0], b[3]) or (a[0], a[1]) in gone or (b[0], b[1]) in gone:
                continue
            shared = sorted(a[2].keys() & b[2].keys())
            same = [f for f in shared if same_object(a[2][f].mask, b[2][f].mask)]
            loser = min(a, b, key=rank)
            drop |= {(loser[0], loser[1], f) for f in same}
            if a[0] != b[0] and len(shared) >= 3 and len(same) >= 0.5 * len(shared):
                gone.add((loser[0], loser[1]))
                drop |= {(loser[0], loser[1], f) for f in loser[2]}
    return drop


def label_owner(t, tasks) -> dict:
    """frame → Member whose label t's detections take on that frame.

    A real task's frames belong to its members (earliest window first). An auto task's
    frames take the label of a real prompt of the same category whose text is the generic
    noun ("hands", "faces"), where that prompt's window covers the frame.
    """
    owner = {}
    sources = [o for o in tasks if not o.auto and o.category == t.category and o.sam_text == o.generic_text] if t.auto else []
    for task in sources + [t]:
        for m in sorted(task.members, key=lambda m: m.out[0]):
            for f in range(*m.out):
                owner.setdefault(f, m)
    return owner


def run_unit(seg, stem, shot_idx, tasks, categories, tmp_root):
    fps, _, W, H = video_meta(stem)
    s, e = tasks[0].shot
    frames = read_frames(stem, s, e)
    tmp = tempfile.mkdtemp(prefix=f"{stem}_{shot_idx}_", dir=tmp_root)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])

    results = []
    for t in tasks:
        a, b = t.frames
        sess = tempfile.mkdtemp(prefix="sess_", dir=tmp)
        for i in range(a, b):
            os.symlink(os.path.join(tmp, f"{i - s:05d}.jpg"), os.path.join(sess, f"{i - a:05d}.jpg"))
        print(f"  [{stem} shot {shot_idx}] {t.sam_text!r} frames {a}-{b} "
              f"members {[m.prompt_idx for m in t.members]} clicks {len(t.clicks)}", flush=True)
        results.append((t, seg.run(t, sess, frames[a - s:b - s], a)))
    for cat in sorted({t.category for t in tasks} & CUT_TEXT.keys()):
        sess = tempfile.mkdtemp(prefix="parts_", dir=tmp)
        for i in range(e - s):
            os.symlink(os.path.join(tmp, f"{i:05d}.jpg"), os.path.join(sess, f"{i:05d}.jpg"))
        cut = seg.parts(sess, CUT_TEXT[cat])
        ears = seg.parts(sess, EAR_TEXT) if cat == "face" else None
        for t, r in results:
            if t.category != cat:
                continue
            for f, dets in r.frames.items():
                dets = [Det(d.obj_id, drop_strays(d.mask), d.prob) for d in subtract_parts(dets, cut[f - s])]
                r.frames[f] = attach_ears(dets, ears[f - s]) if ears else dets
    shutil.rmtree(tmp)

    drop = dedup(results)
    black = {s + i for i, f in enumerate(frames) if f.mean() < BLACK_LEVEL}
    rows, masks, clicks = [], {}, []
    for ti, (t, r) in enumerate(results):
        owner = label_owner(t, [t for t, _ in results])
        for f, dets in sorted(r.frames.items()):
            m = owner.get(f)
            if m is None or f in black:
                continue
            union = np.zeros((H, W), bool)
            for d in dets:
                if (ti, d.obj_id, f) in drop:
                    continue
                d.mask.paste(union)
                bx, by, bw, bh = d.mask.bbox
                ys, xs = np.nonzero(d.mask.m)
                meta = r.tracks[d.obj_id]
                rows.append(dict(
                    video_name=stem, row_idx=m.prompt_idx, prompt_idx=m.prompt_idx, prompt=m.prompt_name,
                    sam_text=t.sam_text, text_used=r.text_used, category=t.category, shot_idx=shot_idx,
                    frame_idx_global=f, time_ms=round(f / fps * 1000, 2), obj_id=d.obj_id + AUTO_ID0 * t.auto,
                    source=meta["source"], clicked=meta["clicked"], prob=round(d.prob, 4),
                    bbox_x=bx, bbox_y=by, bbox_w=bw, bbox_h=bh,
                    mask_area_px=d.mask.area, mask_area_frac=round(d.mask.area / (W * H), 6),
                    centroid_x=int(bx + xs.mean()), centroid_y=int(by + ys.mean()),
                    video_width=W, video_height=H))
            if union.any():
                key = mask_key(m.prompt_idx, m.prompt_name, f)
                masks[key] = masks[key] | union if key in masks else union
        for entry in r.click_log if not t.auto else []:   # an auto task's clicks are copies of other prompts' excludes
            c = entry["click"]
            clicks.append(dict(video_name=stem, shot_idx=shot_idx, sam_text=t.sam_text, text_used=r.text_used,
                               prompt_idx=c.prompt_idx, frame=c.frame, frame_used=entry["frame_used"],
                               x=c.x, y=c.y, label=c.label,
                               status=entry["status"]))

    base = shard_base(stem, shot_idx, categories)
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table in ((".csv", rows), (".clicks.csv", clicks)):
        if table:
            pd.DataFrame(table).to_csv(f"{base}.tmp{suffix}", index=False)
            os.replace(f"{base}.tmp{suffix}", f"{base}{suffix}")
    for suffix, table in ((".csv", rows), (".clicks.csv", clicks)):
        if not table and Path(f"{base}{suffix}").exists():   # a rerun that found nothing
            os.remove(f"{base}{suffix}")
    np.savez_compressed(f"{base}.tmp.npz", **masks)
    os.replace(f"{base}.tmp.npz", f"{base}.npz")
    write_stamp(base, signature(tasks))                 # last: a matching stamp marks the unit done
    print(f"  [{stem} shot {shot_idx}] done: {len(rows)} rows, {len(masks)} masks, "
          f"dropped {len(drop)} duplicate detection(s)", flush=True)


def cmd_run(args):
    units = sorted(all_units(args.categories, args.video, args.shot),
                   key=lambda u: -sum(t.frames[1] - t.frames[0] for t in u[2]))
    todo = [u for u in units if not is_fresh(shard_base(*u[:2], args.categories), signature(u[2]))]
    print(f"GPU {args.gpu}: {len(todo)}/{len(units)} shots unfinished", flush=True)
    if not todo:
        return
    from dotenv import load_dotenv
    load_dotenv(C.ENV_FILE)
    seg = Segmenter(args.gpu)
    tmp_root = C.TMP_DIR
    tmp_root.mkdir(parents=True, exist_ok=True)
    for stem, k, tasks in todo:
        base = shard_base(stem, k, args.categories)
        if not claim(base, signature(tasks)):
            continue
        print(f"[claimed] {stem} shot {k} ({len(tasks)} tasks)", flush=True)
        run_unit(seg, stem, k, tasks, args.categories, tmp_root)
        os.remove(f"{base}.lock")
    seg.pred.shutdown()


def cmd_merge(args):
    for vdir in sorted(Path(C.OUTPUT_DIR).iterdir()):
        sh = vdir / "shards"
        if not sh.is_dir() or (args.video and vdir.name not in args.video):
            continue
        done = sorted(sh.glob("*.npz"))
        frames = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in done if p.with_suffix(".csv").exists()])
        clicks = [pd.read_csv(p.with_suffix(".clicks.csv")) for p in done if p.with_suffix(".clicks.csv").exists()]
        masks = {}
        for p in done:
            with np.load(p) as z:
                masks.update({k: z[k] for k in z.files})
        frames.sort_values(["frame_idx_global", "prompt_idx", "obj_id"]).to_csv(vdir / "frames.csv", index=False)
        (pd.concat(clicks) if clicks else pd.DataFrame(columns=CLICK_COLS)).to_csv(vdir / "clicks.csv", index=False)
        np.savez_compressed(vdir / "masks.npz", **masks)
        print(f"{vdir.name}: {len(done)} shards → {len(frames)} rows, {len(masks)} masks")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--categories", nargs="+", default=["face", "hand"], choices=["face", "hand", "other"])
    r.add_argument("--video", nargs="+")
    r.add_argument("--shot", nargs="+", type=int, help="shot indices (with --video), for re-runs")
    m = sub.add_parser("merge")
    m.add_argument("--video", nargs="+")
    args = ap.parse_args()
    {"run": cmd_run, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    main()
