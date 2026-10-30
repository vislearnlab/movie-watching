"""
Build the SAM3 prompt table from the Datavyu exports.

For every video, the newest Datavyu round that exists (see config.ROUNDS) is
used; videos without a Datavyu export fall back to the round-0 spreadsheet.
Then the review corrections in inputs/corrections/{video}.csv are applied
(apply_corrections), and the AUTO prompts ("face (unlabelled)", "hand (unlabelled)")
cover every shot of every video with the generic text, so faces and hands with
no prompt still get a mask.

Corrections (one row each; columns CORR_COLS; coordinates in 1280×720 video pixels):
  prompt   new prompt: prompt_idx c<n>, prompt_name, optional sam_text, and shot or start_s/end_s
  text     sam_text sent to SAM3 for prompt_idx (the label and category keep prompt_name);
           "none" = no text detection, only the prompt's boxes and clicks
  window   new shot or start_s/end_s for prompt_idx; its clicks outside the new window are dropped
  box      x0 y0 x1 y1 on frame: the object in this box belongs to prompt_idx (replaces any
           track it overlaps, then propagates both ways)
  label    new prompt_name for prompt_idx (same category; sam_text unchanged)
  include / exclude   click x y on frame for prompt_idx (marked review: an exclude on a box
           track trims that track, see segment.trim_box_tracks)

Output: config.PROMPTS_CSV — one row per prompt:

  video_stem, video_block, video_path, round
  prompt_idx      Datavyu id (non-numeric ids like "14a" remapped to unique ints)
  prompt_idx_raw  Datavyu id as typed
  prompt_name     label as typed by the annotator
  sam_text        normalised text sent to SAM3 ("children faces" → "child face")
  generic_text    fallback noun ("face"/"hand") when the specific text finds nothing
  category        face | hand | other (from prompt_name)
  auto            True for the AUTO prompts
  start_ms, end_ms  prompt window (repaired, see window_note)
  window_note     why the window was changed, if it was
  clicks          JSON [{"t_ms", "x", "y", "label", "review"}] in video pixels; label 1 =
                  include, 0 = exclude; review = from a correction. t_ms is when the click was made —
                  the point is only valid on that frame.
  boxes           JSON [{"frame", "x0", "y0", "x1", "y1"}] from box corrections
  include_coords / exclude_coords   legacy [[x, y], …] lists (no times)

Clicks that reference a prompt_idx with no prompt row are reported and dropped.

Usage:
    python pipeline/prepare_prompts.py           # → config.PROMPTS_CSV
    python pipeline/prepare_prompts.py --check   # validate corrections only, write nothing
"""

import argparse
import json
import re
import sys

import cv2
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import config as C
from plan import load_shots

COORD_COLS  = ["coords.ordinal", "coords.onset", "coords.offset", "coords.prompt_idx",
               "coords.x", "coords.y", "coords.include"]
PROMPT_COLS = ["prompts.ordinal", "prompts.onset", "prompts.offset",
               "prompts.prompt_idx", "prompts.prompt_name"]

# Word-level singularisation for SAM3 text prompts. SAM3 is an instance detector,
# so the singular noun phrase is the natural query and returns every instance.
SINGULAR = {
    "faces": "face", "hands": "hand", "children": "child", "kids": "kid",
    "birds": "bird", "people": "person", "adults": "adult", "muppets": "muppet",
    "chickens": "chicken", "sheep": "sheep", "feet": "foot", "wings": "wing",
}


CORR_COLS = ["video", "kind", "prompt_idx", "prompt_name", "sam_text", "shot", "start_s", "end_s",
             "frame", "x", "y", "x0", "y0", "x1", "y1", "note"]
KINDS = {"prompt", "text", "label", "window", "box", "include", "exclude"}
AUTO = {"c999": ("face (unlabelled)", "face"), "c998": ("hand (unlabelled)", "hand")}   # id → (name, sam_text)
CORR_ID0 = 1_000_000   # prompt_idx of correction prompt c<n> in video i: CORR_ID0 + 1000·i + n


def sam_text_of(name: str) -> str:
    words = re.sub(r"\s+", " ", name.strip().lower()).split(" ")
    return " ".join(SINGULAR.get(w, w) for w in words)


def category_of(text: str) -> str:
    if re.search(r"\bface\b", text):
        return "face"
    if re.search(r"\bhand\b", text):
        return "hand"
    return "other"


def parse_int(s) -> int:
    """'1,857' → 1857 (Datavyu exports thousands separators)."""
    return int(str(s).replace(",", "").strip())


def screen_to_video(sx, sy, coord_system, W, H):
    if coord_system == "fullscreen":
        vx, vy = sx / C.SCREEN_W * W, sy / C.SCREEN_H * H
    elif coord_system == "letterbox":
        lb = C.LETTERBOX
        vx = (sx - lb["left"]) / (lb["right"] - lb["left"]) * W
        vy = (sy - lb["top"]) / (lb["bottom"] - lb["top"]) * H
    else:
        raise ValueError(coord_system)
    return min(max(round(vx), 0), W - 1), min(max(round(vy), 0), H - 1)


def video_meta(stem):
    cap = cv2.VideoCapture(str(C.video_path(stem)))
    meta = dict(W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=cap.get(cv2.CAP_PROP_FPS), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta


def load_datavyu(stem, cfg, meta):
    for rnd in C.ROUNDS:
        path = C.EXPORTS_DIR / rnd / cfg["datavyu"]
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No Datavyu export for {stem} in {C.ROUNDS}")
    csys = cfg["coords"][rnd]

    raw = pd.read_csv(path, dtype=str)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    prompts = raw.loc[raw["prompts.ordinal"].notna(), PROMPT_COLS]
    coords  = raw.loc[raw["coords.ordinal"].notna(), COORD_COLS]

    clicks: dict[str, list] = {}
    for _, c in coords.iterrows():
        x, y = screen_to_video(parse_int(c["coords.x"]), parse_int(c["coords.y"]), csys, meta["W"], meta["H"])
        pid = c["coords.prompt_idx"].strip()
        pid = C.CLICK_REMAP.get((stem, pid), pid)
        clicks.setdefault(pid, []).append(dict(
            t_ms=parse_int(c["coords.onset"]), x=x, y=y,
            label=1 if str(c["coords.include"]).strip().lower() == "yes" else 0, review=False,
        ))

    rows = []
    for _, p in prompts.iterrows():
        pid = p["prompts.prompt_idx"].strip()
        rows.append(dict(
            prompt_idx_raw=pid, prompt_name=str(p["prompts.prompt_name"]).strip(),
            start_ms=parse_int(p["prompts.onset"]), end_ms=parse_int(p["prompts.offset"]),
            clicks=sorted(clicks.pop(pid, []), key=lambda c: c["t_ms"]), boxes=[],
        ))
        rows[-1]["sam_text"] = sam_text_of(rows[-1]["prompt_name"])
    for pid, cl in clicks.items():
        print(f"  WARNING {stem}: {len(cl)} click(s) for prompt_idx {pid!r} with no prompt row — dropped")
    print(f"  {stem:15s} {rnd} [{csys}] {path.name}: {len(rows)} prompts, "
          f"{sum(len(r['clicks']) for r in rows)} clicks")
    return rnd, rows


def load_round0(stem, cfg):
    df = pd.read_csv(C.ROUND0_CSV)
    df = df[df["video_block"] == cfg["block"]]
    rows = [dict(prompt_idx_raw=str(r.prompt_idx), prompt_name=str(r.prompt).strip(),
                 sam_text=sam_text_of(str(r.prompt)),
                 start_ms=int(round(r.start_time_s * 1000)), end_ms=int(round(r.end_time_s * 1000)),
                 clicks=[], boxes=[]) for r in df.itertuples()]
    print(f"  {stem:15s} round0 (spreadsheet): {len(rows)} prompts, no clicks")
    return "round0", rows


def repair_window(r, duration_ms):
    """Datavyu leaves offset=0 for an unset end and sometimes end < start.

    Clicks are the most reliable timing evidence, so a broken window becomes
    [start, max(start, last click)] and is flagged; annotate_videos.py then
    snaps it to the enclosing shot.
    """
    s, e = r["start_ms"], r["end_ms"]
    note = ""
    if e <= s:
        click_ts = [c["t_ms"] for c in r["clicks"]]
        e = max([s] + click_ts)
        note = f"end<=start ({r['start_ms']}–{r['end_ms']}); window set to start..last click, snapped to shot"
    s = max(0, min(s, duration_ms))
    e = max(0, min(e, duration_ms))
    return s, e, note


def load_corrections() -> pd.DataFrame:
    """All inputs/corrections/*.csv rows, in file order."""
    files = sorted(C.CORRECTIONS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=CORR_COLS)
    df = pd.concat([pd.read_csv(p, dtype={"prompt_idx": str}, comment="#") for p in files], ignore_index=True)
    assert list(df.columns) == CORR_COLS, f"corrections columns must be {CORR_COLS}, got {list(df.columns)}"
    bad = df[~df.kind.isin(KINDS) | ~df.video.isin(C.VIDEOS)]
    assert bad.empty, f"unknown kind or video in corrections:\n{bad}"
    return df


def apply_corrections(rows: list[dict], corr: pd.DataFrame, shots: list[tuple[int, int]], fps: float) -> list[dict]:
    """rows with one video's corrections applied (see module docstring); unknown prompt_idx raises KeyError."""
    by_id = {r["prompt_idx_raw"]: r for r in rows}

    def window(c):
        if pd.notna(c.shot):
            s, e = shots[int(c.shot)]
            return s * 1000 / fps, e * 1000 / fps
        assert pd.notna(c.start_s) and pd.notna(c.end_s), f"{c.kind} {c.prompt_idx}: needs shot or start_s/end_s"
        return c.start_s * 1000, c.end_s * 1000

    for c in corr.itertuples():
        pid = str(c.prompt_idx).strip()
        if c.kind == "prompt":
            assert re.fullmatch(r"c\d+", pid) and pid not in by_id, f"new prompt id must be a fresh c<n>: {pid!r}"
            s, e = window(c)
            by_id[pid] = dict(prompt_idx_raw=pid, prompt_name=c.prompt_name.strip(),
                              sam_text=c.sam_text if isinstance(c.sam_text, str) else sam_text_of(c.prompt_name),
                              start_ms=s, end_ms=e, clicks=[], boxes=[])
            rows.append(by_id[pid])
            continue
        if pid not in by_id:
            raise KeyError(f"{c.video} {c.kind}: prompt_idx {pid!r} is not a prompt_idx_raw of this video. Sheet labels "
                           f"show prompt_idx (a number); corrections need prompt_idx_raw from prompts.csv. "
                           f"Known: {sorted(by_id)}")
        r = by_id[pid]
        if c.kind == "text":
            r["sam_text"] = c.sam_text.strip()
        elif c.kind == "label":
            name = c.prompt_name.strip()
            assert category_of(sam_text_of(name)) == category_of(sam_text_of(r["prompt_name"])), \
                f"{c.video} label {pid}: {r['prompt_name']!r} → {name!r} changes the category"
            r["prompt_name"] = name
        elif c.kind == "window":   # the reviewer moved the prompt: clicks outside its new window were misplaced
            r["start_ms"], r["end_ms"] = window(c)
            r["clicks"] = [k for k in r["clicks"] if r["start_ms"] <= k["t_ms"] <= r["end_ms"]]
        elif c.kind == "box":
            r["boxes"].append(dict(frame=int(c.frame), x0=int(c.x0), y0=int(c.y0), x1=int(c.x1), y1=int(c.y1)))
        else:
            r["clicks"].append(dict(t_ms=(int(c.frame) + 0.5) * 1000 / fps, x=int(c.x), y=int(c.y),
                                    label=int(c.kind == "include"), review=True))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate corrections, write nothing")
    args = ap.parse_args()
    corrections = load_corrections()
    out = []
    for vi, (stem, cfg) in enumerate(C.VIDEOS.items()):
        meta = video_meta(stem)
        duration_ms = int(meta["n"] / meta["fps"] * 1000)
        rnd, rows = load_datavyu(stem, cfg, meta) if cfg["datavyu"] else load_round0(stem, cfg)
        n_orig = len(rows)
        rows += [dict(prompt_idx_raw=pid, prompt_name=name, sam_text=text,       # corrections may exclude on these
                      start_ms=0, end_ms=duration_ms, clicks=[], boxes=[]) for pid, (name, text) in AUTO.items()
                 if text in cfg.get("auto", ("face", "hand"))]
        corr = corrections[corrections.video == stem]
        rows = apply_corrections(rows, corr, load_shots(stem, meta["fps"], meta["n"]), meta["fps"])
        if len(corr):
            print(f"  {stem:15s} {len(corr)} corrections ({sum(r["prompt_idx_raw"].startswith("c") for r in rows[n_orig:]) - sum(r["prompt_idx_raw"] in AUTO for r in rows[n_orig:])} new prompts)")
        for i, r in enumerate(rows):
            s, e, note = repair_window(r, duration_ms)
            cat = category_of(sam_text_of(r["prompt_name"]))
            fix = i >= n_orig
            out.append(dict(
                video_stem=stem, video_block=cfg["block"], video_path=str(C.video_path(stem)),
                round="correction" if fix else rnd,
                prompt_idx=CORR_ID0 + 1000 * vi + int(r["prompt_idx_raw"][1:]) if fix else None,
                prompt_idx_raw=r["prompt_idx_raw"], prompt_name=r["prompt_name"],
                sam_text=r["sam_text"], generic_text=cat if cat != "other" else "",
                category=cat, auto=r["prompt_idx_raw"] in AUTO, start_ms=s, end_ms=e, window_note=note,
                clicks=json.dumps(r["clicks"]), boxes=json.dumps(r["boxes"]),
                include_coords=json.dumps([[c["x"], c["y"]] for c in r["clicks"] if c["label"] == 1]),
                exclude_coords=json.dumps([[c["x"], c["y"]] for c in r["clicks"] if c["label"] == 0]),
            ))

    df = pd.DataFrame(out)
    # prompt_idx must be a unique int (it doubles as a SAM3 obj_id seed and output key);
    # keep numeric Datavyu ids where they are unique across all videos, remap the rest.
    # Correction and auto prompts already have their own fixed range (CORR_ID0).
    datavyu = df.prompt_idx.isna()
    raw = df.loc[datavyu, "prompt_idx_raw"].astype(str)
    numeric = raw.str.fullmatch(r"\d+")
    dup = raw.duplicated(keep=False)
    next_id = raw[numeric].astype(int).max() + 1
    idx = []
    for r, is_num, is_dup in zip(raw, numeric, dup):
        if is_num and not is_dup:
            idx.append(int(r))
        else:
            idx.append(next_id); next_id += 1
    assert next_id < CORR_ID0
    df.loc[datavyu, "prompt_idx"] = idx
    df["prompt_idx"] = df.prompt_idx.astype(int)
    assert df["prompt_idx"].is_unique

    n_fix = (df["window_note"] != "").sum()
    print(f"\n{len(df)} prompts ({(df.category == 'face').sum()} face, {(df.category == 'hand').sum()} hand, "
          f"{(df.category == 'other').sum()} other; {df.auto.sum()} auto); {len(corrections)} corrections; "
          f"{n_fix} broken windows repaired")
    for r in df[df["window_note"] != ""].itertuples():
        print(f"  {r.video_stem} #{r.prompt_idx_raw} {r.prompt_name!r}: {r.window_note}")
    if args.check:
        print("corrections OK (--check: nothing written)")
        return
    df.to_csv(C.PROMPTS_CSV, index=False)
    print(f"Saved → {C.PROMPTS_CSV}")


if __name__ == "__main__":
    main()
