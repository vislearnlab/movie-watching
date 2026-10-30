"""
Turn prompt windows and clicks into per-shot SAM3 tasks.

A task is one SAM3 session: one text prompt on one shot. Prompts with the same
text in the same shot share a task, so the same objects are never segmented
twice under one label. Shots come from scene-cuts/all_clips_cuts_frames.csv,
so no mask is ever propagated across a cut.
"""

import json
import math
from dataclasses import dataclass, field

import pandas as pd

import config as C

MIN_OVERLAP_S    = 1.0   # a prompt covers a shot if it overlaps ≥ min(this, MIN_OVERLAP_FRAC · shot)
MIN_OVERLAP_FRAC = 0.5
SNAP_S           = 1.0   # window edges this close to a cut are moved onto the cut
CONTEXT_S        = 1.0   # extra frames given to SAM3 on each side of the output range
CLICK_SLACK      = 2     # Datavyu click times can be off by this many frames either way


@dataclass
class Member:
    prompt_idx: int
    prompt_name: str
    out: tuple[int, int]          # global frames [a, b) written to the outputs


@dataclass
class Click:
    prompt_idx: int
    frame: int                    # global frame index
    x: int
    y: int
    label: int                    # 1 include, 0 exclude
    review: bool = False          # from a review correction (newer than the annotators' clicks)


@dataclass
class Box:
    prompt_idx: int
    frame: int                    # global frame index
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class Task:
    stem: str
    shot_idx: int
    shot: tuple[int, int]         # global frames [s, e)
    sam_text: str
    generic_text: str
    category: str
    members: list[Member] = field(default_factory=list)
    clicks: list[Click] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    auto: bool = False            # a per-video "… (unlabelled)" prompt; loses every dedup
    exclusive: bool = False       # the only real prompt of its category in the shot: may keep unanchored generic tracks

    @property
    def frames(self) -> tuple[int, int]:
        """Session range: union of member outputs plus context, clipped to the shot."""
        a = min(m.out[0] for m in self.members)
        b = max(m.out[1] for m in self.members)
        return max(self.shot[0], a - self.ctx), min(self.shot[1], b + self.ctx)

    ctx: int = 0


def frame_of(t_ms: float, fps: float) -> int:
    """Frame on screen at t_ms (Datavyu shows the frame whose interval contains t)."""
    return math.floor(t_ms / 1000 * fps + 1e-6)


def load_shots(stem: str, fps: float, n_frames: int) -> list[tuple[int, int]]:
    """[start, end) frame ranges between consecutive cuts (refined by refine_cuts.py)."""
    cuts = pd.read_csv(C.CUT_FRAMES_CSV)
    cut_frames = sorted(cuts.loc[cuts.video_name == stem, "cut_frame"])
    edges = [0] + [f for f in cut_frames if 0 < f < n_frames] + [n_frames]
    return list(zip(edges[:-1], edges[1:]))


def shot_ranges(start_f: int, end_f: int, click_frames: list[int],
                shots: list[tuple[int, int]], fps: float) -> dict[int, tuple[int, int]]:
    """Map one prompt onto shots: {shot_idx: (a, b)} output range per covered shot.

    A window shorter than one frame stands for the whole shot containing its start.
    """
    snap, min_ov = round(SNAP_S * fps), MIN_OVERLAP_S * fps
    out = {}
    for k, (s, e) in enumerate(shots):
        in_shot = [f for f in click_frames if s <= f < e]
        a, b = max(start_f, s), min(end_f, e)
        if end_f - start_f < 1:
            covered = s <= start_f < e
            a, b = s, e
        else:
            covered = b - a >= min(min_ov, MIN_OVERLAP_FRAC * (e - s))
        if not (covered or in_shot):
            continue
        if not covered:                       # only clicks place the prompt here: trust them for the shot
            out[k] = (s, e)
            continue
        a, b = min([a] + in_shot), max([b] + [f + 1 for f in in_shot])
        if a - s <= snap:
            a = s
        if e - b <= snap:
            b = e
        out[k] = (a, b)
    return out


def snap_clicks(click_frames: list[int], start_f: int, end_f: int,
                shots: list[tuple[int, int]], fps: float) -> list[int]:
    """Move clicks that land ≤ CLICK_SLACK frames outside a shot the prompt window covers into it."""
    covered = [shots[k] for k in shot_ranges(start_f, end_f, [], shots, fps)]
    out = []
    for f in click_frames:
        if not any(s <= f < e for s, e in covered):
            for s, e in covered:
                if s - CLICK_SLACK <= f < s:
                    f = s
                elif e <= f < e + CLICK_SLACK:
                    f = e - 1
        out.append(f)
    return out


def plan_video(prompts: pd.DataFrame, fps: float, n_frames: int, categories: list[str]) -> list[Task]:
    """Tasks for one video's prompt rows whose category is in categories (face | hand | other)."""
    stem = prompts.video_stem.iloc[0]
    prompts = prompts[prompts.category.isin(categories)]
    shots = load_shots(stem, fps, n_frames)
    tasks: dict[tuple[int, str, bool], Task] = {}
    for p in prompts.itertuples():
        start_f, end_f = frame_of(p.start_ms, fps), frame_of(p.end_ms, fps)
        raw = json.loads(p.clicks)
        frames = snap_clicks([frame_of(c["t_ms"], fps) for c in raw], start_f, end_f, shots, fps)
        clicks = [Click(p.prompt_idx, f, c["x"], c["y"], c["label"], c["review"]) for f, c in zip(frames, raw)]
        boxes = [Box(p.prompt_idx, **b) for b in json.loads(p.boxes)]
        ranges = shot_ranges(start_f, end_f, frames + [b.frame for b in boxes], shots, fps)
        for k, out in ranges.items():
            in_shot = lambda f: shots[k][0] <= f < shots[k][1]
            t = tasks.setdefault((k, p.sam_text, bool(p.auto)), Task(
                stem, k, shots[k], p.sam_text, p.generic_text if isinstance(p.generic_text, str) else "",
                p.category, ctx=round(CONTEXT_S * fps), auto=bool(p.auto)))
            t.members.append(Member(p.prompt_idx, p.prompt_name, out))
            t.clicks += [c for c in clicks if in_shot(c.frame)]
            t.boxes += [b for b in boxes if in_shot(b.frame)]
    def covers_shot(t):
        out = {f for m in t.members for f in range(*m.out)}
        return out >= set(range(*t.shot))

    # an auto task repeats a real prompt that already runs the generic text over the whole shot
    tasks = {key: t for key, t in tasks.items() if not (t.auto and any(
        not o.auto and (o.shot_idx, o.category) == (t.shot_idx, t.category) and o.sam_text == o.generic_text
        and covers_shot(o) for o in tasks.values()))}
    for t in tasks.values():
        if t.auto:   # an object excluded for its prompt must not come back as "(unlabelled)"
            t.clicks += [c for o in tasks.values() if not o.auto and (o.shot_idx, o.category) == (t.shot_idx, t.category)
                         for c in o.clicks if c.label == 0]
        t.exclusive = not t.auto and not any(o is not t and not o.auto and (o.shot_idx, o.category) == (t.shot_idx, t.category)
                                             for o in tasks.values())
    return sorted(tasks.values(), key=lambda t: (t.shot_idx, t.auto, t.sam_text))
