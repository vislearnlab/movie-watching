"""
Run one plan.Task through SAM3.

1. Text prompt on the session frames; SAM3 detects and tracks every instance.
   If the specific text ("adult face") covers under MIN_COVERAGE of the output
   frames, the generic noun ("face") is run too. When another prompt of the same
   category shares the shot, a generic track keeps the prompt's label only if it
   is clicked or overlaps one of the specific text's tracks (unanchored_generic);
   otherwise "muppet face" would also label the adult next to the muppet. A
   prompt that is alone in its category (task.exclusive) keeps every generic track.
   Survivors compete with the specific tracks in annotate_videos.dedup.
2. Exclude clicks remove the track under the click.
3. Review boxes (corrections) replace every track they overlap on their frame
   (replaced_by_box) with a tracker object seeded from the box. A box on an object
   an earlier box seeded adds a conditioning frame to it (box_target).
4. Include clicks not already on a track (within ±NEARBY_FRAMES, since SAM3
   starts tracks late) seed a new tracker object from a
   box (the image detector's best candidate under the click) plus the click,
   placed on the click's frame and propagated both ways. A bare click is
   never used as a prompt: SAM picks the whole body/background for it.
5. A reviewer's exclude click on a box-seeded track trims that track from the click
   onward, on the side away from its box frames (trim_box_tracks): the object left.
6. Per-shot part passes (Segmenter.parts, one text pass per shot shared by its
   tasks; run in annotate_videos.run_unit): hair is cut from faces and forearms from
   hands (subtract_parts), detached stray pieces are dropped (drop_strays), and face
   masks get the ears that touch them (attach_ears).

Tracks that SAM3 starts a few frames late (its hotstart confirmation delay) are
re-seeded from their own first box and propagated back to the session start.

Each click is matched on the nearest frame within ±CLICK_SLACK where it lands
on something, since Datavyu click times can be a frame or two off.
"""

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

import config as C
from plan import CLICK_SLACK

sys.path.insert(0, str(C.SAM3_REPO))
from sam3.model.sam3_image_processor import Sam3Processor   # noqa: E402
from sam3.model_builder import build_sam3_video_predictor   # noqa: E402

TOL_PX     = 30     # a click within this distance of a mask counts as on it (annotators click the head)
SAME_OBJ   = 0.6    # |A∩B| / min(|A|, |B|) above this: two masks are the same object
BOX_THRESH = 0.05   # image-detector score floor when looking for a box under a click
CLICK_ID0  = 10_000 # obj_ids for click-seeded objects (detector ids count up from 0)
MIN_COVERAGE = 0.5  # specific text on fewer output frames than this: also run the generic noun
HELD_ID0   = 5_000  # obj_ids for specific-text tracks held while the session runs the generic noun
NEARBY_FRAMES = 15  # SAM3 confirms new detections after a hotstart delay, so a track can start this late
LATE_START = 20     # tracks first seen in (0, LATE_START] are hotstart gaps and get backfilled to frame 0
NO_TEXT    = "none"  # sam_text for box/click-only prompts (corrections): no text or generic pass
EAR_TEXT   = "ear"  # face masks get the ears that touch them (SAM3 "face" stops before the ears)
EAR_GAP_PX = 15
EAR_MAX_FRAC = 0.5  # an ear is at most this fraction of its face's (pre-ear) area
CUT_TEXT   = {"face": "hair", "hand": "forearm"}   # part masks cut from each category's masks
KEEP_FRAC  = 0.2    # a cut leaving less than this fraction of a mask means the part was the object: skip it
STRAY_FRAC = 0.2    # a mask piece smaller than this fraction of the largest piece …
STRAY_GAP_PX = 15   # … and farther than this from it is a stray and is dropped


@dataclass
class Crop:
    """Boolean mask stored as its bounding-box crop."""
    y0: int
    x0: int
    m: np.ndarray

    @classmethod
    def of(cls, full: np.ndarray) -> "Crop | None":
        ys, xs = np.nonzero(full)
        if len(ys) == 0:
            return None
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        return cls(int(y0), int(x0), full[y0:y1, x0:x1].copy())

    @property
    def bbox(self):  # x, y, w, h
        return self.x0, self.y0, self.m.shape[1], self.m.shape[0]

    @property
    def area(self) -> int:
        return int(self.m.sum())

    def dist(self, x: int, y: int) -> float:
        """Euclidean distance in px from (x, y) to the nearest mask pixel (0 inside)."""
        ys, xs = np.nonzero(self.m)
        return float(np.sqrt(((xs + self.x0 - x) ** 2 + (ys + self.y0 - y) ** 2).min()))

    def gap(self, other: "Crop") -> float:
        """Smallest pixel distance between the two masks (1 = adjacent, 0 = overlapping)."""
        y0, x0 = min(self.y0, other.y0), min(self.x0, other.x0)
        y1 = max(self.y0 + self.m.shape[0], other.y0 + other.m.shape[0])
        x1 = max(self.x0 + self.m.shape[1], other.x0 + other.m.shape[1])
        a, b = np.zeros((y1 - y0, x1 - x0), bool), np.zeros((y1 - y0, x1 - x0), bool)
        Crop(self.y0 - y0, self.x0 - x0, self.m).paste(a)
        Crop(other.y0 - y0, other.x0 - x0, other.m).paste(b)
        dist = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
        return float(dist[b].min())

    def union(self, other: "Crop") -> "Crop":
        y1 = max(self.y0 + self.m.shape[0], other.y0 + other.m.shape[0])
        x1 = max(self.x0 + self.m.shape[1], other.x0 + other.m.shape[1])
        full = np.zeros((y1, x1), bool)
        self.paste(full); other.paste(full)
        return Crop.of(full)

    def minus(self, other: "Crop") -> np.ndarray:
        """self.m with other's pixels removed (same shape as self.m)."""
        h, w = self.m.shape
        oh, ow = other.m.shape
        ya, yb = max(self.y0, other.y0), min(self.y0 + h, other.y0 + oh)
        xa, xb = max(self.x0, other.x0), min(self.x0 + w, other.x0 + ow)
        out = self.m.copy()
        if ya < yb and xa < xb:
            out[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0] &= \
                ~other.m[ya - other.y0:yb - other.y0, xa - other.x0:xb - other.x0]
        return out

    def paste(self, canvas: np.ndarray) -> None:
        h, w = self.m.shape
        canvas[self.y0:self.y0 + h, self.x0:self.x0 + w] |= self.m



def same_object(a: Crop, b: Crop) -> bool:
    """Overlap relative to the smaller mask, so a face region inside a whole head still matches."""
    H = max(a.y0 + a.m.shape[0], b.y0 + b.m.shape[0])
    W = max(a.x0 + a.m.shape[1], b.x0 + b.m.shape[1])
    ca, cb = np.zeros((H, W), bool), np.zeros((H, W), bool)
    a.paste(ca); b.paste(cb)
    return (ca & cb).sum() / min(a.area, b.area) > SAME_OBJ


def slack_frames(lf: int, n: int, slack: int = CLICK_SLACK) -> list[int]:
    """lf, lf-1, lf+1, … within ±slack, clipped to [0, n)."""
    order = [lf] + [lf + s * d for d in range(1, slack + 1) for s in (-1, 1)]
    return [f for f in order if 0 <= f < n]


def tracks_under(outs: dict, c, lf: int, slack: int = CLICK_SLACK) -> tuple[int, list[int]]:
    """(frame, [obj_id]) for the closest track within TOL_PX on the nearest frame within ±slack; [] if none."""
    for f in slack_frames(lf, len(outs), slack):
        dists = [(d.mask.dist(c.x, c.y), d.obj_id) for d in outs[f]]
        dists = [t for t in dists if t[0] <= TOL_PX]
        if dists:
            return f, [min(dists)[1]]
    return lf, []


def late_tracks(outs: dict[int, list["Det"]]) -> dict[int, int]:
    """{obj_id: first frame} for tracks that first appear in frames 1..LATE_START."""
    first = {}
    for f in sorted(outs):
        for d in outs[f]:
            first.setdefault(d.obj_id, f)
    return {oid: f for oid, f in first.items() if 0 < f <= LATE_START}


def unanchored_generic(frames: dict[int, list["Det"]], tracks: dict[int, dict]) -> set[int]:
    """Generic-noun tracks with no click and never the same object as a specific-text track.

    The generic noun ("face") finds every face in the shot; only those tied to the prompt's
    own detections or to a click carry the prompt's label.
    """
    anchored = {oid for oid, m in tracks.items() if m["source"] != "generic" or m["clicked"]}
    for dets in frames.values():
        specific = [d for d in dets if tracks[d.obj_id]["source"] != "generic"]
        for d in dets:
            if d.obj_id not in anchored and any(same_object(d.mask, s.mask) for s in specific):
                anchored.add(d.obj_id)
    return {oid for oid, m in tracks.items() if m["source"] == "generic"} - anchored


def replaced_by_box(dets: list["Det"], b) -> set[int]:
    """obj_ids of tracks that are the same object as the box region (a plan.Box) on its frame."""
    full = np.zeros((C.VIDEO_SIZE[1], C.VIDEO_SIZE[0]), bool)
    full[b.y0:b.y1, b.x0:b.x1] = True
    rect = Crop.of(full)
    return {d.obj_id for d in dets if same_object(d.mask, rect)}


def drifted_onto_exclude(outs: dict[int, list["Det"]], excludes, f0: int, tracks: dict[int, dict]) -> set[int]:
    """Tracks under an exclude click after seeding: a click-seeded object can drift onto an excluded spot.

    Review boxes are exempt: they are newer evidence than the annotators' excludes.
    """
    hits = {oid for c in excludes for oid in tracks_under(outs, c, c.frame - f0)[1]}
    return {oid for oid in hits if tracks[oid]["source"] != "box"}


def box_target(under: set[int], tracks: dict[int, dict], next_id: int) -> int:
    """obj_id a review box seeds: the earlier box-seeded object under it (adding a conditioning
    frame), else a new object. Two objects on one head would split its pixels between them."""
    own = sorted(oid for oid in under if tracks[oid]["source"] == "box")
    return own[0] if own else next_id


def attach_ears(faces: list["Det"], ears: list[Crop]) -> list["Det"]:
    """Union each ear into the nearest face within EAR_GAP_PX that is ≥ 1/EAR_MAX_FRAC times its size."""
    masks = [d.mask for d in faces]
    for ear in ears:
        gaps = [(masks[i].gap(ear), i) for i, d in enumerate(faces) if ear.area <= EAR_MAX_FRAC * d.mask.area]
        gaps = [g for g in gaps if g[0] <= EAR_GAP_PX]
        if gaps:
            i = min(gaps)[1]
            masks[i] = masks[i].union(ear)
    return [Det(d.obj_id, m, d.prob) for d, m in zip(faces, masks)]


def subtract_parts(dets: list["Det"], parts: list[Crop]) -> list["Det"]:
    """Each mask minus every part mask, unless that leaves under KEEP_FRAC of it."""
    out = []
    for d in dets:
        m = d.mask.m
        for p in parts:
            m = Crop(d.mask.y0, d.mask.x0, m).minus(p)
        cut = Crop.of(np.pad(m, ((d.mask.y0, 0), (d.mask.x0, 0))))
        keep = cut is not None and cut.area >= KEEP_FRAC * d.mask.area
        out.append(Det(d.obj_id, cut, d.prob) if keep else d)
    return out


def drop_strays(mask: Crop) -> Crop:
    """mask without pieces under STRAY_FRAC of the largest piece's area and over STRAY_GAP_PX from it."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    dist = cv2.distanceTransform((lab != big).astype(np.uint8), cv2.DIST_L2, 5)
    keep = lab == big
    for i in range(1, n):
        piece = lab == i
        if i != big and (areas[i - 1] >= STRAY_FRAC * areas[big - 1] or dist[piece].min() <= STRAY_GAP_PX):
            keep |= piece
    return Crop.of(np.pad(keep, ((mask.y0, 0), (mask.x0, 0))))


def trim_box_tracks(outs: dict[int, list["Det"]], excludes, f0: int, tracks: dict[int, dict]) -> set[tuple[int, int]]:
    """(obj_id, local frame) to drop: a review exclude on a box track cuts it from the click to the
    next box frame (or session end) on the side away from its nearest box frame; never on a box frame."""
    drop = set()
    for c in excludes:
        if not c.review:
            continue
        f, hit = tracks_under(outs, c, c.frame - f0)
        for oid in hit:
            if tracks[oid]["source"] != "box":
                continue
            bf = tracks[oid]["box_frames"]
            if f in bf:                    # the box on this frame says the object is there
                continue
            nearest = min(bf, key=lambda b: abs(b - f))
            if f > nearest:
                span = range(f, min([b for b in bf if b > f], default=len(outs)))
            else:
                span = range(max([b for b in bf if b < f], default=-1) + 1, f + 1)
            drop |= {(oid, lf) for lf in span if any(d.obj_id == oid for d in outs[lf])}
    return drop


@dataclass
class Det:
    obj_id: int
    mask: Crop
    prob: float


@dataclass
class TaskResult:
    """frames: global frame → detections. tracks: obj_id → {source, group, clicked}.

    source: text | generic | box | click_box. group: held | session (tracks in one group never overlap).
    """
    frames: dict[int, list[Det]]
    tracks: dict[int, dict]
    text_used: str
    click_log: list[dict]


class Segmenter:
    def __init__(self, gpu: int):
        self.pred = build_sam3_video_predictor(gpus_to_use=[gpu])
        self.proc = Sam3Processor(self.pred.model.detector, device=f"cuda:{gpu}",
                                  confidence_threshold=BOX_THRESH)

    def _req(self, **kw):
        return self.pred.handle_request(request=kw)

    def _propagate(self, sid, start=None) -> dict[int, list[Det]]:
        out = {}
        for r in self.pred.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=sid, start_frame_index=start)):
            o = r["outputs"]
            dets = []
            for oid, p, m in zip(o["out_obj_ids"], o["out_probs"], o["out_binary_masks"]):
                crop = Crop.of(m)
                if crop is not None:
                    dets.append(Det(int(oid), crop, float(p)))
            out[r["frame_index"]] = dets
        return out

    def click_box(self, frame_bgr: np.ndarray, x: int, y: int, texts: list[str]):
        """(score, box x0 y0 x1 y1) of the best image-detector candidate under the click, or None.

        Candidates containing the click beat ones merely within TOL_PX; then the higher score wins.
        """
        st = self.proc.set_image(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        best = None
        for text in texts:
            st = self.proc.set_text_prompt(text, st)
            for box, score, mask in zip(st["boxes"].cpu().numpy(), st["scores"].cpu().numpy(), st["masks"]):
                crop = Crop.of(mask[0].cpu().numpy() > 0)
                if crop is None or (d := crop.dist(x, y)) > TOL_PX:
                    continue
                key = (d == 0, float(score))
                if best is None or key > best[0]:
                    best = (key, box)
        return None if best is None else (best[0][1], best[1])

    def _text_pass(self, sid, text) -> dict[int, list[Det]]:
        self._req(type="add_prompt", session_id=sid, frame_index=0, text=text)
        return self._propagate(sid)

    def parts(self, frames_dir: str, text: str) -> dict[int, list[Crop]]:
        """frame (local to frames_dir) → masks from one text pass ("ear", "hair", …); shared by a shot's tasks."""
        sid = self._req(type="start_session", resource_path=frames_dir, offload_video_to_cpu=True)["session_id"]
        out = {lf: [d.mask for d in dets] for lf, dets in self._text_pass(sid, text).items()}
        self._req(type="close_session", session_id=sid)
        torch.cuda.empty_cache()
        return out

    def run(self, task, frames_dir: str, frames: list[np.ndarray], f0: int) -> TaskResult:
        """frames_dir/frames hold the task's session frames; f0 is the first one's global index."""
        H, W = frames[0].shape[:2]
        sid = self._req(type="start_session", resource_path=frames_dir, offload_video_to_cpu=True)["session_id"]
        in_range = [lf for m in task.members for lf in range(m.out[0] - f0, m.out[1] - f0)]

        # held: fixed specific-text tracks kept when the session moves on to the generic noun
        if task.sam_text == NO_TEXT:
            # box/click-only. SAM3 adds points only after a propagation, and a propagation needs a
            # prompt: run the generic noun once and remove everything it found.
            for oid in {d.obj_id for ds in self._text_pass(sid, task.generic_text).values() for d in ds}:
                self._req(type="remove_object", session_id=sid, obj_id=oid)
            outs = {lf: [] for lf in range(len(frames))}
        else:
            outs = self._text_pass(sid, task.sam_text)
        tracks = {d.obj_id: dict(source="text", group="session", clicked=False) for ds in outs.values() for d in ds}
        held = {lf: [] for lf in outs}
        texts = [task.sam_text]
        coverage = np.mean([bool(outs[lf]) for lf in in_range])
        if task.generic_text and task.sam_text not in (task.generic_text, NO_TEXT) and coverage < MIN_COVERAGE:
            held = {lf: [Det(d.obj_id + HELD_ID0, d.mask, d.prob) for d in ds] for lf, ds in outs.items()}
            tracks = {oid + HELD_ID0: dict(meta, group="held") for oid, meta in tracks.items()}
            outs = self._text_pass(sid, task.generic_text)
            tracks |= {d.obj_id: dict(source="generic", group="session", clicked=False)
                       for ds in outs.values() for d in ds}
            texts.append(task.generic_text)
        both = lambda: {lf: held[lf] + outs[lf] for lf in outs}

        def drop(ids):
            nonlocal outs, held
            for oid in ids:
                if tracks.pop(oid)["group"] == "session":
                    self._req(type="remove_object", session_id=sid, obj_id=oid)
            outs = {k: [d for d in v if d.obj_id not in ids] for k, v in outs.items()}
            held = {k: [d for d in v if d.obj_id not in ids] for k, v in held.items()}

        # Hotstart backfill: re-seed late-starting tracks from their own first box and
        # propagate both ways, so they reach back to the start of the session.
        late = late_tracks(outs)
        for oid, lf in late.items():
            x, y, w, h = next(d.mask.bbox for d in outs[lf] if d.obj_id == oid)
            self._req(type="add_prompt", session_id=sid, frame_index=lf, obj_id=oid,
                      points=torch.tensor([[x / W, y / H], [(x + w) / W, (y + h) / H]], dtype=torch.float32),
                      point_labels=torch.tensor([2, 3], dtype=torch.int32))
        if late:
            outs = {k: [d for d in v if d.obj_id in tracks] for k, v in
                    self._propagate(sid, start=min(late.values())).items()}

        log = []
        excludes = [c for c in task.clicks if c.label == 0]
        for c in excludes:
            f, hit = tracks_under(both(), c, c.frame - f0)
            drop(hit)
            log.append(dict(click=c, frame_used=f0 + f, status="removed" if hit else "no_track"))

        next_id = CLICK_ID0

        def seed(lf, pts, labels, source, obj_id):
            nonlocal outs
            self._req(type="add_prompt", session_id=sid, frame_index=lf, obj_id=obj_id,
                      points=torch.tensor(pts, dtype=torch.float32),
                      point_labels=torch.tensor(labels, dtype=torch.int32))
            tracks[obj_id] = dict(source=source, group="session", clicked=True)
            outs = {k: [d for d in v if d.obj_id in tracks] for k, v in self._propagate(sid, start=lf).items()}

        for b in sorted(task.boxes, key=lambda b: b.frame):
            lf = b.frame - f0
            under = replaced_by_box(both()[lf], b)
            target = box_target(under, tracks, next_id)
            box_frames = tracks[target]["box_frames"] if target in tracks else []
            drop(under - {target})
            seed(lf, [[b.x0 / W, b.y0 / H], [b.x1 / W, b.y1 / H]], [2, 3], "box", target)
            tracks[target]["box_frames"] = box_frames + [lf]
            next_id += target == next_id

        for c in sorted((c for c in task.clicks if c.label == 1), key=lambda c: c.frame):
            f, hit = tracks_under(both(), c, c.frame - f0)
            status = "already_tracked"
            if not hit:
                f, hit = tracks_under(both(), c, c.frame - f0, slack=NEARBY_FRAMES)
                status = "tracked_nearby"
            if hit:
                tracks[hit[0]]["clicked"] = True
                log.append(dict(click=c, frame_used=f0 + f, status=status))
                continue
            cands = [(b, f) for f in slack_frames(c.frame - f0, len(frames))
                     if (b := self.click_box(frames[f], c.x, c.y, texts)) is not None]
            if not cands:
                log.append(dict(click=c, frame_used=c.frame, status="no_candidate"))
                continue
            (_, (x0, y0, x1, y1)), lf = max(cands, key=lambda bf: bf[0][0])
            seed(lf, [[x0 / W, y0 / H], [x1 / W, y1 / H], [c.x / W, c.y / H]], [2, 3, 1], "click_box", next_id)
            log.append(dict(click=c, frame_used=f0 + lf, status="added"))
            next_id += 1

        drop(drifted_onto_exclude(both(), excludes, f0, tracks))
        trim = trim_box_tracks(both(), excludes, f0, tracks)
        outs = {lf: [d for d in v if (d.obj_id, lf) not in trim] for lf, v in outs.items()}

        if not task.exclusive:
            drop(unanchored_generic(both(), tracks))
        final = both()

        self._req(type="close_session", session_id=sid)
        torch.cuda.empty_cache()
        return TaskResult({f0 + lf: v for lf, v in final.items()}, tracks, "|".join(texts), log)
