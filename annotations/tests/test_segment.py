import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from plan import Click
from segment import Crop, Det, same_object, slack_frames, tracks_under


def square(x, y, r=10):
    m = np.zeros((720, 1280), bool)
    m[y - r:y + r, x - r:x + r] = True
    return Crop.of(m)


def test_slack_order_and_clipping():
    assert slack_frames(5, 100) == [5, 4, 6, 3, 7]
    assert slack_frames(0, 3) == [0, 1, 2]


def test_click_one_frame_off_finds_track():
    outs = {f: [] for f in range(10)}
    outs[6] = [Det(7, square(400, 300), 0.9)]
    assert tracks_under(outs, Click(1, 5, 400, 300, 1), 5) == (6, [7])


def test_click_beyond_slack_finds_nothing():
    outs = {f: [] for f in range(10)}
    outs[8] = [Det(7, square(400, 300), 0.9)]
    assert tracks_under(outs, Click(1, 5, 400, 300, 1), 5) == (5, [])


def test_crop_distance():
    a = square(400, 300)                       # covers x 390..409
    assert a.dist(400, 300) == 0
    assert a.dist(409 + 25, 300) == 25


def test_nearest_track_wins_within_tolerance():
    outs = {0: [Det(1, square(400, 300), 0.9), Det(2, square(440, 300), 0.9)]}
    # click at x=455: 6 px from track 2, 46 px from track 1
    assert tracks_under(outs, Click(1, 0, 455, 300, 1), 0) == (0, [2])


def test_click_on_hair_above_face_counts():
    # annotators click the head: 25 px above the face mask must still hit it
    outs = {0: [Det(1, square(400, 300), 0.9)]}
    assert tracks_under(outs, Click(1, 0, 400, 300 - 10 - 25, 1), 0) == (0, [1])


def test_same_object_uses_overlap_of_smaller_mask():
    head = square(400, 300, r=60)
    eyes = square(400, 280, r=15)              # fully inside the head, IoU ≈ 0.06
    assert same_object(head, eyes)
    assert not same_object(head, square(700, 300, r=15))


def test_ear_touching_face_is_merged():
    from segment import attach_ears
    face, ear = square(400, 300, 30), square(400 + 30 + 8, 300, 8)   # ear starts at the face's edge
    out = attach_ears([Det(1, face, 0.9)], [ear])
    assert out[0].obj_id == 1 and out[0].mask.area == face.area + ear.area


def test_far_or_oversized_ear_is_not_merged():
    from segment import attach_ears
    face = square(400, 300, 30)
    far = square(400 + 30 + 40, 300, 8)        # x 462..477: 33 px from the face, > EAR_GAP_PX
    huge = square(400 + 30 + 40, 300, 40)      # bigger than the face
    half = square(400 + 30 + 22, 300, 22)      # touching, but 54% of the face's area > EAR_MAX_FRAC
    assert attach_ears([Det(1, face, 0.9)], [far, huge, half])[0].mask.area == face.area


def test_ear_between_two_faces_goes_to_nearest():
    from segment import attach_ears
    a, b = square(300, 300, 30), square(400, 300, 30)   # a: x 270..329, b: x 370..429
    ear = square(348, 300, 8)                            # x 340..355: 11 px from a, 15 px from b
    out = attach_ears([Det(1, a, 0.9), Det(2, b, 0.9)], [ear])
    assert out[0].mask.area == a.area + ear.area and out[1].mask.area == b.area


def test_gap_between_masks():
    assert square(400, 300, 10).gap(square(400 + 10 + 5, 300, 5)) == 1   # x 390..409 vs 410..419: adjacent


def test_track_starting_after_hotstart_is_found_with_wide_slack():
    # sesameus_1 shot 3: the muppet track appears ~10 frames after the shot starts, where its click is
    from segment import NEARBY_FRAMES
    outs = {f: [] for f in range(30)}
    for f in range(11, 30):
        outs[f] = [Det(1, square(400, 300), 0.9)]
    c = Click(4, 0, 400, 300, 1)
    assert tracks_under(outs, c, 0) == (0, [])
    assert tracks_under(outs, c, 0, slack=NEARBY_FRAMES) == (11, [1])


def test_generic_track_kept_only_when_anchored():
    # sesameus_2 shot 10: "muppet face" fell back to "face", which also found an adult and a child
    from segment import unanchored_generic
    muppet_eyes, muppet_head = square(900, 200, 15), square(900, 210, 50)
    adult, child = square(500, 200, 30), square(100, 300, 20)
    frames = {0: [Det(5000, muppet_eyes, 0.6), Det(1, muppet_head, 0.7), Det(2, adult, 0.8), Det(3, child, 0.7)],
              1: [Det(1, muppet_head, 0.7), Det(2, adult, 0.8), Det(3, child, 0.7)]}
    tracks = {5000: dict(source="text", group="held", clicked=False),
              1: dict(source="generic", group="session", clicked=False),
              2: dict(source="generic", group="session", clicked=False),
              3: dict(source="generic", group="session", clicked=True)}   # an annotator clicked the child
    assert unanchored_generic(frames, tracks) == {2}


def test_late_tracks_are_those_starting_inside_the_hotstart_window():
    from segment import LATE_START, late_tracks
    outs = {f: [] for f in range(60)}
    for f in range(0, 60):
        outs[f].append(Det(1, square(100, 100), 0.9))      # present from the first frame
    for f in range(12, 60):
        outs[f].append(Det(2, square(400, 300), 0.9))      # hotstart gap: starts at 12
    for f in range(LATE_START + 10, 60):
        outs[f].append(Det(3, square(800, 300), 0.9))      # walks in later: not backfilled
    assert late_tracks(outs) == {2: 12}


def test_box_replaces_partial_mask_inside_it_but_not_neighbours():
    from plan import Box
    from segment import replaced_by_box
    # muppet hat+nose (inside the head box) is replaced; the face next to it is kept
    dets = [Det(1, square(300, 250, 30), 0.9), Det(2, square(520, 300, 40), 0.9)]
    assert replaced_by_box(dets, Box(1, 0, 200, 150, 450, 450)) == {1}


def test_second_box_on_a_box_seeded_object_refines_it():
    from segment import box_target
    tracks = {0: dict(source="text"), 10000: dict(source="box"), 10001: dict(source="click_box")}
    assert box_target({0, 10000}, tracks, next_id=10002) == 10000     # refine the earlier box's object
    assert box_target({0, 10001}, tracks, next_id=10002) == 10002     # anything else: a new object


def test_exclude_recheck_spares_review_boxes_but_drops_drifted_click_seeds():
    from segment import drifted_onto_exclude
    outs = {0: [Det(10000, square(400, 300), 0.9), Det(10001, square(800, 300), 0.9)]}
    tracks = {10000: dict(source="box"), 10001: dict(source="click_box")}
    excludes = [Click(1, 0, 400, 300, 0), Click(1, 0, 800, 300, 0)]
    assert drifted_onto_exclude(outs, excludes, 0, tracks) == {10001}


def rect(x0, y0, x1, y1):
    m = np.zeros((720, 1280), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_forearm_part_is_cut_from_hand_leaving_exactly_the_hand():
    from segment import subtract_parts
    hand, forearm = rect(400, 300, 440, 340), rect(440, 310, 600, 330)
    (d,) = subtract_parts([Det(1, Crop.of(hand | forearm), 0.9)], [Crop.of(rect(438, 305, 620, 335))])
    full = np.zeros((720, 1280), bool); d.mask.paste(full)
    assert (full == (hand & ~rect(438, 305, 620, 335))).all()


def test_part_covering_most_of_a_face_leaves_it_unchanged():
    from segment import subtract_parts
    face = Crop.of(rect(400, 300, 500, 400))
    (d,) = subtract_parts([Det(1, face, 0.9)], [Crop.of(rect(400, 300, 500, 385))])   # 85% of the face
    assert (d.mask.bbox, d.mask.area) == (face.bbox, face.area)


def test_far_small_blob_is_dropped_but_strand_split_halves_are_kept():
    from segment import drop_strays
    face, blob = rect(400, 300, 500, 400), rect(700, 600, 722, 622)      # blob: 4.8% of the face, 200 px away
    assert drop_strays(Crop.of(face | blob)).area == face.sum()
    halves = rect(400, 300, 448, 400) | rect(452, 300, 500, 400)         # one face split by a 4 px hair strand
    assert drop_strays(Crop.of(halves)).area == 2 * 48 * 100


def test_review_exclude_trims_a_box_track_from_the_click_away_from_its_box():
    from segment import trim_box_tracks
    outs = {lf: [Det(10000, square(400, 300), 0.9)] for lf in range(40)}
    tracks = {10000: dict(source="box", box_frames=[10])}
    review = Click(1, 30, 400, 300, 0, review=True)
    assert trim_box_tracks(outs, [review], 0, tracks) == {(10000, lf) for lf in range(30, 40)}
    before = Click(1, 5, 400, 300, 0, review=True)       # before the box: trim back to the session start
    assert trim_box_tracks(outs, [before], 0, tracks) == {(10000, lf) for lf in range(0, 6)}
    annotator = Click(1, 30, 400, 300, 0)
    assert trim_box_tracks(outs, [annotator], 0, tracks) == set()


def test_trim_stops_at_the_next_box_frame():
    from segment import trim_box_tracks
    outs = {lf: [Det(10000, square(400, 300), 0.9)] for lf in range(40)}
    tracks = {10000: dict(source="box", box_frames=[10, 35])}
    assert trim_box_tracks(outs, [Click(1, 20, 400, 300, 0, review=True)], 0, tracks) == \
        {(10000, lf) for lf in range(20, 35)}


def test_exclude_on_a_box_frame_keeps_the_box_track():
    from segment import trim_box_tracks
    outs = {lf: [Det(10000, square(400, 300), 0.9)] for lf in range(40)}
    tracks = {10000: dict(source="box", box_frames=[30])}
    assert trim_box_tracks(outs, [Click(1, 30, 400, 300, 0, review=True)], 0, tracks) == set()
