import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from annotate_videos import dedup
from segment import Det, TaskResult
from test_segment import square

FRAMES = range(5)


def result(tracks, frames_of=None, auto=False):
    """tracks: {obj_id: (Crop, meta)}, on every frame unless frames_of[obj_id] says otherwise."""
    frames_of = frames_of or {}
    return (SimpleNamespace(auto=auto), TaskResult({f: [Det(oid, c, 0.8) for oid, (c, _) in tracks.items() if f in frames_of.get(oid, FRAMES)]
                              for f in FRAMES},
                             {oid: meta for oid, (_, meta) in tracks.items()}, "", []))


def meta(source, group, clicked=False):
    return dict(source=source, group=group, clicked=clicked)


def test_generic_head_loses_to_specific_eyes_in_same_task():
    r = result({5000: (square(400, 280, 15), meta("text", "held")),
                0:    (square(400, 300, 60), meta("generic", "session"))})
    assert dedup([r]) == {(0, 0, f) for f in FRAMES}


def test_generic_survives_where_specific_track_is_missing():
    # pixar lone bird: specific text finds the tuft on 2 frames, generic finds the face on all 5
    r = result({5000: (square(400, 280, 15), meta("text", "held")),
                0:    (square(400, 300, 60), meta("generic", "session"))}, frames_of={5000: [0, 1]})
    assert dedup([r]) == {(0, 0, 0), (0, 0, 1)}


def test_tracks_of_one_session_are_never_compared():
    r = result({0: (square(400, 300, 60), meta("text", "session")),
                1: (square(400, 280, 15), meta("text", "session"))})
    assert dedup([r]) == set()


def test_clicked_track_beats_unclicked_specific_across_tasks():
    a = result({0: (square(400, 300, 20), meta("text", "session"))})
    b = result({10000: (square(402, 300, 20), meta("click_box", "session", clicked=True))})
    assert dedup([a, b]) == {(0, 0, f) for f in FRAMES}   # across tasks the whole track goes


def test_cross_task_brief_overlap_drops_loser_only_where_they_meet():
    # a clicked label covers the object on 2 of 5 frames: the weaker label goes on those 2 frames only
    a = result({0: (square(400, 300, 20), meta("text", "session"))})
    b = result({0: (square(402, 300, 20), meta("text", "session", clicked=True))}, frames_of={0: [3, 4]})
    assert dedup([a, b]) == {(0, 0, 3), (0, 0, 4)}


def test_auto_face_loses_even_to_unclicked_generic():
    auto = result({0: (square(400, 300, 20), meta("text", "session", clicked=True))}, auto=True)
    prompt = result({0: (square(401, 300, 20), meta("generic", "session"))})
    assert dedup([auto, prompt]) == {(0, 0, f) for f in FRAMES}
