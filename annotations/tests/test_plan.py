import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from plan import shot_ranges
from prepare_prompts import category_of, sam_text_of, screen_to_video

FPS = 30.0
# Sesame US 1 cuts at 2.098 s, 3.481 s, 8.018 s, 10.171 s
SHOTS = [(0, 63), (63, 104), (104, 241), (241, 305), (305, 900)]


def test_window_snaps_to_its_shot_and_ignores_spillover():
    # "adult face" 3000–8000 ms: 0.48 s spills into the 2.1–3.5 s shot, which must be dropped
    assert shot_ranges(90, 240, [], SHOTS, FPS) == {2: (104, 241)}


def test_zero_length_window_means_whole_enclosing_shot():
    assert shot_ranges(150, 150, [], SHOTS, FPS) == {2: (104, 241)}


def test_click_outside_window_adds_its_shot():
    assert shot_ranges(90, 240, [270], SHOTS, FPS) == {2: (104, 241), 3: (241, 305)}


def test_window_inside_long_shot_is_not_snapped():
    assert shot_ranges(500, 560, [], SHOTS, FPS) == {4: (500, 560)}


def test_click_extends_range_within_shot():
    assert shot_ranges(500, 560, [600], SHOTS, FPS) == {4: (500, 601)}


def test_letterbox_corners_map_to_video_corners():
    assert screen_to_video(46, 53, "letterbox", 1280, 720) == (0, 0)
    assert screen_to_video(1873, 1079, "letterbox", 1280, 720) == (1279, 719)
    assert screen_to_video(960, 540, "fullscreen", 1280, 720) == (640, 360)


def test_prompt_text_normalisation():
    assert sam_text_of("Children  Faces") == "child face"
    assert sam_text_of("small birds faces") == "small bird face"
    assert category_of("muppet hand") == "hand"
    assert category_of("bucket") == "other"


def test_click_time_maps_to_displayed_frame():
    from plan import frame_of
    # Datavyu shows the frame whose interval contains t: 3536 ms at 29.97 fps is frame 105.97 → 105
    assert frame_of(3536, 30000 / 1001) == 105
    assert frame_of(1400, 25.0) == 35          # exact boundary stays on its frame


def test_click_just_before_cut_moves_into_prompts_shot():
    from plan import snap_clicks
    shots = [(0, 119), (119, 250)]
    # frank_complex: "baby face" window and click both at 4720 ms = frame 118; the cut is at frame 119
    assert snap_clicks([118], 118, 250, shots, 25.0) == [119]
    # a click far from the cut keeps its frame (and its own shot)
    assert snap_clicks([60], 118, 250, shots, 25.0) == [60]


def test_generic_tracks_kept_only_when_no_other_prompt_shares_the_category():
    import pandas as pd
    from plan import plan_video
    # sesameindia_1 shot 23 (frames 1360–1396 @ 25 fps): one face prompt, one hand prompt, the auto face
    row = dict(video_stem="sesameindia_1", clicks="[]", boxes="[]", auto=False)
    df = pd.DataFrame([row | dict(prompt_idx=1, prompt_name="children faces", sam_text="child face",
                                  generic_text="face", category="face", start_ms=54400, end_ms=55840),
                       row | dict(prompt_idx=2, prompt_name="hands", sam_text="hand",
                                  generic_text="hand", category="hand", start_ms=54400, end_ms=55840),
                       row | dict(prompt_idx=3, prompt_name="face (unlabelled)", sam_text="face",
                                  generic_text="face", category="face", start_ms=54400, end_ms=55840, auto=True),
                       row | dict(prompt_idx=4, prompt_name="muppet face", sam_text="muppet face",
                                  generic_text="face", category="face", start_ms=0, end_ms=1000)])
    tasks = {(t.shot_idx, t.sam_text): t for t in plan_video(df, 25.0, 1500, ["face", "hand"])}
    assert tasks[(23, "child face")].exclusive and tasks[(23, "hand")].exclusive
    assert not tasks[(23, "face")].exclusive          # the auto prompt never keeps unanchored generic tracks
    assert tasks[(0, "muppet face")].exclusive


def test_auto_task_inherits_excludes_and_generic_labels():
    import json
    import pandas as pd
    from plan import plan_video
    from annotate_videos import label_owner
    # sesameindia_1 shot 1 (frames 20–209): a generic "hands" prompt on frames 75–150 with one include and one exclude
    clicks = json.dumps([dict(t_ms=4000, x=10, y=10, label=1, review=False), dict(t_ms=4400, x=20, y=20, label=0, review=False)])
    row = dict(video_stem="sesameindia_1", boxes="[]", generic_text="hand", category="hand", sam_text="hand")
    df = pd.DataFrame([row | dict(prompt_idx=2, prompt_name="hands", clicks=clicks, auto=False, start_ms=3000, end_ms=6000),
                       row | dict(prompt_idx=3, prompt_name="hand (unlabelled)", clicks="[]", auto=True,
                                  start_ms=0, end_ms=60000)])
    tasks = [t for t in plan_video(df, 25.0, 1500, ["hand"]) if t.shot_idx == 1]
    auto = next(t for t in tasks if t.auto)
    assert [(c.x, c.label) for c in auto.clicks] == [(20, 0)]
    owner = label_owner(auto, tasks)
    assert (owner[120].prompt_idx, owner[30].prompt_idx) == (2, 3)   # inside the "hands" window: its label


def test_auto_task_skipped_where_a_generic_prompt_covers_the_whole_shot():
    import pandas as pd
    from plan import plan_video
    # sesameindia_1 shot 23 (frames 1360–1396): "hands" covers it all; "face" prompt covers only part of shot 22
    row = dict(video_stem="sesameindia_1", clicks="[]", boxes="[]")
    df = pd.DataFrame([row | dict(prompt_idx=2, prompt_name="hands", sam_text="hand", generic_text="hand",
                                  category="hand", auto=False, start_ms=54400, end_ms=55840),
                       row | dict(prompt_idx=3, prompt_name="hand (unlabelled)", sam_text="hand", generic_text="hand",
                                  category="hand", auto=True, start_ms=0, end_ms=60000),
                       row | dict(prompt_idx=4, prompt_name="face (unlabelled)", sam_text="face", generic_text="face",
                                  category="face", auto=True, start_ms=0, end_ms=60000)])
    auto = {(t.shot_idx, t.category) for t in plan_video(df, 25.0, 1500, ["face", "hand"]) if t.auto}
    assert (23, "hand") not in auto and (23, "face") in auto and (22, "hand") in auto
