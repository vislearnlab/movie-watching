import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from plan import frame_of
from prepare_prompts import CORR_COLS, apply_corrections

FPS = 29.97
SHOTS = [(0, 63), (63, 104), (104, 241)]


def corr(**kw):
    return pd.DataFrame([{c: kw.get(c) for c in CORR_COLS}])


def rows():
    return [dict(prompt_idx_raw="7", prompt_name="muppet face", sam_text="muppet face",
                 start_ms=0, end_ms=3000, clicks=[], boxes=[])]


def test_shot_window_maps_back_to_exactly_that_shot():
    out = apply_corrections(rows(), corr(video="v", kind="prompt", prompt_idx="c1", prompt_name="dog face", shot=1),
                            SHOTS, FPS)
    new = out[-1]
    assert (frame_of(new["start_ms"], FPS), frame_of(new["end_ms"], FPS)) == (63, 104)
    assert new["sam_text"] == "dog face"


def test_click_correction_lands_on_its_frame():
    out = apply_corrections(rows(), corr(video="v", kind="include", prompt_idx="7", frame=101, x=5, y=6), SHOTS, FPS)
    (c,) = out[0]["clicks"]
    assert (frame_of(c["t_ms"], FPS), c["x"], c["y"], c["label"]) == (101, 5, 6, 1)


def test_box_and_text_corrections_attach_to_the_prompt():
    c = pd.concat([corr(video="v", kind="box", prompt_idx="7", frame=70, x0=1, y0=2, x1=30, y1=40),
                   corr(video="v", kind="text", prompt_idx="7", sam_text="blue furry monster head")])
    (r,) = apply_corrections(rows(), c, SHOTS, FPS)
    assert r["boxes"] == [dict(frame=70, x0=1, y0=2, x1=30, y1=40)]
    assert (r["sam_text"], r["prompt_name"]) == ("blue furry monster head", "muppet face")


def test_unknown_prompt_fails_loudly():
    with pytest.raises(KeyError):
        apply_corrections(rows(), corr(video="v", kind="include", prompt_idx="99", frame=1, x=1, y=1), SHOTS, FPS)


def test_window_correction_drops_clicks_outside_the_new_window():
    r = rows()
    r[0]["clicks"] = [dict(t_ms=1000, x=1, y=1, label=1), dict(t_ms=5000, x=2, y=2, label=1)]
    (out,) = apply_corrections(r, corr(video="v", kind="window", prompt_idx="7", shot=2), SHOTS, FPS)
    assert [c["x"] for c in out["clicks"]] == [2]       # shot 2 is frames 104–240 (3.47–8.04 s)


def test_label_correction_renames_the_prompt_and_keeps_its_text():
    (r,) = apply_corrections(rows(), corr(video="v", kind="label", prompt_idx="7", prompt_name="grover face"), SHOTS, FPS)
    assert (r["prompt_name"], r["sam_text"]) == ("grover face", "muppet face")


def test_label_correction_cannot_change_the_category():
    with pytest.raises(AssertionError):
        apply_corrections(rows(), corr(video="v", kind="label", prompt_idx="7", prompt_name="muppet hand"), SHOTS, FPS)


def test_correction_clicks_are_marked_as_review_clicks():
    out = apply_corrections(rows(), corr(video="v", kind="exclude", prompt_idx="7", frame=101, x=5, y=6), SHOTS, FPS)
    assert out[0]["clicks"][0]["review"] is True
