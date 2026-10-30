import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

import cv2

import config as C
from refine_cuts import refine_cut


def test_pixar_cut_one_frame_early_is_found():
    # checked by eye: frame 799 already shows the two small birds; the reference cut (33.367 s) maps to 800
    cap = cv2.VideoCapture(str(C.video_path("pixar_birds")))
    assert refine_cut(cap, 800) == (799, "hard")


def test_fade_through_black_cuts_at_first_fade_in_frame():
    # slow_people 15.099 s: mean intensity falls to 1.1 at f453 and rises from f454
    cap = cv2.VideoCapture(str(C.video_path("slow_people")))
    f = round(15.099 * cap.get(cv2.CAP_PROP_FPS))
    assert refine_cut(cap, f) == (454, "fade")
