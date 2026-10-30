import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from evaluate import hit


def test_hit_respects_euclidean_tolerance():
    m = np.zeros((720, 1280), bool)
    m[300, 400] = True
    assert hit([m], 430, 300, 30)          # 30 px away
    assert not hit([m], 422, 322, 30)      # 31.1 px diagonally, inside the square window
    assert not hit([], 400, 300, 30)
