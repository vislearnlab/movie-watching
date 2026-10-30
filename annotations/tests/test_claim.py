import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from annotate_videos import claim, write_stamp, iteration_of


def test_a_shot_is_claimed_once(tmp_path):
    base = tmp_path / "shards" / "face_hand_shot003"
    assert claim(base, "sig") and not claim(base, "sig")


def test_finished_shot_is_not_claimed(tmp_path):
    base = tmp_path / "shards" / "face_hand_shot004"
    base.parent.mkdir(parents=True)
    Path(f"{base}.npz").touch()
    write_stamp(base, "sig")
    assert not claim(base, "sig")


def test_changed_tasks_rerun_the_shot_and_count_iterations(tmp_path):
    base = tmp_path / "shards" / "face_hand_shot005"
    base.parent.mkdir(parents=True)
    Path(f"{base}.npz").touch()          # pass-1 shard: no stamp, counts as iteration 1
    assert iteration_of(base) == 1
    write_stamp(base, "old")
    assert iteration_of(base) == 2 and claim(base, "new")
