"""
Rebuild a missing *_validation_summary.csv from the raw *_validation.csv samples.

Per-point accuracy = mean angle between gaze and target over the point's valid samples, from an eye
assumed centered in front of the screen (screen geometry: METHODS_DECISIONS.md, 2025-09-25). Tobii
used the measured eye position, so this is an approximation (METHODS_DECISIONS.md, 2026-11-03).
The "mean" row is the average of the point accuracies, as Tobii's is. Pixel columns are left blank.

    python scripts/preprocessing/qc/rebuild_validation_summary.py path/to/PID_..._validation.csv [--check]

--check compares against an existing summary instead of writing one.
"""
import argparse
import os

import numpy as np
import pandas as pd

W_CM, H_CM, D_CM = 53.14, 29.89, 70.0
W_PX, H_PX = 1920, 1080
COLS = ["point", "point_index", "presentation_order", "stimulus_x", "stimulus_y", "validation_step",
        "Mean_accuracy_degrees_left", "Mean_accuracy_degrees_right",
        "Mean_accuracy_pixels_left", "Mean_accuracy_pixels_right"]


def angle_deg(x, y, tx, ty):
    def ray(px, py):
        return np.stack([(px - W_PX / 2) * W_CM / W_PX, (py - H_PX / 2) * H_CM / H_PX, np.full_like(px, D_CM)], -1)
    a, b = ray(x, y), ray(tx, ty)
    cos = (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rebuild(raw):
    v = raw.copy()
    for e in ("left", "right"):
        valid = v[f"{e}_valid"].astype(str).str.lower().isin(["true", "1"])
        acc = angle_deg(v[f"{e}_x"].to_numpy(float), v[f"{e}_y"].to_numpy(float),
                        v["stimulus_x"].to_numpy(float), v["stimulus_y"].to_numpy(float))
        v[f"Mean_accuracy_degrees_{e}"] = np.where(valid, acc, np.nan)
    rows = []
    for step, s in v.groupby("validation_step", sort=False):
        pts = (s.groupby("presentation_order", sort=True)
                .agg(point_index=("point_index", "first"), stimulus_x=("stimulus_x", "first"),
                     stimulus_y=("stimulus_y", "first"),
                     Mean_accuracy_degrees_left=("Mean_accuracy_degrees_left", "mean"),
                     Mean_accuracy_degrees_right=("Mean_accuracy_degrees_right", "mean"))
                .reset_index())
        pts["point"] = [f"Point {i}" for i in pts["presentation_order"]]
        pts["validation_step"] = step
        mean = dict(point="mean", validation_step=step,
                    Mean_accuracy_degrees_left=pts["Mean_accuracy_degrees_left"].mean(),
                    Mean_accuracy_degrees_right=pts["Mean_accuracy_degrees_right"].mean())
        rows += [mean] + pts.to_dict("records")
    out = pd.DataFrame(rows).reindex(columns=COLS)
    out[["Mean_accuracy_degrees_left", "Mean_accuracy_degrees_right"]] = \
        out[["Mean_accuracy_degrees_left", "Mean_accuracy_degrees_right"]].round(4)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("validation_csv")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    out_path = args.validation_csv.replace("_validation.csv", "_validation_summary.csv")
    new = rebuild(pd.read_csv(args.validation_csv))
    if args.check:
        old = pd.read_csv(out_path)
        m = old.merge(new, on=["validation_step", "point"], suffixes=("_logged", "_rebuilt"))
        for e in ("left", "right"):
            d = m[f"Mean_accuracy_degrees_{e}_rebuilt"] - m[f"Mean_accuracy_degrees_{e}_logged"]
            print(f"{e}: n={len(d)} diff mean {d.mean():+.3f}, max |diff| {d.abs().max():.3f}")
        return
    if os.path.exists(out_path):
        raise SystemExit(f"{out_path} exists; use --check to compare")
    new.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({new['validation_step'].nunique()} validations)")


if __name__ == "__main__":
    main()
