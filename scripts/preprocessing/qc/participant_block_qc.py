"""
Per-block QC for named participants: calibration accuracy, eyes analyzed, inclusion (I2MC's <= 2 deg rule,
METHODS_DECISIONS.md 2026-11-03), and % usable gaze. Also writes the validation samples and angular rings
that plot_participant_qc.R draws on the validation screen.

Raw files are copied from data/raw/<group>/<PID>/ to data/local_cache/qc_raw/<group>/<PID>/ and read from there.

Outputs (data/results/qc/):
    <PID>_block_qc.csv            one row per block
    <PID>_trial_qc.csv            % valid samples (analyzed eyes) per trial
    <PID>_validation_samples.csv  gaze samples from the validation used for each block
    validation_rings.csv          2 deg rings around each target + reference rings around screen center

    python scripts/preprocessing/qc/participant_block_qc.py MW056 MW057
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fixations"))
import backfill_validation_outcomes as bf
import i2mc_fixations as i2mc
from rebuild_validation_summary import W_CM, H_CM, D_CM, W_PX, H_PX

ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "local_cache", "qc_raw")
OUT_DIR = os.path.join(ROOT, "data", "results", "qc")
GROUPS = ["adults", "kids", "infants"]
TARGETS = [(192, 108), (1728, 108), (960, 540), (192, 972), (1728, 972)]
REF_RINGS_DEG = [1, 2, 5, 10, 20]


def copy_participant(pid):
    """Copy one participant's raw CSVs to the local cache; returns (group, local dir) or raises."""
    for g in GROUPS:
        src = os.path.join(RAW_DIR, g, pid)
        if os.path.isdir(src):
            break
    else:
        raise FileNotFoundError(f"{pid}: no folder in data/raw/{{{','.join(GROUPS)}}}")
    csvs = [f for f in sorted(os.listdir(src)) if f.endswith(".csv") and not f.startswith(".")]
    empty = [f for f in csvs if os.path.getsize(os.path.join(src, f)) == 0]
    if not csvs or empty:
        raise ValueError(f"{pid}: {len(empty)}/{len(csvs)} raw files are empty on the server (upload incomplete?)")
    dst = os.path.join(CACHE_DIR, g, pid)
    os.makedirs(dst, exist_ok=True)
    for f in csvs:
        s, d = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.exists(d) or os.path.getmtime(s) > os.path.getmtime(d):
            shutil.copyfile(s, d)  # copy2 fails on SMB file flags
            os.utime(d, (os.path.getatime(s), os.path.getmtime(s)))
    return g, dst


def used_steps(tro, vs):
    """block_index -> validation_step used, plus attempt count; same selection as i2mc.compute_block_calibration."""
    steps = list(dict.fromkeys(vs.loc[vs["point"] == "mean", "validation_step"]))
    out = {}
    for b, first in tro.groupby("block_index")["total_trial_index"].min().items():
        pat = r"pre_validation_(\d+)" if b == 0 else rf"block_validation_trial{first}_(\d+)"
        att = sorted((int(m.group(1)), s) for s in steps if (m := re.fullmatch(pat, s)))
        out[b] = (att[-1][1] if att else None, len(att))
    return out


def ring(tx, ty, r_deg, n=180):
    """Screen-pixel polygon of all points r_deg of visual angle from (tx, ty), eye centered at D_CM."""
    t = np.array([(tx - W_PX / 2) * W_CM / W_PX, (ty - H_PX / 2) * H_CM / H_PX, D_CM])
    t /= np.linalg.norm(t)
    u = np.cross(t, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(t, u)
    phi = np.linspace(0, 2 * np.pi, n)
    r = np.radians(r_deg)
    d = np.cos(r) * t[None] + np.sin(r) * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
    p = d * (D_CM / d[:, 2:3])
    return p[:, 0] * W_PX / W_CM + W_PX / 2, p[:, 1] * H_PX / H_CM + H_PX / 2


def write_rings(path):
    rows = []
    for kind, centers, radii in [("threshold", TARGETS, [i2mc.MAX_CALIBRATION_DEG]), ("reference", [(960, 540)], REF_RINGS_DEG)]:
        for tx, ty in centers:
            for r in radii:
                x, y = ring(tx, ty, r)
                rows.append(pd.DataFrame(dict(kind=kind, target_x=tx, target_y=ty, radius_deg=r, x=x, y=y, order=range(len(x)))))
    pd.concat(rows).round(2).to_csv(path, index=False)


def block_qc(pid, group, pdir, cfg):
    files = i2mc.find_participant_files(pdir)
    vs = pd.read_csv(files["validation_summary"])
    if "outcome" not in vs.columns or vs["outcome"].isna().any():
        vs, _ = bf.backfill_file(files["validation_summary"], cfg, baby=group == "infants")  # in memory only
    tro = pd.read_csv(files["trial_order"])
    gaze = i2mc.read_csv_multi_encoding(files["gaze"])
    cal = i2mc.compute_block_calibration(tro, vs)
    steps = used_steps(tro, vs)
    means = vs[vs["point"] == "mean"].drop_duplicates("validation_step", keep="last").set_index("validation_step")

    rows, trials = [], []
    for b, d in tro.groupby("block_index"):
        deg, eyes = cal[b]
        step, n_att = steps[b]
        included = deg is not None and not pd.isna(deg) and round(deg, 2) <= i2mc.MAX_CALIBRATION_DEG
        n_samp = n_valid = n_trials_ok = 0
        for ti, v in zip(d["total_trial_index"], d["video_name"]):
            t = i2mc.extract_trial_gaze(gaze, v)
            _, invalid = i2mc.trial_to_i2mc_frame(t, eyes) if t is not None else (None, 1.0)
            n = 0 if t is None else int((t["trial_time"].notna() & (t["trial_time"] >= 0)).sum())
            n_samp += n
            n_valid += round(n * (1 - invalid))
            n_trials_ok += invalid <= i2mc.MAX_INVALID_FRAC
            trials.append(dict(pid=pid, block_index=b, trial_index=ti, video=v, n_samples=n,
                               pct_valid_samples=round(100 * (1 - invalid), 1), included=included))
        pct_valid = 100 * n_valid / n_samp if n_samp else 0.0
        m = means.loc[step] if step in means.index else {}
        rows.append(dict(
            pid=pid, group=group, block_index=b, block_id=d["block_id"].iloc[0], n_trials=len(d),
            validation_step=step, n_attempts=n_att,
            deg_left=m.get("Mean_accuracy_degrees_left"), deg_right=m.get("Mean_accuracy_degrees_right"),
            eyes_analyzed=eyes, deg_analyzed=deg, included=included,
            pct_valid_samples=round(pct_valid, 1), pct_usable=round(pct_valid if included else 0.0, 1),
            n_trials_valid_enough=n_trials_ok))
    return pd.DataFrame(rows), pd.DataFrame(trials), vs, steps, files


def validation_samples(pid, files, qc):
    raw = pd.read_csv(os.path.join(os.path.dirname(files["gaze"]), os.path.basename(files["gaze"])[:-4] + "_validation.csv"))
    out = []
    for _, r in qc.iterrows():
        s = raw[raw["validation_step"] == r["validation_step"]].copy()
        eyes = ["left", "right"] if r["eyes_analyzed"] == "both" else [r["eyes_analyzed"]]
        valid = {e: s[f"{e}_valid"].astype(str).str.lower().isin(["true", "1"]) for e in eyes}
        s["x"] = pd.concat([s[f"{e}_x"].where(valid[e]) for e in eyes], axis=1).mean(axis=1)
        s["y"] = pd.concat([s[f"{e}_y"].where(valid[e]) for e in eyes], axis=1).mean(axis=1)
        s = s.dropna(subset=["x", "y"])
        out.append(pd.DataFrame(dict(pid=pid, block_index=r["block_index"], validation_step=r["validation_step"],
                                     target_x=s["stimulus_x"], target_y=s["stimulus_y"], x=s["x"].round(1), y=s["y"].round(1))))
    return pd.concat(out)


def point_accuracy(vs, qc):
    """Per-target accuracy (deg) of the analyzed eye(s) and presentation order for each block's validation."""
    pts = vs[vs["point"].astype(str).str.startswith("Point")]
    out = []
    for _, r in qc.iterrows():
        p = pts[pts["validation_step"] == r["validation_step"]]
        cols = [f"Mean_accuracy_degrees_{e}" for e in (["left", "right"] if r["eyes_analyzed"] == "both" else [r["eyes_analyzed"]])]
        order = p["presentation_order"] if "presentation_order" in p else np.nan  # missing in pre-2026-03 summaries
        out.append(pd.DataFrame(dict(block_index=r["block_index"], target_x=p["stimulus_x"], target_y=p["stimulus_y"],
                                     point_deg=p[cols].mean(axis=1).round(2), presentation_order=order)))
    return pd.concat(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pids", nargs="+")
    args = p.parse_args()
    with open(bf.CONFIG) as f:
        cfg = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    write_rings(os.path.join(OUT_DIR, "validation_rings.csv"))
    for pid in args.pids:
        try:
            group, pdir = copy_participant(pid)
        except (FileNotFoundError, ValueError) as e:
            print(f"SKIP {e}")
            continue
        qc, trials, vs, _, files = block_qc(pid, group, pdir, cfg)
        trials.to_csv(os.path.join(OUT_DIR, f"{pid}_trial_qc.csv"), index=False)
        qc = qc.merge(point_accuracy(vs, qc).groupby("block_index")["point_deg"].max().rename("worst_point_deg"),
                      on="block_index", how="left")
        qc.to_csv(os.path.join(OUT_DIR, f"{pid}_block_qc.csv"), index=False)
        samples = validation_samples(pid, files, qc).merge(point_accuracy(vs, qc), on=["block_index", "target_x", "target_y"], how="left")
        samples.to_csv(os.path.join(OUT_DIR, f"{pid}_validation_samples.csv"), index=False)
        print(qc[["pid", "block_index", "block_id", "validation_step", "n_attempts", "eyes_analyzed", "deg_analyzed",
                  "worst_point_deg", "included", "pct_valid_samples", "pct_usable"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
