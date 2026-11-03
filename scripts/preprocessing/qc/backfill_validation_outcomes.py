"""
Backfill validation_passed / eyes_used / outcome into *_validation_summary.csv
files written before the experiment logged them (2026-10-30).

Re-applies the pass rule the experiment used at the time to the per-point,
per-eye accuracies already in the file. The old experiment stopped at the
first pass, so each calibration routine's last attempt is the one in use.

Rules by session date (from the {Sub}_{YYYYmmdd_HHMMSS}_ filename):
  before 2025-12-12  fail only if either eye's mean error > 25 deg (no 1-eye pass)
  2025-12-12 on      average of both eyes, else either eye alone; thresholds from
                     experiment/config.yaml (unchanged since 2025-12-12)
  2026-06-02 on      baby groups (--baby_groups) use the baby point counts

Default is a dry run: prints a summary and writes a per-attempt log. --write
rewrites each file in place, after copying the original to --backup_dir.

Usage
-----
    python preprocessing/qc/backfill_validation_outcomes.py                # dry run
    python preprocessing/qc/backfill_validation_outcomes.py --write
"""

import argparse
import glob
import os
import re
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(__file__)
RAW_DIR = os.path.join(HERE, "..", "..", "..", "data", "raw")
LOG_DIR = os.path.join(HERE, "..", "..", "..", "data", "preprocessed", "log-files")
BACKUP_DIR = os.path.join(HERE, "..", "..", "..", "data", "local_cache", "validation_summary_backup")
CONFIG = os.path.join(HERE, "..", "..", "..", "experiment", "config.yaml")

POINT_RULE_START = "20251212"   # commit 756a496: per-point thresholds + single-eye pass
BABY_RULE_START = "20260602"    # commit 5a5398b: --baby point counts
OLD_RULE_MAX_DEG = 25.0         # pre-2025-12-12 rule: fail if either eye's mean > 25 deg
NEW_COLS = ["validation_passed", "eyes_used", "outcome"]


def evaluate(left, right, good, bad, min_good, max_bad):
    """Same rule as experiment/psychopy_tobii_infant.evaluate_validation; returns both/left/right/none."""
    avg = [(l + r) / 2 for l, r in zip(left, right)]
    passes = {}
    for name, vals in (("both", avg), ("left", left), ("right", right)):
        n_good = sum(v < good for v in vals)
        n_bad = sum(v > bad for v in vals)
        passes[name] = n_good >= min_good and n_bad <= max_bad
    if passes["both"]:
        return "both"
    if passes["left"] and passes["right"]:
        # Old code accepted without recording which eye; use the more accurate one
        return "left" if np.nanmean(left) <= np.nanmean(right) else "right"
    return "left" if passes["left"] else "right" if passes["right"] else "none"


def session_date(path):
    m = re.search(r"_(\d{8})_\d{6}_validation_summary\.csv$", os.path.basename(path))
    return m.group(1) if m else None


def attempts(df):
    """Contiguous runs of rows sharing a validation_step = one validation attempt each."""
    run_id = (df["validation_step"] != df["validation_step"].shift()).cumsum()
    return [g for _, g in df.groupby(run_id, sort=False)]


def backfill_file(path, cfg, baby):
    date = session_date(path)
    df = pd.read_csv(path)
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[NEW_COLS] = df[NEW_COLS].astype(object)

    runs = attempts(df)
    log = []
    for k, run in enumerate(runs):
        step = str(run["validation_step"].iloc[0])
        if run["outcome"].notna().any():
            continue  # logged live by the new experiment code
        routine, _, n = step.rpartition("_")
        nxt = runs[k + 1]["validation_step"].iloc[0] if k + 1 < len(runs) else None
        nxt_routine, _, nxt_n = str(nxt).rpartition("_")
        is_last = not (nxt is not None and nxt_routine == routine and nxt_n.isdigit() and n.isdigit() and int(nxt_n) > int(n))

        mean = run[run["point"] == "mean"]
        pts = run[run["point"].astype(str).str.startswith("Point")]
        if "presentation_order" in pts.columns:  # absent in pre-2026-03 files, already in presentation order
            pts = pts.sort_values("presentation_order")
        if date is None:
            rule, passed = "unknown_date", "none"
        elif date < POINT_RULE_START:
            rule = "pre_20251212_mean25"
            l, r = (mean["Mean_accuracy_degrees_left"].iloc[0], mean["Mean_accuracy_degrees_right"].iloc[0]) if len(mean) else (np.nan, np.nan)
            passed = "none" if (np.isnan(l) or np.isnan(r) or l > OLD_RULE_MAX_DEG or r > OLD_RULE_MAX_DEG) else "both"
        else:
            mode = "initial" if routine.startswith("pre_validation") else "later"
            use_baby = baby and date >= BABY_RULE_START
            g = cfg["general_validation"]
            min_good, max_bad = (g["min_good_points_baby"], g["max_bad_points_baby"]) if use_baby else (g["min_good_points"], g["max_bad_points"])
            good, bad = cfg[f"{mode}_validation"]["good_threshold"], cfg[f"{mode}_validation"]["bad_threshold"]
            rule = f"{mode}{'_baby' if use_baby else ''}"
            # Missing points count as bad, as in the experiment
            left = list(pts["Mean_accuracy_degrees_left"]) + [bad + .01] * (5 - len(pts))
            right = list(pts["Mean_accuracy_degrees_right"]) + [bad + .01] * (5 - len(pts))
            passed = evaluate(left, right, good, bad, min_good, max_bad)

        if not is_last:
            eyes, outcome = np.nan, "retried"
        elif passed == "none":
            eyes, outcome = "both", "proceeded_failed"
        else:
            eyes, outcome = passed, "accepted"
        df.loc[run.index, NEW_COLS] = [passed, eyes, outcome]
        log.append(dict(file=os.path.basename(path), session_date=date, validation_step=step, rule=rule,
                        validation_passed=passed, eyes_used=eyes, outcome=outcome))
    return df, log


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", default=RAW_DIR)
    p.add_argument("--log_dir", default=LOG_DIR)
    p.add_argument("--backup_dir", default=BACKUP_DIR)
    p.add_argument("--groups", default="adults,infants,kids")
    p.add_argument("--baby_groups", default="infants", help="Groups run with --baby (from 2026-06-02)")
    p.add_argument("--write", action="store_true", help="Rewrite files in place (default: dry run)")
    args = p.parse_args()

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    baby_groups = set(args.baby_groups.split(",")) if args.baby_groups else set()

    all_log = []
    for group in args.groups.split(","):
        files = sorted(glob.glob(os.path.join(args.raw_dir, group, "*", "*_validation_summary.csv")))
        for path in files:
            if os.path.basename(path).startswith("."):
                continue
            df, log = backfill_file(path, cfg, group in baby_groups)
            for row in log:
                row["group"] = group
            all_log += log
            if session_date(path) == POINT_RULE_START:
                print(f"  NOTE {group}/{os.path.basename(path)}: recorded on the day the rule changed; check which code ran")
            if args.write and log:
                dest = os.path.join(args.backup_dir, group)
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(path, os.path.join(dest, os.path.basename(path)))
                df.to_csv(path, index=False)
        print(f"[{group}] {len(files)} file(s)")

    if not all_log:
        print("Nothing to backfill.")
        return
    log_df = pd.DataFrame(all_log)
    in_use = log_df[log_df["outcome"] != "retried"]
    print("\nCalibrations in use (last attempt of each routine):")
    print(in_use.groupby(["group", "rule", "eyes_used"]).size().to_string())
    os.makedirs(args.log_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.log_dir, f"validation_outcome_backfill_{'write' if args.write else 'dryrun'}_{stamp}.csv")
    log_df.to_csv(out, index=False)
    print(f"\n{'Rewrote' if args.write else 'Dry run, nothing changed'}; per-attempt log -> {out}")


if __name__ == "__main__":
    main()
