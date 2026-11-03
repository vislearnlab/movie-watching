"""
Which participants/blocks ran on a 1-eye calibration, and how that changes the
I2MC calibration check (accepted eye's accuracy vs. the old both-eyes average).

Reads each participant's *_validation_summary.csv (backfilling eyes_used in
memory if the file predates it; see backfill_validation_outcomes.py) and
*_trial_order.csv, maps blocks to validations exactly as i2mc_fixations.py
does, and writes:
  {out_csv}  one row per participant x block
  {out_md}   readable summary

Usage
-----
    python preprocessing/qc/summarize_eye_calibration_impact.py --groups adults,kids
"""

import argparse
import os
import sys
from datetime import date

import pandas as pd
import yaml

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "fixations"))
import backfill_validation_outcomes as bf
import i2mc_fixations as i2mc

ROOT = os.path.join(HERE, "..", "..", "..")
OUT_CSV = os.path.join(ROOT, "data", "results", "calibration_eyes", "eye_calibration_impact_by_block.csv")
OUT_MD = os.path.join(ROOT, "documentation", "one_eye_calibration_impact.md")


def participant_rows(group, pdir, cfg, max_deg):
    files = i2mc.find_participant_files(pdir)
    pid = os.path.basename(pdir)
    if files["validation_summary"] is None or files["trial_order"] is None:
        return [], [dict(group=group, pid=pid, issue="missing validation_summary or trial_order")]
    vs = pd.read_csv(files["validation_summary"])
    if "outcome" not in vs.columns or vs["outcome"].isna().any():
        vs, _ = bf.backfill_file(files["validation_summary"], cfg, baby=False)
    tro = pd.read_csv(files["trial_order"])

    new = i2mc.compute_block_calibration(tro, vs, max_deg)
    old = i2mc.compute_block_calibration(tro, vs.drop(columns=["eyes_used"]))
    accepted = i2mc.compute_block_calibration(tro, vs, max_deg=-1)  # forces the accepted eye's own accuracy
    block_ids = tro.groupby("block_index")["block_id"].first() if "block_id" in tro.columns else {}
    rows = []
    for b, (deg_new, eyes) in new.items():
        deg_old = old[b][0]
        deg_acc, eyes_acc = accepted[b]
        # Same rule as i2mc_fixations.py: no matching validation is kept, NaN accuracy is excluded
        excl_old = deg_old is not None and (pd.isna(deg_old) or round(deg_old, 2) > max_deg)
        excl_new = deg_new is not None and (pd.isna(deg_new) or round(deg_new, 2) > max_deg)
        rows.append(dict(group=group, pid=pid, block_index=b, block_id=block_ids.get(b, ""),
                         eyes_accepted=eyes_acc, eyes_analyzed=eyes, deg_both_eyes=deg_old,
                         deg_accepted_eye=deg_acc, deg_analyzed_eyes=deg_new,
                         excluded_before=excl_old, excluded_after=excl_new))

    # Mid-block forced recalibrations aren't used by I2MC; report any that ended on 1 eye
    issues = []
    forced = vs[(vs["point"] == "mean") & vs["validation_step"].str.contains("forced")
                & vs["eyes_used"].isin(["left", "right"])]
    for _, r in forced.iterrows():
        issues.append(dict(group=group, pid=pid, issue=f"{r['validation_step']}: mid-block recalibration "
                           f"accepted on {r['eyes_used']} eye (not used by I2MC)"))
    return rows, issues


def write_md(df, issues, path, groups, max_deg):
    one = df[df["eyes_accepted"] != "both"]
    lines = [f"# 1-eye calibrations: impacted participants and blocks", "",
             f"Generated {date.today()} by `scripts/preprocessing/qc/summarize_eye_calibration_impact.py` "
             f"(groups: {', '.join(groups)}). Infants not yet checked.", "",
             "Before 2026-10-30 the experiment accepted a calibration that passed on one eye only, without "
             "logging which. Which eye passed was reconstructed from the per-eye validation accuracies "
             "(`METHODS_DECISIONS.md`, 2026-10-30). I2MC uses both eyes whenever their averaged error is "
             f"≤ {max_deg}°, and falls back to the experiment-accepted eye only when it isn't "
             "(`METHODS_DECISIONS.md`, 2026-11-03).", "", "## Overview", "",
             "| Group | Participants | Blocks | Accepted on 1 eye | Analyzed on 1 eye | Participants analyzed on 1 eye |",
             "|---|---|---|---|---|---|"]
    for g, d in df.groupby("group"):
        o = d[d["eyes_accepted"] != "both"]
        a = d[(d["eyes_analyzed"] != "both") & ~d["excluded_after"]]
        lines.append(f"| {g} | {d['pid'].nunique()} | {len(d)} | {len(o)} (left {sum(o.eyes_accepted == 'left')}, "
                     f"right {sum(o.eyes_accepted == 'right')}) | {len(a)} | {a['pid'].nunique()} |")
    changed = df[df["excluded_before"] != df["excluded_after"]]
    lines += ["", f"## Effect on the I2MC calibration check (≤ {max_deg}°)", "",
              f"- Blocks newly **included** (both-eyes average > {max_deg}°, accepted eye ≤ {max_deg}°): "
              f"{sum(changed.excluded_before)} -- these are the only blocks whose fixations change.",
              f"- Blocks newly **excluded**: {sum(changed.excluded_after)}", "", "## Blocks accepted on 1 eye", ""]
    if one.empty:
        lines.append("None.")
    else:
        lines += ["| Group | Participant | Block | Block ID | Accepted eye | Both-eyes error (°) | Accepted-eye error (°) | Analyzed | I2MC check |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in one.sort_values(["group", "pid", "block_index"]).iterrows():
            status = ("excluded" if r.excluded_after else "included") + (
                " (was excluded)" if r.excluded_before and not r.excluded_after else
                " (was included)" if r.excluded_after and not r.excluded_before else "")
            analyzed = "—" if r.excluded_after else r.eyes_analyzed
            lines.append(f"| {r.group} | {r.pid} | {r.block_index} | {r.block_id} | {r.eyes_accepted} | "
                         f"{r.deg_both_eyes:.2f} | {r.deg_accepted_eye:.2f} | {analyzed} | {status} |")
    if issues:
        lines += ["", "## Other notes", ""] + [f"- {i['group']}/{i['pid']}: {i['issue']}" for i in issues]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_dir", default=bf.RAW_DIR)
    p.add_argument("--groups", default="adults,kids")
    p.add_argument("--out_csv", default=OUT_CSV)
    p.add_argument("--out_md", default=OUT_MD)
    p.add_argument("--max_calibration_deg", type=float, default=i2mc.MAX_CALIBRATION_DEG)
    args = p.parse_args()

    with open(bf.CONFIG) as f:
        cfg = yaml.safe_load(f)
    groups = args.groups.split(",")
    rows, issues = [], []
    for g in groups:
        gdir = os.path.join(args.raw_dir, g)
        for pdir in sorted(d for d in (os.path.join(gdir, x) for x in os.listdir(gdir)) if os.path.isdir(d)):
            r, i = participant_rows(g, pdir, cfg, args.max_calibration_deg)
            rows += r; issues += i
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    write_md(df, issues, args.out_md, groups, args.max_calibration_deg)
    print(f"{len(df)} blocks, {sum(df.eyes_accepted != 'both')} accepted on 1 eye, {sum((df.eyes_analyzed != 'both') & ~df.excluded_after)} analyzed on 1 eye -> {args.out_md}, {args.out_csv}")


if __name__ == "__main__":
    main()
