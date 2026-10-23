"""
Exploratory: the Overall IG pipeline (unchopped, 20 knots, floor) rebuilt on a dense grid of
~10,000 locations instead of the 120 tiles. Same cells, same nested leave-one-clip-out, same
smoothing (gaze sigma=46 from fixations; features plotted at patch centers, sigma 46/92).

Output: figures/checks/overall_ig_dense_<n>.png, data/results/information_gain/summary_dense_<n>.csv

Usage: python scripts/analysis/information_gain/explore_dense_grid.py [NX NY]  (default 133 75)
"""

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

import config as C
from common import group_average
from load_features import frame_feature, load_meaning_raw, meaning_frames, segment_mean
from partition import partition
from plots import headline
from scoring import MODELS
from segments import segment_table

sys.path.insert(0, str(C.PROJECT_ROOT / "scripts" / "analysis"))
from calc_concordance_group import build_participant_arrays, closed_form_tile_values, load_video_fixations  # noqa: E402
from tile_pooling import CENTERS  # noqa: E402

NX, NY = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (133, 75)
SEGMENT_SET = "cuts"
LN2 = np.log(2)
QX, QY = np.meshgrid((np.arange(NX) + 0.5) * 1920 / NX, (np.arange(NY) + 0.5) * 1080 / NY)
Q = np.column_stack([QX.ravel(), QY.ravel()])
N = len(Q)
UNIF = np.full(N, 1 / N)  # equal-area cells


def smoothing_to_grid(points_xy, sigma):
    d2 = ((Q[:, None, :] - points_xy[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2 * sigma ** 2))
    return w / w.sum(1, keepdims=True)


def ll(g, q):
    return (g * np.log2(q)).sum(-1)


def cell_weights(cells):
    return 1.0 / cells.groupby("participant").participant.transform("size").to_numpy() / cells.participant.nunique()


def build_gaze(cells):
    G = np.zeros((len(cells), N), dtype=np.float32)
    segs = segment_table(SEGMENT_SET).set_index(["video", "seg_idx"])
    for clip, idx in cells.groupby("video").indices.items():
        fix = load_video_fixations(C.FIXATION_DIR, clip, C.FIXATION_PARAM_CODE, C.PARTICIPANT_SUMMARY_PATH)
        arrays = build_participant_arrays(fix[(fix.endT - fix.startT) <= C.MAX_FIXATION_MS])
        for i in idx:
            r = cells.iloc[i]
            seg = segs.loc[(clip, r.seg_idx)]
            v = closed_form_tile_values(*arrays[r.participant], seg.seg_start_s * 1000, seg.seg_end_s * 1000,
                                        C.SIGMA_PX, Q[:, 0], Q[:, 1])
            G[i] = v / v.sum()
    return G


def build_features(keys):
    segs = segment_table(SEGMENT_SET).set_index(["video", "seg_idx"])
    raw = {c: r for c in C.CLIPS if (r := load_meaning_raw(c)) is not None}
    meaning = meaning_frames(raw)
    w_fine = smoothing_to_grid(CENTERS[["center_x", "center_y"]].to_numpy(), C.SIGMA_FEATURE_FINE)
    out = {f: {} for f in ["saliency", "motion", "meaning_fine", "meaning_coarse"]}
    for clip in C.CLIPS:
        sal, mot, m = frame_feature(clip, "saliency"), frame_feature(clip, "motion"), meaning[clip]
        w_coarse = smoothing_to_grid(m[3], C.SIGMA_FEATURE_COARSE)
        for k in [k for k in keys if k[0] == clip]:
            s, e = segs.loc[k, "seg_start_s"], segs.loc[k, "seg_end_s"]
            out["saliency"][k] = w_fine @ segment_mean(sal, s, e)
            out["motion"][k] = w_fine @ segment_mean(mot, s, e, needs_prev=True)
            out["meaning_fine"][k] = w_fine @ segment_mean((m[0], m[1]), s, e)
            out["meaning_coarse"][k] = w_coarse @ segment_mean((m[0], m[2]), s, e)

    def mm(d):  # joint min-max across all segments and locations
        lo, hi = min(v.min() for v in d.values()), max(v.max() for v in d.values())
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}
    fine, coarse = mm(out["meaning_fine"]), mm(out["meaning_coarse"])
    return {"saliency": mm(out["saliency"]), "motion": mm(out["motion"]),
            "meaning": mm({k: (fine[k] + coarse[k]) / 2 for k in keys})}


def fit_knots(X, F, A):
    """20-knot monotone curve; sparse basis (2 nonzeros per location)."""
    pos = X * (C.N_KNOTS - 1)
    k = np.minimum(np.floor(pos).astype(int), C.N_KNOTS - 2)
    a = pos - k

    def obj(th):
        v = np.concatenate([[1.0], 1.0 + np.cumsum(np.exp(th))])
        f = v[k] * (1 - a) + v[k + 1] * a
        fb = f * F
        Z = fb.sum(1, keepdims=True)
        W = A.sum(1, keepdims=True)
        val = (A * (np.log(fb) - np.log(Z))).sum() / LN2
        d = A / f - W * F / Z
        g_v = np.bincount(k.ravel(), (d * (1 - a)).ravel(), C.N_KNOTS) + np.bincount((k + 1).ravel(), (d * a).ravel(), C.N_KNOTS)
        grad = np.exp(th) * np.cumsum(g_v[1:][::-1])[::-1]
        return -val, -grad / LN2
    x0 = np.full(C.N_KNOTS - 1, np.log(1 / (C.N_KNOTS - 1)))
    th = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=[(-20, 10)] * (C.N_KNOTS - 1)).x
    return np.concatenate([[1.0], 1.0 + np.cumsum(np.exp(th))])


def main():
    cells = pd.read_parquet(C.CACHE_DIR / f"ceiling_{SEGMENT_SET}.parquet")[["video", "seg_idx", "participant"]].reset_index(drop=True)
    print(f"grid {NX}x{NY} = {N} locations; {len(cells)} cells")
    G = build_gaze(cells)
    seg_idx = cells.groupby(["video", "seg_idx"]).indices
    clip_sum = {c: G[idx].astype(np.float64).sum(0) for c, idx in cells.groupby("video").indices.items()}
    clip_n = cells.video.value_counts().to_dict()

    def pooled(excl):
        keep = [c for c in clip_sum if c not in excl]
        v = sum(clip_sum[c] for c in keep) / sum(clip_n[c] for c in keep)
        return v / v.sum()

    feats = build_features(list(seg_idx))
    rows = []
    for clip in C.CLIPS:
        train = cells.video != clip
        tcells = cells[train]
        w = np.zeros(len(cells)); w[train.to_numpy()] = cell_weights(tcells)
        # eps: inner leave-one-clip-out on training clips (aggregated weighted gaze per clip).
        A_clip = {d: (G[idx] * w[idx, None]).sum(0) for d, idx in tcells.groupby("video").indices.items()}
        P_in = {d: pooled({clip, d}) for d in A_clip}
        eps = minimize_scalar(lambda e: -sum(ll(A_clip[d], (1 - e) * P_in[d] + e * UNIF) for d in A_clip),
                              bounds=(1e-6, 0.5), method="bounded").x
        floor_c = (1 - eps) * pooled({clip}) + eps * UNIF
        inner_floor = {d: (1 - eps) * P_in[d] + eps * UNIF for d in A_clip}

        # lam: inner leave-one-clip-out with leave-one-participant-out others.
        blocks = []
        for (v, s), idx in seg_idx.items():
            if v == clip:
                continue
            Gs = G[idx].astype(np.float64)
            blocks.append((Gs, (Gs.sum(0) - Gs) / (len(Gs) - 1), inner_floor[v], w[idx]))
        lam = minimize_scalar(lambda L: -sum((wi * ll(Gs, (1 - L) * O + L * f)).sum() for Gs, O, f, wi in blocks),
                              bounds=(1e-4, 0.999), method="bounded").x

        # Feature curves fit on training segments (floors exclude the held-out clip and the segment's clip).
        tkeys = [k for k in seg_idx if k[0] != clip]
        A = np.array([(G[seg_idx[k]] * w[seg_idx[k], None]).sum(0) for k in tkeys])
        F = np.array([inner_floor[k[0]] for k in tkeys])
        curves = {f: fit_knots(np.array([feats[f][k] for k in tkeys]), F, A) for f in C.FEATURES}

        for k in [k for k in seg_idx if k[0] == clip]:
            idx = seg_idx[k]
            Gs = G[idx].astype(np.float64)
            llf = ll(Gs, floor_c)
            others = (Gs.sum(0) - Gs) / (len(Gs) - 1)
            q = {}
            for f, ab in zip(C.FEATURES, "SMX"):
                x = feats[f][k] * (C.N_KNOTS - 1)
                kk = np.minimum(np.floor(x).astype(int), C.N_KNOTS - 2)
                fx = curves[f][kk] * (1 - (x - kk)) + curves[f][kk + 1] * (x - kk)
                qq = np.maximum(fx * floor_c / (fx * floor_c).sum(), 1e-12)
                q[ab] = qq / qq.sum()
            row = cells.iloc[idx][["video", "seg_idx", "participant"]].copy()
            row["ig_ceiling"] = ll(Gs, (1 - lam) * others + lam * floor_c) - llf
            for name, parts in MODELS.items():
                row[f"ig_{name}"] = ll(Gs, np.mean([q[p] for p in parts], axis=0)) - llf
            rows.append(row)
        print(f"  {clip:15s} eps={eps:.3f} lam={lam:.3f}")

    df = pd.concat(rows, ignore_index=True)
    ig = {m: group_average(df, f"ig_{m}") for m in MODELS}
    s = pd.Series(dict(ceiling=group_average(df, "ig_ceiling"), **{f"ig_{m}": v for m, v in ig.items()},
                       **partition(ig), n_participants=df.participant.nunique(),
                       n_segments=df.groupby(["video", "seg_idx"]).ngroups))
    s.to_frame("value").to_csv(C.RESULTS_DIR / f"summary_dense_{N}.csv")
    print(s.round(4).to_string())
    caption = (f"Same pipeline as the 120-tile Overall IG, on a {NX}x{NY} grid ({N} locations, ~14.4px cells). "
               "Combined models are fixed equal-weight mixtures; each feature model includes the floor (beta=1). "
               f"{int(s.n_participants)} adults, {int(s.n_segments)} segments, leave-one-clip-out. No confidence intervals.")
    fig = headline(s, f"unchopped, 20 knots, floor, {N} locations", caption)
    out = C.CHECK_FIG_DIR / f"overall_ig_dense_{N}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
