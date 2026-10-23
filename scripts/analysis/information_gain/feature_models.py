"""
Single-feature models, leave-one-clip-out: q_t ∝ f(x_t) * floor_t^beta, normalized. f = monotone
piecewise-linear with N_KNOTS equidistant knots on [0, 1] (Kümmerer et al. 2015) or exp(a*x).
Fit by maximizing group-average log-likelihood on training clips; training-segment floors exclude
both the held-out clip and the segment's own clip. Also predicts the shuffled-feature control
(each held-out segment given the feature vector of a random segment from a different clip).

Output: predictions_<segment_set>_<variant>.parquet (video, seg_idx, feature, shuffled, tile_000..)
        curves_<segment_set>_<variant>.parquet (held-out clip, feature, fitted f at the knots)

Usage: python scripts/analysis/information_gain/feature_models.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config as C
from common import TILE_COLS, UNIFORM, clip_sums, load_gaze, pooled_excluding

LN2 = np.log(2)
KNOTS = np.linspace(0, 1, C.N_KNOTS)
VARIANTS = {  # name: (nonlinearity, beta)
    "knots_floor": ("knots", C.FLOOR_BETA),
    "exp_floor": ("exp", C.FLOOR_BETA),
    "knots_nofloor": ("knots", 0.0),
}


def basis(x: np.ndarray, n_knots: int = C.N_KNOTS) -> np.ndarray:
    """(..., N_KNOTS) linear-interpolation weights so that f(x) = basis(x) @ knot_values."""
    pos = np.clip(x, 0, 1) * (n_knots - 1)
    k = np.minimum(np.floor(pos).astype(int), n_knots - 2)
    a = pos - k
    B = np.zeros(x.shape + (n_knots,))
    np.put_along_axis(B, k[..., None], (1 - a)[..., None], axis=-1)
    np.put_along_axis(B, (k + 1)[..., None], a[..., None], axis=-1)
    return B


def knot_values(theta: np.ndarray) -> np.ndarray:
    # f(0) fixed at 1 (scale cancels in normalization); non-negative increments keep f monotone.
    return np.concatenate([[1.0], 1.0 + np.cumsum(np.exp(theta))])


def f_of(x, params, nonlin):
    return basis(x, len(params) + 1) @ knot_values(params) if nonlin == "knots" else np.exp(params[0] * x)


def neg_ll_and_grad(params, X, Bx, floors, G, nonlin):
    """Objective: -sum_s sum_t G_st log2 q_st, with G already carrying the group-average cell weights."""
    if nonlin == "knots":
        v = knot_values(params)
        f = Bx @ v
    else:
        f = np.exp(params[0] * X)
    fb = f * floors
    Z = fb.sum(1, keepdims=True)
    W = G.sum(1, keepdims=True)
    ll = (G * (np.log(fb) - np.log(Z))).sum() / LN2
    # d ll / d f_st = G_st / f_st - W_s * floor_st / Z_s
    dl_df = G / f - W * floors / Z
    if nonlin == "knots":
        g_v = np.einsum("st,stk->k", dl_df, Bx)
        e = np.exp(params)
        grad = e * np.cumsum(g_v[1:][::-1])[::-1]
    else:
        grad = np.array([(dl_df * f * X).sum()])
    return -ll, -grad / LN2


def fit(X, floors, G, nonlin, n_knots=C.N_KNOTS):
    Bx = basis(X, n_knots) if nonlin == "knots" else None
    if nonlin == "knots":
        x0, bounds = np.full(n_knots - 1, np.log(1 / (n_knots - 1))), [(-20, 10)] * (n_knots - 1)
    else:
        x0, bounds = np.array([1.0]), [(0.0, 100.0)]
    res = minimize(neg_ll_and_grad, x0, args=(X, Bx, floors, G, nonlin), jac=True,
                   method="L-BFGS-B", bounds=bounds)
    return res.x


def predict(x, params, floor, nonlin, beta):
    q = f_of(x, params, nonlin) * floor ** beta
    q = q / q.sum(-1, keepdims=True)
    q = np.maximum(q, 1e-12)
    return q / q.sum(-1, keepdims=True)


def weighted_segment_gaze(cells: pd.DataFrame, gaze_mat: np.ndarray) -> dict:
    """Group-average weights: each participant's cells share weight 1/n_p, participants equal."""
    w = 1.0 / cells.groupby("participant").participant.transform("size").to_numpy() / cells.participant.nunique()
    out = {}
    for key, idx in cells.groupby(["video", "seg_idx"]).indices.items():
        out[key] = (gaze_mat[idx] * w[idx, None]).sum(0)
    return out


def run(segment_set: str, variant: str, rng: np.random.Generator, n_knots: int = C.N_KNOTS, suffix: str = ""):
    nonlin, beta = VARIANTS[variant]
    gaze = load_gaze(segment_set)
    cells = pd.read_parquet(C.CACHE_DIR / f"ceiling_{segment_set}.parquet")[["video", "seg_idx", "participant"]]
    gaze = gaze.merge(cells, on=["video", "seg_idx", "participant"])  # exactly the ceiling's cells
    sums, counts = clip_sums(gaze)
    floors = pd.read_parquet(C.CACHE_DIR / f"floor_{segment_set}.parquet").set_index("video")
    feats = pd.read_csv(C.CACHE_DIR / f"features_{segment_set}.csv")

    preds, curves = [], []
    for feature in C.FEATURES:
        fx = {k: g.sort_values("tile")[feature].to_numpy() for k, g in feats.groupby(["video", "seg_idx"])}
        fx = {k: v for k, v in fx.items() if not np.isnan(v).any()}
        all_keys = list(fx)
        for clip in C.CLIPS:
            eps = floors.loc[clip, "eps"]
            train = gaze[(gaze.video != clip) & gaze.set_index(["video", "seg_idx"]).index.isin(all_keys)]
            Gs = weighted_segment_gaze(train.reset_index(drop=True), train[TILE_COLS].to_numpy())
            keys = sorted(Gs)
            X = np.array([fx[k] for k in keys])
            Fl = np.array([((1 - eps) * pooled_excluding(sums, counts, {clip, k[0]}) + eps * UNIFORM) ** beta
                           for k in keys])
            params = fit(X, Fl, np.array([Gs[k] for k in keys]), nonlin, n_knots)
            curve = f_of(np.linspace(0, 1, n_knots) if nonlin == "knots" else KNOTS, params, nonlin)
            curves.append(dict(held_out=clip, feature=feature, **{f"k{i:02d}": c for i, c in enumerate(curve / curve[0])}))

            floor_c = floors.loc[clip, TILE_COLS].to_numpy(dtype=float)
            held = [k for k in all_keys if k[0] == clip]
            other_clip = [k for k in all_keys if k[0] != clip]
            for k in held:
                shuf = other_clip[rng.integers(len(other_clip))]
                for shuffled, xk in [(False, fx[k]), (True, fx[shuf])]:
                    q = predict(xk, params, floor_c, nonlin, beta)
                    preds.append(dict(video=k[0], seg_idx=k[1], feature=feature, shuffled=shuffled,
                                      **dict(zip(TILE_COLS, q))))
        print(f"  {variant:14s} {feature:9s} done")
    tag = f"{segment_set}_{variant}{suffix}"
    pd.DataFrame(preds).to_parquet(C.CACHE_DIR / f"predictions_{tag}.parquet", index=False)
    pd.DataFrame(curves).to_parquet(C.CACHE_DIR / f"curves_{tag}.parquet", index=False)


def main():
    for segment_set in C.SEGMENT_SETS:
        for variant in VARIANTS:
            run(segment_set, variant, np.random.default_rng(C.RNG_SEED))


if __name__ == "__main__":
    main()
