"""v16: benchmark hierarchy IID < vol2 < m3 < U (and R), on saved forecasts.

Builds a daily panel per rule and asset with the probability vectors of five
models, then reports the telescoping score and return layers, a magnitude-
weighted direction score, calendar segments and rolling-frequency baselines.

Inputs (not distributed): price CSVs, the v13 primary forecasts.csv files and
the v13 sens/<rule>_<asset>.csv files. Usage:

    python -I layers.py --prices DIR --runs DIR --sens DIR --out DIR
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, pandas as pd

SEED, BLOCK, DRAWS, ALPHA, COST, MIN_TRAIN = 20261113, 20, 5000, 1.0, 0.0005, 2520
ASSETS = ("SPY", "QQQ", "IWM")
RULES = ("primary", "fixed", "roll2520", "roll1260")
MODELS = ("iid", "vol2", "m3", "U", "R")
LAYERS = (("vol2", "iid"), ("m3", "vol2"), ("U", "m3"), ("U", "iid"), ("U", "vol2"), ("R", "U"), ("R", "m3"))
SEGMENTS = (("2015-2018", "2015-01-01", "2018-12-31"), ("2019-2022", "2019-01-01", "2022-12-31"),
            ("2023-2025", "2023-01-01", "2025-12-31"))


def block_indices(n):
    rng = np.random.default_rng(SEED)
    per = int(np.ceil(n / BLOCK))
    st = rng.integers(0, n - BLOCK + 1, (DRAWS, per))
    return (st[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(DRAWS, -1)[:, :n]


def interval(x, idx):
    x = np.asarray(x, float)
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return [float(x.mean()), float(lo), float(hi)]


def classify(v, lo, hi):
    return np.where(v <= lo, 0, np.where(v <= hi, 1, 2))


def benchmark_probs(s):
    """vol2 and m3 forecasts from a labelled history s (current state s[-1])."""
    dest = s[1:]
    nu, nl = (dest == 2).sum() + ALPHA, (dest == 0).sum() + ALPHA
    q = nu / (nu + nl)
    g = (s == 1).astype(int)
    G = np.zeros((2, 2)); np.add.at(G, (g[:-1], g[1:]), 1.0); G += ALPHA
    m2 = G[g[-1], 1] / G[g[-1]].sum()
    C = np.zeros((3, 3)); np.add.at(C, (s[:-1], s[1:]), 1.0)
    P = (C + ALPHA) / (C + ALPHA).sum(axis=1, keepdims=True)
    m3 = P[s[-1], 1]
    return q, P, [(1 - m2) * (1 - q), m2, (1 - m2) * q], [(1 - m3) * (1 - q), m3, (1 - m3) * q], float(nu + nl)


def panel(prices, saved, rule):
    px = pd.read_csv(prices, parse_dates=["Date"]).set_index("Date").loc[:"2025-12-31"]
    r = (px.Close / px.Open - 1.0).to_numpy(float)
    pos = {d: i for i, d in enumerate(px.index)}
    f = pd.read_csv(saved, parse_dates=["signal_date", "target_date"])
    f = f[f.split == "final_test"].reset_index(drop=True)
    fixed = np.quantile(r[:MIN_TRAIN], [1 / 3, 2 / 3])
    rows, err = [], 0.0
    for _, row in f.iterrows():
        o = pos[row.signal_date]
        if rule == "fixed":
            lo, hi = fixed; hist = r[: o + 1]
        elif rule.startswith("roll"):
            w = int(rule[4:]); hist = r[o + 1 - w: o + 1]; lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
        else:
            hist = r[: o + 1]; lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
        s = classify(hist, lo, hi)
        y = int(classify(np.array([r[o + 1]]), lo, hi)[0])
        if y != int(row.target_state):
            raise AssertionError("target label mismatch")
        q, P, p2, p3, denom = benchmark_probs(s)
        pu = [row.p_markov_down, row.p_markov_neutral, row.p_markov_up]
        err = max(err, float(np.abs(P[s[-1]] - pu).max()))
        rec = dict(date=row.target_date, current=int(s[-1]), y=y, r=float(r[o + 1]), q=float(q),
                   half_bound=1.0 / denom)
        probs = dict(iid=[row.p_iid_down, row.p_iid_neutral, row.p_iid_up], vol2=p2, m3=p3, U=pu,
                     R=[row.p_reversible_down, row.p_reversible_neutral, row.p_reversible_up])
        for k, p in probs.items():
            rec[f"p_{k}_0"], rec[f"p_{k}_1"], rec[f"p_{k}_2"] = map(float, p)
        rows.append(rec)
    if err > 1e-12:
        raise AssertionError(f"unrestricted forecast mismatch {err}")
    return pd.DataFrame(rows), err


def derived(L):
    out = {}
    y, r = L.y.to_numpy(), L.r.to_numpy()
    outer = y != 1
    wabs = np.abs(r) / np.abs(r[outer]).mean()
    for k in MODELS:
        p = L[[f"p_{k}_0", f"p_{k}_1", f"p_{k}_2"]].to_numpy()
        m = p[:, 1]; q = p[:, 2] / (p[:, 0] + p[:, 2])
        ll = -np.log(p[np.arange(len(y)), y])
        A = np.where(y == 1, -np.log(m), -np.log(1 - m))
        B = np.where(outer, -np.where(y == 2, np.log(q), np.log(1 - q)), 0.0)
        Bw = B * np.where(outer, wabs, 0.0)
        w = np.clip(0.5 + 0.5 * (p[:, 2] - p[:, 0]), 0, 1)
        out[k] = dict(ll=ll, A=A, B=B, Bw=Bw, w=w, gross=w * r, net=w * r - 2 * COST * w)
    return out


def summarize(L, idx=None):
    D = derived(L)
    idx = block_indices(len(L)) if idx is None else idx
    res = {"levels": {k: float(D[k]["ll"].mean()) for k in MODELS},
           "mean_abs_tilt": {k: float(np.abs(D[k]["w"] - 0.5).mean()) for k in MODELS},
           "q_range": [float(L.q.min()), float(L.q.max())],
           "max_abs_tilt_vol2": float(np.abs(D["vol2"]["w"] - 0.5).max()),
           "max_tilt_over_bound_vol2": float((np.abs(D["vol2"]["w"] - 0.5) / L.half_bound).max()),
           "max_tilt_over_bound": {k: float((np.abs(D[k]["w"] - 0.5) / L.half_bound).max()) for k in ("iid", "vol2", "m3")},
           "max_abs_tilt": {k: float(np.abs(D[k]["w"] - 0.5).max()) for k in MODELS}}
    for a, b in LAYERS:
        res[f"{a}-{b}"] = {key: interval(D[a][key] - D[b][key], idx) for key in ("ll", "A", "B", "Bw", "net", "gross")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices"); ap.add_argument("--runs"); ap.add_argument("--sens"); ap.add_argument("--out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    result = {"settings": dict(seed=SEED, block=BLOCK, draws=DRAWS, alpha=ALPHA, cost=COST)}
    for rule in RULES:
        for t in ASSETS:
            saved = (glob.glob(os.path.join(a.runs, t, "run_*", "forecasts.csv"))[0] if rule == "primary"
                     else os.path.join(a.sens, f"{rule}_{t}.csv"))
            L, err = panel(os.path.join(a.prices, f"{t}.csv"), saved, rule)
            res = summarize(L); res["max_unrestricted_error"] = err; res["n"] = len(L)
            if rule == "primary":
                res["segments"] = {}
                for name, lo, hi in SEGMENTS:
                    M = L[(L.date >= lo) & (L.date <= hi)].reset_index(drop=True)
                    res["segments"][name] = summarize(M)
                D = derived(L)
                res["state_weights_U"] = {int(s): float(D["U"]["w"][L.current == s].mean()) for s in (0, 1, 2)}
                res["state_mean_return_bp"] = state_means(L)
                res["rolling_frequency"] = rolling_frequency(a, t, L)
            result[f"{rule}:{t}"] = res
            print(rule, t, "n", len(L), "U err", f"{err:.1e}", flush=True)
    cells = [(t, k) for t in ASSETS for k in ("0", "1", "2")]
    boots = np.column_stack([result[f"primary:{t}"]["state_mean_return_bp"][k].pop("boot") for t, k in cells])
    est = np.array([result[f"primary:{t}"]["state_mean_return_bp"][k]["mean"] for t, k in cells])
    se = boots.std(axis=0, ddof=1)
    q95 = float(np.quantile(np.max(np.abs((boots - est) / se), axis=1), 0.95))
    for (t, k), e, sd in zip(cells, est, se):
        result[f"primary:{t}"]["state_mean_return_bp"][k]["simultaneous_ci"] = [float(e - q95 * sd), float(e + q95 * sd)]
    result["state_mean_critical_value"] = q95
    with open(os.path.join(a.out, "layers.json"), "w") as fh:
        json.dump(result, fh, indent=1)


def rolling_frequency(a, t, L):
    """Class frequencies of the last W labelled sessions, under the day's primary cut points."""
    px = pd.read_csv(os.path.join(a.prices, f"{t}.csv"), parse_dates=["Date"]).set_index("Date").loc[:"2025-12-31"]
    r = (px.Close / px.Open - 1.0).to_numpy(float); pos = {d: i for i, d in enumerate(px.index)}
    sig = px.index[[pos[d] - 1 for d in L.date]]
    D = derived(L); idx = block_indices(len(L)); out = {}
    lls = {W: [] for W in (252, 504, 1260)}
    for d, y in zip(sig, L.y):
        o = pos[d]; hist = r[: o + 1]; lo, hi = np.quantile(hist, [1 / 3, 2 / 3]); s = classify(hist, lo, hi)
        for W in lls:
            c = np.bincount(s[-W:], minlength=3) + ALPHA; lls[W].append(-np.log(c[y] / c.sum()))
    for W, v in lls.items():
        v = np.array(v)
        out[str(W)] = {"U-roll": interval(D["U"]["ll"] - v, idx), "roll-iid": interval(v - D["iid"]["ll"], idx)}
        for name, lo, hi in SEGMENTS:
            mask = ((L.date >= lo) & (L.date <= hi)).to_numpy(); sidx = block_indices(int(mask.sum()))
            out[str(W)][name] = {"U-roll": interval((D["U"]["ll"] - v)[mask], sidx),
                                 "roll-iid": interval((v - D["iid"]["ll"])[mask], sidx)}
    return out


def state_means(L):
    """Mean next-session return (bp) by current state, with pointwise and nine-cell
    simultaneous intervals from the common moving-block draws (ratio of resampled sums)."""
    idx = block_indices(len(L)); r = L.r.to_numpy() * 1e4; cur = L.current.to_numpy()
    out = {}
    for s in (0, 1, 2):
        ind = (cur == s).astype(float)
        boot = (r * ind)[idx].sum(axis=1) / ind[idx].sum(axis=1)
        est = float(r[cur == s].mean())
        out[str(s)] = dict(mean=est, se=float(boot.std(ddof=1)),
                           ci=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                           boot=boot)
    return out


if __name__ == "__main__":
    main()
