"""Score components, circulation statistics, allocation sensitivities and bin counts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import ALPHA, Config, block_indices, load_models, percentile_ci

MODELS = ("iid", "markov", "reversible")
LABEL = {"iid": "IID", "markov": "U", "reversible": "R"}


def losses(frame: pd.DataFrame):
    y = frame["target_state"].to_numpy()
    out = {}
    for m in MODELS:
        P = frame[[f"p_{m}_down", f"p_{m}_neutral", f"p_{m}_up"]].to_numpy()
        out[m] = -np.log(P[np.arange(len(frame)), y])
    return out


def components(frame: pd.DataFrame):
    """A: middle versus outer; B: upper versus lower, outer-state weighted."""
    y = frame["target_state"].to_numpy()
    out = {}
    for m in MODELS:
        p1, p2, p3 = (frame[f"p_{m}_{k}"].to_numpy() for k in ("down", "neutral", "up"))
        q = p3 / (p1 + p3)
        z = (y == 1).astype(float)
        u = (y == 2).astype(float)
        A = -(z * np.log(p2) + (1 - z) * np.log1p(-p2))
        B = -(1 - z) * (u * np.log(q) + (1 - u) * np.log1p(-q))
        out[m] = (A, B)
    return out


def component_table(cfg: Config):
    idx_cache = {}
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        if len(f) not in idx_cache:
            idx_cache[len(f)] = block_indices(len(f))
        idx = idx_cache[len(f)]
        comp = components(f)
        for m in ("markov", "reversible"):
            row = dict(asset=t, model=LABEL[m])
            for name, k in (("A", 0), ("B", 1)):
                v = comp[m][k]
                d = v - comp["iid"][k]
                lo, hi = percentile_ci(d, idx)
                row[f"{name}"] = float(v.mean())
                row[f"d{name}"] = float(d.mean())
                row[f"d{name}_ci"] = [lo, hi]
            rows.append(row)
    return rows


def relative_components(cfg: Config):
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        idx = block_indices(len(f))
        comp = components(f)
        row = dict(asset=t)
        for name, k in (("A", 0), ("B", 1)):
            d = comp["reversible"][k] - comp["markov"][k]
            lo, hi = percentile_ci(d, idx)
            row[f"d{name}"] = float(d.mean())
            row[f"d{name}_ci"] = [lo, hi]
        rows.append(row)
    return rows


def derivatives(cfg: Config):
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        for m in MODELS:
            p1, p2, p3 = (f[f"p_{m}_{k}"].to_numpy() for k in ("down", "neutral", "up"))
            q = p3 / (p1 + p3)
            rows.append(dict(asset=t, model=m, q_mean=float(q.mean()),
                             dw_dm=float(np.abs(2 * q - 1).mean() / 2),
                             dw_dq=float((1 - p2).mean())))
    return rows


def circulation(cfg: Config):
    """Per-origin fitted matrices from the frozen estimator, plus the scalar j_t."""
    models, states = load_models(cfg)
    rows = []
    for t in cfg.assets:
        prices = cfg.prices_frame(t)
        r = (prices["Close"] / prices["Open"] - 1.0).to_numpy(float)
        dates = prices.index
        for o in range(2519, len(prices) - 1):
            hist = states.classify_returns(r[: o + 1], states.fit_thresholds(r[: o + 1]))
            P = models.markov_matrix(hist, alpha=ALPHA)
            R = models.reversible_matrix(hist, alpha=ALPHA)[0]
            pi = _stationary(P)
            j = float(pi[0] * P[0, 1] - pi[1] * P[1, 0])
            rows.append(dict(asset=t, signal_date=dates[o], target_date=dates[o + 1],
                             j=j, pi0=pi[0], pi1=pi[1], pi2=pi[2],
                             **{f"P{i}{k}": float(P[i - 1, k - 1]) for i in (1, 2, 3) for k in (1, 2, 3)},
                             **{f"R{i}{k}": float(R[i - 1, k - 1]) for i in (1, 2, 3) for k in (1, 2, 3)}))
    frame = pd.DataFrame(rows)
    frame["split"] = np.where(frame.target_date <= "2009-12-31", "development",
                              np.where(frame.target_date <= "2014-12-31", "validation", "final_test"))
    frame.to_csv(cfg.work / "results" / "matrices.csv", index=False, date_format="%Y-%m-%d")
    return frame


def _stationary(P):
    w, v = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    x = np.real(v[:, k])
    return np.abs(x / x.sum())


def circulation_summary(frame: pd.DataFrame, assets):
    rows = []
    for t in assets:
        d = frame[(frame.asset == t) & (frame.split == "final_test")]
        j = d["j"].to_numpy()
        diff = np.abs(d[[f"P{i}{k}" for i in (1, 2, 3) for k in (1, 2, 3)]].to_numpy()
                     - d[[f"R{i}{k}" for i in (1, 2, 3) for k in (1, 2, 3)]].to_numpy())
        mask = ~np.eye(3, dtype=bool)
        rows.append(dict(asset=t, n=len(d), mean=float(j.mean()) * 100, min=float(j.min()) * 100,
                         max=float(j.max()) * 100, ar1=float(pd.Series(j).autocorr(1)),
                         mean_abs_diff=float(diff[:, mask.ravel()].mean()),
                         max_abs_diff=float(diff[:, mask.ravel()].max()),
                         share_negative=float((j < 0).mean()),
                         pi_dev=float(np.abs(d[["pi0", "pi1", "pi2"]].to_numpy() - 1 / 3).max())))
    return rows


def bins(cfg: Config):
    """Bin occupancy aggregated over the three one-versus-rest classes."""
    edges = np.linspace(0, 1, 11)
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        for m in MODELS:
            empty, smallest, largest = 0, None, 0
            for name in ("down", "neutral", "up"):
                counts = np.bincount(
                    np.digitize(f[f"p_{m}_{name}"].to_numpy(), edges[1:-1], right=False), minlength=10)
                empty += int((counts == 0).sum())
                largest = max(largest, int(counts.max()))
                if counts[counts > 0].size:
                    smallest = int(counts[counts > 0].min()) if smallest is None else min(smallest, int(counts[counts > 0].min()))
            rows.append(dict(asset=t, model=m, empty=empty, smallest=smallest, largest=largest))
    return rows

def calibration(cfg: Config, draws: int = 5000, block: int = 20, seed: int = 20261113):
    """Count-weighted calibration error per asset and model, with a block bootstrap.

    The statistic is the mean over the three one-versus-rest classes of
    sum_b (n_b / N) * |mean forecast in b - observed frequency in b|.
    """
    from .core import SEED, BLOCK, DRAWS, block_indices

    edges = np.linspace(0, 1, 11)
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        y = f["target_state"].to_numpy()
        n = len(f)
        idx = block_indices(n, block, draws, seed)
        for m in MODELS:
            onehot = np.eye(3)[y]
            bin_index = [np.digitize(f[f"p_{m}_{name}"].to_numpy(), edges[1:-1], right=False)
                         for name in ("down", "neutral", "up")]
            prob = [f[f"p_{m}_{name}"].to_numpy() for name in ("down", "neutral", "up")]
            stats = np.empty(draws)
            for start in range(0, draws, 500):
                chunk = idx[start:start + 500]
                stats[start:start + 500] = _ece_chunk(chunk, bin_index, prob, onehot, edges)
            lo, hi = np.percentile(stats, [2.5, 97.5])
            observed = float(_ece_chunk(np.arange(n)[None, :], bin_index, prob, onehot, edges)[0])
            rows.append(dict(asset=t, model=m, ece=observed, bootstrap_mean=float(stats.mean()), ci=[float(lo), float(hi)],
                             bins_30=int(_bins_with_support(bin_index, 30))))
    return rows


def _ece_chunk(chunk, bin_index, prob, onehot, edges):
    out = np.empty(len(chunk))
    for j, days in enumerate(chunk):
        total = 0.0
        n = len(days)
        for k in range(3):
            b = bin_index[k][days]
            counts = np.bincount(b, minlength=10)
            mass = np.bincount(b, weights=prob[k][days], minlength=10)
            obs = np.bincount(b, weights=onehot[days, k], minlength=10)
            mask = counts > 0
            mean_p = mass[mask] / counts[mask]
            freq = obs[mask] / counts[mask]
            total += float(np.sum(counts[mask] / n * np.abs(mean_p - freq)))
        out[j] = total / 3.0
    return out


def _bins_with_support(bin_index, threshold):
    return sum(int((np.bincount(b, minlength=10) >= threshold).sum()) for b in bin_index)

def calibration_test(cfg: Config, draws: int = 500, seed: int = 20261115):
    """Independent-draw reference diagnostic for observed bin-level error.

    Freezes the full forecast sequence and draws outcomes independently.
    This stronger reference model does not regenerate adaptive forecasts and
    is not a general test of sequential calibration under serial dependence.
    """
    edges = np.linspace(0, 1, 11)
    rng = np.random.default_rng(seed)
    rows = []
    for t in cfg.assets:
        f = cfg.final(t)
        n = len(f)
        for m in MODELS:
            prob = np.column_stack([f[f"p_{m}_{name}"].to_numpy() for name in ("down", "neutral", "up")])
            cum = np.cumsum(prob, axis=1)
            bin_index = [np.digitize(prob[:, k], edges[1:-1], right=False) for k in range(3)]
            counts = [np.bincount(b, minlength=10) for b in bin_index]
            mean_p = [np.bincount(bin_index[k], weights=prob[:, k], minlength=10) / np.maximum(counts[k], 1)
                      for k in range(3)]
            observed = _ece_observed(f, m, edges, bin_index, counts, mean_p, n)
            null = np.empty(draws)
            for b in range(draws):
                u = rng.random(n)
                y = (cum < u[:, None]).sum(axis=1)
                total = 0.0
                for k in range(3):
                    obs = np.bincount(bin_index[k], weights=(y == k).astype(float), minlength=10)
                    freq = np.divide(obs, counts[k], out=np.zeros(10), where=counts[k] > 0)
                    total += float(np.sum(counts[k][counts[k] > 0] / n * np.abs(mean_p[k][counts[k] > 0] - freq[counts[k] > 0])))
                null[b] = total / 3.0
            p_value = (1 + int(np.sum(null >= observed))) / (draws + 1)
            rows.append(dict(asset=t, model=m, ece=float(observed), p_value=float(p_value),
                             null_mean=float(null.mean()), null_q95=float(np.quantile(null, 0.95))))
    return rows


def _ece_observed(f, model, edges, bin_index, counts, mean_p, n):
    y = f["target_state"].to_numpy()
    total = 0.0
    for k in range(3):
        obs = np.bincount(bin_index[k], weights=(y == k).astype(float), minlength=10)
        freq = np.divide(obs, counts[k], out=np.zeros(10), where=counts[k] > 0)
        mask = counts[k] > 0
        total += float(np.sum(counts[k][mask] / n * np.abs(mean_p[k][mask] - freq[mask])))
    return total / 3.0
