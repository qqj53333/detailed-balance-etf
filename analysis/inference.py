"""Familywise inference over the six score contrasts, calendar segments and rolling paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import BLOCK, DRAWS, SEED, Config, block_indices, percentile_ci
from .scores import losses


def six_contrasts(cfg: Config):
    """Daily paired differences, aligned on common target dates."""
    series, dates = {}, None
    for t in cfg.assets:
        f = cfg.final(t)
        if dates is None:
            dates = f["target_date"].to_numpy()
        elif not (dates == f["target_date"].to_numpy()).all():
            raise ValueError("assets do not share final-period dates")
        ll = losses(f)
        series[(t, "ui")] = ll["markov"] - ll["iid"]
        series[(t, "ru")] = ll["reversible"] - ll["markov"]
    names = [f"{t}:{k}" for t in cfg.assets for k in ("ui", "ru")]
    matrix = np.column_stack([series[(t, k)] for t in cfg.assets for k in ("ui", "ru")])
    return names, matrix


def romano_wolf(matrix: np.ndarray, names):
    n = len(matrix)
    idx = block_indices(n)
    means = matrix[idx].mean(axis=1)
    obs = matrix.mean(axis=0)
    se = means.std(axis=0, ddof=1)
    t_obs = obs / se
    t_star = (means - obs) / se
    max_abs = np.max(np.abs(t_star), axis=1)

    p_single = np.array([(1 + np.sum(max_abs >= abs(v))) / (DRAWS + 1) for v in t_obs])
    p_raw = np.array([(1 + np.sum(np.abs(t_star[:, j]) >= abs(t_obs[j]))) / (DRAWS + 1)
                      for j in range(len(names))])

    order = np.argsort(-np.abs(t_obs))
    p_step = np.empty(len(names))
    for k, c in enumerate(order):
        remaining = order[k:]
        stat = np.max(np.abs(t_star[:, remaining]), axis=1)
        p_step[c] = (1 + np.sum(stat >= abs(t_obs[c]))) / (DRAWS + 1)
    for k in range(1, len(names)):
        p_step[order[k]] = max(p_step[order[k]], p_step[order[k - 1]])

    holm = np.empty(len(names))
    running = 0.0
    for k, c in enumerate(np.argsort(p_raw)):
        running = max(running, min(1.0, (len(names) - k) * p_raw[c]))
        holm[c] = running

    q95 = float(np.quantile(max_abs, 0.95))
    pointwise = {}
    for j, nm in enumerate(names):
        resampled = matrix[idx][:, :, j].mean(axis=1)
        lo, hi = np.percentile(resampled, [2.5, 97.5])
        pointwise[nm] = [float(lo), float(hi)]
    rows = []
    for j, nm in enumerate(names):
        asset, kind = nm.split(":")
        rows.append(dict(contrast=nm, asset=asset, kind=kind, mean=float(obs[j]),
                         se=float(se[j]), t=float(t_obs[j]), p_raw=float(p_raw[j]),
                         p_holm=float(holm[j]), p_stepdown=float(p_step[j]),
                         p_single=float(p_single[j]),
                         simultaneous_ci=[float(obs[j] - q95 * se[j]), float(obs[j] + q95 * se[j])]))
    return dict(q95=q95, seed=SEED, block=BLOCK, draws=DRAWS, rows=rows, pointwise=pointwise)


def segments(cfg: Config):
    bounds = [("2015-2018", "2015-01-01", "2018-12-31"),
              ("2019-2022", "2019-01-01", "2022-12-31"),
              ("2023-2025", "2023-01-01", "2025-12-31")]
    out = {}
    for t in cfg.assets:
        f = cfg.final(t)
        ll = losses(f)
        ui = ll["markov"] - ll["iid"]
        ru = ll["reversible"] - ll["markov"]
        dates = pd.to_datetime(f["target_date"])
        rows = []
        for name, a, b in bounds:
            mask = (dates >= a) & (dates <= b)
            x, y = ui[mask.to_numpy()], ru[mask.to_numpy()]
            idx = block_indices(len(x))
            lo1, hi1 = percentile_ci(x, idx)
            lo2, hi2 = percentile_ci(y, idx)
            rows.append(dict(seg=name, n=int(mask.sum()), ui=float(x.mean()), ui_ci=[lo1, hi1],
                             ru=float(y.mean()), ru_ci=[lo2, hi2]))
        out[t] = rows
    return out


def rolling_paths(cfg: Config):
    out = {}
    for t in cfg.assets:
        f = cfg.final(t)
        ll = losses(f)
        frame = pd.DataFrame(dict(date=pd.to_datetime(f["target_date"]),
                                  ui=ll["markov"] - ll["iid"],
                                  ru=ll["reversible"] - ll["markov"])).set_index("date").sort_index()
        out[t] = frame.rolling(252).mean().dropna()
    return out

def brier_contrasts(cfg: Config):
    """Six paired Brier-score contrasts: model minus IID, two per asset."""
    series, dates = {}, None
    for t in cfg.assets:
        f = cfg.final(t)
        if dates is None:
            dates = f["target_date"].to_numpy()
        y = f["target_state"].to_numpy()
        score = {}
        for m in ("iid", "markov", "reversible"):
            P = f[[f"p_{m}_down", f"p_{m}_neutral", f"p_{m}_up"]].to_numpy()
            onehot = np.eye(3)[y]
            score[m] = ((P - onehot) ** 2).sum(axis=1)
        series[(t, "ui")] = score["markov"] - score["iid"]
        series[(t, "ru")] = score["reversible"] - score["markov"]
    names = [f"{t}:{k}" for t in cfg.assets for k in ("ui", "ru")]
    return names, np.column_stack([series[(t, k)] for t in cfg.assets for k in ("ui", "ru")])


def net_returns(cfg: Config, ticker: str, cost: float = 0.0005):
    f = cfg.final(ticker)
    r = f["target_intraday_return"].to_numpy()
    out = {}
    for m in ("iid", "markov", "reversible"):
        w = (0.5 + 0.5 * (f[f"p_{m}_up"] - f[f"p_{m}_down"])).to_numpy()
        out[m] = w * r - cost * 2 * w
    return out


def economic_contrasts(cfg: Config, cost: float = 0.0005):
    """Six paired mean-net-return contrasts: model minus IID, two per asset."""
    series = {}
    for t in cfg.assets:
        net = net_returns(cfg, t, cost)
        series[(t, "ui")] = net["markov"] - net["iid"]
        series[(t, "ru")] = net["reversible"] - net["markov"]
    names = [f"{t}:{k}" for t in cfg.assets for k in ("ui", "ru")]
    return names, np.column_stack([series[(t, k)] for t in cfg.assets for k in ("ui", "ru")])

def sharpe_draws(cfg: Config, cost: float = 0.0005, chunk: int = 500):
    """Bootstrap draws of the Sharpe ratio for each model and asset.

    Returns a dict (asset, model) -> array of length DRAWS. Sharpe is computed on
    the resampled days exactly as in the point estimate: sqrt(252)*mean/std.
    """
    out = {}
    for t in cfg.assets:
        net = net_returns(cfg, t, cost)
        n = len(net["iid"])
        idx = block_indices(n)
        for m in ("iid", "markov", "reversible"):
            values = np.empty(len(idx))
            for start in range(0, len(idx), chunk):
                block = idx[start:start + chunk]
                sample = net[m][block]
                means = sample.mean(axis=1)
                sds = sample.std(axis=1, ddof=1)
                values[start:start + chunk] = np.sqrt(252.0) * means / sds
            out[(t, m)] = values
    return out


def economic_family(cfg: Config, cost: float = 0.0005, chunk: int = 500):
    """Twelve economic contrasts: six mean net returns and six Sharpe differences."""
    net = {t: net_returns(cfg, t, cost) for t in cfg.assets}
    sharpe = sharpe_draws(cfg, cost, chunk)
    mean_draws = {t: {} for t in cfg.assets}
    for t in cfg.assets:
        n = len(net[t]["iid"])
        idx = block_indices(n)
        for m in ("iid", "markov", "reversible"):
            values = np.empty(len(idx))
            for start in range(0, len(idx), chunk):
                block = idx[start:start + chunk]
                values[start:start + chunk] = net[t][m][block].mean(axis=1)
            mean_draws[t][m] = values

    pairs = (("markov", "iid", "ui"), ("reversible", "markov", "ru"))
    columns, names, obs = [], [], []
    for t in cfg.assets:
        for a, b, kind in pairs:
            columns.append(mean_draws[t][a] - mean_draws[t][b])
            names.append(f"{t}:{kind}:net")
            obs.append(float((net[t][a] - net[t][b]).mean()))
    for t in cfg.assets:
        for a, b, kind in pairs:
            columns.append(sharpe[(t, a)] - sharpe[(t, b)])
            names.append(f"{t}:{kind}:sharpe")
            obs.append(_sharpe(net[t][a]) - _sharpe(net[t][b]))
    return names, np.column_stack(columns), np.array(obs)


def _sharpe(values):
    return float(np.sqrt(252.0) * values.mean() / values.std(ddof=1))


def family_stepdown(matrix: np.ndarray, obs: np.ndarray, names):
    """Romano-Wolf stepdown for statistics whose bootstrap draws are given."""
    draws = matrix.shape[0]
    se = matrix.std(axis=0, ddof=1)
    t_obs = obs / se
    t_star = (matrix - obs) / se
    max_abs = np.max(np.abs(t_star), axis=1)
    order = np.argsort(-np.abs(t_obs))
    p_step = np.empty(len(names))
    for k, c in enumerate(order):
        remaining = order[k:]
        stat = np.max(np.abs(t_star[:, remaining]), axis=1)
        p_step[c] = (1 + np.sum(stat >= abs(t_obs[c]))) / (draws + 1)
    for k in range(1, len(names)):
        p_step[order[k]] = max(p_step[order[k]], p_step[order[k - 1]])
    p_raw = np.array([(1 + np.sum(np.abs(t_star[:, j]) >= abs(t_obs[j]))) / (draws + 1)
                      for j in range(len(names))])
    holm = np.empty(len(names))
    running = 0.0
    for k, c in enumerate(np.argsort(p_raw)):
        running = max(running, min(1.0, (len(names) - k) * p_raw[c]))
        holm[c] = running
    q95 = float(np.quantile(max_abs, 0.95))
    rows = []
    for j, nm in enumerate(names):
        asset, kind, metric = nm.split(":")
        rows.append(dict(contrast=nm, asset=asset, kind=kind, metric=metric,
                         mean=float(obs[j]), se=float(se[j]), t=float(t_obs[j]),
                         p_raw=float(p_raw[j]), p_holm=float(holm[j]),
                         p_stepdown=float(p_step[j]),
                         simultaneous_ci=[float(obs[j] - q95 * se[j]), float(obs[j] + q95 * se[j])]))
    return dict(q95=q95, draws=draws, rows=rows)