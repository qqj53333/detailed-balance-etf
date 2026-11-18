"""Alternative estimation rules for the threshold and window sensitivity analysis.

The primary rule reproduces the frozen pipeline from the reconstruction; the four
alternatives change only where cut points and transition counts come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from scipy.special import softmax

from .core import ALPHA, MIN_TRAIN, Config, load_models

PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

RULES = {
    "primary": dict(mode="expanding", window=None, start=None),
    "fixed": dict(mode="fixed", window=None, start=None),
    "roll2520": dict(mode="rolling", window=2520, start=None),
    "roll1260": dict(mode="rolling", window=1260, start=None),
    "common": dict(mode="expanding", window=None, start="2000-05-26"),
}


def _prep(states, alpha=ALPHA):
    counts = np.zeros((3, 3))
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    smoothed = counts + alpha
    c = np.array([smoothed[i, i] if i == j else smoothed[i, j] + smoothed[j, i] for i, j in PAIRS])
    return c, smoothed.sum(axis=1), np.log(c / c.sum())[:-1]


def _flow(params):
    masses = softmax(np.concatenate([params, [0.0]]))
    flow = np.zeros((3, 3))
    for k, (i, j) in enumerate(PAIRS):
        if i == j:
            flow[i, i] = masses[k]
        else:
            flow[i, j] = masses[k] / 2.0
            flow[j, i] = masses[k] / 2.0
    return flow, flow.sum(axis=1)


def _nll(params, c, r):
    _, pi = _flow(params)
    masses = softmax(np.concatenate([params, [0.0]]))
    return float(-np.sum(c * np.log(masses)) + np.sum(r * np.log(pi)))


def _grad(params, c, r):
    masses = softmax(np.concatenate([params, [0.0]]))
    _, pi = _flow(params)
    gm = -c / masses
    for k, (i, j) in enumerate(PAIRS):
        gm[k] += r[i] / pi[i] if i == j else 0.5 * (r[i] / pi[i] + r[j] / pi[j])
    g = masses * (gm - float(np.dot(masses, gm)))
    return g[:-1]


def _masses_flow(masses):
    flow = np.zeros((3, 3))
    for k, (i, j) in enumerate(PAIRS):
        if i == j:
            flow[i, i] = masses[k]
        else:
            flow[i, j] = masses[k] / 2.0
            flow[j, i] = masses[k] / 2.0
    return flow, flow.sum(axis=1)


def _masses_nll(masses, c, r):
    _, pi = _masses_flow(masses)
    return float(-np.sum(c * np.log(masses)) + np.sum(r * np.log(pi)))


def _masses_grad(masses, c, r):
    _, pi = _masses_flow(masses)
    g = -c / masses
    for k, (i, j) in enumerate(PAIRS):
        g[k] += r[i] / pi[i] if i == j else 0.5 * (r[i] / pi[i] + r[j] / pi[j])
    return g


def reversible_fit(models, states, alpha=ALPHA):
    """Frozen estimator first; on a line-search precision limit, solve the same
    objective directly and accept only a genuinely stationary point.

    The fallback minimises over the six flow masses on the simplex, then polishes
    the KKT system grad(m) = lambda, sum(m) = 1 with a root solver. Stationarity is
    measured with the projected gradient, because on the simplex only directions
    orthogonal to the unit vector are feasible.
    """
    try:
        return models.reversible_matrix(states, alpha=alpha)[0], "package", 0.0
    except RuntimeError:
        c, r, _ = _prep(states, alpha)
        cons = [{"type": "eq", "fun": lambda m: m.sum() - 1.0, "jac": lambda m: np.ones(6)}]
        res = minimize(_masses_nll, c / c.sum(), args=(c, r), jac=_masses_grad, method="SLSQP",
                       bounds=[(1e-14, None)] * 6, constraints=cons,
                       options={"maxiter": 3000, "ftol": 1e-16})
        masses = res.x

        def kkt(x):
            m, lam = x[:6], x[6]
            return np.concatenate([_masses_grad(m, c, r) - lam, [m.sum() - 1.0]])

        sol = least_squares(kkt, np.concatenate([masses, [0.0]]),
                            xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
        if (np.all(np.isfinite(sol.x)) and np.all(sol.x[:6] > 1e-12)
                and _masses_nll(sol.x[:6], c, r) <= _masses_nll(masses, c, r) + 1e-9):
            masses = sol.x[:6]

        g = _masses_grad(masses, c, r)
        gnorm = float(np.abs(g - g.mean()).max())
        flow, pi = _masses_flow(masses)
        if (not np.isfinite(gnorm) or gnorm > 1e-8 or not np.isfinite(flow).all()
                or not (flow > 0).all() or not (pi > 0).all()):
            raise RuntimeError(f"Fallback reversible fit failed validation: projected gradient={gnorm}")
        return flow / pi[:, None], "fallback", gnorm


def run_rule(cfg: Config, ticker: str, rule: str) -> pd.DataFrame:
    models, states = load_models(cfg)
    spec = RULES[rule]
    prices = cfg.prices_frame(ticker)
    if spec["start"]:
        prices = prices.loc[spec["start"]:]
    r = (prices["Close"] / prices["Open"] - 1.0).to_numpy(float)
    dates = prices.index
    fixed = None
    rows = []
    for o in range(MIN_TRAIN - 1, len(prices) - 1):
        if spec["mode"] == "expanding":
            cut = states.fit_thresholds(r[: o + 1]); hist = r[: o + 1]
        elif spec["mode"] == "fixed":
            fixed = fixed or states.fit_thresholds(r[: o + 1])
            cut = fixed; hist = r[: o + 1]
        else:
            w = spec["window"]
            cut = states.fit_thresholds(r[o + 1 - w: o + 1]); hist = r[o + 1 - w: o + 1]
        hist_states = states.classify_returns(hist, cut)
        current = int(hist_states[-1])
        p_iid = models.iid_probabilities(hist_states, alpha=ALPHA)
        P = models.markov_matrix(hist_states, alpha=ALPHA)
        R, how, gnorm = reversible_fit(models, hist_states)
        target = float(r[o + 1])
        rows.append(dict(
            ticker=ticker, rule=rule, signal_date=dates[o], target_date=dates[o + 1],
            current_state=int(current), target_state=int(states.classify_returns(np.array([target]), cut)[0]),
            target_intraday_return=target, fit=how, gradient=gnorm,
            **{f"p_iid_{k}": float(p_iid[i]) for i, k in enumerate(("down", "neutral", "up"))},
            **{f"p_markov_{k}": float(P[current, i]) for i, k in enumerate(("down", "neutral", "up"))},
            **{f"p_reversible_{k}": float(R[current, i]) for i, k in enumerate(("down", "neutral", "up"))},
        ))
    frame = pd.DataFrame(rows)
    frame["split"] = np.where(frame.target_date <= "2009-12-31", "development",
                              np.where(frame.target_date <= "2014-12-31", "validation", "final_test"))
    frame.to_csv(cfg.work / "sens" / f"{rule}_{ticker}.csv", index=False, date_format="%Y-%m-%d")
    return frame