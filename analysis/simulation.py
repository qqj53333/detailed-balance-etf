"""Controlled simulation: expected cost of imposing detailed balance under known truth."""

from __future__ import annotations

import numpy as np

from .core import Config, load_models
from .rules import reversible_fit

PI = np.ones(3) / 3.0
DIAG = 0.35
CYCLE = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
SCENARIOS = {"reversible": 0.0, "weak_irreversible": -0.007, "empirical_irreversible": -0.014}
LENGTHS = (250, 500, 1260, 2520, 5000)
REPS = 500
SEED = 20261115


def symmetric_flows():
    diag = DIAG / 3.0
    off = (1.0 / 3.0 - diag) / 2.0
    S = np.full((3, 3), off)
    np.fill_diagonal(S, diag)
    return S * ((1.0 / 3.0) / S.sum(axis=1, keepdims=True))


def build(j, S=None):
    S = symmetric_flows() if S is None else S
    flow = S + (j / 2.0) * CYCLE
    if not (flow > 0).all():
        raise ValueError("circulation too large for a positive flow matrix")
    P = flow / PI[:, None]
    assert np.allclose(P.sum(axis=1), 1.0) and np.allclose(PI @ P, PI)
    return P, float(PI[0] * P[0, 1] - PI[1] * P[1, 0])


def sample_chains(P, n, reps, rng):
    cum = np.cumsum(P, axis=1)
    state = rng.choice(3, size=reps, p=PI)
    out = np.empty((n, reps), dtype=np.int8)
    out[0] = state
    for t in range(1, n):
        u = rng.random(reps)
        state = (cum[state] < u[:, None]).sum(axis=1)
        out[t] = state
    return out


def expected_loss(P, Q):
    return float(-np.sum(PI * np.sum(P * np.log(Q), axis=1)))


def unrestricted_fit(states, alpha=1.0):
    counts = np.zeros((3, 3))
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    smoothed = counts + alpha
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def run(cfg: Config, reps: int = REPS, seed: int = SEED):
    models, _ = load_models(cfg)
    rng = np.random.default_rng(seed)
    cells, fallbacks = [], 0
    for name, j in SCENARIOS.items():
        P, actual = build(j)
        for n in LENGTHS:
            paths = sample_chains(P, n, reps, rng)
            du = np.empty(reps)
            dr = np.empty(reps)
            for k in range(reps):
                states = paths[:, k]
                du[k] = expected_loss(P, unrestricted_fit(states))
                Q, how, _ = reversible_fit(models, states)
                if how == "fallback":
                    fallbacks += 1
                dr[k] = expected_loss(P, Q)
            d = dr - du
            cells.append(dict(scenario=name, J=j, a=actual, n=n, reps=reps,
                              L_U=float(du.mean()), L_R=float(dr.mean()),
                              delta=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(reps))))
    asym = {}
    for name, j in SCENARIOS.items():
        P, _ = build(j)
        Q, _, _ = reversible_fit(models, sample_chains(P, 200000, 1, np.random.default_rng(13))[:, 0])
        asym[f"{j:.3f}"] = float(expected_loss(P, Q) - expected_loss(P, P))
    return dict(scenarios=SCENARIOS, lengths=list(LENGTHS), reps=reps, cells=cells,
                fallbacks=fallbacks, total=len(SCENARIOS) * len(LENGTHS) * reps, asymptotic=asym)
