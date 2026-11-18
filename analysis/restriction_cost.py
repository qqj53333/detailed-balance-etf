"""v16: closed-form reference for the cost of imposing detailed balance.

For a positive transition matrix P with stationary distribution pi, let
F = diag(pi) P and S = (F + F^T)/2. The reversible KL projection is
P*_{ij} = S_{ij}/pi_i and its per-step cost is D = sum_ij F_ij log(F_ij/S_ij).
The first-order excess risk of an extra free parameter is 1/(2n). This script
(i) compares D - 1/(2n) with the controlled simulation of Table A8,
(ii) evaluates D_t - 1/(2 n_t) at every final-period origin and from the matrix
frozen at the first final-period origin (information through 31 December 2014), and (iii) checks the symmetric-count
reversible estimator against the saved reversible forecasts.

Usage: python -I restriction_cost.py --prices DIR --runs DIR --out DIR
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np, pandas as pd

ALPHA = 1.0


def stationary(P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))]); return pi / pi.sum()


def projection_cost(P):
    pi = stationary(P); F = pi[:, None] * P; S = (F + F.T) / 2
    return float((F * np.log(F / S)).sum()), float(F[0, 1] - F[1, 0])


def sim_matrix(j, d=0.35):
    off = (1 - d) / 2
    P0 = np.full((3, 3), off); np.fill_diagonal(P0, d)
    C = np.zeros((3, 3)); C[0, 1] = C[1, 2] = C[2, 0] = 1; C[1, 0] = C[2, 1] = C[0, 2] = -1
    F = P0 / 3 + (j / 2) * C
    return F / F.sum(axis=1, keepdims=True)


SIM = {0.0: ([-2.087, -1.008, -0.431, -0.214, -0.101], [0.137, 0.062, 0.027, 0.013, 0.006]),
       -0.007: ([-1.626, -0.661, -0.061, 0.132, 0.245], [0.132, 0.061, 0.026, 0.013, 0.007]),
       -0.014: ([-0.638, 0.374, 0.950, 1.165, 1.257], [0.134, 0.072, 0.023, 0.012, 0.007])}
NS = (250, 500, 1260, 2520, 5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices"); ap.add_argument("--runs"); ap.add_argument("--out")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    out = {"simulation": {}}
    for j, (vals, ses) in SIM.items():
        D, _ = projection_cost(sim_matrix(j))
        pred = [1e3 * (D - 1 / (2 * n)) for n in NS]
        out["simulation"][str(j)] = dict(D=D, n_star=(1 / (2 * D) if D > 0 else None), predicted=pred,
                                         reported=vals, z=[(v - p) / s for v, p, s in zip(vals, pred, ses)])
    for t in ("SPY", "QQQ", "IWM"):
        px = pd.read_csv(os.path.join(a.prices, f"{t}.csv"), parse_dates=["Date"]).set_index("Date").loc[:"2025-12-31"]
        r = (px.Close / px.Open - 1).to_numpy(float); pos = {d: i for i, d in enumerate(px.index)}
        f = pd.read_csv(glob.glob(os.path.join(a.runs, t, "run_*", "forecasts.csv"))[0], parse_dates=["signal_date"])
        fin = f[f.split == "final_test"].reset_index(drop=True); val = fin.iloc[0]

        def counts(o):
            hist = r[: o + 1]; lo, hi = np.quantile(hist, [1 / 3, 2 / 3])
            s = np.where(hist <= lo, 0, np.where(hist <= hi, 1, 2))
            C = np.zeros((3, 3)); np.add.at(C, (s[:-1], s[1:]), 1.0)
            return C + ALPHA, s[-1], len(s) - 1

        Cf, _, _ = counts(pos[val.signal_date])
        Dfrozen, jfrozen = projection_cost(Cf / Cf.sum(1, keepdims=True))
        Ds, js, inv, symerr, relerr = [], [], [], 0.0, 0.0
        for _, row in fin.iterrows():
            C, cur, n = counts(pos[row.signal_date])
            P = C / C.sum(1, keepdims=True); D, jj = projection_cost(P)
            Ds.append(D); js.append(jj); inv.append(1 / (2 * n))
            Psym = (C + C.T) / (C.sum(1) + C.sum(0))[:, None]
            saved = np.array([row.p_reversible_down, row.p_reversible_neutral, row.p_reversible_up])
            symerr = max(symerr, float(np.abs(Psym[cur] - saved).max()))
            relerr = max(relerr, float(np.abs(C.sum(1) - C.sum(0)).max()))
        Ds, inv = np.array(Ds), np.array(inv)
        ll = lambda c: -np.log(fin[c].to_numpy())
        y = fin.target_state.to_numpy()
        pick = lambda m: np.log(fin[[f"p_{m}_down", f"p_{m}_neutral", f"p_{m}_up"]].to_numpy()[np.arange(len(y)), y])
        actual = float((pick("markov") - pick("reversible")).mean())
        out[t] = dict(mean_D=float(Ds.mean()), mean_inv2n=float(inv.mean()),
                      fitted_reference=float((Ds - inv).mean()),
                      frozen_signal_date=str(val.signal_date.date()), frozen_D=Dfrozen,
                      frozen_circulation_pp=100 * jfrozen,
                      frozen_reference=float(Dfrozen - inv.mean()), actual_R_minus_U=actual,
                      mean_circulation_pp=100 * float(np.mean(js)),
                      symmetric_estimator_max_abs_diff=symerr, max_row_minus_col_count=relerr)
    with open(os.path.join(a.out, "restriction_cost.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
