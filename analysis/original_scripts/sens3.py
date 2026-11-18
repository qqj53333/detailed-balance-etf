
import sys, os
sys.path.insert(0, r"F:\MKV\_work\code_v5\markov-reversibility-code\src")
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from markov_research.models import iid_probabilities, markov_matrix, reversible_matrix
from markov_research.states import fit_thresholds, classify_returns, STATE_NAMES

PAIRS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def _prep(states, alpha=1.0):
    C = np.zeros((3,3)); np.add.at(C,(states[:-1], states[1:]),1.0)
    sc = C + alpha
    c = np.array([sc[i,i] if i==j else sc[i,j]+sc[j,i] for (i,j) in PAIRS])
    r = sc.sum(1)
    logits = np.log(c/c.sum())
    return c, r, logits[:-1]-logits[-1]

def _flow(x):
    m = softmax(np.concatenate([x,[0.0]])); F = np.zeros((3,3))
    for k,(i,j) in enumerate(PAIRS):
        if i==j: F[i,i]=m[k]
        else: F[i,j]=m[k]/2; F[j,i]=m[k]/2
    return m, F, F.sum(1)

def _nll(x, c, r):
    m,_,pi = _flow(x)
    return float(-np.sum(c*np.log(m)) + np.sum(r*np.log(pi)))

def _grad(x, c, r):
    m,_,pi = _flow(x)
    gm = -c/m
    for k,(i,j) in enumerate(PAIRS):
        if i==j: gm[k] += r[i]/pi[i]
        else: gm[k] += 0.5*(r[i]/pi[i] + r[j]/pi[j])
    g = m*(gm - float(np.dot(m,gm)))
    return g[:-1]

def fit_rev(states, alpha=1.0):
    try:
        P, pi, resid, conv, gn, nit = reversible_matrix(states, alpha=alpha)
        return P, pi, resid, "package", 0.0
    except RuntimeError:
        c, r, x0 = _prep(states, alpha)
        res = minimize(_nll, x0, args=(c,r), method="BFGS", jac=_grad,
                       options={"maxiter":800, "gtol":1e-8})
        gnorm = float(np.linalg.norm(_grad(res.x,c,r), np.inf))
        best = (res.fun, res.x)
        if gnorm > 1e-5:
            from scipy.optimize import minimize as mz
            r2 = mz(lambda x: _nll(x,c,r), res.x, method="Nelder-Mead",
                    options={"maxiter":4000,"xatol":1e-10,"fatol":1e-12})
            g2 = float(np.linalg.norm(_grad(r2.x,c,r), np.inf))
            if r2.fun < best[0]: best = (r2.fun, r2.x); gnorm = g2
        m, F, pi = _flow(best[1]); P = F/pi[:,None]
        resid = float(np.max(np.abs(pi[:,None]*P - (pi[:,None]*P).T)))
        return P, pi, resid, "fallback", gnorm

SPECS = {
    "primary":   dict(mode="expanding", window=None,  common_start=None),
    "fixed":     dict(mode="fixed",     window=None,  common_start=None),
    "roll2520":  dict(mode="rolling",   window=2520,  common_start=None),
    "roll1260":  dict(mode="rolling",   window=1260,  common_start=None),
    "common":    dict(mode="expanding", window=None,  common_start="2000-05-26"),
}

def run(ticker, spec, min_train=2520, alpha=1.0, out_dir=r"F:\MKV\_work\sens"):
    cfg = SPECS[spec]
    prices = pd.read_csv(rf"F:\MKV\_work\data\{ticker}_adj.csv", parse_dates=["Date"]).set_index("Date").sort_index().loc[:"2025-12-31"]
    if cfg["common_start"]: prices = prices.loc[cfg["common_start"]:]
    r = (prices["Close"]/prices["Open"]-1.0).to_numpy(float)
    dates = prices.index; n = len(prices)
    fixed_th = None; rows = []; nfb = 0; maxg = 0.0; bad = 0
    for o in range(min_train-1, n-1):
        if cfg["mode"] == "expanding": th = fit_thresholds(r[:o+1]); hist = r[:o+1]
        elif cfg["mode"] == "fixed":
            if fixed_th is None: fixed_th = fit_thresholds(r[:o+1])
            th = fixed_th; hist = r[:o+1]
        else:
            w = cfg["window"]; th = fit_thresholds(r[o+1-w:o+1]); hist = r[o+1-w:o+1]
        st = classify_returns(hist, th); cs = int(st[-1])
        p_iid = iid_probabilities(st, alpha=alpha); P = markov_matrix(st, alpha=alpha)
        R, pi_r, resid, how, gn = fit_rev(st, alpha)
        if how == "fallback":
            nfb += 1; maxg = max(maxg, gn); bad += (gn > 1e-5)
        tr = float(r[o+1]); ts = int(classify_returns(np.array([tr]), th)[0])
        d = dict(ticker=ticker, spec=spec, signal_date=dates[o], target_date=dates[o+1],
                 current_state=STATE_NAMES[cs], target_state=ts, target_intraday_return=tr, fit=how)
        for nm, p in (("iid", p_iid), ("markov", P[cs]), ("reversible", R[cs])):
            d[f"p_{nm}_down"], d[f"p_{nm}_neutral"], d[f"p_{nm}_up"] = map(float, p)
        rows.append(d)
    df = pd.DataFrame(rows)
    df["split"] = np.where(df.target_date <= "2009-12-31", "development",
                  np.where(df.target_date <= "2014-12-31", "validation", "final_test"))
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(rf"{out_dir}\{spec}_{ticker}.csv", index=False, date_format="%Y-%m-%d")
    print(f"{ticker} {spec} n={len(df)} fallback={nfb} weak={bad} maxg={maxg:.1e}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--ticker"); ap.add_argument("--spec")
    a = ap.parse_args(); run(a.ticker, a.spec)
