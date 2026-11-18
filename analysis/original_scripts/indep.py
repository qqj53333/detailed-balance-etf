
import pandas as pd, numpy as np, json
from scipy.optimize import minimize
from scipy.stats import dirichlet

PAIRS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def nll(masses, c, r):
    m = np.asarray(masses, float)
    if np.any(m <= 0): return 1e12
    flow = np.zeros((3,3)); pi = np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i == j: flow[i,i] = m[k]; pi[i] += m[k]
        else: flow[i,j] = m[k]/2; flow[j,i] = m[k]/2; pi[i] += m[k]/2; pi[j] += m[k]/2
    return float(-np.sum(c*np.log(m)) + np.sum(r*np.log(pi)))

def fit_counts(N, alpha=1.0, x0=None, method="SLSQP"):
    s = N + alpha
    c = np.array([s[i,i] if i==j else s[i,j]+s[j,i] for (i,j) in PAIRS])
    r = s.sum(1)
    if x0 is None: x0 = np.full(6, 1/6)
    res = minimize(nll, x0, args=(c, r), method=method,
                   bounds=[(1e-12, None)]*6,
                   constraints=[{"type":"eq","fun":lambda m: m.sum()-1, "jac": lambda m: np.ones(6)}],
                   options={"maxiter":500, "ftol":1e-14})
    return res.x, float(res.fun), bool(res.success)

def build_P(m):
    flow = np.zeros((3,3)); pi = np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i == j: flow[i,i] = m[k]; pi[i] += m[k]
        else: flow[i,j] = m[k]/2; flow[j,i] = m[k]/2; pi[i] += m[k]/2; pi[j] += m[k]/2
    return flow/pi[:,None], pi

def counts_from_states(st):
    C = np.zeros((3,3))
    np.add.at(C, (st[:-1], st[1:]), 1.0)
    return C

def pkg_nll(m, N, alpha=1.0):
    s = N + alpha
    c = np.array([s[i,i] if i==j else s[i,j]+s[j,i] for (i,j) in PAIRS])
    r = s.sum(1)
    return nll(m, c, r)

rng = np.random.default_rng(20261115)
out = []
for t in ("SPY","QQQ","IWM"):
    m = pd.read_csv(rf"F:\MKV\_work\data\{t}_matrices.csv", parse_dates=["signal_date"])
    pr = pd.read_csv(rf"F:\MKV\_work\data\{t}_adj.csv", parse_dates=["Date"]).set_index("Date")
    ret = (pr["Close"]/pr["Open"]-1.0).to_numpy()
    dates = pr.index
    ft = m[m.split=="final_test"].reset_index(drop=True)
    gaps, mism, ll_gap = [], 0, []
    worst = 0
    for n, row in ft.iterrows():
        o = dates.get_loc(row.signal_date)
        tr = ret[:o+1]
        lo, hi = np.quantile(tr, [1/3, 2/3])
        st = np.where(tr<=lo,0,np.where(tr<=hi,1,2))
        N = counts_from_states(st)
        m_pkg = np.array([row[f"R{i}{j}"] for i in range(1,4) for j in range(1,4)])
        # package flow masses implied by its P and pi
        pi_pkg = np.array([row.pi_r0,row.pi_r1,row.pi_r2])
        flow_pkg = pi_pkg[:,None]*m_pkg.reshape(3,3)
        m_pkg_masses = np.array([flow_pkg[0,0],flow_pkg[1,1],flow_pkg[2,2],
                                 2*flow_pkg[0,1],2*flow_pkg[0,2],2*flow_pkg[1,2]])
        f_pkg = pkg_nll(m_pkg_masses, N)
        m_ind, f_ind, ok = fit_counts(N)
        P_ind, pi_ind = build_P(m_ind)
        gap = f_pkg - f_ind
        gaps.append(gap)
        if gap > 1e-9: mism += 1; worst = max(worst, gap)
        ll_gap.append(np.abs(P_ind - m_pkg.reshape(3,3)).max())
    gaps = np.array(gaps)
    print(f"{t}: n={len(gaps)} pkg NLL - indep NLL: max {gaps.max():.2e} mean {gaps.mean():.2e} | pkg worse by >1e-9 at {mism} origins"
          f" | max|P diff| {np.max(ll_gap):.2e}")
    out.append((t, float(gaps.max()), int(mism), float(np.max(ll_gap))))
json.dump(out, open(r"F:\MKV\_work\indep_check.json","w"))
