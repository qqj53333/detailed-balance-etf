
import sys
sys.path.insert(0, r"F:\MKV\_work\code_v5\markov-reversibility-code\src")
import numpy as np, json
from scipy.optimize import minimize
from markov_research.models import reversible_matrix

PI = np.ones(3)/3.0
diag = 0.35/3.0; off = (1.0/3.0 - diag)/2.0
S = np.array([[diag,off,off],[off,diag,off],[off,off,diag]])
S = S * ((1.0/3.0) / S.sum(axis=1, keepdims=True))
C = np.array([[0.0,1.0,-1.0],[-1.0,0.0,1.0],[1.0,-1.0,0.0]])
PAIRS = ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))

def build(J):
    F = S + (J/2.0)*C
    P = F/PI[:,None]
    return P, float(PI[0]*P[0,1]-PI[1]*P[1,0])

def sample_chain(P, n, reps, rng):
    cum = np.cumsum(P, axis=1)
    st = rng.choice(3, size=reps, p=PI)
    out = np.empty((n, reps), dtype=np.int8)
    out[0] = st
    for t in range(1, n):
        u = rng.random(reps)
        st = (cum[st] < u[:,None]).sum(axis=1)
        out[t] = st
    return out

def expl(P, Q):
    return float(-np.sum(PI*np.sum(P*np.log(Q), axis=1)))

def fit_u(st, alpha=1.0):
    Cm = np.zeros((3,3)); np.add.at(Cm,(st[:-1],st[1:]),1.0)
    Sm = Cm + alpha
    return Sm/Sm.sum(1, keepdims=True)

def _terms(st, alpha):
    Cm = np.zeros((3,3)); np.add.at(Cm,(st[:-1],st[1:]),1.0)
    sc = Cm + alpha
    c = np.array([sc[i,i] if i==j else sc[i,j]+sc[j,i] for (i,j) in PAIRS])
    return c, sc.sum(1)

def _nll(m, c, r):
    F = np.zeros((3,3)); pi = np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i==j: F[i,i]=m[k]; pi[i]+=m[k]
        else: F[i,j]=m[k]/2; F[j,i]=m[k]/2; pi[i]+=m[k]/2; pi[j]+=m[k]/2
    return float(-np.sum(c*np.log(m)) + np.sum(r*np.log(pi)))

def fit_slsqp(st, alpha=1.0):
    c, r = _terms(st, alpha)
    cons = [{"type":"eq","fun":lambda m: m.sum()-1,"jac":lambda m: np.ones(6)}]
    best = None
    for start in (c/c.sum(), np.full(6,1/6)):
        res = minimize(_nll, start, args=(c,r), method="SLSQP", bounds=[(1e-12,None)]*6,
                       constraints=cons, options={"maxiter":1000,"ftol":1e-14})
        if best is None or res.fun < best.fun: best = res
    m = best.x; F = np.zeros((3,3)); pi = np.zeros(3)
    for k,(i,j) in enumerate(PAIRS):
        if i==j: F[i,i]=m[k]; pi[i]+=m[k]
        else: F[i,j]=m[k]/2; F[j,i]=m[k]/2; pi[i]+=m[k]/2; pi[j]+=m[k]/2
    return F/pi[:,None]

rng = np.random.default_rng(20261115)
scenarios = {"reversible":0.0, "weak_irreversible":-0.007, "empirical_irreversible":-0.014}
lengths = [250, 500, 1260, 2520, 5000]
REPS = 500
res = []; fallbacks = 0
for name, J in scenarios.items():
    P, a = build(J)
    for n in lengths:
        paths = sample_chain(P, n, REPS, rng)
        dU = np.empty(REPS); dR = np.empty(REPS)
        for k in range(REPS):
            st = paths[:, k]
            dU[k] = expl(P, fit_u(st))
            try:
                dR[k] = expl(P, reversible_matrix(st)[0])
            except RuntimeError:
                dR[k] = expl(P, fit_slsqp(st)); fallbacks += 1
        d = dR - dU
        res.append(dict(scenario=name, J=J, a=a, n=n, reps=REPS, L_U=float(dU.mean()), L_R=float(dR.mean()),
                        delta=float(d.mean()), se=float(d.std(ddof=1)/np.sqrt(REPS))))
        print(f"{name:24s} n={n:5d} L_U={dU.mean():.6f} delta={d.mean():+.6f} se={res[-1]['se']:.6f}", flush=True)
res.append(dict(fallbacks=fallbacks, total=len(scenarios)*len(lengths)*REPS))
json.dump(res, open(r"F:\MKV\_work\simulation.json","w"), indent=2)
print("fallbacks", fallbacks)
