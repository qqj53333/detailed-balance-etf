
import pandas as pd, numpy as np, glob, json
rng_seed = 20261113
B = 20; NBOOT = 5000

def load(t):
    d = sorted(glob.glob(rf"F:\MKV\_work\runs\{t}\run_*"))[-1]
    f = pd.read_csv(d + r"\forecasts.csv")
    return f[f.split == "final_test"].reset_index(drop=True)

def components(df, model):
    p1 = df[f"p_{model}_down"].to_numpy(); p2 = df[f"p_{model}_neutral"].to_numpy(); p3 = df[f"p_{model}_up"].to_numpy()
    m = p2; a = p3/(p1+p3)
    y = df["target_state"].to_numpy()   # 0 down, 1 neutral, 2 up
    z = (y == 1).astype(float); u = (y == 2).astype(float)
    A = -(z*np.log(m) + (1-z)*np.log1p(-m))
    Bc = -(1-z)*(u*np.log(a) + (1-u)*np.log1p(-a))
    return A, Bc

def block_ci(series, B=20, nboot=NBOOT, seed=rng_seed):
    x = np.asarray(series, float); n = x.size
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n/B)); out = np.empty(nboot)
    for i in range(nboot):
        starts = rng.integers(0, n-B+1, nb)
        idx = (starts[:,None] + np.arange(B)[None,:]).ravel()[:n]
        out[i] = x[idx].mean()
    return np.percentile(out, [2.5, 97.5])

rows = []
for t in ["SPY","QQQ","IWM"]:
    df = load(t)
    comp = {mo: components(df, mo) for mo in ("iid","markov","reversible")}
    tot = {mo: comp[mo][0] + comp[mo][1] for mo in comp}
    for mo in ("iid","markov","reversible"):
        assert np.allclose(tot[mo].mean(), -np.log(np.clip(
            df[[f"p_{mo}_down",f"p_{mo}_neutral",f"p_{mo}_up"]].to_numpy()[np.arange(len(df)), df.target_state.to_numpy()],1e-15,1)).mean())
    for ref in ("iid",):
        for mo in ("markov","reversible"):
            for k, lab in ((0,"middle_vs_outer"),(1,"high_vs_low_conditional")):
                diff = comp[mo][k] - comp[ref][k]
                lo, hi = block_ci(diff)
                rows.append(dict(asset=t, model=mo, component=lab, mean_loss=comp[mo][k].mean(),
                                 diff_vs_iid=diff.mean(), ci_lo=lo, ci_hi=hi))
            td = tot[mo]-tot[ref]; lo,hi = block_ci(td)
            rows.append(dict(asset=t, model=mo, component="total", mean_loss=tot[mo].mean(),
                             diff_vs_iid=td.mean(), ci_lo=lo, ci_hi=hi))
res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(6).to_string(index=False))
