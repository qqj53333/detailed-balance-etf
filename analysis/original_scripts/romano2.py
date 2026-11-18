
import pandas as pd, numpy as np, glob, json
def load(t):
    d = sorted(glob.glob(rf"F:\MKV\_work\runs\{t}\run_*"))[-1]
    f = pd.read_csv(d + r"\forecasts.csv"); return f[f.split=="final_test"].reset_index(drop=True)
assets = ["SPY","QQQ","IWM"]; data = {}
for t in assets:
    f = load(t); y = f.target_state.to_numpy()
    ll = {mo: -np.log(f[[f"p_{mo}_down",f"p_{mo}_neutral",f"p_{mo}_up"]].to_numpy()[np.arange(len(f)),y]) for mo in ("iid","markov","reversible")}
    data[t] = dict(ui=ll["markov"]-ll["iid"], ru=ll["reversible"]-ll["markov"])
D = np.column_stack([data[t][k] for t in assets for k in ("ui","ru")])
names = [f"{t}:{k}" for t in assets for k in ("ui","ru")]
n = len(D); B, L, seed = 5000, 20, 20261113
rng = np.random.default_rng(seed); nb = int(np.ceil(n/L))
idx = (rng.integers(0, n-L+1, (B, nb))[:,:,None] + np.arange(L)[None,None,:]).reshape(B,-1)[:,:n]
means = np.array([D[idx[b]].mean(0) for b in range(B)])
obs = D.mean(0); se = means.std(0, ddof=1); t_obs = obs/se; t_star = (means-obs)/se

# single-step max-|t| p
maxabs = np.max(np.abs(t_star), axis=1)
p_single = np.array([(1+np.sum(maxabs >= abs(t)))/(B+1) for t in t_obs])
# stepdown
order = np.argsort(-np.abs(t_obs)); p_step = np.empty(6)
for k,c in enumerate(order):
    rem = order[k:]; stat = np.max(np.abs(t_star[:,rem]), axis=1)
    p_step[c] = (1+np.sum(stat >= abs(t_obs[c])))/(B+1)
for k in range(1,6): p_step[order[k]] = max(p_step[order[k]], p_step[order[k-1]])
# per-contrast bootstrap p (two-sided)
p_raw = np.array([(1+np.sum(np.abs(t_star[:,j]) >= abs(t_obs[j])))/(B+1) for j in range(6)])
# Holm on raw
order_h = np.argsort(p_raw); p_holm = np.empty(6)
run = 0.0
for k,c in enumerate(order_h):
    run = max(run, min(1.0, (6-k)*p_raw[c])); p_holm[c] = run
q95 = np.quantile(maxabs, 0.95)
print(f"{'contrast':10s} {'mean':>11s} {'t':>6s} {'p_raw':>7s} {'Holm':>7s} {'stepdown':>9s} {'single':>7s}")
for j,nm in enumerate(names):
    print(f"{nm:10s} {obs[j]:+11.6f} {t_obs[j]:6.2f} {p_raw[j]:7.3f} {p_holm[j]:7.3f} {p_step[j]:9.4f} {p_single[j]:7.3f}")
print("q95", round(float(q95),4))
json.dump(dict(names=names, mean=obs.tolist(), se=se.tolist(), t=t_obs.tolist(), p_raw=p_raw.tolist(),
               p_holm=p_holm.tolist(), p_stepdown=p_step.tolist(), p_single=p_single.tolist(), q95=float(q95),
               ci=[[float(obs[j]-q95*se[j]), float(obs[j]+q95*se[j])] for j in range(6)]),
          open(r"F:\MKV\_work\rw.json","w"), indent=2)
