
import pandas as pd, numpy as np, glob, json
def load(t):
    d = sorted(glob.glob(rf"F:\MKV\_work\runs\{t}\run_*"))[-1]
    f = pd.read_csv(d + r"\forecasts.csv"); f=f[f.split=="final_test"].reset_index(drop=True)
    y = f.target_state.to_numpy()
    ll = {mo: -np.log(f[[f"p_{mo}_down",f"p_{mo}_neutral",f"p_{mo}_up"]].to_numpy()[np.arange(len(f)),y]) for mo in ("iid","markov","reversible")}
    return pd.DataFrame(dict(date=pd.to_datetime(f.target_date), ui=ll["markov"]-ll["iid"], ru=ll["reversible"]-ll["markov"]))
def ci(x, B=5000, L=20, seed=20261113):
    rng=np.random.default_rng(seed); n=x.size; nb=int(np.ceil(n/L))
    idx=(rng.integers(0,n-L+1,(B,nb))[:,:,None]+np.arange(L)[None,None,:]).reshape(B,-1)[:,:n]
    m=x[idx].mean(1); return np.percentile(m,[2.5,97.5])
segs = [("2015-2018","2015-01-01","2018-12-31"),("2019-2022","2019-01-01","2022-12-31"),("2023-2025","2023-01-01","2025-12-31")]
out = {}
for t in ("SPY","QQQ","IWM"):
    d = load(t); rows=[]
    for nm,a,b in segs:
        s = d[(d.date>=a)&(d.date<=b)]
        a1=ci(s.ui.to_numpy()); a2=ci(s.ru.to_numpy())
        rows.append(dict(seg=nm, n=len(s), ui=s.ui.mean(), ui_lo=a1[0], ui_hi=a1[1], ru=s.ru.mean(), ru_lo=a2[0], ru_hi=a2[1]))
    out[t]=rows
    print(t)
    for r in rows: print(f"  {r['seg']} n={r['n']:5d} U-IID {r['ui']:+.5f} [{r['ui_lo']:+.5f},{r['ui_hi']:+.5f}]  R-U {r['ru']:+.5f} [{r['ru_lo']:+.5f},{r['ru_hi']:+.5f}]")
    # rolling 252
    d = d.set_index("date").sort_index()
    roll = d.rolling(252).mean()
    roll.to_csv(rf"F:\MKV\_work\roll252_{t}.csv")
json.dump(out, open(r"F:\MKV\_work\segments.json","w"), indent=2)
