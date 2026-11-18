
import pandas as pd, numpy as np, glob
rows=[]
for t in ("SPY","QQQ","IWM"):
    d = sorted(glob.glob(rf"F:\MKV\_work\runs\{t}\run_*"))[-1]
    f = pd.read_csv(d + r"\forecasts.csv"); f=f[f.split=="final_test"]
    for mo in ("iid","markov","reversible"):
        P = f[[f"p_{mo}_down",f"p_{mo}_neutral",f"p_{mo}_up"]].to_numpy()
        edges = np.linspace(0,1,11)
        for k,nm in enumerate(("Down","Neutral","Up")):
            b = np.digitize(P[:,k], edges[1:-1], right=False)
            cnt = np.bincount(b, minlength=10)
            rows.append(dict(asset=t, model=mo, state=nm, empty=int((cnt==0).sum()),
                             min_count=int(cnt[cnt>0].min()), max_count=int(cnt.max())))
df = pd.DataFrame(rows)
print(df.groupby(["asset","model"]).agg(empty_bins=("empty","sum"), min_count=("min_count","min"), max_count=("max_count","max")).to_string())
