
import pandas as pd, numpy as np, glob
rows=[]
for t in ("SPY","QQQ","IWM"):
    d = sorted(glob.glob(rf"F:\MKV\_work\runs\{t}\run_*"))[-1]
    f = pd.read_csv(d + r"\forecasts.csv"); f=f[f.split=="final_test"].reset_index(drop=True)
    for mo in ("iid","markov","reversible"):
        p1,p2,p3 = (f[f"p_{mo}_{k}"].to_numpy() for k in ("down","neutral","up"))
        m = p2; q = p3/(p1+p3)
        dwdm = np.abs(2*q-1)/2          # |dw/dm|
        dwdq = 1-m                      # dw/dq
        rows.append(dict(asset=t, model=mo, q_mean=q.mean(), q_sd=q.std(),
                         abs2q1_mean=np.abs(2*q-1).mean(), dwdm_mean=dwdm.mean(),
                         dwdm_med=np.median(dwdm), dwdq_mean=dwdq.mean(), w_sd=((0.5+0.5*(p3-p1)).std())))
df = pd.DataFrame(rows); pd.set_option("display.width",200)
print(df.round(4).to_string(index=False))
