
import pandas as pd, numpy as np
pd.set_option("display.width", 220)
for t in ("SPY","QQQ","IWM"):
    m = pd.read_csv(rf"F:\MKV\_work\data\{t}_matrices.csv", parse_dates=["signal_date","target_date"])
    ft = m[m.split=="final_test"].copy()
    a = ft.circulation.to_numpy()
    Pm = ft[[f"P{i}{j}" for i in (1,2,3) for j in (1,2,3)]].to_numpy().reshape(-1,3,3)
    Rm = ft[[f"R{i}{j}" for i in (1,2,3) for j in (1,2,3)]].to_numpy().reshape(-1,3,3)
    # reversible circulation (should be ~0)
    pir = ft[["pi_r0","pi_r1","pi_r2"]].to_numpy()
    a_rev = pir[:,0]*Rm[:,0,1] - pir[:,1]*Rm[:,1,0]
    # per-origin max abs off-diagonal difference
    dd = np.abs(Pm - Rm)
    mask = ~np.eye(3, dtype=bool)
    ft["a"] = a
    ft["year"] = ft.target_date.dt.year
    yr = ft.groupby("year")["a"].agg(["mean","std", lambda x: (x>0).mean()]).rename(columns={"<lambda_0>":"share_pos"})
    print(f"=== {t}: mean a {a.mean():+.5f} median {np.median(a):+.5f} sd {a.std():.5f} share>0 {(a>0).mean():.3f} "
          f"ac1 {pd.Series(a).autocorr(1):.3f} | a_rev max {np.abs(a_rev).max():.2e} | maxoff-diag diff {dd[:,mask].max():.4f} "
          f"mean off-diag diff {dd[:,mask].mean():.4f}")
    print(yr.round(4).T.to_string())
