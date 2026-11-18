
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})
fig, ax = plt.subplots(figsize=(6.4, 2.9))
styles = {"SPY": ("#1f4e79", "-"), "QQQ": ("#b35c1e", "--"), "IWM": ("#2e6b3e", "-.")}
stats = {}
for t in ("SPY","QQQ","IWM"):
    m = pd.read_csv(rf"F:\MKV\_work\data\{t}_matrices.csv", parse_dates=["signal_date"])
    ft = m[m.split=="final_test"]
    a = 100*ft.circulation.to_numpy(); stats[t] = (a.mean(), a.min(), a.max())
    c, ls = styles[t]
    ax.plot(ft.signal_date, a, lw=1.0, color=c, ls=ls, label=f"{t} (mean {a.mean():+.2f})")
ax.axhline(0, color="black", lw=0.9, ls=(0,(4,3)))
ax.annotate("detailed balance", xy=(pd.Timestamp("2016-01-01"), 0), xytext=(0, 4),
            textcoords="offset points", fontsize=7.5)
ax.set_ylabel("Circulation $a_t$ (percentage points/day)")
ax.set_xlabel("Target session")
ax.set_ylim(-1.92, 0.24)
ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=3)
fig.tight_layout()
fig.savefig(r"F:\MKV\_work\figures\figure5.pdf", bbox_inches="tight")
fig.savefig(r"F:\MKV\_work\figures\figure5.png", bbox_inches="tight")
for t,v in stats.items(): print(t, [round(x,3) for x in v])
