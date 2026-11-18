
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300,
                     "font.family": "serif", "mathtext.fontset": "dejavuserif"})
styles = {"SPY": ("#1f4e79","-"), "QQQ": ("#b35c1e","--"), "IWM": ("#2e6b3e","-.")}
fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
for t in ("SPY","QQQ","IWM"):
    d = pd.read_csv(rf"F:\MKV\_work\roll252_{t}.csv", parse_dates=["date"]).set_index("date").sort_index()
    c, ls = styles[t]
    axes[0].plot(d.index, 1e3*d.ui, lw=1.0, color=c, ls=ls, label=t)
    axes[1].plot(d.index, 1e3*d.ru, lw=1.0, color=c, ls=ls, label=t)
for ax, lab in zip(axes, ("Unrestricted $-$ IID", "Reversible $-$ unrestricted")):
    ax.axhline(0, color="black", lw=0.8, ls=(0,(4,3)))
    ax.set_ylabel(lab + "\n(252-session mean, bps)")
axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
axes[1].set_xlabel("Target session")
fig.tight_layout()
fig.savefig(r"F:\MKV\_work\figures\figure6_rolling.pdf", bbox_inches="tight")
fig.savefig(r"F:\MKV\_work\figures\figure6_rolling.png", bbox_inches="tight")
print("ok")
