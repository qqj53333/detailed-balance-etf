"""Figures 3 and 4 of the manuscript, regenerated from the analysis outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STYLE = {"SPY": ("#1f4e79", "-"), "QQQ": ("#b35c1e", "--"), "IWM": ("#2e6b3e", "-.")}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 300, "savefig.dpi": 300,
                         "font.family": "serif", "mathtext.fontset": "dejavuserif"})
    return plt


def circulation(frames: pd.DataFrame, out: Path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    for asset, (colour, style) in STYLE.items():
        d = frames[(frames.asset == asset) & (frames.split == "final_test")]
        ax.plot(d.signal_date, 100 * d.j.to_numpy(), lw=1.0, color=colour, ls=style,
                label=f"{asset} (mean {100*d.j.mean():+.2f})")
    ax.axhline(0, color="black", lw=0.9, ls=(0, (4, 3)))
    ax.annotate("detailed balance", xy=(pd.Timestamp("2016-01-01"), 0), xytext=(0, 4),
                textcoords="offset points", fontsize=7.5)
    ax.set_ylabel("Circulation $j_t$ (percentage points/day)")
    ax.set_xlabel("Signal session")
    ax.set_ylim(-1.92, 0.24)
    ax.legend(frameon=False, fontsize=7.5, loc="lower center", ncol=3)
    fig.tight_layout()
    _save(fig, out, "figure5_circulation")


def rolling(paths: dict, out: Path):
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for asset, (colour, style) in STYLE.items():
        d = paths[asset]
        axes[0].plot(d.index, 1e3 * d.ui, lw=1.0, color=colour, ls=style, label=asset)
        axes[1].plot(d.index, 1e3 * d.ru, lw=1.0, color=colour, ls=style, label=asset)
    for ax, label in zip(axes, ("Unrestricted $-$ IID", "Reversible $-$ unrestricted")):
        ax.axhline(0, color="black", lw=0.8, ls=(0, (4, 3)))
        ax.set_ylabel(label + "\n(252-session mean, $10^{-3}$ nats)")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left")
    axes[1].set_xlabel("Target session")
    fig.tight_layout()
    _save(fig, out, "figure6_rolling")


def _save(fig, out: Path, stem: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
