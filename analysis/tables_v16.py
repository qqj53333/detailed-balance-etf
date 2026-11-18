"""Generate the v16 tables from revision_audit/v16_results/*.json.

Usage: python -I analysis/tables_v16.py --results revision_audit/v16_results --out source/tables
"""
from __future__ import annotations
import argparse, json, os

ASSETS = ("SPY", "QQQ", "IWM")
NOTE = r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize "


def sgn(x, nd):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$").replace("+", "$+$")


def ci(v, scale, nd):
    m, lo, hi = (x * scale for x in v)
    return f"{sgn(m, nd)} [{sgn(lo, nd)}, {sgn(hi, nd)}]"


def layers_table(L):
    rows = []
    names = (("vol2-iid", "Persistence", r"$\mathrm{V}-\mathrm{IID}$"),
             ("m3-vol2", "Middle information", r"$\mathrm{M}-\mathrm{V}$"),
             ("U-m3", "Direction", r"$\mathrm{U}-\mathrm{M}$"),
             ("U-iid", "Total", r"$\mathrm{U}-\mathrm{IID}$"))
    for i, t in enumerate(ASSETS):
        v = L[f"primary:{t}"]
        for key, label, short in names:
            share = 100 * v[key]["ll"][0] / v["U-iid"]["ll"][0]
            rows.append(f"{t} & {label} & {short} & {ci(v[key]['ll'], 1e3, 2)} & {share:.0f} & {ci(v[key]['net'], 1e4, 5)}\\\\".replace(f"& {share:.0f} &", "& " + f"{share:.0f}".replace("-", "$-$") + " &"))
        if i < 2:
            rows.append(r"\midrule")
    tilt = ", ".join(f"{L[f'primary:{t}']['max_abs_tilt_vol2']:.1e}".replace("e-0", r"\times10^{-") + "}" for t in ASSETS)
    return (r"""\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{4pt}
\caption{Exact benchmark layers of the unrestricted-minus-IID comparison}\label{tab:layers}
\footnotesize\begin{tabular}{lllrrr}\toprule
Asset & Layer & Contrast & Log loss ($10^{-3}$ nats) & Share (\%) & Net return (bps/day)\\
 & & & [95\% CI] & & [95\% CI]\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
""" + NOTE + r"""V is the two-state middle-versus-outer chain, M uses the unrestricted middle probability, and both use the marginal upper share $\bar q_t$ (Section~\ref{sec:benchmarks}). The first three rows of each asset add exactly to the fourth, for log loss and for net return. Share is the layer's fraction of the total log-loss difference. Net returns use 5 bps per side. Intervals are pointwise percentile intervals from 5,000 paired moving blocks of length 20, computed on the saved forecasts without refitting. The largest absolute deviation of the V allocation from one-half is $""" + tilt.replace(", ", "$, $") + r"""$ for SPY, QQQ and IWM. V and M were specified after the primary results were known.}
\end{table}
""")


def rules_table(L):
    rows = []
    label = {"primary": "Primary", "fixed": "Fixed", "roll2520": "Roll 2,520", "roll1260": "Roll 1,260"}
    for i, t in enumerate(ASSETS):
        for rule in ("primary", "fixed", "roll2520", "roll1260"):
            v = L[f"{rule}:{t}"]; d = v["U-m3"]
            share = 100 * v["vol2-iid"]["ll"][0] / v["U-iid"]["ll"][0]
            rows.append(f"{t} & {label[rule]} & {share:.0f} & {v['mean_abs_tilt']['vol2']:.5f} & {ci(d['B'], 1e3, 2)} & {ci(d['Bw'], 1e3, 2)} & {ci(d['net'], 1e4, 3)}\\\\")
        if i < 2:
            rows.append(r"\midrule")
    return (r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{2pt}
\caption{Benchmark layers under the threshold and window rules}\label{tab:layerrules}
\begin{tabular}{llrrrrr}\toprule
Asset & Rule & V share & $|w^{\mathrm V}-\tfrac12|$ & Direction & Weighted direction & Direction net\\
 & & (\%) & & ($10^{-3}$ nats) & ($10^{-3}$) & (bps/day)\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
""" + NOTE + r"""V share is $(\bar\ell^{\mathrm V}-\bar\ell^{\mathrm{IID}})/(\bar\ell^{\mathrm U}-\bar\ell^{\mathrm{IID}})$ in percent. The second column is the mean of $|w^{\mathrm V}_t-\tfrac12|$. The direction column, $\mathrm{U}-\mathrm{M}$, is the paired difference in the conditional upper-versus-lower term of Equation~\eqref{eq:scorefactor}; it equals the total $\mathrm{U}-\mathrm{M}$ log-loss difference. The weighted column multiplies that term on each outer-tercile day by $|r_{t+1}|$ divided by the mean of $|r_{t+1}|$ over outer-tercile days; it is a magnitude-weighted diagnostic, not a proper score for the tercile event. Net returns use 5 bps per side. Each rule labels history with its own cut points, as in Table~\ref{tab:sensitivity}. Under fixed cut points the marginal upper share departs from one-half, so V, M and IID all carry a common tilt. Intervals are pointwise percentile intervals from 5,000 paired moving blocks of length 20 without refitting; rows share sessions and assets.}
\end{table}
""")


def restriction_table(C, L):
    rows = []
    for t in ASSETS:
        c = C[t]; ru = L[f"primary:{t}"]["R-U"]["ll"]
        rows.append(f"{t} & {sgn(c['mean_circulation_pp'], 2)} & {1e3 * c['mean_D']:.3f} & {1e3 * c['mean_inv2n']:.3f} & {1e3 * c['fitted_reference']:.3f} & {sgn(c['frozen_circulation_pp'], 2)} & {1e3 * c['frozen_reference']:.3f} & {ci(ru, 1e3, 3)}\\\\")
    sim = []
    lab = {"0.0": "Reversible truth ($j=0$)", "-0.007": "Weak circulation ($j=-0.7$)", "-0.014": "Fitted magnitude ($j=-1.4$)"}
    for j, s in C["simulation"].items():
        cells = " & ".join(f"{sgn(p, 3)} / {sgn(r, 3)}" for p, r in zip(s["predicted"], s["reported"]))
        nstar = "---" if s["n_star"] is None or s["n_star"] > 1e9 else f"{s['n_star']:,.0f}"
        sim.append(f"{lab[j]} & {nstar} & {cells}\\\\")
    zmax = max(abs(z) for s in C["simulation"].values() for z in s["z"])
    return (r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}
\caption{Closed-form reference for the cost of detailed balance}\label{tab:restriction}
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrr}\toprule
\multicolumn{8}{l}{\emph{Panel A. Final-period reference values ($10^{-3}$ nats unless stated)}}\\
Asset & Mean $j_t$ (pp) & Mean $D_t$ & Mean $1/(2n_t)$ & Fitted reference & Frozen $j$ (pp) & Frozen reference & Observed $\mathrm{R}-\mathrm{U}$ [95\% CI]\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}}

\medskip
\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrr}\toprule
\multicolumn{7}{l}{\emph{Panel B. Controlled simulation: closed form / Monte Carlo ($10^{-3}$ nats)}}\\
Process & $n^\ast$ & $n=250$ & $n=500$ & $n=1{,}260$ & $n=2{,}520$ & $n=5{,}000$\\\midrule
""" + "\n".join(sim) + r"""
\bottomrule\end{tabular}}
""" + NOTE + r"""$D_t=\sum_{ij}F_{ij}\log(F_{ij}/S_{ij})$ with $F=\operatorname{diag}(\pi)\hat P_t$ and $S=(F+F^\top)/2$, computed from the smoothed unrestricted matrix at each origin; $n_t$ is the number of historical transitions. The fitted reference averages $D_t-1/(2n_t)$ over origins and uses matrices that incorporate earlier final-period outcomes. The frozen reference uses only the matrix of the first final-period origin (history through 31 December 2014) and the mean of $1/(2n_t)$. Both are first-order references under a stationary chain, not tests, and neither models the updating cut points. Observed values and intervals repeat Table~\ref{tab:intervals}. In Panel B, $n^\ast=1/(2D)$ is the approximate crossover length, and Monte Carlo values repeat Table~\ref{tab:simulation}; the largest absolute difference is """ + f"{zmax:.2f}" + r""" Monte Carlo standard errors.}
\end{table}
""")


def rollfreq_table(L):
    rows = []
    for i, t in enumerate(ASSETS):
        v = L[f"primary:{t}"]["rolling_frequency"]
        for W in ("252", "504", "1260"):
            x = v[W]
            rows.append(f"{t} & {int(W):,} & {ci(x['roll-iid'], 1e3, 2)} & {ci(x['U-roll'], 1e3, 2)} & {sgn(1e3 * x['2015-2018']['U-roll'][0], 2)} & {sgn(1e3 * x['2019-2022']['U-roll'][0], 2)} & {sgn(1e3 * x['2023-2025']['U-roll'][0], 2)}\\\\")
        if i < 2:
            rows.append(r"\midrule")
    return (r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}
\caption{Rolling class-frequency forecasts under the primary cut points}\label{tab:rollfreq}
\begin{tabular}{lrrrrrr}\toprule
Asset & Window & $\mathrm{Roll}-\mathrm{IID}$ & $\mathrm{U}-\mathrm{Roll}$ & \multicolumn{3}{c}{$\mathrm{U}-\mathrm{Roll}$ by segment}\\
 & (sessions) & [95\% CI] & [95\% CI] & 2015--2018 & 2019--2022 & 2023--2025\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
""" + NOTE + r"""Log-loss differences in $10^{-3}$ nats; negative values favour the first model. The rolling forecast uses the smoothed frequencies of the most recent $W$ historical labels, relabelled with the cut points of the primary rule at each origin, and no state dependence. Window lengths were chosen after the primary results were known. Intervals are pointwise percentile intervals from 5,000 paired moving blocks of length 20; segment intervals are in the replication results.}
\end{table}
""")


def statemeans_table(L):
    rows = []
    name = {"0": "Lower", "1": "Middle", "2": "Upper"}
    for i, t in enumerate(ASSETS):
        v = L[f"primary:{t}"]
        for s in ("0", "1", "2"):
            x = v["state_mean_return_bp"][s]
            w = 100 * v["state_weights_U"][s]
            rows.append(f"{t} & {name[s]} & {w:.3f} & {sgn(x['mean'], 2)} & [{sgn(x['ci'][0], 2)}, {sgn(x['ci'][1], 2)}] & [{sgn(x['simultaneous_ci'][0], 2)}, {sgn(x['simultaneous_ci'][1], 2)}]\\\\")
        if i < 2:
            rows.append(r"\midrule")
    return (r"""\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{5pt}
\caption{Unrestricted exposure and next-session return by current state}\label{tab:statemeans}
\begin{tabular}{llrrrr}\toprule
Asset & Current state & Mean $w^{\mathrm U}$ (\%) & Mean $r_{t+1}$ (bps) & Pointwise 95\% CI & Simultaneous 95\% CI\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}
""" + NOTE + r"""Final-period descriptive summaries by the state of the origin session under the primary rule. Intervals resample sessions with the common moving blocks and recompute each conditional mean as a ratio of resampled sums; the simultaneous interval uses the single-step max-$|t|$ critical value """ + f"{L['state_mean_critical_value']:.3f}" + r""" over the nine cells. These are conditional averages over the evaluation period, not a trading rule; selecting states from them and reporting returns in the same period would be in-sample.}
\end{table}
""")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results"); ap.add_argument("--out")
    a = ap.parse_args()
    L = json.load(open(os.path.join(a.results, "layers.json")))
    C = json.load(open(os.path.join(a.results, "restriction_cost.json")))
    for name, text in (("tab_layers", layers_table(L)), ("tab_layerrules", rules_table(L)),
                       ("tab_restriction", restriction_table(C, L)), ("tab_rollfreq", rollfreq_table(L)),
                       ("tab_statemeans", statemeans_table(L))):
        with open(os.path.join(a.out, name + ".tex"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        print("wrote", name)


if __name__ == "__main__":
    main()
