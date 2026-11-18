"""LaTeX table bodies generated from the analysis outputs.

Each function returns a complete table environment. Captions and footnotes are
fixed text; every number is read from the JSON written by the pipeline stages.
"""

from __future__ import annotations

ASSETS = ("SPY", "QQQ", "IWM")
RULE_ORDER = ("primary", "fixed", "roll2520", "roll1260", "common")
RULE_LABEL = {"primary": "Primary", "fixed": "Fixed cut points", "roll2520": "Rolling 2,520",
              "roll1260": "Rolling 1,260", "common": "Common start"}
MODEL_LABEL = {"iid": "IID", "markov": "Unrestricted", "reversible": "Reversible"}
SCENARIO_LABEL = {"reversible": r"Reversible truth ($j=0$)",
                  "weak_irreversible": r"Weak circulation ($j=-0.7$)",
                  "empirical_irreversible": r"Fitted magnitude ($j=-1.4$)"}


def _signed(x, digits=5):
    return (f"{x:+.{digits}f}").replace("-", "$-$").replace("+", "$+$")


def _plain(x, digits=5):
    return f"{x:.{digits}f}"


def _ci(pair, digits=5):
    return "[" + _signed(pair[0], digits) + ", " + _signed(pair[1], digits) + "]"


def _ci_plain(pair, digits=4):
    return "[" + f"{pair[0]:.{digits}f}" + ", " + f"{pair[1]:.{digits}f}" + "]"


def components(results):
    rows = []
    for asset in ASSETS:
        for model in ("U", "R"):
            row = next(r for r in results["components"] if r["asset"] == asset and r["model"] == model)
            rows.append(
                f"{asset} & {model} & {_plain(row['A'])} & {_signed(row['dA'])} & {_ci(row['dA_ci'])} & "
                f"{_plain(row['B'])} & {_signed(row['dB'])} & {_ci(row['dB_ci'])}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{7pt}",
            r"\caption{Components of the final-period log loss}\label{tab:components}",
            r"\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{llrrrrrr}\toprule",
            r" & & \multicolumn{3}{c}{Middle versus outer} & \multicolumn{3}{c}{Upper versus lower}\\ \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"Asset & Model & $A$ & $\Delta A$ & 95\% CI & $B$ & $\Delta B$ & 95\% CI\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize $A$ is $-\log m_t$ or $-\log(1-m_t)$ according to whether the target is the middle tercile; $B$ is the upper-versus-lower term of Equation~\eqref{eq:scorefactor}; it is zero on middle-tercile days and is averaged over all evaluation days, so it is an outer-state-weighted conditional loss, not the mean conditional loss on outer days. $A+B$ equals the total log loss of Table~\ref{tab:scores}. $\Delta$ is the paired difference against IID. Intervals are percentile intervals from 5,000 paired moving blocks of length 20, computed on the saved forecasts without refitting or multiplicity adjustment. Values are computed on the reconstructed panel described in Appendix~\ref{app:reconstruction}; the QQQ totals differ from Table~\ref{tab:scores} in the sixth decimal.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def circulation(results):
    rows = []
    for asset in ASSETS:
        r = next(x for x in results["circulation_summary"] if x["asset"] == asset)
        rows.append(f"{asset} & {_signed(r['mean'], 2)} & {_signed(r['min'], 2)} & {_signed(r['max'], 2)} & "
                    f"{_plain(r['ar1'], 3)} & {_plain(r['mean_abs_diff'], 4)} & {_plain(r['max_abs_diff'], 4)}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{7pt}",
            r"\caption{Circulation of the fitted unrestricted chains}\label{tab:circulation}",
            r"\begin{tabular}{lrrrrrr}\toprule",
            r"Asset & Mean $j_t$ & Minimum & Maximum & AR(1) & Mean $|P_{ij}-R_{ij}|$ & Maximum\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize $j_t=J_{12}=J_{23}=J_{31}$ in percentage points of daily probability flow, computed at each of the 2,766 final-period origins from the fitted unrestricted matrix; detailed balance is $j_t=0$ and the reversible fit imposes it by construction. The sign is negative at every origin for every asset, so no share-of-origins column is reported. AR(1) is the first-order autocorrelation of the path. The two right-hand columns summarize the absolute difference between the fitted unrestricted and reversible off-diagonal entries, averaged and maximized over origins and over the six off-diagonal cells, in probability units.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def joint(results):
    rows = []
    for asset in ASSETS:
        for kind, label in (("ui", r"$\mathrm{U}-\mathrm{IID}$"), ("ru", r"$\mathrm{R}-\mathrm{U}$")):
            r = next(x for x in results["joint"]["rows"] if x["asset"] == asset and x["kind"] == kind)
            pointwise = results["joint"]["pointwise"][f"{asset}:{kind}"]
            rows.append(f"{asset} & {label} & {_signed(r['mean'], 6)} & {_ci(pointwise, 6)} & "
                        f"{r['p_stepdown']:.3f} & {r['p_holm']:.3f} & {_ci(r['simultaneous_ci'], 6)}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{5pt}",
            r"\caption{Familywise inference for the six log-loss contrasts}\label{tab:joint}",
            r"\footnotesize\begin{tabular}{llrrrrr}\toprule",
            r"Asset & Contrast & Estimate & Pointwise 95\% CI & Romano--Wolf $p$ & Holm $p$ & Simultaneous CI\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize U: unrestricted; R: reversible. All six contrasts use 2,766 common target sessions and the 5,000 paired moving-block draws of Section~\ref{sec:joint}, reindexed jointly across assets and contrasts. The Romano--Wolf $p$-value is stepdown-adjusted for the family of six; Holm's column applies a closed-form procedure to the same raw $p$-values. The simultaneous interval uses a single-step max-$|t|$ critical value of " + f"{results['joint']['q95']:.3f}" + r" and is more conservative than the stepdown test, so the two need not agree at the boundary. Pointwise intervals use the reconstructed panel and are not familywise; QQQ differs slightly from the archived estimates in Table~\ref{tab:intervals}. Separate adjustments for Brier and economic contrasts are reported in Appendix~\ref{app:families}; component intervals remain pointwise. No correction controls all families jointly.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def sensitivity(results):
    rows, first = [], True
    for asset in ASSETS:
        if not first:
            rows.append(r"\midrule")
        first = False
        for rule in RULE_ORDER:
            r = next(x for x in results["sensitivity"] if x["asset"] == asset and x["rule"] == rule)
            rows.append(f"{asset} & {RULE_LABEL[rule]} & {_signed(r['du'], 4)} {_ci(r['du_ci'], 4)} & "
                        f"{_signed(r['dr'], 4)} {_ci(r['dr_ci'], 4)} & {r['net_bp']:.3f} {_ci(r['net_ci'], 3)}" + r"\\")
    levels = {asset: [next(r for r in results["sensitivity"] if r["asset"] == asset and r["rule"] == k)["llu"]
                      for k in RULE_ORDER] for asset in ASSETS}
    shares = [r["fallback"] / r["origins_total"] for r in results["sensitivity"]
              if r.get("origins_total") and r["rule"] in ("fixed", "roll2520", "roll1260")]
    lo_share, hi_share = min(shares), max(shares)
    level_text = ", ".join(f"{v:.4f}" for v in levels["SPY"])
    level_text_q = ", ".join(f"{v:.4f}" for v in levels["QQQ"])
    head = [r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
            r"\caption{Threshold and window sensitivity of the final-period comparisons}\label{tab:sensitivity}",
            r"\begin{tabular}{llrrr}\toprule",
            r"Asset & Rule & $\Delta(\mathrm U-\mathrm{IID})$ log loss & $\Delta(\mathrm R-\mathrm U)$ log loss & $\Delta$ net return (bps/day)\\",
            r" & & [95\% CI] & [95\% CI] & [95\% CI]\\\midrule"]
    foot = (r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize U: unrestricted; R: reversible. Each rule re-estimates cut points and state counts as described in Section~\ref{sec:sensitivity}; the common-start rule applies 26 May 2000 to all three assets and leaves IWM unchanged. $\Delta$ net is the paired mean net-return difference at 5 bps per side. Intervals are percentile intervals from 5,000 paired moving blocks of length 20, conditional on the forecasts generated under each rule, without refitting and without multiplicity adjustment. Mean unrestricted levels, which are not comparable across rules, are "
            + level_text + " for SPY and " + level_text_q + " for QQQ under the five rules in order. Under the fixed and rolling rules the retained solver reaches a line-search precision limit at between "
            + f"{100*lo_share:.1f}" + r"\% and " + f"{100*hi_share:.1f}" + r"\% of estimation origins; the v13 fallback re-solves the same likelihood on simplex flow masses and accepts only positive finite fits with projected-gradient infinity norm at most $10^{-8}$. The primary rule has no fallbacks; this replaces the older softmax-gradient acceptance rule.}")
    tail = [r"\bottomrule\end{tabular}", foot, r"\end{table}"]
    return "\n".join(head + rows + tail)


def segments(results):
    rows, first = [], True
    for asset in ASSETS:
        if not first:
            rows.append(r"\midrule")
        first = False
        for r in results["segments"][asset]:
            label = r["seg"].replace("-", "--")
            rows.append(f"{asset} & {label} & {r['n']} & {_signed(r['ui'])} {_ci(r['ui_ci'])} & "
                        f"{_signed(r['ru'])} {_ci(r['ru_ci'])}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{6pt}",
            r"\caption{Final-period comparisons by fixed calendar segment}\label{tab:segments}",
            r"\begin{tabular}{llrrr}\toprule",
            r"Asset & Segment & Sessions & $\Delta(\mathrm U-\mathrm{IID})$ & $\Delta(\mathrm R-\mathrm U)$\\",
            r" & & & [95\% CI] & [95\% CI]\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize Segment boundaries were fixed for this revision and are exploratory. Each interval is a percentile interval from 5,000 paired moving blocks of length 20 computed within the segment, conditional on the saved forecasts and without refitting. Nine segment-specific intervals are not a segment-difference test; adjacent segments share no sessions but the point estimates are not independent of the full-period result.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def simulation(results):
    sim = results["simulation"]
    lengths = sim["lengths"]
    head = [r"\begin{table}[!htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
            r"\caption{Expected cost of imposing detailed balance in the controlled simulation}\label{tab:simulation}",
            r"\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrrrr}\toprule",
            r"Process & $j$ & " + " & ".join(f"$n={n:,}$".replace(",", "{,}") for n in lengths) + r"\\\midrule"]
    rows = []
    for scenario in ("reversible", "weak_irreversible", "empirical_irreversible"):
        cells = [next(c for c in sim["cells"] if c["scenario"] == scenario and c["n"] == n) for n in lengths]
        j = cells[0]["J"]
        rows.append(f"{SCENARIO_LABEL[scenario]} & {100*j:+.1f} & "
                    + " & ".join(f"{1e3*c['delta']:+.3f} ({1e3*c['se']:.3f})" for c in cells) + r"\\")
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize Entries are $10^{3}$ times the difference in expected log loss between the restricted and unrestricted estimators, evaluated against the generating matrix; Monte Carlo standard errors are in parentheses, from 500 simulated chains per cell. $j$ is the circulation $J_{12}=J_{23}=J_{31}$ in percentage points of daily flow. Positive values mean that imposing detailed balance costs expected log loss. The generating matrices have a uniform stationary distribution, baseline transition probabilities of 0.35 on the diagonal and 0.325 off the diagonal; stationary flows are one-third of those probabilities. The design was fixed after the empirical results were seen and is calibrated to them, so the agreement reported in Section~\ref{sec:simulation} is an explanation, not a confirmation.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def bins(results):
    rows, first = [], True
    for asset in ASSETS:
        if not first:
            rows.append(r"\midrule")
        first = False
        for model in ("iid", "markov", "reversible"):
            r = next(x for x in results["bins"] if x["asset"] == asset and x["model"] == model)
            rows.append(f"{asset} & {MODEL_LABEL[model]} & {r['empty']} & {r['smallest']} & {r['largest']}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{8pt}",
            r"\caption{Calibration bin occupancy in the final period}\label{tab:bins}",
            r"\begin{tabular}{llrrr}\toprule",
            r"Asset & Model & Empty bins (of 30) & Smallest non-empty & Largest\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize Bins are the ten fixed edges of Figure~\ref{fig:calibration}, counted separately for each of the three one-versus-rest classes, so thirty bins are possible. Most are empty: the forecast probabilities concentrate near one third, and a bin holding two observations provides little precision about calibration. The counts here are the only measure of how much support each plotted marker has; the summary statistic and reference diagnostic built from the same bins are in Table~\ref{tab:calibration}.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)





def build_all(results):
    return {name: fn(results) for name, fn in BUILDERS.items()}

def joint_family(results, key, caption, label, scale=1.0, digits=6, units=""):
    rows = []
    for asset in ASSETS:
        for kind, name in (("ui", r"$\mathrm{U}-\mathrm{IID}$"), ("ru", r"$\mathrm{R}-\mathrm{U}$")):
            r = next(x for x in results[key]["rows"] if x["asset"] == asset and x["kind"] == kind)
            point = results[key]["pointwise"][f"{asset}:{kind}"]
            rows.append(f"{asset} & {name} & {_signed(r['mean']*scale, digits)} & "
                        f"{_ci([point[0]*scale, point[1]*scale], digits)} & "
                        f"{r['p_stepdown']:.3f} & {r['p_holm']:.3f}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{6pt}",
            r"\caption{" + caption + r"}\label{" + label + "}",
            r"\footnotesize\begin{tabular}{llrrrr}\toprule",
            r"Asset & Contrast & Estimate" + units + r" & Pointwise 95\% CI & Romano--Wolf $p$ & Holm $p$\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize Separate family, adjusted within itself only. Estimates are model minus IID for the first contrast and reversible minus unrestricted for the second; for scores a negative difference favours the first model, for returns a positive difference does. Intervals are pointwise percentile intervals from the same 5,000 paired moving-block draws, reindexed jointly across the six contrasts; the $p$-values are stepdown-adjusted within this family and do not control errors jointly with Table~\ref{tab:joint}.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def calibration(results):
    rows = []
    first = True
    for asset in ASSETS:
        if not first:
            rows.append(r"\midrule")
        first = False
        for model in ("iid", "markov", "reversible"):
            r = next(x for x in results["calibration"] if x["asset"] == asset and x["model"] == model)
            rows.append(f"{asset} & {MODEL_LABEL[model]} & {r['ece']:.4f} & {_ci_plain(r['ci'])} & "
                        f"{r['p_value']:.3f} & {r['bins_30']}" + r"\\")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{6pt}",
            r"\caption{Observed calibration error and independent-draw reference diagnostic}\label{tab:calibration}",
            r"\begin{tabular}{llrrrr}\toprule",
            r"Asset & Model & Observed error & 95\% CI & Reference $p$ & Bins with $n\ge30$\\\midrule"]
    tail = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize The statistic is the mean over the three one-versus-rest classes of $\sum_b (n_b/N)\lvert\bar p_b-\hat o_b\rvert$ over the ten fixed bins of Figure~\ref{fig:calibration}. The interval is a percentile interval from 5,000 paired moving blocks of length 20 computed by resampling days and recomputing every bin. The reference $p$ uses 500 simulations with independent outcomes drawn from each day's frozen forecast and the plus-one correction. It is a diagnostic under that stronger reference model, not a general sequential-calibration test; values are not adjusted across the nine rows. The minimum possible value is $1/501$. Three to seven of the thirty possible bins hold thirty or more observations, so the statistic is dominated by a few cells.}",
            r"\end{table}"]
    return "\n".join(head + rows + tail)


def joint_brier(results):
    return joint_family(results, "joint_brier",
                        r"Brier-score contrasts, adjusted within their own family", "tab:jointbrier")


def joint_economic(results):
    rows = []
    for metric, scale, digits in (("net", 1e4, 4), ("sharpe", 1.0, 4)):
        for asset in ASSETS:
            for kind, name in (("ui", r"$\mathrm{U}-\mathrm{IID}$"), ("ru", r"$\mathrm{R}-\mathrm{U}$")):
                r = next(x for x in results["joint_economic"]["rows"]
                         if x["asset"] == asset and x["kind"] == kind and x["metric"] == metric)
                lo, hi = r["simultaneous_ci"]
                rows.append(f"{asset} & {name} & {_signed(r['mean']*scale, digits)} & "
                            f"{_ci([lo*scale, hi*scale], digits)} & {r['p_stepdown']:.3f} & {r['p_holm']:.3f}" + r"\\")
        if metric == "net":
            rows.append(r"\midrule")
            rows.append(r"\multicolumn{6}{l}{\emph{Sharpe ratio (conventional $\sqrt{252}$ scaling)}}\\")
            rows.append(r"\addlinespace[2pt]")
    head = [r"\begin{table}[!htbp]\centering\small\setlength{\tabcolsep}{5pt}",
            r"\caption{Economic contrasts, adjusted within their own family}\label{tab:jointecon}",
            r"\footnotesize\begin{tabular}{llrrrr}\toprule",
            r"Asset & Contrast & Estimate & Simultaneous 95\% CI & Romano--Wolf $p$ & Holm $p$\\\midrule",
            r"\multicolumn{6}{l}{\emph{Mean net return, 5 bps per side (bps/day)}}\\",
            r"\addlinespace[2pt]"]
    foot = [r"\bottomrule\end{tabular}",
            r"\par\smallskip\parbox{0.98\linewidth}{\footnotesize One exploratory family of twelve contrasts, adjusted within itself, with no joint error control across this family and Table~\ref{tab:joint}. Estimates are unrestricted minus IID for the first contrast and reversible minus unrestricted for the second; for both metrics a positive difference favours the first model. Intervals are simultaneous, from the single-step max-$|t|$ critical value of " + f"{results['joint_economic']['q95']:.3f}" + r"; the $p$-values are stepdown-adjusted. The Sharpe statistic is recomputed inside every bootstrap draw from the resampled days, so its sampling variability is that of a ratio rather than of a mean.}",
            r"\end{table}"]
    return "\n".join(head + rows + foot)


BUILDERS = {"tab_components.tex": components, "tab_circulation.tex": circulation,
            "tab_joint.tex": joint, "tab_sensitivity.tex": sensitivity,
            "tab_segments.tex": segments, "tab_simulation.tex": simulation,
            "tab_bins.tex": bins,
            "tab_jointbrier.tex": joint_brier, "tab_jointecon.tex": joint_economic,
            "tab_calibration.tex": calibration}