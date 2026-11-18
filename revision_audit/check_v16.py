"""Checks for revision v16: every new number in the text against the v16 results.

    python revision_audit/check_v16.py

Reads source/body.tex, source/references.tex, source/tables/*.tex and
revision_audit/v16_results/{layers,restriction_cost}.json. Standard library only.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODY = (ROOT / "source" / "body.tex").read_text(encoding="utf-8")
APPX = (ROOT / "source" / "appendix_body.tex").read_text(encoding="utf-8")
REFS = (ROOT / "source" / "references.tex").read_text(encoding="utf-8")
TAB = ROOT / "source" / "tables"
L = json.loads((ROOT / "revision_audit" / "v16_results" / "layers.json").read_text())
C = json.loads((ROOT / "revision_audit" / "v16_results" / "restriction_cost.json").read_text())
A = ("SPY", "QQQ", "IWM")
RULES = ("primary", "fixed", "roll2520", "roll1260")
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + str(detail)))
    if not ok:
        failures.append(name)


def text(fragment):
    return fragment in BODY


def excl(v):
    return v[1] * v[2] > 0


def z(v):
    return v[0] / ((v[2] - v[1]) / 3.92)


P = {t: L[f"primary:{t}"] for t in A}
abstract = BODY.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
check("abstract at most 250 words", len(abstract.split()) <= 250, len(abstract.split()))

# Settings and reproduction of the saved forecasts
check("bootstrap settings unchanged", L["settings"] == dict(seed=20261113, block=20, draws=5000, alpha=1.0, cost=0.0005))
for k, v in L.items():
    if ":" in k:
        check(f"{k}: unrestricted forecasts reproduced to 1e-16", v["max_unrestricted_error"] < 1e-16, v["max_unrestricted_error"])
        check(f"{k}: 2,766 targets", v["n"] == 2766)
for t in A:
    for key in ("ll", "net"):
        parts = sum(P[t][k][key][0] for k in ("vol2-iid", "m3-vol2", "U-m3"))
        check(f"{t}: layers add exactly to U-IID ({key})", abs(parts - P[t]["U-iid"][key][0]) < 1e-15)
    check(f"{t}: direction layer equals the conditional score component", abs(P[t]["U-m3"]["ll"][0] - P[t]["U-iid"]["B"][0]) < 1e-15)
    check(f"{t}: U-IID log loss matches Table 9 (reconstructed panel)",
          abs(P[t]["U-iid"]["ll"][0] - {"SPY": -0.005314, "QQQ": -0.011699, "IWM": -0.004942}[t]) < 5e-7)

# Abstract and Section 4.4 numbers
shares = [round(100 * P[t]["vol2-iid"]["ll"][0] / P[t]["U-iid"]["ll"][0]) for t in A]
check("persistence shares 82/82/66", shares == [82, 82, 66], shares)
check("abstract and text state 82, 82 and 66 per cent", "82\\%, 82\\% and 66\\%" in abstract and text("this is 82\\%, 82\\% and 66\\%"))
for t, s in zip(A, ("4.36", "9.60", "3.24")):
    check(f"{t}: persistence layer {s} and excludes zero", f"{-1e3 * P[t]['vol2-iid']['ll'][0]:.2f}" == s and excl(P[t]["vol2-iid"]["ll"]))
for t, s in zip(A, ("0.99", "2.02", "0.39")):
    check(f"{t}: middle layer {s} with interval containing zero", f"{-1e3 * P[t]['m3-vol2']['ll'][0]:.2f}" == s and not excl(P[t]["m3-vol2"]["ll"]))
for t, s in zip(A, ("+0.04", "-0.07", "-1.31")):
    check(f"{t}: direction layer {s}", f"{1e3 * P[t]['U-m3']['ll'][0]:+.2f}" == s)
check("direction layer excludes zero only for IWM", [excl(P[t]["U-m3"]["ll"]) for t in A] == [False, False, True])
for t, s in zip(A, ("-0.96", "-2.10", "-1.70")):
    check(f"{t}: U-V {s} with interval containing zero", f"{1e3 * P[t]['U-vol2']['ll'][0]:.2f}" == s and not excl(P[t]["U-vol2"]["ll"]))
check("U-V intervals quoted", text("$[-3.75,+1.93]$, $[-4.56,+0.66]$ and $[-3.80,+0.32]$"))
check("V and M allocations within 2e-4 of one-half", all(P[t]["max_abs_tilt"][m] < 2e-4 for t in A for m in ("vol2", "m3")))
check("V and M net layers below 2e-5 bps/day", all(abs(P[t][k]["net"][0]) * 1e4 < 2e-5 for t in A for k in ("vol2-iid", "m3-vol2")))
for t, s in zip(A, ("0.0425", "0.0200", "0.0728")):
    check(f"{t}: direction net {s} bps/day", f"{1e4 * P[t]['U-m3']['net'][0]:.4f}" == s)
check("SPY direction log-loss interval [-1.04,+1.05]", [round(1e3 * x, 2) for x in P["SPY"]["U-m3"]["ll"][1:]] == [-1.04, 1.05])
check("abstract incremental net 0.043, 0.020, 0.073", "0.043, 0.020 and 0.073" in abstract)

# Property 1
for rule in ("primary", "roll2520", "roll1260"):
    for t in A:
        r = L[f"{rule}:{t}"]["max_tilt_over_bound"]
        check(f"{rule} {t}: Equation (15) holds at every origin for IID, V, M", max(r.values()) <= 1.0, r)
qf = [L[f"fixed:{t}"]["q_range"] for t in ("SPY", "QQQ")]
check("fixed cut points: marginal upper share 0.515 to 0.531 for SPY and QQQ",
      round(min(q[0] for q in qf), 3) == 0.515 and round(max(q[1] for q in qf), 3) == 0.531)
fshare = [round(100 * (L[f"fixed:{t}"]["vol2-iid"]["net"][0] + L[f"fixed:{t}"]["m3-vol2"]["net"][0]) / L[f"fixed:{t}"]["U-iid"]["net"][0]) for t in A]
check("fixed-rule first two layers 15/22/2 per cent of net difference", fshare == [15, 22, 2], fshare)

# Rule counts and the weighted direction term
allc = [L[f"{r}:{t}"] for r in RULES for t in A]
vs = [100 * v["vol2-iid"]["ll"][0] / v["U-iid"]["ll"][0] for v in allc]
check("persistence share ranges 56 to 108 per cent", round(min(vs)) == 56 and round(max(vs)) == 108, (min(vs), max(vs)))
check("direction log-loss excludes zero in 1 of 12", sum(excl(v["U-m3"]["ll"]) for v in allc) == 1)
check("direction net excludes zero in 9 of 12", sum(excl(v["U-m3"]["net"]) for v in allc) == 9)
check("abstract states nine of twelve and one", "nine of twelve cases, its log-loss interval in one" in abstract)
check("weighted direction favours U in all 12", all(v["U-m3"]["Bw"][0] < 0 for v in allc))
check("unweighted direction favours U in 9 of 12", sum(v["U-m3"]["B"][0] < 0 for v in allc) == 9)
check("weighting raises |estimate/SE| in every case", all(abs(z(v["U-m3"]["Bw"])) > abs(z(v["U-m3"]["B"])) for v in allc))
wex = [(r, t) for r in RULES for t in A if excl(L[f"{r}:{t}"]["U-m3"]["Bw"])]
check("weighted direction excludes zero in 3 of 12, all IWM", len(wex) == 3 and all(t == "IWM" for _, t in wex), wex)

# Rolling-frequency forecasts
for t, s in zip(A, ("9.13", "10.18", "2.43")):
    v = P[t]["rolling_frequency"]["252"]["U-roll"]
    check(f"{t}: rolling-252 beats U by {s} with interval containing zero", f"{1e3 * v[0]:.2f}" == s and not excl(v))
r1260 = {t: P[t]["rolling_frequency"]["1260"] for t in A}
check("rolling-1,260 beats U in 2015-2018 for all three (excluding zero)",
      all(r1260[t]["2015-2018"]["U-roll"][0] > 0 and excl(r1260[t]["2015-2018"]["U-roll"]) for t in A))
check("rolling-1,260 loses to U in 2019-2022 for QQQ and IWM only (excluding zero)",
      [excl(r1260[t]["2019-2022"]["U-roll"]) and r1260[t]["2019-2022"]["U-roll"][0] < 0 for t in A] == [False, True, True])

# Restriction cost
for t, f, o, fr in zip(A, ("1.289", "0.355", "1.131"), ("1.315", "0.685", "1.595"), ("1.087", "0.564", "0.650")):
    c = C[t]; ru = P[t]["R-U"]["ll"]
    check(f"{t}: fitted reference {f}", f"{1e3 * c['fitted_reference']:.3f}" == f)
    check(f"{t}: observed R-U {o}", f"{1e3 * c['actual_R_minus_U']:.3f}" == o and abs(c["actual_R_minus_U"] - ru[0]) < 1e-12)
    check(f"{t}: frozen reference {fr} from the 2014-12-31 origin", f"{1e3 * c['frozen_reference']:.3f}" == fr and c["frozen_signal_date"] == "2014-12-31")
    check(f"{t}: both references inside the R-U interval", ru[1] < c["fitted_reference"] < ru[2] and ru[1] < c["frozen_reference"] < ru[2])
    check(f"{t}: symmetric estimator within 1.4e-4 and counts within one", c["symmetric_estimator_max_abs_diff"] <= 1.4e-4 + 1e-6 and c["max_row_minus_col_count"] <= 1)
ratios = [C[t][k] / ((P[t]["R-U"]["ll"][2] - P[t]["R-U"]["ll"][1]) / 3.92) for t in A for k in ("fitted_reference", "frozen_reference")]
check("references are 0.7 to 1.4 standard errors", round(min(ratios), 1) >= 0.7 and round(max(ratios), 1) <= 1.4, [round(r, 2) for r in ratios])
zs = [abs(x) for s in C["simulation"].values() for x in s["z"]]
check("closed form within 1.27 Monte Carlo SE", f"{max(zs):.2f}" == "1.27")
check("crossover lengths 1,474 and 368", round(C["simulation"]["-0.007"]["n_star"]) == 1474 and round(C["simulation"]["-0.014"]["n_star"]) == 368)
for t, a_, b_ in (("SPY", "-0.13", "+1.44"), ("IWM", "-0.08", "+1.67"), ("QQQ", "+0.34", "+0.35")):
    check(f"{t}: R-U component departures {a_}/{b_}",
          f"{1e3 * P[t]['R-U']['A'][0]:+.2f}" == a_ and f"{1e3 * P[t]['R-U']['B'][0]:+.2f}" == b_)

# Conditional returns and exposures
for t, s in zip(A, ("7.61", "13.37", "7.64")):
    check(f"{t}: lower-state mean return {s}", f"{P[t]['state_mean_return_bp']['0']['mean']:.2f}" == s)
q = P["QQQ"]["state_mean_return_bp"]["0"]
check("QQQ lower-state intervals", [round(x, 2) for x in q["ci"]] == [5.73, 22.53] and [round(x, 2) for x in q["simultaneous_ci"]] == [1.68, 25.05])
check("QQQ intervals contain 10 bps", q["ci"][0] < 10 < q["ci"][1] and q["simultaneous_ci"][0] < 10 < q["simultaneous_ci"][1])
for t, s in zip(A, ("1.67", "0.28", "1.50")):
    check(f"{t}: extra exposure after lower state {s} pp", f"{100 * (P[t]['state_weights_U']['0'] - 0.5):.2f}" == s)
w = P["QQQ"]["state_weights_U"]
check("QQQ highest mean exposure follows middle sessions (51.04%)", max(w, key=w.get) == "1" and f"{100 * w['1']:.2f}" == "51.04")

# Segments and sensitivity
seg = [round(100 * P[t]["segments"]["2015-2018"]["vol2-iid"]["ll"][0] / P[t]["segments"]["2015-2018"]["U-iid"]["ll"][0]) for t in A]
check("2015-2018 persistence shares 70/77/59", seg == [70, 77, 59], seg)
qs = P["QQQ"]["segments"]["2023-2025"]
check("QQQ 2023-2025 persistence layer +8.88 [+0.36,+17.83] of +11.06",
      [round(1e3 * x, 2) for x in qs["vol2-iid"]["ll"]] == [8.88, 0.36, 17.83] and round(1e3 * qs["U-iid"]["ll"][0], 2) == 11.06)
shr = [round(100 * (1 - L[f"roll1260:{t}"]["U-iid"]["ll"][0] / P[t]["U-iid"]["ll"][0])) for t in A]
check("rolling-1,260 estimates 67/60/51 per cent smaller", shr == [67, 60, 51], shr)

# References, disclosure and wiring
check("Wolfer and Watanabe cited with Theorem 6.1", "WolferWatanabe2021" in REFS and "Theorem~6.1]{WolferWatanabe2021}" in BODY)
check("Tong cited for the parameter term, Akaike removed", "Tong1975" in REFS and "citep{Tong1975}" in BODY and "Akaike1974" not in REFS + BODY)
check("layer named middle-versus-outer persistence, not volatility persistence", "volatility persistence" not in BODY.lower() and "middle-versus-outer persistence" in BODY)
check("subtitle present in all three title entry points", all("Where Score Gains Meet the Decision Rule" in (ROOT / "source" / f).read_text(encoding="utf-8") for f in ("preprint.tex", "manuscript.tex", "blind_manuscript.tex")))
DECL = BODY.split(r"\section*{Declaration of Generative AI in Scientific Writing}")[1]
check("AI declaration names every system",
      all(x in DECL for x in ("ChatGPT", "Codex", "Gemini", "DeepSeek Harness", "Anthropic Claude")))
check("AI declaration states that use was not limited to language editing", "not only for language editing" in DECL)
check("AI declaration discloses corrected AI errors", "reversed sign" in DECL and "bootstrap draws" in DECL)
check("AI declaration does not claim the author recomputed results", "written and run by the AI systems" in DECL
      and "independently recomputed" not in DECL)
check("appendix points to the main declaration instead of a second list", "Declaration of Generative AI" in APPX)
for name in ("tab_layers", "tab_layerrules", "tab_restriction", "tab_rollfreq", "tab_statemeans"):
    check(f"{name}.tex generated and included", (TAB / f"{name}.tex").exists()
          and (f"tables/{name}" in APPX or f"tables/{name}" in (ROOT / "source" / "tables.tex").read_text(encoding="utf-8")))
check("post hoc status stated in abstract and limitations",
      "specified after the primary results" in abstract and "were all specified after the primary results" in BODY)

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
