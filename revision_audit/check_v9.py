"""Consistency checks for the additions reported in revision v9.

    python revision_audit/check_v9.py

Standard library only. Verifies that the claims made in Sections 4.2, 4.7 and 4.8
follow from the numbers in v9_additions.json. It does not re-run the experiment.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADD = json.loads((ROOT / "v9_additions.json").read_text(encoding="utf-8"))
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


rows = ADD["joint_inference"]["rows"]
ui = [r for r in rows if r["contrast"].endswith("ui")]
ru = [r for r in rows if r["contrast"].endswith("ru")]

decisive = [r["contrast"] for r in ui if not (r["simultaneous_ci"][0] <= 0 <= r["simultaneous_ci"][1])]
check("exactly one U-IID contrast is decisive", decisive == ["QQQ:ui"], str(decisive))
check("stepdown rejects QQQ and IWM", [r["contrast"] for r in ui if r["p_stepdown"] < 0.05] == ["QQQ:ui", "IWM:ui"])
check("IWM is rejected by stepdown but not by Holm", next(r for r in ui if r["contrast"] == "IWM:ui")["p_holm"] > 0.05)
check("no R-U contrast reaches 0.15", min(r["p_stepdown"] for r in ru) >= 0.15,
      str(min(r["p_stepdown"] for r in ru)))
check("IWM lies on the boundary (stepdown < 0.05 < single-step p)",
      next(r for r in ui if r["contrast"] == "IWM:ui")["p_stepdown"] < 0.05
      < next(r for r in ui if r["contrast"] == "IWM:ui")["p_single"])
check("simultaneous interval excludes zero only for QQQ",
      [r["contrast"] for r in ui if not (r["simultaneous_ci"][0] <= 0 <= r["simultaneous_ci"][1])] == ["QQQ:ui"])

segs = ADD["segments"]
flat = [(a, s) for a in ("SPY", "QQQ", "IWM") for s in segs[a]]
check("nine segment rows", len(flat) == 9, str(len(flat)))
late = {a: next(s for s in segs[a] if s["seg"] == "2023-2025") for a in segs}
check("U-IID changes sign in 2023-2025 for SPY and QQQ", late["SPY"]["ui"] > 0 and late["QQQ"]["ui"] > 0,
      f"{late['SPY']['ui']:.5f} {late['QQQ']['ui']:.5f}")
check("QQQ 2023-2025 interval excludes zero", late["QQQ"]["ui_lo"] > 0)
check("IWM 2023-2025 is essentially zero", abs(late["IWM"]["ui"]) < 5e-4)
check("R-U positive in all nine segments", all(s["ru"] > 0 for _, s in flat))

cells = {(c["scenario"], c["n"]): c for c in ADD["simulation"]["cells"]}
check("fifteen simulation cells", len(cells) == 15, str(len(cells)))
d5000 = [cells[(s, 5000)]["delta"] for s in ("reversible", "weak_irreversible", "empirical_irreversible")]
check("cost rises with circulation at n=5000", d5000[0] < d5000[1] < d5000[2], str([round(v, 6) for v in d5000]))
check("reversible truth favours the restriction at every length",
      all(cells[("reversible", n)]["delta"] < 0 for n in (250, 500, 1260, 2520, 5000)))
check("strong circulation is costly at the empirical lengths (2520 and above)",
      cells[("empirical_irreversible", 2520)]["delta"] > 0 and cells[("empirical_irreversible", 5000)]["delta"] > 0)
check("strong circulation crosses from benefit to cost between 250 and 500",
      cells[("empirical_irreversible", 250)]["delta"] < 0 < cells[("empirical_irreversible", 500)]["delta"])
asym = ADD["simulation"]["asymptotic_cost"]
check("asymptotic cost increases with circulation", asym["0.000"] < asym["-0.007"] < asym["-0.014"])

bins = ADD["calibration_bins"]
check("at least 23 of 30 bins are empty everywhere",
      all(v[0] >= 23 for a in bins for v in bins[a].values()),
      str(min(v[0] for a in bins for v in bins[a].values())))
check("at least one bin holds two observations", any(v[1] <= 2 for a in bins for v in bins[a].values()))

fb = ADD["solver_fallbacks"]
worst = max(fb["counts"][a][s] / fb["denominators"][a] for a in fb["counts"] for s in fb["counts"][a])
best = min(fb["counts"][a][s] / fb["denominators"][a] for a in fb["counts"] for s in fb["counts"][a])
check("fallback share lies between 0.1% and 4.6%", 0.001 < best and worst < 0.047,
      f"{best:.4f} .. {worst:.4f}")

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
