"""Consistency checks for the additions reported in revision v11.

    python revision_audit/check_v11.py

Standard library only. Verifies that the claims in Appendix A.5 follow from the
numbers in v11_additions.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADD = json.loads((ROOT / "v11_additions.json").read_text(encoding="utf-8"))
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


brier = ADD["brier"]["rows"]
ui = {r["asset"]: r for r in brier if r["kind"] == "ui"}
check("Brier: QQQ survives", ui["QQQ"]["p_stepdown"] < 0.05)
check("Brier: IWM survives", ui["IWM"]["p_stepdown"] < 0.05)
check("Brier: SPY does not", ui["SPY"]["p_stepdown"] > 0.05)

econ = ADD["economic"]["rows"]
check("Economic family has twelve contrasts", len(econ) == 12, str(len(econ)))
net = {(r["asset"], r["kind"]): r for r in econ if r["metric"] == "net"}
sharpe = {(r["asset"], r["kind"]): r for r in econ if r["metric"] == "sharpe"}
check("Net: only IWM U-IID survives",
      [k for k, r in net.items() if r["p_stepdown"] < 0.05] == [("IWM", "ui")],
      str([k for k, r in net.items() if r["p_stepdown"] < 0.05]))
check("Net: IWM R-U falls just outside", net[("IWM", "ru")]["p_stepdown"] > 0.05)
check("Sharpe: SPY and IWM both contrasts survive",
      all(sharpe[k]["p_stepdown"] < 0.05 for k in (("SPY", "ui"), ("SPY", "ru"), ("IWM", "ui"), ("IWM", "ru"))))
check("Sharpe: those reversible contrasts are negative",
      sharpe[("SPY", "ru")]["mean"] < 0 and sharpe[("IWM", "ru")]["mean"] < 0)
check("Sharpe: QQQ does not survive",
      sharpe[("QQQ", "ui")]["p_stepdown"] > 0.05 and sharpe[("QQQ", "ru")]["p_stepdown"] > 0.05)

cal = ADD["calibration"]
check("Calibration: nine rows", len(cal) == 9, str(len(cal)))
check("Calibration: every model rejected at 5%", all(r["p_value"] < 0.05 for r in cal),
      str([round(r["p_value"], 3) for r in cal]))
check("Calibration: strongest rejection is the 0.002 floor",
      abs(min(r["p_value"] for r in cal) - 0.002) < 5e-4, str(min(r["p_value"] for r in cal)))
check("Calibration: observed error exceeds the null 95th percentile everywhere",
      all(r["ece"] > r["null_q95"] for r in cal))
check("Calibration: observed error between 0.02 and 0.05", all(0.02 < r["ece"] < 0.05 for r in cal))
check("Calibration: three to seven populated bins",
      all(3 <= r["bins_30"] <= 7 for r in cal), str([r["bins_30"] for r in cal]))

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
