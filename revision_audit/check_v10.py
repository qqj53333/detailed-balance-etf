"""Consistency checks for the additions reported in revision v10.

    python revision_audit/check_v10.py

Standard library only. Verifies that the claims in Appendix A.5 follow from the
numbers in v10_additions.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADD = json.loads((ROOT / "v10_additions.json").read_text(encoding="utf-8"))
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


brier = ADD["brier"]["rows"]
ui_brier = {r["asset"]: r for r in brier if r["kind"] == "ui"}
check("Brier: QQQ survives at 5%", ui_brier["QQQ"]["p_stepdown"] < 0.05, str(ui_brier["QQQ"]["p_stepdown"]))
check("Brier: IWM survives at 5%", ui_brier["IWM"]["p_stepdown"] < 0.05, str(ui_brier["IWM"]["p_stepdown"]))
check("Brier: SPY does not survive", ui_brier["SPY"]["p_stepdown"] > 0.05, str(ui_brier["SPY"]["p_stepdown"]))
check("Brier: no reversible contrast survives",
      all(r["p_stepdown"] > 0.05 for r in brier if r["kind"] == "ru"))

econ = ADD["economic"]["rows"]
econ_map = {(r["asset"], r["kind"]): r for r in econ}
check("Economic: IWM U-IID survives", econ_map[("IWM", "ui")]["p_stepdown"] < 0.05)
check("Economic: IWM R-U survives", econ_map[("IWM", "ru")]["p_stepdown"] < 0.05)
check("Economic: that R-U estimate is negative", econ_map[("IWM", "ru")]["mean"] < 0)
check("Economic: SPY and QQQ do not survive",
      econ_map[("SPY", "ui")]["p_stepdown"] > 0.05 and econ_map[("QQQ", "ui")]["p_stepdown"] > 0.05)

cal = ADD["calibration"]
check("Calibration: nine rows", len(cal) == 9, str(len(cal)))
check("Calibration: errors between 0.02 and 0.05", all(0.02 < r["ece"] < 0.05 for r in cal),
      str([round(r["ece"], 4) for r in cal]))
check("Calibration: every interval contains the estimate",
      all(r["ci"][0] <= r["ece"] <= r["ci"][1] for r in cal))
check("Calibration: intervals are at least half as wide as the estimate",
      all((r["ci"][1] - r["ci"][0]) > 0.5 * r["ece"] for r in cal))
check("Calibration: at most seven bins hold thirty observations",
      all(r["bins_30"] <= 7 for r in cal), str([r["bins_30"] for r in cal]))
check("Calibration: at least three do", all(r["bins_30"] >= 3 for r in cal),
      str([r["bins_30"] for r in cal]))

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
