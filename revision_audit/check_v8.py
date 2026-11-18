"""Consistency checks for the sensitivity table reported in revision v8.

    python revision_audit/check_v8.py

Standard library only. Verifies that the claims made in Section 4.5 follow from
the numbers in v8_additions.json. It does not re-run the experiment.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADD = json.loads((ROOT / "v8_additions.json").read_text(encoding="utf-8"))
rows = ADD["table"]

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


def contains_zero(ci):
    return ci[0] <= 0.0 <= ci[1]


check("fifteen asset-by-rule rows", len(rows) == 15, str(len(rows)))

neg = [r for r in rows if r["du"] < 0]
check("unrestricted-minus-IID negative in every row", len(neg) == 15, str(len(neg)))
check(
    "unrestricted-minus-IID range -0.0017 to -0.0148",
    abs(min(r["du"] for r in rows) + 0.014777) < 5e-5 and abs(max(r["du"] for r in rows) + 0.001744) < 5e-5,
    f"{min(r['du'] for r in rows):.4f} .. {max(r['du'] for r in rows):.4f}",
)

pos = [r for r in rows if r["dr"] > 0]
check("reversible-minus-unrestricted positive in every row", len(pos) == 15, str(len(pos)))

excl = [r for r in rows if not contains_zero(r["dr_ci"])]
check("exactly one R-U interval excludes zero", len(excl) == 1, str([(r["asset"], r["rule"]) for r in excl]))
check(
    "that interval puts the reversible chain on the worse side",
    bool(excl) and excl[0]["dr"] > 0 and excl[0]["asset"] == "IWM" and excl[0]["rule"] == "fixed",
    str(excl),
)

loose = [r for r in rows if contains_zero(r["du_ci"])]
check("five U-IID intervals contain zero", len(loose) == 5, str(len(loose)))
check(
    "which five: roll1260 all three, roll2520 QQQ, common SPY",
    sorted((r["rule"], r["asset"]) for r in loose)
    == sorted([("roll1260", "SPY"), ("roll1260", "QQQ"), ("roll1260", "IWM"),
               ("roll2520", "QQQ"), ("common", "SPY")]),
    str(sorted((r["rule"], r["asset"]) for r in loose)),
)

nets = [r for r in rows if contains_zero(r["net_ci"])]
check("four net intervals contain zero", len(nets) == 4, str(len(nets)))
check(
    "net differences stay positive and within 0.02-0.14 bps/day",
    all(r["net_bp"] > 0 for r in rows) and min(r["net_bp"] for r in rows) > 0.019 and max(r["net_bp"] for r in rows) < 0.145,
    f"{min(r['net_bp'] for r in rows):.3f} .. {max(r['net_bp'] for r in rows):.3f}",
)

primary_iwm = next(r for r in rows if r["asset"] == "IWM" and r["rule"] == "primary")
common_iwm = next(r for r in rows if r["asset"] == "IWM" and r["rule"] == "common")
check(
    "common start leaves IWM unchanged",
    abs(primary_iwm["du"] - common_iwm["du"]) < 1e-9 and abs(primary_iwm["dr"] - common_iwm["dr"]) < 1e-9,
    f"{primary_iwm['du']} vs {common_iwm['du']}",
)

spy_p = next(r for r in rows if r["asset"] == "SPY" and r["rule"] == "primary")
spy_c = next(r for r in rows if r["asset"] == "SPY" and r["rule"] == "common")
check(
    "common start moves the SPY estimate from -0.00531 to -0.00536",
    abs(spy_p["du"] + 0.005314) < 1e-4 and abs(spy_c["du"] + 0.005356) < 1e-4,
    f"{spy_p['du']:.5f} -> {spy_c['du']:.5f}",
)

levels = {(r["asset"], r["rule"]): r["mean_logloss_unrestricted"] for r in rows}
check(
    "fixed cut points lower the unrestricted level for SPY and QQQ",
    levels[("SPY", "fixed")] < levels[("SPY", "primary")] and levels[("QQQ", "fixed")] < levels[("QQQ", "primary")],
    str(levels),
)

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
