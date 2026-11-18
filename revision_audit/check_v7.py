"""Consistency checks for the additions reported in revision v7.

Run from the package directory with the standard library only:

    python revision_audit/check_v7.py

The script verifies internal arithmetic of the numbers stated in the manuscript.
It does not rerun the market-data experiment, does not read row-level data, and
does not establish that the reported estimates are correct; it only checks that
the published numbers are mutually consistent.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADD = json.loads((ROOT / "v7_additions.json").read_text(encoding="utf-8"))

TOTALS = {  # Table 2, six decimals
    ("SPY", "U"): 1.093333, ("SPY", "R"): 1.094648,
    ("QQQ", "U"): 1.086977, ("QQQ", "R"): 1.087662,
    ("IWM", "U"): 1.093705, ("IWM", "R"): 1.095300,
}

failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


# 1. Components sum to the reported total log loss.
for key, row in ADD["components"]["table"].items():
    asset, model = key.split("_")
    total = row["A"] + row["B"]
    published = TOTALS[(asset, model)]
    check(
        f"components add to total: {key}",
        abs(total - published) <= 1e-4,
        f"{total:.6f} vs {published:.6f}",
    )

# 2. Every interval contains its point estimate.
for key, row in ADD["components"]["table"].items():
    for comp in ("dA", "dB"):
        lo, hi = row[comp + "_ci"]
        check(
            f"interval contains estimate: {key} {comp}",
            lo <= row[comp] <= hi,
            f"{row[comp]} not in [{lo}, {hi}]",
        )
for asset, row in ADD["return_components"].items():
    if asset == "note":
        continue
    for comp in ("dG", "dR"):
        lo, hi = row[comp + "_ci"]
        check(
            f"interval contains estimate: return {asset} {comp}",
            lo <= row[comp] <= hi,
            f"{row[comp]} not in [{lo}, {hi}]",
        )

# 3. Net equals gross minus the incremental cost term, at 5 bps per side.
c = 0.0005
for asset, row in ADD["return_components"].items():
    if asset == "note":
        continue
    turnover = ADD["break_even"]["mean_turnover_difference_round_trips"][asset]
    implied = row["dG"] - 1e4 * c * turnover
    check(
        f"net = gross - cost: {asset}",
        abs(implied - row["dR"]) <= 5e-3,
        f"{implied:.5f} vs {row['dR']:.5f}",
    )

# 4. Circulation consistency: a_t equals one third of the pairwise difference,
#    and the reported mean lies inside the reported range.
for asset in ("SPY", "QQQ", "IWM"):
    row = ADD["circulation"][asset]
    check(
        f"circulation mean inside range: {asset}",
        row["min"] <= row["mean"] <= row["max"],
        str(row),
    )
    check(
        f"circulation mean negative: {asset}",
        row["mean"] < 0,
        str(row["mean"]),
    )
    check(
        f"reversible matrix moves entries by half the pairwise difference: {asset}",
        abs(row["mean_abs_P_minus_R"] - abs(row["mean"]) / 100.0 * 1.5) <= 6e-3,
        f"{row['mean_abs_P_minus_R']:.4f}",
    )

# 5. Fixed-half cost illustration reproduces the published counterfactual.
wealth = (1.0 - c) ** 2766
check("cost-only wealth", abs(wealth - 0.25073821172899147) <= 1e-12, f"{wealth:.12f}")

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
