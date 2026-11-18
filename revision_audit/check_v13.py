"""Checks for the framing introduced in revision v13.

    python revision_audit/check_v13.py

The v13 wording was rewritten in v14 and v15, so the assertions below test the
*content* of the framing rather than exact phrases: either phrasing of the same
claim satisfies them. Revision-specific checks for later edits live in
check_v14.py and check_v15.py.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BODY = (ROOT.parent / "source" / "body.tex").read_text(encoding="utf-8")
failures = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "  " + detail))
    if not ok:
        failures.append(name)


abstract = BODY.split(r"\begin{abstract}")[1].split(r"\end{abstract}")[0]
words = len(abstract.split())
check("abstract between 190 and 250 words", 190 <= words <= 250, str(words))  # v16: limit raised to 250 for the layer decomposition
check("abstract states the conditional nature",
      bool(re.search(r"conditional|sensitive to the adjustment|reverses", abstract)))
check("abstract reports the cost-to-gain comparison",
      bool(re.search(r"five basis points|cost", abstract)) and "basis points per day" in abstract)

conclusion = BODY.split(r"\section{Limitations and conclusion}")[1]
check("conclusion separates the economic result from a model verdict",
      bool(re.search(r"specific to the decision rule|arithmetic rather than a verdict", conclusion)))
check("conclusion names the transferable contribution",
      bool(re.search(r"integrated diagnostic template|protocol rather than the tickers|exact accounting for modest forecasting gains", conclusion)))
check("conclusion states that no reversible advantage is shown",
      bool(re.search(r"No forecast advantage from detailed balance|no reversible advantage is detected|neither demonstrate a forecast advantage from detailed balance", conclusion)))

intro = BODY.split(r"\section{Introduction and related research}")[1].split(r"\section{Design and estimators}")[0]
check("introduction states what transfers",
      bool(re.search(r"transferable contribution|What generalises beyond", intro)))

gains = [0.043, 0.020, 0.073]
round_trip_bp = 5.0
ratios = [round_trip_bp / g for g in gains]
check("round-trip cost is 60 to 250 times the daily gain",
      all(60 <= r <= 250 for r in ratios), str([round(r) for r in ratios]))

repro = ROOT.parent / "REPRODUCTION.md"
check("reproduction record present", repro.exists())
if repro.exists():
    text = repro.read_text(encoding="utf-8")
    check("record reports zero mismatches", "0 mismatches" in text)
    check("record reports identical forecast hashes", "identical" in text and "SHA-256" in text)
    check("record lists what is not established", "does not establish" in text)

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
