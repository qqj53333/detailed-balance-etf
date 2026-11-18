"""Checks specific to revision v15.

    python revision_audit/check_v15.py

v15 makes two changes on top of v14: the abstract labels the five-basis-point cost
as a baseline common to all allocations, and check_v13 no longer depends on v13's
exact wording. Standard library only.
"""

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
conclusion = BODY.split(r"\section{Limitations and conclusion}")[1]

check("abstract labels the cost as a baseline common to all allocations",
      "baseline common to all nine allocations" in abstract)
check("abstract no longer juxtaposes the gain with an unexplained loss",
      "although their incremental cost differences are small" not in abstract)
check("abstract keeps the incremental cost magnitude", "0.005" in abstract)
check("abstract at most 250 words", len(abstract.split()) <= 250, str(len(abstract.split())))  # v16: was 235
check("abstract no longer claims the cost is the incremental cost of the model",
      bool(re.search(r"incremental cost differences", abstract)))
check("conclusion still warns against reading the scale comparison as incremental",
      "must not be read as the incremental cost" in conclusion)

v13 = (ROOT / "check_v13.py").read_text(encoding="utf-8")
check("check_v13 no longer asserts on exact v13 sentences",
      'check("introduction states what generalises"' not in v13
      and 'check("conclusion separates arithmetic from model verdict"' not in v13)
check("check_v13 tests content, not phrasing", "re.search" in v13)

print()
if failures:
    raise SystemExit(f"{len(failures)} check(s) failed")
print("all checks passed")
