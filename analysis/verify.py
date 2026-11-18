"""Compare regenerated numbers with the published audit files."""

from __future__ import annotations

import json
from pathlib import Path

from .core import Config

TOL = 1e-5
P_TOL = 1e-3  # p-values are published rounded to three or four decimals


def _close(a, b, tol=TOL):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return abs(float(a) - float(b)) <= tol


def compare(cfg: Config) -> bool:
    audit = cfg.package / "revision_audit"
    v7 = json.loads((audit / "v7_additions.json").read_text(encoding="utf-8"))
    v8 = json.loads((audit / "v8_additions.json").read_text(encoding="utf-8"))
    v9 = json.loads((audit / "v9_additions.json").read_text(encoding="utf-8"))
    res = cfg.work / "results"
    problems = []

    def check(name, got, want):
        tol = P_TOL if " p_" in f" {name}" or name.endswith(("p_raw", "p_holm", "p_stepdown", "p_single")) else TOL
        if not _close(got, want, tol):
            problems.append(f"{name}: got {got!r}, published {want!r}")

    # Table 7: score components (published in the v7 audit).
    comps = json.loads((res / "components.json").read_text(encoding="utf-8"))
    for row in comps:
        want = v7["components"]["table"][f"{row['asset']}_{row['model']}"]
        for key in ("A", "dA", "dA_ci", "B", "dB", "dB_ci"):
            check(f"component {row['asset']}_{row['model']} {key}", row[key], want[key])

    # Table A9: bin counts.
    bins = json.loads((res / "bins.json").read_text(encoding="utf-8"))
    for row in bins:
        want = v9["calibration_bins"][row["asset"]][row["model"]]
        check(f"bins {row['asset']} {row['model']}", [row["empty"], row["smallest"], row["largest"]], want)

    # Table A6: estimation rules (published in the v8 audit).
    sens = json.loads((res / "sensitivity.json").read_text(encoding="utf-8"))
    for row in sens:
        want = next(w for w in v8["table"] if w["asset"] == row["asset"] and w["rule"] == row["rule"])
        check(f"sensitivity {row['asset']} {row['rule']} du", row["du"], want["du"])
        check(f"sensitivity {row['asset']} {row['rule']} du_ci", row["du_ci"], want["du_ci"])
        check(f"sensitivity {row['asset']} {row['rule']} dr", row["dr"], want["dr"])
        check(f"sensitivity {row['asset']} {row['rule']} net", row["net_bp"], want["net_bp"])

    # Table 9: joint inference.
    joint = json.loads((res / "joint.json").read_text(encoding="utf-8"))
    for row, want in zip(sorted(joint["rows"], key=lambda r: r["contrast"]),
                         sorted(v9["joint_inference"]["rows"], key=lambda r: r["contrast"])):
        for key in ("mean", "se", "t", "p_raw", "p_holm", "p_stepdown", "p_single",
                    "simultaneous_ci"):
            check(f"joint {row['contrast']} {key}", row[key], want[key])

    # Table A7: calendar segments.
    segs = json.loads((res / "segments.json").read_text(encoding="utf-8"))
    for asset in v9["segments"]:
        for got, want in zip(segs[asset], v9["segments"][asset]):
            check(f"segment {asset} {want['seg']} ui", got["ui"], want["ui"])
            check(f"segment {asset} {want['seg']} ui_ci", got["ui_ci"], [want["ui_lo"], want["ui_hi"]])
            check(f"segment {asset} {want['seg']} ru", got["ru"], want["ru"])
            check(f"segment {asset} {want['seg']} ru_ci", got["ru_ci"], [want["ru_lo"], want["ru_hi"]])
            check(f"segment {asset} {want['seg']} n", got["n"], want["n"])

    # Table A8: simulation.
    sim = json.loads((res / "simulation.json").read_text(encoding="utf-8"))
    for cell in sim["cells"]:
        want = next(c for c in v9["simulation"]["cells"]
                    if c["scenario"] == cell["scenario"] and c["n"] == cell["n"])
        check(f"simulation {cell['scenario']} n={cell['n']} delta", cell["delta"], want["delta"])

    total = (len(comps) * 6 + len(bins) * 3 + len(sens) * 4 + len(joint["rows"]) * 8
             + 9 * 5 + len(sim["cells"]))
    print(f"{total} comparisons, {len(problems)} mismatch(es)")
    for p in problems[:20]:
        print("  " + p)
    return not problems
