"""Driver for the derived analyses, including v12 calibration corrections.

Usage:
    python -m analysis.pipeline --work out --prices prices --package . --stages all

Stages may be run individually. Each writes JSON/CSV under <work>/results and
figures under <work>/figures. The final verify stage compares the generated
numbers with the published ones in revision_audit/v9_additions.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import figures, inference, scores, simulation, tables
from .core import Config, write_json
from .rules import RULES, run_rule

STAGES = ("code", "forecasts", "matrices", "scores", "inference", "sensitivity",
          "simulation", "figures", "tables", "verify")


def stage_code(cfg: Config):
    from .core import extract_code
    src = extract_code(cfg)
    print(f"frozen code available at {src}")


def stage_forecasts(cfg: Config):
    from .core import run_cli
    for ticker in cfg.assets:
        path = run_cli(cfg, ticker)
        print(f"{ticker}: forecasts in {path}")


def stage_matrices(cfg: Config):
    frame = scores.circulation(cfg)
    write_json(cfg.work / "results" / "circulation_summary.json",
               scores.circulation_summary(frame, cfg.assets))
    print(f"matrices: {len(frame)} origins")


def stage_scores(cfg: Config):
    write_json(cfg.work / "results" / "components.json", scores.component_table(cfg))
    write_json(cfg.work / "results" / "components_ru.json", scores.relative_components(cfg))
    write_json(cfg.work / "results" / "derivatives.json", scores.derivatives(cfg))
    write_json(cfg.work / "results" / "bins.json", scores.bins(cfg))
    calibrated = scores.calibration(cfg)
    tested = {(r["asset"], r["model"]): r for r in scores.calibration_test(cfg)}
    for row in calibrated:
        extra = tested[(row["asset"], row["model"])]
        row["p_value"] = extra["p_value"]
        row["null_q95"] = extra["null_q95"]
    write_json(cfg.work / "results" / "calibration.json", calibrated)
    print("score components, sensitivities and bin counts written")


def stage_inference(cfg: Config):
    names, matrix = inference.six_contrasts(cfg)
    joint = inference.romano_wolf(matrix, names)
    write_json(cfg.work / "results" / "joint.json", joint)
    names_b, matrix_b = inference.brier_contrasts(cfg)
    write_json(cfg.work / "results" / "joint_brier.json", inference.romano_wolf(matrix_b, names_b))
    names_e, matrix_e, obs_e = inference.economic_family(cfg)
    write_json(cfg.work / "results" / "joint_economic.json",
               inference.family_stepdown(matrix_e, obs_e, names_e))
    segs = inference.segments(cfg)
    write_json(cfg.work / "results" / "segments.json", segs)
    paths = inference.rolling_paths(cfg)
    for asset, frame in paths.items():
        frame.to_csv(cfg.work / "results" / f"rolling_{asset}.csv")
    print("joint inference, segments and rolling paths written")


def stage_sensitivity(cfg: Config):
    for rule in RULES:
        for ticker in cfg.assets:
            frame = run_rule(cfg, ticker, rule)
            print(f"{rule} {ticker}: {len(frame)} origins")
    summarise(cfg)


def summarise(cfg: Config):
    rows = []
    for rule in RULES:
        for ticker in cfg.assets:
            path = cfg.work / "sens" / f"{rule}_{ticker}.csv"
            if not path.exists():
                continue
            f = pd.read_csv(path)
            origins_total = len(f)
            f = f[f.split == "final_test"].reset_index(drop=True)
            ll = scores.losses(f)
            du = ll["markov"] - ll["iid"]
            dr = ll["reversible"] - ll["markov"]
            idx = inference.block_indices(len(f))
            lo1, hi1 = inference.percentile_ci(du, idx)
            lo2, hi2 = inference.percentile_ci(dr, idx)
            wm = (0.5 + 0.5 * (f.p_markov_up - f.p_markov_down)).to_numpy()
            wi = (0.5 + 0.5 * (f.p_iid_up - f.p_iid_down)).to_numpy()
            r = f.target_intraday_return.to_numpy()
            net = (wm * r - 2 * 0.0005 * wm) - (wi * r - 2 * 0.0005 * wi)
            lo3, hi3 = inference.percentile_ci(net, idx)
            rows.append(dict(asset=ticker, rule=rule, n=len(f), origins_total=origins_total,
                             fallback=int((f.fit == "fallback").sum()),
                             llu=float(ll["markov"].mean()), du=float(du.mean()), du_ci=[lo1, hi1],
                             dr=float(dr.mean()), dr_ci=[lo2, hi2],
                             net_bp=float(net.mean() * 1e4), net_ci=[lo3 * 1e4, hi3 * 1e4]))
    write_json(cfg.work / "results" / "sensitivity.json", rows)


def stage_simulation(cfg: Config):
    result = simulation.run(cfg)
    write_json(cfg.work / "results" / "simulation.json", result)
    print(f"simulation: {len(result['cells'])} cells, {result['fallbacks']} fallbacks")


def stage_figures(cfg: Config):
    matrices = cfg.work / "results" / "matrices.csv"
    if matrices.exists():
        figures.circulation(pd.read_csv(matrices, parse_dates=["signal_date"]), cfg.work / "figures")
    paths = {a: pd.read_csv(cfg.work / "results" / f"rolling_{a}.csv", parse_dates=["date"]).set_index("date")
             for a in cfg.assets}
    figures.rolling(paths, cfg.work / "figures")
    print("figures written")


def stage_tables(cfg: Config):
    import json
    res = cfg.work / "results"
    results = {name: json.loads((res / f"{name}.json").read_text(encoding="utf-8"))
               for name in ("components", "circulation_summary", "joint", "joint_brier",
                            "joint_economic", "sensitivity", "segments", "simulation",
                            "bins", "calibration")}
    rendered = tables.build_all(results)
    target = cfg.package / "source" / "tables"
    target.mkdir(parents=True, exist_ok=True)
    for name, body in rendered.items():
        (target / name).write_text(body + "\n", encoding="utf-8")
        (cfg.work / "tables").mkdir(parents=True, exist_ok=True)
        (cfg.work / "tables" / name).write_text(body + "\n", encoding="utf-8")
    print(f"{len(rendered)} tables written to {target}")


def stage_verify(cfg: Config):
    from .verify import compare
    if not compare(cfg):
        raise SystemExit("verification failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the revision numbers")
    parser.add_argument("--work", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--package", default=".")
    parser.add_argument("--assets", default="SPY,QQQ,IWM")
    parser.add_argument("--stages", default="all")
    args = parser.parse_args(argv)

    cfg = Config(work=args.work, prices=args.prices, package=args.package,
                 code=Path(args.work) / "code",
                 assets=tuple(a.strip() for a in args.assets.split(",")))

    chosen = STAGES[:-1] if args.stages == "all" else tuple(s.strip() for s in args.stages.split(","))
    for name in chosen:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}")
        print(f"\n=== {name} ===")
        globals()[f"stage_{name}"](cfg)
    if args.stages == "all":
        print("\n=== verify ===")
        stage_verify(cfg)


if __name__ == "__main__":
    main()
