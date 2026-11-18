# Reproduction pipeline

The pipeline regenerates the derived analysis stages from three price files
and the frozen code package, including the secondary families and corrected
v12 calibration summaries. Its aggregate verify stage covers the older
v7--v9 results only; `revision_audit/check_v14.py` checks the current review corrections. Older
check scripts are retained as historical records and may assert superseded prose
or solver behavior.
The full pipeline was not rerun for this editorial revision.

    python -m analysis.pipeline --work out --prices prices --package . --stages all

The final stage compares the regenerated numbers with the published ones in
`revision_audit/v9_additions.json` and exits non-zero on a mismatch.

## Inputs

Only one input is external:

- `--prices DIR`: three CSV files named `SPY.csv`, `QQQ.csv`, `IWM.csv`, each with
  columns `Date, Open, Close` and the same adjustment convention as the frozen
  adapter. These are not distributed with the package. `fetch_prices.py` can
  retrieve the series from the public vendor endpoint, subject to the vendor's
  terms; the authors make no claim that redistribution is permitted.

Everything else is derived: the frozen code is unpacked from
`public_code_release_candidate_v5.zip`, the forecast panels are rebuilt by running
that unmodified package, and all later stages read only those outputs.

## Stages

| Stage | What it does | Main output |
|---|---|---|
| `code` | unpacks the frozen release candidate under `<work>/code` | — |
| `forecasts` | runs the frozen CLI for each asset | `<work>/runs/<ticker>/run_*/forecasts.csv` |
| `matrices` | refits the walk-forward and records every 3x3 matrix | `results/matrices.csv`, `circulation_summary.json` |
| `scores` | score components, allocation derivatives, bin counts, calibration diagnostics | `components.json`, `components_ru.json`, `derivatives.json`, `bins.json`, `calibration.json` |
| `inference` | separate log-loss, Brier and economic families, segments, rolling paths | `joint.json`, `joint_brier.json`, `joint_economic.json`, `segments.json`, `rolling_*.csv` |
| `sensitivity` | five estimation rules | `sens/*.csv`, `sensitivity.json` |
| `simulation` | controlled simulation and asymptotic costs | `simulation.json` |
| `figures` | Figures 3 and 4 | `figures/figure5_circulation.pdf`, `figures/figure6_rolling.pdf` |
| `tables` | renders Tables 7, 8, 9 and A6--A12 from the JSON outputs | `source/tables/*.tex` |
| `verify` | compares against the published audit file | exit status |

Stages can be run separately, for example
`--stages forecasts,scores`. The tables stage writes into the manuscript source,
so `--stages ... ,tables` followed by `tectonic source/preprint.tex` rebuilds the
PDF from the price files alone.

## Settings

Bootstrap: 5,000 paired non-circular moving blocks, length 20, seed 20261113, no
refitting. Simulation: seed 20261115, 500 replications per cell. Estimation uses
the frozen estimator; where it stops on a line-search precision limit, the same
objective is re-solved over positive simplex flow masses by SLSQP and polished
using least-squares stationarity equations. The projected-gradient ceiling is
1e-8. This v13 implementation is unchanged in v14; v14 corrects the outdated
softmax-gradient description in the manuscript and table note.


## Layout

The modules are `core`, `rules`, `scores`, `inference`, `simulation`, `figures`,
`pipeline` and `verify`. `original_scripts/` keeps the ad-hoc scripts that were run
while preparing the revision, for provenance; they use absolute paths and are
superseded by the modules.

## Requirements

Python 3.12 with numpy, pandas, scipy. Figures additionally need matplotlib.

## Verification

The verify stage makes 231 comparisons against the three audit files: 18 score
components (Table 7), 9 bin aggregates (Table A9), 60 sensitivity entries
(Table A6), 48 joint-inference entries (Table 9), 45 segment entries (Table A7)
and 15 simulation cells (Table A8). Tolerances are 1e-5 for estimates and
intervals, which are stored rounded to five decimals in the audit files, and 1e-3
for p-values, which are stored rounded to three or four decimals.

The v13 record reports a complete run from the retained reconstructed prices.
The v14 review checked the existing run artifacts and repeated the aggregate
comparison, without launching a new full model refit or simulation run.


## What the verification does and does not establish

A successful verify stage checks only the listed archived comparisons against
the generated aggregate results, within the stated tolerances. It does not establish that the price files are the archived snapshot:
the vendor revises history, and the hashes are reported in Appendix B.1 rather
than matched.

## Descriptive Sharpe check (v14)

```text
python -m analysis.sharpe_diagnostic --forecast-root RUN/runs --output moments.json
```

This reads existing final-period forecasts and computes the mean and sample
standard deviation of net returns, the conventional Sharpe ratio, and the exact
symmetric mean/scale decomposition. It does not perform new significance tests
or identify volatility timing. No row-level data are written to the JSON output.
