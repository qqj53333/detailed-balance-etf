# Full-pipeline reproduction record

Date: 2026-11-16. Package: v13 (identical analysis code to v12 apart from the
fallback solver described below). This record answers the question "was the whole
experiment rerun from prices with the corrected code?" — it was.

## Command

    python -m analysis.pipeline --work F:\MKV\_work\v13run \
        --prices F:\MKV\_work\prices \
        --package . --stages all

Wall time on this machine: about 15 minutes (nine processes, single-threaded).

Environment: Python 3.12.14, numpy 2.3.5, pandas 3.0.1, scipy 1.18.1,
matplotlib 3.11.2, Tectonic 0.15.0.

## Inputs

Three adjusted daily bars, Date/Open/Close, 1993-01-29 to 2026-02-27 truncated at
2025-12-31 by the pipeline:

| File | Bytes | SHA-256 (first 16) |
|---|---|---|
| SPY.csv | 402,619 | 7974388c7d0723b8 |
| QQQ.csv | 329,396 | 1e228a8d91fe97a3 |
| IWM.csv | 315,253 | 9dc11b1bbfcb6bab |

They are not distributed with the package; they are reproducible from the public
vendor endpoint with `analysis/fetch_prices.py` subject to the vendor's terms.

## What the run produced

| Stage | Output | Result |
|---|---|---|
| code | frozen release unpacked | untouched |
| forecasts | 5,768 / 4,226 / 3,918 rows | SHA-256 **identical** to the hashes recorded in `v12_calibration_recheck.json` |
| matrices | 13,912 origins | regenerated |
| scores | components, derivatives, bins, calibration | calibration matches `v12_calibration_recheck.json` to 0.0 |
| inference | 6 log-loss, 6 Brier, 12 economic contrasts, segments, paths | regenerated |
| sensitivity | five rules × three assets, 15 panels | regenerated |
| simulation | 15 cells, 110 solver fallbacks | regenerated |
| figures | Figures 3 and 4 | regenerated |
| tables | ten `.tex` files | **byte-identical** to the v12 versions |
| verify | 231 comparisons | **0 mismatches** |

Two checks matter most. The regenerated forecast panels hash to exactly the values
v12 recorded, so the reproduction is bit-level for the objects every later number
depends on. And all ten generated tables are byte-identical to the shipped ones,
including the sensitivity table, so the corrected solver below moves no reported
digit.

## Code change relative to v12

v12 tightened the fallback acceptance rule to reject fits whose gradient exceeds
2.1×10⁻⁵ under the softmax parameterisation. Running the full pipeline under that
rule aborted at `fixed SPY`: the fallback had stopped at a point whose raw
gradient was 5.1×10⁻⁵ even though the objective was numerically flat.

v13 replaces the fallback with a simplex formulation that is polished by solving
the KKT system `grad(m) = lambda, sum(m) = 1` with a least-squares root solver,
and measures stationarity with the *projected* gradient, which is the correct
criterion on the simplex. Across the 15 sensitivity panels the fallback now runs
14 times for SPY's fixed rule and never exceeds a projected gradient of
1.4×10⁻¹²; no fit is rejected. The fitted matrices are unchanged to display
precision: every table is byte-identical.

This is a numerical fix, not a change to the estimator or the research design.

## What this reproduction does not establish

- **Bootstrap coverage.** The intervals are still conditional percentile intervals
  under block resampling; their coverage under structural change is untested.
- **Data provenance.** The price files are a fresh retrieval of the same series,
  not the archived snapshot. Row counts and the published estimates match, but the
  fingerprints differ, as Appendix B.1 explains.
- **Independent replication.** This is the same code on a reconstructed panel, not
  a separate implementation on new data.
- **Calibration.** The parametric bootstrap remains a reference-model diagnostic,
  not a general test of sequential calibration.
