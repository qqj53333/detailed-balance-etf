# Detailed Balance in Three-State ETF Forecasting and Intraday Allocation

Jinziqi Zhang · Preprint v17 · 18 November 2026

The main paper and appendix compile together from source/preprint.tex. The empirical comparison uses next-session open-to-close returns, three ETFs, and an expanding primary rule. The benchmark layers are exploratory. Read AUDIT.md for the scope of the release checks and CHANGELOG.md for the v16 changes.

## Reproduce

Use Python 3.12 and an empty output directory. Run from the extracted package root. The price directory must contain SPY.csv, QQQ.csv and IWM.csv, with unique ascending Date and positive, same-basis Open and Close columns. See REPRODUCTION.md for source vintages, input fingerprints and earlier run evidence. The frozen estimator and its original protocol, data terms, tests and historical lockfile are in public_code_release_candidate_v5.zip. An authorized new vendor download can differ from the retained reconstruction.

    python -m pip install -r requirements.txt
    python analysis/fetch_prices.py --out prices
    python -m analysis.pipeline --work run --prices prices --package . --stages all
    python analysis/layers.py --prices prices --runs run/runs --sens run/sens --out run/v17
    python analysis/restriction_cost.py --prices prices --runs run/runs --out run/v17
    python analysis/tables_v16.py --results run/v17 --out source/tables
    python analysis/audit_primary.py prices run/runs
    tectonic source/preprint.tex

Skip the fetch if supplying authorized inputs. Use a fresh working copy because the table stages write TeX files. Use one run per ticker; the independent checker rejects ambiguous directories. A changed vendor history can cause verification against published estimates to fail. To build the paper alone, run just the last command with the included tables and figures. Tectonic 0.15 with its default v33 bundle was used for this release. The historical verify stage checks 231 aggregate values. The separate v17 checker independently fits all three primary models, reconstructs the binary benchmark and compares the daily predictions and dates. The release includes aggregate checks only. Do not commit prices or row-level run directories.

## Scope of the archive

analysis/ contains the pipeline and independent check. revision_audit/ retains historical records, including checks tied to earlier manuscript text. figures/ includes all six print figures. The pipeline generates the circulation and rolling-score figures; the four wealth and calibration graphics are supplied as finished source assets. REPRODUCTION.md describes runs by version. The final-period price reconstruction and raw simulation draws are not distributed. The repository includes the [v17 preprint](Preprint_v17.pdf).
