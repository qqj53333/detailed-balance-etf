"""Shared configuration, input handling and resampling utilities.

All paths are supplied by the caller; nothing here depends on the directory in
which the revision was originally produced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261113
BLOCK = 20
DRAWS = 5000
ALPHA = 1.0
MIN_TRAIN = 2520
COST = 0.0005
SPLITS = (("development", "2009-12-31"), ("validation", "2014-12-31"), ("final_test", "2025-12-31"))


@dataclass
class Config:
    work: Path
    prices: Path
    package: Path
    code: Path
    assets: tuple = ("SPY", "QQQ", "IWM")

    def __post_init__(self):
        self.work = Path(self.work).resolve()
        self.prices = Path(self.prices).resolve()
        self.package = Path(self.package).resolve()
        self.code = Path(self.code).resolve()
        for sub in ("code", "runs", "sens", "results", "figures"):
            (self.work / sub).mkdir(parents=True, exist_ok=True)

    def run_dir(self, ticker: str) -> Path:
        runs = sorted((self.work / "runs" / ticker).glob("run_*"))
        if not runs:
            raise FileNotFoundError(f"no run_* directory under {self.work / 'runs' / ticker}")
        return runs[-1]

    def forecasts(self, ticker: str) -> pd.DataFrame:
        return pd.read_csv(self.run_dir(ticker) / "forecasts.csv", parse_dates=["signal_date", "target_date"])

    def final(self, ticker: str) -> pd.DataFrame:
        f = self.forecasts(ticker)
        return f[f["split"] == "final_test"].reset_index(drop=True)

    def prices_frame(self, ticker: str) -> pd.DataFrame:
        p = pd.read_csv(self.prices / f"{ticker}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        return p.loc[:"2025-12-31"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_code(cfg: Config) -> Path:
    """Unpack the frozen release candidate and return its source directory."""
    zpath = cfg.package / "public_code_release_candidate_v5.zip"
    target = cfg.code
    marker = target / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target)
        marker.write_text(sha256(zpath), encoding="utf-8")
    src = next(target.glob("*/src"))
    return src


def load_models(cfg: Config):
    src = extract_code(cfg)
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from markov_research import models, states  # noqa: E402

    return models, states


def run_cli(cfg: Config, ticker: str) -> Path:
    """Run the unmodified frozen package on the supplied prices."""
    src = extract_code(cfg)
    out = cfg.work / "runs" / ticker
    cmd = [sys.executable, "-m", "markov_research.cli", "--ticker", ticker,
           "--input-csv", str(cfg.prices / f"{ticker}.csv"), "--phase", "final",
           "--output", str(out)]
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env, cwd=str(src.parent))
    return sorted(out.glob("run_*"))[-1]


def block_indices(n: int, block: int = BLOCK, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """Common non-circular moving-block index matrix, shape (draws, n)."""
    rng = np.random.default_rng(seed)
    per = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, (draws, per))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)
    return idx[:, :n]


def percentile_ci(values, idx: np.ndarray) -> tuple:
    values = np.asarray(values, dtype=float)
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
