"""Optional helper: retrieve adjusted daily bars from the public vendor endpoint.

The vendor's terms govern use. This script exists so that a reader with the right
to obtain the data can rebuild the input files; it is not run by the pipeline.

    python analysis/fetch_prices.py --out prices
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
       "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits&includeAdjustedClose=true")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(ticker: str, start: str = "1993-01-01", end: str = "2026-03-01") -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    request = urllib.request.Request(URL.format(ticker=ticker, p1=p1, p2=p2), headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    result = payload["chart"]["result"][0]
    index = (pd.to_datetime(result["timestamp"], unit="s", utc=True)
             .tz_convert("America/New_York").normalize().tz_localize(None))
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    frame = pd.DataFrame({"Open": quote["open"], "Close": quote["close"], "AdjClose": adjusted},
                         index=index).dropna()
    frame = frame[(frame > 0).all(axis=1)]
    factor = frame["AdjClose"] / frame["Close"]
    out = pd.DataFrame({"Open": frame["Open"] * factor, "Close": frame["AdjClose"]})
    out.index.name = "Date"
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="prices")
    parser.add_argument("--tickers", default="SPY,QQQ,IWM")
    args = parser.parse_args()
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers.split(","):
        frame = fetch(ticker.strip())
        frame.to_csv(target / f"{ticker.strip()}.csv", date_format="%Y-%m-%d")
        print(f"{ticker}: {len(frame)} rows, {frame.index[0].date()} to {frame.index[-1].date()}")


if __name__ == "__main__":
    main()
