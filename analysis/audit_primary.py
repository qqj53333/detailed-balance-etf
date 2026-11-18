"""Independent audit of the final-period forecasts. Arguments: prices_dir runs_dir."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def audit(asset, prices, runs):
    bars = pd.read_csv(prices / (asset + ".csv"), index_col="Date")
    assert bars.index.is_unique and bars.index.is_monotonic_increasing
    returns = (bars.Close / bars.Open - 1).to_numpy()
    paths = list((runs / asset).glob("run_*/forecasts.csv"))
    assert len(paths) == 1
    saved = pd.read_csv(paths[0]).query("split == 'final_test'").set_index("target_date")
    dates = bars.loc["2015":"2025-12-31"].index
    assert dates.equals(saved.index) and len(dates) == 2766
    forecasts = []
    targets = []
    binary_scores = []
    direction_net = []
    for day in dates:
        i = bars.index.get_loc(day)
        assert saved.at[day, "signal_date"] == bars.index[i-1]
        thresholds = np.quantile(returns[:i], [1/3, 2/3])
        s = np.searchsorted(thresholds, returns[:i])
        y = np.searchsorted(thresholds, returns[i])
        assert y == saved.at[day, "target_state"]
        assert abs(returns[i] - saved.at[day, "target_intraday_return"]) < 1e-14
        counts = np.bincount(3*s[:-1]+s[1:], minlength=9).reshape(3,3)
        iid = counts.sum(0)+1
        c = counts+1
        row = c.sum(1)
        # Solve the reversible likelihood score equations in stationary masses.
        pi = (row + c.sum(0))/(2*c.sum())
        for _ in range(200):
            scale = row/pi
            flow = (c+c.T)/(scale[:,None]+scale[None,:])
            new = flow.sum(1)/flow.sum()
            if np.max(abs(new-pi)) < 1e-13:
                break
            pi = new
        else:
            raise AssertionError("reversible fit failed")
        k = s[-1]
        forecasts.append([iid/iid.sum(), c[k]/row[k], flow[k]/flow[k].sum()])
        # Binary transitions pool the two outer origins before smoothing.
        group = [1] if k == 1 else [0,2]
        pooled = counts[group].sum(axis=0)
        middle = (pooled[1]+1)/(pooled.sum()+2)
        upper_share = iid[2]/(iid[0]+iid[2])
        binary = [(1-middle)*(1-upper_share), middle, (1-middle)*upper_share]
        binary_scores.append(-np.log(binary[y]))
        u = c[k]/row[k]
        delta_weight = u[2] - (1-u[1])*upper_share
        direction_net.append(delta_weight*(returns[i]-.001))
        targets.append(y)
    p = np.array(forecasts)
    old = saved[[f"p_{m}_{s}" for m in ("iid","markov","reversible")
                 for s in ("down","neutral","up")]].to_numpy().reshape(p.shape)
    errors = abs(p-old).max(axis=(0,2))
    ll = -np.log(p[np.arange(len(p)),:,targets])
    print(asset, "persistence share", (np.mean(binary_scores)-ll[:,0].mean())/(ll[:,1].mean()-ll[:,0].mean()), "direction net bp",1e4*np.mean(direction_net))
    weights = p[:,:,2]+p[:,:,1]/2
    net = weights*(saved.target_intraday_return.to_numpy()[:,None]-.001)
    print(asset, "max probability errors",errors,"mean loss",ll.mean(0),
          "U-IID net bp",1e4*(net[:,1]-net[:,0]).mean(),
          "terminal wealth",np.prod(1+net,axis=0))
    assert max(errors[:2]) < 1e-12 and errors[2] < 1e-8
    return np.diff(ll, axis=1)


if __name__ == "__main__":
    diffs = np.column_stack([audit(a,*map(Path,sys.argv[1:])) for a in ("SPY","QQQ","IWM")])
    n = len(diffs)
    rng = np.random.default_rng(20261113)
    sums = np.vstack([np.zeros(6), diffs.cumsum(0)])
    samples = []
    for starts in rng.integers(n-19,size=(5000,(n+19)//20)):
        ends = starts+20
        ends[-1] = starts[-1]+(n-1)%20+1
        samples.append((sums[ends]-sums[starts]).sum(0)/n)
    mean = diffs.mean(0)
    se = np.std(samples,axis=0,ddof=1)
    maxima = abs((samples-mean)/se).max(1)
    print("six simultaneous max-t p",[(1+sum(maxima>=abs(x)))/5001 for x in mean/se])
