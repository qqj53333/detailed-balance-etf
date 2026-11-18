"""Descriptive Sharpe decomposition on existing forecast records; no new tests.

python -m analysis.sharpe_diagnostic --forecast-root RUN/runs --output audit.json
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def symmetric_components(mu_a,sd_a,mu_b,sd_b):
    if sd_a <= 0 or sd_b <= 0:
        raise ValueError('Positive standard deviations are required')
    mean=np.sqrt(252)*(mu_a-mu_b)*(.5/sd_a+.5/sd_b)
    scale=np.sqrt(252)*(.5*mu_a+.5*mu_b)*(1/sd_a-1/sd_b)
    return float(mean),float(scale)

def run(forecast_root):
    rows={}
    for asset in ('SPY','QQQ','IWM'):
        files=sorted((Path(forecast_root)/asset).glob('run_*/forecasts.csv'))
        if not files:raise FileNotFoundError(f'No forecasts for {asset}')
        path=files[-1]
        f=pd.read_csv(path);f=f[f.split=='final_test'].reset_index(drop=True)
        metrics={}
        for model in ('iid','markov','reversible'):
            w=f[f'weight_{model}'].to_numpy()
            net=w*(f.target_intraday_return.to_numpy()-0.001)
            mu,sd=float(net.mean()),float(net.std(ddof=1))
            metrics[model]=dict(mean_net=mu,sd_net=sd,sharpe=float(np.sqrt(252)*mu/sd))
        u,i=metrics['markov'],metrics['iid']
        mean,scale=symmetric_components(u['mean_net'],u['sd_net'],i['mean_net'],i['sd_net'])
        delta=u['sharpe']-i['sharpe']
        if abs(delta-mean-scale)>1e-12:raise AssertionError('Decomposition mismatch')
        rows[asset]=dict(n=len(f),forecast_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            metrics=metrics,sharpe_difference=delta,mean_component=mean,scale_component=scale)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(run(args.forecast_root),indent=2),encoding='utf-8')
    print('Saved descriptive moments and exact Sharpe components to',out)

if __name__=='__main__':main()
