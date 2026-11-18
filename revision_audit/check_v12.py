"""Focused regression and document-consistency checks; no market-data refit.

Run from the package root: python revision_audit/check_v12.py
Requires the analysis dependencies (NumPy, pandas, SciPy).
"""
from pathlib import Path
import sys, json, re
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import scores, simulation, tables, rules

passed = []
def check(name, condition):
    if not condition:
        raise AssertionError(name)
    passed.append(name)
    print('PASS', name)

# Balanced outcomes and constant uniform probabilities have exactly zero
# empirical bin error. Their bootstrap mean is positive and must not replace it.
frame = pd.DataFrame({'target_state': np.tile(np.arange(3),10)})
for m in scores.MODELS:
    for k in ('down','neutral','up'):
        frame[f'p_{m}_{k}'] = 1/3
cfg = SimpleNamespace(assets=('TEST',), final=lambda _: frame)
cal = scores.calibration(cfg, draws=30, block=2, seed=71)
check('calibration point estimate is observed zero, not bootstrap mean',
      all(abs(r['ece']) < 1e-14 and r['bootstrap_mean'] > .001 for r in cal))

audit = json.loads((ROOT/'revision_audit/v12_calibration_recheck.json').read_text())
old = json.loads((ROOT/'revision_audit/v11_additions.json').read_text())
oldmap = {(r['asset'],r['model']):r for r in old['calibration']}
check('all nine old calibration point estimates were bootstrap means',
      len(audit['calibration'])==9 and all(r['bootstrap_mean_matches_v11'] for r in audit['calibration']))
check('calibration intervals and reference probabilities preserved',
      all(np.allclose(r['ci'],oldmap[(r['asset'],r['model'])]['ci'],rtol=0,atol=1e-12)
          and abs(r['p_value']-oldmap[(r['asset'],r['model'])]['p_value'])<1e-14
          for r in audit['calibration']))
check('calibration table generated from corrected audit',
      (ROOT/'source/tables/tab_calibration.tex').read_text().strip()==tables.calibration({'calibration':audit['calibration']}).strip())

v9=json.loads((ROOT/'revision_audit/v9_additions.json').read_text())
sim=dict(v9['simulation'],lengths=[250,500,1260,2520,5000])
generated=tables.simulation({'simulation':sim})
check('simulation circulation displayed in percentage points',
      '& -0.7 &' in generated and '& -1.4 &' in generated and '& -0.0 &' not in generated)
check('simulation table agrees with generator',
      (ROOT/'source/tables/tab_simulation.tex').read_text().strip()==generated.strip())
for j in [0.,-.007,-.014]:
    P, actual=simulation.build(j)
    check(f'generating matrix valid and circulation correct ({j})',
          np.all(P>0) and np.allclose(P.sum(axis=1),1) and np.allclose(np.ones(3)/3@P,np.ones(3)/3)
          and abs(actual-j)<1e-14)

def fail_original(*args,**kwargs):
    raise RuntimeError('synthetic solver failure')
fake_models=SimpleNamespace(reversible_matrix=fail_original)
states=np.tile(np.arange(3),5)

# v13 replaced the softmax fallback with a simplex solver polished by a KKT root
# search, so the rejection test now forces a non-constant flow gradient instead of
# patching the old parameterisation.
bad_grad=lambda m,c,r: np.array([0.,1.,0.,0.,0.,0.])
feasible=SimpleNamespace(x=np.full(7,1/7))
with patch.object(rules,'least_squares',return_value=feasible), patch.object(rules,'_masses_grad',bad_grad):
    try:
        rules.reversible_fit(fake_models,states)
    except RuntimeError:
        rejected=True
    else:
        rejected=False
check('fallback refuses a fit above the documented gradient limit',rejected)
P, how, grad=rules.reversible_fit(fake_models,states)
check('fallback accepts a finite feasible fit within tolerance',
      how=='fallback' and np.allclose(P.sum(axis=1),1) and grad<=1e-8)

source='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'source').rglob('*.tex'))
labels=re.findall(r'\\label\{([^}]+)\}',source)
refs=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source)
cites=[k.strip() for group in re.findall(r'\\cite[pt]?\{([^}]+)\}',source) for k in group.split(',')]
bibs=re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',source)
check('every source reference and citation has a definition', set(refs)<=set(labels) and set(cites)<=set(bibs))
body=(ROOT/'source/body.tex').read_text()
check('IWM stepdown rejection and method sensitivity both stated',
      'significant under stepdown ($p=0.048$)' in body and 'Holm ($p=0.056$)' in body)
check('obsolete simulation value removed', '5.8\\times10^{-4}' not in body)
component_audit=json.loads((ROOT/'revision_audit/v12_component_recheck.json').read_text())
qqq=next(r for r in component_audit['components'] if r['asset']=='QQQ' and r['model']=='U')
check('QQQ middle-component interval excludes zero in both prose and table',
      component_audit['table_matches_saved_source'] and qqq['dA_ci'][1]<0
      and '[-0.01850,-0.00456]' in body and 'QQQ interval just fails' not in body)
print(f'{len(passed)} focused checks passed; no full refit or new bootstrap coverage claim.')
