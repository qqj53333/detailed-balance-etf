"""Checks of review corrections, saved moments and generated table consistency."""
from pathlib import Path
import sys,json,re,math
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.sharpe_diagnostic import symmetric_components

checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name);print('PASS',name)
body=(ROOT/'source/body.tex').read_text(encoding='utf-8')
tables=(ROOT/'source/tables.tex').read_text(encoding='utf-8')
audit=json.loads((ROOT/'revision_audit/v14_review_diagnostics.json').read_text())
moments=json.loads((ROOT/'revision_audit/v14_sharpe_moments.json').read_text())
check('all existing forecast hashes match recorded v12 values',all(r['matches_v12_hash'] for r in audit['assets'].values()))
check('ten v13 generated tables matched the v13 package',len(audit['table_matches'])==10 and all(audit['table_matches'].values()))
check('archived verification passed all 231 comparisons',audit['archived_verify_passed'] and '231 comparisons, 0 mismatch(es)' in audit['archived_verify_output'])
for asset,row in moments.items():
    check(f'{asset}: independently recomputed descriptive moments match review audit',
          abs(row['sharpe_difference']-audit['assets'][asset]['sharpe_difference'])<1e-12)
    u,i=row['metrics']['markov'],row['metrics']['iid']
    mean,scale=symmetric_components(u['mean_net'],u['sd_net'],i['mean_net'],i['sd_net'])
    check(f'{asset}: mean and scale components sum to ratio difference',abs(mean+scale-row['sharpe_difference'])<1e-12)
spy=moments['SPY']
check('SPY standard deviation increases, not decreases',spy['metrics']['markov']['sd_net']>spy['metrics']['iid']['sd_net'])
check('SPY negative means imply positive scale contribution here',spy['metrics']['iid']['mean_net']<0 and spy['metrics']['markov']['mean_net']<0 and spy['scale_component']>0)
check('abstract qualifies primary log-loss intervals',('No primary reversible-minus-unrestricted log-loss interval excludes zero.' in body) or bool(re.search(r'Reversible-minus-unrestricted log-loss differences.{0,80}intervals containing zero',body.split(r'\end{abstract}')[0])))  # v16: content, not the v14 sentence
check('no universal score-factorization claim retained','apply to any proper score paired with any decision rule' not in body)
check('paired-variance correction direction is not asserted','parameter count alone does not determine the direction' in body)
check('SPY cost columns unchanged and correctly ordered',r'\mbox{31.21\%} & \mbox{-0.47\%} & \mbox{-67.05\%}' in tables)
check('QQQ cost columns unchanged and correctly ordered',r'\mbox{41.66\%} & \mbox{7.40\%} & \mbox{-64.52\%}' in tables)
source='\n'.join(p.read_text(encoding='utf-8') for p in (ROOT/'source').rglob('*.tex'))
labels=set(re.findall(r'\\label\{([^}]+)\}',source))
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',source))
cites={x.strip() for g in re.findall(r'\\cite[pt]?(?:\[[^\]]*\])?\{([^}]+)\}',source) for x in g.split(',')}
bibs=set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',source))
check('all source references and citations resolve',refs<=labels and cites<=bibs)
print(len(checks),'focused v14 checks passed. No new full pipeline run or causal attribution.')
