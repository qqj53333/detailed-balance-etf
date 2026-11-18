"""Recompute v6 descriptive additions from the supplied v5 aggregate summaries.

Standard library only. This does not refit forecasts or access market data.
Run from any directory: python revision_audit/recompute.py
"""
from pathlib import Path
import json
import math
import random

folder = Path(__file__).resolve().parent
reported = json.loads((folder / 'reported_inputs.json').read_text())
weights = json.loads((folder / 'mechanism_descriptives_v5.json').read_text())
expected = json.loads((folder / 'derived_values.json').read_text())

actual = {}
for asset, values in reported.items():
    cost_difference = 2 * 0.0005 * (
        weights[asset]['markov']['w_mean'] - weights[asset]['iid']['w_mean']
    )
    net_difference = values['mean_net_difference']
    actual[asset] = {
        'relative_logloss_reduction_percent': -100 * values['logloss_difference'] / values['iid_logloss'],
        'gross_difference_bps_per_day': 10000 * (net_difference + cost_difference),
        'cost_difference_bps_per_day': 10000 * cost_difference,
        'net_difference_bps_per_day': 10000 * net_difference,
        'annualized_mean_net_difference_percentage_points': 252 * 100 * net_difference,
    }
    for key, value in actual[asset].items():
        assert math.isclose(value, expected[asset][key], rel_tol=1e-12, abs_tol=1e-12)
wealth = (1 - 0.0005) ** 2766
assert math.isclose(wealth, expected['cost_only_illustration']['wealth'], abs_tol=1e-12)

# Independently check the two new algebraic identities on positive probabilities
# and arbitrary signed returns; no claims about empirical market behavior follow.
rng = random.Random(20261115)
for _ in range(1000):
    raw = [rng.uniform(0.01, 1) for _ in range(3)]
    p = [value / sum(raw) for value in raw]
    m = p[1]
    a = p[2] / (p[0] + p[2])
    assert math.isclose(p[2] - p[0], (1 - m) * (2 * a - 1), abs_tol=1e-12)
    for y in range(3):
        z, u = int(y == 1), int(y == 2)
        factorized = -z * math.log(m) - (1-z) * math.log(1-m) - (1-z) * (u*math.log(a) + (1-u)*math.log(1-a))
        assert math.isclose(-math.log(p[y]), factorized, abs_tol=1e-12)
    w = 0.5 + 0.5 * (p[2] - p[0])
    r = rng.uniform(-0.2, 0.2)
    c = rng.uniform(0, 0.0025)
    direct = w*r - 2*c*w
    separated = 0.5*(r-2*c) + (w-0.5)*r - 2*c*(w-0.5)
    assert math.isclose(direct, separated, abs_tol=1e-12)

print(json.dumps(actual, indent=2))
print(f'Cost-only terminal wealth: {wealth:.12f}; loss: {100*(1-wealth):.6f}%')
print('PASS: aggregate calculations and 1,000 algebraic identity checks.')
print('No market-data refit, empirical score-component attribution, or new confidence intervals were performed.')
