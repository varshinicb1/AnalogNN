import torch
import numpy as np
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nas.analog_nas import ScalingLawRobustnessScorer
from energy.analog_energy_model import AnalogEnergyModel

DIGITAL_ACCURACY = 0.95
INPUT_DIM = 784
OUTPUT_DIM = 10
TECH_NODES = ['28nm', '14nm', '7nm']


def compute_network_energy(em, depth, width, input_dim, output_dim):
    """Compute total energy per inference for a full network."""
    torch.manual_seed(42)

    if depth == 1:
        w = torch.randn(output_dim, input_dim) * 0.1
        x = torch.randn(1, input_dim)
        result = em.full_layer_energy(w, x)
        return result['total_energy_J']

    total = 0.0

    w = torch.randn(width, input_dim) * 0.1
    x = torch.randn(1, input_dim)
    result = em.full_layer_energy(w, x)
    total += result['total_energy_J']

    for _ in range(depth - 2):
        w = torch.randn(width, width) * 0.1
        x = torch.randn(1, width)
        result = em.full_layer_energy(w, x)
        total += result['total_energy_J']

    w = torch.randn(output_dim, width) * 0.1
    x = torch.randn(1, width)
    result = em.full_layer_energy(w, x)
    total += result['total_energy_J']

    return total


def compute_pareto(results):
    for r in results:
        dominated = False
        for other in results:
            if other is r:
                continue
            if (other['analog_accuracy'] >= r['analog_accuracy'] and
                other['efficiency'] >= r['efficiency'] and
                (other['analog_accuracy'] > r['analog_accuracy'] or
                 other['efficiency'] > r['efficiency'])):
                dominated = True
                break
        r['pareto'] = not dominated


arch_configs = []
for depth in [1, 2, 3, 4]:
    for width in [32, 64, 128, 256, 512]:
        arch_configs.append((depth, width))

scorer = ScalingLawRobustnessScorer(noise_sigma=0.05)

all_results = {}

for tech_node in TECH_NODES:
    em = AnalogEnergyModel(tech_node=tech_node)

    results = []
    for depth, width in arch_configs:
        pred_drop = scorer.predict_drop(depth, width)
        analog_acc = max(DIGITAL_ACCURACY - pred_drop, 0.0)
        energy_J = compute_network_energy(em, depth, width, INPUT_DIM, OUTPUT_DIM)
        energy_uJ = energy_J * 1e6
        efficiency = analog_acc / energy_uJ

        results.append(OrderedDict([
            ('depth', depth),
            ('width', width),
            ('predicted_accuracy_drop', pred_drop),
            ('analog_accuracy', analog_acc),
            ('energy_J', energy_J),
            ('energy_uJ', energy_uJ),
            ('efficiency', efficiency),
            ('pareto', False),
        ]))

    compute_pareto(results)
    all_results[tech_node] = results


for tech_node in TECH_NODES:
    results = all_results[tech_node]

    print(f"\n{'='*80}")
    print(f"  ENERGY BENCHMARK \u2014 Analog NN Architecture Sweep ({tech_node})")
    print(f"{'='*80}")
    header = f"{'Depth':<6} {'Width':<6} {'Pred Acc':<10} {'Drop':<8} {'Energy (µJ)':<12} {'Eff (acc/µJ)':<15} {'Pareto':<8}"
    sep = f"{'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*15} {'-'*8}"
    print(header)
    print(sep)

    for r in results:
        print(f"{r['depth']:<6} {r['width']:<6} {r['analog_accuracy']*100:>6.2f}%  "
              f"{r['predicted_accuracy_drop']*100:>5.2f}%  {r['energy_uJ']:>10.4f}  "
              f"{r['efficiency']:>13.6f}  {'Yes' if r['pareto'] else 'No':<8}")

    best_efficient = max(results, key=lambda r: r['efficiency'])
    budget_candidates = [r for r in results if r['energy_uJ'] <= 10]
    best_accurate_budget = max(budget_candidates, key=lambda r: r['analog_accuracy']) if budget_candidates else max(results, key=lambda r: r['analog_accuracy'])
    pareto_archs = [r for r in results if r['pareto']]
    best_product = min(results, key=lambda r: (1 - r['analog_accuracy']) * r['energy_J'])

    print(f"\n  === {tech_node}: Key Findings ===")
    print(f"  Most energy-efficient:")
    print(f"    D={best_efficient['depth']}, W={best_efficient['width']}: "
          f"acc={best_efficient['analog_accuracy']*100:.2f}%, "
          f"energy={best_efficient['energy_uJ']:.4f} µJ, "
          f"eff={best_efficient['efficiency']:.4f} acc/µJ")
    print(f"  Most accurate within 10µJ:")
    print(f"    D={best_accurate_budget['depth']}, W={best_accurate_budget['width']}: "
          f"acc={best_accurate_budget['analog_accuracy']*100:.2f}%, "
          f"energy={best_accurate_budget['energy_uJ']:.4f} µJ")
    print(f"  Pareto-optimal architectures ({len(pareto_archs)}):")
    for p in pareto_archs:
        print(f"    D={p['depth']}, W={p['width']}: "
              f"acc={p['analog_accuracy']*100:.2f}%, "
              f"energy={p['energy_uJ']:.4f} µJ, "
              f"eff={p['efficiency']:.4f} acc/µJ")
    print(f"  Best (1-acc)*energy product:")
    print(f"    D={best_product['depth']}, W={best_product['width']}: "
          f"acc={best_product['analog_accuracy']*100:.2f}%, "
          f"energy={best_product['energy_uJ']:.4f} µJ, "
          f"product={(1-best_product['analog_accuracy'])*best_product['energy_J']:.6e}")


print(f"\n{'='*80}")
print("  CROSS-TECH NODE COMPARISON")
print(f"{'='*80}")
cross_header = f"{'Arch':<12} {'28nm acc':<10} {'28nm µJ':<10} {'28nm eff':<12} {'14nm µJ':<10} {'7nm µJ':<10} {'7nm eff':<12}"
print(cross_header)
print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")
for depth, width in arch_configs:
    r28 = next(r for r in all_results['28nm'] if r['depth'] == depth and r['width'] == width)
    r14 = next(r for r in all_results['14nm'] if r['depth'] == depth and r['width'] == width)
    r7 = next(r for r in all_results['7nm'] if r['depth'] == depth and r['width'] == width)
    print(f"D={depth} W={width:<4} {r28['analog_accuracy']*100:>6.2f}%   "
          f"{r28['energy_uJ']:>8.4f}  {r28['efficiency']:>10.4f}   "
          f"{r14['energy_uJ']:>8.4f}  {r7['energy_uJ']:>8.4f}  {r7['efficiency']:>10.4f}")


print(f"\n{'='*80}")
print("  SUMMARY: BEST ARCHITECTURE PER NODE")
print(f"{'='*80}")
for tech_node in TECH_NODES:
    results = all_results[tech_node]
    best = max(results, key=lambda r: r['efficiency'])
    print(f"  {tech_node}: D={best['depth']}, W={best['width']}  "
          f"acc={best['analog_accuracy']*100:.2f}%  "
          f"energy={best['energy_uJ']:.4f} µJ  "
          f"eff={best['efficiency']:.4f} acc/µJ")


output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'energy_benchmark_results.json')
serializable = {}
for tech_node, results in all_results.items():
    serializable[tech_node] = []
    for r in results:
        d = dict(r)
        d['predicted_accuracy_drop'] = float(d['predicted_accuracy_drop'])
        d['analog_accuracy'] = float(d['analog_accuracy'])
        d['energy_J'] = float(d['energy_J'])
        d['energy_uJ'] = float(d['energy_uJ'])
        d['efficiency'] = float(d['efficiency'])
        d['pareto'] = bool(d['pareto'])
        serializable[tech_node].append(d)

with open(output_path, 'w') as f:
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to {output_path}")
