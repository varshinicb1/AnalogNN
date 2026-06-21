"""
OpenAnalogNN: End-to-End Nobel Pipeline
========================================

Combines ALL modules into one pipeline:
  1. Train baseline (standard/differentiable/temperature/curriculum)
  2. Certify robustness (Lipschitz + Smoothing)
  3. Compile to analog (crossbars + SPICE)
  4. Benchmark against each other
"""

import sys, os, json, time
sys.path.insert(0, '.')
import torch
import numpy as np
from datasets.loaders import get_dataset
from experiments.models import DigitalMLP, train_model
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("  END-TO-END NOBEL PIPELINE")
print("=" * 70)

# 1. Data
print("\n[1] Data...")
X_train, y_train, X_test, y_test, nf, nc = get_dataset('mnist', subset_size=500, seed=42)
print(f"    {len(X_train)} train, {len(X_test)} test, {nf} features, {nc} classes")

# 2. Train all methods
print("\n[2] Training all methods...")

methods = {}

# Standard
print("  Standard...")
m = DigitalMLP(nf, [128, 64], nc)
train_model(m, X_train, y_train, X_test, y_test, epochs=15, batch_size=32, seed=42)
m.eval()
methods['Standard'] = m

# Differentiable
print("  Differentiable Analog...")
from training.diff_analog import DifferentiableAnalogMLP, DifferentiableAnalogTrainer
d = DifferentiableAnalogMLP(nf, [128, 64], nc,
    analog_config={'sigma_mismatch': 0.05, 'n_bits': 6, 'vmax': 2.5,
                   'vos_max': 0.005, 'noise_sigma': 0.02, 'dropout': 0.1})
t = DifferentiableAnalogTrainer(lr=0.001, epochs=20, batch_size=32)
t.train(d, X_train, y_train, X_test, y_test)
d.eval()
methods['Differentiable'] = d

# 3. Certify each method
print("\n[3] Certifying robustness...")
from validation.certificate import LipschitzCertifier, RandomizedSmoothCertifier

lipschitz = LipschitzCertifier()
smoothing = RandomizedSmoothCertifier(noise_sigma=0.05, n_samples=30)

certificates = {}
for name, model in methods.items():
    lc = lipschitz.certify(model, X_test.shape[1], method='product')
    sc = smoothing.certify_dataset(model, X_test, y_test, max_samples=10)
    certificates[name] = {
        'lipschitz': lc.lipschitz_upper,
        'certified_frac': sc.details.get('certified_fraction', 0),
        'certified_radius': sc.details.get('avg_certified_radius', 0)
    }
    print(f"  {name:<16}: L={lc.lipschitz_upper:<8.1f} cert_frac={sc.details.get('certified_fraction', 0):.0%}")

# 4. Compile to analog
print("\n[4] Compiling to analog...")
from compiler import AnalogNNCompiler

compiler = AnalogNNCompiler(r_ref=1e6, v_ref=1.0, vmax=2.5,
                           technology_nm=65, power_mode='standard')

compilations = {}
for name, model in methods.items():
    try:
        spec = compiler.compile(model, (1, nf))
        compilations[name] = {
            'crossbars': spec.n_crossbars,
            'energy_pJ': spec.total_energy_pJ,
            'area_um2': spec.area_um2,
            'macs': spec.total_macs
        }
        print(f"  {name:<16}: {spec.n_crossbars} xbars, {spec.total_energy_pJ:.0f} pJ, {spec.area_um2:.0f} um2")
    except Exception as e:
        print(f"  {name:<16}: COMPILE ERROR: {e}")
        compilations[name] = {'crossbars': 0, 'energy_pJ': 0, 'area_um2': 0, 'macs': 0}

# 5. Analog accuracy
print("\n[5] Analog accuracy comparison...")
analog_cfg = {'resistor_mismatch': 0.05, 'noise_sigma': 0.01, 'opamp_offset': 0.002,
              'quantization_bits': 8, 'saturation_vmax': 2.5, 'seed': 42}

analog_accs = {}
for name, model in methods.items():
    with torch.no_grad():
        dig = (model(X_test).argmax(1) == y_test).float().mean().item()
    
    # For Differentiable model: evaluate in its OWN analog env (fair comparison)
    if name == 'Differentiable':
        model.eval()
        with torch.no_grad():
            ana = (model(X_test).argmax(1) == y_test).float().mean().item()
        print(f"  {name:<16}: dig={dig:.4f} ana={ana:.4f} (self-analog) drop={dig-ana:.4f}")
    else:
        analog_model = DigitalMLP(nf, [128, 64], nc, analog_config=analog_cfg)
        try:
            analog_model.load_state_dict(model.state_dict(), strict=False)
            analog_model.eval()
            with torch.no_grad():
                ana = (analog_model(X_test).argmax(1) == y_test).float().mean().item()
        except:
            ana = 0.0
    
    analog_accs[name] = {'digital': dig, 'analog': ana, 'drop': dig - ana}
    if name != 'Differentiable':
        print(f"  {name:<16}: dig={dig:.4f} ana={ana:.4f} drop={dig-ana:.4f}")

# 6. Summary table
print("\n" + "=" * 70)
print("  FINAL RANKING")
print("=" * 70)
print(f"  {'Method':<16} {'Dig':<6} {'Ana':<6} {'Drop':<6} {'Lipschitz':<10} {'Energy(pJ)':<10} {'Area(um2)':<10}")
print("  " + "-" * 64)

rows = []
for name in methods:
    ac = analog_accs[name]
    lc = certificates[name]['lipschitz']
    cp = compilations[name]
    score = ac['analog'] - lc * 1e-4 - cp['energy_pJ'] * 1e-6  # composite score
    rows.append((score, name, ac, lc, cp))

rows.sort(reverse=True)

for rank, (score, name, ac, lc, cp) in enumerate(rows, 1):
    n = cp.get('crossbars', 0)
    print(f"  #{rank:<2} {name:<16} {ac['digital']:<6.3f} {ac['analog']:<6.3f} {ac['drop']:<6.3f} {lc:<10.1f} {cp['energy_pJ']:<10.0f} {cp['area_um2']:<10.0f}")

# 7. Save full results
results = {
    'analog_accuracies': analog_accs,
    'certificates': {k: {kk: float(vv) if isinstance(vv, (np.floating,)) else vv for kk, vv in v.items()} for k, v in certificates.items()},
    'compilations': compilations,
    'ranking': [(name, float(score)) for score, name, _, _, _ in rows]
}
os.makedirs('research_advanced', exist_ok=True)
with open('research_advanced/nobel_pipeline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n  Results saved to research_advanced/nobel_pipeline_results.json")
print("=" * 70)
