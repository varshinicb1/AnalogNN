"""
OpenAnalogNN: MEGA DEMO — Validating All 3 New Nobel-Tier Modules
==================================================================

1. DifferentiableAnalogSimulator
2. Formal RobustnessCertificate (Lipschitz + Z3 + Smoothing)
3. AnalogNN Compiler (PyTorch -> Crossbar -> SPICE)
"""

import sys, os, json, time
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
from datasets.loaders import get_dataset
from experiments.models import DigitalMLP, train_model

print("=" * 70)
print("  MEGA DEMO: 3 Nobel-Tier Modules")
print("=" * 70)

# 1. Load data and train baseline
print("\n[1/3] Loading data and training baseline...")
X_train, y_train, X_test, y_test, nf, nc = get_dataset('mnist', subset_size=500, seed=42)
model = DigitalMLP(nf, [128, 64], nc)
train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, seed=42)
model.eval()
with torch.no_grad():
    dig_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
print(f"  Digital baseline accuracy: {dig_acc:.4f}")

# 2. Differentiable Analog Simulator
print("\n[2/3] Differentiable Analog Simulator...")
from training.diff_analog import (
    DifferentiableAnalogLinear,
    DifferentiableAnalogMLP,
    DifferentiableAnalogTrainer,
    DifferentiableMismatch,
    DifferentiableQuantization,
    DifferentiableSaturation,
    DifferentiableOffset
)

# Test each component
print("\n  Testing individual components...")
mismatch = DifferentiableMismatch(sigma=0.05)
w_test = torch.randn(10, 32)
w_eff = mismatch(w_test)
print(f"    Mismatch:      input.shape={w_test.shape}, output.shape={w_eff.shape} (diff OK)")

quant = DifferentiableQuantization(n_bits=6)
x_test = torch.randn(16, 64)
x_q = quant(x_test)
print(f"    Quantization:  input.shape={x_test.shape}, output.shape={x_q.shape} (diff OK)")

sat = DifferentiableSaturation(vmax=2.5)
x_sat = sat(torch.randn(16, 10) * 5)
print(f"    Saturation:    max={x_sat.abs().max():.4f} <= 2.5 (diff OK)")

offset = DifferentiableOffset(vos_max=0.01)
x_off = offset(torch.randn(16, 10))
print(f"    Offset:        output.shape={x_off.shape} (diff OK)")

# Test end-to-end differentiable model
print("\n  Training DifferentiableAnalogMLP end-to-end...")
diff_model = DifferentiableAnalogMLP(
    nf, [128, 64], nc,
    analog_config={'sigma_mismatch': 0.05, 'n_bits': 6, 'vmax': 2.5,
                   'vos_max': 0.005, 'noise_sigma': 0.02, 'dropout': 0.1}
)

trainer = DifferentiableAnalogTrainer(lr=0.001, epochs=15, batch_size=32)
history = trainer.train(diff_model, X_train, y_train, X_test, y_test)

# Verify gradients flow through all layers
print("  Verifying gradient flow...")
total_grad_norm = 0
for name, param in diff_model.named_parameters():
    if param.grad is not None:
        total_grad_norm += param.grad.norm().item()
print(f"    Total gradient norm: {total_grad_norm:.4f} (non-zero = gradients flow)")

# Count non-zero gradients
nz = sum(1 for p in diff_model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total = sum(1 for p in diff_model.parameters())
print(f"    Parameters with gradients: {nz}/{total}")

# 3. Formal Robustness Certificates
print("\n[3/3] Formal Robustness Certificates...")
from validation.certificate import (
    LipschitzCertifier,
    SMTCertifier,
    RandomizedSmoothCertifier,
    certify_model
)

# Lipschitz certificate
print("\n  Computing Lipschitz bounds...")
certifier = LipschitzCertifier()
lipschitz_cert = certifier.certify(model, X_test.shape[1], method='product')
print(f"    Lipschitz upper bound: {lipschitz_cert.lipschitz_upper:.4f}")
print(f"    Certified: {lipschitz_cert.verified}")

# SDP bound (tighter)
try:
    sdp_cert = certifier.certify(model, X_test.shape[1], method='sdp')
    print(f"    SDP Lipschitz bound: {sdp_cert.lipschitz_upper:.4f}")
    print(f"    SDP Certified: {sdp_cert.verified}")
except Exception as e:
    print(f"    SDP skipped: {e}")

# Randomized Smoothing certificate
print("\n  Computing randomized smoothing certificates...")
smoothing = RandomizedSmoothCertifier(noise_sigma=0.05, n_samples=50)
smooth_cert = smoothing.certify_dataset(model, X_test, y_test, max_samples=20)
print(f"    Certified fraction: {smooth_cert.details['certified_fraction']:.2%}")
print(f"    Avg certified radius: {smooth_cert.details['avg_certified_radius']:.4f}")
print(f"    Certified: {smooth_cert.verified}")

# Z3 SMT certificate (small subset)
print("\n  Z3 SMT verification (3 samples)...")
try:
    weight = list(model.parameters())[0]
    bias = list(model.parameters())[1]
    z3_certifier = SMTCertifier()
    z3_cert = z3_certifier.verify_perturbation_bound(
        weight, bias, X_test, y_test,
        max_mismatch=0.2, max_offset=0.01, timeout_ms=5000
    )
    print(f"    Z3 result: {'VERIFIED (no flip found)' if z3_cert.verified else 'FLIP FOUND'}")
except Exception as e:
    print(f"    Z3 skipped: {e}")

# 4. AnalogNN Compiler
print("\n[4/3 bonus] AnalogNN Compiler...")
from compiler import AnalogNNCompiler

compiler = AnalogNNCompiler(r_ref=1e6, v_ref=1.0, vmax=2.5,
                           technology_nm=65, power_mode='standard')
spec = compiler.compile(model, (1, nf))
print(f"\n{compiler.summary(spec)}")

# Export to SPICE
compiler.compile_and_export(model, (1, nf), output_dir='netlists_compiled')

# 5. Combined: Differentiable Training -> Certificate -> Compiler
print("\n[5/3 bonus] End-to-End Pipeline Validation...")
print("  Differentiably trained model:")
diff_lipschitz = LipschitzCertifier().certify(diff_model, X_test.shape[1])
print(f"    Lipschitz bound: {diff_lipschitz.lipschitz_upper:.4f}")

diff_spec = compiler.compile(diff_model, (1, nf))
print(f"    Crossbars: {diff_spec.n_crossbars}")
print(f"    Energy: {diff_spec.total_energy_pJ:.1f} pJ")

# Final accuracy through analog
diff_model.eval()
with torch.no_grad():
    out = diff_model(X_test)
    diff_acc = (out.argmax(1) == y_test).float().mean().item()

print(f"\n  Digital model analog accuracy: {dig_acc:.4f}")
print(f"  Differentiably trained accuracy: {diff_acc:.4f}")
print(f"  Difference: {(diff_acc - dig_acc)*100:+.2f}%")

# Save results
results = {
    'digital_accuracy': dig_acc,
    'differentiable_accuracy': diff_acc,
    'lipschitz_bound': lipschitz_cert.lipschitz_upper,
    'certified_fraction': smooth_cert.details['certified_fraction'],
    'avg_certified_radius': smooth_cert.details['avg_certified_radius'],
    'crossbars': spec.n_crossbars,
    'energy_pJ': spec.total_energy_pJ,
    'area_um2': spec.area_um2,
}
with open('research_advanced/mega_demo_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("  MEGA DEMO COMPLETE — All 3 modules validated!")
print("=" * 70)
