"""Verify shifted match between ngspice and solver outputs."""

import re
import numpy as np
import torch

# Parse ngspice raw file
with open(r'C:\Users\varsh\AppData\Local\Temp\layer_raw.raw') as f:
    content = f.read()

var_match = re.search(r'Variables:\n(.*?)Values:', content, re.DOTALL)
val_match = re.search(r'Values:\n(.*)', content, re.DOTALL)
var_lines = var_match.group(1).strip().split('\n')
val_str = val_match.group(1).strip()
variables = []
for line in var_lines:
    parts = line.strip().split()
    if len(parts) >= 3:
        variables.append(parts[1])
values = [float(v) for v in val_str.split()]

# Extract output nodes (names like v(node_out_0))
ngspice_out = {}
for var, val in zip(variables, values):
    if 'node_out_' in var and 'pos_' not in var and 'neg_' not in var and 'sub_' not in var:
        ngspice_out[var] = val

print('ngspice output nodes: ' + str(len(ngspice_out)))
for k, v in sorted(ngspice_out.items()):
    m = re.search(r'node_out_(\d+)', k)
    if not m:
        continue
    idx = int(m.group(1))
    if idx < 5 or idx >= 30:
        print('  ' + k + ' = ' + str(round(v, 6)))

# Get solver outputs
from research.real_spice_comparison import train_mini_mlp
from spice.fallback_solver import FallbackNodalSolver

model = train_mini_mlp(seed=42)
W1, b1 = model[0].weight.data, model[0].bias.data
x = torch.randn(16)
solver_cfg = {
    'resistor_mismatch': 0.0,
    'enable_mismatch': True,
    'drift_time': 0.0,
    'drift_tau': 1.0,
    'enable_drift': True,
    'opamp_offset': 0.0,
    'enable_offset': True,
    'saturation_vmax': 2.5,
    'enable_saturation': True,
    'seed': 42,
}
solver_out = FallbackNodalSolver.solve_closed_form(W1, b1, x.unsqueeze(0), solver_cfg).squeeze(0)
solver_out = solver_out.detach().numpy()

print('\nSolver outputs (first 5): ' + str(solver_out[:5].tolist()))
print('Solver outputs (last 5):  ' + str(solver_out[-5:].tolist()))

# Compare shifted: ngspice node_out_{i+1} vs solver_out_{i}
diffs_shifted = []
for ng_key, ng_val in sorted(ngspice_out.items()):
    m2 = re.search(r'node_out_(\d+)', ng_key)
    if not m2:
        continue
    idx = int(m2.group(1))
    if idx == 0:
        continue
    sol_idx = idx - 1
    if sol_idx < len(solver_out):
        diffs_shifted.append(abs(ng_val - solver_out[sol_idx]))

print('\nShifted comparison (ngspice[i+1] vs solver[i]):')
print('  Count: ' + str(len(diffs_shifted)))
print('  Max diff: ' + format(max(diffs_shifted), '.2e'))
print('  Mean diff: ' + format(np.mean(diffs_shifted), '.2e'))
matches = sum(1 for d in diffs_shifted if d < 1e-4)
print('  Match at 1e-4: ' + str(matches) + '/' + str(len(diffs_shifted)))
matches_1e3 = sum(1 for d in diffs_shifted if d < 1e-3)
print('  Match at 1e-3: ' + str(matches_1e3) + '/' + str(len(diffs_shifted)))

# Check node_out_0
ng0 = ngspice_out.get('v(node_out_0)', None)
if ng0 is not None:
    print('\nngspice node_out_0 = ' + str(round(ng0, 6)))
    # Compare directly with solver[0]
    print('Solver[0] = ' + str(round(float(solver_out[0]), 6)))
    print('Diff = ' + str(round(abs(ng0 - float(solver_out[0])), 6)))

# Direct comparison (same index)
diffs_direct = []
for ng_key, ng_val in sorted(ngspice_out.items()):
    m3 = re.search(r'node_out_(\d+)', ng_key)
    if not m3:
        continue
    idx = int(m3.group(1))
    if idx < len(solver_out):
        diffs_direct.append(abs(ng_val - float(solver_out[idx])))

print('\nDirect comparison (same index):')
print('  Max diff: ' + format(max(diffs_direct), '.2e'))
print('  Mean diff: ' + format(np.mean(diffs_direct), '.2e'))
