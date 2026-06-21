"""
SPICE Real Simulation: ngspice vs FallbackSolver Comparison
===========================================================

After fixing the op-amp model (limit -> max(min(...)) in ngspice 46),
this script runs the actual netlist through ngspice and compares
outputs with our FallbackNodalSolver.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import re
import subprocess


def train_mini_mlp(input_dim=16, hidden_dim=32, output_dim=10, seed=42):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    X = torch.randn(200, input_dim)
    y = torch.randint(0, output_dim, (200,))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    for ep in range(20):
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        opt.step()
    return model


def generate_netlist_with_control(weight, bias, x, vmax=2.5, r_ref=10000.0):
    """Generate a SPICE netlist with embedded .control block for batch output."""
    from circuit_ir.mapping import map_layer_to_circuit
    from circuit_ir.exporters.ngspice_exporter import NgspiceExporter

    circuit = map_layer_to_circuit(weight, bias, x, r_ref=r_ref, v_ref=1.0, name='layer')

    n_out = weight.shape[0]
    print_cmds = ['op']
    for i in range(n_out):
        print_cmds.append(f'print v(node_out_{i})')
    print_cmds.append(f'write {os.environ["TEMP"]}/layer_raw.raw all')

    netlist_content = NgspiceExporter.export(circuit, analysis_cmds=print_cmds, vmax=vmax)
    return netlist_content


def parse_ngspice_output(raw_path):
    """Parse ngspice raw file into dict of voltage values."""
    voltages = {}
    try:
        with open(raw_path, 'r') as f:
            content = f.read()

        # Find variables and values
        var_match = re.search(r'Variables:\n(.*?)Values:', content, re.DOTALL)
        val_match = re.search(r'Values:\n(.*)', content, re.DOTALL)

        if var_match and val_match:
            var_lines = var_match.group(1).strip().split('\n')
            val_str = val_match.group(1).strip()

            variables = []
            for line in var_lines:
                parts = line.strip().split()
                if len(parts) >= 3:
                    variables.append(parts[1])

            values = [float(v) for v in val_str.split()]

            for var, val in zip(variables, values):
                voltages[var] = val
    except Exception as e:
        print(f"  Parse error: {e}")

    return voltages


def compare_ngspice_solver():
    """Run ngspice on generated netlist, compare with solver."""
    print("=" * 70)
    print("NGSPICE vs FALLBACK SOLVER COMPARISON")
    print("=" * 70)

    # Setup
    ngspice_path = r"C:\Users\varsh\AppData\Local\Temp\ngspice_extract\Spice64\bin\ngspice.exe"
    td = os.environ['TEMP']

    # Train model
    model = train_mini_mlp(seed=42)
    W1, b1 = model[0].weight.data, model[0].bias.data
    x = torch.randn(16)

    # Generate netlist
    netlist = generate_netlist_with_control(W1, b1, x, vmax=2.5, r_ref=10000.0)
    netlist_path = os.path.join(td, 'layer0_sim.cir')
    with open(netlist_path, 'w') as f:
        f.write(netlist)

    print(f"  Netlist: {netlist_path}")
    print(f"  Netlist size: {len(netlist)} chars")

    # Run ngspice
    result = subprocess.run(
        [ngspice_path, '-b', netlist_path],
        capture_output=True, text=True, timeout=60
    )
    print(f"  ngspice exit code: {result.returncode}")
    if result.stdout:
        print(f"  ngspice stdout: {result.stdout[:200]}")
    if result.stderr:
        print(f"  ngspice stderr: {result.stderr[:200]}")

    # Parse raw file (ngspice writes raw)
    raw_path = os.path.join(td, 'layer_raw.raw')
    ngspice_v = parse_ngspice_output(raw_path)

    print(f"\n  Parsed {len(ngspice_v)} ngspice variables")
    out_keys = sorted([k for k in ngspice_v.keys() if 'node_out_' in k and not 'pos_' in k and not 'neg_' in k])
    print(f"  Output nodes found: {len(out_keys)}")
    if not out_keys:
        # Try alternate
        raw_path2 = os.path.join(td, 'layer_raw.raw').replace('/', '\\')
        if os.path.exists(raw_path2):
            ngspice_v = parse_ngspice_output(raw_path2)
            out_keys = sorted([k for k in ngspice_v.keys() if 'node_out_' in k and not 'pos_' in k and not 'neg_' in k])
            print(f"  After retry: {len(out_keys)} output nodes")

    # Fallback solver
    from spice.fallback_solver import FallbackNodalSolver
    solver_cfg = {
        'resistor_mismatch': 0.0, 'enable_mismatch': True,
        'drift_time': 0.0, 'drift_tau': 1.0, 'enable_drift': True,
        'opamp_offset': 0.0, 'enable_offset': True,
        'saturation_vmax': 2.5, 'enable_saturation': True, 'seed': 42,
    }
    solver_out = FallbackNodalSolver.solve_closed_form(W1, b1, x.unsqueeze(0), solver_cfg).squeeze(0)

    # Print comparison
    n_out = W1.shape[0]
    print(f"\n  {'Node':<20} {'ngspice':<15} {'Solver':<15} {'Diff':<15}")
    print(f"  {'-'*65}")

    diffs = []
    for i in range(min(n_out, 32)):
        ng_val = ngspice_v.get(f'v(node_out_{i})', None)
        if ng_val is None:
            # Try without v() wrapper
            ng_val = ngspice_v.get(f'node_out_{i}', None)
        if ng_val is None:
            # The raw file might have different naming
            # Check all keys for match
            for k, v in ngspice_v.items():
                if f'node_out_{i}' in k or f'node_out_{i}' in k:
                    ng_val = v
                    break

        sol_val = solver_out[i].item()
        if ng_val is not None:
            diff = abs(ng_val - sol_val)
            diffs.append(diff)
            print(f"  node_out_{i:<14} {ng_val:<15.6f} {sol_val:<15.6f} {diff:<15.2e}")
        else:
            print(f"  node_out_{i:<14} {'N/A':<15} {sol_val:<15.6f} {'N/A':<15}")

    if diffs:
        print(f"\n  {'':20} {'':15} {'':15} {'-'*15}")
        print(f"  {'':20} {'':15} {'':15} MAX: {max(diffs):.2e}")
        print(f"  {'':20} {'':15} {'':15} MEAN: {np.mean(diffs):.2e}")
        print(f"  {'':20} {'':15} {'':15} MIN:  {min(diffs):.2e}")

    match = len(diffs) > 0 and max(diffs) < 1e-3
    print(f"\n  SPICE vs SOLVER MATCH: {'YES' if match else 'NO'}")

    # Also test with all read node voltages (saturation check)
    sat_count = sum(1 for v in ngspice_v.values() if isinstance(v, (int, float)) and abs(v) > 3.0)
    print(f"  Saturated nodes (>3V): {sat_count}")

    return {
        'match': match,
        'n_outputs': len(diffs),
        'max_diff': max(diffs) if diffs else float('inf'),
        'mean_diff': float(np.mean(diffs)) if diffs else float('inf'),
        'saturated_nodes': sat_count,
    }


if __name__ == '__main__':
    result = compare_ngspice_solver()

    path = 'research_advanced/real_spice_comparison.json'
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results: {path}")
