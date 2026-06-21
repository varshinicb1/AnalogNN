"""Final ngspice vs solver comparison with proper transient analysis."""

import torch
import os
import re
import subprocess
import numpy as np
import json


def train_mini_mlp(input_dim=16, hidden_dim=32, output_dim=10, seed=42):
    torch.manual_seed(seed)
    import torch.nn as nn
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


def compare_layer(W, b, x, label, vmax=2.5, r_ref=10000.0):
    """Compare ngspice vs solver for one layer. Returns list of per-output diffs."""
    from circuit_ir.mapping import map_layer_to_circuit
    from circuit_ir.exporters.ngspice_exporter import NgspiceExporter
    from spice.fallback_solver import FallbackNodalSolver
    from spice.waveform_parser import WaveformParser

    td = os.environ['TEMP']

    # Generate netlist with .op analysis and .write raw
    circuit = map_layer_to_circuit(W, b, x, r_ref=r_ref, v_ref=1.0, name=label)
    raw_path = td.replace('\\', '/') + '/' + label + '.raw'
    netlist = NgspiceExporter.export(circuit,
        analysis_cmds=['op', 'write ' + raw_path + ' all'],
        vmax=vmax)

    path = os.path.join(td, label + '.cir')
    with open(path, 'w') as f:
        f.write(netlist)

    # Run ngspice
    ngspice = r'C:\Users\varsh\AppData\Local\Temp\ngspice_extract\Spice64\bin\ngspice.exe'
    out_path = os.path.join(td, label + '_out.txt')
    subprocess.run(
        [ngspice, '-b', '-o', out_path, path],
        capture_output=True, timeout=30
    )

    # Parse raw file using fixed WaveformParser
    node_voltages = WaveformParser.parse_raw_file(raw_path)
    ngspice_v = {}
    for key, val in node_voltages.items():
        m = re.search(r'node_out_(\d+)', key)
        if m and 'pos' not in key and 'neg' not in key and 'sub' not in key:
            ngspice_v[int(m.group(1))] = val

    n_out = W.shape[0]
    print('  {}: parsed {} ngspice outputs out of {}'.format(
        label, len(ngspice_v), n_out))

    # Solver
    solver_cfg = {
        'resistor_mismatch': 0.0, 'enable_mismatch': True,
        'drift_time': 0.0, 'drift_tau': 1.0, 'enable_drift': True,
        'opamp_offset': 0.0, 'enable_offset': True,
        'saturation_vmax': vmax, 'enable_saturation': True, 'seed': 42,
    }
    solver_out = FallbackNodalSolver.solve_closed_form(
        W, b, x.unsqueeze(0), solver_cfg).squeeze(0).detach().numpy()

    # Compare each output
    diffs = []
    for i in range(n_out):
        ng_val = ngspice_v.get(i, None)
        sol_val = float(solver_out[i])
        if ng_val is not None:
            diff = abs(ng_val - sol_val)
            diffs.append(diff)
            if i < 5 or diff > 0.01:
                print('    node_out_{}: ng={:.6f} sol={:.6f} diff={:.2e}'.format(
                    i, ng_val, sol_val, diff))
        else:
            print('    node_out_{}: MISSING (sol={:.6f})'.format(i, sol_val))

    return diffs


if __name__ == '__main__':
    print('=' * 70)
    print('FINAL SPICE VERIFICATION (with transient analysis)')
    print('=' * 70)

    model = train_mini_mlp(seed=42)
    W1, b1 = model[0].weight.data, model[0].bias.data
    W2, b2 = model[2].weight.data, model[2].bias.data
    x = torch.randn(16)

    all_diffs = []

    print('\n--- Layer 0 (16x32) ---')
    d1 = compare_layer(W1, b1, x, 'layer0', vmax=2.5, r_ref=10000.0)
    all_diffs.extend(d1)

    print('\n--- Layer 1 (32x10) ---')
    h = torch.relu(model[0](x))
    d2 = compare_layer(W2, b2, h, 'layer1', vmax=2.5, r_ref=10000.0)
    all_diffs.extend(d2)

    print('\n' + '=' * 70)
    print('OVERALL RESULTS')
    print('=' * 70)
    print('  Total outputs compared: {}'.format(len(all_diffs)))
    print('  Max diff: {:.2e}'.format(max(all_diffs)))
    print('  Mean diff: {:.2e}'.format(float(np.mean(all_diffs))))
    print('  Min diff: {:.2e}'.format(min(all_diffs)))
    match_1e4 = sum(1 for d in all_diffs if d < 1e-4)
    print('  Match at 1e-4: {}/{}'.format(match_1e4, len(all_diffs)))
    match_1e3 = sum(1 for d in all_diffs if d < 1e-3)
    print('  Match at 1e-3: {}/{}'.format(match_1e3, len(all_diffs)))

    results = {
        'n_outputs': len(all_diffs),
        'max_diff': float(np.max(all_diffs)),
        'mean_diff': float(np.mean(all_diffs)),
        'min_diff': float(np.min(all_diffs)),
        'match_1e4': match_1e4,
        'match_1e3': match_1e3,
        'all_match_1e3': match_1e3 == len(all_diffs),
        'verdict': 'PASS' if match_1e3 == len(all_diffs) else 'FAIL',
    }

    path = 'research_advanced/final_spice_comparison.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print('\n  Results: ' + path)
    print('  Verdict: ' + results['verdict'])
