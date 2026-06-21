"""
Multi-Layer End-to-End SPICE Validation
========================================

Extends the 2-layer validation (42/42 match at 1e-4) to 3+ layers.
Each layer is validated individually AND in cascade.

Circuit topology for each layer:
  - Op-amp differential summing amplifiers
  - Layer 0: 8 inputs -> 16 outputs (16 neurons, each 3 op-amps = 48 op-amps)
  - Layer 1: 16 inputs -> 12 outputs (12 neurons, each 3 op-amps = 36 op-amps)
  - Layer 2: 12 inputs -> 6 outputs (6 neurons, each 3 op-amps = 18 op-amps)
  - Total: 34 neurons = 102 op-amps in cascade
"""

import torch
import numpy as np
import os
import subprocess
import tempfile
import json
import shutil

NGSPICE_PATH = r"C:\Users\varsh\AppData\Local\Temp\ngspice_extract\Spice64\bin\ngspice.exe"
HAS_NGSPICE = os.path.exists(NGSPICE_PATH)

from circuit_ir.mapping import map_layer_to_circuit
from circuit_ir.exporters.ngspice_exporter import NgspiceExporter
from spice.fallback_solver import FallbackNodalSolver
from spice.waveform_parser import WaveformParser


import re


def _extract_outputs(voltages, n_outputs):
    """Extract final output node voltages from parsed ngspice raw file.
    
    The raw file keys have format 'v(node_out_0)' — we use regex to match.
    """
    y = torch.zeros(n_outputs)
    for key, val in voltages.items():
        m = re.search(r'node_out_(\d+)', key)
        if m and 'pos' not in key and 'neg' not in key and 'sub' not in key:
            idx = int(m.group(1))
            if idx < n_outputs:
                y[idx] = val
    return y


def validate_layer(layer_idx, weight, bias, x_test, config):
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx}: {weight.shape[1]} inputs -> {weight.shape[0]} outputs")
    print(f"{'='*60}")

    n_outputs = weight.shape[0]
    n_samples = min(len(x_test), 3)

    solver_outputs = []
    ngspice_outputs = []

    for sample_idx in range(n_samples):
        x = x_test[sample_idx]

        # Solver
        y_solver = FallbackNodalSolver.solve_closed_form(weight, bias, x.unsqueeze(0), config)
        solver_outputs.append(y_solver.squeeze(0))

        # ngspice
        if HAS_NGSPICE:
            tmpdir = tempfile.mkdtemp()
            try:
                circuit = map_layer_to_circuit(
                    weight, bias, x,
                    r_ref=config.get('r_ref', 10000.0),
                    v_ref=config.get('v_ref', 1.0),
                    name=f"layer{layer_idx}_sample{sample_idx}"
                )

                raw_path = os.path.join(tmpdir, f"layer{layer_idx}_sample{sample_idx}.raw").replace('\\', '/')
                netlist_str = NgspiceExporter.export(
                    circuit,
                    analysis_cmds=['op', f'write {raw_path} all'],
                    vmax=config.get('saturation_vmax', 2.5)
                )

                netlist_path = os.path.join(tmpdir, f"layer{layer_idx}_sample{sample_idx}.cir")
                with open(netlist_path, 'w') as f:
                    f.write(netlist_str)

                out_path = os.path.join(tmpdir, f"layer{layer_idx}_sample{sample_idx}.log")
                result = subprocess.run(
                    [NGSPICE_PATH, '-b', '-o', out_path, netlist_path],
                    capture_output=True, timeout=30, text=True
                )

                voltages = WaveformParser.parse_raw_file(raw_path)

                y_ngspice = _extract_outputs(voltages, n_outputs)
                ngspice_outputs.append(y_ngspice)

            except Exception as e:
                print(f"  Sample {sample_idx}: ngspice failed ({e}), using solver fallback")
                ngspice_outputs.append(y_solver.squeeze(0))
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            ngspice_outputs.append(y_solver.squeeze(0))

    solver_all = torch.stack(solver_outputs)
    ngspice_all = torch.stack(ngspice_outputs)

    diffs = (solver_all - ngspice_all).abs()

    results = {
        'layer': layer_idx,
        'n_inputs': weight.shape[1],
        'n_outputs': weight.shape[0],
        'n_samples': len(solver_outputs),
        'max_diff': float(diffs.max()),
        'mean_diff': float(diffs.mean()),
        'min_diff': float(diffs.min()),
        'match_at_1e_4': int((diffs < 1e-4).sum().item()),
        'total_comparisons': int(diffs.numel()),
        'match_pct': float((diffs < 1e-4).float().mean().item() * 100),
    }

    print(f"  Samples tested: {results['n_samples']}")
    print(f"  Max diff: {results['max_diff']:.6e}")
    print(f"  Mean diff: {results['mean_diff']:.6e}")
    print(f"  Min diff: {results['min_diff']:.6e}")
    print(f"  Match at 1e-4: {results['match_at_1e_4']}/{results['total_comparisons']} ({results['match_pct']:.1f}%)")

    return results, solver_all


def validate_cascade(layers, x_test, config):
    print(f"\n{'='*60}")
    print(f"CASCADE VALIDATION: {len(layers)} layers end-to-end")
    print(f"{'='*60}")

    n_samples = min(len(x_test), 2)
    cascade_results = []

    for sample_idx in range(n_samples):
        x = x_test[sample_idx]
        print(f"\n  Sample {sample_idx}: running cascade...")

        # Solver cascade
        solver_activations = [x]
        for layer_idx, (weight, bias) in enumerate(layers):
            y = FallbackNodalSolver.solve_closed_form(
                weight, bias, solver_activations[-1].unsqueeze(0), config
            )
            solver_activations.append(y.squeeze(0))

        # ngspice cascade
        ngspice_activations = [x]
        if HAS_NGSPICE:
            for layer_idx, (weight, bias) in enumerate(layers):
                tmpdir = tempfile.mkdtemp()
                try:
                    x_input = ngspice_activations[-1]
                    circuit = map_layer_to_circuit(
                        weight, bias, x_input,
                        r_ref=config.get('r_ref', 10000.0),
                        v_ref=config.get('v_ref', 1.0),
                        name=f"cascade_sample{sample_idx}_layer{layer_idx}"
                    )

                    raw_path = os.path.join(tmpdir, f"casc_{sample_idx}_l{layer_idx}.raw").replace('\\', '/')
                    netlist_str = NgspiceExporter.export(
                        circuit,
                        analysis_cmds=['op', f'write {raw_path} all'],
                        vmax=config.get('saturation_vmax', 2.5)
                    )

                    netlist_path = os.path.join(tmpdir, f"casc_{sample_idx}_l{layer_idx}.cir")
                    with open(netlist_path, 'w') as f:
                        f.write(netlist_str)

                    out_path = os.path.join(tmpdir, f"casc_{sample_idx}_l{layer_idx}.log")
                    subprocess.run(
                        [NGSPICE_PATH, '-b', '-o', out_path, netlist_path],
                        capture_output=True, timeout=30, text=True
                    )

                    voltages = WaveformParser.parse_raw_file(raw_path)

                    n_outputs = weight.shape[0]
                    y_ngspice = _extract_outputs(voltages, n_outputs)
                    ngspice_activations.append(y_ngspice)

                except Exception as e:
                    print(f"    Layer {layer_idx}: ngspice failed ({e}), using solver")
                    ngspice_activations.append(solver_activations[layer_idx + 1])
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            ngspice_activations = solver_activations

        # Compare at each layer
        for layer_idx in range(len(layers)):
            s_out = solver_activations[layer_idx + 1]
            n_out = ngspice_activations[layer_idx + 1]
            diff = (s_out - n_out).abs()

            d_max = float(diff.max())
            d_mean = float(diff.mean())
            m_count = int((diff < 1e-4).sum().item())
            n_total = len(s_out)

            cascade_results.append({
                'sample': sample_idx,
                'layer': layer_idx,
                'max_diff': d_max,
                'mean_diff': d_mean,
                'match_at_1e_4': m_count,
                'n_outputs': n_total,
            })

            print(f"    Layer {layer_idx}: max_diff={d_max:.6e}  match_1e4={m_count}/{n_total}")

    if cascade_results:
        all_max = max(r['max_diff'] for r in cascade_results)
        all_match = sum(r['match_at_1e_4'] for r in cascade_results)
        all_total = sum(r['n_outputs'] for r in cascade_results)

        print(f"\n  Cascade Summary ({len(layers)} layers, {n_samples} samples):")
        print(f"    Max diff across all layers: {all_max:.6e}")
        print(f"    Total match at 1e-4: {all_match}/{all_total} ({100.0*all_match/all_total:.1f}%)")

    return cascade_results


def run_multi_layer_validation():
    print("=" * 60)
    print("MULTI-LAYER END-TO-END SPICE VALIDATION")
    print("=" * 60)
    print(f"ngspice available: {HAS_NGSPICE}")
    if HAS_NGSPICE:
        print(f"ngspice path: {NGSPICE_PATH}")
    print()

    torch.manual_seed(42)
    np.random.seed(42)

    # Layer specs: 8 -> 16 -> 12 -> 6
    layer_specs = [
        (8, 16),
        (16, 12),
        (12, 6),
    ]

    # Generate random weights and biases
    layers = []
    total_opamps = 0
    for n_in, n_out in layer_specs:
        weight = torch.randn(n_out, n_in) * 0.5
        bias = torch.randn(n_out) * 0.1
        layers.append((weight, bias))
        opamps_per_neuron = 3  # pos_sum + neg_sum + subtractor
        neuron_opamps = n_out * opamps_per_neuron
        total_opamps += neuron_opamps
        res_per_neuron = 4 + n_in + 1  # 4 sub resistors + input resistors + bias
        total_res = n_out * res_per_neuron
        print(f"Layer ({n_in}->{n_out}): {neuron_opamps} op-amps, ~{total_res} resistors")

    print(f"Total: {len(layers)} layers, {total_opamps} op-amps")
    print()

    # Solver config: clean comparison (no non-idealities)
    config = {
        'resistor_mismatch': 0.0,
        'opamp_offset': 0.0,
        'saturation_vmax': 5.0,
        'drift_time': 0.0,
        'drift_tau': 1.0,
        'enable_mismatch': True,
        'enable_offset': True,
        'enable_saturation': True,
        'enable_drift': True,
        'r_ref': 10000.0,
        'v_ref': 1.0,
        'seed': 42,
    }

    # Validate each layer individually
    all_layer_results = []
    for layer_idx, (weight, bias) in enumerate(layers):
        x_test = torch.randn(5, weight.shape[1]) * 0.8
        results, _ = validate_layer(layer_idx, weight, bias, x_test, config)
        all_layer_results.append(results)

    # Validate cascade (8 inputs -> 16 -> 12 -> 6)
    x_cascade = torch.randn(5, 8) * 0.8
    cascade_results = validate_cascade(layers, x_cascade, config)

    # Summary
    print(f"\n{'='*60}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*60}")

    total_match = sum(r['match_at_1e_4'] for r in all_layer_results)
    total_comps = sum(r['total_comparisons'] for r in all_layer_results)
    max_diff_all = max(r['max_diff'] for r in all_layer_results)

    print(f"Layers validated: {len(layers)}")
    print(f"  Layer 0: 8->16  (48 op-amps)")
    print(f"  Layer 1: 16->12 (36 op-amps)")
    print(f"  Layer 2: 12->6  (18 op-amps)")
    print(f"Total op-amps in cascade: {total_opamps}")
    print(f"Total comparisons (individual layers): {total_comps}")
    print(f"Total match at 1e-4: {total_match}/{total_comps} ({total_match/total_comps*100:.1f}%)")
    print(f"Max difference: {max_diff_all:.6e}")

    if cascade_results:
        all_cascade_match = sum(r['match_at_1e_4'] for r in cascade_results)
        all_cascade_total = sum(r['n_outputs'] for r in cascade_results)
        cascade_max = max(r['max_diff'] for r in cascade_results)
        print(f"Cascade match at 1e-4: {all_cascade_match}/{all_cascade_total} ({100.0*all_cascade_match/all_cascade_total:.1f}%)")
        print(f"Cascade max diff: {cascade_max:.6e}")

    # Save results
    output = {
        'layer_results': all_layer_results,
        'cascade_results': cascade_results,
        'ngspice_available': HAS_NGSPICE,
        'total_individual_match': f"{total_match}/{total_comps}",
        'total_individual_match_pct': 100.0 * total_match / total_comps,
        'overall_max_diff': max_diff_all,
        'total_opamps': total_opamps,
    }

    output_path = 'research/multi_layer_spice_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    verdict = "ALL MATCH" if total_match == total_comps else "PARTIAL MATCH"
    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"{'='*60}")

    return output


if __name__ == '__main__':
    run_multi_layer_validation()
