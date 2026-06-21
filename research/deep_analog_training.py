"""
Deep Analog Neural Network Training: Multi-architecture, Multi-mode Comparison
==============================================================================
Trains synthetic 10-class datasets with CIFAR-10 and SVHN dimensions,
on 3 architectures x 3 modes each.
Measures accuracy, energy, and efficiency.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.models import DigitalMLP, train_model, evaluate_model
from analog_layers.analog_linear import AnalogLinear
from energy.analog_energy_model import AnalogEnergyModel


ARCHITECTURES = {
    'S': [128],
    'M': [256, 128],
    'L': [512, 256, 128],
}

DATASET_CONFIGS = [
    ('CIFAR10', 768, 10),
    ('SVHN', 256, 10),
]

EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
SEED = 42
N_TRAIN = 1000
N_TEST = 200

STD_DEPLOY_CONFIG = {'noise_sigma': 0.05, 'resistor_mismatch': 0.01}
DIFF_ANALOG_CONFIG = {'noise_sigma': 0.03}


class AnalogDeploymentModel(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.network = sequential

    def forward(self, x):
        return self.network(x)


def build_analog_from_digital(digital_model, analog_config):
    layers = []
    for module in digital_model.network:
        if isinstance(module, nn.Linear):
            layers.append(AnalogLinear.from_digital(module, analog_config))
        else:
            layers.append(module)
    return AnalogDeploymentModel(nn.Sequential(*layers))


def compute_model_energy(model, sample_input, energy_model):
    total_energy = 0.0
    x = sample_input
    with torch.no_grad():
        for module in model.network:
            if isinstance(module, (nn.Linear, AnalogLinear)):
                result = energy_model.full_layer_energy(module.weight, x)
                total_energy += result['energy_per_inference_J']
            x = module(x)
    return total_energy


def make_synthetic_data(num_features, num_classes, n_train, n_test, seed=42):
    """Generate Gaussian cluster data with class-specific means."""
    rng = np.random.RandomState(seed)
    X_all = np.zeros((n_train + n_test, num_features), dtype=np.float32)
    y_all = np.zeros(n_train + n_test, dtype=np.int64)
    for i in range(n_train + n_test):
        label = i % num_classes
        mean = np.full(num_features, label * 0.4 / num_classes)
        cov = 0.25 + 0.08 * label
        X_all[i] = rng.randn(num_features) * cov + mean
        y_all[i] = label
    X_train = torch.tensor(X_all[:n_train])
    y_train = torch.tensor(y_all[:n_train])
    X_test = torch.tensor(X_all[n_train:])
    y_test = torch.tensor(y_all[n_train:])
    return X_train, y_train, X_test, y_test


def run():
    energy_model = AnalogEnergyModel(tech_node='28nm')
    all_results = []

    for dataset_name, num_features, num_classes in DATASET_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Generating {dataset_name} ({num_features} features, {num_classes} classes)...")
        print(f"{'='*80}")

        X_train, y_train, X_test, y_test = make_synthetic_data(
            num_features, num_classes, N_TRAIN, N_TEST, seed=SEED
        )
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Class distribution: train={np.bincount(y_train.numpy())}")

        sample = X_test[:BATCH_SIZE]
        input_dim = X_train.shape[1]

        for arch_name, hidden_dims in ARCHITECTURES.items():
            print(f"\n  --- Architecture {arch_name}: {hidden_dims} ---")

            # Mode 1: Digital Baseline
            print(f"    Mode: Digital Baseline")
            model_digital = DigitalMLP(input_dim, hidden_dims, num_classes, analog_config=None)
            t0 = time.time()
            train_model(model_digital, X_train, y_train, X_test, y_test,
                        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=SEED)
            t_digital = time.time() - t0
            eval_result = evaluate_model(model_digital, X_test, y_test)
            acc_digital = eval_result['accuracy']

            all_results.append({
                'dataset': dataset_name, 'architecture': arch_name,
                'mode': 'Digital', 'train_time_s': round(float(t_digital), 2),
                'accuracy': round(float(acc_digital) * 100, 1),
                'energy_uj': 'N/A', 'efficiency': 'N/A',
            })
            print(f"      Acc: {acc_digital*100:.1f}% | Time: {t_digital:.2f}s")

            # Mode 2: Standard Deploy (train digital, deploy analog)
            print(f"    Mode: StdDeploy (noise=0.05, mismatch=0.01)")
            analog_deploy = build_analog_from_digital(model_digital, STD_DEPLOY_CONFIG)
            eval_std = evaluate_model(analog_deploy, X_test, y_test)
            acc_std = eval_std['accuracy']
            energy_std = compute_model_energy(analog_deploy, sample, energy_model)
            energy_std_uj = energy_std * 1e6
            eff_std = (acc_std * 100) / energy_std_uj if energy_std_uj > 0 else 0

            all_results.append({
                'dataset': dataset_name, 'architecture': arch_name,
                'mode': 'StdDeploy', 'train_time_s': 0,
                'accuracy': round(float(acc_std) * 100, 1),
                'energy_uj': round(float(energy_std_uj), 4),
                'efficiency': round(float(eff_std), 2),
            })
            print(f"      Acc: {acc_std*100:.1f}% | Energy: {energy_std_uj:.4f} uJ | Eff: {eff_std:.2f}")

            # Mode 3: Differentiable Analog MLP (train-through analog)
            print(f"    Mode: DiffAnalo (noise=0.03, train-through)")
            model_analog = DigitalMLP(input_dim, hidden_dims, num_classes,
                                      analog_config=DIFF_ANALOG_CONFIG)
            t0 = time.time()
            train_model(model_analog, X_train, y_train, X_test, y_test,
                        epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, seed=SEED)
            t_analog = time.time() - t0
            eval_analog = evaluate_model(model_analog, X_test, y_test)
            acc_analog = eval_analog['accuracy']
            energy_analog = compute_model_energy(model_analog, sample, energy_model)
            energy_analog_uj = energy_analog * 1e6
            eff_analog = (acc_analog * 100) / energy_analog_uj if energy_analog_uj > 0 else 0

            all_results.append({
                'dataset': dataset_name, 'architecture': arch_name,
                'mode': 'DiffAnalo', 'train_time_s': round(float(t_analog), 2),
                'accuracy': round(float(acc_analog) * 100, 1),
                'energy_uj': round(float(energy_analog_uj), 4),
                'efficiency': round(float(eff_analog), 2),
            })
            print(f"      Acc: {acc_analog*100:.1f}% | Time: {t_analog:.2f}s | Energy: {energy_analog_uj:.4f} uJ | Eff: {eff_analog:.2f}")

    # === Comparison Table ===
    print(f"\n\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    header = f"{'Dataset':<10} {'Arch':<6} {'Mode':<12} {'Acc %':<8} {'Energy (uJ)':<13} {'Eff (acc/uJ)':<13}"
    sep = f"{'-'*10} {'-'*6} {'-'*12} {'-'*8} {'-'*13} {'-'*13}"
    print(header)
    print(sep)
    for r in all_results:
        acc = f"{r['accuracy']:.1f}"
        e = f"{r['energy_uj']:.4f}" if r['energy_uj'] != 'N/A' else '-'
        eff = f"{r['efficiency']:.2f}" if r['efficiency'] != 'N/A' else '-'
        print(f"{r['dataset']:<10} {r['architecture']:<6} {r['mode']:<12} {acc:<8} {e:<13} {eff:<13}")

    # === Summary: Ranked by Efficiency ===
    print(f"\n\n{'='*100}")
    print("SUMMARY - Ranked by Energy-Accuracy Efficiency")
    print(f"{'='*100}")
    analog_entries = [r for r in all_results if r['efficiency'] != 'N/A']
    analog_entries.sort(key=lambda r: r['efficiency'], reverse=True)
    header2 = f"{'Rank':<6} {'Dataset':<10} {'Arch':<6} {'Mode':<12} {'Acc %':<8} {'Energy (uJ)':<13} {'Eff (acc/uJ)':<13}"
    print(header2)
    print(sep)
    for i, r in enumerate(analog_entries, 1):
        acc = f"{r['accuracy']:.1f}"
        e = f"{r['energy_uj']:.4f}"
        eff = f"{r['efficiency']:.2f}"
        print(f"{i:<6} {r['dataset']:<10} {r['architecture']:<6} {r['mode']:<12} {acc:<8} {e:<13} {eff:<13}")

    # Save
    output_path = Path(__file__).parent / 'deep_analog_training_results.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == '__main__':
    run()
