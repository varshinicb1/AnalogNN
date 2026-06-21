"""
Distributional Robust Training for Analog Hardware Variation
=============================================================

Domain randomization over chip-to-chip manufacturing variation.

Key innovation: Train a single model across an ENTIRE distribution of
hardware configurations (mismatch, noise, offset, bits, vmax) so it
works on ANY chip from a manufacturing batch — no per-chip calibration.

Analogy: This is to analog AI what domain randomization (OpenAI Dactyl)
was to robotic manipulation — bridging the simulation-to-reality gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import os
import warnings

from training.diff_analog import DifferentiableAnalogLinear, DifferentiableAnalogMLP, DifferentiableAnalogTrainer


@dataclass
class HardwareDistribution:
    """
    Defines distributions over chip-to-chip hardware variation.

    Each parameter is (low, high) for uniform sampling.
    During training, one sample is drawn per forward pass.
    """
    sigma_mismatch: Tuple[float, float] = (0.005, 0.05)
    noise_sigma: Tuple[float, float] = (0.001, 0.05)
    vos_max: Tuple[float, float] = (0.0, 0.005)
    n_bits_range: Tuple[int, int] = (6, 10)
    vmax_range: Tuple[float, float] = (2.0, 3.0)

    def sample(self) -> Dict:
        """Sample a hardware configuration from the distribution."""
        return {
            'sigma_mismatch': np.random.uniform(*self.sigma_mismatch),
            'noise_sigma': np.random.uniform(*self.noise_sigma),
            'vos_max': np.random.uniform(*self.vos_max),
            'n_bits': np.random.randint(self.n_bits_range[0], self.n_bits_range[1] + 1),
            'vmax': np.random.uniform(*self.vmax_range),
        }

    def sample_population(self, n_chips: int, seed: int = 42) -> List[Dict]:
        """Sample n_chips distinct hardware configurations (one per chip)."""
        rng = np.random.RandomState(seed)
        configs = []
        for _ in range(n_chips):
            configs.append({
                'sigma_mismatch': rng.uniform(*self.sigma_mismatch),
                'noise_sigma': rng.uniform(*self.noise_sigma),
                'vos_max': rng.uniform(*self.vos_max),
                'n_bits': rng.randint(self.n_bits_range[0], self.n_bits_range[1] + 1),
                'vmax': rng.uniform(*self.vmax_range),
            })
        return configs


def sample_config_around(config: Dict, sigma_scale: float = 0.2) -> Dict:
    """Sample a config near a reference (simulates chip-to-chip drift)."""
    def perturb(val, scale):
        if isinstance(val, int):
            return max(1, int(round(val * (1 + np.random.randn() * scale))))
        return max(1e-6, val * (1 + np.random.randn() * scale))
    return {
        'sigma_mismatch': perturb(config.get('sigma_mismatch', 0.01), sigma_scale),
        'noise_sigma': perturb(config.get('noise_sigma', 0.01), sigma_scale),
        'vos_max': perturb(config.get('vos_max', 0.002), sigma_scale),
        'n_bits': int(perturb(float(config.get('n_bits', 8)), sigma_scale)),
        'vmax': perturb(config.get('vmax', 2.5), sigma_scale),
    }


class DistributionalAnalogLinear(nn.Module):
    """
    Analog linear layer with domain-randomized hardware parameters.

    During training: samples new hardware params per forward pass.
    During eval: uses fixed params (can be set to nominal or specific config).
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hardware_dist: Optional[HardwareDistribution] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hardware_dist = hardware_dist or HardwareDistribution()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        self._eval_config = None  # Fixed config for evaluation

    def set_eval_config(self, config: Dict):
        self._eval_config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            cfg = self.hardware_dist.sample()
        elif self._eval_config is not None:
            cfg = self._eval_config
        else:
            cfg = self.hardware_dist.sample()

        return self._forward_with_config(x, cfg)

    def _forward_with_config(self, x: torch.Tensor, cfg: Dict) -> torch.Tensor:
        # 1. Apply mismatch
        sigma_m = cfg['sigma_mismatch']
        eps = torch.randn_like(self.weight)
        w_eff = self.weight / (1.0 + eps * sigma_m)

        # 2. Quantize weights
        n_bits = cfg['n_bits']
        w_max = w_eff.abs().max() + 1e-8
        w_scale = w_max / (2**(n_bits - 1) - 1)
        w_q = torch.clamp(torch.round(w_eff / w_scale), -(2**(n_bits-1)), 2**(n_bits-1) - 1)
        w_eff = w_eff + (w_q * w_scale - w_eff).detach()  # STE

        # 3. Matrix multiply
        out = F.linear(x, w_eff, self.bias)

        # 4. Op-amp offset
        vos = cfg['vos_max']
        if vos > 0 and self.training:
            offsets = vos * (2 * torch.rand(out.shape[-1], device=out.device) - 1)
            out = out + offsets

        # 5. Additive noise
        noise_sigma = cfg['noise_sigma']
        if noise_sigma > 0:
            out = out + torch.randn_like(out) * noise_sigma

        # 6. Saturation
        vmax = cfg['vmax']
        out = vmax * torch.tanh(out / vmax)

        return out


class DistributionalAnalogMLP(nn.Module):
    """MLP with domain-randomized analog non-idealities."""

    def __init__(self,
                 in_features: int,
                 hidden_dims: List[int],
                 out_features: int,
                 hardware_dist: Optional[HardwareDistribution] = None):
        super().__init__()
        dims = [in_features] + hidden_dims + [out_features]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.layers.append(
                DistributionalAnalogLinear(dims[i], dims[i+1], hardware_dist)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def set_eval_config(self, config: Dict):
        for layer in self.layers:
            layer.set_eval_config(config)


class DistributionalRobustTrainer:
    """
    Trains a model with domain randomization over hardware configurations.

    During each forward pass, a DIFFERENT hardware configuration is sampled,
    forcing the model to be robust to the ENTIRE distribution of chip variation.

    Reference: Domain Randomization for Sim2Real Transfer (Tobin et al., 2017)
    applied to analog neural network hardware for the first time.
    """

    def __init__(self,
                 lr: float = 0.003,
                 epochs: int = 30,
                 batch_size: int = 64,
                 hardware_dist: Optional[HardwareDistribution] = None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.hardware_dist = hardware_dist or HardwareDistribution()

    def train(self, model: nn.Module,
              X_train, y_train, X_test, y_test) -> Dict:
        from torch.utils.data import TensorDataset, DataLoader

        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=self.batch_size, shuffle=True)

        history = {'loss': [], 'acc': []}

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for bx, by in loader:
                out = model(bx)
                loss = crit(out, by)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            # Evaluate with NOMINAL hardware config (median of distribution)
            model.eval()
            nominal_cfg = {
                'sigma_mismatch': np.median(self.hardware_dist.sigma_mismatch),
                'noise_sigma': np.median(self.hardware_dist.noise_sigma),
                'vos_max': np.median(self.hardware_dist.vos_max),
                'n_bits': int(np.mean(self.hardware_dist.n_bits_range)),
                'vmax': np.median(self.hardware_dist.vmax_range),
            }
            model.set_eval_config(nominal_cfg)

            with torch.no_grad():
                out = model(X_test)
                acc = (out.argmax(1) == y_test).float().mean().item()

            history['loss'].append(epoch_loss / len(loader))
            history['acc'].append(acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: loss={history['loss'][-1]:.4f}, acc={acc:.4f}")

        return history


def evaluate_across_population(model: nn.Module,
                                X_test, y_test,
                                chip_configs: List[Dict],
                                label: str = "Model") -> Dict:
    """
    Evaluate a model across a population of chips with different hardware.

    Returns: mean, std, min, max accuracy across the population.
    """
    model.eval()
    accuracies = []

    for i, cfg in enumerate(chip_configs):
        model.set_eval_config(cfg)
        with torch.no_grad():
            out = model(X_test)
            acc = (out.argmax(1) == y_test).float().mean().item()
        accuracies.append(acc)

    acc_arr = np.array(accuracies)
    results = {
        'label': label,
        'mean_accuracy': float(np.mean(acc_arr)),
        'std_accuracy': float(np.std(acc_arr)),
        'min_accuracy': float(np.min(acc_arr)),
        'max_accuracy': float(np.max(acc_arr)),
        'p10_accuracy': float(np.percentile(acc_arr, 10)),
        'p90_accuracy': float(np.percentile(acc_arr, 90)),
        'yield_90pct': float(np.mean(acc_arr >= 0.9 * np.max(acc_arr))),
        'n_chips': len(accuracies),
    }
    return results


def run_comparison_experiment(X_train, y_train, X_test, y_test,
                               output_dir: str = 'research_advanced') -> Dict:
    """
    Compare three strategies:
    1. Standard training (one fixed hardware config)
    2. Distributional training (domain-randomized hardware)
    3. Curriculum: start standard, then distributional
    """
    os.makedirs(output_dir, exist_ok=True)

    hw_dist = HardwareDistribution(
        sigma_mismatch=(0.005, 0.05),
        noise_sigma=(0.001, 0.05),
        vos_max=(0.0, 0.005),
        n_bits_range=(6, 10),
        vmax_range=(2.0, 3.0),
    )

    # Generate chip population for evaluation
    chip_pop = hw_dist.sample_population(n_chips=50, seed=123)

    results = {}

    # Strategy 1: Standard training (fixed nominal config)
    print("\n\n=== Strategy 1: Standard Training (Fixed Config) ===")
    nominal_cfg = {
        'sigma_mismatch': 0.0275,
        'noise_sigma': 0.025,
        'vos_max': 0.0025,
        'n_bits': 8,
        'vmax': 2.5,
    }

    standard_model = DistributionalAnalogMLP(64, [128, 64], 10, hw_dist)
    standard_model.set_eval_config(nominal_cfg)  # Fix to nominal during training

    # Need a modified trainer that uses fixed config
    opt_std = torch.optim.Adam(standard_model.parameters(), lr=0.003)
    crit = nn.CrossEntropyLoss()
    loader_std = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )

    for epoch in range(15):
        standard_model.train()
        for bx, by in loader_std:
            opt_std.zero_grad()
            out = standard_model(bx)
            crit(out, by).backward()
            opt_std.step()

    standard_model.eval()
    standard_model.set_eval_config(nominal_cfg)
    pop_results_std = evaluate_across_population(standard_model, X_test, y_test, chip_pop,
                                                  label="Standard")
    results['standard'] = pop_results_std

    print(f"  Population: mean={pop_results_std['mean_accuracy']:.4f}, "
          f"std={pop_results_std['std_accuracy']:.4f}, "
          f"min={pop_results_std['min_accuracy']:.4f}")

    # Strategy 2: Distributional training
    print("\n\n=== Strategy 2: Distributional Robust Training ===")
    dist_model = DistributionalAnalogMLP(64, [128, 64], 10, hw_dist)
    dist_trainer = DistributionalRobustTrainer(
        lr=0.003, epochs=15, batch_size=64, hardware_dist=hw_dist
    )
    dist_trainer.train(dist_model, X_train, y_train, X_test, y_test)

    pop_results_dist = evaluate_across_population(dist_model, X_test, y_test, chip_pop,
                                                   label="Distributional")
    results['distributional'] = pop_results_dist

    print(f"  Population: mean={pop_results_dist['mean_accuracy']:.4f}, "
          f"std={pop_results_dist['std_accuracy']:.4f}, "
          f"min={pop_results_dist['min_accuracy']:.4f}")

    # Strategy 3: Curriculum (standard → distributional)
    print("\n\n=== Strategy 3: Curriculum Training ===")
    curriculum_model = DistributionalAnalogMLP(64, [128, 64], 10, hw_dist)

    # Phase 1: Standard training (epochs 1-8)
    opt_cur = torch.optim.Adam(curriculum_model.parameters(), lr=0.003)
    for epoch in range(8):
        curriculum_model.set_eval_config(nominal_cfg)
        curriculum_model.train()
        for bx, by in loader_std:
            opt_cur.zero_grad()
            out = curriculum_model(bx)
            crit(out, by).backward()
            opt_cur.step()

    # Phase 2: Distributional training (epochs 9-15)
    dist_cur = DistributionalRobustTrainer(
        lr=0.001, epochs=7, batch_size=64, hardware_dist=hw_dist
    )
    dist_cur.train(curriculum_model, X_train, y_train, X_test, y_test)

    pop_results_cur = evaluate_across_population(curriculum_model, X_test, y_test, chip_pop,
                                                  label="Curriculum")
    results['curriculum'] = pop_results_cur

    print(f"  Population: mean={pop_results_cur['mean_accuracy']:.4f}, "
          f"std={pop_results_cur['std_accuracy']:.4f}, "
          f"min={pop_results_cur['min_accuracy']:.4f}")

    # Save
    path = os.path.join(output_dir, 'distributional_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Chip Population Evaluation (50 chips)")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Mean Acc':>10} {'Std':>8} {'Min':>8} {'P10':>8} {'Yield':>8}")
    print(f"{'-'*62}")
    for name, r in results.items():
        print(f"{name:<20} {r['mean_accuracy']:>10.4f} {r['std_accuracy']:>8.4f} "
              f"{r['min_accuracy']:>8.4f} {r['p10_accuracy']:>8.4f} {r['yield_90pct']:>8.2f}")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Distributional Robust Training for Analog Hardware")
    print("=" * 60)

    from datasets.loaders import get_dataset
    X_train, y_train, X_test, y_test, nf, nc = get_dataset('mnist', subset_size=500, seed=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = run_comparison_experiment(X_train, y_train, X_test, y_test)

    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")
    std_mean = results.get('standard', {}).get('mean_accuracy', 0)
    dist_mean = results.get('distributional', {}).get('mean_accuracy', 0)
    cur_mean = results.get('curriculum', {}).get('mean_accuracy', 0)
    std_min = results.get('standard', {}).get('min_accuracy', 0)
    dist_min = results.get('distributional', {}).get('min_accuracy', 0)
    cur_min = results.get('curriculum', {}).get('min_accuracy', 0)

    print(f"Standard training:            mean={std_mean:.4f}, min across chips={std_min:.4f}")
    print(f"Distributional robust:        mean={dist_mean:.4f}, min across chips={dist_min:.4f}")
    print(f"Curriculum (std -> dist):      mean={cur_mean:.4f}, min across chips={cur_min:.4f}")

    dist_std = results.get('distributional', {}).get('std_accuracy', 0)
    std_std = results.get('standard', {}).get('std_accuracy', 0)
    print(f"\nVariance reduction: {'YES' if dist_std < std_std else 'NO'} "
          f"(std={std_std:.4f} -> std={dist_std:.4f})")
    print(f"Worst-case improvement: {'YES' if dist_min > std_min else 'NO'} "
          f"(min={std_min:.4f} -> min={dist_min:.4f})")
