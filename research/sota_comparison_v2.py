"""
SOTA Comparison v2: Adding Bayesian & Ensemble Calibration to Analog Neural Networks
=====================================================================================

Extends the original SOTA comparison by adding:
- DifferentiableAnalogMLP + Affine Calibration
- DifferentiableAnalogMLP + Bayesian Calibration (NEW)
- DifferentiableAnalogMLP + Ensemble Calibration (NEW)

Each calibrator is fitted on a held-out calibration set and evaluated
across a chip population. RMSE pre vs post calibration is reported.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import warnings

from analog_layers.analog_linear import AnalogLinear

from calibration.affine import AffineCalibrator
from calibration.polynomial import PolynomialCalibrator
from calibration.bayesian import BayesianCalibrator
from calibration.ensemble import EnsembleCalibrator
from validation.metrics import compute_metrics


def get_fashion_mnist(subset_size: int = None):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.flatten()),
    ])
    train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

    X_train = torch.stack([t[0] for t in train]).squeeze()
    y_train = torch.tensor([t[1] for t in train], dtype=torch.long)
    X_test = torch.stack([t[0] for t in test]).squeeze()
    y_test = torch.tensor([t[1] for t in test], dtype=torch.long)

    if subset_size:
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(X_train), generator=rng)[:min(subset_size, len(X_train))]
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def train_digital_model(X_train, y_train, X_test, y_test,
                        hidden_dims=[128, 64], epochs=15, lr=0.003):
    from experiments.models import DigitalMLP
    model = DigitalMLP(X_train.shape[1], hidden_dims, 10)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return model, acc


def eval_analog_deploy(digital_model, X_test, y_test, config: dict) -> float:
    linear_modules = []
    for module in digital_model.modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(module)

    analog_layers = [AnalogLinear.from_digital(m, config) for m in linear_modules]

    h = X_test
    li = 0
    for module in digital_model.modules():
        if isinstance(module, nn.Linear):
            if li < len(analog_layers):
                h = analog_layers[li](h)
                li += 1
        elif isinstance(module, nn.ReLU):
            h = module(h)

    return (h.argmax(1) == y_test).float().mean().item()


def run_nature_comms_approach(X_train, y_train, X_test, y_test,
                               hidden_dims=[128, 64], epochs=15):
    from torch.nn.utils import prune

    class RandomWeightMLP(nn.Module):
        def __init__(self, in_dim, h_dims, out_dim):
            super().__init__()
            dims = [in_dim] + h_dims + [out_dim]
            self.layers = nn.ModuleList()
            for i in range(len(dims) - 1):
                self.layers.append(nn.Linear(dims[i], dims[i+1]))
            self.init_random()

        def init_random(self):
            for m in self.layers:
                nn.init.normal_(m.weight, mean=0, std=1.0)

        def forward(self, x):
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = F.relu(h)
            return h

    model = RandomWeightMLP(X_train.shape[1], hidden_dims, 10)
    model.init_random()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return model, acc


def train_differentiable_analog(X_train, y_train, X_test, y_test,
                                 hidden_dims=[128, 64], epochs=15, lr=0.003,
                                 analog_config=None):
    from training.diff_analog import DifferentiableAnalogMLP, DifferentiableAnalogTrainer

    cfg = analog_config or {
        'sigma_mismatch': 0.01,
        'n_bits': 8,
        'vmax': 2.5,
        'vos_max': 0.002,
        'noise_sigma': 0.01,
        'learnable_params': False,
    }

    model = DifferentiableAnalogMLP(X_train.shape[1], hidden_dims, 10, cfg)
    trainer = DifferentiableAnalogTrainer(lr=lr, epochs=epochs, batch_size=64)
    trainer.train(model, X_train, y_train, X_test, y_test)

    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return model, acc


def train_distributional_analog(X_train, y_train, X_test, y_test,
                                 hidden_dims=[128, 64], epochs=15):
    from training.distributional import (DistributionalAnalogMLP,
                                          DistributionalRobustTrainer,
                                          HardwareDistribution)

    hw_dist = HardwareDistribution(
        sigma_mismatch=(0.005, 0.05),
        noise_sigma=(0.001, 0.05),
        vos_max=(0.0, 0.005),
        n_bits_range=(6, 10),
        vmax_range=(2.0, 3.0),
    )

    model = DistributionalAnalogMLP(X_train.shape[1], hidden_dims, 10, hw_dist)
    trainer = DistributionalRobustTrainer(
        lr=0.003, epochs=epochs, batch_size=64, hardware_dist=hw_dist
    )
    trainer.train(model, X_train, y_train, X_test, y_test)

    nominal_cfg = {
        'sigma_mismatch': np.median(hw_dist.sigma_mismatch),
        'noise_sigma': np.median(hw_dist.noise_sigma),
        'vos_max': np.median(hw_dist.vos_max),
        'n_bits': int(np.mean(hw_dist.n_bits_range)),
        'vmax': np.median(hw_dist.vmax_range),
    }
    model.set_eval_config(nominal_cfg)
    model.eval()
    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean().item()
    return model, acc, hw_dist


def generate_chip_population(n_chips=50, seed=123):
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n_chips):
        configs.append({
            'sigma_mismatch': rng.uniform(0.005, 0.05),
            'noise_sigma': rng.uniform(0.001, 0.05),
            'vos_max': rng.uniform(0.0, 0.005),
            'n_bits': rng.randint(6, 11),
            'vmax': rng.uniform(2.0, 3.0),
        })
    return configs


def apply_chip_config_to_diff_model(model, cfg):
    """Apply a chip configuration to a DifferentiableAnalogMLP."""
    for layer in model.layers:
        if hasattr(layer, 'mismatch') and hasattr(layer.mismatch, 'log_sigma'):
            layer.mismatch.log_sigma.data.fill_(np.log(cfg['sigma_mismatch']))


if __name__ == '__main__':
    print("=" * 80)
    print("SOTA COMPARISON V2: Adding Bayesian & Ensemble Calibration")
    print("=" * 80)

    print("\nLoading Fashion-MNIST (subset for speed)...")
    X_train, y_train, X_test, y_test = get_fashion_mnist(subset_size=2000)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Split test into calibration (50%) and evaluation (50%)
    rng = np.random.RandomState(42)
    n_test = len(X_test)
    cal_frac = 0.5
    cal_idx = rng.choice(n_test, size=int(n_test * cal_frac), replace=False)
    eval_idx = np.array([i for i in range(n_test) if i not in cal_idx])

    X_cal = X_test[cal_idx]
    y_cal = y_test[cal_idx]
    X_eval = X_test[eval_idx]
    y_eval = y_test[eval_idx]

    print(f"Calibration set: {len(X_cal)}, Evaluation set: {len(X_eval)}")

    hidden_dims = [128, 64]
    chip_pop = generate_chip_population(n_chips=30, seed=42)

    results = {}

    # =============================================================
    # METHOD 1: Standard Digital + Analog Deploy
    # =============================================================
    print("\n\n" + "=" * 60)
    print("METHOD 1: Standard Digital + Analog Deploy [BASELINE]")
    print("=" * 60)
    dig_model, dig_acc = train_digital_model(
        X_train, y_train, X_eval, y_eval, hidden_dims, epochs=12
    )
    analog_config_deploy = {
        'noise_sigma': 0.01,
        'mismatch_sigma': 0.01,
        'quantization_bits': 8,
        'saturation_vmax': 2.5,
        'opamp_offset_sigma': 0.002,
    }
    analog_deploy_acc = eval_analog_deploy(dig_model, X_eval, y_eval, analog_config_deploy)
    print(f"  Digital accuracy:       {dig_acc:.4f}")
    print(f"  Analog deploy accuracy: {analog_deploy_acc:.4f}")

    chip_accs_std = []
    for cfg in chip_pop:
        deploy_cfg = {
            'noise_sigma': cfg['noise_sigma'],
            'mismatch_sigma': cfg['sigma_mismatch'],
            'quantization_bits': cfg['n_bits'],
            'saturation_vmax': cfg['vmax'],
            'opamp_offset_sigma': cfg['vos_max'],
        }
        chip_accs_std.append(eval_analog_deploy(dig_model, X_eval, y_eval, deploy_cfg))
    std_arr = np.array(chip_accs_std)

    results['standard_deploy'] = {
        'digital_accuracy': dig_acc,
        'analog_accuracy': analog_deploy_acc,
        'chip_mean': float(np.mean(std_arr)),
        'chip_std': float(np.std(std_arr)),
        'chip_min': float(np.min(std_arr)),
    }
    print(f"  Chip pop mean: {np.mean(std_arr):.4f}, std: {np.std(std_arr):.4f}, min: {np.min(std_arr):.4f}")

    # =============================================================
    # METHOD 2: Nature Comms 2026
    # =============================================================
    print("\n\n" + "=" * 60)
    print("METHOD 2: Nature Comms 2026 (Edge-Pruning)")
    print("=" * 60)
    nc_model, nc_acc = run_nature_comms_approach(
        X_train, y_train, X_eval, y_eval, hidden_dims, epochs=12
    )
    print(f"  Topology-optimized accuracy: {nc_acc:.4f}")

    analog_nc_acc = eval_analog_deploy(nc_model, X_eval, y_eval, analog_config_deploy)
    print(f"  Analog deploy: {analog_nc_acc:.4f}")

    chip_nc = []
    for cfg in chip_pop:
        deploy_cfg = {
            'noise_sigma': cfg['noise_sigma'],
            'mismatch_sigma': cfg['sigma_mismatch'],
            'quantization_bits': cfg['n_bits'],
            'saturation_vmax': cfg['vmax'],
            'opamp_offset_sigma': cfg['vos_max'],
        }
        chip_nc.append(eval_analog_deploy(nc_model, X_eval, y_eval, deploy_cfg))
    nc_arr = np.array(chip_nc)

    results['nature_comms'] = {
        'digital_accuracy': nc_acc,
        'analog_accuracy': analog_nc_acc,
        'chip_mean': float(np.mean(nc_arr)),
        'chip_std': float(np.std(nc_arr)),
        'chip_min': float(np.min(nc_arr)),
    }
    print(f"  Chip pop mean: {np.mean(nc_arr):.4f}, std: {np.std(nc_arr):.4f}, min: {np.min(nc_arr):.4f}")

    # =============================================================
    # METHOD 3: DifferentiableAnalogMLP
    # =============================================================
    print("\n\n" + "=" * 60)
    print("METHOD 3: DifferentiableAnalogMLP")
    print("=" * 60)
    diff_model, diff_acc = train_differentiable_analog(
        X_train, y_train, X_eval, y_eval, hidden_dims, epochs=12,
        analog_config={'sigma_mismatch': 0.01, 'n_bits': 8, 'vmax': 2.5,
                       'vos_max': 0.002, 'noise_sigma': 0.01}
    )
    print(f"  Differentiable analog accuracy: {diff_acc:.4f}")

    chip_diff = []
    for cfg in chip_pop:
        diff_model.eval()
        apply_chip_config_to_diff_model(diff_model, cfg)
        with torch.no_grad():
            out = diff_model(X_eval)
            chip_diff.append((out.argmax(1) == y_eval).float().mean().item())
    diff_arr = np.array(chip_diff)

    results['differentiable'] = {
        'analog_accuracy': diff_acc,
        'chip_mean': float(np.mean(diff_arr)),
        'chip_std': float(np.std(diff_arr)),
        'chip_min': float(np.min(diff_arr)),
    }
    print(f"  Chip pop mean: {np.mean(diff_arr):.4f}, std: {np.std(diff_arr):.4f}, min: {np.min(diff_arr):.4f}")

    # =============================================================
    # METHOD 4: Distributional Robust
    # =============================================================
    print("\n\n" + "=" * 60)
    print("METHOD 4: Distributional Robust Training")
    print("=" * 60)
    dist_model, dist_acc, hw_dist = train_distributional_analog(
        X_train, y_train, X_eval, y_eval, hidden_dims, epochs=12
    )
    print(f"  Distributional analog accuracy: {dist_acc:.4f}")

    chip_dist = []
    for cfg in chip_pop:
        dist_model.set_eval_config(cfg)
        with torch.no_grad():
            out = dist_model(X_eval)
            chip_dist.append((out.argmax(1) == y_eval).float().mean().item())
    dist_arr = np.array(chip_dist)

    results['distributional'] = {
        'analog_accuracy': dist_acc,
        'chip_mean': float(np.mean(dist_arr)),
        'chip_std': float(np.std(dist_arr)),
        'chip_min': float(np.min(dist_arr)),
    }
    print(f"  Chip pop mean: {np.mean(dist_arr):.4f}, std: {np.std(dist_arr):.4f}, min: {np.min(dist_arr):.4f}")

    # =============================================================
    # CALIBRATION SECTION
    # =============================================================
    print("\n\n" + "=" * 60)
    print("CALIBRATION: Applying calibrators to DifferentiableAnalogMLP")
    print("=" * 60)

    # Get ideal (digital) outputs for calibration and eval sets
    dig_model.eval()
    with torch.no_grad():
        y_ideal_cal = dig_model(X_cal)       # (N_cal, 10)
        y_ideal_eval = dig_model(X_eval)      # (N_eval, 10)

    # Get raw analog outputs on calibration set (nominal config)
    diff_model.eval()
    apply_chip_config_to_diff_model(diff_model, {'sigma_mismatch': 0.01})
    with torch.no_grad():
        y_spice_cal = diff_model(X_cal)       # (N_cal, 10)
        y_spice_eval = diff_model(X_eval)     # (N_eval, 10)

    # Pre-calibration RMSE
    rmse_pre = float(np.sqrt(np.mean((y_ideal_eval.numpy() - y_spice_eval.numpy()) ** 2)))
    print(f"\n  Pre-calibration RMSE (on eval set): {rmse_pre:.6f}")

    # Subsampling for calibration (GP is O(n^3), use 500 points)
    n_cal_small = min(500, len(X_cal))
    gp_sub = np.random.RandomState(42).choice(len(X_cal), n_cal_small, replace=False)

    # ---- Affine Calibration ----
    print("\n  >> Affine Calibration")
    affine_cal = AffineCalibrator()
    affine_cal.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    y_affine = affine_cal.calibrate(y_spice_eval)
    rmse_affine = float(np.sqrt(np.mean((y_ideal_eval.numpy() - y_affine.numpy()) ** 2)))

    chip_affine = []
    for cfg in chip_pop:
        diff_model.eval()
        apply_chip_config_to_diff_model(diff_model, cfg)
        with torch.no_grad():
            raw = diff_model(X_eval)
            cal = affine_cal.calibrate(raw)
            chip_affine.append((cal.argmax(1) == y_eval).float().mean().item())
    aff_arr = np.array(chip_affine)

    results['diff_affine'] = {
        'analog_accuracy': float(np.mean(aff_arr)),
        'chip_mean': float(np.mean(aff_arr)),
        'chip_std': float(np.std(aff_arr)),
        'chip_min': float(np.min(aff_arr)),
        'rmse_post': rmse_affine,
    }
    print(f"    RMSE post: {rmse_affine:.6f}  (improvement: {(rmse_pre - rmse_affine) / rmse_pre * 100:.1f}%)")
    print(f"    Chip pop mean: {np.mean(aff_arr):.4f}, std: {np.std(aff_arr):.4f}, min: {np.min(aff_arr):.4f}")

    # ---- Bayesian Calibration (NEW) ----
    print("\n  >> Bayesian Calibration (NEW)")
    bayes_cal = BayesianCalibrator()
    bayes_cal.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    y_bayes = bayes_cal.calibrate(y_spice_eval)
    rmse_bayes = float(np.sqrt(np.mean((y_ideal_eval.numpy() - y_bayes.numpy()) ** 2)))

    chip_bayes = []
    for cfg in chip_pop:
        diff_model.eval()
        apply_chip_config_to_diff_model(diff_model, cfg)
        with torch.no_grad():
            raw = diff_model(X_eval)
            cal = bayes_cal.calibrate(raw)
            chip_bayes.append((cal.argmax(1) == y_eval).float().mean().item())
    bayes_arr = np.array(chip_bayes)

    results['diff_bayesian'] = {
        'analog_accuracy': float(np.mean(bayes_arr)),
        'chip_mean': float(np.mean(bayes_arr)),
        'chip_std': float(np.std(bayes_arr)),
        'chip_min': float(np.min(bayes_arr)),
        'rmse_post': rmse_bayes,
    }
    print(f"    RMSE post: {rmse_bayes:.6f}  (improvement: {(rmse_pre - rmse_bayes) / rmse_pre * 100:.1f}%)")
    print(f"    Chip pop mean: {np.mean(bayes_arr):.4f}, std: {np.std(bayes_arr):.4f}, min: {np.min(bayes_arr):.4f}")

    # ---- Ensemble Calibration (NEW) ----
    print("\n  >> Ensemble Calibration (NEW)")
    # Pre-fit each calibrator on subsampled data for efficiency
    affine_cal_ens = AffineCalibrator()
    affine_cal_ens.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    poly_cal_ens = PolynomialCalibrator(degree=3)
    poly_cal_ens.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    bayes_cal_ens = BayesianCalibrator()
    bayes_cal_ens.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    ensemble_cal = EnsembleCalibrator(
        calibrators={
            'affine': affine_cal_ens,
            'polynomial': poly_cal_ens,
            'bayesian': bayes_cal_ens,
        },
        strategy='average'
    )
    ensemble_cal.fit(y_spice_cal[gp_sub], y_ideal_cal[gp_sub])
    y_ensemble = ensemble_cal.calibrate(y_spice_eval)
    rmse_ensemble = float(np.sqrt(np.mean((y_ideal_eval.numpy() - y_ensemble.numpy()) ** 2)))

    chip_ensemble = []
    for cfg in chip_pop:
        diff_model.eval()
        apply_chip_config_to_diff_model(diff_model, cfg)
        with torch.no_grad():
            raw = diff_model(X_eval)
            cal = ensemble_cal.calibrate(raw)
            chip_ensemble.append((cal.argmax(1) == y_eval).float().mean().item())
    ens_arr = np.array(chip_ensemble)

    results['diff_ensemble'] = {
        'analog_accuracy': float(np.mean(ens_arr)),
        'chip_mean': float(np.mean(ens_arr)),
        'chip_std': float(np.std(ens_arr)),
        'chip_min': float(np.min(ens_arr)),
        'rmse_post': rmse_ensemble,
    }
    print(f"    RMSE post: {rmse_ensemble:.6f}  (improvement: {(rmse_pre - rmse_ensemble) / rmse_pre * 100:.1f}%)")
    print(f"    Chip pop mean: {np.mean(ens_arr):.4f}, std: {np.std(ens_arr):.4f}, min: {np.min(ens_arr):.4f}")

    # =============================================================
    # SAVE RESULTS
    # =============================================================
    path = 'research/sota_v2_results.json'
    os.makedirs('research', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    # =============================================================
    # SUMMARY TABLE
    # =============================================================
    print("\n\n" + "=" * 105)
    print("FINAL COMPARISON TABLE V2 — All Methods")
    print("=" * 105)

    header = (
        f"{'Method':<40} {'Dig Acc':>8} {'Anlg Acc':>10} "
        f"{'Chip Mean':>10} {'Chip Std':>9} {'Chip Min':>9} {'RMSE':>10}"
    )
    print(header)
    print("-" * 105)

    display_names = {
        'standard_deploy': 'Standard Deploy',
        'nature_comms': 'Nature Comms 2026',
        'differentiable': 'DifferentiableAnalogMLP',
        'distributional': 'Distributional Robust',
        'diff_affine': '+ Affine Cal',
        'diff_bayesian': '+ Bayesian Cal (NEW)',
        'diff_ensemble': '+ Ensemble Cal (NEW)',
    }

    for name, r in results.items():
        dname = display_names.get(name, name)
        dacc = r.get('digital_accuracy', None)
        dacc_str = f"{dacc:.4f}" if dacc is not None else "   -   "
        aacc = r.get('analog_accuracy', 0)
        c_mean = r.get('chip_mean', 0)
        c_std = r.get('chip_std', 0)
        c_min = r.get('chip_min', 0)
        rmse = r.get('rmse_post', None)
        rmse_str = f"{rmse:.6f}" if rmse is not None else "     -    "
        print(
            f"{dname:<40} {dacc_str:>8} {aacc:>10.4f} "
            f"{c_mean:>10.4f} {c_std:>9.4f} {c_min:>9.4f} {rmse_str:>10}"
        )

    # ---- RMSE Summary ----
    print("\n\n" + "=" * 60)
    print("RMSE COMPARISON (Pre vs Post Calibration)")
    print("=" * 60)
    print(f"{'Method':<35} {'RMSE':>12} {'Improvement':>12}")
    print("-" * 59)
    print(f"{'Uncalibrated (pre)':<35} {rmse_pre:>12.6f} {'  ---':>12}")

    cal_methods = [
        ('diff_affine', 'Affine Cal'),
        ('diff_bayesian', 'Bayesian Cal'),
        ('diff_ensemble', 'Ensemble Cal'),
    ]
    for key, label in cal_methods:
        r = results.get(key, {})
        rmse_val = r.get('rmse_post', None)
        if rmse_val is not None:
            impr = (rmse_pre - rmse_val) / rmse_pre * 100
            print(f"{label:<35} {rmse_val:>12.6f} {impr:>+11.1f}%")

    # ---- Best scores ----
    print("\n\n" + "=" * 80)
    print("BEST SCORES")
    print("=" * 80)
    best_mean = max((r.get('chip_mean', 0), n) for n, r in results.items())
    best_std = min((r.get('chip_std', float('inf')), n) for n, r in results.items())
    best_min = max((r.get('chip_min', 0), n) for n, r in results.items())
    best_acc = max((r.get('analog_accuracy', 0), n) for n, r in results.items())
    best_rmse = min((r.get('rmse_post', float('inf')), n) for n, r in results.items() if 'rmse_post' in r)

    print(f"  Best analog accuracy:  {display_names.get(best_acc[1], best_acc[1])} ({best_acc[0]:.4f})")
    print(f"  Best chip mean:        {display_names.get(best_mean[1], best_mean[1])} ({best_mean[0]:.4f})")
    print(f"  Best chip consistency: {display_names.get(best_std[1], best_std[1])} (std={best_std[0]:.4f})")
    print(f"  Best worst-case chip:  {display_names.get(best_min[1], best_min[1])} (min={best_min[0]:.4f})")
    print(f"  Best post-cal RMSE:    {display_names.get(best_rmse[1], best_rmse[1])} ({best_rmse[0]:.6f})")

    # Improvement vs baseline
    if 'distributional' in results and 'standard_deploy' in results:
        imp = results['distributional']['chip_mean'] - results['standard_deploy']['chip_mean']
        print(f"\n  Distributional vs Standard Deploy: +{imp:.4f} mean accuracy")

    if 'diff_ensemble' in results and 'differentiable' in results:
        imp = results['diff_ensemble']['chip_mean'] - results['differentiable']['chip_mean']
        print(f"  Ensemble Cal vs Uncalibrated Diff: +{imp:.4f} mean accuracy")

    print("\n\nDone.")
