"""
Analog Scaling Law: accuracy_drop = f(depth, width, noise_sigma)
==============================================================

Goal: Achieve R2 > 0.80 with comprehensive grid search and
multiplicative interaction terms (depth x noise propagates).

Theorem 8 gives us the physical prior: additive perturbations
propagate ~5.2x more through deeper ReLU networks, so the
scaling law MUST include a depth x noise interaction term.

Model:
    log(drop) = log(a) + alpha·log(D) + beta·log(W) + gamma·log(N)
                       + delta·log(D)·log(N)   [interaction]

This captures: depth amplifies noise effects multiplicatively.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os
import time
import warnings
import itertools

from experiments.models import DigitalMLP
from analog_layers.analog_linear import AnalogLinear


def train_and_eval(depth: int, width: int, noise_sigma: float,
                   X_train, y_train, X_test, y_test,
                   epochs: int = 15, lr: float = 0.003,
                   seed: int = 0) -> Dict:
    """Train a DigitalMLP, then deploy on AnalogLinear — measure accuracy drop."""
    torch.manual_seed(seed)
    hidden_dims = [width] * (depth - 1)

    model = DigitalMLP(X_train.shape[1], hidden_dims, 10)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    dig_out = model(X_test)
    dig_acc = (dig_out.argmax(1) == y_test).float().mean().item()

    # Convert to analog and evaluate
    analog_config = {
        'noise_sigma': noise_sigma,
        'mismatch_sigma': noise_sigma,
        'quantization_bits': 8,
        'saturation_vmax': 2.5,
        'opamp_offset_sigma': noise_sigma * 0.2,
        'drift_time': 0.0,
    }

    analog_model = AnalogMLP.from_digital(model, analog_config)
    analog_model.eval()
    analog_out = analog_model(X_test)
    analog_acc = (analog_out.argmax(1) == y_test).float().mean().item()

    accuracy_drop = dig_acc - analog_acc

    return {
        'depth': depth,
        'width': width,
        'noise_sigma': noise_sigma,
        'digital_accuracy': dig_acc,
        'analog_accuracy': analog_acc,
        'accuracy_drop': accuracy_drop,
        'n_params': sum(p.numel() for p in model.parameters()),
    }


class AnalogMLP(nn.Module):
    """Digital MLP with analog evaluation."""

    @classmethod
    def from_digital(cls, digital_model: DigitalMLP, analog_config: Dict) -> 'AnalogMLP':
        model = cls()
        dig_net = digital_model.network
        idx = 0
        for dig_layer in dig_net:
            if isinstance(dig_layer, nn.Linear):
                analog = AnalogLinear.from_digital(dig_layer, analog_config)
                model.add_module(f'analog_{idx}', analog)
                idx += 1
            elif isinstance(dig_layer, nn.ReLU):
                model.add_module(f'relu_{idx}', nn.ReLU())
                idx += 1
        return model

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def run_scaling_law_experiment(X_train, y_train, X_test, y_test,
                                grid: Dict = None,
                                n_seeds: int = 3,
                                output_dir: str = 'research_advanced'):
    """
    Systematic sweep over depth, width, noise — multiple seeds per config.

    Grid defaults cover [depth: 1-6, width: 32-512, noise: 0.001-0.2]
    with geometric spacing in noise for balanced log coverage.
    """
    if grid is None:
        grid = {
            'depths': [1, 2, 3, 4, 5, 6],
            'widths': [32, 64, 128, 256, 512],
            'noise_levels': [0.001, 0.003, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        }

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    configs = list(itertools.product(
        grid['depths'], grid['widths'], grid['noise_levels']
    ))
    total = len(configs) * n_seeds
    count = 0

    print(f"Analog Scaling Law: {len(configs)} configs x {n_seeds} seeds = {total} runs")
    print(f"  depths={grid['depths']}, widths={grid['widths']}")
    print(f"  noise_levels={grid['noise_levels']}")
    print(f"{'='*80}")

    for depth, width, noise in configs:
        seed_results = []
        for seed in range(n_seeds):
            count += 1
            print(f"[{count}/{total}] D={depth}, W={width}, N={noise:.4f}, seed={seed}", end="")
            t0 = time.time()

            result = train_and_eval(
                depth, width, noise,
                X_train, y_train, X_test, y_test,
                epochs=10, lr=0.003,
                seed=seed
            )
            seed_results.append(result)

            elapsed = time.time() - t0
            print(f"  drop={result['accuracy_drop']:.4f}  ({elapsed:.1f}s)")

        avg_drop = np.mean([r['accuracy_drop'] for r in seed_results])
        std_drop = np.std([r['accuracy_drop'] for r in seed_results])
        all_results.append({
            'depth': depth,
            'width': width,
            'noise_sigma': noise,
            'accuracy_drop_mean': float(avg_drop),
            'accuracy_drop_std': float(std_drop),
            'n_params': seed_results[0]['n_params'],
            'seed_results': [
                {
                    'seed': s,
                    'digital_accuracy': r['digital_accuracy'],
                    'analog_accuracy': r['analog_accuracy'],
                    'accuracy_drop': r['accuracy_drop'],
                }
                for s, r in enumerate(seed_results)
            ],
        })

    # === Fitting ===
    print(f"\n{'='*80}")
    print("Fitting Scaling Law (with depthxnoise interaction)...")
    print(f"{'='*80}")

    # Prepare data: use mean drop per config
    X_mat = []
    y_vec = []
    weights = []

    for r in all_results:
        drop = max(r['accuracy_drop_mean'], 1e-10)
        X_mat.append([
            np.log(r['depth']),
            np.log(r['width']),
            np.log(r['noise_sigma']),
            np.log(r['depth']) * np.log(r['noise_sigma']),  # interaction
        ])
        y_vec.append(np.log(drop))
        weights.append(1.0 / (r['accuracy_drop_std'] + 0.001))

    X_mat = np.array(X_mat)
    y_vec = np.array(y_vec)
    weights = np.array(weights)
    W_sqrt = np.diag(np.sqrt(weights))

    # Weighted OLS: minimize sum w_i * (y_i - x_i @ beta)^2
    Aw = W_sqrt @ np.column_stack([np.ones(len(X_mat)), X_mat])
    bw = W_sqrt @ y_vec
    coeffs, residuals, rank, s = np.linalg.lstsq(Aw, bw, rcond=None)

    log_a = coeffs[0]
    alpha = coeffs[1]
    beta_ = coeffs[2]
    gamma = coeffs[3]
    delta = coeffs[4]
    a = np.exp(log_a)

    y_pred = np.column_stack([np.ones(len(X_mat)), X_mat]) @ coeffs
    ss_res = np.sum(weights * (y_vec - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_vec - np.mean(y_vec)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.sum((y_vec - y_pred) ** 2) / len(y_vec))

    scaling_law = {
        'formula': 'drop = a * D^α * W^β * N^γ * exp(δ·log(D)·log(N))',
        'a': float(a),
        'alpha': float(alpha),
        'beta': float(beta_),
        'gamma': float(gamma),
        'delta': float(delta),
        'r_squared': float(r2),
        'rmse': float(rmse),
        'n_configs': len(all_results),
        'n_seeds': n_seeds,
    }

    print(f"\nWeighted Fit (inverse std weighting):")
    print(f"  drop = {a:.6f} x D^{alpha:.4f} x W^{beta_:.4f} x N^{gamma:.4f}")
    print(f"        x exp({delta:.4f} · log(D) · log(N))")
    print(f"  R2 = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    # === Also fit without interaction for comparison ===
    X_no_int = X_mat[:, :3]
    A_no_int = np.column_stack([np.ones(len(X_no_int)), X_no_int])
    coeffs_ni, _, _, _ = np.linalg.lstsq(A_no_int, y_vec, rcond=None)
    y_pred_ni = A_no_int @ coeffs_ni
    ss_res_ni = np.sum((y_vec - y_pred_ni) ** 2)
    r2_ni = 1 - ss_res_ni / ss_tot

    print(f"\nWithout interaction term:")
    print(f"  drop = {np.exp(coeffs_ni[0]):.6f} x D^{coeffs_ni[1]:.4f} x W^{coeffs_ni[2]:.4f} x N^{coeffs_ni[3]:.4f}")
    print(f"  R2 = {r2_ni:.4f}")
    print(f"  Improvement from interaction: deltaR2 = {r2 - r2_ni:.4f}")

    # === Ridge regression for comparison ===
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(np.column_stack([np.ones(len(X_mat)), X_mat]))
        # Don't scale the intercept... actually let's just scale the features
        X_scaled2 = scaler.fit_transform(X_mat)
        A_scaled = np.column_stack([np.ones(len(X_scaled2)), X_scaled2])

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), fit_intercept=False)
        ridge.fit(A_scaled, y_vec, sample_weight=weights)
        y_pred_ridge = ridge.predict(A_scaled)
        ss_res_ridge = np.sum(weights * (y_vec - y_pred_ridge) ** 2)
        r2_ridge = 1 - ss_res_ridge / ss_tot

        print(f"\nRidge regression (CV-optimized α={ridge.alpha_:.4f}):")
        print(f"  R2 = {r2_ridge:.4f}")
        scaling_law['ridge_r_squared'] = float(r2_ridge)
        scaling_law['ridge_alpha'] = float(ridge.alpha_)
        scaling_law['ridge_coefficients'] = ridge.coef_.tolist()
    except ImportError:
        print("  (sklearn not available, skipping Ridge)")

    # === Random Forest for non-parametric comparison ===
    try:
        from sklearn.ensemble import RandomForestRegressor

        rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, min_samples_leaf=5)
        rf.fit(X_mat, y_vec, sample_weight=weights)
        y_pred_rf = rf.predict(X_mat)
        ss_res_rf = np.sum(weights * (y_vec - y_pred_rf) ** 2)
        r2_rf = 1 - ss_res_rf / ss_tot

        print(f"\nRandom Forest (non-parametric):")
        print(f"  R2 = {r2_rf:.4f}")
        print(f"  Feature importance: depth={rf.feature_importances_[0]:.3f}, "
              f"width={rf.feature_importances_[1]:.3f}, "
              f"noise={rf.feature_importances_[2]:.3f}, "
              f"interaction={rf.feature_importances_[3]:.3f}")
        scaling_law['rf_r_squared'] = float(r2_rf)
        scaling_law['rf_feature_importances'] = rf.feature_importances_.tolist()
    except ImportError:
        print("  (sklearn not available, skipping RF)")

    # === Depth-only analysis ===
    print(f"\n{'='*80}")
    print("Depth-Only Analysis (averaged over width and noise)")
    print(f"{'='*80}")
    by_depth = defaultdict(list)
    for r in all_results:
        by_depth[r['depth']].append(r['accuracy_drop_mean'])
    for d in sorted(by_depth.keys()):
        vals = np.array(by_depth[d])
        print(f"  D={d}: mean={np.mean(vals):.4f}, median={np.median(vals):.4f}, "
              f"std={np.std(vals):.4f}, max={np.max(vals):.4f}")

    # Fit depth-only: drop = a * D^alpha (using means per depth)
    depth_means = {d: np.mean(by_depth[d]) for d in sorted(by_depth.keys())}
    X_d = np.log(list(depth_means.keys()))
    y_d = np.log([max(v, 1e-10) for v in depth_means.values()])
    A_d = np.column_stack([np.ones(len(X_d)), X_d])
    c_d, _, _, _ = np.linalg.lstsq(A_d, y_d, rcond=None)
    y_pred_d = A_d @ c_d
    ss_res_d = np.sum((y_d - y_pred_d) ** 2)
    ss_tot_d = np.sum((y_d - np.mean(y_d)) ** 2)
    r2_d = 1 - ss_res_d / ss_tot_d

    print(f"\n  Depth-only power law: drop = {np.exp(c_d[0]):.6f} x D^{c_d[1]:.4f}")
    print(f"  R2 = {r2_d:.4f}")
    print(f"  → depth exponent α ≈ {c_d[1]:.2f}")

    # Save
    output = {
        'configs': grid,
        'scaling_law': scaling_law,
        'depth_only': {
            'a': float(np.exp(c_d[0])),
            'alpha': float(c_d[1]),
            'r_squared': float(r2_d),
        },
        'results': all_results,
    }

    path = os.path.join(output_dir, 'scaling_law_results.json')
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {path}")

    return output


def derive_design_equation(scaling_law: Dict, target_drop: float = 0.02) -> Dict:
    """
    Derive required noise sigma for a given architecture and target drop.

    With interaction term:
    drop = a x D^α x W^β x N^γ x exp(δ·log(D)·log(N))
         = a x D^α x W^β x N^{γ + δ·log(D)}

    Solve: N = (drop / (a x D^α x W^β))^{1/(γ + δ·log(D))}
    """
    a = scaling_law['a']
    alpha = scaling_law['alpha']
    beta_ = scaling_law['beta']
    gamma = scaling_law['gamma']
    delta = scaling_law.get('delta', 0.0)

    def required_noise(depth: int, width: int, drop: float = target_drop) -> float:
        if a <= 0 or (gamma + delta * np.log(depth)) <= 0:
            return float('inf')
        denominator = a * depth**alpha * width**beta_
        if denominator <= 0:
            return float('inf')
        raw = (drop / denominator) ** (1.0 / (gamma + delta * np.log(depth)))
        return min(float(raw), 1e6)

    equations = {}
    architectures = [
        (1, 32), (1, 128),
        (2, 64), (2, 256),
        (3, 128), (3, 512),
        (4, 256), (5, 512), (6, 512),
    ]

    print(f"\n{'='*80}")
    print(f"Design Equations: Required noise σ for {target_drop*100:.1f}% accuracy drop")
    print(f"{'='*80}")

    for depth, width in architectures:
        req = required_noise(depth, width)
        label = f"σ ≤ {req:.5f}" if req < 1e3 else "unconstrained"
        equations[f'{depth}L_{width}W'] = {
            'depth': depth,
            'width': width,
            'required_noise_sigma': float(req) if req < 1e6 else float('inf'),
        }
        print(f"  {depth}-layer, {width}-wide: {label}")

    return equations


def plot_scaling_law(output: Dict, output_dir: str = 'research_advanced'):
    """Generate publication-quality plots of the scaling law."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    results = output['results']
    sl = output['scaling_law']
    a = sl['a']
    alpha = sl['alpha']
    beta_ = sl['beta']
    gamma = sl['gamma']
    delta = sl.get('delta', 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Predicted vs actual (log-log)
    ax = axes[0, 0]
    drops = np.array([max(r['accuracy_drop_mean'], 1e-10) for r in results])
    depths = np.array([r['depth'] for r in results])
    widths = np.array([r['width'] for r in results])
    noises = np.array([r['noise_sigma'] for r in results])

    predicted = a * depths**alpha * widths**beta_ * noises**gamma * \
                np.exp(delta * np.log(depths) * np.log(noises))

    ax.scatter(predicted, drops, alpha=0.5, s=20)
    min_val = min(np.min(predicted), np.min(drops))
    max_val = max(np.max(predicted), np.max(drops))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted drop')
    ax.set_ylabel('Actual drop')
    ax.set_title(f'Scaling Law Fit (R2={sl["r_squared"]:.4f})')
    ax.grid(True, alpha=0.3)

    # 2. Depth vs accuracy drop (noise-colored)
    ax = axes[0, 1]
    unique_depths = sorted(set(r['depth'] for r in results))
    for d in unique_depths:
        d_vals = [r for r in results if r['depth'] == d]
        drops_d = [max(r['accuracy_drop_mean'], 1e-10) for r in d_vals]
        noises_d = [r['noise_sigma'] for r in d_vals]
        sc = ax.scatter(noises_d, drops_d, c=[d]*len(noises_d),
                        cmap='viridis', alpha=0.6, s=30, label=f'D={d}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Noise σ')
    ax.set_ylabel('Accuracy drop')
    ax.set_title('Drop vs Noise (colored by depth)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Depth-only: mean drop per depth
    ax = axes[1, 0]
    depth_only = output.get('depth_only', {})
    by_depth = defaultdict(list)
    for r in results:
        by_depth[r['depth']].append(r['accuracy_drop_mean'])
    depths_sorted = sorted(by_depth.keys())
    means = [np.mean(by_depth[d]) for d in depths_sorted]
    stds = [np.std(by_depth[d]) for d in depths_sorted]
    ax.errorbar(depths_sorted, means, yerr=stds, fmt='o-', capsize=5)
    d_fit = np.linspace(min(depths_sorted), max(depths_sorted), 100)
    a_d = depth_only.get('a', 0.01)
    alpha_d = depth_only.get('alpha', 3.0)
    ax.plot(d_fit, a_d * d_fit**alpha_d, 'r--', label=f'D^{alpha_d:.2f}')
    ax.set_xlabel('Depth')
    ax.set_ylabel('Mean accuracy drop')
    ax.set_title(f'Depth-Only Scaling (α={alpha_d:.2f}, R2={depth_only.get("r_squared", 0):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Residuals plot
    ax = axes[1, 1]
    log_drops = np.log(drops)
    log_pred = np.log(predicted)
    residuals = log_drops - log_pred
    ax.scatter(log_pred, residuals, alpha=0.5, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=1)
    ax.set_xlabel('Log predicted drop')
    ax.set_ylabel('Log residual')
    ax.set_title(f'Residuals (RMSE={sl["rmse"]:.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scaling_law_fig.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {path}")


if __name__ == '__main__':
    print("=" * 80)
    print("ANALOG SCALING LAW — Improved Fit with Interaction Term")
    print("=" * 80)

    print("\nLoading MNIST data...")
    from datasets.loaders import get_dataset
    X_train, y_train, X_test, y_test, nf, nc = get_dataset('mnist', subset_size=1000, seed=42)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = run_scaling_law_experiment(
        X_train, y_train, X_test, y_test,
        grid={
            'depths': [1, 2, 3, 4, 5, 6],
            'widths': [32, 64, 128, 256],
            'noise_levels': [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2],
        },
        n_seeds=3,
    )

    equations = derive_design_equation(results['scaling_law'])

    plot_scaling_law(results)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    sl = results['scaling_law']
    print(f"Scaling Law:")
    print(f"  drop = {sl['a']:.6f} x D^{sl['alpha']:.3f} x W^{sl['beta']:.3f} x N^{sl['gamma']:.3f}")
    print(f"        x exp({sl.get('delta', 0):.4f} · log(D) · log(N))")
    print(f"  R2 = {sl['r_squared']:.4f}")

    if 'ridge_r_squared' in sl:
        print(f"  Ridge R2 = {sl['ridge_r_squared']:.4f}")
    if 'rf_r_squared' in sl:
        print(f"  Random Forest R2 = {sl['rf_r_squared']:.4f}")

    do = results['depth_only']
    print(f"\nDepth-only: drop ∝ D^{do['alpha']:.2f} (R2={do['r_squared']:.3f})")
    print(f"  → Deep networks suffer ~D^{do['alpha']:.1f} accuracy drop amplification")
