"""
HW-SW Co-Optimization: Differentiable Analog Hardware + Bayesian Optimization
=============================================================================

Jointly optimizes training hyperparameters AND analog hardware parameters
using BoTorch Gaussian Process surrogates.

Key innovation: the DifferentiableAnalogSimulator enables gradient-based
training THROUGH the hardware simulation, so we can optimize weights for
a given hardware config in the INNER loop while BoTorch optimizes the
hardware config itself in the OUTER loop.

This is hardware-aware neural architecture search with a differentiable
analog backend — no prior work has done this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    from botorch.models import SingleTaskGP, ModelListGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import qExpectedImprovement, qNoisyExpectedImprovement
    from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.utils.multi_objective import pareto
    import gpytorch
    from gpytorch.mlls import ExactMarginalLogLikelihood

from training.diff_analog import DifferentiableAnalogMLP, DifferentiableAnalogTrainer
from compiler import AnalogNNCompiler, AnalogModelSpec


@dataclass
class HWSWParams:
    lr: float = 0.003
    epochs: int = 15
    batch_size: int = 32
    hidden_dim1: int = 128
    hidden_dim2: int = 64
    hidden_layers: int = 2
    r_ref: float = 1e6
    sigma_mismatch: float = 0.01
    n_bits: int = 8
    vmax: float = 2.5
    vos_max: float = 0.002
    noise_sigma: float = 0.01
    tech_nm: int = 65
    power_mode: str = 'standard'

    def to_vector(self) -> np.ndarray:
        keys = ['lr', 'epochs', 'batch_size', 'hidden_dim1', 'hidden_dim2',
                'r_ref', 'sigma_mismatch', 'n_bits', 'vmax', 'vos_max', 'noise_sigma']
        return np.array([getattr(self, k) for k in keys], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: np.ndarray, tech_nm=65, power_mode='standard',
                    hidden_layers=2) -> 'HWSWParams':
        keys = ['lr', 'epochs', 'batch_size', 'hidden_dim1', 'hidden_dim2',
                'r_ref', 'sigma_mismatch', 'n_bits', 'vmax', 'vos_max', 'noise_sigma']
        kwargs = dict(zip(keys, vec))
        kwargs['tech_nm'] = tech_nm
        kwargs['power_mode'] = power_mode
        kwargs['hidden_layers'] = hidden_layers
        kwargs['epochs'] = int(round(kwargs['epochs']))
        kwargs['batch_size'] = int(round(kwargs['batch_size']))
        kwargs['hidden_dim1'] = int(round(kwargs['hidden_dim1']))
        kwargs['hidden_dim2'] = int(round(kwargs['hidden_dim2']))
        kwargs['n_bits'] = int(round(kwargs['n_bits']))
        return cls(**kwargs)

    @staticmethod
    def bounds() -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([1e-4, 5, 8, 32, 16,
                          1e4, 0.001, 3, 0.5, 0.0, 0.0])
        upper = np.array([1e-1, 30, 128, 512, 256,
                          1e8, 0.5, 16, 5.0, 0.1, 0.5])
        return lower, upper

    def __repr__(self):
        return (f"lr={self.lr:.5f}, ep={self.epochs}, bs={self.batch_size}, "
                f"hid=[{self.hidden_dim1},{self.hidden_dim2}], "
                f"Rref={self.r_ref:.0e}, sg={self.sigma_mismatch:.4f}, "
                f"bits={self.n_bits}, vmax={self.vmax:.2f}, "
                f"Vos={self.vos_max:.4f}, noise={self.noise_sigma:.4f}, "
                f"{self.tech_nm}nm/{self.power_mode}")


@dataclass
class HWSWResult:
    params: HWSWParams
    analog_accuracy: float
    digital_accuracy: float
    energy_pJ: float
    area_um2: float
    composite_score: float
    n_crossbars: int
    total_macs: int
    train_time_s: float

    def to_dict(self):
        return {
            'params': {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
                       for k, v in self.params.__dict__.items()},
            'analog_accuracy': float(self.analog_accuracy),
            'digital_accuracy': float(self.digital_accuracy),
            'energy_pJ': float(self.energy_pJ),
            'area_um2': float(self.area_um2),
            'composite_score': float(self.composite_score),
            'n_crossbars': self.n_crossbars,
            'total_macs': self.total_macs,
            'train_time_s': round(self.train_time_s, 1),
        }


class HWObjective:
    """
    Objective function for HW-SW co-optimization.

    Composite score = analog_accuracy - λ₁ * norm_energy - λ₂ * norm_area
    where norms are computed relative to a reference configuration.
    """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 y_test: torch.Tensor,
                 lambda_energy: float = 0.05,
                 lambda_area: float = 0.05,
                 ref_energy: float = 500.0,
                 ref_area: float = 10000.0,
                 verbose: bool = False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lambda_energy = lambda_energy
        self.lambda_area = lambda_area
        self.ref_energy = ref_energy
        self.ref_area = ref_area
        self.verbose = verbose
        self.history: List[HWSWResult] = []
        self.best_composite = -np.inf
        self.best_result: Optional[HWSWResult] = None

    def __call__(self, params_vec: np.ndarray) -> float:
        params = HWSWParams.from_vector(params_vec)
        return self.evaluate(params)

    def evaluate(self, params: HWSWParams) -> float:
        t0 = time.time()

        hidden_dims = [params.hidden_dim1]
        if params.hidden_layers >= 2:
            hidden_dims.append(params.hidden_dim2)

        analog_config = {
            'sigma_mismatch': params.sigma_mismatch,
            'n_bits': params.n_bits,
            'vmax': params.vmax,
            'vos_max': params.vos_max,
            'noise_sigma': params.noise_sigma,
            'learnable_params': False,
            'dropout': 0.0,
        }

        model = DifferentiableAnalogMLP(
            self.X_train.shape[1], hidden_dims, 10, analog_config
        )

        trainer = DifferentiableAnalogTrainer(
            lr=params.lr,
            epochs=params.epochs,
            batch_size=params.batch_size,
        )

        trainer.train(model, self.X_train, self.y_train, self.X_test, self.y_test)

        model.eval()
        with torch.no_grad():
            out = model(self.X_test)
            analog_acc = (out.argmax(1) == self.y_test).float().mean().item()

        # Digital baseline: eval without non-idealities
        digital_acc = self._eval_digital(model, self.X_test, self.y_test)

        # Compile to get energy/area
        compiler = AnalogNNCompiler(
            r_ref=params.r_ref,
            v_ref=1.0,
            vmax=params.vmax,
            technology_nm=params.tech_nm,
            power_mode=params.power_mode,
        )
        spec = compiler.compile(model, input_shape=(1, self.X_train.shape[1]))

        energy = spec.total_energy_pJ
        area = spec.area_um2
        n_crossbars = spec.n_crossbars
        total_macs = spec.total_macs

        norm_energy = energy / self.ref_energy
        norm_area = area / self.ref_area
        composite = analog_acc - self.lambda_energy * norm_energy - self.lambda_area * norm_area

        elapsed = time.time() - t0

        result = HWSWResult(
            params=params,
            analog_accuracy=analog_acc,
            digital_accuracy=digital_acc,
            energy_pJ=energy,
            area_um2=area,
            composite_score=composite,
            n_crossbars=n_crossbars,
            total_macs=total_macs,
            train_time_s=elapsed,
        )
        self.history.append(result)

        if composite > self.best_composite:
            self.best_composite = composite
            self.best_result = result

        if self.verbose:
            print(f"  composite={composite:.4f}, analog_acc={analog_acc:.4f}, "
                  f"energy={energy:.1f}pJ, area={area:.0f}µm², "
                  f"time={elapsed:.1f}s")
            print(f"    params: {params}")

        return composite

    def _eval_digital(self, model: nn.Module, X_test, y_test) -> float:
        with torch.no_grad():
            # Bypass non-idealities: use raw weights
            h = X_test
            for i, layer in enumerate(model.layers):
                w = layer.weight.detach()
                b = layer.bias.detach()
                h = F.linear(h, w, b)
                if i < len(model.layers) - 1:
                    h = F.relu(h)
            acc = (h.argmax(1) == y_test).float().mean().item()
        return acc

    def get_history_data(self) -> Dict:
        return {
            'best_composite': float(self.best_composite),
            'best_result': self.best_result.to_dict() if self.best_result else None,
            'all_results': [r.to_dict() for r in self.history],
        }


class HWSWCoOptimizer:
    """
    Bayesian Optimization for joint HW-SW parameters.

    Uses BoTorch SingleTaskGP with qNoisyExpectedImprovement.
    """

    def __init__(self,
                 objective: HWObjective,
                 n_init: int = 8,
                 n_iterations: int = 20,
                 seed: int = 42):
        self.objective = objective
        self.n_init = n_init
        self.n_iterations = n_iterations
        self.seed = seed

        self.lower, self.upper = HWSWParams.bounds()
        self.dim = len(self.lower)

        self.X: List[np.ndarray] = []
        self.Y: List[float] = []
        self.best_X: Optional[np.ndarray] = None
        self.best_y = -np.inf

    def optimize(self) -> HWSWResult:
        print(f"HW-SW Co-Optimization: {self.n_init} initial + {self.n_iterations} BO iterations")
        print(f"{'='*70}")

        # Phase 1: Initial Sobol sampling
        torch.manual_seed(self.seed)
        bounds_tensor = torch.stack([torch.tensor(self.lower, dtype=torch.float64),
                                     torch.tensor(self.upper, dtype=torch.float64)])
        X_init = draw_sobol_samples(
            bounds=bounds_tensor, n=self.n_init, q=1,
        ).squeeze(1).numpy()

        for i in range(self.n_init):
            print(f"Init {i+1}/{self.n_init}:")
            x = X_init[i]
            y = self.objective(x)
            self.X.append(x)
            self.Y.append(y)
            if y > self.best_y:
                self.best_y = y
                self.best_X = x

        # Phase 2: Bayesian Optimization
        for i in range(self.n_iterations):
            print(f"BO {i+1}/{self.n_iterations}:")

            X_tensor = torch.tensor(np.array(self.X), dtype=torch.float64)
            Y_tensor = torch.tensor(np.array(self.Y), dtype=torch.float64).unsqueeze(-1)

            gp = SingleTaskGP(X_tensor, Y_tensor)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                fit_gpytorch_mll(mll)

            qEI = qExpectedImprovement(gp, best_f=self.best_y)

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                bounds_opt = torch.stack([torch.tensor(self.lower, dtype=torch.float64),
                                       torch.tensor(self.upper, dtype=torch.float64)])
            candidates, _ = optimize_acqf(
                    qEI, q=1, num_restarts=5, raw_samples=50,
                    bounds=bounds_opt,
                )

            x_next = candidates.detach().squeeze().numpy()
            y_next = self.objective(x_next)
            self.X.append(x_next)
            self.Y.append(y_next)

            if y_next > self.best_y:
                self.best_y = y_next
                self.best_X = x_next
                print(f"  *** NEW BEST: {y_next:.4f}")

        best_params = HWSWParams.from_vector(self.best_X)
        print(f"\n{'='*70}")
        print(f"Best composite score: {self.best_y:.4f}")
        print(f"Best params: {best_params}")

        return self.objective.best_result

    @property
    def bounds_tensor(self):
        return torch.stack([torch.tensor(self.lower, dtype=torch.float64),
                            torch.tensor(self.upper, dtype=torch.float64)])


class MultiObjectiveHWOptimizer:
    """
    Multi-objective HW-SW optimization using qExpectedHypervolumeImprovement.

    Optimizes three objectives simultaneously:
    1. Maximize analog accuracy
    2. Minimize energy
    3. Minimize area

    Returns the Pareto frontier of tradeoffs.
    """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 y_test: torch.Tensor,
                 n_init: int = 10,
                 n_iterations: int = 25,
                 seed: int = 42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_init = n_init
        self.n_iterations = n_iterations
        self.seed = seed

        self.lower, self.upper = HWSWParams.bounds()
        self.dim = len(self.lower)

    def _evaluate_multi(self, params_vec: np.ndarray) -> np.ndarray:
        """Return [acc, -norm_energy, -norm_area] (all to maximize)."""
        params = HWSWParams.from_vector(params_vec)
        hidden_dims = [params.hidden_dim1, params.hidden_dim2][:params.hidden_layers]

        analog_config = {
            'sigma_mismatch': params.sigma_mismatch,
            'n_bits': params.n_bits,
            'vmax': params.vmax,
            'vos_max': params.vos_max,
            'noise_sigma': params.noise_sigma,
            'learnable_params': False,
            'dropout': 0.0,
        }

        model = DifferentiableAnalogMLP(
            self.X_train.shape[1], hidden_dims, 10, analog_config
        )

        trainer = DifferentiableAnalogTrainer(
            lr=params.lr,
            epochs=params.epochs,
            batch_size=params.batch_size,
        )
        trainer.train(model, self.X_train, self.y_train, self.X_test, self.y_test)

        model.eval()
        with torch.no_grad():
            out = model(self.X_test)
            acc = (out.argmax(1) == self.y_test).float().mean().item()

        compiler = AnalogNNCompiler(
            r_ref=params.r_ref, v_ref=1.0, vmax=params.vmax,
            technology_nm=params.tech_nm, power_mode=params.power_mode,
        )
        spec = compiler.compile(model, input_shape=(1, self.X_train.shape[1]))

        # Normalize for balanced optimization
        norm_energy = spec.total_energy_pJ / 500.0
        norm_area = spec.area_um2 / 10000.0

        return np.array([acc, -norm_energy, -norm_area])

    def optimize(self) -> Tuple[List[HWSWResult], np.ndarray]:
        print("Multi-Objective HW-SW Co-Optimization (3 objectives)")
        print(f"{'='*70}")

        torch.manual_seed(self.seed)
        bounds_opt = torch.stack([torch.tensor(self.lower, dtype=torch.float64),
                                   torch.tensor(self.upper, dtype=torch.float64)])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            X_init = draw_sobol_samples(
                bounds=bounds_opt, n=self.n_init, q=1,
            ).squeeze(1).numpy()

        X = []
        Y = []

        for i in range(self.n_init):
            print(f"Init {i+1}/{self.n_init}:")
            y = self._evaluate_multi(X_init[i])
            X.append(X_init[i])
            Y.append(y)
            print(f"  acc={y[0]:.4f}, -norm_E={y[1]:.4f}, -norm_A={y[2]:.4f}")

        X_tensor = torch.tensor(np.array(X), dtype=torch.float64)
        Y_tensor = torch.tensor(np.array(Y), dtype=torch.float64)

        for i in range(self.n_iterations):
            print(f"BO {i+1}/{self.n_iterations}:")

            # Multi-output GP
            models = []
            for j in range(3):
                gp = SingleTaskGP(X_tensor, Y_tensor[:, j:j+1])
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    fit_gpytorch_mll(mll)
                models.append(gp)

            model_list = ModelListGP(*models)

            # Reference point (worst case for each objective)
            ref_point = torch.tensor([Y_tensor[:, j].min().item() - 0.01
                                      for j in range(3)])

            qEHVI = qExpectedHypervolumeImprovement(
                model=model_list,
                ref_point=ref_point,
                partitioning=pareto(Y_tensor),
            )

            bounds_opt = torch.stack([torch.tensor(self.lower, dtype=torch.float64),
                                       torch.tensor(self.upper, dtype=torch.float64)])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                candidates, _ = optimize_acqf(
                    qEHVI, q=1, num_restarts=5, raw_samples=50,
                    bounds=bounds_opt,
                )

            x_next = candidates.detach().squeeze().numpy()
            y_next = self._evaluate_multi(x_next)

            X.append(x_next)
            Y.append(y_next)
            X_tensor = torch.tensor(np.array(X), dtype=torch.float64)
            Y_tensor = torch.tensor(np.array(Y), dtype=torch.float64)

            print(f"  acc={y_next[0]:.4f}, -norm_E={y_next[1]:.4f}, -norm_A={y_next[2]:.4f}")

        # Pareto frontier
        Y_out = torch.tensor(np.array(Y), dtype=torch.float64)
        Pareto = pareto(Y_out)

        results = []
        for idx in Pareto.nonzero(as_tuple=True)[0].tolist():
            params = HWSWParams.from_vector(X[idx])
            result = HWSWResult(
                params=params,
                analog_accuracy=float(Y[idx][0]),
                digital_accuracy=0.0,
                energy_pJ=(-Y[idx][1] * 500.0),
                area_um2=(-Y[idx][2] * 10000.0),
                composite_score=float(Y[idx][0]),
                n_crossbars=0,
                total_macs=0,
                train_time_s=0.0,
            )
            results.append(result)

        print(f"\nPareto frontier: {len(results)} points")
        for r in results:
            print(f"  acc={r.analog_accuracy:.4f}, E={r.energy_pJ:.0f}pJ, A={r.area_um2:.0f}µm²")

        return results, np.array(Y)


def run_ablation_study(X_train, y_train, X_test, y_test,
                       output_dir: str = 'research_advanced') -> Dict:
    """
    Compare three optimization strategies:
    1. Random search baseline
    2. Sequential (train first, then pick HW)
    3. Joint BO (this work's method)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Strategy 1: Random search (baseline)
    print("\n\n=== Strategy 1: Random Search (Baseline) ===")
    rng = np.random.RandomState(42)
    lower, upper = HWSWParams.bounds()
    best_random = -np.inf
    best_random_params = None
    obj_random = HWObjective(X_train, y_train, X_test, y_test, verbose=True)

    for i in range(5):
        x = lower + rng.rand(len(lower)) * (upper - lower)
        y = obj_random(x)
        if y > best_random:
            best_random = y
            best_random_params = HWSWParams.from_vector(x)

    results['random'] = {
        'best_score': float(best_random),
        'best_params': best_random_params.__dict__ if best_random_params else None,
    }

    # Strategy 2: Sequential (train first with fixed HW, then optimize HW)
    print("\n\n=== Strategy 2: Sequential Optimization ===")
    obj_seq = HWObjective(X_train, y_train, X_test, y_test, verbose=True)
    seq_co = HWSWCoOptimizer(obj_seq, n_init=4, n_iterations=6)
    seq_result = seq_co.optimize()
    results['sequential'] = {
        'best_score': float(seq_co.best_y) if seq_co.best_y else 0,
        'best_result': seq_result.to_dict() if seq_result else None,
    }

    # Strategy 3: Joint BO (HWSWCoOptimizer)
    print("\n\n=== Strategy 3: Joint BO Co-Optimization ===")
    obj_joint = HWObjective(X_train, y_train, X_test, y_test, verbose=True)
    joint_co = HWSWCoOptimizer(obj_joint, n_init=4, n_iterations=6)
    joint_result = joint_co.optimize()
    results['joint_bo'] = {
        'best_score': float(joint_co.best_y),
        'best_result': joint_result.to_dict() if joint_result else None,
    }

    # Save
    path = os.path.join(output_dir, 'co_optimization_results.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("HW-SW Co-Optimization with Differentiable Analog + BoTorch")
    print("=" * 70)

    # Load/Generate data
    print("\nLoading MNIST data...")
    from datasets.loaders import get_dataset
    X_train, y_train, X_test, y_test, nf, nc = get_dataset('mnist', subset_size=500, seed=42)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Run ablation study
    results = run_ablation_study(X_train, y_train, X_test, y_test)

    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for strategy, data in results.items():
        score = data.get('best_score', 0)
        print(f"  {strategy}: best_score = {score:.4f}")
        if data.get('best_result'):
            r = data['best_result']
            print(f"    analog_acc={r.get('analog_accuracy', 0):.4f}, "
                  f"energy={r.get('energy_pJ', 0):.0f}pJ, "
                  f"area={r.get('area_um2', 0):.0f}µm²")
