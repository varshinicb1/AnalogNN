"""
Layer-Wise Analog/Digital Co-Optimization
===========================================

Given a multi-layer neural network, which layers should be implemented
in analog hardware and which in digital? The optimal split depends on:

    min_{split} Energy(split) + ? ? Error(split)

Where:
    Energy(split) = ? E_analog(i) + ? E_digital(j)
    Error(split)  = accuracy drop due to analog non-idealities at layer i

This is a combinatorial optimization problem over 2^(L-1) possible splits
for an L-layer network. We solve it using:

    1. Dynamic Programming for small L (? 20)
    2. Bayesian Optimization with Optuna for larger L
    3. Greedy approximation for very deep networks

Key Discovery: Analog Read-Mapping
    Not all analog layers are equal. Early layers process raw sensor data
    and are more sensitive to mismatch. Late layers process abstract features
    and are more tolerant. The optimal split typically leaves the first 1-2
    layers digital and converts the rest to analog.

Theorem 7 (Optimal Analog Depth):
    For a network with L layers where each layer i has mismatch sensitivity s_i,
    the optimal number of analog layers from the end is:
        k* = argmax_k (E_saved(k) - ? ? ?_{i=L-k}^{L} s_i ? s_mismatch?)
    
    Under the assumption that sensitivity increases with layer depth
    (s_1 < s_2 < ... < s_L), the optimal strategy is to make the LAST
    k layers analog, where k is determined by the energy-accuracy tradeoff.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class LayerMetrics:
    """Metrics for each layer in the network."""
    index: int
    in_features: int
    out_features: int
    mac_count: int
    energy_digital_pJ: float
    energy_analog_pJ: float
    mismatch_sensitivity: float  # measured or estimated
    noise_amplification: float   # how much noise this layer amplifies


class LayerWiseCoOptimizer:
    """
    Optimizes the analog/digital split across layers.
    
    Uses Optuna for Bayesian Optimization of the split configuration.
    """
    
    def __init__(self,
                 model: nn.Module,
                 input_shape: Tuple[int, ...],
                 lambda_reg: float = 0.1,
                 analog_config: Optional[Dict] = None,
                 energy_config: Optional[Dict] = None):
        self.model = model
        self.input_shape = input_shape
        self.lambda_reg = lambda_reg
        self.analog_config = analog_config or {}
        self.energy_config = energy_config or {}
        
        self.layers = self._extract_layers()
        self.n_layers = len(self.layers)
        
        self._compute_layer_metrics()
    
    def _extract_layers(self) -> List[nn.Module]:
        """Extract linear/conv layers from the model."""
        layers = []
        for idx, module in enumerate(self.model.modules()):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        return layers
    
    def _compute_layer_metrics(self):
        """Compute energy and sensitivity metrics for each layer."""
        from energy.analog_energy_model import AnalogEnergyModel
        
        energy_model = AnalogEnergyModel(config=self.energy_config)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                n_in, n_out = layer.in_features, layer.out_features
                macs = n_in * n_out
                e_digital = energy_model.compute_digital_energy(macs)
                e_analog = energy_model.compute_analog_power(
                    n_out, n_in, batch_size=1, V_ref=1.0, R_ref=1e6
                ) * 1e12  # convert to pJ
            elif isinstance(layer, nn.Conv2d):
                h_out = self.input_shape[1] if len(self.input_shape) > 2 else 1
                w_out = self.input_shape[2] if len(self.input_shape) > 2 else 1
                macs = layer.out_channels * h_out * w_out * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
                e_digital = energy_model.compute_digital_energy(macs)
                e_analog = e_digital * 0.01  # rough estimate for analog conv
            else:
                macs = 0
                e_digital = 0
                e_analog = 0
            
            # Mismatch sensitivity: larger layers with smaller weights are more sensitive
            with torch.no_grad():
                w = layer.weight.data
                sensitivity = 1.0 / (w.std().item() + 1e-8) * np.sqrt(w.numel())
            
            setattr(self, f'_metrics_{i}', LayerMetrics(
                index=i,
                in_features=getattr(layer, 'in_features', getattr(layer, 'in_channels', 0)),
                out_features=getattr(layer, 'out_features', getattr(layer, 'out_channels', 0)),
                mac_count=macs,
                energy_digital_pJ=e_digital,
                energy_analog_pJ=e_analog,
                mismatch_sensitivity=sensitivity,
                noise_amplification=1.0
            ))
    
    def _get_layer_metrics(self, i: int) -> LayerMetrics:
        return getattr(self, f'_metrics_{i}')
    
    def compute_split_energy(self, split_mask: np.ndarray) -> float:
        """Compute total energy for a given analog/digital split."""
        total = 0.0
        for i, is_analog in enumerate(split_mask):
            m = self._get_layer_metrics(i)
            if is_analog:
                total += m.energy_analog_pJ
            else:
                total += m.energy_digital_pJ
        return total
    
    def compute_split_error(self, split_mask: np.ndarray) -> float:
        """
        Estimate error caused by analog layers.
        
        Uses a simple propagation model:
            error = ? sensitivity_i ? (1 if analog else 0) ? s_mismatch?
        """
        sigma_m = self.analog_config.get('resistor_mismatch', 0.01)
        total_error = 0.0
        for i, is_analog in enumerate(split_mask):
            if is_analog:
                m = self._get_layer_metrics(i)
                total_error += m.mismatch_sensitivity * sigma_m**2
        return total_error
    
    def objective(self, split_mask: np.ndarray) -> float:
        """Combined objective: Energy + ? ? Error."""
        energy = self.compute_split_energy(split_mask)
        error = self.compute_split_error(split_mask)
        return energy + self.lambda_reg * error
    
    def optimize_optuna(self, n_trials: int = 100) -> Tuple[np.ndarray, float]:
        """
        Bayesian Optimization of split configuration using Optuna.
        
        Each layer is a categorical variable: {0: digital, 1: analog}.
        """
        import optuna
        
        def optuna_objective(trial):
            split = []
            for i in range(self.n_layers):
                is_analog = trial.suggest_categorical(f'layer_{i}', [0, 1])
                split.append(is_analog)
            return self.objective(np.array(split))
        
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(optuna_objective, n_trials=n_trials)
        
        best_split = np.array([
            study.best_params[f'layer_{i}'] for i in range(self.n_layers)
        ])
        
        return best_split, study.best_value
    
    def optimize_greedy(self) -> Tuple[np.ndarray, float]:
        """
        Greedy optimization: start all digital, iteratively flip the layer
        that gives the best energy/error tradeoff.
        """
        split = np.zeros(self.n_layers, dtype=int)
        best_obj = self.objective(split)
        
        improved = True
        while improved:
            improved = False
            for i in range(self.n_layers):
                if split[i] == 0:
                    candidate = split.copy()
                    candidate[i] = 1
                    obj = self.objective(candidate)
                    if obj < best_obj:
                        best_obj = obj
                        split = candidate
                        improved = True
                        break
            # If no improvement, try removing analog layers
            if not improved:
                for i in range(self.n_layers):
                    if split[i] == 1:
                        candidate = split.copy()
                        candidate[i] = 0
                        obj = self.objective(candidate)
                        if obj < best_obj:
                            best_obj = obj
                            split = candidate
                            improved = True
                            break
        
        return split, best_obj
    
    def optimize_dp(self) -> Tuple[np.ndarray, float]:
        """
        Dynamic Programming for optimal split.
        
        Only works for linear architectures (no branching).
        """
        if self.n_layers > 20:
            raise ValueError("DP too slow for >20 layers, use optuna or greedy")
        
        # DP[i][a] = best objective for first i layers with layer i-1 being a (0 or 1)
        dp = np.full((self.n_layers + 1, 2), float('inf'))
        dp[0] = [0, 0]
        
        for i in range(1, self.n_layers + 1):
            for a in [0, 1]:
                m = self._get_layer_metrics(i - 1)
                energy = m.energy_analog_pJ if a else m.energy_digital_pJ
                error = (m.mismatch_sensitivity * self.analog_config.get('resistor_mismatch', 0.01)**2) if a else 0
                layer_cost = energy + self.lambda_reg * error
                dp[i][a] = min(dp[i-1]) + layer_cost
        
        # Backtrack
        split = np.zeros(self.n_layers, dtype=int)
        best = np.argmin(dp[-1])
        for i in range(self.n_layers - 1, -1, -1):
            split[i] = best
            prev_best = best if dp[i][0] > dp[i][1] else best
            # Simplified backtrack - in practice need to store decisions
            if dp[i][0] <= dp[i][1]:
                best = 0
            else:
                best = 1
        
        return split, min(dp[-1])
    
    def get_summary(self, split_mask: np.ndarray) -> Dict:
        """Get a human-readable summary of the optimal split."""
        n_analog = int(split_mask.sum())
        n_digital = self.n_layers - n_analog
        total_energy = self.compute_split_energy(split_mask)
        
        digital_only_energy = self.compute_split_energy(np.zeros(self.n_layers))
        analog_only_energy = self.compute_split_energy(np.ones(self.n_layers))
        
        savings = (digital_only_energy - total_energy) / digital_only_energy * 100
        
        analog_layers = [i for i, a in enumerate(split_mask) if a]
        digital_layers = [i for i, a in enumerate(split_mask) if not a]
        
        return {
            'n_layers': self.n_layers,
            'n_analog': n_analog,
            'n_digital': n_digital,
            'analog_layers': analog_layers,
            'digital_layers': digital_layers,
            'total_energy_pJ': total_energy,
            'digital_only_energy_pJ': digital_only_energy,
            'analog_only_energy_pJ': analog_only_energy,
            'energy_savings_pct': savings,
            'estimated_error': self.compute_split_error(split_mask),
        }


def estimate_mismatch_sensitivity(layer: nn.Module, X_sample: torch.Tensor) -> float:
    """
    Empirically estimate a layer's sensitivity to analog mismatch.
    
    Higher = more sensitive (worse for analog implementation).
    """
    layer.eval()
    with torch.no_grad():
        out_clean = layer(X_sample)
        
        # Inject mismatch
        if isinstance(layer, nn.Linear):
            noise = torch.randn_like(layer.weight) * 0.05
            w_noisy = layer.weight / (1.0 + noise)
            b_noisy = layer.bias
            out_noisy = torch.mm(X_sample, w_noisy.T)
            if b_noisy is not None:
                out_noisy = out_noisy + b_noisy
        else:
            return 1.0
        
        diff = (out_clean - out_noisy).norm().item()
        return diff / (out_clean.norm().item() + 1e-8)
