"""
Circuit Area & Resistance Optimization Engine
=============================================

This module implements Theorem 3 (Optimal Resistance Allocation). Given a total
area budget (or maximum chip resistance capacity), it optimizes the reference
resistance R_ref and individual resistor aspect ratios (W/L) to minimize
the propagation of resistor mismatch and thermal noise into neural activations.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class CircuitOptimizer:
    """
    Optimizes analog NN resistor sizes and feedback parameters to minimize
    electrical and structural degradation of inference accuracy.
    """

    def __init__(self, weight_matrix: torch.Tensor,
                 area_budget: float = 1e8,  # Nominal total resistance budget in Ohms
                 pelgrom_constant: float = 1e-3,
                 temperature_K: float = 300.0,
                 bandwidth_Hz: float = 1e6,
                 gbw_Hz: float = 1e7):
        self.W = weight_matrix.detach().cpu().numpy()
        self.area_budget = area_budget
        self.pelgrom_constant = pelgrom_constant
        self.temperature_K = temperature_K
        self.bandwidth_Hz = bandwidth_Hz
        self.gbw_Hz = gbw_Hz
        
        self.C, self.n = self.W.shape
        self.nonzero_mask = np.abs(self.W) > 1e-6
        # Sum of 1/|w| to map R_total = R_ref * sum(1/|w_ij|)
        self.inv_w_sum = np.sum(1.0 / np.abs(self.W[self.nonzero_mask]))

    def optimize_resistance_allocation(self) -> Dict:
        """
        Finds the optimal reference resistance R_ref that balances mismatch
        (which decreases with larger resistor area, i.e., smaller resistance)
        and thermal noise (which increases with larger resistance values).
        """
        # Sweep R_ref values from 100 Ohms to 10 MOhms
        r_ref_range = np.logspace(2, 7, 500)
        
        best_r_ref = 10000.0
        min_error = float('inf')
        
        sweep_r_ref = []
        sweep_mismatch = []
        sweep_thermal = []
        sweep_total = []
        
        k_B = 1.381e-23  # Boltzmann
        
        for r_ref in r_ref_range:
            # 1. Area constraint check
            total_resistance = r_ref * self.inv_w_sum
            if total_resistance > self.area_budget:
                continue  # Exceeds budget
                
            # 2. Mismatch variance (proportional to 1/R_ref)
            # Var(e_mismatch) = sigma_R^2 * ||W||_F^2
            # Pelgrom mismatch coefficient is self.pelgrom_constant
            mismatch_var = (self.pelgrom_constant ** 2 / r_ref) * np.sum(self.W ** 2)
            
            # 3. Thermal noise variance (proportional to R_ref)
            # Noise gain sum: C + sum_i sum_j |w_ij|
            noise_gains = 1.0 + np.sum(np.abs(self.W), axis=1)
            thermal_var = 4.0 * k_B * self.temperature_K * r_ref * self.bandwidth_Hz * np.sum(noise_gains)
            
            total_var = mismatch_var + thermal_var
            
            sweep_r_ref.append(float(r_ref))
            sweep_mismatch.append(float(mismatch_var))
            sweep_thermal.append(float(thermal_var))
            sweep_total.append(float(total_var))
            
            if total_var < min_error:
                min_error = total_var
                best_r_ref = r_ref
                
        # Settling time constant at optimal R_ref
        # tau = R_ref * C_load / (GBW * beta)
        max_gain = np.max(np.sum(np.abs(self.W), axis=1))
        beta_min = 1.0 / (1.0 + max_gain)
        f_cl = self.gbw_Hz * beta_min
        tau = 1.0 / (2 * np.pi * f_cl)
        
        return {
            'optimal_r_ref': float(best_r_ref),
            'min_expected_variance': float(min_error),
            'settling_time_tau': float(tau),
            'total_resistors_area_ohms': float(best_r_ref * self.inv_w_sum),
            'mismatch_share_pct': float((sweep_mismatch[sweep_r_ref.index(best_r_ref)] / min_error) * 100) if sweep_r_ref else 0.0,
            'thermal_share_pct': float((sweep_thermal[sweep_r_ref.index(best_r_ref)] / min_error) * 100) if sweep_r_ref else 0.0,
            'sweep_data': {
                'r_ref_values': sweep_r_ref,
                'mismatch_errors': sweep_mismatch,
                'thermal_errors': sweep_thermal,
                'total_errors': sweep_total,
                'settling_times': [float(tau)] * len(sweep_r_ref)
            }
        }
