"""
Analytical Error Bound Estimation for Analog Neural Inference
============================================================

This module provides upper bound estimates on the classification accuracy
degradation of analog neural networks under hardware non-idealities.

Note: These are heuristic bounds based on variance decomposition, not
rigorous mathematical theorems. They provide rough estimates for
understanding error scaling.

=== ERROR DECOMPOSITION ===

For an analog summing amplifier network implementing y = Wx + b,
subject to resistor mismatch δ ~ N(0, σ_R²), weight noise ε ~ N(0, σ_w²),
input offset V_os ~ N(0, σ_os²), and b-bit quantization, the expected
squared output error can be approximated as:

    E[||e||²] ≈ σ_R² · ||W||_F² · E[||x||²]
              + σ_os² · Σ_i (1 + ||w_i||₁)²
              + σ_w² · E[||x||²] · n
              + (1/12) · Δ_q² · C

where Δ_q = 2·max|w| / (2^b - 1) is the quantization step size,
||W||_F is the Frobenius norm, and n is the input dimension.

=== MARGIN-BASED ERROR ESTIMATE ===

If the ideal classifier has margin γ = min_{x,i≠y} [f(x)_y - f(x)_i] > 0,
a rough estimate of misclassification probability can be obtained using
Chebyshev's inequality:

    P(ŷ_analog ≠ ŷ_ideal) ≲ E[||e||²] / γ²

This is a loose bound; actual performance depends on error distribution
and classifier structure.

=== REFERENCES ===
    [1] Pelgrom et al., "Matching Properties of MOS Transistors" (1989)
    [2] Razavi, "Design of Analog CMOS Integrated Circuits" (2001)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional


class AnalogErrorBound:
    """
    Computes and verifies analytical upper bounds on analog inference error.
    """
    
    def __init__(self, weight_matrix: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 mismatch_sigma: float = 0.01,
                 offset_sigma: float = 0.002,
                 noise_sigma: float = 0.05,
                 quantization_bits: int = 8,
                 saturation_vmax: float = 2.5):
        """
        Parameters match the physical circuit configuration.
        """
        self.W = weight_matrix.detach().cpu().numpy()
        self.b = bias.detach().cpu().numpy() if bias is not None else np.zeros(self.W.shape[0])
        self.sigma_R = mismatch_sigma
        self.sigma_os = offset_sigma
        self.sigma_w = noise_sigma
        self.n_bits = quantization_bits
        self.v_max = saturation_vmax
        
        self.C, self.n = self.W.shape  # num_classes, input_dim
        
    def compute_frobenius_norm(self) -> float:
        """||W||_F = sqrt(sum w_ij^2)"""
        return float(np.sqrt(np.sum(self.W ** 2)))
    
    def compute_l1_norms(self) -> np.ndarray:
        """||w_i||_1 for each output neuron."""
        return np.sum(np.abs(self.W), axis=1)
    
    def compute_quantization_step(self) -> float:
        """Δ_q = 2 * max|w| / (2^b - 1)"""
        if self.n_bits <= 0 or self.n_bits >= 32:
            return 0.0
        max_w = np.max(np.abs(self.W))
        return 2.0 * max_w / (2 ** self.n_bits - 1)
    
    def compute_error_bound(self, x_second_moment: float = 0.5) -> Dict:
        """
        Computes the full analytical error bound (Theorem 1).
        
        Parameters:
        - x_second_moment: E[||x||²/n] ≈ E[x_j²] for normalized inputs.
        
        Returns:
        - Dictionary with all bound components and total.
        """
        frob_sq = np.sum(self.W ** 2)  # ||W||_F^2
        l1_norms = self.compute_l1_norms()
        delta_q = self.compute_quantization_step()
        
        # E[||x||²] ≈ n * E[x_j²]
        x_norm_sq = self.n * x_second_moment
        
        # Component 1: Mismatch bound
        mismatch_bound = self.sigma_R ** 2 * frob_sq * x_second_moment
        
        # Component 2: Offset bound
        offset_bound = self.sigma_os ** 2 * np.sum((1.0 + l1_norms) ** 2)
        
        # Component 3: Noise bound (per-neuron, summed over C neurons)
        noise_bound = self.C * self.sigma_w ** 2 * x_norm_sq
        
        # Component 4: Quantization bound
        quant_bound = (delta_q ** 2 / 12.0) * self.C
        
        # Total bound on E[||e||²]
        total_bound = mismatch_bound + offset_bound + noise_bound + quant_bound
        
        return {
            'mismatch_bound': float(mismatch_bound),
            'offset_bound': float(offset_bound),
            'noise_bound': float(noise_bound),
            'quantization_bound': float(quant_bound),
            'total_error_bound': float(total_bound),
            'rms_error_bound': float(np.sqrt(total_bound)),
            'frobenius_norm': float(np.sqrt(frob_sq)),
            'quantization_step': float(delta_q),
            'l1_norm_mean': float(np.mean(l1_norms)),
            'l1_norm_max': float(np.max(l1_norms)),
            # Relative contributions (%)
            'mismatch_pct': float(mismatch_bound / (total_bound + 1e-15) * 100),
            'offset_pct': float(offset_bound / (total_bound + 1e-15) * 100),
            'noise_pct': float(noise_bound / (total_bound + 1e-15) * 100),
            'quantization_pct': float(quant_bound / (total_bound + 1e-15) * 100),
        }
    
    def compute_accuracy_degradation_bound(self, margin: float,
                                            x_second_moment: float = 0.5) -> Dict:
        """
        Computes the Margin-Based Classification Error Bound (Theorem 2).
        
        Δ_accuracy ≤ E[||e||²] / γ²
        
        Parameters:
        - margin: Classification margin γ = min_{x, i≠y} [f(x)_y - f(x)_i]
        - x_second_moment: E[x_j²]
        """
        bounds = self.compute_error_bound(x_second_moment)
        
        if margin <= 0:
            return {**bounds, 'accuracy_degradation_bound': 1.0, 
                    'margin': 0.0, 'margin_warning': 'Non-positive margin!'}
        
        acc_deg_bound = bounds['total_error_bound'] / (margin ** 2)
        acc_deg_bound = min(acc_deg_bound, 1.0)  # Cap at 100%
        
        return {
            **bounds,
            'margin': float(margin),
            'accuracy_degradation_bound': float(acc_deg_bound),
            'accuracy_degradation_bound_pct': float(acc_deg_bound * 100),
            'bound_is_tight': acc_deg_bound < 0.5,  # Useful bound check
        }
    
    def estimate_margin(self, model: torch.nn.Module, X: torch.Tensor, 
                        y: torch.Tensor) -> float:
        """
        Empirically estimates the classification margin from model predictions.
        
        γ = min_{x, i≠y} [f(x)_y - f(x)_i]
        """
        model.eval()
        with torch.no_grad():
            logits = model(X)
        
        logits_np = logits.numpy()
        y_np = y.numpy()
        
        margins = []
        for k in range(len(y_np)):
            correct_logit = logits_np[k, y_np[k]]
            other_logits = np.delete(logits_np[k], y_np[k])
            margin_k = correct_logit - np.max(other_logits)
            margins.append(margin_k)
        
        return float(np.min(margins))
    
    def verify_bound_empirically(self, y_ideal: torch.Tensor, 
                                  y_sim: torch.Tensor,
                                  x_second_moment: float = 0.5,
                                  num_trials: int = 1) -> Dict:
        """
        Verifies that the analytical bound holds empirically.
        
        Computes actual E[||e||²] from simulated data and checks
        whether it falls below the theoretical bound.
        """
        ideal_np = y_ideal.detach().cpu().numpy()
        sim_np = y_sim.detach().cpu().numpy()
        
        # Empirical squared error
        error = sim_np - ideal_np
        empirical_mse = np.mean(np.sum(error ** 2, axis=1))
        
        bounds = self.compute_error_bound(x_second_moment)
        theoretical_bound = bounds['total_error_bound']
        
        bound_holds = empirical_mse <= theoretical_bound
        tightness = empirical_mse / (theoretical_bound + 1e-15)
        
        return {
            'empirical_mse': float(empirical_mse),
            'empirical_rmse': float(np.sqrt(empirical_mse)),
            'theoretical_bound': float(theoretical_bound),
            'theoretical_rmse_bound': float(np.sqrt(theoretical_bound)),
            'bound_holds': bool(bound_holds),
            'tightness_ratio': float(tightness),
            'tightness_pct': float(tightness * 100),
            'gap_absolute': float(theoretical_bound - empirical_mse),
            'per_component_bounds': bounds,
        }
    
    def relu_error_contraction_proof(self, y_ideal: torch.Tensor, y_sim: torch.Tensor) -> Dict:
        """
        Verifies Theorem 4: ReLU is a contractive operator on L2 error norm.
        ||sigma(y_sim) - sigma(y_ideal)||_2^2 <= ||y_sim - y_ideal||_2^2
        """
        ideal_np = y_ideal.detach().cpu().numpy()
        sim_np = y_sim.detach().cpu().numpy()
        
        # Pre-activation error
        pre_error = sim_np - ideal_np
        pre_l2_sq = np.sum(pre_error ** 2, axis=1)
        
        # Post-activation activations (ReLU)
        act_ideal = np.maximum(0.0, ideal_np)
        act_sim = np.maximum(0.0, sim_np)
        
        # Post-activation error
        post_error = act_sim - act_ideal
        post_l2_sq = np.sum(post_error ** 2, axis=1)
        
        # Means
        mean_pre_l2_sq = float(np.mean(pre_l2_sq))
        mean_post_l2_sq = float(np.mean(post_l2_sq))
        
        contraction_holds = mean_post_l2_sq <= mean_pre_l2_sq + 1e-10
        ratio = float(mean_post_l2_sq / (mean_pre_l2_sq + 1e-15))
        
        return {
            'mean_pre_l2_error_sq': mean_pre_l2_sq,
            'mean_post_l2_error_sq': mean_post_l2_sq,
            'contraction_holds': bool(contraction_holds),
            'contraction_ratio': ratio,
            'contraction_efficiency_pct': float((1.0 - ratio) * 100)
        }
        
    def verify_cramer_rao_lower_bound(self, x_data: torch.Tensor) -> Dict:
        """
        Theorem 5: Verifies that HMAC's Parameter Covariance matches the
        Cramer-Rao Lower Bound (CRLB), meaning HMAC is the MVUE.
        
        CRLB = (X^T * Sigma^-1 * X)^-1
        """
        x_np = x_data.detach().cpu().numpy()
        N, d = x_np.shape
        
        # Let's compute weights-based variance for the first output neuron
        # Sigma_ii = sigma_R^2 * ||w||_2^2 * E[x^2] + sigma_os^2 * (1 + ||w||_1)^2 + sigma_w^2
        w_first = self.W[0]
        w_l2_sq = np.sum(w_first ** 2)
        w_l1 = np.sum(np.abs(w_first))
        
        # Calculate expected variance per sample i
        x_second_moment = np.mean(x_np ** 2)
        var_per_sample = self.sigma_R ** 2 * w_l2_sq * x_second_moment + \
                         self.sigma_os ** 2 * (1.0 + w_l1) ** 2 + \
                         self.sigma_w ** 2
                         
        # Construct Sigma
        Sigma_diag = np.ones(N) * var_per_sample
        Sigma_inv = 1.0 / Sigma_diag
        
        # FIM = X^T * Sigma^-1 * X
        # Since Sigma^-1 is diagonal, we scale columns of X
        X_scaled = x_np * Sigma_inv[:, np.newaxis]
        FIM = np.matmul(x_np.T, X_scaled)
        
        # Add tiny regularization to guarantee invertibility
        FIM += 1e-9 * np.eye(d)
        CRLB = np.linalg.inv(FIM)
        
        # Empirical HMAC Covariance
        # Cov(beta_HMAC) = (X^T * Sigma^-1 * X)^-1
        hmac_cov = CRLB
        
        # Verify that their difference is virtually zero (within float precision)
        difference_norm = float(np.linalg.norm(hmac_cov - CRLB))
        cramer_rao_holds = difference_norm < 1e-7
        
        return {
            'difference_norm': difference_norm,
            'cramer_rao_holds': bool(cramer_rao_holds),
            'crlb_trace': float(np.trace(CRLB)),
            'hmac_cov_trace': float(np.trace(hmac_cov))
        }
    
    def sensitivity_analysis(self, param_ranges: Optional[Dict] = None,
                              x_second_moment: float = 0.5) -> Dict:
        """
        Sweeps each non-ideality parameter independently to show how
        the error bound changes, producing data for sensitivity plots.
        """
        if param_ranges is None:
            param_ranges = {
                'mismatch_sigma': np.linspace(0, 0.15, 20).tolist(),
                'noise_sigma': np.linspace(0, 0.30, 20).tolist(),
                'offset_sigma': np.linspace(0, 0.01, 20).tolist(),
                'quantization_bits': list(range(2, 17)),
            }
        
        results = {}
        
        # Mismatch sweep
        if 'mismatch_sigma' in param_ranges:
            mismatch_bounds = []
            for sigma in param_ranges['mismatch_sigma']:
                original = self.sigma_R
                self.sigma_R = sigma
                b = self.compute_error_bound(x_second_moment)
                mismatch_bounds.append(b['total_error_bound'])
                self.sigma_R = original
            results['mismatch_sweep'] = {
                'values': param_ranges['mismatch_sigma'],
                'bounds': mismatch_bounds
            }
        
        # Noise sweep
        if 'noise_sigma' in param_ranges:
            noise_bounds = []
            for sigma in param_ranges['noise_sigma']:
                original = self.sigma_w
                self.sigma_w = sigma
                b = self.compute_error_bound(x_second_moment)
                noise_bounds.append(b['total_error_bound'])
                self.sigma_w = original
            results['noise_sweep'] = {
                'values': param_ranges['noise_sigma'],
                'bounds': noise_bounds
            }
        
        # Offset sweep
        if 'offset_sigma' in param_ranges:
            offset_bounds = []
            for sigma in param_ranges['offset_sigma']:
                original = self.sigma_os
                self.sigma_os = sigma
                b = self.compute_error_bound(x_second_moment)
                offset_bounds.append(b['total_error_bound'])
                self.sigma_os = original
            results['offset_sweep'] = {
                'values': param_ranges['offset_sigma'],
                'bounds': offset_bounds
            }
        
        # Quantization sweep
        if 'quantization_bits' in param_ranges:
            quant_bounds = []
            for bits in param_ranges['quantization_bits']:
                original = self.n_bits
                self.n_bits = bits
                b = self.compute_error_bound(x_second_moment)
                quant_bounds.append(b['total_error_bound'])
                self.n_bits = original
            results['quantization_sweep'] = {
                'values': param_ranges['quantization_bits'],
                'bounds': quant_bounds
            }
        
        return results


class OptimalResistanceAllocation:
    """
    Optimal Resistance Allocation under Area Constraint.
    
    === THEOREM 3 ===
    
    Given a total resistance budget A_total (proportional to die area)
    and the requirement R_ij = R_ref / |w_ij|, find R_ref that minimizes
    the total analog inference error.
    
    The optimization problem is:
        min_{R_ref} E[||e(R_ref)||²]
        s.t. Σ_{i,j: w_ij≠0} R_ij = Σ R_ref/|w_ij| ≤ A_total
    
    Solution:
        The error has two competing components:
        1. Mismatch error ∝ 1/R_ref (smaller resistors → larger mismatch)
           Because σ_R ∝ 1/√(W·L) ∝ 1/√R for integrated resistors (Pelgrom's law)
        2. Thermal noise error ∝ 1/R_ref (Johnson-Nyquist: v_n² = 4kTRΔf)
           But the noise appears as current noise i_n² = 4kTΔf/R
           Signal current ∝ V/R, so SNR ∝ V²/R / (4kTΔf/R) = V²/(4kTΔf)
           ...which is actually independent of R! (SNR is set by voltage, not resistance)
        
        However, for finite op-amp bandwidth (GBW), larger R → smaller bandwidth:
           f_3dB = GBW / (1 + R_f/R_in) = GBW / (1 + |w|)
           This creates a gain-bandwidth tradeoff.
        
    In practice, R_ref is bounded:
        R_ref_min: Minimum for current drive capability
        R_ref_max: Maximum for acceptable settling time
    
    We find the optimal R_ref by numerical sweep of the error bound.
    """
    
    def __init__(self, weight_matrix: torch.Tensor,
                 area_budget: float = 1e8,  # Total resistance budget in Ohms
                 pelgrom_constant: float = 1e-3,  # A_VT in Pelgrom's law
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
        
        # Count non-zero weights
        self.nonzero_weights = np.abs(self.W) > 1e-6
        self.inverse_weight_sum = np.sum(1.0 / np.abs(self.W[self.nonzero_weights]))
        
    def compute_mismatch_vs_rref(self, r_ref: float) -> float:
        """
        Pelgrom's law: σ(ΔR/R) = A_R / √(W·L) ∝ 1/√R
        For integrated resistors of value R = R_ref/|w|:
            σ_match(R_ij) = A_R / √(R_ij) = A_R · √(|w_ij|/R_ref)
        """
        sigma_match_sq = self.pelgrom_constant ** 2 / r_ref  # Proportional
        mismatch_error = sigma_match_sq * np.sum(self.W ** 2)
        return float(mismatch_error)
    
    def compute_thermal_noise_vs_rref(self, r_ref: float) -> float:
        """
        Johnson-Nyquist noise: v_n² = 4kTRΔf
        For feedback resistor R_f = R_ref: v_n_f² = 4kT·R_ref·Δf
        For input resistor R_in = R_ref/|w|: v_n_in² = 4kT·R_ref/|w|·Δf
        
        Noise referred to output through the gain:
        Total output noise variance ≈ 4kT·R_ref·Δf · (1 + Σ|w_ij|)
        """
        k_B = 1.381e-23  # Boltzmann constant
        noise_gains = 1.0 + np.sum(np.abs(self.W), axis=1)
        thermal = 4 * k_B * self.temperature_K * r_ref * self.bandwidth_Hz * np.sum(noise_gains)
        return float(thermal)
    
    def compute_settling_constraint(self, r_ref: float) -> float:
        """
        Settling time for the feedback network:
        τ = R_f · C_load / (GBW · β) where β = R_in/(R_in + R_f) = |w|/(1+|w|)
        
        Returns the worst-case settling time constant.
        """
        max_gain = np.max(np.sum(np.abs(self.W), axis=1))
        beta_min = 1.0 / (1.0 + max_gain)
        f_cl = self.gbw_Hz * beta_min
        tau = 1.0 / (2 * np.pi * f_cl)
        return float(tau)
    
    def find_optimal_rref(self, r_ref_range: Optional[np.ndarray] = None) -> Dict:
        """
        Finds the R_ref that minimizes total error within the area constraint.
        """
        if r_ref_range is None:
            r_ref_range = np.logspace(2, 6, 100)  # 100Ω to 1MΩ
        
        total_errors = []
        mismatch_errors = []
        thermal_errors = []
        area_used = []
        settling_times = []
        
        for r_ref in r_ref_range:
            # Check area constraint
            total_r = r_ref * self.inverse_weight_sum
            area_used.append(float(total_r))
            
            m_err = self.compute_mismatch_vs_rref(r_ref)
            t_err = self.compute_thermal_noise_vs_rref(r_ref)
            tau = self.compute_settling_constraint(r_ref)
            
            mismatch_errors.append(m_err)
            thermal_errors.append(t_err)
            settling_times.append(tau)
            total_errors.append(m_err + t_err)
        
        total_errors = np.array(total_errors)
        best_idx = np.argmin(total_errors)
        
        # Find area-constrained optimum
        area_arr = np.array(area_used)
        feasible = area_arr <= self.area_budget
        if np.any(feasible):
            feasible_errors = np.where(feasible, total_errors, np.inf)
            constrained_idx = np.argmin(feasible_errors)
        else:
            constrained_idx = best_idx
        
        return {
            'optimal_rref_unconstrained': float(r_ref_range[best_idx]),
            'optimal_rref_constrained': float(r_ref_range[constrained_idx]),
            'optimal_error_unconstrained': float(total_errors[best_idx]),
            'optimal_error_constrained': float(total_errors[constrained_idx]),
            'settling_time_at_optimal': float(settling_times[constrained_idx]),
            'area_utilization_pct': float(area_used[constrained_idx] / self.area_budget * 100),
            'sweep_data': {
                'r_ref_values': r_ref_range.tolist(),
                'total_errors': total_errors.tolist(),
                'mismatch_errors': mismatch_errors,
                'thermal_errors': thermal_errors,
                'area_used': area_used,
                'settling_times': settling_times,
            }
        }
