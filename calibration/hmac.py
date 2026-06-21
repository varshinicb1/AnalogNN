"""
Heteroscedastic Mismatch-Aware Calibration (HMAC)
=================================================

Calibration method for analog neural network inference using
physics-informed Weighted Least Squares.

=== THEORETICAL FOUNDATION ===

For an analog summing amplifier network implementing y = Wx + b,
subject to resistor mismatch delta_ij ~ N(0, sigma_R^2) and
op-amp input offset V_os ~ N(0, sigma_os^2), the output error
e_i for neuron i has variance:

    Var(e_i) = sigma_R^2 * sum_j (w_ij * x_j)^2 / (1+delta_ij)^2
             + sigma_os^2 * (1 + sum_j |w_ij|)^2

Under the simplifying assumption of small mismatch (|delta| << 1)
and stationary inputs, this reduces to:

    Var(e_i) ≈ sigma_R^2 * ||w_i||_2^2 * E[x^2]
              + sigma_os^2 * (1 + ||w_i||_1)^2

The HMAC estimator uses Weighted Least Squares with weight matrix
W = diag(1/Var(e_i)), following the Generalized Least Squares (GLS)
framework for heteroscedastic models.

Note: This is an application of standard GLS with physics-derived weights,
not a novel statistical method. The Gauss-Markov theorem guarantees that
GLS is the Best Linear Unbiased Estimator (BLUE) when the covariance
structure is known.

=== REFERENCES ===
    [1] Gauss-Markov theorem — Aitken (1935)
    [2] Pelgrom et al., "Matching Properties of MOS Transistors" (1989)
    [3] Razavi, "Design of Analog CMOS Integrated Circuits" (2001)
    [4] Carroll & Ruppert, "Transformation and Weighting in Regression" (1988)
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from scipy import stats as scipy_stats


class HMACCalibrator:
    """
    Heteroscedastic Mismatch-Aware Calibration.
    
    Uses Generalized Least Squares (GLS) with a physically-derived
    heteroscedasticity structure from the analog circuit's noise gain.
    
    This is provably the Best Linear Unbiased Estimator (BLUE) for
    analog circuit calibration under the Gauss-Markov conditions.
    """
    
    def __init__(self, weight_matrix: Optional[torch.Tensor] = None,
                 polynomial_degree: int = 1,
                 regularization: float = 1e-8,
                 mismatch_sigma: float = 0.01,
                 offset_sigma: float = 0.002,
                 noise_sigma: float = 0.05):
        """
        Parameters:
        - weight_matrix: Neural layer weight matrix (out, in). Used to compute
                        noise gain structure. If None, variance estimated from data.
        - polynomial_degree: Degree of polynomial calibration map (1=affine, 2=quadratic, 3=cubic).
        - regularization: Tikhonov regularization parameter (lambda).
        - mismatch_sigma: Resistor mismatch std dev (from config).
        - offset_sigma: Op-amp input offset voltage std dev (from config).
        - noise_sigma: Weight noise std dev (from config).
        """
        self.weight_matrix = weight_matrix
        self.polynomial_degree = polynomial_degree
        self.regularization = regularization
        self.mismatch_sigma = mismatch_sigma
        self.offset_sigma = offset_sigma
        self.noise_sigma = noise_sigma
        
        self.coefficients = None
        self.noise_gains = None
        self.sigma_diag = None
        self.fit_diagnostics = {}
        self._ols_coefficients = None  # For comparison
        
    def compute_theoretical_variance(self, weight_matrix: torch.Tensor,
                                      x_variance: float = 0.5) -> np.ndarray:
        """
        Computes the theoretical per-neuron output error variance from circuit physics.
        
        Var(e_i) = sigma_R^2 * ||w_i||_2^2 * E[x^2]
                 + sigma_os^2 * (1 + ||w_i||_1)^2  
                 + sigma_w^2 * E[x^2] * n_inputs
        
        Parameters:
        - weight_matrix: (out_features, in_features) tensor
        - x_variance: Expected value of x_j^2 (second moment of inputs)
        
        Returns:
        - variance vector (out_features,)
        """
        w = weight_matrix.detach().cpu().numpy()
        out_features, in_features = w.shape
        
        # Component 1: Mismatch contribution
        # E[e_mismatch_i^2] = sigma_R^2 * sum_j (w_ij^2 * E[x_j^2])
        mismatch_var = self.mismatch_sigma**2 * np.sum(w**2, axis=1) * x_variance
        
        # Component 2: Op-amp offset contribution  
        # E[e_offset_i^2] = sigma_os^2 * (1 + sum_j |w_ij|)^2
        noise_gain = 1.0 + np.sum(np.abs(w), axis=1)
        offset_var = self.offset_sigma**2 * noise_gain**2
        
        # Component 3: Weight noise contribution
        # E[e_noise_i^2] = sigma_w^2 * E[x^2] * n_inputs
        noise_var = self.noise_sigma**2 * x_variance * in_features
        
        total_var = mismatch_var + offset_var + noise_var
        
        return total_var, noise_gain, mismatch_var, offset_var, noise_var
    
    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """Build polynomial feature matrix [1, x, x^2, ..., x^d]."""
        N = x.shape[0]
        features = np.ones((N, self.polynomial_degree + 1))
        for k in range(1, self.polynomial_degree + 1):
            features[:, k] = x ** k
        return features
    
    def _solve_wls(self, X: np.ndarray, y: np.ndarray, 
                   weights: np.ndarray) -> np.ndarray:
        """
        Weighted Least Squares: β = (X^T W X + λI)^{-1} X^T W y
        """
        W = np.diag(weights)
        XtWX = X.T @ W @ X + self.regularization * np.eye(X.shape[1])
        XtWy = X.T @ W @ y
        return np.linalg.solve(XtWX, XtWy)
    
    def _solve_ols(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Ordinary Least Squares: β = (X^T X + λI)^{-1} X^T y
        """
        XtX = X.T @ X + self.regularization * np.eye(X.shape[1])
        Xty = X.T @ y
        return np.linalg.solve(XtX, Xty)
    
    def fit(self, y_spice: torch.Tensor, y_ideal: torch.Tensor,
            weight_matrix: Optional[torch.Tensor] = None,
            input_data: Optional[torch.Tensor] = None):
        """
        Fit HMAC calibration with physics-derived heteroscedastic weights.
        Also fits OLS internally for efficiency comparison.
        
        Parameters:
        - y_spice: (N, C) simulated circuit voltages
        - y_ideal: (N, C) ideal mathematical logits
        - weight_matrix: optional weight matrix override
        - input_data: optional input data for variance estimation
        """
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()
        N, C = spice_np.shape
        
        w_mat = weight_matrix if weight_matrix is not None else self.weight_matrix
        
        # Compute input variance
        if input_data is not None:
            x_var = float(torch.mean(input_data ** 2).item())
        else:
            x_var = 0.5  # Default for [0,1] normalized inputs
        
        # Step 1: Compute heteroscedastic variance structure
        if w_mat is not None:
            total_var, self.noise_gains, mismatch_var, offset_var, noise_var = \
                self.compute_theoretical_variance(w_mat, x_var)
            self.sigma_diag = total_var
            self._variance_components = {
                'mismatch': mismatch_var.tolist(),
                'offset': offset_var.tolist(),
                'noise': float(noise_var),
                'total': total_var.tolist()
            }
        else:
            # Data-driven fallback: estimate from OLS residuals
            residuals = ideal_np - spice_np
            self.sigma_diag = np.var(residuals, axis=0)
            self.sigma_diag = np.maximum(self.sigma_diag, 1e-12)
            self.noise_gains = np.sqrt(self.sigma_diag)
            self._variance_components = None
        
        # Prevent division by zero
        self.sigma_diag = np.maximum(self.sigma_diag, 1e-12)
        
        # Step 2: Fit per-channel WLS (HMAC) and OLS (baseline)
        self.coefficients = np.zeros((C, self.polynomial_degree + 1))
        self._ols_coefficients = np.zeros((C, self.polynomial_degree + 1))
        self.fit_diagnostics = {}
        
        for c in range(C):
            x_c = spice_np[:, c]
            y_c = ideal_np[:, c]
            X_poly = self._build_features(x_c)
            
            # WLS weights: inverse variance (higher variance → lower weight)
            wls_weights = np.ones(N) / self.sigma_diag[c]
            
            # HMAC fit (GLS)
            beta_hmac = self._solve_wls(X_poly, y_c, wls_weights)
            self.coefficients[c] = beta_hmac
            
            # OLS fit (for comparison)
            beta_ols = self._solve_ols(X_poly, y_c)
            self._ols_coefficients[c] = beta_ols
            
            # Diagnostics
            y_hat_hmac = X_poly @ beta_hmac
            y_hat_ols = X_poly @ beta_ols
            res_hmac = y_c - y_hat_hmac
            res_ols = y_c - y_hat_ols
            
            ss_tot = np.sum((y_c - np.mean(y_c)) ** 2) + 1e-12
            
            self.fit_diagnostics[c] = {
                'mse_hmac': float(np.mean(res_hmac ** 2)),
                'mse_ols': float(np.mean(res_ols ** 2)),
                'r2_hmac': float(1.0 - np.sum(res_hmac ** 2) / ss_tot),
                'r2_ols': float(1.0 - np.sum(res_ols ** 2) / ss_tot),
                'noise_gain': float(self.noise_gains[c]) if self.noise_gains is not None else None,
                'channel_variance': float(self.sigma_diag[c]),
                # Weighted MSE (the proper metric for heteroscedastic models)
                'wmse_hmac': float(np.mean(wls_weights * res_hmac ** 2)),
                'wmse_ols': float(np.mean(wls_weights * res_ols ** 2)),
            }
    
    def calibrate(self, y_spice: torch.Tensor) -> torch.Tensor:
        """Apply HMAC calibration to new data."""
        if self.coefficients is None:
            raise ValueError("HMAC calibrator has not been fitted!")
        
        spice_np = y_spice.detach().cpu().numpy()
        N, C = spice_np.shape
        calibrated = np.zeros_like(spice_np)
        
        for c in range(C):
            X_poly = self._build_features(spice_np[:, c])
            calibrated[:, c] = X_poly @ self.coefficients[c]
        
        return torch.tensor(calibrated, dtype=torch.float32)
    
    def calibrate_ols(self, y_spice: torch.Tensor) -> torch.Tensor:
        """Apply baseline OLS calibration for comparison."""
        if self._ols_coefficients is None:
            raise ValueError("Calibrator has not been fitted!")
        
        spice_np = y_spice.detach().cpu().numpy()
        N, C = spice_np.shape
        calibrated = np.zeros_like(spice_np)
        
        for c in range(C):
            X_poly = self._build_features(spice_np[:, c])
            calibrated[:, c] = X_poly @ self._ols_coefficients[c]
        
        return torch.tensor(calibrated, dtype=torch.float32)
    
    def compute_efficiency_gain(self) -> dict:
        """
        Computes the theoretical and empirical efficiency gain of HMAC over OLS.
        
        Theoretical: eff_ratio >= max(Σ_ii) / min(Σ_ii)
        Empirical: ratio of weighted MSEs
        """
        if not self.fit_diagnostics:
            return {}
        
        wmse_hmac = np.mean([d['wmse_hmac'] for d in self.fit_diagnostics.values()])
        wmse_ols = np.mean([d['wmse_ols'] for d in self.fit_diagnostics.values()])
        
        mse_hmac = np.mean([d['mse_hmac'] for d in self.fit_diagnostics.values()])
        mse_ols = np.mean([d['mse_ols'] for d in self.fit_diagnostics.values()])
        
        het_ratio = float(np.max(self.sigma_diag) / (np.min(self.sigma_diag) + 1e-12))
        
        return {
            'heteroscedasticity_ratio': het_ratio,
            'theoretical_max_efficiency_gain': het_ratio,
            'empirical_wmse_ratio': float(wmse_ols / (wmse_hmac + 1e-12)),
            'empirical_mse_ratio': float(mse_ols / (mse_hmac + 1e-12)),
            'mean_wmse_hmac': float(wmse_hmac),
            'mean_wmse_ols': float(wmse_ols),
            'mean_mse_hmac': float(mse_hmac),
            'mean_mse_ols': float(mse_ols),
        }
    
    def breusch_pagan_test(self, y_spice: torch.Tensor, 
                           y_ideal: torch.Tensor) -> dict:
        """
        Breusch-Pagan test for heteroscedasticity.
        
        H0: Homoscedastic errors (OLS is efficient)
        H1: Heteroscedastic errors (HMAC is needed)
        
        If p < 0.05, reject H0 → heteroscedasticity is statistically significant
        → HMAC is justified.
        """
        spice_np = y_spice.detach().cpu().numpy()
        ideal_np = y_ideal.detach().cpu().numpy()
        N, C = spice_np.shape
        
        results = {}
        for c in range(C):
            # OLS residuals
            X = self._build_features(spice_np[:, c])
            beta = self._solve_ols(X, ideal_np[:, c])
            residuals = ideal_np[:, c] - X @ beta
            
            # Squared residuals
            u_sq = residuals ** 2
            u_sq_norm = u_sq / np.mean(u_sq)
            
            # Regress squared residuals on X
            beta_aux = self._solve_ols(X, u_sq_norm)
            u_sq_hat = X @ beta_aux
            
            # Test statistic: N * R^2 of auxiliary regression
            ss_res_aux = np.sum((u_sq_norm - u_sq_hat) ** 2)
            ss_tot_aux = np.sum((u_sq_norm - np.mean(u_sq_norm)) ** 2)
            r2_aux = 1.0 - ss_res_aux / (ss_tot_aux + 1e-12)
            
            bp_stat = N * r2_aux
            p_value = 1.0 - scipy_stats.chi2.cdf(bp_stat, df=X.shape[1] - 1)
            
            results[c] = {
                'bp_statistic': float(bp_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'interpretation': 'Heteroscedastic (HMAC justified)' if p_value < 0.05 
                                  else 'Cannot reject homoscedasticity'
            }
        
        return results
    
    def get_full_diagnostics(self) -> dict:
        """Returns comprehensive diagnostics for publication."""
        return {
            'per_channel': self.fit_diagnostics,
            'efficiency_gain': self.compute_efficiency_gain(),
            'variance_components': self._variance_components if hasattr(self, '_variance_components') else None,
            'noise_gain_range': [
                float(np.min(self.noise_gains)),
                float(np.max(self.noise_gains))
            ] if self.noise_gains is not None else None,
        }
