"""
Residual Analysis Engine
========================

Provides rigorous statistical analysis of calibration residuals to:
1. Diagnose systematic vs random error components
2. Detect heteroscedasticity patterns
3. Quantify per-channel error structure
4. Generate publication-quality residual diagnostic plots

This is essential for justifying the HMAC calibration approach —
showing that standard OLS residuals exhibit weight-dependent
heteroscedasticity, while HMAC residuals are properly homoscedastic.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from typing import Dict, Optional, Tuple
import os


class ResidualAnalyzer:
    """
    Comprehensive residual analysis for analog calibration validation.
    """
    
    def __init__(self, y_ideal: torch.Tensor, y_sim: torch.Tensor,
                 y_cal_hmac: Optional[torch.Tensor] = None,
                 y_cal_ols: Optional[torch.Tensor] = None,
                 weight_matrix: Optional[torch.Tensor] = None):
        """
        Parameters:
        - y_ideal: (N, C) ideal digital logits
        - y_sim: (N, C) raw analog simulation outputs
        - y_cal_hmac: (N, C) HMAC-calibrated outputs
        - y_cal_ols: (N, C) OLS-calibrated outputs (for comparison)
        - weight_matrix: (C, n) weight matrix for noise gain computation
        """
        self.ideal = y_ideal.detach().cpu().numpy()
        self.sim = y_sim.detach().cpu().numpy()
        self.cal_hmac = y_cal_hmac.detach().cpu().numpy() if y_cal_hmac is not None else None
        self.cal_ols = y_cal_ols.detach().cpu().numpy() if y_cal_ols is not None else None
        
        self.N, self.C = self.ideal.shape
        
        if weight_matrix is not None:
            w = weight_matrix.detach().cpu().numpy()
            self.noise_gains = 1.0 + np.sum(np.abs(w), axis=1)
        else:
            self.noise_gains = None
        
        # Compute residuals
        self.res_raw = self.sim - self.ideal
        self.res_hmac = (self.cal_hmac - self.ideal) if self.cal_hmac is not None else None
        self.res_ols = (self.cal_ols - self.ideal) if self.cal_ols is not None else None
    
    def compute_per_channel_statistics(self) -> Dict:
        """
        Per-channel residual statistics: mean, std, skewness, kurtosis.
        Tests for normality (Shapiro-Wilk) on each channel.
        """
        results = {}
        
        for name, residuals in [('raw', self.res_raw), 
                                 ('hmac', self.res_hmac), 
                                 ('ols', self.res_ols)]:
            if residuals is None:
                continue
            
            channel_stats = []
            for c in range(self.C):
                r = residuals[:, c]
                
                # Shapiro-Wilk normality test (max 5000 samples)
                n_test = min(len(r), 5000)
                sw_stat, sw_p = scipy_stats.shapiro(r[:n_test])
                
                channel_stats.append({
                    'channel': c,
                    'mean': float(np.mean(r)),
                    'std': float(np.std(r, ddof=1)),
                    'variance': float(np.var(r, ddof=1)),
                    'skewness': float(scipy_stats.skew(r)),
                    'kurtosis': float(scipy_stats.kurtosis(r)),
                    'min': float(np.min(r)),
                    'max': float(np.max(r)),
                    'shapiro_stat': float(sw_stat),
                    'shapiro_p': float(sw_p),
                    'is_normal': sw_p > 0.05,
                    'noise_gain': float(self.noise_gains[c]) if self.noise_gains is not None else None,
                })
            
            results[name] = channel_stats
        
        return results
    
    def heteroscedasticity_analysis(self) -> Dict:
        """
        Quantifies heteroscedasticity in residuals by:
        1. Computing per-channel variance
        2. Correlating variance with noise gain
        3. Computing the heteroscedasticity ratio
        4. Running Levene's test for equality of variances
        """
        results = {}
        
        for name, residuals in [('raw', self.res_raw),
                                 ('hmac', self.res_hmac),
                                 ('ols', self.res_ols)]:
            if residuals is None:
                continue
            
            channel_vars = np.var(residuals, axis=0, ddof=1)
            het_ratio = float(np.max(channel_vars) / (np.min(channel_vars) + 1e-15))
            
            # Correlation with noise gain
            if self.noise_gains is not None:
                corr, p_corr = scipy_stats.pearsonr(self.noise_gains, channel_vars)
                spearman_corr, p_spearman = scipy_stats.spearmanr(self.noise_gains, channel_vars)
            else:
                corr, p_corr = 0.0, 1.0
                spearman_corr, p_spearman = 0.0, 1.0
            
            # Levene's test (comparing channels with lowest vs highest variance)
            sorted_idx = np.argsort(channel_vars)
            low_var_channel = residuals[:, sorted_idx[0]]
            high_var_channel = residuals[:, sorted_idx[-1]]
            levene_stat, levene_p = scipy_stats.levene(low_var_channel, high_var_channel)
            
            # Bartlett's test (all channels)
            channel_residuals = [residuals[:, c] for c in range(self.C)]
            try:
                bartlett_stat, bartlett_p = scipy_stats.bartlett(*channel_residuals)
            except Exception:
                bartlett_stat, bartlett_p = 0.0, 1.0
            
            results[name] = {
                'channel_variances': channel_vars.tolist(),
                'heteroscedasticity_ratio': het_ratio,
                'variance_noise_gain_pearson_r': float(corr),
                'variance_noise_gain_pearson_p': float(p_corr),
                'variance_noise_gain_spearman_r': float(spearman_corr),
                'variance_noise_gain_spearman_p': float(p_spearman),
                'levene_statistic': float(levene_stat),
                'levene_p_value': float(levene_p),
                'levene_significant': levene_p < 0.05,
                'bartlett_statistic': float(bartlett_stat),
                'bartlett_p_value': float(bartlett_p),
                'bartlett_significant': bartlett_p < 0.05,
            }
        
        return results
    
    def plot_residual_diagnostics(self, save_dir: str = "./figures") -> list:
        """
        Generates a comprehensive 6-panel residual diagnostic figure.
        
        Panel layout:
        [1] Raw residuals histogram    [2] Q-Q plot of raw residuals
        [3] Residual vs noise gain     [4] Variance vs noise gain  
        [5] HMAC vs OLS residual std   [6] Residual autocorrelation
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_files = []
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Residual Diagnostic Analysis — OpenAnalogNN', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        colors = {'raw': '#e74c3c', 'ols': '#3498db', 'hmac': '#2ecc71'}
        
        # Panel 1: Residual histograms
        ax = axes[0, 0]
        for name, res, color in [('Raw', self.res_raw, colors['raw']),
                                   ('OLS', self.res_ols, colors['ols']),
                                   ('HMAC', self.res_hmac, colors['hmac'])]:
            if res is not None:
                ax.hist(res.flatten(), bins=60, alpha=0.5, color=color, 
                       label=f'{name} (σ={np.std(res):.4f})', density=True)
        ax.set_xlabel('Residual Value (V)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Residual Distributions', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, frameon=True, facecolor='white')
        
        # Panel 2: Q-Q plot
        ax = axes[0, 1]
        for name, res, color in [('Raw', self.res_raw, colors['raw']),
                                   ('HMAC', self.res_hmac, colors['hmac'])]:
            if res is not None:
                sorted_res = np.sort(res.flatten())
                n = len(sorted_res)
                theoretical = scipy_stats.norm.ppf(np.linspace(0.001, 0.999, n))
                ax.scatter(theoretical[::max(1, n//500)], sorted_res[::max(1, n//500)], 
                          s=3, alpha=0.5, color=color, label=name)
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax.set_ylabel('Sample Quantiles (V)', fontsize=11)
        ax.set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, frameon=True, facecolor='white')
        
        # Panel 3: Residual std per channel vs noise gain
        ax = axes[0, 2]
        if self.noise_gains is not None:
            for name, res, color, marker in [('Raw', self.res_raw, colors['raw'], 'o'),
                                               ('OLS', self.res_ols, colors['ols'], 's'),
                                               ('HMAC', self.res_hmac, colors['hmac'], '^')]:
                if res is not None:
                    channel_std = np.std(res, axis=0, ddof=1)
                    ax.scatter(self.noise_gains, channel_std, color=color, 
                             marker=marker, s=80, label=name, edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Noise Gain (1 + Σ|wᵢⱼ|)', fontsize=11)
            ax.set_ylabel('Per-Channel Residual Std (V)', fontsize=11)
            ax.set_title('Heteroscedasticity Structure', fontsize=13, fontweight='bold')
            ax.legend(fontsize=9, frameon=True, facecolor='white')
        else:
            ax.text(0.5, 0.5, 'No weight matrix\nprovided', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
        
        # Panel 4: Variance bar chart per channel
        ax = axes[1, 0]
        x_pos = np.arange(self.C)
        bar_width = 0.25
        for i, (name, res, color) in enumerate([('Raw', self.res_raw, colors['raw']),
                                                   ('OLS', self.res_ols, colors['ols']),
                                                   ('HMAC', self.res_hmac, colors['hmac'])]):
            if res is not None:
                channel_vars = np.var(res, axis=0, ddof=1)
                ax.bar(x_pos + i * bar_width, channel_vars, bar_width, 
                      color=color, label=name, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Output Channel', fontsize=11)
        ax.set_ylabel('Residual Variance (V²)', fontsize=11)
        ax.set_title('Per-Channel Variance Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos + bar_width)
        ax.set_xticklabels([str(i) for i in range(self.C)])
        ax.legend(fontsize=9, frameon=True, facecolor='white')
        
        # Panel 5: Cumulative error distribution
        ax = axes[1, 1]
        for name, res, color in [('Raw', self.res_raw, colors['raw']),
                                   ('OLS', self.res_ols, colors['ols']),
                                   ('HMAC', self.res_hmac, colors['hmac'])]:
            if res is not None:
                abs_errors = np.abs(res.flatten())
                sorted_errors = np.sort(abs_errors)
                cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
                ax.plot(sorted_errors, cdf, color=color, linewidth=2, label=name)
        ax.set_xlabel('Absolute Error (V)', fontsize=11)
        ax.set_ylabel('Cumulative Probability', fontsize=11)
        ax.set_title('Error CDF Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, frameon=True, facecolor='white')
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Per-channel RMSE improvement
        ax = axes[1, 2]
        if self.res_hmac is not None and self.res_ols is not None:
            rmse_ols = np.sqrt(np.mean(self.res_ols ** 2, axis=0))
            rmse_hmac = np.sqrt(np.mean(self.res_hmac ** 2, axis=0))
            improvement_pct = (rmse_ols - rmse_hmac) / (rmse_ols + 1e-12) * 100
            
            bar_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in improvement_pct]
            ax.bar(range(self.C), improvement_pct, color=bar_colors, 
                  edgecolor='black', linewidth=0.5)
            ax.axhline(y=0, color='black', linewidth=0.8)
            ax.set_xlabel('Output Channel', fontsize=11)
            ax.set_ylabel('HMAC Improvement over OLS (%)', fontsize=11)
            ax.set_title('Per-Channel HMAC Advantage', fontsize=13, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Need both HMAC\nand OLS results', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(save_dir, 'residual_diagnostics.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        saved_files.append(path)
        
        return saved_files
    
    def compute_summary_table(self) -> Dict:
        """
        Generates a summary table comparing Raw, OLS, and HMAC calibration.
        """
        table = {}
        for name, res in [('Raw (Uncalibrated)', self.res_raw),
                           ('OLS Calibrated', self.res_ols),
                           ('HMAC Calibrated', self.res_hmac)]:
            if res is None:
                continue
            table[name] = {
                'RMSE': float(np.sqrt(np.mean(res ** 2))),
                'MAE': float(np.mean(np.abs(res))),
                'Max Error': float(np.max(np.abs(res))),
                'Std Dev': float(np.std(res)),
                'Het Ratio': float(np.max(np.var(res, axis=0)) / (np.min(np.var(res, axis=0)) + 1e-15)),
            }
        return table
