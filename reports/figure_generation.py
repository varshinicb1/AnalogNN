"""
Publication-Quality Figure Generation Engine
===========================================

Creates publication-ready vector figures (PDF/PNG) for academic reports.
Generates:
1. HMAC vs OLS calibration parity plots.
2. Calibration sample efficiency curves.
3. Parameter sensitivity sweeps (accuracy bound vs non-idealities).
4. Optimal resistance allocation tradeoffs (mismatch vs thermal noise).
5. 6-panel residual diagnostics.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple, Optional


class PublicationFigureEngine:
    """
    Plots high-resolution figures matching IEEE style guides.
    """

    def __init__(self, output_dir: str = "./figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure IEEE styling parameters
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14
        plt.rcParams['grid.alpha'] = 0.3
        
        # Professional color palette
        self.colors = {
            'primary': '#1f77b4',     # Professional blue
            'secondary': '#aec7e8',   # Light blue
            'accent': '#ff7f0e',      # Accent orange
            'success': '#2ca02c',     # Safe green
            'danger': '#d62728',      # Warning red
            'purple': '#9467bd',      # Dark purple
            'gray': '#7f7f7f'         # Neutral gray
        }

    def plot_calibration_parity(self, y_ideal: np.ndarray, y_pre: np.ndarray, 
                                y_post: np.ndarray, filename: str = "calibration_parity.png"):
        """
        Generates side-by-side scatter plots showing uncalibrated vs calibrated parity.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        
        # Flat data for scatter
        ideal_flat = y_ideal.flatten()
        pre_flat = y_pre.flatten()
        post_flat = y_post.flatten()
        
        # Determine limits
        min_val = min(ideal_flat.min(), pre_flat.min(), post_flat.min()) - 0.2
        max_val = max(ideal_flat.max(), pre_flat.max(), post_flat.max()) + 0.2
        
        # Panel 1: Pre-calibration
        axes[0].scatter(ideal_flat, pre_flat, color=self.colors['danger'], alpha=0.4, s=6, label='Raw Simulated')
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Parity')
        axes[0].set_xlabel('Ideal Target Activation (V)')
        axes[0].set_ylabel('Simulated Activation (V)')
        axes[0].set_title('Pre-Calibration Parity')
        axes[0].grid(True)
        axes[0].legend(loc='upper left')
        axes[0].set_xlim([min_val, max_val])
        axes[0].set_ylim([min_val, max_val])
        
        # Panel 2: Post-calibration
        axes[1].scatter(ideal_flat, post_flat, color=self.colors['success'], alpha=0.4, s=6, label='HMAC Calibrated')
        axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
        axes[1].set_xlabel('Ideal Target Activation (V)')
        axes[1].set_title('Post-Calibration Parity')
        axes[1].grid(True)
        axes[1].legend(loc='upper left')
        axes[1].set_xlim([min_val, max_val])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

    def plot_sample_efficiency(self, efficiency_data: Dict, filename: str = "sample_efficiency.png"):
        """
        Plots RMSE vs calibration training set size to compare data efficiency.
        """
        fig, ax = plt.subplots(figsize=(7, 4.5))
        
        sizes = efficiency_data['sizes']
        results = efficiency_data['results']
        
        markers = {'Affine': 'o', 'Quadratic': 's', 'Learned (MLP)': '^', 'HMAC (Linear)': 'D', 'HMAC (Quadratic)': '*'}
        line_colors = {
            'Affine': self.colors['gray'],
            'Quadratic': self.colors['primary'],
            'Learned (MLP)': self.colors['accent'],
            'HMAC (Linear)': self.colors['purple'],
            'HMAC (Quadratic)': self.colors['success']
        }
        
        for name, rmses in results.items():
            ax.plot(sizes, rmses, label=name, marker=markers.get(name, 'o'), 
                    color=line_colors.get(name), linewidth=1.8, markersize=6)
            
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Number of Calibration Training Samples (Log Scale)')
        ax.set_ylabel('Test Root Mean Squared Error (RMSE) (Log Scale)')
        ax.set_title('Calibration Sample Efficiency & Overfitting Benchmark')
        ax.grid(True, which='both', linestyle='--', alpha=0.4)
        ax.legend(frameon=True, facecolor='white', loc='upper right')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

    def plot_sensitivity_analysis(self, sensitivity_data: Dict, filename: str = "sensitivity_analysis.png"):
        """
        Plots accuracy degradation bounds or MSE bounds under parameter sweeps.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # 1. Mismatch sweep
        ax = axes[0, 0]
        data = sensitivity_data.get('mismatch_sweep', {})
        if data:
            ax.plot(data['values'], data['bounds'], color=self.colors['primary'], linewidth=2)
            ax.set_xlabel('Resistor Mismatch Standard Deviation ($\\sigma_R$)')
            ax.set_ylabel('Total Output MSE Bound (V$^2$)')
            ax.set_title('Sensitivity to Component Mismatch')
            ax.grid(True)
            
        # 2. Noise sweep
        ax = axes[0, 1]
        data = sensitivity_data.get('noise_sweep', {})
        if data:
            ax.plot(data['values'], data['bounds'], color=self.colors['accent'], linewidth=2)
            ax.set_xlabel('Temporal Weight Noise ($\\sigma_w$)')
            ax.set_ylabel('Total Output MSE Bound (V$^2$)')
            ax.set_title('Sensitivity to Temporal Noise')
            ax.grid(True)
            
        # 3. Offset sweep
        ax = axes[1, 0]
        data = sensitivity_data.get('offset_sweep', {})
        if data:
            ax.plot(data['values'], data['bounds'], color=self.colors['danger'], linewidth=2)
            ax.set_xlabel('Op-Amp Input Offset Voltage ($V_{os}$)')
            ax.set_ylabel('Total Output MSE Bound (V$^2$)')
            ax.set_title('Sensitivity to Op-Amp DC Offset')
            ax.grid(True)
            
        # 4. Quantization sweep
        ax = axes[1, 1]
        data = sensitivity_data.get('quantization_sweep', {})
        if data:
            ax.plot(data['values'], data['bounds'], color=self.colors['success'], linewidth=2)
            ax.set_xlabel('DAC/ADC Quantization Bits ($n_{bits}$)')
            ax.set_ylabel('Total Output MSE Bound (V$^2$)')
            ax.set_title('Sensitivity to Quantization')
            ax.grid(True)
            
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

    def plot_optimal_resistance_allocation(self, sweep_data: Dict, optimal_rref: float, 
                                           filename: str = "optimal_resistance.png"):
        """
        Plots the optimal resistance tradeoff showing mismatch vs thermal noise.
        """
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        
        r_refs = np.array(sweep_data['r_ref_values'])
        total_err = np.array(sweep_data['total_errors'])
        mismatch_err = np.array(sweep_data['mismatch_errors'])
        thermal_err = np.array(sweep_data['thermal_errors'])
        
        ax1.loglog(r_refs, mismatch_err, '--', color=self.colors['danger'], label='Mismatch Error ($\\propto 1/R_{ref}$)')
        ax1.loglog(r_refs, thermal_err, '--', color=self.colors['primary'], label='Thermal Noise Error ($\\propto R_{ref}$)')
        ax1.loglog(r_refs, total_err, '-', color=self.colors['purple'], linewidth=2.2, label='Total Expected MSE')
        
        ax1.axvline(x=optimal_rref, color='black', linestyle=':', label=f'Optimal $R_{{ref}}$ = {optimal_rref/1000:.1f} k$\\Omega$')
        
        ax1.set_xlabel('Reference Resistance $R_{ref}$ ($\\Omega$, Log Scale)')
        ax1.set_ylabel('Expected Output Voltage MSE (V$^2$, Log Scale)')
        ax1.set_title('Optimal Resistance Allocation & Physics Tradeoffs')
        ax1.grid(True, which='both', alpha=0.3)
        ax1.legend(loc='lower left')
        
        # Add a secondary axis showing worst-case settling time constraint
        ax2 = ax1.twinx()
        settling_times = np.array(sweep_data['settling_times']) * 1e6 # convert to microseconds
        ax2.loglog(r_refs, settling_times, ':', color=self.colors['gray'], label='Settling Time $\\tau$ ($\\mu$s)')
        ax2.set_ylabel('Worst-Case Settling Time $\\tau$ ($\\mu$s, Log Scale)', color=self.colors['gray'])
        ax2.tick_params(axis='y', labelcolor=self.colors['gray'])
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
