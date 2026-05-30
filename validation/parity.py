import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_parity(y_ideal: torch.Tensor, y_sim: torch.Tensor, y_cal: torch.Tensor | None,
                metrics: dict, save_dir: str = "./figures") -> str:
    """
    Generates a high-quality parity plot comparing:
    - Ideal vs Simulated
    - Ideal vs Calibrated
    """
    os.makedirs(save_dir, exist_ok=True)
    
    ideal_np = y_ideal.detach().cpu().numpy().flatten()
    sim_np = y_sim.detach().cpu().numpy().flatten()
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    if y_cal is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(7, 5))
        ax2 = None
        
    # Minimum/Maximum boundary for the line y = x
    min_val = min(ideal_np.min(), sim_np.min())
    max_val = max(ideal_np.max(), sim_np.max())
    if y_cal is not None:
        cal_np = y_cal.detach().cpu().numpy().flatten()
        min_val = min(min_val, cal_np.min())
        max_val = max(max_val, cal_np.max())
        
    diag_x = np.linspace(min_val, max_val, 100)
    
    # 1. Pre-Calibration Plot
    ax1.scatter(ideal_np, sim_np, alpha=0.5, color='#e377c2', edgecolors='none', label='Data Points')
    ax1.plot(diag_x, diag_x, color='#7f7f7f', linestyle='--', linewidth=1.5, label='Ideal y = x')
    ax1.set_title('Pre-Calibration Parity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Ideal Logits (V)', fontsize=11)
    ax1.set_ylabel('Simulated Nodal Voltages (V)', fontsize=11)
    
    pre_text = (
        f"RMSE: {metrics.get('rmse_pre_calibration', 0.0):.4f} V\n"
        f"Corr R: {metrics.get('correlation_pre_calibration', 0.0):.4f}\n"
        f"Sim Acc: {metrics.get('accuracy_sim', 0.0)*100:.2f}%"
    )
    ax1.text(0.05, 0.95, pre_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))
    ax1.legend(loc='lower right', frameon=True, facecolor='white')
    
    # 2. Post-Calibration Plot
    if ax2 is not None and y_cal is not None:
        ax2.scatter(ideal_np, cal_np, alpha=0.5, color='#17becf', edgecolors='none', label='Data Points')
        ax2.plot(diag_x, diag_x, color='#7f7f7f', linestyle='--', linewidth=1.5, label='Ideal y = x')
        ax2.set_title('Post-Calibration Parity', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Ideal Logits (V)', fontsize=11)
        ax2.set_ylabel('Calibrated Logits (V)', fontsize=11)
        
        post_text = (
            f"RMSE: {metrics.get('rmse_post_calibration', 0.0):.4f} V\n"
            f"Corr R: {metrics.get('correlation_post_calibration', 0.0):.4f}\n"
            f"Cal Acc: {metrics.get('accuracy_calibrated', 0.0)*100:.2f}%\n"
            f"Err Reduction: {metrics.get('calibration_improvement_pct', 0.0):.1f}%"
        )
        ax2.text(0.05, 0.95, post_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))
        ax2.legend(loc='lower right', frameon=True, facecolor='white')
        
    plt.suptitle('Cross-Layer Calibration & Parity Analysis', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'parity_analysis.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path
