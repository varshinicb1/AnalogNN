import torch
import numpy as np
from scipy.stats import pearsonr

def compute_metrics(y_ideal: torch.Tensor, y_sim: torch.Tensor, y_cal: torch.Tensor | None, y_true: torch.Tensor,
                    rest_of_network_fn = None) -> dict:
    """
    Computes scientific validation metrics between layers:
    - RMSE
    - Pearson Correlation (R)
    - Classification Accuracy Drop (using rest_of_network_fn to get class logits if provided)
    """
    ideal_np = y_ideal.detach().cpu().numpy()
    sim_np = y_sim.detach().cpu().numpy()
    true_np = y_true.detach().cpu().numpy()
    
    # Compute class logits if rest_of_network_fn is provided
    if rest_of_network_fn is not None:
        with torch.no_grad():
            ideal_logits = rest_of_network_fn(y_ideal).detach().cpu().numpy()
            sim_logits = rest_of_network_fn(y_sim).detach().cpu().numpy()
        ideal_preds = np.argmax(ideal_logits, axis=1)
        sim_preds = np.argmax(sim_logits, axis=1)
    else:
        ideal_preds = np.argmax(ideal_np, axis=1)
        sim_preds = np.argmax(sim_np, axis=1)
        
    acc_ideal = np.mean(ideal_preds == true_np)
    acc_sim = np.mean(sim_preds == true_np)
    
    # RMSE pre-calibration
    rmse_pre = np.sqrt(np.mean((ideal_np - sim_np) ** 2))
    
    # Correlation pre-calibration
    r_pre_list = []
    for i in range(ideal_np.shape[1]):
        if np.std(sim_np[:, i]) > 1e-8 and np.std(ideal_np[:, i]) > 1e-8:
            r, _ = pearsonr(sim_np[:, i], ideal_np[:, i])
            r_pre_list.append(r)
    r_pre = np.mean(r_pre_list) if r_pre_list else 0.0
    
    metrics = {
        'accuracy_ideal': acc_ideal,
        'accuracy_sim': acc_sim,
        'accuracy_drop_pre': acc_ideal - acc_sim,
        'rmse_pre_calibration': rmse_pre,
        'correlation_pre_calibration': r_pre
    }
    
    if y_cal is not None:
        cal_np = y_cal.detach().cpu().numpy()
        if rest_of_network_fn is not None:
            with torch.no_grad():
                cal_logits = rest_of_network_fn(y_cal).detach().cpu().numpy()
            cal_preds = np.argmax(cal_logits, axis=1)
        else:
            cal_preds = np.argmax(cal_np, axis=1)
        acc_cal = np.mean(cal_preds == true_np)
        
        # RMSE post-calibration
        rmse_post = np.sqrt(np.mean((ideal_np - cal_np) ** 2))
        
        # Correlation post-calibration
        r_post_list = []
        for i in range(ideal_np.shape[1]):
            if np.std(cal_np[:, i]) > 1e-8 and np.std(ideal_np[:, i]) > 1e-8:
                r, _ = pearsonr(cal_np[:, i], ideal_np[:, i])
                r_post_list.append(r)
        r_post = np.mean(r_post_list) if r_post_list else 0.0
        
        metrics.update({
            'accuracy_calibrated': acc_cal,
            'accuracy_drop_post': acc_ideal - acc_cal,
            'calibration_improvement_pct': ((rmse_pre - rmse_post) / (rmse_pre + 1e-12)) * 100.0,
            'rmse_post_calibration': rmse_post,
            'correlation_post_calibration': r_post
        })
        
    return metrics
