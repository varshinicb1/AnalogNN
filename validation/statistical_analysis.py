import numpy as np
import scipy.stats as stats
import os

class StatisticalAnalysis:
    @staticmethod
    def aggregate_runs(runs_data: list[dict]) -> dict:
        """
        Aggregates multiple randomized experimental runs to compute:
        - Mean
        - Standard Deviation (std)
        - Standard Error of the Mean (sem)
        - 95% Confidence Intervals (CI)
        """
        if not runs_data:
            return {}
            
        keys = runs_data[0].keys()
        stats_summary = {}
        
        for key in keys:
            # Extract values for this metric across runs
            vals = []
            for run in runs_data:
                if key not in run:
                    continue
                val = run[key]
                if hasattr(val, 'item'): # Torch tensor or NumPy scalar
                    val = val.item()
                if isinstance(val, (int, float, np.integer, np.floating)):
                    vals.append(float(val))
                    
            if not vals:
                continue
                
            vals = np.array(vals)
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            sem = std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            
            # 95% Confidence Interval
            if len(vals) > 1 and sem > 1e-12:
                ci = stats.t.interval(0.95, df=len(vals)-1, loc=mean, scale=sem)
                ci_margin = ci[1] - mean
                if np.isnan(ci_margin):
                    ci_margin = 0.0
            else:
                ci_margin = 0.0
                
            stats_summary[key] = {
                'mean': mean,
                'std': std,
                'sem': sem,
                'ci_margin': ci_margin,
                'raw': vals.tolist()
            }
            
        return stats_summary

    @staticmethod
    def generate_latex_table(stats_summary: dict, output_filepath: str):
        """
        Generates a publication-grade LaTeX formatted tabular block from the statistical summaries.
        """
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{OpenAnalogNN Cross-Layer Validation and Calibration Performance Summary}")
        latex.append(r"\label{tab:analog_stats}")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\hline")
        latex.append(r"\textbf{Performance Metric} & \textbf{Digital Baseline} & \textbf{Uncalibrated Analog} & \textbf{Calibrated Analog} \\")
        latex.append(r"\hline")
        
        # Accuracy row
        acc_ideal = stats_summary.get('accuracy_ideal', {}).get('mean', 0.0) * 100.0
        acc_ideal_ci = stats_summary.get('accuracy_ideal', {}).get('ci_margin', 0.0) * 100.0
        
        acc_sim = stats_summary.get('accuracy_sim', {}).get('mean', 0.0) * 100.0
        acc_sim_ci = stats_summary.get('accuracy_sim', {}).get('ci_margin', 0.0) * 100.0
        
        acc_cal = stats_summary.get('accuracy_calibrated', {}).get('mean', 0.0) * 100.0
        acc_cal_ci = stats_summary.get('accuracy_calibrated', {}).get('ci_margin', 0.0) * 100.0
        
        latex.append(f"Classification Accuracy (\\%) & {acc_ideal:.2f} $\\pm$ {acc_ideal_ci:.2f} & {acc_sim:.2f} $\\pm$ {acc_sim_ci:.2f} & {acc_cal:.2f} $\\pm$ {acc_cal_ci:.2f} \\\\")
        
        # RMSE rows
        rmse_pre = stats_summary.get('rmse_pre_calibration', {}).get('mean', 0.0)
        rmse_pre_ci = stats_summary.get('rmse_pre_calibration', {}).get('ci_margin', 0.0)
        
        rmse_post = stats_summary.get('rmse_post_calibration', {}).get('mean', 0.0)
        rmse_post_ci = stats_summary.get('rmse_post_calibration', {}).get('ci_margin', 0.0)
        
        latex.append(f"Root Mean Squared Error (RMSE) & --- & {rmse_pre:.4f} $\\pm$ {rmse_pre_ci:.4f} & {rmse_post:.4f} $\\pm$ {rmse_post_ci:.4f} \\\\")
        
        # Correlation rows
        r_pre = stats_summary.get('correlation_pre_calibration', {}).get('mean', 0.0)
        r_pre_ci = stats_summary.get('correlation_pre_calibration', {}).get('ci_margin', 0.0)
        
        r_post = stats_summary.get('correlation_post_calibration', {}).get('mean', 0.0)
        r_post_ci = stats_summary.get('correlation_post_calibration', {}).get('ci_margin', 0.0)
        
        latex.append(f"Pearson Correlation ($R$) & 1.000 & {r_pre:.4f} $\\pm$ {r_pre_ci:.4f} & {r_post:.4f} $\\pm$ {r_post_ci:.4f} \\\\")
        
        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        with open(output_filepath, "w") as f:
            f.write("\n".join(latex))
            
        return "\n".join(latex)
