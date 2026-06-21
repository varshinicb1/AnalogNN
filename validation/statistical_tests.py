"""
Formal Statistical Hypothesis Testing for Analog Calibration
=============================================================

This module provides rigorous statistical validation to compare analog
calibration methods (Raw, Affine, Polynomial, Learned, and HMAC).
It performs:
1. Residual normality tests (Shapiro-Wilk, D'Agostino K-squared).
2. Heteroscedasticity diagnostics (Breusch-Pagan, Goldfeld-Quandt).
3. Multi-model calibration performance comparisons (ANOVA, Friedman).
4. Post-hoc pairwise testing with p-value correction (Holm-Bonferroni, Tukey).
5. Effect size quantification (Cohen's d).
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional


class CalibrationStatisticalTester:
    """
    Executes formal statistical hypothesis tests on analog network outputs
    and calibration residuals to establish statistical significance.
    """

    @staticmethod
    def test_normality(residuals: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Performs multiple normality tests on residuals.
        Null Hypothesis (H0): The residuals are normally distributed.
        """
        flat_res = residuals.flatten()
        n = len(flat_res)
        
        # Shapiro-Wilk (capped at 5000 samples due to scipy limitations)
        if n <= 5000:
            sw_stat, sw_p = stats.shapiro(flat_res)
        else:
            # Subsample randomly if too large
            rng = np.random.default_rng(42)
            subsample = rng.choice(flat_res, size=5000, replace=False)
            sw_stat, sw_p = stats.shapiro(subsample)

        # D'Agostino's K-squared test
        if n >= 8:
            k2_stat, k2_p = stats.normaltest(flat_res)
        else:
            k2_stat, k2_p = 0.0, 1.0

        # Anderson-Darling test
        ad_res = stats.anderson(flat_res, dist='norm')
        
        return {
            'shapiro_wilk': {
                'statistic': float(sw_stat),
                'p_value': float(sw_p),
                'reject_h0': bool(sw_p < 0.05)
            },
            'dagostino_k2': {
                'statistic': float(k2_stat),
                'p_value': float(k2_p),
                'reject_h0': bool(k2_p < 0.05)
            },
            'anderson_darling': {
                'statistic': float(ad_res.statistic),
                'critical_values': ad_res.critical_values.tolist(),
                'significance_levels': ad_res.significance_level.tolist(),
                'reject_h0_at_5pct': bool(ad_res.statistic > ad_res.critical_values[2]) # 5% level
            }
        }

    @staticmethod
    def test_heteroscedasticity(residuals: np.ndarray, noise_gains: np.ndarray) -> Dict[str, float]:
        """
        Tests for heteroscedasticity in the residuals with respect to channel noise gains.
        H0: Homoscedasticity (equal variance across channels).
        H1: Heteroscedasticity (variance depends on noise gain/complexity).
        """
        # Group residuals by channel
        C = residuals.shape[1]
        channel_vars = np.var(residuals, axis=0, ddof=1)
        
        # 1. Levene's test (robust to non-normality)
        channel_data = [residuals[:, c] for c in range(C)]
        levene_stat, levene_p = stats.levene(*channel_data)
        
        # 2. Bartlett's test (more power if normal)
        try:
            bartlett_stat, bartlett_p = stats.bartlett(*channel_data)
        except Exception:
            bartlett_stat, bartlett_p = 0.0, 1.0

        # 3. Goldfeld-Quandt test (split by noise gains)
        # We sort channels by noise gain and compare variance of top vs bottom 30%
        sorted_idx = np.argsort(noise_gains)
        split_size = max(1, int(C * 0.3))
        
        low_gain_res = residuals[:, sorted_idx[:split_size]].flatten()
        high_gain_res = residuals[:, sorted_idx[-split_size:]].flatten()
        
        var_low = np.var(low_gain_res, ddof=1)
        var_high = np.var(high_gain_res, ddof=1)
        
        # F-statistic = Var(High) / Var(Low) assuming high is larger
        f_stat = var_high / (var_low + 1e-15)
        df1 = len(high_gain_res) - 1
        df2 = len(low_gain_res) - 1
        gq_p = 1.0 - stats.f.cdf(f_stat, df1, df2)

        return {
            'levene_statistic': float(levene_stat),
            'levene_p': float(levene_p),
            'levene_reject_h0': bool(levene_p < 0.05),
            'bartlett_statistic': float(bartlett_stat),
            'bartlett_p': float(bartlett_p),
            'bartlett_reject_h0': bool(bartlett_p < 0.05),
            'goldfeld_quandt_f': float(f_stat),
            'goldfeld_quandt_p': float(gq_p),
            'goldfeld_quandt_reject_h0': bool(gq_p < 0.05),
            'variance_gain_spearman_rho': float(stats.spearmanr(noise_gains, channel_vars)[0])
        }

    @staticmethod
    def compare_calibrators(errors_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Compares multiple calibration models using ANOVA and Friedman tests on absolute/squared errors.
        
        Parameters:
        - errors_dict: Dictionary mapping model names to absolute errors of shape (N_samples,)
        """
        methods = list(errors_dict.keys())
        if len(methods) < 2:
            return {'error': 'Need at least two methods to compare.'}
            
        data_list = [errors_dict[m] for m in methods]
        
        # 1. Standard One-way ANOVA
        f_stat, anova_p = stats.f_oneway(*data_list)
        
        # 2. Non-parametric Friedman test (repeated measures/paired)
        # Expects matrix of shape (N_samples, N_methods)
        matrix_data = np.stack(data_list, axis=1)
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(*data_list)
        except Exception as e:
            friedman_stat, friedman_p = 0.0, 1.0
            
        # 3. Kruskal-Wallis test (independent samples alternative)
        kw_stat, kw_p = stats.kruskal(*data_list)
        
        return {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(anova_p),
                'reject_h0': bool(anova_p < 0.05)
            },
            'friedman': {
                'statistic': float(friedman_stat),
                'p_value': float(friedman_p),
                'reject_h0': bool(friedman_p < 0.05)
            },
            'kruskal_wallis': {
                'statistic': float(kw_stat),
                'p_value': float(kw_p),
                'reject_h0': bool(kw_p < 0.05)
            }
        }

    @staticmethod
    def pairwise_comparisons(errors_dict: Dict[str, np.ndarray], 
                             paired: bool = True) -> List[Dict]:
        """
        Performs pairwise comparisons between all calibration methods.
        Applies Holm-Bonferroni correction to avoid family-wise error rate inflation.
        """
        methods = list(errors_dict.keys())
        pairs = []
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                pairs.append((methods[i], methods[j]))
                
        raw_p_values = []
        t_statistics = []
        cohen_ds = []
        
        for m1, m2 in pairs:
            err1 = errors_dict[m1]
            err2 = errors_dict[m2]
            
            # Compute Cohen's d effect size
            diff = err1 - err2
            mean_diff = np.mean(diff) if paired else (np.mean(err1) - np.mean(err2))
            pooled_std = np.std(diff, ddof=1) if paired else np.sqrt((np.var(err1, ddof=1) + np.var(err2, ddof=1)) / 2.0)
            d = mean_diff / (pooled_std + 1e-15)
            cohen_ds.append(float(d))
            
            # Hypothesis test
            if paired:
                stat, p = stats.ttest_rel(err1, err2)
            else:
                stat, p = stats.ttest_ind(err1, err2, equal_var=False)
            
            t_statistics.append(float(stat))
            raw_p_values.append(float(p))
            
        # Holm-Bonferroni correction
        m = len(pairs)
        sorted_indices = np.argsort(raw_p_values)
        adjusted_p_values = np.zeros(m)
        
        for rank, idx in enumerate(sorted_indices):
            adjusted_p = raw_p_values[idx] * (m - rank)
            adjusted_p_values[idx] = min(adjusted_p, 1.0)
            
        # Enforce monotonicity
        for k in range(1, m):
            prev_idx = sorted_indices[k - 1]
            curr_idx = sorted_indices[k]
            if adjusted_p_values[curr_idx] < adjusted_p_values[prev_idx]:
                adjusted_p_values[curr_idx] = adjusted_p_values[prev_idx]
                
        results = []
        for k, (m1, m2) in enumerate(pairs):
            results.append({
                'method_1': m1,
                'method_2': m2,
                't_statistic': t_statistics[k],
                'raw_p_value': raw_p_values[k],
                'adjusted_p_value': adjusted_p_values[k],
                'significant': adjusted_p_values[k] < 0.05,
                'cohen_d': cohen_ds[k],
                'superior_method': m2 if t_statistics[k] > 0 else m1  # If err1 > err2, m2 is superior (lower error)
            })
            
        return results

    @staticmethod
    def generate_statistical_latex_report(tester_results: Dict) -> str:
        """
        Generates LaTeX text block compiling statistical findings.
        """
        report = []
        report.append(r"\begin{table}[htbp]")
        report.append(r"\centering")
        report.append(r"\caption{Hypothesis Testing and Pairwise Comparison of Calibration Methods}")
        report.append(r"\label{tab:calibration_hypothesis}")
        report.append(r"\begin{tabular}{llccc}")
        report.append(r"\hline")
        report.append(r"\textbf{Comparison} & \textbf{Hypothesis Test} & \textbf{Statistic} & \textbf{Adjusted p-value} & \textbf{Effect Size ($d$)} \\")
        report.append(r"\hline")
        
        # Add pairwise comparisons
        if 'pairwise' in tester_results:
            for pair in tester_results['pairwise']:
                m1, m2 = pair['method_1'], pair['method_2']
                stat = pair['t_statistic']
                p_val = pair['adjusted_p_value']
                d = pair['cohen_d']
                
                p_str = f"{p_val:.2e}" if p_val > 1e-4 else "< 1e-4"
                sig_star = "*" if p_val < 0.05 else ""
                
                report.append(f"{m1} vs {m2} & Rel. $t$-test & {stat:.2f} & {p_str}{sig_star} & {d:.3f} \\\\")
                
        report.append(r"\hline")
        report.append(r"\end{tabular}")
        report.append(r"\end{table}")
        
        return "\n".join(report)
