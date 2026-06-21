"""
Calibration Benchmarking Suite
==============================

Evaluates and compares the performance of different calibration models
(Affine, Polynomial, Learned MLP, and HMAC) across multiple hardware 
settings, noise levels, mismatch levels, and calibration sample sizes.
"""

import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Type

from calibration.affine import AffineCalibrator
from calibration.polynomial import PolynomialCalibrator
from calibration.learned import LearnedCalibrator
from calibration.hmac import HMACCalibrator
from validation.metrics import compute_metrics
from validation.statistical_tests import CalibrationStatisticalTester


class CalibrationBenchmarker:
    """
    Benchmarks calibrators under systematic sweeps of noise, mismatch,
    and sample sizes to compare performance, fit time, and data efficiency.
    """

    def __init__(self, y_ideal: torch.Tensor, y_sim: torch.Tensor,
                 weight_matrix: torch.Tensor, input_data: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 rest_of_network_fn = None):
        self.y_ideal = y_ideal
        self.y_sim = y_sim
        self.W = weight_matrix
        self.input_data = input_data
        self.labels = labels if labels is not None else torch.zeros(y_ideal.shape[0], dtype=torch.long)
        self.rest_of_network_fn = rest_of_network_fn
        
        self.N, self.C = y_ideal.shape

    def run_benchmark(self, train_ratio: float = 0.5, seed: int = 42) -> Dict:
        """
        Splits data into train/test calibration sets, fits all calibrators,
        and evaluates their post-calibration performance.
        """
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Split indexes
        split_idx = int(self.N * train_ratio)
        indices = np.random.permutation(self.N)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        # Datasets
        y_ideal_train, y_ideal_test = self.y_ideal[train_idx], self.y_ideal[test_idx]
        y_sim_train, y_sim_test = self.y_sim[train_idx], self.y_sim[test_idx]
        labels_train, labels_test = self.labels[train_idx], self.labels[test_idx]
        
        input_train = self.input_data[train_idx] if self.input_data is not None else None
        
        # Setup calibrators
        calibrators = {
            'Affine': PolynomialCalibrator(degree=1),
            'Quadratic': PolynomialCalibrator(degree=2),
            'Cubic': PolynomialCalibrator(degree=3),
            'Learned (MLP)': LearnedCalibrator(hidden_dim=32, epochs=150),
            'HMAC (Linear)': HMACCalibrator(weight_matrix=self.W, polynomial_degree=1),
            'HMAC (Quadratic)': HMACCalibrator(weight_matrix=self.W, polynomial_degree=2),
        }
        
        results = {}
        
        # Raw uncalibrated baseline
        raw_metrics = compute_metrics(
            y_ideal_test, y_sim_test, y_sim_test, labels_test,
            rest_of_network_fn=self.rest_of_network_fn
        )
        results['Uncalibrated'] = {
            'rmse': float(raw_metrics['rmse_pre_calibration']),
            'mae': float(np.mean(np.abs(y_sim_test.numpy() - y_ideal_test.numpy()))),
            'accuracy': float(raw_metrics['accuracy_sim']),
            'correlation': float(raw_metrics['correlation_pre_calibration']),
            'fit_time': 0.0,
            'eval_time': 0.0
        }
        
        absolute_errors = {'Uncalibrated': np.mean(np.abs(y_sim_test.numpy() - y_ideal_test.numpy()), axis=1)}
        residuals = {'Uncalibrated': (y_sim_test - y_ideal_test).numpy()}
        
        for name, cal in calibrators.items():
            # Time the fit phase
            start_fit = time.perf_counter()
            if isinstance(cal, HMACCalibrator):
                cal.fit(y_sim_train, y_ideal_train, weight_matrix=self.W, input_data=input_train)
            elif isinstance(cal, LearnedCalibrator):
                cal.fit(y_sim_train, y_ideal_train)
            else:
                cal.fit(y_sim_train, y_ideal_train)
            fit_time = time.perf_counter() - start_fit
            
            # Time the evaluation phase
            start_eval = time.perf_counter()
            y_cal_test = cal.calibrate(y_sim_test)
            eval_time = time.perf_counter() - start_eval
            
            # Compute metrics
            m = compute_metrics(
                y_ideal_test, y_sim_test, y_cal_test, labels_test,
                rest_of_network_fn=self.rest_of_network_fn
            )
            
            results[name] = {
                'rmse': float(m['rmse_post_calibration']),
                'mae': float(np.mean(np.abs(y_cal_test.numpy() - y_ideal_test.numpy()))),
                'accuracy': float(m['accuracy_calibrated']),
                'correlation': float(m['correlation_post_calibration']),
                'fit_time': float(fit_time),
                'eval_time': float(eval_time)
            }
            
            absolute_errors[name] = np.mean(np.abs(y_cal_test.numpy() - y_ideal_test.numpy()), axis=1)
            residuals[name] = (y_cal_test - y_ideal_test).numpy()
            
        # Run statistical tests
        stat_tester = CalibrationStatisticalTester()
        comparisons = stat_tester.compare_calibrators(absolute_errors)
        pairwise = stat_tester.pairwise_comparisons(absolute_errors)
        
        return {
            'metrics': results,
            'statistical_tests': comparisons,
            'pairwise': pairwise,
            'residuals': residuals,
            'y_ideal_test': y_ideal_test.numpy(),
            'y_sim_test': y_sim_test.numpy()
        }

    def benchmark_sample_efficiency(self, train_sizes: List[int], seed: int = 42) -> Dict:
        """
        Benchmarks calibrator RMSE as a function of the number of calibration samples.
        This demonstrates HMAC's stability with very small datasets due to the physical priors.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Reserve 30% for final evaluation, sweep sizes within remaining 70%
        split_idx = int(self.N * 0.7)
        indices = np.random.permutation(self.N)
        train_pool_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        y_ideal_test = self.y_ideal[test_idx]
        y_sim_test = self.y_sim[test_idx]
        labels_test = self.labels[test_idx]
        
        efficiency_data = {
            'sizes': train_sizes,
            'results': {name: [] for name in ['Affine', 'Quadratic', 'Learned (MLP)', 'HMAC (Linear)', 'HMAC (Quadratic)']}
        }
        
        for size in train_sizes:
            if size > len(train_pool_idx):
                for name in efficiency_data['results']:
                    efficiency_data['results'][name].append(np.nan)
                continue
                
            sub_train_idx = train_pool_idx[:size]
            y_ideal_train = self.y_ideal[sub_train_idx]
            y_sim_train = self.y_sim[sub_train_idx]
            input_train = self.input_data[sub_train_idx] if self.input_data is not None else None
            
            calibrators = {
                'Affine': PolynomialCalibrator(degree=1),
                'Quadratic': PolynomialCalibrator(degree=2),
                'Learned (MLP)': LearnedCalibrator(hidden_dim=32, epochs=100),
                'HMAC (Linear)': HMACCalibrator(weight_matrix=self.W, polynomial_degree=1),
                'HMAC (Quadratic)': HMACCalibrator(weight_matrix=self.W, polynomial_degree=2),
            }
            
            for name, cal in calibrators.items():
                try:
                    if isinstance(cal, HMACCalibrator):
                        cal.fit(y_sim_train, y_ideal_train, weight_matrix=self.W, input_data=input_train)
                    else:
                        cal.fit(y_sim_train, y_ideal_train)
                    y_cal_test = cal.calibrate(y_sim_test)
                    rmse = float(np.sqrt(np.mean((y_cal_test.numpy() - y_ideal_test.numpy()) ** 2)))
                except Exception as e:
                    # Fix: Raise exception instead of silently returning NaN
                    # This prevents masking bugs and makes debugging easier
                    raise RuntimeError(f"Calibration failed for {name} with size {size}: {str(e)}") from e
                efficiency_data['results'][name].append(rmse)
                
        return efficiency_data
