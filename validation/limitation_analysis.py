"""
Boundary and Failure Mode Limitation Analysis
=============================================

This module runs extreme stress-testing on the analog layers and calibration
framework to identify structural limitation envelopes (failure bounds).
Specifically, it identifies:
1. Mismatch Thresholds: Mismatch levels where calibrators fail to recover accuracy.
2. Saturation Boundaries: Input scales that drive the op-amps into saturation.
3. Quantization Cliff: The minimum DAC/ADC resolution required for calibration.
4. Channel Information Capacity: Shannon capacity limit under specific SNR.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from calibration.hmac import HMACCalibrator
from validation.metrics import compute_metrics


class CalibrationLimitationAnalyzer:
    """
    Stress-tests the calibration framework to identify physical limitations
    and failure modes of analog neural networks.
    """

    def __init__(self, y_ideal: torch.Tensor, y_sim: torch.Tensor,
                 weight_matrix: torch.Tensor, labels: torch.Tensor,
                 input_data: Optional[torch.Tensor] = None,
                 rest_of_network_fn = None):
        self.y_ideal = y_ideal
        self.y_sim = y_sim
        self.W = weight_matrix
        self.labels = labels
        self.input_data = input_data
        self.rest_of_network_fn = rest_of_network_fn
        
        self.N, self.C = y_ideal.shape

    def find_mismatch_cliff(self, mismatch_range: List[float], seed: int = 42) -> Dict:
        """
        Sweeps mismatch std dev to identify the "cliff" where calibration accuracy drops.
        """
        from analog_layers.analog_linear import AnalogLinear
        
        accuracies_uncal = []
        accuracies_hmac = []
        rmses_uncal = []
        rmses_hmac = []
        
        # Split data for calibration fitting
        split_idx = int(self.N * 0.5)
        
        for mismatch in mismatch_range:
            # Re-simulate with this mismatch
            config = {
                'resistor_mismatch': mismatch,
                'enable_mismatch': True,
                'quantization_bits': 8,
                'noise_sigma': 0.0,
                'opamp_offset': 0.0,
                'saturation_vmax': 5.0
            }
            
            in_features = self.W.shape[1]
            out_features = self.W.shape[0]
            
            # Recreate simulated output with this mismatch
            analog_layer = AnalogLinear(in_features, out_features, bias=False, config=config)
            analog_layer.weight.data.copy_(self.W)
            
            # Predict
            with torch.no_grad():
                y_mismatched = analog_layer(self.input_data)
                
            # Split
            y_mismatched_train, y_mismatched_test = y_mismatched[:split_idx], y_mismatched[split_idx:]
            y_ideal_train, y_ideal_test = self.y_ideal[:split_idx], self.y_ideal[split_idx:]
            labels_test = self.labels[split_idx:]
            
            # Fit HMAC
            hmac = HMACCalibrator(weight_matrix=self.W, polynomial_degree=1, mismatch_sigma=mismatch)
            try:
                hmac.fit(y_mismatched_train, y_ideal_train, weight_matrix=self.W, input_data=self.input_data[:split_idx])
                y_cal_test = hmac.calibrate(y_mismatched_test)
                
                # Metrics
                m_cal = compute_metrics(
                    y_ideal_test, y_mismatched_test, y_cal_test, labels_test,
                    rest_of_network_fn=self.rest_of_network_fn
                )
                acc_hmac = float(m_cal['accuracy_calibrated'])
                rmse_hmac = float(m_cal['rmse_post_calibration'])
            except Exception:
                acc_hmac = 0.0
                rmse_hmac = 99.0
                
            m_uncal = compute_metrics(
                y_ideal_test, y_mismatched_test, y_mismatched_test, labels_test,
                rest_of_network_fn=self.rest_of_network_fn
            )
            
            accuracies_uncal.append(float(m_uncal['accuracy_sim']))
            accuracies_hmac.append(acc_hmac)
            rmses_uncal.append(float(m_uncal['rmse_pre_calibration']))
            rmses_hmac.append(rmse_hmac)
            
        return {
            'mismatch_values': mismatch_range,
            'accuracy_uncalibrated': accuracies_uncal,
            'accuracy_calibrated_hmac': accuracies_hmac,
            'rmse_uncalibrated': rmses_uncal,
            'rmse_calibrated_hmac': rmses_hmac,
            # Cliff threshold is defined as mismatch value where calibrated accuracy drops below 90% of ideal
            'cliff_mismatch_std': float(mismatch_range[np.where(np.array(accuracies_hmac) < 0.9 * accuracies_hmac[0])[0][0]]) \
                                  if any(np.array(accuracies_hmac) < 0.9 * accuracies_hmac[0]) else None
        }

    def compute_shannon_capacity(self, snr_db_range: List[float]) -> Dict:
        """
        Computes the theoretical Shannon information capacity limit of the analog crossbar channel.
        C = B * log2(1 + SNR)
        Shows how noise limits the equivalent bit-resolution of weights and activations.
        """
        capacity_bits_per_weight = []
        for snr_db in snr_db_range:
            snr = 10 ** (snr_db / 10.0)
            capacity = float(0.5 * np.log2(1 + snr)) # bits per analog channel usage
            capacity_bits_per_weight.append(capacity)
            
        return {
            'snr_db': snr_db_range,
            'capacity_bits': capacity_bits_per_weight
        }

    def analyze_saturation_failures(self, input_scales: List[float], vmax: float = 2.5) -> Dict:
        """
        Analyzes how input voltage scaling causes op-amp saturation and leads to calibration breakdown.
        """
        from analog_layers.analog_linear import AnalogLinear
        
        saturation_ratios = []
        accuracies_hmac = []
        
        split_idx = int(self.N * 0.5)
        
        for scale in input_scales:
            scaled_inputs = self.input_data * scale
            
            config = {
                'resistor_mismatch': 0.01,
                'enable_mismatch': True,
                'quantization_bits': 8,
                'noise_sigma': 0.0,
                'opamp_offset': 0.0,
                'saturation_vmax': vmax
            }
            
            in_features = self.W.shape[1]
            out_features = self.W.shape[0]
            
            # Predict ideal (linear) vs simulated (saturated)
            bias_term = torch.zeros(out_features)
            y_ideal_scaled = torch.matmul(scaled_inputs, self.W.t())
            
            analog_layer = AnalogLinear(in_features, out_features, bias=False, config=config)
            analog_layer.weight.data.copy_(self.W)
            
            with torch.no_grad():
                y_sim_scaled = analog_layer(scaled_inputs)
                
            # Count how many outputs saturated
            sat_mask = (torch.abs(y_sim_scaled) >= 0.99 * vmax)
            sat_ratio = float(sat_mask.float().mean().item())
            saturation_ratios.append(sat_ratio)
            
            # Fit/evaluate calibrator on scaled outputs
            y_sim_train, y_sim_test = y_sim_scaled[:split_idx], y_sim_scaled[split_idx:]
            y_ideal_train, y_ideal_test = y_ideal_scaled[:split_idx], y_ideal_scaled[split_idx:]
            labels_test = self.labels[split_idx:]
            
            hmac = HMACCalibrator(weight_matrix=self.W, polynomial_degree=1)
            try:
                hmac.fit(y_sim_train, y_ideal_train, weight_matrix=self.W, input_data=scaled_inputs[:split_idx])
                y_cal_test = hmac.calibrate(y_sim_test)
                m = compute_metrics(
                    y_ideal_test, y_sim_test, y_cal_test, labels_test,
                    rest_of_network_fn=self.rest_of_network_fn
                )
                acc = float(m['accuracy_calibrated'])
            except Exception:
                acc = 0.0
                
            accuracies_hmac.append(acc)
            
        return {
            'input_scales': input_scales,
            'saturation_ratios': saturation_ratios,
            'accuracies_hmac': accuracies_hmac
        }
