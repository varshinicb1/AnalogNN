"""
Enhanced Experiment Runner for OpenAnalogNN
==========================================

Orchestrates the full validation pipeline using modular stages.
Breaks circular dependencies by delegating to separate stage modules.
"""

import os
import torch
import json
from typing import Dict, Tuple, Optional

from datasets.loaders import get_dataset
from experiments.config_loader import load_config
from experiments.pipeline_stages import (
    train_baseline_stage,
    simulate_analog_stage,
    calibration_benchmark_stage,
    parity_evaluation_stage,
    circuit_optimization_stage,
    limitation_analysis_stage,
    statistical_trials_stage
)
from reports.figure_generation import PublicationFigureEngine
from analog_layers.analog_linear import AnalogLinear


class ExperimentRunner:
    """
    Orchestrates training, analog simulation sweeps, calibration,
    optimal resistance sizing, limitation checks, and report building.
    """

    def __init__(self, config_path: str = "./configs/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        self.seed = self.config.get('seed', 42)
        self.figure_engine = PublicationFigureEngine(output_dir="./figures")
        
    def load_data_and_train_baseline(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Loads the configured dataset and trains the digital baseline model.
        """
        d_cfg = self.config['dataset']
        
        print(f"Loading dataset: {d_cfg['name']} (Subset size: {d_cfg['subset_size']})...")
        X_train, y_train, X_test, y_test, num_features, num_classes = get_dataset(
            name=d_cfg['name'],
            subset_size=d_cfg['subset_size'],
            downsample_size=d_cfg.get('downsample_size', 8),
            seed=self.seed
        )
        
        print("Training Digital MLP Baseline...")
        model, history = train_baseline_stage(
            X_train, y_train, X_test, y_test, num_features, num_classes, self.config
        )
        
        return X_train, y_train, X_test, y_test, model

    def run_full_pipeline(self) -> Dict:
        """
        Runs the full OpenAnalogNN scientific validation pipeline using modular stages.
        """
        print("=== OpenAnalogNN Scientific Validation Pipeline ===")
        
        # 1. Digital Baseline
        X_train, y_train, X_test, y_test, digital_model = self.load_data_and_train_baseline()
        
        # Extract first linear layer for analog mapping
        digital_linear = None
        for layer in digital_model.network:
            if isinstance(layer, (torch.nn.Linear, AnalogLinear)):
                digital_linear = layer
                break
                
        if digital_linear is None:
            raise ValueError("No linear layer found in baseline model.")
            
        weight = digital_linear.weight.data
        bias = digital_linear.bias.data if digital_linear.bias is not None else None
        
        # Build rest of network for end-to-end evaluation
        rest_layers = []
        found_linear = False
        for layer in digital_model.network:
            if found_linear:
                rest_layers.append(layer)
            elif layer is digital_linear:
                found_linear = True
        rest_of_network = torch.nn.Sequential(*rest_layers)
        
        # 2. Analog Simulation
        y_ideal, y_sim = simulate_analog_stage(weight, bias, X_test, self.config)
        
        # 3. Calibration Benchmarking
        benchmark_results = calibration_benchmark_stage(
            y_ideal, y_sim, weight, X_test, y_test, rest_of_network, self.config
        )
        
        # 4. Parity Evaluation
        parity_results = parity_evaluation_stage(weight, bias, X_test, y_test, self.config)
        
        # 5. Circuit Optimization
        opt_results = circuit_optimization_stage(weight, self.config)
        
        # 6. Limitation Analysis
        limitation_results = limitation_analysis_stage(
            y_ideal, y_sim, weight, y_test, X_test, rest_of_network, self.config
        )
        
        # 7. Statistical Trials
        stats_summary = statistical_trials_stage(
            weight, bias, X_test, y_ideal, y_test, rest_of_network, self.config
        )
        
        # 8. Generate Figures and Reports
        self._generate_reports(benchmark_results, opt_results, limitation_results, 
                               stats_summary, weight, bias, X_test, y_ideal, y_sim, y_test)
        
        print("Validation pipeline successfully executed! Reports and figures compiled.")
        return {
            'benchmark': benchmark_results['metrics'],
            'parity': parity_results['accuracies'],
            'optimization': opt_results,
            'limitation': limitation_results,
            'statistics': stats_summary
        }
    
    def _generate_reports(self, benchmark_results, opt_results, limitation_results,
                         stats_summary, weight, bias, X_test, y_ideal, y_sim, y_test):
        """Generate figures and reports from pipeline results."""
        from validation.residual_analysis import ResidualAnalyzer
        from validation.error_bounds import AnalogErrorBound
        from spice.netlist_generator import NetlistGenerator
        from validation.statistical_analysis import StatisticalAnalysis
        import copy
        
        os.makedirs("./reports/paper_ready", exist_ok=True)
        os.makedirs("./netlists", exist_ok=True)
        
        # Parity plots
        y_ideal_test = benchmark_results['y_ideal_test']
        y_sim_test = benchmark_results['y_sim_test']
        y_post_hmac = benchmark_results['residuals']['HMAC (Linear)'] + y_ideal_test
        
        self.figure_engine.plot_calibration_parity(
            y_ideal=y_ideal_test, y_pre=y_sim_test, y_post=y_post_hmac,
            filename="calibration_parity.png"
        )
        
        # Residual diagnostics
        y_cal_hmac = torch.tensor(benchmark_results['residuals']['HMAC (Linear)'] + y_ideal_test)
        y_cal_ols = torch.tensor(benchmark_results['residuals']['Affine'] + y_ideal_test)
        
        analyzer = ResidualAnalyzer(
            y_ideal=torch.tensor(y_ideal_test),
            y_sim=torch.tensor(y_sim_test),
            y_cal_hmac=y_cal_hmac,
            y_cal_ols=y_cal_ols,
            weight_matrix=weight
        )
        analyzer.plot_residual_diagnostics(save_dir="./figures")
        
        # Error bounds analysis
        bound_calc = AnalogErrorBound(
            weight_matrix=weight, bias=bias,
            mismatch_sigma=self.config['analog'].get('resistor_mismatch', 0.01),
            offset_sigma=self.config['analog'].get('opamp_offset', 0.002),
            noise_sigma=self.config['analog'].get('noise_sigma', 0.05),
            quantization_bits=self.config['analog'].get('quantization_bits', 6)
        )
        sensitivity_data = bound_calc.sensitivity_analysis()
        self.figure_engine.plot_sensitivity_analysis(sensitivity_data, filename="sensitivity_analysis.png")
        
        # SPICE netlists
        NetlistGenerator.generate(
            weight=weight, bias=bias, x=X_test[0],
            r_ref=self.config['circuit']['r_ref'],
            v_ref=self.config['circuit']['v_ref'],
            vmax=self.config['analog']['saturation_vmax'],
            output_dir="./netlists",
            filename="analog_layer_ngspice.cir",
            backend="ngspice"
        )
        NetlistGenerator.generate(
            weight=weight, bias=bias, x=X_test[0],
            r_ref=self.config['circuit']['r_ref'],
            v_ref=self.config['circuit']['v_ref'],
            vmax=self.config['analog']['saturation_vmax'],
            output_dir="./netlists",
            filename="analog_layer_ltspice.cir",
            backend="ltspice"
        )
        
        # LaTeX table
        StatisticalAnalysis.generate_latex_table(stats_summary, "./reports/paper_ready/performance_table.tex")
        
        # JSON report
        report_data = {
            'benchmark': benchmark_results['metrics'],
            'optimization': opt_results,
            'limitation': limitation_results,
            'statistics': stats_summary
        }
        
        with open("./reports/paper_ready/report_data.json", "w") as f:
            json.dump(report_data, f, indent=4)

