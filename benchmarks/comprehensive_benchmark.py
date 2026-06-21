"""
Comprehensive Benchmark Suite
==============================

Runs all benchmarks:
1. Core analog layer performance
2. NAS architecture search
3. Adversarial training robustness
4. Energy efficiency analysis
5. HuggingFace model evaluation (if transformers installed)
"""

import torch
import numpy as np
import json
import time
from typing import Dict
from datetime import datetime


class ComprehensiveBenchmark:
    """
    Runs comprehensive benchmarks across all OpenAnalogNN features.
    
    Generates a detailed report with:
    - Performance metrics
    - Energy efficiency analysis
    - Robustness evaluation
    - Comparison with digital baselines
    """
    
    def __init__(self, config_path: str = "./configs/config.yaml"):
        from experiments.config_loader import load_config
        self.config = load_config(config_path)
        self.results = {}
        
    def run_all(self, 
                skip_huggingface: bool = False,
                skip_nas: bool = False,
                quick_mode: bool = False) -> Dict:
        """
        Run all benchmarks.
        
        Args:
            skip_huggingface: Skip HuggingFace benchmarks (requires transformers)
            skip_nas: Skip NAS (computationally expensive)
            quick_mode: Use smaller datasets/fewer iterations for faster testing
        
        Returns:
            Dictionary with all benchmark results
        """
        print("=" * 80)
        print("OpenAnalogNN Comprehensive Benchmark Suite")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # 1. Core analog layers
        print("[1/6] Core Analog Layer Benchmarks")
        print("-" * 80)
        self.results['core_layers'] = self._benchmark_core_layers(quick_mode)
        print()
        
        # 2. Circuit simulation
        print("[2/6] Circuit Simulation Benchmarks")
        print("-" * 80)
        self.results['circuit_sim'] = self._benchmark_circuit_simulation(quick_mode)
        print()
        
        # 3. Calibration methods
        print("[3/6] Calibration Method Benchmarks")
        print("-" * 80)
        self.results['calibration'] = self._benchmark_calibration(quick_mode)
        print()
        
        # 4. Energy efficiency
        print("[4/6] Energy Efficiency Analysis")
        print("-" * 80)
        self.results['energy'] = self._benchmark_energy_efficiency(quick_mode)
        print()
        
        # 5. NAS (optional)
        if not skip_nas:
            print("[5/6] Neural Architecture Search")
            print("-" * 80)
            self.results['nas'] = self._benchmark_nas(quick_mode)
            print()
        else:
            print("[5/6] Neural Architecture Search (SKIPPED)")
            print()
        
        # 6. HuggingFace (optional)
        if not skip_huggingface:
            print("[6/6] HuggingFace LLM Benchmarks")
            print("-" * 80)
            try:
                self.results['huggingface'] = self._benchmark_huggingface(quick_mode)
            except ImportError as e:
                print(f"Skipped: {e}")
                self.results['huggingface'] = {'status': 'skipped', 'reason': str(e)}
            print()
        else:
            print("[6/6] HuggingFace LLM Benchmarks (SKIPPED)")
            print()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _benchmark_core_layers(self, quick_mode: bool) -> Dict:
        """Benchmark core analog layer performance."""
        from analog_layers.analog_linear import AnalogLinear
        
        n_runs = 10 if quick_mode else 100
        sizes = [(128, 64), (256, 128), (512, 256)] if quick_mode else [(128, 64), (256, 128), (512, 256), (1024, 512)]
        
        results = {}
        
        for in_dim, out_dim in sizes:
            # Create analog layer
            config = {
                'resistor_mismatch': 0.01,
                'noise_sigma': 0.01,
                'opamp_offset': 0.002
            }
            analog_layer = AnalogLinear(in_dim, out_dim, config=config)
            digital_layer = torch.nn.Linear(in_dim, out_dim)
            
            # Copy weights
            with torch.no_grad():
                digital_layer.weight.copy_(analog_layer.weight)
                digital_layer.bias.copy_(analog_layer.bias)
            
            # Benchmark
            x = torch.randn(32, in_dim)
            
            # Warmup
            for _ in range(10):
                _ = analog_layer(x)
                _ = digital_layer(x)
            
            # Analog timing
            start = time.time()
            for _ in range(n_runs):
                _ = analog_layer(x)
            analog_time = (time.time() - start) / n_runs
            
            # Digital timing
            start = time.time()
            for _ in range(n_runs):
                _ = digital_layer(x)
            digital_time = (time.time() - start) / n_runs
            
            # Accuracy
            with torch.no_grad():
                analog_out = analog_layer(x)
                digital_out = digital_layer(x)
                error = torch.mean(torch.abs(analog_out - digital_out)).item()
                accuracy = 1.0 - error
            
            results[f'{in_dim}x{out_dim}'] = {
                'analog_latency_ms': analog_time * 1000,
                'digital_latency_ms': digital_time * 1000,
                'speedup': digital_time / analog_time,
                'accuracy': accuracy,
                'mean_absolute_error': error
            }
            
            print(f"  {in_dim}x{out_dim}: Analog={analog_time*1000:.3f}ms, "
                  f"Digital={digital_time*1000:.3f}ms, Accuracy={accuracy:.4f}")
        
        return results
    
    def _benchmark_circuit_simulation(self, quick_mode: bool) -> Dict:
        """Benchmark circuit simulation performance."""
        from spice.fallback_solver import FallbackNodalSolver
        
        n_samples = 10 if quick_mode else 100
        sizes = [(64, 32), (128, 64)] if quick_mode else [(64, 32), (128, 64), (256, 128)]
        
        results = {}
        
        for out_dim, in_dim in sizes:
            weight = torch.randn(out_dim, in_dim)
            bias = torch.randn(out_dim)
            x = torch.randn(n_samples, in_dim)
            
            config = {
                'resistor_mismatch': 0.01,
                'noise_sigma': 0.01,
                'seed': 42
            }
            
            start = time.time()
            output = FallbackNodalSolver.solve_closed_form(weight, bias, x, config)
            elapsed = time.time() - start
            
            results[f'{out_dim}x{in_dim}'] = {
                'n_samples': n_samples,
                'total_time_ms': elapsed * 1000,
                'time_per_sample_ms': (elapsed / n_samples) * 1000,
                'throughput_samples_per_sec': n_samples / elapsed
            }
            
            print(f"  {out_dim}x{in_dim}: {elapsed*1000:.2f}ms for {n_samples} samples "
                  f"({n_samples/elapsed:.0f} samples/sec)")
        
        return results
    
    def _benchmark_calibration(self, quick_mode: bool) -> Dict:
        """Benchmark calibration methods."""
        from calibration.affine import AffineCalibrator
        from calibration.polynomial import PolynomialCalibrator
        from calibration.learned import LearnedCalibrator
        from calibration.hmac import HMACCalibrator
        
        n_samples = 100 if quick_mode else 1000
        n_features = 10
        
        # Generate synthetic data
        y_ideal = torch.randn(n_samples, n_features)
        y_sim = y_ideal + torch.randn(n_samples, n_features) * 0.1
        
        results = {}
        
        # Affine
        start = time.time()
        cal = AffineCalibrator()
        cal.fit(y_sim, y_ideal)
        y_cal = cal.calibrate(y_sim)
        elapsed = time.time() - start
        
        error = torch.mean(torch.abs(y_cal - y_ideal)).item()
        results['affine'] = {
            'fit_time_ms': elapsed * 1000,
            'calibration_error': error
        }
        print(f"  Affine: {elapsed*1000:.2f}ms, Error={error:.4f}")
        
        # Polynomial
        start = time.time()
        cal = PolynomialCalibrator(degree=2)
        cal.fit(y_sim, y_ideal)
        y_cal = cal.calibrate(y_sim)
        elapsed = time.time() - start
        
        error = torch.mean(torch.abs(y_cal - y_ideal)).item()
        results['polynomial'] = {
            'fit_time_ms': elapsed * 1000,
            'calibration_error': error
        }
        print(f"  Polynomial: {elapsed*1000:.2f}ms, Error={error:.4f}")
        
        # HMAC
        weight_matrix = torch.randn(n_features, 20)
        start = time.time()
        cal = HMACCalibrator(weight_matrix=weight_matrix, polynomial_degree=1)
        cal.fit(y_sim, y_ideal, weight_matrix=weight_matrix, input_data=torch.randn(n_samples, 20))
        y_cal = cal.calibrate(y_sim)
        elapsed = time.time() - start
        
        error = torch.mean(torch.abs(y_cal - y_ideal)).item()
        results['hmac'] = {
            'fit_time_ms': elapsed * 1000,
            'calibration_error': error
        }
        print(f"  HMAC: {elapsed*1000:.2f}ms, Error={error:.4f}")
        
        return results
    
    def _benchmark_energy_efficiency(self, quick_mode: bool) -> Dict:
        """Benchmark energy efficiency across all power modes."""
        from energy.analog_energy_model import AnalogEnergyModel
        
        sizes = [(128, 64), (256, 128)] if quick_mode else [(128, 64), (256, 128), (512, 256)]
        power_modes = ['standard', 'low', 'ultra_low']
        
        results = {}
        
        for tech_node in ['65nm', '28nm', '7nm']:
            tech_results = {}
            
            for in_dim, out_dim in sizes:
                weight = torch.randn(out_dim, in_dim)
                x = torch.randn(32, in_dim)
                
                best_efficiency = 0
                best_mode = ''
                best_details = {}
                
                for mode in power_modes:
                    energy_model = AnalogEnergyModel(tech_node=tech_node, power_mode=mode)
                    energy = energy_model.compare_with_digital(weight, x)
                    
                    eff = energy['efficiency_vs_gpu']
                    if eff > best_efficiency:
                        best_efficiency = eff
                        best_mode = mode
                        best_details = {
                            'analog_energy_J': energy['analog']['total_energy_J'],
                            'gpu_energy_J': energy['digital']['gpu_J'],
                            'efficiency_vs_gpu': eff,
                            'tops_per_watt': energy['analog']['tops_per_watt'],
                            'power_mode': mode,
                            'clock_freq_Hz': energy_model.clock_freq_Hz,
                            'analog_fJ_per_mac': energy['analog_fJ_per_mac'],
                            'gpu_fJ_per_mac': energy['gpu_fJ_per_mac'],
                        }
                
                tech_results[f'{in_dim}x{out_dim}'] = best_details
                print(f"  {tech_node} {in_dim}x{out_dim} ({best_mode}): "
                      f"Analog={best_details['analog_fJ_per_mac']:.1f}fJ/MAC, "
                      f"GPU={best_details['gpu_fJ_per_mac']:.0f}fJ/MAC, "
                      f"Efficiency={best_efficiency:.1f}x")
            
            results[tech_node] = tech_results
        
        return results
    
    def _benchmark_nas(self, quick_mode: bool) -> Dict:
        """Benchmark NAS."""
        from nas.analog_nas import AnalogNASSearch
        
        # Small dataset for quick testing
        X_train = torch.randn(100 if quick_mode else 500, 20)
        y_train = torch.randint(0, 5, (100 if quick_mode else 500,))
        X_test = torch.randn(50 if quick_mode else 200, 20)
        y_test = torch.randint(0, 5, (50 if quick_mode else 200,))
        
        config = {
            'resistor_mismatch': 0.01,
            'noise_sigma': 0.01
        }
        
        nas = AnalogNASSearch(
            input_dim=20,
            output_dim=5,
            analog_config=config
        )
        
        start = time.time()
        best_arch = nas.search(
            X_train, y_train, X_test, y_test,
            max_candidates=5 if quick_mode else 20,
            epochs=10 if quick_mode else 30
        )
        elapsed = time.time() - start
        
        return {
            'search_time_sec': elapsed,
            'best_architecture': {
                'hidden_dims': best_arch.hidden_dims,
                'accuracy': best_arch.accuracy,
                'robustness': best_arch.analog_robustness_score,
                'energy_efficiency': best_arch.energy_efficiency
            },
            'n_candidates_evaluated': len(nas.results)
        }
    
    def _benchmark_huggingface(self, quick_mode: bool) -> Dict:
        """Benchmark HuggingFace models."""
        from huggingface.analog_transformers import AnalogLLMBenchmark
        
        benchmark = AnalogLLMBenchmark(
            model_name='gpt2',
            analog_config={
                'resistor_mismatch': 0.01,
                'noise_sigma': 0.01
            }
        )
        
        # Load models
        digital_model, analog_model = benchmark.load_model()
        
        # Benchmark inference
        inference_results = benchmark.benchmark_inference(
            prompt="Hello world",
            max_new_tokens=10 if quick_mode else 50,
            n_runs=3 if quick_mode else 10
        )
        
        # Energy analysis
        energy_results = benchmark.energy_analysis()
        
        return {
            'inference': inference_results,
            'energy': energy_results
        }
    
    def _generate_summary(self):
        """Generate benchmark summary."""
        print("=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        
        # Core layers
        if 'core_layers' in self.results:
            avg_accuracy = np.mean([
                v['accuracy'] for v in self.results['core_layers'].values()
            ])
            print(f"Core Layers: Avg Accuracy = {avg_accuracy:.4f}")
        
        # Energy
        if 'energy' in self.results:
            for tech, tech_results in self.results['energy'].items():
                avg_efficiency = np.mean([
                    v['efficiency_vs_gpu'] for v in tech_results.values()
                ])
                print(f"Energy ({tech}): Avg Efficiency vs GPU = {avg_efficiency:.1f}x")
        
        # NAS
        if 'nas' in self.results:
            best = self.results['nas']['best_architecture']
            print(f"NAS: Best Architecture = {best['hidden_dims']}, "
                  f"Accuracy = {best['accuracy']:.4f}, "
                  f"Robustness = {best['robustness']:.4f}")
        
        print()
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def save_results(self, output_path: str = "./benchmarks/results/comprehensive_benchmark.json"):
        """Save benchmark results to JSON."""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert tensors to Python types
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        
        results_converted = convert(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    benchmark = ComprehensiveBenchmark()
    results = benchmark.run_all(quick_mode=True)
    benchmark.save_results()
