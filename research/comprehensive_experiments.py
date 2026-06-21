"""
Comprehensive Research Experiments
===================================

Systematic empirical study to discover novel phenomena in analog neural networks:
1. Phase transitions: Critical mismatch thresholds where accuracy collapses
2. Scaling laws: How robustness scales with network depth/width
3. Calibration effectiveness: Which methods work best for which architectures
4. Adversarial training transfer: Does adversarial training help analog robustness?
5. Energy-accuracy Pareto frontiers

This is real research - we're looking for unexpected patterns and empirical laws
that haven't been documented before.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from datetime import datetime

from datasets.loaders import get_dataset
from experiments.models import DigitalMLP, train_model
from analog_layers.analog_linear import AnalogLinear
from calibration.affine import AffineCalibrator
from calibration.polynomial import PolynomialCalibrator
from calibration.hmac import HMACCalibrator
from calibration.learned import LearnedCalibrator
from energy.analog_energy_model import AnalogEnergyModel


class ResearchExperiment:
    """
    Systematic research experiments to discover novel phenomena.
    """
    
    def __init__(self, output_dir: str = "./research_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'experiments': []
            }
        }
        
    def download_datasets(self):
        """Download and cache real datasets."""
        print("=" * 80)
        print("Downloading Real Datasets")
        print("=" * 80)
        
        datasets = {}
        
        # Iris (small, fast)
        print("\n[1/3] Loading Iris dataset...")
        X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
            name='iris',
            subset_size=100,
            downsample_size=8,
            seed=42
        )
        datasets['iris'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'n_features': n_features, 'n_classes': n_classes
        }
        print(f"  Iris: {len(X_train)} train, {len(X_test)} test, {n_features} features, {n_classes} classes")
        
        # MNIST downsampled (medium)
        print("\n[2/3] Loading MNIST (8x8 downsampled)...")
        X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
            name='mnist',
            subset_size=500,
            downsample_size=8,
            seed=42
        )
        datasets['mnist'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'n_features': n_features, 'n_classes': n_classes
        }
        print(f"  MNIST: {len(X_train)} train, {len(X_test)} test, {n_features} features, {n_classes} classes")
        
        # Fashion-MNIST downsampled (medium)
        print("\n[3/3] Loading Fashion-MNIST (8x8 downsampled)...")
        X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
            name='fashion',
            subset_size=500,
            downsample_size=8,
            seed=42
        )
        datasets['fashion'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'n_features': n_features, 'n_classes': n_classes
        }
        print(f"  Fashion: {len(X_train)} train, {len(X_test)} test, {n_features} features, {n_classes} classes")
        
        return datasets
    
    def experiment_1_phase_transitions(self, datasets: dict):
        """
        Experiment 1: Mismatch Phase Transitions
        
        Hypothesis: There exists a critical mismatch threshold where accuracy
        suddenly collapses, analogous to phase transitions in physics.
        
        Method: Sweep resistor_mismatch from 0% to 50% and measure accuracy.
        Look for sharp drops (phase transitions).
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Mismatch Phase Transitions")
        print("=" * 80)
        
        results = []
        
        # Sweep mismatch levels
        mismatch_levels = np.linspace(0.0, 0.5, 51)  # 0% to 50% in 51 steps
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # Train baseline model
            model = DigitalMLP(
                input_dim=data['n_features'],
                hidden_dims=[128, 64],
                output_dim=data['n_classes']
            )
            
            history = train_model(
                model=model,
                X_train=data['X_train'],
                y_train=data['y_train'],
                X_test=data['X_test'],
                y_test=data['y_test'],
                epochs=50,
                lr=0.001,
                batch_size=32,
                seed=42
            )
            
            # Baseline accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(data['X_test'])
                predictions = torch.argmax(outputs, dim=1)
                baseline_acc = (predictions == data['y_test']).float().mean().item()
            
            print(f"  Baseline accuracy: {baseline_acc:.4f}")
            
            # Sweep mismatch
            accuracies = []
            for mismatch in mismatch_levels:
                # Create analog model with this mismatch level
                analog_config = {
                    'resistor_mismatch': mismatch,
                    'noise_sigma': 0.0,  # Disable other non-idealities
                    'opamp_offset': 0.0,
                    'quantization_bits': 0,
                    'saturation_vmax': 0.0,
                    'seed': 42
                }
                
                analog_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[128, 64],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                
                # Copy weights
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                # Evaluate
                analog_model.eval()
                with torch.no_grad():
                    outputs = analog_model(data['X_test'])
                    predictions = torch.argmax(outputs, dim=1)
                    acc = (predictions == data['y_test']).float().mean().item()
                
                accuracies.append(acc)
                
                if mismatch in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    print(f"    Mismatch {mismatch*100:.0f}%: Accuracy {acc:.4f}")
            
            # Find phase transition (largest drop)
            drops = np.diff(accuracies)
            max_drop_idx = np.argmin(drops)
            critical_mismatch = mismatch_levels[max_drop_idx]
            max_drop = drops[max_drop_idx]
            
            print(f"  Phase transition at {critical_mismatch*100:.1f}% mismatch (drop: {max_drop:.4f})")
            
            results.append({
                'dataset': dataset_name,
                'baseline_acc': baseline_acc,
                'mismatch_levels': mismatch_levels.tolist(),
                'accuracies': accuracies,
                'critical_mismatch': critical_mismatch,
                'max_drop': max_drop
            })
        
        # Save results
        self.results['phase_transitions'] = results
        self.results['metadata']['experiments'].append('phase_transitions')
        
        # Plot
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            axes[i].plot(result['mismatch_levels'], result['accuracies'], 'b-', linewidth=2)
            axes[i].axvline(result['critical_mismatch'], color='r', linestyle='--', 
                          label=f'Critical: {result["critical_mismatch"]*100:.1f}%')
            axes[i].set_xlabel('Resistor Mismatch (fraction)')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(f'{result["dataset"].upper()} - Phase Transition')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase_transitions.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'phase_transitions.png'}")
        
        return results
    
    def experiment_2_scaling_laws(self, datasets: dict):
        """
        Experiment 2: Architecture Scaling Laws
        
        Hypothesis: There are predictable scaling laws relating network
        depth/width to analog robustness.
        
        Method: Train networks with varying depth and width, measure robustness.
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Architecture Scaling Laws")
        print("=" * 80)
        
        results = []
        
        # Architecture configurations
        architectures = [
            # Varying depth (fixed width=64)
            {'hidden_dims': [64], 'name': 'D1-W64'},
            {'hidden_dims': [64, 64], 'name': 'D2-W64'},
            {'hidden_dims': [64, 64, 64], 'name': 'D3-W64'},
            {'hidden_dims': [64, 64, 64, 64], 'name': 'D4-W64'},
            # Varying width (fixed depth=2)
            {'hidden_dims': [32, 32], 'name': 'D2-W32'},
            {'hidden_dims': [64, 64], 'name': 'D2-W64'},
            {'hidden_dims': [128, 128], 'name': 'D2-W128'},
            {'hidden_dims': [256, 256], 'name': 'D2-W256'},
        ]
        
        # Fixed analog config for robustness testing
        analog_config = {
            'resistor_mismatch': 0.05,  # 5% mismatch
            'noise_sigma': 0.01,
            'opamp_offset': 0.002,
            'quantization_bits': 8,
            'saturation_vmax': 2.5,
            'seed': 42
        }
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            arch_results = []
            
            for arch in architectures:
                # Train clean model
                model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=arch['hidden_dims'],
                    output_dim=data['n_classes']
                )
                
                history = train_model(
                    model=model,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_test=data['X_test'],
                    y_test=data['y_test'],
                    epochs=50,
                    lr=0.001,
                    batch_size=32,
                    seed=42
                )
                
                # Clean accuracy
                model.eval()
                with torch.no_grad():
                    outputs = model(data['X_test'])
                    predictions = torch.argmax(outputs, dim=1)
                    clean_acc = (predictions == data['y_test']).float().mean().item()
                
                # Analog accuracy (average over 5 seeds)
                analog_accs = []
                for seed in range(5):
                    analog_config['seed'] = seed
                    
                    analog_model = DigitalMLP(
                        input_dim=data['n_features'],
                        hidden_dims=arch['hidden_dims'],
                        output_dim=data['n_classes'],
                        analog_config=analog_config
                    )
                    analog_model.load_state_dict(model.state_dict(), strict=False)
                    
                    analog_model.eval()
                    with torch.no_grad():
                        outputs = analog_model(data['X_test'])
                        predictions = torch.argmax(outputs, dim=1)
                        acc = (predictions == data['y_test']).float().mean().item()
                        analog_accs.append(acc)
                
                analog_acc = np.mean(analog_accs)
                robustness = analog_acc / clean_acc if clean_acc > 0 else 0
                
                # Count parameters
                n_params = sum(p.numel() for p in model.parameters())
                
                arch_results.append({
                    'name': arch['name'],
                    'hidden_dims': arch['hidden_dims'],
                    'depth': len(arch['hidden_dims']),
                    'width': max(arch['hidden_dims']),
                    'n_params': n_params,
                    'clean_acc': clean_acc,
                    'analog_acc': analog_acc,
                    'robustness': robustness
                })
                
                print(f"  {arch['name']}: Clean={clean_acc:.4f}, Analog={analog_acc:.4f}, "
                      f"Robustness={robustness:.4f}, Params={n_params}")
            
            results.append({
                'dataset': dataset_name,
                'architectures': arch_results
            })
        
        # Save results
        self.results['scaling_laws'] = results
        self.results['metadata']['experiments'].append('scaling_laws')
        
        # Plot scaling laws
        n_datasets = len(results)
        fig, axes = plt.subplots(2, n_datasets, figsize=(6*n_datasets, 10))
        
        if n_datasets == 1:
            axes = axes.reshape(2, 1)
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            archs = result['architectures']
            
            # Depth vs Robustness
            depths = [a['depth'] for a in archs if a['width'] == 64]
            robustness_depth = [a['robustness'] for a in archs if a['width'] == 64]
            
            axes[0, i].plot(depths, robustness_depth, 'bo-', linewidth=2, markersize=8)
            axes[0, i].set_xlabel('Network Depth')
            axes[0, i].set_ylabel('Robustness (Analog/Clean)')
            axes[0, i].set_title(f'{dataset_name.upper()} - Depth Scaling')
            axes[0, i].grid(True, alpha=0.3)
            
            # Width vs Robustness
            widths = [a['width'] for a in archs if a['depth'] == 2]
            robustness_width = [a['robustness'] for a in archs if a['depth'] == 2]
            
            axes[1, i].plot(widths, robustness_width, 'rs-', linewidth=2, markersize=8)
            axes[1, i].set_xlabel('Network Width')
            axes[1, i].set_ylabel('Robustness (Analog/Clean)')
            axes[1, i].set_title(f'{dataset_name.upper()} - Width Scaling')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'scaling_laws.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'scaling_laws.png'}")
        
        return results
    
    def experiment_3_calibration_effectiveness(self, datasets: dict):
        """
        Experiment 3: Calibration Effectiveness vs Architecture
        
        Hypothesis: Different calibration methods work better for different
        architectures. There may be predictable patterns.
        
        Method: Test multiple calibration methods on multiple architectures.
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: Calibration Effectiveness")
        print("=" * 80)
        
        results = []
        
        architectures = [
            {'hidden_dims': [64], 'name': 'Shallow'},
            {'hidden_dims': [128, 64], 'name': 'Deep'},
            {'hidden_dims': [256, 128], 'name': 'Wide'},
        ]
        
        analog_config = {
            'resistor_mismatch': 0.05,
            'noise_sigma': 0.01,
            'opamp_offset': 0.002,
            'quantization_bits': 8,
            'saturation_vmax': 2.5,
            'seed': 42
        }
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            arch_results = []
            
            for arch in architectures:
                # Train model
                model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=arch['hidden_dims'],
                    output_dim=data['n_classes']
                )
                
                history = train_model(
                    model=model,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_test=data['X_test'],
                    y_test=data['y_test'],
                    epochs=50,
                    lr=0.001,
                    batch_size=32,
                    seed=42
                )
                
                # Get ideal outputs
                model.eval()
                with torch.no_grad():
                    y_ideal = model(data['X_test'])
                
                # Get analog outputs
                analog_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=arch['hidden_dims'],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                analog_model.eval()
                with torch.no_grad():
                    y_sim = analog_model(data['X_test'])
                
                # Test calibration methods
                calibration_results = {}
                
                # 1. Affine calibration
                cal = AffineCalibrator()
                cal.fit(y_sim, y_ideal)
                y_cal = cal.calibrate(y_sim)
                error = torch.mean(torch.abs(y_cal - y_ideal)).item()
                calibration_results['Affine'] = error
                print(f"  {arch['name']} - Affine: Error={error:.4f}")
                
                # 2. Polynomial calibration
                cal = PolynomialCalibrator(degree=2)
                cal.fit(y_sim, y_ideal)
                y_cal = cal.calibrate(y_sim)
                error = torch.mean(torch.abs(y_cal - y_ideal)).item()
                calibration_results['Polynomial'] = error
                print(f"  {arch['name']} - Polynomial: Error={error:.4f}")
                
                # 3. HMAC calibration
                weight_matrix = model.network[0].weight if hasattr(model.network[0], 'weight') else torch.randn(data['n_classes'], data['n_features'])
                cal = HMACCalibrator(weight_matrix=weight_matrix, polynomial_degree=1)
                cal.fit(y_sim, y_ideal, weight_matrix=weight_matrix, input_data=data['X_test'])
                y_cal = cal.calibrate(y_sim)
                error = torch.mean(torch.abs(y_cal - y_ideal)).item()
                calibration_results['HMAC'] = error
                print(f"  {arch['name']} - HMAC: Error={error:.4f}")
                
                # 4. Learned calibration
                cal = LearnedCalibrator(hidden_dim=32, epochs=50, lr=0.001)
                cal.fit(y_sim, y_ideal)
                y_cal = cal.calibrate(y_sim)
                error = torch.mean(torch.abs(y_cal - y_ideal)).item()
                calibration_results['Learned'] = error
                print(f"  {arch['name']} - Learned: Error={error:.4f}")
                
                arch_results.append({
                    'name': arch['name'],
                    'hidden_dims': arch['hidden_dims'],
                    'calibration_errors': calibration_results
                })
            
            results.append({
                'dataset': dataset_name,
                'architectures': arch_results
            })
        
        # Save results
        self.results['calibration_effectiveness'] = results
        self.results['metadata']['experiments'].append('calibration_effectiveness')
        
        # Plot heatmap
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            archs = result['architectures']
            
            # Build matrix
            methods = list(archs[0]['calibration_errors'].keys())
            arch_names = [a['name'] for a in archs]
            
            matrix = np.zeros((len(arch_names), len(methods)))
            for j, arch in enumerate(archs):
                for k, method in enumerate(methods):
                    matrix[j, k] = arch['calibration_errors'][method]
            
            # Plot heatmap
            im = axes[i].imshow(matrix, cmap='RdYlGn_r', aspect='auto')
            axes[i].set_xticks(range(len(methods)))
            axes[i].set_xticklabels(methods, rotation=45, ha='right')
            axes[i].set_yticks(range(len(arch_names)))
            axes[i].set_yticklabels(arch_names)
            axes[i].set_title(f'{dataset_name.upper()} - Calibration Error')
            
            # Add text annotations
            for j in range(len(arch_names)):
                for k in range(len(methods)):
                    axes[i].text(k, j, f'{matrix[j, k]:.3f}',
                               ha='center', va='center', fontsize=10)
            
            plt.colorbar(im, ax=axes[i])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'calibration_heatmap.png'}")
        
        return results
    
    def experiment_4_energy_accuracy_tradeoff(self, datasets: dict):
        """
        Experiment 4: Energy-Accuracy Pareto Frontier
        
        Hypothesis: There's a predictable tradeoff between energy consumption
        and accuracy, with diminishing returns at high energy.
        
        Method: Measure energy and accuracy for different architectures.
        """
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: Energy-Accuracy Tradeoff")
        print("=" * 80)
        
        results = []
        
        architectures = [
            {'hidden_dims': [32]},
            {'hidden_dims': [64]},
            {'hidden_dims': [128]},
            {'hidden_dims': [64, 64]},
            {'hidden_dims': [128, 64]},
            {'hidden_dims': [128, 128]},
        ]
        
        energy_model = AnalogEnergyModel(tech_node='28nm')
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            arch_results = []
            
            for arch in architectures:
                # Train model
                model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=arch['hidden_dims'],
                    output_dim=data['n_classes']
                )
                
                history = train_model(
                    model=model,
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_test=data['X_test'],
                    y_test=data['y_test'],
                    epochs=50,
                    lr=0.001,
                    batch_size=32,
                    seed=42
                )
                
                # Accuracy
                model.eval()
                with torch.no_grad():
                    outputs = model(data['X_test'])
                    predictions = torch.argmax(outputs, dim=1)
                    acc = (predictions == data['y_test']).float().mean().item()
                
                # Energy
                energy = energy_model.estimate_model_energy(model, data['X_test'][:10])
                
                arch_results.append({
                    'hidden_dims': arch['hidden_dims'],
                    'accuracy': acc,
                    'energy_J': energy['total_energy_J'],
                    'static_power_W': energy['total_static_power_W'],
                    'tops_per_watt': energy['avg_tops_per_watt']
                })
                
                print(f"  {arch['hidden_dims']}: Acc={acc:.4f}, "
                      f"Energy={energy['total_energy_J']:.2e}J, "
                      f"TOPS/W={energy['avg_tops_per_watt']:.2f}")
            
            results.append({
                'dataset': dataset_name,
                'architectures': arch_results
            })
        
        # Save results
        self.results['energy_accuracy'] = results
        self.results['metadata']['experiments'].append('energy_accuracy')
        
        # Plot Pareto frontier
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            archs = result['architectures']
            
            energies = [a['energy_J'] for a in archs]
            accuracies = [a['accuracy'] for a in archs]
            
            axes[i].scatter(energies, accuracies, s=100, alpha=0.7)
            
            # Label points
            for j, arch in enumerate(archs):
                axes[i].annotate(str(arch['hidden_dims']), 
                               (energies[j], accuracies[j]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[i].set_xlabel('Energy per Inference (J)')
            axes[i].set_ylabel('Accuracy')
            axes[i].set_title(f'{dataset_name.upper()} - Energy-Accuracy Tradeoff')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'energy_accuracy_pareto.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'energy_accuracy_pareto.png'}")
        
        return results
    
    def run_all_experiments(self):
        """Run all research experiments."""
        print("=" * 80)
        print("COMPREHENSIVE RESEARCH EXPERIMENTS")
        print("=" * 80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Download datasets
        datasets = self.download_datasets()
        
        # Run experiments
        self.experiment_1_phase_transitions(datasets)
        self.experiment_2_scaling_laws(datasets)
        self.experiment_3_calibration_effectiveness(datasets)
        self.experiment_4_energy_accuracy_tradeoff(datasets)
        
        # Save all results
        with open(self.output_dir / 'research_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("EXPERIMENTS COMPLETE")
        print("=" * 80)
        print(f"Results saved to: {self.output_dir}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """Generate a summary report of findings."""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        report = []
        report.append("# Research Findings Summary")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Phase transitions
        if 'phase_transitions' in self.results:
            report.append("## 1. Phase Transitions")
            report.append("\nCritical mismatch thresholds where accuracy collapses:\n")
            for result in self.results['phase_transitions']:
                report.append(f"- **{result['dataset'].upper()}**: "
                            f"{result['critical_mismatch']*100:.1f}% mismatch "
                            f"(max drop: {result['max_drop']:.4f})")
            report.append("")
        
        # Scaling laws
        if 'scaling_laws' in self.results:
            report.append("## 2. Scaling Laws")
            report.append("\nArchitecture robustness patterns:\n")
            for result in self.results['scaling_laws']:
                report.append(f"\n### {result['dataset'].upper()}")
                for arch in result['architectures']:
                    report.append(f"- {arch['name']}: Robustness = {arch['robustness']:.4f}")
            report.append("")
        
        # Calibration effectiveness
        if 'calibration_effectiveness' in self.results:
            report.append("## 3. Calibration Effectiveness")
            report.append("\nBest calibration methods by architecture:\n")
            for result in self.results['calibration_effectiveness']:
                report.append(f"\n### {result['dataset'].upper()}")
                for arch in result['architectures']:
                    best_method = min(arch['calibration_errors'].items(), 
                                    key=lambda x: x[1])
                    report.append(f"- **{arch['name']}**: {best_method[0]} "
                                f"(error: {best_method[1]:.4f})")
            report.append("")
        
        # Energy-accuracy
        if 'energy_accuracy' in self.results:
            report.append("## 4. Energy-Accuracy Tradeoffs")
            report.append("\nMost efficient architectures:\n")
            for result in self.results['energy_accuracy']:
                report.append(f"\n### {result['dataset'].upper()}")
                # Sort by efficiency (accuracy / energy)
                archs = sorted(result['architectures'],
                             key=lambda a: a['accuracy'] / a['energy_J'],
                             reverse=True)
                for arch in archs[:3]:  # Top 3
                    efficiency = arch['accuracy'] / arch['energy_J']
                    report.append(f"- {arch['hidden_dims']}: "
                                f"Acc={arch['accuracy']:.4f}, "
                                f"Energy={arch['energy_J']:.2e}J, "
                                f"Efficiency={efficiency:.2e}")
            report.append("")
        
        # Save report
        report_text = '\n'.join(report)
        with open(self.output_dir / 'summary_report.md', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nSaved: {self.output_dir / 'summary_report.md'}")


if __name__ == "__main__":
    experiment = ResearchExperiment()
    results = experiment.run_all_experiments()
