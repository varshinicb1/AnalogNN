import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import json

from datasets.loaders import get_dataset
from experiments.models import DigitalMLP, train_model, evaluate_model
from analog_layers.analog_linear import AnalogLinear
from spice.spice_runner import SpiceRunner
from validation.metrics import compute_metrics
from calibration.polynomial import PolynomialCalibrator

class ExperimentRunner:
    def __init__(self, config_path: str = "./configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.seed = self.config.get('seed', 42)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def load_data_and_train_baseline(self):
        """
        Loads the configured dataset, trains the digital MLP baseline model,
        and returns data tensors and the trained model.
        """
        d_cfg = self.config['dataset']
        m_cfg = self.config['model']
        
        print(f"Loading dataset: {d_cfg['name']} (Subset size: {d_cfg['subset_size']})...")
        X_train, y_train, X_test, y_test, num_features, num_classes = get_dataset(
            name=d_cfg['name'],
            subset_size=d_cfg['subset_size'],
            downsample_size=d_cfg.get('downsample_size', 8),
            seed=self.seed
        )
        
        print("Training Digital MLP Baseline...")
        model = DigitalMLP(num_features, m_cfg['hidden_dims'], num_classes)
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=m_cfg['epochs'],
            lr=m_cfg['lr'],
            batch_size=m_cfg['batch_size'],
            seed=self.seed
        )
        
        # Plot training curves
        from experiments.models import plot_training_curves
        plot_training_curves(history, "./figures")
        
        return X_train, y_train, X_test, y_test, model

    def run_sweeps(self, X_test: torch.Tensor, y_test: torch.Tensor, digital_model: torch.nn.Module,
                   save_dir: str = "./figures") -> dict:
        """
        Sweeps noise levels, mismatch levels, and quantization resolutions,
        producing publication-ready robustness curves.
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs("./benchmarks", exist_ok=True)
        
        # Convert digital model to analog layers (we assume mapping a single AnalogLinear layer)
        # In our MLP sequence, we target the first Linear layer for our SPICE/analog sweeps
        # to ensure fast simulation.
        digital_linear = None
        for layer in digital_model.network:
            if isinstance(layer, torch.nn.Linear):
                digital_linear = layer
                break
                
        if digital_linear is None:
            raise ValueError("No linear layer found in baseline model to simulate.")
            
        print("Starting Parametric Sweeps...")
        
        # 1. Sweep Weight Noise Standard Deviation (sigma in [0.0, 0.25])
        noise_sigmas = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25]
        noise_accs = []
        noise_rmses = []
        
        for sigma in noise_sigmas:
            cfg = self.config.copy()
            cfg['analog']['noise_sigma'] = sigma
            cfg['analog']['enable_noise'] = True
            
            # Setup layer
            analog_layer = AnalogLinear.from_digital(digital_linear, config=cfg['analog'])
            runner = SpiceRunner(config=cfg)
            
            # Run inference
            with torch.no_grad():
                y_sim = runner.run(analog_layer.weight, analog_layer.bias, X_test,
                                   r_ref=self.config['circuit']['r_ref'],
                                   v_ref=self.config['circuit']['v_ref'])
                y_ideal = digital_linear(X_test)
                
            metrics = compute_metrics(y_ideal, y_sim, y_cal=None, y_true=y_test)
            noise_accs.append(metrics['accuracy_sim'])
            noise_rmses.append(metrics['rmse_pre_calibration'])
            
        # 2. Sweep Resistor Mismatch (delta in [0%, 10%])
        mismatches = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
        mismatch_accs = []
        mismatch_rmses = []
        
        for delta in mismatches:
            cfg = self.config.copy()
            cfg['analog']['resistor_mismatch'] = delta
            cfg['analog']['enable_mismatch'] = True
            cfg['analog']['enable_noise'] = False # Isolate mismatch
            
            analog_layer = AnalogLinear.from_digital(digital_linear, config=cfg['analog'])
            runner = SpiceRunner(config=cfg)
            
            with torch.no_grad():
                y_sim = runner.run(analog_layer.weight, analog_layer.bias, X_test,
                                   r_ref=self.config['circuit']['r_ref'],
                                   v_ref=self.config['circuit']['v_ref'])
                y_ideal = digital_linear(X_test)
                
            metrics = compute_metrics(y_ideal, y_sim, y_cal=None, y_true=y_test)
            mismatch_accs.append(metrics['accuracy_sim'])
            mismatch_rmses.append(metrics['rmse_pre_calibration'])
            
        # 3. Sweep Quantization Resolution (n_bits in [2, 8])
        bits_list = [2, 3, 4, 5, 6, 8, 12]
        quant_accs = []
        quant_rmses = []
        
        for bits in bits_list:
            cfg = self.config.copy()
            cfg['analog']['quantization_bits'] = bits
            cfg['analog']['enable_quantization'] = True
            cfg['analog']['enable_noise'] = False
            cfg['analog']['enable_mismatch'] = False
            
            analog_layer = AnalogLinear.from_digital(digital_linear, config=cfg['analog'])
            runner = SpiceRunner(config=cfg)
            
            with torch.no_grad():
                y_sim = runner.run(analog_layer.weight, analog_layer.bias, X_test,
                                   r_ref=self.config['circuit']['r_ref'],
                                   v_ref=self.config['circuit']['v_ref'])
                y_ideal = digital_linear(X_test)
                
            metrics = compute_metrics(y_ideal, y_sim, y_cal=None, y_true=y_test)
            quant_accs.append(metrics['accuracy_sim'])
            quant_rmses.append(metrics['rmse_pre_calibration'])

        # --- Plots ---
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Noise Robustness Plot
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        color = '#1f77b4'
        ax1.set_xlabel('Weight Noise Standard Deviation ($\\sigma_w$)', fontsize=12)
        ax1.set_ylabel('Inference Accuracy', color=color, fontsize=12)
        ax1.plot(noise_sigmas, noise_accs, marker='o', linewidth=2, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = '#d62728'
        ax2.set_ylabel('Signal Output RMSE (V)', color=color, fontsize=12)
        ax2.plot(noise_sigmas, noise_rmses, marker='s', linewidth=2, color=color, linestyle='--', label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Synaptic Weight Noise Robustness Analysis', fontsize=13, fontweight='bold')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_noise.png'), dpi=300)
        plt.close()
        
        # Resistor Mismatch Plot
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        color = '#2ca02c'
        ax1.set_xlabel('Resistor Mismatch Tolerance ($\\sigma_R$)', fontsize=12)
        ax1.set_ylabel('Inference Accuracy', color=color, fontsize=12)
        ax1.plot(mismatches, mismatch_accs, marker='o', linewidth=2, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = '#9467bd'
        ax2.set_ylabel('Signal Output RMSE (V)', color=color, fontsize=12)
        ax2.plot(mismatches, mismatch_rmses, marker='s', linewidth=2, color=color, linestyle='--', label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Device Mismatch Tolerance Analysis', fontsize=13, fontweight='bold')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_mismatch.png'), dpi=300)
        plt.close()

        # Quantization Resolution Plot
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        color = '#ff7f0e'
        ax1.set_xlabel('Quantization Bit Resolution ($n_{bits}$)', fontsize=12)
        ax1.set_ylabel('Inference Accuracy', color=color, fontsize=12)
        ax1.plot(bits_list, quant_accs, marker='o', linewidth=2, color=color, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = '#17becf'
        ax2.set_ylabel('Signal Output RMSE (V)', color=color, fontsize=12)
        ax2.plot(bits_list, quant_rmses, marker='s', linewidth=2, color=color, linestyle='--', label='RMSE')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('DAC/ADC Quantization Robustness Analysis', fontsize=13, fontweight='bold')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'robustness_quantization.png'), dpi=300)
        plt.close()

        # Save results to disk
        sweep_data = {
            'noise': {'sigmas': noise_sigmas, 'accuracies': noise_accs, 'rmses': noise_rmses},
            'mismatch': {'deltas': mismatches, 'accuracies': mismatch_accs, 'rmses': mismatch_rmses},
            'quantization': {'bits': bits_list, 'accuracies': quant_accs, 'rmses': quant_rmses}
        }
        with open("./benchmarks/sweep_results.json", "w") as f:
            json.dump(sweep_data, f, indent=4)
            
        print("Parametric sweeps successfully completed! Visualizations saved in ./figures.")
        return sweep_data
