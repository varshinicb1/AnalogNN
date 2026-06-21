"""
Example: Analog-Aware Neural Architecture Search
=================================================

Demonstrates how to use NAS to discover architectures optimized
for analog hardware constraints.

This example:
1. Defines a search space of candidate architectures
2. Evaluates each under analog non-idealities
3. Finds the Pareto-optimal architectures
4. Compares with standard architectures
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from nas.analog_nas import AnalogNASSearch


def main():
    print("=" * 80)
    print("Analog-Aware Neural Architecture Search Example")
    print("=" * 80)
    
    # Create synthetic dataset (simulating a classification task)
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 500
    n_features = 20
    n_classes = 5
    
    X_train = torch.randn(n_samples, n_features)
    y_train = torch.randint(0, n_classes, (n_samples,))
    X_test = torch.randn(200, n_features)
    y_test = torch.randint(0, n_classes, (200,))
    
    print(f"\nDataset: {n_samples} training samples, {n_features} features, {n_classes} classes")
    
    # Define analog hardware constraints
    analog_config = {
        'resistor_mismatch': 0.02,  # 2% resistor tolerance
        'noise_sigma': 0.01,        # 1% thermal noise
        'opamp_offset': 0.002,      # 2mV op-amp offset
        'quantization_bits': 8,     # 8-bit DAC/ADC
        'saturation_vmax': 2.5      # 2.5V supply rails
    }
    
    print(f"\nAnalog Hardware Constraints:")
    for key, value in analog_config.items():
        print(f"  {key}: {value}")
    
    # Define search space
    search_space = {
        'hidden_dims': [
            [64],           # Shallow narrow
            [128],          # Shallow wide
            [256],          # Shallow very wide
            [64, 64],       # Deep narrow
            [128, 64],      # Deep tapered
            [64, 128],      # Deep expanding
            [128, 128],     # Deep wide
            [256, 128],     # Deep very wide
        ]
    }
    
    print(f"\nSearch Space: {len(search_space['hidden_dims'])} candidate architectures")
    
    # Run NAS
    nas = AnalogNASSearch(
        input_dim=n_features,
        output_dim=n_classes,
        analog_config=analog_config,
        search_space=search_space
    )
    
    print("\nStarting NAS search...")
    best_arch = nas.search(
        X_train, y_train, X_test, y_test,
        max_candidates=len(search_space['hidden_dims']),
        epochs=30
    )
    
    # Get Pareto front
    pareto_front = nas.get_pareto_front()
    
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")
    
    print(f"\nBest Architecture: {best_arch.hidden_dims}")
    print(f"  Accuracy: {best_arch.accuracy:.4f}")
    print(f"  Analog Robustness: {best_arch.analog_robustness_score:.4f}")
    print(f"  Energy Efficiency: {best_arch.energy_efficiency:.2e} ops/J")
    
    print(f"\nPareto-Optimal Architectures ({len(pareto_front)} found):")
    for i, arch in enumerate(pareto_front, 1):
        print(f"  {i}. {arch.hidden_dims}: "
              f"Acc={arch.accuracy:.4f}, "
              f"Robust={arch.analog_robustness_score:.4f}, "
              f"Energy={arch.energy_efficiency:.2e}")
    
    # Visualization
    print(f"\n{'=' * 80}")
    print("Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Accuracy vs Robustness
    accuracies = [r.accuracy for r in nas.results]
    robustness = [r.analog_robustness_score for r in nas.results]
    
    axes[0, 0].scatter(accuracies, robustness, alpha=0.6, label='Candidates')
    
    # Highlight Pareto front
    pareto_acc = [r.accuracy for r in pareto_front]
    pareto_rob = [r.analog_robustness_score for r in pareto_front]
    axes[0, 0].scatter(pareto_acc, pareto_rob, c='red', s=100, marker='*', 
                       label='Pareto Front', zorder=5)
    
    axes[0, 0].set_xlabel('Clean Accuracy')
    axes[0, 0].set_ylabel('Analog Robustness')
    axes[0, 0].set_title('Accuracy vs Analog Robustness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Energy Efficiency
    energies = [r.energy_efficiency for r in nas.results]
    
    axes[0, 1].scatter(accuracies, energies, alpha=0.6, label='Candidates')
    
    pareto_energy = [r.energy_efficiency for r in pareto_front]
    axes[0, 1].scatter(pareto_acc, pareto_energy, c='red', s=100, marker='*',
                       label='Pareto Front', zorder=5)
    
    axes[0, 1].set_xlabel('Clean Accuracy')
    axes[0, 1].set_ylabel('Energy Efficiency (ops/J)')
    axes[0, 1].set_title('Accuracy vs Energy Efficiency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Architecture depth analysis
    depths = [len(r.hidden_dims) for r in nas.results]
    
    axes[1, 0].scatter(depths, accuracies, alpha=0.6, label='Accuracy')
    axes[1, 0].scatter(depths, robustness, alpha=0.6, label='Robustness')
    axes[1, 0].set_xlabel('Network Depth (layers)')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Depth vs Performance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Architecture width analysis
    widths = [max(r.hidden_dims) for r in nas.results]
    
    axes[1, 1].scatter(widths, accuracies, alpha=0.6, label='Accuracy')
    axes[1, 1].scatter(widths, robustness, alpha=0.6, label='Robustness')
    axes[1, 1].set_xlabel('Max Hidden Dimension')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Width vs Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nas_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved: nas_results.png")
    
    plt.show()
    
    # Comparison with standard architectures
    print(f"\n{'=' * 80}")
    print("Comparison with Standard Architectures")
    print(f"{'=' * 80}")
    
    from experiments.models import DigitalMLP
    from analog_layers.analog_linear import AnalogLinear
    
    standard_archs = {
        'Shallow [64]': [64],
        'Shallow [128]': [128],
        'Deep [64, 64]': [64, 64],
        'Deep [128, 128]': [128, 128],
        f'NAS Best {best_arch.hidden_dims}': best_arch.hidden_dims
    }
    
    for name, hidden_dims in standard_archs.items():
        # Create model
        model = DigitalMLP(
            input_dim=n_features,
            hidden_dims=hidden_dims,
            output_dim=n_classes
        )
        
        # Evaluate clean accuracy
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            predictions = torch.argmax(outputs, dim=1)
            clean_acc = (predictions == y_test).float().mean().item()
        
        # Evaluate robust accuracy (with analog noise)
        robust_accs = []
        for trial in range(5):
            # Create analog version
            analog_model = DigitalMLP(
                input_dim=n_features,
                hidden_dims=hidden_dims,
                output_dim=n_classes,
                analog_config={**analog_config, 'seed': trial}
            )
            
            # Copy weights
            analog_model.load_state_dict(model.state_dict(), strict=False)
            
            analog_model.eval()
            with torch.no_grad():
                outputs = analog_model(X_test)
                predictions = torch.argmax(outputs, dim=1)
                robust_acc = (predictions == y_test).float().mean().item()
                robust_accs.append(robust_acc)
        
        avg_robust = np.mean(robust_accs)
        degradation = (clean_acc - avg_robust) / clean_acc * 100
        
        print(f"\n{name}:")
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        print(f"  Robust Accuracy: {avg_robust:.4f} (±{np.std(robust_accs):.4f})")
        print(f"  Degradation: {degradation:.2f}%")
    
    print(f"\n{'=' * 80}")
    print("NAS Example Complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
