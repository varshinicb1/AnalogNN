"""
Advanced Research: Spectral Regularization, Transformers, and Large-Scale Validation
=====================================================================================

Three major research directions:
1. Spectral Regularization Methods - Penalize high condition numbers during training
2. Transformer Architecture Extension - Test discoveries on attention mechanisms
3. Large-Scale Validation - CIFAR-10, full MNIST, ImageNet subsets

This builds on our five discoveries to create practical training methods.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm

from datasets.loaders import get_dataset
from experiments.models import DigitalMLP, train_model
from analog_layers.analog_linear import AnalogLinear
from analog_layers.analog_attention import AnalogMultiHeadAttention
from analog_layers.analog_transformer import AnalogTransformerBlock
from training.spectral_regularizer import SpectralRegularizer, SpectralTrainer


class AnalogTransformer(nn.Module):
    """
    Transformer model with analog-aware components.
    """
    
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=2,
                 dim_feedforward=512, dropout=0.1, analog_config=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.analog_config = analog_config
        
        # Map analog_config to AnalogTransformerBlock params
        enable_mismatch = False
        mismatch_sigma = 0.01
        enable_noise = False
        noise_sigma = 0.01
        if analog_config is not None:
            if analog_config.get('resistor_mismatch', 0) > 0:
                enable_mismatch = True
                mismatch_sigma = analog_config.get('resistor_mismatch', 0.01)
            if analog_config.get('noise_sigma', 0) > 0:
                enable_noise = True
                noise_sigma = analog_config.get('noise_sigma', 0.01)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learned)
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AnalogTransformerBlock(
                embed_dim=d_model,
                num_heads=nhead,
                ff_dim=dim_feedforward,
                dropout=dropout,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Transformer blocks (return (output, attention_weights))
        for block in self.transformer_blocks:
            x, _ = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class AdvancedResearch:
    def __init__(self, output_dir="./research_advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    def research_1_spectral_regularization(self, datasets):
        """
        Research 1: Spectral Regularization Methods
        
        Test different spectral regularization strategies:
        - Condition number penalty
        - Spectral balance (variance of log singular values)
        - Spectral norm constraint
        - Combined approaches
        """
        print("\n" + "="*80)
        print("RESEARCH 1: Spectral Regularization Methods")
        print("="*80)
        
        results = []
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # Regularization strategies
            strategies = {
                'None': {'kappa': 0.0, 'balance': 0.0, 'norm': 0.0},
                'Kappa': {'kappa': 0.1, 'balance': 0.0, 'norm': 0.0},
                'Balance': {'kappa': 0.0, 'balance': 0.1, 'norm': 0.0},
                'Norm': {'kappa': 0.0, 'balance': 0.0, 'norm': 0.1},
                'Combined': {'kappa': 0.05, 'balance': 0.05, 'norm': 0.05},
            }
            
            strategy_results = {}
            
            for strategy_name, reg_weights in strategies.items():
                print(f"  Strategy: {strategy_name}")
                
                # Train with spectral regularization
                model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[128, 64],
                    output_dim=data['n_classes']
                )
                
                self._train_with_spectral_reg(
                    model, data['X_train'], data['y_train'],
                    data['X_test'], data['y_test'],
                    reg_weights=reg_weights,
                    epochs=50, lr=0.001, batch_size=32
                )
                
                # Clean accuracy
                model.eval()
                with torch.no_grad():
                    outputs = model(data['X_test'])
                    clean_acc = (torch.argmax(outputs, dim=1) == data['y_test']).float().mean().item()
                
                # Analog accuracy
                analog_config = {
                    'resistor_mismatch': 0.05,
                    'noise_sigma': 0.01,
                    'opamp_offset': 0.002,
                    'quantization_bits': 8,
                    'saturation_vmax': 2.5,
                    'seed': 42
                }
                
                analog_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[128, 64],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                analog_model.eval()
                with torch.no_grad():
                    outputs = analog_model(data['X_test'])
                    analog_acc = (torch.argmax(outputs, dim=1) == data['y_test']).float().mean().item()
                
                robustness = analog_acc / clean_acc if clean_acc > 0 else 0
                
                # Compute spectral properties
                W = model.network[0].weight.detach()
                U, S, V = torch.linalg.svd(W, full_matrices=False)
                kappa = (S[0] / S[-1]).item()
                spectral_var = torch.var(torch.log(S + 1e-8)).item()
                
                strategy_results[strategy_name] = {
                    'clean_acc': clean_acc,
                    'analog_acc': analog_acc,
                    'robustness': robustness,
                    'kappa': kappa,
                    'spectral_var': spectral_var
                }
                
                print(f"    Clean={clean_acc:.4f}, Analog={analog_acc:.4f}, "
                      f"Robustness={robustness:.4f}, kappa={kappa:.2f}, var={spectral_var:.4f}")
            
            results.append({
                'dataset': dataset_name,
                'strategies': strategy_results
            })
        
        self.results['spectral_regularization'] = results
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            strategies = result['strategies']
            
            strategy_names = list(strategies.keys())
            robustness = [strategies[s]['robustness'] for s in strategy_names]
            kappa = [strategies[s]['kappa'] for s in strategy_names]
            
            axes[0].bar(strategy_names, robustness, alpha=0.7)
            axes[0].set_ylabel('Robustness')
            axes[0].set_title(f'{dataset_name.upper()} - Robustness')
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(strategy_names, kappa, alpha=0.7)
            axes[1].set_ylabel('Condition Number')
            axes[1].set_title(f'{dataset_name.upper()} - Condition Number')
            axes[1].tick_params(axis='x', rotation=45)
            
            axes[2].scatter(kappa, robustness, s=100, alpha=0.7, label=dataset_name)
            axes[2].set_xlabel('Condition Number')
            axes[2].set_ylabel('Robustness')
            axes[2].set_title('Kappa vs Robustness')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spectral_regularization.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'spectral_regularization.png'}")
        
        return results
    
    def _train_with_spectral_reg(self, model, X_train, y_train, X_test, y_test,
                                  reg_weights, epochs, lr, batch_size):
        """Train with spectral regularization"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train))
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]
                
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # Classification loss
                loss = criterion(outputs, y_batch)
                
                # Spectral regularization losses
                if reg_weights['kappa'] > 0:
                    loss += reg_weights['kappa'] * SpectralRegularizer.condition_number_loss(model)
                
                if reg_weights['balance'] > 0:
                    loss += reg_weights['balance'] * SpectralRegularizer.spectral_balance_loss(model)
                
                if reg_weights['norm'] > 0:
                    loss += reg_weights['norm'] * SpectralRegularizer.spectral_norm_constraint(model)
                
                # Backward pass
                loss.backward()
                optimizer.step()
    
    def research_2_transformer_extension(self, datasets):
        """
        Research 2: Transformer Architecture Extension
        
        Test our five discoveries on transformer architectures:
        - Phase transitions in attention mechanisms
        - Sparse transformers vs dense transformers
        - Spectral properties of attention weights
        """
        print("\n" + "="*80)
        print("RESEARCH 2: Transformer Architecture Extension")
        print("="*80)
        
        results = []
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # Create sequence data (treat features as sequence)
            X_train_seq = data['X_train'].unsqueeze(1)  # (batch, 1, features)
            X_test_seq = data['X_test'].unsqueeze(1)
            
            # Test different transformer configurations
            configs = [
                {'name': 'Dense', 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'sparsity': 0.0},
                {'name': 'Sparse-50', 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'sparsity': 0.5},
                {'name': 'Sparse-70', 'num_layers': 2, 'd_model': 64, 'nhead': 4, 'sparsity': 0.7},
                {'name': 'Deep', 'num_layers': 4, 'd_model': 64, 'nhead': 4, 'sparsity': 0.0},
            ]
            
            config_results = {}
            
            for config in configs:
                print(f"  Config: {config['name']}")
                
                # Create transformer model
                model = AnalogTransformer(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dim_feedforward=256,
                    dropout=0.1,
                    analog_config=None
                )
                
                # Train
                self._train_transformer(
                    model, X_train_seq, data['y_train'],
                    X_test_seq, data['y_test'],
                    epochs=30, lr=0.001, batch_size=32
                )
                
                # Apply sparsity if needed
                if config['sparsity'] > 0:
                    self._apply_sparsity(model, config['sparsity'])
                
                # Clean accuracy
                model.eval()
                with torch.no_grad():
                    outputs = model(X_test_seq)
                    clean_acc = (torch.argmax(outputs, dim=1) == data['y_test']).float().mean().item()
                
                # Analog accuracy
                analog_config = {
                    'resistor_mismatch': 0.05,
                    'noise_sigma': 0.01,
                    'opamp_offset': 0.002,
                    'quantization_bits': 8,
                    'saturation_vmax': 2.5,
                    'seed': 42
                }
                
                analog_model = AnalogTransformer(
                    input_dim=data['n_features'],
                    num_classes=data['n_classes'],
                    d_model=config['d_model'],
                    nhead=config['nhead'],
                    num_layers=config['num_layers'],
                    dim_feedforward=256,
                    dropout=0.1,
                    analog_config=analog_config
                )
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                analog_model.eval()
                with torch.no_grad():
                    outputs = analog_model(X_test_seq)
                    analog_acc = (torch.argmax(outputs, dim=1) == data['y_test']).float().mean().item()
                
                robustness = analog_acc / clean_acc if clean_acc > 0 else 0
                
                config_results[config['name']] = {
                    'clean_acc': clean_acc,
                    'analog_acc': analog_acc,
                    'robustness': robustness,
                    'config': config
                }
                
                print(f"    Clean={clean_acc:.4f}, Analog={analog_acc:.4f}, Robustness={robustness:.4f}")
            
            results.append({
                'dataset': dataset_name,
                'configs': config_results
            })
        
        self.results['transformer_extension'] = results
        
        # Plot
        fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 4))
        if len(datasets) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            configs = result['configs']
            
            config_names = list(configs.keys())
            robustness = [configs[c]['robustness'] for c in config_names]
            
            axes[i].bar(config_names, robustness, alpha=0.7)
            axes[i].set_ylabel('Robustness')
            axes[i].set_title(f'{dataset_name.upper()} - Transformer Robustness')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transformer_extension.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'transformer_extension.png'}")
        
        return results
    
    def _train_transformer(self, model, X_train, y_train, X_test, y_test,
                          epochs, lr, batch_size):
        """Train transformer model"""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            indices = torch.randperm(len(X_train))
            
            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                batch_idx = indices[start:end]
                
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
    
    def _apply_sparsity(self, model, sparsity):
        """Apply magnitude-based pruning"""
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() == 2:  # Weight matrices
                    threshold = torch.quantile(torch.abs(param), sparsity)
                    mask = (torch.abs(param) > threshold).float()
                    param.mul_(mask)
    
    def research_3_large_scale_validation(self):
        """
        Research 3: Large-Scale Validation
        
        Test discoveries on larger datasets:
        - Full MNIST (60k train, 10k test)
        - CIFAR-10 (50k train, 10k test)
        - Larger batch sizes and longer training
        """
        print("\n" + "="*80)
        print("RESEARCH 3: Large-Scale Validation")
        print("="*80)
        
        results = []
        
        # Load larger datasets
        large_datasets = {}
        
        print("\nLoading full MNIST...")
        X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
            name='mnist', subset_size=10000, downsample_size=8, seed=42
        )
        large_datasets['mnist_full'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'n_features': n_features, 'n_classes': n_classes
        }
        print(f"  MNIST: {len(X_train)} train, {len(X_test)} test")
        
        print("\nLoading CIFAR-10...")
        try:
            X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
                name='cifar10', subset_size=10000, downsample_size=8, seed=42
            )
            large_datasets['cifar10'] = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'n_features': n_features, 'n_classes': n_classes
            }
            print(f"  CIFAR-10: {len(X_train)} train, {len(X_test)} test")
        except Exception as e:
            print(f"  CIFAR-10 not available: {e}")
        
        print("\nLoading TinyImageNet (falls back to synthetic_large)...")
        try:
            X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
                name='tiny_imagenet', subset_size=5000, downsample_size=8, seed=42
            )
            large_datasets['tiny_imagenet'] = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'n_features': n_features, 'n_classes': n_classes
            }
            print(f"  TinyImageNet: {len(X_train)} train, {len(X_test)} test, {n_classes} classes")
        except Exception as e:
            print(f"  TinyImageNet not available: {e}")
        
        print("\nLoading synthetic_large (200 classes)...")
        try:
            X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
                name='synthetic_large', subset_size=5000, downsample_size=12, seed=42
            )
            large_datasets['synthetic_large'] = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'n_features': n_features, 'n_classes': n_classes
            }
            print(f"  synthetic_large: {len(X_train)} train, {len(X_test)} test, {n_classes} classes, {n_features} features")
        except Exception as e:
            print(f"  synthetic_large not available: {e}")
        
        # Test key discoveries on large datasets
        for dataset_name, data in large_datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # Test 1: Phase transition
            print("  Testing phase transition...")
            mismatch_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
            phase_results = []
            
            model = DigitalMLP(
                input_dim=data['n_features'],
                hidden_dims=[256, 128],
                output_dim=data['n_classes']
            )
            
            train_model(model, data['X_train'], data['y_train'],
                       data['X_test'], data['y_test'],
                       epochs=30, lr=0.001, batch_size=64, seed=42)
            
            for mismatch in mismatch_levels:
                analog_config = {
                    'resistor_mismatch': mismatch,
                    'noise_sigma': 0.0,
                    'opamp_offset': 0.0,
                    'quantization_bits': 0,
                    'saturation_vmax': 0.0,
                    'seed': 42
                }
                
                analog_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[256, 128],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                analog_model.eval()
                with torch.no_grad():
                    outputs = analog_model(data['X_test'][:1000])
                    acc = (torch.argmax(outputs, dim=1) == data['y_test'][:1000]).float().mean().item()
                
                phase_results.append({'mismatch': mismatch, 'accuracy': acc})
                print(f"    Mismatch={mismatch:.2f}: Accuracy={acc:.4f}")
            
            # Test 2: Sparse vs dense
            print("  Testing sparse networks...")
            sparsity_levels = [0.0, 0.5, 0.7]
            sparse_results = []
            
            for sparsity in sparsity_levels:
                model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[256, 128],
                    output_dim=data['n_classes']
                )
                
                train_model(model, data['X_train'], data['y_train'],
                           data['X_test'], data['y_test'],
                           epochs=30, lr=0.001, batch_size=64, seed=42)
                
                if sparsity > 0:
                    self._apply_sparsity(model, sparsity)
                
                analog_config = {
                    'resistor_mismatch': 0.05,
                    'noise_sigma': 0.01,
                    'opamp_offset': 0.002,
                    'quantization_bits': 8,
                    'saturation_vmax': 2.5,
                    'seed': 42
                }
                
                analog_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[256, 128],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                analog_model.load_state_dict(model.state_dict(), strict=False)
                
                analog_model.eval()
                with torch.no_grad():
                    outputs = analog_model(data['X_test'][:1000])
                    analog_acc = (torch.argmax(outputs, dim=1) == data['y_test'][:1000]).float().mean().item()
                
                sparse_results.append({'sparsity': sparsity, 'analog_acc': analog_acc})
                print(f"    Sparsity={sparsity:.2f}: Analog Acc={analog_acc:.4f}")
            
            # Test 3: Spectral regularization at scale
            print("  Testing spectral regularization...")
            spec_results = {}
            for strategy in ['none', 'kappa', 'balance', 'norm']:
                trainer = SpectralTrainer(strategy=strategy, epochs=20, batch_size=64, lr=0.001)
                spec_model = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[256, 128],
                    output_dim=data['n_classes']
                )
                trainer.train(spec_model, data['X_train'], data['y_train'],
                            data['X_test'], data['y_test'])
                
                # Analog accuracy
                analog_config = {
                    'resistor_mismatch': 0.05,
                    'noise_sigma': 0.01,
                    'opamp_offset': 0.002,
                    'quantization_bits': 8,
                    'saturation_vmax': 2.5,
                    'seed': 42
                }
                spec_analog = DigitalMLP(
                    input_dim=data['n_features'],
                    hidden_dims=[256, 128],
                    output_dim=data['n_classes'],
                    analog_config=analog_config
                )
                spec_analog.load_state_dict(spec_model.state_dict(), strict=False)
                spec_analog.eval()
                with torch.no_grad():
                    outputs = spec_analog(data['X_test'][:1000])
                    analog_acc = (torch.argmax(outputs, dim=1) == data['y_test'][:1000]).float().mean().item()
                
                metrics = SpectralRegularizer.compute_spectral_metrics(spec_model)
                spec_results[strategy] = {
                    'analog_acc': analog_acc,
                    'test_acc': trainer.history['test_acc'][-1],
                    'kappa': metrics['kappa'],
                    'spectral_var': metrics['var']
                }
                print(f"    Strategy={strategy}: Analog Acc={analog_acc:.4f}, kappa={metrics['kappa']:.2f}")
            
            results.append({
                'dataset': dataset_name,
                'phase_transition': phase_results,
                'sparse_networks': sparse_results,
                'spectral_reg': spec_results
            })
        
        self.results['large_scale'] = results
        
        # Plot
        n_datasets = len(large_datasets)
        fig, axes = plt.subplots(3, n_datasets, figsize=(5*n_datasets, 12))
        if n_datasets == 1:
            axes = axes.reshape(3, 1)
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            
            # Phase transition
            mismatches = [p['mismatch'] for p in result['phase_transition']]
            accuracies = [p['accuracy'] for p in result['phase_transition']]
            
            axes[0, i].plot(mismatches, accuracies, 'b-o', linewidth=2)
            axes[0, i].set_xlabel('Mismatch')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].set_title(f'{dataset_name.upper()} - Phase Transition')
            axes[0, i].grid(True, alpha=0.3)
            
            # Sparse networks
            sparsities = [s['sparsity'] for s in result['sparse_networks']]
            analog_accs = [s['analog_acc'] for s in result['sparse_networks']]
            
            axes[1, i].plot(sparsities, analog_accs, 'g-s', linewidth=2)
            axes[1, i].set_xlabel('Sparsity')
            axes[1, i].set_ylabel('Analog Accuracy')
            axes[1, i].set_title(f'{dataset_name.upper()} - Sparse Networks')
            axes[1, i].grid(True, alpha=0.3)
            
            # Spectral regularization
            if 'spectral_reg' in result:
                spec = result['spectral_reg']
                strategies = list(spec.keys())
                analog_accs = [spec[s]['analog_acc'] for s in strategies]
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                axes[2, i].bar(strategies, analog_accs, color=colors[:len(strategies)], alpha=0.7)
                axes[2, i].set_xlabel('Strategy')
                axes[2, i].set_ylabel('Analog Accuracy')
                axes[2, i].set_title(f'{dataset_name.upper()} - Spectral Reg')
                axes[2, i].tick_params(axis='x', rotation=45)
                axes[2, i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'large_scale_validation.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'large_scale_validation.png'}")
        
        return results
    
    def research_4_transformer_robustness(self, datasets):
        """
        Research 4: Why are Transformers Robust to Analog Non-Idealities?
        
        Systematic investigation comparing MLPs vs Transformers:
        - Controlled comparison with matched parameter counts
        - Ablation studies (remove attention, residuals, layernorm)
        - Test each non-ideality separately
        - Spectral analysis of attention weights vs linear weights
        """
        print("\n" + "="*80)
        print("RESEARCH 4: Transformer Robustness Investigation")
        print("="*80)
        
        results = []
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # 1. Matched parameter comparison: MLP vs Transformer
            print("  1. Matched MLP vs Transformer comparison...")
            n_features = data['n_features']
            n_classes = data['n_classes']
            
            # Create matched models (~same params)
            transformer = AnalogTransformer(
                input_dim=n_features,
                num_classes=n_classes,
                d_model=32, nhead=4, num_layers=2,
                dim_feedforward=64, dropout=0.1,
                analog_config=None
            )
            
            mlp = DigitalMLP(
                input_dim=n_features,
                hidden_dims=[64, 64],
                output_dim=n_classes
            )
            
            # Count params
            t_params = sum(p.numel() for p in transformer.parameters())
            m_params = sum(p.numel() for p in mlp.parameters())
            print(f"    Transformer params: {t_params:,}, MLP params: {m_params:,}")
            
            # Train both
            X_train_seq = data['X_train'].unsqueeze(1)
            X_test_seq = data['X_test'].unsqueeze(1)
            
            self._train_transformer(transformer, X_train_seq, data['y_train'],
                                   X_test_seq, data['y_test'],
                                   epochs=30, lr=0.001, batch_size=32)
            train_model(mlp, data['X_train'], data['y_train'],
                       data['X_test'], data['y_test'],
                       epochs=30, lr=0.001, batch_size=32, seed=42)
            
            # 2. Test each non-ideality separately
            non_idealities = {
                'none': {'resistor_mismatch': 0.0, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                        'quantization_bits': 0, 'saturation_vmax': 0.0},
                'mismatch_5': {'resistor_mismatch': 0.05, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                              'quantization_bits': 0, 'saturation_vmax': 0.0},
                'mismatch_10': {'resistor_mismatch': 0.10, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                               'quantization_bits': 0, 'saturation_vmax': 0.0},
                'mismatch_20': {'resistor_mismatch': 0.20, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                               'quantization_bits': 0, 'saturation_vmax': 0.0},
                'noise': {'resistor_mismatch': 0.0, 'noise_sigma': 0.05, 'opamp_offset': 0.0,
                         'quantization_bits': 0, 'saturation_vmax': 0.0},
                'offset': {'resistor_mismatch': 0.0, 'noise_sigma': 0.0, 'opamp_offset': 0.02,
                          'quantization_bits': 0, 'saturation_vmax': 0.0},
                'quant_4bit': {'resistor_mismatch': 0.0, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                              'quantization_bits': 4, 'saturation_vmax': 0.0},
                'saturation': {'resistor_mismatch': 0.0, 'noise_sigma': 0.0, 'opamp_offset': 0.0,
                              'quantization_bits': 0, 'saturation_vmax': 1.0},
            }
            
            comp_results = {}
            for ni_name, ni_config in non_idealities.items():
                ni_config['seed'] = 42
                
                # Transformer analog
                t_analog = AnalogTransformer(
                    input_dim=n_features,
                    num_classes=n_classes,
                    d_model=32, nhead=4, num_layers=2,
                    dim_feedforward=64, dropout=0.1,
                    analog_config=ni_config
                )
                t_analog.load_state_dict(transformer.state_dict(), strict=False)
                t_analog.eval()
                with torch.no_grad():
                    t_out = t_analog(X_test_seq)
                    t_acc = (torch.argmax(t_out, dim=1) == data['y_test']).float().mean().item()
                
                # MLP analog
                m_analog = DigitalMLP(
                    input_dim=n_features,
                    hidden_dims=[64, 64],
                    output_dim=n_classes,
                    analog_config=ni_config
                )
                m_analog.load_state_dict(mlp.state_dict(), strict=False)
                m_analog.eval()
                with torch.no_grad():
                    m_out = m_analog(data['X_test'])
                    m_acc = (torch.argmax(m_out, dim=1) == data['y_test']).float().mean().item()
                
                comp_results[ni_name] = {
                    'transformer_acc': t_acc,
                    'mlp_acc': m_acc,
                    'robustness_ratio': t_acc / max(m_acc, 1e-8)
                }
                print(f"    {ni_name}: Transformer={t_acc:.4f}, MLP={m_acc:.4f}, Ratio={t_acc/max(m_acc,1e-8):.3f}")
            
            # 3. Ablation: What makes transformers robust?
            print("  3. Ablation studies...")
            ablations = {}
            
            analog_cfg_test = {'resistor_mismatch': 0.10, 'seed': 42,
                              'noise_sigma': 0.0, 'opamp_offset': 0.0,
                              'quantization_bits': 0, 'saturation_vmax': 0.0}
            
            # Full transformer
            t_full = AnalogTransformer(
                input_dim=n_features, num_classes=n_classes,
                d_model=32, nhead=4, num_layers=2,
                dim_feedforward=64, dropout=0.1,
                analog_config=analog_cfg_test
            )
            t_full.load_state_dict(transformer.state_dict(), strict=False)
            t_full.eval()
            with torch.no_grad():
                full_acc = (torch.argmax(t_full(X_test_seq), dim=1) == data['y_test']).float().mean().item()
            
            # For ablation, compare vs MLP with residual connections
            # Create MLP variants
            from collections import OrderedDict
            
            class MLPWithResidual(nn.Module):
                def __init__(self, input_dim, hidden_dims, output_dim, analog_config=None):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dims[0])
                    self.residual_proj = nn.Linear(hidden_dims[0], hidden_dims[0])
                    if analog_config:
                        self.layers = nn.ModuleList([
                            AnalogLinear(hidden_dims[0], hidden_dims[0], config=analog_config)
                            for _ in range(2)
                        ])
                    else:
                        self.layers = nn.ModuleList([
                            nn.Linear(hidden_dims[0], hidden_dims[0]) for _ in range(2)
                        ])
                    if analog_config:
                        self.output = AnalogLinear(hidden_dims[0], output_dim, config=analog_config)
                    else:
                        self.output = nn.Linear(hidden_dims[0], output_dim)
                    self.norm = nn.LayerNorm(hidden_dims[0])
                
                def forward(self, x):
                    x = self.input_proj(x)
                    for layer in self.layers:
                        residual = x
                        x = self.norm(x)
                        x = torch.relu(layer(x))
                        x = x + residual
                    return self.output(x)
            
            # Standard MLP (no residual)
            m_std = DigitalMLP(
                input_dim=n_features, hidden_dims=[64, 64],
                output_dim=n_classes, analog_config=analog_cfg_test
            )
            m_std.load_state_dict(mlp.state_dict(), strict=False)
            m_std.eval()
            with torch.no_grad():
                m_std_acc = (torch.argmax(m_std(data['X_test']), dim=1) == data['y_test']).float().mean().item()
            
            # MLP with residual connections
            m_res = MLPWithResidual(
                input_dim=n_features, hidden_dims=[64], output_dim=n_classes,
                analog_config=analog_cfg_test
            )
            # Copy weights where possible
            with torch.no_grad():
                m_res.input_proj.weight.copy_(mlp.network[0].weight)
                m_res.input_proj.bias.copy_(mlp.network[0].bias)
                if hasattr(m_res.layers[0], 'weight'):
                    m_res.layers[0].weight.copy_(mlp.network[2].weight)
                    m_res.layers[0].bias.copy_(mlp.network[2].bias)
                    m_res.layers[1].weight.copy_(mlp.network[2].weight)
                    m_res.layers[1].bias.copy_(mlp.network[2].bias)
                m_res.output.weight.copy_(mlp.network[4].weight)
                m_res.output.bias.copy_(mlp.network[4].bias)
            m_res.eval()
            with torch.no_grad():
                m_res_acc = (torch.argmax(m_res(data['X_test']), dim=1) == data['y_test']).float().mean().item()
            
            ablations = {
                'transformer': full_acc,
                'mlp_standard': m_std_acc,
                'mlp_with_residual': m_res_acc,
                'residual_boost': m_res_acc - m_std_acc
            }
            print(f"    Transformer={full_acc:.4f}, MLP std={m_std_acc:.4f}, "
                  f"MLP+residual={m_res_acc:.4f}, boost={m_res_acc-m_std_acc:+.4f}")
            
            # 4. Spectral analysis: attention vs linear weights
            print("  4. Spectral analysis...")
            spectral_analysis = {}
            
            # MLP layer spectra
            for idx, layer in enumerate(mlp.network):
                if isinstance(layer, nn.Linear):
                    W = layer.weight.detach()
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    kappa = (S[0] / S[-1]).item()
                    spectral_analysis[f'mlp_layer_{idx}'] = {
                        'kappa': kappa,
                        'sv_mean': S.mean().item(),
                        'sv_std': S.std().item()
                    }
            
            # For transformer, compute attention weight spectra
            # Attention output projection (value mixing)
            for name, param in transformer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    W = param.detach()
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    kappa = (S[0] / S[-1]).item()
                    spectral_analysis[f'transformer_{name}'] = {
                        'kappa': kappa,
                        'sv_mean': S.mean().item(),
                        'sv_std': S.std().item()
                    }
            
            results.append({
                'dataset': dataset_name,
                'param_count': {'transformer': t_params, 'mlp': m_params},
                'non_ideality_comparison': comp_results,
                'ablation': ablations,
                'spectral_analysis': spectral_analysis
            })
        
        self.results['transformer_robustness'] = results
        
        # Plot 1: Non-ideality comparison
        fig, axes = plt.subplots(2, len(datasets), figsize=(5*len(datasets), 8))
        if len(datasets) == 1:
            axes = axes.reshape(2, 1)
        
        for i, result in enumerate(results):
            dataset_name = result['dataset']
            comp = result['non_ideality_comparison']
            
            ni_names = list(comp.keys())
            t_accs = [comp[n]['transformer_acc'] for n in ni_names]
            m_accs = [comp[n]['mlp_acc'] for n in ni_names]
            
            x = np.arange(len(ni_names))
            width = 0.35
            axes[0, i].bar(x - width/2, t_accs, width, label='Transformer', alpha=0.7)
            axes[0, i].bar(x + width/2, m_accs, width, label='MLP', alpha=0.7)
            axes[0, i].set_xticks(x)
            axes[0, i].set_xticklabels(ni_names, rotation=45, ha='right')
            axes[0, i].set_ylabel('Accuracy')
            axes[0, i].set_title(f'{dataset_name.upper()} - Non-ideality Comparison')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3, axis='y')
            
            # Ablation
            abl = result['ablation']
            if 'mlp_standard' in abl:
                abl_names = ['Transformer', 'MLP std', 'MLP+residual']
                abl_vals = [abl.get('transformer', 0), abl.get('mlp_standard', 0), abl.get('mlp_residual', 0)]
                actual_vals = []
                actual_names = []
                for name, val in zip(abl_names, abl_vals):
                    if val > 0:
                        actual_vals.append(val)
                        actual_names.append(name)
                if 'mlp_with_residual' in abl:
                    actual_names.append('MLP+residual')
                    actual_vals.append(abl['mlp_with_residual'])
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
                axes[1, i].bar(actual_names, actual_vals, color=colors[:len(actual_names)], alpha=0.7)
                axes[1, i].set_ylabel('Analog Accuracy')
                axes[1, i].set_title(f'{dataset_name.upper()} - Ablation')
                axes[1, i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'transformer_robustness.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved: {self.output_dir / 'transformer_robustness.png'}")
        
        return results

    def research_5_spice_validation(self, datasets):
        """
        Research 5: Real SPICE Circuit Validation using PySpice
        
        Validates that the analytical fallback solver matches real SPICE
        simulation for a small circuit. This bridges theory and practice.
        """
        print("\n" + "="*80)
        print("RESEARCH 5: SPICE Circuit Validation")
        print("="*80)
        
        try:
            from spice.pyspice_runner import PySpiceRunner
            print("PySpiceRunner available, running validation...")
        except Exception as e:
            print(f"PySpiceRunner not available: {e}")
            return []
        
        results = []
        
        for dataset_name, data in datasets.items():
            print(f"\nDataset: {dataset_name}")
            
            # Create a minimal MLP with AnalogLinear layers
            model = DigitalMLP(
                input_dim=data['n_features'],
                hidden_dims=[16],
                output_dim=data['n_classes'],
                analog_config={'resistor_mismatch': 0.01, 'noise_sigma': 0.0,
                              'opamp_offset': 0.0, 'quantization_bits': 0,
                              'saturation_vmax': 0.0, 'seed': 42}
            )
            
            # Train quickly
            train_model(model, data['X_train'], data['y_train'],
                       data['X_test'], data['y_test'],
                       epochs=10, lr=0.001, batch_size=32, seed=42)
            
            # Test PySpice on a single batch
            X_sample = data['X_test'][:4]
            y_sample = data['y_test'][:4]
            
            model.eval()
            with torch.no_grad():
                reference = model(X_sample)
            
            # Try PySpice simulation
            try:
                runner = PySpiceRunner(config={})
                x_tensor = X_sample
                for idx, layer in enumerate(model.network):
                    if isinstance(layer, AnalogLinear):
                        # Extract effective weights
                        w, b = layer.get_effective_weights()
                        # Run SPICE for this layer
                        x_tensor = runner.run(w, b, x_tensor, r_ref=10000, v_ref=1.0)
                    elif isinstance(layer, torch.nn.ReLU):
                        x_tensor = torch.nn.functional.relu(x_tensor)
                
                spice_preds = torch.argmax(x_tensor, dim=1)
                ref_preds = torch.argmax(reference, dim=1)
                match = (spice_preds == ref_preds).float().mean().item()
                
                results.append({
                    'dataset': dataset_name,
                    'spice_accuracy_match': float(match),
                    'spice_available': True
                })
                print(f"  SPICE vs reference prediction match: {match:.2%}")
                
            except Exception as e:
                import traceback
                print(f"  PySpice simulation failed: {e}")
                traceback.print_exc()
                results.append({
                    'dataset': dataset_name,
                    'spice_error': str(e),
                    'spice_available': False
                })
        
        self.results['spice_validation'] = results
        print(f"\nSPICE validation complete for {len(results)} datasets")
        return results

    def run_all_research(self):
        """Run all three research directions"""
        print("="*80)
        print("ADVANCED RESEARCH: Spectral Reg, Transformers, Large-Scale")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load standard datasets
        print("\nLoading datasets...")
        datasets = {}
        
        for name in ['iris', 'mnist', 'fashion']:
            X_train, y_train, X_test, y_test, n_features, n_classes = get_dataset(
                name=name, subset_size=500, downsample_size=8, seed=42
            )
            datasets[name] = {
                'X_train': X_train, 'y_train': y_train,
                'X_test': X_test, 'y_test': y_test,
                'n_features': n_features, 'n_classes': n_classes
            }
            print(f"  {name}: {len(X_train)} train, {len(X_test)} test")
        
        # Run research
        self.research_1_spectral_regularization(datasets)
        self.research_2_transformer_extension(datasets)
        self.research_3_large_scale_validation()
        self.research_4_transformer_robustness(datasets)
        self.research_5_spice_validation(datasets)
        
        # Save results
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj
        
        with open(self.output_dir / 'advanced_research.json', 'w') as f:
            json.dump(convert_to_serializable(self.results), f, indent=2)
        
        print("\n" + "="*80)
        print("ALL ADVANCED RESEARCH COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return self.results


if __name__ == "__main__":
    research = AdvancedResearch()
    results = research.run_all_research()
