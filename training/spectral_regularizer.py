"""
Spectral Regularization for Analog-Robust Training
===================================================

Integrates spectral regularization methods into the training pipeline.
Based on Discovery 5: High condition number networks are spectrally fragile.

Strategies:
- Kappa penalty: penalizes high condition number
- Balance penalty: penalizes variance in log singular values
- Norm constraint: penalizes large spectral norm
- Combined: all three with tuned weights
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from torch.utils.data import TensorDataset, DataLoader


class SpectralRegularizer:
    """Spectral regularization methods for analog-robust training."""

    @staticmethod
    def condition_number_loss(model, target_kappa=5.0):
        loss = 0.0
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                W = module.weight
                with torch.no_grad():
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    kappa = S[0] / (S[-1] + 1e-8)
                if kappa > target_kappa:
                    loss += (kappa - target_kappa) ** 2
                    count += 1
        return loss / max(count, 1)

    @staticmethod
    def spectral_balance_loss(model):
        loss = 0.0
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                W = module.weight
                with torch.no_grad():
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    log_S = torch.log(S + 1e-8)
                    loss += torch.var(log_S)
                    count += 1
        return loss / max(count, 1)

    @staticmethod
    def spectral_norm_constraint(model, max_norm=2.0):
        loss = 0.0
        count = 0
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                W = module.weight
                with torch.no_grad():
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    sigma_max = S[0]
                if sigma_max > max_norm:
                    loss += (sigma_max - max_norm) ** 2
                    count += 1
        return loss / max(count, 1)

    @staticmethod
    def compute_spectral_metrics(model):
        metrics = {'kappa': [], 'var': [], 'max_sv': []}
        for module in model.modules():
            if isinstance(module, nn.Linear) and module.weight.dim() >= 2:
                W = module.weight
                with torch.no_grad():
                    S = torch.linalg.svd(W, full_matrices=False)[1]
                    metrics['kappa'].append((S[0] / (S[-1] + 1e-8)).item())
                    metrics['var'].append(torch.var(torch.log(S + 1e-8)).item())
                    metrics['max_sv'].append(S[0].item())
        return {k: (sum(v) / len(v)) if v else 0.0 for k, v in metrics.items()}


class SpectralTrainer:
    """Training wrapper that applies spectral regularization."""

    STRATEGIES = {
        'none': {'kappa': 0.0, 'balance': 0.0, 'norm': 0.0},
        'kappa': {'kappa': 0.1, 'balance': 0.0, 'norm': 0.0},
        'balance': {'kappa': 0.0, 'balance': 0.1, 'norm': 0.0},
        'norm': {'kappa': 0.0, 'balance': 0.0, 'norm': 0.1},
        'combined': {'kappa': 0.05, 'balance': 0.05, 'norm': 0.05},
        'kappa_strong': {'kappa': 0.25, 'balance': 0.0, 'norm': 0.0},
        'balance_strong': {'kappa': 0.0, 'balance': 0.25, 'norm': 0.0},
    }

    def __init__(self, strategy: str = 'kappa', lr: float = 0.001,
                 epochs: int = 50, batch_size: int = 32):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(self.STRATEGIES.keys())}")
        self.strategy = strategy
        self.reg_weights = self.STRATEGIES[strategy]
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = {'train_loss': [], 'test_acc': [], 'kappa': [], 'spectral_var': []}

    def train(self, model: nn.Module, X_train, y_train, X_test, y_test):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                if self.reg_weights['kappa'] > 0:
                    loss += self.reg_weights['kappa'] * SpectralRegularizer.condition_number_loss(model)
                if self.reg_weights['balance'] > 0:
                    loss += self.reg_weights['balance'] * SpectralRegularizer.spectral_balance_loss(model)
                if self.reg_weights['norm'] > 0:
                    loss += self.reg_weights['norm'] * SpectralRegularizer.spectral_norm_constraint(model)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                acc = (torch.argmax(outputs, dim=1) == y_test).float().mean().item()

            metrics = SpectralRegularizer.compute_spectral_metrics(model)
            self.history['train_loss'].append(epoch_loss / len(loader))
            self.history['test_acc'].append(acc)
            self.history['kappa'].append(metrics['kappa'])
            self.history['spectral_var'].append(metrics['var'])

        return self.history
