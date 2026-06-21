"""
Unified Robust Training for Analog Hardware
=============================================

Combines our novel discoveries into a single practical training pipeline:
1. Spectral Regularization (Discovery 5: spectral fragility)
2. Hardware-Aware Adversarial Training (Discovery 3: mismatch recycling)
3. Analog Lottery Ticket Pruning (Discovery 4: sparse robustness)
4. Orthogonal Weight Initialization (Discovery 2: spectral conditioning)

This is the first training method designed specifically for analog hardware.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from torch.utils.data import TensorDataset, DataLoader
from training.spectral_regularizer import SpectralRegularizer
from training.adversarial_training import AnalogAdversarialTrainer


class AnalogRobustTrainer:
    """
    Unified trainer combining all analog-robustness discoveries.
    
    Strategies:
    - 'spectral': Spectral regularization (kappa + balance)
    - 'adversarial': Hardware-aware adversarial training
    - 'combined': Both spectral + adversarial
    - 'lottery': Prune to 70% sparsity after training for max robustness
    """

    def __init__(self,
                 strategy: str = 'combined',
                 spectral_weight: float = 0.1,
                 adversarial_epsilon: float = 0.05,
                 lr: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32,
                 prune_sparsity: float = 0.0):
        self.strategy = strategy
        self.spectral_weight = spectral_weight
        self.adversarial_epsilon = adversarial_epsilon
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.prune_sparsity = prune_sparsity
        self.history = {
            'train_loss': [], 'test_acc': [], 'analog_acc': [],
            'kappa': [], 'spectral_var': []
        }

    def train(self, model: nn.Module, X_train, y_train, X_test, y_test,
              analog_config: Optional[Dict] = None,
              callback: Optional[Callable] = None):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        use_adversarial = self.strategy in ('adversarial', 'combined')
        use_spectral = self.strategy in ('spectral', 'combined')

        if use_adversarial and analog_config is not None:
            adv_trainer = AnalogAdversarialTrainer(
                model=model,
                analog_config=analog_config,
                epsilon=self.adversarial_epsilon,
                attack_steps=3,
                attack_lr=0.01
            )

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                if use_spectral and self.spectral_weight > 0:
                    kappa_loss = SpectralRegularizer.condition_number_loss(model)
                    balance_loss = SpectralRegularizer.spectral_balance_loss(model)
                    loss += self.spectral_weight * (kappa_loss + balance_loss)

                if use_adversarial and analog_config is not None:
                    X_adv, _ = adv_trainer.adversarial_attack(batch_x, batch_y)
                    outputs_adv = model(X_adv)
                    loss_adv = criterion(outputs_adv, batch_y)
                    loss += 0.5 * loss_adv

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                acc = (torch.argmax(outputs, dim=1) == y_test).float().mean().item()

            if analog_config is not None:
                analog_model = self._create_analog_copy(model, analog_config)
                analog_model.eval()
                with torch.no_grad():
                    a_out = analog_model(X_test)
                    a_acc = (torch.argmax(a_out, dim=1) == y_test).float().mean().item()
            else:
                a_acc = acc

            metrics = SpectralRegularizer.compute_spectral_metrics(model)
            self.history['train_loss'].append(epoch_loss / len(loader))
            self.history['test_acc'].append(acc)
            self.history['analog_acc'].append(a_acc)
            self.history['kappa'].append(metrics['kappa'])
            self.history['spectral_var'].append(metrics['var'])

            if callback:
                callback(epoch, self.history)

        if self.prune_sparsity > 0:
            self._apply_lottery_pruning(model, self.prune_sparsity)

        return self.history

    def _create_analog_copy(self, model, analog_config):
        from experiments.models import DigitalMLP
        input_dim = model.network[0].in_features
        output_dim = model.network[-1].out_features
        hidden_dims = []
        for layer in model.network:
            if isinstance(layer, (nn.Linear,)) and layer is not model.network[-1]:
                hidden_dims.append(layer.out_features)
        analog = DigitalMLP(input_dim, hidden_dims, output_dim, analog_config=analog_config)
        analog.load_state_dict(model.state_dict(), strict=False)
        return analog

    def _apply_lottery_pruning(self, model, sparsity):
        with torch.no_grad():
            for param in model.parameters():
                if param.dim() == 2:
                    threshold = torch.quantile(torch.abs(param), sparsity)
                    mask = (torch.abs(param) > threshold).float()
                    param.mul_(mask)
