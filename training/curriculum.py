"""
Curriculum Learning for Analog Robustness
===========================================

Gradually increases analog non-ideality exposure during training,
following the principle that networks learn better when difficulty
increases progressively.

Strategies:
- Mismatch curriculum: 0% -> target_mismatch over epochs
- Spectral curriculum: 0 -> target_weight over epochs  
- Combined: both curricula simultaneously
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional, Callable
from torch.utils.data import TensorDataset, DataLoader
from training.spectral_regularizer import SpectralRegularizer


class CurriculumScheduler:
    """
    Schedules regularization parameters over training epochs.
    
    Supports linear, exponential, and cosine schedules.
    """
    
    def __init__(self, epochs: int, schedule_type: str = 'cosine'):
        self.epochs = epochs
        self.schedule_type = schedule_type
    
    def get_weight(self, epoch: int, start: float = 0.0,
                   end: float = 1.0, warmup: float = 0.1) -> float:
        """Get regularization weight for current epoch."""
        if epoch / self.epochs < warmup:
            return start
        progress = (epoch / self.epochs - warmup) / (1 - warmup)
        progress = min(max(progress, 0.0), 1.0)
        
        if self.schedule_type == 'linear':
            return start + (end - start) * progress
        elif self.schedule_type == 'exponential':
            return start + (end - start) * (1 - math.exp(-3 * progress))
        elif self.schedule_type == 'cosine':
            return start + (end - start) * (1 - math.cos(math.pi * progress / 2))
        else:  # step
            return end if progress > 0.5 else start


class CurriculumRobustTrainer:
    """
    Trains networks with curriculum learning for analog robustness.
    
    Gradually increases:
    - Spectral regularization strength
    - Mismatch level (for recycling)
    - Adversarial perturbation budget
    
    This mimics how biological systems develop robustness through
    progressive exposure to environmental variability.
    """
    
    def __init__(self,
                 lr: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32,
                 final_spectral_weight: float = 0.05,
                 final_mismatch: float = 0.02,
                 schedule_type: str = 'cosine',
                 orthogonal_init: bool = True):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.final_spectral_weight = final_spectral_weight
        self.final_mismatch = final_mismatch
        self.scheduler = CurriculumScheduler(epochs, schedule_type)
        self.orthogonal_init = orthogonal_init
        self.history = {
            'train_loss': [], 'test_acc': [], 'analog_acc': [],
            'spectral_weight': [], 'mismatch': [], 'kappa': []
        }
    
    def train(self, model: nn.Module, X_train, y_train, X_test, y_test,
              analog_config: Optional[Dict] = None,
              callback: Optional[Callable] = None):
        
        if self.orthogonal_init and hasattr(model, 'network'):
            for layer in model.network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            # Curriculum schedule
            spectral_weight = self.scheduler.get_weight(
                epoch, end=self.final_spectral_weight)
            mismatch = self.scheduler.get_weight(
                epoch, end=self.final_mismatch)
            
            # Create epoch-specific analog config
            epoch_analog_config = None
            if analog_config is not None and mismatch > 0:
                epoch_analog_config = dict(analog_config)
                epoch_analog_config['resistor_mismatch'] = mismatch
            
            model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                if spectral_weight > 0:
                    kappa_loss = SpectralRegularizer.condition_number_loss(model)
                    balance_loss = SpectralRegularizer.spectral_balance_loss(model)
                    loss += spectral_weight * (kappa_loss + balance_loss)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                outputs = model(X_test)
                acc = (torch.argmax(outputs, dim=1) == y_test).float().mean().item()
            
            if epoch_analog_config is not None:
                analog_model = self._copy_to_analog(model, epoch_analog_config, X_test.shape[1], y_test.max().item()+1)
                analog_model.eval()
                with torch.no_grad():
                    a_out = analog_model(X_test)
                    a_acc = (torch.argmax(a_out, dim=1) == y_test).float().mean().item()
            else:
                a_acc = 0.0
            
            metrics = SpectralRegularizer.compute_spectral_metrics(model)
            self.history['train_loss'].append(epoch_loss / len(loader))
            self.history['test_acc'].append(acc)
            self.history['analog_acc'].append(a_acc)
            self.history['spectral_weight'].append(spectral_weight)
            self.history['mismatch'].append(mismatch)
            self.history['kappa'].append(metrics['kappa'])
            
            if callback:
                callback(epoch, self.history)
        
        return self.history
    
    def _copy_to_analog(self, model, analog_config, n_features, n_classes):
        from experiments.models import DigitalMLP
        hidden_dims = []
        for layer in model.network:
            if isinstance(layer, nn.Linear) and layer is not model.network[-1]:
                hidden_dims.append(layer.out_features)
        analog = DigitalMLP(n_features, hidden_dims, n_classes, analog_config=analog_config)
        analog.load_state_dict(model.state_dict(), strict=False)
        return analog
