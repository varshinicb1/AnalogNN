"""
Temperature-Aware Analog Computing
===================================

Resistors have non-zero Temperature Coefficients (TCR):
    R(T) = R? ? (1 + ?(T ? T?) + ?(T ? T?)?)
    
Where:
    ? ? ?100 to ?1000 ppm/?C (1st order)
    ? ? ?0.5 ppm/?C? (2nd order, for precision) 

At T = 85?C (typical automotive junction temp):
    R(T) = R? ? (1 + 500e-6 ? 60 + 0.5e-6 ? 3600)
         = R? ? (1 + 0.03 + 0.0018) = 1.0318 ? R? -> 3.18% error

Across an entire crossbar array, temperature gradients create systematic
error patterns that look like spatially-correlated mismatch. Training
with temperature variation produces thermally-robust networks.

Novel Contribution: Temperature-Invariant Representations (TIR)
    Networks trained with temperature as an explicit random variable
    learn features that are invariant to thermal drift. This is a form
    of domain randomization specific to analog hardware.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Callable


# Temperature coefficients for common resistor types (ppm/?C)
RESISTOR_TCR = {
    'standard': {'alpha': 100, 'beta': 0.5},       # Standard thick film
    'precision': {'alpha': 25, 'beta': 0.1},         # Precision thin film
    'ultra_precision': {'alpha': 5, 'beta': 0.02},   # Ultra-precision foil
    'integrated': {'alpha': 800, 'beta': 2.0},        # On-chip polysilicon
    'ideal': {'alpha': 0, 'beta': 0},                  # Ideal (for comparison)
}

# Silicon process temperature ranges (?C)
TEMPERATURE_RANGES = {
    'commercial': (0, 70),
    'industrial': (-40, 85),
    'automotive': (-40, 125),
    'military': (-55, 125),
    'lab': (20, 30),            # Controlled lab environment
    'server': (20, 80),         # Server rack with cooling
    'edge_outdoor': (-20, 60),  # Edge device outdoors
}


def apply_temperature_to_weights(weight: torch.Tensor,
                                 T: float,
                                 T_ref: float = 25.0,
                                 resistor_type: str = 'standard') -> torch.Tensor:
    """
    Simulate temperature effect on resistor values.
    
    R(T) = R? ? (1 + ???T + ???T?)
    w_eff = w ? R? / R(T)  (effective weight at temperature T)
    
    The weight changes because the resistor ratio R_f/R_ij changes with temp.
    If all resistors have the SAME TCR (monolithic integration), the ratio
    is temp-independent. But in practice, TCR mismatch exists:
        w_eff = w ? (1 + ???T + ???T?)^{-1}
    """
    tcr = RESISTOR_TCR[resistor_type]
    delta_T = T - T_ref
    temp_factor = 1.0 + tcr['alpha'] * 1e-6 * delta_T + tcr['beta'] * 1e-6 * delta_T**2
    
    return weight / temp_factor


class TemperatureAnalogLinear(nn.Module):
    """
    Analog linear layer with temperature-aware resistor model.
    
    At each forward pass, the temperature is sampled (uniformly or via
    a profile) and weights are adjusted accordingly.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 T_range: tuple = (20, 80),
                 resistor_type: str = 'standard',
                 enable_mismatch: bool = True,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = True,
                 noise_sigma: float = 0.01):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.T_range = T_range
        self.resistor_type = resistor_type
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        self.enable_noise = enable_noise
        self.noise_sigma = noise_sigma
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.T_ref = 25.0
    
    def get_effective_weight(self, T: Optional[float] = None) -> torch.Tensor:
        """Compute temperature- and mismatch-adjusted effective weight."""
        w = self.weight
        
        if T is None:
            T = np.random.uniform(*self.T_range)
        
        # Temperature effect
        tcr = RESISTOR_TCR[self.resistor_type]
        delta_T = T - self.T_ref
        temp_factor = 1.0 + tcr['alpha'] * 1e-6 * delta_T + tcr['beta'] * 1e-6 * delta_T**2
        w_eff = w / temp_factor
        
        # Mismatch effect (spatially correlated across the array)
        if self.enable_mismatch and self.mismatch_sigma > 0:
            noise = torch.randn_like(w_eff) * self.mismatch_sigma
            w_eff = w_eff / (1.0 + noise)
        
        return w_eff
    
    def forward(self, x: torch.Tensor, T: Optional[float] = None) -> torch.Tensor:
        w_eff = self.get_effective_weight(T)
        
        out = torch.mm(x, w_eff.T)
        
        if self.bias is not None:
            out = out + self.bias
        
        # Thermal noise (Johnson-Nyquist: scales with sqrt(T))
        if self.enable_noise and self.noise_sigma > 0:
            T_use = T if T is not None else np.random.uniform(*self.T_range)
            temp_noise_factor = np.sqrt((T_use + 273.15) / (self.T_ref + 273.15))
            noise = torch.randn_like(out) * self.noise_sigma * temp_noise_factor
            out = out + noise
        
        return out


class TemperatureProfile:
    """
    Generate realistic temperature profiles for training.
    
    Supports:
        - Constant: fixed temperature
        - Sine wave: diurnal cycling
        - Step: sudden temperature changes (e.g., device power-on)
        - Random walk: Brownian temperature drift
        - Realistic: from weather station data simulation
    """
    
    @staticmethod
    def constant(T: float, n: int) -> np.ndarray:
        return np.full(n, T)
    
    @staticmethod
    def sine_wave(T_min: float, T_max: float, n: int, period: int = 24) -> np.ndarray:
        """Diurnal temperature cycle (24-hour default)."""
        t = np.arange(n)
        return T_min + (T_max - T_min) * (0.5 + 0.5 * np.sin(2 * np.pi * t / period))
    
    @staticmethod
    def step(T_cold: float, T_hot: float, n: int, switch_at: float = 0.3) -> np.ndarray:
        """Sudden temperature change (e.g., device power-on)."""
        switch_idx = int(n * switch_at)
        temps = np.full(n, T_cold)
        temps[switch_idx:] = T_hot
        return temps
    
    @staticmethod
    def random_walk(T_mean: float, T_std: float, n: int, sigma: float = 1.0) -> np.ndarray:
        """Brownian temperature drift."""
        walk = np.random.randn(n) * sigma
        return T_mean + T_std * np.cumsum(walk) / np.sqrt(n)
    
    @staticmethod
    def realistic(n: int, seed: int = 42) -> np.ndarray:
        """Realistic temperature profile combining diurnal + random walk."""
        rng = np.random.RandomState(seed)
        diurnal = TemperatureProfile.sine_wave(20, 35, n, period=24)
        walk = TemperatureProfile.random_walk(0, 2, n, sigma=0.5)
        dust = rng.randn(n) * 0.3
        return diurnal + walk + dust


class TemperatureAwareTrainer:
    """
    Trains neural networks with temperature as an explicit random variable.
    
    Key innovation: During training, each batch sees a different temperature.
    This forces the network to learn representations that are invariant to
    temperature-induced weight drift.
    
    This is a form of Domain Randomization (DR) for analog hardware.
    Unlike standard DR which randomizes simulation parameters,
    we randomize the temperature based on realistic physical profiles.
    """
    
    def __init__(self,
                 T_range: tuple = (20, 80),
                 resistor_type: str = 'standard',
                 profile_type: str = 'realistic',
                 lr: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 32):
        self.T_range = T_range
        self.resistor_type = resistor_type
        self.profile_type = profile_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = {'loss': [], 'acc': [], 'T_samples': []}
    
    def _sample_temperature(self, step: int, total_steps: int) -> float:
        """Sample temperature based on profile type."""
        if self.profile_type == 'uniform':
            return np.random.uniform(*self.T_range)
        elif self.profile_type == 'realistic':
            profile = TemperatureProfile.realistic(total_steps)
            return profile[min(step, len(profile) - 1)]
        elif self.profile_type == 'sine':
            profile = TemperatureProfile.sine_wave(self.T_range[0], self.T_range[1], total_steps, period=24)
            return profile[min(step, len(profile) - 1)]
        elif self.profile_type == 'step':
            profile = TemperatureProfile.step(self.T_range[0], self.T_range[1], total_steps)
            return profile[min(step, len(profile) - 1)]
        else:
            return np.random.uniform(*self.T_range)
    
    def _apply_temperature_to_model(self, model, T):
        """Apply temperature drift to model weights IN PLACE."""
        tcr = RESISTOR_TCR[self.resistor_type]
        delta_T = T - 25.0
        temp_factor = 1.0 + tcr['alpha'] * 1e-6 * delta_T + tcr['beta'] * 1e-6 * delta_T**2
        
        originals = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                originals[name] = param.data.clone()
                param.data.div_(temp_factor)
        return originals
    
    def _restore_model_weights(self, model, originals):
        for name, param in model.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])
    
    def train(self, model: nn.Module, 
              X_train, y_train, X_test, y_test,
              analog_config: Optional[Dict] = None,
              callback: Optional[Callable] = None):
        from torch.utils.data import TensorDataset, DataLoader
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_steps = self.epochs * len(loader)
        global_step = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_x, batch_y in loader:
                T = self._sample_temperature(global_step, total_steps)
                self.history['T_samples'].append(T)
                
                # Apply temperature to model weights IN PLACE, then forward
                originals = self._apply_temperature_to_model(model, T)
                out = model(batch_x)
                self._restore_model_weights(model, originals)
                
                loss = criterion(out, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                global_step += 1
            
            with torch.no_grad():
                out = model(X_test)
                acc = (out.argmax(1) == y_test).float().mean().item()
            
            self.history['loss'].append(epoch_loss / len(loader))
            self.history['acc'].append(acc)
            
            if callback:
                callback(epoch, self.history)
        
        return self.history
    
    def evaluate_at_temperature(self, model, X_test, y_test,
                                T_test: float = 25.0) -> float:
        """Evaluate model at a specific temperature through analog hardware."""
        from experiments.models import DigitalMLP
        
        nf = X_test.shape[1]
        nc = y_test.max().item() + 1
        analog_cfg = {'resistor_mismatch': 0.01, 'noise_sigma': 0.01,
                      'opamp_offset': 0.002, 'quantization_bits': 8,
                      'saturation_vmax': 2.5, 'temperature': T_test,
                      'resistor_type': self.resistor_type}
        analog_model = DigitalMLP(nf, [128, 64], nc, analog_config=analog_cfg)
        analog_model.load_state_dict(model.state_dict(), strict=False)
        
        with torch.no_grad():
            out = analog_model(X_test)
            acc = (out.argmax(1) == y_test).float().mean().item()
        
        return acc
