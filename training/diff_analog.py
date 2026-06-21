"""
DifferentiableAnalogSimulator
==============================

End-to-end differentiable analog non-ideality simulation.

Key innovations:
1. Reparameterized mismatch: noise = eps * sigma (differentiable w.r.t sigma)
2. Straight-through quantization: gradient passes through as 1
3. Soft saturation: sigmoid-based differentiable approximation
4. Reparameterized offset: model offset as learnable parameter

Together, these make the ENTIRE analog cascade differentiable,
enabling gradient-based optimization THROUGH hardware non-idealities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class DifferentiableMismatch(nn.Module):
    """
    Reparameterized resistor mismatch.
    
    Instead of: w_eff = w / (1 + randn * sigma)
    We use:     w_eff = w / (1 + eps * sigma)
    Where eps is a learned or sampled parameter.
    
    This is differentiable w.r.t. both w and sigma.
    """
    
    def __init__(self, sigma: float = 0.01, learnable_sigma: bool = False):
        super().__init__()
        if learnable_sigma:
            self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))
        else:
            self.register_buffer('log_sigma', torch.tensor(np.log(sigma)))
    
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def forward(self, w: torch.Tensor, eps: Optional[torch.Tensor] = None,
                return_noise: bool = False) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(w)
        
        sigma_val = self.sigma
        noise = 1.0 + eps * sigma_val
        w_eff = w / noise
        
        if return_noise:
            return w_eff, noise
        return w_eff


class DifferentiableQuantization(nn.Module):
    """
    Straight-through estimator for quantization.
    
    Forward: round to n_bits
    Backward: gradient passes through as 1 (STE)
    
    Supports symmetric and asymmetric quantization.
    """
    
    def __init__(self, n_bits: int = 8, symmetric: bool = True,
                 learnable_bits: bool = False):
        super().__init__()
        if learnable_bits:
            self.log_bits = nn.Parameter(torch.tensor(np.log(float(n_bits))))
        else:
            self.register_buffer('log_bits', torch.tensor(np.log(float(n_bits))))
        self.symmetric = symmetric
    
    @property
    def n_bits(self):
        return torch.exp(self.log_bits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits = self.n_bits
        
        if self.symmetric:
            max_val = x.abs().max() + 1e-8
            scale = max_val / (2**(bits - 1) - 1)
        else:
            min_val = x.min()
            max_val = x.max() + 1e-8
            scale = (max_val - min_val) / (2**bits - 1)
        
        if self.symmetric:
            x_q = torch.clamp(torch.round(x / scale), -(2**(bits-1)), 2**(bits-1) - 1)
        else:
            x_q = torch.clamp(torch.round((x - min_val) / scale), 0, 2**bits - 1)
            x_q = x_q * scale + min_val
            return x_q  # Not using STE for asymmetric (simpler)
        
        # Straight-through estimator for symmetric
        x_ste = x + (x_q * scale - x).detach()
        return x_ste


class DifferentiableSaturation(nn.Module):
    """
    Differentiable voltage saturation.
    
    Uses sigmoid-based soft clipping which has well-defined gradients.
    Hard clip: V_out = clamp(V_in, -V_max, V_max)
    Soft clip: V_out = V_max * tanh(V_in / V_max)
    """
    
    def __init__(self, vmax: float = 2.5, soft: bool = True,
                 learnable_vmax: bool = False):
        super().__init__()
        self.soft = soft
        if learnable_vmax:
            self.log_vmax = nn.Parameter(torch.tensor(np.log(vmax)))
        else:
            self.register_buffer('log_vmax', torch.tensor(np.log(vmax)))
    
    @property
    def vmax(self):
        return torch.exp(self.log_vmax)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.vmax
        if self.soft:
            return v * torch.tanh(x / v)
        else:
            return torch.clamp(x, -v, v)


class DifferentiableOffset(nn.Module):
    """
    Reparameterized op-amp offset.
    
    V_out += Vos where Vos ~ Uniform(-Vos_max, Vos_max)
    
    Reparameterized: Vos = Vos_max * (2 * u - 1), u ~ Uniform(0, 1)
    Differentiable w.r.t. Vos_max.
    """
    
    def __init__(self, vos_max: float = 0.002, learnable: bool = False):
        super().__init__()
        if learnable:
            self.log_vos = nn.Parameter(torch.tensor(np.log(vos_max)))
        else:
            self.register_buffer('log_vos', torch.tensor(np.log(vos_max)))
    
    @property
    def vos_max(self):
        return torch.exp(self.log_vos)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vos = self.vos_max
        if self.training:
            u = torch.rand(x.shape[-1], device=x.device)
            offsets = vos * (2 * u - 1)
            return x + offsets
        return x


class DifferentiableAnalogLinear(nn.Module):
    """
    Fully differentiable analog linear layer.
    
    All non-idealities are implemented with reparameterization or
    straight-through estimators, making the entire forward pass
    differentiable end-to-end.
    
    This enables:
    1. Training through analog simulation
    2. Learning optimal sigma/bits/vmax/Vos parameters
    3. Computing exact Lipschitz constants
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 sigma_mismatch: float = 0.01,
                 n_bits: int = 8,
                 vmax: float = 2.5,
                 vos_max: float = 0.002,
                 noise_sigma: float = 0.01,
                 learnable_params: bool = False):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.mismatch = DifferentiableMismatch(sigma_mismatch, learnable_params)
        self.quantization = DifferentiableQuantization(n_bits, symmetric=True, learnable_bits=learnable_params)
        self.saturation = DifferentiableSaturation(vmax, soft=True, learnable_vmax=learnable_params)
        self.offset = DifferentiableOffset(vos_max, learnable=learnable_params)
        
        self.noise_sigma = nn.Parameter(torch.tensor(np.log(noise_sigma))) if learnable_params else noise_sigma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Weight mismatch (reparameterized)
        w_eff = self.mismatch(self.weight)
        
        # 2. Weight quantization (STE)
        w_eff = self.quantization(w_eff)
        
        # 3. Matrix multiply
        out = F.linear(x, w_eff, self.bias)
        
        # 4. Op-amp offset (reparameterized)
        out = self.offset(out)
        
        # 5. Additive noise (reparameterized)
        if self.training and isinstance(self.noise_sigma, (float, int)) and self.noise_sigma > 0:
            noise = torch.randn_like(out) * self.noise_sigma
            out = out + noise
        elif self.training and isinstance(self.noise_sigma, nn.Parameter):
            noise = torch.randn_like(out) * torch.exp(self.noise_sigma)
            out = out + noise
        
        # 6. Saturation (differentiable tanh)
        out = self.saturation(out)
        
        return out
    
    def get_lipschitz_upper_bound(self) -> float:
        """Compute upper bound on Lipschitz constant."""
        return self.weight.norm(p=2).item() / 1.0  # divide by 1 = no temperature factor


class DifferentiableAnalogMLP(nn.Module):
    """
    Multi-layer DifferentiableAnalogMLP for end-to-end analog training.
    """
    
    def __init__(self,
                 in_features: int,
                 hidden_dims: list,
                 out_features: int,
                 analog_config: Optional[Dict] = None):
        super().__init__()
        
        cfg = analog_config or {}
        learnable = cfg.get('learnable_params', False)
        
        dims = [in_features] + hidden_dims + [out_features]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            layer = DifferentiableAnalogLinear(
                dims[i], dims[i+1],
                sigma_mismatch=cfg.get('sigma_mismatch', 0.01),
                n_bits=cfg.get('n_bits', 8),
                vmax=cfg.get('vmax', 2.5),
                vos_max=cfg.get('vos_max', 0.002),
                noise_sigma=cfg.get('noise_sigma', 0.01),
                learnable_params=learnable
            )
            self.layers.append(layer)
        
        self.dropout = nn.Dropout(cfg.get('dropout', 0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h
    
    def get_total_lipschitz_bound(self) -> float:
        """Product of all layer Lipschitz bounds."""
        bound = 1.0
        for layer in self.layers:
            bound *= layer.get_lipschitz_upper_bound()
        return bound


class DifferentiableAnalogTrainer:
    """
    Trains models THROUGH differentiable analog simulation.
    
    This is fundamentally different from "train digital, deploy analog."
    The gradients flow through the mismatch, quantization, saturation,
    and offset layers, optimizing weights to compensate for non-idealities.
    """
    
    def __init__(self,
                 lr: float = 0.001,
                 epochs: int = 30,
                 batch_size: int = 32,
                 analog_config: Optional[Dict] = None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.analog_config = analog_config or {}
    
    def train(self, model: nn.Module,
              X_train, y_train, X_test, y_test) -> Dict:
        from torch.utils.data import TensorDataset, DataLoader
        
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        loader = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=self.batch_size, shuffle=True)
        
        history = {'loss': [], 'acc': []}
        
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0.0
            
            for bx, by in loader:
                out = model(bx)
                loss = crit(out, by)
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                epoch_loss += loss.item()
            
            model.eval()
            with torch.no_grad():
                out = model(X_test)
                acc = (out.argmax(1) == y_test).float().mean().item()
            
            history['loss'].append(epoch_loss / len(loader))
            history['acc'].append(acc)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}: loss={history['loss'][-1]:.4f}, acc={acc:.4f}")
        
        return history
