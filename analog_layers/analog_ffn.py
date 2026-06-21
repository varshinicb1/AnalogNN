"""
Analog Feed-Forward Network (FFN)
==================================

Implements analog circuit versions of transformer feed-forward blocks.
This includes:
- Two linear layers (using differential summing-subtractor)
- GeLU activation (analog approximation using piecewise-linear circuits)
- Dropout (analog stochastic switching)

Theoretical Basis:
- FFN(x) = GeLU(Linear1(x)) -> Linear2
- GeLU(x) = x * Φ(x) where Φ is the CDF of standard normal
- Analog GeLU approximated using piecewise-linear or tanh-based circuits
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional
from .analog_linear import AnalogLinear


class AnalogGeLU(nn.Module):
    """
    Analog GeLU activation approximation.
    
    GeLU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal.
    
    Analog implementations:
    1. Piecewise-linear approximation using op-amp comparators
    2. Tanh-based approximation: GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
    3. Current-mode implementation using differential pairs
    
    This module implements the tanh-based approximation which is
    circuit-friendly (tanh can be implemented with differential pairs).
    """
    
    def __init__(self,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        super().__init__()
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        self.enable_noise = enable_noise
        self.noise_sigma = noise_sigma
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analog GeLU approximation.
        
        Uses the tanh-based approximation which is circuit-friendly:
        GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))
        
        Args:
            x: Input tensor
        
        Returns:
            GeLU output with analog non-idealities
        """
        # Tanh-based GeLU approximation
        # This is more circuit-friendly than the exact formula
        c = math.sqrt(2 / math.pi)
        gelu_approx = 0.5 * x * (1 + torch.tanh(c * (x + 0.044715 * x**3)))
        
        # Add mismatch if enabled
        if self.enable_mismatch and self.training:
            delta = torch.randn_like(gelu_approx) * self.mismatch_sigma
            gelu_approx = gelu_approx * (1 + delta)
        
        # Add noise if enabled
        if self.enable_noise and self.training:
            noise = torch.randn_like(gelu_approx) * self.noise_sigma
            gelu_approx = gelu_approx + noise
        
        return gelu_approx


class AnalogFeedForward(nn.Module):
    """
    Analog feed-forward network for transformer blocks.
    
    Architecture:
    1. Linear layer 1 (expand dimension by 4x)
    2. GeLU activation (analog approximation)
    3. Linear layer 2 (project back to original dimension)
    4. Dropout (analog stochastic switching)
    
    This is the standard transformer FFN adapted for analog hardware.
    """
    
    def __init__(self,
                 embed_dim: int,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog feed-forward network.
        
        Args:
            embed_dim: Embedding dimension
            ff_dim: Feed-forward dimension (default: 4 * embed_dim)
            dropout: Dropout rate
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        self.ff_dim = ff_dim or 4 * embed_dim
        
        # Linear layer 1: expand
        self.linear1 = AnalogLinear(
            in_features=embed_dim,
            out_features=self.ff_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Analog GeLU activation
        self.activation = AnalogGeLU(
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Linear layer 2: project back
        self.linear2 = AnalogLinear(
            in_features=self.ff_dim,
            out_features=embed_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of analog feed-forward network.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
        
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        # Linear layer 1
        x = self.linear1(x)
        
        # GeLU activation
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Linear layer 2
        x = self.linear2(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x


class AnalogDropout(nn.Module):
    """
    Analog dropout using stochastic switching.
    
    In analog hardware, dropout can be implemented using:
    1. Random switching transistors (stochastic behavior)
    2. Current mirrors with random enable signals
    3. Sample-and-hold circuits with random timing
    
    This module simulates the analog dropout behavior.
    """
    
    def __init__(self, p: float = 0.1):
        """
        Initialize analog dropout.
        
        Args:
            p: Dropout probability
        """
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply analog dropout.
        
        Args:
            x: Input tensor
        
        Returns:
            Output with dropout applied
        """
        if not self.training:
            return x
        
        # Analog dropout: random switching
        mask = torch.rand_like(x) > self.p
        return x * mask.float() / (1 - self.p)
