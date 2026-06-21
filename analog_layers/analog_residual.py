"""
Analog Residual Connections
===========================

Implements analog circuit versions of residual connections.
This includes:
- Residual addition (analog summing amplifier)
- Residual scaling (analog multiplication)
- Residual projection (when dimensions don't match)

Theoretical Basis:
- Residual(x, F(x)) = x + F(x)
- Analog implementation uses summing amplifiers
- When dimensions don't match, use projection: W_proj * x + F(x)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .analog_linear import AnalogLinear


class AnalogResidualConnection(nn.Module):
    """
    Analog residual connection (addition).
    
    Residual(x, F(x)) = x + F(x)
    
    Analog implementation:
    - Summing amplifier (op-amp with multiple inputs)
    - When x and F(x) are voltages, simple voltage adder
    - When dimensions don't match, use projection layer first
    
    This is the core of ResNet and transformer architectures.
    """
    
    def __init__(self,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog residual connection.
        
        Args:
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        self.enable_noise = enable_noise
        self.noise_sigma = noise_sigma
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Analog residual addition.
        
        Args:
            x: Input tensor (skip connection)
            residual: Residual tensor (from network)
        
        Returns:
            Sum: x + residual with analog non-idealities
        """
        # Analog addition (summing amplifier)
        output = x + residual
        
        # Add mismatch if enabled
        if self.enable_mismatch and self.training:
            delta = torch.randn_like(output) * self.mismatch_sigma
            output = output * (1 + delta)
        
        # Add noise if enabled
        if self.enable_noise and self.training:
            noise = torch.randn_like(output) * self.noise_sigma
            output = output + noise
        
        return output


class AnalogResidualProjection(nn.Module):
    """
    Analog residual connection with projection.
    
    When input and output dimensions don't match, we need to project:
    output = W_proj * x + F(x)
    
    Analog implementation:
    - Projection layer (analog linear layer)
    - Summing amplifier for addition
    
    This is used in transformers when changing embedding dimensions.
    """
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog residual projection.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        # Projection layer (analog linear)
        self.projection = AnalogLinear(
            in_features=in_features,
            out_features=out_features,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Residual addition
        self.residual = AnalogResidualConnection(
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Analog residual projection.
        
        Args:
            x: Input tensor (skip connection)
            residual: Residual tensor (from network)
        
        Returns:
            Sum: W_proj * x + residual with analog non-idealities
        """
        # Project x if dimensions don't match
        if x.shape[-1] != residual.shape[-1]:
            x_proj = self.projection(x)
        else:
            x_proj = x
        
        # Add residual
        output = self.residual(x_proj, residual)
        
        return output


class AnalogGatedResidual(nn.Module):
    """
    Gated residual connection with learnable gate.
    
    output = gate * x + (1 - gate) * residual
    
    This allows the network to learn how much to use the residual.
    Analog implementation uses variable gain amplifiers.
    """
    
    def __init__(self,
                 features: int,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize gated residual connection.
        
        Args:
            features: Feature dimension
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        # Learnable gate parameter (analog variable resistor)
        self.gate = nn.Parameter(torch.ones(features))
        
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        self.enable_noise = enable_noise
        self.noise_sigma = noise_sigma
        
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Gated residual connection.
        
        Args:
            x: Input tensor (skip connection)
            residual: Residual tensor (from network)
        
        Returns:
            Gated sum: gate * x + (1 - gate) * residual
        """
        # Apply gate (analog variable gain amplifier)
        if self.enable_mismatch and self.training:
            delta = torch.randn_like(self.gate) * self.mismatch_sigma
            gate = self.gate * (1 + delta)
        else:
            gate = self.gate
        
        # Gated combination
        output = gate * x + (1 - gate) * residual
        
        # Add noise if enabled
        if self.enable_noise and self.training:
            noise = torch.randn_like(output) * self.noise_sigma
            output = output + noise
        
        return output
