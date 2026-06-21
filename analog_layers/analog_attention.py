"""
Analog Attention Mechanism
==========================

Implements analog circuit versions of transformer attention blocks.
This includes:
- Q, K, V projections (using differential summing-subtractor)
- Attention score computation (analog multiplication/division)
- Softmax approximation (current-mode exponential circuits)
- Weighted sum (analog multiplication)

Theoretical Basis:
- Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
- Analog implementation uses current-mode circuits for multiplication
- Softmax approximated using Gilbert cell or translinear loops
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from .analog_linear import AnalogLinear


class AnalogMultiplication(nn.Module):
    """
    Analog multiplication using Gilbert cell topology.
    
    Multiplies two analog signals using current-mode circuits.
    V_out = k * V1 * V2 where k is a scaling factor.
    
    This is the core operation for attention score computation.
    """
    
    def __init__(self, 
                 scaling_factor: float = 1.0,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Analog multiplication: out = k * x * y
        
        Args:
            x: First input tensor
            y: Second input tensor
        
        Returns:
            Product with analog non-idealities
        """
        # Ideal multiplication
        out = self.scaling_factor * x * y
        
        # Add mismatch if enabled
        if self.enable_mismatch and self.training:
            delta = torch.randn_like(out) * self.mismatch_sigma
            out = out * (1 + delta)
        
        return out


class AnalogSoftmax(nn.Module):
    """
    Analog softmax approximation using translinear loops.
    
    Softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    
    Analog implementation uses:
    - Exponential function via transistor I-V characteristics
    - Current division for normalization
    - Gilbert cell for ratio computation
    
    This is a critical component for attention mechanisms.
    """
    
    def __init__(self,
                 temperature: float = 1.0,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.enable_mismatch = enable_mismatch
        self.mismatch_sigma = mismatch_sigma
        
    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Analog softmax approximation.
        
        Args:
            x: Input tensor (attention scores)
            dim: Dimension to apply softmax over
        
        Returns:
            Softmax output with analog non-idealities
        """
        # Scale by temperature
        x_scaled = x / self.temperature
        
        # Numerical stability (max subtraction)
        x_max = torch.max(x_scaled, dim=dim, keepdim=True)[0]
        x_stable = x_scaled - x_max
        
        # Exponential (analog: transistor exponential I-V)
        exp_x = torch.exp(x_stable)
        
        # Normalization (analog: current division)
        sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
        softmax = exp_x / (sum_exp + 1e-8)
        
        # Add mismatch if enabled
        if self.enable_mismatch and self.training:
            delta = torch.randn_like(softmax) * self.mismatch_sigma
            softmax = softmax * (1 + delta)
        
        return softmax


class AnalogAttention(nn.Module):
    """
    Complete analog attention mechanism.
    
    Architecture:
    1. Q, K, V projections (AnalogLinear with differential summing)
    2. Attention scores: QK^T (AnalogMultiplication)
    3. Scaling: / sqrt(d_k) (Analog division)
    4. Softmax (AnalogSoftmax)
    5. Weighted sum: softmax * V (AnalogMultiplication)
    
    This enables transformer-style attention on analog hardware.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Q, K, V projections using analog linear layers
        self.q_proj = AnalogLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        self.k_proj = AnalogLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        self.v_proj = AnalogLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Output projection
        self.out_proj = AnalogLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Analog multiplication for attention scores
        self.attention_mult = AnalogMultiplication(
            scaling_factor=1.0 / np.sqrt(self.head_dim),
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma
        )
        
        # Analog softmax
        self.softmax = AnalogSoftmax(
            temperature=1.0,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma
        )
        
        # Analog multiplication for weighted sum
        self.weighted_sum_mult = AnalogMultiplication(
            scaling_factor=1.0,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of analog attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            output: Attention output [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V projections
        Q = self.q_proj(x)  # [batch_size, seq_len, embed_dim]
        K = self.k_proj(x)  # [batch_size, seq_len, embed_dim]
        V = self.v_proj(x)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: QK^T
        # In analog: multiply Q and K^T using Gilbert cells
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Apply scaling (built into attention_mult in analog)
        # Here we apply it numerically for simulation
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Analog softmax
        attention_weights = self.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum: attention_weights * V
        # In analog: multiply using Gilbert cells
        output = torch.matmul(attention_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, attention_weights


class AnalogMultiHeadAttention(nn.Module):
    """
    Multi-head attention with analog circuit implementation.
    
    This is the standard transformer attention block adapted for analog hardware.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        super().__init__()
        self.attention = AnalogAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention(x, mask)
