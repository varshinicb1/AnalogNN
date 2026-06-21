"""
Analog Transformer Block
=======================

Implements complete analog transformer block combining:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections

This is the building block for analog LLMs.

Theoretical Basis:
- Transformer Block = LayerNorm(x + Attention(x)) + LayerNorm(x + FFN(x))
- Analog implementation uses the component blocks designed separately
- Enables full transformer architectures on analog hardware
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .analog_attention import AnalogMultiHeadAttention
from .analog_ffn import AnalogFeedForward
from .analog_norm import AnalogLayerNorm, AnalogRMSNorm
from .analog_residual import AnalogResidualConnection, AnalogResidualProjection


class AnalogTransformerBlock(nn.Module):
    """
    Complete analog transformer block.
    
    Architecture:
    1. LayerNorm
    2. Multi-head self-attention
    3. Residual connection
    4. LayerNorm
    5. Feed-forward network
    6. Residual connection
    
    This is the standard transformer block adapted for analog hardware.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = "layer",  # "layer" or "rms"
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension (default: 4 * embed_dim)
            dropout: Dropout rate
            norm_type: Normalization type ("layer" or "rms")
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # First LayerNorm
        if norm_type == "layer":
            self.norm1 = AnalogLayerNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        else:
            self.norm1 = AnalogRMSNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        
        # Multi-head attention
        self.attention = AnalogMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Residual connection 1
        self.residual1 = AnalogResidualConnection(
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Second LayerNorm
        if norm_type == "layer":
            self.norm2 = AnalogLayerNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        else:
            self.norm2 = AnalogRMSNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        
        # Feed-forward network
        self.ffn = AnalogFeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        # Residual connection 2
        self.residual2 = AnalogResidualConnection(
            enable_mismatch=enable_mismatch,
            mismatch_sigma=mismatch_sigma,
            enable_noise=enable_noise,
            noise_sigma=noise_sigma
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of analog transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
        
        Returns:
            output: Output tensor [batch_size, seq_len, embed_dim]
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        # Save input for residual
        residual = x
        
        # LayerNorm
        x = self.norm1(x)
        
        # Self-attention
        attn_output, attention_weights = self.attention(x, mask)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        x = self.residual1(residual, attn_output)
        
        # Save for second residual
        residual = x
        
        # LayerNorm
        x = self.norm2(x)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output)
        
        # Residual connection
        x = self.residual2(residual, ffn_output)
        
        return x, attention_weights


class AnalogTransformerEncoder(nn.Module):
    """
    Analog transformer encoder (stack of transformer blocks).
    
    This is the encoder portion of the transformer architecture,
    used for tasks like sequence classification, encoding, etc.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = "layer",
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog transformer encoder.
        
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            norm_type: Normalization type
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            AnalogTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                norm_type=norm_type,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        if norm_type == "layer":
            self.final_norm = AnalogLayerNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        else:
            self.final_norm = AnalogRMSNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        
    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass of analog transformer encoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
        
        Returns:
            output: Output tensor [batch_size, seq_len, embed_dim]
            attention_weights_list: List of attention weights from each layer
        """
        attention_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights_list.append(attn_weights)
        
        x = self.final_norm(x)
        
        return x, attention_weights_list


class AnalogTransformerDecoder(nn.Module):
    """
    Analog transformer decoder (stack of transformer blocks with cross-attention).
    
    This is the decoder portion of the transformer architecture,
    used for autoregressive generation tasks.
    """
    
    def __init__(self,
                 embed_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 ff_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 norm_type: str = "layer",
                 enable_mismatch: bool = False,
                 mismatch_sigma: float = 0.01,
                 enable_noise: bool = False,
                 noise_sigma: float = 0.01):
        """
        Initialize analog transformer decoder.
        
        Args:
            embed_dim: Embedding dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            norm_type: Normalization type
            enable_mismatch: Enable resistor mismatch
            mismatch_sigma: Mismatch standard deviation
            enable_noise: Enable thermal noise
            noise_sigma: Noise standard deviation
        """
        super().__init__()
        
        # For simplicity, decoder uses same blocks as encoder
        # In a full implementation, would add cross-attention
        self.layers = nn.ModuleList([
            AnalogTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                norm_type=norm_type,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        if norm_type == "layer":
            self.final_norm = AnalogLayerNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        else:
            self.final_norm = AnalogRMSNorm(
                normalized_shape=embed_dim,
                enable_mismatch=enable_mismatch,
                mismatch_sigma=mismatch_sigma,
                enable_noise=enable_noise,
                noise_sigma=noise_sigma
            )
        
    def forward(self,
                x: torch.Tensor,
                encoder_output: Optional[torch.Tensor] = None,
                self_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        """
        Forward pass of analog transformer decoder.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            encoder_output: Optional encoder output for cross-attention
            self_mask: Optional self-attention mask
            cross_mask: Optional cross-attention mask
        
        Returns:
            output: Output tensor [batch_size, seq_len, embed_dim]
            attention_weights_list: List of attention weights
        """
        attention_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, self_mask)
            attention_weights_list.append(attn_weights)
        
        x = self.final_norm(x)
        
        return x, attention_weights_list
