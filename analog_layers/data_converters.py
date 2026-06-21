"""
Analog-to-Digital and Digital-to-Analog Converters for Neuromorphic Projections
==============================================================================

Models high-fidelity converter non-idealities:
1. Resistor-divider DAC with INL/DNL non-linearities, offset, and gain errors.
2. SAR ADC with comparator noise, static offsets, and quantizer deviations.
"""

import torch
import torch.nn as nn
import numpy as np

class AnalogDAC(nn.Module):
    """
    Models a non-ideal Digital-to-Analog Converter (DAC).
    Converts digital embedding states (normalized [-1, 1]) to physical voltages.
    """
    def __init__(self, bits=8, v_ref=2.5, dnl_sigma=0.01, inl_sigma=0.02, offset_error=0.005, gain_error=0.005):
        super().__init__()
        self.bits = bits
        self.v_ref = v_ref
        self.dnl_sigma = dnl_sigma
        self.inl_sigma = inl_sigma
        self.offset_error = offset_error
        self.gain_error = gain_error
        self.levels = 2 ** bits
        
        # Generate non-uniform quantization step sizes
        steps = np.ones(self.levels)
        if self.dnl_sigma > 0:
            dnl = np.random.normal(0, self.dnl_sigma, self.levels)
            dnl = np.maximum(dnl, -0.9)  # Prevent negative step sizes
            steps += dnl
            
        steps /= steps.sum()  # Normalize to total range
        cum_steps = np.cumsum(steps)
        self.cum_steps = np.insert(cum_steps, 0, 0.0)

    def forward(self, x):
        # Scale to DAC integer levels [0, levels - 1]
        x_scaled = (x + 1.0) / 2.0  # Normalize to [0, 1]
        x_quant = torch.clamp(torch.round(x_scaled * (self.levels - 1)), 0, self.levels - 1)
        
        # Map indices to the actual physical non-linear steps
        cum_steps_t = torch.tensor(self.cum_steps, dtype=x.dtype, device=x.device)
        indices = x_quant.long()
        v_fraction = cum_steps_t[indices]
        
        # Scale back to voltage reference range [-V_ref, V_ref]
        v_out = (v_fraction * 2.0 - 1.0) * self.v_ref
        
        # Apply offset and gain errors
        v_out = v_out * (1.0 + self.gain_error) + self.offset_error
        return v_out


class AnalogADC(nn.Module):
    """
    Models a non-ideal Analog-to-Digital Converter (ADC).
    Quantizes output analog voltages back into normalized digital states [-1, 1].
    """
    def __init__(self, bits=8, v_ref=2.5, comparator_noise=0.002, dnl_sigma=0.01, inl_sigma=0.02, offset_error=0.005):
        super().__init__()
        self.bits = bits
        self.v_ref = v_ref
        self.comparator_noise = comparator_noise
        self.dnl_sigma = dnl_sigma
        self.inl_sigma = inl_sigma
        self.offset_error = offset_error
        self.levels = 2 ** bits
        
        # Generate non-uniform threshold spacing for quantization levels
        steps = np.ones(self.levels)
        if self.dnl_sigma > 0:
            dnl = np.random.normal(0, self.dnl_sigma, self.levels)
            dnl = np.maximum(dnl, -0.9)
            steps += dnl
            
        steps /= steps.sum()
        cum_steps = np.cumsum(steps)
        self.cum_steps = np.insert(cum_steps, 0, 0.0)

    def forward(self, v_in):
        # Apply static offset and comparator noise in physical domain
        v_noisy = v_in + self.offset_error
        if self.comparator_noise > 0:
            v_noisy = v_noisy + torch.randn_like(v_in) * self.comparator_noise
            
        # Normalize voltage to [0, 1] relative to [-V_ref, V_ref]
        v_norm = (v_noisy / self.v_ref + 1.0) / 2.0
        v_norm = torch.clamp(v_norm, 0.0, 1.0)
        
        # Map physical voltage fractions to digital level codes using bucketization
        cum_steps_t = torch.tensor(self.cum_steps, dtype=v_in.dtype, device=v_in.device)
        
        shape = v_norm.shape
        v_flat = v_norm.flatten()
        indices = torch.bucketize(v_flat, cum_steps_t) - 1
        indices = torch.clamp(indices, 0, self.levels - 1)
        
        # Re-scale back to [-1, 1] floating-point output
        digital_fraction = indices.float() / (self.levels - 1)
        digital_out = (digital_fraction * 2.0 - 1.0)
        
        return digital_out.view(shape)
