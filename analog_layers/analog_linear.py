import torch
import torch.nn as nn
import math

from analog_layers.noise_models import apply_weight_noise, apply_activation_noise
from analog_layers.drift_models import apply_drift
from analog_layers.quantization import quantize_tensor
from analog_layers.saturation import apply_saturation
from analog_layers.mismatch import apply_mismatch
from analog_layers.thermal_noise import apply_thermal_noise
from analog_layers.temperature_dependence import apply_temperature_drift, RESISTOR_TCR

class AnalogLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, config: dict = None, **kwargs):
        super(AnalogLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Nominal weights and biases
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        merged = dict(config or {})
        merged.update(kwargs)
        self.set_config(merged)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def set_config(self, config: dict):
        """
        Sets the analog non-ideality configuration parameters.
        """
        if 'mismatch_sigma' in config and 'resistor_mismatch' not in config:
            config['resistor_mismatch'] = config['mismatch_sigma']
        self.noise_sigma = float(config.get('noise_sigma', 0.0) or 0.0)
        self.quantization_bits = int(config.get('quantization_bits', 0) or 0)
        self.drift_tau = float(config.get('drift_tau', 0.0) or 0.0)
        self.drift_time = float(config.get('drift_time', 0.0) or 0.0)
        self.saturation_vmax = float(config.get('saturation_vmax', 0.0) or 0.0)
        self.resistor_mismatch = float(config.get('resistor_mismatch', 0.0) or 0.0)
        self.opamp_offset = float(config.get('opamp_offset', 0.0) or 0.0)
        self.pelgrom_matching = bool(config.get('pelgrom_matching', False))
        self.thermal_noise = float(config.get('thermal_noise', 0.0) or 0.0)
        self.temperature_K = float(config.get('temperature_K', 300.0) or 300.0)
        self.resistor_type = str(config.get('resistor_type', 'standard'))
        self.temperature_C = float(config.get('temperature_C', 25.0) or 25.0)
        self.resistance_ohm = float(config.get('resistance_ohm', 1e4) or 1e4)
        self.bandwidth_Hz = float(config.get('bandwidth_Hz', 1e6) or 1e6)
        
        # Flags to toggle individual non-idealities
        self.enable_mismatch = config.get('enable_mismatch', True)
        self.enable_quantization = config.get('enable_quantization', True)
        self.enable_drift = config.get('enable_drift', True)
        self.enable_noise = config.get('enable_noise', True)
        self.enable_offset = config.get('enable_offset', True)
        self.enable_saturation = config.get('enable_saturation', True)
        self.enable_thermal = config.get('enable_thermal', True)
        self.enable_temperature = config.get('enable_temperature', True)

    @classmethod
    def from_digital(cls, digital_linear: nn.Linear, config: dict = None):
        """
        Creates an AnalogLinear layer preloaded with weights/biases from a digital PyTorch nn.Linear layer.
        """
        has_bias = digital_linear.bias is not None
        analog_layer = cls(
            in_features=digital_linear.in_features,
            out_features=digital_linear.out_features,
            bias=has_bias,
            config=config
        )
        with torch.no_grad():
            analog_layer.weight.copy_(digital_linear.weight)
            if has_bias:
                analog_layer.bias.copy_(digital_linear.bias)
        return analog_layer

    def get_effective_weights(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Applies all weight-level non-idealities sequentially and returns the effective weight and bias.
        """
        w_eff = self.weight.clone()
        b_eff = self.bias.clone() if self.bias is not None else None
        
        # 0. Temperature drift (TCR changes resistance, then mismatch adds variation)
        if self.enable_temperature and self.resistor_type != 'ideal':
            w_eff = apply_temperature_drift(w_eff, self.temperature_C, resistor_type=self.resistor_type)
            if b_eff is not None:
                b_eff = apply_temperature_drift(b_eff, self.temperature_C, resistor_type=self.resistor_type)
                
        # 1. Resistor Mismatch (static, depends on physical tolerance)
        if self.enable_mismatch and self.resistor_mismatch > 0.0:
            w_eff = apply_mismatch(w_eff, self.resistor_mismatch, self.pelgrom_matching)
            if b_eff is not None:
                b_eff = apply_mismatch(b_eff, self.resistor_mismatch, self.pelgrom_matching)
                
        # 2. Weight Quantization (resolution limit)
        if self.enable_quantization and self.quantization_bits > 0:
            w_eff = quantize_tensor(w_eff, self.quantization_bits, symmetric=True)
            if b_eff is not None:
                b_eff = quantize_tensor(b_eff, self.quantization_bits, symmetric=True)
                
        # 3. Drift (decay over time)
        if self.enable_drift and self.drift_time > 0.0 and self.drift_tau > 0.0:
            w_eff = apply_drift(w_eff, self.drift_time, self.drift_tau)
            if b_eff is not None:
                b_eff = apply_drift(b_eff, self.drift_time, self.drift_tau)
                
        # 4. Weight Noise (dynamic, temporal fluctuation)
        if self.enable_noise and self.noise_sigma > 0.0:
            w_eff = apply_weight_noise(w_eff, self.noise_sigma, training=self.training)
            # Biases are represented as hardware voltages or DAC feeds, also subject to noise
            if b_eff is not None:
                b_eff = apply_weight_noise(b_eff, self.noise_sigma, training=self.training)
                
        return w_eff, b_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get weight and bias after hardware non-idealities
        w_eff, b_eff = self.get_effective_weights()
        
        # Run linear operation
        y = torch.matmul(x, w_eff.t())
        if b_eff is not None:
            y = y + b_eff
            
        # 5. Op-Amp Input Offset Voltage
        # The offset causes a systematic output offset related to the gain:
        # V_out_offset = (1 + sum_j |w_j|) * V_os
        # Fix: Use proper circuit topology - for differential summing architecture,
        # the noise gain is computed per output neuron based on its input resistor network
        if self.enable_offset and self.opamp_offset > 0.0:
            # Calculate closed-loop noise gain: 1 + sum_j (|w_ij|)
            # This represents the noise gain of the summing amplifier before subtraction
            noise_gain = 1.0 + torch.sum(torch.abs(w_eff), dim=1) # Shape: (out_features,)
            # V_os is randomly distributed per op-amp (different for positive/negative summers)
            # For differential architecture, we model offset on both summers
            v_os_pos = torch.randn(self.out_features, device=y.device) * self.opamp_offset
            v_os_neg = torch.randn(self.out_features, device=y.device) * self.opamp_offset
            # Net offset after subtraction: offset_neg - offset_pos
            offset_error = noise_gain * (v_os_neg - v_os_pos)
            y = y + offset_error.unsqueeze(0) # Broadcast to batch size
            
        # 6. Thermal Noise (Johnson-Nyquist from resistors and op-amps)
        if self.enable_thermal and self.thermal_noise > 0.0:
            y = apply_thermal_noise(y, self.temperature_K, self.resistance_ohm, self.bandwidth_Hz)
            
        # 7. Saturation (supply rails limit output voltage range)
        if self.enable_saturation and self.saturation_vmax > 0.0:
            y = apply_saturation(y, self.saturation_vmax)
            
        return y
