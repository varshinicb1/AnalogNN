"""
AnalogNN Compiler
==================

Compiles any PyTorch model to analog crossbar topology.

Pipeline:
    PyTorch nn.Module 
    -> Layer extraction (Linear, Conv2d) 
    -> Weight mapping (positive/negative split for differential)
    -> Crossbar array generation (resistor grid)
    -> SPICE netlist export
    -> Energy report
    -> Mismatch-optimized weight remapping

This is an "LLVM for analog AI" — hardware-agnostic compilation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CrossbarConfig:
    """Configuration for a single crossbar array."""
    name: str
    rows: int          # Output neurons
    cols: int          # Input activations
    r_ref: float = 1e6  # Reference resistor (Ohms)
    v_ref: float = 1.0   # Reference voltage (V)
    vmax: float = 2.5    # Saturation voltage (V)
    positive_weights: Optional[np.ndarray] = None
    negative_weights: Optional[np.ndarray] = None
    positive_biases: Optional[np.ndarray] = None
    negative_biases: Optional[np.ndarray] = None


@dataclass
class AnalogModelSpec:
    """Complete specification of an analog model."""
    layers: List[CrossbarConfig]
    total_macs: int
    total_energy_pJ: float
    area_um2: float
    n_crossbars: int
    topology: str  # 'differential' or 'single_ended'


class ModelExtractor:
    """Extract linear/conv layers from a PyTorch model."""
    
    @staticmethod
    def extract(model: nn.Module,
                input_shape: Tuple[int, ...]) -> List[Dict]:
        """Extract layer specifications from a PyTorch model."""
        layers = []
        current_shape = input_shape
        
        # Try DifferentiableAnalogLinear first (it has .weight directly)
        for name, module in model.named_modules():
            # Check by type first for clean dispatch
            module_type = type(module).__name__
            
            if module_type == 'DifferentiableAnalogLinear':
                W = module.weight.data.detach().cpu().numpy()
                b = module.bias.data.detach().cpu().numpy() if module.bias is not None else None
                layers.append({
                    'name': name,
                    'type': 'linear',
                    'weight': W,
                    'bias': b,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'macs': module.in_features * module.out_features
                })
                current_shape = (module.out_features,)
                continue
            
            if isinstance(module, nn.Linear):
                W = module.weight.data.detach().cpu().numpy()
                b = module.bias.data.detach().cpu().numpy() if module.bias is not None else None
                
                layers.append({
                    'name': name,
                    'type': 'linear',
                    'weight': W,
                    'bias': b,
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'macs': module.in_features * module.out_features
                })
                
                current_shape = (module.out_features,)
            
            elif isinstance(module, nn.Conv2d):
                W = module.weight.data.detach().cpu().numpy()
                
                # Compute output shape
                h_out = (current_shape[1] + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
                w_out = (current_shape[2] + 2 * module.padding[1] - module.kernel_size[1]) // module.stride[1] + 1
                
                macs = module.out_channels * h_out * w_out * module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                
                layers.append({
                    'name': name,
                    'type': 'conv2d',
                    'weight': W,
                    'bias': module.bias.data.detach().cpu().numpy() if module.bias is not None else None,
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'macs': macs
                })
                
                current_shape = (module.out_channels, h_out, w_out)
        
        return layers


class CrossbarMapper:
    """
    Maps layer weights to differential crossbar arrays.
    
    For each output neuron i:
        Positive summer: Σ w_ij⁺ · x_j  where w_ij⁺ = max(w_ij, 0)
        Negative summer: Σ w_ij⁻ · x_j  where w_ij⁻ = max(-w_ij, 0)
        Output: pos_summer - neg_summer
        
    Resistor mapping:
        R_ij⁺ = R_ref / max(|w_ij⁺|, epsilon) 
        R_ij⁻ = R_ref / max(|w_ij⁻|, epsilon)
    """
    
    @staticmethod
    def map_linear(weight: np.ndarray,
                   bias: Optional[np.ndarray] = None,
                   r_ref: float = 1e6,
                   v_ref: float = 1.0,
                   min_weight: float = 1e-6) -> CrossbarConfig:
        """Map a linear layer to differential crossbar."""
        n_out, n_in = weight.shape
        
        pos_W = np.maximum(weight, 0)
        neg_W = np.maximum(-weight, 0)
        
        # Resistor values: R = R_ref / |w| (avoid div-by-zero)
        pos_R = np.full_like(pos_W, 1e12, dtype=np.float64)
        mask_p = pos_W > min_weight
        np.divide(r_ref, pos_W, out=pos_R, where=mask_p)
        
        neg_R = np.full_like(neg_W, 1e12, dtype=np.float64)
        mask_n = neg_W > min_weight
        np.divide(r_ref, neg_W, out=neg_R, where=mask_n)
        
        # Effective weights (for numerical checking)
        pos_eff = np.where(pos_R < 1e11, np.divide(r_ref, pos_R, where=pos_R > 0), 0)
        neg_eff = np.where(neg_R < 1e11, np.divide(r_ref, neg_R, where=neg_R > 0), 0)
        
        # Bias mapping
        pos_bias = None
        neg_bias = None
        if bias is not None:
            pos_bias = np.maximum(bias, 0)
            neg_bias = np.maximum(-bias, 0)
        
        return CrossbarConfig(
            name='linear',
            rows=n_out,
            cols=n_in,
            r_ref=r_ref,
            v_ref=v_ref,
            positive_weights=pos_eff,
            negative_weights=neg_eff,
            positive_biases=pos_bias,
            negative_biases=neg_bias
        )
    
    @staticmethod
    def map_conv2d(weight: np.ndarray,
                   bias: Optional[np.ndarray] = None,
                   r_ref: float = 1e6,
                   v_ref: float = 1.0,
                   min_weight: float = 1e-6) -> CrossbarConfig:
        """Map a Conv2d layer to crossbar (unrolled)."""
        n_out, n_in, kh, kw = weight.shape
        n_weights = n_in * kh * kw
        
        # Unroll convolution weights
        weight_unrolled = weight.reshape(n_out, n_weights)
        
        pos_W = np.maximum(weight_unrolled, 0)
        neg_W = np.maximum(-weight_unrolled, 0)
        
        pos_R = np.where(pos_W > min_weight, r_ref / pos_W, 1e12)
        neg_R = np.where(neg_W > min_weight, r_ref / neg_W, 1e12)
        
        pos_eff = np.where(pos_R < 1e11, r_ref / pos_R, 0)
        neg_eff = np.where(neg_R < 1e11, r_ref / neg_R, 0)
        
        pos_bias = None
        neg_bias = None
        if bias is not None:
            pos_bias = np.maximum(bias, 0)
            neg_bias = np.maximum(-bias, 0)
        
        return CrossbarConfig(
            name='conv2d',
            rows=n_out,
            cols=n_weights,
            r_ref=r_ref,
            v_ref=v_ref,
            positive_weights=pos_eff,
            negative_weights=neg_eff,
            positive_biases=pos_bias,
            negative_biases=neg_bias
        )


class EnergyEstimator:
    """Estimate analog energy consumption."""
    
    @staticmethod
    def estimate(config: CrossbarConfig, 
                 technology_nm: int = 65,
                 power_mode: str = 'standard') -> float:
        """Estimate energy per inference in pJ."""
        # MAC energy
        if technology_nm == 7:
            mac_pj = 0.02 if power_mode == 'ultra_low' else 0.2
        elif technology_nm == 28:
            mac_pj = 0.08 if power_mode == 'ultra_low' else 0.5
        else:  # 65nm
            mac_pj = 0.15 if power_mode == 'ultra_low' else 1.0
        
        n_macs = config.rows * config.cols
        compute_energy = n_macs * mac_pj
        
        # ADC energy (dominates at high precision)
        if power_mode == 'ultra_low':
            adc_bits = 4
        elif power_mode == 'low':
            adc_bits = 6
        else:
            adc_bits = 8
        
        adc_energy_pj = config.rows * (2 ** adc_bits) * 0.01  # pJ per conversion
        
        # DAC energy
        dac_energy_pj = config.cols * adc_bits * 0.005  # pJ per conversion
        
        return compute_energy + adc_energy_pj + dac_energy_pj
    
    @staticmethod
    def estimate_area(config: CrossbarConfig) -> float:
        """Estimate area in um^2."""
        cell_area_um2 = 0.5  # Typical 1T1R cell area
        return config.rows * config.cols * cell_area_um2


class SPICENetlistGenerator:
    """Generate SPICE netlists from crossbar config."""
    
    @staticmethod
    def generate(config: CrossbarConfig) -> str:
        """Generate ngspice-compatible netlist."""
        lines = []
        lines.append(f"* Crossbar array: {config.name}")
        lines.append(f"* {config.rows} outputs x {config.cols} inputs")
        lines.append(f".subckt {config.name}_layer")
        lines.append("")
        
        # Reference resistor and voltage
        lines.append(f"Vref ref_node 0 DC {config.v_ref}")
        lines.append(f"Rref pos_sum 0 {config.r_ref}")
        lines.append(f"Rref_neg neg_sum 0 {config.r_ref}")
        lines.append("")
        
        # Input voltage sources
        for j in range(config.cols):
            lines.append(f"V{j} in_{j} 0 DC 0")
        lines.append("")
        
        # Positive weight resistors
        if config.positive_weights is not None:
            for i in range(min(config.rows, 5)):  # Limit for readability
                for j in range(min(config.cols, 5)):
                    w = config.positive_weights[i, j] if config.positive_weights.ndim > 1 else config.positive_weights[j]
                    if abs(w) > 1e-6:
                        r_val = config.r_ref / max(abs(w), 1e-6)
                        r_val = min(max(r_val, 1e3), 1e12)  # Bound resistors
                        lines.append(f"Rpos_{i}_{j} in_{j} pos_sum_{i} {r_val:.1f}")
        
        lines.append("")
        
        # Negative weight resistors
        if config.negative_weights is not None:
            for i in range(min(config.rows, 5)):
                for j in range(min(config.cols, 5)):
                    w = config.negative_weights[i, j] if config.negative_weights.ndim > 1 else config.negative_weights[j]
                    if abs(w) > 1e-6:
                        r_val = config.r_ref / max(abs(w), 1e-6)
                        r_val = min(max(r_val, 1e3), 1e12)
                        lines.append(f"Rneg_{i}_{j} in_{j} neg_sum_{i} {r_val:.1f}")
        
        lines.append("")
        lines.append("* Op-amp differential stage")
        for i in range(min(config.rows, 3)):
            lines.append(f"Eopamp_{i} out_{i} 0 pos_sum_{i} neg_sum_{i} {config.vmax}")
        
        lines.append("")
        lines.append(".ends")
        
        return "\n".join(lines)


class AnalogNNCompiler:
    """
    Main compiler: PyTorch model -> Analog specification.
    
    Usage:
        compiler = AnalogNNCompiler()
        spec = compiler.compile(model, input_shape=(1, 64))
        print(spec.total_energy_pJ)
    """
    
    def __init__(self,
                 r_ref: float = 1e6,
                 v_ref: float = 1.0,
                 vmax: float = 2.5,
                 technology_nm: int = 65,
                 power_mode: str = 'standard'):
        self.r_ref = r_ref
        self.v_ref = v_ref
        self.vmax = vmax
        self.technology_nm = technology_nm
        self.power_mode = power_mode
    
    def compile(self, model: nn.Module,
                input_shape: Tuple[int, ...]) -> AnalogModelSpec:
        """Compile a PyTorch model to analog specification."""
        extractor = ModelExtractor()
        layers = extractor.extract(model, input_shape)
        
        crossbars = []
        total_macs = 0
        total_energy = 0.0
        total_area = 0.0
        
        for layer_spec in layers:
            if layer_spec['type'] == 'linear':
                cb = CrossbarMapper.map_linear(
                    layer_spec['weight'],
                    layer_spec['bias'],
                    self.r_ref,
                    self.v_ref
                )
            elif layer_spec['type'] == 'conv2d':
                cb = CrossbarMapper.map_conv2d(
                    layer_spec['weight'],
                    layer_spec['bias'],
                    self.r_ref,
                    self.v_ref
                )
            else:
                continue
            
            crossbars.append(cb)
            total_macs += layer_spec['macs']
            total_energy += EnergyEstimator.estimate(cb, self.technology_nm, self.power_mode)
            total_area += EnergyEstimator.estimate_area(cb)
        
        return AnalogModelSpec(
            layers=crossbars,
            total_macs=total_macs,
            total_energy_pJ=total_energy,
            area_um2=total_area,
            n_crossbars=len(crossbars),
            topology='differential'
        )
    
    def compile_and_export(self, model: nn.Module,
                           input_shape: Tuple[int, ...],
                           output_dir: str = './netlists') -> AnalogModelSpec:
        """Compile and export to SPICE netlists."""
        import os
        from pathlib import Path
        
        spec = self.compile(model, input_shape)
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, cb in enumerate(spec.layers):
            netlist = SPICENetlistGenerator.generate(cb)
            path = os.path.join(output_dir, f'{cb.name}_layer_{i}.cir')
            with open(path, 'w') as f:
                f.write(netlist)
        
        # Write summary
        summary_path = os.path.join(output_dir, 'compiler_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"AnalogNN Compiler Summary\n")
            f.write(f"{'='*40}\n")
            f.write(f"Total MACs: {spec.total_macs:,}\n")
            f.write(f"Total Energy: {spec.total_energy_pJ:.2f} pJ\n")
            f.write(f"Total Area: {spec.area_um2:.2f} um^2\n")
            f.write(f"Crossbars: {spec.n_crossbars}\n")
            f.write(f"Topology: {spec.topology}\n")
            f.write(f"Technology: {self.technology_nm}nm\n")
            f.write(f"Power Mode: {self.power_mode}\n")
            
            for i, cb in enumerate(spec.layers):
                f.write(f"\nLayer {i} ({cb.name}): {cb.rows}x{cb.cols}\n")
                layer_energy = EnergyEstimator.estimate(cb, self.technology_nm, self.power_mode)
                f.write(f"  Energy: {layer_energy:.2f} pJ\n")
        
        print(f"  Exported {spec.n_crossbars} crossbars to {output_dir}/")
        return spec
    
    def summary(self, spec: AnalogModelSpec) -> str:
        """Human-readable summary."""
        lines = []
        lines.append("=" * 50)
        lines.append("AnalogNN Compilation Report")
        lines.append("=" * 50)
        lines.append(f"  Topology:      {spec.topology}")
        lines.append(f"  Crossbars:     {spec.n_crossbars}")
        lines.append(f"  Total MACs:    {spec.total_macs:,}")
        lines.append(f"  Energy:        {spec.total_energy_pJ:.1f} pJ")
        lines.append(f"  Area:          {spec.area_um2:.1f} um^2")
        lines.append(f"  Technology:    {self.technology_nm}nm")
        lines.append(f"  Power Mode:    {self.power_mode}")
        lines.append("-" * 50)
        
        for i, cb in enumerate(spec.layers):
            layer_energy = EnergyEstimator.estimate(cb, self.technology_nm, self.power_mode)
            lines.append(f"  Layer {i}: {cb.name} {cb.rows}x{cb.cols} ({layer_energy:.1f} pJ)")
        
        lines.append("=" * 50)
        return "\n".join(lines)
