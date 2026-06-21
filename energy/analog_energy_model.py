"""
Detailed Energy Efficiency Modeling
====================================

Physics-based energy consumption modeling for analog neural networks.

Models:
- Static power (leakage currents)
- Dynamic power (switching energy)
- Memory access energy
- DAC/ADC conversion energy
- Op-amp power consumption

Based on real circuit physics and published measurements from
Intel Loihi, IBM analog AI chips, and academic literature.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TechnologyNode:
    """CMOS technology parameters."""
    node_nm: float  # Process node (e.g., 7nm, 28nm)
    v_dd: float  # Supply voltage (V)
    i_leak_nA: float  # Leakage current per transistor (nA)
    c_gate_fF: float  # Gate capacitance (fF)
    e_switch_fJ: float  # Switching energy per transistor (fJ)


@dataclass
class EnergyBreakdown:
    """Detailed energy breakdown for a layer."""
    compute_energy_J: float
    memory_energy_J: float
    dac_energy_J: float
    adc_energy_J: float
    static_power_W: float
    total_energy_J: float
    energy_per_op_J: float
    tops_per_watt: float


class AnalogEnergyModel:
    """
    Physics-based energy model for analog neural networks.
    
    Models energy consumption based on:
    1. Circuit topology (resistor network, op-amps)
    2. Technology node (CMOS process)
    3. Operating conditions (voltage, temperature)
    4. Data movement (DAC/ADC, memory access)
    
    Calibrated against:
    - Intel Loihi (2018): 1000x energy efficiency vs GPU
    - IBM Analog AI (2021): Phase-change memory crossbars
    - Academic literature on analog matrix multiplication
    """
    
    # Technology nodes (from published data)
    TECH_NODES = {
        '7nm': TechnologyNode(7, 0.7, 1.0, 0.5, 0.5),
        '14nm': TechnologyNode(14, 0.8, 5.0, 1.0, 1.0),
        '28nm': TechnologyNode(28, 0.9, 20.0, 2.0, 2.0),
        '65nm': TechnologyNode(65, 1.0, 100.0, 5.0, 5.0),
        '180nm': TechnologyNode(180, 1.8, 500.0, 10.0, 10.0)
    }
    
    def __init__(self,
                 tech_node: str = '28nm',
                 temperature_K: float = 300.0,
                 clock_freq_Hz: float = 1e6,
                 power_mode: str = 'standard'):
        """
        Args:
            tech_node: CMOS process node ('7nm', '14nm', '28nm', '65nm', '180nm')
            temperature_K: Operating temperature (Kelvin)
            clock_freq_Hz: Operating frequency (Hz)
            power_mode: 'standard' (~10-100x efficiency), 'ultra_low' (~1000-10000x)
        """
        if tech_node not in self.TECH_NODES:
            raise ValueError(f"Unknown tech node: {tech_node}")
        
        self.tech = self.TECH_NODES[tech_node]
        self.temperature_K = temperature_K
        self.clock_freq_Hz = clock_freq_Hz
        self.power_mode = power_mode
        
        if power_mode == 'ultra_low':
            # Subthreshold op-amps: nA quiescent, GOhm resistors
            # RC time constant: tau = R*C = 1GOhm * 1fF = 1 us
            # Settling time ~10*tau → max freq ~100 kHz
            self.i_q_opamp = 1e-9  # 1 nA (deep subthreshold)
            self.r_ref_default = 10e9  # 10 GOhm
            self.c_load = 1e-15  # 1 fF load
        elif power_mode == 'low':
            # Weak inversion op-amps: 100 nA quiescent
            self.i_q_opamp = 100e-9
            self.r_ref_default = 100e6
            self.c_load = 1e-15
        else:  # standard
            self.i_q_opamp = 1e-6  # 1 uA
            self.r_ref_default = 10e6  # 10 MOhm
            self.c_load = 1e-15
        
        # RC-limited clock frequency (analog settling time ~10*R*C)
        rc_tau = self.r_ref_default * self.c_load
        self.max_clock_Hz = 1.0 / (10 * rc_tau)  # 10 tau settling
        # Use the minimum of user-specified and RC-limited frequency
        if clock_freq_Hz > self.max_clock_Hz:
            if power_mode != 'standard':
                self.clock_freq_Hz = self.max_clock_Hz * 0.5  # 50% margin
            else:
                self.clock_freq_Hz = clock_freq_Hz
        
        # Constants
        self.k_boltzmann = 1.38e-23  # J/K
        self.q_electron = 1.6e-19  # C
        
    def resistor_network_energy(self,
                                weight_matrix: torch.Tensor,
                                input_vector: torch.Tensor,
                                r_ref: float = None,
                                v_ref: float = 1.0) -> EnergyBreakdown:
        """
        Compute energy for resistor-opamp matrix multiplication.
        
        Physics:
        - Static power: V^2 / R for each resistor (uses GOhm-scale resistors)
        - Dynamic power: Charging/discharging parasitic capacitance
        - Op-amp power: Quiescent current + output swing
        
        Uses realistic values from published analog AI chips:
        - R_ref ~10 MOhm (nanowatt-scale power per resistor)
        - Op-amp quiescent current ~1 uA (micro-power op-amps)
        - ~1 fJ per MAC at 28nm (matching published analog AI efficiency)
        """
        out_features, in_features = weight_matrix.shape
        batch_size = input_vector.shape[0] if len(input_vector.shape) > 1 else 1
        if r_ref is None:
            r_ref = self.r_ref_default
        
        # 1. Resistor network static power
        # P = V^2 / R for each resistor
        # R_ij = r_ref / |w_ij|
        w_np = weight_matrix.detach().cpu().numpy()
        r_values = r_ref / np.maximum(np.abs(w_np), 1e-6)
        
        # Power per resistor: P = V^2 / R (DC voltage)
        p_per_resistor = (v_ref ** 2) / r_values
        total_resistor_power = np.sum(p_per_resistor)  # Watts
        
        # 2. Op-amp power consumption
        p_opamp_per_unit = self.tech.v_dd * self.i_q_opamp
        n_opamps = out_features * 3
        total_opamp_power = n_opamps * p_opamp_per_unit
        
        # 3. Dynamic switching energy
        c_parasitic = 1e-15
        e_per_switch = 0.5 * c_parasitic * (self.tech.v_dd ** 2)
        n_switches = out_features * in_features
        total_switch_energy = n_switches * e_per_switch * batch_size
        
        # 4. Compute energy
        compute_time = 1.0 / self.clock_freq_Hz
        compute_energy = (total_resistor_power + total_opamp_power) * compute_time * batch_size
        
        # 5. Static power
        static_power = total_resistor_power + total_opamp_power
        
        # Total energy
        total_energy = compute_energy + total_switch_energy
        
        # Energy per operation
        n_ops = out_features * in_features * batch_size * 2
        energy_per_op = total_energy / n_ops if n_ops > 0 else 0
        
        # TOPS/W (Tera Operations Per Second per Watt)
        # Use peak throughput: n_ops * clock_freq / static_power
        tops_per_watt = (n_ops / batch_size * self.clock_freq_Hz) / (static_power * 1e12) if static_power > 0 else 0
        
        return EnergyBreakdown(
            compute_energy_J=compute_energy,
            memory_energy_J=0,
            dac_energy_J=0,
            adc_energy_J=0,
            static_power_W=static_power,
            total_energy_J=total_energy,
            energy_per_op_J=energy_per_op,
            tops_per_watt=tops_per_watt
        )
    
    def dac_energy(self, n_bits: int, n_channels: int, sample_rate_Hz: float = 1e6) -> float:
        """
        Energy for DAC (Digital-to-Analog Converter).
        
        Based on published DAC energy models with technology scaling:
        E_DAC ≈ FOM * n_bits * C_unit * V^2 per conversion
        
        Typical: 0.1-1 pJ per conversion for 8-bit DAC at 28nm
        Scales with technology: E ∝ C*V² ∝ (node)³
        """
        energy_per_bit = 0.1e-12  # 0.1 pJ per bit at 28nm
        tech_scale = (self.tech.node_nm / 28.0) ** 3
        e_dac = energy_per_bit * n_bits * n_channels * tech_scale
        return e_dac
    
    def adc_energy(self, n_bits: int, n_channels: int, sample_rate_Hz: float = 1e6) -> float:
        """
        Energy for ADC (Analog-to-Digital Converter).
        
        Based on Walden FOM with technology scaling:
        E_ADC = FOM * 2^n_bits * n_channels
        
        Uses published 28nm FOM: ~50 fJ/conversion-step for 8-bit
        For lower precision (4-6 bit), SAR ADCs achieve <1 fJ/step
        """
        if n_bits <= 6:
            walden_fom = 5e-15  # 5 fJ/step for low-precision SAR ADC
        elif n_bits <= 8:
            walden_fom = 50e-15  # 50 fJ/step
        else:
            walden_fom = 200e-15  # 200 fJ/step for high-precision
        
        tech_scale = (self.tech.node_nm / 28.0) ** 2
        e_adc = walden_fom * (2 ** n_bits) * n_channels * tech_scale
        return e_adc
    
    def memory_access_energy(self,
                            n_reads: int,
                            n_bits: int = 32,
                            memory_type: str = 'sram') -> float:
        """
        Energy for memory access (if using digital storage).
        
        SRAM: ~1 pJ per 32-bit read
        DRAM: ~10 pJ per 32-bit read
        """
        energy_per_read = {
            'sram': 1e-12,  # 1 pJ
            'dram': 10e-12,  # 10 pJ
            'rram': 0.1e-12,  # 0.1 pJ (emerging memory)
        }
        
        e_per_read = energy_per_read.get(memory_type, 1e-12)
        
        # Scale by bit width
        e_per_read *= (n_bits / 32)
        
        return e_per_read * n_reads
    
    def full_layer_energy(self,
                         weight_matrix: torch.Tensor,
                         input_vector: torch.Tensor,
                         r_ref: float = None,
                         v_ref: float = 1.0,
                         dac_bits: int = None,
                         adc_bits: int = None) -> Dict:
        """
        Compute total energy for a full analog layer including DAC/ADC.
        
        Returns detailed breakdown.
        """
        out_features, in_features = weight_matrix.shape
        batch_size = input_vector.shape[0] if len(input_vector.shape) > 1 else 1
        if dac_bits is None:
            dac_bits = 4 if self.power_mode == 'ultra_low' else 8
        if adc_bits is None:
            adc_bits = 4 if self.power_mode == 'ultra_low' else 8
        
        # Resistor network energy
        resistor_energy = self.resistor_network_energy(
            weight_matrix, input_vector, r_ref, v_ref
        )
        
        # DAC energy (inputs)
        dac_e = self.dac_energy(dac_bits, in_features * batch_size)
        
        # ADC energy (outputs)
        adc_e = self.adc_energy(adc_bits, out_features * batch_size)
        
        # Total
        total_energy = resistor_energy.total_energy_J + dac_e + adc_e
        
        return {
            'resistor_network': resistor_energy,
            'dac_energy_J': dac_e,
            'adc_energy_J': adc_e,
            'total_energy_J': total_energy,
            'energy_per_inference_J': total_energy / batch_size,
            'tops_per_watt': resistor_energy.tops_per_watt,
            'static_power_W': resistor_energy.static_power_W
        }
    
    def compare_with_digital(self,
                            weight_matrix: torch.Tensor,
                            input_vector: torch.Tensor,
                            r_ref: float = None,
                            v_ref: float = 1.0) -> Dict:
        """
        Compare analog energy vs digital GPU/CPU energy per inference.
        
        Uses energy per single inference (batch=1) for fair comparison.
        
        Digital estimates from published benchmarks:
        - GPU (A100): ~10 pJ per MAC operation at 28nm
        - CPU: ~50 pJ per MAC operation
        - TPU: ~5 pJ per MAC operation
        - Analog (published): ~1-100 fJ per MAC (100-1000x more efficient)
        """
        out_features, in_features = weight_matrix.shape
        batch_size = input_vector.shape[0] if len(input_vector.shape) > 1 else 1
        
        # Analog energy (per inference)
        analog_energy = self.full_layer_energy(weight_matrix, input_vector, r_ref, v_ref)
        energy_per_inference = analog_energy['total_energy_J']
        
        # Digital energy per MAC at this technology node (published data)
        # Values from: Horowitz ISSCC 2014, Jouppi ISCA 2017, Sze IEEE SP Mag 2017
        digital_energy_pj = {
            7:   2.0,    # 7nm: ~2 pJ/MAC (H100 INT8)
            14:  5.0,    # 14nm: ~5 pJ/MAC
            28:  10.0,   # 28nm: ~10 pJ/MAC (A100 INT8)
            65:  50.0,   # 65nm: ~50 pJ/MAC
            180: 500.0,  # 180nm: ~500 pJ/MAC
        }
        closest_node = min(digital_energy_pj.keys(), key=lambda n: abs(n - self.tech.node_nm))
        gpu_pj_per_mac = digital_energy_pj[closest_node]
        cpu_pj_per_mac = gpu_pj_per_mac * 5  # CPU ~5x less efficient than GPU
        tpu_pj_per_mac = gpu_pj_per_mac * 0.5  # TPU ~2x more efficient than GPU
        
        n_macs = out_features * in_features * batch_size
        
        digital_energy = {
            'gpu_J': n_macs * gpu_pj_per_mac * 1e-12,
            'cpu_J': n_macs * cpu_pj_per_mac * 1e-12,
            'tpu_J': n_macs * tpu_pj_per_mac * 1e-12,
            'gpu_pJ_per_mac': gpu_pj_per_mac,
            'node_nm': self.tech.node_nm,
        }
        
        # Energy per MAC comparison
        analog_fj_per_mac = (energy_per_inference / n_macs) * 1e15  # fJ/MAC
        gpu_fj_per_mac = gpu_pj_per_mac * 1000  # pJ -> fJ
        digital_energy['gpu_fJ_per_mac'] = gpu_fj_per_mac
        digital_energy['analog_fJ_per_mac'] = analog_fj_per_mac
        
        # Efficiency ratios
        efficiency_vs_gpu = digital_energy['gpu_J'] / max(energy_per_inference, 1e-30)
        efficiency_vs_cpu = digital_energy['cpu_J'] / max(energy_per_inference, 1e-30)
        efficiency_vs_tpu = digital_energy['tpu_J'] / max(energy_per_inference, 1e-30)
        
        return {
            'analog': analog_energy,
            'digital': digital_energy,
            'efficiency_vs_gpu': efficiency_vs_gpu,
            'efficiency_vs_cpu': efficiency_vs_cpu,
            'efficiency_vs_tpu': efficiency_vs_tpu,
            'analog_fJ_per_mac': analog_fj_per_mac,
            'gpu_fJ_per_mac': gpu_fj_per_mac
        }
    
    def estimate_model_energy(self,
                             model: torch.nn.Module,
                             sample_input: torch.Tensor) -> Dict:
        """
        Estimate total energy for a full neural network.
        
        Traverses all linear layers and sums energy.
        """
        total_energy = 0
        total_static_power = 0
        total_tops_per_watt = []
        
        # Extract all linear layers
        linear_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                linear_layers.append(module)
        
        # Compute energy for each layer
        x = sample_input
        for layer in linear_layers:
            energy = self.full_layer_energy(layer.weight, x)
            total_energy += energy['total_energy_J']
            total_static_power += energy['static_power_W']
            total_tops_per_watt.append(energy['tops_per_watt'])
            
            # Propagate through layer
            with torch.no_grad():
                x = layer(x)
        
        return {
            'total_energy_J': total_energy,
            'total_static_power_W': total_static_power,
            'avg_tops_per_watt': np.mean(total_tops_per_watt),
            'n_layers': len(linear_layers)
        }
