"""
Realistic Hardware Variation Dataset
=====================================

Generates realistic device-to-device and run-to-run variation patterns
based on physical models validated against SPICE.

Sources of variation:
1. Pelgrom resistor mismatch: sigma_R = A_R / sqrt(W*L)
2. Op-amp input offset: sigma_OS = A_OS / sqrt(W*L)  
3. Temperature spatial gradient: delta_T across chip
4. Temporal drift: G(t) = G_0 * exp(-t/tau)
5. RTS noise (random telegraph signal): burst noise in resistors

Each "chip" has a unique fingerprint of:
- Per-weight mismatch deltas (static, one-time manufacturing)
- Per-neuron offset voltages (static)
- Temperature profile (slowly varying)
- Drift time constant (per-device)

This enables realistic Monte Carlo simulation of chip-to-chip variation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from analog_layers.mismatch import apply_mismatch
from analog_layers.temperature_dependence import RESISTOR_TCR


@dataclass
class ChipFingerprint:
    """Unique fingerprint for a manufactured chip."""
    chip_id: int
    mismatch_map: Dict[str, torch.Tensor]  # Per-weight mismatch deltas
    offset_map: Dict[str, torch.Tensor]    # Per-neuron offsets
    temp_coefficient: float                # Effective TCR (varies per chip)
    drift_tau: float                       # Drift time constant (seconds)
    rts_amplitude: float                   # RTS noise amplitude
    
    def to_dict(self) -> Dict:
        return {
            'chip_id': self.chip_id,
            'mismatch_std': float(self.mismatch_map.get('weight', torch.tensor(0.0)).std()),
            'offset_std': float(self.offset_map.get('bias', torch.tensor(0.0)).std()),
            'temp_coefficient': self.temp_coefficient,
            'drift_tau': self.drift_tau,
            'rts_amplitude': self.rts_amplitude,
        }


class HardwareVariationGenerator:
    """
    Generates realistic hardware variation patterns.
    
    Uses Pelgrom's law for mismatch scaling:
        sigma_R = A_R / sqrt(W*L)
    
    Where:
        A_R = 1e-3 to 3e-3 um (process-dependent)
        W*L = resistor area in um^2
    """
    
    def __init__(self, 
                 pelgrom_constant: float = 2e-3,
                 min_area: float = 0.1,
                 max_area: float = 100.0,
                 temp_range: Tuple[float, float] = (20.0, 80.0),
                 drift_tau_range: Tuple[float, float] = (1e4, 1e6),
                 seed: int = 42):
        self.pelgrom_constant = pelgrom_constant
        self.min_area = min_area
        self.max_area = max_area
        self.temp_range = temp_range
        self.drift_tau_range = drift_tau_range
        self.rng = np.random.RandomState(seed)
    
    def generate_chip_fingerprint(self, 
                                   weight_shape: Tuple[int, int],
                                   chip_id: int = 0) -> ChipFingerprint:
        """
        Generate a unique chip fingerprint.
        
        Args:
            weight_shape: (out_features, in_features)
            chip_id: unique chip identifier
        
        Returns:
            ChipFingerprint with per-device variation parameters
        """
        out_features, in_features = weight_shape
        
        # Generate random resistor areas (Pelgrom scaling)
        # Larger area = better matching
        resistor_areas = self.rng.uniform(self.min_area, self.max_area, 
                                          size=(out_features, in_features))
        
        # Per-weight mismatch: sigma = A_R / sqrt(area)
        mismatch_sigmas = self.pelgrom_constant / np.sqrt(resistor_areas)
        mismatch_deltas = torch.tensor(
            self.rng.normal(0, mismatch_sigmas), dtype=torch.float32
        )
        
        # Op-amp offset: 1-5 mV typical
        offset_map = {
            'bias': torch.tensor(self.rng.normal(0, 0.002, size=out_features), dtype=torch.float32)
        }
        
        # Temperature coefficient (varies per chip due to process variation)
        temp_coeff = float(self.rng.uniform(
            RESISTOR_TCR['standard']['alpha'] * 0.8,
            RESISTOR_TCR['standard']['alpha'] * 1.2
        ))
        
        # Drift time constant (varies per chip)
        drift_tau = float(10 ** self.rng.uniform(
            np.log10(self.drift_tau_range[0]),
            np.log10(self.drift_tau_range[1])
        ))
        
        # RTS noise amplitude (0.1-1% of signal)
        rts_amplitude = float(self.rng.uniform(0.001, 0.01))
        
        return ChipFingerprint(
            chip_id=chip_id,
            mismatch_map={'weight': mismatch_deltas},
            offset_map=offset_map,
            temp_coefficient=temp_coeff,
            drift_tau=drift_tau,
            rts_amplitude=rts_amplitude,
        )
    
    def generate_population(self, 
                            weight_shape: Tuple[int, int],
                            n_chips: int = 100) -> List[ChipFingerprint]:
        """Generate a population of chip fingerprints."""
        return [
            self.generate_chip_fingerprint(weight_shape, chip_id=i)
            for i in range(n_chips)
        ]
    
    @staticmethod
    def apply_chip_variation(weight: torch.Tensor,
                             bias: Optional[torch.Tensor],
                             fingerprint: ChipFingerprint,
                             temperature: float = 25.0,
                             config: Optional[Dict] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply chip-specific variation to weights and biases.
        
        Args:
            weight: digital weight matrix
            bias: digital bias vector
            fingerprint: chip-specific variation parameters
            temperature: operating temperature
            config: additional non-ideality config
        
        Returns:
            (w_eff, b_eff) after applying chip variation
        """
        w_eff = weight.clone()
        b_eff = bias.clone() if bias is not None else None
        
        # 1. Apply chip-specific mismatch
        if fingerprint.mismatch_map:
            delta = fingerprint.mismatch_map.get('weight', torch.zeros_like(w_eff))
            w_eff = w_eff / (1.0 + delta)
        
        # 2. Apply temperature drift
        tcr = fingerprint.temp_coefficient * 1e-6  # Convert to ppm
        delta_T = temperature - 25.0
        temp_factor = 1.0 + tcr * delta_T
        w_eff = w_eff / temp_factor
        
        # 3. Apply offset
        if b_eff is not None and fingerprint.offset_map:
            b_eff = b_eff + fingerprint.offset_map.get('bias', torch.zeros_like(b_eff))
        
        return w_eff, b_eff


class HardwareAwareDataset(torch.utils.data.Dataset):
    """
    Dataset that applies chip-specific hardware variation to each sample.
    
    Each epoch samples a different chip fingerprint and temperature,
    providing domain randomization over hardware variations.
    """
    
    def __init__(self, 
                 base_dataset: torch.utils.data.Dataset,
                 weight_shape: Tuple[int, int],
                 n_chips: int = 50,
                 temp_range: Tuple[float, float] = (20.0, 80.0),
                 seed: int = 42):
        self.base_dataset = base_dataset
        self.weight_shape = weight_shape
        self.temp_range = temp_range
        
        # Pre-generate chip population
        self.generator = HardwareVariationGenerator(seed=seed)
        self.chips = self.generator.generate_population(weight_shape, n_chips)
        
        # Current chip and temperature
        self.current_chip = None
        self.current_temperature = 25.0
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        return self.base_dataset[idx]
    
    def sample_new_chip(self):
        """Sample a random chip fingerprint for a new epoch."""
        self.current_chip = self.chips[np.random.randint(len(self.chips))]
        self.current_temperature = np.random.uniform(*self.temp_range)
        return self.current_chip, self.current_temperature
    
    def apply_variation(self, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        """Apply current chip variation to weights."""
        if self.current_chip is None:
            self.sample_new_chip()
        return HardwareVariationGenerator.apply_chip_variation(
            weight, bias, self.current_chip, self.current_temperature
        )
    
    def get_chip_statistics(self) -> Dict:
        """Return statistics of the chip population."""
        mismatch_stds = [c.mismatch_map['weight'].std().item() for c in self.chips]
        offset_stds = [c.offset_map['bias'].std().item() for c in self.chips]
        drift_taus = [c.drift_tau for c in self.chips]
        
        return {
            'n_chips': len(self.chips),
            'mismatch_std_mean': float(np.mean(mismatch_stds)),
            'mismatch_std_std': float(np.std(mismatch_stds)),
            'offset_std_mean': float(np.mean(offset_stds)),
            'drift_tau_mean': float(np.mean(drift_taus)),
            'drift_tau_range': [float(min(drift_taus)), float(max(drift_taus))],
            'temp_range': list(self.temp_range),
        }


def demo_hardware_dataset():
    """Demonstrate the hardware variation dataset."""
    print("=" * 60)
    print("Hardware Variation Dataset Demo")
    print("=" * 60)
    
    weight_shape = (10, 20)  # 10 outputs, 20 inputs
    
    # Generate chip population
    print(f"\nGenerating 100 chips for weight shape {weight_shape}...")
    generator = HardwareVariationGenerator(seed=42)
    chips = generator.generate_population(weight_shape, n_chips=100)
    
    # Show chip statistics
    mismatch_stds = [c.mismatch_map['weight'].std().item() for c in chips]
    offset_stds = [c.offset_map['bias'].std().item() for c in chips]
    
    print(f"\nChip Population Statistics (n={len(chips)}):")
    print(f"  Mismatch std: mean={np.mean(mismatch_stds):.4f}, "
          f"min={np.min(mismatch_stds):.4f}, max={np.max(mismatch_stds):.4f}")
    print(f"  Offset std: mean={np.mean(offset_stds):.4f}, "
          f"min={np.min(offset_stds):.4f}, max={np.max(offset_stds):.4f}")
    print(f"  Drift tau: mean={np.mean([c.drift_tau for c in chips]):.1f}s, "
          f"range=[{min(c.drift_tau for c in chips):.1f}, {max(c.drift_tau for c in chips):.1f}]")
    
    # Show chip-to-chip variation
    print(f"\nFirst 3 chips:")
    for i, chip in enumerate(chips[:3]):
        stats = chip.to_dict()
        print(f"  Chip {chip.chip_id}: mismatch_std={stats['mismatch_std']:.4f}, "
              f"offset_std={stats['offset_std']:.4f}, "
              f"drift_tau={stats['drift_tau']:.1f}, "
              f"TCR={stats['temp_coefficient']:.1f} ppm/C")
    
    # Demonstrate applying variation
    print(f"\nApplying variation to weights...")
    weight = torch.randn(*weight_shape)
    bias = torch.randn(weight_shape[0])
    
    for temp in [25.0, 50.0, 85.0]:
        w_eff, b_eff = HardwareVariationGenerator.apply_chip_variation(
            weight, bias, chips[0], temperature=temp
        )
        diff_pct = ((w_eff - weight).abs().mean() / weight.abs().mean() * 100).item()
        print(f"  T={temp:.0f}C: weight change = {diff_pct:.2f}%")
    
    # HardwareAwareDataset demo
    print(f"\nHardwareAwareDataset with chip sampling:")
    simple_dataset = torch.utils.data.TensorDataset(
        torch.randn(100, 20), torch.randint(0, 10, (100,))
    )
    hw_dataset = HardwareAwareDataset(simple_dataset, weight_shape, n_chips=50)
    stats = hw_dataset.get_chip_statistics()
    print(f"  Chips: {stats['n_chips']}")
    print(f"  Mismatch mean: {stats['mismatch_std_mean']:.4f}")
    print(f"  Offset mean: {stats['offset_std_mean']:.4f}")
    print(f"  Temp range: {stats['temp_range']}")
    
    print(f"\nHardware variation dataset ready!")
    return chips


if __name__ == '__main__':
    chips = demo_hardware_dataset()
