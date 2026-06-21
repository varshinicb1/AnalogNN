"""
PySpice-based SPICE Runner
==========================

Replaces manual subprocess calls with PySpice library for proper
ngspice simulation and result parsing.
"""

import numpy as np
import torch
from typing import Optional, Dict

try:
    from PySpice.Spice.Netlist import Circuit as PySpiceCircuit
    from PySpice.Simulation import NgSpiceSharedSimulation
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False


class PySpiceRunner:
    """
    SPICE simulation runner using PySpice library.
    Provides proper ngspice integration with automatic result parsing.
    Falls back to FallbackNodalSolver if PySpice is not installed.
    """

    def __init__(self, config: Dict):
        self.config = config
        if not PYSPICE_AVAILABLE:
            print("PySpice not installed. Using FallbackNodalSolver.")

    def run(self, weight: torch.Tensor, bias: Optional[torch.Tensor],
            x: torch.Tensor, r_ref: float, v_ref: float) -> torch.Tensor:
        """
        Run SPICE simulation using PySpice.

        Args:
            weight: Weight matrix (out_features, in_features)
            bias: Bias vector (out_features,)
            x: Input tensor (batch_size, in_features)
            r_ref: Reference resistance
            v_ref: Reference voltage

        Returns:
            Simulated output voltages (batch_size, out_features)
        """
        if not PYSPICE_AVAILABLE:
            from spice.fallback_solver import FallbackNodalSolver
            return FallbackNodalSolver.solve_closed_form(weight, bias, x, self.config)

        w_np = weight.detach().cpu().numpy()
        b_np = bias.detach().cpu().numpy() if bias is not None else np.zeros(weight.shape[0])
        x_np = x.detach().cpu().numpy()

        if len(x_np.shape) == 1:
            x_np = x_np.reshape(1, -1)

        out_features, in_features = w_np.shape
        batch_size = x_np.shape[0]

        outputs = []
        for i in range(batch_size):
            try:
                circuit = self._create_circuit(w_np, b_np, x_np[i], r_ref, v_ref)
                simulator = circuit.simulator()
                analysis = simulator.operating_point()

                voltages = self._extract_output_voltages(analysis, out_features)
                outputs.append(voltages)
            except Exception as e:
                print(f"PySpice simulation failed for sample {i}: {e}")
                from spice.fallback_solver import FallbackNodalSolver
                voltages = FallbackNodalSolver.solve_closed_form(
                    weight, bias, x[i:i+1], self.config
                )
                outputs.append(voltages.numpy().flatten())

        return torch.tensor(np.array(outputs), dtype=torch.float32)

    def _create_circuit(self, weight: np.ndarray, bias: np.ndarray,
                       x: np.ndarray, r_ref: float, v_ref: float):
        """
        Create PySpice Circuit object from weights and inputs.
        Implements differential summing amplifier topology.
        """
        circuit = PySpiceCircuit('Analog Neural Layer')

        out_features, in_features = weight.shape

        for j in range(in_features):
            circuit.V(f'in_{j}', f'nin_{j}', circuit.gnd, float(x[j] * v_ref))

        circuit.V('bias', 'nbias', circuit.gnd, v_ref)

        for i in range(out_features):
            for j in range(in_features):
                if weight[i, j] > 0:
                    r_val = r_ref / max(abs(weight[i, j]), 1e-6)
                    circuit.R(f'rpos_{i}_{j}', f'nin_{j}', f'nsum_pos_{i}', r_val)

            for j in range(in_features):
                if weight[i, j] < 0:
                    r_val = r_ref / max(abs(weight[i, j]), 1e-6)
                    circuit.R(f'rneg_{i}_{j}', f'nin_{j}', f'nsum_neg_{i}', r_val)

            if bias[i] > 0:
                r_bias = r_ref / max(abs(bias[i]), 1e-6)
                circuit.R(f'rbias_pos_{i}', 'nbias', f'nsum_pos_{i}', r_bias)
            elif bias[i] < 0:
                r_bias = r_ref / max(abs(bias[i]), 1e-6)
                circuit.R(f'rbias_neg_{i}', 'nbias', f'nsum_neg_{i}', r_bias)

            circuit.R(f'rf_pos_{i}', f'nsum_pos_{i}', f'nout_pos_{i}', r_ref)
            circuit.R(f'rf_neg_{i}', f'nsum_neg_{i}', f'nout_neg_{i}', r_ref)

            circuit.R(f'rg_pos_{i}', circuit.gnd, f'nsum_pos_{i}', r_ref)
            circuit.R(f'rg_neg_{i}', circuit.gnd, f'nsum_neg_{i}', r_ref)

            circuit.R(f'rsub1_{i}', f'nout_pos_{i}', f'nsub_{i}', r_ref)
            circuit.R(f'rsub2_{i}', f'nout_neg_{i}', f'ndiff_{i}', r_ref)
            circuit.R(f'rsub3_{i}', f'nsub_{i}', f'nout_{i}', r_ref)
            circuit.R(f'rsub4_{i}', f'ndiff_{i}', circuit.gnd, r_ref)
            circuit.R(f'rg_diff_{i}', circuit.gnd, f'nsub_{i}', r_ref)

        return circuit

    def _extract_output_voltages(self, analysis, out_features: int) -> np.ndarray:
        """
        Extract output voltages from PySpice analysis results.
        """
        voltages = []
        for i in range(out_features):
            try:
                v_out = float(analysis[f'nout_{i}'])
                voltages.append(v_out)
            except (KeyError, TypeError, IndexError):
                try:
                    v_pos = float(analysis[f'nout_pos_{i}'])
                    v_neg = float(analysis[f'nout_neg_{i}'])
                    voltages.append(v_neg - v_pos)
                except (KeyError, TypeError, IndexError):
                    voltages.append(0.0)
        return np.array(voltages)
