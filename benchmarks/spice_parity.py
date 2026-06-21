"""
SPICE Parity Benchmarking Suite
===============================

Rigorous validation suite comparing the fidelity of four simulation abstractions:
1. Ideal Float: Pure digital float32 operations.
2. Abstract Analog: Mathematical PyTorch modeling of quantization/noise.
3. Fallback Solver: Analytical physical equations (resistor tolerances, offset).
4. Circuit Graph Solver: Solver solving node voltages directly from Circuit IR.
5. SPICE Solver: Actual ngspice circuit simulation from auto-generated netlists.

Quantifies the exact "abstraction gap" across these layers.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

from spice.spice_runner import SpiceRunner
from spice.fallback_solver import FallbackNodalSolver
from circuit_ir.mapping import map_layer_to_circuit
from validation.metrics import compute_metrics


class SpiceParityBenchmarker:
    """
    Evaluates parity across multiple simulation levels and computes cross-layer
    accuracy and voltage discrepancies.
    """

    def __init__(self, config: dict):
        self.config = config
        self.runner = SpiceRunner(config)
        
    def evaluate_parity(self, weight: torch.Tensor, bias: torch.Tensor | None, x: torch.Tensor,
                        labels: Optional[torch.Tensor] = None) -> Dict:
        """
        Simulates the forward pass of a layer across all 5 abstraction layers
        and computes mutual RMSE, Max Discrepancy, and Pearson R.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        N, in_features = x.shape
        out_features = weight.shape[0]
        
        # Level 1: Ideal digital float
        bias_term = bias if bias is not None else torch.zeros(out_features)
        y_ideal = torch.matmul(x, weight.t()) + bias_term
        
        # Level 2: Abstract Analog Layer (PyTorch)
        from analog_layers.analog_linear import AnalogLinear
        analog_layer = AnalogLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            config=self.config.get('analog', {})
        )
        # Load weights and bias directly to match
        analog_layer.weight.data.copy_(weight)
        if bias is not None:
            analog_layer.bias.data.copy_(bias)
        y_abstract = analog_layer(x)
        
        # Level 3: Fallback Nodal Solver (closed-form physics)
        y_fallback = FallbackNodalSolver.solve_closed_form(weight, bias, x, self.config.get('analog', {}))
        
        # Level 4: Circuit Graph Solver (Solving nodes on Circuit IR graph)
        # We loop through batch because Circuit IR maps single inputs
        y_graph_list = []
        for b in range(N):
            circuit = map_layer_to_circuit(
                weight=weight,
                bias=bias,
                x=x[b],
                r_ref=self.config.get('circuit', {}).get('r_ref', 10000.0),
                v_ref=self.config.get('circuit', {}).get('v_ref', 1.0)
            )
            node_voltages = FallbackNodalSolver.solve_circuit_graph(circuit)
            y_graph_single = [node_voltages.get(f"node_out_{i}", 0.0) for i in range(out_features)]
            y_graph_list.append(y_graph_single)
        y_graph = torch.tensor(y_graph_list, dtype=torch.float32)
        
        # Level 5: SPICE Solver (ngspice netlist)
        # If backend is numerical, runner runs fallback solver, otherwise runs actual ngspice
        y_spice = self.runner.run(
            weight=weight,
            bias=bias,
            x=x,
            r_ref=self.config.get('circuit', {}).get('r_ref', 10000.0),
            v_ref=self.config.get('circuit', {}).get('v_ref', 1.0),
            vmax=self.config.get('analog', {}).get('saturation_vmax', 2.5)
        )
        
        # Standardize labels
        if labels is None:
            labels = torch.zeros(N, dtype=torch.long)
            
        # Compile predictions (detaching to avoid gradient tracking issues)
        predictions = {
            'ideal': y_ideal.detach(),
            'abstract': y_abstract.detach(),
            'fallback': y_fallback.detach(),
            'graph': y_graph.detach(),
            'spice': y_spice.detach()
        }
        
        # Compute discrepancies between all pairs
        discrepancies = {}
        keys = list(predictions.keys())
        
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                k1, k2 = keys[i], keys[j]
                v1, v2 = predictions[k1].numpy(), predictions[k2].numpy()
                
                # RMSE
                rmse = float(np.sqrt(np.mean((v1 - v2) ** 2)))
                # Max absolute error
                max_err = float(np.max(np.abs(v1 - v2)))
                # Correlation
                if np.std(v1) > 1e-8 and np.std(v2) > 1e-8:
                    r = float(np.corrcoef(v1.flatten(), v2.flatten())[0, 1])
                else:
                    r = 1.0 if np.allclose(v1, v2) else 0.0
                    
                discrepancies[f"{k1}_vs_{k2}"] = {
                    'rmse': rmse,
                    'max_discrepancy': max_err,
                    'correlation_r': r
                }
                
        # Compare classification accuracies
        acc_dict = {}
        for name, pred in predictions.items():
            acc = float((torch.argmax(pred, dim=1) == labels).float().mean().item())
            acc_dict[name] = acc
            
        return {
            'discrepancies': discrepancies,
            'accuracies': acc_dict,
            'predictions': {k: v.tolist() for k, v in predictions.items()}
        }

    @staticmethod
    def generate_parity_latex_table(results: Dict) -> str:
        """
        Formats the parity benchmark results as a publication-ready LaTeX table.
        """
        disc = results['discrepancies']
        acc = results['accuracies']
        
        latex = []
        latex.append(r"\begin{table}[htbp]")
        latex.append(r"\centering")
        latex.append(r"\caption{OpenAnalogNN Cross-Abstraction Discrepancy Parity Benchmarking}")
        latex.append(r"\label{tab:spice_parity}")
        latex.append(r"\begin{tabular}{lccc}")
        latex.append(r"\hline")
        latex.append(r"\textbf{Abstraction Comparison} & \textbf{RMSE (V)} & \textbf{Max Discrepancy (V)} & \textbf{Correlation ($R$)} \\")
        latex.append(r"\hline")
        
        for pair, metrics in disc.items():
            name_map = {
                'ideal': 'Ideal Float',
                'abstract': 'Abstract Analog',
                'fallback': 'Fallback Nodal',
                'graph': 'Circuit Graph',
                'spice': 'SPICE Simulation'
            }
            p1, p2 = pair.split('_vs_')
            comp_name = f"{name_map[p1]} vs. {name_map[p2]}"
            
            rmse = metrics['rmse']
            max_d = metrics['max_discrepancy']
            r = metrics['correlation_r']
            
            latex.append(f"{comp_name} & {rmse:.4e} & {max_d:.4e} & {r:.4f} \\\\")
            
        latex.append(r"\hline")
        latex.append(r"\end{tabular}")
        latex.append(r"\end{table}")
        
        return "\n".join(latex)
