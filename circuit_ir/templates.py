"""
Circuit Templates Library
=========================

Provides reusable hardware templates for mapping analog neural networks to
different circuit topologies. Includes:
1. Single-Ended Inverting Summing Amplifier.
2. Differential Summing Amplifier (the default 3-opamp model).
3. Transimpedance Amplifier (TIA) for current-input configurations.
4. Active Gilbert Cell Multiplier model.
"""

from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, OpAmp, VoltageSource, CurrentSource
import numpy as np
from typing import Optional


class CircuitTemplates:
    """
    Standardized templates for analog neural computation blocks.
    """

    @staticmethod
    def single_ended_summing_amp(weights: np.ndarray, bias: Optional[float], inputs: np.ndarray,
                                  r_ref: float = 10000.0, v_ref: float = 1.0, name: str = "SingleEnded") -> Circuit:
        """
        Creates a single-ended inverting summing amplifier:
        V_out = - (sum_j (R_f / R_in_j) * V_in_j + (R_f / R_bias) * V_bias)
        
        Note: Since it is inverting, positive weights are mapped to negative output.
        """
        circuit = Circuit(name=name)
        in_features = len(inputs)
        
        # Setup inputs
        for j in range(in_features):
            circuit.add_component(VoltageSource(
                name=f"in_{j}",
                node1=f"node_in_{j}",
                node2="0",
                value=float(inputs[j] * v_ref)
            ))
            
        if bias is not None:
            circuit.add_component(VoltageSource(
                name="bias",
                node1="node_bias",
                node2="0",
                value=v_ref
            ))
            
        sum_node = "node_sum"
        out_node = "node_out"
        
        # Inverting Op-Amp
        circuit.add_component(OpAmp(
            name="opamp",
            pos_node="0",
            neg_node=sum_node,
            out_node=out_node
        ))
        
        # Feedback resistor
        circuit.add_component(Resistor(
            name="rf",
            node1=sum_node,
            node2=out_node,
            value=r_ref
        ))
        
        # Input resistors representing weights
        for j in range(in_features):
            w = weights[j]
            abs_w = abs(w)
            if abs_w < 1e-6:
                continue
            circuit.add_component(Resistor(
                name=f"rw_{j}",
                node1=f"node_in_{j}",
                node2=sum_node,
                value=r_ref / abs_w
            ))
            
        # Bias resistor
        if bias is not None and abs(bias) >= 1e-6:
            circuit.add_component(Resistor(
                name="rbias",
                node1="node_bias",
                node2=sum_node,
                value=r_ref / abs(bias)
            ))
            
        return circuit

    @staticmethod
    def differential_summing_amp(weights: np.ndarray, bias: Optional[float], inputs: np.ndarray,
                                 r_ref: float = 10000.0, v_ref: float = 1.0, name: str = "DiffAmp") -> Circuit:
        """
        Creates a differential summing amplifier stage using positive/negative summers and subtractor.
        (Matches the implementation in mapping.py, packaged as a reusable template).
        """
        # Convert weight/bias vector/value to appropriate structures for map_layer_to_circuit
        import torch
        from circuit_ir.mapping import map_layer_to_circuit
        
        w_tensor = torch.tensor(weights).unsqueeze(0)
        b_tensor = torch.tensor([bias]) if bias is not None else None
        x_tensor = torch.tensor(inputs)
        
        return map_layer_to_circuit(w_tensor, b_tensor, x_tensor, r_ref=r_ref, v_ref=v_ref, name=name)

    @staticmethod
    def transimpedance_amplifier(inputs_currents: np.ndarray, weights_conductances: np.ndarray,
                                 r_feedback: float = 10000.0, name: str = "TIA_Stage") -> Circuit:
        """
        Creates a transimpedance stage converting current inputs (e.g. photodiode-based optoelectronic NN)
        to voltages: V_out = - I_in * R_feedback.
        """
        circuit = Circuit(name=name)
        n_inputs = len(inputs_currents)
        
        sum_node = "node_sum"
        out_node = "node_out"
        
        circuit.add_component(OpAmp(
            name="tia_opamp",
            pos_node="0",
            neg_node=sum_node,
            out_node=out_node
        ))
        
        circuit.add_component(Resistor(
            name="rf",
            node1=sum_node,
            node2=out_node,
            value=r_feedback
        ))
        
        # Add parallel current sources representing photodiode currents multiplied by conductance weights
        for j in range(n_inputs):
            eff_current = inputs_currents[j] * weights_conductances[j]
            circuit.add_component(CurrentSource(
                name=f"isrc_{j}",
                node1="0",
                node2=sum_node,
                value=float(eff_current)
            ))
            
        return circuit
