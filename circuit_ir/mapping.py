import torch
from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, OpAmp, VoltageSource

def map_layer_to_circuit(weight: torch.Tensor, bias: torch.Tensor | None, x: torch.Tensor,
                         r_ref: float = 10000.0, v_ref: float = 1.0, name: str = "AnalogLayer") -> Circuit:
    """
    Maps an analog neural layer (weights & biases) and a single input activation vector (x)
    to a physical resistor-opamp summing/subtracting circuit representation in the Circuit IR.
    
    Parameters:
    - weight: (out_features, in_features) tensor representing synaptic weights
    - bias: (out_features,) tensor representing biases
    - x: (in_features,) tensor representing input activation voltages
    - r_ref: reference resistance value (nominal feedback resistance)
    - v_ref: voltage representing activation value 1.0
    - name: name of the generated circuit
    
    Returns:
    - A Circuit IR object populated with nodes and components.
    """
    circuit = Circuit(name=name)
    out_features, in_features = weight.shape
    
    # 1. Map input activations to voltage sources
    for j in range(in_features):
        x_val = float(x.detach()[j]) if x.requires_grad else float(x[j])
        # Input voltage source connected between input node and ground (node '0')
        circuit.add_component(VoltageSource(
            name=f"in_{j}",
            node1=f"node_in_{j}",
            node2="0",
            value=x_val * v_ref,
            source_type="DC"
        ))
        
    # 2. Setup the global bias voltage source (V_bias = v_ref)
    has_bias = bias is not None
    if has_bias:
        circuit.add_component(VoltageSource(
            name="bias",
            node1="node_bias",
            node2="0",
            value=v_ref,
            source_type="DC"
        ))
        
    # 3. For each output feature (neuron), construct summing and subtracting amplifiers
    for i in range(out_features):
        # Nodes for output i
        sum_pos = f"node_sum_pos_{i}"
        sum_neg = f"node_sum_neg_{i}"
        out_pos = f"node_out_pos_{i}"
        out_neg = f"node_out_neg_{i}"
        sub_pos = f"node_sub_pos_{i}"
        sub_neg = f"node_sub_neg_{i}"
        final_out = f"node_out_{i}"
        
        # --- Positive Summing Op-Amp (Inverting: OUT = - Sum |w_pos| * V_in) ---
        circuit.add_component(OpAmp(
            name=f"opamp_pos_{i}",
            pos_node="0",
            neg_node=sum_pos,
            out_node=out_pos,
            ref_node="0"
        ))
        # Feedback resistor for positive summing node
        circuit.add_component(Resistor(
            name=f"rf_pos_{i}",
            node1=sum_pos,
            node2=out_pos,
            value=r_ref
        ))
        
        # --- Negative Summing Op-Amp (Inverting: OUT = - Sum |w_neg| * V_in) ---
        circuit.add_component(OpAmp(
            name=f"opamp_neg_{i}",
            pos_node="0",
            neg_node=sum_neg,
            out_node=out_neg,
            ref_node="0"
        ))
        # Feedback resistor for negative summing node
        circuit.add_component(Resistor(
            name=f"rf_neg_{i}",
            node1=sum_neg,
            node2=out_neg,
            value=r_ref
        ))
        
        # --- Map Input Weights to Input Resistors ---
        for j in range(in_features):
            w_val = float(weight[i, j])
            abs_w = abs(w_val)
            
            # Skip extremely small/zero weights to avoid division by zero or infinite nets
            if abs_w < 1e-6:
                continue
                
            r_val = r_ref / abs_w
            
            if w_val > 0:
                # Positive weights map to the positive summer (leads to negative voltage)
                circuit.add_component(Resistor(
                    name=f"rw_pos_{i}_{j}",
                    node1=f"node_in_{j}",
                    node2=sum_pos,
                    value=r_val
                ))
            else:
                # Negative weights map to the negative summer (leads to negative voltage)
                circuit.add_component(Resistor(
                    name=f"rw_neg_{i}_{j}",
                    node1=f"node_in_{j}",
                    node2=sum_neg,
                    value=r_val
                ))
                
        # --- Map Bias to Bias Resistors ---
        if has_bias:
            b_val = float(bias[i])
            abs_b = abs(b_val)
            if abs_b >= 1e-6:
                r_bias_val = r_ref / abs_b
                if b_val > 0:
                    circuit.add_component(Resistor(
                        name=f"rbias_pos_{i}",
                        node1="node_bias",
                        node2=sum_pos,
                        value=r_bias_val
                    ))
                else:
                    circuit.add_component(Resistor(
                        name=f"rbias_neg_{i}",
                        node1="node_bias",
                        node2=sum_neg,
                        value=r_bias_val
                    ))

        # --- Differential Subtractor Stage (V_final = V_out_neg - V_out_pos) ---
        # Note: out_neg represents (- sum_{w<0} |w| x) and out_pos represents (- sum_{w>0} |w| x).
        # We want: final_out = -out_pos - (-out_neg) = out_neg - out_pos = sum_{w>0} |w| x - sum_{w<0} |w| x.
        # This yields exactly the correct weighted sum!
        circuit.add_component(OpAmp(
            name=f"opamp_sub_{i}",
            pos_node=sub_pos,
            neg_node=sub_neg,
            out_node=final_out,
            ref_node="0"
        ))
        
        # Subtractor resistors (all equal to r_ref for unity differential gain)
        # R1: out_pos -> sub_neg
        circuit.add_component(Resistor(
            name=f"rsub1_{i}",
            node1=out_pos,
            node2=sub_neg,
            value=r_ref
        ))
        # R2 (feedback): sub_neg -> final_out
        circuit.add_component(Resistor(
            name=f"rsub2_{i}",
            node1=sub_neg,
            node2=final_out,
            value=r_ref
        ))
        # R3: out_neg -> sub_pos
        circuit.add_component(Resistor(
            name=f"rsub3_{i}",
            node1=out_neg,
            node2=sub_pos,
            value=r_ref
        ))
        # R4: sub_pos -> ground
        circuit.add_component(Resistor(
            name=f"rsub4_{i}",
            node1=sub_pos,
            node2="0",
            value=r_ref
        ))
        
    return circuit
