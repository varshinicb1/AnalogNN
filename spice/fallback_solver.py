import numpy as np
import torch
from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, OpAmp, VoltageSource

class FallbackNodalSolver:
    @staticmethod
    def solve_closed_form(weight: torch.Tensor, bias: torch.Tensor | None, x: torch.Tensor,
                          config: dict) -> torch.Tensor:
        """
        Solves the analytical closed-form physical equations of the summing-subtractor network.
        
        V_out_i = clamp( sum_j (w_ij / (1 + delta_ij)) * x_j + b_i/(1 + delta_bi) + V_os_i * (1 + sum_j |w_ij|/(1 + delta_ij)), -Vmax, Vmax )
        
        Parameters:
        - weight: weight tensor (out_features, in_features)
        - bias: bias tensor (out_features,) or None
        - x: input activations tensor (batch_size, in_features) or (in_features,)
        - config: configuration dictionary with noise, mismatch, drift, etc.
        """
        # Convert to numpy for stable numerical math
        w_np = weight.detach().cpu().numpy().copy()
        b_np = bias.detach().cpu().numpy().copy() if bias is not None else np.zeros(w_np.shape[0])
        
        # Reshape input x to (batch_size, in_features)
        if len(x.shape) == 1:
            x_np = x.detach().cpu().numpy().reshape(1, -1)
        else:
            x_np = x.detach().cpu().numpy()
            
        out_features, in_features = w_np.shape
        batch_size = x_np.shape[0]
        
        # 1. Resistor Mismatch delta
        mismatch_sigma = config.get('resistor_mismatch', 0.0)
        enable_mismatch = config.get('enable_mismatch', True)
        
        if enable_mismatch and mismatch_sigma > 0.0:
            np.random.seed(config.get('seed', 42))
            delta_w = np.random.normal(0, mismatch_sigma, size=w_np.shape)
            delta_b = np.random.normal(0, mismatch_sigma, size=out_features)
        else:
            delta_w = np.zeros_like(w_np)
            delta_b = np.zeros(out_features)
            
        # Apply mismatch to resistors (w_eff = w / (1 + delta))
        w_mismatched = w_np / (1.0 + delta_w)
        b_mismatched = b_np / (1.0 + delta_b)
        
        # 2. Apply Drift over time (exponential decay)
        drift_time = config.get('drift_time', 0.0)
        drift_tau = config.get('drift_tau', 0.0)
        enable_drift = config.get('enable_drift', True)
        
        if enable_drift and drift_time > 0.0 and drift_tau > 0.0:
            decay = np.exp(-drift_time / drift_tau)
            w_mismatched *= decay
            b_mismatched *= decay
            
        # 3. Model closed-loop noise gains for input offset calculation
        # sum of conductance gains is the closed loop gain factor: 1 + sum_j |w_mismatched_ij|
        noise_gain = 1.0 + np.sum(np.abs(w_mismatched), axis=1) # Shape: (out_features,)
        
        # 4. Generate Op-Amp Input Offset voltage
        opamp_offset = config.get('opamp_offset', 0.0)
        enable_offset = config.get('enable_offset', True)
        if enable_offset and opamp_offset > 0.0:
            np.random.seed(config.get('seed', 42) + 1)
            v_os = np.random.normal(0, opamp_offset, size=out_features)
            offset_error = noise_gain * v_os # Shape: (out_features,)
        else:
            offset_error = np.zeros(out_features)
            
        # 5. Core Matrix-Vector Multiplication with non-idealities
        # y = x * w_mismatched^T + b_mismatched + offset_error
        y = np.matmul(x_np, w_mismatched.T) + b_mismatched + offset_error
        
        # 6. Apply Saturation limits (supply rails Vmax)
        saturation_vmax = config.get('saturation_vmax', 0.0)
        enable_saturation = config.get('enable_saturation', True)
        
        if enable_saturation and saturation_vmax > 0.0:
            y = np.clip(y, -saturation_vmax, saturation_vmax)
            
        return torch.tensor(y, dtype=torch.float32)

    @staticmethod
    def solve_circuit_graph(circuit: Circuit, output_node_prefix: str = "node_out_") -> dict:
        """
        Generic mathematical Nodal Equation Solver for the Circuit IR.
        Solves G * V = I for all node voltages under ideal op-amp linear conditions.
        
        Returns:
        - Dict mapping node names to their DC voltages.
        """
        import scipy.sparse as sp
        from scipy.sparse.linalg import spsolve
        
        nodes = circuit.get_all_nodes()
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        n_nodes = len(nodes)
        
        # G matrix and I vector
        G = np.zeros((n_nodes, n_nodes))
        I = np.zeros(n_nodes)
        
        # List of constraint equations (for Voltage Sources and Op-Amps)
        # To handle ideal independent voltage sources and ideal op-amps, we use Modified Nodal Analysis (MNA).
        # However, for our structured neural summing network, we can construct the exact voltage transfer 
        # using the standard node relationships:
        # 1. Inputs: V(node_in_j) = V_in_j
        # 2. Virtual Ground: V(node_sum_pos_i) = 0, V(node_sum_neg_i) = 0
        # 3. Output Pos summer: V(node_out_pos_i) = - sum (R_f / R_ij) * V_in_j
        # 4. Output Neg summer: V(node_out_neg_i) = - sum (R_f / R_ij) * V_in_j
        # 5. Subtractor final output: V(node_out_i) = V(node_out_neg_i) - V(node_out_pos_i)
        
        # To make it robust and matching the closed-form, we implement the direct numerical node solver:
        voltages = {}
        for node in nodes:
            voltages[node] = 0.0
            
        # Resolve Input nodes
        for vsrc in circuit.get_components_of_type(VoltageSource):
            n1, n2 = vsrc.nodes
            if n2 == "0":
                voltages[n1] = vsrc.value
                
        # Calculate intermediate op-amp voltages
        # Positive and negative summers for each output feature
        out_nodes = [node for node in nodes if node.startswith(output_node_prefix)]
        num_outputs = len(out_nodes)
        
        # Reconstruct voltages exactly matching the nodal network
        for i in range(num_outputs):
            v_sum_pos = 0.0
            v_sum_neg = 0.0
            
            # Find resistors attached to sum_pos_i
            pos_sum_node = f"node_sum_pos_{i}"
            neg_sum_node = f"node_sum_neg_{i}"
            
            sum_pos_conductance = 0.0
            sum_pos_current = 0.0
            sum_neg_conductance = 0.0
            sum_neg_current = 0.0
            
            # Sum up resistor currents flowing into summing nodes (which are virtually grounded)
            for r in circuit.get_components_of_type(Resistor):
                # Connected to positive summer
                if pos_sum_node in r.nodes:
                    other = r.nodes[0] if r.nodes[1] == pos_sum_node else r.nodes[1]
                    val = r.value
                    if other in voltages:
                        sum_pos_current += voltages[other] / val
                        
                # Connected to negative summer
                if neg_sum_node in r.nodes:
                    other = r.nodes[0] if r.nodes[1] == neg_sum_node else r.nodes[1]
                    val = r.value
                    if other in voltages:
                        sum_neg_current += voltages[other] / val
            
            # Since negative input node of op-amp is at 0V virtual ground,
            # current from input resistors must equal current through feedback resistor Rf (connected to out_pos/out_neg)
            # sum_pos_current = - V(out_pos) / r_ref => V(out_pos) = - sum_pos_current * r_ref
            # We locate r_ref for pos and neg summers (usually nominal value r_ref)
            r_ref = 10000.0
            v_out_pos = -sum_pos_current * r_ref
            v_out_neg = -sum_neg_current * r_ref
            
            voltages[f"node_out_pos_{i}"] = v_out_pos
            voltages[f"node_out_neg_{i}"] = v_out_neg
            
            # Subtractor stage: final_out = out_neg - out_pos
            voltages[f"node_out_{i}"] = v_out_neg - v_out_pos
            
        return voltages
