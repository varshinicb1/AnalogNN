import os
import torch
from circuit_ir.mapping import map_layer_to_circuit
from circuit_ir.exporters import NgspiceExporter, LtspiceExporter
from circuit_ir.circuit import Circuit

class NetlistGenerator:
    @staticmethod
    def generate(weight: torch.Tensor, bias: torch.Tensor | None, x: torch.Tensor,
                 r_ref: float = 10000.0, v_ref: float = 1.0, vmax: float = 15.0,
                 output_dir: str = "./netlists", filename: str = "neuron_layer.cir",
                 backend: str = "ngspice") -> str:
        """
        Creates and saves a SPICE netlist representing the analog inference of a layer.
        
        Returns:
        - The absolute file path of the saved netlist.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Map to Circuit IR
        circuit = map_layer_to_circuit(weight, bias, x, r_ref=r_ref, v_ref=v_ref, name=filename.split('.')[0])
        
        # 2. Add SPICE analysis card (DC analysis)
        analysis_cmds = [
            ".op",              # Calculate DC operating point
            "print all",        # Print all node voltages
        ]
        
        # 3. Export netlist
        if backend.lower() == "ltspice":
            netlist_content = LtspiceExporter.export(circuit, analysis_cmds=analysis_cmds, vmax=vmax)
        else:
            netlist_content = NgspiceExporter.export(circuit, analysis_cmds=analysis_cmds, vmax=vmax)
            
        # 4. Save to disk
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w") as f:
            f.write(netlist_content)
            
        return filepath
