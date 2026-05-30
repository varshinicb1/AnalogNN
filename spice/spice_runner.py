import subprocess
import shutil
import os
import torch
import numpy as np
from spice.fallback_solver import FallbackNodalSolver
from spice.waveform_parser import WaveformParser

class SpiceRunner:
    def __init__(self, config: dict):
        self.config = config
        self.backend = config.get('circuit', {}).get('backend', 'numerical').lower()
        self.ngspice_path = shutil.which("ngspice")
        
        if self.backend == "ngspice" and not self.ngspice_path:
            print("WARNING: ngspice executable not found in PATH. Gracefully falling back to high-performance Numerical Nodal Solver.")
            self.backend = "numerical"

    def run(self, weight: torch.Tensor, bias: torch.Tensor | None, x: torch.Tensor,
            r_ref: float = 10000.0, v_ref: float = 1.0, vmax: float = 15.0) -> torch.Tensor:
        """
        Runs the circuit simulation for a batch of inputs and returns output voltages.
        
        Parameters:
        - weight: weight tensor (out_features, in_features)
        - bias: bias tensor (out_features,)
        - x: input activations tensor (batch_size, in_features)
        
        Returns:
        - Output voltage tensor of shape (batch_size, out_features)
        """
        if self.backend == "numerical":
            # Run high-speed vectorized analytical solver
            return FallbackNodalSolver.solve_closed_form(weight, bias, x, self.config.get('analog', {}))
            
        # If running real SPICE:
        # Note: Since SPICE runs one circuit at a time, we run it for each input in the batch
        # and gather outputs.
        batch_size = x.size(0)
        out_features = weight.size(0)
        outputs = []
        
        # Temp directory for netlists
        temp_dir = "./spice_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # For small batch validation
        for b in range(batch_size):
            from spice.netlist_generator import NetlistGenerator
            netlist_file = NetlistGenerator.generate(
                weight=weight,
                bias=bias,
                x=x[b],
                r_ref=r_ref,
                v_ref=v_ref,
                vmax=vmax,
                output_dir=temp_dir,
                filename=f"sim_batch_{b}.cir",
                backend="ngspice"
            )
            
            # Execute ngspice in batch mode: ngspice -b -r raw_file netlist_file
            raw_file = os.path.join(temp_dir, f"sim_batch_{b}.raw")
            log_file = os.path.join(temp_dir, f"sim_batch_{b}.log")
            
            try:
                cmd = [self.ngspice_path, "-b", "-r", raw_file, netlist_file]
                with open(log_file, "w") as out_f:
                    subprocess.run(cmd, stdout=out_f, stderr=subprocess.STDOUT, check=True, timeout=5.0)
                
                # Parse node voltages from RAW file
                node_voltages = WaveformParser.parse_raw_file(raw_file)
                
                # Extract output voltages (prefix node_out_i)
                b_output = []
                for i in range(out_features):
                    val = node_voltages.get(f"node_out_{i}", 0.0)
                    b_output.append(val)
                outputs.append(b_output)
                
            except Exception as e:
                print(f"SPICE run failed for batch {b}: {e}. Degrading to numerical solver for this sample.")
                # Local degradation
                x_single = x[b].unsqueeze(0)
                num_out = FallbackNodalSolver.solve_closed_form(weight, bias, x_single, self.config.get('analog', {}))
                outputs.append(num_out.squeeze(0).numpy())
                
        # Clean up temp files
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        return torch.tensor(np.array(outputs), dtype=torch.float32)
