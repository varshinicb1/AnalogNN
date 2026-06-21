from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, Capacitor, OpAmp, VoltageSource, CurrentSource

class NgspiceExporter:
    @staticmethod
    def export(circuit: Circuit, analysis_cmds: list = None, vmax: float = 15.0) -> str:
        """
        Translates a Circuit IR into an ngspice netlist string.
        
        Parameters:
        - circuit: Circuit IR object
        - analysis_cmds: list of SPICE control command strings (e.g., ['.dc V1 0 1 0.1'])
        - vmax: saturation voltage for op-amp subcircuit
        """
        netlist = []
        netlist.append(f"* OpenAnalogNN Auto-Generated ngspice Netlist: {circuit.name}")
        
        # Define Op-Amp subcircuit (with clamping to model saturation)
        netlist.append("* Ideal Op-Amp model with saturation limit")
        netlist.append(".subckt opamp_sat IN+ IN- OUT GND")
        # Behavioral source for clamping: max(min(val, high), low)
        netlist.append(f"B1 OUT GND V=max(min(1e5*(V(IN+)-V(IN-)), {vmax}), {-vmax})")
        netlist.append(".ends opamp_sat")
        netlist.append("")

        # Write components
        netlist.append("* Circuit Components")
        for comp in circuit.components:
            n = comp.nodes
            if isinstance(comp, Resistor):
                netlist.append(f"R_{comp.name} {n[0]} {n[1]} {comp.value}")
            elif isinstance(comp, Capacitor):
                netlist.append(f"C_{comp.name} {n[0]} {n[1]} {comp.value}")
            elif isinstance(comp, OpAmp):
                # Xname pos neg out gnd subcircuit_name
                netlist.append(f"X_{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} opamp_sat")
            elif isinstance(comp, VoltageSource):
                # Vname node1 node2 [type] [value]
                # Default to DC if type is not specified
                src_type = comp.source_type
                val_str = f"{comp.value}" if src_type == "DC" else f"{src_type} {comp.value}"
                netlist.append(f"V_{comp.name} {n[0]} {n[1]} {val_str}")
            elif isinstance(comp, CurrentSource):
                netlist.append(f"I_{comp.name} {n[0]} {n[1]} DC {comp.value}")
        
        netlist.append("")
        
        # Add analysis commands
        if analysis_cmds:
            netlist.append("* Analysis Commands")
            netlist.append(".control")
            for cmd in analysis_cmds:
                netlist.append(cmd)
            netlist.append(".endc")
            
        netlist.append(".end")
        return "\n".join(netlist)
