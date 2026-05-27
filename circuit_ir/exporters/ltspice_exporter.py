from circuit_ir.circuit import Circuit
from circuit_ir.components import Resistor, Capacitor, OpAmp, VoltageSource, CurrentSource

class LtspiceExporter:
    @staticmethod
    def export(circuit: Circuit, analysis_cmds: list = None, vmax: float = 15.0) -> str:
        """
        Translates a Circuit IR into an LTspice netlist string.
        """
        netlist = []
        netlist.append(f"* OpenAnalogNN Auto-Generated LTspice Netlist: {circuit.name}")
        
        # LTspice subcircuit for op-amp with limit
        netlist.append("* Ideal Op-Amp model with saturation limit")
        netlist.append(".subckt opamp_sat IN+ IN- OUT GND")
        netlist.append(f"E1 OUT GND value={{limit(1e5*(V(IN+)-V(IN-)), {-vmax}, {vmax})}}")
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
                netlist.append(f"X_{comp.name} {n[0]} {n[1]} {n[2]} {n[3]} opamp_sat")
            elif isinstance(comp, VoltageSource):
                src_type = comp.source_type
                val_str = f"{comp.value}" if src_type == "DC" else f"{src_type} {comp.value}"
                netlist.append(f"V_{comp.name} {n[0]} {n[1]} {val_str}")
            elif isinstance(comp, CurrentSource):
                netlist.append(f"I_{comp.name} {n[0]} {n[1]} {comp.value}")
        
        netlist.append("")
        
        # Add analysis commands
        if analysis_cmds:
            netlist.append("* Analysis Commands")
            for cmd in analysis_cmds:
                netlist.append(cmd)
            
        netlist.append(".end")
        return "\n".join(netlist)
