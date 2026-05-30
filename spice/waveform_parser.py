import re
import os

class WaveformParser:
    @staticmethod
    def parse_raw_file(raw_filepath: str) -> dict:
        """
        Parses an ASCII-formatted ngspice RAW file or plain text log file
        to extract final DC node voltages.
        
        Returns:
        - Dict mapping node names to voltage values.
        """
        voltages = {}
        if not os.path.exists(raw_filepath):
            return voltages
            
        try:
            with open(raw_filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                
            # Method 1: Parse standard ngspice ASCII raw values
            if "Variables:" in content and "Values:" in content:
                # Extract variables list mapping index to node name
                var_sec = content.split("Variables:")[1].split("Values:")[0].strip()
                variables = []
                for line in var_sec.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        # parts[0]: index, parts[1]: node_name, parts[2]: type
                        variables.append(parts[1].lower())
                
                # Extract values section
                val_sec = content.split("Values:")[1].strip()
                lines = val_sec.splitlines()
                idx = 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        val = float(parts[1])
                        if idx < len(variables):
                            voltages[variables[idx]] = val
                        idx += 1
                    elif len(parts) == 1:
                        # Sometimes values are on a line by themselves
                        val = float(parts[0])
                        if idx < len(variables):
                            voltages[variables[idx]] = val
                        idx += 1
            
            # Method 2: Resilient regex search fallback in log files
            # Look for patterns like "v(node_out_i) = 1.234" or "node_out_i = 1.234"
            patterns = [
                r"v\((node_[a-z0-9_]+)\)\s*=\s*([+-]?\d+\.?\d*[eE]?[+-]?\d*)",
                r"(node_[a-z0-9_]+)\s+(?:voltage|current)\s*=\s*([+-]?\d+\.?\d*[eE]?[+-]?\d*)",
                r"(node_[a-z0-9_]+)\s*=\s*([+-]?\d+\.?\d*[eE]?[+-]?\d*)",
                r"(node_[a-z0-9_]+)\s+([+-]?\d+\.?\d*[eE]?[+-]?\d*)",
            ]
            
            for pat in patterns:
                matches = re.findall(pat, content, re.IGNORECASE)
                for node, val_str in matches:
                    node_lower = node.lower()
                    if node_lower not in voltages:
                        try:
                            voltages[node_lower] = float(val_str)
                        except ValueError:
                            pass
                            
        except Exception as e:
            print(f"Error parsing raw file {raw_filepath}: {e}")
            
        return voltages
