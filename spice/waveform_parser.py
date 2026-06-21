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
                # Find the actual Variables: section header line (not No. Variables:)
                # by finding the second occurrence of "Variables:"
                first = content.index("Variables:")
                second = content.index("Variables:", first + 1)
                var_sec = content[second + 10:content.index("Values:")].strip()
                variables = []
                for line in var_sec.splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # parts[0]: index, parts[1]: node_name
                        variables.append(parts[1].lower())
                
                # Extract values section
                val_sec = content.split("Values:")[1].strip()
                lines = val_sec.splitlines()
                # Find point index line: "0\tval0" or just tokens
                # The first token is the point index, skip it
                all_tokens = val_sec.split()
                if all_tokens and all_tokens[0].lstrip('-').replace('.', '').isdigit():
                    all_tokens = all_tokens[1:]  # skip point index
                for idx, val_str in enumerate(all_tokens):
                    if idx < len(variables):
                        try:
                            voltages[variables[idx]] = float(val_str)
                        except ValueError:
                            pass
            
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
