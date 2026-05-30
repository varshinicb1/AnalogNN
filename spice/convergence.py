import os

class ConvergenceHandler:
    @staticmethod
    def check_log(log_filepath: str) -> dict:
        """
        Scans a SPICE log file for convergence failures or singular matrix errors.
        
        Returns:
        - Dict with status (bool) and diagnostic recommendations.
        """
        status = {
            'converged': True,
            'issues': [],
            'recommendations': []
        }
        
        if not os.path.exists(log_filepath):
            return status
            
        with open(log_filepath, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read().lower()
            
        # Check standard SPICE failure indicators
        if "convergence failure" in log_content or "no convergence" in log_content:
            status['converged'] = False
            status['issues'].append("general convergence failure")
            status['recommendations'].append("Add '.options reltol=0.01' to relax tolerance.")
            
        if "singular matrix" in log_content:
            status['converged'] = False
            status['issues'].append("singular matrix")
            status['recommendations'].append("Ensure there are no floating nodes and every node has a DC path to ground. Add 1e12 shunt resistors if needed.")
            
        if "timestep too small" in log_content:
            status['converged'] = False
            status['issues'].append("timestep too small in transient simulation")
            status['recommendations'].append("Adjust transient step size or set '.options method=gear' to use Gear integration.")
            
        return status
