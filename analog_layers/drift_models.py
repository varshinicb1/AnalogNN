import torch
import numpy as np

def apply_drift(weight: torch.Tensor, time: float, tau: float) -> torch.Tensor:
    """
    Simulates conductance drift in non-volatile memory elements.
    w(t) = w_0 * exp(-t / tau)
    
    Parameters:
    - weight: nominal weights at t=0
    - time: elapsed time (seconds)
    - tau: drift time-constant (seconds)
    """
    if time <= 0.0 or tau <= 0.0:
        return weight
    decay_factor = np.exp(-time / tau)
    return weight * decay_factor
