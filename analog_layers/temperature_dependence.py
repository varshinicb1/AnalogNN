import torch

RESISTOR_TCR = {
    'standard': {'alpha': 100, 'beta': 0.5},
    'precision': {'alpha': 25, 'beta': 0.1},
    'ultra_precision': {'alpha': 5, 'beta': 0.02},
    'integrated': {'alpha': 800, 'beta': 2.0},
    'ideal': {'alpha': 0, 'beta': 0},
}


def apply_temperature_drift(weight: torch.Tensor, temperature_C: float = 25.0, T_ref: float = 25.0, resistor_type: str = 'standard') -> torch.Tensor:
    """
    Apply temperature-dependent resistor drift to weights.
    w_eff = w / (1 + alpha*1e-6*delta_T + beta*1e-6*delta_T^2)
    """
    tcr = RESISTOR_TCR.get(resistor_type, RESISTOR_TCR['standard'])
    delta_T = temperature_C - T_ref
    temp_factor = 1.0 + tcr['alpha'] * 1e-6 * delta_T + tcr['beta'] * 1e-6 * delta_T ** 2
    return weight / temp_factor
