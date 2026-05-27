import torch

def apply_mismatch(weight: torch.Tensor, mismatch_sigma: float) -> torch.Tensor:
    """
    Models manufacturing mismatch of resistors in crossbars.
    R_actual = R_nominal * (1 + delta)
    Since w = G = 1/R, w_mismatched = w / (1 + delta)
    where delta ~ N(0, mismatch_sigma^2)
    """
    if mismatch_sigma <= 0.0:
        return weight
    delta = torch.randn_like(weight) * mismatch_sigma
    # Prevent division by zero or negative resistance in case of extremely high mismatch_sigma
    delta = torch.clamp(delta, min=-0.99)
    return weight / (1.0 + delta)
