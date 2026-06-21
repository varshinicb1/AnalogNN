import torch

def apply_mismatch(weight: torch.Tensor, mismatch_sigma: float, pelgrom_matching: bool = False) -> torch.Tensor:
    """
    Models manufacturing mismatch of resistors in crossbars.
    R_actual = R_nominal * (1 + delta)
    Since w = G = 1/R, w_mismatched = w / (1 + delta)
    where delta ~ N(0, mismatch_sigma^2)
    
    If pelgrom_matching is True:
        delta ~ N(0, (mismatch_sigma * sqrt(|w|))^2)
        modeling how smaller resistances (larger conductances/weights) 
        have larger area and thus smaller relative mismatch, or vice-versa.
    """
    if mismatch_sigma <= 0.0:
        return weight
        
    if pelgrom_matching:
        # relative variance scales with weight magnitude: std = sigma * sqrt(|w|)
        # Avoid zero-weight issues by clamping weights
        w_clamped = torch.clamp(torch.abs(weight), min=1e-6)
        std = mismatch_sigma * torch.sqrt(w_clamped)
        delta = torch.randn_like(weight) * std
    else:
        delta = torch.randn_like(weight) * mismatch_sigma
        
    # Fix: Clamp denominator to prevent division by zero, not numerator
    # This ensures numerical stability while preserving physical meaning
    denominator = 1.0 + delta
    denominator = torch.clamp(denominator, min=1e-6)  # Prevent near-zero denominators
    return weight / denominator

