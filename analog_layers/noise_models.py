import torch

def apply_weight_noise(weight: torch.Tensor, sigma: float, training: bool = False) -> torch.Tensor:
    """
    Applies additive Gaussian noise to the weights.
    w_eff = w + N(0, sigma^2)
    """
    if sigma <= 0.0:
        return weight
    noise = torch.randn_like(weight) * sigma
    return weight + noise

def apply_activation_noise(activation: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Applies additive Gaussian noise to the activations (representing read noise or ADC noise).
    """
    if sigma <= 0.0:
        return activation
    noise = torch.randn_like(activation) * sigma
    return activation + noise
