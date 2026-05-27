import torch

def apply_saturation(tensor: torch.Tensor, vmax: float) -> torch.Tensor:
    """
    Models physical saturation due to hardware supply voltage limits.
    clamp(x, -vmax, vmax)
    """
    if vmax <= 0.0:
        return tensor
    return torch.clamp(tensor, -vmax, vmax)
