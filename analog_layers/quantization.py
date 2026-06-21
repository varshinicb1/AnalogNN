import torch

def quantize_tensor(tensor: torch.Tensor, bits: int, symmetric: bool = True) -> torch.Tensor:
    """
    Quantizes a tensor to simulate finite-resolution DACs/ADCs.
    Q(x) = round(x * (2^n - 1)) / (2^n - 1)
    """
    if bits <= 0 or bits >= 32:
        return tensor
        
    if symmetric:
        # Symmetric quantization: maps [-max_val, max_val] to 2^(bits-1) - 1 levels
        max_val = torch.max(torch.abs(tensor))
        if max_val == 0:
            # Fix: Return zeros tensor for all-zero input, not unchanged tensor
            return torch.zeros_like(tensor)
        levels = 2**(bits - 1) - 1
        scaled = tensor / max_val
        quantized = torch.round(scaled * levels) / levels
        return quantized * max_val
    else:
        # Asymmetric quantization: maps [min_val, max_val] to 2^bits - 1 levels
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        val_range = max_val - min_val
        if val_range == 0:
            return tensor
        levels = 2**bits - 1
        scaled = (tensor - min_val) / val_range
        quantized = torch.round(scaled * levels) / levels
        return quantized * val_range + min_val
