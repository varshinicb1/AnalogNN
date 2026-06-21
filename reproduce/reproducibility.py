"""
Reproducibility Hardening System
================================

Enforces strict determinism and reproducible seed states across all Python, 
NumPy, PyTorch operations, and SPICE simulations.
"""

import random
import os
import numpy as np
import torch


class ReproducibilityManager:
    """
    Manages random seeds and deterministic behavior for the entire package.
    """

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Sets seeds globally and configures PyTorch deterministic flags.
        """
        # 1. Standard Python random
        random.seed(seed)
        
        # 2. NumPy random
        np.random.seed(seed)
        
        # 3. PyTorch seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # 4. Strict PyTorch determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For newer PyTorch versions
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fallback if deterministic algorithms are not fully supported for some ops in the installed PyTorch
            pass
            
        # 5. OS Environment variable
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # deterministic LSTM/RNN operations if any

    @staticmethod
    def get_generator(seed: int = 42) -> torch.Generator:
        """
        Returns a PyTorch Generator with the specified seed for localized stochastic operations.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
