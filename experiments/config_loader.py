"""
Configuration Loader
=====================

Centralized configuration loading with schema validation.
"""

import yaml
from typing import Dict, Any
from configs.config_schema import OpenAnalogNNConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary
    """
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    
    # Validate using Pydantic schema
    validated_config = OpenAnalogNNConfig(**data)
    return validated_config.model_dump()
