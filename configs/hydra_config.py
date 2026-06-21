"""
Hydra + OmegaConf Configuration
===============================

Replaces manual YAML loading with Hydra for hierarchical configuration
management with command-line overrides and composition.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any
import os


class HydraConfigLoader:
    """
    Configuration loader using Hydra and OmegaConf.
    Provides hierarchical config management with validation.
    """

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML using OmegaConf.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to YAML using OmegaConf.
        
        Args:
            config: Configuration dictionary
            output_path: Output YAML file path
        """
        cfg = OmegaConf.create(config)
        OmegaConf.save(cfg, output_path)

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations (override takes precedence).
        
        Args:
            base_config: Base configuration
            override_config: Override configuration
            
        Returns:
            Merged configuration
        """
        base_cfg = OmegaConf.create(base_config)
        override_cfg = OmegaConf.create(override_config)
        merged = OmegaConf.merge(base_cfg, override_cfg)
        return OmegaConf.to_container(merged, resolve=True)

    @staticmethod
    def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        Get nested value from config using dot notation.
        
        Args:
            config: Configuration dictionary
            key_path: Dot-separated key path (e.g., "model.hidden_dims")
            default: Default value if key not found
            
        Returns:
            Value at key path or default
        """
        cfg = OmegaConf.create(config)
        try:
            return OmegaConf.select(cfg, key_path)
        except:
            return default

    @staticmethod
    def validate_config(config: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Schema dictionary with required keys and types
            
        Returns:
            True if valid, False otherwise
        """
        cfg = OmegaConf.create(config)
        schema_cfg = OmegaConf.create(schema)
        
        try:
            OmegaConf.merge(cfg, schema_cfg)
            return True
        except:
            return False


# Hydra decorator for main entry point
@hydra.main(config_path="./configs", config_name="config", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """
    Main entry point using Hydra configuration.
    Command-line overrides: python script.py model.lr=0.001 dataset.name=mnist
    """
    from experiments.runner import ExperimentRunner
    
    # Convert OmegaConf to dict
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Save config for reproducibility
    config_path = os.path.join(hydra.core.utils.get_original_cwd(), "configs", "config.yaml")
    HydraConfigLoader.save_config(config_dict, config_path)
    
    # Run experiment
    runner = ExperimentRunner(config_path)
    results = runner.run_full_pipeline()
    
    print("Experiment completed successfully!")
    return results


if __name__ == "__main__":
    hydra_main()
