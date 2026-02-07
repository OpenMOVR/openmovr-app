"""
Configuration management for MOVR Clinical Analytics.

This module provides unified access to:
- Data dictionary (datadictionary.parquet)
- Field metadata (categories, validation rules)
- Disease configurations
"""

from .field_registry import FieldRegistry
from .disease_config import DiseaseConfig, load_disease_config

# Backward compatibility: Import old config system
# The old src/config.py file is still used by base.py and other modules
import sys
from pathlib import Path

# Import get_config from the old config.py (parent directory)
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    # Import from the config.py module (not the config package)
    # Need to use absolute import path since we're inside config package
    import importlib.util
    import os

    config_module_path = Path(__file__).parent.parent / 'config.py'
    if config_module_path.exists():
        spec = importlib.util.spec_from_file_location("config_module", config_module_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        get_config = config_module.get_config
        ConfigManager = config_module.ConfigManager
    else:
        raise ImportError("config.py not found")
except Exception as e:
    # If old config.py doesn't exist or can't be loaded, create stub
    def get_config():
        raise NotImplementedError(
            f"Old config system not available: {e}. "
            "Use FieldRegistry and DiseaseConfig instead."
        )
    ConfigManager = None

__all__ = [
    'FieldRegistry',
    'DiseaseConfig',
    'load_disease_config',
    'get_config',  # Backward compatibility
    'ConfigManager'  # Backward compatibility
]
