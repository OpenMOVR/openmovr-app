"""
Configuration Management for MOVR Clinical Analytics

Centralized configuration handling with support for environment variables,
JSON configs, and runtime parameters.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "movr"
    username: str = ""
    password: str = ""
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Load database config from environment variables."""
        return cls(
            host=os.getenv('MOVR_DB_HOST', cls.host),
            port=int(os.getenv('MOVR_DB_PORT', cls.port)),
            database=os.getenv('MOVR_DB_NAME', cls.database),
            username=os.getenv('MOVR_DB_USER', cls.username),
            password=os.getenv('MOVR_DB_PASSWORD', cls.password)
        )


@dataclass  
class PathConfig:
    """Path configuration for data and outputs."""
    data_dir: Path = Path("data")
    config_dir: Path = Path("config") 
    output_dir: Path = Path("output")
    notebook_dir: Path = Path("notebooks")
    
    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.config_dir = Path(self.config_dir)
        self.output_dir = Path(self.output_dir)
        self.notebook_dir = Path(self.notebook_dir)


@dataclass
class AnalyticsConfig:
    """Analytics and processing configuration."""
    default_chunk_size: int = 10000
    parallel_workers: int = 4
    cache_enabled: bool = True
    debug_mode: bool = False
    
    # Data validation settings
    strict_validation: bool = True
    allow_missing_values: bool = False
    
    # Output settings
    save_intermediate: bool = False
    output_formats: list = None
    
    def __post_init__(self):
        """Set default output formats."""
        if self.output_formats is None:
            self.output_formats = ['csv', 'parquet']


class ConfigManager:
    """
    Central configuration manager for MOVR analytics.
    
    Loads configuration from multiple sources in priority order:
    1. Environment variables
    2. Local config files (config/local.yaml, config/local.json)  
    3. Default config files (config/default.yaml, config/default.json)
    4. Built-in defaults
    """
    
    def __init__(self, config_dir: Union[str, Path] = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self._data_dictionary = None
        self._db_config = None
        self._path_config = None
        self._analytics_config = None
        self._custom_config = {}
        
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration components."""
        # Load in priority order
        configs = {}
        
        # 1. Default config files
        for filename in ['default.yaml', 'default.yml', 'default.json']:
            config_file = self.config_dir / filename
            if config_file.exists():
                configs.update(self._load_config_file(config_file))
        
        # 2. Local config files (override defaults)
        for filename in ['local.yaml', 'local.yml', 'local.json']:
            config_file = self.config_dir / filename  
            if config_file.exists():
                configs.update(self._load_config_file(config_file))
        
        # 3. Environment variables (override file configs)
        self._load_env_overrides(configs)
        
        # Initialize typed config objects
        self._db_config = DatabaseConfig(**configs.get('database', {}))
        self._path_config = PathConfig(**configs.get('paths', {}))
        self._analytics_config = AnalyticsConfig(**configs.get('analytics', {}))
        self._custom_config = configs.get('custom', {})
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    return json.load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
        return {}
    
    def _load_env_overrides(self, configs: Dict[str, Any]):
        """Apply environment variable overrides."""
        # Database overrides
        if 'database' not in configs:
            configs['database'] = {}
        
        env_mappings = {
            'MOVR_DB_HOST': ('database', 'host'),
            'MOVR_DB_PORT': ('database', 'port', int),
            'MOVR_DB_NAME': ('database', 'database'), 
            'MOVR_DB_USER': ('database', 'username'),
            'MOVR_DB_PASSWORD': ('database', 'password'),
            'MOVR_DATA_DIR': ('paths', 'data_dir'),
            'MOVR_OUTPUT_DIR': ('paths', 'output_dir'),
            'MOVR_DEBUG': ('analytics', 'debug_mode', lambda x: x.lower() == 'true'),
            'MOVR_CHUNK_SIZE': ('analytics', 'default_chunk_size', int),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section = config_path[0]
                key = config_path[1]
                transform = config_path[2] if len(config_path) > 2 else str
                
                if section not in configs:
                    configs[section] = {}
                
                try:
                    configs[section][key] = transform(value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse {env_var}={value}: {e}")
    
    @property
    def db(self) -> DatabaseConfig:
        """Database configuration."""
        return self._db_config
    
    @property  
    def paths(self) -> PathConfig:
        """Path configuration."""
        return self._path_config
    
    @property
    def analytics(self) -> AnalyticsConfig:
        """Analytics configuration."""
        return self._analytics_config
    
    def get_data_dictionary_path(self) -> Path:
        """Get path to data dictionary JSON file."""
        return self.config_dir / "data_dictionary.json"
    
    def get_custom(self, key: str, default: Any = None) -> Any:
        """Get custom configuration value."""
        return self._custom_config.get(key, default)
    
    def update_custom(self, key: str, value: Any):
        """Update custom configuration value."""
        self._custom_config[key] = value
    
    def save_config(self, filename: str = "generated.yaml"):
        """Save current configuration to file."""
        config_data = {
            'database': asdict(self._db_config),
            'paths': {k: str(v) for k, v in asdict(self._path_config).items()},
            'analytics': asdict(self._analytics_config),
            'custom': self._custom_config
        }
        
        output_file = self.config_dir / filename
        with open(output_file, 'w') as f:
            if filename.endswith('.json'):
                json.dump(config_data, f, indent=2, default=str)
            else:
                yaml.dump(config_data, f, default_flow_style=False)
        
        return output_file
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist or can be created."""
        try:
            self.paths.data_dir.mkdir(exist_ok=True, parents=True)
            self.paths.output_dir.mkdir(exist_ok=True, parents=True)
            self.paths.notebook_dir.mkdir(exist_ok=True, parents=True)
            return True
        except Exception as e:
            print(f"Path validation failed: {e}")
            return False


# Global config instance
config = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config


def reload_config():
    """Reload configuration from files."""
    global config
    config = ConfigManager()
    return config