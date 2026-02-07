"""
Host-specific path resolution for MOVR Clinical Analytics

This module provides utilities to resolve paths based on the current hostname,
making notebooks portable across different team members' machines.
"""

import yaml
import socket
import platform
from pathlib import Path
from typing import Optional, Dict, Any
import warnings


class HostPathResolver:
    """
    Resolves paths based on hostname using host_paths.yaml configuration.
    
    Usage:
        resolver = HostPathResolver()
        sharables_path = resolver.get_path('Sharables')
        export_path = resolver.get_path('DataExport')
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize the path resolver.
        
        Args:
            config_file: Path to host_paths.yaml file. If None, uses default location.
        """
        if config_file is None:
            # Default to config/host_paths.yaml relative to project root
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent  # Go up two levels from src/utils/
            config_file = project_root / "config" / "host_paths.yaml"
        
        self.config_file = config_file
        self.hostname = self._get_hostname()
        self.config = self._load_config()
        
    def _get_hostname(self) -> str:
        """Get the current hostname."""
        try:
            return socket.gethostname()
        except Exception:
            return platform.node()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the host paths configuration."""
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            warnings.warn(f"Host paths config file not found: {self.config_file}")
            return {}
        except yaml.YAMLError as e:
            warnings.warn(f"Error parsing host paths config: {e}")
            return {}
    
    def get_path(self, path_type: str, fallback_path: Optional[str] = None) -> Optional[Path]:
        """
        Get a path for the current hostname.
        
        Args:
            path_type: Type of path to retrieve (e.g., 'Sharables', 'DataExport', 'DDR')
            fallback_path: Path to use if hostname not found in config
            
        Returns:
            Path object if found, None if not found and no fallback provided
        """
        if path_type not in self.config:
            if fallback_path:
                warnings.warn(f"Path type '{path_type}' not found in config, using fallback")
                return Path(fallback_path)
            warnings.warn(f"Path type '{path_type}' not found in config")
            return None
        
        hostname_paths = self.config[path_type]
        
        if self.hostname in hostname_paths:
            return Path(hostname_paths[self.hostname])
        
        if fallback_path:
            warnings.warn(f"Hostname '{self.hostname}' not found for path type '{path_type}', using fallback")
            return Path(fallback_path)
        
        # Show available hostnames for debugging
        available_hosts = list(hostname_paths.keys())
        warnings.warn(
            f"Hostname '{self.hostname}' not found for path type '{path_type}'. "
            f"Available hostnames: {available_hosts}"
        )
        return None
    
    def get_export_path(self, export_type: str = 'Sharables', fallback_path: Optional[str] = None) -> Optional[Path]:
        """
        Convenience method for getting export paths.
        
        Args:
            export_type: Type of export ('Sharables', 'DataExport', 'DDR')
            fallback_path: Fallback path if hostname not found
            
        Returns:
            Path for exports or None if not found
        """
        return self.get_path(export_type, fallback_path)
    
    def list_available_paths(self) -> Dict[str, list]:
        """List all available path types and their configured hostnames."""
        result = {}
        for path_type, hostname_paths in self.config.items():
            result[path_type] = list(hostname_paths.keys())
        return result
    
    def get_current_hostname(self) -> str:
        """Get the current hostname."""
        return self.hostname
    
    def validate_path(self, path_type: str) -> tuple[bool, str]:
        """
        Validate that a path exists for the current hostname.
        
        Returns:
            (exists, message) tuple
        """
        path = self.get_path(path_type)
        
        if path is None:
            return False, f"No path configured for {path_type} on hostname {self.hostname}"
        
        if path.exists():
            return True, f"Path exists: {path}"
        else:
            return False, f"Path configured but does not exist: {path}"


def get_sharables_path(fallback_path: Optional[str] = None) -> Optional[Path]:
    """
    Quick function to get the Sharables path for the current hostname.
    
    Args:
        fallback_path: Path to use if hostname not configured
        
    Returns:
        Path to Sharables folder or None
    """
    resolver = HostPathResolver()
    return resolver.get_export_path('Sharables', fallback_path)


def get_data_export_path(fallback_path: Optional[str] = None) -> Optional[Path]:
    """
    Quick function to get the DataExport path for the current hostname.
    
    Args:
        fallback_path: Path to use if hostname not configured
        
    Returns:
        Path to DataExport folder or None
    """
    resolver = HostPathResolver()
    return resolver.get_export_path('DataExport', fallback_path)


def get_ddr_path(fallback_path: Optional[str] = None) -> Optional[Path]:
    """
    Quick function to get the DDR path for the current hostname.
    
    Args:
        fallback_path: Path to use if hostname not configured
        
    Returns:
        Path to DDR folder or None
    """
    resolver = HostPathResolver()
    return resolver.get_export_path('DDR', fallback_path)


if __name__ == "__main__":
    # Test the path resolver
    resolver = HostPathResolver()
    print(f"Current hostname: {resolver.get_current_hostname()}")
    print(f"Available path types: {list(resolver.config.keys())}")
    
    for path_type in ['Sharables', 'DataExport', 'DDR']:
        path = resolver.get_path(path_type)
        exists, message = resolver.validate_path(path_type)
        print(f"{path_type}: {message}")