"""
MOVR BASE Configuration Loader
Loads field mappings from YAML config to ensure consistent field names
across all analysis scripts and notebooks.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

class MovrBaseConfig:
    """Loads and provides access to MOVR BASE field configuration."""
    
    def __init__(self, config_path: str = "config/movr_base.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            return self._config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
    
    @property
    def data_sources(self) -> Dict[str, str]:
        """Get data source mappings."""
        return self._config.get('data_sources', {})
    
    @property
    def ambulatory_config(self) -> Dict[str, Any]:
        """Get ambulatory configuration."""
        return self._config.get('ambulatory_config', {})
    
    def get_ambulatory_fields_for_disease(self, disease: str) -> Dict[str, Any]:
        """
        Get all ambulatory fields for a specific disease.
        
        Args:
            disease: Disease code (e.g., 'DMD', 'ALS', 'SMA')
            
        Returns:
            Dictionary with ambulatory field categories for the disease
        """
        disease_fields = self.ambulatory_config.get('disease_fields', {})
        return disease_fields.get(disease.upper(), {})
    
    def get_ambulatory_fields_by_category(self, category: str, disease: str = None) -> Dict[str, Any]:
        """
        Get ambulatory fields by functional category.
        
        Args:
            category: Functional category (e.g., 'walking_independence', 'wheelchair_use')
            disease: Optional disease filter
            
        Returns:
            Dictionary with fields for the category
        """
        if disease:
            disease_fields = self.get_ambulatory_fields_for_disease(disease)
            return disease_fields.get(category, {})
        else:
            # Return category across all diseases
            all_fields = {}
            disease_fields = self.ambulatory_config.get('disease_fields', {})
            for disease_name, categories in disease_fields.items():
                if category in categories:
                    all_fields[disease_name] = categories[category]
            return all_fields
    
    def get_common_ambulatory_fields(self) -> Dict[str, Any]:
        """Get cross-disease common ambulatory fields."""
        return self.ambulatory_config.get('common_fields', {})
    
    def get_ambulatory_analysis_templates(self) -> Dict[str, Any]:
        """Get predefined analysis templates for ambulatory data."""
        return self.ambulatory_config.get('analysis_templates', {})
    
    def get_ambulatory_shortcuts(self) -> Dict[str, Any]:
        """Get field filtering shortcuts for ambulatory analysis."""
        return self._config.get('ambulatory_shortcuts', {})
    
    def get_timed_walk_fields(self, disease: str = None) -> Dict[str, Any]:
        """
        Get timed walk test fields (ttwr10m and related).
        
        Args:
            disease: Optional disease filter
            
        Returns:
            Dictionary with timed walk fields
        """
        if disease:
            disease_fields = self.get_ambulatory_fields_for_disease(disease)
            return disease_fields.get('mobility_testing', {})
        else:
            common_fields = self.get_common_ambulatory_fields()
            return common_fields.get('timed_walk_test', {})
    
    def get_wheelchair_fields(self, disease: str = None) -> Dict[str, Any]:
        """
        Get wheelchair usage fields.
        
        Args:
            disease: Optional disease filter
            
        Returns:
            Dictionary with wheelchair fields
        """
        if disease:
            disease_fields = self.get_ambulatory_fields_for_disease(disease)
            return disease_fields.get('wheelchair_use', {})
        else:
            common_fields = self.get_common_ambulatory_fields()
            return common_fields.get('wheelchair_usage', {})
    
    def get_walking_milestone_fields(self, disease: str) -> Dict[str, Any]:
        """
        Get walking milestone achievement and loss fields for a disease.
        
        Args:
            disease: Disease code
            
        Returns:
            Dictionary with milestone fields
        """
        disease_fields = self.get_ambulatory_fields_for_disease(disease)
        return disease_fields.get('walking_independence', {})
    
    def list_ambulatory_diseases(self) -> list:
        """Get list of diseases with ambulatory field mappings."""
        disease_fields = self.ambulatory_config.get('disease_fields', {})
        return list(disease_fields.keys())
    
    def list_ambulatory_categories(self) -> Dict[str, str]:
        """Get list of ambulatory categories with descriptions."""
        categories = self.ambulatory_config.get('categories', {})
        return {cat: info.get('description', '') for cat, info in categories.items()}
