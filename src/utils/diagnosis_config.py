"""
Diagnosis Field Configuration Loader

Loads field mappings from YAML config to ensure consistent field names
across all analysis scripts and notebooks.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

class DiagnosisFieldConfig:
    """Loads and provides access to diagnosis field configuration."""
    
    def __init__(self, config_path: str = "config/diagnosis_fields.yaml"):
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
    def diagnosis_fields(self) -> Dict[str, str]:
        """Get diagnosis field mappings."""
        return self._config.get('diagnosis_fields', {})
    
    @property
    def age_fields(self) -> Dict[str, str]:
        """Get age field mappings."""
        return self._config.get('age_fields', {})
    
    @property
    def clinical_fields(self) -> Dict[str, str]:
        """Get clinical measurement field mappings."""
        return self._config.get('clinical_fields', {})
    
    @property
    def inclusion_criteria(self) -> Dict[str, Any]:
        """Get inclusion criteria thresholds."""
        return self._config.get('inclusion_criteria', {})
    
    @property
    def diagnosis_values(self) -> Dict[str, str]:
        """Get diagnosis value mappings."""
        return self._config.get('diagnosis_values', {})
    
    @property
    def disease_type_values(self) -> Dict[str, str]:
        """Get disease type value mappings."""
        return self._config.get('disease_type_values', {})
    
    @property
    def diagnosis_date_fields(self) -> Dict[str, str]:
        """Get diagnosis date field mappings."""
        return self._config.get('diagnosis_date_fields', {})
    
    @property
    def data_sources(self) -> Dict[str, str]:
        """Get data source mappings."""
        return self._config.get('data_sources', {})
    
    def get_field(self, category: str, field_name: str) -> str:
        """Get specific field name from category."""
        category_fields = getattr(self, category, {})
        return category_fields.get(field_name, f"{field_name}_NOT_FOUND")
    
    def print_summary(self):
        """Print a summary of all configured fields."""
        print("📋 DIAGNOSIS FIELD CONFIGURATION SUMMARY")
        print("=" * 50)
        
        print("\n🔬 Diagnosis Fields:")
        for key, value in self.diagnosis_fields.items():
            print(f"  {key}: {value}")
        
        if hasattr(self, 'disease_type_values') and self.disease_type_values:
            print("\n🏷️  Disease Type Values:")
            for key, value in self.disease_type_values.items():
                print(f"  {key}: {value}")
        
        if hasattr(self, 'diagnosis_date_fields') and self.diagnosis_date_fields:
            print("\n📅 Diagnosis Date Fields:")
            for key, value in self.diagnosis_date_fields.items():
                print(f"  {key}: {value}")
        
        print("\n📅 Age Fields:")
        for key, value in self.age_fields.items():
            print(f"  {key}: {value}")
            
        print("\n🏥 Clinical Fields:")
        for key, value in self.clinical_fields.items():
            status = "✓ CONFIRMED" if "field_name_here" not in value else "⚠️ NEEDS CONFIRMATION"
            print(f"  {key}: {value} - {status}")
        
        print("\n📊 Inclusion Criteria:")
        for key, value in self.inclusion_criteria.items():
            print(f"  {key}: {value}")


# Convenience function for quick access
def load_diagnosis_config() -> DiagnosisFieldConfig:
    """Load diagnosis field configuration."""
    return DiagnosisFieldConfig()


if __name__ == "__main__":
    # Test the configuration loader
    config = load_diagnosis_config()
    config.print_summary()