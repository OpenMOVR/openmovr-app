"""
Disease Configuration - Disease-specific field collections and metadata.

Provides convenient access to disease-specific field groupings while
referencing the FieldRegistry as the source of truth.

Author: MOVR Data Science Team
Date: 2025-11-20
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional
from .field_registry import FieldRegistry


class DiseaseConfig:
    """
    Disease-specific configuration and field access.

    Provides convenient methods to work with disease-specific field collections
    while referencing FieldRegistry for actual field metadata.

    Usage:
        registry = FieldRegistry("data/datadictionary.parquet")
        dmd = DiseaseConfig("DMD", registry)

        # Get all mobility fields for DMD
        mobility_fields = dmd.get_domain_fields("mobility")

        # Get fields for trial eligibility pattern
        trial_fields = dmd.get_trial_pattern_fields("ambulatory_status")
    """

    def __init__(self, disease: str, field_registry: FieldRegistry, config_dir: str = "config"):
        """
        Initialize DiseaseConfig.

        Args:
            disease: Disease name (e.g., "DMD", "SMA")
            field_registry: FieldRegistry instance (source of truth)
            config_dir: Path to config directory
        """
        self.disease = disease.upper()
        self.registry = field_registry
        self.config_dir = Path(config_dir)

        # Load disease-specific config
        self.config = self._load_disease_config()

    def _load_disease_config(self) -> Dict:
        """Load disease-specific YAML configuration."""
        config_path = self.config_dir / "diseases" / f"{self.disease.lower()}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Disease configuration not found: {config_path}\n"
                f"Available diseases: {self._list_available_diseases()}"
            )

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _list_available_diseases(self) -> List[str]:
        """List available disease configurations."""
        diseases_dir = self.config_dir / "diseases"
        if not diseases_dir.exists():
            return []

        return [
            f.stem.upper()
            for f in diseases_dir.glob("*.yaml")
            if not f.name.startswith("_")
        ]

    def get_domain_fields(self, domain: str, include_metadata: bool = True) -> List[Dict]:
        """
        Get all fields for a clinical domain.

        Args:
            domain: Clinical domain name (e.g., "mobility", "cardiac")
            include_metadata: Include full field metadata from registry

        Returns:
            List of field names or field info dictionaries

        Example:
            dmd = DiseaseConfig("DMD", registry)
            mobility = dmd.get_domain_fields("mobility")
        """
        if 'clinical_domains' not in self.config:
            raise ValueError(f"No clinical domains defined for {self.disease}")

        if domain not in self.config['clinical_domains']:
            available = list(self.config['clinical_domains'].keys())
            raise ValueError(
                f"Domain '{domain}' not found for {self.disease}. "
                f"Available: {available}"
            )

        domain_config = self.config['clinical_domains'][domain]
        field_names = set()

        # Collect fields from categories
        if 'field_categories' in domain_config:
            for category_path in domain_config['field_categories']:
                category, subcategory = category_path.split('.')
                fields = self.registry.get_fields_by_category(category, subcategory)
                field_names.update(fields)

        # Add explicitly listed fields
        if 'key_fields' in domain_config:
            field_names.update(domain_config['key_fields'])

        # Return with or without metadata
        if include_metadata:
            return [
                self.registry.get_field_info(field)
                for field in sorted(field_names)
                if self.registry.validate_field_exists(field)
            ]
        else:
            return sorted(field_names)

    def get_trial_pattern_fields(self, pattern: str) -> Dict:
        """
        Get fields for a trial eligibility pattern.

        Args:
            pattern: Pattern name (e.g., "ambulatory_status", "cardiac_function")

        Returns:
            Dictionary with pattern info and field metadata

        Example:
            ambulatory_pattern = dmd.get_trial_pattern_fields("ambulatory_status")
            print(ambulatory_pattern['description'])
            print(ambulatory_pattern['fields'])
        """
        if 'trial_patterns' not in self.config:
            raise ValueError(f"No trial patterns defined for {self.disease}")

        if pattern not in self.config['trial_patterns']:
            available = list(self.config['trial_patterns'].keys())
            raise ValueError(
                f"Pattern '{pattern}' not found for {self.disease}. "
                f"Available: {available}"
            )

        pattern_config = self.config['trial_patterns'][pattern]

        # Get field metadata
        fields_info = [
            self.registry.get_field_info(field)
            for field in pattern_config['fields']
            if self.registry.validate_field_exists(field)
        ]

        return {
            'pattern': pattern,
            'description': pattern_config.get('description', ''),
            'fields': fields_info,
            'exclusion_criteria': pattern_config.get('exclusion_threshold', {})
        }

    def get_diagnosis_criteria(self) -> Dict:
        """Get diagnosis identification criteria for this disease."""
        if 'diagnosis' not in self.config:
            raise ValueError(f"No diagnosis criteria defined for {self.disease}")

        return self.config['diagnosis']

    def get_all_disease_fields(self, include_metadata: bool = False) -> List:
        """
        Get all fields across all domains for this disease.

        Args:
            include_metadata: Include full field metadata

        Returns:
            List of unique field names or field info dicts
        """
        all_fields = set()

        if 'clinical_domains' in self.config:
            for domain in self.config['clinical_domains'].keys():
                domain_fields = self.get_domain_fields(domain, include_metadata=False)
                all_fields.update(domain_fields)

        if include_metadata:
            return [
                self.registry.get_field_info(field)
                for field in sorted(all_fields)
                if self.registry.validate_field_exists(field)
            ]
        else:
            return sorted(all_fields)

    def get_domain_names(self) -> List[str]:
        """Get list of available clinical domains for this disease."""
        if 'clinical_domains' not in self.config:
            return []
        return list(self.config['clinical_domains'].keys())

    def get_trial_pattern_names(self) -> List[str]:
        """Get list of available trial patterns for this disease."""
        if 'trial_patterns' not in self.config:
            return []
        return list(self.config['trial_patterns'].keys())

    def summary(self) -> Dict:
        """Get summary of disease configuration."""
        all_fields = self.get_all_disease_fields(include_metadata=False)

        return {
            'disease': self.disease,
            'full_name': self.config.get('full_name', self.disease),
            'total_fields': len(all_fields),
            'clinical_domains': self.get_domain_names(),
            'trial_patterns': self.get_trial_pattern_names(),
            'has_diagnosis_criteria': 'diagnosis' in self.config
        }

    def __repr__(self):
        """String representation."""
        return f"DiseaseConfig(disease='{self.disease}', fields={len(self.get_all_disease_fields())})"


def load_disease_config(disease: str, data_dict_path: str, config_dir: str = "config") -> DiseaseConfig:
    """
    Convenience function to load disease config with registry.

    Args:
        disease: Disease name
        data_dict_path: Path to datadictionary.parquet
        config_dir: Path to config directory

    Returns:
        DiseaseConfig instance

    Example:
        dmd = load_disease_config("DMD", "data/datadictionary.parquet")
        fields = dmd.get_domain_fields("mobility")
    """
    registry = FieldRegistry(data_dict_path, config_dir)
    return DiseaseConfig(disease, registry, config_dir)
