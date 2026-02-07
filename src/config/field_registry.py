"""
Field Registry - Unified field information system.

Combines datadictionary.parquet (source of truth) with enhancement metadata
from field_metadata/ directory.

Author: MOVR Data Science Team
Date: 2025-11-20
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings


class FieldRegistry:
    """
    Unified registry for all field information.

    Combines:
    - datadictionary.parquet (source of truth for field definitions)
    - config/field_metadata/categories.yaml (field categorization)
    - config/field_metadata/validation_rules.yaml (validation metadata)

    Usage:
        registry = FieldRegistry("data/datadictionary.parquet")

        # Get complete field information
        field_info = registry.get_field_info("ambulatn")

        # Get fields by category
        fields = registry.get_fields_by_category("ambulatory", "milestone")

        # Get all fields for a disease
        dmd_fields = registry.get_fields_for_disease("DMD")
    """

    def __init__(self, data_dict_path: str, config_dir: str = "config"):
        """
        Initialize Field Registry.

        Args:
            data_dict_path: Path to datadictionary.parquet (source of truth)
            config_dir: Path to config directory (default: "config")
        """
        self.data_dict_path = Path(data_dict_path)
        self.config_dir = Path(config_dir)

        # Load data dictionary (source of truth)
        self.data_dict = self._load_data_dictionary()

        # Load enhancement metadata
        self.categories = self._load_categories()
        self.validation = self._load_validation()

        # Build field index for fast lookup
        self._build_field_index()

        print(f"FieldRegistry initialized:")
        print(f"  - {len(self.data_dict)} fields in data dictionary")
        print(f"  - {len(self.categories.get('field_categories', {}))} field categories")
        print(f"  - {len(self.validation.get('field_validation', {}))} fields with validation rules")

    def _load_data_dictionary(self) -> pd.DataFrame:
        """Load data dictionary from parquet file."""
        if not self.data_dict_path.exists():
            raise FileNotFoundError(
                f"Data dictionary not found at: {self.data_dict_path}\n"
                f"Please ensure datadictionary.parquet exists."
            )

        df = pd.read_parquet(self.data_dict_path)

        # Standardize column names (handle variations)
        column_mapping = {
            'variable': 'Variable',
            'Variable': 'Variable',
            'Field Name': 'Variable',  # MOVR data dictionary uses this
            'field_name': 'Variable',
            'description': 'Description',
            'Description': 'Description',
            'disease': 'Disease',
            'Disease': 'Disease',
            'crf': 'CRF',
            'CRF': 'CRF'
        }

        df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})

        # Ensure Variable column exists
        if 'Variable' not in df.columns:
            raise ValueError(
                f"Could not find variable/field name column in data dictionary.\n"
                f"Available columns: {df.columns.tolist()}"
            )

        return df

    def _load_yaml(self, filename: str) -> Dict:
        """Load YAML file with error handling."""
        filepath = self.config_dir / filename

        if not filepath.exists():
            warnings.warn(f"Metadata file not found: {filepath}. Using empty dict.")
            return {}

        with open(filepath, 'r') as f:
            return yaml.safe_load(f) or {}

    def _load_categories(self) -> Dict:
        """Load field categories metadata."""
        return self._load_yaml("field_metadata/categories.yaml")

    def _load_validation(self) -> Dict:
        """Load validation rules metadata."""
        return self._load_yaml("field_metadata/validation_rules.yaml")

    def _build_field_index(self):
        """Build fast lookup index for fields."""
        # Create lowercase field name mapping for case-insensitive lookup
        self.field_index = {
            row['Variable'].lower(): idx
            for idx, row in self.data_dict.iterrows()
            if 'Variable' in row
        }

    def get_field_info(self, field_name: str, include_validation: bool = True) -> Dict:
        """
        Get complete information about a field.

        Combines data dictionary info with enhancement metadata.

        Args:
            field_name: Field name to look up
            include_validation: Include validation rules if available

        Returns:
            Dictionary with all field information

        Raises:
            ValueError: If field not found in data dictionary
        """
        # Case-insensitive lookup
        field_lower = field_name.lower()

        if field_lower not in self.field_index:
            raise ValueError(
                f"Field '{field_name}' not found in data dictionary.\n"
                f"Use registry.search_fields('{field_name}') to find similar fields."
            )

        # Get base info from data dictionary
        idx = self.field_index[field_lower]
        field_row = self.data_dict.iloc[idx]
        info = field_row.to_dict()

        # Add validation rules if available
        if include_validation and field_name in self.validation.get('field_validation', {}):
            info['validation'] = self.validation['field_validation'][field_name]

        # Add categories
        info['categories'] = self._get_field_categories(field_name)

        return info

    def _get_field_categories(self, field_name: str) -> List[str]:
        """Find all categories a field belongs to."""
        categories = []

        for category, subcats in self.categories.get('field_categories', {}).items():
            for subcat, config in subcats.items():
                if field_name in config.get('fields', []):
                    categories.append(f"{category}.{subcat}")

        return categories

    def get_fields_by_category(self, category: str, subcategory: Optional[str] = None) -> List[str]:
        """
        Get all fields in a category.

        Args:
            category: Main category (e.g., "ambulatory", "cardiac")
            subcategory: Optional subcategory (e.g., "milestone", "testing")

        Returns:
            List of field names

        Example:
            # Get all ambulatory milestone fields
            fields = registry.get_fields_by_category("ambulatory", "milestone")

            # Get all ambulatory fields (all subcategories)
            fields = registry.get_fields_by_category("ambulatory")
        """
        field_cats = self.categories.get('field_categories', {})

        if category not in field_cats:
            raise ValueError(f"Category '{category}' not found. Available: {list(field_cats.keys())}")

        if subcategory:
            # Get specific subcategory
            if subcategory not in field_cats[category]:
                raise ValueError(
                    f"Subcategory '{subcategory}' not found in '{category}'. "
                    f"Available: {list(field_cats[category].keys())}"
                )
            return field_cats[category][subcategory].get('fields', [])

        else:
            # Get all fields from all subcategories
            fields = []
            for subcat_config in field_cats[category].values():
                fields.extend(subcat_config.get('fields', []))
            return list(set(fields))  # Remove duplicates

    def get_fields_for_disease(self, disease: str) -> pd.DataFrame:
        """
        Get all fields applicable to a disease.

        Queries data dictionary for disease-specific fields.

        Args:
            disease: Disease name (e.g., "DMD", "SMA")

        Returns:
            DataFrame with all fields for that disease
        """
        # Filter by disease column if it exists
        if 'Disease' in self.data_dict.columns:
            # Handle both boolean and string values
            disease_df = self.data_dict[
                (self.data_dict['Disease'] == 'Yes') |
                (self.data_dict['Disease'] == 'Y') |
                (self.data_dict['Disease'] == True)
            ].copy()
        else:
            warnings.warn("Disease column not found in data dictionary. Returning all fields.")
            disease_df = self.data_dict.copy()

        return disease_df

    def get_fields_by_crf(self, crf: str) -> pd.DataFrame:
        """
        Get all fields from a specific CRF.

        Args:
            crf: CRF name (e.g., "Diagnosis", "Encounter")

        Returns:
            DataFrame with all fields from that CRF
        """
        if 'CRF' not in self.data_dict.columns:
            raise ValueError("CRF column not found in data dictionary")

        return self.data_dict[self.data_dict['CRF'] == crf].copy()

    def search_fields(self, search_term: str, search_in: str = 'all') -> pd.DataFrame:
        """
        Search for fields by name or description.

        Args:
            search_term: Term to search for
            search_in: Where to search - 'name', 'description', or 'all'

        Returns:
            DataFrame with matching fields
        """
        search_lower = search_term.lower()

        if search_in in ['name', 'all']:
            name_matches = self.data_dict[
                self.data_dict['Variable'].str.lower().str.contains(search_lower, na=False)
            ]
        else:
            name_matches = pd.DataFrame()

        if search_in in ['description', 'all'] and 'Description' in self.data_dict.columns:
            desc_matches = self.data_dict[
                self.data_dict['Description'].str.lower().str.contains(search_lower, na=False)
            ]
        else:
            desc_matches = pd.DataFrame()

        # Combine and remove duplicates
        results = pd.concat([name_matches, desc_matches]).drop_duplicates(subset=['Variable'])

        return results.sort_values('Variable')

    def get_common_field_info(self, field_key: str) -> Dict:
        """
        Get information about cross-disease common fields.

        Args:
            field_key: Common field key (e.g., "timed_walk_test")

        Returns:
            Dictionary with common field information
        """
        common = self.categories.get('common_fields', {})

        if field_key not in common:
            raise ValueError(
                f"Common field '{field_key}' not found. "
                f"Available: {list(common.keys())}"
            )

        return common[field_key]

    def get_analysis_template_fields(self, template_name: str) -> List[str]:
        """
        Get all fields for an analysis template.

        Args:
            template_name: Template name (e.g., "natural_history", "trial_screening")

        Returns:
            List of field names in the template
        """
        templates = self.categories.get('analysis_templates', {})

        if template_name not in templates:
            raise ValueError(
                f"Analysis template '{template_name}' not found. "
                f"Available: {list(templates.keys())}"
            )

        template_config = templates[template_name]

        # Collect fields from all categories in template
        fields = []
        for category_path in template_config.get('categories', []):
            category, subcategory = category_path.split('.')
            fields.extend(self.get_fields_by_category(category, subcategory))

        return list(set(fields))  # Remove duplicates

    def validate_field_exists(self, field_name: str) -> bool:
        """Check if a field exists in data dictionary."""
        return field_name.lower() in self.field_index

    def get_validation_rules(self, field_name: str) -> Optional[Dict]:
        """Get validation rules for a field (if defined)."""
        return self.validation.get('field_validation', {}).get(field_name)

    def summary(self) -> Dict:
        """Get summary statistics about the registry."""
        return {
            'total_fields': len(self.data_dict),
            'categories': list(self.categories.get('field_categories', {}).keys()),
            'fields_with_validation': len(self.validation.get('field_validation', {})),
            'common_fields': list(self.categories.get('common_fields', {}).keys()),
            'analysis_templates': list(self.categories.get('analysis_templates', {}).keys()),
            'crfs': list(self.data_dict['CRF'].unique()) if 'CRF' in self.data_dict.columns else []
        }
