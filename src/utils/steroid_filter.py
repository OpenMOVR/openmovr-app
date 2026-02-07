#!/usr/bin/env python3
"""
Steroid Filtering Utilities

Provides functions to filter medication DataFrames based on steroid configurations.
Used for identifying patients on corticosteroid therapy for trial inclusion criteria.
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


class SteroidFilter:
    """
    Utility class for filtering medication data based on steroid configurations.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize steroid filter with configuration.
        
        Args:
            config_path: Path to steroids config YAML file
        """
        if config_path is None:
            # Default to config directory
            project_root = Path.cwd()
            if project_root.name == 'notebooks':
                project_root = project_root.parent
            config_path = project_root / 'config' / 'steroids_config.yaml'
            
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load steroid configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Steroid config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing steroid config YAML: {e}")
    
    def get_steroid_list(self, filter_type: str = 'all_steroids') -> List[str]:
        """
        Get list of steroid medications based on filter type.
        
        Args:
            filter_type: Type of filter ('dmd_systemic_only', 'all_steroids', 'oral_only')
            
        Returns:
            List of steroid medication names
        """
        
        if filter_type not in self.config['filtering_options']:
            available = list(self.config['filtering_options'].keys())
            raise ValueError(f"Unknown filter type '{filter_type}'. Available: {available}")
        
        filter_config = self.config['filtering_options'][filter_type]
        steroids = set()
        
        # Include specified categories
        include_categories = filter_config.get('include_categories', [])
        for category in include_categories:
            if category in self.config['steroids']:
                steroids.update(self.config['steroids'][category])
        
        # Exclude specified categories
        exclude_categories = filter_config.get('exclude_categories', [])
        for category in exclude_categories:
            if category in self.config['steroids']:
                exclude_set = set(self.config['steroids'][category])
                steroids = steroids - exclude_set
        
        # Exclude specific medications
        exclude_specific = filter_config.get('exclude_specific', [])
        steroids = steroids - set(exclude_specific)
        
        return sorted(list(steroids))
    
    def filter_medications(self, 
                          df: pd.DataFrame, 
                          filter_type: str = 'dmd_systemic_only',
                          field_name: str = None) -> pd.DataFrame:
        """
        Filter medication DataFrame to only include steroid medications.
        
        Args:
            df: Medication DataFrame to filter
            filter_type: Type of steroid filter to apply
            field_name: Field to match against (default from config)
            
        Returns:
            Filtered DataFrame containing only steroid medications
        """
        
        if field_name is None:
            field_name = self.config['steroids']['field_to_match']
        
        if field_name not in df.columns:
            raise ValueError(f"Field '{field_name}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        # Get steroid list
        steroid_list = self.get_steroid_list(filter_type)
        
        # Filter DataFrame
        steroid_mask = df[field_name].isin(steroid_list)
        filtered_df = df[steroid_mask].copy()
        
        print(f"🔍 STEROID FILTERING RESULTS:")
        print(f"   Filter type: {filter_type}")
        print(f"   Field matched: {field_name}")
        print(f"   Steroid medications in filter: {len(steroid_list)}")
        print(f"   Original records: {len(df):,}")
        print(f"   Steroid records found: {len(filtered_df):,}")
        print(f"   Match rate: {len(filtered_df)/len(df)*100:.1f}%")
        
        if len(filtered_df) > 0:
            unique_steroids = filtered_df[field_name].value_counts()
            print(f"\n💊 STEROIDS FOUND IN DATA:")
            for steroid, count in unique_steroids.head(10).items():
                print(f"   {steroid}: {count:,} records")
            
            if len(unique_steroids) > 10:
                print(f"   ... and {len(unique_steroids) - 10} more steroids")
        
        return filtered_df
    
    def get_patients_on_steroids(self, 
                                df: pd.DataFrame,
                                filter_type: str = 'dmd_systemic_only', 
                                patient_id_field: str = 'FACPATID') -> Set[str]:
        """
        Get set of patient IDs who are on steroid medications.
        
        Args:
            df: Medication DataFrame
            filter_type: Type of steroid filter to apply
            patient_id_field: Patient ID field name
            
        Returns:
            Set of patient IDs on steroid medications
        """
        
        steroid_df = self.filter_medications(df, filter_type)
        
        if patient_id_field not in steroid_df.columns:
            raise ValueError(f"Patient ID field '{patient_id_field}' not found in DataFrame")
        
        patient_ids = set(steroid_df[patient_id_field].dropna().unique())
        
        print(f"\n👥 PATIENTS ON STEROIDS:")
        print(f"   Unique patients: {len(patient_ids)}")
        
        return patient_ids
    
    def analyze_steroid_usage(self, 
                             df: pd.DataFrame,
                             filter_type: str = 'dmd_systemic_only') -> Dict[str, Any]:
        """
        Analyze steroid usage patterns in the medication data.
        
        Args:
            df: Medication DataFrame
            filter_type: Type of steroid filter to apply
            
        Returns:
            Dictionary with analysis results
        """
        
        steroid_df = self.filter_medications(df, filter_type)
        field_name = self.config['steroids']['field_to_match']
        
        analysis = {
            'total_steroid_records': len(steroid_df),
            'unique_patients': steroid_df['FACPATID'].nunique() if 'FACPATID' in steroid_df.columns else None,
            'unique_steroids': steroid_df[field_name].nunique(),
            'steroid_counts': steroid_df[field_name].value_counts().to_dict(),
            'filter_type_used': filter_type,
            'steroid_list_used': self.get_steroid_list(filter_type)
        }
        
        print(f"\n📊 STEROID USAGE ANALYSIS:")
        print(f"   Total steroid records: {analysis['total_steroid_records']:,}")
        if analysis['unique_patients']:
            print(f"   Unique patients on steroids: {analysis['unique_patients']:,}")
        print(f"   Different steroids found: {analysis['unique_steroids']}")
        
        return analysis


# Convenience functions for easy import
def load_steroid_config(config_path: Optional[Path] = None) -> SteroidFilter:
    """Load steroid filter with configuration."""
    return SteroidFilter(config_path)

def filter_steroids(df: pd.DataFrame, 
                   filter_type: str = 'dmd_systemic_only', 
                   config_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Quick function to filter DataFrame for steroid medications.
    
    Args:
        df: Medication DataFrame to filter
        filter_type: Type of steroid filter ('dmd_systemic_only', 'all_steroids', 'oral_only')
        config_path: Path to steroid config file
        
    Returns:
        Filtered DataFrame containing only steroid medications
    """
    steroid_filter = SteroidFilter(config_path)
    return steroid_filter.filter_medications(df, filter_type)

def get_steroid_patients(df: pd.DataFrame, 
                        filter_type: str = 'dmd_systemic_only',
                        config_path: Optional[Path] = None) -> Set[str]:
    """
    Quick function to get patient IDs on steroid medications.
    
    Args:
        df: Medication DataFrame
        filter_type: Type of steroid filter
        config_path: Path to steroid config file
        
    Returns:
        Set of patient IDs on steroid medications
    """
    steroid_filter = SteroidFilter(config_path)
    return steroid_filter.get_patients_on_steroids(df, filter_type)