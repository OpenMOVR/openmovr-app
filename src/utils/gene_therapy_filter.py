#!/usr/bin/env python3
"""
Gene Therapy Filtering Utilities

Provides functions to filter medication DataFrames based on gene therapy configurations.
Organized by 7 disease indications: DMD, BMD, ALS, SMA, LGMD, FSHD, Pompe
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Set


class GeneTherapyFilter:
    """
    Utility class for filtering medication data based on gene therapy configurations.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize gene therapy filter with configuration.
        
        Args:
            config_path: Path to gene therapy config YAML file
        """
        if config_path is None:
            # Default to config directory
            project_root = Path.cwd()
            if project_root.name == 'notebooks':
                project_root = project_root.parent
            config_path = project_root / 'config' / 'gene_therapy_config.yaml'
            
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load gene therapy configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Gene therapy config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing gene therapy config YAML: {e}")
    
    def get_therapy_list(self, 
                        diseases: Optional[List[str]] = None,
                        filter_type: str = 'all_advanced_therapies') -> List[str]:
        """
        Get list of gene therapy medications based on diseases and filter type.
        
        Args:
            diseases: List of diseases to include (e.g., ['DMD', 'SMA'])
            filter_type: Type of filter from filtering_options
            
        Returns:
            List of gene therapy medication names
        """
        
        therapies = set()
        
        # If specific diseases requested, use those
        if diseases:
            for disease in diseases:
                if disease in self.config['gene_therapy_medications']:
                    disease_config = self.config['gene_therapy_medications'][disease]
                    # Add all therapy types for this disease
                    for category, medications in disease_config.items():
                        if isinstance(medications, list):
                            therapies.update(medications)
        
        # Otherwise use filter_type configuration
        elif filter_type in self.config['filtering_options']:
            filter_config = self.config['filtering_options'][filter_type]
            
            # Get diseases to include
            target_diseases = filter_config.get('diseases', 
                             list(self.config['gene_therapy_medications'].keys()))
            
            # Get categories to include
            include_categories = filter_config.get('include_categories', [])
            
            for disease in target_diseases:
                if disease in self.config['gene_therapy_medications']:
                    disease_config = self.config['gene_therapy_medications'][disease]
                    
                    for category in include_categories:
                        if category in disease_config and isinstance(disease_config[category], list):
                            therapies.update(disease_config[category])
        
        else:
            available = list(self.config['filtering_options'].keys())
            raise ValueError(f"Unknown filter type '{filter_type}'. Available: {available}")
        
        return sorted(list(therapies))
    
    def get_therapies_by_disease(self, disease: str) -> Dict[str, List[str]]:
        """
        Get all therapy categories for a specific disease.
        
        Args:
            disease: Disease code (DMD, SMA, etc.)
            
        Returns:
            Dictionary with therapy categories and medications
        """
        
        if disease not in self.config['gene_therapy_medications']:
            available = list(self.config['gene_therapy_medications'].keys())
            raise ValueError(f"Unknown disease '{disease}'. Available: {available}")
        
        disease_config = self.config['gene_therapy_medications'][disease]
        
        # Extract only lists (medication categories)
        therapies = {}
        for category, medications in disease_config.items():
            if isinstance(medications, list):
                therapies[category] = medications
        
        return therapies
    
    def filter_medications(self, 
                          df: pd.DataFrame, 
                          diseases: Optional[List[str]] = None,
                          filter_type: str = 'all_advanced_therapies',
                          field_name: str = None) -> pd.DataFrame:
        """
        Filter medication DataFrame to only include gene therapy medications.
        
        Args:
            df: Medication DataFrame to filter
            diseases: Specific diseases to include
            filter_type: Type of gene therapy filter to apply
            field_name: Field to match against (default from config)
            
        Returns:
            Filtered DataFrame containing only gene therapy medications
        """
        
        if field_name is None:
            field_name = self.config['gene_therapy_medications']['field_to_match']
        
        if field_name not in df.columns:
            raise ValueError(f"Field '{field_name}' not found in DataFrame. Available columns: {list(df.columns)}")
        
        # Get therapy list
        therapy_list = self.get_therapy_list(diseases, filter_type)
        
        # Filter DataFrame (case-insensitive matching)
        therapy_mask = df[field_name].str.lower().isin([t.lower() for t in therapy_list])
        filtered_df = df[therapy_mask].copy()
        
        print(f"🧬 GENE THERAPY FILTERING RESULTS:")
        print(f"   Filter type: {filter_type}")
        print(f"   Diseases: {diseases if diseases else 'All configured'}")
        print(f"   Field matched: {field_name}")
        print(f"   Therapies in filter: {len(therapy_list)}")
        print(f"   Original records: {len(df):,}")
        print(f"   Gene therapy records found: {len(filtered_df):,}")
        print(f"   Match rate: {len(filtered_df)/len(df)*100:.1f}%")
        
        if len(filtered_df) > 0:
            unique_therapies = filtered_df[field_name].value_counts()
            print(f"\n🧬 GENE THERAPIES FOUND IN DATA:")
            for therapy, count in unique_therapies.head(10).items():
                print(f"   {therapy}: {count:,} records")
            
            if len(unique_therapies) > 10:
                print(f"   ... and {len(unique_therapies) - 10} more therapies")
        
        return filtered_df
    
    def get_patients_on_gene_therapy(self, 
                                    df: pd.DataFrame,
                                    diseases: Optional[List[str]] = None,
                                    filter_type: str = 'all_advanced_therapies',
                                    patient_id_field: str = 'FACPATID') -> Set[str]:
        """
        Get set of patient IDs who are on gene therapy medications.
        
        Args:
            df: Medication DataFrame
            diseases: Specific diseases to include
            filter_type: Type of gene therapy filter to apply
            patient_id_field: Patient ID field name
            
        Returns:
            Set of patient IDs on gene therapy medications
        """
        
        therapy_df = self.filter_medications(df, diseases, filter_type)
        
        if patient_id_field not in therapy_df.columns:
            raise ValueError(f"Patient ID field '{patient_id_field}' not found in DataFrame")
        
        patient_ids = set(therapy_df[patient_id_field].dropna().unique())
        
        print(f"\n👥 PATIENTS ON GENE THERAPY:")
        print(f"   Unique patients: {len(patient_ids)}")
        
        return patient_ids
    
    def analyze_by_disease(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze gene therapy usage by disease indication.
        
        Args:
            df: Medication DataFrame
            
        Returns:
            Dictionary with analysis by disease
        """
        
        analysis = {}
        
        for disease in self.config['gene_therapy_medications'].keys():
            if disease == 'description' or disease == 'field_to_match':
                continue
                
            try:
                # Filter for this disease specifically
                disease_therapies = self.filter_medications(df, diseases=[disease])
                
                analysis[disease] = {
                    'total_records': len(disease_therapies),
                    'unique_patients': disease_therapies['FACPATID'].nunique() if 'FACPATID' in disease_therapies.columns else None,
                    'unique_therapies': disease_therapies[self.config['gene_therapy_medications']['field_to_match']].nunique(),
                    'therapy_counts': disease_therapies[self.config['gene_therapy_medications']['field_to_match']].value_counts().to_dict()
                }
                
            except Exception as e:
                analysis[disease] = {
                    'error': str(e),
                    'total_records': 0,
                    'unique_patients': 0,
                    'unique_therapies': 0,
                    'therapy_counts': {}
                }
        
        print(f"\n📊 GENE THERAPY ANALYSIS BY DISEASE:")
        for disease, data in analysis.items():
            if 'error' not in data:
                print(f"   {disease}: {data['total_records']} records, {data.get('unique_patients', 0)} patients")
        
        return analysis


# Convenience functions for easy import
def load_gene_therapy_config(config_path: Optional[Path] = None) -> GeneTherapyFilter:
    """Load gene therapy filter with configuration."""
    return GeneTherapyFilter(config_path)

def filter_gene_therapies(df: pd.DataFrame, 
                         diseases: Optional[List[str]] = None,
                         filter_type: str = 'all_advanced_therapies',
                         config_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Quick function to filter DataFrame for gene therapy medications.
    
    Args:
        df: Medication DataFrame to filter
        diseases: Specific diseases to include (e.g., ['DMD', 'SMA'])
        filter_type: Type of filter ('all_gene_therapy', 'dmd_advanced_therapies', etc.)
        config_path: Path to gene therapy config file
        
    Returns:
        Filtered DataFrame containing only gene therapy medications
    """
    gene_filter = GeneTherapyFilter(config_path)
    return gene_filter.filter_medications(df, diseases, filter_type)

def get_gene_therapy_patients(df: pd.DataFrame, 
                             diseases: Optional[List[str]] = None,
                             filter_type: str = 'all_advanced_therapies',
                             config_path: Optional[Path] = None) -> Set[str]:
    """
    Quick function to get patient IDs on gene therapy medications.
    
    Args:
        df: Medication DataFrame
        diseases: Specific diseases to include
        filter_type: Type of filter
        config_path: Path to gene therapy config file
        
    Returns:
        Set of patient IDs on gene therapy medications
    """
    gene_filter = GeneTherapyFilter(config_path)
    return gene_filter.get_patients_on_gene_therapy(df, diseases, filter_type)