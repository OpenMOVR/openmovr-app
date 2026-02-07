#!/usr/bin/env python3
"""
Business Analytics Utilities for MOVR Clinical Analytics

Streamlined utilities specifically designed for common business use cases:
- Cohort building and participant counts
- Summary statistics and cross-tabulations
- Data export with privacy controls (date shifting, de-identification)
- Standardized reporting formats for academia and industry

Usage in notebooks:
    from src.analytics.business import BusinessAnalyzer
    
    # Quick setup
    analyzer = BusinessAnalyzer()
    
    # Get participant counts
    counts = analyzer.get_participant_counts()
    
    # Export cohort data
    analyzer.export_cohort_data('DMD', include_tables=['demographics', 'medications'])
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from pathlib import Path
import warnings

from .cohorts import MOVRCohortManager
from .filters import filter_disease
from ..data_processing.loader import get_loader


class BusinessAnalyzer:
    """
    Streamlined business analytics for academia and industry requests.
    
    Focuses on common patterns:
    - Participant counts by demographics/diagnosis
    - Summary statistics and cross-tabs
    - Data export with privacy controls
    - Standardized report formats
    """
    
    def __init__(self):
        self.cohort_manager = MOVRCohortManager()
        self.loader = get_loader()
        self._base_cohort = None
        
    def get_participant_counts(self, 
                             by_disease: bool = True,
                             by_demographics: Optional[List[str]] = None,
                             export_csv: bool = False,
                             output_dir: str = "output") -> pd.DataFrame:
        """
        Get comprehensive participant counts across dimensions.
        
        Args:
            by_disease: Include disease breakdown
            by_demographics: List of demographic fields to cross-tabulate
            export_csv: Save results to CSV
            output_dir: Directory for output files
            
        Returns:
            DataFrame with participant counts
        """
        
        if self._base_cohort is None:
            self._base_cohort = self.cohort_manager.get_base_cohort()
            
        results = []
        
        # Overall counts
        total_count = self._base_cohort['count']
        results.append({
            'category': 'Total',
            'subcategory': 'All Participants',
            'count': total_count,
            'percentage': 100.0
        })
        
        # Disease counts
        if by_disease:
            # Use the disease counts from cohort manager
            disease_counts_df = self.cohort_manager.get_disease_counts()
            
            for _, row in disease_counts_df.iterrows():
                results.append({
                    'category': 'Disease',
                    'subcategory': row['disease'],
                    'count': row['patient_count'],
                    'percentage': row['percentage']
                })
        
        # Demographics cross-tabs
        if by_demographics:
            demographics = self._base_cohort['demographics']
            for field in by_demographics:
                if field in demographics.columns:
                    counts = demographics[field].value_counts()
                    for value, count in counts.items():
                        results.append({
                            'category': f'Demographics_{field}',
                            'subcategory': str(value),
                            'count': count,
                            'percentage': round(count / total_count * 100, 1)
                        })
        
        df_results = pd.DataFrame(results)
        
        if export_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"participant_counts_{timestamp}.csv"
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(exist_ok=True)
            df_results.to_csv(output_path, index=False)
            print(f"📊 Participant counts exported to: {output_path}")
            
        return df_results
    
    def get_disease_summary(self, disease: str, 
                          include_demographics: bool = True,
                          include_clinical: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive summary for a specific disease cohort.
        
        Args:
            disease: Disease code (e.g., 'DMD', 'SMA')
            include_demographics: Include demographic summaries
            include_clinical: Include clinical measure summaries
            
        Returns:
            Dictionary with comprehensive disease summary
        """
        
        # Get disease cohort
        cohort = self.cohort_manager.get_disease_cohort(disease)
        
        summary = {
            'disease': disease,
            'total_participants': cohort['count'],
            'generation_date': datetime.now().isoformat(),
            'demographics': {},
            'clinical': {},
            'data_availability': {}
        }
        
        # Demographics summary
        if include_demographics:
            demographics = cohort['demographics']
            
            # Age statistics (if available)
            age_fields = [col for col in demographics.columns if 'age' in col.lower()]
            for field in age_fields[:3]:  # Limit to first 3 age fields
                if demographics[field].notna().sum() > 0:
                    summary['demographics'][field] = {
                        'mean': float(demographics[field].mean()),
                        'median': float(demographics[field].median()),
                        'std': float(demographics[field].std()),
                        'min': float(demographics[field].min()),
                        'max': float(demographics[field].max()),
                        'missing_count': int(demographics[field].isna().sum())
                    }
            
            # Categorical summaries
            categorical_fields = demographics.select_dtypes(include=['object']).columns
            for field in categorical_fields[:5]:  # Limit to first 5 categorical fields
                value_counts = demographics[field].value_counts()
                summary['demographics'][field] = {
                    'unique_values': len(value_counts),
                    'top_values': value_counts.head(3).to_dict(),
                    'missing_count': int(demographics[field].isna().sum())
                }
        
        # Data availability across tables
        patient_ids = cohort['patient_ids']
        available_tables = self._check_data_availability(patient_ids)
        summary['data_availability'] = available_tables
        
        return summary
    
    def export_cohort_data(self, disease: str,
                          include_tables: Optional[List[str]] = None,
                          apply_date_shift: bool = False,
                          date_shift_days: int = None,
                          export_format: str = 'csv',
                          output_dir: str = "output") -> Dict[str, str]:
        """
        Export complete cohort data for sharing with academia/industry.
        
        Args:
            disease: Disease code to export
            include_tables: Specific tables to include (default: all available)
            apply_date_shift: Apply random date shifting for privacy
            date_shift_days: Maximum days to shift (random within range)
            export_format: 'csv', 'parquet', or 'excel'
            output_dir: Output directory
            
        Returns:
            Dictionary with exported file paths
        """
        
        # Get disease cohort
        cohort = self.cohort_manager.get_disease_cohort(disease)
        patient_ids = cohort['patient_ids']
        
        # Determine tables to export
        if include_tables is None:
            include_tables = ['demographics', 'diagnosis', 'encounters', 'medications', 'hospitalizations']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(output_dir) / f"{disease.lower()}_cohort_export_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export core tables
        for table_type in include_tables:
            try:
                if table_type == 'demographics':
                    data = cohort['demographics']
                elif table_type == 'diagnosis':
                    data = cohort['diagnosis']
                elif table_type == 'encounters':
                    data = cohort['encounters']
                else:
                    # Load additional tables
                    table_mapping = {
                        'medications': 'Encounter_Medication',
                        'hospitalizations': 'Log_Hospitalization',
                        'surgeries': 'Log_Surgery',
                        'devices': 'Log_AssistiveDevice'
                    }
                    
                    if table_type in table_mapping:
                        from .cohorts import apply_to_table
                        data = apply_to_table(table_mapping[table_type], patient_ids)
                    else:
                        continue
                
                if data is not None and len(data) > 0:
                    # Apply date shifting if requested
                    if apply_date_shift:
                        data = self._apply_date_shift(data, date_shift_days)
                    
                    # Export in requested format
                    filename = f"{disease.lower()}_{table_type}"
                    if export_format == 'csv':
                        file_path = output_path / f"{filename}.csv"
                        data.to_csv(file_path, index=False)
                    elif export_format == 'parquet':
                        file_path = output_path / f"{filename}.parquet"
                        data.to_parquet(file_path, index=False)
                    elif export_format == 'excel':
                        file_path = output_path / f"{filename}.xlsx"
                        data.to_excel(file_path, index=False)
                    
                    exported_files[table_type] = str(file_path)
                    print(f"📁 Exported {table_type}: {len(data):,} records to {file_path}")
                    
            except Exception as e:
                print(f"⚠️ Could not export {table_type}: {str(e)}")
                continue
        
        # Create summary report
        summary = self.get_disease_summary(disease)
        summary_file = output_path / f"{disease.lower()}_cohort_summary.json"
        
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        exported_files['summary'] = str(summary_file)
        print(f"📋 Cohort summary exported to: {summary_file}")
        
        return exported_files
    
    def _apply_date_shift(self, data: pd.DataFrame, 
                         max_shift_days: Optional[int] = None) -> pd.DataFrame:
        """Apply random date shifting to protect patient privacy."""
        
        if max_shift_days is None:
            max_shift_days = 365  # Default 1 year max shift
            
        data_shifted = data.copy()
        date_columns = []
        
        # Find date columns
        for col in data.columns:
            if 'dt' in col.lower() or 'date' in col.lower():
                if data[col].dtype == 'datetime64[ns]' or 'dt' in str(data[col].dtype):
                    date_columns.append(col)
        
        # Apply consistent shift per patient
        if 'FACPATID' in data.columns and date_columns:
            for patient_id in data['FACPATID'].unique():
                # Generate consistent random shift for this patient
                np.random.seed(hash(str(patient_id)) % 2**32)
                shift_days = np.random.randint(-max_shift_days, max_shift_days + 1)
                shift_delta = timedelta(days=shift_days)
                
                # Apply to all date columns for this patient
                patient_mask = data_shifted['FACPATID'] == patient_id
                for col in date_columns:
                    data_shifted.loc[patient_mask, col] = pd.to_datetime(data_shifted.loc[patient_mask, col]) + shift_delta
        
        return data_shifted
    
    def _check_data_availability(self, patient_ids: List[str]) -> Dict[str, int]:
        """Check data availability across tables for given patient IDs."""
        
        availability = {}
        
        # Standard tables to check
        tables_to_check = [
            'Encounter_Medication',
            'Log_Hospitalization', 
            'Log_Surgery',
            'Encounter_AssistiveDevice',
            'Log_AssistiveDevice'
        ]
        
        for table in tables_to_check:
            try:
                from .cohorts import apply_to_table
                data = apply_to_table(table, patient_ids)
                availability[table] = len(data) if data is not None else 0
            except:
                availability[table] = 0
                
        return availability


# Convenience functions for quick notebook usage
def quick_participant_counts(by_disease: bool = True, 
                           export_csv: bool = False) -> pd.DataFrame:
    """Quick participant counts for notebook usage."""
    analyzer = BusinessAnalyzer()
    return analyzer.get_participant_counts(by_disease=by_disease, export_csv=export_csv)

def quick_disease_summary(disease: str) -> Dict[str, Any]:
    """Quick disease summary for notebook usage."""
    analyzer = BusinessAnalyzer()
    return analyzer.get_disease_summary(disease)

def quick_export(disease: str, tables: Optional[List[str]] = None) -> Dict[str, str]:
    """Quick data export for notebook usage."""
    analyzer = BusinessAnalyzer()
    return analyzer.export_cohort_data(disease, include_tables=tables)