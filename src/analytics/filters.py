#!/usr/bin/env python3
"""
Common data filtering utilities for MOVR analytics notebooks.

These filters provide standardized ways to subset MOVR data for analysis:
- Enrollment validation (requires demographics + diagnosis + encounters)
- MOVR study filtering (2019+ enrollment or missing enrollment date)  
- Disease-specific filtering (DMD, BMD, SMA, etc.)
- Age-based cohort filtering

Usage in notebooks:
    from src.analytics.filters import *
    
    # Apply filters
    validated = filter_validated_enrollment(demographics, diagnosis, encounters)
    movr_only = filter_movr_study(validated['demographics'])
    dmd_patients = filter_disease(validated['diagnosis'], 'DMD')
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import warnings

def filter_validated_enrollment(
    demographics: pd.DataFrame, 
    diagnosis: pd.DataFrame, 
    encounters: pd.DataFrame,
    patient_id_col: str = 'FACPATID'
) -> Dict[str, pd.DataFrame]:
    """
    Filter datasets to include only patients with complete enrollment.
    
    Complete enrollment requires presence in all three core datasets:
    - Demographics_MainData
    - Diagnosis_MainData  
    - Encounter_MainData
    
    Args:
        demographics: Demographics dataframe
        diagnosis: Diagnosis dataframe
        encounters: Encounters dataframe
        patient_id_col: Column name for patient ID (default: FACPATID)
        
    Returns:
        Dictionary with filtered datasets:
        {
            'demographics': filtered demographics,
            'diagnosis': filtered diagnosis, 
            'encounters': filtered encounters,
            'patient_ids': list of validated patient IDs,
            'stats': enrollment statistics
        }
    """
    
    # Get patient IDs from each dataset
    demo_patients = set(demographics[patient_id_col].dropna())
    diag_patients = set(diagnosis[patient_id_col].dropna())
    enc_patients = set(encounters[patient_id_col].dropna())
    
    # Find intersection (patients in all three datasets)
    validated_patients = demo_patients & diag_patients & enc_patients
    
    # Filter datasets
    filtered_demographics = demographics[demographics[patient_id_col].isin(validated_patients)]
    filtered_diagnosis = diagnosis[diagnosis[patient_id_col].isin(validated_patients)]
    filtered_encounters = encounters[encounters[patient_id_col].isin(validated_patients)]
    
    # Generate statistics
    stats = {
        'total_demographics': len(demo_patients),
        'total_diagnosis': len(diag_patients),
        'total_encounters': len(enc_patients),
        'validated_enrollment': len(validated_patients),
        'enrollment_rate': len(validated_patients) / max(len(demo_patients), 1) * 100,
        'excluded_patients': {
            'missing_diagnosis': len(demo_patients - diag_patients),
            'missing_encounters': len(demo_patients - enc_patients),
            'missing_demographics': len((diag_patients | enc_patients) - demo_patients)
        }
    }
    
    print(f"📊 Enrollment Validation Results:")
    print(f"   Demographics patients: {stats['total_demographics']:,}")
    print(f"   Diagnosis patients: {stats['total_diagnosis']:,}")  
    print(f"   Encounters patients: {stats['total_encounters']:,}")
    print(f"   ✅ Validated enrollment: {stats['validated_enrollment']:,} ({stats['enrollment_rate']:.1f}%)")
    print(f"   📋 Filtered datasets:")
    print(f"      Demographics: {len(filtered_demographics):,} records")
    print(f"      Diagnosis: {len(filtered_diagnosis):,} records")
    print(f"      Encounters: {len(filtered_encounters):,} records")
    
    return {
        'demographics': filtered_demographics,
        'diagnosis': filtered_diagnosis,
        'encounters': filtered_encounters,
        'patient_ids': list(validated_patients),
        'stats': stats
    }


def filter_movr_study(
    demographics: pd.DataFrame,
    usndr_col: str = None,  # Auto-detect if None
    patient_id_col: str = 'FACPATID',
    **kwargs  # Accept old parameters for backward compatibility
) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
    """
    Filter demographics to MOVR study patients only.
    
    MOVR study criteria:
    - USNDR field is False/No/0 (not USNDR legacy study), OR
    - USNDR field is missing/NaN (treated as MOVR study)
    
    Logic: If USNDR is True -> Legacy study (exclude)
           If USNDR is False/missing -> MOVR study (include)
    
    Args:
        demographics: Demographics dataframe
        usndr_col: Column name for USNDR flag (auto-detect if None)
        patient_id_col: Column name for patient ID (default: FACPATID)
        **kwargs: Backward compatibility (ignored)
        
    Returns:
        Dictionary with:
        {
            'demographics': filtered demographics,
            'patient_ids': list of MOVR patient IDs,
            'stats': filtering statistics
        }
    """
    
    # Auto-detect USNDR column if not specified
    if usndr_col is None:
        usndr_candidates = [col for col in demographics.columns if 'usndr' in col.lower()]
        if usndr_candidates:
            usndr_col = usndr_candidates[0]  # Use first match
            print(f"🔍 Auto-detected USNDR column: '{usndr_col}'")
        else:
            usndr_col = 'USNDR'  # Default fallback
    
    # Handle missing USNDR column
    if usndr_col not in demographics.columns:
        print(f"⚠️  Warning: USNDR column '{usndr_col}' not found.")
        print(f"   Available columns with 'usndr': {[col for col in demographics.columns if 'usndr' in col.lower()]}")
        # Return all patients as MOVR (conservative approach)
        return {
            'demographics': demographics,
            'patient_ids': demographics[patient_id_col].dropna().tolist(),
            'stats': {
                'total_patients': len(demographics),
                'movr_patients': len(demographics),
                'usndr_patients': 0,
                'missing_usndr': len(demographics),
                'usndr_column_found': False,
                'movr_rate': 100.0
            }
        }
    
    demographics = demographics.copy()
    usndr_series = demographics[usndr_col]
    
    # Handle different USNDR value formats
    # Convert to boolean: True means USNDR (legacy), False/missing means MOVR
    def is_usndr_legacy(value):
        if pd.isna(value):
            return False  # Missing = MOVR study
        
        # Convert to string and check
        str_val = str(value).lower().strip()
        
        # True values (USNDR legacy study)
        if str_val in ['true', '1', '1.0', 'yes', 'y']:
            return True
        
        # False values (MOVR study)  
        if str_val in ['false', '0', '0.0', 'no', 'n', '']:
            return False
            
        # Default: treat unknown values as MOVR
        return False
    
    # Apply USNDR logic
    usndr_flags = usndr_series.apply(is_usndr_legacy)
    
    # MOVR study filter: NOT USNDR (i.e., USNDR is False or missing)
    movr_mask = ~usndr_flags
    
    # Apply filter
    movr_demographics = demographics[movr_mask]
    movr_patient_ids = movr_demographics[patient_id_col].dropna().tolist()
    
    # Generate statistics
    usndr_count = usndr_flags.sum()
    missing_usndr = usndr_series.isna().sum()
    
    stats = {
        'total_patients': len(demographics),
        'movr_patients': len(movr_demographics),
        'usndr_patients': usndr_count,
        'missing_usndr': missing_usndr,
        'usndr_column_found': True,
        'movr_rate': len(movr_demographics) / len(demographics) * 100 if len(demographics) > 0 else 0
    }
    
    print(f"🎯 MOVR Study Filter Results (USNDR-based):")
    print(f"   Total patients: {stats['total_patients']:,}")
    print(f"   ✅ MOVR study patients: {stats['movr_patients']:,} ({stats['movr_rate']:.1f}%)")
    print(f"   📜 USNDR legacy patients: {stats['usndr_patients']:,}")
    print(f"   ❓ Missing USNDR field: {stats['missing_usndr']:,}")
    
    return {
        'demographics': movr_demographics,
        'patient_ids': movr_patient_ids,
        'stats': stats
    }


def filter_disease(
    diagnosis: pd.DataFrame,
    disease: str,
    patient_id_col: str = 'FACPATID'
) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
    """
    Filter diagnosis data for specific disease.
    
    Searches for disease-specific columns and identifies patients with that diagnosis.
    
    Args:
        diagnosis: Diagnosis dataframe
        disease: Disease code (e.g., 'DMD', 'BMD', 'SMA', 'ALS')
        patient_id_col: Column name for patient ID (default: FACPATID)
        
    Returns:
        Dictionary with:
        {
            'diagnosis': filtered diagnosis dataframe,
            'patient_ids': list of patients with disease,
            'disease_columns': list of columns used for filtering,
            'stats': filtering statistics
        }
    """
    
    disease_lower = disease.lower()
    
    # Create a mapping for known disease variations
    disease_mappings = {
        'pompe': ['pom', 'pompe'],
        'pom': ['pom', 'pompe'],
        'lgmd': ['lgmd'],
        'dmd': ['dmd'],
        'bmd': ['bmd'], 
        'sma': ['sma'],
        'als': ['als'],
        'fshd': ['fshd']
    }
    
    # Get possible patterns for this disease
    search_patterns = disease_mappings.get(disease_lower, [disease_lower])
    
    # Find disease-specific columns
    disease_columns = []
    for pattern in search_patterns:
        pattern_cols = [col for col in diagnosis.columns if pattern in col.lower()]
        disease_columns.extend(pattern_cols)
    
    # Remove duplicates
    disease_columns = list(set(disease_columns))
    
    # Also check dstype field if available
    dstype_patients = []
    dstype_used = False
    if 'dstype' in diagnosis.columns:
        # Map common disease codes to dstype values
        dstype_mappings = {
            'pompe': 'Pompe',
            'pom': 'Pompe', 
            'dmd': 'DMD',
            'bmd': 'BMD',
            'sma': 'SMA', 
            'als': 'ALS',
            'fshd': 'FSHD',
            'lgmd': 'LGMD'
        }
        
        dstype_value = dstype_mappings.get(disease_lower)
        if dstype_value:
            dstype_matches = diagnosis[diagnosis['dstype'] == dstype_value]
            dstype_patients = dstype_matches[patient_id_col].dropna().unique().tolist()
            dstype_used = True
            print(f"🎯 Found {len(dstype_patients)} patients via dstype='{dstype_value}'")
    
    if not disease_columns and not dstype_patients:
        print(f"⚠️  Warning: No columns or dstype matches found for disease '{disease}'")
        print(f"   Available disease patterns in columns:")
        disease_patterns = set()
        for col in diagnosis.columns:
            for pattern in ['dmd', 'bmd', 'sma', 'als', 'lgmd', 'fshd', 'pom']:
                if pattern in col.lower():
                    disease_patterns.add(pattern.upper())
        print(f"   {sorted(disease_patterns)}")
        
        if 'dstype' in diagnosis.columns:
            dstype_values = diagnosis['dstype'].value_counts()
            print(f"   Available dstype values: {dict(dstype_values)}")
        
        return {
            'diagnosis': pd.DataFrame(),
            'patient_ids': [],
            'disease_columns': [],
            'stats': {
                'total_patients': len(diagnosis[patient_id_col].dropna().unique()),
                'disease_patients': 0,
                'disease_columns_found': 0,
                'columns_used': []
            }
        }

    print(f"🔍 Found {len(disease_columns)} {disease} columns: {disease_columns[:10]}{'...' if len(disease_columns) > 10 else ''}")
    
    # Combine both approaches: column-based and dstype-based filtering
    all_patient_ids = set()
    
    # Column-based filtering
    if disease_columns:
        # Check for non-null AND non-empty-string values (handles string-converted columns)
        disease_cols_df = diagnosis[disease_columns]
        disease_mask = disease_cols_df.apply(
            lambda col: col.notna() & (col.astype(str).str.strip() != '')
        ).any(axis=1)
        disease_diagnosis_cols = diagnosis[disease_mask]
        column_patient_ids = disease_diagnosis_cols[patient_id_col].dropna().unique()
        all_patient_ids.update(column_patient_ids)
        print(f"📋 Found {len(column_patient_ids)} patients via disease columns")
    
    # dstype-based filtering
    if dstype_patients:
        all_patient_ids.update(dstype_patients)
        print(f"📋 Found {len(dstype_patients)} patients via dstype field")
    
    # Get final patient list and filtered diagnosis
    disease_patient_ids = list(all_patient_ids)
    disease_diagnosis = diagnosis[diagnosis[patient_id_col].isin(disease_patient_ids)]
    
    # Generate statistics
    total_patients = len(diagnosis[patient_id_col].dropna().unique())
    disease_patients = len(disease_patient_ids)
    
    stats = {
        'total_patients': total_patients,
        'disease_patients': disease_patients,
        'disease_rate': disease_patients / total_patients * 100 if total_patients > 0 else 0,
        'disease_columns_found': len(disease_columns),
        'columns_used': disease_columns,
        'disease_records': len(disease_diagnosis)
    }
    
    print(f"🎯 {disease} Disease Filter Results:")
    print(f"   Total patients in diagnosis: {stats['total_patients']:,}")
    print(f"   ✅ {disease} patients: {stats['disease_patients']:,} ({stats['disease_rate']:.1f}%)")
    print(f"   📋 {disease} diagnosis records: {stats['disease_records']:,}")
    print(f"   🔍 Disease columns used: {stats['disease_columns_found']}")
    
    return {
        'diagnosis': disease_diagnosis,
        'patient_ids': disease_patient_ids,
        'disease_columns': disease_columns,
        'stats': stats
    }


def apply_patient_filter_to_datasets(
    datasets: Dict[str, pd.DataFrame],
    patient_ids: List[str],
    patient_id_col: str = 'FACPATID'
) -> Dict[str, pd.DataFrame]:
    """
    Apply patient ID filter to multiple datasets.
    
    Args:
        datasets: Dictionary of dataframes to filter
        patient_ids: List of patient IDs to keep
        patient_id_col: Column name for patient ID
        
    Returns:
        Dictionary of filtered dataframes
    """
    
    patient_set = set(patient_ids)
    filtered_datasets = {}
    
    print(f"🔄 Applying patient filter ({len(patient_ids):,} patients) to datasets:")
    
    for name, df in datasets.items():
        if patient_id_col in df.columns:
            filtered_df = df[df[patient_id_col].isin(patient_set)]
            filtered_datasets[name] = filtered_df
            print(f"   {name}: {len(df):,} → {len(filtered_df):,} records")
        else:
            print(f"   ⚠️  {name}: No {patient_id_col} column found, skipping")
            filtered_datasets[name] = df
    
    return filtered_datasets


def get_age_cohorts(
    demographics: pd.DataFrame,
    age_groups: Dict[str, Tuple[int, int]],
    age_field: Optional[str] = None,
    patient_id_col: str = 'FACPATID'
) -> Dict[str, Dict]:
    """
    Create age-based cohorts from demographics data.
    
    Args:
        demographics: Demographics dataframe
        age_groups: Dictionary of {group_name: (min_age, max_age)}
        age_field: Age column name (auto-detected if None)
        patient_id_col: Column name for patient ID
        
    Returns:
        Dictionary of age cohort information
    """
    
    # Auto-detect age field if not provided
    if age_field is None:
        age_columns = [col for col in demographics.columns if 'age' in col.lower()]
        if age_columns:
            age_field = age_columns[0]
            print(f"🎂 Using age field: {age_field}")
        else:
            print("⚠️  No age columns found. Creating dummy ages for demonstration.")
            demographics = demographics.copy()
            demographics['AGE_YEARS'] = np.random.randint(1, 20, len(demographics))
            age_field = 'AGE_YEARS'
    
    cohorts = {}
    
    print(f"👥 Creating age cohorts:")
    for group_name, (min_age, max_age) in age_groups.items():
        if max_age == 999:  # Handle "X+" groups
            mask = demographics[age_field] >= min_age
        else:
            mask = (demographics[age_field] >= min_age) & (demographics[age_field] < max_age)
        
        cohort_demographics = demographics[mask]
        patient_ids = cohort_demographics[patient_id_col].dropna().unique().tolist()
        
        cohorts[group_name] = {
            'demographics': cohort_demographics,
            'patient_ids': patient_ids,
            'patient_count': len(patient_ids),
            'age_range': (min_age, max_age if max_age != 999 else 'infinity')
        }
        
        print(f"   {group_name}: {len(patient_ids):,} patients")
    
    return cohorts


# Convenience function to apply common filter pipeline
def apply_standard_filters(
    demographics: pd.DataFrame,
    diagnosis: pd.DataFrame, 
    encounters: pd.DataFrame,
    disease: str = 'DMD',
    movr_study_only: bool = True,
    **kwargs  # Backward compatibility for old parameters
) -> Dict[str, Union[pd.DataFrame, List, Dict]]:
    """
    Apply the standard MOVR filtering pipeline:
    1. Enrollment validation (all three datasets)
    2. MOVR study filter (USNDR-based: exclude USNDR=True patients)
    3. Disease filter (specified disease)
    
    Args:
        demographics: Demographics dataframe
        diagnosis: Diagnosis dataframe
        encounters: Encounters dataframe
        disease: Disease to filter for (default: DMD)
        movr_study_only: Whether to apply MOVR study filter (default: True)
        **kwargs: Backward compatibility (ignored)
        
    Returns:
        Dictionary with all filtered datasets and patient IDs
    """
    
    print("🔄 Applying Standard MOVR Filters")
    print("="*50)
    
    # Step 1: Enrollment validation
    print("1️⃣ Enrollment Validation")
    validated = filter_validated_enrollment(demographics, diagnosis, encounters)
    
    # Step 2: MOVR study filter (if requested)
    if movr_study_only:
        print(f"\n2️⃣ MOVR Study Filter (USNDR-based)")
        movr_result = filter_movr_study(validated['demographics'])
        movr_patient_ids = movr_result['patient_ids']
    else:
        print(f"\n2️⃣ MOVR Study Filter: SKIPPED")
        movr_patient_ids = validated['patient_ids']
        movr_result = {'stats': {'movr_patients': len(movr_patient_ids)}}
    
    # Step 3: Disease filter
    print(f"\n3️⃣ {disease} Disease Filter")
    disease_result = filter_disease(validated['diagnosis'], disease)
    
    # Find intersection of MOVR patients and disease patients
    final_patient_ids = list(set(movr_patient_ids) & set(disease_result['patient_ids']))
    
    # Apply final filter to all datasets
    print(f"\n4️⃣ Final Dataset Filtering")
    final_datasets = apply_patient_filter_to_datasets(
        {
            'demographics': validated['demographics'],
            'diagnosis': validated['diagnosis'],  
            'encounters': validated['encounters']
        },
        final_patient_ids
    )
    
    # Final summary
    print(f"\n📊 Final Results Summary:")
    print(f"   ✅ Final cohort: {len(final_patient_ids):,} {disease} patients")
    if movr_study_only:
        print(f"   📅 MOVR study patients only (USNDR-based filter)")
    print(f"   📋 Final datasets:")
    for name, df in final_datasets.items():
        print(f"      {name}: {len(df):,} records")
    
    return {
        'demographics': final_datasets['demographics'],
        'diagnosis': final_datasets['diagnosis'],
        'encounters': final_datasets['encounters'],
        'patient_ids': final_patient_ids,
        'validation_stats': validated['stats'],
        'movr_stats': movr_result['stats'],
        'disease_stats': disease_result['stats'],
        'disease_columns': disease_result['disease_columns']
    }