#!/usr/bin/env python3
"""
MOVR Clinical Analytics - Cohort Management System

This module provides the foundation for all MOVR analytics workflows:
1. Standard base cohort (enrollment validated + MOVR study)
2. Disease-specific cohorts and counts
3. Easy application to any parquet table
4. Field exploration utilities for custom filtering

Common notebook usage:
    from src.analytics.cohorts import get_base_cohort, get_disease_counts, apply_to_table
    
    # Get your standard starting point
    base_cohort = get_base_cohort()
    
    # See all available diseases
    disease_counts = get_disease_counts(base_cohort)
    
    # Apply to any table
    dmd_medications = apply_to_table('Encounter_Medication', base_cohort['DMD'])
"""

import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings
from datetime import datetime, date

from ..data_processing.loader import get_loader
from .filters import filter_validated_enrollment, filter_movr_study, filter_disease, apply_patient_filter_to_datasets


def _compute_facility_info(demographics_df: pd.DataFrame) -> dict:
    """
    Compute facility counts from demographics DataFrame.

    Used by get_base_cohort, get_disease_cohort, and filter_cohort_by_patients
    to ensure consistent facility statistics across all cohort types.

    Args:
        demographics_df: Demographics DataFrame with FACILITY_DISPLAY_ID column

    Returns:
        Dictionary with total_facilities count and list of facility records
    """
    if demographics_df.empty:
        return {'total_facilities': 0, 'facilities': []}

    if 'FACILITY_DISPLAY_ID' not in demographics_df.columns:
        return {'total_facilities': 0, 'facilities': []}

    # Count unique patients per facility
    if 'FACPATID' in demographics_df.columns:
        facility_counts = demographics_df.groupby('FACILITY_DISPLAY_ID')['FACPATID'].nunique()
        facility_counts = facility_counts.reset_index(name='patient_count')
    else:
        # Fallback: count rows per facility
        facility_counts = demographics_df.groupby('FACILITY_DISPLAY_ID').size().reset_index(name='patient_count')

    facility_counts = facility_counts.sort_values('patient_count', ascending=False)

    # Add facility names if available
    if 'FACILITY_NAME' in demographics_df.columns:
        names = demographics_df[['FACILITY_DISPLAY_ID', 'FACILITY_NAME']].drop_duplicates()
        facility_counts = facility_counts.merge(names, on='FACILITY_DISPLAY_ID', how='left')

    return {
        'total_facilities': len(facility_counts),
        'facilities': facility_counts.to_dict('records')
    }


class MOVRCohortManager:
    """
    Manages MOVR clinical cohorts with caching and easy access patterns.
    
    This is your go-to class for all cohort operations in notebooks.
    """
    
    def __init__(self):
        self.loader = get_loader()
        self._base_cohort = None
        self._disease_cohorts = {}
        self._disease_counts = None
        self._available_diseases = None
        
    def get_base_cohort(self, force_refresh: bool = False, include_usndr: bool = False) -> Dict[str, Any]:
        """
        Get the standard MOVR base cohort (enrollment validated + optionally USNDR).
        
        This is your starting point for all analyses.
        Cached for performance - only loads once per session.
        
        Args:
            force_refresh: Force reload even if cached
            include_usndr: Include USNDR legacy patients (default: False, MOVR only)
            
        Returns:
            Dictionary with:
            {
                'demographics': filtered demographics dataframe,
                'diagnosis': filtered diagnosis dataframe, 
                'encounters': filtered encounters dataframe,
                'patient_ids': list of base cohort patient IDs,
                'stats': cohort statistics,
                'count': number of patients
            }
        """
        
        if self._base_cohort is None or force_refresh:
            cohort_type = "enrollment validated + MOVR + USNDR" if include_usndr else "enrollment validated + MOVR study"
            print(f"🔄 Loading MOVR base cohort ({cohort_type})...")
            
            # Load main datasets
            demographics = self.loader.load_demographics()
            diagnosis = self.loader.load_diagnosis()
            encounters = self.loader.load_encounters()
            
            # Apply enrollment validation (core datasets only)
            validated = filter_validated_enrollment(demographics, diagnosis, encounters)
            
            if include_usndr:
                # Use all validated patients (MOVR + USNDR)
                cohort_patient_ids = validated['patient_ids']
                final_datasets = {
                    'demographics': validated['demographics'],
                    'diagnosis': validated['diagnosis'],
                    'encounters': validated['encounters']
                }
                movr_result = None  # No MOVR filtering applied
            else:
                # Apply MOVR study filter (exclude USNDR)
                movr_result = filter_movr_study(validated['demographics'])
                cohort_patient_ids = movr_result['patient_ids']
                
                # Apply MOVR filter to core datasets
                final_datasets = apply_patient_filter_to_datasets(
                    {
                        'demographics': validated['demographics'],
                        'diagnosis': validated['diagnosis'],
                        'encounters': validated['encounters']
                    },
                    cohort_patient_ids
                )
            
            # Load and filter medications separately (not part of enrollment validation)
            medications = self.loader.load_medications()
            medications_filtered = apply_patient_filter_to_datasets(
                {'medications': medications}, cohort_patient_ids
            )
            final_datasets['medications'] = medications_filtered['medications']
            
            # Calculate disease type distribution
            dstype_counts = {}
            if 'dstype' in final_datasets['diagnosis'].columns:
                dstype_counts = final_datasets['diagnosis']['dstype'].value_counts().to_dict()
            
            # Calculate facility distribution using shared helper
            facility_info = _compute_facility_info(final_datasets['demographics'])
            
            self._base_cohort = {
                'demographics': final_datasets['demographics'],
                'diagnosis': final_datasets['diagnosis'],
                'encounters': final_datasets['encounters'],
                'medications': final_datasets['medications'],
                'FACPATID': cohort_patient_ids,  # Changed from patient_ids to FACPATID
                'patient_ids': cohort_patient_ids,  # Keep for backward compatibility
                'dstype_counts': dstype_counts,  # Disease type distribution
                'facility_info': facility_info,  # Facility information
                'stats': {
                    'validation_stats': validated['stats'],
                    'movr_stats': movr_result['stats'] if movr_result else None
                },
                'count': len(cohort_patient_ids),
                'include_usndr': include_usndr  # Track whether USNDR is included
            }
            
            cohort_label = "patients (MOVR + USNDR)" if include_usndr else "MOVR patients"
            print(f"✅ Base cohort ready: {len(cohort_patient_ids):,} {cohort_label}")
            
            # Display disease type distribution
            if dstype_counts:
                print(f"\n📊 DISEASE TYPE DISTRIBUTION:")
                for disease, count in sorted(dstype_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(cohort_patient_ids) * 100) if len(cohort_patient_ids) > 0 else 0
                    print(f"   {disease}: {count:,} patients ({percentage:.1f}%)")
            
            # Display facility information
            if facility_info and facility_info.get('facilities'):
                print(f"\n🏥 FACILITY DISTRIBUTION:")
                print(f"   Total facilities: {facility_info['total_facilities']}")
                print("   Top 10 facilities by patient count:")
                for facility in facility_info['facilities'][:10]:
                    print(f"   {facility['FACILITY_DISPLAY_ID']}: {facility['FACILITY_NAME']} ({facility['patient_count']:,} patients)")
                if facility_info['total_facilities'] > 10:
                    remaining = facility_info['total_facilities'] - 10
                    print(f"   ... and {remaining} more facilities")
        
        return self._base_cohort
    
    def get_disease_counts(self, base_cohort: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get patient counts for all available diseases in the cohort.
        
        Perfect for deciding which disease to analyze.
        
        Args:
            base_cohort: Base cohort dict (uses cached if None)
            
        Returns:
            DataFrame with columns: ['disease', 'patient_count', 'percentage']
        """
        
        if base_cohort is None:
            base_cohort = self.get_base_cohort()
            
        if self._disease_counts is None:
            print("🔍 Analyzing disease distribution in base cohort...")
            
            diagnosis_df = base_cohort['diagnosis']
            total_patients = len(base_cohort['patient_ids'])
            
            # Common diseases to check
            diseases = ['DMD', 'BMD', 'SMA', 'ALS', 'LGMD', 'FSHD', 'POM']
            disease_data = []
            
            for disease in diseases:
                try:
                    disease_result = filter_disease(diagnosis_df, disease)
                    patient_count = len(disease_result['patient_ids'])
                    percentage = (patient_count / total_patients * 100) if total_patients > 0 else 0
                    
                    if patient_count > 0:  # Only include diseases with patients
                        disease_data.append({
                            'disease': disease,
                            'patient_count': patient_count,
                            'percentage': round(percentage, 1),
                            'columns_found': len(disease_result['disease_columns'])
                        })
                except Exception as e:
                    # Skip diseases that cause errors
                    continue
            
            # Sort by patient count
            disease_data.sort(key=lambda x: x['patient_count'], reverse=True)
            
            self._disease_counts = pd.DataFrame(disease_data)
            
            print("📊 Disease distribution:")
            for _, row in self._disease_counts.iterrows():
                print(f"   {row['disease']}: {row['patient_count']:,} patients ({row['percentage']}%)")
        
        return self._disease_counts
    
    def get_disease_cohort(self, disease: str, base_cohort: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get a disease-specific cohort from the base cohort.
        
        Args:
            disease: Disease code (e.g., 'DMD', 'BMD', 'SMA')
            base_cohort: Base cohort dict (uses cached if None)
            
        Returns:
            Dictionary with disease-specific datasets and patient IDs
        """
        
        if base_cohort is None:
            base_cohort = self.get_base_cohort()
            
        cache_key = disease.upper()
        
        if cache_key not in self._disease_cohorts:
            print(f"🎯 Creating {disease} cohort from base cohort...")
            
            # Filter for disease
            disease_result = filter_disease(base_cohort['diagnosis'], disease)
            disease_patient_ids = disease_result['patient_ids']
            
            # Apply disease filter to all base datasets
            disease_datasets = apply_patient_filter_to_datasets(
                {
                    'demographics': base_cohort['demographics'],
                    'diagnosis': base_cohort['diagnosis'],
                    'encounters': base_cohort['encounters'],
                    'medications': base_cohort['medications']
                },
                disease_patient_ids
            )
            
            # Compute facility info for disease cohort
            disease_facility_info = _compute_facility_info(disease_datasets['demographics'])

            self._disease_cohorts[cache_key] = {
                'demographics': disease_datasets['demographics'],
                'diagnosis': disease_datasets['diagnosis'],
                'encounters': disease_datasets['encounters'],
                'medications': disease_datasets['medications'],
                'patient_ids': disease_patient_ids,
                'disease_columns': disease_result['disease_columns'],
                'stats': disease_result['stats'],
                'facility_info': disease_facility_info,
                'count': len(disease_patient_ids)
            }
            
            print(f"✅ {disease} cohort ready: {len(disease_patient_ids):,} patients")
        
        return self._disease_cohorts[cache_key]
    
    def filter_cohort_by_patients(self, 
                                 patient_ids: List[str], 
                                 base_cohort: Optional[Dict] = None,
                                 cohort_name: str = "custom") -> Dict[str, Any]:
        """
        Create a filtered cohort using a custom list of patient IDs.
        
        Filters all datasets in the cohort to only include the specified patients.
        Perfect for creating sub-cohorts based on specific criteria.
        
        Args:
            patient_ids: List of patient IDs (FACPATIDs) to include
            base_cohort: Source cohort to filter (uses base cohort if None)
            cohort_name: Name for the filtered cohort (for caching)
            
        Returns:
            Dictionary with filtered cohort datasets and metadata
            
        Example:
            # Create sub-cohort of ambulatory DMD patients
            dmd_cohort = get_disease_cohort('DMD')
            ambulatory_patients = ['PAT001', 'PAT002', 'PAT003']
            ambulatory_dmd = filter_cohort_by_patients(ambulatory_patients, dmd_cohort)
        """
        
        if base_cohort is None:
            base_cohort = self.get_base_cohort()
        
        patient_set = set(patient_ids)
        original_count = base_cohort['count']
        
        print(f"🎯 Creating {cohort_name} cohort from {len(patient_ids):,} specified patients...")
        print(f"   Original cohort: {original_count:,} patients")
        
        # Validate that the patient IDs exist in the base cohort
        base_patient_set = set(base_cohort['patient_ids'])
        valid_patients = patient_set.intersection(base_patient_set)
        invalid_patients = patient_set - base_patient_set
        
        if invalid_patients:
            print(f"⚠️  {len(invalid_patients)} patient IDs not found in base cohort")
        
        if not valid_patients:
            print(f"❌ No valid patient IDs found in base cohort")
            return {
                'demographics': pd.DataFrame(),
                'diagnosis': pd.DataFrame(),
                'encounters': pd.DataFrame(),
                'medications': pd.DataFrame(),
                'patient_ids': [],
                'count': 0,
                'filter_stats': {
                    'requested_patients': len(patient_ids),
                    'valid_patients': 0,
                    'invalid_patients': len(invalid_patients)
                }
            }
        
        # Apply patient filter to all datasets
        filtered_datasets = apply_patient_filter_to_datasets(
            {
                'demographics': base_cohort['demographics'],
                'diagnosis': base_cohort['diagnosis'],
                'encounters': base_cohort['encounters'],
                'medications': base_cohort['medications']
            },
            list(valid_patients)
        )
        
        # Calculate retention rate
        retention_rate = (len(valid_patients) / original_count * 100) if original_count > 0 else 0

        # Compute facility info for filtered cohort
        filtered_facility_info = _compute_facility_info(filtered_datasets['demographics'])

        filtered_cohort = {
            'demographics': filtered_datasets['demographics'],
            'diagnosis': filtered_datasets['diagnosis'],
            'encounters': filtered_datasets['encounters'],
            'medications': filtered_datasets['medications'],
            'patient_ids': list(valid_patients),
            'count': len(valid_patients),
            'facility_info': filtered_facility_info,
            'filter_stats': {
                'original_cohort_size': original_count,
                'requested_patients': len(patient_ids),
                'valid_patients': len(valid_patients),
                'invalid_patients': len(invalid_patients),
                'retention_rate': round(retention_rate, 1)
            }
        }
        
        # Inherit disease type distribution if available
        if 'dstype_counts' in base_cohort:
            filtered_dstype_counts = {}
            if 'dstype' in filtered_datasets['diagnosis'].columns:
                filtered_dstype_counts = filtered_datasets['diagnosis']['dstype'].value_counts().to_dict()
            filtered_cohort['dstype_counts'] = filtered_dstype_counts
        
        print(f"✅ {cohort_name} cohort ready: {len(valid_patients):,} patients ({retention_rate:.1f}% retention)")
        
        return filtered_cohort

    def apply_to_table(self, table_name: str, patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply a cohort's patient IDs to any MOVR table.
        
        Perfect for analyzing medications, hospitalizations, etc. for your cohort.
        
        Args:
            table_name: Name of the parquet table to load
            patient_ids: List of patient IDs to filter for (if None, loads all records)
            
        Returns:
            Filtered dataframe with only the cohort's data (or all data if patient_ids=None)
        """
        
        if patient_ids is None:
            print(f"📁 Loading {table_name} (all records)...")
        else:
            print(f"📋 Loading {table_name} for {len(patient_ids):,} patients...")
        
        try:
            # Load the table
            table_df = self.loader.load_file(table_name)
            
            # Apply patient filter if patient_ids provided
            if patient_ids is not None and 'FACPATID' in table_df.columns:
                filtered_df = table_df[table_df['FACPATID'].isin(patient_ids)]
                print(f"✅ {table_name}: {len(table_df):,} → {len(filtered_df):,} records")
                return filtered_df
            elif patient_ids is not None and 'FACPATID' not in table_df.columns:
                print(f"⚠️  No FACPATID column in {table_name}")
                return table_df
            else:
                # No filtering - return all records
                print(f"✅ {table_name}: {len(table_df):,} total records")
                return table_df
                
        except Exception as e:
            print(f"❌ Error loading {table_name}: {e}")
            return pd.DataFrame()
    

    def explore_field_values(self, 
                           dataframe: pd.DataFrame, 
                           field_name: str, 
                           max_values: int = 10) -> Dict[str, Any]:
        """
        Explore values in a field for custom filtering decisions.
        
        Perfect for understanding what values are available before filtering.
        
        Args:
            dataframe: DataFrame to explore
            field_name: Column name to explore
            max_values: Maximum unique values to show
            
        Returns:
            Dictionary with field information
        """
        
        if field_name not in dataframe.columns:
            print(f"❌ Field '{field_name}' not found in dataframe")
            return {}
        
        series = dataframe[field_name]
        value_counts = series.value_counts(dropna=False)
        
        info = {
            'total_records': len(dataframe),
            'non_null_records': series.notna().sum(),
            'null_records': series.isna().sum(),
            'unique_values': series.nunique(),
            'value_counts': value_counts.head(max_values).to_dict(),
            'sample_values': list(series.dropna().unique()[:max_values])
        }
        
        print(f"🔍 Field Analysis: {field_name}")
        print(f"   Total records: {info['total_records']:,}")
        print(f"   Non-null: {info['non_null_records']:,} ({info['non_null_records']/info['total_records']*100:.1f}%)")
        print(f"   Unique values: {info['unique_values']:,}")
        print(f"   Top values:")
        for value, count in list(value_counts.head(5).items()):
            print(f"     '{value}': {count:,}")
        
        return info
    
    def create_patient_profile_table(self, base_cohort: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create a comprehensive patient profile table with enrollment, age, and encounter information.
        
        This table provides static (non-time-series) information for each patient including:
        - Basic demographics and facility info
        - Enrollment dates and years since enrollment  
        - Age calculations (current age, age at enrollment, age at diagnosis)
        - Encounter data quality metrics
        - Extensible architecture for future additions (ambulation, etc.)
        
        Args:
            base_cohort: Base cohort dict (uses cached if None)
            
        Returns:
            DataFrame with comprehensive patient profiles
        """
        
        if base_cohort is None:
            base_cohort = self.get_base_cohort()
            
        demographics_df = base_cohort['demographics'].copy()
        diagnosis_df = base_cohort['diagnosis'].copy() 
        encounters_df = base_cohort['encounters'].copy()
        medications_df = base_cohort['medications'].copy()
        
        print(f"🏗️ Building patient profile table for {len(demographics_df)} patients...")
        
        # Start with basic patient and facility information
        profile_data = []
        
        for _, patient in demographics_df.iterrows():
            facpatid = patient['FACPATID']
            
            # Basic information
            profile = {
                'FACPATID': facpatid,
                'FACILITY_DISPLAY_ID': patient.get('FACILITY_DISPLAY_ID', 'Unknown'),
                'FACILITY_NAME': patient.get('FACILITY_NAME', 'Unknown'),
            }
            
            # Enrollment information
            enroll_date = None
            if pd.notna(patient.get('enroldt')):
                try:
                    enroll_date = pd.to_datetime(patient['enroldt'])
                    profile['enrollment_date'] = enroll_date
                    
                    # Calculate years since enrollment
                    years_since_enrollment = (pd.Timestamp.now() - enroll_date).days / 365.25
                    profile['years_since_enrollment'] = round(years_since_enrollment, 1)
                except:
                    profile['enrollment_date'] = None
                    profile['years_since_enrollment'] = None
            else:
                profile['enrollment_date'] = None
                profile['years_since_enrollment'] = None
                
            # Age calculations
            dob = None
            current_age = None
            age_at_enrollment = None
            
            if pd.notna(patient.get('dob')):
                try:
                    dob = pd.to_datetime(patient['dob'])
                    profile['date_of_birth'] = dob
                    
                    # Current age
                    current_age = (pd.Timestamp.now() - dob).days / 365.25
                    profile['current_age'] = round(current_age, 1)
                    
                    # Age at enrollment
                    if enroll_date and dob:
                        age_at_enrollment = (enroll_date - dob).days / 365.25
                        profile['age_at_enrollment'] = round(age_at_enrollment, 1)
                    else:
                        profile['age_at_enrollment'] = None
                        
                except:
                    profile['date_of_birth'] = None
                    profile['current_age'] = None
                    profile['age_at_enrollment'] = None
            else:
                profile['date_of_birth'] = None
                profile['current_age'] = None
                profile['age_at_enrollment'] = None
            
            # Diagnosis information (extensible for disease-specific logic)
            patient_diagnosis = diagnosis_df[diagnosis_df['FACPATID'] == facpatid]
            if not patient_diagnosis.empty:
                diagnosis_record = patient_diagnosis.iloc[0]  # Take first record
                profile['disease_type'] = diagnosis_record.get('dstype', 'Unknown')
                
                # Future: Add diagnosis date and age at diagnosis when available
                profile['diagnosis_date'] = None  # Placeholder for future
                profile['age_at_diagnosis'] = None  # Placeholder for future
            else:
                profile['disease_type'] = 'No Diagnosis Record'
                profile['diagnosis_date'] = None
                profile['age_at_diagnosis'] = None
            
            # Encounter data metrics
            patient_encounters = encounters_df[encounters_df['FACPATID'] == facpatid]
            profile['total_encounters'] = len(patient_encounters)
            
            # Check for encounters outside enrollment period (QA flag)
            encounters_outside_enrollment = False
            if not patient_encounters.empty and enroll_date:
                # Check if any encounters have dates before enrollment
                encounter_date_cols = [col for col in patient_encounters.columns if 'date' in col.lower() or 'dt' in col.lower()]
                
                for date_col in encounter_date_cols:
                    if date_col in patient_encounters.columns:
                        try:
                            encounter_dates = pd.to_datetime(patient_encounters[date_col], errors='coerce')
                            if encounter_dates.notna().any():
                                earliest_encounter = encounter_dates.min()
                                if earliest_encounter < enroll_date:
                                    encounters_outside_enrollment = True
                                    break
                        except:
                            continue
            
            profile['has_encounters_before_enrollment'] = encounters_outside_enrollment
            
            # Future extensibility placeholders
            profile['ambulatory_status'] = None  # For future ambulation analysis
            profile['ambulatory_loss_date'] = None  # Disease-specific milestone
            profile['age_at_ambulatory_loss'] = None  # Disease-specific milestone
            
            profile_data.append(profile)
        
        # Create DataFrame
        profile_df = pd.DataFrame(profile_data)
        
        # Sort by facility and then by FACPATID for easy navigation
        profile_df = profile_df.sort_values(['FACILITY_NAME', 'FACPATID'])
        
        print(f"✅ Patient profile table ready: {len(profile_df)} patients")
        
        return profile_df
    
    def find_fields(self, 
                   dataframe: pd.DataFrame, 
                   search_terms: List[str], 
                   case_sensitive: bool = False) -> List[str]:
        """
        Find columns containing specific search terms.
        
        Perfect for discovering relevant fields for analysis.
        
        Args:
            dataframe: DataFrame to search
            search_terms: List of terms to search for in column names
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching column names
        """
        
        columns = dataframe.columns.tolist()
        matches = []
        
        for term in search_terms:
            if case_sensitive:
                term_matches = [col for col in columns if term in col]
            else:
                term_matches = [col for col in columns if term.lower() in col.lower()]
            matches.extend(term_matches)
        
        # Remove duplicates while preserving order
        unique_matches = list(dict.fromkeys(matches))
        
        print(f"🔍 Found {len(unique_matches)} fields matching {search_terms}:")
        for match in unique_matches:
            print(f"   {match}")
        
        return unique_matches
    
    def filter_by_field_values(self, 
                             dataframe: pd.DataFrame,
                             field_name: str, 
                             values_to_include: List[Any],
                             patient_id_col: str = 'FACPATID') -> List[str]:
        """
        Filter patients based on specific field values.
        
        Perfect for building custom cohorts based on field selections.
        
        Args:
            dataframe: DataFrame to filter
            field_name: Field to filter on
            values_to_include: Values to include in filter
            patient_id_col: Column name for patient IDs
            
        Returns:
            List of patient IDs matching the filter
        """
        
        if field_name not in dataframe.columns:
            print(f"❌ Field '{field_name}' not found")
            return []
        
        filtered_df = dataframe[dataframe[field_name].isin(values_to_include)]
        patient_ids = filtered_df[patient_id_col].dropna().unique().tolist()
        
        print(f"🎯 Filter Results for {field_name}:")
        print(f"   Values included: {values_to_include}")
        print(f"   Records matched: {len(filtered_df):,}")
        print(f"   Patients matched: {len(patient_ids):,}")
        
        return patient_ids
    
    def calculate_age_from_dob(self, 
                              demographics: pd.DataFrame,
                              dob_field: str = None,
                              reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Calculate age from date of birth field.
        
        Args:
            demographics: Demographics dataframe
            dob_field: DOB column name (auto-detect if None)
            reference_date: Date to calculate age from (default: today)
            
        Returns:
            Demographics dataframe with 'calculated_age' column added
        """
        
        if reference_date is None:
            reference_date = datetime.now()
            
        # Auto-detect DOB field if not provided
        if dob_field is None:
            dob_candidates = [col for col in demographics.columns if 'dob' in col.lower()]
            if dob_candidates:
                dob_field = dob_candidates[0]
                print(f"🔍 Auto-detected DOB field: '{dob_field}'")
            else:
                print("❌ No DOB field found")
                return demographics
        
        if dob_field not in demographics.columns:
            print(f"❌ DOB field '{dob_field}' not found")
            return demographics
            
        demographics = demographics.copy()
        
        # Convert DOB to datetime
        try:
            dob_series = pd.to_datetime(demographics[dob_field], errors='coerce')
        except Exception as e:
            print(f"❌ Error parsing DOB field: {e}")
            return demographics
        
        # Calculate age in years
        age_series = (reference_date - dob_series).dt.days / 365.25
        demographics['calculated_age'] = age_series.round(1)
        
        # Stats
        valid_ages = demographics['calculated_age'].dropna()
        
        print(f"🎂 Age Calculation Results:")
        print(f"   DOB field used: {dob_field}")
        print(f"   Reference date: {reference_date.strftime('%Y-%m-%d')}")
        print(f"   Valid ages calculated: {len(valid_ages):,} / {len(demographics):,}")
        
        if len(valid_ages) > 0:
            print(f"   Age range: {valid_ages.min():.1f} - {valid_ages.max():.1f} years")
            print(f"   Median age: {valid_ages.median():.1f} years")
        
        return demographics
    
    def create_age_cohorts(self, 
                          disease_cohort: Dict[str, Any],
                          age_groups: Dict[str, Tuple[float, float]],
                          dob_field: str = None,
                          reference_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
        """
        Create age-based cohorts from a disease cohort.
        
        Perfect for operational reports and age-stratified analysis.
        
        Args:
            disease_cohort: Disease cohort from get_disease_cohort()
            age_groups: Dict of {group_name: (min_age, max_age)}
                       e.g., {'Pediatric': (0, 18), 'Adult': (18, 100)}
            dob_field: DOB column name (auto-detect if None)
            reference_date: Date to calculate age from (default: today)
            
        Returns:
            Dictionary of age cohorts with patient IDs, counts, and export data
        """
        
        print(f"👶 Creating age cohorts from {disease_cohort['count']} patients...")
        
        # Calculate ages
        demographics_with_age = self.calculate_age_from_dob(
            disease_cohort['demographics'], 
            dob_field=dob_field,
            reference_date=reference_date
        )
        
        if 'calculated_age' not in demographics_with_age.columns:
            print("❌ Age calculation failed")
            return {}
        
        age_cohorts = {}
        
        for group_name, (min_age, max_age) in age_groups.items():
            # Create age filter
            if max_age >= 999:  # Handle "X+" groups
                age_mask = demographics_with_age['calculated_age'] >= min_age
            else:
                age_mask = (
                    (demographics_with_age['calculated_age'] >= min_age) & 
                    (demographics_with_age['calculated_age'] < max_age)
                )
            
            # Apply filter
            cohort_demographics = demographics_with_age[age_mask]
            patient_ids = cohort_demographics['FACPATID'].dropna().unique().tolist()
            
            # Create export-ready data
            export_data = cohort_demographics[['FACPATID', 'FACILITY_DISPLAY_ID', 'calculated_age']].copy()
            export_data['age_group_label'] = group_name
            export_data['age_group_range'] = f"{min_age}-{max_age if max_age < 999 else '+'}"
            
            age_cohorts[group_name] = {
                'patient_ids': patient_ids,
                'demographics': cohort_demographics,
                'patient_count': len(patient_ids),
                'age_range': (min_age, max_age if max_age < 999 else None),
                'export_data': export_data,
                'age_stats': {
                    'min_age': cohort_demographics['calculated_age'].min(),
                    'max_age': cohort_demographics['calculated_age'].max(),
                    'median_age': cohort_demographics['calculated_age'].median(),
                    'mean_age': cohort_demographics['calculated_age'].mean()
                }
            }
            
            print(f"   {group_name} ({min_age}-{max_age if max_age < 999 else '+'}): {len(patient_ids):,} patients")
        
        return age_cohorts
    
    def export_age_cohorts(self, 
                          age_cohorts: Dict[str, Dict[str, Any]], 
                          disease_name: str = 'Disease',
                          output_dir: Optional[Path] = None,
                          separate_files: bool = False,
                          include_timestamp: bool = True) -> Dict[str, Path]:
        """
        Export age cohorts to CSV files for operational teams.
        
        Creates export files with patient_id, facility_id, age_group_label, age_group_range.
        
        Args:
            age_cohorts: Age cohorts from create_age_cohorts()
            disease_name: Disease name for file naming
            output_dir: Output directory (default: project output dir)
            separate_files: If True, create separate file per age group
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            Dictionary mapping file types to file paths
        """
        
        if not age_cohorts:
            print("❌ No age cohorts to export")
            return {}
        
        if output_dir is None:
            from ..config import get_config
            output_dir = Path(get_config().paths.output_dir)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if include_timestamp else ''
        exported_files = {}
        
        # Combine all age cohorts into one dataset
        all_cohorts_data = []
        cohort_summary_data = []
        
        for group_name, cohort_info in age_cohorts.items():
            export_data = cohort_info['export_data'].copy()
            all_cohorts_data.append(export_data)
            
            # Summary stats per cohort
            cohort_summary_data.append({
                'age_group_label': group_name,
                'age_group_range': f"{cohort_info['age_range'][0]}-{cohort_info['age_range'][1] if cohort_info['age_range'][1] else '+'}",
                'patient_count': cohort_info['patient_count'],
                'min_age': round(cohort_info['age_stats']['min_age'], 1),
                'max_age': round(cohort_info['age_stats']['max_age'], 1),
                'median_age': round(cohort_info['age_stats']['median_age'], 1),
                'mean_age': round(cohort_info['age_stats']['mean_age'], 1)
            })
        
        # Export combined file
        if all_cohorts_data:
            combined_df = pd.concat(all_cohorts_data, ignore_index=True)
            combined_filename = f"{disease_name.lower()}_age_cohorts{'_' + timestamp if timestamp else ''}.csv"
            combined_path = output_dir / combined_filename
            combined_df.to_csv(combined_path, index=False)
            exported_files['combined'] = combined_path
            
            print(f"📁 Exported combined cohorts: {combined_path}")
            print(f"   Total records: {len(combined_df):,}")
        
        # Export summary file
        if cohort_summary_data:
            summary_df = pd.DataFrame(cohort_summary_data)
            summary_filename = f"{disease_name.lower()}_age_cohorts_summary{'_' + timestamp if timestamp else ''}.csv"
            summary_path = output_dir / summary_filename
            summary_df.to_csv(summary_path, index=False)
            exported_files['summary'] = summary_path
            
            print(f"📊 Exported cohort summary: {summary_path}")
        
        # Export separate files if requested
        if separate_files:
            for group_name, cohort_info in age_cohorts.items():
                group_filename = f"{disease_name.lower()}_{group_name.lower().replace(' ', '_').replace('-', '_')}{'_' + timestamp if timestamp else ''}.csv"
                group_path = output_dir / group_filename
                cohort_info['export_data'].to_csv(group_path, index=False)
                exported_files[group_name] = group_path
                
                print(f"📁 Exported {group_name}: {group_path}")
        
        print(f"\n✅ Export complete! Files saved to: {output_dir}")
        return exported_files


# Global instance for easy importing
_cohort_manager = None

def get_cohort_manager() -> MOVRCohortManager:
    """Get the global cohort manager instance."""
    global _cohort_manager
    if _cohort_manager is None:
        _cohort_manager = MOVRCohortManager()
    return _cohort_manager


# Convenience functions for easy notebook usage
def get_base_cohort(force_refresh: bool = False, include_usndr: bool = False) -> Dict[str, Any]:
    """
    Get the standard MOVR base cohort (enrollment validated + optionally USNDR).
    
    This is your starting point for all analyses.
    
    Args:
        force_refresh: Force reload even if cached
        include_usndr: Include USNDR legacy patients (default: False, MOVR only)
    """
    return get_cohort_manager().get_base_cohort(force_refresh, include_usndr)


def get_disease_counts(base_cohort: Optional[Dict] = None) -> pd.DataFrame:
    """
    Get patient counts for all available diseases.
    
    Perfect for deciding which disease to analyze.
    """
    return get_cohort_manager().get_disease_counts(base_cohort)


def get_disease_cohort(disease: str, base_cohort: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get a disease-specific cohort.
    
    Args:
        disease: Disease code (e.g., 'DMD', 'BMD', 'SMA')
        base_cohort: Base cohort (uses cached if None)
    """
    return get_cohort_manager().get_disease_cohort(disease, base_cohort)


def filter_cohort_by_patients(patient_ids: List[str], 
                             base_cohort: Optional[Dict] = None,
                             cohort_name: str = "custom") -> Dict[str, Any]:
    """
    Create a filtered cohort using a custom list of patient IDs.
    
    Filters all datasets in the cohort to only include the specified patients.
    Perfect for creating sub-cohorts based on specific criteria.
    
    Args:
        patient_ids: List of patient IDs (FACPATIDs) to include
        base_cohort: Source cohort to filter (uses base cohort if None)
        cohort_name: Name for the filtered cohort
        
    Returns:
        Dictionary with filtered cohort datasets and metadata
        
    Examples:
        # Create sub-cohort from DMD patients
        dmd_cohort = get_disease_cohort('DMD')
        ambulatory_patients = ['PAT001', 'PAT002', 'PAT003']
        ambulatory_dmd = filter_cohort_by_patients(ambulatory_patients, dmd_cohort, "ambulatory_dmd")
        
        # Create sub-cohort from base cohort
        trial_eligible = ['PAT010', 'PAT020', 'PAT030']
        trial_cohort = filter_cohort_by_patients(trial_eligible, cohort_name="trial_eligible")
    """
    return get_cohort_manager().filter_cohort_by_patients(patient_ids, base_cohort, cohort_name)


def apply_to_table(table_name: str, patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Apply cohort patient IDs to any MOVR table.
    
    Args:
        table_name: Table to load (e.g., 'Encounter_Medication', 'Log_Hospitalization')
        patient_ids: Patient IDs to filter for (if None, loads all records)
    """
    return get_cohort_manager().apply_to_table(table_name, patient_ids)


def explore_field(dataframe: pd.DataFrame, field_name: str, max_values: int = 10) -> Dict[str, Any]:
    """
    Explore values in a field for filtering decisions.
    
    Args:
        dataframe: DataFrame to explore
        field_name: Column to explore
        max_values: Max unique values to show
    """
    return get_cohort_manager().explore_field_values(dataframe, field_name, max_values)


def find_fields(dataframe: pd.DataFrame, search_terms: List[str]) -> List[str]:
    """
    Find columns containing search terms.
    
    Args:
        dataframe: DataFrame to search
        search_terms: Terms to search for
    """
    return get_cohort_manager().find_fields(dataframe, search_terms)


def filter_by_values(dataframe: pd.DataFrame, 
                    field_name: str, 
                    values: List[Any]) -> List[str]:
    """
    Filter patients by field values.
    
    Args:
        dataframe: DataFrame to filter
        field_name: Field to filter on
        values: Values to include
        
    Returns:
        List of patient IDs
    """
    return get_cohort_manager().filter_by_field_values(dataframe, field_name, values)


def create_patient_profile_table(base_cohort: Optional[Dict] = None) -> pd.DataFrame:
    """
    Create a comprehensive patient profile table with enrollment, age, and encounter information.
    
    Perfect for call preparation and age-related analysis insights.
    
    Args:
        base_cohort: Base cohort (uses cached if None)
        
    Returns:
        DataFrame with comprehensive patient profiles including:
        - FACPATID, facility information
        - Enrollment dates and tenure
        - Age calculations (current, at enrollment, at diagnosis)
        - Encounter volume and QA flags
        - Extensible architecture for future enhancements
    """
    return get_cohort_manager().create_patient_profile_table(base_cohort)


def create_age_cohorts(disease_cohort: Dict[str, Any], 
                      age_groups: Dict[str, Tuple[float, float]],
                      dob_field: str = None,
                      reference_date: Optional[datetime] = None) -> Dict[str, Dict[str, Any]]:
    """
    Create age-based cohorts from a disease cohort.
    
    Perfect for operational reports and age-stratified analysis.
    
    Args:
        disease_cohort: Disease cohort from get_disease_cohort()
        age_groups: Dict of {group_name: (min_age, max_age)}
                   e.g., {'Pediatric': (0, 18), 'Adult': (18, 100)}
        dob_field: DOB column name (auto-detect if None)
        reference_date: Date to calculate age from (default: today)
        
    Returns:
        Dictionary of age cohorts with patient IDs, counts, and export data
    """
    return get_cohort_manager().create_age_cohorts(disease_cohort, age_groups, dob_field, reference_date)


def export_age_cohorts(age_cohorts: Dict[str, Dict[str, Any]], 
                      disease_name: str = 'Disease',
                      output_dir: Optional[Path] = None,
                      separate_files: bool = False,
                      include_timestamp: bool = True) -> Dict[str, Path]:
    """
    Export age cohorts to CSV files for operational teams.
    
    Creates export files with patient_id, facility_id, age_group_label, age_group_range.
    
    Args:
        age_cohorts: Age cohorts from create_age_cohorts()
        disease_name: Disease name for file naming
        output_dir: Output directory (default: project output dir)
        separate_files: If True, create separate file per age group
        include_timestamp: Whether to include timestamp in filename
        
    Returns:
        Dictionary mapping file types to file paths
    """
    return get_cohort_manager().export_age_cohorts(
        age_cohorts, disease_name, output_dir, separate_files, include_timestamp
    )


# Data Dictionary Search Functions
def search_data_dictionary(search_terms: Union[str, List[str]],
                          diseases: Optional[List[str]] = None,
                          crf_filter: Optional[str] = None,
                          case_sensitive: bool = False) -> pd.DataFrame:
    """
    Search for fields in the data dictionary across multiple columns.
    
    Searches in Field Name, Description, and Display Label columns.
    
    Args:
        search_terms: Text or list of texts to search for (multiple terms use OR logic)
        diseases: List of diseases to filter by (e.g., ['DMD', 'ALS'])
        crf_filter: CRF/form to filter by (e.g., 'Diagnosis', 'Encounter')
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        DataFrame with matching fields
        
    Examples:
        # Search for ejection fraction fields in DMD patients
        results = search_data_dictionary('ejection', diseases=['DMD'], crf_filter='Encounter')
        
        # Search for medication OR drug fields
        results = search_data_dictionary(['medication', 'drug'])
        
        # Search in diagnosis forms only
        results = search_data_dictionary('gene', crf_filter='Diagnosis')
    """
    from ..data_processing.data_dictionary import DataDictionary
    
    dd = DataDictionary()
    return dd.search_fields(search_terms, diseases, crf_filter, case_sensitive)


def get_disease_fields(disease: str) -> pd.DataFrame:
    """
    Get all fields applicable to a specific disease.
    
    Args:
        disease: Disease code (e.g., 'DMD', 'ALS', 'SMA')
        
    Returns:
        DataFrame with fields applicable to the disease
    """
    from ..data_processing.data_dictionary import DataDictionary
    
    dd = DataDictionary()
    return dd.get_disease_fields(disease)


def get_crf_fields(crf_name: str) -> pd.DataFrame:
    """
    Get all fields from a specific CRF/form.
    
    Args:
        crf_name: CRF/form name (e.g., 'Diagnosis', 'Encounter')
        
    Returns:
        DataFrame with fields from the specified CRF
    """
    from ..data_processing.data_dictionary import DataDictionary
    
    dd = DataDictionary()
    return dd.get_crf_fields(crf_name)

