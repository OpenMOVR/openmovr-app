"""
Data Cleaning Utilities for MOVR Clinical Analytics

Provides standardized data cleaning functions for MOVR datasets including
missing value handling, outlier detection, data type corrections, and
validation against data dictionary specifications.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import re
from pathlib import Path

from ..config import get_config
from ..data_processing.data_dictionary import DataDictionary


logger = logging.getLogger(__name__)


class MOVRDataCleaner:
    """
    Standardized data cleaning for MOVR clinical datasets.
    
    Features:
    - Data type enforcement
    - Missing value handling  
    - Outlier detection and handling
    - Date/time standardization
    - Text field cleaning
    - Cross-field validation
    """
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.config = get_config()
        self.data_path = Path(data_path) if data_path else self.config.paths.data_dir
        self.data_dict = DataDictionary(str(self.data_path))
        
        # Cleaning statistics
        self.cleaning_log = []
        self._key_fields = None
    
    @property
    def key_fields(self) -> Dict[str, str]:
        """Get key field mappings from data dictionary."""
        if self._key_fields is None:
            try:
                self._key_fields = self.data_dict.get_key_fields()
            except FileNotFoundError:
                self._key_fields = {
                    "participant_id": "FACPATID",
                    "case_id": "CASE_ID",
                    "facility_id": "FACILITY_DISPLAY_ID",
                    "form_name": "SCHEDULED_FORM_NAME",
                    "legacy_flag": "USNDR"
                }
        return self._key_fields
    
    def clean_dataset(self, 
                      df: pd.DataFrame, 
                      table_name: str,
                      strict_mode: bool = None) -> pd.DataFrame:
        """
        Apply comprehensive cleaning to a dataset.
        
        Args:
            df: Input DataFrame
            table_name: Name of the table (for dictionary lookup)
            strict_mode: Whether to apply strict validation rules
            
        Returns:
            Cleaned DataFrame
        """
        strict_mode = strict_mode if strict_mode is not None else self.config.analytics.strict_validation
        
        cleaned_df = df.copy()
        cleaning_stats = {
            'table_name': table_name,
            'original_rows': len(df),
            'original_columns': len(df.columns),
            'timestamp': datetime.now(),
            'operations': []
        }
        
        # 1. Clean column names
        cleaned_df = self._clean_column_names(cleaned_df)
        cleaning_stats['operations'].append('column_name_cleaning')
        
        # 2. Handle missing values
        cleaned_df, missing_stats = self._handle_missing_values(cleaned_df, table_name)
        cleaning_stats['operations'].append(f'missing_values: {missing_stats}')
        
        # 3. Enforce data types
        cleaned_df, type_stats = self._enforce_data_types(cleaned_df, table_name)
        cleaning_stats['operations'].append(f'data_types: {type_stats}')
        
        # 4. Clean text fields
        cleaned_df = self._clean_text_fields(cleaned_df)
        cleaning_stats['operations'].append('text_cleaning')
        
        # 5. Standardize dates
        cleaned_df = self._standardize_dates(cleaned_df)
        cleaning_stats['operations'].append('date_standardization')
        
        # 6. Handle outliers
        cleaned_df, outlier_stats = self._handle_outliers(cleaned_df, table_name)
        cleaning_stats['operations'].append(f'outliers: {outlier_stats}')
        
        # 7. Validate cross-field consistency
        if strict_mode:
            cleaned_df, validation_stats = self._validate_cross_fields(cleaned_df, table_name)
            cleaning_stats['operations'].append(f'validation: {validation_stats}')
        
        # 8. Remove duplicate records
        cleaned_df, duplicate_stats = self._remove_duplicates(cleaned_df, table_name)
        cleaning_stats['operations'].append(f'duplicates: {duplicate_stats}')
        
        # Final statistics
        cleaning_stats['final_rows'] = len(cleaned_df)
        cleaning_stats['final_columns'] = len(cleaned_df.columns)
        cleaning_stats['rows_removed'] = cleaning_stats['original_rows'] - cleaning_stats['final_rows']
        
        self.cleaning_log.append(cleaning_stats)
        
        logger.info(f"Cleaned {table_name}: {cleaning_stats['original_rows']} → {cleaning_stats['final_rows']} rows")
        
        return cleaned_df
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        # Remove extra whitespace and normalize case
        df.columns = df.columns.str.strip()
        
        # Log column name changes
        original_cols = list(df.columns)
        new_cols = []
        
        for col in original_cols:
            # Remove special characters except underscores
            clean_col = re.sub(r'[^\w]', '_', col)
            # Remove multiple underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            # Remove trailing underscores
            clean_col = clean_col.strip('_')
            new_cols.append(clean_col)
        
        df.columns = new_cols
        
        # Log changes
        changes = [(orig, new) for orig, new in zip(original_cols, new_cols) if orig != new]
        if changes:
            logger.info(f"Column name changes: {changes}")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Handle missing values based on data dictionary rules."""
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_stats[column] = missing_count
            
            if missing_count > 0:
                field_info = self.data_dict.get_field_info(column)
                
                # For required fields, flag missing values
                if field_info.get('required', False):
                    logger.warning(f"Missing values in required field {column}: {missing_count}")
                
                # Handle specific data types
                if field_info.get('type') == 'numeric':
                    # For numeric fields, consider imputation or flagging
                    if missing_count / len(df) < 0.1:  # Less than 10% missing
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        df[f"{column}_MISSING"] = df[column].isnull()
                
                elif field_info.get('type') == 'categorical':
                    # For categorical, use "Unknown" or mode
                    df[column] = df[column].fillna('Unknown')
                
                elif field_info.get('type') == 'boolean':
                    # For boolean, be explicit about missing
                    df[f"{column}_MISSING"] = df[column].isnull()
        
        return df, missing_stats
    
    def _enforce_data_types(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Enforce proper data types based on data dictionary."""
        type_stats = {}
        
        for column in df.columns:
            if column.endswith('_MISSING'):  # Skip flag columns
                continue
                
            field_info = self.data_dict.get_field_info(column)
            expected_type = field_info.get('type', '')
            
            if expected_type == 'numeric':
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    type_stats[column] = 'numeric'
                except Exception as e:
                    logger.warning(f"Could not convert {column} to numeric: {e}")
            
            elif expected_type == 'boolean':
                # Handle various boolean representations
                if df[column].dtype == 'object':
                    boolean_map = {'true': True, 'false': False, '1': True, '0': False,
                                   'yes': True, 'no': False, 'y': True, 'n': False}
                    df[column] = df[column].str.lower().map(boolean_map)
                type_stats[column] = 'boolean'
            
            elif expected_type == 'date':
                try:
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                    type_stats[column] = 'date'
                except Exception as e:
                    logger.warning(f"Could not convert {column} to datetime: {e}")
            
            elif expected_type == 'categorical':
                df[column] = df[column].astype('category')
                type_stats[column] = 'categorical'
        
        return df, type_stats
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text fields."""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            if column in df.columns:
                # Remove extra whitespace
                df[column] = df[column].astype(str).str.strip()
                
                # Standardize case for certain fields
                if any(keyword in column.lower() for keyword in ['name', 'description', 'notes']):
                    df[column] = df[column].str.title()
                
                # Remove special characters from ID fields
                if any(keyword in column.upper() for keyword in ['ID', 'CODE']):
                    df[column] = df[column].str.replace(r'[^\w]', '', regex=True)
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date formats."""
        date_columns = df.select_dtypes(include=['datetime64']).columns
        
        for column in date_columns:
            # Ensure consistent format
            df[column] = pd.to_datetime(df[column], errors='coerce')
            
            # Flag future dates that seem unrealistic
            future_threshold = datetime.now() + pd.Timedelta(days=365)
            future_dates = df[column] > future_threshold
            if future_dates.any():
                logger.warning(f"Found {future_dates.sum()} future dates in {column}")
                df[f"{column}_FUTURE_FLAG"] = future_dates
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Detect and handle outliers in numeric fields."""
        outlier_stats = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if column.endswith('_MISSING') or column.endswith('_FLAG'):
                continue
            
            # Use IQR method for outlier detection
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_count = outliers.sum()
            outlier_stats[column] = outlier_count
            
            if outlier_count > 0:
                logger.info(f"Found {outlier_count} outliers in {column}")
                
                # Flag outliers rather than removing them
                df[f"{column}_OUTLIER"] = outliers
                
                # For extreme outliers (>3 IQR), cap values
                extreme_lower = Q1 - 3 * IQR
                extreme_upper = Q3 + 3 * IQR
                extreme_outliers = (df[column] < extreme_lower) | (df[column] > extreme_upper)
                
                if extreme_outliers.any():
                    df.loc[df[column] < extreme_lower, column] = lower_bound
                    df.loc[df[column] > extreme_upper, column] = upper_bound
                    logger.info(f"Capped {extreme_outliers.sum()} extreme outliers in {column}")
        
        return df, outlier_stats
    
    def _validate_cross_fields(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Validate consistency across related fields."""
        validation_stats = {}
        
        # Check for required key field combinations
        key_cols = [self.key_fields["participant_id"], self.key_fields["case_id"]]
        available_key_cols = [col for col in key_cols if col in df.columns]
        
        if available_key_cols:
            # Check for missing key values
            missing_keys = df[available_key_cols].isnull().any(axis=1)
            validation_stats['missing_keys'] = missing_keys.sum()
            
            if missing_keys.any():
                logger.warning(f"Found {missing_keys.sum()} records with missing key fields")
                # Flag these records
                df['INVALID_KEYS'] = missing_keys
        
        # Date field validation
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) >= 2:
            # Check for date ordering issues
            start_date_cols = [col for col in date_columns if 'start' in col.lower()]
            end_date_cols = [col for col in date_columns if 'end' in col.lower()]
            
            for start_col in start_date_cols:
                for end_col in end_date_cols:
                    if start_col in df.columns and end_col in df.columns:
                        invalid_dates = df[start_col] > df[end_col]
                        validation_stats[f'{start_col}_after_{end_col}'] = invalid_dates.sum()
                        
                        if invalid_dates.any():
                            df[f'INVALID_DATE_ORDER_{start_col}_{end_col}'] = invalid_dates
        
        return df, validation_stats
    
    def _remove_duplicates(self, df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Remove duplicate records based on unique key combinations."""
        duplicate_stats = {'removed': 0}
        
        # Define unique key columns based on table type
        unique_cols = [self.key_fields["participant_id"]]
        
        # Add case_id if available
        if self.key_fields["case_id"] in df.columns:
            unique_cols.append(self.key_fields["case_id"])
        
        # Add form name for form-specific tables
        if "SCHEDULED_FORM_NAME" in df.columns:
            unique_cols.append("SCHEDULED_FORM_NAME")
        
        # Check for duplicates
        before_count = len(df)
        df = df.drop_duplicates(subset=unique_cols, keep='first')
        after_count = len(df)
        
        duplicate_stats['removed'] = before_count - after_count
        
        if duplicate_stats['removed'] > 0:
            logger.info(f"Removed {duplicate_stats['removed']} duplicate records from {table_name}")
        
        return df, duplicate_stats
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """Get summary of all cleaning operations performed."""
        if not self.cleaning_log:
            return pd.DataFrame()
        
        summary_data = []
        for log_entry in self.cleaning_log:
            summary_data.append({
                'table_name': log_entry['table_name'],
                'timestamp': log_entry['timestamp'],
                'original_rows': log_entry['original_rows'],
                'final_rows': log_entry['final_rows'],
                'rows_removed': log_entry['rows_removed'],
                'operations': len(log_entry['operations'])
            })
        
        return pd.DataFrame(summary_data)
    
    def clean_demographics_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply demographics-specific cleaning rules."""
        # Validate USNDR field
        if self.key_fields["legacy_flag"] in df.columns:
            usndr_col = self.key_fields["legacy_flag"]
            # Ensure boolean type
            df[usndr_col] = df[usndr_col].fillna(False)
            df[usndr_col] = df[usndr_col].astype(bool)
        
        # Age validations
        age_columns = [col for col in df.columns if 'age' in col.lower()]
        for col in age_columns:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                # Flag negative or extremely high ages
                invalid_ages = (df[col] < 0) | (df[col] > 120 * 12)  # Assume months
                if invalid_ages.any():
                    logger.warning(f"Found {invalid_ages.sum()} invalid ages in {col}")
                    df[f"{col}_INVALID"] = invalid_ages
        
        return df
    
    def clean_encounters_specific(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encounter-specific cleaning rules."""
        # Ensure encounter dates are valid
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            # Encounters shouldn't be in the far future
            future_threshold = datetime.now() + pd.Timedelta(days=30)
            future_encounters = df[col] > future_threshold
            if future_encounters.any():
                logger.warning(f"Found {future_encounters.sum()} future encounter dates in {col}")
                df[f"{col}_FUTURE"] = future_encounters
        
        return df


def get_cleaner(data_path: Optional[Union[str, Path]] = None) -> MOVRDataCleaner:
    """Get a configured data cleaner instance."""
    return MOVRDataCleaner(data_path)