"""
Data Loading Utilities for MOVR Clinical Analytics

Provides standardized data loading functionality for all MOVR parquet files
with built-in validation, caching, and metadata management.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from functools import lru_cache
import pickle
from datetime import datetime

from ..config import get_config
from .data_dictionary import DataDictionary


logger = logging.getLogger(__name__)


class MOVRDataLoader:
    """
    Standardized data loader for MOVR clinical data.
    
    Features:
    - Automatic schema validation
    - Data type enforcement
    - Caching for performance
    - Metadata tracking
    - Error handling and logging
    """
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.config = get_config()
        self.data_path = Path(data_path) if data_path else self.config.paths.data_dir
        self.data_dict = DataDictionary(str(self.data_path))
        
        # Cache settings
        self.cache_enabled = self.config.analytics.cache_enabled
        self.cache_dir = self.data_path / ".cache"
        if self.cache_enabled:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Metadata tracking
        self.load_history = []
        self._key_fields = None
    
    @property
    def key_fields(self) -> Dict[str, str]:
        """Get key field mappings from data dictionary."""
        if self._key_fields is None:
            try:
                self._key_fields = self.data_dict.get_key_fields()
            except FileNotFoundError:
                # Fallback to hardcoded values
                self._key_fields = {
                    "participant_id": "FACPATID",
                    "case_id": "CASE_ID",
                    "facility_id": "FACILITY_DISPLAY_ID", 
                    "form_name": "SCHEDULED_FORM_NAME",
                    "legacy_flag": "USNDR"
                }
        return self._key_fields
    
    def list_available_files(self) -> List[str]:
        """List all available parquet files in data directory."""
        if not self.data_path.exists():
            return []
        
        parquet_files = list(self.data_path.glob("*.parquet"))
        return [f.stem for f in parquet_files]
    
    def _normalize_table_name(self, table_name: str) -> str:
        """
        Normalize table name to match parquet file naming convention.

        Converts CamelCase config names to lowercase file names.
        Examples:
            'Demographics_MainData' -> 'demographics_maindata'
            'Diagnosis_GeneproteinRG' -> 'diagnosis_geneprotein_rg'
            'Encounter_Medication' -> 'encounter_medication_rg'

        Args:
            table_name: Table name from config (e.g., 'Demographics_MainData')

        Returns:
            Normalized filename (without .parquet extension)
        """
        import re

        # Remove .parquet extension if present
        name = table_name.replace('.parquet', '')

        # Convert to lowercase
        normalized = name.lower()

        # Handle common naming variations:
        # 'Encounter_Medication' in config might be 'encounter_medication_rg' in file
        # We'll try the exact name first, then variations

        # Ensure 'rg' suffix has underscore before it
        if normalized.endswith('rg') and not normalized.endswith('_rg'):
            normalized = normalized[:-2] + '_rg'

        return normalized

    def _resolve_file_path(self, filename: str) -> Path:
        """
        Resolve the actual file path for a given filename.

        Handles case-insensitive lookups and name variations.

        Args:
            filename: Table name or filename (with or without .parquet)

        Returns:
            Path to the actual file

        Raises:
            FileNotFoundError: If no matching file found
        """
        # Normalize the filename
        base_name = filename.replace('.parquet', '')
        normalized = self._normalize_table_name(base_name)

        # Try exact match first
        exact_path = self.data_path / f"{normalized}.parquet"
        if exact_path.exists():
            return exact_path

        # Try original name (for backward compatibility)
        original_path = self.data_path / f"{base_name.lower()}.parquet"
        if original_path.exists():
            return original_path

        # Try variations for repeat group tables
        # Config might say 'Encounter_Medication' but file is 'encounter_medication_rg'
        if not normalized.endswith('_rg'):
            rg_path = self.data_path / f"{normalized}_rg.parquet"
            if rg_path.exists():
                return rg_path

        # Case-insensitive glob search as last resort
        available = list(self.data_path.glob("*.parquet"))
        for f in available:
            if f.stem.lower() == normalized.lower():
                return f
            # Also check without _rg suffix
            if f.stem.lower() == normalized.replace('_rg', '').lower():
                return f

        raise FileNotFoundError(
            f"Data file not found: {base_name}\n"
            f"Tried: {normalized}.parquet\n"
            f"Available files: {[f.stem for f in available][:10]}..."
        )

    def load_file(self,
                  filename: str,
                  use_cache: bool = None,
                  validate: bool = True,
                  chunk_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load a single parquet file with optional caching and validation.

        Args:
            filename: Name of parquet file (with or without .parquet extension)
                     Can be CamelCase (e.g., 'Demographics_MainData') or
                     lowercase (e.g., 'demographics_maindata')
            use_cache: Whether to use cached version (None = use config default)
            validate: Whether to run data validation
            chunk_size: For large files, load in chunks

        Returns:
            Loaded DataFrame
        """
        # Resolve the file path (handles case normalization)
        file_path = self._resolve_file_path(filename)

        # Use resolved filename for cache
        filename = file_path.name
        
        # Check cache
        use_cache = use_cache if use_cache is not None else self.cache_enabled
        cache_path = self._get_cache_path(filename)
        
        if use_cache and cache_path.exists():
            cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            
            if cache_time > file_time:
                logger.info(f"Loading {filename} from cache")
                return self._load_from_cache(cache_path)
        
        # Load data
        logger.info(f"Loading {filename} from parquet")
        start_time = datetime.now()
        
        if chunk_size:
            df = self._load_chunked(file_path, chunk_size)
        else:
            df = pd.read_parquet(file_path)
        
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Data validation
        if validate:
            self._validate_data(df, filename)
        
        # Cache if enabled
        if use_cache:
            self._save_to_cache(df, cache_path)
        
        # Track loading metadata
        self.load_history.append({
            'filename': filename,
            'timestamp': datetime.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'load_time_seconds': load_time,
            'from_cache': False
        })
        
        return df
    
    def load_multiple(self, 
                      filenames: List[str],
                      merge_on: Optional[List[str]] = None,
                      how: str = 'inner') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load multiple parquet files.
        
        Args:
            filenames: List of file names to load
            merge_on: Fields to merge on (if None, returns dict of DataFrames)
            how: Type of merge ('inner', 'outer', 'left', 'right')
            
        Returns:
            Merged DataFrame or dictionary of DataFrames
        """
        dataframes = {}
        
        for filename in filenames:
            dataframes[filename] = self.load_file(filename)
        
        if merge_on is None:
            return dataframes
        
        # Merge dataframes
        result = None
        for name, df in dataframes.items():
            if result is None:
                result = df.copy()
            else:
                result = result.merge(df, on=merge_on, how=how, suffixes=('', f'_{name}'))
        
        return result
    
    def load_demographics(self) -> pd.DataFrame:
        """Load demographics data."""
        return self.load_file("demographics_maindata")
    
    def load_encounters(self) -> pd.DataFrame:
        """Load encounter main data."""
        return self.load_file("encounter_maindata")
    
    def load_diagnosis(self) -> pd.DataFrame:
        """Load diagnosis main data."""
        return self.load_file("diagnosis_maindata")
    
    def load_medications(self) -> pd.DataFrame:
        """Load medication data."""
        return self.load_file("combo_drugs")
    
    def load_enrollment_data(self) -> pd.DataFrame:
        """
        Load minimum required data for enrollment validation.
        Merges demographics, diagnosis, and encounter main data.
        """
        merge_keys = [self.key_fields["participant_id"], 
                      self.key_fields["case_id"]]
        
        return self.load_multiple(
            ["demographics_maindata", "diagnosis_maindata", "encounter_maindata"],
            merge_on=merge_keys,
            how='outer'
        )
    
    def load_longitudinal_data(self, table_pattern: str = "encounter_") -> Dict[str, pd.DataFrame]:
        """
        Load all longitudinal data tables matching pattern.
        
        Args:
            table_pattern: Pattern to match table names
            
        Returns:
            Dictionary of DataFrames by table name
        """
        available_files = self.list_available_files()
        matching_files = [f for f in available_files if f.startswith(table_pattern)]
        
        return self.load_multiple(matching_files)
    
    def _load_chunked(self, file_path: Path, chunk_size: int) -> pd.DataFrame:
        """Load large parquet file in chunks."""
        chunks = []
        
        # Read parquet metadata to estimate chunks
        parquet_file = pd.read_parquet(file_path, engine='pyarrow')
        
        for i in range(0, len(parquet_file), chunk_size):
            chunk = parquet_file.iloc[i:i + chunk_size]
            chunks.append(chunk)
        
        return pd.concat(chunks, ignore_index=True)
    
    def _validate_data(self, df: pd.DataFrame, filename: str):
        """Basic data validation."""
        # Check for required key fields
        required_fields = [self.key_fields["participant_id"]]
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            logger.warning(f"Missing required fields in {filename}: {missing_fields}")
        
        # Check for completely empty data
        if len(df) == 0:
            logger.warning(f"Empty dataset loaded: {filename}")
        
        # Check for duplicate rows based on unique key combination
        if filename != "datadictionary.parquet":
            key_cols = [self.key_fields["participant_id"], self.key_fields["case_id"]]
            if "SCHEDULED_FORM_NAME" in df.columns:
                key_cols.append("SCHEDULED_FORM_NAME")
            
            available_cols = [col for col in key_cols if col in df.columns]
            if available_cols:
                duplicates = df.duplicated(subset=available_cols)
                if duplicates.any():
                    logger.warning(f"Found {duplicates.sum()} duplicate records in {filename}")
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path for given filename."""
        cache_name = f"{filename.replace('.parquet', '')}.pkl"
        return self.cache_dir / cache_name
    
    def _save_to_cache(self, df: pd.DataFrame, cache_path: Path):
        """Save DataFrame to cache."""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame:
        """Load DataFrame from cache."""
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    def get_load_stats(self) -> pd.DataFrame:
        """Get statistics about loaded files."""
        if not self.load_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.load_history)
    
    def clear_cache(self):
        """Clear all cached files."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
    
    def get_data_summary(self, filename: str) -> Dict[str, Any]:
        """Get summary information about a dataset without loading it fully."""
        if not filename.endswith('.parquet'):
            filename = f"{filename}.parquet"
        
        file_path = self.data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Use pyarrow to get metadata without loading full dataset
        import pyarrow.parquet as pq
        
        parquet_file = pq.ParquetFile(file_path)
        metadata = parquet_file.metadata
        schema = parquet_file.schema
        
        return {
            'filename': filename,
            'num_rows': metadata.num_rows,
            'num_columns': len(schema.names),
            'columns': schema.names,
            'file_size_bytes': file_path.stat().st_size,
            'created_at': datetime.fromtimestamp(file_path.stat().st_ctime),
            'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime)
        }


def get_loader(data_path: Optional[Union[str, Path]] = None) -> MOVRDataLoader:
    """Get a configured data loader instance."""
    return MOVRDataLoader(data_path)