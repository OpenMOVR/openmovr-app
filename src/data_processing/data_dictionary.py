"""
Data Dictionary Processing Module

Converts datadictionary.parquet to JSON configuration and provides
utilities for accessing field definitions and metadata.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class DataDictionary:
    """
    Manages data dictionary from parquet file and converts to JSON configuration.
    """
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.dict_file = self.data_path / "datadictionary.parquet"
        self.config_file = Path("config") / "data_dictionary.json"
        self._data = None
        self._config = None
    
    def load_parquet(self) -> pd.DataFrame:
        """Load data dictionary from parquet file."""
        if not self.dict_file.exists():
            raise FileNotFoundError(f"Data dictionary not found: {self.dict_file}")
        
        self._data = pd.read_parquet(self.dict_file)
        return self._data
    
    def convert_to_json(self, include_custom_labels: bool = True) -> Dict[str, Any]:
        """
        Convert parquet data dictionary to JSON configuration format.
        
        Args:
            include_custom_labels: Whether to include space for custom labels
            
        Returns:
            Dictionary structure for JSON configuration
        """
        if self._data is None:
            self.load_parquet()
        
        config = {
            "metadata": {
                "source": "datadictionary.parquet",
                "generated_at": pd.Timestamp.now().isoformat(),
                "total_fields": len(self._data)
            },
            "key_fields": {
                "participant_id": "FACPATID",
                "case_id": "CASE_ID", 
                "facility_id": "FACILITY_DISPLAY_ID",
                "form_name": "SCHEDULED_FORM_NAME",
                "legacy_flag": "USNDR"
            },
            "data_types": {
                "demographics": ["Demographics_MainData"],
                "encounters": ["Encounter_MainData", "Encounter_*"],
                "diagnosis": ["Diagnosis_MainData", "Diagnosis_*"],
                "longitudinal": ["Encounter_*", "Log_*"]
            },
            "enrollment_requirements": {
                "required_forms": [
                    "Demographics_MainData",
                    "Diagnosis_MainData", 
                    "Encounter_MainData"
                ],
                "minimum_records": 1
            },
            "fields": {}
        }
        
        # Process each field from data dictionary
        for _, row in self._data.iterrows():
            field_name = row.get('field_name') or row.get('Field_Name') or str(row.iloc[0])
            
            field_config = {
                "table": row.get('table_name', ''),
                "type": row.get('field_type', ''),
                "description": row.get('description', ''),
                "required": row.get('required', False),
                "values": row.get('valid_values', ''),
            }
            
            if include_custom_labels:
                field_config["custom_label"] = ""
                field_config["analytics_notes"] = ""
            
            config["fields"][field_name] = field_config
        
        self._config = config
        return config
    
    def save_json_config(self, output_path: Optional[str] = None) -> Path:
        """Save JSON configuration to file."""
        if self._config is None:
            self.convert_to_json()
        
        output_file = Path(output_path) if output_path else self.config_file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self._config, f, indent=2, default=str)
        
        return output_file
    
    def load_json_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load JSON configuration from file."""
        config_file = Path(config_path) if config_path else self.config_file
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self._config = json.load(f)
        
        return self._config
    
    def search_fields(self, 
                     search_terms: Union[str, List[str]],
                     diseases: Optional[List[str]] = None,
                     crf_filter: Optional[str] = None,
                     case_sensitive: bool = False) -> pd.DataFrame:
        """
        Search for fields in the data dictionary across multiple columns.
        
        Args:
            search_terms: Text or list of texts to search for (multiple terms use OR logic)
            diseases: List of diseases to filter by (e.g., ['DMD', 'ALS'])
            crf_filter: CRF/form to filter by (e.g., 'Diagnosis', 'Encounter')
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            DataFrame with matching fields
            
        Example:
            # Search for ejection fraction fields in DMD patients
            results = dd.search_fields('ejection', diseases=['DMD'], crf_filter='Encounter')
            
            # Search for medication OR drug fields
            results = dd.search_fields(['medication', 'drug'])
        """
        if self._data is None:
            self.load_parquet()
        
        df = self._data.copy()
        
        # Handle both single string and list of strings
        if isinstance(search_terms, str):
            search_terms = [search_terms]
        
        # Prepare search patterns
        if case_sensitive:
            patterns = search_terms
        else:
            patterns = [term.lower() for term in search_terms]
            
        # Search in Field Name, Description, and Display Label
        search_columns = ['Field Name', 'Description', 'Display Label']
        
        # Create OR mask for multiple search terms
        combined_mask = pd.Series(False, index=df.index)
        
        for pattern in patterns:
            if case_sensitive:
                term_mask = df[search_columns].apply(
                    lambda col: col.astype(str).str.contains(pattern, na=False)
                ).any(axis=1)
            else:
                term_mask = df[search_columns].apply(
                    lambda col: col.astype(str).str.lower().str.contains(pattern, na=False)
                ).any(axis=1)
            
            combined_mask |= term_mask
        
        filtered_df = df[combined_mask]
        
        # Filter by diseases if specified
        if diseases:
            disease_mask = pd.Series(False, index=filtered_df.index)
            for disease in diseases:
                if disease in filtered_df.columns:
                    # Any non-null value in disease column indicates applicability
                    disease_mask |= filtered_df[disease].notna()
            filtered_df = filtered_df[disease_mask]
        
        # Filter by CRF if specified
        if crf_filter:
            if case_sensitive:
                crf_mask = filtered_df['CRF'].str.contains(crf_filter, na=False)
            else:
                crf_mask = filtered_df['CRF'].str.lower().str.contains(crf_filter.lower(), na=False)
            filtered_df = filtered_df[crf_mask]
        
        return filtered_df
    
    def get_disease_fields(self, disease: str) -> pd.DataFrame:
        """
        Get all fields applicable to a specific disease.
        
        Args:
            disease: Disease code (e.g., 'DMD', 'ALS', 'SMA')
            
        Returns:
            DataFrame with fields applicable to the disease
        """
        if self._data is None:
            self.load_parquet()
            
        if disease not in self._data.columns:
            available_diseases = [col for col in self._data.columns 
                                if col in ['ALS', 'BMD', 'DMD', 'SMA', 'LGMD', 'FSHD', 'Pompe']]
            raise ValueError(f"Disease '{disease}' not found. Available: {available_diseases}")
        
        return self._data[self._data[disease] == 'Yes']
    
    def get_crf_fields(self, crf_name: str) -> pd.DataFrame:
        """
        Get all fields from a specific CRF/form.
        
        Args:
            crf_name: CRF/form name (e.g., 'Diagnosis', 'Encounter')
            
        Returns:
            DataFrame with fields from the specified CRF
        """
        if self._data is None:
            self.load_parquet()
            
        return self._data[self._data['CRF'].str.contains(crf_name, case=False, na=False)]

        return self._config
    
    def get_field_info(self, field_name: str) -> Dict[str, Any]:
        """Get information about a specific field."""
        if self._config is None:
            try:
                self.load_json_config()
            except FileNotFoundError:
                self.convert_to_json()
        
        return self._config.get("fields", {}).get(field_name, {})
    
    def get_key_fields(self) -> Dict[str, str]:
        """Get mapping of key field types to actual field names."""
        if self._config is None:
            try:
                self.load_json_config()
            except FileNotFoundError:
                self.convert_to_json()
        
        return self._config.get("key_fields", {})
    
    def get_enrollment_requirements(self) -> Dict[str, Any]:
        """Get participant enrollment requirements."""
        if self._config is None:
            try:
                self.load_json_config()
            except FileNotFoundError:
                self.convert_to_json()
        
        return self._config.get("enrollment_requirements", {})


def main():
    """CLI interface for data dictionary processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process MOVR data dictionary")
    parser.add_argument("--data-path", default="data", help="Path to data directory")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--update", action="store_true", help="Update existing JSON config")
    
    args = parser.parse_args()
    
    dd = DataDictionary(args.data_path)
    
    try:
        config = dd.convert_to_json()
        output_file = dd.save_json_config(args.output)
        print(f"Data dictionary converted to: {output_file}")
        print(f"Total fields: {config['metadata']['total_fields']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())