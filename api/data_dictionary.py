"""
Data Dictionary API Layer

Provides a facade over the core data dictionary module for web display.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.data_dictionary import DataDictionary


# Known mislabeled fields - marked for wrong disease in data dictionary
MISLABELED_FIELDS = {
    'LGMD': [
        # FSHD fields incorrectly marked as LGMD
        'fshdcsfw', 'fshdcssgi', 'fshdcsuli', 'fshdcsli',
        'fshdcspgi', 'fshdcsami', 'fshdtcs',
    ],
    # Add other disease mislabelings as discovered
}

# Disease prefixes - fields starting with these belong to that disease
DISEASE_PREFIXES = {
    'ALS': ['als'],
    'BMD': ['bmd'],
    'DMD': ['dmd'],
    'SMA': ['sma'],
    'LGMD': ['lg'],
    'FSHD': ['fshd'],
    'Pompe': ['pom', 'gaa', 'ert'],
}


class DataDictionaryAPI:
    """
    API for data dictionary operations in the webapp.

    Provides clean interface for searching, filtering, and displaying
    field metadata from the MOVR data dictionary.
    """

    _instance: Optional["DataDictionaryAPI"] = None
    _dd: Optional[DataDictionary] = None

    @classmethod
    def _get_dd(cls) -> DataDictionary:
        """Get cached DataDictionary instance."""
        if cls._dd is None:
            cls._dd = DataDictionary()
            cls._dd.load_parquet()
        return cls._dd

    @classmethod
    def load_dictionary(cls) -> pd.DataFrame:
        """
        Load the full data dictionary as a DataFrame.

        Returns:
            DataFrame with all fields and metadata
        """
        return cls._get_dd()._data.copy()

    @classmethod
    def get_forms(cls) -> List[str]:
        """
        Get list of unique forms (File/Form column) in the dictionary.

        Returns:
            Sorted list of form names
        """
        df = cls.load_dictionary()
        if "File/Form" in df.columns:
            forms = df["File/Form"].dropna().unique().tolist()
            return sorted(forms)
        return []

    @classmethod
    def get_excel_tabs(cls) -> List[str]:
        """
        Get list of unique Excel tabs/sheets in the dictionary.

        Returns:
            Sorted list of Excel tab names
        """
        df = cls.load_dictionary()
        if "Excel Tab" in df.columns:
            tabs = df["Excel Tab"].dropna().unique().tolist()
            return sorted(tabs)
        return []

    @classmethod
    def get_field_types(cls) -> List[str]:
        """
        Get list of unique field types in the dictionary.

        Returns:
            Sorted list of field types
        """
        df = cls.load_dictionary()
        if "Field Type" in df.columns:
            types = df["Field Type"].dropna().unique().tolist()
            return sorted(types)
        return []

    @classmethod
    def get_diseases(cls) -> List[str]:
        """
        Get list of diseases with columns in the dictionary.

        Returns:
            List of disease codes
        """
        # Known disease columns in the data dictionary
        disease_cols = ["ALS", "BMD", "DMD", "SMA", "LGMD", "FSHD", "Pompe"]
        df = cls.load_dictionary()
        return [d for d in disease_cols if d in df.columns]

    @classmethod
    def search_fields(
        cls,
        search_text: str = "",
        form_filter: Optional[str] = None,
        excel_tab_filter: Optional[str] = None,
        disease_filter: Optional[str] = None,
        field_type_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Search and filter the data dictionary.

        Args:
            search_text: Text to search in field name, description, label
            form_filter: Filter by File/Form name (Demographics, Diagnosis, etc.)
            excel_tab_filter: Filter by Excel Tab/Sheet name
            disease_filter: Filter by disease applicability
            field_type_filter: Filter by field type

        Returns:
            Filtered DataFrame
        """
        df = cls.load_dictionary()

        # Apply text search
        if search_text and search_text.strip():
            search_lower = search_text.lower()
            search_cols = ["Field Name", "Description", "Display Label"]

            mask = pd.Series(False, index=df.index)
            for col in search_cols:
                if col in df.columns:
                    mask |= df[col].astype(str).str.lower().str.contains(
                        search_lower, na=False
                    )
            df = df[mask]

        # Apply form filter (File/Form column)
        if form_filter and form_filter != "All":
            if "File/Form" in df.columns:
                df = df[df["File/Form"] == form_filter]

        # Apply Excel Tab filter
        if excel_tab_filter and excel_tab_filter != "All":
            if "Excel Tab" in df.columns:
                df = df[df["Excel Tab"] == excel_tab_filter]

        # Apply disease filter
        if disease_filter and disease_filter != "All":
            if disease_filter in df.columns:
                # Disease column has "Yes" for applicable fields
                df = df[df[disease_filter] == "Yes"]

        # Apply field type filter
        if field_type_filter and field_type_filter != "All":
            if "Field Type" in df.columns:
                df = df[df["Field Type"] == field_type_filter]

        return df

    @classmethod
    def get_field_detail(cls, field_name: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific field.

        Args:
            field_name: The field name to look up

        Returns:
            Dictionary with field metadata
        """
        df = cls.load_dictionary()

        if "Field Name" not in df.columns:
            return {}

        match = df[df["Field Name"] == field_name]
        if match.empty:
            return {}

        row = match.iloc[0]
        detail = {}

        # Core info - use actual column names from parquet
        for col in [
            "Field Name",
            "Description",
            "Display Label",
            "Field Type",
            "File/Form",
            "Excel Tab",
            "Column Number",
            "Repeat Group Name",
            "Numeric Ranges",
        ]:
            if col in row.index:
                detail[col] = row[col] if pd.notna(row[col]) else ""

        # Disease applicability
        diseases = cls.get_diseases()
        detail["diseases"] = {}
        for disease in diseases:
            if disease in row.index:
                val = row[disease]
                detail["diseases"][disease] = val == "Yes" if pd.notna(val) else False

        return detail

    @classmethod
    def get_summary_stats(cls) -> Dict[str, Any]:
        """
        Get summary statistics for the data dictionary.

        Returns:
            Dictionary with counts and stats
        """
        df = cls.load_dictionary()

        stats = {
            "total_fields": len(df),
            "form_count": 0,
            "excel_tab_count": 0,
            "field_type_count": 0,
            "disease_coverage": {},
        }

        if "File/Form" in df.columns:
            stats["form_count"] = df["File/Form"].nunique()

        if "Excel Tab" in df.columns:
            stats["excel_tab_count"] = df["Excel Tab"].nunique()

        if "Field Type" in df.columns:
            stats["field_type_count"] = df["Field Type"].nunique()

        # Count fields per disease
        for disease in cls.get_diseases():
            if disease in df.columns:
                count = (df[disease] == "Yes").sum()
                stats["disease_coverage"][disease] = count

        return stats

    @classmethod
    def get_display_columns(cls) -> List[str]:
        """
        Get list of columns suitable for table display.

        Returns:
            List of column names
        """
        return [
            "Field Name",
            "Description",
            "File/Form",
            "Excel Tab",
            "Field Type",
        ]

    @classmethod
    def check_field_validity(cls, field_name: str, disease: str) -> dict:
        """
        Check if a field is valid for a given disease.

        Returns dict with:
        - valid: True if field is appropriate for this disease
        - reason: Explanation if not valid
        """
        field_lower = field_name.lower()

        # Check if explicitly mislabeled
        mislabeled = MISLABELED_FIELDS.get(disease, [])
        if field_name in mislabeled or field_lower in mislabeled:
            return {
                'valid': False,
                'reason': f'Mislabeled - should not be marked for {disease}'
            }

        # Check if field name indicates another disease
        for other_disease, prefixes in DISEASE_PREFIXES.items():
            if other_disease == disease:
                continue
            for prefix in prefixes:
                if field_lower.startswith(prefix):
                    return {
                        'valid': False,
                        'reason': f'Field prefix suggests {other_disease}, not {disease}'
                    }

        return {'valid': True, 'reason': ''}

    @classmethod
    def add_validity_flags(cls, df: pd.DataFrame, disease: str) -> pd.DataFrame:
        """
        Add validity flags to indicate if fields are appropriate for the disease.

        Args:
            df: DataFrame with Field Name column
            disease: Disease to check validity for

        Returns:
            DataFrame with 'Valid for Disease' and 'Validity Note' columns
        """
        if not disease or disease == "All":
            return df

        df = df.copy()

        validity_results = df['Field Name'].apply(
            lambda x: cls.check_field_validity(x, disease) if pd.notna(x) else {'valid': True, 'reason': ''}
        )

        df['Valid'] = validity_results.apply(lambda x: x['valid'])
        df['Validity Note'] = validity_results.apply(lambda x: x['reason'])

        return df

    @classmethod
    def add_required_flag(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'Required' column based on * in Display Label.

        Args:
            df: DataFrame with Display Label column

        Returns:
            DataFrame with 'Required' column added
        """
        df = df.copy()
        if "Display Label" in df.columns:
            df["Required"] = df["Display Label"].astype(str).str.startswith("*")
        else:
            df["Required"] = False
        return df

    @classmethod
    def calculate_completeness(
        cls,
        fields_df: pd.DataFrame,
        disease: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate data completeness (% non-null) for fields.

        Args:
            fields_df: DataFrame of fields from search_fields()
            disease: Disease to filter patients by (optional)

        Returns:
            DataFrame with 'Completeness' column added (% of patients with data)
        """
        from src.analytics.cohorts import get_base_cohort, get_disease_cohort

        fields_df = fields_df.copy()

        # Get patient cohort
        if disease and disease != "All":
            try:
                cohort = get_disease_cohort(disease)
            except:
                cohort = get_base_cohort()
        else:
            cohort = get_base_cohort()

        # Map File/Form to cohort data keys
        form_to_data = {
            "Demographics": cohort.get("demographics", pd.DataFrame()),
            "Diagnosis": cohort.get("diagnosis", pd.DataFrame()),
            "Encounter": cohort.get("encounters", pd.DataFrame()),
            "Log": cohort.get("encounters", pd.DataFrame()),  # Log fields often in encounters
            "Discontinuation": pd.DataFrame(),  # Usually separate
        }

        completeness_values = []
        patient_count = cohort.get("count", 0)

        for _, row in fields_df.iterrows():
            field_name = row.get("Field Name", "")
            form = row.get("File/Form", "")

            data_df = form_to_data.get(form, pd.DataFrame())

            if data_df.empty or field_name not in data_df.columns:
                completeness_values.append(None)
            else:
                # Calculate % non-null
                non_null = data_df[field_name].notna().sum()
                # For encounters, count unique patients with data
                if "FACPATID" in data_df.columns:
                    patients_with_data = data_df[data_df[field_name].notna()]["FACPATID"].nunique()
                    pct = (patients_with_data / patient_count * 100) if patient_count > 0 else 0
                else:
                    pct = (non_null / len(data_df) * 100) if len(data_df) > 0 else 0
                completeness_values.append(round(pct, 1))

        fields_df["Completeness %"] = completeness_values
        return fields_df

    @classmethod
    def format_for_display(cls, df: pd.DataFrame, max_rows: int = 500) -> pd.DataFrame:
        """
        Format DataFrame for web display.

        Args:
            df: Raw DataFrame from search
            max_rows: Maximum rows to return

        Returns:
            Formatted DataFrame with display columns
        """
        display_cols = cls.get_display_columns()
        available_cols = [c for c in display_cols if c in df.columns]

        result = df[available_cols].head(max_rows).copy()

        # Truncate long descriptions for display
        if "Description" in result.columns:
            result["Description"] = result["Description"].astype(str).str[:150]

        return result
