"""
Cohort API Layer

Provides a facade over the core analytics library for cohort operations.
This abstraction makes the webapp easier to extract to a separate repository.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.cohorts import (
    get_base_cohort as _get_base_cohort,
    get_disease_cohort as _get_disease_cohort,
    get_disease_counts as _get_disease_counts,
    apply_to_table as _apply_to_table,
    filter_cohort_by_patients as _filter_cohort_by_patients
)


class CohortAPI:
    """
    API for cohort operations.

    This class provides a clean interface to the core cohort functionality,
    with methods optimized for web display (e.g., converting to dicts, formatting).
    """

    @staticmethod
    def get_base_cohort(include_usndr: bool = False) -> Dict[str, Any]:
        """
        Get the MOVR base cohort.

        Args:
            include_usndr: Include USNDR legacy patients

        Returns:
            Dictionary with cohort data and metadata
        """
        return _get_base_cohort(include_usndr=include_usndr)

    @staticmethod
    def get_disease_cohort(disease: str) -> Dict[str, Any]:
        """
        Get a disease-specific cohort.

        Args:
            disease: Disease code (DMD, ALS, SMA, etc.)

        Returns:
            Dictionary with disease cohort data
        """
        return _get_disease_cohort(disease)

    @staticmethod
    def get_disease_counts(base_cohort: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Get disease distribution as a list of dictionaries (web-friendly).

        Args:
            base_cohort: Base cohort dict (uses cached if None)

        Returns:
            List of dicts with disease, patient_count, percentage
        """
        df = _get_disease_counts(base_cohort)
        return df.to_dict('records')

    @staticmethod
    def get_disease_counts_df(base_cohort: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get disease distribution as DataFrame.

        Args:
            base_cohort: Base cohort dict (uses cached if None)

        Returns:
            DataFrame with disease distribution
        """
        return _get_disease_counts(base_cohort)

    @staticmethod
    def get_available_diseases() -> List[str]:
        """
        Get list of available diseases in the database.

        Returns:
            List of disease codes
        """
        counts_df = _get_disease_counts()
        return counts_df['disease'].tolist()

    @staticmethod
    def apply_to_table(table_name: str, patient_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load a table and filter by patient IDs.

        Args:
            table_name: Name of parquet table
            patient_ids: List of patient IDs to filter (None = all)

        Returns:
            Filtered DataFrame
        """
        return _apply_to_table(table_name, patient_ids)

    @staticmethod
    def filter_cohort(patient_ids: List[str],
                     base_cohort: Optional[Dict] = None,
                     cohort_name: str = "filtered") -> Dict[str, Any]:
        """
        Create a custom cohort from a list of patient IDs.

        Args:
            patient_ids: List of patient IDs
            base_cohort: Source cohort (uses base if None)
            cohort_name: Name for the cohort

        Returns:
            Filtered cohort dictionary
        """
        return _filter_cohort_by_patients(patient_ids, base_cohort, cohort_name)

    @staticmethod
    def apply_filters(
        disease_cohort: Dict[str, Any],
        active_filters: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Apply active filters to a disease cohort and return a filtered cohort.

        Args:
            disease_cohort: Disease cohort dict from get_disease_cohort()
            active_filters: Dict from DiseaseFilterRenderer.render_filters()

        Returns:
            Filtered cohort dict with the same structure as the input.
        """
        if not active_filters:
            return disease_cohort

        table_map = {
            "demographics": "demographics",
            "diagnosis": "diagnosis",
            "encounters": "encounters",
            "medications": "medications",
        }

        # Group filters by source table
        filters_by_table: Dict[str, list] = defaultdict(list)
        for _field, finfo in active_filters.items():
            filters_by_table[finfo["source_table"]].append(finfo)

        patient_sets: List[set] = []

        for table_name, filters in filters_by_table.items():
            df_key = table_map.get(table_name, table_name)
            df = disease_cohort.get(df_key)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue

            mask = pd.Series(True, index=df.index)

            for f in filters:
                wtype = f["widget_type"]
                field = f["field"]
                value = f["value"]

                if wtype == "age_range" and f.get("compute_from") == "dob":
                    dob = pd.to_datetime(df.get("dob"), errors="coerce")
                    age = (pd.Timestamp.now() - dob).dt.days / 365.25
                    mask &= (age >= value[0]) & (age <= value[1])

                elif wtype == "multiselect":
                    col = df[field].astype(str).str.strip()
                    mask &= col.isin(value)

                elif wtype == "text_search":
                    col = df[field].astype(str).str.lower()
                    mask &= col.str.contains(value.lower(), na=False)

                elif wtype == "range_slider":
                    numeric_col = pd.to_numeric(df[field], errors="coerce")
                    mask &= (numeric_col >= value[0]) & (numeric_col <= value[1])

                elif wtype == "date_range":
                    date_col = pd.to_datetime(df[field], errors="coerce")
                    mask &= (date_col >= value[0]) & (date_col <= value[1])

                elif wtype == "checkbox":
                    mask &= df[field].isin([True, "Yes", "yes", 1, "1"])

            facpatid_col = "FACPATID"
            if facpatid_col in df.columns:
                matched = set(df.loc[mask, facpatid_col].dropna().unique())
                patient_sets.append(matched)

        # Intersect patient sets across tables (AND logic)
        if patient_sets:
            final_patients = set.intersection(*patient_sets)
        else:
            final_patients = set(disease_cohort.get("patient_ids", []))

        return _filter_cohort_by_patients(
            list(final_patients), disease_cohort, "filtered"
        )

    @staticmethod
    def get_cohort_summary(cohort: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of cohort statistics (web-friendly format).

        Args:
            cohort: Cohort dictionary

        Returns:
            Summary statistics as dict
        """
        return {
            'total_patients': cohort.get('count', 0),
            'demographics_records': len(cohort.get('demographics', [])),
            'diagnosis_records': len(cohort.get('diagnosis', [])),
            'encounter_records': len(cohort.get('encounters', [])),
            'medication_records': len(cohort.get('medications', [])),
            'dstype_distribution': cohort.get('dstype_counts', {}),
            'facility_count': cohort.get('facility_info', {}).get('total_facilities', 0)
        }
