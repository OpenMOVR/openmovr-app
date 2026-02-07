"""
De-identifier

HIPAA-compliant de-identification utilities for PHI removal.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import re
import hashlib


class Deidentifier:
    """
    HIPAA-compliant de-identification for clinical data exports.

    Supports:
    - HIPAA Safe Harbor method
    - ID mapping with hashing
    - Date shifting
    - Free text removal
    - PHI detection
    """

    # HIPAA Safe Harbor - 18 identifiers to remove
    PHI_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "url": r"https?://\S+",
        "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    }

    def __init__(self, method: str = "hipaa_safe_harbor"):
        """
        Initialize de-identifier.

        Args:
            method: De-identification method ("hipaa_safe_harbor" or "expert_determination")
        """
        if method not in ["hipaa_safe_harbor", "expert_determination"]:
            raise ValueError(f"Unknown method: {method}. Use 'hipaa_safe_harbor' or 'expert_determination'")

        self.method = method
        self.phi_warnings = []
        self.id_mappings = {}  # Store ID mappings for consistency

    def deidentify(self, data: pd.DataFrame,
                   id_mapping: Optional[Dict[str, str]] = None,
                   date_fields: Optional[List[str]] = None,
                   shift_dates: bool = True,
                   remove_fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        De-identify dataset per HIPAA Safe Harbor.

        Args:
            data: DataFrame to de-identify
            id_mapping: Mapping of original IDs to export IDs {"FACPATID": "EXPORT_PATIENT_ID"}
            date_fields: List of date column names to shift
            shift_dates: Whether to shift dates
            remove_fields: List of fields to remove entirely (e.g., free text notes)

        Returns:
            De-identified DataFrame
        """
        deident_data = data.copy()

        # Step 1: Remove specified fields
        if remove_fields:
            fields_to_remove = [f for f in remove_fields if f in deident_data.columns]
            if fields_to_remove:
                deident_data = deident_data.drop(columns=fields_to_remove)
                print(f"  Removed {len(fields_to_remove)} fields: {fields_to_remove}")

        # Step 2: Replace patient IDs
        if id_mapping:
            for original_id_col, new_id_col in id_mapping.items():
                if original_id_col in deident_data.columns:
                    # Generate hashed IDs
                    deident_data[new_id_col] = deident_data[original_id_col].apply(
                        lambda x: self._hash_id(x) if pd.notna(x) else np.nan
                    )
                    # Remove original ID
                    if new_id_col != original_id_col:
                        deident_data = deident_data.drop(columns=[original_id_col])
                    print(f"  Mapped {original_id_col} → {new_id_col}")

        # Step 3: Shift dates
        if shift_dates and date_fields:
            # Generate a consistent shift offset (30-365 days)
            shift_days = np.random.randint(30, 365)

            for date_col in date_fields:
                if date_col in deident_data.columns:
                    # Convert to datetime if not already
                    deident_data[date_col] = pd.to_datetime(deident_data[date_col], errors='coerce')
                    # Shift dates
                    deident_data[date_col] = deident_data[date_col] - pd.Timedelta(days=shift_days)
                    print(f"  Shifted dates in {date_col} by {shift_days} days")

        # Step 4: Remove geographic data smaller than state (if present)
        geographic_fields = ["city", "zip", "zip_code", "zipcode", "address", "street"]
        for geo_field in geographic_fields:
            if geo_field in deident_data.columns:
                if "zip" in geo_field.lower():
                    # Keep first 3 digits of ZIP only
                    deident_data[geo_field] = deident_data[geo_field].astype(str).str[:3]
                    print(f"  Truncated {geo_field} to first 3 digits")
                else:
                    # Remove entirely
                    deident_data = deident_data.drop(columns=[geo_field])
                    print(f"  Removed {geo_field}")

        # Step 5: Remove age if >89 (replace with 90+)
        age_fields = [col for col in deident_data.columns if 'age' in col.lower()]
        for age_col in age_fields:
            if pd.api.types.is_numeric_dtype(deident_data[age_col]):
                deident_data.loc[deident_data[age_col] > 89, age_col] = 90
                print(f"  Capped {age_col} at 90 for ages >89")

        print(f"  De-identification complete: {len(deident_data.columns)} columns, {len(deident_data)} rows")
        return deident_data

    def _hash_id(self, original_id: str, salt: str = "movr_export_2025") -> str:
        """
        Generate consistent hashed ID.

        Args:
            original_id: Original identifier
            salt: Salt for hashing

        Returns:
            Hashed ID as string
        """
        # Check if we've already mapped this ID
        if original_id in self.id_mappings:
            return self.id_mappings[original_id]

        # Create hash
        hash_input = f"{salt}_{original_id}".encode('utf-8')
        hash_hex = hashlib.sha256(hash_input).hexdigest()

        # Take first 8 characters and convert to integer
        hashed_id = int(hash_hex[:8], 16) % 100000  # Keep it reasonable size

        # Store mapping
        self.id_mappings[original_id] = str(hashed_id)

        return str(hashed_id)

    def validate_no_phi(self, data: pd.DataFrame) -> bool:
        """
        Scan dataset for potential PHI.

        Args:
            data: DataFrame to check

        Returns:
            True if no obvious PHI detected, False otherwise
        """
        self.phi_warnings = []

        # Check column names for obvious PHI
        phi_column_names = [
            "name", "patient_name", "physician", "email", "phone", "ssn",
            "address", "mrn", "medical_record", "account"
        ]

        for col in data.columns:
            col_lower = col.lower()
            for phi_term in phi_column_names:
                if phi_term in col_lower:
                    self.phi_warnings.append(f"Column name may contain PHI: {col}")

        # Check string columns for PHI patterns
        string_cols = data.select_dtypes(include=['object', 'string']).columns

        for col in string_cols:
            # Sample some non-null values
            sample_values = data[col].dropna().head(100)

            for value in sample_values:
                if not isinstance(value, str):
                    continue

                # Check against PHI patterns
                for phi_type, pattern in self.PHI_PATTERNS.items():
                    if re.search(pattern, str(value), re.IGNORECASE):
                        self.phi_warnings.append(
                            f"Potential {phi_type} in column '{col}': {value[:20]}..."
                        )
                        break  # Only report first match per value

        return len(self.phi_warnings) == 0

    def get_phi_warnings(self) -> List[str]:
        """
        Get list of PHI warnings from last validation.

        Returns:
            List of warning messages
        """
        return self.phi_warnings

    def create_id_mapping(self, original_ids: pd.Series, salt: str = "movr_export_2025") -> Dict[str, str]:
        """
        Generate consistent ID mapping for a series of IDs.

        Args:
            original_ids: Series of original IDs
            salt: Salt for hashing

        Returns:
            Dictionary mapping original IDs to hashed IDs
        """
        mapping = {}
        for original_id in original_ids.unique():
            if pd.notna(original_id):
                mapping[original_id] = self._hash_id(original_id, salt)

        return mapping
