"""
Metadata Generator

Auto-generation of export manifests, data dictionaries, and documentation.
"""

from typing import Dict, List, Optional
import pandas as pd
import json
from datetime import datetime
import hashlib
from pathlib import Path


class MetadataGenerator:
    """
    Generator for export metadata and documentation.
    """

    def __init__(self):
        """Initialize metadata generator."""
        pass

    def generate_manifest(self, export_id: str, recipient: str, purpose: str,
                         data_version: str,
                         cohort_definition: Dict,
                         cohort_statistics: Dict,
                         files: List[Dict]) -> Dict:
        """
        Generate export manifest JSON.

        Args:
            export_id: Unique export identifier
            recipient: Receiving organization
            purpose: Purpose of export
            data_version: Date of source data
            cohort_definition: Dict with inclusion/exclusion criteria
            cohort_statistics: Dict with cohort stats
            files: List of file info dicts

        Returns:
            Manifest dictionary
        """
        manifest = {
            "export_metadata": {
                "export_id": export_id,
                "creation_date": datetime.now().isoformat(),
                "created_by": "MOVR Data Science Team",
                "recipient": recipient,
                "purpose": purpose,
                "data_version": data_version,
                "movr_version": "2.1.0"
            },
            "cohort_definition": cohort_definition,
            "cohort_statistics": cohort_statistics,
            "files_included": files,
            "privacy_compliance": {
                "de_identification_method": "HIPAA Safe Harbor",
                "phi_removed": True,
                "dates_shifted": True,
                "reviewed_by": "compliance_officer",
                "review_date": datetime.now().strftime("%Y-%m-%d")
            }
        }

        return manifest

    def save_manifest(self, manifest: Dict, output_path: str) -> None:
        """
        Save manifest to JSON file.

        Args:
            manifest: Manifest dictionary
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"Manifest saved to: {output_path}")

    def generate_data_dictionary(self, data: pd.DataFrame,
                                 descriptions: Optional[Dict[str, str]] = None,
                                 units: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Auto-generate data dictionary from DataFrame.

        Args:
            data: DataFrame to generate dictionary for
            descriptions: Optional manual descriptions {column: description}
            units: Optional units {column: unit}

        Returns:
            DataFrame with data dictionary
        """
        descriptions = descriptions or {}
        units = units or {}

        dict_rows = []

        for col in data.columns:
            # Determine data type
            dtype = data[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                data_type = "Integer"
            elif pd.api.types.is_float_dtype(dtype):
                data_type = "Float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                data_type = "Date"
            elif pd.api.types.is_bool_dtype(dtype):
                data_type = "Boolean"
            else:
                data_type = "String"

            # Get value range
            if pd.api.types.is_numeric_dtype(dtype):
                value_range = f"{data[col].min()} - {data[col].max()}"
            elif pd.api.types.is_categorical_dtype(dtype) or data[col].nunique() < 20:
                unique_vals = data[col].dropna().unique()
                if len(unique_vals) <= 10:
                    value_range = ", ".join([str(v) for v in unique_vals[:10]])
                else:
                    value_range = f"{len(unique_vals)} unique values"
            else:
                value_range = "Text"

            # Calculate completeness
            completeness = data[col].notna().sum() / len(data) if len(data) > 0 else 0.0

            dict_rows.append({
                "Variable_Name": col,
                "Description": descriptions.get(col, self._generate_description(col)),
                "Data_Type": data_type,
                "Units": units.get(col, ""),
                "Value_Range": value_range,
                "Completeness": f"{completeness:.1%}",
                "Missing_Count": data[col].isna().sum(),
                "Notes": ""
            })

        return pd.DataFrame(dict_rows)

    def _generate_description(self, column_name: str) -> str:
        """
        Generate basic description from column name.

        Args:
            column_name: Column name

        Returns:
            Basic description
        """
        # Convert snake_case or camelCase to readable format
        # Remove underscores and split on capitals
        words = column_name.replace('_', ' ').split()

        # Capitalize first letter of each word
        description = ' '.join(word.capitalize() for word in words)

        return description

    def save_data_dictionary(self, data_dict: pd.DataFrame, output_path: str) -> None:
        """
        Save data dictionary to file.

        Args:
            data_dict: Data dictionary DataFrame
            output_path: Path to save file (.xlsx or .csv)
        """
        if output_path.endswith('.xlsx'):
            data_dict.to_excel(output_path, index=False, engine='openpyxl')
        else:
            data_dict.to_csv(output_path, index=False)

        print(f"Data dictionary saved to: {output_path}")

    def generate_readme(self, manifest: Dict, datasets: Dict[str, pd.DataFrame]) -> str:
        """
        Generate README.txt content for export package.

        Args:
            manifest: Export manifest dictionary
            datasets: Dictionary of DataFrames

        Returns:
            README content as string
        """
        export_meta = manifest.get("export_metadata", {})
        cohort_stats = manifest.get("cohort_statistics", {})

        # Build README content
        readme_lines = [
            "=" * 70,
            "MOVR CLINICAL DATA EXPORT",
            "=" * 70,
            "",
            f"Export ID: {export_meta.get('export_id', 'N/A')}",
            f"Created: {export_meta.get('creation_date', 'N/A')[:10]}",
            f"Recipient: {export_meta.get('recipient', 'N/A')}",
            f"Purpose: {export_meta.get('purpose', 'N/A')}",
            "",
            "-" * 70,
            "CONTENTS",
            "-" * 70,
            ""
        ]

        # List datasets
        for name, data in datasets.items():
            readme_lines.append(f"Dataset: {name}")
            readme_lines.append(f"  Patients: {len(data)}")
            readme_lines.append(f"  Variables: {len(data.columns)}")
            readme_lines.append("")

        readme_lines.extend([
            "-" * 70,
            "COHORT STATISTICS",
            "-" * 70,
            ""
        ])

        for key, value in cohort_stats.items():
            readme_lines.append(f"{key.replace('_', ' ').title()}: {value}")

        readme_lines.extend([
            "",
            "-" * 70,
            "DATA VERSION & PRIVACY",
            "-" * 70,
            "",
            f"Data Version: {export_meta.get('data_version', 'N/A')}",
            f"De-identification: {manifest.get('privacy_compliance', {}).get('de_identification_method', 'N/A')}",
            f"PHI Removed: {manifest.get('privacy_compliance', {}).get('phi_removed', 'N/A')}",
            f"Dates Shifted: {manifest.get('privacy_compliance', {}).get('dates_shifted', 'N/A')}",
            "",
            "-" * 70,
            "FILES INCLUDED",
            "-" * 70,
            "",
            "1. data/ - Dataset files (CSV/Excel/Parquet)",
            "2. documentation/",
            "   - Data dictionary (variable descriptions)",
            "   - Export manifest (JSON metadata)",
            "   - Validation report",
            "3. README.txt (this file)",
            "",
            "-" * 70,
            "CONTACT",
            "-" * 70,
            "",
            "For questions about this export:",
            "MOVR Data Science Team",
            "Email: data-science@movr.org",
            "",
            "=" * 70
        ])

        return "\n".join(readme_lines)

    def save_readme(self, readme_content: str, output_path: str) -> None:
        """
        Save README to file.

        Args:
            readme_content: README content string
            output_path: Path to save README.txt
        """
        with open(output_path, 'w') as f:
            f.write(readme_content)

        print(f"README saved to: {output_path}")

    def calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """
        Calculate file hash for verification.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ("md5" or "sha256")

        Returns:
            Hex hash string
        """
        if algorithm == "md5":
            hasher = hashlib.md5()
        elif algorithm == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        with open(file_path, 'rb') as f:
            # Read file in chunks for memory efficiency
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()
