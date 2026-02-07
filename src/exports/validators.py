"""
Export Validators

Comprehensive validation utilities for data exports.
"""

from typing import Dict, List
import pandas as pd
import re


class ExportValidator:
    """
    Validator for export data quality and compliance.
    """

    def __init__(self):
        """Initialize validator."""
        self.results = {}

    def check_duplicates(self, data: pd.DataFrame, key: str) -> bool:
        """
        Check for duplicate values in key column.

        Args:
            data: DataFrame to check
            key: Column name to check for duplicates

        Returns:
            True if duplicates found, False otherwise
        """
        if key not in data.columns:
            raise ValueError(f"Key column '{key}' not found in data")

        duplicates = data[key].duplicated().sum()
        return duplicates > 0

    def check_completeness(self, data: pd.DataFrame, required_fields: List[str],
                          threshold: float = 0.95) -> Dict:
        """
        Check data completeness for required fields.

        Args:
            data: DataFrame to check
            required_fields: List of required column names
            threshold: Minimum completeness threshold (0-1)

        Returns:
            Dict with completeness results per field
        """
        completeness = {}

        for field in required_fields:
            if field not in data.columns:
                completeness[field] = {
                    "present": False,
                    "completeness": 0.0,
                    "meets_threshold": False
                }
            else:
                non_null_count = data[field].notna().sum()
                total_count = len(data)
                comp_rate = non_null_count / total_count if total_count > 0 else 0.0

                completeness[field] = {
                    "present": True,
                    "completeness": round(comp_rate, 4),
                    "meets_threshold": comp_rate >= threshold,
                    "missing_count": total_count - non_null_count
                }

        return completeness

    def check_value_ranges(self, data: pd.DataFrame, field: str,
                          min_val: float = None, max_val: float = None) -> Dict:
        """
        Check if values are within expected range.

        Args:
            data: DataFrame to check
            field: Column name
            min_val: Minimum expected value
            max_val: Maximum expected value

        Returns:
            Dict with range validation results
        """
        if field not in data.columns:
            return {"status": "error", "message": f"Field '{field}' not found"}

        values = data[field].dropna()

        if len(values) == 0:
            return {"status": "warning", "message": "No non-null values"}

        result = {
            "field": field,
            "min_value": float(values.min()),
            "max_value": float(values.max()),
            "within_range": True,
            "outliers": 0
        }

        outliers = 0
        if min_val is not None:
            outliers += (values < min_val).sum()
        if max_val is not None:
            outliers += (values > max_val).sum()

        if outliers > 0:
            result["within_range"] = False
            result["outliers"] = outliers

        result["status"] = "passed" if result["within_range"] else "warning"
        return result

    def check_categorical_values(self, data: pd.DataFrame, field: str,
                                 valid_values: List) -> Dict:
        """
        Check if categorical values are all valid.

        Args:
            data: DataFrame to check
            field: Column name
            valid_values: List of valid category values

        Returns:
            Dict with categorical validation results
        """
        if field not in data.columns:
            return {"status": "error", "message": f"Field '{field}' not found"}

        values = data[field].dropna().unique()
        invalid_values = [v for v in values if v not in valid_values]

        result = {
            "field": field,
            "unique_values": len(values),
            "valid_values": valid_values,
            "all_valid": len(invalid_values) == 0,
            "invalid_values": invalid_values
        }

        result["status"] = "passed" if result["all_valid"] else "failed"
        return result

    def validate_export(self, data: pd.DataFrame, data_dictionary: pd.DataFrame = None,
                       export_manifest: Dict = None) -> Dict:
        """
        Run comprehensive export validation.

        Args:
            data: Export DataFrame
            data_dictionary: Optional data dictionary
            export_manifest: Optional export manifest

        Returns:
            Validation results dictionary
        """
        results = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_records": len(data),
            "total_fields": len(data.columns),
            "checks": {}
        }

        # Basic checks
        results["checks"]["non_empty"] = {
            "status": "passed" if len(data) > 0 else "failed",
            "message": f"{len(data)} records"
        }

        results["checks"]["has_columns"] = {
            "status": "passed" if len(data.columns) > 0 else "failed",
            "message": f"{len(data.columns)} columns"
        }

        # If data dictionary provided, check match
        if data_dictionary is not None:
            dict_vars = set(data_dictionary['Variable_Name'].values)
            data_vars = set(data.columns)
            match = dict_vars == data_vars

            results["checks"]["dictionary_match"] = {
                "status": "passed" if match else "warning",
                "message": "Data dictionary matches dataset" if match else "Mismatch between dictionary and data"
            }

        self.results = results
        return results

    def generate_report(self, results: Dict = None, output_path: str = None) -> pd.DataFrame:
        """
        Generate validation report DataFrame.

        Args:
            results: Validation results (uses self.results if None)
            output_path: Optional path to save report

        Returns:
            DataFrame with validation report
        """
        if results is None:
            results = self.results

        if not results:
            return pd.DataFrame()

        # Convert to DataFrame
        report_rows = []
        for check_name, check_result in results.get("checks", {}).items():
            report_rows.append({
                "Check": check_name,
                "Status": check_result.get("status", "unknown"),
                "Message": check_result.get("message", "")
            })

        report_df = pd.DataFrame(report_rows)

        if output_path:
            if output_path.endswith('.xlsx'):
                report_df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                report_df.to_csv(output_path, index=False)
            print(f"Validation report saved to: {output_path}")

        return report_df
