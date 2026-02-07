"""
Industry Exporter

Specialized exporter for pharmaceutical companies and academic partners.
Includes HIPAA-compliant de-identification and comprehensive validation.
"""

from typing import Dict, List, Optional
import pandas as pd
from .base_exporter import BaseExporter
from .deidentifier import Deidentifier
from .validators import ExportValidator


class IndustryExporter(BaseExporter):
    """
    Exporter for industry partner data cuts.

    Features:
    - HIPAA-compliant de-identification
    - Comprehensive validation
    - Multiple output formats
    - Auto-generated documentation
    """

    def __init__(self, export_name: str, recipient: str, purpose: str,
                 data_version: str):
        """
        Initialize industry exporter.

        Args:
            export_name: Unique export identifier
            recipient: Receiving organization name
            purpose: Purpose of export
            data_version: Date of source data (YYYY-MM-DD)
        """
        super().__init__(export_name, recipient, purpose)
        self.data_version = data_version

        # De-identification settings
        self.deidentifier = None
        self.deident_config = {}

        # Validator
        self.validator = ExportValidator()

        # Validation report
        self.validation_report = None

    def configure_deidentification(self, method: str = "hipaa_safe_harbor",
                                  id_mapping: Optional[Dict[str, str]] = None,
                                  shift_dates: bool = True,
                                  remove_fields: Optional[List[str]] = None) -> None:
        """
        Configure PHI de-identification.

        Args:
            method: De-identification method ("hipaa_safe_harbor" or "expert_determination")
            id_mapping: Mapping of original IDs to export IDs {"FACPATID": "EXPORT_PATIENT_ID"}
            shift_dates: Whether to shift dates
            remove_fields: List of fields to remove entirely
        """
        self.deidentifier = Deidentifier(method=method)
        self.deident_config = {
            "id_mapping": id_mapping or {"FACPATID": "EXPORT_PATIENT_ID"},
            "shift_dates": shift_dates,
            "remove_fields": remove_fields or []
        }

        print(f"Configured de-identification: {method}")
        if id_mapping:
            print(f"  ID mapping: {id_mapping}")
        if shift_dates:
            print(f"  Date shifting: Enabled")
        if remove_fields:
            print(f"  Removing fields: {remove_fields}")

    def add_dataset(self, data: pd.DataFrame, name: str,
                   apply_deidentification: bool = True) -> None:
        """
        Add dataset with optional de-identification.

        Args:
            data: DataFrame to add
            name: Dataset name
            apply_deidentification: Whether to apply de-identification
        """
        # Apply de-identification if configured
        if apply_deidentification and self.deidentifier:
            print(f"\nDe-identifying dataset '{name}'...")
            data = self._deidentify_data(data)

        # Call parent method
        super().add_dataset(data, name)

    def _deidentify_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply de-identification to dataset.

        Args:
            data: Original dataset

        Returns:
            De-identified dataset
        """
        if not self.deidentifier:
            raise ValueError("De-identification not configured. Call configure_deidentification() first.")

        # Identify date columns
        date_fields = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                date_fields.append(col)
            elif 'date' in col.lower() or 'dob' in col.lower():
                # Try to parse as date
                try:
                    pd.to_datetime(data[col], errors='coerce')
                    date_fields.append(col)
                except:
                    pass

        # De-identify
        deidentified = self.deidentifier.deidentify(
            data=data,
            id_mapping=self.deident_config.get("id_mapping", {}),
            date_fields=date_fields if self.deident_config.get("shift_dates") else [],
            shift_dates=self.deident_config.get("shift_dates", False),
            remove_fields=self.deident_config.get("remove_fields", [])
        )

        # Validate no PHI
        is_clean = self.deidentifier.validate_no_phi(deidentified)
        if not is_clean:
            warnings = self.deidentifier.get_phi_warnings()
            print(f"  WARNING: Potential PHI detected in {len(warnings)} locations")
            for warning in warnings[:5]:  # Show first 5
                print(f"    - {warning}")

        return deidentified

    def validate(self) -> bool:
        """
        Run comprehensive validation on export.

        Returns:
            True if all validations pass
        """
        if not self.datasets:
            print("ERROR: No datasets to validate")
            return False

        print(f"\nValidating export: {self.export_id}")
        all_passed = True

        # Get primary dataset
        primary_dataset = list(self.datasets.values())[0]
        dataset_name = list(self.datasets.keys())[0]

        # Check 1: No duplicate IDs
        export_id_col = self.deident_config.get("id_mapping", {}).get("FACPATID", "EXPORT_PATIENT_ID")
        if export_id_col in primary_dataset.columns:
            has_duplicates = self.validator.check_duplicates(primary_dataset, export_id_col)
            self.validation_results["no_duplicate_ids"] = {
                "status": "passed" if not has_duplicates else "failed",
                "message": f"Found {primary_dataset[export_id_col].duplicated().sum()} duplicates" if has_duplicates else "No duplicates found"
            }
            if has_duplicates:
                all_passed = False
                print(f"  ✗ Duplicate IDs detected")
            else:
                print(f"  ✓ No duplicate IDs")

        # Check 2: Required fields present
        required_fields = [export_id_col]  # At minimum need patient ID
        missing_fields = [f for f in required_fields if f not in primary_dataset.columns]
        self.validation_results["required_fields"] = {
            "status": "passed" if not missing_fields else "failed",
            "message": f"Missing fields: {missing_fields}" if missing_fields else "All required fields present"
        }
        if missing_fields:
            all_passed = False
            print(f"  ✗ Missing required fields: {missing_fields}")
        else:
            print(f"  ✓ All required fields present")

        # Check 3: Data dictionary matches dataset
        if self.data_dictionary is not None:
            dict_vars = set(self.data_dictionary['Variable_Name'].values)
            data_vars = set(primary_dataset.columns)
            missing_in_dict = data_vars - dict_vars
            missing_in_data = dict_vars - data_vars

            dict_match = len(missing_in_dict) == 0 and len(missing_in_data) == 0
            self.validation_results["dictionary_match"] = {
                "status": "passed" if dict_match else "warning",
                "message": f"Dict has {len(dict_vars)} vars, data has {len(data_vars)} vars"
            }
            if not dict_match:
                print(f"  ⚠ Data dictionary mismatch (not critical)")
                if missing_in_dict:
                    print(f"    Missing in dictionary: {list(missing_in_dict)[:5]}")
                if missing_in_data:
                    print(f"    Missing in data: {list(missing_in_data)[:5]}")
            else:
                print(f"  ✓ Data dictionary matches dataset")

        # Check 4: No obvious PHI
        if self.deidentifier:
            phi_check = self.deidentifier.validate_no_phi(primary_dataset)
            self.validation_results["no_phi"] = {
                "status": "passed" if phi_check else "warning",
                "message": "No obvious PHI detected" if phi_check else "Potential PHI detected - review warnings"
            }
            if not phi_check:
                print(f"  ⚠ Potential PHI detected - manual review recommended")
            else:
                print(f"  ✓ No obvious PHI detected")

        # Check 5: Data completeness
        completeness = self.validator.check_completeness(
            primary_dataset,
            required_fields=[export_id_col],
            threshold=0.95
        )
        self.validation_results["completeness"] = {
            "status": "passed",
            "details": completeness
        }
        print(f"  ✓ Data completeness checked")

        # Overall result
        if all_passed:
            print(f"\n✓ All critical validations passed")
        else:
            print(f"\n✗ Some validations failed - review errors")

        return all_passed

    def generate_validation_report(self) -> None:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            print("No validation results. Run validate() first.")
            return

        # Create validation report DataFrame
        report_rows = []
        for check, result in self.validation_results.items():
            report_rows.append({
                "Check": check,
                "Status": result.get("status", "unknown"),
                "Message": result.get("message", ""),
                "Details": str(result.get("details", ""))
            })

        self.validation_report = pd.DataFrame(report_rows)
        print(f"\nGenerated validation report with {len(report_rows)} checks")

    def generate_manifest(self, inclusion_criteria: List[str],
                         exclusion_criteria: List[str],
                         cohort_statistics: Dict) -> Dict:
        """
        Generate export manifest with cohort details.

        Args:
            inclusion_criteria: List of inclusion criteria strings
            exclusion_criteria: List of exclusion criteria strings
            cohort_statistics: Dict with cohort stats (total_patients, sites, etc.)

        Returns:
            Manifest dictionary
        """
        cohort_definition = {
            "inclusion_criteria": inclusion_criteria,
            "exclusion_criteria": exclusion_criteria
        }

        return super().generate_manifest(
            data_version=self.data_version,
            cohort_definition=cohort_definition,
            cohort_statistics=cohort_statistics
        )

    def _save_documentation(self, docs_dir) -> None:
        """Save documentation including validation report."""
        super()._save_documentation(docs_dir)

        # Save validation report
        if self.validation_report is not None:
            report_path = docs_dir / f"{self.export_name}_validation_report.xlsx"
            self.validation_report.to_excel(report_path, index=False, engine='openpyxl')
            print(f"  Saved: {report_path.name}")
