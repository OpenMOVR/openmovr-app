"""
Trial Exporter

Specialized exporter for clinical trial data cuts with CDISC SDTM support,
visit windowing, and trial-specific metadata tracking.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from .base_exporter import BaseExporter
from .deidentifier import Deidentifier
from .validators import ExportValidator


class TrialExporter(BaseExporter):
    """
    Exporter for clinical trial data cuts.

    Features:
    - Trial-specific metadata (protocol, sponsor, NCT number)
    - Visit windowing logic for longitudinal data
    - Baseline/follow-up data separation
    - CDISC SDTM formatting support
    - Screening vs. enrolled cohort tracking
    """

    def __init__(self, export_name: str, trial_name: str, protocol_number: str,
                 sponsor: str, nct_number: Optional[str] = None):
        """
        Initialize trial exporter.

        Args:
            export_name: Unique export identifier (e.g., "2025-11-20_sardocor_dmd_v1")
            trial_name: Human-readable trial name
            protocol_number: Sponsor protocol number
            sponsor: Sponsoring organization
            nct_number: ClinicalTrials.gov NCT number (optional)
        """
        super().__init__(
            export_name=export_name,
            recipient=sponsor,
            purpose=f"Clinical trial data cut for {trial_name}"
        )

        # Trial-specific metadata
        self.trial_name = trial_name
        self.protocol_number = protocol_number
        self.sponsor = sponsor
        self.nct_number = nct_number

        # Data organization
        self.baseline_data = None
        self.followup_data = None
        self.screening_data = None
        self.visit_windows = {}

        # De-identification and validation
        self.deidentifier = None
        self.deident_config = {}
        self.validator = ExportValidator()

        # CDISC support
        self.cdisc_format = False
        self.output_formats = ["csv"]  # Default

    def configure_deidentification(self, method: str = "hipaa_safe_harbor",
                                  id_mapping: Optional[Dict[str, str]] = None,
                                  shift_dates: bool = True,
                                  date_anchor: str = "enrollment") -> None:
        """
        Configure PHI de-identification for trial exports.

        Args:
            method: De-identification method
            id_mapping: ID field mappings
            shift_dates: Whether to shift dates
            date_anchor: Reference date for relative dates ("enrollment" or "baseline")
        """
        self.deidentifier = Deidentifier(method=method)
        self.deident_config = {
            "id_mapping": id_mapping or {"FACPATID": "SUBJECT_ID"},
            "shift_dates": shift_dates,
            "date_anchor": date_anchor
        }

        print(f"Configured trial de-identification: {method}")
        print(f"  Date anchor: {date_anchor}")

    def add_baseline_data(self, data: pd.DataFrame,
                         apply_deidentification: bool = True) -> None:
        """
        Add baseline/screening data.

        Args:
            data: Baseline dataset
            apply_deidentification: Whether to de-identify
        """
        if apply_deidentification and self.deidentifier:
            print("\nDe-identifying baseline data...")
            data = self._deidentify_data(data)

        self.baseline_data = data.copy()
        self.add_dataset(data, "baseline")
        print(f"Added baseline data: {len(data)} subjects")

    def add_followup_data(self, data: pd.DataFrame,
                         visit_windows: Optional[Dict[str, Tuple[int, int]]] = None,
                         apply_deidentification: bool = True) -> None:
        """
        Add follow-up/longitudinal data with visit windowing.

        Args:
            data: Follow-up dataset with visit dates
            visit_windows: Dict mapping visit names to (day_min, day_max) windows
                          Example: {"Month_3": (75, 105), "Month_6": (165, 195)}
            apply_deidentification: Whether to de-identify
        """
        if apply_deidentification and self.deidentifier:
            print("\nDe-identifying follow-up data...")
            data = self._deidentify_data(data)

        self.followup_data = data.copy()

        if visit_windows:
            self.visit_windows = visit_windows
            print(f"Visit windows configured: {list(visit_windows.keys())}")

        self.add_dataset(data, "followup")
        print(f"Added follow-up data: {len(data)} records")

    def add_screening_data(self, data: pd.DataFrame,
                          eligibility_field: str = "eligible",
                          apply_deidentification: bool = True) -> None:
        """
        Add screening data for trial eligibility.

        Args:
            data: Screening dataset
            eligibility_field: Column indicating eligibility status
            apply_deidentification: Whether to de-identify
        """
        if apply_deidentification and self.deidentifier:
            print("\nDe-identifying screening data...")
            data = self._deidentify_data(data)

        self.screening_data = data.copy()
        self.add_dataset(data, "screening")

        # Calculate eligibility stats
        if eligibility_field in data.columns:
            eligible_count = data[eligibility_field].sum()
            total_count = len(data)
            print(f"Added screening data: {eligible_count}/{total_count} eligible ({eligible_count/total_count*100:.1f}%)")
        else:
            print(f"Added screening data: {len(data)} subjects")

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
            elif 'date' in col.lower() or '_dt' in col.lower():
                date_fields.append(col)

        # De-identify
        deidentified = self.deidentifier.deidentify(
            data=data,
            id_mapping=self.deident_config.get("id_mapping", {}),
            date_fields=date_fields if self.deident_config.get("shift_dates") else [],
            shift_dates=self.deident_config.get("shift_dates", False),
            remove_fields=[]
        )

        # Validate
        is_clean = self.deidentifier.validate_no_phi(deidentified)
        if not is_clean:
            warnings = self.deidentifier.get_phi_warnings()
            print(f"  WARNING: Potential PHI in {len(warnings)} locations")

        return deidentified

    def set_output_format(self, format_type: str) -> None:
        """
        Set output format for trial export.

        Args:
            format_type: "csv", "cdisc_sdtm", "excel", or "sas"
        """
        if format_type == "cdisc_sdtm":
            self.cdisc_format = True
            self.output_formats = ["csv"]  # CDISC uses CSV
            print("CDISC SDTM format enabled")
        elif format_type in ["csv", "excel", "sas"]:
            self.cdisc_format = False
            self.output_formats = [format_type]
            print(f"Output format: {format_type}")
        else:
            raise ValueError(f"Unknown format: {format_type}")

    def apply_visit_windows(self, data: pd.DataFrame,
                           study_day_column: str = "STUDY_DAY") -> pd.DataFrame:
        """
        Apply visit windowing logic to longitudinal data.

        Args:
            data: Follow-up data with study day column
            study_day_column: Column containing days since baseline

        Returns:
            Data with assigned visit windows
        """
        if not self.visit_windows:
            print("No visit windows configured")
            return data

        data = data.copy()
        data['VISIT'] = 'Unscheduled'

        for visit_name, (day_min, day_max) in self.visit_windows.items():
            mask = (data[study_day_column] >= day_min) & (data[study_day_column] <= day_max)
            data.loc[mask, 'VISIT'] = visit_name

        visit_counts = data['VISIT'].value_counts()
        print(f"\nVisit window assignment:")
        for visit, count in visit_counts.items():
            print(f"  {visit}: {count} records")

        return data

    def validate(self) -> bool:
        """
        Run trial-specific validation checks.

        Returns:
            True if all validations pass
        """
        if not self.datasets:
            print("ERROR: No datasets to validate")
            return False

        print(f"\nValidating trial export: {self.export_id}")
        all_passed = True

        # Get subject ID column
        subject_id_col = self.deident_config.get("id_mapping", {}).get("FACPATID", "SUBJECT_ID")

        # Check each dataset
        for dataset_name, dataset in self.datasets.items():
            print(f"\nValidating {dataset_name}:")

            # Check 1: No duplicate subject IDs (within dataset)
            if subject_id_col in dataset.columns:
                has_duplicates = self.validator.check_duplicates(dataset, subject_id_col)
                if has_duplicates:
                    # For longitudinal data, duplicates are expected
                    if dataset_name == "followup":
                        print(f"  ✓ Multiple records per subject (expected for longitudinal)")
                    else:
                        print(f"  ✗ Unexpected duplicate subject IDs")
                        all_passed = False
                else:
                    print(f"  ✓ No duplicate subject IDs")

            # Check 2: Required fields
            required_fields = [subject_id_col]
            missing = [f for f in required_fields if f not in dataset.columns]
            if missing:
                print(f"  ✗ Missing required fields: {missing}")
                all_passed = False
            else:
                print(f"  ✓ All required fields present")

            # Check 3: Data completeness
            completeness = self.validator.check_completeness(
                dataset,
                required_fields=[subject_id_col],
                threshold=0.95
            )
            print(f"  ✓ Completeness: {completeness.get('overall_complete', 0):.1f}%")

        # Check 4: Baseline-followup subject alignment
        if self.baseline_data is not None and self.followup_data is not None:
            baseline_subjects = set(self.baseline_data[subject_id_col].unique())
            followup_subjects = set(self.followup_data[subject_id_col].unique())

            subjects_in_both = len(baseline_subjects & followup_subjects)
            subjects_baseline_only = len(baseline_subjects - followup_subjects)
            subjects_followup_only = len(followup_subjects - baseline_subjects)

            print(f"\nSubject alignment:")
            print(f"  Subjects with baseline and follow-up: {subjects_in_both}")
            print(f"  Subjects with baseline only: {subjects_baseline_only}")
            print(f"  Subjects with follow-up only: {subjects_followup_only}")

            if subjects_followup_only > 0:
                print(f"  ⚠ WARNING: {subjects_followup_only} subjects have follow-up but no baseline")

        # Check 5: CDISC compliance (if enabled)
        if self.cdisc_format:
            print(f"\nCDISC SDTM format validation:")
            print(f"  ⚠ CDISC validation not fully implemented - manual review required")

        # Overall result
        self.validation_results = {"validated": True, "all_passed": all_passed}

        if all_passed:
            print(f"\n✓ All critical validations passed")
        else:
            print(f"\n✗ Some validations failed - review errors")

        return all_passed

    def generate_manifest(self, inclusion_criteria: List[str],
                         exclusion_criteria: List[str],
                         screening_summary: Optional[Dict] = None) -> Dict:
        """
        Generate trial-specific export manifest.

        Args:
            inclusion_criteria: List of inclusion criteria
            exclusion_criteria: List of exclusion criteria
            screening_summary: Optional screening funnel statistics

        Returns:
            Manifest dictionary
        """
        # Calculate cohort statistics
        cohort_stats = {}
        subject_id_col = self.deident_config.get("id_mapping", {}).get("FACPATID", "SUBJECT_ID")

        if self.baseline_data is not None:
            cohort_stats["baseline_subjects"] = len(self.baseline_data[subject_id_col].unique())

        if self.followup_data is not None:
            cohort_stats["followup_subjects"] = len(self.followup_data[subject_id_col].unique())
            cohort_stats["followup_records"] = len(self.followup_data)

        if self.screening_data is not None:
            cohort_stats["screened_subjects"] = len(self.screening_data)

        # Build trial-specific metadata
        trial_metadata = {
            "trial_name": self.trial_name,
            "protocol_number": self.protocol_number,
            "sponsor": self.sponsor,
            "nct_number": self.nct_number,
            "visit_windows": self.visit_windows,
            "cdisc_format": self.cdisc_format
        }

        if screening_summary:
            trial_metadata["screening_summary"] = screening_summary

        # Generate base manifest
        manifest = super().generate_manifest(
            data_version="trial_export",
            cohort_definition={
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria
            },
            cohort_statistics=cohort_stats,
            trial_metadata=trial_metadata
        )

        return manifest

    def generate_screening_flowchart(self, screening_data: pd.DataFrame,
                                    filter_columns: List[str]) -> Dict:
        """
        Generate screening funnel statistics for CONSORT diagram.

        Args:
            screening_data: Screening dataset with filter results
            filter_columns: List of boolean columns representing each filter

        Returns:
            Dictionary with screening funnel counts
        """
        flowchart = {
            "total_screened": len(screening_data)
        }

        # Track subjects at each filter stage
        remaining = screening_data.copy()
        for i, filter_col in enumerate(filter_columns):
            if filter_col in remaining.columns:
                passed = remaining[filter_col] == True
                failed_count = (~passed).sum()

                flowchart[f"stage_{i+1}"] = {
                    "filter": filter_col,
                    "excluded": int(failed_count),
                    "remaining": int(passed.sum())
                }

                remaining = remaining[passed]

        flowchart["final_eligible"] = len(remaining)

        print(f"\nScreening flowchart:")
        print(f"  Total screened: {flowchart['total_screened']}")
        for stage, stats in flowchart.items():
            if isinstance(stats, dict):
                print(f"  {stats['filter']}: excluded {stats['excluded']}, remaining {stats['remaining']}")
        print(f"  Final eligible: {flowchart['final_eligible']}")

        return flowchart

    def _save_documentation(self, docs_dir) -> None:
        """Save trial-specific documentation."""
        super()._save_documentation(docs_dir)

        # Save trial metadata summary
        if self.manifest:
            trial_summary_path = docs_dir / f"{self.export_name}_trial_summary.txt"
            with open(trial_summary_path, 'w') as f:
                f.write(f"CLINICAL TRIAL DATA EXPORT SUMMARY\n")
                f.write(f"=" * 60 + "\n\n")
                f.write(f"Trial Name: {self.trial_name}\n")
                f.write(f"Protocol Number: {self.protocol_number}\n")
                f.write(f"Sponsor: {self.sponsor}\n")
                if self.nct_number:
                    f.write(f"ClinicalTrials.gov: {self.nct_number}\n")
                f.write(f"\nExport Date: {self.creation_date}\n")
                f.write(f"Export ID: {self.export_id}\n")
                f.write(f"\nDatasets Included:\n")
                for name in self.datasets.keys():
                    f.write(f"  - {name}\n")

            print(f"  Saved: {trial_summary_path.name}")
