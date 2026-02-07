"""
Site Exporter

Specialized exporter for facility-specific reports and comparative site analytics.
"""

from typing import Dict, List, Optional
import pandas as pd
from .base_exporter import BaseExporter
from .validators import ExportValidator


class SiteExporter(BaseExporter):
    """
    Exporter for site/facility-specific reports.

    Features:
    - Site-level data aggregation
    - Comparative metrics across facilities
    - Site-specific PHI handling (may retain more identifiers with DUA)
    - Facility performance dashboards
    - Enrollment and data quality metrics by site
    """

    def __init__(self, export_name: str, facility_id: str, facility_name: str,
                 include_comparative: bool = False):
        """
        Initialize site exporter.

        Args:
            export_name: Unique export identifier (e.g., "2025-11-20_houston_site_report_v1")
            facility_id: Facility identifier (e.g., "FACILITY_001")
            facility_name: Human-readable facility name (e.g., "Houston Site")
            include_comparative: Whether to include comparative metrics across all sites
        """
        super().__init__(
            export_name=export_name,
            recipient=facility_name,
            purpose=f"Site report for {facility_name}"
        )

        self.facility_id = facility_id
        self.facility_name = facility_name
        self.include_comparative = include_comparative

        # Site data storage
        self.site_data = None
        self.comparative_data = None
        self.site_metrics = {}

        # Validator
        self.validator = ExportValidator()

        # PHI handling (sites may retain more identifiers)
        self.retain_phi = False  # Set to True if DUA allows

    def add_site_data(self, data: pd.DataFrame, facility_id: Optional[str] = None) -> None:
        """
        Add site-specific data.

        Args:
            data: Site-specific dataset
            facility_id: Optional facility ID to filter by (if data contains multiple sites)
        """
        # Filter to specific facility if needed
        if facility_id and 'FACILITY_DISPLAY_ID' in data.columns:
            data = data[data['FACILITY_DISPLAY_ID'] == facility_id].copy()
            print(f"Filtered to facility {facility_id}: {len(data)} records")

        self.site_data = data.copy()
        self.add_dataset(data, f"site_{self.facility_name.lower().replace(' ', '_')}")
        print(f"Added site data for {self.facility_name}: {len(data)} records")

    def add_comparative_data(self, data: pd.DataFrame,
                            facility_id_column: str = "FACILITY_DISPLAY_ID") -> None:
        """
        Add multi-site data for comparative analysis.

        Args:
            data: Dataset with all sites
            facility_id_column: Column containing facility identifiers
        """
        if not self.include_comparative:
            print("Comparative analysis not enabled. Set include_comparative=True to use this.")
            return

        if facility_id_column not in data.columns:
            raise ValueError(f"Column '{facility_id_column}' not found in data")

        self.comparative_data = data.copy()
        print(f"Added comparative data: {len(data)} total records across {data[facility_id_column].nunique()} sites")

    def calculate_site_metrics(self, patient_id_col: str = "FACPATID") -> Dict:
        """
        Calculate key metrics for the site.

        Args:
            patient_id_col: Column containing patient identifiers

        Returns:
            Dictionary of site metrics
        """
        if self.site_data is None:
            raise ValueError("No site data added. Call add_site_data() first.")

        metrics = {}

        # Enrollment metrics
        if patient_id_col in self.site_data.columns:
            metrics['total_patients'] = self.site_data[patient_id_col].nunique()
            metrics['total_records'] = len(self.site_data)
            metrics['records_per_patient'] = round(
                len(self.site_data) / self.site_data[patient_id_col].nunique(), 1
            )

        # Disease distribution
        if 'disease' in self.site_data.columns:
            disease_counts = self.site_data.groupby('disease')[patient_id_col].nunique().to_dict()
            metrics['disease_distribution'] = disease_counts

        # Data completeness
        completeness = {}
        for col in self.site_data.columns:
            pct_complete = (1 - self.site_data[col].isna().sum() / len(self.site_data)) * 100
            completeness[col] = round(pct_complete, 1)
        metrics['data_completeness'] = completeness
        metrics['avg_completeness'] = round(sum(completeness.values()) / len(completeness), 1)

        # Temporal metrics (if date fields exist)
        date_cols = [col for col in self.site_data.columns if 'date' in col.lower()]
        if date_cols:
            for date_col in date_cols:
                try:
                    dates = pd.to_datetime(self.site_data[date_col], errors='coerce')
                    metrics[f'{date_col}_range'] = {
                        'earliest': str(dates.min()),
                        'latest': str(dates.max())
                    }
                except:
                    pass

        self.site_metrics = metrics

        print(f"\nSite Metrics for {self.facility_name}:")
        print(f"  Total Patients: {metrics.get('total_patients', 'N/A')}")
        print(f"  Total Records: {metrics.get('total_records', 'N/A')}")
        print(f"  Avg Completeness: {metrics.get('avg_completeness', 'N/A')}%")

        if 'disease_distribution' in metrics:
            print(f"  Disease Distribution:")
            for disease, count in metrics['disease_distribution'].items():
                print(f"    - {disease}: {count} patients")

        return metrics

    def calculate_comparative_metrics(self, patient_id_col: str = "FACPATID",
                                     facility_id_col: str = "FACILITY_DISPLAY_ID") -> pd.DataFrame:
        """
        Calculate comparative metrics across all sites.

        Args:
            patient_id_col: Patient identifier column
            facility_id_col: Facility identifier column

        Returns:
            DataFrame with comparative metrics by site
        """
        if self.comparative_data is None:
            raise ValueError("No comparative data added. Call add_comparative_data() first.")

        if not self.include_comparative:
            raise ValueError("Comparative analysis not enabled. Set include_comparative=True.")

        # Calculate metrics per site
        site_metrics = []

        for facility in self.comparative_data[facility_id_col].unique():
            site_subset = self.comparative_data[self.comparative_data[facility_id_col] == facility]

            metrics = {
                'Facility_ID': facility,
                'Total_Patients': site_subset[patient_id_col].nunique() if patient_id_col in site_subset.columns else 0,
                'Total_Records': len(site_subset),
                'Avg_Completeness_Pct': round((1 - site_subset.isna().sum().sum() / site_subset.size) * 100, 1)
            }

            # Disease counts
            if 'disease' in site_subset.columns:
                disease_counts = site_subset.groupby('disease')[patient_id_col].nunique()
                for disease, count in disease_counts.items():
                    metrics[f'{disease}_Patients'] = count

            site_metrics.append(metrics)

        comparative_df = pd.DataFrame(site_metrics)

        # Add ranking
        comparative_df['Patient_Rank'] = comparative_df['Total_Patients'].rank(ascending=False, method='min').astype(int)

        # Highlight current site
        comparative_df['Is_Current_Site'] = comparative_df['Facility_ID'] == self.facility_id

        print(f"\nComparative Metrics Summary:")
        print(f"  Total Sites: {len(comparative_df)}")
        print(f"  Current Site Rank: {comparative_df[comparative_df['Is_Current_Site']]['Patient_Rank'].values[0]}")

        self.add_dataset(comparative_df, "site_comparison")

        return comparative_df

    def generate_site_summary(self, patient_id_col: str = "FACPATID") -> None:
        """
        Generate comprehensive site summary report.

        Args:
            patient_id_col: Patient identifier column
        """
        if self.site_data is None:
            raise ValueError("No site data added. Call add_site_data() first.")

        # Calculate site metrics
        self.calculate_site_metrics(patient_id_col)

        # Calculate comparative metrics if available
        if self.comparative_data is not None and self.include_comparative:
            self.calculate_comparative_metrics(patient_id_col)

        print(f"\nGenerated site summary for {self.facility_name}")

    def validate(self) -> bool:
        """
        Run site-specific validation checks.

        Returns:
            True if all validations pass
        """
        if not self.datasets:
            print("ERROR: No datasets to validate")
            return False

        print(f"\nValidating site export: {self.export_id}")
        all_passed = True

        # Validate site data
        if self.site_data is not None:
            print(f"\nValidating site data for {self.facility_name}:")

            # Check 1: Data not empty
            if len(self.site_data) == 0:
                print(f"  ✗ Site data is empty")
                all_passed = False
            else:
                print(f"  ✓ Site has {len(self.site_data)} records")

            # Check 2: Facility ID consistency
            if 'FACILITY_DISPLAY_ID' in self.site_data.columns:
                unique_facilities = self.site_data['FACILITY_DISPLAY_ID'].unique()
                if len(unique_facilities) == 1:
                    print(f"  ✓ Single facility in dataset: {unique_facilities[0]}")
                else:
                    print(f"  ⚠ Multiple facilities in dataset: {list(unique_facilities)}")

            # Check 3: Data completeness
            completeness = self.validator.check_completeness(
                self.site_data,
                required_fields=[],
                threshold=0.80  # Lower threshold for site reports
            )
            print(f"  ✓ Data completeness: {completeness.get('overall_complete', 0):.1f}%")

        # Validate comparative data
        if self.comparative_data is not None:
            print(f"\nValidating comparative data:")

            # Check: Current site is in comparative data
            if 'FACILITY_DISPLAY_ID' in self.comparative_data.columns:
                if self.facility_id in self.comparative_data['FACILITY_DISPLAY_ID'].values:
                    print(f"  ✓ Current site ({self.facility_id}) found in comparative data")
                else:
                    print(f"  ✗ Current site ({self.facility_id}) NOT found in comparative data")
                    all_passed = False

                total_sites = self.comparative_data['FACILITY_DISPLAY_ID'].nunique()
                print(f"  ✓ Comparative data includes {total_sites} sites")

        # Overall result
        self.validation_results = {"validated": True, "all_passed": all_passed}

        if all_passed:
            print(f"\n✓ All validations passed")
        else:
            print(f"\n✗ Some validations failed - review errors")

        return all_passed

    def generate_manifest(self, reporting_period: Optional[str] = None) -> Dict:
        """
        Generate site-specific export manifest.

        Args:
            reporting_period: Optional reporting period (e.g., "2025-Q3")

        Returns:
            Manifest dictionary
        """
        # Build site-specific metadata
        site_metadata = {
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "reporting_period": reporting_period,
            "site_metrics": self.site_metrics,
            "comparative_included": self.include_comparative
        }

        # Generate base manifest
        manifest = super().generate_manifest(
            data_version="site_report",
            cohort_definition={
                "facility_filter": self.facility_id
            },
            cohort_statistics=self.site_metrics,
            site_metadata=site_metadata
        )

        return manifest

    def generate_enrollment_trend(self, data: pd.DataFrame,
                                  date_column: str = "enrollment_date",
                                  patient_id_col: str = "FACPATID",
                                  frequency: str = "M") -> pd.DataFrame:
        """
        Generate enrollment trend over time for the site.

        Args:
            data: Dataset with enrollment dates
            date_column: Column containing enrollment/visit dates
            patient_id_col: Patient identifier column
            frequency: Frequency for aggregation ('M'=monthly, 'Q'=quarterly, 'Y'=yearly)

        Returns:
            DataFrame with enrollment counts over time
        """
        if date_column not in data.columns:
            raise ValueError(f"Date column '{date_column}' not found")

        # Convert to datetime
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

        # Remove invalid dates
        data = data[data[date_column].notna()]

        # Group by period
        data['period'] = data[date_column].dt.to_period(frequency)

        # Count unique patients per period
        enrollment_trend = data.groupby('period')[patient_id_col].nunique().reset_index()
        enrollment_trend.columns = ['Period', 'New_Enrollments']
        enrollment_trend['Period'] = enrollment_trend['Period'].astype(str)

        # Add cumulative
        enrollment_trend['Cumulative_Enrollments'] = enrollment_trend['New_Enrollments'].cumsum()

        print(f"\nEnrollment Trend ({frequency}):")
        print(f"  Periods: {len(enrollment_trend)}")
        print(f"  Total Enrollments: {enrollment_trend['Cumulative_Enrollments'].iloc[-1]}")

        self.add_dataset(enrollment_trend, "enrollment_trend")

        return enrollment_trend

    def _save_documentation(self, docs_dir) -> None:
        """Save site-specific documentation."""
        super()._save_documentation(docs_dir)

        # Save site summary
        if self.site_metrics:
            summary_path = docs_dir / f"{self.export_name}_site_summary.txt"
            with open(summary_path, 'w') as f:
                f.write(f"SITE REPORT SUMMARY\n")
                f.write(f"=" * 60 + "\n\n")
                f.write(f"Facility: {self.facility_name}\n")
                f.write(f"Facility ID: {self.facility_id}\n")
                f.write(f"Report Date: {self.creation_date}\n")
                f.write(f"\nKey Metrics:\n")

                for key, value in self.site_metrics.items():
                    if not isinstance(value, dict):
                        f.write(f"  {key}: {value}\n")

                if 'disease_distribution' in self.site_metrics:
                    f.write(f"\nDisease Distribution:\n")
                    for disease, count in self.site_metrics['disease_distribution'].items():
                        f.write(f"  {disease}: {count} patients\n")

            print(f"  Saved: {summary_path.name}")
