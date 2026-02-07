"""
Cohort Builder - Orchestrates data loading using disease configs and CRF metadata.

This module bridges the gap between disease configurations (what fields to use)
and actual data loading (where fields come from and how to handle temporal data).

Author: MOVR Data Science Team
Date: 2025-11-20
"""

import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union
import warnings


class CohortBuilder:
    """
    Build cohorts from disease configurations with proper table joining.

    Handles:
    - Static vs longitudinal field sources
    - Visit selection strategies (latest, first, all)
    - Proper key management (FACPATID vs CASE_ID)
    - Table joins (maindata, diagnosis, encounter)

    Usage:
        from src.config import load_disease_config
        from src.analytics.cohort_builder import CohortBuilder

        dmd = load_disease_config("DMD", "data/datadictionary.parquet")
        builder = CohortBuilder(dmd, data_dir="data")

        # Latest visit snapshot
        df = builder.build_cohort(['mobility', 'cardiac'], visit_strategy='latest_visit')

        # Longitudinal dataset
        df_long = builder.build_cohort(['mobility'], visit_strategy='all_visits')
    """

    def __init__(self, disease_config, data_dir: str = "data", config_dir: str = "config"):
        """
        Initialize CohortBuilder.

        Args:
            disease_config: DiseaseConfig instance
            data_dir: Path to parquet data files
            config_dir: Path to config directory (for CRF metadata)
        """
        self.disease_config = disease_config
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)

        # Load CRF metadata
        self.crf_metadata = self._load_crf_metadata()

        # Get data strategy from disease config
        self.data_strategy = disease_config.config.get('data_strategy', {})

    def _load_crf_metadata(self) -> Dict:
        """Load CRF metadata configuration."""
        crf_path = self.config_dir / "crf_metadata.yaml"

        if not crf_path.exists():
            raise FileNotFoundError(
                f"CRF metadata not found: {crf_path}\n"
                f"This file maps fields to their source tables."
            )

        with open(crf_path, 'r') as f:
            return yaml.safe_load(f)

    def build_cohort(
        self,
        domains: List[str],
        visit_strategy: Optional[str] = None,
        include_demographics: bool = True,
        include_diagnosis: bool = True,
        additional_fields: Optional[List[str]] = None,
        include_usndr: bool = False,
        validate_enrollment: bool = True
    ) -> pd.DataFrame:
        """
        Build cohort dataframe with specified domains.

        Args:
            domains: List of clinical domains (e.g., ['mobility', 'cardiac'])
            visit_strategy: 'latest_visit', 'first_visit', or 'all_visits'
                          If None, uses default from disease config
            include_demographics: Include core demographic fields (FACPATID, dob, gender)
            include_diagnosis: Include diagnosis table fields
            additional_fields: Extra fields to include beyond domain fields
            include_usndr: Include USNDR patients (default: False - MOVR study data only)
            validate_enrollment: Validate patient exists in all 3 MainData forms (default: True)

        Returns:
            DataFrame with requested fields
            - One row per patient (latest_visit, first_visit)
            - Multiple rows per patient (all_visits)

        Data Quality Filters (applied by default):
            - Validated enrollment: Patient must exist in Demographics, Diagnosis, AND Encounter
            - MOVR study data only: Excludes USNDR patients (usndr != 1)

        Example:
            # Trial screening with latest data
            df = builder.build_cohort(
                domains=['mobility', 'cardiac', 'pulmonary'],
                visit_strategy='latest_visit'
            )
            # Result: One row per patient with most recent clinical data

            # Natural history longitudinal
            df = builder.build_cohort(
                domains=['mobility'],
                visit_strategy='all_visits'
            )
            # Result: Multiple rows per patient, one per visit
        """
        # Use default visit strategy if not specified
        if visit_strategy is None:
            visit_strategy = self.data_strategy.get('default_mode', 'latest_visit')

        print(f"\n{'='*70}")
        print(f"BUILDING COHORT: {self.disease_config.disease}")
        print(f"{'='*70}")
        print(f"Domains: {domains}")
        print(f"Visit Strategy: {visit_strategy}")
        print(f"Include Demographics: {include_demographics}")
        print(f"Include Diagnosis: {include_diagnosis}")
        print(f"Data Quality Filters:")
        print(f"  - Validate Enrollment: {validate_enrollment}")
        print(f"  - Include USNDR: {include_usndr}")

        # 1. Collect all fields needed
        all_fields = self._collect_fields(domains, additional_fields)
        print(f"\nTotal fields requested: {len(all_fields)}")

        # 2. Categorize fields by table and temporal nature
        field_sources = self._categorize_fields(all_fields)

        print(f"\nField sources:")
        for table, fields in field_sources.items():
            if fields:
                print(f"  {table}: {len(fields)} fields")

        # 3. Load maindata (always one row per patient)
        print(f"\nLoading maindata...")
        df_main = self._load_maindata(
            field_sources.get('maindata', []),
            include_demographics=include_demographics
        )
        print(f"  Loaded {len(df_main)} patients")

        # 4. Join diagnosis if needed
        if include_diagnosis and field_sources.get('diagnosis'):
            print(f"\nJoining diagnosis table...")
            df_diag = self._load_diagnosis(field_sources['diagnosis'])
            df_main = df_main.merge(df_diag, on='FACPATID', how='left')
            print(f"  Joined {len(df_diag)} diagnosis records")

        # 5. Join encounter based on strategy
        if field_sources.get('encounter'):
            print(f"\nJoining encounter table (strategy: {visit_strategy})...")
            df_enc = self._load_encounter(
                field_sources['encounter'],
                visit_strategy=visit_strategy
            )
            df_main = df_main.merge(df_enc, on='FACPATID', how='left')

            if visit_strategy == 'all_visits':
                print(f"  Joined {len(df_enc)} encounter records (longitudinal)")
                print(f"  Result: {len(df_main)} total rows (multiple per patient)")
            else:
                print(f"  Joined {len(df_enc)} patients with encounters")
                print(f"  Result: {len(df_main)} patients (one row per patient)")

        # 6. Apply data quality filters
        print(f"\n{'='*70}")
        print(f"APPLYING DATA QUALITY FILTERS")
        print(f"{'='*70}")

        initial_count = len(df_main)
        initial_patients = df_main['FACPATID'].nunique() if 'FACPATID' in df_main.columns else initial_count

        df_main = self._apply_data_quality_filters(
            df_main,
            include_usndr=include_usndr,
            validate_enrollment=validate_enrollment
        )

        final_count = len(df_main)
        final_patients = df_main['FACPATID'].nunique() if 'FACPATID' in df_main.columns else final_count
        excluded_patients = initial_patients - final_patients

        print(f"  Initial: {initial_patients} patients")
        print(f"  Final: {final_patients} patients")
        print(f"  Excluded: {excluded_patients} patients")

        print(f"\n{'='*70}")
        print(f"COHORT BUILD COMPLETE")
        print(f"{'='*70}")
        print(f"Final shape: {df_main.shape}")
        print(f"Columns: {len(df_main.columns)}")

        return df_main

    def _collect_fields(
        self,
        domains: List[str],
        additional_fields: Optional[List[str]] = None
    ) -> List[str]:
        """Collect all field names needed from specified domains."""
        all_fields = set()

        for domain in domains:
            domain_fields = self.disease_config.get_domain_fields(
                domain,
                include_metadata=False
            )
            all_fields.update(domain_fields)

        if additional_fields:
            all_fields.update(additional_fields)

        return list(all_fields)

    def _categorize_fields(self, fields: List[str]) -> Dict[str, List[str]]:
        """
        Categorize fields by source table.

        Returns:
            Dict mapping table name to list of field names
            e.g., {'maindata': ['dob', 'gender'], 'encounter': ['ambulatn', 'ttwr10m']}
        """
        field_sources = {
            'maindata': [],
            'diagnosis': [],
            'encounter': []
        }

        field_source_map = self.crf_metadata.get('field_sources', {})

        for field in fields:
            if field in field_source_map:
                table = field_source_map[field]['table']
                if table in field_sources:
                    field_sources[table].append(field)
            else:
                # Field not in CRF metadata - try to infer or warn
                warnings.warn(
                    f"Field '{field}' not found in CRF metadata. "
                    f"Cannot determine source table. Skipping."
                )

        return field_sources

    def _load_maindata(
        self,
        fields: List[str],
        include_demographics: bool = True
    ) -> pd.DataFrame:
        """
        Load maindata table (always one row per patient).

        Args:
            fields: List of maindata fields to load
            include_demographics: Include FACPATID, dob, gender, race, ethnicity

        Returns:
            DataFrame with one row per patient
        """
        # Core demographic fields
        required_fields = ['FACPATID']

        if include_demographics:
            required_fields.extend(['dob', 'gender', 'race', 'ethnicity'])

        # Combine with requested fields
        all_fields = list(set(required_fields + fields))

        # Load maindata (try both naming conventions)
        maindata_path = self.data_dir / "Demographics_MainData.parquet"
        if not maindata_path.exists():
            maindata_path = self.data_dir / "maindata.parquet"

        if not maindata_path.exists():
            raise FileNotFoundError(
                f"Maindata file not found. Tried:\n"
                f"  - {self.data_dir / 'Demographics_MainData.parquet'}\n"
                f"  - {self.data_dir / 'maindata.parquet'}"
            )

        # Check which fields actually exist in the file
        # Read just the schema, not the data
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(maindata_path)
        available_columns = parquet_file.schema.names
        fields_to_load = [f for f in all_fields if f in available_columns]

        if len(fields_to_load) < len(all_fields):
            missing = set(all_fields) - set(fields_to_load)
            warnings.warn(f"Some maindata fields not found in file: {missing}")

        df = pd.read_parquet(maindata_path, columns=fields_to_load)

        return df

    def _load_diagnosis(self, fields: List[str]) -> pd.DataFrame:
        """
        Load diagnosis table (one row per patient).

        Args:
            fields: List of diagnosis fields to load

        Returns:
            DataFrame with one row per patient
        """
        # Always need FACPATID for join
        all_fields = list(set(['FACPATID'] + fields))

        # Try both naming conventions
        diagnosis_path = self.data_dir / "Diagnosis_MainData.parquet"
        if not diagnosis_path.exists():
            diagnosis_path = self.data_dir / "diagnosis.parquet"

        if not diagnosis_path.exists():
            warnings.warn(
                f"Diagnosis file not found. Tried:\n"
                f"  - {self.data_dir / 'Diagnosis_MainData.parquet'}\n"
                f"  - {self.data_dir / 'diagnosis.parquet'}"
            )
            # Return empty dataframe with FACPATID
            return pd.DataFrame(columns=['FACPATID'])

        # Check which fields exist
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(diagnosis_path)
        available_columns = parquet_file.schema.names
        fields_to_load = [f for f in all_fields if f in available_columns]

        if len(fields_to_load) < len(all_fields):
            missing = set(all_fields) - set(fields_to_load)
            warnings.warn(f"Some diagnosis fields not found in file: {missing}")

        df = pd.read_parquet(diagnosis_path, columns=fields_to_load)

        return df

    def _load_encounter(
        self,
        fields: List[str],
        visit_strategy: str = 'latest_visit'
    ) -> pd.DataFrame:
        """
        Load encounter table with visit selection strategy.

        Args:
            fields: List of encounter fields to load
            visit_strategy: 'latest_visit', 'first_visit', or 'all_visits'

        Returns:
            DataFrame with visit data
            - One row per patient (latest/first)
            - Multiple rows per patient (all)
        """
        # Always need these for processing
        required_fields = ['FACPATID', 'CASE_ID', 'encntdt']
        all_fields = list(set(required_fields + fields))

        # Try both naming conventions
        encounter_path = self.data_dir / "Encounter_MainData.parquet"
        if not encounter_path.exists():
            encounter_path = self.data_dir / "encounter.parquet"

        if not encounter_path.exists():
            warnings.warn(
                f"Encounter file not found. Tried:\n"
                f"  - {self.data_dir / 'Encounter_MainData.parquet'}\n"
                f"  - {self.data_dir / 'encounter.parquet'}"
            )
            return pd.DataFrame(columns=['FACPATID'])

        # Check which fields exist
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(encounter_path)
        available_columns = parquet_file.schema.names
        fields_to_load = [f for f in all_fields if f in available_columns]

        if len(fields_to_load) < len(all_fields):
            missing = set(all_fields) - set(fields_to_load)
            warnings.warn(f"Some encounter fields not found in file: {missing}")

        # Load encounter data
        df = pd.read_parquet(encounter_path, columns=fields_to_load)

        # Apply visit selection strategy
        if visit_strategy == 'latest_visit':
            # Get most recent visit per patient
            df = df.sort_values('encntdt').groupby('FACPATID', as_index=False).last()

        elif visit_strategy == 'first_visit':
            # Get first visit per patient
            df = df.sort_values('encntdt').groupby('FACPATID', as_index=False).first()

        elif visit_strategy == 'all_visits':
            # Keep all visits (multiple rows per patient)
            df = df.sort_values(['FACPATID', 'encntdt'])

        else:
            raise ValueError(
                f"Invalid visit_strategy: '{visit_strategy}'. "
                f"Must be 'latest_visit', 'first_visit', or 'all_visits'"
            )

        return df

    def _apply_data_quality_filters(
        self,
        df: pd.DataFrame,
        include_usndr: bool = False,
        validate_enrollment: bool = True
    ) -> pd.DataFrame:
        """
        Apply data quality and enrollment validation filters.

        Args:
            df: Input dataframe
            include_usndr: Include USNDR patients (default: False)
            validate_enrollment: Validate patient exists in all 3 MainData forms (default: True)

        Returns:
            Filtered dataframe

        Filters Applied:
            1. Validated Enrollment (if validate_enrollment=True):
               - Patient must exist in Demographics_MainData
               - Patient must exist in Diagnosis_MainData
               - Patient must exist in Encounter_MainData
               This ensures complete enrollment across all three core forms.

            2. MOVR Study Data Only (if include_usndr=False):
               - Excludes patients where usndr == 1
               - Retains only MOVR Data Hub study participants
        """
        if 'FACPATID' not in df.columns:
            warnings.warn("Cannot apply data quality filters: FACPATID column not found")
            return df

        initial_count = df['FACPATID'].nunique()

        # Filter 1: Validated Enrollment
        if validate_enrollment:
            print(f"\n  Applying validated enrollment filter...")

            # Get patients from each MainData table
            demographics_path = self.data_dir / "Demographics_MainData.parquet"
            diagnosis_path = self.data_dir / "Diagnosis_MainData.parquet"
            encounter_path = self.data_dir / "Encounter_MainData.parquet"

            # Load FACPATID from each table
            import pyarrow.parquet as pq

            demographics_patients = set()
            if demographics_path.exists():
                pf = pq.ParquetFile(demographics_path)
                demographics_df = pf.read(columns=['FACPATID']).to_pandas()
                demographics_patients = set(demographics_df['FACPATID'].unique())
                print(f"    Demographics: {len(demographics_patients)} patients")

            diagnosis_patients = set()
            if diagnosis_path.exists():
                pf = pq.ParquetFile(diagnosis_path)
                diagnosis_df = pf.read(columns=['FACPATID']).to_pandas()
                diagnosis_patients = set(diagnosis_df['FACPATID'].unique())
                print(f"    Diagnosis: {len(diagnosis_patients)} patients")

            encounter_patients = set()
            if encounter_path.exists():
                pf = pq.ParquetFile(encounter_path)
                encounter_df = pf.read(columns=['FACPATID']).to_pandas()
                encounter_patients = set(encounter_df['FACPATID'].unique())
                print(f"    Encounter: {len(encounter_patients)} patients")

            # Find patients in all three tables (validated enrollment)
            validated_patients = demographics_patients & diagnosis_patients & encounter_patients

            print(f"    Validated (in all 3 forms): {len(validated_patients)} patients")

            # Filter to validated patients only
            df = df[df['FACPATID'].isin(validated_patients)]

            after_validation = df['FACPATID'].nunique()
            excluded = initial_count - after_validation
            print(f"    Excluded {excluded} patients (not in all 3 MainData forms)")

        # Filter 2: MOVR Study Data Only (exclude USNDR)
        if not include_usndr:
            print(f"\n  Applying MOVR study filter (excluding USNDR)...")

            # Need to check if usndr field exists in df
            if 'usndr' in df.columns:
                before_usndr = len(df)
                # Exclude where usndr == 1 (USNDR patients)
                df = df[df['usndr'] != 1]
                after_usndr = len(df)
                excluded_usndr = before_usndr - after_usndr
                print(f"    Excluded {excluded_usndr} USNDR patients (usndr == 1)")
            else:
                # If usndr not in columns, try to load it from Demographics
                demographics_path = self.data_dir / "Demographics_MainData.parquet"
                if demographics_path.exists():
                    import pyarrow.parquet as pq
                    pf = pq.ParquetFile(demographics_path)
                    if 'usndr' in pf.schema.names:
                        usndr_df = pf.read(columns=['FACPATID', 'usndr']).to_pandas()
                        before_usndr = len(df)
                        # Merge and filter
                        df = df.merge(usndr_df, on='FACPATID', how='left')
                        df = df[df['usndr'] != 1]
                        # Drop usndr column if it wasn't originally there
                        df = df.drop(columns=['usndr'])
                        after_usndr = len(df)
                        excluded_usndr = before_usndr - after_usndr
                        print(f"    Excluded {excluded_usndr} USNDR patients (usndr == 1)")
                    else:
                        print(f"    Warning: usndr field not found - cannot filter USNDR patients")
                else:
                    print(f"    Warning: Demographics file not found - cannot filter USNDR patients")

        return df

    def get_field_sources(self, domains: List[str]) -> Dict[str, List[str]]:
        """
        Get field source mapping for specified domains.

        Useful for understanding where fields come from before loading data.

        Args:
            domains: List of clinical domains

        Returns:
            Dict mapping table names to field lists

        Example:
            sources = builder.get_field_sources(['mobility', 'cardiac'])
            print(f"Diagnosis fields: {sources['diagnosis']}")
            print(f"Encounter fields: {sources['encounter']}")
        """
        all_fields = self._collect_fields(domains, additional_fields=None)
        return self._categorize_fields(all_fields)

    def validate_configuration(self) -> Dict:
        """
        Validate disease configuration and CRF metadata.

        Checks:
        - All domain fields are in CRF metadata
        - Required data files exist
        - Field sources are correctly mapped

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # Check all domains
        domains = self.disease_config.get_domain_names()

        for domain in domains:
            domain_fields = self.disease_config.get_domain_fields(domain, include_metadata=False)

            for field in domain_fields:
                if field not in self.crf_metadata.get('field_sources', {}):
                    results['warnings'].append(
                        f"Domain '{domain}': Field '{field}' not in CRF metadata"
                    )

        # Check data files exist
        for table in ['maindata', 'diagnosis', 'encounter']:
            filepath = self.data_dir / f"{table}.parquet"
            if not filepath.exists():
                results['errors'].append(f"Data file missing: {filepath}")
                results['valid'] = False

        results['info']['domains_checked'] = len(domains)
        results['info']['data_directory'] = str(self.data_dir)

        return results

    def __repr__(self):
        """String representation."""
        return (
            f"CohortBuilder(disease='{self.disease_config.disease}', "
            f"data_dir='{self.data_dir}')"
        )
