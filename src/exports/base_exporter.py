"""
Base Exporter Class

Abstract base class for all MOVR data exporters.
Provides common functionality for export creation, validation, and packaging.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import json
import os


class BaseExporter(ABC):
    """
    Abstract base class for MOVR data exporters.

    All exporters must implement:
    - validate(): Run exporter-specific validation
    - _prepare_data(): Prepare data for export
    """

    def __init__(self, export_name: str, recipient: str, purpose: str):
        """
        Initialize base exporter.

        Args:
            export_name: Unique identifier for this export (e.g., "2025-11-20_pfizer_dmd_v1")
            recipient: Name of receiving organization
            purpose: Brief description of export purpose
        """
        self.export_id = export_name
        self.export_name = export_name
        self.recipient = recipient
        self.purpose = purpose
        self.creation_date = datetime.now().isoformat()

        # Data storage
        self.datasets = {}  # Dict[str, pd.DataFrame]
        self.data_dictionary = None
        self.manifest = {}
        self.validation_results = {}

        # Configuration
        self.output_formats = ["csv"]  # Default
        self.output_dir = None

    def add_dataset(self, data: pd.DataFrame, name: str) -> None:
        """
        Add a DataFrame to the export package.

        Args:
            data: DataFrame to export
            name: Descriptive name for this dataset
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(data)}")

        if data.empty:
            raise ValueError(f"Dataset '{name}' is empty")

        self.datasets[name] = data.copy()
        print(f"Added dataset '{name}' with {len(data)} records and {len(data.columns)} columns")

    def set_output_formats(self, formats: List[str]) -> None:
        """
        Set output file formats.

        Args:
            formats: List of formats ("csv", "excel", "parquet", "sas")
        """
        valid_formats = ["csv", "excel", "parquet", "sas"]
        for fmt in formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid format '{fmt}'. Must be one of {valid_formats}")

        self.output_formats = formats

    def generate_data_dictionary(self, descriptions: Optional[Dict[str, str]] = None,
                                units: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Auto-generate data dictionary from datasets.

        Args:
            descriptions: Optional manual descriptions {column_name: description}
            units: Optional units {column_name: unit}

        Returns:
            DataFrame with data dictionary
        """
        from .metadata import MetadataGenerator

        if not self.datasets:
            raise ValueError("No datasets added. Add data before generating dictionary.")

        # Use first dataset for dictionary generation
        primary_dataset = list(self.datasets.values())[0]

        generator = MetadataGenerator()
        self.data_dictionary = generator.generate_data_dictionary(
            data=primary_dataset,
            descriptions=descriptions or {},
            units=units or {}
        )

        print(f"Generated data dictionary with {len(self.data_dictionary)} variables")
        return self.data_dictionary

    def generate_manifest(self, **kwargs) -> Dict:
        """
        Generate export manifest with metadata.

        Args:
            **kwargs: Additional metadata for manifest

        Returns:
            Dictionary with manifest data
        """
        from .metadata import MetadataGenerator

        if not self.datasets:
            raise ValueError("No datasets added. Add data before generating manifest.")

        generator = MetadataGenerator()

        # Collect file information
        files_info = []
        for name, data in self.datasets.items():
            files_info.append({
                "dataset_name": name,
                "rows": len(data),
                "columns": len(data.columns),
                "estimated_size_mb": round(data.memory_usage(deep=True).sum() / (1024**2), 2)
            })

        self.manifest = generator.generate_manifest(
            export_id=self.export_id,
            recipient=self.recipient,
            purpose=self.purpose,
            data_version=kwargs.get("data_version", "unknown"),
            cohort_definition=kwargs.get("cohort_definition", {}),
            cohort_statistics=kwargs.get("cohort_statistics", {}),
            files=files_info
        )

        # Add any additional kwargs to manifest
        for key, value in kwargs.items():
            if key not in ["data_version", "cohort_definition", "cohort_statistics"]:
                self.manifest[key] = value

        print(f"Generated export manifest for {self.export_id}")
        return self.manifest

    @abstractmethod
    def validate(self) -> bool:
        """
        Run validation checks on export.

        Must be implemented by subclasses.

        Returns:
            True if all validations pass, False otherwise
        """
        pass

    def get_validation_errors(self) -> List[str]:
        """
        Get list of validation errors from last validation run.

        Returns:
            List of error messages
        """
        if not self.validation_results:
            return ["No validation has been run"]

        errors = []
        for check, result in self.validation_results.items():
            if result.get("status") == "failed":
                errors.append(f"{check}: {result.get('message', 'Failed')}")

        return errors if errors else ["No errors"]

    def save_all(self, output_dir: str) -> None:
        """
        Save complete export package to directory.

        Creates:
        - data/ - Dataset files in specified formats
        - documentation/ - Data dictionary, manifest, validation report
        - README.txt - Export summary

        Args:
            output_dir: Directory to save export package
        """
        self.output_dir = Path(output_dir)

        # Create directory structure
        data_dir = self.output_dir / "data"
        docs_dir = self.output_dir / "documentation"
        data_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving export package to: {self.output_dir}")

        # Save datasets
        self._save_datasets(data_dir)

        # Save documentation
        self._save_documentation(docs_dir)

        # Save README
        self._save_readme()

        print(f"\nExport package complete: {self.export_id}")
        print(f"Location: {self.output_dir.absolute()}")

    def _save_datasets(self, data_dir: Path) -> None:
        """Save datasets in specified formats."""
        for dataset_name, data in self.datasets.items():
            base_filename = f"{self.export_name}_{dataset_name}"

            for fmt in self.output_formats:
                if fmt == "csv":
                    filepath = data_dir / f"{base_filename}.csv"
                    data.to_csv(filepath, index=False)
                    print(f"  Saved: {filepath.name}")

                elif fmt == "excel":
                    filepath = data_dir / f"{base_filename}.xlsx"
                    data.to_excel(filepath, index=False, engine='openpyxl')
                    print(f"  Saved: {filepath.name}")

                elif fmt == "parquet":
                    filepath = data_dir / f"{base_filename}.parquet"
                    data.to_parquet(filepath, index=False, engine='pyarrow')
                    print(f"  Saved: {filepath.name}")

                elif fmt == "sas":
                    # SAS XPT format
                    filepath = data_dir / f"{base_filename}.xpt"
                    # Note: Requires additional library for full SAS support
                    print(f"  SAS format requires additional setup: {filepath.name}")

    def _save_documentation(self, docs_dir: Path) -> None:
        """Save documentation files."""
        # Save data dictionary
        if self.data_dictionary is not None:
            dict_path = docs_dir / f"{self.export_name}_data_dictionary.xlsx"
            self.data_dictionary.to_excel(dict_path, index=False, engine='openpyxl')
            print(f"  Saved: {dict_path.name}")

        # Save manifest
        if self.manifest:
            manifest_path = docs_dir / f"{self.export_name}_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(self.manifest, f, indent=2)
            print(f"  Saved: {manifest_path.name}")

        # Save validation results
        if self.validation_results:
            validation_path = docs_dir / f"{self.export_name}_validation.json"
            with open(validation_path, 'w') as f:
                # Convert numpy/pandas types to native Python types for JSON serialization
                import json as json_module
                json_module.dump(self.validation_results, f, indent=2, default=str)
            print(f"  Saved: {validation_path.name}")

    def _save_readme(self) -> None:
        """Generate and save README.txt."""
        from .metadata import MetadataGenerator

        generator = MetadataGenerator()
        readme_content = generator.generate_readme(self.manifest, self.datasets)

        readme_path = self.output_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"  Saved: README.txt")

    def log_to_audit_trail(self, audit_log_path: str) -> None:
        """
        Append export record to audit trail CSV.

        Args:
            audit_log_path: Path to audit log CSV file
        """
        audit_entry = {
            "export_id": self.export_id,
            "timestamp": self.creation_date,
            "recipient": self.recipient,
            "purpose": self.purpose,
            "datasets": ", ".join(self.datasets.keys()),
            "total_records": sum(len(df) for df in self.datasets.values()),
            "output_dir": str(self.output_dir) if self.output_dir else "Not saved",
            "validated": len(self.validation_results) > 0
        }

        # Create or append to audit log
        audit_df = pd.DataFrame([audit_entry])

        if os.path.exists(audit_log_path):
            existing = pd.read_csv(audit_log_path)
            audit_df = pd.concat([existing, audit_df], ignore_index=True)

        audit_df.to_csv(audit_log_path, index=False)
        print(f"\nLogged export to audit trail: {audit_log_path}")
