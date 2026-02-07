"""
Export Configuration Module

Provides centralized configuration for industry data exports with granular PHI control.
Supports multiple PHI profiles from deidentified (default) to full PHI.

Usage:
    from src.config.export_config import ExportConfig, load_export_profile

    # Load a predefined profile
    config = load_export_profile('deidentified')  # Default - no PHI, dates shifted
    config = load_export_profile('dates_only')    # Real dates, no other PHI
    config = load_export_profile('zip_only')      # Only ZIP codes
    config = load_export_profile('full_phi', approval_code="IRB-2025-001")

    # Custom configuration
    config = ExportConfig(
        include_phi_fields=['state', 'zip'],
        shift_dates=True,
        approval_code="CUSTOM-2025-001"
    )

    # Check what fields will be removed/included
    config.print_summary()
    fields_to_remove = config.get_fields_to_remove()

Available Profiles (see config/industry_export_profiles.yaml):
    - deidentified: No PHI, dates shifted (default)
    - dates_only: Real dates, no other PHI
    - location_only: State/zip/country only
    - zip_only: Only ZIP and state
    - contact_info: Full contact for recontact studies
    - full_phi: All PHI fields
    - linkage: Names + DOB for record linkage
"""

import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Union


# =============================================================================
# PHI FIELD DEFINITIONS
# =============================================================================

# All PHI fields that exist in Demographics_PHI.xlsx
ALL_PHI_FIELDS = [
    # Direct identifiers
    'firstname', 'lastname', 'fname', 'mname', 'sname',
    # Contact information
    'street', 'phnum', 'eaddress',
    # Location
    'city', 'city1', 'state', 'zip', 'country', 'country1',
    # Dates (these are in both PHI and noPHI, but values are real in both)
    'dob', 'dob1', 'dob.P', 'dob1.P',
    # Internal identifiers
    'guid1',
]

# PHI fields by category for granular control
PHI_FIELD_GROUPS = {
    'direct_identifiers': ['firstname', 'lastname', 'fname', 'mname', 'sname'],
    'contact': ['street', 'phnum', 'eaddress'],
    'location': ['city', 'city1', 'state', 'zip', 'country', 'country1'],
    'dates': ['dob', 'dob1', 'dob.P', 'dob1.P'],
    'internal_ids': ['guid1'],
}

# IQVIA product/system fields - always excluded
IQVIA_SYSTEM_FIELDS = [
    'MASTER_PATIENT_ID',
    'FORM_VERSION',
    'FORM_STATUS',
    'FACILITY_NAME',
    'CREATED_DT',
    'MODIFIED_DT',
    'CREATED_BY',
    'UPDATED_BY',
    'UPLOADED_BY',
    'Access Case',
]

# Internal/processing fields - always excluded
INTERNAL_FIELDS = [
    'total_days_shifted',
    'Source_sheet',
]


# =============================================================================
# EXPORT CONFIG CLASS
# =============================================================================

@dataclass
class ExportConfig:
    """
    Configuration for industry data exports with granular PHI control.

    Attributes:
        profile_name: Name of the profile being used (for documentation)
        phi_level: PHI level identifier ('none', 'dates', 'location', 'contact', 'full')

        # PHI Control
        include_phi_fields: List of PHI fields to INCLUDE (empty = no PHI)
        exclude_phi_fields: List of PHI fields to EXCLUDE (in addition to defaults)

        # Date Shifting
        shift_dates: If True, applies seed-based date shifting
        year_shift_range: Max years to shift (default: 1)
        day_shift_range: Max days to shift (default: 7)
        bidirectional_shift: If True, shifts can be +/- (default: True)

        # Approval (required for PHI exports)
        requires_approval: Whether this config requires documented approval
        approval_code: Approval/IRB code (required if requires_approval=True)
        approved_by: Name of approver
        approval_date: Date of approval
        justification: Reason for PHI inclusion

        # Source Data
        use_phi_demographics: If True, load from Demographics_PHI.xlsx
        source_data_path: Path to source Excel files

        # Export Options
        remove_iqvia_fields: Remove IQVIA system fields (default: True)
        remove_all_na_columns: Remove columns with all NA values (default: True)
        export_formats: List of formats to generate (['xlsx', 'csv'])
    """

    # Profile identification
    profile_name: str = "custom"
    phi_level: str = "none"
    description: str = ""

    # PHI Control
    include_phi_fields: List[str] = field(default_factory=list)
    exclude_phi_fields: Union[List[str], str] = field(default_factory=lambda: "all")

    # Date Shifting
    shift_dates: bool = True
    year_shift_range: int = 1
    day_shift_range: int = 7
    bidirectional_shift: bool = True

    # Approval
    requires_approval: bool = False
    approval_code: Optional[str] = None
    approved_by: Optional[str] = None
    approval_date: Optional[str] = None
    justification: Optional[str] = None
    irb_approval_required: bool = False

    # Source Data
    use_phi_demographics: bool = False
    source_data_path: Optional[str] = None

    # Export Options
    remove_iqvia_fields: bool = True
    remove_all_na_columns: bool = True
    export_formats: List[str] = field(default_factory=lambda: ['xlsx', 'csv'])
    include_data_dictionary: bool = True
    include_manifest: bool = True
    include_executive_summary: bool = True
    include_readme: bool = True
    include_seed_table: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate approval requirements
        if self.requires_approval and not self.approval_code:
            raise ValueError(
                f"Profile '{self.profile_name}' requires approval. "
                f"Provide approval_code parameter."
            )

        # Determine if PHI demographics should be used
        if self.include_phi_fields and self.include_phi_fields != []:
            # Check if any included fields are actual PHI (not just dates)
            non_date_phi = set(self.include_phi_fields) - set(PHI_FIELD_GROUPS['dates'])
            if non_date_phi:
                self.use_phi_demographics = True

        # If not shifting dates, seed table is not needed
        if not self.shift_dates:
            self.include_seed_table = False

    def get_fields_to_remove(self) -> Set[str]:
        """
        Get complete set of fields to remove from export.

        Returns:
            Set of field names to remove
        """
        fields = set(INTERNAL_FIELDS)

        if self.remove_iqvia_fields:
            fields.update(IQVIA_SYSTEM_FIELDS)

        # Handle PHI exclusion
        if self.exclude_phi_fields == "all":
            # Exclude all PHI except those explicitly included
            phi_to_remove = set(ALL_PHI_FIELDS) - set(self.include_phi_fields)
            fields.update(phi_to_remove)
        elif isinstance(self.exclude_phi_fields, list):
            # Exclude specific fields
            fields.update(self.exclude_phi_fields)

        return fields

    def get_phi_fields_included(self) -> List[str]:
        """Get list of PHI fields that will be included in export."""
        if self.exclude_phi_fields == "all":
            return list(self.include_phi_fields)
        else:
            # Include all PHI except explicitly excluded
            return [f for f in ALL_PHI_FIELDS if f not in self.exclude_phi_fields]

    def get_date_shift_config(self) -> dict:
        """Get date shifting configuration."""
        if not self.shift_dates:
            return {"enabled": False}
        return {
            "enabled": True,
            "year_shift_range": self.year_shift_range,
            "day_shift_range": self.day_shift_range,
            "bidirectional": self.bidirectional_shift,
            "total_range_days": f"±{self.year_shift_range * 365 + self.day_shift_range}",
        }

    def get_demographics_source(self) -> str:
        """Get which demographics file to use."""
        if self.use_phi_demographics:
            return "Demographics_PHI.xlsx"
        return "Demographics_noPHI.xlsx"

    def to_manifest_dict(self) -> dict:
        """Convert configuration to dictionary for export manifest."""
        phi_included = self.get_phi_fields_included()
        return {
            "profile": {
                "name": self.profile_name,
                "phi_level": self.phi_level,
                "description": self.description,
            },
            "phi_configuration": {
                "phi_fields_included": phi_included,
                "phi_field_count": len(phi_included),
                "demographics_source": self.get_demographics_source(),
            },
            "approval": {
                "requires_approval": self.requires_approval,
                "approval_code": self.approval_code,
                "approved_by": self.approved_by,
                "approval_date": self.approval_date,
                "justification": self.justification,
                "irb_approval_required": self.irb_approval_required,
            },
            "date_shifting": self.get_date_shift_config(),
            "field_removal": {
                "remove_iqvia_fields": self.remove_iqvia_fields,
                "remove_all_na_columns": self.remove_all_na_columns,
                "fields_removed_count": len(self.get_fields_to_remove()),
            },
            "export_options": {
                "formats": self.export_formats,
                "include_data_dictionary": self.include_data_dictionary,
                "include_manifest": self.include_manifest,
                "include_executive_summary": self.include_executive_summary,
                "include_readme": self.include_readme,
                "include_seed_table": self.include_seed_table,
            }
        }

    def print_summary(self):
        """Print configuration summary to console."""
        print("\n" + "=" * 70)
        print("EXPORT CONFIGURATION SUMMARY")
        print("=" * 70)

        print(f"\n📋 Profile: {self.profile_name}")
        print(f"   PHI Level: {self.phi_level}")
        if self.description:
            print(f"   Description: {self.description}")

        # PHI Status
        phi_included = self.get_phi_fields_included()
        if phi_included:
            print(f"\n⚠️  PHI INCLUDED ({len(phi_included)} fields)")
            for f in phi_included[:8]:
                print(f"     - {f}")
            if len(phi_included) > 8:
                print(f"     ... and {len(phi_included) - 8} more")
        else:
            print(f"\n✓ NO PHI INCLUDED (fully de-identified)")

        # Date Shifting
        if self.shift_dates:
            print(f"\n📅 Date Shifting: ENABLED")
            print(f"     Year range: ±{self.year_shift_range}")
            print(f"     Day range: ±{self.day_shift_range}")
            total_range = self.year_shift_range * 365 + self.day_shift_range
            print(f"     Total range: ±{total_range} days")
        else:
            print(f"\n📅 Date Shifting: DISABLED (real dates preserved)")

        # Approval
        if self.requires_approval:
            print(f"\n🔐 APPROVAL REQUIRED")
            if self.approval_code:
                print(f"     Code: {self.approval_code}")
            if self.approved_by:
                print(f"     Approved by: {self.approved_by}")
            if self.justification:
                print(f"     Justification: {self.justification}")
        else:
            print(f"\n✓ No approval required (de-identified export)")

        # Demographics Source
        print(f"\n📁 Demographics Source: {self.get_demographics_source()}")

        # Fields to remove
        fields_removed = len(self.get_fields_to_remove())
        print(f"\n🗑️  Fields to remove: {fields_removed}")

        print("\n" + "=" * 70)


# =============================================================================
# PROFILE LOADING
# =============================================================================

def load_export_profile(
    profile_name: str,
    approval_code: Optional[str] = None,
    approved_by: Optional[str] = None,
    justification: Optional[str] = None,
    config_path: Optional[str] = None,
    **overrides
) -> ExportConfig:
    """
    Load an export profile from the YAML configuration.

    Args:
        profile_name: Name of profile to load (e.g., 'deidentified', 'full_phi')
        approval_code: Approval code (required for PHI profiles)
        approved_by: Name of approver (recommended for PHI profiles)
        justification: Reason for PHI inclusion (required for some profiles)
        config_path: Path to config YAML (default: config/industry_export_profiles.yaml)
        **overrides: Additional parameters to override profile defaults

    Returns:
        ExportConfig instance configured according to the profile

    Example:
        # Standard de-identified export
        config = load_export_profile('deidentified')

        # Full PHI export with approval
        config = load_export_profile(
            'full_phi',
            approval_code='IRB-2025-001',
            approved_by='Dr. Smith',
            justification='Patient linkage study'
        )

        # Custom override
        config = load_export_profile(
            'dates_only',
            approval_code='APPROVED-001',
            shift_dates=True  # Override to also shift dates
        )
    """
    # Find config file
    if config_path is None:
        # Try relative to this file, then project root
        module_dir = Path(__file__).parent
        config_path = module_dir.parent.parent / "config" / "industry_export_profiles.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Get profile
    profiles = config_data.get('profiles', {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise ValueError(
            f"Unknown profile '{profile_name}'. "
            f"Available profiles: {available}"
        )

    profile = profiles[profile_name]

    # Build configuration
    include_phi = profile.get('include_phi_fields', [])
    if include_phi == "all":
        include_phi = ALL_PHI_FIELDS.copy()

    exclude_phi = profile.get('exclude_phi_fields', [])

    # Handle date shift config
    date_shift_config = profile.get('date_shift_config', {})

    config = ExportConfig(
        profile_name=profile_name,
        phi_level=profile.get('phi_level', 'none'),
        description=profile.get('description', ''),

        include_phi_fields=include_phi,
        exclude_phi_fields=exclude_phi,

        shift_dates=profile.get('shift_dates', True),
        year_shift_range=date_shift_config.get('year_shift_range', 1),
        day_shift_range=date_shift_config.get('day_shift_range', 7),
        bidirectional_shift=date_shift_config.get('bidirectional', True),

        requires_approval=profile.get('requires_approval', False),
        approval_code=approval_code,
        approved_by=approved_by,
        approval_date=datetime.now().strftime('%Y-%m-%d') if approval_code else None,
        justification=justification,
        irb_approval_required=profile.get('irb_approval_required', False),

        source_data_path=config_data.get('source_data', {}).get('base_path'),
    )

    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def list_profiles(config_path: Optional[str] = None) -> dict:
    """
    List all available export profiles.

    Returns:
        Dictionary of profile names and descriptions
    """
    if config_path is None:
        module_dir = Path(__file__).parent
        config_path = module_dir.parent.parent / "config" / "industry_export_profiles.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return {}

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    profiles = config_data.get('profiles', {})
    return {
        name: {
            'description': p.get('description', ''),
            'phi_level': p.get('phi_level', 'none'),
            'shift_dates': p.get('shift_dates', True),
            'requires_approval': p.get('requires_approval', False),
        }
        for name, p in profiles.items()
    }


def print_available_profiles():
    """Print all available profiles with descriptions."""
    profiles = list_profiles()

    print("\n" + "=" * 70)
    print("AVAILABLE EXPORT PROFILES")
    print("=" * 70)

    for name, info in profiles.items():
        approval = "✓ Requires approval" if info['requires_approval'] else ""
        dates = "Dates shifted" if info['shift_dates'] else "Real dates"
        print(f"\n  {name}")
        print(f"    {info['description']}")
        print(f"    PHI Level: {info['phi_level']} | {dates} {approval}")

    print("\n" + "=" * 70)
    print("Usage: config = load_export_profile('profile_name')")
    print("=" * 70)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def deidentified_export(**kwargs) -> ExportConfig:
    """Create a standard de-identified export configuration."""
    return load_export_profile('deidentified', **kwargs)


def full_phi_export(
    approval_code: str,
    approved_by: str,
    justification: str,
    **kwargs
) -> ExportConfig:
    """Create a full PHI export configuration (requires approval)."""
    return load_export_profile(
        'full_phi',
        approval_code=approval_code,
        approved_by=approved_by,
        justification=justification,
        **kwargs
    )


def dates_only_export(
    approval_code: str,
    approved_by: Optional[str] = None,
    justification: Optional[str] = None,
    **kwargs
) -> ExportConfig:
    """Create export with real dates but no other PHI."""
    return load_export_profile(
        'dates_only',
        approval_code=approval_code,
        approved_by=approved_by,
        justification=justification or "Analysis requires exact dates",
        **kwargs
    )
