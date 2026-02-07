"""
Export table configuration management.

This module provides easy access to predefined sets of tables for industry data cuts.
Table sets are defined in config/export_tables.yaml.

Usage:
    from src.config.export_tables import get_export_tables, list_table_sets

    # Get a predefined table set
    tables = get_export_tables('comprehensive')

    # Get disease-specific tables
    sma_tables = get_export_tables('sma_specific')

    # Get custom combination
    tables = get_export_tables(['standard', 'functional_assessments'])

    # List all available table sets
    available = list_table_sets()
"""

import yaml
from pathlib import Path
from typing import List, Union, Dict

# Path to export tables config
CONFIG_DIR = Path(__file__).parent.parent.parent / 'config'
EXPORT_TABLES_CONFIG = CONFIG_DIR / 'export_tables.yaml'


def load_export_config() -> Dict:
    """Load the export tables configuration from YAML."""
    if not EXPORT_TABLES_CONFIG.exists():
        raise FileNotFoundError(
            f"Export tables configuration not found at {EXPORT_TABLES_CONFIG}"
        )

    with open(EXPORT_TABLES_CONFIG, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_export_tables(table_set: Union[str, List[str]] = 'comprehensive') -> List[str]:
    """
    Get a list of tables for export based on predefined sets.

    Args:
        table_set: Name of table set or list of table set names.
                  Options: 'comprehensive', 'standard', 'minimal',
                          'sma_specific', 'dmd_specific', 'lgmd_specific', 'als_specific',
                          'functional_assessments', 'medications_only', 'genetic_data', 'trial_history'

    Returns:
        List of table names to export (duplicates removed, order preserved)

    Examples:
        >>> # Get comprehensive table set
        >>> tables = get_export_tables('comprehensive')

        >>> # Get SMA-specific tables
        >>> tables = get_export_tables('sma_specific')

        >>> # Combine multiple sets (e.g., standard + functional assessments)
        >>> tables = get_export_tables(['standard', 'functional_assessments'])

        >>> # Use in your export script
        >>> for table_name in get_export_tables('comprehensive'):
        ...     df = loader.load_table(table_name)
    """
    config = load_export_config()

    # Handle single table set name
    if isinstance(table_set, str):
        table_set = [table_set]

    # Collect tables from all requested sets
    all_tables = []
    for set_name in table_set:
        if set_name not in config:
            available = list_table_sets()
            raise ValueError(
                f"Unknown table set: '{set_name}'. "
                f"Available sets: {', '.join(available)}"
            )

        set_config = config[set_name]

        # Skip special sections
        if set_name in ['table_descriptions']:
            continue

        if 'tables' not in set_config:
            raise ValueError(
                f"Table set '{set_name}' does not have a 'tables' key. "
                f"It may be a special section (like 'table_descriptions')."
            )

        all_tables.extend(set_config['tables'])

    # Remove duplicates while preserving order
    seen = set()
    unique_tables = []
    for table in all_tables:
        if table not in seen:
            seen.add(table)
            unique_tables.append(table)

    return unique_tables


def list_table_sets() -> List[str]:
    """
    List all available table set names.

    Returns:
        List of available table set names

    Example:
        >>> available_sets = list_table_sets()
        >>> print("Available table sets:", ", ".join(available_sets))
    """
    config = load_export_config()

    # Filter out special sections
    table_sets = [
        key for key in config.keys()
        if key not in ['table_descriptions'] and isinstance(config[key], dict)
    ]

    return sorted(table_sets)


def get_table_set_description(table_set: str) -> str:
    """
    Get the description of a table set.

    Args:
        table_set: Name of the table set

    Returns:
        Description string

    Example:
        >>> desc = get_table_set_description('comprehensive')
        >>> print(desc)
    """
    config = load_export_config()

    if table_set not in config:
        raise ValueError(f"Unknown table set: '{table_set}'")

    return config[table_set].get('description', 'No description available')


def get_table_description(table_name: str) -> str:
    """
    Get the description of a specific table.

    Args:
        table_name: Name of the table

    Returns:
        Description string or 'No description available'

    Example:
        >>> desc = get_table_description('Demographics_MainData')
        >>> print(desc)
    """
    config = load_export_config()

    descriptions = config.get('table_descriptions', {})
    return descriptions.get(table_name, 'No description available')


def print_table_set_info(table_set: str):
    """
    Print information about a table set including description and table list.

    Args:
        table_set: Name of the table set

    Example:
        >>> print_table_set_info('sma_specific')
    """
    description = get_table_set_description(table_set)
    tables = get_export_tables(table_set)

    print(f"\nTable Set: {table_set}")
    print(f"Description: {description}")
    print(f"Tables ({len(tables)}):")
    for i, table in enumerate(tables, 1):
        desc = get_table_description(table)
        print(f"  {i:2d}. {table}")
        print(f"      {desc}")


def print_all_table_sets():
    """
    Print information about all available table sets.

    Example:
        >>> print_all_table_sets()
    """
    sets = list_table_sets()

    print("="*80)
    print("AVAILABLE TABLE SETS")
    print("="*80)

    for set_name in sets:
        description = get_table_set_description(set_name)
        tables = get_export_tables(set_name)
        print(f"\n{set_name}")
        print(f"  Description: {description}")
        print(f"  Tables: {len(tables)}")


# Convenience exports for backward compatibility
EXPORT_TABLES_COMPREHENSIVE = None  # Lazy loaded
EXPORT_TABLES_STANDARD = None
EXPORT_TABLES_MINIMAL = None


def _get_comprehensive():
    """Lazy load comprehensive table set."""
    global EXPORT_TABLES_COMPREHENSIVE
    if EXPORT_TABLES_COMPREHENSIVE is None:
        EXPORT_TABLES_COMPREHENSIVE = get_export_tables('comprehensive')
    return EXPORT_TABLES_COMPREHENSIVE


def _get_standard():
    """Lazy load standard table set."""
    global EXPORT_TABLES_STANDARD
    if EXPORT_TABLES_STANDARD is None:
        EXPORT_TABLES_STANDARD = get_export_tables('standard')
    return EXPORT_TABLES_STANDARD


def _get_minimal():
    """Lazy load minimal table set."""
    global EXPORT_TABLES_MINIMAL
    if EXPORT_TABLES_MINIMAL is None:
        EXPORT_TABLES_MINIMAL = get_export_tables('minimal')
    return EXPORT_TABLES_MINIMAL


# Make these available as module-level constants (lazy loaded)
def __getattr__(name):
    """Lazy load table set constants."""
    if name == 'EXPORT_TABLES_COMPREHENSIVE':
        return _get_comprehensive()
    elif name == 'EXPORT_TABLES_STANDARD':
        return _get_standard()
    elif name == 'EXPORT_TABLES_MINIMAL':
        return _get_minimal()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


if __name__ == '__main__':
    # Demo usage
    print_all_table_sets()
