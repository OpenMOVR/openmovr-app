"""
Formatting Utilities

Helper functions for formatting data for display.
"""

from typing import Any, Union


def format_number(value: Union[int, float], decimals: int = 0) -> str:
    """
    Format a number with thousands separators.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if decimals == 0:
        return f"{int(value):,}"
    else:
        return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a percentage value.

    Args:
        value: Percentage value (0-100)
        decimals: Number of decimal places

    Returns:
        Formatted string with % sign
    """
    return f"{value:.{decimals}f}%"


def format_metric_title(key: str) -> str:
    """
    Format a metric key into a nice title.

    Args:
        key: Metric key (e.g., 'total_patients')

    Returns:
        Formatted title (e.g., 'Total Patients')
    """
    return key.replace('_', ' ').title()


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
