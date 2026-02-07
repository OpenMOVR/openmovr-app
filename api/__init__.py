"""
API layer for webapp.

This module provides a facade over the core analytics library,
making it easy to extract the webapp to a separate repository.

In snapshot mode (no parquet files), CohortAPI and ReportsAPI
may not be functional — use StatsAPI and LGMDAPI instead.
"""

from .stats import StatsAPI
from .data_dictionary import DataDictionaryAPI

try:
    from .cohorts import CohortAPI
except Exception:
    CohortAPI = None

try:
    from .reports import ReportsAPI
except Exception:
    ReportsAPI = None

try:
    from .lgmd import LGMDAPI
except Exception:
    LGMDAPI = None

__all__ = ['CohortAPI', 'StatsAPI', 'ReportsAPI', 'DataDictionaryAPI', 'LGMDAPI']
