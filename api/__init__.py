"""
API layer for webapp.

This module provides a facade over the core analytics library,
making it easy to extract the webapp to a separate repository.
"""

from .cohorts import CohortAPI
from .stats import StatsAPI
from .reports import ReportsAPI
from .data_dictionary import DataDictionaryAPI
from .lgmd import LGMDAPI

__all__ = ['CohortAPI', 'StatsAPI', 'ReportsAPI', 'DataDictionaryAPI', 'LGMDAPI']
