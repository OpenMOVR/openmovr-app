"""
Caching Utilities

Streamlit caching decorators and cache management.
"""

import streamlit as st
from typing import Dict, Any, List
from api import CohortAPI, StatsAPI


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_snapshot() -> Dict[str, Any]:
    """Get cached database snapshot."""
    return StatsAPI.load_snapshot()


@st.cache_data(ttl=3600)
def get_cached_disease_distribution() -> List[Dict[str, Any]]:
    """Get cached disease distribution."""
    return StatsAPI.get_disease_distribution()


@st.cache_data(ttl=3600)
def get_cached_facility_stats() -> Dict[str, Any]:
    """Get cached facility statistics."""
    return StatsAPI.get_facility_stats()


@st.cache_resource  # Cache indefinitely (resource)
def get_cached_base_cohort(include_usndr: bool = False) -> Dict[str, Any]:
    """
    Get cached base cohort.

    Note: Uses cache_resource for large data objects that should persist.
    """
    return CohortAPI.get_base_cohort(include_usndr=include_usndr)


@st.cache_resource
def get_cached_disease_cohort(disease: str) -> Dict[str, Any]:
    """Get cached disease cohort."""
    return CohortAPI.get_disease_cohort(disease)


def clear_all_caches():
    """Clear all Streamlit caches."""
    st.cache_data.clear()
    st.cache_resource.clear()
