"""
Filter Components

Reusable filter UI components for Streamlit.
"""

import streamlit as st
from typing import List, Optional, Any, Dict
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd


def disease_selector(diseases: List[str],
                    key: str = "disease_select",
                    label: str = "Select Disease",
                    default: Optional[str] = None) -> str:
    """
    Create a disease selection dropdown.

    Args:
        diseases: List of available diseases
        key: Streamlit widget key
        label: Label for the selector
        default: Default selection

    Returns:
        Selected disease
    """
    default_index = 0
    if default and default in diseases:
        default_index = diseases.index(default)

    return st.selectbox(
        label,
        options=diseases,
        index=default_index,
        key=key
    )


def facility_multiselect(facilities: List[Dict[str, Any]],
                        key: str = "facility_select",
                        label: str = "Select Facilities") -> List[str]:
    """
    Create a facility multi-selection widget.

    Args:
        facilities: List of facility dicts with FACILITY_DISPLAY_ID and FACILITY_NAME
        key: Streamlit widget key
        label: Label for the selector

    Returns:
        List of selected facility IDs
    """
    facility_options = {
        f"{f['FACILITY_DISPLAY_ID']}: {f['FACILITY_NAME']}": f['FACILITY_DISPLAY_ID']
        for f in facilities
    }

    selected_names = st.multiselect(
        label,
        options=list(facility_options.keys()),
        key=key
    )

    return [facility_options[name] for name in selected_names]


def patient_count_slider(min_val: int = 0,
                        max_val: int = 1000,
                        default_range: Optional[tuple] = None,
                        key: str = "patient_count_slider",
                        label: str = "Participant Count Range") -> tuple:
    """
    Create a patient count range slider.

    Args:
        min_val: Minimum value
        max_val: Maximum value
        default_range: Default (min, max) tuple
        key: Streamlit widget key
        label: Label for the slider

    Returns:
        Tuple of (min, max) selected values
    """
    if default_range is None:
        default_range = (min_val, max_val)

    return st.slider(
        label,
        min_value=min_val,
        max_value=max_val,
        value=default_range,
        key=key
    )


def include_usndr_toggle(key: str = "include_usndr",
                        label: str = "Include USNDR Legacy Participants",
                        default: bool = False) -> bool:
    """
    Create a toggle for including USNDR patients.

    Args:
        key: Streamlit widget key
        label: Label for the toggle
        default: Default state

    Returns:
        Boolean indicating if USNDR should be included
    """
    return st.checkbox(label, value=default, key=key)


def chart_type_selector(chart_types: List[str],
                       key: str = "chart_type",
                       label: str = "Chart Type",
                       default: str = "bar") -> str:
    """
    Create a chart type selector.

    Args:
        chart_types: List of available chart types
        key: Streamlit widget key
        label: Label for the selector
        default: Default chart type

    Returns:
        Selected chart type
    """
    default_index = 0
    if default in chart_types:
        default_index = chart_types.index(default)

    return st.radio(
        label,
        options=chart_types,
        index=default_index,
        horizontal=True,
        key=key
    )


# ---------------------------------------------------------------------------
# Cascading Disease Filter Renderer
# ---------------------------------------------------------------------------

# Resolve config path relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_FILTER_CONFIG = _PROJECT_ROOT / "config" / "disease_filters.yaml"

# Maximum unique values before switching a multiselect to text search
_MAX_MULTISELECT_OPTIONS = 50


def _load_filter_config(config_path: Path = _DEFAULT_FILTER_CONFIG) -> Dict[str, Any]:
    """Load disease filter configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_filter_config() -> Dict[str, Any]:
    """Public: load the disease filter config (re-reads YAML each call)."""
    return _load_filter_config()


def _get_dataframe_for_source(cohort: Dict[str, Any], source_table: str) -> pd.DataFrame:
    """Map a source_table name from the config to the cohort dict key."""
    table_map = {
        "demographics": "demographics",
        "diagnosis": "diagnosis",
        "encounters": "encounters",
        "medications": "medications",
    }
    key = table_map.get(source_table, source_table)
    df = cohort.get(key)
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def _compute_age_column(df: pd.DataFrame, dob_field: str = "dob") -> Optional[pd.Series]:
    """Compute age at enrollment (enroldt − dob). Falls back to current age if enroldt missing."""
    if dob_field not in df.columns:
        return None
    dob = pd.to_datetime(df[dob_field], errors="coerce")
    if "enroldt" in df.columns:
        enrol = pd.to_datetime(df["enroldt"], errors="coerce")
        age = (enrol - dob).dt.days / 365.25
    else:
        age = (pd.Timestamp.now() - dob).dt.days / 365.25
    age = age[age >= 0]
    return age.dropna()


def _render_single_widget(
    filter_def: Dict[str, Any],
    cohort: Dict[str, Any],
    disease: str,
) -> Optional[Dict[str, Any]]:
    """
    Render one filter widget. Returns a dict describing the active filter,
    or None if the user has not narrowed anything.
    """
    field = filter_def["field"]
    widget_type = filter_def["widget"]
    label = filter_def["label"]
    source_table = filter_def["source_table"]
    # Include per_patient suffix to avoid key collisions for same field with different dedup
    _suffix = f"_{filter_def['per_patient']}" if "per_patient" in filter_def else ""
    key = f"filter_{disease}_{field}{_suffix}"

    df = _get_dataframe_for_source(cohort, source_table)

    # --- Computed age special case ---
    if filter_def.get("compute_from") == "dob":
        age_series = _compute_age_column(df, "dob")
        if age_series is None or age_series.empty:
            st.caption(f"{label}: no data available")
            return None

        min_age = int(age_series.min())
        max_age = int(age_series.max()) + 1
        if min_age == max_age:
            max_age = min_age + 1

        selected = st.slider(
            label,
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
            step=filter_def.get("step", 1),
            key=key,
        )
        if selected != (min_age, max_age):
            return {
                "field": field,
                "value": selected,
                "source_table": source_table,
                "widget_type": "age_range",
                "compute_from": "dob",
            }
        return None

    # --- Check field exists ---
    if field not in df.columns:
        return None  # silently skip

    # --- Multiselect ---
    if widget_type == "multiselect":
        raw_values = df[field].dropna()
        # Convert to string for display consistency
        raw_values = raw_values.astype(str)
        raw_values = raw_values[raw_values.str.strip() != ""]
        unique_vals = sorted(raw_values.unique().tolist())

        if not unique_vals:
            return None

        if len(unique_vals) > _MAX_MULTISELECT_OPTIONS:
            # Too many options — use text search
            search = st.text_input(f"{label} (search)", key=key)
            if search and search.strip():
                return {
                    "field": field,
                    "value": search.strip(),
                    "source_table": source_table,
                    "widget_type": "text_search",
                }
            return None

        selected = st.multiselect(label, options=unique_vals, key=key)
        if selected:
            return {
                "field": field,
                "value": selected,
                "source_table": source_table,
                "widget_type": "multiselect",
            }
        return None

    # --- Range slider ---
    if widget_type == "range_slider":
        numeric_col = pd.to_numeric(df[field], errors="coerce").dropna()
        if numeric_col.empty:
            st.caption(f"{label}: no numeric data")
            return None

        min_val = float(numeric_col.min())
        max_val = float(numeric_col.max())
        if min_val == max_val:
            max_val = min_val + 1.0

        step = float(filter_def.get("step", 1))
        selected = st.slider(
            label,
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=step,
            key=key,
        )
        if selected != (min_val, max_val):
            return {
                "field": field,
                "value": selected,
                "source_table": source_table,
                "widget_type": "range_slider",
            }
        return None

    # --- Date range ---
    if widget_type == "date_range":
        date_col = pd.to_datetime(df[field], errors="coerce").dropna()
        if date_col.empty:
            st.caption(f"{label}: no date data")
            return None

        min_date = date_col.min().date()
        max_date = date_col.max().date()

        selected = st.date_input(
            label,
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key=key,
        )
        # date_input returns a tuple when given a default tuple
        if isinstance(selected, (list, tuple)) and len(selected) == 2:
            if selected[0] != min_date or selected[1] != max_date:
                return {
                    "field": field,
                    "value": (
                        pd.Timestamp(selected[0]),
                        pd.Timestamp(selected[1]),
                    ),
                    "source_table": source_table,
                    "widget_type": "date_range",
                }
        return None

    # --- Checkbox ---
    if widget_type == "checkbox":
        checked = st.checkbox(label, key=key)
        if checked:
            return {
                "field": field,
                "value": True,
                "source_table": source_table,
                "widget_type": "checkbox",
            }
        return None

    return None


class DiseaseFilterRenderer:
    """
    Renders cascading sidebar filters based on the selected disease.

    Usage::

        renderer = DiseaseFilterRenderer()
        active_filters = renderer.render_filters(disease="DMD", cohort=disease_cohort)
    """

    def __init__(self, config_path: Path = _DEFAULT_FILTER_CONFIG):
        self.config = _load_filter_config(config_path)

    def render_filters(
        self, disease: str, cohort: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Render all applicable filters in the sidebar for *disease*.

        Returns a dict keyed by field name where each value is a dict with
        keys: field, value, source_table, widget_type.  Only fields where
        the user has actively narrowed a selection are included.
        """
        active_filters: Dict[str, Dict[str, Any]] = {}

        # --- Universal demographics filters ---
        universal = self.config.get("universal_filters", {})
        demo_filters = universal.get("demographics", [])
        if demo_filters:
            with st.expander("Demographics Filters", expanded=True):
                for fdef in demo_filters:
                    result = _render_single_widget(fdef, cohort, disease)
                    if result is not None:
                        active_filters[result["field"]] = result

        # --- Disease-specific filters ---
        disease_cfg = self.config.get("disease_filters", {}).get(disease, {})

        diag_filters = disease_cfg.get("diagnosis", [])
        if diag_filters:
            with st.expander(f"{disease} Diagnosis Filters", expanded=True):
                for fdef in diag_filters:
                    result = _render_single_widget(fdef, cohort, disease)
                    if result is not None:
                        active_filters[result["field"]] = result

        clinical_filters = disease_cfg.get("clinical", [])
        if clinical_filters:
            with st.expander(f"{disease} Clinical Filters", expanded=False):
                for fdef in clinical_filters:
                    result = _render_single_widget(fdef, cohort, disease)
                    if result is not None:
                        active_filters[result["field"]] = result

        # --- Reset button + summary ---
        st.markdown("---")
        if st.button("Reset All Filters", key=f"reset_filters_{disease}"):
            # Clear all filter keys for this disease from session state
            keys_to_clear = [
                k for k in st.session_state if k.startswith(f"filter_{disease}_")
            ]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

        if active_filters:
            st.caption(f"Active: {len(active_filters)} filter(s)")
        else:
            st.caption("No filters applied")

        return active_filters
