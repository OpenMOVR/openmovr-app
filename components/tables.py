"""
Table Components

Reusable table display components for Streamlit.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any, Optional


def static_table(df: pd.DataFrame) -> None:
    """Render a DataFrame as a static HTML table with no index column."""
    html = df.to_html(index=False, classes="clean-table", border=0)
    st.markdown(html, unsafe_allow_html=True)


def display_disease_table(disease_data: List[Dict[str, Any]],
                         show_columns: Optional[List[str]] = None) -> None:
    """
    Display disease distribution table.

    Args:
        disease_data: List of disease dicts
        show_columns: Optional list of columns to show
    """
    df = pd.DataFrame(disease_data)

    if show_columns:
        df = df[show_columns]

    # Format the display
    if 'percentage' in df.columns:
        df['percentage'] = df['percentage'].apply(lambda x: f"{x:.1f}%")

    if 'patient_count' in df.columns:
        df['patient_count'] = df['patient_count'].apply(lambda x: f"{x:,}")

    # Rename columns for display
    column_renames = {
        'disease': 'Disease',
        'patient_count': 'Participant Count',
        'percentage': 'Percentage',
        'columns_found': 'Fields Found'
    }
    df = df.rename(columns=column_renames)

    static_table(df)


def display_facility_table(facility_data: List[Dict[str, Any]],
                           show_top_n: Optional[int] = None) -> None:
    """
    Display facility distribution table.

    Args:
        facility_data: List of facility dicts
        show_top_n: Show only top N facilities
    """
    df = pd.DataFrame(facility_data)

    if show_top_n:
        df = df.head(show_top_n)

    # Format patient counts
    df['patient_count'] = df['patient_count'].apply(lambda x: f"{x:,}")

    # Rename columns
    df = df.rename(columns={
        'FACILITY_DISPLAY_ID': 'ID',
        'FACILITY_NAME': 'Facility Name',
        'patient_count': 'Participant Count'
    })

    static_table(df)


def display_cohort_summary(cohort_summary: Dict[str, Any]) -> None:
    """
    Display cohort summary statistics in a table.

    Args:
        cohort_summary: Dict with cohort statistics
    """
    summary_data = []

    for key, value in cohort_summary.items():
        # Format the key nicely
        display_key = key.replace('_', ' ').title()

        # Format the value
        if isinstance(value, (int, float)) and key != 'facility_count':
            display_value = f"{value:,}"
        elif isinstance(value, dict):
            display_value = str(len(value)) + " types"
        else:
            display_value = str(value)

        summary_data.append({
            'Metric': display_key,
            'Value': display_value
        })

    df = pd.DataFrame(summary_data)
    static_table(df)


def display_data_preview(df: pd.DataFrame,
                        max_rows: int = 100,
                        show_info: bool = True) -> None:
    """
    Display a data preview with optional info.

    Args:
        df: DataFrame to display
        max_rows: Maximum rows to show
        show_info: Whether to show info about the data
    """
    if show_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")

    # Show preview
    st.dataframe(df.head(max_rows), use_container_width=True, hide_index=True)

    if len(df) > max_rows:
        st.info(f"Showing first {max_rows:,} of {len(df):,} rows")


def create_download_button(df: pd.DataFrame,
                          filename: str,
                          label: str = "Download as CSV",
                          key: Optional[str] = None) -> None:
    """
    Create a download button for a DataFrame.

    Args:
        df: DataFrame to download
        filename: Filename for download
        label: Button label
        key: Optional button key
    """
    csv = df.to_csv(index=False)

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime='text/csv',
        key=key
    )
