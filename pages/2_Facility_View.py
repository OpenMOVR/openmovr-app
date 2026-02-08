"""
Facility View Page

Explore facility distribution and compare facilities.
"""

import sys
from pathlib import Path

# Add both webapp and parent directory to path FIRST (before other imports)
app_dir = Path(__file__).parent.parent
# App root is parent of pages/
sys.path.insert(0, str(app_dir))
# src is in app root

import streamlit as st
import pandas as pd

from api import StatsAPI
from components.charts import create_facility_chart
from components.tables import display_facility_table
from utils.cache import get_cached_facility_stats
from config.settings import PAGE_ICON, DEFAULT_TOP_N_FACILITIES, LOGO_PNG

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

# Page configuration
st.set_page_config(
    page_title="Facility View - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide"
)

# Custom CSS to add branding above page navigation
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        padding-top: 0rem;
    }
    [data-testid="stSidebarNav"]::before {
        content: "OpenMOVR App";
        display: block;
        font-size: 1.4em;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    [data-testid="stSidebarNav"]::after {
        content: "MOVR Data Hub | MOVR 1.0";
        display: block;
        font-size: 0.8em;
        color: #666;
        text-align: center;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Contact info at bottom of sidebar
st.sidebar.markdown("---")
if LOGO_PNG.exists():
    st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.sidebar.image(str(LOGO_PNG), width=160)
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; font-size: 0.8em; color: #888;'>
        <strong>Open Source Project</strong><br>
        <a href="https://openmovr.github.io" target="_blank">openmovr.github.io</a><br>
        <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a><br><br>
        <strong>Created by</strong> Andre D Paredes<br>
        <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a><br>
        <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a> (MDA)<br><br>
        <a href="https://mdausa.tfaforms.net/389761" target="_blank"><strong>Request Data</strong></a>
    </div>
    """,
    unsafe_allow_html=True
)

# Header with branding
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("Facility View")
    st.markdown("### Explore facility distribution and patient counts")

with header_right:
    st.markdown(
        """
        <div style='text-align: right; padding-top: 10px;'>
            <span style='font-size: 1.5em; font-weight: bold; color: #1E88E5;'>OpenMOVR App</span><br>
            <span style='font-size: 0.9em; color: #666; background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px;'>
                Gen1 | v0.1.0 (Prototype)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# Load facility data
with st.spinner("Loading facility data..."):
    try:
        facility_stats = get_cached_facility_stats()
        all_facilities = facility_stats['all_facilities']
        total_facilities = facility_stats['total_facilities']
    except Exception as e:
        st.error(f"Error loading facility data: {e}")
        st.stop()

# Summary metrics
st.markdown("---")
st.subheader("📊 Facility Overview")

col1, col2, col3, col4 = st.columns(4)

# Calculate stats
facility_df = pd.DataFrame(all_facilities)
total_patients = facility_df['patient_count'].sum()
avg_patients = facility_df['patient_count'].mean()
median_patients = facility_df['patient_count'].median()

with col1:
    st.metric(
        "Total Facilities",
        total_facilities,
        help="Number of participating facilities"
    )

with col2:
    st.metric(
        "Total Patients",
        f"{total_patients:,}",
        help="Total patients across all facilities"
    )

with col3:
    st.metric(
        "Avg Patients/Facility",
        f"{avg_patients:.0f}",
        help="Average number of patients per facility"
    )

with col4:
    st.metric(
        "Median Patients",
        f"{median_patients:.0f}",
        help="Median number of patients per facility"
    )

# Top Facilities Section
st.markdown("---")
st.subheader("🔝 Top Facilities by Patient Count")

# Number of facilities slider
top_n = st.slider(
    "Number of facilities to display",
    min_value=5,
    max_value=min(25, total_facilities),
    value=DEFAULT_TOP_N_FACILITIES,
    step=1,
    key="top_n_slider"
)

col1, col2 = st.columns([2, 1])

with col1:
    # Chart
    facility_chart = create_facility_chart(
        all_facilities,
        top_n=top_n,
        chart_type='horizontal_bar'
    )
    st.plotly_chart(facility_chart, use_container_width=True)

with col2:
    # Table
    st.markdown(f"#### Top {top_n} Facilities")
    display_facility_table(all_facilities, show_top_n=top_n)

# All Facilities Table
st.markdown("---")
st.subheader("📋 All Facilities")

# Search filter
search_term = st.text_input(
    "🔍 Search facilities by name or ID",
    placeholder="Enter facility name or ID...",
    key="facility_search"
)

# Filter facilities based on search
if search_term:
    filtered_facilities = [
        f for f in all_facilities
        if search_term.lower() in f['FACILITY_NAME'].lower()
        or search_term.lower() in f['FACILITY_DISPLAY_ID'].lower()
    ]
    st.info(f"Found {len(filtered_facilities)} facilities matching '{search_term}'")
else:
    filtered_facilities = all_facilities

# Display table
facility_table_df = pd.DataFrame(filtered_facilities)

if not facility_table_df.empty:
    # Add rank column
    facility_table_df['Rank'] = range(1, len(facility_table_df) + 1)

    # Format and rename columns
    display_df = facility_table_df[[
        'Rank',
        'FACILITY_DISPLAY_ID',
        'FACILITY_NAME',
        'patient_count'
    ]].copy()

    display_df['patient_count'] = display_df['patient_count'].apply(lambda x: f"{x:,}")

    display_df = display_df.rename(columns={
        'FACILITY_DISPLAY_ID': 'Facility ID',
        'FACILITY_NAME': 'Facility Name',
        'patient_count': 'Patient Count'
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True, height=600)

    # Download button
    csv = facility_table_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Facility Data (CSV)",
        data=csv,
        file_name="movr_facilities.csv",
        mime="text/csv",
        key="download_facilities"
    )
else:
    st.warning("No facilities found matching your search.")

# Distribution Analysis
st.markdown("---")
st.subheader("📈 Facility Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Patient Count Ranges")

    # Create patient count bins
    bins = [0, 10, 25, 50, 100, 200, 1000]
    labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '200+']

    facility_df['patient_range'] = pd.cut(
        facility_df['patient_count'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    range_counts = facility_df['patient_range'].value_counts().sort_index()

    range_df = pd.DataFrame({
        'Patient Range': range_counts.index,
        'Facility Count': range_counts.values
    })

    st.dataframe(range_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### Key Statistics")

    stats_data = {
        'Metric': [
            'Smallest Facility',
            'Largest Facility',
            'Average Size',
            'Median Size',
            'Standard Deviation',
            'Total Patients'
        ],
        'Value': [
            f"{facility_df['patient_count'].min()} patients",
            f"{facility_df['patient_count'].max()} patients",
            f"{facility_df['patient_count'].mean():.1f} patients",
            f"{facility_df['patient_count'].median():.0f} patients",
            f"{facility_df['patient_count'].std():.1f}",
            f"{facility_df['patient_count'].sum():,} patients"
        ]
    }

    st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<strong>OpenMOVR App</strong> | MOVR Data Hub (MOVR 1.0)<br>"
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
    "Facility data from the MOVR database<br>"
    "<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
    "<a href='mailto:andre.paredes@ymail.com'>andre.paredes@ymail.com</a> | "
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
    "</div>",
    unsafe_allow_html=True
)
