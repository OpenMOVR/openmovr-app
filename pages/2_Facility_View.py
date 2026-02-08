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
from components.charts import create_site_map
from components.tables import static_table
from utils.cache import get_cached_facility_stats, get_cached_snapshot
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
    .clean-table { width: 100%; border-collapse: collapse; }
    .clean-table th { text-align: left; padding: 6px 12px; border-bottom: 2px solid #ddd; }
    .clean-table td { padding: 6px 12px; border-bottom: 1px solid #eee; }
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

# Load snapshot for site geography
try:
    snapshot = get_cached_snapshot()
    site_locations = snapshot['facilities'].get('site_locations', [])
except Exception:
    site_locations = []

# Summary metrics
st.markdown("---")
st.subheader("Facility Overview")

col1, col2, col3, col4 = st.columns(4)

facility_df = pd.DataFrame(all_facilities)
total_patients = facility_df['patient_count'].sum()
avg_patients = facility_df['patient_count'].mean()
median_patients = facility_df['patient_count'].median()

with col1:
    st.metric("Total Facilities", total_facilities, help="Number of participating facilities")
with col2:
    st.metric("Total Patients", f"{total_patients:,}", help="Total patients across all facilities")
with col3:
    st.metric("Avg Patients/Site", f"{avg_patients:.0f}", help="Average number of patients per site")
with col4:
    st.metric("Median Patients", f"{median_patients:.0f}", help="Median number of patients per site")

# Site Map
st.markdown("---")
st.subheader("Site Locations")

if site_locations:
    from components.charts import _prepare_site_df

    # Build disease list from actual data
    _all_ds = set()
    for s in site_locations:
        _all_ds.update((s.get("by_disease") or {}).keys())
    _disease_options = ["All Diseases"] + sorted(_all_ds)

    # Filters
    filt1, filt2 = st.columns([1, 1])
    with filt1:
        fv_top_n = st.slider(
            "Number of sites to display",
            min_value=5,
            max_value=min(60, len(site_locations)),
            value=min(20, len(site_locations)),
            step=1,
            key="fv_top_n",
        )
    with filt2:
        fv_disease = st.selectbox(
            "Filter by disease",
            options=_disease_options,
            index=0,
            key="fv_disease_filter",
        )

    fv_ds_filter = None if fv_disease == "All Diseases" else fv_disease
    fv_title = "MOVR Participating Sites"
    if fv_ds_filter:
        fv_title = f"MOVR Sites — {fv_ds_filter} Patients"

    site_map = create_site_map(
        site_locations,
        disease_filter=fv_ds_filter,
        top_n=fv_top_n,
        title=fv_title,
    )
    if site_map:
        st.plotly_chart(site_map, use_container_width=True)
    else:
        st.info("No sites with data for this selection.")

    # State & Region summaries (respecting current filter)
    filtered_df = _prepare_site_df(site_locations, fv_ds_filter, continental_only=True, top_n=fv_top_n)
    if filtered_df is not None and not filtered_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### By State")
            state_counts = {}
            for _, s in filtered_df.iterrows():
                st_name = s['state']
                if st_name not in state_counts:
                    state_counts[st_name] = {"Sites": 0, "Patients": 0}
                state_counts[st_name]["Sites"] += 1
                state_counts[st_name]["Patients"] += s['patient_count']
            state_df = pd.DataFrame([
                {"State": k, "Sites": v["Sites"], "Patients": v["Patients"]}
                for k, v in state_counts.items()
            ]).sort_values("Patients", ascending=False).reset_index(drop=True)
            static_table(state_df)

        with col2:
            st.markdown("#### By Region")
            region_counts = {}
            for _, s in filtered_df.iterrows():
                rgn = s.get('region', 'Unknown') or 'Unknown'
                if rgn not in region_counts:
                    region_counts[rgn] = {"Sites": 0, "Patients": 0}
                region_counts[rgn]["Sites"] += 1
                region_counts[rgn]["Patients"] += s['patient_count']
            region_df = pd.DataFrame([
                {"Region": k, "Sites": v["Sites"], "Patients": v["Patients"]}
                for k, v in region_counts.items()
            ]).sort_values("Patients", ascending=False).reset_index(drop=True)
            static_table(region_df)
else:
    st.info("Site geographic data not available.")

# All Sites Table (anonymized)
st.markdown("---")
st.subheader("All Sites")

if site_locations:
    # Use filtered data
    table_df = _prepare_site_df(site_locations, fv_ds_filter, continental_only=True, top_n=None)

    # Search filter
    search_term = st.text_input(
        "Search sites by city or state",
        placeholder="Enter city or state...",
        key="site_search"
    )

    if table_df is not None and not table_df.empty:
        if search_term:
            table_df = table_df[
                table_df['city'].str.lower().str.contains(search_term.lower(), na=False) |
                table_df['state'].str.lower().str.contains(search_term.lower(), na=False)
            ]
            st.info(f"Found {len(table_df)} sites matching '{search_term}'")

        display_rows = []
        for i, (_, r) in enumerate(table_df.iterrows()):
            row_data = {
                "Rank": i + 1,
                "City": r["city"],
                "State": r["state"],
                "Region": r.get("region", ""),
                "Type": r.get("site_type", ""),
                "Patients": r["patient_count"],
            }
            by_ds = r.get("by_disease")
            if isinstance(by_ds, dict) and by_ds:
                row_data["Diseases"] = ", ".join(sorted(by_ds.keys()))
            display_rows.append(row_data)
        static_table(pd.DataFrame(display_rows))
    else:
        st.warning("No sites found matching your search.")
else:
    static_table(
        facility_df[['FACILITY_DISPLAY_ID', 'patient_count']].rename(
            columns={'FACILITY_DISPLAY_ID': 'Site ID', 'patient_count': 'Patients'}
        )
    )

# Distribution Analysis
st.markdown("---")
st.subheader("Site Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Patient Count Ranges")
    bins = [0, 10, 25, 50, 100, 200, 1000]
    labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '200+']
    facility_df['patient_range'] = pd.cut(
        facility_df['patient_count'], bins=bins, labels=labels, include_lowest=True
    )
    range_counts = facility_df['patient_range'].value_counts().sort_index()
    range_df = pd.DataFrame({'Patient Range': range_counts.index, 'Site Count': range_counts.values})
    static_table(range_df)

with col2:
    st.markdown("#### Key Statistics")
    stats_data = {
        'Metric': ['Smallest Site', 'Largest Site', 'Average Size', 'Median Size', 'Std Dev', 'Total Patients'],
        'Value': [
            f"{facility_df['patient_count'].min()} patients",
            f"{facility_df['patient_count'].max()} patients",
            f"{facility_df['patient_count'].mean():.1f} patients",
            f"{facility_df['patient_count'].median():.0f} patients",
            f"{facility_df['patient_count'].std():.1f}",
            f"{facility_df['patient_count'].sum():,} patients",
        ]
    }
    static_table(pd.DataFrame(stats_data))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<strong>OpenMOVR App</strong> | MOVR Data Hub (MOVR 1.0)<br>"
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
    "Site data from the MOVR database (anonymized)<br>"
    "<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
    "<a href='mailto:andre.paredes@ymail.com'>andre.paredes@ymail.com</a> | "
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
    "</div>",
    unsafe_allow_html=True
)
