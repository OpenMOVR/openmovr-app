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
import plotly.express as px

from api import StatsAPI
from components.charts import create_site_map
from components.tables import static_table
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer, render_page_header
from utils.cache import get_cached_facility_stats, get_cached_snapshot
from config.settings import PAGE_ICON, DEFAULT_TOP_N_FACILITIES

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

# Page configuration
st.set_page_config(
    page_title="Facility View - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide"
)

inject_global_css()
render_sidebar_footer()

render_page_header("Facility View", "Explore facility distribution and participant counts")

# Load facility data
with st.spinner("Loading facility data..."):
    try:
        facility_stats = get_cached_facility_stats()
        all_facilities = facility_stats['all_facilities']
        total_facilities = facility_stats['total_facilities']
        site_locations = facility_stats.get('site_locations', [])
    except Exception as e:
        st.error(f"Error loading facility data: {e}")
        st.stop()

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
    st.metric("Total Participants", f"{total_patients:,}", help="Total participants across all facilities")
with col3:
    st.metric("Avg Participants/Site", f"{avg_patients:.0f}", help="Average number of participants per site")
with col4:
    st.metric("Median Participants", f"{median_patients:.0f}", help="Median number of participants per site")

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
        fv_title = f"MOVR Sites — {fv_ds_filter} Participants"

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
                    state_counts[st_name] = {"Sites": 0, "Participants": 0}
                state_counts[st_name]["Sites"] += 1
                state_counts[st_name]["Participants"] += s['patient_count']
            state_df = pd.DataFrame([
                {"State": k, "Sites": v["Sites"], "Participants": v["Participants"]}
                for k, v in state_counts.items()
            ]).sort_values("Participants", ascending=False).reset_index(drop=True)
            static_table(state_df)

        with col2:
            st.markdown("#### By Region")
            region_counts = {}
            for _, s in filtered_df.iterrows():
                rgn = s.get('region', 'Unknown') or 'Unknown'
                if rgn not in region_counts:
                    region_counts[rgn] = {"Sites": 0, "Participants": 0}
                region_counts[rgn]["Sites"] += 1
                region_counts[rgn]["Participants"] += s['patient_count']
            region_df = pd.DataFrame([
                {"Region": k, "Sites": v["Sites"], "Participants": v["Participants"]}
                for k, v in region_counts.items()
            ]).sort_values("Participants", ascending=False).reset_index(drop=True)
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
                "Participants": r["patient_count"],
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
            columns={'FACILITY_DISPLAY_ID': 'Site ID', 'patient_count': 'Participants'}
        )
    )

# Distribution Analysis
st.markdown("---")
st.subheader("Site Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Participant Count Ranges")
    bins = [0, 10, 25, 50, 100, 200, 1000]
    labels = ['1-10', '11-25', '26-50', '51-100', '101-200', '200+']
    facility_df['patient_range'] = pd.cut(
        facility_df['patient_count'], bins=bins, labels=labels, include_lowest=True
    )
    range_counts = facility_df['patient_range'].value_counts().sort_index()
    range_df = pd.DataFrame({'Participant Range': range_counts.index, 'Site Count': range_counts.values})
    static_table(range_df)

with col2:
    st.markdown("#### Key Statistics")
    stats_data = {
        'Metric': ['Smallest Site', 'Largest Site', 'Average Size', 'Median Size', 'Std Dev', 'Total Participants'],
        'Value': [
            f"{facility_df['patient_count'].min()} participants",
            f"{facility_df['patient_count'].max()} participants",
            f"{facility_df['patient_count'].mean():.1f} participants",
            f"{facility_df['patient_count'].median():.0f} participants",
            f"{facility_df['patient_count'].std():.1f}",
            f"{facility_df['patient_count'].sum():,} participants",
        ]
    }
    static_table(pd.DataFrame(stats_data))

# ===================================================================
# RECRUITMENT OVER TIME BY STATE
# ===================================================================
st.markdown("---")
st.subheader("Recruitment Over Time")

try:
    _snapshot = get_cached_snapshot()
    timeline = _snapshot.get('enrollment_timeline', {})
    by_sdm = timeline.get('by_state_disease_month', [])

    if by_sdm:
        recruit_df = pd.DataFrame(by_sdm)
        recruit_df['month'] = pd.to_datetime(recruit_df['month'])

        # State selector
        all_states = sorted(recruit_df['state'].unique())
        selected_states = st.multiselect(
            "Filter by State",
            options=all_states,
            default=None,
            key="recruit_state_filter",
            help="Leave empty to show all states"
        )

        if selected_states:
            plot_df = recruit_df[recruit_df['state'].isin(selected_states)]
        else:
            plot_df = recruit_df

        # Aggregate by month + disease (across selected states)
        agg_df = plot_df.groupby(['month', 'disease'], as_index=False)['count'].sum()
        agg_df = agg_df.sort_values(['disease', 'month'])
        agg_df['cumulative'] = agg_df.groupby('disease')['count'].cumsum()

        title_suffix = f" ({', '.join(selected_states)})" if selected_states else " (All States)"

        col_cumul, col_monthly = st.columns(2)
        with col_cumul:
            fig = px.line(
                agg_df,
                x='month',
                y='cumulative',
                color='disease',
                title=f'Cumulative Recruitment{title_suffix}',
                labels={'month': '', 'cumulative': 'Cumulative Participants', 'disease': 'Disease'},
            )
            fig.update_layout(height=450, legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(fig, use_container_width=True)

        with col_monthly:
            monthly_df = plot_df.groupby('month', as_index=False)['count'].sum()
            fig_m = px.bar(
                monthly_df,
                x='month',
                y='count',
                title=f'Monthly New Enrollments{title_suffix}',
                labels={'month': '', 'count': 'New Participants'},
                color_discrete_sequence=['#1E88E5'],
            )
            fig_m.update_layout(height=450)
            st.plotly_chart(fig_m, use_container_width=True)

        # Notes
        notes = []
        missing = timeline.get('missing_date_count', 0)
        clamped = timeline.get('pre_study_clamped', 0)
        if missing > 0:
            notes.append(f"{missing} participants with missing enrollment dates defaulted to first encounter or study start (Nov 2018)")
        if clamped > 0:
            notes.append(f"{clamped} pre-study enrollment dates clamped to first encounter or study start")
        notes.append("State is determined by facility location, not participant residence")
        st.caption(" | ".join(notes))
    else:
        st.info("Enrollment timeline data not available in snapshot.")
except Exception as e:
    st.info(f"Recruitment timeline not available: {e}")

# Footer
render_page_footer()
