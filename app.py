"""
OpenMOVR App - Web Application

Main entry point for the Streamlit web application.

Usage:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Add current directory to path for imports
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from config.settings import *
from api import StatsAPI
from utils.cache import get_cached_snapshot
from components.charts import (
    create_disease_distribution_chart,
    create_facility_chart,
    create_data_availability_chart
)
from components.tables import display_disease_table, display_facility_table


# Page configuration — use logo as favicon
_logo_path = Path(__file__).parent / "assets" / "movr_logo_clean_nobackground.png"
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)


def main():
    """Main application entry point."""

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

    # Sidebar: logo (centered) + contact
    with st.sidebar:
        if LOGO_PNG.exists():
            st.markdown(
                "<div style='text-align: center;'>",
                unsafe_allow_html=True
            )
            st.image(str(LOGO_PNG), width=160)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; font-size: 0.8em; color: #888;'>
                <strong>Open Source Project</strong><br>
                <a href="https://openmovr.github.io" target="_blank">openmovr.github.io</a><br>
                <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; font-size: 0.8em; color: #888;'>
                <strong>Created by</strong><br>
                Andre D Paredes<br>
                <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a><br>
                <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a> (MDA)
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; font-size: 0.8em; color: #888;'>
                <strong>Request Data</strong><br>
                <a href="https://mdausa.tfaforms.net/389761" target="_blank">MDA Data Request Form</a>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Header with logo + branding
    header_left, header_right = st.columns([3, 1])

    with header_left:
        st.title(APP_TITLE)
        st.markdown("### Interactive Dashboard for MOVR Clinical Data Analytics")

    with header_right:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 10px;'>
                <span style='font-size: 1.5em; font-weight: bold; color: #1E88E5;'>OpenMOVR App</span><br>
                <span style='font-size: 0.9em; color: #666; background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px;'>
                    Gen1 | v{APP_VERSION} (Prototype)
                </span><br>
                <span style='font-size: 0.75em; color: #999; margin-top: 4px; display: inline-block;'>
                    {STUDY_NAME} | {STUDY_PROTOCOL}<br>
                    Study: {STUDY_START} &ndash; {STUDY_END}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Orientation blurb
    st.markdown(
        f"""
        <div style='background-color: #E3F2FD; border-left: 4px solid #1E88E5; padding: 12px 16px;
        border-radius: 0 4px 4px 0; margin: 0.5rem 0 1rem 0; font-size: 0.95em;'>
        Analytics dashboard for the <strong>{STUDY_NAME}</strong> clinical registry
        ({STUDY_PROTOCOL} study protocol, {STUDY_START} &ndash; {STUDY_END}).
        Explore disease distributions, facility data, and disease-specific profiles
        for rare neuromuscular diseases.
        See <strong>About</strong> for study details and version history.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if snapshot exists
    if not StatsAPI.snapshot_exists():
        st.error(
            "**Database snapshot not found!**\n\n"
            "Please generate the statistics snapshot first:\n\n"
            "```bash\n"
            "python scripts/generate_stats_snapshot.py\n"
            "```"
        )
        st.stop()

    # Load snapshot
    try:
        snapshot = get_cached_snapshot()
        metadata = snapshot['metadata']
        enrollment = snapshot['enrollment']
    except Exception as e:
        st.error(f"Error loading snapshot: {e}")
        st.stop()

    # Snapshot info in sidebar
    with st.sidebar:
        st.info(
            f"**Snapshot Info**\n\n"
            f"Generated: {metadata['generated_timestamp']}\n\n"
            f"Cohort: {metadata['cohort_type']}\n\n"
            f"Age: {StatsAPI.get_snapshot_age()}"
        )

        if st.button("Refresh Snapshot"):
            st.cache_data.clear()
            st.rerun()

    # Key Metrics Row
    st.markdown("---")
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Patients",
            f"{enrollment['total_patients']:,}",
            help="Total MOVR patients with validated enrollment"
        )

    with col2:
        st.metric(
            "Total Facilities",
            snapshot['facilities']['total_facilities'],
            help="Number of participating facilities"
        )

    with col3:
        disease_count = len(snapshot['disease_distribution']['diseases'])
        st.metric(
            "Disease Types",
            disease_count,
            help="Number of disease types in the database"
        )

    with col4:
        enrollment_rate = enrollment['validation_stats']['enrollment_rate']
        st.metric(
            "Enrollment Rate",
            f"{enrollment_rate:.1f}%",
            help="Percentage of records with validated enrollment"
        )

    # Disease Distribution Section
    st.markdown("---")
    st.subheader("Disease Distribution")

    col1, col2 = st.columns([2, 1])

    with col1:
        chart_type = st.radio(
            "Chart Type",
            options=['Bar Chart', 'Pie Chart', 'Horizontal Bar'],
            horizontal=True,
            label_visibility="collapsed"
        )

        chart_type_map = {
            'Bar Chart': 'bar',
            'Pie Chart': 'pie',
            'Horizontal Bar': 'horizontal_bar'
        }

        disease_chart = create_disease_distribution_chart(
            snapshot['disease_distribution']['diseases'],
            chart_type=chart_type_map[chart_type]
        )
        st.plotly_chart(disease_chart, use_container_width=True)

    with col2:
        st.markdown("#### Patient Counts by Disease")
        display_disease_table(
            snapshot['disease_distribution']['diseases'],
            show_columns=['disease', 'patient_count', 'percentage']
        )

    # Facility Distribution Section
    st.markdown("---")
    st.subheader("Top Facilities")

    top_n = st.slider(
        "Number of facilities to display",
        min_value=5,
        max_value=20,
        value=DEFAULT_TOP_N_FACILITIES,
        step=1
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        facility_chart = create_facility_chart(
            snapshot['facilities']['all_facilities'],
            top_n=top_n,
            chart_type='horizontal_bar'
        )
        st.plotly_chart(facility_chart, use_container_width=True)

    with col2:
        st.markdown(f"#### Top {top_n} Facilities")
        display_facility_table(
            snapshot['facilities']['all_facilities'],
            show_top_n=top_n
        )

    # Data Availability Section
    st.markdown("---")
    st.subheader("Data Availability")

    col1, col2 = st.columns([2, 1])

    with col1:
        data_avail_chart = create_data_availability_chart(
            snapshot['data_availability']
        )
        st.plotly_chart(data_avail_chart, use_container_width=True)

    with col2:
        st.markdown("#### Record Counts")
        for data_type, count in snapshot['data_availability'].items():
            display_name = data_type.replace('_records', '').title()
            st.metric(display_name, f"{count:,}")

    # Enrollment Details (Expandable)
    st.markdown("---")
    with st.expander("Enrollment Validation Details", expanded=False):
        val_stats = enrollment['validation_stats']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Demographics", f"{val_stats['total_demographics']:,}")
            st.metric("Total Diagnosis", f"{val_stats['total_diagnosis']:,}")

        with col2:
            st.metric("Total Encounters", f"{val_stats['total_encounters']:,}")
            st.metric("Validated Enrollment", f"{val_stats['validated_enrollment']:,}")

        with col3:
            st.metric("Missing Diagnosis", f"{val_stats['excluded_patients']['missing_diagnosis']:,}")
            st.metric("Missing Encounters", f"{val_stats['excluded_patients']['missing_encounters']:,}")

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"<strong>OpenMOVR App</strong> Gen1 | v{APP_VERSION} (Prototype)<br>"
        f"{STUDY_NAME} Study Protocol ({STUDY_PROTOCOL}) | "
        f"{STUDY_START} &ndash; {STUDY_END} ({STUDY_STATUS})<br>"
        f"<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
        f"Snapshot: {metadata['generated_timestamp']}<br>"
        f"<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
        f"<a href='mailto:andre.paredes@ymail.com'>andre.paredes@ymail.com</a> | "
        f"<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
