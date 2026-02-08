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
import pandas as pd
from config.settings import *
from api import StatsAPI
from utils.cache import get_cached_snapshot
from components.charts import (
    create_disease_distribution_chart,
    create_facility_chart,
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
                <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a><br><br>
                <strong>Created by</strong> Andre D Paredes<br>
                <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a><br>
                <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a> (MDA)<br><br>
                <a href="https://mdausa.tfaforms.net/389761" target="_blank"><strong>Request Data</strong></a>
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
        longitudinal = snapshot.get('longitudinal', {})
        total_enc = longitudinal.get('total_encounters', 0)
        st.metric(
            "Total Encounters",
            f"{total_enc:,}",
            help="Total clinic visit records across all patients"
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

    # Longitudinal Data Section
    st.markdown("---")
    st.subheader("Longitudinal Data")

    longitudinal = snapshot.get('longitudinal', {})
    if longitudinal:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Mean Visits / Patient",
                f"{longitudinal.get('mean_encounters_per_patient', 0)}",
                help="Average clinic visits per patient"
            )
        with col2:
            st.metric(
                "Median Visits / Patient",
                f"{longitudinal.get('median_encounters_per_patient', 0)}",
                help="Median clinic visits per patient"
            )
        with col3:
            st.metric(
                "Patients with 3+ Visits",
                f"{longitudinal.get('patients_3plus_encounters', 0):,}",
                help="Patients with 3 or more clinic encounters"
            )
        with col4:
            st.metric(
                "Patients with 5+ Visits",
                f"{longitudinal.get('patients_5plus_encounters', 0):,}",
                help="Patients with 5 or more clinic encounters"
            )

        # Encounters by disease table
        by_disease = longitudinal.get('by_disease', {})
        if by_disease:
            enc_data = []
            for ds in ["ALS", "DMD", "SMA", "LGMD", "FSHD", "BMD", "Pompe"]:
                info = by_disease.get(ds, {})
                if info:
                    enc_data.append({
                        "Disease": ds,
                        "Patients": f"{info['patients']:,}",
                        "Encounters": f"{info['encounters']:,}",
                        "Mean / Patient": info['mean_per_patient'],
                        "With 3+ Visits": f"{info['patients_3plus']:,}",
                    })
            if enc_data:
                st.dataframe(
                    pd.DataFrame(enc_data),
                    use_container_width=True,
                    hide_index=True,
                )

    # Clinical Data Highlights
    st.markdown("---")
    st.subheader("Clinical Data Highlights")
    st.caption("Unique participants with at least one recorded value per domain.")

    clinical = snapshot.get('clinical_availability', {})
    if clinical:
        # Functional Assessments (with longitudinal counts)
        st.markdown("##### Functional Assessments")
        func_scores = clinical.get('functional_scores', {})
        if func_scores:
            fcols = st.columns(len(func_scores))
            for i, (key, info) in enumerate(func_scores.items()):
                with fcols[i]:
                    st.metric(
                        info.get('label', key),
                        f"{info['patients']:,}",
                        help=f"{info['patients_longitudinal']:,} with repeat assessments"
                    )
                    st.caption(f"{info['patients_longitudinal']:,} longitudinal")

        # Timed Tests | Medications
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Timed Function Tests")
            timed = clinical.get('timed_tests', {})
            t1, t2, t3 = st.columns(3)
            with t1:
                st.metric("10m Walk/Run", f"{timed.get('walk_run_10m', 0):,}")
            with t2:
                st.metric("Stair Climb", f"{timed.get('stair_climb', 0):,}")
            with t3:
                st.metric("Rise from Supine", f"{timed.get('rise_from_supine', 0):,}")

        with col_right:
            st.markdown("##### Medications & Treatments")
            meds = clinical.get('medications', {})
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Glucocorticoids", f"{meds.get('glucocorticoid', 0):,}")
            with m2:
                st.metric("Disease-Modifying", f"{meds.get('disease_modifying_therapy', 0):,}")
            with m3:
                st.metric("ERT (Pompe)", f"{meds.get('ert_current', 0):,}")
            with m4:
                st.metric("Med Reviews", f"{meds.get('medication_reviewed', 0):,}")

        # Pulmonary | Cardiology
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Pulmonary & Respiratory")
            pulm = clinical.get('pulmonary', {})
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric("PFTs", f"{pulm.get('pft_performed', 0):,}")
            with p2:
                st.metric("FVC", f"{pulm.get('fvc', 0):,}")
            with p3:
                st.metric("FEV1", f"{pulm.get('fev1', 0):,}")

        with col_right:
            st.markdown("##### Cardiology")
            cardiac = clinical.get('cardiac', {})
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("ECG", f"{cardiac.get('ecg', 0):,}")
            with c2:
                st.metric("Echo", f"{cardiac.get('echo', 0):,}")
            with c3:
                st.metric("Cardiomyopathy", f"{cardiac.get('cardiomyopathy', 0):,}")

        # Mobility | Vital Signs
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Mobility & Ambulation")
            mob = clinical.get('mobility', {})
            mb1, mb2, mb3, mb4 = st.columns(4)
            with mb1:
                st.metric("Ambulatory Status", f"{mob.get('ambulatory_status', 0):,}")
            with mb2:
                st.metric("Wheelchair", f"{mob.get('wheelchair', 0):,}")
            with mb3:
                st.metric("Assistive Device", f"{mob.get('assistive_device', 0):,}")
            with mb4:
                st.metric("Falls Reported", f"{mob.get('falls', 0):,}")

        with col_right:
            st.markdown("##### Vital Signs & Anthropometrics")
            vitals = clinical.get('vital_signs', {})
            v1, v2, v3 = st.columns(3)
            with v1:
                st.metric("Height", f"{vitals.get('height', 0):,}")
            with v2:
                st.metric("Weight", f"{vitals.get('weight', 0):,}")
            with v3:
                st.metric("BMI", f"{vitals.get('bmi', 0):,}")

        # Nutrition | Orthopedic
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Nutrition & GI")
            nutr = clinical.get('nutrition', {})
            n1, n2, n3 = st.columns(3)
            with n1:
                st.metric("Nutrition Therapy", f"{nutr.get('nutritional_supplementation', 0):,}")
            with n2:
                st.metric("Feeding Method", f"{nutr.get('feeding_method', 0):,}")
            with n3:
                st.metric("Feeding Tube", f"{nutr.get('feeding_tube', 0):,}")

        with col_right:
            st.markdown("##### Orthopedic & Surgical")
            ortho = clinical.get('orthopedic', {})
            o1, o2 = st.columns(2)
            with o1:
                st.metric("Scoliosis", f"{ortho.get('scoliosis', 0):,}")
            with o2:
                st.metric("Spinal X-ray", f"{ortho.get('spinal_xray', 0):,}")

        # Hospitalizations & Care
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### Hospitalizations")
            hosp = clinical.get('hospitalizations', {})
            st.metric("Patients with Hospitalizations", f"{hosp.get('hospitalizations', 0):,}")

        with col_right:
            st.markdown("##### Multidisciplinary Care")
            care = clinical.get('care', {})
            cr1, cr2, cr3 = st.columns(3)
            with cr1:
                st.metric("Care Plans", f"{care.get('multidisciplinary_plan', 0):,}")
            with cr2:
                st.metric("Seen By", f"{care.get('specialists_seen', 0):,}")
            with cr3:
                st.metric("Referred To", f"{care.get('specialists_referred', 0):,}")

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
