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
    create_site_map,
)
from components.tables import display_disease_table, static_table
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer


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

    inject_global_css()
    render_sidebar_footer()

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

    # Compact snapshot info in sidebar
    with st.sidebar:
        st.markdown(
            f"<div style='font-size: 0.7em; color: #999; text-align: center; padding: 4px 0;'>"
            f"Snapshot: {metadata['generated_timestamp']} &middot; {StatsAPI.get_snapshot_age()}"
            f"</div>",
            unsafe_allow_html=True
        )

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

    # Participating Sites Map
    st.markdown("---")
    st.subheader("Participating Sites")

    site_locations = snapshot['facilities'].get('site_locations', [])
    if site_locations:
        site_map = create_site_map(site_locations, title="MOVR Participating Sites")
        if site_map:
            st.plotly_chart(site_map, use_container_width=True)

        # Compact summary
        continental = [s for s in site_locations
                       if s.get('continental', True) and s.get('lat') is not None]
        total_mapped = len(continental)
        states_covered = len(set(s['state'] for s in continental if s.get('state')))
        st.caption(
            f"{total_mapped} sites across {states_covered} states + DC.  "
            "Filter by disease and explore site details on the **Facility View** page."
        )
    else:
        st.info("Site geographic data not available.")

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
                static_table(pd.DataFrame(enc_data))

    # Clinical Data Highlights
    st.markdown("---")
    st.subheader("Clinical Data Highlights")
    st.caption("Unique MOVR participants with at least one recorded value.")

    clinical = snapshot.get('clinical_availability', {})
    if clinical:

        # --- Row 1: Functional Assessments | Timed Tests ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Functional Assessments")
            func_scores = clinical.get('functional_scores', {})
            if func_scores:
                func_data = []
                for key, info in func_scores.items():
                    func_data.append({
                        "Assessment": info.get('label', key),
                        "Participants": f"{info['patients']:,}",
                        "Data Points": f"{info.get('data_points', 0):,}",
                        "Longitudinal (2+)": f"{info['patients_longitudinal']:,}",
                    })
                static_table(pd.DataFrame(func_data))

        with col_right:
            st.markdown("##### Timed Function Tests")
            timed = clinical.get('timed_tests', {})
            timed_labels = {
                'walk_run_10m': '10m Walk/Run',
                'stair_climb': 'Stair Climb (4 stairs)',
                'rise_from_supine': 'Rise from Supine',
            }
            timed_data = []
            for key, label in timed_labels.items():
                info = timed.get(key, {})
                if isinstance(info, dict):
                    timed_data.append({
                        "Test": label,
                        "Participants": f"{info.get('patients', 0):,}",
                        "Data Points": f"{info.get('data_points', 0):,}",
                    })
            if timed_data:
                static_table(pd.DataFrame(timed_data))

        # --- Row 2: Medications (main focus) ---
        st.markdown("##### Medications & Treatments")
        meds = clinical.get('medications', {})
        if meds:
            col_left, col_right = st.columns(2)

            with col_left:
                # Unified drug class table (advanced therapies + standard classes)
                med_rows = []

                # Advanced therapy categories — aggregate across diseases
                gt_by_disease = meds.get('gene_therapy_by_disease', {})
                cat_totals = {}
                for dinfo in gt_by_disease.values():
                    for t in dinfo.get('treatments', []):
                        cat = t['category']
                        cat_totals[cat] = cat_totals.get(cat, 0) + t.get('patients', 0)
                for cat in ['Antisense', 'Gene Therapy', 'SMN2 Modifier',
                            'Exon Skipping', 'ERT']:
                    if cat in cat_totals:
                        med_rows.append({
                            "Drug Class": cat,
                            "Participants": f"{cat_totals[cat]:,}",
                        })

                # Standard drug classes
                for cls_key, label in [
                    ('disease_modifying_als', 'Disease-Modifying (ALS)'),
                    ('cardiac_meds', 'Cardiac'),
                    ('psych_neuro', 'Psych / Neuro'),
                    ('respiratory_meds', 'Respiratory'),
                ]:
                    cls = meds.get(cls_key, {})
                    if cls and cls.get('patients', 0) > 0:
                        med_rows.append({
                            "Drug Class": label,
                            "Participants": f"{cls['patients']:,}",
                        })
                if med_rows:
                    st.markdown("###### Drug Classes Captured")
                    static_table(pd.DataFrame(med_rows))

            with col_right:
                total_pts = meds.get('total_patients', 0)
                total_recs = meds.get('total_records', 0)
                source = meds.get('source', 'encounter_log_meds')
                source_label = "Combo Drugs table" if source == "combo_drugs" else "Encounter + Log tables"
                st.metric("Total Medication Records", f"{total_recs:,}",
                          help=f"Across {total_pts:,} participants ({source_label})")

                # Top prescribed drugs
                top_meds = meds.get('top_medications', [])
                if top_meds:
                    st.markdown("###### Top Prescribed")
                    top_rows = [
                        {"Drug Name": m['name'], "Patients": f"{m['patients']:,}"}
                        for m in top_meds[:10]
                    ]
                    static_table(pd.DataFrame(top_rows))

        # --- Row 3: Clinical Trials | Pulmonary & Cardiology ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Clinical Trial Participation")
            trials = clinical.get('clinical_trials', {})
            if trials and trials.get('patient_breakdown'):
                trial_rows = []
                for cat, pts in trials['patient_breakdown'].items():
                    trial_rows.append({"Status": cat, "Participants": f"{pts:,}"})
                static_table(pd.DataFrame(trial_rows))
            else:
                st.caption("Clinical trial data not available.")

        with col_right:
            st.markdown("##### Pulmonary & Cardiology")
            pulm = clinical.get('pulmonary', {})
            cardiac = clinical.get('cardiac', {})
            pc_rows = [
                {"Domain": "PFTs Performed", "Participants": f"{pulm.get('pft_performed', 0):,}"},
                {"Domain": "FVC", "Participants": f"{pulm.get('fvc', 0):,}"},
                {"Domain": "FEV1", "Participants": f"{pulm.get('fev1', 0):,}"},
                {"Domain": "ECG", "Participants": f"{cardiac.get('ecg', 0):,}"},
                {"Domain": "Echo", "Participants": f"{cardiac.get('echo', 0):,}"},
                {"Domain": "Cardiomyopathy", "Participants": f"{cardiac.get('cardiomyopathy', 0):,}"},
            ]
            static_table(pd.DataFrame(pc_rows))

        # --- Row 4: Hospitalizations, Surgeries & Devices ---
        st.markdown("##### Hospitalizations, Surgeries & Devices")
        hosp = clinical.get('hospitalizations', {})
        surg = clinical.get('surgeries', {})
        devs = clinical.get('devices', {})
        hd_rows = [
            {"Category": "Hospitalizations", "Participants": f"{hosp.get('patients', 0):,}", "Records": f"{hosp.get('records', 0):,}"},
            {"Category": "Surgeries", "Participants": f"{surg.get('patients', 0):,}", "Records": f"{surg.get('records', 0):,}"},
            {"Category": "Assistive Devices", "Participants": f"{devs.get('assistive_patients', 0):,}", "Records": f"{devs.get('assistive_records', 0):,}"},
            {"Category": "Pulmonary Devices", "Participants": f"{devs.get('pulmonary_patients', 0):,}", "Records": f"{devs.get('pulmonary_records', 0):,}"},
        ]
        static_table(pd.DataFrame(hd_rows))
        st.caption("Combined from encounter and log tables.")

    # Footer
    render_page_footer()


if __name__ == "__main__":
    main()
