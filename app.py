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
import plotly.express as px
from config.settings import *
from api import StatsAPI
from utils.cache import get_cached_snapshot
from components.charts import (
    create_disease_distribution_chart,
    create_site_map,
)
from components.tables import display_disease_table, static_table
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_header, render_page_footer


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

    render_page_header(APP_TITLE, "Interactive Dashboard for MOVR Clinical Data Analytics")

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
            "Total Participants",
            f"{enrollment['total_patients']:,}",
            help="Total MOVR participants with validated enrollment"
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
            help="Total clinic visit records across all participants"
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
        st.markdown("#### Participant Counts by Disease")
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
        data_cutoff = longitudinal.get('data_cutoff_date', 'April 2025')

        # --- Row 1: Hero metrics (6 columns) ---
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric(
                "Total Encounters",
                f"{longitudinal.get('total_encounters', 0):,}",
                help="Total number of clinic visits recorded across all participants in the registry."
            )
        with col2:
            person_yrs = longitudinal.get('total_person_years')
            st.metric(
                "Person-Years",
                f"{person_yrs:,.0f}" if person_yrs is not None else "—",
                help=(
                    "The total observation time accumulated across all participants, measured "
                    "in years.  Calculated by adding up each participant's time from enrollment "
                    "to their last clinic visit.  This is a standard measure of how much "
                    "longitudinal data a registry has collected — higher means more data to "
                    "work with for research and trend analysis."
                )
            )
        with col3:
            mean_dur = longitudinal.get('mean_duration_years')
            std_dur = longitudinal.get('std_duration_years')
            dur_label = "—"
            if mean_dur is not None:
                dur_label = f"{mean_dur} yr"
                if std_dur is not None:
                    dur_label = f"{mean_dur} \u00b1 {std_dur} yr"
            st.metric(
                "Registry Duration",
                dur_label,
                help=(
                    "How long participants have been followed on average, from their "
                    "enrollment date to their most recent clinic visit (\u00b1 standard "
                    "deviation).  A longer duration means the registry captures more of "
                    "each participant's disease course over time."
                )
            )
        with col4:
            ivi = longitudinal.get('inter_visit_interval', {})
            ivi_label = "—"
            if ivi.get('median_months') is not None:
                ivi_label = f"{ivi['median_months']} mo"
            st.metric(
                "Visit Interval",
                ivi_label,
                help=(
                    "The typical time between consecutive clinic visits.  "
                    f"Median: {ivi.get('median_months', '—')} months "
                    f"(IQR: {ivi.get('q1_months', '—')}–{ivi.get('q3_months', '—')} months).  "
                    f"Based on {ivi.get('n_intervals', 0):,} consecutive visit pairs "
                    f"across {ivi.get('n_participants', 0):,} participants with 2+ visits.  "
                    "A shorter interval means more frequent data collection and finer-grained "
                    "longitudinal resolution."
                )
            )
        with col5:
            retention = longitudinal.get('retention', {})
            yr1 = retention.get('year_1', {})
            ret_label = "—"
            if yr1.get('rate_pct') is not None:
                ret_label = f"{yr1['rate_pct']}%"
            st.metric(
                "1-Year Retention",
                ret_label,
                help=(
                    "The percentage of participants who returned for at least one visit "
                    "after being enrolled for one year or more.  "
                    f"Of {yr1.get('eligible', 0):,} participants enrolled 1+ years before "
                    f"the data cutoff ({data_cutoff}), {yr1.get('retained', 0):,} had a "
                    "recorded encounter after their 1-year mark.  Higher retention means "
                    "better long-term follow-up."
                )
            )
        with col6:
            active = longitudinal.get('active_last_12m', 0)
            total_pts = longitudinal.get('patients_with_encounters', 1)
            active_pct = round(100.0 * active / total_pts, 1) if total_pts > 0 else 0
            st.metric(
                "Active (12 mo)",
                f"{active:,}",
                help=(
                    f"{active:,} participants ({active_pct}%) had at least one clinic visit "
                    f"in the 12 months before the data cutoff ({data_cutoff}).  "
                    "This shows how many participants are actively being followed in the "
                    "registry versus those with only historical data."
                )
            )

        # --- Row 2: Additional retention + observation period details ---
        yr2 = retention.get('year_2', {})
        yr3 = retention.get('year_3', {})
        obs_dist = longitudinal.get('observation_period_distribution', [])

        if yr2 or yr3 or obs_dist:
            col_ret, col_obs = st.columns(2)

            with col_ret:
                ret_rows = []
                for yr_key, yr_label in [('year_1', '1 Year'), ('year_2', '2 Years'), ('year_3', '3 Years')]:
                    yr_data = retention.get(yr_key, {})
                    if yr_data:
                        ret_rows.append({
                            "Timepoint": yr_label,
                            "Eligible": f"{yr_data['eligible']:,}",
                            "Retained": f"{yr_data['retained']:,}",
                            "Retention Rate": f"{yr_data['rate_pct']}%",
                        })
                if ret_rows:
                    st.markdown("##### Retention Rates")
                    static_table(pd.DataFrame(ret_rows))
                    st.caption(
                        f"As of data cutoff ({data_cutoff}).  "
                        "Eligible = enrolled at least N years before the cutoff.  "
                        "Retained = had at least one visit after the N-year mark from enrollment."
                    )

            with col_obs:
                if obs_dist:
                    st.markdown("##### Observation Period Distribution")
                    obs_rows = [{"Period": d['label'], "Participants": f"{d['count']:,}"}
                                for d in obs_dist]
                    static_table(pd.DataFrame(obs_rows))
                    med_dur = longitudinal.get('median_duration_years')
                    cap = "How long each participant has been observed (enrollment to last visit)."
                    if med_dur is not None:
                        cap += f"  Median: {med_dur} years."
                    st.caption(cap)

        # --- Row 3: Encounters by disease table ---
        by_disease = longitudinal.get('by_disease', {})
        if by_disease:
            st.markdown("##### By Disease")
            enc_data = []
            for ds in ["ALS", "DMD", "SMA", "LGMD", "FSHD", "BMD", "Pompe"]:
                info = by_disease.get(ds, {})
                if info:
                    # Registry duration ± SD
                    md = info.get('mean_duration_years')
                    sd = info.get('std_duration_years')
                    dur_str = '—'
                    if md is not None:
                        dur_str = f"{md} \u00b1 {sd}" if sd is not None else str(md)

                    # Inter-visit interval
                    ds_ivi = info.get('inter_visit_interval', {})
                    ivi_str = '—'
                    if ds_ivi.get('median_months') is not None:
                        ivi_str = f"{ds_ivi['median_months']}"

                    # Person-years
                    py = info.get('total_person_years')
                    py_str = f"{py:,.0f}" if py is not None else '—'

                    # 1-year retention
                    ds_ret = info.get('retention', {}).get('year_1', {})
                    ret_str = f"{ds_ret['rate_pct']}%" if ds_ret.get('rate_pct') is not None else '—'

                    row = {
                        "Disease": ds,
                        "N": f"{info['patients']:,}",
                        "Encounters": f"{info['encounters']:,}",
                        "Person-Yrs": py_str,
                        "Duration (yr)": dur_str,
                        "Visit Interval (mo)": ivi_str,
                        "1-Yr Retention": ret_str,
                        "3+ Visits": f"{info['patients_3plus']:,}",
                    }
                    enc_data.append(row)
            if enc_data:
                static_table(pd.DataFrame(enc_data))

        st.caption(
            f"Study period: {STUDY_START} to {STUDY_END} ({STUDY_STATUS}).  "
            f"Data cutoff: {data_cutoff}.  "
            "Visit Interval = median time between consecutive clinic visits per participant.  "
            "Duration = mean time from enrollment to last visit (\u00b1 SD)."
        )

    # Enrollment Over Time
    timeline = snapshot.get('enrollment_timeline', {})
    by_dm = timeline.get('by_disease_month', [])
    if by_dm:
        st.markdown("---")
        st.subheader("Enrollment Over Time")

        tl_df = pd.DataFrame(by_dm)
        tl_df['month'] = pd.to_datetime(tl_df['month'])
        tl_df = tl_df.sort_values(['disease', 'month'])

        col_cumul, col_monthly = st.columns(2)

        with col_cumul:
            cumul_df = tl_df.copy()
            cumul_df['cumulative'] = cumul_df.groupby('disease')['count'].cumsum()
            fig_cumul = px.line(
                cumul_df,
                x='month',
                y='cumulative',
                color='disease',
                title='Cumulative Enrollment',
                labels={'month': '', 'cumulative': 'Participants', 'disease': 'Disease'},
            )
            fig_cumul.update_layout(height=400, legend=dict(orientation='h', y=-0.15))
            st.plotly_chart(fig_cumul, use_container_width=True)

        with col_monthly:
            # Monthly new enrollments aggregated across diseases
            monthly_total = tl_df.groupby('month', as_index=False)['count'].sum()
            fig_monthly = px.bar(
                monthly_total,
                x='month',
                y='count',
                title='Monthly New Enrollments (All Diseases)',
                labels={'month': '', 'count': 'New Participants'},
                color_discrete_sequence=['#1E88E5'],
            )
            fig_monthly.update_layout(height=400)
            st.plotly_chart(fig_monthly, use_container_width=True)

        notes = []
        missing = timeline.get('missing_date_count', 0)
        clamped = timeline.get('pre_study_clamped', 0)
        if missing > 0:
            notes.append(f"{missing} participants with missing enrollment dates defaulted to first encounter or study start (Nov 2018)")
        if clamped > 0:
            notes.append(f"{clamped} pre-study enrollment dates clamped to first encounter or study start")
        notes.append("Enrollment dates are under active QC review")
        st.caption(" | ".join(notes))

    # Clinical Data Highlights (simplified — per feedback)
    st.markdown("---")
    st.subheader("Clinical Data Highlights")
    st.caption(
        "A snapshot of the breadth and depth of clinical data captured across "
        "all seven diseases.  Explore disease-specific analytics in the "
        "**Disease Explorer** and browse all 1,024 fields in the **Data Dictionary**."
    )

    clinical = snapshot.get('clinical_availability', {})
    if clinical:
        meds = clinical.get('medications', {})
        hosp = clinical.get('hospitalizations', {})
        surg = clinical.get('surgeries', {})
        devs = clinical.get('devices', {})

        # Helper: sort rows by participant count descending
        def _sort_by_participants(rows):
            return sorted(rows, key=lambda r: int(r['Participants'].replace(',', '')), reverse=True)

        # --- Row 1: Medication Highlights | Clinical Record Counts ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### Medication Highlights")
            med_rows = []

            # Disease-modifying (ALS)
            dm_als = meds.get('disease_modifying_als', {})
            als_total = dm_als.get('patients', 0)
            if als_total > 0:
                med_rows.append({
                    "Category": "Disease-Modifying (ALS)",
                    "Participants": f"{als_total:,}",
                    "Records": f"{dm_als.get('records', 0):,}",
                })

            # Gene therapy / advanced therapies (SMA, DMD, ALS, Pompe)
            gt_by_disease = meds.get('gene_therapy_by_disease', {})
            gt_total_p = sum(gt_by_disease.get(ds, {}).get('total_patients', 0)
                            for ds in gt_by_disease)
            gt_total_r = sum(
                sum(t.get('records', 0) for t in gt_by_disease.get(ds, {}).get('treatments', []))
                for ds in gt_by_disease
            )
            if gt_total_p > 0:
                med_rows.append({
                    "Category": "Gene / Advanced Therapy",
                    "Participants": f"{gt_total_p:,}",
                    "Records": f"{gt_total_r:,}",
                })

            # Glucocorticoids (DMD)
            gluc = meds.get('glucocorticoid_encounter', {})
            gluc_total = gluc.get('patients', 0)
            if gluc_total > 0:
                med_rows.append({
                    "Category": "Glucocorticoids (DMD)",
                    "Participants": f"{gluc_total:,}",
                    "Records": f"{gluc.get('data_points', 0):,}",
                })

            # Cardiac medications
            cardiac = meds.get('cardiac_meds', {})
            cardiac_total = cardiac.get('patients', 0)
            if cardiac_total > 0:
                med_rows.append({
                    "Category": "Cardiac Medications",
                    "Participants": f"{cardiac_total:,}",
                    "Records": f"{cardiac.get('records', 0):,}",
                })

            # Psych / neuro
            psych = meds.get('psych_neuro', {})
            psych_total = psych.get('patients', 0)
            if psych_total > 0:
                med_rows.append({
                    "Category": "Psych / Neuro",
                    "Participants": f"{psych_total:,}",
                    "Records": f"{psych.get('records', 0):,}",
                })

            # Respiratory medications
            resp = meds.get('respiratory_meds', {})
            resp_total = resp.get('patients', 0)
            if resp_total > 0:
                med_rows.append({
                    "Category": "Respiratory Medications",
                    "Participants": f"{resp_total:,}",
                    "Records": f"{resp.get('records', 0):,}",
                })

            if med_rows:
                static_table(pd.DataFrame(_sort_by_participants(med_rows)))

            # List drugs included in each category
            drug_lines = []
            als_drugs = list(dm_als.get('drugs', {}).keys())
            if als_drugs:
                drug_lines.append(f"Disease-Modifying (ALS): {', '.join(als_drugs)}")

            # Gene / Advanced therapy per disease
            for ds in sorted(gt_by_disease.keys()):
                ds_treats = gt_by_disease[ds].get('treatments', [])
                if ds_treats:
                    names = [t.get('label', '') for t in ds_treats if t.get('label')]
                    if names:
                        drug_lines.append(f"Gene / Advanced ({ds}): {', '.join(names)}")

            # Glucocorticoids come from encounter eCRF, not combo drugs table
            gluc_vals = gluc.get('values', {})
            gluc_drugs = [v.replace('Yes ', '') for v in gluc_vals if v.startswith('Yes ')]
            if gluc_drugs:
                drug_lines.append(f"Glucocorticoids (DMD): {', '.join(gluc_drugs)} (encounter eCRF)")

            cardiac_drugs = list(cardiac.get('drugs', {}).keys())
            if cardiac_drugs:
                drug_lines.append(f"Cardiac: {', '.join(cardiac_drugs)}")

            psych_drugs = list(psych.get('drugs', {}).keys())
            if psych_drugs:
                drug_lines.append(f"Psych / Neuro: {', '.join(psych_drugs)}")

            resp_drugs = list(resp.get('drugs', {}).keys())
            if resp_drugs:
                drug_lines.append(f"Respiratory: {', '.join(resp_drugs)}")

            if drug_lines:
                st.caption("  \n".join(drug_lines))

        with col_right:
            st.markdown("##### Clinical Record Counts")
            record_rows = [
                {
                    "Category": "Medications",
                    "Participants": f"{meds.get('total_patients', 0):,}",
                    "Records": f"{meds.get('total_records', 0):,}",
                },
                {
                    "Category": "Hospitalizations",
                    "Participants": f"{hosp.get('patients', 0):,}",
                    "Records": f"{hosp.get('records', 0):,}",
                },
                {
                    "Category": "Surgeries",
                    "Participants": f"{surg.get('patients', 0):,}",
                    "Records": f"{surg.get('records', 0):,}",
                },
                {
                    "Category": "Assistive Devices",
                    "Participants": f"{devs.get('assistive_patients', 0):,}",
                    "Records": f"{devs.get('assistive_records', 0):,}",
                },
                {
                    "Category": "Pulmonary Devices",
                    "Participants": f"{devs.get('pulmonary_patients', 0):,}",
                    "Records": f"{devs.get('pulmonary_records', 0):,}",
                },
            ]
            static_table(pd.DataFrame(_sort_by_participants(record_rows)))
            st.caption("Across all seven diseases. See disease-specific breakdowns in the Disease Explorer.")

        # --- Genetics Testing ---
        st.markdown("")
        st.markdown("##### Genetic Testing")

        # Pull genetic confirmation data from disease_profiles
        profiles = snapshot.get('disease_profiles', {})
        _GENETICS_MAP = {
            "DMD":  ("dmdgntcf", "Mutation type, exon deletions/duplications, frame type, dystrophin deficiency"),
            "BMD":  ("bmdgntcf", "Dystrophin deficiency, DNA dosage analysis (MLPA, CGH, Southern Blot)"),
            "SMA":  ("smadgcnf", "SMN1/SMN2 copy number"),
            "LGMD": ("lggntcf", "27 subtypes (LGMD2A, 2I, 2B, ...), muscle biopsy"),
            "ALS":  ("genemut", "C9ORF72, SOD1, FUS, TDP-43/TARDBP"),
            "FSHD": ("fshdel", "4q35 deletion, SMCHD1 mutation"),
            "Pompe": ("pomgntcf", "GAA enzyme activity, up to 3 variants"),
        }

        genetics_rows = []
        for disease in ["DMD", "SMA", "LGMD", "ALS", "BMD", "FSHD", "Pompe"]:
            field_name, description = _GENETICS_MAP.get(disease, (None, ""))
            if not field_name:
                continue
            diag_list = profiles.get(disease, {}).get('diagnosis', [])
            total_records = 0
            for entry in diag_list:
                if entry.get('field') == field_name:
                    vals = entry.get('values', [])
                    total_records = sum(v.get('count', 0) for v in vals)
                    break
            if total_records > 0:
                genetics_rows.append({
                    "Disease": disease,
                    "Records": f"{total_records:,}",
                    "Data Captured": description,
                })

        if genetics_rows:
            genetics_rows.sort(key=lambda r: int(r['Records'].replace(',', '')), reverse=True)
            static_table(pd.DataFrame(genetics_rows))
            st.caption(
                "Genetic confirmation status and related molecular data from the diagnosis eCRF. "
                "110 genetics-related fields across all diseases. "
                "See disease-specific Clinical Summaries for detailed breakdowns."
            )
        else:
            st.caption("Genetic testing data available in the Disease Explorer and Clinical Summaries.")

        # --- Additional Clinical Domains ---
        st.markdown("")
        st.markdown("##### Additional Clinical Domains")

        func = clinical.get('functional_scores', {})
        pulm = clinical.get('pulmonary', {})
        card = clinical.get('cardiac', {})
        mob = clinical.get('mobility', {})
        vitals = clinical.get('vital_signs', {})
        nutr = clinical.get('nutrition', {})
        ortho = clinical.get('orthopedic', {})
        trials = clinical.get('clinical_trials', {})
        care = clinical.get('care', {})
        spin = clinical.get('spinraza', {})

        domain_rows = []

        # Functional scores
        func_pts = max(v.get('patients', 0) for v in func.values()) if func else 0
        func_labels = [v.get('label', k) for k, v in func.items() if v.get('patients', 0) > 0]
        if func_pts > 0:
            domain_rows.append({
                "Domain": "Functional Scores",
                "Participants": f"{func_pts:,}",
                "Includes": ", ".join(func_labels),
            })

        # Care coordination
        care_pts = max(care.get('specialists_seen', 0), care.get('specialists_referred', 0),
                       care.get('multidisciplinary_plan', 0))
        if care_pts > 0:
            domain_rows.append({
                "Domain": "Care Coordination",
                "Participants": f"{care_pts:,}",
                "Includes": "Specialists seen/referred, multidisciplinary care plans",
            })

        # Pulmonary function
        pulm_pts = pulm.get('pft_performed', 0)
        if pulm_pts > 0:
            domain_rows.append({
                "Domain": "Pulmonary Function",
                "Participants": f"{pulm_pts:,}",
                "Includes": "PFTs, FVC, FEV1",
            })

        # Nutrition
        nutr_pts = max(nutr.get('nutritional_supplementation', 0),
                       nutr.get('feeding_method', 0))
        if nutr_pts > 0:
            domain_rows.append({
                "Domain": "Nutrition",
                "Participants": f"{nutr_pts:,}",
                "Includes": "Nutritional supplementation, feeding methods, feeding tubes",
            })

        # Clinical trials
        trial_pts = trials.get('patients', 0)
        if trial_pts > 0:
            domain_rows.append({
                "Domain": "Clinical Trial Participation",
                "Participants": f"{trial_pts:,}",
                "Includes": "Trial enrollment status per encounter",
            })

        # Vital signs
        vitals_pts = max(vitals.get('height', 0), vitals.get('weight', 0))
        if vitals_pts > 0:
            domain_rows.append({
                "Domain": "Vital Signs",
                "Participants": f"{vitals_pts:,}",
                "Includes": "Height, weight, BMI",
            })

        # Mobility
        mob_pts = max(mob.get('ambulatory_status', 0), mob.get('assistive_device', 0))
        if mob_pts > 0:
            domain_rows.append({
                "Domain": "Mobility / Ambulation",
                "Participants": f"{mob_pts:,}",
                "Includes": "Ambulatory status, wheelchair use, assistive devices, falls",
            })

        # Orthopedic
        ortho_pts = max(ortho.get('scoliosis', 0), ortho.get('spinal_xray', 0))
        if ortho_pts > 0:
            domain_rows.append({
                "Domain": "Orthopedic",
                "Participants": f"{ortho_pts:,}",
                "Includes": "Scoliosis screening, spinal x-ray",
            })

        # Cardiac
        card_pts = max(card.get('ecg', 0), card.get('echo', 0), card.get('cardiomyopathy', 0))
        if card_pts > 0:
            domain_rows.append({
                "Domain": "Cardiac",
                "Participants": f"{card_pts:,}",
                "Includes": "ECG, echocardiogram, cardiomyopathy screening",
            })

        # Spinraza eCRF
        spin_pts = spin.get('patients_with_maintenance', 0) + spin.get('patients_eap', 0)
        if spin_pts > 0:
            domain_rows.append({
                "Domain": "Spinraza eCRF (SMA)",
                "Participants": f"{spin_pts:,}",
                "Includes": f"{spin.get('ecrf_fields', 0)} fields, maintenance dosing, EAP",
            })

        if domain_rows:
            domain_rows.sort(key=lambda r: int(r['Participants'].replace(',', '')), reverse=True)
            static_table(pd.DataFrame(domain_rows))
            st.caption(
                "Additional clinical domains captured across the registry. "
                "19 canonical domains, 1,024 fields total. "
                "Browse all fields in the Data Dictionary."
            )

    # Footer
    render_page_footer()


if __name__ == "__main__":
    main()
