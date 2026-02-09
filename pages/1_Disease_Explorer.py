"""
Disease Explorer Page

Explore disease-specific cohorts with cascading filters and interactive charts.
Supports both live mode (parquet files) and snapshot mode (pre-computed JSON).
"""

import sys
from pathlib import Path

# Add both webapp and parent directory to path FIRST (before other imports)
app_dir = Path(__file__).parent.parent
# App root is parent of pages/
sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from api import StatsAPI
from components.charts import (
    create_categorical_bar_chart,
    create_categorical_donut_chart,
    create_numeric_histogram_chart,
    create_facility_distribution_mini_chart,
)
from components.tables import display_cohort_summary, static_table
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer
from utils.cache import get_cached_snapshot
from config.settings import PAGE_ICON, DISEASE_DISPLAY_ORDER, COLOR_SCHEMES

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

# Page configuration
st.set_page_config(
    page_title="Disease Explorer - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide"
)

inject_global_css()


def _unavailable_section(title, detail=None):
    """Render a styled unavailable section placeholder."""
    msg = detail or "This section requires a live data connection and is not available in snapshot mode."
    st.markdown(
        f"""<div style='background-color: #f8f9fa; border: 1px dashed #ccc; border-radius: 8px;
        padding: 2rem; text-align: center; color: #888; margin: 1rem 0;'>
        <strong>{title}</strong><br>
        <span style='font-size: 0.9em;'>{msg}</span>
        </div>""",
        unsafe_allow_html=True,
    )


from components.deep_dive import render_dmd_deep_dive, render_lgmd_deep_dive

# ---------------------------------------------------------------------------
# Deep-dive renderer registry
# Each disease with a deep-dive gets a function here.
# To add a new disease: create render_{disease}_deep_dive() in components/deep_dive.py
# ---------------------------------------------------------------------------
_DEEP_DIVE_RENDERERS = {
    'DMD': render_dmd_deep_dive,
    'LGMD': render_lgmd_deep_dive,
    # 'ALS': render_als_deep_dive,   # planned
    # 'SMA': render_sma_deep_dive,   # planned
}


# Header with branding
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("Disease Explorer")
    st.markdown("### Explore disease-specific cohorts and patient data")

with header_right:
    st.markdown(
        """
        <div style='text-align: right; padding-top: 10px;'>
            <span style='font-size: 1.5em; font-weight: bold; color: #1E88E5;'>OpenMOVR App</span><br>
            <span style='font-size: 0.9em; color: #666; background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px;'>
                Gen1 | v0.2.0
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Detect data mode: live (parquet) vs snapshot
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_has_parquet = any(_DATA_DIR.glob("*.parquet")) if _DATA_DIR.exists() else False

# ---------------------------------------------------------------------------
# Load filter config (shared with DiseaseFilterRenderer)
# ---------------------------------------------------------------------------
from components.filters import load_filter_config as _load_filter_config


# ===================================================================
# SNAPSHOT MODE
# ===================================================================
if not _has_parquet:
    # Load snapshot data
    try:
        snapshot = StatsAPI.load_snapshot()
        diseases = snapshot['disease_distribution']['diseases']
        disease_summary = snapshot['disease_distribution'].get('disease_summary', {})
    except Exception as e:
        st.error(f"Error loading snapshot: {e}")
        st.stop()

    # Prototype banner
    st.warning(
        "**Snapshot Mode (Prototype)** — This page displays pre-computed summary statistics only. "
        "Interactive filters, patient-level data tables, and dynamic charts require a live data connection. "
        "To request access to the full dataset, use the [MDA Data Request Form](https://mdausa.tfaforms.net/389761)."
    )

    # Sidebar — disease selector only (no filters in snapshot mode)
    disease_names = [d for d in DISEASE_DISPLAY_ORDER if d in disease_summary]
    disease_names.extend([d for d in disease_summary if d not in disease_names])

    with st.sidebar:
        st.header("Filters")
        selected_disease = st.selectbox(
            "Select Disease",
            options=disease_names,
            key="disease_select_explorer"
        )
        st.markdown("---")
        st.info("Advanced filters are disabled in snapshot mode.")
        render_sidebar_footer()

    # Find selected disease data
    disease_info = next(
        (d for d in diseases if d['disease'] == selected_disease),
        None
    )
    summary_entry = disease_summary.get(selected_disease, {})

    # ===================================================================
    # METRICS ROW (from snapshot) — always visible above tabs
    # ===================================================================
    st.markdown("---")
    st.subheader(f"{selected_disease} Cohort Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        count = disease_info['patient_count'] if disease_info else summary_entry.get('count', 0)
        st.metric(
            "Total Patients",
            f"{count:,}",
            help=f"Number of {selected_disease} patients in the registry",
        )

    with col2:
        pct = disease_info['percentage'] if disease_info else summary_entry.get('percentage', 0)
        st.metric(
            "% of Registry",
            f"{pct:.1f}%",
            help="Percentage of total MOVR patients",
        )

    with col3:
        total = snapshot['enrollment']['total_patients']
        st.metric("Total Registry", f"{total:,}", help="Total patients across all diseases")

    with col4:
        st.metric(
            "Disease Types",
            len(diseases),
            help="Number of disease types in the database",
        )

    # ===================================================================
    # DISEASE DISTRIBUTION (from snapshot) — always visible above tabs
    # ===================================================================
    st.markdown("---")
    st.subheader("Disease Distribution Overview")

    diseases_df = pd.DataFrame(diseases)

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = px.bar(
            diseases_df.sort_values('patient_count', ascending=True),
            x='patient_count',
            y='disease',
            orientation='h',
            title='Patient Count by Disease',
            labels={'patient_count': 'Patients', 'disease': 'Disease'},
            color='patient_count',
            color_continuous_scale='Blues',
        )
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Patient Counts**")
        table_df = diseases_df[['disease', 'patient_count', 'percentage']].copy()
        table_df.columns = ['Disease', 'Patients', '%']
        table_df['%'] = table_df['%'].apply(lambda x: f"{x:.1f}%")
        static_table(table_df)

    # ===================================================================
    # TABS: Demographics | Diagnosis | Deep Dive (alpha) | Data Summary
    # ===================================================================
    st.markdown("---")
    tab_demo, tab_diag, tab_deep, tab_data = st.tabs([
        "Demographics", "Diagnosis", "Deep Dive (Alpha)", "Data Summary"
    ])

    # --- Tab 1: Demographics ---
    with tab_demo:
        disease_profiles = snapshot.get('disease_profiles', {})
        profile = disease_profiles.get(selected_disease, {})
        demo_snap = profile.get('demographics', {})

        st.subheader("Demographics Overview")

        if demo_snap:
            # Row 1: Age at Enrollment + Age at Diagnosis + Gender
            col_age_enrl, col_age_dx, col_gender = st.columns(3)

            with col_age_enrl:
                age_data = demo_snap.get('age_at_enrollment', [])
                if age_data:
                    fig = go.Figure(go.Bar(
                        x=[d['label'] for d in age_data],
                        y=[d['count'] for d in age_data],
                        marker_color='#636EFA',
                    ))
                    fig.update_layout(
                        title="Age at Enrollment",
                        xaxis_title="Age Range",
                        yaxis_title="Patients",
                        height=350,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Age at enrollment not available.")

            with col_age_dx:
                dx_age_data = demo_snap.get('age_at_diagnosis', [])
                if dx_age_data:
                    fig = go.Figure(go.Bar(
                        x=[d['label'] for d in dx_age_data],
                        y=[d['count'] for d in dx_age_data],
                        marker_color='#FFA726',
                    ))
                    fig.update_layout(
                        title="Age at Diagnosis",
                        xaxis_title="Age Range",
                        yaxis_title="Patients",
                        height=350,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Age at diagnosis not available for this disease.")

            with col_gender:
                gender_data = demo_snap.get('gender', [])
                if gender_data:
                    fig = go.Figure(go.Pie(
                        labels=[d['label'] for d in gender_data],
                        values=[d['count'] for d in gender_data],
                        hole=0.4,
                    ))
                    fig.update_layout(
                        title="Gender Distribution",
                        height=350,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Gender data not available.")

            eth_data = demo_snap.get('ethnicity', [])
            if eth_data:
                labels = [d['label'] for d in eth_data]
                values = [d['count'] for d in eth_data]
                fig = go.Figure(go.Bar(
                    x=values, y=labels,
                    orientation='h',
                    marker_color='#AB63FA',
                ))
                fig.update_layout(
                    title="Race / Ethnicity",
                    xaxis_title="Participants",
                    height=max(250, len(labels) * 30 + 80),
                    margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("*Categories with fewer than 11 participants are grouped per HIPAA small-cell suppression guidelines.*")

            # Row 3: Health Insurance + Education/Employment
            ins_data = demo_snap.get('health_insurance', [])
            edu_data = demo_snap.get('education_level', [])
            emp_data = demo_snap.get('employment_status', [])

            if ins_data or edu_data or emp_data:
                col_ins, col_edu = st.columns(2)

                with col_ins:
                    if ins_data:
                        labels = [d['label'][:50] for d in ins_data]
                        values = [d['count'] for d in ins_data]
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation='h',
                            marker_color='#FF7043',
                        ))
                        fig.update_layout(
                            title="Health Insurance",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=250),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Participants may report multiple insurance types.")

                with col_edu:
                    if edu_data:
                        labels = [d['label'][:40] for d in edu_data]
                        values = [d['count'] for d in edu_data]
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation='h',
                            marker_color='#26A69A',
                        ))
                        fig.update_layout(
                            title="Education Level",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    elif emp_data:
                        labels = [d['label'][:40] for d in emp_data]
                        values = [d['count'] for d in emp_data]
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation='h',
                            marker_color='#26A69A',
                        ))
                        fig.update_layout(
                            title="Employment Status",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            _unavailable_section("Demographics Charts", "Demographic data not available for this disease.")

    # --- Tab 2: Diagnosis ---
    with tab_diag:
        profile = snapshot.get('disease_profiles', {}).get(selected_disease, {})
        diag_snap = profile.get('diagnosis', [])

        if diag_snap:
            st.subheader(f"{selected_disease} Diagnosis Profile")

            for i in range(0, len(diag_snap), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(diag_snap):
                        break
                    dx = diag_snap[idx]
                    with col:
                        if dx['type'] == 'categorical':
                            vals = dx['values']
                            labels = [d['label'][:50] for d in vals]
                            counts = [d['count'] for d in vals]
                            fig = go.Figure(go.Bar(
                                x=counts, y=labels,
                                orientation='h',
                                marker_color='#00CC96',
                            ))
                            fig.update_layout(
                                title=dx['label'],
                                xaxis_title="Patients",
                                height=max(250, len(labels) * 25 + 80),
                                margin=dict(t=40, b=40, l=200),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif dx['type'] == 'numeric':
                            st.markdown(f"**{dx['label']}**")
                            mc1, mc2, mc3 = st.columns(3)
                            with mc1:
                                st.metric("Mean", f"{dx['mean']:.1f}")
                            with mc2:
                                st.metric("Median", f"{dx['median']:.1f}")
                            with mc3:
                                st.metric("N", f"{dx['n']:,}")
        else:
            st.subheader(f"{selected_disease} Diagnosis Profile")
            _unavailable_section("Diagnosis Profile", "Diagnosis data not available for this disease.")

    # --- Tab 3: Deep Dive (Alpha) ---
    with tab_deep:
        renderer = _DEEP_DIVE_RENDERERS.get(selected_disease)
        if renderer:
            st.info(
                "**Alpha Preview** — Deep dive analytics rendered from pre-computed snapshots. "
                "Full interactive deep dives with data tables are available in the "
                "DUA-gated pages (DMD Deep Dive, LGMD Deep Dive)."
            )
            renderer()
        else:
            _disease_placeholders = {
                "ALS": (
                    "A clinical deep dive for ALS is in development. "
                    "Upcoming features include ALSFRS-R longitudinal tracking, "
                    "respiratory function (FVC) trends, loss of ambulation analysis, "
                    "and therapeutic utilization."
                ),
                "SMA": (
                    "A clinical deep dive for SMA is in development. "
                    "Upcoming features include HFMSE/CHOP-INTEND longitudinal tracking, "
                    "respiratory function trends, and therapeutic utilization "
                    "(Spinraza, Zolgensma, Evrysdi)."
                ),
                "BMD": (
                    "A clinical deep dive for BMD is in development. "
                    "Upcoming features include functional outcome tracking, "
                    "cardiac monitoring, and therapeutic utilization."
                ),
                "FSHD": (
                    "A clinical deep dive for FSHD is in development. "
                    "Upcoming features include CSS/Reachability scores, "
                    "respiratory function trends, and genetic characterization."
                ),
                "POM": (
                    "A clinical deep dive for Pompe disease is in development. "
                    "Upcoming features include ERT utilization, respiratory and "
                    "motor function tracking, and longitudinal outcomes."
                ),
            }
            placeholder_msg = _disease_placeholders.get(
                selected_disease,
                f"A clinical deep dive for {selected_disease} has not been built yet. "
                "This feature is under active development.",
            )
            _unavailable_section(f"{selected_disease} Deep Dive", placeholder_msg)

    # --- Tab 4: Data Summary ---
    with tab_data:
        st.subheader("Data Summary")

        # Show what's available from the snapshot for this disease
        disease_profiles = snapshot.get('disease_profiles', {})
        profile = disease_profiles.get(selected_disease, {})
        demo_snap = profile.get('demographics', {})
        diag_snap = profile.get('diagnosis', [])

        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            count = disease_info['patient_count'] if disease_info else 0
            st.metric("Patients", f"{count:,}")
        with col_s2:
            n_demo_fields = len(demo_snap) if isinstance(demo_snap, dict) else 0
            n_diag_fields = len(diag_snap) if isinstance(diag_snap, list) else 0
            st.metric("Profile Sections", n_demo_fields + n_diag_fields)
        with col_s3:
            has_deep_dive = selected_disease in _DEEP_DIVE_RENDERERS
            st.metric("Deep Dive", "Available" if has_deep_dive else "Planned")

        st.markdown("---")
        st.markdown("#### Available Data")

        avail_rows = []
        if demo_snap:
            for key in demo_snap:
                data = demo_snap[key]
                if isinstance(data, list) and data:
                    n = sum(d.get('count', 0) for d in data)
                    avail_rows.append({"Domain": "Demographics", "Section": key.replace("_", " ").title(), "Records": n})
        if diag_snap:
            for dx in diag_snap:
                label = dx.get('label', 'Unknown')
                if dx.get('type') == 'categorical':
                    n = sum(d.get('count', 0) for d in dx.get('values', []))
                elif dx.get('type') == 'numeric':
                    n = dx.get('n', 0)
                else:
                    n = 0
                avail_rows.append({"Domain": "Diagnosis", "Section": label, "Records": n})

        if avail_rows:
            static_table(pd.DataFrame(avail_rows))
        else:
            st.caption("No profile data available for this disease in the snapshot.")

        st.caption(
            "Full data tables and CSV downloads are available in the "
            "**Download Center** and **Deep Dive** pages (provisioned access required)."
        )


# ===================================================================
# LIVE MODE (parquet files available)
# ===================================================================
else:
    from api import CohortAPI
    from components.filters import disease_selector, include_usndr_toggle, DiseaseFilterRenderer
    from utils.cache import get_cached_disease_cohort, get_cached_base_cohort

    # ---------------------------------------------------------------------------
    # Sidebar — disease selector + USNDR toggle + cascading filters
    # ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("Filters")

        available_diseases = CohortAPI.get_available_diseases()
        sorted_diseases = [d for d in DISEASE_DISPLAY_ORDER if d in available_diseases]
        sorted_diseases.extend([d for d in available_diseases if d not in sorted_diseases])

        selected_disease = disease_selector(
            sorted_diseases,
            label="Select Disease",
            key="disease_select_explorer"
        )

        st.markdown("---")

        include_usndr = include_usndr_toggle(
            label="Include USNDR Legacy Patients",
            default=False,
            key="usndr_toggle_explorer"
        )

    # Load disease cohort
    with st.spinner(f"Loading {selected_disease} cohort..."):
        try:
            base_cohort = get_cached_base_cohort(include_usndr=include_usndr)
            disease_cohort = get_cached_disease_cohort(selected_disease)
        except Exception as e:
            st.error(f"Error loading {selected_disease} cohort: {e}")
            st.stop()

    # Render cascading filters in sidebar (after cohort is loaded)
    with st.sidebar:
        st.markdown("---")
        renderer = DiseaseFilterRenderer()
        active_filters = renderer.render_filters(
            disease=selected_disease,
            cohort=disease_cohort,
        )

        # Contact info at bottom
        render_sidebar_footer()

    # Apply filters
    if active_filters:
        with st.spinner("Applying filters..."):
            filtered_cohort = CohortAPI.apply_filters(disease_cohort, active_filters)
    else:
        filtered_cohort = disease_cohort

    # Summaries
    cohort_summary = CohortAPI.get_cohort_summary(filtered_cohort)
    total_unfiltered = disease_cohort.get("count", 0)
    total_filtered = cohort_summary["total_patients"]


    # ===================================================================
    # METRICS ROW — always visible above tabs
    # ===================================================================
    st.markdown("---")
    st.subheader(f"{selected_disease} Cohort Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = None
        if active_filters:
            diff = total_filtered - total_unfiltered
            delta = f"{diff:,}" if diff != 0 else None
        st.metric(
            "Total Patients",
            f"{total_filtered:,}",
            delta=delta,
            help=f"Number of {selected_disease} patients" + (" (filtered)" if active_filters else ""),
        )

    with col2:
        st.metric("Facilities", cohort_summary['facility_count'],
                  help="Facilities with this disease")

    with col3:
        st.metric("Encounters", f"{cohort_summary['encounter_records']:,}",
                  help="Total encounter records")

    with col4:
        st.metric("Medications", f"{cohort_summary['medication_records']:,}",
                  help="Total medication records")

    if active_filters:
        st.info(
            f"Showing **{total_filtered:,}** of **{total_unfiltered:,}** "
            f"{selected_disease} patients ({len(active_filters)} filter(s) active)"
        )

    # ===================================================================
    # TABS: Demographics | Diagnosis | Deep Dive | Data Summary
    # Tab order: Demographics | [Clinical Summary - future] | Diagnosis | Deep Dive | Data Summary
    # ===================================================================
    st.markdown("---")
    tab_demo, tab_diag, tab_deep, tab_data = st.tabs([
        "Demographics", "Diagnosis", "Deep Dive", "Data Summary"
    ])

    # --- Tab 1: Demographics ---
    with tab_demo:
        st.subheader("Demographics Overview")

        demo_df = filtered_cohort.get("demographics", pd.DataFrame())

        # Disease → diagnosis-age field mapping (in diagnosis table)
        _DIAG_AGE_FIELDS = {
            'ALS': 'alsdgnag',
            'DMD': 'dmddgnag',
            'BMD': 'bmddgnag',
            'SMA': 'smadgnag',
            'LGMD': 'lgdgag',
            'FSHD': 'fshdgnag',
        }

        if not demo_df.empty:
            # Row 1: Age at Enrollment + Age at Diagnosis + Gender
            col_age_enrl, col_age_dx, col_gender = st.columns(3)

            with col_age_enrl:
                # Compute age at enrollment from DOB + enrollment date
                if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
                    try:
                        dob = pd.to_datetime(demo_df["dob"], errors="coerce")
                        enrol = pd.to_datetime(demo_df["enroldt"], errors="coerce")
                        enrl_age = ((enrol - dob).dt.days / 365.25).dropna()
                        enrl_age = enrl_age[(enrl_age >= 0) & (enrl_age <= 110)]
                        if len(enrl_age) >= 2:
                            bins = [0, 5, 10, 18, 30, 40, 50, 60, 70, 80, 110]
                            bin_labels = ['0-4', '5-9', '10-17', '18-29', '30-39',
                                          '40-49', '50-59', '60-69', '70-79', '80+']
                            cut = pd.cut(enrl_age, bins=bins, labels=bin_labels, include_lowest=True)
                            vc = cut.value_counts().sort_index()
                            fig = go.Figure(go.Bar(
                                x=vc.index.astype(str), y=vc.values,
                                marker_color='#636EFA',
                            ))
                            fig.update_layout(
                                title=f"Age at Enrollment (median {enrl_age.median():.0f})",
                                xaxis_title="Age Range",
                                yaxis_title="Patients",
                                height=350,
                                margin=dict(t=40, b=40),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.caption("Age at enrollment: insufficient data")
                    except Exception:
                        st.caption("Age at enrollment: error computing")
                else:
                    st.caption("Age at enrollment: DOB or enrollment date not available")

            with col_age_dx:
                diag_age_field = _DIAG_AGE_FIELDS.get(selected_disease)
                diag_df = filtered_cohort.get("diagnosis", pd.DataFrame())
                if (diag_age_field and isinstance(diag_df, pd.DataFrame)
                        and not diag_df.empty and diag_age_field in diag_df.columns):
                    ages = pd.to_numeric(diag_df[diag_age_field], errors="coerce").dropna()
                    ages = ages[(ages >= 0) & (ages <= 110)]
                    if len(ages) >= 2:
                        bins = [0, 5, 10, 18, 30, 40, 50, 60, 70, 80, 110]
                        bin_labels = ['0-4', '5-9', '10-17', '18-29', '30-39',
                                      '40-49', '50-59', '60-69', '70-79', '80+']
                        cut = pd.cut(ages, bins=bins, labels=bin_labels, include_lowest=True)
                        vc = cut.value_counts().sort_index()
                        fig = go.Figure(go.Bar(
                            x=vc.index.astype(str), y=vc.values,
                            marker_color='#FFA726',
                        ))
                        fig.update_layout(
                            title=f"Age at Diagnosis (median {ages.median():.0f})",
                            xaxis_title="Age Range",
                            yaxis_title="Patients",
                            height=350,
                            margin=dict(t=40, b=40),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Age at diagnosis: insufficient data")
                else:
                    st.caption("Age at diagnosis not available for this disease.")

            with col_gender:
                if "gender" in demo_df.columns:
                    fig = create_categorical_donut_chart(
                        demo_df["gender"], title="Gender Distribution"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption("Gender: no data available")
                else:
                    st.caption("Gender field not found")

            # Row 2: Race/Ethnicity + Facility
            col_eth, col_fac = st.columns(2)

            with col_eth:
                if "ethnic" in demo_df.columns:
                    # Clean multi-select: single → keep, multiple → Multiracial
                    _SMALL_CELL = 11  # HIPAA small-cell threshold

                    def _clean_race(v):
                        parts = [p.strip() for p in str(v).split(',')]
                        parts = [p for p in parts
                                 if p and not p.lower().startswith('specify:')]
                        if len(parts) == 0:
                            return None
                        if len(parts) == 1:
                            return parts[0]
                        return 'Multiracial'

                    cleaned_ethnic = demo_df["ethnic"].dropna().map(_clean_race).dropna()
                    if not cleaned_ethnic.empty:
                        vc = cleaned_ethnic.value_counts()
                        reportable = vc[vc >= _SMALL_CELL]
                        suppressed = vc[vc < _SMALL_CELL]
                        if not suppressed.empty:
                            reportable = reportable.copy()
                            reportable['Suppressed (n<11)'] = suppressed.sum()
                        labels = [str(k)[:40] for k in reversed(reportable.index)]
                        values = list(reversed(reportable.values))
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation='h',
                            marker_color='#AB63FA',
                        ))
                        fig.update_layout(
                            title="Race / Ethnicity",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("*Categories with fewer than 11 participants are grouped per HIPAA small-cell suppression guidelines.*")
                    else:
                        st.caption("Race/Ethnicity: no data available")
                else:
                    st.caption("Race/Ethnicity field not found")

            with col_fac:
                # Build location lookup from snapshot for anonymized labels
                _loc_lookup = {}
                try:
                    _snap = get_cached_snapshot()
                    for s in _snap.get('facilities', {}).get('site_locations', []):
                        city, state = s.get('city', ''), s.get('state', '')
                        label = f"{city}, {state}" if city else f"Site {s['facility_id']}"
                        _loc_lookup[str(s['facility_id'])] = label
                except Exception:
                    pass

                fig = create_facility_distribution_mini_chart(
                    demo_df,
                    title="Top Sites (This Cohort)",
                    location_lookup=_loc_lookup if _loc_lookup else None,
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("Site distribution: no data available")

            # Row 3: Health Insurance + Education/Employment
            col_ins, col_edu = st.columns(2)

            with col_ins:
                if "hltin" in demo_df.columns:
                    vals = demo_df["hltin"].dropna()
                    vals = vals[vals.astype(str).str.strip() != ""]
                    if not vals.empty:
                        exploded = vals.str.split(",").explode().str.strip()
                        exploded = exploded[~exploded.str.contains("specify:", case=False, na=False)]
                        exploded = exploded[exploded != ""]
                        if not exploded.empty:
                            vc = exploded.value_counts().head(15)
                            labels = [str(k)[:50] for k in reversed(vc.index)]
                            values = list(reversed(vc.values))
                            fig = go.Figure(go.Bar(
                                x=values, y=labels,
                                orientation="h",
                                marker_color="#FF7043",
                            ))
                            fig.update_layout(
                                title="Health Insurance",
                                xaxis_title="Participants",
                                height=max(250, len(labels) * 30 + 80),
                                margin=dict(t=40, b=40, l=250),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Participants may report multiple insurance types.")

            with col_edu:
                # Consolidate education across form versions
                _edu = None
                for _col in ("edulvl", "edulvl1", "edulvl2"):
                    if _col in demo_df.columns:
                        if _edu is None:
                            _edu = demo_df[_col]
                        else:
                            _edu = _edu.fillna(demo_df[_col])

                if _edu is not None:
                    _null_like = {'', '0', 'nan', 'none', 'null', 'n/a', 'na'}
                    _edu_clean = _edu.dropna()
                    _edu_clean = _edu_clean[~_edu_clean.astype(str).str.strip().str.lower().isin(_null_like)]
                    vc = _edu_clean.value_counts().head(10)
                    if not vc.empty:
                        labels = [str(k)[:40] for k in reversed(vc.index)]
                        values = list(reversed(vc.values))
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation="h",
                            marker_color="#26A69A",
                        ))
                        fig.update_layout(
                            title="Education Level",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif "employ" in demo_df.columns:
                    vc = demo_df["employ"].dropna().value_counts().head(10)
                    if not vc.empty:
                        labels = [str(k)[:40] for k in reversed(vc.index)]
                        values = list(reversed(vc.values))
                        fig = go.Figure(go.Bar(
                            x=values, y=labels,
                            orientation="h",
                            marker_color="#26A69A",
                        ))
                        fig.update_layout(
                            title="Employment Status",
                            xaxis_title="Participants",
                            height=max(250, len(labels) * 30 + 80),
                            margin=dict(t=40, b=40, l=200),
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No demographics data available for this cohort.")

    # --- Tab 2: Diagnosis ---
    with tab_diag:
        filter_config = _load_filter_config()
        disease_cfg = filter_config.get("disease_filters", {}).get(selected_disease, {})

        # Collect all chart-worthy filter definitions (diagnosis + clinical)
        all_chart_defs = []
        for category in ("diagnosis", "clinical"):
            for fdef in disease_cfg.get(category, []):
                all_chart_defs.append((category, fdef))

        if all_chart_defs:
            st.subheader(f"{selected_disease} Diagnosis Profile")

            # Map source_table config key → cohort dict key
            _TABLE_MAP = {
                "demographics": "demographics",
                "diagnosis": "diagnosis",
                "encounters": "encounters",
                "medications": "medications",
            }

            # Render 2 charts per row
            for i in range(0, len(all_chart_defs), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(all_chart_defs):
                        break

                    category, fdef = all_chart_defs[idx]
                    field = fdef["field"]
                    label = fdef["label"]
                    widget = fdef["widget"]
                    source = fdef["source_table"]

                    # Pick color based on category
                    if category == "clinical":
                        cscale = COLOR_SCHEMES.get("clinical", "Blues")
                    else:
                        cscale = COLOR_SCHEMES.get("diagnosis", "Greens")

                    df_key = _TABLE_MAP.get(source, source)
                    df = filtered_cohort.get(df_key, pd.DataFrame())

                    # Per-patient dedup for encounter fields (first/last)
                    per_patient = fdef.get("per_patient")
                    if (per_patient and isinstance(df, pd.DataFrame)
                            and not df.empty and "FACPATID" in df.columns
                            and "encntdt" in df.columns):
                        df_sorted = df.copy()
                        df_sorted["encntdt"] = pd.to_datetime(
                            df_sorted["encntdt"], errors="coerce"
                        )
                        df_sorted = df_sorted.dropna(subset=["encntdt"])
                        df_sorted = df_sorted.sort_values("encntdt")
                        if per_patient == "first":
                            df = df_sorted.groupby("FACPATID").first().reset_index()
                        elif per_patient == "last":
                            df = df_sorted.groupby("FACPATID").last().reset_index()

                    with col:
                        if isinstance(df, pd.DataFrame) and not df.empty and field in df.columns:
                            if widget == "multiselect":
                                fig = create_categorical_bar_chart(
                                    df[field],
                                    title=f"{label}",
                                    color_scale=cscale,
                                )
                            elif widget == "range_slider":
                                fig = create_numeric_histogram_chart(
                                    df[field],
                                    title=f"{label} Distribution",
                                    color="#00CC96" if category == "diagnosis" else "#636EFA",
                                )
                            elif widget == "checkbox":
                                mapped = df[field].map(
                                    {True: "Yes", False: "No", 1: "Yes", 0: "No"}
                                ).fillna(df[field].astype(str))
                                fig = create_categorical_bar_chart(
                                    mapped, title=label, color_scale=cscale
                                )
                            else:
                                fig = None

                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(f"{label}: insufficient data for chart")
                        else:
                            st.caption(f"{label}: no data available")
        else:
            st.subheader(f"{selected_disease} Diagnosis Profile")
            _unavailable_section("Diagnosis Profile", "No diagnosis profile configured for this disease.")

    # --- Tab 3: Deep Dive ---
    with tab_deep:
        renderer = _DEEP_DIVE_RENDERERS.get(selected_disease)
        if renderer:
            renderer()
        else:
            # Disease-specific placeholder descriptions
            _disease_placeholders = {
                "ALS": (
                    "A clinical deep dive for ALS is in development. "
                    "Upcoming features include ALSFRS-R longitudinal tracking, "
                    "respiratory function (FVC) trends, loss of ambulation analysis, "
                    "and therapeutic utilization."
                ),
                "SMA": (
                    "A clinical deep dive for SMA is in development. "
                    "Upcoming features include HFMSE/CHOP-INTEND longitudinal tracking, "
                    "respiratory function trends, and therapeutic utilization "
                    "(Spinraza, Zolgensma, Evrysdi)."
                ),
            }
            placeholder_msg = _disease_placeholders.get(
                selected_disease,
                f"A clinical deep dive for {selected_disease} is in development. "
                "Upcoming features include longitudinal functional outcome tracking, "
                "therapeutic utilization, and detailed cohort characterization.",
            )
            _unavailable_section(f"{selected_disease} Deep Dive", placeholder_msg)

    # --- Tab 4: Data Summary ---
    with tab_data:
        st.subheader("Data Summary")

        sub1, sub2, sub3, sub4 = st.tabs([
            "Cohort",
            "Demographics",
            "Encounters",
            "Medications"
        ])

        with sub1:
            st.markdown("#### Cohort Statistics")
            display_cohort_summary(cohort_summary)

            if cohort_summary.get('dstype_distribution'):
                st.markdown("#### Disease Type Distribution")
                dstype_data = [
                    {'Type': k, 'Count': v}
                    for k, v in cohort_summary['dstype_distribution'].items()
                ]
                static_table(dstype_data)

        with sub2:
            st.markdown(f"#### Demographics ({selected_disease})")
            demographics_df = filtered_cohort['demographics']
            if not demographics_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Patients", f"{len(demographics_df):,}")
                with col2:
                    st.metric("Fields", f"{len(demographics_df.columns):,}")
                with col3:
                    completeness = demographics_df.notna().mean().mean()
                    st.metric("Completeness", f"{completeness:.0%}")
            else:
                st.caption("No demographics data available.")

        with sub3:
            st.markdown(f"#### Encounters ({selected_disease})")
            encounters_df = filtered_cohort['encounters']
            if not encounters_df.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(encounters_df):,}")
                with col2:
                    n_pts = encounters_df['FACPATID'].nunique() if 'FACPATID' in encounters_df.columns else 0
                    st.metric("Patients", f"{n_pts:,}")
                with col3:
                    mean_enc = len(encounters_df) / max(n_pts, 1)
                    st.metric("Mean Visits / Patient", f"{mean_enc:.1f}")
            else:
                st.caption("No encounter data available.")

        with sub4:
            st.markdown(f"#### Medications ({selected_disease})")
            medications_df = filtered_cohort['medications']
            if not medications_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Records", f"{len(medications_df):,}")
                with col2:
                    n_pts = medications_df['FACPATID'].nunique() if 'FACPATID' in medications_df.columns else 0
                    st.metric("Patients", f"{n_pts:,}")
            else:
                st.caption("No medication data available.")

    st.caption(
        "Full data tables and CSV downloads are available in the "
        "**Download Center** (provisioned access required)."
    )

# Footer
render_page_footer()
