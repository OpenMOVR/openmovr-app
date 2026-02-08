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
# src is in app root

import streamlit as st
import pandas as pd
import yaml

from api import StatsAPI
from components.charts import (
    create_age_distribution_chart,
    create_categorical_bar_chart,
    create_categorical_donut_chart,
    create_numeric_histogram_chart,
    create_facility_distribution_mini_chart,
)
from components.tables import display_cohort_summary, static_table
from components.sidebar import inject_global_css, render_sidebar_footer
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
                Gen1 | v0.1.0 (Prototype)
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
_PROJECT_ROOT = Path(__file__).parent.parent
_FILTER_CONFIG_PATH = _PROJECT_ROOT / "config" / "disease_filters.yaml"


@st.cache_data
def _load_filter_config():
    if _FILTER_CONFIG_PATH.exists():
        with open(_FILTER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


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
    # METRICS ROW (from snapshot)
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
    # DISEASE DISTRIBUTION (from snapshot — always available)
    # ===================================================================
    st.markdown("---")
    st.subheader("Disease Distribution Overview")

    import plotly.express as px

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
        # Highlight selected disease
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Patient Counts**")
        table_df = diseases_df[['disease', 'patient_count', 'percentage']].copy()
        table_df.columns = ['Disease', 'Patients', '%']
        table_df['%'] = table_df['%'].apply(lambda x: f"{x:.1f}%")
        # Bold the selected disease
        static_table(table_df)

    # ===================================================================
    # DEMOGRAPHICS OVERVIEW (from snapshot disease_profiles)
    # ===================================================================
    disease_profiles = snapshot.get('disease_profiles', {})
    profile = disease_profiles.get(selected_disease, {})
    demo_snap = profile.get('demographics', {})

    st.markdown("---")
    st.subheader("Demographics Overview")

    if demo_snap:
        import plotly.graph_objects as go

        col_age, col_gender = st.columns(2)

        with col_age:
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
                st.caption("Age distribution not available.")

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
            # Show top 8 categories, combine rest into "Other"
            top = eth_data[:8]
            rest = sum(d['count'] for d in eth_data[8:])
            labels = [d['label'] for d in top]
            values = [d['count'] for d in top]
            if rest > 0:
                labels.append("Other (combined)")
                values.append(rest)
            fig = go.Figure(go.Bar(
                x=values, y=labels,
                orientation='h',
                marker_color='#AB63FA',
            ))
            fig.update_layout(
                title="Race / Ethnicity",
                xaxis_title="Patients",
                height=max(250, len(labels) * 30 + 80),
                margin=dict(t=40, b=40, l=200),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        _unavailable_section("Demographics Charts", "Demographic data not available for this disease.")

    # ===================================================================
    # DIAGNOSIS PROFILE (from snapshot disease_profiles)
    # ===================================================================
    diag_snap = profile.get('diagnosis', [])

    if diag_snap:
        st.markdown("---")
        st.subheader(f"{selected_disease} Diagnosis Profile")

        import plotly.graph_objects as go

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
                        # Shorten long labels
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

    # ===================================================================
    # DATA SUMMARY note
    # ===================================================================
    st.markdown("---")
    st.caption(
        "Full data tables and CSV downloads are available in the "
        "**Download Center** (provisioned access required)."
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
    # METRICS ROW
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
    # DEMOGRAPHICS OVERVIEW — Charts (all diseases)
    # ===================================================================
    st.markdown("---")
    st.subheader("Demographics Overview")

    demo_df = filtered_cohort.get("demographics", pd.DataFrame())

    if not demo_df.empty:
        # Row 1: Age + Gender
        col_age, col_gender = st.columns(2)

        with col_age:
            fig = create_age_distribution_chart(demo_df, title="Patient Age Distribution")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Age distribution: no DOB data available")

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
                fig = create_categorical_bar_chart(
                    demo_df["ethnic"],
                    title="Race / Ethnicity",
                    color_scale=COLOR_SCHEMES.get("demographics", "Purples"),
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
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
    else:
        st.warning("No demographics data available for this cohort.")


    # ===================================================================
    # DISEASE-SPECIFIC DIAGNOSIS PROFILE — Dynamic charts
    # ===================================================================
    filter_config = _load_filter_config()
    disease_cfg = filter_config.get("disease_filters", {}).get(selected_disease, {})

    # Collect all chart-worthy filter definitions (diagnosis + clinical)
    all_chart_defs = []
    for category in ("diagnosis", "clinical"):
        for fdef in disease_cfg.get(category, []):
            all_chart_defs.append((category, fdef))

    if all_chart_defs:
        st.markdown("---")
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

                with col:
                    if isinstance(df, pd.DataFrame) and not df.empty and field in df.columns:
                        if widget == "multiselect":
                            fig = create_categorical_bar_chart(
                                df[field],
                                title=f"{label} Distribution",
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


    # ===================================================================
    # DATA SUMMARY TABS
    # ===================================================================
    st.markdown("---")
    st.subheader("Data Summary")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Cohort",
        "Demographics",
        "Encounters",
        "Medications"
    ])

    with tab1:
        st.markdown("#### Cohort Statistics")
        display_cohort_summary(cohort_summary)

        if cohort_summary.get('dstype_distribution'):
            st.markdown("#### Disease Type Distribution")
            dstype_data = [
                {'Type': k, 'Count': v}
                for k, v in cohort_summary['dstype_distribution'].items()
            ]
            static_table(dstype_data)

    with tab2:
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

    with tab3:
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

    with tab4:
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
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<strong>OpenMOVR App</strong> | MOVR Data Hub (MOVR 1.0)<br>"
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
    "Use the sidebar to select different diseases and apply filters<br>"
    "<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
    "<a href='mailto:andre.paredes@ymail.com'>andre.paredes@ymail.com</a> | "
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
    "</div>",
    unsafe_allow_html=True
)
