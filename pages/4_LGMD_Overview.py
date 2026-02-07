"""
LGMD Overview Page

Dedicated page for LGMD data presentation to Patient Advocacy Groups (PAGs).
Provides comprehensive subtype analysis, genetic confirmation rates, and clinical insights.
"""

import sys
from pathlib import Path

# Add webapp and parent directory to path
app_dir = Path(__file__).parent.parent
# App root is parent of pages/
sys.path.insert(0, str(app_dir))
# src is in app root

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from api import CohortAPI
from components.charts import (
    create_age_distribution_chart,
    create_categorical_bar_chart,
    create_categorical_donut_chart,
)
from config.settings import PAGE_ICON, LOGO_PNG

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

# Page configuration
st.set_page_config(
    page_title="LGMD Overview - OpenMOVR App",
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

# Contact info function for end of sidebar
def _render_sidebar_contact():
    st.sidebar.markdown("---")
    if LOGO_PNG.exists():
        st.sidebar.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.sidebar.image(str(LOGO_PNG), width=160)
        st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888;'>
            <strong>Open Source Project</strong><br>
            <a href="https://openmovr.github.io" target="_blank">openmovr.github.io</a><br>
            <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888;'>
            <strong>Created by</strong><br>
            Andre D Paredes<br>
            <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Header with branding
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("LGMD Registry Overview")
    st.markdown("### Limb-Girdle Muscular Dystrophy Data Summary")

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

st.caption(f"Data as of {datetime.now().strftime('%B %Y')}")


# ---------------------------------------------------------------------------
# Load LGMD Cohort (with snapshot fallback)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_lgmd_data():
    """Load and cache LGMD cohort data. Falls back to snapshot if parquet unavailable."""
    from api.lgmd import LGMDAPI
    return LGMDAPI.load_data()


with st.spinner("Loading LGMD cohort..."):
    try:
        lgmd_cohort = load_lgmd_data()
        is_snapshot_mode = lgmd_cohort.get('_is_snapshot', False)
    except Exception as e:
        st.error(f"Error loading LGMD data: {e}")
        st.info("Generate LGMD snapshot with: `python scripts/generate_lgmd_snapshot.py`")
        st.stop()

# Show data source indicator
if is_snapshot_mode:
    snapshot_data = lgmd_cohort.get('_snapshot_data', {})
    snapshot_date = snapshot_data.get('metadata', {}).get('generated_timestamp', 'Unknown')
    st.warning(f"**Snapshot Mode**: Displaying pre-computed statistics from {snapshot_date}. "
               "Filters are disabled. For full interactivity, load parquet data files.")

# Store data based on mode
if is_snapshot_mode:
    # Snapshot mode - use pre-computed statistics
    snapshot = lgmd_cohort['_snapshot_data']
    total_patient_count = snapshot['summary']['total_patients']
    facility_info = {
        'total_facilities': snapshot['summary']['total_facilities'],
        'facilities': [
            {'FACILITY_DISPLAY_ID': f['id'], 'FACILITY_NAME': f['name'], 'patient_count': f['patients']}
            for f in snapshot['facilities'].get('facilities', [])
        ]
    }
    # Create empty DataFrames for compatibility (won't be used in snapshot mode)
    demo_df_full = pd.DataFrame()
    diag_df_full = pd.DataFrame()
    enc_df_full = pd.DataFrame()
    med_df_full = pd.DataFrame()
else:
    # Live mode - use actual DataFrames
    demo_df_full = lgmd_cohort['demographics']
    diag_df_full = lgmd_cohort['diagnosis']
    enc_df_full = lgmd_cohort['encounters']
    med_df_full = lgmd_cohort['medications']
    total_patient_count = lgmd_cohort['count']


# ---------------------------------------------------------------------------
# Sidebar Filters (Live Mode Only)
# ---------------------------------------------------------------------------
filters_active = False
age_range = (0, 100)
min_age, max_age = 0, 100
selected_subtypes = []

with st.sidebar:
    st.header("Filters")

    if is_snapshot_mode:
        st.info("Filters disabled in snapshot mode. Load parquet files for full interactivity.")
    else:
        # Calculate current age for filtering
        if 'dob' in demo_df_full.columns:
            demo_df_full = demo_df_full.copy()
            demo_df_full['current_age'] = pd.to_datetime(demo_df_full['dob'], errors='coerce').apply(
                lambda x: (datetime.now() - x).days / 365.25 if pd.notna(x) else None
            )
            valid_ages = demo_df_full['current_age'].dropna()
            min_age = int(valid_ages.min()) if len(valid_ages) > 0 else 0
            max_age = int(valid_ages.max()) + 1 if len(valid_ages) > 0 else 100
        else:
            min_age, max_age = 0, 100

        # Age Range Filter
        st.subheader("Age Range")
        age_range = st.slider(
            "Current Age (years)",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
            key="lgmd_age_filter"
        )

        st.markdown("---")

        # Subtype Filter
        st.subheader("LGMD Subtype")
        if 'lgtype' in diag_df_full.columns:
            all_subtypes = diag_df_full['lgtype'].dropna().unique().tolist()
            all_subtypes = sorted([s for s in all_subtypes if s.strip()])

            selected_subtypes = st.multiselect(
                "Select Subtypes",
                options=all_subtypes,
                default=[],  # Empty = all selected
                key="lgmd_subtype_filter",
                help="Leave empty to include all subtypes"
            )
        else:
            selected_subtypes = []

        st.markdown("---")

        # Show filter summary
        filters_active = (age_range != (min_age, max_age)) or (len(selected_subtypes) > 0)
        if filters_active:
            st.info("Filters active")
            if st.button("Clear Filters", key="lgmd_clear"):
                st.rerun()

    # Contact info
    _render_sidebar_contact()


# ---------------------------------------------------------------------------
# Apply Filters (Live Mode Only)
# ---------------------------------------------------------------------------
if is_snapshot_mode:
    # Use snapshot data directly
    patient_count = total_patient_count
    demo_df = pd.DataFrame()
    diag_df = pd.DataFrame()
    enc_df = pd.DataFrame()
    med_df = pd.DataFrame()
else:
    # Live mode: Apply filters
    # Start with all patients
    filtered_patient_ids = set(demo_df_full['FACPATID'].tolist())

    # Apply age filter
    if 'current_age' in demo_df_full.columns:
        age_filtered = demo_df_full[
            (demo_df_full['current_age'] >= age_range[0]) &
            (demo_df_full['current_age'] <= age_range[1])
        ]['FACPATID'].tolist()
        filtered_patient_ids = filtered_patient_ids.intersection(set(age_filtered))

    # Apply subtype filter
    if selected_subtypes:
        subtype_filtered = diag_df_full[
            diag_df_full['lgtype'].isin(selected_subtypes)
        ]['FACPATID'].tolist()
        filtered_patient_ids = filtered_patient_ids.intersection(set(subtype_filtered))

    # Create filtered dataframes
    filtered_patient_ids = list(filtered_patient_ids)
    demo_df = demo_df_full[demo_df_full['FACPATID'].isin(filtered_patient_ids)].copy()
    diag_df = diag_df_full[diag_df_full['FACPATID'].isin(filtered_patient_ids)].copy()
    enc_df = enc_df_full[enc_df_full['FACPATID'].isin(filtered_patient_ids)].copy()
    med_df = med_df_full[med_df_full['FACPATID'].isin(filtered_patient_ids)].copy()
    patient_count = len(filtered_patient_ids)

    # Compute facility info for filtered cohort
    if 'FACILITY_DISPLAY_ID' in demo_df.columns:
        fac_counts = demo_df.groupby('FACILITY_DISPLAY_ID')['FACPATID'].nunique().reset_index(name='patient_count')
        fac_counts = fac_counts.sort_values('patient_count', ascending=False)
        if 'FACILITY_NAME' in demo_df.columns:
            names = demo_df[['FACILITY_DISPLAY_ID', 'FACILITY_NAME']].drop_duplicates()
            fac_counts = fac_counts.merge(names, on='FACILITY_DISPLAY_ID', how='left')
        facility_info = {
            'total_facilities': len(fac_counts),
            'facilities': fac_counts.to_dict('records')
        }
    else:
        facility_info = lgmd_cohort.get('facility_info', {})


# ---------------------------------------------------------------------------
# Key Metrics Row
# ---------------------------------------------------------------------------
st.markdown("---")

# Show filter status (live mode only)
if filters_active and not is_snapshot_mode:
    filter_parts = []
    if age_range != (min_age, max_age):
        filter_parts.append(f"Age: {age_range[0]}-{age_range[1]} years")
    if selected_subtypes:
        if len(selected_subtypes) <= 3:
            filter_parts.append(f"Subtypes: {', '.join(selected_subtypes)}")
        else:
            filter_parts.append(f"Subtypes: {len(selected_subtypes)} selected")

    st.info(
        f"**Filtered View:** {patient_count:,} of {total_patient_count:,} patients | "
        + " | ".join(filter_parts)
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta = None
    if filters_active and not is_snapshot_mode:
        diff = patient_count - total_patient_count
        delta = f"{diff:,}" if diff != 0 else None
    st.metric("Total Patients", f"{patient_count:,}", delta=delta)

with col2:
    facilities = facility_info.get('total_facilities', 0)
    st.metric("Care Sites", facilities)

with col3:
    if is_snapshot_mode:
        conf_pct = snapshot['diagnosis'].get('genetic_confirmation', {}).get('confirmed_percentage', 0)
    else:
        genetic_conf = diag_df['lggntcf'].value_counts()
        confirmed = genetic_conf.get('Yes – Laboratory confirmation', 0) + genetic_conf.get('Yes – In a family member', 0)
        conf_pct = round(confirmed / patient_count * 100) if patient_count > 0 else 0
    st.metric("Genetically Confirmed", f"{conf_pct}%")

with col4:
    if is_snapshot_mode:
        subtypes = snapshot['subtypes'].get('unique_subtypes', 0)
    else:
        subtypes = diag_df['lgtype'].nunique()
    st.metric("Subtypes Represented", subtypes)

with col5:
    if is_snapshot_mode:
        median_dx_age = snapshot['diagnosis'].get('diagnosis_age', {}).get('median')
    else:
        median_dx_age = pd.to_numeric(diag_df['lgdgag'], errors='coerce').median()
    st.metric("Median Dx Age", f"{median_dx_age:.1f} yrs" if median_dx_age is not None else "N/A")


# ---------------------------------------------------------------------------
# Subtype Distribution (Key Section)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("LGMD Subtype Distribution")

# Get subtype data based on mode
if is_snapshot_mode:
    subtype_data = snapshot['subtypes'].get('distribution', [])
    if subtype_data:
        subtype_counts = pd.DataFrame(subtype_data)
        subtype_counts.columns = ['Subtype', 'Patients', 'Percentage', 'Type']
        has_subtype_data = True
    else:
        has_subtype_data = False
elif 'lgtype' in diag_df.columns:
    subtype_counts = diag_df['lgtype'].value_counts().reset_index()
    subtype_counts.columns = ['Subtype', 'Patients']
    subtype_counts = subtype_counts[subtype_counts['Subtype'].str.strip() != '']
    subtype_counts['Percentage'] = (subtype_counts['Patients'] / subtype_counts['Patients'].sum() * 100).round(1)

    # Color by LGMD type (1 vs 2)
    def get_lgmd_type(subtype):
        if 'LGMD1' in str(subtype):
            return 'LGMD Type 1 (Dominant)'
        elif 'LGMD2' in str(subtype):
            return 'LGMD Type 2 (Recessive)'
        else:
            return 'Other/Undetermined'

    subtype_counts['Type'] = subtype_counts['Subtype'].apply(get_lgmd_type)
    has_subtype_data = True
else:
    has_subtype_data = False

if has_subtype_data:
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        # Rename column for snapshot mode compatibility
        if 'lgmd_type' in subtype_counts.columns:
            subtype_counts = subtype_counts.rename(columns={'lgmd_type': 'Type', 'subtype': 'Subtype', 'patients': 'Patients', 'percentage': 'Percentage'})

        fig = px.bar(
            subtype_counts.head(15),
            x='Patients',
            y='Subtype',
            orientation='h',
            color='Type',
            color_discrete_map={
                'LGMD Type 1 (Dominant)': '#636EFA',
                'LGMD Type 2 (Recessive)': '#00CC96',
                'Other/Undetermined': '#AB63FA'
            },
            title='Patient Count by LGMD Subtype'
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Subtype Breakdown**")
        display_df = subtype_counts[['Subtype', 'Patients', 'Percentage']].copy()
        display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x}%")
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=450)
else:
    st.info("Subtype data not available.")


# ---------------------------------------------------------------------------
# Demographics Row
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Patient Demographics")

col_enroll_age, col_age, col_gender, col_ethnic = st.columns(4)

with col_enroll_age:
    if is_snapshot_mode:
        enroll_age_data = snapshot['demographics'].get('enrollment_age', {})
        if enroll_age_data and 'histogram' in enroll_age_data:
            hist_data = enroll_age_data['histogram']
            fig = go.Figure(data=[go.Bar(x=hist_data['bins'], y=hist_data['counts'])])
            fig.update_layout(
                title="Age at Enrollment",
                xaxis_title="Age (years)",
                yaxis_title="Patients",
                showlegend=False
            )
            fig.add_vline(x=enroll_age_data['median'], line_dash="dash", line_color="red",
                         annotation_text=f"Median: {enroll_age_data['median']:.1f}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age at enrollment data not available")
    elif 'enroldt' in demo_df.columns and 'dob' in demo_df.columns:
        enrol_dates = pd.to_datetime(demo_df['enroldt'], errors='coerce')
        dob_dates = pd.to_datetime(demo_df['dob'], errors='coerce')
        age_at_enroll = (enrol_dates - dob_dates).dt.days / 365.25
        age_at_enroll = age_at_enroll.dropna()

        if len(age_at_enroll) > 0:
            fig = px.histogram(
                age_at_enroll,
                nbins=20,
                title="Age at Enrollment",
                labels={'value': 'Age (years)', 'count': 'Patients'},
                color_discrete_sequence=['#1E88E5']
            )
            fig.update_layout(showlegend=False)
            fig.add_vline(x=age_at_enroll.median(), line_dash="dash", line_color="red",
                         annotation_text=f"Median: {age_at_enroll.median():.1f}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age at enrollment data not available")
    else:
        st.caption("Enrollment date not available")

with col_age:
    if is_snapshot_mode:
        current_age_data = snapshot['demographics'].get('current_age', {})
        if current_age_data and 'histogram' in current_age_data:
            hist_data = current_age_data['histogram']
            fig = go.Figure(data=[go.Bar(x=hist_data['bins'], y=hist_data['counts'])])
            fig.update_layout(
                title="Current Age (if alive)",
                xaxis_title="Age (years)",
                yaxis_title="Patients",
                showlegend=False
            )
            fig.add_vline(x=current_age_data['median'], line_dash="dash", line_color="red",
                         annotation_text=f"Median: {current_age_data['median']:.1f}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age data not available")
    else:
        fig = create_age_distribution_chart(demo_df, title="Current Age (if alive)")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age data not available")

with col_gender:
    if is_snapshot_mode:
        gender_data = snapshot['demographics'].get('gender', {}).get('distribution', {})
        if gender_data:
            fig = go.Figure(data=[go.Pie(labels=list(gender_data.keys()), values=list(gender_data.values()), hole=0.4)])
            fig.update_layout(title="Gender Distribution", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    elif 'gender' in demo_df.columns:
        fig = create_categorical_donut_chart(demo_df['gender'], title="Gender Distribution")
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with col_ethnic:
    if is_snapshot_mode:
        ethnic_data = snapshot['demographics'].get('ethnicity', {}).get('distribution', {})
        if ethnic_data:
            fig = go.Figure(data=[go.Bar(x=list(ethnic_data.values()), y=list(ethnic_data.keys()), orientation='h')])
            fig.update_layout(title="Race/Ethnicity", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    elif 'ethnic' in demo_df.columns:
        fig = create_categorical_bar_chart(demo_df['ethnic'], title="Race/Ethnicity", max_categories=8)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Diagnosis Insights Row
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Diagnosis Characteristics")

col_dx_age, col_onset_age, col_genetic = st.columns(3)

with col_dx_age:
    if is_snapshot_mode:
        dx_age_data = snapshot['diagnosis'].get('diagnosis_age', {})
        if dx_age_data and 'histogram' in dx_age_data:
            hist_data = dx_age_data['histogram']
            fig = go.Figure(data=[go.Bar(x=hist_data['bins'], y=hist_data['counts'],
                                         marker_color='#00CC96')])
            fig.update_layout(
                title="Age at Diagnosis",
                xaxis_title="Age (years)",
                yaxis_title="Patients",
                showlegend=False
            )
            fig.add_vline(x=dx_age_data['median'], line_dash="dash", line_color="red",
                         annotation_text=f"Median: {dx_age_data['median']:.1f}")
            st.plotly_chart(fig, use_container_width=True)
    elif 'lgdgag' in diag_df.columns:
        dx_ages = pd.to_numeric(diag_df['lgdgag'], errors='coerce').dropna()
        if len(dx_ages) > 0:
            fig = px.histogram(
                dx_ages,
                nbins=20,
                title="Age at Diagnosis",
                labels={'value': 'Age (years)', 'count': 'Patients'},
                color_discrete_sequence=['#00CC96']
            )
            fig.update_layout(showlegend=False)
            fig.add_vline(x=dx_ages.median(), line_dash="dash", line_color="red",
                         annotation_text=f"Median: {dx_ages.median():.1f}")
            st.plotly_chart(fig, use_container_width=True)

with col_onset_age:
    if is_snapshot_mode:
        onset_age_data = snapshot['diagnosis'].get('onset_age', {})
        if onset_age_data and 'histogram' in onset_age_data:
            hist_data = onset_age_data['histogram']
            fig = go.Figure(data=[go.Bar(x=hist_data['bins'], y=hist_data['counts'],
                                         marker_color='#636EFA')])
            fig.update_layout(
                title="Age at Symptom Onset",
                xaxis_title="Age (years)",
                yaxis_title="Patients",
                showlegend=False
            )
            fig.add_vline(x=onset_age_data['median'], line_dash="dash", line_color="red",
                         annotation_text=f"Median: {onset_age_data['median']:.1f}")
            st.plotly_chart(fig, use_container_width=True)
    elif 'dymonag' in diag_df.columns:
        onset_ages = pd.to_numeric(diag_df['dymonag'], errors='coerce').dropna()
        if len(onset_ages) > 0:
            fig = px.histogram(
                onset_ages,
                nbins=20,
                title="Age at Symptom Onset",
                labels={'value': 'Age (years)', 'count': 'Patients'},
                color_discrete_sequence=['#636EFA']
            )
            fig.update_layout(showlegend=False)
            fig.add_vline(x=onset_ages.median(), line_dash="dash", line_color="red",
                         annotation_text=f"Median: {onset_ages.median():.1f}")
            st.plotly_chart(fig, use_container_width=True)

with col_genetic:
    if is_snapshot_mode:
        genetic_data = snapshot['diagnosis'].get('genetic_confirmation', {}).get('distribution', {})
        if genetic_data:
            fig = go.Figure(data=[go.Pie(labels=list(genetic_data.keys()), values=list(genetic_data.values()), hole=0.4)])
            fig.update_layout(title="Genetic Confirmation Status", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    elif 'lggntcf' in diag_df.columns:
        fig = create_categorical_donut_chart(
            diag_df['lggntcf'].fillna('Unknown'),
            title="Genetic Confirmation Status"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Clinical Characteristics Row
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Clinical Characteristics")

col_biopsy, col_family, col_symptoms = st.columns(3)

with col_biopsy:
    if is_snapshot_mode:
        biopsy_data = snapshot['clinical'].get('muscle_biopsy', {}).get('distribution', {})
        if biopsy_data:
            fig = go.Figure(data=[go.Pie(labels=list(biopsy_data.keys()), values=list(biopsy_data.values()), hole=0.4)])
            fig.update_layout(title="Muscle Biopsy Performed", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    elif 'lgmscbp' in diag_df.columns:
        fig = create_categorical_donut_chart(
            diag_df['lgmscbp'].fillna('Unknown'),
            title="Muscle Biopsy Performed"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with col_family:
    if is_snapshot_mode:
        family_data = snapshot['clinical'].get('family_history', {}).get('distribution', {})
        if family_data:
            fig = go.Figure(data=[go.Pie(labels=list(family_data.keys()), values=list(family_data.values()), hole=0.4)])
            fig.update_layout(title="Family History", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    elif 'lgfam' in diag_df.columns:
        fig = create_categorical_donut_chart(
            diag_df['lgfam'].fillna('Unknown'),
            title="Family History"
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)

with col_symptoms:
    if is_snapshot_mode:
        symptom_data = snapshot['clinical'].get('first_symptoms', {}).get('distribution', {})
        if symptom_data:
            fig = go.Figure(data=[go.Bar(x=list(symptom_data.values()), y=list(symptom_data.keys()),
                                         orientation='h', marker_color='#AB63FA')])
            fig.update_layout(title="First Symptoms Reported", yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    elif 'sym1st' in diag_df.columns:
        # Parse multi-value symptom field
        symptoms = diag_df['sym1st'].dropna()
        symptom_list = []
        for s in symptoms:
            if pd.notna(s) and s.strip():
                # Split by comma for multi-select values
                for symptom in str(s).split(','):
                    symptom_list.append(symptom.strip())

        if symptom_list:
            symptom_counts = pd.Series(symptom_list).value_counts().head(8)
            fig = px.bar(
                x=symptom_counts.values,
                y=symptom_counts.index,
                orientation='h',
                title="First Symptoms Reported",
                labels={'x': 'Mentions', 'y': 'Symptom'},
                color_discrete_sequence=['#AB63FA']
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Ambulatory Status (from encounters)
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Functional Status")

col_amb, col_amb_by_type = st.columns(2)

with col_amb:
    if is_snapshot_mode:
        amb_data = snapshot['ambulatory'].get('current_status', {}).get('distribution', {})
        if amb_data:
            fig = go.Figure(data=[go.Pie(labels=list(amb_data.keys()), values=list(amb_data.values()), hole=0.4)])
            fig.update_layout(title="Current Ambulatory Status", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Ambulatory status data not available")
    elif 'curramb' in enc_df.columns:
        # Get most recent ambulatory status per patient
        amb_status = enc_df[enc_df['curramb'].notna() & (enc_df['curramb'] != '')]
        if not amb_status.empty:
            # Get latest encounter per patient
            if 'encntdt' in amb_status.columns:
                amb_status = amb_status.copy()
                amb_status['encntdt'] = pd.to_datetime(amb_status['encntdt'], errors='coerce')
                latest_amb = amb_status.sort_values('encntdt').groupby('FACPATID').last()
            else:
                latest_amb = amb_status.groupby('FACPATID').last()

            fig = create_categorical_donut_chart(
                latest_amb['curramb'],
                title="Current Ambulatory Status"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

with col_amb_by_type:
    if is_snapshot_mode:
        # Snapshot mode: show by_subtype data if available
        by_subtype = snapshot['ambulatory'].get('by_subtype', {})
        if by_subtype:
            st.caption("Ambulatory status by subtype is available in live mode only.")
        else:
            st.caption("Subtype breakdown not available")
    elif 'curramb' in enc_df.columns and 'lgtype' in diag_df.columns:
        # Ambulatory status by top subtypes
        # Merge diagnosis subtype with encounters
        enc_with_type = enc_df.merge(
            diag_df[['FACPATID', 'lgtype']],
            on='FACPATID',
            how='left'
        )
        enc_with_type = enc_with_type[enc_with_type['curramb'].notna() & (enc_with_type['curramb'] != '')]

        if not enc_with_type.empty:
            # Get latest status per patient
            if 'encntdt' in enc_with_type.columns:
                enc_with_type = enc_with_type.copy()
                enc_with_type['encntdt'] = pd.to_datetime(enc_with_type['encntdt'], errors='coerce')
                latest = enc_with_type.sort_values('encntdt').groupby('FACPATID').last().reset_index()
            else:
                latest = enc_with_type.groupby('FACPATID').last().reset_index()

            # Filter to top 5 subtypes
            top_subtypes = latest['lgtype'].value_counts().head(5).index.tolist()
            latest_top = latest[latest['lgtype'].isin(top_subtypes)]

            if not latest_top.empty:
                cross_tab = pd.crosstab(latest_top['lgtype'], latest_top['curramb'])
                cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

                fig = px.bar(
                    cross_tab_pct.reset_index().melt(id_vars='lgtype'),
                    x='lgtype',
                    y='value',
                    color='curramb',
                    title="Ambulatory Status by Top Subtypes",
                    labels={'value': 'Percentage', 'lgtype': 'Subtype', 'curramb': 'Status'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Geographic Distribution
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Care Site Distribution")

# Get facility data based on mode
if is_snapshot_mode:
    fac_list = snapshot['facilities'].get('facilities', [])
    if fac_list:
        facilities_df = pd.DataFrame(fac_list)
        facilities_df = facilities_df.rename(columns={'name': 'FACILITY_NAME', 'patients': 'patient_count', 'id': 'FACILITY_DISPLAY_ID'})
    else:
        facilities_df = pd.DataFrame()
else:
    if facility_info and facility_info.get('facilities'):
        facilities_df = pd.DataFrame(facility_info['facilities'])
    else:
        facilities_df = pd.DataFrame()

if not facilities_df.empty:
    col_fac_chart, col_fac_table = st.columns([2, 1])

    with col_fac_chart:
        top_facilities = facilities_df.head(10)
        fig = px.bar(
            top_facilities,
            x='patient_count',
            y='FACILITY_NAME',
            orientation='h',
            title=f"Top 10 Care Sites (of {len(facilities_df)} total)",
            labels={'patient_count': 'LGMD Patients', 'FACILITY_NAME': 'Care Site'},
            color_discrete_sequence=['#19D3F3']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_fac_table:
        st.markdown("**All Care Sites**")
        display_fac = facilities_df[['FACILITY_NAME', 'patient_count']].copy()
        display_fac.columns = ['Care Site', 'Patients']
        st.dataframe(display_fac, hide_index=True, use_container_width=True, height=350)
else:
    st.info("Facility data not available.")


# ---------------------------------------------------------------------------
# Subtype Deep Dive (Expandable) - Live Mode Only
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Subtype Details")

if is_snapshot_mode:
    st.info("Subtype deep dive is available in live mode only. Load parquet files for detailed subtype analysis.")
    # Show basic subtype list from snapshot
    subtypes_list = snapshot['subtypes'].get('all_subtypes', [])
    if subtypes_list:
        st.caption(f"Subtypes in database: {', '.join(subtypes_list[:10])}{'...' if len(subtypes_list) > 10 else ''}")
elif 'lgtype' in diag_df.columns:
    subtypes_list = diag_df['lgtype'].dropna().unique().tolist()
    subtypes_list = [s for s in subtypes_list if s.strip()]
    subtypes_list = sorted(subtypes_list)

    selected_subtype = st.selectbox(
        "Select a subtype for detailed breakdown",
        options=subtypes_list,
        key="subtype_detail"
    )

    if selected_subtype:
        subtype_patients = diag_df[diag_df['lgtype'] == selected_subtype]['FACPATID'].tolist()
        subtype_demo = demo_df[demo_df['FACPATID'].isin(subtype_patients)]
        subtype_diag = diag_df[diag_df['FACPATID'].isin(subtype_patients)]
        subtype_enc = enc_df[enc_df['FACPATID'].isin(subtype_patients)]

        with st.expander(f"Details: {selected_subtype} ({len(subtype_patients)} patients)", expanded=True):
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)

            with col_s1:
                st.metric("Patients", len(subtype_patients))

            with col_s2:
                if 'lgdgag' in subtype_diag.columns:
                    med_age = pd.to_numeric(subtype_diag['lgdgag'], errors='coerce').median()
                    st.metric("Median Dx Age", f"{med_age:.1f}" if pd.notna(med_age) else "N/A")

            with col_s3:
                if 'gender' in subtype_demo.columns:
                    male_pct = (subtype_demo['gender'] == 'Male').sum() / len(subtype_demo) * 100
                    st.metric("Male %", f"{male_pct:.0f}%")

            with col_s4:
                if 'lggntcf' in subtype_diag.columns:
                    genetic = subtype_diag['lggntcf'].value_counts()
                    confirmed = genetic.get('Yes – Laboratory confirmation', 0)
                    conf_pct = confirmed / len(subtype_diag) * 100
                    st.metric("Lab Confirmed", f"{conf_pct:.0f}%")

            # Age distributions for this subtype
            col_dx, col_onset = st.columns(2)

            with col_dx:
                if 'lgdgag' in subtype_diag.columns:
                    ages = pd.to_numeric(subtype_diag['lgdgag'], errors='coerce').dropna()
                    if len(ages) > 0:
                        st.markdown(f"**Age at Diagnosis**: {ages.min():.0f} - {ages.max():.0f} yrs (median: {ages.median():.1f})")

            with col_onset:
                if 'dymonag' in subtype_diag.columns:
                    ages = pd.to_numeric(subtype_diag['dymonag'], errors='coerce').dropna()
                    if len(ages) > 0:
                        st.markdown(f"**Symptom Onset**: {ages.min():.0f} - {ages.max():.0f} yrs (median: {ages.median():.1f})")


# ---------------------------------------------------------------------------
# Data Export Section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Data Export")

col_exp1, col_exp2, col_exp3 = st.columns(3)

with col_exp1:
    # Subtype summary export
    if is_snapshot_mode:
        subtype_data = snapshot['subtypes'].get('distribution', [])
        if subtype_data:
            export_subtypes = pd.DataFrame(subtype_data)
            export_subtypes = export_subtypes.rename(columns={'subtype': 'Subtype', 'patients': 'Patient_Count', 'percentage': 'Percentage'})
            export_subtypes = export_subtypes[['Subtype', 'Patient_Count', 'Percentage']]

            st.download_button(
                label="Download Subtype Summary",
                data=export_subtypes.to_csv(index=False),
                file_name="lgmd_subtype_summary.csv",
                mime="text/csv"
            )
    elif 'lgtype' in diag_df.columns:
        export_subtypes = diag_df['lgtype'].value_counts().reset_index()
        export_subtypes.columns = ['Subtype', 'Patient_Count']
        export_subtypes['Percentage'] = (export_subtypes['Patient_Count'] / export_subtypes['Patient_Count'].sum() * 100).round(1)

        st.download_button(
            label="Download Subtype Summary",
            data=export_subtypes.to_csv(index=False),
            file_name="lgmd_subtype_summary.csv",
            mime="text/csv"
        )

with col_exp2:
    # Demographics summary
    if is_snapshot_mode:
        demo_summary = {
            'Total Patients': [snapshot['summary']['total_patients']],
            'Care Sites': [snapshot['summary']['total_facilities']],
            'Median Current Age': [snapshot['demographics'].get('current_age', {}).get('median')],
            'Median Diagnosis Age': [snapshot['diagnosis'].get('diagnosis_age', {}).get('median')],
            'Median Symptom Onset Age': [snapshot['diagnosis'].get('onset_age', {}).get('median')],
        }
    else:
        demo_summary = {
            'Total Patients': [patient_count],
            'Care Sites': [facility_info.get('total_facilities', 0)],
            'Median Current Age': [pd.to_numeric(demo_df.get('dob', pd.Series()).apply(
                lambda x: (datetime.now() - pd.to_datetime(x, errors='coerce')).days / 365.25
                if pd.notna(x) else None
            ), errors='coerce').median()] if not demo_df.empty else [None],
            'Median Diagnosis Age': [pd.to_numeric(diag_df['lgdgag'], errors='coerce').median()] if 'lgdgag' in diag_df.columns else [None],
            'Median Symptom Onset Age': [pd.to_numeric(diag_df['dymonag'], errors='coerce').median()] if 'dymonag' in diag_df.columns else [None],
        }

    st.download_button(
        label="Download Summary Stats",
        data=pd.DataFrame(demo_summary).to_csv(index=False),
        file_name="lgmd_summary_stats.csv",
        mime="text/csv"
    )

with col_exp3:
    # Full patient list (IDs only)
    if is_snapshot_mode:
        patient_ids = snapshot['summary'].get('patient_ids', [])
    else:
        patient_ids = lgmd_cohort.get('patient_ids', [])

    st.download_button(
        label="Download Patient IDs",
        data="\n".join(patient_ids),
        file_name="lgmd_patient_ids.txt",
        mime="text/plain"
    )


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "<strong>OpenMOVR App</strong> | MOVR Data Hub (MOVR 1.0)<br>"
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
    "LGMD Registry Data<br>"
    "<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
    "<a href='mailto:aparedes@mdausa.org'>aparedes@mdausa.org</a> | "
    "<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
    "</div>",
    unsafe_allow_html=True
)
