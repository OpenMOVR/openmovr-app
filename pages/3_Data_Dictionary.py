"""
Data Dictionary Explorer

Browse, search, and filter the MOVR data dictionary fields.
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

from api.data_dictionary import DataDictionaryAPI
from components.tables import static_table
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer
from config.settings import PAGE_ICON

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

# Page configuration
st.set_page_config(
    page_title="Data Dictionary - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide"
)

inject_global_css()

# Header with branding
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("Data Dictionary Explorer")
    st.markdown("### Browse and search MOVR field definitions")
    _source = DataDictionaryAPI.get_data_source()
    if _source == "curated_snapshot":
        st.caption("Source: Curated dictionary with clinical domain classification")
    else:
        st.caption("Source: Raw data dictionary (parquet)")

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
# MOVR Data Overview
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("MOVR Data Overview")

st.markdown(
    """
    The MOVR registry captures clinical data for **7 neuromuscular diseases**
    (ALS, BMD, DMD, SMA, LGMD, FSHD, Pompe). Each disease functions as its
    own sub-registry with **shared** data forms and **disease-specific** eCRFs.
    The eCRFs and data dictionary were developed by the
    [Muscular Dystrophy Association](https://www.mda.org/science/movr2).
    """
)

# --- Demographics & Enrollment ---
st.markdown("#### Demographics & Enrollment (35 fields — shared across all diseases)")
st.markdown(
    """
    Captured once at enrollment. Identical for all 7 registries.

    Includes: age, date of birth, gender, race/ethnicity, health insurance,
    education level, employment status, school enrollment, USNDR status,
    and contact information (name, address, phone, email).
    """
)

# --- Diagnosis ---
st.markdown("#### Diagnosis (171 fields — disease-specific)")
st.markdown(
    """
    Captured once at enrollment. Each disease has its own diagnosis form with
    disease-specific prefixes (als\\*, bmd\\*, dmd\\*, sma\\*, lg\\*, fshd\\*, pom\\*).
    """
)

_dx_overview = {
    "Disease": ["ALS", "BMD", "DMD", "SMA", "LGMD", "FSHD", "Pompe"],
    "Fields": [21, 45, 45, 22, 30, 34, 37],
    "What's Captured": [
        "Diagnosis date/age, symptom onset, El Escorial criteria, body regions affected, gene mutation, family history",
        "Diagnosis date/age, genetic confirmation + report, dystrophin biopsy, DNA dosage analysis, exon mapping (single/multiple), frame type, walked independently milestones, family history",
        "Diagnosis date/age, genetic confirmation + report, dystrophin biopsy, DNA dosage analysis, exon mapping, frame type, family history",
        "Diagnosis date/age, initial method, SMA classification (Type I–IV), genetic confirmation, SMN1/SMN2 copy number, prenatal onset, non-SMN SMA, family history",
        "Diagnosis date/age, symptom onset date/age, LGMD subtype + gene (20+ subtypes), genetic confirmation, gene mutations (1 & 2), muscle biopsy, first symptoms, family history",
        "Diagnosis date/age, first medical attention, FSHD subtype, 4q35 deletion + size, somatic mosaicism, haplotype testing, D4Z4 methylation, SMCHD1 sequencing, first symptoms, loss of ambulation, family history",
        "Diagnosis date/age, initial method, Pompe classification (Infantile/Late-Onset), genetic confirmation, variant 1/2/3, GAA enzyme activity, CRIM status, symptom onset, pre-symptomatic flag, family history",
    ],
}

static_table(pd.DataFrame(_dx_overview))

# --- Encounter & Longitudinal Data ---
st.markdown("#### Encounter & Longitudinal Data (805 fields — per clinic visit)")
st.markdown(
    """
    Collected at **each clinic visit** by MOVR sites — this is the longitudinal
    core of the registry. Encounter fields (651) capture point-in-time
    assessments; Log fields (154) track start/end dates for medications,
    devices, surgeries, and milestones.
    """
)

_enc_overview = {
    "Disease": ["ALS", "BMD", "DMD", "SMA", "LGMD", "FSHD", "Pompe"],
    "Encounter Fields": [135, 206, 226, 358, 96, 133, 183],
    "Log Fields": [55, 55, 55, 55, 59, 72, 133],
    "Unique Strengths": [
        "ALSFRS-R questionnaire (speech, swallowing, handwriting, walking, respiratory), ALSCBS cognitive/behavioral scores, end-of-life planning",
        "Cardiac (Echo, EKG, MRI with LV/RV measurements), glucocorticoid therapy + steroid complications, timed function tests, scoliosis/spine imaging",
        "Deepest cardiac monitoring (59 fields: Echo, EKG, MRI, cardiac devices), glucocorticoids + steroid complications, NSAA, PUL 2.0, timed function tests, scoliosis/spine",
        "Most fields of any disease (358). Gross motor milestones (head control, sitting, crawling, standing, walking), HFMSE, RULM, CHOP-INTEND, Spinraza dosing, extensive pulmonary/ventilation monitoring",
        "Current ambulatory status, timed function tests (6MWT, 10MWT, stair climb, rise from supine), muscle biopsy tracking, assistive devices",
        "FSHD Clinical Scores (facial weakness, scapular/pelvic girdle, limbs, abdominal), timed function tests, FSHD-specific milestones (scapular winging, foot drop, difficulty closing eyes/mouth, Coats syndrome, hearing loss)",
        "Gross motor milestones (58 fields), cardiac monitoring (Echo, MRI), ERT dosing + antibody titers, GAA enzyme activity, cognitive/behavioral screening (MoCA), hearing assessments, neuroimaging",
    ],
}

static_table(pd.DataFrame(_enc_overview))

st.markdown(
    """
    **Shared across all diseases:** vital signs (height, weight, BMI),
    mobility/ambulation status, hospitalizations, medications, multidisciplinary
    care plans (specialists seen/referred), vaccinations, clinical research
    participation, and discontinuation.
    """
)

# --- Clinical Domain Coverage Matrix ---
domain_summary = DataDictionaryAPI.get_domain_summary()
if domain_summary:
    overview_data = {
        "Clinical Domain": [d["domain"] for d in domain_summary],
        "Fields": [d["field_count"] for d in domain_summary],
    }
    for disease in ["ALS", "BMD", "DMD", "SMA", "LGMD", "FSHD", "Pompe"]:
        overview_data[disease] = [d.get(disease, 0) for d in domain_summary]

    overview_df = pd.DataFrame(overview_data)

    st.markdown("#### Clinical Domain Coverage by Disease")
    st.caption(
        "All 1,024 fields classified into 19 clinical domains. "
        "Counts show fields applicable to each disease within that domain."
    )

    static_table(overview_df)

    curated_meta = DataDictionaryAPI.get_curated_metadata()
    if curated_meta:
        corrections = curated_meta.get("mislabel_corrections_applied", 0)
        rate = curated_meta.get("classification_rate", 0)
        st.markdown(
            f"<div style='background-color: #E8F5E9; border-left: 4px solid #4CAF50; padding: 10px 14px; "
            f"border-radius: 0 4px 4px 0; margin: 0.5rem 0 0.5rem 0; font-size: 0.9em;'>"
            f"<strong>Curated Dictionary:</strong> {rate}% of fields classified into clinical domains. "
            f"{corrections} mislabeled field corrections applied (FSHD fields incorrectly marked for LGMD). "
            f"Discontinuation data exists but was <strong>poorly captured</strong> across sites."
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div style='background-color: #FFF3E0; border-left: 4px solid #FF9800; padding: 10px 14px;
    border-radius: 0 4px 4px 0; margin: 0.5rem 0 1rem 0; font-size: 0.9em;'>
    <strong>Tip:</strong> Use the sidebar filters to explore the full data dictionary.
    Filter by Clinical Domain, Disease, Form, or search by keyword.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar Filters (Disease-first hierarchy)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    # 1. Disease filter (PRIMARY - at top)
    diseases = ["All"] + DataDictionaryAPI.get_diseases()
    selected_disease = st.selectbox(
        "1. Disease",
        options=diseases,
        key="dd_disease_filter",
        help="Start here: filter fields by disease applicability"
    )

    st.markdown("---")

    # 2. Clinical Domain filter (NEW - from curated snapshot)
    clinical_domains = DataDictionaryAPI.get_clinical_domains()
    if clinical_domains:
        domain_options = ["All"] + clinical_domains
        selected_domain = st.selectbox(
            "2. Clinical Domain",
            options=domain_options,
            key="dd_domain_filter",
            help="Filter by clinical domain (e.g., Cardiology, Pulmonary)"
        )
    else:
        selected_domain = "All"

    # 3. Form filter (File/Form - Demographics, Diagnosis, etc.)
    forms = ["All"] + DataDictionaryAPI.get_forms()
    selected_form = st.selectbox(
        "3. Form",
        options=forms,
        key="dd_form_filter",
        help="Filter by form type (Demographics, Diagnosis, Encounter, etc.)"
    )

    # 4. Excel Tab filter (more specific sheet names)
    excel_tabs = ["All"] + DataDictionaryAPI.get_excel_tabs()
    selected_excel_tab = st.selectbox(
        "4. Excel Tab / Sheet",
        options=excel_tabs,
        key="dd_excel_tab_filter",
        help="Filter by specific Excel sheet name"
    )

    # 5. Field type filter
    field_types = ["All"] + DataDictionaryAPI.get_field_types()
    selected_field_type = st.selectbox(
        "5. Field Type",
        options=field_types,
        key="dd_type_filter",
        help="Filter by data type (DATE, DECIMAL, RADIO BUTTON, etc.)"
    )

    st.markdown("---")

    # 6. Search box (at bottom for refinement)
    search_text = st.text_input(
        "Search Fields",
        placeholder="Search name, description...",
        key="dd_search",
        help="Search within filtered results"
    )

    st.markdown("---")

    # Reset button
    if st.button("Clear All Filters", key="dd_reset"):
        st.rerun()

    # Contact info
    render_sidebar_footer()


# ---------------------------------------------------------------------------
# Load and Filter Data
# ---------------------------------------------------------------------------
with st.spinner("Loading data dictionary..."):
    try:
        filtered_df = DataDictionaryAPI.search_fields(
            search_text=search_text,
            form_filter=selected_form if selected_form != "All" else None,
            excel_tab_filter=selected_excel_tab if selected_excel_tab != "All" else None,
            disease_filter=selected_disease if selected_disease != "All" else None,
            field_type_filter=selected_field_type if selected_field_type != "All" else None,
            clinical_domain_filter=selected_domain if selected_domain != "All" else None,
        )
        summary_stats = DataDictionaryAPI.get_summary_stats()
    except Exception as e:
        st.error(f"Error loading data dictionary: {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Metrics Row
# ---------------------------------------------------------------------------
st.markdown("---")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric(
        "Total Fields",
        f"{summary_stats['total_fields']:,}",
        help="Total fields in the data dictionary"
    )

with col2:
    _domain_count = len(DataDictionaryAPI.get_clinical_domains())
    st.metric(
        "Clinical Domains",
        _domain_count if _domain_count else "—",
        help="Clinical domain categories"
    )

with col3:
    st.metric(
        "Forms",
        summary_stats['form_count'],
        help="Number of form types"
    )

with col4:
    st.metric(
        "Excel Tabs",
        summary_stats['excel_tab_count'],
        help="Number of Excel sheets/tabs"
    )

with col5:
    st.metric(
        "Field Types",
        summary_stats['field_type_count'],
        help="Unique field types"
    )

with col6:
    st.metric(
        "Matching",
        f"{len(filtered_df):,}",
        help="Fields matching current filters"
    )


# Show active filters (in hierarchical order)
active_filters = []
if selected_disease != "All":
    active_filters.append(f"Disease: {selected_disease}")
if selected_domain != "All":
    active_filters.append(f"Domain: {selected_domain}")
if selected_form != "All":
    active_filters.append(f"Form: {selected_form}")
if selected_excel_tab != "All":
    active_filters.append(f"Tab: {selected_excel_tab}")
if selected_field_type != "All":
    active_filters.append(f"Type: {selected_field_type}")
if search_text:
    active_filters.append(f'Search: "{search_text}"')

if active_filters:
    st.info(f"Filtering: {' → '.join(active_filters)}")


# ---------------------------------------------------------------------------
# Fields Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Fields")

if filtered_df.empty:
    st.warning("No fields match the current filters.")
else:
    # Add required flag
    filtered_df = DataDictionaryAPI.add_required_flag(filtered_df)

    # Add validity flags if disease is selected
    if selected_disease != "All":
        filtered_df = DataDictionaryAPI.add_validity_flags(filtered_df, selected_disease)
        invalid_count = (~filtered_df["Valid"]).sum() if "Valid" in filtered_df.columns else 0
    else:
        invalid_count = 0

    # Count required fields
    required_count = filtered_df["Required"].sum() if "Required" in filtered_df.columns else 0

    # Show count info
    total_matching = len(filtered_df)
    showing = min(100, total_matching)

    col_info, col_calc = st.columns([2, 1])

    with col_info:
        info_parts = [f"{total_matching} fields"]
        if required_count > 0:
            info_parts.append(f"{required_count} required")
        if invalid_count > 0:
            info_parts.append(f"{invalid_count} potentially mislabeled")

        if showing < total_matching:
            st.caption(f"Showing {showing} of {' | '.join(info_parts)}")
        else:
            st.caption(" | ".join(info_parts))

        if invalid_count > 0:
            st.warning(f"Found {invalid_count} fields that may be incorrectly marked for {selected_disease}")

    with col_calc:
        # Completeness calculation button
        calc_completeness = st.button(
            "Calculate Completeness %",
            key="dd_calc_complete",
            help="Calculate % of patients with data for each field (may take a moment)"
        )

    # Prepare display dataframe
    display_cols = ["Field Name", "Description", "Clinical Domain", "File/Form", "Excel Tab", "Field Type", "Required"]
    if selected_disease != "All":
        display_cols.extend(["Valid", "Validity Note"])
    available_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[available_cols].head(100).copy()

    # Truncate long descriptions
    if "Description" in display_df.columns:
        display_df["Description"] = display_df["Description"].astype(str).str[:100]

    # Calculate completeness if requested
    if calc_completeness:
        with st.spinner("Calculating data completeness..."):
            try:
                completeness_df = DataDictionaryAPI.calculate_completeness(
                    filtered_df.head(100),
                    disease=selected_disease if selected_disease != "All" else None
                )
                if "Completeness %" in completeness_df.columns:
                    display_df["Completeness %"] = completeness_df["Completeness %"]
                st.success("Completeness calculated!")
            except Exception as e:
                st.error(f"Error calculating completeness: {e}")

    # Add visual indicators
    if "Required" in display_df.columns:
        display_df["Required"] = display_df["Required"].apply(lambda x: "* REQUIRED" if x else "")

    if "Valid" in display_df.columns:
        display_df["Valid"] = display_df["Valid"].apply(lambda x: "OK" if x else "INVALID")

    # Build column config
    col_config = {
        "Required": st.column_config.TextColumn(
            "Required",
            help="Fields marked with * are required",
            width="small",
        ),
    }

    if "Completeness %" in display_df.columns:
        col_config["Completeness %"] = st.column_config.ProgressColumn(
            "Completeness %",
            help="% of patients with data",
            min_value=0,
            max_value=100,
            format="%.1f%%",
        )

    if "Valid" in display_df.columns:
        col_config["Valid"] = st.column_config.TextColumn(
            "Valid",
            help="Whether field is appropriate for selected disease",
            width="small",
        )
        col_config["Validity Note"] = st.column_config.TextColumn(
            "Issue",
            help="Reason why field may be mislabeled",
            width="medium",
        )

    # Display table (static — no CSV export on public pages)
    static_table(display_df)

    caption_parts = ["'* REQUIRED' = mandatory field"]
    if selected_disease != "All":
        caption_parts.append("'INVALID' = field may be mislabeled for this disease")
    if showing < total_matching:
        caption_parts.append(f"Showing {showing} of {total_matching} — use sidebar filters to narrow results")
    st.caption(" | ".join(caption_parts))


# ---------------------------------------------------------------------------
# Field Detail Expander
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Field Details")

if not filtered_df.empty and "Field Name" in filtered_df.columns:
    field_names = filtered_df["Field Name"].dropna().tolist()

    if field_names:
        selected_field = st.selectbox(
            "Select a field to view details",
            options=field_names,
            key="dd_field_select"
        )

        if selected_field:
            detail = DataDictionaryAPI.get_field_detail(selected_field)

            if detail:
                with st.expander(f"Details: {selected_field}", expanded=True):
                    # Core info in columns
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("**Field Name**")
                        st.code(detail.get("Field Name", ""))

                        st.markdown("**Display Label**")
                        st.write(detail.get("Display Label", "-"))

                        st.markdown("**Field Type**")
                        st.write(detail.get("Field Type", "-"))

                    with col_b:
                        st.markdown("**Form**")
                        st.write(detail.get("File/Form", "-"))

                        st.markdown("**Excel Tab**")
                        st.write(detail.get("Excel Tab", "-"))

                        st.markdown("**Column Number**")
                        st.write(detail.get("Column Number", "-"))

                    # Repeat Group if present
                    repeat_group = detail.get("Repeat Group Name", "")
                    if repeat_group:
                        st.markdown("**Repeat Group**")
                        st.write(repeat_group)

                    # Description
                    st.markdown("**Description**")
                    desc = detail.get("Description", "-")
                    st.write(desc if desc else "-")

                    # Numeric ranges if present
                    numeric_ranges = detail.get("Numeric Ranges", "")
                    if numeric_ranges:
                        st.markdown("**Numeric Ranges**")
                        st.write(numeric_ranges)

                    # Disease applicability
                    disease_info = detail.get("diseases", {})
                    if disease_info:
                        st.markdown("**Disease Applicability**")
                        disease_badges = []
                        for disease, applicable in disease_info.items():
                            if applicable:
                                disease_badges.append(disease)

                        if disease_badges:
                            st.write(", ".join(disease_badges))
                        else:
                            st.write("Not disease-specific (universal field)")
            else:
                st.warning(f"No details found for field: {selected_field}")
else:
    st.info("Apply filters above to search for fields, then select one to view details.")


# ---------------------------------------------------------------------------
# Disease Coverage Summary
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Disease Coverage Summary")

disease_coverage = summary_stats.get("disease_coverage", {})
if disease_coverage:
    coverage_df = pd.DataFrame([
        {"Disease": disease, "Fields": count}
        for disease, count in disease_coverage.items()
    ])
    coverage_df = coverage_df.sort_values("Fields", ascending=False)

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.bar_chart(coverage_df.set_index("Disease")["Fields"])

    with col_table:
        static_table(coverage_df)
else:
    st.info("No disease coverage data available.")


# Footer
render_page_footer()
