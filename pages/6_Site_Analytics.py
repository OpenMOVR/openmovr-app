"""
Site Analytics Page

Provisioned-access page for site-level analytics.
Requires authentication — intended for participating sites, DUA holders,
and approved researchers.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from config.settings import PAGE_ICON, DISEASE_DISPLAY_ORDER, COLOR_SCHEMES
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer
from utils.access import require_access
from components.charts import (
    create_age_distribution_chart,
    create_categorical_bar_chart,
    create_categorical_donut_chart,
    create_numeric_histogram_chart,
)

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="Site Analytics - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
header_left, header_right = st.columns([3, 1])
with header_left:
    st.title("Site Analytics")
    st.markdown("### Site-level clinical data analytics")
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
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Access gate
# ---------------------------------------------------------------------------
require_access(
    page_title="Site Analytics",
    description=(
        "Site-level analytics including facility identification, per-site patient "
        "demographics, disease breakdowns, and benchmark comparisons.\n\n"
        "Available to participating sites, researchers, PAGs, and patients "
        "with an approved Data Use Agreement.  All other inquiries should be "
        "directed to the MOVR team.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)


# ===================================================================
# AUTHENTICATED — Load data
# ===================================================================
_DATA_DIR = Path(__file__).parent.parent / "data"
_has_parquet = any(_DATA_DIR.glob("*.parquet")) if _DATA_DIR.exists() else False

if not _has_parquet:
    st.error(
        "**Site Analytics requires live data.**\n\n"
        "Parquet files are not available in this deployment. "
        "This feature is designed for local/provisioned environments with access to the MOVR dataset."
    )
    render_sidebar_footer()
    st.stop()

from api import CohortAPI
from utils.cache import get_cached_base_cohort

# Load base cohort
with st.spinner("Loading MOVR base cohort..."):
    try:
        base_cohort = get_cached_base_cohort(include_usndr=False)
    except Exception as e:
        st.error(f"Error loading base cohort: {e}")
        st.stop()

enc = base_cohort["encounters"]
demo = base_cohort["demographics"]
diag = base_cohort["diagnosis"]

# Load filter config for disease-specific fields
_FILTER_CONFIG_PATH = Path(__file__).parent.parent / "config" / "disease_filters.yaml"


@st.cache_data
def _load_filter_config():
    if _FILTER_CONFIG_PATH.exists():
        with open(_FILTER_CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {}


# Load site metadata for name resolution
_SITES_EXCEL = _DATA_DIR / "MOVR Sites - Tracker Information - EK.xlsx"


@st.cache_data
def _load_site_metadata():
    if not _SITES_EXCEL.exists():
        return {}
    sites = pd.read_excel(_SITES_EXCEL)
    lookup = {}
    for _, row in sites.iterrows():
        fid = str(int(row["FACILITY_DISPLAY_ID"]))
        lookup[fid] = {
            "name": str(row.get("FACILITY_NAME", "")).strip(),
            "movr_name": str(row.get("MOVR Site Name", "")).strip(),
            "city": str(row.get("City", "")).strip(),
            "state": str(row.get("State", "")).strip(),
        }
    return lookup


site_meta = _load_site_metadata()

# Build site lists
all_facility_ids = sorted(enc["FACILITY_DISPLAY_ID"].dropna().unique().astype(str))
facility_labels = {}
for fid in all_facility_ids:
    meta = site_meta.get(fid, {})
    name = meta.get("name", "")
    city = meta.get("city", "")
    state = meta.get("state", "")
    if name and city:
        facility_labels[fid] = f"{name} — {city}, {state} (ID: {fid})"
    elif name:
        facility_labels[fid] = f"{name} (ID: {fid})"
    else:
        facility_labels[fid] = f"Site {fid}"

# State list
all_states = sorted(set(
    site_meta[fid]["state"]
    for fid in all_facility_ids
    if fid in site_meta and site_meta[fid].get("state")
))


# ---------------------------------------------------------------------------
# Sidebar — site selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Site Selection")

    # State filter
    state_filter = st.selectbox(
        "Filter by State",
        options=["All States"] + all_states,
        key="sa_state_filter",
    )

    # Filter facility list by state
    if state_filter != "All States":
        available_fids = [
            fid for fid in all_facility_ids
            if site_meta.get(fid, {}).get("state") == state_filter
        ]
    else:
        available_fids = all_facility_ids

    # Site selector
    label_to_fid = {facility_labels[fid]: fid for fid in available_fids}
    selected_label = st.selectbox(
        "Select Site",
        options=sorted(label_to_fid.keys()),
        key="sa_site_select",
    )
    selected_fid = label_to_fid.get(selected_label, available_fids[0] if available_fids else None)

    st.markdown("---")
    st.info(f"Site ID: **{selected_fid}**")

    if st.button("Log Out", key="sa_logout"):
        st.session_state["site_access_granted"] = False
        st.rerun()

    render_sidebar_footer()


if not selected_fid:
    st.warning("No site selected.")
    st.stop()


# ---------------------------------------------------------------------------
# Filter data to selected site
# ---------------------------------------------------------------------------
site_enc = enc[enc["FACILITY_DISPLAY_ID"].astype(str) == selected_fid].copy()
site_demo = demo[demo["FACPATID"].isin(site_enc["FACPATID"].unique())].copy()
site_diag = diag[diag["FACPATID"].isin(site_enc["FACPATID"].unique())].copy()

site_patients = site_enc["FACPATID"].nunique()
site_encounters = len(site_enc)
site_diseases = sorted(site_enc["dstype"].dropna().unique()) if "dstype" in site_enc.columns else []

meta = site_meta.get(selected_fid, {})
site_display = meta.get("name", f"Site {selected_fid}")
site_location = f"{meta.get('city', '')}, {meta.get('state', '')}".strip(", ")


# ===================================================================
# SITE OVERVIEW
# ===================================================================
st.markdown("---")
st.subheader(f"{site_display}")
if site_location:
    st.caption(site_location)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Patients", f"{site_patients:,}")
with col2:
    st.metric("Encounters", f"{site_encounters:,}")
with col3:
    st.metric("Disease Types", len(site_diseases))
with col4:
    overall_pts = enc["FACPATID"].nunique()
    pct = (site_patients / overall_pts * 100) if overall_pts else 0
    st.metric("% of Registry", f"{pct:.1f}%")

# Disease breakdown at this site
if site_diseases:
    ds_rows = []
    for ds in DISEASE_DISPLAY_ORDER:
        if ds not in site_diseases and ds != "POM":
            continue
        # Handle POM/Pompe naming
        ds_check = ds
        if ds == "POM" and "Pompe" in site_diseases:
            ds_check = "Pompe"
        elif ds == "POM" and ds not in site_diseases:
            continue
        ds_mask = site_enc["dstype"] == ds_check
        ds_pts = int(site_enc.loc[ds_mask, "FACPATID"].nunique())
        ds_enc_count = int(ds_mask.sum())
        if ds_pts > 0:
            ds_rows.append({
                "Disease": ds_check,
                "Patients": ds_pts,
                "Encounters": ds_enc_count,
                "% of Site": f"{ds_pts / site_patients * 100:.1f}%",
            })
    if ds_rows:
        st.markdown("##### Disease Distribution at This Site")
        st.dataframe(pd.DataFrame(ds_rows), use_container_width=True, hide_index=True)


# ===================================================================
# DEMOGRAPHICS — Age, Gender, Race/Ethnicity
# ===================================================================
st.markdown("---")
st.subheader("Demographics")

if not site_demo.empty:
    col_age, col_gender = st.columns(2)

    with col_age:
        fig = create_age_distribution_chart(site_demo, title="Patient Age Distribution")
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Age distribution: no DOB data")

    with col_gender:
        if "gender" in site_demo.columns:
            fig = create_categorical_donut_chart(site_demo["gender"], title="Gender")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Gender: no data")

    if "ethnic" in site_demo.columns:
        fig = create_categorical_bar_chart(
            site_demo["ethnic"], title="Race / Ethnicity",
            color_scale=COLOR_SCHEMES.get("demographics", "Purples"),
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No demographics data for this site.")


# ===================================================================
# MEDICATIONS OVERVIEW (all diseases at this site)
# ===================================================================
st.markdown("---")
st.subheader("Medications Overview")

# Load medication repeat group tables
@st.cache_data
def _load_meds():
    enc_path = _DATA_DIR / "encounter_medication_rg.parquet"
    log_path = _DATA_DIR / "log_medication_rg.parquet"
    frames = []
    for p in [enc_path, log_path]:
        if p.exists():
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


all_meds = _load_meds()
site_patient_ids = set(site_enc["FACPATID"].unique())

if not all_meds.empty and "FACPATID" in all_meds.columns:
    site_meds = all_meds[all_meds["FACPATID"].isin(site_patient_ids)]

    if not site_meds.empty and "medname" in site_meds.columns:
        med_counts = site_meds["medname"].dropna().value_counts().head(20)
        if not med_counts.empty:
            med_df = pd.DataFrame({"Medication": med_counts.index, "Records": med_counts.values})
            col_med_chart, col_med_table = st.columns([2, 1])
            with col_med_chart:
                fig = px.bar(
                    med_df, x="Records", y="Medication", orientation="h",
                    title="Top 20 Medications Prescribed",
                    color="Records", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=max(400, len(med_df) * 25),
                    yaxis=dict(categoryorder="total ascending"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_med_table:
                st.markdown("**Medication Summary**")
                med_df["Patients"] = [
                    int(site_meds[site_meds["medname"] == m]["FACPATID"].nunique())
                    for m in med_df["Medication"]
                ]
                st.dataframe(med_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No coded medications at this site.")
    else:
        st.caption("No medication data for this site.")
else:
    st.caption("Medication data not available.")


# ===================================================================
# PER-DISEASE SECTIONS
# ===================================================================
st.markdown("---")
st.subheader("Disease-Specific Analytics")
st.caption("Each disease present at this site is shown below with site vs. overall comparison.")

filter_config = _load_filter_config()
disease_filters = filter_config.get("disease_filters", {})

# Map source_table key → cohort key
_TABLE_MAP = {
    "demographics": "demographics",
    "diagnosis": "diagnosis",
    "encounters": "encounters",
    "medications": "medications",
}

for ds in DISEASE_DISPLAY_ORDER:
    # Handle POM/Pompe
    ds_label = ds
    ds_check = ds
    if ds == "POM":
        if "Pompe" in site_diseases:
            ds_check = "Pompe"
            ds_label = "Pompe"
        elif "POM" not in site_diseases:
            continue
    elif ds not in site_diseases:
        continue

    ds_mask_site = site_enc["dstype"] == ds_check
    ds_site_pts = site_enc.loc[ds_mask_site, "FACPATID"].unique()

    if len(ds_site_pts) == 0:
        continue

    ds_mask_overall = enc["dstype"] == ds_check
    ds_overall_pts = enc.loc[ds_mask_overall, "FACPATID"].unique()

    with st.expander(f"{ds_label} — {len(ds_site_pts):,} patients at this site  |  {len(ds_overall_pts):,} overall", expanded=False):

        # Site vs Overall metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Site Patients", f"{len(ds_site_pts):,}")
        with m2:
            st.metric("Overall Patients", f"{len(ds_overall_pts):,}")
        with m3:
            share = len(ds_site_pts) / len(ds_overall_pts) * 100 if len(ds_overall_pts) else 0
            st.metric("Site Share", f"{share:.1f}%")

        # Demographics comparison: Age
        site_ds_demo = demo[demo["FACPATID"].isin(ds_site_pts)]
        overall_ds_demo = demo[demo["FACPATID"].isin(ds_overall_pts)]

        col_site, col_overall = st.columns(2)

        with col_site:
            fig = create_age_distribution_chart(
                site_ds_demo,
                title=f"Age — This Site ({len(ds_site_pts):,})",
                color="#1E88E5",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Age: insufficient data")

        with col_overall:
            fig = create_age_distribution_chart(
                overall_ds_demo,
                title=f"Age — All Sites ({len(ds_overall_pts):,})",
                color="#90CAF9",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Age: insufficient data")

        # Gender comparison
        if "gender" in demo.columns:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig = create_categorical_donut_chart(
                    site_ds_demo["gender"], title="Gender — This Site"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            with col_g2:
                fig = create_categorical_donut_chart(
                    overall_ds_demo["gender"], title="Gender — All Sites"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

        # Disease-specific fields from filter config
        ds_cfg_key = ds if ds != "Pompe" else "POM"
        ds_cfg = disease_filters.get(ds_cfg_key, {})
        chart_defs = []
        for category in ("diagnosis", "clinical"):
            for fdef in ds_cfg.get(category, []):
                chart_defs.append((category, fdef))

        if chart_defs:
            st.markdown("**Disease-Specific Variables — Site vs Overall**")

            site_ds_enc = enc[enc["FACPATID"].isin(ds_site_pts)]
            overall_ds_enc = enc[enc["FACPATID"].isin(ds_overall_pts)]
            site_ds_diag = diag[diag["FACPATID"].isin(ds_site_pts)]
            overall_ds_diag = diag[diag["FACPATID"].isin(ds_overall_pts)]

            ds_tables = {
                "demographics": (site_ds_demo, overall_ds_demo),
                "diagnosis": (site_ds_diag, overall_ds_diag),
                "encounters": (site_ds_enc, overall_ds_enc),
            }

            for category, fdef in chart_defs:
                field = fdef["field"]
                label = fdef["label"]
                widget = fdef["widget"]
                source = fdef["source_table"]

                site_df, overall_df = ds_tables.get(source, (pd.DataFrame(), pd.DataFrame()))

                cscale = COLOR_SCHEMES.get(category, "Blues")

                col_s, col_o = st.columns(2)

                for col, df, suffix in [(col_s, site_df, "This Site"), (col_o, overall_df, "All Sites")]:
                    with col:
                        if isinstance(df, pd.DataFrame) and not df.empty and field in df.columns:
                            if widget == "multiselect":
                                fig = create_categorical_bar_chart(
                                    df[field], title=f"{label} — {suffix}",
                                    color_scale=cscale,
                                )
                            elif widget == "range_slider":
                                fig = create_numeric_histogram_chart(
                                    df[field], title=f"{label} — {suffix}",
                                )
                            elif widget == "checkbox":
                                mapped = df[field].map(
                                    {True: "Yes", False: "No", 1: "Yes", 0: "No"}
                                ).fillna(df[field].astype(str))
                                fig = create_categorical_bar_chart(
                                    mapped, title=f"{label} — {suffix}",
                                    color_scale=cscale,
                                )
                            else:
                                fig = None

                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.caption(f"{label}: insufficient data")
                        else:
                            st.caption(f"{label}: no data")

        # Functional scores comparison (if available)
        _SCORE_FIELDS = {
            "ALS": [("alsfrstl", "ALSFRS-R Total")],
            "SMA": [("hfmsesc", "HFMSE"), ("cittlscr", "CHOP-INTEND"), ("rulmcs", "RULM")],
        }
        score_fields = _SCORE_FIELDS.get(ds_cfg_key, [])
        if score_fields:
            site_ds_enc = enc[enc["FACPATID"].isin(ds_site_pts)]
            overall_ds_enc = enc[enc["FACPATID"].isin(ds_overall_pts)]

            for field, label in score_fields:
                if field not in enc.columns:
                    continue
                site_vals = pd.to_numeric(site_ds_enc[field], errors="coerce").dropna()
                overall_vals = pd.to_numeric(overall_ds_enc[field], errors="coerce").dropna()
                if len(site_vals) < 2 and len(overall_vals) < 2:
                    continue

                col_s, col_o = st.columns(2)
                with col_s:
                    fig = create_numeric_histogram_chart(
                        site_vals, title=f"{label} — This Site",
                        color="#1E88E5",
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(f"{label}: insufficient site data")

                with col_o:
                    fig = create_numeric_histogram_chart(
                        overall_vals, title=f"{label} — All Sites",
                        color="#90CAF9",
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(f"{label}: insufficient data")


# Footer
render_page_footer()
