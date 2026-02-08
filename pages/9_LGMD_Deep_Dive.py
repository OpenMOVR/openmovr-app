"""
LGMD Deep Dive — DUA-Gated Page

Standalone page requiring provisioned access.  Shows all LGMD deep-dive
figures (downloadable via Plotly modebar) plus summary and patient-level
data tables with CSV export.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
from config.settings import PAGE_ICON, APP_VERSION
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer
from components.deep_dive import render_lgmd_deep_dive
from utils.access import require_access
from api.lgmd import LGMDAPI

_logo_path = app_dir / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="LGMD Deep Dive - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Access gate ----
require_access(
    page_title="LGMD Deep Dive",
    description=(
        "In-depth clinical analytics for the **Limb-Girdle Muscular Dystrophy (LGMD)** cohort.  "
        "Includes subtype distribution, diagnostic journey, functional outcomes, "
        "medication utilization, and downloadable data tables.\n\n"
        "Available to participating sites, researchers, PAGs, and patients "
        "with an approved Data Use Agreement.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)

# ---- Header ----
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("LGMD Deep Dive")
    st.markdown("### Subtypes, diagnostic journey, functional outcomes & more")

with header_right:
    st.markdown(
        f"""
        <div style='text-align: right; padding-top: 10px;'>
            <span style='font-size: 1.5em; font-weight: bold; color: #1E88E5;'>OpenMOVR App</span><br>
            <span style='font-size: 0.9em; color: #666; background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px;'>
                Gen1 | v{APP_VERSION}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Deep-dive charts (all downloadable via Plotly modebar) ----
render_lgmd_deep_dive()

# ---- Data Tables ----
st.markdown("---")
st.subheader("Data Tables")

tab_summary, tab_patient = st.tabs(["Summary Tables", "Patient-Level Data"])

# ===== Tab 1: Summary tables from snapshot =====
with tab_summary:
    lgmd_snap = LGMDAPI.get_snapshot()
    if not lgmd_snap:
        st.warning("LGMD snapshot not available.")
    else:
        # -- Subtypes --
        subtypes = lgmd_snap.get("subtypes", {})
        dist = subtypes.get("distribution", [])
        if dist:
            st.markdown("#### Subtype Distribution")
            sub_rows = []
            for d in dist:
                sub_rows.append({
                    "Subtype": d["subtype"],
                    "Patients": d["patients"],
                    "%": f"{d['percentage']:.1f}%",
                    "Type": d.get("lgmd_type", ""),
                })
            sub_df = pd.DataFrame(sub_rows)
            st.dataframe(sub_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Subtypes CSV",
                data=sub_df.to_csv(index=False),
                file_name="openmovr_lgmd_subtypes.csv",
                mime="text/csv",
                key="dl_lgmd_subtypes",
            )

        # -- Diagnostic Journey --
        journey = lgmd_snap.get("diagnostic_journey", {})
        if journey.get("available"):
            st.markdown("#### Diagnostic Journey")
            jour_rows = []
            for measure_key, label in [
                ("onset_age", "Age at Symptom Onset (yrs)"),
                ("diagnosis_age", "Age at Diagnosis (yrs)"),
                ("delay", "Diagnostic Delay (yrs)"),
            ]:
                m = journey.get(measure_key, {})
                if m.get("count") and not m.get("suppressed"):
                    jour_rows.append({
                        "Measure": label,
                        "N": m["count"],
                        "Median": m.get("median"),
                        "Q1": m.get("q1"),
                        "Q3": m.get("q3"),
                        "Min": m.get("min"),
                        "Max": m.get("max"),
                    })
            if jour_rows:
                jour_df = pd.DataFrame(jour_rows)
                st.dataframe(jour_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Diagnostic Journey CSV",
                    data=jour_df.to_csv(index=False),
                    file_name="openmovr_lgmd_diagnostic_journey.csv",
                    mime="text/csv",
                    key="dl_lgmd_journey",
                )

        # -- Functional Scores --
        func = lgmd_snap.get("functional_scores", {})
        if func.get("available"):
            st.markdown("#### Functional Outcomes Summary")
            func_rows = []
            for measure_key, label in [
                ("fvc_pct", "FVC % Predicted"),
                ("timed_10m_walk", "Timed 10m Walk (s)"),
            ]:
                m = func.get(measure_key, {})
                if m.get("count"):
                    func_rows.append({
                        "Measure": label,
                        "N": m["count"],
                        "Median": m.get("median"),
                        "Q1": m.get("q1"),
                        "Q3": m.get("q3"),
                        "Min": m.get("min"),
                        "Max": m.get("max"),
                    })
            # Ambulatory status
            amb = func.get("ambulatory_status", {})
            amb_dist = amb.get("distribution", {})
            if amb_dist:
                for status, count in amb_dist.items():
                    func_rows.append({
                        "Measure": f"Ambulatory: {status}",
                        "N": count,
                        "Median": "",
                        "Q1": "",
                        "Q3": "",
                        "Min": "",
                        "Max": "",
                    })
            if func_rows:
                func_df = pd.DataFrame(func_rows)
                st.dataframe(func_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Functional Scores CSV",
                    data=func_df.to_csv(index=False),
                    file_name="openmovr_lgmd_functional_scores.csv",
                    mime="text/csv",
                    key="dl_lgmd_func",
                )

        # -- Medications --
        meds = lgmd_snap.get("medications", {})
        if meds.get("available"):
            st.markdown("#### Medication Utilization")
            med_rows = []
            # Categories
            categories = meds.get("categories", {})
            for cat_name, cat_data in categories.items():
                med_rows.append({
                    "Type": "Category",
                    "Name": cat_name,
                    "Patients": cat_data.get("patients", 0),
                    "% Cohort": f"{cat_data.get('percentage', 0):.1f}%",
                })
            # Top drugs
            top_drugs = meds.get("top_drugs", [])
            for d in top_drugs:
                med_rows.append({
                    "Type": "Drug",
                    "Name": d["drug"],
                    "Patients": d["patients"],
                    "% Cohort": f"{d.get('percentage', 0):.1f}%",
                })
            if med_rows:
                med_df = pd.DataFrame(med_rows)
                st.dataframe(med_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Medications CSV",
                    data=med_df.to_csv(index=False),
                    file_name="openmovr_lgmd_medications.csv",
                    mime="text/csv",
                    key="dl_lgmd_meds",
                )

        # -- State Distribution --
        state_dist = lgmd_snap.get("state_distribution", {})
        states = state_dist.get("states", [])
        if states:
            st.markdown("#### State Distribution")
            state_df = pd.DataFrame(states)
            st.dataframe(state_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download State Distribution CSV",
                data=state_df.to_csv(index=False),
                file_name="openmovr_lgmd_state_distribution.csv",
                mime="text/csv",
                key="dl_lgmd_states",
            )

        # -- Facilities --
        fac_list = lgmd_snap.get("facilities", {}).get("facilities", [])
        if fac_list:
            st.markdown("#### Care Sites")
            fac_df = pd.DataFrame(fac_list)
            st.dataframe(fac_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Care Sites CSV",
                data=fac_df.to_csv(index=False),
                file_name="openmovr_lgmd_care_sites.csv",
                mime="text/csv",
                key="dl_lgmd_sites",
            )

# ===== Tab 2: Patient-level data (parquet only) =====
with tab_patient:
    _DATA_DIR = app_dir / "data"
    _has_parquet = any(_DATA_DIR.glob("*.parquet")) if _DATA_DIR.exists() else False

    if not _has_parquet:
        st.info(
            "Patient-level data requires a live parquet connection and is not "
            "available in snapshot mode."
        )
    else:
        st.warning(
            "Patient-level data contains individual records.  Handle per your "
            "Data Use Agreement.  Do not share outside approved personnel."
        )

        show_patient = st.checkbox("Show patient-level data tables", value=False, key="lgmd_show_patient")

        if show_patient:
            with st.spinner("Loading LGMD patient data..."):
                try:
                    demo_df = pd.read_parquet(_DATA_DIR / "combo_demographics.parquet")
                    diag_df = pd.read_parquet(_DATA_DIR / "combo_diagnosis.parquet")
                    enc_df = pd.read_parquet(_DATA_DIR / "combo_encounters.parquet")

                    # Filter to LGMD
                    if "dx" in diag_df.columns:
                        lgmd_patients = diag_df[diag_df["dx"].str.contains("Limb-girdle|LGMD", case=False, na=False)]["FACPATID"].unique()
                    elif "disease" in demo_df.columns:
                        lgmd_patients = demo_df[demo_df["disease"].str.contains("Limb-girdle|LGMD", case=False, na=False)]["FACPATID"].unique()
                    else:
                        lgmd_patients = demo_df["FACPATID"].unique()

                    demo_lgmd = demo_df[demo_df["FACPATID"].isin(lgmd_patients)]
                    diag_lgmd = diag_df[diag_df["FACPATID"].isin(lgmd_patients)]
                    enc_lgmd = enc_df[enc_df["FACPATID"].isin(lgmd_patients)]

                    # Latest encounter per patient
                    if "encntdt" in enc_lgmd.columns:
                        enc_lgmd = enc_lgmd.copy()
                        enc_lgmd["encntdt"] = pd.to_datetime(enc_lgmd["encntdt"], errors="coerce")
                        enc_latest = enc_lgmd.sort_values("encntdt").groupby("FACPATID").last().reset_index()
                    else:
                        enc_latest = enc_lgmd.groupby("FACPATID").last().reset_index()

                    st.markdown("#### Demographics")
                    st.caption(f"{len(demo_lgmd)} LGMD patients")
                    st.dataframe(demo_lgmd, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Demographics CSV",
                        data=demo_lgmd.to_csv(index=False),
                        file_name="openmovr_lgmd_demographics_patient.csv",
                        mime="text/csv",
                        key="dl_lgmd_demo_pt",
                    )

                    st.markdown("#### Diagnosis")
                    st.dataframe(diag_lgmd, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Diagnosis CSV",
                        data=diag_lgmd.to_csv(index=False),
                        file_name="openmovr_lgmd_diagnosis_patient.csv",
                        mime="text/csv",
                        key="dl_lgmd_diag_pt",
                    )

                    st.markdown("#### Latest Encounters")
                    st.caption(f"{len(enc_latest)} patients with encounter data")
                    st.dataframe(enc_latest, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Encounters CSV",
                        data=enc_latest.to_csv(index=False),
                        file_name="openmovr_lgmd_encounters_patient.csv",
                        mime="text/csv",
                        key="dl_lgmd_enc_pt",
                    )

                    # Medications (if available)
                    med_path = _DATA_DIR / "combo_drugs.parquet"
                    if med_path.exists():
                        med_df = pd.read_parquet(med_path)
                        med_lgmd = med_df[med_df["FACPATID"].isin(lgmd_patients)]
                        st.markdown("#### Medications")
                        st.caption(f"{len(med_lgmd)} medication records")
                        st.dataframe(med_lgmd, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Medications CSV",
                            data=med_lgmd.to_csv(index=False),
                            file_name="openmovr_lgmd_medications_patient.csv",
                            mime="text/csv",
                            key="dl_lgmd_med_pt",
                        )

                except Exception as e:
                    st.error(f"Error loading patient data: {e}")

# ---- Footer ----
render_page_footer()
