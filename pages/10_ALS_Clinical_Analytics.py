"""
ALS Clinical Analytics — DUA-Gated Page

Standalone page requiring provisioned access.  Shows all ALS clinical
summary figures (downloadable via Plotly modebar) plus summary and
participant-level data tables with CSV export.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
from config.settings import PAGE_ICON
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer, render_page_header
from components.clinical_summary import render_als_clinical_summary
from utils.access import require_access
from api.als import ALSAPI

_logo_path = app_dir / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="ALS Clinical Analytics - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Access gate ----
require_access(
    page_title="ALS Clinical Analytics",
    description=(
        "Clinical summary analytics for the **Amyotrophic Lateral Sclerosis (ALS)** cohort.  "
        "Includes ALSFRS-R functional scores, disease progression milestones, respiratory function, "
        "medication utilization, and downloadable data tables.\n\n"
        "Available to participating sites, researchers, PAGs, and participants "
        "with an approved Data Use Agreement.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)

# ---- Header ----
render_page_header("ALS Clinical Analytics")

# ---- Clinical summary charts (all downloadable via Plotly modebar) ----
render_als_clinical_summary()

# ---- Data Tables ----
st.markdown("---")
st.subheader("Data Tables")

tab_summary, tab_patient = st.tabs(["Summary Tables", "Participant-Level Data"])

# ===== Tab 1: Summary tables from snapshot =====
with tab_summary:
    als_snap = ALSAPI.get_snapshot()
    if not als_snap:
        st.warning("ALS snapshot not available.")
    else:
        # -- ALSFRS-R --
        alsfrs = als_snap.get("alsfrs", {})
        if alsfrs.get("available"):
            st.markdown("#### ALSFRS-R Scores")
            # Severity band table
            severity = alsfrs.get("severity_bands", [])
            if severity:
                sev_rows = [{
                    "Severity Band": b["label"],
                    "Participants": "<11" if b.get("suppressed") else b.get("count", 0),
                    "% of Cohort": f"{b.get('percentage', 0):.1f}%",
                } for b in severity]
                sev_df = pd.DataFrame(sev_rows)
                st.dataframe(sev_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download ALSFRS-R Severity CSV",
                    data=sev_df.to_csv(index=False),
                    file_name="openmovr_als_alsfrs_severity.csv",
                    mime="text/csv",
                    key="dl_als_alsfrs",
                )

            # Longitudinal table
            longitudinal = alsfrs.get("total_score", {}).get("longitudinal", [])
            if longitudinal:
                st.markdown("**Longitudinal ALSFRS-R (by years since enrollment)**")
                long_df = pd.DataFrame(longitudinal)
                st.dataframe(long_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Longitudinal CSV",
                    data=long_df.to_csv(index=False),
                    file_name="openmovr_als_alsfrs_longitudinal.csv",
                    mime="text/csv",
                    key="dl_als_long",
                )

        # -- El Escorial --
        dx = als_snap.get("diagnosis", {})
        ee = dx.get("el_escorial", {})
        if ee:
            st.markdown("#### El Escorial Classification")
            ee_dist = ee.get("distribution", {})
            if ee_dist:
                ee_rows = [{"Classification": k, "Participants": v} for k, v in ee_dist.items()]
                ee_df = pd.DataFrame(ee_rows)
                st.dataframe(ee_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download El Escorial CSV",
                    data=ee_df.to_csv(index=False),
                    file_name="openmovr_als_el_escorial.csv",
                    mime="text/csv",
                    key="dl_als_ee",
                )

        # -- Medications --
        meds = als_snap.get("medications", {})
        if meds.get("available"):
            st.markdown("#### Medication Utilization")
            med_rows = []
            for drug_name, drug_data in meds.get("als_drugs", {}).items():
                med_rows.append({
                    "Drug": drug_name,
                    "Participants": "<11" if drug_data.get("suppressed") else drug_data.get("count", 0),
                    "% of Cohort": f"{drug_data.get('percentage', 0):.1f}%",
                })
            top_drugs = meds.get("top_drugs", [])
            for d in top_drugs:
                if d["drug"] not in [r["Drug"].split(" (")[0] for r in med_rows]:
                    med_rows.append({
                        "Drug": d["drug"],
                        "Participants": d["patients"],
                        "% of Cohort": f"{d.get('percentage', 0):.1f}%",
                    })
            if med_rows:
                med_df = pd.DataFrame(med_rows)
                st.dataframe(med_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Medications CSV",
                    data=med_df.to_csv(index=False),
                    file_name="openmovr_als_medications.csv",
                    mime="text/csv",
                    key="dl_als_meds",
                )

        # -- State Distribution --
        state_dist = als_snap.get("state_distribution", {})
        states = state_dist.get("states", [])
        if states:
            st.markdown("#### State Distribution")
            state_df = pd.DataFrame(states)
            st.dataframe(state_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download State Distribution CSV",
                data=state_df.to_csv(index=False),
                file_name="openmovr_als_state_distribution.csv",
                mime="text/csv",
                key="dl_als_states",
            )

        # -- Milestones --
        ms = als_snap.get("milestones", {})
        if ms.get("available"):
            st.markdown("#### Disease Milestones")
            ms_rows = []
            for key, label in [
                ("onset_age", "Age at Symptom Onset (yrs)"),
                ("diagnosis_age", "Age at Diagnosis (yrs)"),
                ("diagnostic_delay", "Diagnostic Delay (yrs)"),
            ]:
                m = ms.get(key, {})
                if m.get("count"):
                    ms_rows.append({
                        "Measure": label,
                        "N": m["count"],
                        "Median": m.get("median"),
                        "Mean": m.get("mean"),
                        "Min": m.get("min"),
                        "Max": m.get("max"),
                    })
            if ms_rows:
                ms_df = pd.DataFrame(ms_rows)
                st.dataframe(ms_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Milestones CSV",
                    data=ms_df.to_csv(index=False),
                    file_name="openmovr_als_milestones.csv",
                    mime="text/csv",
                    key="dl_als_ms",
                )

        # -- Facilities --
        fac_list = als_snap.get("facilities", {}).get("facilities", [])
        if fac_list:
            st.markdown("#### Care Sites")
            fac_df = pd.DataFrame(fac_list)
            st.dataframe(fac_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Care Sites CSV",
                data=fac_df.to_csv(index=False),
                file_name="openmovr_als_care_sites.csv",
                mime="text/csv",
                key="dl_als_sites",
            )

# ===== Tab 2: Participant-level data (parquet only) =====
with tab_patient:
    _DATA_DIR = app_dir / "data"
    _has_parquet = any(_DATA_DIR.glob("*.parquet")) if _DATA_DIR.exists() else False

    if not _has_parquet:
        st.info(
            "Participant-level data requires a live parquet connection and is not "
            "available in snapshot mode."
        )
    else:
        st.warning(
            "Participant-level data contains individual records.  Handle per your "
            "Data Use Agreement.  Do not share outside approved personnel."
        )

        show_patient = st.checkbox("Show participant-level data tables", value=False, key="als_show_patient")

        if show_patient:
            with st.spinner("Loading ALS participant data..."):
                try:
                    demo_df = pd.read_parquet(_DATA_DIR / "combo_demographics.parquet")
                    diag_df = pd.read_parquet(_DATA_DIR / "combo_diagnosis.parquet")
                    enc_df = pd.read_parquet(_DATA_DIR / "combo_encounters.parquet")

                    # Filter to ALS
                    if "dstype" in demo_df.columns:
                        als_patients = demo_df[demo_df["dstype"] == "ALS"]["FACPATID"].unique()
                    elif "dx" in diag_df.columns:
                        als_patients = diag_df[diag_df["dx"].str.contains("ALS", case=False, na=False)]["FACPATID"].unique()
                    else:
                        als_patients = demo_df["FACPATID"].unique()

                    demo_als = demo_df[demo_df["FACPATID"].isin(als_patients)]
                    diag_als = diag_df[diag_df["FACPATID"].isin(als_patients)]
                    enc_als = enc_df[enc_df["FACPATID"].isin(als_patients)]

                    # Latest encounter per patient
                    if "encntdt" in enc_als.columns:
                        enc_als = enc_als.copy()
                        enc_als["encntdt"] = pd.to_datetime(enc_als["encntdt"], errors="coerce")
                        enc_latest = enc_als.sort_values("encntdt").groupby("FACPATID").last().reset_index()
                    else:
                        enc_latest = enc_als.groupby("FACPATID").last().reset_index()

                    st.markdown("#### Demographics")
                    st.caption(f"{len(demo_als)} ALS participants")
                    st.dataframe(demo_als, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Demographics CSV",
                        data=demo_als.to_csv(index=False),
                        file_name="openmovr_als_demographics_patient.csv",
                        mime="text/csv",
                        key="dl_als_demo_pt",
                    )

                    st.markdown("#### Diagnosis")
                    st.dataframe(diag_als, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Diagnosis CSV",
                        data=diag_als.to_csv(index=False),
                        file_name="openmovr_als_diagnosis_patient.csv",
                        mime="text/csv",
                        key="dl_als_diag_pt",
                    )

                    st.markdown("#### Latest Encounters")
                    st.caption(f"{len(enc_latest)} participants with encounter data")
                    st.dataframe(enc_latest, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Encounters CSV",
                        data=enc_latest.to_csv(index=False),
                        file_name="openmovr_als_encounters_patient.csv",
                        mime="text/csv",
                        key="dl_als_enc_pt",
                    )

                    # Medications (if available)
                    med_path = _DATA_DIR / "encounter_medication_rg.parquet"
                    if not med_path.exists():
                        med_path = _DATA_DIR / "combo_drugs.parquet"
                    if med_path.exists():
                        med_df = pd.read_parquet(med_path)
                        med_als = med_df[med_df["FACPATID"].isin(als_patients)]
                        st.markdown("#### Medications")
                        st.caption(f"{len(med_als)} medication records")
                        st.dataframe(med_als, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Medications CSV",
                            data=med_als.to_csv(index=False),
                            file_name="openmovr_als_medications_patient.csv",
                            mime="text/csv",
                            key="dl_als_med_pt",
                        )

                except Exception as e:
                    st.error(f"Error loading participant data: {e}")

# ---- Footer ----
render_page_footer()
