"""
DMD Clinical Analytics — DUA-Gated Page

Standalone page requiring provisioned access.  Shows all DMD clinical
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
from components.clinical_summary import render_dmd_clinical_summary
from utils.access import require_access
from api.dmd import DMDAPI

_logo_path = app_dir / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="DMD Clinical Analytics - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Access gate ----
require_access(
    page_title="DMD Clinical Analytics",
    description=(
        "Clinical summary analytics for the **Duchenne Muscular Dystrophy (DMD)** cohort.  "
        "Includes exon-skipping therapy utilization, functional outcomes, genetic profiles, "
        "and downloadable data tables.\n\n"
        "Available to participating sites, researchers, PAGs, and participants "
        "with an approved Data Use Agreement.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)

# ---- Header ----
render_page_header("DMD Clinical Analytics")

# ---- Clinical summary charts (all downloadable via Plotly modebar) ----
render_dmd_clinical_summary()

# ---- Data Tables ----
st.markdown("---")
st.subheader("Data Tables")

tab_summary, tab_patient = st.tabs(["Summary Tables", "Participant-Level Data"])

# ===== Tab 1: Summary tables from snapshot =====
with tab_summary:
    dmd_snap = DMDAPI.get_snapshot()
    if not dmd_snap:
        st.warning("DMD snapshot not available.")
    else:
        # -- Therapeutics --
        tx = dmd_snap.get("therapeutics", {})
        drugs = tx.get("drugs", [])
        if drugs:
            st.markdown("#### Exon-Skipping Therapeutics")
            tx_rows = []
            for d in drugs:
                on_v = d.get("on_therapy", {})
                am_not = d.get("amenable_not_on_therapy", {})
                am_tot = d.get("total_amenable", {})
                tx_rows.append({
                    "Drug": d["drug_name"],
                    "On Therapy": "<11" if on_v.get("suppressed") else on_v.get("count", 0),
                    "Amenable (not on)": "<11" if am_not.get("suppressed") else am_not.get("count", 0),
                    "Total Amenable": "<11" if am_tot.get("suppressed") else am_tot.get("count", 0),
                    "% Cohort": f"{d.get('pct_of_cohort', 0):.1f}%",
                })
            tx_df = pd.DataFrame(tx_rows)
            st.dataframe(tx_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Therapeutics CSV",
                data=tx_df.to_csv(index=False),
                file_name="openmovr_dmd_therapeutics.csv",
                mime="text/csv",
                key="dl_dmd_tx",
            )

        # -- Steroids --
        steroids = dmd_snap.get("steroids", {})
        if steroids.get("available"):
            st.markdown("#### Glucocorticoid Use")
            steroid_rows = []
            for period_key, period_label in [
                ("glcouse_first_encounter", "First Encounter"),
                ("glcouse_last_encounter", "Last Encounter"),
            ]:
                period = steroids.get(period_key, {})
                dist = period.get("distribution", {})
                for status, count in dist.items():
                    steroid_rows.append({
                        "Period": period_label,
                        "Status": status,
                        "Participants": count,
                    })
            if steroid_rows:
                st_df = pd.DataFrame(steroid_rows)
                st.dataframe(st_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Steroid Use CSV",
                    data=st_df.to_csv(index=False),
                    file_name="openmovr_dmd_steroids.csv",
                    mime="text/csv",
                    key="dl_dmd_steroids",
                )

        # -- Genetics --
        genetics = dmd_snap.get("genetics", {})
        if genetics.get("available"):
            st.markdown("#### Genetic & Mutation Profile")
            gen_rows = []
            mt = genetics.get("mutation_type", {})
            for d in mt.get("distribution", []):
                gen_rows.append({
                    "Category": "Mutation Type",
                    "Label": d.get("label", ""),
                    "Count": "<11" if d.get("suppressed") else d.get("count", 0),
                })
            ft = genetics.get("frame_type", {})
            for d in ft.get("distribution", []):
                gen_rows.append({
                    "Category": "Frame Type",
                    "Label": d.get("label", ""),
                    "Count": "<11" if d.get("suppressed") else d.get("count", 0),
                })
            gc = genetics.get("genetic_confirmation", {})
            for status, count in gc.get("distribution", {}).items():
                gen_rows.append({
                    "Category": "Genetic Confirmation",
                    "Label": status,
                    "Count": count,
                })
            if gen_rows:
                gen_df = pd.DataFrame(gen_rows)
                st.dataframe(gen_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Genetics CSV",
                    data=gen_df.to_csv(index=False),
                    file_name="openmovr_dmd_genetics.csv",
                    mime="text/csv",
                    key="dl_dmd_genetics",
                )

        # -- State Distribution --
        state_dist = dmd_snap.get("state_distribution", {})
        states = state_dist.get("states", [])
        if states:
            st.markdown("#### State Distribution")
            state_df = pd.DataFrame(states)
            st.dataframe(state_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download State Distribution CSV",
                data=state_df.to_csv(index=False),
                file_name="openmovr_dmd_state_distribution.csv",
                mime="text/csv",
                key="dl_dmd_states",
            )

        # -- Functional Scores --
        func = dmd_snap.get("functional_scores", {})
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
            loa = func.get("loss_of_ambulation", {})
            if loa:
                age_data = loa.get("age_at_loss_years", {})
                func_rows.append({
                    "Measure": "Age at Loss of Ambulation (yrs)",
                    "N": loa.get("total_with_data", 0),
                    "Median": age_data.get("median"),
                    "Q1": age_data.get("q1"),
                    "Q3": age_data.get("q3"),
                    "Min": age_data.get("min"),
                    "Max": age_data.get("max"),
                })
            if func_rows:
                func_df = pd.DataFrame(func_rows)
                st.dataframe(func_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Functional Scores CSV",
                    data=func_df.to_csv(index=False),
                    file_name="openmovr_dmd_functional_scores.csv",
                    mime="text/csv",
                    key="dl_dmd_func",
                )

        # -- Facilities --
        fac_list = dmd_snap.get("facilities", {}).get("facilities", [])
        if fac_list:
            st.markdown("#### Care Sites")
            fac_df = pd.DataFrame(fac_list)
            st.dataframe(fac_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Care Sites CSV",
                data=fac_df.to_csv(index=False),
                file_name="openmovr_dmd_care_sites.csv",
                mime="text/csv",
                key="dl_dmd_sites",
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

        show_patient = st.checkbox("Show participant-level data tables", value=False, key="dmd_show_patient")

        if show_patient:
            with st.spinner("Loading DMD participant data..."):
                try:
                    demo_df = pd.read_parquet(_DATA_DIR / "combo_demographics.parquet")
                    diag_df = pd.read_parquet(_DATA_DIR / "combo_diagnosis.parquet")
                    enc_df = pd.read_parquet(_DATA_DIR / "combo_encounters.parquet")

                    # Filter to DMD
                    dmd_diseases = ["Duchenne muscular dystrophy"]
                    if "dx" in diag_df.columns:
                        dmd_patients = diag_df[diag_df["dx"].str.contains("Duchenne", case=False, na=False)]["FACPATID"].unique()
                    elif "disease" in demo_df.columns:
                        dmd_patients = demo_df[demo_df["disease"].str.contains("Duchenne", case=False, na=False)]["FACPATID"].unique()
                    else:
                        dmd_patients = demo_df["FACPATID"].unique()

                    demo_dmd = demo_df[demo_df["FACPATID"].isin(dmd_patients)]
                    diag_dmd = diag_df[diag_df["FACPATID"].isin(dmd_patients)]
                    enc_dmd = enc_df[enc_df["FACPATID"].isin(dmd_patients)]

                    # Latest encounter per patient
                    if "encntdt" in enc_dmd.columns:
                        enc_dmd = enc_dmd.copy()
                        enc_dmd["encntdt"] = pd.to_datetime(enc_dmd["encntdt"], errors="coerce")
                        enc_latest = enc_dmd.sort_values("encntdt").groupby("FACPATID").last().reset_index()
                    else:
                        enc_latest = enc_dmd.groupby("FACPATID").last().reset_index()

                    st.markdown("#### Demographics")
                    st.caption(f"{len(demo_dmd)} DMD participants")
                    st.dataframe(demo_dmd, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Demographics CSV",
                        data=demo_dmd.to_csv(index=False),
                        file_name="openmovr_dmd_demographics_patient.csv",
                        mime="text/csv",
                        key="dl_dmd_demo_pt",
                    )

                    st.markdown("#### Diagnosis")
                    st.dataframe(diag_dmd, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Diagnosis CSV",
                        data=diag_dmd.to_csv(index=False),
                        file_name="openmovr_dmd_diagnosis_patient.csv",
                        mime="text/csv",
                        key="dl_dmd_diag_pt",
                    )

                    st.markdown("#### Latest Encounters")
                    st.caption(f"{len(enc_latest)} participants with encounter data")
                    st.dataframe(enc_latest, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Encounters CSV",
                        data=enc_latest.to_csv(index=False),
                        file_name="openmovr_dmd_encounters_patient.csv",
                        mime="text/csv",
                        key="dl_dmd_enc_pt",
                    )

                    # Medications (if available)
                    med_path = _DATA_DIR / "combo_drugs.parquet"
                    if med_path.exists():
                        med_df = pd.read_parquet(med_path)
                        med_dmd = med_df[med_df["FACPATID"].isin(dmd_patients)]
                        st.markdown("#### Medications")
                        st.caption(f"{len(med_dmd)} medication records")
                        st.dataframe(med_dmd, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Medications CSV",
                            data=med_dmd.to_csv(index=False),
                            file_name="openmovr_dmd_medications_patient.csv",
                            mime="text/csv",
                            key="dl_dmd_med_pt",
                        )

                except Exception as e:
                    st.error(f"Error loading participant data: {e}")

# ---- Footer ----
render_page_footer()
