"""
SMA Clinical Analytics — DUA-Gated Page

Standalone page requiring provisioned access.  Shows all SMA clinical
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
from components.clinical_summary import render_sma_clinical_summary
from utils.access import require_access
from api.sma import SMAAPI

_logo_path = app_dir / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="SMA Clinical Analytics - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Access gate ----
require_access(
    page_title="SMA Clinical Analytics",
    description=(
        "Clinical summary analytics for the **Spinal Muscular Atrophy (SMA)** cohort.  "
        "Includes HFMSE, CHOP-INTEND, and RULM motor scores, SMA type classification, "
        "SMN2 genetics, therapeutic utilization (Spinraza, Evrysdi, Zolgensma), "
        "respiratory function, and downloadable data tables.\n\n"
        "Available to participating sites, researchers, PAGs, and participants "
        "with an approved Data Use Agreement.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)

# ---- Header ----
render_page_header("SMA Clinical Analytics")

# ---- Clinical summary charts (all downloadable via Plotly modebar) ----
render_sma_clinical_summary()

# ---- Data Tables ----
st.markdown("---")
st.subheader("Data Tables")

tab_summary, tab_patient = st.tabs(["Summary Tables", "Participant-Level Data"])

# ===== Tab 1: Summary tables from snapshot =====
with tab_summary:
    sma_snap = SMAAPI.get_snapshot()
    if not sma_snap:
        st.warning("SMA snapshot not available.")
    else:
        # -- Motor Scores --
        motor = sma_snap.get("motor_scores", {})
        if motor.get("available"):
            for score_key, score_label, max_score in [
                ("hfmse", "HFMSE", 66),
                ("chop_intend", "CHOP-INTEND", 64),
                ("rulm", "RULM", 37),
            ]:
                score_data = motor.get(score_key, {})
                ts = score_data.get("total_score", {})
                if ts.get("count"):
                    st.markdown(f"#### {score_label} Scores")
                    # By SMA type table
                    by_type = score_data.get("by_sma_type", {})
                    if by_type:
                        type_rows = []
                        for sma_type, stats in by_type.items():
                            type_rows.append({
                                "SMA Type": sma_type,
                                "N": stats.get("count", 0),
                                "Median": stats.get("median"),
                                "Mean": f"{stats.get('mean', 0):.1f}",
                                "Min": stats.get("min"),
                                "Max": stats.get("max"),
                            })
                        type_df = pd.DataFrame(type_rows)
                        st.dataframe(type_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            f"Download {score_label} by SMA Type CSV",
                            data=type_df.to_csv(index=False),
                            file_name=f"openmovr_sma_{score_key}_by_type.csv",
                            mime="text/csv",
                            key=f"dl_sma_{score_key}_type",
                        )

                    # Longitudinal table
                    longitudinal = score_data.get("longitudinal", [])
                    if longitudinal:
                        st.markdown(f"**Longitudinal {score_label} (by years since enrollment)**")
                        long_df = pd.DataFrame(longitudinal)
                        st.dataframe(long_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            f"Download {score_label} Longitudinal CSV",
                            data=long_df.to_csv(index=False),
                            file_name=f"openmovr_sma_{score_key}_longitudinal.csv",
                            mime="text/csv",
                            key=f"dl_sma_{score_key}_long",
                        )

        # -- Classification --
        classification = sma_snap.get("classification", {})
        if classification.get("available"):
            sma_type = classification.get("sma_type", {})
            dist = sma_type.get("distribution", {})
            if dist:
                st.markdown("#### SMA Type Distribution")
                type_rows = [{"SMA Type": k, "Participants": v} for k, v in dist.items()]
                type_df = pd.DataFrame(type_rows)
                st.dataframe(type_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download SMA Type CSV",
                    data=type_df.to_csv(index=False),
                    file_name="openmovr_sma_type_distribution.csv",
                    mime="text/csv",
                    key="dl_sma_type",
                )

        # -- Therapeutics --
        tx = sma_snap.get("therapeutics", {})
        if tx.get("available"):
            st.markdown("#### SMA Therapeutics")
            drug_rows = []
            for drug_name, drug_data in tx.get("sma_drugs", {}).items():
                drug_rows.append({
                    "Drug": drug_name,
                    "Participants": "<11" if drug_data.get("suppressed") else drug_data.get("count", 0),
                    "% of Cohort": f"{drug_data.get('percentage', 0):.1f}%",
                })
            if drug_rows:
                drug_df = pd.DataFrame(drug_rows)
                st.dataframe(drug_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download SMA Therapeutics CSV",
                    data=drug_df.to_csv(index=False),
                    file_name="openmovr_sma_therapeutics.csv",
                    mime="text/csv",
                    key="dl_sma_tx",
                )

            # Therapy by type cross-tab
            therapy_by_type = tx.get("therapy_by_sma_type", {})
            if therapy_by_type:
                st.markdown("**Therapy by SMA Type**")
                cross_rows = []
                for sma_type, data in therapy_by_type.items():
                    cross_rows.append({
                        "SMA Type": sma_type,
                        "Spinraza": data.get("Spinraza (nusinersen)", 0),
                        "Evrysdi": data.get("Evrysdi (risdiplam)", 0),
                        "Zolgensma": data.get("Zolgensma (onasemnogene)", 0),
                        "Total on Therapy": data.get("total_on_therapy", 0),
                        "Total Participants": data.get("total_patients", 0),
                    })
                cross_df = pd.DataFrame(cross_rows)
                st.dataframe(cross_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download Therapy by Type CSV",
                    data=cross_df.to_csv(index=False),
                    file_name="openmovr_sma_therapy_by_type.csv",
                    mime="text/csv",
                    key="dl_sma_tx_type",
                )

        # -- Genetics --
        genetics = sma_snap.get("genetics", {})
        if genetics.get("available"):
            smn2 = genetics.get("smn2_copy_number", {})
            dist = smn2.get("distribution", {})
            if dist:
                st.markdown("#### SMN2 Copy Number")
                smn2_rows = [{"SMN2 Copies": k, "Participants": v} for k, v in dist.items()]
                smn2_df = pd.DataFrame(smn2_rows)
                st.dataframe(smn2_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download SMN2 CSV",
                    data=smn2_df.to_csv(index=False),
                    file_name="openmovr_sma_smn2.csv",
                    mime="text/csv",
                    key="dl_sma_smn2",
                )

        # -- State Distribution --
        state_dist = sma_snap.get("state_distribution", {})
        states = state_dist.get("states", [])
        if states:
            st.markdown("#### State Distribution")
            state_df = pd.DataFrame(states)
            st.dataframe(state_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download State Distribution CSV",
                data=state_df.to_csv(index=False),
                file_name="openmovr_sma_state_distribution.csv",
                mime="text/csv",
                key="dl_sma_states",
            )

        # -- Facilities --
        fac_list = sma_snap.get("facilities", {}).get("facilities", [])
        if fac_list:
            st.markdown("#### Care Sites")
            fac_df = pd.DataFrame(fac_list)
            st.dataframe(fac_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Care Sites CSV",
                data=fac_df.to_csv(index=False),
                file_name="openmovr_sma_care_sites.csv",
                mime="text/csv",
                key="dl_sma_sites",
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

        show_patient = st.checkbox("Show participant-level data tables", value=False, key="sma_show_patient")

        if show_patient:
            with st.spinner("Loading SMA participant data..."):
                try:
                    demo_df = pd.read_parquet(_DATA_DIR / "combo_demographics.parquet")
                    diag_df = pd.read_parquet(_DATA_DIR / "combo_diagnosis.parquet")
                    enc_df = pd.read_parquet(_DATA_DIR / "combo_encounters.parquet")

                    # Filter to SMA
                    if "dstype" in demo_df.columns:
                        sma_patients = demo_df[demo_df["dstype"] == "SMA"]["FACPATID"].unique()
                    elif "smaclass" in diag_df.columns:
                        sma_patients = diag_df[diag_df["smaclass"].notna()]["FACPATID"].unique()
                    else:
                        sma_patients = demo_df["FACPATID"].unique()

                    demo_sma = demo_df[demo_df["FACPATID"].isin(sma_patients)]
                    diag_sma = diag_df[diag_df["FACPATID"].isin(sma_patients)]
                    enc_sma = enc_df[enc_df["FACPATID"].isin(sma_patients)]

                    # Latest encounter per patient
                    if "encntdt" in enc_sma.columns:
                        enc_sma = enc_sma.copy()
                        enc_sma["encntdt"] = pd.to_datetime(enc_sma["encntdt"], errors="coerce")
                        enc_latest = enc_sma.sort_values("encntdt").groupby("FACPATID").last().reset_index()
                    else:
                        enc_latest = enc_sma.groupby("FACPATID").last().reset_index()

                    st.markdown("#### Demographics")
                    st.caption(f"{len(demo_sma)} SMA participants")
                    st.dataframe(demo_sma, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Demographics CSV",
                        data=demo_sma.to_csv(index=False),
                        file_name="openmovr_sma_demographics_patient.csv",
                        mime="text/csv",
                        key="dl_sma_demo_pt",
                    )

                    st.markdown("#### Diagnosis")
                    st.dataframe(diag_sma, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Diagnosis CSV",
                        data=diag_sma.to_csv(index=False),
                        file_name="openmovr_sma_diagnosis_patient.csv",
                        mime="text/csv",
                        key="dl_sma_diag_pt",
                    )

                    st.markdown("#### Latest Encounters")
                    st.caption(f"{len(enc_latest)} participants with encounter data")
                    st.dataframe(enc_latest, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Encounters CSV",
                        data=enc_latest.to_csv(index=False),
                        file_name="openmovr_sma_encounters_patient.csv",
                        mime="text/csv",
                        key="dl_sma_enc_pt",
                    )

                    # Medications (if available)
                    med_path = _DATA_DIR / "encounter_medication_rg.parquet"
                    if not med_path.exists():
                        med_path = _DATA_DIR / "combo_drugs.parquet"
                    if med_path.exists():
                        med_df = pd.read_parquet(med_path)
                        med_sma = med_df[med_df["FACPATID"].isin(sma_patients)]
                        st.markdown("#### Medications")
                        st.caption(f"{len(med_sma)} medication records")
                        st.dataframe(med_sma, use_container_width=True, hide_index=True)
                        st.download_button(
                            "Download Medications CSV",
                            data=med_sma.to_csv(index=False),
                            file_name="openmovr_sma_medications_patient.csv",
                            mime="text/csv",
                            key="dl_sma_med_pt",
                        )

                except Exception as e:
                    st.error(f"Error loading participant data: {e}")

# ---- Footer ----
render_page_footer()
