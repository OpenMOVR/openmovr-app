"""
Download Center

Provisioned-access page for downloading snapshot data as CSV/JSON.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
import pandas as pd
from config.settings import PAGE_ICON, LOGO_PNG
from utils.access import require_access

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="Download Center - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

# Branding CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    [data-testid="stSidebarNav"]::before {
        content: "OpenMOVR App"; display: block; font-size: 1.4em;
        font-weight: bold; color: #1E88E5; text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    [data-testid="stSidebarNav"]::after {
        content: "MOVR Data Hub | MOVR 1.0"; display: block;
        font-size: 0.8em; color: #666; text-align: center;
        padding-bottom: 1rem; margin-bottom: 1rem; border-bottom: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    if LOGO_PNG.exists():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(str(LOGO_PNG), width=160)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

# ---- Access gate ----
require_access(
    page_title="Download Center",
    description=(
        "Export aggregated statistics, gene therapy breakdowns, and clinical "
        "data summaries for your own analyses.  Free for researchers, sites, "
        "and patients with a DUA.  Industry access requires a commercial license.\n\n"
        "**[Request Access](https://mdausa.tfaforms.net/389761)**"
    ),
)

# ---- Load snapshot ----
snapshot_path = Path(__file__).parent.parent / "stats" / "database_snapshot.json"
if not snapshot_path.exists():
    st.error("Snapshot not found.  Run `python scripts/generate_stats_snapshot.py` first.")
    st.stop()

with open(snapshot_path) as f:
    snapshot = json.load(f)

metadata = snapshot["metadata"]
st.caption(f"Snapshot generated: {metadata['generated_timestamp']}")

st.markdown("---")

# ---- 1. Disease Distribution ----
st.subheader("Disease Distribution")

diseases = snapshot["disease_distribution"]["diseases"]
if diseases:
    disease_df = pd.DataFrame(diseases)
    if "percentage" in disease_df.columns:
        disease_df["percentage"] = disease_df["percentage"].apply(lambda x: f"{x:.1f}%")
    if "patient_count" in disease_df.columns:
        disease_df["patient_count"] = disease_df["patient_count"].apply(lambda x: f"{x:,}")
    disease_df = disease_df.rename(columns={
        "disease": "Disease",
        "patient_count": "Patients",
        "percentage": "%",
    })
    st.dataframe(disease_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Disease Distribution CSV",
        data=pd.DataFrame(diseases).to_csv(index=False),
        file_name="openmovr_disease_distribution.csv",
        mime="text/csv",
        key="dl_disease",
    )

# ---- 2. Gene Therapy / Advanced Therapies ----
st.markdown("---")
st.subheader("Advanced Therapies by Disease")

meds = snapshot.get("clinical_availability", {}).get("medications", {})
gt = meds.get("gene_therapy_by_disease", {})
if gt:
    gt_rows = []
    for disease, dinfo in gt.items():
        for t in dinfo.get("treatments", []):
            gt_rows.append({
                "Disease": disease,
                "Category": t["category"],
                "Treatment": t["label"],
                "Patients": t["patients"],
                "Records": t["records"],
            })
    gt_df = pd.DataFrame(gt_rows)
    st.dataframe(gt_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Advanced Therapies CSV",
        data=gt_df.to_csv(index=False),
        file_name="openmovr_advanced_therapies.csv",
        mime="text/csv",
        key="dl_gene_therapy",
    )
else:
    st.caption("Gene therapy data not available in snapshot.")

# ---- 3. Top Medications ----
st.markdown("---")
st.subheader("Top Prescribed Medications")

top_meds = meds.get("top_medications", [])
if top_meds:
    top_df = pd.DataFrame(top_meds).rename(columns={
        "name": "Medication",
        "patients": "Patients",
        "records": "Records",
    })
    st.dataframe(top_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Top Medications CSV",
        data=top_df.to_csv(index=False),
        file_name="openmovr_top_medications.csv",
        mime="text/csv",
        key="dl_top_meds",
    )

# ---- 4. Clinical Availability ----
st.markdown("---")
st.subheader("Clinical Data Availability")

clinical = snapshot.get("clinical_availability", {})
avail_rows = []

func_scores = clinical.get("functional_scores", {})
for key, info in func_scores.items():
    avail_rows.append({
        "Domain": "Functional Assessment",
        "Measure": info.get("label", key),
        "Participants": info["patients"],
        "Data Points": info.get("data_points", 0),
        "Longitudinal (2+)": info.get("patients_longitudinal", 0),
    })

timed = clinical.get("timed_tests", {})
timed_labels = {"walk_run_10m": "10m Walk/Run", "stair_climb": "Stair Climb", "rise_from_supine": "Rise from Supine"}
for key, label in timed_labels.items():
    info = timed.get(key, {})
    if isinstance(info, dict):
        avail_rows.append({
            "Domain": "Timed Test",
            "Measure": label,
            "Participants": info.get("patients", 0),
            "Data Points": info.get("data_points", 0),
            "Longitudinal (2+)": "",
        })

pulm = clinical.get("pulmonary", {})
for key, label in [("pft_performed", "PFTs Performed"), ("fvc", "FVC"), ("fev1", "FEV1")]:
    avail_rows.append({
        "Domain": "Pulmonary",
        "Measure": label,
        "Participants": pulm.get(key, 0),
        "Data Points": "",
        "Longitudinal (2+)": "",
    })

cardiac = clinical.get("cardiac", {})
for key, label in [("ecg", "ECG"), ("echo", "Echo"), ("cardiomyopathy", "Cardiomyopathy")]:
    avail_rows.append({
        "Domain": "Cardiology",
        "Measure": label,
        "Participants": cardiac.get(key, 0),
        "Data Points": "",
        "Longitudinal (2+)": "",
    })

trials = clinical.get("clinical_trials", {})
if trials.get("patient_breakdown"):
    for cat, pts in trials["patient_breakdown"].items():
        avail_rows.append({
            "Domain": "Clinical Trials",
            "Measure": cat,
            "Participants": pts,
            "Data Points": "",
            "Longitudinal (2+)": "",
        })

hosp = clinical.get("hospitalizations", {})
if hosp:
    avail_rows.append({
        "Domain": "Hospitalizations",
        "Measure": "Events",
        "Participants": hosp.get("patients", 0),
        "Data Points": hosp.get("records", 0),
        "Longitudinal (2+)": "",
    })

surg = clinical.get("surgeries", {})
if surg and surg.get("patients", 0) > 0:
    avail_rows.append({
        "Domain": "Surgeries",
        "Measure": "Procedures",
        "Participants": surg.get("patients", 0),
        "Data Points": surg.get("records", 0),
        "Longitudinal (2+)": "",
    })

if avail_rows:
    avail_df = pd.DataFrame(avail_rows)
    st.dataframe(avail_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Clinical Availability CSV",
        data=avail_df.to_csv(index=False),
        file_name="openmovr_clinical_availability.csv",
        mime="text/csv",
        key="dl_clinical",
    )

# ---- 5. Longitudinal Summary ----
st.markdown("---")
st.subheader("Longitudinal Summary by Disease")

longitudinal = snapshot.get("longitudinal", {})
by_disease = longitudinal.get("by_disease", {})
if by_disease:
    long_rows = []
    for ds, info in by_disease.items():
        long_rows.append({
            "Disease": ds,
            "Patients": info["patients"],
            "Encounters": info["encounters"],
            "Mean / Patient": info["mean_per_patient"],
            "With 3+ Visits": info["patients_3plus"],
        })
    long_df = pd.DataFrame(long_rows)
    st.dataframe(long_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Longitudinal Summary CSV",
        data=long_df.to_csv(index=False),
        file_name="openmovr_longitudinal_summary.csv",
        mime="text/csv",
        key="dl_longitudinal",
    )

# ---- 6. Facility Summary (with site names from local metadata) ----
st.markdown("---")
st.subheader("Facility Summary")

all_facs = snapshot.get("facilities", {}).get("all_facilities", [])
site_locs = snapshot.get("facilities", {}).get("site_locations", [])

if all_facs:
    fac_df = pd.DataFrame(all_facs)

    # Enrich with city/state from site_locations (no facility names in snapshot)
    loc_lookup = {s["facility_id"]: s for s in site_locs} if site_locs else {}
    fac_df["City"] = fac_df["FACILITY_DISPLAY_ID"].astype(str).map(
        lambda fid: loc_lookup.get(fid, {}).get("city", "")
    )
    fac_df["State"] = fac_df["FACILITY_DISPLAY_ID"].astype(str).map(
        lambda fid: loc_lookup.get(fid, {}).get("state", "")
    )

    fac_df = fac_df.rename(columns={
        "FACILITY_DISPLAY_ID": "Site ID",
        "patient_count": "Patients",
    })
    display_cols = ["Site ID", "City", "State", "Patients"]
    display_cols = [c for c in display_cols if c in fac_df.columns]
    fac_df = fac_df[display_cols]

    st.dataframe(fac_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download Facility Summary CSV",
        data=fac_df.to_csv(index=False),
        file_name="openmovr_facility_summary.csv",
        mime="text/csv",
        key="dl_facilities",
    )

# ---- 7. Full Snapshot JSON ----
st.markdown("---")
st.subheader("Full Snapshot (JSON)")

st.caption(
    "The complete database snapshot in JSON format.  Contains all aggregated "
    "statistics, clinical availability, facility data, and gene therapy breakdowns.  "
    "No PHI -- field-level metadata and aggregate counts only."
)

snapshot_json = json.dumps(snapshot, indent=2, default=str)
st.download_button(
    "Download Full Snapshot JSON",
    data=snapshot_json,
    file_name="openmovr_database_snapshot.json",
    mime="application/json",
    key="dl_snapshot",
)

# ---- Footer ----
st.markdown("---")
st.caption(
    "All data is aggregated and de-identified.  No protected health information (PHI) "
    "is included in any download.  Please cite OpenMOVR App and the MOVR Data Hub "
    "in any publications or presentations using this data."
)
