"""
About Page

Study context, what this app is, who it's for, and version history.
"""

import sys
from pathlib import Path

# Add webapp directory to path
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from config.settings import (
    PAGE_ICON, APP_VERSION, LOGO_PNG,
    STUDY_NAME, STUDY_PROTOCOL, STUDY_START, STUDY_END, STUDY_STATUS,
)

# Page configuration
_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"
st.set_page_config(
    page_title="About - OpenMOVR App",
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

# Sidebar
with st.sidebar:
    if LOGO_PNG.exists():
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(str(LOGO_PNG), width=160)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.8em; color: #888;'>
            <strong>Open Source Project</strong><br>
            <a href="https://openmovr.github.io" target="_blank">openmovr.github.io</a><br>
            <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a><br><br>
            <strong>Created by</strong> Andre D Paredes<br>
            <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a><br>
            <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a> (MDA)<br><br>
            <a href="https://mdausa.tfaforms.net/389761" target="_blank"><strong>Request Data</strong></a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Header
header_left, header_right = st.columns([3, 1])

with header_left:
    st.title("About OpenMOVR App")

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
# What Is This
# ---------------------------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    **OpenMOVR App** is an independently developed, open-source analytics
    dashboard for the **MOVR** (Outcome and Value Research) clinical registry.
    It provides interactive visualizations of patient demographics, disease
    distributions, facility participation, and disease-specific clinical
    profiles for rare neuromuscular diseases including ALS, DMD, SMA, LGMD,
    FSHD, BMD, and Pompe disease.

    This project was created by **Andre D Paredes** to drive study impact and
    benefit the community of clinicians, researchers, and patient advocacy
    groups who rely on MOVR data. The underlying registry data and eCRF
    infrastructure were developed and funded by the
    **[Muscular Dystrophy Association (MDA)](https://www.mda.org/science/movr2)**.

    This prototype operates on **pre-computed aggregate statistics** (snapshot
    mode). No patient-level data is exposed. Features that require a live
    data connection are clearly marked as unavailable.
    """
)

# ---------------------------------------------------------------------------
# Study Protocol
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Study Protocol")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        | | |
        |---|---|
        | **Study** | {STUDY_NAME} |
        | **Protocol** | {STUDY_PROTOCOL} |
        | **Sponsor** | [Muscular Dystrophy Association (MDA)](https://www.mda.org/science/movr2) |
        | **Start Date** | {STUDY_START} |
        | **End Date** | {STUDY_END} |
        | **Status** | {STUDY_STATUS} |
        """
    )

with col2:
    st.markdown(
        f"""
        | | |
        |---|---|
        | **App** | OpenMOVR App (Gen1) |
        | **Current Version** | v{APP_VERSION} |
        | **Mode** | Prototype (Snapshot) |
        | **Data Source** | Pre-computed aggregate statistics |
        | **Repository** | [OpenMOVR/openmovr-app](https://github.com/OpenMOVR/openmovr-app) |
        | **Created by** | Andre D Paredes ([andre.paredes@ymail.com](mailto:andre.paredes@ymail.com)) |
        """
    )

st.info(
    f"The **{STUDY_NAME}** operated under the **{STUDY_PROTOCOL}** study protocol "
    f"from **{STUDY_START}** through **{STUDY_END}**, sponsored by the "
    f"[Muscular Dystrophy Association (MDA)](https://www.mda.org/science/movr2). "
    f"OpenMOVR App is an independent open-source project providing analytics for this registry data."
)

# ---------------------------------------------------------------------------
# What's In The App
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Pages in This App")

st.markdown(
    """
    | Page | Description |
    |------|-------------|
    | **Dashboard** | Registry-wide key metrics, disease distribution, top facilities, and data availability |
    | **Disease Explorer** | Select a disease to see patient counts and registry share. In live mode: demographics, diagnosis profiles, data tables, and patient-level filtering |
    | **Facility View** | Facility rankings, search, patient count distributions, and CSV export |
    | **Data Dictionary** | Browse and search all MOVR field definitions. Filter by disease, form, field type. Identifies required and potentially mislabeled fields |
    | **LGMD Overview** | Dedicated page for Limb-Girdle Muscular Dystrophy: subtype distribution, demographics, diagnosis characteristics, clinical features, ambulatory status, and care sites |
    | **About** | Study context, version history, and roadmap (this page) |
    """
)

# ---------------------------------------------------------------------------
# Version History
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Version History")

with st.expander("v0.1.0 — Initial Prototype Release (February 2026)", expanded=True):
    st.markdown(
        """
        First public deployment of the OpenMOVR App (Gen1) dashboard on Streamlit Cloud.

        **What's included:**
        - Dashboard with key metrics, disease distribution, facility overview, data availability
        - Disease Explorer with snapshot mode (aggregate counts per disease)
        - Facility View with full facility list, search, and distribution analysis
        - Data Dictionary with disease-first filtering, required field indicators, mislabeled field detection
        - LGMD Overview with subtype analysis, demographics, diagnosis, clinical characteristics, ambulatory status
        - About page with study protocol info and version history

        **Architecture:**
        - Dual-mode operation: snapshot (JSON) for public deployment, live (parquet) for local use
        - API facade layer with fault-tolerant imports
        - Snapshot fallback with styled "unavailable" placeholders for live-only features
        - MOVR logo as browser favicon across all pages
        """
    )

# ---------------------------------------------------------------------------
# Roadmap
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Roadmap")

st.markdown(
    """
    Items under consideration for future releases:

    - **Per-disease snapshots** — Pre-computed demographics, diagnosis profiles, and clinical summaries for Disease Explorer
    - **Custom domain** — `app.openmovr.io` via CNAME to Streamlit Cloud
    - **Additional disease overview pages** — DMD, SMA, ALS dedicated pages similar to LGMD Overview
    - **Enhanced exports** — Aggregate summary downloads across all pages
    - **Live mode deployment** — Authenticated access with full parquet data for interactive filtering
    """
)

# Footer
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #666;'>"
    f"<strong>OpenMOVR App</strong> Gen1 | v{APP_VERSION} (Prototype)<br>"
    f"<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a><br>"
    f"{STUDY_NAME} Study Protocol ({STUDY_PROTOCOL}) | "
    f"{STUDY_START} &ndash; {STUDY_END} ({STUDY_STATUS})<br>"
    f"<span style='font-size: 0.9em;'>Created by Andre D Paredes | "
    f"<a href='mailto:andre.paredes@ymail.com'>andre.paredes@ymail.com</a> | "
    f"<a href='https://openmovr.github.io' target='_blank'>openmovr.github.io</a></span>"
    f"</div>",
    unsafe_allow_html=True
)
