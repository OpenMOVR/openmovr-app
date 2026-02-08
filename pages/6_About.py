"""
About Page

Project information, access tiers, and version history.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from config.settings import (
    PAGE_ICON, APP_VERSION, STUDY_NAME, STUDY_PROTOCOL,
    STUDY_START, STUDY_END, STUDY_STATUS, LOGO_PNG,
)

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="About - OpenMOVR App",
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
    [data-testid="stTable"] thead tr th:first-child,
    [data-testid="stTable"] tbody tr td:first-child {
        display: none;
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

# ---- Page content ----
st.title("About OpenMOVR App")

st.markdown(
    f"""
    **OpenMOVR App** is an open-source analytics dashboard for the
    **{STUDY_NAME}** clinical registry ({STUDY_PROTOCOL} study protocol,
    {STUDY_START} -- {STUDY_END}, {STUDY_STATUS}).

    The registry captures longitudinal clinical data for **seven rare
    neuromuscular diseases**: ALS, DMD, BMD, SMA, LGMD, FSHD, and Pompe
    disease, across 60+ participating sites in the United States.
    """
)

# ---- Access Tiers ----
st.markdown("---")
st.subheader("Access Tiers")

st.markdown(
    """
    OpenMOVR App operates on a **two-tier access model** designed to maximize
    the value of MOVR data while sustaining development.
    """
)

col_pub, col_prov = st.columns(2)

with col_pub:
    st.markdown(
        """
        <div style='background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 16px; border-radius: 0 4px 4px 0; min-height: 360px;'>
        <h4 style='margin-top:0;'>Public Access</h4>
        <p style='font-size:0.95em;'>
        Available to everyone -- no login required.
        </p>
        <ul style='font-size:0.9em;'>
            <li>Aggregated enrollment &amp; disease distribution</li>
            <li>Anonymized site map</li>
            <li>Disease Explorer with demographic breakdowns</li>
            <li>LGMD Overview &amp; future disease deep-dives</li>
            <li>Curated Data Dictionary (1,024 clinical fields)</li>
            <li>Clinical data highlights (functional scores, medications,
                trials, hospitalizations)</li>
        </ul>
        <p style='font-size:0.85em; color:#666;'>
        Tables are displayed as static views.  For downloadable data,
        see Provisioned Access.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_prov:
    st.markdown(
        """
        <div style='background: #E3F2FD; border-left: 4px solid #1E88E5;
        padding: 16px; border-radius: 0 4px 4px 0; min-height: 360px;'>
        <h4 style='margin-top:0;'>Provisioned Access</h4>
        <p style='font-size:0.95em;'>
        Requires an access key.
        </p>
        <ul style='font-size:0.9em;'>
            <li><strong>Site Analytics</strong> -- site-level reports with
                facility names, site-vs-overall comparisons, per-disease
                breakdowns</li>
            <li><strong>Download Center</strong> -- export tables as CSV,
                download snapshot data for custom analytics</li>
            <li><em>Coming soon:</em> Cohort Builder, custom longitudinal
                analytics</li>
        </ul>
        <p style='font-size:0.85em; color:#666;'>
        <strong>Free</strong> for participating sites, researchers with an
        approved DUA, and patients.<br>
        <strong>Commercial license</strong> required for pharma / industry.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

st.markdown(
    """
    <div style='text-align: center; margin: 1rem 0;'>
    <a href="https://mdausa.tfaforms.net/389761" target="_blank"
       style='background: #1E88E5; color: white; padding: 10px 24px;
       border-radius: 4px; text-decoration: none; font-weight: bold;'>
       Request Access
    </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Roadmap ----
st.markdown("---")
st.subheader("Roadmap")

st.markdown(
    """
    | Feature | Status |
    |---------|--------|
    | Public dashboard with aggregated statistics | Available |
    | Anonymized site map with disease filters | Available |
    | Disease Explorer & LGMD deep-dive | Available |
    | Curated Data Dictionary (19 clinical domains) | Available |
    | Gene therapy & advanced therapies tracking | Available |
    | Site Analytics (provisioned) | Available |
    | Download Center (provisioned) | Available |
    | Cohort Builder with custom filters | Planned |
    | Custom longitudinal analytics | Planned |
    | Additional disease deep-dives (SMA, ALS, DMD) | Planned |
    | Patient-facing data summaries | Planned |
    """
)

# ---- Study Details ----
st.markdown("---")
st.subheader("Study Details")

st.markdown(
    f"""
    | | |
    |---|---|
    | **Registry** | {STUDY_NAME} |
    | **Protocol** | {STUDY_PROTOCOL} |
    | **Study Period** | {STUDY_START} -- {STUDY_END} |
    | **Status** | {STUDY_STATUS} |
    | **Diseases** | ALS, DMD, BMD, SMA, LGMD, FSHD, Pompe |
    | **Data Sponsor** | Muscular Dystrophy Association (MDA) |
    | **Sites** | 60+ across the United States |

    *The MDA sponsors the MOVR study and data collection.  The OpenMOVR App
    is an independent open-source project created by Andre D Paredes.*
    """
)

# ---- Version ----
st.markdown("---")
st.subheader("Version History")

st.markdown(
    f"""
    **v{APP_VERSION}** (Gen1 Prototype)
    - Public dashboard with aggregated statistics
    - Disease Explorer, Facility View, Data Dictionary
    - LGMD Overview with PAG presentation features
    - Site map with data-driven patient counts
    - Advanced therapies tracking from Combo Drugs table
    - Site Analytics with provisioned access
    - Download Center for data exports
    """
)

# ---- Footer ----
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    <strong>Created by</strong> Andre D Paredes<br>
    <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a> |
    <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a> (MDA)<br><br>
    <a href="https://openmovr.github.io" target="_blank">openmovr.github.io</a> |
    <a href="https://github.com/OpenMOVR/openmovr-app" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
