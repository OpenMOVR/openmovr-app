"""
About Page

Project information, study details, access tiers, and version history.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from config.settings import (
    PAGE_ICON, APP_VERSION, STUDY_NAME, STUDY_PROTOCOL,
    STUDY_START, STUDY_END, STUDY_STATUS,
)
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="About - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Page content ----
st.title("About OpenMOVR App")

st.markdown(
    f"""
    **OpenMOVR App** is an open-source analytics dashboard for the
    **{STUDY_NAME}** clinical registry ({STUDY_PROTOCOL} study protocol,
    {STUDY_START} -- {STUDY_END}, {STUDY_STATUS}).

    The registry captures longitudinal clinical data for **seven rare
    neuromuscular diseases** -- ALS, DMD, BMD, SMA, LGMD, FSHD, and Pompe
    disease -- across 60+ participating sites in the United States.

    OpenMOVR App is built for the **research community first** -- clinical
    researchers, academic institutions engaged in basic and translational
    science, and patient advocacy groups working to advance disease
    understanding and drug development.  It leverages modern data
    visualization and cohort-browsing technology to make registry insights
    accessible and actionable, with capabilities that will grow over time.

    All data access is governed by the **MOVR Data Hub governance policy**
    and is designed to support the registry's mission of driving progress
    in neuromuscular disease research.
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

# ---- Access Tiers ----
st.markdown("---")
st.subheader("Access Tiers")

col_pub, col_prov = st.columns(2)

with col_pub:
    st.markdown(
        """
        <div style='background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 16px; border-radius: 0 4px 4px 0; min-height: 380px;'>
        <h4 style='margin-top:0;'>Public Access</h4>
        <p style='font-size:0.95em;'>
        Available to everyone &mdash; no login required.
        </p>
        <ul style='font-size:0.9em;'>
            <li>Aggregated enrollment &amp; disease distribution</li>
            <li>Anonymized site map</li>
            <li>Disease Explorer with demographic breakdowns</li>
            <li>Disease-specific overview (prototype)</li>
            <li>Curated Data Dictionary (1,024 clinical fields)</li>
            <li>Clinical data highlights (functional scores, medications,
                trials, hospitalizations)</li>
        </ul>
        <p style='font-size:0.85em; color:#666;'>
        All data shown is pre-computed aggregated statistics (snapshots)
        &mdash; no individual-level data is connected or displayed.
        For downloadable data, see Provisioned Access.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_prov:
    st.markdown(
        """
        <div style='background: #E3F2FD; border-left: 4px solid #1E88E5;
        padding: 16px; border-radius: 0 4px 4px 0; min-height: 380px;'>
        <h4 style='margin-top:0;'>Provisioned Access</h4>
        <p style='font-size:0.95em;'>
        Requires a signed Data Use Agreement (DUA).
        </p>
        <ul style='font-size:0.9em;'>
            <li><strong>Site Analytics</strong> &mdash; site-level reports with
                facility names, site-vs-overall comparisons, per-disease
                breakdowns</li>
            <li><strong>Download Center</strong> &mdash; export tables as CSV,
                download snapshot data for custom analytics</li>
            <li><strong>Clinical Summaries</strong> &mdash; disease-specific
                clinical analytics organized by clinical domain</li>
            <li><em>Coming soon:</em> Cohort Builder, custom longitudinal
                analytics</li>
        </ul>
        <p style='font-size:0.85em; color:#666;'>
        Available to <strong>participating sites</strong>, <strong>PAGs</strong>,
        <strong>researchers</strong>, and <strong>patients</strong> under the
        MOVR Data Hub governance policy.<br><br>
        All other inquiries &mdash; including those from organizations seeking
        data access &mdash; should be directed to the MOVR team to discuss
        alignment with the registry's mission and objectives.
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
       Sign the DUA
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
    | Public dashboard with aggregated statistics | Prototype |
    | Anonymized site map with disease filters | Prototype |
    | Disease Explorer & disease-specific overview | Prototype |
    | Curated Data Dictionary (19 clinical domains) | Prototype |
    | Gene therapy & advanced therapies tracking | Prototype |
    | Site Analytics (provisioned) | Prototype |
    | Download Center (provisioned) | Prototype |
    | Cohort Builder with custom filters | Planned |
    | Custom longitudinal analytics | Planned |
    | ALS Clinical Summary (ALSFRS-R, milestones, respiratory) | Prototype |
    | Additional clinical summaries (SMA, BMD) | Planned |
    | Patient-facing data summaries | Planned |
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
render_page_footer()
