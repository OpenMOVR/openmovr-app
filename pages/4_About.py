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
from config.contact import (
    ADMIN_EMAIL, FEEDBACK_FORM_URL, FEEDBACK_FORM_ENABLED,
    GITHUB_ISSUES_URL
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
    OpenMOVR App is an open-source analytics dashboard for the
    {STUDY_NAME} clinical registry ({STUDY_PROTOCOL} study protocol,
    {STUDY_START} to {STUDY_END}, {STUDY_STATUS}).

    The registry captures longitudinal clinical data for seven rare
    neuromuscular diseases (ALS, DMD, BMD, SMA, LGMD, FSHD, and Pompe
    disease) across 60+ participating sites in the United States.

    The app is built for the research community: clinical researchers,
    academic institutions, and participant advocacy groups working to
    advance disease understanding and drug development.  It uses modern
    data visualization and cohort-browsing tools to make registry insights
    accessible and actionable, with capabilities that will grow over time.

    All data access is governed by the MOVR Data Hub governance policy
    and supports the registry's mission of driving progress in
    neuromuscular disease research.
    """
)

# ---- Impact & Vision ----
st.markdown("---")
st.subheader("Why OpenMOVR")

st.markdown(
    """
    We want participants to land here and see what their enrollment is
    contributing to.  We want researchers and clinicians to see what
    questions this data can answer.  That is the idea behind OpenMOVR.

    The longer-term vision includes cohort-building tools, longitudinal
    trend analysis, and purpose-built applications aligned with the
    registry's mission.  More is being built.
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
    | **Study Period** | {STUDY_START} to {STUDY_END} |
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
            <li>Clinical Summary Preview per disease (prototype)</li>
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
            <li><strong>Clinical Analytics</strong> &mdash; disease-specific
                analytics (DMD, LGMD, ALS, SMA) with motor scores,
                therapeutics, genetics, longitudinal trends, and data tables</li>
            <li><em>Coming soon:</em> Cohort Builder, custom longitudinal
                analytics</li>
        </ul>
        <p style='font-size:0.85em; color:#666;'>
        Available to <strong>participating sites</strong>, <strong>PAGs</strong>,
        <strong>researchers</strong>, and <strong>participants</strong> under the
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
    | Anonymized site map with per-disease filters | Prototype |
    | Disease Explorer with cascading filters | Prototype |
    | Curated Data Dictionary (1,024 fields, 19 clinical domains) | Prototype |
    | DMD Clinical Analytics (genetics, therapeutics, functional outcomes) | Prototype |
    | LGMD Clinical Analytics (subtypes, diagnostic journey, medications) | Prototype |
    | ALS Clinical Analytics (ALSFRS-R, milestones, respiratory) | Prototype |
    | SMA Clinical Analytics (motor scores, SMN2 genetics, therapeutics) | Prototype |
    | Disease-specific therapies tracking | Prototype |
    | Site Analytics (DUA required) | Prototype |
    | Download Center (DUA required) | Prototype |
    | Additional clinical analytics (BMD, FSHD, Pompe) | Planned |
    | Cohort Builder with custom filters | Planned |
    | Longitudinal trend analysis | Planned |
    | Participant-facing data summaries | Planned |
    """
)

# ---- Version ----
st.markdown("---")
st.subheader("Version History")

st.markdown(
    f"""
    **v{APP_VERSION}** (Gen1 Prototype)
    - Public dashboard with aggregated enrollment, disease distribution, and longitudinal metrics
    - Disease Explorer with demographic breakdowns, diagnosis profiles, and Clinical Summary Preview
    - Clinical Analytics for DMD, LGMD, ALS, and SMA organized by clinical domains (DUA required)
    - Curated Data Dictionary with 1,024 fields classified into 19 clinical domains
    - Anonymized site map with per-disease filtering across 60+ participating sites
    - Disease-specific therapies tracking (gene therapy, antisense, ERT, disease-modifying)
    - Cumulative and monthly enrollment charts per disease and across registry
    - Site Analytics with site-vs-overall comparisons (DUA required)
    - Download Center for data exports (DUA required)
    - Facility View with top-site rankings and geographic distribution
    """
)

# ---- Get Help / Contact ----
st.markdown("---")
st.subheader("Get Help or Provide Feedback")

col_users, col_devs = st.columns(2)

with col_users:
    st.markdown("### For Users")
    st.markdown(
        f"""
        **Clinicians • Researchers • Data Managers**
        
        - **Report issues or suggest features**:  
          [Click here to submit feedback]({FEEDBACK_FORM_URL})
          
        - **Data access or study questions**:  
          [{ADMIN_EMAIL}](mailto:{ADMIN_EMAIL})
          
        - **Study participants in MOVR**:  
          [Visit the Pilot Page](https://openmovr.github.io/pilot/)
        
        <div style='background: #FFF3E0; padding: 12px; border-radius: 4px; 
                    margin-top: 12px; border-left: 3px solid #FF9800;'>
        <strong>Privacy Note:</strong> Do not include patient names, medical 
        record numbers, or other PHI in feedback submissions.
        </div>
        """,
        unsafe_allow_html=True
    )

with col_devs:
    st.markdown("### For Developers & Initiative")
    st.markdown(
        f"""
        **Open Source Contributors & OpenMOVR Initiative**
        
        - **OpenMOVR Initiative inquiries**:  
          [andre.paredes@ymail.com](mailto:andre.paredes@ymail.com)
        
        - **Report technical issues**:  
          [GitHub Issues]({GITHUB_ISSUES_URL})
          
        - **Submit code contributions**:  
          [GitHub Repository](https://github.com/OpenMOVR/openmovr-app)
          
        - **View documentation**:  
          See `docs/` folder in repository
        
        Contributions welcome! See CONTRIBUTING.md for guidelines.
        """
    )

st.info(
    "**Tip**: Use the **'Report Issue or Feedback'** button at the top of each page or in the sidebar "
    "for quick access to the feedback form."
)

# ---- Footer ----
render_page_footer()
