"""
Shared sidebar branding, CSS, and footer for all pages.

Call ``inject_global_css()`` once at the top of every page (after
``st.set_page_config``) and ``render_sidebar_footer()`` at the end of
each page's sidebar block to keep the navigation consistent.
"""

import base64

import streamlit as st

from config.settings import (
    LOGO_JPG, LOGO_PNG, APP_VERSION, STUDY_NAME,
)
from config.contact import (
    ADMIN_NAME, ADMIN_EMAIL, DEVELOPER_NAME, DEVELOPER_EMAIL,
    SHOW_DEVELOPER_CONTACT, SHOW_FEEDBACK_BUTTON, FEEDBACK_FORM_URL,
    FEEDBACK_FORM_ENABLED
)


def _banner_provenance() -> dict:
    """Read data-provenance dates out of the statistics snapshot.

    Returns ``{}`` if the snapshot is unavailable so the banner degrades to
    its generic form rather than failing the page.
    """
    try:
        from utils.cache import get_cached_snapshot
        snapshot = get_cached_snapshot()
        meta = snapshot.get("metadata", {})
        cohort = snapshot.get("community_profile", {}).get("cohort", {})
    except Exception:
        return {}

    generated = str(meta.get("generated_timestamp", ""))[:10]
    return {
        "generated": generated,
        "enrollment_last": cohort.get("enrollment_last"),
        "participants": cohort.get("participants"),
        "facilities": cohort.get("facilities"),
    }


def _render_provenance_banner() -> None:
    """Render the data-provenance banner at the top of every page.

    This banner is what a reader citing a figure from this app relies on, so
    it states three separate dates and never conflates them:

    * the data extract -- what the numbers describe
    * the generation date -- when the aggregates were computed
    * the access date -- supplied by the reader, not by us

    The app being a prototype is a statement about the interface, not about
    the underlying registry data, and is worded so it cannot be read as
    casting doubt on the figures themselves.
    """
    p = _banner_provenance()

    if p.get("enrollment_last") and p.get("generated"):
        scope = (
            f"Aggregated statistics from the MDA {STUDY_NAME} Study"
        )
        if p.get("participants"):
            scope += (
                f" &mdash; {p['participants']:,} participants"
            )
            if p.get("facilities"):
                scope += f" at {p['facilities']} clinical sites"
        provenance = (
            f"<strong>Data extract:</strong> participants enrolled through "
            f"{p['enrollment_last']}. "
            f"<strong>Statistics generated:</strong> {p['generated']}."
        )
    else:
        scope = f"Aggregated statistics from the MDA {STUDY_NAME} Study"
        provenance = (
            "<strong>Data extract and generation dates:</strong> see the "
            "Community Snapshot page."
        )

    feedback_button = ""
    if SHOW_FEEDBACK_BUTTON and FEEDBACK_FORM_ENABLED:
        feedback_button = f'''
        <div style="text-align: center; margin-top: 8px;">
            <a href="{FEEDBACK_FORM_URL}" target="_blank"
               style="display: inline-block; padding: 0.3rem 0.8rem;
                      background-color: #1E88E5; color: white;
                      text-decoration: none; border-radius: 4px;
                      font-size: 0.8em;">
                Report Issue or Feedback
            </a>
        </div>
        '''

    st.markdown(
        f"""
        <div style='background-color: #F1F8FF; border: 1px solid #BBDEFB;
        border-left: 4px solid #1E88E5;
        padding: 10px 16px; border-radius: 4px; margin-bottom: 1rem;
        font-size: 0.83em; color: #1A3C5A; text-align: center;
        line-height: 1.7;'>
        {scope}.<br>
        {provenance}<br>
        <span style='color: #4A6580;'>
        Figures are pre-computed aggregates; no individual-level data is
        connected or displayed. Counts below 11 are suppressed. Denominators
        and field definitions are stated on each page.
        Interface features are under active development; the statistics
        themselves are final for the data extract above.
        </span>
        {feedback_button}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_global_css() -> None:
    """Inject the global CSS and prototype banner shared by every page.

    Includes:
    - Data provenance banner
    - Sidebar nav branding (title, subtitle, PUBLIC / DUA REQUIRED labels)
    - White sidebar / light-grey page background
    - ``.clean-table`` styling for static tables
    """
    _render_provenance_banner()
    st.markdown(
        """
        <style>
        /* --- Page & sidebar colours --- */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff;
        }
        [data-testid="stSidebar"] {
            background-color: #f5f5f5;
        }

        /* --- Nav branding --- */
        [data-testid="stSidebarNav"] {
            padding-top: 7rem;
            position: relative;
        }
        [data-testid="stSidebarNav"]::before {
            content: "OpenMOVR App";
            position: absolute;
            top: 0.5rem;
            left: 0; right: 0;
            text-align: center;
            font-size: 1.4em;
            font-weight: bold;
            color: #1E88E5;
        }
        [data-testid="stSidebarNav"]::after {
            content: "Open Source Project\\A Data Source: MDA MOVR Data Hub\\A Gen1 | v0.2.0";
            position: absolute;
            top: 2.5rem;
            left: 0; right: 0;
            white-space: pre-line;
            text-align: center;
            font-size: 0.65em;
            color: #888;
            line-height: 1.6;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
        }

        /* --- Table styling --- */
        .clean-table { width: 100%; border-collapse: collapse; font-size: 0.85em; }
        .clean-table th { text-align: left; padding: 3px 8px; border-bottom: 2px solid #ddd; }
        .clean-table td { padding: 3px 8px; border-bottom: 1px solid #eee; }

        /* --- PUBLIC label above first nav item --- */
        [data-testid="stSidebarNav"] li:first-child {
            margin-top: 0.5rem; padding-top: 0.5rem;
        }
        [data-testid="stSidebarNav"] li:first-child::before {
            content: "PUBLIC"; display: block; font-size: 0.7em;
            color: #4CAF50; font-weight: bold; padding: 0 14px 4px;
            letter-spacing: 0.05em;
        }

        /* --- DUA REQUIRED separator --- */
        [data-testid="stSidebarNav"] li:nth-last-child(7) {
            margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #ddd;
        }
        [data-testid="stSidebarNav"] li:nth-last-child(7)::before {
            content: "DUA REQUIRED"; display: block; font-size: 0.7em;
            color: #1E88E5; font-weight: bold; padding: 0 14px 4px;
            letter-spacing: 0.05em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render the shared sidebar footer (contact + feedback + logo).

    Call this at the *end* of your ``with st.sidebar:`` block (or after
    all page-specific sidebar widgets) so the footer sits below filters.
    """
    st.sidebar.markdown("---")
    
    # Feedback button for clinicians
    if SHOW_FEEDBACK_BUTTON:
        if FEEDBACK_FORM_ENABLED:
            st.sidebar.markdown(
                f"""
                <div style='text-align: center; margin-bottom: 0.5rem;'>
                    <a href="{FEEDBACK_FORM_URL}" target="_blank" 
                       style="display: inline-block; padding: 0.4rem 1rem; 
                              background-color: #1E88E5; color: white; 
                              text-decoration: none; border-radius: 4px;
                              font-size: 0.85em;">
                        Report Issue or Feedback
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # Show email button if form not set up yet
            st.sidebar.markdown(
                f"""
                <div style='text-align: center; margin-bottom: 0.5rem;'>
                    <a href="mailto:{ADMIN_EMAIL}?subject=OpenMOVR App Feedback" 
                       style="display: inline-block; padding: 0.4rem 1rem; 
                              background-color: #1E88E5; color: white; 
                              text-decoration: none; border-radius: 4px;
                              font-size: 0.85em;">
                        Send Feedback
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Contact information
    contact_html = f"""
        <div style='text-align: center; font-size: 0.75em; color: #999;'>
            Data/Support: <a href="mailto:{ADMIN_EMAIL}">{ADMIN_EMAIL}</a>
    """
    
    if SHOW_DEVELOPER_CONTACT:
        contact_html += f"""<br>OpenMOVR Initiative: <a href="mailto:{DEVELOPER_EMAIL}">{DEVELOPER_EMAIL}</a>"""
    
    contact_html += "</div>"
    
    st.sidebar.markdown(contact_html, unsafe_allow_html=True)
    _logo = LOGO_JPG if LOGO_JPG.exists() else LOGO_PNG
    if _logo.exists():
        _b64 = base64.b64encode(_logo.read_bytes()).decode()
        _mime = "image/jpeg" if _logo.suffix == ".jpg" else "image/png"
        st.sidebar.markdown(
            f'<div style="text-align: center; padding: 0.5rem 0;">'
            f'<img src="data:{_mime};base64,{_b64}" width="140" '
            f'style="display: inline-block;">'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render the shared page header with title (left) and branding (right).

    Call this at the top of every page after ``inject_global_css()`` and
    ``render_sidebar_footer()``.
    """
    header_left, header_right = st.columns([3, 1])

    with header_left:
        st.title(title)
        if subtitle:
            st.markdown(f"### {subtitle}")

    with header_right:
        st.markdown(
            f"""
            <div style='text-align: right; padding-top: 10px;'>
                <span style='font-size: 1.5em; font-weight: bold; color: #1E88E5;'>OpenMOVR App</span><br>
                <span style='font-size: 0.9em; color: #666; background-color: #E3F2FD; padding: 4px 8px; border-radius: 4px;'>
                    Gen1 | v{APP_VERSION}
                </span><br>
                <span style='font-size: 0.75em; color: #999; margin-top: 4px; display: inline-block;'>
                    Data Source: MDA {STUDY_NAME}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_footer() -> None:
    """Render the shared page footer (data source, version, contact).

    Call this at the bottom of every page to keep footers consistent.
    """
    st.markdown("---")
    
    footer_html = (
        f"<div style='text-align: center; color: #888; font-size: 0.85em;'>"
        f"Data Source: <a href='https://mdausa.tfaforms.net/389761' target='_blank' "
        f"style='color: #1E88E5;'>MDA {STUDY_NAME} Study</a><br>"
        f"Independently built via the "
        f"<a href='https://openmovr.github.io' target='_blank' "
        f"style='color: #1E88E5;'>OpenMOVR Initiative</a><br>"
        f"Gen1 | v{APP_VERSION}<br>"
        f"Data/Support: <a href='mailto:{ADMIN_EMAIL}' style='color: #999;'>{ADMIN_EMAIL}</a>"
    )
    
    if SHOW_DEVELOPER_CONTACT:
        footer_html += f" | OpenMOVR: <a href='mailto:{DEVELOPER_EMAIL}' style='color: #999;'>{DEVELOPER_EMAIL}</a>"
    
    footer_html += "</div>"
    
    st.markdown(footer_html, unsafe_allow_html=True)
