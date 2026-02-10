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


def _render_prototype_banner() -> None:
    """Render a small prototype banner at the top of every page."""
    st.markdown(
        """
        <div style='background-color: #FFF3E0; border: 1px solid #FFB74D;
        padding: 10px 16px; border-radius: 4px; margin-bottom: 1rem;
        font-size: 0.83em; color: #E65100; text-align: center; line-height: 1.6;'>
        <strong>Proof-of-Concept Prototype</strong> &mdash;
        No individual-level data is accessible. No database is connected.
        All statistics are pre-computed and fully aggregated.
        Analytics are a preview and under active development.
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_global_css() -> None:
    """Inject the global CSS and prototype banner shared by every page.

    Includes:
    - Prototype status banner
    - Sidebar nav branding (title, subtitle, PUBLIC / DUA REQUIRED labels)
    - White sidebar / light-grey page background
    - ``.clean-table`` styling for static tables
    """
    _render_prototype_banner()
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
            content: "Open Source Project\\A Data Source: MDA MOVR Data Hub\\A Gen1 | v0.2.0 (Prototype)";
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
    """Render the shared sidebar footer (contact + centred logo).

    Call this at the *end* of your ``with st.sidebar:`` block (or after
    all page-specific sidebar widgets) so the footer sits below filters.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; font-size: 0.75em; color: #999;'>
            Contact:
            <a href="mailto:andre.paredes@ymail.com">andre.paredes@ymail.com</a> |
            <a href="mailto:aparedes@mdausa.org">aparedes@mdausa.org</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
                    Gen1 | v{APP_VERSION} (Prototype)
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
    st.markdown(
        f"<div style='text-align: center; color: #888; font-size: 0.85em;'>"
        f"Data Source: <a href='https://mdausa.tfaforms.net/389761' target='_blank' "
        f"style='color: #1E88E5;'>MDA {STUDY_NAME} Study</a><br>"
        f"Independently built via the "
        f"<a href='https://openmovr.github.io' target='_blank' "
        f"style='color: #1E88E5;'>OpenMOVR Initiative</a><br>"
        f"Gen1 | v{APP_VERSION} (Prototype)<br>"
        f"<a href='mailto:andre.paredes@ymail.com' style='color: #999;'>"
        f"andre.paredes@ymail.com</a>"
        f"</div>",
        unsafe_allow_html=True,
    )
