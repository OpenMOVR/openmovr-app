"""
Ask OpenMOVR - Structured Data Chatbot

Public page — no DUA required. Answers questions about the MOVR
registry using pre-computed snapshot statistics (no LLM).
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from config.settings import PAGE_ICON, APP_VERSION
from components.sidebar import inject_global_css, render_sidebar_footer, render_page_footer
from components.chatbot import ChatbotRenderer

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="Ask OpenMOVR",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

# ---- Sidebar ----
with st.sidebar:
    st.markdown(
        """
        <div style='background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 12px; border-radius: 0 4px 4px 0; font-size: 0.85em;'>
        <strong>Ask OpenMOVR</strong><br>
        Ask questions about the MOVR registry and get instant answers
        from pre-computed summary statistics.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---- Main content ----
renderer = ChatbotRenderer()
renderer.render()

# ---- Footer ----
render_page_footer()
