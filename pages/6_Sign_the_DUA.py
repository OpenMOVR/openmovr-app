"""
Sign the DUA — redirects to MDA MOVR Data Use Agreement request form.
"""

import sys
from pathlib import Path

app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

import streamlit as st
from components.sidebar import inject_global_css, render_sidebar_footer

_logo_path = Path(__file__).parent.parent / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(page_title="Sign the DUA", page_icon=str(_logo_path) if _logo_path.exists() else "\U0001f4cb", layout="wide")

inject_global_css()
render_sidebar_footer()

st.title("Data Use Agreement")

st.markdown(
    """
    Access to **Site Analytics** and **Download Center** requires a signed
    Data Use Agreement (DUA) with the Muscular Dystrophy Association.
    """
)

st.link_button(
    "Sign the DUA \u2192",
    "https://mdausa.tfaforms.net/389761",
    type="primary",
)

st.caption("You will be redirected to the MDA MOVR DUA request form.")
