"""
Sign the DUA — redirects to MDA MOVR Data Use Agreement request form.
"""

import streamlit as st

st.set_page_config(page_title="Sign the DUA", page_icon="\U0001f4cb", layout="wide")

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
