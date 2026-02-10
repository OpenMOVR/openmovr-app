"""
Access Control Utilities

Centralized access gate for provisioned pages.
Supports environment variables, Streamlit secrets, and password input.
"""

import os
import streamlit as st


def _get_access_key() -> str:
    """Resolve access key: env var > st.secrets.  No hardcoded fallback."""
    key = os.environ.get("OPENMOVR_SITE_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENMOVR_SITE_KEY", "")
        except Exception:
            pass
    return key


def has_access() -> bool:
    """Return True if the current session has provisioned access."""
    return bool(st.session_state.get("provisioned_access"))


def require_access(
    page_title: str = "Provisioned Access Required",
    description: str | None = None,
) -> bool:
    """Show access gate if not authenticated.  Returns True when granted.

    Call at the top of any provisioned page.  If access is not yet granted
    the function renders the login form and calls ``st.stop()`` so the
    rest of the page never executes.
    """
    if has_access():
        return True

    st.title(page_title)

    if description:
        st.info(description)
    else:
        st.info(
            "This page requires provisioned access.  "
            "Available to participating sites, researchers, PAGs, and participants "
            "with an approved Data Use Agreement.  All other inquiries should be "
            "directed to the MOVR team.\n\n"
            "**[Request Access](https://mdausa.tfaforms.net/389761)**"
        )

    access_key = _get_access_key()
    if not access_key:
        st.warning(
            "Access keys have not been configured for this deployment.  "
            f"Contact the administrator at {st.secrets.get('ADMIN_EMAIL', 'mdamovr@mdausa.org')}."
        )
        st.stop()
        return False

    with st.form("access_form"):
        pwd = st.text_input("Access Key", type="password")
        submitted = st.form_submit_button("Unlock")

        if submitted:
            if pwd == access_key:
                st.session_state["provisioned_access"] = True
                st.rerun()
            else:
                st.error("Invalid access key.")

    st.stop()
    return False  # unreachable, but keeps type checkers happy
