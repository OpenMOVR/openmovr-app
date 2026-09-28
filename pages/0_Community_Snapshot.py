"""
Community Snapshot Page

Study-wide demographics, health insurance coverage and minimum diagnosis
elements across all seven disease types in the MOVR Data Hub Study.

This page exists so that aggregate community-level figures -- health
insurance coverage in particular -- can be cited from a stable public
source. Every figure is pre-computed and fully aggregated; no
individual-level data is connected or displayed.
"""

import sys
from pathlib import Path

# Add app root to path FIRST (before other imports)
app_dir = Path(__file__).parent.parent
sys.path.insert(0, str(app_dir))

from datetime import date

import json
import pandas as pd
import plotly.express as px
import streamlit as st

from api import StatsAPI
from components.sidebar import (
    inject_global_css, render_sidebar_footer, render_page_footer,
    render_page_header,
)
from components.tables import static_table
from config.settings import APP_VERSION, PAGE_ICON, STUDY_NAME

_logo_path = app_dir / "assets" / "movr_logo_clean_nobackground.png"

st.set_page_config(
    page_title="Community Snapshot - OpenMOVR App",
    page_icon=str(_logo_path) if _logo_path.exists() else PAGE_ICON,
    layout="wide",
)

inject_global_css()
render_sidebar_footer()

render_page_header(
    "Community Snapshot",
    "Demographics, health insurance and diagnosis across all disease types",
)

# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

try:
    profile = StatsAPI.get_community_profile()
    snapshot_meta = StatsAPI.get_snapshot_metadata()
except FileNotFoundError as exc:
    st.error(f"Statistics snapshot not available.\n\n{exc}")
    st.stop()

if not profile:
    st.error(
        "The community profile is missing from the statistics snapshot. "
        "Regenerate it with `python scripts/generate_stats_snapshot.py`."
    )
    st.stop()

cohort = profile.get("cohort", {})
demographics = profile.get("demographics", {})
insurance = profile.get("health_insurance", {})
diagnosis = profile.get("diagnosis_minimum", {})

overall = insurance.get("overall", {})
measures = overall.get("measures", {})

generated = str(snapshot_meta.get("generated_timestamp", ""))[:10]


def _fmt_measure(key: str) -> str:
    """Format one derived insurance measure, honouring suppression."""
    m = measures.get(key)
    if not m:
        return "n/a"
    if m.get("suppressed"):
        return "suppressed"
    return f"{m['pct_of_responders']}%"


def _measure_n(key: str) -> str:
    m = measures.get(key)
    if not m or m.get("suppressed"):
        return ""
    return f"n = {m['count']:,}"


# --------------------------------------------------------------------------
# What this page covers
# --------------------------------------------------------------------------

st.markdown(
    f"""
    <div style='background-color:#E3F2FD; border-left:4px solid #1E88E5;
    padding:12px 16px; border-radius:4px; font-size:0.9em; line-height:1.7;'>
    <strong>{cohort.get('label', STUDY_NAME)}</strong><br>
    <strong>{cohort.get('participants', 0):,} participants</strong> across
    <strong>{len(cohort.get('diseases', []))} disease types</strong>
    ({', '.join(cohort.get('diseases', []))}) at
    <strong>{cohort.get('facilities', 0)} clinical sites</strong>.<br>
    Enrollment {cohort.get('enrollment_first', 'n/a')} to
    {cohort.get('enrollment_last', 'n/a')}. Statistics generated
    {generated}.<br>
    <span style='color:#555;'>Cohort: {cohort.get('cohort_definition', '')}.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("")

# --------------------------------------------------------------------------
# Key measures
# --------------------------------------------------------------------------

st.header("Health Insurance Coverage")

st.markdown(
    f"Of {overall.get('participants', 0):,} participants, "
    f"**{overall.get('responders', 0):,}** reported at least one identifiable "
    f"insurance type at enrollment. The percentages below use that figure as "
    f"the denominator. Participants may report more than one insurance type, "
    f"so categories sum to more than 100%."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Medicaid", _fmt_measure("medicaid"), _measure_n("medicaid"),
          delta_color="off")
c2.metric("Medicare", _fmt_measure("medicare"), _measure_n("medicare"),
          delta_color="off")
c3.metric("Any public coverage", _fmt_measure("any_public"),
          _measure_n("any_public"), delta_color="off")
c4.metric("Private or group", _fmt_measure("private_or_group"),
          _measure_n("private_or_group"), delta_color="off")

c5, c6 = st.columns(2)
c5.metric("Medicaid and Medicare (dual)",
          _fmt_measure("medicaid_and_medicare"),
          _measure_n("medicaid_and_medicare"), delta_color="off")
c6.metric("No insurance / self-pay", _fmt_measure("uninsured_self_pay"),
          _measure_n("uninsured_self_pay"), delta_color="off")

st.caption(
    "\"Any public coverage\" counts participants reporting Medicaid, Medicare, "
    "or Veterans Administration / military insurance."
)

st.warning(
    "**Read the disease and age breakdowns before citing a single figure.** "
    "The insurance mix differs sharply across the community. ALS accounts for "
    "roughly half of all participants and is predominantly Medicare-covered, "
    "while the pediatric-onset conditions are predominantly Medicaid-covered. "
    "A study-wide percentage describes the registry as a whole, not any one "
    "disease population or age group."
)

# --------------------------------------------------------------------------
# Overall distribution
# --------------------------------------------------------------------------

categories = overall.get("categories", [])

if categories:
    left, right = st.columns([3, 2])

    with left:
        chart_rows = [c for c in categories if not c.get("suppressed")]
        if chart_rows:
            chart_df = pd.DataFrame({
                "Insurance type": [c["label"] for c in chart_rows],
                "Participants": [c["count"] for c in chart_rows],
                "Percent": [c["pct_of_responders"] for c in chart_rows],
            })
            fig = px.bar(
                chart_df,
                x="Participants",
                y="Insurance type",
                orientation="h",
                text="Percent",
                color="Participants",
                color_continuous_scale="Blues",
                title="Reported insurance type, all participants",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                height=max(350, len(chart_rows) * 48),
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending", title=""),
                xaxis_title="Participants reporting",
                margin=dict(r=80),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        table_df = pd.DataFrame([
            {
                "Insurance type": c["label"],
                "Participants": f"{c['count']:,}",
                "% of responders": f"{c['pct_of_responders']}%",
            }
            for c in categories
        ])
        static_table(table_df)
        st.caption(
            f"Denominator: {overall.get('responders', 0):,} participants who "
            f"reported an identifiable insurance type. "
            f"{overall.get('no_informative_response', 0):,} answered only "
            f"\"Unknown\" or \"Not Reported\" and are excluded."
        )

# --------------------------------------------------------------------------
# By disease and by age band
# --------------------------------------------------------------------------


def _group_entry(name: str) -> dict:
    for e in insurance.get("by_group", []):
        if e.get("group") == name:
            return e
    return {}


def _group_pct(name: str):
    e = _group_entry(name)
    m = e.get("measures", {}).get("medicaid", {})
    return m.get("pct_of_responders", "n/a")


def _group_n(name: str):
    e = _group_entry(name)
    return f"{e.get('responders', 0):,}"


def _measure_cell(entry: dict, key: str) -> str:
    m = entry.get("measures", {}).get(key)
    if not m:
        return "n/a"
    if m.get("suppressed"):
        return "suppressed"
    return f"{m['pct_of_responders']}% ({m['count']:,})"


tab_group, tab_disease, tab_age, tab_method = st.tabs(
    ["By disease group", "By disease", "By age at enrollment",
     "Methods and definitions"]
)

with tab_group:
    by_group = insurance.get("by_group", [])
    if not by_group:
        st.info("No disease-group insurance data in this snapshot.")
    else:
        st.markdown(
            "ALS is roughly half of the registry, is adult-onset, and is "
            "predominantly Medicare-covered. A study-wide percentage is "
            "therefore driven by ALS and describes the other six conditions "
            "poorly. These groupings let a figure be cited against the "
            "population it actually describes."
        )
        rows = [
            {
                "Group": e.get("group", ""),
                "Definition": e.get("definition", ""),
                "Participants": f"{e.get('participants', 0):,}",
                "Reported insurance": f"{e.get('responders', 0):,}",
                "Medicaid": _measure_cell(e, "medicaid"),
                "Medicare": _measure_cell(e, "medicare"),
                "Any public": _measure_cell(e, "any_public"),
                "Private or group": _measure_cell(e, "private_or_group"),
            }
            for e in by_group
        ]
        static_table(pd.DataFrame(rows))
        st.caption(
            "Percentages are of participants within that group who reported "
            "an identifiable insurance type, shown with the underlying count. "
            "Groups overlap: Duchenne, Becker and SMA are a subset of the "
            "non-ALS group."
        )
        st.download_button(
            "Download insurance by disease group (CSV)",
            pd.DataFrame(rows).to_csv(index=False),
            file_name="movr_insurance_by_disease_group.csv",
            mime="text/csv",
        )

with tab_disease:
    by_disease = insurance.get("by_disease", [])
    if not by_disease:
        st.info("No per-disease insurance data in this snapshot.")
    else:
        rows = [
            {
                "Disease": e.get("disease", ""),
                "Participants": f"{e.get('participants', 0):,}",
                "Reported insurance": f"{e.get('responders', 0):,}",
                "Medicaid": _measure_cell(e, "medicaid"),
                "Medicare": _measure_cell(e, "medicare"),
                "Any public": _measure_cell(e, "any_public"),
                "Private or group": _measure_cell(e, "private_or_group"),
            }
            for e in by_disease
        ]
        static_table(pd.DataFrame(rows))
        st.caption(
            "Percentages are of participants within that disease who reported "
            "an identifiable insurance type, shown with the underlying count. "
            "Participants may report more than one type."
        )

        chart_rows = [
            {
                "Disease": e.get("disease", ""),
                "Medicaid": e["measures"]["medicaid"]["pct_of_responders"],
                "Medicare": e["measures"]["medicare"]["pct_of_responders"],
                "Private or group":
                    e["measures"]["private_or_group"]["pct_of_responders"],
            }
            for e in by_disease
            if e.get("measures")
        ]
        if chart_rows:
            melted = pd.DataFrame(chart_rows).melt(
                id_vars="Disease", var_name="Insurance type",
                value_name="Percent",
            )
            fig = px.bar(
                melted,
                x="Disease",
                y="Percent",
                color="Insurance type",
                barmode="group",
                title="Insurance coverage by disease "
                      "(% of participants reporting insurance)",
            )
            fig.update_layout(height=430, yaxis_title="% of responders",
                              xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download insurance by disease (CSV)",
            pd.DataFrame(rows).to_csv(index=False),
            file_name="movr_insurance_by_disease.csv",
            mime="text/csv",
        )

with tab_age:
    by_age = insurance.get("by_age_band", [])
    if not by_age:
        st.info("No age-band insurance data in this snapshot.")
    else:
        rows = [
            {
                "Age at enrollment": e.get("band", ""),
                "Participants": f"{e.get('participants', 0):,}",
                "Reported insurance": f"{e.get('responders', 0):,}",
                "Medicaid": _measure_cell(e, "medicaid"),
                "Medicare": _measure_cell(e, "medicare"),
                "Any public": _measure_cell(e, "any_public"),
                "Private or group": _measure_cell(e, "private_or_group"),
            }
            for e in by_age
        ]
        static_table(pd.DataFrame(rows))
        st.caption(
            "Age is calculated at the date of enrollment, not as of today. "
            "Participants enrolled as children may since have reached "
            "adulthood, and coverage may have changed accordingly."
        )

        chart_rows = [
            {
                "Age at enrollment": e.get("band", ""),
                "Medicaid": e["measures"]["medicaid"]["pct_of_responders"],
                "Medicare": e["measures"]["medicare"]["pct_of_responders"],
                "Private or group":
                    e["measures"]["private_or_group"]["pct_of_responders"],
            }
            for e in by_age
            if e.get("measures")
        ]
        if chart_rows:
            melted = pd.DataFrame(chart_rows).melt(
                id_vars="Age at enrollment", var_name="Insurance type",
                value_name="Percent",
            )
            fig = px.bar(
                melted,
                x="Age at enrollment",
                y="Percent",
                color="Insurance type",
                barmode="group",
                title="Insurance coverage by age at enrollment "
                      "(% of participants reporting insurance)",
            )
            fig.update_layout(height=430, yaxis_title="% of responders",
                              xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download insurance by age band (CSV)",
            pd.DataFrame(rows).to_csv(index=False),
            file_name="movr_insurance_by_age_band.csv",
            mime="text/csv",
        )

with tab_method:
    st.markdown(
        f"""
**Source field.** `{insurance.get('field', 'hltin')}` --
"{insurance.get('question_label', 'Health insurance type')}",
{insurance.get('collected', 'self-reported at enrollment').lower()}.

**Multi-select.** Participants may select more than one insurance type, so
category percentages sum to more than 100%. A participant reporting both
Medicaid and private insurance is counted in both rows.

**Denominator.** {insurance.get('denominator_note', '')}

**Small cells.** {insurance.get('small_cell_policy', '')} This follows HIPAA
de-identification practice and means some categories show as "suppressed"
rather than a count.

**Point in time.** Coverage is recorded at enrollment. Enrollment in this
cohort spans {cohort.get('enrollment_first', 'n/a')} to
{cohort.get('enrollment_last', 'n/a')}, so these figures describe coverage at
the time each participant joined the registry, not current coverage.

**Cohort.** {cohort.get('cohort_definition', '')}. USNDR legacy participants
are excluded, consistent with every other figure published in this app.

**Not a prevalence estimate.** MOVR is a clinic-based registry of participants
enrolled at {cohort.get('facilities', 0)} participating neuromuscular centers.
It is not a probability sample of everyone in the United States living with
these conditions, and these percentages should not be read as national
prevalence estimates.
        """
    )

# --------------------------------------------------------------------------
# Citation
# --------------------------------------------------------------------------

st.markdown("")
with st.expander("How to cite this page", expanded=False):
    accessed = date.today().strftime("%B %d, %Y").replace(" 0", " ")
    st.markdown(
        f"""
Suggested citation:

> OpenMOVR Initiative. *Community Snapshot: MDA {STUDY_NAME} Study.*
> OpenMOVR App, Gen1 v{APP_VERSION}. Statistics generated {generated}.
> Accessed {accessed}.

When citing a specific figure, always name the population and the
denominator, because the insurance mix differs sharply between disease
groups. Both of the following are accurate and they are not
interchangeable:

- "Among {cohort.get('participants', 0):,} participants in the MDA
  {STUDY_NAME} Study who reported an insurance type at enrollment
  (n = {overall.get('responders', 0):,}),
  {measures.get('medicaid', {}).get('pct_of_responders', 'n/a')}% reported
  Medicaid coverage."
- "Among MOVR participants with a muscular dystrophy, spinal muscular
  atrophy or Pompe disease -- that is, excluding ALS -- {_group_pct(
  'Muscular dystrophies, SMA and Pompe disease')}% reported Medicaid
  coverage (n = {_group_n('Muscular dystrophies, SMA and Pompe disease')})."

A figure cited without its population is not supported by this page.

For figures beyond those published here, or for participant-level analyses,
use the MOVR data request process linked in **Sign the DUA**.
        """
    )

# --------------------------------------------------------------------------
# Demographics
# --------------------------------------------------------------------------

st.markdown("---")
st.header("Demographics")
st.caption(
    "Collected at enrollment and identical in definition across all disease "
    "types."
)


def _dist_table(rows: list, label: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            label: r["label"],
            "Participants": f"{r['count']:,}",
            "%": f"{r.get('pct', '')}%" if r.get("pct") is not None else "",
        }
        for r in rows
    ])


d1, d2 = st.columns(2)

with d1:
    gender = demographics.get("gender", [])
    if gender:
        st.subheader("Sex")
        static_table(_dist_table(gender, "Sex"))

with d2:
    race = demographics.get("race_ethnicity", [])
    if race:
        st.subheader("Race and ethnicity")
        static_table(_dist_table(race, "Race / ethnicity"))
        st.caption(
            "Participants selecting more than one category are reported as "
            "Multiracial."
        )

age_hist = demographics.get("age_at_enrollment", [])
age_summary = demographics.get("age_at_enrollment_summary", {})

if age_hist:
    st.subheader("Age at enrollment")
    fig = px.bar(
        pd.DataFrame({
            "Age band": [b["label"] for b in age_hist],
            "Participants": [b["count"] for b in age_hist],
        }),
        x="Age band",
        y="Participants",
        text="Participants",
        color="Participants",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(height=380, showlegend=False, coloraxis_showscale=False,
                      xaxis_title="", yaxis_title="Participants")
    st.plotly_chart(fig, use_container_width=True)
    if age_summary.get("median") is not None:
        st.caption(
            f"n = {age_summary.get('n', 0):,}. "
            f"Median {age_summary['median']} years, "
            f"mean {age_summary.get('mean')} years, "
            f"range {age_summary.get('min')}-{age_summary.get('max')} years. "
            "The distribution is bimodal: the pediatric-onset conditions "
            "enroll children, while ALS and FSHD enroll adults."
        )

e1, e2 = st.columns(2)

with e1:
    employment = demographics.get("employment_status", [])
    if employment:
        st.subheader("Employment status")
        static_table(_dist_table(employment, "Employment status"))
        st.caption(
            "Recorded at enrollment. Not collected for all participants, and "
            "not applicable to participants enrolled as children."
        )

with e2:
    education = demographics.get("education_level", [])
    if education:
        st.subheader("Education level")
        static_table(_dist_table(education, "Highest level completed"))
        st.caption(
            "Recorded at enrollment. \"Never attended / Kindergarten only\" "
            "largely reflects participants enrolled as young children."
        )

# --------------------------------------------------------------------------
# Minimum diagnosis
# --------------------------------------------------------------------------

st.markdown("---")
st.header("Diagnosis")
if diagnosis.get("note"):
    st.caption(diagnosis["note"])

age_dx = diagnosis.get("age_at_diagnosis", [])
gen_cf = diagnosis.get("genetic_confirmation", [])

if age_dx:
    st.subheader("Age at diagnosis")
    dx_df = pd.DataFrame([
        {
            "Disease": r["disease"],
            "n": f"{r['n']:,}",
            "Median (years)": r["median"],
            "Mean (years)": r["mean"],
            "Range (years)": f"{r['min']}-{r['max']}",
        }
        for r in age_dx
    ])
    static_table(dx_df)
    fig = px.bar(
        pd.DataFrame({
            "Disease": [r["disease"] for r in age_dx],
            "Median age at diagnosis": [r["median"] for r in age_dx],
        }),
        x="Disease",
        y="Median age at diagnosis",
        text="Median age at diagnosis",
        color="Median age at diagnosis",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, coloraxis_showscale=False, xaxis_title="",
                      yaxis_title="Median age at diagnosis (years)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pompe is not shown: age at diagnosis is not captured in a directly "
        "comparable field. Values are restricted to 0-110 years to exclude "
        "data-entry errors."
    )

if gen_cf:
    st.subheader("Genetic confirmation of diagnosis")
    cf_df = pd.DataFrame([
        {
            "Disease": r["disease"],
            "Answered": f"{r['answered']:,}",
            "Genetically confirmed": f"{r['confirmed']:,}",
            "% confirmed": f"{r['pct']}%",
        }
        for r in gen_cf
    ])
    static_table(cf_df)
    st.caption(
        "Confirmation by laboratory testing or via a tested family member. "
        "Participants answering \"Unknown\" are excluded from the denominator. "
        "ALS and FSHD are not shown: they record genetic findings through "
        "different fields (gene mutation and 4q35 deletion respectively) that "
        "are not comparable to a single confirmation flag."
    )

# --------------------------------------------------------------------------
# Full download
# --------------------------------------------------------------------------

st.markdown("---")
st.subheader("Download")
st.caption(
    "Aggregated statistics only. No individual-level data is included."
)

st.download_button(
    "Download this community snapshot (JSON)",
    json.dumps(profile, indent=2),
    file_name="movr_community_snapshot.json",
    mime="application/json",
)

render_page_footer()
