"""
Shared clinical summary renderers for DMD, LGMD, ALS, and SMA.

These functions are used by both the Disease Explorer (embedded tab) and
the standalone DUA-gated clinical summary pages. Sections are organized
by the 19 canonical clinical domains from config/clinical_domains.yaml.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from api import StatsAPI
from api.dmd import DMDAPI
from api.lgmd import LGMDAPI
from api.als import ALSAPI
from api.sma import SMAAPI
from utils.access import has_access
from components.tables import static_table


def _render_enrollment_charts(disease: str):
    """Render cumulative + monthly enrollment charts for a single disease."""
    try:
        snapshot = StatsAPI.load_snapshot()
    except Exception:
        return
    tl = snapshot.get('enrollment_timeline', {})
    by_dm = tl.get('by_disease_month', [])
    ds_data = [r for r in by_dm if r.get('disease') == disease]
    if not ds_data:
        return

    df = pd.DataFrame(ds_data)
    df['month'] = pd.to_datetime(df['month'])
    df = df.sort_values('month')
    df['cumulative'] = df['count'].cumsum()

    c1, c2 = st.columns(2)
    ds_key = disease.lower().replace(" ", "_")
    with c1:
        fig = px.line(
            df, x='month', y='cumulative',
            title=f'Cumulative {disease} Enrollment',
            labels={'month': '', 'cumulative': 'Participants'},
            color_discrete_sequence=['#1E88E5'],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key=f"enroll_cumul_{ds_key}")
    with c2:
        fig = px.bar(
            df, x='month', y='count',
            title=f'Monthly New {disease} Enrollments',
            labels={'month': '', 'count': 'New Participants'},
            color_discrete_sequence=['#1E88E5'],
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True, key=f"enroll_monthly_{ds_key}")


# ======================================================================
# LGMD Clinical Summary
# ======================================================================

_ALL_DOMAINS = [
    "Demographics & Enrollment", "Disease Classification & Diagnosis",
    "Genetics & Molecular Testing", "Family History",
    "Mobility & Ambulation", "Motor Function Assessments",
    "Timed Function Tests", "Disease Milestones & Progression",
    "Pulmonary & Respiratory", "Cardiology", "Nutrition & GI",
    "Medications & Treatments", "Orthopedic & Surgical",
    "Hospitalizations", "Vital Signs & Anthropometrics",
    "Cognition & Behavioral", "Multidisciplinary Care",
    "Clinical Research", "Discontinuation",
]


def render_lgmd_clinical_summary():
    """Render LGMD clinical summary sections organized by clinical domains."""
    lgmd_snap = LGMDAPI.get_snapshot()
    if not lgmd_snap:
        return

    summary = lgmd_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("LGMD Clinical Summary")

    # =================================================================
    # Disease Classification & Diagnosis — Subtype distribution
    # =================================================================
    subtypes = lgmd_snap.get("subtypes", {})
    dist = subtypes.get("distribution", [])
    diagnosis = lgmd_snap.get("diagnosis", {})
    gc = diagnosis.get("genetic_confirmation", {})

    if dist:
        st.markdown("##### Disease Classification & Diagnosis")
        st.caption("_Subtype distribution_")

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total LGMD Participants", f"{total_patients:,}")
        with m2:
            st.metric("Unique Subtypes", subtypes.get("unique_subtypes", 0))
        with m3:
            pct = gc.get("confirmed_percentage", 0)
            st.metric("Genetic Confirmation", f"{pct:.1f}%")

        # Chart + table
        col_chart, col_table = st.columns([2, 1])

        with col_chart:
            top = dist[:15]
            labels = [d["subtype"] for d in reversed(top)]
            counts = [d["patients"] for d in reversed(top)]
            types = [d.get("lgmd_type", "") for d in reversed(top)]
            colors = ["#E53935" if t == "LGMD Type 1 (Dominant)" else "#1E88E5" for t in types]
            fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h", marker_color=colors))
            fig.update_layout(
                title=f"Top {len(top)} LGMD Subtypes",
                xaxis_title="Participants",
                height=max(350, len(top) * 28 + 80),
                margin=dict(t=40, b=40, l=250),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Red = Dominant (Type 1), Blue = Recessive (Type 2)")

        with col_table:
            rows = [{"Subtype": d["subtype"], "N": d["patients"],
                     "%": f"{d['percentage']:.1f}%"} for d in top]
            static_table(pd.DataFrame(rows))

    # =================================================================
    # Disease Milestones & Progression — Diagnostic journey
    # =================================================================
    journey = lgmd_snap.get("diagnostic_journey", {})
    if journey.get("available"):
        st.markdown("---")
        st.markdown("##### Disease Milestones & Progression")
        st.caption("_Diagnostic journey_")
        st.markdown(
            "LGMD participants often experience a significant delay between symptom onset "
            "and clinical diagnosis due to the heterogeneity of limb-girdle phenotypes."
        )

        # Metric row
        delay = journey.get("delay", {})
        onset_data = journey.get("onset_age", {})
        dx_data = journey.get("diagnosis_age", {})

        m1, m2, m3 = st.columns(3)
        with m1:
            if onset_data.get("count"):
                st.metric("Median Onset Age", f"{onset_data['median']:.1f} yrs",
                          help=f"n={onset_data['count']}")
        with m2:
            if dx_data.get("count"):
                st.metric("Median Diagnosis Age", f"{dx_data['median']:.1f} yrs",
                          help=f"n={dx_data['count']}")
        with m3:
            if delay.get("count") and not delay.get("suppressed"):
                st.metric("Median Diagnostic Delay", f"{delay['median']:.1f} yrs",
                          help=f"n={delay['count']}")

        # Charts row
        col_delay, col_ages = st.columns(2)

        with col_delay:
            if delay.get("histogram") and not delay.get("suppressed"):
                hist = delay["histogram"]
                med = delay.get("median", 0)
                bins_list = hist["bins"]
                bar_colors = []
                for b in bins_list:
                    parts = b.split("-")
                    if len(parts) == 2:
                        lo, hi = float(parts[0]), float(parts[1])
                        bar_colors.append("#E53935" if lo <= med < hi else "#1E88E5")
                    else:
                        bar_colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=bins_list, y=hist["counts"], marker_color=bar_colors,
                ))
                fig.update_layout(
                    title=f"Diagnostic Delay (median {med:.1f} yrs)",
                    xaxis_title="Years from Onset to Diagnosis",
                    yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red bar = bin containing median")

        with col_ages:
            # Onset vs Diagnosis side-by-side
            if onset_data.get("histogram") and dx_data.get("histogram"):
                onset_hist = onset_data["histogram"]
                dx_hist = dx_data["histogram"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=onset_hist["bins"], y=onset_hist["counts"],
                    name=f"Onset Age (n={onset_data['count']})",
                    marker_color="#636EFA", opacity=0.7,
                ))
                fig.add_trace(go.Bar(
                    x=dx_hist["bins"], y=dx_hist["counts"],
                    name=f"Diagnosis Age (n={dx_data['count']})",
                    marker_color="#00CC96", opacity=0.7,
                ))
                fig.update_layout(
                    title="Onset Age vs Diagnosis Age",
                    xaxis_title="Age (years)", yaxis_title="Participants",
                    barmode="overlay", height=350,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Pulmonary & Respiratory (+ Timed Function Tests, Mobility & Ambulation)
    # =================================================================
    func = lgmd_snap.get("functional_scores", {})
    if func.get("available"):
        st.markdown("---")
        st.markdown("##### Pulmonary & Respiratory")
        st.caption("_Also includes: Timed Function Tests, Mobility & Ambulation_")

        fvc = func.get("fvc_pct", {})
        tw = func.get("timed_10m_walk", {})
        amb_func = func.get("ambulatory_status", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            if fvc.get("count"):
                st.metric("FVC % Predicted (median)", f"{fvc['median']:.0f}%",
                          help=f"n={fvc['count']} participants")
        with m2:
            if tw.get("count"):
                st.metric("Timed 10m Walk (median)", f"{tw['median']:.1f}s",
                          help=f"n={tw['count']} participants")
        with m3:
            amb_dist = amb_func.get("distribution", {})
            if amb_dist:
                not_amb = amb_dist.get("Not ambulatory", 0)
                total_amb = amb_func.get("total_reported", 1)
                st.metric("Not Ambulatory", f"{not_amb / total_amb * 100:.0f}%",
                          help=f"{not_amb} of {total_amb} participants reporting")

        # Chart row
        charts_shown = []
        if fvc.get("histogram"):
            charts_shown.append("fvc")
        if amb_dist:
            charts_shown.append("amb")
        amb_by_subtype = lgmd_snap.get("ambulatory", {}).get("by_subtype", {})
        if amb_by_subtype:
            charts_shown.append("amb_sub")

        cols = st.columns(len(charts_shown)) if charts_shown else []
        col_idx = 0

        if "fvc" in charts_shown:
            with cols[col_idx]:
                hist = fvc["histogram"]
                fvc_colors = []
                for b in hist["bins"]:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 40:
                            fvc_colors.append("#E53935")
                        elif mid < 60:
                            fvc_colors.append("#FF9800")
                        elif mid < 80:
                            fvc_colors.append("#FFC107")
                        else:
                            fvc_colors.append("#4CAF50")
                    else:
                        fvc_colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=hist["bins"], y=hist["counts"], marker_color=fvc_colors,
                ))
                fig.update_layout(
                    title=f"FVC % Predicted (n={fvc['count']})",
                    xaxis_title="FVC %", yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red <40% | Orange 40-60% | Yellow 60-80% | Green >80%")
            col_idx += 1

        if "amb" in charts_shown:
            with cols[col_idx]:
                fig = go.Figure(go.Pie(
                    labels=list(amb_dist.keys()),
                    values=list(amb_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title="Ambulatory Status", height=350,
                    margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)
            col_idx += 1

        if "amb_sub" in charts_shown:
            with cols[col_idx]:
                # Ambulatory by subtype grouped bar
                status_labels = ["Ambulatory without difficulty", "Ambulatory with difficulty", "Not ambulatory"]
                status_colors = {"Ambulatory without difficulty": "#4CAF50",
                                 "Ambulatory with difficulty": "#FFC107",
                                 "Not ambulatory": "#E53935"}
                fig = go.Figure()
                for status in status_labels:
                    vals = [amb_by_subtype.get(st_name, {}).get(status, 0) for st_name in amb_by_subtype]
                    fig.add_trace(go.Bar(
                        name=status, x=list(amb_by_subtype.keys()), y=vals,
                        marker_color=status_colors.get(status, "#999"),
                    ))
                fig.update_layout(
                    title="Ambulatory Status by Subtype (Top 5)",
                    xaxis_title="Subtype", yaxis_title="Participants",
                    barmode="group", height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Medications & Treatments
    # =================================================================
    meds = lgmd_snap.get("medications", {})
    if meds.get("available"):
        st.markdown("---")
        st.markdown("##### Medications & Treatments")
        st.markdown(
            "LGMD management focuses on supportive care: cardiac protection, pain management, "
            "respiratory support, and nutritional supplementation. No disease-modifying therapies "
            "are currently approved for LGMD."
        )

        col_cat, col_top = st.columns(2)

        with col_cat:
            categories = meds.get("categories", {})
            if categories:
                cat_names = list(categories.keys())
                cat_pts = [categories[c]["patients"] for c in cat_names]
                cat_pcts = [categories[c]["percentage"] for c in cat_names]
                # Sort by patient count
                sorted_idx = sorted(range(len(cat_pts)), key=lambda i: cat_pts[i], reverse=True)
                fig = go.Figure(go.Bar(
                    y=[cat_names[i] for i in reversed(sorted_idx)],
                    x=[cat_pts[i] for i in reversed(sorted_idx)],
                    orientation="h",
                    marker_color="#1E88E5",
                    text=[f"{cat_pcts[i]:.0f}%" for i in reversed(sorted_idx)],
                    textposition="auto",
                ))
                fig.update_layout(
                    title="Drug Categories (% of LGMD cohort)",
                    xaxis_title="Participants",
                    height=350,
                    margin=dict(t=40, b=40, l=150),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_top:
            top_drugs = meds.get("top_drugs", [])
            if top_drugs:
                labels = [d["drug"] for d in reversed(top_drugs[:10])]
                counts = [d["patients"] for d in reversed(top_drugs[:10])]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h",
                    marker_color="#AB63FA",
                ))
                fig.update_layout(
                    title="Top Medications (by unique participants)",
                    xaxis_title="Participants",
                    height=350,
                    margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Genetics & Molecular Testing (+ Disease Classification & Diagnosis)
    # =================================================================
    clinical = lgmd_snap.get("clinical", {})
    if clinical.get("available"):
        st.markdown("---")
        st.markdown("##### Genetics & Molecular Testing")
        st.caption("_Also includes: Disease Classification & Diagnosis_")

        col_gen, col_biopsy, col_sym = st.columns(3)

        with col_gen:
            gc_dist = gc.get("distribution", {})
            if gc_dist:
                fig = go.Figure(go.Pie(labels=list(gc_dist.keys()), values=list(gc_dist.values()), hole=0.4))
                fig.update_layout(title="Genetic Confirmation", height=300,
                                  margin=dict(t=40, b=20), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

        with col_biopsy:
            mb = clinical.get("muscle_biopsy", {}).get("distribution", {})
            if mb:
                fig = go.Figure(go.Pie(labels=list(mb.keys()), values=list(mb.values()), hole=0.4))
                fig.update_layout(title="Muscle Biopsy", height=300,
                                  margin=dict(t=40, b=20), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

        with col_sym:
            fs = clinical.get("first_symptoms", {}).get("distribution", {})
            if fs:
                items = sorted(fs.items(), key=lambda x: x[1], reverse=True)[:8]
                labels = [x[0] for x in reversed(items)]
                counts = [x[1] for x in reversed(items)]
                fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h",
                                       marker_color="#AB63FA"))
                fig.update_layout(title="First Symptoms (Top 8)",
                                  xaxis_title="Participants",
                                  height=max(250, len(labels) * 28 + 80),
                                  margin=dict(t=40, b=40, l=200))
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Demographics & Enrollment (+ Multidisciplinary Care)
    # =================================================================
    state_dist = lgmd_snap.get("state_distribution", {})
    fac_list = lgmd_snap.get("facilities", {}).get("facilities", [])

    if state_dist.get("available") or fac_list:
        st.markdown("---")
        st.markdown("##### Demographics & Enrollment")
        st.caption("_Also includes: Multidisciplinary Care_")

        col_state, col_sites = st.columns(2)

        with col_state:
            states = state_dist.get("states", [])
            if states:
                labels = [s["state"] for s in reversed(states[:15])]
                counts = [s["total"] for s in reversed(states[:15])]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#00BCD4",
                ))
                fig.update_layout(
                    title=f"Participant Distribution ({state_dist.get('total_states_mapped', 0)} states)",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=180),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("States with <11 participants grouped as 'Other States'")

        with col_sites:
            if fac_list and has_access():
                top_sites = fac_list[:10]
                labels = [s.get("name", f"Site {s.get('id', '')}") for s in reversed(top_sites)]
                counts = [s.get("patients", 0) for s in reversed(top_sites)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#FF7043",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Top 10 Care Sites**")
                st.info("Not available — requires DUA.  "
                        "[Request Access](https://mdausa.tfaforms.net/389761)")

    _render_enrollment_charts("LGMD")

    # =================================================================
    # Other Clinical Domains
    # =================================================================
    _lgmd_covered = {
        "Disease Classification & Diagnosis",
        "Disease Milestones & Progression",
        "Pulmonary & Respiratory",
        "Timed Function Tests",
        "Mobility & Ambulation",
        "Medications & Treatments",
        "Genetics & Molecular Testing",
        "Demographics & Enrollment",
        "Multidisciplinary Care",
    }
    uncovered = [d for d in _ALL_DOMAINS if d not in _lgmd_covered]
    with st.expander(f"Other Clinical Domains ({len(uncovered)} not yet available)"):
        for d in uncovered:
            st.markdown(f"- **{d}**")
        st.caption("Additional domains will be added as analytics are developed.")


# ======================================================================
# DMD Clinical Summary
# ======================================================================

def render_dmd_clinical_summary():
    """Render DMD clinical summary sections organized by clinical domains."""
    dmd_snap = DMDAPI.get_snapshot()
    if not dmd_snap:
        return

    summary = dmd_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("DMD Clinical Summary")

    # =================================================================
    # Medications & Treatments — Exon-skipping therapeutics
    # =================================================================
    tx = dmd_snap.get("therapeutics", {})
    if tx.get("available"):
        st.markdown("##### Medications & Treatments")
        st.caption("_Exon-skipping therapeutics: amenability & utilization_")
        st.markdown(
            "Exon-skipping therapies target specific deletion mutations in the dystrophin gene. "
            "Participants are classified as *amenable* based on mutation type and exon range."
        )

        # Metric row
        total_amenable = tx.get("total_amenable_any_drug", {})
        total_on = tx.get("total_on_any_exon_skipping", {})
        amen_count = total_amenable.get("count", 0)
        on_count = total_on.get("count", 0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total DMD Participants", f"{total_patients:,}")
        with m2:
            pct_amen = f"({amen_count / total_patients * 100:.1f}% of cohort)" if total_patients else ""
            st.metric("Amenable to Any Exon-Skipping", f"{amen_count:,}", delta=pct_amen)
        with m3:
            pct_on = f"({on_count / amen_count * 100:.1f}% of amenable)" if amen_count else ""
            st.metric("On Any Exon-Skipping Therapy", f"{on_count:,}", delta=pct_on)

        # Grouped horizontal bar chart + table
        drugs = tx.get("drugs", [])
        if drugs:
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                drug_names = []
                on_vals = []
                amen_vals = []
                for d in reversed(drugs):
                    drug_names.append(d["drug_name"])
                    on_v = d.get("on_therapy", {})
                    am_v = d.get("amenable_not_on_therapy", {})
                    on_vals.append(on_v.get("count", 0) if not on_v.get("suppressed") else 0)
                    amen_vals.append(am_v.get("count", 0) if not am_v.get("suppressed") else 0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=drug_names, x=on_vals, orientation="h",
                    name="On Therapy", marker_color="#1E88E5",
                ))
                fig.add_trace(go.Bar(
                    y=drug_names, x=amen_vals, orientation="h",
                    name="Amenable, Not on Therapy", marker_color="#FF7043",
                ))
                fig.update_layout(
                    barmode="group",
                    title="Exon-Skipping Drug Utilization vs. Amenability",
                    xaxis_title="Participants",
                    height=max(300, len(drugs) * 60 + 80),
                    margin=dict(t=40, b=40, l=220),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                rows = []
                for d in drugs:
                    on_v = d.get("on_therapy", {})
                    am_not = d.get("amenable_not_on_therapy", {})
                    am_tot = d.get("total_amenable", {})
                    rows.append({
                        "Drug": d["drug_name"].split(" (")[0],
                        "On Therapy": "<11" if on_v.get("suppressed") else on_v.get("count", 0),
                        "Amenable (not on)": "<11" if am_not.get("suppressed") else am_not.get("count", 0),
                        "Total Amenable": "<11" if am_tot.get("suppressed") else am_tot.get("count", 0),
                        "% Cohort": f"{d.get('pct_of_cohort', 0):.1f}%",
                    })
                static_table(pd.DataFrame(rows))
                st.caption(
                    "*Categories with fewer than 11 participants are suppressed "
                    "per HIPAA small-cell guidelines.*"
                )

    # =================================================================
    # Medications & Treatments: Glucocorticoids
    # =================================================================
    steroids = dmd_snap.get("steroids", {})
    if steroids.get("available"):
        st.markdown("##### Medications & Treatments: Glucocorticoids")

        col_glc, col_drugs, col_compare = st.columns(3)

        with col_glc:
            last_enc = steroids.get("glcouse_last_encounter", {})
            dist = last_enc.get("distribution", {})
            if dist:
                fig = go.Figure(go.Pie(
                    labels=list(dist.keys()),
                    values=list(dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title="Steroid Use (Last Encounter)",
                    height=320, margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_drugs:
            sm = steroids.get("steroid_medications", {})
            breakdown = sm.get("breakdown", [])
            if breakdown:
                labels = [b["drug_class"] for b in reversed(breakdown)]
                counts = [b["count"] for b in reversed(breakdown)]
                colors = ["#7B1FA2", "#AB47BC", "#CE93D8"][:len(labels)]
                fig = go.Figure(go.Bar(
                    x=counts, y=labels, orientation="h",
                    marker_color=list(reversed(colors)),
                ))
                fig.update_layout(
                    title="Steroid Drug Class (Unique Participants)",
                    xaxis_title="Participants",
                    height=320, margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_compare:
            first_enc = steroids.get("glcouse_first_encounter", {})
            last_enc = steroids.get("glcouse_last_encounter", {})
            f_dist = first_enc.get("distribution", {})
            l_dist = last_enc.get("distribution", {})

            # Count "currently taking" at first vs last encounter
            taking_keys = [k for k in (set(f_dist) | set(l_dist)) if "currently" in k.lower()]
            first_taking = sum(f_dist.get(k, 0) for k in taking_keys)
            last_taking = sum(l_dist.get(k, 0) for k in taking_keys)
            first_total = first_enc.get("total_reported", 1)
            last_total = last_enc.get("total_reported", 1)

            st.markdown("**Steroid Use Rates**")
            st.metric(
                "At Enrollment",
                f"{first_taking / first_total * 100:.0f}%"
                if first_total else "N/A",
                delta=f"{first_taking:,} of {first_total:,}",
            )
            st.metric(
                "Last Encounter",
                f"{last_taking / last_total * 100:.0f}%"
                if last_total else "N/A",
                delta=f"{last_taking:,} of {last_total:,}",
            )

    # =================================================================
    # Pulmonary & Respiratory (+ Timed Function Tests, Disease Milestones & Progression)
    # =================================================================
    func = dmd_snap.get("functional_scores", {})
    if func.get("available"):
        st.markdown("##### Pulmonary & Respiratory")
        st.caption("_Also includes: Timed Function Tests, Disease Milestones & Progression_")

        # Metric row
        fvc_data = func.get("fvc_pct", {})
        tw_data = func.get("timed_10m_walk", {})
        loa_data = func.get("loss_of_ambulation", {})

        fm1, fm2, fm3 = st.columns(3)
        with fm1:
            if fvc_data.get("count"):
                st.metric(
                    "FVC % Predicted (median)",
                    f"{fvc_data['median']}%",
                    delta=f"n = {fvc_data['count']} participants",
                )
        with fm2:
            if tw_data.get("count"):
                st.metric(
                    "Timed 10m Walk (median)",
                    f"{tw_data['median']}s",
                    delta=f"n = {tw_data['count']} participants",
                )
        with fm3:
            if loa_data:
                age_loss = loa_data.get("age_at_loss_years", {})
                st.metric(
                    "Age at Loss of Ambulation (median)",
                    f"{age_loss.get('median', 'N/A')} years",
                    delta=f"n = {loa_data.get('total_with_data', 0)} participants",
                )

        # Histogram row
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            hist = fvc_data.get("histogram", {})
            if hist:
                bins = hist["bins"]
                counts = hist["counts"]
                median_val = fvc_data.get("median", 0)
                colors = []
                for b in bins:
                    parts = b.split("-")
                    mid = (float(parts[0]) + float(parts[1])) / 2 if len(parts) == 2 else 0
                    if mid < 40:
                        colors.append("#E53935")  # severe
                    elif mid < 60:
                        colors.append("#FF7043")  # moderate
                    elif mid < 80:
                        colors.append("#FFA726")  # mild
                    else:
                        colors.append("#66BB6A")  # normal
                fig = go.Figure(go.Bar(x=bins, y=counts, marker_color=colors))
                fig.update_layout(
                    title=f"FVC % Predicted (median {median_val}%)",
                    xaxis_title="FVC %", yaxis_title="Participants",
                    height=300, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red <40% severe, Orange <60% moderate, Yellow <80% mild, Green normal")

        with fc2:
            hist = tw_data.get("histogram", {})
            if hist:
                fig = go.Figure(go.Bar(
                    x=hist["bins"], y=hist["counts"], marker_color="#1E88E5",
                ))
                fig.update_layout(
                    title=f"Timed 10m Walk (median {tw_data.get('median', '')}s)",
                    xaxis_title="Seconds", yaxis_title="Participants",
                    height=300, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        with fc3:
            age_hist = loa_data.get("age_at_loss_years", {}).get("histogram", {}) if loa_data else {}
            if age_hist:
                median_age = loa_data.get("age_at_loss_years", {}).get("median", 0)
                bins = age_hist["bins"]
                counts = age_hist["counts"]
                colors = ["#E53935" if b.split("-")[0].strip() and float(b.split("-")[0]) <= median_age < float(b.split("-")[1]) else "#42A5F5" for b in bins if "-" in b]
                if len(colors) != len(bins):
                    colors = ["#42A5F5"] * len(bins)
                fig = go.Figure(go.Bar(x=bins, y=counts, marker_color=colors))
                fig.update_layout(
                    title=f"Age at Loss of Ambulation (median {median_age} yr)",
                    xaxis_title="Age (years)", yaxis_title="Participants",
                    xaxis_type="category",
                    height=300, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Longitudinal trend row
        fvc_long = fvc_data.get("longitudinal", [])
        tw_long = tw_data.get("longitudinal", [])
        if fvc_long or tw_long:
            lc1, lc2 = st.columns(2)

            with lc1:
                if fvc_long:
                    years = [str(p["year"]) for p in fvc_long]
                    medians = [p["median"] for p in fvc_long]
                    q1s = [p["q1"] for p in fvc_long]
                    q3s = [p["q3"] for p in fvc_long]
                    ns = [p["n"] for p in fvc_long]

                    fig = go.Figure()
                    # IQR ribbon
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=q3s + q1s[::-1],
                        fill="toself", fillcolor="rgba(30,136,229,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR (Q1-Q3)", showlegend=True,
                    ))
                    # Median line
                    fig.add_trace(go.Scatter(
                        x=years, y=medians,
                        mode="lines+markers", line=dict(color="#1E88E5", width=3),
                        name="Median",
                        text=[f"n={n}" for n in ns], textposition="top center",
                    ))
                    fig.update_layout(
                        title="FVC % Predicted Over Time",
                        xaxis_title="Years Since Enrollment",
                        yaxis_title="FVC % Predicted",
                        height=350, yaxis=dict(range=[0, 120]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with lc2:
                if tw_long:
                    years = [str(p["year"]) for p in tw_long]
                    medians = [p["median"] for p in tw_long]
                    q1s = [p["q1"] for p in tw_long]
                    q3s = [p["q3"] for p in tw_long]
                    ns = [p["n"] for p in tw_long]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=q3s + q1s[::-1],
                        fill="toself", fillcolor="rgba(255,112,67,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR (Q1-Q3)", showlegend=True,
                    ))
                    fig.add_trace(go.Scatter(
                        x=years, y=medians,
                        mode="lines+markers", line=dict(color="#FF7043", width=3),
                        name="Median",
                        text=[f"n={n}" for n in ns], textposition="top center",
                    ))
                    fig.update_layout(
                        title="Timed 10m Walk Over Time",
                        xaxis_title="Years Since Enrollment",
                        yaxis_title="Seconds",
                        height=350,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "*Functional outcomes reflect latest available encounter per participant. "
            "Longitudinal trends show median with interquartile range (IQR) by years since enrollment.*"
        )

    # =================================================================
    # Genetics & Molecular Testing
    # =================================================================
    genetics = dmd_snap.get("genetics", {})
    if genetics.get("available"):
        st.markdown("##### Genetics & Molecular Testing")

        col_gc, col_mut, col_frame = st.columns(3)

        with col_gc:
            gc = genetics.get("genetic_confirmation", {})
            if gc:
                pct = gc.get("confirmed_percentage", 0)
                st.metric("Genetic Confirmation", f"{pct:.1f}%")
                dist = gc.get("distribution", {})
                if dist:
                    fig = go.Figure(go.Pie(
                        labels=list(dist.keys()), values=list(dist.values()), hole=0.4,
                    ))
                    fig.update_layout(
                        title="Genetic Confirmation", height=300,
                        margin=dict(t=40, b=20), showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col_mut:
            mt = genetics.get("mutation_type", {})
            dist = mt.get("distribution", [])
            if dist:
                items = [d for d in dist if not d.get("suppressed")]
                suppressed = [d for d in dist if d.get("suppressed")]
                if suppressed:
                    items.append({"label": "Suppressed (n<11)", "count": len(suppressed)})
                labels = [d["label"] for d in reversed(items)]
                counts = [d["count"] for d in reversed(items)]
                fig = go.Figure(go.Bar(
                    x=counts, y=labels, orientation="h", marker_color="#00897B",
                ))
                fig.update_layout(
                    title="Mutation Type", xaxis_title="Participants",
                    height=max(300, len(items) * 28 + 80),
                    margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_frame:
            ft = genetics.get("frame_type", {})
            dist = ft.get("distribution", [])
            if dist:
                items = [d for d in dist if not d.get("suppressed")]
                labels = [d["label"] for d in reversed(items)]
                counts = [d["count"] for d in reversed(items)]
                fig = go.Figure(go.Bar(
                    x=counts, y=labels, orientation="h", marker_color="#26A69A",
                ))
                fig.update_layout(
                    title="Frame Type", xaxis_title="Participants",
                    height=300, margin=dict(t=40, b=40, l=150),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Demographics & Enrollment — Geographic distribution
    # =================================================================
    state_dist = dmd_snap.get("state_distribution", {})
    if state_dist.get("available"):
        states = state_dist.get("states", [])
        if states:
            st.markdown("##### Demographics & Enrollment")
            st.caption("_Geographic distribution & therapy utilization by state_")

            # Top 15 states bar chart
            top_states = states[:15]
            state_names = [s["state"] for s in reversed(top_states)]
            totals = [s["total"] for s in reversed(top_states)]
            on_therapy = [s["on_therapy"] for s in reversed(top_states)]

            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=state_names, x=on_therapy, orientation="h",
                    name="On Exon-Skipping Therapy", marker_color="#1E88E5",
                ))
                # "Rest" = total minus on_therapy
                rest = [t - o for t, o in zip(totals, on_therapy)]
                fig.add_trace(go.Bar(
                    y=state_names, x=rest, orientation="h",
                    name="Other DMD Participants", marker_color="#BDBDBD",
                ))
                fig.update_layout(
                    barmode="stack",
                    title="Top States by DMD Participant Count",
                    xaxis_title="Participants",
                    height=max(350, len(top_states) * 28 + 80),
                    margin=dict(t=40, b=40, l=120),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                rows = []
                for s in top_states:
                    rows.append({
                        "State": s["state"],
                        "Total": s["total"],
                        "On Therapy": s["on_therapy"],
                        "Amenable": s["amenable_not_on"],
                    })
                static_table(pd.DataFrame(rows))
                st.caption(
                    "*States with fewer than 11 DMD participants are grouped per "
                    "HIPAA small-cell suppression guidelines.*"
                )

    _render_enrollment_charts("DMD")

    # =================================================================
    # Mobility & Ambulation (+ Multidisciplinary Care)
    # =================================================================
    amb = dmd_snap.get("ambulatory", {})
    amb_dist = amb.get("current_status", {}).get("distribution", {})
    fac_list = dmd_snap.get("facilities", {}).get("facilities", [])

    if amb_dist or fac_list:
        st.markdown("##### Mobility & Ambulation")
        st.caption("_Also includes: Multidisciplinary Care_")
        col_amb, col_sites = st.columns(2)

        with col_amb:
            if amb_dist:
                fig = go.Figure(go.Pie(
                    labels=list(amb_dist.keys()),
                    values=list(amb_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title="Ambulatory Status (Latest Encounter)",
                    height=300, margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_sites:
            if fac_list and has_access():
                top_sites = fac_list[:10]
                labels = [s.get("name", f"Site {s.get('id', i + 1)}") for i, s in enumerate(top_sites)]
                counts = [s.get("patients", 0) for s in reversed(top_sites)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h",
                    marker_color="#00BCD4",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Top 10 Care Sites**")
                st.info("Not available — requires DUA.  "
                        "[Request Access](https://mdausa.tfaforms.net/389761)")

    # =================================================================
    # Other Clinical Domains
    # =================================================================
    _dmd_covered = {
        "Medications & Treatments",
        "Pulmonary & Respiratory",
        "Timed Function Tests",
        "Disease Milestones & Progression",
        "Genetics & Molecular Testing",
        "Demographics & Enrollment",
        "Mobility & Ambulation",
        "Multidisciplinary Care",
    }
    uncovered = [d for d in _ALL_DOMAINS if d not in _dmd_covered]
    with st.expander(f"Other Clinical Domains ({len(uncovered)} not yet available)"):
        for d in uncovered:
            st.markdown(f"- **{d}**")
        st.caption("Additional domains will be added as analytics are developed.")


# ======================================================================
# ALS Clinical Summary
# ======================================================================

def render_als_clinical_summary():
    """Render ALS clinical summary sections organized by clinical domains."""
    als_snap = ALSAPI.get_snapshot()
    if not als_snap:
        return

    summary = als_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("ALS Clinical Summary")

    # =================================================================
    # Motor Function Assessments — ALSFRS-R (Hero Section)
    # =================================================================
    alsfrs = als_snap.get("alsfrs", {})
    if alsfrs.get("available"):
        st.markdown("##### Motor Function Assessments")
        st.caption("_ALSFRS-R (Revised ALS Functional Rating Scale) — 0-48 scale, higher = better function_")

        ts = alsfrs.get("total_score", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total ALS Participants", f"{total_patients:,}")
        with m2:
            if ts.get("count"):
                st.metric("ALSFRS-R Median", f"{ts['median']:.0f} / 48",
                          delta=f"n = {ts['count']} participants")
        with m3:
            long_n = alsfrs.get("patients_with_longitudinal", 0)
            if long_n:
                st.metric("Longitudinal Participants", f"{long_n:,}",
                          delta="2+ ALSFRS-R measurements")

        # Chart row: histogram + longitudinal
        col_hist, col_long = st.columns(2)

        with col_hist:
            hist = ts.get("histogram", {})
            if hist:
                bins_list = hist["bins"]
                counts = hist["counts"]
                # Color by severity band
                colors = []
                for b in bins_list:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 12:
                            colors.append("#B71C1C")   # very severe - dark red
                        elif mid < 24:
                            colors.append("#E53935")   # severe - red
                        elif mid < 36:
                            colors.append("#FFA726")   # moderate - orange
                        else:
                            colors.append("#4CAF50")   # mild - green
                    else:
                        colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=bins_list, y=counts, marker_color=colors,
                ))
                fig.update_layout(
                    title=f"ALSFRS-R Score Distribution (n={ts.get('count', 0)})",
                    xaxis_title="ALSFRS-R Total Score",
                    yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Green = Mild (36-48) | Orange = Moderate (24-35) | Red = Severe (12-23) | Dark Red = Very Severe (0-11)")

        with col_long:
            longitudinal = ts.get("longitudinal", [])
            if longitudinal:
                years = [str(p["year"]) for p in longitudinal]
                medians = [p["median"] for p in longitudinal]
                q1s = [p["q1"] for p in longitudinal]
                q3s = [p["q3"] for p in longitudinal]
                ns = [p["n"] for p in longitudinal]

                fig = go.Figure()
                # IQR ribbon
                fig.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=q3s + q1s[::-1],
                    fill="toself", fillcolor="rgba(30,136,229,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IQR (Q1-Q3)", showlegend=True,
                ))
                # Median line
                fig.add_trace(go.Scatter(
                    x=years, y=medians,
                    mode="lines+markers", line=dict(color="#1E88E5", width=3),
                    name="Median",
                    text=[f"n={n}" for n in ns], textposition="top center",
                ))
                fig.update_layout(
                    title="ALSFRS-R Score Over Time",
                    xaxis_title="Years Since Enrollment",
                    yaxis_title="ALSFRS-R Total Score",
                    height=350, yaxis=dict(range=[0, 50]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Longitudinal ALSFRS-R data not available in snapshot.")

        # Severity band table
        severity = alsfrs.get("severity_bands", [])
        if severity:
            rows = []
            for band in severity:
                rows.append({
                    "Severity": band["label"],
                    "Participants": "<11" if band.get("suppressed") else band.get("count", 0),
                    "%": f"{band.get('percentage', 0):.1f}%",
                })
            static_table(pd.DataFrame(rows))

    # =================================================================
    # Disease Classification & Diagnosis — El Escorial + Body Region
    # =================================================================
    dx = als_snap.get("diagnosis", {})
    ee = dx.get("el_escorial", {})
    br = dx.get("body_region_onset", {})

    if ee or br:
        st.markdown("---")
        st.markdown("##### Disease Classification & Diagnosis")
        st.caption("_El Escorial diagnostic criteria & body region of symptom onset_")

        col_ee, col_br = st.columns(2)

        with col_ee:
            if ee:
                dist = ee.get("distribution", {})
                if dist:
                    fig = go.Figure(go.Pie(
                        labels=list(dist.keys()),
                        values=list(dist.values()),
                        hole=0.4,
                    ))
                    fig.update_layout(
                        title=f"El Escorial Criteria (n={ee.get('total_reported', 0)})",
                        height=350, margin=dict(t=40, b=20), showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col_br:
            if br:
                dist = br.get("distribution", {})
                if dist:
                    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                    labels = [x[0] for x in reversed(items)]
                    counts = [x[1] for x in reversed(items)]
                    fig = go.Figure(go.Bar(
                        x=counts, y=labels, orientation="h",
                        marker_color="#1E88E5",
                    ))
                    fig.update_layout(
                        title=f"Body Region First Affected (n={br.get('total_reported', 0)})",
                        xaxis_title="Participants",
                        height=350,
                        margin=dict(t=40, b=40, l=200),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Disease Milestones & Progression
    # =================================================================
    ms = als_snap.get("milestones", {})
    if ms.get("available"):
        st.markdown("---")
        st.markdown("##### Disease Milestones & Progression")
        st.caption("_Diagnostic journey & functional milestones_")

        # Metric row
        onset = ms.get("onset_age", {})
        dx_age = ms.get("diagnosis_age", {})
        delay = ms.get("diagnostic_delay", {})

        m1, m2, m3 = st.columns(3)
        with m1:
            if onset.get("count"):
                st.metric("Median Onset Age", f"{onset['median']:.1f} yrs",
                          help=f"n={onset['count']}")
        with m2:
            if dx_age.get("count"):
                st.metric("Median Diagnosis Age", f"{dx_age['median']:.1f} yrs",
                          help=f"n={dx_age['count']}")
        with m3:
            if delay.get("count"):
                st.metric("Median Diagnostic Delay", f"{delay['median']:.1f} yrs",
                          help=f"n={delay['count']}")

        # Chart row: delay + onset vs diagnosis
        col_delay, col_ages = st.columns(2)

        with col_delay:
            if delay.get("histogram"):
                hist = delay["histogram"]
                med = delay.get("median", 0)
                bins_list = hist["bins"]
                bar_colors = []
                for b in bins_list:
                    parts = b.split("-")
                    if len(parts) == 2:
                        lo, hi = float(parts[0]), float(parts[1])
                        bar_colors.append("#E53935" if lo <= med < hi else "#1E88E5")
                    else:
                        bar_colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=bins_list, y=hist["counts"], marker_color=bar_colors,
                ))
                fig.update_layout(
                    title=f"Diagnostic Delay (median {med:.1f} yrs)",
                    xaxis_title="Years from Onset to Diagnosis",
                    yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red bar = bin containing median")

        with col_ages:
            if onset.get("histogram") and dx_age.get("histogram"):
                onset_hist = onset["histogram"]
                dx_hist = dx_age["histogram"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=onset_hist["bins"], y=onset_hist["counts"],
                    name=f"Onset Age (n={onset['count']})",
                    marker_color="#636EFA", opacity=0.7,
                ))
                fig.add_trace(go.Bar(
                    x=dx_hist["bins"], y=dx_hist["counts"],
                    name=f"Diagnosis Age (n={dx_age['count']})",
                    marker_color="#00CC96", opacity=0.7,
                ))
                fig.update_layout(
                    title="Onset Age vs Diagnosis Age",
                    xaxis_title="Age (years)", yaxis_title="Participants",
                    barmode="overlay", height=350,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Milestone summary row
        loa = ms.get("loss_of_ambulation", {})
        speech = ms.get("loss_of_speech", {})
        gast = ms.get("gastrostomy", {})
        niv = ms.get("niv_initiation", {})

        milestone_items = []
        if loa.get("distribution"):
            yes_count = sum(v for k, v in loa["distribution"].items()
                           if "yes" in k.lower())
            milestone_items.append(("Loss of Ambulation", yes_count, loa.get("total_reported", 0)))
        if speech.get("distribution"):
            yes_count = sum(v for k, v in speech["distribution"].items()
                           if "yes" in k.lower())
            milestone_items.append(("Loss of Speech", yes_count, speech.get("total_reported", 0)))
        if gast.get("distribution"):
            yes_count = sum(v for k, v in gast["distribution"].items()
                           if "yes" in k.lower())
            milestone_items.append(("Gastrostomy", yes_count, gast.get("total_reported", 0)))
        if niv.get("distribution"):
            yes_count = sum(v for k, v in niv["distribution"].items()
                           if "yes" in k.lower())
            milestone_items.append(("NIV Initiation", yes_count, niv.get("total_reported", 0)))

        if milestone_items:
            cols = st.columns(len(milestone_items))
            for i, (label, yes_n, total_n) in enumerate(milestone_items):
                with cols[i]:
                    pct = round(yes_n / total_n * 100, 0) if total_n else 0
                    st.metric(label, f"{pct:.0f}%",
                              delta=f"{yes_n} of {total_n} reporting")

    # =================================================================
    # Pulmonary & Respiratory
    # =================================================================
    resp = als_snap.get("respiratory", {})
    if resp.get("available"):
        st.markdown("---")
        st.markdown("##### Pulmonary & Respiratory")
        st.caption("_FVC % predicted & respiratory support_")

        fvc = resp.get("fvc_pct", {})
        niv_status = resp.get("niv_status", {})
        trach_status = resp.get("trach_status", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            if fvc.get("count"):
                st.metric("FVC % Predicted (median)", f"{fvc['median']:.0f}%",
                          help=f"n={fvc['count']} participants")
        with m2:
            niv_dist = niv_status.get("distribution", {})
            if niv_dist:
                yes_niv = sum(v for k, v in niv_dist.items() if "yes" in k.lower())
                total_niv = niv_status.get("total_reported", 1)
                st.metric("NIV Use", f"{yes_niv / total_niv * 100:.0f}%",
                          help=f"{yes_niv} of {total_niv} reporting")
        with m3:
            trach_dist = trach_status.get("distribution", {})
            if trach_dist:
                yes_trach = sum(v for k, v in trach_dist.items() if "yes" in k.lower())
                total_trach = trach_status.get("total_reported", 1)
                st.metric("Tracheostomy", f"{yes_trach / total_trach * 100:.0f}%",
                          help=f"{yes_trach} of {total_trach} reporting")

        # Chart row: FVC histogram + longitudinal
        charts_shown = []
        if fvc.get("histogram"):
            charts_shown.append("fvc_hist")
        if fvc.get("longitudinal"):
            charts_shown.append("fvc_long")
        elif niv_dist:
            charts_shown.append("niv_pie")

        cols = st.columns(len(charts_shown)) if charts_shown else []
        col_idx = 0

        if "fvc_hist" in charts_shown:
            with cols[col_idx]:
                hist = fvc["histogram"]
                fvc_colors = []
                for b in hist["bins"]:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 40:
                            fvc_colors.append("#E53935")
                        elif mid < 60:
                            fvc_colors.append("#FF9800")
                        elif mid < 80:
                            fvc_colors.append("#FFC107")
                        else:
                            fvc_colors.append("#4CAF50")
                    else:
                        fvc_colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=hist["bins"], y=hist["counts"], marker_color=fvc_colors,
                ))
                fig.update_layout(
                    title=f"FVC % Predicted (n={fvc['count']})",
                    xaxis_title="FVC %", yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red <40% | Orange 40-60% | Yellow 60-80% | Green >80%")
            col_idx += 1

        if "fvc_long" in charts_shown:
            with cols[col_idx]:
                longitudinal = fvc["longitudinal"]
                years = [str(p["year"]) for p in longitudinal]
                medians = [p["median"] for p in longitudinal]
                q1s = [p["q1"] for p in longitudinal]
                q3s = [p["q3"] for p in longitudinal]
                ns = [p["n"] for p in longitudinal]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=q3s + q1s[::-1],
                    fill="toself", fillcolor="rgba(30,136,229,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IQR (Q1-Q3)",
                ))
                fig.add_trace(go.Scatter(
                    x=years, y=medians,
                    mode="lines+markers", line=dict(color="#1E88E5", width=3),
                    name="Median",
                    text=[f"n={n}" for n in ns], textposition="top center",
                ))
                fig.update_layout(
                    title="FVC % Predicted Over Time",
                    xaxis_title="Years Since Enrollment",
                    yaxis_title="FVC % Predicted",
                    height=350, yaxis=dict(range=[0, 120]),
                )
                st.plotly_chart(fig, use_container_width=True)
            col_idx += 1

        if "niv_pie" in charts_shown:
            with cols[col_idx]:
                fig = go.Figure(go.Pie(
                    labels=list(niv_dist.keys()),
                    values=list(niv_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title="NIV Status", height=350,
                    margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Medications & Treatments
    # =================================================================
    meds = als_snap.get("medications", {})
    if meds.get("available"):
        st.markdown("---")
        st.markdown("##### Medications & Treatments")
        st.markdown(
            "ALS disease-modifying therapies include riluzole (the first FDA-approved treatment), "
            "edaravone (Radicava), and emerging therapies. Nuedexta addresses pseudobulbar affect."
        )

        als_drugs = meds.get("als_drugs", {})

        # Metric row for key drugs
        drug_items = list(als_drugs.items())
        if drug_items:
            cols = st.columns(min(len(drug_items), 4))
            for i, (drug_name, drug_data) in enumerate(drug_items[:4]):
                with cols[i]:
                    if drug_data.get("suppressed"):
                        st.metric(drug_name.split(" (")[0], "<11 participants")
                    else:
                        st.metric(
                            drug_name.split(" (")[0],
                            f"{drug_data.get('percentage', 0):.1f}%",
                            delta=f"{drug_data.get('count', 0)} participants",
                        )

        # Charts
        col_drugs, col_top = st.columns(2)

        with col_drugs:
            # ALS drug utilization bar
            drug_labels = []
            drug_counts = []
            for name, data in als_drugs.items():
                if not data.get("suppressed") and data.get("count", 0) > 0:
                    drug_labels.append(name)
                    drug_counts.append(data["count"])
            if drug_labels:
                fig = go.Figure(go.Bar(
                    y=list(reversed(drug_labels)),
                    x=list(reversed(drug_counts)),
                    orientation="h",
                    marker_color="#1E88E5",
                ))
                fig.update_layout(
                    title="ALS Disease Therapies (Unique Participants)",
                    xaxis_title="Participants",
                    height=max(300, len(drug_labels) * 50 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_top:
            top_drugs = meds.get("top_drugs", [])
            if top_drugs:
                labels = [d["drug"] for d in reversed(top_drugs[:10])]
                counts = [d["patients"] for d in reversed(top_drugs[:10])]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h",
                    marker_color="#AB63FA",
                ))
                fig.update_layout(
                    title="Top Medications (by unique participants)",
                    xaxis_title="Participants",
                    height=350,
                    margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Genetics & Molecular Testing
    # =================================================================
    gm = dx.get("gene_mutation", {}) if dx else {}
    fh = dx.get("family_history", {}) if dx else {}

    if gm or fh:
        st.markdown("---")
        st.markdown("##### Genetics & Molecular Testing")
        st.caption("_Gene mutation testing & familial ALS_")

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            if gm.get("tested_percentage"):
                st.metric("Genetic Testing Rate", f"{gm['tested_percentage']:.1f}%",
                          help=f"{gm.get('tested_count', 0)} of {gm.get('total_reported', 0)} tested")
        with m2:
            if fh.get("familial_percentage"):
                st.metric("Familial ALS", f"{fh['familial_percentage']:.1f}%",
                          help=f"{fh.get('familial_count', 0)} participants")
        with m3:
            st.metric("Total ALS Participants", f"{total_patients:,}")

        col_gm, col_fh = st.columns(2)

        with col_gm:
            gm_dist = gm.get("distribution", {})
            if gm_dist:
                fig = go.Figure(go.Pie(
                    labels=list(gm_dist.keys()),
                    values=list(gm_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title=f"Gene Mutation Testing (n={gm.get('total_reported', 0)})",
                    height=350, margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_fh:
            fh_dist = fh.get("distribution", {})
            if fh_dist:
                fig = go.Figure(go.Pie(
                    labels=list(fh_dist.keys()),
                    values=list(fh_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title=f"Family History of ALS/FTD (n={fh.get('total_reported', 0)})",
                    height=350, margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Demographics & Enrollment (+ Multidisciplinary Care)
    # =================================================================
    state_dist = als_snap.get("state_distribution", {})
    fac_list = als_snap.get("facilities", {}).get("facilities", [])

    if state_dist.get("available") or fac_list:
        st.markdown("---")
        st.markdown("##### Demographics & Enrollment")
        st.caption("_Also includes: Multidisciplinary Care_")

        col_state, col_sites = st.columns(2)

        with col_state:
            states = state_dist.get("states", [])
            if states:
                labels = [s["state"] for s in reversed(states[:15])]
                counts = [s["total"] for s in reversed(states[:15])]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#00BCD4",
                ))
                fig.update_layout(
                    title=f"Participant Distribution ({state_dist.get('total_states_mapped', 0)} states)",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=180),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("States with <11 participants grouped as 'Other States'")

        with col_sites:
            if fac_list and has_access():
                top_sites = fac_list[:10]
                labels = [s.get("name", f"Site {s.get('id', '')}") for s in reversed(top_sites)]
                counts = [s.get("patients", 0) for s in reversed(top_sites)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#FF7043",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Top 10 Care Sites**")
                st.info("Not available — requires DUA.  "
                        "[Request Access](https://mdausa.tfaforms.net/389761)")

    _render_enrollment_charts("ALS")

    # =================================================================
    # Other Clinical Domains
    # =================================================================
    _als_covered = {
        "Motor Function Assessments",
        "Disease Classification & Diagnosis",
        "Disease Milestones & Progression",
        "Pulmonary & Respiratory",
        "Medications & Treatments",
        "Genetics & Molecular Testing",
        "Demographics & Enrollment",
        "Multidisciplinary Care",
    }
    uncovered = [d for d in _ALL_DOMAINS if d not in _als_covered]
    with st.expander(f"Other Clinical Domains ({len(uncovered)} not yet available)"):
        for d in uncovered:
            st.markdown(f"- **{d}**")
        st.caption("Additional domains will be added as analytics are developed.")


# ======================================================================
# SMA Clinical Summary
# ======================================================================

def _short_sma_type(label: str) -> str:
    """Shorten SMA type labels for chart display."""
    if "Type 0" in label:
        return "Type 0"
    if "Type 1" in label:
        return "Type 1"
    if "Type 2" in label:
        return "Type 2"
    if "Type 3" in label:
        return "Type 3"
    if "Pre-symptomatic" in label:
        return "Pre-sympt."
    return label[:20]


def render_sma_clinical_summary():
    """Render SMA clinical summary sections organized by clinical domains."""
    sma_snap = SMAAPI.get_snapshot()
    if not sma_snap:
        return

    summary = sma_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("SMA Clinical Summary")

    # =================================================================
    # Motor Function Assessments — HFMSE + CHOP-INTEND (Hero Section)
    # =================================================================
    motor = sma_snap.get("motor_scores", {})
    if motor.get("available"):
        st.markdown("##### Motor Function Assessments")
        st.caption(
            "_HFMSE (0-66, Type 2/3), CHOP-INTEND (0-64, Type 1/pre-symptomatic), "
            "RULM (0-37, upper limb) — higher = better function_"
        )

        hfmse = motor.get("hfmse", {})
        chop = motor.get("chop_intend", {})
        rulm = motor.get("rulm", {})

        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total SMA Participants", f"{total_patients:,}")
        with m2:
            ts = hfmse.get("total_score", {})
            if ts.get("count"):
                st.metric("HFMSE Median", f"{ts['median']:.0f} / 66",
                          delta=f"n = {ts['count']} participants")
        with m3:
            ts = chop.get("total_score", {})
            if ts.get("count"):
                st.metric("CHOP-INTEND Median", f"{ts['median']:.0f} / 64",
                          delta=f"n = {ts['count']} participants")
        with m4:
            ts = rulm.get("total_score", {})
            if ts.get("count"):
                st.metric("RULM Median", f"{ts['median']:.0f} / 37",
                          delta=f"n = {ts['count']} participants")

        # Histogram row: HFMSE + CHOP-INTEND
        col_hfmse, col_chop = st.columns(2)

        with col_hfmse:
            ts = hfmse.get("total_score", {})
            hist = ts.get("histogram", {})
            if hist:
                bins_list = hist["bins"]
                counts = hist["counts"]
                colors = []
                for b in bins_list:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 15:
                            colors.append("#E53935")
                        elif mid < 30:
                            colors.append("#FF9800")
                        elif mid < 50:
                            colors.append("#FFC107")
                        else:
                            colors.append("#4CAF50")
                    else:
                        colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=bins_list, y=counts, marker_color=colors,
                ))
                fig.update_layout(
                    title=f"HFMSE Score Distribution (n={ts.get('count', 0)})",
                    xaxis_title="HFMSE Total Score (0-66)",
                    yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_chop:
            ts = chop.get("total_score", {})
            hist = ts.get("histogram", {})
            if hist:
                bins_list = hist["bins"]
                counts = hist["counts"]
                colors = []
                for b in bins_list:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 16:
                            colors.append("#E53935")
                        elif mid < 32:
                            colors.append("#FF9800")
                        elif mid < 48:
                            colors.append("#FFC107")
                        else:
                            colors.append("#4CAF50")
                    else:
                        colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=bins_list, y=counts, marker_color=colors,
                ))
                fig.update_layout(
                    title=f"CHOP-INTEND Score Distribution (n={ts.get('count', 0)})",
                    xaxis_title="CHOP-INTEND Total Score (0-64)",
                    yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Red = Severe | Orange = Moderate | Yellow = Mild | Green = Near-normal function. "
            "HFMSE is used primarily for Type 2/3; CHOP-INTEND for Type 1 and pre-symptomatic."
        )

        # Longitudinal trend row: HFMSE + CHOP-INTEND
        hfmse_long = hfmse.get("longitudinal", [])
        chop_long = chop.get("longitudinal", [])
        if hfmse_long or chop_long:
            lc1, lc2 = st.columns(2)

            with lc1:
                if hfmse_long:
                    years = [str(p["year"]) for p in hfmse_long]
                    medians = [p["median"] for p in hfmse_long]
                    q1s = [p["q1"] for p in hfmse_long]
                    q3s = [p["q3"] for p in hfmse_long]
                    ns = [p["n"] for p in hfmse_long]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=q3s + q1s[::-1],
                        fill="toself", fillcolor="rgba(30,136,229,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR (Q1-Q3)", showlegend=True,
                    ))
                    fig.add_trace(go.Scatter(
                        x=years, y=medians,
                        mode="lines+markers", line=dict(color="#1E88E5", width=3),
                        name="Median",
                        text=[f"n={n}" for n in ns], textposition="top center",
                    ))
                    fig.update_layout(
                        title="HFMSE Score Over Time",
                        xaxis_title="Years Since Enrollment",
                        yaxis_title="HFMSE Total Score",
                        height=350, yaxis=dict(range=[0, 70]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with lc2:
                if chop_long:
                    years = [str(p["year"]) for p in chop_long]
                    medians = [p["median"] for p in chop_long]
                    q1s = [p["q1"] for p in chop_long]
                    q3s = [p["q3"] for p in chop_long]
                    ns = [p["n"] for p in chop_long]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years + years[::-1],
                        y=q3s + q1s[::-1],
                        fill="toself", fillcolor="rgba(255,112,67,0.15)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="IQR (Q1-Q3)", showlegend=True,
                    ))
                    fig.add_trace(go.Scatter(
                        x=years, y=medians,
                        mode="lines+markers", line=dict(color="#FF7043", width=3),
                        name="Median",
                        text=[f"n={n}" for n in ns], textposition="top center",
                    ))
                    fig.update_layout(
                        title="CHOP-INTEND Score Over Time",
                        xaxis_title="Years Since Enrollment",
                        yaxis_title="CHOP-INTEND Total Score",
                        height=350, yaxis=dict(range=[0, 68]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # By-SMA-type grouped bar
        hfmse_by_type = hfmse.get("by_sma_type", {})
        chop_by_type = chop.get("by_sma_type", {})
        if hfmse_by_type or chop_by_type:
            st.markdown("**Motor Scores by SMA Type**")
            col_bt1, col_bt2 = st.columns(2)

            with col_bt1:
                if hfmse_by_type:
                    types = list(hfmse_by_type.keys())
                    short_labels = [_short_sma_type(t) for t in types]
                    medians = [hfmse_by_type[t].get("median", 0) for t in types]
                    ns = [hfmse_by_type[t].get("count", 0) for t in types]
                    fig = go.Figure(go.Bar(
                        x=short_labels, y=medians,
                        marker_color="#1E88E5",
                        text=[f"n={n}" for n in ns],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="HFMSE Median by SMA Type",
                        yaxis_title="Median Score",
                        height=350, yaxis=dict(range=[0, 70]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_bt2:
                if chop_by_type:
                    types = list(chop_by_type.keys())
                    short_labels = [_short_sma_type(t) for t in types]
                    medians = [chop_by_type[t].get("median", 0) for t in types]
                    ns = [chop_by_type[t].get("count", 0) for t in types]
                    fig = go.Figure(go.Bar(
                        x=short_labels, y=medians,
                        marker_color="#FF7043",
                        text=[f"n={n}" for n in ns],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="CHOP-INTEND Median by SMA Type",
                        yaxis_title="Median Score",
                        height=350, yaxis=dict(range=[0, 68]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # RULM summary
        rulm_ts = rulm.get("total_score", {})
        if rulm_ts.get("count"):
            with st.expander(f"RULM Upper Limb Function (n={rulm_ts['count']})"):
                rm1, rm2, rm3 = st.columns(3)
                with rm1:
                    st.metric("RULM Median", f"{rulm_ts['median']:.0f} / 37")
                with rm2:
                    st.metric("RULM Mean", f"{rulm_ts['mean']:.1f}")
                with rm3:
                    long_n = rulm.get("patients_with_longitudinal", 0)
                    st.metric("Longitudinal Participants", f"{long_n:,}")

                rulm_hist = rulm_ts.get("histogram", {})
                if rulm_hist:
                    fig = go.Figure(go.Bar(
                        x=rulm_hist["bins"], y=rulm_hist["counts"],
                        marker_color="#AB63FA",
                    ))
                    fig.update_layout(
                        title=f"RULM Score Distribution (n={rulm_ts['count']})",
                        xaxis_title="RULM Total Score (0-37)",
                        yaxis_title="Participants",
                        showlegend=False, height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "*Motor scores reflect latest available encounter per participant. "
            "Longitudinal trends show median with interquartile range (IQR) by years since enrollment.*"
        )

    # =================================================================
    # Disease Classification & Diagnosis
    # =================================================================
    classification = sma_snap.get("classification", {})
    if classification.get("available"):
        st.markdown("---")
        st.markdown("##### Disease Classification & Diagnosis")
        st.caption("_SMA type classification, diagnosis method, and age at diagnosis_")

        sma_type = classification.get("sma_type", {})
        dx_method = classification.get("diagnosis_method", {})
        gc = classification.get("genetic_confirmation", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("SMA Participants Classified",
                      f"{sma_type.get('total_reported', 0):,}")
        with m2:
            pct = gc.get("confirmed_percentage", 0)
            st.metric("Genetically Confirmed", f"{pct:.1f}%")
        with m3:
            dx_age = classification.get("diagnosis_age", {})
            if dx_age.get("count"):
                st.metric("Median Diagnosis Age", f"{dx_age['median']:.0f} yr",
                          help=f"n={dx_age['count']}")

        # Charts: SMA type distribution + diagnosis method
        col_type, col_dx = st.columns(2)

        with col_type:
            dist = sma_type.get("distribution", {})
            if dist:
                items = [(k, v) for k, v in dist.items() if "Suppressed" not in k]
                items.sort(key=lambda x: x[1], reverse=True)
                labels = [_short_sma_type(k) for k, _ in reversed(items)]
                counts = [v for _, v in reversed(items)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h",
                    marker_color="#1E88E5",
                ))
                fig.update_layout(
                    title="SMA Type Distribution",
                    xaxis_title="Participants",
                    height=max(300, len(items) * 40 + 80),
                    margin=dict(t=40, b=40, l=120),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_dx:
            dx_dist = dx_method.get("distribution", {})
            if dx_dist:
                fig = go.Figure(go.Pie(
                    labels=list(dx_dist.keys()),
                    values=list(dx_dist.values()),
                    hole=0.4,
                ))
                fig.update_layout(
                    title=f"Diagnosis Method (n={dx_method.get('total_reported', 0)})",
                    height=350, margin=dict(t=40, b=20), showlegend=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Age at diagnosis histogram
        dx_age = classification.get("diagnosis_age", {})
        dx_hist = dx_age.get("histogram", {})
        if dx_hist:
            col_dx_hist, col_dx_table = st.columns([2, 1])
            with col_dx_hist:
                fig = go.Figure(go.Bar(
                    x=dx_hist["bins"], y=dx_hist["counts"],
                    marker_color="#00897B",
                ))
                fig.update_layout(
                    title=f"Age at Diagnosis (median {dx_age.get('median', 0)} yr, n={dx_age.get('count', 0)})",
                    xaxis_title="Age (years)",
                    yaxis_title="Participants",
                    xaxis_type="category",
                    showlegend=False, height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            with col_dx_table:
                rows = [{"Age Band": b, "N": c}
                        for b, c in zip(dx_hist["bins"], dx_hist["counts"])]
                static_table(pd.DataFrame(rows))

    # =================================================================
    # Genetics & Molecular Testing
    # =================================================================
    genetics = sma_snap.get("genetics", {})
    if genetics.get("available"):
        st.markdown("---")
        st.markdown("##### Genetics & Molecular Testing")
        st.caption("_SMN2 copy number, SMN1 status, and familial SMA_")

        smn2 = genetics.get("smn2_copy_number", {})
        fh = genetics.get("family_history", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("SMN2 Reported", f"{smn2.get('total_reported', 0):,}")
        with m2:
            fam_pct = fh.get("familial_percentage", 0)
            st.metric("Familial SMA", f"{fam_pct:.1f}%",
                      help=f"{fh.get('familial_count', 0)} participants")
        with m3:
            gc = classification.get("genetic_confirmation", {}) if classification.get("available") else {}
            st.metric("Genetic Confirmation", f"{gc.get('confirmed_percentage', 0):.1f}%")

        # Charts: SMN2 distribution + SMN2 by SMA type
        col_smn2, col_cross = st.columns(2)

        with col_smn2:
            dist = smn2.get("distribution", {})
            if dist:
                items = [(k, v) for k, v in dist.items() if "Suppressed" not in k]
                items.sort(key=lambda x: x[1], reverse=True)
                labels = [k for k, _ in reversed(items)]
                counts = [v for _, v in reversed(items)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h",
                    marker_color="#00897B",
                ))
                fig.update_layout(
                    title="SMN2 Copy Number Distribution",
                    xaxis_title="Participants",
                    height=max(250, len(items) * 35 + 80),
                    margin=dict(t=40, b=40, l=100),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_cross:
            smn2_by_type = genetics.get("smn2_by_sma_type", {})
            if smn2_by_type:
                # Build grouped bar of SMN2 copies by SMA type
                sma_types = [t for t in smn2_by_type.keys() if "Unknown" not in t]
                smn2_values = ["2", "3", "4"]
                colors = {"2": "#E53935", "3": "#1E88E5", "4": "#4CAF50"}

                fig = go.Figure()
                for smn2_val in smn2_values:
                    vals = [smn2_by_type.get(t, {}).get(smn2_val, 0) for t in sma_types]
                    fig.add_trace(go.Bar(
                        name=f"SMN2 = {smn2_val}",
                        x=[_short_sma_type(t) for t in sma_types],
                        y=vals,
                        marker_color=colors.get(smn2_val, "#999"),
                    ))
                fig.update_layout(
                    title="SMN2 Copy Number by SMA Type",
                    barmode="group",
                    xaxis_title="SMA Type",
                    yaxis_title="Participants",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Medications & Treatments — SMA Therapeutics
    # =================================================================
    tx = sma_snap.get("therapeutics", {})
    if tx.get("available"):
        st.markdown("---")
        st.markdown("##### Medications & Treatments")
        st.markdown(
            "Three disease-modifying therapies are available for SMA: "
            "**Spinraza** (nusinersen, intrathecal), **Evrysdi** (risdiplam, oral), "
            "and **Zolgensma** (onasemnogene, gene therapy)."
        )

        sma_drugs = tx.get("sma_drugs", {})
        total_on = tx.get("total_on_therapy", {})

        # Metric row
        drug_items = list(sma_drugs.items())
        cols = st.columns(min(len(drug_items) + 1, 4))
        with cols[0]:
            on_count = total_on.get("count", 0)
            on_pct = total_on.get("percentage", 0)
            st.metric("On Any SMA Therapy", f"{on_count:,}",
                      delta=f"{on_pct:.1f}% of cohort")
        for i, (drug_name, drug_data) in enumerate(drug_items[:3]):
            with cols[i + 1]:
                if drug_data.get("suppressed"):
                    st.metric(drug_name.split(" (")[0], "<11 participants")
                else:
                    st.metric(
                        drug_name.split(" (")[0],
                        f"{drug_data.get('percentage', 0):.1f}%",
                        delta=f"{drug_data.get('count', 0)} participants",
                    )

        # Charts: drug utilization + therapy by SMA type
        col_drugs, col_by_type = st.columns(2)

        with col_drugs:
            drug_labels = []
            drug_counts = []
            for name, data in sma_drugs.items():
                if not data.get("suppressed") and data.get("count", 0) > 0:
                    drug_labels.append(name)
                    drug_counts.append(data["count"])
            if drug_labels:
                fig = go.Figure(go.Bar(
                    y=list(reversed(drug_labels)),
                    x=list(reversed(drug_counts)),
                    orientation="h",
                    marker_color="#1E88E5",
                ))
                fig.update_layout(
                    title="SMA Drug Utilization (Unique Participants)",
                    xaxis_title="Participants",
                    height=max(250, len(drug_labels) * 50 + 80),
                    margin=dict(t=40, b=40, l=220),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_by_type:
            therapy_by_type = tx.get("therapy_by_sma_type", {})
            if therapy_by_type:
                sma_types = [t for t in therapy_by_type.keys()
                             if "Unknown" not in t and therapy_by_type[t].get("total_patients", 0) >= 11]
                drug_names = ["Spinraza (nusinersen)", "Evrysdi (risdiplam)", "Zolgensma (onasemnogene)"]
                drug_colors = {"Spinraza (nusinersen)": "#1E88E5",
                               "Evrysdi (risdiplam)": "#4CAF50",
                               "Zolgensma (onasemnogene)": "#FF7043"}

                fig = go.Figure()
                for drug in drug_names:
                    vals = [therapy_by_type.get(t, {}).get(drug, 0) for t in sma_types]
                    fig.add_trace(go.Bar(
                        name=drug.split(" (")[0],
                        x=[_short_sma_type(t) for t in sma_types],
                        y=vals,
                        marker_color=drug_colors.get(drug, "#999"),
                    ))
                fig.update_layout(
                    title="Therapy by SMA Type",
                    barmode="group",
                    xaxis_title="SMA Type",
                    yaxis_title="Participants",
                    height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

        # Top drugs table
        top_drugs = tx.get("top_drugs", [])
        if top_drugs:
            with st.expander(f"Top Medications (all drugs, {len(top_drugs)} shown)"):
                rows = [{"Drug": d["drug"],
                         "Participants": d["patients"],
                         "% of Cohort": f"{d.get('percentage', 0):.1f}%"}
                        for d in top_drugs]
                static_table(pd.DataFrame(rows))

    # =================================================================
    # Pulmonary & Respiratory
    # =================================================================
    resp = sma_snap.get("respiratory", {})
    if resp.get("available"):
        st.markdown("---")
        st.markdown("##### Pulmonary & Respiratory")
        st.caption("_FVC % predicted & longitudinal trends_")

        fvc = resp.get("fvc_pct", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            if fvc.get("count"):
                st.metric("FVC % Predicted (median)", f"{fvc['median']:.0f}%",
                          help=f"n={fvc['count']} participants")
        with m2:
            if fvc.get("count"):
                st.metric("FVC Mean", f"{fvc['mean']:.1f}%")
        with m3:
            amb = sma_snap.get("milestones", {}).get("ambulatory_status", {})
            amb_dist = amb.get("distribution", {})
            not_amb = amb_dist.get("Not ambulatory", 0)
            total_amb = amb.get("total_reported", 1)
            if amb_dist:
                st.metric("Not Ambulatory", f"{not_amb / total_amb * 100:.0f}%",
                          help=f"{not_amb} of {total_amb} reporting")

        # Charts: FVC histogram + longitudinal
        col_fvc_hist, col_fvc_long = st.columns(2)

        with col_fvc_hist:
            hist = fvc.get("histogram", {})
            if hist:
                fvc_colors = []
                for b in hist["bins"]:
                    parts = b.split("-")
                    if len(parts) == 2:
                        mid = (float(parts[0]) + float(parts[1])) / 2
                        if mid < 40:
                            fvc_colors.append("#E53935")
                        elif mid < 60:
                            fvc_colors.append("#FF9800")
                        elif mid < 80:
                            fvc_colors.append("#FFC107")
                        else:
                            fvc_colors.append("#4CAF50")
                    else:
                        fvc_colors.append("#1E88E5")
                fig = go.Figure(go.Bar(
                    x=hist["bins"], y=hist["counts"], marker_color=fvc_colors,
                ))
                fig.update_layout(
                    title=f"FVC % Predicted (n={fvc['count']})",
                    xaxis_title="FVC %", yaxis_title="Participants",
                    showlegend=False, height=350,
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Red <40% | Orange 40-60% | Yellow 60-80% | Green >80%")

        with col_fvc_long:
            longitudinal = fvc.get("longitudinal", [])
            if longitudinal:
                years = [str(p["year"]) for p in longitudinal]
                medians = [p["median"] for p in longitudinal]
                q1s = [p["q1"] for p in longitudinal]
                q3s = [p["q3"] for p in longitudinal]
                ns = [p["n"] for p in longitudinal]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=years + years[::-1],
                    y=q3s + q1s[::-1],
                    fill="toself", fillcolor="rgba(30,136,229,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="IQR (Q1-Q3)",
                ))
                fig.add_trace(go.Scatter(
                    x=years, y=medians,
                    mode="lines+markers", line=dict(color="#1E88E5", width=3),
                    name="Median",
                    text=[f"n={n}" for n in ns], textposition="top center",
                ))
                fig.update_layout(
                    title="FVC % Predicted Over Time",
                    xaxis_title="Years Since Enrollment",
                    yaxis_title="FVC % Predicted",
                    height=350, yaxis=dict(range=[0, 120]),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Demographics & Enrollment (+ Multidisciplinary Care)
    # =================================================================
    state_dist = sma_snap.get("state_distribution", {})
    fac_list = sma_snap.get("facilities", {}).get("facilities", [])

    if state_dist.get("available") or fac_list:
        st.markdown("---")
        st.markdown("##### Demographics & Enrollment")
        st.caption("_Also includes: Multidisciplinary Care_")

        col_state, col_sites = st.columns(2)

        with col_state:
            states = state_dist.get("states", [])
            if states:
                labels = [s["state"] for s in reversed(states[:15])]
                counts = [s["total"] for s in reversed(states[:15])]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#00BCD4",
                ))
                fig.update_layout(
                    title=f"Participant Distribution ({state_dist.get('total_states_mapped', 0)} states)",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=180),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("States with <11 participants grouped as 'Other States'")

        with col_sites:
            if fac_list and has_access():
                top_sites = fac_list[:10]
                labels = [s.get("name", f"Site {s.get('id', '')}") for s in reversed(top_sites)]
                counts = [s.get("patients", 0) for s in reversed(top_sites)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#FF7043",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Participants",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("**Top 10 Care Sites**")
                st.info("Not available — requires DUA.  "
                        "[Request Access](https://mdausa.tfaforms.net/389761)")

    _render_enrollment_charts("SMA")

    # =================================================================
    # Other Clinical Domains
    # =================================================================
    _sma_covered = {
        "Motor Function Assessments",
        "Disease Classification & Diagnosis",
        "Genetics & Molecular Testing",
        "Medications & Treatments",
        "Pulmonary & Respiratory",
        "Demographics & Enrollment",
        "Multidisciplinary Care",
        "Mobility & Ambulation",
    }
    uncovered = [d for d in _ALL_DOMAINS if d not in _sma_covered]
    with st.expander(f"Other Clinical Domains ({len(uncovered)} not yet available)"):
        for d in uncovered:
            st.markdown(f"- **{d}**")
        st.caption("Additional domains will be added as analytics are developed.")
