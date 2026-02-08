"""
Shared deep-dive renderers for DMD and LGMD.

These functions are used by both the Disease Explorer (embedded tab) and
the standalone DUA-gated deep-dive pages.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from api.dmd import DMDAPI
from api.lgmd import LGMDAPI
from components.tables import static_table


# ======================================================================
# LGMD Deep Dive
# ======================================================================

def render_lgmd_deep_dive():
    """Render LGMD-specific deep-dive sections from the LGMD snapshot."""
    lgmd_snap = LGMDAPI.get_snapshot()
    if not lgmd_snap:
        return

    summary = lgmd_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("LGMD Deep-Dive")

    # =================================================================
    # Section 1: Subtype Distribution (hero)
    # =================================================================
    subtypes = lgmd_snap.get("subtypes", {})
    dist = subtypes.get("distribution", [])
    diagnosis = lgmd_snap.get("diagnosis", {})
    gc = diagnosis.get("genetic_confirmation", {})

    if dist:
        st.markdown("##### Subtype Distribution")

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total LGMD Patients", f"{total_patients:,}")
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
                xaxis_title="Patients",
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
    # Section 2: Diagnostic Journey
    # =================================================================
    journey = lgmd_snap.get("diagnostic_journey", {})
    if journey.get("available"):
        st.markdown("---")
        st.markdown("##### Diagnostic Journey")
        st.markdown(
            "LGMD patients often experience a significant delay between symptom onset "
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
                    yaxis_title="Patients",
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
                    xaxis_title="Age (years)", yaxis_title="Patients",
                    barmode="overlay", height=350,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Section 3: Functional Outcomes
    # =================================================================
    func = lgmd_snap.get("functional_scores", {})
    if func.get("available"):
        st.markdown("---")
        st.markdown("##### Functional Outcomes")

        fvc = func.get("fvc_pct", {})
        tw = func.get("timed_10m_walk", {})
        amb_func = func.get("ambulatory_status", {})

        # Metric row
        m1, m2, m3 = st.columns(3)
        with m1:
            if fvc.get("count"):
                st.metric("FVC % Predicted (median)", f"{fvc['median']:.0f}%",
                          help=f"n={fvc['count']} patients")
        with m2:
            if tw.get("count"):
                st.metric("Timed 10m Walk (median)", f"{tw['median']:.1f}s",
                          help=f"n={tw['count']} patients")
        with m3:
            amb_dist = amb_func.get("distribution", {})
            if amb_dist:
                not_amb = amb_dist.get("Not ambulatory", 0)
                total_amb = amb_func.get("total_reported", 1)
                st.metric("Not Ambulatory", f"{not_amb / total_amb * 100:.0f}%",
                          help=f"{not_amb} of {total_amb} patients reporting")

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
                    xaxis_title="FVC %", yaxis_title="Patients",
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
                    xaxis_title="Subtype", yaxis_title="Patients",
                    barmode="group", height=350,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    margin=dict(b=80),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Section 4: Medication Utilization
    # =================================================================
    meds = lgmd_snap.get("medications", {})
    if meds.get("available"):
        st.markdown("---")
        st.markdown("##### Medication Utilization")
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
                    xaxis_title="Patients",
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
                    title="Top Medications (by unique patients)",
                    xaxis_title="Patients",
                    height=350,
                    margin=dict(t=40, b=40, l=200),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Section 5: Clinical Characteristics
    # =================================================================
    clinical = lgmd_snap.get("clinical", {})
    if clinical.get("available"):
        st.markdown("---")
        st.markdown("##### Clinical Characteristics")

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
                                  xaxis_title="Patients",
                                  height=max(250, len(labels) * 28 + 80),
                                  margin=dict(t=40, b=40, l=200))
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Section 6: State Distribution + Care Sites
    # =================================================================
    state_dist = lgmd_snap.get("state_distribution", {})
    fac_list = lgmd_snap.get("facilities", {}).get("facilities", [])

    if state_dist.get("available") or fac_list:
        st.markdown("---")
        st.markdown("##### Geographic Distribution & Care Sites")

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
                    title=f"Patient Distribution ({state_dist.get('total_states_mapped', 0)} states)",
                    xaxis_title="Patients",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=180),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("States with <11 patients grouped as 'Other States'")

        with col_sites:
            if fac_list:
                top_sites = fac_list[:10]
                labels = [s.get("name", f"Site {s.get('id', '')}") for s in reversed(top_sites)]
                counts = [s.get("patients", 0) for s in reversed(top_sites)]
                fig = go.Figure(go.Bar(
                    y=labels, x=counts, orientation="h", marker_color="#FF7043",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Patients",
                    height=max(350, len(labels) * 28 + 80),
                    margin=dict(t=40, b=40, l=250),
                )
                st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# DMD Deep Dive
# ======================================================================

def render_dmd_deep_dive():
    """Render DMD-specific deep-dive sections from the DMD snapshot."""
    dmd_snap = DMDAPI.get_snapshot()
    if not dmd_snap:
        return

    summary = dmd_snap.get("summary", {})
    total_patients = summary.get("total_patients", 0)

    st.subheader("DMD Deep-Dive")

    # =================================================================
    # Section 1: Exon-Skipping Therapeutics (hero section)
    # =================================================================
    tx = dmd_snap.get("therapeutics", {})
    if tx.get("available"):
        st.markdown("##### Exon-Skipping Therapeutics: Amenability & Utilization")
        st.markdown(
            "Exon-skipping therapies target specific deletion mutations in the dystrophin gene. "
            "Patients are classified as *amenable* based on mutation type and exon range."
        )

        # Metric row
        total_amenable = tx.get("total_amenable_any_drug", {})
        total_on = tx.get("total_on_any_exon_skipping", {})
        amen_count = total_amenable.get("count", 0)
        on_count = total_on.get("count", 0)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total DMD Patients", f"{total_patients:,}")
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
                    xaxis_title="Patients",
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
    # Section 2: Glucocorticoid Use
    # =================================================================
    steroids = dmd_snap.get("steroids", {})
    if steroids.get("available"):
        st.markdown("##### Glucocorticoid Use in DMD")

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
                    title="Steroid Drug Class (Unique Patients)",
                    xaxis_title="Patients",
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
    # Section 2.5: Functional Outcomes
    # =================================================================
    func = dmd_snap.get("functional_scores", {})
    if func.get("available"):
        st.markdown("##### Functional Outcomes")

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
                    delta=f"n = {fvc_data['count']} patients",
                )
        with fm2:
            if tw_data.get("count"):
                st.metric(
                    "Timed 10m Walk (median)",
                    f"{tw_data['median']}s",
                    delta=f"n = {tw_data['count']} patients",
                )
        with fm3:
            if loa_data:
                age_loss = loa_data.get("age_at_loss_years", {})
                st.metric(
                    "Age at Loss of Ambulation (median)",
                    f"{age_loss.get('median', 'N/A')} years",
                    delta=f"n = {loa_data.get('total_with_data', 0)} patients",
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
                    xaxis_title="FVC %", yaxis_title="Patients",
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
                    xaxis_title="Seconds", yaxis_title="Patients",
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
                    xaxis_title="Age (years)", yaxis_title="Patients",
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
            "*Functional outcomes reflect latest available encounter per patient. "
            "Longitudinal trends show median with interquartile range (IQR) by years since enrollment.*"
        )

    # =================================================================
    # Section 3: Genetic & Mutation Profile
    # =================================================================
    genetics = dmd_snap.get("genetics", {})
    if genetics.get("available"):
        st.markdown("##### Genetic & Mutation Profile")

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
                    title="Mutation Type", xaxis_title="Patients",
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
                    title="Frame Type", xaxis_title="Patients",
                    height=300, margin=dict(t=40, b=40, l=150),
                )
                st.plotly_chart(fig, use_container_width=True)

    # =================================================================
    # Section 4: State-Level Distribution
    # =================================================================
    state_dist = dmd_snap.get("state_distribution", {})
    if state_dist.get("available"):
        states = state_dist.get("states", [])
        if states:
            st.markdown("##### Geographic Distribution: Therapy Utilization by State")

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
                    name="Other DMD Patients", marker_color="#BDBDBD",
                ))
                fig.update_layout(
                    barmode="stack",
                    title="Top States by DMD Patient Count",
                    xaxis_title="Patients",
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
                    "*States with fewer than 11 DMD patients are grouped per "
                    "HIPAA small-cell suppression guidelines.*"
                )

    # =================================================================
    # Section 5: Care Sites & Ambulatory Status
    # =================================================================
    amb = dmd_snap.get("ambulatory", {})
    amb_dist = amb.get("current_status", {}).get("distribution", {})
    fac_list = dmd_snap.get("facilities", {}).get("facilities", [])

    if amb_dist or fac_list:
        st.markdown("##### Care Sites & Ambulatory Status")
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
            if fac_list:
                top_sites = fac_list[:10]
                labels = [f"Site {s.get('id', i + 1)}" for i, s in enumerate(top_sites)]
                counts = [s.get("patients", 0) for s in top_sites]
                fig = go.Figure(go.Bar(
                    x=counts, y=list(reversed(labels)), orientation="h",
                    marker_color="#00BCD4",
                ))
                fig.update_layout(
                    title="Top 10 Care Sites",
                    xaxis_title="Patients", height=350,
                    margin=dict(t=40, b=40, l=120),
                )
                st.plotly_chart(fig, use_container_width=True)
