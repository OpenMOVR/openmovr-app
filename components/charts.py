"""
Chart Components

Reusable chart components using Plotly for interactive visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def create_disease_distribution_chart(disease_data: List[Dict[str, Any]],
                                      chart_type: str = 'bar') -> go.Figure:
    """
    Create disease distribution chart.

    Args:
        disease_data: List of dicts with disease, patient_count, percentage
        chart_type: 'bar', 'pie', or 'horizontal_bar'

    Returns:
        Plotly figure
    """
    df = pd.DataFrame(disease_data)

    if chart_type == 'pie':
        fig = px.pie(
            df,
            values='patient_count',
            names='disease',
            title='Disease Distribution',
            hover_data=['percentage']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')

    elif chart_type == 'horizontal_bar':
        fig = px.bar(
            df,
            x='patient_count',
            y='disease',
            orientation='h',
            title='Disease Distribution',
            text='patient_count',
            color='patient_count',
            color_continuous_scale='Blues'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})

    else:  # bar (default)
        fig = px.bar(
            df,
            x='disease',
            y='patient_count',
            title='Disease Distribution',
            text='patient_count',
            color='patient_count',
            color_continuous_scale='Blues'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False)

    fig.update_layout(
        xaxis_title="Disease" if chart_type != 'horizontal_bar' else "Patient Count",
        yaxis_title="Patient Count" if chart_type != 'horizontal_bar' else "Disease",
        height=400
    )

    return fig


def create_facility_chart(facility_data: List[Dict[str, Any]],
                         top_n: int = 10,
                         chart_type: str = 'bar') -> go.Figure:
    """
    Create facility distribution chart.

    Args:
        facility_data: List of facility dicts
        top_n: Number of top facilities to show
        chart_type: 'bar' or 'horizontal_bar'

    Returns:
        Plotly figure
    """
    df = pd.DataFrame(facility_data[:top_n])

    if chart_type == 'horizontal_bar':
        fig = px.bar(
            df,
            x='patient_count',
            y='FACILITY_NAME',
            orientation='h',
            title=f'Top {top_n} Facilities by Patient Count',
            text='patient_count',
            color='patient_count',
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(
            showlegend=False,
            yaxis={'categoryorder':'total ascending', 'type': 'category'},
            height=max(400, top_n * 40)
        )
    else:  # bar
        fig = px.bar(
            df,
            x='FACILITY_NAME',
            y='patient_count',
            title=f'Top {top_n} Facilities by Patient Count',
            text='patient_count',
            color='patient_count',
            color_continuous_scale='Viridis'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, xaxis={'tickangle': -45})

    return fig


def create_data_availability_chart(data_availability: Dict[str, int]) -> go.Figure:
    """
    Create data availability chart.

    Args:
        data_availability: Dict of data_type -> record_count

    Returns:
        Plotly figure
    """
    df = pd.DataFrame([
        {'Data Type': key.replace('_records', '').title(), 'Record Count': value}
        for key, value in data_availability.items()
    ])

    fig = px.bar(
        df,
        x='Data Type',
        y='Record Count',
        title='Data Availability by Table',
        text='Record Count',
        color='Record Count',
        color_continuous_scale='Greens'
    )
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    fig.update_layout(showlegend=False, height=400)

    return fig


def create_metric_card(title: str, value: Any, delta: Optional[str] = None) -> Dict[str, Any]:
    """
    Create data for a metric card (used with st.metric).

    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta/change indicator

    Returns:
        Dict with metric data
    """
    return {
        'title': title,
        'value': value,
        'delta': delta
    }


# ---------------------------------------------------------------------------
# Disease Explorer — Demographics & Diagnosis Charts
# ---------------------------------------------------------------------------

def create_age_distribution_chart(
    df: pd.DataFrame,
    dob_field: str = "dob",
    title: str = "Age Distribution",
    color: str = "#636EFA",
) -> Optional[go.Figure]:
    """
    Create an age distribution histogram from a DOB column.

    Computes age from *dob_field*, bins into ~20 bins, and adds a
    dashed vertical line at the median age.

    Returns None if there is insufficient data.
    """
    if dob_field not in df.columns:
        return None

    dob = pd.to_datetime(df[dob_field], errors="coerce")
    age = ((pd.Timestamp.now() - dob).dt.days / 365.25).dropna()

    if len(age) < 2:
        return None

    median_age = age.median()
    mean_age = age.mean()

    fig = px.histogram(
        age,
        nbins=20,
        title=title,
        color_discrete_sequence=[color],
        labels={"value": "Age (years)", "count": "Patients"},
    )

    fig.add_vline(
        x=median_age,
        line_dash="dash",
        line_color="#333",
        annotation_text=f"Median: {median_age:.0f}",
        annotation_position="top right",
    )

    fig.update_layout(
        height=400,
        xaxis_title="Age (years)",
        yaxis_title="Number of Patients",
        showlegend=False,
        annotations=[
            dict(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"n={len(age):,}  |  Mean: {mean_age:.1f}  |  Median: {median_age:.1f}",
                showarrow=False, font=dict(size=11, color="#555"),
                xanchor="right",
            )
        ],
    )
    return fig


def create_categorical_bar_chart(
    series: pd.Series,
    title: str,
    color_scale: str = "Blues",
    max_categories: int = 15,
) -> Optional[go.Figure]:
    """
    Horizontal bar chart for a categorical column.

    Sorted descending by count, capped at *max_categories*.
    Returns None if no valid data.
    """
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]

    if clean.empty:
        return None

    counts = clean.value_counts()
    total = len(counts)
    counts = counts.head(max_categories)

    chart_df = pd.DataFrame({"Category": counts.index, "Count": counts.values})

    subtitle = f"  (top {max_categories} of {total})" if total > max_categories else ""

    fig = px.bar(
        chart_df,
        x="Count",
        y="Category",
        orientation="h",
        title=f"{title}{subtitle}",
        text="Count",
        color="Count",
        color_continuous_scale=color_scale,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        height=max(350, len(counts) * 35),
        showlegend=False,
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Patients",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return fig


def create_categorical_donut_chart(
    series: pd.Series,
    title: str,
    max_slices: int = 8,
) -> Optional[go.Figure]:
    """
    Donut chart (pie with hole) for low-cardinality categorical data.

    Best for fields with 2-6 distinct values (e.g. gender).
    Returns None if no valid data.
    """
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]

    if clean.empty:
        return None

    counts = clean.value_counts().head(max_slices)
    chart_df = pd.DataFrame({"Category": counts.index, "Count": counts.values})

    fig = px.pie(
        chart_df,
        values="Count",
        names="Category",
        title=title,
        hole=0.4,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=400, showlegend=True)
    return fig


def create_numeric_histogram_chart(
    series: pd.Series,
    title: str,
    nbins: int = 20,
    color: str = "#00CC96",
) -> Optional[go.Figure]:
    """
    Histogram for a numeric column (diagnosis age, onset age, etc.).

    Adds a dashed vertical median line and summary stats annotation.
    Returns None if insufficient data.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if len(numeric) < 2:
        return None

    median_val = numeric.median()
    mean_val = numeric.mean()

    fig = px.histogram(
        numeric,
        nbins=nbins,
        title=title,
        color_discrete_sequence=[color],
        labels={"value": title.replace(" Distribution", ""), "count": "Patients"},
    )

    fig.add_vline(
        x=median_val,
        line_dash="dash",
        line_color="#333",
        annotation_text=f"Median: {median_val:.1f}",
        annotation_position="top right",
    )

    fig.update_layout(
        height=400,
        xaxis_title=title.replace(" Distribution", ""),
        yaxis_title="Number of Patients",
        showlegend=False,
        annotations=[
            dict(
                x=0.98, y=0.95, xref="paper", yref="paper",
                text=f"n={len(numeric):,}  |  Mean: {mean_val:.1f}  |  Median: {median_val:.1f}",
                showarrow=False, font=dict(size=11, color="#555"),
                xanchor="right",
            )
        ],
    )
    return fig


def create_facility_distribution_mini_chart(
    df: pd.DataFrame,
    title: str = "Top Sites (This Cohort)",
    top_n: int = 10,
    facility_col: str = "FACILITY_DISPLAY_ID",
    location_lookup: Optional[Dict[str, str]] = None,
) -> Optional[go.Figure]:
    """
    Horizontal bar of top *top_n* sites within a cohort DataFrame.

    Uses *location_lookup* (facility_id → "City, ST") when provided.
    Falls back to facility_col as string so Plotly treats it as categorical.
    """
    if facility_col not in df.columns:
        return None

    counts = df[facility_col].dropna().value_counts().head(top_n)

    if counts.empty:
        return None

    # Build display labels: prefer location lookup, fall back to ID as string
    if location_lookup:
        labels = [location_lookup.get(str(fid), str(fid)) for fid in counts.index]
    else:
        labels = [str(fid) for fid in counts.index]

    chart_df = pd.DataFrame({"Site": labels, "Patients": counts.values})

    fig = px.bar(
        chart_df,
        x="Patients",
        y="Site",
        orientation="h",
        title=title,
        text="Patients",
        color="Patients",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        height=max(350, top_n * 35),
        showlegend=False,
        yaxis=dict(categoryorder="total ascending", type="category"),
        xaxis_title="Patients",
        yaxis_title="",
        coloraxis_showscale=False,
    )
    return fig


# ---------------------------------------------------------------------------
# US Site Map
# ---------------------------------------------------------------------------

def _prepare_site_df(
    site_locations: List[Dict[str, Any]],
    disease_filter: Optional[str] = None,
    continental_only: bool = True,
    top_n: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """Shared helper: filter site_locations → DataFrame ready for map/table.

    When *disease_filter* is set (e.g. "ALS"), patient_count is replaced
    with the per-disease count from ``by_disease``.  Sites with 0 patients
    for that disease are dropped.
    """
    if not site_locations:
        return None

    df = pd.DataFrame(site_locations)

    # Apply disease filter — swap total for per-disease count
    if disease_filter and "by_disease" in df.columns:
        df["patient_count"] = df["by_disease"].apply(
            lambda d: d.get(disease_filter, 0) if isinstance(d, dict) else 0
        )
        df = df[df["patient_count"] > 0]

    if continental_only:
        df = df[df["continental"] == True]

    df = df.dropna(subset=["lat", "lon"])

    if df.empty:
        return None

    # Sort descending and apply top_n
    df = df.sort_values("patient_count", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)

    return df


def create_site_map(
    site_locations: List[Dict[str, Any]],
    continental_only: bool = True,
    disease_filter: Optional[str] = None,
    top_n: Optional[int] = None,
    title: str = "MOVR Participating Sites",
) -> Optional[go.Figure]:
    """
    Scatter-geo map of MOVR sites on a US continental projection.

    Args:
        site_locations: list of site dicts from the snapshot
        continental_only: exclude non-continental US territories
        disease_filter: if set, use per-disease count (e.g. "ALS")
        top_n: if set, show only the top N sites by patient count
        title: chart title

    Returns None if no valid locations.
    """
    df = _prepare_site_df(site_locations, disease_filter, continental_only, top_n)
    if df is None:
        return None

    # Build hover text
    def _hover(r):
        lines = [
            f"<b>{r['city']}, {r['state']}</b>",
            f"Patients: {r['patient_count']:,}",
        ]
        if r.get("region"):
            lines.append(f"Region: {r['region']}")
        if r.get("site_type"):
            lines.append(f"Type: {r['site_type']}")
        # Show disease breakdown if available and no single-disease filter
        by_ds = r.get("by_disease")
        if isinstance(by_ds, dict) and by_ds:
            ds_str = " | ".join(f"{k}: {v}" for k, v in sorted(by_ds.items()))
            lines.append(f"Diseases: {ds_str}")
        return "<br>".join(lines)

    df["hover"] = df.apply(_hover, axis=1)

    # Size: scale so smallest sites are still visible
    min_size = 8
    max_size = 35
    counts = df["patient_count"].values.astype(float)
    if counts.max() > counts.min():
        scaled = (counts - counts.min()) / (counts.max() - counts.min())
        df["marker_size"] = min_size + scaled * (max_size - min_size)
    else:
        df["marker_size"] = (min_size + max_size) / 2

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lat=df["lat"],
        lon=df["lon"],
        text=df["hover"],
        hoverinfo="text",
        marker=dict(
            size=df["marker_size"],
            color=df["patient_count"],
            colorscale="Blues",
            cmin=0,
            cmax=df["patient_count"].max(),
            colorbar=dict(title="Patients"),
            line=dict(width=0.5, color="#333"),
            opacity=0.85,
        ),
    ))

    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#f0f0f0",
        showlakes=True,
        lakecolor="white",
        showcountries=False,
        showsubunits=True,
        subunitcolor="#ccc",
    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
        geo=dict(bgcolor="white"),
    )

    return fig
