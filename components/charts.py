"""
Chart Components

Reusable chart components using Plotly for interactive visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
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
    title: str = "Top Facilities (This Cohort)",
    top_n: int = 10,
    facility_col: str = "FACILITY_DISPLAY_ID",
) -> Optional[go.Figure]:
    """
    Horizontal bar of top *top_n* facilities within a cohort DataFrame.

    Prefers FACILITY_NAME for display. Falls back to facility_col if names
    are unavailable, casting IDs to strings so Plotly treats them as
    categorical rather than numeric.
    """
    if facility_col not in df.columns:
        return None

    counts = df[facility_col].dropna().value_counts().head(top_n)

    if counts.empty:
        return None

    # Build display labels: prefer FACILITY_NAME, fall back to ID as string
    if "FACILITY_NAME" in df.columns:
        id_to_name = (
            df[[facility_col, "FACILITY_NAME"]]
            .drop_duplicates()
            .set_index(facility_col)["FACILITY_NAME"]
            .to_dict()
        )
        labels = [id_to_name.get(fid, str(fid)) for fid in counts.index]
    else:
        labels = [str(fid) for fid in counts.index]

    chart_df = pd.DataFrame({"Facility": labels, "Patients": counts.values})

    fig = px.bar(
        chart_df,
        x="Patients",
        y="Facility",
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
