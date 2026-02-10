#!/usr/bin/env python3
"""
Generate ALS Clinical Summary Snapshot

Creates a JSON file with pre-computed ALS statistics including
ALSFRS-R scores, El Escorial criteria, respiratory function,
medication utilization, and disease milestones.

Usage:
    python scripts/generate_als_snapshot.py
    python scripts/generate_als_snapshot.py --output stats/als_snapshot.json
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import numpy as np

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.cohorts import get_disease_cohort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HIPAA Safe Harbor small-cell threshold
SMALL_CELL = 11

# ALSFRS-R severity bands (0-48 scale, higher = better)
SEVERITY_BANDS = [
    ("Very Severe (0-11)", 0, 12),
    ("Severe (12-23)", 12, 24),
    ("Moderate (24-35)", 24, 36),
    ("Mild (36-48)", 36, 49),
]

# ALS-appropriate age bands (adult-onset)
_AGE_BANDS = [
    ("<40", 0, 40),
    ("40-49", 40, 50),
    ("50-59", 50, 60),
    ("60-69", 60, 70),
    ("70+", 70, 200),
]

# ALS disease-modifying / symptomatic drugs to track
ALS_DRUGS = {
    "Riluzole": ["riluzole", "rilutek", "tiglutik", "exservan"],
    "Radicava (edaravone)": ["radicava", "edaravone"],
    "Nuedexta": ["nuedexta", "dextromethorphan/quinidine"],
    "Relyvrio": ["relyvrio", "amx0035", "sodium phenylbutyrate"],
    "Qalsody (tofersen)": ["qalsody", "tofersen"],
}


# ---------------------------------------------------------------------------
# HIPAA suppression helpers
# ---------------------------------------------------------------------------

def _suppress(count: int) -> dict:
    """Return count with suppression flag if below HIPAA threshold."""
    if count < SMALL_CELL:
        return {"count": 0, "suppressed": True}
    return {"count": int(count), "suppressed": False}


def _suppress_distribution(dist: dict, min_cell: int = SMALL_CELL) -> dict:
    """Suppress small categories in a distribution dict."""
    suppressed_total = 0
    cleaned = {}
    for label, count in dist.items():
        if count < min_cell:
            suppressed_total += count
        else:
            cleaned[label] = int(count)
    if suppressed_total > 0:
        cleaned["Suppressed (n<11)"] = int(suppressed_total)
    return cleaned


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _score_stats(series: pd.Series) -> dict:
    """Compute basic stats for a numeric series."""
    series = series.dropna()
    if series.empty:
        return {"count": 0}
    return {
        "count": int(len(series)),
        "median": round(float(series.median()), 1),
        "mean": round(float(series.mean()), 1),
        "min": round(float(series.min()), 1),
        "max": round(float(series.max()), 1),
    }


def _score_histogram(series: pd.Series, bins: list) -> dict:
    """Build a histogram dict from a numeric series and bin edges."""
    series = series.dropna()
    if series.empty:
        return {}
    hist, edges = np.histogram(series, bins=bins)
    return {
        "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
        "counts": [int(c) for c in hist],
    }


def _latest_per_patient(enc_df: pd.DataFrame, patient_ids: list) -> pd.DataFrame:
    """Get the latest encounter per patient, sorted by encntdt."""
    df = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
    if "encntdt" not in df.columns or df.empty:
        return pd.DataFrame()
    df["encntdt"] = pd.to_datetime(df["encntdt"], errors="coerce")
    df = df.dropna(subset=["encntdt"])
    return df.sort_values("encntdt").groupby("FACPATID").last().reset_index()


# ---------------------------------------------------------------------------
# ALSFRS-R computation
# ---------------------------------------------------------------------------

def _compute_alsfrs(enc_df: pd.DataFrame, demo_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute ALSFRS-R total score stats, sub-scores, severity bands, and longitudinal."""
    stats = {"available": False}

    if "alsfrstl" not in enc_df.columns:
        return stats

    als_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()

    # --- Total score from latest encounter ---
    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return stats

    total_scores = pd.to_numeric(latest["alsfrstl"], errors="coerce").dropna()
    total_scores = total_scores[(total_scores >= 0) & (total_scores <= 48)]

    if len(total_scores) < SMALL_CELL:
        return stats

    stats["available"] = True

    # Basic stats + histogram
    bins = list(range(0, 52, 4))  # 0-4, 4-8, ..., 44-48
    stats["total_score"] = {
        **_score_stats(total_scores),
        "histogram": _score_histogram(total_scores, bins),
    }

    # Severity bands
    severity = []
    for label, lo, hi in SEVERITY_BANDS:
        band_count = int(((total_scores >= lo) & (total_scores < hi)).sum())
        severity.append({
            "label": label,
            **_suppress(band_count),
            "percentage": round(band_count / len(total_scores) * 100, 1) if total_scores.any() else 0,
        })
    stats["severity_bands"] = severity

    # By age band
    enrol_ages = {}
    if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
        dob = pd.to_datetime(demo_df.set_index("FACPATID")["dob"], errors="coerce")
        enrol = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
        age = ((enrol - dob).dt.days / 365.25).dropna()
        enrol_ages = age.to_dict()

    age_band_data = {}
    for label, lo, hi in _AGE_BANDS:
        mask = latest["FACPATID"].map(
            lambda pid: lo <= enrol_ages.get(pid, -1) < hi
        )
        subset = pd.to_numeric(
            latest.loc[mask, "alsfrstl"], errors="coerce"
        ).dropna()
        subset = subset[(subset >= 0) & (subset <= 48)]
        if len(subset) >= SMALL_CELL:
            age_band_data[label] = _score_stats(subset)
        elif len(subset) > 0:
            age_band_data[label] = {"count": 0, "suppressed": True}
    stats["total_score"]["by_age_band"] = age_band_data

    # --- Longitudinal ALSFRS-R ---
    als_enc["encntdt"] = pd.to_datetime(als_enc["encntdt"], errors="coerce")
    als_enc["alsfrstl_num"] = pd.to_numeric(als_enc["alsfrstl"], errors="coerce")
    enrol_dt = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
    als_enc["enroldt"] = als_enc["FACPATID"].map(enrol_dt)
    als_enc["years_since_enrol"] = (
        (als_enc["encntdt"] - als_enc["enroldt"]).dt.days / 365.25
    )
    alsfrs_long = als_enc[["alsfrstl_num", "years_since_enrol"]].dropna()
    alsfrs_long = alsfrs_long[
        (alsfrs_long["alsfrstl_num"] >= 0) & (alsfrs_long["alsfrstl_num"] <= 48)
    ]
    alsfrs_long["year_bucket"] = (
        alsfrs_long["years_since_enrol"].clip(lower=0).astype(int).clip(upper=6)
    )

    longitudinal = []
    for yr in range(7):
        bucket = alsfrs_long[alsfrs_long["year_bucket"] == yr]["alsfrstl_num"]
        if len(bucket) >= SMALL_CELL:
            longitudinal.append({
                "year": yr if yr < 6 else "6+",
                "median": round(float(bucket.median()), 1),
                "q1": round(float(bucket.quantile(0.25)), 1),
                "q3": round(float(bucket.quantile(0.75)), 1),
                "n": int(len(bucket)),
            })
    stats["total_score"]["longitudinal"] = longitudinal

    # Count patients with multiple measurements
    multi = als_enc[als_enc["alsfrstl_num"].notna()].groupby("FACPATID").size()
    stats["patients_with_longitudinal"] = int((multi > 1).sum())

    # --- Sub-scores (from latest encounter) ---
    sub_score_fields = {
        "speech": "Speech",
        "slvatn": "Salivation",
        "swallow": "Swallowing",
        "handwrit": "Handwriting",
        "drshygn": "Dressing & Hygiene",
        "turnbed": "Turning in Bed",
        "walking": "Walking",
        "clmbstrs": "Climbing Stairs",
        "dyspnea": "Dyspnea",
        "orthpnea": "Orthopnea",
        "respinsf": "Respiratory Insufficiency",
    }
    sub_scores = {}
    for field, label in sub_score_fields.items():
        if field in latest.columns:
            vals = latest[field].dropna()
            vals = vals[vals.astype(str).str.strip() != ""]
            if len(vals) >= SMALL_CELL:
                vc = vals.value_counts()
                sub_scores[field] = {
                    "label": label,
                    "distribution": _suppress_distribution(
                        {str(k): int(v) for k, v in vc.items()}
                    ),
                    "total_reported": int(vc.sum()),
                }
    stats["sub_scores"] = sub_scores

    return stats


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def _compute_diagnosis(diag_df: pd.DataFrame, patient_count: int) -> dict:
    """Compute El Escorial, body region onset, gene mutation, family history."""
    stats = {"available": True}

    # El Escorial criteria
    if "escorial" in diag_df.columns:
        vals = diag_df["escorial"].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        if not vals.empty:
            vc = vals.value_counts()
            stats["el_escorial"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # Body regions first affected
    if "bdypt" in diag_df.columns:
        vals = diag_df["bdypt"].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        if not vals.empty:
            vc = vals.value_counts()
            stats["body_region_onset"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # Gene mutation status
    if "genemut" in diag_df.columns:
        vals = diag_df["genemut"].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        if not vals.empty:
            vc = vals.value_counts()
            tested = int(vc.sum()) - int(vc.get("Not Tested", 0)) - int(vc.get("Unknown", 0))
            stats["gene_mutation"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
                "tested_count": tested,
                "tested_percentage": round(tested / patient_count * 100, 1) if patient_count else 0,
            }

    # Family history
    if "famhst" in diag_df.columns:
        vals = diag_df["famhst"].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        if not vals.empty:
            vc = vals.value_counts()
            familial = int(vc.get("Yes", 0))
            stats["family_history"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
                "familial_count": familial,
                "familial_percentage": round(familial / patient_count * 100, 1) if patient_count else 0,
            }

    return stats


# ---------------------------------------------------------------------------
# Milestones
# ---------------------------------------------------------------------------

def _compute_milestones(diag_df: pd.DataFrame, enc_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute disease milestones: onset/diagnosis age, delay, LOA, speech, gastrostomy, NIV."""
    stats = {"available": True}

    # --- Onset age ---
    if "alsonsag" in diag_df.columns:
        onset_ages = pd.to_numeric(diag_df["alsonsag"], errors="coerce").dropna()
        onset_ages = onset_ages[(onset_ages >= 0) & (onset_ages <= 110)]
        if len(onset_ages) >= SMALL_CELL:
            bins = list(range(0, int(onset_ages.max()) + 10, 5))
            if len(bins) < 2:
                bins = [0, 5, 10]
            stats["onset_age"] = {
                **_score_stats(onset_ages),
                "histogram": _score_histogram(onset_ages, bins),
            }

    # --- Diagnosis age ---
    if "alsdgnag" in diag_df.columns:
        dx_ages = pd.to_numeric(diag_df["alsdgnag"], errors="coerce").dropna()
        dx_ages = dx_ages[(dx_ages >= 0) & (dx_ages <= 110)]
        if len(dx_ages) >= SMALL_CELL:
            bins = list(range(0, int(dx_ages.max()) + 10, 5))
            if len(bins) < 2:
                bins = [0, 5, 10]
            stats["diagnosis_age"] = {
                **_score_stats(dx_ages),
                "histogram": _score_histogram(dx_ages, bins),
            }

    # --- Diagnostic delay (diagnosis age - onset age) ---
    if "alsonsag" in diag_df.columns and "alsdgnag" in diag_df.columns:
        onset = pd.to_numeric(diag_df["alsonsag"], errors="coerce")
        dx = pd.to_numeric(diag_df["alsdgnag"], errors="coerce")
        delay = (dx - onset).dropna()
        delay = delay[(delay >= 0) & (delay <= 30)]
        if len(delay) >= SMALL_CELL:
            bins = [0, 0.5, 1, 2, 3, 5, 10, 20, 30]
            stats["diagnostic_delay"] = {
                **_score_stats(delay),
                "histogram": _score_histogram(delay, bins),
            }

    # --- Functional milestones from encounters ---
    als_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)]
    latest = _latest_per_patient(enc_df, patient_ids)

    # Loss of ambulation
    if "amblloss" in als_enc.columns:
        amb_vals = als_enc.groupby("FACPATID")["amblloss"].last().dropna()
        amb_vals = amb_vals[amb_vals.astype(str).str.strip() != ""]
        if not amb_vals.empty:
            vc = amb_vals.value_counts()
            stats["loss_of_ambulation"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # Loss of speech
    if "spchloss" in als_enc.columns:
        speech_vals = als_enc.groupby("FACPATID")["spchloss"].last().dropna()
        speech_vals = speech_vals[speech_vals.astype(str).str.strip() != ""]
        if not speech_vals.empty:
            vc = speech_vals.value_counts()
            stats["loss_of_speech"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # Gastrostomy
    if "gstrmy" in als_enc.columns:
        gast_vals = als_enc.groupby("FACPATID")["gstrmy"].last().dropna()
        gast_vals = gast_vals[gast_vals.astype(str).str.strip() != ""]
        if not gast_vals.empty:
            vc = gast_vals.value_counts()
            stats["gastrostomy"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # NIV initiation
    if "fstniv" in als_enc.columns:
        niv_vals = als_enc.groupby("FACPATID")["fstniv"].last().dropna()
        niv_vals = niv_vals[niv_vals.astype(str).str.strip() != ""]
        if not niv_vals.empty:
            vc = niv_vals.value_counts()
            stats["niv_initiation"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    return stats


# ---------------------------------------------------------------------------
# Respiratory
# ---------------------------------------------------------------------------

def _compute_respiratory(enc_df: pd.DataFrame, demo_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute FVC % predicted, NIV/tracheostomy status."""
    stats = {"available": False}

    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return stats

    als_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()

    # --- FVC % Predicted ---
    if "fvcpctpd" in latest.columns:
        fvc = pd.to_numeric(latest["fvcpctpd"], errors="coerce").dropna()
        fvc = fvc[(fvc > 0) & (fvc <= 200)]
        if len(fvc) >= SMALL_CELL:
            stats["available"] = True
            bins = list(range(0, 160, 10))
            stats["fvc_pct"] = {
                **_score_stats(fvc),
                "histogram": _score_histogram(fvc, bins),
            }

            # Longitudinal FVC
            als_enc["encntdt"] = pd.to_datetime(als_enc["encntdt"], errors="coerce")
            als_enc["fvcpctpd_num"] = pd.to_numeric(als_enc["fvcpctpd"], errors="coerce")
            enrol_dt = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
            als_enc["enroldt"] = als_enc["FACPATID"].map(enrol_dt)
            als_enc["years_since_enrol"] = (
                (als_enc["encntdt"] - als_enc["enroldt"]).dt.days / 365.25
            )
            fvc_long = als_enc[["fvcpctpd_num", "years_since_enrol"]].dropna()
            fvc_long = fvc_long[(fvc_long["fvcpctpd_num"] > 0) & (fvc_long["fvcpctpd_num"] <= 200)]
            fvc_long["year_bucket"] = fvc_long["years_since_enrol"].clip(lower=0).astype(int).clip(upper=6)

            longitudinal = []
            for yr in range(7):
                bucket = fvc_long[fvc_long["year_bucket"] == yr]["fvcpctpd_num"]
                if len(bucket) >= SMALL_CELL:
                    longitudinal.append({
                        "year": yr if yr < 6 else "6+",
                        "median": round(float(bucket.median()), 1),
                        "q1": round(float(bucket.quantile(0.25)), 1),
                        "q3": round(float(bucket.quantile(0.75)), 1),
                        "n": int(len(bucket)),
                    })
            stats["fvc_pct"]["longitudinal"] = longitudinal

    # --- NIV status ---
    if "fstniv" in latest.columns:
        niv_vals = latest["fstniv"].dropna()
        niv_vals = niv_vals[niv_vals.astype(str).str.strip() != ""]
        if not niv_vals.empty:
            stats["available"] = True
            vc = niv_vals.value_counts()
            stats["niv_status"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    # --- Tracheostomy status ---
    if "trach" in latest.columns:
        trach_vals = latest["trach"].dropna()
        trach_vals = trach_vals[trach_vals.astype(str).str.strip() != ""]
        if not trach_vals.empty:
            stats["available"] = True
            vc = trach_vals.value_counts()
            stats["trach_status"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in vc.items()}
                ),
                "total_reported": int(vc.sum()),
            }

    return stats


# ---------------------------------------------------------------------------
# Medications
# ---------------------------------------------------------------------------

def _compute_medications(patient_ids: list, total_patients: int) -> dict:
    """Compute ALS medication utilization from encounter_medication_rg.parquet."""
    stats = {"available": False}

    # Load medication records
    med_path = Path(__file__).parent.parent / "data" / "encounter_medication_rg.parquet"
    if not med_path.exists():
        # Fall back to combo_drugs
        med_path = Path(__file__).parent.parent / "data" / "combo_drugs.parquet"
    if not med_path.exists():
        return stats

    try:
        meds_df = pd.read_parquet(med_path)
        meds_df = meds_df[meds_df["FACPATID"].isin(patient_ids)]
    except Exception:
        return stats

    if meds_df.empty:
        return stats

    stats["available"] = True

    # --- Specific ALS drugs ---
    drug_stats = {}
    for drug_name, search_terms in ALS_DRUGS.items():
        mask = pd.Series(False, index=meds_df.index)
        for term in search_terms:
            for col in ["medname", "StandardName", "Medications", "medoth"]:
                if col in meds_df.columns:
                    mask |= meds_df[col].str.contains(term, case=False, na=False)
        patients_on = meds_df.loc[mask, "FACPATID"].dropna().nunique()
        drug_stats[drug_name] = {
            **_suppress(patients_on),
            "percentage": round(patients_on / total_patients * 100, 1) if total_patients else 0,
        }
    stats["als_drugs"] = drug_stats

    # --- Top medications overall ---
    # Use the same multi-column search approach as als_drugs so that any
    # drug appearing in both charts shows identical counts.
    _MED_COLS = [c for c in ["StandardName", "medname", "Medications", "medoth"]
                 if c in meds_df.columns]
    if _MED_COLS:
        # Build a unified drug name per row: first non-empty value wins
        unified = meds_df[_MED_COLS[0]].copy()
        for col in _MED_COLS[1:]:
            unified = unified.fillna(meds_df[col])
        unified = unified.dropna()
        unified = unified[unified.astype(str).str.strip() != ""]
        unified = unified[~unified.isin(["Other", "other", ""])]

        tmp = meds_df.loc[unified.index].copy()
        tmp["_unified_drug"] = unified
        med_patient = tmp.groupby("_unified_drug")["FACPATID"].nunique()
        med_patient = med_patient.sort_values(ascending=False).head(15)
        top_drugs = []
        for med, count in med_patient.items():
            if count >= SMALL_CELL:
                top_drugs.append({
                    "drug": str(med),
                    "patients": int(count),
                    "percentage": round(count / total_patients * 100, 1),
                })

        # Reconcile: for known ALS drugs, use the multi-column search
        # count from als_drugs (which merges brand/generic synonyms across
        # all columns) so both charts show identical numbers.
        _als_drug_terms = {}  # lowercase search term → canonical als_drugs name
        for drug_name, terms in ALS_DRUGS.items():
            for t in terms:
                _als_drug_terms[t.lower()] = drug_name
        for td in top_drugs:
            td_lower = td["drug"].lower()
            canonical = _als_drug_terms.get(td_lower)
            if canonical and canonical in drug_stats:
                ds = drug_stats[canonical]
                if not ds.get("suppressed"):
                    td["patients"] = ds["count"]
                    td["percentage"] = ds["percentage"]

        stats["top_drugs"] = top_drugs

    return stats


# ---------------------------------------------------------------------------
# State distribution
# ---------------------------------------------------------------------------

def _compute_state_distribution(demo_df: pd.DataFrame, site_locations: list, patient_ids: list) -> dict:
    """Compute per-state patient counts."""
    # Build facility → state mapping
    state_map = {}
    for loc in site_locations:
        fid = loc.get("facility_id", "")
        state = loc.get("state", "")
        if fid and state:
            state_map[str(fid)] = state

    als_demo = demo_df[demo_df["FACPATID"].isin(patient_ids)].copy()
    als_demo["state"] = als_demo["FACILITY_DISPLAY_ID"].astype(str).map(state_map)

    patient_state = dict(zip(als_demo["FACPATID"], als_demo["state"]))

    state_data = {}
    for pid in patient_ids:
        state = patient_state.get(pid)
        if not state or pd.isna(state):
            state = "Unknown"
        state_data[state] = state_data.get(state, 0) + 1

    # HIPAA: suppress states with < SMALL_CELL patients
    states_list = []
    suppressed_total = 0
    for state, count in sorted(state_data.items(), key=lambda x: x[1], reverse=True):
        if state == "Unknown":
            continue
        if count < SMALL_CELL:
            suppressed_total += count
        else:
            states_list.append({
                "state": state,
                "total": int(count),
            })

    if suppressed_total > 0:
        states_list.append({
            "state": "Other States (n<11 each)",
            "total": int(suppressed_total),
        })

    return {
        "available": True,
        "total_states_mapped": len([s for s in state_data if s != "Unknown"]),
        "states": states_list,
    }


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def _compute_demographics(demo_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    """Compute diagnosis age histogram and gender distribution."""
    stats = {"available": True}

    # Age at diagnosis
    if "alsdgnag" in diag_df.columns:
        als_diag = diag_df[diag_df["FACPATID"].isin(demo_df["FACPATID"])]
        dx_ages = pd.to_numeric(als_diag["alsdgnag"], errors="coerce").dropna()
        dx_ages = dx_ages[(dx_ages >= 0) & (dx_ages <= 110)]
        if len(dx_ages) > 0:
            bins = list(range(0, int(dx_ages.max()) + 10, 5))
            if len(bins) < 2:
                bins = [0, 5, 10]
            stats["diagnosis_age"] = {
                **_score_stats(dx_ages),
                "histogram": _score_histogram(dx_ages, bins),
            }

    # Gender
    if "gender" in demo_df.columns:
        gender_counts = demo_df["gender"].value_counts()
        stats["gender"] = {
            "distribution": {str(k): int(v) for k, v in gender_counts.items()},
        }

    return stats


# ---------------------------------------------------------------------------
# Ambulatory
# ---------------------------------------------------------------------------

def _compute_data_quality_flags(diag_df: pd.DataFrame) -> dict:
    """Flag data quality issues for QA review.

    Checks numeric age fields for out-of-range values that are excluded
    from analytics (values < 0 or > 110).
    """
    flags = []

    _AGE_FIELDS = {
        "alsonsag": "Symptom Onset Age",
        "alsdgnag": "Diagnosis Age",
    }
    for field, label in _AGE_FIELDS.items():
        if field not in diag_df.columns:
            continue
        vals = pd.to_numeric(diag_df[field], errors="coerce").dropna()
        neg = vals[vals < 0]
        over = vals[vals > 110]
        if not neg.empty or not over.empty:
            flags.append({
                "field": field,
                "label": label,
                "issue": "out_of_range",
                "negative_values": int(len(neg)),
                "over_110_values": int(len(over)),
                "excluded_total": int(len(neg) + len(over)),
                "note": (
                    f"{int(len(neg) + len(over))} record(s) excluded from "
                    f"analytics: {int(len(neg))} negative, "
                    f"{int(len(over))} over 110."
                ),
            })

    return {
        "flags": flags,
        "total_flags": len(flags),
    }


def _compute_ambulatory(enc_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute ambulatory status from most recent encounter."""
    if "curramb" not in enc_df.columns:
        return {"available": False}

    als_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)]
    amb_data = als_enc[als_enc["curramb"].notna() & (als_enc["curramb"].astype(str).str.strip() != "")]

    if amb_data.empty:
        return {"available": False}

    if "encntdt" in amb_data.columns:
        amb_data = amb_data.copy()
        amb_data["encntdt"] = pd.to_datetime(amb_data["encntdt"], errors="coerce")
        latest = amb_data.sort_values("encntdt").groupby("FACPATID").last()
    else:
        latest = amb_data.groupby("FACPATID").last()

    amb_counts = latest["curramb"].value_counts()
    return {
        "available": True,
        "current_status": {
            "distribution": _suppress_distribution(
                {str(k): int(v) for k, v in amb_counts.items()}
            ),
        },
    }


# ---------------------------------------------------------------------------
# Facilities
# ---------------------------------------------------------------------------

def _compute_facility_stats(facility_info: dict) -> dict:
    """Compute facility distribution statistics."""
    if not facility_info or not facility_info.get("facilities"):
        return {"available": False}

    facilities = facility_info["facilities"]
    return {
        "available": True,
        "total_facilities": len(facilities),
        "facilities": [
            {
                "id": f.get("FACILITY_DISPLAY_ID", ""),
                "name": f.get("FACILITY_NAME", ""),
                "patients": int(f.get("patient_count", 0)),
            }
            for f in facilities
        ],
    }


# ---------------------------------------------------------------------------
# Main snapshot generation
# ---------------------------------------------------------------------------

def generate_als_snapshot() -> dict:
    """Generate comprehensive ALS clinical summary statistics snapshot."""

    print("Loading ALS cohort...")
    als_cohort = get_disease_cohort("ALS")

    demo_df = als_cohort["demographics"]
    diag_df = als_cohort["diagnosis"]
    enc_df = als_cohort["encounters"]
    patient_count = als_cohort["count"]
    patient_ids = als_cohort["patient_ids"]
    facility_info = als_cohort.get("facility_info", {})

    print(f"Processing {patient_count} ALS patients...")

    # Load state mapping from database_snapshot
    site_locations = []
    db_snap_path = Path(__file__).parent.parent / "stats" / "database_snapshot.json"
    if db_snap_path.exists():
        with open(db_snap_path) as f:
            db_snap = json.load(f)
        site_locations = db_snap.get("facilities", {}).get("site_locations", [])
        print(f"  {len(site_locations)} facility locations loaded for state mapping")

    # --- Data quality flags ---
    dq_flags = _compute_data_quality_flags(diag_df)

    snapshot = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": "ALS",
            "description": "Amyotrophic Lateral Sclerosis Clinical Summary Statistics",
        },
        "summary": {
            "total_patients": patient_count,
            "total_facilities": facility_info.get("total_facilities", 0),
        },
        "alsfrs": _compute_alsfrs(enc_df, demo_df, patient_ids),
        "diagnosis": _compute_diagnosis(diag_df, patient_count),
        "milestones": _compute_milestones(diag_df, enc_df, patient_ids),
        "respiratory": _compute_respiratory(enc_df, demo_df, patient_ids),
        "medications": _compute_medications(patient_ids, patient_count),
        "state_distribution": _compute_state_distribution(
            demo_df, site_locations, patient_ids
        ),
        "demographics": _compute_demographics(demo_df, diag_df),
        "ambulatory": _compute_ambulatory(enc_df, patient_ids),
        "facilities": _compute_facility_stats(facility_info),
        "data_quality": dq_flags,
    }

    return snapshot


def save_snapshot(snapshot: dict, output_path: Path):
    """Save snapshot to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\nSnapshot saved to: {output_path}")


def print_summary(snapshot: dict):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("ALS CLINICAL SUMMARY SNAPSHOT SUMMARY")
    print("=" * 70)

    print(f"\nGenerated: {snapshot['metadata']['generated_timestamp']}")
    print(f"Total ALS Patients: {snapshot['summary']['total_patients']:,}")
    print(f"Total Facilities: {snapshot['summary']['total_facilities']}")

    # ALSFRS-R
    alsfrs = snapshot.get("alsfrs", {})
    if alsfrs.get("available"):
        ts = alsfrs.get("total_score", {})
        print(f"\nALSFRS-R Total Score:")
        print(f"  Median: {ts.get('median')}, Mean: {ts.get('mean')}, n={ts.get('count')}")
        print(f"  Longitudinal: {len(ts.get('longitudinal', []))} year-buckets")
        print(f"  Patients with longitudinal data: {alsfrs.get('patients_with_longitudinal', 0)}")
        print(f"  Severity bands:")
        for band in alsfrs.get("severity_bands", []):
            print(f"    {band['label']}: {band.get('count', 0)} ({band.get('percentage', 0)}%)")
        print(f"  Sub-scores tracked: {len(alsfrs.get('sub_scores', {}))}")

    # Diagnosis
    dx = snapshot.get("diagnosis", {})
    ee = dx.get("el_escorial", {})
    if ee:
        print(f"\nEl Escorial Criteria: {ee.get('total_reported', 0)} patients classified")
    br = dx.get("body_region_onset", {})
    if br:
        print(f"Body Region Onset: {br.get('total_reported', 0)} patients")
    gm = dx.get("gene_mutation", {})
    if gm:
        print(f"Gene Mutation Testing: {gm.get('tested_percentage', 0)}% tested")
    fh = dx.get("family_history", {})
    if fh:
        print(f"Familial ALS: {fh.get('familial_percentage', 0)}% ({fh.get('familial_count', 0)} patients)")

    # Milestones
    ms = snapshot.get("milestones", {})
    if ms.get("onset_age"):
        print(f"\nOnset Age: median {ms['onset_age'].get('median')} years")
    if ms.get("diagnosis_age"):
        print(f"Diagnosis Age: median {ms['diagnosis_age'].get('median')} years")
    if ms.get("diagnostic_delay"):
        print(f"Diagnostic Delay: median {ms['diagnostic_delay'].get('median')} years")

    # Respiratory
    resp = snapshot.get("respiratory", {})
    if resp.get("fvc_pct"):
        fvc = resp["fvc_pct"]
        print(f"\nFVC % Predicted: median={fvc.get('median')}%, n={fvc.get('count')}")

    # Medications
    meds = snapshot.get("medications", {})
    if meds.get("available"):
        print("\nALS Medications:")
        for drug_name, drug_data in meds.get("als_drugs", {}).items():
            flag = " (suppressed)" if drug_data.get("suppressed") else ""
            print(f"  {drug_name}: {drug_data.get('count', 0)} patients ({drug_data.get('percentage', 0)}%){flag}")

    # States
    sd = snapshot.get("state_distribution", {})
    if sd.get("available"):
        print(f"\nState Distribution ({sd.get('total_states_mapped', 0)} states mapped)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate ALS clinical summary snapshot")
    parser.add_argument(
        "--output", "-o",
        default="stats/als_snapshot.json",
        help="Output file path (default: stats/als_snapshot.json)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress summary output")

    args = parser.parse_args()

    try:
        snapshot = generate_als_snapshot()
        output_path = Path(args.output)
        save_snapshot(snapshot, output_path)

        if not args.quiet:
            print_summary(snapshot)

        print("\nALS snapshot generation complete!")
        return 0

    except Exception as e:
        print(f"\nError generating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
