#!/usr/bin/env python3
"""
Generate SMA Clinical Summary Snapshot

Creates a JSON file with pre-computed SMA statistics including
HFMSE/CHOP-INTEND/RULM motor scores, SMA type classification,
SMN2 copy number genetics, treatment utilization, and respiratory function.

Usage:
    python scripts/generate_sma_snapshot.py
    python scripts/generate_sma_snapshot.py --output stats/sma_snapshot.json
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

SMALL_CELL = 11

# SMA-appropriate age bands (heavily pediatric)
_AGE_BANDS = [
    ("0-2", 0, 3),
    ("2-5", 3, 6),
    ("5-10", 6, 11),
    ("10-18", 11, 19),
    ("18-30", 19, 31),
    ("30+", 31, 200),
]

# SMA disease-modifying therapies
SMA_DRUGS = {
    "Spinraza (nusinersen)": ["spinraza", "nusinersen"],
    "Evrysdi (risdiplam)": ["evrysdi", "risdiplam"],
    "Zolgensma (onasemnogene)": ["zolgensma", "onasemnogene", "avxs-101"],
}


# ---------------------------------------------------------------------------
# HIPAA suppression helpers
# ---------------------------------------------------------------------------

def _suppress(count: int) -> dict:
    if count < SMALL_CELL:
        return {"count": 0, "suppressed": True}
    return {"count": int(count), "suppressed": False}


def _suppress_distribution(dist: dict, min_cell: int = SMALL_CELL) -> dict:
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
    series = series.dropna()
    if series.empty:
        return {}
    hist, edges = np.histogram(series, bins=bins)
    return {
        "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
        "counts": [int(c) for c in hist],
    }


def _latest_per_patient(enc_df: pd.DataFrame, patient_ids: list) -> pd.DataFrame:
    df = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
    if "encntdt" not in df.columns or df.empty:
        return pd.DataFrame()
    df["encntdt"] = pd.to_datetime(df["encntdt"], errors="coerce")
    df = df.dropna(subset=["encntdt"])
    return df.sort_values("encntdt").groupby("FACPATID").last().reset_index()


# ---------------------------------------------------------------------------
# Classification & Diagnosis
# ---------------------------------------------------------------------------

def _compute_classification(diag_df: pd.DataFrame, total_patients: int) -> dict:
    stats = {"available": False}
    if diag_df.empty:
        return stats
    stats["available"] = True

    # SMA type distribution
    if "smaclass" in diag_df.columns:
        vc = diag_df["smaclass"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        stats["sma_type"] = {
            "distribution": _suppress_distribution(dist),
            "total_reported": int(len(vc)),
        }

    # Diagnosis method
    if "smadgmad" in diag_df.columns:
        vc = diag_df["smadgmad"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        stats["diagnosis_method"] = {
            "distribution": _suppress_distribution(dist),
            "total_reported": int(len(vc)),
        }

    # Genetic confirmation
    if "smadgcnf" in diag_df.columns:
        vc = diag_df["smadgcnf"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        yes_count = sum(v for k, v in dist.items() if "yes" in k.lower() or "confirmed" in k.lower())
        stats["genetic_confirmation"] = {
            "distribution": _suppress_distribution(dist),
            "confirmed_percentage": round(yes_count / total_patients * 100, 1) if total_patients else 0,
        }

    # Age at diagnosis
    if "smadgnag" in diag_df.columns:
        dx_ages = pd.to_numeric(diag_df["smadgnag"], errors="coerce").dropna()
        dx_ages = dx_ages[(dx_ages >= 0) & (dx_ages <= 110)]
        if len(dx_ages) > 0:
            bins = [0, 0.5, 1, 2, 5, 10, 18, 30, 50, 80]
            stats["diagnosis_age"] = {
                **_score_stats(dx_ages),
                "histogram": _score_histogram(dx_ages, bins),
            }

    return stats


# ---------------------------------------------------------------------------
# Genetics & Molecular Testing
# ---------------------------------------------------------------------------

def _compute_genetics(diag_df: pd.DataFrame, total_patients: int) -> dict:
    stats = {"available": False}
    if diag_df.empty:
        return stats
    stats["available"] = True

    # SMN2 copy number (most prognostically important)
    if "smn2mutn" in diag_df.columns:
        vc = diag_df["smn2mutn"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        stats["smn2_copy_number"] = {
            "distribution": _suppress_distribution(dist),
            "total_reported": int(len(vc)),
        }

    # SMN1 copy number
    if "smn1cn" in diag_df.columns:
        vc = diag_df["smn1cn"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        stats["smn1_copy_number"] = {
            "distribution": _suppress_distribution(dist),
            "total_reported": int(len(vc)),
        }

    # SMN2 by SMA type cross-tabulation
    if "smn2mutn" in diag_df.columns and "smaclass" in diag_df.columns:
        cross = {}
        for sma_type in sorted(diag_df["smaclass"].dropna().unique()):
            mask = diag_df["smaclass"] == sma_type
            smn2 = diag_df.loc[mask, "smn2mutn"].dropna()
            smn2 = smn2[smn2.astype(str).str.strip() != ""]
            if len(smn2) >= SMALL_CELL:
                dist = {str(k): int(v) for k, v in smn2.value_counts().items()}
                cross[str(sma_type)] = _suppress_distribution(dist)
        if cross:
            stats["smn2_by_sma_type"] = cross

    # Family history
    if "nwfammbr" in diag_df.columns:
        vc = diag_df["nwfammbr"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        dist = {str(k): int(v) for k, v in vc.value_counts().items()}
        yes_count = sum(v for k, v in dist.items() if "yes" in k.lower())
        stats["family_history"] = {
            "distribution": _suppress_distribution(dist),
            "familial_count": int(yes_count),
            "familial_percentage": round(yes_count / total_patients * 100, 1) if total_patients else 0,
        }

    # Non-SMN SMA
    if "nonsma" in diag_df.columns:
        vc = diag_df["nonsma"].dropna()
        vc = vc[vc.astype(str).str.strip() != ""]
        if len(vc) > 0:
            dist = {str(k): int(v) for k, v in vc.value_counts().items()}
            stats["non_smn_sma"] = {
                "distribution": _suppress_distribution(dist),
            }

    return stats


# ---------------------------------------------------------------------------
# Motor Function Assessments (HFMSE, CHOP-INTEND, RULM)
# ---------------------------------------------------------------------------

def _compute_motor_score(enc_df: pd.DataFrame, demo_df: pd.DataFrame,
                         diag_df: pd.DataFrame, patient_ids: list,
                         field: str, valid_range: tuple, hist_bins: list,
                         label: str) -> dict:
    """Compute stats for a single motor score field."""
    if field not in enc_df.columns:
        return {"available": False}

    # Latest score per patient
    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty or field not in latest.columns:
        return {"available": False}

    scores = pd.to_numeric(latest[field], errors="coerce").dropna()
    scores = scores[(scores >= valid_range[0]) & (scores <= valid_range[1])]
    if len(scores) < SMALL_CELL:
        return {"available": False}

    result = {
        "available": True,
        "total_score": {
            **_score_stats(scores),
            "histogram": _score_histogram(scores, hist_bins),
        },
    }

    # By SMA type
    if "smaclass" in diag_df.columns:
        type_map = dict(zip(diag_df["FACPATID"], diag_df["smaclass"]))
        latest_with_type = latest.copy()
        latest_with_type["sma_type"] = latest_with_type["FACPATID"].map(type_map)
        by_type = {}
        for sma_type in sorted(latest_with_type["sma_type"].dropna().unique()):
            type_scores = pd.to_numeric(
                latest_with_type.loc[latest_with_type["sma_type"] == sma_type, field],
                errors="coerce"
            ).dropna()
            type_scores = type_scores[(type_scores >= valid_range[0]) & (type_scores <= valid_range[1])]
            if len(type_scores) >= SMALL_CELL:
                by_type[str(sma_type)] = _score_stats(type_scores)
        if by_type:
            result["by_sma_type"] = by_type

    # By age band
    if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
        age_map = {}
        try:
            dob = pd.to_datetime(demo_df["dob"], errors="coerce")
            enrol = pd.to_datetime(demo_df["enroldt"], errors="coerce")
            ages = ((enrol - dob).dt.days / 365.25).dropna()
            ages = ages[(ages >= 0) & (ages <= 110)]
            age_map = dict(zip(demo_df.loc[ages.index, "FACPATID"], ages))
        except Exception:
            pass

        if age_map:
            latest_with_age = latest.copy()
            latest_with_age["age"] = latest_with_age["FACPATID"].map(age_map)
            by_age = {}
            for band_label, low, high in _AGE_BANDS:
                mask = (latest_with_age["age"] >= low) & (latest_with_age["age"] < high)
                band_scores = pd.to_numeric(
                    latest_with_age.loc[mask, field], errors="coerce"
                ).dropna()
                band_scores = band_scores[(band_scores >= valid_range[0]) & (band_scores <= valid_range[1])]
                if len(band_scores) >= SMALL_CELL:
                    by_age[band_label] = _score_stats(band_scores)
            if by_age:
                result["by_age_band"] = by_age

    # Longitudinal (year-bucketed)
    all_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
    if "encntdt" in all_enc.columns and "enroldt" in demo_df.columns:
        all_enc["encntdt"] = pd.to_datetime(all_enc["encntdt"], errors="coerce")
        enrol_map = dict(zip(
            demo_df["FACPATID"],
            pd.to_datetime(demo_df["enroldt"], errors="coerce")
        ))
        all_enc["enroldt"] = all_enc["FACPATID"].map(enrol_map)
        all_enc["years_since"] = (all_enc["encntdt"] - all_enc["enroldt"]).dt.days / 365.25
        all_enc[field] = pd.to_numeric(all_enc[field], errors="coerce")
        valid = all_enc.dropna(subset=[field, "years_since"])
        valid = valid[(valid[field] >= valid_range[0]) & (valid[field] <= valid_range[1])]
        valid = valid[valid["years_since"] >= 0]

        # Patients with longitudinal data (2+ measurements)
        pt_counts = valid.groupby("FACPATID").size()
        result["patients_with_longitudinal"] = int((pt_counts >= 2).sum())
        result["total_data_points"] = int(len(valid))

        # Year buckets
        longitudinal = []
        for yr in range(7):
            if yr < 6:
                bucket = valid[(valid["years_since"] >= yr) & (valid["years_since"] < yr + 1)]
                yr_label = yr
            else:
                bucket = valid[valid["years_since"] >= 6]
                yr_label = "6+"
            if len(bucket) >= SMALL_CELL:
                longitudinal.append({
                    "year": yr_label,
                    "median": round(float(bucket[field].median()), 1),
                    "q1": round(float(bucket[field].quantile(0.25)), 1),
                    "q3": round(float(bucket[field].quantile(0.75)), 1),
                    "n": int(len(bucket)),
                })
        if longitudinal:
            result["longitudinal"] = longitudinal

    return result


def _compute_motor_scores(enc_df: pd.DataFrame, demo_df: pd.DataFrame,
                          diag_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute all SMA motor function scores."""
    hfmse = _compute_motor_score(
        enc_df, demo_df, diag_df, patient_ids,
        field="hfmsesc", valid_range=(0, 66),
        hist_bins=list(range(0, 70, 5)),
        label="HFMSE",
    )

    chop = _compute_motor_score(
        enc_df, demo_df, diag_df, patient_ids,
        field="cittlscr", valid_range=(0, 64),
        hist_bins=list(range(0, 68, 4)),
        label="CHOP-INTEND",
    )

    rulm = _compute_motor_score(
        enc_df, demo_df, diag_df, patient_ids,
        field="rulmcs", valid_range=(0, 37),
        hist_bins=list(range(0, 40, 3)),
        label="RULM",
    )

    available = any(s.get("available") for s in [hfmse, chop, rulm])
    return {
        "available": available,
        "hfmse": hfmse,
        "chop_intend": chop,
        "rulm": rulm,
    }


# ---------------------------------------------------------------------------
# Medications & Treatments
# ---------------------------------------------------------------------------

def _compute_therapeutics(patient_ids: list, total_patients: int,
                          diag_df: pd.DataFrame) -> dict:
    stats = {"available": False}

    med_path = Path(__file__).parent.parent / "data" / "encounter_medication_rg.parquet"
    if not med_path.exists():
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

    # Per-drug utilization
    drug_stats = {}
    all_on_therapy = set()
    for drug_name, search_terms in SMA_DRUGS.items():
        mask = pd.Series(False, index=meds_df.index)
        for term in search_terms:
            for col in ["medname", "StandardName", "Medications", "medoth"]:
                if col in meds_df.columns:
                    mask |= meds_df[col].str.contains(term, case=False, na=False)
        pts_on = set(meds_df.loc[mask, "FACPATID"].dropna())
        all_on_therapy |= pts_on
        drug_stats[drug_name] = {
            **_suppress(len(pts_on)),
            "percentage": round(len(pts_on) / total_patients * 100, 1) if total_patients else 0,
        }
    stats["sma_drugs"] = drug_stats
    stats["total_on_therapy"] = {
        "count": int(len(all_on_therapy)),
        "percentage": round(len(all_on_therapy) / total_patients * 100, 1) if total_patients else 0,
    }

    # Therapy by SMA type
    if "smaclass" in diag_df.columns:
        type_map = dict(zip(diag_df["FACPATID"], diag_df["smaclass"]))
        therapy_by_type = {}
        for sma_type in sorted(diag_df["smaclass"].dropna().unique()):
            type_pts = set(diag_df.loc[diag_df["smaclass"] == sma_type, "FACPATID"])
            type_meds = meds_df[meds_df["FACPATID"].isin(type_pts)]
            type_total = len(type_pts)
            if type_total < SMALL_CELL:
                continue
            type_drugs = {}
            type_any = set()
            for drug_name, search_terms in SMA_DRUGS.items():
                mask = pd.Series(False, index=type_meds.index)
                for term in search_terms:
                    for col in ["medname", "StandardName", "Medications", "medoth"]:
                        if col in type_meds.columns:
                            mask |= type_meds[col].str.contains(term, case=False, na=False)
                pts = set(type_meds.loc[mask, "FACPATID"].dropna())
                type_any |= pts
                count = len(pts)
                type_drugs[drug_name] = int(count) if count >= SMALL_CELL else 0
            type_drugs["total_on_therapy"] = int(len(type_any))
            type_drugs["total_patients"] = int(type_total)
            therapy_by_type[str(sma_type)] = type_drugs
        if therapy_by_type:
            stats["therapy_by_sma_type"] = therapy_by_type

    # Spinraza maintenance dose data
    spin_path = Path(__file__).parent.parent / "data" / "encounter_spinrazamaintenancedose_rg.parquet"
    if spin_path.exists():
        try:
            spin_df = pd.read_parquet(spin_path)
            spin_df = spin_df[spin_df["FACPATID"].isin(patient_ids)]
            stats["spinraza_maintenance"] = {
                "total_records": int(len(spin_df)),
                "unique_patients": int(spin_df["FACPATID"].nunique()) if not spin_df.empty else 0,
            }
        except Exception:
            pass

    # Top medications overall
    med_col = "medname" if "medname" in meds_df.columns else "StandardName" if "StandardName" in meds_df.columns else None
    if med_col:
        vals = meds_df[med_col].dropna()
        vals = vals[vals.astype(str).str.strip() != ""]
        vals = vals[~vals.isin(["Other", "other", ""])]
        med_patient = meds_df[meds_df[med_col].isin(vals)].groupby(med_col)["FACPATID"].nunique()
        med_patient = med_patient.sort_values(ascending=False).head(15)
        top_drugs = []
        for med, count in med_patient.items():
            if count >= SMALL_CELL:
                top_drugs.append({
                    "drug": str(med),
                    "patients": int(count),
                    "percentage": round(count / total_patients * 100, 1),
                })
        stats["top_drugs"] = top_drugs

    return stats


# ---------------------------------------------------------------------------
# Pulmonary & Respiratory
# ---------------------------------------------------------------------------

def _compute_respiratory(enc_df: pd.DataFrame, demo_df: pd.DataFrame,
                         patient_ids: list) -> dict:
    stats = {"available": False}

    if "fvcpctpd" not in enc_df.columns:
        return stats

    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return stats

    fvc = pd.to_numeric(latest["fvcpctpd"], errors="coerce").dropna()
    fvc = fvc[(fvc >= 0) & (fvc <= 200)]
    if len(fvc) < SMALL_CELL:
        return stats

    stats["available"] = True
    stats["fvc_pct"] = {
        **_score_stats(fvc),
        "histogram": _score_histogram(fvc, list(range(0, 160, 10))),
    }

    # Longitudinal FVC
    all_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
    all_enc["encntdt"] = pd.to_datetime(all_enc["encntdt"], errors="coerce")
    enrol_map = dict(zip(
        demo_df["FACPATID"],
        pd.to_datetime(demo_df["enroldt"], errors="coerce")
    ))
    all_enc["enroldt"] = all_enc["FACPATID"].map(enrol_map)
    all_enc["years_since"] = (all_enc["encntdt"] - all_enc["enroldt"]).dt.days / 365.25
    all_enc["fvcpctpd"] = pd.to_numeric(all_enc["fvcpctpd"], errors="coerce")
    valid = all_enc.dropna(subset=["fvcpctpd", "years_since"])
    valid = valid[(valid["fvcpctpd"] >= 0) & (valid["fvcpctpd"] <= 200)]
    valid = valid[valid["years_since"] >= 0]

    longitudinal = []
    for yr in range(7):
        if yr < 6:
            bucket = valid[(valid["years_since"] >= yr) & (valid["years_since"] < yr + 1)]
        else:
            bucket = valid[valid["years_since"] >= 6]
        if len(bucket) >= SMALL_CELL:
            longitudinal.append({
                "year": yr if yr < 6 else "6+",
                "median": round(float(bucket["fvcpctpd"].median()), 1),
                "q1": round(float(bucket["fvcpctpd"].quantile(0.25)), 1),
                "q3": round(float(bucket["fvcpctpd"].quantile(0.75)), 1),
                "n": int(len(bucket)),
            })
    if longitudinal:
        stats["fvc_pct"]["longitudinal"] = longitudinal

    return stats


# ---------------------------------------------------------------------------
# Milestones & Ambulatory Status
# ---------------------------------------------------------------------------

def _compute_milestones(enc_df: pd.DataFrame, patient_ids: list) -> dict:
    stats = {"available": False}
    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return stats

    stats["available"] = True

    # Ambulatory status
    if "curramb" in latest.columns:
        amb = latest["curramb"].dropna()
        amb = amb[amb.astype(str).str.strip() != ""]
        if len(amb) > 0:
            dist = {str(k): int(v) for k, v in amb.value_counts().items()}
            stats["ambulatory_status"] = {
                "distribution": _suppress_distribution(dist),
                "total_reported": int(len(amb)),
            }

    # Wheelchair use
    if "whlchr" in latest.columns:
        wc = latest["whlchr"].dropna()
        wc = wc[wc.astype(str).str.strip() != ""]
        if len(wc) > 0:
            dist = {str(k): int(v) for k, v in wc.value_counts().items()}
            stats["wheelchair"] = {
                "distribution": _suppress_distribution(dist),
                "total_reported": int(len(wc)),
            }

    # Ventilation support
    for field, label in [("respcare", "Respiratory Care"), ("ventstat", "Ventilation Status")]:
        if field in latest.columns:
            vc = latest[field].dropna()
            vc = vc[vc.astype(str).str.strip() != ""]
            if len(vc) > 0:
                dist = {str(k): int(v) for k, v in vc.value_counts().items()}
                stats[field] = {
                    "distribution": _suppress_distribution(dist),
                    "total_reported": int(len(vc)),
                }

    return stats


# ---------------------------------------------------------------------------
# State distribution (same pattern as ALS)
# ---------------------------------------------------------------------------

def _compute_state_distribution(demo_df: pd.DataFrame, site_locations: list,
                                patient_ids: list) -> dict:
    state_map = {}
    for loc in site_locations:
        fid = loc.get("facility_id", "")
        state = loc.get("state", "")
        if fid and state:
            state_map[str(fid)] = state

    sma_demo = demo_df[demo_df["FACPATID"].isin(patient_ids)].copy()
    sma_demo["state"] = sma_demo["FACILITY_DISPLAY_ID"].astype(str).map(state_map)

    patient_state = dict(zip(sma_demo["FACPATID"], sma_demo["state"]))

    state_data = {}
    for pid in patient_ids:
        state = patient_state.get(pid)
        if not state or pd.isna(state):
            state = "Unknown"
        state_data[state] = state_data.get(state, 0) + 1

    states_list = []
    suppressed_total = 0
    for state, count in sorted(state_data.items(), key=lambda x: x[1], reverse=True):
        if state == "Unknown":
            continue
        if count < SMALL_CELL:
            suppressed_total += count
        else:
            states_list.append({"state": state, "total": int(count)})

    if suppressed_total > 0:
        states_list.append({"state": "Other States (n<11 each)", "total": int(suppressed_total)})

    return {
        "available": True,
        "total_states_mapped": len([s for s in state_data if s != "Unknown"]),
        "states": states_list,
    }


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def _compute_demographics(demo_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    stats = {"available": True}

    # Age at diagnosis
    if "smadgnag" in diag_df.columns:
        sma_diag = diag_df[diag_df["FACPATID"].isin(demo_df["FACPATID"])]
        dx_ages = pd.to_numeric(sma_diag["smadgnag"], errors="coerce").dropna()
        dx_ages = dx_ages[(dx_ages >= 0) & (dx_ages <= 110)]
        if len(dx_ages) > 0:
            bins = [0, 0.5, 1, 2, 5, 10, 18, 30, 50, 80]
            stats["diagnosis_age"] = {
                **_score_stats(dx_ages),
                "histogram": _score_histogram(dx_ages, bins),
            }

    # Age at enrollment
    if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
        try:
            dob = pd.to_datetime(demo_df["dob"], errors="coerce")
            enrol = pd.to_datetime(demo_df["enroldt"], errors="coerce")
            ages = ((enrol - dob).dt.days / 365.25).dropna()
            ages = ages[(ages >= 0) & (ages <= 110)]
            if len(ages) > 0:
                bins = [0, 2, 5, 10, 18, 30, 50, 80]
                stats["enrollment_age"] = {
                    **_score_stats(ages),
                    "histogram": _score_histogram(ages, bins),
                }
        except Exception:
            pass

    # Gender
    if "gender" in demo_df.columns:
        gender_counts = demo_df["gender"].value_counts()
        stats["gender"] = {
            "distribution": {str(k): int(v) for k, v in gender_counts.items()},
        }

    return stats


# ---------------------------------------------------------------------------
# Ambulatory status (from latest encounter)
# ---------------------------------------------------------------------------

def _compute_ambulatory(enc_df: pd.DataFrame, patient_ids: list) -> dict:
    if "curramb" not in enc_df.columns:
        return {"available": False}

    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty or "curramb" not in latest.columns:
        return {"available": False}

    amb_data = latest[latest["curramb"].notna() & (latest["curramb"].astype(str).str.strip() != "")]
    if amb_data.empty:
        return {"available": False}

    amb_counts = amb_data["curramb"].value_counts()
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

def generate_sma_snapshot() -> dict:
    print("Loading SMA cohort...")
    sma_cohort = get_disease_cohort("SMA")

    demo_df = sma_cohort["demographics"]
    diag_df = sma_cohort["diagnosis"]
    enc_df = sma_cohort["encounters"]
    patient_count = sma_cohort["count"]
    patient_ids = sma_cohort["patient_ids"]
    facility_info = sma_cohort.get("facility_info", {})

    print(f"Processing {patient_count} SMA patients...")

    # Load state mapping from database_snapshot
    site_locations = []
    db_snap_path = Path(__file__).parent.parent / "stats" / "database_snapshot.json"
    if db_snap_path.exists():
        with open(db_snap_path) as f:
            db_snap = json.load(f)
        site_locations = db_snap.get("facilities", {}).get("site_locations", [])
        print(f"  {len(site_locations)} facility locations loaded for state mapping")

    snapshot = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": "SMA",
            "description": "Spinal Muscular Atrophy Clinical Summary Statistics",
        },
        "summary": {
            "total_patients": patient_count,
            "total_facilities": facility_info.get("total_facilities", 0),
        },
        "classification": _compute_classification(diag_df, patient_count),
        "genetics": _compute_genetics(diag_df, patient_count),
        "motor_scores": _compute_motor_scores(enc_df, demo_df, diag_df, patient_ids),
        "therapeutics": _compute_therapeutics(patient_ids, patient_count, diag_df),
        "respiratory": _compute_respiratory(enc_df, demo_df, patient_ids),
        "milestones": _compute_milestones(enc_df, patient_ids),
        "state_distribution": _compute_state_distribution(demo_df, site_locations, patient_ids),
        "demographics": _compute_demographics(demo_df, diag_df),
        "ambulatory": _compute_ambulatory(enc_df, patient_ids),
        "facilities": _compute_facility_stats(facility_info),
    }

    return snapshot


def save_snapshot(snapshot: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\nSnapshot saved to: {output_path}")


def print_summary(snapshot: dict):
    print("\n" + "=" * 70)
    print("SMA CLINICAL SUMMARY SNAPSHOT SUMMARY")
    print("=" * 70)

    print(f"\nGenerated: {snapshot['metadata']['generated_timestamp']}")
    print(f"Total SMA Patients: {snapshot['summary']['total_patients']:,}")
    print(f"Total Facilities: {snapshot['summary']['total_facilities']}")

    # Classification
    cls = snapshot.get("classification", {})
    if cls.get("available"):
        sma_type = cls.get("sma_type", {})
        print(f"\nSMA Type Distribution: {sma_type.get('total_reported', 0)} classified")
        for k, v in sma_type.get("distribution", {}).items():
            print(f"  {k}: {v}")

    # Genetics
    gen = snapshot.get("genetics", {})
    if gen.get("available"):
        smn2 = gen.get("smn2_copy_number", {})
        print(f"\nSMN2 Copy Number: {smn2.get('total_reported', 0)} reported")
        fh = gen.get("family_history", {})
        if fh:
            print(f"Familial SMA: {fh.get('familial_percentage', 0)}%")

    # Motor scores
    motor = snapshot.get("motor_scores", {})
    if motor.get("available"):
        for name, key in [("HFMSE", "hfmse"), ("CHOP-INTEND", "chop_intend"), ("RULM", "rulm")]:
            m = motor.get(key, {})
            if m.get("available"):
                ts = m.get("total_score", {})
                print(f"\n{name}: median={ts.get('median')}, n={ts.get('count')}")
                print(f"  Longitudinal patients: {m.get('patients_with_longitudinal', 0)}")

    # Therapeutics
    tx = snapshot.get("therapeutics", {})
    if tx.get("available"):
        print("\nSMA Therapeutics:")
        for drug, data in tx.get("sma_drugs", {}).items():
            flag = " (suppressed)" if data.get("suppressed") else ""
            print(f"  {drug}: {data.get('count', 0)} ({data.get('percentage', 0)}%){flag}")
        tot = tx.get("total_on_therapy", {})
        print(f"  Total on therapy: {tot.get('count', 0)} ({tot.get('percentage', 0)}%)")

    # Respiratory
    resp = snapshot.get("respiratory", {})
    if resp.get("fvc_pct"):
        fvc = resp["fvc_pct"]
        print(f"\nFVC % Predicted: median={fvc.get('median')}%, n={fvc.get('count')}")

    # States
    sd = snapshot.get("state_distribution", {})
    if sd.get("available"):
        print(f"\nState Distribution ({sd.get('total_states_mapped', 0)} states mapped)")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate SMA clinical summary snapshot")
    parser.add_argument(
        "--output", "-o",
        default="stats/sma_snapshot.json",
        help="Output file path (default: stats/sma_snapshot.json)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress summary output")

    args = parser.parse_args()

    try:
        snapshot = generate_sma_snapshot()
        output_path = Path(args.output)
        save_snapshot(snapshot, output_path)

        if not args.quiet:
            print_summary(snapshot)

        print("\nSMA snapshot generation complete!")
        return 0

    except Exception as e:
        print(f"\nError generating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
