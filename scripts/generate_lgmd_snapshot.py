#!/usr/bin/env python3
"""
Generate LGMD Deep Dive Snapshot

Creates a JSON file with pre-computed LGMD statistics for the OpenMOVR App.
This enables the LGMD Deep Dive to work without loading parquet files directly.

Usage:
    python scripts/generate_lgmd_snapshot.py
    python scripts/generate_lgmd_snapshot.py --output stats/lgmd_snapshot.json
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

# LGMD-appropriate age bands (adult-onset disease)
_LGMD_AGE_BANDS = [
    ("<18", 0, 18),
    ("18-30", 18, 31),
    ("30-40", 30, 41),
    ("40-50", 40, 51),
    ("50+", 50, 200),
]

# Clinically relevant medication categories for LGMD
DRUG_CATEGORIES = {
    "Cardiac / CV": [
        "lisinopril", "losartan", "metoprolol", "carvedilol", "enalapril",
        "ramipril", "amlodipine", "atenolol", "valsartan", "diltiazem", "aspirin",
    ],
    "Pain / Neuro": [
        "gabapentin", "ibuprofen", "acetaminophen", "naproxen", "tramadol",
        "pregabalin", "meloxicam", "cymbalta", "duloxetine",
    ],
    "Respiratory": [
        "albuterol", "ipratropium", "fluticasone", "tiotropium", "budesonide",
    ],
    "Corticosteroids": [
        "prednisone", "prednisolone", "deflazacort", "methylprednisolone",
        "dexamethasone",
    ],
    "Supplements": [
        "cholecalciferol", "vitamin d", "vitamin b12", "calcium", "magnesium",
        "coq10", "creatine", "ascorbic",
    ],
}


# ---------------------------------------------------------------------------
# HIPAA suppression helpers (same pattern as DMD snapshot)
# ---------------------------------------------------------------------------

def _suppress(count: int) -> dict:
    """Return count with suppression flag if below HIPAA threshold."""
    if count < SMALL_CELL:
        return {"count": 0, "suppressed": True}
    return {"count": int(count), "suppressed": False}


def _suppress_distribution(dist: dict, min_cell: int = SMALL_CELL) -> dict:
    """Suppress small categories, rolling into 'Suppressed (n<11)'."""
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
# Score computation helpers (same pattern as DMD snapshot)
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
# Existing compute helpers (unchanged)
# ---------------------------------------------------------------------------

def _compute_subtype_stats(diag_df: pd.DataFrame, total_patients: int) -> dict:
    """Compute subtype distribution statistics."""

    if 'lgtype' not in diag_df.columns:
        return {"available": False}

    subtype_counts = diag_df['lgtype'].value_counts()
    subtype_counts = subtype_counts[subtype_counts.index.astype(str).str.strip() != '']

    subtypes_list = []
    for subtype, count in subtype_counts.items():
        if 'LGMD1' in str(subtype):
            lgmd_type = 'LGMD Type 1 (Dominant)'
        elif 'LGMD2' in str(subtype):
            lgmd_type = 'LGMD Type 2 (Recessive)'
        else:
            lgmd_type = 'Other/Undetermined'

        subtypes_list.append({
            "subtype": str(subtype),
            "patients": int(count),
            "percentage": round(count / total_patients * 100, 1),
            "lgmd_type": lgmd_type
        })

    return {
        "available": True,
        "unique_subtypes": len(subtypes_list),
        "distribution": subtypes_list,
        "all_subtypes": sorted([s['subtype'] for s in subtypes_list])
    }


def _compute_demographic_stats(demo_df: pd.DataFrame) -> dict:
    """Compute demographic statistics."""

    stats = {"available": True}

    if 'dob' in demo_df.columns:
        dob_dates = pd.to_datetime(demo_df['dob'], errors='coerce')
        current_ages = (datetime.now() - dob_dates).dt.days / 365.25
        current_ages = current_ages.dropna()

        if len(current_ages) > 0:
            age_bins = list(range(0, int(current_ages.max()) + 10, 5))
            hist, edges = np.histogram(current_ages, bins=age_bins)

            stats["current_age"] = {
                "min": round(float(current_ages.min()), 1),
                "max": round(float(current_ages.max()), 1),
                "median": round(float(current_ages.median()), 1),
                "mean": round(float(current_ages.mean()), 1),
                "count": int(len(current_ages)),
                "histogram": {
                    "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
                    "counts": [int(c) for c in hist]
                }
            }

    if 'enroldt' in demo_df.columns and 'dob' in demo_df.columns:
        enrol_dates = pd.to_datetime(demo_df['enroldt'], errors='coerce')
        dob_dates = pd.to_datetime(demo_df['dob'], errors='coerce')
        age_at_enroll = (enrol_dates - dob_dates).dt.days / 365.25
        age_at_enroll = age_at_enroll.dropna()

        if len(age_at_enroll) > 0:
            age_bins = list(range(0, int(age_at_enroll.max()) + 10, 5))
            hist, edges = np.histogram(age_at_enroll, bins=age_bins)

            stats["enrollment_age"] = {
                "min": round(float(age_at_enroll.min()), 1),
                "max": round(float(age_at_enroll.max()), 1),
                "median": round(float(age_at_enroll.median()), 1),
                "mean": round(float(age_at_enroll.mean()), 1),
                "count": int(len(age_at_enroll)),
                "histogram": {
                    "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
                    "counts": [int(c) for c in hist]
                }
            }

    if 'gender' in demo_df.columns:
        gender_counts = demo_df['gender'].value_counts()
        stats["gender"] = {
            "distribution": {str(k): int(v) for k, v in gender_counts.items()},
            "total": int(gender_counts.sum())
        }

    if 'ethnic' in demo_df.columns:
        ethnic_counts = demo_df['ethnic'].value_counts().head(10)
        stats["ethnicity"] = {
            "distribution": {str(k): int(v) for k, v in ethnic_counts.items()},
            "total": int(demo_df['ethnic'].notna().sum())
        }

    return stats


def _compute_diagnosis_stats(diag_df: pd.DataFrame, patient_count: int) -> dict:
    """Compute diagnosis-related statistics."""

    stats = {"available": True}

    if 'lgdgag' in diag_df.columns:
        dx_ages = pd.to_numeric(diag_df['lgdgag'], errors='coerce').dropna()
        if len(dx_ages) > 0:
            age_bins = list(range(0, int(dx_ages.max()) + 10, 5))
            hist, edges = np.histogram(dx_ages, bins=age_bins)

            stats["diagnosis_age"] = {
                "min": round(float(dx_ages.min()), 1),
                "max": round(float(dx_ages.max()), 1),
                "median": round(float(dx_ages.median()), 1),
                "mean": round(float(dx_ages.mean()), 1),
                "count": int(len(dx_ages)),
                "histogram": {
                    "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
                    "counts": [int(c) for c in hist]
                }
            }

    if 'dymonag' in diag_df.columns:
        onset_ages = pd.to_numeric(diag_df['dymonag'], errors='coerce').dropna()
        if len(onset_ages) > 0:
            age_bins = list(range(0, int(onset_ages.max()) + 10, 5))
            hist, edges = np.histogram(onset_ages, bins=age_bins)

            stats["onset_age"] = {
                "min": round(float(onset_ages.min()), 1),
                "max": round(float(onset_ages.max()), 1),
                "median": round(float(onset_ages.median()), 1),
                "mean": round(float(onset_ages.mean()), 1),
                "count": int(len(onset_ages)),
                "histogram": {
                    "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
                    "counts": [int(c) for c in hist]
                }
            }

    if 'lggntcf' in diag_df.columns:
        genetic_counts = diag_df['lggntcf'].fillna('Unknown').value_counts()
        confirmed = (genetic_counts.get('Yes – Laboratory confirmation', 0) +
                    genetic_counts.get('Yes – In a family member', 0))

        stats["genetic_confirmation"] = {
            "distribution": {str(k): int(v) for k, v in genetic_counts.items()},
            "confirmed_count": int(confirmed),
            "confirmed_percentage": round(confirmed / patient_count * 100, 1) if patient_count > 0 else 0
        }

    return stats


def _compute_clinical_stats(diag_df: pd.DataFrame, enc_df: pd.DataFrame) -> dict:
    """Compute clinical characteristics statistics."""

    stats = {"available": True}

    if 'lgmscbp' in diag_df.columns:
        biopsy_counts = diag_df['lgmscbp'].fillna('Unknown').value_counts()
        stats["muscle_biopsy"] = {
            "distribution": {str(k): int(v) for k, v in biopsy_counts.items()}
        }

    if 'lgfam' in diag_df.columns:
        family_counts = diag_df['lgfam'].fillna('Unknown').value_counts()
        stats["family_history"] = {
            "distribution": {str(k): int(v) for k, v in family_counts.items()}
        }

    if 'sym1st' in diag_df.columns:
        symptoms = diag_df['sym1st'].dropna()
        symptom_list = []
        for s in symptoms:
            if pd.notna(s) and str(s).strip():
                for symptom in str(s).split(','):
                    symptom_list.append(symptom.strip())

        if symptom_list:
            symptom_counts = pd.Series(symptom_list).value_counts().head(10)
            stats["first_symptoms"] = {
                "distribution": {str(k): int(v) for k, v in symptom_counts.items()}
            }

    return stats


def _compute_ambulatory_stats(enc_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    """Compute ambulatory status statistics."""

    if 'curramb' not in enc_df.columns:
        return {"available": False}

    stats = {"available": True}

    amb_status = enc_df[enc_df['curramb'].notna() & (enc_df['curramb'] != '')]

    if amb_status.empty:
        return {"available": False}

    if 'encntdt' in amb_status.columns:
        amb_status = amb_status.copy()
        amb_status['encntdt'] = pd.to_datetime(amb_status['encntdt'], errors='coerce')
        latest_amb = amb_status.sort_values('encntdt').groupby('FACPATID').last()
    else:
        latest_amb = amb_status.groupby('FACPATID').last()

    amb_counts = latest_amb['curramb'].value_counts()
    stats["current_status"] = {
        "distribution": {str(k): int(v) for k, v in amb_counts.items()}
    }

    if 'lgtype' in diag_df.columns:
        enc_with_type = enc_df.merge(
            diag_df[['FACPATID', 'lgtype']],
            on='FACPATID',
            how='left'
        )
        enc_with_type = enc_with_type[enc_with_type['curramb'].notna() & (enc_with_type['curramb'] != '')]

        if not enc_with_type.empty:
            if 'encntdt' in enc_with_type.columns:
                enc_with_type = enc_with_type.copy()
                enc_with_type['encntdt'] = pd.to_datetime(enc_with_type['encntdt'], errors='coerce')
                latest = enc_with_type.sort_values('encntdt').groupby('FACPATID').last().reset_index()
            else:
                latest = enc_with_type.groupby('FACPATID').last().reset_index()

            top_subtypes = latest['lgtype'].value_counts().head(5).index.tolist()
            latest_top = latest[latest['lgtype'].isin(top_subtypes)]

            if not latest_top.empty:
                by_subtype = {}
                for subtype in top_subtypes:
                    subtype_data = latest_top[latest_top['lgtype'] == subtype]
                    status_counts = subtype_data['curramb'].value_counts()
                    by_subtype[str(subtype)] = {str(k): int(v) for k, v in status_counts.items()}

                stats["by_subtype"] = by_subtype

    return stats


def _compute_facility_stats(facility_info: dict) -> dict:
    """Compute facility distribution statistics."""

    if not facility_info or not facility_info.get('facilities'):
        return {"available": False}

    facilities = facility_info['facilities']

    return {
        "available": True,
        "total_facilities": len(facilities),
        "facilities": [
            {
                "id": f.get('FACILITY_DISPLAY_ID', ''),
                "name": f.get('FACILITY_NAME', ''),
                "patients": int(f.get('patient_count', 0))
            }
            for f in facilities
        ]
    }


# ---------------------------------------------------------------------------
# NEW: Functional scores
# ---------------------------------------------------------------------------

def _compute_functional_scores(
    enc_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    patient_ids: list,
) -> dict:
    """Compute LGMD functional outcome measures: FVC, timed walk, ambulatory status."""
    stats = {"available": True}

    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return {"available": False}

    # Enrollment age for stratification
    enrol_ages = {}
    if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
        dob = pd.to_datetime(demo_df.set_index("FACPATID")["dob"], errors="coerce")
        enrol = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
        age = ((enrol - dob).dt.days / 365.25).dropna()
        enrol_ages = age.to_dict()

    def _by_age_band(df_with_pid: pd.DataFrame, value_col: str) -> dict:
        """Stratify a score by LGMD-appropriate age band at enrollment."""
        bands = {}
        for label, lo, hi in _LGMD_AGE_BANDS:
            mask = df_with_pid["FACPATID"].map(
                lambda pid: lo <= enrol_ages.get(pid, -1) < hi
            )
            subset = df_with_pid.loc[mask, value_col].dropna()
            if len(subset) >= SMALL_CELL:
                bands[label] = _score_stats(subset)
            elif len(subset) > 0:
                bands[label] = {"count": 0, "suppressed": True}
        return bands

    # --- FVC % Predicted ---
    if "fvcpctpd" in latest.columns:
        fvc = pd.to_numeric(latest["fvcpctpd"], errors="coerce").dropna()
        fvc = fvc[(fvc > 0) & (fvc <= 200)]
        if len(fvc) >= SMALL_CELL:
            bins = list(range(0, 160, 10))
            stats["fvc_pct"] = {
                **_score_stats(fvc),
                "histogram": _score_histogram(fvc, bins),
                "by_age_band": _by_age_band(
                    latest[["FACPATID", "fvcpctpd"]].assign(
                        fvcpctpd=pd.to_numeric(latest["fvcpctpd"], errors="coerce")
                    ),
                    "fvcpctpd",
                ),
            }

            # Longitudinal FVC (all encounters, not just latest)
            all_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
            all_enc["encntdt"] = pd.to_datetime(all_enc["encntdt"], errors="coerce")
            all_enc["fvcpctpd"] = pd.to_numeric(all_enc["fvcpctpd"], errors="coerce")
            enrol_dt = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
            all_enc["enroldt"] = all_enc["FACPATID"].map(enrol_dt)
            all_enc["years_since_enrol"] = (
                (all_enc["encntdt"] - all_enc["enroldt"]).dt.days / 365.25
            )
            fvc_long = all_enc[["fvcpctpd", "years_since_enrol"]].dropna()
            fvc_long = fvc_long[(fvc_long["fvcpctpd"] > 0) & (fvc_long["fvcpctpd"] <= 200)]
            fvc_long["year_bucket"] = fvc_long["years_since_enrol"].clip(lower=0).astype(int).clip(upper=6)

            longitudinal = []
            for yr in range(7):
                bucket = fvc_long[fvc_long["year_bucket"] == yr]["fvcpctpd"]
                if len(bucket) >= SMALL_CELL:
                    longitudinal.append({
                        "year": yr if yr < 6 else "6+",
                        "median": round(float(bucket.median()), 1),
                        "q1": round(float(bucket.quantile(0.25)), 1),
                        "q3": round(float(bucket.quantile(0.75)), 1),
                        "n": int(len(bucket)),
                    })
            stats["fvc_pct"]["longitudinal"] = longitudinal

    # --- Timed 10m Walk ---
    if "ttwr10m" in latest.columns:
        ttwr = pd.to_numeric(latest["ttwr10m"], errors="coerce").dropna()
        ttwr = ttwr[(ttwr > 0) & (ttwr <= 200)]
        if len(ttwr) >= SMALL_CELL:
            bins = list(range(0, 30, 3))
            stats["timed_10m_walk"] = {
                **_score_stats(ttwr),
                "histogram": _score_histogram(ttwr, bins),
                "by_age_band": _by_age_band(
                    latest[["FACPATID", "ttwr10m"]].assign(
                        ttwr10m=pd.to_numeric(latest["ttwr10m"], errors="coerce")
                    ),
                    "ttwr10m",
                ),
            }

    # --- Ambulatory Status (latest per patient) ---
    if "curramb" in latest.columns:
        amb = latest["curramb"].dropna()
        amb = amb[amb.astype(str).str.strip() != ""]
        if len(amb) >= SMALL_CELL:
            stats["ambulatory_status"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in amb.value_counts().items()}
                ),
                "total_reported": int(len(amb)),
            }

    return stats


# ---------------------------------------------------------------------------
# NEW: State distribution
# ---------------------------------------------------------------------------

def _compute_state_distribution(
    demo_df: pd.DataFrame,
    site_locations: list,
    patient_ids: list,
) -> dict:
    """Compute per-state patient counts with HIPAA suppression."""

    # Build facility → state mapping
    state_map = {}
    for loc in site_locations:
        fid = loc.get("facility_id", "")
        state = loc.get("state", "")
        if fid and state:
            state_map[str(fid)] = state

    # Map each patient to a state
    lgmd_demo = demo_df[demo_df["FACPATID"].isin(patient_ids)].copy()
    lgmd_demo["state"] = lgmd_demo["FACILITY_DISPLAY_ID"].astype(str).map(state_map)

    # Aggregate by state
    state_data = {}
    for _, row in lgmd_demo.iterrows():
        state = row.get("state")
        if not state or pd.isna(state):
            state = "Unknown"
        state_data[state] = state_data.get(state, 0) + 1

    # HIPAA suppress states with < SMALL_CELL patients
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
# NEW: Medication utilization
# ---------------------------------------------------------------------------

def _compute_medication_stats(
    meds_df: pd.DataFrame,
    patient_ids: list,
    patient_count: int,
) -> dict:
    """Compute medication utilization by drug category and top individual drugs."""

    if meds_df is None or meds_df.empty:
        return {"available": False}

    lgmd_meds = meds_df[meds_df["FACPATID"].isin(patient_ids)]
    if lgmd_meds.empty:
        return {"available": False}

    std_col = "StandardName" if "StandardName" in lgmd_meds.columns else "Medications"
    drug_lower = lgmd_meds[std_col].dropna().str.lower()

    # Drug categories
    categories = {}
    for cat_name, cat_drugs in DRUG_CATEGORIES.items():
        pattern = "|".join(cat_drugs)
        mask = drug_lower.str.contains(pattern, case=False, na=False)
        pts = lgmd_meds.loc[mask[mask].index, "FACPATID"].nunique()
        if pts >= SMALL_CELL:
            categories[cat_name] = {
                "patients": int(pts),
                "percentage": round(pts / patient_count * 100, 1),
                "records": int(mask.sum()),
            }

    # Top individual medications
    drug_counts = lgmd_meds[std_col].dropna().value_counts()
    top_drugs = []
    for drug_name, count in drug_counts.head(20).items():
        pts = lgmd_meds[lgmd_meds[std_col] == drug_name]["FACPATID"].nunique()
        if pts >= SMALL_CELL:
            top_drugs.append({
                "drug": str(drug_name),
                "patients": int(pts),
                "records": int(count),
            })
        if len(top_drugs) >= 15:
            break

    return {
        "available": True,
        "total_records": int(len(lgmd_meds)),
        "total_patients_with_meds": int(lgmd_meds["FACPATID"].nunique()),
        "categories": categories,
        "top_drugs": top_drugs,
    }


# ---------------------------------------------------------------------------
# NEW: Diagnostic journey
# ---------------------------------------------------------------------------

def _compute_diagnostic_journey(diag_df: pd.DataFrame, patient_count: int) -> dict:
    """Compute diagnostic delay: time from symptom onset to diagnosis."""

    stats = {"available": True}

    dx_age = pd.to_numeric(diag_df.get("lgdgag"), errors="coerce") if "lgdgag" in diag_df.columns else pd.Series(dtype=float)
    onset_age = pd.to_numeric(diag_df.get("dymonag"), errors="coerce") if "dymonag" in diag_df.columns else pd.Series(dtype=float)

    mask = dx_age.notna() & onset_age.notna()
    delays = dx_age[mask] - onset_age[mask]
    # Clean: only non-negative delays under 60 years
    delays = delays[(delays >= 0) & (delays < 60)]

    if len(delays) >= SMALL_CELL:
        bins = [0, 2, 5, 10, 15, 20, 30, 60]
        stats["delay"] = {
            **_score_stats(delays),
            "histogram": _score_histogram(delays, bins),
        }
    else:
        stats["delay"] = {"count": 0, "suppressed": True}

    # Also include onset age and diagnosis age distributions for side-by-side comparison
    onset_clean = onset_age.dropna()
    onset_clean = onset_clean[onset_clean >= 0]
    if len(onset_clean) >= SMALL_CELL:
        bins_age = list(range(0, int(onset_clean.max()) + 10, 5))
        stats["onset_age"] = {
            **_score_stats(onset_clean),
            "histogram": _score_histogram(onset_clean, bins_age),
        }

    dx_clean = dx_age.dropna()
    dx_clean = dx_clean[dx_clean >= 0]
    if len(dx_clean) >= SMALL_CELL:
        bins_age = list(range(0, int(dx_clean.max()) + 10, 5))
        stats["diagnosis_age"] = {
            **_score_stats(dx_clean),
            "histogram": _score_histogram(dx_clean, bins_age),
        }

    return stats


# ---------------------------------------------------------------------------
# Main snapshot generation
# ---------------------------------------------------------------------------

def generate_lgmd_snapshot() -> dict:
    """Generate comprehensive LGMD deep dive statistics snapshot."""

    print("Loading LGMD cohort...")
    lgmd_cohort = get_disease_cohort('LGMD')

    demo_df = lgmd_cohort['demographics']
    diag_df = lgmd_cohort['diagnosis']
    enc_df = lgmd_cohort['encounters']
    patient_count = lgmd_cohort['count']
    patient_ids = lgmd_cohort['patient_ids']
    facility_info = lgmd_cohort.get('facility_info', {})

    print(f"Processing {patient_count} LGMD patients...")

    # Load medications
    meds_df = None
    combo_path = Path(__file__).parent.parent / "data" / "combo_drugs.parquet"
    if combo_path.exists():
        print("Loading combo_drugs.parquet...")
        meds_df = pd.read_parquet(combo_path)
        meds_df = meds_df[meds_df["FACPATID"].isin(patient_ids)]
        print(f"  {len(meds_df)} LGMD medication records")
    else:
        print("WARNING: combo_drugs.parquet not found — medication analysis will be limited")

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
            "disease": "LGMD",
            "description": "Limb-Girdle Muscular Dystrophy Deep Dive Statistics"
        },

        "summary": {
            "total_patients": patient_count,
            "total_facilities": facility_info.get('total_facilities', 0),
            "patient_ids": lgmd_cohort['patient_ids']
        },

        "subtypes": _compute_subtype_stats(diag_df, patient_count),
        "demographics": _compute_demographic_stats(demo_df),
        "diagnosis": _compute_diagnosis_stats(diag_df, patient_count),
        "clinical": _compute_clinical_stats(diag_df, enc_df),
        "ambulatory": _compute_ambulatory_stats(enc_df, diag_df),
        "facilities": _compute_facility_stats(facility_info),
        "functional_scores": _compute_functional_scores(enc_df, demo_df, patient_ids),
        "state_distribution": _compute_state_distribution(demo_df, site_locations, patient_ids),
        "medications": _compute_medication_stats(meds_df, patient_ids, patient_count),
        "diagnostic_journey": _compute_diagnostic_journey(diag_df, patient_count),
    }

    return snapshot


def save_snapshot(snapshot: dict, output_path: Path):
    """Save snapshot to JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\nSnapshot saved to: {output_path}")


def print_summary(snapshot: dict):
    """Print human-readable summary."""

    print("\n" + "="*70)
    print("LGMD DEEP DIVE SNAPSHOT SUMMARY")
    print("="*70)

    print(f"\nGenerated: {snapshot['metadata']['generated_timestamp']}")
    print(f"\nTotal LGMD Patients: {snapshot['summary']['total_patients']:,}")
    print(f"Total Facilities: {snapshot['summary']['total_facilities']}")

    if snapshot['subtypes'].get('available'):
        print(f"\nSubtypes Represented: {snapshot['subtypes']['unique_subtypes']}")
        print("Top 5 Subtypes:")
        for s in snapshot['subtypes']['distribution'][:5]:
            print(f"   {s['subtype']}: {s['patients']} ({s['percentage']}%)")

    if snapshot['diagnosis'].get('genetic_confirmation'):
        gc = snapshot['diagnosis']['genetic_confirmation']
        print(f"\nGenetic Confirmation: {gc['confirmed_percentage']}% ({gc['confirmed_count']} patients)")

    if snapshot['diagnosis'].get('diagnosis_age'):
        da = snapshot['diagnosis']['diagnosis_age']
        print(f"Median Age at Diagnosis: {da['median']} years")

    # Functional scores
    fs = snapshot.get('functional_scores', {})
    if fs.get('fvc_pct'):
        fvc = fs['fvc_pct']
        print(f"\nFVC % Predicted: median {fvc['median']}%, n={fvc['count']}")
    if fs.get('timed_10m_walk'):
        tw = fs['timed_10m_walk']
        print(f"Timed 10m Walk: median {tw['median']}s, n={tw['count']}")

    # Diagnostic journey
    dj = snapshot.get('diagnostic_journey', {})
    if dj.get('delay') and not dj['delay'].get('suppressed'):
        d = dj['delay']
        print(f"\nDiagnostic Delay: median {d['median']} years (n={d['count']})")

    # State distribution
    sd = snapshot.get('state_distribution', {})
    if sd.get('states'):
        print(f"\nState Distribution: {sd['total_states_mapped']} states")
        for s in sd['states'][:5]:
            print(f"   {s['state']}: {s['total']}")

    # Medications
    mx = snapshot.get('medications', {})
    if mx.get('categories'):
        print(f"\nMedication Categories:")
        for cat, info in mx['categories'].items():
            print(f"   {cat}: {info['patients']} patients ({info['percentage']}%)")

    if snapshot['facilities'].get('available'):
        print(f"\nTop 3 Facilities:")
        for f in snapshot['facilities']['facilities'][:3]:
            print(f"   {f['name']}: {f['patients']} patients")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate LGMD deep dive snapshot")
    parser.add_argument("--output", "-o",
                       default="stats/lgmd_snapshot.json",
                       help="Output file path (default: stats/lgmd_snapshot.json)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress summary output")

    args = parser.parse_args()

    try:
        snapshot = generate_lgmd_snapshot()

        output_path = Path(args.output)
        save_snapshot(snapshot, output_path)

        if not args.quiet:
            print_summary(snapshot)

        print("\nLGMD snapshot generation complete!")
        return 0

    except Exception as e:
        print(f"\nError generating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
