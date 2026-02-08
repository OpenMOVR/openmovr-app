#!/usr/bin/env python3
"""
Generate DMD Deep Dive Snapshot

Creates a JSON file with pre-computed DMD statistics including
exon-skipping amenability analysis, steroid use, and state distribution.

Usage:
    python scripts/generate_dmd_snapshot.py
    python scripts/generate_dmd_snapshot.py --output stats/dmd_snapshot.json
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
# Constants ported from scripts/legacy/registry_meds_DMD_summary.py
# ---------------------------------------------------------------------------

# Mutation types that qualify a patient as amenable to exon-skipping therapy
AMENABLE_MUT_TYPES = ["Deletion", "Nonsense mutation", "Missense"]

# FDA-approved and pipeline exon-skipping therapies for DMD
# range: [from_exon, to_exon] inclusive — patient's deleted exon range must overlap
# search: case-insensitive terms to match in StandardName/Medications medication fields
EXON_SKIP_DRUGS = {
    "Exondys 51 (eteplirsen)": {
        "target_exon": 51,
        "range": [50, 52],
        "search": ["Exondys", "Eteplirsen", "eteplirsen"],
    },
    "Vyondys 53 (golodirsen)": {
        "target_exon": 53,
        "range": [52, 54],
        "search": ["Golodirsen", "Vyondys", "golodirsen"],
    },
    "Viltepso (viltolarsen)": {
        "target_exon": 53,
        "range": [52, 54],
        "search": ["Viltepso", "Viltolarsen", "viltolarsen"],
    },
    "Amondys 45 (casimersen)": {
        "target_exon": 45,
        "range": [44, 46],
        "search": ["Casimersen", "Amondys", "casimersen"],
    },
    "AOC 1044": {
        "target_exon": 44,
        "range": [43, 45],
        "search": ["AOC 1044", "AOC1044"],
    },
}

# Steroid drug classes for medication-level breakdown
STEROID_CLASSES = {
    "Deflazacort / Emflaza": ["deflazacort", "emflaza", "calcort"],
    "Prednisone / Prednisolone": ["prednisone", "prednisolone", "rayos", "deltasone", "sterapred"],
    "Vamorolone / Agamree": ["vamorolone", "agamree", "vbp-15"],
}

# HIPAA Safe Harbor small-cell threshold
SMALL_CELL = 11


# ---------------------------------------------------------------------------
# HIPAA suppression helper
# ---------------------------------------------------------------------------

def _suppress(count: int) -> dict:
    """Return count with suppression flag if below HIPAA threshold."""
    if count < SMALL_CELL:
        return {"count": 0, "suppressed": True}
    return {"count": int(count), "suppressed": False}


def _suppress_distribution(dist: dict, min_cell: int = SMALL_CELL) -> dict:
    """Suppress small categories in a distribution dict, rolling into 'Suppressed (n<11)'."""
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
# Snapshot computation helpers
# ---------------------------------------------------------------------------

def _compute_genetics(diag_df: pd.DataFrame, patient_count: int) -> dict:
    """Compute genetic and mutation profile distributions."""
    stats = {"available": True}

    # Genetic confirmation
    if "dmdgntcf" in diag_df.columns:
        counts = diag_df["dmdgntcf"].dropna().value_counts()
        confirmed = int(counts.get("Yes - Laboratory confirmation", 0) +
                        counts.get("Yes - In a family member", 0))
        stats["genetic_confirmation"] = {
            "distribution": _suppress_distribution(
                {str(k): int(v) for k, v in counts.items()}
            ),
            "confirmed_count": confirmed,
            "confirmed_percentage": round(confirmed / patient_count * 100, 1) if patient_count else 0,
        }

    # Mutation type (dna field)
    if "dna" in diag_df.columns:
        counts = diag_df["dna"].dropna()
        counts = counts[counts.astype(str).str.strip() != ""]
        vc = counts.value_counts()
        stats["mutation_type"] = {
            "distribution": [
                {"label": str(k), **_suppress(int(v))}
                for k, v in vc.items()
            ],
            "total_reported": int(vc.sum()),
        }

    # Exon type
    if "exontype" in diag_df.columns:
        counts = diag_df["exontype"].dropna()
        counts = counts[counts.astype(str).str.strip() != ""]
        vc = counts.value_counts()
        stats["exon_type"] = {
            "distribution": [
                {"label": str(k), **_suppress(int(v))}
                for k, v in vc.items()
            ],
        }

    # Frame type
    if "frametype" in diag_df.columns:
        counts = diag_df["frametype"].dropna()
        counts = counts[counts.astype(str).str.strip() != ""]
        vc = counts.value_counts()
        stats["frame_type"] = {
            "distribution": [
                {"label": str(k), **_suppress(int(v))}
                for k, v in vc.items()
            ],
        }

    # Dystrophin deficiency
    if "dmddysdf" in diag_df.columns:
        counts = diag_df["dmddysdf"].dropna()
        counts = counts[counts.astype(str).str.strip() != ""]
        vc = counts.value_counts()
        stats["dystrophin_deficiency"] = {
            "distribution": [
                {"label": str(k), **_suppress(int(v))}
                for k, v in vc.items()
            ],
        }

    return stats


def _compute_amenability(diag_df: pd.DataFrame, meds_df: pd.DataFrame, total_patients: int) -> dict:
    """Compute exon-skipping amenability and therapy utilization per drug."""

    diag = diag_df.copy()
    diag["fromexon"] = pd.to_numeric(diag["fromexon"], errors="coerce")
    diag["toexon"] = pd.to_numeric(diag["toexon"], errors="coerce")

    has_amenable_mut = diag["dna"].isin(AMENABLE_MUT_TYPES) if "dna" in diag.columns else pd.Series(False, index=diag.index)
    has_exon_data = diag["fromexon"].notna() & diag["toexon"].notna()

    # Build set of patients on each drug from medication records
    def _patients_on_drug(search_terms):
        """Find unique patients whose medication records match search terms."""
        if meds_df is None or meds_df.empty:
            return set()
        mask = pd.Series(False, index=meds_df.index)
        for term in search_terms:
            for col in ["StandardName", "Medications"]:
                if col in meds_df.columns:
                    mask |= meds_df[col].str.contains(term, case=False, na=False)
        return set(meds_df.loc[mask, "FACPATID"].dropna().unique())

    drugs_output = []
    all_amenable_patients = set()
    all_on_therapy_patients = set()

    for drug_name, info in EXON_SKIP_DRUGS.items():
        r = info["range"]

        # Amenable: mutation type + exon overlap
        amenable_mask = has_amenable_mut & has_exon_data & (diag["fromexon"] <= r[1]) & (diag["toexon"] >= r[0])
        amenable_patients = set(diag.loc[amenable_mask, "FACPATID"].dropna().unique())

        # On therapy
        on_therapy_patients = _patients_on_drug(info["search"])

        amenable_not_on = amenable_patients - on_therapy_patients
        total_amenable = len(amenable_patients)

        all_amenable_patients |= amenable_patients
        all_on_therapy_patients |= on_therapy_patients

        on_therapy_count = len(on_therapy_patients)
        amenable_not_on_count = len(amenable_not_on)

        drugs_output.append({
            "drug_name": drug_name,
            "target_exon": info["target_exon"],
            "on_therapy": _suppress(on_therapy_count),
            "amenable_not_on_therapy": _suppress(amenable_not_on_count),
            "total_amenable": _suppress(total_amenable),
            "pct_of_cohort": round(total_amenable / total_patients * 100, 1) if total_patients else 0,
        })

    return {
        "available": True,
        "drugs": drugs_output,
        "amenable_mutation_types": AMENABLE_MUT_TYPES,
        "total_amenable_any_drug": _suppress(len(all_amenable_patients)),
        "total_on_any_exon_skipping": _suppress(len(all_on_therapy_patients)),
        "total_non_amenable": _suppress(total_patients - len(all_amenable_patients)),
    }


def _compute_steroid_stats(enc_df: pd.DataFrame, meds_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute glucocorticoid use from encounters and medication records."""
    stats = {"available": True}

    # --- From encounter glcouse field (per-patient dedup) ---
    if "glcouse" in enc_df.columns and "FACPATID" in enc_df.columns:
        enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
        enc_with_glc = enc[enc["glcouse"].notna() & (enc["glcouse"].astype(str).str.strip() != "")]

        if not enc_with_glc.empty and "encntdt" in enc_with_glc.columns:
            enc_with_glc = enc_with_glc.copy()
            enc_with_glc["encntdt"] = pd.to_datetime(enc_with_glc["encntdt"], errors="coerce")
            enc_sorted = enc_with_glc.dropna(subset=["encntdt"]).sort_values("encntdt")

            # First encounter per patient
            first = enc_sorted.groupby("FACPATID").first().reset_index()
            first_counts = first["glcouse"].value_counts()
            stats["glcouse_first_encounter"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in first_counts.items()}
                ),
                "total_reported": int(first_counts.sum()),
            }

            # Last encounter per patient
            last = enc_sorted.groupby("FACPATID").last().reset_index()
            last_counts = last["glcouse"].value_counts()
            stats["glcouse_last_encounter"] = {
                "distribution": _suppress_distribution(
                    {str(k): int(v) for k, v in last_counts.items()}
                ),
                "total_reported": int(last_counts.sum()),
            }

    # --- From medication records (specific steroid drugs) ---
    if meds_df is not None and not meds_df.empty:
        dmd_meds = meds_df[meds_df["FACPATID"].isin(patient_ids)]
        steroid_breakdown = []
        total_steroid_patients = set()

        for class_name, terms in STEROID_CLASSES.items():
            mask = pd.Series(False, index=dmd_meds.index)
            for term in terms:
                for col in ["StandardName", "Medications"]:
                    if col in dmd_meds.columns:
                        mask |= dmd_meds[col].str.contains(term, case=False, na=False)
            patients_on = set(dmd_meds.loc[mask, "FACPATID"].dropna().unique())
            total_steroid_patients |= patients_on
            steroid_breakdown.append({
                "drug_class": class_name,
                **_suppress(len(patients_on)),
            })

        stats["steroid_medications"] = {
            "total_patients_on_steroids": _suppress(len(total_steroid_patients)),
            "breakdown": steroid_breakdown,
        }

    return stats


def _compute_state_distribution(
    demo_df: pd.DataFrame,
    meds_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    site_locations: list,
    patient_ids: list,
) -> dict:
    """Compute per-state patient counts and therapy utilization."""

    # Build facility → state mapping
    state_map = {}
    for loc in site_locations:
        fid = loc.get("facility_id", "")
        state = loc.get("state", "")
        if fid and state:
            state_map[str(fid)] = state

    # Map each patient to a state
    dmd_demo = demo_df[demo_df["FACPATID"].isin(patient_ids)].copy()
    dmd_demo["state"] = dmd_demo["FACILITY_DISPLAY_ID"].astype(str).map(state_map)

    # Patient → state lookup
    patient_state = dict(zip(dmd_demo["FACPATID"], dmd_demo["state"]))

    # --- Compute which patients are on any exon-skipping therapy ---
    on_therapy_patients = set()
    if meds_df is not None and not meds_df.empty:
        dmd_meds = meds_df[meds_df["FACPATID"].isin(patient_ids)]
        for drug_info in EXON_SKIP_DRUGS.values():
            mask = pd.Series(False, index=dmd_meds.index)
            for term in drug_info["search"]:
                for col in ["StandardName", "Medications"]:
                    if col in dmd_meds.columns:
                        mask |= dmd_meds[col].str.contains(term, case=False, na=False)
            on_therapy_patients |= set(dmd_meds.loc[mask, "FACPATID"].dropna().unique())

    # --- Compute which patients are amenable ---
    diag = diag_df[diag_df["FACPATID"].isin(patient_ids)].copy()
    diag["fromexon"] = pd.to_numeric(diag["fromexon"], errors="coerce")
    diag["toexon"] = pd.to_numeric(diag["toexon"], errors="coerce")
    has_amenable_mut = diag["dna"].isin(AMENABLE_MUT_TYPES) if "dna" in diag.columns else pd.Series(False, index=diag.index)
    has_exon_data = diag["fromexon"].notna() & diag["toexon"].notna()

    amenable_patients = set()
    for drug_info in EXON_SKIP_DRUGS.values():
        r = drug_info["range"]
        amenable_mask = has_amenable_mut & has_exon_data & (diag["fromexon"] <= r[1]) & (diag["toexon"] >= r[0])
        amenable_patients |= set(diag.loc[amenable_mask, "FACPATID"].dropna().unique())

    # --- Aggregate by state ---
    state_data = {}
    for pid in patient_ids:
        state = patient_state.get(pid)
        if not state or pd.isna(state):
            state = "Unknown"
        if state not in state_data:
            state_data[state] = {"total": 0, "on_therapy": 0, "amenable_not_on": 0}
        state_data[state]["total"] += 1
        if pid in on_therapy_patients:
            state_data[state]["on_therapy"] += 1
        elif pid in amenable_patients:
            state_data[state]["amenable_not_on"] += 1

    # HIPAA: suppress states with < SMALL_CELL total patients
    states_list = []
    suppressed_total = {"total": 0, "on_therapy": 0, "amenable_not_on": 0}
    for state, counts in sorted(state_data.items(), key=lambda x: x[1]["total"], reverse=True):
        if state == "Unknown":
            continue
        if counts["total"] < SMALL_CELL:
            suppressed_total["total"] += counts["total"]
            suppressed_total["on_therapy"] += counts["on_therapy"]
            suppressed_total["amenable_not_on"] += counts["amenable_not_on"]
        else:
            states_list.append({
                "state": state,
                "total": int(counts["total"]),
                "on_therapy": int(counts["on_therapy"]),
                "amenable_not_on": int(counts["amenable_not_on"]),
            })

    if suppressed_total["total"] > 0:
        states_list.append({
            "state": "Other States (n<11 each)",
            "total": int(suppressed_total["total"]),
            "on_therapy": int(suppressed_total["on_therapy"]),
            "amenable_not_on": int(suppressed_total["amenable_not_on"]),
        })

    return {
        "available": True,
        "total_states_mapped": len([s for s in state_data if s != "Unknown"]),
        "states": states_list,
    }


def _compute_demographics(demo_df: pd.DataFrame) -> dict:
    """Compute diagnosis age histogram and gender distribution."""
    stats = {"available": True}

    # Age at diagnosis
    diag_df_full = None
    try:
        diag_df_full = pd.read_parquet(Path(__file__).parent.parent / "data" / "diagnosis_maindata.parquet")
    except Exception:
        pass

    if diag_df_full is not None and "dmddgnag" in diag_df_full.columns:
        dmd_diag = diag_df_full[diag_df_full["FACPATID"].isin(demo_df["FACPATID"])]
        dx_ages = pd.to_numeric(dmd_diag["dmddgnag"], errors="coerce").dropna()
        dx_ages = dx_ages[dx_ages >= 0]
        if len(dx_ages) > 0:
            bins = list(range(0, int(dx_ages.max()) + 5, 2))
            if len(bins) < 2:
                bins = [0, 5]
            hist, edges = np.histogram(dx_ages, bins=bins)
            stats["diagnosis_age"] = {
                "min": round(float(dx_ages.min()), 1),
                "max": round(float(dx_ages.max()), 1),
                "median": round(float(dx_ages.median()), 1),
                "mean": round(float(dx_ages.mean()), 1),
                "count": int(len(dx_ages)),
                "histogram": {
                    "bins": [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(hist))],
                    "counts": [int(c) for c in hist],
                },
            }

    # Gender
    if "gender" in demo_df.columns:
        gender_counts = demo_df["gender"].value_counts()
        stats["gender"] = {
            "distribution": {str(k): int(v) for k, v in gender_counts.items()},
        }

    return stats


def _compute_ambulatory(enc_df: pd.DataFrame, patient_ids: list) -> dict:
    """Compute ambulatory status from most recent encounter."""
    if "curramb" not in enc_df.columns:
        return {"available": False}

    dmd_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)]
    amb_data = dmd_enc[dmd_enc["curramb"].notna() & (dmd_enc["curramb"].astype(str).str.strip() != "")]

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
# Functional scores
# ---------------------------------------------------------------------------

# DMD-appropriate age bands (pediatric focus)
_AGE_BANDS = [
    ("<5", 0, 5),
    ("5-12", 5, 13),
    ("13-17", 13, 18),
    ("18-25", 18, 26),
    ("26+", 26, 200),
]


def _latest_per_patient(enc_df: pd.DataFrame, patient_ids: list) -> pd.DataFrame:
    """Get the latest encounter per patient, sorted by encntdt."""
    df = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
    if "encntdt" not in df.columns or df.empty:
        return pd.DataFrame()
    df["encntdt"] = pd.to_datetime(df["encntdt"], errors="coerce")
    df = df.dropna(subset=["encntdt"])
    return df.sort_values("encntdt").groupby("FACPATID").last().reset_index()


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


def _compute_functional_scores(
    enc_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    diag_df: pd.DataFrame,
    patient_ids: list,
) -> dict:
    """Compute DMD functional outcome measures: FVC, timed walk, ambulatory status, loss of ambulation."""
    stats = {"available": True}

    latest = _latest_per_patient(enc_df, patient_ids)
    if latest.empty:
        return {"available": False}

    # --- Enrollment age for stratification ---
    enrol_ages = {}
    if "dob" in demo_df.columns and "enroldt" in demo_df.columns:
        dob = pd.to_datetime(demo_df.set_index("FACPATID")["dob"], errors="coerce")
        enrol = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
        age = ((enrol - dob).dt.days / 365.25).dropna()
        enrol_ages = age.to_dict()

    def _by_age_band(series_with_pid: pd.DataFrame, value_col: str) -> dict:
        """Stratify a score by age band at enrollment."""
        bands = {}
        for label, lo, hi in _AGE_BANDS:
            mask = series_with_pid["FACPATID"].map(
                lambda pid: lo <= enrol_ages.get(pid, -1) < hi
            )
            subset = series_with_pid.loc[mask, value_col].dropna()
            if len(subset) >= SMALL_CELL:
                bands[label] = _score_stats(subset)
            elif len(subset) > 0:
                bands[label] = {"count": 0, "suppressed": True}
        return bands

    # =====================================================================
    # A. FVC % Predicted (primary respiratory measure for DMD)
    # =====================================================================
    if "fvcpctpd" in latest.columns:
        fvc = pd.to_numeric(latest["fvcpctpd"], errors="coerce").dropna()
        fvc = fvc[(fvc > 0) & (fvc <= 200)]  # reasonable range
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

    # =====================================================================
    # B. Time to Walk/Run 10m (motor function for ambulatory DMD)
    # =====================================================================
    if "ttwr10m" in latest.columns:
        ttwr = pd.to_numeric(latest["ttwr10m"], errors="coerce").dropna()
        ttwr = ttwr[(ttwr > 0) & (ttwr <= 200)]  # reasonable range (seconds)
        if len(ttwr) >= SMALL_CELL:
            bins = list(range(0, 70, 5))
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

            # Longitudinal timed walk
            all_enc = enc_df[enc_df["FACPATID"].isin(patient_ids)].copy()
            all_enc["encntdt"] = pd.to_datetime(all_enc["encntdt"], errors="coerce")
            all_enc["ttwr10m"] = pd.to_numeric(all_enc["ttwr10m"], errors="coerce")
            enrol_dt = pd.to_datetime(demo_df.set_index("FACPATID")["enroldt"], errors="coerce")
            all_enc["enroldt"] = all_enc["FACPATID"].map(enrol_dt)
            all_enc["years_since_enrol"] = (
                (all_enc["encntdt"] - all_enc["enroldt"]).dt.days / 365.25
            )
            tw_long = all_enc[["ttwr10m", "years_since_enrol"]].dropna()
            tw_long = tw_long[(tw_long["ttwr10m"] > 0) & (tw_long["ttwr10m"] <= 200)]
            tw_long["year_bucket"] = tw_long["years_since_enrol"].clip(lower=0).astype(int).clip(upper=6)

            longitudinal = []
            for yr in range(7):
                bucket = tw_long[tw_long["year_bucket"] == yr]["ttwr10m"]
                if len(bucket) >= SMALL_CELL:
                    longitudinal.append({
                        "year": yr if yr < 6 else "6+",
                        "median": round(float(bucket.median()), 1),
                        "q1": round(float(bucket.quantile(0.25)), 1),
                        "q3": round(float(bucket.quantile(0.75)), 1),
                        "n": int(len(bucket)),
                    })
            stats["timed_10m_walk"]["longitudinal"] = longitudinal

    # =====================================================================
    # C. Loss of Ambulation (from diagnosis milestones)
    # =====================================================================
    if "amblostmnths" in diag_df.columns:
        dmd_diag = diag_df[diag_df["FACPATID"].isin(patient_ids)]
        loa_months = pd.to_numeric(dmd_diag["amblostmnths"], errors="coerce").dropna()
        loa_months = loa_months[loa_months > 0]
        loa_years = loa_months / 12.0

        if len(loa_years) >= SMALL_CELL:
            bins = list(range(0, int(loa_years.max()) + 5, 2))
            if len(bins) < 2:
                bins = [0, 5]
            stats["loss_of_ambulation"] = {
                "total_with_data": int(len(loa_years)),
                "age_at_loss_years": {
                    **_score_stats(loa_years),
                    "histogram": _score_histogram(loa_years, bins),
                },
            }

    # =====================================================================
    # D. Ambulatory status transitions (from encounter curramb)
    # =====================================================================
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
# Main snapshot generation
# ---------------------------------------------------------------------------

def generate_dmd_snapshot() -> dict:
    """Generate comprehensive DMD deep dive statistics snapshot."""

    print("Loading DMD cohort...")
    dmd_cohort = get_disease_cohort("DMD")

    demo_df = dmd_cohort["demographics"]
    diag_df = dmd_cohort["diagnosis"]
    enc_df = dmd_cohort["encounters"]
    patient_count = dmd_cohort["count"]
    patient_ids = dmd_cohort["patient_ids"]
    facility_info = dmd_cohort.get("facility_info", {})

    print(f"Processing {patient_count} DMD patients...")

    # Load combo_drugs for medication matching
    meds_df = None
    combo_path = Path(__file__).parent.parent / "data" / "combo_drugs.parquet"
    if combo_path.exists():
        print("Loading combo_drugs.parquet...")
        meds_df = pd.read_parquet(combo_path)
        meds_df = meds_df[meds_df["FACPATID"].isin(patient_ids)]
        print(f"  {len(meds_df)} DMD medication records")
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
            "disease": "DMD",
            "description": "Duchenne Muscular Dystrophy Deep Dive Statistics",
        },
        "summary": {
            "total_patients": patient_count,
            "total_facilities": facility_info.get("total_facilities", 0),
        },
        "genetics": _compute_genetics(diag_df, patient_count),
        "therapeutics": _compute_amenability(diag_df, meds_df, patient_count),
        "steroids": _compute_steroid_stats(enc_df, meds_df, patient_ids),
        "state_distribution": _compute_state_distribution(
            demo_df, meds_df, diag_df, site_locations, patient_ids
        ),
        "functional_scores": _compute_functional_scores(enc_df, demo_df, diag_df, patient_ids),
        "demographics": _compute_demographics(demo_df),
        "ambulatory": _compute_ambulatory(enc_df, patient_ids),
        "facilities": _compute_facility_stats(facility_info),
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
    print("DMD DEEP DIVE SNAPSHOT SUMMARY")
    print("=" * 70)

    print(f"\nGenerated: {snapshot['metadata']['generated_timestamp']}")
    print(f"Total DMD Patients: {snapshot['summary']['total_patients']:,}")
    print(f"Total Facilities: {snapshot['summary']['total_facilities']}")

    # Genetics
    gc = snapshot["genetics"].get("genetic_confirmation", {})
    if gc:
        print(f"\nGenetic Confirmation: {gc.get('confirmed_percentage', 0)}% ({gc.get('confirmed_count', 0)} patients)")

    # Therapeutics
    tx = snapshot.get("therapeutics", {})
    if tx.get("available"):
        total_amen = tx.get("total_amenable_any_drug", {})
        total_on = tx.get("total_on_any_exon_skipping", {})
        print(f"\nExon-Skipping Therapeutics:")
        print(f"  Amenable to any drug: {total_amen.get('count', '?')}")
        print(f"  On any exon-skipping: {total_on.get('count', '?')}")
        for d in tx.get("drugs", []):
            on = d["on_therapy"]
            amen = d["total_amenable"]
            flag_on = " (suppressed)" if on.get("suppressed") else ""
            flag_am = " (suppressed)" if amen.get("suppressed") else ""
            print(f"  {d['drug_name']}: on={on['count']}{flag_on}, amenable={amen['count']}{flag_am}")

    # Steroids
    st = snapshot.get("steroids", {})
    if st.get("steroid_medications"):
        sm = st["steroid_medications"]
        print(f"\nSteroid Medications ({sm.get('total_patients_on_steroids', {}).get('count', '?')} patients):")
        for b in sm.get("breakdown", []):
            print(f"  {b['drug_class']}: {b['count']}")

    # States
    sd = snapshot.get("state_distribution", {})
    if sd.get("available"):
        print(f"\nState Distribution ({sd.get('total_states_mapped', 0)} states mapped):")
        for s in sd.get("states", [])[:5]:
            print(f"  {s['state']}: {s['total']} total, {s['on_therapy']} on therapy")

    # Functional scores
    fs = snapshot.get("functional_scores", {})
    if fs.get("available"):
        print("\nFunctional Scores:")
        if "fvc_pct" in fs:
            fvc = fs["fvc_pct"]
            print(f"  FVC % Predicted: median={fvc.get('median')}, n={fvc.get('count')} patients")
            print(f"    Longitudinal: {len(fvc.get('longitudinal', []))} year-buckets")
        if "timed_10m_walk" in fs:
            tw = fs["timed_10m_walk"]
            print(f"  Timed 10m Walk: median={tw.get('median')}s, n={tw.get('count')} patients")
        if "loss_of_ambulation" in fs:
            loa = fs["loss_of_ambulation"]
            age = loa.get("age_at_loss_years", {})
            print(f"  Loss of Ambulation: n={loa.get('total_with_data')}, median age={age.get('median')} years")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate DMD deep dive snapshot")
    parser.add_argument(
        "--output", "-o",
        default="stats/dmd_snapshot.json",
        help="Output file path (default: stats/dmd_snapshot.json)",
    )
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress summary output")

    args = parser.parse_args()

    try:
        snapshot = generate_dmd_snapshot()
        output_path = Path(args.output)
        save_snapshot(snapshot, output_path)

        if not args.quiet:
            print_summary(snapshot)

        print("\nDMD snapshot generation complete!")
        return 0

    except Exception as e:
        print(f"\nError generating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
