#!/usr/bin/env python3
"""
Generate LGMD Overview Snapshot

Creates a JSON file with pre-computed LGMD statistics for the OpenMOVR App.
This enables the LGMD Overview page to work without loading parquet files directly.

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


def generate_lgmd_snapshot() -> dict:
    """
    Generate comprehensive LGMD statistics snapshot.

    Returns:
        Dictionary with all LGMD snapshot statistics
    """

    print("Loading LGMD cohort...")
    lgmd_cohort = get_disease_cohort('LGMD')

    demo_df = lgmd_cohort['demographics']
    diag_df = lgmd_cohort['diagnosis']
    enc_df = lgmd_cohort['encounters']
    patient_count = lgmd_cohort['count']
    facility_info = lgmd_cohort.get('facility_info', {})

    print(f"Processing {patient_count} LGMD patients...")

    snapshot = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "disease": "LGMD",
            "description": "Limb-Girdle Muscular Dystrophy Overview Statistics"
        },

        "summary": {
            "total_patients": patient_count,
            "total_facilities": facility_info.get('total_facilities', 0),
            "patient_ids": lgmd_cohort['patient_ids']  # For filtering
        },

        "subtypes": _compute_subtype_stats(diag_df, patient_count),
        "demographics": _compute_demographic_stats(demo_df),
        "diagnosis": _compute_diagnosis_stats(diag_df, patient_count),
        "clinical": _compute_clinical_stats(diag_df, enc_df),
        "ambulatory": _compute_ambulatory_stats(enc_df, diag_df),
        "facilities": _compute_facility_stats(facility_info),
    }

    return snapshot


def _compute_subtype_stats(diag_df: pd.DataFrame, total_patients: int) -> dict:
    """Compute subtype distribution statistics."""

    if 'lgtype' not in diag_df.columns:
        return {"available": False}

    subtype_counts = diag_df['lgtype'].value_counts()
    subtype_counts = subtype_counts[subtype_counts.index.astype(str).str.strip() != '']

    subtypes_list = []
    for subtype, count in subtype_counts.items():
        # Categorize by LGMD type
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

    # Current age calculation
    if 'dob' in demo_df.columns:
        dob_dates = pd.to_datetime(demo_df['dob'], errors='coerce')
        current_ages = (datetime.now() - dob_dates).dt.days / 365.25
        current_ages = current_ages.dropna()

        if len(current_ages) > 0:
            # Histogram bins for age distribution
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

    # Age at enrollment
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

    # Gender distribution
    if 'gender' in demo_df.columns:
        gender_counts = demo_df['gender'].value_counts()
        stats["gender"] = {
            "distribution": {str(k): int(v) for k, v in gender_counts.items()},
            "total": int(gender_counts.sum())
        }

    # Ethnicity distribution
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

    # Age at diagnosis
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

    # Age at symptom onset
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

    # Genetic confirmation status
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

    # Muscle biopsy
    if 'lgmscbp' in diag_df.columns:
        biopsy_counts = diag_df['lgmscbp'].fillna('Unknown').value_counts()
        stats["muscle_biopsy"] = {
            "distribution": {str(k): int(v) for k, v in biopsy_counts.items()}
        }

    # Family history
    if 'lgfam' in diag_df.columns:
        family_counts = diag_df['lgfam'].fillna('Unknown').value_counts()
        stats["family_history"] = {
            "distribution": {str(k): int(v) for k, v in family_counts.items()}
        }

    # First symptoms
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

    # Get most recent ambulatory status per patient
    amb_status = enc_df[enc_df['curramb'].notna() & (enc_df['curramb'] != '')]

    if amb_status.empty:
        return {"available": False}

    # Get latest encounter per patient
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

    # Ambulatory status by top subtypes
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

            # Top 5 subtypes with ambulatory data
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


def save_snapshot(snapshot: dict, output_path: Path):
    """Save snapshot to JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\nSnapshot saved to: {output_path}")


def print_summary(snapshot: dict):
    """Print human-readable summary."""

    print("\n" + "="*70)
    print("LGMD OVERVIEW SNAPSHOT SUMMARY")
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

    if snapshot['facilities'].get('available'):
        print(f"\nTop 3 Facilities:")
        for f in snapshot['facilities']['facilities'][:3]:
            print(f"   {f['name']}: {f['patients']} patients")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate LGMD overview snapshot")
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
