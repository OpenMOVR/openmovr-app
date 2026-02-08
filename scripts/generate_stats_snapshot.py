#!/usr/bin/env python3
"""
Generate Database Statistics Snapshot

Creates a JSON file with commonly-requested database statistics that can be
quickly queried without running full analyses.

Usage:
    python scripts/generate_stats_snapshot.py
    python scripts/generate_stats_snapshot.py --output stats/current_stats.json

This creates a snapshot with:
- Total patient counts (MOVR vs USNDR)
- Disease-specific patient counts
- Enrollment statistics
- Facility distribution
- Data completeness metrics
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.cohorts import get_base_cohort, get_disease_counts
from src.config import get_config


def generate_snapshot(include_usndr: bool = False) -> dict:
    """
    Generate comprehensive statistics snapshot.

    Args:
        include_usndr: Whether to include USNDR legacy patients in counts

    Returns:
        Dictionary with all snapshot statistics
    """

    print("🔄 Generating database statistics snapshot...")

    # Load base cohort (MOVR only by default)
    base_cohort = get_base_cohort(include_usndr=include_usndr)

    # Get disease distribution
    disease_counts_df = get_disease_counts(base_cohort)

    # Build snapshot
    snapshot = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "generated_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "includes_usndr": include_usndr,
            "cohort_type": "MOVR + USNDR" if include_usndr else "MOVR only"
        },

        "enrollment": {
            "total_patients": base_cohort['count'],
            "description": "Patients with validated enrollment (Demographics + Diagnosis + Encounters)",
            "validation_stats": base_cohort['stats']['validation_stats'],
            "movr_stats": base_cohort['stats']['movr_stats'] if not include_usndr else None
        },

        "disease_distribution": {
            "total_with_disease_data": int(disease_counts_df['patient_count'].sum()) if not disease_counts_df.empty else 0,
            "diseases": disease_counts_df.to_dict('records') if not disease_counts_df.empty else [],
            "disease_summary": {
                row['disease']: {
                    "count": int(row['patient_count']),
                    "percentage": float(row['percentage'])
                }
                for _, row in disease_counts_df.iterrows()
            } if not disease_counts_df.empty else {}
        },

        "facilities": {
            "total_facilities": base_cohort['facility_info'].get('total_facilities', 0),
            "top_10_facilities": base_cohort['facility_info'].get('facilities', [])[:10] if base_cohort['facility_info'] else [],
            "all_facilities": base_cohort['facility_info'].get('facilities', []) if base_cohort['facility_info'] else []
        },

        "disease_type_distribution": {
            "dstype_counts": base_cohort.get('dstype_counts', {}),
            "description": "Disease types as coded in dstype field"
        },

        "data_availability": {
            "demographics_records": len(base_cohort['demographics']),
            "diagnosis_records": len(base_cohort['diagnosis']),
            "encounter_records": len(base_cohort['encounters']),
            "medication_records": len(base_cohort['medications'])
        },

        "longitudinal": _compute_longitudinal_stats(base_cohort),

        "clinical_availability": _compute_clinical_availability(base_cohort),
    }

    return snapshot


def _compute_longitudinal_stats(base_cohort: dict) -> dict:
    """Compute encounter longitudinality statistics."""
    enc = base_cohort['encounters']
    if enc.empty or 'FACPATID' not in enc.columns:
        return {}

    enc_per_pt = enc.groupby('FACPATID').size()

    stats = {
        "total_encounters": len(enc),
        "patients_with_encounters": int(enc['FACPATID'].nunique()),
        "mean_encounters_per_patient": round(float(enc_per_pt.mean()), 1),
        "median_encounters_per_patient": int(enc_per_pt.median()),
        "patients_3plus_encounters": int((enc_per_pt >= 3).sum()),
        "patients_5plus_encounters": int((enc_per_pt >= 5).sum()),
    }

    # By disease
    if 'dstype' in enc.columns:
        by_disease = {}
        for ds in sorted(enc['dstype'].dropna().unique()):
            ds_enc = enc[enc['dstype'] == ds]
            ds_per_pt = ds_enc.groupby('FACPATID').size()
            by_disease[ds] = {
                "patients": int(ds_enc['FACPATID'].nunique()),
                "encounters": int(len(ds_enc)),
                "mean_per_patient": round(float(ds_per_pt.mean()), 1),
                "patients_3plus": int((ds_per_pt >= 3).sum()),
            }
        stats["by_disease"] = by_disease

    return stats


def _compute_clinical_availability(base_cohort: dict) -> dict:
    """Compute clinical domain data availability (patients with data)."""
    enc = base_cohort['encounters']
    if enc.empty or 'FACPATID' not in enc.columns:
        return {}

    def _patients_with(col):
        if col not in enc.columns:
            return 0
        has = enc[col].notna() & (enc[col].astype(str).str.strip() != '')
        return int(enc.loc[has, 'FACPATID'].nunique())

    def _patients_multi(col):
        if col not in enc.columns:
            return 0
        has = enc[col].notna() & (enc[col].astype(str).str.strip() != '')
        pt_counts = enc.loc[has].groupby('FACPATID').size()
        return int((pt_counts > 1).sum())

    return {
        "functional_scores": {
            "alsfrs_r": {"patients": _patients_with('alsfrstl'), "patients_longitudinal": _patients_multi('alsfrstl'), "label": "ALSFRS-R"},
            "hfmse": {"patients": _patients_with('hfmsesc'), "patients_longitudinal": _patients_multi('hfmsesc'), "label": "HFMSE"},
            "chop_intend": {"patients": _patients_with('cittlscr'), "patients_longitudinal": _patients_multi('cittlscr'), "label": "CHOP-INTEND"},
            "rulm": {"patients": _patients_with('rulmcs'), "patients_longitudinal": _patients_multi('rulmcs'), "label": "RULM"},
        },
        "pulmonary": {
            "pft_performed": _patients_with('pftest'),
            "fvc": _patients_with('fvcrslt'),
            "fev1": _patients_with('fev1rslt'),
        },
        "cardiac": {
            "ecg": _patients_with('ecgrslt'),
            "echo": _patients_with('echorslt'),
            "cardiomyopathy": _patients_with('crdmyo'),
        },
        "medications": {
            "glucocorticoid": _patients_with('glcouse'),
            "ambulatory_status": _patients_with('curramb'),
        },
        "timed_tests": {
            "walk_run_10m": _patients_with('ttwr10m'),
            "stair_climb": _patients_with('ttcstr'),
            "rise_from_supine": _patients_with('ttrsupn'),
        },
        "care": {
            "multidisciplinary_plan": _patients_with('mltcrpl'),
            "hospitalizations": _patients_with('hospbtwvt'),
        },
    }


def save_snapshot(snapshot: dict, output_path: Path):
    """Save snapshot to JSON file with pretty formatting."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"\n✅ Snapshot saved to: {output_path}")

    # Also create a timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.parent / f"{output_path.stem}_{timestamp}.json"
    with open(backup_path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)

    print(f"📁 Backup saved to: {backup_path}")


def print_summary(snapshot: dict):
    """Print human-readable summary of the snapshot."""

    print("\n" + "="*70)
    print("DATABASE STATISTICS SNAPSHOT SUMMARY")
    print("="*70)

    print(f"\n📅 Generated: {snapshot['metadata']['generated_timestamp']}")
    print(f"📊 Cohort Type: {snapshot['metadata']['cohort_type']}")

    print(f"\n👥 ENROLLMENT")
    print(f"   Total Patients: {snapshot['enrollment']['total_patients']:,}")

    print(f"\n🧬 DISEASE DISTRIBUTION")
    for disease, stats in sorted(
        snapshot['disease_distribution']['disease_summary'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    ):
        print(f"   {disease}: {stats['count']:,} patients ({stats['percentage']:.1f}%)")

    print(f"\n🏥 FACILITIES")
    print(f"   Total Facilities: {snapshot['facilities']['total_facilities']}")
    print(f"   Top 5 Facilities:")
    for facility in snapshot['facilities']['top_10_facilities'][:5]:
        print(f"      {facility['FACILITY_DISPLAY_ID']}: {facility['FACILITY_NAME']} ({facility['patient_count']:,} patients)")

    print(f"\n📋 DATA AVAILABILITY")
    print(f"   Demographics: {snapshot['data_availability']['demographics_records']:,} records")
    print(f"   Diagnosis: {snapshot['data_availability']['diagnosis_records']:,} records")
    print(f"   Encounters: {snapshot['data_availability']['encounter_records']:,} records")
    print(f"   Medications: {snapshot['data_availability']['medication_records']:,} records")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Generate database statistics snapshot")
    parser.add_argument("--output", "-o",
                       default="stats/database_snapshot.json",
                       help="Output file path (default: stats/database_snapshot.json)")
    parser.add_argument("--include-usndr", action="store_true",
                       help="Include USNDR legacy patients in counts")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress summary output")

    args = parser.parse_args()

    try:
        # Generate snapshot
        snapshot = generate_snapshot(include_usndr=args.include_usndr)

        # Save to file
        output_path = Path(args.output)
        save_snapshot(snapshot, output_path)

        # Print summary unless quiet mode
        if not args.quiet:
            print_summary(snapshot)

        print("\n✅ Snapshot generation complete!")
        print(f"\n💡 TIP: Claude Code can now instantly answer questions like:")
        print("   - 'How many DMD patients do we have?'")
        print("   - 'What's the total enrollment count?'")
        print("   - 'Show me the facility distribution'")
        print(f"\n   Just ask, and I'll read: {output_path}")

        return 0

    except Exception as e:
        print(f"\n❌ Error generating snapshot: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
