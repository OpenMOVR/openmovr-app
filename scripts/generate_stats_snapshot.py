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

    patient_ids = set(base_cohort['patient_ids'])

    def _patients_with(col):
        if col not in enc.columns:
            return 0
        has = enc[col].notna() & (enc[col].astype(str).str.strip() != '')
        return int(enc.loc[has, 'FACPATID'].nunique())

    def _data_points(col):
        if col not in enc.columns:
            return 0
        has = enc[col].notna() & (enc[col].astype(str).str.strip() != '')
        return int(has.sum())

    def _patients_multi(col):
        if col not in enc.columns:
            return 0
        has = enc[col].notna() & (enc[col].astype(str).str.strip() != '')
        pt_counts = enc.loc[has].groupby('FACPATID').size()
        return int((pt_counts > 1).sum())

    # Load additional parquet tables and filter to cohort patients
    data_dir = Path(__file__).parent.parent / 'data'

    def _load_filtered(filename):
        path = data_dir / filename
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        if 'FACPATID' in df.columns:
            df = df[df['FACPATID'].isin(patient_ids)]
        return df

    enc_meds = _load_filtered('encounter_medication_rg.parquet')
    log_meds = _load_filtered('log_medication_rg.parquet')
    enc_devices = _load_filtered('encounter_assistivedevice_rg.parquet')
    log_devices = _load_filtered('log_assistivedevice_rg.parquet')
    enc_hosp = _load_filtered('encounter_hospitalization_rg.parquet')
    log_hosp = _load_filtered('log_hospitalization_rg.parquet')
    log_surgery = _load_filtered('log_surgery_rg.parquet')
    spin_maint = _load_filtered('encounter_spinrazamaintenancedose_rg.parquet')
    enc_pulm_dev = _load_filtered('encounter_pulmonarydevice_rg.parquet')
    log_pulm_dev = _load_filtered('log_pulmonarydevice_rg.parquet')

    # --- Medication drug class breakdown ---
    med_stats = _compute_medication_stats(enc, enc_meds, log_meds)

    # --- Spinraza eCRF ---
    spin_cols = [c for c in enc.columns if any(
        kw in c.lower() for kw in ['spin', 'spiclin', 'sptrt', 'splst']
    ) and c.lower() not in ('respinsf', 'spincurv', 'stdsplst', 'wlksplst')]
    spin_maint_cols = [c for c in spin_maint.columns if c not in (
        'CASE_ID', 'PATIENT_DISPLAY_ID', 'MASTER_PATIENT_ID',
        'SCHEDULED_FORM_NAME', 'FORM_VERSION', 'FACILITY_DISPLAY_ID',
        'FACILITY_NAME', 'CREATED_DT', 'MODIFIED_DT', 'FORM_STATUS',
        'CREATED_BY', 'UPDATED_BY', 'UPLOADED_BY', 'Access Case', 'FACPATID'
    )]
    spinraza_stats = {
        "ecrf_fields": len(spin_cols) + len(spin_maint_cols),
        "maintenance_records": len(spin_maint),
        "patients_with_maintenance": int(spin_maint['FACPATID'].nunique()) if not spin_maint.empty else 0,
        "patients_eap": _patients_with('spiclin'),
    }

    # --- Devices (encounter + log combined) ---
    dev_enc_pts = set(enc_devices['FACPATID'].dropna()) if not enc_devices.empty else set()
    dev_log_pts = set(log_devices['FACPATID'].dropna()) if not log_devices.empty else set()
    pulm_enc_pts = set(enc_pulm_dev['FACPATID'].dropna()) if not enc_pulm_dev.empty else set()
    pulm_log_pts = set(log_pulm_dev['FACPATID'].dropna()) if not log_pulm_dev.empty else set()

    devices_stats = {
        "assistive_patients": len(dev_enc_pts | dev_log_pts),
        "assistive_records": len(enc_devices) + len(log_devices),
        "pulmonary_patients": len(pulm_enc_pts | pulm_log_pts),
        "pulmonary_records": len(enc_pulm_dev) + len(log_pulm_dev),
    }

    # --- Hospitalizations (encounter + log combined) ---
    hosp_enc_pts = set(enc_hosp['FACPATID'].dropna()) if not enc_hosp.empty else set()
    hosp_log_pts = set(log_hosp['FACPATID'].dropna()) if not log_hosp.empty else set()
    hosp_stats = {
        "patients": len(hosp_enc_pts | hosp_log_pts),
        "records": len(enc_hosp) + len(log_hosp),
    }
    # Top reasons from encounter table
    if not enc_hosp.empty and 'hsprsn' in enc_hosp.columns:
        reasons = enc_hosp['hsprsn'].dropna()
        reasons = reasons[reasons.astype(str).str.strip() != '']
        hosp_stats["top_reasons"] = {
            str(k): int(v) for k, v in reasons.value_counts().head(8).items()
        }

    # --- Surgeries (log) ---
    surg_stats = {"patients": 0, "records": 0}
    if not log_surgery.empty:
        surg_stats["patients"] = int(log_surgery['FACPATID'].nunique())
        surg_stats["records"] = len(log_surgery)
        if 'srgtype' in log_surgery.columns:
            stypes = log_surgery['srgtype'].dropna()
            stypes = stypes[stypes.astype(str).str.strip() != '']
            surg_stats["types"] = {
                str(k): int(v) for k, v in stypes.value_counts().items()
                if str(k).strip()
            }

    return {
        "functional_scores": {
            "alsfrs_r": {"patients": _patients_with('alsfrstl'), "data_points": _data_points('alsfrstl'), "patients_longitudinal": _patients_multi('alsfrstl'), "label": "ALSFRS-R"},
            "hfmse": {"patients": _patients_with('hfmsesc'), "data_points": _data_points('hfmsesc'), "patients_longitudinal": _patients_multi('hfmsesc'), "label": "HFMSE"},
            "chop_intend": {"patients": _patients_with('cittlscr'), "data_points": _data_points('cittlscr'), "patients_longitudinal": _patients_multi('cittlscr'), "label": "CHOP-INTEND"},
            "rulm": {"patients": _patients_with('rulmcs'), "data_points": _data_points('rulmcs'), "patients_longitudinal": _patients_multi('rulmcs'), "label": "RULM"},
        },
        "timed_tests": {
            "walk_run_10m": {"patients": _patients_with('ttwr10m'), "data_points": _data_points('ttwr10m')},
            "stair_climb": {"patients": _patients_with('ttcstr'), "data_points": _data_points('ttcstr')},
            "rise_from_supine": {"patients": _patients_with('ttrsupn'), "data_points": _data_points('ttrsupn')},
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
        "medications": med_stats,
        "spinraza": spinraza_stats,
        "mobility": {
            "ambulatory_status": _patients_with('curramb'),
            "wheelchair": _patients_with('whlchr'),
            "assistive_device": _patients_with('asstdvc'),
            "falls": _patients_with('apprxfall'),
        },
        "devices": devices_stats,
        "vital_signs": {
            "height": _patients_with('height'),
            "weight": _patients_with('wgt'),
            "bmi": _patients_with('bmi'),
        },
        "nutrition": {
            "nutritional_supplementation": _patients_with('nutrthrp'),
            "feeding_method": _patients_with('feedmthd'),
            "feeding_tube": _patients_with('feedtube'),
        },
        "orthopedic": {
            "scoliosis": _patients_with('scliosis'),
            "spinal_xray": _patients_with('xrybtwvt'),
        },
        "hospitalizations": hosp_stats,
        "surgeries": surg_stats,
        "care": {
            "multidisciplinary_plan": _patients_with('mltcrpl'),
            "specialists_seen": _patients_with('ptsn'),
            "specialists_referred": _patients_with('rfrto'),
        },
    }


def _compute_medication_stats(enc, enc_meds, log_meds):
    """Compute medication breakdown by drug class from repeat group tables."""
    # Combine encounter + log medication tables
    all_meds = pd.concat([enc_meds, log_meds], ignore_index=True) if not log_meds.empty else enc_meds

    if all_meds.empty:
        return {}

    total_pts = int(all_meds['FACPATID'].nunique()) if 'FACPATID' in all_meds.columns else 0

    def _class_stats(names, label):
        mask = all_meds['medname'].isin(names)
        recs = int(mask.sum())
        pts = int(all_meds.loc[mask, 'FACPATID'].nunique()) if recs > 0 else 0
        result = {"patients": pts, "records": recs, "label": label}
        # Per-drug breakdown
        drugs = {}
        for name in names:
            drug_mask = all_meds['medname'] == name
            drug_recs = int(drug_mask.sum())
            if drug_recs > 0:
                drug_pts = int(all_meds.loc[drug_mask, 'FACPATID'].nunique())
                drugs[name] = {"patients": drug_pts, "records": drug_recs}
        result["drugs"] = drugs
        return result

    # Also search medoth and medname1-Description for gene therapies
    def _search_meds(keywords):
        pts = set()
        recs = 0
        for kw in keywords:
            for col in ['medname', 'medoth']:
                if col in all_meds.columns:
                    vals = all_meds[col].fillna('').astype(str)
                    matches = vals.str.lower().str.contains(kw.lower(), na=False)
                    recs += int(matches.sum())
                    pts |= set(all_meds.loc[matches, 'FACPATID'].dropna())
        return {"patients": len(pts), "records": recs}

    # Glucocorticoid encounter field (glcouse) breakdown
    glc_enc = {}
    if 'glcouse' in enc.columns:
        glc_vals = enc['glcouse'].dropna()
        glc_vals = glc_vals[glc_vals.astype(str).str.strip() != '']
        glc_enc["data_points"] = len(glc_vals)
        glc_enc["patients"] = int(enc.loc[glc_vals.index, 'FACPATID'].nunique())
        glc_enc["values"] = {
            str(k): int(v) for k, v in glc_vals.value_counts().items()
        }

    return {
        "total_records": len(all_meds),
        "total_patients": total_pts,
        "glucocorticoids_med_table": _class_stats(
            ['Prednisone', 'Deflazacort', 'Emflaza'], "Glucocorticoids (Med Table)"
        ),
        "glucocorticoid_encounter": glc_enc,
        "disease_modifying_als": _class_stats(
            ['Riluzole', 'Riluzole, oral suspension (TiGlutik)', 'Radicava', 'Relyvrio'],
            "Disease-Modifying (ALS)"
        ),
        "gene_therapy_antisense": _class_stats(
            ['Spinraza', 'Zolgensma', 'Exondys-51', 'Evrysdi'],
            "Gene Therapy & Antisense"
        ),
        "exon_skipping_search": _search_meds(
            ['exondys', 'vyondys', 'viltepso', 'casimersen', 'amondys',
             'eteplirsen', 'golodirsen', 'gene therapy', 'gene skipping']
        ),
        "sma_treatments_search": _search_meds(
            ['spinraza', 'nusinersen', 'zolgensma', 'onasemnogene',
             'risdiplam', 'evrysdi']
        ),
        "cardiac_meds": _class_stats(
            ['Lisinopril', 'Losartan', 'Enalapril', 'Carvedilol', 'Metoprolol',
             'Aldactone (Spironolactone)', 'Digoxin', 'Benazepril', 'Ramipril'],
            "Cardiac Medications"
        ),
        "psych_neuro": _class_stats(
            ['Zyprexa', 'Nuedexta', 'Gabapentin', 'Zoloft (SSRI)',
             'Lorazepam', 'Amitriptyline', 'Baclofen'],
            "Psych / Neuro"
        ),
        "respiratory_meds": _class_stats(
            ['Albuterol', 'Albuterol inhaled', 'Albuterol oral', 'Budesonide'],
            "Respiratory"
        ),
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
