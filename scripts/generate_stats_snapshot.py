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
            # Strip facility names from public snapshot — names only
            # available via provisioned access + live Excel metadata
            "all_facilities": [
                {
                    "FACILITY_DISPLAY_ID": f.get("FACILITY_DISPLAY_ID"),
                    "patient_count": f.get("patient_count"),
                }
                for f in (base_cohort['facility_info'].get('facilities', []) if base_cohort['facility_info'] else [])
            ],
            "site_locations": _build_site_geography(base_cohort),
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

        "disease_profiles": _compute_disease_profiles(base_cohort),
    }

    return snapshot


def _compute_disease_profiles(base_cohort: dict) -> dict:
    """Compute per-disease demographic overview and diagnosis profile snapshots."""
    demo = base_cohort['demographics']
    diag = base_cohort['diagnosis']
    if demo.empty or 'dstype' not in demo.columns:
        return {}

    def _value_counts(series, top_n=15):
        """Return value counts as list of {label, count} dicts."""
        vals = series.dropna()
        vals = vals[vals.astype(str).str.strip() != '']
        if vals.empty:
            return []
        vc = vals.value_counts().head(top_n)
        return [{"label": str(k), "count": int(v)} for k, v in vc.items()]

    def _age_histogram(dob_series, enrol_series):
        """Compute age-at-enrollment histogram bins."""
        try:
            dob = pd.to_datetime(dob_series, format='mixed', dayfirst=False)
            enrol = pd.to_datetime(enrol_series, format='mixed', dayfirst=False)
            ages = ((enrol - dob).dt.days / 365.25).dropna()
            ages = ages[(ages >= 0) & (ages <= 110)]
            if ages.empty:
                return []
            bins = [0, 5, 10, 18, 30, 40, 50, 60, 70, 80, 110]
            labels = ['0-4', '5-9', '10-17', '18-29', '30-39', '40-49',
                      '50-59', '60-69', '70-79', '80+']
            cut = pd.cut(ages, bins=bins, labels=labels, include_lowest=True)
            vc = cut.value_counts().sort_index()
            return [{"label": str(k), "count": int(v)} for k, v in vc.items() if v > 0]
        except Exception:
            return []

    # Disease-specific diagnosis fields to snapshot
    _DIAGNOSIS_FIELDS = {
        'ALS': [
            {'field': 'alselesc', 'label': 'El Escorial Criteria'},
            {'field': 'genemut', 'label': 'Gene Mutation'},
            {'field': 'alsdgnag', 'label': 'Age at Diagnosis', 'numeric': True},
        ],
        'DMD': [
            {'field': 'dmdgntcf', 'label': 'Genetic Confirmation'},
            {'field': 'dmdfam', 'label': 'Family History'},
        ],
        'BMD': [
            {'field': 'bmdgntcf', 'label': 'Genetic Confirmation'},
            {'field': 'bmdfam', 'label': 'Family History'},
        ],
        'SMA': [
            {'field': 'smaclass', 'label': 'SMA Classification'},
            {'field': 'smadgcnf', 'label': 'Genetic Confirmation'},
        ],
        'LGMD': [
            {'field': 'lgtype', 'label': 'LGMD Subtype', 'top_n': 20},
            {'field': 'lggntcf', 'label': 'Genetic Confirmation'},
            {'field': 'lgmscbp', 'label': 'Muscle Biopsy'},
        ],
        'FSHD': [
            {'field': 'fshdel', 'label': '4q35 Deletion'},
        ],
        'Pompe': [],
    }

    profiles = {}
    all_diseases = sorted(demo['dstype'].dropna().unique())

    for disease in all_diseases:
        ds_pts = set(demo.loc[demo['dstype'] == disease, 'FACPATID'])
        ds_demo = demo[demo['FACPATID'].isin(ds_pts)]
        ds_diag = diag[diag['FACPATID'].isin(ds_pts)] if not diag.empty else pd.DataFrame()

        profile = {
            "patient_count": len(ds_pts),
            "demographics": {
                "gender": _value_counts(ds_demo['gender']) if 'gender' in ds_demo.columns else [],
                "ethnicity": _value_counts(ds_demo['ethnic']) if 'ethnic' in ds_demo.columns else [],
                "age_at_enrollment": _age_histogram(
                    ds_demo.get('dob', pd.Series(dtype='object')),
                    ds_demo.get('enroldt', pd.Series(dtype='object')),
                ),
            },
            "diagnosis": [],
        }

        # Disease-specific diagnosis distributions
        for fdef in _DIAGNOSIS_FIELDS.get(disease, []):
            field = fdef['field']
            if ds_diag.empty or field not in ds_diag.columns:
                continue
            top_n = fdef.get('top_n', 15)
            if fdef.get('numeric'):
                # For numeric fields, compute summary stats
                vals = pd.to_numeric(ds_diag[field], errors='coerce').dropna()
                if not vals.empty:
                    profile["diagnosis"].append({
                        "field": field,
                        "label": fdef['label'],
                        "type": "numeric",
                        "n": int(len(vals)),
                        "mean": round(float(vals.mean()), 1),
                        "median": round(float(vals.median()), 1),
                        "min": round(float(vals.min()), 1),
                        "max": round(float(vals.max()), 1),
                    })
            else:
                counts = _value_counts(ds_diag[field], top_n=top_n)
                if counts:
                    profile["diagnosis"].append({
                        "field": field,
                        "label": fdef['label'],
                        "type": "categorical",
                        "values": counts,
                    })

        profiles[disease] = profile

    print(f"   Disease profiles: {len(profiles)} diseases computed")
    return profiles


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


def _build_site_geography(base_cohort: dict) -> list:
    """Build site locations from actual encounter data + metadata for geography.

    Patient counts and per-disease breakdowns come from the data itself
    (encounters grouped by FACILITY_DISPLAY_ID × dstype).  Location metadata
    (city, state, ZIP → lat/lon) comes from the MOVR Sites tracker Excel.
    Sites that appear in the data but not the metadata are included with
    null coordinates.  No facility names are stored.
    """
    enc = base_cohort['encounters']
    if enc.empty or 'FACILITY_DISPLAY_ID' not in enc.columns:
        return []

    # --- Source of truth: patient counts from data ---
    fac_total = (
        enc.groupby('FACILITY_DISPLAY_ID')['FACPATID']
        .nunique()
        .reset_index()
        .rename(columns={'FACPATID': 'patient_count'})
    )
    fac_total['facility_id'] = fac_total['FACILITY_DISPLAY_ID'].astype(str)

    # Per-disease counts per facility
    disease_counts = {}
    if 'dstype' in enc.columns:
        fac_ds = (
            enc.groupby(['FACILITY_DISPLAY_ID', 'dstype'])['FACPATID']
            .nunique()
            .reset_index()
            .rename(columns={'FACPATID': 'patients'})
        )
        for _, row in fac_ds.iterrows():
            fid = str(row['FACILITY_DISPLAY_ID'])
            ds = str(row['dstype'])
            disease_counts.setdefault(fid, {})[ds] = int(row['patients'])

    all_diseases = sorted(enc['dstype'].dropna().unique()) if 'dstype' in enc.columns else []

    # --- Location metadata from Excel (optional enrichment) ---
    excel_path = Path(__file__).parent.parent / 'data' / 'MOVR Sites - Tracker Information - EK.xlsx'
    meta_lookup = {}
    if excel_path.exists():
        sites_meta = pd.read_excel(excel_path)
        try:
            import pgeocode
            nomi = pgeocode.Nominatim('us')
        except ImportError:
            nomi = None
            print("⚠️  pgeocode not installed – coordinates unavailable")

        for _, mrow in sites_meta.iterrows():
            mid = str(int(mrow['FACILITY_DISPLAY_ID']))
            zipcode = str(int(mrow['Zip'])).zfill(5)

            lat, lon = None, None
            if nomi is not None:
                geo = nomi.query_postal_code(zipcode)
                lat = float(geo.latitude) if pd.notna(geo.latitude) else None
                lon = float(geo.longitude) if pd.notna(geo.longitude) else None

            state = str(mrow['State']).strip()
            meta_lookup[mid] = {
                "city": str(mrow['City']).strip(),
                "state": state,
                "region": str(mrow.get('Region', '')).strip(),
                "site_type": str(mrow.get('Site Type', '')).strip(),
                "lat": lat,
                "lon": lon,
                "continental": state not in ('HI', 'AK', 'PR', 'GU', 'VI', 'AS', 'MP'),
            }
    else:
        print("⚠️  Site tracker Excel not found – locations unavailable")

    # --- Merge data counts with metadata ---
    locations = []
    for _, row in fac_total.iterrows():
        fid = row['facility_id']
        meta = meta_lookup.get(fid, {})
        ds_counts = disease_counts.get(fid, {})

        locations.append({
            "facility_id": fid,
            "city": meta.get("city", ""),
            "state": meta.get("state", ""),
            "region": meta.get("region", ""),
            "site_type": meta.get("site_type", ""),
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "continental": meta.get("continental", True),
            "patient_count": int(row['patient_count']),
            "by_disease": ds_counts,
        })

    locations.sort(key=lambda x: x['patient_count'], reverse=True)

    mapped = sum(1 for loc in locations if loc['lat'] is not None)
    in_data_only = sum(1 for loc in locations if not meta_lookup.get(loc['facility_id']))
    print(f"📍 Site geography: {len(locations)} sites from data, "
          f"{mapped} geocoded, {in_data_only} data-only (no metadata)")
    if all_diseases:
        print(f"   Disease types: {', '.join(all_diseases)}")
    return locations


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
        "clinical_trials": _compute_trial_stats(enc),
        "care": {
            "multidisciplinary_plan": _patients_with('mltcrpl'),
            "specialists_seen": _patients_with('ptsn'),
            "specialists_referred": _patients_with('rfrto'),
        },
    }


def _compute_trial_stats(enc):
    """Compute clinical trial participation from clntrlyn encounter field."""
    if 'clntrlyn' not in enc.columns:
        return {}
    vals = enc['clntrlyn'].dropna()
    vals = vals[vals.astype(str).str.strip() != '']
    patients = int(enc.loc[vals.index, 'FACPATID'].nunique())
    breakdown = {str(k): int(v) for k, v in vals.value_counts().items()}
    # Count unique patients per category
    patient_breakdown = {}
    for cat in vals.unique():
        mask = enc['clntrlyn'] == cat
        patient_breakdown[str(cat)] = int(enc.loc[mask, 'FACPATID'].nunique())
    return {
        "patients": patients,
        "data_points": len(vals),
        "breakdown": breakdown,
        "patient_breakdown": patient_breakdown,
    }


def _compute_medication_stats(enc, enc_meds, log_meds):
    """Compute medication stats from combo_drugs (preferred) + repeat group tables."""
    import yaml

    # Combine encounter + log medication tables (for overall totals)
    all_meds = pd.concat([enc_meds, log_meds], ignore_index=True) if not log_meds.empty else enc_meds
    total_pts = int(all_meds['FACPATID'].nunique()) if not all_meds.empty and 'FACPATID' in all_meds.columns else 0

    # --- Load combo_drugs (preferred source per gene therapy config) ---
    data_dir = Path(__file__).parent.parent / 'data'
    combo_path = data_dir / 'combo_drugs.parquet'
    patient_ids = set(enc['FACPATID'].dropna().unique()) if 'FACPATID' in enc.columns else set()

    combo = pd.DataFrame()
    if combo_path.exists():
        combo = pd.read_parquet(combo_path)
        if 'FACPATID' in combo.columns:
            combo = combo[combo['FACPATID'].isin(patient_ids)]
        print(f"   combo_drugs: {len(combo):,} records, {combo['FACPATID'].nunique():,} cohort patients")

    # --- Load gene therapy config ---
    config_path = data_dir.parent / 'config' / 'gene_therapy_config.yaml'
    gt_config = {}
    if config_path.exists():
        with open(config_path) as f:
            gt_config = yaml.safe_load(f)

    # --- Gene therapy by disease from combo_drugs ---
    # Merge brand + generic names into deduplicated treatment families
    # so patients aren't double-counted (Spinraza = nusinersen, etc.)
    _TREATMENT_FAMILIES = {
        'SMA': [
            {'label': 'Spinraza (nusinersen)', 'category': 'Antisense',
             'search': ['Spinraza', 'nusinersen']},
            {'label': 'Zolgensma (onasemnogene)', 'category': 'Gene Therapy',
             'search': ['Zolgensma', 'onasemnogene']},
            {'label': 'Evrysdi (risdiplam)', 'category': 'SMN2 Modifier',
             'search': ['Evrysdi', 'risdiplam']},
        ],
        'DMD': [
            {'label': 'Exondys 51 (eteplirsen)', 'category': 'Exon Skipping',
             'search': ['Exondys', 'eteplirsen']},
            {'label': 'Vyondys 53 (golodirsen)', 'category': 'Exon Skipping',
             'search': ['Vyondys', 'golodirsen']},
            {'label': 'Amondys 45 (casimersen)', 'category': 'Exon Skipping',
             'search': ['Amondys', 'casimersen']},
            {'label': 'Viltepso (viltolarsen)', 'category': 'Exon Skipping',
             'search': ['Viltepso', 'viltolarsen']},
        ],
        'ALS': [
            {'label': 'Qalsody (tofersen)', 'category': 'Antisense',
             'search': ['Qalsody', 'tofersen']},
        ],
        'Pompe': [
            {'label': 'Lumizyme (alglucosidase alfa)', 'category': 'ERT',
             'search': ['Lumizyme', 'alglucosidase alfa']},
            {'label': 'Nexviazyme (avalglucosidase alfa)', 'category': 'ERT',
             'search': ['Nexviazyme', 'avalglucosidase alfa']},
        ],
    }

    gene_therapy_by_disease = {}
    if not combo.empty:
        for disease, families in _TREATMENT_FAMILIES.items():
            disease_patient_ids = set()
            treatments = []

            for fam in families:
                fam_pts = set()
                fam_recs = 0
                for term in fam['search']:
                    mask = combo['StandardName'].str.contains(
                        term, case=False, na=False
                    )
                    recs = int(mask.sum())
                    if recs > 0:
                        fam_pts |= set(combo.loc[mask, 'FACPATID'].dropna())
                        fam_recs += recs

                if fam_pts:
                    treatments.append({
                        "label": fam['label'],
                        "category": fam['category'],
                        "patients": len(fam_pts),
                        "records": fam_recs,
                    })
                    disease_patient_ids |= fam_pts

            if treatments:
                gene_therapy_by_disease[disease] = {
                    "total_patients": len(disease_patient_ids),
                    "treatments": treatments,
                }

        if gene_therapy_by_disease:
            parts = [f"{d}={v['total_patients']} pts" for d, v in gene_therapy_by_disease.items()]
            print(f"   Gene therapy by disease: {', '.join(parts)}")

    # --- Top medications from combo_drugs ---
    top_medications = []
    if not combo.empty and 'StandardName' in combo.columns:
        top = (
            combo.groupby('StandardName')['FACPATID']
            .nunique()
            .sort_values(ascending=False)
            .head(25)
        )
        for name, pts in top.items():
            recs = int((combo['StandardName'] == name).sum())
            top_medications.append({
                "name": str(name),
                "patients": int(pts),
                "records": recs,
            })

    # --- Drug class stats from combo_drugs (or fallback to enc/log meds) ---
    def _class_stats_combo(keywords, label):
        """Match drug classes in combo_drugs by keyword search on StandardName."""
        if combo.empty:
            return {"patients": 0, "records": 0, "label": label, "drugs": {}}

        all_pts = set()
        all_recs = 0
        drugs = {}
        for kw in keywords:
            mask = combo['StandardName'].str.contains(kw, case=False, na=False)
            recs = int(mask.sum())
            if recs > 0:
                pts = set(combo.loc[mask, 'FACPATID'].dropna())
                drugs[kw] = {"patients": len(pts), "records": recs}
                all_pts |= pts
                all_recs += recs
        return {"patients": len(all_pts), "records": all_recs, "label": label, "drugs": drugs}

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

    combo_stats = {
        "total_records": len(combo) if not combo.empty else len(all_meds),
        "total_patients": int(combo['FACPATID'].nunique()) if not combo.empty else total_pts,
        "source": "combo_drugs" if not combo.empty else "encounter_log_meds",
        "gene_therapy_by_disease": gene_therapy_by_disease,
        "top_medications": top_medications,
        "glucocorticoid_encounter": glc_enc,
        "disease_modifying_als": _class_stats_combo(
            ['riluzole', 'Radicava', 'edaravone', 'Relyvrio'],
            "Disease-Modifying (ALS)"
        ),
        "cardiac_meds": _class_stats_combo(
            ['lisinopril', 'losartan', 'enalapril', 'carvedilol', 'metoprolol',
             'spironolactone', 'digoxin', 'benazepril', 'ramipril'],
            "Cardiac Medications"
        ),
        "psych_neuro": _class_stats_combo(
            ['olanzapine', 'Nuedexta', 'gabapentin', 'sertraline',
             'lorazepam', 'amitriptyline', 'baclofen'],
            "Psych / Neuro"
        ),
        "respiratory_meds": _class_stats_combo(
            ['albuterol', 'budesonide'],
            "Respiratory"
        ),
    }

    return combo_stats


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
    print(f"   Top 5 Sites (by patient count):")
    for facility in snapshot['facilities']['all_facilities'][:5]:
        print(f"      Site {facility['FACILITY_DISPLAY_ID']}: {facility['patient_count']:,} patients")

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
