"""
LGMD API Layer

Provides LGMD cohort data from either live parquet files or pre-computed snapshots.
Falls back to snapshot when parquet files are unavailable (e.g., Streamlit Cloud deployment).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime


class LGMDAPI:
    """
    API for LGMD cohort data.

    Tries to load live data from parquet files first.
    Falls back to snapshot data if parquet files are unavailable.
    """

    _snapshot: Optional[Dict] = None
    _snapshot_path: Path = Path(__file__).parent.parent / "stats" / "lgmd_snapshot.json"
    _live_data: Optional[Dict] = None
    _data_source: str = "unknown"

    @classmethod
    def get_data_source(cls) -> str:
        """Return the current data source ('live' or 'snapshot')."""
        return cls._data_source

    @classmethod
    def load_data(cls) -> Dict[str, Any]:
        """
        Load LGMD data from parquet or snapshot.

        Returns:
            Dictionary with LGMD cohort data
        """
        # Try live data first
        try:
            from src.analytics.cohorts import get_disease_cohort
            cls._live_data = get_disease_cohort('LGMD')
            cls._data_source = "live"
            return cls._live_data
        except Exception as e:
            print(f"Could not load live data: {e}")

        # Fall back to snapshot
        if cls._load_snapshot():
            cls._data_source = "snapshot"
            return cls._get_snapshot_as_cohort()

        raise RuntimeError("No LGMD data available. Run generate_lgmd_snapshot.py or provide parquet files.")

    @classmethod
    def _load_snapshot(cls) -> bool:
        """Load snapshot from JSON file."""
        if cls._snapshot is not None:
            return True

        # Check multiple possible locations
        possible_paths = [
            cls._snapshot_path,
            Path(__file__).parent.parent / "stats" / "lgmd_snapshot.json",
            Path("stats/lgmd_snapshot.json"),
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        cls._snapshot = json.load(f)
                    return True
                except Exception as e:
                    print(f"Error loading snapshot from {path}: {e}")

        return False

    @classmethod
    def _get_snapshot_as_cohort(cls) -> Dict[str, Any]:
        """Convert snapshot format to cohort-like format for compatibility."""
        if cls._snapshot is None:
            return {}

        return {
            "count": cls._snapshot['summary']['total_patients'],
            "patient_ids": cls._snapshot['summary'].get('patient_ids', []),
            "facility_info": {
                "total_facilities": cls._snapshot['summary']['total_facilities'],
                "facilities": cls._snapshot['facilities'].get('facilities', [])
            },
            "_is_snapshot": True,
            "_snapshot_data": cls._snapshot
        }

    @classmethod
    def get_snapshot(cls) -> Optional[Dict]:
        """Get the raw snapshot data."""
        if cls._snapshot is None:
            cls._load_snapshot()
        return cls._snapshot

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get summary statistics."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            snapshot = data['_snapshot_data']
            return {
                "total_patients": snapshot['summary']['total_patients'],
                "total_facilities": snapshot['summary']['total_facilities'],
                "unique_subtypes": snapshot['subtypes'].get('unique_subtypes', 0),
                "genetic_confirmed_pct": snapshot['diagnosis'].get('genetic_confirmation', {}).get('confirmed_percentage', 0),
                "median_dx_age": snapshot['diagnosis'].get('diagnosis_age', {}).get('median', None),
                "data_source": "snapshot",
                "snapshot_date": snapshot['metadata'].get('generated_timestamp', 'Unknown')
            }
        else:
            # Calculate from live data
            diag_df = data.get('diagnosis', pd.DataFrame())
            patient_count = data.get('count', 0)

            # Genetic confirmation
            if 'lggntcf' in diag_df.columns:
                genetic_counts = diag_df['lggntcf'].value_counts()
                confirmed = genetic_counts.get('Yes – Laboratory confirmation', 0) + genetic_counts.get('Yes – In a family member', 0)
                genetic_pct = round(confirmed / patient_count * 100) if patient_count > 0 else 0
            else:
                genetic_pct = 0

            # Subtypes
            subtypes = diag_df['lgtype'].nunique() if 'lgtype' in diag_df.columns else 0

            # Median dx age
            median_dx = pd.to_numeric(diag_df['lgdgag'], errors='coerce').median() if 'lgdgag' in diag_df.columns else None

            return {
                "total_patients": patient_count,
                "total_facilities": data.get('facility_info', {}).get('total_facilities', 0),
                "unique_subtypes": subtypes,
                "genetic_confirmed_pct": genetic_pct,
                "median_dx_age": median_dx,
                "data_source": "live"
            }

    @classmethod
    def get_subtype_distribution(cls) -> List[Dict]:
        """Get subtype distribution data."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data']['subtypes'].get('distribution', [])
        else:
            diag_df = data.get('diagnosis', pd.DataFrame())
            if 'lgtype' not in diag_df.columns:
                return []

            subtype_counts = diag_df['lgtype'].value_counts()
            subtype_counts = subtype_counts[subtype_counts.index.astype(str).str.strip() != '']
            total = subtype_counts.sum()

            result = []
            for subtype, count in subtype_counts.items():
                if 'LGMD1' in str(subtype):
                    lgmd_type = 'LGMD Type 1 (Dominant)'
                elif 'LGMD2' in str(subtype):
                    lgmd_type = 'LGMD Type 2 (Recessive)'
                else:
                    lgmd_type = 'Other/Undetermined'

                result.append({
                    "subtype": str(subtype),
                    "patients": int(count),
                    "percentage": round(count / total * 100, 1) if total > 0 else 0,
                    "lgmd_type": lgmd_type
                })

            return result

    @classmethod
    def get_demographic_stats(cls) -> Dict[str, Any]:
        """Get demographic statistics."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data'].get('demographics', {})
        else:
            demo_df = data.get('demographics', pd.DataFrame())
            return cls._compute_live_demographics(demo_df)

    @classmethod
    def _compute_live_demographics(cls, demo_df: pd.DataFrame) -> Dict:
        """Compute demographic stats from live data."""
        stats = {"available": True}

        if 'dob' in demo_df.columns:
            dob_dates = pd.to_datetime(demo_df['dob'], errors='coerce')
            current_ages = (datetime.now() - dob_dates).dt.days / 365.25
            current_ages = current_ages.dropna()

            if len(current_ages) > 0:
                stats["current_age"] = {
                    "min": round(float(current_ages.min()), 1),
                    "max": round(float(current_ages.max()), 1),
                    "median": round(float(current_ages.median()), 1),
                    "count": int(len(current_ages))
                }

        if 'gender' in demo_df.columns:
            gender_counts = demo_df['gender'].value_counts()
            stats["gender"] = {
                "distribution": {str(k): int(v) for k, v in gender_counts.items()}
            }

        if 'ethnic' in demo_df.columns:
            ethnic_counts = demo_df['ethnic'].value_counts().head(10)
            stats["ethnicity"] = {
                "distribution": {str(k): int(v) for k, v in ethnic_counts.items()}
            }

        return stats

    @classmethod
    def get_diagnosis_stats(cls) -> Dict[str, Any]:
        """Get diagnosis statistics."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data'].get('diagnosis', {})
        else:
            diag_df = data.get('diagnosis', pd.DataFrame())
            patient_count = data.get('count', 0)
            return cls._compute_live_diagnosis(diag_df, patient_count)

    @classmethod
    def _compute_live_diagnosis(cls, diag_df: pd.DataFrame, patient_count: int) -> Dict:
        """Compute diagnosis stats from live data."""
        stats = {"available": True}

        if 'lgdgag' in diag_df.columns:
            dx_ages = pd.to_numeric(diag_df['lgdgag'], errors='coerce').dropna()
            if len(dx_ages) > 0:
                stats["diagnosis_age"] = {
                    "min": round(float(dx_ages.min()), 1),
                    "max": round(float(dx_ages.max()), 1),
                    "median": round(float(dx_ages.median()), 1),
                    "count": int(len(dx_ages))
                }

        if 'dymonag' in diag_df.columns:
            onset_ages = pd.to_numeric(diag_df['dymonag'], errors='coerce').dropna()
            if len(onset_ages) > 0:
                stats["onset_age"] = {
                    "min": round(float(onset_ages.min()), 1),
                    "max": round(float(onset_ages.max()), 1),
                    "median": round(float(onset_ages.median()), 1),
                    "count": int(len(onset_ages))
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

    @classmethod
    def get_clinical_stats(cls) -> Dict[str, Any]:
        """Get clinical characteristics."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data'].get('clinical', {})
        else:
            diag_df = data.get('diagnosis', pd.DataFrame())
            enc_df = data.get('encounters', pd.DataFrame())
            return cls._compute_live_clinical(diag_df, enc_df)

    @classmethod
    def _compute_live_clinical(cls, diag_df: pd.DataFrame, enc_df: pd.DataFrame) -> Dict:
        """Compute clinical stats from live data."""
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

    @classmethod
    def get_ambulatory_stats(cls) -> Dict[str, Any]:
        """Get ambulatory status statistics."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data'].get('ambulatory', {})
        else:
            enc_df = data.get('encounters', pd.DataFrame())
            diag_df = data.get('diagnosis', pd.DataFrame())

            if 'curramb' not in enc_df.columns:
                return {"available": False}

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
            return {
                "available": True,
                "current_status": {
                    "distribution": {str(k): int(v) for k, v in amb_counts.items()}
                }
            }

    @classmethod
    def get_facility_stats(cls) -> Dict[str, Any]:
        """Get facility distribution."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data'].get('facilities', {})
        else:
            facility_info = data.get('facility_info', {})
            if not facility_info or not facility_info.get('facilities'):
                return {"available": False}

            return {
                "available": True,
                "total_facilities": len(facility_info['facilities']),
                "facilities": [
                    {
                        "id": f.get('FACILITY_DISPLAY_ID', ''),
                        "name": f.get('FACILITY_NAME', ''),
                        "patients": int(f.get('patient_count', 0))
                    }
                    for f in facility_info['facilities']
                ]
            }

    @classmethod
    def get_all_subtypes(cls) -> List[str]:
        """Get list of all subtype values."""
        data = cls.load_data()

        if data.get('_is_snapshot'):
            return data['_snapshot_data']['subtypes'].get('all_subtypes', [])
        else:
            diag_df = data.get('diagnosis', pd.DataFrame())
            if 'lgtype' not in diag_df.columns:
                return []

            subtypes = diag_df['lgtype'].dropna().unique().tolist()
            return sorted([s for s in subtypes if str(s).strip()])

    @classmethod
    def get_functional_scores(cls) -> Dict[str, Any]:
        """Get functional outcome scores (FVC, timed walk, ambulatory status)."""
        snap = cls.get_snapshot()
        if snap:
            return snap.get('functional_scores', {})
        return {}

    @classmethod
    def get_state_distribution(cls) -> Dict[str, Any]:
        """Get geographic state distribution."""
        snap = cls.get_snapshot()
        if snap:
            return snap.get('state_distribution', {})
        return {}

    @classmethod
    def get_medication_stats(cls) -> Dict[str, Any]:
        """Get medication utilization by category and top drugs."""
        snap = cls.get_snapshot()
        if snap:
            return snap.get('medications', {})
        return {}

    @classmethod
    def get_diagnostic_journey(cls) -> Dict[str, Any]:
        """Get diagnostic delay and onset/diagnosis age data."""
        snap = cls.get_snapshot()
        if snap:
            return snap.get('diagnostic_journey', {})
        return {}
