"""
ALS API Layer

Provides ALS clinical summary data from either live parquet files or pre-computed snapshots.
Falls back to snapshot when parquet files are unavailable (e.g., Streamlit Cloud deployment).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class ALSAPI:
    """
    API for ALS clinical summary data.

    Tries to load live data from parquet files first.
    Falls back to snapshot data if parquet files are unavailable.
    """

    _snapshot: Optional[Dict] = None
    _snapshot_path: Path = Path(__file__).parent.parent / "stats" / "als_snapshot.json"

    @classmethod
    def _load_snapshot(cls) -> bool:
        """Load snapshot from JSON file."""
        if cls._snapshot is not None:
            return True

        possible_paths = [
            cls._snapshot_path,
            Path(__file__).parent.parent / "stats" / "als_snapshot.json",
            Path("stats/als_snapshot.json"),
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, "r") as f:
                        cls._snapshot = json.load(f)
                    return True
                except Exception as e:
                    print(f"Error loading ALS snapshot from {path}: {e}")

        return False

    @classmethod
    def get_snapshot(cls) -> Optional[Dict]:
        """Get the raw snapshot data."""
        if cls._snapshot is None:
            cls._load_snapshot()
        return cls._snapshot

    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """Get summary statistics."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("summary", {})

    @classmethod
    def get_alsfrs(cls) -> Dict[str, Any]:
        """Get ALSFRS-R score analysis."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("alsfrs", {})

    @classmethod
    def get_diagnosis(cls) -> Dict[str, Any]:
        """Get diagnosis data (El Escorial, body region, gene mutation, family history)."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("diagnosis", {})

    @classmethod
    def get_milestones(cls) -> Dict[str, Any]:
        """Get disease milestones (onset/diagnosis age, delay, LOA, speech, gastrostomy, NIV)."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("milestones", {})

    @classmethod
    def get_respiratory(cls) -> Dict[str, Any]:
        """Get respiratory function (FVC, NIV, tracheostomy)."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("respiratory", {})

    @classmethod
    def get_medications(cls) -> Dict[str, Any]:
        """Get medication utilization."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("medications", {})

    @classmethod
    def get_state_distribution(cls) -> Dict[str, Any]:
        """Get geographic distribution by state."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("state_distribution", {})

    @classmethod
    def get_demographics(cls) -> Dict[str, Any]:
        """Get demographics (diagnosis age, gender)."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("demographics", {})

    @classmethod
    def get_ambulatory(cls) -> Dict[str, Any]:
        """Get ambulatory status."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("ambulatory", {})

    @classmethod
    def get_facilities(cls) -> Dict[str, Any]:
        """Get facility distribution."""
        snap = cls.get_snapshot()
        if not snap:
            return {}
        return snap.get("facilities", {})
