"""
Statistics API Layer

Provides access to database statistics from the snapshot file.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StatsAPI:
    """
    API for database statistics.

    Reads from the stats snapshot JSON for fast access to common metrics.
    """

    SNAPSHOT_PATH = Path(__file__).parent.parent / "stats" / "database_snapshot.json"

    @classmethod
    def load_snapshot(cls) -> Dict[str, Any]:
        """
        Load the database statistics snapshot.

        Returns:
            Dictionary with snapshot data

        Raises:
            FileNotFoundError: If snapshot doesn't exist
        """
        if not cls.SNAPSHOT_PATH.exists():
            raise FileNotFoundError(
                f"Statistics snapshot not found at {cls.SNAPSHOT_PATH}\n"
                f"Run: python scripts/generate_stats_snapshot.py"
            )

        with open(cls.SNAPSHOT_PATH, 'r') as f:
            return json.load(f)

    @classmethod
    def get_enrollment_stats(cls) -> Dict[str, Any]:
        """Get enrollment statistics."""
        snapshot = cls.load_snapshot()
        return snapshot['enrollment']

    @classmethod
    def get_disease_distribution(cls) -> List[Dict[str, Any]]:
        """Get disease distribution list."""
        snapshot = cls.load_snapshot()
        return snapshot['disease_distribution']['diseases']

    @classmethod
    def get_disease_summary(cls) -> Dict[str, Dict[str, int]]:
        """Get disease summary dict (disease -> {count, percentage})."""
        snapshot = cls.load_snapshot()
        return snapshot['disease_distribution']['disease_summary']

    @classmethod
    def get_facility_stats(cls) -> Dict[str, Any]:
        """Get facility statistics."""
        snapshot = cls.load_snapshot()
        return snapshot['facilities']

    @classmethod
    def get_top_facilities(cls, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get top N facilities by patient count.

        Args:
            n: Number of facilities to return

        Returns:
            List of facility dicts
        """
        snapshot = cls.load_snapshot()
        return snapshot['facilities']['top_10_facilities'][:n]

    @classmethod
    def get_all_facilities(cls) -> List[Dict[str, Any]]:
        """Get all facilities."""
        snapshot = cls.load_snapshot()
        return snapshot['facilities']['all_facilities']

    @classmethod
    def get_data_availability(cls) -> Dict[str, int]:
        """Get data availability metrics."""
        snapshot = cls.load_snapshot()
        return snapshot['data_availability']

    @classmethod
    def get_snapshot_metadata(cls) -> Dict[str, Any]:
        """Get snapshot metadata (when generated, cohort type, etc.)."""
        snapshot = cls.load_snapshot()
        return snapshot['metadata']

    @classmethod
    def get_total_patients(cls) -> int:
        """Get total patient count."""
        snapshot = cls.load_snapshot()
        return snapshot['enrollment']['total_patients']

    @classmethod
    def get_patient_count_for_disease(cls, disease: str) -> Optional[int]:
        """
        Get patient count for a specific disease.

        Args:
            disease: Disease code (DMD, ALS, etc.)

        Returns:
            Patient count or None if disease not found
        """
        summary = cls.get_disease_summary()
        disease_data = summary.get(disease.upper())
        return disease_data['count'] if disease_data else None

    @classmethod
    def snapshot_exists(cls) -> bool:
        """Check if snapshot file exists."""
        return cls.SNAPSHOT_PATH.exists()

    @classmethod
    def get_snapshot_age(cls) -> Optional[str]:
        """
        Get how old the snapshot is.

        Returns:
            Human-readable age string or None if no snapshot
        """
        if not cls.snapshot_exists():
            return None

        metadata = cls.get_snapshot_metadata()
        generated_at = datetime.fromisoformat(metadata['generated_at'])
        age = datetime.now() - generated_at

        if age.days > 0:
            return f"{age.days} day{'s' if age.days != 1 else ''} ago"
        elif age.seconds > 3600:
            hours = age.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif age.seconds > 60:
            minutes = age.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "just now"
