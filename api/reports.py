"""
Reports API Layer

Provides report generation capabilities.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ReportsAPI:
    """
    API for report generation.

    Wraps the report generation scripts for use in the webapp.
    """

    SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

    @classmethod
    def generate_snapshot(cls, include_usndr: bool = False) -> Dict[str, Any]:
        """
        Generate a new database snapshot.

        Args:
            include_usndr: Include USNDR patients

        Returns:
            Result dictionary with status and path
        """
        script = cls.SCRIPTS_DIR / "generate_stats_snapshot.py"

        cmd = ["python", str(script)]
        if include_usndr:
            cmd.append("--include-usndr")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': None,
                'error': 'Snapshot generation timed out (>5 minutes)'
            }
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'error': str(e)
            }

    @classmethod
    def generate_database_report(cls,
                                 format: str = 'excel',
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a database report.

        Args:
            format: Report format (excel, markdown, csv)
            output_path: Optional custom output path

        Returns:
            Result dictionary with status and path
        """
        script = cls.SCRIPTS_DIR / "generate_database_report.py"

        cmd = ["python", str(script), "--format", format]
        if output_path:
            cmd.extend(["--output", output_path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Try to extract the output path from stdout
            output_file = None
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'saved to:' in line.lower():
                        output_file = line.split('saved to:')[-1].strip()
                        break

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'output_file': output_file,
                'error': result.stderr if result.returncode != 0 else None
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': None,
                'output_file': None,
                'error': 'Report generation timed out (>5 minutes)'
            }
        except Exception as e:
            return {
                'success': False,
                'output': None,
                'output_file': None,
                'error': str(e)
            }
