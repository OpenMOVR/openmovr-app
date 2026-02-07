"""
Webapp Configuration

Settings and configuration for the MOVR Clinical Analytics webapp.
"""

from pathlib import Path

# App metadata
APP_TITLE = "OpenMOVR App"
APP_ICON = ""  # No icon in titles
APP_VERSION = "0.1.0"

# Layout settings
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Page configuration
PAGE_TITLE = "OpenMOVR App"
PAGE_ICON = "📊"  # Fallback emoji; pages use logo PNG when available

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
LOGO_PNG = ASSETS_DIR / "movr_logo_clean_nobackground.png"
LOGO_JPG = ASSETS_DIR / "movr_logo_clean.jpg"
DATA_DIR = PROJECT_ROOT / "data"
STATS_DIR = PROJECT_ROOT / "stats"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Cache settings
CACHE_TTL_SECONDS = 3600  # 1 hour

# Display settings
DEFAULT_TOP_N_FACILITIES = 10
MAX_TABLE_ROWS = 1000
DEFAULT_CHART_TYPE = "bar"

# Available diseases (in display order)
DISEASE_DISPLAY_ORDER = ['ALS', 'DMD', 'SMA', 'LGMD', 'FSHD', 'BMD', 'POM']

# Color schemes
COLOR_SCHEMES = {
    'disease': 'Blues',
    'facility': 'Viridis',
    'data_availability': 'Greens',
    'metrics': 'Purples',
    'demographics': 'Purples',
    'diagnosis': 'Greens',
    'clinical': 'Blues',
}

# Study metadata
STUDY_NAME = "MOVR Data Hub"
STUDY_PROTOCOL = "MOVR 1.0"
STUDY_START = "November 1, 2018"
STUDY_END = "April 1, 2025"
STUDY_STATUS = "Concluded"

# Feature flags (for gradual rollout)
FEATURES = {
    'disease_explorer': True,
    'facility_view': True,
    'query_builder': False,  # Not yet implemented
    'report_generator': True,
    'export_data': True,
    'advanced_filters': True
}
