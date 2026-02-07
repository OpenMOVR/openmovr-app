"""
Webapp Configuration

Settings and configuration for the MOVR Clinical Analytics webapp.
"""

from pathlib import Path

# App metadata
APP_TITLE = "OpenMOVR App"
APP_ICON = ""  # No icon in titles
APP_VERSION = "0.2.0"

# Layout settings
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Page configuration
PAGE_TITLE = "OpenMOVR"
PAGE_ICON = "📊"

# Paths
WEBAPP_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = WEBAPP_ROOT.parent
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

# Feature flags (for gradual rollout)
FEATURES = {
    'disease_explorer': True,
    'facility_view': True,
    'query_builder': False,  # Not yet implemented
    'report_generator': True,
    'export_data': True,
    'advanced_filters': True
}
