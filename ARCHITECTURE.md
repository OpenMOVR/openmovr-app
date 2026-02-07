# OpenMOVR App Architecture

## Overview

OpenMOVR App is a Streamlit-based dashboard for MOVR clinical data analytics. It's designed to run in two modes:

1. **Snapshot Mode** - Uses pre-computed JSON statistics (no PHI, deployable publicly)
2. **Live Mode** - Uses parquet data files (full interactivity, requires data access)

## Directory Structure

```
openmovr-app/
├── app.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── .streamlit/config.toml      # Streamlit theme/config
│
├── pages/                      # Streamlit multi-page app
│   ├── 1_Disease_Explorer.py   # Disease cohort filtering
│   ├── 2_Facility_View.py      # Facility distribution
│   ├── 3_Data_Dictionary.py    # Field browser with completeness
│   └── 4_LGMD_Overview.py      # LGMD-specific PAG presentation
│
├── api/                        # Data Access Layer (facade pattern)
│   ├── __init__.py
│   ├── cohorts.py              # CohortAPI - cohort operations
│   ├── stats.py                # StatsAPI - snapshot statistics
│   ├── lgmd.py                 # LGMDAPI - LGMD-specific data
│   ├── data_dictionary.py      # DataDictionaryAPI - field metadata
│   └── reports.py              # ReportsAPI - report generation
│
├── components/                 # Reusable UI Components
│   ├── charts.py               # Plotly chart factories
│   ├── tables.py               # DataFrame display helpers
│   └── filters.py              # Filter widgets
│
├── config/                     # Configuration
│   ├── settings.py             # App settings (title, theme, etc.)
│   ├── disease_filters.yaml    # Disease-specific filter definitions
│   └── *.yaml                  # Other config files
│
├── utils/                      # Webapp utilities
│   ├── cache.py                # Streamlit caching helpers
│   └── formatting.py           # Display formatting
│
├── src/                        # Core Analytics Library
│   ├── analytics/
│   │   ├── cohorts.py          # Cohort management (get_base_cohort, get_disease_cohort)
│   │   ├── filters.py          # MOVR/USNDR filtering
│   │   └── base.py             # BaseAnalyzer class
│   ├── data_processing/
│   │   ├── loader.py           # Parquet data loading with caching
│   │   └── data_dictionary.py  # Field metadata operations
│   └── utils/
│       └── diagnosis_config.py # Disease field mappings
│
├── stats/                      # Pre-computed Snapshots (NO PHI)
│   ├── database_snapshot.json  # Overall database statistics
│   └── lgmd_snapshot.json      # LGMD-specific statistics
│
├── data/                       # Parquet Files (GITIGNORED - may contain PHI)
│   └── README.md               # Instructions for obtaining data
│
└── scripts/                    # Utility Scripts
    ├── generate_stats_snapshot.py   # Regenerate database snapshot
    └── generate_lgmd_snapshot.py    # Regenerate LGMD snapshot
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT PAGES                          │
│  (Disease Explorer, Facility View, Data Dictionary, LGMD)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          API LAYER                              │
│  CohortAPI, StatsAPI, LGMDAPI, DataDictionaryAPI                │
│  - Provides clean interface for pages                           │
│  - Handles snapshot vs live mode switching                      │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│     SNAPSHOT MODE       │     │       LIVE MODE         │
│  stats/*.json           │     │  src/analytics/         │
│  (pre-computed, no PHI) │     │  data/*.parquet         │
│  Fast, limited features │     │  Full interactivity     │
└─────────────────────────┘     └─────────────────────────┘
```

## Key Design Decisions

### 1. Dual-Mode Operation
The app gracefully degrades when parquet files aren't available. This enables:
- Public deployment on Streamlit Cloud (snapshot mode)
- Private deployment with full data (live mode)

### 2. API Facade Pattern
The `api/` layer abstracts data access. Pages don't directly import from `src/`.
This makes it easy to:
- Switch between snapshot and live mode
- Mock data for testing
- Extract webapp to separate repo

### 3. Snapshot Pre-computation
JSON snapshots contain aggregate statistics with no PHI:
- Patient counts by disease
- Facility distributions
- Age histograms (binned)
- Subtype breakdowns

### 4. Disease-First Filtering
Data Dictionary uses disease as primary filter, then form/table.
This matches how users think: "What LGMD fields are in Encounters?"

## Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub (public or private repo)
2. Connect at share.streamlit.io
3. App runs at `your-app.streamlit.app`

### Custom Domain
1. Deploy to Streamlit Cloud
2. Configure custom domain in Streamlit settings
3. Point DNS CNAME to Streamlit

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Regenerating Snapshots

When data updates:
```bash
# From movr-clinical-analytics repo (has parquet files)
python scripts/generate_stats_snapshot.py
python scripts/generate_lgmd_snapshot.py

# Copy to openmovr-app
cp stats/*.json /path/to/openmovr-app/stats/
```

## Adding New Disease Overview Pages

1. Create snapshot generator: `scripts/generate_{disease}_snapshot.py`
2. Create API: `api/{disease}.py` with snapshot fallback
3. Create page: `pages/X_{Disease}_Overview.py`
4. Update `api/__init__.py` to export new API

## Contact

Andre D Paredes
aparedes@mdausa.org
