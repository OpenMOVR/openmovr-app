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
│   ├── 1_Disease_Explorer.py   # Disease cohort filtering + clinical summary tabs
│   ├── 2_Facility_View.py      # Facility distribution + site map
│   ├── 3_Data_Dictionary.py    # Curated field browser (19 clinical domains)
│   ├── 4_About.py              # Study info, access tiers, roadmap
│   ├── 5_Sign_the_DUA.py       # DUA information + access request
│   ├── 6_Site_Analytics.py     # [DUA] Site-level reports
│   ├── 7_Download_Center.py    # [DUA] CSV/JSON data exports
│   ├── 8_DMD_Clinical_Summary.py      # [DUA] DMD analytics + data tables
│   └── 9_LGMD_Clinical_Summary.py     # [DUA] LGMD analytics + data tables
│
├── api/                        # Data Access Layer (facade pattern)
│   ├── __init__.py
│   ├── cohorts.py              # CohortAPI - cohort operations
│   ├── stats.py                # StatsAPI - snapshot statistics
│   ├── dmd.py                  # DMDAPI - DMD clinical summary data
│   ├── lgmd.py                 # LGMDAPI - LGMD clinical summary data
│   ├── data_dictionary.py      # DataDictionaryAPI - field metadata
│   └── reports.py              # ReportsAPI - report generation
│
├── components/                 # Reusable UI Components
│   ├── sidebar.py              # Global CSS, sidebar footer, page footer
│   ├── clinical_summary.py            # Shared clinical summary renderers (DMD, LGMD)
│   ├── charts.py               # Plotly chart factories
│   ├── tables.py               # DataFrame display helpers
│   └── filters.py              # Filter widgets
│
├── config/                     # Configuration
│   ├── settings.py             # App settings (title, version, paths, etc.)
│   ├── disease_filters.yaml    # Disease-specific filter definitions
│   ├── clinical_domains.yaml   # Curated clinical domain classifications
│   └── *.yaml                  # Other config files
│
├── utils/                      # Webapp utilities
│   ├── access.py               # Access key auth (require_access, has_access)
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
│   ├── dmd_snapshot.json       # DMD-specific statistics
│   ├── lgmd_snapshot.json      # LGMD-specific statistics
│   └── curated_dictionary.json # Field metadata (1,024 fields, 19 domains)
│
├── data/                       # Parquet Files (GITIGNORED - may contain PHI)
│   └── README.md               # Instructions for obtaining data
│
├── assets/                     # Logo and branding assets
│
└── scripts/                    # Utility Scripts
    ├── generate_stats_snapshot.py      # Regenerate database snapshot
    ├── generate_dmd_snapshot.py        # Regenerate DMD snapshot
    ├── generate_lgmd_snapshot.py       # Regenerate LGMD snapshot
    └── generate_curated_dictionary.py  # Regenerate curated dictionary
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT PAGES                          │
│  Public: Disease Explorer, Facility View, Data Dictionary       │
│  DUA: Site Analytics, Download Center, DMD/LGMD Clinical Summary       │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
          ┌──────────────┐    ┌──────────────────┐
          │  components/  │    │   utils/access.py │
          │  clinical_summary.py │    │   (access gate)   │
          │  sidebar.py   │    └──────────────────┘
          │  charts.py    │
          └──────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                          API LAYER                              │
│  StatsAPI, CohortAPI, DMDAPI, LGMDAPI, DataDictionaryAPI        │
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

### 3. Shared Clinical Summary Renderers
`components/clinical_summary.py` contains disease-specific chart renderers shared by:
- The Disease Explorer page (embedded in the Clinical Summary tab, live mode)
- Standalone DUA-gated pages (DMD/LGMD Clinical Summary)

This avoids code duplication while keeping pages independent.

### 4. Access Key Authentication
DUA-gated pages use `utils/access.py`:
- `require_access()` shows login form and calls `st.stop()` if not authenticated
- Access key resolved from `OPENMOVR_SITE_KEY` env var or `st.secrets`
- Session state `provisioned_access` persists across page navigations

### 5. Snapshot Pre-computation
JSON snapshots contain aggregate statistics with no PHI:
- Patient counts by disease
- Facility distributions
- Age histograms (binned)
- Subtype breakdowns
- Therapeutic utilization
- Functional scores (medians, IQR)
- HIPAA small-cell suppression (n<11)

### 6. Disease-First Filtering
Data Dictionary uses disease as primary filter, then form/table.
This matches how users think: "What LGMD fields are in Encounters?"

## DUA-Gated Pages

Pages 5-9 require provisioned access. The access system:

1. **Sign the DUA** (page 5) — informational, links to MDA access request form
2. **Site Analytics** (page 6) — requires access key
3. **Download Center** (page 7) — requires access key, CSV/JSON exports
4. **DMD Clinical Summary** (page 8) — requires access key, charts + data tables
5. **LGMD Clinical Summary** (page 9) — requires access key, charts + data tables

Clinical Summary pages include two tabs:
- **Summary Tables** — aggregated data from snapshots with CSV download
- **Patient-Level Data** — individual records from parquet (live mode only), behind a toggle checkbox

## Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub (public or private repo)
2. Connect at share.streamlit.io
3. App runs at `your-app.streamlit.app`
4. Configure `OPENMOVR_SITE_KEY` in Streamlit secrets

### Custom Domain
1. Deploy to Streamlit Cloud
2. Configure custom domain in Streamlit settings
3. Point DNS CNAME to Streamlit

### Local Development
```bash
pip install -r requirements.txt
export OPENMOVR_SITE_KEY="your-key"  # for DUA pages
streamlit run app.py
```

## Regenerating Snapshots

When data updates:
```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_dmd_snapshot.py
python scripts/generate_lgmd_snapshot.py
python scripts/generate_curated_dictionary.py
```

## Adding New Disease Clinical Summarys

1. Create snapshot generator: `scripts/generate_{disease}_snapshot.py`
2. Create API: `api/{disease}.py` with `{Disease}API` class
3. Add renderer: `render_{disease}_clinical_summary()` in `components/clinical_summary.py`
4. Register in Disease Explorer: add to `_CLINICAL_SUMMARY_RENDERERS` dict
5. Create DUA page: `pages/X_{Disease}_Clinical_Summary.py`
6. Update sidebar CSS: adjust `nth-last-child` in `components/sidebar.py`
7. Update `api/__init__.py` to export new API

## Contact

Andre D Paredes
andre.paredes@ymail.com
