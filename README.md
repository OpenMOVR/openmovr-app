# OpenMOVR App

Open-source analytics dashboard for the **MDA MOVR Data Hub** clinical registry. Built for the research community — clinical researchers, academic institutions, patient advocacy groups — to explore longitudinal data across seven rare neuromuscular diseases: ALS, DMD, BMD, SMA, LGMD, FSHD, and Pompe disease.

Independently built via the [OpenMOVR Initiative](https://openmovr.github.io).

## Features

### Public Access (no login required)

- **Dashboard** — Aggregated enrollment statistics, disease distribution charts, participating sites map, longitudinal summaries, clinical data highlights (functional scores, medications, trials, hospitalizations), and cumulative/monthly enrollment charts
- **Disease Explorer** — Per-disease cohort views with demographics, diagnosis profiles, and Clinical Summary Preview (DMD exon-skipping therapeutics, LGMD subtype distribution, ALS ALSFRS-R scores, SMA motor function, and more)
- **Facility View** — Anonymized site map with disease filters, participant count ranges, recruitment over time by state, and site distribution analysis
- **Data Dictionary** — Curated dictionary covering 1,024 clinical fields across 19 clinical domains, with disease filtering, required-field indicators, and mislabeled-field detection
- **About** — Study details, access tiers, roadmap, and version history

### Provisioned Access (DUA required)

- **Sign the DUA** — Information and links for requesting data access
- **Site Analytics** — Site-level reports with facility names, site-vs-overall comparisons, per-disease breakdowns, and disease-specific variable charts
- **Download Center** — Export disease distribution, advanced therapies, top medications, clinical availability, longitudinal summaries, facility data, and full snapshot as CSV or JSON
- **DMD Clinical Analytics** — Exon-skipping therapeutics, glucocorticoids, functional outcomes (FVC, timed walk, loss of ambulation with longitudinal trends), genetics/mutations, state distribution, data tables with CSV export
- **LGMD Clinical Analytics** — Subtype distribution, diagnostic journey (onset vs diagnosis age, diagnostic delay), functional outcomes, medication utilization, genetics, data tables with CSV export
- **ALS Clinical Analytics** — ALSFRS-R scores with longitudinal trends, El Escorial classification, disease milestones, respiratory function, medication utilization, genetics, data tables with CSV export
- **SMA Clinical Analytics** — Motor scores (HFMSE, CHOP-INTEND, RULM with longitudinal trends), SMA type classification, SMN2 genetics with cross-tabulations, therapeutics (Spinraza, Evrysdi, Zolgensma), respiratory function, data tables with CSV export

All public data is pre-computed aggregated statistics (snapshots) — no individual-level data is connected or displayed. No database is connected. DUA-gated pages require an access key and support both snapshot (summary tables) and live (participant-level data) modes.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Access Key Configuration

DUA-gated pages require an access key. Set via environment variable or Streamlit secrets:

```bash
export OPENMOVR_SITE_KEY="your-access-key"
```

Or in `.streamlit/secrets.toml`:
```toml
OPENMOVR_SITE_KEY = "your-access-key"
```

## Data Modes

1. **Snapshot Mode** (default) — Uses pre-computed JSON files in `stats/`. Works without parquet files. Powers the public deployment on Streamlit Cloud. DUA pages show summary tables only.

2. **Live Mode** — Uses parquet data files in `data/`. Full interactivity with cohort filters, real-time calculations, site analytics, and participant-level data exports.

## Directory Structure

```
openmovr-app/
├── app.py                     # Main Streamlit entry point
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── api/                       # Data access facade
│   ├── stats.py               # StatsAPI — snapshot statistics
│   ├── cohorts.py             # CohortAPI — cohort operations
│   ├── dmd.py                 # DMDAPI — DMD clinical analytics data
│   ├── lgmd.py                # LGMDAPI — LGMD clinical analytics data
│   ├── als.py                 # ALSAPI — ALS clinical analytics data
│   ├── sma.py                 # SMAAPI — SMA clinical analytics data
│   ├── data_dictionary.py     # DataDictionaryAPI — field metadata
│   └── reports.py             # ReportsAPI — report generation
├── components/                # Shared UI components
│   ├── sidebar.py             # Sidebar branding, CSS, page footer
│   ├── clinical_summary.py    # DMD, LGMD, ALS & SMA clinical summary renderers
│   ├── charts.py              # Plotly chart factories
│   ├── tables.py              # DataFrame display helpers
│   └── filters.py             # Filter widgets
├── config/                    # Settings, disease filters, clinical domains, export profiles
├── pages/                     # Streamlit multi-page app
│   ├── 1_Disease_Explorer.py  # Disease cohort filtering + Clinical Summary Preview tabs
│   ├── 2_Facility_View.py     # Facility distribution + site map
│   ├── 3_Data_Dictionary.py   # Curated field browser
│   ├── 4_About.py             # Study info + version history
│   ├── 5_Sign_the_DUA.py      # DUA information + access request
│   ├── 6_Site_Analytics.py    # [DUA] Site-level reports
│   ├── 7_Download_Center.py   # [DUA] CSV/JSON data exports
│   ├── 8_DMD_Clinical_Analytics.py   # [DUA] DMD analytics + data tables
│   ├── 9_LGMD_Clinical_Analytics.py  # [DUA] LGMD analytics + data tables
│   ├── 10_ALS_Clinical_Analytics.py  # [DUA] ALS analytics + data tables
│   └── 11_SMA_Clinical_Analytics.py  # [DUA] SMA analytics + data tables
├── utils/                     # Webapp utilities
│   ├── access.py              # Access key authentication (require_access, has_access)
│   ├── cache.py               # Streamlit caching helpers
│   └── formatting.py          # Display formatting
├── src/                       # Core analytics library (cohorts, cleaning, exports, validation)
├── stats/                     # Pre-computed snapshots (JSON, no PHI)
├── data/                      # Parquet data files (gitignored, may contain PHI)
├── assets/                    # Logo and branding assets
└── scripts/                   # Snapshot generation scripts
```

## Data Files

### Snapshots (`stats/`)

Pre-computed JSON files with aggregate statistics. No PHI — safe to commit:

- `database_snapshot.json` — Overall enrollment, disease distribution, facilities, longitudinal data, clinical availability, medications, gene therapy breakdowns
- `dmd_snapshot.json` — DMD therapeutics (exon-skipping amenability/utilization), steroids, functional scores, genetics, state distribution, ambulatory status, facilities
- `lgmd_snapshot.json` — LGMD subtypes, diagnosis, clinical characteristics, ambulatory status, functional scores, medications, diagnostic journey, state distribution
- `als_snapshot.json` — ALS ALSFRS-R scores, El Escorial classification, disease milestones, respiratory function, medications, genetics, state distribution
- `sma_snapshot.json` — SMA type classification, SMN2 genetics, motor scores (HFMSE, CHOP-INTEND, RULM), therapeutics, respiratory function, state distribution
- `curated_dictionary.json` — Field-level metadata for 1,024 clinical fields across 19 domains

### Parquet Files (`data/`)

Raw data files for live mode. May contain PHI — **do not commit to public repos**. Contact the MDA MOVR team for data access.

## Regenerating Snapshots

```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_dmd_snapshot.py
python scripts/generate_lgmd_snapshot.py
python scripts/generate_als_snapshot.py
python scripts/generate_sma_snapshot.py
python scripts/generate_curated_dictionary.py
```

## Deployment

Deployed on [Streamlit Cloud](https://share.streamlit.io) in snapshot mode. The public deployment requires no parquet files — all data comes from pre-computed JSON snapshots.

## Created By

Andre D Paredes
andre.paredes@ymail.com

---

**OpenMOVR App** | Gen1 | v0.2.0
Data Source: [MDA MOVR Data Hub Study](https://mdausa.tfaforms.net/389761)
