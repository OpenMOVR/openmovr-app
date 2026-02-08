# OpenMOVR App

Open-source analytics dashboard for the **MDA MOVR Data Hub** clinical registry. Built for the research community — clinical researchers, academic institutions, patient advocacy groups — to explore longitudinal data across seven rare neuromuscular diseases: ALS, DMD, BMD, SMA, LGMD, FSHD, and Pompe disease.

Independently built via the [OpenMOVR Initiative](https://openmovr.github.io).

## Features

### Public Access (no login required)

- **Dashboard** — Aggregated enrollment statistics, disease distribution charts, participating sites map, longitudinal summaries, and clinical data highlights (functional scores, medications, trials, hospitalizations)
- **Disease Explorer** — Per-disease cohort views with demographics, diagnosis profiles, and disease-specific deep-dives (LGMD subtype distribution, genetic confirmation, clinical characteristics)
- **Facility View** — Anonymized site map with disease filters, patient count ranges, and site distribution analysis
- **Data Dictionary** — Curated dictionary covering 1,024 clinical fields across 19 clinical domains, with disease filtering, required-field indicators, and mislabeled-field detection
- **About** — Study details, access tiers, roadmap, and version history

### Provisioned Access (DUA required)

- **Site Analytics** — Site-level reports with facility names, site-vs-overall comparisons, per-disease breakdowns, and disease-specific variable charts
- **Download Center** — Export disease distribution, advanced therapies, top medications, clinical availability, longitudinal summaries, facility data, and full snapshot as CSV or JSON

All public data is pre-computed aggregated statistics (snapshots) — no individual-level data is connected or displayed.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Data Modes

1. **Snapshot Mode** (default) — Uses pre-computed JSON files in `stats/`. Works without parquet files. Powers the public deployment on Streamlit Cloud.

2. **Live Mode** — Uses parquet data files in `data/`. Full interactivity with cohort filters, real-time calculations, and site analytics.

## Directory Structure

```
openmovr-app/
├── app.py                     # Main Streamlit entry point
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── api/                       # Data access facade (StatsAPI, CohortAPI, LGMDAPI, DataDictionaryAPI)
├── components/                # Shared UI (sidebar branding, charts, tables, filters)
├── config/                    # Settings, disease filters, clinical domains, export profiles
├── pages/                     # Streamlit multi-page app
│   ├── 1_Disease_Explorer.py
│   ├── 2_Facility_View.py
│   ├── 3_Data_Dictionary.py
│   ├── 4_About.py
│   ├── 5_Sign_the_DUA.py
│   ├── 6_Site_Analytics.py
│   └── 7_Download_Center.py
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
- `lgmd_snapshot.json` — LGMD-specific subtypes, diagnosis, clinical characteristics, ambulatory status
- `curated_dictionary.json` — Field-level metadata for 1,024 clinical fields across 19 domains

### Parquet Files (`data/`)

Raw data files for live mode. May contain PHI — **do not commit to public repos**. Contact the MDA MOVR team for data access.

## Regenerating Snapshots

```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_lgmd_snapshot.py
python scripts/generate_curated_dictionary.py
```

## Deployment

Deployed on [Streamlit Cloud](https://share.streamlit.io) in snapshot mode. The public deployment requires no parquet files — all data comes from pre-computed JSON snapshots.

## Created By

Andre D Paredes
andre.paredes@ymail.com

---

**OpenMOVR App** | Gen1 | v0.1.0 (Prototype)
Data Source: [MDA MOVR Data Hub Study](https://mdausa.tfaforms.net/389761)
