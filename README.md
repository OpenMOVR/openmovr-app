# OpenMOVR App

Interactive dashboard for MOVR Clinical Data Analytics.

## Features

- **Dashboard**: Overview of MOVR database statistics
- **Disease Explorer**: Filter and explore disease-specific cohorts
- **Facility View**: Analyze facility distribution and patient counts
- **Data Dictionary**: Browse field definitions with completeness metrics
- **LGMD Overview**: Dedicated LGMD analysis for PAG presentations

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Data Modes

The app runs in two modes:

1. **Snapshot Mode** (default): Uses pre-computed statistics from `stats/` directory. Works without parquet files. Some interactive features are limited.

2. **Live Mode**: Uses parquet data files from `data/` directory. Full interactivity with filters and real-time calculations.

## Directory Structure

```
openmovr-app/
├── app.py                 # Main Streamlit entry point
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml        # Streamlit configuration
├── api/                   # API layer (data access facade)
├── components/            # Reusable UI components
├── pages/                 # Streamlit multi-page app
├── config/                # Configuration files
├── src/                   # Core analytics library
├── stats/                 # Pre-computed snapshots (JSON)
├── data/                  # Parquet data files (optional)
└── scripts/               # Utility scripts
```

## Data Files

### Snapshots (stats/)

Pre-computed JSON files with aggregate statistics. These contain no PHI and are safe to commit:

- `database_snapshot.json`: Overall database statistics
- `lgmd_snapshot.json`: LGMD-specific statistics

### Parquet Files (data/)

Raw data files for live mode. These may contain PHI:

- **Do NOT commit to public repos**
- For private repos, ensure proper access controls
- Contact MDA MOVR team for data access

## Regenerating Snapshots

When data is updated, regenerate snapshots:

```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_lgmd_snapshot.py
```

## Deployment to Streamlit Cloud

1. Push this repo to GitHub (ensure no PHI in parquet files if public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repo
5. Main file: `app.py`
6. Click "Deploy"

## Created By

Andre D Paredes  
aparedes@mdausa.org

---

**OpenMOVR App** | MOVR Data Hub | MOVR 1.0
