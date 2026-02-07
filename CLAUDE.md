# CLAUDE.md - Instructions for Claude Code

This file helps Claude Code understand the OpenMOVR App codebase and continue previous work.

## Project Summary

**OpenMOVR App** is a Streamlit dashboard for MOVR (Muscular Dystrophy Association registry) clinical data analytics. It displays patient statistics, disease distributions, and facility information for rare neuromuscular diseases (DMD, SMA, ALS, LGMD, etc.).

## Key Context

### Dual-Mode Operation
- **Snapshot Mode**: Uses pre-computed JSON files in `stats/`. No parquet files needed. Limited interactivity.
- **Live Mode**: Uses parquet files in `data/`. Full filtering and real-time calculations.

The API layer (`api/`) automatically switches between modes based on data availability.

### Data Privacy
- `data/*.parquet` files are GITIGNORED (may contain PHI)
- `stats/*.json` files are safe to commit (aggregated, no PHI)
- Never commit parquet files to public repos

## Architecture

```
Pages (UI) → API Layer (facade) → Snapshots OR Core Library (src/)
```

- `app.py` - Main dashboard
- `pages/` - Disease Explorer, Facility View, Data Dictionary, LGMD Overview
- `api/` - Data access facade (CohortAPI, StatsAPI, LGMDAPI, DataDictionaryAPI)
- `src/` - Core analytics library (cohort management, data loading)
- `stats/` - Pre-computed JSON snapshots
- `config/` - App settings, disease filters

## Common Tasks

### Add a new disease overview page
1. Create `scripts/generate_{disease}_snapshot.py`
2. Create `api/{disease}.py` with `{Disease}API` class
3. Create `pages/X_{Disease}_Overview.py`
4. Update `api/__init__.py`

### Regenerate snapshots
```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_lgmd_snapshot.py
```

### Update disease filters
Edit `config/disease_filters.yaml`

### Test imports
```bash
python -c "from api import StatsAPI, CohortAPI, LGMDAPI; print('OK')"
```

## Previous Work (Context for Continuation)

### Session: LGMD Overview & Deployment (Feb 2026)
Created:
- LGMD Overview page (`pages/4_LGMD_Overview.py`) with PAG presentation features
- LGMD snapshot system (`scripts/generate_lgmd_snapshot.py`, `api/lgmd.py`)
- Snapshot fallback for all pages
- Branding: "OpenMOVR App" / "MOVR Data Hub | MOVR 1.0"
- Creator attribution: Andre D Paredes (aparedes@mdausa.org)
- Extracted standalone repo from movr-clinical-analytics

### Key Decisions Made
1. Disease-first filtering in Data Dictionary (disease → form → field type)
2. Mislabeled field detection (FSHD fields incorrectly marked for LGMD)
3. Required field indicators (fields with * in Display Label)
4. Age at enrollment histogram (more accurate than current age)
5. Parquet files excluded from git via .gitignore

### Deployment Status
- Repo ready at `/home/aparedes/MDA/openmovr-app/`
- Not yet pushed to GitHub
- Target: Streamlit Cloud deployment
- Desired URL: something like `app.openmovr.io` or similar

## Deployment Options

GitHub Pages CANNOT run Streamlit (static only). Options:

1. **Streamlit Cloud** (recommended)
   - Free tier available
   - URL: `openmovr-app.streamlit.app`
   - Custom domain possible

2. **Redirect from GitHub Pages**
   - Create `app.openmovr.github.io` repo
   - Add index.html that redirects to Streamlit URL

3. **Custom domain**
   - Point `app.openmovr.io` to Streamlit Cloud

## Testing the App

```bash
cd /home/aparedes/MDA/openmovr-app
pip install -r requirements.txt
streamlit run app.py
```

## File Locations

| Purpose | Location |
|---------|----------|
| Main app | `app.py` |
| Pages | `pages/*.py` |
| API layer | `api/*.py` |
| Core library | `src/` |
| Snapshots | `stats/*.json` |
| Data (local) | `data/*.parquet` |
| Config | `config/` |

## Contact

Owner: Andre D Paredes (aparedes@mdausa.org)
