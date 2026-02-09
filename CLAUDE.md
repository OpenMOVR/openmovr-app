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
- HIPAA small-cell suppression: counts <11 are suppressed

### Access Control
- DUA-gated pages use `utils/access.py` with `require_access()`
- Access key from `OPENMOVR_SITE_KEY` env var or `st.secrets`
- Session state `provisioned_access` persists across pages

## Architecture

```
Pages (UI) → API Layer (facade) → Snapshots OR Core Library (src/)
```

- `app.py` - Main dashboard
- `pages/` - 9 pages: Disease Explorer, Facility View, Data Dictionary, About, Sign the DUA, Site Analytics, Download Center, DMD Deep Dive, LGMD Deep Dive
- `api/` - Data access facade (StatsAPI, CohortAPI, DMDAPI, LGMDAPI, DataDictionaryAPI)
- `components/` - Shared UI (sidebar, deep_dive renderers, charts, tables, filters)
- `src/` - Core analytics library (cohort management, data loading)
- `stats/` - Pre-computed JSON snapshots (database, DMD, LGMD, curated dictionary)
- `config/` - App settings, disease filters, clinical domains
- `utils/` - Access control, caching, formatting

## Common Tasks

### Add a new disease deep dive
1. Create `scripts/generate_{disease}_snapshot.py`
2. Create `api/{disease}.py` with `{Disease}API` class
3. Add `render_{disease}_deep_dive()` in `components/deep_dive.py`
4. Register in Disease Explorer `_DEEP_DIVE_RENDERERS` dict
5. Create `pages/X_{Disease}_Deep_Dive.py` (DUA-gated)
6. Update sidebar CSS `nth-last-child` in `components/sidebar.py`
7. Update `api/__init__.py`

### Regenerate snapshots
```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_dmd_snapshot.py
python scripts/generate_lgmd_snapshot.py
python scripts/generate_curated_dictionary.py
```

### Update disease filters
Edit `config/disease_filters.yaml`

### Test imports
```bash
python -c "from api import StatsAPI, CohortAPI, LGMDAPI; print('OK')"
```

## Previous Work (Context for Continuation)

### Session 1: LGMD Overview & Deployment (Feb 2026)
- LGMD snapshot system (`scripts/generate_lgmd_snapshot.py`, `api/lgmd.py`)
- Snapshot fallback for all pages
- Branding: "OpenMOVR App" / "MOVR Data Hub | MOVR 1.0"
- Creator attribution: Andre D Paredes
- Extracted standalone repo from movr-clinical-analytics
- Curated data dictionary: 1,024 fields, 19 clinical domains

### Session 2: Deep Dives & DUA Pages (Feb 2026)
- DMD deep dive: exon-skipping therapeutics, steroids, functional outcomes (FVC, timed walk, loss of ambulation with longitudinal trends), genetics/mutations, state distribution, ambulatory status
- LGMD deep dive: subtypes, diagnostic journey (onset vs dx age, median 4.7yr delay), functional outcomes (FVC, timed walk, ambulatory), medications (cardiac, pain, supplements), clinical characteristics, geographic distribution
- Extracted deep-dive renderers to `components/deep_dive.py` (shared by Disease Explorer and standalone pages)
- Created DUA-gated standalone pages: `pages/8_DMD_Deep_Dive.py`, `pages/9_LGMD_Deep_Dive.py`
- Data tables with CSV export (summary tables from snapshot + patient-level from parquet with toggle)
- Version bump to v0.2.0 across all files
- Merged to main via PR #1 (clinical summary) and PR #2 (DUA deep dive pages)

### Key Decisions Made
1. Disease-first filtering in Data Dictionary (disease → form → field type)
2. Mislabeled field detection (FSHD fields incorrectly marked for LGMD)
3. Required field indicators (fields with * in Display Label)
4. Age at enrollment histogram (more accurate than current age)
5. Parquet files excluded from git via .gitignore
6. LGMD age bands: <18, 18-30, 30-40, 40-50, 50+ (adult-appropriate)
7. Deep-dive renderers in shared module to avoid code duplication
8. DUA pages have both summary (snapshot) and patient-level (parquet) data tabs

### Deployment Status
- GitHub: `OpenMOVR/openmovr-app` (public repo)
- Branch: `main` (up to date)
- Target: Streamlit Cloud deployment
- Desired URL: `app.openmovr.io` or `openmovr-app.streamlit.app`

## Testing the App

```bash
cd /home/aparedes/MDA/openmovr-app
pip install -r requirements.txt
export OPENMOVR_SITE_KEY="your-key"  # for DUA pages
streamlit run app.py
```

## File Locations

| Purpose | Location |
|---------|----------|
| Main app | `app.py` |
| Pages | `pages/*.py` (9 pages) |
| API layer | `api/*.py` (stats, cohorts, dmd, lgmd, data_dictionary, reports) |
| Deep-dive renderers | `components/deep_dive.py` |
| Access control | `utils/access.py` |
| Core library | `src/` |
| Snapshots | `stats/*.json` (database, dmd, lgmd, curated_dictionary) |
| Data (local) | `data/*.parquet` |
| Config | `config/` |

## Contact

Owner: Andre D Paredes (andre.paredes@ymail.com)
