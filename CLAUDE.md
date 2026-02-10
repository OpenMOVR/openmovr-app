# CLAUDE.md - Instructions for Claude Code

This file helps Claude Code understand the OpenMOVR App codebase and continue previous work.

## Project Summary

**OpenMOVR App** is a Streamlit dashboard for MOVR (Muscular Dystrophy Association registry) clinical data analytics. It displays participant statistics, disease distributions, and facility information for rare neuromuscular diseases (DMD, SMA, ALS, LGMD, etc.).

> **IRB Language**: All user-facing text uses "participants" (not "patients") per IRB requirements — these are enrolled participants, not a site's total patient volume. Internal variable names and data keys still use `patient_count`, `total_patients`, etc.

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
- `pages/` - 11 pages: Disease Explorer, Facility View, Data Dictionary, About, Sign the DUA, Site Analytics, Download Center, DMD Clinical Analytics, LGMD Clinical Analytics, ALS Clinical Analytics, SMA Clinical Analytics
- `api/` - Data access facade (StatsAPI, CohortAPI, DMDAPI, LGMDAPI, ALSAPI, SMAAPI, DataDictionaryAPI)
- `components/` - Shared UI (sidebar, clinical_summary renderers, charts, tables, filters)
- `src/` - Core analytics library (cohort management, data loading)
- `stats/` - Pre-computed JSON snapshots (database, DMD, LGMD, curated dictionary)
- `config/` - App settings, disease filters, clinical domains
- `utils/` - Access control, caching, formatting

## Common Tasks

### Add a new disease clinical summary
1. Create `scripts/generate_{disease}_snapshot.py`
2. Create `api/{disease}.py` with `{Disease}API` class
3. Add `render_{disease}_clinical_summary()` in `components/clinical_summary.py`
4. Register in Disease Explorer `_CLINICAL_SUMMARY_RENDERERS` dict
5. Create `pages/X_{Disease}_Clinical_Summary.py` (DUA-gated)
6. Update sidebar CSS `nth-last-child` in `components/sidebar.py`
7. Update `api/__init__.py`

### Regenerate snapshots
```bash
python scripts/generate_stats_snapshot.py
python scripts/generate_dmd_snapshot.py
python scripts/generate_lgmd_snapshot.py
python scripts/generate_als_snapshot.py
python scripts/generate_sma_snapshot.py
python scripts/generate_curated_dictionary.py
```

### Update disease filters
Edit `config/disease_filters.yaml`

### Test imports
```bash
python -c "from api import StatsAPI, CohortAPI, LGMDAPI, ALSAPI, SMAAPI; print('OK')"
```

## Previous Work (Context for Continuation)

### Session 1: LGMD Overview & Deployment (Feb 2026)
- LGMD snapshot system (`scripts/generate_lgmd_snapshot.py`, `api/lgmd.py`)
- Snapshot fallback for all pages
- Branding: "OpenMOVR App" / "MOVR Data Hub | MOVR 1.0"
- Creator attribution: Andre D Paredes
- Extracted standalone repo from movr-clinical-analytics
- Curated data dictionary: 1,024 fields, 19 clinical domains

### Session 2: Clinical Summaries & DUA Pages (Feb 2026)
- DMD clinical summary: exon-skipping therapeutics, steroids, functional outcomes (FVC, timed walk, loss of ambulation with longitudinal trends), genetics/mutations, state distribution, ambulatory status
- LGMD clinical summary: subtypes, diagnostic journey (onset vs dx age, median 4.7yr delay), functional outcomes (FVC, timed walk, ambulatory), medications (cardiac, pain, supplements), clinical characteristics, geographic distribution
- Extracted clinical summary renderers to `components/clinical_summary.py` (shared by Disease Explorer and standalone pages)
- Created DUA-gated standalone pages: `pages/8_DMD_Clinical_Summary.py`, `pages/9_LGMD_Clinical_Summary.py`
- Data tables with CSV export (summary tables from snapshot + patient-level from parquet with toggle)
- Version bump to v0.2.0 across all files
- Sections organized by canonical clinical domains from `config/clinical_domains.yaml`

### Key Decisions Made
1. Disease-first filtering in Data Dictionary (disease → form → field type)
2. Mislabeled field detection (FSHD fields incorrectly marked for LGMD)
3. Required field indicators (fields with * in Display Label)
4. Age at enrollment histogram (more accurate than current age)
5. Parquet files excluded from git via .gitignore
6. LGMD age bands: <18, 18-30, 30-40, 40-50, 50+ (adult-appropriate)
7. Clinical summary renderers in shared module to avoid code duplication
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
| Pages | `pages/*.py` (11 pages) |
| API layer | `api/*.py` (stats, cohorts, dmd, lgmd, als, sma, data_dictionary, reports) |
| Clinical summary renderers | `components/clinical_summary.py` |
| Access control | `utils/access.py` |
| Core library | `src/` |
| Snapshots | `stats/*.json` (database, dmd, lgmd, als, sma, curated_dictionary) |
| Data (local) | `data/*.parquet` |
| Config | `config/` |

## Contact

MDA MOVR Data Hub Team: mdamovr@mdausa.org
Developer: Andre D Paredes (andre.paredes@ymail.com)
