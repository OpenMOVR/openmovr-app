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
- `pages/` - 12 pages: Community Snapshot, Disease Explorer, Facility View, Data Dictionary, About, Sign the DUA, Site Analytics, Download Center, DMD Clinical Analytics, LGMD Clinical Analytics, ALS Clinical Analytics, SMA Clinical Analytics
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

### Add a study-wide (cross-disease) metric

Per-disease numbers live in `disease_profiles`; anything that describes the
whole registry lives in `community_profile`. Health insurance is the worked
example -- follow the same path for any new cross-disease measure.

1. Compute it in `_compute_community_profile()` in
   `scripts/generate_stats_snapshot.py`. Suppress counts < 11 (`_MIN_CELL`).
2. Add an accessor to `api/stats.py` (see `get_community_profile`).
3. Render it in `pages/0_Community_Snapshot.py`, and on the dashboard in
   `app.py` if an external reader should find it without clicking through.
4. Regenerate: `.venv/bin/python scripts/generate_stats_snapshot.py`
5. Smoke-test every page (see below), then commit and push to `main`.

**Report a rate against the population it describes, not just the registry.**
ALS is ~48% of MOVR, is adult-onset and is predominantly Medicare-covered, so
a single study-wide percentage is driven by ALS and describes the other six
conditions poorly. Medicaid is 22.3% study-wide but 38.1% excluding ALS and
44.3% in DMD/BMD/SMA. Publish the groupings together.

### Smoke-test every page

Renders all 13 pages and surfaces exceptions. Faster and more reliable than
clicking through the app:

```bash
OPENMOVR_SITE_KEY=testkey .venv/bin/python - <<'EOF'
from streamlit.testing.v1 import AppTest
import glob
for t in ["app.py"] + sorted(glob.glob("pages/*.py")):
    at = AppTest.from_file(t, default_timeout=200); at.run()
    print(t, "OK" if not at.exception and not at.error else "FAIL")
    for e in at.exception: print("  !!", str(e.value)[:160])
EOF
```

Without `OPENMOVR_SITE_KEY` the six DUA pages fail on missing secrets. That
is environmental, not a real failure.

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

## Gotchas

- **Use `.venv/bin/python`.** System python has no pandas or streamlit.
- **Cohort switching is safe, but know how it behaves.** `get_base_cohort()`
  caches keyed on `include_usndr`, so switching between MOVR-only (n=3,444)
  and MOVR+USNDR (n=6,021) rebuilds and clears the derived disease caches.
  Internal helpers use `_current_base_cohort()`, which keeps working against
  whichever cohort you loaded rather than reverting to the default. Switching
  back and forth in a loop will reload the data each time.
- **`pgeocode` is not installed and is not in `requirements.txt`.** Snapshot
  regeneration falls back to coordinates already in the committed snapshot
  (56/60 sites). If you ever regenerate with an empty `stats/`, the site map
  goes blank.
- **Deploy target is `main`.** Streamlit Cloud auto-deploys from `main` only.
  The local checkout has historically sat on `feature/longitudinal-snapshots`
  -- check `git branch --show-current` before committing, or the push lands
  on GitHub without changing the live app.
- **`OpenMOVR/openmovr-app` is public.** Pushing publishes. `data/*.parquet`
  is gitignored; `stats/*.json` is aggregated and safe.

## Key Cohort Numbers

From the 2025-03-25 data extract, MOVR-only validated cohort:

| | |
|---|---|
| Participants | 3,444 (MOVR+USNDR: 6,021) |
| Facilities | 60 |
| Insurance responders | 3,300 (144 answered only Unknown/Not Reported) |
| Medicaid | 22.3% all / 38.1% excl. ALS / 44.3% DMD+BMD+SMA / 5.2% ALS |
| Medicaid by age at enrollment | 47.1% under 19 / 14.8% 19-64 / 3.8% 65+ |
| Any public coverage | 56.2% (Medicaid, Medicare or VA/military) |

`hltin` is a multi-select, so categories sum to over 100%. Percentages use
participants giving an informative answer as the denominator, excluding those
answering only "Unknown" or "Not Reported".

## Related Repos

- `../movr-clinical-analytics` -- the main MOVR 1.0 analysis and the
  export/tokenization pipeline. This app was extracted from it; that repo
  keeps `webapp/README.md` as a pointer. Its `CLAUDE.md` forbids Claude
  attribution in commits and changelogs; this repo does not.
- `../openmovr.github.io` -- the public docs and pilot site.

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

### Session 3: Community Snapshot & Citable Provenance (Sep 2026)

Driver: MDA Access Policy needed a publicly citable Medicaid figure for a
National Health Law Program amicus brief on Medicaid work requirements. The
figure they wanted to cite (~40% of the community on Medicaid) was not
supported study-wide -- MOVR is 22.3% -- but is 38.1% excluding ALS, which
is the grouping that resolved it.

- `pages/0_Community_Snapshot.py`: first public page; study-wide
  demographics, insurance (overall / by disease / by disease group / by age
  band), minimum diagnosis, methods tab, suggested citation
- `community_profile` added to `database_snapshot.json`; `_MIN_CELL` = 11
- Dashboard "Community and Coverage" block linking through to it
- Prototype banner replaced with a provenance banner carrying the data
  extract date, statistics generation date and app version
- v0.3.0, `CHANGELOG.md` added, About page keeps prior releases
- Fixed: snapshot generator no longer wipes site coordinates without
  pgeocode; sidebar CSS no longer hardcodes the version

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
- Branch: `main` (Streamlit Cloud auto-deploys on push to `main`)
- Live: `openmovr-app.streamlit.app` (custom domain `app.openmovr.io` optional, see DEPLOYMENT.md)
- Desired URL: `app.openmovr.io` or `openmovr-app.streamlit.app`

## Testing the App

```bash
cd /home/andre/MDA/openmovr-app
pip install -r requirements.txt
export OPENMOVR_SITE_KEY="your-key"  # for DUA pages
streamlit run app.py
```

## File Locations

| Purpose | Location |
|---------|----------|
| Main app | `app.py` |
| Pages | `pages/*.py` (12 pages) |
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
