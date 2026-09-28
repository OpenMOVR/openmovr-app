# Changelog

Releases of the OpenMOVR App. Figures cited from this app should name the
version they were read from; each release below states what changed.

Dates are release dates. They are not the date of the underlying data — every
page carries the data extract date in its banner, and that date changes only
when the registry data cut is refreshed.

## v0.3.0 — 2026-09-28

### Added

- **Community Snapshot page** (`pages/0_Community_Snapshot.py`), the first
  public page, covering study-wide demographics, health insurance and the
  minimum diagnosis elements collected comparably across all seven disease
  types. Previously insurance was reported only per disease, so no
  community-level coverage figure could be cited from a public source.
- Health insurance reported four ways: overall, by disease, by named disease
  group, and by age band at enrollment. ALS is roughly half the registry, is
  adult-onset and is predominantly Medicare-covered, so a single study-wide
  percentage is driven by ALS and describes the other six conditions poorly.
  Reporting the groupings side by side is what makes a cited figure
  unambiguous.
- Methods tab stating the source field, the denominator rule, multi-select
  handling, small-cell suppression and the point-in-time nature of the data.
- Suggested citation, with worked examples showing that a figure cited
  without its population is not supported by the page.
- `community_profile` section in `stats/database_snapshot.json`, exposed
  through `StatsAPI.get_community_profile()` and
  `StatsAPI.get_insurance_profile()`.
- CSV and JSON downloads for each breakdown.

### Changed

- The prototype banner is now a data provenance banner. It states three
  dates that a citation needs and keeps them distinct: the data extract
  (what the numbers describe), the statistics generation date (when the
  aggregates were computed) and the app version. The previous banner led
  with "Proof-of-Concept Prototype", which read as a caveat on the data
  rather than on the interface. The development caveat remains, scoped to
  the interface.
- Version badges keep the version, which a citation needs, and drop
  "(Prototype)". The per-feature maturity table on the About page is
  unchanged.
- The About page version history now retains prior releases, so a figure
  cited against a given version can be identified later.

### Fixed

- The snapshot generator silently wrote null coordinates for every site when
  `pgeocode` was unavailable, which empties the Facility View map.
  `pgeocode` is an optional dependency and needs network access on first
  use, so this was easy to trip into. Coordinates already in the committed
  snapshot are now reused as a fallback.
- The app version in the sidebar CSS was hardcoded and had to be updated by
  hand alongside `config/settings.py`. It is now substituted from
  `APP_VERSION`.
- Timestamped snapshot backups written by the generator are gitignored.

## v0.2.0 — 2026-02-10

### Added

- Public dashboard with aggregated enrollment, disease distribution and
  longitudinal metrics.
- Disease Explorer with demographic breakdowns, diagnosis profiles and
  Clinical Summary Preview.
- Clinical Analytics for DMD, LGMD, ALS and SMA, organized by clinical
  domain (DUA required).
- Curated Data Dictionary covering 1,024 fields across 19 clinical domains.
- Anonymized site map with per-disease filtering across 60+ participating
  sites.
- Disease-specific therapies tracking (gene therapy, antisense, ERT,
  disease-modifying).
- Cumulative and monthly enrollment charts, per disease and across the
  registry.
- Site Analytics with site-vs-overall comparisons (DUA required).
- Download Center for data exports (DUA required).
- Facility View with top-site rankings and geographic distribution.
