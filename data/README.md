# Data Directory

This directory is for MOVR parquet data files.

## Files NOT included in git

Parquet files are excluded via `.gitignore` because they may contain PHI.

## Required files for Live Mode

To run the app in Live Mode (full interactivity), place these parquet files here:

- `demographics_maindata.parquet`
- `diagnosis_maindata.parquet`
- `encounter_maindata.parquet`
- `log_maindata.parquet`
- `datadictionary.parquet`

Plus any additional tables you need (medications, surgeries, etc.)

## Obtaining Data

Contact the MDA MOVR team for data access:
- Email: aparedes@mdausa.org

## Snapshot Mode

Without parquet files, the app runs in Snapshot Mode using pre-computed 
statistics from `stats/database_snapshot.json` and `stats/lgmd_snapshot.json`.
