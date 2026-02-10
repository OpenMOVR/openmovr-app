#!/usr/bin/env python3
"""
Generate Curated Data Dictionary Snapshot

Reads the raw datadictionary.parquet, classifies every field into a clinical
domain using rules from config/clinical_domains.yaml, fixes known mislabelings,
and outputs stats/curated_dictionary.json.

Usage:
    python scripts/generate_curated_dictionary.py
    python scripts/generate_curated_dictionary.py --output stats/curated_dictionary.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DISEASE_COLUMNS = ["ALS", "BMD", "DMD", "SMA", "LGMD", "FSHD", "Pompe"]


def load_config(config_path: Path) -> dict:
    """Load clinical domain classification config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_dictionary(parquet_path: Path) -> pd.DataFrame:
    """Load the raw data dictionary parquet."""
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} fields from {parquet_path.name}")
    return df


def apply_mislabel_corrections(df: pd.DataFrame, corrections: list) -> int:
    """Fix mislabeled disease assignments. Returns count of corrections."""
    count = 0
    for fix in corrections:
        field = fix["field_name"]
        remove_col = fix["remove_from"]
        add_col = fix["add_to"]
        mask = df["Field Name"] == field
        if mask.any():
            df.loc[mask, remove_col] = ""
            df.loc[mask, add_col] = "Yes"
            count += mask.sum()
    return count


def normalize_disease_columns(df: pd.DataFrame) -> None:
    """Convert disease columns from 'Yes'/'' to True/False."""
    for col in DISEASE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: True if str(x).strip() == "Yes" else False)


def classify_field(row: pd.Series, config: dict) -> str:
    """
    Classify a single field into a clinical domain.

    Priority:
      1. Non-Main-Data Excel Tab → excel_tab_domains
      2. Main Data + Demographics/Discontinuation → form_domains
      3. Main Data + Diagnosis → diagnosis_prefix_domains
      4. Main Data + Encounter/Log → keyword_rules (first match wins)
      5. Fallback → "Unclassified"
    """
    excel_tab = str(row.get("Excel Tab", "")).strip()
    file_form = str(row.get("File/Form", "")).strip()
    field_name = str(row.get("Field Name", "")).strip()
    description = str(row.get("Description", "")).strip()

    # 1. Non-Main-Data Excel Tab
    if excel_tab != "Main Data":
        domain = config.get("excel_tab_domains", {}).get(excel_tab)
        if domain:
            return domain

    # 2. Form-level defaults (Demographics, Discontinuation)
    form_domain = config.get("form_domains", {}).get(file_form)
    if form_domain:
        return form_domain

    # 3. Diagnosis form → disease prefix matching
    if file_form == "Diagnosis":
        for rule in config.get("diagnosis_prefix_domains", []):
            prefix = rule["prefix"]
            if field_name.startswith(prefix) or field_name.lower().startswith(prefix.lower()):
                return rule["domain"]
        # Fallback for unmatched diagnosis fields
        return "Disease Classification & Diagnosis"

    # 4. Encounter/Log → keyword rules (first match wins)
    if file_form in ("Encounter", "Log"):
        for rule in config.get("keyword_rules", []):
            match_type = rule["match"]
            value = rule["value"]

            if match_type == "field_exact":
                if field_name == value:
                    return rule["domain"]
            elif match_type == "field_prefix":
                if field_name.startswith(value) or field_name.lower().startswith(value.lower()):
                    return rule["domain"]
            elif match_type == "desc_contains":
                if value in description:
                    return rule["domain"]

    return "Unclassified"


def classify_all_fields(df: pd.DataFrame, config: dict) -> pd.Series:
    """Classify all fields, return Series of domain labels."""
    return df.apply(lambda row: classify_field(row, config), axis=1)


def build_domain_summary(df: pd.DataFrame) -> list:
    """Build domain × disease matrix for overview.

    field_count uses unique Field Name values (not rows) so that fields
    appearing in multiple forms (Demographics, Encounter, Log, etc.) are
    only counted once.
    """
    summary = []
    for domain in df["Clinical Domain"].unique():
        domain_df = df[df["Clinical Domain"] == domain]
        entry = {
            "domain": domain,
            "field_count": int(domain_df["Field Name"].nunique()),
        }
        for col in DISEASE_COLUMNS:
            if col in df.columns:
                # Count unique field names where disease flag is True
                ds_fields = domain_df.loc[domain_df[col] == True, "Field Name"]
                entry[col] = int(ds_fields.nunique())
        summary.append(entry)

    # Sort by canonical order if available
    return summary


def build_snapshot(df: pd.DataFrame, config: dict, corrections_applied: int) -> dict:
    """Build the full JSON snapshot."""
    canonical = config.get("canonical_domains", [])

    # Sort domain summary by canonical order
    domain_summary = build_domain_summary(df)
    domain_order = {d: i for i, d in enumerate(canonical)}
    domain_summary.sort(key=lambda x: domain_order.get(x["domain"], 999))

    # Build fields list
    fields = []
    for _, row in df.iterrows():
        field = {
            "field_name": row["Field Name"],
            "description": row["Description"],
            "display_label": row.get("Display Label", ""),
            "file_form": row["File/Form"],
            "excel_tab": row["Excel Tab"],
            "clinical_domain": row["Clinical Domain"],
            "field_type": row.get("Field Type", ""),
            "numeric_ranges": row.get("Numeric Ranges", ""),
            "is_required": "*" in str(row.get("Display Label", "")),
        }
        for col in DISEASE_COLUMNS:
            if col in df.columns:
                field[col.lower()] = bool(row[col])
        fields.append(field)

    # Classification stats
    total = len(df)
    classified = len(df[df["Clinical Domain"] != "Unclassified"])
    unclassified = total - classified

    snapshot = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "data/datadictionary.parquet",
            "config": "config/clinical_domains.yaml",
            "total_fields": total,
            "classified_fields": classified,
            "unclassified_fields": unclassified,
            "classification_rate": round(classified / total * 100, 1) if total else 0,
            "mislabel_corrections_applied": corrections_applied,
            "domain_count": len([d for d in df["Clinical Domain"].unique() if d != "Unclassified"]),
        },
        "canonical_domains": canonical,
        "domain_summary": domain_summary,
        "fields": fields,
    }

    return snapshot


def print_qa_report(df: pd.DataFrame, snapshot: dict):
    """Print QA report for review."""
    meta = snapshot["metadata"]

    print("\n" + "=" * 70)
    print("CURATED DATA DICTIONARY — QA REPORT")
    print("=" * 70)

    print(f"\n  Total fields:       {meta['total_fields']}")
    print(f"  Classified:         {meta['classified_fields']}")
    print(f"  Unclassified:       {meta['unclassified_fields']}")
    print(f"  Classification %:   {meta['classification_rate']}%")
    print(f"  Corrections applied: {meta['mislabel_corrections_applied']}")
    print(f"  Domains used:       {meta['domain_count']}")

    print("\n  DOMAIN DISTRIBUTION:")
    print(f"  {'Domain':<40} {'Count':>6}")
    print("  " + "-" * 48)
    for entry in snapshot["domain_summary"]:
        print(f"  {entry['domain']:<40} {entry['field_count']:>6}")

    # Show unclassified fields
    unclassified = df[df["Clinical Domain"] == "Unclassified"]
    if not unclassified.empty:
        print(f"\n  WARNING: {len(unclassified)} UNCLASSIFIED FIELDS:")
        for _, row in unclassified.iterrows():
            print(f"    {row['Field Name']}: {row['Description'][:60]} [{row['File/Form']}]")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Generate curated data dictionary snapshot")
    parser.add_argument("--output", "-o",
                        default="stats/curated_dictionary.json",
                        help="Output file path")
    parser.add_argument("--parquet", "-p",
                        default="data/datadictionary.parquet",
                        help="Raw data dictionary parquet path")
    parser.add_argument("--config", "-c",
                        default="config/clinical_domains.yaml",
                        help="Clinical domains config path")
    parser.add_argument("--quiet", "-q", action="store_true")

    args = parser.parse_args()

    parquet_path = PROJECT_ROOT / args.parquet
    config_path = PROJECT_ROOT / args.config
    output_path = PROJECT_ROOT / args.output

    if not parquet_path.exists():
        print(f"  Error: {parquet_path} not found")
        return 1
    if not config_path.exists():
        print(f"  Error: {config_path} not found")
        return 1

    print("Generating curated data dictionary...")

    # Load
    config = load_config(config_path)
    df = load_raw_dictionary(parquet_path)

    # Fix mislabels
    corrections = apply_mislabel_corrections(
        df, config.get("mislabeled_corrections", [])
    )
    print(f"  Applied {corrections} mislabel corrections")

    # Normalize disease columns
    normalize_disease_columns(df)

    # Classify
    df["Clinical Domain"] = classify_all_fields(df, config)

    # Build snapshot
    snapshot = build_snapshot(df, config, corrections)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\n  Saved to: {output_path}")

    # QA
    if not args.quiet:
        print_qa_report(df, snapshot)

    return 0


if __name__ == "__main__":
    sys.exit(main())
