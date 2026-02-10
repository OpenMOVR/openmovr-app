"""
Structured Chatbot Engine for Ask OpenMOVR.

No LLM -- uses keyword matching to route questions to pre-built
response handlers that pull from snapshot JSON files.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st

# ── Snapshot Loader ─────────────────────────────────────────────

_STATS_DIR = Path(__file__).parent.parent / "stats"


@st.cache_data(ttl=300)
def _load_all_snapshots() -> Dict:
    snapshots = {}
    _FILES = {
        "database": "database_snapshot.json",
        "als": "als_snapshot.json",
        "dmd": "dmd_snapshot.json",
        "sma": "sma_snapshot.json",
        "lgmd": "lgmd_snapshot.json",
        "dictionary": "curated_dictionary.json",
    }
    for key, fname in _FILES.items():
        path = _STATS_DIR / fname
        if path.exists():
            with open(path) as f:
                snapshots[key] = json.load(f)
    return snapshots


# ── Question Template ───────────────────────────────────────────

@dataclass
class QuestionTemplate:
    id: str
    category: str
    display_text: str
    # List of OR-groups; ALL groups must have >= 1 match
    required_keywords: List[List[str]]
    boost_keywords: List[str]
    # None = registry-wide, "inferred" = from user input
    disease: Optional[str]
    handler: Callable


# ── Disease Aliases ─────────────────────────────────────────────

_DISEASE_ALIASES = {
    "amyotrophic lateral sclerosis": "ALS",
    "amyotrophic": "ALS",
    "als": "ALS",
    "duchenne": "DMD",
    "dmd": "DMD",
    "spinal muscular atrophy": "SMA",
    "sma": "SMA",
    "limb-girdle": "LGMD",
    "limb girdle": "LGMD",
    "lgmd": "LGMD",
    "becker": "BMD",
    "bmd": "BMD",
    "facioscapulohumeral": "FSHD",
    "fshd": "FSHD",
    "pompe": "Pompe",
}

# Map disease code to snapshot key
_DISEASE_SNAP_KEY = {
    "ALS": "als",
    "DMD": "dmd",
    "SMA": "sma",
    "LGMD": "lgmd",
}


# ── Question Matcher ────────────────────────────────────────────

class QuestionMatcher:
    def __init__(self, templates: List[QuestionTemplate]):
        self.templates = templates

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^\w\s-]", "", text.lower().strip())

    @staticmethod
    def _extract_disease(text: str) -> Optional[str]:
        lower = text.lower()
        # Check longest aliases first to avoid partial matches
        for alias in sorted(_DISEASE_ALIASES, key=len, reverse=True):
            if alias in lower:
                return _DISEASE_ALIASES[alias]
        return None

    def match(
        self, user_input: str
    ) -> Tuple[Optional[QuestionTemplate], Optional[str]]:
        normalized = self._normalize(user_input)
        disease = self._extract_disease(user_input)

        best: Optional[QuestionTemplate] = None
        best_score = 0

        for tmpl in self.templates:
            # Disease constraint: skip if user mentioned a DIFFERENT disease,
            # but allow when user didn't mention any (keywords are specific enough)
            if tmpl.disease and tmpl.disease != "inferred":
                if disease and disease != tmpl.disease:
                    continue

            # Required keyword groups (OR within, AND across)
            all_met = True
            for or_group in tmpl.required_keywords:
                if not any(kw in normalized for kw in or_group):
                    all_met = False
                    break
            if not all_met:
                continue

            score = len(tmpl.required_keywords)
            for bkw in tmpl.boost_keywords:
                if bkw in normalized:
                    score += 1
            if tmpl.disease == "inferred" and disease:
                score += 2

            if score > best_score:
                best_score = score
                best = tmpl

        if best_score < 2:
            return None, disease
        # Use template's hard-coded disease as default when user didn't specify
        if not disease and best.disease and best.disease != "inferred":
            disease = best.disease
        return best, disease


# ── Response Handlers ───────────────────────────────────────────

def _src() -> str:
    return "\n\n*Source: MOVR registry snapshot (aggregated statistics, no individual data)*"


# --- Registry Overview ---

def _handle_total_participants(kb: Dict, disease: Optional[str]) -> str:
    db = kb["database"]
    total = db["enrollment"]["total_patients"]
    fac = db["facilities"]["total_facilities"]
    diseases = db["disease_distribution"]["diseases"]
    lines = [
        f"The MOVR registry includes **{total:,} participants** with validated "
        f"enrollment across **{fac} sites** and **{len(diseases)} disease types**.",
        "",
        "Disease breakdown:",
    ]
    for d in sorted(diseases, key=lambda x: -x["patient_count"]):
        lines.append(f"- {d['disease']}: {d['patient_count']:,} ({d['percentage']}%)")
    lines.append(_src())
    return "\n".join(lines)


def _handle_diseases_covered(kb: Dict, disease: Optional[str]) -> str:
    diseases = kb["database"]["disease_distribution"]["diseases"]
    names = [f"**{d['disease']}** ({d['patient_count']:,})" for d in sorted(diseases, key=lambda x: -x["patient_count"])]
    return (
        f"MOVR covers **{len(diseases)} neuromuscular diseases**: "
        + ", ".join(names)
        + "."
        + _src()
    )


def _handle_site_count(kb: Dict, disease: Optional[str]) -> str:
    fac = kb["database"]["facilities"]["total_facilities"]
    locs = kb["database"]["facilities"].get("site_locations", [])
    states = set(s["state"] for s in locs if s.get("state"))
    return (
        f"There are **{fac} participating sites** across "
        f"**{len(states)} states** in the United States."
        + _src()
    )


def _handle_encounter_count(kb: Dict, disease: Optional[str]) -> str:
    lng = kb["database"].get("longitudinal", {})
    total_enc = lng.get("total_encounters", 0)
    pts = lng.get("patients_with_encounters", 0)
    mean = lng.get("mean_encounters_per_patient", 0)
    median = lng.get("median_encounters_per_patient", 0)
    return (
        f"The registry contains **{total_enc:,} encounter records** across "
        f"{pts:,} participants.\n\n"
        f"- Mean visits per participant: {mean}\n"
        f"- Median visits per participant: {median}\n"
        f"- Participants with 3+ visits: {lng.get('patients_3plus_encounters', 0):,}"
        + _src()
    )


# --- Disease Counts (parametric) ---

def _handle_disease_count(kb: Dict, disease: Optional[str]) -> str:
    if not disease:
        return _handle_total_participants(kb, disease)
    summary = kb["database"]["disease_distribution"].get("disease_summary", {})
    ds = summary.get(disease, {})
    if not ds:
        return f"I don't have specific count data for **{disease}** in the registry."
    total = kb["database"]["enrollment"]["total_patients"]
    return (
        f"There are **{ds['count']:,} {disease} participants** in the registry, "
        f"representing **{ds['percentage']}%** of the total {total:,} enrollment."
        + _src()
    )


# --- Demographics ---

def _handle_gender(kb: Dict, disease: Optional[str]) -> str:
    if not disease:
        return "Please specify a disease (e.g., \"What is the gender breakdown for ALS?\")."
    profile = kb["database"].get("disease_profiles", {}).get(disease, {})
    gender = profile.get("demographics", {}).get("gender", [])
    if not gender:
        return f"Gender data is not available for {disease} in the snapshot."
    lines = [f"**{disease} gender distribution:**", ""]
    total = sum(g["count"] for g in gender)
    for g in gender:
        pct = round(g["count"] / total * 100, 1) if total else 0
        lines.append(f"- {g['label']}: {g['count']:,} ({pct}%)")
    lines.append(_src())
    return "\n".join(lines)


def _handle_age_distribution(kb: Dict, disease: Optional[str]) -> str:
    if not disease:
        return "Please specify a disease (e.g., \"What is the age distribution for DMD?\")."
    profile = kb["database"].get("disease_profiles", {}).get(disease, {})
    age_data = profile.get("demographics", {}).get("age_at_enrollment", [])
    if not age_data:
        return f"Age distribution data is not available for {disease} in the snapshot."
    lines = [f"**{disease} age at enrollment:**", ""]
    for a in age_data:
        lines.append(f"- {a['label']}: {a['count']:,}")
    lines.append(_src())
    return "\n".join(lines)


# --- ALS-Specific ---

def _handle_alsfrs(kb: Dict, disease: Optional[str]) -> str:
    als = kb.get("als", {})
    alsfrs = als.get("alsfrs", {})
    if not alsfrs.get("available"):
        return "ALSFRS-R data is not available in the ALS snapshot."
    ts = alsfrs["total_score"]
    sev = alsfrs.get("severity_bands", [])
    lines = [
        f"The median ALSFRS-R total score is **{ts['median']}** "
        f"(mean {ts['mean']}, range {ts['min']}-{ts['max']}) "
        f"based on **{ts['count']:,} participants**.",
        "",
        "Severity distribution:",
    ]
    for b in sev:
        if not b.get("suppressed"):
            lines.append(f"- {b['label']}: {b.get('count', 0):,} ({b.get('percentage', 0)}%)")
    lines.append(_src())
    return "\n".join(lines)


def _handle_als_medications(kb: Dict, disease: Optional[str]) -> str:
    meds = kb.get("als", {}).get("medications", {})
    if not meds.get("available"):
        return "ALS medication data is not available in the snapshot."
    als_drugs = meds.get("als_drugs", {})
    lines = ["**ALS disease-modifying and symptomatic medications:**", ""]
    for name, data in als_drugs.items():
        if data.get("suppressed"):
            lines.append(f"- {name}: <11 participants (suppressed)")
        else:
            lines.append(f"- {name}: **{data['count']:,}** participants ({data['percentage']}%)")
    lines.append(_src())
    return "\n".join(lines)


def _handle_el_escorial(kb: Dict, disease: Optional[str]) -> str:
    dx = kb.get("als", {}).get("diagnosis", {})
    ee = dx.get("el_escorial", {})
    if not ee:
        return "El Escorial classification data is not available."
    dist = ee.get("distribution", {})
    lines = [
        f"**El Escorial classification** ({ee.get('total_reported', 0):,} participants classified):",
        "",
    ]
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {count:,}")
    lines.append(_src())
    return "\n".join(lines)


def _handle_als_body_region(kb: Dict, disease: Optional[str]) -> str:
    dx = kb.get("als", {}).get("diagnosis", {})
    br = dx.get("body_region_onset", {})
    if not br:
        return "Body region onset data is not available."
    dist = br.get("distribution", {})
    lines = [
        f"**ALS body region of onset** ({br.get('total_reported', 0):,} participants):",
        "",
    ]
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {count:,}")
    lines.append(_src())
    return "\n".join(lines)


def _handle_als_fvc(kb: Dict, disease: Optional[str]) -> str:
    resp = kb.get("als", {}).get("respiratory", {})
    fvc = resp.get("fvc_pct", {})
    if not fvc:
        return "ALS FVC data is not available."
    return (
        f"**ALS FVC % Predicted**: median **{fvc['median']}%**, "
        f"mean {fvc['mean']}%, range {fvc['min']}-{fvc['max']}%, "
        f"based on **{fvc['count']:,} participants**."
        + _src()
    )


def _handle_als_onset_age(kb: Dict, disease: Optional[str]) -> str:
    ms = kb.get("als", {}).get("milestones", {})
    onset = ms.get("onset_age", {})
    dx_age = ms.get("diagnosis_age", {})
    delay = ms.get("diagnostic_delay", {})
    lines = ["**ALS disease milestones:**", ""]
    if onset:
        lines.append(f"- Symptom onset age: median **{onset['median']} years** (mean {onset['mean']}, n={onset['count']:,})")
    if dx_age:
        lines.append(f"- Diagnosis age: median **{dx_age['median']} years** (mean {dx_age['mean']}, n={dx_age['count']:,})")
    if delay:
        lines.append(f"- Diagnostic delay: median **{delay['median']} years** (mean {delay['mean']}, n={delay['count']:,})")
    lines.append(_src())
    return "\n".join(lines)


# --- DMD-Specific ---

def _handle_dmd_therapeutics(kb: Dict, disease: Optional[str]) -> str:
    tx = kb.get("dmd", {}).get("therapeutics", {})
    if not tx:
        return "DMD therapeutics data is not available."
    drugs = tx.get("drugs", [])
    lines = ["**DMD exon-skipping therapeutics:**", ""]
    for d in drugs:
        name = d.get("drug_name", "")
        total_amenable = d.get("total_amenable", {})
        on_therapy = d.get("on_therapy", {})
        amenable_ct = total_amenable.get("count", 0) if isinstance(total_amenable, dict) else total_amenable
        on_ct = on_therapy.get("count", 0) if isinstance(on_therapy, dict) else on_therapy
        suppressed = on_therapy.get("suppressed", False) if isinstance(on_therapy, dict) else False
        if suppressed:
            lines.append(f"- {name}: {amenable_ct} amenable, <11 on therapy (suppressed)")
        else:
            lines.append(f"- {name}: {amenable_ct} amenable, **{on_ct} on therapy**")
    lines.append(_src())
    return "\n".join(lines)


def _handle_dmd_genetics(kb: Dict, disease: Optional[str]) -> str:
    gen = kb.get("dmd", {}).get("genetics", {})
    if not gen.get("available"):
        return "DMD genetics data is not available."
    conf = gen.get("genetic_confirmation", {})
    mut = gen.get("mutation_type", {})
    lines = ["**DMD genetics:**", ""]
    if conf:
        lines.append(f"- Genetic confirmation: **{conf.get('confirmed_percentage', 0)}%** ({conf.get('confirmed_count', 0):,} participants)")
    if mut:
        dist = mut.get("distribution", [])
        for entry in dist:
            if not entry.get("suppressed"):
                lines.append(f"- {entry['label']}: {entry['count']:,}")
    lines.append(_src())
    return "\n".join(lines)


def _handle_dmd_steroids(kb: Dict, disease: Optional[str]) -> str:
    steroids = kb.get("dmd", {}).get("steroids", {})
    if not steroids.get("available"):
        return "DMD steroid data is not available."
    first = steroids.get("glcouse_first_encounter", {}).get("distribution", {})
    lines = ["**DMD glucocorticoid use (at first encounter):**", ""]
    for label, count in sorted(first.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {count:,}")
    lines.append(_src())
    return "\n".join(lines)


def _handle_dmd_ambulatory(kb: Dict, disease: Optional[str]) -> str:
    amb = kb.get("dmd", {}).get("ambulatory", {})
    if not amb.get("available", False):
        return "DMD ambulatory status data is not available."
    dist = amb.get("current_status", {}).get("distribution", {})
    lines = ["**DMD ambulatory status (latest encounter):**", ""]
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {count:,}")
    lines.append(_src())
    return "\n".join(lines)


# --- SMA-Specific ---

def _handle_sma_types(kb: Dict, disease: Optional[str]) -> str:
    cls = kb.get("sma", {}).get("classification", {})
    if not cls:
        return "SMA classification data is not available."
    sma_type = cls.get("sma_type", {})
    dist = sma_type.get("distribution", {})
    lines = [
        f"**SMA type distribution** ({sma_type.get('total_reported', 0):,} participants classified):",
        "",
    ]
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {label}: {count:,}")
    lines.append(_src())
    return "\n".join(lines)


def _handle_sma_therapies(kb: Dict, disease: Optional[str]) -> str:
    tx = kb.get("sma", {}).get("therapeutics", {})
    if not tx:
        return "SMA therapeutics data is not available."
    sma_drugs = tx.get("sma_drugs", {})
    total = tx.get("total_on_therapy", {})
    lines = ["**SMA disease-modifying therapies:**", ""]
    for name, data in sma_drugs.items():
        if data.get("suppressed"):
            lines.append(f"- {name}: <11 participants (suppressed)")
        else:
            lines.append(f"- {name}: **{data['count']:,}** participants ({data['percentage']}%)")
    if total:
        lines.append(f"\nTotal on any therapy: **{total['count']:,}** ({total['percentage']}%)")
    lines.append(_src())
    return "\n".join(lines)


def _handle_smn2(kb: Dict, disease: Optional[str]) -> str:
    gen = kb.get("sma", {}).get("genetics", {})
    if not gen:
        return "SMA genetics data is not available."
    smn2 = gen.get("smn2_copy_number", {})
    dist = smn2.get("distribution", {})
    if not dist:
        return "SMN2 copy number data is not available."
    lines = [
        f"**SMN2 copy number distribution** ({smn2.get('total_reported', 0):,} participants):",
        "",
    ]
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        lines.append(f"- {label} copies: {count:,}")
    conf = gen.get("genetic_confirmation", {})
    if conf:
        lines.append(f"\nGenetic confirmation rate: **{conf.get('confirmed_pct', 0)}%**")
    lines.append(_src())
    return "\n".join(lines)


def _handle_sma_motor(kb: Dict, disease: Optional[str]) -> str:
    ms = kb.get("sma", {}).get("motor_scores", {})
    if not ms:
        return "SMA motor score data is not available."
    lines = ["**SMA motor function scores:**", ""]
    for key, label in [("hfmse", "HFMSE (0-66, Type 2/3)"), ("chop_intend", "CHOP-INTEND (0-64, Type 1)"), ("rulm", "RULM (0-37)")]:
        score = ms.get(key, {})
        stats = score.get("total_score", {})
        if stats and stats.get("count", 0) > 0:
            lines.append(f"- {label}: median **{stats['median']}**, mean {stats['mean']}, n={stats['count']:,}")
    lines.append(_src())
    return "\n".join(lines)


# --- LGMD-Specific ---

def _handle_lgmd_subtypes(kb: Dict, disease: Optional[str]) -> str:
    sub = kb.get("lgmd", {}).get("subtypes", {})
    if not sub.get("available"):
        return "LGMD subtype data is not available."
    dist = sub.get("distribution", [])
    total = sum(d.get("patients", 0) for d in dist) if dist else 0
    lines = [
        f"**{sub.get('unique_subtypes', 0)} LGMD subtypes** identified "
        f"({total:,} participants classified):",
        "",
    ]
    for d in sorted(dist, key=lambda x: -x.get("patients", 0))[:10]:
        lines.append(f"- {d['subtype']}: {d['patients']:,} ({d.get('percentage', 0)}%)")
    if len(dist) > 10:
        lines.append(f"- ... and {len(dist) - 10} more subtypes")
    lines.append(_src())
    return "\n".join(lines)


def _handle_lgmd_delay(kb: Dict, disease: Optional[str]) -> str:
    dj = kb.get("lgmd", {}).get("diagnostic_journey", {})
    delay = dj.get("delay", {})
    if not delay:
        return "LGMD diagnostic delay data is not available."
    return (
        f"**LGMD diagnostic delay**: median **{delay.get('median', '?')} years** "
        f"(mean {delay.get('mean', '?')} years, range {delay.get('min', '?')}-{delay.get('max', '?')} years, "
        f"n={delay.get('count', 0):,}).\n\n"
        f"The delay represents the time between symptom onset and confirmed diagnosis."
        + _src()
    )


def _handle_lgmd_medications(kb: Dict, disease: Optional[str]) -> str:
    meds = kb.get("lgmd", {}).get("medications", {})
    if not meds.get("available"):
        return "LGMD medication data is not available."
    cats = meds.get("categories", {})
    lines = [
        f"**{meds.get('total_patients_with_meds', 0):,} LGMD participants** with medication data.",
        "",
        "Medication categories:",
    ]
    for cat, data in sorted(cats.items(), key=lambda x: -x[1].get("patients", 0)):
        lines.append(f"- {cat}: {data['patients']:,} participants ({data['percentage']}%)")
    top = meds.get("top_drugs", [])
    if top:
        lines.append("\nTop medications:")
        for d in top[:5]:
            lines.append(f"- {d['drug']}: {d['patients']:,} participants")
    lines.append(_src())
    return "\n".join(lines)


# --- Data Dictionary ---

def _handle_field_count(kb: Dict, disease: Optional[str]) -> str:
    meta = kb.get("dictionary", {}).get("metadata", {})
    domains = kb.get("dictionary", {}).get("canonical_domains", [])
    return (
        f"MOVR captures **{meta.get('total_fields', 0):,} clinical fields** "
        f"across **{len(domains)} clinical domains**, classified at "
        f"{meta.get('classification_rate', 0)}% coverage."
        + _src()
    )


def _handle_clinical_domains(kb: Dict, disease: Optional[str]) -> str:
    summary = kb.get("dictionary", {}).get("domain_summary", [])
    if not summary:
        return "Clinical domain data is not available."
    lines = [
        f"**{len(summary)} clinical domains** in the MOVR data dictionary:",
        "",
    ]
    for d in summary:
        lines.append(f"- {d['domain']}: {d['field_count']} fields")
    lines.append(_src())
    return "\n".join(lines)


# ── Template Registry ───────────────────────────────────────────

def _build_templates() -> List[QuestionTemplate]:
    return [
        # Registry Overview
        QuestionTemplate("encounter_count", "Registry Overview",
            "How many encounters are in the registry?",
            [["encounter", "visit"],
             ["how many", "total", "count", "number", "registry", "data"]],
            ["longitudinal", "record", "registry"], None, _handle_encounter_count),
        QuestionTemplate("total_participants", "Registry Overview",
            "How many participants are in the registry?",
            [["how many", "total", "count", "number", "size"],
             ["participant", "patient", "people", "enrolled", "registry", "enrollment"]],
            ["registry", "movr", "total", "overall"], None, _handle_total_participants),
        QuestionTemplate("diseases_covered", "Registry Overview",
            "What diseases does MOVR cover?",
            [["disease", "diagnos", "condition", "cover"]],
            ["movr", "registry", "types", "which"], None, _handle_diseases_covered),
        QuestionTemplate("site_count", "Registry Overview",
            "How many sites participate?",
            [["site", "facilit", "center", "clinic", "hospital"]],
            ["how many", "count", "participate", "state"], None, _handle_site_count),

        # Disease Counts (parametric)
        QuestionTemplate("disease_count", "Disease Counts",
            "How many participants for a specific disease?",
            [["how many", "count", "number", "total"],
             ["participant", "patient", "people", "enrolled"]],
            ["registry"], "inferred", _handle_disease_count),

        # Demographics
        QuestionTemplate("gender", "Demographics",
            "What is the gender breakdown?",
            [["gender", "sex", "male", "female"]],
            ["breakdown", "distribution", "split"], "inferred", _handle_gender),
        QuestionTemplate("age_distribution", "Demographics",
            "What is the age distribution?",
            [["age"],
             ["distribution", "range", "breakdown", "histogram", "enrollment"]],
            [], "inferred", _handle_age_distribution),

        # ALS
        QuestionTemplate("alsfrs", "ALS",
            "What is the median ALSFRS-R score?",
            [["alsfrs", "als functional"]],
            ["score", "median", "severity"], "ALS", _handle_alsfrs),
        QuestionTemplate("als_meds", "ALS",
            "What ALS medications are tracked?",
            [["medication", "drug", "treatment", "therap", "riluzole", "radicava", "nuedexta"]],
            ["als", "tracked", "utilization"], "ALS", _handle_als_medications),
        QuestionTemplate("el_escorial", "ALS",
            "What is the El Escorial classification?",
            [["el escorial", "escorial", "classification"]],
            ["als", "criteria", "escorial", "classification"], "ALS", _handle_el_escorial),
        QuestionTemplate("als_body_region", "ALS",
            "What is the ALS body region of onset?",
            [["body region", "onset region", "bulbar", "limb onset"]],
            ["als"], "ALS", _handle_als_body_region),
        QuestionTemplate("als_fvc", "ALS",
            "What is the FVC for ALS participants?",
            [["fvc", "lung function", "pulmonary", "respiratory"]],
            ["als", "predicted", "percent"], "ALS", _handle_als_fvc),
        QuestionTemplate("als_onset", "ALS",
            "What is the ALS onset and diagnosis age?",
            [["onset", "diagnosis age", "diagnostic delay", "symptom"]],
            ["als", "age", "delay", "milestone"], "ALS", _handle_als_onset_age),

        # DMD
        QuestionTemplate("dmd_therapeutics", "DMD",
            "What DMD exon-skipping therapies are tracked?",
            [["exon", "therap", "treatment", "eteplirsen", "golodirsen", "casimersen", "viltolarsen"]],
            ["dmd", "duchenne", "skipping"], "DMD", _handle_dmd_therapeutics),
        QuestionTemplate("dmd_genetics", "DMD",
            "What is the DMD genetic confirmation rate?",
            [["genetic", "mutation", "confirmation", "dystrophin"]],
            ["dmd", "duchenne", "rate"], "DMD", _handle_dmd_genetics),
        QuestionTemplate("dmd_steroids", "DMD",
            "What steroids do DMD participants take?",
            [["steroid", "glucocorticoid", "deflazacort", "prednisone"]],
            ["dmd", "duchenne"], "DMD", _handle_dmd_steroids),
        QuestionTemplate("dmd_ambulatory", "DMD",
            "What is the ambulatory status for DMD?",
            [["ambulat", "walk", "wheelchair", "mobility"]],
            ["dmd", "duchenne", "status"], "DMD", _handle_dmd_ambulatory),

        # SMA
        QuestionTemplate("sma_types", "SMA",
            "What SMA types are in the registry?",
            [["sma", "spinal muscular"],
             ["type", "classification", "class"]],
            ["registry", "distribution"], "SMA", _handle_sma_types),
        QuestionTemplate("sma_therapies", "SMA",
            "What SMA therapies are tracked?",
            [["therap", "treatment", "drug", "medication", "spinraza", "zolgensma", "evrysdi", "nusinersen", "risdiplam"]],
            ["sma", "spinal muscular"], "SMA", _handle_sma_therapies),
        QuestionTemplate("smn2", "SMA",
            "What is the SMN2 copy number distribution?",
            [["smn2", "smn 2", "copy number", "genetic"]],
            ["sma", "distribution"], "SMA", _handle_smn2),
        QuestionTemplate("sma_motor", "SMA",
            "What are the SMA motor function scores?",
            [["motor", "hfmse", "chop-intend", "chop intend", "rulm", "function"]],
            ["sma", "score"], "SMA", _handle_sma_motor),

        # LGMD
        QuestionTemplate("lgmd_subtypes", "LGMD",
            "What LGMD subtypes are there?",
            [["subtype", "lgmd type", "capn3", "fkrp"]],
            ["lgmd", "limb-girdle", "limb girdle", "distribution"], "LGMD", _handle_lgmd_subtypes),
        QuestionTemplate("lgmd_delay", "LGMD",
            "What is the LGMD diagnostic delay?",
            [["diagnostic delay", "delay", "diagnosis time", "journey"]],
            ["lgmd", "limb-girdle"], "LGMD", _handle_lgmd_delay),
        QuestionTemplate("lgmd_meds", "LGMD",
            "What medications do LGMD participants take?",
            [["medication", "drug", "treatment", "therap"]],
            ["lgmd", "limb-girdle"], "LGMD", _handle_lgmd_medications),

        # Data Dictionary
        QuestionTemplate("field_count", "Data Dictionary",
            "How many fields does MOVR capture?",
            [["field", "variable"],
             ["how many", "total", "count", "capture"]],
            ["dictionary", "movr"], None, _handle_field_count),
        QuestionTemplate("clinical_domains", "Data Dictionary",
            "What clinical domains are in the data?",
            [["domain", "categor"]],
            ["clinical", "dictionary", "data"], None, _handle_clinical_domains),
    ]


# ── Suggested Question Categories ───────────────────────────────

SUGGESTED_CATEGORIES = {
    "Registry Overview": [
        "How many participants are in the registry?",
        "What diseases does MOVR cover?",
        "How many sites participate?",
    ],
    "ALS": [
        "What is the median ALSFRS-R score?",
        "What ALS medications are tracked?",
        "What is the El Escorial classification?",
    ],
    "DMD": [
        "What DMD exon-skipping therapies are tracked?",
        "What steroids do DMD participants take?",
        "What is the DMD genetic confirmation rate?",
    ],
    "SMA": [
        "What SMA types are in the registry?",
        "What SMA therapies are tracked?",
        "What are the SMA motor function scores?",
    ],
    "LGMD": [
        "What LGMD subtypes are there?",
        "What is the LGMD diagnostic delay?",
    ],
}

_WELCOME = (
    "Welcome to **Ask OpenMOVR**! I can answer questions about the MOVR "
    "neuromuscular disease registry using pre-computed summary statistics.\n\n"
    "Try clicking one of the suggested questions below, or type your own. "
    "I can tell you about participant counts, demographics, disease-specific "
    "clinical data (ALS, DMD, SMA, LGMD), medications, functional scores, and more.\n\n"
    "*All data shown is aggregated -- no individual-level data is used or displayed.*"
)

_FALLBACK = (
    "I'm not sure how to answer that. Here are some topics I can help with:\n\n"
    "- **Registry overview**: total participants, diseases, sites, encounters\n"
    "- **Disease counts**: how many participants per disease\n"
    "- **Demographics**: gender and age distributions per disease\n"
    "- **ALS**: ALSFRS-R scores, medications, El Escorial, FVC, onset age\n"
    "- **DMD**: genetics, exon-skipping therapies, steroids, ambulatory status\n"
    "- **SMA**: types, therapies (Spinraza, Zolgensma, Evrysdi), SMN2, motor scores\n"
    "- **LGMD**: subtypes, diagnostic delay, medications\n"
    "- **Data dictionary**: clinical domains, field counts\n\n"
    "Try asking something like \"How many ALS participants are there?\" or "
    "\"What is the median ALSFRS-R score?\""
)


# ── Chat Renderer ───────────────────────────────────────────────

class ChatbotRenderer:
    def __init__(self):
        self.kb = _load_all_snapshots()
        self.templates = _build_templates()
        self.matcher = QuestionMatcher(self.templates)

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    def _process_input(self, user_input: str):
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )
        template, disease = self.matcher.match(user_input)
        if template:
            response = template.handler(self.kb, disease)
        else:
            if disease:
                response = (
                    f"I don't have a specific answer for that **{disease}** question. "
                    + _FALLBACK
                )
            else:
                response = _FALLBACK
        st.session_state.chat_history.append(
            {"role": "assistant", "content": response}
        )

    def _render_suggested(self):
        for category, questions in SUGGESTED_CATEGORIES.items():
            st.markdown(f"**{category}**")
            cols = st.columns(min(len(questions), 3))
            for i, q in enumerate(questions):
                with cols[i % 3]:
                    if st.button(
                        q,
                        key=f"suggest_{category}_{i}_{len(st.session_state.chat_history)}",
                        use_container_width=True,
                    ):
                        self._process_input(q)
                        st.rerun()

    def render(self):
        # Welcome message
        if not st.session_state.chat_history:
            with st.chat_message("assistant"):
                st.markdown(_WELCOME)
            self._render_suggested()
        else:
            # Render history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            # Suggested follow-ups
            with st.expander("Suggested questions"):
                self._render_suggested()

        # Text input
        if user_input := st.chat_input("Ask a question about MOVR data..."):
            self._process_input(user_input)
            st.rerun()
