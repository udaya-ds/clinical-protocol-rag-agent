"""
Generates synthetic CDISC ADaM-style datasets (ADSL, ADAE) for the Gout
trial (protocol BP-202606-797), consistent with that protocol's known
facts: 90 subjects, 3 arms (Belimumab Medium Dose, Hydroxychloroquine Low
Dose, Standard of Care), ~37 sites.

This is synthetic data generated for portfolio/demo purposes - no real
patient data, same spirit as the CDISC-generated synthetic protocol PDFs
already in this project. Variable names and structure follow real CDISC
ADaM conventions (ADSL subject-level, ADAE adverse events) so the dataset
is authentically recognizable to anyone with clinical programming
experience, not just generic made-up column names.

Run: python data/sas_datasets/generate_adam_datasets.py
"""

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)  # reproducible

OUT_DIR = Path(__file__).parent
STUDYID = "BP-202606-797"
N_SUBJECTS = 90
N_SITES = 37

ARMS = [
    ("Belimumab Medium Dose", "BELI"),
    ("Hydroxychloroquine Low Dose", "HCQ"),
    ("Standard of Care", "SOC"),
]

RACES = ["WHITE", "BLACK OR AFRICAN AMERICAN", "ASIAN", "AMERICAN INDIAN OR ALASKA NATIVE", "OTHER"]
RACE_WEIGHTS = [0.62, 0.18, 0.12, 0.03, 0.05]

DISPOSITIONS = ["COMPLETED", "ADVERSE EVENT", "WITHDRAWAL BY SUBJECT", "LOST TO FOLLOW-UP", "LACK OF EFFICACY"]
DISPOSITION_WEIGHTS = [0.78, 0.07, 0.06, 0.04, 0.05]

AE_TERMS = [
    # (AEDECOD, AEBODSYS, base_severity_skew) - skew toward MILD for common/benign events
    ("Headache", "Nervous system disorders", "mild"),
    ("Nausea", "Gastrointestinal disorders", "mild"),
    ("Diarrhoea", "Gastrointestinal disorders", "mild"),
    ("Gout Flare", "Musculoskeletal and connective tissue disorders", "moderate"),
    ("Arthralgia", "Musculoskeletal and connective tissue disorders", "mild"),
    ("Injection Site Reaction", "General disorders and administration site conditions", "mild"),
    ("Upper Respiratory Tract Infection", "Infections and infestations", "mild"),
    ("Hypertension", "Vascular disorders", "moderate"),
    ("Fatigue", "General disorders and administration site conditions", "mild"),
    ("Rash", "Skin and subcutaneous tissue disorders", "mild"),
    ("Elevated Liver Enzymes", "Investigations", "moderate"),
    ("Renal Impairment", "Renal and urinary disorders", "severe"),
    ("Anaphylactic Reaction", "Immune system disorders", "severe"),
]

SEVERITY_BY_SKEW = {
    "mild": ["MILD", "MILD", "MILD", "MODERATE"],
    "moderate": ["MILD", "MODERATE", "MODERATE", "SEVERE"],
    "severe": ["MODERATE", "SEVERE", "SEVERE"],
}

OUTCOMES = ["RECOVERED/RESOLVED", "RECOVERED/RESOLVED", "RECOVERING/RESOLVING", "NOT RECOVERED/NOT RESOLVED"]

STUDY_START = date(2026, 1, 12)


def random_date(start: date, max_offset_days: int) -> date:
    return start + timedelta(days=random.randint(0, max_offset_days))


def generate_adsl() -> list[dict]:
    rows = []
    # Distribute subjects roughly evenly across the 3 arms (30/30/30)
    arm_assignments = []
    per_arm = N_SUBJECTS // len(ARMS)
    for arm_name, arm_code in ARMS:
        arm_assignments.extend([(arm_name, arm_code)] * per_arm)
    while len(arm_assignments) < N_SUBJECTS:
        arm_assignments.append(random.choice(ARMS))
    random.shuffle(arm_assignments)

    for i in range(1, N_SUBJECTS + 1):
        usubjid = f"{STUDYID}-{i:03d}"
        arm_name, arm_code = arm_assignments[i - 1]
        age = random.randint(18, 75)
        sex = random.choice(["M", "F"])
        race = random.choices(RACES, weights=RACE_WEIGHTS, k=1)[0]
        siteid = f"{random.randint(1, N_SITES):03d}"
        randdt = random_date(STUDY_START, 90)
        disposition = random.choices(DISPOSITIONS, weights=DISPOSITION_WEIGHTS, k=1)[0]
        saffl = "Y" if random.random() > 0.02 else "N"  # ~98% in safety population
        ittfl = "Y" if random.random() > 0.03 else "N"

        rows.append({
            "STUDYID": STUDYID,
            "USUBJID": usubjid,
            "SUBJID": f"{i:03d}",
            "SITEID": siteid,
            "AGE": age,
            "AGEU": "YEARS",
            "SEX": sex,
            "RACE": race,
            "ARM": arm_name,
            "ARMCD": arm_code,
            "ACTARM": arm_name,
            "ACTARMCD": arm_code,
            "RANDDT": randdt.isoformat(),
            "SAFFL": saffl,
            "ITTFL": ittfl,
            "DCDECOD": disposition,
            "EOSSTT": "COMPLETED" if disposition == "COMPLETED" else "DISCONTINUED",
        })
    return rows


def generate_adae(adsl_rows: list[dict]) -> list[dict]:
    rows = []
    for subj in adsl_rows:
        usubjid = subj["USUBJID"]
        arm_name = subj["ARM"]
        randdt = date.fromisoformat(subj["RANDDT"])

        # Not every subject has an AE - roughly 65% do, with a plausible
        # count distribution (most have 1-2, a few have more)
        if random.random() > 0.65:
            continue
        n_aes = random.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05], k=1)[0]

        for seq in range(1, n_aes + 1):
            aedecod, aebodsys, skew = random.choice(AE_TERMS)
            aesev = random.choice(SEVERITY_BY_SKEW[skew])
            # Serious AEs are rare and correlate with SEVERE severity
            aeser = "Y" if (aesev == "SEVERE" and random.random() < 0.4) else "N"
            aerel = random.choices(
                ["RELATED", "POSSIBLY RELATED", "NOT RELATED"], weights=[0.3, 0.3, 0.4], k=1
            )[0]
            aestdt = random_date(randdt, 80)
            aeout = random.choice(OUTCOMES)
            duration = random.randint(1, 21)
            aeendt = aestdt + timedelta(days=duration) if aeout != "NOT RECOVERED/NOT RESOLVED" else None

            rows.append({
                "STUDYID": STUDYID,
                "USUBJID": usubjid,
                "AESEQ": seq,
                "AEDECOD": aedecod,
                "AEBODSYS": aebodsys,
                "AESEV": aesev,
                "AESER": aeser,
                "AEREL": aerel,
                "AESTDTC": aestdt.isoformat(),
                "AEENDTC": aeendt.isoformat() if aeendt else "",
                "AEOUT": aeout,
                "TRTA": arm_name,
            })
    return rows


def write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    adsl = generate_adsl()
    adae = generate_adae(adsl)

    write_csv(adsl, OUT_DIR / "adsl.csv")
    write_csv(adae, OUT_DIR / "adae.csv")

    print(f"Generated ADSL: {len(adsl)} subjects -> {OUT_DIR / 'adsl.csv'}")
    print(f"Generated ADAE: {len(adae)} adverse event records -> {OUT_DIR / 'adae.csv'}")

    # Quick sanity summary
    arm_counts = {}
    for r in adsl:
        arm_counts[r["ARM"]] = arm_counts.get(r["ARM"], 0) + 1
    print("\nSubjects per arm:")
    for arm, count in arm_counts.items():
        print(f"  {arm}: {count}")

    subjects_with_ae = len({r["USUBJID"] for r in adae})
    print(f"\nSubjects with at least one AE: {subjects_with_ae}/{len(adsl)}")
    serious_ae_count = sum(1 for r in adae if r["AESER"] == "Y")
    print(f"Serious AEs: {serious_ae_count}/{len(adae)}")
