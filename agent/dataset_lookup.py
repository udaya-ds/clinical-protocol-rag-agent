"""
Natural-language lookup over the synthetic ADaM datasets (ADSL, ADAE),
via LLM-generated SQL.

Honest scope note: we don't have a real SAS or R runtime available (SAS is
proprietary/licensed; this is a portfolio project, not a SAS-licensed
environment). So the actual EXECUTION path is: the LLM generates standard
SQL, which runs against an in-memory SQLite database built from the ADaM
CSVs. The LLM is ALSO asked to generate the equivalent PROC SQL syntax -
this is displayed alongside the answer (showing the authentic SAS-style
translation) but is NOT executed, since no SAS engine exists here to run
it against.

Safety: generated SQL is validated as read-only (SELECT-only) before
execution - this is the same OWASP-LLM06 (Excessive Agency) concern as
elsewhere in this project's guardrails, applied to a new risk surface: an
LLM generating executable code rather than just retrieved text. A
malicious or malformed query (DROP, DELETE, UPDATE, INSERT, ATTACH,
PRAGMA, etc.) is rejected before it ever reaches sqlite3.execute().
"""

from __future__ import annotations
import csv
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o"

DATASET_DIR = PROJECT_ROOT / "data" / "sas_datasets"

SCHEMA_DESCRIPTION = """\
Table ADSL (Subject-Level Analysis Dataset - one row per subject):
  STUDYID (text), USUBJID (text, primary key), SUBJID (text), SITEID (text),
  AGE (integer), AGEU (text, always 'YEARS'), SEX (text, 'M'/'F'),
  RACE (text), ARM (text, planned treatment arm), ARMCD (text, short arm code),
  ACTARM (text, actual treatment arm), ACTARMCD (text),
  RANDDT (text, ISO date), SAFFL (text, 'Y'/'N' safety population flag),
  ITTFL (text, 'Y'/'N' intent-to-treat flag),
  DCDECOD (text, disposition: COMPLETED/ADVERSE EVENT/WITHDRAWAL BY SUBJECT/
           LOST TO FOLLOW-UP/LACK OF EFFICACY),
  EOSSTT (text, 'COMPLETED' or 'DISCONTINUED')

Table ADAE (Adverse Events Analysis Dataset - one row per adverse event,
multiple rows per subject possible, join to ADSL via USUBJID):
  STUDYID (text), USUBJID (text, foreign key to ADSL), AESEQ (integer),
  AEDECOD (text, adverse event term), AEBODSYS (text, body system),
  AESEV (text, 'MILD'/'MODERATE'/'SEVERE'),
  AESER (text, 'Y'/'N' serious adverse event flag),
  AEREL (text, 'RELATED'/'POSSIBLY RELATED'/'NOT RELATED' to study drug),
  AESTDTC (text, ISO start date), AEENDTC (text, ISO end date or blank if ongoing),
  AEOUT (text, outcome), TRTA (text, actual treatment arm at time of event)
"""

SYSTEM_PROMPT = f"""You are a clinical data analyst assistant that translates natural \
language questions into SQL queries against ADaM-standard clinical trial datasets.

{SCHEMA_DESCRIPTION}

Rules:
- Generate a single READ-ONLY SQLite SELECT query that answers the question. \
NEVER generate INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, or PRAGMA statements.
- ALSO generate the equivalent SAS PROC SQL syntax for the same query, for display \
purposes only (it will not be executed, since no SAS environment exists here - this \
is purely to show the SAS-equivalent translation).
- Return ONLY a JSON object with exactly two keys: "sql" (the executable SQLite \
query) and "proc_sql" (the equivalent PROC SQL text, as a multi-line string starting \
with "PROC SQL;" and ending with "QUIT;").
- No markdown, no commentary, no code fences - just the JSON object.
"""


_UNSAFE_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)


def is_safe_select(sql: str) -> tuple[bool, str]:
    """Read-only validation gate. Returns (is_safe, reason_if_not)."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False, "empty query"
    if not stripped.upper().startswith("SELECT") and not stripped.upper().startswith("WITH"):
        return False, "query must start with SELECT or WITH (a read-only statement)"
    if _UNSAFE_SQL_PATTERN.search(stripped):
        return False, "query contains a write/schema-modifying keyword, which is not permitted"
    if ";" in stripped:
        return False, "multiple statements (semicolon-separated) are not permitted"
    return True, ""


def _build_connection() -> sqlite3.Connection:
    """Load the ADSL/ADAE CSVs into a fresh in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    missing_files = []
    for table_name, filename in [("ADSL", "adsl.csv"), ("ADAE", "adae.csv")]:
        path = DATASET_DIR / filename
        if not path.exists():
            # Confirmed real failure mode: silently `continue`-ing past a
            # missing CSV meant a query against a table that was never
            # created failed with a generic "no such table: ADAE" SQLite
            # error, giving zero indication that the actual problem was a
            # missing data file at a specific, checkable path. Surface that
            # path explicitly instead.
            missing_files.append(str(path))
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            missing_files.append(f"{path} (exists but is empty)")
            continue
        columns = list(rows[0].keys())
        col_defs = ", ".join(f'"{c}" TEXT' for c in columns)  # TEXT for everything - simplest, avoids type-coercion edge cases
        conn.execute(f'CREATE TABLE {table_name} ({col_defs})')
        placeholders = ", ".join("?" for _ in columns)
        conn.executemany(
            f'INSERT INTO {table_name} VALUES ({placeholders})',
            [tuple(row[c] for c in columns) for row in rows],
        )
    conn.commit()

    if missing_files:
        raise FileNotFoundError(
            f"Could not load the following dataset file(s): {missing_files}. "
            f"Run `python data/sas_datasets/generate_adam_datasets.py` to generate them, "
            f"or check that they're saved at the expected path: {DATASET_DIR}"
        )

    return conn


def generate_query(question: str) -> dict:
    """LLM call: question -> {"sql": ..., "proc_sql": ...}"""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=800,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    import json
    # strict=False tolerates literal control characters (raw newlines, tabs)
    # inside JSON string values - LLMs frequently emit these for multi-line
    # content like a PROC SQL block, even when explicitly told to return
    # valid JSON. Strict parsing would reject this with a confusing
    # "Invalid control character" error on otherwise-correct output.
    return json.loads(raw, strict=False)


def answer_dataset_question(question: str) -> dict:
    """Full pipeline: generate SQL -> validate -> execute -> return structured
    results plus the SAS-equivalent display text."""
    generated = generate_query(question)
    sql = generated.get("sql", "")
    proc_sql = generated.get("proc_sql", "")

    safe, reason = is_safe_select(sql)
    if not safe:
        return {
            "question": question,
            "sql": sql,
            "proc_sql": proc_sql,
            "blocked": True,
            "block_reason": reason,
            "results": None,
        }

    conn = _build_connection()
    try:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        error = None
    except Exception as e:
        rows = None
        error = str(e)
    finally:
        conn.close()

    return {
        "question": question,
        "sql": sql,
        "proc_sql": proc_sql,
        "blocked": False,
        "results": rows,
        "error": error,
    }


if __name__ == "__main__":
    question = input("Enter a question about the trial dataset: ").strip()
    result = answer_dataset_question(question)

    print(f"\nGenerated SQL:\n  {result['sql']}")
    print(f"\nEquivalent PROC SQL (display only, not executed):\n{result['proc_sql']}")

    if result["blocked"]:
        print(f"\nBLOCKED - {result['block_reason']}")
    elif result.get("error"):
        print(f"\nQuery execution error: {result['error']}")
    else:
        print(f"\nResults ({len(result['results'])} rows):")
        for row in result["results"][:20]:
            print(f"  {row}")
