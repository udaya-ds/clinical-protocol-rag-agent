"""
Structured extraction pipeline: raw protocol text -> validated StudyProtocol JSON.

Follows the standard extract -> validate -> repair loop pattern:

    1. Prompt an LLM to extract fields into JSON
    2. Validate against a Pydantic schema
    3. If invalid, feed the error back to the LLM and ask it to repair
    4. Retry up to N times before giving up
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI
from pydantic import ValidationError

from extraction_schema import StudyProtocol
from guardrails import screen_document

client = OpenAI()  # expects OPENAI_API_KEY in env
MODEL = "gpt-4o"


SYSTEM_PROMPT = """You are an expert clinical data extraction assistant specialized \
in clinical trial protocol documents.

Your task is to read a clinical trial protocol (any phase, any therapeutic area, any \
study design) and extract its key attributes into a structured JSON object that \
exactly matches the required schema.

Rules:
- Return ONLY a valid JSON object. No markdown, no commentary, no code fences.
- If a field is genuinely absent from the text, return null (for optional fields) \
or an empty list [] (for list fields). Never fabricate information not present in \
the text.
- Numeric fields (e.g. treatment_duration_weeks, planned_enrollment, number_of_sites) \
must be extracted as integers, not strings.
- inclusion/exclusion criteria and objectives/endpoints should be returned as lists \
of strings, with each list item being one criterion/objective/endpoint, with leading \
numbering removed.
- treatment_arms should be a list of objects, each with arm_label (e.g. "Arm 1") and \
description (e.g. "Gabapentin Low Dose").
- IMPORTANT: "exclusion criteria" means PRE-ENROLLMENT eligibility exclusions, not \
"discontinuation criteria" / "withdrawal criteria" (reasons a participant might stop \
or be withdrawn AFTER already enrolling). These two lists can use similar-sounding \
language (e.g. both may mention pregnancy or lab-confirmed infection) - only extract \
the pre-enrollment exclusion list for the "exclusion" field.
"""

SCHEMA_HINT = StudyProtocol.model_json_schema()
FIELD_NAMES = list(SCHEMA_HINT.get("properties", {}).keys())

# --- Chunking config ---
# Char-based sizing (no extra tiktoken dependency needed). ~4 chars/token is a
# safe-enough approximation for chunk planning purposes (we're not trying to
# hit the limit exactly, just stay comfortably under it).
MAX_CHUNK_CHARS = 30000   # ~7500 tokens of protocol text per chunk
CHUNK_OVERLAP_CHARS = 800  # avoid losing criteria/objectives split across a boundary


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split protocol text into overlapping character chunks."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunk_extraction_prompt(chunk: str, chunk_idx: int, total_chunks: int, source_filename: str) -> str:
    """Lightweight per-chunk prompt: field names only, not the full JSON schema,
    to keep repeated per-chunk calls cheap on tokens."""
    return f"""This is chunk {chunk_idx + 1} of {total_chunks} from a clinical trial \
protocol document ("{source_filename}"). Chunks are split by character position and \
may begin or end mid-section.

Extract whatever values you can find in THIS CHUNK for the following fields:
{json.dumps(FIELD_NAMES)}

Rules:
- Return ONLY a valid JSON object with exactly these keys.
- If a field is not present in this chunk, return null (or [] for list-type fields \
like criteria/objectives/endpoints/treatment_arms). Do NOT guess or carry over \
values from outside this chunk.
- Never fabricate information not present in this chunk.
- inclusion/exclusion criteria, objectives, and endpoints are lists of strings (one \
item per criterion/objective/endpoint, no leading numbering).
- treatment_arms is a list of objects with arm_label and description.
- Numeric fields must be integers, not strings.
- IMPORTANT: "exclusion criteria" means PRE-ENROLLMENT eligibility exclusions (found \
in a "Study Population" / "Eligibility" section, deciding whether someone may join \
the study at all). Do NOT confuse this with "discontinuation criteria" / "withdrawal \
criteria" (found in a "Discontinuation of Study Intervention" or similar section), \
which describe reasons a participant might STOP or be WITHDRAWN from the study after \
already enrolling. These two lists often sit close together in the same document and \
can use similar-sounding language (e.g. both may mention pregnancy, lab-confirmed \
infection, or serious adverse events) - only extract the pre-enrollment exclusion \
list for the "exclusion" field, even if a discontinuation/withdrawal list also \
appears in this chunk.

Chunk text:
---
{chunk}
---

Return ONLY the JSON object."""


import re as _re

# Keywords suggesting a "primary"/"objectives" candidate is actually about
# SAFETY/REACTOGENICITY, not the trial's actual primary EFFICACY endpoint.
# Confirmed real failure mode: Pfizer's merge picked a reactogenicity-focused
# chunk's endpoint list (local reactions, AEs, GMTs...) over the real
# efficacy endpoint (VE = 100x(1-IRR)), purely because that chunk happened to
# get processed first and "first non-empty wins" had no way to tell the two
# apart.
_SAFETY_KEYWORDS = (
    "local reaction", "systemic event", "adverse event", " ae ", " aes ",
    "serious ae", " sae", "geometric mean", "gmt", "gmc", "gmfr",
    "laboratory value", "reactogenicity", "neutralizing titer",
)


def _looks_safety_focused(items: list[str]) -> bool:
    joined = " ".join(items).lower()
    return any(kw in joined for kw in _SAFETY_KEYWORDS)


def _normalize_for_dedup(text: str) -> str:
    """Normalize for near-duplicate detection - lowercase, strip
    punctuation/whitespace differences (e.g. '5 x 10^10' vs '5 x 1010'
    should be recognized as the same value, not two separate list items)."""
    return _re.sub(r"[^a-z0-9]+", "", text.lower())


_DRUG_NAME_PATTERN = _re.compile(r"\b[A-Z]{2,}[A-Za-z]*\d[A-Za-z0-9]*\b")  # e.g. AZD1222, BNT162b2


def _arm_group_key(item) -> str | None:
    """Extract a grouping key for a treatment_arms candidate - the drug
    name if one is detectable (e.g. 'AZD1222'), or 'placebo'/'standard of
    care' for those special-cased arms. Returns None if no group key is
    detectable (kept as its own standalone item rather than risking a wrong
    merge)."""
    text = _item_text(item)
    m = _DRUG_NAME_PATTERN.search(text)
    if m:
        return m.group(0).upper()
    lowered = text.lower()
    if "placebo" in lowered:
        return "placebo"
    if "standard of care" in lowered:
        return "standard of care"
    return None


def _merge_treatment_arms(candidates: list[list]) -> list:
    """treatment_arms needs the OPPOSITE strategy from other list fields.
    Confirmed real failure: naive "longest list wins" picked a verbose,
    polluted candidate (supply/sourcing descriptions, near-duplicate dosing
    strings) over the clean canonical pair ('AZD1222', 'Placebo') from the
    Synopsis, precisely BECAUSE the messy candidate was longer. For arms,
    shorter/cleaner descriptions are the more trustworthy signal, not list
    completeness - there's no risk of a real arm being "incomplete" the way
    an exclusion-criteria list can be, since an arm description is usually
    a single short label, not a list of independent items.

    Strategy: pool every candidate item across all chunks, group items
    referring to the same drug/treatment (e.g. 'AZD1222' detected in both
    'AZD1222' and '2 IM doses of 5x1010 vp AZD1222'), and within each
    group keep only the SHORTEST description - the verbose, supply-chain-
    style descriptions are reliably longer than the canonical Synopsis-
    style label for the same arm.
    """
    all_items = [item for c in candidates for item in c]
    if not all_items:
        return []

    groups: dict[str, list] = {}
    standalone: list = []
    for item in all_items:
        key = _arm_group_key(item)
        if key is None:
            standalone.append(item)
        else:
            groups.setdefault(key, []).append(item)

    result = [min(group_items, key=lambda it: len(_item_text(it))) for group_items in groups.values()]

    # De-dupe standalone items (no detected group key) the normal way, in
    # case the same ungroupable text appears verbatim in multiple chunks.
    seen_standalone = set()
    for item in standalone:
        key = _normalize_for_dedup(_item_text(item))
        if key not in seen_standalone:
            seen_standalone.add(key)
            result.append(item)

    return result


def _merge_list_field(candidates: list[list], field_name: str) -> list:
    """Merge a list-type field across chunks. Confirmed real failure modes
    this addresses:
    - AZ's treatment_arms ended up polluted with verbose supply/sourcing
      descriptions alongside the clean canonical arm labels, because naive
      union just concatenates every chunk's list regardless of quality.
    - Near-duplicate entries with cosmetic formatting differences (e.g.
      '5 x 10^10 vp' vs '5 x 1010 vp') counted as two separate items under
      exact-string deduplication.
    - AZ's exclusion criteria came back with 0/3 expected phrases found -
      the single most complete candidate list is generally more trustworthy
      than gluing together fragments from several chunks, which risks both
      incompleteness (no single chunk had the FULL list) and pollution
      (a less-relevant chunk's short/wrong list getting unioned in).

    Strategy: prefer the LONGEST non-empty candidate list as the base (most
    likely to be the complete, correctly-scoped one), then only add items
    from OTHER candidates if they're not a near-duplicate of something
    already present - rather than blindly unioning every chunk's list
    together regardless of quality.
    """
    non_empty = [c for c in candidates if c]
    if not non_empty:
        return []

    non_empty.sort(key=len, reverse=True)
    merged = list(non_empty[0])
    seen = {_normalize_for_dedup(_item_text(item)) for item in merged}

    for candidate_list in non_empty[1:]:
        for item in candidate_list:
            key = _normalize_for_dedup(_item_text(item))
            if key and key not in seen:
                seen.add(key)
                merged.append(item)
    return merged


def _item_text(item) -> str:
    """List items are usually strings, but treatment_arms items are dicts
    ({'arm_label':..., 'description':...}) - get a comparable text
    representation either way."""
    if isinstance(item, dict):
        return f"{item.get('arm_label', '')} {item.get('description', '')}"
    return str(item)


def _merge_primary_endpoint(candidates: list[list[str]]) -> list[str]:
    """Specifically for endpoints.primary / similar "primary X" fields:
    prefer a candidate that does NOT look safety/reactogenicity-focused,
    rather than just taking whichever chunk was processed first. Falls back
    to the longest-list strategy if every candidate looks safety-focused
    (better to return something than nothing) or none do (ambiguous either
    way, so completeness wins)."""
    non_empty = [c for c in candidates if c]
    if not non_empty:
        return []

    efficacy_focused = [c for c in non_empty if not _looks_safety_focused(c)]
    pool = efficacy_focused if efficacy_focused else non_empty
    return _merge_list_field(pool, "endpoints.primary")


def merge_partial_extractions(partials: list[dict], source_filename: str) -> dict:
    """Merge per-chunk partial dicts into one.

    Scalars: first non-empty value wins (unchanged - works fine for facts
    that appear identically regardless of which chunk found them first,
    e.g. protocol_number, planned_enrollment).

    List fields get smarter, field-aware merging (see _merge_list_field /
    _merge_primary_endpoint) instead of naive union, since naive union was
    confirmed to produce both incompleteness and pollution depending on the
    field and document.
    """
    # Collect every chunk's value for each key, preserving chunk order, so
    # list-type fields can be merged with full visibility into all
    # candidates rather than one-at-a-time accumulation.
    all_keys = {k for p in partials for k in p if k != "source_file"}
    candidates_by_key = {k: [p.get(k) for p in partials if p.get(k) not in (None, "", [])] for k in all_keys}

    merged: dict = {}
    for key, candidates in candidates_by_key.items():
        if not candidates:
            continue

        if all(isinstance(c, list) for c in candidates):
            if key == "treatment_arms":
                merged[key] = _merge_treatment_arms(candidates)
            elif key == "primary" or key.endswith(".primary"):
                merged[key] = _merge_primary_endpoint(candidates)
            else:
                merged[key] = _merge_list_field(candidates, key)
        elif all(isinstance(c, dict) for c in candidates):
            # Nested objects (eligibility, endpoints) - recurse one level so
            # their own list sub-fields (inclusion/exclusion, primary/
            # secondary) get the smarter merge treatment too, instead of the
            # whole nested dict just taking "first non-empty wins."
            sub_partials = candidates
            merged[key] = merge_partial_extractions(sub_partials, source_filename)
            merged[key].pop("source_file", None)
        else:
            # Plain scalar - first non-empty wins, as before.
            merged[key] = candidates[0]

    merged["source_file"] = source_filename
    return merged


def build_extraction_prompt(protocol_text: str, source_filename: str) -> str:
    return f"""Extract the clinical trial protocol information from the text below \
and return a JSON object matching this schema:

{json.dumps(SCHEMA_HINT, indent=2)}

Set "source_file" to: "{source_filename}"

Protocol text:
---
{protocol_text}
---

Return ONLY the JSON object."""


def build_repair_prompt(broken_json: str, error_message: str) -> str:
    return f"""The following JSON failed schema validation.

Candidate JSON:
{broken_json}

Validation error:
{error_message}

Fix the JSON so it passes validation against this schema:
{json.dumps(SCHEMA_HINT, indent=2)}

Return ONLY the corrected JSON object, no explanation, no markdown."""


def call_llm(system_msg: str, user_prompt: str, max_tokens: int = 2000) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,  # deterministic, faithful extraction - not creative writing
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def _clean(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _parse_partial(raw: str) -> dict:
    """Best-effort JSON parse of a single chunk's extraction output. Returns {} on
    failure rather than raising, so one bad chunk doesn't sink the whole document."""
    try:
        return json.loads(_clean(raw))
    except json.JSONDecodeError:
        return {}


def extract_protocol(protocol_text: str, source_filename: str, max_retries: int = 3) -> StudyProtocol:
    """Extract+validate+repair loop. Returns a validated StudyProtocol instance.

    Large documents are split into char-based chunks first (each chunk extracted
    independently with a lightweight field-only prompt), then merged into a single
    candidate JSON object before the validate/repair loop runs. This keeps any one
    request well under the per-minute token limit instead of sending the entire
    protocol (plus the full JSON schema) in one shot.
    """
    chunks = chunk_text(protocol_text)

    if len(chunks) == 1:
        prompt = build_extraction_prompt(protocol_text, source_filename)
        raw_output = call_llm(SYSTEM_PROMPT, prompt)
    else:
        print(f"  (splitting into {len(chunks)} chunks for extraction)")
        partials = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = build_chunk_extraction_prompt(chunk, i, len(chunks), source_filename)
            chunk_raw = call_llm(SYSTEM_PROMPT, chunk_prompt, max_tokens=3000)
            partial = _parse_partial(chunk_raw)
            if not partial:
                print(f"    chunk {i + 1}/{len(chunks)}: failed to parse, skipping")
            partials.append(partial)

        merged = merge_partial_extractions(partials, source_filename)
        raw_output = json.dumps(merged)

    attempt = 0
    last_error = None
    # Repaired JSON must be at least as large as the broken JSON, so give
    # the repair call headroom proportional to the current output size.
    repair_max_tokens = max(4000, len(raw_output) // 3)
    while attempt < max_retries:
        try:
            clean = _clean(raw_output)
            return StudyProtocol.model_validate_json(clean)
        except ValidationError as e:
            last_error = e
            attempt += 1
            repair_prompt = build_repair_prompt(_clean(raw_output), e.json())
            raw_output = call_llm(SYSTEM_PROMPT, repair_prompt, max_tokens=4096)
        except json.JSONDecodeError as e:
            last_error = e
            attempt += 1
            repair_prompt = build_repair_prompt(_clean(raw_output), str(e))
            raw_output = call_llm(SYSTEM_PROMPT, repair_prompt, max_tokens=4096)

    raise RuntimeError(f"Failed to extract valid protocol JSON after {max_retries} retries: {last_error}")


if __name__ == "__main__":
    from pathlib import Path

    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir = Path(__file__).parent.parent / "data" / "structured"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structured_protocols.json"

    results = []
    skipped = []
    failed = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")

        screening = screen_document(text, txt_path.name)
        if not screening["safe_to_process"]:
            print(f"SKIPPING {txt_path.name} - failed guardrail screening: {screening}")
            skipped.append(screening)
            continue

        print(f"Extracting {txt_path.name} ...")
        try:
            protocol = extract_protocol(text, source_filename=txt_path.name)
        except Exception as e:
            # Confirmed real failure mode: without this try/except, one
            # document's RuntimeError (after exhausting max_retries) used to
            # crash the ENTIRE script - silently abandoning every file after
            # it in the batch, with the output file left stale from whatever
            # the last successful run happened to produce. Catching it here
            # means one bad document gets clearly reported and skipped, while
            # every other document - before or after it - still gets
            # processed and saved.
            print(f"  FAILED to extract {txt_path.name}: {e}")
            failed.append({"source_file": txt_path.name, "error": str(e)})
            continue

        results.append(protocol.model_dump())
        print(f"  -> {protocol.protocol_number} | {protocol.indication} | {protocol.study_design}")

        # Save incrementally after each successful extraction, not just once
        # at the end - so a crash on a LATER file still leaves every
        # already-completed result on disk, rather than risking the whole
        # batch's output (e.g. if something outside this try/except, like a
        # KeyboardInterrupt, stops the run early).
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if skipped:
        print(f"\n{len(skipped)} document(s) skipped by guardrails - see above.")
    if failed:
        print(f"\n{len(failed)} document(s) FAILED extraction - see above. "
              f"Other documents were still processed and saved normally.")

    print(f"\nSaved {len(results)} structured protocols -> {out_path}")
