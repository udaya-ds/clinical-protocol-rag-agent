"""
Chunk protocol text by section, attaching metadata for filtered retrieval.

The CDISC-generated synthetic protocols (and most real protocols) follow a
numbered section structure (1. Synopsis, 2. Introduction..., 3.1 Inclusion
Criteria, etc.). Splitting on these headers - rather than a fixed character
window - keeps each chunk semantically coherent (e.g. "eligibility criteria"
never gets split mid-list), which matters a lot for retrieval precision.

Real-world protocol PDFs introduce noise that clean synthetic documents
don't have: a Table of Contents with dotted page leaders ("1.3 Schedule of
Activities ..... 19"), amendment/change-log tables that reference section
numbers as table cells, and repeated running headers/footers on every page
(e.g. "CONFIDENTIAL AND PROPRIETARY 12 of 110"). Left unhandled, these all
get misidentified as real section headers, producing duplicate or truncated
chunks. _strip_boilerplate() and _looks_like_toc_line() guard against this.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from pathlib import Path

# Matches top-level numbered sections like "5. Study Population" and
# sub-sections nested up to 4 levels deep, e.g. "3.1 Inclusion Criteria",
# "6.2.1 Dose Preparation", "6.2.1.2 Placebo". Real protocols routinely
# nest deeper than the synthetic corpus's max-2-level structure (confirmed
# directly: AstraZeneca's placebo-preparation details live under
# "6.2.1.2", four levels deep) - capping at 2 levels silently merges all
# deeper content into one oversized, multi-topic chunk, which measurably
# hurts BOTH dense and BM25 retrieval precision for questions about any
# single sub-topic within it.
#
# Title character class includes straight (') and curly (') apostrophes -
# confirmed directly: "8.3.6 Hy's Law" (curly apostrophe in the source
# text) failed to match without this, silently merging that subsection
# into the PRECEDING section ("8.3.5 Adverse Events Based on Examinations
# and Tests") instead of getting its own chunk. Any other possessive/
# contraction section title (e.g. a hypothetical "Investigator's
# Brochure") would hit the same gap without this fix.
#
# Title character class also includes digits - confirmed directly: Pfizer's
# "3.2 For Phase 2/3" (a real subsection under "3. OBJECTIVES, ESTIMANDS,
# AND ENDPOINTS") never matched without this, silently merging that entire
# subsection's content into whichever earlier section happened to be the
# nearest valid chunk boundary. IMPORTANT: allowing digits here, on its
# own, would reintroduce a regression - bare (no-dot) numbered list items
# that happen to mention something with a digit (e.g. an exclusion
# criterion mentioning "SARS-CoV-2") would start matching too. This is
# guarded against in _is_real_section() below: digits in the title are
# only trusted for multi-level (dotted) section numbers, never for bare
# single-level numbers.
SECTION_PATTERN = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][A-Za-z0-9/ \-'\u2019]{3,60})$",
    re.MULTILINE,
)

# A line is a TOC/List-of-Tables entry if it has a title followed by dots
# (or many spaces) leading to a trailing page number, e.g.
# "1.3 Schedule of Activities ................ 19"
TOC_LINE_PATTERN = re.compile(r"\.{3,}\s*\d{1,4}\s*$|\s{4,}\d{1,4}\s*$")

# Repeated running header/footer boilerplate seen on every page of many
# real protocol exports (sponsor name + confidentiality notice + page count).
BOILERPLATE_PATTERNS = [
    re.compile(r"^CONFIDENTIAL(?:\s+AND\s+PROPRIETARY)?\s*\d*\s*(?:of\s*\d+)?\s*$", re.I),
    re.compile(r"^Clinical Study Protocol.*\d{4}.*$"),  # title+date repeated per page
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.I),
]


def _strip_boilerplate(text: str) -> str:
    """Remove lines that are repeated running headers/footers rather than
    actual content, so they can't be mistaken for section structure."""
    lines = text.split("\n")
    kept = [ln for ln in lines if not any(p.match(ln.strip()) for p in BOILERPLATE_PATTERNS)]
    return "\n".join(kept)


def _looks_like_toc_or_reference(text: str, match_start: int, match_end: int) -> bool:
    """True if this section-header-shaped match is actually a Table of
    Contents entry, List of Tables entry, or an amendment-log table cell
    referencing a section number - not a real section header."""
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_end)
    line_end = line_end if line_end != -1 else len(text)
    full_line = text[line_start:line_end]
    return bool(TOC_LINE_PATTERN.search(full_line))


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    protocol_number: str | None
    section_number: str
    section_title: str
    text: str

    def to_metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("chunk_id")
        return {k: (v if v is not None else "") for k, v in d.items()}


def _extract_protocol_number(text: str) -> str | None:
    m = re.search(r"Protocol\s+(?:Number\s+)?([A-Z0-9\-]{6,})", text)
    return m.group(1) if m else None


def _is_real_section(m: re.Match, text: str) -> bool:
    """True if this section-header-shaped regex match is an actual section
    header, not noise.

    Three failure modes to guard against, all confirmed directly against
    real protocols (synthetic protocols never trigger any of them):
    1. TOC/List-of-Tables/amendment-log entries - handled by
       _looks_like_toc_or_reference().
    2. Bare numbered LIST ITEMS within a section's body text (e.g. a
       numbered exclusion criterion like "1 History of allergy to any
       component of the vaccine") getting matched as if they were their
       own top-level section. Confirmed: this exact case swallowed the
       rest of a real "5.2 Exclusion Criteria" section (3,361 chars,
       including a "Prior/Concomitant Therapy" subsection) into a fake
       section mislabeled "1 History of allergy to any component of the
       vaccine" instead.

       Heuristic: word count, not casing. An earlier version of this check
       used "ALL-CAPS vs lowercase," but that doesn't generalize - the
       synthetic corpus's legitimate top-level sections are Title Case
       ("Synopsis", "Study Population"), not ALL-CAPS, and would have been
       incorrectly rejected by a casing-based rule (confirmed: it dropped
       145 chunks down to 95 when tried). Word count holds up across both
       conventions instead: every legitimate section title seen so far
       (synthetic Title Case AND real-document ALL-CAPS) is 6 words or
       fewer, while numbered list items disguised as section headers tend
       to be longer, sentence-like phrases (the "History of allergy..."
       case is 9 words). A bare (no-dot) section number whose title is
       longer than 6 words is treated as a list item, not a real header.
       Multi-level numbers (X.Y, X.Y.Z, X.Y.Z.W) are NOT subject to this
       check - those are legitimate real subsections (e.g. "6.2.1.2
       Placebo") regardless of title length.
    3. Bare numbered LIST ITEMS whose title happens to contain a digit
       (e.g. exclusion criterion "4 History of laboratory-confirmed
       SARS-CoV-2 infection" - only 5 words, so check #2 above wouldn't
       catch it). This guard only exists BECAUSE digits were added to the
       title character class to fix a real bug (Pfizer's "3.2 For Phase
       2/3" needed digits allowed to match at all) - that fix would have
       reintroduced exactly this kind of false positive without this
       guard. Multi-level (dotted) section numbers are NOT subject to this
       check either - digits in a dotted subsection's title (e.g. "For
       Phase 2/3") are trusted, since multi-level numbering itself is
       already a strong, specific signal of a genuine subsection, not
       inline list-item numbering (lists in every document seen so far
       use bare single numbers, never dotted multi-level numbers).
    """
    if "." not in m.group("num"):
        if len(m.group("title").split()) > 6:
            return False
        if any(c.isdigit() for c in m.group("title")):
            return False
    return not _looks_like_toc_or_reference(text, m.start(), m.end())


def chunk_protocol_text(text: str, source_file: str) -> list[Chunk]:
    """Split a protocol's full text into one chunk per numbered section."""
    text = _strip_boilerplate(text)
    protocol_number = _extract_protocol_number(text)

    raw_matches = list(SECTION_PATTERN.finditer(text))
    # Drop matches that are actually TOC/List-of-Tables entries, amendment-
    # log table references, or bare numbered list items - not real section
    # headers.
    matches = [m for m in raw_matches if _is_real_section(m, text)]

    chunks: list[Chunk] = []

    if not matches:
        # Fallback: no recognizable section headers, treat as one chunk.
        return [Chunk(
            chunk_id=f"{source_file}::full",
            source_file=source_file,
            protocol_number=protocol_number,
            section_number="0",
            section_title="Full Document",
            text=text.strip(),
        )]

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if len(section_text) < 20:
            continue
        chunks.append(Chunk(
            chunk_id=f"{source_file}::{m.group('num')}::{i}",
            source_file=source_file,
            protocol_number=protocol_number,
            section_number=m.group("num"),
            section_title=m.group("title").strip(),
            text=section_text,
        ))

    return chunks


def chunk_all(processed_dir: str | Path) -> list[Chunk]:
    processed_dir = Path(processed_dir)
    all_chunks: list[Chunk] = []
    for txt_path in sorted(processed_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        all_chunks.extend(chunk_protocol_text(text, txt_path.name))
    return all_chunks


if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    chunks = chunk_all(processed_dir)
    print(f"Produced {len(chunks)} chunks from {len(list(processed_dir.glob('*.txt')))} documents\n")
    for c in chunks[:8]:
        print(f"[{c.source_file}] {c.section_number} {c.section_title} ({len(c.text)} chars)")