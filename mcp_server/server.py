"""
MCP server exposing the clinical protocol RAG pipeline as callable tools.

This wraps the existing pipeline (embeddings/embed_store.py,
agent/agentic_pipeline.py) rather than reimplementing any retrieval or
generation logic - the MCP layer is purely a tool-serving interface on top
of work already done elsewhere in this repo.

Run standalone for local testing:
    python mcp_server/server.py
More commonly, an MCP client (Claude Desktop, Claude Code, or the included
mcp_server/client_demo.py) launches this script as a subprocess over stdio.
"""

from __future__ import annotations
import os

# Must be set BEFORE importing anything that loads HuggingFace/sentence-
# transformers models (agentic_pipeline -> embed_store -> sentence_transformers).
# Confirmed real failure mode: under MCP's stdio transport, a tool call
# crashed with "fastmcp.exceptions.ToolError: ... [Errno 9] Bad file
# descriptor" - tqdm's progress bars (used throughout the HuggingFace/
# sentence-transformers model-loading chain, e.g. the "Loading weights"
# bars seen in every other context this pipeline runs in) write repeatedly
# to stderr, and something about this subprocess's stdio environment makes
# stderr unusable for that kind of repeated, in-place-updating write.
# Disabling progress bars entirely here removes the problem at its source,
# and serves no purpose in a non-interactive server anyway.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
from pathlib import Path

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent"))
sys.path.insert(0, str(PROJECT_ROOT / "embeddings"))
sys.path.insert(0, str(PROJECT_ROOT / "ingestion"))

import agentic_pipeline
import deep_research_router as dr

mcp = FastMCP("Clinical-Protocol-Server")

STRUCTURED_PATH = PROJECT_ROOT / "data" / "structured" / "structured_protocols.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _load_structured() -> list[dict]:
    if STRUCTURED_PATH.exists():
        return json.loads(STRUCTURED_PATH.read_text())
    return []


@mcp.tool()
def list_protocols() -> str:
    """List all indexed clinical trial protocols with their key identifying
    info (protocol number, phase, indication, study design). Use this to
    discover what's available before calling the other tools, or when the
    user doesn't specify a particular protocol by name."""
    structured = _load_structured()
    if structured:
        summary = [
            {
                "source_file": p.get("source_file"),
                "protocol_number": p.get("protocol_number"),
                "phase": p.get("phase"),
                "indication": p.get("indication"),
                "study_design": p.get("study_design"),
            }
            for p in structured
        ]
        return json.dumps({"status": "found", "count": len(summary), "protocols": summary}, indent=2)

    # Fallback if extraction hasn't been run yet - just list raw files.
    files = sorted(p.name for p in PROCESSED_DIR.glob("*.txt")) if PROCESSED_DIR.exists() else []
    return json.dumps({
        "status": "found_unstructured",
        "message": "No structured extraction found - showing raw processed filenames. "
                    "Run agent/extract_protocol.py for richer metadata.",
        "files": files,
    }, indent=2)


@mcp.tool()
def _status_for(protocol_filename: str | None) -> str:
    """Returns 'found' if protocol_filename is None (no filter requested,
    valid) or resolves to a real indexed protocol; 'not_found' if a
    non-empty filter was given but doesn't match anything indexed. Uses
    the same resolve_protocol_filter() / ProtocolNotFoundError mechanism
    agentic_pipeline.ask() relies on internally, rather than string-
    matching the prose answer text (which would be fragile and could
    silently break if that message's wording ever changes)."""
    if not protocol_filename:
        return "found"
    try:
        agentic_pipeline.resolve_protocol_filter(protocol_filename)
        return "found"
    except agentic_pipeline.ProtocolNotFoundError:
        return "not_found"


@mcp.tool()
def get_eligibility_criteria(protocol_filename: str) -> str:
    """Get inclusion and exclusion criteria for a specific protocol.
    protocol_filename should match a filename from list_protocols()
    (e.g. 'study_protocol_phase_3_20260619_045037.txt')."""
    answer = agentic_pipeline.ask(
        "What are the inclusion and exclusion criteria for this trial?",
        protocol_filter=protocol_filename,
    )
    return json.dumps({
        "status": _status_for(protocol_filename),
        "protocol_filename": protocol_filename,
        "answer": answer,
    }, indent=2)


@mcp.tool()
def get_primary_endpoint(protocol_filename: str) -> str:
    """Get the primary (and secondary, if mentioned) efficacy endpoint(s)
    for a specific protocol. protocol_filename should match a filename
    from list_protocols()."""
    answer = agentic_pipeline.ask(
        "What is the primary efficacy endpoint for this trial? Include secondary endpoints if relevant.",
        protocol_filter=protocol_filename,
    )
    return json.dumps({
        "status": _status_for(protocol_filename),
        "protocol_filename": protocol_filename,
        "answer": answer,
    }, indent=2)


@mcp.tool()
def ask_protocol_question(question: str, protocol_filename: str | None = None) -> str:
    """Ask any natural-language question about the indexed clinical trial
    protocol(s), answered by the full multi-agent pipeline (Eligibility,
    Endpoint, and Design specialist agents + Synthesis, with guardrails
    screening retrieved context before generation). If protocol_filename
    is omitted, searches across all indexed protocols rather than one
    specific trial."""
    answer = agentic_pipeline.ask(question, protocol_filter=protocol_filename)
    return json.dumps({
        "status": _status_for(protocol_filename),
        "question": question,
        "protocol_filename": protocol_filename or "all",
        "answer": answer,
    }, indent=2)


@mcp.tool()
def deep_research(question: str, protocol_filename: str | None = None) -> str:
    """Run a Deep Research query that automatically routes the question to the
    right tools — protocol document RAG, ADaM trial dataset SQL (with PROC SQL
    display), live web search, or all three in parallel — then synthesizes a
    structured markdown report with tables, citations, and key takeaways.

    Use this instead of ask_protocol_question when the question might need
    data from multiple sources, e.g. 'What does the protocol say about the
    primary endpoint and how many subjects had serious adverse events?'
    (needs both RAG and SQL), or 'What is Belimumab's mechanism of action
    and how does the trial design test it?' (needs web search + RAG).

    protocol_filename is optional — if provided, scopes the protocol RAG
    component to one specific trial. Omit to search all indexed protocols."""
    result = dr.deep_research(question, protocol_filter=protocol_filename)
    return json.dumps({
        "question": result["question"],
        # Scoped specifically to protocol resolution, not the overall call -
        # deep_research can still succeed via DATASET_SQL or WEB_SEARCH even
        # if the requested protocol_filename isn't indexed for RAG, so this
        # is reported separately rather than as a single pass/fail status.
        "protocol_status": _status_for(protocol_filename),
        "tools_used": result["tools_used"],
        "routing_reasoning": result["routing_reasoning"],
        "report": result["report"],
    }, indent=2)


if __name__ == "__main__":
    # show_banner=False is required, not cosmetic - confirmed directly that
    # FastMCP's startup banner prints to the SAME stdout stream the stdio
    # transport uses for JSON-RPC protocol messages. A client launching this
    # as a subprocess reads the banner text, fails to parse it as a valid
    # protocol message, and closes the connection - which then crashes the
    # server with a BrokenPipeError when it tries to write to the now-closed
    # pipe. This produces a generic "Connection closed" error on the client
    # side with no indication of the real cause.
    mcp.run(show_banner=False)
