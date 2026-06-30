"""
Multi-agent pipeline over the clinical protocol vector index: specialized
agents + LangGraph orchestration.

Three specialized retrieval agents (Eligibility, Endpoint, Design) each
query a metadata-filtered slice of the ChromaDB index, run results through
the guardrails context scanner, and produce a grounded, section-scoped
answer. A fourth General agent runs broader, unfiltered retrieval across
the whole (optionally protocol-scoped) index - a deliberate fallback for
questions that don't map to any of the three specialists' fixed section
lists (e.g. drug/placebo preparation, safety monitoring, statistical
methods, data management - real questions a real protocol can be asked,
but outside what eligibility/endpoint/design were ever scoped to find). A
Synthesis agent combines all four outputs into one final response with
explicit source citations (protocol + section) for traceability.

This is the "agentic AI" centerpiece of the project: not just RAG Q&A, but
multiple coordinated agents with distinct scopes, composed via a graph.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "embeddings"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from guardrails import scan_context

import embed_store
from reranker import rerank

client = OpenAI()
MODEL = "gpt-4o"

# Real protocol chunks can be much larger than synthetic ones - some
# sections run 6,000-19,000+ characters (confirmed directly during testing
# on real AstraZeneca/Pfizer protocols). Combining several such chunks into
# one prompt can exceed token-per-minute rate limits even though each
# individual chunk is "just text" - this caps each chunk's contribution to
# the prompt so the total stays bounded regardless of which chunks
# retrieval happens to surface. Same fix already applied in
# local_test_data/run_real_protocol_test.py; ported here since this is the
# pipeline Streamlit/MCP actually run against.
MAX_CHARS_PER_CHUNK_IN_PROMPT = 3000

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


class ProtocolNotFoundError(Exception):
    """Raised when a non-empty protocol_filter doesn't match ANY indexed
    protocol. Distinguishes 'this document isn't indexed here at all' from
    'the document is indexed but doesn't mention this' - two very
    different facts that previously looked identical to the end user
    (both produced a generic 'could not be determined from protocol
    excerpts' answer after the full agent pipeline ran and found nothing,
    since ChromaDB's exact-match `where` filter silently matches zero
    chunks for an unindexed source_file). Confirmed real-world case: a
    protocol_filter referencing a real-world document (e.g. a Pfizer/AZ
    trial) requested against an environment where only the synthetic
    corpus is indexed (e.g. the public git-committed version of this
    repo, which deliberately excludes real protocol PDFs)."""

    def __init__(self, requested: str, available: list[str]):
        self.requested = requested
        self.available = available
        super().__init__(f"Protocol '{requested}' not found. Available: {available}")


def resolve_protocol_filter(user_input: str | None) -> str | None:
    """Match a partial/imprecise filename (e.g. 'D8110C00001_CSP', typed by
    a person who doesn't know or remember the exact processed filename)
    against the real filenames in data/processed/, returning the exact
    string that's actually stored as `source_file` metadata in the index.

    Without this, a protocol_filter that doesn't EXACTLY match the stored
    metadata value silently matches zero chunks for every agent (ChromaDB's
    `where` filter is an exact-equality match, not substring) - which looks
    identical to "the protocol genuinely has no relevant content," but is
    actually just a typo/abbreviation mismatch. This was confirmed as the
    root cause of exactly that confusing symptom during testing.
    """
    if not user_input:
        return None

    # Strip stray quote characters and whitespace - very common when a
    # filename gets pasted from a file explorer's "Copy path" feature
    # (which often wraps it in quotes) or typed with extra whitespace.
    # Without this, the quoted/padded string can never match or even
    # substring-match the real filename, silently producing zero results
    # for every agent - confirmed as the root cause of this exact symptom
    # during testing.
    cleaned = user_input.strip().strip("'\"")
    if cleaned != user_input:
        print(f"(Stripped quotes/whitespace from protocol filter: {user_input!r} -> {cleaned!r})")
    user_input = cleaned

    if not PROCESSED_DIR.exists():
        return user_input  # can't resolve, pass through unchanged

    available = [p.name for p in PROCESSED_DIR.glob("*.txt")]

    # Exact match first (handles the case where the user already typed it correctly).
    if user_input in available:
        return user_input

    # Case-insensitive substring match - the typical case (user typed a
    # shortened/partial name without the exact suffix/extension).
    matches = [f for f in available if user_input.lower() in f.lower()]
    if len(matches) == 1:
        print(f"(Resolved protocol filter '{user_input}' -> '{matches[0]}')")
        return matches[0]
    if len(matches) > 1:
        print(f"WARNING: '{user_input}' matches multiple protocols {matches} - "
              f"using the first match. Be more specific to avoid ambiguity.")
        return matches[0]

    raise ProtocolNotFoundError(user_input, available)

# Each specialized agent is scoped to specific section titles in the index,
# mirroring how a human reviewer would only look at the relevant part of a
# protocol rather than the whole document. The General agent (handled
# separately below, not in this dict) deliberately has NO fixed section
# list - it's the fallback for everything these three don't cover.
AGENT_SECTION_FILTERS = {
    "eligibility": ["Inclusion Criteria", "Exclusion Criteria", "Study Population"],
    "endpoint": ["Efficacy Assessments", "Primary Efficacy Endpoint", "Secondary Efficacy Endpoints"],
    "design": ["Study Design", "Overview of Study Design", "Visit Schedule"],
}


class PipelineState(TypedDict):
    question: str
    protocol_filter: str | None  # optional source_file to restrict to one protocol
    eligibility_answer: str
    endpoint_answer: str
    design_answer: str
    general_answer: str
    final_answer: str


def _log_retrieval_error(message: str) -> None:
    """Best-effort error logging to stderr - wrapped so a logging failure
    can NEVER itself become a new crash. Confirmed real failure mode:
    under MCP's stdio transport, stderr can apparently be closed/redirected
    in a way that makes writing to it raise its own OSError ("Bad file
    descriptor") - which, since it occurs INSIDE an except block, is a NEW
    exception that escapes uncaught rather than being handled by the
    original try/except. Diagnostic logging must never be allowed to crash
    the operation it's trying to help debug - if logging itself fails,
    silently give up on logging this one error rather than propagating a
    second failure on top of the first."""
    try:
        print(message, file=sys.stderr)
    except Exception:
        pass


def _retrieve_for_agent(question: str, section_titles: list[str], protocol_filter: str | None,
                         n_results: int = 4, candidate_pool_size: int = 4) -> list[dict]:
    """Retrieve chunks for one agent, filtered to its scoped sections
    (and optionally to a single protocol).

    Two-stage retrieval: pull a larger candidate pool per section using
    embedding similarity (fast, approximate), then rerank the merged pool
    with a cross-encoder (slower, but reads query+chunk together for much
    more precise relevance) and keep only the true top n_results. This is
    what filters out noisy matches - e.g. an amendment-log table reference
    that happens to share vocabulary with the query - before they ever
    reach the LLM.
    """
    # ChromaDB `where` only supports one equality per key without $or/$and
    # composition in the simplest form, so we query per-section and merge.
    all_hits = []
    for title in section_titles:
        where = {"section_title": title}
        if protocol_filter:
            where = {"$and": [{"section_title": title}, {"source_file": protocol_filter}]}
        try:
            hits = embed_store.query(question, n_results=candidate_pool_size, where=where)
            all_hits.extend(hits)
        except Exception as e:
            # Never silently swallow this. Confirmed real failure mode: a
            # concurrent process (Streamlit) holding ChromaDB's SQLite-backed
            # storage open caused queries from a separately-launched MCP
            # server subprocess to fail - and because this was a bare
            # `except Exception: continue`, every retrieval call returned
            # empty with ZERO visible indication of why, looking identical
            # to "genuinely found nothing." Logging goes through
            # _log_retrieval_error() (fail-safe to stderr), not a bare
            # print() - confirmed separately that stderr itself can be
            # unwritable under MCP's stdio transport, which would otherwise
            # turn a logging attempt into a brand new crash.
            _log_retrieval_error(f"[retrieval error] section '{title}': {e}")
            continue

    return rerank(question, all_hits, top_k=n_results)


def _retrieve_general(question: str, protocol_filter: str | None,
                       n_results: int = 4, candidate_pool_size: int = 12) -> list[dict]:
    """Broader, UNFILTERED retrieval across all sections of the (optionally
    protocol-scoped) index - the fallback path for questions that don't map
    to eligibility/endpoint/design's fixed section lists. Real protocols
    have far more content than those three categories cover (drug/placebo
    preparation, safety monitoring, statistical methods, data management,
    etc.) - this agent exists so a question about any of that doesn't
    silently fall through every specialist's scope with no agent able to
    find it at all.

    Uses a larger candidate pool than the scoped agents (12 vs 4) since
    it's searching the whole document rather than a pre-narrowed slice."""
    where = {"source_file": protocol_filter} if protocol_filter else None
    try:
        hits = embed_store.query(question, n_results=candidate_pool_size, where=where)
    except Exception as e:
        _log_retrieval_error(f"[general retrieval error]: {e}")
        hits = []
    return rerank(question, hits, top_k=n_results)


def _answer_from_chunks(question: str, chunks: list[dict], agent_role: str) -> str:
    """Run retrieved chunks through the guardrails context scanner (LLM04/
    LLM08 - poisoned/untrusted context), then generate a grounded answer
    that cites source protocol + section."""
    if not chunks:
        return "No relevant information found in the indexed protocols."

    rag_chunks = [{"source": c["metadata"]["source_file"], "text": c["text"]} for c in chunks]
    scan_reports = scan_context(rag_chunks, policy="pharma_gxp")

    safe_chunks = [
        c for c, report in zip(chunks, scan_reports)
        if report.action.value != "block"
    ]
    if not safe_chunks:
        return "Retrieved context was flagged by guardrails and withheld from generation."

    def _truncate(text: str) -> str:
        if len(text) <= MAX_CHARS_PER_CHUNK_IN_PROMPT:
            return text
        return text[:MAX_CHARS_PER_CHUNK_IN_PROMPT] + "\n[...truncated - chunk exceeded prompt budget...]"

    context_blocks = "\n\n".join(
        f"[Source: {c['metadata']['source_file']} | Section: {c['metadata']['section_title']}]\n{_truncate(c['text'])}"
        for c in safe_chunks
    )

    system_msg = (
        f"You are a specialized clinical protocol {agent_role} agent. Answer ONLY using the "
        "provided source excerpts. Always cite which protocol file and section each fact comes "
        "from. If the excerpts don't contain enough information to answer, say so explicitly "
        "rather than inferring or guessing."
    )
    user_msg = f"Question: {question}\n\nSource excerpts:\n{context_blocks}"

    response = client.chat.completions.create(
        model=MODEL, max_tokens=600, temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def eligibility_node(state: PipelineState) -> dict:
    chunks = _retrieve_for_agent(state["question"], AGENT_SECTION_FILTERS["eligibility"], state.get("protocol_filter"))
    return {"eligibility_answer": _answer_from_chunks(state["question"], chunks, "eligibility-criteria")}


def endpoint_node(state: PipelineState) -> dict:
    chunks = _retrieve_for_agent(state["question"], AGENT_SECTION_FILTERS["endpoint"], state.get("protocol_filter"))
    return {"endpoint_answer": _answer_from_chunks(state["question"], chunks, "efficacy-endpoint")}


def design_node(state: PipelineState) -> dict:
    chunks = _retrieve_for_agent(state["question"], AGENT_SECTION_FILTERS["design"], state.get("protocol_filter"))
    return {"design_answer": _answer_from_chunks(state["question"], chunks, "study-design")}


def general_node(state: PipelineState) -> dict:
    chunks = _retrieve_general(state["question"], state.get("protocol_filter"))
    return {"general_answer": _answer_from_chunks(state["question"], chunks, "general-purpose research")}


NO_INFO_PLACEHOLDER = "No relevant information found in the indexed protocols."


def synthesis_node(state: PipelineState) -> dict:
    """Combine all four agents' answers into one coherent, still-traceable
    final response.

    Filters out agents that found nothing BEFORE building the prompt,
    rather than relying on the LLM to correctly weight a real answer
    against a numeric "majority" of empty findings. Confirmed via direct
    testing that the previous prose-only instruction ("give the general
    agent's findings primary weight...") was NOT reliable: the general
    agent produced a correct, cited answer in isolation, but synthesis
    still echoed "no relevant information... from ANY of the agents" when
    given all 4 raw answers together, apparently defaulting to a consensus
    framing across the 3-vs-1 split rather than following the weighting
    instruction. Removing the noise at the Python level - rather than
    hoping the LLM reliably follows a complex prose instruction every time
    - is the more robust fix.
    """
    agent_outputs = [
        ("Eligibility", state["eligibility_answer"]),
        ("Endpoint", state["endpoint_answer"]),
        ("Design", state["design_answer"]),
        ("General-purpose", state["general_answer"]),
    ]

    informative = [
        (label, answer) for label, answer in agent_outputs
        if answer.strip() and NO_INFO_PLACEHOLDER not in answer
    ]

    if not informative:
        # No LLM call needed - every agent genuinely found nothing.
        return {"final_answer": NO_INFO_PLACEHOLDER}

    combined = "\n\n".join(f"{label} agent findings:\n{answer}" for label, answer in informative)

    system_msg = (
        "You are a synthesis agent. Combine the specialist agent findings below into one "
        "clear, well-organized answer to the original question. Every agent shown below DID "
        "find relevant information - agents that found nothing have already been removed, so "
        "do not say information is missing or unavailable when real findings are provided "
        "below. Preserve all source citations (protocol file + section) exactly as given - "
        "never drop or invent citations, and never state a citation that doesn't appear "
        "verbatim in the findings below."
    )
    user_msg = f"Original question: {state['question']}\n\n{combined}"

    response = client.chat.completions.create(
        model=MODEL, max_tokens=800, temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )
    return {"final_answer": response.choices[0].message.content}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("eligibility", eligibility_node)
    graph.add_node("endpoint", endpoint_node)
    graph.add_node("design", design_node)
    graph.add_node("general", general_node)
    graph.add_node("synthesis", synthesis_node)

    # True parallel fan-out: all four agents run concurrently from START,
    # since they query independent slices of the index and don't depend on
    # each other's output. They converge (fan-in) at synthesis, which only
    # runs once all four have completed.
    graph.add_edge(START, "eligibility")
    graph.add_edge(START, "endpoint")
    graph.add_edge(START, "design")
    graph.add_edge(START, "general")
    graph.add_edge("eligibility", "synthesis")
    graph.add_edge("endpoint", "synthesis")
    graph.add_edge("design", "synthesis")
    graph.add_edge("general", "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


def ask(question: str, protocol_filter: str | None = None) -> str:
    try:
        protocol_filter = resolve_protocol_filter(protocol_filter)
    except ProtocolNotFoundError as e:
        # Short-circuit BEFORE running any agents - there is no point
        # querying ChromaDB with a filter guaranteed to match zero chunks.
        # This produces an honest, specific answer ("this protocol isn't
        # indexed here") instead of a misleading generic one ("the
        # protocol excerpts don't mention this") that the old silent-
        # proceed behavior produced after the full pipeline ran and found
        # nothing.
        available_list = "\n".join(f"  - {f}" for f in e.available) if e.available else "  (none indexed)"
        return (
            f"Protocol '{e.requested}' was not found among the indexed protocols in "
            f"this environment - it may not have been uploaded/processed here (for "
            f"example, real third-party protocol documents are deliberately excluded "
            f"from the public version of this repository). This is different from "
            f"'the protocol doesn't mention this topic' - the document itself isn't "
            f"available to search.\n\nProtocols actually indexed here:\n{available_list}"
        )

    app = build_graph()
    result = app.invoke({
        "question": question,
        "protocol_filter": protocol_filter,
        "eligibility_answer": "",
        "endpoint_answer": "",
        "design_answer": "",
        "general_answer": "",
        "final_answer": "",
    })
    return result["final_answer"]


if __name__ == "__main__":
    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered, exiting.")
        sys.exit(0)

    protocol_filter_input = input(
        "Optional: limit to one protocol filename (press Enter to search all): "
    ).strip()
    protocol_filter = protocol_filter_input or None

    print(f"\nQuestion: {question}")
    if protocol_filter:
        print(f"Protocol filter: {protocol_filter}")
    print()

    answer = ask(question, protocol_filter=protocol_filter)
    print(answer)