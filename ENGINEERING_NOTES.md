# Engineering Notes

This file contains the detailed build/verification history for each
component — what was tested, how, and real bugs found and fixed along the
way. The main [README](README.md) keeps a short, scannable feature list;
this file is for anyone who wants the full story behind it.

## Status detail

- **PDF text extraction** — straightforward `pypdf`-based batch extraction.

- **Structured extraction schema (Pydantic) + extract/validate/repair loop**
  — LLM extracts fields into JSON, validated against a Pydantic schema; on
  failure, the error is fed back to the LLM with a repair prompt, retried
  up to N times.

- **Input-side guardrails** (OWASP Top 10 for LLM Apps, 2025) — see
  `GUARDRAILS.md` and `agent/guardrails.py`; documents are structurally
  screened and checked for injection patterns before entering the
  extraction pipeline.

- **Chunking + ChromaDB embedding** (`ingestion/chunker.py` splits
  protocols by numbered section for semantic coherence; `embeddings/
  embed_store.py` embeds with SentenceTransformer and supports
  metadata-filtered retrieval, e.g. filter to "Inclusion Criteria"
  sections only). Text-only — CLIP/image embedding skipped, not relevant
  to text protocols.
  The chunker was hardened through four distinct, confirmed real-world
  bugs found via stress-testing against real (non-synthetic) protocols:
  digit-containing subsection titles (e.g. "For Phase 2/3") breaking the
  title regex and silently merging content into the wrong chunk; bare
  numbered list items being mistaken for section headers; curly
  apostrophes (e.g. "Hy's Law") failing to match; and section nesting
  deeper than 2 levels (e.g. "6.2.1.2 Placebo") being flattened into an
  oversized parent chunk. Each fix was verified with a before/after
  character-offset check confirming the target content actually moved
  into its own correctly-labeled chunk, plus a full regression check
  confirming the synthetic corpus still produces exactly 145 chunks.

- **Multi-agent LangGraph pipeline** (`agent/agentic_pipeline.py`) —
  three specialized agents (Eligibility, Endpoint, Design), each scoped
  to fixed section titles, plus a fourth **General-purpose agent** with
  NO section filter — all four run in true parallel fan-out, passing
  results through `scan_context()` guardrails (blocking poisoned chunks
  before generation), then converging at a Synthesis agent that
  preserves source citations.
  The general agent was added after a real finding during testing: a
  question about placebo preparation fell completely outside all three
  specialists' scope (drug/placebo prep lives under "Preparation/
  Handling/Storage/Accountability," not eligibility/endpoint/design), and
  the LLM filled the gap with a specific-sounding but ungrounded citation
  instead of saying so. Verified directly: with the three specialists
  mocked to find nothing (matching the original failure) and the general
  agent given the real confirmed placebo-prep chunk, the pipeline
  produces a correctly grounded, properly cited answer end-to-end.
  A separate, real synthesis bug was also found and fixed: when 3 of 4
  agents found nothing and 1 found the right answer, the synthesis LLM
  sometimes got pulled toward a "no information found" framing despite
  explicit instructions to weight the informative agent's answer as
  primary — fixed by filtering empty agent responses out of the synthesis
  prompt entirely at the code level, rather than relying on the LLM to
  correctly weigh a 3-vs-1 split through prose instructions alone.
  Two genuine thread-safety race conditions were also found and fixed:
  the embedding model and the ChromaDB client were both being lazily
  constructed with no locking, so concurrent agent threads could each
  build their own separate instance simultaneously. Fixed with
  double-checked locking; verified directly by stress-testing with 5
  genuinely concurrent threads and an artificial delay specifically
  designed to widen the race window, confirming exactly one instance gets
  constructed and shared across all threads.

- **Cross-encoder reranking** (`embeddings/reranker.py`) — two-stage
  retrieval: embedding similarity pulls a larger candidate pool per
  section, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
  reranks for precision before chunks reach the LLM. Verified with a
  mocked model that genuinely relevant chunks correctly outrank noisy
  matches (e.g. amendment-log table references) that share surface
  vocabulary with the query but aren't actually responsive.

- **RAG evaluation** (`evaluation/eval_rag.py`, `evaluation/
  rag_eval_set.json`, `evaluation/ir_metrics.py`) — measures retrieval
  correctness AND final answer quality separately, with a diagnostic
  distinguishing retrieval failures from generation failures. Adds
  standard IR ranking metrics (MRR, MAP, NDCG@k) on top of simple
  hit-rate, since hit-rate alone can't tell a relevant-chunk-at-rank-1
  apart from a relevant-chunk-at-rank-3 — verified `ir_metrics.py`'s
  formulas against hand-calculated worked examples (matches exactly) and
  confirmed the integration correctly differentiates two "HIT" cases by
  ranking quality alone.

- **HyDE query transformation** (`embeddings/hyde.py`) — generates a
  hypothetical, corpus-styled passage and embeds it AS A DOCUMENT (no
  query-instruction prefix) instead of embedding the raw question. Wired
  as an explicit alternate path (`embed_store.query_with_hyde()`) rather
  than the default, since our questions are typically specific enough
  that direct retrieval already performs well. Tested head-to-head
  against plain query retrieval on a real, specific factual question
  (gout trial primary endpoint): both found the right chunk at rank 1
  with similar scores, but HyDE's second-best result pulled in the wrong
  protocol while plain query's stayed within the correct one — concrete
  evidence that HyDE's value is for vague/underspecified queries, not
  already-specific ones, supporting the decision to keep it opt-in rather
  than default.

- **Hybrid search** (`embeddings/hybrid_search.py`) — BM25 (sparse/
  keyword) retrieval fused with dense embedding retrieval via Reciprocal
  Rank Fusion. Specifically targets a real gap: clinical protocols are
  full of exact identifiers (protocol numbers, drug names) that
  embeddings don't reliably handle. Verified directly against the real
  corpus with a striking result: for an exact protocol-number query
  ("BP-202606-731"), dense embedding retrieval didn't return the correct
  protocol anywhere in its top 5 results at all (best score still
  noise-level, ~0.94 cosine distance) — while BM25 found it at rank 1
  with a clear, decisive score margin (18.987 vs. the next-best 11.934,
  a 59% gap). RRF fusion math also confirmed correct (a chunk ranked in
  both lists outranks one ranked #1 in only one list).

- **Streamlit demo UI** (`app/streamlit_app.py`) — two tabs: Protocol
  Q&A (shows the final synthesized answer alongside each specialist
  agent's retrieved context, with visible guardrail status per chunk and
  rerank scores) and Deep Research (shows the router's tool selection,
  reasoning, the synthesized report, and raw per-tool outputs including
  a rendered SQL results table and PROC SQL display). Confirmed working
  live end-to-end, including diagnosing and fixing a `torchvision`
  missing-dependency error that only manifested under Streamlit's
  long-running process (not the one-shot CLI).

- **MCP server** (`mcp_server/server.py`, `mcp_server/client_demo.py`) —
  exposes `list_protocols`, `get_eligibility_criteria`,
  `get_primary_endpoint`, `ask_protocol_question`, and `deep_research`
  (5 tools total), wrapping the existing pipelines rather than
  reimplementing any logic. Several real, confirmed bugs found and fixed
  specifically in this subprocess/stdio context:
  - `.env` loading needs an explicit file path, since MCP spawns the
    server as a subprocess with its own clean environment — the original
    `load_dotenv()` silently failed to find the API key.
  - FastMCP's startup banner prints to the same stdout stream the stdio
    transport uses for its own JSON-RPC protocol, corrupting the
    connection (`show_banner=False` fixes this).
  - HuggingFace's model-loading progress bars (tqdm) write repeatedly to
    stderr, which crashed with `[Errno 9] Bad file descriptor` under this
    specific subprocess's stdio environment (`HF_HUB_DISABLE_PROGRESS_BARS=1`
    fixes this).
  - ChromaDB's SQLite-backed storage doesn't handle concurrent access
    well — running Streamlit and the MCP server against the same
    `chroma_db/` folder simultaneously caused silent retrieval failures
    that looked identical to "no relevant content found." Previously-
    silent exception handlers in the retrieval layer were changed to log
    the real error to stderr (verified safe — doesn't corrupt the stdout
    JSON-RPC channel) so this class of failure is now diagnosable instead
    of silent.
  - Added explicit `status: "found"/"not_found"` reporting when a
    requested `protocol_filename` doesn't match anything indexed,
    distinguishing "this document isn't available here" from "the
    document doesn't mention this" — previously both looked identical to
    the caller, including in the underlying CLI/Streamlit paths, where
    the same `ProtocolNotFoundError` short-circuit was added.

  Verified all 5 tools are correctly discovered and `list_protocols`
  returns real data from this repo's actual processed protocols.

- **Evaluation set** (`evaluation/eval_set.json` + `evaluation/
  run_eval.py`) — hand-verified ground truth for all 5 synthetic
  protocols (protocol number, phase, indication, design, enrollment,
  sites, arms, eligibility counts/key phrases, primary endpoint). Scorer
  logic verified using a simulated extraction output with deliberately
  injected errors — correctly caught both a wrong scalar value and a
  truncated eligibility list, while passing all correct fields. A real
  scoring bug was also found and fixed: fields with no ground truth yet
  (`expected: null`) were being auto-failed against any real extracted
  value, rather than being correctly skipped as "not yet verified."

- **ADaM synthetic trial dataset** (`data/sas_datasets/`) — CDISC-
  standard ADSL (subject-level: demographics, arm assignment,
  disposition) and ADAE (adverse events: term, severity, seriousness,
  relatedness) datasets, deterministically generated (seeded) and tied to
  the Gout trial protocol (BP-202606-797) for narrative consistency: 90
  subjects across 3 arms (30/30/30), 126 AE records with realistic
  severity/seriousness distributions. Synthetic — no real patient data,
  same spirit as the CDISC-generated protocol PDFs.

- **Text-to-SQL dataset lookup** (`agent/dataset_lookup.py`) — natural
  language question -> LLM-generated SQL (executed against an in-memory
  SQLite database built from the ADaM CSVs) + the equivalent PROC SQL
  syntax (displayed for SAS-audience readability, not executed — no
  licensed SAS environment exists here to run it against). Read-only
  validation gate rejects any non-SELECT statement before execution (the
  same OWASP-LLM06 Excessive Agency concern as elsewhere in this
  project's guardrails, applied to a new risk surface: an LLM generating
  executable code). Verified directly: 10/10 safety-validation test
  cases correct including a SQL-injection-style multi-statement attempt;
  real aggregation/filter/join queries execute correctly against the
  actual generated data; and a real bug was found and fixed —
  `json.loads(strict=False)` needed to tolerate literal newlines LLMs
  commonly emit inside multi-line PROC SQL string values, which strict
  JSON parsing rejects by default. A separate silent-failure bug was
  also fixed: a missing/misnamed CSV file used to produce a generic
  "no such table" SQLite error with no indication the real problem was a
  missing file at a specific path — now raises a clear, actionable
  `FileNotFoundError` instead.

- **Deep Research router** (`agent/deep_research_router.py`, `agent/
  web_search_tool.py`) — an LLM classifies each question's intent
  (temperature=0) and routes it to one or more of: protocol RAG, ADaM
  dataset SQL, or live web search (Tavily), running selected tools in
  true parallel (`ThreadPoolExecutor`, same fan-out pattern as the
  4-agent pipeline) before synthesizing a structured markdown report
  (Summary / Findings-by-source / Key Takeaways) with citations. Routing
  logic verified with 4 test cases (single-tool x2, multi-tool, and
  `ALL`-expansion) — all correct. Full end-to-end pipeline (route ->
  parallel execution -> fusion -> report) verified with mocked tool
  results, confirming exactly 2 LLM calls total (router + report
  generator, not one call per tool) and correct true-parallel execution.

  **Honest caveat**: live web search itself could not be tested in the
  sandbox this was originally built in (no external network access, no
  Tavily key available there) — `web_search_tool.py`'s request-building
  and response-parsing logic was verified by code review and mocked
  tests only at that point; the actual Tavily API integration needed
  (and received) verification with a real key in the developer's own
  environment.
