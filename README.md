# Clinical Trial Protocol RAG + Agentic Extraction

A portfolio project applying RAG and agentic AI techniques to clinical trial
protocol documents — built to demonstrate the same skills that show up
repeatedly in current pharma AI Engineer job postings.

- Citation-grounded RAG over regulatory/clinical documents (mirrors tools like
  Elsevier's PharmaPendium AI)
- Agentic, multi-step structured extraction with auditability (mirrors published
  approaches for automated clinical trial protocol information extraction)
- Multi-agent orchestration (LangGraph) and MCP-style tool serving
- Deep Research-style multi-tool orchestration: an LLM router decides whether
  a question needs protocol text (RAG), structured trial data (text-to-SQL
  over ADaM datasets, with PROC SQL display for a SAS-literate audience), live
  web search, or a combination — then synthesizes a cited report

## Data

Sample protocols in `data/raw/` are **synthetic, CDISC-standard-aligned protocols**
generated via the [CDISC Dataset Generator](https://cdiscdataset.com)'s Custom
Study Protocol Generator. They contain no real patient data, no real sponsor, and
no real investigational product — generated explicitly for training/testing/
educational purposes. Safe to version-control and share publicly.

(Earlier in this project's development, a real sponsor's protocol — found publicly
on ClinicalTrials.gov's large-docs CDN — was considered but intentionally excluded
from this repo, since it carried a "for non-commercial use, subject to terms of use"
restriction. Real protocols may be used locally for one-off pipeline smoke-testing,
but should not be committed to a public portfolio repo.)

## Pipeline

```
data/raw/*.pdf
    │  ingestion/pdf_extract.py
    ▼
data/processed/*.txt
    │  agent/extract_protocol.py  (LLM extract -> Pydantic validate -> auto-repair loop,
    │                              guardrails-screened before processing)
    ▼
data/structured/structured_protocols.json
    │  ingestion/chunker.py  (section-based chunking)
    │  embeddings/embed_store.py  (BGE embeddings -> ChromaDB)
    ▼
Vector index (chunked by section: eligibility, endpoints, design, etc.)
    │  retrieval: embedding similarity (embed_store.query) OR
    │             hybrid BM25+dense fusion (embeddings/hybrid_search.py) OR
    │             HyDE hypothetical-passage embedding (embeddings/hyde.py)
    │  --> embeddings/reranker.py  (cross-encoder reranks candidate pool)
    │  --> agent/guardrails.py  (scan_context: blocks poisoned/untrusted chunks)
    ▼
agent/agentic_pipeline.py
    (LangGraph multi-agent, parallel fan-out: Eligibility / Endpoint / Design
     agents -> Synthesis agent)
    ▼
Structured, traceable answer (cites which protocol + section it came from)
    │
    ├──> app/streamlit_app.py        (interactive demo UI: Protocol Q&A tab +
    │                                  Deep Research tab)
    ├──> mcp_server/server.py        (MCP tools: list_protocols,
    │                                  get_eligibility_criteria,
    │                                  get_primary_endpoint,
    │                                  ask_protocol_question, deep_research)
    └──> evaluation/eval_rag.py      (retrieval + answer quality scoring:
                                       hit-rate, MRR, MAP, NDCG, keyword recall)

Deep Research (separate, parallel-tool orchestration layer):

  User question
       │
       ▼
  agent/deep_research_router.py  (LLM intent router, temperature=0)
       │
       ├─────────────┬──────────────────┐
       ▼              ▼                  ▼
  PROTOCOL_RAG    DATASET_SQL        WEB_SEARCH
  (agentic_       (agent/            (agent/
   pipeline.ask)   dataset_lookup.py: web_search_tool.py:
                    text -> SQL,       Tavily API)
                    SQLite execution
                    + PROC SQL display)
       │              │                  │
       └──────────────┴──────────────────┘
                       ▼
            Result fusion + structured
            markdown report (Summary /
            Findings / Key Takeaways)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste in your real OPENAI_API_KEY
# (optional) also add TAVILY_API_KEY if you want the Deep Research
# pipeline's web search tool to work - get a free key at https://tavily.com.
# Without it, Deep Research still works for protocol/dataset questions;
# only the WEB_SEARCH branch will report itself unavailable.
```

## Running the pipeline so far

```bash
# 1. Extract raw text from protocol PDFs
python ingestion/pdf_extract.py

# 2. Run LLM extraction + validation + auto-repair
python agent/extract_protocol.py

# 3. Build the chunked, embedded ChromaDB index
#    (this also runs a plain-query vs. HyDE comparison automatically - see
#    embeddings/hyde.py and embed_store.py's __main__ block)
python embeddings/embed_store.py

# 4. Ask a question through the multi-agent pipeline
#    (cross-encoder reranking runs automatically inside this - see
#    embeddings/reranker.py, wired into agentic_pipeline._retrieve_for_agent)
python agent/agentic_pipeline.py

# 5. Score extraction accuracy against hand-verified ground truth
python evaluation/run_eval.py

# 6. Score RETRIEVAL and ANSWER quality (separate from extraction accuracy) -
#    includes hit-rate, MRR, MAP, NDCG, and answer keyword recall, plus a
#    diagnostic distinguishing retrieval failures from generation failures
python evaluation/eval_rag.py

# 7. (Optional, standalone) See hybrid BM25+dense search in action - NOT
#    wired into the main pipeline by default; demonstrates the technique
#    against an exact-identifier query where dense retrieval alone struggles
python embeddings/hybrid_search.py

# 8. Generate the synthetic ADaM trial datasets (ADSL subject-level + ADAE
#    adverse events). Deterministic (seeded), tied to the Gout trial
#    protocol (BP-202606-797) for narrative consistency. Run this BEFORE
#    steps 9-10 - both Streamlit's "Deep Research" tab and the MCP
#    server's `deep_research` tool can route a question to the dataset
#    SQL lookup, which needs these CSVs to exist; without them, that
#    specific branch fails with a clear (but avoidable) FileNotFoundError.
python data/sas_datasets/generate_adam_datasets.py

# 9. Launch the interactive demo UI (Protocol Q&A tab + Deep Research tab)
streamlit run app/streamlit_app.py

# 10. (Optional) Verify the MCP server and run a tool-calling demo
#     (covers all 5 tools, including deep_research)
python mcp_server/client_demo.py

# 11. (Optional, standalone) Run the Deep Research CLI directly - same
#     pipeline as the Streamlit tab / MCP tool, useful for quick one-off
#     questions without launching the full UI.
python agent/deep_research_router.py
```

## Status

- [x] PDF text extraction
- [x] Structured extraction schema (Pydantic) + extract/validate/repair loop
- [x] Input-side guardrails (OWASP Top 10 for LLM Apps, 2025) — see `GUARDRAILS.md`
      and `agent/guardrails.py`; documents are structurally screened and checked
      for injection patterns before entering the extraction pipeline
- [x] Chunking + ChromaDB embedding (`ingestion/chunker.py` splits protocols by
      numbered section for semantic coherence; `embeddings/embed_store.py`
      embeds with SentenceTransformer and supports metadata-filtered retrieval,
      e.g. filter to "Inclusion Criteria" sections only). Text-only — CLIP/image
      embedding skipped, not relevant to text protocols. **Note**: model
      download requires internet access; ChromaDB plumbing verified
      independently with dummy embeddings in this sandbox.
- [x] Multi-agent LangGraph pipeline (`agent/agentic_pipeline.py`) — three
      specialized agents (Eligibility, Endpoint, Design), each scoped to
      fixed section titles, plus a fourth **General-purpose agent** with NO
      section filter — all four run in true parallel fan-out, passing
      results through `scan_context()` guardrails (blocking poisoned chunks
      before generation), then converging at a Synthesis agent that
      preserves source citations. The general agent was added after a real
      finding during testing: a question about placebo preparation fell
      completely outside all three specialists' scope (drug/placebo prep
      lives under "Preparation/Handling/Storage/Accountability," not
      eligibility/endpoint/design), and the LLM filled the gap with a
      specific-sounding but ungrounded citation instead of saying so.
      Verified directly: with the three specialists mocked to find nothing
      (matching the original failure) and the general agent given the real
      confirmed placebo-prep chunk, the pipeline now produces a correctly
      grounded, properly cited answer end-to-end.
- [x] Cross-encoder reranking (`embeddings/reranker.py`) — two-stage
      retrieval: embedding similarity pulls a larger candidate pool per
      section, then a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
      reranks for precision before chunks reach the LLM. Verified with a
      mocked model that genuinely relevant chunks correctly outrank noisy
      matches (e.g. amendment-log table references) that share surface
      vocabulary with the query but aren't actually responsive.
- [x] RAG evaluation (`evaluation/eval_rag.py`, `evaluation/rag_eval_set.json`,
      `evaluation/ir_metrics.py`) — measures retrieval correctness AND final
      answer quality separately, with a diagnostic distinguishing retrieval
      failures from generation failures. Adds standard IR ranking metrics
      (MRR, MAP, NDCG@k) on top of simple hit-rate, since hit-rate alone
      can't tell a relevant-chunk-at-rank-1 apart from a relevant-chunk-at-
      rank-3 — verified `ir_metrics.py`'s formulas against hand-calculated
      worked examples (matches exactly) and confirmed the integration
      correctly differentiates two "HIT" cases by ranking quality alone.
- [x] HyDE query transformation (`embeddings/hyde.py`) — generates a
      hypothetical, corpus-styled passage and embeds it AS A DOCUMENT (no
      query-instruction prefix) instead of embedding the raw question.
      Wired as an explicit alternate path (`embed_store.query_with_hyde()`)
      rather than the default, since our questions are typically specific
      enough that direct retrieval already performs well - kept as a
      documented, testable alternative for vaguer queries.
- [x] Hybrid search (`embeddings/hybrid_search.py`) — BM25 (sparse/keyword)
      retrieval fused with dense embedding retrieval via Reciprocal Rank
      Fusion. Specifically targets a real gap: clinical protocols are full
      of exact identifiers (protocol numbers, drug names) that embeddings
      don't reliably handle. Verified directly against the real corpus:
      BM25 correctly identified the right protocol for an exact protocol-
      number query by a clear score margin, and RRF fusion math confirmed
      correct (a chunk ranked in both lists outranks one ranked #1 in only
      one list).
- [x] Streamlit demo UI (`app/streamlit_app.py`) — interactive Q&A interface
      showing the final synthesized answer alongside each specialist
      agent's retrieved context, with visible guardrail status per chunk
      (allow / redact / block) and rerank scores. Verified the app starts
      cleanly with a headless smoke test in this sandbox; full end-to-end
      use needs your `OPENAI_API_KEY` + built index.
- [x] MCP server (`mcp_server/server.py`, `mcp_server/client_demo.py`) —
      exposes `list_protocols`, `get_eligibility_criteria`,
      `get_primary_endpoint`, `ask_protocol_question`, and `deep_research`
      (5 tools total) wrapping the existing pipelines rather than
      reimplementing any logic. Caught and fixed two real bugs while
      testing: (1) `.env` loading needs an explicit file path, since MCP
      spawns the server as a subprocess with its own clean environment —
      the original `load_dotenv()` silently failed to find the API key
      until this was fixed; (2) FastMCP's startup banner AND HuggingFace's
      model-loading progress bars both write to the same stdout/stderr
      channels the stdio transport uses for its own JSON-RPC protocol —
      confirmed directly that this corrupted the connection
      (`show_banner=False` and `HF_HUB_DISABLE_PROGRESS_BARS=1` fix both).
      Also added explicit `status: "found"/"not_found"` reporting when a
      requested `protocol_filename` doesn't match anything indexed,
      distinguishing "this document isn't available here" from "the
      document doesn't mention this" — previously both looked identical
      to the caller. Verified all 5 tools are correctly discovered and
      `list_protocols` returns real data from this repo's actual processed
      protocols.
- [x] Evaluation set (`evaluation/eval_set.json` + `evaluation/run_eval.py`) —
      hand-verified ground truth for all 5 synthetic protocols (protocol
      number, phase, indication, design, enrollment, sites, arms, eligibility
      counts/key phrases, primary endpoint). Scorer logic verified in this
      sandbox using a simulated extraction output with deliberately injected
      errors — correctly caught both a wrong scalar value and a truncated
      eligibility list, while passing all correct fields.
- [x] ADaM synthetic trial dataset (`data/sas_datasets/`) — CDISC-standard
      ADSL (subject-level: demographics, arm assignment, disposition) and
      ADAE (adverse events: term, severity, seriousness, relatedness)
      datasets, deterministically generated (seeded) and tied to the Gout
      trial protocol (BP-202606-797) for narrative consistency: 90 subjects
      across 3 arms (30/30/30), 126 AE records with realistic severity/
      seriousness distributions. Synthetic — no real patient data, same
      spirit as the CDISC-generated protocol PDFs.
- [x] Text-to-SQL dataset lookup (`agent/dataset_lookup.py`) — natural
      language question -> LLM-generated SQL (executed against an in-memory
      SQLite database built from the ADaM CSVs) + the equivalent PROC SQL
      syntax (displayed for SAS-audience readability, not executed — no
      licensed SAS environment exists here to run it against). Read-only
      validation gate rejects any non-SELECT statement before execution
      (the same OWASP-LLM06 Excessive Agency concern as elsewhere in this
      project's guardrails, applied to a new risk surface: an LLM
      generating executable code). Verified directly: 10/10 safety-
      validation test cases correct including a SQL-injection-style
      multi-statement attempt, real aggregation/filter/join queries
      execute correctly against the actual generated data, and a real bug
      was found and fixed — `json.loads(strict=False)` needed to tolerate
      literal newlines LLMs commonly emit inside multi-line PROC SQL
      string values, which strict JSON parsing rejects by default.
- [x] Deep Research router (`agent/deep_research_router.py`,
      `agent/web_search_tool.py`) — an LLM classifies each question's
      intent (temperature=0) and routes it to one or more of: protocol RAG,
      ADaM dataset SQL, or live web search (Tavily), running selected tools
      in true parallel (`ThreadPoolExecutor`, same fan-out pattern as the
      4-agent pipeline) before synthesizing a structured markdown report
      (Summary / Findings-by-source / Key Takeaways) with citations.
      Routing logic verified with 4 test cases (single-tool x2, multi-tool,
      and `ALL`-expansion) — all correct. Full end-to-end pipeline (route
      -> parallel execution -> fusion -> report) verified with mocked tool
      results, confirming exactly 2 LLM calls total (router + report
      generator, not one call per tool) and correct true-parallel
      execution. **Honest caveat**: live web search itself could not be
      tested in this sandbox (no external network access, no Tavily key
      available here) — `web_search_tool.py`'s request-building and
      response-parsing logic is verified by code review and mocked tests
      only; the actual Tavily API integration needs verification with a
      real key in your own environment.

## Known limitations — chunked extraction on real (non-synthetic) protocols

`agent/extract_protocol.py`'s chunked-extraction path (used for documents
too large for a single LLM call — i.e. real protocols, not the synthetic
corpus) was stress-tested against the two real protocols in
`local_test_data/`. This surfaced a real, instructive split between what's
fixable with better prompting/merge logic and what's a harder, inherent
limitation of the approach:

**Confirmed fixed**, with before/after evidence:
- **Exclusion criteria vs. discontinuation criteria conflation** (AZ
  protocol) — these two lists use similar language ("pregnancy,"
  "lab-confirmed infection," "serious adverse event") but mean different
  things (pre-enrollment exclusion vs. mid-study withdrawal). Adding an
  explicit disambiguation rule to the extraction prompt took this from
  0/3 expected phrases found to 3/3.
- **Merge-time pollution and wrong-candidate selection** for
  `treatment_arms` and `endpoints.primary` — replaced naive "first/longest
  list wins" with field-aware merge logic (shortest-description-per-drug-
  group for arms; safety-keyword filtering for the primary endpoint).
  Verified directly against the exact confirmed failure cases.

**Confirmed NOT fixed, and likely inherent to the approach**:
- **Pfizer's primary endpoint** still extracts safety/reactogenicity
  content instead of the efficacy endpoint. Root cause: Pfizer's protocol
  is a single combined Phase 1/2/3 trial that legitimately lists *several*
  distinct "primary" endpoints side by side (Phase 1 reactogenicity, an
  immunogenicity non-inferiority comparison, AND Phase 2/3 efficacy) — so
  "which one is *the* primary endpoint" is itself a document-specific
  judgment call, not a clear extraction error a keyword filter can
  reliably resolve.
- **Run-to-run extraction variance** — re-running extraction on the
  identical AZ document produced *different* (and differently wrong)
  `treatment_arms` results across runs (verbose dosing descriptions in one
  run, overly generic `['saline', 'vaccine']` in another), despite
  `temperature=0`. For sufficiently long/complex source text, the model's
  chosen level of abstraction when summarizing isn't perfectly stable run
  to run, and the merge logic can only choose among whatever candidates a
  given run actually produced.

**Practical takeaway**: the chunked extraction approach is reliable for
facts that appear cleanly and consistently in one chunk (protocol number,
enrollment, phase, most eligibility criteria), but is inherently less
reliable for fields requiring document-wide disambiguation between
multiple superficially-similar candidates. The Q&A pipeline
(`agent/agentic_pipeline.py`) does not share this limitation to the same
degree, since its multi-agent retrieval is scoped and reranked per-query
rather than merged once across the whole document up front.

## Future work (deliberately not built — documented gaps, not oversights)

- **Embedding quantization** — scalar/binary quantization for storage/speed
  at scale; not implemented since this corpus (a few hundred chunks) is far
  too small for quantization to matter in practice.
- **Knowledge graph retrieval** — modeling entity relationships (e.g. Drug →
  Trial → Indication → Endpoint) for relationship-style queries text
  retrieval handles poorly. A genuinely separate sub-project in scope
  (entity extraction + graph schema + a graph library), not a quick add-on.
- **Embedding fine-tuning** — training the embedding model on real clinical
  protocol query-document pairs, rather than relying on a strong pretrained
  general model (BGE-large).
