# Clinical Trial Protocol RAG + Agentic Extraction

A portfolio project applying RAG and agentic AI techniques to clinical trial
protocol documents.

- Citation-grounded RAG over regulatory/clinical documents
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

All components below are built and verified — see [ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)
for the full build/verification history, including real bugs found and
fixed along the way (chunking edge cases, thread-safety races, MCP
subprocess quirks, and more).

- [x] PDF text extraction (`ingestion/pdf_extract.py`)
- [x] Structured extraction with Pydantic validation + auto-repair
      (`agent/extract_protocol.py`, `agent/extraction_schema.py`)
- [x] Input-side guardrails, OWASP Top 10 for LLM Apps (`agent/guardrails.py`,
      `GUARDRAILS.md`)
- [x] Section-based chunking + ChromaDB embedding (`ingestion/chunker.py`,
      `embeddings/embed_store.py`)
- [x] 4-agent LangGraph pipeline with true parallel fan-out + synthesis
      (`agent/agentic_pipeline.py`)
- [x] Cross-encoder reranking (`embeddings/reranker.py`)
- [x] RAG evaluation: hit-rate, MRR, MAP, NDCG, retrieval-vs-generation
      diagnostics (`evaluation/eval_rag.py`, `evaluation/ir_metrics.py`)
- [x] HyDE query transformation, opt-in alternate retrieval path
      (`embeddings/hyde.py`)
- [x] Hybrid BM25 + dense search via Reciprocal Rank Fusion
      (`embeddings/hybrid_search.py`)
- [x] Streamlit demo UI — Protocol Q&A + Deep Research tabs
      (`app/streamlit_app.py`)
- [x] MCP server, 5 tools (`mcp_server/server.py`, `mcp_server/client_demo.py`)
- [x] Extraction accuracy evaluation set (`evaluation/eval_set.json`,
      `evaluation/run_eval.py`)
- [x] Synthetic ADaM trial dataset, ADSL + ADAE (`data/sas_datasets/`)
- [x] Text-to-SQL dataset lookup with PROC SQL display
      (`agent/dataset_lookup.py`)
- [x] Deep Research router: intent classification -> parallel tool
      execution -> synthesized report (`agent/deep_research_router.py`,
      `agent/web_search_tool.py`)

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
- **The combined Phase 1/2/3 trial's primary endpoint** still extracts
  safety/reactogenicity content instead of the efficacy endpoint. Root
  cause: that protocol is a single combined Phase 1/2/3 trial that
  legitimately lists *several* distinct "primary" endpoints side by side
  (Phase 1 reactogenicity, an immunogenicity non-inferiority comparison,
  AND Phase 2/3 efficacy) — so "which one is *the* primary endpoint" is
  itself a document-specific judgment call, not a clear extraction error
  a keyword filter can reliably resolve.
- **Run-to-run extraction variance** — re-running extraction on the
  identical real-protocol document produced *different* (and differently
  wrong) `treatment_arms` results across runs (verbose dosing descriptions
  in one run, overly generic `['saline', 'vaccine']` in another), despite
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
