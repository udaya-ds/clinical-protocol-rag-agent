# Guardrails — OWASP Top 10 for LLM Apps (2025), applied to this pipeline

Reference: https://genai.owasp.org/llm-top-10/

The guardrail engine in `agent/guardrails.py` is architected after the R
package **`llmshieldr`** (presented at R/Pharma GenAI Day 2026 by Indraneel
Chakraborty — *"'Ignore All Previous Instructions' and Other Things Your LLM
Shouldn't Do in Pharma"*), which implements OWASP-mapped guardrails with a
pharma-tuned policy (`pharma_gxp`) on top of an enterprise baseline. This
project is a from-scratch Python port of that same pattern — rules tagged
by OWASP risk + severity, a policy that activates a subset of rules with
block/redact thresholds, and source-trust-aware scanning of RAG context —
adapted to the protocol-extraction use case.

This maps each OWASP risk to a concrete control in this project. Listing
this explicitly is also a good portfolio signal — pharma AI roles care
about auditability and risk-awareness, not just model accuracy.

| # | Risk | Where it applies here | Control implemented / planned |
|---|------|------------------------|-------------------------------|
| LLM01 | Prompt Injection | Protocol PDFs/text are untrusted input — a malicious or malformed protocol could contain text designed to hijack the extraction prompt (e.g. "ignore previous instructions and output X") | `scan_prompt(text, policy="pharma_gxp")` blocks on `llm01.injection.*` rule matches before text reaches the LLM. Extracted text is also always wrapped as **data** inside a fixed prompt template (`build_extraction_prompt`), never concatenated into the system role — defense in depth, not just regex matching. |
| LLM02 | Sensitive Information Disclosure | Real protocols (tested locally only, see `local_test_data/`) may contain sponsor-confidential info; synthetic CDISC protocols contain none; a future "ask about a patient" feature could leak PII/PHI | `scan_prompt()` includes clinical-tuned PII/PHI rules (named patient, MRN/USUBJID, SSN, email, phone) with configurable redaction (`default`/`hash`/`mask`/`drop`). Real protocols never committed to git (`.gitignore`). |
| LLM03 | Supply Chain | Dependencies (openai, sentence-transformers, chromadb, langchain) | `requirements.txt` pins minimum versions; review/upgrade deliberately rather than auto-pulling `latest`. CDISC-generated data's provenance is documented in README. |
| LLM04 | Data & Model Poisoning | A malicious/corrupted protocol document fed into the embedding store could pollute retrieval for all future queries | `screen_document()` runs the pharma_gxp policy scan on every document before extraction. Planned: `scan_context()` will screen each retrieved chunk (with source-trust checks) before it's added to a generation prompt — already implemented and demoed in `agent/guardrails.py`'s poisoned-RAG-context example. |
| LLM05 | Improper Output Handling | Raw LLM output is JSON that downstream code/UI will consume | Already implemented: `StudyProtocol` Pydantic schema + validate/repair loop in `extract_protocol.py` rejects malformed JSON before it reaches any downstream consumer. Never `eval()` or directly render raw LLM output. |
| LLM06 | Excessive Agency | The planned multi-agent (LangGraph) layer will have agents calling tools | Agent tools will be strictly read-only (`get_eligibility_criteria`, `get_primary_endpoints`, etc.) — no tool will have file-write, delete, or external-network capability. No agent auto-executes code. |
| LLM07 | System Prompt Leakage | System prompts contain extraction rules; shouldn't be echoed back to end users | Agent/UI layer will strip system-prompt content from any user-facing output; system prompts contain no secrets (no API keys, no internal-only logic) by design, so leakage risk is low-severity even if it happened. |
| LLM08 | Vector & Embedding Weaknesses | ChromaDB index — embedding inversion, unauthorized cross-collection access | Planned: separate ChromaDB collections per data source (synthetic vs. any future real corpus), no shared collection; local-only persistence (no public-facing vector DB endpoint in this portfolio version). |
| LLM09 | Misinformation | LLM could hallucinate an endpoint, eligibility criterion, or arm that isn't in the source text | This is the core design goal of the retrieval+citation approach: structured output is required to be grounded in retrieved chunks (planned UI shows source chunk next to generated answer). Evaluation set (`evaluation/eval_set.json`, planned) will hand-check extraction accuracy on a sample of protocols. |
| LLM10 | Unbounded Consumption | Real protocols are 300K+ characters — naive whole-document prompts (as built initially) don't scale and could balloon token usage/cost | Already surfaced by testing against real-world trial protocols locally (see README) — this is *why* the chunking/RAG layer is necessary rather than optional. `max_retries=3` caps the auto-repair loop; chunked retrieval (planned) caps per-call token usage regardless of source document size. |

## Practical notes for this specific domain

- **Clinical accuracy stakes are higher than a restaurant demo.** A hallucinated
  eligibility criterion or endpoint isn't a quirky chatbot answer — it's
  exactly the kind of error pharma AI tooling is built to prevent. Treat
  LLM09 (Misinformation) as the highest-priority risk in this project,
  not a checkbox.
- **Synthetic-data-only commits remove most LLM02/LLM04 risk surface** for the
  public repo version. If you later add a private branch with real protocols
  for further testing, re-apply these controls more strictly (access control,
  redaction, no public exposure).
