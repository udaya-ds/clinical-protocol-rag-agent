"""
Cross-encoder reranking for retrieved chunks.

embed_store.query() uses embedding similarity (a "bi-encoder": query and
chunk are embedded independently, then compared by distance). This is fast
but approximate - it can't always tell "primary endpoint" apart from
"secondary endpoint" mentions that embed similarly but mean different
things, and it can let noisy/irrelevant chunks (e.g. amendment-log table
references) slip into the top results just because they share vocabulary
with the query.

A cross-encoder reads the query and each candidate chunk TOGETHER in one
forward pass, producing a much more precise relevance score - at the cost
of being slower per-pair, which is why it's used as a second-stage filter
over a small candidate pool (e.g. top 12-20 from the bi-encoder), not as
the first-stage retriever over the whole index.
"""

from __future__ import annotations
import threading
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Same thread-safety concern as embeddings/embed_store.py's _get_model() -
# agentic_pipeline.py's specialist agents run via true parallel fan-out, so
# this lazy singleton needs a lock or concurrent threads can each construct
# their own model instance, defeating the point of caching.
_model: CrossEncoder | None = None
_model_lock = threading.Lock()


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check after acquiring the lock
                _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 4) -> list[dict]:
    """Re-score a candidate pool of chunks against the query using a
    cross-encoder, returning the top_k most relevant, each annotated with
    a `rerank_score` (higher = more relevant).

    `chunks` is expected to be the same dict shape embed_store.query()
    returns: {"text": ..., "metadata": ..., "distance": ...}.
    """
    if not chunks:
        return []

    model = _get_model()
    pairs = [(query, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked = []
    for chunk, score in scored[:top_k]:
        chunk_with_score = dict(chunk)
        chunk_with_score["rerank_score"] = float(score)
        reranked.append(chunk_with_score)
    return reranked


if __name__ == "__main__":
    # Quick sanity check with synthetic candidates - a noisy/irrelevant
    # chunk should score lower than a genuinely relevant one even if both
    # happen to share surface vocabulary with the query.
    query = "What is the primary efficacy endpoint for this trial?"
    fake_chunks = [
        {"text": "Primary Efficacy Endpoint: Progression-free survival (PFS).",
         "metadata": {"source_file": "a.txt", "section_title": "Efficacy Assessments"}, "distance": 0.2},
        {"text": "Table 3: Schedule of Activities, Section 1.3, Efficacy Endpoint references page 19.",
         "metadata": {"source_file": "a.txt", "section_title": "Table of Contents"}, "distance": 0.25},
        {"text": "Secondary Efficacy Endpoints: time to clinical improvement, quality of life score.",
         "metadata": {"source_file": "a.txt", "section_title": "Efficacy Assessments"}, "distance": 0.3},
    ]
    results = rerank(query, fake_chunks, top_k=3)
    for r in results:
        print(f"[{r['rerank_score']:.3f}] {r['text'][:70]}")