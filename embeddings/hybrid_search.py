"""
Hybrid search: combines sparse (BM25 keyword) retrieval with dense
(embedding similarity) retrieval, fused via Reciprocal Rank Fusion (RRF).

Why this matters for THIS corpus specifically: clinical protocols are full
of exact identifiers - protocol numbers ("BP-202606-731"), specific drug
names, exact dosage strings. Dense/embedding retrieval is good at semantic
meaning but can struggle with exact-token matches like these (an embedding
model has no special reason to place "BP-202606-731" close to anything in
particular). BM25 (sparse, term-frequency-based) is the opposite: terrible
at "what concept is this about" but excellent at "does this exact term
appear." Fusing both gets the benefit of each.

BM25 index here is built in-memory from the chunk corpus at call time
(not persisted to disk) - at this corpus's scale (a few hundred chunks),
rebuilding it per call is fast enough that persistence isn't worth the
added complexity. For a much larger corpus, you'd want to build it once
and cache/persist it, the same way embed_store.py persists ChromaDB.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))
from chunker import chunk_all, Chunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer - deliberately basic since
    BM25's strength here is exact/near-exact term matching (protocol
    numbers, drug names), not linguistic sophistication."""
    return TOKEN_PATTERN.findall(text.lower())


def _filter_chunks(chunks: list[Chunk], section_titles: list[str] | None,
                    protocol_filter: str | None) -> list[Chunk]:
    filtered = chunks
    if section_titles is not None:
        filtered = [c for c in filtered if c.section_title in section_titles]
    if protocol_filter is not None:
        filtered = [c for c in filtered if c.source_file == protocol_filter]
    return filtered


def bm25_search(question: str, processed_dir: str | Path, n_results: int = 10,
                 section_titles: list[str] | None = None,
                 protocol_filter: str | None = None) -> list[dict]:
    """BM25 keyword search over the chunk corpus, optionally filtered the
    same way embed_store/agentic_pipeline filter (by section title and/or
    source protocol file)."""
    all_chunks = chunk_all(processed_dir)
    candidates = _filter_chunks(all_chunks, section_titles, protocol_filter)
    if not candidates:
        return []

    tokenized_corpus = [tokenize(c.text) for c in candidates]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    out = []
    for chunk, score in ranked[:n_results]:
        out.append({
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.to_metadata(),
            "bm25_score": float(score),
        })
    return out


def reciprocal_rank_fusion(dense_results: list[dict], sparse_results: list[dict],
                            k: int = 60, top_k: int = 5) -> list[dict]:
    """Fuse two ranked result lists (matched by chunk `id`) using RRF:
    score(doc) = sum over each list it appears in of 1 / (k + rank),
    where rank is 1-indexed position in that list. A doc appearing near
    the top of EITHER list scores well; appearing near the top of BOTH
    scores best. k=60 is the standard default from the original RRF paper -
    it dampens the impact of any single very-high rank so no one list
    dominates the fused result.
    """
    fused_scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        fused_scores[item["id"]] = fused_scores.get(item["id"], 0.0) + 1.0 / (k + rank)
        chunk_lookup[item["id"]] = item

    for rank, item in enumerate(sparse_results, start=1):
        fused_scores[item["id"]] = fused_scores.get(item["id"], 0.0) + 1.0 / (k + rank)
        chunk_lookup.setdefault(item["id"], item)

    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

    out = []
    for cid in ranked_ids[:top_k]:
        result = dict(chunk_lookup[cid])
        result["rrf_score"] = fused_scores[cid]
        out.append(result)
    return out


def hybrid_query(question: str, processed_dir: str | Path, embed_store_module,
                  n_results: int = 5, candidate_pool: int = 10,
                  where: dict | None = None,
                  section_titles: list[str] | None = None,
                  protocol_filter: str | None = None) -> list[dict]:
    """Run both dense (via the passed-in embed_store module, so this stays
    decoupled from which embedding model/index embed_store.py uses) and
    sparse (BM25) retrieval, then fuse with RRF."""
    dense_results = embed_store_module.query(question, n_results=candidate_pool, where=where)
    sparse_results = bm25_search(question, processed_dir, n_results=candidate_pool,
                                  section_titles=section_titles, protocol_filter=protocol_filter)
    return reciprocal_rank_fusion(dense_results, sparse_results, top_k=n_results)


if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"

    # The exact scenario hybrid search is meant to help with: a query built
    # around an exact identifier, which dense/embedding retrieval has no
    # particular reason to handle well.
    test_query = "BP-202606-731"
    print(f"BM25-only search for exact identifier: {test_query!r}\n")
    results = bm25_search(test_query, processed_dir, n_results=3)
    for r in results:
        print(f"  [bm25={r['bm25_score']:.3f}] {r['metadata']['source_file']} | "
              f"{r['metadata']['section_title']}")

    print("\n(Compare against embed_store.query() for the same identifier to see "
          "whether dense retrieval alone finds the right protocol as reliably.)")
