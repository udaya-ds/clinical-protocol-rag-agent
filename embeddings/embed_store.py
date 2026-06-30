"""
Embed protocol chunks and store them in ChromaDB for similarity +
metadata-filtered retrieval.

Text-only: image/multimodal embedding is intentionally skipped here since
clinical trial protocols in this corpus are text, not images.
"""

from __future__ import annotations
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "ingestion"))

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_all, Chunk

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # stronger retrieval quality than MiniLM, larger/slower

# BGE models are trained with an instruction prefix applied ONLY to queries,
# never to the documents/chunks being indexed - this asymmetry is documented
# BGE behavior, not optional. Skipping it on queries measurably hurts
# retrieval quality (the query and passage embeddings are no longer in the
# space the model was actually trained to compare).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

COLLECTION_NAME = "clinical_protocols"
DB_PATH = str(Path(__file__).parent.parent / "chroma_db")

# Cached at module level so repeated calls (e.g. three specialist agents
# each calling query() once per scoped section, multiple times per question)
# reuse the same loaded model instead of constructing a fresh
# SentenceTransformer on every single call. Guarded by a lock because
# agentic_pipeline.py's specialist agents run via TRUE PARALLEL fan-out
# (LangGraph executes independent nodes concurrently, typically via a
# thread pool) - without the lock, two threads can both see `_model is
# None` at the same instant and each construct their own instance, which
# is exactly what caused multiple "Loading weights" events despite caching
# being in place.
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()

# Same thread-safety concern as the embedding model above, but for the
# ChromaDB client itself: get_collection() used to construct a fresh
# chromadb.PersistentClient on EVERY call, with no caching at all. With
# agentic_pipeline.py's true parallel agent fan-out, many threads can end
# up simultaneously trying to open a persistent client against the same
# on-disk SQLite-backed path - confirmed in practice to produce "Could not
# connect to tenant default_tenant" errors under concurrent access. Caching
# the client with the same double-checked-lock pattern fixes this.
_chroma_client = None
_chroma_client_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # re-check after acquiring the lock
                _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_collection(reset: bool = False):
    global _chroma_client
    if _chroma_client is None:
        with _chroma_client_lock:
            if _chroma_client is None:  # re-check after acquiring the lock
                _chroma_client = chromadb.PersistentClient(path=DB_PATH)

    if reset:
        try:
            _chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return _chroma_client.get_or_create_collection(COLLECTION_NAME)


def build_index(processed_dir: str | Path, reset: bool = True) -> int:
    """Chunk all processed protocols, embed them, and load into ChromaDB."""
    chunks: list[Chunk] = chunk_all(processed_dir)
    if not chunks:
        print("No chunks found - run ingestion/pdf_extract.py first.")
        return 0

    model = _get_model()
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    collection = get_collection(reset=reset)
    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.to_metadata() for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into ChromaDB collection '{COLLECTION_NAME}'")
    return len(chunks)


def query(text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Similarity search, optionally filtered by metadata
    (e.g. where={"section_title": "Inclusion Criteria"})."""
    model = _get_model()
    # Instruction prefix applies to the QUERY only - never to indexed chunks.
    query_embedding = model.encode([QUERY_INSTRUCTION + text], convert_to_numpy=True).tolist()

    collection = get_collection(reset=False)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
    )

    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return out


def query_with_hyde(text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
    """Same as query(), but embeds a HyDE-generated hypothetical passage
    instead of the raw question (see embeddings/hyde.py). Uses an extra LLM
    call per query, so it's slower/costlier than query() - intended as an
    alternative to compare against, not a default replacement, since our
    questions are typically specific enough that direct retrieval already
    works well."""
    sys.path.insert(0, str(Path(__file__).parent))
    from hyde import embed_hyde_query

    model = _get_model()
    hyde_embedding, hypothetical_passage = embed_hyde_query(text, model)

    collection = get_collection(reset=False)
    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=n_results,
        where=where,
    )

    out = []
    for i in range(len(results["ids"][0])):
        out.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "hypothetical_passage": hypothetical_passage,
        })
    return out


if __name__ == "__main__":
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    build_index(processed_dir, reset=True)

    print("\n--- Test query: 'eligibility criteria for enrollment' ---")
    for r in query("eligibility criteria for enrollment", n_results=3):
        print(f"[{r['distance']:.3f}] {r['metadata']['source_file']} | "
              f"{r['metadata']['section_number']} {r['metadata']['section_title']}")

    print("\n--- Test query: 'primary efficacy endpoint', filtered to Efficacy Assessments section ---")
    for r in query("primary efficacy endpoint", n_results=3,
                    where={"section_title": "Efficacy Assessments"}):
        print(f"[{r['distance']:.3f}] {r['metadata']['source_file']} | "
              f"{r['metadata']['section_number']} {r['metadata']['section_title']}")

    print("\n--- Comparison: plain query vs. HyDE for the same question ---")
    test_question = "What is the primary efficacy endpoint for the gout trial?"
    print(f"Question: {test_question}\n")

    print("Plain query embedding:")
    for r in query(test_question, n_results=3):
        print(f"  [{r['distance']:.3f}] {r['metadata']['source_file']} | {r['metadata']['section_title']}")

    print("\nHyDE (hypothetical passage embedding):")
    hyde_results = query_with_hyde(test_question, n_results=3)
    if hyde_results:
        print(f"  Hypothetical passage used: {hyde_results[0]['hypothetical_passage']!r}\n")
    for r in hyde_results:
        print(f"  [{r['distance']:.3f}] {r['metadata']['source_file']} | {r['metadata']['section_title']}")
