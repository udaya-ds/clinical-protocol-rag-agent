"""
HyDE (Hypothetical Document Embeddings) for retrieval.

Standard query embedding compares the QUESTION's embedding against chunk
embeddings - but a question ("What is the primary efficacy endpoint?") and
its answer ("Progression-free survival (PFS).") often sit in noticeably
different regions of embedding space, even when they're a perfect match.

HyDE sidesteps this by generating a hypothetical ANSWER first - a passage
written in the same style as the actual corpus (a protocol excerpt, not a
literal direct answer) - and embedding THAT instead. A plausible answer
passage tends to land much closer in embedding space to the real answer
passage than the bare question does.

Important detail: the hypothetical passage is embedded as a DOCUMENT, not
a query - no BGE query-instruction prefix here. The point of HyDE is to
make the query-side embedding look like corpus content, so embedding it
as a query would partially undo that.
"""

from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "agent"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from openai import OpenAI

client = OpenAI()
MODEL = "gpt-4o-mini"  # cheap/fast is fine here - this is a short generation, not the final answer

HYDE_SYSTEM_PROMPT = """You write short, plausible excerpts from clinical trial protocol \
documents. Given a question about a hypothetical trial, write a 2-4 sentence passage in \
the same style, tone, and structure as an actual protocol section that WOULD answer this \
question - as if quoting directly from a real protocol's Inclusion Criteria, Efficacy \
Assessments, or Study Design section (whichever fits the question).

Do not write "Here is a passage" or any meta-commentary. Do not answer in a conversational \
tone. Write ONLY the hypothetical protocol excerpt itself, as plain document text."""


def generate_hypothetical_passage(question: str) -> str:
    """Generate a short, corpus-styled hypothetical passage that would
    answer the question, for use as the retrieval query embedding."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=150,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content.strip()


def embed_hyde_query(question: str, embed_model) -> list[float]:
    """Generate the hypothetical passage and embed it AS A DOCUMENT
    (no query-instruction prefix) using the provided SentenceTransformer
    instance, so the caller controls which embedding model is used
    (keeping this module decoupled from embed_store.py's specific model
    choice)."""
    passage = generate_hypothetical_passage(question)
    embedding = embed_model.encode([passage], convert_to_numpy=True).tolist()[0]
    return embedding, passage


if __name__ == "__main__":
    # Demonstrate the transformation itself (no embedding model needed to
    # see this part working - just the LLM call).
    question = "What is the primary efficacy endpoint for the gout trial?"
    passage = generate_hypothetical_passage(question)
    print(f"Question: {question}\n")
    print(f"Hypothetical passage (this gets embedded, not the question):\n{passage}")
