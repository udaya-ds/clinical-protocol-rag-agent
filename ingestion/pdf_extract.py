"""
Extracts raw text from clinical trial protocol PDFs.

These are CDISC Dataset Generator synthetic protocols (cdiscdataset.com),
generated purely for training/portfolio purposes - no real patient or
sponsor data. Safe to commit alongside the repo.

process_pdfs_directory() batch-processes an entire folder of protocol PDFs
with per-file error handling and empty-text warnings, so one corrupt or
scanned/image-only PDF doesn't halt the whole run. Pages are joined with
"\n\n" rather than concatenated directly, so section headers don't get
glued together across page boundaries (which would break chunker.py's
section-detection regex).
"""

from __future__ import annotations
import glob
import os
from pathlib import Path

from pypdf import PdfReader
from tqdm import tqdm


def extract_text(pdf_path: str | Path) -> str:
    """Extract all text from a single PDF, page by page."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def process_pdfs_directory(directory_path: str | Path) -> list[dict]:
    """Process every PDF in a directory. Returns a list of
    {"text": ..., "source": filename} dicts, skipping files that fail to
    extract or yield no text (e.g. scanned/image-only PDFs that would need
    OCR), and printing a warning/error for each so nothing fails silently.
    """
    pdf_files = sorted(glob.glob(os.path.join(str(directory_path), "*.pdf")))
    documents: list[dict] = []

    for pdf_file in tqdm(pdf_files, desc="Processing PDF files"):
        try:
            text = extract_text(pdf_file)
            if text.strip():
                documents.append({
                    "text": text,
                    "source": os.path.basename(pdf_file),
                })
            else:
                print(f"Warning: No text extracted from {pdf_file} (likely scanned/image-only - needs OCR)")
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    return documents


def extract_all(raw_dir: str | Path) -> dict[str, str]:
    """Backwards-compatible wrapper: {filename: text} instead of the
    list-of-dicts shape, for callers that used the earlier interface."""
    return {doc["source"]: doc["text"] for doc in process_pdfs_directory(raw_dir)}


if __name__ == "__main__":
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    documents = process_pdfs_directory(raw_dir)

    for doc in documents:
        out_path = processed_dir / (Path(doc["source"]).stem + ".txt")
        out_path.write_text(doc["text"], encoding="utf-8")
        print(f"Extracted {doc['source']} -> {out_path.name} ({len(doc['text'])} chars)")

    print(f"\nProcessed {len(documents)}/{len(list(raw_dir.glob('*.pdf')))} PDFs successfully.")
