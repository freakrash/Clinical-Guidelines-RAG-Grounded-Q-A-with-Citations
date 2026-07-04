"""Ingest clinical guideline PDFs into a FAISS vector index.

Usage:
    python -m src.ingest --docs ./guidelines --index ./index
"""

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for each page."""
    reader = PdfReader(str(pdf_path))
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            yield i, text


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150):
    """Character chunks with overlap, snapped to sentence boundaries where possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            period = text.rfind(". ", start + chunk_size // 2, end)
            if period != -1:
                end = period + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 50]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs", required=True, help="Folder of guideline PDFs")
    p.add_argument("--index", default="index", help="Output index folder")
    args = p.parse_args()

    model = SentenceTransformer(EMBED_MODEL)
    records, texts = [], []

    for pdf in sorted(Path(args.docs).glob("*.pdf")):
        for page_num, page_text in extract_pages(pdf):
            for chunk in chunk_text(page_text):
                records.append({"source": pdf.name, "page": page_num, "text": chunk})
                texts.append(chunk)
        print(f"Ingested {pdf.name}")

    print(f"Embedding {len(texts)} chunks...")
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype=np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine similarity on normalized vectors
    index.add(emb)

    out = Path(args.index)
    out.mkdir(exist_ok=True)
    faiss.write_index(index, str(out / "chunks.faiss"))
    with open(out / "records.json", "w") as f:
        json.dump(records, f)
    print(f"Wrote index with {index.ntotal} vectors to {out}/")


if __name__ == "__main__":
    main()
