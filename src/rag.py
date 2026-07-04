"""Retrieval + grounded answer generation with page-level citations.

Usage:
    export ANTHROPIC_API_KEY=...
    python -m src.rag --index ./index --question "First-line therapy for stage 2 hypertension?"
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

from .ingest import EMBED_MODEL

SYSTEM_PROMPT = """You are a clinical evidence assistant. Answer ONLY from the provided guideline excerpts.
Rules:
- Cite every claim as [source, p. N].
- If the excerpts don't contain the answer, say so explicitly — never guess.
- Note guideline disagreements when excerpts conflict.
- End with: "For clinical decision support only — verify against the full guideline."
"""


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float


class ClinicalRAG:
    def __init__(self, index_dir: str):
        idx = Path(index_dir)
        self.index = faiss.read_index(str(idx / "chunks.faiss"))
        self.records = json.load(open(idx / "records.json"))
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = Anthropic()

    def retrieve(self, question: str, k: int = 6) -> list[Hit]:
        q = self.embedder.encode([question], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype=np.float32), k)
        return [
            Hit(self.records[i]["text"], self.records[i]["source"],
                self.records[i]["page"], float(s))
            for s, i in zip(scores[0], ids[0]) if i != -1
        ]

    def answer(self, question: str, k: int = 6) -> dict:
        hits = self.retrieve(question, k)
        context = "\n\n".join(
            f"[{h.source}, p. {h.page}] (relevance {h.score:.2f})\n{h.text}" for h in hits
        )
        msg = self.client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f"Guideline excerpts:\n\n{context}\n\nQuestion: {question}"}],
        )
        return {"answer": msg.content[0].text, "sources": hits}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="index")
    p.add_argument("--question", required=True)
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args()

    rag = ClinicalRAG(args.index)
    result = rag.answer(args.question, args.k)
    print(result["answer"])
    print("\n--- Retrieved sources ---")
    for h in result["sources"]:
        print(f"  {h.source} p.{h.page} (score {h.score:.2f})")


if __name__ == "__main__":
    main()
