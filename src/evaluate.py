"""Retrieval evaluation: recall@k and MRR against a hand-labeled question set.

Create eval_set.json like:
[
  {"question": "First-line therapy for stage 2 hypertension?",
   "relevant": [{"source": "jnc8.pdf", "page": 12}]}
]

Usage:
    python -m src.evaluate --index ./index --eval-set eval_set.json
"""

import argparse
import json

from .rag import ClinicalRAG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="index")
    p.add_argument("--eval-set", required=True)
    p.add_argument("--k", type=int, default=6)
    args = p.parse_args()

    rag = ClinicalRAG(args.index)
    eval_set = json.load(open(args.eval_set))

    hits_at_k, reciprocal_ranks = 0, []
    for item in eval_set:
        results = rag.retrieve(item["question"], k=args.k)
        relevant = {(r["source"], r["page"]) for r in item["relevant"]}
        rank = next((i + 1 for i, h in enumerate(results)
                     if (h.source, h.page) in relevant), None)
        if rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_set)
    print(f"Questions: {n}")
    print(f"Recall@{args.k}: {hits_at_k / n:.3f}")
    print(f"MRR: {sum(reciprocal_ranks) / n:.3f}")


if __name__ == "__main__":
    main()
