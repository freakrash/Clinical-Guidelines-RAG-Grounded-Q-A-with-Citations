# Clinical Guidelines RAG — Grounded Q&A with Citations

Retrieval-augmented generation over clinical guideline PDFs (e.g., WHO, NICE, AHA). Every answer is grounded in retrieved excerpts and cited to source and page — the model is instructed to refuse rather than guess when evidence is missing.

Example interaction (illustrative — regenerate with your own guideline set):

**Question:** *"What is the first-line pharmacological therapy for stage 2 hypertension?"*

**Answer:** Thiazide-type diuretics, CCBs, ACE inhibitors, or ARBs are recommended as initial therapy; stage 2 typically warrants two first-line agents from different classes [aha-hypertension-2017.pdf, p. 42]. *For clinical decision support only — verify against the full guideline.*

## Why this design matters for medical use

- **Page-level citations** — clinicians can verify every claim against the source PDF
- **Refusal over hallucination** — system prompt forbids answering beyond retrieved evidence
- **Conflict surfacing** — flags when guidelines disagree instead of silently picking one
- **Measurable retrieval quality** — recall@k and MRR evaluation against a labeled question set

## Stack

Sentence-Transformers embeddings (all-MiniLM-L6-v2) → FAISS cosine index → Claude for grounded synthesis. Sentence-boundary chunking with overlap, page-level metadata carried through the pipeline.

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# 1. Build the index from a folder of guideline PDFs
python -m src.ingest --docs ./guidelines --index ./index

# 2. Ask a question
python -m src.rag --index ./index --question "First-line therapy for stage 2 hypertension?"

# 3. Evaluate retrieval on a labeled set
python -m src.evaluate --index ./index --eval-set eval_set.json
```

## Structure

```
src/
  ingest.py    # PDF → chunks → embeddings → FAISS
  rag.py       # Retrieval + cited answer generation
  evaluate.py  # recall@k, MRR
```

## Notes

Decision-support demo on public guidelines — not a medical device; outputs must be verified by a qualified clinician.
