# Advanced RAG

Hybrid retrieval with query rewriting and reranking: the step from naive RAG towards a production pipeline. One of six RAG variants in a portfolio (Naive, Advanced, Production, Agentic, Graph, Multimodal) measured on the same corpus and dataset; the production system is [Personal-Knowledge-RAG](https://github.com/yashbhosale789/Personal-Knowledge-RAG).

## Pipeline

`query → LLM rewrite → dense + BM25 → reciprocal rank fusion → cross-encoder rerank → LLM answer`

| Stage | Choice |
|---|---|
| Loading | LangChain `PyPDFLoader` (one document per page); `.txt`, `.md`, `.docx` also read |
| Chunking | `RecursiveCharacterTextSplitter`, 512 characters, 64 overlap; chunks also saved to `storage/chunks.json` for BM25 |
| Embedding | `BAAI/bge-small-en-v1.5` (384 dims, local), query prefix `Represent this sentence: ` |
| Rewrite | LLM turns the question into a document-shaped search query (falls back to the question on error) |
| Retrieval | dense (Chroma) top 20 + BM25 (`rank-bm25`, whitespace tokens) top 20 |
| Fusion | reciprocal rank fusion, k = 60, top 10 |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 to generation |
| LLM | OpenRouter `nvidia/nemotron-3-nano-30b-a3b:free` (rewrite and answer) |

## Run

```bash
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
echo OPENROUTER_API_KEY=your_key > .env

# Index a corpus folder in place (default: data/documents). Delete storage/ before re-indexing:
# chunks get random ids, so a second ingest duplicates them in Chroma.
RAG_DATA_DIR=/path/to/corpus RAG_DATA_GLOB="*.pdf" python scripts/ingest.py
python scripts/advanced_query.py   # full pipeline; scripts/query.py is the naive dense-only path
```

## Retrieval results (provisional)

Measured 2026-11-07 on the 71 answerable questions of the Personal-Knowledge-RAG golden dataset (`eval/data/golden_dataset.json`, id `8f89d42be9`), retrieval only ($0, no LLM calls), with `python scripts/retrieval_eval.py <golden_dataset.json>`; per-question results for every stage are in `eval/retrieval_results.json`. **The LLM rewrite was skipped: queries are the raw questions.** The script scores `retrieve_stages()`, the same code the pipeline runs, with metric definitions copied from that project's eval harness. A hit means the labelled evidence span (source file + text, chunk-independent) is inside a retrieved chunk. **Provisional:** every variant is re-measured through one shared eval kit before the final cross-variant table.

Corpus: the 236 PDFs below, 795 pages → 2,222 chunks (id `2222:9e32e2edff`).

| Variant | Recall@1 | Recall@3 | Recall@5 | Success@3 | MRR |
|---|---|---|---|---|---|
| Naive RAG (dense top 5) | 0.41 | 0.65 | 0.73 | 0.59 | 0.59 |
| **Advanced RAG** (this repo, raw question, reranked) | 0.58 | 0.74 | 0.79 | 0.68 | 0.73 |
| Production RAG, raw question (top 3) | 0.70 | 0.83 | 0.83 | 0.82 | 0.81 |
| Production RAG, with query rewrite (top 3) | 0.68 | 0.88 | 0.88 | 0.85 | 0.83 |

Stages of this pipeline: dense recall@20 0.88, BM25 recall@20 0.74, fused recall@10 0.81, reranked recall@5 0.79.

Paired comparison (bootstrap CI + sign test, Bonferroni over 6 comparisons, α 0.0083):

- vs Production RAG with the same raw questions: no distinguishable difference on any metric at n = 71 (recall@3 −0.09, CI [−0.21, +0.03]).
- vs Production RAG with rewrites: recall@3 lower and distinguishable (−0.13, CI [−0.24, −0.04]); MRR, recall@1 and recall@5 not distinguishable.
- vs Naive RAG: not distinguishable (MRR +0.14, sign test p 0.013).

The differences from Production RAG are smaller chunks (512 vs 1024 characters), no PDF text clean-up (kerned words such as "A ventro" stay split, which hurts BM25), top 5 instead of 3, and a different rewrite prompt and model.

## Corpus and licence

The measurement corpus is [RAG-Multi-Corpus](https://github.com/udayallu/RAG-Multi-Corpus) (commit `39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3`): every PDF under `datasets/<Company>/pdf/`, 236 files, synthetic documents of five fictional companies. It is MIT-licensed, Copyright (c) 2025 UDAY ALLU. This repository ships one of its files as a sample (`data/documents/`, with the licence in `data/documents/LICENSE`); download the rest from the upstream repository.
