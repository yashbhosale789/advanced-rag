"""Retrieval-only eval (no LLM, $0): recall@k, success@k and MRR on every answerable question.

The dataset is the Personal-Knowledge-RAG golden dataset (eval/data/golden_dataset.json): each
answerable item labels evidence spans as {"source": file name, "evidence": text}. A retrieved chunk
is a hit when it comes from that file and contains the span (case, punctuation and whitespace
ignored), so the labels do not depend on how a pipeline chunks. The metric definitions copy that
project's harness so the numbers are comparable; they are provisional until a shared eval kit
re-measures every RAG variant.

Every stage of the pipeline is scored (dense, bm25, fused, reranked). Queries are the raw
questions: the LLM query rewrite (stage 1) is skipped, so the eval makes no LLM call.

Usage: python scripts/retrieval_eval.py <golden_dataset.json> [--out eval/retrieval_results.json]
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def chunk_source(chunk: dict) -> str:
    meta = chunk["metadata"]
    return re.split(r"[\\/]", meta.get("filename") or meta.get("source", ""))[-1]


def score(item: dict, ranked: list[dict], ks: tuple[int, ...]) -> dict[str, float]:
    cols = [(chunk_source(c), norm(c["text"])) for c in ranked]
    matrix = [
        [src == rel["source"] and norm(rel["evidence"]) in text for src, text in cols]
        for rel in item["relevant"]
    ]
    any_of = item.get("any_of", False)
    out: dict[str, float] = {}
    for k in ks:
        covered = [any(row[:k]) for row in matrix]
        out[f"recall@{k}"] = float(any(covered)) if any_of else sum(covered) / len(covered)
        out[f"success@{k}"] = float(any(covered) if any_of else all(covered))
    first = [row.index(True) + 1 for row in matrix if True in row]
    out["mrr"] = 1 / min(first) if first else 0.0
    return out


# deeper k per stage, as in Personal-Knowledge-RAG's harness; recall@1/3/5 everywhere
STAGE_KS = {"dense": (20,), "bm25": (20,), "fused": (10,), "reranked": ()}


def stage_ks(stage: str, depth: int) -> tuple[int, ...]:
    return tuple(sorted({1, 3, 5} | {k for k in STAGE_KS.get(stage, ()) if k <= depth}))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "eval" / "retrieval_results.json")
    args = p.parse_args()

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"  # models come from the local cache, as in the query scripts
    from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, TOP_K
    from src.pipeline.advanced_rag import retrieve_stages

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    texts = sorted(c["text"] for c in json.loads(CHUNKS_PATH.read_text(encoding="utf-8")))
    ids = {
        # same id as Personal-Knowledge-RAG's store.content_id (parsed, so CRLF-independent)
        "dataset": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()[:10],
        "corpus": f"{len(texts)}:{hashlib.sha256(chr(0).join(texts).encode()).hexdigest()[:10]}",
    }
    settings = {"chunk": f"{CHUNK_SIZE}/{CHUNK_OVERLAP}", "top_k": TOP_K, "rewrite": False}
    rows = []
    for item in dataset:
        if item["expected_outcome"] != "answer":
            continue
        stages = retrieve_stages(item["question"], top_k=TOP_K)
        rows.append(
            {
                "id": item["id"],
                "split": item["split"],
                "query": item["question"],
                **ids,
                "settings": settings,
                "metrics": {s: score(item, r, stage_ks(s, len(r))) for s, r in stages.items()},
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{args.out.name}: {ids} {settings}  n={len(rows)}")
    for stage, metrics in rows[0]["metrics"].items():
        cells = "  ".join(f"{m}={sum(r['metrics'][stage][m] for r in rows) / len(rows):.2f}" for m in metrics)
        print(f"  {stage:<9} {cells}")


if __name__ == "__main__":
    main()
