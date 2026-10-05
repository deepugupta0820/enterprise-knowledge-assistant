"""Evaluate the enterprise RAG retriever using Precision@K, Recall@K and MRR.

The relevance unit is (source filename, 1-based PDF page). This is intentionally
more stable than evaluating raw LangChain chunk IDs.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow this script to be run as: python evaluation/evaluate_retrieval.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DB_FAISS_PATH, EMBEDDING_MODEL  # noqa: E402
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402


def normalize_source(source: str) -> str:
    """Normalize an absolute/relative source path to its PDF filename."""
    return os.path.basename(source).strip().lower()


def page_key(doc) -> tuple[str, int]:
    """Return the stable relevance key for a LangChain Document."""
    source = normalize_source(doc.metadata.get("source", ""))
    # LangChain's PDF loaders use zero-based page metadata.
    page = int(doc.metadata.get("page", 0)) + 1
    return source, page


def ground_truth_keys(item: dict) -> set[tuple[str, int]]:
    return {
        (normalize_source(x["source"]), int(x["page"]))
        for x in item["relevant_pages"]
    }


def unique_ranked_pages(retrieved: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Collapse multiple chunks from the same page while preserving first rank."""
    seen = set()
    pages = []
    for item in retrieved:
        if item not in seen:
            seen.add(item)
            pages.append(item)
    return pages


def precision_at_k(retrieved: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int) -> float:
    """Fraction of the first k retrieved results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(x in relevant for x in top_k) / len(top_k)


def recall_at_k(retrieved: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int) -> float:
    """Fraction of all relevant pages retrieved in the first k results."""
    if not relevant or k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[tuple[str, int]], relevant: set[tuple[str, int]], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant result; 0 if none is found."""
    ranked = retrieved if k is None else retrieved[:k]
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(db, dataset: list[dict], ks: list[int]) -> tuple[list[dict], dict]:
    retriever = db.as_retriever(search_kwargs={"k": max(ks)})
    rows = []

    for item in dataset:
        docs = retriever.invoke(item["question"])
        retrieved_chunks = [page_key(doc) for doc in docs]
        retrieved = unique_ranked_pages(retrieved_chunks)
        relevant = ground_truth_keys(item)

        row = {
            "question": item["question"],
            "relevant_pages": sorted(relevant),
            "retrieved_pages": retrieved,
            "retrieved_chunk_pages": retrieved_chunks,
            "metrics": {},
        }
        for k in ks:
            row["metrics"][f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
            row["metrics"][f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
            row["metrics"][f"mrr@{k}"] = reciprocal_rank(retrieved, relevant, k)
        rows.append(row)

    summary = {}
    for k in ks:
        summary[f"precision@{k}"] = sum(r["metrics"][f"precision@{k}"] for r in rows) / len(rows)
        summary[f"recall@{k}"] = sum(r["metrics"][f"recall@{k}"] for r in rows) / len(rows)
        summary[f"mrr@{k}"] = sum(r["metrics"][f"mrr@{k}"] for r in rows) / len(rows)
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate FAISS retrieval quality.")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "evaluation" / "eval_dataset.json"))
    parser.add_argument("--k", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "evaluation" / "evaluation_results.json"))
    args = parser.parse_args()

    ks = sorted(set(k for k in args.k if k > 0))
    if not ks:
        raise ValueError("At least one positive K is required.")

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not os.path.exists(DB_FAISS_PATH):
        raise RuntimeError(
            "FAISS index not found. First run the Streamlit app, upload the evaluation PDFs, "
            "and click 'Process & Index Knowledge Base'."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    rows, summary = evaluate(db, dataset, ks)
    result = {
        "description": "Page-level retrieval evaluation for the Enterprise Knowledge Assistant",
        "top_k_used_for_retrieval": max(ks),
        "num_queries": len(dataset),
        "metrics": summary,
        "per_query": rows,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nRetrieval Evaluation")
    print("=" * 24)
    print(f"Queries: {len(dataset)}")
    for k in ks:
        print(f"Precision@{k}: {summary[f'precision@{k}']:.4f}")
        print(f"Recall@{k}:    {summary[f'recall@{k}']:.4f}")
        print(f"MRR@{k}:       {summary[f'mrr@{k}']:.4f}")
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
