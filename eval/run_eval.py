from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retriever import KnowledgeRetriever  # noqa: E402


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_source(path_text: str) -> str:
    return path_text.replace("\\", "/")


def evaluate_case(
    retriever: KnowledgeRetriever,
    case: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    docs = retriever.similarity_search(case["question"], k=k)
    context = "\n".join(doc.page_content for doc in docs)
    context_lower = context.lower()

    expected_keywords = case.get("expected_keywords", [])
    expected_sources = case.get("expected_sources", [])

    keyword_hits = [kw for kw in expected_keywords if kw.lower() in context_lower]
    source_hits = []

    normalized_sources = {
        normalize_source(str(doc.metadata.get("source", ""))) for doc in docs
    }
    for source in expected_sources:
        if normalize_source(source) in normalized_sources:
            source_hits.append(source)

    return {
        "id": case["id"],
        "question": case["question"],
        "category": case.get("category", "unknown"),
        "retrieved_sources": sorted(normalized_sources),
        "keyword_hits": keyword_hits,
        "keyword_hit_count": len(keyword_hits),
        "keyword_total": len(expected_keywords),
        "source_hits": source_hits,
        "source_hit_count": len(source_hits),
        "source_total": len(expected_sources),
        "retrieval_hit": bool(keyword_hits),
        "source_hit": bool(source_hits),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    retrieval_hits = sum(1 for item in results if item["retrieval_hit"])
    source_hits = sum(1 for item in results if item["source_hit"])
    total_keyword_hits = sum(item["keyword_hit_count"] for item in results)
    total_keywords = sum(item["keyword_total"] for item in results)

    categories: dict[str, dict[str, int]] = {}
    for item in results:
        category = item["category"]
        bucket = categories.setdefault(
            category,
            {"cases": 0, "retrieval_hits": 0, "source_hits": 0},
        )
        bucket["cases"] += 1
        bucket["retrieval_hits"] += int(item["retrieval_hit"])
        bucket["source_hits"] += int(item["source_hit"])

    return {
        "cases": total,
        "retrieval_hit_rate": retrieval_hits / total if total else 0.0,
        "source_hit_rate": source_hits / total if total else 0.0,
        "keyword_coverage_rate": (
            total_keyword_hits / total_keywords if total_keywords else 0.0
        ),
        "category_breakdown": categories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline retrieval evaluation for MedAgent."
    )
    parser.add_argument(
        "--dataset",
        default=str(Path("eval") / "qa_dataset.jsonl"),
        help="Path to the evaluation dataset.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Top-k documents to retrieve for each question.",
    )
    parser.add_argument(
        "--min-retrieval-hit-rate",
        type=float,
        default=0.0,
        help="Fail if retrieval hit rate is below this threshold.",
    )
    parser.add_argument(
        "--min-source-hit-rate",
        type=float,
        default=0.0,
        help="Fail if source hit rate is below this threshold.",
    )
    parser.add_argument(
        "--min-keyword-coverage-rate",
        type=float,
        default=0.0,
        help="Fail if keyword coverage rate is below this threshold.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write JSON evaluation results.",
    )
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    dataset = load_dataset(dataset_path)

    retriever = KnowledgeRetriever()
    mode = "vector" if retriever.vectorstore is not None else "keyword_fallback"

    results = [evaluate_case(retriever, case, args.k) for case in dataset]
    summary = summarize(results)

    print("=== MedAgent Retrieval Evaluation ===")
    print(f"Dataset: {dataset_path}")
    print(f"Cases: {summary['cases']}")
    print(f"Retriever mode: {mode}")
    if retriever.init_error:
        print(f"Retriever note: {retriever.init_error}")
    print(f"Retrieval hit rate: {summary['retrieval_hit_rate']:.2%}")
    print(f"Source hit rate: {summary['source_hit_rate']:.2%}")
    print(f"Keyword coverage rate: {summary['keyword_coverage_rate']:.2%}")
    print("")
    print("Category breakdown:")
    for category, bucket in sorted(summary["category_breakdown"].items()):
        cases = bucket["cases"]
        retrieval_rate = bucket["retrieval_hits"] / cases if cases else 0.0
        source_rate = bucket["source_hits"] / cases if cases else 0.0
        print(
            f"- {category}: cases={cases}, retrieval_hit_rate={retrieval_rate:.2%}, "
            f"source_hit_rate={source_rate:.2%}"
        )

    print("")
    print("Case details:")
    for item in results:
        print(
            f"- {item['id']}: retrieval_hit={item['retrieval_hit']}, "
            f"source_hit={item['source_hit']}, "
            f"keywords={item['keyword_hit_count']}/{item['keyword_total']}, "
            f"sources={item['source_hit_count']}/{item['source_total']}"
        )

    if args.output:
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    failures = []
    if summary["retrieval_hit_rate"] < args.min_retrieval_hit_rate:
        failures.append(
            f"retrieval_hit_rate {summary['retrieval_hit_rate']:.2%} < {args.min_retrieval_hit_rate:.2%}"
        )
    if summary["source_hit_rate"] < args.min_source_hit_rate:
        failures.append(
            f"source_hit_rate {summary['source_hit_rate']:.2%} < {args.min_source_hit_rate:.2%}"
        )
    if summary["keyword_coverage_rate"] < args.min_keyword_coverage_rate:
        failures.append(
            f"keyword_coverage_rate {summary['keyword_coverage_rate']:.2%} < {args.min_keyword_coverage_rate:.2%}"
        )

    if failures:
        print("")
        print("Evaluation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
