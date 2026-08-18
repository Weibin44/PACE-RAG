"""Materialize 2WikiMultiHopQA cohort parts."""

from __future__ import annotations

import json
from pathlib import Path

from pace.cohort.builder import build_cohort_batch
from pace.cohort.gold import load_2wiki_gold_queries
from pace.cohort.io import write_cohort_batch
from pace.cohort.materialize import retain_complete_samples
from pace.cohort.retrieval import (
    CorpusReader,
    load_retrieval_results,
)


def _load_complete_query_ids(
    report_path: Path,
) -> tuple[str, ...]:
    """Load corpus-complete query IDs from an audit report."""

    report = json.loads(
        report_path.read_text(encoding="utf-8")
    )
    query_ids = report.get("complete_query_ids")

    if not isinstance(query_ids, list):
        raise ValueError(
            f"{report_path} must contain complete_query_ids"
        )

    normalized = tuple(str(query_id) for query_id in query_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"{report_path} contains duplicate query IDs"
        )

    return normalized


def _write_coverage(
    path: Path,
    samples,
) -> None:
    """Atomically write Top-100 evidence coverage."""

    rows = [
        {
            "query_id": sample.query_id,
            "gold_fact_count": len(sample.gold_fact_ids),
            "covered_fact_count": len(
                sample.gold_fact_ids & sample.retrieved_fact_ids
            ),
            "all_facts_covered": (
                sample.complete_evidence_retrieved
            ),
        }
        for sample in samples
    ]

    payload = {
        "corpus_complete_queries": len(rows),
        "top_k": 100,
        "top_k_complete_queries": sum(
            row["all_facts_covered"] for row in rows
        ),
        "top_k_complete_rate": (
            sum(row["all_facts_covered"] for row in rows)
            / len(rows)
            if rows
            else 0.0
        ),
        "per_query": rows,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_2wiki_cohort(
    labels_path: Path,
    retrieval_dir: Path,
    corpus_coverage_report: Path,
    output_dir: Path,
    corpus: CorpusReader,
    *,
    num_parts: int = 8,
) -> int:
    """Compute Top-100 coverage and materialize complete queries."""

    if num_parts <= 0:
        raise ValueError("number of parts must be positive")

    complete_query_ids = _load_complete_query_ids(
        corpus_coverage_report
    )
    total = 0

    for part_index in range(num_parts):
        part_name = (
            f"part{part_index:02d}-of-{num_parts:02d}"
        )
        query_ids = complete_query_ids[
            part_index::num_parts
        ]

        indices, scores = load_retrieval_results(
            retrieval_dir
            / f"dense_top100.{part_name}.npz"
        )

        if len(indices) != len(query_ids):
            raise ValueError(
                f"{part_name}: retrieval contains "
                f"{len(indices)} rows, expected "
                f"{len(query_ids)}"
            )

        queries = load_2wiki_gold_queries(
            labels_path,
            query_ids,
        )
        candidates = build_cohort_batch(
            queries,
            indices,
            scores,
            corpus,
            "2wiki",
        )

        _write_coverage(
            output_dir
            / "coverage"
            / f"top100_coverage.{part_name}.json",
            candidates,
        )

        samples = retain_complete_samples(candidates)
        write_cohort_batch(
            output_dir
            / "batches"
            / part_name
            / "samples_sentence_labels.json",
            samples,
        )
        total += len(samples)

    return total
