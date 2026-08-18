"""Evaluate document selection at D followed by final selection at K."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from pace.config import BATCH_SIZES
from pace.data.batched import load_batched_examples
from pace.data.musique import load_musique_examples
from pace.data.schema import EvidenceExample
from pace.evaluation.calibration import (
    load_calibration_manifest,
)
from pace.evaluation.metrics import evidence_scores
from pace.evaluation.rankings import rank_example


METHODS = (
    ("soft_anchor", "ours"),
    ("rocchio_prf", "rocchio_prf"),
    ("mmr", "mmr"),
    ("dartboard", "dartboard"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("hotpot", "2wiki", "musique"),
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--similarity-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--d-values",
        default="20,30,40,50,60,70,80,90,100",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    return parser


def load_examples(
    args: argparse.Namespace,
) -> list[EvidenceExample]:
    if args.dataset == "musique":
        loader = load_musique_examples(
            args.source,
            args.cache_dir,
            args.similarity_cache,
            candidate_limit=20,
        )
    else:
        loader = load_batched_examples(
            args.source,
            args.cache_dir,
            args.similarity_cache,
            candidate_limit=100,
        )

    return list(loader)


def subset_example(
    example: EvidenceExample,
    indices: list[int],
) -> EvidenceExample:
    selected = np.asarray(indices, dtype=np.int64)

    subset = EvidenceExample(
        query_id=example.query_id,
        question=example.question,
        gold_fact_ids=example.gold_fact_ids,
        candidates=tuple(
            example.candidates[index]
            for index in indices
        ),
        reranker_scores=example.reranker_scores[
            selected
        ],
        query_weights=example.query_weights,
        document_features=example.document_features[
            selected
        ],
        document_similarity=example.document_similarity[
            np.ix_(selected, selected)
        ],
        query_similarity=example.query_similarity[
            selected
        ],
    )
    subset.validate()
    return subset


def final_k_order(
    example: EvidenceExample,
    selected_d: list[int],
    method: str,
    top_k: int,
    parameters,
) -> list[int]:
    if method != "ours":
        reranker_scores = example.reranker_scores[
            selected_d
        ]
        local_order = np.argsort(
            -reranker_scores,
            kind="stable",
        )[:top_k]
    else:
        subset = subset_example(
            example,
            selected_d,
        )
        local_order = rank_example(
            subset,
            "K",
            "ours",
            parameters,
            limit=top_k,
        )

    return [
        selected_d[int(index)]
        for index in local_order
    ]


def evaluate(
    args: argparse.Namespace,
) -> tuple[list[dict], int]:
    d_values = sorted(
        {
            int(value)
            for value in args.d_values.split(",")
        }
    )
    if not d_values:
        raise ValueError("at least one D value is required")
    if d_values[0] < args.top_k:
        raise ValueError("D must be greater than or equal to K")

    examples = load_examples(args)
    calibration = load_calibration_manifest(
        args.calibration_manifest
    )

    if calibration.dataset != args.dataset:
        raise ValueError(
            "calibration dataset does not match evaluation dataset"
        )

    examples = [
        example
        for example in examples
        if example.query_id
        not in calibration.calibration_query_ids
    ]
    if not examples:
        raise ValueError("no evaluation queries remain")

    maximum_d = max(
        example.candidate_count
        for example in examples
    )
    if d_values[-1] > maximum_d:
        raise ValueError(
            f"D cannot exceed candidate maximum {maximum_d}"
        )

    fixed_method = f"fixed_d{maximum_d}"
    output_methods = (
        (fixed_method, "standard"),
        *METHODS,
    )

    totals = {}
    for output_method, _ in output_methods:
        values = (
            (maximum_d,)
            if output_method == fixed_method
            else tuple(d_values)
        )
        for d_value in values:
            totals[(output_method, d_value)] = {
                "complete_recall_at_D": 0.0,
                "supporting_recall_at_D": 0.0,
                "complete_recall_at_K": 0.0,
                "supporting_recall_at_K": 0.0,
            }

    for example in examples:
        for output_method, ranking_method in output_methods:
            d_order = rank_example(
                example,
                "D",
                ranking_method,
                calibration.parameters_by_stage["D"],
                limit=maximum_d,
            )
            values = (
                (maximum_d,)
                if output_method == fixed_method
                else tuple(d_values)
            )

            for d_value in values:
                selected_d = d_order[:d_value]
                selected_k = final_k_order(
                    example,
                    selected_d,
                    ranking_method,
                    args.top_k,
                    calibration.parameters_by_stage["K"],
                )

                d_scores = evidence_scores(
                    example,
                    selected_d,
                )
                k_scores = evidence_scores(
                    example,
                    selected_k,
                )
                result = totals[
                    (output_method, d_value)
                ]
                result["complete_recall_at_D"] += (
                    d_scores["complete_evidence_recall"]
                )
                result["supporting_recall_at_D"] += (
                    d_scores["supporting_fact_recall"]
                )
                result["complete_recall_at_K"] += (
                    k_scores["complete_evidence_recall"]
                )
                result["supporting_recall_at_K"] += (
                    k_scores["supporting_fact_recall"]
                )

    rows = []
    query_count = len(examples)

    for output_method, _ in output_methods:
        values = (
            (maximum_d,)
            if output_method == fixed_method
            else tuple(d_values)
        )
        for d_value in values:
            result = totals[(output_method, d_value)]
            rows.append(
                {
                    "dataset": args.dataset,
                    "method": output_method,
                    "D": d_value,
                    "K": args.top_k,
                    **{
                        metric: value / query_count
                        for metric, value in result.items()
                    },
                }
            )

    return rows, query_count


def write_results(
    args: argparse.Namespace,
    rows: list[dict],
    query_count: int,
) -> None:
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with (
        args.output_dir / "joint_recall.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "complete": True,
        "dataset": args.dataset,
        "num_evaluation_queries": query_count,
        "d_values": sorted(
            {
                int(value)
                for value in args.d_values.split(",")
            }
        ),
        "top_k": args.top_k,
        "methods": list(
            dict.fromkeys(
                row["method"]
                for row in rows
            )
        ),
        "batch_sizes": BATCH_SIZES.manifest_dict(),
    }
    (
        args.output_dir / "manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    rows, query_count = evaluate(args)
    write_results(
        args,
        rows,
        query_count,
    )


if __name__ == "__main__":
    main()
