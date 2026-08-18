"""Generate Top-100 retrieval results from a BERGEN SPLADE index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace.cohort.gold import (
    load_2wiki_questions,
    load_hotpot_gold_queries,
)
from pace.cohort.splade_retrieval import (
    retrieve_questions,
)


def add_common_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    datasets = parser.add_subparsers(
        dest="dataset",
        required=True,
    )

    hotpot = datasets.add_parser("hotpot")
    add_common_arguments(hotpot)
    hotpot.add_argument(
        "--evaluation",
        type=Path,
        required=True,
    )
    hotpot.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    hotpot.add_argument(
        "--batch-index",
        type=int,
        required=True,
    )
    hotpot.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    hotpot.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    two_wiki = datasets.add_parser("2wiki")
    add_common_arguments(two_wiki)
    two_wiki.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    two_wiki.add_argument(
        "--coverage-report",
        type=Path,
        required=True,
    )
    two_wiki.add_argument(
        "--part-index",
        type=int,
        required=True,
    )
    two_wiki.add_argument(
        "--num-parts",
        type=int,
        default=8,
    )

    return parser


def select_hotpot_batch(
    args: argparse.Namespace,
):
    """Select one deterministic HotpotQA batch."""

    if args.batch_size <= 0:
        raise ValueError(
            "batch size must be positive"
        )
    if args.batch_index < 0:
        raise ValueError(
            "batch index must be non-negative"
        )

    queries = load_hotpot_gold_queries(
        args.evaluation,
        args.labels,
        seed=args.seed,
    )
    start = args.batch_index * args.batch_size
    stop = min(
        start + args.batch_size,
        len(queries),
    )

    if start >= len(queries):
        raise ValueError(
            "batch index exceeds available queries"
        )

    return queries[start:stop], start, stop


def select_2wiki_part(
    args: argparse.Namespace,
):
    """Select one round-robin 2Wiki retrieval part."""

    if args.num_parts <= 0:
        raise ValueError(
            "number of parts must be positive"
        )
    if not 0 <= args.part_index < args.num_parts:
        raise ValueError(
            "part index must satisfy "
            "0 <= part_index < num_parts"
        )

    report = json.loads(
        args.coverage_report.read_text(
            encoding="utf-8"
        )
    )
    complete_query_ids = report.get(
        "complete_query_ids"
    )
    if not isinstance(complete_query_ids, list):
        raise ValueError(
            "coverage report must contain "
            "complete_query_ids"
        )

    questions = {
        query.query_id: query
        for query in load_2wiki_questions(
            args.labels
        )
    }
    selected_ids = [
        str(query_id)
        for query_id in complete_query_ids[
            args.part_index :: args.num_parts
        ]
    ]

    missing = [
        query_id
        for query_id in selected_ids
        if query_id not in questions
    ]
    if missing:
        raise KeyError(
            f"unknown 2Wiki query IDs: {missing[:5]}"
        )

    return tuple(
        questions[query_id]
        for query_id in selected_ids
    )


def run_retrieval(
    args: argparse.Namespace,
) -> tuple[Path, int]:
    """Run one Hotpot batch or one 2Wiki part."""

    if args.candidate_count <= 0:
        raise ValueError(
            "candidate count must be positive"
        )

    if args.dataset == "hotpot":
        queries, start, stop = (
            select_hotpot_batch(args)
        )
        output_path = (
            args.output_dir
            / f"{start:05d}_{stop:05d}"
            / "dense_top100.npz"
        )
    else:
        queries = select_2wiki_part(args)
        part_name = (
            f"part{args.part_index:02d}"
            f"-of-{args.num_parts:02d}"
        )
        output_path = (
            args.output_dir
            / f"dense_top100.{part_name}.npz"
        )

    retrieve_questions(
        [query.question for query in queries],
        args.index_dir,
        output_path,
        device=args.device,
        candidate_count=args.candidate_count,
        local_files_only=args.local_files_only,
    )

    return output_path, len(queries)


def main() -> None:
    args = build_parser().parse_args()
    output_path, query_count = run_retrieval(args)

    print(
        f"Retrieved {query_count} queries into "
        f"{output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

