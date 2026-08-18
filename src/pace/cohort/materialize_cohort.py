"""Materialize cohorts from BERGEN retrieval results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace.cohort.corpus import BergenCorpus
from pace.cohort.hotpot import (
    materialize_hotpot_cohort,
)
from pace.cohort.two_wiki import (
    materialize_2wiki_cohort,
)
from pace.config import BATCH_SIZES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=("hotpot", "2wiki"),
        required=True,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
    )
    parser.add_argument(
        "--corpus-coverage-report",
        type=Path,
    )
    parser.add_argument(
        "--retrieval-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser


def write_manifest(
    path: Path,
    manifest: dict,
) -> None:
    """Atomically write a cohort manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp"
    )
    temporary.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_cohort(
    args: argparse.Namespace,
) -> dict:
    """Run dataset-specific cohort materialization."""

    if args.dataset == "hotpot":
        if args.evaluation is None:
            raise ValueError(
                "HotpotQA requires --evaluation"
            )
        if args.corpus_coverage_report is not None:
            raise ValueError(
                "HotpotQA does not use "
                "--corpus-coverage-report"
            )
    else:
        if args.evaluation is not None:
            raise ValueError(
                "2Wiki does not use --evaluation"
            )
        if args.corpus_coverage_report is None:
            raise ValueError(
                "2Wiki requires "
                "--corpus-coverage-report"
            )

    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "complete": False,
        "dataset": args.dataset,
        "labels": str(args.labels),
        "evaluation": (
            str(args.evaluation)
            if args.evaluation is not None
            else None
        ),
        "corpus_coverage_report": (
            str(args.corpus_coverage_report)
            if args.corpus_coverage_report is not None
            else None
        ),
        "retrieval_dir": str(args.retrieval_dir),
        "corpus_dir": str(args.corpus_dir),
        "corpus_format": (
            "bergen_huggingface_dataset"
        ),
        "candidate_count": 100,
        "batch_sizes": BATCH_SIZES.manifest_dict(),
    }
    write_manifest(
        manifest_path,
        manifest,
    )

    corpus = BergenCorpus.from_disk(
        args.corpus_dir
    )

    if args.dataset == "hotpot":
        query_count = materialize_hotpot_cohort(
            args.evaluation,
            args.labels,
            args.retrieval_dir,
            args.output_dir,
            corpus,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        manifest["batch_size"] = args.batch_size
        manifest["seed"] = args.seed
    else:
        query_count = materialize_2wiki_cohort(
            args.labels,
            args.retrieval_dir,
            args.corpus_coverage_report,
            args.output_dir,
            corpus,
            num_parts=args.num_parts,
        )
        manifest["num_parts"] = args.num_parts

    manifest.update(
        {
            "complete": True,
            "num_queries": query_count,
        }
    )
    write_manifest(
        manifest_path,
        manifest,
    )
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = materialize_cohort(args)

    print(
        f"Completed {manifest['dataset']} cohort: "
        f"{manifest['num_queries']} queries",
        flush=True,
    )


if __name__ == "__main__":
    main()
