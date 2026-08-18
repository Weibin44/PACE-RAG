"""Generate the 2Wiki BERGEN corpus-coverage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace.cohort.corpus import BergenCorpus
from pace.cohort.corpus_coverage import (
    audit_2wiki_corpus_coverage,
)
from pace.config import BATCH_SIZES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
    )
    return parser


def write_report(
    path: Path,
    report: dict,
) -> None:
    """Atomically write the coverage report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp"
    )
    temporary.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_audit(
    args: argparse.Namespace,
) -> dict:
    """Audit all 2Wiki gold facts against the corpus."""

    if args.batch_size <= 0:
        raise ValueError(
            "batch size must be positive"
        )

    examples = json.loads(
        args.labels.read_text(encoding="utf-8")
    )
    if not isinstance(examples, list):
        raise ValueError(
            "2Wiki labels must contain a JSON list"
        )

    corpus = BergenCorpus.from_disk(
        args.corpus_dir
    )
    report = audit_2wiki_corpus_coverage(
        examples,
        corpus.iter_contents(
            batch_size=args.batch_size
        ),
    )
    report.update(
        {
            "complete": True,
            "dataset": "2wiki",
            "corpus_dir": str(args.corpus_dir),
            "scan_batch_size": args.batch_size,
            "batch_sizes": (
                BATCH_SIZES.manifest_dict()
            ),
        }
    )

    write_report(
        args.output,
        report,
    )
    return report


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(args)

    print(
        "2Wiki corpus-complete queries: "
        f"{report['queries_with_all_facts']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
