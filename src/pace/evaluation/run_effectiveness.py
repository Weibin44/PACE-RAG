"""Run effectiveness evaluation on one dataset."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from pace.config import BATCH_SIZES
from pace.data.batched import load_batched_examples
from pace.data.musique import load_musique_examples
from pace.data.schema import EvidenceExample
from pace.evaluation.calibration import (
    CalibrationManifest,
    calibrate_stage,
    load_calibration_manifest,
    save_calibration_manifest,
    select_calibration_ids,
)
from pace.evaluation.evaluator import (
    EvaluationResult,
    evaluate_adaptive_k,
    evaluate_curves,
    write_adaptive_point_csv,
    write_curves_csv,
)

ExampleFactory = Callable[[], Iterator[EvidenceExample]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
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
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--calibration-queries",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--parameter-step",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--adaptive-buffer",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--adaptive-search-fraction",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adaptive-min-documents",
        type=int,
        default=5,
    )
    return parser


def build_example_factory(
    args: argparse.Namespace,
) -> ExampleFactory:
    """Construct the dataset-specific example loader."""

    if args.dataset in {"hotpot", "2wiki"}:
        return lambda: load_batched_examples(
            args.source,
            args.cache_dir,
            args.similarity_cache,
            candidate_limit=100,
        )

    return lambda: load_musique_examples(
        args.source,
        args.cache_dir,
        args.similarity_cache,
        candidate_limit=20,
    )


def load_or_calibrate(
    args: argparse.Namespace,
    factory: ExampleFactory,
    d_maximum: int,
) -> tuple[CalibrationManifest, bool]:
    """Reuse an existing manifest or perform calibration."""

    if args.calibration_manifest.exists():
        manifest = load_calibration_manifest(
            args.calibration_manifest
        )
        if manifest.dataset != args.dataset:
            raise ValueError(
                "calibration manifest dataset does not match "
                f"{args.dataset}"
            )
        return manifest, True

    calibration_ids = select_calibration_ids(
        (query.query_id for query in factory()),
        args.calibration_queries,
    )

    print("Calibrating D-stage baselines...", flush=True)
    d_parameters = calibrate_stage(
        factory(),
        calibration_ids,
        "D",
        d_maximum,
        parameter_step=args.parameter_step,
    )

    print("Calibrating K-stage baselines...", flush=True)
    k_parameters = calibrate_stage(
        factory(),
        calibration_ids,
        "K",
        args.max_k,
        parameter_step=args.parameter_step,
    )

    manifest = CalibrationManifest(
        dataset=args.dataset,
        calibration_query_ids=calibration_ids,
        parameters_by_stage={
            "D": d_parameters,
            "K": k_parameters,
        },
    )
    save_calibration_manifest(
        args.calibration_manifest,
        manifest,
    )
    return manifest, False


def main() -> None:
    args = build_parser().parse_args()

    if args.max_k <= 0:
        raise ValueError("max-k must be positive")

    d_maximum = 20 if args.dataset == "musique" else 100
    factory = build_example_factory(args)

    manifest, calibration_reused = load_or_calibrate(
        args,
        factory,
        d_maximum,
    )

    print("Evaluating D-stage curves...", flush=True)
    d_result = evaluate_curves(
        factory(),
        "D",
        manifest.parameters_by_stage["D"],
        cutoffs=range(1, d_maximum + 1),
        excluded_query_ids=manifest.calibration_query_ids,
    )

    print("Evaluating K-stage curves...", flush=True)
    k_result = evaluate_curves(
        factory(),
        "K",
        manifest.parameters_by_stage["K"],
        cutoffs=range(1, args.max_k + 1),
        excluded_query_ids=manifest.calibration_query_ids,
    )

    if d_result.query_count != k_result.query_count:
        raise ValueError(
            "D-stage and K-stage query counts do not match"
        )

    combined_result = EvaluationResult(
        rows=d_result.rows + k_result.rows,
        query_count=d_result.query_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_curves_csv(
        args.output_dir / "curves.csv",
        combined_result,
    )

    adaptive_result = evaluate_adaptive_k(
        factory(),
        excluded_query_ids=manifest.calibration_query_ids,
        buffer=args.adaptive_buffer,
        search_fraction=args.adaptive_search_fraction,
        min_documents=args.adaptive_min_documents,
    )
    write_adaptive_point_csv(
        args.output_dir / "adaptive_point.csv",
        adaptive_result,
    )

    run_manifest = {
        "complete": True,
        "dataset": args.dataset,
        "num_evaluation_queries": combined_result.query_count,
        "num_calibration_queries": len(
            manifest.calibration_query_ids
        ),
        "d_maximum": d_maximum,
        "k_maximum": args.max_k,
        "parameter_step": args.parameter_step,
        "calibration_reused": calibration_reused,
        "calibration_manifest": str(
            args.calibration_manifest
        ), 
        "adaptive_k": {
            "buffer": args.adaptive_buffer,
            "search_fraction": args.adaptive_search_fraction,
            "min_documents": args.adaptive_min_documents,
        },
        "batch_sizes": BATCH_SIZES.manifest_dict(), 
    }
    (
        args.output_dir / "manifest.json"
    ).write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Completed {args.dataset}: "
        f"{combined_result.query_count} evaluation queries",
        flush=True,
    )


if __name__ == "__main__":
    main()