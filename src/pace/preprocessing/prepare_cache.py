"""Generate reusable effectiveness caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pace.config import BATCH_SIZES
from pace.preprocessing.batched_cache import (
    write_batched_coverage_features,
    write_batched_reranker_scores,
)
from pace.preprocessing.modeling import (
    RERANKER_MODEL_NAME,
    SPLADE_MODEL_NAME,
    encode_splade,
    load_reranker_model,
    load_splade_model,
    score_reranker_pairs,
)
from pace.preprocessing.musique_cache import (
    write_musique_coverage_features,
    write_musique_reranker_scores,
)
from pace.preprocessing.similarity_cache import (
    write_batched_splade_similarity,
    write_musique_splade_similarity,
)


STAGES = (
    "coverage_features",
    "reranker_scores",
    "splade_similarity",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("hotpot", "2wiki", "musique"),
        required=True,
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
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
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
    )
    parser.add_argument(
        "--batch-name",
        help="Generate only one HotpotQA/2Wiki batch.",
    )
    return parser


def stage_max_length(
    dataset: str,
    stage: str,
) -> int:
    """Return the exact tokenizer length used originally."""

    if stage == "coverage_features" and dataset != "musique":
        return 128

    return 256


def cache_manifest_path(
    cache_dir: Path,
    dataset: str,
    stage: str,
    batch_name: str | None,
) -> Path:
    """Return one unambiguous manifest path per cache task."""

    scope = batch_name if batch_name is not None else "all"
    return (
        cache_dir
        / "manifests"
        / f"{dataset}_{stage}_{scope}.json"
    )


def write_manifest(
    path: Path,
    values: dict,
) -> None:
    """Atomically write a JSON manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp.json"
    )
    temporary.write_text(
        json.dumps(values, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_cache(args: argparse.Namespace) -> dict:
    """Generate one requested cache stage."""

    if args.dataset == "musique" and args.batch_name is not None:
        raise ValueError(
            "MuSiQue does not use named batches"
        )

    max_length = stage_max_length(
        args.dataset,
        args.stage,
    )
    model_name = (
        RERANKER_MODEL_NAME
        if args.stage == "reranker_scores"
        else SPLADE_MODEL_NAME
    )

    manifest_path = cache_manifest_path(
        args.cache_dir,
        args.dataset,
        args.stage,
        args.batch_name,
    )
    manifest = {
        "complete": False,
        "dataset": args.dataset,
        "stage": args.stage,
        "source": str(args.source),
        "cache_dir": str(args.cache_dir),
        "batch_name": args.batch_name,
        "model": model_name,
        "requested_device": args.device,
        "local_files_only": args.local_files_only,
        "max_length": max_length,
        "batch_sizes": {
            **BATCH_SIZES.manifest_dict(),
            "splade_encoder": BATCH_SIZES.splade_encoder,
        },
    }
    write_manifest(manifest_path, manifest)

    if args.stage == "reranker_scores":
        bundle = load_reranker_model(
            device=args.device,
            local_files_only=args.local_files_only,
        )

        def score_pairs(pairs):
            return score_reranker_pairs(
                bundle.model,
                bundle.tokenizer,
                pairs,
                str(bundle.device),
                max_length=max_length,
            )

        if args.dataset == "musique":
            query_count = write_musique_reranker_scores(
                args.source,
                args.cache_dir,
                score_pairs,
            )
        else:
            query_count = write_batched_reranker_scores(
                args.source,
                args.cache_dir,
                score_pairs,
                batch_name=args.batch_name,
            )
    else:
        bundle = load_splade_model(
            device=args.device,
            local_files_only=args.local_files_only,
        )

        def encode_texts(texts):
            return encode_splade(
                bundle.model,
                bundle.tokenizer,
                texts,
                str(bundle.device),
                max_length=max_length,
            )

        if args.stage == "coverage_features":
            if args.dataset == "musique":
                query_count = write_musique_coverage_features(
                    args.source,
                    args.cache_dir,
                    encode_texts,
                )
            else:
                query_count = write_batched_coverage_features(
                    args.source,
                    args.cache_dir,
                    encode_texts,
                    batch_name=args.batch_name,
                )
        elif args.dataset == "musique":
            query_count = write_musique_splade_similarity(
                args.source,
                args.cache_dir,
                encode_texts,
            )
        else:
            query_count = write_batched_splade_similarity(
                args.source,
                args.cache_dir,
                encode_texts,
                batch_name=args.batch_name,
            )

    manifest.update(
        complete=True,
        num_queries=query_count,
        resolved_device=str(bundle.device),
        dtype=str(bundle.dtype).removeprefix("torch."),
    )
    write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_cache(args)

    print(
        f"Completed {manifest['dataset']} "
        f"{manifest['stage']}: "
        f"{manifest['num_queries']} queries",
        flush=True,
    )


if __name__ == "__main__":
    main()