"""Cache writers for batched HotpotQA and 2Wiki cohorts."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch


TextEncoder = Callable[[Sequence[str]], torch.Tensor]
PairScorer = Callable[
    [Sequence[tuple[str, str]]],
    np.ndarray,
]


def iter_batches(
    source: Path,
    batch_name: str | None = None,
) -> Iterator[tuple[str, list[dict]]]:
    """Yield named batches from a materialized cohort."""

    paths = sorted(
        (source / "batches").glob(
            "*/samples_sentence_labels.json"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"no cohort batches found under {source}"
        )

    found_requested_batch = False

    for path in paths:
        name = path.parent.name
        if batch_name is not None and name != batch_name:
            continue

        found_requested_batch = True
        values = json.loads(
            path.read_text(encoding="utf-8")
        )
        yield name, values["samples"]

    if batch_name is not None and not found_requested_batch:
        raise ValueError(
            f"batch {batch_name!r} was not found"
        )


def atomic_save_npz(
    path: Path,
    **arrays: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def atomic_save_npy(
    path: Path,
    values: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp.npy"
    )
    np.save(temporary, values)
    temporary.replace(path)


def _coverage_cache_is_valid(
    path: Path,
    document_count: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(path) as values:
            query_weights = values["query_weights"]
            document_features = values[
                "document_features"
            ]
            return (
                document_features.shape[0] == document_count
                and document_features.shape[1]
                == len(query_weights)
            )
    except (OSError, KeyError, ValueError):
        return False


def write_batched_coverage_features(
    source: Path,
    cache_dir: Path,
    encode_texts: TextEncoder,
    *,
    batch_name: str | None = None,
) -> int:
    """Write query-active SPLADE document features."""

    query_count = 0

    for name, samples in iter_batches(
        source,
        batch_name,
    ):
        output_dir = (
            cache_dir / "coverage_features" / name
        )

        for index, sample in enumerate(samples):
            candidates = sample["candidates"]
            query_count += 1

            path = output_dir / f"{index:03d}.npz"
            if _coverage_cache_is_valid(
                path,
                len(candidates),
            ):
                continue

            query = encode_texts(
                [sample["question"]]
            )[0]
            documents = encode_texts(
                [
                    candidate["text"]
                    for candidate in candidates
                ]
            )

            active = query > 0
            query_weights = query[active]
            document_features = documents[:, active]

            atomic_save_npz(
                path,
                query_weights=(
                    query_weights.numpy().astype(np.float32)
                ),
                document_features=(
                    document_features.numpy().astype(np.float16)
                ),
            )

    return query_count


def write_batched_reranker_scores(
    source: Path,
    cache_dir: Path,
    score_pairs: PairScorer,
    *,
    batch_name: str | None = None,
) -> int:
    """Write one reranker matrix per source batch."""

    query_count = 0

    for name, samples in iter_batches(
        source,
        batch_name,
    ):
        if not samples:
            continue

        document_count = len(samples[0]["candidates"])
        if any(
            len(sample["candidates"]) != document_count
            for sample in samples
        ):
            raise ValueError(
                f"{name}: candidate counts are inconsistent"
            )

        query_count += len(samples)
        output = (
            cache_dir / "reranker_scores" / f"{name}.npy"
        )
        expected_shape = (
            len(samples),
            document_count,
        )

        if output.is_file():
            try:
                if (
                    np.load(output, mmap_mode="r").shape
                    == expected_shape
                ):
                    continue
            except (OSError, ValueError):
                pass

        pairs = [
            (
                sample["question"],
                candidate["text"],
            )
            for sample in samples
            for candidate in sample["candidates"]
        ]
        scores = np.asarray(
            score_pairs(pairs),
            dtype=np.float32,
        )

        if scores.shape != (len(pairs),):
            raise ValueError(
                f"{name}: scorer returned {scores.shape}, "
                f"expected {(len(pairs),)}"
            )

        atomic_save_npy(
            output,
            scores.reshape(expected_shape),
        )

    return query_count