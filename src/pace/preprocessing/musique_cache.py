"""Cache writers for the MuSiQue closed-context dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pace.preprocessing.batched_cache import (
    PairScorer,
    TextEncoder,
    atomic_save_npy,
    atomic_save_npz,
)


MAX_PARAGRAPHS = 20


def load_musique_source(path: Path) -> list[dict]:
    """Load and validate answerable MuSiQue examples."""

    if not path.is_file():
        raise FileNotFoundError(
            f"MuSiQue source not found: {path}"
        )

    examples = [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not examples:
        raise ValueError("MuSiQue source is empty")

    for index, example in enumerate(examples):
        paragraphs = example.get("paragraphs", [])
        if not paragraphs:
            raise ValueError(
                f"MuSiQue example {index} has no paragraphs"
            )
        if len(paragraphs) > MAX_PARAGRAPHS:
            raise ValueError(
                f"MuSiQue example {index} has more than "
                f"{MAX_PARAGRAPHS} paragraphs"
            )
        if any(
            "is_supporting" not in paragraph
            for paragraph in paragraphs
        ):
            raise ValueError(
                f"MuSiQue example {index} has no evidence labels"
            )

    return examples


def paragraph_text(paragraph: dict) -> str:
    """Construct the exact text encoded in the original experiment."""

    return (
        f'{paragraph["title"]}: '
        f'{paragraph["paragraph_text"]}'
    )


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
            dense_scores = values["dense_scores"]

            return (
                document_features.shape[0] == document_count
                and document_features.shape[1]
                == len(query_weights)
                and dense_scores.shape == (document_count,)
            )
    except (OSError, KeyError, ValueError):
        return False


def write_musique_coverage_features(
    source: Path,
    cache_dir: Path,
    encode_texts: TextEncoder,
) -> int:
    """Write query-active SPLADE features and dense scores."""

    examples = load_musique_source(source)
    output_dir = cache_dir / "coverage_features"

    for index, example in enumerate(examples):
        paragraphs = example["paragraphs"]
        path = output_dir / f"{index:04d}.npz"

        if _coverage_cache_is_valid(
            path,
            len(paragraphs),
        ):
            continue

        query = encode_texts(
            [example["question"]]
        )[0]
        documents = encode_texts(
            [
                paragraph_text(paragraph)
                for paragraph in paragraphs
            ]
        )

        active = query > 0
        query_weights = query[active]
        document_features = documents[:, active]
        dense_scores = (
            document_features * query_weights
        ).sum(dim=1)

        atomic_save_npz(
            path,
            query_weights=(
                query_weights.numpy().astype(np.float32)
            ),
            document_features=(
                document_features.numpy().astype(np.float16)
            ),
            dense_scores=(
                dense_scores.numpy().astype(np.float32)
            ),
        )

    return len(examples)


def write_musique_reranker_scores(
    source: Path,
    cache_dir: Path,
    score_pairs: PairScorer,
) -> int:
    """Write padded MuSiQue reranker scores."""

    examples = load_musique_source(source)
    output = cache_dir / "reranker_scores.npy"
    expected_shape = (len(examples), MAX_PARAGRAPHS)

    if output.is_file():
        try:
            if (
                np.load(output, mmap_mode="r").shape
                == expected_shape
            ):
                return len(examples)
        except (OSError, ValueError):
            pass

    pairs = [
        (
            example["question"],
            paragraph_text(paragraph),
        )
        for example in examples
        for paragraph in example["paragraphs"]
    ]
    scores = np.asarray(
        score_pairs(pairs),
        dtype=np.float32,
    )

    if scores.shape != (len(pairs),):
        raise ValueError(
            f"scorer returned {scores.shape}, "
            f"expected {(len(pairs),)}"
        )

    padded = np.full(
        expected_shape,
        -np.inf,
        dtype=np.float32,
    )

    offset = 0
    for index, example in enumerate(examples):
        count = len(example["paragraphs"])
        padded[index, :count] = scores[
            offset : offset + count
        ]
        offset += count

    atomic_save_npy(output, padded)
    return len(examples)