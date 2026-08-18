"""Full-vocabulary SPLADE cosine-similarity cache writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pace.preprocessing.batched_cache import (
    TextEncoder,
    atomic_save_npz,
    iter_batches,
)
from pace.preprocessing.musique_cache import (
    load_musique_source,
    paragraph_text,
)


def _is_top_complete(sample: dict) -> bool:
    gold_fact_ids = {
        fact["fact_id"]
        for fact in sample["facts"]
    }
    available_fact_ids = {
        fact_id
        for candidate in sample["candidates"]
        for fact_id in candidate.get(
            "covered_facts",
            [],
        )
    }
    return gold_fact_ids <= available_fact_ids


def _similarity_cache_is_valid(
    path: Path,
    document_count: int,
) -> bool:
    if not path.is_file():
        return False

    try:
        with np.load(path) as values:
            return (
                values["query_similarity"].shape
                == (document_count,)
                and values["document_similarity"].shape
                == (document_count, document_count)
            )
    except (OSError, KeyError, ValueError):
        return False


def _similarities(
    vectors: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute query-document and document-document cosine similarity."""

    if vectors.ndim != 2 or len(vectors) < 2:
        raise ValueError(
            "vectors must contain one query and at least one document"
        )

    normalized = vectors / vectors.norm(
        dim=1,
        keepdim=True,
    ).clamp_min(1e-8)

    query = normalized[0]
    documents = normalized[1:]

    query_similarity = documents @ query
    document_similarity = documents @ documents.T

    return (
        query_similarity.numpy().astype(np.float32),
        document_similarity.numpy().astype(np.float16),
    )


def write_batched_splade_similarity(
    source: Path,
    cache_dir: Path,
    encode_texts: TextEncoder,
    *,
    batch_name: str | None = None,
) -> int:
    """Write similarity cache for Top-100-complete batched queries."""

    query_count = 0

    for name, samples in iter_batches(
        source,
        batch_name,
    ):
        output_dir = (
            cache_dir / "splade_similarity" / name
        )

        for index, sample in enumerate(samples):
            if not _is_top_complete(sample):
                continue

            candidates = sample["candidates"]
            query_count += 1
            path = output_dir / f"{index:03d}.npz"

            if _similarity_cache_is_valid(
                path,
                len(candidates),
            ):
                continue

            vectors = encode_texts(
                [
                    sample["question"],
                    *(
                        candidate["text"]
                        for candidate in candidates
                    ),
                ]
            )
            query_similarity, document_similarity = (
                _similarities(vectors)
            )

            atomic_save_npz(
                path,
                query_similarity=query_similarity,
                document_similarity=document_similarity,
            )

    return query_count


def write_musique_splade_similarity(
    source: Path,
    cache_dir: Path,
    encode_texts: TextEncoder,
) -> int:
    """Write similarity cache for all MuSiQue queries."""

    examples = load_musique_source(source)
    output_dir = cache_dir / "splade_similarity"

    for index, example in enumerate(examples):
        paragraphs = example["paragraphs"]
        path = output_dir / f"{index:04d}.npz"

        if _similarity_cache_is_valid(
            path,
            len(paragraphs),
        ):
            continue

        vectors = encode_texts(
            [
                example["question"],
                *(
                    paragraph_text(paragraph)
                    for paragraph in paragraphs
                ),
            ]
        )
        query_similarity, document_similarity = (
            _similarities(vectors)
        )

        atomic_save_npz(
            path,
            query_similarity=query_similarity,
            document_similarity=document_similarity,
        )

    return len(examples)