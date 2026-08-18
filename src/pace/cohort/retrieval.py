"""Read and materialize dense retrieval results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from pace.cohort.schema import (
    CorpusPassage,
    RetrievedPassage,
)


class CorpusReader(Protocol):
    """Minimal corpus interface required by retrieval."""

    def fetch(
        self,
        indices: Sequence[int],
    ) -> Mapping[int, CorpusPassage]:
        ...


def load_retrieval_results(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate a dense retrieval NPZ file."""

    with np.load(path) as values:
        if "indices" not in values or "scores" not in values:
            raise ValueError(
                "retrieval file must contain indices and scores"
            )
        indices = np.array(
            values["indices"],
            copy=True,
        )
        scores = np.array(
            values["scores"],
            copy=True,
        )

    if indices.ndim != 2 or scores.ndim != 2:
        raise ValueError(
            "retrieval arrays must be two-dimensional"
        )
    if indices.shape != scores.shape:
        raise ValueError(
            "retrieval indices and scores must have equal shapes"
        )
    if not np.issubdtype(
        indices.dtype,
        np.integer,
    ):
        raise ValueError(
            "retrieval indices must be integers"
        )
    if np.any(indices < 0):
        raise ValueError(
            "retrieval indices must be non-negative"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "retrieval scores must be finite"
        )

    return (
        indices.astype(np.int64, copy=False),
        scores.astype(np.float32, copy=False),
    )


def retrieved_passages_from_lookup(
    passages: Mapping[int, CorpusPassage],
    indices: Sequence[int],
    scores: Sequence[float],
) -> tuple[RetrievedPassage, ...]:
    """Convert one retrieval row using prefetched passages."""

    index_array = np.asarray(indices)
    score_array = np.asarray(scores)

    if index_array.ndim != 1 or score_array.ndim != 1:
        raise ValueError(
            "one retrieval row must be one-dimensional"
        )
    if len(index_array) != len(score_array):
        raise ValueError(
            "retrieval row indices and scores must align"
        )

    integer_indices = [
        int(index)
        for index in index_array
    ]
    if len(set(integer_indices)) != len(integer_indices):
        raise ValueError(
            "retrieval row contains duplicate corpus indices"
        )

    missing = [
        index
        for index in integer_indices
        if index not in passages
    ]
    if missing:
        raise KeyError(
            f"corpus did not return indices: {missing[:5]}"
        )

    return tuple(
        RetrievedPassage(
            document_id=passages[index].document_id,
            text=passages[index].text,
            retriever_score=float(score),
            retrieval_rank=rank,
            corpus_index=index,
        )
        for rank, (index, score) in enumerate(
            zip(integer_indices, score_array),
            start=1,
        )
    )


def retrieved_passages_from_row(
    corpus: CorpusReader,
    indices: Sequence[int],
    scores: Sequence[float],
) -> tuple[RetrievedPassage, ...]:
    """Fetch and convert one ranked retrieval row."""

    index_array = np.asarray(indices)
    if index_array.ndim != 1:
        raise ValueError(
            "one retrieval row must be one-dimensional"
        )

    integer_indices = [
        int(index)
        for index in index_array
    ]
    passages = corpus.fetch(integer_indices)

    return retrieved_passages_from_lookup(
        passages,
        integer_indices,
        scores,
    )

def save_retrieval_results(
    path: Path,
    indices: np.ndarray,
    scores: np.ndarray,
) -> None:
    """Atomically save validated retrieval results."""

    index_array = np.asarray(indices)
    score_array = np.asarray(scores)

    if index_array.ndim != 2 or score_array.ndim != 2:
        raise ValueError(
            "retrieval arrays must be two-dimensional"
        )
    if index_array.shape != score_array.shape:
        raise ValueError(
            "retrieval indices and scores must have equal shapes"
        )
    if not np.issubdtype(
        index_array.dtype,
        np.integer,
    ):
        raise ValueError(
            "retrieval indices must be integers"
        )
    if np.any(index_array < 0):
        raise ValueError(
            "retrieval indices must be non-negative"
        )
    if not np.all(np.isfinite(score_array)):
        raise ValueError(
            "retrieval scores must be finite"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp.npz"
    )
    np.savez(
        temporary,
        indices=index_array.astype(
            np.int64,
            copy=False,
        ),
        scores=score_array.astype(
            np.float32,
            copy=False,
        ),
    )
    temporary.replace(path)
