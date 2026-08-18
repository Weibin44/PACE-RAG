"""Build evidence-labelled cohort batches."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pace.cohort.matching import DatasetName
from pace.cohort.materialize import (
    materialize_sample,
    retain_complete_samples,
)
from pace.cohort.retrieval import (
    CorpusReader,
    retrieved_passages_from_lookup,
)
from pace.cohort.schema import (
    CohortSample,
    GoldQuery,
)


def build_cohort_batch(
    queries: Sequence[GoldQuery],
    indices: np.ndarray,
    scores: np.ndarray,
    corpus: CorpusReader,
    dataset: DatasetName,
    *,
    retain_complete: bool = False,
) -> tuple[CohortSample, ...]:
    """Build one cohort batch from aligned retrieval arrays."""

    index_array = np.asarray(indices)
    score_array = np.asarray(scores)

    if index_array.ndim != 2 or score_array.ndim != 2:
        raise ValueError(
            "batch retrieval arrays must be two-dimensional"
        )
    if index_array.shape != score_array.shape:
        raise ValueError(
            "batch indices and scores must have equal shapes"
        )
    if len(queries) != len(index_array):
        raise ValueError(
            "query count does not match retrieval rows"
        )

    passage_lookup = corpus.fetch(
        index_array.reshape(-1).tolist()
    )

    samples = tuple(
        materialize_sample(
            query,
            retrieved_passages_from_lookup(
                passage_lookup,
                index_array[position],
                score_array[position],
            ),
            dataset,
        )
        for position, query in enumerate(queries)
    )

    if retain_complete:
        return retain_complete_samples(samples)

    return samples