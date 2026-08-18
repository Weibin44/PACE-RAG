"""Materialize evidence-labelled samples from retrieval results."""

from __future__ import annotations

from collections.abc import Sequence

from pace.cohort.matching import (
    DatasetName,
    covered_fact_ids,
)
from pace.cohort.schema import (
    CohortCandidate,
    CohortSample,
    GoldQuery,
    RetrievedPassage,
)


def label_passage(
    passage: RetrievedPassage,
    query: GoldQuery,
    dataset: DatasetName,
) -> CohortCandidate:
    """Attach covered gold-fact IDs to one passage."""

    return CohortCandidate(
        document_id=passage.document_id,
        text=passage.text,
        retriever_score=passage.retriever_score,
        retrieval_rank=passage.retrieval_rank,
        covered_fact_ids=covered_fact_ids(
            passage.text,
            query.facts,
            dataset,
        ),
        corpus_index=passage.corpus_index,
    )


def materialize_sample(
    query: GoldQuery,
    passages: Sequence[RetrievedPassage],
    dataset: DatasetName,
) -> CohortSample:
    """Create one evidence-labelled cohort sample."""

    return CohortSample(
        query_id=query.query_id,
        question=query.question,
        facts=query.facts,
        candidates=tuple(
            label_passage(
                passage,
                query,
                dataset,
            )
            for passage in passages
        ),
    )


def retain_complete_samples(
    samples: Sequence[CohortSample],
) -> tuple[CohortSample, ...]:
    """Keep samples whose complete gold evidence was retrieved."""

    return tuple(
        sample
        for sample in samples
        if sample.complete_evidence_retrieved
    )
