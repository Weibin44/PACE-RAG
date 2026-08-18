"""Read evidence-labelled cohort batches into canonical schemas."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pace.cohort.schema import (
    CohortCandidate,
    CohortSample,
    GoldFact,
)


def _document_id(candidate: Mapping[str, Any]) -> str:
    for key in ("doc_id", "corpus_index", "doc_index"):
        if key in candidate:
            return str(candidate[key])

    raise KeyError("candidate has no document identifier")


def _corpus_index(
    candidate: Mapping[str, Any],
) -> int | None:
    for key in ("corpus_index", "doc_index"):
        if key in candidate:
            return int(candidate[key])

    return None


def candidate_from_dict(
    candidate: Mapping[str, Any],
    position: int,
) -> CohortCandidate:
    """Convert one legacy candidate dictionary."""

    score = candidate.get(
        "dense_score",
        candidate.get("score"),
    )
    if score is None:
        raise KeyError("candidate has no retriever score")

    return CohortCandidate(
        document_id=_document_id(candidate),
        text=str(candidate["text"]),
        retriever_score=float(score),
        retrieval_rank=int(
            candidate.get("dense_rank", position + 1)
        ),
        covered_fact_ids=frozenset(
            str(value)
            for value in candidate.get(
                "covered_facts",
                [],
            )
        ),
        corpus_index=_corpus_index(candidate),
    )


def fact_from_dict(
    fact: Mapping[str, Any],
) -> GoldFact:
    """Convert one legacy gold-fact dictionary."""

    sentence_index = fact.get(
        "sentence_id",
        fact.get("sentence_index"),
    )
    if sentence_index is None:
        raise KeyError("fact has no sentence index")

    return GoldFact(
        fact_id=str(fact["fact_id"]),
        title=str(fact["title"]),
        sentence_index=int(sentence_index),
        text=str(fact["text"]),
    )


def sample_from_dict(
    sample: Mapping[str, Any],
) -> CohortSample:
    """Convert one legacy cohort sample."""

    return CohortSample(
        query_id=str(sample["q_id"]),
        question=str(sample["question"]),
        facts=tuple(
            fact_from_dict(fact)
            for fact in sample["facts"]
        ),
        candidates=tuple(
            candidate_from_dict(candidate, position)
            for position, candidate in enumerate(
                sample["candidates"]
            )
        ),
    )


def load_cohort_batch(
    path: Path,
) -> tuple[CohortSample, ...]:
    """Load one samples_sentence_labels.json file."""

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    samples = payload.get("samples")

    if not isinstance(samples, list):
        raise ValueError(
            "cohort batch must contain a samples list"
        )

    return tuple(
        sample_from_dict(sample)
        for sample in samples
    )


def fact_to_dict(fact: GoldFact) -> dict[str, Any]:
    """Serialize one gold fact."""

    return {
        "fact_id": fact.fact_id,
        "title": fact.title,
        "sentence_id": fact.sentence_index,
        "text": fact.text,
    }


def candidate_to_dict(
    candidate: CohortCandidate,
) -> dict[str, Any]:
    """Serialize one evidence-labelled candidate."""

    output = {
        "doc_id": candidate.document_id,
        "dense_rank": candidate.retrieval_rank,
        "dense_score": candidate.retriever_score,
        "text": candidate.text,
        "covered_facts": sorted(
            candidate.covered_fact_ids
        ),
    }

    if candidate.corpus_index is not None:
        output["corpus_index"] = candidate.corpus_index

    return output


def sample_to_dict(
    sample: CohortSample,
) -> dict[str, Any]:
    """Serialize one canonical cohort sample."""

    return {
        "q_id": sample.query_id,
        "question": sample.question,
        "facts": [
            fact_to_dict(fact)
            for fact in sample.facts
        ],
        "candidates": [
            candidate_to_dict(candidate)
            for candidate in sample.candidates
        ],
    }


def write_cohort_batch(
    path: Path,
    samples: Sequence[CohortSample],
) -> None:
    """Atomically write one normalized cohort batch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp{path.suffix}"
    )
    payload = {
        "samples": [
            sample_to_dict(sample)
            for sample in samples
        ]
    }

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)