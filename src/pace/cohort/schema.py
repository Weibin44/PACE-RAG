"""Canonical schemas for evidence-labelled retrieval cohorts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldFact:
    """One gold supporting sentence."""

    fact_id: str
    title: str
    sentence_index: int
    text: str


@dataclass(frozen=True)
class Query:
    """One question before gold-evidence resolution."""

    query_id: str
    question: str


@dataclass(frozen=True)
class GoldQuery(Query):
    """One question with its gold supporting evidence."""

    facts: tuple[GoldFact, ...]

    @property
    def gold_fact_ids(self) -> frozenset[str]:
        return frozenset(
            fact.fact_id
            for fact in self.facts
        )


@dataclass(frozen=True)
class CorpusPassage:
    """One passage loaded from the BERGEN corpus."""

    corpus_index: int
    document_id: str
    text: str


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage returned by the external retriever."""

    document_id: str
    text: str
    retriever_score: float
    retrieval_rank: int
    corpus_index: int | None = None


@dataclass(frozen=True)
class CohortCandidate:
    """One retrieved passage with its covered gold facts."""

    document_id: str
    text: str
    retriever_score: float
    retrieval_rank: int
    covered_fact_ids: frozenset[str]
    corpus_index: int | None = None


@dataclass(frozen=True)
class CohortSample(GoldQuery):
    """One query and its evidence-labelled retrieved passages."""

    candidates: tuple[CohortCandidate, ...]

    @property
    def retrieved_fact_ids(self) -> frozenset[str]:
        return frozenset(
            fact_id
            for candidate in self.candidates
            for fact_id in candidate.covered_fact_ids
        )

    @property
    def complete_evidence_retrieved(self) -> bool:
        return self.gold_fact_ids <= self.retrieved_fact_ids
