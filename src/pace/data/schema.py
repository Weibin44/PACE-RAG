"""Shared data structures for effectiveness and serving experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """One retrieved candidate document."""

    document_id: str
    text: str
    retriever_score: float
    covered_fact_ids: frozenset[str]


@dataclass(frozen=True)
class EvidenceExample:
    """One query and all cached features required by ranking methods."""

    query_id: str
    question: str
    gold_fact_ids: frozenset[str]
    candidates: tuple[Candidate, ...]
    reranker_scores: np.ndarray
    query_weights: np.ndarray
    document_features: np.ndarray
    document_similarity: np.ndarray
    query_similarity: np.ndarray

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    def validate(self) -> None:
        """Validate that all cached arrays match the candidate pool."""
        count = self.candidate_count

        if self.reranker_scores.shape != (count,):
            raise ValueError("reranker_scores must have shape [documents]")

        if self.document_features.shape[0] != count:
            raise ValueError("document_features must have one row per document")

        if self.document_similarity.shape != (count, count):
            raise ValueError(
                "document_similarity must have shape [documents, documents]"
            )

        if self.query_similarity.shape != (count,):
            raise ValueError("query_similarity must have shape [documents]")

        if self.document_features.shape[1] != len(self.query_weights):
            raise ValueError(
                "document feature width must match query_weights"
            )