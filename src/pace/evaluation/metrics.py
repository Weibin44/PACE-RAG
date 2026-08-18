
"""Evidence-ranking metrics shared by all datasets."""

from __future__ import annotations

from collections.abc import Sequence

from pace.data.schema import EvidenceExample


def evidence_scores(
    example: EvidenceExample,
    order: Sequence[int],
) -> dict[str, float]:
    """Compute evidence metrics for one selected document prefix."""

    selected = [example.candidates[index] for index in order]

    covered_fact_ids = {
        fact_id
        for candidate in selected
        for fact_id in candidate.covered_fact_ids
        if fact_id in example.gold_fact_ids
    }

    relevant_documents = sum(
        bool(candidate.covered_fact_ids & example.gold_fact_ids)
        for candidate in selected
    )

    returned = len(selected)
    gold_count = len(example.gold_fact_ids)

    supporting_fact_recall = (
        len(covered_fact_ids) / gold_count
        if gold_count
        else 1.0
    )

    return {
        "complete_evidence_recall": float(
            covered_fact_ids >= example.gold_fact_ids
        ),
        "supporting_fact_recall": supporting_fact_recall,
        "precision": (
            relevant_documents / returned
            if returned
            else 0.0
        ),
        "returned_k": returned,
    }


# """Pure evidence-ranking and HotpotQA supporting-fact metrics."""

# from __future__ import annotations

# import numpy as np


# def evidence_scores(sample: dict, order: list[int]) -> dict[str, float]:
#     """Compute query-level metrics for one returned document prefix."""
#     gold = {fact["fact_id"] for fact in sample["facts"]}
#     selected = [sample["candidates"][index] for index in order]
#     covered = {
#         fact_id
#         for candidate in selected
#         for fact_id in candidate.get("covered_facts", [])
#         if fact_id in gold
#     }
#     relevant_documents = sum(
#         bool(set(candidate.get("covered_facts", [])) & gold) for candidate in selected
#     )
#     returned = len(selected)
#     recall = len(covered) / len(gold) if gold else 1.0
#     return {
#         "complete_evidence_recall": float(covered >= gold),
#         "supporting_fact_recall": recall,
#         "precision": relevant_documents / returned if returned else 0.0,
#         "returned_k": returned,
#     }


