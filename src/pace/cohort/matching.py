"""Match retrieved passages to gold supporting facts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from pace.cohort.schema import GoldFact


DatasetName = Literal["hotpot", "2wiki"]


def normalize_hotpot(text: str) -> str:
    """Apply the original HotpotQA matching normalization."""

    normalized = unicodedata.normalize(
        "NFKC",
        text,
    ).lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_2wiki(text: str) -> str:
    """Apply the original 2Wiki matching normalization."""

    normalized = unicodedata.normalize(
        "NFKD",
        text,
    ).casefold()
    return " ".join(
        re.findall(
            r"[\w]+",
            normalized,
            flags=re.UNICODE,
        )
    )


def hotpot_covered_fact_ids(
    passage: str,
    facts: Sequence[GoldFact],
) -> frozenset[str]:
    """Find HotpotQA facts covered by one BERGEN passage."""

    normalized_passage = normalize_hotpot(passage)
    passage_title = normalize_hotpot(
        passage.split(":", 1)[0]
    )

    return frozenset(
        fact.fact_id
        for fact in facts
        if passage_title == normalize_hotpot(fact.title)
        and normalize_hotpot(fact.text)
        and normalize_hotpot(fact.text)
        in normalized_passage
    )


def two_wiki_covered_fact_ids(
    passage: str,
    facts: Sequence[GoldFact],
) -> frozenset[str]:
    """Find 2Wiki facts covered by one BERGEN passage."""

    normalized_passage = normalize_2wiki(passage)

    return frozenset(
        fact.fact_id
        for fact in facts
        if normalized_passage.startswith(
            normalize_2wiki(fact.title) + " "
        )
        and normalize_2wiki(fact.text)
        in normalized_passage
    )


def covered_fact_ids(
    passage: str,
    facts: Sequence[GoldFact],
    dataset: DatasetName,
) -> frozenset[str]:
    """Dispatch to the dataset-specific matching policy."""

    if dataset == "hotpot":
        return hotpot_covered_fact_ids(
            passage,
            facts,
        )
    if dataset == "2wiki":
        return two_wiki_covered_fact_ids(
            passage,
            facts,
        )

    raise ValueError(
        f"unsupported matching dataset: {dataset}"
    )
