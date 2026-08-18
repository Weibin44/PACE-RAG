"""Load questions and sentence-level gold evidence."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pace.cohort.schema import (
    GoldFact,
    GoldQuery,
    Query,
)


def resolve_gold_facts(
    example: Mapping[str, Any],
) -> tuple[GoldFact, ...]:
    """Resolve supporting-fact labels against context sentences."""

    contexts: dict[str, list[list[str]]] = defaultdict(list)
    for title, sentences in example["context"]:
        contexts[str(title)].append(
            [str(sentence) for sentence in sentences]
        )

    facts = []
    for title, sentence_index in example["supporting_facts"]:
        title = str(title)
        sentence_index = int(sentence_index)

        matching_contexts = [
            sentences
            for sentences in contexts.get(title, [])
            if sentence_index < len(sentences)
        ]
        if not matching_contexts:
            raise ValueError(
                "unresolvable supporting fact: "
                f"{example.get('_id')} "
                f"{title}::{sentence_index}"
            )

        # 2Wiki can contain duplicate titles with different contexts.
        sentences = max(matching_contexts, key=len)
        facts.append(
            GoldFact(
                fact_id=f"{title}::{sentence_index}",
                title=title,
                sentence_index=sentence_index,
                text=sentences[sentence_index],
            )
        )

    return tuple(facts)


def _load_json_list(path: Path) -> list[dict]:
    values = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(values, list):
        raise ValueError(
            f"{path} must contain a JSON list"
        )
    return values


def load_hotpot_gold_queries(
    evaluation_path: Path,
    labels_path: Path,
    *,
    seed: int = 2026,
) -> tuple[GoldQuery, ...]:
    """Reproduce the stable HotpotQA query order."""

    evaluation = {
        str(example["q_id"]): example
        for example in _load_json_list(evaluation_path)
    }
    labels = {
        str(example["_id"]): example
        for example in _load_json_list(labels_path)
    }

    query_ids = sorted(
        set(evaluation) & set(labels)
    )
    random.Random(seed).shuffle(query_ids)

    return tuple(
        GoldQuery(
            query_id=query_id,
            question=str(
                evaluation[query_id]["question"]
            ),
            facts=resolve_gold_facts(
                labels[query_id]
            ),
        )
        for query_id in query_ids
    )


def load_2wiki_questions(
    labels_path: Path,
) -> tuple[Query, ...]:
    """Load all 2Wiki questions in official dev order."""

    return tuple(
        Query(
            query_id=str(example["_id"]),
            question=str(example["question"]),
        )
        for example in _load_json_list(labels_path)
    )


def load_2wiki_gold_queries(
    labels_path: Path,
    query_ids: Sequence[str],
) -> tuple[GoldQuery, ...]:
    """Resolve selected 2Wiki queries in the requested order."""

    labels = {
        str(example["_id"]): example
        for example in _load_json_list(labels_path)
    }

    missing = [
        query_id
        for query_id in query_ids
        if query_id not in labels
    ]
    if missing:
        raise KeyError(
            f"unknown 2Wiki query IDs: {missing[:5]}"
        )

    return tuple(
        GoldQuery(
            query_id=query_id,
            question=str(labels[query_id]["question"]),
            facts=resolve_gold_facts(labels[query_id]),
        )
        for query_id in query_ids
    )
