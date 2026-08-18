"""Audit 2Wiki gold evidence against the full BERGEN corpus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pace.cohort.matching import normalize_2wiki


FactKey = tuple[str, str]


def audit_2wiki_corpus_coverage(
    examples: Sequence[Mapping[str, Any]],
    corpus_contents: Iterable[str],
) -> dict:
    """Find queries whose complete evidence exists in the corpus."""

    facts: dict[FactKey, dict] = {}
    query_facts: dict[str, list[FactKey]] = {}
    titles_by_lead: dict[
        str,
        set[str],
    ] = defaultdict(set)
    facts_by_title: dict[
        str,
        list[FactKey],
    ] = defaultdict(list)

    for example in examples:
        contexts: dict[
            str,
            list[list[str]],
        ] = defaultdict(list)

        for title, sentences in example["context"]:
            contexts[str(title)].append(
                [str(sentence) for sentence in sentences]
            )

        keys = []
        for title, sentence_index in example[
            "supporting_facts"
        ]:
            title = str(title)
            sentence_index = int(sentence_index)

            candidates = [
                sentences
                for sentences in contexts.get(title, [])
                if sentence_index < len(sentences)
            ]
            sentence = (
                max(candidates, key=len)[sentence_index]
                if candidates
                else None
            )

            normalized_title = normalize_2wiki(title)
            normalized_sentence = (
                normalize_2wiki(sentence)
                if sentence is not None
                else (
                    "__invalid_sentence_id_"
                    f"{sentence_index}"
                )
            )
            key = (
                normalized_title,
                normalized_sentence,
            )

            facts.setdefault(
                key,
                {
                    "title": title,
                    "sentence": sentence,
                    "sentence_id": sentence_index,
                    "label_resolvable": (
                        sentence is not None
                    ),
                },
            )
            keys.append(key)

            lead = normalize_2wiki(
                title.split(":", 1)[0]
            )
            titles_by_lead[lead].add(
                normalized_title
            )

        query_facts[str(example["_id"])] = keys

    for key in facts:
        facts_by_title[key[0]].append(key)

    matched_titles: set[str] = set()
    matched_facts: set[FactKey] = set()
    corpus_chunk_count = 0

    for content in corpus_contents:
        corpus_chunk_count += 1
        lead = normalize_2wiki(
            content.split(":", 1)[0]
        )
        candidate_titles = titles_by_lead.get(
            lead,
        )
        if not candidate_titles:
            continue

        normalized_content = normalize_2wiki(
            content
        )
        for title in candidate_titles:
            if not normalized_content.startswith(
                title + " "
            ):
                continue

            matched_titles.add(title)
            for key in facts_by_title[title]:
                if (
                    facts[key]["label_resolvable"]
                    and key[1] in normalized_content
                ):
                    matched_facts.add(key)

    complete_query_ids = [
        query_id
        for query_id, keys in query_facts.items()
        if all(
            key in matched_facts
            for key in keys
        )
    ]
    queries_with_any_fact = sum(
        any(
            key in matched_facts
            for key in keys
        )
        for keys in query_facts.values()
    )
    resolvable_facts = {
        key
        for key, value in facts.items()
        if value["label_resolvable"]
    }

    return {
        "dev_queries": len(examples),
        "corpus_chunks": corpus_chunk_count,
        "unique_gold_titles": len(
            {key[0] for key in facts}
        ),
        "matched_gold_titles": len(matched_titles),
        "unique_gold_facts": len(facts),
        "resolvable_gold_facts": len(
            resolvable_facts
        ),
        "invalid_sentence_labels": (
            len(facts) - len(resolvable_facts)
        ),
        "matched_gold_facts": len(matched_facts),
        "queries_with_any_fact": (
            queries_with_any_fact
        ),
        "queries_with_all_facts": len(
            complete_query_ids
        ),
        "complete_query_ids": complete_query_ids,
    }
