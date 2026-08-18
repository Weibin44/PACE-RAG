"""Tests for dataset-specific evidence matching."""

import pytest

from pace.cohort.matching import (
    covered_fact_ids,
    normalize_2wiki,
    normalize_hotpot,
)
from pace.cohort.schema import GoldFact


FACT = GoldFact(
    fact_id="Example (film)::0",
    title="Example (film)",
    sentence_index=0,
    text="This is an example sentence.",
)


def test_hotpot_normalization():
    assert normalize_hotpot(
        "  Example—TEXT!  "
    ) == "example text"


def test_hotpot_requires_title_and_sentence():
    passage = (
        "Example (film): "
        "This is an example sentence!"
    )

    assert covered_fact_ids(
        passage,
        [FACT],
        "hotpot",
    ) == frozenset({"Example (film)::0"})

    assert not covered_fact_ids(
        "Other title: This is an example sentence.",
        [FACT],
        "hotpot",
    )


def test_2wiki_normalization():
    assert normalize_2wiki(
        "Example—TEXT!"
    ) == "example text"


def test_2wiki_requires_title_prefix():
    passage = (
        "Example (film): "
        "This is an example sentence."
    )

    assert covered_fact_ids(
        passage,
        [FACT],
        "2wiki",
    ) == frozenset({"Example (film)::0"})

    assert not covered_fact_ids(
        "Other: Example (film) "
        "This is an example sentence.",
        [FACT],
        "2wiki",
    )


def test_matching_rejects_unknown_dataset():
    with pytest.raises(
        ValueError,
        match="unsupported matching dataset",
    ):
        covered_fact_ids(
            "passage",
            [FACT],
            "unknown",  # type: ignore[arg-type]
        )