"""Read passages from a BERGEN Hugging Face corpus."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pace.cohort.schema import CorpusPassage


class BergenCorpus:
    """Read-only adapter around a BERGEN corpus dataset."""
    def __len__(self) -> int:
        return len(self._dataset)

    def iter_contents(
        self,
        *,
        batch_size: int = 65536,
    ):
        """Yield all corpus contents in stable dataset order."""

        if batch_size <= 0:
            raise ValueError(
                "batch size must be positive"
            )

        for start in range(
            0,
            len(self),
            batch_size,
        ):
            stop = min(
                start + batch_size,
                len(self),
            )
            material = self._dataset[start:stop]
            contents = material["content"]

            if len(contents) != stop - start:
                raise ValueError(
                    "BERGEN corpus returned an "
                    "unexpected row count"
                )

            yield from (
                str(content)
                for content in contents
            )

    def __init__(self, dataset: Any):
        columns = set(dataset.column_names)
        if "content" not in columns:
            raise ValueError(
                "BERGEN corpus must contain a content column"
            )

        self._dataset = dataset
        self._has_document_ids = "id" in columns

    @classmethod
    def from_disk(
        cls,
        path: Path,
    ) -> BergenCorpus:
        """Load a BERGEN corpus saved by datasets.save_to_disk."""

        from datasets import load_from_disk

        return cls(load_from_disk(str(path)))

    def fetch(
        self,
        indices: Sequence[int],
    ) -> dict[int, CorpusPassage]:
        """Fetch unique corpus passages by integer index."""

        unique_indices = sorted(
            {int(index) for index in indices}
        )
        if any(index < 0 for index in unique_indices):
            raise ValueError(
                "corpus indices must be non-negative"
            )
        if not unique_indices:
            return {}

        material = self._dataset[unique_indices]
        contents = material["content"]

        if self._has_document_ids:
            document_ids = material["id"]
        else:
            document_ids = unique_indices

        if (
            len(contents) != len(unique_indices)
            or len(document_ids) != len(unique_indices)
        ):
            raise ValueError(
                "BERGEN corpus returned an unexpected row count"
            )

        return {
            corpus_index: CorpusPassage(
                corpus_index=corpus_index,
                document_id=str(document_id),
                text=str(content),
            )
            for corpus_index, document_id, content in zip(
                unique_indices,
                document_ids,
                contents,
            )
        }
