"""Repository-wide fixed experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSizes:
    """Fixed batch sizes required by all experiments."""

    reranker_pair: int = 8
    llm_generator: int = 10
    provence_compressor: int = 4
    splade_encoder: int = 8

    def manifest_dict(self) -> dict[str, int]:
        """Return stable names used in experiment manifests."""

        return {
            "reranker": self.reranker_pair,
            "llm": self.llm_generator,
            "provence": self.provence_compressor,
        }


BATCH_SIZES = BatchSizes()