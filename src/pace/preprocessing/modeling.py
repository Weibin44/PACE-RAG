"""Shared neural-model inference utilities for cache generation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from pace.config import BATCH_SIZES
from dataclasses import dataclass
from typing import Any


SPLADE_MODEL_NAME = "naver/splade-v3"
RERANKER_MODEL_NAME = "naver/trecdl22-crossencoder-debertav3"


@dataclass(frozen=True)
class ModelBundle:
    """Loaded tokenizer, model, and runtime configuration."""

    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype


def resolve_device(value: str) -> torch.device:
    """Resolve auto/CPU/CUDA device selection."""

    if value == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return device


def inference_dtype(device: torch.device) -> torch.dtype:
    """Use reproducible GPU FP16 and safe CPU FP32 inference."""

    return (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )


def load_splade_model(
    *,
    device: str = "auto",
    local_files_only: bool = False,
    model_name: str = SPLADE_MODEL_NAME,
) -> ModelBundle:
    """Load SPLADE with explicit online/offline behavior."""

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
    )

    resolved_device = resolve_device(device)
    dtype = inference_dtype(resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = (
        AutoModelForMaskedLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(resolved_device)
    )

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=dtype,
    )


def load_reranker_model(
    *,
    device: str = "auto",
    local_files_only: bool = False,
    model_name: str = RERANKER_MODEL_NAME,
) -> ModelBundle:
    """Load the cross-encoder reranker."""

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    resolved_device = resolve_device(device)
    dtype = inference_dtype(resolved_device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .eval()
        .to(resolved_device)
    )

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=resolved_device,
        dtype=dtype,
    )

def splade_pool(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply SPLADE log-ReLU max pooling."""

    if logits.ndim != 3:
        raise ValueError("SPLADE logits must have shape [batch, tokens, vocab]")
    if attention_mask.shape != logits.shape[:2]:
        raise ValueError(
            "attention mask must match SPLADE batch and token dimensions"
        )

    return torch.max(
        torch.log1p(torch.relu(logits))
        * attention_mask.unsqueeze(-1),
        dim=1,
    ).values.float().cpu()


def encode_splade(
    model,
    tokenizer,
    texts: Sequence[str],
    device: str,
    *,
    max_length: int,
) -> torch.Tensor:
    """Encode text into full-vocabulary SPLADE vectors."""

    if not texts:
        raise ValueError("at least one text is required")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    outputs = []

    with torch.inference_mode():
        for start in range(
            0,
            len(texts),
            BATCH_SIZES.splade_encoder,
        ):
            batch = tokenizer(
                list(
                    texts[
                        start : start
                        + BATCH_SIZES.splade_encoder
                    ]
                ),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**batch).logits
            outputs.append(
                splade_pool(
                    logits,
                    batch.attention_mask,
                )
            )

    return torch.cat(outputs, dim=0)


def scalar_scores_from_logits(
    logits: torch.Tensor,
) -> torch.Tensor:
    """Convert classifier logits to one score per query-document pair."""

    if logits.ndim == 1:
        return logits.detach().float().cpu()

    if logits.ndim != 2:
        raise ValueError(
            "reranker logits must have shape [batch] or [batch, classes]"
        )

    if logits.shape[1] == 1:
        return logits[:, 0].detach().float().cpu()

    return logits[:, -1].detach().float().cpu()


def score_reranker_pairs(
    model,
    tokenizer,
    pairs: Sequence[tuple[str, str]],
    device: str,
    *,
    max_length: int = 256,
) -> np.ndarray:
    """Score query-document pairs using the fixed reranker batch size."""

    if not pairs:
        return np.empty(0, dtype=np.float32)
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    scores = []

    with torch.inference_mode():
        for start in range(
            0,
            len(pairs),
            BATCH_SIZES.reranker_pair,
        ):
            batch_pairs = pairs[
                start : start + BATCH_SIZES.reranker_pair
            ]
            encoded = tokenizer(
                [pair[0] for pair in batch_pairs],
                [pair[1] for pair in batch_pairs],
                padding=True,
                truncation="only_second",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            logits = model(**encoded).logits
            scores.append(scalar_scores_from_logits(logits))

    return torch.cat(scores).numpy().astype(np.float32)