"""SPLADE retrieval over precomputed BERGEN index shards."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from pace.cohort.retrieval import (
    save_retrieval_results,
)
from pace.preprocessing.modeling import (
    ModelBundle,
    encode_splade,
    inference_dtype,
    resolve_device,
    load_splade_model,
)

def encode_retrieval_queries(
    bundle: ModelBundle,
    questions: Sequence[str],
    *,
    max_length: int = 128,
) -> torch.Tensor:
    """Encode retrieval queries with the fixed SPLADE batch size."""

    if not questions:
        raise ValueError(
            "at least one question is required"
        )

    return encode_splade(
        bundle.model,
        bundle.tokenizer,
        questions,
        str(bundle.device),
        max_length=max_length,
    )

def embedding_shard_paths(
    index_dir: Path,
) -> tuple[Path, ...]:
    """Return embedding shards in numeric order."""

    paths = list(
        index_dir.glob("embedding_chunk_*.pt")
    )
    if not paths:
        raise FileNotFoundError(
            f"no embedding shards found under {index_dir}"
        )

    try:
        return tuple(
            sorted(
                paths,
                key=lambda path: int(
                    path.stem.rsplit("_", 1)[1]
                ),
            )
        )
    except ValueError as error:
        raise ValueError(
            "embedding shard names must end in integers"
        ) from error


def merge_topk(
    best_scores: torch.Tensor,
    best_indices: torch.Tensor,
    shard_scores: torch.Tensor,
    *,
    corpus_offset: int,
    candidate_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one index shard into the current global Top-K."""

    if candidate_count <= 0:
        raise ValueError(
            "candidate count must be positive"
        )
    if corpus_offset < 0:
        raise ValueError(
            "corpus offset must be non-negative"
        )
    if (
        best_scores.ndim != 2
        or best_indices.ndim != 2
        or shard_scores.ndim != 2
    ):
        raise ValueError(
            "Top-K tensors must be two-dimensional"
        )
    if best_scores.shape != best_indices.shape:
        raise ValueError(
            "best scores and indices must align"
        )
    if best_scores.shape[0] != shard_scores.shape[0]:
        raise ValueError(
            "query counts must match"
        )
    if best_scores.shape[1] != candidate_count:
        raise ValueError(
            "best tensors must use candidate_count columns"
        )

    local_count = min(
        candidate_count,
        shard_scores.shape[1],
    )
    local_scores, local_indices = torch.topk(
        shard_scores,
        local_count,
        dim=1,
    )
    local_indices = (
        local_indices.cpu()
        + corpus_offset
    )

    combined_scores = torch.cat(
        [
            best_scores.cpu(),
            local_scores.cpu(),
        ],
        dim=1,
    )
    combined_indices = torch.cat(
        [
            best_indices.cpu(),
            local_indices,
        ],
        dim=1,
    )

    merged_scores, positions = torch.topk(
        combined_scores,
        candidate_count,
        dim=1,
    )
    merged_indices = torch.gather(
        combined_indices,
        1,
        positions,
    )

    return merged_scores, merged_indices

def retrieve_topk(
    query_vectors: torch.Tensor,
    index_dir: Path,
    *,
    device: str = "auto",
    candidate_count: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve global Top-K from all BERGEN index shards."""

    if query_vectors.ndim != 2:
        raise ValueError(
            "query vectors must be two-dimensional"
        )
    if len(query_vectors) == 0:
        raise ValueError(
            "at least one query vector is required"
        )
    if candidate_count <= 0:
        raise ValueError(
            "candidate count must be positive"
        )

    resolved_device = resolve_device(device)
    dtype = inference_dtype(resolved_device)

    normalized_queries = torch.nn.functional.normalize(
        query_vectors.float(),
        dim=1,
    )
    sparse_queries = (
        normalized_queries
        .to(
            device=resolved_device,
            dtype=dtype,
        )
        .to_sparse()
    )

    query_count = len(query_vectors)
    best_scores = torch.full(
        (query_count, candidate_count),
        -torch.inf,
        dtype=torch.float32,
    )
    best_indices = torch.full(
        (query_count, candidate_count),
        -1,
        dtype=torch.long,
    )

    corpus_offset = 0

    for shard_path in embedding_shard_paths(
        index_dir
    ):
        documents = torch.load(
            shard_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(documents, torch.Tensor):
            raise ValueError(
                f"{shard_path} does not contain a tensor"
            )
        if documents.ndim != 2:
            raise ValueError(
                f"{shard_path} must be two-dimensional"
            )
        if documents.shape[0] == 0:
            raise ValueError(
                f"{shard_path} is empty"
            )
        if documents.shape[1] != query_vectors.shape[1]:
            raise ValueError(
                f"{shard_path} vocabulary dimension "
                "does not match queries"
            )

        documents = documents.to(
            device=resolved_device,
            dtype=dtype,
        )
        shard_scores = torch.sparse.mm(
            sparse_queries,
            documents.T,
        ).to_dense()

        best_scores, best_indices = merge_topk(
            best_scores,
            best_indices,
            shard_scores,
            corpus_offset=corpus_offset,
            candidate_count=candidate_count,
        )

        corpus_offset += documents.shape[0]
        del documents, shard_scores

    if corpus_offset < candidate_count:
        raise ValueError(
            "index contains fewer passages than "
            "candidate_count"
        )
    if torch.any(best_indices < 0):
        raise ValueError(
            "retrieval did not produce enough candidates"
        )

    return (
        best_indices.numpy().astype(
            np.int64,
            copy=False,
        ),
        best_scores.numpy().astype(
            np.float32,
            copy=False,
        ),
    )

def retrieve_questions(
    questions: Sequence[str],
    index_dir: Path,
    output_path: Path,
    *,
    device: str = "auto",
    candidate_count: int = 100,
    local_files_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode questions, retrieve Top-K, and save results."""

    bundle = load_splade_model(
        device=device,
        local_files_only=local_files_only,
    )
    query_vectors = encode_retrieval_queries(
        bundle,
        questions,
        max_length=128,
    )

    # Retrieval no longer needs the SPLADE model. Move it away
    # before loading document-index shards onto the GPU.
    bundle.model.to("cpu")
    if bundle.device.type == "cuda":
        torch.cuda.empty_cache()

    indices, scores = retrieve_topk(
        query_vectors,
        index_dir,
        device=str(bundle.device),
        candidate_count=candidate_count,
    )
    save_retrieval_results(
        output_path,
        indices,
        scores,
    )

    return indices, scores

