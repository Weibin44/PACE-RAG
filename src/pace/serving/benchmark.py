#!/usr/bin/env python3
"""Open-loop RAG serving benchmark with optional online Provence."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from pace.config import BATCH_SIZES
from pace.evaluation.calibration import load_calibration_manifest
from pace.methods.baselines import (
    dartboard_cosine_order,
    dartboard_hybrid_order,
    mmr_order,
    rocchio_prf_order,
)
from pace.methods.pace import (
    greedy_evidence_frontloading_order,
    soft_anchor_relevance_components,
)
from pace.methods.utils import minmax_normalize
from pace.preprocessing.modeling import (
    scalar_scores_from_logits as tensor_scores_from_logits,
)


def scalar_scores_from_logits(
    logits: torch.Tensor,
) -> list[float]:
    return tensor_scores_from_logits(logits).tolist()


def normalize_scores(
    values: np.ndarray,
    method: str,
) -> np.ndarray:
    if method != "minmax":
        raise ValueError(
            f"unsupported normalization: {method}"
        )
    return minmax_normalize(values)


def sqrt_quality_order(
    query_weights: np.ndarray,
    document_features: np.ndarray,
    relevance: np.ndarray,
    limit: int,
) -> list[int]:
    return greedy_evidence_frontloading_order(
        query_weights,
        document_features,
        relevance,
        limit,
    )


def evidence_scores(
    sample: dict[str, Any],
    order: list[int],
) -> dict[str, float]:
    gold = {
        fact["fact_id"]
        for fact in sample["facts"]
    }
    selected = [
        sample["candidates"][index]
        for index in order
    ]
    covered = {
        fact_id
        for candidate in selected
        for fact_id in candidate.get(
            "covered_facts",
            [],
        )
        if fact_id in gold
    }
    relevant_documents = sum(
        bool(
            set(
                candidate.get(
                    "covered_facts",
                    [],
                )
            )
            & gold
        )
        for candidate in selected
    )
    returned = len(selected)
    recall = (
        len(covered) / len(gold)
        if gold
        else 1.0
    )

    return {
        "complete_evidence_recall": float(
            covered >= gold
        ),
        "supporting_fact_recall": recall,
        "precision": (
            relevant_documents / returned
            if returned
            else 0.0
        ),
        "returned_k": returned,
    }


def has_complete_evidence(
    sample: dict[str, Any],
) -> bool:
    gold = {
        fact["fact_id"]
        for fact in sample["facts"]
    }
    retrieved = {
        fact_id
        for candidate in sample["candidates"]
        for fact_id in candidate.get(
            "covered_facts",
            [],
        )
    }
    return gold <= retrieved


@dataclass
class Ewma:
    value: float
    decay: float = 0.9

    def update(
        self,
        observation: float,
    ) -> None:
        if observation > 0:
            self.value = (
                self.decay * self.value
                + (1.0 - self.decay) * observation
            )


@dataclass(frozen=True)
class DepthDecision:
    depth: int
    reranker_wait: float
    llm_wait: float
    reranker_pairs_per_second: float


class QueueAwareDepthController:
    def __init__(
        self,
        *,
        min_depth: int,
        max_depth: int,
        reranker_batch_size: int,
        llm_batch_size: int,
        initial_reranker_throughput: float,
        initial_llm_batch_seconds: float,
        ewma_decay: float = 0.9,
    ) -> None:
        if not 1 <= min_depth <= max_depth:
            raise ValueError(
                "require 1 <= min_depth <= max_depth"
            )

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reranker_batch_size = (
            reranker_batch_size
        )
        self.llm_batch_size = llm_batch_size
        self.reranker_throughput = Ewma(
            initial_reranker_throughput,
            ewma_decay,
        )
        self.llm_batch_seconds = Ewma(
            initial_llm_batch_seconds,
            ewma_decay,
        )

    def decide(
        self,
        *,
        pending_reranker_pairs: int,
        llm_queue_size: int,
        llm_active_remaining_seconds: float,
    ) -> DepthDecision:
        throughput = max(
            self.reranker_throughput.value,
            1e-6,
        )
        reranker_wait = (
            pending_reranker_pairs / throughput
        )
        queued_waves = math.ceil(
            llm_queue_size / self.llm_batch_size
        )
        llm_wait = max(
            llm_active_remaining_seconds,
            0.0,
        ) + (
            queued_waves
            * self.llm_batch_seconds.value
        )
        excess_pairs = throughput * max(
            reranker_wait - llm_wait,
            0.0,
        )
        batches_to_remove = math.floor(
            excess_pairs
            / self.reranker_batch_size
        )
        depth = max(
            self.min_depth,
            self.max_depth
            - batches_to_remove
            * self.reranker_batch_size,
        )

        return DepthDecision(
            depth=depth,
            reranker_wait=reranker_wait,
            llm_wait=llm_wait,
            reranker_pairs_per_second=throughput,
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--calibration-manifest", type=Path, required=True)

    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workload-dir", type=Path, required=True)

    p.add_argument("--qps-values", default="0.5,0.75,1.0,1.15,1.3,1.45,1.6,1.8")
    p.add_argument(
        "--methods",
        default=(
            "fixed_d100,queue_adaptive_dense,queue_adaptive_prf,"
            "queue_adaptive_mmr,queue_adaptive_dartboard,"
            "queue_adaptive_coverage_only_d,queue_adaptive_sqrt,"
            "queue_adaptive_anchor_only,queue_adaptive_soft_anchor"
        ),
    )
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--repetition-offset", type=int, default=0)
    p.add_argument("--order-offset", type=int, default=0)
    p.add_argument("--warmup-seconds", type=float, default=60)
    p.add_argument("--measurement-seconds", type=float, default=300)
    p.add_argument("--request-timeout-seconds", type=float, default=300)
    p.add_argument(
        "--no-request-deadline", action="store_true",
        help="Never discard requests; stop arrivals at the window boundary and drain all queues.",
    )
    p.add_argument("--unique-measurement-pass", action="store_true")
    p.add_argument(
        "--prime-pipeline", action="store_true",
        help="Run one unmeasured request through the empty pipeline before timing arrivals.",
    )


    p.add_argument("--doc-count", type=int, default=100)
    p.add_argument("--candidate-pool-size", type=int, default=100)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument(
        "--natural-eos", action="store_true",
        help="Treat --output-tokens as a maximum and allow generation to stop at EOS",
    )


    p.add_argument("--max-batch-wait-ms", type=float, default=10)
    p.add_argument("--reranker-max-length", type=int, default=256)


    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--provence-device", default="cuda:2")
    p.add_argument("--provence-model", default="naver/provence-reranker-debertav3-v1")


    p.add_argument("--reranker-model", default="naver/trecdl22-crossencoder-debertav3")
    p.add_argument("--frontend-device", default="cuda:0")
    p.add_argument("--generator-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--llm-device", default="cuda:1")


    p.add_argument("--dynamic-d-min", type=int, default=20)
    p.add_argument("--initial-reranker-throughput", type=float, default=64.0)
    p.add_argument("--initial-llm-batch-seconds", type=float, default=1.0)
    p.add_argument("--controller-ewma-decay", type=float, default=0.9)
    
    args = p.parse_args()
    args.reranker_batch_size = BATCH_SIZES.reranker_pair
    args.provence_batch_size = BATCH_SIZES.provence_compressor
    args.llm_batch_size = BATCH_SIZES.llm_generator
    return args





def normalize_by_max(values: np.ndarray) -> np.ndarray:
    """Map non-negative retrieval scores to [0, 1] without changing order."""
    values = np.maximum(np.asarray(values, dtype=np.float32), 0)
    maximum = float(values.max()) if values.size else 0.0
    return values / maximum if maximum > 0 else np.zeros_like(values)



@dataclass
class Request:
    request_id: str
    sample: dict[str, Any]
    measured: bool
    arrival: float
    scheduled_offset: float
    actual_enqueue_offset: float
    workload_id: str
    query_id: str
    deadline: float
    future: asyncio.Future
    selected_contexts: list[str] = field(default_factory=list)
    compressed_context: str | None = None
    stamps: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodSpec:
    use_provence: bool = False
    dynamic_depth: bool = False
    d_coverage_policy: str | None = None
    k_coverage_policy: str | None = None
    literature_policy: str | None = None
    use_compressed_context: bool = False


METHOD_SPECS = {
    "fixed_d100": MethodSpec(use_provence=True, use_compressed_context=True),
    "queue_adaptive_dense": MethodSpec(
        use_provence=True, use_compressed_context=True, dynamic_depth=True,
    ),
    "queue_adaptive_sqrt": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True,
        d_coverage_policy="sqrt_quality", k_coverage_policy="sqrt_quality",
    ),
    "queue_adaptive_soft_anchor": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True,
        d_coverage_policy="soft_anchor", k_coverage_policy="soft_anchor",
    ),
    "queue_adaptive_anchor_only": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True,
        d_coverage_policy="anchor_only", k_coverage_policy="anchor_only",
    ),
    "queue_adaptive_coverage_only_d": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True, d_coverage_policy="coverage_only",
    ),
    "queue_adaptive_prf": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True, literature_policy="rocchio_prf",
    ),
    "queue_adaptive_mmr": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True, literature_policy="mmr",
    ),
    "queue_adaptive_dartboard": MethodSpec(
        use_provence=True, use_compressed_context=True,
        dynamic_depth=True, literature_policy="dartboard",
    ),
}


class ServingPipeline:
    def __init__(self, a, method, models, selection_cache):
        self.a, self.method = a, method
        self.spec = METHOD_SPECS[method]
        self.selection_cache = selection_cache
        self.pending_reranker_pairs = 0
        self.llm_active_started: float | None = None
        self.llm_active_expected_seconds = 0.0
        self.controller = QueueAwareDepthController(
            min_depth=a.dynamic_d_min,
            max_depth=a.doc_count,
            reranker_batch_size=a.reranker_batch_size,
            llm_batch_size=a.llm_batch_size,
            initial_reranker_throughput=a.initial_reranker_throughput,
            initial_llm_batch_seconds=a.initial_llm_batch_seconds,
            ewma_decay=a.controller_ewma_decay,
        )
        self.rr_tok, self.rr_model, self.prov_model, self.llm_tok, self.llm_model = models
        # The fast tokenizer is shared by the Provence and LLM workers but is not
        # safe to borrow concurrently from their separate execution threads.
        self.llm_tokenizer_lock = threading.Lock()
        self.rr_queue = asyncio.Queue()
        self.prov_queue = asyncio.Queue()
        self.llm_queue = asyncio.Queue()
        # Serialize only when reranker and Provence are explicitly co-located.
        self.front_lock = (
            asyncio.Lock() if a.frontend_device == a.provence_device else None
        )
        self.tasks: list[asyncio.Task] = []


    def literature_order(
        self,
        req,
        stage: str,
        raw_scores: np.ndarray,
        original_indices: np.ndarray,
        limit: int,
    ) -> list[int]:
        """Apply a calibrated literature baseline to the requested candidate view."""
        cached = self.selection_cache[req.query_id]
        similarities = cached["literature_similarities"][np.ix_(
            original_indices, original_indices
        )]
        policy = self.spec.literature_policy
        parameters = self.a.baseline_parameters
        if policy == "rocchio_prf":
            query_similarity = cached["query_similarity"][original_indices]
            feedback = np.argsort(-raw_scores, kind="stable").tolist()
            return rocchio_prf_order(
                query_similarity,
                similarities,
                feedback,
                alpha=float(parameters[f"{stage}_rocchio_alpha"]),
                beta=1.0 - float(parameters[f"{stage}_rocchio_alpha"]),
                feedback_depth=int(parameters[f"{stage}_rocchio_depth"]),
            )[:limit]
        if policy == "mmr":
            relevance = normalize_scores(raw_scores, "minmax")
            return mmr_order(
                relevance,
                similarities,
                diversity=float(parameters[f"{stage}_mmr"]),
                limit=limit,
            )
        if policy == "dartboard":
            sigma = float(parameters[f"{stage}_dartboard"])
            if stage == "D":
                query_similarity = cached["query_similarity"][original_indices]
                return dartboard_cosine_order(
                    query_similarity, similarities, sigma=sigma, limit=limit
                )
            return dartboard_hybrid_order(
                raw_scores, similarities, sigma=sigma, limit=limit
            )
        raise ValueError(f"Unknown literature policy: {policy}")

    def coverage_order(
        self,
        req,
        relevance: np.ndarray,
        original_indices: np.ndarray,
        limit: int,
        policy: str,
        stage: str,
    ) -> list[int]:
        """Rank a candidate view with one query-conditioned coverage variant."""
        cached = self.selection_cache[req.query_id]
        features = cached["document_features"][original_indices]
        if policy == "coverage_only":
            quality_relevance = np.ones(len(original_indices), dtype=np.float32)
        elif policy == "sqrt_quality":
            quality_relevance = relevance
        elif policy in {"soft_anchor", "anchor_only"}:
            similarities = cached["similarities"][np.ix_(
                original_indices, original_indices
            )]
            query_rel, anchor_rel, _, effective_anchors = (
                soft_anchor_relevance_components(relevance, similarities)
            )
            quality_relevance = (
                anchor_rel
                if policy == "anchor_only"
                else 1.0 - (1.0 - query_rel) * (1.0 - anchor_rel)
            )
            req.stamps[f"{stage.lower()}_effective_anchors"] = effective_anchors
            req.stamps["effective_anchors"] = effective_anchors
        else:
            raise ValueError(f"Unknown coverage policy: {policy}")
        return sqrt_quality_order(
            cached["query_weights"], features, quality_relevance, limit
        )



    async def start(self):
        self.tasks = [asyncio.create_task(self.llm_worker()), 
                      asyncio.create_task(self.reranker_worker())]
        if self.spec.use_provence:
            self.tasks.append(asyncio.create_task(self.provence_worker()))

    async def stop(self):
        await self.rr_queue.join()
        if self.spec.use_provence:
            await self.prov_queue.join()
        await self.llm_queue.join()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


    def rerank_batch_sync(self, question, texts):
        encoded = self.rr_tok(
            [question] * len(texts),
            texts,
            padding=True,
            truncation="only_second",
            max_length=self.a.reranker_max_length,
            return_tensors="pt",
        ).to(self.a.frontend_device)

        with torch.inference_mode():
            outputs = self.rr_model(**encoded, return_dict=True)

        torch.cuda.synchronize(torch.device(self.a.frontend_device))
        return scalar_scores_from_logits(outputs.logits)

    def llm_active_remaining(self) -> float:
        if self.llm_active_started is None:
            return 0.0
        elapsed = time.perf_counter() - self.llm_active_started
        return max(self.llm_active_expected_seconds - elapsed, 0.0)

    def prepare_queue_aware_request(self, req) -> None:
        """Freeze D and candidate order once, immediately before enqueueing."""
        candidates = req.sample["candidates"][: self.a.doc_count]
        decision = self.controller.decide(
            pending_reranker_pairs=self.pending_reranker_pairs,
            llm_queue_size=self.llm_queue.qsize(),
            llm_active_remaining_seconds=self.llm_active_remaining(),
        )
        depth = decision.depth if self.spec.dynamic_depth else self.a.doc_count
        order = list(range(self.a.doc_count))
        if self.spec.literature_policy:
            selection_started = time.perf_counter()
            raw_scores = np.asarray(
                [candidate["dense_score"] for candidate in candidates],
                dtype=np.float32,
            )
            order = self.literature_order(
                req, "D", raw_scores, np.arange(len(candidates)), depth
            )
            req.stamps["candidate_selection_seconds"] = (
                time.perf_counter() - selection_started
            )
        elif self.spec.d_coverage_policy:
            selection_started = time.perf_counter()
            relevance = normalize_by_max(np.asarray(
                [candidate["dense_score"] for candidate in candidates],
                dtype=np.float32,
            ))
            order = self.coverage_order(
                req,
                relevance,
                np.arange(len(candidates), dtype=np.int64),
                depth,
                self.spec.d_coverage_policy,
                "D",
            )
            req.stamps["candidate_selection_seconds"] = (
                time.perf_counter() - selection_started
            )
        req.stamps.update({
            "selected_D": depth,
            "candidate_order": order[:depth],
            "controller_reranker_wait_seconds": decision.reranker_wait,
            "controller_llm_wait_seconds": decision.llm_wait,
            "controller_reranker_pairs_per_second": decision.reranker_pairs_per_second,
        })
        self.pending_reranker_pairs += depth

    def select_final_k(
        self,
        req,
        selected_original_indices: list[int],
        reranker_scores: list[float],
    ) -> tuple[list[int], list[int]]:
        """Select K after reranking, using the method's own selection rule."""
        selection_started = time.perf_counter()
        if not self.spec.k_coverage_policy:
            local_order = sorted(
                range(len(reranker_scores)),
                key=reranker_scores.__getitem__,
                reverse=True,
            )[: self.a.top_k]
        else:
            original_indices = np.asarray(selected_original_indices, dtype=np.int64)
            relevance = normalize_scores(
                np.asarray(reranker_scores, dtype=np.float32), "minmax"
            )
            local_order = self.coverage_order(
                req,
                relevance,
                original_indices,
                self.a.top_k,
                self.spec.k_coverage_policy,
                "K",
            )
        req.stamps["k_selection_seconds"] = time.perf_counter() - selection_started
        final_indices = [selected_original_indices[index] for index in local_order]
        return local_order, final_indices

    async def reranker_worker(self):
        while True:
            req = await self.rr_queue.get()
            processed_pairs = 0
            try:
                if time.perf_counter() > req.deadline:
                    self.fail_timeout(req, "reranker")
                    continue
                req.stamps["reranker_batch_start"] = time.perf_counter()
                req.stamps["reranker_pair_batch_size"] = self.a.reranker_batch_size
                original_candidates = req.sample["candidates"][: self.a.doc_count]
                selected_original_indices = req.stamps.get(
                    "candidate_order", list(range(len(original_candidates)))
                )
                candidates = [original_candidates[index] for index in selected_original_indices]
                texts = [row["text"] for row in candidates]
                scores = []
                for offset in range(0, len(texts), self.a.reranker_batch_size):
                    batch_started = time.perf_counter()
                    args = (
                        req.sample["question"],
                        texts[offset : offset + self.a.reranker_batch_size],
                    )
                    if self.front_lock is None:
                        batch_scores = await asyncio.to_thread(self.rerank_batch_sync, *args)
                    else:
                        async with self.front_lock:
                            batch_scores = await asyncio.to_thread(self.rerank_batch_sync, *args)
                    scores.extend(batch_scores)
                    batch_elapsed = time.perf_counter() - batch_started
                    self.pending_reranker_pairs = max(
                        0, self.pending_reranker_pairs - len(batch_scores)
                    )
                    processed_pairs += len(batch_scores)
                    self.controller.reranker_throughput.update(
                        len(batch_scores) / max(batch_elapsed, 1e-9)
                    )
                   
                req.stamps["reranked_documents"] = len(scores)
                req.stamps["reranker_batch_end"] = time.perf_counter()
                ranked_local, final_indices = self.select_final_k(
                    req, selected_original_indices, scores
                )
                ranked = [texts[index] for index in ranked_local]
                req.stamps["final_k_indices"] = final_indices
                req.stamps["reranked_documents"] = len(scores)
                req.selected_contexts = ranked
                if self.spec.use_provence:
                    req.stamps["provence_queue_enter"] = time.perf_counter()
                    await self.prov_queue.put(req)
                else:
                    req.stamps["llm_queue_enter"] = time.perf_counter()
                    await self.llm_queue.put(req)
            except Exception as exc:
                self.fail(req, exc)
            finally:
                unprocessed = int(req.stamps.get("selected_D", 0)) - processed_pairs
                self.pending_reranker_pairs = max(
                    0, self.pending_reranker_pairs - max(unprocessed, 0)
                )
                self.rr_queue.task_done()

    def provenance_batch_sync(self, requests):
        with torch.inference_mode():
            result = self.prov_model.process(
                [r.sample["question"] for r in requests],
                [r.selected_contexts for r in requests],
                threshold=self.a.threshold, batch_size=self.a.provence_batch_size,
                always_select_title=True, enable_warnings=False, reorder=False,
            )
        torch.cuda.synchronize(torch.device(self.a.provence_device))
        return [x if isinstance(x, str) else "\n".join(x) for x in result["pruned_context"]]

    async def collect_batch(self, queue, first, max_size):
        batch = [first]
        deadline = time.perf_counter() + self.a.max_batch_wait_ms / 1000
        while len(batch) < max_size:
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        return batch

    async def provence_worker(self):
        while True:
            first = await self.prov_queue.get()
            batch = await self.collect_batch(self.prov_queue, first, self.a.provence_batch_size)
            active = [r for r in batch if time.perf_counter() <= r.deadline]
            for req in batch:
                if req not in active:
                    self.fail_timeout(req, "provence")
            try:
                if active:
                    stamp = time.perf_counter()
                    for req in active:
                        req.stamps["provence_batch_start"] = stamp
                        req.stamps["provence_batch_size"] = len(active)
                    if self.front_lock is None:
                        outputs = await asyncio.to_thread(self.provenance_batch_sync, active)
                    else:
                        async with self.front_lock:
                            outputs = await asyncio.to_thread(self.provenance_batch_sync, active)
                    stamp = time.perf_counter()
                    for req, output in zip(active, outputs):
                        req.stamps["provence_batch_end"] = stamp
                        req.compressed_context = output
                        req.stamps["context_tokens_before_provence"] = self.count_llm_tokens(
                            "\n\n".join(req.selected_contexts)
                        )
                        req.stamps["context_tokens_after_provence"] = self.count_llm_tokens(output)
                        req.stamps["llm_queue_enter"] = time.perf_counter()
                        await self.llm_queue.put(req)
            except Exception as exc:
                for req in active:
                    self.fail(req, exc)
            finally:
                for _ in batch:
                    self.prov_queue.task_done()

    def make_prompt(self, req):
        context = (
            req.compressed_context
            if self.spec.use_compressed_context
            else "\n\n".join(req.selected_contexts)
        )
        if self.a.natural_eos:
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question briefly using only the provided evidence.",
                },
                {
                    "role": "user",
                    "content": f"Evidence:\n{context}\n\nQuestion: {req.sample['question']}",
                },
            ]
            return self.llm_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        return f"Use the evidence to answer briefly.\nEvidence:\n{context}\nQuestion: {req.sample['question']}\nAnswer:"

    def count_llm_tokens(self, text):
        with self.llm_tokenizer_lock:
            return len(self.llm_tok.encode(text, add_special_tokens=False))

    def llm_batch_sync(self, requests):
        with self.llm_tokenizer_lock:
            prompts = [self.make_prompt(r) for r in requests]
            encoded = self.llm_tok(
                prompts, padding=True, return_tensors="pt"
            ).to(self.a.llm_device)
        input_tokens = encoded.attention_mask.sum(dim=1).tolist()
        generation_args = {"max_new_tokens": self.a.output_tokens}
        stop_ids = [self.llm_tok.eos_token_id]
        if self.a.natural_eos:
            endoftext_id = self.llm_tok.convert_tokens_to_ids("<|endoftext|>")
            if endoftext_id is not None and endoftext_id not in stop_ids:
                stop_ids.append(endoftext_id)
            generation_args["eos_token_id"] = stop_ids
        else:
            generation_args["min_new_tokens"] = self.a.output_tokens
            generation_args["eos_token_id"] = self.llm_tok.eos_token_id
        with torch.inference_mode():
            output_ids = self.llm_model.generate(
                **encoded, **generation_args,
                do_sample=False, pad_token_id=self.llm_tok.pad_token_id,
            )
        torch.cuda.synchronize(torch.device(self.a.llm_device))
        generated = output_ids[:, encoded.input_ids.shape[1]:]
        results = []
        for input_count, token_row in zip(input_tokens, generated):
            ids = token_row.tolist()
            stop_position = next(
                (position for position, token_id in enumerate(ids) if token_id in stop_ids),
                None,
            )
            if stop_position is None:
                answer_ids = ids
                stop_reason = "max_tokens"
            else:
                answer_ids = ids[:stop_position]
                stop_reason = "eos"
            answer = self.llm_tok.decode(answer_ids, skip_special_tokens=True).strip()
            results.append((int(input_count), len(answer_ids), answer, stop_reason))
        return results

    async def llm_worker(self):
        while True:
            first = await self.llm_queue.get()
            batch = await self.collect_batch(self.llm_queue, first, self.a.llm_batch_size)
            active = [r for r in batch if time.perf_counter() <= r.deadline]
            for req in batch:
                if req not in active:
                    self.fail_timeout(req, "llm")
            try:
                if active:
                    stamp = time.perf_counter()
                    self.llm_active_started = stamp
                    self.llm_active_expected_seconds = self.controller.llm_batch_seconds.value
                    for req in active:
                        req.stamps["llm_batch_start"] = stamp
                        req.stamps["llm_batch_size"] = len(active)
                    token_counts = await asyncio.to_thread(self.llm_batch_sync, active)
                    stamp = time.perf_counter()
                    observed_batch_seconds = stamp - self.llm_active_started
                    self.controller.llm_batch_seconds.update(observed_batch_seconds)
                    self.llm_active_started = None
                    for req, (input_tokens, output_tokens, output_text, stop_reason) in zip(active, token_counts):
                        req.stamps["llm_batch_end"] = stamp
                        req.stamps["completion_time"] = stamp
                        req.stamps["llm_input_tokens"] = input_tokens
                        req.stamps["llm_output_tokens"] = output_tokens
                        req.stamps["llm_output_text"] = output_text
                        req.stamps["llm_stop_reason"] = stop_reason
                        if stamp > req.deadline:
                            req.stamps["timeout_stage"] = "completion_deadline"
                        if not req.future.done():
                            req.future.set_result(req)
            except Exception as exc:
                self.llm_active_started = None
                for req in active:
                    self.fail(req, exc)
            finally:
                for _ in batch:
                    self.llm_queue.task_done()

    def fail_timeout(self, req, stage):
        req.stamps["timeout_stage"] = stage
        req.stamps["completion_time"] = time.perf_counter()
        if not req.future.done():
            req.future.set_result(req)

    def fail(self, req, exc):
        req.stamps["error"] = repr(exc)
        req.stamps["completion_time"] = time.perf_counter()
        if not req.future.done():
            req.future.set_result(req)


def request_row(req, method, qps, repetition):
    s = req.stamps
    def delta(end, start):
        return s[end] - s[start] if end in s and start in s else None
    row = {
        "request_id": req.request_id, "method": method, "offered_qps": qps,
        "repetition": repetition, "measured": req.measured, "arrival_time": req.arrival,
        "scheduled_arrival_offset": req.scheduled_offset,
        "actual_enqueue_offset": req.actual_enqueue_offset,
        "arrival_lag_seconds": req.actual_enqueue_offset - req.scheduled_offset,
        "workload_id": req.workload_id, "query_id": req.query_id,
        **s,
        "reranker_queue_seconds": delta("reranker_batch_start", "reranker_queue_enter"),
        "reranker_service_seconds": delta("reranker_batch_end", "reranker_batch_start"),
        "provence_queue_seconds": delta("provence_batch_start", "provence_queue_enter"),
        "provence_service_seconds": delta("provence_batch_end", "provence_batch_start"),
        "llm_queue_seconds": delta("llm_batch_start", "llm_queue_enter"),
        "llm_service_seconds": delta("llm_batch_end", "llm_batch_start"),
        "end_to_end_seconds": s.get("completion_time", time.perf_counter()) - req.arrival,
        "timed_out": "timeout_stage" in s,
        "failed": "error" in s,
    }
    if req.sample.get("facts"):
        d_metrics = evidence_scores(req.sample, s.get("candidate_order", []))
        k_metrics = evidence_scores(req.sample, s.get("final_k_indices", []))
        row.update({f"D_{key}": value for key, value in d_metrics.items()})
        row.update({f"K_{key}": value for key, value in k_metrics.items()})
    return row


def append_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        fields = sorted(set(fields).union(*(row.keys() for row in existing)))
    if existing and "request_id" in fields:
        merged = {row["request_id"]: row for row in existing}
        merged.update({row["request_id"]: row for row in rows})
        existing, rows = [], list(merged.values())
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(existing); writer.writerows(rows)
    temp.replace(path)


def workload_path(a, qps, repetition):
    qps_name = str(qps).replace(".", "p")
    return a.workload_dir / f"qps_{qps_name}_rep_{repetition}.json"


def load_or_create_workload(a, qps, repetition, samples):
    path = workload_path(a, qps, repetition)
    path.parent.mkdir(parents=True, exist_ok=True)
    query_ids = [sample["q_id"] for sample in samples]
    seed = a.seed + repetition * 1000 + int(qps * 100)
    expected = {
        "version": 2, "qps": qps, "repetition": repetition, "seed": seed,
        "warmup_seconds": a.warmup_seconds,
        "measurement_mode": (
            "one_unique_pass" if a.unique_measurement_pass else "fixed_duration"
        ),
        "measurement_seconds": (
            None if a.unique_measurement_pass else a.measurement_seconds
        ),
        "num_queries": len(samples), "query_ids": query_ids,
    }
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"Workload mismatch in {path}: {key}")
    else:
        rng = random.Random(seed)
        arrivals, offset, counter = [], 0.0, 0
        while offset < a.warmup_seconds:
            arrivals.append({"offset": offset, "query_index": counter % len(samples)})
            counter += 1
            offset += rng.expovariate(qps)
        if a.unique_measurement_pass:
            order = list(range(len(samples)))
            rng.shuffle(order)
            offset = a.warmup_seconds
            for query_index in order:
                arrivals.append({"offset": offset, "query_index": query_index})
                offset += rng.expovariate(qps)
        else:
            duration = a.warmup_seconds + a.measurement_seconds
            while offset < duration:
                arrivals.append({"offset": offset, "query_index": counter % len(samples)})
                counter += 1
                offset += rng.expovariate(qps)
        payload = {**expected, "arrivals": arrivals}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["workload_id"] = hashlib.sha256(canonical.encode()).hexdigest()
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    canonical_payload = {k: v for k, v in payload.items() if k != "workload_id"}
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(canonical.encode()).hexdigest()
    if payload.get("workload_id") != actual_hash:
        raise RuntimeError(f"Invalid workload hash: {path}")
    return payload


async def run_point(a, method, qps, repetition, samples, models, workload, selection_cache):
    pipeline = ServingPipeline(a, method, models, selection_cache)
    await pipeline.start()
    loop = asyncio.get_running_loop()
    if a.prime_pipeline:
        now = time.perf_counter()
        prime = Request(
            request_id=f"prime-D{a.doc_count}-{method}", sample=samples[0], measured=False,
            arrival=now, scheduled_offset=0.0, actual_enqueue_offset=0.0,
            workload_id="pipeline-prime", query_id=samples[0]["q_id"],
            deadline=math.inf, future=loop.create_future(),
        )
        pipeline.prepare_queue_aware_request(prime)
        prime.stamps["reranker_queue_enter"] = now
        await pipeline.rr_queue.put(prime)
        await prime.future
    started = time.perf_counter()
    requests = []
    for counter, item in enumerate(workload["arrivals"]):
        scheduled_offset = float(item["offset"])
        scheduled_arrival = started + scheduled_offset
        await asyncio.sleep(max(0, scheduled_arrival - time.perf_counter()))
        now = time.perf_counter()
        query_index = int(item["query_index"])
        sample = samples[query_index]
        future = loop.create_future()
        req = Request(
            request_id=f"{workload['workload_id'][:12]}-D{a.doc_count}-{method}-{counter}",
            sample=sample,
            measured=a.warmup_seconds <= scheduled_offset,
            arrival=now, scheduled_offset=scheduled_offset,
            actual_enqueue_offset=now - started, workload_id=workload["workload_id"],
            query_id=sample["q_id"],
            deadline=(math.inf if a.no_request_deadline else now + a.request_timeout_seconds),
            future=future,
        )
        requests.append(req)
        pipeline.prepare_queue_aware_request(req)
        req.stamps["reranker_queue_enter"] = now
        await pipeline.rr_queue.put(req)
    await asyncio.gather(*(req.future for req in requests))
    await pipeline.stop()
    return [request_row(req, method, qps, repetition) for req in requests]


def load_models(a):
    dtype = torch.float16
    rr_tok = AutoTokenizer.from_pretrained(a.reranker_model, local_files_only=a.local_files_only)
    rr_model = AutoModelForSequenceClassification.from_pretrained(
        a.reranker_model, torch_dtype=dtype, local_files_only=a.local_files_only,
        low_cpu_mem_usage=True,
    ).eval().to(a.frontend_device)
    prov = AutoModel.from_pretrained(
        a.provence_model, torch_dtype=dtype, local_files_only=a.local_files_only,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval().to(a.provence_device)
    llm_tok = AutoTokenizer.from_pretrained(a.generator_model, local_files_only=a.local_files_only)
    llm_tok.pad_token = llm_tok.eos_token; llm_tok.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(
        a.generator_model, torch_dtype=dtype, local_files_only=a.local_files_only,
        low_cpu_mem_usage=True,
    ).eval().to(a.llm_device)
    return rr_tok, rr_model, prov, llm_tok, llm


def load_hotpot_cohort(
    source: Path,
    cache_dir: Path,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    """Load HotpotQA candidates and all selection features from unified caches."""
    coverage_cache = cache_dir / "coverage_features"
    similarity_cache = cache_dir / "splade_similarity"
    samples: list[dict[str, Any]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}
    for sample_path in sorted((source / "batches").glob("*/samples_sentence_labels.json")):
        batch_name = sample_path.parent.name
        batch_samples = json.loads(sample_path.read_text())["samples"]
        for index, sample in enumerate(batch_samples):
            if not has_complete_evidence(sample):
                continue
            candidates = sample["candidates"][:candidate_count]
            if len(candidates) != candidate_count:
                raise RuntimeError(f"{sample['q_id']} has only {len(candidates)} candidates")
            for candidate in candidates:
                candidate.setdefault("score", candidate["dense_score"])
            sample = {**sample, "candidates": candidates}
            feature_path = coverage_cache / batch_name / f"{index:03d}.npz"
            if not feature_path.exists():
                raise FileNotFoundError(feature_path)
            with np.load(feature_path) as values:
                cache[sample["q_id"]] = {
                    "query_weights": values["query_weights"].astype(np.float32),
                    "document_features": values["document_features"].astype(np.float32),
                }
            similarity_path = similarity_cache / batch_name / f"{index:03d}.npz"
            if not similarity_path.exists():
                raise FileNotFoundError(similarity_path)
            with np.load(similarity_path) as values:
                document_similarity = values["document_similarity"].astype(np.float32)
                cache[sample["q_id"]].update(
                    query_similarity=values["query_similarity"].astype(np.float32),
                    similarities=document_similarity,
                    literature_similarities=document_similarity,
                )
            samples.append(sample)
    return samples, cache


def _busy_capacity(rows, stage):
    """Return query capacity from unique online batches, excluding queue time."""
    batches = {}
    for row in rows:
        start, end = row.get(f"{stage}_batch_start"), row.get(f"{stage}_batch_end")
        if start is None or end is None:
            continue
        key = (float(start), float(end))
        if stage == "reranker":
            size = 1
        else:
            size = int(float(row.get(f"{stage}_batch_size") or 1))
        batches[key] = size
    busy = sum(end - start for start, end in batches)
    capacity = sum(batches.values()) / busy if busy > 0 else None
    occupancy = statistics.mean(batches.values()) if batches else None
    return busy, capacity, occupancy


def summarize(rows, warmup_seconds, measurement_seconds):
    measured = [r for r in rows if r["measured"]]
    completed = [r for r in measured if not r["timed_out"] and not r["failed"]]
    window_completed = [
        row for row in completed
        if float(row["actual_enqueue_offset"]) + float(row["end_to_end_seconds"])
        <= warmup_seconds + measurement_seconds
    ]
    result = {
        "num_measured": len(measured), "num_completed": len(completed),
        "num_failed": sum(bool(r["failed"]) for r in measured),
        "num_completed_within_window": len(window_completed),
        "achieved_qps": len(window_completed) / measurement_seconds,
        "throughput_ratio": (
            len(window_completed) / (float(measured[0]["offered_qps"]) * measurement_seconds)
            if measured else None
        ),
    }
    result["timeout_rate"] = sum(bool(r["timed_out"]) for r in measured) / len(measured) if measured else float("nan")
    result["failure_rate"] = result["num_failed"] / len(measured) if measured else float("nan")
    lags = [float(r["arrival_lag_seconds"]) for r in measured]
    result["p95_arrival_lag_seconds"] = float(np.percentile(lags, 95)) if lags else None
    for key in ["end_to_end_seconds", "reranker_queue_seconds", "provence_queue_seconds", "llm_queue_seconds"]:
        vals = [float(r[key]) for r in completed if r.get(key) is not None]
        result[f"mean_{key}"] = statistics.mean(vals) if vals else None
        result[f"p95_{key}"] = float(np.percentile(vals, 95)) if vals else None
    for key in [
        "selected_D", "reranked_documents", "llm_input_tokens",
        "llm_output_tokens", "context_tokens_before_provence",
        "context_tokens_after_provence", "provence_batch_size", "llm_batch_size",
        "controller_reranker_wait_seconds", "controller_llm_wait_seconds",
        "candidate_selection_seconds", "k_selection_seconds",
        "effective_anchors", "d_effective_anchors", "k_effective_anchors",
        "D_complete_evidence_recall", "D_supporting_fact_recall",
        "K_complete_evidence_recall", "K_supporting_fact_recall",
    ]:
        vals = [float(r[key]) for r in completed if r.get(key) is not None]
        result[f"mean_{key}"] = statistics.mean(vals) if vals else None
    for stage in ("reranker", "provence", "llm"):
        busy, capacity, occupancy = _busy_capacity(completed, stage)
        result[f"{stage}_busy_seconds"] = busy
        result[f"{stage}_busy_capacity_qps"] = capacity
        result[f"{stage}_mean_batch_occupancy"] = occupancy
    rr_capacity = result["reranker_busy_capacity_qps"]
    llm_capacity = result["llm_busy_capacity_qps"]
    result["llm_to_reranker_capacity_ratio"] = (
        llm_capacity / rr_capacity if rr_capacity and llm_capacity else None
    )
    result["eos_rate"] = (
        sum(row.get("llm_stop_reason") == "eos" for row in completed) / len(completed)
        if completed else None
    )
    return result


async def main_async(a):
    a.output_dir.mkdir(parents=True, exist_ok=True)
    effective_batch_sizes = {
        "reranker": a.reranker_batch_size,
        "llm": a.llm_batch_size,
        "provence": a.provence_batch_size,
    }
    if effective_batch_sizes != {"reranker": 8, "llm": 10, "provence": 4}:
        raise ValueError(
            "Required serving batch sizes are reranker=8, llm=10, provence=4; "
            f"got {effective_batch_sizes}"
        )
    if len({a.frontend_device, a.llm_device, a.provence_device}) != 3:
        raise ValueError("Reranker, LLM, and Provence must use three distinct devices")
    if a.candidate_pool_size < max(a.doc_count, a.top_k):
        raise ValueError("candidate-pool-size must cover doc-count and top-k")
    samples, selection_cache = load_hotpot_cohort(
        a.source,
        a.cache_dir,
        a.candidate_pool_size,
    )
    calibration = load_calibration_manifest(a.calibration_manifest)
    if calibration.dataset != "hotpot":
        raise ValueError("calibration manifest must be for HotpotQA")
    if len(calibration.calibration_query_ids) != 100:
        raise ValueError("the HotpotQA calibration split must contain 100 queries")
    excluded_ids = set(calibration.calibration_query_ids)
    samples = [sample for sample in samples if sample["q_id"] not in excluded_ids]
    if len(samples) != 1087:
        raise ValueError(f"expected 1087 evaluation queries; got {len(samples)}")
    candidate_payload = [
        {"query_id": sample["q_id"], "doc_ids": [c["doc_id"] for c in sample["candidates"]]}
        for sample in samples
    ]
    candidate_hash = hashlib.sha256(
        json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    qps_values = [float(x) for x in a.qps_values.split(",")]
    methods = a.methods.split(",")
    unsupported = set(methods) - set(METHOD_SPECS)
    if unsupported:
        raise ValueError(f"Unsupported methods: {sorted(unsupported)}")
    literature_methods = [
        method for method in methods if METHOD_SPECS[method].literature_policy
    ]
    a.baseline_parameters = {}
    if literature_methods:
        for stage, parameters in calibration.parameters_by_stage.items():
            a.baseline_parameters.update({
                f"{stage}_rocchio_alpha": parameters.rocchio_alpha,
                f"{stage}_rocchio_depth": parameters.rocchio_depth,
                f"{stage}_mmr": parameters.mmr_diversity,
                f"{stage}_dartboard": parameters.dartboard_sigma,
            })
    coverage_methods = [
        method for method in methods
        if METHOD_SPECS[method].d_coverage_policy
        or METHOD_SPECS[method].k_coverage_policy
    ]
    if coverage_methods and not selection_cache:
        raise ValueError("coverage methods require cached SPLADE features")
    anchor_methods = [
        method for method in methods
        if METHOD_SPECS[method].d_coverage_policy in {"soft_anchor", "anchor_only"}
        or METHOD_SPECS[method].k_coverage_policy in {"soft_anchor", "anchor_only"}
    ]
    if anchor_methods:
        if any("similarities" not in values for values in selection_cache.values()):
            raise ValueError("soft-anchor similarity cache is incomplete")
    if not a.top_k <= a.dynamic_d_min <= a.doc_count:
        raise ValueError("require top-k <= dynamic-d-min <= doc-count")
    models = load_models(a)
    summary_path = a.output_dir / "summary.jsonl"
    existing = []
    if summary_path.exists():
        existing = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    completed_keys = {(r["method"], float(r["offered_qps"]), int(r["repetition"])) for r in existing}
    manifest = {
        "num_queries": len(samples),
        "doc_count": a.doc_count, "candidate_pool_size": a.candidate_pool_size,
        "qps_values": qps_values, "repetitions": a.repetitions,
        "repetition_offset": a.repetition_offset, "seed": a.seed,
        "warmup_seconds": a.warmup_seconds, "measurement_seconds": a.measurement_seconds,
        "request_deadline_enabled": not a.no_request_deadline,
        "pipeline_primed": a.prime_pipeline,
        "natural_eos": a.natural_eos, "max_output_tokens": a.output_tokens,
        "dynamic_d_min": a.dynamic_d_min,
        "dynamic_d_max": a.doc_count,
        "controller": "queue_aware_depth_v1",
        "controller_ewma_decay": a.controller_ewma_decay,
        "k_selection_policy": {
            "fixed_d100": "reranker_top_k",
            "queue_adaptive_dense": "reranker_top_k",
            "queue_adaptive_sqrt": "reranker_sqrt_quality_coverage",
            "queue_adaptive_soft_anchor": (
                "reranker_soft_anchor_noisy_or_sqrt_quality_coverage"
            ),
            "queue_adaptive_anchor_only": (
                "reranker_anchor_only_sqrt_quality_coverage"
            ),
            "queue_adaptive_coverage_only_d": "reranker_top_k",
            "queue_adaptive_prf": "reranker_top_k",
            "queue_adaptive_mmr": "reranker_top_k",
            "queue_adaptive_dartboard": "reranker_top_k",
        },
        "source": str(a.source),
        "cache_dir": str(a.cache_dir),
        "calibration_manifest": str(a.calibration_manifest),
        "excluded_calibration_queries": sorted(excluded_ids),
        "baseline_parameters": a.baseline_parameters,
        "methods": methods,
        "reranker_batch_size": a.reranker_batch_size,
        "provence_batch_size": a.provence_batch_size, "llm_batch_size": a.llm_batch_size,
        "frontend_device": a.frontend_device, "llm_device": a.llm_device,
        "provence_device": a.provence_device,
        "candidate_pool_hash": candidate_hash,
    }
    (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for qps_index, qps in enumerate(qps_values):
        for repetition in range(a.repetition_offset, a.repetition_offset + a.repetitions):
            workload = load_or_create_workload(a, qps, repetition, samples)
            run_methods = methods if (qps_index + repetition + a.order_offset) % 2 == 0 else list(reversed(methods))
            for method in run_methods:
                key = (method, qps, repetition)
                if key in completed_keys:
                    print(f"SKIP completed method={method} qps={qps} repetition={repetition}", flush=True)
                    continue
                print(f"START method={method} qps={qps} repetition={repetition}", flush=True)
                rows = await run_point(
                    a, method, qps, repetition, samples, models, workload, selection_cache
                )
                append_csv(a.output_dir / "request_traces.csv", rows)
                summary = {
                    "method": method, "offered_qps": qps, "repetition": repetition,
                    "doc_count": a.doc_count, "workload_id": workload["workload_id"],
                    **summarize(rows, a.warmup_seconds, a.measurement_seconds),
                }
                with summary_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(summary) + "\n")
                print(f"DONE {summary}", flush=True)


def main():
    a = parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
