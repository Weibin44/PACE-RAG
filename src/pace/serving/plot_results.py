#!/usr/bin/env python3
"""Create paper-layout plots for the HotpotQA online-serving experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = (
    ("fixed_d100", "Fixed D=100", "#000000", 1.9),
    ("queue_adaptive_dense", "Dense", "#7F7F7F", 1.9),
    ("queue_adaptive_prf", "Rocchio PRF", "#0072B2", 1.9),
    ("queue_adaptive_mmr", "MMR", "#009E73", 1.9),
    ("queue_adaptive_dartboard", "Dartboard", "#E69F00", 1.9),
    ("queue_adaptive_coverage_only_d", "w/o query & anchor relevance", "#59A14F", 1.9),
    ("queue_adaptive_sqrt", "w/o anchor relevance", "#F2CF5B", 1.9),
    ("queue_adaptive_anchor_only", "w/o query relevance", "#A65628", 1.9),
    ("queue_adaptive_soft_anchor", "Ours (PACE)", "#D62728", 1.9),
)
RECALL_MARKERS = {
    "fixed_d100": None,
    "queue_adaptive_dense": "s",
    "queue_adaptive_prf": "v",
    "queue_adaptive_mmr": "<",
    "queue_adaptive_dartboard": ">",
    "queue_adaptive_coverage_only_d": "P",
    "queue_adaptive_sqrt": "^",
    "queue_adaptive_anchor_only": "D",
    "queue_adaptive_soft_anchor": "o",
}


def marker_positions(point_count: int) -> list[int]:
    """Return four approximately equally spaced marker positions."""
    if point_count <= 4:
        return list(range(point_count))
    return [
        index * point_count // 5
        for index in range(1, 5)
    ]


def rows_for(rows: list[dict], method: str) -> list[dict]:
    return sorted(
        (row for row in rows if row["method"] == method),
        key=lambda row: float(row["offered_qps"]),
    )


def plot_metric(axis, rows, field: str, methods=METHODS, show_markers=False) -> None:
    for method, label, color, linewidth in methods:
        points = rows_for(rows, method)
        if points and all(row.get(field) is not None for row in points):
            axis.plot(
                [float(row["offered_qps"]) for row in points],
                [float(row[field]) for row in points],
                color=color, linewidth=linewidth, linestyle="-", label=label,
                marker=RECALL_MARKERS[method] if show_markers else None,
                markersize=4.2, markevery=marker_positions(len(points)),
            )
    axis.set_xlabel("QPS")
    axis.grid(alpha=0.22, which="both")


def common_legend(fig, axis, columns: int = 5) -> None:
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=columns, frameon=False, fontsize=10.5,
        handlelength=2.5, markerscale=1.2, columnspacing=1.5,
        handletextpad=0.6,
    )


def latency_figure(rows: list[dict], output: Path, statistic: str = "mean") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 4.25))
    latency_methods = tuple(
        method for method in METHODS
        if method[0] in {"fixed_d100", "queue_adaptive_soft_anchor"}
    )
    if statistic == "p95":
        panels = (
            ("p95_end_to_end_seconds", "P95 end-to-end latency", True),
            ("p95_reranker_queue_seconds", "P95 reranker queue time", True),
            ("p95_llm_queue_seconds", "P95 LLM queue time", False),
        )
    else:
        panels = (
            ("mean_end_to_end_seconds", "Total latency", True),
            ("mean_reranker_queue_seconds", "Reranker queue time", True),
            ("mean_llm_queue_seconds", "LLM queue time", False),
        )
    for axis, (field, title, logarithmic) in zip(axes, panels):
        plot_metric(axis, rows, field, latency_methods, show_markers=True)
        axis.set_title(title)
        axis.set_ylabel("Seconds")
        if logarithmic:
            axis.set_yscale("log")
    common_legend(fig, axes[0], columns=2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def recall_figure(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.35), sharex=True)
    panels = (
        ("mean_D_complete_evidence_recall", "Complete Evidence Recall@D"),
        ("mean_D_supporting_fact_recall", "Supporting Evidence Recall@D"),
        ("mean_K_complete_evidence_recall", "Complete Evidence Recall@5"),
        ("mean_K_supporting_fact_recall", "Supporting Evidence Recall@5"),
    )
    for axis, (field, title) in zip(axes, panels):
        plot_metric(axis, rows, field, show_markers=True)
        axis.set_title(title)
        axis.set_ylabel("Recall")
        axis.margins(y=0.08)
    common_legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def selected_d_figure(rows: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    plot_metric(axis, rows, "mean_selected_D", show_markers=True)
    axis.set_title("Selected D")
    axis.set_ylabel("Number of documents")
    axis.set_ylim(45, 102)
    common_legend(fig, axis, columns=3)
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (args.input_dir / "summary.jsonl").read_text().splitlines()
        if line.strip()
    ]
    latency_figure(rows, output_dir / "hotpot_online_latency.pdf")
    latency_figure(
        rows, output_dir / "hotpot_online_latency_p95.pdf", statistic="p95"
    )
    recall_figure(rows, output_dir / "hotpot_online_recall.pdf")
    selected_d_figure(rows, output_dir / "hotpot_online_selected_D.pdf")


if __name__ == "__main__":
    main()
