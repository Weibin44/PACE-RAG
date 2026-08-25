"""Plot the final cross-dataset effectiveness figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = (
    ("hotpot", "HotpotQA"),
    ("musique", "MuSiQue"),
    ("2wiki", "2WikiMultiHopQA"),
)

COMPARISON_METHODS = (
    ("standard", "#000000", "Standard dense", "-", "s", 2.4),
    ("rocchio_prf", "#0072B2", "Rocchio PRF", "--", "v", 1.9),
    ("mmr", "#009E73", "MMR", "-.", "<", 1.9),
    ("dartboard", "#E69F00", "Dartboard", ":", ">", 2.2),
    ("ours", "#D62728", "Ours (PACE)", "-", "o", 3.2),
)

ABLATION_METHODS = (
    (
        "coverage_only",
        "w/o query & anchor relevance",
        "#7F7F7F",
        ":",
        "P",
        2.4,
    ),
    (
        "query_only",
        "w/o anchor relevance",
        "#E69F00",
        "--",
        "^",
        2.1,
    ),
    (
        "anchor_only",
        "w/o query relevance",
        "#A65628",
        "-.",
        "D",
        2.1,
    ),
    (
        "ours",
        "Ours (PACE)",
        "#D62728",
        "-",
        "o",
        3.2,
    ),
)

D_METRICS = (
    (
        "complete_evidence_recall",
        "Complete Evidence Recall",
        "complete_evidence_recall_vs_d.pdf",
    ),
    (
        "supporting_fact_recall",
        "Supporting Evidence Recall",
        "supporting_evidence_recall_vs_d.pdf",
    ),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        return list(csv.DictReader(handle))


def d_rows(
    curves: list[dict[str, str]],
    method: str,
) -> list[dict[str, str]]:
    return [
        row
        for row in curves
        if row["stage"] == "D"
        and row["method"] == method
    ]


def marker_positions(point_count: int) -> list[int]:
    """Return four approximately equally spaced marker positions."""
    if point_count <= 4:
        return list(range(point_count))
    return [
        index * point_count // 5
        for index in range(1, 5)
    ]


def configure_axis(
    axis,
    curves: list[dict[str, str]],
    dataset_label: str,
) -> None:
    maximum_d = max(
        int(row["cutoff"])
        for row in curves
        if row["stage"] == "D"
    )
    axis.set_title(dataset_label)
    axis.set_xlabel("D")
    axis.set_xlim(1, maximum_d)
    axis.set_ylim(0, 1.0)
    axis.grid(alpha=0.2)


def plot_comparison(
    input_root: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, filename in D_METRICS:
        figure, axes = plt.subplots(
            1,
            3,
            figsize=(15.2, 4.25),
            sharey=True,
        )

        for axis, (directory, label) in zip(
            axes,
            DATASETS,
        ):
            curves = read_csv(
                input_root / directory / "curves.csv"
            )
            for (
                method,
                color,
                method_label,
                linestyle,
                marker,
                linewidth,
            ) in COMPARISON_METHODS:
                rows = d_rows(curves, method)
                axis.plot(
                    [
                        int(row["cutoff"])
                        for row in rows
                    ],
                    [
                        float(row[metric])
                        for row in rows
                    ],
                    color=color,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    marker=marker,
                    markevery=marker_positions(len(rows)),
                    markersize=5.2,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=method_label,
                )

            adaptive_path = (
                input_root
                / directory
                / "adaptive_point.csv"
            )
            if adaptive_path.exists():
                adaptive = read_csv(adaptive_path)[0]
                axis.scatter(
                    float(adaptive["mean_cutoff"]),
                    float(adaptive[metric]),
                    color="#7B61A8",
                    marker="x",
                    s=58,
                    linewidth=2,
                    zorder=7,
                    label="Adaptive-K",
                )

            configure_axis(axis, curves, label)

        axes[0].set_ylabel(ylabel)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=6,
            frameon=False,
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        figure.savefig(
            output_dir / filename,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_ablation(
    input_root: Path,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 4.25),
        sharey=True,
    )

    for axis, (directory, label) in zip(
        axes,
        DATASETS,
    ):
        curves = read_csv(
            input_root / directory / "curves.csv"
        )

        for (
            method,
            method_label,
            color,
            linestyle,
            marker,
            linewidth,
        ) in ABLATION_METHODS:
            rows = d_rows(curves, method)
            axis.plot(
                [
                    int(row["cutoff"])
                    for row in rows
                ],
                [
                    float(
                        row[
                            "complete_evidence_recall"
                        ]
                    )
                    for row in rows
                ],
                label=method_label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                markevery=marker_positions(len(rows)),
                markersize=5.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=linewidth,
            )

        configure_axis(axis, curves, label)

    axes[0].set_ylabel("Complete Evidence Recall")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=2.5,
        columnspacing=1.7,
        handletextpad=0.6,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    plot_comparison(
        args.input_root,
        args.output_root / "cross_dataset_D",
    )
    plot_ablation(
        args.input_root,
        args.output_root
        / "cross_dataset_ablation"
        / "complete_evidence_recall_ablation.pdf",
    )


if __name__ == "__main__":
    main()
