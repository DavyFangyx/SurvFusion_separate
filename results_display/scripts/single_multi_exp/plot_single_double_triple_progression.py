"""
Plot single/bimodal/trimodal progression curves for multimodal experiments.

This script aligns results from:
    results_display/MSingle_Multi_Test/
    results_display/Multi_model_test/
    results_display/Multi_model_test2/

Outputs:
    results_display/Multi_exp_sum/
        progression_points.csv
        figures/{dataset}__{tri_model}.png
        figures/{dataset}__{tri_model}.pdf

Figure layout:
    9 figures = 3 datasets x 3 trimodal models
    Each figure contains 3 subplots for target modalities: C / G / P(WSI)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_deployment.registry import get_dataset_config, list_enabled_studies

DATASETS = list_enabled_studies()
TRI_MODELS = [
    "survtri_snn_concat",
    "survtri_mlp_concat",
    "survtri_mlp_mhsa",
]
TARGET_MODALITIES = ["C", "G", "P"]
MODALITY_ORDER = ["C", "G", "P"]

DATASET_LABELS = {
    study: get_dataset_config(study).display_name.replace("TCGA-", "").replace("TCGA_", "")
    for study in DATASETS
}

TRI_MODEL_LABELS = {
    "survtri_snn_concat": "SNN Concat",
    "survtri_mlp_concat": "MLP Concat",
    "survtri_mlp_mhsa": "MLP MHSA",
}

TARGET_COLORS = {
    "C": "#3A7D44",
    "G": "#C1666B",
    "P": "#2E5EAA",
}

SINGLE_SPECS = {
    "C": {
        "csv": "results_display/MSingle_Multi_Test/Clinictest_Li/Clinictest_Li_combined.csv",
        "embedding": "L4",
        "models": {
            "survtri_snn_concat": "snn_clinic_flatten",
            "survtri_mlp_concat": "mlp_clinic_flatten",
            "survtri_mlp_mhsa": "mlp_clinic_flatten",
        },
    },
    "G": {
        "csv": "results_display/MSingle_Multi_Test/Gengtest_F/Gengtest_F_combined.csv",
        "embedding": "scFoundation_embedding_cell_norm",
        "models": {
            "survtri_snn_concat": "snn_gene_f",
            "survtri_mlp_concat": "mlp_gene_f",
            "survtri_mlp_mhsa": "mlp_gene_f",
        },
    },
    "P": {
        "csv": "results_display/MSingle_Multi_Test/WSItest_F/WSItest_F_combined.csv",
        "embedding": "uni_v2",
        "models": {
            "survtri_snn_concat": "mlp_wsi",
            "survtri_mlp_concat": "mlp_wsi",
            "survtri_mlp_mhsa": "mlp_wsi",
        },
    },
}

PAIR_NAME_MAP = {
    frozenset({"C", "G"}): "gene_clinic",
    frozenset({"C", "P"}): "wsi_clinic",
    frozenset({"G", "P"}): "wsi_gene",
}

PAIR_STAGE_LABELS = {
    "gene_clinic": "C+G",
    "wsi_clinic": "C+P",
    "wsi_gene": "G+P",
}


@dataclass
class PointRecord:
    dataset: str
    tri_model: str
    target_modality: str
    stage_index: int
    stage_key: str
    stage_label: str
    source_level: str
    source_model: str
    source_condition: str
    metric_mean: float
    metric_std: float


def ensure_output_dirs(project_root: Path, out_dir: Path | None) -> tuple[Path, Path]:
    base_dir = out_dir or (project_root / "results_display" / "Multi_exp_sum")
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, fig_dir


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def pick_single_metric(
    df: pd.DataFrame,
    dataset: str,
    embedding: str,
    model: str,
    metric: str,
) -> tuple[float, float]:
    hit = df[
        (df["dataset"] == dataset)
        & (df["embedding"] == embedding)
        & (df["model"] == model)
    ]
    if hit.empty:
        raise KeyError(f"Single result missing: {dataset} / {embedding} / {model}")
    row = hit.iloc[0]
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def pick_bimodal_metric(
    df: pd.DataFrame,
    dataset: str,
    pair_name: str,
    tri_model: str,
    metric: str,
) -> tuple[float, float]:
    hit = df[
        (df["study"] == dataset)
        & (df["modality_pair"] == pair_name)
        & (df["model"] == tri_model)
    ]
    if hit.empty:
        raise KeyError(f"Bimodal result missing: {dataset} / {pair_name} / {tri_model}")
    row = hit.iloc[0]
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def pick_trimodal_metric(
    df: pd.DataFrame,
    dataset: str,
    tri_model: str,
    tri_heads: int,
    metric: str,
) -> tuple[float, float]:
    hit = df[
        (df["study"] == dataset)
        & (df["clinic"] == "L4")
        & (df["gene"] == "gene_norm")
        & (df["wsi"] == "uni_v2")
        & (df["heads"] == tri_heads)
        & (df["model"] == tri_model)
    ]
    if hit.empty:
        raise KeyError(
            f"Trimodal result missing: {dataset} / heads={tri_heads} / {tri_model}"
        )
    row = hit.iloc[0]
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def single_stage_label(modality: str) -> str:
    return modality


def pair_stage_label(target: str, other: str) -> str:
    return f"{target}+{other}"


def tri_stage_label(target: str) -> str:
    others = other_modalities(target)
    return "+".join([target] + others)


def pair_name(a: str, b: str) -> str:
    return PAIR_NAME_MAP[frozenset({a, b})]


def other_modalities(target: str) -> list[str]:
    return [m for m in MODALITY_ORDER if m != target]


def collect_points(project_root: Path, metric: str, tri_heads: int) -> pd.DataFrame:
    single_frames = {
        modality: load_csv(project_root / spec["csv"])
        for modality, spec in SINGLE_SPECS.items()
    }
    bimodal_df = load_csv(
        project_root / "results_display" / "Multi_model_test2" / "Multi_model_test2_combined.csv"
    )
    trimodal_df = load_csv(
        project_root / "results_display" / "Multi_model_test" / "Multi_model_test_combined.csv"
    )

    rows: list[PointRecord] = []
    for dataset in DATASETS:
        for tri_model in TRI_MODELS:
            tri_mean, tri_std = pick_trimodal_metric(
                trimodal_df,
                dataset=dataset,
                tri_model=tri_model,
                tri_heads=tri_heads,
                metric=metric,
            )

            for target in TARGET_MODALITIES:
                others = other_modalities(target)
                target_single_model = SINGLE_SPECS[target]["models"][tri_model]
                target_single_mean, target_single_std = pick_single_metric(
                    single_frames[target],
                    dataset=dataset,
                    embedding=SINGLE_SPECS[target]["embedding"],
                    model=target_single_model,
                    metric=metric,
                )

                for stage_index, modality in enumerate(others):
                    single_model = SINGLE_SPECS[modality]["models"][tri_model]
                    single_mean, single_std = pick_single_metric(
                        single_frames[modality],
                        dataset=dataset,
                        embedding=SINGLE_SPECS[modality]["embedding"],
                        model=single_model,
                        metric=metric,
                    )
                    rows.append(
                        PointRecord(
                            dataset=dataset,
                            tri_model=tri_model,
                            target_modality=target,
                            stage_index=stage_index,
                            stage_key=f"single_{modality}",
                            stage_label=single_stage_label(modality),
                            source_level="single",
                            source_model=single_model,
                            source_condition=SINGLE_SPECS[modality]["embedding"],
                            metric_mean=single_mean,
                            metric_std=single_std,
                        )
                    )

                rows.append(
                    PointRecord(
                        dataset=dataset,
                        tri_model=tri_model,
                        target_modality=target,
                        stage_index=2,
                        stage_key=f"single_{target}",
                        stage_label=single_stage_label(target),
                        source_level="single",
                        source_model=target_single_model,
                        source_condition=SINGLE_SPECS[target]["embedding"],
                        metric_mean=target_single_mean,
                        metric_std=target_single_std,
                    )
                )

                for offset, modality in enumerate(others, start=3):
                    combo_name = pair_name(target, modality)
                    pair_mean, pair_std = pick_bimodal_metric(
                        bimodal_df,
                        dataset=dataset,
                        pair_name=combo_name,
                        tri_model=tri_model,
                        metric=metric,
                    )
                    rows.append(
                        PointRecord(
                            dataset=dataset,
                            tri_model=tri_model,
                            target_modality=target,
                            stage_index=offset,
                            stage_key=f"bimodal_{combo_name}",
                            stage_label=pair_stage_label(target, modality),
                            source_level="bimodal",
                            source_model=tri_model,
                            source_condition=combo_name,
                            metric_mean=pair_mean,
                            metric_std=pair_std,
                        )
                    )

                rows.append(
                    PointRecord(
                        dataset=dataset,
                        tri_model=tri_model,
                        target_modality=target,
                        stage_index=5,
                        stage_key="trimodal",
                        stage_label=tri_stage_label(target),
                        source_level="trimodal",
                        source_model=tri_model,
                        source_condition=f"L4__gene_norm__uni_v2__h{tri_heads:02d}",
                        metric_mean=tri_mean,
                        metric_std=tri_std,
                    )
                )

    df = pd.DataFrame([row.__dict__ for row in rows]).sort_values(
        by=["dataset", "tri_model", "target_modality", "stage_index"],
        kind="stable",
    )
    return df


def prettify_target_label(target: str) -> str:
    return {
        "C": "Clinical",
        "G": "Gene",
        "P": "WSI",
    }[target]


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_model_figure(
    subset: pd.DataFrame,
    dataset: str,
    tri_model: str,
    metric_label: str,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    fig.patch.set_facecolor("#FBFAF7")

    y_min = max(0.0, float(subset["metric_mean"].min()) - 0.08)
    y_max = min(1.0, float(subset["metric_mean"].max()) + 0.08)

    for ax, target in zip(axes, TARGET_MODALITIES):
        target_df = subset[subset["target_modality"] == target].sort_values("stage_index")
        x = target_df["stage_index"].to_numpy(dtype=float)
        y = target_df["metric_mean"].to_numpy(dtype=float)
        err = target_df["metric_std"].to_numpy(dtype=float)
        labels = target_df["stage_label"].tolist()
        color = TARGET_COLORS[target]

        ax.set_facecolor("#FFFDF8")
        ax.grid(axis="y", color="#D9D5CB", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=6.5,
            capsize=4,
            elinewidth=1.4,
            markerfacecolor="#FFFFFF",
            markeredgewidth=1.8,
            markeredgecolor=color,
        )
        for xi, yi in zip(x, y):
            ax.text(
                xi,
                yi + 0.02,
                f"{yi:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#2B2B2B",
            )

        ax.set_title(
            f"{prettify_target_label(target)} progression",
            fontsize=12.5,
            pad=10,
            color=color,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=11.5, fontweight="bold")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Stage", fontsize=10)

    axes[0].set_ylabel(metric_label, fontsize=10.5)
    fig.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)} | {TRI_MODEL_LABELS.get(tri_model, tri_model)}",
        fontsize=16,
        x=0.06,
        ha="left",
        y=1.04,
    )
    fig.text(
        0.06,
        0.97,
        "Order: other singles -> target single -> target-related bimodals -> trimodal",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, fig_dir / f"{dataset}__{tri_model}")


def plot_all(points_df: pd.DataFrame, metric: str, fig_dir: Path) -> None:
    metric_label = metric.replace("_mean", "").replace("_", " ")
    for dataset in DATASETS:
        for tri_model in TRI_MODELS:
            subset = points_df[
                (points_df["dataset"] == dataset)
                & (points_df["tri_model"] == tri_model)
            ]
            if subset.empty:
                continue
            plot_dataset_model_figure(
                subset=subset,
                dataset=dataset,
                tri_model=tri_model,
                metric_label=metric_label,
                fig_dir=fig_dir,
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Plot aligned single/bimodal/trimodal progression curves"
    )
    parser.add_argument(
        "--metric",
        default="test_cindex",
        help="Metric prefix to plot, e.g. test_cindex or test_iauc",
    )
    parser.add_argument(
        "--tri_heads",
        type=int,
        default=8,
        help="Heads value used to pick trimodal results from Multi_model_test",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display" / "Multi_exp_sum",
        help="Directory for exported figures and CSV tables",
    )
    args = parser.parse_args()

    base_dir, fig_dir = ensure_output_dirs(project_root, args.output_dir)
    points_df = collect_points(project_root, metric=args.metric, tri_heads=args.tri_heads)
    points_path = base_dir / "progression_points.csv"
    points_df.to_csv(points_path, index=False)
    plot_all(points_df, metric=args.metric, fig_dir=fig_dir)

    print(f"Output directory: {base_dir}")
    print(f"Point table:       {points_path}")
    print(f"Figures:           {fig_dir}")
    print(f"Metric:            {args.metric}")
    print(f"Trimodal heads:    {args.tri_heads}")


if __name__ == "__main__":
    main()
