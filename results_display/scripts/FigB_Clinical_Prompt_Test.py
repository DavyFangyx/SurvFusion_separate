"""
Plot FigB clinical prompt trends from results/FigB_Clinical Prompt Test.

Outputs:
    results_display/FigB_Clinical Prompt Test/summary/clinical_prompt_long.csv
    results_display/FigB_Clinical Prompt Test/summary/clinical_traditional_long.csv
    results_display/FigB_Clinical Prompt Test/figures/clinical_prompt_vs_traditional.png
    results_display/FigB_Clinical Prompt Test/figures/by_dataset/<study>.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataset_deployment.registry import get_dataset_config, list_enabled_studies
except Exception:
    get_dataset_config = None
    list_enabled_studies = None


RESULTS_SERIES = "FigB_Clinical Prompt Test"
METRIC = "test_cindex"
PROMPT_LEVELS = [f"L{i}" for i in range(6)]
TRAD_LEVELS = [f"D{i}" for i in range(6)]

MODEL_SPECS = {
    "mlp_clinic_mean": {
        "label": "MLP mean",
        "color": "#1b9e77",
        "marker": "o",
        "model_family": "MLP",
        "pooling": "mean",
    },
    "mlp_clinic_flatten": {
        "label": "MLP flatten",
        "color": "#d95f02",
        "marker": "s",
        "model_family": "MLP",
        "pooling": "flatten",
    },
    "snn_clinic_mean": {
        "label": "SNN mean",
        "color": "#7570b3",
        "marker": "^",
        "model_family": "SNN",
        "pooling": "mean",
    },
    "snn_clinic_flatten": {
        "label": "SNN flatten",
        "color": "#e7298a",
        "marker": "D",
        "model_family": "SNN",
        "pooling": "flatten",
    },
}


def dataset_label(study: str) -> str:
    if get_dataset_config is not None:
        try:
            label = get_dataset_config(study).display_name
            return label.replace("TCGA-", "").replace("TCGA_", "")
        except Exception:
            pass
    return study.replace("tcga_", "").upper()


def ordered_studies(values: list[str]) -> list[str]:
    if list_enabled_studies is None:
        return sorted(values)
    study_order = {study: idx for idx, study in enumerate(list_enabled_studies())}
    return sorted(values, key=lambda item: (study_order.get(item, 999), item))


def load_metric(csv_path: Path, metric: str) -> tuple[float, float, int] | None:
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[WARN] Cannot read {csv_path}: {exc}")
        return None

    if metric not in df.columns:
        print(f"[WARN] Missing {metric} in {csv_path}")
        return None

    values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        print(f"[WARN] Empty {metric} in {csv_path}")
        return None

    return float(np.mean(values)), float(np.std(values)), int(len(values))


def collect_long_table(results_root: Path) -> pd.DataFrame:
    series_dir = results_root / RESULTS_SERIES
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {series_dir}")

    rows: list[dict[str, object]] = []
    for run_dir in sorted(series_dir.iterdir()):
        if not run_dir.is_dir() or "__" not in run_dir.name:
            continue

        study, clinic_code = run_dir.name.split("__", 1)
        if clinic_code not in PROMPT_LEVELS + TRAD_LEVELS:
            continue

        for model_dir in sorted(run_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_dir.name not in MODEL_SPECS:
                continue

            csv_path = model_dir / "test_result.csv"
            if not csv_path.exists():
                continue

            stats = load_metric(csv_path, METRIC)
            if stats is None:
                continue

            mean, std, fold_num = stats
            rows.append(
                {
                    "study": study,
                    "study_label": dataset_label(study),
                    "clinic_code": clinic_code,
                    "encoding_family": clinic_code[0],
                    "level": int(clinic_code[1:]),
                    "model": model_dir.name,
                    "model_label": MODEL_SPECS[model_dir.name]["label"],
                    "model_family": MODEL_SPECS[model_dir.name]["model_family"],
                    "pooling": MODEL_SPECS[model_dir.name]["pooling"],
                    "mean": mean,
                    "std": std,
                    "fold_num": fold_num,
                }
            )

    if not rows:
        raise RuntimeError(f"No usable results found under {series_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["study_label", "encoding_family", "level", "model_label"],
        kind="stable",
    ).reset_index(drop=True)
    return df


def tint_towards_gray(color: str, gray: float = 0.55, mix: float = 0.58) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color), dtype=float)
    target = np.asarray([gray, gray, gray], dtype=float)
    return tuple((1.0 - mix) * base + mix * target)


def ensure_dirs(output_root: Path) -> tuple[Path, Path]:
    summary_dir = output_root / "summary"
    figure_dir = output_root / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir, figure_dir


def savefig(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def infer_ylim(prompt_df: pd.DataFrame, traditional_df: pd.DataFrame) -> tuple[float, float]:
    all_lower = np.concatenate(
        [
            (prompt_df["mean"] - prompt_df["std"]).to_numpy(dtype=float),
            (traditional_df["mean"] - traditional_df["std"]).to_numpy(dtype=float) if not traditional_df.empty else np.array([], dtype=float),
        ]
    )
    all_upper = np.concatenate(
        [
            (prompt_df["mean"] + prompt_df["std"]).to_numpy(dtype=float),
            (traditional_df["mean"] + traditional_df["std"]).to_numpy(dtype=float) if not traditional_df.empty else np.array([], dtype=float),
        ]
    )
    ymin = max(0.0, float(np.nanmin(all_lower)) - 0.02)
    ymax = min(1.0, float(np.nanmax(all_upper)) + 0.03)
    if ymax - ymin < 0.08:
        mid = (ymin + ymax) / 2.0
        ymin = max(0.0, mid - 0.04)
        ymax = min(1.0, mid + 0.04)
    return ymin, ymax


def draw_study_panel(
    ax: plt.Axes,
    study: str,
    study_prompt: pd.DataFrame,
    study_traditional: pd.DataFrame,
    ymin: float,
    ymax: float,
) -> None:
    x = np.arange(len(PROMPT_LEVELS))

    for model_name, spec in MODEL_SPECS.items():
        model_prompt = (
            study_prompt[study_prompt["model"].eq(model_name)]
            .sort_values(by=["level"], kind="stable")
        )
        if model_prompt.empty:
            continue

        means = model_prompt["mean"].to_numpy(dtype=float)
        stds = model_prompt["std"].to_numpy(dtype=float)
        lower = np.clip(means - stds, 0.0, 1.0)
        upper = np.clip(means + stds, 0.0, 1.0)

        ax.plot(
            x,
            means,
            color=spec["color"],
            marker=spec["marker"],
            lw=2.0,
            ms=5.5,
            label=spec["label"],
            zorder=3,
        )
        ax.fill_between(x, lower, upper, color=spec["color"], alpha=0.16, zorder=2)

        model_traditional = (
            study_traditional[study_traditional["model"].eq(model_name)]
            .sort_values(by=["level"], kind="stable")
        )
        if not model_traditional.empty:
            trad_means = model_traditional["mean"].to_numpy(dtype=float)
            trad_stds = model_traditional["std"].to_numpy(dtype=float)
            trad_lower = np.clip(trad_means - trad_stds, 0.0, 1.0)
            trad_upper = np.clip(trad_means + trad_stds, 0.0, 1.0)
            trad_color = tint_towards_gray(spec["color"])
            ax.plot(
                x,
                trad_means,
                color=trad_color,
                marker=spec["marker"],
                lw=1.8,
                ms=5.0,
                linestyle="--",
                alpha=0.95,
                zorder=2,
            )
            ax.fill_between(x, trad_lower, trad_upper, color=trad_color, alpha=0.12, zorder=1)

    ax.set_title(dataset_label(study), fontsize=12, loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(PROMPT_LEVELS)
    ax.set_ylim(ymin, ymax)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_prompt_vs_traditional(prompt_df: pd.DataFrame, traditional_df: pd.DataFrame, figure_dir: Path) -> None:
    studies = ordered_studies(prompt_df["study"].drop_duplicates().tolist())
    if not studies:
        raise RuntimeError("No prompt L-group results found for plotting.")

    n = len(studies)
    ncols = 3 if n <= 9 else 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 5.3, nrows * 4.1),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    ymin, ymax = infer_ylim(prompt_df, traditional_df)

    for idx, study in enumerate(studies):
        ax = axes[idx // ncols, idx % ncols]
        study_prompt = prompt_df[prompt_df["study"].eq(study)].copy()
        study_traditional = traditional_df[traditional_df["study"].eq(study)].copy()
        draw_study_panel(ax, study, study_prompt, study_traditional, ymin, ymax)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    for row in axes[:, 0]:
        row.set_ylabel("Test C-index")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Prompt Layer")

    line_handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            lw=2.0,
            ms=6,
            label=spec["label"],
        )
        for spec in MODEL_SPECS.values()
    ]
    baseline_handle = Line2D(
        [0],
        [0],
        color="#777777",
        linestyle="--",
        lw=1.5,
        label="Traditional D0-D5",
    )
    prompt_style_handle = Line2D(
        [0],
        [0],
        color="#444444",
        linestyle="-",
        lw=1.8,
        label="Prompt L0-L5",
    )
    fig.legend(
        handles=line_handles + [prompt_style_handle, baseline_handle],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Clinical Prompt Layer Trend Across Datasets\nSolid: L0-L5 prompt layers; dashed gray-tinted: D0-D5 traditional encodings",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
    savefig(fig, figure_dir / "clinical_prompt_vs_traditional")


def plot_per_dataset(prompt_df: pd.DataFrame, traditional_df: pd.DataFrame, figure_dir: Path) -> None:
    out_dir = figure_dir / "by_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = ordered_studies(prompt_df["study"].drop_duplicates().tolist())
    ymin, ymax = infer_ylim(prompt_df, traditional_df)

    line_handles = [
        Line2D(
            [0],
            [0],
            color=spec["color"],
            marker=spec["marker"],
            lw=2.0,
            ms=6,
            label=spec["label"],
        )
        for spec in MODEL_SPECS.values()
    ]
    baseline_handle = Line2D(
        [0],
        [0],
        color="#777777",
        linestyle="--",
        lw=1.5,
        label="Traditional D0-D5",
    )
    prompt_style_handle = Line2D(
        [0],
        [0],
        color="#444444",
        linestyle="-",
        lw=1.8,
        label="Prompt L0-L5",
    )

    for study in studies:
        study_prompt = prompt_df[prompt_df["study"].eq(study)].copy()
        study_traditional = traditional_df[traditional_df["study"].eq(study)].copy()
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        draw_study_panel(ax, study, study_prompt, study_traditional, ymin, ymax)
        ax.set_xlabel("Prompt Layer")
        ax.set_ylabel("Test C-index")
        fig.legend(
            handles=line_handles + [prompt_style_handle, baseline_handle],
            loc="lower center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
        fig.suptitle(
            f"{dataset_label(study)} Clinical Prompt Trend",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.93])
        savefig(fig, out_dir / study)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot FigB clinical prompt trends.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Root results directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results_display" / RESULTS_SERIES,
        help="Output directory under results_display.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary_dir, figure_dir = ensure_dirs(args.output_root)
    long_df = collect_long_table(args.results_root)
    prompt_df = long_df[long_df["encoding_family"].eq("L")].copy()
    traditional_df = long_df[long_df["encoding_family"].eq("D")].copy()

    long_path = summary_dir / "clinical_prompt_long.csv"
    traditional_path = summary_dir / "clinical_traditional_long.csv"
    long_df.to_csv(long_path, index=False)
    traditional_df.to_csv(traditional_path, index=False)

    plot_prompt_vs_traditional(prompt_df, traditional_df, figure_dir)
    plot_per_dataset(prompt_df, traditional_df, figure_dir)

    print(f"[OK] long summary: {long_path}")
    print(f"[OK] traditional summary: {traditional_path}")
    print(f"[OK] figures: {figure_dir / 'clinical_prompt_vs_traditional.png'}")


if __name__ == "__main__":
    main()
