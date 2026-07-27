"""
Render figure outputs for Single_Multi_Test summary analysis.

Inputs:
    results_display/Single_Multi_Test/Sum/*.csv

Outputs:
    results_display/Single_Multi_Test/Sum/figures/*.png
    results_display/Single_Multi_Test/Sum/figures/*.pdf
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import friedmanchisquare


DATASET_LABELS = {
    "tcga_kich": "KICH",
    "tcga_kirc": "KIRC",
    "tcga_kirp": "KIRP",
}

MODALITY_COLORS = {
    "WSI": "#355C7D",
    "Gene": "#C06C84",
    "Clinical": "#6C9A8B",
}

MODEL_COLORS = {
    "mlp": "#1768AC",
    "snn": "#DB3A34",
}

POOLING_STYLES = {
    "mean": "--",
    "flatten": "-",
}

PANORAMA_CMAP = LinearSegmentedColormap.from_list(
    "panorama_cmap",
    ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
)

DELTA_CMAP = LinearSegmentedColormap.from_list(
    "delta_cmap",
    ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
)


def ensure_dirs(project_root: Path) -> tuple[Path, Path, Path]:
    sum_dir = project_root / "results_display/Single_Multi_Test/Sum"
    internal_dir = sum_dir / "_internal"
    fig_dir = sum_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return sum_dir, internal_dir, fig_dir


def savefig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=360, bbox_inches="tight")
    plt.close(fig)


def resolve_csv(sum_dir: Path, internal_dir: Path, filename: str) -> Path:
    primary = sum_dir / filename
    if primary.exists():
        return primary
    fallback = internal_dir / filename
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Cannot find required CSV: {filename}")


def tidy_config_label(label: str) -> str:
    return (
        label.replace("Clinical | ", "C | ")
        .replace("Gene | ", "G | ")
        .replace("WSI | ", "W | ")
        .replace("scFoundation", "scF")
        .replace("flatten", "flat")
    )


def modality_breaks(config_meta: pd.DataFrame) -> list[int]:
    mods = config_meta["modality"].tolist()
    breaks: list[int] = []
    for idx in range(1, len(mods)):
        if mods[idx] != mods[idx - 1]:
            breaks.append(idx)
    return breaks


def cleanup_old_outputs(fig_dir: Path) -> None:
    for path in fig_dir.glob("*"):
        if path.is_file():
            path.unlink()


def plot_panorama_per_dataset(sum_dir: Path, internal_dir: Path, fig_dir: Path) -> None:
    mean_df = pd.read_csv(resolve_csv(sum_dir, internal_dir, "panorama_matrix_mean.csv"))
    std_df = pd.read_csv(resolve_csv(sum_dir, internal_dir, "panorama_matrix_std.csv"))
    datasets = ["tcga_kich", "tcga_kirc", "tcga_kirp"]
    config_labels = mean_df["config_label"].tolist()
    modality_meta = mean_df[["config_label", "modality"]].copy()
    breaks = modality_breaks(modality_meta)
    n_rows = len(config_labels)

    for dataset in datasets:
        dataset_col = mean_df[[dataset]].to_numpy(dtype=float)
        dataset_std_col = std_df[[dataset]].to_numpy(dtype=float)
        local_min = float(np.nanmin(dataset_col))
        local_max = float(np.nanmax(dataset_col))
        pad = max(0.01, (local_max - local_min) * 0.08)
        vmin = local_min - pad
        vmax = local_max + pad
        midpoint = (local_min + local_max) / 2
        fig_h = max(12.5, n_rows * 0.34)
        fig, ax = plt.subplots(figsize=(7.8, fig_h))
        im = ax.imshow(
            dataset_col,
            aspect="auto",
            cmap=PANORAMA_CMAP,
            norm=PowerNorm(gamma=0.75, vmin=vmin, vmax=vmax),
        )
        for row in range(n_rows):
            value = dataset_col[row, 0]
            std = dataset_std_col[row, 0]
            text_color = "white" if value > midpoint else "#0f172a"
            ax.text(
                0,
                row,
                f"{value:.3f}\n±{std:.3f}",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )

        ax.set_title(
            f"2. {DATASET_LABELS[dataset]} Panorama Heatmap",
            fontsize=16,
            pad=10,
            loc="left",
        )
        ax.set_xticks([0])
        ax.set_xticklabels(["C-index\n(mean±std)"], fontsize=9)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([tidy_config_label(x) for x in config_labels], fontsize=7.8)
        for y in breaks:
            ax.axhline(y - 0.5, color="#ffffff", lw=3)
            ax.axhline(y - 0.5, color="#4f4f4f", lw=0.7)
        cbar = fig.colorbar(im, ax=ax, shrink=0.28, pad=0.03)
        cbar.set_label(
            f"Test C-index\nrange {local_min:.3f}-{local_max:.3f}",
            fontsize=10,
        )

        modality_runs = []
        start = 0
        mods = modality_meta["modality"].tolist()
        for idx in range(1, len(mods) + 1):
            if idx == len(mods) or mods[idx] != mods[start]:
                center = (start + idx - 1) / 2
                modality_runs.append((mods[start], center))
                start = idx
        for modality, center in modality_runs:
            ax.text(
                -0.78,
                center,
                modality,
                rotation=90,
                fontsize=10,
                fontweight="bold",
                color=MODALITY_COLORS[modality],
                va="center",
                ha="center",
                transform=ax.transData,
                clip_on=False,
            )
        savefig(fig, fig_dir / f"2_panorama_{dataset}_heatmap")


def build_delta_row_labels(df: pd.DataFrame) -> list[str]:
    excluded = {
        "tcga_kich",
        "tcga_kirc",
        "tcga_kirp",
        "positive_count",
        "available_n",
        "consistency",
        "mean_delta",
    }
    label_cols = [col for col in df.columns if col not in excluded]
    labels = []
    for _, row in df.iterrows():
        parts = [str(row[col]) for col in label_cols if pd.notna(row[col])]
        parts.append(f"[{row['consistency']}]")
        labels.append(" | ".join(parts))
    return labels


def delta_title(file_stem: str) -> str:
    mapping = {
        "delta_gene_norm_vs_raw": "3. Gene: norm vs raw",
        "delta_gene_cell_vs_gene": "3. Gene: cell vs gene",
        "delta_gene_scfoundation_vs_csvraw": "3. Gene: scFoundation vs csvraw",
        "delta_wsi_uni_v2_vs_v1": "3. WSI: UNI v2 vs v1",
        "delta_clinical_flatten_vs_mean": "3. Clinical: flatten vs mean",
        "delta_model_snn_vs_mlp_gene": "3. Model: SNN vs MLP on Gene",
        "delta_model_snn_vs_mlp_clinical": "3. Model: SNN vs MLP on Clinical",
    }
    return mapping.get(file_stem, file_stem)


def plot_delta_heatmaps(sum_dir: Path, internal_dir: Path, fig_dir: Path) -> None:
    delta_files = [
        "delta_gene_norm_vs_raw.csv",
        "delta_gene_cell_vs_gene.csv",
        "delta_gene_scfoundation_vs_csvraw.csv",
        "delta_wsi_uni_v2_vs_v1.csv",
        "delta_clinical_flatten_vs_mean.csv",
        "delta_model_snn_vs_mlp_gene.csv",
        "delta_model_snn_vs_mlp_clinical.csv",
    ]
    datasets = ["tcga_kich", "tcga_kirc", "tcga_kirp"]

    matrices: list[tuple[str, pd.DataFrame]] = []
    for file_name in delta_files:
        df = pd.read_csv(resolve_csv(sum_dir, internal_dir, file_name))
        matrices.append((file_name.replace(".csv", ""), df))

    for stem, df in matrices:
        vals = df[datasets].to_numpy(dtype=float)
        labels = build_delta_row_labels(df)
        local_abs = float(np.nanmax(np.abs(vals)))
        local_abs = max(local_abs, 1e-6)
        norm = TwoSlopeNorm(vmin=-local_abs, vcenter=0.0, vmax=local_abs)
        fig_h = max(3.4, vals.shape[0] * 0.62 + 1.2)
        fig, ax = plt.subplots(figsize=(8.4, fig_h))
        im = ax.imshow(vals, aspect="auto", cmap=DELTA_CMAP, norm=norm)
        ax.set_title(delta_title(stem), fontsize=15, pad=8, loc="left")
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels([DATASET_LABELS[x] for x in datasets], fontsize=10)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                value = vals[i, j]
                color = "white" if abs(value) > local_abs * 0.42 else "#111827"
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=color)
        cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.03)
        cbar.set_label(f"Delta C-index\nrange ±{local_abs:.3f}", fontsize=10)
        savefig(fig, fig_dir / f"3_{stem}")


def plot_clinical_trends(sum_dir: Path, internal_dir: Path, fig_dir: Path) -> None:
    trend_df = pd.read_csv(resolve_csv(sum_dir, internal_dir, "clinical_layer_trend_long.csv"))
    rho_df = pd.read_csv(resolve_csv(sum_dir, internal_dir, "4_clinical_layer_spearman_summary.csv"))
    datasets = ["tcga_kich", "tcga_kirc", "tcga_kirp"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, dataset in zip(axes, datasets):
        sub = trend_df[trend_df["dataset"].eq(dataset)].copy()
        for (model_base, pooling), group in sub.groupby(["model_base", "pooling"]):
            group = group.sort_values("layer", kind="stable")
            x = group["layer"].to_numpy(dtype=float)
            y = group["test_cindex_mean"].to_numpy(dtype=float)
            yerr = group["test_cindex_std"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_base],
                linestyle=POOLING_STYLES[pooling],
                linewidth=2.2,
            )
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=MODEL_COLORS[model_base],
                alpha=0.13,
            )

        rho_lines = []
        rho_sub = rho_df[rho_df["dataset"].eq(dataset)].copy()
        for _, row in rho_sub.sort_values(["model_base", "pooling"], kind="stable").iterrows():
            rho_lines.append(f"{row['model_base'].upper()}-{row['pooling']}: rho={row['spearman_rho']:.2f}")
        ax.text(
            0.02,
            0.02,
            "\n".join(rho_lines),
            transform=ax.transAxes,
            fontsize=8.2,
            va="bottom",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d0d0"},
        )
        ax.set_title(DATASET_LABELS[dataset], fontsize=13)
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"L{i}" for i in range(6)])
        ax.set_xlabel("Clinical layer depth", fontsize=10)
        ax.grid(alpha=0.18, axis="y")

    axes[0].set_ylabel("Test C-index", fontsize=11)
    legend_elements = [
        Line2D([0], [0], color=MODEL_COLORS["mlp"], lw=2.4, label="MLP"),
        Line2D([0], [0], color=MODEL_COLORS["snn"], lw=2.4, label="SNN"),
        Line2D([0], [0], color="#555555", lw=2.4, linestyle=POOLING_STYLES["mean"], label="mean"),
        Line2D([0], [0], color="#555555", lw=2.4, linestyle=POOLING_STYLES["flatten"], label="flatten"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("4. Clinical Layer Trend Across Datasets", fontsize=16, y=1.08)
    savefig(fig, fig_dir / "4_clinical_layer_trends")


def plot_average_rank(sum_dir: Path, internal_dir: Path, fig_dir: Path) -> None:
    rank_df = pd.read_csv(resolve_csv(sum_dir, internal_dir, "5_average_rank_all_configs.csv"))
    rank_df["display_label"] = rank_df["config_label"].apply(tidy_config_label)
    rank_df = rank_df.sort_values(["average_rank", "config_label"], kind="stable").reset_index(drop=True)

    fig_h = max(11.5, len(rank_df) * 0.23)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    y = np.arange(len(rank_df))
    colors = rank_df["modality"].map(MODALITY_COLORS).tolist()
    ax.hlines(
        y,
        xmin=0,
        xmax=rank_df["average_rank"].to_numpy(dtype=float),
        color="#d1d5db",
        lw=1.3,
    )
    ax.errorbar(
        rank_df["average_rank"].to_numpy(dtype=float),
        y,
        xerr=rank_df["rank_std"].fillna(0).to_numpy(dtype=float),
        fmt="none",
        ecolor="#6b7280",
        elinewidth=1.1,
        capsize=2.5,
        zorder=2,
    )
    ax.scatter(
        rank_df["average_rank"].to_numpy(dtype=float),
        y,
        s=52,
        c=colors,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(rank_df["display_label"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank across KICH/KIRC/KIRP (lower is better)", fontsize=11)
    ax.set_title("5. Cross-Dataset Robustness by Average Rank", fontsize=15, pad=10)
    ax.grid(axis="x", alpha=0.2)

    legend_elements = [
        Line2D([0], [0], color=color, lw=8, label=modality)
        for modality, color in MODALITY_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=False)

    friedman_note = ""
    rank_matrix = pd.read_csv(resolve_csv(sum_dir, internal_dir, "average_rank_long.csv"))
    pivot = rank_matrix.pivot(index="dataset", columns="config_label", values="dataset_rank")
    if pivot.shape[0] >= 3 and pivot.shape[1] >= 2:
        ordered = [pivot[col].to_numpy(dtype=float) for col in pivot.columns]
        stat, pvalue = friedmanchisquare(*ordered)
        friedman_note = f"Friedman chi2={stat:.2f}, p={pvalue:.3g}"

    if friedman_note:
        ax.text(
            0.99,
            0.01,
            friedman_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#4a4a4a",
        )

    savefig(fig, fig_dir / "5_average_rank_all_configs")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sum_dir, internal_dir, fig_dir = ensure_dirs(project_root)
    cleanup_old_outputs(fig_dir)
    plot_panorama_per_dataset(sum_dir, internal_dir, fig_dir)
    plot_delta_heatmaps(sum_dir, internal_dir, fig_dir)
    plot_clinical_trends(sum_dir, internal_dir, fig_dir)
    plot_average_rank(sum_dir, internal_dir, fig_dir)
    print(f"Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()
