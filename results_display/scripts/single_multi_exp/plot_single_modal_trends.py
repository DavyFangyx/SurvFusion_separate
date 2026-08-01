"""
Render single-modal trend and conclusion figures from the collected long table.

Input:
    results_display/Single_Modal_Trend/summary/single_modal_cindex_long.csv

Outputs:
    results_display/Single_Modal_Trend/summary/*.csv
    results_display/Single_Modal_Trend/figures/**/*.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SERIES_ORDER = ["Clinictest_Li", "WSItest_F", "Genetest"]
SERIES_LABELS = {
    "Clinictest_Li": "Clinical",
    "WSItest_F": "WSI",
    "Genetest": "Gene",
}
SERIES_COLORS = {
    "Clinictest_Li": "#6C9A8B",
    "WSItest_F": "#355C7D",
    "Genetest": "#C06C84",
}
MODEL_COLORS = {
    "Cox": "#2F4858",
    "MLP-mean": "#1F78B4",
    "MLP-flatten": "#5AA9E6",
    "SNN-mean": "#D1495B",
    "SNN-flatten": "#EDAE49",
    "MLP": "#1768AC",
    "SNN": "#DB3A34",
    "ABMIL": "#5B8E7D",
    "TransMIL": "#9C6644",
}
INPUT_COLORS = {
    "L0": "#264653",
    "L1": "#2A9D8F",
    "L2": "#8AB17D",
    "L3": "#E9C46A",
    "L4": "#F4A261",
    "L5": "#E76F51",
    "CSV raw": "#6D597A",
    "scF gene raw": "#B56576",
    "scF gene norm": "#E56B6F",
    "scF cell raw": "#355070",
    "scF cell norm": "#6D597A",
    "UNI v1": "#4C6E91",
    "UNI v2": "#7DA0CA",
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    summary_dir = base_dir / "summary"
    fig_dir = base_dir / "figures"
    dirs = {
        "summary": summary_dir,
        "fig_root": fig_dir,
        "by_dataset_model": fig_dir / "by_dataset_model",
        "by_dataset_input": fig_dir / "by_dataset_input",
        "conclusion": fig_dir / "conclusion",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def savefig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=320, bbox_inches="tight")
    plt.close(fig)


def load_long_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sort_cols = [
        "series",
        "dataset_order",
        "dataset",
        "input_order",
        "input_key",
        "model_order",
        "model_key",
    ]
    return df.sort_values(by=sort_cols, kind="stable").reset_index(drop=True)


def aggregate_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "series",
                "modality",
                "dataset",
                "dataset_label",
                "dataset_order",
                "model_key",
                "model_label",
                "model_order",
            ],
            as_index=False,
        )
        .agg(
            mean=("mean", "mean"),
            std_across_inputs=("mean", "std"),
            input_count=("input_key", "nunique"),
            best_input_mean=("mean", "max"),
        )
        .fillna({"std_across_inputs": 0.0})
    )
    return grouped.sort_values(
        by=["series", "dataset_order", "dataset", "model_order", "model_key"],
        kind="stable",
    )


def aggregate_input_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(
            [
                "series",
                "modality",
                "dataset",
                "dataset_label",
                "dataset_order",
                "input_key",
                "input_label",
                "input_order",
            ],
            as_index=False,
        )
        .agg(
            mean=("mean", "mean"),
            std_across_models=("mean", "std"),
            model_count=("model_key", "nunique"),
            best_model_mean=("mean", "max"),
        )
        .fillna({"std_across_models": 0.0})
    )
    return grouped.sort_values(
        by=["series", "dataset_order", "dataset", "input_order", "input_key"],
        kind="stable",
    )


def aggregate_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    combo_df = df.copy()
    combo_df["combo_key"] = combo_df["input_key"] + "__" + combo_df["model_key"]
    combo_df["combo_label"] = combo_df["input_label"] + " + " + combo_df["model_label"]
    combo_df["combo_order"] = combo_df["input_order"] * 100 + combo_df["model_order"]
    return combo_df.sort_values(
        by=["series", "dataset_order", "dataset", "combo_order", "combo_key"],
        kind="stable",
    ).reset_index(drop=True)


def rank_summary(df: pd.DataFrame, value_col: str, rank_name: str) -> pd.DataFrame:
    ranked = df.copy()
    ranked[rank_name] = ranked.groupby(["series", "dataset"])[value_col].rank(
        method="min",
        ascending=False,
    )
    return ranked


def average_rank_table(ranked: pd.DataFrame, entity_cols: list[str], rank_name: str, extra_sort_col: str) -> pd.DataFrame:
    cols = ["series", "modality", *entity_cols]
    agg = (
        ranked.groupby(cols, as_index=False)
        .agg(
            average_rank=(rank_name, "mean"),
            win_count=(rank_name, lambda s: int(np.sum(np.isclose(s, 1.0)))),
            dataset_count=("dataset", "nunique"),
        )
        .sort_values(by=["series", "average_rank", extra_sort_col], kind="stable")
    )
    return agg


def winner_table(df: pd.DataFrame, value_col: str, entity_key: str, entity_label: str) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for (_, _), group in df.groupby(["series", "dataset"], sort=False):
        rows.append(group.sort_values(by=[value_col, entity_key], ascending=[False, True]).iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out[
        [
            "series",
            "modality",
            "dataset",
            "dataset_label",
            entity_key,
            entity_label,
            value_col,
        ]
    ]


def best_config_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for dataset, group in df.groupby("dataset", sort=False):
        rows.append(
            group.sort_values(
                by=["mean", "series", "model_key", "input_key"],
                ascending=[False, True, True, True],
            ).iloc[0]
        )
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out[
        [
            "dataset",
            "dataset_label",
            "series",
            "modality",
            "input_key",
            "input_label",
            "model_key",
            "model_label",
            "mean",
            "std",
            "fold_num",
        ]
    ].sort_values(by=["dataset_label"], kind="stable")


def dataset_column_order(summary_df: pd.DataFrame) -> list[str]:
    ordered = (
        summary_df[["dataset_label", "dataset_order", "dataset"]]
        .drop_duplicates()
        .sort_values(by=["dataset_order", "dataset"], kind="stable")
    )
    return ordered["dataset_label"].tolist()


def plot_single_series_line(
    x_labels: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    color: str,
    title: str,
    xlabel: str,
    out_base: Path,
) -> None:
    x = np.arange(len(x_labels))
    fig, ax = plt.subplots(figsize=(max(6.8, len(x_labels) * 1.2), 4.6))
    lower = np.clip(means - stds, 0.0, 1.0)
    upper = np.clip(means + stds, 0.0, 1.0)

    ax.plot(x, means, color=color, marker="o", lw=2.2, ms=6)
    if len(x_labels) >= 2:
        ax.fill_between(x, lower, upper, color=color, alpha=0.18)
    else:
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=color, elinewidth=1.8, capsize=5)

    for idx, value in enumerate(means):
        ax.text(idx, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test C-index")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=20 if len(x_labels) > 4 else 0, ha="right" if len(x_labels) > 4 else "center")
    ax.set_ylim(max(0.0, lower.min() - 0.03), min(1.0, upper.max() + 0.06))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, out_base)


def plot_trend_figures(df: pd.DataFrame, out_dir: Path) -> None:
    for (series, dataset, model_key), group in df.groupby(["series", "dataset", "model_key"], sort=False):
        ordered = group.sort_values(by=["input_order", "input_key"], kind="stable")
        series_dir = out_dir / "by_dataset_model" / series
        series_dir.mkdir(parents=True, exist_ok=True)
        plot_single_series_line(
            x_labels=ordered["input_label"].tolist(),
            means=ordered["mean"].to_numpy(dtype=float),
            stds=ordered["std"].to_numpy(dtype=float),
            color=MODEL_COLORS.get(model_key, SERIES_COLORS.get(series, "#355C7D")),
            title=f"{ordered.iloc[0]['dataset_label']} | {SERIES_LABELS.get(series, series)} | {ordered.iloc[0]['model_label']}",
            xlabel="Input",
            out_base=series_dir / f"{dataset}__{model_key.lower().replace(' ', '_')}",
        )

    for (series, dataset, input_key), group in df.groupby(["series", "dataset", "input_key"], sort=False):
        ordered = group.sort_values(by=["model_order", "model_key"], kind="stable")
        series_dir = out_dir / "by_dataset_input" / series
        series_dir.mkdir(parents=True, exist_ok=True)
        input_label = ordered.iloc[0]["input_label"]
        plot_single_series_line(
            x_labels=ordered["model_label"].tolist(),
            means=ordered["mean"].to_numpy(dtype=float),
            stds=ordered["std"].to_numpy(dtype=float),
            color=INPUT_COLORS.get(input_label, SERIES_COLORS.get(series, "#355C7D")),
            title=f"{ordered.iloc[0]['dataset_label']} | {SERIES_LABELS.get(series, series)} | {input_label}",
            xlabel="Model",
            out_base=series_dir / f"{dataset}__{input_key.lower().replace(' ', '_')}",
        )


def draw_heatmap(
    pivot: pd.DataFrame,
    title: str,
    subtitle: str,
    out_base: Path,
) -> None:
    values = pivot.to_numpy(dtype=float)
    fig_w = max(7.2, pivot.shape[1] * 1.15)
    fig_h = max(4.6, pivot.shape[0] * 0.72 + 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.YlGnBu
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    threshold = (vmin + vmax) / 2 if np.isfinite(vmin) and np.isfinite(vmax) else 0.0
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isnan(value):
                ax.text(col_idx, row_idx, "-", ha="center", va="center", fontsize=8.5, color="#6B7280")
                continue
            color = "white" if value >= threshold else "#102A43"
            ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=color)

    ax.set_title(f"{title}\n{subtitle}", fontsize=13, loc="left")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=25, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_xlabel("Dataset")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean C-index", fontsize=10)
    savefig(fig, out_base)


def plot_rank_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_base: Path) -> None:
    ordered = df.sort_values(by=[value_col, label_col], ascending=[True, True], kind="stable")
    fig_h = max(3.8, len(ordered) * 0.55)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    colors = [MODEL_COLORS.get(label, INPUT_COLORS.get(label, "#4C6E91")) for label in ordered[label_col]]
    ax.barh(ordered[label_col], ordered[value_col], color=colors, alpha=0.9)
    for y, value in zip(ordered[label_col], ordered[value_col]):
        ax.text(value + 0.02, y, f"{value:.2f}", va="center", fontsize=8.5)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    savefig(fig, out_base)


def plot_win_bar(df: pd.DataFrame, label_col: str, title: str, out_base: Path) -> None:
    ordered = df.sort_values(by=["win_count", label_col], ascending=[False, True], kind="stable")
    fig_h = max(3.8, len(ordered) * 0.55)
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    colors = [MODEL_COLORS.get(label, INPUT_COLORS.get(label, "#4C6E91")) for label in ordered[label_col]]
    ax.barh(ordered[label_col], ordered["win_count"], color=colors, alpha=0.9)
    for y, value in zip(ordered[label_col], ordered["win_count"]):
        ax.text(value + 0.03, y, f"{int(value)}", va="center", fontsize=8.5)
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel("Win Count")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    savefig(fig, out_base)


def plot_stability_scatter(df: pd.DataFrame, label_col: str, title: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    ax.scatter(df["mean_across_datasets"], df["std_across_datasets"], s=85, color="#355C7D", alpha=0.85)
    for _, row in df.iterrows():
        ax.text(
            row["mean_across_datasets"] + 0.0025,
            row["std_across_datasets"] + 0.001,
            str(row[label_col]),
            fontsize=8.5,
        )
    ax.set_title(title, fontsize=13, loc="left")
    ax.set_xlabel("Mean C-index across datasets")
    ax.set_ylabel("Std across datasets")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    savefig(fig, out_base)


def plot_conclusion_figures(
    model_summary: pd.DataFrame,
    input_summary: pd.DataFrame,
    combo_rank: pd.DataFrame,
    model_rank: pd.DataFrame,
    input_rank: pd.DataFrame,
    out_dir: Path,
) -> None:
    for series in SERIES_ORDER:
        series_model = model_summary[model_summary["series"] == series].copy()
        series_input = input_summary[input_summary["series"] == series].copy()
        if series_model.empty or series_input.empty:
            continue

        series_dir = out_dir / "conclusion" / series
        series_dir.mkdir(parents=True, exist_ok=True)
        label = SERIES_LABELS.get(series, series)
        dataset_order = dataset_column_order(series_model)

        model_pivot = (
            series_model.sort_values(by=["model_order", "model_key"], kind="stable")
            .pivot(index="model_label", columns="dataset_label", values="mean")
            .reindex(columns=dataset_order)
        )
        input_pivot = (
            series_input.sort_values(by=["input_order", "input_key"], kind="stable")
            .pivot(index="input_label", columns="dataset_label", values="mean")
            .reindex(columns=dataset_order)
        )
        draw_heatmap(
            model_pivot,
            title=f"{label}: Model x Dataset",
            subtitle="Cell value = mean C-index averaged over inputs",
            out_base=series_dir / "model_dataset_heatmap",
        )
        draw_heatmap(
            input_pivot,
            title=f"{label}: Input x Dataset",
            subtitle="Cell value = mean C-index averaged over models",
            out_base=series_dir / "input_dataset_heatmap",
        )

        local_model_rank = model_rank[model_rank["series"] == series].copy()
        local_input_rank = input_rank[input_rank["series"] == series].copy()
        local_combo_rank = combo_rank[combo_rank["series"] == series].copy()
        plot_rank_bar(
            local_model_rank,
            label_col="model_label",
            value_col="average_rank",
            title=f"{label}: Model Average Rank",
            out_base=series_dir / "model_average_rank",
        )
        plot_rank_bar(
            local_input_rank,
            label_col="input_label",
            value_col="average_rank",
            title=f"{label}: Input Average Rank",
            out_base=series_dir / "input_average_rank",
        )
        plot_rank_bar(
            local_combo_rank,
            label_col="combo_label",
            value_col="average_rank",
            title=f"{label}: Input + Model Average Rank",
            out_base=series_dir / "combo_average_rank",
        )
        plot_win_bar(
            local_model_rank,
            label_col="model_label",
            title=f"{label}: Model Win Count",
            out_base=series_dir / "model_win_count",
        )
        plot_win_bar(
            local_input_rank,
            label_col="input_label",
            title=f"{label}: Input Win Count",
            out_base=series_dir / "input_win_count",
        )

        model_stability = (
            series_model.groupby(["model_label"], as_index=False)
            .agg(
                mean_across_datasets=("mean", "mean"),
                std_across_datasets=("mean", "std"),
            )
            .fillna({"std_across_datasets": 0.0})
        )
        input_stability = (
            series_input.groupby(["input_label"], as_index=False)
            .agg(
                mean_across_datasets=("mean", "mean"),
                std_across_datasets=("mean", "std"),
            )
            .fillna({"std_across_datasets": 0.0})
        )
        plot_stability_scatter(
            model_stability,
            label_col="model_label",
            title=f"{label}: Model Stability",
            out_base=series_dir / "model_stability",
        )
        plot_stability_scatter(
            input_stability,
            label_col="input_label",
            title=f"{label}: Input Stability",
            out_base=series_dir / "input_stability",
        )


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Plot single-modal trends and summary figures")
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=project_root / "results_display/Single_Modal_Trend/summary/single_modal_cindex_long.csv",
        help="Collected long-format CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display/Single_Modal_Trend",
        help="Base output directory",
    )
    args = parser.parse_args()

    dirs = ensure_dirs(args.output_dir)
    df = load_long_df(args.input_csv)

    model_summary = aggregate_model_summary(df)
    input_summary = aggregate_input_summary(df)
    combo_summary = aggregate_combo_summary(df)
    model_ranked = rank_summary(model_summary, "mean", "rank")
    input_ranked = rank_summary(input_summary, "mean", "rank")
    combo_ranked = rank_summary(combo_summary, "mean", "rank")
    model_rank = average_rank_table(model_ranked, ["model_key", "model_label", "model_order"], "rank", "model_order")
    input_rank = average_rank_table(input_ranked, ["input_key", "input_label", "input_order"], "rank", "input_order")
    combo_rank = average_rank_table(
        combo_ranked,
        [
            "input_key",
            "input_label",
            "input_order",
            "model_key",
            "model_label",
            "model_order",
            "combo_key",
            "combo_label",
            "combo_order",
        ],
        "rank",
        "combo_order",
    )
    best_model = winner_table(model_summary, "mean", "model_key", "model_label")
    best_input = winner_table(input_summary, "mean", "input_key", "input_label")
    best_config = best_config_table(df)
    model_win = model_rank[["series", "modality", "model_key", "model_label", "win_count"]].copy()
    input_win = input_rank[["series", "modality", "input_key", "input_label", "win_count"]].copy()

    model_summary.to_csv(dirs["summary"] / "model_dataset_summary.csv", index=False)
    input_summary.to_csv(dirs["summary"] / "input_dataset_summary.csv", index=False)
    model_rank.to_csv(dirs["summary"] / "model_average_rank.csv", index=False)
    input_rank.to_csv(dirs["summary"] / "input_average_rank.csv", index=False)
    combo_rank.to_csv(dirs["summary"] / "combo_average_rank.csv", index=False)
    model_win.to_csv(dirs["summary"] / "model_win_count.csv", index=False)
    input_win.to_csv(dirs["summary"] / "input_win_count.csv", index=False)
    best_model.to_csv(dirs["summary"] / "best_model_per_dataset.csv", index=False)
    best_input.to_csv(dirs["summary"] / "best_input_per_dataset.csv", index=False)
    best_config.to_csv(dirs["summary"] / "best_single_modal_config_per_dataset.csv", index=False)

    plot_trend_figures(df, dirs["fig_root"])
    plot_conclusion_figures(
        model_summary,
        input_summary,
        combo_rank,
        model_rank,
        input_rank,
        dirs["fig_root"],
    )

    print(f"[WRITE] {args.output_dir.relative_to(project_root)}")
    print("Done.")


if __name__ == "__main__":
    main()
