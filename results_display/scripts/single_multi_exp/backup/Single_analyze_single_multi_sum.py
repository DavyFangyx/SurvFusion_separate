"""
Build downstream summary-analysis CSVs for Single_Multi_Test.

Inputs:
    results_display/Single_Multi_Test/*/*_combined.csv
    results/Single_Multi_Test/**/splits_0.csv
    datasets_csv/metadata/tcga_*.csv

Outputs:
    results_display/Single_Multi_Test/Sum/*.csv

This script focuses on analysis-ready CSVs rather than LaTeX/figures.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_deployment.registry import get_dataset_config, list_enabled_studies

DATASETS = list_enabled_studies()
DATASET_LABELS = {
    study: get_dataset_config(study).display_name.replace("TCGA-", "").replace("TCGA_", "")
    for study in DATASETS
}
SERIES_ORDER = {
    "WSItest_F": 0,
    "Gengtest_CSVRAW": 1,
    "Gengtest_F": 1,
    "Clinictest_Li": 2,
}
MODALITY_ORDER = {"WSI": 0, "Gene": 1, "Clinical": 2}
MODEL_BASE_ORDER = {"abmil": 0, "mlp": 1, "snn": 2, "transmil": 3}
POOLING_ORDER = {"mean": 0, "flatten": 1}
GENE_SOURCE_ORDER = {"csvraw": 0, "scFoundation": 1}
GENE_REPR_ORDER = {"gene": 0, "cell": 1}
GENE_PREP_ORDER = {"raw": 0, "norm": 1}
WSI_EMBED_ORDER = {"uni_v1": 0, "uni_v2": 1}


def parse_gene_foundation_embedding(embedding: str) -> tuple[str, str]:
    match = re.fullmatch(r"scFoundation_embedding_(cell|gene)_(raw|norm)", embedding)
    if not match:
        raise ValueError(f"Unexpected gene foundation embedding: {embedding}")
    return match.group(1), match.group(2)


def parse_clinical_layer(embedding: str) -> int:
    match = re.fullmatch(r"L(\d+)", embedding)
    if not match:
        raise ValueError(f"Unexpected clinical embedding: {embedding}")
    return int(match.group(1))


def build_analysis_frame(project_root: Path) -> pd.DataFrame:
    combined_paths = [
        project_root / "results_display/Single_Multi_Test/Gengtest_CSVRAW/Gengtest_CSVRAW_combined.csv",
        project_root / "results_display/Single_Multi_Test/Gengtest_F/Gengtest_F_combined.csv",
        project_root / "results_display/Single_Multi_Test/Clinictest_Li/Clinictest_Li_combined.csv",
        project_root / "results_display/Single_Multi_Test/WSItest_F/WSItest_F_combined.csv",
    ]

    frames = [pd.read_csv(path) for path in combined_paths]
    df = pd.concat(frames, ignore_index=True)

    df["dataset_label"] = df["dataset"].map(DATASET_LABELS)
    df["series_order"] = df["series"].map(SERIES_ORDER)

    df["modality"] = "Other"
    df["model_base"] = df["model"].str.split("_").str[0]
    df["model_display"] = df["model_base"].str.upper()
    df["pooling"] = pd.NA
    df["layer"] = pd.NA
    df["gene_source"] = pd.NA
    df["gene_repr"] = pd.NA
    df["gene_prep"] = pd.NA
    df["wsi_encoder"] = pd.NA

    gene_csvraw_mask = df["series"].eq("Gengtest_CSVRAW")
    gene_foundation_mask = df["series"].eq("Gengtest_F")
    clinical_mask = df["series"].eq("Clinictest_Li")
    wsi_mask = df["series"].eq("WSItest_F")

    df.loc[gene_csvraw_mask | gene_foundation_mask, "modality"] = "Gene"
    df.loc[clinical_mask, "modality"] = "Clinical"
    df.loc[wsi_mask, "modality"] = "WSI"

    df.loc[gene_csvraw_mask, "gene_source"] = "csvraw"
    df.loc[gene_csvraw_mask, "gene_repr"] = "gene"
    df.loc[gene_csvraw_mask, "gene_prep"] = "raw"

    for idx in df.index[gene_foundation_mask]:
        gene_repr, gene_prep = parse_gene_foundation_embedding(df.at[idx, "embedding"])
        df.at[idx, "gene_source"] = "scFoundation"
        df.at[idx, "gene_repr"] = gene_repr
        df.at[idx, "gene_prep"] = gene_prep

    df.loc[clinical_mask, "pooling"] = df.loc[clinical_mask, "model"].str.split("_").str[-1]
    df.loc[clinical_mask, "layer"] = df.loc[clinical_mask, "embedding"].apply(parse_clinical_layer)

    df.loc[wsi_mask, "wsi_encoder"] = df.loc[wsi_mask, "embedding"]
    df.loc[df["model_base"].eq("abmil"), "model_display"] = "ABMIL"
    df.loc[df["model_base"].eq("mlp"), "model_display"] = "MLP"
    df.loc[df["model_base"].eq("snn"), "model_display"] = "SNN"
    df.loc[df["model_base"].eq("transmil"), "model_display"] = "TransMIL"

    def build_config_label(row: pd.Series) -> str:
        if row["modality"] == "WSI":
            return f"WSI | {row['wsi_encoder']} | {row['model_display']}"
        if row["modality"] == "Gene":
            if row["gene_source"] == "csvraw":
                return f"Gene | csvraw | {row['model_display']}"
            return (
                f"Gene | scFoundation | {row['gene_repr']}_{row['gene_prep']} | "
                f"{row['model_display']}"
            )
        if row["modality"] == "Clinical":
            return f"Clinical | {row['pooling']} | L{int(row['layer'])} | {row['model_display']}"
        return f"{row['series']} | {row['embedding']} | {row['model']}"

    df["config_label"] = df.apply(build_config_label, axis=1)

    def row_order(row: pd.Series) -> tuple[int, int, int, int, int, int]:
        modality_order = MODALITY_ORDER.get(row["modality"], 99)
        if row["modality"] == "WSI":
            return (
                modality_order,
                WSI_EMBED_ORDER.get(row["wsi_encoder"], 99),
                MODEL_BASE_ORDER.get(row["model_base"], 99),
                0,
                0,
                0,
            )
        if row["modality"] == "Gene":
            return (
                modality_order,
                GENE_SOURCE_ORDER.get(row["gene_source"], 99),
                GENE_REPR_ORDER.get(row["gene_repr"], 99),
                GENE_PREP_ORDER.get(row["gene_prep"], 99),
                MODEL_BASE_ORDER.get(row["model_base"], 99),
                0,
            )
        if row["modality"] == "Clinical":
            return (
                modality_order,
                POOLING_ORDER.get(row["pooling"], 99),
                int(row["layer"]),
                MODEL_BASE_ORDER.get(row["model_base"], 99),
                0,
                0,
            )
        return (99, 99, 99, 99, 99, 99)

    df["row_order"] = list(map(row_order, [row for _, row in df.iterrows()]))
    df["row_order_num"] = pd.Series(range(len(df)), index=df.index)
    df = df.sort_values(["dataset", "row_order", "config_label"], kind="stable").reset_index(drop=True)
    df["dataset_rank"] = (
        df.groupby("dataset")["test_cindex_mean"].rank(ascending=False, method="min").astype(int)
    )
    return df


def collect_sample_ids(split_df: pd.DataFrame) -> set[str]:
    ids: set[str] = set()
    for col in ["train", "val", "test"]:
        if col not in split_df.columns:
            continue
        ids.update(split_df[col].dropna().astype(str).tolist())
    return ids


def build_modality_sample_stats(project_root: Path) -> pd.DataFrame:
    metadata_cache = {
        dataset: pd.read_csv(project_root / f"datasets_csv/metadata/{dataset}.csv")
        for dataset in DATASETS
    }

    records: list[dict[str, object]] = []
    results_root = project_root / "results/Single_Multi_Test"

    for series in ["Gengtest_CSVRAW", "Gengtest_F", "Clinictest_Li", "WSItest_F"]:
        series_dir = results_root / series
        modality = (
            "Gene" if series in {"Gengtest_CSVRAW", "Gengtest_F"} else
            "Clinical" if series == "Clinictest_Li" else
            "WSI"
        )

        for dataset in DATASETS:
            sample_ids: set[str] = set()
            experiment_dirs = sorted(series_dir.glob(f"{dataset}__*"))
            for experiment_dir in experiment_dirs:
                split_paths = sorted(experiment_dir.glob("*/splits_0.csv"))
                for split_path in split_paths[:1]:
                    split_df = pd.read_csv(split_path)
                    sample_ids.update(collect_sample_ids(split_df))

            if not sample_ids:
                continue

            meta_df = metadata_cache[dataset]
            subset = meta_df[meta_df["slide_id"].astype(str).isin(sample_ids)].copy()
            n = len(subset)
            censored = int(subset["censorship"].sum())
            events = int(n - censored)
            records.append(
                {
                    "dataset": dataset,
                    "dataset_label": DATASET_LABELS[dataset],
                    "series": series,
                    "modality": modality,
                    "sample_n": n,
                    "event_n": events,
                    "censor_n": censored,
                    "event_rate": events / n if n else np.nan,
                    "censor_rate": censored / n if n else np.nan,
                    "survival_months_min": subset["survival_months"].min() if n else np.nan,
                    "survival_months_median": subset["survival_months"].median() if n else np.nan,
                    "survival_months_max": subset["survival_months"].max() if n else np.nan,
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values(["dataset", "modality", "series"], kind="stable").reset_index(drop=True)
    return df


def write_dataset_fingerprint(
    df: pd.DataFrame,
    sample_stats: pd.DataFrame,
    out_dir: Path,
    internal_dir: Path,
) -> None:
    modality_stats = (
        df.groupby(["dataset", "modality"])
        .agg(
            cindex_min=("test_cindex_mean", "min"),
            cindex_max=("test_cindex_mean", "max"),
            cindex_mean=("test_cindex_mean", "mean"),
            avg_std=("test_cindex_std", "mean"),
            max_std=("test_cindex_std", "max"),
            config_n=("config_label", "count"),
        )
        .reset_index()
    )

    dataset_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        row: dict[str, object] = {
            "item_no": "1",
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
        }

        subset_samples = sample_stats[sample_stats["dataset"].eq(dataset)]
        event_rate = np.nan
        censor_rate = np.nan
        sample_n = np.nan
        event_n = np.nan
        censor_n = np.nan
        for modality in ["Gene", "Clinical", "WSI"]:
            sample_row = subset_samples[subset_samples["modality"].eq(modality)]
            if not sample_row.empty:
                sample_row = sample_row.iloc[0]
                if pd.isna(sample_n):
                    sample_n = int(sample_row["sample_n"])
                    event_n = int(sample_row["event_n"])
                    censor_n = int(sample_row["censor_n"])
                    event_rate = float(sample_row["event_rate"])
                    censor_rate = float(sample_row["censor_rate"])

            stats_row = modality_stats[
                modality_stats["dataset"].eq(dataset) & modality_stats["modality"].eq(modality)
            ]
            if not stats_row.empty:
                stats_row = stats_row.iloc[0]
                key = modality.lower()
                row[f"{key}_metric_range"] = (
                    f"{float(stats_row['cindex_min']):.3f}-{float(stats_row['cindex_max']):.3f}"
                )
                row[f"{key}_avg_std"] = float(stats_row["avg_std"])
            else:
                key = modality.lower()
                row[f"{key}_metric_range"] = "-"
                row[f"{key}_avg_std"] = np.nan

        row["sample_n"] = sample_n
        row["event_n"] = event_n
        row["censor_n"] = censor_n
        row["event_rate"] = event_rate
        row["censor_rate"] = censor_rate
        row["event_censor_ratio"] = (
            f"{int(event_n)}/{int(censor_n)}" if not pd.isna(event_n) and not pd.isna(censor_n) else "-"
        )
        dataset_rows.append(row)

    fingerprint_df = pd.DataFrame(dataset_rows)[
        [
            "item_no",
            "dataset_label",
            "sample_n",
            "event_censor_ratio",
            "event_rate",
            "censor_rate",
            "wsi_metric_range",
            "wsi_avg_std",
            "gene_metric_range",
            "gene_avg_std",
            "clinical_metric_range",
            "clinical_avg_std",
        ]
    ].rename(
        columns={
            "dataset_label": "dataset",
            "sample_n": "n",
            "event_censor_ratio": "event/censor",
            "event_rate": "event_rate",
            "censor_rate": "censor_rate",
            "wsi_metric_range": "WSI_range",
            "wsi_avg_std": "WSI_avg_std",
            "gene_metric_range": "Gene_range",
            "gene_avg_std": "Gene_avg_std",
            "clinical_metric_range": "Clinical_range",
            "clinical_avg_std": "Clinical_avg_std",
        }
    )
    fingerprint_df.to_csv(out_dir / "1_dataset_fingerprint.csv", index=False)
    sample_stats.to_csv(internal_dir / "dataset_modality_sample_stats.csv", index=False)
    modality_stats.to_csv(internal_dir / "dataset_modality_metric_stats.csv", index=False)


def write_panorama_tables(df: pd.DataFrame, out_dir: Path, internal_dir: Path) -> None:
    panorama_long = df[
        [
            "dataset",
            "dataset_label",
            "modality",
            "config_label",
            "series",
            "embedding",
            "model",
            "model_display",
            "test_cindex_mean",
            "test_cindex_std",
            "dataset_rank",
            "row_order",
        ]
    ].copy()
    panorama_long.to_csv(internal_dir / "panorama_long.csv", index=False)

    matrix_mean = (
        panorama_long.pivot(index="config_label", columns="dataset", values="test_cindex_mean")
        .reset_index()
    )
    matrix_std = (
        panorama_long.pivot(index="config_label", columns="dataset", values="test_cindex_std")
        .reset_index()
    )
    matrix_rank = (
        panorama_long.pivot(index="config_label", columns="dataset", values="dataset_rank")
        .reset_index()
    )

    config_meta = (
        panorama_long.sort_values(["row_order", "config_label"])
        .drop_duplicates("config_label")[["config_label", "modality", "row_order"]]
        .copy()
    )
    config_meta["row_order_text"] = config_meta["row_order"].astype(str)
    config_meta = config_meta.drop(columns=["row_order"])

    matrix_mean = config_meta.merge(matrix_mean, on="config_label", how="left")
    matrix_std = config_meta.merge(matrix_std, on="config_label", how="left")
    matrix_rank = config_meta.merge(matrix_rank, on="config_label", how="left")

    matrix_mean.to_csv(internal_dir / "panorama_matrix_mean.csv", index=False)
    matrix_std.to_csv(internal_dir / "panorama_matrix_std.csv", index=False)
    matrix_rank.to_csv(internal_dir / "panorama_matrix_rank.csv", index=False)

    for dataset in DATASETS:
        dataset_long = panorama_long[panorama_long["dataset"].eq(dataset)].copy()
        dataset_long = dataset_long.sort_values(["row_order", "config_label"], kind="stable")
        dataset_long.to_csv(out_dir / f"2_panorama_{dataset}.csv", index=False)


def make_pair_table(
    df: pd.DataFrame,
    family: str,
    direction: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    group_cols: list[str],
    extra_cols: list[str],
) -> pd.DataFrame:
    merged = left_df.merge(
        right_df,
        on=group_cols,
        how="inner",
        suffixes=("_left", "_right"),
    )
    merged["comparison_family"] = family
    merged["direction"] = direction
    merged["delta_cindex_mean"] = merged["test_cindex_mean_left"] - merged["test_cindex_mean_right"]
    merged["delta_cindex_std"] = merged["test_cindex_std_left"] - merged["test_cindex_std_right"]
    keep_cols = (
        ["comparison_family", "direction"] +
        group_cols +
        extra_cols +
        [
            "config_label_left",
            "config_label_right",
            "test_cindex_mean_left",
            "test_cindex_std_left",
            "test_cindex_mean_right",
            "test_cindex_std_right",
            "delta_cindex_mean",
            "delta_cindex_std",
        ]
    )
    keep_cols = [col for col in keep_cols if col in merged.columns]
    return merged[keep_cols].copy()


def matrix_from_pairs(
    pairs: pd.DataFrame,
    row_cols: list[str],
    out_path: Path,
    dataset_cols: list[str] | None = None,
) -> None:
    if dataset_cols is None:
        dataset_cols = DATASETS

    matrix = (
        pairs.pivot_table(
            index=row_cols,
            columns="dataset",
            values="delta_cindex_mean",
            aggfunc="first",
        )
        .reset_index()
    )
    for dataset in dataset_cols:
        if dataset not in matrix.columns:
            matrix[dataset] = np.nan
    matrix = matrix[row_cols + dataset_cols]

    value_cols = dataset_cols
    matrix["positive_count"] = (matrix[value_cols] > 0).sum(axis=1)
    matrix["available_n"] = matrix[value_cols].notna().sum(axis=1)
    matrix["consistency"] = matrix["positive_count"].astype(str) + "/" + matrix["available_n"].astype(str)
    matrix["mean_delta"] = matrix[value_cols].mean(axis=1)
    matrix.to_csv(out_path, index=False)


def write_delta_tables(df: pd.DataFrame, out_dir: Path, internal_dir: Path) -> None:
    gene_df = df[df["modality"].eq("Gene")].copy()
    clinical_df = df[df["modality"].eq("Clinical")].copy()
    wsi_df = df[df["modality"].eq("WSI")].copy()

    pair_tables: list[pd.DataFrame] = []

    norm_df = gene_df[gene_df["gene_prep"].eq("norm")].copy()
    raw_df = gene_df[gene_df["gene_prep"].eq("raw") & gene_df["gene_source"].eq("scFoundation")].copy()
    pairs = make_pair_table(
        df,
        "gene_norm_vs_raw",
        "norm - raw",
        norm_df,
        raw_df,
        ["dataset", "gene_repr", "model_base"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["gene_repr", "model_base", "direction"],
        out_dir / "delta_gene_norm_vs_raw.csv",
    )
    matrix_from_pairs(
        pairs,
        ["gene_repr", "model_base", "direction"],
        internal_dir / "delta_gene_norm_vs_raw.csv",
    )

    cell_df = gene_df[gene_df["gene_source"].eq("scFoundation") & gene_df["gene_repr"].eq("cell")].copy()
    gene_repr_df = gene_df[gene_df["gene_source"].eq("scFoundation") & gene_df["gene_repr"].eq("gene")].copy()
    pairs = make_pair_table(
        df,
        "gene_cell_vs_gene",
        "cell - gene",
        cell_df,
        gene_repr_df,
        ["dataset", "gene_prep", "model_base"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["gene_prep", "model_base", "direction"],
        out_dir / "delta_gene_cell_vs_gene.csv",
    )
    matrix_from_pairs(
        pairs,
        ["gene_prep", "model_base", "direction"],
        internal_dir / "delta_gene_cell_vs_gene.csv",
    )

    foundation_fair = gene_df[
        gene_df["gene_source"].eq("scFoundation") &
        gene_df["gene_repr"].eq("gene") &
        gene_df["gene_prep"].eq("norm")
    ].copy()
    csvraw_df = gene_df[gene_df["gene_source"].eq("csvraw")].copy()
    pairs = make_pair_table(
        df,
        "gene_scfoundation_vs_csvraw",
        "scFoundation(gene_norm) - csvraw",
        foundation_fair,
        csvraw_df,
        ["dataset", "model_base"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["model_base", "direction"],
        out_dir / "delta_gene_scfoundation_vs_csvraw.csv",
    )
    matrix_from_pairs(
        pairs,
        ["model_base", "direction"],
        internal_dir / "delta_gene_scfoundation_vs_csvraw.csv",
    )

    uni_v2_df = wsi_df[wsi_df["wsi_encoder"].eq("uni_v2")].copy()
    uni_v1_df = wsi_df[wsi_df["wsi_encoder"].eq("uni_v1")].copy()
    pairs = make_pair_table(
        df,
        "wsi_uni_v2_vs_v1",
        "uni_v2 - uni_v1",
        uni_v2_df,
        uni_v1_df,
        ["dataset", "model_base"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["model_base", "direction"],
        out_dir / "delta_wsi_uni_v2_vs_v1.csv",
    )
    matrix_from_pairs(
        pairs,
        ["model_base", "direction"],
        internal_dir / "delta_wsi_uni_v2_vs_v1.csv",
    )

    flatten_df = clinical_df[clinical_df["pooling"].eq("flatten")].copy()
    mean_df = clinical_df[clinical_df["pooling"].eq("mean")].copy()
    pairs = make_pair_table(
        df,
        "clinical_flatten_vs_mean",
        "flatten - mean",
        flatten_df,
        mean_df,
        ["dataset", "layer", "model_base"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["layer", "model_base", "direction"],
        out_dir / "delta_clinical_flatten_vs_mean.csv",
    )
    matrix_from_pairs(
        pairs,
        ["layer", "model_base", "direction"],
        internal_dir / "delta_clinical_flatten_vs_mean.csv",
    )

    gene_snn_df = gene_df[gene_df["model_base"].eq("snn")].copy()
    gene_mlp_df = gene_df[gene_df["model_base"].eq("mlp")].copy()
    pairs = make_pair_table(
        df,
        "model_snn_vs_mlp_gene",
        "snn - mlp",
        gene_snn_df,
        gene_mlp_df,
        ["dataset", "gene_source", "gene_repr", "gene_prep"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["gene_source", "gene_repr", "gene_prep", "direction"],
        out_dir / "delta_model_snn_vs_mlp_gene.csv",
    )
    matrix_from_pairs(
        pairs,
        ["gene_source", "gene_repr", "gene_prep", "direction"],
        internal_dir / "delta_model_snn_vs_mlp_gene.csv",
    )

    clinical_snn_df = clinical_df[clinical_df["model_base"].eq("snn")].copy()
    clinical_mlp_df = clinical_df[clinical_df["model_base"].eq("mlp")].copy()
    pairs = make_pair_table(
        df,
        "model_snn_vs_mlp_clinical",
        "snn - mlp",
        clinical_snn_df,
        clinical_mlp_df,
        ["dataset", "pooling", "layer"],
        [],
    )
    pair_tables.append(pairs)
    matrix_from_pairs(
        pairs,
        ["pooling", "layer", "direction"],
        out_dir / "delta_model_snn_vs_mlp_clinical.csv",
    )
    matrix_from_pairs(
        pairs,
        ["pooling", "layer", "direction"],
        internal_dir / "delta_model_snn_vs_mlp_clinical.csv",
    )

    delta_long = pd.concat(pair_tables, ignore_index=True)
    delta_long.to_csv(internal_dir / "delta_comparisons_long.csv", index=False)


def write_clinical_trend_tables(df: pd.DataFrame, out_dir: Path, internal_dir: Path) -> None:
    clinical_df = df[df["modality"].eq("Clinical")].copy()
    clinical_df["layer"] = pd.to_numeric(clinical_df["layer"], errors="coerce")
    clinical_df["test_cindex_mean"] = pd.to_numeric(clinical_df["test_cindex_mean"], errors="coerce")
    clinical_df["test_cindex_std"] = pd.to_numeric(clinical_df["test_cindex_std"], errors="coerce")
    trend_long = clinical_df[
        [
            "dataset",
            "dataset_label",
            "model_base",
            "model_display",
            "pooling",
            "layer",
            "config_label",
            "test_cindex_mean",
            "test_cindex_std",
        ]
    ].copy()
    trend_long = trend_long.sort_values(
        ["dataset", "model_base", "pooling", "layer"],
        kind="stable",
    )
    trend_long.to_csv(internal_dir / "clinical_layer_trend_long.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for (dataset, model_base, pooling), group in trend_long.groupby(["dataset", "model_base", "pooling"]):
        group = group.sort_values("layer", kind="stable")
        if group["layer"].isna().any():
            continue
        rho = group["layer"].corr(group["test_cindex_mean"], method="spearman")
        slope = np.polyfit(group["layer"], group["test_cindex_mean"], deg=1)[0]
        best_row = group.loc[group["test_cindex_mean"].idxmax()]
        summary_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model_base": model_base,
                "pooling": pooling,
                "spearman_rho": rho,
                "linear_slope": slope,
                "best_layer": int(best_row["layer"]),
                "best_cindex_mean": float(best_row["test_cindex_mean"]),
                "best_cindex_std": float(best_row["test_cindex_std"]),
            }
        )
        best_rows.append(
            {
                "dataset": dataset,
                "dataset_label": DATASET_LABELS[dataset],
                "model_base": model_base,
                "pooling": pooling,
                "best_layer": int(best_row["layer"]),
                "best_config_label": best_row["config_label"],
                "best_cindex_mean": float(best_row["test_cindex_mean"]),
                "best_cindex_std": float(best_row["test_cindex_std"]),
            }
        )

    pd.DataFrame(summary_rows).sort_values(
        ["dataset", "model_base", "pooling"], kind="stable"
    ).to_csv(out_dir / "4_clinical_layer_spearman_summary.csv", index=False)
    pd.DataFrame(best_rows).sort_values(
        ["dataset", "model_base", "pooling"], kind="stable"
    ).to_csv(out_dir / "4_clinical_best_layer_summary.csv", index=False)


def write_average_rank_tables(df: pd.DataFrame, out_dir: Path, internal_dir: Path) -> None:
    rank_long = df[
        [
            "dataset",
            "dataset_label",
            "modality",
            "config_label",
            "test_cindex_mean",
            "test_cindex_std",
            "dataset_rank",
        ]
    ].copy()
    rank_long.to_csv(internal_dir / "average_rank_long.csv", index=False)

    rank_summary = (
        rank_long.groupby(["modality", "config_label"])
        .agg(
            average_rank=("dataset_rank", "mean"),
            rank_std=("dataset_rank", "std"),
            average_cindex=("test_cindex_mean", "mean"),
            average_metric_std=("test_cindex_std", "mean"),
        )
        .reset_index()
    )

    rank_matrix = rank_long.pivot(index="config_label", columns="dataset", values="dataset_rank").reset_index()
    rank_summary = rank_summary.merge(rank_matrix, on="config_label", how="left")
    rank_summary = rank_summary.sort_values(
        ["average_rank", "modality", "config_label"],
        kind="stable",
    )
    rank_summary.to_csv(out_dir / "5_average_rank_all_configs.csv", index=False)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "results_display/Single_Multi_Test/Sum"
    internal_dir = out_dir / "_internal"
    out_dir.mkdir(parents=True, exist_ok=True)
    internal_dir.mkdir(parents=True, exist_ok=True)

    analysis_df = build_analysis_frame(project_root)
    sample_stats = build_modality_sample_stats(project_root)

    analysis_df.to_csv(internal_dir / "analysis_master_long.csv", index=False)
    write_dataset_fingerprint(analysis_df, sample_stats, out_dir, internal_dir)
    write_panorama_tables(analysis_df, out_dir, internal_dir)
    write_delta_tables(analysis_df, out_dir, internal_dir)
    write_clinical_trend_tables(analysis_df, out_dir, internal_dir)
    write_average_rank_tables(analysis_df, out_dir, internal_dir)

    print(f"Wrote analysis CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
