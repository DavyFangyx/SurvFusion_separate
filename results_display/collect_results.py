"""
Collect test_result.csv files under results/ and export summary CSVs.

Supported directory structure:
    results/{series}/{dataset__embedding}/{model}/test_result.csv

For each series, this script outputs into results_display/{series}/:

1. Per-experiment table:
   {dataset__embedding}.csv
   rows = models
   cols = mean of requested metrics across folds

2. Pivot summary table:
   {series}_summary.csv
   rows = (model_type, model)
   cols = {dataset__embedding}
   values = test_cindex formatted as "mean ± std"

3. Flat combined table:
   {series}_combined.csv
   rows = one row per (dataset, embedding, model)
   cols = dataset / embedding / model / model_type / fold_num / metrics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- model type mapping ----------
MODEL_TYPE_MAP: dict[str, str] = {
    # 单模态 G
    "mlp_gene": "单模态 G",
    "snn_gene": "单模态 G",
    "mlp_gene_f": "单模态 G",
    "snn_gene_f": "单模态 G",
    # 单模态 WSI
    "abmil_wsi": "单模态 WSI",
    "mlp_wsi": "单模态 WSI",
    "transmil_wsi": "单模态 WSI",
    # 单模态 C
    "mlp_clinic_mean": "单模态 C",
    "mlp_clinic_flatten": "单模态 C",
    "snn_clinic_mean": "单模态 C",
    "snn_clinic_flatten": "单模态 C",
    "clinic_cox": "单模态 C",
    # 多模态基线 WSI+G
    "porpoise": "多模态 WSI+G",
    "survpath": "多模态 WSI+G",
    "mcat": "多模态 WSI+G",
    # 主模型 WSI+G+C
    "survpgc_f": "多模态 WSI+G+C",
    "survfusion_separate": "多模态 WSI+G+C",
    # 消融 WSI+C
    "survpc_f": "消融 WSI+C",
    # 消融 G+C
    "survgc_f": "消融 G+C",
}

TYPE_ORDER = [
    "单模态 G",
    "单模态 WSI",
    "单模态 C",
    "多模态 WSI+G",
    "多模态 WSI+G+C",
    "消融 WSI+C",
    "消融 G+C",
]

SCALAR_METRICS = [
    "test_cindex",
    "test_cindex_ipcw",
    "test_IBS",
    "test_iauc",
    "test_loss",
]

TARGET_SERIES = [
    "Clinictest_Li",
    "Gengtest_CSVRAW",
    "Gengtest_F",
    "WSItest_F",
]


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _sort_key_model_type(model_type: str) -> tuple[int, str]:
    return (
        TYPE_ORDER.index(model_type) if model_type in TYPE_ORDER else 99,
        model_type,
    )


def parse_experiment_name(exp_name: str) -> tuple[str, str]:
    """Split {dataset}__{embedding}. If no separator, embedding = 'default'."""
    if "__" in exp_name:
        dataset, embedding = exp_name.split("__", 1)
        return dataset, embedding
    return exp_name, "default"


def load_result_df(csv_path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        print(f"  [WARN] Cannot read {csv_path}: {e}")
        return None


def load_fold_values(csv_path: Path, metric: str) -> np.ndarray | None:
    df = load_result_df(csv_path)
    if df is None or metric not in df.columns:
        return None
    vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
    return vals if len(vals) > 0 else None


def load_all_metrics(csv_path: Path, metrics: list[str]) -> dict[str, float] | None:
    df = load_result_df(csv_path)
    if df is None:
        return None

    row: dict[str, float] = {}
    for metric in metrics:
        if metric not in df.columns:
            continue
        vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
        if len(vals) > 0:
            row[metric] = float(np.mean(vals))
    return row if row else None


def build_experiment_table(exp_dir: Path, metrics: list[str]) -> pd.DataFrame | None:
    records = []
    for model_dir in sorted(exp_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "test_result.csv"
        if not csv_path.exists():
            continue
        row = load_all_metrics(csv_path, metrics)
        if row is None:
            continue
        row["model"] = model_dir.name
        records.append(row)

    if not records:
        return None

    df = pd.DataFrame(records).set_index("model")
    for col in df.columns:
        df[col] = df[col].apply(lambda v: f"{v:.4f}")
    return df.sort_index()


def build_series_summary(series_dir: Path, experiments: list[str]) -> pd.DataFrame:
    data: dict[tuple[str, str], dict[str, str]] = {}

    for exp_name in experiments:
        exp_dir = series_dir / exp_name
        if not exp_dir.is_dir():
            continue
        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            csv_path = model_dir / "test_result.csv"
            if not csv_path.exists():
                continue

            model = model_dir.name
            vals = load_fold_values(csv_path, "test_cindex")
            if vals is None:
                continue

            model_type = MODEL_TYPE_MAP.get(model, "其他")
            key = (model_type, model)
            data.setdefault(key, {})[exp_name] = _fmt(float(np.mean(vals)), float(np.std(vals)))

    if not data:
        return pd.DataFrame()

    rows = []
    for model_type, model in sorted(
        data.keys(),
        key=lambda x: (_sort_key_model_type(x[0]), x[1]),
    ):
        row = {"model_type": model_type, "model": model}
        for exp_name in experiments:
            row[exp_name] = data[(model_type, model)].get(exp_name, "-")
        rows.append(row)

    return pd.DataFrame(rows).set_index(["model_type", "model"])


def build_series_combined(series_dir: Path, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for exp_dir in sorted(series_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        exp_name = exp_dir.name
        dataset, embedding = parse_experiment_name(exp_name)

        for model_dir in sorted(exp_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            csv_path = model_dir / "test_result.csv"
            if not csv_path.exists():
                continue

            df = load_result_df(csv_path)
            if df is None:
                continue

            row: dict[str, object] = {
                "series": series_dir.name,
                "experiment": exp_name,
                "dataset": dataset,
                "embedding": embedding,
                "model_type": MODEL_TYPE_MAP.get(model_dir.name, "其他"),
                "model": model_dir.name,
                "fold_num": len(df),
            }

            for metric in metrics:
                if metric not in df.columns:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
                    row[metric] = "-"
                    continue

                vals = pd.to_numeric(df[metric], errors="coerce").dropna().values
                if len(vals) == 0:
                    row[f"{metric}_mean"] = np.nan
                    row[f"{metric}_std"] = np.nan
                    row[metric] = "-"
                    continue

                mean = float(np.mean(vals))
                std = float(np.std(vals))
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[metric] = _fmt(mean, std)

            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["model_type_order"] = df["model_type"].apply(
        lambda x: TYPE_ORDER.index(x) if x in TYPE_ORDER else 99
    )
    df = df.sort_values(
        by=["dataset", "embedding", "model_type_order", "model"],
        kind="stable",
    ).drop(columns=["model_type_order"])

    front_cols = [
        "series",
        "experiment",
        "dataset",
        "embedding",
        "model_type",
        "model",
        "fold_num",
    ]
    metric_cols = []
    for metric in metrics:
        metric_cols.extend([metric, f"{metric}_mean", f"{metric}_std"])

    existing_cols = [col for col in front_cols + metric_cols if col in df.columns]
    return df[existing_cols]


def collect(results_dir: Path, output_dir: Path, metrics: list[str]) -> None:
    for series_name in TARGET_SERIES:
        series_dir = results_dir / series_name
        if not series_dir.is_dir():
            print(f"[SKIP] Missing series: {series_dir}")
            continue

        series_out = output_dir / series_name
        series_out.mkdir(parents=True, exist_ok=True)

        experiments = sorted([p.name for p in series_dir.iterdir() if p.is_dir()])
        if not experiments:
            print(f"[SKIP] Empty series: {series_dir}")
            continue

        print(f"[SERIES] {series_name}")

        for exp_name in experiments:
            exp_dir = series_dir / exp_name
            df = build_experiment_table(exp_dir, metrics)
            if df is None:
                continue
            out_path = series_out / f"{exp_name}.csv"
            df.to_csv(out_path)
            print(f"  [L1] {out_path.relative_to(output_dir.parent)}")

        summary_df = build_series_summary(series_dir, experiments)
        if not summary_df.empty:
            summary_path = series_out / f"{series_name}_summary.csv"
            summary_df.to_csv(summary_path)
            print(f"  [L2] {summary_path.relative_to(output_dir.parent)}")

        combined_df = build_series_combined(series_dir, metrics)
        if not combined_df.empty:
            combined_path = series_out / f"{series_name}_combined.csv"
            combined_df.to_csv(combined_path, index=False)
            print(f"  [L3] {combined_path.relative_to(output_dir.parent)}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Collect experiment result summaries")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=root / "results",
        help="Path to results directory (default: <project_root>/results)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=root / "results_display",
        help="Path to output directory (default: <project_root>/results_display)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=SCALAR_METRICS,
        help="Metrics to include in exported CSVs",
    )
    args = parser.parse_args()

    print(f"Scanning: {args.results_dir}")
    print(f"Output:   {args.output_dir}")
    print(f"Metrics:  {args.metrics}")
    print(f"Series:   {TARGET_SERIES}\n")

    collect(args.results_dir, args.output_dir, args.metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
