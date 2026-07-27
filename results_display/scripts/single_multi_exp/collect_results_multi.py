"""
Collect Multi_model_test results and export summary CSVs.

Supported directory structure:
    results/Multi_model_test/{condition}/{model}/test_result.csv

Where condition looks like:
    {study}__{clinic}__{gene}__{wsi}__h{heads}

This script outputs into results_display/Multi_model_test/:

1. Per-condition table:
   {condition}.csv
   rows = models
   cols = mean of requested metrics across folds

2. Dataset-head summary table:
   Multi_model_test_summary.csv
   rows = (model_type, model)
   cols = {study}__h{heads}
   values = test_cindex formatted as "mean ± std"

3. Flat combined table:
   Multi_model_test_combined.csv
   rows = one row per (condition, model)
   cols = study / heads / condition / model / model_type / fold_num / metrics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_TYPE_MAP: dict[str, str] = {
    "survpgc_f": "主模型",
    "survtri_mlp_concat": "三模态消融",
    "survtri_mlp_mhsa": "三模态消融",
    "survtri_snn_concat": "三模态消融",
    "survtri_snn_mhsa": "三模态消融",
}

TYPE_ORDER = [
    "主模型",
    "三模态消融",
]

MODEL_ORDER = [
    "survpgc_f",
    "survtri_mlp_concat",
    "survtri_mlp_mhsa",
    "survtri_snn_concat",
    "survtri_snn_mhsa",
]

SCALAR_METRICS = [
    "test_cindex",
    "test_cindex_ipcw",
    "test_IBS",
    "test_iauc",
    "test_loss",
]


def _fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def _sort_key_model_type(model_type: str) -> tuple[int, str]:
    return (
        TYPE_ORDER.index(model_type) if model_type in TYPE_ORDER else 99,
        model_type,
    )


def _sort_key_model(model: str) -> tuple[int, str]:
    return (
        MODEL_ORDER.index(model) if model in MODEL_ORDER else 99,
        model,
    )


def _sort_key_condition(condition: str) -> tuple[str, int, str]:
    parts = condition.split("__")
    study = parts[0] if parts else condition
    heads = extract_heads(condition)
    return (study, heads if heads is not None else 999, condition)


def parse_condition_name(condition: str) -> dict[str, object]:
    parts = condition.split("__")
    parsed: dict[str, object] = {
        "condition": condition,
        "study": parts[0] if len(parts) > 0 else condition,
        "clinic": parts[1] if len(parts) > 1 else "",
        "gene": parts[2] if len(parts) > 2 else "",
        "wsi": parts[3] if len(parts) > 3 else "",
        "heads": extract_heads(condition),
    }
    parsed["study_heads"] = (
        f"{parsed['study']}__h{int(parsed['heads']):02d}"
        if parsed["heads"] is not None
        else str(parsed["study"])
    )
    return parsed


def extract_heads(condition: str) -> int | None:
    for part in reversed(condition.split("__")):
        if part.startswith("h") and part[1:].isdigit():
            return int(part[1:])
    return None


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


def build_condition_table(condition_dir: Path, metrics: list[str]) -> pd.DataFrame | None:
    records = []
    for model_dir in sorted(condition_dir.iterdir(), key=lambda p: _sort_key_model(p.name)):
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
    return df.sort_index(key=lambda idx: idx.map(lambda x: _sort_key_model(str(x))))


def build_summary(results_dir: Path, conditions: list[str]) -> pd.DataFrame:
    data: dict[tuple[str, str], dict[str, str]] = {}
    summary_cols: list[str] = []

    for condition in conditions:
        condition_dir = results_dir / condition
        if not condition_dir.is_dir():
            continue

        parsed = parse_condition_name(condition)
        summary_col = str(parsed["study_heads"])
        if summary_col not in summary_cols:
            summary_cols.append(summary_col)

        for model_dir in sorted(condition_dir.iterdir(), key=lambda p: _sort_key_model(p.name)):
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
            data.setdefault(key, {})[summary_col] = _fmt(
                float(np.mean(vals)),
                float(np.std(vals)),
            )

    if not data:
        return pd.DataFrame()

    rows = []
    for model_type, model in sorted(
        data.keys(),
        key=lambda x: (_sort_key_model_type(x[0]), _sort_key_model(x[1])),
    ):
        row = {"model_type": model_type, "model": model}
        for summary_col in summary_cols:
            row[summary_col] = data[(model_type, model)].get(summary_col, "-")
        rows.append(row)

    return pd.DataFrame(rows).set_index(["model_type", "model"])


def build_combined(results_dir: Path, conditions: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for condition in conditions:
        condition_dir = results_dir / condition
        if not condition_dir.is_dir():
            continue

        parsed = parse_condition_name(condition)

        for model_dir in sorted(condition_dir.iterdir(), key=lambda p: _sort_key_model(p.name)):
            if not model_dir.is_dir():
                continue

            csv_path = model_dir / "test_result.csv"
            if not csv_path.exists():
                continue

            df = load_result_df(csv_path)
            if df is None:
                continue

            row: dict[str, object] = {
                "series": results_dir.name,
                "condition": condition,
                "study": parsed["study"],
                "clinic": parsed["clinic"],
                "gene": parsed["gene"],
                "wsi": parsed["wsi"],
                "heads": parsed["heads"],
                "study_heads": parsed["study_heads"],
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
    df["model_order"] = df["model"].apply(
        lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 99
    )
    df = df.sort_values(
        by=["study", "heads", "model_type_order", "model_order", "model"],
        kind="stable",
    ).drop(columns=["model_type_order", "model_order"])

    front_cols = [
        "series",
        "condition",
        "study",
        "clinic",
        "gene",
        "wsi",
        "heads",
        "study_heads",
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
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(
        [p.name for p in results_dir.iterdir() if p.is_dir()],
        key=_sort_key_condition,
    )
    if not conditions:
        print(f"[SKIP] Empty results dir: {results_dir}")
        return

    print(f"[SERIES] {results_dir.name}")

    for condition in conditions:
        condition_dir = results_dir / condition
        df = build_condition_table(condition_dir, metrics)
        if df is None:
            continue
        out_path = output_dir / f"{condition}.csv"
        df.to_csv(out_path)
        print(f"  [L1] {out_path.relative_to(output_dir.parent)}")

    summary_df = build_summary(results_dir, conditions)
    if not summary_df.empty:
        summary_path = output_dir / f"{results_dir.name}_summary.csv"
        summary_df.to_csv(summary_path)
        print(f"  [L2] {summary_path.relative_to(output_dir.parent)}")

    combined_df = build_combined(results_dir, conditions, metrics)
    if not combined_df.empty:
        combined_path = output_dir / f"{results_dir.name}_combined.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"  [L3] {combined_path.relative_to(output_dir.parent)}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Collect Multi_model_test result summaries")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=project_root / "results" / "Multi_model_test",
        help="Path to Multi_model_test directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display" / "Multi_model_test",
        help="Path to output directory",
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
    print(f"Metrics:  {args.metrics}\n")

    collect(args.results_dir, args.output_dir, args.metrics)
    print("\nDone.")


if __name__ == "__main__":
    main()
