"""
Collect single-modal result summaries for downstream plotting.

Scans the following directory structure by default:
    results/{series}/{dataset__input}/{model}/test_result.csv

Exports one long-format CSV:
    results_display/Single_Modal_Trend/summary/single_modal_cindex_long.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dataset_deployment.registry import get_dataset_config, list_enabled_studies
except Exception:
    get_dataset_config = None
    list_enabled_studies = None


DEFAULT_SERIES = ["Clinictest_Li", "WSItest_F", "Genetest"]
METRIC = "test_cindex"

SERIES_MODALITY = {
    "Clinictest_Li": "Clinical",
    "WSItest_F": "WSI",
    "Genetest": "Gene",
}

MODEL_META = {
    "Clinictest_Li": {
        "clinic_cox": {"model_key": "Cox", "model_label": "Cox", "model_order": 0},
        "mlp_clinic_mean": {
            "model_key": "MLP-mean",
            "model_label": "MLP mean",
            "model_order": 1,
        },
        "mlp_clinic_flatten": {
            "model_key": "MLP-flatten",
            "model_label": "MLP flatten",
            "model_order": 2,
        },
        "snn_clinic_mean": {
            "model_key": "SNN-mean",
            "model_label": "SNN mean",
            "model_order": 3,
        },
        "snn_clinic_flatten": {
            "model_key": "SNN-flatten",
            "model_label": "SNN flatten",
            "model_order": 4,
        },
    },
    "WSItest_F": {
        "mlp_wsi": {"model_key": "MLP", "model_label": "MLP", "model_order": 0},
        "abmil_wsi": {"model_key": "ABMIL", "model_label": "ABMIL", "model_order": 1},
        "transmil_wsi": {
            "model_key": "TransMIL",
            "model_label": "TransMIL",
            "model_order": 2,
        },
    },
    "Genetest": {
        "mlp_gene": {"model_key": "MLP", "model_label": "MLP", "model_order": 0},
        "mlp_gene_f": {"model_key": "MLP", "model_label": "MLP", "model_order": 0},
        "snn_gene": {"model_key": "SNN", "model_label": "SNN", "model_order": 1},
        "snn_gene_f": {"model_key": "SNN", "model_label": "SNN", "model_order": 1},
    },
}

GENE_INPUT_LABELS = {
    "csvraw": "CSV raw",
    "scFoundation_embedding_gene_raw": "scF gene raw",
    "scFoundation_embedding_gene_norm": "scF gene norm",
    "scFoundation_embedding_cell_raw": "scF cell raw",
    "scFoundation_embedding_cell_norm": "scF cell norm",
}

GENE_INPUT_ORDER = {
    "csvraw": 0,
    "scFoundation_embedding_gene_raw": 1,
    "scFoundation_embedding_gene_norm": 2,
    "scFoundation_embedding_cell_raw": 3,
    "scFoundation_embedding_cell_norm": 4,
}

WSI_INPUT_LABELS = {
    "uni_v1": "UNI v1",
    "uni_v2": "UNI v2",
}


def project_root_from_script() -> Path:
    return PROJECT_ROOT


def dataset_order_map() -> dict[str, int]:
    if list_enabled_studies is not None:
        studies = list_enabled_studies()
        return {study: idx for idx, study in enumerate(studies)}
    return {}


DATASET_ORDER = dataset_order_map()


def dataset_label(dataset: str) -> str:
    if get_dataset_config is not None:
        try:
            label = get_dataset_config(dataset).display_name
            return label.replace("TCGA-", "").replace("TCGA_", "")
        except Exception:
            pass
    return dataset.replace("tcga_", "").upper()


def parse_experiment_name(experiment: str) -> tuple[str, str]:
    if "__" not in experiment:
        return experiment, "default"
    dataset, input_key = experiment.split("__", 1)
    return dataset, input_key


def infer_input_meta(series: str, input_key: str) -> tuple[str, int]:
    if series == "Clinictest_Li":
        match = re.fullmatch(r"L(\d+)", input_key)
        if match:
            idx = int(match.group(1))
            return input_key, idx
        return input_key, 999

    if series == "WSItest_F":
        if input_key in WSI_INPUT_LABELS:
            version = int(input_key.split("_v")[-1])
            return WSI_INPUT_LABELS[input_key], version - 1
        return input_key.replace("_", " "), 999

    if series == "Genetest":
        label = GENE_INPUT_LABELS.get(input_key, input_key.replace("_", " "))
        order = GENE_INPUT_ORDER.get(input_key, 999)
        return label, order

    return input_key, 999


def infer_model_meta(series: str, model_raw: str) -> dict[str, object]:
    series_meta = MODEL_META.get(series, {})
    meta = series_meta.get(model_raw)
    if meta is not None:
        return meta
    return {
        "model_key": model_raw,
        "model_label": model_raw,
        "model_order": 999,
    }


def load_metric(csv_path: Path, metric: str) -> tuple[float, float, int] | None:
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as exc:
        print(f"[WARN] Cannot read {csv_path}: {exc}")
        return None

    if metric not in df.columns:
        print(f"[WARN] Missing {metric} in {csv_path}")
        return None

    values = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
    if len(values) == 0:
        print(f"[WARN] Empty {metric} in {csv_path}")
        return None

    return float(np.mean(values)), float(np.std(values)), int(len(values))


def collect_rows(results_dir: Path, series_names: list[str], metric: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for series in series_names:
        series_dir = results_dir / series
        if not series_dir.is_dir():
            print(f"[SKIP] Missing series directory: {series_dir}")
            continue

        print(f"[SERIES] {series}")

        for exp_dir in sorted(series_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            dataset, input_key = parse_experiment_name(exp_dir.name)
            input_label, input_order = infer_input_meta(series, input_key)

            for model_dir in sorted(exp_dir.iterdir()):
                if not model_dir.is_dir():
                    continue

                csv_path = model_dir / "test_result.csv"
                if not csv_path.exists():
                    continue

                metric_stats = load_metric(csv_path, metric)
                if metric_stats is None:
                    continue

                mean, std, fold_num = metric_stats
                model_meta = infer_model_meta(series, model_dir.name)
                rows.append(
                    {
                        "series": series,
                        "modality": SERIES_MODALITY.get(series, series),
                        "experiment": exp_dir.name,
                        "dataset": dataset,
                        "dataset_label": dataset_label(dataset),
                        "dataset_order": DATASET_ORDER.get(dataset, 999),
                        "input_key": input_key,
                        "input_label": input_label,
                        "input_order": input_order,
                        "model_raw": model_dir.name,
                        "model_key": model_meta["model_key"],
                        "model_label": model_meta["model_label"],
                        "model_order": model_meta["model_order"],
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "fold_num": fold_num,
                    }
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df.sort_values(
        by=[
            "series",
            "dataset_order",
            "dataset",
            "model_order",
            "model_key",
            "input_order",
            "input_key",
        ],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect single-modal trend summaries")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=project_root / "results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display/Single_Modal_Trend/summary",
        help="Directory for summary CSV outputs",
    )
    parser.add_argument(
        "--series",
        nargs="+",
        default=DEFAULT_SERIES,
        help="Single-modal result series to scan",
    )
    parser.add_argument(
        "--metric",
        default=METRIC,
        help="Metric column to summarize from test_result.csv",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scanning: {args.results_dir}")
    print(f"Series:   {args.series}")
    print(f"Metric:   {args.metric}")
    print(f"Output:   {args.output_dir}\n")

    df = collect_rows(args.results_dir, args.series, args.metric)
    if df.empty:
        print("No result rows collected.")
        return

    out_path = args.output_dir / "single_modal_cindex_long.csv"
    df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(project_root)}")
    print("Done.")


if __name__ == "__main__":
    main()
