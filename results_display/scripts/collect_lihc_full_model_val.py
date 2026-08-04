"""
Collect LIHC full-model validation results into one CSV table.

Scans:
    results/LIHC_full_model_val/{experiment}/{model}/test_result.csv

Outputs:
    results_display/LIHC_full_model_val/summary/lihc_full_model_val_cindex_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


MODEL_TYPE_MAP = {
    "abmil_wsi": "WSI",
    "mlp_wsi": "WSI",
    "transmil_wsi": "WSI",
    "clinic_cox": "Clinical",
    "mlp_clinic_mean": "Clinical",
    "mlp_clinic_flatten": "Clinical",
    "snn_clinic_mean": "Clinical",
    "snn_clinic_flatten": "Clinical",
    "mlp_gene": "Gene",
    "snn_gene": "Gene",
    "mlp_gene_f": "Gene",
    "snn_gene_f": "Gene",
    "survpc_f": "Multi",
    "porpoise": "Multi",
    "survpath": "Multi",
    "mcat": "Multi",
    "survgc_f": "Multi",
    "survpgc_f": "Multi",
}

TYPE_ORDER = {
    "WSI": 0,
    "Clinical": 1,
    "Gene": 2,
    "Multi": 3,
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cindex_stats(csv_path: Path) -> tuple[float, float, int] | None:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values: list[float] = []
            for row in reader:
                raw = row.get("test_cindex", "")
                if raw is None or raw == "":
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
    except Exception as exc:
        print(f"[WARN] Cannot read {csv_path}: {exc}")
        return None

    if len(values) == 0:
        print(f"[WARN] Empty test_cindex in {csv_path}")
        return None

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return mean, std, len(values)


def collect(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for experiment_dir in sorted(results_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue

        for model_dir in sorted(experiment_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            csv_path = model_dir / "test_result.csv"
            model = model_dir.name
            model_type = MODEL_TYPE_MAP.get(model, "Other")
            if not csv_path.exists():
                print(f"[WARN] Missing test_result.csv: {model_dir}")
                rows.append(
                    {
                        "study": "tcga_lihc",
                        "experiment": experiment_dir.name,
                        "model_type": model_type,
                        "model": model,
                        "test_cindex_mean": "",
                        "test_cindex_5fold_std": "",
                        "fold_num": 0,
                        "test_cindex_mean_std": "-",
                        "status": "missing_test_result",
                    }
                )
                continue

            stats = load_cindex_stats(csv_path)
            if stats is None:
                rows.append(
                    {
                        "study": "tcga_lihc",
                        "experiment": experiment_dir.name,
                        "model_type": model_type,
                        "model": model,
                        "test_cindex_mean": "",
                        "test_cindex_5fold_std": "",
                        "fold_num": 0,
                        "test_cindex_mean_std": "-",
                        "status": "unreadable_test_result",
                    }
                )
                continue

            mean, std, fold_num = stats
            rows.append(
                {
                    "study": "tcga_lihc",
                    "experiment": experiment_dir.name,
                    "model_type": model_type,
                    "model": model,
                    "test_cindex_mean": mean,
                    "test_cindex_5fold_std": std,
                    "fold_num": fold_num,
                    "test_cindex_mean_std": f"{mean:.4f} ± {std:.4f}",
                    "status": "ok",
                }
            )

    if not rows:
        return []

    return sorted(
        rows,
        key=lambda row: (
            TYPE_ORDER.get(str(row["model_type"]), 99),
            str(row["experiment"]),
            str(row["model"]),
        ),
    )


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect LIHC full-model validation cindex summary")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=project_root / "results/LIHC_full_model_val",
        help="LIHC full-model validation results directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=project_root / "results_display/LIHC_full_model_val/summary",
        help="Summary CSV output directory",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = collect(args.results_dir)
    if not summary_rows:
        raise SystemExit(f"No usable results found under {args.results_dir}")

    out_path = args.output_dir / "lihc_full_model_val_cindex_summary.csv"
    fieldnames = [
        "study",
        "experiment",
        "model_type",
        "model",
        "test_cindex_mean",
        "test_cindex_5fold_std",
        "fold_num",
        "test_cindex_mean_std",
        "status",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[WRITE] {out_path.relative_to(project_root)}")
    print("Done.")


if __name__ == "__main__":
    main()
