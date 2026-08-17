"""
Collect Table 1 c-index summaries for L0 baselines and ours.

Inputs:
    results/Table1_Cindex_Main/L0_*_full_model_val
    results/Table1_Cindex_Main/L0_*_poe_model_val

Outputs:
    results_display/Table1_Cindex_Main/L0/summary.csv
    results_display/Table1_Cindex_Main/L0/baselines/summary/
    results_display/Table1_Cindex_Main/L0/ours/summary/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


GROUP_DIR = "Table1_Cindex_Main"
LAYER_DIR = "L0"

STUDY_SPECS = [
    ("BRCA", "tcga_brca"),
    ("COAD", "tcga_coad"),
    ("KICH", "tcga_kich"),
    ("KIRC", "tcga_kirc"),
    ("KIRP", "tcga_kirp"),
    ("LIHC", "tcga_lihc"),
    ("PRAD", "tcga_prad"),
    ("READ", "tcga_read"),
]

BASELINE_MODEL_SPECS = [
    ("P", "abmil_wsi"),
    ("P", "mlp_wsi"),
    ("P", "transmil_wsi"),
    ("C", "clinic_cox"),
    ("C", "mlp_clinic_mean"),
    ("C", "mlp_clinic_flatten"),
    ("C", "snn_clinic_mean"),
    ("C", "snn_clinic_flatten"),
    ("G", "mlp_gene"),
    ("G", "snn_gene"),
    ("G", "mlp_gene_f"),
    ("G", "snn_gene_f"),
    ("P+C", "survpc_f"),
    ("P+G", "porpoise"),
    ("P+G", "survpath"),
    ("P+G", "mcat"),
    ("C+G", "survgc_f"),
    ("P+C+G", "survpgc_f"),
]

OURS_MODEL_SPECS = [("Ours", "A"), ("Ours", "B"), ("Ours", "C")]

TYPE_ORDER = {"P": 0, "C": 1, "G": 2, "P+C": 3, "P+G": 4, "C+G": 5, "P+C+G": 6, "Ours": 7, "Other": 99}

GROUP_CONFIG = {
    "baselines": {
        "results_suffix": "full_model_val",
        "model_specs": BASELINE_MODEL_SPECS,
    },
    "ours": {
        "results_suffix": "poe_model_val",
        "model_specs": OURS_MODEL_SPECS,
    },
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_cindex_stats(csv_path: Path) -> tuple[float, float] | None:
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values: list[float] = []
            for row in reader:
                raw = row.get("test_cindex", "")
                if raw in ("", None):
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
    except Exception as exc:
        print(f"[WARN] Cannot read {csv_path}: {exc}")
        return None

    if not values:
        print(f"[WARN] Empty test_cindex in {csv_path}")
        return None

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return mean, std


def parse_mean_from_text(value: str) -> float | None:
    text = str(value).strip()
    if text in {"", "-", "nan"}:
        return None
    if "±" in text:
        text = text.split("±", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        return None


def format_mean(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def ensure_clean_csv_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for csv_path in path.glob("*.csv"):
        csv_path.unlink()


def resolve_group_study_dir(results_root: Path, group_dir: str, study_token: str, results_suffix: str) -> Path | None:
    folder_name = f"L0_{study_token}_{results_suffix}"
    candidates = [
        results_root / GROUP_DIR / folder_name,
        results_root / folder_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def extract_model_name(kind: str, csv_path: Path) -> str:
    model_dir = csv_path.parent.name
    if kind == "ours":
        return model_dir.rsplit("__", 1)[-1]
    return model_dir


def collect_group(results_root: Path, kind: str) -> dict[str, dict[str, str]]:
    cfg = GROUP_CONFIG[kind]
    study_to_model_text: dict[str, dict[str, str]] = {}

    for study_token, study in STUDY_SPECS:
        study_dir = resolve_group_study_dir(results_root, GROUP_DIR, study_token, cfg["results_suffix"])
        study_to_model_text.setdefault(study, {})
        if study_dir is None:
            print(f"[WARN] Missing study directory: L0_{study_token}_{cfg['results_suffix']}")
            continue

        print(f"[KIND] {kind} | {study_dir.name}")
        for csv_path in sorted(study_dir.rglob("test_result.csv")):
            model = extract_model_name(kind, csv_path)
            stats = load_cindex_stats(csv_path)
            if stats is None:
                continue
            mean, std = stats
            study_to_model_text[study][model] = f"{mean:.4f} ± {std:.4f}"

    return study_to_model_text


def write_group_outputs(
    kind: str,
    study_to_model_text: dict[str, dict[str, str]],
    output_root: Path,
) -> pd.DataFrame:
    cfg = GROUP_CONFIG[kind]
    out_dir = output_root / GROUP_DIR / LAYER_DIR / kind / "summary"
    ensure_clean_csv_dir(out_dir)

    studies = [study for _, study in STUDY_SPECS]
    models = [model for _, model in cfg["model_specs"]]
    model_to_study_text: dict[str, dict[str, str]] = {model: {} for model in models}

    for study in studies:
        model_map = study_to_model_text.get(study, {})
        study_rows: list[dict[str, object]] = []

        for model_type, model in cfg["model_specs"]:
            value = model_map.get(model, "-")
            study_rows.append(
                {
                    "study": study,
                    "model_type": model_type,
                    "model": model,
                    "test_cindex_mean_std": value,
                }
            )
            model_to_study_text.setdefault(model, {})[study] = value

        study_df = pd.DataFrame(study_rows)
        study_df["type_order"] = study_df["model_type"].map(lambda value: TYPE_ORDER.get(str(value), 99))
        study_df = study_df.sort_values(by=["type_order", "model"], kind="stable").drop(columns=["type_order"])
        study_df.to_csv(out_dir / f"{study}_model_summary.csv", index=False)

    matrix_rows: list[dict[str, object]] = []
    for model_type, model in cfg["model_specs"]:
        row: dict[str, object] = {"model_type": model_type, "model": model}
        values: list[float] = []
        for study in studies:
            text = model_to_study_text.get(model, {}).get(study, "-")
            row[study] = text
            parsed = parse_mean_from_text(text)
            if parsed is not None:
                values.append(parsed)
        row["mean"] = format_mean(sum(values) / len(values) if values else None)
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df["type_order"] = matrix_df["model_type"].map(lambda value: TYPE_ORDER.get(str(value), 99))
    matrix_df = (
        matrix_df.sort_values(by=["type_order", "model"], kind="stable")
        .drop(columns=["type_order"])
        .reset_index(drop=True)
    )
    matrix_df = matrix_df[["model_type", "model", *studies, "mean"]]
    matrix_path = out_dir / "table1_cindex_main_cindex_summary.csv"
    matrix_df.to_csv(matrix_path, index=False)

    print(f"[WRITE] {out_dir.relative_to(output_root)}")
    for study in studies:
        print(f"[WRITE] {out_dir.name}/{study}_model_summary.csv")
    print(f"[WRITE] {matrix_path.relative_to(output_root)}")
    return matrix_df


def write_total_summary(
    baseline_matrix_df: pd.DataFrame,
    ours_matrix_df: pd.DataFrame,
    output_root: Path,
) -> None:
    out_path = output_root / GROUP_DIR / LAYER_DIR / "summary.csv"
    total_df = pd.concat([baseline_matrix_df, ours_matrix_df], ignore_index=True)
    total_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(output_root)}")


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect Table 1 c-index summaries")
    parser.add_argument(
        "--results_root",
        type=Path,
        default=project_root / "results",
        help="Root results directory",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=project_root / "results_display",
        help="Root output directory",
    )
    args = parser.parse_args()

    baselines_text = collect_group(args.results_root, "baselines")
    ours_text = collect_group(args.results_root, "ours")

    baseline_matrix_df = write_group_outputs("baselines", baselines_text, args.output_root)
    ours_matrix_df = write_group_outputs("ours", ours_text, args.output_root)
    write_total_summary(baseline_matrix_df, ours_matrix_df, args.output_root)
    print("Done.")


if __name__ == "__main__":
    main()
