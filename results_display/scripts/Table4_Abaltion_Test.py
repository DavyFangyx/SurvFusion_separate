"""
Collect Table 4 c-index summaries.

Default input:
    results/Table4_Abaltion_Test/{study}__{input}/{ablation_model}/test_result.csv

Default outputs:
    results_display/Table4_Abaltion_Test/summary.csv
    results_display/Table4_Abaltion_Test/summary_5datasets.csv
    results_display/Table4_Abaltion_Test/summary/{study}_model_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


GROUP_DIR = "Table4_Abaltion_Test"
DEFAULT_MODEL = "survtri_poe_vae_b_nopretrain"

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
FIVE_DATASET_STUDIES = ["tcga_brca", "tcga_coad", "tcga_kirc", "tcga_kirp", "tcga_lihc"]

MODEL_SPECS = [("Ablation", DEFAULT_MODEL)]
TYPE_ORDER = {"Ablation": 0, "Other": 99}


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


def collect_group(
    results_root: Path,
    *,
    group_dir: str,
    model_name: str,
) -> dict[str, dict[str, str]]:
    results_dir = results_root / group_dir
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results directory: {results_dir}")

    study_to_model_text: dict[str, dict[str, str]] = {study: {} for _, study in STUDY_SPECS}
    valid_studies = {study for _, study in STUDY_SPECS}

    for csv_path in sorted(results_dir.rglob("test_result.csv")):
        model_dir = csv_path.parent
        run_dir = model_dir.parent
        if model_dir.name != model_name or "__" not in run_dir.name:
            continue

        study_token = run_dir.name.split("__", 1)[0]
        if study_token not in valid_studies:
            continue

        stats = load_cindex_stats(csv_path)
        if stats is None:
            continue

        mean, std = stats
        study_to_model_text[study_token][model_dir.name] = f"{mean:.4f} ± {std:.4f}"

    return study_to_model_text


def build_matrix_frame(
    study_to_model_text: dict[str, dict[str, str]],
    studies: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_type, model in MODEL_SPECS:
        row: dict[str, object] = {"model_type": model_type, "model": model}
        values: list[float] = []
        for study in studies:
            text = study_to_model_text.get(study, {}).get(model, "-")
            row[study] = text
            parsed = parse_mean_from_text(text)
            if parsed is not None:
                values.append(parsed)
        row["mean"] = format_mean(sum(values) / len(values) if values else None)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["type_order"] = df["model_type"].map(lambda value: TYPE_ORDER.get(str(value), 99))
    df = (
        df.sort_values(by=["type_order", "model"], kind="stable")
        .drop(columns=["type_order"])
        .reset_index(drop=True)
    )
    return df[["model_type", "model", *studies, "mean"]]


def write_per_study_outputs(
    matrix_df: pd.DataFrame,
    output_root: Path,
    *,
    group_dir: str,
) -> None:
    out_dir = output_root / group_dir / "summary"
    ensure_clean_csv_dir(out_dir)

    studies = [column for column in matrix_df.columns if column not in {"model_type", "model", "mean"}]
    for study in studies:
        study_rows = [
            {
                "study": study,
                "model_type": row["model_type"],
                "model": row["model"],
                "test_cindex_mean_std": row[study],
            }
            for _, row in matrix_df.iterrows()
        ]
        study_df = pd.DataFrame(study_rows)
        study_df.to_csv(out_dir / f"{study}_model_summary.csv", index=False)

    print(f"[WRITE] {out_dir.relative_to(output_root)}")
    for study in studies:
        print(f"[WRITE] {out_dir.name}/{study}_model_summary.csv")


def write_summary(matrix_df: pd.DataFrame, output_root: Path, *, group_dir: str, file_name: str) -> None:
    out_path = output_root / group_dir / file_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path.relative_to(output_root)}")


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect Table 4 c-index summaries")
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
    parser.add_argument(
        "--group-dir",
        default=GROUP_DIR,
        help="Grouped results/output directory name",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help="Ablation model directory name",
    )
    args = parser.parse_args()

    study_to_model_text = collect_group(
        args.results_root,
        group_dir=args.group_dir,
        model_name=args.model_name,
    )

    matrix_df = build_matrix_frame(
        study_to_model_text,
        [study for _, study in STUDY_SPECS],
    )
    subset_df = build_matrix_frame(
        study_to_model_text,
        FIVE_DATASET_STUDIES,
    )

    write_per_study_outputs(
        matrix_df,
        args.output_root,
        group_dir=args.group_dir,
    )
    write_summary(
        matrix_df,
        args.output_root,
        group_dir=args.group_dir,
        file_name="summary.csv",
    )
    write_summary(
        subset_df,
        args.output_root,
        group_dir=args.group_dir,
        file_name="summary_5datasets.csv",
    )
    print("Done.")


if __name__ == "__main__":
    main()
