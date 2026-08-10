"""
Collect LIHC-family model-val summaries in the requested table format.

Outputs per kind:
    1. N study-level tables
       columns: study, model_type, model, test_cindex_mean_std
    2. 1 study-model matrix
       rows: model_type, model
       columns: study
       values: test_cindex_mean_std

Scans:
    results/*_full_model_val
    results/*_poe_model_val

Also writes:
    results_display/tables/cindex汇总表.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


FULL_MODEL_SPECS = [
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

POE_MODEL_SPECS = [
    ("Ours", "A"),
    ("Ours", "B"),
    ("Ours", "C"),
]

FULL_TYPE_ORDER = {"P": 0, "C": 1, "G": 2, "P+C": 3, "P+G": 4, "C+G": 5, "P+C+G": 6, "Ours": 7, "Other": 99}
POE_TYPE_ORDER = {"Ours": 7, "Other": 99}
STUDY_ORDER = [
    "tcga_brca",
    "tcga_coad",
    "tcga_kich",
    "tcga_kirc",
    "tcga_kirp",
    "tcga_lihc",
    "tcga_prad",
    "tcga_read",
]

KIND_CONFIG = {
    "full": {
        "results_glob": "*_full_model_val",
        "output_dir": "LIHC_full_model_val",
        "matrix_name": "lihc_full_model_val_cindex_summary.csv",
        "model_specs": FULL_MODEL_SPECS,
        "type_order": FULL_TYPE_ORDER,
    },
    "poe": {
        "results_glob": "*_poe_model_val",
        "output_dir": "LIHC_poe_model_val",
        "matrix_name": "lihc_poe_model_val_cindex_summary.csv",
        "model_specs": POE_MODEL_SPECS,
        "type_order": POE_TYPE_ORDER,
    },
}


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def study_from_dataset_dirname(name: str) -> str:
    token = name.split("_", 1)[0]
    return f"tcga_{token.lower()}"


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


def format_ranked_value(value: str, rank: float | None) -> str:
    text = str(value).strip()
    if text in {"", "-", "nan"}:
        return "-"
    if rank is None or pd.isna(rank):
        return text
    return f"{text} ({int(rank)})"


def append_top_marker(value: str, marker: str | None) -> str:
    text = str(value).strip()
    if text in {"", "-", "nan"} or marker is None:
        return text if text not in {"", "nan"} else "-"
    return f"{text} [{marker}]"


def extract_model_name(kind: str, csv_path: Path) -> str:
    model_dir = csv_path.parent.name
    if kind == "poe":
        return model_dir.rsplit("__", 1)[-1]
    return model_dir


def collect_kind(results_root: Path, kind: str) -> dict[str, dict[str, str]]:
    cfg = KIND_CONFIG[kind]
    study_to_model_text: dict[str, dict[str, str]] = {}

    for study_dir in sorted(results_root.glob(cfg["results_glob"])):
        if not study_dir.is_dir():
            continue

        study = study_from_dataset_dirname(study_dir.name)
        study_to_model_text.setdefault(study, {})
        print(f"[KIND] {kind} | {study_dir.name}")

        for csv_path in sorted(study_dir.rglob("test_result.csv")):
            model = extract_model_name(kind, csv_path)
            stats = load_cindex_stats(csv_path)
            if stats is None:
                continue

            mean, std = stats
            study_to_model_text[study][model] = f"{mean:.4f} ± {std:.4f}"

    return study_to_model_text


def ensure_clean_csv_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for csv_path in path.glob("*.csv"):
        csv_path.unlink()


def write_kind_outputs(
    kind: str,
    study_to_model_text: dict[str, dict[str, str]],
    output_root: Path,
) -> pd.DataFrame:
    cfg = KIND_CONFIG[kind]
    out_dir = output_root / cfg["output_dir"] / "summary"
    ensure_clean_csv_dir(out_dir)

    study_rows: list[dict[str, object]] = []
    studies = sorted(study_to_model_text.keys())
    models = [model for _, model in cfg["model_specs"]]
    model_to_study_text: dict[str, dict[str, str]] = {model: {} for model in models}

    for study in studies:
        model_map = study_to_model_text.get(study, {})
        study_table_rows: list[dict[str, object]] = []

        for model_type, model in cfg["model_specs"]:
            value = model_map.get(model, "-")
            study_table_rows.append(
                {
                    "study": study,
                    "model_type": model_type,
                    "model": model,
                    "test_cindex_mean_std": value,
                }
            )
            model_to_study_text.setdefault(model, {})[study] = value

        study_df = pd.DataFrame(study_table_rows)
        study_df["type_order"] = study_df["model_type"].map(lambda x: cfg["type_order"].get(x, 99))
        study_df = study_df.sort_values(
            by=["type_order", "model"],
            kind="stable",
        ).drop(columns=["type_order"])

        study_path = out_dir / f"{study}_model_summary.csv"
        study_df.to_csv(study_path, index=False)
        study_rows.append({"study": study, "path": study_path.name})

    matrix_rows: list[dict[str, object]] = []
    model_type_map = {model: model_type for model_type, model in cfg["model_specs"]}
    for model in models:
        row: dict[str, object] = {
            "model_type": model_type_map.get(model, "Other"),
            "model": model,
        }
        for study in studies:
            row[study] = model_to_study_text.get(model, {}).get(study, "-")
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df = matrix_df[["model_type", "model", *studies]]
    matrix_path = out_dir / cfg["matrix_name"]
    matrix_df.to_csv(matrix_path, index=False)

    print(f"[WRITE] {out_dir.relative_to(output_root)}")
    for row in study_rows:
        print(f"[WRITE] {out_dir.name}/{row['path']}")
    print(f"[WRITE] {matrix_path.relative_to(output_root)}")
    return matrix_df


def write_combined_outputs(
    matrices: dict[str, pd.DataFrame],
    output_root: Path,
) -> None:
    if not matrices:
        return

    tables_dir = output_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[pd.DataFrame] = []
    for kind in ("full", "poe"):
        matrix_df = matrices.get(kind)
        if matrix_df is None or matrix_df.empty:
            continue
        combined_rows.append(matrix_df.copy())

    if not combined_rows:
        return

    base_df = pd.concat(combined_rows, ignore_index=True)
    all_study_cols = [study for study in STUDY_ORDER if study in base_df.columns]

    def build_ranked_table(source_df: pd.DataFrame, selected_studies: list[str]) -> pd.DataFrame:
        table_df = source_df[["model_type", "model", *selected_studies]].copy()

        for study in selected_studies:
            table_df[f"{study}__mean_numeric"] = table_df[study].map(parse_mean_from_text)

        table_df["mean"] = table_df[selected_studies].apply(
            lambda row: format_mean(
                pd.Series([parse_mean_from_text(row[col]) for col in selected_studies]).mean(skipna=True)
            ),
            axis=1,
        )
        table_df["mean__mean_numeric"] = table_df["mean"].map(parse_mean_from_text)

        ours_mask = table_df["model"].isin(["A", "B", "C"])
        rank_target_cols = [*selected_studies, "mean"]
        for col in rank_target_cols:
            numeric_col = f"{col}__mean_numeric"
            table_df[f"{col}__rank"] = table_df[numeric_col].rank(method="min", ascending=False)
            table_df.loc[ours_mask, col] = table_df.loc[ours_mask].apply(
                lambda row: format_ranked_value(row[col], row[f"{col}__rank"]),
                axis=1,
            )

            distinct_values = sorted(
                {value for value in table_df[numeric_col].dropna().tolist()},
                reverse=True,
            )
            best_value = distinct_values[0] if len(distinct_values) >= 1 else None
            second_value = distinct_values[1] if len(distinct_values) >= 2 else None

            def marker_for_row(row: pd.Series) -> str | None:
                current = row[numeric_col]
                if pd.isna(current):
                    return None
                if best_value is not None and current == best_value:
                    return "best"
                if second_value is not None and current == second_value:
                    return "second"
                return None

            table_df[col] = table_df.apply(
                lambda row: append_top_marker(row[col], marker_for_row(row)),
                axis=1,
            )

        table_df = table_df[["model_type", "model", *selected_studies, "mean"]]
        table_df["type_order"] = table_df["model_type"].map(
            lambda value: FULL_TYPE_ORDER.get(str(value), POE_TYPE_ORDER.get(str(value), 99))
        )
        return (
            table_df.sort_values(by=["type_order", "model"], kind="stable")
            .drop(columns=["type_order"])
            .reset_index(drop=True)
        )

    combined_df = build_ranked_table(base_df, all_study_cols)
    combined_path = tables_dir / "cindex汇总表.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"[WRITE] {combined_path.relative_to(output_root)}")

    reduced_studies = [
        study for study in all_study_cols if study not in {"tcga_kich", "tcga_prad", "tcga_read"}
    ]
    reduced_df = build_ranked_table(base_df, reduced_studies)
    reduced_path = tables_dir / "cindex汇总表_去除kich_prad_read.csv"
    reduced_df.to_csv(reduced_path, index=False)
    print(f"[WRITE] {reduced_path.relative_to(output_root)}")

    poe_rank_path = tables_dir / "cindex汇总表_POE排名.csv"
    if poe_rank_path.exists():
        poe_rank_path.unlink()


def main() -> None:
    project_root = project_root_from_script()
    parser = argparse.ArgumentParser(description="Collect LIHC full/poe model-val summaries")
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
        "--kinds",
        nargs="+",
        choices=["full", "poe"],
        default=["full", "poe"],
        help="Which result families to collect",
    )
    args = parser.parse_args()

    matrices: dict[str, pd.DataFrame] = {}
    for kind in args.kinds:
        study_to_model_text = collect_kind(args.results_root, kind)
        if not study_to_model_text:
            print(f"[SKIP] No usable rows for kind={kind}")
            continue
        matrices[kind] = write_kind_outputs(kind, study_to_model_text, args.output_root)

    write_combined_outputs(matrices, args.output_root)

    print("Done.")


if __name__ == "__main__":
    main()
