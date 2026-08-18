"""
Collect Table 1 c-index summaries for a given layer prefix.

Default inputs:
    results/Table1_Cindex_Main/L0Test/L0_*_full_model_val
    results/Table1_Cindex_Main/L0Test/L0_*_poe_model_val

Default outputs:
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
DEFAULT_LAYER_DIR = "L0"

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


def append_rank_suffix(value: str, rank: int) -> str:
    text = str(value).strip()
    if text in {"", "-"}:
        return text
    if text.endswith(")") and "(" in text:
        return text
    return f"{text}({rank})"


def ensure_clean_csv_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for csv_path in path.glob("*.csv"):
        csv_path.unlink()


def resolve_group_study_dir(
    results_root: Path,
    group_dir: str,
    layer_dir: str,
    test_dir: str | None,
    study_token: str,
    results_suffix: str,
) -> Path | None:
    folder_name = f"{layer_dir}_{study_token}_{results_suffix}"
    candidates = [
        results_root / group_dir / test_dir / folder_name if test_dir else None,
        results_root / group_dir / folder_name,
        results_root / folder_name,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def extract_model_name(kind: str, csv_path: Path) -> str:
    model_dir = csv_path.parent.name
    if kind == "ours":
        return model_dir.rsplit("__", 1)[-1]
    return model_dir


def collect_group(
    results_root: Path,
    kind: str,
    *,
    group_dir: str,
    layer_dir: str,
    test_dir: str | None,
) -> dict[str, dict[str, str]]:
    cfg = GROUP_CONFIG[kind]
    study_to_model_text: dict[str, dict[str, str]] = {}

    for study_token, study in STUDY_SPECS:
        study_dir = resolve_group_study_dir(
            results_root,
            group_dir,
            layer_dir,
            test_dir,
            study_token,
            cfg["results_suffix"],
        )
        study_to_model_text.setdefault(study, {})
        if study_dir is None:
            print(f"[WARN] Missing study directory: {layer_dir}_{study_token}_{cfg['results_suffix']}")
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


def build_global_rank_map(
    study_to_model_text: dict[str, dict[str, str]],
    model_specs: list[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    model_names = [model for _, model in model_specs]
    study_to_model_rank: dict[str, dict[str, int]] = {}

    for study, model_map in study_to_model_text.items():
        ranked_values: list[tuple[float, str]] = []
        for model in model_names:
            mean = parse_mean_from_text(model_map.get(model, "-"))
            if mean is None:
                continue
            ranked_values.append((mean, model))

        ranked_values.sort(key=lambda item: (-item[0], item[1]))
        study_to_model_rank[study] = {model: rank for rank, (_, model) in enumerate(ranked_values, start=1)}

    return study_to_model_rank


def build_mean_rank_map(
    study_to_model_text: dict[str, dict[str, str]],
    model_specs: list[tuple[str, str]],
    studies: list[str],
) -> dict[str, int]:
    model_names = [model for _, model in model_specs]
    ranked_values: list[tuple[float, str]] = []

    for model in model_names:
        values: list[float] = []
        for study in studies:
            mean = parse_mean_from_text(study_to_model_text.get(study, {}).get(model, "-"))
            if mean is not None:
                values.append(mean)
        if not values:
            continue
        ranked_values.append((sum(values) / len(values), model))

    ranked_values.sort(key=lambda item: (-item[0], item[1]))
    return {model: rank for rank, (_, model) in enumerate(ranked_values, start=1)}


def annotate_models_with_ranks(
    study_to_model_text: dict[str, dict[str, str]],
    study_to_model_rank: dict[str, dict[str, int]],
    target_models: set[str],
) -> dict[str, dict[str, str]]:
    ranked_study_to_model_text: dict[str, dict[str, str]] = {}

    for study, model_map in study_to_model_text.items():
        ranked_model_map = dict(model_map)
        rank_map = study_to_model_rank.get(study, {})
        for model in target_models:
            if model not in ranked_model_map:
                continue
            rank = rank_map.get(model)
            if rank is None:
                continue
            ranked_model_map[model] = append_rank_suffix(ranked_model_map[model], rank)
        ranked_study_to_model_text[study] = ranked_model_map

    return ranked_study_to_model_text


def write_group_outputs(
    kind: str,
    study_to_model_text: dict[str, dict[str, str]],
    output_root: Path,
    *,
    group_dir: str,
    layer_dir: str,
    study_to_model_rank: dict[str, dict[str, int]] | None = None,
    mean_rank_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    cfg = GROUP_CONFIG[kind]
    out_dir = output_root / group_dir / layer_dir / kind / "summary"
    ensure_clean_csv_dir(out_dir)
    if kind == "ours" and study_to_model_rank is not None:
        study_to_model_text = annotate_models_with_ranks(
            study_to_model_text,
            study_to_model_rank,
            {model for _, model in cfg["model_specs"]},
        )

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
    if kind == "ours" and mean_rank_map is not None:
        for index, row in matrix_df.iterrows():
            model = str(row["model"])
            rank = mean_rank_map.get(model)
            if rank is None:
                continue
            matrix_df.at[index, "mean"] = append_rank_suffix(str(row["mean"]), rank)
    matrix_path = out_dir / "table1_cindex_main_cindex_summary.csv"
    matrix_df.to_csv(matrix_path, index=False)

    print(f"[WRITE] {out_dir.relative_to(output_root)}")
    for study in studies:
        print(f"[WRITE] {out_dir.name}/{study}_model_summary.csv")
    print(f"[WRITE] {matrix_path.relative_to(output_root)}")
    return matrix_df


def build_summary_frame(
    baseline_matrix_df: pd.DataFrame,
    ours_matrix_df: pd.DataFrame,
    studies: list[str],
    ours_mean_rank_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    base_cols = ["model_type", "model", *studies]
    total_df = pd.concat(
        [baseline_matrix_df[base_cols], ours_matrix_df[base_cols]],
        ignore_index=True,
    )

    mean_values: list[str] = []
    for _, row in total_df.iterrows():
        values: list[float] = []
        for study in studies:
            parsed = parse_mean_from_text(str(row[study]))
            if parsed is not None:
                values.append(parsed)
        mean_text = format_mean(sum(values) / len(values) if values else None)
        if str(row["model_type"]) == "Ours" and ours_mean_rank_map is not None:
            rank = ours_mean_rank_map.get(str(row["model"]))
            if rank is not None:
                mean_text = append_rank_suffix(mean_text, rank)
        mean_values.append(mean_text)

    total_df = total_df.copy()
    total_df["mean"] = mean_values
    return total_df[["model_type", "model", *studies, "mean"]]


def write_total_summary(
    summary_df: pd.DataFrame,
    output_root: Path,
    *,
    group_dir: str,
    layer_dir: str,
    file_name: str,
) -> None:
    out_path = output_root / group_dir / layer_dir / file_name
    summary_df.to_csv(out_path, index=False)
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
    parser.add_argument(
        "--group-dir",
        default=GROUP_DIR,
        help="Grouped results/output directory name",
    )
    parser.add_argument(
        "--layer-dir",
        default=DEFAULT_LAYER_DIR,
        help="Layer prefix to collect, such as L0 or L4",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Optional extra subdirectory under the grouped results directory; defaults to {layer-dir}Test",
    )
    args = parser.parse_args()

    test_dir = args.test_dir if args.test_dir is not None else f"{args.layer_dir}Test"

    baselines_text = collect_group(
        args.results_root,
        "baselines",
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
        test_dir=test_dir,
    )
    ours_text = collect_group(
        args.results_root,
        "ours",
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
        test_dir=test_dir,
    )
    combined_text: dict[str, dict[str, str]] = {}
    for study, model_map in baselines_text.items():
        combined_text.setdefault(study, {}).update(model_map)
    for study, model_map in ours_text.items():
        combined_text.setdefault(study, {}).update(model_map)
    global_rank_map = build_global_rank_map(
        combined_text,
        BASELINE_MODEL_SPECS + OURS_MODEL_SPECS,
    )
    full_mean_rank_map = build_mean_rank_map(
        combined_text,
        BASELINE_MODEL_SPECS + OURS_MODEL_SPECS,
        [study for _, study in STUDY_SPECS],
    )
    five_dataset_mean_rank_map = build_mean_rank_map(
        combined_text,
        BASELINE_MODEL_SPECS + OURS_MODEL_SPECS,
        FIVE_DATASET_STUDIES,
    )

    baseline_matrix_df = write_group_outputs(
        "baselines",
        baselines_text,
        args.output_root,
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
    )
    ours_matrix_df = write_group_outputs(
        "ours",
        ours_text,
        args.output_root,
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
        study_to_model_rank=global_rank_map,
        mean_rank_map=full_mean_rank_map,
    )
    full_summary_df = build_summary_frame(
        baseline_matrix_df,
        ours_matrix_df,
        [study for _, study in STUDY_SPECS],
        ours_mean_rank_map=full_mean_rank_map,
    )
    five_dataset_summary_df = build_summary_frame(
        baseline_matrix_df,
        ours_matrix_df,
        FIVE_DATASET_STUDIES,
        ours_mean_rank_map=five_dataset_mean_rank_map,
    )
    write_total_summary(
        full_summary_df,
        args.output_root,
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
        file_name="summary.csv",
    )
    write_total_summary(
        five_dataset_summary_df,
        args.output_root,
        group_dir=args.group_dir,
        layer_dir=args.layer_dir,
        file_name="summary_5datasets.csv",
    )
    print("Done.")


if __name__ == "__main__":
    main()
