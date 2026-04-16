import argparse
import csv
import random
from pathlib import Path


DEFAULT_METADATA_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/datasets_csv/metadata"
DEFAULT_OUTPUT_ROOT = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/splits/5foldcv"

SUBTYPE_CONFIG = {
    "kich": {"metadata_csv": "tcga_kich.csv", "output_dir": "tcga_kich"},
    "kirc": {"metadata_csv": "tcga_kirc.csv", "output_dir": "tcga_kirc"},
    "kirp": {"metadata_csv": "tcga_kirp.csv", "output_dir": "tcga_kirp"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 5-fold split CSVs for SurvPGC.")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--subtype", choices=sorted(SUBTYPE_CONFIG.keys()), default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def save_split_csv(train_ids: list[str], val_ids: list[str], output_path: Path) -> None:
    max_len = max(len(train_ids), len(val_ids))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "train", "val", "test"])
        for index in range(max_len):
            train_value = train_ids[index] if index < len(train_ids) else ""
            val_value = val_ids[index] if index < len(val_ids) else ""
            writer.writerow([index, train_value, val_value, val_value])


def load_case_ids(metadata_csv: Path) -> list[str]:
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        case_ids = set()
        for row in reader:
            normalized_row = {
                str(key).replace("\ufeff", "").strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            case_id = normalized_row.get("case_id", "")
            if case_id:
                case_ids.add(case_id)
        return sorted(case_ids)


def build_folds(case_ids: list[str], n_splits: int, seed: int) -> list[list[str]]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(n_splits)]
    for index, case_id in enumerate(shuffled):
        folds[index % n_splits].append(case_id)
    return folds


def generate_one(metadata_csv: Path, output_dir: Path, n_splits: int, seed: int) -> None:
    case_ids = load_case_ids(metadata_csv)
    if len(case_ids) < n_splits:
        raise ValueError(f"Not enough cases ({len(case_ids)}) for {n_splits} folds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    folds = build_folds(case_ids, n_splits, seed)

    for fold, val_ids in enumerate(folds):
        train_ids = [case_id for fold_index, fold_cases in enumerate(folds) if fold_index != fold for case_id in fold_cases]
        save_split_csv(train_ids, val_ids, output_dir / f"splits_{fold}.csv")

    print(f"Saved {n_splits} split files to {output_dir}")
    print(f"Unique cases: {len(case_ids)}")


def main() -> None:
    args = parse_args()

    if args.metadata_csv and args.output_dir:
        generate_one(Path(args.metadata_csv), Path(args.output_dir), args.n_splits, args.seed)
        return

    if args.subtype:
        config = SUBTYPE_CONFIG[args.subtype]
        generate_one(Path(DEFAULT_METADATA_DIR) / config["metadata_csv"], Path(DEFAULT_OUTPUT_ROOT) / config["output_dir"], args.n_splits, args.seed)
        return

    for subtype in ["kich", "kirc", "kirp"]:
        config = SUBTYPE_CONFIG[subtype]
        generate_one(Path(DEFAULT_METADATA_DIR) / config["metadata_csv"], Path(DEFAULT_OUTPUT_ROOT) / config["output_dir"], args.n_splits, args.seed)


if __name__ == "__main__":
    main()