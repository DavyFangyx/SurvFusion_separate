import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


DEFAULT_JSON_PATH = "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json"
DEFAULT_INDEX_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/patients_index"
DEFAULT_PATCH_DIR = "/data/fangyuxuan/projects/medical_dl/trident_project/TRIDENT_workspace/20.0x_256px_0px_overlap/patches"
DEFAULT_OUTPUT_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/datasets_csv/metadata"

SUBTYPE_CONFIG = {
    "kich": {"index_csv": "A_both_KICH.csv", "output_csv": "tcga_kich.csv"},
    "kirc": {"index_csv": "A_both_KIRC.csv", "output_csv": "tcga_kirc.csv"},
    "kirp": {"index_csv": "A_both_KIRP.csv", "output_csv": "tcga_kirp.csv"},
}


def clean_row(row: dict[str, str]) -> dict[str, str]:
    cleaned = {}
    for key, value in row.items():
        normalized_key = str(key).replace("\ufeff", "").strip()
        normalized_value = "" if value is None else str(value).strip()
        cleaned[normalized_key] = normalized_value
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metadata CSV for SurvPGC.")
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--index-csv", default=None)
    parser.add_argument("--patch-dir", default=DEFAULT_PATCH_DIR)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--subtype", choices=sorted(SUBTYPE_CONFIG.keys()), default=None)
    return parser.parse_args()


def load_case_lookup(json_path: Path) -> dict[str, dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        cases = json.load(handle)
    return {case["submitter_id"]: case for case in cases}


def pick_primary_diagnosis(case: dict) -> dict:
    diagnoses = case.get("diagnoses", [])
    for diagnosis in diagnoses:
        if str(diagnosis.get("diagnosis_is_primary_disease", "")).lower() == "true":
            return diagnosis
    if diagnoses:
        return diagnoses[0]
    return {}


def build_slide_lookup(patch_dir: Path) -> dict[str, list[str]]:
    slide_lookup: dict[str, list[str]] = defaultdict(list)
    for patch_file in sorted(patch_dir.glob("*_patches.h5")):
        slide_name = patch_file.name.replace("_patches.h5", ".svs")
        patient_id = slide_name[:12]
        slide_lookup[patient_id].append(slide_name)
    return slide_lookup


def extract_survival(case: dict, diagnosis: dict) -> tuple[float | None, int | None]:
    demographic = case.get("demographic", {})

    days_to_death = demographic.get("days_to_death")
    if days_to_death not in (None, "", "'--"):
        return float(days_to_death) / 30.44, 0

    last_follow_up = diagnosis.get("days_to_last_follow_up")
    if last_follow_up not in (None, "", "'--"):
        return float(last_follow_up) / 30.44, 1

    for follow_up in reversed(case.get("follow_ups", [])):
        candidate = follow_up.get("days_to_follow_up")
        if candidate not in (None, "", "'--"):
            return float(candidate) / 30.44, 1

    return None, None


def extract_age(case: dict) -> int | None:
    demographic = case.get("demographic", {})
    age = demographic.get("age_at_index")
    if age not in (None, "", "'--"):
        return int(age)

    days_to_birth = demographic.get("days_to_birth")
    if days_to_birth not in (None, "", "'--"):
        return int(abs(float(days_to_birth)) / 365.25)
    return None


def extract_is_female(case: dict) -> int:
    gender = str(case.get("demographic", {}).get("gender", "")).lower()
    return 1 if gender == "female" else 0


def extract_oncotree_code(case: dict) -> str:
    project_id = case.get("project", {}).get("project_id", "")
    if project_id.startswith("TCGA-"):
        return project_id.split("-", 1)[1]
    return project_id or "N/A"


def load_index_rows(index_csv: Path) -> list[dict[str, str]]:
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return [clean_row(row) for row in csv.DictReader(handle)]


def build_metadata_rows(index_rows: list[dict[str, str]], case_lookup: dict[str, dict], slide_lookup: dict[str, list[str]]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing_patients: list[str] = []

    for record in index_rows:
        patient_id = str(record.get("patient_id", "")).strip()
        if not patient_id:
            continue
        case = case_lookup.get(patient_id)
        slides = slide_lookup.get(patient_id, [])

        if case is None or not slides:
            missing_patients.append(patient_id)
            continue

        diagnosis = pick_primary_diagnosis(case)
        survival_months, censorship = extract_survival(case, diagnosis)
        if survival_months is None or censorship is None:
            missing_patients.append(patient_id)
            continue

        for slide_id in slides:
            rows.append(
                {
                    "case_id": patient_id,
                    "slide_id": slide_id,
                    "age": extract_age(case),
                    "site": patient_id.split("-")[1],
                    "survival_months": survival_months,
                    "censorship": censorship,
                    "is_female": extract_is_female(case),
                    "oncotree_code": extract_oncotree_code(case),
                    "rna_file_name": record.get("rna_file_name", ""),
                }
            )

    return rows, missing_patients


def generate_one(index_csv: Path, output_csv: Path, case_lookup: dict[str, dict], slide_lookup: dict[str, list[str]]) -> None:
    index_rows = load_index_rows(index_csv)
    rows, missing_patients = build_metadata_rows(index_rows, case_lookup, slide_lookup)
    rows.sort(key=lambda item: (item["case_id"], item["slide_id"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "slide_id",
        "age",
        "site",
        "survival_months",
        "censorship",
        "is_female",
        "oncotree_code",
        "rna_file_name",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + fieldnames)
        for index, row in enumerate(rows):
            writer.writerow([index] + [row.get(field, "") for field in fieldnames])

    print(f"Saved metadata CSV to {output_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Patients with missing JSON/WSI/survival data: {len(set(missing_patients))}")
    if missing_patients:
        print("First missing patients:", sorted(set(missing_patients))[:10])


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    patch_dir = Path(args.patch_dir)

    case_lookup = load_case_lookup(json_path)
    slide_lookup = build_slide_lookup(patch_dir)

    if args.index_csv and args.output_csv:
        generate_one(Path(args.index_csv), Path(args.output_csv), case_lookup, slide_lookup)
        return

    if args.subtype:
        config = SUBTYPE_CONFIG[args.subtype]
        generate_one(Path(DEFAULT_INDEX_DIR) / config["index_csv"], Path(DEFAULT_OUTPUT_DIR) / config["output_csv"], case_lookup, slide_lookup)
        return

    for subtype in ["kich", "kirc", "kirp"]:
        config = SUBTYPE_CONFIG[subtype]
        generate_one(Path(DEFAULT_INDEX_DIR) / config["index_csv"], Path(DEFAULT_OUTPUT_DIR) / config["output_csv"], case_lookup, slide_lookup)


if __name__ == "__main__":
    main()