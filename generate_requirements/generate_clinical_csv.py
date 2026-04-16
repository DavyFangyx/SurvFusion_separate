import argparse
import csv
import json
from pathlib import Path


DEFAULT_JSON_PATH = "/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json"
DEFAULT_INDEX_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/patients_index"
DEFAULT_OUTPUT_DIR = "/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/datasets_csv/clinical_data"

SUBTYPE_CONFIG = {
    "kich": {"index_csv": "A_both_KICH.csv", "output_csv": "tcga_kich_clinical.csv"},
    "kirc": {"index_csv": "A_both_KIRC.csv", "output_csv": "tcga_kirc_clinical.csv"},
    "kirp": {"index_csv": "A_both_KIRP.csv", "output_csv": "tcga_kirp_clinical.csv"},
}


def clean_row(row: dict[str, str]) -> dict[str, str]:
    cleaned = {}
    for key, value in row.items():
        normalized_key = str(key).replace("\ufeff", "").strip()
        normalized_value = "" if value is None else str(value).strip()
        cleaned[normalized_key] = normalized_value
    return cleaned


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clinical CSV for SurvPGC.")
    parser.add_argument("--json-path", default=DEFAULT_JSON_PATH)
    parser.add_argument("--index-csv", default=None)
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


def normalize_stage(raw_stage: str | None) -> str:
    if not raw_stage:
        return "N/A"
    cleaned = str(raw_stage).strip()
    if cleaned.lower().startswith("stage "):
        cleaned = cleaned[6:]
    return cleaned or "N/A"


def normalize_grade(raw_grade: str | None) -> str:
    if not raw_grade:
        return "N/A"
    cleaned = str(raw_grade).strip()
    return cleaned or "N/A"


def extract_subtype(case: dict) -> str:
    project_id = case.get("project", {}).get("project_id", "")
    if project_id.startswith("TCGA-"):
        return project_id.split("-", 1)[1]
    return project_id or "N/A"


def build_clinical_row(patient_id: str, case: dict) -> dict:
    diagnosis = pick_primary_diagnosis(case)
    return {
        "case_id": patient_id,
        "stage": normalize_stage(diagnosis.get("ajcc_pathologic_stage")),
        "subtype": extract_subtype(case),
        "grade": normalize_grade(diagnosis.get("tumor_grade")),
    }


def generate_one(index_csv: Path, output_csv: Path, case_lookup: dict[str, dict]) -> None:
    with index_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        patient_ids = []
        for row in reader:
            normalized_row = clean_row(row)
            patient_id = normalized_row.get("patient_id", "")
            if patient_id:
                patient_ids.append(patient_id)

    rows = []
    missing_patients = []
    for patient_id in sorted(set(patient_ids)):
        case = case_lookup.get(patient_id)
        if case is None:
            missing_patients.append(patient_id)
            continue
        rows.append(build_clinical_row(patient_id, case))

    rows.sort(key=lambda item: item["case_id"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "case_id", "stage", "subtype", "grade"])
        for index, row in enumerate(rows):
            writer.writerow([index, row["case_id"], row["stage"], row["subtype"], row["grade"]])

    print(f"Saved clinical CSV to {output_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Missing patients in JSON: {len(missing_patients)}")
    if missing_patients:
        print("First missing patients:", missing_patients[:10])


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    case_lookup = load_case_lookup(json_path)

    if args.index_csv and args.output_csv:
        generate_one(Path(args.index_csv), Path(args.output_csv), case_lookup)
        return

    if args.subtype:
        config = SUBTYPE_CONFIG[args.subtype]
        generate_one(Path(DEFAULT_INDEX_DIR) / config["index_csv"], Path(DEFAULT_OUTPUT_DIR) / config["output_csv"], case_lookup)
        return

    for subtype in ["kich", "kirc", "kirp"]:
        config = SUBTYPE_CONFIG[subtype]
        generate_one(Path(DEFAULT_INDEX_DIR) / config["index_csv"], Path(DEFAULT_OUTPUT_DIR) / config["output_csv"], case_lookup)


if __name__ == "__main__":
    main()