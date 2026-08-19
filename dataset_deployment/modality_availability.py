from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import torch

from dataset_deployment.registry import get_dataset_config


MODALITY_KEYS = ("wsi", "gene", "clinic")
PATIENT_TABLE_COLUMNS = {"submitter_id", "has_P", "has_C", "has_G"}


def _normalize_case_id(case_id: str) -> str:
    return str(case_id).strip().upper()[:12]


@lru_cache(maxsize=None)
def load_patient_availability_lookup(study: str, repo_root: str = ".") -> dict[str, dict[str, bool]]:
    config = get_dataset_config(study)
    patient_table_path = Path(repo_root).resolve() / config.patient_table_csv
    if not patient_table_path.exists():
        raise FileNotFoundError(
            f"Patient availability table not found for study '{study}': {patient_table_path}"
        )

    patient_df = pd.read_csv(patient_table_path, encoding="utf-8-sig")
    missing_columns = PATIENT_TABLE_COLUMNS.difference(patient_df.columns)
    if missing_columns:
        raise ValueError(
            f"Patient availability table missing columns {sorted(missing_columns)}: {patient_table_path}"
        )

    lookup: dict[str, dict[str, bool]] = {}
    for row in patient_df.itertuples(index=False):
        case_id = _normalize_case_id(row.submitter_id)
        record = {
            "wsi": bool(int(row.has_P)),
            "gene": bool(int(row.has_G)),
            "clinic": bool(int(row.has_C)),
        }
        if not any(record.values()):
            raise ValueError(
                f"Invalid availability row with no active modality for case '{case_id}': {patient_table_path}"
            )
        lookup[case_id] = record

    return lookup


def get_case_availability(case_id: str, study: str, repo_root: str = ".") -> dict[str, bool]:
    normalized_case_id = _normalize_case_id(case_id)
    lookup = load_patient_availability_lookup(study, repo_root=repo_root)
    try:
        return dict(lookup[normalized_case_id])
    except KeyError as exc:
        raise KeyError(
            f"Case '{normalized_case_id}' is missing from the patient availability table for study '{study}'."
        ) from exc


def build_batch_availability(case_ids: list[str], study: str, repo_root: str = ".") -> dict[str, torch.BoolTensor]:
    batch = {name: [] for name in MODALITY_KEYS}
    for case_id in case_ids:
        case_avail = get_case_availability(case_id, study, repo_root=repo_root)
        for name in MODALITY_KEYS:
            batch[name].append(case_avail[name])

    return {
        name: torch.tensor(values, dtype=torch.bool)
        for name, values in batch.items()
    }
