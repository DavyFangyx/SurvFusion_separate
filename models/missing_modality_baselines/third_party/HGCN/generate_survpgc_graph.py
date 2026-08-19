from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import h5py
import joblib
import pandas as pd
import torch
from torch_geometric.data import Data

from dataset_deployment.registry import get_dataset_config
from utils.file_utils import _save_pkl


def _load_h5_slide(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(path, "r") as handle:
        features = torch.from_numpy(handle["features"][:]).float()
        coords = torch.from_numpy(handle["coords"][:]).long()
    if features.shape[0] != coords.shape[0]:
        raise ValueError(f"Patch count mismatch in {path}: {features.shape[0]} vs {coords.shape[0]}")
    return features, coords


def _build_8_neighborhood_edge_index(coords: torch.Tensor) -> torch.Tensor:
    coord_to_idx = {tuple(map(int, coord.tolist())): idx for idx, coord in enumerate(coords)}
    src = []
    dst = []
    for idx, (x, y) in enumerate(coords.tolist()):
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = coord_to_idx.get((int(x + dx), int(y + dy)))
                if neighbor is not None:
                    src.append(idx)
                    dst.append(neighbor)
    return torch.tensor([src, dst], dtype=torch.long)


def _build_full_edge_index(num_nodes: int) -> torch.Tensor:
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def _resolve_slide_h5(root: Path, slide_id: str) -> Path:
    candidate = root / f"{slide_id}.h5"
    if candidate.exists():
        return candidate
    if slide_id.endswith(".svs"):
        candidate = root / f"{slide_id[:-4]}.h5"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing WSI h5 for slide `{slide_id}` under {root}")


def _load_case_graph(case_id: str, case_df: pd.DataFrame, workspace_root: Path) -> Data:
    wsi_root = workspace_root / "P" / "uni_v1_h5"
    gene_root = workspace_root / "G" / "scFoundation_embedding_gene_raw"
    clinic_root = workspace_root / "C" / "L4"

    slide_ids = [str(value) for value in case_df["slide_id"].dropna().astype(str).tolist()]
    if not slide_ids:
        raise ValueError(f"Case `{case_id}` has no slide ids.")

    img_features = []
    img_edges = []
    node_offset = 0
    for slide_id in slide_ids:
        slide_path = _resolve_slide_h5(wsi_root, slide_id)
        features, coords = _load_h5_slide(slide_path)
        edge_index = _build_8_neighborhood_edge_index(coords) + node_offset
        img_features.append(features)
        img_edges.append(edge_index)
        node_offset += int(features.shape[0])

    x_img = torch.cat(img_features, dim=0)
    edge_index_image = torch.cat(img_edges, dim=1)

    gene_path = gene_root / f"{case_id}.pt"
    clinic_path = clinic_root / f"{case_id}.pt"
    x_rna = torch.load(gene_path, map_location="cpu").float()
    x_cli = torch.load(clinic_path, map_location="cpu").float()

    edge_index_rna = _build_full_edge_index(int(x_rna.shape[0]))
    edge_index_cli = _build_full_edge_index(int(x_cli.shape[0]))
    edge_index_model = _build_full_edge_index(3)

    censorship = int(case_df["censorship"].iloc[0])
    survival_months = float(case_df["survival_months"].iloc[0])

    return Data(
        x_img=x_img,
        x_rna=x_rna,
        x_cli=x_cli,
        sur_type=torch.tensor([censorship], dtype=torch.long),
        data_id=case_id,
        data_type=["img", "rna", "cli"],
        edge_index_model=edge_index_model,
        edge_index_image=edge_index_image,
        edge_index_rna=edge_index_rna,
        edge_index_cli=edge_index_cli,
        surv_time=torch.tensor([survival_months], dtype=torch.float32),
    )


def build_graph_pack(study: str) -> tuple[list[str], dict[str, list[float]], dict[str, Data]]:
    config = get_dataset_config(study)
    workspace_root = Path(config.workspace_root)
    metadata = pd.read_csv(config.metadata_csv)
    metadata["case_id"] = metadata["case_id"].astype(str)
    case_df = metadata.copy()
    label_df = metadata.drop_duplicates("case_id").copy().set_index("case_id")

    patients = []
    sur_and_time = {}
    all_data = {}
    for case_id, group in case_df.groupby("case_id", sort=False):
        case_id = str(case_id)
        data = _load_case_graph(case_id, group, workspace_root)
        all_data[case_id] = data
        patients.append(case_id)
        sur_and_time[case_id] = [int(label_df.loc[case_id, "censorship"]), float(label_df.loc[case_id, "survival_months"])]

    return patients, sur_and_time, all_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HGCN graph inputs from SurvPGC workspace features.")
    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patients, sur_and_time, all_data = build_graph_pack(args.study)
    output_dir = Path(args.output_dir or Path("models/missing_modality_baselines/third_party/HGCN/data_split") / args.study)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_pkl(str(output_dir / "patients.pkl"), patients)
    _save_pkl(str(output_dir / "sur_and_time.pkl"), sur_and_time)
    _save_pkl(str(output_dir / "all_data.pkl"), all_data)
    print(f"Saved HGCN graph pack to {output_dir}")


if __name__ == "__main__":
    main()
