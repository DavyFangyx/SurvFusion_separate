import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
from models import Custom3DCNN, PatchEmbeddings
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations

from dataset_deployment.registry import get_dataset_config


def _resolve_split_csv_path(args):
    split_csv = getattr(args, "split_csv", None)
    if split_csv:
        return split_csv
    if args.data == "adni":
        return "./data/adni/PTID_splits.csv"
    if args.data == "mimic":
        return "./data/mimic/PTID_splits_mimic.csv"
    raise ValueError(f"Unsupported dataset for split csv resolution: {args.data}")


def _load_split_csv(split_csv_path):
    split_df = pd.read_csv(split_csv_path)
    first_col = str(split_df.columns[0]) if len(split_df.columns) else ""
    if first_col.startswith("Unnamed"):
        split_df = split_df.drop(columns=split_df.columns[0])
    missing = [col for col in ("train", "val", "test") if col not in split_df.columns]
    if missing:
        raise ValueError(f"{split_csv_path} missing columns: {missing}")
    train_ids = split_df["train"].dropna().astype(str).tolist()
    valid_ids = split_df["val"].dropna().astype(str).tolist()
    test_ids = split_df["test"].dropna().astype(str).tolist()
    return train_ids, valid_ids, test_ids


def _load_case_level_tensor(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.float()


def _resolve_slide_pt_path(root: Path, slide_id: str) -> Path:
    slide_stub = slide_id[:-4] if slide_id.endswith(".svs") else slide_id
    return root / f"{slide_stub}.pt"


def _pool_wsi_feature(case_slides: pd.DataFrame, wsi_root: Path) -> torch.Tensor:
    slide_tensors = []
    for slide_id in case_slides["slide_id"].dropna().astype(str).tolist():
        slide_path = _resolve_slide_pt_path(wsi_root, slide_id)
        if not slide_path.exists():
            raise FileNotFoundError(f"Missing WSI tensor: {slide_path}")
        slide_tensor = _load_case_level_tensor(slide_path)
        if slide_tensor.dim() == 1:
            slide_tensor = slide_tensor.unsqueeze(0)
        slide_tensors.append(slide_tensor)
    if not slide_tensors:
        raise ValueError("No WSI tensors available for the current case.")
    return torch.cat(slide_tensors, dim=0).mean(dim=0)


def load_and_preprocess_survpgc_data(args, modality_dict):
    config = get_dataset_config(args.study)
    workspace_root = Path(config.workspace_root)
    metadata_df = pd.read_csv(config.metadata_csv)
    metadata_df["case_id"] = metadata_df["case_id"].astype(str)
    case_df = metadata_df.copy()
    label_df = metadata_df.drop_duplicates("case_id").copy()
    label_df = label_df.set_index("case_id")
    censors = label_df["censorship"].fillna(0).astype(np.int64).values
    survival_times = label_df["survival_months"].fillna(0).astype(np.float32).values

    train_ids, valid_ids, test_ids = _load_split_csv(_resolve_split_csv_path(args))

    wsi_root = workspace_root / "P" / "uni_v1"
    gene_root = workspace_root / "G" / "scFoundation_embedding_gene_raw"
    clinic_root = workspace_root / "C" / "L4"

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {case_id: idx for idx, case_id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((censors.shape[0], 3), dtype=bool)
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    for case_id, row in label_df.iterrows():
        idx = id_to_idx[case_id]
        update_modality_combinations(idx, '')
        observed_idx_arr[idx, modality_dict['wsi']] = True
        observed_idx_arr[idx, modality_dict['genomic']] = True
        observed_idx_arr[idx, modality_dict['clinical']] = True

    wsi_rows = []
    gene_rows = []
    clinic_rows = []
    wsi_available = []
    gene_available = []
    clinic_available = []

    for case_id, group in case_df.groupby("case_id", sort=False):
        wsi_tensor = _pool_wsi_feature(group.reset_index(), wsi_root)
        gene_tensor = _load_case_level_tensor(gene_root / f"{case_id}.pt").reshape(-1)
        clinic_tensor = _load_case_level_tensor(clinic_root / f"{case_id}.pt").reshape(-1)
        wsi_rows.append(wsi_tensor.unsqueeze(0).numpy())
        gene_rows.append(gene_tensor.unsqueeze(0).numpy())
        clinic_rows.append(clinic_tensor.unsqueeze(0).numpy())
        wsi_available.append(True)
        gene_available.append(True)
        clinic_available.append(True)
        update_modality_combinations(id_to_idx[case_id], 'W')
        update_modality_combinations(id_to_idx[case_id], 'G')
        update_modality_combinations(id_to_idx[case_id], 'C')

    wsi_arr = np.vstack(wsi_rows).astype(np.float32)
    gene_arr = np.vstack(gene_rows).astype(np.float32)
    clinic_arr = np.vstack(clinic_rows).astype(np.float32)

    data_dict['wsi'] = wsi_arr
    data_dict['genomic'] = gene_arr
    data_dict['clinical'] = clinic_arr

    encoder_dict['wsi'] = PatchEmbeddings(wsi_arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
    encoder_dict['genomic'] = PatchEmbeddings(gene_arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
    encoder_dict['clinical'] = PatchEmbeddings(clinic_arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
    input_dims['wsi'] = wsi_arr.shape[1]
    input_dims['genomic'] = gene_arr.shape[1]
    input_dims['clinical'] = clinic_arr.shape[1]

    combination_to_index = get_modality_combinations(args.modality)
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0)
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*[set(train_idxs), set(valid_idxs), set(test_idxs)]) if len(train_idxs) > 0 else set()
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    case_ids = list(label_df.index)

    return (
        data_dict,
        encoder_dict,
        survival_times,
        censors,
        case_ids,
        train_idxs,
        valid_idxs,
        test_idxs,
        input_dims,
        transforms,
        masks,
        observed_idx_arr,
        full_modality_index,
    )

class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, preprocessed=False, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.preprocessed = preprocessed
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]

        # Sort ids by the number of available modalities
        self.sorted_ids = sorted(np.arange(len(ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
            if (modality == 'image') & (not self.preprocessed):
                subj1 = data[idx]
                subj_gm_3d = np.zeros(self.masks.shape, dtype=np.float32)
                subj_gm_3d.ravel()[self.masks] = subj1
                subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
                if self.transforms:
                    subj_gm_3d = self.transforms(subj_gm_3d)
                sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
                sample_data[modality] = np.array(sample)

        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        return sample_data, label, mc, observed


class SurvPGCMultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, survival_times, censors, input_dims, transforms, masks, case_ids, preprocessed=False, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.survival_times = survival_times
        self.censors = censors
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.case_ids = np.array(case_ids)
        self.preprocessed = preprocessed
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.survival_new = self.survival_times[ids]
        self.censor_new = self.censors[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]
        self.case_ids_new = self.case_ids[ids]

        self.sorted_ids = sorted(np.arange(len(ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.survival_new = self.survival_new[self.sorted_ids]
        self.censor_new = self.censor_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]
        self.case_ids_new = self.case_ids_new[self.sorted_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
        return sample_data, self.survival_new[idx], self.censor_new[idx], self.mc_new[idx], self.observed_new[idx], self.case_ids_new[idx]

def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_image_data(image_path, label_df, id_to_idx):
    # Load and preprocess image data
    image_data = np.load(os.path.join(image_path, 'ADNI_G.npy'), mmap_mode='r')
    mask_path = os.path.join(image_path, 'BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz')
    
    subject_ids = []
    dates = []
    with open('./data/adni/image/ADNI_subj.txt', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split('_')
            subject_id = '_'.join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)

    df = pd.DataFrame({
            'PTID': subject_ids,
            'date': dates
        })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=False)
    idx = df.groupby('PTID')['date'].idxmax()

    # Creating the subset DataFrame using the indexes
    subdf = df.loc[idx]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = pd.merge(subdf, label_df, on='PTID', how='left')

    image_data = image_data[merged_df['index']]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    data = nib.load(mask_path).get_fdata()
    mean = image_data.mean()
    std = image_data.std()     
    # mean = data.mean()
    # std = data.std()
    mask_gm = (data == 150).ravel()
    
    return tmp, filtered_idx, mean, std, mask_gm


def load_and_preprocess_data(args, modality_dict):
    # Paths
    image_path = './data/adni/image'
    preprocessed_image_path = './data/adni/image/UCSFFSX7_09Jun2025.csv'
    genomic_path = './data/adni/genomic/genomic_merged.h5ad'
    clinical_path = './data/adni/clinical/clinical_merged'
    biospecimen_path = './data/adni/biospecimen/biospecimen_merged'
    label_df = pd.read_csv('./data/adni/label.csv', index_col='PTID')
    label_df['DIAGNOSIS'] -= 1
    labels = label_df['DIAGNOSIS'].values.astype(np.int64)
    n_labels = len(set(labels))

    train_ids, valid_ids, test_ids = _load_split_csv(_resolve_split_csv_path(args))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],4), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'I' in args.modality or 'i' in args.modality:
        if args.preprocessed:
            df = pd.read_csv(preprocessed_image_path)
        
            # filter the latest record per subject using update_stamp
            df['update_stamp'] = pd.to_datetime(df['update_stamp'], errors='coerce')
            idx = df.groupby('PTID')['update_stamp'].idxmax()
            df = df.loc[idx].reset_index(drop=True)
            df.index = df['PTID']

            # select brain-related features ending with CV, TA, or SV.
            feature_cols = [col for col in df.columns if (
                col.endswith('CV') or col.endswith('TA') or col.endswith('SV')) and col.startswith('ST')
            ]
            df = df[feature_cols]

            if args.initial_filling == 'mean':
                df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)

            scaler = StandardScaler()
            brain_values = df.apply(pd.to_numeric, errors='coerce')  
            arr = scaler.fit_transform(brain_values.fillna(0)) 
            
            new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
            filtered_idx = new_idx[new_idx != -1]
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')
            tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
            tmp[filtered_idx] = arr[new_idx != -1]
            observed_idx_arr[filtered_idx, modality_dict['image']] = True
            data_dict['image'] = tmp.astype(np.float32)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
            input_dims['image'] = df.shape[1]

        else:
            arr, filtered_idx, mean, std, mask = load_and_preprocess_image_data(image_path, label_df, id_to_idx)
            observed_idx_arr[:, modality_dict['image']] = arr[:, 0] != -2
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')

            data_dict['image'] = np.array(arr)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = torch.nn.Sequential(
                Custom3DCNN(hidden_dim=args.hidden_dim).to(args.device),
                PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(args.device)
                )
            input_dims['image'] = arr.shape[1]
            transforms['image'] = Compose([
                                        ToTensor(),
                                        Normalize(mean=[mean], std=[std]),
                                    ])
            masks['image'] = mask

    if 'G' in args.modality or 'g' in args.modality:
        df = sc.read_h5ad(genomic_path).to_df()
        if args.initial_filling == 'mean':
            df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0) # use mode as genotype values are 0,1,2
        arr = df.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(arr)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['genomic']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'G')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['genomic'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['genomic'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['genomic'] = df.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        if args.initial_filling == 'mean':
            path = clinical_path + '_mean.csv'
        else:
            path = clinical_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        columns_to_exclude = [col for col in df.columns if col.startswith('PTCOGBEG') or col.startswith('PTADDX') or col.startswith('PTADBEG')]
        if len(columns_to_exclude) > 0:
            df = df.drop(columns_to_exclude, axis=1)
        arr = df.values.astype(np.float32)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['clinical']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['clinical'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['clinical'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['clinical'] = df.shape[1]

    if 'B' in args.modality or 'b' in args.modality:
        if args.initial_filling == 'mean':
            path = biospecimen_path + '_mean.csv'
        else:
            path = biospecimen_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        arr = df.values
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['biospecimen']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'B')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['biospecimen'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['biospecimen'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['biospecimen'] = df.shape[1]

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def load_and_preprocess_data_mimic(args, modality_dict):
    # Paths
    lab_path = './data/mimic/lab_x'
    note_path = './data/mimic/note_x'
    code_path = './data/mimic/code_x'
    label_df = pd.read_csv('./data/mimic/labels.csv', index_col='subject_id')
    labels = label_df['one_year_mortality'].values.astype(np.int64)
    n_labels = len(set(labels))

    train_ids, valid_ids, test_ids = _load_split_csv(_resolve_split_csv_path(args))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], args.n_full_modalities), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'L' in args.modality or 'l' in args.modality:
        path = lab_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['lab']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'L')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['lab'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['lab'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['lab'] = arr.shape[1]

    if 'N' in args.modality or 'n' in args.modality:
        path = note_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['note']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'N')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['note'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['note'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['note'] = arr.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        path = code_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['code']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['code'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['code'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['code'] = arr.shape[1]
    
    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def collate_fn(batch):
    data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, preprocessed, use_common_ids=True):
    if ('image' in list(data_dict.keys())) & (not preprocessed):
        train_transfrom = val_transform = test_transform = transforms['image']
        # val_transform = test_transform = False
        mask = masks['image']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, preprocessed, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, preprocessed, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, preprocessed, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader


def survpgc_collate_fn(batch):
    data, survival_times, censors, mcs, observeds, case_ids = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    survival_times = torch.tensor(survival_times, dtype=torch.float32)
    censors = torch.tensor(censors, dtype=torch.float32)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    case_ids = list(case_ids)
    return collated_data, survival_times, censors, mcs, observeds, case_ids


def create_survpgc_loaders(data_dict, observed_idx, survival_times, censors, case_ids, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, preprocessed, use_common_ids=True):
    train_dataset = SurvPGCMultiModalDataset(data_dict, observed_idx, train_ids, survival_times, censors, input_dims, transforms, masks, case_ids, preprocessed, use_common_ids)
    valid_dataset = SurvPGCMultiModalDataset(data_dict, observed_idx, valid_ids, survival_times, censors, input_dims, transforms, masks, case_ids, preprocessed, use_common_ids)
    test_dataset = SurvPGCMultiModalDataset(data_dict, observed_idx, test_ids, survival_times, censors, input_dims, transforms, masks, case_ids, preprocessed, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=survpgc_collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=survpgc_collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=survpgc_collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=survpgc_collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index
