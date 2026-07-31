# TCGA Dataset File Trees For AI Parsing

Base root:

```text
/data/lizhe/Medteam_projects/
```

Important modalities:

```text
WSI     = whole-slide image data, usually *.svs
Gene    = bulk RNA-seq gene expression data, usually *.rna_seq.augmented_star_gene_counts.tsv
Clinic  = clinical.cart.*.json
```

Common normal patterns:

```text
<dataset>/WSI/<file_uuid>/
├── *.svs
└── logs/
    └── *.svs.parcel

<dataset>/Bulk_RNA/<file_uuid>/
├── *.rna_seq.augmented_star_gene_counts.tsv
├── annotations.txt                  # optional, not always present
└── logs/
    └── *.rna_seq.augmented_star_gene_counts.tsv.parcel
```

---

## TCGA-BRCA

Root:

```text
/data/lizhe/Medteam_projects/TCGA-BRCA/
```

Top-level tree:

```text
TCGA-BRCA/
├── Bulk_RNA/             # Gene data
├── Bulk_RNA_case/        # Gene case metadata + clinic
├── Mutation/
├── Mutation_case/
├── Mythylation/
├── Mythylation_case/
├── RPPA/
├── RPPA_case/
├── WSI/                  # WSI data
├── WSI_case/             # WSI case metadata + clinic
├── TCGA_info
└── gdc-client
```

Path map:

```yaml
dataset: TCGA-BRCA
root: /data/lizhe/Medteam_projects/TCGA-BRCA
gene_root: /data/lizhe/Medteam_projects/TCGA-BRCA/Bulk_RNA
wsi_root: /data/lizhe/Medteam_projects/TCGA-BRCA/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA-BRCA/Bulk_RNA_case/clinical.cart.2026-06-29.json
  - /data/lizhe/Medteam_projects/TCGA-BRCA/WSI_case/clinical.cart.2026-06-29.json
  - /data/lizhe/Medteam_projects/TCGA-BRCA/Mutation_case/clinical.cart.2026-06-29.json
  - /data/lizhe/Medteam_projects/TCGA-BRCA/Mythylation_case/clinical.cart.2026-06-29.json
  - /data/lizhe/Medteam_projects/TCGA-BRCA/RPPA_case/clinical.cart.2026-06-29.json
case_dirs:
  - Bulk_RNA_case
  - WSI_case
  - Mutation_case
  - Mythylation_case
  - RPPA_case
counts:
  gene_uuid_dirs: 1118
  wsi_uuid_dirs: 1133
notes:
  - case directory suffix is singular: *_case
  - has additional modalities: Mutation, Mythylation, RPPA
```

Representative Gene file tree:

```text
TCGA-BRCA/Bulk_RNA/0019c951-16c5-48d0-85c8-58d96b12d330/
├── ba295155-272e-43eb-9d6a-e4c9c392e68b.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── ba295155-272e-43eb-9d6a-e4c9c392e68b.rna_seq.augmented_star_gene_counts.tsv.parcel
```

Representative WSI file tree:

```text
TCGA-BRCA/WSI/0018cc22-498a-45b8-bfd4-b0fe0c3a2f0a/
├── TCGA-S3-AA15-01Z-00-DX1.A2456A4A-E6E8-4429-8F09-B997AA497BB0.svs
└── logs/
    └── TCGA-S3-AA15-01Z-00-DX1.A2456A4A-E6E8-4429-8F09-B997AA497BB0.svs.parcel
```

---

## TCGA_LIHC

Root:

```text
/data/lizhe/Medteam_projects/TCGA_LIHC/
```

Top-level tree:

```text
TCGA_LIHC/
├── Bulk_RNA/                                  # Gene data
├── WSI/                                       # WSI data
├── clinical/                                  # Clinic metadata
├── clinical.cart.2026-03-26.json              # older clinic file at root
├── clinical.cart.2026-03-26.tar.gz
├── gdc_manifest.2026-03-26.152118_wsi.txt
├── gdc_manifest.2026-06-10.160348_rna.txt
├── metadata.cart.2026-03-26.json
├── TCGA_info
├── TCGA-RNA-seq数据下载.png
├── 肝癌数据.png
└── gdc-client
```

Path map:

```yaml
dataset: TCGA_LIHC
root: /data/lizhe/Medteam_projects/TCGA_LIHC
gene_root: /data/lizhe/Medteam_projects/TCGA_LIHC/Bulk_RNA
wsi_root: /data/lizhe/Medteam_projects/TCGA_LIHC/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA_LIHC/clinical/clinical.cart.2026-06-01.json
  - /data/lizhe/Medteam_projects/TCGA_LIHC/clinical.cart.2026-03-26.json
case_dirs: []
counts:
  gene_uuid_dirs: 374
  wsi_uuid_dirs: 379
notes:
  - no Bulk_RNA_cases directory
  - no WSI_cases directory
  - clinic is stored in clinical/
  - root also has older clinical.cart.2026-03-26.json
  - RNA 一级子目录名对应 GDC manifest 里的 `file_id`
  - RNA `.tsv` 文件名对应 GDC manifest 里的 `filename`
  - 本地没有现成的 RNA file-to-case 表，因此生成 `rna_clean.csv` / gene manifest 时需要额外补 `file_id/filename -> case_id` 映射
```

Representative Gene file tree:

```text
TCGA_LIHC/Bulk_RNA/0036fcec-eaed-430b-9a23-5efb2d2cc7f2/
├── 32b682ec-8156-44ca-bff0-26155c7fdc12.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── 32b682ec-8156-44ca-bff0-26155c7fdc12.rna_seq.augmented_star_gene_counts.tsv.parcel
```

Representative WSI file tree:

```text
TCGA_LIHC/WSI/0166427c-15c9-4b49-954c-eb3545e5699e/
├── TCGA-BC-A217-01Z-00-DX1.E15D9B32-F809-40D8-846B-2334447B80C0.svs
└── logs/
    └── TCGA-BC-A217-01Z-00-DX1.E15D9B32-F809-40D8-846B-2334447B80C0.svs.parcel
```

---

## TCGA-COAD

Root:

```text
/data/lizhe/Medteam_projects/TCGA-COAD/
```

Top-level tree:

```text
TCGA-COAD/
├── Bulk_RNA/             # Gene data
├── Bulk_RNA_cases/       # Gene case metadata + clinic
├── WSI/                  # WSI data
├── WSI_cases/            # WSI case metadata, no clinic json observed
├── TCGA_info
└── gdc-client
```

Path map:

```yaml
dataset: TCGA-COAD
root: /data/lizhe/Medteam_projects/TCGA-COAD
gene_root: /data/lizhe/Medteam_projects/TCGA-COAD/Bulk_RNA
wsi_root: /data/lizhe/Medteam_projects/TCGA-COAD/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA-COAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json
case_dirs:
  - Bulk_RNA_cases
  - WSI_cases
counts:
  gene_uuid_dirs: 483
  wsi_uuid_dirs: 459
notes:
  - WSI_cases has biospecimen, manifest, sample_sheet, metadata
  - WSI_cases clinical.cart.*.json not observed
```

Representative Gene file tree from user example:

```text
TCGA-COAD/Bulk_RNA/0c62b002-8912-4ce0-a2e5-41bc1e3f9996/
├── annotations.txt
├── f66a9fee-f63a-4398-b242-9c065bbe213d.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── f66a9fee-f63a-4398-b242-9c065bbe213d.rna_seq.augmented_star_gene_counts.tsv.parcel
```

Representative WSI file tree:

```text
TCGA-COAD/WSI/001b7d97-9425-43c3-a9a3-a36cb3d2a591/
├── TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs
└── logs/
    └── TCGA-A6-2686-01Z-00-DX1.0540a027-2a0c-46c7-9af0-7b8672631de7.svs.parcel
```

---

## TCGA-PRAD

Root:

```text
/data/lizhe/Medteam_projects/TCGA-PRAD/
```

Top-level tree:

```text
TCGA-PRAD/
├── Bulk_RNA/             # Gene data
├── Bulk_RNA_cases/       # Gene case metadata + clinic
├── WSI/                  # WSI data
├── WSI_cases/            # WSI case metadata + clinic
├── TCGA_info
└── gdc-client
```

Path map:

```yaml
dataset: TCGA-PRAD
root: /data/lizhe/Medteam_projects/TCGA-PRAD
gene_root: /data/lizhe/Medteam_projects/TCGA-PRAD/Bulk_RNA
wsi_root: /data/lizhe/Medteam_projects/TCGA-PRAD/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA-PRAD/Bulk_RNA_cases/clinical.cart.2026-06-18.json
  - /data/lizhe/Medteam_projects/TCGA-PRAD/WSI_cases/clinical.cart.2026-06-18.json
case_dirs:
  - Bulk_RNA_cases
  - WSI_cases
counts:
  gene_uuid_dirs: 502
  wsi_uuid_dirs: 449
notes:
  - structure is regular
```

Representative Gene file tree:

```text
TCGA-PRAD/Bulk_RNA/0007888f-8d96-4c01-8251-7fef6cc71596/
├── 88215dd0-5841-44f1-9393-eefd8238cbb3.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── 88215dd0-5841-44f1-9393-eefd8238cbb3.rna_seq.augmented_star_gene_counts.tsv.parcel
```

Representative WSI file tree:

```text
TCGA-PRAD/WSI/00784afd-6fc2-4f5e-b07e-0ebb38152339/
├── TCGA-YL-A8SL-01Z-00-DX2.F64D1539-E590-4B51-96B4-1F95CAE6E33A.svs
└── logs/
    └── TCGA-YL-A8SL-01Z-00-DX2.F64D1539-E590-4B51-96B4-1F95CAE6E33A.svs.parcel
```

---

## TCGA-READ

Root:

```text
/data/lizhe/Medteam_projects/TCGA-READ/
```

Top-level tree:

```text
TCGA-READ/
├── Bulk_RNA/             # Gene data
├── Bulk_RNA_cases/       # Gene case metadata + clinic
├── WSI/                  # WSI data
├── WSI_cases/            # WSI case metadata + clinic
├── TCGA_info
└── gdc-client
```

Path map:

```yaml
dataset: TCGA-READ
root: /data/lizhe/Medteam_projects/TCGA-READ
gene_root: /data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA
wsi_root: /data/lizhe/Medteam_projects/TCGA-READ/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA-READ/Bulk_RNA_cases/clinical.cart.2026-06-21.json
  - /data/lizhe/Medteam_projects/TCGA-READ/WSI_cases/clinical.cart.2026-06-21.json
case_dirs:
  - Bulk_RNA_cases
  - WSI_cases
counts:
  gene_uuid_dirs: 167
  wsi_uuid_dirs: 166
notes:
  - structure is regular
```

Representative Gene file tree:

```text
TCGA-READ/Bulk_RNA/00f55a16-0ee5-4939-8efb-de34e68d4ccd/
├── ff11a9e3-d32b-431b-9ebb-c5a3d9eb0e4f.rna_seq.augmented_star_gene_counts.tsv
└── logs/
    └── ff11a9e3-d32b-431b-9ebb-c5a3d9eb0e4f.rna_seq.augmented_star_gene_counts.tsv.parcel
```

Representative WSI file tree:

```text
TCGA-READ/WSI/01c43498-6495-4a01-a60f-dfdebbfc1629/
├── TCGA-F5-6861-01Z-00-DX1.011B771B-F52E-412E-9352-1578349BEAF1.svs
└── logs/
    └── TCGA-F5-6861-01Z-00-DX1.011B771B-F52E-412E-9352-1578349BEAF1.svs.parcel
```

---

## TCGA-STAD

Root:

```text
/data/lizhe/Medteam_projects/TCGA-STAD/
```

Top-level tree:

```text
TCGA-STAD/
├── Bulk_RNA/             # WARNING: contains WSI-like *.svs, not RNA *.tsv
├── Bulk_RNA_cases/       # RNA case metadata + clinic
├── WSI/                  # WSI data
├── WSI_cases/            # WSI case metadata + clinic
├── TCGA_info
├── gdc-client
└── pymp-pbjijnzr         # permission denied during inspection
```

Path map:

```yaml
dataset: TCGA-STAD
root: /data/lizhe/Medteam_projects/TCGA-STAD
gene_root_nominal: /data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA
gene_root_status: invalid_or_suspect
wsi_root: /data/lizhe/Medteam_projects/TCGA-STAD/WSI
clinic_files:
  - /data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA_cases/clinical.cart.2026-06-21.json
  - /data/lizhe/Medteam_projects/TCGA-STAD/WSI_cases/clinical.cart.2026-06-21.json
case_dirs:
  - Bulk_RNA_cases
  - WSI_cases
counts:
  nominal_gene_uuid_dirs: 442
  wsi_uuid_dirs: 442
  bulk_rna_rna_tsv_files: 0
  bulk_rna_svs_files: 442
  bulk_rna_wsi_uuid_overlap: 442
notes:
  - Bulk_RNA is not usable as Gene data as inspected
  - Bulk_RNA appears to mirror WSI
  - for modality stats, count Gene from local RNA *.tsv only
  - current local Gene set is empty, so STAD can still be counted with G = empty
```

Nominal Gene tree, actual content is WSI-like:

```text
TCGA-STAD/Bulk_RNA/0006d12e-8e38-4f64-a339-12a1dfb26fa7/
├── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs
└── logs/
    └── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs.parcel
```

Representative WSI file tree:

```text
TCGA-STAD/WSI/0006d12e-8e38-4f64-a339-12a1dfb26fa7/
├── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs
└── logs/
    └── TCGA-BR-A44T-01Z-00-DX1.46AA24E7-F2C9-418B-90AA-D6DA2896F5DE.svs.parcel
```

Specific checked file:

```yaml
path: /data/lizhe/Medteam_projects/TCGA-STAD/Bulk_RNA/00bb8e3b-d868-4e8f-bb11-1c58b579b25d/logs/TCGA-BR-6710-01Z-00-DX1.9368e11d-9000-4e84-8772-a0ee54932049.svs.parcel
size_bytes: 32416
size_human: 32K
interpretation: small WSI parcel, still WSI-like, not RNA parcel
```

---

## Machine-Readable Summary

```yaml
base_root: /data/lizhe/Medteam_projects
datasets:
  TCGA-BRCA:
    root: /data/lizhe/Medteam_projects/TCGA-BRCA
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - Bulk_RNA_case/clinical.cart.2026-06-29.json
      - WSI_case/clinical.cart.2026-06-29.json
    case_style: singular_case
    special: has Mutation, Mythylation, RPPA

  TCGA_LIHC:
    root: /data/lizhe/Medteam_projects/TCGA_LIHC
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - clinical/clinical.cart.2026-06-01.json
      - clinical.cart.2026-03-26.json
    case_style: no_case_dirs
    special: clinic directory at root

  TCGA-COAD:
    root: /data/lizhe/Medteam_projects/TCGA-COAD
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - Bulk_RNA_cases/clinical.cart.2026-06-18.json
    case_style: plural_cases
    special: WSI_cases clinical json not observed

  TCGA-PRAD:
    root: /data/lizhe/Medteam_projects/TCGA-PRAD
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - Bulk_RNA_cases/clinical.cart.2026-06-18.json
      - WSI_cases/clinical.cart.2026-06-18.json
    case_style: plural_cases
    special: regular

  TCGA-READ:
    root: /data/lizhe/Medteam_projects/TCGA-READ
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - Bulk_RNA_cases/clinical.cart.2026-06-21.json
      - WSI_cases/clinical.cart.2026-06-21.json
    case_style: plural_cases
    special: regular

  TCGA-STAD:
    root: /data/lizhe/Medteam_projects/TCGA-STAD
    gene: Bulk_RNA
    gene_status: suspect_contains_svs_not_rna
    wsi: WSI
    clinic:
      - Bulk_RNA_cases/clinical.cart.2026-06-21.json
      - WSI_cases/clinical.cart.2026-06-21.json
    case_style: plural_cases
    special: Bulk_RNA mirrors WSI; for stats, treat local Gene as empty rather than counting mirrored WSI as Gene

  kindey_cancer_TCGA:
    root: /data/lizhe/Medteam_projects/kindey_cancer_TCGA
    gene: Bulk_RNA
    wsi: WSI
    clinic:
      - clinical/clinical.cart.2026-03-17.json
    case_style: single_clinic_multi_project
    projects:
      - TCGA-KIRC
      - TCGA-KIRP
      - TCGA-KICH
    special: Bulk_RNA filenames are UUID-like; use manifest/API to map file_id to submitter_id
```
