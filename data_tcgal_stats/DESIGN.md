# TCGA PCG 模态统计设计

目标：按病人级 `submitter_id` 统计 P/WSI、C/Clinic、G/Gene 三路模态可用情况，并输出 7 类患者分布。

## 数据集

- 统计：`TCGA_LIHC`、`TCGA-BRCA`、`TCGA-COAD`、`TCGA-PRAD`、`TCGA-READ`、`TCGA-STAD`
- 统计：`kindey_cancer_TCGA` 按 `project_id` 拆成 `TCGA-KIRC`、`TCGA-KIRP`、`TCGA-KICH`

## 模态定义

- `P`：WSI，基于本地 `WSI/**/*.svs` 文件判断可用。
- `C`：Clinic，基于所有配置的 `clinical.cart.*.json` 中 `submitter_id` 并集判断可用。
- `G`：Gene，基于本地 `Bulk_RNA/**/*.rna_seq.augmented_star_gene_counts.tsv` 文件判断可用。

## ID 规则

- 主键：病人级 `submitter_id`，例如 `TCGA-AA-3561`。
- WSI：优先从 metadata/sample sheet 映射；没有时从 `.svs` 文件名提取前三段 TCGA ID。
- Gene：优先从 `gdc_sample_sheet.*.tsv` 的 `Case ID` 映射；其次从 `metadata.cart.*.json` 的 associated entities 映射。
- `TCGA_LIHC` 的 RNA 本地没有 sample sheet；脚本默认会根据 manifest 的 file id 向 GDC API 补 file-to-case 映射。
- `kindey_cancer_TCGA` 按 `project_id` 拆成 `TCGA-KIRC`、`TCGA-KIRP`、`TCGA-KICH` 三个子目录。
- `TCGA-STAD` 的本地 `Bulk_RNA/` 没有 RNA `.tsv`，统计时将 Gene 视为空集，而不会把镜像的 `.svs` 误当作 Gene。
- 不允许把 file UUID 或下载目录 UUID 当作病人 ID。

## 分类

- `PCG`：全部完整（PCG）
- `CG`：仅缺 P（CG）
- `PG`：仅缺 C（PG）
- `PC`：仅缺 G（PC）
- `G`：缺 PC（仅 G）
- `C`：缺 PG（仅 C）
- `P`：缺 CG（仅 P）

分母为 P/C/G 三表 outer join 后的病人全集。

## 输出

- `data_tcgal_stats/汇总.csv`：所有已统计数据集的 7 类汇总长表。
- `data_tcgal_stats/<dataset>/<dataset>_patients.csv`：病人级 P/C/G 明细。
- `data_tcgal_stats/<dataset>/<dataset>_<class>.csv`：每个分类一个患者表，包含 `submitter_id` 和可获得的 `case_uuid`。
