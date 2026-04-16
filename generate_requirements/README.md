# generate_requirements

这个目录包含 3 个数据生成脚本，用于为 SurvPGC 项目准备 KICH、KIRC、KIRP 三个亚型的 metadata、clinical 和 5-fold split 文件。

## 文件说明

### 1. generate_clinical_csv.py

用途：
- 读取病人级 clinical JSON。
- 按 [patients_index](../patients_index) 中的三份亚型索引文件，分别生成 3 个 clinical CSV。

默认输出：
- [tcga_kich_clinical.csv](../datasets_csv/clinical_data/tcga_kich_clinical.csv)
- [tcga_kirc_clinical.csv](../datasets_csv/clinical_data/tcga_kirc_clinical.csv)
- [tcga_kirp_clinical.csv](../datasets_csv/clinical_data/tcga_kirp_clinical.csv)

生成字段：
- `case_id`：来自索引 CSV 的 `patient_id`
- `stage`：来自 clinical JSON 中主诊断的 `ajcc_pathologic_stage`，并去掉前缀 `Stage `，例如 `Stage III` -> `III`
- `subtype`：来自 clinical JSON 的 `project.project_id`，取 `TCGA-` 后半段，例如 `TCGA-KIRC` -> `KIRC`
- `grade`：来自 clinical JSON 中主诊断的 `tumor_grade`，缺失时写为 `N/A`

主诊断选择规则：
- 优先选 `diagnosis_is_primary_disease == true` 的诊断记录
- 若没有该标记，则退回使用 `diagnoses` 中第一条记录

注意：
- KICH 和 KIRP 的 `grade` 在源 JSON 中大量缺失，因此结果中会出现大量 `N/A`
- `subtype` 是项目亚型标签，不是更细粒度的病理 subtype

### 2. generate_metadata_csv.py

用途：
- 读取病人级 clinical JSON
- 读取亚型索引 CSV
- 读取本地 WSI patch 文件目录
- 生成每个亚型对应的 metadata CSV

默认输出：
- [tcga_kich.csv](../datasets_csv/metadata/tcga_kich.csv)
- [tcga_kirc.csv](../datasets_csv/metadata/tcga_kirc.csv)
- [tcga_kirp.csv](../datasets_csv/metadata/tcga_kirp.csv)

这份表是“病人信息 + slide 级展开”的表：
- 一个病人有 1 张 slide，就有 1 行
- 一个病人有多张 slide，就有多行

生成字段：
- `case_id`：来自索引 CSV 的 `patient_id`
- `slide_id`：来自 patch 文件名，脚本扫描 patch 目录中的 `*_patches.h5`，替换为 `.svs`
- `age`：优先取 clinical JSON 中 `demographic.age_at_index`；若缺失，则用 `abs(days_to_birth) / 365.25` 近似换算
- `site`：由 `case_id` 的第二段编码生成，例如 `TCGA-B0-5088` -> `B0`
- `survival_months`：优先取 `days_to_death`，否则取主诊断中的 `days_to_last_follow_up`，再否则取 `follow_ups` 中最近可用的 `days_to_follow_up`；最后统一除以 `30.44` 转为月
- `censorship`：若使用 `days_to_death`，记为 `0`；若使用随访时间，记为 `1`
- `is_female`：来自 `demographic.gender`，`female` 记为 `1`，其余记为 `0`
- `oncotree_code`：来自 `project.project_id`，取 `TCGA-` 后半段，例如 `TCGA-KIRP` -> `KIRP`
- `rna_file_name`：直接来自亚型索引 CSV 的 `rna_file_name`

过滤规则：
- 如果病人不在 clinical JSON 中，跳过
- 如果病人没有对应 patch 文件，跳过
- 如果病人无法解析出 survival 时间，跳过

注意：
- `slide_id` 以本地 patch 文件是否存在为准，因此 metadata 表只覆盖“当前可用于训练的 WSI 病例”
- `site` 和 `oncotree_code` 当前不是训练关键字段，其中 `site` 是编码拆分结果，不是严格临床部位字段

### 3. generate_5fold_splits.py

用途：
- 从每个亚型的 metadata CSV 中读取 `case_id`
- 生成对应亚型的 5 折 split 文件

默认输出目录：
- [tcga_kich](../splits/5foldcv/tcga_kich)
- [tcga_kirc](../splits/5foldcv/tcga_kirc)
- [tcga_kirp](../splits/5foldcv/tcga_kirp)

输出格式：
- 每个目录生成 `splits_0.csv` 到 `splits_4.csv`
- 每个 split 文件包含 `train`、`val`、`test` 三列
- 当前实现中 `test` 与 `val` 相同，保持与项目现有读取方式兼容

生成逻辑：
- 先从 metadata 中提取唯一 `case_id`
- 使用固定随机种子打乱
- 按轮转方式分到 5 个 fold
- 每次取 1 个 fold 作为 `val/test`，其余 4 个 fold 作为 `train`

## 默认输入数据来源

- 病人临床 JSON：
  - `/data/lizhe/Medteam_projects/kindey_cancer_TCGA/clinical/clinical.cart.2026-03-17.json`
- 病人索引 CSV：
  - [A_both_KICH.csv](../patients_index/A_both_KICH.csv)
  - [A_both_KIRC.csv](../patients_index/A_both_KIRC.csv)
  - [A_both_KIRP.csv](../patients_index/A_both_KIRP.csv)
- WSI patch 目录：
  - `/data/fangyuxuan/projects/medical_dl/trident_project/TRIDENT_workspace/20.0x_256px_0px_overlap/patches`

## 运行方式

在当前目录执行：

```bash
python generate_clinical_csv.py
python generate_metadata_csv.py
python generate_5fold_splits.py
```

也可以按单个亚型运行，例如：

```bash
python generate_clinical_csv.py --subtype kirc
python generate_metadata_csv.py --subtype kirc
python generate_5fold_splits.py --subtype kirc
```

## 当前已知情况

- KICH、KIRC 生成结果与索引数量对齐
- KIRP 有 3 个病例缺少本地 WSI patch，因此不会进入 metadata 和 split
- clinical 文件中的 `grade` 缺失情况主要取决于源 clinical JSON，而不是脚本错误