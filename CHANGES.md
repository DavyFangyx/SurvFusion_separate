# Changelog: clinic_num_tokens 参数化 & 多 slide 重复加载修复

## 问题背景

不同数据集的 prompt clinic 文件（`clinic_dir/*.pt`）中 token 数 `n` 不同，形如 `[n, 512]`。
原始代码在两处将 `n` 硬编码为 6，并且数据集加载会对多 slide 病例重复拼接 clinic/gene token，
导致 token 数变成 `slide_count × n`，与模型中固定偏移量切片（`mm_embed[:, :num_clinic, :]`）冲突。

---

## 修改文件一览

### 1. `datasets/dataset_survival.py`

**函数** `_load_gene_embs_from_path`（原第 876-885 行）  
**函数** `_load_clinic_embs_from_prompt`（原第 888-906 行）

**改动**：两个函数均由"对每个 slide_id 循环加载再 cat"改为"只用第一个 slide 的 case_id 前缀加载一次"。

**原因**：clinic 和 gene 文件均以 case_id 为单位存储，同一病例的所有 slide 指向同一个 `.pt` 文件。
原实现对多 slide 病例会重复拼接相同内容，产生 `[slide_count×n, 512]`，使模型 forward 中按
固定偏移量切分 token 的逻辑静默出错（clinic token 被混入 WSI 那侧，部分信息丢失）。

---

### 2. `utils/core_utils.py`

**函数** `_init_model`

**改动**：
- 在模型构建前，对 prompt clinic 类 modality（`survpgc_f`、`survgc_f`、`survpc_f`、`clinic_mlp`），
  通过 `glob` 取 `clinic_dir` 中的一个 `.pt` 样本文件，读取其 `shape[0]` 得到 `clinic_num_tokens`。
- 将 `clinic_num_tokens` 传入上述四个模型的构造函数。

**未改动**：`survpc`、`mlppc_concat`（走 one-hot/tabular clinic，与 `[n, 512]` prompt clinic 无关）。

---

### 3. `models/model_SurvPGC_foundation.py`

**类** `SurvPGC_F.__init__`

**改动**：新增参数 `clinic_num_tokens=6`，`self.num_clinic = clinic_num_tokens`（原为硬编码 `6`）。

`forward` 和 `captum` 中的切片逻辑（`mm_embed2[:, :self.num_clinic, :]`）无需修改，已通过 `self.num_clinic` 间接引用。

---

### 4. `models/model_SurvGC_foundation.py`

**类** `SurvGC_F.__init__`

**改动**：同上，新增 `clinic_num_tokens=6`，`self.num_clinic = clinic_num_tokens`。

---

### 5. `models/model_SurvPC_foundation.py`

**类** `SurvPC_F.__init__`

**改动**：同上，新增 `clinic_num_tokens=6`，`self.num_clinic = clinic_num_tokens`。

---

### 6. `models/model_MLPSingle.py`

**类** `MLPSingle.__init__`

**改动**：新增参数 `clinic_num_tokens=6`，将 `to_logits` 输入维度由硬编码的 `projection_dim*6//4`
改为 `projection_dim * clinic_num_tokens // 4`。

**原因**：`self.net` 将 `[n, 512]` 映射到 `[n, projection_dim//4]`，
`data.view(1, -1)` 展平为 `[1, n * projection_dim//4]`，
`to_logits` 的输入维度必须与之匹配，因此依赖 `clinic_num_tokens`。

---

## 未改动范围

| 模型/文件 | 原因 |
|-----------|------|
| `model_SurvPC.py` | 走 one-hot clinic（`clinic_size` 标量输入），不涉及 `[n, 512]` |
| `model_MLPPC_concat.py` | 同上，`clinic_dim` 为标量维度 |
| `model_SNNSingle.py` | 使用 `torch.mean(h, dim=0)` 对 token 聚合，无 token 数依赖 |
| `utils/process_args.py` | `clinic_num_tokens` 改为从数据动态推断，无需 CLI 参数 |
