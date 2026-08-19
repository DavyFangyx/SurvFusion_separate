# HGCN / Flex-MoE 接入调研报告

## 1. Split 格式

### HGCN
- 原版 `data_split/*.pkl` 是 `list[5]`，每个元素是 `[train_ids, val_ids, test_ids]` 三段。
- 每段内容是 `numpy.ndarray` 的 patient id 列表。
- 证据：[`HGCN_code/train.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py:479>) 直接按 `seed_fit_split[n_fold-1][0/1/2]` 读。
- 现有样例：`models/missing_modality_baselines/third_party/HGCN/data_split/lihc_split.pkl`。

### Flex-MoE
- 原版 split 是 `json`。
- 结构固定为 `{"training": [...], "validation": [...], "testing": [...]}`。
- 证据：[`flex-moe/data.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/flex-moe/data.py:126>)、[`flex-moe/main.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/flex-moe/main.py:118>)。

## 2. Results 格式

### 现有项目
- 当前主线是 `split_*_results.pkl` + `test_result.csv`。
- `test_result.csv` 是 fold 级汇总，包含 `test_cindex / test_cindex_ipcw / test_IBS / test_iauc / test_loss` 等。
- `split_*_results.pkl` 是 patient-level 字典，键为病例 id。
- 证据：[`main.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/main.py:58>)、[`utils/core_utils.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/utils/core_utils.py:1459>)。

### HGCN 原版
- 原版没有你现在这种统一 `test_result.csv` 落盘。
- 训练里真正保存的是：
  - 每折模型权重 `torch.save(...)`
  - `all_gnn_time.pkl`
  - `all_each_model_time.pkl`
- 这两个 pkl 本质上是 `list[seed] -> dict[patient_id -> score]`。
- `prediction()` / `_summary()` 内部能拿到 `patient_results`，结构和你现在的 patient-level pkl 很接近：`time / risk / censorship / clinical / logits`。
- 证据：[`HGCN_code/train.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py:616>)、[`utils/core_utils.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/utils/core_utils.py:1459>)。

### Flex-MoE 原版
- 原版不输出 survival 风格 results。
- 它主要保存 checkpoint：`./saves/seed_*_modality_*.pth`，结果只写进 `./logs/{data}/{modality}.txt`。
- 指标是 `Accuracy / F1 / AUC`，没有 `cindex`。
- 证据：[`flex-moe/main.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/flex-moe/main.py:172>)。

## 3. 缺失模态机制

### HGCN
- `prediction()` 没有单独的“空模态兜底”分支。
- 它把 `use_type/data_type` 当成“当前参与计算的模态集合”，`forward()` 只对其中出现的 `img/rna/cli` 分支做图卷积和池化。
- 也就是说，缺失模态不是在 `prediction()` 里临时补节点，而是要么在上游数据对象里已经被标成“不参与本次计算”，要么在调用 `forward()` 时就不把它传进去。
- 多模态时，MAE 分支会对当前可见模态做 mask/reconstruct，再把补出的表示加回现有模态。
- 单模态训练时如果该模态在样本里不存在，代码直接 `pass`。
- 结论：**缺失不是在 `prediction()` 里补节点，而是靠 `data_type + MAE` 路径处理；没有看到“零向量占位节点”的显式构造。**
- 证据：[`HGCN_code/train.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py:45>)、[`HGCN_code/mae_model.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/mae_model.py:479>)。

### HGCN 的分支
- `forward()` 里实际存在的模态分支只有 `img`、`rna`、`cli`。
- `prediction()` 里额外统计了单模态和双模态结果：`img / rna / cli / imgrna / imgcli / rnacli`。
- 证据：[`HGCN_code/mae_model.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/mae_model.py:479>)、[`HGCN_code/train.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py:549>).

### Flex-MoE
- 缺失处理是软的：
  - observed modality 走 encoder
  - missing modality 用 `missing_embeds` 顶上
- 这和你现有项目的缺失策略是同一类接口语义。
- 证据：[`flex-moe/main.py`](</data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/models/missing_modality_baselines/third_party/flex-moe/main.py:77>)。

## 4. 结论

- HGCN：split 可直接适配为 fold 级 `pkl`，但 results 需要补成你现在的 `test_result.csv + split_*_results.pkl` 形式。
- Flex-MoE：split 需要从你当前 split 体系转成 `json`；results 需要额外做 survival/cindex 语义适配，因为原版只产分类指标。
- 对 HGCN 的“缺 gene”场景，原版不是靠补节点解决，而是靠 `data_type` 控制分支 + MAE 重建表示。若要完全保持图结构不变，需要再确认数据构造阶段是否给缺失模态保留空张量占位。
