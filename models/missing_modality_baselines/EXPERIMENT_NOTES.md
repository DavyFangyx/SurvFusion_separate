# 对比模型改造说明

这份文件只记录目前已经实际改过的内容。
每一条都按“原来是什么 / 现在改成什么 / 为什么这么改”来写。

## 1. HGCN

### 1.1 输入维度改造
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/mae_model.py`
- 原来：
  - `fusion_model_mae_2.__init__(in_feats, ...)`
  - 三个 `SAGEConv` 都共用同一个 `in_feats`
  - 默认等价于 `x_img / x_rna / x_cli` 都按 1024 维输入
- 现在：
  - 改成 `fusion_model_mae_2(img_in_feats, rna_in_feats, cli_in_feats, ...)`
  - `img_gnn_2` 输入维度是 `1024`
  - `rna_gnn_2` 输入维度是 `768`
  - `cli_gnn_2` 输入维度是 `512`
- 原因：
  - `z_temp/Table2/部署指南.md` 明确要求 HGCN 三路输入维度改为 `1024 / 768 / 512`
  - 这里只改输入维度，不改图卷积、池化、MAE、Mixer、输出结构

### 1.2 HGCN 训练入口参数同步
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - 实例化模型时传 `in_feats=1024`
- 现在：
  - 改成显式传
    - `img_in_feats=1024`
    - `rna_in_feats=768`
    - `cli_in_feats=512`
- 原因：
  - 与 1.1 的模型签名保持一致
  - 只改实例化参数，不改训练流程

### 1.3 HGCN split 读取方式改造
- 文件：`models/missing_modality_baselines/third_party/HGCN/HGCN_code/train.py`
- 原来：
  - 固定 split 来自 `seed_fit_split.pkl`
- 现在：
  - 新增 `_resolve_split_dir()`
  - 新增 `_load_split_csv()`
  - `if_fit_split=True` 时改为读取 `splits_{fold}.csv`
  - 新增参数：
    - `--split_root`
    - `--split_dir`
- 原因：
  - 你上一轮明确要求这两个模型只用 csv split
  - 这里只改 split 来源，不改 fold 训练逻辑

### 1.4 HGCN 图构造重建为脚本
- 文件：`models/missing_modality_baselines/third_party/HGCN/generate_survpgc_graph.py`
- 原来：
  - 仓库里只有原始 `gendata.ipynb` 思路
  - 没有针对当前 SurvPGC 特征目录的可执行图构造脚本
- 现在：
  - 新增一个独立脚本，把当前项目的冻结特征重建成 HGCN 需要的 `Data(...)`
  - 脚本输出：
    - `patients.pkl`
    - `sur_and_time.pkl`
    - `all_data.pkl`
- 具体做了什么：
  - WSI：
    - 从 `SurvPGC_Workspace/<study>/P/uni_v1_h5` 读 `.h5`
    - 读取 `features`
    - 读取 `coords`
    - 断言 `coords.shape[0] == features.shape[0]`
    - 用 `coords` 按 8 邻域建 `edge_index_image`
  - Gene：
    - 从 `SurvPGC_Workspace/<study>/G/scFoundation_embedding_gene_raw/<case_id>.pt` 读
    - 保持 `(4, 768)`
    - 生成全连接去自环 `edge_index_rna`
  - Clinic：
    - 从 `SurvPGC_Workspace/<study>/C/L4/<case_id>.pt` 读
    - 保持 `(n_c, 512)`
    - 生成全连接去自环 `edge_index_cli`
  - 同时写入：
    - `data_id`
    - `sur_type`
    - `surv_time`
    - `data_type=['img','rna','cli']`
    - `edge_index_model`
- 原因：
  - 你问得对，这一步不是“小调整”，而是把当前项目特征重新封装成 HGCN 的原生图输入
  - 我之前在 md 里提到了“新增脚本”，但没有把脚本里每一处实际构造动作展开写清楚，这里补全

## 2. Flex-MoE

### 2.1 不是“新增 WSI 分支”，而是“新增 SurvPGC 数据路由”
- 文件：`models/missing_modality_baselines/third_party/flex-moe/data.py`
- 原来：
  - 这个第三方 dataloader 只支持它原生的
    - ADNI 路由
    - MIMIC 路由
  - ADNI 的 image 分支是给它自己的影像数据准备的，不是给当前 SurvPGC 的 WSI/Gene/Clinic 冻结特征准备的
- 现在：
  - 我新增的是 `load_and_preprocess_survpgc_data()`
  - 它的含义是：
    - 给 Flex-MoE 增加一个“当前项目数据格式”的加载分支
    - 不是增加一个新的模型结构分支
    - 也不是替换它原有的 MoE 主体
- 原因：
  - ADNI / MIMIC 原 loader 读取的是它原论文自己的数据格式
  - 当前项目要喂给它的是 SurvPGC 的三模态冻结特征，所以必须单独加一个“数据接线分支”

### 2.2 Flex-MoE 的 WSI 输入怎么改的
- 文件：`models/missing_modality_baselines/third_party/flex-moe/data.py`
- 原来：
  - ADNI image 分支有两种：
    - 预处理影像表格特征直接进 `PatchEmbeddings`
    - 原始 3D 影像走 `Custom3DCNN -> PatchEmbeddings`
- 现在：
  - 在新的 SurvPGC 数据路由里，WSI 不再走 `Custom3DCNN`
  - 改成：
    - 从 `SurvPGC_Workspace/<study>/P/uni_v1` 读取每张 slide 的 patch 特征 `(n_patch, 1024)`
    - 同一 case 若有多张 slide，就把这些 slide 的 patch 特征先拼起来
    - 然后对 patch 维做 mean pooling
    - 得到 case 级 WSI 向量 `(1024,)`
    - batch 后就是 `(B, 1024)`
    - 再送入原始 `PatchEmbeddings(1024, num_patches, hidden_dim)`
- 原因：
  - `z_temp/Table2/部署指南.md` 对 Flex-MoE 的明确要求是：
    - WSI 先压成定长 `(B, 1024)`
    - 然后再进 `PatchEmbeddings`
  - 这里的 pooling 只发生在 WSI
  - 这一步不是 flatten

### 2.3 Flex-MoE 的 Gene 输入怎么改的
- 文件：`models/missing_modality_baselines/third_party/flex-moe/data.py`
- 原来：
  - ADNI genomic 分支读它自己的 `h5ad`
- 现在：
  - 改成从
    - `SurvPGC_Workspace/<study>/G/scFoundation_embedding_gene_raw/<case_id>.pt`
    - 读取 `(4, 768)`
  - 然后 reshape / flatten 成 `(3072,)`
  - batch 后就是 `(B, 3072)`
  - 再送入原始 `PatchEmbeddings(3072, num_patches, hidden_dim)`
- 原因：
  - guide 要求 Gene 对 Flex-MoE 走
    - `flatten -> (B, 3072) -> PatchEmbeddings`

### 2.4 Flex-MoE 的 Clinic 输入怎么改的
- 文件：`models/missing_modality_baselines/third_party/flex-moe/data.py`
- 原来：
  - ADNI clinical 分支读它自己的 clinical csv
- 现在：
  - 改成从
    - `SurvPGC_Workspace/<study>/C/L4/<case_id>.pt`
    - 读取 `(n_c, 512)`
  - 然后 flatten 成 `(n_c * 512,)`
  - batch 后是 `(B, n_c * 512)`
  - 再送入原始 `PatchEmbeddings(n_c * 512, num_patches, hidden_dim)`
- 原因：
  - guide 要求 Clinic 对 Flex-MoE 走
    - `flatten -> (B, n_c * 512) -> PatchEmbeddings`

### 2.5 Flex-MoE 的“pooling”和“flatten”不是同一件事
- WSI：
  - 先从变长 patch 序列 `(n_patch, 1024)` 做 mean pooling
  - 结果是定长 `(1024,)`
  - 这里没有 flatten 的意义，因为 pooling 后已经是一维向量
- Gene：
  - 输入本来是 `(4, 768)`
  - 这里没有 pooling，只有 flatten
  - 输出是 `(3072,)`
- Clinic：
  - 输入本来是 `(n_c, 512)`
  - 这里也没有 pooling，只有 flatten
  - 输出是 `(n_c * 512,)`
- 原因：
  - guide 对三种模态的处理要求本来就不同
  - 我之前总结得太压缩，导致“WSI 也 flatten 了”的阅读印象不清楚，这里明确分开写

### 2.6 Flex-MoE 的 split 输入怎么改的
- 文件：`models/missing_modality_baselines/third_party/flex-moe/data.py`
- 原来：
  - 用 `json` split
- 现在：
  - 用 csv split
  - 读取列：
    - `train`
    - `val`
    - `test`
- 原因：
  - 这是你上一轮明确要求的改动

### 2.7 Flex-MoE 主入口怎么改的
- 文件：`models/missing_modality_baselines/third_party/flex-moe/main.py`
- 原来：
  - `--data` 默认 `adni`
  - 没有当前项目数据源路由
  - 没有 `--study`
  - 没有 `--split_csv`
- 现在：
  - 新增 `--study`
  - 已有的 `--split_csv` 被真正用于 `survpgc` 路由
  - `train_and_evaluate()` 里新增：
    - `if args.data == 'survpgc': ...`
  - 若 `args.data == 'survpgc' and args.modality == 'IGCB'`
    - 自动改成 `WGC`
  - 若 `args.data == 'survpgc' and not args.split_csv`
    - 直接报错
- 原因：
  - `IGCB` 是原论文 4 模态命名
  - 当前项目只有三模态 WSI/Gene/Clinic，所以要切到 `WGC`
  - `split_csv` 必须显式给，不然第三方脚本不知道当前跑哪一折

### 2.8 Flex-MoE 里没有改动的部分
- 文件：`models/missing_modality_baselines/third_party/flex-moe/models.py`
- 保持不动：
  - `FlexMoE`
  - `TransformerEncoderLayer`
  - `FMoETransformerMLP`
  - `AddtionalNoisyGate`
  - `PatchEmbeddings`
  - `missing_embeds`
  - `expert_indices -> router -> gate_loss`
- 原因：
  - 你的要求是“除了输入形状之外，其他结构都不能动”
  - 这部分我没有改

## 3. 运行路径兼容

### 3.1 third_party 子目录下直跑时的 import 路径
- 文件：
  - `models/missing_modality_baselines/third_party/flex-moe/data.py`
  - `models/missing_modality_baselines/third_party/flex-moe/main.py`
  - `models/missing_modality_baselines/third_party/HGCN/generate_survpgc_graph.py`
- 原来：
  - 这些脚本默认只按第三方目录自身的相对路径跑
- 现在：
  - 在文件顶部补了 repo root 到 `sys.path`
- 原因：
  - 新增代码引用了当前仓库的
    - `dataset_deployment`
    - `utils`
  - 不补路径会 import 失败

## 4. 验证

- 已执行：
  - `python -m py_compile`
- 通过文件：
  - `third_party/HGCN/HGCN_code/mae_model.py`
  - `third_party/HGCN/HGCN_code/train.py`
  - `third_party/HGCN/generate_survpgc_graph.py`
  - `third_party/flex-moe/data.py`
  - `third_party/flex-moe/main.py`
