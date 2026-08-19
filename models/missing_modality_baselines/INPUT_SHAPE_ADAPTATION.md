# 输入形状改造说明

这份文件只记录“输入形状”相关的改动。
不记录融合机制、不记录 loss 机制、不记录调度机制。

## 1. HGCN

### 1.1 三路输入宽度改造
- 文件：`third_party/HGCN/HGCN_code/mae_model.py`
- 原来：
  - `fusion_model_mae_2` 只有一个 `in_feats`
  - 三个 `SAGEConv` 共用同一个输入宽度
- 现在：
  - 改成三个显式参数：
    - `img_in_feats=1024`
    - `rna_in_feats=768`
    - `cli_in_feats=512`
- 结果：
  - `img_gnn_2` 读 `1024`
  - `rna_gnn_2` 读 `768`
  - `cli_gnn_2` 读 `512`
- 说明：
  - 只改输入宽度，不改 `SAGEConv` 以外的结构

### 1.2 训练入口同步
- 文件：`third_party/HGCN/HGCN_code/train.py`
- 原来：
  - 实例化模型时只传一个 `in_feats`
- 现在：
  - 显式传入三路输入宽度
- 说明：
  - 这是为了和 1.1 的新签名对齐

### 1.3 图构造脚本
- 文件：`third_party/HGCN/generate_survpgc_graph.py`
- 原来：
  - 只有 notebook 思路，没有当前项目可直接执行的图构造脚本
- 现在：
  - 新增独立脚本
  - 从当前 workspace 特征重建 HGCN 需要的 `Data(...)`
- 具体输入形状：
  - WSI：`x_img: (N_img, 1024)`
  - Gene：`x_rna: (4, 768)`
  - Clinic：`x_cli: (n_c, 512)`
- 具体处理：
  - WSI：
    - 从 `P/uni_v1_h5` 读 `features/coords`
    - 用 `coords` 建 8 邻域边
  - Gene：
    - 从 `G/scFoundation_embedding_gene_raw` 读 `.pt`
    - 保持 `(4, 768)`
  - Clinic：
    - 从 `C/L4` 读 `.pt`
    - 保持 `(n_c, 512)`
- 输出：
  - `patients.pkl`
  - `sur_and_time.pkl`
  - `all_data.pkl`

## 2. Flex-MoE

### 2.1 这里新增的不是“模型分支”，而是“数据路由分支”
- 文件：`third_party/flex-moe/data.py`
- 原来：
  - 只有 ADNI / MIMIC 的原生数据路由
- 现在：
  - 新增 `load_and_preprocess_survpgc_data()`
- 说明：
  - 这不是新增一个新模型
  - 只是给原有 Flex-MoE 增加一个当前项目数据源入口

### 2.2 WSI 输入形状
- 文件：`third_party/flex-moe/data.py`
- 原来：
  - ADNI 的 image 分支走它自己的图像处理逻辑
- 现在：
  - SurvPGC 路由里：
    - 每个 case 的 WSI 先从 `P/uni_v1` 读出 patch 特征
    - 若一个 case 有多张 slide，就先把这些 slide 的 patch 特征拼起来
    - 再对 patch 维做 mean pooling
    - 得到 case 级 WSI 向量 `(1024,)`
    - batch 后是 `(B, 1024)`
    - 再送入原始 `PatchEmbeddings(1024, num_patches, hidden_dim)`
- 说明：
  - 这一步是 pooling，不是 flatten
  - pooling 发生在进入 `PatchEmbeddings` 之前

### 2.3 Gene 输入形状
- 文件：`third_party/flex-moe/data.py`
- 原来：
  - ADNI genomic 分支读它自己的文件
- 现在：
  - 从 `G/scFoundation_embedding_gene_raw/<case_id>.pt` 读
  - 输入 `(4, 768)`
  - flatten 成 `(3072,)`
  - batch 后是 `(B, 3072)`
  - 再进 `PatchEmbeddings(3072, num_patches, hidden_dim)`

### 2.4 Clinic 输入形状
- 文件：`third_party/flex-moe/data.py`
- 原来：
  - ADNI clinical 分支读它自己的文件
- 现在：
  - 从 `C/L4/<case_id>.pt` 读
  - 输入 `(n_c, 512)`
  - flatten 成 `(n_c * 512,)`
  - batch 后是 `(B, n_c * 512)`
  - 再进 `PatchEmbeddings(n_c * 512, num_patches, hidden_dim)`

### 2.5 主入口路由
- 文件：`third_party/flex-moe/main.py`
- 原来：
  - 没有当前项目数据源入口
- 现在：
  - 新增 `--study`
  - `--data survpgc` 时自动使用三模态命名 `WGC`
  - `--data survpgc` 时必须提供 `--split_csv`

## 3. 形状验证

- 已确认的真实 shape：
  - WSI：`(28885, 1024)`
  - Gene：`(4, 768)`
  - Clinic：`(21, 512)`
- `python -m py_compile` 已通过
