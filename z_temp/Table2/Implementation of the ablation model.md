# 缺失模态基线实现与改装说明（面向执行 AI）

## 0. 任务定位与铁律

本项目主模型为三模态 PoE-VAE 生存预测（WSI/Clinic/Gene）。本文档规定
六个对比基线如何改装接入本仓库。

铁律（违反即返工）：
1. **机制本体不许动**：每个基线赖以成立的融合/缺失机制必须保持原始实现，
   不得"顺手优化"、不得替换等价实现、不得改超参默认值以外的结构。
2. **除机制本体外，一切外围必须统一**：特征来源、pooling、mask 约定、
   生存头、loss、优化器、训练预算、split、评估口径。
3. 任何一处无法统一的地方，**在代码注释和实验记录里显式标注**，不得静默处理。
4. 不得引入主模型没有的额外信息（例如缺失指示位、模态 ID embedding、
   额外的对比学习目标），除非该机制是这个基线本身自带的。

---

## 1. 统一基础设施（六个基线全部遵守）

### 1.1 特征与数据口径

- 三模态特征全部为**冻结的预提取特征**，基线不得对特征提取器做任何微调。
- 各模态输入维度如下（固定，硬编码入模型 `__init__`）：

| 模态 | 特征提取器 | 输入维度 | 备注 |
|---|---|---|---|
| WSI | UNI 1 | `(n_patch, 1024)` | n_patch 逐病人变化 |
| Clinic | COACH | `(n_c, 512)` | n_c 同一实验组内固定，初始化时检测 |
| Gene | scFoundation | `(4, 768)` | 固定 |

- 隐变量维度 **`d_z = 128`**，硬编码，不做成超参。
- 复用现有 `survtri_*` 系列的 dataset / collate_fn，WSI 走 padding + padding_mask。
- split 复用 `splits/5foldcv/<study>/splits_*.csv`，五折，不重新划分。

### 1.2 模态可用性约定（所有基线共用同一份）

    avail: Dict[str, BoolTensor]  # {'wsi': (B,), 'gene': (B,), 'clinic': (B,)}
    # True = 该样本该模态可用；恒定保证每个样本至少一个 True

- 7 个非空子集：`W / G / C / WG / WC / GC / WGC`。
- 每个基线负责把这份 `avail` 翻译成自己原生的缺失接口（见第 3 节），
  **不得反向修改 `avail` 的语义**。
- 推理期缺失评估：按 7 子集分组报告 c-index，并报 Δ 列（相对全模态的下降）。
- 训练期缺失（Table 3）：沿用 EMMS 协议，总缺失率 60%，三向分配
  `(60,0,0)/(0,60,0)/(0,0,60)/(20,20,20)/(30,30,0)`，site-stratified 5 折。
  缺失 mask 由 dataloader 统一生成并下发，六个基线共用同一份 mask 文件，
  保证同一 fold 同一样本在所有基线上缺的是同一批模态。

### 1.3 统一生存头（复用本项目已有实现）

融合表示 h (B, d_fuse) -> fuse_fc -> classifier -> Cox risk score (B, 1)

- `bag_loss = cox_surv`，`label_dim = 1`。
- 各基线自带的预测头（HGCN 的 Cox 头、Flex-MoE 的分类头、MultiVae 的
  生成式目标下游）**一律替换为上述统一头**，并在实现文件顶部注明
  "原实现的 XXX head 已替换为 项目统一 Cox head（为保证可比性）"。
- 生成式基线（MultiVae 系）保留自身的重建 + KL 目标，
  总 loss = 原目标 + λ_surv * L_cox，λ_surv 取固定值 1.0（不调参）。

### 1.4 训练与评估协议

- 优化器、lr scheduler、weight decay、early stopping 规则与主模型一致。
- best checkpoint 从 `epoch >= 10` 起开始保存（与现有逻辑一致），
  因此 `--max_epochs` 不得低于 11。
- 每个基线使用与主模型相同的训练预算（epochs、batch size 等）。
- 最终报告：5 癌种（brca/coad/kirc/kirp/lihc）× 5 折，均值 ± 标准差。

### 1.5 接入方式

- 每个基线一个独立文件，放 `models/` 下，通过 `main.py --modality <name>` 路由：
  `mvae_poe` / `mopoe` / `flex_moe` / `hgcn` / `concat_zero` / `concat_mean`
- 日志键名规范：`loss/*`、`z/*`，生存阶段记 `loss/surv`，验证记 `val/c_index`。
- 第三方代码（MultiVae / flex-moe / HGCN）以 `third_party/<name>/` 形式原样保留，
  改装逻辑写在本仓库的 wrapper 文件里，**不直接改第三方源码**；
  确需改动时用最小 patch 并在 `third_party/<name>/PATCHES.md` 记录每一处。

---

## 2. Concat 地板组（`concat_zero` / `concat_mean`）完整规格

### 2.1 结构

WSI (B, n_patch, 1024) + mask ──mean-pool over valid patches──> x_w (B, 1024)
Gene (B, 4, 768) ──flatten──────────────────────> x_g (B, 3072)
Clin (B, n_c, 512) ──flatten──────────────────────> x_c (B, n_c * 512)

x_m ──[若该模态缺失，整条向量替换为 fill_m]──> MLP_m ──> h_m (B, d_z)

h = concat(h_w, h_g, h_c) (B, 3*d_z) ──> fuse_fc ──> classifier ──> risk

- pooling 方式**必须与现有单模态基线保持一致**：WSI 用 masked mean pooling
  （对齐 `mlp_wsi`），Gene / Clinic 用 flatten（对齐 `mlp_gene_f` /
  `mlp_clinic_flatten`，你的实验已证明 flatten 显著优于 mean）。
  禁止在这里使用 attention pooling —— 那会变成 ABMIL，不再是地板。
- `MLP_m` = `Linear(D_m, d_z) -> ReLU -> Dropout -> Linear(d_z, d_z)`，
  最多两层，禁止加深。
- 除填充常数外，`concat_zero` 与 `concat_mean` **必须是同一份代码**，
  由 `--concat_impute {zero, mean}` 切换。

### 2.2 填充规则（本组的全部实验内容都在这里）

1. **填充位置：MLP 之前，原始/pooled 特征空间。**
   绝不允许"先过 MLP 再把输出置零"——带 bias 的线性层会把零输入映射成一个
   常数向量，等价于隐式的可学习 missing token，破坏地板语义。
2. `zero` 模式：`fill_m = zeros(D_m)`。
3. `mean` 模式：`fill_m = 该 fold 训练集上、该模态可用样本的 pooled 特征均值`。
   - **每折单独计算**，只用 train split，不得混入 val/test（防泄漏）。
   - 在训练开始前一次性算好，`register_buffer` 存入模型，随 ckpt 保存，
     val/test 直接读 buffer，不得在推理时重算。
4. **禁止对输入特征做逐维标准化（z-score）**。若用 train mean 标准化，
   均值即为 0，两组会在数值上完全等价，实验作废。
   需要归一化时只能用 `LayerNorm`（作用在 MLP 内部、填充之后）。
5. **禁止拼接缺失指示位 / mask embedding / 模态 ID**。地板就是要没有这些。

### 2.3 训练与测试的缺失设置

- Table 1 / Table 2 场景：**全模态训练**（所有 `avail` 置 True，不做模态 dropout），
  仅在测试时按 7 子集触发填充。
- Table 3 场景：训练期即按统一下发的缺失 mask 触发填充，逻辑与测试期完全一致。
- 提供 `--concat_train_missing {none, protocol}` 开关区分这两种场景。

### 2.4 代码模板

    import torch
    import torch.nn as nn

    def masked_mean(x, mask):
        """x: (B, N, D); mask: (B, N) True=valid"""
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(1) / m.sum(1).clamp(min=1.0)

    class ModalityMLP(nn.Module):
        """每模态一个,最多两层,禁止加深"""
        def __init__(self, d_in, d_z, dropout=0.25):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, d_z),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_z, d_z),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class ConcatBaseline(nn.Module):
        """
        Concat + zero/mean imputation 地板基线。
        zero 与 mean 两组共用本类,唯一差别是 fill_* buffer 的取值。
        """

        def __init__(self, d_wsi=1024, d_gene_flat=3072, d_clinic_flat=None,
                     d_z=128, mmhid=256, dropout=0.25,
                     impute="zero", label_dim=1):
            super().__init__()
            assert impute in ("zero", "mean")
            self.impute = impute
            self.d_clinic_flat = d_clinic_flat

            self.mlp_wsi    = ModalityMLP(d_wsi,         d_z, dropout)
            self.mlp_gene   = ModalityMLP(d_gene_flat,   d_z, dropout)
            self.mlp_clinic = ModalityMLP(d_clinic_flat, d_z, dropout)

            self.register_buffer("fill_wsi",    torch.zeros(d_wsi))
            self.register_buffer("fill_gene",   torch.zeros(d_gene_flat))
            self.register_buffer("fill_clinic", torch.zeros(d_clinic_flat))

            self.fuse_fc = nn.Sequential(
                nn.Linear(3 * d_z, mmhid),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(mmhid, label_dim)

        @torch.no_grad()
        def set_impute_stats(self, stats: dict):
            """
            训练开始前调用一次。stats 由 train split 单独统计:
            {'wsi': (1024,), 'gene': (3072,), 'clinic': (n_c*512,)}
            impute='zero' 时不调用(保持全零)。
            """
            assert self.impute == "mean", "zero 模式不得写入均值统计量"
            self.fill_wsi.copy_(stats["wsi"])
            self.fill_gene.copy_(stats["gene"])
            self.fill_clinic.copy_(stats["clinic"])

        @staticmethod
        def _fill(x, avail, fill_vec):
            """avail: (B,) bool。整条向量替换,发生在 MLP 之前。"""
            keep = avail.view(-1, 1).to(x.dtype)
            return x * keep + fill_vec.unsqueeze(0) * (1.0 - keep)

        def forward(self, wsi, wsi_mask, gene, clinic, avail):
            x_w = masked_mean(wsi, wsi_mask)          # (B, 1024)
            x_g = gene.flatten(1)                     # (B, 3072)
            x_c = clinic.flatten(1)                   # (B, n_c*512)

            x_w = self._fill(x_w, avail["wsi"],    self.fill_wsi)
            x_g = self._fill(x_g, avail["gene"],   self.fill_gene)
            x_c = self._fill(x_c, avail["clinic"], self.fill_clinic)

            h = torch.cat([self.mlp_wsi(x_w),
                           self.mlp_gene(x_g),
                           self.mlp_clinic(x_c)], dim=1)

            risk = self.classifier(self.fuse_fc(h))   # (B, label_dim)
            return risk

---

## 3. 其他四个基线的改装说明

### 3.1 MultiVae-PoE（即 MVAE）— 成本：低

- **保留不动**：Product-of-Experts 联合后验（缺失时只乘可用专家）、
  VAE 目标（重建 + KL）、`MultimodalBaseDataset / IncompleteDataset` 数据协议。
- **改装**：
  - encoder 换成本项目的投影 encoder：冻结特征 → 隐维 d_z 的对角高斯后验
    （三模态各一份权重）。WSI 变长序列先按 2.1 的 masked mean pooling 压成定长，
    或复用主模型 MIL resampler —— 二选一但**六个基线必须选同一种**并注明。
  - decoder 重建的目标是冻结特征本身（不是像素/token 原始输入）。
  - 共享隐 z 上挂项目统一 Cox 头，总 loss = ELBO + λ_surv * L_cox，
    λ_surv = 1.0（固定）。
  - 跑本项目训练循环，不用 MultiVae 自带 trainer。
- **缺失接口**：原生 `masks={模态: BoolTensor}` + `IncompleteDataset`，
  由 1.2 的 `avail` 直接转换。

### 3.2 MultiVae-MoPoE — 成本：低

- **保留不动**：先对每个非空模态子集做 PoE、再对所有子集后验做 MoE 的聚合规则，
  以及 `joint_elbo` 目标。
- **改装**：与 3.1 共用**完全同一套**投影 encoder / decoder / 生存头。
  两者之间的唯一差别必须只有模型类的选择。
- **缺失接口**：同 3.1；缺失时只枚举可用模态的子集。

### 3.3 Flex-MoE — 成本：中

- **保留不动**：`FixedFMoE` / `FMoETransformerMLP` / `AddtionalNoisyGate`
  的 MoE 路由逻辑、missing-modality bank、`assign_expert` /
  `set_full_modality` 的调用时序、`gate_loss` 及其在总 loss 中的权重。
- **改装**：
  - `Custom3DCNN` + `PatchEmbeddings` 换成线性/MLP 投影：
    冻结特征 → `d_z` 的 token 序列（token 数与原实现一致）。
  - 分类头 → 项目统一 Cox 头；分类 loss → `cox_surv`；
    `gate_loss` 保留并按原权重加入总 loss。
- **缺失接口**：原生 `is_full_modality` + 模态掩码，7 子集映射到它的掩码格式。
  注意 `set_full_modality` 必须在每个 batch 前按该 batch 的实际情况正确设置。

### 3.4 HGCN — 成本：高（PyG 格式转换是主要工程量）

- **保留不动**：`SAGEConv` 模态图卷积、`my_GlobalAttention` 池化、
  `PretrainVisionTransformer` 在线 MAE（这是它的缺失机制，绝不能绕过）、
  `MixerBlock`。
- **改装**：
  - 冻结特征直接作为 `img / rna / cli` 三类节点特征。
  - 写一个 PyG 图转换器（仿其 `gendata.ipynb`），把每个病人封成图对象；
    这是本基线的主要工作量，建议先做单癌种小样本打通再全量转换。
  - **自带的 Cox 头换成项目统一头 + 统一 loss，并在文件顶部和论文附录中注明**
    （HGCN 原本就是生存模型，替换头部是为了严格可比，必须说明这一点）。
- **缺失接口**：原生 `use_type` / `in_mask` / `train_use_type`，
  7 子集直接对应它的单模态 / 双模态 / 全模态分支，不需要额外映射逻辑。
