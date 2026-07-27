# 三模态 PoE-VAE 生存预测模型 — 设计说明

本文档整合此前讨论中确定的所有设计决策，作为后续撰写 Codex 执行 prompt 的直接依据。凡本文档未覆盖的实现细节，均属于遗留待定项，需在写执行 prompt 前单独确认。

---

## 0. 总体结构

三种模态（WSI / Clinic / Gene）各自通过一个 Encoder 输出对角高斯 (μᵢ, logσ²ᵢ)，经 Generalized PoE 融合为单一联合高斯 (μ_joint, σ_joint)，重参数化采样得到共享隐变量 z（训练时采样，推理时取 μ_joint）。三个 Decoder 从 z 分别重建各模态对应的定长目标向量。下游生存预测 head 复用已有工作实现，不在本文档中设计。

代码组织：共享组件全部放在 `model_utils.py`（Encoder、PoE、重参数化、Decoder、Loss 等基础模块），Model_A / Model_B / Model_C 三个训练模式各自在独立文件中，通过拼装 `model_utils.py` 里的组件构建，不重复实现底层逻辑。

---

## 1. 数据与输入维度

| 模态 | 特征提取器 | 输入维度 | 备注 |
|---|---|---|---|
| WSI | UNI V2 | (n_patch, 768) | n_patch 逐病人变化 |
| Clinic | COACH | (n_c, 768) | n_c 同一实验组内固定，初始化时检测 |
| Gene | scFoundation | (4, 768) | 固定 |

**隐变量维度 d_z = 128，硬编码，不做成超参。**

---

## 2. Encoder 设计

### 2.1 Gene / Clinic — Encoder_TRANSFORMER

不做输入端降维，保持 d_model=768 直接进 Transformer，仅在输出端投影到 d_z：

```
输入 x: (nfeats, B, 768)   # Gene: nfeats=4；Clinic: nfeats=n_c

模块:
1. 可学习 muQuery, sigmaQuery: (768,)
2. 拼接 [muQuery, sigmaQuery, x] -> (nfeats+2, B, 768)
3. TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=1024, activation="gelu")
   num_layers = 1~2
4. 取输出的前两个 token: mu_token, logvar_token  (B, 768)
5. mu    = Linear(768, 128)(mu_token)
   logvar = Linear(768, 128)(logvar_token)
6. logvar.clamp(min=-4, max=2)

输出: mu, logvar  shape = (B, 128)
```

Gene 和 Clinic 共用这一套结构定义，各自单独实例化一份权重（不共享参数），因为 nfeats 不同。

### 2.2 WSI — VIBTrans（NystromAttention 版本）

```
输入 features: (B, n_patch, 768)

1. Padding mask:
   - 将 n_patch pad 到接近正方形 H*W
   - 生成 padding_mask: (B, H*W)，标记哪些位置是真实 patch、哪些是 padding

2. 拼接 muQuery, sigmaQuery token:
   x: (B, H*W+2, 768)

3. TransLayer 1 (NystromAttention)
   - 必须将 padding_mask 传入 attention，屏蔽 padding token 的注意力权重

4. PPEG (卷积式位置编码)
   - 只对 feature token 做，reshape 前需将 padding 位置置 0
   - 卷积后仍按 padding_mask 保持 padding 位置无效（不参与后续统计）

5. TransLayer 2 (NystromAttention，同样传入 padding_mask)

6. LayerNorm

7. 拆分:
   mu_token, logvar_token: (B, 768)
   feat_tokens: (B, H*W, 768)   # 含 padding，需配合 padding_mask 使用

8. mu_wsi    = Linear(768, 128)(mu_token)
   logvar_wsi = Linear(768, 128)(logvar_token)
   logvar_wsi.clamp(min=-4, max=2)

输出:
mu_wsi, logvar_wsi: (B, 128)
h_wsi_tokens: (B, H*W, 768)  + padding_mask   # 供 2.3 节重建目标使用
```

### 2.3 WSI 重建目标（防止表征塌缩，独立池化头）

**不使用 mu_wsi 作为重建目标。** 单独设计一个与 mu 分支不共享参数的池化头：

```
输入: h_wsi_tokens (B, H*W, 768), padding_mask

1. mask 均值池化（只对真实 patch 做平均，忽略 padding）:
   pooled = masked_mean(h_wsi_tokens, padding_mask)   # (B, 768)

2. 独立线性层（参数与 mu/logvar 分支完全分离）:
   wsi_target = Linear(768, 768)(pooled)   # 或直接用 pooled 本身作为 target，不加线性层

输出: wsi_target (B, 768)   # 作为 Decoder 重建的 ground truth
```

该目标向量与 mu_wsi 来自同一 backbone 输出但走独立的投影头，二者参数不共享，避免 decoder 直接照抄 mu_wsi 导致的平凡解。

---

## 3. Generalized PoE 融合

三个模态专家 + 1 个固定先验专家：

```
先验专家: μ0 = 0, τ0 = 1（固定，不参与 softmax，始终存在）

模态专家权重: 每个模态 i 有一个可学习 logit a_i
α_i = softmax(a_1, a_2, a_3)_i   # 仅在该模态可用时参与 softmax

τ_i = 1 / exp(logσ²_i)  (即 1/σ_i²)

τ_joint  = τ0 + Σ_{i ∈ available} α_i * τ_i
μ_joint  = (τ0*μ0 + Σ_{i ∈ available} α_i * τ_i * μ_i) / τ_joint
σ_joint² = 1 / τ_joint
```

**缺失模态处理（训练时的模态 dropout 和推理/验证时的真实缺失，逻辑一致）：**
softmax 只在当前可用（未被 dropout / 未缺失）的模态子集上计算，即分母重新归一化，而不是对缺失模态的 α 置零后不归一。

---

## 4. 重参数化

```
训练: z = μ_joint + σ_joint ⊙ ε,  ε ~ N(0, I)
推理: z = μ_joint   （不采样，避免下游 c-index 抖动）
```

---

## 5. Decoder 设计

Decoder 输入仅为 z，**不拼接任何 condition**：

```
输入: z (B, 128)

每个模态一个独立 decoder_i:
  Linear(128, hidden) -> ELU -> Linear(hidden, output_dim_i)

- decoder_wsi:    output_dim = 768                → 对应 wsi_target (2.3节)
- decoder_gene:   output_dim = 4 * 768，reshape 为 (4, 768)   → 对应原始 Gene token 特征
- decoder_clinic: output_dim = n_c * 768，reshape 为 (n_c, 768) → 对应原始 Clinic token 特征
                  （n_c 在模型初始化时按当前实验组检测值写死到 decoder 里）
```

---

## 6. 损失函数

### 6.1 重建损失（每模态独立学习观测方差）

对每个模态 m 学一个标量可学习参数 logσ²_m：

```
L_rec_m = MSE_m / (2 * σ²_m) + (dim_m / 2) * log(σ²_m)
L_rec   = Σ_m L_rec_m       # m ∈ {wsi, gene, clinic}
```

dim_m 取该模态重建目标的展平总维度（wsi=768，gene=4*768，clinic=n_c*768）。

### 6.2 KL / Jeffreys 散度

第一阶段训练全程使用 Jeffreys 散度（对称化 KL），作用于 joint 后验 q(z|m1,m2,m3) 与先验 N(0,I) 之间：

```
J = 1/2 * Σ_j ( σ_j² + 1/σ_j² - 2 + μ_j² * (1 + 1/σ_j²) )
```

logσ²（各专家及 joint）均需 clamp 到 [-4, 2]。

### 6.3 总损失与 warm-up

```
L = L_rec + β(t) * J

β(t): 前 N_warmup 个 epoch 从 0 线性升到目标值 β_target，之后保持不变
```

（β_target 和 N_warmup 作为超参，具体数值留待调参，不在本设计文档中固定。）

---

## 7. 模态 dropout

- 概率 p = 0.2，每个模态独立 Bernoulli 采样是否丢弃（每个 batch 内每个样本独立）
- 约束：任意样本不能三个模态同时被丢弃，至少保留一个

---

## 8. 训练流程与代码组织

### 8.1 共享组件（model_utils.py）
- `GeneClinicEncoder`（2.1）
- `WSIEncoderVIBTrans`（含 TransLayer / PPEG，2.2）
- `WSITargetPoolingHead`（2.3）
- `GeneralizedPoE`（3）
- `reparameterize`（4）
- `Decoder_Share`（5，三个模态各一份权重）
- `ReconstructionLoss`（6.1，含每模态可学习 logσ²_m）
- `JeffreysDivergence`（6.2）
- `modality_dropout`（7）

### 8.2 Model_A（当前优先实现）
1. 用上述组件构建完整 VAE（Enc + PoE + Dec），无监督训练：`L = L_rec + β(t)*J`，训练期启用模态 dropout。
2. 训练完成后，**冻结 Encoder 与 PoE 的全部参数**。
3. 对每个病人（含验证集中模态缺失的样本），用其可用模态过一遍冻结的 Enc + PoE，取 `z = μ_joint`（不采样）。
4. 在冻结的 z 上训练一个线性探针 head（线性 Cox 或线性 bin-hazard，用于诊断）。
5. 报告 c-index：若 ≈ 0.5，说明重建目标学到的表示对预后无信息，需直接转 Model_C。

### 8.3 Model_B（后续）
- 复用 Model_A 训练好的 Enc/PoE 权重作为初始化，不冻结
- Encoder 学习率 = head 学习率 × 0.1
- 下游 head 换成已有工作的生存预测 head（复用实现，接口见第 9 节）

### 8.4 Model_C（后续）
- Enc + Head 从头联合训练：`L = L_rec + β*J + λ*L_surv`
- λ 需扫 {0.1, 1, 10}

---

## 9. 生存预测 head（不在本文档中设计）

Model_B / Model_C 使用的生存预测 head 直接复用已有工作的实现，届时以"参考 xxxx.py"的形式在 Codex 执行 prompt 中给出源码引用，本设计文档只约定接口：

```
输入: z  (B, 128)
输出: 与参考实现一致（如离散时间 bin 的 hazard logits）
```

Model_A 的线性探针 head 除外——那是一个单独的、极简的线性层，用于诊断，需要自己实现（不复用参考代码）。

---

## 10. 遗留待确认项

- β_target、N_warmup、各 decoder hidden 维度、TransformerEncoderLayer 的 nhead/层数等具体超参数值，尚未固定，写执行 prompt 前需给出默认值。
- Model_A 线性探针具体用线性 Cox 还是线性 bin-hazard，需要确认（建议线性 Cox，实现最简单，作为纯诊断够用）。
- 生存 head 的参考实现文件（xxxx.py）尚未提供，Model_B/C 的执行 prompt 需等该文件确定后再写。
