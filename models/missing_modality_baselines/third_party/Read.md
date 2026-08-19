# 五个项目模型与代码结构简述

本文简要归纳 `flex-moe`、`HGCN`、`MoPoE`、`MVAE`、`MultiVae` 五个项目中包含的主模型、基线/分支模型、模型接口与工具代码。

## 1. flex-moe

`flex-moe` 的主模型是 `FlexMoE`，用于处理任意模态组合与缺失模态场景。

- 主模型：`FlexMoE`
- 核心结构：Transformer 编码层 + Sparse MoE 专家路由 + 缺失模态 bank + MLP 预测头
- 模态分支：`image`、`genomic`、`clinical`、`biospecimen`
- 编码器：
  - `Custom3DCNN`：原始 3D 医学图像编码
  - `PatchEmbeddings`：表格特征或预处理图像特征转 patch token
  - `MLP`：预测头或普通前馈层
- MoE 组件：
  - `FixedFMoE`
  - `FMoETransformerMLP`
  - `AddtionalNoisyGate`

主要接口：

- `FlexMoE.forward(*inputs, expert_indices=None, is_full_modality=None)`
- `FlexMoE.gate_loss()`
- `FlexMoE.assign_expert(combination)`
- `FlexMoE.set_full_modality(is_full_modality)`

工具与入口代码：

- `main.py`：训练、验证、测试入口
- `data.py`：ADNI/MIMIC 数据加载、缺失模态处理、DataLoader 构建
- `utils.py`：随机种子、日志工具
- `run_adni_igcb.sh`：ADNI 示例运行脚本

## 2. HGCN

`HGCN` 的主模型是 `fusion_model_mae_2`，用于多模态癌症生存预测。

- 主模型：`fusion_model_mae_2`
- 核心结构：图卷积网络 + 模态内 attention pooling + 在线 masked autoencoder + Cox 生存预测头
- 模态分支：`img`、`rna`、`cli`
- 单模态分支：`img`、`rna`、`cli`
- 双模态分支：`imgrna`、`imgcli`、`rnacli`
- 全模态融合：`img + rna + cli`
- 关键组件：
  - `SAGEConv`：模态图卷积
  - `my_GlobalAttention`：模态特征池化
  - `PretrainVisionTransformer`：MAE 缺失模态建模
  - `MixerBlock`：MAE 输出后的模态混合

主要接口：

- `fusion_model_mae_2.forward(all_thing, train_use_type=None, use_type=None, in_mask=[], mix=False)`
- 返回 `(one_x, multi_x), save_fea, (att_2, att_3), fea_dict`
- `use_type` 控制当前使用的模态组合
- `train_use_type` 控制训练阶段的完整模态集合

工具与入口代码：

- `HGCN_code/train.py`：主训练、交叉验证、C-index 评估
- `HGCN_code/mae_model.py`：HGCN 与 MAE 主体模型
- `HGCN_code/mae_utils.py`：mask 生成、Transformer block、位置编码
- `HGCN_code/util.py`：C-index、患者信息、日志、学习率调整
- `cut_and_pretrain/cut_and_pretrain.py`：病理 WSI 切 patch 与预训练准备
- `gendata.ipynb`：数据封装为 PyTorch Geometric 格式

## 3. MultiVae

`MultiVae` 是一个较完整的多模态 VAE Python 包，核心不是单个实验脚本，而是一套可复用的模型、训练器、数据集、评估器和采样器框架。

- 统一输入形式：
  - 所有多模态模型的训练/推理输入都是 `MultimodalBaseDataset` 或 `IncompleteDataset`
  - 核心字段是 `data={模态名: Tensor}`，例如 `{"mnist": x1, "svhn": x2}`、`{"image": x_img, "text": x_txt}` 或 `{"m0": x0, "m1": x1, ...}`
  - 缺失模态数据使用 `masks={模态名: BoolTensor}` 标记每个样本的模态是否可用
  - 每个模态需要配套一个 encoder 和 decoder，通常以 `encoders={模态名: encoder}`、`decoders={模态名: decoder}` 传入模型
  - 因此 `MultiVae` 中的 `MVAE`、`MMVAE` 等不是固定“文本+图像”网络；文本/图像分支取决于用户传入的 encoder/decoder

- 主框架：
  - `BaseModel`
  - `BaseMultiVAE`
  - `BaseMultiVAEConfig`
- 自动加载接口：
  - `AutoModel`
  - `AutoConfig`
- 已实现模型：
  - `MVAE`
  - `MMVAE`
  - `MMVAEPlus`
  - `MoPoE`
  - `JMVAE`
  - `JNF`
  - `TELBO`
  - `MVTCAE`
  - `Nexus`
  - `CVAE`
  - `MHVAE`
  - `DMVAE`
  - `CMVAE`
  - `CRMVAE`

已实现模型的输入与核心结构：

- `MVAE`
  - 输入：任意多个模态的 `data` 字典；可使用完整模态，也可在支持的设置下使用可观测模态子集
  - 核心结构：每个模态独立 encoder，使用 Product-of-Experts 聚合为共享 latent space，再由各模态 decoder 重建
  - 典型用途：弱监督多模态生成、缺失模态推理、跨模态生成
- `MMVAE`
  - 输入：任意多个模态的 `data` 字典；缺失模态时只对可用模态做 mixture
  - 核心结构：每个模态一个 encoder/expert，使用 Mixture-of-Experts 聚合后进行多模态重建，训练中使用 IWAE/DReG 形式的目标
  - 典型用途：多模态生成、模态间翻译、部分模态输入
- `MMVAEPlus`
  - 输入：任意多个模态的 `data` 字典；支持部分可观测模态
  - 核心结构：共享 latent `z` + 每个模态私有 latent `w_m`，共享部分使用 MoE，私有部分使用模态专属先验
  - 典型用途：保留共享语义，同时建模模态私有信息
- `MoPoE`
  - 输入：任意多个模态的 `data` 字典；缺失时只枚举可用模态子集
  - 核心结构：对每个非空模态子集先做 Product-of-Experts，再对所有子集 posterior 做 Mixture
  - 典型用途：任意模态组合下的联合生成与跨模态生成
- `JMVAE`
  - 输入：完整多模态 `data` 字典；训练阶段依赖 joint encoder，一般不适合直接训练缺失模态样本
  - 核心结构：joint encoder 编码所有模态，同时训练 unimodal surrogate encoders 与各模态 decoders
  - 典型用途：联合多模态表示学习、双模态或多模态生成
- `JNF`
  - 输入：完整多模态 `data` 字典；依赖 joint encoder，不能直接用于部分可观测训练
  - 核心结构：JMVAE 类结构 + unimodal encoders 使用 normalizing flows，通常用多阶段训练器训练
  - 典型用途：提升单模态 posterior 表达能力
- `TELBO`
  - 输入：完整多模态 `data` 字典；依赖 joint encoder 与 unimodal encoders
  - 核心结构：joint encoder + unimodal encoders + Triple ELBO，两阶段训练
  - 典型用途：视觉-语言等多模态生成任务
- `MVTCAE`
  - 输入：任意多个模态的 `data` 字典；支持部分缺失模态
  - 核心结构：类似 MVAE 的 PoE 聚合，但目标函数来自 Total Correlation Analysis，并加入相关正则项
  - 典型用途：多视角表示学习
- `Nexus`
  - 输入：任意多个模态的 `data` 字典；可在可用模态上计算损失
  - 核心结构：两级 latent，底层是模态专属 latent，上层是共享 latent，并带 forced perceptual dropout 思路
  - 典型用途：层级式跨模态推理
- `CVAE`
  - 输入：主模态 `y` 与条件模态 `x`，由 `main_modality` 和 `conditioning_modalities` 指定
  - 核心结构：conditional encoder、conditional decoder，可选条件 prior network，建模 `p(y|x)`
  - 典型用途：条件生成，例如由一个或多个条件模态生成目标模态
- `MHVAE`
  - 输入：任意多个模态的 `data` 字典
  - 核心结构：层级 latent groups，多层 posterior/prior block，并在每一层使用 PoE 融合多模态信息
  - 典型用途：医学图像等层级多模态生成
- `DMVAE`
  - 输入：任意多个模态的 `data` 字典；支持部分缺失模态
  - 核心结构：共享 latent `z_s` + 模态私有 latent `z_p_i`，共享 posterior 使用 PoE
  - 典型用途：共享/私有因素解耦的多模态表示学习
- `CMVAE`
  - 输入：任意多个模态的 `data` 字典；支持部分缺失模态
  - 核心结构：基于 `MMVAEPlus`，在共享 latent 上加入 Gaussian mixture prior 和聚类变量
  - 典型用途：多模态生成聚类
- `CRMVAE`
  - 输入：任意多个模态的 `data` 字典；支持不完整数据
  - 核心结构：基于 MVTCAE，使用 PoE joint posterior，并额外加入 unimodal reconstruction 项
  - 典型用途：不完整多模态数据下的鲁棒生成与重建

如果具体实验使用图文数据，分支通常可以写成：

- 核心结构：文本 encoder + 图像 encoder + 共享 latent space + 文本/图像 decoder
- 文本分支：Embedding / RNN / Transformer / MLP 等，由实验代码自定义
- 图像分支：CNN / ResNet / MLP / 图像 embedding encoder 等，由实验代码自定义
- 注意：`MultiVae` 本身不内置 FND 分类头；FND 分类头属于单独 `MVAE` 目录中的假新闻检测实现
- 神经网络分支：
  - `models/nn/default_architectures.py`
  - `models/nn/mmnist.py`
  - `models/nn/svhn.py`
  - `models/nn/cub.py`
  - `models/nn/base_architectures.py`

主要接口：

- `BaseModel.forward(inputs, **kwargs)`
- `BaseMultiVAE.encode(...)`
- `BaseMultiVAE.predict(...)`
- `BaseMultiVAE.forward(inputs, **kwargs)`
- `BaseMultiVAE.generate_from_prior(n_samples, **kwargs)`

工具与入口代码：

- `src/multivae/models/`：所有模型、配置类、基础类
- `src/multivae/trainers/`：基础训练器与多阶段训练器
- `src/multivae/data/`：多模态数据集、缺失模态数据集、MMNIST、MNIST-SVHN、CUB 等数据封装
- `src/multivae/metrics/`：coherence、FID、likelihood、latent clustering、reconstruction、visualization 等评估器
- `src/multivae/samplers/`：Gaussian Mixture、MAF、IAF 等采样器
- `examples/`：不同模型的训练示例，包括 `cmvae`、`crmvae`、`dmvae`、`mhvae`、`mmvae_plus`、`mopoe`、`mvtcae`
- `tests/`：模型、数据集、训练器、评估器和采样器测试
- `docs/`：文档与 API 说明

## 总体对比

| 项目 | 主模型 | 主要任务 | 基线/分支形式 |
| --- | --- | --- | --- |
| `flex-moe` | `FlexMoE` | 缺失模态下的多模态分类 | 不同模态组合、各模态 encoder、MoE experts |
| `HGCN` | `fusion_model_mae_2` | 癌症生存预测 | 单模态、双模态、全模态分支 |
| `MoPoE` | `BaseMMVae` 及其子类 | 多模态生成与表示学习 | `poe`、`moe`、`jsd`、`joint_elbo` |
| `MVAE` | `MVAE` | 图文假新闻检测与表示学习 | 文本分支、图像分支、重建分支、分类分支 |
| `MultiVae` | `BaseMultiVAE` 及多个子模型 | 通用多模态 VAE 建模、训练、评估 | `MVAE`、`MMVAE`、`MoPoE`、`JMVAE`、`Nexus` 等模型族 |
