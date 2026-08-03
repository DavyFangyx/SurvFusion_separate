# WandB ID 规则说明

本文档记录**当前代码实际实现**的 WandB 字段与标签规则，主要给后续 AI / 脚本对照使用。

当前实现来源：
- [utils/wandb_utils.py](/data/fangyuxuan/projects/medical_dl/SurvPGC_github_init/utils/wandb_utils.py)

---

## 1. 两套 run 协议

当前 WandB 上传分成两套协议：

- 普通训练：`run_kind = "train"`
- Optuna 调参：`run_kind = "optuna"`

二者的区别主要体现在：

- `group`
- `name`
- `tags`
- `config` 中的补充字段

---

## 2. 普通训练协议

### 2.1 通用生成规则

若 `args.use_optuna` 不存在或为 `False`，则走普通训练协议。

```text
project = args.wandb_project
entity  = args.wandb_entity

group   = f"{args.exp_group}_train"

stage_tag = stage_name or "single"

job_type = 调用处传入的 job_type

name = f"{args.run_name}_{stage_tag}_fold{fold}"
```

### 2.2 普通训练 tags

```text
tags = [
    args.exp_group,
    "train",
    args.run_name,
    stage_tag,
    f"fold_{fold}",
    args.modality,
    f"poe_{args.poe_variant}",
]
```

### 2.3 普通训练 config 关键字段

```text
config = {
    "experiment_tag": args.exp_group,
    "run_kind": "train",
    "model": args.run_name,
    "stage": stage_tag,
    "fold": fold,
    "modality": args.modality,
    "poe_variant": args.poe_variant,
    ...
}
```

### 2.4 普通训练下各组字段示例

#### Model_A Stage1

```text
project  = args.wandb_project
group    = f"{args.exp_group}_train"
job_type = "stage1_vae_pretrain"
name     = f"{args.run_name}_stage1_fold{fold}"
```

#### Model_A Stage2

```text
project  = args.wandb_project
group    = f"{args.exp_group}_train"
job_type = "stage2_linear_probe"
name     = f"{args.run_name}_stage2_fold{fold}"
```

#### Model_B Stage1

```text
project  = args.wandb_project
group    = f"{args.exp_group}_train"
job_type = "stage1_vae_pretrain"
name     = f"{args.run_name}_stage1_fold{fold}"
```

#### Model_B Stage2

```text
project  = args.wandb_project
group    = f"{args.exp_group}_train"
job_type = "stage2_survival_finetune"
name     = f"{args.run_name}_stage2_fold{fold}"
```

#### Model_C

```text
project  = args.wandb_project
group    = f"{args.exp_group}_train"
job_type = "stage2_joint_train"
name     = f"{args.run_name}_stage2_fold{fold}"
```

说明：
- 当前 Model_C 仍只有一个 run，但沿用 `stage2` 命名。

---

## 3. Optuna 调参协议

### 3.1 通用生成规则

若 `args.use_optuna == True`，则走 Optuna 协议。

其中：

```text
experiment_tag = args.optuna_experiment_tag
model_or_run   = args.optuna_base_run_name
trial_tag      = args.optuna_trial_tag   # 例如 trial_0003
stage_tag      = stage_name or "single"
```

生成规则：

```text
project = args.wandb_project
entity  = args.wandb_entity

group   = f"{experiment_tag}_optuna"

job_type = 调用处传入的 job_type

name = f"{model_or_run}_{trial_tag}_{stage_tag}_fold{fold}"
```

### 3.2 Optuna tags

```text
tags = [
    experiment_tag,
    "optuna",
    model_or_run,
    trial_tag,
    stage_tag,
    f"fold_{fold}",
    args.modality,
    f"poe_{args.poe_variant}",
]
```

### 3.3 Optuna config 关键字段

```text
config = {
    "experiment_tag": experiment_tag,
    "run_kind": "optuna",
    "model": model_or_run,
    "trial": trial.number,
    "trial_tag": trial_tag,
    "stage": stage_tag,
    "fold": fold,
    "modality": args.modality,
    "poe_variant": args.poe_variant,
    ...
}
```

### 3.4 Optuna 下各组字段示例

假设：

```text
base_args.exp_group = "poe_vae_optuna"
base_args.run_name  = "lihc_poeB"
trial.number        = 7
trial_tag           = "trial_0007"
fold                = 2
```

则：

```text
experiment_tag = "poe_vae_optuna"
model_or_run   = "lihc_poeB"
group          = "poe_vae_optuna_optuna"
```

#### Optuna Model_A Stage1

```text
project  = args.wandb_project
group    = f"{experiment_tag}_optuna"
job_type = "stage1_vae_pretrain"
name     = f"{model_or_run}_{trial_tag}_stage1_fold{fold}"
```

#### Optuna Model_A Stage2

```text
project  = args.wandb_project
group    = f"{experiment_tag}_optuna"
job_type = "stage2_linear_probe"
name     = f"{model_or_run}_{trial_tag}_stage2_fold{fold}"
```

#### Optuna Model_B Stage1

```text
project  = args.wandb_project
group    = f"{experiment_tag}_optuna"
job_type = "stage1_vae_pretrain"
name     = f"{model_or_run}_{trial_tag}_stage1_fold{fold}"
```

#### Optuna Model_B Stage2

```text
project  = args.wandb_project
group    = f"{experiment_tag}_optuna"
job_type = "stage2_survival_finetune"
name     = f"{model_or_run}_{trial_tag}_stage2_fold{fold}"
```

#### Optuna Model_C

```text
project  = args.wandb_project
group    = f"{experiment_tag}_optuna"
job_type = "stage2_joint_train"
name     = f"{model_or_run}_{trial_tag}_stage2_fold{fold}"
```

---

## 4. 具体示例

假设普通训练：

```text
wandb_project = "SurvPGC_MultiVAE"
wandb_entity  = "davyfangyuxuan-nanjing-university-of-aeronautics-and-ast"
exp_group     = "poe_vae_test"
run_name      = "model_A"
fold          = 0
```

则普通训练 Model_A：

### Stage1

```text
project  = "SurvPGC_MultiVAE"
group    = "poe_vae_test_train"
job_type = "stage1_vae_pretrain"
name     = "model_A_stage1_fold0"
```

### Stage2

```text
project  = "SurvPGC_MultiVAE"
group    = "poe_vae_test_train"
job_type = "stage2_linear_probe"
name     = "model_A_stage2_fold0"
```

---

## 5. 备注

- Model_A / Model_B 当前会拆成两个独立 WandB run：
  `stage1`
  `stage2`
- 两段 run 的 step 计数器独立，不再共用。
- 普通训练必须带 `train` 标签。
- Optuna 调参必须带 `optuna` 标签。
