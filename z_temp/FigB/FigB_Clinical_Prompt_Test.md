# FigB Clinical Prompt Test

在仓库根目录执行。下面保留一条最小可用的 `python main.py` 测试命令。

## 通用说明

- 实验组目录：`results/FigB_Clinical Prompt Test/`
- 数据集范围：全部 `enabled` 数据集
- Clinic 特征范围：`L0-L5` 与 `D0-D5`
- 模型范围：`mlp/snn × mean/flatten`
- 交叉验证：`5foldcv`

## 单条测试命令

示例：`tcga_read + D0 + mlp_clinic_mean`

```bash
python main.py \
  --study tcga_read \
  --exp_group 'FigB_Clinical Prompt Test' \
  --run_name tcga_read__D0 \
  --clinic_dir ./SurvPGC_Workspace/tcga_read/C/D1 \
  --modality snn_clinic_mean \
  --max_epochs 12 \
  --wandb_mode disabled
```

## 替换规则
clinic_mlp
clinic_snn
- 改数据集：同步替换 `--study`、`--run_name`、`--clinic_dir`
- 改 Clinic 编码：同步替换 `--run_name` 和 `--clinic_dir` 里的 `Lx/Dx`
- 改模型：只需替换 `--modality`
- `label_file`、`clinical_file`、`omics_dir`、`split_dir`、`data_root_dir` 会按 `study` 自动推断
