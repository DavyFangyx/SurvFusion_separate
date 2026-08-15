#!/bin/bash
# 生成 FigB_Clinical Prompt Test 的 Clinic 单模态配置与直跑命令文档。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
DOC_DIR="${DOC_DIR:-$SCRIPT_DIR/z_temp/FigB}"
DOC_PATH="$DOC_DIR/FigB_Clinical_Prompt_Test.md"
mkdir -p "$OUT_DIR" "$DOC_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

CLINIC_EXPERIMENTS=(
    L0
    L1
    L2
    L3
    L4
    L5
    D0
    D1
    D2
    D3
    D4
    D5
)

MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
)

create_conf() {
    local target="$1"
    local content="$2"
    if [ -e "$target" ]; then
        skipped=$((skipped + 1))
        return
    fi
    printf '%s\n' "$content" > "$target"
    created=$((created + 1))
}

seq=0
created=0
skipped=0

for study in "${STUDIES[@]}"; do
    for clinic in "${CLINIC_EXPERIMENTS[@]}"; do
        for preset in "${MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "figb_clinic_prompt__%03d__%s__%s__%s.conf" "$seq" "$study" "$clinic" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP='FigB_Clinical Prompt Test'
RUN_NAME=${study}__${clinic}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$clinic
WSI_EXPERIMENT=uni_v1
GENE_EXPERIMENT=scFoundation_embedding_gene_raw
WHICH_SPLITS=5foldcv
K=5
WANDB_MODE=disabled
EOF
)"
        done
    done
done

cat > "$DOC_PATH" <<'EOF'
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
  --clinic_dir ./SurvPGC_Workspace/tcga_read/C/D0 \
  --modality mlp_clinic_mean \
  --max_epochs 12 \
  --wandb_mode disabled
```

## 替换规则

- 改数据集：同步替换 `--study`、`--run_name`、`--clinic_dir`
- 改 Clinic 编码：同步替换 `--run_name` 和 `--clinic_dir` 里的 `Lx/Dx`
- 改模型：只需替换 `--modality`
- `label_file`、`clinical_file`、`omics_dir`、`split_dir`、`data_root_dir` 会按 `study` 自动推断
EOF

echo "Enabled studies: ${#STUDIES[@]}"
printf 'Studies:'
for study in "${STUDIES[@]}"; do
    printf ' %s' "$study"
done
printf '\n'

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Markdown command doc: $DOC_PATH"
echo "Expected result folders:"
echo "  results/FigB_Clinical Prompt Test/<study>__<Lx_or_Dx>/<clinic_model>/"
