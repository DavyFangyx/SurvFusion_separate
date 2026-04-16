#!/bin/bash
set -o pipefail

# ============================================================
# 一键并行运行所有模型对比实验
# 修改下方【参数配置区】后直接运行: 
# conda activate SurvPGC
# bash run_all_models.sh GPU=
# ============================================================

# ============================================================
# 【参数配置区】—— 每次实验只需修改这里
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"  # 确保相对路径（splits/、main.py）始终可找到

# 命令行指定 GPU
# 不指定则使用系统默认（所有卡）
for arg in "$@"; do
    [[ $arg == GPU=* ]] && export CUDA_VISIBLE_DEVICES="${arg#GPU=}"
done

# TEST_SET 代表本次测试系列（如 atcga_kirc / tcga_kirc），日志将汇集到 results/summary_${TEST_SET}_logs/
TEST_SET="stcga_kirc"
# RUN_NAME 代表同系列内的具体实验变体（如不同 clinic_dir），只影响日志文件名
RUN_NAME="O_origin"
CLINIC_DIR="$SCRIPT_DIR/SurvPGC_Workspace/C/O_origin/npy"
# A_profile  B_noNM     B_noTM   B_staging    D_noProg
# B_noM      B_noStage  B_noTN   C_treatment  O_origin
# B_noN      B_noT      B_noTNM  D_clinical_summary
# 【模型开关】—— 将不需要运行的模型设为 false
# 单模态 G（只用基因组）
RUN_OMICS=false
RUN_SNN=false
# 单模态 WSI（只用病理图像）
RUN_ABMIL_WSI=false
RUN_MLP_WSI=false
RUN_TRANSMIL_WSI=false
# 单模态 C
RUN_CLINIC_MLP=false
RUN_CLINIC_SNN=false
# 多模态（WSI + G）
RUN_PORPOISE=false
RUN_SURVPATH=false
# 多模态（WSI + G + C）
RUN_SURVPGC_F=false
RUN_SURVFUSION_F=false
# SURVFUSION_F消融
RUN_SURVFUSION_MLPCONTACT=false    # 消融（WSI + G + C 拼接后MLP）
RUN_SURVFUSION_MHSACONCAT=false    # 消融（WSI + G + C 拼接后多头自注意力）
RUN_SURVFUSION_NOALIGN=false       # 消融（WSI + G + C 无对齐损失）
RUN_SURVFUSION_SEPARATE=true     # 两阶段分离训练（Stage1: 对齐损失, Stage2: 生存损失）
# 消融（WSI + C（去掉G））
RUN_SURVPC=false
RUN_SURVPC_F=false
# 消融（WSI + C 直接拼接（无注意力））
RUN_MLPPC_CONCAT=false
# 消融（G + C（去掉WSI））
RUN_SURVGC_F=false


# STUDY 必须与 splits/5foldcv/ 下的目录名一致，代码内部会用它构建路径
STUDY="tcga_kirc"
LABEL_FILE="$SCRIPT_DIR/datasets_csv/metadata/tcga_kirc.csv"
OMICS_DIR="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/kirc"
DATA_ROOT_DIR="$SCRIPT_DIR/SurvPGC_Workspace/P/uni_v1"
GENE_DIR="$SCRIPT_DIR/SurvPGC_Workspace/G/max_seq_len_2048/npy"
SPLIT_DIR="$SCRIPT_DIR/splits/5foldcv/tcga_kirc"


TYPE_OF_PATH="combine"
K=1
MAX_EPOCHS=20
MAX_EPOCHS_STAGE1=40
BATCH_SIZE_STAGE1=32
BAG_LOSS="nll_surv"
LR=0.0005
LR_STAGE1=0.0001
REG=0.001
SEED=1

# porpoise 专用 fusion 参数（concat 或 bilinear）
PORPOISE_FUSION="bilinear"


# ============================================================
# 以下无需修改
# ============================================================

RESULTS_DIR="./results"
TMP_DIR="/tmp/survpgc_${TEST_SET}_${RUN_NAME}"
SET_LOG_DIR="./results/summary_${TEST_SET}_logs"
SUMMARY_LOG="${SET_LOG_DIR}/summary_${TEST_SET}_${RUN_NAME}.log"
LOSS_CURVE_DIR="${SET_LOG_DIR}/loss_curves_${RUN_NAME}"
EXPORT_LOSS_CURVES=true
RESULTS_ROOT_DISPLAY="$SCRIPT_DIR/results/results"

mkdir -p "$TMP_DIR" "$RESULTS_DIR" "$SET_LOG_DIR" "$LOSS_CURVE_DIR"

echo "TEST_SET: $TEST_SET  |  RUN_NAME: $RUN_NAME" > "$SUMMARY_LOG"
echo "Start time: $(date)" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

TOTAL_MODELS=0
DONE_MODELS=0
count_model() { TOTAL_MODELS=$((TOTAL_MODELS + 1)); }
$RUN_OMICS        && count_model
$RUN_SNN          && count_model
$RUN_ABMIL_WSI    && count_model
$RUN_MLP_WSI      && count_model
$RUN_TRANSMIL_WSI && count_model
$RUN_PORPOISE     && count_model
$RUN_SURVPATH     && count_model
$RUN_SURVPGC_F    && count_model
$RUN_SURVPC       && count_model
$RUN_SURVPC_F     && count_model
$RUN_MLPPC_CONCAT && count_model
$RUN_SURVGC_F     && count_model
$RUN_CLINIC_MLP   && count_model
$RUN_CLINIC_SNN   && count_model

launch_model() {
    local modality=$1
    local extra_args=$2
    local tmp_out="$TMP_DIR/${modality}.log"
    local exit_code

    echo "[START] $modality"

    python -u main.py \
      --modality "$modality" \
      --study "$STUDY" \
            --test_set "$TEST_SET" \
            --run_name "$RUN_NAME" \
      --label_file "$LABEL_FILE" \
      --omics_dir "$OMICS_DIR" \
      --data_root_dir "$DATA_ROOT_DIR" \
      --clinic_dir "$CLINIC_DIR" \
      --gene_dir "$GENE_DIR" \
      --split_dir "$SPLIT_DIR" \
      --type_of_path "$TYPE_OF_PATH" \
      --k "$K" \
      --max_epochs "$MAX_EPOCHS" \
      --bag_loss "$BAG_LOSS" \
      --lr "$LR" \
      --reg "$REG" \
      --seed "$SEED" \
      --results_dir "$RESULTS_DIR" \
      $extra_args \
      2>&1 | tee "$tmp_out"
        exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -ne 0 ]; then
                printf "%s:\n[运行失败，exit code: %s]\n日志: %s\n\n" "$modality" "$exit_code" "$tmp_out" >> "$SUMMARY_LOG"
                echo "[FAIL] $modality  | exit code: $exit_code | log: $tmp_out"
                return "$exit_code"
        fi

    cindex_line=$(grep -i "Final test c-index" "$tmp_out" | tail -1)
    result="${cindex_line:-[未找到 c-index，exit code: $?]}"

    if $EXPORT_LOSS_CURVES; then
        curve_csv="${LOSS_CURVE_DIR}/${modality}_${RUN_NAME}_loss_curve.csv"
        python - "$tmp_out" "$curve_csv" <<'PY'
import csv
import re
import sys

log_path, out_csv = sys.argv[1], sys.argv[2]
rows = []
current_stage = "default"
epoch_stage_counter = {"stage1": 0, "stage2": 0, "default": 0}
pending = {}

re_stage1 = re.compile(r"Stage\s*1")
re_stage2 = re.compile(r"Stage\s*2")
re_train = re.compile(r"Epoch:\s*(\d+),\s*train_loss:\s*([0-9.]+)(?:,\s*train_c_index:\s*([0-9.]+))?")
re_val_align = re.compile(r"Val\s+Align\s+Loss:\s*([0-9.]+)")
re_val_stage2 = re.compile(r"Val\s+Loss:\s*([0-9.]+),\s*Val\s+C-Index:\s*([0-9.]+)")

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if re_stage1.search(line):
            current_stage = "stage1"
            continue
        if re_stage2.search(line):
            current_stage = "stage2"
            continue

        m = re_train.search(line)
        if m:
            printed_epoch = int(m.group(1))
            train_loss = float(m.group(2))
            train_c_index = m.group(3)
            train_c_index = float(train_c_index) if train_c_index is not None else ""
            epoch_stage_counter[current_stage] += 1
            stage_epoch = epoch_stage_counter[current_stage] - 1
            rec = {
                "stage": current_stage,
                "epoch": stage_epoch,
                "printed_epoch": printed_epoch,
                "train_loss": train_loss,
                "val_loss": "",
                "train_c_index": train_c_index,
                "val_c_index": "",
            }
            rows.append(rec)
            pending[current_stage] = rec
            continue

        m = re_val_align.search(line)
        if m and "stage1" in pending:
            pending["stage1"]["val_loss"] = float(m.group(1))
            continue

        m = re_val_stage2.search(line)
        if m and "stage2" in pending:
            pending["stage2"]["val_loss"] = float(m.group(1))
            pending["stage2"]["val_c_index"] = float(m.group(2))
            continue

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["stage", "epoch", "printed_epoch", "train_loss", "val_loss", "train_c_index", "val_c_index"],
    )
    writer.writeheader()
    writer.writerows(rows)
PY
    fi

    DONE_MODELS=$((DONE_MODELS + 1))
    printf "%s:\n%s\n\n" "$modality" "$result" >> "$SUMMARY_LOG"
    echo "[DONE] $modality  ($DONE_MODELS/$TOTAL_MODELS)  |  $result"
}

echo "========================================================"
echo "  实验配置: STUDY=$STUDY  TEST_SET=$TEST_SET  RUN_NAME=$RUN_NAME"
[ -n "$CUDA_VISIBLE_DEVICES" ] && echo "  GPU: $CUDA_VISIBLE_DEVICES" || echo "  GPU: 系统默认"
echo "  开始时间: $(date)"
echo "  汇总日志: $SUMMARY_LOG"
echo "  模型结果目录根: $RESULTS_ROOT_DISPLAY"
echo "========================================================"
echo ""

# ---------- 单模态 G ----------
$RUN_OMICS        && launch_model "omics"
$RUN_SNN          && launch_model "snn"

# ---------- 单模态 WSI ----------
$RUN_ABMIL_WSI    && launch_model "abmil_wsi"
$RUN_MLP_WSI      && launch_model "mlp_wsi"
$RUN_TRANSMIL_WSI && launch_model "transmil_wsi"

# ---------- 多模态基线 WSI + G ----------
$RUN_PORPOISE     && launch_model "porpoise"     "--fusion $PORPOISE_FUSION"
$RUN_SURVPATH     && launch_model "survpath"

# ---------- 主模型 WSI + G + C ----------
$RUN_SURVPGC_F    && launch_model "survpgc_f"
$RUN_SURVFUSION_F          && launch_model "survfusion_f"

# ---------- 消融：融合策略 & 对齐损失 ----------
$RUN_SURVFUSION_MLPCONTACT && launch_model "survfusion_mlpcontact"
$RUN_SURVFUSION_MHSACONCAT && launch_model "survfusion_mhsaconcat"
$RUN_SURVFUSION_NOALIGN    && launch_model "survfusion_noalign"
$RUN_SURVFUSION_SEPARATE   && launch_model "survfusion_separate" "--lr_stage1 $LR_STAGE1 --max_epochs_stage1 $MAX_EPOCHS_STAGE1 --batch_size_stage1 $BATCH_SIZE_STAGE1"

# ---------- 消融：WSI + C ----------
$RUN_SURVPC       && launch_model "survpc"
$RUN_SURVPC_F     && launch_model "survpc_f"

# ---------- 消融：WSI + C 拼接 ----------
$RUN_MLPPC_CONCAT && launch_model "mlppc_concat"

# ---------- 消融：G + C ----------
$RUN_SURVGC_F     && launch_model "survgc_f"

# ---------- 消融：只用 C ----------
$RUN_CLINIC_MLP   && launch_model "clinic_mlp"
$RUN_CLINIC_SNN   && launch_model "clinic_snn"

echo ""
echo "========================================================"
echo "  所有模型运行完毕"
echo "  结束时间: $(date)" | tee -a "$SUMMARY_LOG"
echo "  汇总日志: $SUMMARY_LOG"
echo "  Loss 曲线目录: $LOSS_CURVE_DIR"
echo "========================================================"
echo ""
echo "===== 汇总结果 ====="
cat "$SUMMARY_LOG"
