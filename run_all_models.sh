#!/bin/bash
set -o pipefail

# ============================================================
# 一键运行模型对比实验
# 用法:
#   conda activate SurvPGC
#   bash run_all_models.sh GPU=
#
# 结果目录结构:
#   results/{EXP_GROUP}/{RUN_NAME}/{modality}/
#     test_result.csv     split_*_results.pkl     experiment.txt
#
# EXP_GROUP 对应 4 类实验:
#   clinic_test    — 不同 clinic 数据对照
#   gene_test      — 不同 gene 数据对照
#   param_tuning   — 单模型超参调整
#   ablation       — 消融实验
#   实验一 ablation_Model_structure  — noalign / joint / separate_mhsa
#   实验二 ablation_Fusion_method    — mhsa / concat / mean_concat
#   实验三 ablation_CLIP_weights     — 只开 MHSA，脚本自动遍历 {1,2,3}³
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- GPU 指定 ----------
for arg in "$@"; do
    [[ $arg == GPU=* ]] && export CUDA_VISIBLE_DEVICES="${arg#GPU=}"
done

# ---- 数据集配置 ----
STUDY="tcga_kirc"
WHICH_SPLITS="5foldcv"

# ============================================================
# 【参数配置区】—— 每次实验只需修改这里
# ============================================================

# EXP_GROUP: clinic_test | gene_test | param_tuning | default
# ablation_Model_structure ablation_Fusion_method  ablation_CLIP_weights

EXP_GROUP="ablation_Model_structure"

# RUN_NAME: 若留空，会按实验类型自动生成
RUN_NAME="CLIP_LAMBDA005"

# ---- 数据集配置 ----
STUDY="tcga_kirc"
WHICH_SPLITS="5foldcv"
# joint 联合训练对齐损失权重 λ
CLIP_LAMBDA=0.01
# mhsa 注意力头数
NUM_HEADS=8
# CLIP 三对权重；RUN_CLIP_WEIGHTS_SWEEP=true 时此处无效，自动遍历 {1,2,3}³
CLIP_WEIGHT_IT=1.0
CLIP_WEIGHT_IS=1.0
CLIP_WEIGHT_TS=1.0
RUN_CLIP_WEIGHTS_SWEEP=false
# WSI 特征目录
WSI_EXPERIMENT="uni_v1"

# ---- 训练超参 ----
TYPE_OF_PATH="combine"
K=5
MAX_EPOCHS=12
BAG_LOSS="nll_surv"
LR=0.0005
REG=0.001
SEED=1
# fusion 参数（PORPOISE 仅支持 bilinear，统一使用）
FUSION="bilinear"

# ---- Stage-1/2 超参（仅survfusion_* ）----
LR_STAGE1=0.0001
MAX_EPOCHS_STAGE1=30
BATCH_SIZE_STAGE1=32
BATCH_SIZE_STAGE2=32    # survfusion_separate Stage2 及普通模型的 train batch_size

# clinic 特征目录（实验一改这里）:
# A_profile  B_noNM     B_noTM   B_staging    D_noProg
# B_noM      B_noStage  B_noTN   C_treatment  O_origin
# B_noN      B_noT      B_noTNM  D_clinical_summary
# O_simple
CLINIC_EXPERIMENT="O_simple"
# gene 特征目录（实验二改这里）:
# scFoundation_embedding_cell_norm
# scFoundation_embedding_cell_raw
# scFoundation_embedding_gene_norm
# scFoundation_embedding_gene_raw
GENE_EXPERIMENT="scFoundation_embedding_gene_raw"
# WSI 特征目录
WSI_EXPERIMENT="uni_v1"

# ---- 训练超参 ----
TYPE_OF_PATH="combine"
K=5
MAX_EPOCHS=12
BAG_LOSS="nll_surv"
LR=0.0005
REG=0.001
SEED=1
# fusion 参数（PORPOISE 仅支持 bilinear，统一使用）
FUSION="bilinear"

# ---- Stage-1/2 超参（仅survfusion_* ）----
LR_STAGE1=0.0001
MAX_EPOCHS_STAGE1=30
BATCH_SIZE_STAGE1=32
BATCH_SIZE_STAGE2=1    # survfusion_separate Stage2 及普通模型的 train batch_size

# ---- 模型开关 ----
# 单模态 C
RUN_CLINIC_COX=false
RUN_CLINIC_MLP=false
RUN_CLINIC_SNN=false
# 单模态 G
RUN_OMICS=false
RUN_SNN=false
# 单模态 WSI
RUN_MLP_WSI=false
RUN_ABMIL_WSI=false
RUN_TRANSMIL_WSI=false

# 多模态基线 WSI+G
RUN_PORPOISE=false
RUN_SURVPATH=false
RUN_MCAT=false
# 主模型 WSI+G+C
RUN_SURVPGC_F=false
# 消融：WSI+C（去掉 G）
RUN_SURVPC_F=false
# 消融：G+C（去掉 WSI）
RUN_SURVGC_F=false

# 实验一：模型结构  实验三：CLIP权重（27组合自动循环）
RUN_SURVFUSION_NOALIGN=false
RUN_SURVFUSION_JOINT=true
RUN_SURVFUSION_SEPARATE_MHSA=true
# 实验二：融合方式
RUN_SURVFUSION_SEPARATE_CONCAT=false
RUN_SURVFUSION_SEPARATE_MEAN=false

# ============================================================
# 以下无需修改
# ============================================================

RESULTS_BASE="./results"
STUDY_SUBTYPE="${STUDY#tcga_}"

LABEL_FILE="$SCRIPT_DIR/datasets_csv/metadata/${STUDY}.csv"
OMICS_DIR="$SCRIPT_DIR/datasets_csv/raw_rna_data/${TYPE_OF_PATH}/${STUDY_SUBTYPE}"
DATA_ROOT_DIR="$SCRIPT_DIR/SurvPGC_Workspace/P/${WSI_EXPERIMENT}"
CLINIC_DIR="$SCRIPT_DIR/SurvPGC_Workspace/C/${CLINIC_EXPERIMENT}"
GENE_DIR="$SCRIPT_DIR/SurvPGC_Workspace/G/${GENE_EXPERIMENT}"
SPLIT_DIR="$SCRIPT_DIR/splits/${WHICH_SPLITS}/${STUDY}"

if [ -z "$RUN_NAME" ]; then
    case "$EXP_GROUP" in
        clinic_test)
            RUN_NAME="$CLINIC_EXPERIMENT"
            ;;
        gene_test)
            RUN_NAME="${GENE_EXPERIMENT}_${TYPE_OF_PATH}"
            ;;
        param_tuning)
            RUN_NAME="lr${LR}_reg${REG}_ep${MAX_EPOCHS}_seed${SEED}"
            ;;
        ablation)
            RUN_NAME="model_ablation"
            ;;
        *)
            RUN_NAME="default"
            ;;
    esac
fi

require_path() {
    local target_path=$1
    local description=$2

    if [ ! -e "$target_path" ]; then
        echo "[ERROR] ${description} 不存在: $target_path"
        exit 1
    fi
}

infer_wsi_encoding_dim() {
    local data_root_dir="$1"
    python -c '
import glob
import os
import sys
import torch

data_root_dir = sys.argv[1]
pt_files = sorted(glob.glob(os.path.join(data_root_dir, "*.pt")))
if not pt_files:
    raise FileNotFoundError(f"No .pt files found under {data_root_dir}")

sample = torch.load(pt_files[0], map_location="cpu")
if sample.ndim < 2:
    raise ValueError(f"Expected WSI embedding tensor with ndim >= 2, got shape {tuple(sample.shape)} from {pt_files[0]}")

print(int(sample.shape[-1]))
' "$data_root_dir"
}

require_path "$LABEL_FILE" "label_file"
require_path "$OMICS_DIR" "omics_dir"
require_path "$DATA_ROOT_DIR" "data_root_dir"
require_path "$CLINIC_DIR" "clinic_dir"
require_path "$GENE_DIR" "gene_dir"
require_path "$SPLIT_DIR" "split_dir"

ENCODING_DIM="$(infer_wsi_encoding_dim "$DATA_ROOT_DIR")"

TOTAL_MODELS=0
DONE_MODELS=0
count_model() { TOTAL_MODELS=$((TOTAL_MODELS + 1)); }
$RUN_OMICS              && count_model
$RUN_SNN                && count_model
$RUN_ABMIL_WSI          && count_model
$RUN_MLP_WSI            && count_model
$RUN_TRANSMIL_WSI       && count_model
$RUN_CLINIC_MLP         && count_model
$RUN_CLINIC_SNN         && count_model
$RUN_CLINIC_COX         && count_model
$RUN_PORPOISE           && count_model
$RUN_SURVPATH           && count_model
$RUN_MCAT               && count_model
$RUN_SURVPGC_F                    && count_model
if $RUN_SURVFUSION_SEPARATE_MHSA; then
    if $RUN_CLIP_WEIGHTS_SWEEP; then
        for _i in $(seq 27); do count_model; done
    else
        count_model
    fi
fi
$RUN_SURVFUSION_SEPARATE_CONCAT  && count_model
$RUN_SURVFUSION_SEPARATE_MEAN    && count_model
$RUN_SURVFUSION_NOALIGN          && count_model
$RUN_SURVFUSION_JOINT            && count_model
$RUN_SURVPC_F                    && count_model
$RUN_SURVGC_F                    && count_model

if [ "$TOTAL_MODELS" -eq 0 ]; then
    echo "[ERROR] 没有启用任何模型，请把至少一个 RUN_* 开关设为 true"
    exit 1
fi

launch_model() {
    local modality=$1
    shift
    local exit_code
    local cmd=(
        python -u main.py
        --modality "$modality"
        --study "$STUDY"
        --exp_group "$EXP_GROUP"
        --run_name "$RUN_NAME"
        --results_dir "$RESULTS_BASE"
        --label_file "$LABEL_FILE"
        --omics_dir "$OMICS_DIR"
        --data_root_dir "$DATA_ROOT_DIR"
        --clinic_dir "$CLINIC_DIR"
        --gene_dir "$GENE_DIR"
        --split_dir "$SPLIT_DIR"
        --which_splits "$WHICH_SPLITS"
        --type_of_path "$TYPE_OF_PATH"
        --k "$K"
        --max_epochs "$MAX_EPOCHS"
        --bag_loss "$BAG_LOSS"
        --lr "$LR"
        --reg "$REG"
        --seed "$SEED"
        --batch_size "$BATCH_SIZE_STAGE2"
        --encoding_dim "$ENCODING_DIM"
    )

    echo "[START] $modality"
    echo "        results/${EXP_GROUP}/${RUN_NAME}/${modality}/"

    if [ "$#" -gt 0 ]; then
        cmd+=("$@")
    fi

    "${cmd[@]}"
    exit_code=$?

    if [ "$exit_code" -ne 0 ]; then
        echo "[FAIL] $modality  | exit code: $exit_code"
        return "$exit_code"
    fi

    DONE_MODELS=$((DONE_MODELS + 1))
    echo "[DONE] $modality  ($DONE_MODELS/$TOTAL_MODELS)"
}

echo "========================================================"
echo "  实验配置: STUDY=$STUDY  EXP_GROUP=$EXP_GROUP  RUN_NAME=$RUN_NAME"
[ -n "$CUDA_VISIBLE_DEVICES" ] && echo "  GPU: $CUDA_VISIBLE_DEVICES" || echo "  GPU: 系统默认"
echo "  CLINIC_EXPERIMENT: $CLINIC_EXPERIMENT"
echo "  GENE_EXPERIMENT:   $GENE_EXPERIMENT"
echo "  WSI_EXPERIMENT:    $WSI_EXPERIMENT"
echo "  WSI_ENCODING_DIM:  $ENCODING_DIM"
echo "  TYPE_OF_PATH:      $TYPE_OF_PATH"
echo "  WHICH_SPLITS:      $WHICH_SPLITS"
echo "  开始时间: $(date)"
echo "  结果根目录: $SCRIPT_DIR/results/${EXP_GROUP}/${RUN_NAME}/"
echo "========================================================"
echo ""

# ---------- 单模态 G ----------
$RUN_OMICS        && launch_model "omics"
$RUN_SNN          && launch_model "snn"

# ---------- 单模态 WSI ----------
$RUN_ABMIL_WSI    && launch_model "abmil_wsi"
$RUN_MLP_WSI      && launch_model "mlp_wsi"
$RUN_TRANSMIL_WSI && launch_model "transmil_wsi"

# ---------- 单模态 C ----------
$RUN_CLINIC_MLP   && launch_model "clinic_mlp"
$RUN_CLINIC_SNN   && launch_model "clinic_snn"
$RUN_CLINIC_COX   && launch_model "clinic_cox" --bag_loss "cox_surv"

# ---------- 多模态基线 WSI+G ----------
$RUN_PORPOISE     && launch_model "porpoise" --fusion "$FUSION"
$RUN_SURVPATH     && launch_model "survpath"
$RUN_MCAT         && launch_model "mcat" --fusion "$FUSION"

# ---------- 主模型 WSI+G+C ----------
$RUN_SURVPGC_F            && launch_model "survpgc_f"

_sep_base_args=(
    --lr_stage1 "$LR_STAGE1"
    --max_epochs_stage1 "$MAX_EPOCHS_STAGE1"
    --batch_size_stage1 "$BATCH_SIZE_STAGE1"
)

$RUN_SURVFUSION_NOALIGN   && launch_model "survfusion_noalign" \
    --batch_size 32 \
    --num_heads "$NUM_HEADS"

$RUN_SURVFUSION_JOINT     && launch_model "survfusion_joint" \
    --batch_size 32 \
    --num_heads "$NUM_HEADS" \
    --clip_lambda "$CLIP_LAMBDA"

if $RUN_SURVFUSION_SEPARATE_MHSA; then
    if $RUN_CLIP_WEIGHTS_SWEEP; then
        for w_it in 1 2 3; do
            for w_is in 1 2 3; do
                for w_ts in 1 2 3; do
                    launch_model "survfusion_separate" \
                        "${_sep_base_args[@]}" \
                        --fusion_type "mhsa" \
                        --num_heads "$NUM_HEADS" \
                        --clip_weight_IT "$w_it" \
                        --clip_weight_IS "$w_is" \
                        --clip_weight_TS "$w_ts"
                done
            done
        done
    else
        launch_model "survfusion_separate" \
            "${_sep_base_args[@]}" \
            --fusion_type "mhsa" \
            --num_heads "$NUM_HEADS" \
            --clip_weight_IT "$CLIP_WEIGHT_IT" \
            --clip_weight_IS "$CLIP_WEIGHT_IS" \
            --clip_weight_TS "$CLIP_WEIGHT_TS"
    fi
fi

$RUN_SURVFUSION_SEPARATE_CONCAT && launch_model "survfusion_separate" \
    "${_sep_base_args[@]}" --fusion_type "concat"

$RUN_SURVFUSION_SEPARATE_MEAN   && launch_model "survfusion_separate" \
    "${_sep_base_args[@]}" --fusion_type "mean_concat"

# ---------- 消融：WSI+C ----------
$RUN_SURVPC_F     && launch_model "survpc_f"

# ---------- 消融：G+C ----------
$RUN_SURVGC_F     && launch_model "survgc_f"

echo ""
echo "========================================================"
echo "  所有模型运行完毕  ($DONE_MODELS/$TOTAL_MODELS)"
echo "  结束时间: $(date)"
echo "  结果根目录: $SCRIPT_DIR/results/${EXP_GROUP}/${RUN_NAME}/"
echo "========================================================"
