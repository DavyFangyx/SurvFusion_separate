#!/bin/bash
set -o pipefail

# ============================================================
# 一键运行模型对比实验
# 用法:
#   conda activate SurvPGC
#   bash run_all_models.sh [GPU=0]
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
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- GPU 指定 ----------
for arg in "$@"; do
    [[ $arg == GPU=* ]] && export CUDA_VISIBLE_DEVICES="${arg#GPU=}"
done

# ============================================================
# 【参数配置区】—— 每次实验只需修改这里
#
# 推荐的四类实验设计：
# 1) clinic_test:
#    - 改 CLINIC_EXPERIMENT
#    - 跑所有包含 C 模态的模型
#    - RUN_NAME 建议与 clinic 特征名保持一致
#
# 2) gene_test:
#    - 改 GENE_EXPERIMENT / TYPE_OF_PATH
#    - 跑所有包含 G 模态的模型
#    - RUN_NAME 建议与 gene 特征名保持一致
#
# 3) param_tuning:
#    - 固定数据与模型，只改超参数
#    - RUN_NAME 建议编码主要超参，例如 lr5e4_reg1e3
#
# 4) ablation:
#    - 固定最佳数据与超参，只改模型开关
#    - RUN_NAME 建议描述消融主题，例如 main_vs_wc
# ============================================================

# EXP_GROUP: 本次属于哪类实验
#   clinic_test | gene_test | param_tuning | ablation | default
EXP_GROUP="clinic_test"

# RUN_NAME: 同系列内的具体变体名。
# 若留空，会按实验类型自动生成。
RUN_NAME=""

# ---- 数据集配置 ----
STUDY="tcga_kirc"
WHICH_SPLITS="5foldcv"

# clinic 特征目录（实验一改这里）:
# A_profile  B_noNM     B_noTM   B_staging    D_noProg
# B_noM      B_noStage  B_noTN   C_treatment  O_origin
# B_noN      B_noT      B_noTNM  D_clinical_summary
# O_simple
CLINIC_EXPERIMENT="O_origin"

# gene 特征目录（实验二改这里）:
# scFoundation_embedding_cell_norm
# scFoundation_embedding_cell_raw
# scFoundation_embedding_gene_norm
# scFoundation_embedding_gene_raw
GENE_EXPERIMENT="scFoundation_embedding_cell_norm"

# WSI 特征目录
WSI_EXPERIMENT="uni_v1"

# ---- 训练超参 ----
TYPE_OF_PATH="combine"
K=5
MAX_EPOCHS=20
BAG_LOSS="nll_surv"
LR=0.0005
REG=0.001
SEED=1

# porpoise 专用 fusion 参数
PORPOISE_FUSION="bilinear"

# ---- 模型开关 ----
# 单模态 C
RUN_CLINIC_MLP=false
RUN_CLINIC_SNN=false
RUN_CLINIC_COX=false
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

# 主模型 WSI+G+C
RUN_SURVPGC_F=true
# 消融：WSI+C（去掉 G）
RUN_SURVPC=false
RUN_SURVPC_F=false
# 消融：WSI+C 直接拼接
RUN_MLPPC_CONCAT=false

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

require_path "$LABEL_FILE" "label_file"
require_path "$OMICS_DIR" "omics_dir"
require_path "$DATA_ROOT_DIR" "data_root_dir"
require_path "$CLINIC_DIR" "clinic_dir"
require_path "$GENE_DIR" "gene_dir"
require_path "$SPLIT_DIR" "split_dir"

TOTAL_MODELS=0
DONE_MODELS=0
count_model() { TOTAL_MODELS=$((TOTAL_MODELS + 1)); }
$RUN_OMICS        && count_model
$RUN_SNN          && count_model
$RUN_ABMIL_WSI    && count_model
$RUN_MLP_WSI      && count_model
$RUN_TRANSMIL_WSI && count_model
$RUN_CLINIC_MLP   && count_model
$RUN_CLINIC_SNN   && count_model
$RUN_CLINIC_COX   && count_model
$RUN_PORPOISE     && count_model
$RUN_SURVPATH     && count_model
$RUN_SURVPGC_F    && count_model
$RUN_SURVPC       && count_model
$RUN_SURVPC_F     && count_model
$RUN_MLPPC_CONCAT && count_model

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
$RUN_PORPOISE     && launch_model "porpoise" --fusion "$PORPOISE_FUSION"
$RUN_SURVPATH     && launch_model "survpath"

# ---------- 主模型 WSI+G+C ----------
$RUN_SURVPGC_F    && launch_model "survpgc_f"

# ---------- 消融：WSI+C ----------
$RUN_SURVPC       && launch_model "survpc"
$RUN_SURVPC_F     && launch_model "survpc_f"

# ---------- 消融：WSI+C 直接拼接 ----------
$RUN_MLPPC_CONCAT && launch_model "mlppc_concat"

echo ""
echo "========================================================"
echo "  所有模型运行完毕  ($DONE_MODELS/$TOTAL_MODELS)"
echo "  结束时间: $(date)"
echo "  结果根目录: $SCRIPT_DIR/results/${EXP_GROUP}/${RUN_NAME}/"
echo "========================================================"
