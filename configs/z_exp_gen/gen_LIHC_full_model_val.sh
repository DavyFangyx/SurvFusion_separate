#!/bin/bash
# configs/z_exp_gen/gen_LIHC_full_model_val.sh
# 生成 tcga_lihc 指定特征组合下的全模型验证配置。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

EXP_GROUP="${EXP_GROUP:-LIHC_full_model_val}"
STUDY="tcga_lihc"
CLINIC_EXPERIMENT="L4"
GENE_EXPERIMENT="scFoundation_embedding_cell_norm"
WSI_EXPERIMENT="uni_v1"
BATCH_SIZE="${BATCH_SIZE:-1}"
RUN_NAME_WSI="${RUN_NAME_WSI:-${STUDY}__${WSI_EXPERIMENT}}"
RUN_NAME_CLINIC="${RUN_NAME_CLINIC:-${STUDY}__${CLINIC_EXPERIMENT}}"
RUN_NAME_GENE_RAW="${RUN_NAME_GENE_RAW:-${STUDY}__csvraw}"
RUN_NAME_GENE_F="${RUN_NAME_GENE_F:-${STUDY}__${GENE_EXPERIMENT}}"
RUN_NAME_MULTI="${RUN_NAME_MULTI:-${STUDY}__L4__cell_norm__uni_v1}"

WSI_MODELS=(
    abmil_wsi
    mlp_wsi
    transmil_wsi
)

CLINIC_MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
    clinic_cox
)

GENE_MODELS=(
    mlp_gene
    snn_gene
    mlp_gene_f
    snn_gene_f
)

MULTI_MODELS=(
    survpc_f
    porpoise
    survpath
    mcat
    survgc_f
    survpgc_f
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

for preset in "${WSI_MODELS[@]}"; do
    seq=$((seq + 1))
    fname=$(printf "lihc_full_model_val__%03d__P__%s.conf" "$seq" "$preset")
    target="$OUT_DIR/$fname"
    create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$RUN_NAME_WSI
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
BATCH_SIZE=$BATCH_SIZE
EOF
)"
done

for preset in "${CLINIC_MODELS[@]}"; do
    seq=$((seq + 1))
    fname=$(printf "lihc_full_model_val__%03d__C__%s.conf" "$seq" "$preset")
    target="$OUT_DIR/$fname"
    create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$RUN_NAME_CLINIC
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
BATCH_SIZE=$BATCH_SIZE
EOF
)"
done

for preset in "${GENE_MODELS[@]}"; do
    seq=$((seq + 1))
    fname=$(printf "lihc_full_model_val__%03d__G__%s.conf" "$seq" "$preset")
    target="$OUT_DIR/$fname"
    run_name="$RUN_NAME_GENE_F"
    case "$preset" in
        mlp_gene|snn_gene)
            run_name="$RUN_NAME_GENE_RAW"
            ;;
    esac
    create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
BATCH_SIZE=$BATCH_SIZE
EOF
)"
done

for preset in "${MULTI_MODELS[@]}"; do
    seq=$((seq + 1))
    modality_tag="PCG"
    case "$preset" in
        survpc_f)
            modality_tag="PC"
            ;;
        porpoise|survpath|mcat)
            modality_tag="PG"
            ;;
        survgc_f)
            modality_tag="CG"
            ;;
    esac

    fname=$(printf "lihc_full_model_val__%03d__%s__%s.conf" "$seq" "$modality_tag" "$preset")
    target="$OUT_DIR/$fname"
    create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$RUN_NAME_MULTI
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$CLINIC_EXPERIMENT
GENE_EXPERIMENT=$GENE_EXPERIMENT
WSI_EXPERIMENT=$WSI_EXPERIMENT
BATCH_SIZE=$BATCH_SIZE
EOF
)"
done

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Study: $STUDY"
echo "Clinic embedding: $CLINIC_EXPERIMENT"
echo "Gene embedding: $GENE_EXPERIMENT"
echo "WSI embedding: $WSI_EXPERIMENT"
echo "Run name (WSI): $RUN_NAME_WSI"
echo "Run name (Clinic): $RUN_NAME_CLINIC"
echo "Run name (Gene raw): $RUN_NAME_GENE_RAW"
echo "Run name (Gene FM): $RUN_NAME_GENE_F"
echo "Run name (Multi): $RUN_NAME_MULTI"
echo "Experiment group: $EXP_GROUP"
