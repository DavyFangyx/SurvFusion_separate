#!/bin/bash
# configs/z_exp_gen/Table1_cindex_main/gen_ours/_common.bash
# 共享的 SurvTriPoEVAE A/B/C 与 B_nopretrain 配置生成逻辑。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
[_common.bash] 这是共享脚本，不会单独生成 config。

请运行下面任一入口脚本：
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_brca_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_coad_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kich_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kirc_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_kirp_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_lihc_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_prad_poe_model_val.sh
  bash configs/z_exp_gen/Table1_cindex_main/gen_ours/gen_tcga_read_poe_model_val.sh
EOF
    exit 2
fi

generate_poe_model_val_configs() {
    : "${STUDY:?STUDY is required}"
    : "${EXP_GROUP:?EXP_GROUP is required}"

    local script_dir out_dir clinic_experiment gene_experiment wsi_experiment
    local batch_size batch_size_stage1
    local max_epochs max_epochs_stage1 warmup_epochs run_name_base file_prefix exp_group_prefix
    local seq created skipped preset fname target

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
    out_dir="${OUT_DIR:-$script_dir/configs/queue}"
    clinic_experiment="${CLINIC_EXPERIMENT:-L0}"
    gene_experiment="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
    wsi_experiment="${WSI_EXPERIMENT:-uni_v1}"
    batch_size="${BATCH_SIZE:-128}"
    batch_size_stage1="${BATCH_SIZE_STAGE1:-128}"
    max_epochs="${MAX_EPOCHS:-20}"
    max_epochs_stage1="${MAX_EPOCHS_STAGE1:-10}"
    warmup_epochs="${WARMUP_EPOCHS:-3}"
    exp_group_prefix=""
    if [[ "$clinic_experiment" == "L0" ]]; then
        exp_group_prefix="L0_"
    fi

    run_name_base="${RUN_NAME_BASE:-${STUDY}__${clinic_experiment}__cell_norm__${wsi_experiment}}"
    if [[ "$EXP_GROUP" != "${exp_group_prefix}"* ]]; then
        EXP_GROUP="${exp_group_prefix}${EXP_GROUP}"
    fi
    file_prefix="${FILE_PREFIX:-${exp_group_prefix}${STUDY#tcga_}_poe_model_val}"

    local -a poe_presets=(
        survtri_poe_vae_A
        survtri_poe_vae_B
        survtri_poe_vae_B_nopretrain
        survtri_poe_vae_C
    )

    mkdir -p "$out_dir"

    create_conf() {
        local target_path="$1"
        local content="$2"
        if [ -e "$target_path" ]; then
            skipped=$((skipped + 1))
            return
        fi
        printf '%s\n' "$content" > "$target_path"
        created=$((created + 1))
    }

    seq=0
    created=0
    skipped=0

    for preset in "${poe_presets[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__PCG__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=${run_name_base}__${preset}
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BAG_LOSS=cox_surv
BATCH_SIZE=$batch_size
BATCH_SIZE_STAGE1=$batch_size_stage1
MAX_EPOCHS=$max_epochs
MAX_EPOCHS_STAGE1=$max_epochs_stage1
WARMUP_EPOCHS=$warmup_epochs
EOF
)"
    done

    echo "Generated $created new configs in $out_dir"
    echo "Total indexed configs this round: $seq"
    echo "Skipped existing configs: $skipped"
    echo "Study: $STUDY"
    echo "Experiment group: $EXP_GROUP"
    echo "Clinic embedding: $clinic_experiment"
    echo "Gene embedding: $gene_experiment"
    echo "WSI embedding: $wsi_experiment"
    echo "Run name base: $run_name_base"
}
