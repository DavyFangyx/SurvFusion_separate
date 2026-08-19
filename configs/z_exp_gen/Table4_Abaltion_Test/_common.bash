#!/bin/bash
# configs/z_exp_gen/Table4_Abaltion_Test/_common.bash
# 共享的 Table4_Abaltion_Test 配置生成逻辑。

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    cat <<'EOF'
[_common.bash] 这是共享脚本，不会单独生成 config。

请运行下面任一入口脚本：
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_brca_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_coad_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kich_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kirc_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_kirp_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_lihc_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_prad_ablation_test.sh
  bash configs/z_exp_gen/Table4_Abaltion_Test/gen_tcga_read_ablation_test.sh
EOF
    exit 2
fi

generate_table4_ablation_test_configs() {
    : "${STUDY:?STUDY is required}"
    : "${EXP_GROUP:?EXP_GROUP is required}"

    local script_dir out_dir clinic_experiment gene_experiment wsi_experiment
    local batch_size max_epochs warmup_epochs run_name_base file_prefix
    local seq created skipped preset fname target gene_tag

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
    out_dir="${OUT_DIR:-$script_dir/configs/queue}"
    clinic_experiment="${CLINIC_EXPERIMENT:-L0}"
    gene_experiment="${GENE_EXPERIMENT:-scFoundation_embedding_cell_norm}"
    wsi_experiment="${WSI_EXPERIMENT:-uni_v1}"
    batch_size="${BATCH_SIZE:-128}"
    max_epochs="${MAX_EPOCHS:-20}"
    warmup_epochs="${WARMUP_EPOCHS:-3}"

    case "$gene_experiment" in
        scFoundation_embedding_cell_norm) gene_tag="cell_norm" ;;
        scFoundation_embedding_cell_raw) gene_tag="cell_raw" ;;
        scFoundation_embedding_gene_norm) gene_tag="gene_norm" ;;
        scFoundation_embedding_gene_raw) gene_tag="gene_raw" ;;
        *) gene_tag="$gene_experiment" ;;
    esac

    run_name_base="${RUN_NAME_BASE:-${STUDY}__${gene_tag}__${wsi_experiment}}"
    file_prefix="${FILE_PREFIX:-Table4_Abaltion_Test_${STUDY#tcga_}_ablation_test}"

    local -a ablation_presets=(
        survtri_poe_vae_B_nopretrain
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

    for preset in "${ablation_presets[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "%s__%03d__%s.conf" "$file_prefix" "$seq" "$preset")
        target="$out_dir/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=$EXP_GROUP
RUN_NAME=$run_name_base
PRESET=$preset
STUDY=$STUDY
CLINIC_EXPERIMENT=$clinic_experiment
GENE_EXPERIMENT=$gene_experiment
WSI_EXPERIMENT=$wsi_experiment
BAG_LOSS=cox_surv
BATCH_SIZE=$batch_size
MAX_EPOCHS=$max_epochs
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
