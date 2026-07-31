#!/bin/bash
# configs/z_exp_gen/gen_Single_modal_full_test.sh
# 一次性生成单模态测试配置。
# 先对所有注册数据集做 readiness 验证，再仅为当前可运行组合生成 config。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/configs/queue}"
mkdir -p "$OUT_DIR"

INCLUDE_CLINIC_COX="${INCLUDE_CLINIC_COX:-true}"
GENERATE_ONLY_READY="${GENERATE_ONLY_READY:-true}"
VALIDATION_DIR="${VALIDATION_DIR:-$SCRIPT_DIR/configs/z_exp_gen/_validation}"
REPORT_PATH="${REPORT_PATH:-$VALIDATION_DIR/single_modal_full_test_readiness.csv}"
mkdir -p "$VALIDATION_DIR"

mapfile -t STUDIES < <(PYTHONPATH="$SCRIPT_DIR" python -c 'from dataset_deployment.registry import list_enabled_studies; print("\n".join(list_enabled_studies()))')

WSI_EXPERIMENTS=(
    uni_v1
)

WSI_MODELS=(
    abmil_wsi
    mlp_wsi
    transmil_wsi
)

GENE_F_EXPERIMENTS=(
    scFoundation_embedding_cell_norm
    scFoundation_embedding_cell_raw
    scFoundation_embedding_gene_norm
    scFoundation_embedding_gene_raw
)

GENE_F_MODELS=(
    mlp_gene_f
    snn_gene_f
)

GENE_CSVRAW_MODELS=(
    mlp_gene
    snn_gene
)

CLINIC_EXPERIMENTS=(
    L0
    L1
    L2
    L4
    L3
    L5
)

CLINIC_MODELS=(
    mlp_clinic_mean
    mlp_clinic_flatten
    snn_clinic_mean
    snn_clinic_flatten
)

if [ "$INCLUDE_CLINIC_COX" = "true" ]; then
    CLINIC_MODELS+=(clinic_cox)
fi

pt_count() {
    local target_dir="$1"
    PYTHONPATH="$SCRIPT_DIR" python -c '
from pathlib import Path
import sys
target = Path(sys.argv[1])
if not target.exists():
    print(-1)
else:
    print(sum(1 for _ in target.glob("*.pt")))
' "$target_dir"
}

gene_resolvable_count() {
    local target_dir="$1"
    PYTHONPATH="$SCRIPT_DIR" python -c '
from dataset_deployment.workspace_features import build_case_feature_index
from pathlib import Path
import sys
target = Path(sys.argv[1])
if not target.exists():
    print(-1)
else:
    print(len(build_case_feature_index(target)))
' "$target_dir"
}

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

report_tmp="$(mktemp /tmp/single_modal_full_test_readiness.XXXXXX)"
printf 'series,study,variant,ready,reason,metadata_exists,split_exists,omics_exists,feature_dir,pt_count,resolvable_case_count\n' > "$report_tmp"

record_report() {
    printf '%s\n' "$1" >> "$report_tmp"
}

# ---- WSItest_F ----
for study in "${STUDIES[@]}"; do
    study_subtype="${study#tcga_}"
    metadata_csv="$SCRIPT_DIR/datasets_csv/metadata/${study}.csv"
    split_dir="$SCRIPT_DIR/splits/5foldcv/${study}"
    omics_csv="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/${study_subtype}/rna_clean.csv"
    metadata_exists=false
    split_exists=false
    omics_exists=false
    [ -f "$metadata_csv" ] && metadata_exists=true
    [ -d "$split_dir" ] && [ -f "$split_dir/splits_0.csv" ] && split_exists=true
    [ -f "$omics_csv" ] && omics_exists=true

    for wsi in "${WSI_EXPERIMENTS[@]}"; do
        feature_dir="$SCRIPT_DIR/SurvPGC_Workspace/${study}/P/${wsi}"
        dir_count="$(pt_count "$feature_dir")"
        ready=true
        reason="ok"
        if [ "$metadata_exists" != true ]; then
            ready=false
            reason="missing_metadata"
        elif [ "$split_exists" != true ]; then
            ready=false
            reason="missing_splits"
        elif [ "$dir_count" -le 0 ]; then
            ready=false
            reason="missing_wsi_features"
        fi
        record_report "WSItest_F,$study,$wsi,$ready,$reason,$metadata_exists,$split_exists,$omics_exists,$feature_dir,$dir_count,$dir_count"

        if [ "$GENERATE_ONLY_READY" = "true" ] && [ "$ready" != true ]; then
            continue
        fi
        for preset in "${WSI_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "wsitest_f__%03d__%s__%s__%s.conf" "$seq" "$study" "$wsi" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=WSItest_F
RUN_NAME=${study}__${wsi}
PRESET=$preset
STUDY=$study
WSI_EXPERIMENT=$wsi
EOF
)"
        done
    done
done

# ---- Gengtest_F ----
for study in "${STUDIES[@]}"; do
    study_subtype="${study#tcga_}"
    metadata_csv="$SCRIPT_DIR/datasets_csv/metadata/${study}.csv"
    split_dir="$SCRIPT_DIR/splits/5foldcv/${study}"
    omics_csv="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/${study_subtype}/rna_clean.csv"
    metadata_exists=false
    split_exists=false
    omics_exists=false
    [ -f "$metadata_csv" ] && metadata_exists=true
    [ -d "$split_dir" ] && [ -f "$split_dir/splits_0.csv" ] && split_exists=true
    [ -f "$omics_csv" ] && omics_exists=true

    for gene in "${GENE_F_EXPERIMENTS[@]}"; do
        feature_dir="$SCRIPT_DIR/SurvPGC_Workspace/${study}/G/${gene}"
        dir_count="$(pt_count "$feature_dir")"
        resolvable_count="$(gene_resolvable_count "$feature_dir")"
        ready=true
        reason="ok"
        if [ "$metadata_exists" != true ]; then
            ready=false
            reason="missing_metadata"
        elif [ "$split_exists" != true ]; then
            ready=false
            reason="missing_splits"
        elif [ "$dir_count" -le 0 ]; then
            ready=false
            reason="missing_gene_features"
        elif [ "$resolvable_count" -le 0 ]; then
            ready=false
            reason="unresolvable_gene_filenames"
        fi
        record_report "Gengtest_F,$study,$gene,$ready,$reason,$metadata_exists,$split_exists,$omics_exists,$feature_dir,$dir_count,$resolvable_count"

        if [ "$GENERATE_ONLY_READY" = "true" ] && [ "$ready" != true ]; then
            continue
        fi
        for preset in "${GENE_F_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "gengtest_f__%03d__%s__%s__%s.conf" "$seq" "$study" "$gene" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=Gengtest_F
RUN_NAME=${study}__${gene}
PRESET=$preset
STUDY=$study
GENE_EXPERIMENT=$gene
EOF
)"
        done
    done
done

# ---- Gengtest_CSVRAW ----
for study in "${STUDIES[@]}"; do
    study_subtype="${study#tcga_}"
    metadata_csv="$SCRIPT_DIR/datasets_csv/metadata/${study}.csv"
    split_dir="$SCRIPT_DIR/splits/5foldcv/${study}"
    omics_csv="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/${study_subtype}/rna_clean.csv"
    feature_dir="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/${study_subtype}"
    metadata_exists=false
    split_exists=false
    omics_exists=false
    [ -f "$metadata_csv" ] && metadata_exists=true
    [ -d "$split_dir" ] && [ -f "$split_dir/splits_0.csv" ] && split_exists=true
    [ -f "$omics_csv" ] && omics_exists=true
    ready=true
    reason="ok"
    if [ "$metadata_exists" != true ]; then
        ready=false
        reason="missing_metadata"
    elif [ "$split_exists" != true ]; then
        ready=false
        reason="missing_splits"
    elif [ "$omics_exists" != true ]; then
        ready=false
        reason="missing_rna_clean"
    fi
    record_report "Gengtest_CSVRAW,$study,csvraw,$ready,$reason,$metadata_exists,$split_exists,$omics_exists,$feature_dir,$([ -f "$omics_csv" ] && echo 1 || echo 0),$([ -f "$omics_csv" ] && echo 1 || echo 0)"

    if [ "$GENERATE_ONLY_READY" = "true" ] && [ "$ready" != true ]; then
        continue
    fi
    for preset in "${GENE_CSVRAW_MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "gengtest_csvraw__%03d__%s__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"
        create_conf "$target" "$(cat <<EOF
EXP_GROUP=Gengtest_CSVRAW
RUN_NAME=${study}__csvraw
PRESET=$preset
STUDY=$study
EOF
)"
    done
done

# ---- Clinictest_Li ----
for study in "${STUDIES[@]}"; do
    study_subtype="${study#tcga_}"
    metadata_csv="$SCRIPT_DIR/datasets_csv/metadata/${study}.csv"
    split_dir="$SCRIPT_DIR/splits/5foldcv/${study}"
    omics_csv="$SCRIPT_DIR/datasets_csv/raw_rna_data/combine/${study_subtype}/rna_clean.csv"
    metadata_exists=false
    split_exists=false
    omics_exists=false
    [ -f "$metadata_csv" ] && metadata_exists=true
    [ -d "$split_dir" ] && [ -f "$split_dir/splits_0.csv" ] && split_exists=true
    [ -f "$omics_csv" ] && omics_exists=true

    for clinic in "${CLINIC_EXPERIMENTS[@]}"; do
        feature_dir="$SCRIPT_DIR/SurvPGC_Workspace/${study}/C/${clinic}"
        dir_count="$(pt_count "$feature_dir")"
        ready=true
        reason="ok"
        if [ "$metadata_exists" != true ]; then
            ready=false
            reason="missing_metadata"
        elif [ "$split_exists" != true ]; then
            ready=false
            reason="missing_splits"
        elif [ "$dir_count" -le 0 ]; then
            ready=false
            reason="missing_clinic_features"
        fi
        record_report "Clinictest_Li,$study,$clinic,$ready,$reason,$metadata_exists,$split_exists,$omics_exists,$feature_dir,$dir_count,$dir_count"

        if [ "$GENERATE_ONLY_READY" = "true" ] && [ "$ready" != true ]; then
            continue
        fi
        for preset in "${CLINIC_MODELS[@]}"; do
            seq=$((seq + 1))
            fname=$(printf "clinictest_li__%03d__%s__%s__%s.conf" "$seq" "$study" "$clinic" "$preset")
            target="$OUT_DIR/$fname"
            create_conf "$target" "$(cat <<EOF
EXP_GROUP=Clinictest_Li
RUN_NAME=${study}__${clinic}
PRESET=$preset
STUDY=$study
CLINIC_EXPERIMENT=$clinic
EOF
)"
        done
    done
done

mv "$report_tmp" "$REPORT_PATH"

echo "Generated $created new configs in $OUT_DIR"
echo "Total indexed configs this round: $seq"
echo "Skipped existing configs: $skipped"
echo "Clinic Cox included: $INCLUDE_CLINIC_COX"
echo "Generate only ready combos: $GENERATE_ONLY_READY"
echo "Validation report: $REPORT_PATH"
