#!/bin/bash
# exp_gen/gen_Gengtest_CSVRAW.sh
# 生成 Gengtest_CSVRAW 的 config：3 个 study × 2 个模型 = 6 个 config
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$SCRIPT_DIR/configs/queue"
mkdir -p "$OUT_DIR"

STUDIES=(
    tcga_kich
    tcga_kirc
    tcga_kirp
)

MODELS=(
    mlp_gene
    snn_gene
)

seq=0
created=0
skipped=0
for study in "${STUDIES[@]}"; do
    for preset in "${MODELS[@]}"; do
        seq=$((seq + 1))
        fname=$(printf "gengtest_csvraw__%03d__%s__%s.conf" "$seq" "$study" "$preset")
        target="$OUT_DIR/$fname"

        if [ -e "$target" ]; then
            skipped=$((skipped + 1))
            continue
        fi

        cat > "$target" <<EOF
EXP_GROUP=Gengtest_CSVRAW
RUN_NAME=${study}__csvraw
PRESET=$preset
STUDY=$study
EOF
        created=$((created + 1))
    done
done

echo "Generated $created new configs in $OUT_DIR  (Gengtest_CSVRAW total: $seq, skipped existing: $skipped)"
