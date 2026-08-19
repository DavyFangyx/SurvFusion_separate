#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/gen_tcga_brca_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_coad_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_kich_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_kirc_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_kirp_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_lihc_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_prad_ablation_test.sh"
bash "$SCRIPT_DIR/gen_tcga_read_ablation_test.sh"
