#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/gen_tcga_brca_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_coad_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kich_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirc_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_kirp_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_lihc_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_prad_poe_model_val.sh"
bash "$SCRIPT_DIR/gen_tcga_read_poe_model_val.sh"
