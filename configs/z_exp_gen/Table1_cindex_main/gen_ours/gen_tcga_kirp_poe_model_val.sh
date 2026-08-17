#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_kirp"
EXP_GROUP="${EXP_GROUP:-L0_KIRP_poe_model_val}"

generate_poe_model_val_configs
