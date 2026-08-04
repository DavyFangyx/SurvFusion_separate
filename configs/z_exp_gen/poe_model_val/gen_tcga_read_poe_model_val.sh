#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_read"
EXP_GROUP="${EXP_GROUP:-READ_poe_model_val}"

generate_poe_model_val_configs
