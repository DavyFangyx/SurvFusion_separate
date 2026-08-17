#!/bin/bash
# 生成 tcga_kich 指定特征组合下的全模型验证配置。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.bash"

STUDY="tcga_kich"
EXP_GROUP="${EXP_GROUP:-L0_KICH_full_model_val}"

generate_full_model_val_configs
