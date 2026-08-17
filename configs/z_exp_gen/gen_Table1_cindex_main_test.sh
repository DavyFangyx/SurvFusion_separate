#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/Table1_cindex_main/gen_baselines/gen_baselines.sh"
bash "$SCRIPT_DIR/Table1_cindex_main/gen_ours/gen_ours.sh"
