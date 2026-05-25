#!/bin/bash
# bg.sh — 后台启动 scheduler
# 用法: CUDA_VISIBLE_DEVICES=N bash bg.sh <日志文件名>
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "用法: CUDA_VISIBLE_DEVICES=N bash bg.sh <日志文件名>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$1"

nohup bash scheduler.sh > "$LOG_FILE" 2>&1 &
PID=$!

echo "PID: $PID"
echo "日志: $SCRIPT_DIR/$LOG_FILE"
echo "停止: kill $PID"
