#!/bin/bash

# 保留的PID（你想保留的进程）
KEEP_PID=2006538

echo "⏳ 实时监控中，保留 PID=$KEEP_PID，其余进程将被杀死。"

while true; do
    # 获取当前所有使用GPU的 python 进程 PID 列表（排除 KEEP_PID）
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -E '^[0-9]+$' | grep -v "$KEEP_PID")

    for pid in $PIDS; do
        if ps -p "$pid" > /dev/null; then
            echo "⚠️ 杀死进程 $pid ($(ps -p $pid -o cmd=))"
            kill -9 "$pid"
        fi
    done

    sleep 1
done
