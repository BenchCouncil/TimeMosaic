#!/bin/bash

# ===================== 配置参数 =====================
MAX_JOBS=2
TOTAL_GPUS=2
MAX_RETRIES=1
LOG_DIR="logs_DUET"

mkdir -p "$LOG_DIR"

SCRIPT_LIST=(
# "/root/daye/AGPT/scripts/Duet/ECL.sh"
# "/root/daye/AGPT/scripts/Duet/ETTh1.sh"
# "/root/daye/AGPT/scripts/Duet/ETTm1.sh"
# "/root/daye/AGPT/scripts/Duet/ETTh2.sh"
# "/root/daye/AGPT/scripts/Duet/Exchange.sh"
# "/root/daye/AGPT/scripts/Duet/ETTm2.sh"
"/root/daye/AGPT/scripts/Duet/Traffic.sh"
# "/root/daye/AGPT/scripts/Duet/PEMS.sh"
"/root/daye/AGPT/scripts/Duet/Solar.sh"
"/root/daye/AGPT/scripts/Duet/Weather.sh"
"/root/daye/AGPT/scripts/Duet/Wind1.sh"
"/root/daye/AGPT/scripts/Duet/Wind2.sh"
"/root/daye/AGPT/scripts/Duet/Wind3.sh"
"/root/daye/AGPT/scripts/Duet/Wind4.sh"
)



# ===================== 工具函数 =====================

check_jobs() {
    while true; do
        local running_jobs
        running_jobs=$(jobs -p | wc -l)
        if [ "$running_jobs" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

get_gpu_allocation() {
    local job_number=$1
    local gpu_id=$((job_number % TOTAL_GPUS))
    echo $gpu_id
}

run_with_retry() {
    local script_path=$1
    local gpu_id=$2
    local attempt=0

    local clean_path=${script_path#/root/daye/AGPT/}
    local log_name="${clean_path//\//_}.log"
    local log_file="$LOG_DIR/$log_name"

    while [ $attempt -le $MAX_RETRIES ]; do
        echo "[INFO] Running $script_path on GPU $gpu_id (Attempt $((attempt + 1)))"
        CUDA_VISIBLE_DEVICES=$gpu_id bash "$script_path" > "$log_file" 2>&1
        local status=$?
        if [ $status -eq 0 ]; then
            echo "[SUCCESS] $script_path succeeded. Log saved to $log_file"
            break
        else
            echo "[WARNING] $script_path failed (Attempt $((attempt + 1))). Retrying..."
            ((attempt++))
            sleep 2
        fi
    done

    if [ $attempt -gt $MAX_RETRIES ]; then
        echo "[ERROR] $script_path failed after $MAX_RETRIES attempts. See $log_file"
    fi
}

# ===================== 主调度逻辑 =====================
job_number=0
for script_path in "${SCRIPT_LIST[@]}"; do
    check_jobs
    gpu_id=$(get_gpu_allocation $job_number)
    # run_with_retry "$script_path" "$gpu_id" &
    # ((job_number++))
    run_with_retry "$script_path" "$gpu_id" &
    ((job_number++))
done

wait
echo "[ALL DONE] 所有脚本执行完毕，日志存于 $LOG_DIR"

# git add .
# git commit -m "init"
# git push -u origin main