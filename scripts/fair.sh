#!/bin/bash

MAX_JOBS=4
TOTAL_GPUS=2
MAX_RETRIES=1

mkdir -p logs
> failures.txt

declare -a models=("TimeMosaic" "SimpleTM" "TimeFilter" "xPatch" "PatchMLP" "Duet" "iTransformer" "TimeMixer" "PatchTST" "DLinear" "FreTS" "LightTS")

datasets=(
  "ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 ETTh1"
  "ETTh2 ./dataset/ETT-small/ ETTh2.csv 7 ETTh2"
  "ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 ETTm1"
  "ETTm2 ./dataset/ETT-small/ ETTm2.csv 7 ETTm2"
  "Exchange ./dataset/exchange_rate/ exchange_rate.csv 8 custom"
  "Weather ./dataset/weather/ weather.csv 21 custom"
)

d_model=512
d_ff=2048
e_layers=2
n_heads=8
seq_lens=(320)
pred_lens=(96 192 336 720)

SEMAPHORE=/tmp/gs_semaphore
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE
for ((i=0;i<${MAX_JOBS};i++)); do echo >&9; done

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "✅ [GPU $gpu_id] Success: $model_id"
      break
    else
      echo "❌ [GPU $gpu_id] Failed: $model_id (Attempt $((attempt+1)))"
      attempt=$((attempt + 1))
      if (( attempt > MAX_RETRIES )); then
        echo "$cmd" >> failures.txt
      fi
    fi
  done

  echo >&9
}

job_index=0

for model_name in "${models[@]}"; do
  for dataset_config in "${datasets[@]}"; do
    set -- $dataset_config
    dataset=$1; root_path=$2; data_path=$3; enc_in=$4; data_flag=$5

    for seq_len in "${seq_lens[@]}"; do
      for pred_len in "${pred_lens[@]}"; do

        task_name="long_term_forecast"
        if [[ "$model_name" == "TimeFilter" ]]; then
          task_name="Exp_TimeFilter"
        elif [[ "$model_name" == "PathFormer" ]]; then
          task_name="Exp_PathFormer"
        elif [[ "$model_name" == "Duet" ]]; then
          task_name="Exp_DUET"
        elif [[ "$model_name" == "TimeMosaic" ]]; then
          task_name="TimeMosaic"
        fi

        model_id="${dataset}_${seq_len}_${pred_len}_${d_model}_${d_ff}_${loss}_e${e_layers}_h${n_heads}_${model_name}"
        log_file="logs/${model_id}.log"

        cmd="python -u run.py \
          --task_name $task_name \
          --is_training 1 \
          --root_path $root_path \
          --data_path $data_path \
          --model_id $model_id \
          --model $model_name \
          --data $data_flag \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --e_layers $e_layers \
          --d_layers 1 \
          --factor 3 \
          --enc_in $enc_in \
          --dec_in $enc_in \
          --c_out $enc_in \
          --des Exp \
          --n_heads $n_heads \
          --d_model $d_model \
          --d_ff $d_ff \
          --itr 1"

        read -u9
        gpu_id=$(( job_index % TOTAL_GPUS ))
        run_job $gpu_id "$cmd" "$log_file" "$model_id" &
        job_index=$((job_index + 1))
      done
    done
  done
done

wait
exec 9>&-