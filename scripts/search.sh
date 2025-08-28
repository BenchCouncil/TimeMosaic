#!/bin/bash

MAX_JOBS=8
TOTAL_GPUS=2
MAX_RETRIES=0

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

seq_lens=(96 192 320 512)
pred_lens=(96 192 336 720)
d_model_list=(16 128 512)
e_layers_list=(1 3 5)
# epoch_list=(10 50 100)
epoch_list=(50)
# lr_list=(1e-5 1e-4 1e-3 1e-2 0.05)
lr_list=(1e-4 1e-3 1e-2)

batch_size=32
n_heads=8
d_layers=1

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
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \

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
        for d_model in "${d_model_list[@]}"; do
          d_ff=$((d_model * 4))
          for e_layers in "${e_layers_list[@]}"; do
            for epochs in "${epoch_list[@]}"; do
              for lr in "${lr_list[@]}"; do

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

                model_id="${dataset}_${seq_len}_${pred_len}_${d_model}_${d_ff}_e${e_layers}_ep${epochs}_lr${lr}_${model_name}"
                log_file="logs/${model_id}.log"

                if [[ -f "$log_file" ]] && grep -q "mse:" "$log_file"; then
                  echo "⏩ Skip (already has mse:) $model_id"
                  continue
                fi

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
                  --d_layers $d_layers \
                  --factor 3 \
                  --enc_in $enc_in \
                  --dec_in $enc_in \
                  --c_out $enc_in \
                  --des Exp \
                  --n_heads $n_heads \
                  --d_model $d_model \
                  --d_ff $d_ff \
                  --itr 1 \
                  --train_epochs $epochs \
                  --learning_rate $lr \
                  --batch_size $batch_size \
                  --num_workers 1"

                read -u9
                gpu_id=$(( job_index % TOTAL_GPUS ))
                run_job $gpu_id "$cmd" "$log_file" "$model_id" &
                job_index=$((job_index + 1))

              done
            done
          done
        done
      done
    done
  done

done

wait
exec 9>&-
