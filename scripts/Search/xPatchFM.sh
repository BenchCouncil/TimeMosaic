#!/bin/bash 

MAX_JOBS=4
TOTAL_GPUS=4
MAX_RETRIES=0

mkdir -p logs
> failures.txt  # 清空失败日志

declare -a models=("xPatchFM")

datasets=(
  "ETTh1 ./dataset/ETT-small/ ETTh1.csv 7 ETTh1"
  "ETTh2 ./dataset/ETT-small/ ETTh2.csv 7 ETTh2"
  "ETTm1 ./dataset/ETT-small/ ETTm1.csv 7 ETTm1"
  "ETTm2 ./dataset/ETT-small/ ETTm2.csv 7 ETTm2"
  "Exchange ./dataset/exchange_rate/ exchange_rate.csv 8 custom"
  "Weather ./dataset/weather/ weather.csv 21 custom"
  "ECL ./dataset/electricity/ electricity.csv 321 custom"
)

d_pairs=("32 128" "128 256" "256 1024" "512 2048")
losses=("MAE" "MSE")
channels=("CD" "CI")
seq_lens=(96 192 336 720)
pred_lens=(96 192 336 720)
label_len=48

gate_hidden_units_list=(32 64)
dnn_hidden_units_list=("128 128" "256 256" "64 128 64")
cin_layer_size_list=(
  "256 128 128 64"
  "256 128 64"
  "512 128 64"
  "256 128"
  "512 64"
)

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
        for loss in "${losses[@]}"; do
          for channel in "${channels[@]}"; do
            for pair in "${d_pairs[@]}"; do
              d_model=$(echo $pair | cut -d' ' -f1)
              d_ff=$(echo $pair | cut -d' ' -f2)

              for gate_hidden in "${gate_hidden_units_list[@]}"; do
                for dnn_hidden in "${dnn_hidden_units_list[@]}"; do
                  for cin_layers in "${cin_layer_size_list[@]}"; do

                    model_id="${dataset}_${seq_len}_${pred_len}_${d_model}_${d_ff}_${loss}_${channel}_g${gate_hidden}_${model_name}"
                    log_file="logs/${model_id}.log"

                    cmd="python -u run.py \
                      --task_name long_term_forecast \
                      --is_training 1 \
                      --root_path $root_path \
                      --data_path $data_path \
                      --model_id $model_id \
                      --model $model_name \
                      --data $data_flag \
                      --features M \
                      --seq_len $seq_len \
                      --label_len $label_len \
                      --pred_len $pred_len \
                      --d_layers 1 \
                      --factor 3 \
                      --enc_in $enc_in \
                      --dec_in $enc_in \
                      --c_out $enc_in \
                      --des Exp \
                      --d_model $d_model \
                      --d_ff $d_ff \
                      --loss $loss \
                      --channel $channel \
                      --itr 1 \
                      --gate_hidden_units $gate_hidden \
                      --dnn_hidden_units $dnn_hidden \
                      --cin_layer_size $cin_layers"

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
  done
done

wait
exec 9>&-
