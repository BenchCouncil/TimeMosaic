#!/bin/bash

model_name=AGPT_loss_G
label_len=48
d_layers=1
factor=3
enc_in=7
dec_in=7
c_out=7
des='Exp'
itr=1
data_name=ETTm1
dataset_path=./dataset/ETT-small/
data_file=ETTm1.csv

# 支持的输入长度和预测长度
seq_lens=(96 192 288 384 512 736 1024)
pred_lens=(96 192 336 720)

# 每个 pred_len 对应的参数配置
declare -A e_layers_map
declare -A n_heads_map
declare -A batch_size_map

e_layers_map[96]=1
e_layers_map[192]=3
e_layers_map[336]=1
e_layers_map[720]=3

n_heads_map[96]=2
n_heads_map[192]=2
n_heads_map[336]=4
n_heads_map[720]=4

batch_size_map[96]=32
batch_size_map[192]=128
batch_size_map[336]=128
batch_size_map[720]=128

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do

    # 可选过滤逻辑，避免预测长度超过输入长度
    if [ "$pred_len" -le "$seq_len" ]; then
      e_layers=${e_layers_map[$pred_len]}
      n_heads=${n_heads_map[$pred_len]}
      batch_size=${batch_size_map[$pred_len]}

      echo "Running ETTm1 with seq_len=$seq_len, pred_len=$pred_len, e_layers=$e_layers, n_heads=$n_heads, batch_size=$batch_size"

      python -u run.py \
        --task_name AGPT_loss \
        --is_training 1 \
        --root_path $dataset_path \
        --data_path $data_file \
        --model_id ${data_name}_${seq_len}_${pred_len} \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers $d_layers \
        --factor $factor \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --des "$des" \
        --n_heads $n_heads \
        --batch_size $batch_size \
        --itr $itr
    fi

  done
done
