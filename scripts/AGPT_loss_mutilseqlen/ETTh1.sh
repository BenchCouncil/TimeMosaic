#!/bin/bash

model_name=AGPT_loss_G
label_len=48
e_layers=1
d_layers=1
factor=3
enc_in=7
dec_in=7
c_out=7
itr=1
data_name=ETTh1
dataset_path=./dataset/ETT-small/
data_file=ETTh1.csv

# 循环的 seq_len 列表
seq_lens=(96 192 288 384 512 736 1024)

# 目标预测长度
pred_lens=(96 192 336 720)

# 每个 pred_len 对应的 n_heads 参数
declare -A n_heads_map
n_heads_map[96]=2
n_heads_map[192]=8
n_heads_map[336]=8
n_heads_map[720]=16

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    n_heads=${n_heads_map[$pred_len]}
    
    echo "Running ETTh1 with seq_len=$seq_len, pred_len=$pred_len, n_heads=$n_heads"
    
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
      --des 'Exp' \
      --n_heads $n_heads \
      --itr $itr
  done
done
