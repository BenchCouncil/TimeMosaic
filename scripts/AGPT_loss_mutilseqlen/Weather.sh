#!/bin/bash

# 通用参数设置
model_name=AGPT_loss_G
label_len=48
e_layers=2
d_layers=1
factor=3
enc_in=21
dec_in=21
c_out=21
des='Exp'
itr=1
train_epochs=10
data_name=custom

# 数据集路径
root_path=./dataset/weather/
data_path=weather.csv

# 遍历的 seq_len 和 pred_len
seq_lens=(96 192 288 384 512 736 1024)
pred_lens=(96 192 336 720)

# 不同 pred_len 对应的超参
declare -A n_heads_map
declare -A batch_size_map

n_heads_map[96]=4
n_heads_map[192]=16
n_heads_map[336]=4
n_heads_map[720]=4

batch_size_map[96]=32
batch_size_map[192]=32
batch_size_map[336]=128
batch_size_map[720]=128

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    if [ "$pred_len" -le "$seq_len" ]; then
      n_heads=${n_heads_map[$pred_len]}
      batch_size=${batch_size_map[$pred_len]}

      echo "Running weather with seq_len=$seq_len, pred_len=$pred_len, n_heads=$n_heads, batch_size=$batch_size"

      python -u run.py \
        --task_name AGPT_loss \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id weather_${seq_len}_${pred_len} \
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
        --des $des \
        --itr $itr \
        --n_heads $n_heads \
        --batch_size $batch_size \
        --train_epochs $train_epochs
    fi
  done
done
