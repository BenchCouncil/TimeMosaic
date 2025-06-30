#!/bin/bash

model_name=AGPT_loss_G
label_len=48
e_layers=3
d_layers=1
factor=3
enc_in=7
dec_in=7
c_out=7
n_heads=4
itr=1
data_name=ETTh2
dataset_path=./dataset/ETT-small/
data_file=ETTh2.csv

# 可变 seq_len 和 pred_len
seq_lens=(96 192 288 384 512 736 1024)
pred_lens=(96 192 336 720)

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    echo "Running ETTh2 with seq_len=$seq_len, pred_len=$pred_len..."

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
