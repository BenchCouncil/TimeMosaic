#!/bin/bash

model_name=AGPT_loss_G
label_len=48
batch_size=16
itr=1

seq_lens=(96 192 288 384 512 736 1024)
# seq_lens=(96 192 288 384 512 736 1024 1536 2048 3072 4096)
pred_lens=(96 192 336 720)

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    echo "Running model with seq_len=$seq_len, pred_len=$pred_len..."
    
    python -u run.py \
      --task_name AGPT_loss \
      --is_training 1 \
      --use_multi_gpu \
      --devices 0,1,2,3 \
      --root_path ./dataset/electricity/ \
      --data_path electricity.csv \
      --model_id ECL_${seq_len}_${pred_len} \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 321 \
      --dec_in 321 \
      --c_out 321 \
      --des 'Exp' \
      --batch_size $batch_size \
      --itr $itr
  done
done
