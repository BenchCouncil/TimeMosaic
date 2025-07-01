#!/bin/bash

model_name=AGPT_loss_G
label_len=0
learning_rate=0.001
batch_size=32
train_epochs=10
patience=3
itr=1
des='Exp'

# 数据集路径
root_path=./dataset/Solar/
data_path=solar_AL.txt
data_name=Solar

# 模型结构参数
e_layers=3
d_layers=1
factor=3
enc_in=137
dec_in=137
c_out=137
d_model=512
d_ff=2048

# 输入长度与预测长度列表
seq_lens=(96 192 288 384 512 736 1024)
pred_lens=(96 192 336 720)

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do
    if [ "$pred_len" -le "$seq_len" ]; then
      echo "Running Solar with seq_len=$seq_len, pred_len=$pred_len"

      python -u run.py \
        --task_name AGPT_loss \
        --is_training 1 \
        --use_multi_gpu \
        --devices 0,1,2,3 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id solar_${seq_len}_${pred_len} \
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
        --d_model $d_model \
        --d_ff $d_ff \
        --des $des \
        --itr $itr \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience
    fi
  done
done
