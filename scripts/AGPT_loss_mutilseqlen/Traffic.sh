#!/bin/bash

# 通用参数
model_name=AGPT_loss_G
label_len=48
e_layers=2
d_layers=1
factor=3
enc_in=862
dec_in=862
c_out=862
d_model=512
d_ff=512
batch_size=4
itr=1
des='Exp'

# 数据集信息
root_path=./dataset/traffic/
data_path=traffic.csv
data_name=custom

# 输入长度与预测长度列表
seq_lens=(96 192 288 384 512 736 1024)
pred_lens=(96 192 336 720)

for seq_len in "${seq_lens[@]}"; do
  for pred_len in "${pred_lens[@]}"; do

    # 可选：跳过 pred_len > seq_len 的非法组合
    if [ "$pred_len" -le "$seq_len" ]; then
      echo "Running traffic with seq_len=$seq_len, pred_len=$pred_len"

      python -u run.py \
        --task_name AGPT_loss \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id traffic_${seq_len}_${pred_len} \
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
        --batch_size $batch_size \
        --itr $itr
    fi

  done
done
