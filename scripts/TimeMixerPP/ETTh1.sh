model_name=TimeMixerPP

seq_len=96
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=64
patience=10
batch_size=512

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path  ./dataset/ETT-small/\
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --enc_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window