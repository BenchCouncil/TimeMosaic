
model_name=AGPT_PT

python -u run.py \
  --task_name AGPT_loss \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location1.csv \
  --model_id Wind1_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --channel CD \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1

python -u run.py \
  --task_name AGPT_loss \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location1.csv \
  --model_id Wind1_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --channel CD \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python -u run.py \
  --task_name AGPT_loss \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location1.csv \
  --model_id Wind1_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --channel CD \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python -u run.py \
  --task_name AGPT_loss \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location1.csv \
  --model_id Wind1_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --channel CD \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1