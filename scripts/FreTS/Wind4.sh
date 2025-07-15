
model_name=FreTS

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location4.csv \
  --model_id Wind4_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --n_heads 2 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location4.csv \
  --model_id Wind4_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location4.csv \
  --model_id Wind4_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --n_heads 8 \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/wind/ \
  --data_path Location4.csv \
  --model_id Wind4_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --target Power \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --n_heads 16 \
  --itr 1